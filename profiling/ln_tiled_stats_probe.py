"""
Row 47, accuracy kill test. It writes no Metal.

WHAT ROW 47 CLAIMS

Row 46 deleted the LayerNorm write. It left `layer_norm_stats()`, a pass that
reads a whole activation to write two floats for each row. That pass runs at
104 GB/s against a 128 GB/s roof, so no better kernel wins it. Only fewer
bytes win it.

Every activation the pass reads was written by a GEMM one stage earlier, and
that GEMM holds the value in registers when it stores it. So the GEMM
epilogue can produce the statistics, and the 65 MiB read never happens.

WHY THIS TEST EXISTS

The obstacle is arithmetic, not Metal.

`layer_norm_stats()` sees a whole row, so it runs TWO passes over registers:
it takes the mean first, then it sums `(v - mean)^2`. That form never
cancels.

A GEMM threadgroup sees ONE TILE of the row. Shape 6 splits `d_model = 128`
into 2 tiles at `bn = 64`, and shape 8 splits 1024 into 16. So the epilogue
cannot centre against a mean it does not have, and the tiles must combine.

Two ways to combine, and they are not equally safe:

    naive   each tile writes the raw sum and the raw sum of squares.
            The reduce takes `var = Q/D - mean^2`.
            This is the uncentred form. `fast_layernorm` refuses it for the
            whole row, because it cancels when the mean is large against the
            standard deviation, and a residual stream drifts.

    chan    each tile centres against ITS OWN mean, which it can do, because
            the tile is in registers. It writes `{n, mean, M2}`. The reduce
            combines the tiles by Chan's formula, which is stable.

This script measures both against float64, at the real shape, on real
activations. It reports the error of the statistics, and then the error of
the projection that row 46 builds on them, which is the number the harness
sees.

Run it:

    .venv/bin/python3 profiling/ln_tiled_stats_probe.py
    .venv/bin/python3 profiling/ln_tiled_stats_probe.py --shape 8
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from appendix_cases import APPENDIX_SHAPES  # noqa: E402
from torch_transformer_benchmark import (  # noqa: E402
    BaselineTransformer,
    LAYER_NORM_EPS,
)

# The tile the model uses. Every shape picks `32x64x16x2x2` today, so the row
# splits into `d_model / 64` tiles. See the `plan` string in scoreboard.json.
TILE_N = 64


def drift(x: np.ndarray) -> float:
    """Return the worst row ratio |mean| / std. It decides the naive form."""
    mean = x.mean(axis=-1)
    std = x.std(axis=-1)
    return float(np.max(np.abs(mean) / np.maximum(std, 1e-30)))


def stats_whole_row(x: np.ndarray, dtype: type):
    """
    Today's kernel. Two passes over the whole row, the second one centred.

    This is `fast_layernorm._STATS_SOURCE`.
    """
    v = x.astype(dtype)
    mean = v.mean(axis=-1, keepdims=True)
    var = ((v - mean) ** 2).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + dtype(LAYER_NORM_EPS))
    return mean, rstd


def stats_tiled_naive(x: np.ndarray, dtype: type, tile: int):
    """
    Each tile writes the raw sum and the raw sum of squares. The reduce takes
    the uncentred variance.
    """
    v = x.astype(dtype)
    rows, width = v.shape
    tiles = v.reshape(rows, width // tile, tile)

    # The epilogue. One accumulator pair for each tile.
    part_s = tiles.sum(axis=-1, dtype=dtype)
    part_q = (tiles * tiles).sum(axis=-1, dtype=dtype)

    # The reduce kernel, over `width / tile` partials for each row.
    total_s = part_s.sum(axis=-1, dtype=dtype)[:, None]
    total_q = part_q.sum(axis=-1, dtype=dtype)[:, None]
    n = dtype(width)
    mean = total_s / n
    var = total_q / n - mean * mean
    var = np.maximum(var, dtype(0.0))
    rstd = 1.0 / np.sqrt(var + dtype(LAYER_NORM_EPS))
    return mean, rstd


def stats_tiled_chan(x: np.ndarray, dtype: type, tile: int):
    """
    Each tile centres against its own mean, then Chan's formula combines the
    tiles. The tile is in registers, so the centring costs nothing.
    """
    v = x.astype(dtype)
    rows, width = v.shape
    tiles = v.reshape(rows, width // tile, tile)

    # The epilogue. Two passes over the tile, the second one centred. This is
    # the same arithmetic as today's kernel, on a narrower row.
    part_mean = tiles.mean(axis=-1, dtype=dtype)
    centred = tiles - part_mean[:, :, None]
    part_m2 = (centred * centred).sum(axis=-1, dtype=dtype)

    # The reduce kernel. It walks the tiles in order.
    n_a = dtype(tile)
    mean_a = part_mean[:, 0].copy()
    m2_a = part_m2[:, 0].copy()
    for index in range(1, tiles.shape[1]):
        n_b = dtype(tile)
        mean_b = part_mean[:, index]
        m2_b = part_m2[:, index]
        n_ab = dtype(n_a + n_b)
        delta = mean_b - mean_a
        mean_a = mean_a + delta * (n_b / n_ab)
        m2_a = m2_a + m2_b + delta * delta * (n_a * n_b / n_ab)
        n_a = n_ab

    mean = mean_a[:, None]
    var = (m2_a / dtype(width))[:, None]
    rstd = 1.0 / np.sqrt(var + dtype(LAYER_NORM_EPS))
    return mean, rstd


def project(x: np.ndarray, mean, rstd, gain, bias, weight, proj_bias,
            dtype: type):
    """
    Row 46's absorbed projection, over RAW x, with the given statistics.

    This is `steel_gemm.layer_norm_constants()` plus the epilogue.
    """
    v = x.astype(dtype)
    g = gain.astype(dtype)
    b = bias.astype(dtype)
    w = weight.astype(dtype)

    bw = g[:, None] * w
    c1 = bw.sum(axis=0)
    c2 = b @ w + proj_bias.astype(dtype)

    acc = v @ bw
    return rstd * (acc - mean * c1) + c2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, default=6,
                        help="the appendix case id. The default is shape 6, "
                             "which is 66.5%% of the FLOP weight")
    parser.add_argument("--rows", type=int, default=4096,
                        help="rows to test. A float64 copy of the full shape "
                             "6 chunk does not fit")
    parser.add_argument("--tile", type=int, default=TILE_N,
                        help="the GEMM tile width bn. The model uses 64")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shape = next(s for s in APPENDIX_SHAPES if s.case_id == args.shape)
    config = shape.config()
    torch.manual_seed(args.seed)
    model = BaselineTransformer(config).eval()

    n_tiles = shape.d_model // args.tile
    print(f"Shape {shape.case_id}: B{shape.batch_size} D{shape.d_model} "
          f"H{shape.num_heads} S{shape.seq_len} L{shape.num_layers}")
    print(f"{args.rows} rows, bn = {args.tile}, so {n_tiles} tiles for each "
          f"row")
    print(f"atol = 0.002, and the model sits at 3.3e-06\n")

    batch = min(shape.batch_size, max(1, args.rows // shape.seq_len))
    x = torch.randn(batch, shape.seq_len, shape.d_model)
    captured = []
    with torch.no_grad():
        h = x
        for block in model.layers:
            captured.append(h.detach().clone())
            h = block(h, None, config.causal)

    header = (f"{'layer':>5} {'|mean|/std':>11} | {'rstd today':>11} "
              f"{'rstd naive':>11} {'rstd chan':>11} | {'proj today':>11} "
              f"{'proj naive':>11} {'proj chan':>11}")
    print(header)
    print("-" * len(header))

    worst_naive = 0.0
    worst_chan = 0.0
    worst_today = 0.0
    for index, block in enumerate(model.layers):
        flat = captured[index].reshape(-1, shape.d_model).numpy()[:args.rows]
        x64 = flat.astype(np.float64)

        gain = block.norm1.weight.detach().numpy()
        bias = block.norm1.bias.detach().numpy()
        weight = block.attention.q_proj.weight.detach().numpy().T
        proj_bias = block.attention.q_proj.bias.detach().numpy()

        m64, r64 = stats_whole_row(x64, np.float64)
        m_now, r_now = stats_whole_row(x64, np.float32)
        m_nai, r_nai = stats_tiled_naive(x64, np.float32, args.tile)
        m_chn, r_chn = stats_tiled_chan(x64, np.float32, args.tile)

        # The statistics error, relative, because rstd is not O(1).
        def rel(r):
            return float(np.max(np.abs(r - r64) / np.abs(r64)))

        ref = project(x64, m64, r64, gain, bias, weight, proj_bias,
                      np.float64)
        p_now = project(x64, m_now, r_now, gain, bias, weight, proj_bias,
                        np.float32)
        p_nai = project(x64, m_nai, r_nai, gain, bias, weight, proj_bias,
                        np.float32)
        p_chn = project(x64, m_chn, r_chn, gain, bias, weight, proj_bias,
                        np.float32)

        e_now = float(np.max(np.abs(p_now - ref)))
        e_nai = float(np.max(np.abs(p_nai - ref)))
        e_chn = float(np.max(np.abs(p_chn - ref)))
        worst_today = max(worst_today, e_now)
        worst_naive = max(worst_naive, e_nai)
        worst_chan = max(worst_chan, e_chn)

        print(f"{index:>5} {drift(flat):>11.2f} | {rel(r_now):>11.3e} "
              f"{rel(r_nai):>11.3e} {rel(r_chn):>11.3e} | {e_now:>11.3e} "
              f"{e_nai:>11.3e} {e_chn:>11.3e}")

    print()
    print(f"worst projection error, today: {worst_today:.3e}")
    print(f"worst projection error, naive: {worst_naive:.3e}  "
          f"({worst_naive / max(worst_today, 1e-30):.1f}x today)")
    print(f"worst projection error, chan : {worst_chan:.3e}  "
          f"({worst_chan / max(worst_today, 1e-30):.1f}x today)")
    print(f"budget (atol):                 {2e-3:.3e}")
    print()
    for name, worst in (("naive", worst_naive), ("chan", worst_chan)):
        ratio = worst / max(worst_today, 1e-30)
        if ratio <= 2.0:
            verdict = "SAFE. It holds today's error."
        elif worst <= 2e-4:
            verdict = "MARGINAL. It costs bits, but it keeps headroom."
        else:
            verdict = "FAIL. Do not build it."
        print(f"{name:>5}: {verdict}")


if __name__ == "__main__":
    main()

"""
Row 43 route 2, accuracy kill test. It writes no Metal.

WHAT ROUTE 2 CLAIMS

A LayerNorm is affine in the row once the mean and the rstd are known, so it
distributes through the matmul that follows it. Write `m` and `r` for the row
mean and rstd, `w` and `b` for the LayerNorm gain and bias, `c` for the
deferred residual bias of row 36, and `B` for the GEMM weight:

    y[i,j]   = ((x[i,j] + c_j) - m_i) * r_i * w_j + b_j
    out[i,n] = sum_j y[i,j] * B[j,n]
             = r_i * ((X @ Bw)[i,n] + c3[n] - m_i * c1[n]) + c2[n]

with three constants that depend on the weights alone:

    Bw[j,n] = w_j * B[j,n]        the weight, with its rows scaled
    c1[n]   = sum_j w_j * B[j,n]
    c2[n]   = sum_j b_j * B[j,n]
    c3[n]   = sum_j c_j * w_j * B[j,n]

So the LayerNorm never writes an activation. It becomes three vectors built
at weight build time, plus two floats per row, plus an epilogue. That is the
article's optimization #1 (move constant work to load time) applied to its
optimization #3 (fuse the norm into the next kernel).

WHY THIS TEST EXISTS

The speed case is sound. The risk is arithmetic.

Today the matmul runs over NORMALIZED values, which are O(1). Route 2 runs it
over RAW `x`, then subtracts `m_i * c1[n]`. A transformer accumulates its
residual stream, so `x` drifts away from zero, and the subtraction is a
difference of two large close numbers. That is catastrophic cancellation, and
it is the reason LayerNorm exists.

The gate is `atol = 0.002` and the model sits at `max_abs = 2.4e-06`, so
there is about 800x of headroom, or about 9 bits. This script measures how
many bits route 2 actually spends.

METHOD

Take real activations from the real model, at the real shape. Compute the
projection three ways:

    float64        the reference
    today          float32 LayerNorm, then float32 matmul
    route 2        float32, with the LayerNorm absorbed into the weights

Report the error of each against the reference. Route 2 passes only when its
error stays the same order as today's.

Run it:

    .venv/bin/python3 profiling/probes/ln_absorb_probe.py
    .venv/bin/python3 profiling/probes/ln_absorb_probe.py --shape 8 --layers 4
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from appendix_cases import APPENDIX_SHAPES  # noqa: E402
from torch_transformer_benchmark import (  # noqa: E402
    BaselineTransformer,
    LAYER_NORM_EPS,
)


def drift(x: np.ndarray) -> float:
    """
    Return the worst row ratio |mean| / std.

    This is the number that decides route 2. It says how large the term that
    cancels is, against the term that survives. A ratio of 1 costs nothing.
    A ratio of 1000 costs about 10 bits.
    """
    mean = x.mean(axis=-1)
    std = x.std(axis=-1)
    return float(np.max(np.abs(mean) / np.maximum(std, 1e-30)))


def project_today(x64: np.ndarray, gain: np.ndarray, bias: np.ndarray,
                  weight: np.ndarray, proj_bias: np.ndarray,
                  dtype: type) -> np.ndarray:
    """LayerNorm in float32, then the matmul. This is what the model runs."""
    x = x64.astype(dtype)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + dtype(LAYER_NORM_EPS))
    y = (x - mean) * rstd * gain.astype(dtype) + bias.astype(dtype)
    return y @ weight.astype(dtype) + proj_bias.astype(dtype)


def project_route2(x64: np.ndarray, gain: np.ndarray, bias: np.ndarray,
                   weight: np.ndarray, proj_bias: np.ndarray,
                   dtype: type) -> np.ndarray:
    """
    The absorbed form. The three constants are built once, in float32, the
    way a weight build would build them.
    """
    x = x64.astype(dtype)
    g = gain.astype(dtype)
    b = bias.astype(dtype)
    w = weight.astype(dtype)

    # Built at weight build time.
    bw = g[:, None] * w
    c1 = bw.sum(axis=0)
    c2 = b @ w + proj_bias.astype(dtype)

    # Two floats for each row. A separate pass writes them.
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + dtype(LAYER_NORM_EPS))

    # The GEMM, over RAW x, then the epilogue.
    acc = x @ bw
    return rstd * (acc - mean * c1) + c2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, default=6,
                        help="the appendix case id. The default is shape 6, "
                             "which is 66.5%% of the FLOP weight")
    parser.add_argument("--rows", type=int, default=4096,
                        help="rows to test. The full shape 6 chunk does not "
                             "fit a float64 copy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shape = next(s for s in APPENDIX_SHAPES if s.case_id == args.shape)
    config = shape.config()
    torch.manual_seed(args.seed)
    model = BaselineTransformer(config).eval()

    print(f"Shape {shape.case_id}: B{shape.batch_size} D{shape.d_model} "
          f"H{shape.num_heads} S{shape.seq_len} L{shape.num_layers}")
    print(f"{args.rows} rows, atol = 0.002, and the model sits at 2.4e-06\n")

    # Run a real forward, and capture the residual stream at each block. The
    # drift of `x` is the whole question, and only a real forward has it.
    batch = min(shape.batch_size, max(1, args.rows // shape.seq_len))
    x = torch.randn(batch, shape.seq_len, shape.d_model)
    captured = []
    with torch.no_grad():
        h = x
        for block in model.layers:
            captured.append(h.detach().clone())
            h = block(h, None, config.causal)

    header = (f"{'layer':>5} {'|mean|/std':>11} {'today':>11} "
              f"{'route 2':>11} {'ratio':>8}")
    print(header)
    print("-" * len(header))

    worst = 0.0
    for index, block in enumerate(model.layers):
        flat = captured[index].reshape(-1, shape.d_model).numpy()
        flat = flat[:args.rows]
        x64 = flat.astype(np.float64)

        gain = block.norm1.weight.detach().numpy()
        bias = block.norm1.bias.detach().numpy()
        # The qkv projection. torch stores Linear as [out, in], so transpose.
        weight = block.attention.q_proj.weight.detach().numpy().T
        proj_bias = block.attention.q_proj.bias.detach().numpy()

        ref = project_today(x64, gain, bias, weight, proj_bias, np.float64)
        now = project_today(x64, gain, bias, weight, proj_bias, np.float32)
        new = project_route2(x64, gain, bias, weight, proj_bias, np.float32)

        err_now = float(np.max(np.abs(now - ref)))
        err_new = float(np.max(np.abs(new - ref)))
        ratio = err_new / max(err_now, 1e-30)
        worst = max(worst, err_new)
        print(f"{index:>5} {drift(flat):>11.2f} {err_now:>11.3e} "
              f"{err_new:>11.3e} {ratio:>7.1f}x")

    print()
    print(f"worst route 2 error: {worst:.3e}")
    print(f"budget (atol):       {2e-3:.3e}")
    print(f"headroom:            {2e-3 / max(worst, 1e-30):.0f}x")
    print()
    if worst > 2e-4:
        print("VERDICT: route 2 spends most of the accuracy budget. The end "
              "to end error compounds over the layers and the two norms, so "
              "this is a FAIL. Do not build it.")
    else:
        print("VERDICT: route 2 holds the accuracy budget on this test. The "
              "test covers ONE projection, not the whole model, so the "
              "scoreboard accuracy table is still the gate that decides.")


if __name__ == "__main__":
    main()

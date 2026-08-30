"""
Measure what a GEMM tile costs, when a fusion forces the tile.

WHY THIS EXISTS

Rows 43 and 44 both fuse a neighbouring stage into the steel GEMM, and both
constrain the tile to do it. A constrained tile is not free. This script
measures the constraint before the kernel gets built, so a dead route dies
in one minute instead of one day.

    row 43 route 1   the row statistics need the whole row in one A tile,
                     so `bk` must equal K. The 32 KiB threadgroup then caps
                     the tile at `bm + bn <= 62`.
    row 44           the second GEMM needs a whole `hidden` row, so `bn`
                     must cover all of `ffn_dim`.

Run it:

    .venv/bin/python3 profiling/tile_probe.py            # both
    .venv/bin/python3 profiling/tile_probe.py --row 43
    .venv/bin/python3 profiling/tile_probe.py --row 44

RESULT ON AN M3 PRO, MLX 0.32.2

Row 43 route 1 is dead. The best `bk = 128` tile is 0.543x. Row 44 is
affordable. The `bn = 128` tile it needs is 0.884x, and the threadgroup
holds `hidden` beside As and Bs.

Both rows are recorded in OPTIMIZATIONS.md. Read that first.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import mlx.nn as mlx_nn  # noqa: E402

from steel_gemm import fits_threadgroup, steel_addmm, tgp_floats  # noqa: E402

REPEATS = 50
WARMUP = 5

# The shape 6 chunk. `plan_kernels()` gives shape 6 a batch chunk of 1024, so
# one GEMM sees M = 1024 * 128 rows. K is d_model and ffn_dim is d_model, so
# `ffn_in` and `ffn_out` are both K = N = 128.
M, K = 131072, 128
QKV_N = 3 * K
FFN_N = K


def timed(build: Callable[[], mx.array]) -> float:
    """Return the median wall time in ms. It synchronizes every repeat."""
    for _ in range(WARMUP):
        mx.eval(build())
    mx.synchronize()
    samples: List[float] = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        mx.eval(build())
        mx.synchronize()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


def legal(bm: int, bn: int, bk: int, wm: int, wn: int, m: int, n: int,
          k: int) -> bool:
    """
    Return True when the steel kernel accepts the tile.

    Three rules, and all three come from the headers:

    1. The tile must divide the problem, or the kernel takes its safe path.
    2. `bm % (wm * 8)` and `bn % (wn * 8)` must be zero. Below that the MMA
       fragment count is zero, and Metal refuses a zero length array.
    3. As and Bs must fit the 32 KiB threadgroup.
    """
    if m % bm or n % bn or k % bk:
        return False
    if bm % (wm * 8) or bn % (wn * 8):
        return False
    return fits_threadgroup(bm, bn, bk, False, True)


def operands(n: int) -> Tuple[mx.array, mx.array, mx.array]:
    a = mx.random.normal((M, K))
    w = mx.random.normal((n, K))
    b = mx.random.normal((n,))
    mx.eval(a, w, b)
    return a, w, b


def run_row_43() -> None:
    """Measure the `bk = K` tile that route 1 of row 43 forces."""
    print("=" * 78)
    print("Row 43 route 1: the row statistics force bk = K = 128")
    print("A tile needs bm + bn <= 62 at bk = 128, against bm32 bn64 today.")
    print("=" * 78)

    candidates = [(32, 64, 16, 2, 2)]
    for bm in (16, 32, 48):
        for bn in (16, 32):
            for wm, wn in ((1, 1), (2, 1), (1, 2), (2, 2)):
                candidates.append((bm, bn, 128, wm, wn))

    for name, n in (("qkv proj", QKV_N), ("ffn_in", FFN_N)):
        a, w, b = operands(n)
        ref = mx.addmm(b, a, w.T)
        mx.eval(ref)
        print(f"\n{name}: M={M} K={K} N={n}")
        base = None
        for bm, bn, bk, wm, wn in candidates:
            if not legal(bm, bn, bk, wm, wn, M, n, K):
                continue
            try:
                out = steel_addmm(b, a, w, gelu=False, bm=bm, bn=bn, bk=bk,
                                  wm=wm, wn=wn)
                mx.eval(out)
            except Exception:
                print(f"  bm{bm:>3} bn{bn:>3} bk{bk:>3} wm{wm} wn{wn}: "
                      f"does not compile")
                continue
            err = float(mx.max(mx.abs(out - ref)))
            ms = timed(lambda: steel_addmm(b, a, w, gelu=False, bm=bm, bn=bn,
                                           bk=bk, wm=wm, wn=wn))
            if base is None:
                base = ms
            print(f"  bm{bm:>3} bn{bn:>3} bk{bk:>3} wm{wm} wn{wn}: "
                  f"{ms:8.3f} ms  {base / ms:.3f}x  max_abs={err:.2e}")


def run_row_44() -> None:
    """Measure the full width N tile that row 44 forces, and the fit."""
    print("=" * 78)
    print("Row 44: the second GEMM needs a whole hidden row, so bn = ffn_dim")
    print("The threadgroup must hold `hidden` beside As and Bs.")
    print("=" * 78)

    a, w, b = operands(FFN_N)
    ref = mlx_nn.gelu(mx.addmm(b, a, w.T))
    mx.eval(ref)
    print(f"\nffn_in + gelu: M={M} K={K} N={FFN_N}")
    print(f"  mx pair (addmm then gelu): "
          f"{timed(lambda: mlx_nn.gelu(mx.addmm(b, a, w.T))):8.3f} ms")

    candidates = [(32, 64, 16, 2, 2),
                  (16, 128, 16, 1, 2), (16, 128, 16, 2, 2),
                  (32, 128, 16, 2, 2), (32, 128, 16, 2, 4),
                  (64, 128, 16, 2, 2), (64, 128, 16, 4, 2)]
    base = None
    for bm, bn, bk, wm, wn in candidates:
        if not legal(bm, bn, bk, wm, wn, M, FFN_N, K):
            continue
        floats_a, floats_b = tgp_floats(bm, bn, bk, False, True)
        gemm_kib = (floats_a + floats_b) * 4 / 1024
        hidden_kib = bm * FFN_N * 4 / 1024
        try:
            out = steel_addmm(b, a, w, gelu=True, bm=bm, bn=bn, bk=bk,
                              wm=wm, wn=wn)
            mx.eval(out)
        except Exception:
            print(f"  bm{bm:>3} bn{bn:>3} bk{bk:>3} wm{wm} wn{wn}: "
                  f"does not compile")
            continue
        err = float(mx.max(mx.abs(out - ref)))
        ms = timed(lambda: steel_addmm(b, a, w, gelu=True, bm=bm, bn=bn,
                                       bk=bk, wm=wm, wn=wn))
        if base is None:
            base = ms
        verdict = "fits" if gemm_kib + hidden_kib <= 32 else "TOO BIG"
        print(f"  bm{bm:>3} bn{bn:>3} bk{bk:>3} wm{wm} wn{wn}: {ms:8.3f} ms  "
              f"{base / ms:.3f}x  max_abs={err:.2e}  "
              f"gemm {gemm_kib:5.1f} + hidden {hidden_kib:5.1f} KiB "
              f"-> {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=int, choices=(43, 44), default=None,
                        help="measure one row only. The default measures both")
    args = parser.parse_args()

    if args.row in (None, 43):
        run_row_43()
    if args.row is None:
        print()
    if args.row in (None, 44):
        run_row_44()


if __name__ == "__main__":
    main()

"""
Row 50. Fold the final LayerNorm into the `ffn_out` GEMM epilogue.

The final LayerNorm is the one LayerNorm with no GEMM below it, so row 46
cannot reach it. This probe reaches it from the GEMM ABOVE it instead. That
needs a full row tile (`bn == N`) and one simdgroup for each row (`wn == 1`),
so the two `simd_shuffle_xor` steps of row 47 reduce the whole row.

MEASURE IT INTERLEAVED. A first version of this file timed each side in its
own block, ran the big case first, and read 0.534x at M=8192 where an
interleaved A/B reads 1.074x. The allocator state left by the big case moved
a 0.3 ms reading by 2x. Every number below therefore alternates the order
each round, as WORKFLOW.md requires under about 2 ms.

Run:

    .venv/bin/python3 profiling/final_ln_probe.py
"""

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fast_layernorm as fl
from steel_gemm import choose_final_ln_tile, steel_addmm

EPS = 1e-5


def ab(fa, fb, rounds: int = 60):
    """Time two callables, alternating the order each round."""
    for _ in range(5):
        mx.eval(fa())
        mx.synchronize()
        mx.eval(fb())
        mx.synchronize()
    times = {fa: [], fb: []}
    for r in range(rounds):
        for fn in ((fa, fb) if r % 2 == 0 else (fb, fa)):
            start = time.perf_counter()
            mx.eval(fn())
            mx.synchronize()
            times[fn].append((time.perf_counter() - start) * 1e3)
    out = []
    for fn in (fa, fb):
        got = sorted(times[fn])
        out.append(got[len(got) // 2])
    return out


def case(m: int, n: int, k: int, label: str) -> None:
    tile = choose_final_ln_tile(m, n, k)
    if tile is None:
        print(f"{m:>8} {n:>5} {k:>5} | {label:<24} no full row tile fits")
        return
    bm, bn, bk, wm, wn = tile

    mx.random.seed(0)
    a = mx.random.normal((m, k)).astype(mx.float32)
    w = (mx.random.normal((n, k)) * 0.1).astype(mx.float32)
    resid = mx.random.normal((m, n)).astype(mx.float32)
    gain = mx.random.normal((n,)).astype(mx.float32)
    lnb = mx.random.normal((n,)).astype(mx.float32)
    carry = (mx.random.normal((n,)) * 0.1).astype(mx.float32)
    mx.eval(a, w, resid, gain, lnb, carry)

    # What the model ran before row 50: the GEMM, then a LayerNorm pass.
    def today():
        return fl.layer_norm(mx.addmm(resid, a, w.T), gain, lnb, EPS,
                             pre_bias=carry)

    def fused():
        return steel_addmm(resid, a, w, transpose_b=True, bm=bm, bn=bn, bk=bk,
                           wm=wm, wn=wn, final_gain=gain, final_bias=lnb,
                           row_carry=carry, final_eps=EPS)

    err = float(mx.max(mx.abs(fused() - today())))
    t_today, t_fused = ab(today, fused)
    print(f"{m:>8} {n:>5} {k:>5} | {label:<24} {t_today:8.4f} {t_fused:8.4f} "
          f"{t_today / t_fused:6.3f}x   max_abs {err:.2e}")


if __name__ == "__main__":
    print(f"{'M':>8} {'N':>5} {'K':>5} | {'shape':<24} {'today':>8} "
          f"{'fused':>8} {'ratio':>7}   interleaved, median of 60")
    case(1024 * 128, 128, 128, "shape 6, one chunk")
    case(64 * 1024, 128, 128, "shape 13")
    case(128 * 128, 128, 128, "shape 5")
    case(64 * 128, 128, 128, "shape 1")
    case(64 * 128, 32, 32, "shape 7")
    case(64 * 32, 128, 128, "shape 12")
    case(8192, 1024, 1024, "shape 8")

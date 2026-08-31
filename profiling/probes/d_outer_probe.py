"""
Check the D-blocked attention kernel, then time it against the path shape 8
runs today.

Run it before any model integration. Row 41 did the same for the narrow
block, and one measurement killed that idea before the integration work.

    .venv/bin/python3 profiling/probes/d_outer_probe.py
    .venv/bin/python3 profiling/probes/d_outer_probe.py --repeats 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import d_outer_attention as do

# Shape 8 attention: B64 D1024 H4 S128 gives head_dim 256.
SHAPE_8 = dict(batch=64, heads=4, seq=128, head_dim=256)

# The block shapes that fit 32 KiB. `bq` fixes the warp count: wm = bq // 8.
CANDIDATES: List[Tuple[int, int, int]] = [
    (bq, bk, bdc)
    for bq in (8, 16)
    for bk in (8, 16, 32, 64)
    for bdc in (16, 32, 64, 128)
]


def reference(q, k, v, scale):
    """The path shape 8 runs today: MLX picks its fallback at head_dim 256."""
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                                mask="causal")


def make_inputs(batch, heads, seq, head_dim, seed=0):
    mx.random.seed(seed)
    shape = (batch, heads, seq, head_dim)
    q = mx.random.normal(shape, dtype=mx.float32)
    k = mx.random.normal(shape, dtype=mx.float32)
    v = mx.random.normal(shape, dtype=mx.float32)
    mx.eval(q, k, v)
    mx.synchronize()
    return q, k, v


def check(bq, bk, bdc, cfg=SHAPE_8, seed=0) -> Tuple[bool, str]:
    q, k, v = make_inputs(**cfg, seed=seed)
    scale = 1.0 / (cfg["head_dim"] ** 0.5)

    want = reference(q, k, v, scale)
    mx.eval(want)
    mx.synchronize()

    try:
        got = do.d_blocked_attention(q, k, v, scale, causal=True,
                                     bq=bq, bk=bk, bdc=bdc)
        mx.eval(got)
        mx.synchronize()
    except Exception as exc:  # a compile failure lands here
        return False, str(exc).splitlines()[0][:160]

    err = float(mx.max(mx.abs(got - want)))
    ok = err < 2e-5
    return ok, f"max_abs={err:.2e}"


def median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def time_call(fn, q, k, v, scale, repeats: int) -> float:
    """
    Median wall time of one call, in ms.

    `mx.eval` and `mx.synchronize` go INSIDE the loop. MLX is lazy: a loop
    that overwrites one output and evaluates it once builds the graph
    `repeats` times and runs it once. That read 0.03 ms for a call the
    stage roofline puts at 2.65 ms. Both paths pay the same round trip, so
    the ratio stands. See `timed()` in `stage_roofline.py`.
    """
    for _ in range(3):
        mx.eval(fn(q, k, v, scale))
    mx.synchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn(q, k, v, scale))
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return median(samples)


def bench(bq, bk, bdc, cfg=SHAPE_8, repeats=100, rounds=3):
    """Interleave the two paths, and alternate which one runs first."""
    q, k, v = make_inputs(**cfg)
    scale = 1.0 / (cfg["head_dim"] ** 0.5)

    def theirs(q, k, v, scale):
        return reference(q, k, v, scale)

    def ours(q, k, v, scale):
        return do.d_blocked_attention(q, k, v, scale, causal=True,
                                      bq=bq, bk=bk, bdc=bdc)

    ref_ms: List[float] = []
    new_ms: List[float] = []
    for r in range(rounds):
        if r % 2 == 0:
            ref_ms.append(time_call(theirs, q, k, v, scale, repeats))
            new_ms.append(time_call(ours, q, k, v, scale, repeats))
        else:
            new_ms.append(time_call(ours, q, k, v, scale, repeats))
            ref_ms.append(time_call(theirs, q, k, v, scale, repeats))
    return median(ref_ms), median(new_ms)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if do.upstream_moved():
        print("WARNING: Apple's steel_attention.h changed since this fork "
              "was read. Check the softmax, the online rescale and the "
              "causal mask before you trust a number.\n")

    cfg = SHAPE_8
    print(f"shape 8 attention: B{cfg['batch']} H{cfg['heads']} "
          f"S{cfg['seq']} head_dim={cfg['head_dim']}, causal, float32")
    print(f"threadgroup budget: {do.THREADGROUP_BYTES / 1024:.0f} KiB\n")

    print(f"{'bq':>3}{'bk':>4}{'bdc':>5}{'warps':>7}{'smem KiB':>10}"
          f"  {'accuracy':<22}")
    print("-" * 56)

    good: List[Tuple[int, int, int]] = []
    for bq, bk, bdc in CANDIDATES:
        if not do.supports(cfg["seq"], cfg["head_dim"], bq, bk, bdc):
            print(f"{bq:>3}{bk:>4}{bdc:>5}{bq // 8:>7}"
                  f"{do.smem_kib(bq, bk, cfg['head_dim'], bdc):>10.2f}"
                  f"  does not fit")
            continue
        ok, info = check(bq, bk, bdc, cfg)
        mark = "PASS" if ok else "FAIL"
        print(f"{bq:>3}{bk:>4}{bdc:>5}{bq // 8:>7}"
              f"{do.smem_kib(bq, bk, cfg['head_dim'], bdc):>10.2f}"
              f"  {mark} {info}")
        if ok:
            good.append((bq, bk, bdc))

    if args.check_only or not good:
        return 0 if good else 1

    print(f"\ntiming: {args.rounds} interleaved rounds of {args.repeats} "
          f"repeats, median of rounds")
    print(f"{'bq':>3}{'bk':>4}{'bdc':>5}{'fallback ms':>13}"
          f"{'d-blocked ms':>14}{'speedup':>9}")
    print("-" * 48)
    best = None
    for bq, bk, bdc in good:
        ref_ms, new_ms = bench(bq, bk, bdc, cfg, args.repeats, args.rounds)
        ratio = ref_ms / new_ms
        print(f"{bq:>3}{bk:>4}{bdc:>5}{ref_ms:>13.4f}{new_ms:>14.4f}"
              f"{ratio:>8.3f}x")
        if best is None or ratio > best[0]:
            best = (ratio, bq, bk, bdc)

    if best:
        ratio, bq, bk, bdc = best
        print(f"\nbest: bq{bq} bk{bk} bdc{bdc} at {ratio:.3f}x")
        if ratio <= 1.0:
            print("It loses. Record it and stop, as rows 26 and 41 did.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

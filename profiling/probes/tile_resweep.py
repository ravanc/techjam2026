#!/usr/bin/env python3
"""Re-sweep the steel GEMM tile, with the row 46 and row 47 epilogues ON.

WHY THIS EXISTS

`_TILES` in `steel_gemm.py` was ordered by a sweep of a PLAIN GEMM. Two
epilogues have arrived since:

    row 46   the LayerNorm epilogue on `qkv proj` and `ffn_in`. It reads two
             floats for the row and two (N,) vectors.
    row 47   the row statistics epilogue on `out proj` and `ffn_out`. It
             writes `wn * (N / bn)` partial planes.

Both change what a tile costs, and neither row re-swept the tile. Row 47 in
particular gets cheaper as `bn` grows: at N=128 a `bn64 wn2` tile writes 4
partial planes and a `bn128 wn2` tile writes 2.

    .venv/bin/python3 profiling/probes/tile_resweep.py --grid coarse
    .venv/bin/python3 profiling/probes/tile_resweep.py --grid full --stages ffn_in

Every stage runs at the shape 6 chunk, which carries 66.5% of the FLOP
weight: M = 1024 * 128 = 131072, K = d_model = 128.

The baseline of each stage is the tile `choose_tile()` picks today.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx  # noqa: E402

from steel_gemm import (  # noqa: E402
    choose_tile, fits_threadgroup, loader_geometry_ok, row_stats_reduce,
    steel_addmm,
)

M, K = 131072, 128
REPEATS = 30
WARMUP = 3

BM = [16, 32, 64]
BN = [32, 64, 128]
BK = [8, 16, 32]
WMWN = [(2, 2), (4, 1), (1, 4), (4, 2), (2, 4)]

# Each stage names its N, the stored layout of B, and the epilogue it carries.
STAGES = {
    "qkv proj": dict(n=3 * K, transpose_b=False, kind="ln", gelu=False),
    "out proj": dict(n=K, transpose_b=True, kind="stats", gelu=False),
    "ffn_in":   dict(n=K, transpose_b=True, kind="ln", gelu=True),
    "ffn_out":  dict(n=K, transpose_b=True, kind="stats", gelu=False),
}


def operands(spec: dict) -> dict:
    """Build one set of operands for a stage. Every tile reuses them."""
    n, transpose_b = spec["n"], spec["transpose_b"]
    a = mx.random.normal((M, K))
    b = (mx.random.normal((n, K)) if transpose_b
         else mx.random.normal((K, n)))
    out = dict(a=a, b=b, transpose_b=transpose_b, gelu=spec["gelu"])
    if spec["kind"] == "ln":
        out["bias"] = mx.random.normal((n,))
        out["rowstat"] = mx.random.normal((M, 2))
        out["lnc1"] = mx.random.normal((n,))
        out["lnc2"] = mx.random.normal((n,))
    else:
        # Row 47 takes the residual as a MATRIX C, and it writes partials.
        out["bias"] = mx.random.normal((M, n))
        out["row_carry"] = mx.random.normal((n,))
    mx.eval(*[v for v in out.values() if isinstance(v, mx.array)])
    mx.synchronize()
    return out


def run_once(spec: dict, ops: dict, tile: Tuple[int, int, int, int, int]):
    """Run the stage once with this tile. Include the row 47 reduce."""
    bm, bn, bk, wm, wn = tile
    if spec["kind"] == "ln":
        return steel_addmm(
            ops["bias"], ops["a"], ops["b"],
            transpose_b=ops["transpose_b"], gelu=ops["gelu"],
            bm=bm, bn=bn, bk=bk, wm=wm, wn=wn,
            rowstat=ops["rowstat"], lnc1=ops["lnc1"], lnc2=ops["lnc2"])
    # The reduce is part of the cost. Row 47 cannot use the partials without
    # it, and its size follows `wn * (N / bn)`, so it belongs in the timing.
    x, partials = steel_addmm(
        ops["bias"], ops["a"], ops["b"], transpose_b=ops["transpose_b"],
        bm=bm, bn=bn, bk=bk, wm=wm, wn=wn,
        row_stats=True, row_carry=ops["row_carry"])
    return x, row_stats_reduce(partials, spec["n"], 1e-5)


def build_ok(spec: dict, ops: dict, tile) -> bool:
    """Compile and run the tile once. False when it does not build."""
    try:
        for _ in range(WARMUP):
            mx.eval(run_once(spec, ops, tile))
        mx.synchronize()
        return True
    except Exception:
        return False


def paired(spec: dict, ops: dict, tile, base_tile, repeats: int):
    """Time the candidate against today's tile, ALTERNATING each repeat.

    A plain sweep cannot score these tiles. The machine drifts inside one
    run: an early reading of the tile in use gave 2.0771 ms and a later one
    in the same process gave 1.9488 ms, which is 6.6%. That is larger than
    the difference between the tiles. So every repeat runs both sides, and
    the order swaps on every other repeat.
    """
    cand: List[float] = []
    base: List[float] = []
    for index in range(repeats):
        order = [(tile, cand), (base_tile, base)]
        if index % 2:
            order.reverse()
        for which, sink in order:
            start = time.perf_counter()
            mx.eval(run_once(spec, ops, which))
            mx.synchronize()
            sink.append((time.perf_counter() - start) * 1e3)
    return statistics.median(cand), statistics.median(base)


def legal(spec: dict, grid: str) -> List[Tuple[int, int, int, int, int]]:
    n, transpose_b = spec["n"], spec["transpose_b"]
    pairs = [(2, 2)] if grid == "coarse" else WMWN
    out = []
    for bm in BM:
        for bn in BN:
            for bk in BK:
                for wm, wn in pairs:
                    if M % bm or n % bn or K % bk:
                        continue
                    if not fits_threadgroup(bm, bn, bk, False, transpose_b):
                        continue
                    if not loader_geometry_ok(bm, bn, bk, wm, wn, False,
                                              transpose_b):
                        continue
                    out.append((bm, bn, bk, wm, wn))
    return out


def main() -> int:
    global REPEATS
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=("coarse", "full"), default="coarse")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--only", default="",
                    help="test these tiles only, e.g. 64x32x16x4x1,32x64x32x2x2")
    args = ap.parse_args()
    REPEATS = args.repeats

    print(f"M={M} K={K}, the shape 6 chunk. grid={args.grid} "
          f"repeats={REPEATS}")
    print()

    for name in [s.strip() for s in args.stages.split(",") if s.strip()]:
        spec = STAGES[name]
        tiles = legal(spec, args.grid)
        if args.only:
            picked = {tuple(int(v) for v in t.split("x"))
                      for t in args.only.split(",") if t}
            tiles = [t for t in tiles if t in picked]
        today = choose_tile(M, spec["n"], K, transpose_b=spec["transpose_b"])
        ops = operands(spec)

        print(f"=== {name}  N={spec['n']} transpose_b={spec['transpose_b']} "
              f"epilogue={spec['kind']}  {len(tiles)} tiles")
        if today is None:
            print("    no tile divides this stage")
            print()
            continue
        print(f"    today  {'x'.join(map(str, today))}")

        rows = []
        for index, tile in enumerate(tiles, 1):
            print(f"\r    {index}/{len(tiles)}", end="", flush=True)
            if not build_ok(spec, ops, tile):
                continue
            ms, base_ms = paired(spec, ops, tile, today, args.repeats)
            rows.append((ms / base_ms, ms, base_ms, tile))
        print("\r" + " " * 20 + "\r", end="")

        rows.sort()
        print(f"    {'tile':<18}{'ms':>9}{'today ms':>10}{'ratio':>9}")
        for ratio, ms, base_ms, tile in rows[:args.top]:
            mark = " <- today" if tuple(today) == tile else ""
            print(f"    {'x'.join(map(str, tile)):<18}{ms:>9.4f}"
                  f"{base_ms:>10.4f}{1.0 / ratio:>8.3f}x{mark}")
        for place, (ratio, ms, base_ms, tile) in enumerate(rows, 1):
            if tuple(today) == tile and place > args.top:
                print(f"    ... today is #{place} of {len(rows)}, "
                      f"self ratio {1.0 / ratio:.3f}x")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

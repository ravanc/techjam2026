"""
A/B one KernelPlan field against another, on one shape, interleaved.

WHY THIS EXISTS

A sweep compares two runs that are minutes or hours apart, so its ratio holds
the machine as well as the code. A shape under about 2 ms cannot be scored
that way: OPTIMIZATIONS.md row 47 read 0.922x at shape 12 from two sweeps,
and this script read 1.008x to 1.013x for the same change.

This script builds the SAME model twice in one process, sets `plan_override`
on each, and alternates the order each round so neither side always runs on
a cold chip.

It toggles row 50 today. Change the `dataclasses.replace` call below to
toggle another field.

    .venv/bin/python3 profiling/probes/plan_ab.py --cases 12
    .venv/bin/python3 profiling/probes/plan_ab.py --cases 1,3,4,12 --repeats 150
"""
import sys, time, dataclasses, argparse
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, mlx.core as mx
from appendix_cases import APPENDIX_SHAPES
from torch_transformer_benchmark import (
    UserOptimizedTransformer, plan_kernels)

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="12")
ap.add_argument("--repeats", type=int, default=200)
ap.add_argument("--rounds", type=int, default=5)
args = ap.parse_args()

def build(shape, off):
    cfg = shape.config()
    torch.manual_seed(0)
    m = UserOptimizedTransformer(cfg).eval()
    plan = plan_kernels(cfg, 4)
    if off:
        # The field under test. Row 50 today.
        plan = dataclasses.replace(plan, final_ln=None)
    m.plan_override = plan
    return m, plan

def time_once(model, x, repeats):
    for _ in range(5):
        model(x); mx.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]

for case in [int(c) for c in args.cases.split(",")]:
    shape = next(s for s in APPENDIX_SHAPES if s.case_id == case)
    torch.manual_seed(0)
    x = torch.randn(shape.batch_size, shape.seq_len, shape.d_model)

    on, plan_on = build(shape, off=False)
    off, plan_off = build(shape, off=True)
    rows = (plan_on.batch_chunk or shape.batch_size) * shape.seq_len

    a, b = [], []
    for r in range(args.rounds):
        # Alternate the order each round, so neither side always runs cold.
        if r % 2 == 0:
            b.append(time_once(off, x, args.repeats))
            a.append(time_once(on, x, args.repeats))
        else:
            a.append(time_once(on, x, args.repeats))
            b.append(time_once(off, x, args.repeats))
    a.sort(); b.sort()
    ma, mb = a[len(a)//2], b[len(b)//2]
    print(f"case {case:>2}  B{shape.batch_size} S{shape.seq_len} "
          f"D{shape.d_model} L{shape.num_layers}  rows={rows}")
    print(f"    OFF    {mb:8.4f} ms   {['%.4f'%v for v in b]}")
    print(f"    ON     {ma:8.4f} ms   {['%.4f'%v for v in a]}")
    print(f"    ratio      {mb/ma:8.4f}x")

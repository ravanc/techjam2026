"""
Force `steel_attention=True` on every appendix shape, and compare it against
the plan the model uses today.

Shapes 1-7, 11, 12 and 13 already run the steel kernel, so ON equals OFF
there. Those rows are the null control: they give the noise floor of this
harness. Shapes 8, 9 and 10 are the real test.

The repeat count scales to a fixed time budget, so shape 6 does not hold the
GPU for an hour.

    .venv/bin/python3 profiling/probes/steel_gate_ab.py

See OPTIMIZATIONS.md row 56.
"""
import sys, os, time, dataclasses, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, mlx.core as mx
from appendix_cases import APPENDIX_SHAPES
from torch_transformer_benchmark import UserOptimizedTransformer, plan_kernels

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10,11,12,13")
ap.add_argument("--budget-ms", type=float, default=1500.0)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--max-repeats", type=int, default=200)
args = ap.parse_args()


def build(shape, steel):
    cfg = shape.config()
    torch.manual_seed(0)
    model = UserOptimizedTransformer(cfg).eval()
    model.plan_override = dataclasses.replace(
        plan_kernels(cfg, 4), steel_attention=steel)
    return model


def time_once(model, x, repeats):
    for _ in range(3):
        model(x); mx.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x)
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


print(f"{'#':>3} {'shape':24s} {'today':>10} {'steel ON':>10} "
      f"{'ratio':>9}  {'rep':>4}  note")
for case in [int(c) for c in args.cases.split(",")]:
    shape = next(s for s in APPENDIX_SHAPES if s.case_id == case)
    cfg = shape.config()
    head_dim = cfg.d_model // cfg.num_heads
    today_steel = plan_kernels(cfg, 4).steel_attention
    torch.manual_seed(0)
    x = torch.randn(shape.batch_size, shape.seq_len, shape.d_model)
    name = (f"B{shape.batch_size} D{shape.d_model} "
            f"H{shape.num_heads} S{shape.seq_len} d{head_dim}")

    off = build(shape, today_steel)
    try:
        one = time_once(off, x, 3)
    except RuntimeError as error:
        print(f"{case:>3} {name:24s} {'':>10} {'':>10} {'':>9}  "
              f"{'':>4}  today FAILS: {str(error).splitlines()[-1]}")
        continue
    repeats = max(5, min(args.max_repeats, int(args.budget_ms / max(one, 1e-3))))

    try:
        on = build(shape, True)
        time_once(on, x, 3)
    except RuntimeError as error:
        print(f"{case:>3} {name:24s} {'':>10} {'':>10} {'':>9}  "
              f"{'':>4}  steel ON FAILS: {str(error).splitlines()[-1]}")
        del off
        continue

    a, b = [], []
    for index in range(args.rounds):
        if index % 2 == 0:
            b.append(time_once(off, x, repeats))
            a.append(time_once(on, x, repeats))
        else:
            a.append(time_once(on, x, repeats))
            b.append(time_once(off, x, repeats))
    a.sort(); b.sort()
    median_on, median_off = a[len(a)//2], b[len(b)//2]
    note = "null control" if today_steel else "REAL TEST"
    print(f"{case:>3} {name:24s} {median_off:>10.4f} {median_on:>10.4f} "
          f"{median_off/median_on:>8.4f}x  {repeats:>4}  {note}")
    del off, on

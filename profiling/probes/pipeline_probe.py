"""
Row 48. Can the framework boundary hide behind the GPU? It cannot.

WHAT THE IDEA WAS

`forward()` converts the input to MLX before it queues any GPU work, so the
GPU idles through the copy. At shape 6 that copy is 13.95 ms of a 446.5 ms
call, 3.1%, and the boundary as a whole is 5.4%. The obvious fix is to
convert chunk i+1 while the GPU runs chunk i.

WHY IT CANNOT WORK, AND THE MEASUREMENT THAT PROVES IT

This machine has UNIFIED memory. The CPU copy and the GPU kernels read and
write the same DRAM through the same controller, so they do not overlap:
they contend. There is no separate bus to hide a transfer on, which is the
thing that makes this trick pay on a discrete GPU.

The shape 6 block is memory bound. `stage_roofline.py` puts `out proj` at
105% and `ffn_out` at 110% of the 128 GB/s roof. So every byte the CPU moves
comes straight out of the GPU's throughput.

Arm B adds 625 MiB of UNRELATED CPU memcpy to the chunk loop. That memcpy
costs 34.53 ms when it runs alone. Inside the loop it costs 31.31 ms, so
**91% of it does not hide**.

Run it:

    .venv/bin/python3 profiling/probes/pipeline_probe.py
    .venv/bin/python3 profiling/probes/pipeline_probe.py --case 6 --repeats 7
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import mlx.core as mx  # noqa: E402

from appendix_cases import APPENDIX_SHAPES  # noqa: E402
from torch_transformer_benchmark import (  # noqa: E402
    UserOptimizedTransformer, _to_mlx, _to_torch)


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=6,
                        help="the appendix case id. It must select a chunked "
                             "plan; only shape 6 does")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    shape = next(s for s in APPENDIX_SHAPES if s.case_id == args.case)
    config = shape.config()
    torch.manual_seed(0)
    model = UserOptimizedTransformer(config).eval()
    x = torch.randn(shape.batch_size, shape.seq_len, shape.d_model)
    mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool)

    model(x)
    mx.synchronize()
    chunk = model.plan.batch_chunk
    if not chunk:
        raise SystemExit(
            f"shape {args.case} does not chunk, so it has no loop to "
            f"pipeline. Use shape 6.")
    call = model._mlx_call[False]
    batch = x.shape[0]
    layers, final = model._mlx_layers, model._mlx_final

    bulk_x = _to_mlx(x)
    bulk_mask = _to_mlx(mask)
    mx.eval(bulk_x, bulk_mask)

    # One chunk of unrelated bytes, for the contention arm.
    junk_src = np.random.rand(
        chunk, shape.seq_len, shape.d_model).astype(np.float32)

    def join(parts):
        out = mx.concatenate(parts, axis=0)
        mx.eval(out)
        return out

    def today():
        """Convert everything first, then loop. This is the model today."""
        mlx_x, mlx_mask = _to_mlx(x), _to_mlx(mask)
        parts = []
        for start in range(0, batch, chunk):
            stop = min(start + chunk, batch)
            part = call(mlx_x[start:stop], mlx_mask[start:stop], layers, *final)
            mx.eval(part)
            parts.append(part)
        return join(parts)

    def convert_in_loop():
        """Convert chunk by chunk, still evaluating each chunk at once."""
        parts = []
        for start in range(0, batch, chunk):
            stop = min(start + chunk, batch)
            part = call(_to_mlx(x[start:stop]), _to_mlx(mask[start:stop]),
                        layers, *final)
            mx.eval(part)
            parts.append(part)
        return join(parts)

    def pipeline():
        """Convert chunk i+1 while the GPU runs chunk i."""
        parts = []
        pending = None
        for start in range(0, batch, chunk):
            stop = min(start + chunk, batch)
            part = call(_to_mlx(x[start:stop]), _to_mlx(mask[start:stop]),
                        layers, *final)
            if pending is not None:
                mx.eval(pending)
            parts.append(part)
            pending = part
        mx.eval(pending)
        return join(parts)

    def loop_alone():
        """The loop with the input already converted. The contention control."""
        parts = []
        for start in range(0, batch, chunk):
            stop = min(start + chunk, batch)
            part = call(bulk_x[start:stop], bulk_mask[start:stop],
                        layers, *final)
            mx.eval(part)
            parts.append(part)
        return join(parts)

    def loop_with_memcpy():
        """`loop_alone` plus an UNRELATED CPU memcpy of one chunk each pass."""
        parts = []
        for start in range(0, batch, chunk):
            stop = min(start + chunk, batch)
            part = call(bulk_x[start:stop], bulk_mask[start:stop],
                        layers, *final)
            junk = junk_src.copy()
            mx.eval(part)
            parts.append(part)
            del junk
        return join(parts)

    def loop_lagged():
        """Keep the bulk convert, but never let the GPU queue drain."""
        parts = []
        pending = None
        for start in range(0, batch, chunk):
            stop = min(start + chunk, batch)
            part = call(bulk_x[start:stop], bulk_mask[start:stop],
                        layers, *final)
            if pending is not None:
                mx.eval(pending)
            parts.append(part)
            pending = part
        mx.eval(pending)
        return join(parts)

    arms = [
        ("today", today),
        ("convert in loop", convert_in_loop),
        ("pipeline", pipeline),
        ("loop alone", loop_alone),
        ("loop + memcpy", loop_with_memcpy),
        ("loop, lagged eval", loop_lagged),
    ]

    # Correctness. Every arm must return the same bytes.
    reference = None
    for name, fn in arms:
        out = fn()
        mx.eval(out)
        mx.synchronize()
        result = _to_torch(out, dtype=x.dtype, device=x.device).clone()
        if reference is None:
            reference = result
        elif not torch.equal(reference, result):
            raise SystemExit(f"{name} does not match `today`")
    print("every arm is bit equal")

    peaks = {}
    for name, fn in arms:
        mx.synchronize()
        mx.reset_peak_memory()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        peaks[name] = mx.get_peak_memory() / 2**30
        out = None

    results = {name: [] for name, _ in arms}
    for round_index in range(args.rounds):
        order = arms if round_index % 2 == 0 else list(reversed(arms))
        for name, fn in order:
            for _ in range(2):
                out = fn()
                mx.synchronize()
                out = None
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                out = fn()
                mx.eval(out)
                mx.synchronize()
                samples.append((time.perf_counter() - start) * 1e3)
                out = None
            results[name].append(median(samples))

    base = median(results["today"])
    alone = median(results["loop alone"])
    n_chunks = -(-batch // chunk)
    print(f"\nshape {args.case}: B{batch} chunk={chunk}, {n_chunks} chunks")
    print(f"{'arm':<20}{'ms':>10}{'vs today':>10}{'peak GiB':>10}")
    for name, _ in arms:
        value = median(results[name])
        print(f"{name:<20}{value:>10.3f}{base / value:>9.3f}x"
              f"{peaks[name]:>10.2f}")

    # The contention calibration. Time the same memcpy with no GPU work.
    for _ in range(3):
        junk = junk_src.copy()
    start = time.perf_counter()
    for _ in range(n_chunks):
        junk = junk_src.copy()
    solo = (time.perf_counter() - start) * 1e3
    inside = median(results["loop + memcpy"]) - alone
    print()
    print(f"{n_chunks} x {junk_src.nbytes / 2**20:.1f} MiB of CPU memcpy")
    print(f"  alone                 {solo:.2f} ms")
    print(f"  added to the loop     {inside:+.2f} ms")
    print(f"  hidden                {(1 - inside / solo) * 100:.0f}%")
    print()
    if inside > 0.5 * solo:
        print("VERDICT: the CPU copy and the GPU kernels CONTEND for one "
              "memory system. Overlapping them cannot pay. Convert in bulk, "
              "while the GPU is idle.")
    else:
        print("VERDICT: the copy hides. Pipeline the loop.")


if __name__ == "__main__":
    main()

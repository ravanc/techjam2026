#!/usr/bin/env python3
"""Run the transformer benchmark under hardware profiling.

Two modes:

  --mode signpost   Emit os_signpost intervals around each model's forward
                    passes so an Instruments recording (see trace.sh) shows
                    which GPU work belongs to which implementation.

  --mode gputrace   Write a .gputrace bundle per backend for the Xcode Metal
                    debugger, giving per-kernel timings and occupancy.

Both modes reuse the models and inputs from torch_transformer_benchmark.py so
what gets profiled is what gets benchmarked.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROFILING = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PROFILING)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import gpucapture  # noqa: E402
import signposts  # noqa: E402


def build(config, device, dtype, seed):
    """Construct baseline + optimized models with identical weights."""
    import torch

    import torch_transformer_benchmark as B

    torch.manual_seed(seed)
    baseline = B.BaselineTransformer(config).to(device=device, dtype=dtype).eval()
    optimized = B.UserOptimizedTransformer(config).to(device=device, dtype=dtype).eval()
    B.copy_model_weights(baseline, optimized, strict=True)

    x, valid_mask = B.generate_random_case(
        config, device, dtype, seed=seed, padding_ratio=0.0, input_scale=1.0
    )
    return baseline, optimized, x, valid_mask


def run_signposts(args) -> int:
    import torch

    import torch_transformer_benchmark as B

    device = B.resolve_device(args.device)
    dtype = B.resolve_dtype(args.dtype)
    config = B.TransformerConfig(
        args.batch_size, args.seq_len, args.d_model,
        args.heads, args.ffn_dim, args.layers, args.causal,
    )
    config.validate()

    baseline, optimized, x, valid_mask = build(config, device, dtype, args.seed)

    print(f"device={device} dtype={dtype} signposts={signposts.available()}")
    if not signposts.available():
        print("[warning] signpost shim unavailable; run `make -C profiling/tools`")

    def drive(model, label):
        with torch.inference_mode():
            for _ in range(args.warmup):  # keep warmup out of the trace
                model(x, valid_mask)
            _sync(device)
            with signposts.interval(f"{label}-total"):
                for i in range(args.iterations):
                    with signposts.interval(f"{label}-iter"):
                        model(x, valid_mask)
                        _sync(device)
        signposts.event(f"{label}-done")

    drive(baseline, "baseline")
    drive(optimized, "optimized")
    print(f"emitted signposts for {args.iterations} iterations per model")
    return 0


def _sync(device) -> None:
    """Wait for GPU work so signpost intervals bound real completion."""
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    try:
        import mlx.core as mx

        mx.synchronize()
    except ImportError:
        pass


def run_gputrace(args) -> int:
    # Metal reads MTL_CAPTURE_ENABLED at init, so fix the environment first.
    gpucapture.ensure_capture_env()

    import torch

    import torch_transformer_benchmark as B

    device = B.resolve_device(args.device)
    dtype = B.resolve_dtype(args.dtype)
    config = B.TransformerConfig(
        args.batch_size, args.seq_len, args.d_model,
        args.heads, args.ffn_dim, args.layers, args.causal,
    )
    config.validate()

    baseline, optimized, x, valid_mask = build(config, device, dtype, args.seed)
    outdir = os.path.abspath(args.outdir)

    with torch.inference_mode():
        # Warm up outside capture so one-time allocation and shader compilation
        # do not dominate the trace.
        for _ in range(args.warmup):
            baseline(x, valid_mask)
            optimized(x, valid_mask)
        _sync(device)

        written = []
        if device.type == "mps":
            path = os.path.join(outdir, "baseline_mps.gputrace")
            with gpucapture.mps_capture(path):
                for _ in range(args.iterations):
                    baseline(x, valid_mask)
            written.append(path)
        else:
            print(f"[skip] baseline capture needs --device mps (got {device.type})")

        path = os.path.join(outdir, "optimized_mlx.gputrace")
        with gpucapture.mlx_capture(path):
            for _ in range(args.iterations):
                optimized(x, valid_mask)
        written.append(path)

    for p in written:
        print(f"wrote {p}")
    print("open with: open <path>   (requires Xcode)")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("signpost", "gputrace"), default="signpost")
    p.add_argument("--device", default="mps", help="mps, cpu, cuda, auto")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                   default="float32")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=2048)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--outdir", default=os.path.join(_PROFILING, "traces"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "signpost":
        return run_signposts(args)
    return run_gputrace(args)


if __name__ == "__main__":
    raise SystemExit(main())

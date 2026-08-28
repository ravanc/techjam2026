#!/usr/bin/env python3
"""
Run the default benchmark across three backends and compare them side by side:

    cpu  - BaselineTransformer, torch CPU kernels
    mps  - BaselineTransformer, torch Metal kernels
    mlx  - UserOptimizedTransformer, the MLX implementation

All three are given byte-identical inputs (see bench_cases.py) and identical
weights, so latency and accuracy are directly comparable.

Timing notes:

  * Every backend is timed with a device synchronize on both sides of the call.
    MPS and MLX dispatch asynchronously, so the harness's non-CUDA path
    (`benchmark_once`, which only wraps `perf_counter_ns` around the call and
    only synchronizes for CUDA) measures enqueue time on those backends, not
    execution time.
  * Backends are timed round-robin, and the order is reversed on odd rounds, to
    spread thermal drift and clock ramping across all of them.

Accuracy is measured against the CPU baseline at the same dtype, which is the
comparison the harness itself makes.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from bench_cases import TestCase, make_accuracy_cases, make_timing_case
from torch_transformer_benchmark import (
    BaselineTransformer,
    TimingResult,
    TransformerConfig,
    UserOptimizedTransformer,
    compare_outputs,
    copy_model_weights,
    resolve_dtype,
)

ALL_BACKENDS = ("cpu", "mps", "mlx")


def make_sync(device: torch.device) -> Callable[[], None]:
    """A blocking synchronize for `device`, so timing covers execution."""
    if device.type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    if device.type == "mps":
        return torch.mps.synchronize
    return lambda: None


@dataclass
class Backend:
    name: str
    label: str
    model: nn.Module
    device: torch.device
    dtype: torch.dtype
    sync: Callable[[], None]
    samples_ms: List[float] = field(default_factory=list)
    accuracy: Optional[Dict] = None

    def run(self, case: TestCase) -> torch.Tensor:
        """Run one case and bring the output back to the CPU in float32."""
        local = case.to(self.device, self.dtype)
        with torch.inference_mode():
            output = self.model(local.x, local.valid_mask)
        self.sync()
        return output.detach().to(device="cpu", dtype=torch.float32)


def build_backends(
    names: List[str],
    config: TransformerConfig,
    dtype: torch.dtype,
    seed: int,
    mlx_torch_device: str,
) -> List[Backend]:
    """Build one model per backend, all sharing the CPU baseline's weights."""
    torch.manual_seed(seed)
    reference_model = BaselineTransformer(config)

    backends: List[Backend] = []
    for name in names:
        if name == "mlx":
            model: nn.Module = UserOptimizedTransformer(config)
            copy_model_weights(reference_model, model, strict=True)
            device = torch.device(mlx_torch_device)
            label = f"mlx (torch tensors on {device.type})"
        else:
            model = copy.deepcopy(reference_model)
            device = torch.device(name)
            label = f"torch baseline on {name}"

        model = model.to(device=device, dtype=dtype).eval()
        backends.append(
            Backend(
                name=name,
                label=label,
                model=model,
                device=device,
                sync=make_sync(device),
                dtype=dtype,
            )
        )
    return backends


def check_accuracy(
    backends: List[Backend],
    cases: List[TestCase],
    reference: Backend,
    rtol: float,
    atol: float,
) -> None:
    print("\n=== Accuracy vs the CPU baseline ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")

    references = [reference.run(case) for case in cases]

    for backend in backends:
        max_abs = 0.0
        max_rel = 0.0
        failed = 0
        total = 0
        for case, expected in zip(cases, references):
            result = compare_outputs(expected, backend.run(case), rtol=rtol, atol=atol)
            max_abs = max(max_abs, result.max_abs_error)
            max_rel = max(max_rel, result.max_relative_error)
            failed += result.failed_elements
            total += result.total_elements

        backend.accuracy = {
            "passed": failed == 0,
            "max_abs_error": max_abs,
            "max_relative_error": max_rel,
            "failed_elements": failed,
            "total_elements": total,
        }
        note = " (reference)" if backend is reference else ""
        print(
            f"{backend.name:<4}: {'PASS' if failed == 0 else 'FAIL'} | "
            f"max_abs={max_abs:.6g} | max_rel={max_rel:.6g} | "
            f"failed={failed}/{total}{note}"
        )


def warmup(backend: Backend, case: TestCase, iterations: int) -> None:
    local = case.to(backend.device, backend.dtype)
    with torch.inference_mode():
        for _ in range(iterations):
            backend.model(local.x, local.valid_mask)
    backend.sync()


def time_backend(backend: Backend, case: TestCase, iterations: int) -> List[float]:
    """Time `iterations` calls, synchronizing on both sides of each one."""
    local = case.to(backend.device, backend.dtype)
    samples: List[float] = []
    with torch.inference_mode():
        for _ in range(iterations):
            backend.sync()
            start = time.perf_counter_ns()
            backend.model(local.x, local.valid_mask)
            backend.sync()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def run_benchmark(
    backends: List[Backend],
    case: TestCase,
    warmup_iterations: int,
    repeats: int,
    rounds: int,
) -> None:
    print("\n=== Performance benchmark ===")
    print("each call is bracketed by a device synchronize, so async dispatch is timed")
    print(f"warmup={warmup_iterations}, repeats={repeats}, rounds={rounds}")

    for backend in backends:
        warmup(backend, case, warmup_iterations)

    for round_index in range(rounds):
        # Reverse the order on odd rounds so no backend always runs on a cold
        # or a heat-soaked chip.
        order = backends if round_index % 2 == 0 else list(reversed(backends))
        for backend in order:
            backend.samples_ms.extend(time_backend(backend, case, repeats))


def report(backends: List[Backend], config: TransformerConfig, reference: Backend) -> None:
    tokens_per_call = config.batch_size * config.seq_len
    base_median = TimingResult(reference.samples_ms).median_ms

    header = (
        f"{'backend':<6} {'median':>10} {'mean':>10} {'p90':>10} "
        f"{'min':>10} {'tokens/s':>13} {'vs cpu':>8}"
    )
    print("\n=== Results ===")
    print(header)
    print("-" * len(header))

    for backend in backends:
        result = TimingResult(backend.samples_ms)
        throughput = tokens_per_call * 1000.0 / result.median_ms
        speedup = base_median / result.median_ms
        print(
            f"{backend.name:<6} {result.median_ms:>10.4f} {result.mean_ms:>10.4f} "
            f"{result.p90_ms:>10.4f} {result.min_ms:>10.4f} "
            f"{throughput:>13.0f} {speedup:>7.3f}x"
        )
    print("\n(times in ms; 'vs cpu' is median-latency speedup over the CPU baseline)")


def collect(backends: List[Backend], config: TransformerConfig, dtype: torch.dtype) -> Dict:
    return {
        "config": {
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "d_model": config.d_model,
            "num_heads": config.num_heads,
            "ffn_dim": config.ffn_dim,
            "num_layers": config.num_layers,
            "causal": config.causal,
        },
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "backends": {
            backend.name: {
                "label": backend.label,
                "accuracy": backend.accuracy,
                "timing_ms": {
                    "median": TimingResult(backend.samples_ms).median_ms,
                    "mean": TimingResult(backend.samples_ms).mean_ms,
                    "p90": TimingResult(backend.samples_ms).p90_ms,
                    "min": TimingResult(backend.samples_ms).min_ms,
                    "samples": len(backend.samples_ms),
                },
            }
            for backend in backends
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the benchmark across cpu, mps, and the MLX implementation"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--backends",
        default=",".join(ALL_BACKENDS),
        help=f"comma-separated subset of {','.join(ALL_BACKENDS)}",
    )
    parser.add_argument(
        "--mlx-torch-device",
        default="cpu",
        help="where the MLX backend's torch input/output tensors live",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)

    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--json", help="write the full results to this path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    names = [name.strip() for name in args.backends.split(",") if name.strip()]
    unknown = [name for name in names if name not in ALL_BACKENDS]
    if unknown:
        raise SystemExit(f"unknown backend(s): {unknown}; choose from {list(ALL_BACKENDS)}")
    if "cpu" not in names:
        raise SystemExit("the cpu backend is the accuracy and speedup reference; keep it")
    if "mps" in names and not torch.backends.mps.is_available():
        print("[warning] MPS is unavailable on this machine; skipping the mps backend")
        names.remove("mps")

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()
    torch.set_float32_matmul_precision(args.matmul_precision)

    backends = build_backends(names, config, dtype, args.seed, args.mlx_torch_device)
    reference = next(backend for backend in backends if backend.name == "cpu")

    print("=== Configuration ===")
    print(config)
    print(f"dtype={dtype}, torch={torch.__version__}")
    for backend in backends:
        print(f"  {backend.name:<4} -> {backend.label}")

    if not args.skip_accuracy:
        cases = make_accuracy_cases(
            config,
            seed=args.seed,
            trials=args.accuracy_trials,
            padding_ratio=args.padding_ratio,
            input_scale=args.input_scale,
        )
        check_accuracy(backends, cases, reference, rtol=args.rtol, atol=args.atol)

    timing_case = make_timing_case(
        config,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
    )
    run_benchmark(backends, timing_case, args.warmup, args.repeats, args.rounds)
    report(backends, config, reference)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(collect(backends, config, dtype), handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

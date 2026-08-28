#!/usr/bin/env python3
"""
The test shapes of Appendix 3.7, built with the shared case generator.

Each row of the appendix table becomes one `TransformerConfig`:

    Batch Size -> batch_size    Seq Len -> seq_len    Causal   -> causal
    QKV Dim    -> d_model       Layers  -> num_layers FFN Dim  -> ffn_dim
    Heads      -> num_heads

Run the script with no argument to build every shape and print its cost:

    .venv/bin/python3 appendix_cases.py

Select a subset with `--cases`, and add `--run` for the real benchmark:

    .venv/bin/python3 appendix_cases.py --cases 1,7-9 --run

Shape 14 (B=32, D=1024, H=16, S=100000, L=2) is disabled. Its input alone is
12.2 GiB in float32, and `BaselineSelfAttention` materializes a B x H x S x S
score matrix, which is 18.6 TiB at that shape.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from bench_cases import make_accuracy_cases, make_case, make_timing_case
from test_backends import (
    ALL_BACKENDS,
    build_backends,
    check_accuracy,
    collect,
    report,
    run_benchmark,
)
from torch_transformer_benchmark import TransformerConfig, resolve_dtype

GIB = 1024 ** 3


@dataclass(frozen=True)
class Shape:
    """One row of the appendix table."""

    case_id: int
    batch_size: int
    d_model: int
    num_heads: int
    seq_len: int
    num_layers: int
    causal: bool
    ffn_dim: int
    enabled: bool = True
    note: str = ""

    def config(self) -> TransformerConfig:
        return TransformerConfig(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
            causal=self.causal,
        )

    def input_bytes(self, itemsize: int = 4) -> int:
        return self.batch_size * self.seq_len * self.d_model * itemsize

    def attention_bytes(self, itemsize: int = 4) -> int:
        """One B x H x S x S score matrix, as the baseline materializes it."""
        return (
            self.batch_size
            * self.num_heads
            * self.seq_len
            * self.seq_len
            * itemsize
        )


APPENDIX_SHAPES: List[Shape] = [
    Shape(1, 64, 128, 4, 128, 4, True, 128),
    Shape(2, 1, 128, 4, 128, 4, True, 128),
    Shape(3, 4, 128, 4, 128, 4, True, 128),
    Shape(4, 16, 128, 4, 128, 4, True, 128),
    Shape(5, 128, 128, 4, 128, 4, True, 128),
    Shape(6, 10000, 128, 4, 128, 4, True, 128),
    Shape(7, 64, 32, 4, 128, 4, True, 32),
    Shape(8, 64, 1024, 4, 128, 4, True, 1024),
    Shape(9, 64, 128, 1, 128, 4, True, 128),
    Shape(10, 64, 128, 2, 128, 4, True, 128),
    Shape(11, 64, 128, 16, 128, 4, True, 128),
    Shape(12, 64, 128, 4, 32, 4, True, 128),
    Shape(13, 64, 128, 4, 1024, 4, True, 128),
    Shape(
        14,
        32,
        1024,
        16,
        100000,
        2,
        True,
        1024,
        enabled=False,
        note="12.2 GiB input, 18.6 TiB score matrix; too large for this machine",
    ),
]

SHAPES_BY_ID: Dict[int, Shape] = {shape.case_id: shape for shape in APPENDIX_SHAPES}


def parse_selection(text: str) -> List[int]:
    """Turn "1,3,5-8" into [1, 3, 5, 6, 7, 8]. "all" gives every enabled id."""
    if text.strip().lower() == "all":
        return [shape.case_id for shape in APPENDIX_SHAPES if shape.enabled]

    selected: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low_text, _, high_text = part.partition("-")
            low, high = int(low_text), int(high_text)
            if low > high:
                raise SystemExit(f"bad range: {part}")
            ids = range(low, high + 1)
        else:
            ids = [int(part)]
        for case_id in ids:
            if case_id not in SHAPES_BY_ID:
                raise SystemExit(
                    f"unknown case {case_id}; choose from 1-{len(APPENDIX_SHAPES)}"
                )
            if case_id not in selected:
                selected.append(case_id)
    if not selected:
        raise SystemExit("--cases selected nothing")
    return selected


def print_table() -> None:
    header = (
        f"{'#':>3}  {'Batch':>6} {'QKV':>5} {'Heads':>5} {'Seq':>7} "
        f"{'Lyr':>3} {'Causal':>6} {'FFN':>5}  {'Input':>10} {'Attn':>12}"
    )
    print(header)
    print("-" * len(header))
    for shape in APPENDIX_SHAPES:
        mark = "" if shape.enabled else "  [disabled]"
        print(
            f"{shape.case_id:>3}  {shape.batch_size:>6} {shape.d_model:>5} "
            f"{shape.num_heads:>5} {shape.seq_len:>7} {shape.num_layers:>3} "
            f"{str(shape.causal).upper():>6} {shape.ffn_dim:>5}  "
            f"{shape.input_bytes() / GIB:>8.3f} GiB "
            f"{shape.attention_bytes() / GIB:>10.1f} GiB{mark}"
        )
        if shape.note:
            print(f"{'':>5}{shape.note}")


def free_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def describe_case(shape: Shape, args: argparse.Namespace) -> None:
    """Build the accuracy and timing inputs and print what they hold."""
    config = shape.config()
    config.validate()

    case = make_case(config, args.seed, args.padding_ratio, args.input_scale)
    timing = make_timing_case(config, args.seed, args.padding_ratio, args.input_scale)
    print(
        f"    x{tuple(case.x.shape)} {case.x.dtype} "
        f"mask{tuple(case.valid_mask.shape)} "
        f"valid={int(case.valid_mask.sum())}/{case.valid_mask.numel()} "
        f"mean={case.x.mean().item():+.5f} std={case.x.std().item():.5f}"
    )
    print(f"    accuracy seed={case.seed}, timing seed={timing.seed}")
    del case, timing
    free_memory()


def run_case(shape: Shape, args: argparse.Namespace, names: List[str]) -> Optional[Dict]:
    """Run the cross-backend benchmark for one shape."""
    config = shape.config()
    config.validate()
    dtype = resolve_dtype(args.dtype)

    backends = build_backends(names, config, dtype, args.seed, args.mlx_torch_device)
    reference = next(backend for backend in backends if backend.name == "cpu")
    for backend in backends:
        print(f"    {backend.name:<4} -> {backend.label}")

    if not args.skip_accuracy:
        cases = make_accuracy_cases(
            config,
            seed=args.seed,
            trials=args.accuracy_trials,
            padding_ratio=args.padding_ratio,
            input_scale=args.input_scale,
        )
        check_accuracy(backends, cases, reference, rtol=args.rtol, atol=args.atol)
        del cases

    timing_case = make_timing_case(
        config,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
    )
    run_benchmark(backends, timing_case, args.warmup, args.repeats, args.rounds)
    report(backends, config, reference)

    result = collect(backends, config, dtype)
    result["case_id"] = shape.case_id
    del backends, reference, timing_case
    free_memory()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build (and optionally benchmark) the Appendix 3.7 shapes"
    )
    parser.add_argument(
        "--cases",
        default="all",
        help='which shapes to use, e.g. "1,3,5-8" or "all" (the default)',
    )
    parser.add_argument("--list", action="store_true", help="print the table and stop")
    parser.add_argument(
        "--run",
        action="store_true",
        help="run the cross-backend benchmark for each selected shape",
    )
    parser.add_argument(
        "--budget-gb",
        type=float,
        default=8.0,
        help="skip a shape whose float32 input is larger than this",
    )

    parser.add_argument(
        "--backends",
        default=",".join(ALL_BACKENDS),
        help=f"comma-separated subset of {','.join(ALL_BACKENDS)}",
    )
    parser.add_argument("--mlx-torch-device", default="cpu")
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

    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument("--json", help="write the results of every shape to this path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print_table()
        return 0

    names = [name.strip() for name in args.backends.split(",") if name.strip()]
    unknown = [name for name in names if name not in ALL_BACKENDS]
    if unknown:
        raise SystemExit(
            f"unknown backend(s): {unknown}; choose from {list(ALL_BACKENDS)}"
        )
    if args.run:
        if "cpu" not in names:
            raise SystemExit(
                "the cpu backend is the accuracy and speedup reference; keep it"
            )
        if "mps" in names and not torch.backends.mps.is_available():
            print("[warning] MPS is unavailable on this machine; dropping the mps backend")
            names.remove("mps")
    torch.set_float32_matmul_precision(args.matmul_precision)

    selected = parse_selection(args.cases)
    results: List[Dict] = []
    skipped: List[int] = []

    for case_id in selected:
        shape = SHAPES_BY_ID[case_id]
        input_gib = shape.input_bytes() / GIB
        print(
            f"\n=== Case {shape.case_id} ===  B={shape.batch_size} "
            f"D={shape.d_model} H={shape.num_heads} S={shape.seq_len} "
            f"L={shape.num_layers} FFN={shape.ffn_dim} "
            f"causal={str(shape.causal).upper()}"
        )
        print(
            f"    input={input_gib:.3f} GiB, "
            f"score matrix={shape.attention_bytes() / GIB:.3f} GiB per layer"
        )

        if not shape.enabled:
            print(f"    skipped: {shape.note}")
            skipped.append(case_id)
            continue
        if input_gib > args.budget_gb:
            print(f"    skipped: over the {args.budget_gb} GiB budget")
            skipped.append(case_id)
            continue

        if args.run:
            results.append(run_case(shape, args, names))
        else:
            describe_case(shape, args)

    done = [case_id for case_id in selected if case_id not in skipped]
    print(f"\ncases done: {done}")
    if skipped:
        print(f"cases skipped: {skipped}")

    if args.json and results:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

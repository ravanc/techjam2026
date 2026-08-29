#!/usr/bin/env python3
"""
The scoreboard: CPU, MPS and MLX timing for every appendix shape.

Every number here is a stopwatch reading or a ratio of two stopwatch
readings, with one exception that is labelled PROVISIONAL. It reports four
things for each shape, and it never drops a column:

    CPU              torch baseline on the CPU. The accuracy and speed reference.
    MPS              torch baseline on the GPU. The gain of the device alone.
    MLX              UserOptimizedTransformer. The gain of the kernels.
    achieved TFLOP/s Model FLOPs divided by measured time.

Run every shape and write the results:

    .venv/bin/python3 scoreboard.py --label "what changed"

Run a subset, which is much faster while you tune:

    .venv/bin/python3 scoreboard.py --cases 1,7,11,13 --label "trying X"

Print the recorded runs:

    .venv/bin/python3 scoreboard.py --show-history

The repeat count falls as the shape grows, so one sweep stays inside about
ten minutes. The CPU baseline is most of that time: at shape 6 one CPU call
takes 13.6 s, because `BaselineSelfAttention` materializes a 2.4 GiB score
matrix. `--repeats` overrides the automatic count.

Outputs:
    profiling/scoreboard.json   the newest sweep, overwritten each run
    profiling/history.jsonl     append-only, one line per sweep
    references/scoreboard.md    the tables, ready to read

MFU is reported once, in its own section, and it is marked PROVISIONAL:
its denominator was never verified. See `flops.py`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx
import torch

from appendix_cases import SHAPES_BY_ID, Shape, parse_selection
from bench_cases import make_case, make_timing_case
from flops import (
    MEASURED_TFLOPS,
    MFU_DISCLAIMER,
    PROVISIONAL_PEAK_TFLOPS,
    achieved_tflops,
    model_flops,
    provisional_mfu,
)
from test_backends import build_backends, time_backend, warmup
from torch_transformer_benchmark import resolve_dtype

BACKENDS = ("cpu", "mps", "mlx")

# Rough CPU baseline rate, used only to choose a repeat count. The baseline
# materializes a B x H x S x S score matrix, so it runs well under the 1.42
# TFLOP/s matmul rate measured on this CPU.
CPU_FLOPS_ESTIMATE = 200e9

# Target seconds of timed work per backend per round.
TARGET_SECONDS = 2.0


@dataclass
class CaseResult:
    """One shape, every backend. Times only."""

    case_id: int
    config: Dict
    flops: int
    tokens: int
    median_ms: Dict[str, float] = field(default_factory=dict)
    samples: Dict[str, List[float]] = field(default_factory=dict)
    accuracy: Dict[str, Dict] = field(default_factory=dict)
    failed: Dict[str, str] = field(default_factory=dict)
    plan: Optional[str] = None
    repeats: int = 0
    rounds: int = 0

    def speedup(self, name: str) -> Optional[float]:
        base = self.median_ms.get("cpu")
        mine = self.median_ms.get(name)
        return None if base is None or mine is None else base / mine

    def tflops(self, name: str) -> Optional[float]:
        mine = self.median_ms.get(name)
        return None if mine is None else achieved_tflops(self.flops, mine)

    def tokens_per_second(self, name: str) -> Optional[float]:
        mine = self.median_ms.get(name)
        return None if mine is None else self.tokens * 1000.0 / mine

    def mfu(self, name: str) -> Optional[float]:
        """PROVISIONAL. See `flops.PROVISIONAL_PEAK_TFLOPS`."""
        mine = self.median_ms.get(name)
        return None if mine is None else provisional_mfu(self.flops, mine)


def number(value: Optional[float], fmt: str = "{:.3f}") -> str:
    """Format a value, or an em dash when it is missing."""
    return fmt.format(value) if value is not None else "—"


def choose_repeats(flops: int, override: Optional[int]) -> int:
    if override:
        return override
    seconds = flops / CPU_FLOPS_ESTIMATE
    return max(3, min(50, int(TARGET_SECONDS / max(seconds, 1e-6))))


def free_memory() -> None:
    gc.collect()
    mx.clear_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def accuracy_of(reference: torch.Tensor, other: torch.Tensor,
                rtol: float, atol: float) -> Dict:
    difference = (other - reference).abs()
    within = (difference <= atol) | (difference <= rtol * reference.abs())
    return {
        "passed": bool(within.all()),
        "max_abs_error": float(difference.max()),
        "failed_elements": int((~within).sum()),
        "total_elements": int(within.numel()),
    }


def run_case(shape: Shape, args: argparse.Namespace, names: List[str]) -> CaseResult:
    config = shape.config()
    config.validate()
    dtype = resolve_dtype(args.dtype)
    flops = model_flops(config)

    result = CaseResult(
        case_id=shape.case_id,
        config=config.__dict__.copy(),
        flops=flops,
        tokens=config.batch_size * config.seq_len,
        repeats=choose_repeats(flops, args.repeats),
        rounds=args.rounds,
    )

    backends = build_backends(names, config, dtype, args.seed, args.mlx_torch_device)
    for backend in backends:
        if backend.name == "mlx":
            # Force the weight build so the plan exists before it is reported.
            backend.model._build_mlx_weights()
            result.plan = backend.model.plan.describe()

    print(
        f"    flops={flops / 1e9:.2f} G  tokens={result.tokens:,}  "
        f"repeats={result.repeats} rounds={result.rounds}"
    )
    if result.plan:
        print(f"    plan: {result.plan}")

    # --- accuracy, one trial, against the CPU baseline ---
    if not args.skip_accuracy:
        case = make_case(config, args.seed, args.padding_ratio)
        reference = None
        for backend in backends:
            try:
                output = backend.run(case)
            except Exception as error:  # noqa: BLE001
                result.failed[backend.name] = f"accuracy: {type(error).__name__}"
                print(f"    {backend.name:<4} accuracy FAILED {type(error).__name__}")
                continue
            if backend.name == "cpu":
                reference = output
            elif reference is not None:
                result.accuracy[backend.name] = accuracy_of(
                    reference, output, args.rtol, args.atol
                )
            del output
        del case, reference
        free_memory()

    # --- timing, one backend at a time so a failure is contained ---
    timing_case = make_timing_case(config, args.seed, args.padding_ratio)
    for round_index in range(args.rounds):
        # Reverse on odd rounds so no backend always runs on a cold chip.
        order = backends if round_index % 2 == 0 else list(reversed(backends))
        for backend in order:
            if backend.name in result.failed:
                continue
            try:
                if round_index == 0:
                    warmup(backend, timing_case, args.warmup)
                samples = time_backend(backend, timing_case, result.repeats)
            except Exception as error:  # noqa: BLE001
                result.failed[backend.name] = f"{type(error).__name__}: {error}"[:120]
                print(f"    {backend.name:<4} timing FAILED {type(error).__name__}")
                continue
            result.samples.setdefault(backend.name, []).extend(samples)

    for name, samples in result.samples.items():
        result.median_ms[name] = statistics.median(samples)

    del backends, timing_case
    free_memory()
    return result


def print_case(result: CaseResult) -> None:
    print(
        f"    {'backend':<5} {'median ms':>11} {'vs CPU':>9} "
        f"{'TFLOP/s':>9} {'token/s':>12}  accuracy"
    )
    for name in BACKENDS:
        if name in result.failed:
            print(f"    {name:<5} {'--':>11}  {result.failed[name]}")
            continue
        if name not in result.median_ms:
            continue
        accuracy = result.accuracy.get(name)
        if name == "cpu":
            note = "reference"
        elif accuracy is None:
            note = "not checked"
        else:
            note = (
                f"{'PASS' if accuracy['passed'] else 'FAIL'} "
                f"max_abs={accuracy['max_abs_error']:.2e}"
            )
        print(
            f"    {name:<5} {result.median_ms[name]:>11.4f} "
            f"{number(result.speedup(name), '{:.3f}') + 'x':>9} "
            f"{number(result.tflops(name)):>9} "
            f"{number(result.tokens_per_second(name), '{:.0f}'):>12}  {note}"
        )


def git_state() -> Dict:
    """Which build produced a reading. A reading without this is hard to place."""

    def run(*command: str) -> str:
        try:
            return subprocess.run(
                command, capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {
        "commit": run("git", "rev-parse", "--short", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def summarize(results: List[CaseResult]) -> Dict:
    scored = [r for r in results if r.speedup("mlx") is not None]
    if not scored:
        return {"cases_scored": 0}
    speedups = [r.speedup("mlx") for r in scored]
    rates = [r.tflops("mlx") for r in scored]
    return {
        "cases_scored": len(scored),
        "median_speedup_mlx": statistics.median(speedups),
        "min_speedup_mlx": min(speedups),
        "max_speedup_mlx": max(speedups),
        "median_tflops_mlx": statistics.median(rates),
        "max_tflops_mlx": max(rates),
    }


def append_history(results: List[CaseResult], payload: Dict,
                   path: str, label: str) -> Dict:
    """
    Append one line to the run history.

    `profiling/scoreboard.json` holds the newest run only, and every run
    overwrites it. This file never loses a reading.
    """
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "git": git_state(),
        "dtype": payload["dtype"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "summary": summarize(results),
        "cases": {
            str(r.case_id): {
                "median_ms": r.median_ms,
                "speedup_mlx": r.speedup("mlx"),
                "tflops_mlx": r.tflops("mlx"),
                "plan": r.plan,
                "flops": r.flops,
            }
            for r in results
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def load_history(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def commit_label(entry: Dict) -> str:
    commit = entry.get("git", {}).get("commit", "") or "?"
    return commit + ("*" if entry.get("git", {}).get("dirty") else "")


def print_history(entries: List[Dict]) -> None:
    if not entries:
        print("no history yet; run the sweep once")
        return
    header = (
        f"{'when':<20} {'commit':<9} {'label':<26} {'cases':>5} "
        f"{'med speedup':>12} {'med TFLOP/s':>12}"
    )
    print(header)
    print("-" * len(header))
    for entry in entries:
        summary = entry.get("summary", {})
        print(
            f"{entry['timestamp']:<20} {commit_label(entry):<9} "
            f"{(entry.get('label') or '')[:26]:<26} "
            f"{summary.get('cases_scored', 0):>5} "
            f"{number(summary.get('median_speedup_mlx'), '{:.2f}') + 'x':>12} "
            f"{number(summary.get('median_tflops_mlx')):>12}"
        )

    print("\nPer-case MLX median ms across the last runs:")
    case_ids = sorted({int(c) for e in entries for c in e.get("cases", {})})
    recent = entries[-6:]
    print(f"{'case':>5} " + " ".join(f"{e['timestamp'][5:16]:>12}" for e in recent))
    for case_id in case_ids:
        cells = []
        for entry in recent:
            case = entry.get("cases", {}).get(str(case_id))
            value = (case or {}).get("median_ms", {}).get("mlx")
            cells.append(number(value, "{:.2f}"))
        print(f"{case_id:>5} " + " ".join(f"{c:>12}" for c in cells))


def write_markdown(results: List[CaseResult], path: str, dtype_name: str,
                   elapsed: float, history: Optional[List[Dict]] = None) -> None:
    lines: List[str] = []
    add = lines.append

    add("# Scoreboard")
    add("")
    add("Timing for every Appendix 3.7 shape. Regenerate with:")
    add("")
    add('    .venv/bin/python3 scoreboard.py --label "what changed"')
    add("")
    add("Every number below is a stopwatch reading or a ratio of two stopwatch")
    add("readings. `TFLOP/s` divides the model FLOP count by measured time; the")
    add("FLOP count comes from the model definition, not from a specification.")
    add("")
    add(f"- dtype: `{dtype_name}`")
    add("- CPU is the reference for both accuracy and speedup.")
    add("- Each call is bracketed by a device synchronize. Rounds alternate the")
    add("  backend order, so no backend always runs on a cold chip.")
    add(f"- sweep took {elapsed / 60:.1f} minutes")
    add("")
    add("MFU appears once, in its own section, and it is provisional. See")
    add("[../flops.py](../flops.py).")
    add("")

    add("## Speedup against the CPU baseline")
    add("")
    add("| # | Shape | CPU ms | MPS ms | MLX ms | MPS vs CPU | **MLX vs CPU** | MLX vs MPS |")
    add("|---:|---|---:|---:|---:|---:|---:|---:|")
    for result in results:
        config = result.config
        tag = (f"B{config['batch_size']} D{config['d_model']} "
               f"H{config['num_heads']} S{config['seq_len']}")

        def cell(name: str) -> str:
            if name in result.failed:
                return "OOM"
            return number(result.median_ms.get(name))

        mps_up = result.speedup("mps")
        mlx_up = result.speedup("mlx")
        mlx_mps = None
        if "mps" in result.median_ms and "mlx" in result.median_ms:
            mlx_mps = result.median_ms["mps"] / result.median_ms["mlx"]
        add(
            f"| {result.case_id} | {tag} | {cell('cpu')} | {cell('mps')} | "
            f"{cell('mlx')} | {number(mps_up, '{:.2f}') + 'x'} | "
            f"**{number(mlx_up, '{:.2f}') + 'x'}** | "
            f"{number(mlx_mps, '{:.2f}') + 'x'} |"
        )
    add("")

    add("## Achieved arithmetic rate")
    add("")
    add("Model FLOPs divided by measured time. Compare against the matmul rates")
    add("measured on this machine: MLX float32 reaches 4.06 TFLOP/s on a square")
    add("matmul, torch MPS 4.16, torch CPU 1.42.")
    add("")
    add("| # | Shape | GFLOP | CPU TFLOP/s | MPS TFLOP/s | **MLX TFLOP/s** | MLX token/s | Plan chosen |")
    add("|---:|---|---:|---:|---:|---:|---:|---|")
    for result in results:
        config = result.config
        tag = (f"B{config['batch_size']} D{config['d_model']} "
               f"H{config['num_heads']} S{config['seq_len']}")
        add(
            f"| {result.case_id} | {tag} | {result.flops / 1e9:.2f} | "
            f"{number(result.tflops('cpu'))} | {number(result.tflops('mps'))} | "
            f"**{number(result.tflops('mlx'))}** | "
            f"{number(result.tokens_per_second('mlx'), '{:,.0f}')} | "
            f"{result.plan or '—'} |"
        )
    add("")

    add("## MFU — PROVISIONAL, do not quote without this note")
    add("")
    add(f"> **{MFU_DISCLAIMER}**")
    add("")
    add("The numerator and the time are objective. The denominator is not:")
    add(f"{PROVISIONAL_PEAK_TFLOPS} TFLOP/s came from "
        "`14 cores x 128 ALUs x 2 x 1.398 GHz`,")
    add("and the clock and ALU count were never checked against a source. Every")
    add("number in this table scales linearly with it. Fix the constant in")
    add("`flops.py` before this table means anything.")
    add("")
    add("| # | Shape | GFLOP | FLOP share | MLX MFU | MPS MFU |")
    add("|---:|---|---:|---:|---:|---:|")
    total_flops = sum(r.flops for r in results if r.speedup("mlx") is not None)
    for result in results:
        config = result.config
        tag = (f"B{config['batch_size']} D{config['d_model']} "
               f"H{config['num_heads']} S{config['seq_len']}")
        share = result.flops / total_flops if total_flops else None
        add(
            f"| {result.case_id} | {tag} | {result.flops / 1e9:.2f} | "
            f"{number(share and share * 100, '{:.1f}') + '%'} | "
            f"{number(result.mfu('mlx') and result.mfu('mlx') * 100, '{:.1f}') + '%'} | "
            f"{number(result.mfu('mps') and result.mfu('mps') * 100, '{:.1f}') + '%'} |"
        )
    add("")
    mfus = [r.mfu("mlx") for r in results if r.mfu("mlx") is not None]
    if mfus and total_flops:
        weighted = sum(
            r.mfu("mlx") * r.flops for r in results if r.mfu("mlx") is not None
        ) / total_flops
        add(f"- unweighted mean MLX MFU: **{statistics.mean(mfus) * 100:.2f}%**")
        add(f"- FLOP-weighted mean MLX MFU: **{weighted * 100:.2f}%**")
        add("")
        add("Shape 6 alone is two thirds of the FLOP weight, so a FLOP-weighted")
        add("score is mostly a score on shape 6.")
        add("")

    add("## Accuracy")
    add("")
    add("Against the CPU baseline, at `atol=0.002` and `rtol=0.02`.")
    add("")
    add("| # | MPS | MLX |")
    add("|---:|---|---|")
    for result in results:
        def verdict(name: str) -> str:
            accuracy = result.accuracy.get(name)
            if accuracy is None:
                return "—"
            state = "PASS" if accuracy["passed"] else "FAIL"
            return (f"{state} `max_abs={accuracy['max_abs_error']:.2e}` "
                    f"({accuracy['failed_elements']}/"
                    f"{accuracy['total_elements']} failed)")
        add(f"| {result.case_id} | {verdict('mps')} | {verdict('mlx')} |")
    add("")

    summary = summarize(results)
    if summary.get("cases_scored"):
        add("## Summary")
        add("")
        add("| Metric | Value |")
        add("|---|---:|")
        add(f"| Shapes scored | {summary['cases_scored']} |")
        add(f"| Median MLX speedup over CPU | **{summary['median_speedup_mlx']:.2f}x** |")
        add(f"| Range of MLX speedup | {summary['min_speedup_mlx']:.2f}x "
            f"to {summary['max_speedup_mlx']:.2f}x |")
        add(f"| Median MLX rate | {summary['median_tflops_mlx']:.3f} TFLOP/s |")
        add(f"| Best MLX rate | {summary['max_tflops_mlx']:.3f} TFLOP/s |")
        add("")

    if history:
        add("## History")
        add("")
        add("Every recorded sweep. The data is in "
            "[../profiling/history.jsonl](../profiling/history.jsonl).")
        add("Print it with `.venv/bin/python3 scoreboard.py --show-history`.")
        add("")
        add("| When | Commit | Label | Shapes | Median speedup | Median TFLOP/s |")
        add("|---|---|---|---:|---:|---:|")
        for entry in history:
            item = entry.get("summary", {})
            add(
                f"| {entry['timestamp']} | `{commit_label(entry)}` | "
                f"{entry.get('label') or '—'} | {item.get('cases_scored', 0)} | "
                f"{number(item.get('median_speedup_mlx'), '{:.2f}') + 'x'} | "
                f"{number(item.get('median_tflops_mlx'))} |"
            )
        add("")
        add("A `*` on the commit means the working tree had uncommitted changes,")
        add("so that reading cannot be reproduced exactly.")
        add("")

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU/MPS/MLX timing for every appendix shape"
    )
    parser.add_argument("--cases", default="all", help='e.g. "1,3,5-8" or "all"')
    parser.add_argument(
        "--label", default="", help="short note recorded with this run in the history"
    )
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--mlx-torch-device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--budget-gb", type=float, default=8.0)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--repeats", type=int, default=None,
        help="override the automatic per-shape repeat count",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--json", default="profiling/scoreboard.json")
    parser.add_argument("--markdown", default="references/scoreboard.md")
    parser.add_argument("--history", default="profiling/history.jsonl")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--show-history", action="store_true", help="print the recorded runs and stop"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.show_history:
        print_history(load_history(args.history))
        return 0

    names = [name.strip() for name in args.backends.split(",") if name.strip()]
    if "cpu" not in names:
        raise SystemExit("the cpu backend is the reference; keep it")
    if "mps" in names and not torch.backends.mps.is_available():
        print("[warning] MPS is unavailable; dropping it")
        names.remove("mps")

    print("measured matmul rates on this machine (TFLOP/s): " + ", ".join(
        f"{name}={rate}" for name, rate in MEASURED_TFLOPS.items()
    ))

    selected = parse_selection(args.cases)
    results: List[CaseResult] = []
    started = time.perf_counter()

    for case_id in selected:
        shape = SHAPES_BY_ID[case_id]
        input_gib = shape.input_bytes() / 1024**3
        print(
            f"\n=== Case {shape.case_id} ===  B={shape.batch_size} "
            f"D={shape.d_model} H={shape.num_heads} S={shape.seq_len} "
            f"L={shape.num_layers} FFN={shape.ffn_dim}"
        )
        if not shape.enabled:
            print(f"    skipped: {shape.note}")
            continue
        if input_gib > args.budget_gb:
            print(f"    skipped: input {input_gib:.2f} GiB is over the "
                  f"{args.budget_gb} GiB budget")
            continue
        result = run_case(shape, args, names)
        print_case(result)
        results.append(result)

    elapsed = time.perf_counter() - started
    payload = {
        "dtype": args.dtype,
        "measured_tflops": MEASURED_TFLOPS,
        "torch_version": torch.__version__,
        "elapsed_seconds": elapsed,
        "summary": summarize(results),
        "cases": [
            {
                "case_id": r.case_id,
                "config": r.config,
                "flops": r.flops,
                "tokens": r.tokens,
                "plan": r.plan,
                "repeats": r.repeats,
                "rounds": r.rounds,
                "median_ms": r.median_ms,
                "speedup_vs_cpu": {name: r.speedup(name) for name in r.median_ms},
                "tflops": {name: r.tflops(name) for name in r.median_ms},
                "accuracy": r.accuracy,
                "failed": r.failed,
            }
            for r in results
        ],
    }
    with open(args.json, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.json}")

    if not args.no_history:
        entry = append_history(results, payload, args.history, args.label)
        print(f"appended reading {len(load_history(args.history))} to {args.history} "
              f"(label={entry['label'] or 'none'}, commit={commit_label(entry)})")

    write_markdown(
        results, args.markdown, args.dtype, elapsed,
        history=None if args.no_history else load_history(args.history),
    )
    print(f"wrote {args.markdown}")

    summary = payload["summary"]
    if summary.get("cases_scored"):
        print(f"\n=== Summary over {summary['cases_scored']} shapes ===")
        print(f"  median MLX speedup over CPU : {summary['median_speedup_mlx']:.2f}x")
        print(f"  range                       : {summary['min_speedup_mlx']:.2f}x "
              f"to {summary['max_speedup_mlx']:.2f}x")
        print(f"  median MLX rate             : {summary['median_tflops_mlx']:.3f} TFLOP/s")
    print(f"  sweep took {elapsed / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

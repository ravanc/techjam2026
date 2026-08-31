#!/usr/bin/env python3
"""Run the given harness on every appendix shape, and collect the data.

`torch_transformer_benchmark.py` takes one shape for each call. This script
calls it once for each shape, as a subprocess, with the default flags. It does
not import the harness and it does not change it. The subprocess output is the
record.

    .venv/bin/python3 profiling/tools/harness_sweep.py
    .venv/bin/python3 profiling/tools/harness_sweep.py --cases 1,7,8

For each run it writes:

  profiling/results/harness/<stamp>/case_NN.log   the full stdout of the call
  profiling/results/harness/<stamp>/results.json  the parsed numbers
  profiling/results/harness/<stamp>/results.md    the same numbers as a table
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from appendix_cases import APPENDIX_SHAPES, SHAPES_BY_ID, parse_selection  # noqa: E402

HARNESS = ROOT / "torch_transformer_benchmark.py"
PYTHON = ROOT / ".venv" / "bin" / "python3"
OUT_ROOT = ROOT / "profiling" / "results" / "harness"

TIMING_RE = re.compile(
    r"^(baseline|optimized)\s*:\s*"
    r"median=(?P<median>[\d.]+) ms \| "
    r"mean=(?P<mean>[\d.]+) ms \| "
    r"p90=(?P<p90>[\d.]+) ms \| "
    r"min=(?P<min>[\d.]+) ms \| "
    r"throughput=(?P<tps>[\d.]+) token/s"
)
SPEEDUP_RE = re.compile(r"^speedup\s*:\s*(?P<speedup>[\d.]+)x")
ACCURACY_RE = re.compile(
    r"^summary: (?P<status>PASS|FAIL) \| "
    r"max_abs=(?P<max_abs>\S+) \| max_rel=(?P<max_rel>\S+) \| "
    r"failed=(?P<failed>\d+)/(?P<total>\d+)"
)


def git_state() -> Dict[str, object]:
    def run(cmd: List[str]) -> str:
        return subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
    }


def build_command(shape, args: argparse.Namespace) -> List[str]:
    command = [
        str(PYTHON),
        str(HARNESS),
        "--batch-size", str(shape.batch_size),
        "--seq-len", str(shape.seq_len),
        "--d-model", str(shape.d_model),
        "--heads", str(shape.num_heads),
        "--ffn-dim", str(shape.ffn_dim),
        "--layers", str(shape.num_layers),
        "--device", args.device,
        "--dtype", args.dtype,
    ]
    if shape.causal:
        command.append("--causal")
    for name in ("warmup", "repeats", "accuracy_trials", "seed"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    if args.benchmark_rounds is not None:
        command += ["--benchmark-rounds", str(args.benchmark_rounds)]
    return command


def parse_output(text: str) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for line in text.splitlines():
        timing = TIMING_RE.match(line)
        if timing:
            parsed[timing.group(1)] = {
                "median_ms": float(timing.group("median")),
                "mean_ms": float(timing.group("mean")),
                "p90_ms": float(timing.group("p90")),
                "min_ms": float(timing.group("min")),
                "tokens_per_second": float(timing.group("tps")),
            }
            continue
        speedup = SPEEDUP_RE.match(line)
        if speedup:
            parsed["speedup"] = float(speedup.group("speedup"))
            continue
        accuracy = ACCURACY_RE.match(line)
        if accuracy:
            parsed["accuracy"] = {
                "passed": accuracy.group("status") == "PASS",
                "max_abs": float(accuracy.group("max_abs")),
                "max_rel": float(accuracy.group("max_rel")),
                "failed_elements": int(accuracy.group("failed")),
                "total_elements": int(accuracy.group("total")),
            }
    return parsed


def write_markdown(path: Path, record: Dict[str, object]) -> None:
    lines = [
        "# Harness sweep",
        "",
        f"- script: `torch_transformer_benchmark.py` (unchanged)",
        f"- date: {record['timestamp']}",
        f"- commit: {record['git']['commit']}"
        + ("*" if record["git"]["dirty"] else ""),
        f"- device: {record['device']}, dtype: {record['dtype']}",
        f"- torch: {record['torch_version']}",
        f"- elapsed: {record['elapsed_seconds']:.1f} s",
        "",
        "`baseline` is `BaselineTransformer`. `optimized` is "
        "`UserOptimizedTransformer`. Both times are the median of the harness "
        "rounds.",
        "",
        "| # | Shape (B, D, H, S, L) | Accuracy | baseline ms | optimized ms |"
        " speedup | optimized token/s |",
        "|---:|---|:---:|---:|---:|---:|---:|",
    ]
    for case in record["cases"]:
        shape = case["shape"]
        label = (
            f"{shape['batch_size']}, {shape['d_model']}, {shape['num_heads']}, "
            f"{shape['seq_len']}, {shape['num_layers']}"
        )
        if case.get("error"):
            lines.append(f"| {case['case_id']} | {label} | ERROR | — | — | — | — |")
            continue
        accuracy = case.get("accuracy") or {}
        status = "PASS" if accuracy.get("passed") else "FAIL"
        base = case["baseline"]["median_ms"]
        opt = case["optimized"]["median_ms"]
        lines.append(
            f"| {case['case_id']} | {label} | {status} | {base:.4f} | "
            f"{opt:.4f} | {case['speedup']:.3f}x | "
            f"{case['optimized']['tokens_per_second']:,.0f} |"
        )
    summary = record["summary"]
    lines += [
        "",
        "## Summary",
        "",
        f"- cases run: {summary['cases_run']}",
        f"- cases that pass accuracy: {summary['cases_passed']}",
        f"- median speedup: {summary['median_speedup']:.3f}x",
        f"- minimum speedup: {summary['min_speedup']:.3f}x "
        f"(shape {summary['min_speedup_case']})",
        f"- maximum speedup: {summary['max_speedup']:.3f}x "
        f"(shape {summary['max_speedup_case']})",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run torch_transformer_benchmark.py on every appendix shape"
    )
    parser.add_argument("--cases", default="all", help='e.g. "1,3,5-8" or "all"')
    parser.add_argument("--device", default="cpu", help="passed to the harness")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--benchmark-rounds", type=int, default=None)
    parser.add_argument("--accuracy-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands and stop"
    )
    args = parser.parse_args()

    case_ids = parse_selection(args.cases)
    shapes = [SHAPES_BY_ID[case_id] for case_id in case_ids]

    if args.dry_run:
        for shape in shapes:
            print(" ".join(build_command(shape, args)))
        return 0

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch  # after the argument check, so --help stays fast

    started = time.time()
    cases: List[Dict[str, object]] = []

    for index, shape in enumerate(shapes, start=1):
        command = build_command(shape, args)
        print(
            f"[{index}/{len(shapes)}] shape {shape.case_id}: "
            f"B={shape.batch_size} D={shape.d_model} H={shape.num_heads} "
            f"S={shape.seq_len} L={shape.num_layers}",
            flush=True,
        )
        case_started = time.time()
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True
        )
        case_elapsed = time.time() - case_started
        output = completed.stdout + completed.stderr
        (out_dir / f"case_{shape.case_id:02d}.log").write_text(
            " ".join(command) + "\n\n" + output
        )

        record: Dict[str, object] = {
            "case_id": shape.case_id,
            "shape": {
                "batch_size": shape.batch_size,
                "d_model": shape.d_model,
                "num_heads": shape.num_heads,
                "seq_len": shape.seq_len,
                "num_layers": shape.num_layers,
                "causal": shape.causal,
                "ffn_dim": shape.ffn_dim,
            },
            "command": command,
            "returncode": completed.returncode,
            "elapsed_seconds": case_elapsed,
            "log": f"case_{shape.case_id:02d}.log",
        }
        record.update(parse_output(output))

        if "speedup" not in record:
            record["error"] = "the harness printed no speedup line"
            print(f"    ERROR (exit {completed.returncode}); see the log", flush=True)
        else:
            accuracy = record.get("accuracy") or {}
            print(
                f"    accuracy={'PASS' if accuracy.get('passed') else 'FAIL'} | "
                f"baseline={record['baseline']['median_ms']:.4f} ms | "
                f"optimized={record['optimized']['median_ms']:.4f} ms | "
                f"speedup={record['speedup']:.3f}x | "
                f"({case_elapsed:.1f} s)",
                flush=True,
            )
        cases.append(record)

    scored = [case for case in cases if "speedup" in case]
    speedups = sorted(float(case["speedup"]) for case in scored)
    summary: Dict[str, object] = {
        "cases_run": len(cases),
        "cases_scored": len(scored),
        "cases_passed": sum(
            1 for case in scored if (case.get("accuracy") or {}).get("passed")
        ),
    }
    if scored:
        middle = len(speedups) // 2
        summary["median_speedup"] = (
            speedups[middle]
            if len(speedups) % 2
            else 0.5 * (speedups[middle - 1] + speedups[middle])
        )
        best = max(scored, key=lambda case: case["speedup"])
        worst = min(scored, key=lambda case: case["speedup"])
        summary["max_speedup"] = float(best["speedup"])
        summary["max_speedup_case"] = best["case_id"]
        summary["min_speedup"] = float(worst["speedup"])
        summary["min_speedup_case"] = worst["case_id"]

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "harness": "torch_transformer_benchmark.py",
        "git": git_state(),
        "device": args.device,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "elapsed_seconds": time.time() - started,
        "summary": summary,
        "cases": cases,
    }

    (out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if scored:
        write_markdown(out_dir / "results.md", result)
    latest = OUT_ROOT / "latest.json"
    latest.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\nwrote {out_dir}/results.json")
    if scored:
        print(f"wrote {out_dir}/results.md")
    print(f"wrote {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

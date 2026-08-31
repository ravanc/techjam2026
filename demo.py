#!/usr/bin/env python3
"""
A watchable run of the benchmark, for a demo or for a long sweep.

`scoreboard.py` prints nothing while a shape runs. Shape 6 holds the terminal
for about a minute, and one CPU call inside it takes 13.6 s. This script runs
the same work and shows the elapsed time while it runs, so the screen always
moves.

    .venv/bin/python3 demo.py                       # every enabled shape
    .venv/bin/python3 demo.py --cases 1,6,13        # a subset
    .venv/bin/python3 demo.py --cases 1,6,13 --quick # one round, fast
    .venv/bin/python3 demo.py --pause 2             # hold each shape on screen

The numbers come from the same helpers `scoreboard.py` uses, with the same
defaults, so a reading here matches a reading there. The timing loop below
mirrors `test_backends.time_backend` exactly. Keep the two the same.

This script WRITES NOTHING. It does not touch `scoreboard.json`,
`history.jsonl` or the CPU cache. `scoreboard.py` stays the only script that
records a reading. `--cpu-cache` here reads the cache file and never counts a
use against it.

Do not run this beside another measurement. The script looks for one and
stops. Read the rules for a measurement in CLAUDE.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx
import torch

from appendix_cases import SHAPES_BY_ID, Shape, parse_selection
from bench_cases import make_case, make_timing_case
from flops import (
    PEAK_TFLOPS,
    SUSTAINED_TFLOPS,
    achieved_tflops,
    model_flops,
)
from scoreboard import (
    accuracy_of,
    choose_repeats,
    cpu_cache_key,
    load_cpu_cache,
)
from test_backends import ALL_BACKENDS, build_backends, warmup
from torch_transformer_benchmark import resolve_dtype

# The label each backend carries on screen.
TITLES = {
    "cpu": "CPU  torch baseline",
    "mps": "MPS  torch baseline",
    "mlx": "MLX  this project",
}


# --------------------------------------------------------------------------
# the screen
# --------------------------------------------------------------------------

class Screen:
    """
    Write to the terminal. Rewrite one line while a step runs.

    A pipe gets plain lines instead, because a pipe cannot move the cursor.
    """

    def __init__(self, stream=sys.stdout, color: Optional[bool] = None):
        self.stream = stream
        self.tty = stream.isatty()
        self.color = self.tty if color is None else color

    def paint(self, text: str, style: str = "") -> str:
        if not self.color or not style:
            return text
        return f"{style}{text}\x1b[0m"

    def line(self, text: str = "") -> None:
        self.stream.write(text + "\n")
        self.stream.flush()

    def rewrite(self, text: str) -> None:
        """Replace the current line. Do nothing on a pipe."""
        if not self.tty:
            return
        self.stream.write("\r" + text + "\x1b[K")
        self.stream.flush()

    def hide_cursor(self) -> None:
        if self.tty:
            self.stream.write("\x1b[?25l")
            self.stream.flush()

    def show_cursor(self) -> None:
        if self.tty:
            self.stream.write("\x1b[?25h")
            self.stream.flush()


DIM = "\x1b[2m"
BOLD = "\x1b[1m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
CYAN = "\x1b[36m"

SPINNER = "|/-\\"


class Step:
    """
    One step of work, with the elapsed time on screen while it runs.

    A thread rewrites the line about ten times a second. The main thread does
    the work, and it sets `note` to show the progress inside the step.

        with Step(screen, "cpu", "torch baseline") as step:
            step.note = "run 2 / 3"
            ...
        step.finish("15777.204 ms")
    """

    def __init__(self, screen: Screen, name: str, note: str = "",
                 indent: str = "  "):
        self.screen = screen
        self.name = name
        self.note = note
        self.indent = indent
        self.started = 0.0
        self.elapsed = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame = 0

    def __enter__(self) -> "Step":
        self.started = time.perf_counter()
        if self.screen.tty:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
        # A pipe gets no line here. `finish()` prints one line for each step,
        # and it carries the elapsed time.
        return self

    def __exit__(self, *_exception) -> None:
        self._settle()

    def _settle(self) -> None:
        """Stop the thread and hold the elapsed time. Safe to call twice."""
        if self._stop.is_set():
            return
        self.elapsed = time.perf_counter() - self.started
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _tick(self) -> None:
        while not self._stop.wait(0.1):
            self._frame += 1
            spin = SPINNER[self._frame % len(SPINNER)]
            elapsed = time.perf_counter() - self.started
            body = f"{self.indent}{self.name:<22} {elapsed:7.1f} s  {spin}"
            if self.note:
                body += f"  {self.note}"
            self.screen.rewrite(self.screen.paint(body, DIM))

    def finish(self, result: str, style: str = "") -> None:
        """Print the final line of the step, with the time it took."""
        self._settle()
        body = f"{self.indent}{self.name:<22} {self.elapsed:7.1f} s     "
        self.screen.rewrite("")
        self.screen.line(
            self.screen.paint(body, DIM) + self.screen.paint(result, style)
        )


def bar(value: float, longest: float, width: int = 22) -> str:
    """A bar for one time, against the slowest time on the shape."""
    if longest <= 0:
        return ""
    filled = max(1, int(round(width * value / longest)))
    return "█" * filled + "·" * (width - filled)


def clock(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes:d}m {rest:02d}s"


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------

@dataclass
class ShapeResult:
    case_id: int
    flops: int
    tokens: int
    median_ms: Dict[str, float] = field(default_factory=dict)
    accuracy: Dict[str, Dict] = field(default_factory=dict)
    failed: Dict[str, str] = field(default_factory=dict)
    cached_cpu: Optional[str] = None

    def speedup(self, name: str) -> Optional[float]:
        base = self.median_ms.get("cpu")
        mine = self.median_ms.get(name)
        return None if base is None or mine is None else base / mine


def busy_gpu() -> List[str]:
    """Look for another measurement. Rule 1 of a measurement in CLAUDE.md."""
    try:
        listing = subprocess.run(
            ["ps", "-Ao", "pid,etime,%cpu,command"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    mine = {str(os.getpid()), str(os.getppid())}
    found = []
    for row in listing.splitlines():
        fields = row.split(None, 3)
        if len(fields) < 4 or fields[0] in mine:
            continue
        # Match the interpreter itself, not a wrapper that names it. `script`
        # and `time` both carry the python path in their own command line.
        program = fields[3].split()[0]
        if not program.endswith(".venv/bin/python3"):
            continue
        if "shell-snapshots" in fields[3]:
            continue
        found.append(row.strip())
    return found


def free_memory() -> None:
    gc.collect()
    mx.clear_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def time_backend_live(backend, local, count: int, step: Step,
                      done: int, total: int) -> List[float]:
    """
    Time `count` calls and show the progress.

    This mirrors `test_backends.time_backend`. The only difference is the
    `step.note` line, and it sits outside the timed region. Change both
    functions together.
    """
    samples: List[float] = []
    with torch.inference_mode():
        for index in range(count):
            step.note = f"call {done + index + 1} / {total}"
            backend.sync()
            start = time.perf_counter_ns()
            backend.model(local.x, local.valid_mask)
            backend.sync()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def read_cpu_cache(cache: Dict, config: Dict, args, repeats: int) -> Optional[Dict]:
    """Read a stored CPU reading. Never count a use, never write the file."""
    key = cpu_cache_key(config, args, repeats)
    return cache.get("entries", {}).get(key)


def run_shape(shape: Shape, args, names: List[str], screen: Screen,
              cache: Optional[Dict], position: str) -> ShapeResult:
    config = shape.config()
    config.validate()
    dtype = resolve_dtype(args.dtype)
    flops = model_flops(config)
    repeats = choose_repeats(flops, args.repeats)
    total_calls = repeats * args.rounds

    result = ShapeResult(
        case_id=shape.case_id,
        flops=flops,
        tokens=config.batch_size * config.seq_len,
    )

    title = (f"shape {shape.case_id}  B{shape.batch_size} D{shape.d_model} "
             f"H{shape.num_heads} S{shape.seq_len} L{shape.num_layers}"
             f"{' causal' if shape.causal else ''}")
    screen.line()
    screen.line(screen.paint(f"{position}  {title}", BOLD))
    screen.line(screen.paint(
        f"  {flops / 1e9:.2f} GFLOP   {result.tokens:,} tokens   "
        f"{repeats} calls x {args.rounds} rounds", DIM))

    # --- build one model for each backend, from the same weights ---
    plan = None
    with Step(screen, "build models", "shared weights") as step:
        backends = build_backends(names, config, dtype, args.seed,
                                  args.mlx_torch_device)
        for backend in backends:
            if backend.name == "mlx":
                backend.model._build_mlx_weights()
                plan = backend.model.plan.describe()
    step.finish(f"{len(backends)} models")
    if plan is not None:
        screen.line(screen.paint(f"  plan: {plan}", DIM))

    # --- accuracy, against the CPU baseline ---
    if not args.skip_accuracy:
        case = make_case(config, args.seed, args.padding_ratio)
        reference = None
        for backend in backends:
            if backend.name == "cpu":
                with Step(screen, "accuracy reference", "cpu baseline") as step:
                    reference = backend.run(case)
                step.finish("built")
                continue
            with Step(screen, f"accuracy {backend.name}", "one trial") as step:
                try:
                    output = backend.run(case)
                except Exception as error:  # noqa: BLE001
                    result.failed[backend.name] = f"accuracy: {type(error).__name__}"
                    step.finish(f"FAILED {type(error).__name__}", RED)
                    continue
            if reference is None:
                del output
                continue
            check = accuracy_of(reference, output, args.rtol, args.atol)
            result.accuracy[backend.name] = check
            verdict = "PASS" if check["passed"] else "FAIL"
            step.finish(
                f"{verdict}  max_abs {check['max_abs_error']:.2e}  "
                f"atol {args.atol:g}  "
                f"{check['failed_elements']:,} / {check['total_elements']:,} bad",
                GREEN if check["passed"] else RED,
            )
            del output
        del case, reference
        free_memory()

    # --- the CPU reading, from the cache when the user asks for it ---
    served = None
    if cache is not None and "cpu" in names and "cpu" not in result.failed:
        served = read_cpu_cache(cache, config.__dict__.copy(), args, repeats)
        if served is not None:
            result.median_ms["cpu"] = served["median_ms"]
            result.cached_cpu = served["measured_at"]
            screen.line(
                f"  {'CPU  torch baseline':<22} {'cached':>7}     "
                + screen.paint(
                    f"{served['median_ms']:10.3f} ms   read from the cache, "
                    f"measured {served['measured_at']}", DIM)
            )

    # --- timing, one backend at a time ---
    timing_case = make_timing_case(config, args.seed, args.padding_ratio)
    samples: Dict[str, List[float]] = {}
    order = [b for b in backends if not (b.name == "cpu" and served is not None)]

    for backend in order:
        if backend.name in result.failed:
            continue
        local = timing_case.to(backend.device, backend.dtype)
        with Step(screen, TITLES.get(backend.name, backend.name),
                  "warmup") as step:
            try:
                warmup(backend, timing_case, args.warmup)
                for round_index in range(args.rounds):
                    samples.setdefault(backend.name, []).extend(
                        time_backend_live(backend, local, repeats, step,
                                          round_index * repeats, total_calls)
                    )
            except Exception as error:  # noqa: BLE001
                result.failed[backend.name] = f"{type(error).__name__}: {error}"[:120]
                step.finish(f"FAILED {type(error).__name__}", RED)
                del local
                continue
        median = statistics.median(samples[backend.name])
        result.median_ms[backend.name] = median
        speedup = result.speedup(backend.name)
        tail = f"{median:10.3f} ms"
        if speedup is not None and backend.name != "cpu":
            tail += f"   {speedup:6.2f}x vs CPU"
        if backend.name == "mlx":
            tail += (f"   {achieved_tflops(flops, median):5.3f} TFLOP/s"
                     f"   MFU {achieved_tflops(flops, median) / PEAK_TFLOPS:.0%}")
        step.finish(tail, BOLD if backend.name == "mlx" else "")
        del local

    del backends, timing_case
    free_memory()

    # --- the bars, so the shape reads at a glance ---
    if len(result.median_ms) > 1:
        longest = max(result.median_ms.values())
        screen.line()
        for name in ("cpu", "mps", "mlx"):
            if name not in result.median_ms:
                continue
            value = result.median_ms[name]
            style = CYAN if name == "mlx" else DIM
            screen.line(
                f"  {name.upper():<4} "
                + screen.paint(bar(value, longest), style)
                + f" {value:10.3f} ms"
            )

    return result


# --------------------------------------------------------------------------
# the summary
# --------------------------------------------------------------------------

def print_summary(results: List[ShapeResult], screen: Screen,
                  wall_seconds: float) -> None:
    screen.line()
    screen.line(screen.paint("=" * 78, DIM))
    screen.line(screen.paint("Summary", BOLD))
    screen.line()
    header = (f"{'#':>3}  {'CPU ms':>11}  {'MPS ms':>10}  {'MLX ms':>10}  "
              f"{'MLX vs CPU':>10}  {'TFLOP/s':>8}  {'acc':>5}")
    screen.line(header)
    screen.line(screen.paint("-" * len(header), DIM))

    for result in results:
        def cell(name: str, width: int = 10) -> str:
            value = result.median_ms.get(name)
            return f"{value:{width}.3f}" if value is not None else f"{'—':>{width}}"

        speedup = result.speedup("mlx")
        speed_text = f"{speedup:9.2f}x" if speedup is not None else f"{'—':>10}"
        mlx_ms = result.median_ms.get("mlx")
        rate = f"{achieved_tflops(result.flops, mlx_ms):8.3f}" if mlx_ms else f"{'—':>8}"
        check = result.accuracy.get("mlx")
        verdict = "—" if check is None else ("PASS" if check["passed"] else "FAIL")
        mark = "†" if result.cached_cpu else " "
        screen.line(
            f"{result.case_id:>3}  {cell('cpu', 11)}{mark} {cell('mps')}  "
            f"{cell('mlx')}  {speed_text}  {rate}  {verdict:>5}"
        )

    # Sum a backend only when every shape holds a reading for it. A partial
    # sum compares a different set of shapes and it means nothing.
    totals: Dict[str, Optional[float]] = {}
    for name in ALL_BACKENDS:
        readings = [r.median_ms[name] for r in results if name in r.median_ms]
        totals[name] = sum(readings) if len(readings) == len(results) else None

    def total_cell(name: str, width: int = 10) -> str:
        value = totals[name]
        return f"{value:{width}.3f}" if value is not None else f"{'—':>{width}}"

    screen.line(screen.paint("-" * len(header), DIM))
    row = (f"{'sum':>3}  {total_cell('cpu', 11)}  {total_cell('mps')}  "
           f"{total_cell('mlx')}  ")
    if totals["cpu"] is not None and totals["mlx"]:
        row += screen.paint(f"{totals['cpu'] / totals['mlx']:9.2f}x", BOLD)
    screen.line(row)
    if any(r.cached_cpu for r in results):
        screen.line()
        screen.line(screen.paint(
            "† the CPU reading came from the cache, so the speedup mixes "
            "two sweeps.", DIM))
    screen.line()
    screen.line(screen.paint(
        f"MFU denominator {PEAK_TFLOPS:.3f} TFLOP/s. This GPU sustains "
        f"{SUSTAINED_TFLOPS:.2f} TFLOP/s, so read an MFU against 82%.", DIM))
    screen.line(screen.paint(
        f"demo took {clock(wall_seconds)}. This script recorded nothing. "
        f"Run scoreboard.py for a reading that counts.", DIM))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the benchmark with the elapsed time on screen.",
    )
    parser.add_argument("--cases", default="all", help='e.g. "1,3,5-8" or "all"')
    parser.add_argument("--backends", default=",".join(ALL_BACKENDS))
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--mlx-torch-device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=None,
                        help="override the automatic call count")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="one round and three calls; for a rehearsal only")
    parser.add_argument("--cpu-cache", action="store_true",
                        help="read the stored CPU reading; never writes it")
    parser.add_argument("--cpu-cache-path",
                        default="profiling/results/cpu_cache.json")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds to hold each shape on screen")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="run even when another measurement is active")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.rounds = 1
        args.repeats = args.repeats or 3
        args.warmup = min(args.warmup, 2)

    screen = Screen(color=False if args.no_color else None)

    other = busy_gpu()
    if other and not args.force:
        screen.line(screen.paint("another measurement is running:", RED))
        for row in other:
            screen.line(f"  {row}")
        screen.line("Two runs share one GPU, so both readings are false.")
        screen.line("Wait for it to end, or pass --force to run anyway.")
        return 1

    names = [n.strip() for n in args.backends.split(",") if n.strip()]
    for name in names:
        if name not in ALL_BACKENDS:
            raise SystemExit(f"unknown backend {name}; choose from {ALL_BACKENDS}")
    if "mps" in names and not torch.backends.mps.is_available():
        screen.line(screen.paint("[warning] MPS is unavailable; dropping it", RED))
        names.remove("mps")

    cache = load_cpu_cache(args.cpu_cache_path) if args.cpu_cache else None
    case_ids = parse_selection(args.cases)

    screen.line(screen.paint(
        "UserOptimizedTransformer against the torch baseline", BOLD))
    screen.line(screen.paint(
        f"shapes {', '.join(str(i) for i in case_ids)}   dtype {args.dtype}   "
        f"backends {', '.join(names)}", DIM))
    screen.line(screen.paint(
        "Every call is bracketed by a device synchronize.", DIM))
    if args.quick:
        screen.line(screen.paint(
            "--quick is on. These numbers are for a rehearsal, not a reading.",
            RED))

    started = time.perf_counter()
    results: List[ShapeResult] = []
    screen.hide_cursor()
    try:
        for index, case_id in enumerate(case_ids, start=1):
            shape = SHAPES_BY_ID[case_id]
            if not shape.enabled:
                screen.line(f"\nshape {case_id} skipped: {shape.note}")
                continue
            results.append(run_shape(
                shape, args, names, screen, cache, f"[{index}/{len(case_ids)}]"
            ))
            if args.pause > 0 and index < len(case_ids):
                time.sleep(args.pause)
    except KeyboardInterrupt:
        screen.line()
        screen.line(screen.paint("stopped by the user", RED))
    finally:
        screen.show_cursor()

    if results:
        print_summary(results, screen, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

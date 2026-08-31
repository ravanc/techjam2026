#!/usr/bin/env python3
"""Measure GPU idle time on the real timeline, for one appendix shape.

The residual accounting bounds GPU idle from a slope fit. This tool reads the
gaps directly. Instruments records a `Metal System Trace`, and `xctrace export`
gives the GPU execution track as XML. So no Xcode window is needed.

Two steps:

    ./profiling/gpu_timeline.sh --case 6 --iterations 3
    .venv/bin/python3 profiling/gpu_timeline.py report profiling/traces/gpu_timeline.trace

Step 1 records. It drives `UserOptimizedTransformer` alone, and it puts one
`mlx-forward` signpost around each forward pass.

Step 2 reports. For each `mlx-forward` window it clips the GPU intervals of the
python process, joins the overlaps, and prints the busy time, the idle time and
the largest gaps.

The `metal-gpu-intervals` table holds one row for each command encoder that the
GPU ran, with a start and a duration in nanoseconds. It covers every process on
the device, so the report keeps only the rows whose label names the recorded
process.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

SUBSYSTEM = "com.techjam.profiling"
REGION = "mlx-forward"


# --------------------------------------------------------------------------
# Step 1: record
# --------------------------------------------------------------------------

def run_record(args) -> int:
    import torch

    import signposts
    from appendix_cases import SHAPES_BY_ID
    from bench_cases import make_timing_case
    from torch_transformer_benchmark import (
        BaselineTransformer,
        UserOptimizedTransformer,
        copy_model_weights,
    )

    shape = SHAPES_BY_ID[args.case]
    config = shape.config()
    config.validate()

    torch.manual_seed(args.seed)
    reference = BaselineTransformer(config)
    model = UserOptimizedTransformer(config)
    copy_model_weights(reference, model, strict=True)
    del reference
    model = model.eval()

    case = make_timing_case(config, args.seed)
    x, mask = case.x, case.valid_mask

    print(f"case {args.case}: {config}")
    print(f"signposts={signposts.available()}")
    if not signposts.available():
        print("[warning] no signpost shim; run `make -C profiling`")

    import mlx.core as mx

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x, mask)
        mx.synchronize()

        signposts.event("record-begin")
        for _ in range(args.iterations):
            with signposts.interval(REGION):
                model(x, mask)
                mx.synchronize()
        signposts.event("record-end")

    print(f"ran {args.iterations} forward passes under signpost {REGION!r}")
    return 0


# --------------------------------------------------------------------------
# Step 2: report
# --------------------------------------------------------------------------

def export(trace: str, schema: str) -> bytes:
    """Pull one table out of the trace as XML."""
    xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    proc = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", trace, "--xpath", xpath],
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.exit(f"xctrace export failed:\n{proc.stderr.decode(errors='replace')}")
    return proc.stdout


def rows(xml: bytes):
    """Yield one dict for each row, with the raw text and the display form.

    The export interns a repeated value: an element carries either an `id` that
    defines it, or a `ref` that points at one defined before. So a reader must
    hold a table of the ids it saw.
    """
    root = ET.fromstring(xml)
    node = root.find("node")
    if node is None:
        return
    columns = [c.findtext("mnemonic") for c in node.find("schema").findall("col")]

    seen: dict[str, tuple[str, str]] = {}

    def value(el) -> tuple[str, str]:
        ref = el.get("ref")
        if ref is not None:
            return seen.get(ref, ("", ""))
        pair = (el.text or "", el.get("fmt") or "")
        ident = el.get("id")
        if ident is not None:
            seen[ident] = pair
        return pair

    for row in node.findall("row"):
        out: dict[str, tuple[str, str]] = {}
        for index, el in enumerate(row):
            if index >= len(columns):
                break
            if el.tag == "sentinel":
                continue
            out[columns[index]] = value(el)
        yield out


def as_int(pair: tuple[str, str]) -> int:
    try:
        return int(pair[0])
    except (TypeError, ValueError):
        return 0


def windows(trace: str, region: str) -> list[tuple[int, int]]:
    """Return the (start, stop) of every signpost interval with this name.

    The `os-signpost` table names its time column `time`, and it gives the
    interval id in `identifier`.
    """
    open_at: dict[str, int] = {}
    found: list[tuple[int, int]] = []
    for row in rows(export(trace, "os-signpost")):
        if row.get("subsystem", ("", ""))[1] != SUBSYSTEM:
            continue
        if row.get("message", ("", ""))[1] != region:
            continue
        spid = row.get("identifier", ("", ""))[1]
        kind = row.get("event-type", ("", ""))[1]
        stamp = as_int(row.get("time", ("0", "")))
        if kind == "Begin":
            open_at[spid] = stamp
        elif kind == "End" and spid in open_at:
            found.append((open_at.pop(spid), stamp))
    found.sort()
    return found


def device_active(trace: str) -> list[tuple[int, int]]:
    """Return every interval where the GPU device itself reports Active.

    This track counts every process on the device, so it is a ceiling on our
    own busy time, not a substitute for it.
    """
    out: list[tuple[int, int]] = []
    for row in rows(export(trace, "metal-gpu-state-intervals")):
        if row.get("state", ("", ""))[1] != "Active":
            continue
        start = as_int(row.get("start", ("0", "")))
        length = as_int(row.get("duration", ("0", "")))
        if length > 0:
            out.append((start, start + length))
    out.sort()
    return out


def gpu_intervals(trace: str, process: str) -> list[tuple[int, int, str]]:
    """Return every GPU execution interval of one process, sorted by start."""
    out: list[tuple[int, int, str]] = []
    for row in rows(export(trace, "metal-gpu-intervals")):
        label = row.get("event-label", ("", ""))[1]
        if process not in label:
            continue
        start = as_int(row.get("start", ("0", "")))
        length = as_int(row.get("duration", ("0", "")))
        if length <= 0:
            continue
        out.append((start, start + length, label))
    out.sort()
    return out


def join(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Join the overlapping spans, so busy time is never counted twice."""
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(spans):
        if merged and start <= merged[-1][1]:
            if stop > merged[-1][1]:
                merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return merged


def processes(trace: str) -> list[str]:
    """List the processes that ran GPU work, with their total GPU time."""
    label_re = re.compile(r"\(([^()]+) \((\d+)\)\)")
    total: dict[str, int] = {}
    for row in rows(export(trace, "metal-gpu-intervals")):
        found = label_re.search(row.get("event-label", ("", ""))[1])
        name = f"{found.group(1)} ({found.group(2)})" if found else "?"
        total[name] = total.get(name, 0) + as_int(row.get("duration", ("0", "")))
    return [f"{k:<28} {v / 1e6:9.3f} ms" for k, v in
            sorted(total.items(), key=lambda kv: -kv[1])]


def clip(spans, start: int, stop: int) -> list[tuple[int, int]]:
    """Cut the spans to one window, and join the overlaps."""
    return join([
        (max(a, start), min(b, stop))
        for a, b in spans
        if b > start and a < stop
    ])


def run_report(args) -> int:
    found = windows(args.trace, args.region)
    if not found:
        print(f"no {args.region!r} signpost in {args.trace}")
        print("record with: ./profiling/gpu_timeline.sh --case 6")
        return 1

    spans = [(a, b) for a, b, _ in gpu_intervals(args.trace, args.process)]
    if not spans:
        print(f"no GPU interval names {args.process!r}. GPU work in this trace:")
        for line in processes(args.trace):
            print(f"  {line}")
        return 1
    active = device_active(args.trace)

    print(f"trace     {args.trace}")
    print(f"region    {args.region} x{len(found)}")
    print(f"process   {args.process}, {len(spans)} GPU encoder intervals")
    print()
    print("busy = our own encoders. active = the whole device, every process.")
    print()
    print(f"{'#':>3} {'window ms':>10} {'busy ms':>9} {'head ms':>8} "
          f"{'inner ms':>9} {'tail ms':>8} {'inner %':>8} {'gaps':>5} "
          f"{'active %':>9}")
    print("-" * 86)

    idle_share: list[float] = []
    inner_share: list[float] = []
    heads: list[float] = []
    active_share: list[float] = []
    all_gaps: list[tuple[int, int, int]] = []   # (length, offset in window, index)

    for index, (start, stop) in enumerate(found, 1):
        merged = clip(spans, start, stop)
        span = stop - start
        busy = sum(b - a for a, b in merged)
        idle = span - busy
        # Split the idle time three ways. The head is the CPU work before the
        # first kernel, and the tail is the wait after the last one. Both are
        # the framework boundary. Only the inner gaps are a GPU stall.
        head = merged[0][0] - start
        tail = stop - merged[-1][1]
        gaps: list[int] = []
        edge = merged[0][1]
        for a, b in merged[1:]:
            if a > edge:
                gaps.append(a - edge)
                all_gaps.append((a - edge, edge - start, index))
            edge = b
        inner = sum(gaps)
        on = sum(b - a for a, b in clip(active, start, stop))
        idle_share.append(100.0 * idle / span if span else 0.0)
        inner_share.append(100.0 * inner / span if span else 0.0)
        heads.append(head / 1e6)
        active_share.append(100.0 * on / span if span else 0.0)
        print(f"{index:>3} {span / 1e6:>10.3f} {busy / 1e6:>9.3f} "
              f"{head / 1e6:>8.3f} {inner / 1e6:>9.3f} {tail / 1e6:>8.3f} "
              f"{inner_share[-1]:>8.2f} {len(gaps):>5} "
              f"{active_share[-1]:>9.2f}")

    print("-" * 86)
    print(f"median idle {statistics.median(idle_share):.2f}%, of which the "
          f"head is {statistics.median(heads):.3f} ms")
    print(f"median INNER idle {statistics.median(inner_share):.2f}%  "
          f"<- the only part a kernel change can win")
    print(f"median device active {statistics.median(active_share):.2f}%")

    if all_gaps:
        print()
        print(f"the {min(args.top, len(all_gaps))} largest gaps:")
        print(f"{'gap ms':>9} {'at ms':>9} {'window':>7}   (head gaps excluded)")
        for length, offset, index in sorted(all_gaps, reverse=True)[:args.top]:
            print(f"{length / 1e6:>9.3f} {offset / 1e6:>9.3f} {index:>7}")

        cut = args.gap_floor * 1e6
        small = [g for g, _, _ in all_gaps if g < cut]
        large = [g for g, _, _ in all_gaps if g >= cut]
        print()
        print(f"{len(small)} gaps under {args.gap_floor:g} ms hold "
              f"{sum(small) / 1e6 / len(found):.3f} ms per window")
        print(f"{len(large)} gaps over {args.gap_floor:g} ms hold "
              f"{sum(large) / 1e6 / len(found):.3f} ms per window")

    print()
    print("GPU time by process over the whole trace:")
    for line in processes(args.trace):
        print(f"  {line}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    rec = sub.add_parser("record", help="drive the MLX model under signposts")
    rec.add_argument("--case", type=int, default=6)
    rec.add_argument("--warmup", type=int, default=3)
    rec.add_argument("--iterations", type=int, default=3)
    rec.add_argument("--seed", type=int, default=1234)

    rep = sub.add_parser("report", help="read the GPU gaps out of a trace")
    rep.add_argument("trace")
    rep.add_argument("--region", default=REGION)
    rep.add_argument("--process", default="python3")
    rep.add_argument("--top", type=int, default=15)
    rep.add_argument("--gap-floor", type=float, default=0.05,
                     help="the gap size that separates launch cost from a stall")

    args = ap.parse_args()
    if args.mode == "record":
        return run_record(args)
    return run_report(args)


if __name__ == "__main__":
    raise SystemExit(main())

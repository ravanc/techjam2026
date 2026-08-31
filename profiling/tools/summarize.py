#!/usr/bin/env python3
"""Summarize os_signpost intervals from an Instruments .trace file.

Pairs Begin/End signpost events emitted by signposts.py and reports per-region
wall-clock statistics, so a recording can be read without opening the
Instruments UI.

Usage:
    python3 profiling/tools/summarize.py profiling/traces/benchmark.trace
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

SUBSYSTEM = "com.techjam.profiling"
XPATH = '/trace-toc/run[@number="1"]/data/table[@schema="os-signpost"]'


def export(trace: str) -> bytes:
    proc = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", trace, "--xpath", XPATH],
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.exit(f"xctrace export failed:\n{proc.stderr.decode(errors='replace')}")
    return proc.stdout


def parse(xml: bytes):
    """Yield (timestamp_ns, event_type, signpost_id, subsystem, message) rows.

    The export format interns repeated values: an element carries either an
    `id` defining a value or a `ref` pointing at one defined earlier, so
    resolving refs against a table of seen ids is required to read any row.
    """
    root = ET.fromstring(xml)
    node = root.find("node")
    if node is None:
        return

    schema = node.find("schema")
    columns = [c.findtext("mnemonic") for c in schema.findall("col")]

    values: dict[str, str] = {}

    def resolve(el) -> str:
        ref = el.get("ref")
        if ref is not None:
            return values.get(ref, "")
        # `fmt` carries the display form; fall back to element text.
        val = el.get("fmt")
        if val is None:
            val = (el.text or "")
        ident = el.get("id")
        if ident is not None:
            values[ident] = val
        return val

    for row in node.findall("row"):
        fields: dict[str, str] = {}
        for idx, el in enumerate(row):
            if idx >= len(columns):
                break
            if el.tag == "sentinel":
                continue
            fields[columns[idx]] = resolve(el)

        raw_time = row[0].get("ref")
        ts = values.get(raw_time) if raw_time else (row[0].text or "0")
        # `fmt` on event-time is a display string; the element text is ns.
        try:
            ts_ns = int(row[0].text) if row[0].text else 0
        except (TypeError, ValueError):
            ts_ns = 0

        yield (
            ts_ns,
            fields.get("event-type", ""),
            fields.get("identifier", ""),
            fields.get("subsystem", ""),
            fields.get("message", "") or fields.get("name", ""),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--subsystem", default=SUBSYSTEM)
    args = ap.parse_args()

    open_intervals: dict[str, tuple[int, str]] = {}
    durations: dict[str, list[float]] = defaultdict(list)
    events: dict[str, int] = defaultdict(int)

    for ts, etype, spid, subsystem, message in parse(export(args.trace)):
        if subsystem != args.subsystem:
            continue
        label = message or "(unnamed)"
        if etype == "Begin":
            open_intervals[spid] = (ts, label)
        elif etype == "End":
            start = open_intervals.pop(spid, None)
            if start is not None:
                durations[start[1]].append((ts - start[0]) / 1e6)
        elif etype == "Event":
            events[label] += 1

    if not durations and not events:
        print(f"no signposts found for subsystem {args.subsystem!r}")
        print("did the recording include: --instrument os_signpost ?")
        return 1

    print(f"{'region':<24}{'n':>5}{'median ms':>12}{'mean ms':>11}"
          f"{'min ms':>10}{'max ms':>10}")
    print("-" * 72)
    for label in sorted(durations):
        d = durations[label]
        print(f"{label:<24}{len(d):>5}{statistics.median(d):>12.4f}"
              f"{statistics.fmean(d):>11.4f}{min(d):>10.4f}{max(d):>10.4f}")

    if events:
        print()
        for label, count in sorted(events.items()):
            print(f"event: {label} x{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

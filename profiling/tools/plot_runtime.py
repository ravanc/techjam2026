#!/usr/bin/env python3
"""Plot the measured runtime of the 13 test shapes on CPU, MPS and MLX.

    .venv/bin/python3 profiling/tools/plot_runtime.py

The y axis is logarithmic. The measured times run from 0.61 ms to 15,420 ms,
a spread of 25,000x, so a linear axis hides twelve of the thirteen shapes. Read
the printed value on each bar for the exact figure: on a log axis the height of
a bar shows the ratio between shapes, not the value itself.

It writes a PNG at 200 dpi for slides and documents, and an SVG and a PDF that
stay sharp at every size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "profiling" / "results" / "harness"

# CPU and MLX from the harness run; MPS from the scoreboard run. Milliseconds,
# median of the timed calls.
SHAPES = [
    #  #   label            cpu         mps        mlx
    (1,  "base",          45.9754,    17.4401,    3.2652),
    (2,  "batch 1",        1.4456,     1.5256,    0.6104),
    (3,  "batch 4",        4.0834,     2.0156,    0.6723),
    (4,  "batch 16",      13.9391,     4.7035,    1.2029),
    (5,  "batch 128",    102.2755,    33.7087,    5.8422),
    (6,  "batch 10,000", 15419.7160, 2771.8542, 436.8107),
    (7,  "dim 32",        23.6896,    11.5201,    1.0029),
    (8,  "dim 1024",     461.1526,   167.5706,  115.5255),
    (9,  "1 head",        26.3907,     8.1035,    3.3803),
    (10, "2 heads",       37.3346,    13.0954,    3.2754),
    (11, "16 heads",     121.8538,    42.5131,    3.2514),
    (12, "seq 32",         7.5305,     3.4665,    1.1551),
    (13, "seq 1024",    1845.8785,   577.3611,   37.4570),
]

# dataviz reference palette, categorical slots 1-3. validate_palette.js passes
# every check in both modes; aqua warns on contrast against a light surface, and
# the value label on every bar is the documented relief.
LIGHT = {
    "cpu": "#2a78d6", "mps": "#eb6834", "mlx": "#1baf7a",
    "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e",
    "faint": "#8a8880", "grid": "#d8d7d3", "axis": "#3c3b38",
}
DARK = {
    "cpu": "#3987e5", "mps": "#d95926", "mlx": "#199e70",
    "surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7",
    "faint": "#8a8880", "grid": "#33332f", "axis": "#9a9992",
}


def compact(value: float) -> str:
    """Short enough to sit above a 20px bar, precise enough to be useful."""
    if value >= 10000:
        return f"{value / 1000:.1f}k"
    if value >= 1000:
        return f"{value / 1000:.2f}k"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw(mode: str, out_stem: Path) -> None:
    c = LIGHT if mode == "light" else DARK

    plt.rcParams.update({
        # DejaVu Sans is matplotlib's own face; keeping it is part of the look.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.facecolor": c["surface"],
        "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"],
        "text.color": c["text"],
        "axes.labelcolor": c["text"],
        "axes.edgecolor": c["axis"],
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
    })

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(13.0, 9.2), height_ratios=[1.0, 0.92])

    positions = list(range(len(SHAPES)))
    bar_w = 0.27
    series = [
        ("CPU (torch baseline)", 2, c["cpu"]),
        ("MPS (same model, GPU)", 3, c["mps"]),
        ("MLX (optimized)", 4, c["mlx"]),
    ]

    # ---- panel A: absolute runtime, log ---------------------------------
    # A log axis has no zero, so the bars stand on an explicit floor below the
    # smallest measurement rather than on an undefined baseline.
    floor = 0.35
    for offset, (label, column, colour) in zip((-bar_w, 0.0, bar_w), series):
        values = [row[column] for row in SHAPES]
        xs = [p + offset for p in positions]
        ax.bar(xs, [v - floor for v in values], bottom=floor, width=bar_w,
               label=label, color=colour, edgecolor="black", linewidth=0.5,
               zorder=3)
        for x, value in zip(xs, values):
            ax.text(x, value * 1.15, compact(value), ha="center", va="bottom",
                    fontsize=7.2, color=c["text"], rotation=90, zorder=4)

    ax.set_yscale("log")
    ax.set_ylim(floor, 400000)
    ax.set_ylabel("Runtime (ms)", fontsize=11)
    ax.set_title("A.  Absolute Runtime (log ms)", fontsize=12.5,
                 fontweight="bold", color=c["text"], loc="left", pad=10)
    ax.grid(axis="y", which="major", color=c["grid"], linewidth=0.8, zorder=0)
    ax.grid(axis="y", which="minor", color=c["grid"], linewidth=0.4,
            alpha=0.6, zorder=0)
    ax.legend(loc="upper left", frameon=True, fontsize=9, framealpha=1.0,
              edgecolor=c["axis"], facecolor=c["surface"], ncol=3,
              labelcolor=c["text"])

    # ---- panel B: speedup, linear ---------------------------------------
    # The log panel compresses ratios: a 47x win looks like a small step. A
    # linear axis is what makes the size of the win visible, and speedup is the
    # measure that fits on one — raw ms on a linear axis hides twelve shapes.
    gains = [
        ("MLX vs CPU", 2, c["cpu"]),
        ("MLX vs MPS", 3, c["mps"]),
    ]
    gap = 0.19
    for offset, (label, column, colour) in zip((-gap, gap), gains):
        values = [row[column] / row[4] for row in SHAPES]
        xs = [p + offset for p in positions]
        bx.bar(xs, values, width=0.34, label=label, color=colour,
               edgecolor="black", linewidth=0.5, zorder=3)
        for x, value in zip(xs, values):
            bx.text(x, value + 0.9, f"{value:.1f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=c["text"], zorder=4)

    bx.axhline(1.0, color=c["axis"], linewidth=1.0, zorder=2)
    bx.set_ylim(0, 56)
    bx.set_ylabel("Speedup (x)", fontsize=11)
    bx.set_xlabel("Test case", fontsize=11, labelpad=7)
    bx.set_title("B.  Speedup", fontsize=12.5, fontweight="bold",
                 color=c["text"], loc="left", pad=10)
    bx.grid(axis="y", color=c["grid"], linewidth=0.8, zorder=0)
    bx.legend(loc="upper left", frameon=True, fontsize=9, framealpha=1.0,
              edgecolor=c["axis"], facecolor=c["surface"], ncol=2,
              labelcolor=c["text"])

    # ---- shared chrome ---------------------------------------------------
    for axis in (ax, bx):
        axis.set_xticks(positions)
        axis.set_xlim(-0.62, len(SHAPES) - 0.38)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", which="major", direction="out",
                         length=4.5, width=0.9, labelsize=9)
        axis.tick_params(axis="y", which="minor", direction="out",
                         length=2.5, width=0.7)
        for side in ("top", "right", "bottom", "left"):
            axis.spines[side].set_visible(True)
            axis.spines[side].set_linewidth(0.9)

    for axis in (ax, bx):
        axis.set_xticklabels([str(row[0]) for row in SHAPES], fontsize=9.5)

    fig.suptitle("Runtime and speedup across 13 cases", fontsize=16,
                 fontweight="bold", color=c["text"], y=0.982)
    fig.subplots_adjust(left=0.060, right=0.988, top=0.915, bottom=0.078,
                        hspace=0.28)

    for suffix, kwargs in ((".png", {"dpi": 200}), (".svg", {}), (".pdf", {})):
        target = out_stem.with_suffix(suffix)
        fig.savefig(target, **kwargs)
        print(f"wrote {target}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot runtime for the 13 shapes")
    parser.add_argument("--mode", choices=["light", "dark", "both"], default="both")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in (["light", "dark"] if args.mode == "both" else [args.mode]):
        stem = "runtime-by-shape" + ("" if mode == "light" else "-dark")
        draw(mode, out_dir / stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

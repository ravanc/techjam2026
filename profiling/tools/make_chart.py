#!/usr/bin/env python3
"""Draw the headline figure: where the speedup comes from.

The total speedup is a product, not a sum: CPU/MLX = (CPU/MPS) x (MPS/MLX).
On a log axis a product becomes a sum, so the two factors stack end to end and
the bar length stays true. That is the one form that shows the split honestly.

    .venv/bin/python3 profiling/tools/make_chart.py

It writes an SVG. Import the SVG into Canva or a document; it stays sharp at
every size.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "profiling" / "results" / "harness" / "speedup-by-shape.svg"

# case: (cpu ms, mlx ms) from the harness run, mps ms from the scoreboard run.
HARNESS: Dict[int, Tuple[float, float]] = {
    1: (45.9754, 3.2652), 2: (1.4456, 0.6104), 3: (4.0834, 0.6723),
    4: (13.9391, 1.2029), 5: (102.2755, 5.8422), 6: (15419.7160, 436.8107),
    7: (23.6896, 1.0029), 8: (461.1526, 115.5255), 9: (26.3907, 3.3803),
    10: (37.3346, 3.2754), 11: (121.8538, 3.2514), 12: (7.5305, 1.1551),
    13: (1845.8785, 37.4570),
}
MPS: Dict[int, float] = {
    1: 17.4401, 2: 1.5256, 3: 2.0156, 4: 4.7035, 5: 33.7087, 6: 2771.8542,
    7: 11.5201, 8: 167.5706, 9: 8.1035, 10: 13.0954, 11: 42.5131,
    12: 3.4665, 13: 577.3611,
}
# What each shape moves away from the base. The chart must read without the table.
LABEL: Dict[int, str] = {
    1: "base", 2: "batch 1", 3: "batch 4", 4: "batch 16", 5: "batch 128",
    6: "batch 10,000", 7: "dim 32", 8: "dim 1024", 9: "1 head", 10: "2 heads",
    11: "16 heads", 12: "seq 32", 13: "seq 1024",
}
# Share of the 1,800 GFLOP total. Only the two that decide a weighted score
# get called out, so the labels stay selective.
FLOP_SHARE: Dict[int, str] = {6: "66.5% of FLOPs", 8: "21.3%"}

# dataviz reference palette, categorical slots 1 and 2. Both modes pass every
# check of scripts/validate_palette.js: CVD dE 24.7 light, 26.8 dark.
LIGHT = {
    "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e",
    "faint": "#8a8880", "grid": "#e8e7e4", "rule": "#d7d5d0",
    "device": "#2a78d6", "kernel": "#eb6834",
}
DARK = {
    "surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7",
    "faint": "#8a8880", "grid": "#2a2a28", "rule": "#3a3a37",
    "device": "#3987e5", "kernel": "#d95926",
}
# For embedding in a page that owns the theme: the host stylesheet defines the
# values under all three theme scopes, so one SVG serves light and dark.
CSSVAR = {key: f"var(--viz-{key})" for key in LIGHT}

W = 940
# Thin marks: a 19px bar keeps a 600px fill from reading as a saturated block.
ROW_H, ROW_GAP = 19, 12
PAD_L, PAD_R = 196, 132
PAD_T, PAD_B = 172, 76
TICKS = [1, 2, 5, 10, 20, 50]
X_MIN, X_MAX = 0.86, 58.0


def rows() -> List[Tuple[int, float, float, float]]:
    out = []
    for case, (cpu, mlx) in HARNESS.items():
        mps = MPS[case]
        out.append((case, cpu / mps, mps / mlx, cpu / mlx))
    out.sort(key=lambda r: -r[3])
    return out


def rounded_end(x: float, y: float, width: float, height: float,
                radius: float, side: str) -> str:
    """A rect with only the outer end rounded, so the fills meet the baseline flat."""
    radius = max(0.0, min(radius, abs(width) / 2, height / 2))
    if width <= 0.4:
        return f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(width, 0.6):.2f}" height="{height:.2f}"/>'
    if side == "right":
        return (f'<path d="M{x:.2f},{y:.2f} H{x + width - radius:.2f} '
                f'a{radius:.2f},{radius:.2f} 0 0 1 {radius:.2f},{radius:.2f} '
                f'V{y + height - radius:.2f} '
                f'a{radius:.2f},{radius:.2f} 0 0 1 -{radius:.2f},{radius:.2f} '
                f'H{x:.2f} Z"/>')
    if side == "top":
        radius = max(0.0, min(radius, abs(width) / 2, abs(height) / 2))
        if height <= 0.4:
            return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
                    f'height="{max(height, 0.6):.2f}"/>')
        return (f'<path d="M{x:.2f},{y + height:.2f} V{y + radius:.2f} '
                f'a{radius:.2f},{radius:.2f} 0 0 1 {radius:.2f},-{radius:.2f} '
                f'H{x + width - radius:.2f} '
                f'a{radius:.2f},{radius:.2f} 0 0 1 {radius:.2f},{radius:.2f} '
                f'V{y + height:.2f} Z"/>')
    return (f'<path d="M{x + width:.2f},{y:.2f} H{x + radius:.2f} '
            f'a{radius:.2f},{radius:.2f} 0 0 0 -{radius:.2f},{radius:.2f} '
            f'V{y + height - radius:.2f} '
            f'a{radius:.2f},{radius:.2f} 0 0 0 {radius:.2f},{radius:.2f} '
            f'H{x + width:.2f} Z"/>')



def build_grouped(mode: str) -> str:
    """A grouped bar chart: two bars per shape, speedup against the CPU baseline.

    Raw milliseconds span 25,000x across the 13 shapes, so a linear axis would
    hide twelve of them and a log axis would break the bars: bar length must stay
    proportional, and a log scale has no zero. Speedup is the measure that is
    both linear and zero-based, so the bars stay honest.
    """
    c = {"light": LIGHT, "dark": DARK, "css": CSSVAR}[mode]
    order = sorted(HARNESS)
    width, height = 1020, 540
    pad_l, pad_r, pad_t, pad_b = 62, 22, 128, 96
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    y_max = 52.0
    group_w = plot_w / len(order)
    bar_w = min(26.0, group_w * 0.32)
    base = pad_t + plot_h

    def y_of(value: float) -> float:
        return base - (value / y_max) * plot_h

    parts = []
    add = parts.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">')
    if mode != "css":
        add(f'<rect width="{width}" height="{height}" fill="{c["surface"]}"/>')

    add(f'<text x="{pad_l}" y="42" font-size="24" font-weight="700" '
        f'fill="{c["text"]}">Speedup over the CPU baseline</text>')
    add(f'<text x="{pad_l}" y="66" font-size="13.5" fill="{c["muted"]}">'
        f'All 13 test shapes. Taller is faster. The CPU is 1&#215; by definition.</text>')

    lx = pad_l
    add(f'<rect x="{lx}" y="86" width="11" height="11" rx="2.5" fill="{c["device"]}"/>')
    add(f'<text x="{lx + 18}" y="95.5" font-size="12.5" fill="{c["muted"]}">'
        f'MPS &#8212; the same model on the GPU</text>')
    add(f'<rect x="{lx + 236}" y="86" width="11" height="11" rx="2.5" fill="{c["kernel"]}"/>')
    add(f'<text x="{lx + 254}" y="95.5" font-size="12.5" fill="{c["muted"]}">'
        f'MLX &#8212; the optimized model</text>')

    for tick in [0, 10, 20, 30, 40, 50]:
        gy = y_of(tick)
        add(f'<line x1="{pad_l}" y1="{gy:.2f}" x2="{pad_l + plot_w}" y2="{gy:.2f}" '
            f'stroke="{c["grid"]}" stroke-width="1"/>')
        add(f'<text x="{pad_l - 10}" y="{gy + 4:.2f}" font-size="11.5" '
            f'text-anchor="end" fill="{c["faint"]}">{tick}&#215;</text>')

    # The 1x rule is the whole point of the chart: below it, the change lost.
    one = y_of(1.0)
    add(f'<line x1="{pad_l}" y1="{one:.2f}" x2="{pad_l + plot_w}" y2="{one:.2f}" '
        f'stroke="{c["rule"]}" stroke-width="1"/>')

    for index, case in enumerate(order):
        cpu, mlx = HARNESS[case]
        mps_up, mlx_up = cpu / MPS[case], cpu / mlx
        cx = pad_l + index * group_w + group_w / 2
        x1, x2 = cx - bar_w - 2, cx + 2

        add(f'<g class="bar" data-case="{case}" data-label="{LABEL[case]}" '
            f'data-cpu="{cpu:.4f}" data-mps="{MPS[case]:.4f}" data-mlx="{mlx:.4f}" '
            f'data-mpsup="{mps_up:.2f}" data-mlxup="{mlx_up:.2f}">')
        for x, value, fill in ((x1, mps_up, c["device"]), (x2, mlx_up, c["kernel"])):
            top = y_of(value)
            add(f'<g fill="{fill}">'
                + rounded_end(x, top, bar_w, base - top, 3, "top") + '</g>')
            add(f'<text x="{x + bar_w / 2:.2f}" y="{top - 6:.2f}" font-size="10.5" '
                f'font-weight="600" text-anchor="middle" fill="{fill}">'
                f'{value:.1f}</text>')
        add('</g>')

        add(f'<text x="{cx:.2f}" y="{base + 20:.2f}" font-size="12" font-weight="700" '
            f'text-anchor="middle" fill="{c["text"]}">{case}</text>')
        add(f'<text x="{cx:.2f}" y="{base + 35:.2f}" font-size="10.5" '
            f'text-anchor="middle" fill="{c["muted"]}">{LABEL[case]}</text>')

    add(f'<text x="{pad_l}" y="{height - 22}" font-size="11" fill="{c["faint"]}">'
        f'Shape 2 is the one case where the GPU alone loses to the CPU (0.9&#215;): at '
        f'batch 1 there is too little work to cover dispatch.</text>')
    add('</svg>')
    return "\n".join(parts)

def build(mode: str) -> str:
    c = {"light": LIGHT, "dark": DARK, "css": CSSVAR}[mode]
    data = rows()
    plot_w = W - PAD_L - PAD_R
    height = PAD_T + len(data) * (ROW_H + ROW_GAP) - ROW_GAP + PAD_B

    lo, hi = math.log(X_MIN), math.log(X_MAX)

    def x_of(value: float) -> float:
        return PAD_L + (math.log(value) - lo) / (hi - lo) * plot_w

    origin = x_of(1.0)
    parts: List[str] = []
    add = parts.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'viewBox="0 0 {W} {height}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">')
    if mode != "css":
        add(f'<rect width="{W}" height="{height}" fill="{c["surface"]}"/>')

    # --- masthead -------------------------------------------------------
    add(f'<text x="{PAD_L}" y="46" font-size="25" font-weight="700" '
        f'fill="{c["text"]}">Where the speedup comes from</text>')
    add(f'<text x="{PAD_L}" y="70" font-size="13.5" fill="{c["muted"]}">'
        f'Each bar is one test shape. The two segments multiply to the total.</text>')

    # Hero number. The chart exists to make this one line believable.
    add(f'<text x="{PAD_L}" y="116" font-size="34" font-weight="700" '
        f'fill="{c["text"]}">11.59&#215;</text>')
    add(f'<text x="{PAD_L + 124}" y="116" font-size="15" fill="{c["muted"]}">'
        f'median across the 13 shapes</text>')
    add(f'<text x="{PAD_L}" y="138" font-size="14" fill="{c["muted"]}">'
        f'<tspan fill="{c["device"]}" font-weight="700">2.85&#215;</tspan> from the device'
        f'<tspan fill="{c["faint"]}">  &#215;  </tspan>'
        f'<tspan fill="{c["kernel"]}" font-weight="700">4.00&#215;</tspan> from the kernels'
        f'</text>')

    # --- legend ---------------------------------------------------------
    lx = W - PAD_R - 232
    add(f'<rect x="{lx}" y="96" width="11" height="11" rx="2.5" fill="{c["device"]}"/>')
    add(f'<text x="{lx + 18}" y="105.5" font-size="12.5" fill="{c["muted"]}">'
        f'Device &#8212; MPS vs CPU</text>')
    add(f'<rect x="{lx}" y="118" width="11" height="11" rx="2.5" fill="{c["kernel"]}"/>')
    add(f'<text x="{lx + 18}" y="127.5" font-size="12.5" fill="{c["muted"]}">'
        f'Kernels &#8212; MLX vs MPS</text>')

    plot_top = PAD_T - 18
    plot_bottom = plot_top + len(data) * (ROW_H + ROW_GAP) - ROW_GAP + 12

    # --- grid, drawn under the bars, one shade off the surface ----------
    for tick in TICKS:
        gx = x_of(tick)
        strong = tick == 1
        add(f'<line x1="{gx:.2f}" y1="{plot_top:.2f}" x2="{gx:.2f}" y2="{plot_bottom:.2f}" '
            f'stroke="{c["rule"] if strong else c["grid"]}" stroke-width="1"/>')
        add(f'<text x="{gx:.2f}" y="{plot_bottom + 20:.2f}" font-size="11.5" '
            f'text-anchor="middle" fill="{c["faint"]}">{tick}&#215;</text>')

    # --- bars -----------------------------------------------------------
    for index, (case, device, kernel, total) in enumerate(data):
        y = PAD_T - 18 + index * (ROW_H + ROW_GAP)
        x_dev = x_of(max(device, X_MIN))
        x_tot = x_of(total)

        add(f'<text x="{PAD_L - 76}" y="{y + ROW_H / 2 + 4.5:.2f}" font-size="13" '
            f'font-weight="600" text-anchor="end" fill="{c["text"]}">{case}</text>')
        add(f'<text x="{PAD_L - 68}" y="{y + ROW_H / 2 + 4.5:.2f}" font-size="12" '
            f'fill="{c["muted"]}">{LABEL[case]}</text>')

        tip = (f'data-case="{case}" data-label="{LABEL[case]}" '
               f'data-device="{device:.3f}" data-kernel="{kernel:.3f}" '
               f'data-total="{total:.3f}" data-cpu="{HARNESS[case][0]:.4f}" '
               f'data-mps="{MPS[case]:.4f}" data-mlx="{HARNESS[case][1]:.4f}"')
        add(f'<g class="bar" {tip}>')

        if device >= 1.0:
            # device runs origin -> x_dev, kernels carry on to the total.
            # A 2px surface gap separates the two fills.
            add(f'<g fill="{c["device"]}">'
                + rounded_end(origin, y, x_dev - origin, ROW_H, 4, "right") + '</g>')
            add(f'<g fill="{c["kernel"]}">'
                + rounded_end(x_dev + 2, y, x_tot - x_dev - 2, ROW_H, 4, "right") + '</g>')
        else:
            # Shape 2 only: MPS is slower than CPU, so the device factor is a
            # loss and its segment runs left of the 1x rule.
            add(f'<g fill="{c["kernel"]}">'
                + rounded_end(x_dev, y, x_tot - x_dev, ROW_H, 4, "right") + '</g>')
            add(f'<g fill="{c["device"]}">'
                + rounded_end(x_dev, y, origin - x_dev, ROW_H, 4, "left") + '</g>')
            add(f'<text x="{x_tot + 52:.2f}" y="{y + ROW_H / 2 + 4:.2f}" font-size="10.5" '
                f'fill="{c["device"]}">device 0.95&#215;, a loss</text>')

        add('</g>')
        add(f'<text x="{x_tot + 10:.2f}" y="{y + ROW_H / 2 + 4.5:.2f}" font-size="13" '
            f'font-weight="700" fill="{c["text"]}">{total:.1f}&#215;</text>')
        if case in FLOP_SHARE:
            add(f'<text x="{x_tot + 52:.2f}" y="{y + ROW_H / 2 + 4:.2f}" font-size="10.5" '
                f'fill="{c["faint"]}">{FLOP_SHARE[case]}</text>')

    # --- footnote -------------------------------------------------------
    foot_y = plot_bottom + 46
    add(f'<line x1="{PAD_L}" y1="{foot_y - 20:.2f}" x2="{W - PAD_R + 60}" '
        f'y2="{foot_y - 20:.2f}" stroke="{c["grid"]}" stroke-width="1"/>')
    add(f'<text x="{PAD_L}" y="{foot_y:.2f}" font-size="11" fill="{c["faint"]}">'
        f'Log scale, so the segments add. Shape 2 is the one case where the GPU '
        f'loses to the CPU: at batch 1 there is too little work to cover dispatch.</text>')

    add('</svg>')
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the speedup decomposition")
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--mode", choices=["light", "dark", "css", "both"],
                        default="both")
    parser.add_argument("--form", choices=["grouped", "decomposition"],
                        default="grouped",
                        help="grouped: two bars per shape. decomposition: the "
                             "log-scale split into device and kernels.")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    modes = ["light", "dark", "css"] if args.mode == "both" else [args.mode]

    suffix = {"light": "", "dark": "-dark", "css": "-themed"}
    for mode in modes:
        target = (out if mode == "light"
                  else out.with_name(out.stem + suffix[mode] + out.suffix))
        draw = build_grouped if args.form == 'grouped' else build
        target.write_text(draw(mode))
        print(f"wrote {target}")

    print("\nsegment check (device x kernel == total):")
    for case, device, kernel, total in rows():
        assert abs(device * kernel - total) < 1e-9
        print(f"  shape {case:>2}: {device:.3f} x {kernel:.3f} = {total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

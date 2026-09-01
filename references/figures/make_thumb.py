"""Make an opaque thumbnail from a transparent figure.

The two `kernel-dispatch` PNGs draw on a transparent background, so a viewer
that picks the wrong variant hides the text. A thumbnail slot does not let
the user pick, so it needs an opaque file. This script bakes the background
into the PNG, adds a margin, and draws a border.

Run it from this directory:

    ../../.venv/bin/python3 make_thumb.py
"""

from PIL import Image, ImageDraw

PAD = 90
BORDER = 3

# (source, output, background, border)
FIGURES = [
    ("kernel-dispatch-dark.png", "kernel-dispatch-thumb-dark.png",
     (13, 17, 23), (46, 56, 71)),
    ("kernel-dispatch.png", "kernel-dispatch-thumb-light.png",
     (255, 255, 255), (208, 213, 221)),
    ("kernel-dispatch-shape6-dark.png", "kernel-dispatch-shape6-thumb-dark.png",
     (13, 17, 23), (46, 56, 71)),
    ("kernel-dispatch-shape6.png", "kernel-dispatch-shape6-thumb-light.png",
     (255, 255, 255), (208, 213, 221)),
]


def make_card(src_path, out_path, ground, edge):
    src = Image.open(src_path).convert("RGBA")
    art = src.crop(src.getchannel("A").getbbox())
    out = Image.new("RGB", (art.width + 2 * PAD, art.height + 2 * PAD), ground)
    out.paste(art, (PAD, PAD), art)
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width - 1, out.height - 1], outline=edge, width=BORDER)
    out.save(out_path, optimize=True)
    print(out_path, out.size)


for figure in FIGURES:
    make_card(*figure)

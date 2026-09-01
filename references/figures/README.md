# Figures

Diagrams that explain the design. They are not measurements. Every number a
diagram shows must also appear in a reference file or in `OPTIMIZATIONS.md`,
and that text is the source of truth. If a diagram and a table disagree,
correct the diagram.

| File | Subject |
|---|---|
| `kernel-dispatch.png` | The `plan_kernels()` decision tree: which kernel each of the three stages takes, and the condition that selects it |
| `kernel-dispatch-shape6.png` | The same tree, resolved for shape 6, with the kernel that each step of one layer becomes |
| `kernel-dispatch-thumb-dark.png` | The dispatch tree on an opaque dark background, for a thumbnail |
| `kernel-dispatch-thumb-light.png` | The dispatch tree on an opaque white background, for a thumbnail |
| `kernel-dispatch-shape6-thumb-dark.png` | The shape 6 tree on an opaque dark background, for a thumbnail |
| `kernel-dispatch-shape6-thumb-light.png` | The shape 6 tree on an opaque white background, for a thumbnail |

## Two files for each diagram

Each diagram has a `-dark` variant. The background of both is transparent.

- `<name>.png` draws in dark ink, for a white page.
- `<name>-dark.png` draws in light ink, for a dark page.

A viewer that shows the plain file on a dark background hides most of the
text. That is the expected result, not a broken export. Pick the variant
that matches the page. In Markdown on GitHub, use both:

    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="kernel-dispatch-dark.png">
      <img src="kernel-dispatch.png" alt="the plan_kernels() dispatch tree">
    </picture>

## The thumbnails

A thumbnail slot does not let the viewer pick a variant, and it puts the
image on a page background that you do not control. So each `-thumb-` file
carries the background in the pixels. Every one has a margin of 90 px and a
border of 3 px.

| File | Ink | Background | Size |
|---|---|---|---|
| `kernel-dispatch-thumb-dark.png` | light | `#0d1117` | 3098 x 2052 |
| `kernel-dispatch-thumb-light.png` | dark | white | 3098 x 2052 |
| `kernel-dispatch-shape6-thumb-dark.png` | light | `#0d1117` | 3098 x 2244 |
| `kernel-dispatch-shape6-thumb-light.png` | dark | white | 3098 x 2244 |

Take the dark file first. It stays readable on a white page and on a dark
page. Take the light file when the page must look light.

`make_thumb.py` rebuilds all four from the transparent PNGs:

    ../../.venv/bin/python3 make_thumb.py

## There is no source file

**The four transparent PNGs are the only copy.** No `.tex`, `.svg` or other
source is in this repository, so no script can rebuild them. Git tracks them
for that reason. Do not delete one expecting to regenerate it. The
thumbnails are the exception: `make_thumb.py` rebuilds them.

To change a diagram you must redraw it. If you do, commit the source next to
the PNG, and delete this section.

The charts under `profiling/results/harness/` are the opposite case:
`profiling/tools/plot_runtime.py` rebuilds them from a sweep, so they are
reproducible.

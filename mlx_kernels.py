"""
Find the MLX Metal kernel headers, whatever the install layout.

WHY THIS EXISTS

`steel_gemm.py` and `steel_attention.py` inline MLX's own steel headers,
because `mx.fast.metal_kernel` cannot `#include` them. MLX embeds a fixed
header set in the binary, and the steel headers are not in it. So both
modules read the headers off disk.

Both modules held one fixed path:

    <repo>/.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels

That path makes three assumptions. The venv carries the name `.venv`. The
venv sits in the repo. The venv runs Python 3.13. Break one assumption, and
`_read()` raised `FileNotFoundError` at the FIRST kernel build, which is
deep inside a timed run.

This module removes the three assumptions, and it degrades instead of
raising. `choose_tile()` returns None and `supports()` returns False when
the headers are absent. The model then takes the plain MLX path, and the
run completes with a slower number instead of a traceback.

HOW IT SEARCHES

1. `MLX_KERNELS_DIR`, when the user sets it. This overrides everything.
   Point it at the `kernels` directory itself.
2. The installed `mlx` package, through `mlx.__path__[0]`. This handles a
   venv at any place, a conda environment and a system install.
   Do NOT use `mlx.__file__`. It is None on this install, because `mlx` is
   a namespace package.
3. A `.venv` in the repo, at any Python version, through a glob. This is
   the fallback for a `mlx` package that reports no usable path.

Each candidate must hold `steel/defines.h`. A directory that does not hold
it is not the kernels directory, so the search continues.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

# The path from the package root to the kernels, inside an MLX install.
_INCLUDE_TAIL = os.path.join("include", "mlx", "backend", "metal", "kernels")

# The file that proves a directory is the kernels directory. Both
# `steel_gemm.py` and `steel_attention.py` read it first in their chains.
_SENTINEL = os.path.join("steel", "defines.h")

_REPO = os.path.dirname(os.path.abspath(__file__))


def _valid(path: Optional[str]) -> bool:
    """Return True when `path` holds the steel headers."""
    return bool(path) and os.path.isfile(os.path.join(path, _SENTINEL))


def _candidates() -> List[Tuple[str, str]]:
    """Return every path to try, with the source that produced it."""
    found: List[Tuple[str, str]] = []

    override = os.environ.get("MLX_KERNELS_DIR")
    if override:
        found.append((override, "MLX_KERNELS_DIR"))

    try:
        import mlx  # noqa: PLC0415

        for root in list(getattr(mlx, "__path__", [])):
            found.append((os.path.join(root, _INCLUDE_TAIL), "mlx.__path__"))
    except ImportError:
        pass

    pattern = os.path.join(
        _REPO, ".venv", "lib", "python*", "site-packages", "mlx", _INCLUDE_TAIL
    )
    for root in sorted(glob.glob(pattern)):
        found.append((root, "repo .venv"))

    return found


def find_kernels() -> Optional[str]:
    """Return the kernels directory, or None when no candidate holds it."""
    for path, _source in _candidates():
        if _valid(path):
            return os.path.abspath(path)
    return None


KERNELS: Optional[str] = find_kernels()


def available() -> bool:
    """Return True when the steel headers are on disk."""
    return KERNELS is not None


def why_missing() -> str:
    """Return one line that names every path this module tried."""
    if KERNELS is not None:
        return f"the MLX kernel headers are at {KERNELS}"
    tried = [f"{path} (from {source})" for path, source in _candidates()]
    if not tried:
        tried = ["nothing; the mlx package did not import"]
    return (
        "this module did not find the MLX Metal kernel headers. It looked "
        "for " + _SENTINEL + " in: " + "; ".join(tried) + ". Install mlx in "
        "the active environment, or set MLX_KERNELS_DIR to the kernels "
        "directory."
    )


def read(rel: str) -> str:
    """Read one header. Raise a message that names the search when absent."""
    if KERNELS is None:
        raise FileNotFoundError(why_missing())
    return open(os.path.join(KERNELS, rel)).read()

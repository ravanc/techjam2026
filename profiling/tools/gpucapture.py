"""GPU trace capture helpers for MLX and PyTorch/MPS.

Produces .gputrace bundles that open in Xcode's Metal debugger, where you get
per-kernel timing, occupancy, and the shader profiler.

Two platform quirks are handled here, both of which produce confusing failures
if you hit them by hand:

1. MTL_CAPTURE_ENABLED=1 must be present in the environment *before* Metal
   initializes. Setting it from Python after the framework has loaded is too
   late, so `ensure_capture_env` re-execs the interpreter with it set.

2. PyTorch's `is_metal_capture_enabled()` reports False until the MPS backend
   has actually been initialized, because the check depends on a live Metal
   device. Querying it before touching MPS makes capture look unsupported when
   it is merely uninitialized. `mps_capture` forces initialization first.
"""

from __future__ import annotations

import glob
import os
import sys
from contextlib import contextmanager

CAPTURE_ENV = "MTL_CAPTURE_ENABLED"


def capture_env_ready() -> bool:
    """True when the process was started with Metal capture enabled."""
    return os.environ.get(CAPTURE_ENV) == "1"


def ensure_capture_env() -> None:
    """Re-exec this process with MTL_CAPTURE_ENABLED=1 if it is not set.

    Metal reads the variable when it initializes, so it cannot be set usefully
    from inside a process that has already loaded the framework.
    """
    if capture_env_ready():
        return
    env = dict(os.environ)
    env[CAPTURE_ENV] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


def _require_env(backend: str) -> None:
    if not capture_env_ready():
        raise RuntimeError(
            f"{backend} GPU capture needs {CAPTURE_ENV}=1 set before the process "
            f"starts. Call gpucapture.ensure_capture_env() at the top of your "
            f"script, or run it as `{CAPTURE_ENV}=1 python ...`."
        )


def _prepare_path(path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        # Metal refuses to overwrite an existing trace document.
        import shutil

        shutil.rmtree(path, ignore_errors=True)
    return path


@contextmanager
def mlx_capture(path: str):
    """Capture MLX GPU work into a .gputrace bundle."""
    import mlx.core as mx

    _require_env("MLX")
    path = _prepare_path(path)
    mx.metal.start_capture(path)
    try:
        yield
    finally:
        # Pending work must be flushed or it lands outside the capture.
        mx.synchronize()
        mx.metal.stop_capture()


@contextmanager
def mps_capture(path: str):
    """Capture PyTorch MPS GPU work into a .gputrace bundle."""
    import torch

    _require_env("MPS")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend is not available on this machine")

    # Force MPS initialization before querying capture support; the check
    # depends on a live Metal device and reports False otherwise.
    torch.zeros(1, device="mps")
    torch.mps.synchronize()

    if not torch.mps.profiler.is_metal_capture_enabled():
        raise RuntimeError("torch reports Metal capture unavailable after MPS init")

    path = _prepare_path(path)

    # torch builds the trace name as f"{counter:04d}-{fname}.gputrace". Handing
    # it an absolute path therefore creates a nested tree under "./0000-/"
    # instead of the file you asked for, so capture into the target directory
    # under a bare stem and rename the result afterwards.
    outdir = os.path.dirname(path)
    stem = os.path.basename(path)
    if stem.endswith(".gputrace"):
        stem = stem[: -len(".gputrace")]

    prev_cwd = os.getcwd()
    os.chdir(outdir)
    try:
        with torch.mps.profiler.metal_capture(stem):
            yield
            torch.mps.synchronize()
    finally:
        os.chdir(prev_cwd)

    produced = sorted(
        p for p in glob.glob(os.path.join(outdir, f"*-{stem}.gputrace"))
    )
    if not produced:
        raise RuntimeError(f"MPS capture produced no trace for {stem!r} in {outdir}")
    os.rename(produced[-1], path)

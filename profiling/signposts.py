"""os_signpost bridge so Python-side regions show up in Instruments.

Regions are emitted into the "Points of Interest" category, which Instruments
graphs alongside the Metal/GPU tracks. Without them a GPU trace shows a wall of
kernels with no indication of which model or layer produced them.

The actual signpost calls live in a small C shim (src/signpost_shim.c) because
os_signpost_* are macros requiring compile-time format strings; calling the
underlying libSystem entry point through ctypes traps. Build the shim with:

    make -C profiling

If the shim is missing this module degrades to no-ops, so instrumented code
stays runnable when not profiling.
"""

from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "libtechjam_signpost.dylib")

_lib = None
try:
    _lib = ctypes.CDLL(_LIB_PATH)
    _lib.tj_signpost_enabled.restype = ctypes.c_int
    _lib.tj_signpost_enabled.argtypes = []
    _lib.tj_signpost_id.restype = ctypes.c_uint64
    _lib.tj_signpost_id.argtypes = []
    _lib.tj_interval_begin.restype = None
    _lib.tj_interval_begin.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
    _lib.tj_interval_end.restype = None
    _lib.tj_interval_end.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
    _lib.tj_event.restype = None
    _lib.tj_event.argtypes = [ctypes.c_char_p]
except OSError:
    _lib = None


def available() -> bool:
    """True when the shim is loaded and signposts will reach Instruments."""
    return _lib is not None and bool(_lib.tj_signpost_enabled())


def event(name: str) -> None:
    """Emit a single point-in-time marker."""
    if _lib is not None:
        _lib.tj_event(name.encode())


@contextmanager
def interval(name: str):
    """Bracket a region as a named interval in Instruments."""
    if _lib is None:
        yield
        return
    spid = _lib.tj_signpost_id()
    raw = name.encode()
    _lib.tj_interval_begin(spid, raw)
    try:
        yield
    finally:
        _lib.tj_interval_end(spid, raw)

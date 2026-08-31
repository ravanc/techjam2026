#!/usr/bin/env bash
# Record a Metal System Trace of the MLX model alone, for one appendix shape.
#
# Read the result with:
#   .venv/bin/python3 profiling/tools/gpu_timeline.py report profiling/traces/gpu_timeline.trace
#
# Usage:
#   ./profiling/tools/gpu_timeline.sh [--case 6] [--iterations 3]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python3"
OUT="${OUT:-$ROOT/profiling/traces/gpu_timeline.trace}"

if [ ! -x "$PY" ]; then
  echo "error: $PY not found; create the venv first" >&2
  exit 1
fi

if [ ! -f "$ROOT/profiling/tools/libtechjam_signpost.dylib" ]; then
  echo "building signpost shim..." >&2
  make -C "$ROOT/profiling/tools"
fi

mkdir -p "$(dirname "$OUT")"
rm -rf "$OUT"

exec xcrun xctrace record \
  --template "Metal System Trace" \
  --instrument "os_signpost" \
  --instrument "Points of Interest" \
  --output "$OUT" \
  --no-prompt \
  --launch -- "$PY" "$ROOT/profiling/tools/gpu_timeline.py" record "$@"

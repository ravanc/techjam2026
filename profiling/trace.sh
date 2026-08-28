#!/usr/bin/env bash
# Record an Instruments trace of the transformer benchmark.
#
# The stock "Metal System Trace" template does not record user os_signpost
# subsystems, so the signpost instruments are added explicitly; without them
# the trace contains Apple's Metal signposts but none of our region labels.
#
# Usage:
#   ./profiling/trace.sh [output.trace] [-- extra args for profile_benchmark.py]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python3"
TEMPLATE="${TEMPLATE:-Metal System Trace}"
OUT="${1:-$ROOT/profiling/traces/benchmark.trace}"
shift || true
[ "${1:-}" = "--" ] && shift

if [ ! -x "$PY" ]; then
  echo "error: $PY not found; create the venv first" >&2
  exit 1
fi

if [ ! -f "$ROOT/profiling/libtechjam_signpost.dylib" ]; then
  echo "building signpost shim..." >&2
  make -C "$ROOT/profiling"
fi

mkdir -p "$(dirname "$OUT")"
rm -rf "$OUT"

exec xcrun xctrace record \
  --template "$TEMPLATE" \
  --instrument "os_signpost" \
  --instrument "Points of Interest" \
  --output "$OUT" \
  --no-prompt \
  --launch -- "$PY" "$ROOT/profiling/profile_benchmark.py" --mode signpost "$@"

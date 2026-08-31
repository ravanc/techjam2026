# Hardware profiling for MLX / MPS

Instruments + Metal debugger tooling for `torch_transformer_benchmark.py`, so
GPU work can be attributed to a specific model and kernel.

To learn how to use these tools for optimization, read [WORKFLOW.md](WORKFLOW.md).
This file covers the setup and the platform problems.

## The four directories

| Directory | Holds | Git |
|---|---|---|
| `probes/` | One script for one question. `OPTIMIZATIONS.md` names the probe that decided each row | tracked |
| `tools/` | Instruments and Metal capture, the signpost shim, and the trace readers | tracked, except the built `.dylib` |
| `results/` | The JSON and JSONL that a run writes | tracked, except `cpu_cache.json` |
| `traces/` | Recorded Instruments and Metal traces. One reaches 500 MB | ignored |

Add a new probe to `probes/`. Add a new capture or reader to `tools/`.

## Requirements

Already satisfied on this machine: Xcode 16.4, Instruments, and the Metal
toolchain (`xcrun metal`). The venv provides torch 2.13 (MPS) and mlx 0.32.

Build the signpost shim once:

    make -C profiling/tools

Recording a locally-launched process works without enabling developer mode.
Only attaching to an already-running process needs it:

    sudo DevToolsSecurity -enable

## 1. Instruments timeline (which model, which kernel)

    ./profiling/tools/trace.sh                      # -> profiling/traces/benchmark.trace
    ./profiling/tools/trace.sh out.trace -- --iterations 50
    open profiling/traces/benchmark.trace

This records the **Metal System Trace** template plus the `os_signpost` and
`Points of Interest` instruments. The extra instruments matter: the stock Metal
template does not record user signpost subsystems, so without them the trace
contains Apple's own Metal signposts and none of the `baseline-*` /
`optimized-*` region labels.

Read it without the GUI:

    .venv/bin/python3 profiling/tools/summarize.py profiling/traces/benchmark.trace

    region                      n   median ms    mean ms    min ms    max ms
    ------------------------------------------------------------------------
    baseline-iter              10     20.0516    20.0538   19.4703   21.0130
    optimized-iter             10     15.4604    15.7413   14.7966   18.5760

Use `TEMPLATE="Time Profiler" ./profiling/tools/trace.sh` for CPU-side work, or
`TEMPLATE="Metal System Trace"` (default) for GPU scheduling and occupancy.

## 2. Metal GPU captures (per-kernel timings)

    .venv/bin/python3 profiling/tools/profile_benchmark.py --mode gputrace --iterations 3
    open profiling/traces/optimized_mlx.gputrace

Writes one `.gputrace` per backend for the Xcode Metal debugger: per-kernel
timing, occupancy, and the shader profiler.

Keep `--iterations` small — each capture is roughly 500 MB for 3 iterations,
since Metal archives buffer contents. `profiling/traces/` is gitignored.

## Instrumenting your own code

```python
import sys; sys.path.insert(0, "profiling")
import signposts, gpucapture

with signposts.interval("attention"):
    ...                       # shows as a named interval in Instruments

gpucapture.ensure_capture_env()          # must run before Metal initializes
with gpucapture.mlx_capture("out.gputrace"):
    ...
```

## Platform quirks handled here

These all produce misleading failures if you hit them by hand:

- **`MTL_CAPTURE_ENABLED=1` must be set before Metal initializes.** Setting it
  from Python after import is too late; `ensure_capture_env()` re-execs the
  interpreter with it set.
- **`torch.mps.profiler.is_metal_capture_enabled()` returns False until MPS is
  initialized**, because the check needs a live Metal device. Querying it first
  thing makes capture look unsupported when it is merely uninitialized.
  `mps_capture` forces initialization before checking.
- **torch names captures `f"{counter:04d}-{fname}.gputrace"`.** An absolute path
  therefore creates a nested tree under `./0000-/` rather than the file you
  asked for, so `mps_capture` captures under a bare stem and renames.
- **`os_signpost` cannot be called through ctypes.** It is a macro requiring a
  compile-time format string in the image's `__TEXT,__os_log` section; calling
  the libSystem entry point with a Python string traps with SIGTRAP. Hence the
  C shim in `src/`.

## Caveat: the benchmark under-reports MPS baseline latency

`benchmark_once` in `torch_transformer_benchmark.py` uses timing events only on
CUDA and never synchronizes on MPS. Because torch's MPS backend is
asynchronous, it times *enqueue* for the torch baseline while the MLX path
blocks on completion (it converts back through numpy). Measured here:

    baseline enqueue-only (what the benchmark reports)  2.84 ms
    baseline with torch.mps.synchronize()              19.54 ms
    optimized                                          14.83 ms

So the reported `0.129x` MPS speedup is an artifact; with both sides
synchronized the MLX path is roughly 1.3x faster. The profiling scripts here
synchronize explicitly (`_sync`), which is why their numbers differ.

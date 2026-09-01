# Optimizing Transformers on Apple Silicon

This project optimizes the forward pass of the provided `BaselineTransformer` for a consumer-grade Apple Silicon GPU.

The final implementation runs through MLX while keeping the PyTorch interface expected by the benchmark harness. Rather than using one implementation for every input, it builds a shape-aware `KernelPlan` that selects between MLX operations and custom Metal kernels based on the workload.

Across the 13 runnable benchmark shapes, our implementation is **27.80x faster than the PyTorch CPU baseline** and **5.72x faster than the same PyTorch baseline running on MPS**. All 13 shapes pass the provided float32 accuracy checks with zero failed elements.

| Implementation | Total time (13 shapes) | vs. CPU |
|---|---:|---:|
| PyTorch CPU | 17,763.7 ms | 1.00x |
| PyTorch MPS | 3,654.9 ms | 4.86x |
| **Our MLX implementation** | **639.0 ms** | **27.80x** |

We also run test case 14 separately. The provided PyTorch attention implementation cannot run this shape because it materializes an attention score matrix that would require approximately 18.6 TiB. Our implementation avoids materializing this matrix and completes the case in approximately **8.41 minutes**.

## How we approached it

We treated optimization as an iterative process:

```text
Profile
  ↓
Identify bottleneck
  ↓
Form hypothesis
  ↓
Implement / microbenchmark
  ↓
Test correctness
  ↓
Benchmark across shapes
  ↓
Keep / revert
  │
  └──────────────→ Repeat
```

We used roofline analysis and targeted microbenchmarks to determine whether individual stages were limited by compute, memory traffic, or launch overhead. This mattered because the appropriate optimization depends on the bottleneck: a memory-bound operation, for example, generally benefits more from removing memory passes or fusing operations than from reducing arithmetic.

We tested over 40 optimization hypotheses during development. The optimizations that survived include:

- shape-aware kernel selection
- custom builds of Apple's Steel attention and GEMM kernels
- single-pass LayerNorm
- fused QKV projection
- GEMM epilogue fusion
- removal of unnecessary tensor copies and memory passes
- deferred residual biases
- LayerNorm and row-statistic fusion
- batch chunking for very large inputs
- zero-copy output from MLX memory where possible

The important part is that these optimizations are **not applied blindly**. Different input shapes behave very differently on the GPU, so `plan_kernels()` chooses an appropriate path before execution.

For example, shape 6 has `head_dim=32`, where MLX does not normally dispatch to its fused attention kernel. We compile Apple's Steel attention kernel at this width instead. Shape 8 has `head_dim=256`, where the Steel configurations we tested either exceed the GPU's threadgroup-memory limit or are slower than the existing MLX path, so the planner deliberately falls back to MLX.

The complete history of what we tried, including optimizations that failed, is in [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md).

## Setup

### Requirements

This project targets Apple Silicon.

We developed and measured it on:

| | |
|---|---|
| Machine | MacBook Pro, Apple M3 Pro |
| GPU | 14-core integrated Apple GPU |
| Memory | 18 GiB unified memory |
| Python | 3.13 |
| MLX | 0.32.2 |
| PyTorch | 2.13.0 |

Python 3.10 or later should work.

### Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/ravanc/techjam2026
cd techjam2026

python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

Run the quick path tests to check that the installation and custom kernels work:

```bash
.venv/bin/python3 tests/test_paths.py
```

The custom kernels use Metal headers included with MLX. If these headers cannot be found, the custom paths disable themselves and the model falls back to standard MLX operations rather than failing.

## Reproducing the results

### 1. Check correctness and kernel dispatch

```bash
.venv/bin/python3 tests/test_paths.py
```

This checks that the benchmark shapes select the expected kernel plans and tests the individual custom kernel paths.

For padded and ragged inputs:

```bash
.venv/bin/python3 tests/test_padding.py
```

This runs 18 additional cases, including padded batches, ragged batches, and an empty sample.

### 2. Run the benchmark demo

For a visual run of the benchmark:

```bash
.venv/bin/python3 demo.py
```

`demo.py` runs the benchmark while displaying the progress and elapsed time on screen, making it the easiest way to see the optimized implementation running.


### 3. Run the original benchmark harness

```bash
.venv/bin/python3 torch_transformer_benchmark.py
```

This runs the provided benchmark harness and compares the optimized implementation against `BaselineTransformer` for correctness and performance.

### 4. Run test case 14

```bash
.venv/bin/python3 shape14_harness.py
```

This takes approximately 8.5 minutes on our M3 Pro.

Test case 14 is kept separate because the provided PyTorch baseline cannot allocate its attention score matrix, so there is no meaningful baseline runtime or speedup to report.

### Benchmarking note

Avoid running other GPU-heavy applications or benchmark processes at the same time. Since this is a personal laptop rather than a dedicated compute node, other processes can noticeably affect benchmark results.

Short benchmark runs can also produce misleading medians, so the default repeat counts should be used when reporting performance.

## Repository layout

The benchmark harness imports the kernel modules by their plain name, so those
files stay at the top level.

| Path | Holds |
|---|---|
| `torch_transformer_benchmark.py` | The baseline model, the optimized model, and the provided harness |
| `steel_attention.py`, `steel_gemm.py`, `fast_layernorm.py`, `mlx_kernels.py` | The custom Metal kernels and the MLX bindings |
| `appendix_cases.py`, `bench_cases.py`, `test_backends.py`, `flops.py` | The shapes, the inputs, the three backends, and the FLOP model |
| `demo.py`, `scoreboard.py`, `shape14_harness.py` | The entry points: the visual run, the graded sweep, and shape 14 |
| `tests/` | The correctness tests: the kernel plans, and the padded batch |
| `docs/` | The optimization loop, and the chat logs of the project |
| `references/` | Measured facts: the shapes, the machine, the MLX kernels, and the figures |
| `profiling/` | The probes, the Instruments tools, and every recorded result |
| `OPTIMIZATIONS.md` | Every optimization tried, kept or reverted, with its number |

## Limitations

### The kernel plan is tuned for one machine

The dispatch thresholds and tile choices were measured on an M3 Pro with 14 GPU cores. The implementation remains correct on other compatible Apple Silicon machines, but the choices we found for this GPU will not necessarily be optimal on an M4, M4 Max, or another Apple GPU.

### The project optimizes the float32 forward pass

The benchmark's accuracy requirements prevent us from using the usual reduced-precision optimizations. The project therefore focuses on float32 and does not implement training, a backward pass, optimizer operations, or KV caching.

### Benchmarking on a personal laptop is noisy

Unlike a dedicated compute machine, the CPU and GPU are shared with browsers, the desktop environment, and other applications. We therefore use repeated measurements and MPS as an additional control when evaluating small improvements.

### AI still required human steering

We used Claude Code throughout the project to investigate bottlenecks, write kernels, build profiling tools, and test optimization hypotheses.

The biggest limitation was that a plausible AI explanation was not necessarily the correct one.

One example was Q, K and V layout. We initially suspected that strided memory access was slowing attention and tried making the tensors contiguous. The improvement was effectively noise. Further profiling revealed that the custom Steel kernel was already making hidden contiguous copies internally. The eventual optimization was the opposite: remove those copies and pass the real tensor strides directly to the kernel.

This shaped our workflow: **AI suggestions were hypotheses, not conclusions**. An optimization was only kept after profiling supported its mechanism, the correctness tests passed, and controlled benchmarks showed an actual improvement.

## Future Developments

With more time, we would push this towards autonomous kernel optimization: specialized agents could independently investigate memory layout, fusion, mathematical transformations, and GPU-specific kernels, while a deterministic benchmark and correctness harness decides what gets kept. The optimization log would act as shared memory so that agents do not repeatedly rediscover approaches that have already failed.

We would also replace the fixed dispatch thresholds and tile choices with an autotuner. The autotuner benchmarks the candidate kernels on first use, and caches the resulting plan for the current GPU.

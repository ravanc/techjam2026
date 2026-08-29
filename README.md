# Transformer forward pass, optimized for Apple silicon

`UserOptimizedTransformer` runs the forward pass of `BaselineTransformer` in
MLX on the GPU, behind the torch interface the harness expects. The baseline
does not change. It is the reference for both accuracy and speed.

Run the graded sweep:

    .venv/bin/python3 scoreboard.py --label "what changed"

Read [CLAUDE.md](CLAUDE.md) before you change anything. It holds the rules
for a measurement, and the list of reference files to read first.

## Result

All 13 runnable Appendix 3.7 shapes, float32. `MLX` is
`UserOptimizedTransformer`. `CPU` is the torch baseline, and it is the
reference the harness uses. `MPS` is the same torch baseline on the GPU,
through Metal.

| # | Shape | CPU ms | MPS ms | MLX ms | MPS vs CPU | **MLX vs CPU** | MLX vs MPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 61.252 | 17.943 | 5.502 | 3.41x | **11.13x** | 3.26x |
| 2 | B1 D128 H4 S128 | 2.571 | 1.889 | 0.650 | 1.36x | **3.96x** | 2.91x |
| 3 | B4 D128 H4 S128 | 7.169 | 2.217 | 0.877 | 3.23x | **8.17x** | 2.53x |
| 4 | B16 D128 H4 S128 | 19.373 | 4.887 | 1.764 | 3.96x | **10.98x** | 2.77x |
| 5 | B128 D128 H4 S128 | 139.502 | 35.015 | 11.264 | 3.98x | **12.38x** | 3.11x |
| 6 | B10000 D128 H4 S128 | 14639.960 | 2693.553 | 852.760 | 5.44x | **17.17x** | 3.16x |
| 7 | B64 D32 H4 S128 | 25.959 | 11.533 | 1.395 | 2.25x | **18.60x** | 8.27x |
| 8 | B64 D1024 H4 S128 | 470.854 | 166.595 | 128.377 | 2.83x | **3.67x** | 1.30x |
| 9 | B64 D128 H1 S128 | 35.099 | 7.904 | 4.438 | 4.44x | **7.91x** | 1.78x |
| 10 | B64 D128 H2 S128 | 46.106 | 13.077 | 4.346 | 3.53x | **10.61x** | 3.01x |
| 11 | B64 D128 H16 S128 | 130.289 | 42.589 | 5.342 | 3.06x | **24.39x** | 7.97x |
| 12 | B64 D128 H4 S32 | 12.224 | 3.220 | 1.481 | 3.80x | **8.25x** | 2.17x |
| 13 | B64 D128 H4 S1024 | 1899.325 | 577.590 | 59.429 | 3.29x | **31.96x** | 9.72x |

| Metric | Value |
|---|---:|
| Median MLX speedup over CPU | **10.98x** |
| Range | 3.67x to 31.96x |
| Median MLX rate | 1.561 TFLOP/s |
| Best MLX rate | 3.346 TFLOP/s (shape 8) |

Accuracy: all 13 shapes PASS at `atol=0.002`, `rtol=0.02`, against the CPU
baseline. `max_abs` runs 9.54e-07 to 2.65e-06.

**Read the MPS column, not only the CPU column.** The CPU baseline moves with
machine load and chip temperature. It drifted up to 45.9% between two sweeps
on an unchanged baseline. MPS is stable to a few percent, so it is the better
control when you compare two builds.

Shape 14 (B32 D1024 H16 S100000 L2) is disabled. Its input alone is 12.2 GiB
and the machine holds 12.0 GiB, so no backend runs it here.

## Current bottlenecks

Measured with `profiling/stage_roofline.py` at shape 6, which carries 66.5%
of the FLOP weight. One layer, one chunk of 1024 rows, real layer time
20.71 ms. The roofs are 4.06 TFLOP/s and 128 GB/s, both measured here.

| Stage | ms | share | Limit | Against its own roof | Headroom |
|---|---:|---:|---|---:|---|
| sdpa (attention) | 5.57 | 27% | IO | 38% of bandwidth | **2.50x. Row 32** |
| qkv proj | 3.55 | 17% | COMPUTE | 89% of matmul peak | little |
| ffn_in + gelu | 2.34 | 11% | COMPUTE | 45% of matmul peak | **0.97 ms. Row 33** |
| residual add x2 | 3.37 | 16% | IO | 93% of bandwidth | none |
| ln1 + ln2 | 2.24 | 11% | IO | 94% of bandwidth | none. Row 31 took it |
| out proj | 1.32 | 6% | COMPUTE | 80% of matmul peak | little |
| ffn_out | 1.32 | 6% | COMPUTE | 80% of matmul peak | little |
| merge heads | 1.10 | 5% | IO | 96% of bandwidth | none |
| split+transpose | free | — | — | a strided view, not a copy | — |

Shape 6 is **67% IO-bound**. Four open items remain:

| # | Bottleneck | Size | Why it is still open |
|---:|---|---|---|
| 32 | The attention kernel reads q, k and v with a stride | shape 6 sdpa is 5.58 ms strided against 2.24 ms contiguous, **2.50x** | A `mx.contiguous` first does not pay: the copy costs 3.43 ms and saves 3.34 ms. The win needs the QKV projection to write the head layout directly, or a kernel that reads the stride well |
| 33 | GELU runs as a separate memory pass | 0.97 ms for each layer and chunk, **4.5%** of shape 6 | The GELU kernel is efficient at 115 GB/s. The extra pass is the cost. `mx.compile` does not fuse it, and MLX exposes no matmul epilogue |
| 21, 26 | MLX never calls its own `bd192` and `bd256` attention kernels | shape 8, 21.3% of the FLOP weight | `head_dim` 256 takes the fallback. `head_dim` cannot pad down, and a head cannot split. The threadgroup memory for `bd256` exceeds the 32 KiB limit |
| — | Small shapes are launch-bound | shapes 2, 3, 7 and 12 | Under 0.2% of the FLOP weight together. Not worth the effort |

Shape 8 is the exception to all of this. It is genuinely compute-bound: its
QKV projection runs at **99.5% of the matmul peak** at 351 FLOP/byte. Little
is left in it except the fallback attention kernel.

## What is in the model

The full log is [OPTIMIZATIONS.md](OPTIMIZATIONS.md). Its source of truth
table is the only place that states the status of an optimization. The
largest wins:

| # | Optimization | Effect |
|---:|---|---|
| 1 | MLX behind the torch interface | 4.4x to 7.4x against torch CPU |
| 25 | Hoist MLX's `steel_attention` and compile it at an unshipped `head_dim` | **1.308x** FLOP-weighted |
| 31 | A single-pass LayerNorm kernel for a row width under 256 | **1.205x** FLOP-weighted |
| 29 | `mx.addmm` for every projection, so the GPU adds the bias inside the matmul | 1.096x FLOP-weighted |
| 7 | A shape-aware kernel plan (`KernelPlan`) | 1.57x at shape 13 |
| 10 | Batch chunking, full depth for each chunk | peak 9.16 GiB to 2.68 GiB |
| 23 | Return the output as a view of MLX memory, not a copy | 71.6 ms of 1590.2 ms at shape 6 |

Two of these are custom Metal kernels: `steel_attention.py` and
`fast_layernorm.py`.

## Layout

| File | Role |
|---|---|
| `torch_transformer_benchmark.py` | The baseline model, the MLX model, and the harness |
| `steel_attention.py` | MLX's flash attention kernel, compiled at a `head_dim` MLX does not ship |
| `fast_layernorm.py` | A single-pass LayerNorm kernel for a row width under 256 |
| `scoreboard.py` | The graded run over all 13 shapes |
| `flops.py` | The FLOP model and the measured matmul rates |
| `appendix_cases.py` | The 14 shapes as code |
| `bench_cases.py` | Deterministic input generation, shared by every backend |
| `test_backends.py` | Cross-backend comparison: CPU, MPS, MLX |
| `test_padding.py` | Padded and ragged batches, including an empty sample |
| `profiling/stage_roofline.py` | Splits one block into stages and names each limit |
| `profiling/sdpa_dispatch.py` | Finds which `head_dim` values reach the fused SDPA kernel |
| `references/` | Measured facts: the machine, the shapes, the MLX kernels, the scoreboard |

## Reproduce

    .venv/bin/python3 torch_transformer_benchmark.py     # the harness
    .venv/bin/python3 test_padding.py                    # padded batches
    .venv/bin/python3 scoreboard.py --label "..."        # the full sweep
    .venv/bin/python3 profiling/stage_roofline.py --shapes 6

Check that no other run holds the GPU before you measure. Two runs share one
GPU, and each one makes the other reading false:

    ps -Ao pid,etime,%cpu,command | grep "[.]venv/bin/python3" | grep -v shell-snapshots

Machine: Apple M3 Pro, 14 GPU cores, 18 GiB unified memory, macOS 24.6,
Python 3.13.5, torch 2.13.0, mlx 0.32.2. See
[references/machine.md](references/machine.md).

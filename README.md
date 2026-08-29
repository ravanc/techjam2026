# Transformer forward pass, optimized for Apple silicon

`UserOptimizedTransformer` runs the forward pass of `BaselineTransformer` in
MLX on the GPU, behind the torch interface the harness expects. The baseline
does not change. It is the reference for both accuracy and speed.

Run the graded sweep:

    .venv/bin/python3 scoreboard.py --cpu-cache --label "what changed"

Read [CLAUDE.md](CLAUDE.md) before you change anything. It holds the rules
for a measurement, and the list of reference files to read first.

## Result

All 13 runnable Appendix 3.7 shapes, float32. `MLX` is
`UserOptimizedTransformer`. `CPU` is the torch baseline, and it is the
reference the harness uses. `MPS` is the same torch baseline on the GPU,
through Metal.

| # | Shape | CPU ms † | MPS ms | MLX ms | MPS vs CPU | **MLX vs CPU** | MLX vs MPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 61.252 | 17.309 | 3.788 | 3.54x | **16.17x** | 4.57x |
| 2 | B1 D128 H4 S128 | 2.571 | 1.644 | 0.652 | 1.56x | **3.94x** | 2.52x |
| 3 | B4 D128 H4 S128 | 7.169 | 2.024 | 0.757 | 3.54x | **9.47x** | 2.67x |
| 4 | B16 D128 H4 S128 | 19.373 | 4.726 | 1.375 | 4.10x | **14.09x** | 3.44x |
| 5 | B128 D128 H4 S128 | 139.502 | 33.535 | 7.359 | 4.16x | **18.96x** | 4.56x |
| 6 | B10000 D128 H4 S128 | 14639.960 | 2669.151 | 567.842 | 5.48x | **25.78x** | 4.70x |
| 7 | B64 D32 H4 S128 | 25.959 | 11.242 | 1.124 | 2.31x | **23.10x** | 10.00x |
| 8 | B64 D1024 H4 S128 | 470.854 | 165.156 | 127.557 | 2.85x | **3.69x** | 1.29x |
| 9 | B64 D128 H1 S128 | 35.099 | 7.932 | 3.923 | 4.42x | **8.95x** | 2.02x |
| 10 | B64 D128 H2 S128 | 46.106 | 12.966 | 3.850 | 3.56x | **11.98x** | 3.37x |
| 11 | B64 D128 H16 S128 | 130.289 | 41.966 | 3.850 | 3.10x | **33.84x** | 10.90x |
| 12 | B64 D128 H4 S32 | 12.224 | 3.341 | 1.309 | 3.66x | **9.34x** | 2.55x |
| 13 | B64 D128 H4 S1024 | 1899.325 | 562.926 | 45.231 | 3.37x | **41.99x** | 12.45x |

| Metric | Value |
|---|---:|
| Median MLX speedup over CPU | **14.09x** |
| Range | 3.69x to 41.99x |
| Median MLX rate | 2.231 TFLOP/s |
| Best MLX rate | 4.178 TFLOP/s (shape 13) |

**† the CPU column came from the cache**, not from this sweep, so the two
speedup columns against it mix two sweeps. See the `--cpu-cache` rule in
[CLAUDE.md](CLAUDE.md). The MLX and MPS columns are measured fresh, and the
MLX column is the one that decides whether a change won.

Accuracy: all 13 shapes PASS at `atol=0.002`, `rtol=0.02`, against the CPU
baseline, with **zero failed elements** on every shape, including
0 / 163,840,000 at shape 6. `max_abs` runs 1.19e-06 to 2.65e-06.

**Read the MPS column, not only the CPU column.** The CPU baseline moves with
machine load and chip temperature. It drifted up to 45.9% between two sweeps
on an unchanged baseline. MPS is stable to a few percent, so it is the better
control when you compare two builds.

Shape 14 (B32 D1024 H16 S100000 L2) is disabled. Its input alone is 12.2 GiB
and the machine holds 12.0 GiB, so no backend runs it here.

## Current bottlenecks

Measured with `profiling/stage_roofline.py` after row 36. Shape 6 carries
66.5% of the FLOP weight: one layer, one chunk of 1024 rows, real layer time
**14.22 ms**. The roofs are 4.06 TFLOP/s and 128 GB/s, both measured here.

| Stage | ms | share | Limit | Against its own roof | Headroom |
|---|---:|---:|---|---:|---|
| qkv proj | 3.61 | 24% | COMPUTE | 88% of matmul peak | little |
| sdpa (attention) | 2.39 | 16% | IO | 88% of bandwidth | little. Row 34 took it |
| ffn_in + gelu | 2.34 | 15% | COMPUTE | 45% of matmul peak | **1.01 ms. Row 33** |
| ffn_out (+residual) | 1.69 | 11% | IO | 93% of bandwidth | none. Row 36 took it |
| out proj (+residual) | 1.69 | 11% | IO | 93% of bandwidth | none. Row 36 took it |
| ln1 + ln2 | 2.28 | 15% | IO | 92% of bandwidth | none. Rows 31 and 36 took it |
| merge heads | 1.14 | 8% | IO | 92% of bandwidth | none |
| split+transpose | free | — | — | a strided view, not a copy | — |

Row 36 removed the two separate residual adds, so the block is nine stages,
not eleven. **Seven of the eight measurable stages now run at 87% or more of
the roof that binds them.** The ninth, `split+transpose`, sits at the launch
floor because it is a view, not work.

`ffn_in + gelu` is the one stage with real headroom. Compare it against
`ffn_out`, which does the **same** 4.295 GFLOP:

| | GFLOP | MiB moved | ms |
|---|---:|---:|---:|
| `ffn_in + gelu` | 4.295 | 128.1 | 2.34 |
| `ffn_out (+residual)` | 4.295 | 192.1 | 1.69 |

`ffn_out` moves 50% more bytes, because it now carries the residual too, and
it still finishes 0.65 ms sooner. The whole difference is the GELU, which is
one extra pass over DRAM.

### What is still open

| # | Bottleneck | Size | Why it is still open |
|---:|---|---|---|
| 33 | GELU runs as a separate memory pass | 1.01 ms for each layer and chunk, **7%** of shape 6 | The GELU kernel is efficient at 115 GB/s. The extra pass is the cost, and `mx.compile` does not fuse it. The steel GEMM header exposes an `apply_epilogue` hook, so the row 25 hoisting method can probably reach it |
| 37 | Shape 8 cannot reach row 36 | about 1.6 ms of its 32.5 ms layer, **0.9%** FLOP-weighted | `mx.fast.layer_norm` takes no `pre_bias`, and `fast_layernorm` serves a row width under 256. A wide variant needs 32 floats per lane against 8 today, so register pressure is the open question |
| 21, 26 | MLX never calls its own `bd192` and `bd256` attention kernels | shape 8, 21.3% of the FLOP weight | `head_dim` 256 takes the fallback. `head_dim` cannot pad down, and a head cannot split. The threadgroup memory for `bd256` exceeds the 32 KiB limit |
| — | Small shapes are launch-bound | shapes 2, 3, 7 and 12 | Under 0.2% of the FLOP weight together. Not worth the effort |

Shape 8 is the exception to all of this. It is genuinely compute-bound: its
QKV projection runs at **100.1% of the measured matmul peak** at 351
FLOP/byte, and its other three matmuls reach 89% to 99%. Nothing is left in
it except the fallback attention kernel and row 37.

Shape 13 is close behind. Its `sdpa` is 45% of the layer and already holds
80% of the matmul peak.

## What is in the model

The full log is [OPTIMIZATIONS.md](OPTIMIZATIONS.md). Its source of truth
table is the only place that states the status of an optimization. The
largest wins:

| # | Optimization | Effect |
|---:|---|---|
| 1 | MLX behind the torch interface | 4.4x to 7.4x against torch CPU |
| 25 | Hoist MLX's `steel_attention` and compile it at an unshipped `head_dim` | **1.308x** FLOP-weighted |
| 31 | A single-pass LayerNorm kernel for a row width under 256 | **1.205x** FLOP-weighted |
| 34 | Read q, k and v as strided views, and write the head layout directly | **1.239x** FLOP-weighted |
| 36 | Defer the residual biases, and give the residual add to the GEMM C operand | **1.132x** FLOP-weighted |
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
| `fast_layernorm.py` | A single-pass LayerNorm kernel for a row width under 256, with the row 36 `pre_bias` hook |
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
    .venv/bin/python3 scoreboard.py --cpu-cache --label "..."   # the full sweep
    .venv/bin/python3 profiling/stage_roofline.py --shapes 6

Check that no other run holds the GPU before you measure. Two runs share one
GPU, and each one makes the other reading false:

    ps -Ao pid,etime,%cpu,command | grep "[.]venv/bin/python3" | grep -v shell-snapshots

Machine: Apple M3 Pro, 14 GPU cores, 18 GiB unified memory, macOS 24.6,
Python 3.13.5, torch 2.13.0, mlx 0.32.2. See
[references/machine.md](references/machine.md).

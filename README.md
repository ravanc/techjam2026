# Transformer forward pass, optimized for Apple silicon

`UserOptimizedTransformer` runs the forward pass of `BaselineTransformer` in
MLX on the GPU, behind the torch interface the harness expects. The baseline
does not change. It is the reference for both accuracy and speed.

Install:

    python3 -m venv .venv
    .venv/bin/python3 -m pip install -r requirements.txt

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
| 1 | B64 D128 H4 S128 | 61.252 | 17.404 | 3.460 | 3.52x | **17.70x** | 5.03x |
| 2 | B1 D128 H4 S128 | 2.571 | 1.594 | 0.563 | 1.61x | **4.57x** | 2.83x |
| 3 | B4 D128 H4 S128 | 7.169 | 1.960 | 0.632 | 3.66x | **11.34x** | 3.10x |
| 4 | B16 D128 H4 S128 | 19.373 | 4.748 | 1.197 | 4.08x | **16.18x** | 3.97x |
| 5 | B128 D128 H4 S128 | 139.502 | 34.021 | 7.235 | 4.10x | **19.28x** | 4.70x |
| 6 | B10000 D128 H4 S128 | 15777.204 | 2604.115 | 480.810 | 6.06x | **32.81x** | 5.42x |
| 7 | B64 D32 H4 S128 | 25.959 | 11.382 | 1.070 | 2.28x | **24.26x** | 10.64x |
| 8 | B64 D1024 H4 S128 | 470.782 | 166.892 | 120.164 | 2.82x | **3.92x** | 1.39x |
| 9 | B64 D128 H1 S128 | 35.099 | 8.271 | 3.667 | 4.24x | **9.57x** | 2.26x |
| 10 | B64 D128 H2 S128 | 46.106 | 13.063 | 3.582 | 3.53x | **12.87x** | 3.65x |
| 11 | B64 D128 H16 S128 | 130.289 | 42.156 | 3.714 | 3.09x | **35.08x** | 11.35x |
| 12 | B64 D128 H4 S32 | 12.224 | 3.306 | 1.133 | 3.70x | **10.79x** | 2.92x |
| 13 | B64 D128 H4 S1024 | 1899.325 | 569.699 | 42.224 | 3.33x | **44.98x** | 13.49x |

The MLX column totals **669.5 ms** over the 13 shapes. That total is the
score this project moves. Rows 46 and 37 took it from 722.3 ms, and about a
fifth of that is machine drift: the MPS control moved 1.9% across the same
two sweeps, so the attributable gain is about **1.068x**.

**† the CPU column came from the cache**, not from this sweep, so the two
speedup columns against it mix two sweeps. See the `--cpu-cache` rule in
[CLAUDE.md](CLAUDE.md). The MLX and MPS columns are measured fresh, and the
MLX column is the one that decides whether a change won.

Accuracy: all 13 shapes PASS at `atol=0.002`, `rtol=0.02`, against the CPU
baseline, with **zero failed elements** on every shape, including
0 / 163,840,000 at shape 6. `max_abs` runs 1.07e-06 to 3.34e-06, against an `atol` of 0.002.

The shape 6 CPU entry above drifted from 14639.960 ms to 15777.204 ms on an
unchanged baseline. Nothing in the model caused that. It is why the rule
below exists.

**Read the MPS column, not only the CPU column.** The CPU baseline moves with
machine load and chip temperature. It drifted up to 45.9% between two sweeps
on an unchanged baseline. MPS is stable to a few percent, so it is the better
control when you compare two builds.

Shape 14 (B32 D1024 H16 S100000 L2) is disabled. Its input alone is 12.2 GiB
and the machine holds 12.0 GiB, so no backend runs it here.

## Current bottlenecks

Measured with `profiling/stage_roofline.py --shapes 6` after row 46. Shape 6
carries 66.5% of the FLOP weight: one layer, one chunk of 1024 rows, real
layer time **12.02 ms**, stage sum 11.63 ms.

| Stage | ms | share | Limit | Against its own roof | Headroom |
|---|---:|---:|---|---:|---|
| qkv proj (+layer norm) | 3.560 | 31% | COMPUTE | 89% of matmul peak | little. Row 46 took the LayerNorm |
| sdpa (attention) | 2.218 | 19% | IO | at the bandwidth roof | none. Row 34 took it |
| ffn_out (+residual) | 1.665 | 14% | IO | at the bandwidth roof | none. Row 36 took it |
| out proj (+residual) | 1.621 | 14% | IO | at the bandwidth roof | none. Row 36 took it |
| ffn_in + gelu (+layer norm) | 1.496 | 13% | COMPUTE | 71% of matmul peak | little. Rows 33 and 46 took it |
| ln1 stats + ln2 stats | 1.071 | 9% | IO | at the bandwidth roof | none. Row 46 halved it |
| merge heads | free | — | — | a reshape, not a copy | — |
| split+transpose | free | — | — | a strided view, not a copy | — |

The block is eight stages, not eleven: row 36 removed the two separate
residual adds and row 33 removed the GELU pass.

**Read `%comp` and `%mem` in that tool as a rank, not as a fraction of the
roof.** It subtracts the launch floor from the stage time but not from the
roof, so both columns read high and can pass 100%. See
[references/machine.md](references/machine.md). The table above says "at the
bandwidth roof" rather than a percentage for that reason. The claim still
holds where it matters: a standalone `fast_layernorm` at the shape 6 `ln1`
size runs at 107.5 GB/s and a plain `x * 2.0` at the same size runs at
108.9 GB/s, so the LayerNorm was already at copy speed.

That is why row 46 is a byte optimization and not a kernel optimization. No
better LayerNorm kernel could win `ln1`, because the old one already ran at
copy speed. Only moving fewer bytes could. A LayerNorm is affine in the row,
so it distributes through the GEMM below it and folds into that GEMM's
weights at build time:

    out[i,n] = P_i * (x @ Bw + c3)[i,n] - Q_i * c1[n] + c2[n]

`Bw`, `c1`, `c2` and `c3` depend on the weights alone, so they are built
once. What is left at run time is two floats for each row:

| at the shape 6 chunk | ms | MiB moved |
|---|---:|---:|
| `fast_layernorm`, writing a whole activation | 1.108 | 128.0 |
| the statistics kernel, writing 2 floats per row | **0.545** | **65.0** |

### What is still open

Rows 43 and 44 both tried to delete a DRAM round trip inside the FFN, and
both were measured and reverted. Row 46 deleted the LayerNorm round trip
instead, and it is what shipped. See OPTIMIZATIONS.md.


| # | Bottleneck | Size | Why it is still open |
|---:|---|---|---|
| 21, 26 | MLX never calls its own `bd192` and `bd256` attention kernels | shape 8, 21.3% of the FLOP weight | `head_dim` 256 takes the fallback. `head_dim` cannot pad down, and a head cannot split. The threadgroup memory for `bd256` exceeds the 32 KiB limit |
| — | qkv proj | 3.56 ms, 31% of the shape 6 layer | The largest stage, and it already runs at 89% of the matmul peak. It holds 43% of the block FLOPs because it is three projections in one |
| — | Small shapes are launch-bound | shapes 2, 3, 7 and 12 | Under 0.2% of the FLOP weight together. Shape 2 declines every fused path because it has 128 rows, under the 512 row gate |

Shape 8 is compute-bound: its four GEMMs are most of its layer and they run
near the measured matmul peak. Its remaining slack is the fallback attention
kernel and the carry note under row 37.

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
| 34 | Read q, k and v as strided views, and write the head layout directly | **1.239x** FLOP-weighted |
| 31 | A single-pass LayerNorm kernel for a row width under 256 | **1.205x** FLOP-weighted |
| 36 | Defer the residual biases, and give the residual add to the GEMM C operand | **1.132x** FLOP-weighted |
| 29 | `mx.addmm` for every projection, so the GPU adds the bias inside the matmul | 1.096x FLOP-weighted |
| 33 | Fold GELU into the `ffn_in` matmul epilogue | 1.064x FLOP-weighted |
| 46 | Absorb the LayerNorm into the GEMM weights, and apply it in the epilogue | **1.060x** FLOP-weighted |
| 37 | Defer the residual biases on shape 8 too, by adding the carry once before the final LayerNorm | shape 8 **1.040x**, on a 0.991x control |
| 7 | A shape-aware kernel plan (`KernelPlan`) | 1.57x at shape 13 |
| 10 | Batch chunking, full depth for each chunk | peak 9.16 GiB to 2.68 GiB |
| 23 | Return the output as a view of MLX memory, not a copy | 71.6 ms of 1590.2 ms at shape 6 |

Three of these are custom Metal kernels: `steel_attention.py`,
`steel_gemm.py` and `fast_layernorm.py`. All three hoist a kernel that MLX
already ships and compile it in a way MLX does not expose.

[agent_loop.md](agent_loop.md) holds the loop that produces these: how a
candidate is screened, the four gates it must pass to be kept, and a log of
what each turn measured.

## Layout

| File | Role |
|---|---|
| `torch_transformer_benchmark.py` | The baseline model, the MLX model, and the harness |
| `steel_attention.py` | MLX's flash attention kernel, compiled at a `head_dim` MLX does not ship |
| `steel_gemm.py` | MLX's steel GEMM, with epilogues MLX does not expose: GELU (row 33) and the LayerNorm (row 46) |
| `fast_layernorm.py` | A single-pass LayerNorm for a row width under 256, the row 36 `pre_bias` hook, and the row 46 statistics kernel |
| `scoreboard.py` | The graded run over all 13 shapes |
| `flops.py` | The FLOP model and the measured matmul rates |
| `appendix_cases.py` | The 14 shapes as code |
| `bench_cases.py` | Deterministic input generation, shared by every backend |
| `test_backends.py` | Cross-backend comparison: CPU, MPS, MLX |
| `test_padding.py` | Padded and ragged batches, including an empty sample |
| `profiling/WORKFLOW.md` | How to find the next optimization. Read it first |
| `profiling/stage_roofline.py` | Splits one block into stages and names each limit |
| `profiling/sdpa_dispatch.py` | Finds which `head_dim` values reach the fused SDPA kernel |
| `profiling/tile_probe.py` | What a GEMM tile costs, when a fusion forces the tile. It killed row 43 |
| `profiling/ln_absorb_probe.py` | The accuracy screen for row 46, against a float64 reference |
| `agent_loop.md` | The optimization loop: the screens, the four gates, and the run log |
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

`steel_gemm.py` and `steel_attention.py` read the Metal headers from the
installed `mlx` package. `mlx_kernels.py` finds them. It asks `mlx` for its
own path, so a venv at any place and at any Python version works. Set
`MLX_KERNELS_DIR` to override the search. When the headers are absent, both
modules turn themselves off and the model takes the plain MLX path, so the
run completes with a slower number instead of an error.

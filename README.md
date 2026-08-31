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

All 13 runnable Appendix 3.7 shapes, float32, at commit `311a420`. `MLX` is
`UserOptimizedTransformer`. `CPU` is the torch baseline, and it is the
reference the harness uses. `MPS` is the same torch baseline on the GPU,
through Metal.

| # | Shape | CPU ms † | MPS ms | MLX ms | MPS vs CPU | **MLX vs CPU** | MLX vs MPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 44.227 | 17.440 | 3.204 | 2.54x | **13.80x** | 5.44x |
| 2 | B1 D128 H4 S128 | 1.695 | 1.526 | 0.610 | 1.11x | **2.78x** | 2.50x |
| 3 | B4 D128 H4 S128 | 4.700 | 2.016 | 0.680 | 2.33x | **6.91x** | 2.96x |
| 4 | B16 D128 H4 S128 | 13.364 | 4.703 | 1.301 | 2.84x | **10.27x** | 3.62x |
| 5 | B128 D128 H4 S128 | 114.944 | 33.709 | 6.019 | 3.41x | **19.10x** | 5.60x |
| 6 | B10000 D128 H4 S128 | 14946.867 | 2771.854 | 454.810 | 5.39x | **32.86x** | 6.09x |
| 7 | B64 D32 H4 S128 | 27.267 | 11.520 | 1.012 | 2.37x | **26.95x** | 11.39x |
| 8 | B64 D1024 H4 S128 | 473.041 | 167.571 | 119.196 | 2.82x | **3.97x** | 1.41x |
| 9 | B64 D128 H1 S128 | 32.895 | 8.103 | 3.447 | 4.06x | **9.54x** | 2.35x |
| 10 | B64 D128 H2 S128 | 42.969 | 13.095 | 3.374 | 3.28x | **12.74x** | 3.88x |
| 11 | B64 D128 H16 S128 | 133.328 | 42.513 | 3.425 | 3.14x | **38.93x** | 12.41x |
| 12 | B64 D128 H4 S32 | 10.042 | 3.467 | 1.183 | 2.90x | **8.49x** | 2.93x |
| 13 | B64 D128 H4 S1024 | 1918.386 | 577.361 | 40.689 | 3.32x | **47.15x** | 14.19x |
| | **total** | **17763.7** | **3654.9** | **639.0** | 4.86x | **27.80x** | 5.72x |

Median speedup over the CPU baseline: **12.74x**. Range 2.78x to 47.15x.

The MLX total over the 13 shapes is the score this project moves. Where it
came from:

| Sweep | MLX total | Against the sweep before |
|---|---:|---|
| attempts 1 to 11 | 2074.3 ms | the first full sweep |
| row 25, steel attention | 1475.9 ms | 1.308x FLOP-weighted |
| row 29, `mx.addmm` | 1298.3 ms | 1.096x |
| row 31, single-pass LayerNorm | 1077.6 ms | 1.205x |
| row 34, strided q, k and v | 869.7 ms | 1.239x |
| row 36, deferred residual bias | 768.6 ms | 1.132x |
| row 33, GELU in the epilogue | 722.3 ms | 1.064x |
| row 46, LayerNorm in the epilogue | 681.4 ms | 1.060x |
| row 47, LayerNorm statistics in the epilogue | 638.7 ms | 1.075x |
| row 50, final LayerNorm in the epilogue | 628.8 ms | 1.019x by A/B |
| this sweep, `311a420` | 639.0 ms | machine drift, see below |

The last two rows differ by 1.6% with no code change between them. The MPS
control moved the same way (3625.3 ms to 3654.9 ms, 0.8%), so that is the
machine, not the model. Read a small difference against its own MPS control.

**† the CPU column came from the cache**, not from this sweep, so the two
speedup columns against it mix two sweeps. See the `--cpu-cache` rule in
[CLAUDE.md](CLAUDE.md). The MLX and MPS columns are measured fresh, and the
MLX column is the one that decides whether a change won.

Accuracy: all 13 shapes PASS at `atol=0.002`, `rtol=0.02`, against the CPU
baseline, with **zero failed elements** on every shape, including
0 / 163,840,000 at shape 6. `max_abs` runs 1.19e-06 to 3.46e-06, against an
`atol` of 0.002. `test_padding.py` adds 18 padded and ragged cases.

**Read the MPS column, not only the CPU column.** The CPU baseline moves with
machine load and chip temperature. It drifted up to 45.9% between two sweeps
on an unchanged baseline. MPS is stable to a few percent, so it is the better
control when you compare two builds.

### Shape 14

**The torch baseline cannot run shape 14, and no machine can. The MLX path
runs it in 8.41 minutes.** Those are two separate facts, and an earlier
version of this file ran them together.

`BaselineSelfAttention` materializes a `B x H x S x S` score matrix, which is
18.6 TiB at this shape and 596 GiB for a single batch row of a single layer.
So the CPU and MPS baselines stop with an allocation error before they
compute anything. They have no run time here, and there is no speedup to
report.

The MLX path runs it because `head_dim = 64` reaches MLX's fused flash
kernel, which holds no score matrix. `plan_kernels()` already splits the
batch into 32 chunks of one 391 MiB sequence.

| Item | Value |
|---|---|
| full batch, compute | **504.5 s = 8.41 min** |
| per row, median | 15.766 s |
| tokens/s | 6,343 |
| rate | 5.356 counted TFLOP/s, 2.758 executed TFLOP/s |
| MFU | 108.3% counted, **55.8% executed** |
| accuracy | PASS. `max_abs` 2.03e-06, 0 of 1,048,576 elements failed |

It runs under `shape14_harness.py`, not the graded sweep, because the sweep
needs a baseline for both of its jobs. The reading lives in
`profiling/results/shape14.json`. See `OPTIMIZATIONS.md` row 55 and
[references/test-shapes.md](references/test-shapes.md).

## Model FLOPs Utilization

    MFU = model_flops / seconds / 4.946 TFLOP/s

The denominator is `14 cores x 128 ALUs x 2 x 1.380 GHz`, and every term in
it is checked on this machine. The core count comes from `system_profiler`,
the 1380 MHz top state from the GPU DVFS table `voltage-states9` in the pmgr
device tree, and the ALU count is bounded below at 105 by the measured
matmul rate.

Reported for MLX only. **Read it against 82%, not 100%**: the GPU does not
hold its 1380 MHz top state. A saturating FMA loop sustains 3.92 TFLOP/s
(`profiling/probes/alu_peak.py`) and a plain matmul reaches 4.06 TFLOP/s
(`flops.py --peak`). Both imply a clock near 1.1 GHz, and no kernel can
close that gap.

Two columns, because a causal shape has two honest numerators. **counted**
divides by the full `S x S` attention, which is what the baseline computes,
and it is the graded number. **executed** divides by the causal triangle,
which is what the kernel really runs.

| # | Shape | GFLOP counted | GFLOP executed | exec/cnt | FLOP share | MLX TFLOP/s | MFU counted | **MFU executed** |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 8.59 | 7.52 | 87.6% | 0.4% | 2.681 | 54.2% | **47.5%** |
| 2 | B1 D128 H4 S128 | 0.13 | 0.12 | 87.6% | 0.0% | 0.220 | 4.4% | **3.9%** |
| 3 | B4 D128 H4 S128 | 0.54 | 0.47 | 87.6% | 0.0% | 0.790 | 16.0% | **14.0%** |
| 4 | B16 D128 H4 S128 | 2.15 | 1.88 | 87.6% | 0.1% | 1.651 | 33.4% | **29.2%** |
| 5 | B128 D128 H4 S128 | 17.18 | 15.05 | 87.6% | 0.9% | 2.854 | 57.7% | **50.6%** |
| 6 | B10000 D128 H4 S128 | 1,342.18 | 1,175.72 | 87.6% | 66.5% | 2.951 | 59.7% | **52.3%** |
| 7 | B64 D32 H4 S128 | 0.94 | 0.67 | 71.7% | 0.0% | 0.929 | 18.8% | **13.5%** |
| 8 | B64 D1024 H4 S128 | 429.50 | 420.97 | 98.0% | 21.3% | 3.603 | 72.9% | **71.4%** |
| 9 | B64 D128 H1 S128 | 8.59 | 7.52 | 87.6% | 0.4% | 2.492 | 50.4% | **44.1%** |
| 10 | B64 D128 H2 S128 | 8.59 | 7.52 | 87.6% | 0.4% | 2.546 | 51.5% | **45.1%** |
| 11 | B64 D128 H16 S128 | 8.59 | 7.52 | 87.6% | 0.4% | 2.508 | 50.7% | **44.4%** |
| 12 | B64 D128 H4 S32 | 1.74 | 1.68 | 96.3% | 0.1% | 1.475 | 29.8% | **28.7%** |
| 13 | B64 D128 H4 S1024 | 188.98 | 120.33 | 63.7% | 9.4% | 4.644 | 93.9% | **59.8%** |
| 14 | B32 D1024 H16 S100000 | 2,701,970.64 | 1,391,263.74 | 51.5% | excluded | 5.356 | 108.3% | **55.8%** |

Shapes 1 to 13 come from the sweep in
[references/scoreboard.md](references/scoreboard.md). Shape 14 comes from
`shape14_harness.py`, and it is excluded from every mean below.

### The score

| Over shapes 1 to 13 | counted | executed |
|---|---:|---:|
| unweighted mean | 45.64% | **38.80%** |
| **FLOP-weighted mean** | **65.44%** | **56.84%** |

Best single shape: 93.9% counted at shape 13, and 71.4% executed at shape 8.
Both sit near the 82% practical ceiling on their own numerator.

**Do not weight shape 14 into that mean.** Shape 14 is 2.70 PFLOP against
2.02 TFLOP for all 13 others combined, which is 1,339x. It would carry
**99.93%** of the FLOP weight, and the weighted score would stop being a
score on the model and become a score on shape 14 alone. For the same
reason, read the 13-shape weighted number knowing shape 6 already carries
66.5% of it.

### Why counted and executed differ, and by how much

The gap is **not** a constant factor. It runs from 98.0% at shape 8, which
has almost no triangle to skip, to 51.5% at shape 14, which is 97%
attention. Shape 14's counted MFU passes 100% for that reason, and that is
not an error in the stopwatch. The 87.6% shared by most shapes is
arithmetic: at `S=128` the triangle saves half of attention, and attention
is 25% of the work.

Quote **counted** as the score, because it is the graded numerator. Quote
**executed** when you judge a kernel against the 82% ceiling.

`flops.py` holds the FLOP model. It counts matmuls only.

## Current bottlenecks

    .venv/bin/python3 profiling/probes/stage_roofline.py --shapes 6

Measured at commit `311a420`. Shape 6 carries 66.5% of the FLOP weight: one
layer, one chunk of 1024 rows, real layer time **12.66 ms**, raw stage sum
12.92 ms, floor 0.128 ms.

| Stage | raw ms | share | Limit | Against its own roof | Headroom |
|---|---:|---:|---|---:|---|
| qkv proj (+layer norm) | 3.687 | 29% | COMPUTE | 89% of matmul peak | little. Row 46 took the LayerNorm |
| sdpa (attention) | 2.527 | 20% | IO | at the bandwidth roof | none. Row 34 took it |
| ffn_out (+residual) | 1.803 | 14% | IO | at the bandwidth roof | none. Rows 36, 47 and 50 took it |
| out proj (+residual) | 1.800 | 14% | IO | at the bandwidth roof | none. Rows 36 and 47 took it |
| ffn_in + gelu (+layer norm) | 1.639 | 13% | COMPUTE | 70% of matmul peak | little. Rows 33 and 46 took it |
| ln1 stats + ln2 stats | 1.370 | 11% | IO | at the bandwidth roof | **row 47 already removed these two from the model.** See the caveat below |
| merge heads | 0.041 | — | LAUNCH | — | a reshape, not a copy |
| split+transpose | 0.049 | — | LAUNCH | — | a strided view, not a copy |

**The probe is one row behind the model.** It produces the LayerNorm
statistics with a standalone `layer_norm_stats` kernel, which is the row 46
form. The model now takes them in the epilogue of the GEMM above (row 47),
so the two `stats` rows are an upper bound on work the model no longer does,
and the shipped block is eight stages, not ten. Row 36 removed the two
separate residual adds and row 33 removed the GELU pass. Fixing the probe to
follow `plan_kernels()` is open work.

**Read `%comp` and `%mem` in that tool as a rank, not as a fraction of the
roof.** It subtracts the launch floor from the stage time but not from the
roof, so both columns read high and can pass 100%. See
[references/machine.md](references/machine.md). The table above says "at the
bandwidth roof" rather than a percentage for that reason. The claim still
holds where it matters: a standalone `fast_layernorm` at the shape 6 `ln1`
size runs at 107.5 GB/s and a plain `x * 2.0` at the same size runs at
108.9 GB/s, so the LayerNorm was already at copy speed.

That is why rows 46, 47 and 50 are byte optimizations and not kernel
optimizations. No better LayerNorm kernel could win `ln1`, because the old
one already ran at copy speed. Only moving fewer bytes could. A LayerNorm is
affine in the row, so it distributes through the GEMM below it and folds
into that GEMM's weights at build time:

    out[i,n] = P_i * (x @ Bw + c3)[i,n] - Q_i * c1[n] + c2[n]

`Bw`, `c1`, `c2` and `c3` depend on the weights alone, so they are built
once. What is left at run time is two floats for each row:

| at the shape 6 chunk | ms | MiB moved |
|---|---:|---:|
| `fast_layernorm`, writing a whole activation | 1.108 | 128.0 |
| the statistics kernel, writing 2 floats per row (row 46) | 0.545 | 65.0 |
| the same statistics, in the GEMM epilogue (row 47) | **0** | **0** |

Row 50 closes the last one. The FINAL LayerNorm has no GEMM below it, so
row 46 cannot reach it; the GEMM **above** it can, and that fusion is 1.399x
at the GEMM alone.

### What is still open

Five rows of 55 are OPEN. Every other route into the shape 6 block is now
measured and closed.

| # | Bottleneck | Size | Why it is still open |
|---:|---|---|---|
| 21, 26, 41, 49 | MLX never calls its own `bd192` and `bd256` attention kernels | shape 8, 21.3% of the FLOP weight | `head_dim` 256 takes the fallback. `head_dim` cannot pad down, and a head cannot split. Rows 41 and 49 built the narrow-block and D-blocked routes: both are correct and both LOSE (0.904x and 0.832x). Recheck after an MLX upgrade |
| 16 | Quantization by `mx.quantize` | every shape | not measured. It will fail the accuracy test |
| 19 | Place the `S >= 4*D` threshold by sweeping `seq_len` | `head_dim` 32 to 48 | not measured at the model level |
| 22 | Block a causal wide head that misses the fused set | `head_dim > 128` and long `S` | no appendix shape reaches it. `plan_kernels()` refuses to block because it tests `effective_head_dim < 64` |
| 45 | Build the deferred bias `carry` at weight build time | shape 2 only | about 0.1% FLOP-weighted, under the 1% noise floor. A sweep cannot measure it |
| — | qkv proj | 3.69 ms, 29% of the shape 6 layer | The largest stage, and it already runs at 89% of the matmul peak. It holds 43% of the block FLOPs because it is three projections in one. Row 40 proved the tile is already right, and row 54 re-proved it with the epilogues on |
| — | Small shapes are launch-bound | shapes 2, 3, 7 and 12 | Under 0.2% of the FLOP weight together. Shape 2 declines every fused path because it has 128 rows, under the 512 row gate |

Shape 8 is compute-bound: its four GEMMs run at 99-101% of the measured
matmul peak and carry 25.2 ms of its 30.18 ms layer. Its remaining slack is
the fallback attention kernel alone.

Shape 13 is close behind. Its `sdpa` is 45% of the layer and already holds
80% of the matmul peak.

Four classes are closed by measurement, not by opinion:

| # | Class | Why it is closed |
|---:|---|---|
| 48 | transfer/compute overlap | 0.974x. Unified memory makes the CPU copy and the GPU kernels contend, so the copy does not hide |
| 52 | a persistent kernel over the launch gaps | the whole prize is 0.06% of the call. GPU idle between kernels is 1.05%, and 9 or 10 chunk-boundary `mx.eval` gaps hold all of it |
| 53 | fusing attention into the out projection | 25.5 KiB of the 32 KiB threadgroup budget, against 9.0 KiB today. Row 44 measured that band at 0.996x to 0.896x |
| 54 | a better steel GEMM tile | 129 tiles on each of four shape 6 stages. No tile wins outside the noise floor |

## What is in the model

The full log is [OPTIMIZATIONS.md](OPTIMIZATIONS.md). Its source of truth
table is the only place that states the status of an optimization.

**55 rows: 26 KEPT, 14 REVERTED, 7 RULED OUT, 5 OPEN.** The reverted rows
are kept on purpose. They stop repeated work.

The largest wins:

| # | Optimization | Effect |
|---:|---|---|
| 1 | MLX behind the torch interface | 4.4x to 7.4x against torch CPU |
| 25 | Hoist MLX's `steel_attention` and compile it at an unshipped `head_dim` | **1.308x** FLOP-weighted |
| 34 | Read q, k and v as strided views, and write the head layout directly | **1.239x** FLOP-weighted |
| 31 | A single-pass LayerNorm kernel for a row width under 256 | **1.205x** FLOP-weighted |
| 36 | Defer the residual biases, and give the residual add to the GEMM C operand | **1.132x** FLOP-weighted |
| 29 | `mx.addmm` for every projection, so the GPU adds the bias inside the matmul | 1.096x FLOP-weighted |
| 47 | Take the LayerNorm statistics in the epilogue of the GEMM that writes the activation | **1.075x** FLOP-weighted |
| 33 | Fold GELU into the `ffn_in` matmul epilogue | 1.064x FLOP-weighted |
| 46 | Absorb the LayerNorm into the GEMM weights, and apply it in the epilogue | **1.060x** FLOP-weighted |
| 37 | Defer the residual biases on shape 8 too, by adding the carry once before the final LayerNorm | shape 8 **1.040x**, on a 0.991x control |
| 50 | Apply the final LayerNorm in the epilogue of the GEMM above it | 1.019x FLOP-weighted by a 3x13 controlled A/B. 1.399x at the GEMM alone |
| 7 | A shape-aware kernel plan (`KernelPlan`) | 1.57x at shape 13 |
| 10 | Batch chunking, full depth for each chunk | peak 9.16 GiB to 2.68 GiB |
| 23 | Return the output as a view of MLX memory, not a copy | 71.6 ms of 1590.2 ms at shape 6 |
| 51 | Refuse a steel tile whose block loader has inexact thread geometry | a bug fix. MLX truncates `TCOLS` with no guard, so an undispatched tile answers silently wrong |

Three of these are custom Metal kernels: `steel_attention.py`,
`steel_gemm.py` and `fast_layernorm.py`. All three hoist a kernel that MLX
already ships and compile it in a way MLX does not expose. Six epilogues now
live in `steel_gemm.py`: GELU (row 33), the LayerNorm (row 46), the row
statistics (row 47) and the final LayerNorm (row 50).

[agent_loop.md](agent_loop.md) holds the loop that produces these: how a
candidate is screened, the four gates it must pass to be kept, and a log of
what each turn measured.

## Layout

    torch_transformer_benchmark.py   the models and the harness
    scoreboard.py                    the graded sweep, shapes 1 to 13
    demo.py                          the same work, with the elapsed time on screen
    shape14_harness.py               shape 14 alone. See OPTIMIZATIONS.md row 55
    steel_gemm.py  steel_attention.py  fast_layernorm.py  mlx_kernels.py
    appendix_cases.py  bench_cases.py  flops.py
    test_backends.py  test_padding.py  test_paths.py

    profiling/probes/     one question, one script. Each row names its probe
    profiling/tools/      Instruments and Metal capture, charts, and the signpost shim
    profiling/results/    the JSON a run writes. Git tracks all but cpu_cache
    profiling/results/harness/  one harness sweep, and its charts. Git keeps
                                the JSON and the SVG, and ignores the PNG,
                                the PDF and the raw case logs
    profiling/traces/     recorded traces. Git ignores them, they reach 1 GB

    references/           measured facts, one subject per file
    references/figures/   diagrams. No script rebuilds them, so git tracks them
    references/wwdc/      Apple GPU talk transcripts

### The model and the harness

| File | Role |
|---|---|
| `torch_transformer_benchmark.py` | The baseline model, the MLX model, and the harness |
| `steel_attention.py` | MLX's flash attention kernel, compiled at a `head_dim` MLX does not ship |
| `steel_gemm.py` | MLX's steel GEMM, with epilogues MLX does not expose: GELU (row 33), the LayerNorm (row 46), the row statistics (row 47) and the final LayerNorm (row 50). It also holds the row 51 loader-geometry guard |
| `fast_layernorm.py` | A single-pass LayerNorm for a row width under 256, the row 36 `pre_bias` hook, and the row 46 statistics kernel |
| `mlx_kernels.py` | Finds the Metal headers inside the installed `mlx` package |
| `bench_cases.py` | Deterministic input generation, shared by every backend |
| `appendix_cases.py` | The 14 shapes as code. It runs 1 to 13 |
| `flops.py` | The FLOP model and the measured matmul rates |

### Run it

| File | Role |
|---|---|
| `scoreboard.py` | The graded run over all 13 shapes. Always pass `--cpu-cache` and `--label` |
| `demo.py` | The same run, with the elapsed time and the progress on screen. It records nothing. Use it to watch a long sweep, or to show the work |
| `shape14_harness.py` | Shape 14 alone. Its baseline cannot run, so this file brings its own accuracy checks and reports no speedup. Row 55 |
| `test_backends.py` | Cross-backend comparison: CPU, MPS, MLX |
| `test_padding.py` | Padded and ragged batches, including an empty sample. 18 cases |
| `test_paths.py` | The dispatch paths of the 14 shapes: the plan, an A/B against an all-off plan, and the kernel units. 18 s |

### Probes — one question each

| File | Question it answers |
|---|---|
| `profiling/probes/stage_roofline.py` | Splits one block into stages and names each limit |
| `profiling/probes/sdpa_dispatch.py` | Which `head_dim` values reach the fused SDPA kernel. Row 20 |
| `profiling/probes/alu_peak.py` | What a saturating FMA loop sustains. It sets the 82% ceiling |
| `profiling/probes/zero_copy.py` | Whether the torch output aliases MLX memory. Row 23 |
| `profiling/probes/tile_probe.py` | What a GEMM tile costs, when a fusion forces the tile. It killed row 43 |
| `profiling/probes/tile_resweep.py` | Re-sweeps the steel GEMM tile with the epilogues on, paired against the tile in use. Row 54 |
| `profiling/probes/ln_absorb_probe.py` | The accuracy screen for row 46, against a float64 reference |
| `profiling/probes/ln_tiled_stats_probe.py` | The accuracy screen for row 47: a tiled reduction against a float64 reference |
| `profiling/probes/final_ln_probe.py` | The final LayerNorm fusion at the GEMM alone. Row 50 |
| `profiling/probes/plan_ab.py` | A/B one `KernelPlan` field in one process, interleaved. Use it on a shape under 2 ms, where a sweep cannot decide |
| `profiling/probes/chain_probe.py` | Whether `ffn_in` and `ffn_out` fit in one kernel. Row 44 |
| `profiling/probes/attn_out_budget.py` | Threadgroup budget of a fused attention -> out projection. Row 53 ruled it out on paper |
| `profiling/probes/d_outer_probe.py`, `d_outer_attention.py` | Blocking the head dimension so `head_dim = 256` fits the threadgroup. Row 49 |
| `profiling/probes/pipeline_probe.py` | Whether the framework boundary can hide behind the GPU. It cannot: unified memory makes the two contend. Row 48 |

### Tools

| File | Role |
|---|---|
| `profiling/tools/gpu_timeline.py` | GPU idle read off the Metal timeline, in the terminal. Splits it into head, inner and tail. Row 52 |
| `profiling/tools/harness_sweep.py` | Runs the harness once for each shape as a subprocess, and collects the output |
| `profiling/tools/plot_runtime.py` | The runtime chart of the 13 shapes, on a log axis |
| `profiling/tools/make_chart.py` | The headline figure: the speedup split into its CPU->MPS and MPS->MLX factors, on a log axis |
| `profiling/tools/gpucapture.py`, `trace.sh`, `signposts.py`, `summarize.py`, `Makefile` | Instruments and Metal capture, and the signpost shim |

### Documents

| File | Role |
|---|---|
| `CLAUDE.md` | The rules: the scope, the measurement protocol, and what a run must report |
| `OPTIMIZATIONS.md` | Every optimization tried, with its status and its measurement |
| `agent_loop.md` | The optimization loop: the screens, the four gates, and the run log |
| `profiling/WORKFLOW.md` | How to find the next optimization. Read it first |
| `references/` | Measured facts: the machine, the shapes, the MLX kernels, the scoreboard |
| `references/figures/` | The `plan_kernels()` dispatch tree, and the same tree resolved for shape 6. **No source file exists**, so git tracks the PNGs |
| `references/wwdc/` | Apple GPU talk transcripts, for the machine model |

## Reproduce

    .venv/bin/python3 torch_transformer_benchmark.py     # the harness
    .venv/bin/python3 test_padding.py                    # padded batches
    .venv/bin/python3 test_paths.py                      # the dispatch paths
    .venv/bin/python3 scoreboard.py --cpu-cache --label "..."   # the full sweep
    .venv/bin/python3 demo.py --cases 1,6,13 --cpu-cache        # watch it run
    .venv/bin/python3 shape14_harness.py                 # shape 14. 8.5 minutes
    .venv/bin/python3 profiling/probes/stage_roofline.py --shapes 6

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

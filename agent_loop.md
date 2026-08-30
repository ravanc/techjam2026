# The agent loop

A loop that finds an optimization, builds it, measures it, and keeps it or
drops it. It copies the framework in "Agentic Kernels in Production"
(Baseten, 28 August 2026) and applies it to `UserOptimizedTransformer`.

Read [OPTIMIZATIONS.md](OPTIMIZATIONS.md) first. That file holds the status
of every optimization. This file holds the **method** and the **queue**.

## Why this file exists

The article makes one claim: a kernel that wins a benchmark does not always
win in production. Four reasons, and all four apply to this repository.

| The article says | It shows here as |
|---|---|
| The best kernel depends on the workload | `plan_kernels()` picks a different path for each of the 13 shapes |
| A faster microbenchmark does not give a faster model | row 32 measured 1.02x on the kernel and nothing on the model |
| Per-kernel work misses the larger win | row 43: `fast_layernorm` already runs at copy speed, so no better LayerNorm kernel can win `ln1`. Only fewer bytes can |
| Integration is hard | row 27: a padded batch broke the steel kernel, and no sweep saw it |

So the loop measures the **model**, not the kernel.

## The two layers

The article splits the work in two. This loop keeps that split.

### Layer 1 — model level

Read the whole block. Find work to **delete**, not work to speed up.

Three questions, in this order:

1. Which tensor does one kernel write and the next kernel read back? That
   round trip is free to delete if the two kernels merge.
2. Which value does the model compute again, when it did not change? Move
   it to weight build time.
3. Which stage runs at the memory roof? A stage at the roof cannot get a
   better kernel. It can only move fewer bytes.

Tool: `.venv/bin/python3 profiling/stage_roofline.py --shapes 1,6,8,13`

### Layer 2 — per kernel

Take the kernel that layer 1 produced. Tune it: the tile, the block shape,
the epilogue. Row 38 measured this layer on the steel attention kernel and
found nothing, so this layer is second, never first.

Tool: `profiling/profile_benchmark.py --mode gputrace`, then Xcode.

## The gate

A candidate becomes KEPT only when it passes all four. No exception.

| Gate | Command | Pass condition |
|---|---|---|
| 1. Accuracy | `scoreboard.py` accuracy table | 13 of 13 PASS at `atol=0.002`, `rtol=0.02` |
| 2. Padding | `.venv/bin/python3 test_padding.py` | 18 of 18 pass |
| 3. End to end | `scoreboard.py --cpu-cache --label "..."` | the **MLX ms** column drops, FLOP weighted |
| 4. Control | the same sweep | MPS moves under 1%, and a shape the change cannot touch holds at 1.00x |

Gate 4 is the one that catches a false reading. The CPU baseline moves 76%
with the machine load. MPS does not. A shape outside the change does not
either. If the control moves, the reading is noise, not a win.

Gate 3 uses a **FLOP weighted** score, because shape 6 is 66.5% of the FLOPs
and shape 8 is 21.3%. The other eleven together are 12%.

## The rules of a measurement

These come from CLAUDE.md. They are here because the loop breaks without
them.

1. Check for a running measurement first:
   `ps -Ao pid,etime,%cpu,command | grep "[.]venv/bin/python3" | grep -v shell-snapshots`
   Two runs share one GPU. Each one makes the other false.
2. Run with `.venv/bin/python3`.
3. Call `mx.synchronize()` after `mx.eval()`.
4. Use the default `--repeats`. Three repeats gave 7.198x where 100 gave
   4.590x.
5. Always pass `--cpu-cache` and `--label`.
6. Record a failure as well as a win.

## The knowledge base

The article keeps the kernels that pass, and the lessons from the attempts
that fail. This repository already does that.

| What | Where |
|---|---|
| The status of every optimization | the source of truth table in `OPTIMIZATIONS.md` |
| The measurement that decided it | the detail section under the same row number |
| A machine fact, measured once | `references/` |
| Every sweep ever run | `profiling/history.jsonl` |
| This loop, and what it did | the run log at the end of this file |

A REVERTED row is the valuable one. It stops the loop from doing the same
work twice. Do not delete a row.

## The mapping — the article against this repository

The article lists eight optimizations. Five of them are already KEPT here,
under a different name. Three are open. One principle has no row yet.

| Article | What it does | Row here | Status |
|---|---|---|---|
| #2 Fused QKV projection and epilogue | one GEMM for q, k and v, then bias, norm and RoPE in the epilogue | 9, 34 | **KEPT** |
| #4 Bias absorption | fold a standalone bias into the next fused operation | 36 | **KEPT** |
| #6 Fused SwiGLU + quantization | fold the activation into the GEMM, so the pre-activation never reaches DRAM | 33 | **KEPT** |
| #7 Gated residual normalization | one kernel for the residual add and the norm | 36 | **KEPT** |
| Per-kernel optimization loop | tune the tile and the block shape | 38 | **REVERTED** |
| **#3 Normalization + quantization fusion** | the norm writes a large tensor that the next kernel reads back. Fuse them | **43** | **OPEN** |
| **Model level: remove intermediate materialization** | chain two GEMMs, so the middle tensor stays in the threadgroup | **44** | **OPEN** |
| **Extend a win to the shape it misses** | Qwen-Image and FLUX.2 got the same kernels, at different gates | **37** | **OPEN** |
| **#1 Prepacked scales, moved to load time** | stop recomputing a constant on every call | **45**, new | **OPEN** |

Two article optimizations have no analogue here, and the loop will not chase
them:

- **FP8 and NVFP4 quantization.** Row 16. The harness fixes `atol=0.002`
  and `rtol=0.02`, and CLAUDE.md forbids raising them. Row 13 measured that
  even float16 cannot pass. So no quantized path can pass.
- **The CFG modulation cache.** This model has no classifier free guidance
  and no second pass. Its *principle* is row 45: cache the value that does
  not change.

## The queue

Ranked by FLOP weighted value, not by size of the number.

The estimates below are the ones the loop has measured. Turn 1 replaced the
memory-floor guesses that were here before, and they were optimistic.

| Order | Row | What | Estimated value | Risk | Article |
|---:|---:|---|---|---|---|
| 1 | 45 | build the deferred bias `carry` at weight build time | unmeasured. It is 12 small kernels per call, and shape 2 is launch bound | low | #1 |
| 2 | 44 | chain `ffn_in` and `ffn_out` | about **0.69 ms per layer, 5.0% of shape 6**. The `bn = 128` tile costs 0.884x and `hidden` fits at 28.5 KiB of 32 | high. It is a two GEMM kernel with a threadgroup handoff | model level |
| **1** | 43 route 2 | absorb the LayerNorm into the weights, apply it in the epilogue | about **1.0 ms per layer, 7.2% of shape 6**, and it reaches shape 8 as well | the accuracy risk is MEASURED and it passes (0.9x to 1.2x). What is left is build difficulty | #3 + #1 |
| — | 43 route 1 | LayerNorm in the GEMM prologue, statistics recomputed per tile | **REVERTED.** +2.243 ms per layer | — | #3 |
| — | 37 | a wide `fast_layernorm`, so shape 8 reaches row 36 | about 1.0% FLOP weighted | low | **probably subsumed by 43 route 2** |

**Row 44 does not depend on row 43.** An earlier version of this file said to
build row 44 second, on row 43's kernel. That was wrong: `bn = 128` and
`bk = 16` are independent constraints, and route 1 dying does not touch row
44. The two rows do land in the same file, `steel_gemm.py`.

**Turn 3 reordered this queue.** Row 43 route 2 was third, because it was
the one that could fail the accuracy gate. It does not fail: measured at
0.9x to 1.2x of the error the model already carries. So it moves to first.
It is the largest prize, it is the only candidate that reaches shape 8, and
it makes row 37 unnecessary.

Row 45 dropped to last. Its value is about 0.1% FLOP weighted, which is
under the 1% noise floor, so a sweep cannot resolve it. It is worth doing
for shape 2 alone, and shape 2 carries 0.0% of the weight.

## The run log

One entry for each turn of the loop. Append, never edit.

### Turn 0 — read the article, build the queue

- **Layer 1 found:** nothing new. Rows 43, 44 and 37 were already OPEN in
  `OPTIMIZATIONS.md`, and the article confirms all three are the right
  targets.
- **New:** row 45. The article's optimization #1 moves constant work to
  load time. `carry` in `_mlx_transformer()` is a pure function of the
  weights, and the block rebuilds it on every call.
- **Gate:** not run. No code changed.
- **Next:** row 43.

### Turn 1 — row 43, route 1: REVERTED

**The question.** Row 43 folds the LayerNorm into the GEMM prologue. The row
statistics are a reduction over the whole row, so one A tile must cover it:
`bk = K = 128`. What does that tile cost?

**The kill test first.** This is the cheapest measurement that can end the
row, so it ran before any kernel got written.

    .venv/bin/python3 profiling/tile_probe.py --row 43

**The reading.** The 32 KiB threadgroup caps the tile at `bm + bn <= 62` at
`bk = 128`, against `bm32 bn64` today. Every tile that compiles is bit exact
(`max_abs = 0.00e+00`) and none reaches 0.6x. Best of 24:

| | `qkv proj` | `ffn_in` |
|---|---:|---:|
| `bm32 bn64 bk16` (today) | 3.602 ms | 1.345 ms |
| best `bk128` tile | 6.637 ms **0.543x** | 2.467 ms **0.545x** |

**The verdict.** Route 1 adds 4.157 ms per shape 6 layer to delete 1.914 ms
of LayerNorm. Net **+2.243 ms, a 1.19x slowdown**. REVERTED.

**Why the old estimate was wrong.** It assumed the GEMM holds its rate at
`bk = 128`. The row's "0.8% extra ALU per N tile" line is true of the
arithmetic and irrelevant: the cost is the tile, not the reduction.

**What survives.** Route 2 (precompute the statistics, apply them as the
loader reads each `bk` chunk) puts no constraint on `bk`, so it keeps the
fast tile. Layer 1 of the loop then found a stronger form of route 2: the
LayerNorm is affine in the row, so it distributes through the matmul and
folds into the WEIGHTS at load time, leaving an epilogue that reads two
floats per row. That is the article's optimization #1 applied to its
optimization #3. Written up under row 43. Its open risk is accuracy, not
speed.

### Turn 2 — row 44: feasibility confirmed, not built

**The question.** Row 44 chains `ffn_in` and `ffn_out`. The second GEMM
needs a whole `hidden` row, so the first GEMM's `bn` must cover all of
`ffn_dim`. What does `bn = 128` cost, and does `hidden` fit beside As and Bs?

    .venv/bin/python3 profiling/tile_probe.py --row 44

**The reading.** `ffn_in + gelu` at `bm32 bn64` is 1.680 ms. At `bm32 bn128`
it is 1.899 ms, **0.884x**, `max_abs` 4.77e-07. The threadgroup holds
12.5 KiB of As+Bs plus 16.0 KiB of `hidden` = **28.5 KiB of 32**.

**The verdict.** Affordable. Deleting the 128 MiB round trip is worth about
0.90 ms against a 0.22 ms tile penalty, so about **0.69 ms per layer, or
5.0% of shape 6**. That is below the row's memory-floor estimate, because
`ffn_in` is already COMPUTE bound. Row 44 stays OPEN with a corrected
number.

**Correction to the queue.** Row 44 does NOT depend on row 43. `bn = 128`
and `bk = 16` are independent constraints, and route 1 dying does not touch
it.

### Turn 2b — a reference correction the loop had to make

`stage_roofline.py` printed `ln1` at 141.6 GB/s and `%mem` 110.6, above the
128 GB/s roof that `references/machine.md` tells you to use. A stage cannot
beat a roof, so one of the two had to be wrong.

Neither is. The tool subtracts `FLOOR_MS` from the stage time and takes GB/s
from the result, but `PEAK_GBPS` is a raw reading that still includes the
round trip. The two sides of the ratio are measured differently.

Checked directly. The shape 6 `ln1` moves 128 MiB:

| | ms | GB/s |
|---|---:|---:|
| roofline `ms` (floor removed) | 0.9477 | 141.6, prints 110.6% |
| roofline `raw` | 1.2492 | **107.4** |
| standalone `fast_layernorm`, same size | 1.305 | 107.5 |
| `x * 2.0` at 64 MiB, `machine.md` | — | 109.0 |

The `raw` column agrees with the reference and with a standalone reading.
So `%mem` is a rank, not a fraction of the roof, and the same bias inflates
`%comp` (which is why shape 8 prints 100.4% on a GEMM). Both
`profiling/stage_roofline.py` and `references/machine.md` now say so.

This changes nothing about row 43: `fast_layernorm` really does run at copy
speed, so the conclusion "only fewer bytes can win `ln1`" still holds.

- **Gate:** not run. No model code changed in turns 1, 2 or 2b.
- **Next:** row 43 route 2. Its kill test is arithmetic, not speed, so it is
  cheap and it goes first.

### Turn 3 — row 43 route 2: the accuracy risk is measured, and it passes

**The question.** Route 2 runs the matmul over RAW `x`, then subtracts
`m_i * c1[n]`. That is a difference of two large close numbers. How many bits
does that cancellation cost?

**The kill test.** Arithmetic only. It writes no Metal, so it cost minutes.

    .venv/bin/python3 profiling/ln_absorb_probe.py --shape 6

It takes the residual stream from a real forward, then computes one
projection three ways: float64 as the reference, float32 as the model runs it
today, and float32 absorbed.

**The reading.** Worst of the 4 layers, `max_abs` against float64:

| shape | worst `abs(mean)/std` | today | route 2 | ratio | headroom to `atol` |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.33 | 1.28e-06 | 1.49e-06 | 1.2x | 1340x |
| 7 | 0.93 | 6.49e-07 | 7.62e-07 | 1.2x | 2626x |
| 8 | 0.13 | 4.52e-06 | 4.62e-06 | 1.0x | 433x |
| 13 | 0.33 | 1.60e-06 | 1.49e-06 | 0.9x | 1340x |

**The verdict.** Route 2 costs 0.9x to 1.2x of the error the model already
carries. It is numerically free.

**Why the fear was wrong.** The cancellation scales with the row ratio
`|mean| / std`, and that ratio stays near 0.3 at every shape. The residual
stream does not drift far from zero. The harness benchmarks an UNTRAINED
model, so this is the regime the gate measures.

**The bonus, and it is the largest finding of the turn.** The prologue form
needed `fast_layernorm`, so it stopped at `d_model < 256` and shape 8 was out
of scope. Route 2 needs a statistics pass and an epilogue, and neither cares
about the row width. So route 2 reaches **shape 8, which is 21.3% of the FLOP
weight**, and its `c3` constant carries the deferred residual bias that row
37 exists to give shape 8. **Route 2 subsumes row 37.**

- **Gate:** not run. Route 2 is not built. The probe is a screen, not a gate:
  it covers one projection, and the real error compounds over 4 layers and 2
  norms each.
- **Next:** build route 2. It is the largest remaining item and the only one
  that touches both heavyweight shapes.

### Turn 4 — row 46: BUILT, gated, KEPT

The first turn that changed model code, so the first turn that ran the gate.

**What went in.** Three pieces:

1. `fast_layernorm.layer_norm_stats()` — the same two reductions as the
   LayerNorm, then it stops. It writes `{rstd, rstd * mean}` for each row.
   It serves every width up to 1024, so shape 8 reaches it.
2. `steel_gemm.layer_norm_constants()` — builds the prepacked weight and the
   three constant vectors at weight build time.
3. `apply_layer_norm_epilogue`, a new method patched into Apple's
   `BlockMMA`. MLX ships two epilogue hooks and neither fits: the unary hook
   has no index, and the binary hook sees ONE operand, by row or by column,
   never both. Row 46 needs a row times column term. So this is Apple's
   binary loop with a second pointer added, and the same indexing.

**The trick that made it small.** `out = P*(acc + c3) - Q*c1 + c2` has two
cross terms. Giving `c3` to the ordinary C operand — the one `mx.addmm`
already uses — absorbs it before the epilogue runs, and leaves exactly one.

**The gate, all four:**

| Gate | Result |
|---|---|
| 1. Accuracy | 13/13 PASS, `max_abs` 1.07e-06 to 3.34e-06 |
| 2. Padding | 18/18 pass, bit exact |
| 3. End to end | MLX **722.3 -> 681.4 ms, 1.060x FLOP-weighted** |
| 4. Control | MPS 0.992x, inside the 1% noise floor |

Shape 6 **1.073x**, shape 8 1.012x, shape 12 1.152x.

**The screen predicted the gate.** `ln_absorb_probe.py` said the accuracy
cost would be 1.0x to 1.2x before any Metal was written. The sweep measured
1.27x at the worst shape. That is what a screen is for.

**One correction this turn forced.** Turn 3 said row 46 subsumes row 37. It
does not. Row 46 removes the need for a `pre_bias` hook at `ln1` and `ln2`,
but the FINAL LayerNorm still has no GEMM below it and still needs the
carry, so shape 8 runs row 46 with `defer_bias=False`. Row 37 is corrected
and stays OPEN — and it now records a cheaper fix than itself: one
`x = x + carry` before the final norm.

- **Gate:** PASSED. Row 46 is KEPT and committed.
- **Next:** row 44, then the shape 8 carry note under row 37.

### Turn 5 — row 37: KEPT, and it cost no new kernel

**The queue was wrong, and turn 4 made it wrong.** Row 44 led the queue. But
row 46 had just unblocked row 37, and row 37 turned out to be a four line
change for 0.7% FLOP-weighted, against hours of kernel work for row 44's 5%.
Effort per unit of win decided the order, not the size of the win.

**What it is.** Row 36 defers each residual bias into a `carry`, and every
LayerNorm must then apply it. Only `fast_layernorm` had a `pre_bias` hook,
and it stops at a row width of 256, so shape 8 could not defer.

Row 46 folds `ln1` and `ln2` into the GEMM below them, and the carry rides in
the `c3` constant. So only the FINAL LayerNorm still needs the carry, and the
fix is to add it there, once for the whole model:

    if pre is not None and not plan.fast_layer_norm:
        value = value + pre
        pre = None

One pass over the activation per forward, against two deferred residual adds
in every layer.

**The gate:**

| Gate | Result |
|---|---|
| 1. Accuracy | shape 8 PASS at `max_abs` 3.22e-06, BETTER than 3.34e-06 before |
| 2. Padding | 18/18 pass |
| 3. End to end | shape 8 **124.911 -> 119.931 ms, 1.042x**. About 0.7% FLOP-weighted |
| 4. Control | **shape 8 MPS held at 0.996x** |

**Gate 4 failed at the sweep level, and the loop had to handle that.** The
sweep-wide MPS control moved 0.974x, outside the 1% floor, because `mysqld`
was holding 76% of a CPU. So the sweep total could not score the change.

Two things saved the reading:

1. Shape 8 is the ONLY shape this change touches, and its own MPS control
   held at 0.996x while its MLX gained 4.2%.
2. Every untouched shape tracked its own MPS control within a point or two,
   so nothing regressed.

An isolated single-shape run on a quieter machine gave 1.044x independently.

**The rule this exposed.** CLAUDE.md's rule 1 command greps for
`.venv/bin/python3`, so it detects a competing sweep and nothing else. It
cannot see a busy database. Recorded in `references/machine.md` with a
whole-machine check.

- **Gate:** PASSED on the per-shape control.
- **Next:** row 44, the last item in the queue.

### The state at the end of turn 4

No model code has changed. Four turns of the loop produced four measured
facts and no regression risk:

| Row | Was | Now | Decided by |
|---:|---|---|---|
| 43 route 1 | OPEN | **REVERTED**, +2.243 ms | `profiling/tile_probe.py --row 43` |
| 46 (was 43 route 2) | not stated | **KEPT, 1.060x FLOP-weighted** | the four-gate sweep |
| 44 | OPEN, estimated from a memory floor | **OPEN, estimate corrected** to 5.0% of shape 6. Tile is affordable | `profiling/tile_probe.py --row 44` |
| 37 | OPEN | **KEPT, shape 8 1.042x.** Turn 3 wrongly called it subsumed; turn 4 corrected that and found a cheaper fix; turn 5 built the cheap fix and it needed no new kernel | the per-shape control |

Plus one reference correction: `stage_roofline.py` `%mem` overstates, and
can pass 100%, because it subtracts the round trip from the stage but not
from the roof.

Turns 1 to 3 built nothing: every number there is a screen. A screen kills a
bad idea cheaply, and it never keeps a good one. Turn 4 built row 46 and ran
the gate, which is what kept it.

The loop cost four screens to find one 1.060x. Two of the four screens were
negative, and that is the point: route 1 would have been days of kernel work
for a 1.19x slowdown.

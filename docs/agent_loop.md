# The agent loop

A loop that finds an optimization, builds it, measures it, and keeps it or
drops it. It copies the framework in "Agentic Kernels in Production"
(Baseten, 28 August 2026) and applies it to `UserOptimizedTransformer`.

Read [OPTIMIZATIONS.md](../OPTIMIZATIONS.md) first. That file holds the status
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

Tool: `.venv/bin/python3 profiling/probes/stage_roofline.py --shapes 1,6,8,13`

### Layer 2 — per kernel

Take the kernel that layer 1 produced. Tune it: the tile, the block shape,
the epilogue. Row 38 measured this layer on the steel attention kernel and
found nothing, so this layer is second, never first.

Tool: `profiling/tools/profile_benchmark.py --mode gputrace`, then Xcode.

## The gate

A candidate becomes KEPT only when it passes all four. No exception.

| Gate | Command | Pass condition |
|---|---|---|
| 1. Accuracy | `scoreboard.py` accuracy table | 13 of 13 PASS at `atol=0.002`, `rtol=0.02` |
| 2. Padding | `.venv/bin/python3 tests/test_padding.py` | 18 of 18 pass |
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
| Every sweep ever run | `profiling/results/history.jsonl` |
| This loop, and what it did | the run log at the end of this file |

A REVERTED row is the valuable one. It stops the loop from doing the same
work twice. Do not delete a row.

## The mapping — the article against this repository

The article lists eight optimizations. Read the status column of
`OPTIMIZATIONS.md` before you trust this table; the table below is a map from
the article to a row number, not a second source of truth.

| Article | What it does | Row here | Status |
|---|---|---|---|
| #2 Fused QKV projection and epilogue | one GEMM for q, k and v, then bias, norm and RoPE in the epilogue | 9, 34 | **KEPT** |
| #4 Bias absorption | fold a standalone bias into the next fused operation | 36 | **KEPT** |
| #6 Fused SwiGLU + quantization | fold the activation into the GEMM, so the pre-activation never reaches DRAM | 33 | **KEPT** |
| #7 Gated residual normalization | one kernel for the residual add and the norm | 36 | **KEPT** |
| Per-kernel optimization loop | tune the tile and the block shape | 38 | **REVERTED** |
| **#3 Normalization + quantization fusion** | the norm writes a large tensor that the next kernel reads back. Fuse them | 43 REVERTED, then **46** | **KEPT, 1.060x.** Row 47 takes the same principle one level further and is OPEN |
| **Model level: remove intermediate materialization** | chain two GEMMs, so the middle tensor stays in the threadgroup | **44** | **REVERTED.** Built, correct, 0.969x |
| **Extend a win to the shape it misses** | Qwen-Image and FLUX.2 got the same kernels, at different gates | **37** | **KEPT.** Shape 8 1.042x, and it needed no new kernel |
| **#1 Prepacked scales, moved to load time** | stop recomputing a constant on every call | **45**, new | **OPEN**, and mostly delivered by row 46's weight build. What is left is 0.1% FLOP-weighted |

Two article optimizations have no analogue here, and the loop will not chase
them:

- **FP8 and NVFP4 quantization.** Row 16. The harness fixes `atol=0.002`
  and `rtol=0.02`, and CLAUDE.md forbids raising them. Row 13 measured that
  even float16 cannot pass. So no quantized path can pass.
- **The CFG modulation cache.** This model has no classifier free guidance
  and no second pass. Its *principle* is row 45: cache the value that does
  not change.

## The queue

**SUPERSEDED. The live queue is "The queue, after turn 7" at the end of this
file.** This section is what the loop believed at turn 3, and the run log
below shows how each row was decided. Keep it for the reasoning, not for the
order.

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

    .venv/bin/python3 profiling/probes/tile_probe.py --row 43

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

    .venv/bin/python3 profiling/probes/tile_probe.py --row 44

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
`profiling/probes/stage_roofline.py` and `references/machine.md` now say so.

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

    .venv/bin/python3 profiling/probes/ln_absorb_probe.py --shape 6

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

### Turn 6 — row 44: built, correct, REVERTED

The last item in the queue, and the only substantial one left.

**What it did.** Kept `hidden` in threadgroup memory so `ffn_out` reads it
from there instead of DRAM. That deletes a 128 MiB round trip, about 1.1 ms
of a 12.0 ms shape 6 layer.

**The screens passed.** All three prerequisites were checked before a line
was written: `BlockMMA::mma()` already takes a threadgroup A pointer,
`MMATile::store` has a threadgroup overload, and the budget fits at 29.0 KiB
of 32 with phase 2's `Bs2` aliased onto the dead phase 1 buffers.

**The gate failed on speed.** Best of 40 configurations is **0.969x**, and a
repeat gave 0.960x. Nothing wins.

**Why.** One threadgroup must own all of `ffn_dim` to hold the row, and that
is what makes the kernel threadgroup-hungry. Threadgroup memory limits
residency. The controlled pair, at a fixed `bm = 32`:

| | threadgroup | best |
|---|---:|---:|
| `bk = 8` | 24.0 KiB | 0.996x |
| `bk = 16` | 29.0 KiB | 0.896x |

Same tile, more threadgroup memory, worse time. And a small `bm` lowers the
memory but wastes the tile: `bm8` reaches 0.694x. The kernel is squeezed
between a tile too small to be efficient and a threadgroup too large to be
resident.

**The screen under-predicted the loss, and that is worth recording.**
`tile_probe.py` measured the `bn = ffn_dim` tile at 0.884x. The full kernel
came in at 0.969x — BETTER than the tile alone, because the deleted round
trip does pay for part of it. So the screen was directionally right about
the cost and could not see the occupancy term. A screen bounds one effect.
It does not predict a kernel.

**Where the code went.** `profiling/probes/chain_probe.py`, not the model. It is a
working kernel that loses, so it belongs with the measurement, not in
`steel_gemm.py`. `store_result_tgp` stays in the mma patch, marked as used
by the probe alone.

- **Gate:** FAILED on gate 3. Row 44 is REVERTED.
- **Next:** the queue is empty. See below.

### The queue is empty

Every row that was OPEN when this loop started is now decided:

| Row | Outcome |
|---:|---|
| 43 | REVERTED. The prologue forces `bk = K` and the tile costs 0.543x |
| 46 | KEPT. **1.060x FLOP-weighted** |
| 37 | KEPT. Shape 8 **1.040x**, and it needed no new kernel |
| 44 | REVERTED. Built, correct, 0.969x |

What remains in `OPTIMIZATIONS.md` is out of reach rather than untried:

- **16** quantization cannot pass `atol = 0.002`. Row 13 measured that even
  float16 fails.
- **21** MLX never calls its own `bd192` and `bd256` kernels, and Python
  cannot reach around it. Recheck after an MLX upgrade.
- **19, 22** apply to no appendix shape.

So the next real move is not another row. It is one of:

1. Re-measure the roofline and look for a stage that moved. Rows 46 and 37
   changed the block, and layer 1 of this loop reads that table.
2. Fix `PROVISIONAL_PEAK_TFLOPS` in `flops.py`. Every MFU number scales with
   a constant that was asserted from memory and never checked. The graded
   score depends on it.

Item 2 is the larger one. The MFU is the graded metric and its denominator
is currently unverified.

### The state at the end of turn 4

No model code has changed. Four turns of the loop produced four measured
facts and no regression risk:

| Row | Was | Now | Decided by |
|---:|---|---|---|
| 43 route 1 | OPEN | **REVERTED**, +2.243 ms | `profiling/probes/tile_probe.py --row 43` |
| 46 (was 43 route 2) | not stated | **KEPT, 1.060x FLOP-weighted** | the four-gate sweep |
| 44 | OPEN, estimated from a memory floor | **OPEN, estimate corrected** to 5.0% of shape 6. Tile is affordable | `profiling/probes/tile_probe.py --row 44` |
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

### Turn 7 — the roofline re-measure, and the queue refills

The queue was empty, so the loop ran the first of the two moves it left
itself: re-measure the stage roofline, because rows 46 and 37 changed the
block and layer 1 of this loop reads that table.

    .venv/bin/python3 profiling/probes/stage_roofline.py --shapes 1,6,8,13

**The tool crashed first, and the crash is itself a finding.** Shape 8 raised
`TypeError` in `do_norm()`. Row 37 changed `plan_kernels()` so that
`defer_bias` no longer implies `fast_layer_norm`, and shape 8 is the one
shape that now defers WITHOUT it. `mx.fast.layer_norm` takes no `pre_bias`.
The model was correct; the profiler still held the old call. Fixed by
copying the model's own branch into the tool. **A tool that follows the plan
must be updated in the same change as the plan.**

**What the block looks like now.** Shape 6, floor 0.3214 ms:

| stage | ms | raw | limit | %roof |
|---|---:|---:|---|---:|
| qkv proj (+layer norm) | 3.321 | 3.642 | COMPUTE | 95.6 |
| sdpa | 2.021 | 2.342 | IO | 103.8 |
| out proj (+residual) | 1.494 | 1.815 | IO | 105.3 |
| ffn_out (+residual) | 1.435 | 1.756 | IO | 109.7 |
| ffn_in + gelu (+layer norm) | 1.278 | 1.599 | COMPUTE | 82.8 |
| ln1 stats + ln2 stats | 0.668 | 1.311 | IO | at the roof |

**Every stage of shape 6 now sits at a roof.** The only stage under 90% of
its own roof is `ffn_in` at 82.8% of compute. Shape 8 is stronger still: four
GEMMs at 99-101% of the matmul peak carry 25.2 ms of its 30.18 ms layer.

So layer 2 of this loop (tune a kernel) has nothing left to tune. Question 3
of layer 1 answers the whole table: a stage at the roof can only get faster
by moving fewer bytes.

**The one stage that can lose bytes: row 47, NEW.** The two
`layer_norm_stats` passes read 65.0 MiB each to write two floats for each
row, and every byte they read was written by a GEMM one stage earlier
(`ffn_out` of the layer below for `ln1`, `out proj` of the same layer for
`ln2`). Put the reduction in that GEMM's epilogue and the read disappears:
**5.5% of shape 6, 0.6% of shape 8, about 3.8% FLOP-weighted.** Written up
as row 47, with the tiled-partials design that avoids both `bn = ffn_dim`
(row 44 killed it) and float atomics (the padding gate checks bit equality).

**Row 45 was OPEN in this file and MISSING from the source of truth table.**
CLAUDE.md says the table is the only place that states a status. Added, with
the correction that row 46 already delivered most of it at weight build time.

**Three reference corrections, all forced by the readings:**

1. **The floor is not 0.13 to 0.17 ms and it is not a constant.** Three runs
   minutes apart gave 0.3049, 0.1468 and 0.3214 ms, while `mysqld` held 96%
   of one CPU. The round trip is CPU-side work. `references/machine.md` now
   says 0.15 to 0.32 ms and says to measure it in the run that uses it.
2. **`raw` is the reproducible column, `ms` is derived.** The floor comes off
   every stage, so a sub-floor stage is noise: shape 8 `ln1 stats` read
   0.2602 ms in one run and 0.0691 ms in the next, from the floor alone,
   which is where the 380% `%mem` readings come from. Every stage above 1 ms
   repeated within 5%, and shape 8 `qkv proj` within 0.1%.
3. **"The sum of stages is larger than the real layer" is false as written.**
   The two sums BRACKET the real layer, and the bracket held at all four
   shapes:

   | shape | `ms` sum | real per layer | `raw` sum |
   |---:|---:|---:|---:|
   | 1 | 0.438 | 0.846 | 2.326 |
   | 6 | 10.215 | 12.070 | 12.552 |
   | 8 | 28.042 | 30.180 | 30.664 |
   | 13 | 8.574 | 10.381 | 10.917 |

   The `ms` sum subtracts the floor once for each of 9 stages; the real model
   pays it once for the forward. So `ms` under-counts by about 9 x floor.
   **There is no unexplained gap in the shape 6 layer.** An earlier reading
   of the same table looked like 2.0 ms of missing work, and it was the floor
   accounting, not missing work.

- **Gate:** not run. No model code changed. `stage_roofline.py`, `WORKFLOW.md`
  and `machine.md` changed, and none of the three is in the forward path.
- **Next:** the queue below.

### Turn 8 — row 47: BUILT, gated, KEPT

**1.075x FLOP-weighted.** MLX 686.6 ms to 638.7 ms over the 13 shapes.
Shape 6 1.091x, shape 8 1.027x, shape 13 1.045x, against a 1.006x median MPS
control. The queue's estimate was 3.8%; the measurement is 7.5%, because the
estimate used the floor-subtracted `ms` column and the stage is really
1.311 ms raw of a 12.552 ms shape 6 layer.

**Two screens went first, and both were cheap.**

1. `profiling/probes/ln_tiled_stats_probe.py`, the accuracy screen. A tile cannot
   centre against a mean it does not have, so the epilogue must use the
   uncentred variance, which `fast_layernorm` refuses for a whole row. The
   screen measured the residual drift of this model at **0.12 to 0.33**,
   far below the cancellation regime, and the uncentred form then gives the
   same projection error as today on shapes 6, 8 and 13. It also measured
   Chan's stable combine and found it slightly WORSE at shape 8, so the
   build took the simple form.
2. A speed screen on `steel_addmm` with a matrix C operand. `out proj` and
   `ffn_out` ran `mx.addmm`, so they had to move to the hoisted GEMM before
   an epilogue could reach them. They tie (1.009x, 0.988x, 1.039x) and stay
   bit exact, which repeats row 40's proof that MLX dispatches `addmm` to
   this same kernel.

**What row 44 taught, and this row used.** Row 44 died on threadgroup
memory. This epilogue spends none: the four lanes of one MMA fragment row
are `lane`, `lane ^ 1`, `lane ^ 8` and `lane ^ 9`, so two `simd_shuffle_xor`
steps reduce a row with no barrier.

The queue is empty again except for row 45, which is under the noise floor.

### Turn 9 — row 48: built, correct, REVERTED

The block is at its roofs, so this turn looked OUTSIDE the block, at
`forward()`. Nothing had re-read it since row 23, and its share had tripled:
the kernels got 3.7x faster and the framework boundary did not. A fresh
breakdown put **4.5% FLOP-weighted outside the kernels**, more than any row
still OPEN.

The prototype converts chunk `i+1` while the GPU runs chunk `i`. It is bit
equal and it gives **0.974x**.

**It fails on the memory system, not on the code.** A controlled arm adds an
UNRELATED 625 MiB CPU memcpy to the chunk loop. That memcpy costs 31.40 ms
alone and **+45.06 ms inside the loop**: overlapping is worse than serial.
Unified memory gives the CPU and the GPU one controller, and the shape 6
block already runs at 105% and 110% of the memory roof, so every byte the
CPU moves is taken from the GPU.

This rules out the whole class: double buffering, a copy stream, a background
conversion thread. Do not spend another turn on it.

**The lesson.** On a discrete GPU the transfer crosses PCIe while the GPU
reads VRAM, so overlap is free. Do not carry that instinct to this machine.
Before proposing any overlap here, ask which memory system each side uses.

## The queue, after turn 9

| Order | Row | What | Estimated value | Risk |
|---:|---:|---|---|---|
| 1 | 45 | Build the deferred bias `carry` at weight build time | about 0.1% FLOP-weighted, under the noise floor | low. Row 46 already did most of it |

Rows 47 and 48 are done. Rows 16, 19, 21 and 22 stay out of reach: 16 cannot
pass the tolerance, 21 waits for an MLX upgrade, and 19 and 22 apply to no
appendix shape.

**There is no queue left with a measured prize.** Every stage of the block is
at a roof, and row 48 closed the one region outside it. The next turn has to
find a NEW prize, not work an old one: run `profiling/probes/stage_roofline.py`
again, and read every row EXCEPT `ln1 stats` and `ln2 stats`, which row 47
made stale.

## The queue, after turn 7

| Order | Row | What | Estimated value | Risk |
|---:|---:|---|---|---|
| 1 | — | Fix `PROVISIONAL_PEAK_TFLOPS` in `flops.py` | no speed at all. It is the GRADED metric, and every MFU scales with a constant asserted from memory and never checked | none. It is a source lookup, not a kernel |
| 2 | 47 | Produce the LayerNorm statistics in the producing GEMM's epilogue | **3.8% FLOP-weighted**, upper bound | high. A row spans several N tiles, so the reduction crosses threadgroups. Screen the accuracy first |
| 3 | 45 | Build the deferred bias `carry` at weight build time | about 0.1% FLOP-weighted, under the noise floor | low. Row 46 already did most of it |

Item 1 leads on value for the grade, not on speed. Items 2 and 3 are the only
rows in `OPTIMIZATIONS.md` that are OPEN and reachable: 16 cannot pass the
tolerance, 21 waits for an MLX upgrade, and 19 and 22 apply to no appendix
shape.

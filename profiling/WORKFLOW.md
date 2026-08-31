# How to use the profiler for optimization

This guide tells you how to find slow code and how to make it faster.
For the setup steps and the platform problems, read `README.md`.

## The five tools

### 1. The timeline — find the slow model

```
./profiling/tools/trace.sh
.venv/bin/python3 profiling/tools/summarize.py profiling/traces/benchmark.trace
```

The summary prints the median, the mean, the minimum, and the maximum time
for each region. It groups the times by model.

```
region                      n   median ms    mean ms    min ms    max ms
------------------------------------------------------------------------
baseline-iter              20     20.1336    20.1265   19.5421   20.7394
optimized-iter             20     14.7423    14.8990   14.5080   16.1950
```

Use this tool first. It shows you where to look.

### 2. The kernel view — find the slow GPU kernel

```
.venv/bin/python3 profiling/tools/profile_benchmark.py --mode gputrace --iterations 3
open profiling/traces/optimized_mlx.gputrace
```

Xcode shows the time for each GPU kernel. The shader profiler shows the cost
of each instruction. It tells you if a kernel waits for memory or for math.

### 3. The stage roofline — find the slow stage, and its limit

```
.venv/bin/python3 profiling/probes/stage_roofline.py --shapes 1,6,8,13
```

It splits one transformer block into eleven stages, times each one alone,
and names what limits it: the arithmetic units, the memory system, or the
kernel launch. For each stage it gives the FLOPs, the compulsory DRAM
traffic, the arithmetic intensity, and the two achieved rates against the
measured roofs (4.06 TFLOP/s and 128 GB/s, ridge 31.7 FLOP/byte).

It subtracts the `mx.eval` + `mx.synchronize` round trip, which is 0.15 to
0.32 ms and dominates every small shape. A stage that does not clear the
floor prints `at floor` and is marked LAUNCH.

**The floor moves between runs, and the tool measures it once per run.**
Three runs on 30 August 2026 gave 0.3049, 0.1468 and 0.3214 ms. Every stage
loses that number, so a stage under about 0.5 ms is not reproducible in the
`ms` column: the shape 8 `ln1 stats` read 0.2602 ms and then 0.0691 ms from
the floor alone. The `raw` column holds: every stage above 1 ms repeated
within 5%, and the shape 8 `qkv proj` within 0.1%. **Rank by `raw`.**

**The tool times ONE LAYER. It never sees the final LayerNorm.** That
LayerNorm runs once for the whole forward, outside the layer, so no row of
the table holds it and the `real per layer` line spreads it over every layer.
It cost 1.2478 ms for each shape 6 chunk, which is 2.8% of the shape, and
four rows of OPTIMIZATIONS.md named it without measuring it. Row 50 now folds
it into the last `ffn_out`. **Look outside the layer as well as inside it.**

**The `ln1 stats` and `ln2 stats` rows are stale since row 47.** The tool
builds a block from the plan and times each stage alone, and it does not
model the statistics epilogue. A shape that selects row 47 no longer runs
those two passes at all: the GEMM above each one produces the statistics in
its own epilogue. Read the two rows as the prize row 47 took, not as current
cost. Every other row still describes what the model runs.

Two extra diagnostics print under each table:

- **SDPA peak memory** against the operands alone. It says whether the
  attention kernel wrote the `B x H x S x S` score matrix to DRAM.
- **Strided against contiguous operands.** The head layout is a transpose,
  and an MLX transpose is a free view, so the layout cost hides inside the
  attention kernel call. This pair separates the two. **Read the gap
  carefully.** It does not prove that the kernel reads the stride badly. It
  can also be a copy that the kernel launch runs for you: a
  `mx.fast.metal_kernel` with `ensure_row_contiguous=True` copies every
  operand that is not row contiguous. That is what the gap was for the steel
  kernel. See OPTIMIZATIONS.md rows 32 and 34.

Compare `sum of stages` against `real per layer`. The two sums bracket it:
the `raw` sum is above the real layer and the `ms` sum is below it, because
`ms` subtracts the floor once for each of the 9 stages while the real model
pays it once for the whole forward. **The bracket now fails at shape 6:**
the `ms` sum 11.755 is ABOVE the real layer 11.546. The `stats` rows below
say why.

| shape | `ms` sum | real per layer | `raw` sum | when |
|---:|---:|---:|---:|---|
| 1 | 0.438 | 0.846 | 2.326 | 30 August 2026 |
| 6 | 11.755 | 11.546 | 12.880 | 31 August 2026 |
| 8 | 29.539 | 29.813 | 30.775 | 31 August 2026 |
| 13 | 8.574 | 10.381 | 10.917 | 30 August 2026 |

**Drop the two `stats` rows and the `raw` sum stops bracketing: it lands on
the real layer.** Row 47 removes those two passes, so the tool times work
that the model never runs. Measured 31 August 2026:

| shape | `raw` sum, all rows | `raw` sum, no `stats` rows | real per layer |
|---:|---:|---:|---:|
| 6 | 12.880 | 11.491 | 11.546 |
| 8 | 30.775 | 29.958 | 29.813 |

Both land within 0.5% of the real layer. That is the check that the
`stats` rows do not run, and it is a better bracket than the raw sum.

The bracket is tight at a large shape (4% wide at shape 8) and useless at a
small one (shape 1 spans 5x). When the real layer falls outside the bracket,
the stage model is wrong: at shape 6 it now does, because the two `stats`
rows time work that row 47 removed.

**The gap is not fusion.** `mx.compile` does not fuse the elementwise stages
of this block. Measured at the shape 6 chunk (B1024 S128 D128), median of 30
repeats:

| chain | eager | `mx.compile` | ratio |
|---|---:|---:|---:|
| `addmm` then `gelu` | 2.4414 ms | 2.4414 ms | 1.00x |
| add then LayerNorm | 2.8309 ms | 2.8347 ms | 1.00x |
| add, LayerNorm, add | 4.2828 ms | 4.3166 ms | 0.99x |

The byte count confirms it. One activation is 64 MiB. An unfused add plus
LayerNorm moves 320 MiB, which is 2.71 ms at the 124 GB/s that this size
reaches. The measurement is 2.83 ms. A fused pair would move 192 MiB, which
is 1.62 ms. The chain therefore runs both passes, and each pass runs at the
bandwidth roof.

So an elementwise stage cannot get faster from a better kernel. It can only
get faster from a kernel that moves fewer bytes.

### 4. The GPU gap report — find out if the GPU waits

```
./profiling/tools/gpu_timeline.sh --case 6 --iterations 5
.venv/bin/python3 profiling/tools/gpu_timeline.py report \
    profiling/traces/gpu_timeline.trace
```

Tool 2 needs Xcode. This one does not. It records the same `Metal System
Trace` and reads it with `xctrace export`, so it prints in the terminal.

It puts one `mlx-forward` signpost around each forward pass, clips the GPU
encoder intervals of the process to that window, and splits the idle time
three ways:

| Column | What it is |
|---|---|
| `head` | the CPU work before the first kernel. The `_to_mlx` input copy |
| `inner` | the gaps between two kernels. The only part a kernel change can win |
| `tail` | the wait after the last kernel |

Use it before you build anything that claims to remove a launch cost or a
stall. It gives the size of the prize first. Row 52 used it to rule out the
persistent kernel: the inner idle at shape 6 is 1.05%, 10 chunk boundaries
hold all of it, and the 90 launch gaps together hold 0.06%.

Read `head` from a late window. The first windows pay first touch, and the
head decayed from 141.965 ms to 15.9 ms over five passes in one run.

### 5. Your own labels — find the slow layer

```python
import sys; sys.path.insert(0, "profiling")
import signposts

with signposts.interval("attention"):
    ...
```

The label shows in the timeline. Put a label on each layer.
Then you can see which layer costs the most time.

## The optimization loop

Do these steps in this order:

1. Run `trace.sh` and `summarize.py`. Find the slow model.
2. Put signposts in that model. Run again. Find the slow layer.
3. Run `gpu_timeline.sh`. Find out if the GPU waits, and where.
4. Capture a `.gputrace` file. Open it in Xcode. Find the slow kernel.
5. Read the shader profiler. Decide if the kernel is memory-bound or math-bound.
6. Change the code.
7. Do step 1 again. Make sure the time decreased.

## A small shape needs an A/B, not two sweeps

A sweep compares two runs that are minutes or hours apart, so its ratio holds
the machine as well as the code. Below about 2 ms a shape moves further with
the machine than with any optimization.

Measured: row 47 read **0.922x** at shape 12 from two sweeps, and
**1.008x to 1.013x** from an interleaved A/B of the same change. Shape 12's
own MPS control moved 0.955x on that sweep pair, and MPS runs none of the
code under test.

    .venv/bin/python3 profiling/probes/plan_ab.py --cases 12

`plan_ab.py` builds the same model twice in one process, sets
`plan_override` on each, and alternates the order each round. Use it, or read
the MPS control and say the reading is inconclusive. Do not record a small
shape's two-sweep ratio as a result.

## Always synchronize before you measure

The MPS backend runs the GPU work in the background. If you do not
synchronize, you measure the queue time. You do not measure the real time.

This error is in `benchmark_once` in `torch_transformer_benchmark.py`. The
function synchronizes for CUDA but not for MPS. The measured results were:

| Measurement | Time |
|---|---|
| Baseline, queue time only | 2.84 ms |
| Baseline, after `torch.mps.synchronize()` | 19.54 ms |
| Optimized (MLX) | 14.83 ms |

The error makes the baseline look 7 times faster than it is. The MLX path
is approximately 1.3 times faster than the MPS baseline. The numbers in
`backend_comparison.json` agree: 20.08 ms for MPS and 15.30 ms for MLX.

The scripts in this directory synchronize for you. Your own timing code must
also synchronize.

## What the profiler cannot measure

The profiler cannot give you GPU cache misses. It cannot give you GPU branch
mispredictions. The M3 Pro GPU makes only one counter public: `GPUTimestamp`.

Use the occupancy data and the shader profiler instead. These are the correct
measurements for a GPU.

CPU cache misses are available. They come from a different template:

```
TEMPLATE="CPU Counters" ./profiling/tools/trace.sh
```

Use this template only to study the Python overhead and the numpy overhead.
The mathematics runs on the GPU.

## Control the disk space

Each `.gputrace` file uses approximately 575 MB for 3 iterations. Keep the
`--iterations` value small. Delete the old traces. Git ignores the `traces/`
directory, so these files stay out of a commit.

# Language

Write all output in ASD-STE100 Simplified Technical English. This applies to
your replies, to the reference files, and to the comments in the code.

1. Use short sentences. Put one idea in each sentence.
2. Use the active voice. Name the agent of each action.
3. Use one word for one meaning. Do not use a synonym for variety.
4. Use a simple tense: present, past or future. Do not use a perfect tense.
5. Do not put more than three nouns together.
6. Start an instruction with the verb.
7. Do not use a metaphor, a joke or a filler phrase.

A technical name keeps its usual form. `mx.fast.scaled_dot_product_attention`,
`head_dim` and `KernelPlan` are names, not noun clusters.

# Scope of optimization

Optimize `UserOptimizedTransformer` only.

`BaselineTransformer` is the benchmark. It is the reference for accuracy and
the reference for speed. Do not change it, do not port it, and do not move it
to a different framework or device. A change to the baseline makes every
number in `OPTIMIZATIONS.md` meaningless.

This also applies to the harness in `torch_transformer_benchmark.py`:
`compare_outputs`, `benchmark_models` and the CLI defaults stay as they are.
Do not raise `--atol` or `--rtol` to make a test pass.

Read-only copies of the baseline on other devices are permitted for
measurement, but only in a separate file (`test_backends.py` does this for
MPS). The class in `torch_transformer_benchmark.py` does not change.

# References

Read the relevant file before you start work. Each one holds measured facts
that are costly to derive again.

| File | Read it before you |
|---|---|
| [references/README.md](references/README.md) | add a reference, or look for one |
| [references/test-shapes.md](references/test-shapes.md) | reason about a test shape, or pick a shape to optimize |
| [references/machine.md](references/machine.md) | trust a number, or write a timing loop |
| [references/mlx-tensorops.md](references/mlx-tensorops.md) | choose an MLX kernel, or add a `KernelPlan` branch |
| [OPTIMIZATIONS.md](OPTIMIZATIONS.md) | try an optimization, in case it already failed |
| [profiling/WORKFLOW.md](profiling/WORKFLOW.md) | look for the next optimization |

Keep them current. When you measure something that one of these files states,
and your result disagrees, correct the file in the same change.

# The source of truth table

`OPTIMIZATIONS.md` starts with a **source of truth** table. It is the only
place that states the status of an optimization. Read it before you try an
optimization, and before you describe the model to anybody.

One row for each optimization, with these columns:

| Column | Meaning |
|---|---|
| **#** | The row number. The detail section uses the same number. |
| **Optimization** | What it does, in one line. |
| **Status** | `KEPT`, `REVERTED`, `RULED OUT` or `OPEN`. |
| **Lives in** | The function and line that holds it, or `—` if it is not in the code. |
| **Applies where** | The shapes or the conditions that use it. `—` if none. |
| **Measured effect** | The number that decided the status. |

Status:

- **KEPT** — it is in `UserOptimizedTransformer` now.
- **REVERTED** — it was measured, and it lost. Do not try it again.
- **RULED OUT** — it is out of scope, or it cannot pass.
- **OPEN** — it is not tried yet.

Rules:

1. Update the table in the same change that adds, keeps or reverts an
   optimization. A change that does not touch the table is not complete.
2. Give every row a detail section below, with the measurement.
3. The table wins. If a section disagrees with the table, correct the
   table, then correct the section.
4. Do not delete a row. A `REVERTED` row stops repeated work.
5. Check the "Lives in" line numbers when you move code.

# Rules for a measurement

1. Look for a running measurement before you start one. Two runs share one
   GPU, and each one makes the other reading false. Run this command first:

       ps -Ao pid,etime,%cpu,command | grep "[.]venv/bin/python3" | grep -v shell-snapshots

   If the command prints a process, wait for that process to end. Do not
   start a second run beside it, and do not trust a number that you took
   beside it. Measured: shape 12 gave 2.591 ms and 5.021 ms in two runs of
   one script, while a `scoreboard.py` sweep held the GPU at 228% CPU.
2. Run with `.venv/bin/python3`. Never use the system interpreter.
3. Call `mx.synchronize()` after `mx.eval()`. `mx.eval()` alone times the
   CPU graph build, not the GPU.
4. Use the default `--repeats`. A short run gives a false median: `--repeats
   3` gave 7.198x where 100 repeats gave 4.590x on the same build.
5. Give the command that produced a number, next to the number.
6. Record a failure as well as a success. A recorded failure stops repeated
   work.
7. **Always run `scoreboard.py` with `--cpu-cache`.** The flag is off by
   default in the script, so pass it every time:

       .venv/bin/python3 scoreboard.py --cpu-cache --label "what changed"

   The CPU baseline takes most of the sweep time, and `BaselineTransformer`
   never changes, so measuring it again on every sweep wastes minutes.

   Know what this costs. A cached reading comes from an earlier sweep, under
   a different machine load and at a different chip temperature. So the
   **speedup** column mixes two sweeps and is approximate. The **MLX ms**
   column does not: it is measured fresh every sweep, and it is the column
   that decides whether an optimization won. Compare MLX ms against MLX ms.

   Run without the flag only when the user asks for a clean CPU reference.

## The CPU cache

`BaselineTransformer` never changes, so its time only moves with the machine.
`--cpu-cache` stores one entry for each shape in `profiling/cpu_cache.json`.
The file is machine-local, and git ignores it.

- Each entry counts its uses. It serves five sweeps. The sixth sweep measures
  the CPU again, writes the new reading, and sets the count back to zero.
- Each entry holds its own count. One shape does not expire another shape.
- The key holds the shape config, the dtype, the seed, the padding ratio, the
  warmup count, the repeat count, the round count and the torch version. A
  change to any one of these makes a new entry.
- Every output marks a cached reading: the console line, `scoreboard.json`,
  `history.jsonl`, and a `†` in `references/scoreboard.md`.
- Run `scoreboard.py --clear-cpu-cache` to delete the file.

# What a test run must report

Every benchmark run reports the same four things, for every shape it runs.
Do not drop a column, and do not report only the best shape.

| Column | Meaning |
|---|---|
| **CPU** | torch baseline on the CPU. The accuracy reference and the speed reference. |
| **MPS** | torch baseline on the GPU, through Metal. Shows how much of the gain is the device alone. |
| **MLX** | `UserOptimizedTransformer`. Shows how much of the gain is the kernels. |
| **MFU (MLX)** | Model FLOPs Utilization of the MLX path. |

Report speedup against the **CPU** column, because it is the reference the
harness uses. Also give the MPS column, because the CPU baseline moves with
machine load and thermal state, and MPS does not.

MFU is the graded score, and it is reported for **MLX only**:

    MFU = model_flops / seconds / peak_flops_per_second

The denominator is **4.946 TFLOP/s**, and every term in it is now checked
on this machine: `14 cores x 128 ALUs x 2 x 1.380 GHz`. The core count comes
from `system_profiler`, the 1380 MHz top state from the GPU DVFS table
`voltage-states9` in the pmgr device tree, and the ALU count is bounded below
at 105 by the measured matmul rate. An earlier version used 1.398 GHz, which
was asserted from memory and is wrong by 1.3%.

**Read every MFU against 82%, not against 100%.** The GPU does not hold its
top DVFS state. A saturating FMA loop sustains 3.92 TFLOP/s
(`profiling/alu_peak.py`) and a plain matmul reaches 4.06 TFLOP/s
(`flops.py --peak`), and both imply a clock near 1.1 GHz. No kernel can close
that gap, so 4.06 TFLOP/s is the ceiling a kernel competes against. Give the
speedup and the achieved TFLOP/s first: they are stopwatch readings.

`flops.py` holds the FLOP model. It counts matmuls only, and it counts the
full S x S attention, not the causal triangle. Do not change it without a
note in the file.

Write the result of a full sweep to `profiling/scoreboard.json` and to the
table in `references/scoreboard.md`.

# Keep every reading

`profiling/scoreboard.json` holds the newest sweep only, and every run
overwrites it. `profiling/history.jsonl` never loses a reading: one line for
each sweep, with the commit that produced it.

    .venv/bin/python3 scoreboard.py --label "what changed"   # appends
    .venv/bin/python3 scoreboard.py --show-history           # the trend

Always pass `--label`. A reading with no label is hard to place later. A `*`
after the commit means the working tree was dirty, so that reading cannot be
reproduced exactly.

# Where to put effort

Use the **FLOP share** column of `references/scoreboard.md`. It is the weight
a shape carries in a FLOP-weighted score, and it comes from the shape alone,
so it is exact.

Shape 6 is 66.5% of the total FLOPs and shape 8 is 21.3%. Those two decide a
FLOP-weighted score. The other eleven together are 12%. Do not optimize a
shape that carries little weight, whatever its rate looks like.

An earlier version of this file described a measured "ceiling" for each shape
and a priority ranking built on it. **That was removed.** It stacked three
unvalidated constants, and its first version produced a ceiling below the
measured runtime. Do not reintroduce it without validating each constant.

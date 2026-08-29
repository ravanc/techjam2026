Use ASD-STE100 in your output.

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

# Rules for a measurement

1. Run with `.venv/bin/python3`. Never use the system interpreter.
2. Call `mx.synchronize()` after `mx.eval()`. `mx.eval()` alone times the
   CPU graph build, not the GPU.
3. Use the default `--repeats`. A short run gives a false median: `--repeats
   3` gave 7.198x where 100 repeats gave 4.590x on the same build.
4. Give the command that produced a number, next to the number.
5. Record a failure as well as a success. A recorded failure stops repeated
   work.

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

**The MFU numbers are PROVISIONAL. Do not quote one without the
disclaimer.** The numerator and the time are objective. The denominator is
not: 5.01 TFLOP/s came from `14 cores x 128 ALUs x 2 x 1.398 GHz`, and the
clock and the ALU count were asserted from memory, never checked against a
source. Every MFU scales linearly with it.

Fix `PROVISIONAL_PEAK_TFLOPS` in `flops.py` before trusting any MFU. Until
then, lead with the speedup and the achieved TFLOP/s, which are stopwatch
readings, and treat MFU as a rough guide only.

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

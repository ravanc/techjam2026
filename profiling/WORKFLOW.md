# How to use the profiler for optimization

This guide tells you how to find slow code and how to make it faster.
For the setup steps and the platform problems, read `README.md`.

## The four tools

### 1. The timeline — find the slow model

```
./profiling/trace.sh
.venv/bin/python3 profiling/summarize.py profiling/traces/benchmark.trace
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
.venv/bin/python3 profiling/profile_benchmark.py --mode gputrace --iterations 3
open profiling/traces/optimized_mlx.gputrace
```

Xcode shows the time for each GPU kernel. The shader profiler shows the cost
of each instruction. It tells you if a kernel waits for memory or for math.

### 3. The stage roofline — find the slow stage, and its limit

```
.venv/bin/python3 profiling/stage_roofline.py --shapes 1,6,8,13
```

It splits one transformer block into eleven stages, times each one alone,
and names what limits it: the arithmetic units, the memory system, or the
kernel launch. For each stage it gives the FLOPs, the compulsory DRAM
traffic, the arithmetic intensity, and the two achieved rates against the
measured roofs (4.06 TFLOP/s and 128 GB/s, ridge 31.7 FLOP/byte).

It subtracts the `mx.eval` + `mx.synchronize` round trip, which is 0.13 to
0.17 ms and dominates every small shape. A stage that does not clear the
floor prints `at floor` and is marked LAUNCH.

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

Compare `sum of stages` against `real per layer`. The sum is the larger
number, because `mx.compile` fuses the elementwise stages and the GPU
overlaps the layers. When the two are far apart, the stage model is wrong.

### 4. Your own labels — find the slow layer

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
3. Capture a `.gputrace` file. Open it in Xcode. Find the slow kernel.
4. Read the shader profiler. Decide if the kernel is memory-bound or math-bound.
5. Change the code.
6. Do step 1 again. Make sure the time decreased.

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
TEMPLATE="CPU Counters" ./profiling/trace.sh
```

Use this template only to study the Python overhead and the numpy overhead.
The mathematics runs on the GPU.

## Control the disk space

Each `.gputrace` file uses approximately 575 MB for 3 iterations. Keep the
`--iterations` value small. Delete the old traces. Git ignores the `traces/`
directory, so these files stay out of a commit.

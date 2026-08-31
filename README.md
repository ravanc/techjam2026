# A shape-aware GPU transformer kernel for Apple silicon

**TechJam 2026 — Problem 3: Implement a GPU Kernel for a Transformer Layer**
Team: PB & Techjam

## Project overview

The benchmark gives a reference transformer forward pass in PyTorch and asks
for a faster implementation that returns the same answer. The reference is
`BaselineTransformer`. It never changes. Only `UserOptimizedTransformer`
changes.

Fourteen input shapes are given in advance. They span seven orders of
magnitude of work, from 0.13 GFLOP to 2.7 PFLOP. **No single kernel is correct
across that range**, so the deliverable is not one kernel. It is a dispatcher
that picks a kernel path from the shape.

`plan_kernels()` reads the shape once per model and returns a `KernelPlan`:
which attention kernel, which normalization kernel, which GEMM epilogues,
whether to chunk the batch, and which tile to use. Every threshold in it comes
from a measurement on the target machine, not from a rule of thumb.

The implementation runs in MLX on the Metal GPU, behind the torch interface
the harness expects. Two kernels are custom Metal, built by compiling Apple's
own kernel templates at parameters Apple does not ship.

**Result: a median of 11.6x against the torch CPU reference across the 13
runnable shapes, from 2.4x to 49.3x, with every shape passing the accuracy
test.** The optimization chain took the suite total from 1298.3 ms to
628.8 ms, which is **2.065x from the kernel work alone**.

### The three ideas the project rests on

1. **Vendor kernels are templates. Instantiate the one you need.**
   `mx.fast.scaled_dot_product_attention` reaches its fused flash kernel for
   `head_dim` in {64, 72, 80, 96, 128} and silently falls back for every other
   width, materializing a `B x H x S x S` score matrix. Most of the appendix
   falls off that cliff. MLX ships the Metal source of that kernel as a C++
   template whose width parameter is only constrained by `BD % 8 == 0`. We
   read the headers off disk and compile the same kernel at head widths 8 and
   32. No new arithmetic is written. The same argument applies to CUTLASS on
   NVIDIA hardware.
2. **On a machine at its roofline ridge, the byte count is the runtime.**
   The ridge of the target machine is about 32 FLOP/byte, and a square float32
   projection at `d_model = 128` has an arithmetic intensity of exactly
   `d/4 = 32`. Twelve of the fourteen shapes sit on that knee. So almost every
   optimization we kept removes bytes, not arithmetic.
3. **An affine operation distributes through a matmul.** A LayerNorm is affine
   in the row, so instead of fusing it into the next kernel we fold it into
   that kernel's weights at build time and pass two floats per row. The full
   normalization pass disappears. Three rows of the log come from this one
   observation.

## Setup and installation

Requirements: an Apple silicon Mac (M1 or later), macOS with the Metal
toolchain, and Python 3.9 or later.

```bash
git clone https://github.com/ravanc/techjam2026.git
cd techjam2026
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

`requirements.txt` pins exact versions, because the benchmark compares
measured times and a different torch version gives a different CPU baseline:

```
mlx==0.32.2        numpy==2.5.2        torch==2.13.0
```

Confirm the install and read the machine you are on:

```bash
.venv/bin/python3 -c "import mlx.core as mx; print(mx.device_info())"
.venv/bin/python3 flops.py --peak
```

There is no CUDA path. MLX ships its GPU backend as `mlx-metal`, an arm64
macOS wheel, so this runs on Apple silicon only. See *Limitations*.

## Steps to reproduce our results

Run these in order. Close other applications first: a browser or an editor
holding the GPU changes the numbers (see *Limitations*).

```bash
# 1. correctness, on the harness's own default configuration
.venv/bin/python3 torch_transformer_benchmark.py

# 2. correctness on padded and ragged batches, which the sweep never runs
.venv/bin/python3 test_padding.py

# 3. the graded sweep over all 13 runnable Appendix 3.7 shapes
.venv/bin/python3 scoreboard.py --label "reproduction"

# 4. where the time goes, one shape at a time
.venv/bin/python3 profiling/stage_roofline.py --shapes 6,8,13
```

Step 3 takes about 10 minutes on an idle machine. It writes
`profiling/scoreboard.json`, appends one line to `profiling/history.jsonl`,
and rewrites `references/scoreboard.md`.

Pass `--cpu-cache` to reuse the CPU baseline from an earlier sweep and save
several minutes. Do not use it for a reported number: a cached reading comes
from a different sweep at a different chip temperature, so the speedup column
then mixes two runs. Every cached reading is marked with a dagger.

### Results

All 13 runnable shapes, float32, `atol=0.002`, `rtol=0.02`. `CPU` is the torch
reference and the accuracy reference. `MPS` is the same reference on the GPU
through PyTorch's Metal backend, which shows how much of the gain is the
device alone. `MLX` is our implementation.

| # | Shape (B, D, H, S) | Max abs error | Accuracy | CPU (ms) | MPS (ms) | MLX (ms) | MLX vs CPU | MLX vs MPS |
|---:|---|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 64, 128, 4, 128 | 1.91e-06 | PASS | 45.98 | 17.44 | 3.27 | **14.08x** | 5.34x |
| 2 | 1, 128, 4, 128 | 1.31e-06 | PASS | 1.45 | 1.53 | 0.61 | **2.37x** | 2.50x |
| 3 | 4, 128, 4, 128 | 1.43e-06 | PASS | 4.08 | 2.02 | 0.67 | **6.07x** | 3.00x |
| 4 | 16, 128, 4, 128 | 1.43e-06 | PASS | 13.94 | 4.70 | 1.20 | **11.59x** | 3.91x |
| 5 | 128, 128, 4, 128 | 2.38e-06 | PASS | 102.28 | 33.71 | 5.84 | **17.51x** | 5.77x |
| 6 | 10000, 128, 4, 128 | 2.38e-06 | PASS | 15419.72 | 2771.85 | 436.81 | **35.30x** | 6.35x |
| 7 | 64, 32, 4, 128 | 1.67e-06 | PASS | 23.69 | 11.52 | 1.00 | **23.62x** | 11.49x |
| 8 | 64, 1024, 4, 128 | 3.58e-06 | PASS | 461.15 | 167.57 | 115.53 | **3.99x** | 1.45x |
| 9 | 64, 128, 1, 128 | 1.67e-06 | PASS | 26.39 | 8.10 | 3.38 | **7.81x** | 2.40x |
| 10 | 64, 128, 2, 128 | 1.91e-06 | PASS | 37.33 | 13.10 | 3.28 | **11.40x** | 4.00x |
| 11 | 64, 128, 16, 128 | 1.67e-06 | PASS | 121.85 | 42.51 | 3.25 | **37.48x** | 13.08x |
| 12 | 64, 128, 4, 32 | 1.55e-06 | PASS | 7.53 | 3.47 | 1.16 | **6.52x** | 3.00x |
| 13 | 64, 128, 4, 1024 | 1.91e-06 | PASS | 1845.88 | 577.36 | 37.46 | **49.28x** | 15.41x |

| Metric | Value |
|---|---|
| Median speedup against the CPU reference | **11.59x** |
| Range | 2.37x to 49.28x |
| Accuracy | **13 of 13 PASS**, zero failed elements, worst absolute error 3.58e-06 against a tolerance of 0.002 |
| Suite total, MLX | **613.5 ms** |

Shape 14 (B=32, D=1024, H=16, S=100000, L=2) is disabled. Its input alone is
12.2 GiB and the reference materializes an 18.6 TiB score matrix, so **the
reference cannot run it on any machine that exists**. See *Limitations*.

**Read the MPS column, not only the CPU column.** The CPU reference moves with
machine load and chip temperature; it drifted 45.9% between two sweeps on
unchanged code. MPS is stable to a few percent, so it is the better control
when comparing two builds.

### The optimization chain

The seven changes that carry the result. Each row is a measured pair of suite
totals. The full log is [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

| Change | Before (ms) | After (ms) | Ratio |
|---|---:|---:|---:|
| Single-pass row moments | 1298.3 | 1077.6 | 1.205x |
| Strided operands, no copy | 1077.6 | 869.7 | 1.239x |
| Deferred bias, residual to C | 869.7 | 768.6 | 1.132x |
| Nonlinearity in the epilogue | 768.6 | 722.3 | 1.064x |
| Normalize folded into weights | 722.3 | 681.4 | 1.060x |
| Row moments in the epilogue | 686.6 | 638.7 | 1.075x |
| Final normalize in the epilogue | 635.2 | 628.8 | 1.019x |
| **Whole chain** | **1298.3** | **628.8** | **2.065x** |

## The AI-assisted workflow

The problem statement asks what AI tools were used. The tools matter less than
the protocol that made their output trustworthy.

**Tools.** Claude Code (Anthropic) as the agent, driven from a repository
whose rules live in `CLAUDE.md`; Xcode Instruments and Metal GPU capture for
kernel-level profiling; custom profilers under `profiling/`.

**The loop.** Ideate, screen, build, profile, then keep or revert. It is
written down in [agent_loop.md](agent_loop.md). Two layers, in order: first
find work to *delete* at the model level, then tune the kernel that produced.
Tuning first is a trap, and we measured it — retuning the attention block
shape over five cases found nothing above 1.002x.

**What made it work.** Three rules, all learned by getting them wrong first.

1. **A source of truth table.** `OPTIMIZATIONS.md` holds one row per
   optimization with its status and the measurement that decided it. A
   `REVERTED` row is never deleted, because a recorded failure stops repeated
   work. 51 rows: 26 kept, 13 reverted, 6 ruled out, 6 open.
2. **A stable control on every claim.** The CPU reference moves 76% with
   machine load. The MPS column runs unchanged code, so if it moves more than
   about 1%, the machine moved and the reading cannot score a change. Several
   rows were rejected on this basis alone.
3. **The agent does not score its own work.** Every claimed win passes four
   gates: accuracy on 13 shapes, correctness on 18 padded cases, an
   improvement in the MLX column, and a control that did not move.

**Reverted work is reported, not hidden.** The clearest example: we pipelined
the CPU-to-GPU batch loop so the copy would hide behind compute. It measured
0.974x. Apple silicon has unified memory, so the CPU copy and the GPU kernels
contend for one memory controller and there is no separate bus to overlap
with. That is a correct CUDA instinct that does not survive the platform, and
it cost a day. It is row 48.

## Limitations, and what we would improve with more time

**Apple silicon only.** MLX has no CUDA backend, and the two custom kernels
inline Apple's Metal source. The *method* ports — shape-aware dispatch,
roofline attribution, template instantiation, the measurement protocol — but
the kernels do not. We would port the dispatcher to CUTLASS on an NVIDIA part
to show that the argument, not the code, is the contribution.

**Float32 only.** The harness fixes `atol=0.002` and `rtol=0.02`. No
half-precision implementation passes: float16 fails 49 of 1.5M elements and
bfloat16 fails 172,424. PyTorch's own
`F.scaled_dot_product_attention` fails the same test, so this is arithmetic,
not a framework problem. Quantization fails for the same reason and worse.

**Thresholds are tuned to one machine.** Chunk size, row-count gates, block
sizes and tile choices were all measured on one laptop. We would write a
calibration script that re-derives the four roofs — matmul peak, streaming
bandwidth, launch floor, and the normalization crossover — on any machine and
writes a machine profile that `plan_kernels()` reads. Dispatch would then key
on hardware as well as shape.

**Measurement noise on a shared desktop.** Without a dedicated node, browsers
and editors take variable CPU and GPU bandwidth between runs. We saw a single
contended sweep report a control column 12x slower than the same sweep on an
idle machine. Our defence is the MPS control and repeated sweeps; a proper fix
is a locked machine.

**Shape 14 is unrun.** It is the one appendix shape with no result. The
reference cannot produce an answer for it at any size, so there is nothing to
compare against, and one forward pass is roughly 8 minutes, which does not fit
the harness's 320-pass timing protocol. We have designed the path — stream the
batch conversion so peak GPU memory stays flat, and validate attention at the
true sequence length against a float64 row-wise reference, since a single
output row costs `O(S*d)` rather than `O(S^2)` — but it is not yet in the
model.

**Agent behaviour needed steering.** Agents held to self-imposed guardrails
too strictly. Asked to try Apple's flash attention implementation, the agent
repeatedly diverted to other changes. It also defaults to industrial-GPU
assumptions and does not take the initiative to research platform specifics
such as unified memory. The mitigation we would apply next time is to brief
the agent with the platform research up front, rather than expect it to find
it.

**Future work: autonomous kernel optimization.** Recent work — Baseten's
*Agentic Kernels in Production* and *KernelArc* (arXiv 2608.17071) — runs
strategy-specialized agents with shared memory and a deterministic guard, with
no human in the loop. Our source-of-truth table is already that shared memory,
and our four gates are already that guard, but a human runs them. The
remaining step is to make the guard deterministic code, so an agent proposes
an edit and never decides whether its own edit is an improvement.

## Team member contributions

<!-- TODO: fill in before submission. One line per member. -->

| Member | Contribution |
|---|---|
| | |

## Repository layout

| File | Role |
|---|---|
| `torch_transformer_benchmark.py` | The reference model, our model, and the harness |
| `steel_attention.py` | Apple's flash attention kernel, compiled at a head width Apple does not ship |
| `steel_gemm.py` | Apple's GEMM, with epilogues for the activation, the normalization and the row statistics |
| `fast_layernorm.py` | A single-pass normalization kernel for a narrow row |
| `scoreboard.py` | The graded sweep over all 13 shapes |
| `appendix_cases.py` | The 14 Appendix 3.7 shapes as code |
| `test_padding.py` | Padded and ragged batches, including an empty sample |
| `test_backends.py` | CPU, MPS and MLX side by side |
| `flops.py` | The FLOP model and the measured matmul rates |
| `OPTIMIZATIONS.md` | Every optimization tried, kept or reverted, with its measurement |
| `agent_loop.md` | The AI-assisted method and its run log |
| `references/` | Measured facts: the machine, the shapes, the MLX kernels |
| `profiling/` | Roofline, kernel trace and probe tooling |

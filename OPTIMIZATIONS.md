# Optimization log

A record of every optimization tried on `torch_transformer_benchmark.py`.
Add a new row for each attempt. Keep the failures. They stop repeated work.

## Source of truth

**This table is the only place that states the status of an optimization.**
Read it first. Update it in the same change that adds, keeps or reverts an
optimization. The section below each row holds the measurement that decided
the status. If a section and this table disagree, this table is wrong: fix
it.

Status:

- **KEPT** — it is in `UserOptimizedTransformer` now.
- **REVERTED** — it was measured, and it lost. Do not try it again.
- **RULED OUT** — it is out of scope, or it cannot pass.
- **OPEN** — it is not tried yet.

| # | Optimization | Status | Lives in | Applies where | Measured effect |
|---:|---|---|---|---|---|
| 1 | MLX behind the torch interface | **KEPT** | `_mlx_transformer()` L512, `UserOptimizedTransformer` L620 | every shape | 4.4x to 7.4x against torch CPU. float32 PASS |
| 2 | `mx.compile` on the forward pass | **KEPT** | `make_call()` L745 | every shape | 14.203 ms to 13.401 ms (6%) |
| 3 | float32 LayerNorm accumulation | **KEPT** | `norm()` L543, weight build L687 | every shape | no measured gain. Correct policy. Free |
| 4 | Explicit float32 softmax, not fused | **REVERTED** | — | — | float16 49 to 50 failures. No gain. More code |
| 5 | `torch.compile` over the MLX class | **RULED OUT** | — | — | crash. Dynamo cannot trace an MLX object |
| 6 | Mask form selected by the input | **KEPT** | `_mlx_transformer()` L557 | seq_len >= 512, no padding | 3.9% at seq 512. 1.7% at seq 2048. none at seq 128 |
| 7 | Shape-aware kernel plan (`KernelPlan`) | **KEPT** | `plan_kernels()` L343 | every shape. It selects 8, 9 and 10 | 1.57x at shape 13. 1.31x at shape 6 |
| 8 | Blocked causal attention | **KEPT** | `_attention()` L459 | causal, effective `head_dim < 64`, and not on the steel path. **No appendix shape selects it now** | 1.63x at seq 1024, bit exact, when it was the best option. It works around the FALLBACK kernel. Row 25 removes the fallback instead, so every shape that used to block now takes the steel kernel. The code stays for a shape row 25 cannot take |
| 9 | Fused `[D, 3D]` QKV matmul | **KEPT** | weight build L738, use L580 | every shape. Always on | 1.99x at batch 1. 1.00x at d_model 1024 |
| 10 | Batch chunking, full depth for each chunk | **KEPT** | `forward()` L778 | one activation over 64 MiB. Shape 6 only | peak 9.16 GiB to 2.68 GiB, and 1.05x faster |
| 11 | Pad `head_dim` from 8 up to 32 | **REVERTED** | — | — | 0.91x at shape 7. Blocking beats it. Wrong target: 32 is still the fallback kernel. See row 17 |
| 12 | Full port of the file to MLX | **RULED OUT** | — | — | out of scope. The baseline does not change |
| 13 | float16 and bfloat16 accuracy | **RULED OUT** | — | — | no implementation can pass. torch `F.sdpa` fails too |
| 14 | Hand-written Metal kernel for `head_dim = 8` | **KEPT**, as row 25 | `steel_attention.py` | shapes 7 and 11 | done without writing new arithmetic: row 25 compiles Apple's own kernel at `head_dim = 8`. Shape 11 **2.19x**, shape 7 1.48x |
| 17 | Pad `head_dim` to 64 to reach the fused SDPA kernel | **KEPT** | `plan_kernels()` L433 | **nothing now.** Row 25 wins wherever the pad applied, and `plan_kernels()` prefers it | 1.646x at shape 13 when it was the best option. Row 25 gives shape 13 **2.14x over the padded path** at the attention step, with no widened projection. The pad stays in the code for a shape the steel kernel cannot take |
| 18 | MLX dispatches SDPA to two different kernels | **KEPT** (a fact, not a change) | `SDPA_FUSED_MIN_HEAD_DIM` L320 | `head_dim` in {64, 72, 80, 96, 128} gets the fused kernel. Everything else materializes `B x H x S x S` | peak memory at B8 H8 S1024: 16 MiB fused against 264 MiB fallback. **The set is NOT the range 64..128.** See row 20 |
| 19 | Sweep `seq_len` to place the `S >= 4*D` threshold | **OPEN** | — | `head_dim` 32..48 | not measured at the model level. Row 20 measures the crossover at the attention kernel alone |
| 20 | The fused SDPA set is discrete, not a range | **KEPT** (a fact, not a change) | `profiling/sdpa_dispatch.py` | `head_dim` 1..288, every mask kind, every dtype, `S` 512 and 1024, `B*H` 1 to 256 | the set is {64, 72, 80, 96, 128}. 65, 100 and 127 fall back. Always pad to the SMALLEST member at or above `head_dim`: a larger target lost all 44 rows |
| 21 | MLX never calls its own `bd192` and `bd256` kernels | **OPEN** (an MLX gap) | — | shape 8, `head_dim=256`, 21.3% of the FLOP-weighted score | the metallib holds `steel_attention_*_bd192_*` and `*_bd256_*` for all 3 dtypes. All 32 dispatch combinations tried take the fallback. No Python workaround: `head_dim` cannot pad down and a head cannot split. Recheck after an MLX upgrade |
| 22 | Block a causal wide head that misses the fused set | **OPEN** | — | `head_dim > 128` and long `S`. No appendix shape reaches it | `head_dim=256`, B64 H4: S=128 full wins, S=512 blk128 gives 1.32x, S=1024 blk128 gives **1.55x**. `plan_kernels()` refuses to block because it tests `effective_head_dim < 64` |
| 23 | Return the output as a view of MLX memory, not a copy | **KEPT** | `_to_torch()` L188 | every shape. float32 and float16 alias. bfloat16 and a device change still copy | **71.6 ms of 1590.2 ms at shape 6 (1.047x)**. 1.5 ms of 136 ms at shape 8. Bit exact by `torch.equal` |
| 15 | Fused FFN | **OPEN** | — | every shape | not measured. Check `mx.compile` first |
| 16 | Quantization by `mx.quantize` | **OPEN** | — | every shape | not measured. It will fail the accuracy test |
| 25 | Hoist MLX's `steel_attention` and compile it at an unshipped `head_dim` | **KEPT** | `steel_attention.py`, routed at `_attention()` L479, gated at `plan_kernels()` L419 | causal, no padded batch, `head_dim % 8 == 0`, `head_dim` not already fused, threadgroup fits. Shapes 1-7, 11, 12, 13 | **1.308x FLOP-weighted** (MLX 1218.8 ms to 932.2 ms). Shape 6 1.32x, shape 13 1.47x, shape 11 2.19x. MLX against MPS 1.60x to 2.12x FLOP-weighted. All accuracy PASS, max_abs 1.31e-06 to 1.91e-06 |
| 26 | Reach the steel kernel at `head_dim = 256` for shape 8 | **OPEN** | — | shape 8, 21.3% of the FLOP weight | not measured. `bq32_bk32_bd256` needs 68.5 KiB of threadgroup memory against a 32 KiB limit, and `bk16` still needs 41 KiB. `bk8` would fit. Shape 8 is only 7.6% attention, so the ceiling is small |
| 27 | Gate the steel kernel on a string mask | **KEPT** (a bug fix) | `_attention()` L479 | every padded batch on a shape that selects the steel kernel | a padded causal batch went **FAIL** (822894/1048576 elements wrong) to **PASS** (`max_abs=3.04e-06`). No sweep saw the bug: `--padding-ratio` defaults to 0.0 everywhere. `test_padding.py` now covers it |
| 28 | Drop the token masks that cannot change the output | **KEPT** | `_mlx_transformer()` L570, L611, L617 | every shape. The unpadded graph now holds no mask operation | bit exact on 18 cases (`test_padding.py`). **1.048x FLOP-weighted** (MLX 841.6 ms to 803.2 ms). Shape 6 1.050x. MPS held at 1.010x and CPU at 1.005x on the same sweep, so the noise floor is about 1% |
| 29 | `mx.addmm` for every projection, in place of `h @ w + b` | **KEPT** | `_mlx_transformer()` L587-L616 | every shape. Always on | **1.096x FLOP-weighted** (MLX 803.2 ms to 732.6 ms). Shape 6 1.099x, shape 8 1.040x, shape 13 1.092x. Bit exact in float32 on 16 cases |
| 30 | Flatten the block to rank 2 before each projection | **REVERTED** | — | — | 0.992x to 1.000x against rank 3, on 4 projection sizes. MLX already collapses a rank 3 by rank 2 matmul into one GEMM |
| 31 | Single-pass LayerNorm kernel for a narrow row | **KEPT** | `fast_layernorm.py`, chosen at `_mlx_transformer()` L563, gated at `plan_kernels()` L465 | `d_model < 256`, float32. Shapes 1-7 and 9-13. Shape 8 keeps MLX | **1.205x FLOP-weighted** (MLX 1298.3 ms to 1077.6 ms over the 13 shapes). Shape 7 **3.41x**, shape 10 1.42x, shape 9 1.41x, shape 6 **1.23x**, shape 13 1.17x. Shape 8 1.00x, so the gate is correct. All 13 shapes PASS, `max_abs` 9.5e-07 to 2.65e-06. 18/18 padding cases bit exact |
| 32 | Give the attention kernel contiguous q, k and v | **OPEN** | — | every shape on the steel path | not tried at the model level. An MLX transpose is a free strided view, so the head layout costs nothing as a stage and costs inside the attention kernel instead. Shape 6: SDPA is 5.54 ms on the strided view and **2.13 ms on contiguous copies, 2.60x**. A `mx.contiguous` first does not pay (3.29 ms of copy makes the total 5.42 ms, only 1.02x). The win needs the QKV projection to write the head layout directly, or a kernel that reads the stride well |
| 33 | Fold GELU into the FFN matmul epilogue | **OPEN** | — | every shape. `ffn_in` only | not tried. GELU runs as a separate kernel and costs a whole extra read plus write. At the shape 6 chunk (64 MiB activation): `mx.addmm` alone 1.417 ms, `addmm` then GELU 2.385 ms, so GELU adds **0.968 ms**. GELU alone is 1.166 ms at 115 GB/s, which is 90% of the copy roof, so the kernel itself is efficient. It is the extra pass that costs. `mx.compile` does NOT fuse it: 2.378 ms compiled against 2.385 ms plain. That is 4.5% of the shape 6 runtime. MLX exposes no matmul epilogue, so Python may not be able to reach it |

Line numbers are in `torch_transformer_benchmark.py`.

## Test conditions

| Item | Value |
|---|---|
| Machine | Apple M3 Pro |
| Python | 3.13 in `.venv` |
| torch | 2.13.0 (CPU device by default; MPS is available and measured below) |
| mlx | 0.32.2 (GPU device) |
| numpy | 2.5.2 |
| Model | batch 8, seq 128, d_model 512, heads 8, ffn 2048, layers 6 |
| Thresholds | `atol=0.002`, `rtol=0.02` |

Run the full benchmark. Do not trust a short run:

```
.venv/bin/python3 torch_transformer_benchmark.py
```

A run with `--repeats 3` gave 7.198x. The same build with the default 100
repeats gave 4.590x. Three samples give a false median.

## Current result

Latest run, after attempts 7 to 11:

```
float32 : PASS | max_abs=2.98e-06 | failed=0/2621440
baseline : median=59.3431 ms | 17256 token/s
optimized: median=13.3549 ms | 76676 token/s
speedup  : 4.444x
```

**Do not quote one speedup number.** The ratio is unstable, and the cause is
the baseline, not the optimized model:

| Model | Range seen across runs |
|---|---|
| optimized (MLX) | 13.4 ms to 15.3 ms — stable, about 12% spread |
| baseline (torch CPU) | 59.3 ms to 104.1 ms — 76% spread |
| speedup | 4.44x to 7.36x |

The torch CPU baseline moves with the machine load and the thermal state. The
MLX model does not. A speedup between **4.4x and 7.4x** is the honest claim.
Compare against the MPS baseline (below) for a number that does not move.

The run above has the **fastest optimized median recorded** (13.355 ms) and a
fast baseline (59.3 ms), so its ratio is at the low end. Read the optimized
column, not the ratio.

This shape gets `head_dim = 64`, so `plan_kernels()` chooses the unblocked
path. Attempts 7, 8, 10 and 11 do not apply here. They apply to the appendix
shapes. See attempt 7.

## Attempts

Rows 1 to 11 of the source of truth table. Each section holds the
measurement that decided the status. Rows 12 to 16 are further down: see
"Known limit", "Not tried yet" and "Ruled out".

### 1. MLX behind the torch interface — KEPT

Only `UserOptimizedTransformer` changed. The harness has no change. The
class keeps its torch parameters, so `load_state_dict()`, `.to()` and
`.eval()` operate as before. `forward()` converts to MLX, calculates, and
converts back.

It uses `mx.fast.scaled_dot_product_attention`, `mx.fast.layer_norm` and
`mx.compile`.

The MLX weights build at the first call, not in `__init__`. The harness
copies the weights and moves the model after `__init__`. The warmup loop
pays this cost, so it stays out of the measurement.

float32 passes on every shape tested: with padding, with `--causal`, with
both, and with odd sizes (batch 3, seq 71, d_model 256, ffn 999).

### 2. `mx.compile` — KEPT

| State | Median |
|---|---|
| `use_mlx_compile = False` | 14.203 ms |
| `use_mlx_compile = True` | 13.401 ms |

Set `UserOptimizedTransformer.use_mlx_compile = False` to disable it.

### 3. float32 LayerNorm — KEPT

The torch baseline accumulates its LayerNorm in float32 for every input
type. The MLX code now does the same, and holds the LayerNorm weights in
float32. This gave no measured accuracy gain, but it is the correct policy
and it costs nothing.

### 4. Explicit float32 softmax — REVERTED

The baseline computes its softmax in float32 at line 111. I copied this
with an explicit score matmul, mask and softmax, in place of the fused
kernel. The aim was to pass the half precision tests.

| Path | float16 failures | bfloat16 failures |
|---|---|---|
| Explicit float32 softmax | 50 / 1572864 | 173161 / 1572864 |
| Fused `mx.fast.sdpa` | 49 / 1572864 | 172424 / 1572864 |

No gain. The fused kernel is simpler. Reverted.

Note: an early comparison seemed to favour the explicit path. That test was
wrong. Forcing `compute_dtype=float32` had also promoted the whole network
to float32, so it did not measure what it claimed. The table above is from
clean CLI runs.

### 5. `torch.compile` over the MLX class — FAILS

```
TypeError: cannot create weak reference to 'mlx.gc_func' object
```

Dynamo cannot trace the MLX objects. Do not use `--compile-user`. Use
`mx.compile` inside the class instead. `--compile-baseline` still operates.

### 6. Mask form selected by the input — KEPT

An array mask is necessary only when the batch has padding. Without padding
the kernel can get `"causal"` or no mask, which are its faster paths.
`forward()` tests `valid_token_mask.all()` and selects the form.

`padded` changes the graph, so it cannot be a traced argument of
`mx.compile`. The class holds one compiled variant for each form. Two
variants compile at most.

Measured in one process, 300 samples for each path, rounds interleaved:

| seq_len | mask=None | array mask | Gain | Saved |
|---|---|---|---|---|
| 128 | 15.250 ms | 15.292 ms | none (noise) | 0.04 ms |
| 512 | 85.091 ms | 88.557 ms | 3.9% | 3.47 ms |
| 2048 | 388.507 ms | 395.345 ms | 1.7% | 6.84 ms |

The gain is zero at the default shape, because the mask array is only
8 x 128 x 128 there.

The time saved grows with `seq_len`, but the share does not. It is largest at
seq 512 and smaller at seq 2048, because attention itself grows with
`seq_len**2` and quickly becomes the whole cost. Do not expect more than
about 4% from this change at any shape.

Kept because it costs two lines and it never makes the model slower.

### 7. Shape-aware kernel plan — KEPT

The Appendix 3.7 shapes span 7 orders of magnitude of work, from 0.13 GFLOP
at shape 2 to 2.7 PFLOP at shape 14. One kernel path is not correct across
that range.

`plan_kernels()` reads the shape and returns a `KernelPlan` with three
decisions: `fuse_qkv`, `causal_block` and `batch_chunk`. Attempts 8, 9 and
10 are those three decisions. Every threshold comes from a measurement. The
full tables are in [references/mlx-tensorops.md](references/mlx-tensorops.md).

Measured on each appendix shape, float32, no padding. The "before" column
forces `KernelPlan(fuse_qkv=False, causal_block=None, batch_chunk=None)`,
which is the single path of attempts 1 to 6. The "after" column lets
`plan_kernels()` choose.

| # | Shape | Before | After | Gain | Plan chosen |
|---:|---|---:|---:|---:|---|
| 13 | B64 D128 H4 S1024 | 285.551 ms | 181.712 ms | **1.57x** | blk64 |
| 6 | B10000 D128 H4 S128 | 2093.1 ms | 1602.7 ms | **1.31x** | blk64 + chunk 1024 |
| 11 | B64 D128 H16 S128 | 21.002 ms | 17.955 ms | **1.17x** | blk32 |
| 7 | B64 D32 H4 S128 | 8.046 ms | 7.166 ms | **1.12x** | blk32 |
| 1 | B64 D128 H4 S128 | 10.784 ms | 10.245 ms | 1.05x | blk64 |
| 2 | B1 D128 H4 S128 | 0.785 ms | 0.752 ms | 1.04x | full |
| 4 | B16 D128 H4 S128 | 2.605 ms | 2.514 ms | 1.04x | blk64 |
| 5 | B128 D128 H4 S128 | 21.222 ms | 20.450 ms | 1.04x | blk64 |
| 3 | B4 D128 H4 S128 | 1.150 ms | 1.125 ms | 1.02x | full |
| 10 | B64 D128 H2 S128 | 7.307 ms | 7.218 ms | 1.01x | full |
| 8 | B64 D1024 H4 S128 | 137.876 ms | 138.038 ms | 1.00x | full |
| 12 | B64 D128 H4 S32 | 2.434 ms | 2.464 ms | 0.99x | full |
| 9 | B64 D128 H1 S128 | 7.147 ms | 7.290 ms | 0.98x | full |

Where the plan chooses `full`, the two columns run the same code. The 0.98x
and 0.99x rows are therefore measurement noise, not a loss.

The default shape of this log (batch 8, seq 128, d_model 512, heads 8) gets
`head_dim = 64`, so the plan chooses `full` and only attempt 9 applies. This
work does not move the headline number. It moves the appendix shapes.

Accuracy is unchanged. Every runnable appendix shape, at `padding_ratio` 0.0
and 0.3: `max_abs_error` between 1.19e-06 and 2.50e-06, and 0 failed
elements at `atol=0.002`, `rtol=0.02`.

Set `UserOptimizedTransformer.plan_override` to a `KernelPlan` before the
first call to force a path. The table above was measured that way.

### 8. Blocked causal attention — KEPT

`mx.fast.scaled_dot_product_attention` does **not** skip the masked
triangle. It calculates the whole square and then applies the mask. At
B=64, H=4, S=1024, head_dim=32:

| mask argument | Time |
|---|---:|
| `None` | 38.5 ms |
| `"causal"` | 52.7 ms |
| explicit bool array | 55.1 ms |

`"causal"` is slower than no mask at all. This does not contradict attempt
6: `"causal"` still beats the array mask by 4.5% here, which agrees with the
3.9% that attempt 6 measured at seq 512. Attempt 6 picks the better of two
masks. This attempt removes the masked work.

Split the query into blocks. Block `i` covers rows `[start, stop)` and
receives only keys `[0, stop)`. The masked tiles are never built. MLX aligns
a `"causal"` mask to the **end** of the key sequence, so a query block of
length `stop - start` against keys `[0, stop)` gets exactly the right rows.

**It is bit exact.** `max_abs_diff = 0.0` against the unblocked call. It is
not an approximation.

Block size sweep, milliseconds. The best of each row is marked:

| B | H | S | head_dim | full | blk16 | blk32 | blk64 | Best |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 4 | 1024 | 32 | 52.00 | 48.50 | 36.32 | **31.90** | blk64, 1.63x |
| 64 | 16 | 128 | 8 | 3.408 | 3.236 | **2.649** | 2.709 | blk32, 1.29x |
| 64 | 4 | 128 | 8 | 0.855 | 0.872 | **0.687** | 0.693 | blk32, 1.24x |
| 64 | 4 | 128 | 32 | 1.001 | 1.136 | 0.917 | **0.895** | blk64, 1.12x |
| 1024 | 4 | 128 | 32 | 15.58 | 17.19 | 14.07 | **14.07** | blk64, 1.11x |
| 128 | 4 | 128 | 32 | 2.049 | 2.318 | 1.911 | **1.904** | blk64, 1.08x |
| 16 | 4 | 128 | 32 | 0.267 | 0.338 | 0.257 | **0.247** | blk64, 1.08x |
| 64 | 2 | 128 | 64 | **0.194** | 0.500 | 0.310 | 0.281 | full |
| 64 | 1 | 128 | 128 | **0.208** | 0.522 | 0.318 | 0.303 | full |
| 64 | 4 | 128 | 256 | **2.549** | 4.919 | 3.726 | 3.154 | full |
| 64 | 4 | 32 | 32 | **0.139** | 0.193 | — | — | full |
| 4 | 4 | 128 | 32 | **0.088** | 0.246 | 0.134 | 0.093 | full |
| 1 | 4 | 128 | 32 | **0.050** | 0.245 | 0.129 | 0.074 | full |

Three rules cover all 13 rows. Do not block when:

1. **`head_dim >= 64`.** A wide head is already efficient, so blocking only
   adds kernel launches. `mx.fast.sdpa` reaches 1390 GFLOP/s at
   `head_dim=128`, 760 at 256, 270 at 32 and 78 at 8. Peak float32 matmul on
   this GPU is about 3500 GFLOP/s.
2. **`seq_len <= 64`.** There is no triangle worth skipping.
3. **`batch * heads < 64`.** The GPU is not full, so launch cost is larger
   than the saved arithmetic.

Otherwise block, at 32 for `head_dim <= 16` and at 64 above it.

### 9. Fused QKV matmul — KEPT

One `[D, 3D]` matmul in place of three `[D, D]` matmuls. It removes two
kernel launches for each layer.

| B | S | D | 3 separate | 1 fused | Gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 0.0439 ms | 0.0220 ms | **1.99x** |
| 64 | 128 | 32 | 0.1094 ms | 0.0961 ms | 1.14x |
| 64 | 128 | 128 | 0.6202 ms | 0.5524 ms | 1.12x |
| 128 | 128 | 128 | 1.0695 ms | 1.0560 ms | 1.01x |
| 1024 | 128 | 128 | 6.8189 ms | 6.7662 ms | 1.01x |
| 64 | 1024 | 128 | 3.5790 ms | 3.5689 ms | 1.00x |
| 64 | 128 | 1024 | 14.3045 ms | 14.2692 ms | 1.00x |
| 64 | 32 | 128 | 0.1319 ms | 0.1340 ms | 0.98x |
| 4 | 128 | 128 | 0.0436 ms | 0.0460 ms | 0.95x |

The gain is large only at batch 1, where the launch is most of the call. It
never loses outside noise, so it is always on. It also makes the compiled
graph smaller.

Build the weight once, when the MLX weights are copied, not for each call:

    qkvw = mx.concatenate([qw, kw, vw], axis=0).T    # [D, 3D]
    qkvb = mx.concatenate([qb, kb, vb], axis=0)      # [3D]

`mx.split(h @ qkvw + qkvb, 3, axis=-1)` returns q, k and v in that order.

### 10. Batch chunking — KEPT

Shape 6 of the appendix is batch 10000. One activation is 625 MiB, and a
layer holds five of them. The GPU working set is 12.0 GiB.

Run the **whole depth** for one batch chunk, then the next. Only one chunk of
intermediates is live at a time. One layer of shape 6:

| chunk | Time | Peak memory |
|---:|---:|---:|
| 10000 (no chunk) | 284.1 ms | 8.55 GiB |
| 4096 | 286.1 ms | 6.71 GiB |
| 2048 | 276.8 ms | 6.10 GiB |
| 1024 | 271.9 ms | 5.60 GiB |
| 512 | 271.6 ms | 5.60 GiB |
| 256 | 271.4 ms | 3.86 GiB |

**Chunking is not slower.** It is slightly faster, because a chunk fits the
cache better. It costs nothing and it removes the memory limit.

The rule: keep one chunk activation at or under 64 MiB.

    chunk = 64 MiB // (seq_len * d_model * itemsize)

This gives chunk=1024 for shape 6 and no chunking for every other shape of
the table. On the whole model, shape 6 goes from 9.16 GiB peak to 2.68 GiB.

### 11. Pad `head_dim` from 8 up to 32 — REVERTED

`mx.fast.sdpa` reaches only 78 GFLOP/s at `head_dim = 8`, against 270 at 32
and 1390 at 128. Shapes 7 and 11 of the appendix both have `head_dim = 8`.

Padding q, k and v with zeros up to `head_dim = 32` is exact. Zeros add
nothing to the dot product, and the extra output columns are discarded. But
it multiplies the arithmetic by 4 while efficiency rises only 3.4x.

| Shape | Result |
|---|---|
| 7 (B64 D32 H4) | 0.91x — slower |
| 11 (B64 D128 H16) | 1.17x |

Blocking (attempt 8) gives 1.24x and 1.29x on the same two shapes, and it
adds no arithmetic. Blocking wins. Reverted.

### 17. Pad `head_dim` to 64 to reach the fused SDPA kernel — KEPT

Attempt 11 padded 8 up to 32 and lost. It aimed at the wrong target. Row 18
shows why: 32 is still on the fallback kernel, so that test never reached
the fused kernel.

#### Why it works

The gain is **not** arithmetic. The padded path does *more* arithmetic than
the path it replaces. The gain is DRAM traffic that stops happening.

The two kernels of row 18 differ in what they put in memory:

| | fallback | fused (flash) |
|---|---|---|
| `S x S` scores | written to DRAM, read back for the mask, read again for softmax, read again for `@V` | held in registers and threadgroup memory, never written to DRAM |
| causal triangle | built, then masked away | skipped, measured 0.55x |
| DRAM cost | grows as `S * S` | grows as `S * head_dim` |

So the fallback pays for the score matrix and the fused kernel does not.
That matrix is the largest object in the layer. At shape 13:

| Item, one layer | Size |
|---|---:|
| `S x S` score matrix, `B*H*S*S*4` | **1.000 GiB** |
| q, k, v operands, `3*B*S*D*4` | 96.0 MiB |
| ratio | **10.7x** |

Move 1.000 GiB across 4 layers, two or three times each, at the 150 GB/s of
this machine:

| passes over the score matrix | traffic | time |
|---:|---:|---:|
| 2 | 8.00 GiB | 57.3 ms |
| 3 | 12.00 GiB | 85.9 ms |

**Measured saving: 182.790 - 111.195 = 71.6 ms.** It sits between the two
estimates. The win is the score-matrix traffic, and nothing else.

Against that, the pad costs `rho = 2`:

| Item | Before | After |
|---|---:|---:|
| QKV projection | 25.77 GFLOP | 51.54 GFLOP, about +8.6 ms |
| attention matmuls | 137.44 GFLOP | 274.88 GFLOP, then 0.55x back from the causal skip |

Trading 8.6 ms of extra projection for 71.6 ms of removed traffic is the
whole optimization.

#### Why it works only at a long sequence

The score matrix is `S x S` and the operands are `S x head_dim`. Their ratio
is `S / head_dim`. That single number decides how much traffic the fused
kernel removes:

| # | S | head_dim | `S / head_dim` | result |
|---:|---:|---:|---:|---|
| 13 | 1024 | 32 | **32** | 1.644x |
| 11 | 128 | 8 | 16 | needs `rho = 8`, loses |
| 7 | 128 | 8 | 16 | needs `rho = 8`, loses |
| 1, 5, 6 | 128 | 32 | 4 | inside the noise floor |
| 12 | 32 | 32 | 1 | loses |
| 9 | 128 | 128 | 1 | already fused |

At `S = D` the score matrix is no larger than the activations, so there is
little traffic to remove and the pad only adds arithmetic. The threshold
`S >= 4*D` is a statement about this ratio.

#### Why `head_dim = 8` cannot use it

Shapes 7 and 11 have the second largest `S / head_dim` in the table, so they
have traffic worth removing. They still lose, for a reason outside
attention: reaching 64 from 8 needs `rho = 8`, and `rho` multiplies the QKV
projection, which is `6*D*D` per token and has nothing to do with attention.
An 8x projection is not payable at any sequence length. Shape 11 also loses
blocking, which had given it 1.29x. That is why it fell to 0.756x.

Those two shapes need a kernel, not a pad. See row 14.

The pad is exact. Zeros in q and k add nothing to the dot product, zeros in
v add nothing to the output, and the extra output columns get discarded.
`scale` stays at the TRUE `head_dim`, so the softmax does not move. The pad
is folded into the fused QKV weight, which becomes `[D, 3*H*64]` with zero
columns, so the projection writes the wide layout and costs no extra pass.

**A blanket `head_dim < 64` rule loses.** Full sweep, MLX ms. The noise
floor is +-4%, taken from shapes 8, 9 and 10, whose path did not change:

| # | D | S | head_dim | rho | before | after | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 32 | 2 | 10.141 | 9.497 | 1.068x |
| 4 | 128 | 128 | 32 | 2 | 2.495 | 2.696 | 0.925x |
| 5 | 128 | 128 | 32 | 2 | 20.527 | 18.582 | 1.105x |
| 6 | 128 | 128 | 32 | 2 | 1677.834 | 1718.289 | 0.976x |
| 7 | 32 | 128 | 8 | 8 | 7.100 | 7.942 | 0.894x |
| 11 | 128 | 128 | 8 | 8 | 17.257 | 22.819 | **0.756x** |
| 12 | 128 | 32 | 32 | 2 | 2.386 | 2.656 | 0.898x |
| 13 | 128 | 1024 | 32 | 2 | 182.790 | **111.044** | **1.646x** |

    .venv/bin/python3 scoreboard.py --cases all \
        --label "pad head_dim to 64 to reach the fused SDPA kernel"

FLOP-weighted, the blanket rule gives **0.983x**, a net loss. Shape 6 is
66.5% of the weight and it measured 0.976x.

The pad multiplies the QKV projection and the attention arithmetic by
`rho = 64 / head_dim`. Per token per layer the projection is `6*D*D` and
attention is `4*S*D`. Two conditions gate it:

1. `rho <= 2`, so `head_dim >= 32`. At `head_dim = 8` the projection grows
   8x. Shape 11 also loses blocking, which had given it 1.29x. That is why
   it falls to 0.756x.
2. `S >= 4*D`, so attention dominates the layer. Every shape with `S == D`
   landed inside the noise floor.

Shape 11 also failed a mechanistic cost model built on arithmetic alone:
the model predicted the sign on only 6 of 10 shapes, and every miss sat
within 10% of 1.0. Arithmetic does not decide the small cases. The memory
tail and the launch count do.

**Threshold 2 rests on ONE point.** See row 19.

Accuracy is unchanged: `max_abs` 1.07e-06 to 2.86e-06 at `atol=0.002` and
`rtol=0.02`, over all 13 shapes, at `padding_ratio` 0.0 and 0.3.

### 18. MLX dispatches SDPA to two different kernels — a fact

`mx.fast.scaled_dot_product_attention` uses a fused flash kernel **only for
`head_dim` 64 to 128**. Outside that range it accepts the call, returns a
correct answer, and uses a fallback that materializes `B x H x S x S`.

Measured by peak GPU memory at B=8, H=8, S=2048, where the score matrix
would be 1024 MiB. The column is `peak - base`:

| head_dim | 8 | 16 | 32 | 48 | 64 | 72 | 96 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| peak MiB | 1048 | 1068 | 1108 | 1148 | **128** | **144** | **192** | **256** | 1668 |

Of the appendix shapes, only 9 (`head_dim=128`) and 10 (`head_dim=64`)
reach the fused kernel by themselves.

This corrects two earlier readings:

- The "18x efficiency curve against `head_dim`" of attempt 8 is not a
  curve. It is a cliff between two kernels.
- `mask="causal"` is slower than `mask=None` on the **fallback** only. On
  the fused kernel it is **0.55x**, measured at B=32, H=8, S=1024,
  `head_dim=64`: 11.08 ms against 20.31 ms. The fused kernel skips the
  masked blocks itself, so blocking above it only adds launches.

### 19. Sweep `seq_len` to place the `S >= 4*D` threshold — OPEN

Row 17 holds one measured point on the winning side: shape 13, at `S = 8*D`.
The threshold sits at `S >= 4*D`, between shape 1 (`S = D`, no gain) and
shape 13. Nothing between those two is measured.

Hold `D=128`, `head_dim=32`, `B=64`. Sweep `S` through 128, 256, 512, 1024
and 2048, with the padded and the unpadded path, end to end. Move the
threshold to where the curve crosses 1.0. Do not move it on a model: the
arithmetic model already failed on 4 of 10 shapes.

## Cost of the framework boundary (row 23)

`forward()` converts the input to MLX and the output back to torch. The
timed region holds both conversions. `benchmark_once()` throws the returned
tensor away, but it builds the tensor before the stopwatch stops.

The share grows with the size of the activation:

    .venv/bin/python3 profiling/zero_copy.py

| Shape | total ms | to MLX (in) | `.all()` | MLX calculation | to torch (out) | torch share |
|---|---:|---:|---:|---:|---:|---:|
| 2 B1 D128 S128 | 0.781 | 0.044 | 0.001 | 0.767 | 0.002 | 5.9% |
| 1 B64 D128 S128 | 10.060 | 0.099 | 0.001 | 9.643 | 0.155 | 2.5% |
| 8 B64 D1024 S128 | 136.151 | 2.012 | 0.001 | 131.400 | 1.496 | 2.6% |
| 6 B10000 D128 S128 | 1677.975 | 17.439 | 0.052 | 1494.072 | 59.246 | 4.6% |

An earlier version of this section reported 0.8% at the default size only.
That number is correct for that size. It hid the cost at shape 6, which
carries 66.5% of the FLOP-weighted score.

### The output copy was avoidable

MLX allocates every array in unified memory, so the CPU reads an MLX buffer
with no transfer. `np.asarray` returns a view of that buffer. `np.array`
allocates and copies. `_to_torch()` called `np.array`.

Copy rates at 64 MiB, from the same script:

| Direction | Result | Rate |
|---|---|---|
| `mx.array(numpy)` | copy | 24 to 51 GB/s |
| `np.array(mlx)` | copy | 15 to 22 GB/s |
| `np.asarray(mlx)` | **view** | 0.0008 ms at 625 MiB |

The input copy stays. MLX allocates from its own pool, so `mx.array` must
own the bytes. The output copy is now a view.

Measured at shape 6, 12 interleaved samples for each arm, same process and
same model instance:

| `_to_torch` | median | min |
|---|---:|---:|
| `np.array` (copy) | 1590.2 ms | 1554.4 ms |
| `np.asarray` (view) | 1518.6 ms | 1500.5 ms |

Gain: 71.6 ms on the median, 1.047x. Interleave the samples. A plain
before/after at shape 6 gave the wrong sign, because the shape drifts by
about 200 ms between runs.

### The view is honest and it is safe

`mx.eval(output)` in `forward()` already finishes the GPU work. After it,
`np.asarray` costs 0.02 ms and a read of the first element costs 0.02 ms.
So the removed copy was overhead, not a disguised wait. `mx.eval` blocks:
a 15-step matmul chain took 510.27 ms with `mx.eval` alone and 509.39 ms
with `mx.eval` plus `mx.synchronize()`.

Checks that passed:

- `torch.equal` against the copying version at shapes 1, 8 and 6. Bit exact.
- The harness at its defaults: PASS, `max_abs=2.98023e-06`, 5.486x.
- Lifetime. The chain tensor -> ndarray -> memoryview -> `mx.array` holds a
  reference. The values survived 50 rounds of MLX allocator churn after the
  source array went out of scope.
- float32 and float16 alias the MLX buffer. bfloat16 casts first, and a
  dtype or device change in `.to()` copies, so both give an independent
  tensor.

One condition: the returned tensor shares memory with the MLX array. A
write to the tensor changes the array. `compare_outputs()` only reads it.

## Backend comparison: how much is the GPU, how much is MLX

`test_backends.py` runs the same weights and byte-identical inputs on three
backends, with a device synchronize on both sides of each call. Results are
in `profiling/backend_comparison.json`, 300 samples for each backend:

| Backend | Model | Median | Against CPU |
|---|---|---|---|
| cpu | baseline, torch CPU kernels | 91.72 ms | 1.00x |
| mps | baseline, torch Metal kernels | 20.08 ms | 4.57x |
| mlx | optimized, MLX | 15.30 ms | 5.99x |

All three pass the float32 accuracy test against the CPU baseline.

This gives the split the log wanted:

- **The GPU gives 4.57x.** The same baseline code, moved to Metal.
- **MLX gives 1.31x more** on that same GPU. This part is the kernels:
  `mx.fast.sdpa`, `mx.fast.layer_norm` and `mx.compile`.

Use the MPS row for a stable comparison. The CPU row moves with the machine
state; the two GPU rows do not.

## Known limit: float16 and bfloat16 cannot pass (row 13)

| Type | Failures at `atol=0.002` |
|---|---|
| float32 | 0 / 2621440 |
| float16 | 49 / 1572864 (0.003%) |
| bfloat16 | 172424 / 1572864 (11%) |

No implementation can pass these. Proof by control experiment: I wrote a
second optimized model in pure PyTorch with
`torch.nn.functional.scaled_dot_product_attention`, which line 192 of the
file names as an example optimization.

| bfloat16, baseline against: | Failures |
|---|---|
| torch `F.sdpa` (no MLX) | 34687 / 524288 |
| MLX (this class) | 57482 / 524288 |

PyTorch's own suggested optimization fails.

The cause is arithmetic. One bfloat16 step at magnitude 1.0 is 0.0078. The
atol is 0.002. The step is 4 times the tolerance, so the atol test cannot
absorb one different rounding. The rtol test then fails wherever the
reference value is near zero. For float16 the step is 0.00098, just under
atol, which is why float16 almost passes.

These thresholds are float32 thresholds. Do not raise `--atol` to hide
this.

## Tools

| Tool | Use |
|---|---|
| `test_backends.py` | The three backends, side by side. Writes the JSON above. |
| `test_padding.py` | The padded batch. The sweep never runs one. Rows 27 and 28. |
| `appendix_cases.py` | The Appendix 3.7 shapes. `--cases 1,7-9 --run`. |
| `profiling/` | Instruments traces and Metal GPU captures. See its README. |

Shape 14 of the appendix (B=32, D=1024, H=16, S=100000, L=2) is disabled.
The input alone is 12.2 GiB in float32, and `BaselineSelfAttention` builds a
B x H x S x S score matrix, which is 18.6 TiB at that shape.

## Not tried yet (rows 14 to 16)

- **A hand-written Metal kernel by `mx.fast.metal_kernel`, for
  `head_dim = 8`.** This is the largest gap that remains. `mx.fast.sdpa`
  reaches 1390 GFLOP/s at `head_dim = 128` but only 78 at 8, an 18x swing.
  Shapes 7 and 11 both sit in it, and blocking recovers only 1.24x to 1.29x.
  A kernel that holds a whole 8-wide head in registers should recover more.
  It fits the design as one more `KernelPlan` branch.
- **Fused FFN.** Check whether `mx.compile` already joins the GELU with the
  two matmuls. If not, write a custom Metal kernel.
- **Quantization.** `mx.quantize` for the linear layers. This will fail the
  accuracy test, so measure the speed only.

## Ruled out (row 12)

- **Full port of the file to MLX.** Out of scope. `BaselineTransformer` is
  the benchmark: it is the reference for accuracy and for speed, so it does
  not change and it does not move to a different framework. Only
  `UserOptimizedTransformer` is optimized. See `CLAUDE.md`.

  The question this port was wanted for is answered anyway. The backend
  comparison gives the split (GPU 4.57x, MLX kernels 1.31x) with a read-only
  copy of the baseline in `test_backends.py`, which leaves the class itself
  untouched.

## Done since the first version of this log

- ~~**A torch MPS baseline.**~~ Done. See the backend comparison above.
- ~~**One matmul for q, k and v.**~~ Done. Attempt 9.

## 25. Hoist MLX's `steel_attention` and compile it at an unshipped `head_dim`

`mx.fast.scaled_dot_product_attention` reaches the fused flash kernel for
five head widths only: 64, 72, 80, 96 and 128 (row 20). Every other width
takes a fallback that materializes the whole `B x H x S x S` score matrix.
Ten of the fourteen appendix shapes have `head_dim` 8 or 32, so ten shapes
were on the fallback.

MLX gives no way to name a kernel and run it. `mx.fast` holds only
`scaled_dot_product_attention`, which applies the width check first, and
`metal_kernel`, which compiles source you supply. But the wheel ships the
Metal SOURCE of the fused kernel:

    .venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/
      kernels/steel/attn/

The kernel is a C++ template. `BD` is the head width and the template only
needs `BD % 8 == 0`, because its MMA fragment is 8 wide. Apple compiled five
values. Nothing stops us from compiling another one.

`steel_attention.py` inlines the nine-header dependency chain and makes
three edits, listed in that file. The arithmetic is untouched.

**It is the real flash kernel.** Peak GPU memory at B=8, H=8, S=1024,
`head_dim=32`, where the score matrix would be 256 MiB:

| path | extra peak |
|---|---:|
| `mx.fast.scaled_dot_product_attention` | 273.0 MiB |
| hoisted steel `bd32` | **8.0 MiB** |

Attention step alone, causal float32, against the path each shape used
before:

| case | path before | before | after | gain |
|---|---|---:|---:|---:|
| shape 6 chunk, B1024 H4 S128 D32 | fallback blk64 | 13.838 | 2.261 | **6.12x** |
| shape 11, B64 H16 S128 D8 | fallback blk32 | 2.921 | 0.358 | **8.16x** |
| shape 1, B64 H4 S128 D32 | fallback blk64 | 1.070 | 0.323 | 3.31x |
| shape 13, B64 H4 S1024 D32 | pad64 fused | 11.061 | 5.166 | 2.14x |
| shape 7, B64 H4 S128 D8 | fallback blk32 | 0.834 | 0.318 | 2.62x |

Full sweep, MLX milliseconds. Reproduce with
`.venv/bin/python3 scoreboard.py --label "..."`:

| # | steel | before | after | gain | FLOP share |
|---:|:---:|---:|---:|---:|---:|
| 6 | yes | 1772.615 | 1346.945 | **1.32x** | 66.5% |
| 8 | no | 135.822 | 135.210 | 1.00x | 21.3% |
| 13 | yes | 111.195 | 75.789 | **1.47x** | 9.4% |
| 5 | yes | 20.236 | 17.548 | 1.15x | 0.9% |
| 11 | yes | 17.074 | 7.800 | **2.19x** | 0.4% |
| 1 | yes | 9.966 | 8.004 | 1.25x | 0.4% |
| 9 | no | 7.051 | 6.855 | 1.03x | 0.4% |
| 10 | no | 6.935 | 6.824 | 1.02x | 0.4% |
| 7 | yes | 6.945 | 4.701 | **1.48x** | 0.0% |
| 4 | yes | 2.507 | 2.227 | 1.13x | 0.1% |
| 12 | yes | 2.407 | 2.142 | 1.12x | 0.1% |
| 3 | yes | 1.128 | 1.106 | 1.02x | 0.0% |
| 2 | yes | 0.775 | 0.644 | 1.20x | 0.0% |

**FLOP-weighted: 1218.8 ms to 932.2 ms, which is 1.308x.**

Accuracy PASS on every shape, `max_abs` 1.31e-06 to 1.91e-06.

**Read the MPS column, not the CPU column.** The CPU baseline of that sweep
ran on a loaded machine: shape 6 went 18943 ms to 24763 ms between sweeps on
a baseline this project forbids changing. So the CPU speedup moved from
10.57x to 17.97x FLOP-weighted, and most of that is the reference, not the
kernel. MPS held steady (shape 8 165.4 to 165.8, shape 11 41.9 to 42.0), and
against MPS the gain is **1.60x to 2.12x FLOP-weighted**.

Limits:

1. The kernel takes a string causal mask only. A padded batch keeps the old
   path. This is a correctness gate. **The first version of row 25 stated
   this rule but never applied it.** Row 27 applies it.
2. `head_dim` must be a multiple of 8.
3. Q plus K threadgroup memory must fit 32 KiB, which caps `head_dim` at 96
   with `BQ=32, BK=32`. See row 26.

## 26. Reach the steel kernel at `head_dim = 256` for shape 8

Shape 8 is `d_model=1024`, `num_heads=4`, so `head_dim=256`, and it carries
21.3% of the FLOP weight. It is the largest shape row 25 cannot take.

`BQ=32, BK=32, BD=256` needs 68.5 KiB of threadgroup memory against a 32 KiB
limit. `BK=16` needs about 41 KiB, still over. `BK=8` would fit and is not
tried.

**The ceiling is small.** Shape 8 is 7.6% attention by measured time, not the
4% the FLOP table implies, because `d_model=1024` makes the projections 92%
of the work. A perfect attention kernel would give about 1.5% of the
weighted score.

## 27. Gate the steel kernel on a string mask — a bug fix

`plan_kernels()` sets `steel` from the shape alone. The shape does not say if
a batch has padding, so `plan.steel_attention` stayed True for a padded
batch. `_attention()` then ran:

```python
return steel_sdpa(q, k, v, scale, causal=mask == "causal")
```

`mask` is an `mx.array` for a padded batch. `mx.array == "causal"` returns
the Python `False`, not an array:

```
>>> mx.ones((2, 2), dtype=mx.bool_) == "causal"
False
```

So the kernel dropped the token mask **and** the causal mask, for the whole
batch. Every sample lost its causal mask, not only the padded samples.

Reproduce the failure on the parent commit:

    .venv/bin/python3 torch_transformer_benchmark.py --causal --heads 16 \
        --padding-ratio 0.25 --accuracy-trials 2
    summary: FAIL | max_abs=2.71282 | max_rel=321908 | failed=822894/1048576

The same command after the fix:

    summary: PASS | max_abs=3.03611e-06 | max_rel=2.173 | failed=0/1572864

The fix adds `isinstance(mask, str)` to the steel branch. A padded batch
takes `mx.fast.scaled_dot_product_attention` with the array mask.

**No recorded sweep saw this.** `--padding-ratio` defaults to 0.0 in
`torch_transformer_benchmark.py`, in `scoreboard.py` and in
`appendix_cases.py`, so every number in this log ran on an unpadded batch.
The padded graph has no coverage in the sweep. Add a padded run when you
change the mask path.

## 28. Drop the token masks that cannot change the output — KEPT

The baseline applies `masked_fill` three times for each layer: on the
attention output (L124), at the end of the block (L147), and after the final
LayerNorm (L174). `_mlx_transformer()` copied all three. Two of them cannot
change the result.

**The attention output mask is redundant.** Attention is the only operation
that mixes the positions, and it runs before the mask. LayerNorm, GELU and
both FFN matmuls act on one position at a time. So a value at a padded
position never reaches a valid position, and the mask at the end of the
block removes it. A NaN from a fully masked query row goes the same way:
`mx.where` selects a value, so it does not spread the NaN.

**All three are no-ops without padding.** `forward()` sets `padded = not
valid_token_mask.all()`, and `padded` selects a separate compiled graph
(row 6). Inside the unpadded graph every token is valid, so each
`mx.where(valid_tokens, y, 0)` returns `y`. The unpadded graph now holds no
mask operation.

The end-of-block mask stays for a padded batch. It clears the FFN output.
The final mask stays as well: LayerNorm returns the bias at a zeroed
position, not zero.

Bit exact. 18 cases, causal and non-causal, `head_dim` 8, 16 and 64, with an
all-valid mask, a ragged mask, and a mask that empties one whole sample:

    PYTHONPATH=. .venv/bin/python3 <scratchpad>/check_exact.py
    ALL BIT EXACT

Full sweep, MLX milliseconds, against the sweep at commit `92be893`:

    .venv/bin/python3 scoreboard.py --label "rows 27 and 28, re-measured on AC power"

| # | before | after | gain | FLOP share |
|---:|---:|---:|---:|---:|
| 6 | 1211.252 | 1153.598 | **1.050x** | 66.5% |
| 8 | 134.105 | 133.563 | 1.004x | 21.3% |
| 13 | 75.603 | 75.672 | 0.999x | 9.4% |
| 5 | 15.281 | 15.153 | 1.008x | 0.9% |
| 11 | 7.754 | 7.698 | 1.007x | 0.4% |
| 1 | 7.670 | 7.594 | 1.010x | 0.4% |
| 9 | 6.819 | 6.869 | 0.993x | 0.4% |
| 10 | 6.728 | 6.705 | 1.003x | 0.4% |
| 7 | 4.742 | 4.784 | 0.991x | 0.0% |
| 4 | 2.197 | 2.149 | 1.022x | 0.1% |
| 12 | 2.146 | 2.139 | 1.003x | 0.1% |
| 3 | 0.977 | 0.976 | 1.001x | 0.0% |
| 2 | 0.635 | 0.618 | 1.028x | 0.0% |

**FLOP-weighted MLX: 841.6 ms to 803.2 ms, which is 1.048x.**

The two reference backends held still across the same pair of sweeps, so the
noise floor is about 1% and the MLX gain sits above it:

| Backend | before | after | ratio |
|---|---:|---:|---:|
| MLX | 841.6 ms | 803.2 ms | **1.048x** |
| MPS | 1868.8 ms | 1850.9 ms | 1.010x |
| CPU | 9552.6 ms | 9600.6 ms | 1.005x |

Shape 6 carries the gain, at 1.050x and 66.5% of the FLOP weight. That fits
the change. Shape 6 is B=10000 and it runs in chunks of 1024, so it moves
more activation bytes than any other shape, and this change removes one
elementwise pass over `B x S x D` for each layer. Shape 8 gains almost
nothing (1.004x), which also fits: `d_model=1024` makes the projections 92%
of its work.

`mx.compile` joins each removed `mx.where` with the add beside it, so the
change removes arithmetic and one input, not a kernel launch. Only the final
mask is a kernel of its own.

**Measure this on AC power.** The first sweep of this change ran on battery.
The CPU baseline came out 1.9x slow, MPS 6% slow, and MLX 2% fast, which
gave a useless comparison. That reading stays in `profiling/history.jsonl`
with the label `rows 27 and 28, ON BATTERY`. Do not compare against it.

Accuracy on the sweep: PASS on all 13 shapes, `max_abs` 1.07e-06 to
2.65e-06.

## 29. `mx.addmm` for every projection — KEPT

`_mlx_transformer()` wrote each projection as `h @ w + b`. MLX runs that as
two kernels: the matmul writes the output to DRAM, and the add reads the
whole output back, adds the bias, and writes it again.

`mx.addmm(b, h, w)` gives the bias to the matmul as its C operand, so the
GPU adds it inside the matmul kernel. One kernel replaces two, and one full
pass over the output disappears.

**`mx.compile` does not do this.** That was the surprise. The compiled graph
keeps the add as its own kernel, and the compiled time equals the raw time:

    .venv/bin/python3 <scratchpad>/addmm_micro.py

| Projection | `h @ w + b` raw | compiled | `mx.addmm` raw | compiled |
|---|---:|---:|---:|---:|
| B64 S128 D128->384 (shape 1) | 0.766 ms | 0.778 ms | 0.645 ms | 0.688 ms |
| B1024 S128 D128->384 (shape 6 chunk) | 6.768 ms | 6.753 ms | **3.749 ms** | 3.742 ms |
| B64 S1024 D128->384 (shape 13) | 3.503 ms | 3.507 ms | **1.998 ms** | 1.997 ms |
| B64 S128 D1024->3072 (shape 8) | 14.327 ms | 14.218 ms | 12.725 ms | 12.704 ms |

The gain follows the output size, because the kernel that disappears is
memory-bound. Shape 8 gains least: its matmul is arithmetic-bound, so the
extra pass costs a smaller share.

Seven call sites changed: the fused QKV, the three unfused QKV projections,
the attention output, and both FFN matmuls.

Full sweep, MLX milliseconds, against the sweep at the same commit:

    .venv/bin/python3 scoreboard.py --label "row 29: mx.addmm for every projection"

| # | before | after | gain | FLOP share |
|---:|---:|---:|---:|---:|
| 6 | 1153.598 | 1050.066 | **1.099x** | 66.5% |
| 8 | 133.563 | 128.463 | 1.040x | 21.3% |
| 13 | 75.672 | 69.288 | **1.092x** | 9.4% |
| 5 | 15.153 | 13.571 | 1.117x | 0.9% |
| 11 | 7.698 | 7.061 | 1.090x | 0.4% |
| 1 | 7.594 | 6.889 | 1.102x | 0.4% |
| 9 | 6.869 | 6.268 | 1.096x | 0.4% |
| 10 | 6.705 | 6.151 | 1.090x | 0.4% |
| 7 | 4.784 | 4.761 | 1.005x | 0.0% |
| 4 | 2.149 | 2.091 | 1.028x | 0.1% |
| 12 | 2.139 | 2.007 | 1.066x | 0.1% |
| 3 | 0.976 | 0.956 | 1.021x | 0.0% |
| 2 | 0.618 | 0.748 | 0.827x | 0.0% |

**FLOP-weighted MLX: 803.2 ms to 732.6 ms, which is 1.096x.**

Shape 2 is the one row that moved backward. It runs one sample in 0.7 ms,
where a kernel launch is most of the time, and a paired test of the same two
builds gave 1.004x for it. Treat the sweep row as noise, not as a
regression.

The two reference backends held still across the pair of sweeps, so the
noise floor is about 1% to 2%:

| Backend | before | after | ratio |
|---|---:|---:|---:|
| MLX | 803.2 ms | 732.6 ms | **1.096x** |
| MPS | 1850.9 ms | 1872.1 ms | 0.989x |
| CPU | 9600.6 ms | 9388.9 ms | 1.023x |

A paired test confirms the sweep. It alternates the two builds inside one
process, one call each, 30 rounds, so a drift of the machine hits both:

    .venv/bin/python3 <scratchpad>/ab_paired.py 6,8 10

| # | 6 | 8 | 13 | 1 | 11 | 5 | 9 | 10 | 12 | 7 | 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gain | 1.118x | 1.049x | 1.089x | 1.108x | 1.120x | 1.120x | 1.105x | 1.098x | 1.046x | 1.013x | 1.004x |

**Accuracy.** All 13 shapes PASS, `max_abs` 1.31e-06 to 1.91e-06. The output
is bit exact against the build before this change, on 16 float32 cases:
shapes 1, 2, 7, 8, 9, 11, 12 and 13, each with `--padding-ratio` 0.0 and
0.3, every one `max_abs=0.0`.

float16 is **not** bit exact: shape 8 differs by 7.81e-03, which is one
float16 step at that magnitude. The bias enters the sum at a different point
in the kernel, so half precision rounds differently. Row 13 rules out
float16 anyway.

## 30. Flatten the block to rank 2 before each projection — REVERTED

The idea copies a known torch pattern. `F.linear` on a `[B, S, D]` input
takes a rank 3 path, so a caller flattens to `[B*S, D]` first and calls
`torch.addmm`, which gives one large GEMM in place of a batched call.

MLX does not need it. MLX collapses the leading dimensions of a rank 3 by
rank 2 matmul into one GEMM already, so the flatten removes nothing.

Paired measurement, both forms alternating inside one process, 200 rounds
each, on an idle machine:

    .venv/bin/python3 <scratchpad>/rank_paired.py

| Projection | rank 3 `mx.addmm` | rank 2 `mx.addmm` | gain |
|---|---:|---:|---:|
| B64 S128 D128->384 (shape 1) | 0.457 ms | 0.461 ms | 0.992x |
| B1024 S128 D128->384 (shape 6 chunk) | 3.728 ms | 3.727 ms | 1.000x |
| B64 S1024 D128->384 (shape 13) | 1.992 ms | 1.994 ms | 0.999x |
| B64 S128 D1024->3072 (shape 8) | 12.717 ms | 12.717 ms | 1.000x |

An unpaired run of the same test, taken while a `scoreboard.py` sweep held
the GPU, gave 1.42x on one row and 0.79x on another. That reading was noise.
It is the reason measurement rule 1 exists.

The rank 2 form also costs code. Attention needs `[B, S, H, head_dim]`, so a
rank 2 block must reshape into attention and out of it, and `valid_tokens`
changes shape as well. It buys nothing. Do not try it again.

Row 29 keeps the `mx.addmm`, at rank 3.

## 31. Single-pass LayerNorm kernel for a narrow row — KEPT

Found by `profiling/stage_roofline.py`. The two LayerNorm calls of one block
do **zero matmul FLOPs** and take **26% of the shape 6 layer time**.

    .venv/bin/python3 profiling/stage_roofline.py --shapes 6

| Stage of one shape 6 layer, one chunk of 1024 | ms | share |
|---|---:|---:|
| ln1 | 3.710 | 12.9% |
| ln2 | 3.610 | 12.6% |
| everything else | 21.36 | 74.5% |
| real layer time | 28.68 | |

### The cause is the layer_norm kernel alone

`mx.fast.layer_norm` loses throughput as the row gets narrower. Measured at a
constant 64 MiB, so the row count rises as `D` falls:

| row width `D` | 32 | 64 | 96 | **128** | 192 | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| copy `x * 2` | 108 | 109 | 108 | 109 | 108 | 106 | 108 | 110 | 110 |
| `fast.rms_norm` | 74 | 106 | 107 | **107** | 105 | 107 | 107 | 109 | 108 |
| `fast.layer_norm` | **5.5** | **12** | **21** | **33** | 87 | 105 | 107 | 107 | 106 |
| layer_norm / copy | 19.8x | 8.8x | 5.0x | **3.3x** | 1.2x | 1.0x | 1.0x | 1.0x | 1.0x |

At `D >= 256` layer_norm reaches copy speed. Below 256 it degrades as about
`1/D`. That is the signature of a kernel that gives one thread or one small
group to each row, and does not vectorize along the row.

The two pieces layer_norm is built from both hold full speed at every width:

| operation at `D` = | 32 | 128 | 256 | 1024 |
|---|---:|---:|---:|---:|
| `mx.mean(x, axis=-1)`, GB/s of the read | 93 | 97 | 93 | 98 |
| `mx.fast.rms_norm`, GB/s | 74 | 107 | 109 | 109 |

So the memory system is not the limit. The reduction width is not the limit.
The weight and the bias are not the limit either: layer_norm with no weight
and no bias still gives 36 GB/s at `D = 128`, against 33 with both.

Twelve of the fourteen appendix shapes use `d_model` 128 or 32, so they all
take the slow path.

### What was tried

`layer_norm(x) == rms_norm(x - mean(x)) + bias`. The rewrite is accurate
(`max_abs_diff` 9.5e-07 to 1.4e-06, inside the 0.002 threshold), but it wins
only where `B * S` is small:

| B, S, D | `fast.layer_norm` | `rms_norm(x - mean)` | gain |
|---|---:|---:|---:|
| 64, 128, 128 (shape 1) | 0.382 ms | 0.259 ms | **1.47x** |
| 64, 128, 32 (shape 7) | 0.528 ms | 0.207 ms | **2.55x** |
| 1024, 128, 128 (shape 6 chunk) | 3.733 ms | 3.720 ms | 1.00x |
| 64, 1024, 128 (shape 13) | 1.922 ms | 1.943 ms | 0.99x |

Shape 6 and shape 13 carry 76% of the FLOP weight and neither moves, so the
rewrite as written is not worth taking. The mean pass costs at a large
`B * S` exactly what the layer_norm kernel's extra pass costs.

### Why the rewrite ties at the shapes that matter

The rewrite needs 3 to 4 passes over DRAM: one to read for the mean, one to
subtract it, one for `rms_norm`, one to add the bias. `mx.compile` merges
some of them, but not all. `mx.fast.layer_norm` needs about 3.3 passes in one
kernel. The two costs match, so the times match.

At a small `B * S` the rewrite still wins, because each pass is then short
enough that the launch cost decides the result, and layer_norm pays its own
launch as well.

### The prize

`mx.fast.rms_norm` proves the hardware runs this shape of work at 107 GB/s.
A single-pass LayerNorm holds `sum(x)` and `sum(x*x)` in one accumulator
pair, so it needs one read and one write, exactly like rms_norm.

| Shape | FLOP share | LayerNorm share of the layer | time saved at copy speed |
|---:|---:|---:|---:|
| 6 | 66.5% | 25.5% (7.32 of 28.68 ms) | 5.09 ms, 17.7% |
| 8 | 21.3% | 4.0%, `d_model = 1024` | none. Already at copy speed |
| 13 | 9.4% | 20.8% (3.67 of 17.64 ms) | 2.52 ms, 14.3% |

That is about **1.15x to 1.20x** on the FLOP-weighted sweep. It is larger
than row 28 (1.048x) and row 29 (1.096x).

### The kernel

`fast_layernorm.py`. One SIMD group of 32 lanes takes one row. Lane `l` reads
elements `l`, `l + 32`, `l + 64`, so the 32 lanes of one step read 32 adjacent
floats. That read is coalesced. The lane keeps its values in registers, and
the two reductions then run over the registers, not over DRAM:

1. Sum, then `simd_sum`, gives the mean.
2. Sum of `(v - mean)^2`, then `simd_sum`, gives the variance.

So the kernel makes ONE pass over memory. The centred form is deliberate. The
one-pass `E[x^2] - mean^2` form is cheaper, but it cancels when the mean is
large against the standard deviation. The centred form matches the torch
baseline, and the extra reduction is free because the kernel is memory bound.

The kernel holds `ceil(D / 32)` floats per lane, so it serves `D < 256`. At
256 and above `mx.fast.layer_norm` already reaches copy speed, and
`plan_kernels()` leaves that work with MLX.

### The measurement

    .venv/bin/python3 scoreboard.py --cpu-cache --label "row 31: single-pass LayerNorm kernel at d_model < 256"

Compare the MLX column against the row 29 sweep. Do NOT compare the speedup
against the CPU: the CPU baseline drifted up to 45.9% between the two sweeps,
which inflates every CPU-relative number.

| # | Shape | MLX before | MLX now | gain |
|---:|---|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 6.889 | 5.502 | 1.25x |
| 2 | B1 D128 H4 S128 | 0.748 | 0.650 | 1.15x |
| 3 | B4 D128 H4 S128 | 0.956 | 0.877 | 1.09x |
| 4 | B16 D128 H4 S128 | 2.091 | 1.764 | 1.19x |
| 5 | B128 D128 H4 S128 | 13.571 | 11.264 | 1.20x |
| 6 | B10000 D128 H4 S128 | 1050.066 | 852.760 | **1.23x** |
| 7 | B64 D32 H4 S128 | 4.761 | 1.395 | **3.41x** |
| 8 | B64 D1024 H4 S128 | 128.463 | 128.377 | 1.00x |
| 9 | B64 D128 H1 S128 | 6.268 | 4.438 | 1.41x |
| 10 | B64 D128 H2 S128 | 6.151 | 4.346 | 1.42x |
| 11 | B64 D128 H16 S128 | 7.061 | 5.342 | 1.32x |
| 12 | B64 D128 H4 S32 | 2.007 | 1.481 | 1.36x |
| 13 | B64 D128 H4 S1024 | 69.288 | 59.429 | 1.17x |
| | **total** | **1298.32** | **1077.62** | **1.205x** |

Shape 7 gives 3.41x because `d_model` is 32, where `mx.fast.layer_norm` runs
at 5.5 GB/s against 108 for a copy. That shape carries almost no FLOP weight,
so it does not drive the total. Shape 6 does.

### The controls

Two columns must NOT improve, because this change touches neither. Both held:

| # | CPU before -> now | MPS before -> now |
|---:|---|---|
| 1 | 41.975 -> 61.252 (**+45.9%**) | 17.313 -> 17.943 (+3.6%) |
| 6 | 13698.8 -> 14640.0 (+6.9%) | 2680.8 -> 2693.6 (+0.5%) |
| 8 | 471.9 -> 470.9 (-0.2%) | 165.5 -> 166.6 (+0.7%) |
| 13 | 1858.0 -> 1899.3 (+2.2%) | 564.8 -> 577.6 (+2.3%) |

MPS moved between +0.5% and +3.6%, all in the SLOWER direction. So the GPU
did not get faster between the sweeps, and the 1.205x is the kernel.

Shape 8 gives MLX 1.00x. That is the gate working: `d_model` is 1024, so
`plan_kernels()` sets `fast_ln=False` and MLX keeps the work.

### The stage-level proof

    .venv/bin/python3 profiling/stage_roofline.py --shapes 6

The LayerNorm stage now runs at the copy roof. Nothing else moved:

| Shape 6 stage, one layer, one chunk | before | after | change |
|---|---:|---:|---|
| ln1 | 3.71 ms, 36.2 GB/s | **1.12 ms, 119.8 GB/s** | **3.31x** |
| ln2 | 3.61 ms, 37.2 GB/s | **1.12 ms, 119.9 GB/s** | **3.22x** |
| qkv proj | 3.64 ms | 3.55 ms | — |
| sdpa | 5.64 ms | 5.57 ms | — |
| out proj | 1.34 ms | 1.32 ms | — |
| **real layer time** | **28.68 ms** | **20.71 ms** | **1.385x** |

The roof for a data movement stage is 128 GB/s. The kernel reaches 119.8, so
it is at 94% of the roof and there is almost nothing left in it. LayerNorm
fell from 25.5% of the layer to 10.8%.

The stage sum and the real layer time now agree to 0.5% (20.81 against
20.71 ms), so the stage model holds.

Note: `stage_roofline.py` profiles the LayerNorm that `plan_kernels()`
selects. An earlier version called `mx.fast.layer_norm` directly and reported
the old rate after this change landed. That was a bug in the tool, and it is
fixed.

### Accuracy

All 13 shapes PASS at `atol=0.002`, `rtol=0.02`. `max_abs` runs 9.54e-07 to
2.65e-06, which is the same band as the previous sweep.

`test_padding.py` passes all 18 cases, bit exact. That set includes a fully
empty sample. A row of zeros gives mean 0 and variance 0, so `rstd` becomes
`1/sqrt(eps)`, which is finite. The kernel returns the bias there, exactly as
`mx.fast.layer_norm` does.

The harness at its own config (`d_model = 512`) gives PASS `max_abs=2.98e-06`
and 4.974x. That config does not reach the kernel, so it only shows that
nothing broke:

    .venv/bin/python3 torch_transformer_benchmark.py

## 32. Give the attention kernel contiguous q, k and v — OPEN

Found by `profiling/stage_roofline.py`. An MLX transpose is a **free strided
view**, not a copy:

| B1024 S128 H4 hd32 | ms |
|---|---:|
| `mx.split` alone | 0.0425 |
| `mx.split` + `transpose(0, 2, 1, 3)` | 0.0415 |
| the same, then `mx.contiguous` | 3.5495 |

So the head layout costs nothing as a stage. An earlier version of the stage
profiler reported 3.25 ms for it, and that number was **wrong**: the profiler
added an `mx.concatenate` to force the copy, and measured its own artifact.

The cost is real, but it lands inside the attention kernel, which reads q, k
and v with a stride instead of in order:

| Shape | sdpa on the strided view | sdpa on contiguous copies | ratio | cost of the copy |
|---|---:|---:|---:|---:|
| 6 (B1024 chunk, S128, hd32) | 5.539 ms | **2.131 ms** | **2.60x** | 3.291 ms |
| 13 (B64, S1024, hd32) | 6.625 ms | 5.148 ms | 1.29x | 1.644 ms |
| 11 (B64, S128, H16, hd8) | 0.536 ms | 0.318 ms | 1.68x | 0.365 ms |
| 8 (B64, S128, hd256, fallback) | 2.622 ms | 2.712 ms | 0.97x | 1.839 ms |

Shape 6 loses **3.41 ms of 5.54 ms**, 61% of its attention time, to
non-coalesced reads alone.

### Why a `mx.contiguous` first does not pay

At shape 6 the copy costs 3.291 ms and saves 3.408 ms. The total goes 5.539
to 5.422 ms, which is 1.02x and inside the noise floor. The copy itself runs
at 98 GB/s, near the roof, so there is nothing to tune in it.

**The row stays OPEN.** Two paths could collect the 3.41 ms:

1. Make the QKV projection write `[B, H, S, head_dim]` directly. The matmul
   output is indexed `(b, s)` by `(h, d)`, so `s` must move inside `h`. That
   is a transpose of the matmul output, and it is not free.
2. Change the read pattern of `steel_attention.py` to suit the stride.

Shape 8 is the control: it takes the FALLBACK kernel, and the ratio there is
0.97x. The strided penalty belongs to the steel kernel, not to every kernel.


## 33. Fold GELU into the FFN matmul epilogue — OPEN

Found by `profiling/stage_roofline.py`. The `ffn_in + gelu` stage reaches 45%
of the matmul peak. `ffn_out` runs the same size matmul and reaches 80%. The
difference is the GELU.

Measured at the shape 6 chunk FFN size, 131072 rows by 128, a 64 MiB
activation:

| step | ms |
|---|---:|
| `mx.addmm` alone | 1.417 |
| `mx.addmm` then GELU | 2.385 |
| GELU alone, on an evaluated input | 1.166 |
| `mx.compile` of `gelu(addmm(...))` | 2.378 |

GELU adds 0.968 ms. A separate read plus write of 64 MiB at the measured
128 GB/s roof costs 1.049 ms. The two agree, so GELU is exactly one extra
pass over DRAM.

The GELU kernel is not slow. Alone it reaches 115 GB/s, which is 90% of the
copy roof. The cost is the pass itself, not the arithmetic.

**`mx.compile` does not fuse it.** 2.378 ms compiled against 2.385 ms plain
is inside the noise. MLX fuses elementwise chains, but it does not fuse an
elementwise operation into a GEMM epilogue.

At shape 6 this is 0.968 ms for each layer and each chunk, so 4 layers by 10
chunks is 38.7 ms of the 852.8 ms runtime, or **4.5%**.

**The row stays OPEN, and it may not be reachable.** MLX exposes no matmul
epilogue through its Python API. `steel_attention.py` shows one way around a
missing dispatch, by hoisting the Metal source of a kernel MLX already ships.
The same trick would need a GEMM template with an epilogue hook. Nobody
checked whether the MLX steel GEMM headers offer one.

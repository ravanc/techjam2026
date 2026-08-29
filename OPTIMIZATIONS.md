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
| 1 | MLX behind the torch interface | **KEPT** | `_mlx_transformer()` L537, `UserOptimizedTransformer` L668 | every shape | 4.4x to 7.4x against torch CPU. float32 PASS |
| 2 | `mx.compile` on the forward pass | **KEPT** | `make_call()` L793 | every shape | 14.203 ms to 13.401 ms (6%) |
| 3 | float32 LayerNorm accumulation | **KEPT** | `norm()` L572, weight build L731 | every shape | no measured gain. Correct policy. Free |
| 4 | Explicit float32 softmax, not fused | **REVERTED** | — | — | float16 49 to 50 failures. No gain. More code |
| 5 | `torch.compile` over the MLX class | **RULED OUT** | — | — | crash. Dynamo cannot trace an MLX object |
| 6 | Mask form selected by the input | **KEPT** | `_mlx_transformer()` L586 | seq_len >= 512, no padding | 3.9% at seq 512. 1.7% at seq 2048. none at seq 128 |
| 7 | Shape-aware kernel plan (`KernelPlan`) | **KEPT** | `plan_kernels()` L354 | every shape. It selects 8, 9 and 10 | 1.57x at shape 13. 1.31x at shape 6 |
| 8 | Blocked causal attention | **KEPT** | `_attention()` L477 | causal, effective `head_dim < 64`, and not on the steel path. **No appendix shape selects it now** | 1.63x at seq 1024, bit exact, when it was the best option. It works around the FALLBACK kernel. Row 25 removes the fallback instead, so every shape that used to block now takes the steel kernel. The code stays for a shape row 25 cannot take |
| 9 | Fused `[D, 3D]` QKV matmul | **KEPT** | weight build L786, use L621 | every shape. Always on | 1.99x at batch 1. 1.00x at d_model 1024 |
| 10 | Batch chunking, full depth for each chunk | **KEPT** | `forward()` L826 | one activation over 64 MiB. Shape 6 only | peak 9.16 GiB to 2.68 GiB, and 1.05x faster |
| 11 | Pad `head_dim` from 8 up to 32 | **REVERTED** | — | — | 0.91x at shape 7. Blocking beats it. Wrong target: 32 is still the fallback kernel. See row 17 |
| 12 | Full port of the file to MLX | **RULED OUT** | — | — | out of scope. The baseline does not change |
| 13 | float16 and bfloat16 accuracy | **RULED OUT** | — | — | no implementation can pass. torch `F.sdpa` fails too |
| 14 | Hand-written Metal kernel for `head_dim = 8` | **KEPT**, as row 25 | `steel_attention.py` | shapes 7 and 11 | done without writing new arithmetic: row 25 compiles Apple's own kernel at `head_dim = 8`. Shape 11 **2.19x**, shape 7 1.48x |
| 17 | Pad `head_dim` to 64 to reach the fused SDPA kernel | **KEPT** | `plan_kernels()` L444 | **nothing now.** Row 25 wins wherever the pad applied, and `plan_kernels()` prefers it | 1.646x at shape 13 when it was the best option. Row 25 gives shape 13 **2.14x over the padded path** at the attention step, with no widened projection. The pad stays in the code for a shape the steel kernel cannot take |
| 18 | MLX dispatches SDPA to two different kernels | **KEPT** (a fact, not a change) | `SDPA_FUSED_MIN_HEAD_DIM` L331 | `head_dim` in {64, 72, 80, 96, 128} gets the fused kernel. Everything else materializes `B x H x S x S` | peak memory at B8 H8 S1024: 16 MiB fused against 264 MiB fallback. **The set is NOT the range 64..128.** See row 20 |
| 19 | Sweep `seq_len` to place the `S >= 4*D` threshold | **OPEN** | — | `head_dim` 32..48 | not measured at the model level. Row 20 measures the crossover at the attention kernel alone |
| 20 | The fused SDPA set is discrete, not a range | **KEPT** (a fact, not a change) | `profiling/sdpa_dispatch.py` | `head_dim` 1..288, every mask kind, every dtype, `S` 512 and 1024, `B*H` 1 to 256 | the set is {64, 72, 80, 96, 128}. 65, 100 and 127 fall back. Always pad to the SMALLEST member at or above `head_dim`: a larger target lost all 44 rows |
| 21 | MLX never calls its own `bd192` and `bd256` kernels | **OPEN** (an MLX gap) | — | shape 8, `head_dim=256`, 21.3% of the FLOP-weighted score | the metallib holds `steel_attention_*_bd192_*` and `*_bd256_*` for all 3 dtypes. All 32 dispatch combinations tried take the fallback. No Python workaround: `head_dim` cannot pad down and a head cannot split. Recheck after an MLX upgrade |
| 22 | Block a causal wide head that misses the fused set | **OPEN** | — | `head_dim > 128` and long `S`. No appendix shape reaches it | `head_dim=256`, B64 H4: S=128 full wins, S=512 blk128 gives 1.32x, S=1024 blk128 gives **1.55x**. `plan_kernels()` refuses to block because it tests `effective_head_dim < 64` |
| 23 | Return the output as a view of MLX memory, not a copy | **KEPT** | `_to_torch()` L188 | every shape. float32 and float16 alias. bfloat16 and a device change still copy | **71.6 ms of 1590.2 ms at shape 6 (1.047x)**. 1.5 ms of 136 ms at shape 8. Bit exact by `torch.equal` |
| 15 | Fused FFN | **KEPT**, as row 33 | `steel_gemm.py` | float32, `ffn_in`, rows >= 512 | `mx.compile` does NOT fuse the FFN: 2.237 ms compiled against 2.242 ms plain at the shape 6 chunk. Row 33 fuses it in the GEMM epilogue instead, for **1.064x FLOP-weighted** |
| 16 | Quantization by `mx.quantize` | **OPEN** | — | every shape | not measured. It will fail the accuracy test |
| 25 | Hoist MLX's `steel_attention` and compile it at an unshipped `head_dim` | **KEPT** | `steel_attention.py`, routed at `_attention()` L497, gated at `plan_kernels()` L430 | causal, no padded batch, `head_dim % 8 == 0`, `head_dim` not already fused, threadgroup fits. Shapes 1-7, 11, 12, 13 | **1.308x FLOP-weighted** (MLX 1218.8 ms to 932.2 ms). Shape 6 1.32x, shape 13 1.47x, shape 11 2.19x. MLX against MPS 1.60x to 2.12x FLOP-weighted. All accuracy PASS, max_abs 1.31e-06 to 1.91e-06 |
| 26 | Reach the steel kernel at `head_dim = 256` for shape 8 | **REVERTED** | — | — | measured. `bk8` does fit, and it LOSES. Three block shapes fit 32 KiB at `bd256`, and the best of them runs **0.904x** against the MLX fallback shape 8 uses today. See row 41 |
| 27 | Gate the steel kernel on a string mask | **KEPT** (a bug fix) | `_mlx_transformer()` L604 | every padded batch on a shape that selects the steel kernel | a padded causal batch went **FAIL** (822894/1048576 elements wrong) to **PASS** (`max_abs=3.04e-06`). No sweep saw the bug: `--padding-ratio` defaults to 0.0 everywhere. `test_padding.py` now covers it |
| 28 | Drop the token masks that cannot change the output | **KEPT** | `_mlx_transformer()` L599, L659, L665 | every shape. The unpadded graph now holds no mask operation | bit exact on 18 cases (`test_padding.py`). **1.048x FLOP-weighted** (MLX 841.6 ms to 803.2 ms). Shape 6 1.050x. MPS held at 1.010x and CPU at 1.005x on the same sweep, so the noise floor is about 1% |
| 29 | `mx.addmm` for every projection, in place of `h @ w + b` | **KEPT** | `_mlx_transformer()` L621-L658 | every shape. Always on | **1.096x FLOP-weighted** (MLX 803.2 ms to 732.6 ms). Shape 6 1.099x, shape 8 1.040x, shape 13 1.092x. Bit exact in float32 on 16 cases |
| 30 | Flatten the block to rank 2 before each projection | **REVERTED** | — | — | 0.992x to 1.000x against rank 3, on 4 projection sizes. MLX already collapses a rank 3 by rank 2 matmul into one GEMM |
| 31 | Single-pass LayerNorm kernel for a narrow row | **KEPT** | `fast_layernorm.py`, chosen at `_mlx_transformer()` L570, gated at `plan_kernels()` L465 | `d_model < 256`, float32. Shapes 1-7 and 9-13. Shape 8 keeps MLX | **1.205x FLOP-weighted** (MLX 1298.3 ms to 1077.6 ms over the 13 shapes). Shape 7 **3.41x**, shape 10 1.42x, shape 9 1.41x, shape 6 **1.23x**, shape 13 1.17x. Shape 8 1.00x, so the gate is correct. All 13 shapes PASS, `max_abs` 9.5e-07 to 2.65e-06. 18/18 padding cases bit exact |
| 32 | Give the attention kernel contiguous q, k and v | **REVERTED** | — | — | 1.02x, inside the noise floor: the `mx.contiguous` costs 3.29 ms and saves 3.41 ms. **Row 34 supersedes it and inverts it.** The 3.41 ms was never the read pattern. It was a hidden copy that `ensure_row_contiguous=True` ran inside the kernel launch. Do not make q, k and v contiguous. Tell the kernel their strides. This row also mislabelled its own measurement: see the detail section |
| 33 | Fold GELU into the FFN matmul epilogue | **KEPT** | `steel_gemm.py` `steel_addmm()` L376, tile at `choose_tile()` L288; gated at `plan_kernels()` L508; used at `_mlx_transformer()` L734 | float32, `ffn_in` only, rows >= 512, and a tile that divides M, N and K. Shapes 1 and 3-13. Shape 2 keeps the MLX pair | **1.064x FLOP-weighted** (MLX 768.6 ms to 722.3 ms over the 13 shapes). Shape 6 **1.082x**, shape 3 1.068x, shape 5 1.042x, shape 7 1.042x, shape 13 1.030x, shape 8 1.009x. At the `ffn_in` stage alone: shape 7 **4.49x**, shape 6 1.51x, shape 13 1.47x. **Row 40 attributes that 1.51x: it is the epilogue, not the tile.** `steel_addmm(gelu=False)` ties `mx.addmm` on the same GEMM. Control: MPS held at 1.005x across the two sweeps. All 13 shapes PASS, `max_abs` 1.07e-06 to 2.62e-06. 18/18 padding cases pass |
| 34 | Read q, k and v as strided views, and write the head layout directly | **KEPT** | `steel_attention.py` `steel_attention()`, called at `_attention()` L511, merge at `_mlx_transformer()` L631 | every shape on the steel path: 1-7, 11, 12, 13 | **1.239x FLOP-weighted** (MLX 1077.6 ms to 869.7 ms). Shape 5 1.336x, shape 6 **1.290x**, shape 1 1.301x, shape 11 1.225x, shape 13 1.182x. Two controls: MPS held at **1.000x** across the two sweeps, and the three non-steel shapes 8, 9 and 10 moved 1.006x, 1.002x and 1.013x. All 13 shapes PASS, `max_abs` 9.54e-07 to 2.65e-06. 18/18 padding cases bit exact |
| 35 | Fuse the residual add into the LayerNorm kernel | **RULED OUT** | — | — | **Superseded by row 36, never built.** Row 36 reaches the same prize from the other side: it gives the residual add to the GEMM C operand and defers the bias into the LayerNorm. So the residual add is no longer a kernel, and there is nothing left for this row to fuse. The measurement below still stands and is why row 36 exists |
| 36 | Defer the residual biases, and give the residual add to the GEMM C operand | **KEPT** | `fast_layernorm.py` `layer_norm(pre_bias=)` L154; block at `_mlx_transformer()` L653, L696 and L705; gated at `plan_kernels()` L480 | float32, unpadded, `d_model < 256`. Shapes 1-7 and 9-13. Shape 8 keeps the plain path | **1.132x FLOP-weighted** (MLX 869.7 ms to 768.6 ms over the 13 shapes). Shape 6 **1.164x**, shape 11 1.133x, shape 9 1.128x, shape 1 1.117x, shape 10 1.114x, shape 13 1.112x. Two controls: **shape 8 holds at 1.000x** with `defer_bias=False`, and MPS held at 1.013x across the two sweeps. All 13 shapes PASS, `max_abs` 1.19e-06 to 2.65e-06. 18/18 padding cases bit exact |
| 37 | A wide `fast_layernorm` variant, so shape 8 reaches row 36 | **OPEN** | — | shape 8 only (`d_model` 1024), 21.3% of the FLOP weight | not tried. Shape 8 is the one shape row 36 cannot serve, because `mx.fast.layer_norm` has no `pre_bias` argument. The C operand is nearly free there: **0.043 ms** against 0.708 ms for the separate add, because the shape is compute bound and the extra read hides under the matmul. Two residual adds is about **1.53 ms of the 32.51 ms layer, or 4.7%**, so about **1.0% FLOP-weighted**. Measured directly after row 36: the two residual adds are 0.785 ms and 0.834 ms, and the C operand would cost 0.043 ms each. The cost is a `fast_layernorm` that holds `ceil(1024/32) = 32` floats per lane, against 8 at `d_model` 248 today. Register pressure is the open question, and row 31 measured that MLX already runs at copy speed at that width, so the LayerNorm itself has nothing to win. The `pre_bias` hook is the only reason to build it |
| 38 | Retune the steel attention block shape (`bq`, `bk`, `wm`) against the MFA parameter table | **REVERTED** | — | — | the default `bq=32, bk=32, wm=4` is best on every steel shape. The best other config anywhere is **1.002x** (shape 13), inside the 1% noise floor. MFA's own `bq=16` gives 1.000x, 0.956x, 0.929x, 0.824x and 0.895x on the five cases |
| 39 | Shift the edge block in the steel GEMM, so an unaligned tile stays branch free | **RULED OUT** | — | — | nothing to unlock. `choose_tile()` never returns `None` on any appendix shape: every M, N and K is a power of two, and at least two tiles divide all 13. Shape 2 loses the fused GELU to the `rows >= 512` gate, not to divisibility |
| 40 | Route the projections with no epilogue through the hoisted steel GEMM | **REVERTED** | — | — | `mx.addmm` is already optimal. Over 8 tiles on 5 real projection sizes, the best steel tile reaches **1.005x**, and `max_abs` is **0.00e+00** everywhere, which proves MLX dispatches `addmm` to the same steel kernel with a good tile. Shape 6 `qkv proj` sits at 83.5% of matmul peak because K=128 is short, not because of the tile |
| 41 | Reach the steel attention kernel at `head_dim = 256` with a narrow block | **REVERTED** | — | — | supersedes row 26. Three block shapes fit 32 KiB at `bd256`. Best is `bq16 bk8 wm2` at **0.904x** against the MLX fallback; `bq8 bk8` gives 0.553x and `bq8 bk16` gives 0.359x. Bit accurate (`max_abs` 1.19e-06), just slower |
| 42 | Combine the batch chunks with something faster than `mx.concatenate` | **REVERTED** | — | — | `mx.concatenate` already runs at **119.9 GB/s** on the shape 6 output (625 MiB in 10.185 ms), against a 128 GB/s roof. Every alternative is about 7x worse, because a CPU-side copy of unified memory runs at 14-17 GB/s: `torch.cat` 72.3 ms, `torch.copy_` 77.0 ms, numpy slice assign 84.7 ms. All three are bit equal |

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

## 26. Reach the steel kernel at `head_dim = 256` for shape 8 — REVERTED

Shape 8 is `d_model=1024`, `num_heads=4`, so `head_dim=256`, and it carries
21.3% of the FLOP weight. It is the largest shape row 25 cannot take.

`BQ=32, BK=32, BD=256` needs 68.5 KiB of threadgroup memory against a 32 KiB
limit. `BK=16` needs about 41 KiB, still over.

This row guessed that `BK=8` would fit, and never measured whether it would be
faster. **Row 41 measured it. `BK=8` fits, and it loses: 0.904x against the
MLX fallback.** Read row 41 before you try this again.

**The ceiling is small.** Shape 8 is 7.1% attention by measured time, not the
4% the FLOP table implies, because `d_model=1024` makes the projections 92%
of the work. `stage_roofline.py --shapes 8` gives `sdpa` 2.2808 ms and
`merge heads` 0.4113 ms of a 32.06 ms layer. A perfect attention kernel would
give about 1.0% of the weighted score.

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

## 32. Give the attention kernel contiguous q, k and v — REVERTED

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

**Two corrections. Read them before you use the table below.**

1. The table says "sdpa". Rows 6, 13 and 11 did NOT call
   `mx.fast.scaled_dot_product_attention`. They called `steel_attention.py`,
   because `_attention()` routes those shapes to the steel kernel. Measured
   at the shape 6 dimensions, `mx.fast.scaled_dot_product_attention` takes
   **16.313 ms on the strided view and 15.807 ms on contiguous copies**. Its
   ratio is 1.03x, not 2.60x. Only shape 8, the control row, is really SDPA.
2. The diagnosis "non-coalesced reads" was **wrong**. The kernel never read
   the stride. `steel_attention.py` passed `ensure_row_contiguous=True`, so
   MLX copied q, k and v into fresh contiguous buffers before every launch.
   The 3.41 ms was that hidden copy. The "contiguous copies" column is fast
   because MLX skips the copy when the input is already contiguous, not
   because the kernel reads it better. Row 34 removes the copy and keeps the
   views, and it wins 1.290x at shape 6.

The cost is real, and it lands inside the attention kernel launch:

| Shape | steel on the strided view | steel on contiguous copies | ratio | cost of the copy |
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

**The row is REVERTED, and row 34 replaces it.** This row asked the wrong
question. It assumed the kernel must receive contiguous arrays, and it looked
for a cheaper way to produce them. Both paths it lists are dead ends:

1. Make the QKV projection write `[B, H, S, head_dim]` directly. The matmul
   output is indexed `(b, s)` by `(h, d)`, so `s` must move inside `h`. That
   is a transpose of the matmul output, and it is not free.
2. Change the read pattern of `steel_attention.py` to suit the stride.

The kernel already reads any stride. It needs no contiguous array at all. See
row 34.

Shape 8 is the control: it takes the FALLBACK kernel, and the ratio there is
0.97x. That control is still valid, and it now reads differently. Shape 8
shows no penalty because MLX's own SDPA never ran the extra copy. Only
`steel_attention.py` did, and only because this project asked it to.


## 33. Fold GELU into the FFN matmul epilogue — KEPT

`steel_gemm.py` hoists MLX's own steel GEMM out of its headers and compiles
it at a new epilogue. The method is row 25's, applied to `steel/gemm/` in
place of `steel/attn/`. `_mlx_transformer()` then runs ONE kernel where MLX
ran two.

### Why it was worth doing

Found by `profiling/stage_roofline.py`. Every other stage of the shape 6
block sits at 88% to 98% of a roof. `ffn_in + gelu` sat at 45.9%.

Measured at the shape 6 chunk FFN size, 131072 rows by 128, 100 repeats:

| step | ms |
|---|---:|
| `mx.addmm` alone | 1.266 |
| `mx.addmm` then GELU | 2.242 |
| GELU alone, on an evaluated input | 1.013 |
| `mx.compile` of `gelu(addmm(...))` | 2.237 |

GELU added 0.977 ms. A separate read plus write of 64 MiB at the measured
128 GB/s roof costs 1.049 ms. The two agree, so GELU was exactly one extra
pass over DRAM.

The GELU kernel was never slow. Alone it reached 132.5 GB/s, which is above
the 128 GB/s copy roof for this size. The cost was the pass itself.

**`mx.compile` does not fuse it.** 2.237 ms compiled against 2.242 ms plain
is inside the noise. MLX fuses elementwise chains, but it does not fuse an
elementwise operation into a GEMM epilogue. This also closes the
"check `mx.compile` first" note on row 15.

### The hook

`steel/gemm/mma.h` line 593 holds a UNARY epilogue:

    template <typename UnaryEpilogue>
    METAL_FUNC void apply_epilogue(
        thread const UnaryEpilogue& epilogue_op) thread {
      for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
        Ctile.elems()[i] = epilogue_op.apply(Ctile.elems()[i]);
      }
    }

`Ctile` is the accumulator, in registers, before the store. The kernel
already calls the BINARY form, `apply_epilogue(C, ldc, fdc, TransformAdd)`,
which is how `mx.addmm` adds the bias. `steel_gemm.py` adds a `TransformGelu`
and calls the unary form straight after, at all four exit paths. So the
order is: accumulate, add the bias, apply GELU, store. The extra pass is
gone.

### Five edits to Apple's kernel, and no others

1. `[[kernel]]` becomes a plain function, so it can be a callee.
2. The `[[function_constant]]` flags become `constexpr bool`.
3. `constant GEMMParams*` becomes `thread GEMMParams*`.
4. The `threadgroup` arrays move to the caller.
5. The `has_batch` branch goes. This module never batches, and Metal type
   checks a dead branch: that branch reads `batch_strides` through
   `const constant auto*`, and a hoisted callee gets `thread` pointers.

The arithmetic is Apple's. See the `steel_gemm.py` docstring.

### The GEMM is not larger than the attention kernel

An earlier version of this section warned that the GEMM would be a bigger
job than row 25, because `steel_gemm_fused.h`, `steel_gemm_splitk.h` and
`steel_gemm_masked.h` are separate kernels. **That was wrong on both counts.**

The header chain is the same size, because row 25 already inlines the four
shared `steel/` headers and `steel/gemm/params.h`:

| Hoist | kernel-specific headers | kernel file | total |
|---|---:|---:|---:|
| `steel/attn/` (row 25) | 1199 | 476 | 1675 |
| `steel/gemm/` (row 33) | 1432 | 346 | 1778 |

And only one of the three kernels matters. `splitk` serves a small M with a
large K, and `masked`, `gather` and `segmented` serve other operations.
`ffn_in` is a plain `[M, K] @ [K, N]` with a bias, so `steel_gemm_fused` is
the only file to hoist.

### The plain path is bit exact

With `gelu=False` the hoisted kernel reproduces `mx.addmm` exactly:
`mx.array_equal` is True and `max_abs` is 0.0. That was the gate. It proves
the hoist itself is faithful, before any new arithmetic goes in.

### GELU is 1 ULP off, and cannot be closer

`mlx_nn.gelu` is `x * (1 + mx.erf(x / sqrt(2))) / 2`. `steel_gemm.py` inlines
MLX's own `erf.h`, so both call the SAME approximation. The result still
differs by 1 ULP, and three expression orders all give the same 4.77e-07:

| form | max_abs | differing |
|---|---:|---:|
| `x * 0.5f * (1 + erf(x * 0.70710678f))` | 4.768e-07 | 140462 / 1048576 |
| `(x * (1 + erf(x / 1.41421356f))) / 2` | 4.768e-07 | 147929 / 1048576 |
| the same with `metal::precise::divide` | 4.768e-07 | 147929 / 1048576 |

So the difference is the JIT's math flags, not the expression. The first
form has the fewest differing elements, so it is the one in the code.
4.77e-07 is 4000 times inside the harness `atol=0.002`, and the model's
`max_abs` was already 2.65e-06 before this row.

### The tile

`choose_tile()` takes the first tile that divides M, N and K, so the kernel
always runs its aligned path. Measured at the shape 6 `ffn_in` size, against
2.300 ms for the MLX pair:

    bm32 bn64  1.527 ms      bm64 bn32  1.531 ms     bm32 bn32  1.593 ms
    bm64 bn64  1.648 ms      bm64 bn128 1.907 ms     bm128 bn64 1.947 ms

### The row threshold

Measured at the real `ffn_in` size of each shape, fused against the MLX pair:

| rows | M | K | N | MLX ms | fused ms | speedup |
|---|---:|---:|---:|---:|---:|---:|
| shape 2 | 128 | 128 | 128 | 0.0500 | 0.0615 | **0.81x** |
| shape 3 | 512 | 128 | 128 | 0.0855 | 0.0395 | 2.17x |
| shapes 4, 12 | 2048 | 128 | 128 | 0.0749 | 0.0477 | 1.57x |
| shapes 1, 9, 10, 11 | 8192 | 128 | 128 | 0.1785 | 0.1403 | 1.27x |
| shape 5 | 16384 | 128 | 128 | 0.2952 | 0.2244 | 1.32x |
| shape 13 | 65536 | 128 | 128 | 1.1943 | 0.8142 | 1.47x |
| shape 6 chunk | 131072 | 128 | 128 | 2.2957 | 1.5259 | 1.51x |
| shape 7 | 8192 | 32 | 32 | 0.1503 | 0.0335 | **4.49x** |
| shape 8 | 8192 | 1024 | 1024 | 4.7809 | 4.4603 | 1.07x |

Only M=128 loses, so `MIN_FUSED_GELU_ROWS` is 512. Shape 2 is the only
shape below it, and it is kernel launch bound end to end.

Shape 7 gains 4.49x because MLX's GEMM is poor at K=N=32, not because the
epilogue saves more there.

### The model level result

    .venv/bin/python3 scoreboard.py --cpu-cache --label "row 33: fold GELU into the ffn_in GEMM epilogue"

**1.064x FLOP-weighted**, MLX 768.6 ms to 722.3 ms over the 13 shapes.

| case | FLOP weight | MLX before | MLX after | speedup | MPS control |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4% | 3.788 | 3.656 | 1.036x | 1.002x |
| 2 | 0.0% | 0.652 | 0.634 | 1.029x | 1.106x |
| 3 | 0.0% | 0.757 | 0.709 | 1.068x | 1.021x |
| 4 | 0.1% | 1.375 | 1.337 | 1.029x | 1.007x |
| 5 | 0.9% | 7.359 | 7.065 | 1.042x | 1.004x |
| **6** | **66.5%** | 567.842 | 524.885 | **1.082x** | 1.009x |
| 7 | 0.0% | 1.124 | 1.079 | 1.042x | 0.984x |
| 8 | 21.3% | 127.557 | 126.408 | 1.009x | 0.996x |
| 9 | 0.4% | 3.923 | 3.861 | 1.016x | 0.989x |
| 10 | 0.4% | 3.850 | 3.742 | 1.029x | 0.995x |
| 11 | 0.4% | 3.850 | 3.754 | 1.026x | 0.995x |
| 12 | 0.1% | 1.309 | 1.285 | 1.018x | 0.975x |
| 13 | 9.4% | 45.231 | 43.893 | 1.030x | 0.992x |

MPS summed 3533.9 ms against 3515.3 ms across the two sweeps, which is
1.005x. So the machine did not change and the gain is the kernel.

Shape 2 moved 1.029x with `fuse_gelu=None`, so that is the noise floor of
this pair of sweeps, not an effect.

All 13 shapes PASS, `max_abs` 1.07e-06 to 2.62e-06. `test_padding.py` passes
18 of 18; its third shape (B4 S128, ffn 512) gives 512 rows and does select
the fused tile.

## 34. Read q, k and v as strided views, and write the head layout directly — KEPT

The model builds q, k and v as free strided views of one fused QKV buffer.
`mx.split`, `reshape` and `transpose` all return the SAME base pointer, and
they cost 0.035 ms at the shape 6 chunk. Row 32 already recorded that.

`steel_attention.py` then threw the views away. It passed
`ensure_row_contiguous=True` to `mx.fast.metal_kernel`, so MLX copied all
three into fresh contiguous buffers before every launch. Measured at the
shape 6 chunk, B1024 S128 D128 H4:

| step | ms |
|---|---:|
| the kernel on the strided views, as the model called it | 5.364 |
| the kernel on ready-made contiguous arrays | 2.223 |
| `mx.contiguous` of q, k and v alone | 3.328 |

The copy was 59% of the attention call, and it bought nothing.

### The kernel never needed it

`steel/attn/kernels/steel_attention.h` advances its pointers through
`params->Q_strides`, and it hands `params->Q_strides[2]` to the block loader:

    Q += tidl.z * params->Q_strides[0] +   // Batch
         tidl.y * params->Q_strides[1] +   // Head
         tidl.x * BQ * params->Q_strides[2];  // Sequence

`params.h` names the contract on line 33: `Query strides (B, H, L, D = 1)`.
The kernel requires the LAST axis to be contiguous, and nothing else. A view
of the fused buffer satisfies that: D stays contiguous, and only the batch,
the head and the sequence strides change.

`mx.fast.metal_kernel` passes the true strides of every array it binds, in a
`q_strides` buffer. I checked it with a probe kernel. For a `[B, S, 3D]`
buffer split three ways and viewed as `[B, H, S, D]` it reports
`[49152, 32, 384, 1]`, which is exactly `(S*3D, head_dim, 3D, 1)`. MLX also
applies the data offset of the view, so k and v read from the right place.

So `_source()` now copies `q_strides`, `k_strides` and `v_strides` into
AttnParams, and `ensure_row_contiguous` is False. The strides are the only
values in the kernel that are not literals.

### The output copy went the same way

`context.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)` merged the
heads. `transpose` makes the array strided, so `reshape` must materialize.
That copy cost 1.197 ms for each layer at the shape 6 chunk, against 1.185 ms
for a plain copy of the same bytes. It ran at copy speed, so there was
nothing to tune in it.

The kernel writes O through `params->O_strides` as well. `head_last=True`
allocates the output as `[B, S, H, D]` and bakes the strides that place
element `(b, h, s, d)` there. The merge is then `reshape(B, S, D)` on a
contiguous array, which is a free view.

Together the two changes remove 4.34 ms from each shape 6 layer.

### The measurement

    .venv/bin/python3 scoreboard.py --cpu-cache \
        --label "steel reads strided q,k,v and writes head-last: no copy"

| # | Shape | steel? | MLX before | MLX after | gain |
|---:|---|:---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | yes | 5.502 | 4.230 | **1.301x** |
| 2 | B1 D128 H4 S128 | yes | 0.650 | 0.631 | 1.030x |
| 3 | B4 D128 H4 S128 | yes | 0.877 | 0.772 | 1.136x |
| 4 | B16 D128 H4 S128 | yes | 1.764 | 1.396 | 1.263x |
| 5 | B128 D128 H4 S128 | yes | 11.264 | 8.428 | **1.336x** |
| 6 | B10000 D128 H4 S128 | yes | 852.760 | 660.808 | **1.290x** |
| 7 | B64 D32 H4 S128 | yes | 1.395 | 1.145 | 1.219x |
| 8 | B64 D1024 H4 S128 | no | 128.377 | 127.573 | 1.006x |
| 9 | B64 D128 H1 S128 | no | 4.438 | 4.427 | 1.002x |
| 10 | B64 D128 H2 S128 | no | 4.346 | 4.290 | 1.013x |
| 11 | B64 D128 H16 S128 | yes | 5.342 | 4.362 | 1.225x |
| 12 | B64 D128 H4 S32 | yes | 1.481 | 1.382 | 1.072x |
| 13 | B64 D128 H4 S1024 | yes | 59.429 | 50.300 | 1.182x |

Total MLX time 1077.6 ms to 869.7 ms, so **1.239x FLOP-weighted**.

### The controls

1. **MPS held at 1.000x.** Total MPS time was 3578.0 ms before and 3578.2 ms
   after. The machine did not drift between the two sweeps, so the MLX gain
   is the change and not the weather.
2. **The three non-steel shapes did not move.** Shapes 8, 9 and 10 set
   `steel=False`, and they gave 1.006x, 1.002x and 1.013x. Row 28 measured
   the noise floor at about 1%, so all three are inside it. The gain belongs
   to the steel path alone.

The CPU column came from `profiling/cpu_cache.json` on this sweep, so the
"vs CPU" speedups mix two sweeps. **Do not quote them.** The MLX column is a
fresh stopwatch reading, and it is what this row rests on.

### Accuracy

All 13 shapes PASS, `max_abs` 9.54e-07 to 2.65e-06, against `atol=0.002`.

`test_padding.py` gives 18 of 18 PASS, every case bit exact. A padded batch
must not take the steel kernel, because the kernel handles a string mask
only. Row 27 established that gate. This change moves the gate up one level,
from `_attention()` into `_mlx_transformer()` L604, because the caller now
needs the same answer to choose the head merge. The condition is unchanged:

    steel = plan.steel_attention and isinstance(mask, str)

`mask` does not change inside the layer loop, so testing it once is also
cheaper than testing it for each layer.

A unit check compared the kernel against `mx.fast.scaled_dot_product_attention`
over `head_dim` 8 and 32, `S` 96, 100, 128 and 256, causal and not causal. The
strided path and the contiguous path agree **bit exact** (`max_abs = 0.0`),
and both sit 5.4e-07 to 1.3e-06 from SDPA. `S = 100` covers the ragged case,
where `S` is not a multiple of the 32-row query block.

## 35. Fuse the residual add into the LayerNorm kernel — RULED OUT

**Superseded by row 36. This kernel was never built.** Row 36 removed the
residual add by a different route, so nothing is left here to fuse. The
measurement below still stands, and it is what led to row 36. Read it for the
byte accounting, not for a plan.

The block runs this pair twice:

```python
x1 = x + attn_out              # one kernel
h2 = norm(x1, n2w, n2b, EPS)   # a second kernel
```

The add writes `x1` to DRAM. The LayerNorm reads it back. Nothing else uses
`x1` except the next residual, which reads it again.

### The measurement

Shape 6 chunk, B1024 S128 D128, float32. One activation is 64 MiB. Median of
30 repeats, with `mx.eval` and `mx.synchronize` in the loop.

```
.venv/bin/python3 profiling/stage_roofline.py --shapes 6
```

| chain | eager | `mx.compile` | ratio |
|---|---:|---:|---:|
| add alone | 1.740 ms | — | — |
| LayerNorm alone | 1.243 ms | — | — |
| add then LayerNorm | 2.831 ms | 2.835 ms | 1.00x |
| add, LayerNorm, add | 4.283 ms | 4.317 ms | 0.99x |

**`mx.compile` does not fuse the pair.** This matches row 33, where it does
not fuse the GELU either.

### The byte count agrees

| path | traffic | time at 124 GB/s |
|---|---:|---:|
| two kernels: read `x`, read `y`, write `x1`, read `x1`, write `h2` | 320 MiB | 2.71 ms |
| one kernel, two outputs: read `x`, read `y`, write `x1`, write `h2` | 256 MiB | 2.17 ms |
| one kernel, one output: read `x`, read `y`, write `h2` | 192 MiB | 1.62 ms |

The measured 2.831 ms sits on the two-kernel line. Each pass therefore runs
at the bandwidth roof already. A faster kernel cannot help. Only a kernel
that moves fewer bytes can.

### The estimate

**The one-output line does not apply.** The residual output stays live: the
block computes `x = x + attention` at L654, normalizes it at L656, and then
reads the same `x` again at L658. The second pair behaves the same way
across the layer boundary. So the fused kernel must write two outputs, and
the reachable line is 256 MiB.

One fused pair saves **0.54 ms**. A block holds two pairs, so the estimate
is **1.08 ms of the 16.65 ms shape 6 layer, or 6.5%**.

**The estimate is arithmetic, not a measurement.** It assumes the fused
kernel reaches the same 124 GB/s. Measure it before you claim it.

### Why it is cheap to try

`fast_layernorm.py` already owns this kernel below `d_model = 256`, which
covers shapes 1 to 7 and 9 to 13. The add becomes one extra input operand
and one extra load in the existing kernel. It is not a new kernel.

Shape 8 uses `mx.fast.layer_norm`, so it needs the same treatment as row 33,
or it keeps the two-kernel path.

### The risk

The gain is 6.5% of one shape 6 layer, and it rests on one arithmetic step.
A fused kernel that writes two outputs may not hold 124 GB/s, because it
writes to two buffers instead of one. If it holds only 110 GB/s, the saving
disappears. Measure a two-output kernel at this shape before you build the
plan branch.

## 36. Defer the residual biases, and give the residual add to the GEMM C operand — KEPT

Found by the stage roofline sweep over all 13 shapes. It removes **two whole
elementwise kernels** from every block.

### The problem

The block ran this twice for each layer:

```python
x = x + mx.addmm(layer["ob"], context, layer["ow"].T)
```

That is two kernels. The GEMM writes its result to DRAM, and the add reads it
back with `x` and writes again. At the shape 6 chunk the pair moves 192 MiB
for zero matmul FLOPs, and the stage roofline measured it at 1.635 ms and
1.700 ms, which is 20% of the layer.

### The lever

`mx.addmm(c, a, b)` computes `c + a @ b`. Nobody had checked whether `c` may
be the **whole activation** instead of a broadcast bias. It may, and the
steel GEMM applies it from the accumulator tile, before the write:

| shape | GEMM alone | `x + addmm(bias,·)` | `addmm(x,·)` full C | cost of C | saving |
|---|---:|---:|---:|---:|---:|
| 6, 131072x128 | 1.438 | 2.921 | 1.750 | 0.312 | 1.171 ms |
| 13, 65536x128 | 0.796 | 1.621 | 0.990 | 0.194 | 0.631 ms |
| 8, 8192x1024 | 4.417 | 5.168 | 4.459 | **0.043** | 0.708 ms |

On shape 8 the C operand is nearly free, because that shape is compute bound
and the extra read hides under the matmul.

### The obstacle

`addmm` gives one C, and the block needs both the residual **and** the bias
vector. Adding the bias back as its own kernel destroys the win: 2.855 ms
against 3.002 ms is nothing.

### The answer: never add the bias to an activation

Carry the bias as a `(d_model,)` vector. Let `true_x = x + carry`:

```python
x     = mx.addmm(x, context, layer["ow"].T)     # residual in C, no bias
carry = carry + layer["ob"]                     # 128 elements

h     = norm(x, layer["n2w"], layer["n2b"], carry)   # pre_bias absorbs it

x     = mx.addmm(x, h, layer["fow"].T)
carry = carry + layer["fob"]
```

Every LayerNorm already reads the activation, so `pre_bias` costs one vector
load for each row. The **final** LayerNorm absorbs the accumulated `carry`.
So the model never adds a bias to a full activation, not once.

It is exact, not an approximation. Only the rounding order changes.

### The gates

    defer = plan.defer_bias and not padded and not half

- **`padded`** keeps the plain path. The layer ends with
  `mx.where(valid_tokens, x, 0)`, and that does not commute with a deferred
  bias: zeroing `x` does not zero `x + carry`. `padded` is a compile time
  flag, so the two graphs stay separate.
- **`half`** keeps the plain path, so a float16 or bfloat16 run rounds
  exactly as it did before. See row 13 for why those types cannot pass.
- **`d_model >= 256`** keeps the plain path, because only `fast_layernorm`
  takes a `pre_bias`. That is shape 8 alone, and it is the control below.

### The result

    .venv/bin/python3 scoreboard.py --cpu-cache --label "row 36: ..."

**1.132x FLOP-weighted**, MLX 869.7 ms to 768.6 ms over the 13 shapes.

| # | shape | weight | MLX before | MLX after | MLX | MPS control |
|---:|---|---:|---:|---:|---:|---:|
| 6 | B10000 D128 H4 S128 | 66.5% | 660.808 | 567.842 | **1.164x** | 1.017x |
| 13 | B64 D128 H4 S1024 | 9.4% | 50.299 | 45.231 | 1.112x | 0.996x |
| 11 | B64 D128 H16 S128 | 0.4% | 4.362 | 3.850 | 1.133x | 0.998x |
| 9 | B64 D128 H1 S128 | 0.4% | 4.427 | 3.923 | 1.128x | 1.010x |
| 1 | B64 D128 H4 S128 | 0.4% | 4.230 | 3.788 | 1.117x | 0.999x |
| 10 | B64 D128 H2 S128 | 0.4% | 4.290 | 3.850 | 1.114x | 0.993x |
| 5 | B128 D128 H4 S128 | 0.9% | 8.428 | 7.359 | 1.145x | 0.998x |
| **8** | **B64 D1024 H4 S128** | **21.3%** | **127.573** | **127.557** | **1.000x** | 1.001x |

**Shape 8 is the control.** Its plan sets `defer_bias=False`, and it did not
move. MPS held at 1.013x over the whole sweep. So the gain is this change and
not the machine.

Shape 2 went 0.968x. It is 0.13 GFLOP and every stage sits at the launch
floor, so that is jitter, not a regression. Its FLOP weight is 0.0%.

### Accuracy

All 13 shapes PASS at `atol=0.002` and `rtol=0.02`, `max_abs` 1.19e-06 to
2.65e-06. That is the same band as before the change.

`test_padding.py` is 18/18 bit exact. That test rebuilds a full mask variant
by rewriting the source text of `_mlx_transformer()`, and this change moved
the line it patched, so its patch target moved with it. The padded path
itself did not change.

### What it does not do

Shape 8 carries 21.3% of the FLOP weight and gains nothing, because
`mx.fast.layer_norm` has no `pre_bias` argument. The C operand is nearly free
there (0.043 ms), so a wide `fast_layernorm` variant would collect about
1.53 ms of its 32.51 ms layer, or 4.7%. That is row 37, and it is OPEN.

## 37. A wide `fast_layernorm` variant, so shape 8 reaches row 36 — OPEN

Shape 8 is the only shape that row 36 cannot serve. Its `d_model` is 1024, and
`plan_kernels()` sends every width of 256 and above to `mx.fast.layer_norm`,
which takes no `pre_bias`. So shape 8 keeps two separate residual add kernels
and holds at 1.000x through the row 36 sweep.

### The prize

Measured at the shape 8 projection size, 8192 rows by 1024:

| step | ms |
|---|---:|
| GEMM alone | 4.417 |
| `x + mx.addmm(bias, h, w)`, what shape 8 runs | 5.168 |
| `mx.addmm(x, h, w)`, C is full size | 4.459 |
| **cost of the full C operand** | **0.043** |

The C operand is almost free here. Shape 8 is compute bound, so the extra read
hides under the matmul. Two residual adds per block is **1.53 ms of the
32.51 ms layer, or 4.7%**. At a 21.3% FLOP weight that is about **1.0%
FLOP-weighted**. Measured after row 36: the two adds are 0.785 ms and
0.834 ms, and the C operand would cost 0.043 ms each.

### The cost

`fast_layernorm` gives one SIMD group of 32 lanes to a row, so each lane holds
`ceil(D / 32)` floats. At `d_model` 1024 that is **32 floats per lane**,
against 8 at the widest width it serves today. Register pressure is the open
question. If the kernel spills, it loses more than the 0.043 ms it saves.

Row 31 measured that `mx.fast.layer_norm` already runs at copy speed at
`D >= 256`. So the LayerNorm itself has nothing to win at this width. **The
`pre_bias` hook is the only reason to build this.**

An alternative is a `pre_bias` variant that does not normalize at all, and
instead folds the deferred bias into whatever kernel reads the activation
next. That was not explored.

## 38. Retune the steel attention block shape against the MFA parameter table — REVERTED

`steel_attention()` takes `bq`, `bk`, `wm` and `wn`, and every call site uses
the default `bq=32, bk=32, wm=4, wn=1`. Nothing ever swept them.

`metal-flash-attention` publishes a parameter table for this GPU family. Its
FP32 forward table for Apple9 gives a parallelization block of **16** at every
head width, and a traversal block of 32 at `head_dim <= 48` and 128 at
`head_dim <= 8`. So the table predicted a gain on every steel shape.

It does not transfer.

### The constraint

`steel_attention.h` line 174:

```c
static_assert(BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0, ...);
constexpr int TQ = BQ / (kNWarps * kFragSize);
static_assert(TQ == 1, "Check TQ");
```

`kNWarps = WM * WN` and `kFragSize = 8`, so `BQ = WM * WN * 8` exactly. `bq`
and `wm` are one knob. `bq=16` forces `wm=2`, a 64 thread threadgroup against
128 today.

### The measurement

100 repeats, median, `mx.synchronize()` after each `mx.eval()`. q, k and v are
strided views of one fused QKV buffer, which is what the model gives the
kernel. Ratios are against `bq=32, bk=32, wm=4`.

Shape 6 chunk, B1024 H4 S128 `head_dim` 32, which carries 66.5% of the weight:

| bq | wm | bk | ms | ratio |
|---:|---:|---:|---:|---:|
| 16 | 2 | 16 | 2.3743 | 1.000x |
| 16 | 2 | 32 | 2.7289 | 0.870x |
| 16 | 2 | 64 | 3.4236 | 0.693x |
| 32 | 4 | 16 | 2.3804 | 0.997x |
| **32** | **4** | **32** | **2.3734** | **1.000x** |
| 32 | 4 | 64 | 2.9591 | 0.802x |
| 32 | 4 | 128 | 4.0846 | 0.581x |
| 64 | 8 | 16 | 2.8220 | 0.841x |
| 64 | 8 | 32 | 2.7509 | 0.863x |
| 64 | 8 | 64 | 2.7989 | 0.848x |
| 64 | 8 | 128 | 3.7853 | 0.627x |

The best other config on each remaining case:

| Case | B, H, S, head_dim | Best other config | Ratio |
|---|---|---|---:|
| shape 13 | 64, 4, 1024, 32 | `bq=32 bk=16` | 1.002x |
| shape 1 | 64, 4, 128, 32 | `bq=32 bk=16` | 0.946x |
| shape 7 | 64, 4, 128, 8 | `bq=32 bk=64` | 0.859x |
| shape 11 | 64, 16, 128, 8 | `bq=32 bk=64` | 0.902x |

Accuracy held on every config: `max_abs` 1.01e-06 to 1.91e-06.

### Why the table does not transfer

1. **MFA caches Q and O in registers, and MLX does not.** MFA affords a 16 row
   block because it also blocks over `head_dim`. MLX's kernel stages Q, K and
   V through threadgroup memory at the full head width. At `bq=16` the
   threadgroup halves to 64 threads, but every warp still loads the same K and
   V fragments, so the K and V traffic for each output row doubles.

2. **A smaller `bk` removes arithmetic that is not the limit.** `bk` sets how
   tightly the kernel follows the causal triangle, because
   `kb_lim = (q_max + BK - 1) / BK` skips a K block only when the whole block
   sits above the diagonal. At `seq=128`, `bk=16` computes 9216 cells against
   10240 at `bk=32`, a 10% cut. It buys nothing: `stage_roofline.py` puts this
   stage at 101.5% of the measured bandwidth roof, so the stage is IO bound
   and the removed arithmetic was already free.

### Two configs do not compile

`bq=16, bk=128` and `bq=64, bk=16` both fail with `[metal::Device] Unable to
build metal library from source`. Neither was a contender, so the cause was
not investigated.

## 39. Shift the edge block in the steel GEMM — RULED OUT

`metal-flash-attention` never bounds checks a GEMM edge. It moves the last
block backwards so it overlaps the previous one
(`GEMMKernel+Source.swift:140`):

```c
constant ushort M_shift = (M < M_group) ? 0 : registerM - M_remainder;
```

Every access stays in bounds, the inner loop needs no test, and the overlap is
harmless because the same values are recomputed. `createStoreC()` then shifts
the garbage zone from bottom right to top left before the store.

The motivation was `choose_tile()`, which returns `None` when no tile divides
M, N and K, and then drops the shape to the MLX pair.

**That gate never fires.** Running `choose_tile()`'s own divisibility test on
the real `ffn_in` dimensions of all 13 shapes:

| # | M | N | K | Tiles that divide |
|---:|---:|---:|---:|---|
| 1 | 8192 | 128 | 128 | all four |
| 2 | 128 | 128 | 128 | all four |
| 3 | 512 | 128 | 128 | all four |
| 6 | 131072 | 128 | 128 | all four |
| 7 | 8192 | 32 | 32 | 64x32, 32x32 |
| 8 | 8192 | 1024 | 1024 | all four |
| 13 | 65536 | 128 | 128 | all four |

Every appendix dimension is a power of two. Shape 2 keeps the MLX pair because
of the `rows >= MIN_FUSED_GELU_ROWS` gate, which row 33 measured and which is
correct: M=128 runs at 0.81x because shape 2 is kernel launch bound.

Edge shifting would only matter for a shape whose M, N or K is not a multiple
of 32 or 16. The appendix has none.

## 40. Route the projections with no epilogue through the hoisted steel GEMM — REVERTED

Row 33 measured `steel_addmm(gelu=True)` at 1.51x over `mlx_nn.gelu(mx.addmm())`
at the shape 6 `ffn_in` stage. That number bundles two changes: the fused GELU
epilogue and a different tile. If the tile carried any of it, the same tile
would help `qkv proj`, `out proj` and `ffn_out`, none of which have an epilogue
to fuse. `qkv proj` alone is **27.8% of the shape 6 layer**.

It carries none of it. `steel_addmm(gelu=False)` against `mx.addmm` on the same
GEMM, 8 tiles, 3 interleaved rounds of 100 repeats each:

| Projection | M | N | K | Best steel tile | vs `mx.addmm` |
|---|---:|---:|---:|---|---:|
| shape 6 qkv | 131072 | 384 | 128 | `32x32x16` | 0.978x |
| shape 6 out | 131072 | 128 | 128 | `32x64x16` | 0.991x |
| shape 6 ffn_in | 131072 | 128 | 128 | `32x64x16` | **1.005x** |
| shape 8 qkv | 8192 | 3072 | 1024 | `64x64x16` | 0.996x |
| shape 8 out | 8192 | 1024 | 1024 | `64x64x16` | **1.027x** |

`max_abs` is **0.00e+00** on all 40 readings. That is the proof: MLX dispatches
`mx.addmm` to the same steel kernel this module hoists, and MLX already picks a
good tile. There is no tile to win.

**So row 33's 1.51x is entirely the epilogue.** The table entry for row 33 now
says so.

### The one sub-noise candidate

Shape 8 `ffn_in` with the fused GELU prefers `64x64x16` over today's
`32x64x16`, reproducibly: 4.8374, 4.8482, 4.8379 ms against 5.0857 ms, a
**1.051x** with a 0.2% spread. Today's tile gives only 1.008x against the MLX
pair there, so the fusion is break even at shape 8.

It is not worth the code. `ffn_in` is 14% of the shape 8 layer and shape 8 is
21.3% of the weight, so this is about **0.16% FLOP-weighted**, well under the
1% noise floor row 28 measured. Taking it would also make `choose_tile()`
depend on N and K, which is new complexity for a sub-noise gain.

### Why `qkv proj` sits at 83.5% of peak

It is a short-K GEMM: K=128 against N=384. The arithmetic intensity is fixed by
the shape, not by the kernel. `stage_roofline.py` names it COMPUTE bound at
83.5% of matmul peak, and no tile moved it.

## 41. Reach the steel attention kernel at `head_dim = 256` with a narrow block — REVERTED

This supersedes row 26, which guessed that `bk=8` would fit and never measured
whether it was faster.

`bk=8` does fit. Three block shapes fit the 32 KiB threadgroup at `bd=256`,
because `BQ = WM * WN * 8` lets `bq` go below 32:

| bq | wm | bk | Q_smem | KV_smem | total |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 8 | 8.12 KiB | 12.00 KiB | 20.12 KiB |
| 8 | 1 | 16 | 8.12 KiB | 20.00 KiB | 28.12 KiB |
| 16 | 2 | 8 | 16.25 KiB | 12.00 KiB | 28.25 KiB |

All three are slower than the fallback shape 8 runs today. Shape 8 attention,
B64 H4 S128 `head_dim` 256, causal, 3 interleaved rounds of 100 repeats:

| Path | ms | speedup | max_abs |
|---|---:|---:|---:|
| **MLX fallback (today)** | **3.0840** | **1.000x** | — |
| steel `bq16 bk8 wm2` | 3.4134 | 0.904x | 1.19e-06 |
| steel `bq8 bk8 wm1` | 5.5801 | 0.553x | 1.19e-06 |
| steel `bq8 bk16 wm1` | 8.6021 | 0.359x | 1.19e-06 |

Every one is bit accurate. They are just slower. At `BD=256` with `BK=8` the
kernel runs 16 K block iterations, each staging a 256 wide operand, with one
or two warps in the threadgroup. The threadgroup traffic per unit of work is
what kills it.

### The `d_outer` alternative, and why it is not worth building

`metal-flash-attention` solves exactly this by adding a third block dimension
over `head_dim` and spilling the O accumulator to device memory on purpose
(`AttentionKernel+OuterProduct.swift:451`). Its threadgroup allocation is the
**maximum** over operands, not the sum, so `head_dim=256` costs 8 KiB rather
than 68.5 KiB.

Three measurements say do not build it:

1. **The ceiling is about 1% FLOP-weighted.** `stage_roofline.py --shapes 8`
   puts `sdpa` at 2.2808 ms and `merge heads` at 0.4113 ms of a 32.06 ms
   layer, so 8.4% of shape 8. Shape 8 is 21.3% of the weight. Even a 2x
   returns about 1.0%.
2. **The stage is IO bound, not compute bound.** It runs at 46% of the
   bandwidth roof with the arithmetic units idle. `d_outer` solves register
   pressure to raise ALU utilization. It does not create bandwidth.
3. **Row 21's premise is weaker than recorded.** The profiler reports
   `sdpa peak memory: 80.0 MiB allocated` against `128.0 MiB` for operands and
   output alone, so the MLX fallback is **not** round tripping the
   `B x H x S x S` scores through DRAM at S=128. The score matrix is 16 MiB.

Against that, `d_outer` is a new attention kernel, not an edit. MLX's kernel
has no head block dimension to set.

## 42. Combine the batch chunks with something faster than `mx.concatenate` — REVERTED

`forward()` runs shape 6 in 10 chunks and joins them with `mx.concatenate`.
The output is 625 MiB, so the join reads and writes 1.25 GiB.

`mx.concatenate` is already at the roof. 625 MiB in **10.185 ms is 119.9 GB/s**,
against the 128 GB/s measured on this machine. Every alternative is about 7x
worse, because it moves the bytes on the CPU:

| Route | ms | GB/s | bit equal |
|---|---:|---:|---|
| **`mx.concatenate`** | **10.185** | **119.9** | — |
| `torch.cat` on aliased views | 72.317 | 16.9 | yes |
| `torch.copy_` into a preallocated tensor | 77.016 | 15.8 | yes |
| numpy slice assign on aliased views | 84.725 | 14.4 | yes |

Unified memory means `_to_torch()` can alias an MLX buffer with no copy
(row 23), but it does not make a CPU memcpy fast. Leave the join on the GPU.

### A false lead, recorded so it is not chased again

Timing the whole chunk loop with and without the join suggested the join cost
36.5 ms, not 10.2 ms. The difference is not copy cost. It is allocator
pressure: holding all 10 chunks plus the output alive is 1.25 GiB. Measure the
join on its own, not by subtracting two loop timings.

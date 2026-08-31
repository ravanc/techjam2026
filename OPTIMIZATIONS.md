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
| 33 | Fold GELU into the FFN matmul epilogue | **KEPT** | `steel_gemm.py` `steel_addmm()` L833, tile at `choose_tile()` L679; gated at `plan_kernels()` L508; used at `_mlx_transformer()` L734 | float32, `ffn_in` only, rows >= 512, and a tile that divides M, N and K. Shapes 1 and 3-13. Shape 2 keeps the MLX pair | **1.064x FLOP-weighted** (MLX 768.6 ms to 722.3 ms over the 13 shapes). Shape 6 **1.082x**, shape 3 1.068x, shape 5 1.042x, shape 7 1.042x, shape 13 1.030x, shape 8 1.009x. At the `ffn_in` stage alone: shape 7 **4.49x**, shape 6 1.51x, shape 13 1.47x. **Row 40 attributes that 1.51x: it is the epilogue, not the tile.** `steel_addmm(gelu=False)` ties `mx.addmm` on the same GEMM. Control: MPS held at 1.005x across the two sweeps. All 13 shapes PASS, `max_abs` 1.07e-06 to 2.62e-06. 18/18 padding cases pass |
| 34 | Read q, k and v as strided views, and write the head layout directly | **KEPT** | `steel_attention.py` `steel_attention()`, called at `_attention()` L511, merge at `_mlx_transformer()` L631 | every shape on the steel path: 1-7, 11, 12, 13 | **1.239x FLOP-weighted** (MLX 1077.6 ms to 869.7 ms). Shape 5 1.336x, shape 6 **1.290x**, shape 1 1.301x, shape 11 1.225x, shape 13 1.182x. Two controls: MPS held at **1.000x** across the two sweeps, and the three non-steel shapes 8, 9 and 10 moved 1.006x, 1.002x and 1.013x. All 13 shapes PASS, `max_abs` 9.54e-07 to 2.65e-06. 18/18 padding cases bit exact |
| 35 | Fuse the residual add into the LayerNorm kernel | **RULED OUT** | — | — | **Superseded by row 36, never built.** Row 36 reaches the same prize from the other side: it gives the residual add to the GEMM C operand and defers the bias into the LayerNorm. So the residual add is no longer a kernel, and there is nothing left for this row to fuse. The measurement below still stands and is why row 36 exists |
| 36 | Defer the residual biases, and give the residual add to the GEMM C operand | **KEPT** | `fast_layernorm.py` `layer_norm(pre_bias=)` L220; block at `_mlx_transformer()` L653, L696 and L705; gated at `plan_kernels()` L480 | float32, unpadded, `d_model < 256`. Shapes 1-7 and 9-13. Shape 8 keeps the plain path | **1.132x FLOP-weighted** (MLX 869.7 ms to 768.6 ms over the 13 shapes). Shape 6 **1.164x**, shape 11 1.133x, shape 9 1.128x, shape 1 1.117x, shape 10 1.114x, shape 13 1.112x. Two controls: **shape 8 holds at 1.000x** with `defer_bias=False`, and MPS held at 1.013x across the two sweeps. All 13 shapes PASS, `max_abs` 1.19e-06 to 2.65e-06. 18/18 padding cases bit exact |
| 37 | Defer the residual biases on shape 8, by adding the carry once before the final LayerNorm | **KEPT** | the gate at `plan_kernels()` `use_defer_bias`; the add at `_mlx_transformer()` `norm()` | float32, unpadded, and both `ln1` and `ln2` folded by row 46. **Shape 8** |  **shape 8 1.042x** (MLX 124.911 ms to 119.931 ms), against a **clean 0.996x MPS control on the same shape**. An isolated single-shape run gave 1.044x, so the two agree. Worth about **0.7% FLOP-weighted**: 4.98 ms of the 681.4 ms total. The wide `fast_layernorm` this row originally proposed was NOT built, and it is not needed. **What this row became.** It was "a wide `fast_layernorm`, so shape 8 reaches row 36". Row 46 replaced the need for that. Row 36 needed a `pre_bias` hook on every LayerNorm, and only `fast_layernorm` had one, so shape 8 (`d_model` 1024) could not defer. Row 46 folds `ln1` and `ln2` into the GEMM below them and carries the bias in its `c3` constant, so neither needs a hook. Only the FINAL LayerNorm still does, and it has no GEMM below it. So the fix is one `x = x + carry` before that single call: **one pass over the activation for the whole model, against two deferred residual adds in every layer**. No new kernel. An earlier note here said it would. That was wrong: row 46 removes the need for a `pre_bias` hook at `ln1` and `ln2`, because the carry goes into its `c3` constant, but the FINAL LayerNorm still has no GEMM below it and still needs the carry. So shape 8 runs row 46 with `defer_bias=False` today. **The cheap fix is not this row at all**: add `x = x + carry` once before the final norm, which is ONE elementwise pass for the whole model instead of two for each layer. Measure that before building a wide `fast_layernorm`. The rest of this row still stands: Shape 8 is the one shape row 36 cannot serve, because `mx.fast.layer_norm` has no `pre_bias` argument. The C operand is nearly free there: **0.043 ms** against 0.708 ms for the separate add, because the shape is compute bound and the extra read hides under the matmul. Two residual adds is about **1.53 ms of the 32.51 ms layer, or 4.7%**, so about **1.0% FLOP-weighted**. Measured directly after row 36: the two residual adds are 0.785 ms and 0.834 ms, and the C operand would cost 0.043 ms each. The cost is a `fast_layernorm` that holds `ceil(1024/32) = 32` floats per lane, against 8 at `d_model` 248 today. Register pressure is the open question, and row 31 measured that MLX already runs at copy speed at that width, so the LayerNorm itself has nothing to win. The `pre_bias` hook is the only reason to build it |
| 38 | Retune the steel attention block shape (`bq`, `bk`, `wm`) against the MFA parameter table | **REVERTED** | — | — | the default `bq=32, bk=32, wm=4` is best on every steel shape. The best other config anywhere is **1.002x** (shape 13), inside the 1% noise floor. MFA's own `bq=16` gives 1.000x, 0.956x, 0.929x, 0.824x and 0.895x on the five cases |
| 39 | Shift the edge block in the steel GEMM, so an unaligned tile stays branch free | **RULED OUT** | — | — | nothing to unlock. `choose_tile()` never returns `None` on any appendix shape: every M, N and K is a power of two, and at least two tiles divide all 13. Shape 2 loses the fused GELU to the `rows >= 512` gate, not to divisibility |
| 40 | Route the projections with no epilogue through the hoisted steel GEMM | **REVERTED** | — | — | `mx.addmm` is already optimal. Over 8 tiles on 5 real projection sizes, the best steel tile reaches **1.005x**, and `max_abs` is **0.00e+00** everywhere, which proves MLX dispatches `addmm` to the same steel kernel with a good tile. Shape 6 `qkv proj` sits at 83.5% of matmul peak because K=128 is short, not because of the tile |
| 41 | Reach the steel attention kernel at `head_dim = 256` with a narrow block | **REVERTED** | — | — | supersedes row 26. Three block shapes fit 32 KiB at `bd256`. Best is `bq16 bk8 wm2` at **0.904x** against the MLX fallback; `bq8 bk8` gives 0.553x and `bq8 bk16` gives 0.359x. Bit accurate (`max_abs` 1.19e-06), just slower |
| 42 | Combine the batch chunks with something faster than `mx.concatenate` | **REVERTED** | — | — | `mx.concatenate` already runs at **119.9 GB/s** on the shape 6 output (625 MiB in 10.185 ms), against a 128 GB/s roof. Every alternative is about 7x worse, because a CPU-side copy of unified memory runs at 14-17 GB/s: `torch.cat` 72.3 ms, `torch.copy_` 77.0 ms, numpy slice assign 84.7 ms. All three are bit equal |
| 43 | Fold the LayerNorm into the prologue of the following GEMM | **REVERTED** | — | float32, `fast_layernorm` shapes with a large activation. Shape 6 above all | the prize is real and re-measured: `ln1` and `ln2` are **1.914 ms of a 13.634 ms shape 6 layer (14.0%)**, and no better LayerNorm kernel can win them, because `fast_layernorm` already runs at copy speed. Only fewer bytes can. **Route 1 (recompute the statistics inside the tile) is measured DEAD.** It needs one A tile to cover a whole row, so `bk = K = 128`, and the 32 KiB threadgroup then caps the tile at `bm + bn <= 62` against `bm32 bn64` today. Every one of the 24 tiles that compiles is bit exact (`max_abs` 0.00e+00) and **at best 0.543x**: `qkv proj` 3.602 -> 6.637 ms and `ffn_in` 1.345 -> 2.467 ms, so route 1 ADDS 4.157 ms to save 1.914 ms. The prize is real, so it moves to **row 46**, which reaches it without a prologue |
| 44 | Chain `ffn_in` and `ffn_out` into one kernel | **REVERTED** | `profiling/chain_probe.py`, not in the model | — | built, correct, and it LOSES. Best of 40 configurations is **0.969x**, and a repeat run gave 0.960x. The chain deletes the 128 MiB `hidden` round trip, worth about 1.1 ms, but one threadgroup must own all of `ffn_dim` to hold the row, and the threadgroup memory that costs takes more than the round trip gives. Controlled pair at `bm = 32`: `bk8` uses 24.0 KiB and reaches 0.996x, `bk16` uses 29.0 KiB and reaches 0.896x — same tile, more threadgroup, worse time. Relative error 3e-07, so it is correct, just slower |
| 45 | Build the deferred bias `carry` at weight build time | **OPEN** | partly delivered: `TorchToMLX` weight build L984-L1000 folds the carry into row 46's `ln1c3`/`ln2c3` | what remains is the run-time accumulation at `_mlx_transformer()` L821 and L849, which serves the final LayerNorm and any shape that declines row 46 (shape 2) | not measured, and a sweep CANNOT measure it. It is 2 kernels for each layer on a `(d_model,)` vector, so about 8 launches of 0.004 ms each for the whole forward. That is about **0.1% FLOP-weighted**, under the 1% noise floor. Worth doing for shape 2 alone, and shape 2 carries 0.0% of the FLOP weight |
| 46 | Absorb the LayerNorm into the weights, and apply it in the GEMM epilogue | **KEPT** | `fast_layernorm.py` `layer_norm_stats()` L305; `steel_gemm.py` `layer_norm_constants()` L1173 and the `apply_layer_norm_epilogue` patch at L136; gated at `plan_kernels()`; used at `_mlx_transformer()` | float32, unpadded, no padded head, rows >= 512, and a tile that divides M, N and K. **Shapes 1 and 3-13, shape 8 included.** Shape 2 keeps the plain path | **1.060x FLOP-weighted** (MLX 722.3 ms to 681.4 ms over the 13 shapes). Shape 6 **1.073x**, shape 12 1.152x, shape 4 1.139x, shape 3 1.103x, shape 13 1.057x, shape 8 1.012x. Two controls: MPS held at **0.992x** across the two sweeps, and shape 2, which declines this row, moved 0.974x on MLX while its own MPS control moved 0.919x, so that is machine noise on a 0.65 ms shape and not a regression. All 13 shapes PASS, `max_abs` 1.07e-06 to 3.34e-06. 18/18 padding cases pass. It supersedes row 43. It supersedes row 43, which tried the same prize with a prologue and lost the tile. A LayerNorm is affine in the row, so it distributes through the matmul that follows it and folds into three constants built at weight build time. What is left at run time is one GEMM over RAW `x` and an epilogue that reads two floats for the row, so **the LayerNorm never writes an activation**. Worth about **1.0 ms per shape 6 layer, or 7.2%**: it replaces `ln1` and `ln2` (1.914 ms) with two statistics passes (about 0.92 ms). **The accuracy risk is measured and it passes**: 0.9x to 1.2x of the error the model already carries, on shapes 6, 7, 8 and 13, by `profiling/ln_absorb_probe.py`. Unlike row 43 it has no `d_model < 256` gate, so it reaches shape 8 (21.3% of the FLOP weight) and it subsumes row 37 |
| 48 | Hide the framework boundary behind the GPU, by pipelining the chunk loop | **REVERTED** | `profiling/pipeline_probe.py`, not in the model | — | built, correct, and it LOSES: **0.974x**. Unified memory means the CPU copy and the GPU kernels contend for one memory system, so the copy does not hide. An unrelated 625 MiB CPU memcpy costs 31.40 ms alone and **+45.06 ms inside the loop**, so overlapping is WORSE than serial. A second run gave 34.53 ms alone and +31.31 ms inside, 91% unhidden. A lagged eval with the bulk convert kept gives 1.004x, inside the noise floor. **This rules out the whole class of transfer-overlap ideas on this machine.** The boundary is real (5.4% of shape 6, about 4.5% FLOP-weighted) but it is not recoverable this way |
| 47 | Produce the LayerNorm statistics in the epilogue of the GEMM that writes the activation | **KEPT** | `steel_gemm.py` `_ROW_STATS_EPILOGUE` and `row_stats_reduce()`; gated at `plan_kernels()` `fuse_stats_out` and `fuse_stats_ffn`; used at `_mlx_transformer()` | float32, unpadded, `use_defer_bias`, rows >= 512, and a tile that divides M, N and K. Shapes 1 and 3-13, shape 8 included. Shape 2 keeps the plain path | **1.075x FLOP-weighted** (MLX 686.6 ms to 638.7 ms over the 13 shapes). Shape 6 **1.091x**, shape 3 1.098x, shape 7 1.089x, shape 5 1.081x, shape 11 1.067x, shape 13 1.045x, shape 8 1.027x. Control: MPS held at a 1.006x median across the two sweeps, and shape 6's own MPS moved 0.970x, so the machine was if anything slower. Shape 12 read 0.922x in the sweep; a controlled interleaved A/B says **1.008x to 1.013x**, so that reading was machine drift and not a regression. All 13 shapes PASS, `max_abs` 1.19e-06 to 3.46e-06. 18/18 padding cases bit exact |
| 49 | Block the head dimension of K and V, so `head_dim = 256` fits the threadgroup | **REVERTED** | `profiling/d_outer_attention.py` and `profiling/d_outer_probe.py`, not in the model | — | built, correct on **26 of 26** block shapes (`max_abs` 1.31e-06 to 1.55e-06), and every one LOSES. Best is `bq16 bk8 bdc64` at **0.832x**, which is worse than row 41's **0.904x** for the same `bq16 bk8` block with no D block. **The premise is wrong**: row 41 read `BK = 8` as the handicap, and this row removed the threadgroup limit that forced it. A larger `BK` then made it slower, not faster. Every `bk8` config beats every `bk32` config at the same `bq`. The stage is IO bound, and `BK` does not change the bytes |

| 50 | Apply the FINAL LayerNorm in the epilogue of the GEMM above it | **KEPT** | `steel_gemm.py` `_FINAL_LN_EPILOGUE` L298 and `choose_final_ln_tile()` L714; gated at `plan_kernels()` L633; used at `_mlx_transformer()` L856 | float32, unpadded, and a full row tile (`bn == d_model`, `wn == 1`) that fits the threadgroup AND has exact loader geometry, which holds for `d_model` in {8, 16, 32, 64, 128}. **Shapes 1-7 and 9-13.** Shape 8 cannot: `bn = 1024` needs 80 KiB | **1.019x FLOP-weighted** by a controlled A/B on all 13 shapes (`profiling/plan_ab.py`), and 1.011x from the two sweeps (MLX 635.2 ms to 628.8 ms). Shape 6 **1.0211x** with no overlap between the two distributions (OFF min 441.06 ms, ON max 436.36 ms), shape 11 1.0315x, shape 7 1.0195x, shape 5 1.0180x, shape 9 1.0134x, shape 1 1.0128x, shape 13 1.0117x, shape 12 1.0117x. **Shape 8 is the null control**: it takes no tile, so both sides run identical code, and it reads 0.9911x, 1.0027x and 0.9994x over three runs. That is the noise floor of the A/B. **Every one of the 13 shapes ran the A/B three times.** Shape 6 repeats at 1.0211x, 1.0210x and 1.0202x, a 0.09 pp spread, while its own OFF median drifted 2.8%. Every shape that takes the row has a median above 1.010x, and only 3 of the 39 readings fall below 1.000x (two are the null control). Take the WORST of three runs on every shape at once and it is still **1.0185x FLOP-weighted**. MPS held at a **1.000x median** across the two sweeps. No shape loses: the lowest is shape 3 at 0.9994x. At the GEMM alone the fusion is **1.399x** at the shape 6 chunk and 1.412x at shape 13 (`profiling/final_ln_probe.py`). All 13 shapes PASS, `max_abs` 1.43e-06 to 2.38e-06. 18/18 padding cases pass. **Rows 37, 45, 46 and 47 each named this LayerNorm as the one that stays, and none of them measured it.** Every one asked whether a GEMM sits BELOW it, which is what row 46 needs, and the answer is no. The GEMM ABOVE it works instead |

| 51 | Refuse a steel tile whose block loader has inexact thread geometry | **KEPT** (a bug fix) | `steel_gemm.py` `_loader_ok()` and `loader_geometry_ok()` L689; applied in `choose_tile()`, `choose_final_ln_tile()` and `steel_addmm()` | every steel GEMM. It changes NO appendix plan: all 13 are identical before and after | found while building row 50. MLX's `BlockLoader` derives `n_reads`, `TCOLS` and `TROWS` by TRUNCATING integer division with no guard, so a tile it never dispatches gives a **silently wrong answer**. Measured on a plain GEMM with no epilogue (M=1024 K=128, `bm32 bk16`, 128 threads): `BN` 32, 64 and 128 give `max_abs` **0.00e+00**, while `BN` 48 and 96 give **5.2e+00**, and `BN` 160 does not compile (`TCOLS` is 0). At `BN = 96`: `n_reads = 12`, `TCOLS = 16 / 12 = 1`, `TROWS = 128`, so 128 threads load 128 rows of a 96 row tile. The predicate reproduces the measured set exactly: it admits {8, 16, 32, 64, 128} and rejects 24, 40, 48, 56, 72, 80, 88, 96, 104, 112, 120 and 160. It was latent before row 50, because `_TILES` holds only `bn` 32 and 64, and all four tiles pass. 84/84 configurations over 7 `d_model`, 3 `seq_len`, 2 batch and both causal settings now PASS |

| 52 | Close the kernel launch gaps with a persistent kernel | **RULED OUT** | `profiling/gpu_timeline.py`, not in the model | — | **read off the Metal timeline, not from a slope fit.** GPU idle BETWEEN the kernels of a shape 6 forward is **1.05%** of the call. 9 or 10 gaps of about 0.45 ms hold all of it, and shape 6 runs exactly 10 chunks, so those gaps are the `mx.eval` at the chunk boundary, not a launch bubble. Every other gap is under 0.05 ms, and the 90 of them together hold **0.269 ms of a 462 ms window, which is 0.06%**. That 0.06% is the whole prize of a persistent kernel. The idle that IS large sits before the first kernel, and it is the `_to_mlx` input copy that row 48 already ruled out |

| 53 | Fuse the attention and the out projection into one kernel | **RULED OUT** | `profiling/attn_out_budget.py`, not in the model | — | stopped on the budget, before any kernel was written. The out projection mixes every head, so one threadgroup must own `bq` query rows over the FULL `d_model`. At shape 6 that is **25.50 KiB of the 32 KiB budget with aliasing, and 29.62 KiB without**, against the **9.00 KiB** the steel attention kernel uses today. Row 44 measured that exact band on this machine: 24.0 KiB gave 0.996x and 29.0 KiB gave 0.896x. Same prize as row 44 too, and the same size: a 128 MiB round trip at the shape 6 chunk. **Shape 8 cannot hold it at all** (197 KiB), and nor can shapes 9 and 10 |

| 54 | Re-sweep the steel GEMM tile, with the row 46 and row 47 epilogues on | **REVERTED** | `profiling/tile_resweep.py`, `_TILES` unchanged | — | 129 tiles on each of the four shape 6 GEMM stages, each one paired against today's tile and alternated every repeat. **No tile wins.** `qkv proj` and `ffn_in` already run their best tile: the top candidate over 129 is 0.997x and 1.007x. The two row 47 stages prefer `32x64x32x2x2`, but `out proj` reads 1.025x, 1.018x and 1.012x and `ffn_out` reads 1.030x, 0.979x and 1.006x, while the null control (today against today) moves 0.976x to 1.009x. **The `bn = 128` hypothesis is refuted**: a `bn128` tile never reached the top of any stage. The largest reproducible effect is `32x64x16x1x4` on `ffn_in` at 1.007x, 1.008x and 1.008x, which is **0.07% FLOP-weighted** |

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

**The table above is stale, and its SHARES are stale by more than its
milliseconds.** The kernels are 3.7x faster since it was taken, and the
boundary is not, so the same copy is now 3.1% of shape 6 rather than 1.0%.
Row 48 re-measured it and tried to hide it behind the GPU. It cannot be
hidden: unified memory makes the copy and the kernels contend. Read row 48
for the current numbers.

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

    <mlx package>/include/mlx/backend/metal/kernels/steel/attn/

`mlx_kernels.py` finds that directory. It asks the installed `mlx` package
for its own path, so a venv at any place and at any Python version works.
Set `MLX_KERNELS_DIR` to override the search. When the headers are absent,
`steel_attention.supports()` returns False and `steel_gemm.choose_tile()`
returns None, so the model takes the plain MLX path and the run completes.

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

**The ceiling is small.** Shape 8 is 8.9% attention by measured time, not the
4% the FLOP table implies, because `d_model=1024` makes the projections 89%
of the work. `stage_roofline.py --shapes 8` on 31 August 2026 gives `sdpa`
2.651 ms and `merge heads` 0.723 ms of a 29.813 ms layer. Rows 46 and 47 cut
the layer from 32.06 ms, so attention holds a larger share than this row first
recorded. The `sdpa` floor is 1.049 ms at its byte count, so a perfect
attention kernel gives 1.06x on shape 8, which is about 1.2% of the weighted
score.

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

## 37. Defer the residual biases on shape 8 — KEPT

**Built as something other than what this row proposed.** The original plan
was a wide `fast_layernorm`, so shape 8 could reach row 36's `pre_bias` hook.
That kernel was never written, and it is not needed. Row 46 removed the
reason it existed.

### Why row 46 unblocked it

Row 36 defers each residual bias into a `carry` vector, and every LayerNorm
in the block must then apply that carry. Only `fast_layernorm` had a
`pre_bias` hook, and it serves a row width under 256, so shape 8 could not
defer at all.

Row 46 folds `ln1` and `ln2` into the GEMM below each of them, and the carry
rides in the `c3` constant that the GEMM's C operand adds. Neither norm needs
a hook any more.

That leaves exactly one LayerNorm that still needs the carry: the final one,
which has no GEMM below it. So the whole fix is:

    if pre is not None and not plan.fast_layer_norm:
        value = value + pre     # once, for the whole model
        pre = None

`layer_norm(x, w, b, eps, pre_bias=c)` is `layer_norm(x + c, w, b, eps)` by
definition, so the arithmetic is unchanged.

**The trade.** One pass over the activation, once per forward, against two
deferred residual adds in every layer. Shape 8 has 4 layers, so it trades one
pass for eight.

The gate widened to match:

    use_defer_bias = use_fast_ln or (
        fuse_ln_qkv is not None and fuse_ln_ffn is not None)

Both tiles are required. If only one norm folds, the other still needs a hook
that `mx.fast.layer_norm` does not have.

### The measurement

    .venv/bin/python3 scoreboard.py --cpu-cache --label "row 37 cheap form, re-measured: shape 8 defers its residual biases"

| | MLX before | MLX after | ratio | MPS control |
|---|---:|---:|---:|---:|
| shape 8 | 124.911 | **119.931** | **1.042x** | 165.374 -> 165.999, **0.996x** |

An isolated single-shape run on a quieter machine gave 124.911 -> 119.678,
**1.044x**. The two agree.

Worth about **0.7% FLOP-weighted**: 4.98 ms of the 681.4 ms total.

Accuracy on shape 8: PASS, `max_abs` 3.22e-06 against 3.34e-06 before, so it
got slightly BETTER. That is expected: a deferred bias keeps the residual
stream smaller for longer. 18 of 18 padding cases pass.

### Read the total of that sweep with care

**The sweep-wide MPS control moved 0.974x, which is outside the 1% noise
floor, so the sweep total cannot score this change.** `mysqld` was holding
76% of a CPU for the whole run. The per-shape control is what decides here,
and shape 8's own MPS held at 0.996x while its MLX gained 4.2%.

Every shape this change does NOT touch tracked its own MPS control:

| shape | MLX ratio | MPS control |
|---:|---:|---:|
| 1 | 0.976x | 0.972x |
| 4 | 0.942x | 0.936x |
| 5 | 0.962x | 0.976x |
| 13 | 0.966x | 0.971x |

So nothing regressed. The machine was slower, and the untouched shapes
followed it.

### The old plan, kept because it is still the fallback

The wide `fast_layernorm` is still the answer if a future shape folds only
one of its two norms. What it would cost is unchanged from the original
note:

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
   on 31 August 2026 puts `sdpa` at 2.651 ms and `merge heads` at 0.723 ms of
   a 29.813 ms layer, so 11.3% of shape 8. Rows 46 and 47 cut the layer from
   32.06 ms, which raises the share from the 8.4% this row first recorded.
   Shape 8 is 21.3% of the weight. The `sdpa` floor is 1.049 ms, so a perfect
   kernel returns about 1.2%.
2. **The stage is IO bound, not compute bound.** It runs at 40% of the
   bandwidth roof and 20% of the matmul roof, so the arithmetic units idle. `d_outer` solves register
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

## 43. Fold the LayerNorm into the prologue of the following GEMM — REVERTED

A stage roofline run on all 13 shapes placed this row. Command:

    .venv/bin/python3 profiling/stage_roofline.py --shapes 6,8
    .venv/bin/python3 profiling/stage_roofline.py --shapes 1,2,3,4,5,7,9,10,11,12,13
    .venv/bin/python3 profiling/stage_roofline.py --shapes 13

### What the run measured

Shape 6 carries 66.5% of the FLOP weight. Its layer splits like this:

| stage | ms | FLOP/B | %comp | %mem | limit |
|---|---:|---:|---:|---:|---|
| `ln1` | 1.087 | 0.00 | 0.0 | **96.5** | IO |
| `qkv proj` | 3.617 | 47.96 | **87.7** | 58.0 | COMPUTE |
| `sdpa` (steel) | 2.185 | 16.12 | 48.8 | **96.0** | IO |
| `out proj` | 1.653 | 21.33 | 64.0 | **95.2** | IO |
| `ln2` | 1.072 | 0.00 | 0.0 | **97.8** | IO |
| `ffn_in + gelu` | 1.379 | 31.98 | 76.7 | 76.1 | COMPUTE |
| `ffn_out` | 1.620 | 21.33 | 65.3 | **97.1** | IO |

The sum of the stages is 12.613 ms. The real layer is 12.939 ms, so the stage
model accounts for 97.5% of the time.

**Read the `%mem` column as a rank, not as a fraction of the roof.** It
overstates, because `ms` has `FLOOR_MS` removed and the 128 GB/s roof does
not. A later run of the same tool printed `ln1` at 110.6%. See
`references/machine.md`.

**The conclusion survives the correction: the IO stages are at the roof.**
Checked directly, and not through the biased column: a standalone
`fast_layernorm` at the shape 6 `ln1` size takes 1.305 ms for 128 MiB, which
is 107.5 GB/s, and a plain `x * 2.0` at the same size reaches 108.9 GB/s. The
LayerNorm already runs at copy speed. A better LayerNorm kernel cannot win it.
Only fewer bytes can.

The ridge point of this machine is 31.7 FLOP/byte, and every IO stage sits
below it: a LayerNorm does zero matmul FLOPs, and the two residual
projections sit at 21.3 FLOP/byte. Those stages cannot reach the arithmetic
peak whatever the kernel does.

### The prize

`ln1` and `ln2` together are 2.158 ms, or 17.1% of the shape 6 layer, on the
run above. **A later run measured 1.914 ms of a 13.634 ms layer, or 14.0%**,
and that is the figure the route 1 arithmetic below uses. The two runs differ
by machine load, not by a code change. Row 46 carries the current number.

`ln1` reads `x` (64 MiB) and writes `h1` (64 MiB). `qkv proj` then reads `h1`
back. A GEMM prologue that normalizes `x` in place deletes the write and the
re-read, which is 128 MiB, or about 1.0 ms at the 123.5 GB/s the stage reaches.

`qkv proj` has room to absorb it: 87.7% of matmul peak against 58.0% of the
bandwidth roof. Row 37 measured the same effect on the C operand and found it
nearly free on a compute bound stage. `ffn_in` is tighter, at 76.7% and 76.1%.

### The open question, and the answer

The GEMM re-reads each A row once for each N tile. `qkv proj` has N = 384, so
a `bn` of 64 gives 6 tiles and 6 recomputes of the row statistics. Two routes:

1. Recompute the mean and the variance in each tile. It moves the fewest bytes
   and costs redundant reduction work.
2. Compute the statistics in a first pass that writes 2 floats for each row
   (about 1 MiB), then apply them in the prologue. It saves 63 MiB of the 128
   MiB and adds no redundant work.

**Route 1 is measured, and it is dead.** Route 2 is now the only route.

### Route 1 is REVERTED. The tile it forces costs more than the LayerNorm

Route 1 needs the whole row inside one A tile, because the mean and the
variance are reductions over the row. So `bk` must equal `K`, which is 128 at
every shape except 7 and 8.

The 32 KiB threadgroup then caps the tile. `steel/gemm/gemm.h` sizes the two
buffers as `bm * (bk + 4)` and `bn * (bk + 4)` floats, so:

    (bm + bn) * (bk + 4) * 4 <= 32768   ->   at bk = 128,  bm + bn <= 62

The model runs `bm32 bn64` today, which is 96. So route 1 must halve the tile
area. Measured at the real shape 6 chunk sizes (M = 131072, K = 128), median
of 50 repeats, against `bm32 bn64 bk16`:

| tile | `qkv proj` N=384 | `ffn_in` N=128 |
|---|---:|---:|
| `bm32 bn64 bk16` (today) | **3.602 ms** | **1.345 ms** |
| `bm32 bn16 bk128 wm2 wn2` | 6.637 ms  0.543x | 2.467 ms  0.545x |
| `bm16 bn32 bk128 wm1 wn2` | 6.677 ms  0.540x | 2.523 ms  0.533x |
| `bm16 bn16 bk128 wm2 wn2` | 7.527 ms  0.479x | 2.658 ms  0.506x |
| `bm16 bn32 bk128 wm1 wn1` | 10.070 ms 0.358x | 3.530 ms  0.381x |

Every tile that compiles is **bit exact** (`max_abs = 0.00e+00`), so this is a
speed loss and nothing else. A tile also needs `bm % (wm*8) == 0` and
`bn % (wn*8) == 0`, or the MMA fragment count is zero and Metal refuses the
source. That rules out `bn = 8` entirely.

The arithmetic that kills the row:

| | ms per shape 6 layer |
|---|---:|
| route 1 adds to `qkv proj` | +3.035 |
| route 1 adds to `ffn_in` | +1.122 |
| **route 1 adds** | **+4.157** |
| it deletes `ln1` and `ln2` | -1.914 |
| **net** | **+2.243, a 1.19x slowdown** |

Command:

    .venv/bin/python3 profiling/tile_probe.py --row 43

Two runs of that command agree on the verdict and differ inside the noise:
the best `bk = 128` tile gave 0.543x then 0.528x on `qkv proj`, and 0.545x
then 0.579x on `ffn_in`. Nothing reaches 0.6x.

The earlier estimate in this row assumed the GEMM would hold its rate at
`bk = 128`. It does not. That assumption was the whole basis of the "0.8%
extra ALU per N tile" line, which is true of the arithmetic and irrelevant:
the cost is the tile, not the reduction.

### Route 1 could not have served shape 8 either

Shape 8 carries 21.3% of the FLOP weight and is compute bound: its four GEMMs
are 25.638 ms of a 31.388 ms layer (81.7%), and they run at 95.8% to **100.4%**
of the 4.06 TFLOP/s roof. A reading above 100% is the `%comp` bias described
in `references/machine.md`: the stage time has `FLOOR_MS` removed and the roof
does not, so the ratio reads high. It is not a stage that beat physics.

Shape 8 has no GEMM headroom, and route 1 spends the GEMM to buy the
LayerNorm. Its only other slack is the FALLBACK `sdpa` at 2.433 ms (21.9%
comp, 43.1% mem), and rows 21, 26 and 41 close that route.

**This does not rule out row 46.** Row 46 does not touch the tile, so it
takes no GEMM rate from shape 8. It removes a LayerNorm pass instead.

### What survives: row 46

Route 2 puts no constraint on `bk`, because the statistics arrive as two
floats per row, so it keeps the `bm32 bn64 bk16` tile. It grew into its own
optimization, and it now lives at **row 46** with its own measurement.

## 44. Chain `ffn_in` and `ffn_out` into one kernel — REVERTED

It was built, it is correct, and it loses. The kernel and the sweep live in
`profiling/chain_probe.py`:

    .venv/bin/python3 profiling/chain_probe.py

### What it does

`ffn_in` writes `hidden` to DRAM and `ffn_out` reads it straight back. At the
shape 6 chunk that round trip is 128 MiB, about 1.1 ms of a 12.0 ms layer.

The chained kernel keeps `hidden` in threadgroup memory:

    phase 1   hidden = gelu(layer_norm(x) @ W1 + b1)   -> threadgroup
    phase 2   out    = residual + hidden @ W2          -> device

Three pieces made it possible, and all three were checked before it was
written:

1. `BlockMMA::mma()` already takes a THREADGROUP pointer for its A operand
   and strides it by the template constant `lda_tgp`. So phase 2 reads
   `hidden` in place, with `lda_tgp = ffn_dim + 4`.
2. `MMATile::store` has a threadgroup overload, so `store_result_tgp` writes
   the phase 1 accumulator into `hidden`.
3. The budget fits, but only with aliasing: phase 2's `Bs2` reuses the dead
   `As1` and `Bs1`. Peak 29.0 KiB of 32.0 at `bm32 bk16`. Without the
   aliasing it is 39.0 KiB and does not fit at all.

It is possible only because `appendix_cases.py` sets `ffn_dim == d_model` on
every shape. A normal transformer with `ffn_dim = 4 * d_model` cannot hold a
row block of `hidden` at all.

### Why it loses

One threadgroup must own `bm` rows and ALL of `ffn_dim`, so `bn1 = ffn_dim`
and there is a single N tile. That is exactly what makes the whole `hidden`
row available to phase 2. It is also what makes the kernel
threadgroup-hungry, and threadgroup memory is what limits how many
threadgroups stay resident on a core.

Best of 40 configurations, at the shape 6 chunk (M = 131072, K = 128,
N = 128), against the three kernels the model runs today:

| tile | threadgroup | best ratio |
|---|---:|---:|
| `bm32 bk8 wm4 wn2` | 24.0 KiB | **0.969x** |
| `bm32 bk8 wm2 wn4` | 24.0 KiB | 0.994x on one run, 0.968x on another |
| `bm16 bk16 wm2 wn4` | 19.5 KiB | 0.966x |
| `bm32 bk16 wm2 wn4` | 29.0 KiB | 0.896x |
| `bm32 bk16 wm4 wn1` | 29.0 KiB | 0.614x |
| `bm8 bk16 wm1 wn4` | 14.8 KiB | 0.694x |

**The controlled pair is the evidence.** Hold `bm = 32` and change only
`bk`:

| | threadgroup | best |
|---|---:|---:|
| `bk = 8` | 24.0 KiB | **0.996x** |
| `bk = 16` | 29.0 KiB | **0.896x** |

Same tile shape, more threadgroup memory, worse time. Occupancy is the
mechanism.

The two ends of the sweep confirm it from the other side. A small `bm`
lowers threadgroup use but wastes the tile: `bm8` reaches only 0.694x. So
the kernel is squeezed between a tile too small to be efficient and a
threadgroup too large to be resident, and no point between them wins.

### It is correct, so this is a speed result and nothing else

Relative error against the four-kernel path is 3.6e-07 at 1024 rows,
4.4e-07 at 4096 rows, and 2.4e-07 at `d_model` 32. The arithmetic is right.

### What this does NOT rule out

The round trip is still 128 MiB and still real. What failed is deleting it
by holding `hidden` in threadgroup memory. Any future attempt has to find a
way that does not spend the occupancy, and this row does not know one.

Do not try the same shape of kernel again. The tile probe in
`profiling/tile_probe.py` measured the `bn = ffn_dim` tile at 0.884x before
this was built, and the full kernel came in at 0.969x, so the extra loss
beyond the tile is the occupancy.

## 46. Absorb the LayerNorm into the weights, and apply it in the GEMM epilogue — KEPT

**This row supersedes row 43.** Row 43 chased the same prize with a GEMM
prologue and lost, because the prologue forces `bk = K` and that tile runs at
0.543x. This row reaches the prize from the other side: it never builds a
prologue at all.

Derived while measuring row 43. Read that row first for what failed.

### The algebraic form

Route 2 puts no constraint on `bk`, because the statistics arrive as two
floats per row. The loader applies them as it reads each `bk` chunk. So the
`bm32 bn64 bk16` tile stays.

There is a stronger form. The LayerNorm is affine in the row once the mean and
the rstd are known, so it distributes through the matmul. Write `m` and `r`
for the row mean and rstd, `w` and `b` for the LayerNorm gain and bias, and
`B` for the GEMM weight:

    y[i,j]   = (x[i,j] - m_i) * r_i * w_j + b_j
    out[i,n] = sum_j y[i,j] * B[j,n]
             = r_i * (X @ (w * B))[i,n] - m_i * r_i * (w . B)[n] + (b . B)[n]

Every term that depends on `j` alone is constant across a run:

- `w * B` scales the rows of the weight. Build it once, at weight build time.
- `(w . B)[n]` and `(b . B)[n]` are two `(N,)` vectors. Build them once.

What is left at run time is one GEMM of the RAW `x` against a prepacked
weight, and an epilogue that reads two floats for the row:

    out = r_i * (acc - m_i * c1[n]) + c2[n] + bias[n]

So the LayerNorm needs **no prologue at all**. It becomes an epilogue, and
row 33 already owns that epilogue in `steel_gemm.py`. This is the article's
optimization #1 (move the constant work to load time) applied to its
optimization #3 (fuse the norm into the next kernel). See `agent_loop.md`.

**The open risk was accuracy, not speed. It is now measured, and it passes.**

The reassociation runs the sum over un-normalized `x`, then subtracts
`m_i * c1[n]`. That is a difference of two large close numbers, and
catastrophic cancellation there is the reason LayerNorm exists. So the kill
test for route 2 is arithmetic, and it writes no Metal:

    .venv/bin/python3 profiling/ln_absorb_probe.py --shape 6

It takes the residual stream from a real forward at the real shape, then
computes one projection three ways: float64 as the reference, float32 the way
the model runs it today, and float32 with the LayerNorm absorbed. `max_abs`
against the float64 reference, worst of the 4 layers:

| shape | worst row `abs(mean)/std` | today | route 2 | ratio | headroom |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.33 | 1.28e-06 | 1.49e-06 | 1.2x | 1340x |
| 7 | 0.93 | 6.49e-07 | 7.62e-07 | 1.2x | 2626x |
| 8 | 0.13 | 4.52e-06 | 4.62e-06 | 1.0x | 433x |
| 13 | 0.33 | 1.60e-06 | 1.49e-06 | 0.9x | 1340x |

Route 2 costs between 0.9x and 1.2x of the error the model already carries.

**Why the cancellation does not bite.** It depends on the row ratio
`|mean| / std`, and that ratio stays near 0.3 at every shape. The residual
stream does not drift far from zero. The harness benchmarks an UNTRAINED
model — `generate_random_case()` builds random weights — so this is the
regime that the gate actually measures, not a lucky case.

**This test covers one projection, not the whole model.** The error compounds
over 4 layers and 2 norms per layer. The `scoreboard.py` accuracy table is
still the gate that decides.

### It reaches shape 8, and it makes row 37 unnecessary

The prologue form of this row needed `fast_layernorm`, so it stopped at
`d_model < 256` and shape 8 was out of scope. **Route 2 has no such gate.** It
needs a statistics pass and an epilogue, and neither cares about the row
width. So route 2 serves shape 8, which carries 21.3% of the FLOP weight.

It also carries the deferred residual bias of row 36 through `c3[n]`, which
is the exact thing row 37 exists to give shape 8. So route 2 subsumes row 37.
Build route 2 first, then re-read row 37 to see whether anything is left.

### What it is worth — estimated, then measured

The estimate, before it was built:

| | ms per shape 6 layer |
|---|---:|
| `ln1` + `ln2` today | 1.914 |
| the statistics pass, twice (65 MiB each) | about 0.92 |
| **net** | **about -1.0, or 7.2% of the layer** |

Measured on the two GEMMs alone, at the real shape 6 chunk (M = 131072,
K = 128), median of 50 repeats:

| chain | today | row 46 | |
|---|---:|---:|---:|
| LayerNorm alone | 1.183 | — | |
| statistics alone | — | 0.653 | |
| `ln2` + `ffn_in` + GELU (N=128) | 2.627 | 2.204 | **1.192x** |
| `ln1` + `qkv proj` (N=384) | 4.581 | 4.142 | **1.106x** |

So the epilogue itself is nearly free: the statistics pass saves 0.530 ms
and the chain realizes 0.423 ms of it.

### The sweep that kept it

    .venv/bin/python3 scoreboard.py --cpu-cache --label "row 46: absorb the LayerNorm into the GEMM weights, apply it in the epilogue"

**1.060x FLOP-weighted**, MLX 722.3 ms to 681.4 ms over the 13 shapes.

| # | MLX before | MLX after | ratio | MPS control |
|---:|---:|---:|---:|---:|
| 1 | 3.656 | 3.506 | 1.043x | 1.000x |
| 2 | 0.634 | 0.651 | **0.974x** | 0.919x |
| 3 | 0.709 | 0.643 | 1.103x | 1.013x |
| 4 | 1.337 | 1.174 | 1.139x | 1.036x |
| 5 | 7.065 | 6.681 | 1.057x | 0.994x |
| 6 | 524.885 | 489.310 | **1.073x** | 0.989x |
| 7 | 1.079 | 1.055 | 1.023x | 1.002x |
| 8 | 126.408 | 124.911 | 1.012x | 1.002x |
| 9 | 3.861 | 3.602 | 1.072x | 1.009x |
| 10 | 3.742 | 3.592 | 1.042x | 1.004x |
| 11 | 3.754 | 3.602 | 1.042x | 1.003x |
| 12 | 1.285 | 1.115 | 1.152x | 1.050x |
| 13 | 43.893 | 41.518 | 1.057x | 1.000x |

**Shape 2 went 0.974x, and it is noise, not a regression.** Shape 2 declines
this row: it has 128 rows, under the 512 row gate, so `plan_kernels()` gives
it `fuse_ln_qkv=none`. Its own MPS control moved 0.919x on the same sweep, so
the machine was noisier for tiny shapes. Shape 2 is 0.65 ms and 0.0% of the
FLOP weight.

**Control.** MPS held at 0.992x across the two sweeps, inside the 1% noise
floor.

**Do not read the speedup column of that sweep.** The shape 6 CPU cache entry
expired on it, so shape 6 was re-measured at 15777.204 ms against 14639.960
ms before. The speedup moved from 27.89x to 32.24x on a CPU change, not on
anything this row did. Compare MLX ms against MLX ms.

### Accuracy, as the probe predicted

All 13 shapes PASS. `max_abs` moved from 1.19e-06..2.62e-06 to
1.07e-06..3.34e-06. The worst is shape 8 at 3.34e-06 against 2.62e-06, which
is 1.27x, and `profiling/ln_absorb_probe.py` predicted 1.0x to 1.2x before
the kernel existed. `atol` is 0.002, so shape 8 keeps 599x of headroom.

18 of 18 padding cases pass and stay bit exact. Six of them (`B4 S128 D128
H16`) reach 512 rows and do select this row, so the coverage is real. A
padded batch itself keeps the plain path, because the constants are built
against one value of `defer`.

## 45. Build the deferred bias `carry` at weight build time — OPEN

The article's optimization #1: stop recomputing a constant on every call.

`carry` is the deferred residual bias of row 36. It is a pure function of the
weights, and `_mlx_transformer()` rebuilds the running total on every call:

    carry = ob if carry is None else carry + ob      # L821
    carry = carry + layer["fob"].astype(mx.float32)  # L849

**Row 46 already delivered most of this row.** The weight build at L984-L1000
folds the carry into the `ln1c3` and `ln2c3` constants, so the fused path
reads it, never builds it.

What is left is small:

- the final LayerNorm, which has no GEMM below it and takes the carry through
  row 37's one `x = x + carry`;
- shape 2, which declines row 46 and runs the plain path.

**A sweep cannot score this row.** It is about 8 kernel launches on a
`(d_model,)` vector for the whole forward, at about 0.004 ms each
(`references/machine.md`). That is 0.1% FLOP-weighted against a 1% noise
floor. Do it for correctness of the code, not for a number, and do not
expect the scoreboard to move.

## 47. Produce the LayerNorm statistics in the epilogue of the GEMM that writes the activation — KEPT

Row 46 deleted the LayerNorm. This row deletes what row 46 left behind.

**What row 46 left.** The fused path runs the GEMM over raw `x` and applies
the norm in the epilogue, so no LayerNorm writes an activation any more. But
the epilogue needs the row mean and the row rstd, and `layer_norm_stats()`
made them in a separate pass that **read the whole activation** to write two
floats for each row.

**The prize, before the build.** `stage_roofline.py --shapes 1,6,8,13`,
30 August 2026, floor 0.3214 ms:

| shape | `ln1 stats` ms | `ln2 stats` ms | pair | layer | share |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.334 (raw 0.656) | 0.334 (raw 0.655) | 0.668 | 12.070 | **5.5%** |
| 8 | 0.069 (raw 0.391) | 0.099 (raw 0.420) | 0.168 | 30.180 | 0.6% |

Read the `raw` column at shape 8. Its `ms` values sit under the floor, so
they are not reproducible: an earlier run of the same build gave 0.260 and
0.275 ms for the same two stages.

Each shape 6 pass moved 65.0 MiB and reached **104 GB/s raw** against the
128 GB/s roof of `references/machine.md`. **The stage was at the memory roof,
so no better statistics kernel could win it.** Only fewer bytes could. That is
the same argument that killed a better LayerNorm kernel under row 43, one
level further in.

**The idea.** Every activation these passes read was written by a GEMM one
stage earlier:

| the stats pass | the GEMM that wrote its input |
|---|---|
| `ln1 stats` of layer i | `ffn_out` of layer i-1 |
| `ln2 stats` of layer i | `out proj` of layer i |

That GEMM holds the value in registers at the moment it stores it. So its
epilogue takes the statistics, and the 65 MiB read never happens.

### Step 0. The accuracy screen went first

Row 43 route 1 and row 44 both died AFTER the build. Row 46 lived because a
cheap screen went first. So this row screened first too, with
`profiling/ln_tiled_stats_probe.py`, which writes no Metal.

A threadgroup owns one `bn`-wide tile of the row, not the whole row. Shape 6
splits `d_model = 128` into 2 tiles at `bn = 64`, and shape 8 splits 1024
into 16. So the epilogue cannot centre against a mean it does not have.
`layer_norm_stats()` centres, and `fast_layernorm` refuses the uncentred
form `var = Q/D - mean^2` for exactly that reason: it cancels when the row
mean is large against the standard deviation, and a residual stream drifts.

The screen compared three forms against float64, on real activations from a
real forward:

    today   the whole row, centred
    naive   raw sum and raw sum of squares per tile, then Q/D - mean^2
    chan    each tile centres against its own mean, then Chan's formula

    .venv/bin/python3 profiling/ln_tiled_stats_probe.py --shape 6
    .venv/bin/python3 profiling/ln_tiled_stats_probe.py --shape 8
    .venv/bin/python3 profiling/ln_tiled_stats_probe.py --shape 13

| shape | drift `|mean|/std` | proj today | proj naive | proj chan |
|---:|---:|---:|---:|---:|
| 6 | 0.29 to 0.33 | 1.492e-06 | 1.492e-06 | 1.492e-06 |
| 8 | 0.12 to 0.13 | 4.623e-06 | 4.623e-06 | 4.861e-06 |
| 13 | 0.29 to 0.33 | 1.492e-06 | 1.492e-06 | 1.492e-06 |

**The drift is 0.12 to 0.33, which is far below the cancellation regime.**
A drift of 1 costs nothing and a drift of 1000 costs about 10 bits. The
LayerNorm at the top of every block keeps the residual stream centred, so the
term that cancels never grows. Both tiled forms hold today's error against a
2e-03 budget.

So the build took the **naive** form. It is one add and one multiply-add for
each element, where `chan` needs a tile mean, a centring pass and a combine.
`chan` is also slightly WORSE at shape 8, because 16 combine steps accumulate
their own rounding.

### Step 1. The second kill test: the GEMMs had to move first

`out proj` and `ffn_out` did not run through the hoisted GEMM. They ran
`mx.addmm`, because row 40 measured that `mx.addmm` is already optimal for a
projection with no epilogue. They also take the residual as a **matrix** C
operand, and `steel_addmm()` only took a `(N,)` bias.

The steel kernel already carries the matrix case: it reads C at
`c_row * ldc + c_col * fdc`, so a vector passes `ldc = 0` and a matrix passes
`ldc = N`. `steel_addmm()` now accepts both.

Measured before the epilogue was written, 100 repeats:

| case | M | K | N | `mx.addmm` | steel | ratio | max_abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| shape 6 `out proj` | 131072 | 128 | 128 | 1.7593 | 1.7434 | 1.009x | 0.00e+00 |
| shape 8 `out proj` | 8192 | 1024 | 1024 | 4.4438 | 4.4963 | 0.988x | 0.00e+00 |
| shape 13 `out proj` | 65536 | 128 | 128 | 0.9824 | 0.9456 | 1.039x | 0.00e+00 |

It ties, and `max_abs` is **0.00e+00**, which repeats row 40's proof that MLX
dispatches `addmm` to this same kernel with this same tile. So the move costs
nothing and the epilogue has somewhere to live.

### The kernel

`_ROW_STATS_EPILOGUE` in `steel_gemm.py` is a second new method on
`BlockMMA`, beside row 46's. It runs LAST, directly above the store, so it
reads the value the kernel is about to write.

**The lane reduction, and why it needs no threadgroup memory.** One fragment
row lives in four lanes. `BaseMMAFrag::get_coord` gives
`fm = (qid & 4) + ((lane / 2) % 4)` with `qid = lane / 4`, so the four lanes
of one row are `lane`, `lane ^ 1`, `lane ^ 8` and `lane ^ 9`. Two
`simd_shuffle_xor` steps therefore reduce a row, with no threadgroup memory
and no barrier. **Row 44 is why that matters**: it died because the
threadgroup memory its chain needed cost more than the DRAM round trip it
saved. This row spends none.

**The partials buffer.** Two simdgroups cover each row of the tile, one for
each `simd_group_id % WN`, so the buffer holds `WN * tiles_n` entries for
each row. `row_stats_reduce()` sums them and writes the same `[M, 2]` pair
that `layer_norm_stats()` writes, so row 46's epilogue takes it unchanged.

The layout is `[P][2][M]`, not `[M][P][2]`. The eight leader lanes of a
simdgroup then write eight adjacent floats, and neighbouring threads of the
reduce read neighbouring floats. A `[M][P][2]` layout scatters both.

The byte count, at the shape 6 chunk:

| | today | row 47 |
|---|---:|---:|
| statistics read | 65 MiB | 0 |
| partials write | — | 4 MiB |
| partials read | — | 4 MiB |
| result write | 1 MiB | 1 MiB |

**The deferred bias.** The statistics run over `x + carry`, which is what
`layer_norm_stats(pre_bias=)` computes. The epilogue adds `carry[n]` to the
value it accumulates and NOT to the value it stores, so row 36 still never
adds a bias to an activation. It costs one vector load for the tile.

### It is correct

`max_abs` of the GEMM output against `mx.addmm` is **0.00e+00** on every
case, so adding the epilogue changed no arithmetic in the GEMM. The
statistics agree with `layer_norm_stats()` to the level the screen predicted:

| case | M | K | N | P | out | rstd, relative | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| shape 6 `ffn_out` | 131072 | 128 | 128 | 4 | 0.00e+00 | 1.56e-07 | 8.94e-08 |
| shape 8 `out proj` | 8192 | 1024 | 1024 | 32 | 0.00e+00 | 2.32e-07 | 5.96e-08 |
| shape 13 `ffn_out` | 65536 | 128 | 128 | 4 | 0.00e+00 | 2.04e-07 | 8.94e-08 |

### What it is worth, at the stage

GEMM, then statistics, against GEMM with the epilogue, then reduce. 100
repeats:

| case | M | K | N | GEMM alone | today | row 47 | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| shape 6 | 131072 | 128 | 128 | 1.8101 | 2.4301 | 1.9154 | **1.269x** |
| shape 8 | 8192 | 1024 | 1024 | 4.4919 | 4.7661 | 4.5209 | 1.054x |
| shape 13 | 65536 | 128 | 128 | 1.0173 | 1.3006 | 1.0172 | **1.279x** |
| shape 1 | 8192 | 128 | 128 | 0.2687 | 0.2805 | 0.2673 | 1.049x |

Read the shape 6 row. The statistics cost **0.620 ms** as their own pass and
**0.049 ms** in the epilogue, and the epilogue costs the GEMM 0.056 ms
(1.8101 to 1.8660). So it moves the work to where the value already is, and
the work nearly disappears.

### Where it does not apply

- **`ln1` of layer 0.** No GEMM writes its input. It keeps
  `layer_norm_stats()`.
- **The last `ffn_out`.** The only LayerNorm below it is the final one, which
  is a plain LayerNorm with no GEMM under it. So it takes no statistics.
- Everything between them is free. At 4 layers that is 7 of the 8 passes.

### The sweep that kept it

    .venv/bin/python3 scoreboard.py --cpu-cache --label "Row 47: take the LayerNorm statistics in the GEMM epilogue"

Compare MLX ms against MLX ms. The CPU column came from the cache on 12 of
the 13 shapes, so the speedup column mixes two sweeps.

| shape | FLOP share | MLX before | MLX after | ratio | MPS control |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4% | 3.469 | 3.317 | 1.046x | 1.006x |
| 2 | 0.0% | 0.633 | 0.629 | 1.007x | 0.992x |
| 3 | 0.0% | 0.688 | 0.626 | **1.098x** | 1.017x |
| 4 | 0.1% | 1.168 | 1.166 | 1.002x | 0.996x |
| 5 | 0.9% | 6.532 | 6.044 | 1.081x | 1.005x |
| 6 | **66.5%** | 498.330 | 456.875 | **1.091x** | 0.970x |
| 7 | 0.0% | 1.110 | 1.019 | 1.089x | 1.024x |
| 8 | **21.3%** | 120.930 | 117.722 | 1.027x | 1.010x |
| 9 | 0.4% | 3.819 | 3.521 | 1.085x | 0.999x |
| 10 | 0.4% | 3.579 | 3.402 | 1.052x | 1.020x |
| 11 | 0.4% | 3.622 | 3.396 | 1.067x | 1.008x |
| 12 | 0.1% | 1.109 | 1.203 | 0.922x | 0.955x |
| 13 | 9.4% | 41.566 | 39.793 | 1.045x | 1.010x |
| **sum** | | **686.556** | **638.714** | **1.075x** | |

**The control.** MPS ran the same 13 shapes in both sweeps and held at a
1.006x median. Shape 6's own MPS moved **0.970x**, so the machine was if
anything slower during the second sweep, and the 1.091x on the shape that
carries 66.5% of the weight is conservative.

### Shape 12 did NOT regress. The sweep reading was machine drift

The sweep above reads 0.922x at shape 12. It is wrong, and an A/B says so.

The sweep compares two runs 4.5 hours apart. This A/B builds the same model
twice, once with `fuse_stats_out` and `fuse_stats_ffn` forced to None, and
alternates the order each round so neither side always runs cold:

| shape | rows | activation | row 47 OFF | row 47 ON | ratio |
|---:|---:|---:|---:|---:|---:|
| 3 | 512 | 0.25 MiB | 0.7062 | 0.6772 | 1.043x |
| 4 | 2048 | 1 MiB | 1.2904 | 1.2898 | 1.000x |
| 12 | 2048 | 1 MiB | 1.2313 | 1.2210 | **1.008x** |
| 1 | 8192 | 4 MiB | 3.5009 | 3.3556 | 1.043x |

A second run of shape 12 alone, 200 repeats, gave **1.0131x**, with the two
ranges not overlapping: OFF 1.2370 to 1.2465, ON 1.2224 to 1.2263.

Three things say the sweep reading was the machine:

1. Shape 12's own **MPS control moved 0.955x** across the same two sweeps.
   MPS runs none of this code, so that 4.5% is machine alone.
2. The A/B gives a small WIN twice, with non-overlapping ranges.
3. The earlier sweep read shape 12 at 1.109 ms. The A/B reads 1.22 to 1.23
   ms for **both** arms. The whole shape got about 10% slower between the
   two sweeps, whatever the plan.

**Why shape 12 barely moves either way.** This row deletes a DRAM read, so
the prize scales with the size of the activation, not with the row count:

| shape | rows | activation | gain |
|---:|---:|---:|---:|
| 12 | 2048 | 1 MiB | 1.008x |
| 4 | 2048 | 1 MiB | 1.000x |
| 1 | 8192 | 4 MiB | 1.043x |
| 6 | 131072 | **64 MiB** | 1.091x |

At 1 MiB the activation never leaves the cache, so there is no 65 MiB read
to delete. Shapes 4 and 12 hold the same row count and both sit flat, which
is the mechanism confirming itself. **No gate is needed**: the row costs
nothing where it wins nothing.

**The lesson for the next row.** A shape under about 2 ms cannot be scored
by comparing two sweeps. Its reading moves further with the machine than
with the code. Use an interleaved A/B on that shape, or read only the MPS
control and say the reading is inconclusive.

### Accuracy

All 13 shapes PASS, `max_abs` 1.19e-06 to 3.46e-06 against `atol = 0.002`.
The screen predicted no change and the sweep agrees: the range before this
row was 1.07e-06 to 3.34e-06.

`test_padding.py` gives 18 of 18 bit exact. The padded path declines this
row, exactly as it declines rows 36 and 46, because it ends each layer with
`mx.where(valid_tokens, x, 0)` and that does not commute with a deferred
bias.

### What it does not do

`profiling/stage_roofline.py` still times `ln1 stats` and `ln2 stats` as
their own stages. That tool builds a block from the plan, and it does not
model this epilogue, so its `ln1 stats` and `ln2 stats` rows now describe a
pass the model no longer runs on a shape that fuses. Read the two rows as
the prize this row took, not as current cost.

## 48. Hide the framework boundary behind the GPU — REVERTED

Built as a prototype, correct, and it loses. **Do not try it again, and do
not try any other form of it**: the reason is the memory system, not the
code.

### Why it looked worth doing

Row 23 measured the input copy at 17.4 ms of a 1678 ms shape 6 call and
called it 1.0%. Nobody re-read it after that. The kernels then got 3.7x
faster and the copy did not, so the same milliseconds are now 3.1%.

Measured again on 30 August 2026, with a breakdown of `forward()`:

| shape | FLOP share | `_to_mlx` | `mx.concatenate` | GPU loop | boundary |
|---:|---:|---:|---:|---:|---:|
| 6 | 66.5% | 13.951 | 10.292 | 422.4 | **5.4%** |
| 8 | 21.3% | 2.085 | — | 115.4 | 1.8% |
| 13 | 9.4% | 2.004 | — | 37.7 | **5.1%** |
| 1 | 0.4% | 0.241 | — | 3.3 | 6.8% |

That is about **4.5% FLOP-weighted spent outside the kernels**, which was
more than any row still OPEN. The GPU is idle for all of it.

### What was built

Shape 6 runs 10 chunks. `forward()` converts all 626 MiB before it queues
any GPU work, and `mx.eval(part)` inside the loop drains the GPU on every
pass. The prototype converts chunk `i+1` while the GPU runs chunk `i`:

    part = call(_to_mlx(x[start:stop]), ...)   # queue; returns at once
    if pending is not None:
        mx.eval(pending)                       # wait for the PREVIOUS chunk

`profiling/pipeline_probe.py` holds it, with six arms. Every arm is bit
equal to the model today.

### The measurement

| arm | ms | vs today | peak GiB |
|---|---:|---:|---:|
| today | 448.283 | 1.000x | 4.18 |
| convert in loop | 469.586 | 0.955x | 3.02 |
| **pipeline** | 460.268 | **0.974x** | 3.07 |
| loop alone (input preconverted) | 454.860 | 0.986x | 2.96 |
| loop + unrelated memcpy | 499.921 | 0.897x | 2.96 |
| loop, lagged eval, bulk convert | 446.647 | 1.004x | 2.96 |

### Why it loses. It is unified memory

The copy is not slower when it is split. Timed with NO GPU work at all,
625 MiB through `_to_mlx`:

| route | ms | GB/s |
|---|---:|---:|
| bulk, 1 call | 13.261 | 46.0 |
| 10 chunks | 13.976 | 43.7 |

So the loss is not the chunking. It is contention.

**The controlled test.** Arm `loop + memcpy` adds an UNRELATED 625 MiB CPU
memcpy to the chunk loop. It touches none of the model's data, so it can only
compete for bandwidth:

| run | memcpy alone | added to the loop | hidden |
|---|---:|---:|---:|
| 1 | 34.53 ms | +31.31 ms | 9% |
| 2 | 31.40 ms | **+45.06 ms** | **-43%** |

Run 2 is the whole answer: overlapping the copy with the GPU is **worse than
running it on its own**. There is no separate bus. The CPU and the GPU read
and write the same DRAM through the same controller, and the shape 6 block is
already memory bound — `stage_roofline.py` puts `out proj` at 105% and
`ffn_out` at 110% of the 128 GB/s roof. Every byte the CPU moves comes
straight out of the GPU's throughput.

**This is what makes the trick work on a discrete GPU and fail here.** There
the copy crosses PCIe while the GPU reads its own VRAM, so the two are
genuinely parallel. On an M3 Pro they are the same resource.

### The lagged eval is separately dead

Arm `loop, lagged eval` keeps the fast bulk conversion and only stops the GPU
queue from draining at each chunk boundary. It gives **1.004x**, inside the
1% noise floor. So `mx.eval(part)` inside the loop costs nothing, and it can
stay: it is what bounds the working set.

### What survives

Nothing for speed. One observation worth keeping: converting per chunk cuts
the peak from **4.18 GiB to 3.02 GiB**, because the whole 626 MiB input is
never live beside the chunk intermediates. Row 10 chose chunking for exactly
this reason. If a future shape runs out of the 12 GiB budget, this is the
lever, and it costs about 4.5% of shape 6.

### What it rules out

Every variant of "overlap the transfer with compute" on this machine:
double buffering, a copy stream, a background thread doing `_to_mlx`,
converting the next batch during the current one. They all move bytes through
the one memory system that the kernels are already saturating. The 4.5% is
real and it is not recoverable this way.


## 49. Block the head dimension of K and V, so `head_dim = 256` fits the threadgroup — REVERTED

Built as a prototype, correct on every block shape, and it loses on every
block shape. It closes the last open route to a fused attention kernel at
`head_dim = 256`, which is what shape 8 runs.

### The threadgroup limit is real, and it is 32 KiB

Rows 26 and 41 both assert a 32 KiB threadgroup. Neither measured it. It is
measured now, and they are right. `profiling/tg_limit.py` compiles a kernel
that declares a threadgroup array of a given size:

| asked | result |
|---:|---|
| 16 KiB | loads and runs |
| 32 KiB | loads and runs |
| 33 KiB | `[metal::Device] Unable to load kernel` |
| 40, 48, 64, 96 KiB | same failure |

MLX does not report the limit: `mx.device_info()` gives the device name, the
architecture and three memory sizes, and no threadgroup field. Metal reports
it as `MTLDevice.maxThreadgroupMemoryLength`, which no MLX Python binding
exposes. So the probe is how this project reads it. See
`references/machine.md`.

### What was built

Apple's kernel puts both operand buffers in the threadgroup at the full head
width, so a wide head does not fit:

    Q_smem  = BQ * (BD + 4)                      = 32.5 KiB at BQ 32, BD 256
    KV_smem = max((BK + 4) * BD, BK * (BD + 4))  = 36.0 KiB at BK 32, BD 256

Row 41 shrank `BK` to 8 to fit, and read the result as a handicap: "at
`BD=256` with `BK=8` the kernel runs 16 K block iterations ... The
threadgroup traffic per unit of work is what kills it."

This row removes that constraint instead of accepting it. It blocks the head
dimension of K and V, and keeps Q whole:

    S = Q @ K.T   sums over D, so it accumulates over D chunks.
    O[:, d] += P @ V[:, d]   is independent for each D chunk.

    for each K block:
        S = 0
        for each D chunk:  load K[:, chunk],  S += Q[:, chunk] @ K[:, chunk].T
        softmax(S) -> P            <- needs the whole row of S, and has it
        rescale O by the online factor
        for each D chunk:  load V[:, chunk],  O[:, chunk] += P @ V[:, chunk]

    Q_smem  = BQ * (BD + 4)                        = 16.25 KiB at BQ 16
    KV_smem = max((BK + 4) * BDC, BK * (BDC + 4))  =  9.00 KiB at BK 32, BDC 64

That is 25.25 KiB, and `BK` is free to be 32 or 64 again. The FLOPs and the
DRAM traffic do not change: it reads Q, K and V once each, as Apple's kernel
does. It costs two extra threadgroup barriers for each D chunk, and Q whole
in the threadgroup caps `BQ` at 16, so the threadgroup holds two simdgroups.

This is NOT the `d_outer` of `metal-flash-attention` that row 41 describes.
That one blocks the D axis of the O accumulator and spills O to device
memory on purpose. This one keeps O in registers and blocks the operands, so
there is no spill.

### The measurement

Shape 8 attention, B64 H4 S128 `head_dim` 256, causal, float32. Three
interleaved rounds of 40 repeats, median of rounds, each call bracketed by
`mx.eval` and `mx.synchronize`:

    .venv/bin/python3 profiling/d_outer_probe.py --repeats 40 --rounds 3

Every one of the 26 fitting block shapes is correct, `max_abs` 1.31e-06 to
1.55e-06 against `mx.fast.scaled_dot_product_attention`. The best of each
`bk`, against the 2.59 ms MLX fallback:

| bq | bk | bdc | ms | speedup |
|---:|---:|---:|---:|---:|
| 16 | 8 | 64 | 3.0709 | **0.832x** |
| 16 | 16 | 128 | 3.2325 | 0.800x |
| 8 | 8 | 64 | 3.3865 | 0.745x |
| 16 | 32 | 32 | 4.0708 | 0.640x |
| 16 | 64 | 32 | 4.5986 | 0.550x |

**Row 41 is still the best result at this head width, at 0.904x, and it does
not block D.**

### Why the premise was wrong

Row 41 read `BK = 8` as the handicap. It is not. The sweep holds `bq` fixed
and moves `bk`, and a larger `BK` is always slower:

| bq | bk8 | bk16 | bk32 | bk64 |
|---:|---:|---:|---:|---:|
| 16 | **0.832x** | 0.800x | 0.640x | 0.550x |
| 8 | **0.745x** | 0.654x | 0.495x | — |

So the thing this row was built to buy is a thing the kernel does not want.
The reason is in the roofline, and row 41 already recorded it without
drawing this conclusion: the stage runs at 40% of the bandwidth roof and 20%
of the matmul roof, so it is IO bound. `BK` sets the loop count and the
threadgroup traffic. It does not change the bytes that cross DRAM. Raising
it spends more threadgroup memory, which lowers occupancy on a kernel that
already holds only two simdgroups.

### What this closes

Rows 21, 26 and 41 leave one open route: reach a fused kernel at
`head_dim = 256`. Three approaches are now measured, and all three lose.

| approach | best | row |
|---|---:|---:|
| narrow `BK` to fit the full head | 0.904x | 41 |
| block the head dimension of K and V | 0.832x | 49 |
| `bk8` guess, never measured | — | 26 |

The prize was about 1.2% of the weighted score, from a `sdpa` stage of
2.651 ms against a 1.049 ms byte floor. **Do not try a fourth approach
without a new reason.** The stage is IO bound at `S = 128`, its score
matrix is 16 MiB and never reaches DRAM (`sdpa peak memory: 80.0 MiB`
against 128.0 MiB for the operands alone), so the prize row 25 won at shape
6 does not exist here.

## 50. Apply the final LayerNorm in the epilogue of the GEMM above it — KEPT

Row 46 folds a LayerNorm into the GEMM **below** it. The final LayerNorm has
no GEMM below it. Rows 37, 45, 46 and 47 each said so, and each moved on:

- row 37: "Only the FINAL LayerNorm still does, and it has no GEMM below it."
- row 45: "the final LayerNorm, which has no GEMM below it".
- row 47: "The last `ffn_out`. The only LayerNorm below it is the final one,
  which is a plain LayerNorm with no GEMM under it. So it takes no
  statistics."

**None of them measured what it costs.** Measured now, at the shape 6 chunk
width (M = 131072, d_model = 128):

    .venv/bin/python3 profiling/final_ln_probe.py

| what | time | rate |
|---|---:|---:|
| `fast_layernorm.layer_norm(x, w, b, eps, pre_bias=carry)` | 1.2478 ms | 100.2 GB/s |

That is 12.5 ms of the 452 ms shape 6, or **2.8% of the shape**. It runs at
the memory roof, so no better LayerNorm kernel can win it. Only fewer bytes
can, and row 46's own argument says how: the value was in registers one stage
earlier.

### The GEMM above, not the GEMM below

The activation the final LayerNorm reads is written by the `ffn_out` of the
**last layer**. That GEMM holds the value in its accumulator at the store. So
the LayerNorm runs there, and the activation never makes a second round trip.

### Why nobody could do this before

To centre a row, the tile must own the **whole** row. Row 47 could take only
the raw sums for exactly this reason: a threadgroup owns a `bn` wide piece.

So this epilogue needs a **full row tile**: `bn == N`, and `wn == 1` as well,
so that one simdgroup owns the row and the two `simd_shuffle_xor` steps of
row 47 reduce all of it. `simd_shuffle_xor` is a butterfly, so all four lanes
of a fragment row end with the total. No leader lane, no threadgroup memory
and no barrier.

A full row tile looked expensive, and the `_TILES` comment said so: `bm64
bn128` measured 1.907 ms against 1.527 ms for `bm32 bn64`, a 1.25x loss.
**That measurement is on `ffn_in`**, which has a vector bias and a GELU and is
compute bound. `ffn_out` takes the residual as a **matrix C**, which makes it
IO bound, and there the tile shape does not matter. Measured at the same M, K
and N with a matrix C, 40 repeats:

| tile | time |
|---|---:|
| bm32 bn64 | 1.8055 ms |
| bm32 bn128 | 1.8074 ms |
| bm64 bn128 | 1.8048 ms |

The comment in `steel_gemm.py` now carries both measurements.

### The kernel

`_FINAL_LN_EPILOGUE` is a third method on `BlockMMA`, beside row 46's
`apply_layer_norm_epilogue` and row 47's `write_row_stats`. It reuses the
pointer slots of those two rows, which `ffn_out` never uses: `lnc1` carries
the LayerNorm gain, `lnc2` the LayerNorm bias, and `rowcarry` the deferred
residual bias of row 36. So the kernel needs no new argument.

It makes two passes over the accumulator: one to sum, one to apply. The
variance is the UNCENTRED form, as row 47 uses, and
`profiling/ln_tiled_stats_probe.py` measures that this model stays far below
the cancellation regime.

`wm2` does not compile at `bn = 128`: 64 threads cannot load that threadgroup
tile. `bm32 wm4` is the best of the six configurations that do build, at every
size measured.

### The GEMM alone

`profiling/final_ln_probe.py`, interleaved, median of 60:

| shape | M | N | today | fused | ratio |
|---|---:|---:|---:|---:|---:|
| 6, one chunk | 131072 | 128 | 2.8549 ms | 2.0411 ms | **1.399x** |
| 13 | 65536 | 128 | 1.5226 ms | 1.0787 ms | **1.412x** |
| 5 | 16384 | 128 | 0.5004 ms | 0.4139 ms | 1.209x |
| 7 | 8192 | 32 | 0.1868 ms | 0.1661 ms | 1.124x |
| 1 | 8192 | 128 | 0.3252 ms | 0.3024 ms | 1.075x |
| 12 | 2048 | 128 | 0.1815 ms | 0.1731 ms | 1.049x |
| 8 | 8192 | 1024 | — | — | no full row tile fits |

`max_abs` against the two kernel form is 1.43e-06 to 1.91e-06 on every row.

**Measure this interleaved.** A first version of the probe timed each side in
its own block and ran the big case first. It read **0.534x** at M = 8192 where
the interleaved A/B reads 1.075x. The allocator state that the big case left
behind moved a 0.3 ms reading by 2x. The file now alternates the order each
round.

### Where it does not apply

- **Shape 8.** `bn = N = 1024` puts `1024 * (16 + 4)` floats in the
  threadgroup, which is 80 KiB against the 32 KiB limit.
  `choose_final_ln_tile()` returns None, and shape 8 keeps the separate final
  LayerNorm. Shape 8 is 21.3% of the FLOP weight, so about a fifth of the
  score cannot take this row.
- **A padded batch.** The block clears the padded rows between the GEMM and
  the LayerNorm, so the epilogue would normalize the wrong value.
- **float16 and bfloat16.** The hoisted kernel is compiled for float32.

It does NOT need `defer_bias`. When the block defers, `carry` holds the whole
accumulated bias. When it does not, the projection bias `fob` is the whole
carry, and the epilogue adds it the same way.

### The controlled A/B

`plan_ab.py` builds the same model twice in one process and toggles
`final_ln`, alternating the order each round:

    .venv/bin/python3 profiling/plan_ab.py --cases 6 --repeats 8 --rounds 5
    .venv/bin/python3 profiling/plan_ab.py --cases 13,8,5,1,12 --repeats 30 --rounds 5
    .venv/bin/python3 profiling/plan_ab.py --cases 2,3,7,9,10,11,4 --repeats 60 --rounds 5

| shape | FLOP share | OFF | ON | ratio |
|---:|---:|---:|---:|---:|
| 6 | 66.5% | 442.4843 ms | 433.3615 ms | **1.0211x** |
| 8 | 21.3% | 116.4064 ms | 117.4550 ms | 0.9911x (null control) |
| 13 | 9.4% | 38.0368 ms | 37.5969 ms | 1.0117x |
| 5 | 0.9% | 6.1024 ms | 5.9947 ms | 1.0180x |
| 1 | 0.4% | 3.3741 ms | 3.3315 ms | 1.0128x |
| 9 | 0.4% | 3.5631 ms | 3.5159 ms | 1.0134x |
| 10 | 0.4% | 3.4499 ms | 3.4018 ms | 1.0142x |
| 11 | 0.4% | 3.4855 ms | 3.3789 ms | 1.0315x |
| 4 | 0.1% | 1.1989 ms | 1.1875 ms | 1.0095x |
| 12 | 0.1% | 1.2476 ms | 1.2331 ms | 1.0117x |
| 7 | 0.0% | 1.1080 ms | 1.0869 ms | 1.0195x |
| 3 | 0.0% | 0.7279 ms | 0.7283 ms | 0.9994x |
| 2 | 0.0% | 0.6400 ms | 0.6093 ms | 1.0505x |

**FLOP-weighted: 1.019x.**

### Repeated, to separate it from the noise

The A/B above ran once. **Every one of the 13 shapes then ran it twice more**,
on separate invocations:

    .venv/bin/python3 profiling/plan_ab.py --cases 6,8,13 --repeats 8 --rounds 7
    .venv/bin/python3 profiling/plan_ab.py --cases 1,2,3,4,5,7,9,10,11,12 --repeats 60 --rounds 7

| shape | weight | run 1 | run 2 | run 3 | median | worst | spread |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 66.5% | 1.0211x | 1.0210x | 1.0202x | **1.0210x** | 1.0202x | **0.09 pp** |
| 8 | 21.3% | 0.9911x | 1.0027x | 0.9994x | 0.9994x | 0.9911x | 1.16 pp |
| 13 | 9.4% | 1.0117x | 1.0080x | 1.0085x | 1.0085x | 1.0080x | 0.37 pp |
| 5 | 0.9% | 1.0180x | 1.0158x | 1.0187x | 1.0180x | 1.0158x | 0.29 pp |
| 1 | 0.4% | 1.0128x | 1.0006x | 1.0151x | 1.0128x | 1.0006x | 1.45 pp |
| 9 | 0.4% | 1.0134x | 1.0104x | 1.0118x | 1.0118x | 1.0104x | 0.30 pp |
| 10 | 0.4% | 1.0142x | 1.0134x | 1.0104x | 1.0134x | 1.0104x | 0.38 pp |
| 11 | 0.4% | 1.0315x | 1.0126x | 1.0132x | 1.0132x | 1.0126x | 1.89 pp |
| 4 | 0.1% | 1.0095x | 1.0097x | 1.0102x | 1.0097x | 1.0095x | 0.07 pp |
| 12 | 0.1% | 1.0117x | 1.0070x | 1.0123x | 1.0117x | 1.0070x | 0.53 pp |
| 7 | 0.0% | 1.0195x | 1.0260x | 1.0205x | 1.0205x | 1.0195x | 0.65 pp |
| 3 | 0.0% | 0.9994x | 1.0034x | 1.0168x | 1.0034x | 0.9994x | 1.74 pp |
| 2 | 0.0% | 1.0505x | 1.0154x | 1.0335x | 1.0335x | 1.0154x | 3.51 pp |

**Every one of the 12 shapes that takes the row has a median above 1.010x.**

**Only 3 of the 39 readings fall below 1.000x, and two of them are the null
control.** The third is shape 3 at 0.9994x, which is 0.06% on a 0.73 ms shape.
No shape reads a loss in more than one of its three runs.

**Shape 6 repeats to within 0.09 percentage points**, while the machine itself
moved: its OFF median went 442.48, 454.86 and 450.68 ms over the three runs, a
2.8% drift. The ratio does not follow the drift.

**Shape 8 is the null control and it spreads 1.16 pp around 1.000x.** It takes
no tile, so both sides run identical code. Its spread is 13x that of shape 6,
which is the point: the interleaved A/B measures the change, not the machine,
and the win on shape 6 is an order of magnitude steadier than the noise on a
shape that has no win to measure.

The FLOP-weighted figure barely moves, whichever of the three runs you take:

| pick | FLOP-weighted |
|---|---:|
| per-shape BEST of three | 1.0194x |
| per-shape MEDIAN of three | **1.0192x** |
| per-shape WORST of three | 1.0185x |

**The row does not depend on a single reading.** Take the worst of three runs
on every shape at once and it is still 1.0185x.

Two things make this reading trustworthy:

- **Shape 8 is a null control.** It takes no tile, so `final_ln=None` on both
  sides and the two runs execute identical code. It reads 0.9911x, which is
  the noise floor of this A/B: about 0.9%.
- **Shape 6 does not overlap.** OFF spans 441.06 to 449.60 ms and ON spans
  431.91 to 436.36 ms. The lowest OFF reading is above the highest ON reading.

No shape loses. The lowest is shape 3 at 0.9994x, which is flat.

### The sweep

    .venv/bin/python3 scoreboard.py --cpu-cache --label "Row 50: apply the final LayerNorm in the last ffn_out epilogue"

MLX 635.2 ms to 628.8 ms over the 13 shapes, **1.011x FLOP-weighted**, with
MPS at a **1.000x median** across the two sweeps. The sweep and the A/B
disagree by 0.8 percentage points, and the A/B is the better number: shape 6
carries two thirds of the weight, and a 5 ms move on a 452 ms shape is inside
what two sweeps minutes apart can produce. Both agree on the sign.

### Accuracy

All 13 shapes PASS, `max_abs` 1.43e-06 to 2.38e-06, which is the band the
model already carried. `test_padding.py` gives 18/18 PASS, and every case is
bit exact, because a padded batch declines this row.

### What is left after it

Shape 8 keeps its final LayerNorm, and it is 21.3% of the FLOP weight. To
reach it the epilogue would need a row wider than one threadgroup tile, which
is the same obstacle row 47 met and answered with a two-pass reduce. A
two-pass form here would write the un-normalized activation, read it back,
and normalize it, which is what the model does today.

## 51. Refuse a steel tile whose block loader has inexact thread geometry — KEPT (a bug fix)

Found while building row 50. It is not an optimization. It stops a wrong
answer.

### What MLX does

`steel/gemm/loader.h` gives `BlockLoader` three DEFAULT TEMPLATE ARGUMENTS,
and every one of them is a truncating integer division with no guard:

    n_reads = (BCOLS * BROWS) / tgp_size
    TCOLS   = BCOLS / n_reads
    TROWS   = tgp_size / TCOLS

Then `bi = thread_idx / TCOLS`, and the load loop steps `i` from 0 to `BROWS`
by `TROWS`. The loader is correct only when the three divisions are exact and
`TROWS` divides `BROWS`.

Nothing in MLX checks this, and MLX does not need to: it instantiates only the
tiles in its own dispatch table, and those all have exact geometry. This
module compiles the same kernel at tiles MLX never ships, so it must check.

### What goes wrong

At `BN = 96`, `BK = 16` and 128 threads:

    n_reads = (16 * 96) / 128 = 12
    TCOLS   = 16 / 12        = 1     <- truncated from 1.33
    TROWS   = 128 / 1        = 128

The 128 threads then load 128 rows of a 96 row tile. The loader reads 32 rows
past the operand and writes past the threadgroup buffer. No error is raised.

At `BN = 160` the truncation gives `TCOLS = 0`, and the kernel fails to
compile with "division by zero" in `utils.h`.

### The measurement

A plain GEMM, no epilogue, M=1024 K=128, `bm32 bk16`, against `mx.addmm`:

| BN | `wm4 wn1` | `wm2 wn2` |
|---:|---:|---:|
| 32 | 0.00e+00 | 0.00e+00 |
| 48 | **5.22e+00** | **5.04e+00** |
| 64 | 0.00e+00 | 0.00e+00 |
| 96 | **5.21e+00** | **5.29e+00** |
| 128 | 0.00e+00 | 0.00e+00 |

The error does not depend on `wn`, and it does not depend on the epilogue. It
is the loader.

`loader_geometry_ok()` repeats the three divisions and demands that each one
is exact. It reproduces the measured set exactly: over `BN` from 8 to 296 in
steps of 8, at `bm32 bk16 wm4 wn1`, it admits {8, 16, 32, 64, 128} and rejects
every other value, including all of 24, 40, 48, 56, 72, 80, 88, 96, 104, 112
and 120, which measured wrong, and 160, which does not compile.

### Where it applies

`choose_tile()` and `choose_final_ln_tile()` both skip a tile that fails it,
and `steel_addmm()` raises rather than return a wrong answer.

**It changes no appendix plan.** All four tiles in `_TILES` pass, for both
`transpose_b` settings, so every one of the 13 shapes keeps the plan it had.
That was checked by comparing `plan_kernels(...).describe()` against the plan
recorded in the sweep, for all 13.

### Why it was latent

`_TILES` holds `bn` 32 and 64 only, and `choose_tile()` requires `n % bn == 0`.
So no shape could reach a bad `bn` through it. Row 50 is the first caller that
sets `bn = N` for an arbitrary N, and `d_model = 96` reached the bug at once:
`max_abs` 9.37e-01 against the baseline, where the same model with row 50
off gives 1.19e-06.

### The regression test

84 configurations, over `d_model` in {32, 48, 64, 96, 128, 192, 256},
`seq_len` in {32, 96, 128}, batch in {8, 40} and both causal settings, against
`BaselineTransformer`. All 84 PASS, `max_abs` 9.54e-07 to 1.43e-06.

## 52. Close the kernel launch gaps with a persistent kernel — RULED OUT

Every earlier estimate of GPU idle came from arithmetic. Row 48 subtracted the
`_to_mlx` copy and the `mx.concatenate` from the wall time and called the
remainder the boundary. A residual like that carries the error of every term
it subtracts, so it bounds the idle but it does not locate it.

This row reads the gaps off the Metal timeline instead.

### The tool

`profiling/gpu_timeline.py` records a `Metal System Trace` and reads it with
`xctrace export`, so no Xcode window is needed.

    ./profiling/gpu_timeline.sh --case 6 --iterations 5 --warmup 3
    .venv/bin/python3 profiling/gpu_timeline.py report \
        profiling/traces/gpu_timeline.trace

Step 1 drives `UserOptimizedTransformer` alone and puts one `mlx-forward`
signpost around each forward pass. Step 2 reads three tables out of the trace:

| Table | What it gives |
|---|---|
| `os-signpost` | the start and the stop of each `mlx-forward` window |
| `metal-gpu-intervals` | one row for each command encoder the GPU ran, with a start and a duration in ns, for every process on the device |
| `metal-gpu-state-intervals` | the Active and Idle state of the device itself |

The report clips the encoder rows of the python process to each window, joins
the overlaps, and splits the idle time three ways. The **head** is the time
before the first kernel. The **tail** is the wait after the last one. The
**inner** gaps are the only part that a kernel change can win.

### The measurement

Shape 6, 5 forward passes, 3 warmup passes, on a quiet machine.

```
  #  window ms   busy ms  head ms  inner ms  tail ms  inner %  gaps  active %
  1    609.117   460.404  141.965     6.408    0.340     1.05   103     76.79
  2    484.211   461.757   15.915     5.797    0.742     1.20   101     95.67
  3    508.963   461.955   41.268     5.485    0.255     1.08   100     91.32
  4    511.173   439.050   67.134     4.652    0.337     0.91    94     86.49
  5    462.265   440.729   16.607     4.679    0.250     1.01   101     95.71
```

**The inner idle is 1.05% and it does not move.** Five windows give 1.05,
1.20, 1.08, 0.91 and 1.01 percent, while the head varies by a factor of nine.

### Where the inner idle sits

The inner gaps are not spread over the 100 encoders. They are 9 or 10 large
gaps and about 90 tiny ones.

| Window | encoders | gaps over 0.3 ms | they hold | all inner gaps |
|---:|---:|---:|---:|---:|
| 1 | 104 | 10 | 5.251 ms | 6.408 ms |
| 2 | 102 | 10 | 5.088 ms | 5.797 ms |
| 3 | 101 | 9 | 4.553 ms | 5.485 ms |
| 4 | 95 | 10 | 4.338 ms | 4.652 ms |
| 5 | 102 | 9 | 3.601 ms | 4.679 ms |

**Shape 6 runs exactly 10 chunks.** `CHUNK_ACTIVATION_BYTES` picks
`chunk = 1024` and the batch is 10000, so `forward()` loops 10 times and calls
`mx.eval(part)` at the end of each one. The count of large gaps matches the
count of chunks, window for window. So those gaps are the chunk boundary.
They are the `mx.eval` round trip, and the loop needs it: it is what keeps
one chunk of intermediates live instead of ten.

Everything that is left is the launch cost between two encoders inside one
chunk. **90 gaps under 0.05 ms hold 0.269 ms of a 462 ms window, which is
0.06%.**

### What this rules out

A persistent kernel keeps a layer resident so that consecutive kernels do not
give the GPU back. Its whole prize here is 0.06%. That is 17 times under the
1% noise floor of a sweep, so no build can even be measured, let alone won.

The same number rules out every other member of the class: a kernel graph, a
wider `mx.compile` region, a manual command buffer, an encoder merge. They all
attack the same 0.269 ms.

### The head gap is not this row's target

The head is 15.9 ms in the two clean windows, which agrees with the 13.951 ms
that row 48 measured for `_to_mlx` at shape 6. It is a CPU copy of the 655 MiB
input, and the GPU has nothing to run while it happens. **Row 48 already tried
to hide it and lost at 0.974x**, because unified memory makes the copy and the
kernels contend for one memory system.

Windows 1, 3 and 4 read 141.965, 41.268 and 67.134 ms at the head. That is not
the steady state. It is first touch on freshly mapped pages, and it decays over
the run. Take the head from window 2 or window 5, and treat a single trace as
one reading.

### What it costs to reproduce

28 s of wall time for the recording, and about 60 s for the report, which
runs `xctrace export` three times. The trace is written to
`profiling/traces/gpu_timeline.trace`.

## 53. Fuse the attention and the out projection into one kernel — RULED OUT

Not built. The budget decides it, and the budget takes minutes.

    .venv/bin/python3 profiling/attn_out_budget.py

### The shape of the kernel

The attention writes `context` to DRAM and `attn_out` reads it straight back.
A fused kernel keeps `context` in threadgroup memory and projects it there:

    phase 1   for each head: O_h = softmax(Q K^T) V   -> threadgroup
    phase 2   out = context @ W_o + b_o               -> device

The out projection mixes every head, because `context` is the concatenation of
all of them. **So one threadgroup must own `bq` query rows over the full
`d_model`**, and it must run every head for those rows itself.

### What one threadgroup must hold

| Buffer | Why it is there | Shape 6 |
|---|---|---:|
| `O_tgp`, `bq x (d_model + pad)` | the concatenated output. `BlockMMA::mma()` reads its A operand from threadgroup memory, as row 44 checked | **16.50 KiB** |
| `Q_smem`, `bq x (head_dim + pad)` | phase 1, one head at a time | 4.50 KiB |
| `KV_smem` | phase 1, one head at a time | 4.50 KiB |
| `Bs_o`, `bk_o x (d_model + pad)` | phase 2. `bn_o = d_model`, because the row is whole and there is one N tile | 4.12 KiB at `bk_o = 8` |

Phase 2 reads neither `Q_smem` nor `KV_smem`, so `Bs_o` can alias them. Metal
does not alias two declarations, so that needs one flat buffer, manual offsets
and a barrier between the phases.

| | shape 6 | Metal gives |
|---|---:|---:|
| aliased peak | **25.50 KiB** | 32.0 KiB |
| plain sum | **29.62 KiB** | 32.0 KiB |
| the steel attention kernel today | **9.00 KiB** | |

### Why that is a stop

Row 44 built a kernel with the same structure, on the same machine, and swept
it. Its controlled pair changed nothing but the threadgroup size:

| threadgroup | best ratio |
|---:|---:|
| 24.0 KiB | 0.996x |
| 29.0 KiB | 0.896x |

**25.50 KiB sits inside that band**, and 29.62 KiB is row 44's own losing
point. The aliasing is the only thing that keeps the budget under 26 KiB, and
it buys 4.12 KiB for a barrier and a hand-offset buffer.

The mechanism row 44 named is occupancy: threadgroup memory limits how many
threadgroups stay resident on a core. This kernel asks for **2.8x** what the
attention kernel uses today.

Read the interpolated 0.966x as an indication and nothing more. Row 44 holds
two points, and they come from a GEMM chain, not from attention. The facts that
do not need interpolation are the three numbers above: 25.50, 29.62 and 9.00.

### The prize is the same prize row 44 lost

At the shape 6 chunk, `context` is 131072 rows x 128 floats = 64 MiB, so the
round trip is **128 MiB**, or 1.119 ms of a 12.0 ms layer at 119.9 GB/s. That
is 9.3%.

Row 44's `hidden` round trip at the same chunk is **also 128 MiB**, also about
1.1 ms of the same 12.0 ms layer. So this row would spend more threadgroup
memory than row 44 did, to win the same number row 44 could not keep.

### It does not even reach the whole model

| Shape | aliased peak | fits |
|---:|---:|---|
| 1-6, 12, 13 (`d_model` 128, 4 heads) | 25.50 KiB | yes |
| 7 (`d_model` 32) | 7.50 KiB | yes |
| 11 (16 heads, `head_dim` 8) | 20.62 KiB | yes |
| 10 (2 heads, `head_dim` 64) | 34.00 KiB | **no** |
| 9 (1 head, `head_dim` 128) | 51.00 KiB | **no** |
| 8 (`d_model` 1024) | 197.00 KiB | **no** |

Shape 8 carries 21.3% of the FLOP weight and cannot take it. A wide head makes
both `KV_smem` and `O_tgp` grow, so the fusion is worst exactly where the
activation is largest.

### What would reopen it

A way to project the attention output without holding a whole row. There is
none: the projection is a contraction over `d_model`, and `d_model` is the head
concatenation. Splitting it needs a partial sum in DRAM, which is the round
trip again.

## 54. Re-sweep the steel GEMM tile, with the row 46 and row 47 epilogues on — REVERTED

`_TILES` in `steel_gemm.py` was ordered by a sweep of a PLAIN GEMM. Two
epilogues arrived after it, and neither row re-swept the tile:

- **Row 46** puts the LayerNorm in the epilogue of `qkv proj` and `ffn_in`.
  It reads two floats for the row and two `(N,)` vectors.
- **Row 47** puts the row statistics in the epilogue of `out proj` and
  `ffn_out`. It writes `wn * (N / bn)` partial planes.

Row 47 gives a clear reason to expect `bn = 128` to win: at N = 128 a
`bn64 wn2` tile writes 4 partial planes and a `bn128 wn2` tile writes 2.

    .venv/bin/python3 profiling/tile_resweep.py --grid full --stages ffn_in
    .venv/bin/python3 profiling/tile_resweep.py --grid coarse

### The method

Every stage runs at the shape 6 chunk: M = 131072, K = 128. The grid is
`bm` in {16, 32, 64}, `bn` in {32, 64, 128}, `bk` in {8, 16, 32} and
`(wm, wn)` in {2x2, 4x1, 1x4, 4x2, 2x4}, filtered by divisibility,
`fits_threadgroup()` and row 51's `loader_geometry_ok()`. That leaves 129
tiles for each stage.

**A plain sweep cannot score these tiles.** The first attempt read
`64x32x16x2x2` at 1.172x on `ffn_in`, and a paired run put the same tile at
1.008x. The machine drifted inside one process: an early reading of the tile
in use gave 2.0771 ms and a later one gave 1.9488 ms, which is 6.6%. So the
script runs BOTH tiles on every repeat and swaps the order on every other
one, exactly as `plan_ab.py` does for a plan field.

### The result

| Stage | epilogue | best of 129 | today |
|---|---|---|---|
| `qkv proj` N=384 | row 46 | `32x64x16x1x4` **0.997x** | is the best |
| `ffn_in` N=128 | row 46 | `32x64x16x1x4` **1.007x** | 0.7% behind |
| `out proj` N=128 | row 47 | `32x64x32x2x2` **1.012x to 1.025x** | 1 to 2% behind |
| `ffn_out` N=128 | row 47 | `32x64x32x2x2` **0.979x to 1.030x** | tied |

**The `bn = 128` hypothesis is refuted.** A `bn128` tile reached the top of no
stage. The one candidate that repeats on the row 47 stages is `bk = 32`, not
`bn = 128`, and it does not clear the floor.

### The noise floor of this A/B

Today's tile is in the grid, so it pairs against itself and gives a null
control. Over the runs it read **0.976x, 0.981x, 0.982x, 0.987x, 0.989x,
0.991x, 0.997x, 1.007x and 1.009x**. So the floor is about **1.5%**, and only
an effect above that is real.

By that floor:

- `32x64x32x2x2` on `out proj` sits at the top of the band, not above it.
- `32x64x32x2x2` on `ffn_out` sits inside it. One run read 1.030x and the next
  read 0.979x.
- `64x32x16x4x1` on `ffn_in` read **1.127x** once and then 0.976x, 0.997x and
  0.993x at 60 paired repeats. The 1.127x was its partner's slow reading, not
  its own fast one. **Do not trust a single pair.**

### The one real effect, and why it is not worth taking

`32x64x16x1x4` on `ffn_in` read **1.007x, 1.008x and 1.008x** over three runs
of 60 paired repeats. A 0.1 pp spread over three runs is not noise. It is also
0.7% of a stage that is about 15% of the shape 6 layer, so it is **0.1% of
shape 6 and 0.07% FLOP-weighted**. A sweep cannot see it, so it cannot be
confirmed at the model level.

`_TILES` is one ordered list, and all four consumers and all 13 shapes read
it. Reordering it to take 0.07% moves every shape. Giving the row 47 stages
their own list to take 0.2% adds a second chooser. Neither buys a number a
sweep can measure.

**So `_TILES` does not change.** The value of this row is the negative: the
tile is not where the remaining time is.

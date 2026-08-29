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
| 1 | MLX behind the torch interface | **KEPT** | `_mlx_transformer()` L349, `UserOptimizedTransformer` L433 | every shape | 4.4x to 7.4x against torch CPU. float32 PASS |
| 2 | `mx.compile` on the forward pass | **KEPT** | `make_call()` L535 | every shape | 14.203 ms to 13.401 ms (6%) |
| 3 | float32 LayerNorm accumulation | **KEPT** | `norm()` L380, weight build L474 | every shape | no measured gain. Correct policy. Free |
| 4 | Explicit float32 softmax, not fused | **REVERTED** | — | — | float16 49 to 50 failures. No gain. More code |
| 5 | `torch.compile` over the MLX class | **RULED OUT** | — | — | crash. Dynamo cannot trace an MLX object |
| 6 | Mask form selected by the input | **KEPT** | `_mlx_transformer()` L391 | seq_len >= 512, no padding | 3.9% at seq 512. 1.7% at seq 2048. none at seq 128 |
| 7 | Shape-aware kernel plan (`KernelPlan`) | **KEPT** | `plan_kernels()` L272 | every shape. It selects 8, 9 and 10 | 1.57x at shape 13. 1.31x at shape 6 |
| 8 | Blocked causal attention | **KEPT** | `_attention()` L371 | causal, effective `head_dim < 64`, `seq_len > 64`, `batch * heads >= 64` | 1.63x at seq 1024. Bit exact. It is a workaround for the FALLBACK kernel: the fused kernel skips the triangle itself, at 0.55x. See row 18 |
| 9 | Fused `[D, 3D]` QKV matmul | **KEPT** | weight build L524, use L411 | every shape. Always on | 1.99x at batch 1. 1.00x at d_model 1024 |
| 10 | Batch chunking, full depth for each chunk | **KEPT** | `forward()` L568 | one activation over 64 MiB. Shape 6 only | peak 9.16 GiB to 2.68 GiB, and 1.05x faster |
| 11 | Pad `head_dim` from 8 up to 32 | **REVERTED** | — | — | 0.91x at shape 7. Blocking beats it. Wrong target: 32 is still the fallback kernel. See row 17 |
| 12 | Full port of the file to MLX | **RULED OUT** | — | — | out of scope. The baseline does not change |
| 13 | float16 and bfloat16 accuracy | **RULED OUT** | — | — | no implementation can pass. torch `F.sdpa` fails too |
| 14 | Hand-written Metal kernel for `head_dim = 8` | **OPEN** | — | shapes 7 and 11 | not measured. `mx.fast.sdpa` falls back to a materializing kernel below `head_dim = 64`. Padding cannot rescue these two: 0.894x and 0.756x. See row 17 |
| 17 | Pad `head_dim` to 64 to reach the fused SDPA kernel | **KEPT** | `plan_kernels()` L339 | `head_dim >= 32`, `head_dim < 64` and `seq_len >= 4 * d_model`. Shape 13 only | **1.646x at shape 13**. A blanket `head_dim < 64` rule gives 0.983x FLOP-weighted and loses |
| 18 | MLX dispatches SDPA to two different kernels | **KEPT** (a fact, not a change) | `SDPA_FUSED_MIN_HEAD_DIM` L268 | `head_dim` 64..128 gets the fused kernel. Everything else materializes `B x H x S x S` | peak memory at B8 H8 S2048: 128 MiB fused against 1048 MiB fallback |
| 19 | Sweep `seq_len` to place the `S >= 4*D` threshold | **OPEN** | — | `head_dim` 32..48 | not measured. Row 17 rests on one point |
| 15 | Fused FFN | **OPEN** | — | every shape | not measured. Check `mx.compile` first |
| 16 | Quantization by `mx.quantize` | **OPEN** | — | every shape | not measured. It will fail the accuracy test |

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

## Cost of the framework boundary

The two conversions sit inside the timed region. I measured them:

| Part | Time | Share |
|---|---|---|
| Full `forward()` | 13.422 ms | 100% |
| MLX calculation | 13.256 ms | 98.8% |
| torch to MLX (input) | 0.027 ms | 0.2% |
| MLX to torch (output) | 0.078 ms | 0.6% |

The boundary costs 0.8%. It is small at this size. It becomes important for
a much smaller model.

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

# MLX tensor operations: which kernel at which shape

Measured on this machine. See [machine.md](machine.md) for the limits and the
timing rules, and [test-shapes.md](test-shapes.md) for the shapes.

The code that applies these results is `plan_kernels()` in
`torch_transformer_benchmark.py`. Change a threshold there, not here. Record
the new measurement here.

## The short version

| Condition | Use |
|---|---|
| `head_dim` 64..128 | one full `mx.fast.scaled_dot_product_attention` call |
| `head_dim <= 16`, causal, `S > 64`, `B*H >= 64` | blocked causal, block 32 |
| `head_dim` 17..63, causal, `S > 64`, `B*H >= 64` | blocked causal, block 64 |
| `head_dim` 32..48 and `S >= 4*D` | pad `head_dim` to 64, then one full call |
| always | one fused `[D, 3D]` QKV matmul |
| activation over 64 MiB | chunk the batch, full depth per chunk |

## 0. MLX has two SDPA kernels, and the shape picks one

**This is the most useful fact in this file. Read it before section 1.**

`mx.fast.scaled_dot_product_attention` dispatches to a fused flash kernel
**only for `head_dim` 64 to 128**. Outside that range it accepts the call,
returns a correct answer, and uses a fallback that materializes the whole
`B x H x S x S` score matrix.

Measured by peak GPU memory at B=8, H=8, S=2048, where the score matrix
would be 1024 MiB. The column is `peak - base`, in MiB:

| head_dim | 8 | 16 | 32 | 48 | 64 | 72 | 96 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| peak MiB | 1048 | 1068 | 1108 | 1148 | **128** | **144** | **192** | **256** | 1668 |
| path | fallback | fallback | fallback | fallback | **fused** | **fused** | **fused** | **fused** | fallback |

Reproduce it by `mx.reset_peak_memory()` and `mx.get_peak_memory()` around
one call.

This one fact explains section 1 and section 3. The 18x spread of efficiency
against `head_dim` is not a curve. It is a cliff between two kernels.

Of the appendix shapes, only shape 9 (`head_dim=128`) and shape 10
(`head_dim=64`) reach the fused kernel by themselves.

## 1. The FALLBACK does not skip the causal triangle. The FUSED kernel does.

**Corrected.** An earlier version of this file said that
`mx.fast.scaled_dot_product_attention` never skips the triangle. That is
true of the fallback only, and the original measurement used `head_dim=32`,
which is on the fallback.

Fallback, at B=64, H=4, S=1024, head_dim=32:

| mask argument | Time |
|---|---:|
| `None` | 38.5 ms |
| `"causal"` | 52.7 ms |
| explicit bool array | 55.1 ms |

Fused, at B=32, H=8, S=1024, head_dim=64:

| mask argument | Time |
|---|---:|
| `None` | 20.31 ms |
| `"causal"` | **11.08 ms**, which is 0.55x |

So the two kernels behave in opposite ways:

- On the **fallback**, `"causal"` costs extra. The kernel builds the whole
  square and then masks it. The triangle must be skipped above the kernel,
  by blocking. That is what optimization 8 does.
- On the **fused** kernel, `"causal"` saves 45%. The kernel skips the masked
  blocks itself. Blocking above it only adds kernel launches, and measured
  worse on every shape tried.

Therefore blocked causal attention is a workaround for the fallback path.
Do not apply it once the head reaches the fused range.

## 2. Blocked causal attention

Split the query into blocks. Block `i` covers rows `[start, stop)` and
receives only keys `[0, stop)`. The masked tiles are never built.

    for start in range(0, S, block):
        stop = min(start + block, S)
        part = mx.fast.scaled_dot_product_attention(
            q[:, :, start:stop], k[:, :, :stop], v[:, :, :stop],
            scale=scale, mask="causal")

MLX aligns a `"causal"` mask to the **end** of the key sequence, so a query
block of length `stop - start` against keys `[0, stop)` receives exactly the
correct rows.

**It is bit exact.** `max_abs_diff = 0.0` against the unblocked call. This is
not an approximation.

Block size sweep, milliseconds, best of each row marked:

| B | H | S | head_dim | full | blk16 | blk32 | blk64 | blk128 | Best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 4 | 128 | 32 | 1.001 | 1.136 | 0.917 | **0.895** | — | blk64, 1.12x |
| 64 | 4 | 1024 | 32 | 52.00 | 48.50 | 36.32 | **31.90** | 32.02 | blk64, 1.63x |
| 64 | 4 | 32 | 32 | **0.139** | 0.193 | — | — | — | full |
| 64 | 16 | 128 | 8 | 3.408 | 3.236 | **2.649** | 2.709 | — | blk32, 1.29x |
| 64 | 4 | 128 | 8 | 0.855 | 0.872 | **0.687** | 0.693 | — | blk32, 1.24x |
| 64 | 2 | 128 | 64 | **0.194** | 0.500 | 0.310 | 0.281 | — | full |
| 64 | 1 | 128 | 128 | **0.208** | 0.522 | 0.318 | 0.303 | — | full |
| 64 | 4 | 128 | 256 | **2.549** | 4.919 | 3.726 | 3.154 | — | full |
| 1 | 4 | 128 | 32 | **0.050** | 0.245 | 0.129 | 0.074 | — | full |
| 4 | 4 | 128 | 32 | **0.088** | 0.246 | 0.134 | 0.093 | — | full |
| 16 | 4 | 128 | 32 | 0.267 | 0.338 | 0.257 | **0.247** | — | blk64, 1.08x |
| 128 | 4 | 128 | 32 | 2.049 | 2.318 | 1.911 | **1.904** | — | blk64, 1.08x |
| 1024 | 4 | 128 | 32 | 15.58 | 17.19 | 14.07 | **14.07** | — | blk64, 1.11x |

Three rules cover all 13 rows:

1. **`head_dim >= 64`: do not block.** A wide head is already efficient, so
   blocking only adds kernel launches. See section 3.
2. **`S <= 64`: do not block.** There is no triangle worth skipping.
3. **`B * H < 64`: do not block.** The GPU is not full, so launch cost
   dominates the saved arithmetic.

Otherwise block, at 32 for `head_dim <= 16` and 64 above it.

## 3. SDPA efficiency depends on `head_dim`, by 18x

B=64, H chosen to hold the work near constant, S=128, `mask="causal"`:

| head_dim | GFLOP/s |
|---:|---:|
| 8 | 78 |
| 32 | 270 |
| 128 | 1390 |
| 256 | 760 |

Peak float32 matmul on this GPU is about 3500 GFLOP/s.

**Read this table with section 0.** The step from 270 to 1390 is the step
from the fallback kernel to the fused kernel, not a gradual effect of the
reduction length. 8, 32 and 256 are fallback rows. 64 and 128 are fused
rows.

`head_dim = 8` is the worst case in the appendix table. It appears in shape 7
(D=32, H=4) and shape 11 (D=128, H=16). The reduction is 8 elements long, so
the kernel spends its time on addressing, not on arithmetic.

### Padding `head_dim` to reach the fused kernel

Padding with zeros is exact: zeros in q and k add nothing to the dot
product, zeros in v add nothing to the output, the extra output columns get
discarded, and `scale` stays at the TRUE `head_dim` so the softmax does not
move. Fold the pad into the fused QKV weight, so the projection writes the
wide layout and no extra pass over the data is needed.

The pad multiplies both the QKV projection and the attention arithmetic by
`rho = 64 / head_dim`. Whether it pays depends on `rho` and on how much of
the layer is attention. Per token per layer the cost is `6*D*D` for the QKV
projection against `4*S*D` for attention.

Full-sweep result of a blanket `head_dim < 64` rule, MLX ms before against
after. The noise floor is +-4%, from the three shapes whose path did not
change (8, 9, 10):

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

The blanket rule gives **0.983x FLOP-weighted**, which is a net loss. Two
conditions gate it instead:

1. `rho <= 2`, so `head_dim >= 32`. At `head_dim = 8` the projection grows
   8x and no attention rate recovers it.
2. `S >= 4*D`, so attention dominates the layer. Every shape with `S == D`
   measured inside the noise floor.

**The second threshold rests on ONE measured point, shape 13.** It sits
between shape 1 at `S = D` and shape 13 at `S = 8*D`. Sweep S at fixed D
before you move it.

This supersedes the older result that padding 8 up to 32 is not worth it.
That test was correct but aimed at the wrong target: 32 is still on the
fallback path, so it never reached the fused kernel at all.

## 4. Fused QKV projection

One `[D, 3D]` matmul in place of three `[D, D]` matmuls. It removes two
kernel launches per layer.

| B | S | D | 3 separate | 1 fused | Gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 0.0439 | 0.0220 | **1.99x** |
| 4 | 128 | 128 | 0.0436 | 0.0460 | 0.95x |
| 64 | 128 | 32 | 0.1094 | 0.0961 | 1.14x |
| 64 | 128 | 128 | 0.6202 | 0.5524 | 1.12x |
| 64 | 32 | 128 | 0.1319 | 0.1340 | 0.98x |
| 64 | 1024 | 128 | 3.5790 | 3.5689 | 1.00x |
| 64 | 128 | 1024 | 14.3045 | 14.2692 | 1.00x |
| 128 | 128 | 128 | 1.0695 | 1.0560 | 1.01x |
| 1024 | 128 | 128 | 6.8189 | 6.7662 | 1.01x |

The gain is large only at B=1, where the launch is most of the call. It never
loses outside noise, so it is always on. It also makes the compiled graph
smaller.

Build the weight once, at weight-copy time, not per call:

    qkvw = mx.concatenate([qw, kw, vw], axis=0).T    # [D, 3D]
    qkvb = mx.concatenate([qb, kb, vb], axis=0)      # [3D]

`mx.split(h @ qkvw + qkvb, 3, axis=-1)` returns q, k and v in that order.

## 5. Batch chunking

Shape 6 is B=10000. One activation is 625 MiB, and a layer holds five of
them. The GPU working set is 12.0 GiB.

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

This gives chunk=1024 for shape 6 and no chunking for every other shape in
the table.

## 6. Facts that cost time to find

- **`mx.compile` cannot take a shape-dependent Python branch as an
  argument.** A branch that changes the graph must produce a separate
  compiled variant. The code compiles one variant for the padded mask and
  one for the unpadded mask.
- **`torch.compile` cannot wrap an MLX class.** It raises
  `TypeError: cannot create weak reference to 'mlx.gc_func' object`. Use
  `mx.compile` inside the class. See `OPTIMIZATIONS.md`, attempt 5.
- **`mx.eval()` is not a synchronization point for timing.** Call
  `mx.synchronize()` as well. A timing loop without it measures graph
  building on the CPU.
- **Half precision cannot pass the harness thresholds, and this is not an
  MLX problem.** One bfloat16 step at magnitude 1.0 is 0.0078, which is 4
  times `atol=0.002`. `torch.nn.functional.scaled_dot_product_attention`
  fails the same test. See `OPTIMIZATIONS.md`.

## Not tried yet

- **A hand-written Metal kernel by `mx.fast.metal_kernel`**, for the
  `head_dim = 8` case of section 3. This is the largest gap that remains.
- **Fusing the GELU into the FFN matmuls.** Check first whether `mx.compile`
  already does it.
- **`mx.quantize` on the linear layers.** It will fail the accuracy test.
  Measure the speed only.

## End-to-end result of the dispatcher

`UserOptimizedTransformer` on each appendix shape, float32, no padding. The
"before" column forces `KernelPlan(fuse_qkv=False, causal_block=None,
batch_chunk=None)`, which is the single path used before this work. The
"after" column lets `plan_kernels()` choose.

Reproduce with `.venv/bin/python3 appendix_cases.py --cases all --run`.

| # | Shape | Before | After | Gain | Chosen plan |
|---:|---|---:|---:|---:|---|
| 1 | B64 D128 H4 S128 | 10.784 ms | 10.245 ms | 1.05x | blk64 |
| 2 | B1 D128 H4 S128 | 0.785 ms | 0.752 ms | 1.04x | full |
| 3 | B4 D128 H4 S128 | 1.150 ms | 1.125 ms | 1.02x | full |
| 4 | B16 D128 H4 S128 | 2.605 ms | 2.514 ms | 1.04x | blk64 |
| 5 | B128 D128 H4 S128 | 21.222 ms | 20.450 ms | 1.04x | blk64 |
| 6 | B10000 D128 H4 S128 | 2093.1 ms | 1602.7 ms | **1.31x** | blk64 + chunk 1024 |
| 7 | B64 D32 H4 S128 | 8.046 ms | 7.166 ms | **1.12x** | blk32 |
| 8 | B64 D1024 H4 S128 | 137.876 ms | 138.038 ms | 1.00x | full |
| 9 | B64 D128 H1 S128 | 7.147 ms | 7.290 ms | 0.98x | full |
| 10 | B64 D128 H2 S128 | 7.307 ms | 7.218 ms | 1.01x | full |
| 11 | B64 D128 H16 S128 | 21.002 ms | 17.955 ms | **1.17x** | blk32 |
| 12 | B64 D128 H4 S32 | 2.434 ms | 2.464 ms | 0.99x | full |
| 13 | B64 D128 H4 S1024 | 285.551 ms | 181.712 ms | **1.57x** | blk64 |

Shape 6 also drops peak GPU memory from 9.16 GiB to 2.68 GiB.

Where the plan chooses `full`, the two columns are the same path, so 0.98x
and 0.99x are measurement noise, not a loss.

Accuracy is unchanged. Every shape, with `padding_ratio` 0.0 and 0.3:
`max_abs_error` between 1.19e-06 and 2.50e-06, and 0 failed elements against
`atol=0.002`, `rtol=0.02`.

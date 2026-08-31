# MLX tensor operations: which kernel at which shape

Measured on this machine. See [machine.md](machine.md) for the limits and the
timing rules, and [test-shapes.md](test-shapes.md) for the shapes.

The code that applies these results is `plan_kernels()` in
`torch_transformer_benchmark.py`. Change a threshold there, not here. Record
the new measurement here.

## The short version

| Condition | Use |
|---|---|
| `head_dim` in {64, 72, 80, 96, 128} | one full `mx.fast.scaled_dot_product_attention` call |
| `head_dim` not in that set, causal, no padded batch | `steel_attention.py`: MLX's own fused kernel, compiled at the TRUE width. This beats both rows below, so they no longer select |
| `head_dim <= 16`, causal, `S > 64`, `B*H >= 64` | blocked causal, block 32 |
| `head_dim` 17..63, causal, `S > 64`, `B*H >= 64` | blocked causal, block 64 |
| `head_dim` not in that set, and the steel kernel cannot run | pad up to the next member of the set |
| always | one fused `[D, 3D]` QKV matmul |
| activation over 64 MiB | chunk the batch, full depth per chunk |

## The dispatch, as a diagram

[figures/kernel-dispatch.png](figures/kernel-dispatch.png) draws the whole
`plan_kernels()` tree: the three stages, the kernel each one takes, and the
condition that selects it.
[figures/kernel-dispatch-shape6.png](figures/kernel-dispatch-shape6.png)
resolves that tree for shape 6. Each file has a `-dark` variant for a dark
page. This section and `plan_kernels()` are the source of truth; the diagram
follows them.

## 0. MLX has two SDPA kernels, and the shape picks one

**This is the most useful fact in this file. Read it before section 1.**

`mx.fast.scaled_dot_product_attention` accepts every `head_dim`. It reaches
the fused flash kernel for **five values only**:

    FUSED_HEAD_DIMS = (64, 72, 80, 96, 128)

Outside that set it returns a correct answer through a fallback that
materializes the whole `B x H x S x S` score matrix.

**Corrected.** An earlier version of this file said "`head_dim` 64 to 128",
a contiguous range. That is WRONG. The earlier sweep tested 8, 16, 32, 48,
64, 72, 96, 128 and 256, and every value it tested inside 64..128 happens to
be a member of the set. `head_dim` 65, 100 and 127 are on the fallback.

Measured over `head_dim` 1 to 288, by peak GPU memory. The set does not move
with:

| Axis tested | Values | Result |
|---|---|---|
| mask | `None`, `"causal"`, bool array, float additive array | same set |
| dtype | float32, float16, bfloat16 | same set |
| `S` | 512, 1024 | same set |
| `B * H` | 1, 64, 256 | same set |
| kv heads | equal to q heads, and GQA with 2 | same set |

Reproduce with:

    .venv/bin/python3 profiling/probes/sdpa_dispatch.py --mode path --max-head-dim 288

Peak GPU memory at B=8, H=8, S=1024, where the score matrix is 256 MiB. The
column is `peak - base`, in MiB:

| head_dim | 32 | 63 | **64** | 65 | **72** | **80** | 88 | **96** | 100 | **128** | 192 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| peak MiB | 264 | 272 | **16** | 273 | **18** | **20** | 278 | **24** | 281 | **32** | 352 | 384 |
| path | fb | fb | **fused** | fb | **fused** | **fused** | fb | **fused** | fb | **fused** | fb | fb |

### Why the set has those five values

The kernels are compiled Metal templates. Read the names out of the shipped
metallib:

    strings .venv/lib/python3.13/site-packages/mlx/lib/mlx.metallib \
      | grep -E "^steel_attention" | sed 's/llvm$//' | sort -u

MLX 0.32.2 ships `bd64`, `bd72`, `bd80`, `bd96`, `bd128`, `bd192` and
`bd256`, for each of float32, float16 and bfloat16, and for a bool mask and
a typed mask. So the set is a list of compiled tile widths. It is not a
statement about where flash attention is worth using.

### MLX ships two kernels that it never calls

**`bd192` and `bd256` are in the metallib, and the dispatch never reaches
them.** Every axis in the table above was tested at `head_dim` 192 and 256,
and all 32 combinations took the fallback. This is a gap in the C++ dispatch
guard, not a missing kernel.

It costs shape 8 (`d_model=1024`, `num_heads=4`, so `head_dim=256`) the
fused path. Shape 8 carries 21.3% of the FLOP-weighted score. A `bd256`
`steel_attention` kernel exists on this machine and MLX will not call it.
Check this again after an MLX upgrade.

There is no workaround from Python. `head_dim` cannot be padded down, and a
head cannot be split, because the softmax runs over the full dot product.

Of the appendix shapes, only shape 9 (`head_dim=128`), shape 10 and shape 14
(`head_dim=64`) reach the fused kernel by themselves. Shape 8 has
`head_dim=256` and takes the fallback.

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
Do not apply it once the head reaches the fused set of section 0.

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

1. **`head_dim` in the fused set: do not block.** The fused kernel skips the
   triangle itself, so blocking only adds kernel launches. See section 1.
2. **`S <= 64`: do not block.** There is no triangle worth skipping.
3. **`B * H < 64`: do not block.** The GPU is not full, so launch cost
   dominates the saved arithmetic.

Otherwise block, at 32 for `head_dim <= 16` and 64 above it.

**Corrected.** Rule 1 said "`head_dim >= 64`: do not block", and gave "a wide
head is already efficient" as the reason. The reason is wrong. `head_dim=256`
is on the fallback, so it is NOT efficient, and the rule refuses to block it.
The `head_dim=256` row above measured `full` as the winner, so the row itself
still stands, but only because `S=128` leaves no triangle worth skipping.

Sweep S at `head_dim=256`, B=64, H=4, `mask="causal"`, median ms:

| S | full | blk32 | blk64 | blk128 | Best |
|---:|---:|---:|---:|---:|---|
| 128 | **2.456** | 3.558 | 3.047 | — | full |
| 256 | 7.677 | 9.816 | 7.889 | **7.508** | blk128, 1.02x |
| 512 | 27.922 | 30.964 | 22.573 | **21.174** | blk128, **1.32x** |
| 1024 | 104.439 | 109.669 | 74.589 | **67.181** | blk128, **1.55x** |

So a wide head that misses the fused set must be blocked once `S` is long
enough. `plan_kernels()` does not do this yet. No appendix shape reaches it:
shape 8 is the only `head_dim > 128` shape and it runs at `S=128`.

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

### The pad crossover, measured at the attention kernel

**Where does the pad start to pay?** Time the direct call at the true
`head_dim` against a padded call at each member of the fused set. The pad
cost is inside the number, so this is a lower bound: a model that folds the
pad into the QKV weight pays less.

    .venv/bin/python3 profiling/probes/sdpa_dispatch.py --mode pad \
        --batch 8 --heads 8 --seq 1024 --repeats 30 --check

**Result 1: always target the SMALLEST member of the set at or above
`head_dim`.** A larger target lost in every one of the 44 rows measured. The
fused kernel costs time in proportion to the padded width, at about the same
rate for all five widths, so a wider tile only buys arithmetic nobody wants.

B=8, H=8, S=1024, `mask="causal"`, median ms:

| head_dim | direct | pad64 | pad72 | pad80 | pad96 | pad128 | Best |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 13.153 | **4.255** | 4.837 | 5.720 | 6.791 | 9.254 | pad64, 3.09x |
| 32 | 13.656 | **3.968** | 4.674 | 5.367 | 6.481 | 8.716 | pad64, 3.44x |
| 63 | 14.402 | **3.856** | 4.611 | 5.436 | 6.617 | 8.782 | pad64, 3.74x |
| 65 | 15.623 | — | **4.538** | 5.343 | 6.464 | 8.741 | pad72, 3.44x |
| 76 | 15.807 | — | — | **5.247** | 6.381 | 8.687 | pad80, 3.01x |
| 88 | 16.284 | — | — | — | **6.239** | 8.555 | pad96, 2.61x |
| 100 | 17.072 | — | — | — | — | **8.523** | pad128, 2.00x |
| 127 | 17.833 | — | — | — | — | **8.233** | pad128, 2.17x |

`--check` reports `max_abs_diff` between 8.9e-07 and 2.5e-06 on every row,
which is the float32 reordering noise. The pad is exact.

**Result 2: the crossover moves with `S`, not with the pad ratio.** At
S=1024 every pad wins, even 8 -> 128, a ratio of 16. At S=128 the pad loses
at both ends. The fallback grows as `S*S` while the fused kernel grows as
`S*S/2` with a far better constant, so a long sequence buries any pad ratio.

Best target and gain, at the attention kernel only:

| head_dim | target | S=128 (B64 H4) | S=256 (B8 H8) | S=1024 (B8 H8) |
|---:|---:|---:|---:|---:|
| 8 | 64 | 1.00x (loses) | 1.52x | 3.09x |
| 16 | 64 | 1.03x | 1.57x | 3.31x |
| 32 | 64 | 1.22x | 1.65x | 3.44x |
| 48 | 64 | 1.31x | 1.66x | 3.56x |
| 63 | 64 | 1.46x | 1.96x | 3.74x |
| 65 | 72 | 1.28x | 1.69x | 3.44x |
| 70 | 72 | 1.33x | 1.75x | 3.51x |
| 76 | 80 | 1.24x | 1.56x | 3.01x |
| 88 | 96 | 1.12x | 1.44x | 2.61x |
| 100 | 128 | 1.00x (loses) | 1.11x | 2.00x |
| 112 | 128 | 1.00x (loses) | 1.17x | 2.06x |
| 127 | 128 | 1.10x | 1.27x | 2.17x |

The saddle at S=128 sits near `head_dim=16` for the 64 target, and the
96 -> 128 step fails at `head_dim` 100 and 112.

**This table is the attention kernel alone. It is not the rule for the
model.** The pad also widens the QKV projection and the output projection by
the same ratio, and those carry `6*D*D` per token against `4*S*D` for
attention. The model-level gate stays the one measured below.

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
- **`mx.compile` does not fuse a bias add into a matmul.** `h @ w + b`
  starts a second kernel, which reads and writes the whole output again.
  `mx.addmm(b, h, w)` gives the bias to the matmul as its C operand, and the
  GPU adds it inside the matmul kernel. Measured on one projection at the
  shape 6 dimensions: 6.768 ms to 3.749 ms, and the compiled times match the
  raw times. It gave 1.096x FLOP-weighted over the whole model. See
  `OPTIMIZATIONS.md` row 29.
- **A rank 3 by rank 2 matmul needs no flatten.** MLX collapses the leading
  dimensions into one GEMM already. `[B, S, D] @ [D, O]` and
  `[B*S, D] @ [D, O]` measured 0.992x to 1.000x of each other on four sizes.
  See `OPTIMIZATIONS.md` row 30.
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
- **Fusing the GELU into the FFN matmuls.** DONE, as OPTIMIZATIONS.md row 33.
  `mx.compile` does NOT do it: 2.237 ms compiled against 2.242 ms plain at
  the shape 6 chunk. `steel_gemm.py` hoists MLX's steel GEMM and applies
  GELU through the `apply_epilogue` hook in `steel/gemm/mma.h`, on the
  accumulator tile in registers. 1.064x FLOP-weighted.
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

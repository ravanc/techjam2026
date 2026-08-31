# Test shapes (Appendix 3.7)

The 14 shapes that the grader uses. `appendix_cases.py` holds the same table as
code. This file is the reference: it gives the derived cost of each shape, and
it shows which part of the model each shape loads.

Print the table from the code:

    .venv/bin/python3 appendix_cases.py --list

## The table

| # | Batch Size | QKV Dim | Heads | Seq Len | Layers | Causal | FFN Dim |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 64 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 2 | 1 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 3 | 4 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 4 | 16 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 5 | 128 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 6 | 10000 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 7 | 64 | 32 | 4 | 128 | 4 | TRUE | 32 |
| 8 | 64 | 1024 | 4 | 128 | 4 | TRUE | 1024 |
| 9 | 64 | 128 | 1 | 128 | 4 | TRUE | 128 |
| 10 | 64 | 128 | 2 | 128 | 4 | TRUE | 128 |
| 11 | 64 | 128 | 16 | 128 | 4 | TRUE | 128 |
| 12 | 64 | 128 | 4 | 32 | 4 | TRUE | 128 |
| 13 | 64 | 128 | 4 | 1024 | 4 | TRUE | 128 |
| 14 | 32 | 1024 | 16 | 100000 | 2 | TRUE | 1024 |

Column names map to `TransformerConfig` like this:

| Table column | Field |
|---|---|
| Batch Size | `batch_size` |
| QKV Dim | `d_model` |
| Heads | `num_heads` |
| Seq Len | `seq_len` |
| Layers | `num_layers` |
| Causal | `causal` |
| FFN Dim | `ffn_dim` |

Every shape is causal. Every shape except 14 has 4 layers. In each shape the
FFN dim equals the QKV dim, so the FFN does no expansion.

## The table is one sweep

Shape 1 is the base. Each other shape moves one column away from it. Read the
table in these groups:

| Group | Shapes | What moves | Range |
|---|---|---|---|
| Base | 1 | — | B=64, D=128, H=4, S=128 |
| Batch | 2, 3, 4, **1**, 5, 6 | `batch_size` | 1 -> 10000 |
| Width | 7, **1**, 8 | `d_model` and `ffn_dim` together | 32 -> 1024 |
| Heads | 9, 10, **1**, 11 | `num_heads` (`d_model` fixed) | 1 -> 16 |
| Sequence | 12, **1**, 13 | `seq_len` | 32 -> 1024 |
| Extreme | 14 | all of them at once | — |

The head group changes `head_dim = d_model / num_heads`, not the work. The
width group changes the work as `d_model` squared.

## Derived cost

`act/layer` is one B x S x D activation in float32. `scores/layer` is the
B x H x S x S score matrix that `BaselineSelfAttention` materializes.
`GFLOP` counts the multiply-adds of the whole forward pass.

| # | head_dim | tokens | act/layer | scores/layer | GFLOP | attention share |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 8,192 | 4.0 MiB | 16.0 MiB | 8.6 | 25% |
| 2 | 32 | 128 | 64 KiB | 256 KiB | 0.1 | 25% |
| 3 | 32 | 512 | 256 KiB | 1.0 MiB | 0.5 | 25% |
| 4 | 32 | 2,048 | 1.0 MiB | 4.0 MiB | 2.1 | 25% |
| 5 | 32 | 16,384 | 8.0 MiB | 32.0 MiB | 17.2 | 25% |
| 6 | 32 | 1,280,000 | 625 MiB | 2.4 GiB | 1,342 | 25% |
| 7 | **8** | 8,192 | 1.0 MiB | 16.0 MiB | 0.9 | **57%** |
| 8 | **256** | 8,192 | 32.0 MiB | 16.0 MiB | 430 | **4%** |
| 9 | **128** | 8,192 | 4.0 MiB | 4.0 MiB | 8.6 | 25% |
| 10 | 64 | 8,192 | 4.0 MiB | 8.0 MiB | 8.6 | 25% |
| 11 | **8** | 8,192 | 4.0 MiB | 64.0 MiB | 8.6 | 25% |
| 12 | 32 | 2,048 | 1.0 MiB | 1.0 MiB | 1.7 | 8% |
| 13 | 32 | 65,536 | 32.0 MiB | 1.0 GiB | 189 | **73%** |
| 14 | 64 | 3,200,000 | **12.2 GiB** | **18.6 TiB** | 2,701,971 | 97% |

The total work spans 7 orders of magnitude, from 0.1 GFLOP to 2.7 PFLOP. One
kernel choice cannot be correct across that range.

## What each shape loads

| # | Class | The limit |
|---:|---|---|
| 1 | balanced | the reference point |
| 2 | latency | 0.13 GFLOP. Kernel launch and the framework boundary dominate. The GPU is idle most of the call. |
| 3, 4 | latency | still under-occupies the GPU |
| 5 | balanced | first shape that fills the GPU |
| 6 | memory | 625 MiB per activation. Five live activations per layer exceed the 12 GiB working set. Needs a chunk loop. |
| 7 | latency | head_dim=8, below the SDPA fast-path width. Also the smallest matmuls in the table. |
| 8 | compute | d_model=1024 makes the projections 64x the base cost. Pure matmul throughput. |
| 9 | attention | one head, head_dim=128. No head parallelism, so B alone must fill the GPU. |
| 10 | balanced | head_dim=64, the best SDPA width |
| 11 | attention | 16 heads at head_dim=8. Wide parallelism, narrow reductions. |
| 12 | latency | S=32. The causal mask throws away half of a very small tile. |
| 13 | attention | S=1024 makes attention 73% of the work. Score matrix is 1 GiB per layer. |
| 14 | memory, and no reference | the baseline cannot run it. `shape14_harness.py` runs the MLX path. See below |

## Shape 14

**The baseline cannot run this shape. The MLX path can.** Those are two
separate facts, and an earlier version of this file ran them together and
called the whole shape impossible. Only the first half is true.

| Item | Size in float32 |
|---|---|
| Input `x`, whole batch | 12.2 GiB |
| Input `x`, one batch row | 391 MiB |
| Score matrix, one layer, whole batch | 18.6 TiB |
| Score matrix, one layer, one batch row | 596 GiB |
| Machine working set | 12.0 GiB |

`BaselineSelfAttention` materializes the score matrix, so the torch baseline
stops with an allocation error before it computes anything. It has no run
time at shape 14, on the CPU or on MPS. Do not report a time for it, and do
not report a speedup: there is no baseline time to divide by.

### Why the MLX path runs it

`head_dim = 1024 / 16 = 64`, which is in the fused set. So
`mx.fast.scaled_dot_product_attention` reaches the fused flash kernel, which
holds no score matrix. Measured with `shape14_harness.py --ladder`, at
H=16, head_dim=64, float32, `mask="causal"`, one batch row:

| S | scores if materialized | call adds | ms |
|---:|---:|---:|---:|
| 4,096 | 1.0 GiB | 16.0 MiB | 10.7 |
| 16,384 | 16.0 GiB | 64.0 MiB | 168.7 |
| 65,536 | 256.0 GiB | 256.0 MiB | 3,501 |
| 100,000 | 596.0 GiB | 390.6 MiB | 8,127 |

The "call adds" column is the output alone. Memory grows as `S`, and time
grows as `S * S`. The fused kernel is not a preference here, it is the only
path: the fallback needs 596 GiB for one row of one layer.

`plan_kernels()` already selects `batch_chunk=1` for this shape, so
`forward()` runs one 391 MiB sequence at a time.

### The harness

`shape14_harness.py` runs shape 14, and it runs nothing else. It does not
change `torch_transformer_benchmark.py`, and it does not change
`BaselineTransformer`. It exists because the graded harness needs the
baseline for both of its jobs, and the baseline is gone.

It does not call `UserOptimizedTransformer.forward()`, because that
concatenates all 32 chunk outputs into one 12.2 GiB array. It drives the
chunk loop itself and reduces each row before it takes the next.

Accuracy comes from the causal property: output row `i` uses input rows
`0..i` only.

- `--check` gives the *unmodified* baseline the first 1024 tokens, which it
  can hold, and compares its answer against the first 1024 rows of the full
  answer. Measured: `max_abs = 2.03e-06`, 0 of 1,048,576 elements failed,
  against the harness `atol = 0.002`.
- `--tail` reaches row 99999, which no truncation can reach. It uses
  `frugal_forward()`, a query-blocked copy of the baseline arithmetic.
  `--validate` proves that copy equal to `BaselineTransformer` first, over
  12 shapes, 2 padding settings and 2 query-block counts. **48 of 48 cases
  PASS, and 42 are bit exact.** The 6 that are not read `max_abs` 9.5e-07 to
  1.43e-06, against the harness `atol = 0.002`. All 6 are a blocked run, so
  the difference is the shorter reduction, as expected.
- `--coverage` shows that shape 14 selects no untested kernel branch. Only
  the *value* `batch_chunk=1` is new, and shape 6 runs the same loop at
  `batch_chunk=1024`.

### Measured

`shape14_harness.py --time --rows 32`, at commit 311a420*, one row for each
of the 32 chunks:

| Item | Value |
|---|---|
| per row, median compute | 15.766 s (min 15.681, max 17.700) |
| full batch, compute | **504.5 s = 8.41 min** |
| full batch, including the framework copy | 505.6 s |
| framework boundary | 0.2% of the call |
| tokens/s | 6,343 |
| CPU baseline | cannot run |
| MPS baseline | cannot run |
| speedup | none. There is no baseline time |

The rate depends on which FLOP count you credit:

| | counted (full `S x S`) | executed (causal triangle) |
|---|---:|---:|
| GFLOP | 2,701,971 | 1,391,264 |
| MLX TFLOP/s | 5.356 | **2.758** |
| MFU against 4.946 | 108.3% | **55.8%** |
| against the 4.06 measured ceiling | 131.9% | **67.9%** |

`flops.py` credits the full `S x S`, because that is what the baseline
calculates, and that is the graded MFU. The fused kernel runs the triangle
only, which is 51.5% of the counted work here. So the counted MFU passes
100%, and it is not a measurement error. Read the **executed** column
against the 82% practical ceiling. Shape 13 shows the same effect at 73%
attention; shape 14 shows it at 97%.

The framework boundary is 0.2% here, against 3.4% at shape 6
([machine.md](machine.md)). The copy is the same 391 MiB per chunk, and the
compute it hides behind is 340x longer.

## Notes

- The harness draws its input on the CPU and then moves it, so every backend
  sees identical bytes. See the docstring of `bench_cases.py`.
- `appendix_cases.py --budget-gb` skips a shape whose float32 input is over
  the budget. The default is 8 GiB.
- Set `--padding-ratio` above 0 to add padding. The MLX path then uses an
  array mask in place of the `"causal"` string, which is a slower kernel path.

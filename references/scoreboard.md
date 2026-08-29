# Scoreboard

Timing for every Appendix 3.7 shape. Regenerate with:

    .venv/bin/python3 scoreboard.py --label "what changed"

Every number below is a stopwatch reading or a ratio of two stopwatch
readings. `TFLOP/s` divides the model FLOP count by measured time; the
FLOP count comes from the model definition, not from a specification.

- dtype: `float32`
- CPU is the reference for both accuracy and speedup.
- Each call is bracketed by a device synchronize. Rounds alternate the
  backend order, so no backend always runs on a cold chip.
- sweep took 10.4 minutes

MFU appears once, in its own section, and it is provisional. See
[../flops.py](../flops.py).

## Speedup against the CPU baseline

| # | Shape | CPU ms | MPS ms | MLX ms | MPS vs CPU | **MLX vs CPU** | MLX vs MPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 48.295 | 17.333 | 9.966 | 2.79x | **4.85x** | 1.74x |
| 2 | B1 D128 H4 S128 | 1.767 | 1.663 | 0.775 | 1.06x | **2.28x** | 2.15x |
| 3 | B4 D128 H4 S128 | 5.260 | 2.017 | 1.128 | 2.61x | **4.66x** | 1.79x |
| 4 | B16 D128 H4 S128 | 12.757 | 4.732 | 2.507 | 2.70x | **5.09x** | 1.89x |
| 5 | B128 D128 H4 S128 | 110.068 | 33.998 | 20.236 | 3.24x | **5.44x** | 1.68x |
| 6 | B10000 D128 H4 S128 | 18942.688 | 2792.548 | 1772.615 | 6.78x | **10.69x** | 1.58x |
| 7 | B64 D32 H4 S128 | 26.652 | 11.299 | 6.945 | 2.36x | **3.84x** | 1.63x |
| 8 | B64 D1024 H4 S128 | 464.375 | 165.389 | 135.822 | 2.81x | **3.42x** | 1.22x |
| 9 | B64 D128 H1 S128 | 32.091 | 7.906 | 7.051 | 4.06x | **4.55x** | 1.12x |
| 10 | B64 D128 H2 S128 | 38.433 | 12.589 | 6.935 | 3.05x | **5.54x** | 1.82x |
| 11 | B64 D128 H16 S128 | 124.519 | 41.939 | 17.074 | 2.97x | **7.29x** | 2.46x |
| 12 | B64 D128 H4 S32 | 9.453 | 3.384 | 2.407 | 2.79x | **3.93x** | 1.41x |
| 13 | B64 D128 H4 S1024 | 1997.648 | 562.967 | 111.195 | 3.55x | **17.97x** | 5.06x |

## Achieved arithmetic rate

Model FLOPs divided by measured time. Compare against the matmul rates
measured on this machine: MLX float32 reaches 4.06 TFLOP/s on a square
matmul, torch MPS 4.16, torch CPU 1.42.

| # | Shape | GFLOP | CPU TFLOP/s | MPS TFLOP/s | **MLX TFLOP/s** | MLX token/s | Plan chosen |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | B64 D128 H4 S128 | 8.59 | 0.178 | 0.496 | **0.862** | 822,002 | fuse_qkv=True causal_block=64 batch_chunk=none pad_head_dim=none |
| 2 | B1 D128 H4 S128 | 0.13 | 0.076 | 0.081 | **0.173** | 165,197 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none |
| 3 | B4 D128 H4 S128 | 0.54 | 0.102 | 0.266 | **0.476** | 453,892 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none |
| 4 | B16 D128 H4 S128 | 2.15 | 0.168 | 0.454 | **0.857** | 817,028 | fuse_qkv=True causal_block=64 batch_chunk=none pad_head_dim=none |
| 5 | B128 D128 H4 S128 | 17.18 | 0.156 | 0.505 | **0.849** | 809,658 | fuse_qkv=True causal_block=64 batch_chunk=none pad_head_dim=none |
| 6 | B10000 D128 H4 S128 | 1342.18 | 0.071 | 0.481 | **0.757** | 722,097 | fuse_qkv=True causal_block=64 batch_chunk=1024 pad_head_dim=none |
| 7 | B64 D32 H4 S128 | 0.94 | 0.035 | 0.083 | **0.135** | 1,179,497 | fuse_qkv=True causal_block=32 batch_chunk=none pad_head_dim=none |
| 8 | B64 D1024 H4 S128 | 429.50 | 0.925 | 2.597 | **3.162** | 60,314 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none |
| 9 | B64 D128 H1 S128 | 8.59 | 0.268 | 1.087 | **1.218** | 1,161,876 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none |
| 10 | B64 D128 H2 S128 | 8.59 | 0.224 | 0.682 | **1.239** | 1,181,201 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none |
| 11 | B64 D128 H16 S128 | 8.59 | 0.069 | 0.205 | **0.503** | 479,802 | fuse_qkv=True causal_block=32 batch_chunk=none pad_head_dim=none |
| 12 | B64 D128 H4 S32 | 1.74 | 0.185 | 0.516 | **0.725** | 850,984 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none |
| 13 | B64 D128 H4 S1024 | 188.98 | 0.095 | 0.336 | **1.700** | 589,376 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=64 |

## MFU — PROVISIONAL, do not quote without this note

> **PROVISIONAL: the MFU denominator (5.01 TFLOP/s) was asserted from memory, not verified. Every MFU below scales with it. The measured matmul rate of 4.06 TFLOP/s is the verified alternative.**

The numerator and the time are objective. The denominator is not:
5.01 TFLOP/s came from `14 cores x 128 ALUs x 2 x 1.398 GHz`,
and the clock and ALU count were never checked against a source. Every
number in this table scales linearly with it. Fix the constant in
`flops.py` before this table means anything.

| # | Shape | GFLOP | FLOP share | MLX MFU | MPS MFU |
|---:|---|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 8.59 | 0.4% | 17.2% | 9.9% |
| 2 | B1 D128 H4 S128 | 0.13 | 0.0% | 3.5% | 1.6% |
| 3 | B4 D128 H4 S128 | 0.54 | 0.0% | 9.5% | 5.3% |
| 4 | B16 D128 H4 S128 | 2.15 | 0.1% | 17.1% | 9.1% |
| 5 | B128 D128 H4 S128 | 17.18 | 0.9% | 16.9% | 10.1% |
| 6 | B10000 D128 H4 S128 | 1342.18 | 66.5% | 15.1% | 9.6% |
| 7 | B64 D32 H4 S128 | 0.94 | 0.0% | 2.7% | 1.7% |
| 8 | B64 D1024 H4 S128 | 429.50 | 21.3% | 63.1% | 51.8% |
| 9 | B64 D128 H1 S128 | 8.59 | 0.4% | 24.3% | 21.7% |
| 10 | B64 D128 H2 S128 | 8.59 | 0.4% | 24.7% | 13.6% |
| 11 | B64 D128 H16 S128 | 8.59 | 0.4% | 10.0% | 4.1% |
| 12 | B64 D128 H4 S32 | 1.74 | 0.1% | 14.5% | 10.3% |
| 13 | B64 D128 H4 S1024 | 188.98 | 9.4% | 33.9% | 6.7% |

- unweighted mean MLX MFU: **19.43%**
- FLOP-weighted mean MLX MFU: **27.17%**

Shape 6 alone is two thirds of the FLOP weight, so a FLOP-weighted
score is mostly a score on shape 6.

## Accuracy

Against the CPU baseline, at `atol=0.002` and `rtol=0.02`.

| # | MPS | MLX |
|---:|---|---|
| 1 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 2 | PASS `max_abs=1.07e-06` (0/16384 failed) | PASS `max_abs=9.54e-07` (0/16384 failed) |
| 3 | PASS `max_abs=1.19e-06` (0/65536 failed) | PASS `max_abs=1.31e-06` (0/65536 failed) |
| 4 | PASS `max_abs=1.19e-06` (0/262144 failed) | PASS `max_abs=1.31e-06` (0/262144 failed) |
| 5 | PASS `max_abs=1.43e-06` (0/2097152 failed) | PASS `max_abs=1.43e-06` (0/2097152 failed) |
| 6 | PASS `max_abs=1.91e-06` (0/163840000 failed) | PASS `max_abs=1.91e-06` (0/163840000 failed) |
| 7 | PASS `max_abs=1.19e-06` (0/262144 failed) | PASS `max_abs=1.19e-06` (0/262144 failed) |
| 8 | PASS `max_abs=2.86e-06` (0/8388608 failed) | PASS `max_abs=2.65e-06` (0/8388608 failed) |
| 9 | PASS `max_abs=1.67e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 10 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 11 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 12 | PASS `max_abs=1.91e-06` (0/262144 failed) | PASS `max_abs=1.67e-06` (0/262144 failed) |
| 13 | PASS `max_abs=1.67e-06` (0/8388608 failed) | PASS `max_abs=1.67e-06` (0/8388608 failed) |

## Summary

| Metric | Value |
|---|---:|
| Shapes scored | 13 |
| Median MLX speedup over CPU | **4.85x** |
| Range of MLX speedup | 2.28x to 17.97x |
| Median MLX rate | 0.849 TFLOP/s |
| Best MLX rate | 3.162 TFLOP/s |

## History

Every recorded sweep. The data is in [../profiling/history.jsonl](../profiling/history.jsonl).
Print it with `.venv/bin/python3 scoreboard.py --show-history`.

| When | Commit | Label | Shapes | Median speedup | Median TFLOP/s |
|---|---|---|---:|---:|---:|
| 2026-08-29T11:11:06 | `b37a558*` | attempts 1-11, full sweep | 13 | 5.01x | 0.837 |
| 2026-08-29T11:39:59 | `6f6cfce*` | pad head_dim to 64 to reach the fused SDPA kernel | 13 | 4.76x | 0.797 |
| 2026-08-29T12:05:46 | `6f6cfce*` | pad head_dim to 64, gated to rho<=2 and S>=4D (shape 13 only) | 13 | 4.85x | 0.849 |

A `*` on the commit means the working tree had uncommitted changes,
so that reading cannot be reproduced exactly.


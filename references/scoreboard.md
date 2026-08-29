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
- sweep took 8.7 minutes

MFU appears once, in its own section, and it is provisional. See
[../flops.py](../flops.py).

## Speedup against the CPU baseline

| # | Shape | CPU ms | MPS ms | MLX ms | MPS vs CPU | **MLX vs CPU** | MLX vs MPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 49.136 | 17.310 | 10.141 | 2.84x | **4.85x** | 1.71x |
| 2 | B1 D128 H4 S128 | 1.564 | 1.553 | 0.780 | 1.01x | **2.00x** | 1.99x |
| 3 | B4 D128 H4 S128 | 4.443 | 2.009 | 1.130 | 2.21x | **3.93x** | 1.78x |
| 4 | B16 D128 H4 S128 | 12.699 | 4.700 | 2.495 | 2.70x | **5.09x** | 1.88x |
| 5 | B128 D128 H4 S128 | 113.207 | 34.555 | 20.527 | 3.28x | **5.51x** | 1.68x |
| 6 | B10000 D128 H4 S128 | 13601.131 | 2720.155 | 1677.834 | 5.00x | **8.11x** | 1.62x |
| 7 | B64 D32 H4 S128 | 29.107 | 11.534 | 7.100 | 2.52x | **4.10x** | 1.62x |
| 8 | B64 D1024 H4 S128 | 467.508 | 166.502 | 137.555 | 2.81x | **3.40x** | 1.21x |
| 9 | B64 D128 H1 S128 | 36.296 | 8.013 | 7.243 | 4.53x | **5.01x** | 1.11x |
| 10 | B64 D128 H2 S128 | 42.480 | 12.931 | 7.041 | 3.29x | **6.03x** | 1.84x |
| 11 | B64 D128 H16 S128 | 129.809 | 42.302 | 17.257 | 3.07x | **7.52x** | 2.45x |
| 12 | B64 D128 H4 S32 | 11.788 | 3.425 | 2.386 | 3.44x | **4.94x** | 1.44x |
| 13 | B64 D128 H4 S1024 | 1864.557 | 568.610 | 182.790 | 3.28x | **10.20x** | 3.11x |

## Achieved arithmetic rate

Model FLOPs divided by measured time. Compare against the matmul rates
measured on this machine: MLX float32 reaches 4.06 TFLOP/s on a square
matmul, torch MPS 4.16, torch CPU 1.42.

| # | Shape | GFLOP | CPU TFLOP/s | MPS TFLOP/s | **MLX TFLOP/s** | MLX token/s | Plan chosen |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | B64 D128 H4 S128 | 8.59 | 0.175 | 0.496 | **0.847** | 807,845 | fuse_qkv=True causal_block=64 batch_chunk=none |
| 2 | B1 D128 H4 S128 | 0.13 | 0.086 | 0.086 | **0.172** | 164,024 | fuse_qkv=True causal_block=full batch_chunk=none |
| 3 | B4 D128 H4 S128 | 0.54 | 0.121 | 0.267 | **0.475** | 453,248 | fuse_qkv=True causal_block=full batch_chunk=none |
| 4 | B16 D128 H4 S128 | 2.15 | 0.169 | 0.457 | **0.861** | 820,945 | fuse_qkv=True causal_block=64 batch_chunk=none |
| 5 | B128 D128 H4 S128 | 17.18 | 0.152 | 0.497 | **0.837** | 798,154 | fuse_qkv=True causal_block=64 batch_chunk=none |
| 6 | B10000 D128 H4 S128 | 1342.18 | 0.099 | 0.493 | **0.800** | 762,888 | fuse_qkv=True causal_block=64 batch_chunk=1024 |
| 7 | B64 D32 H4 S128 | 0.94 | 0.032 | 0.081 | **0.132** | 1,153,769 | fuse_qkv=True causal_block=32 batch_chunk=none |
| 8 | B64 D1024 H4 S128 | 429.50 | 0.919 | 2.580 | **3.122** | 59,554 | fuse_qkv=True causal_block=full batch_chunk=none |
| 9 | B64 D128 H1 S128 | 8.59 | 0.237 | 1.072 | **1.186** | 1,130,948 | fuse_qkv=True causal_block=full batch_chunk=none |
| 10 | B64 D128 H2 S128 | 8.59 | 0.202 | 0.664 | **1.220** | 1,163,499 | fuse_qkv=True causal_block=full batch_chunk=none |
| 11 | B64 D128 H16 S128 | 8.59 | 0.066 | 0.203 | **0.498** | 474,701 | fuse_qkv=True causal_block=32 batch_chunk=none |
| 12 | B64 D128 H4 S32 | 1.74 | 0.148 | 0.509 | **0.731** | 858,355 | fuse_qkv=True causal_block=full batch_chunk=none |
| 13 | B64 D128 H4 S1024 | 188.98 | 0.101 | 0.332 | **1.034** | 358,532 | fuse_qkv=True causal_block=64 batch_chunk=none |

## MFU — PROVISIONAL, do not quote without this note

> **PROVISIONAL: the MFU denominator (5.01 TFLOP/s) was asserted from memory, not verified. Every MFU below scales with it. The measured matmul rate of 4.06 TFLOP/s is the verified alternative.**

The numerator and the time are objective. The denominator is not:
5.01 TFLOP/s came from `14 cores x 128 ALUs x 2 x 1.398 GHz`,
and the clock and ALU count were never checked against a source. Every
number in this table scales linearly with it. Fix the constant in
`flops.py` before this table means anything.

| # | Shape | GFLOP | FLOP share | MLX MFU | MPS MFU |
|---:|---|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 8.59 | 0.4% | 16.9% | 9.9% |
| 2 | B1 D128 H4 S128 | 0.13 | 0.0% | 3.4% | 1.7% |
| 3 | B4 D128 H4 S128 | 0.54 | 0.0% | 9.5% | 5.3% |
| 4 | B16 D128 H4 S128 | 2.15 | 0.1% | 17.2% | 9.1% |
| 5 | B128 D128 H4 S128 | 17.18 | 0.9% | 16.7% | 9.9% |
| 6 | B10000 D128 H4 S128 | 1342.18 | 66.5% | 16.0% | 9.8% |
| 7 | B64 D32 H4 S128 | 0.94 | 0.0% | 2.6% | 1.6% |
| 8 | B64 D1024 H4 S128 | 429.50 | 21.3% | 62.3% | 51.5% |
| 9 | B64 D128 H1 S128 | 8.59 | 0.4% | 23.7% | 21.4% |
| 10 | B64 D128 H2 S128 | 8.59 | 0.4% | 24.4% | 13.3% |
| 11 | B64 D128 H16 S128 | 8.59 | 0.4% | 9.9% | 4.1% |
| 12 | B64 D128 H4 S32 | 1.74 | 0.1% | 14.6% | 10.2% |
| 13 | B64 D128 H4 S1024 | 188.98 | 9.4% | 20.6% | 6.6% |

- unweighted mean MLX MFU: **18.30%**
- FLOP-weighted mean MLX MFU: **26.32%**

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
| Median MLX speedup over CPU | **5.01x** |
| Range of MLX speedup | 2.00x to 10.20x |
| Median MLX rate | 0.837 TFLOP/s |
| Best MLX rate | 3.122 TFLOP/s |

## History

Every recorded sweep. The data is in [../profiling/history.jsonl](../profiling/history.jsonl).
Print it with `.venv/bin/python3 scoreboard.py --show-history`.

| When | Commit | Label | Shapes | Median speedup | Median TFLOP/s |
|---|---|---|---:|---:|---:|
| 2026-08-29T11:11:06 | `b37a558*` | attempts 1-11, full sweep | 13 | 5.01x | 0.837 |

A `*` on the commit means the working tree had uncommitted changes,
so that reading cannot be reproduced exactly.


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
- sweep took 15.7 minutes

MFU appears once, in its own section, and it is provisional. See
[../flops.py](../flops.py).

## Speedup against the CPU baseline

| # | Shape | CPU ms | MPS ms | MLX ms | MPS vs CPU | **MLX vs CPU** | MLX vs MPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 52.114 | 17.879 | 8.004 | 2.91x | **6.51x** | 2.23x |
| 2 | B1 D128 H4 S128 | 1.729 | 1.623 | 0.644 | 1.07x | **2.68x** | 2.52x |
| 3 | B4 D128 H4 S128 | 5.997 | 2.290 | 1.106 | 2.62x | **5.42x** | 2.07x |
| 4 | B16 D128 H4 S128 | 22.004 | 5.045 | 2.227 | 4.36x | **9.88x** | 2.27x |
| 5 | B128 D128 H4 S128 | 198.276 | 49.227 | 17.548 | 4.03x | **11.30x** | 2.81x |
| 6 | B10000 D128 H4 S128 | 24762.840 | 2832.288 | 1346.945 | 8.74x | **18.38x** | 2.10x |
| 7 | B64 D32 H4 S128 | 29.116 | 11.549 | 4.701 | 2.52x | **6.19x** | 2.46x |
| 8 | B64 D1024 H4 S128 | 477.166 | 165.823 | 135.210 | 2.88x | **3.53x** | 1.23x |
| 9 | B64 D128 H1 S128 | 35.645 | 7.895 | 6.855 | 4.51x | **5.20x** | 1.15x |
| 10 | B64 D128 H2 S128 | 46.137 | 12.902 | 6.824 | 3.58x | **6.76x** | 1.89x |
| 11 | B64 D128 H16 S128 | 138.431 | 42.029 | 7.800 | 3.29x | **17.75x** | 5.39x |
| 12 | B64 D128 H4 S32 | 10.582 | 3.418 | 2.142 | 3.10x | **4.94x** | 1.60x |
| 13 | B64 D128 H4 S1024 | 1887.825 | 566.156 | 75.789 | 3.33x | **24.91x** | 7.47x |

## Achieved arithmetic rate

Model FLOPs divided by measured time. Compare against the matmul rates
measured on this machine: MLX float32 reaches 4.06 TFLOP/s on a square
matmul, torch MPS 4.16, torch CPU 1.42.

| # | Shape | GFLOP | CPU TFLOP/s | MPS TFLOP/s | **MLX TFLOP/s** | MLX token/s | Plan chosen |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | B64 D128 H4 S128 | 8.59 | 0.165 | 0.480 | **1.073** | 1,023,504 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 2 | B1 D128 H4 S128 | 0.13 | 0.078 | 0.083 | **0.208** | 198,706 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 3 | B4 D128 H4 S128 | 0.54 | 0.090 | 0.234 | **0.486** | 463,121 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 4 | B16 D128 H4 S128 | 2.15 | 0.098 | 0.426 | **0.964** | 919,743 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 5 | B128 D128 H4 S128 | 17.18 | 0.087 | 0.349 | **0.979** | 933,685 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 6 | B10000 D128 H4 S128 | 1342.18 | 0.054 | 0.474 | **0.996** | 950,299 | fuse_qkv=True causal_block=full batch_chunk=1024 pad_head_dim=none steel=True |
| 7 | B64 D32 H4 S128 | 0.94 | 0.032 | 0.081 | **0.200** | 1,742,732 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 8 | B64 D1024 H4 S128 | 429.50 | 0.900 | 2.590 | **3.177** | 60,587 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False |
| 9 | B64 D128 H1 S128 | 8.59 | 0.241 | 1.088 | **1.253** | 1,195,040 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False |
| 10 | B64 D128 H2 S128 | 8.59 | 0.186 | 0.666 | **1.259** | 1,200,385 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False |
| 11 | B64 D128 H16 S128 | 8.59 | 0.062 | 0.204 | **1.101** | 1,050,268 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 12 | B64 D128 H4 S32 | 1.74 | 0.165 | 0.510 | **0.815** | 956,106 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 13 | B64 D128 H4 S1024 | 188.98 | 0.100 | 0.334 | **2.493** | 864,720 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |

## MFU — PROVISIONAL, do not quote without this note

> **PROVISIONAL: the MFU denominator (5.01 TFLOP/s) was asserted from memory, not verified. Every MFU below scales with it. The measured matmul rate of 4.06 TFLOP/s is the verified alternative.**

The numerator and the time are objective. The denominator is not:
5.01 TFLOP/s came from `14 cores x 128 ALUs x 2 x 1.398 GHz`,
and the clock and ALU count were never checked against a source. Every
number in this table scales linearly with it. Fix the constant in
`flops.py` before this table means anything.

| # | Shape | GFLOP | FLOP share | MLX MFU | MPS MFU |
|---:|---|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 8.59 | 0.4% | 21.4% | 9.6% |
| 2 | B1 D128 H4 S128 | 0.13 | 0.0% | 4.2% | 1.7% |
| 3 | B4 D128 H4 S128 | 0.54 | 0.0% | 9.7% | 4.7% |
| 4 | B16 D128 H4 S128 | 2.15 | 0.1% | 19.2% | 8.5% |
| 5 | B128 D128 H4 S128 | 17.18 | 0.9% | 19.5% | 7.0% |
| 6 | B10000 D128 H4 S128 | 1342.18 | 66.5% | 19.9% | 9.5% |
| 7 | B64 D32 H4 S128 | 0.94 | 0.0% | 4.0% | 1.6% |
| 8 | B64 D1024 H4 S128 | 429.50 | 21.3% | 63.4% | 51.7% |
| 9 | B64 D128 H1 S128 | 8.59 | 0.4% | 25.0% | 21.7% |
| 10 | B64 D128 H2 S128 | 8.59 | 0.4% | 25.1% | 13.3% |
| 11 | B64 D128 H16 S128 | 8.59 | 0.4% | 22.0% | 4.1% |
| 12 | B64 D128 H4 S32 | 1.74 | 0.1% | 16.3% | 10.2% |
| 13 | B64 D128 H4 S1024 | 188.98 | 9.4% | 49.8% | 6.7% |

- unweighted mean MLX MFU: **23.04%**
- FLOP-weighted mean MLX MFU: **31.99%**

Shape 6 alone is two thirds of the FLOP weight, so a FLOP-weighted
score is mostly a score on shape 6.

## Accuracy

Against the CPU baseline, at `atol=0.002` and `rtol=0.02`.

| # | MPS | MLX |
|---:|---|---|
| 1 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 2 | PASS `max_abs=1.07e-06` (0/16384 failed) | PASS `max_abs=1.07e-06` (0/16384 failed) |
| 3 | PASS `max_abs=1.19e-06` (0/65536 failed) | PASS `max_abs=1.31e-06` (0/65536 failed) |
| 4 | PASS `max_abs=1.19e-06` (0/262144 failed) | PASS `max_abs=1.31e-06` (0/262144 failed) |
| 5 | PASS `max_abs=1.43e-06` (0/2097152 failed) | PASS `max_abs=1.43e-06` (0/2097152 failed) |
| 6 | PASS `max_abs=1.91e-06` (0/163840000 failed) | PASS `max_abs=1.91e-06` (0/163840000 failed) |
| 7 | PASS `max_abs=1.19e-06` (0/262144 failed) | PASS `max_abs=1.19e-06` (0/262144 failed) |
| 8 | PASS `max_abs=2.86e-06` (0/8388608 failed) | PASS `max_abs=2.65e-06` (0/8388608 failed) |
| 9 | PASS `max_abs=1.67e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 10 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 11 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 12 | PASS `max_abs=1.91e-06` (0/262144 failed) | PASS `max_abs=1.31e-06` (0/262144 failed) |
| 13 | PASS `max_abs=1.67e-06` (0/8388608 failed) | PASS `max_abs=1.67e-06` (0/8388608 failed) |

## Summary

| Metric | Value |
|---|---:|
| Shapes scored | 13 |
| Median MLX speedup over CPU | **6.51x** |
| Range of MLX speedup | 2.68x to 24.91x |
| Median MLX rate | 0.996 TFLOP/s |
| Best MLX rate | 3.177 TFLOP/s |

## History

Every recorded sweep. The data is in [../profiling/history.jsonl](../profiling/history.jsonl).
Print it with `.venv/bin/python3 scoreboard.py --show-history`.

| When | Commit | Label | Shapes | Median speedup | Median TFLOP/s |
|---|---|---|---:|---:|---:|
| 2026-08-29T11:11:06 | `b37a558*` | attempts 1-11, full sweep | 13 | 5.01x | 0.837 |
| 2026-08-29T11:39:59 | `6f6cfce*` | pad head_dim to 64 to reach the fused SDPA kernel | 13 | 4.76x | 0.797 |
| 2026-08-29T12:05:46 | `6f6cfce*` | pad head_dim to 64, gated to rho<=2 and S>=4D (shape 13 only) | 13 | 4.85x | 0.849 |
| 2026-08-29T12:57:29 | `9ddce6d*` | hoist MLX steel_attention as a custom kernel at head_dim 8 and 32 | 13 | 6.51x | 0.996 |

A `*` on the commit means the working tree had uncommitted changes,
so that reading cannot be reproduced exactly.


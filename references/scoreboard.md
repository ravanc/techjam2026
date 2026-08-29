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
- sweep took 2.9 minutes
- **† marks a CPU reading that came from the cache, not from this
  sweep.** The earlier sweep ran under a different machine load, at
  a different chip temperature. The speedup beside a marked reading
  mixes two sweeps. Run without `--cpu-cache` before you report a
  number.

MFU appears once, in its own section, and it is provisional. See
[../flops.py](../flops.py).

## Speedup against the CPU baseline

| # | Shape | CPU ms | MPS ms | MLX ms | MPS vs CPU | **MLX vs CPU** | MLX vs MPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 61.252 † | 17.295 | 4.230 | 3.54x | **14.48x** | 4.09x |
| 2 | B1 D128 H4 S128 | 2.571 † | 1.575 | 0.631 | 1.63x | **4.07x** | 2.50x |
| 3 | B4 D128 H4 S128 | 7.169 † | 2.014 | 0.772 | 3.56x | **9.28x** | 2.61x |
| 4 | B16 D128 H4 S128 | 19.373 † | 4.706 | 1.396 | 4.12x | **13.88x** | 3.37x |
| 5 | B128 D128 H4 S128 | 139.502 † | 33.459 | 8.428 | 4.17x | **16.55x** | 3.97x |
| 6 | B10000 D128 H4 S128 | 14639.960 † | 2715.305 | 660.808 | 5.39x | **22.15x** | 4.11x |
| 7 | B64 D32 H4 S128 | 25.959 † | 11.572 | 1.145 | 2.24x | **22.68x** | 10.11x |
| 8 | B64 D1024 H4 S128 | 470.854 † | 165.298 | 127.573 | 2.85x | **3.69x** | 1.30x |
| 9 | B64 D128 H1 S128 | 35.099 † | 8.015 | 4.427 | 4.38x | **7.93x** | 1.81x |
| 10 | B64 D128 H2 S128 | 46.106 † | 12.879 | 4.290 | 3.58x | **10.75x** | 3.00x |
| 11 | B64 D128 H16 S128 | 130.289 † | 41.882 | 4.362 | 3.11x | **29.87x** | 9.60x |
| 12 | B64 D128 H4 S32 | 12.224 † | 3.393 | 1.382 | 3.60x | **8.84x** | 2.45x |
| 13 | B64 D128 H4 S1024 | 1899.325 † | 560.787 | 50.299 | 3.39x | **37.76x** | 11.15x |

## Achieved arithmetic rate

Model FLOPs divided by measured time. Compare against the matmul rates
measured on this machine: MLX float32 reaches 4.06 TFLOP/s on a square
matmul, torch MPS 4.16, torch CPU 1.42.

| # | Shape | GFLOP | CPU TFLOP/s | MPS TFLOP/s | **MLX TFLOP/s** | MLX token/s | Plan chosen |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | B64 D128 H4 S128 | 8.59 | 0.140 | 0.497 | **2.031** | 1,936,653 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True |
| 2 | B1 D128 H4 S128 | 0.13 | 0.052 | 0.085 | **0.213** | 202,812 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True |
| 3 | B4 D128 H4 S128 | 0.54 | 0.075 | 0.267 | **0.695** | 663,051 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True |
| 4 | B16 D128 H4 S128 | 2.15 | 0.111 | 0.456 | **1.538** | 1,466,787 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True |
| 5 | B128 D128 H4 S128 | 17.18 | 0.123 | 0.513 | **2.038** | 1,943,948 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True |
| 6 | B10000 D128 H4 S128 | 1342.18 | 0.092 | 0.494 | **2.031** | 1,937,023 | fuse_qkv=True causal_block=full batch_chunk=1024 pad_head_dim=none steel=True fast_ln=True |
| 7 | B64 D32 H4 S128 | 0.94 | 0.036 | 0.081 | **0.821** | 7,156,410 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True |
| 8 | B64 D1024 H4 S128 | 429.50 | 0.912 | 2.598 | **3.367** | 64,214 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=False |
| 9 | B64 D128 H1 S128 | 8.59 | 0.245 | 1.072 | **1.940** | 1,850,280 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True |
| 10 | B64 D128 H2 S128 | 8.59 | 0.186 | 0.667 | **2.003** | 1,909,761 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True |
| 11 | B64 D128 H16 S128 | 8.59 | 0.066 | 0.205 | **1.969** | 1,878,172 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True |
| 12 | B64 D128 H4 S32 | 1.74 | 0.143 | 0.514 | **1.262** | 1,481,798 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True |
| 13 | B64 D128 H4 S1024 | 188.98 | 0.099 | 0.337 | **3.757** | 1,302,917 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True |

## MFU — PROVISIONAL, do not quote without this note

> **PROVISIONAL: the MFU denominator (5.01 TFLOP/s) was asserted from memory, not verified. Every MFU below scales with it. The measured matmul rate of 4.06 TFLOP/s is the verified alternative.**

The numerator and the time are objective. The denominator is not:
5.01 TFLOP/s came from `14 cores x 128 ALUs x 2 x 1.398 GHz`,
and the clock and ALU count were never checked against a source. Every
number in this table scales linearly with it. Fix the constant in
`flops.py` before this table means anything.

| # | Shape | GFLOP | FLOP share | MLX MFU | MPS MFU |
|---:|---|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 8.59 | 0.4% | 40.5% | 9.9% |
| 2 | B1 D128 H4 S128 | 0.13 | 0.0% | 4.2% | 1.7% |
| 3 | B4 D128 H4 S128 | 0.54 | 0.0% | 13.9% | 5.3% |
| 4 | B16 D128 H4 S128 | 2.15 | 0.1% | 30.7% | 9.1% |
| 5 | B128 D128 H4 S128 | 17.18 | 0.9% | 40.7% | 10.2% |
| 6 | B10000 D128 H4 S128 | 1342.18 | 66.5% | 40.5% | 9.9% |
| 7 | B64 D32 H4 S128 | 0.94 | 0.0% | 16.4% | 1.6% |
| 8 | B64 D1024 H4 S128 | 429.50 | 21.3% | 67.2% | 51.9% |
| 9 | B64 D128 H1 S128 | 8.59 | 0.4% | 38.7% | 21.4% |
| 10 | B64 D128 H2 S128 | 8.59 | 0.4% | 40.0% | 13.3% |
| 11 | B64 D128 H16 S128 | 8.59 | 0.4% | 39.3% | 4.1% |
| 12 | B64 D128 H4 S32 | 1.74 | 0.1% | 25.2% | 10.3% |
| 13 | B64 D128 H4 S1024 | 188.98 | 9.4% | 75.0% | 6.7% |

- unweighted mean MLX MFU: **36.34%**
- FLOP-weighted mean MLX MFU: **49.38%**

Shape 6 alone is two thirds of the FLOP weight, so a FLOP-weighted
score is mostly a score on shape 6.

## Accuracy

Against the CPU baseline, at `atol=0.002` and `rtol=0.02`.

| # | MPS | MLX |
|---:|---|---|
| 1 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.67e-06` (0/1048576 failed) |
| 2 | PASS `max_abs=1.07e-06` (0/16384 failed) | PASS `max_abs=9.54e-07` (0/16384 failed) |
| 3 | PASS `max_abs=1.19e-06` (0/65536 failed) | PASS `max_abs=1.19e-06` (0/65536 failed) |
| 4 | PASS `max_abs=1.19e-06` (0/262144 failed) | PASS `max_abs=1.19e-06` (0/262144 failed) |
| 5 | PASS `max_abs=1.43e-06` (0/2097152 failed) | PASS `max_abs=1.67e-06` (0/2097152 failed) |
| 6 | PASS `max_abs=1.91e-06` (0/163840000 failed) | PASS `max_abs=1.91e-06` (0/163840000 failed) |
| 7 | PASS `max_abs=1.19e-06` (0/262144 failed) | PASS `max_abs=1.19e-06` (0/262144 failed) |
| 8 | PASS `max_abs=2.86e-06` (0/8388608 failed) | PASS `max_abs=2.65e-06` (0/8388608 failed) |
| 9 | PASS `max_abs=1.67e-06` (0/1048576 failed) | PASS `max_abs=1.67e-06` (0/1048576 failed) |
| 10 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 11 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 12 | PASS `max_abs=1.91e-06` (0/262144 failed) | PASS `max_abs=1.43e-06` (0/262144 failed) |
| 13 | PASS `max_abs=1.67e-06` (0/8388608 failed) | PASS `max_abs=1.55e-06` (0/8388608 failed) |

## Summary

| Metric | Value |
|---|---:|
| Shapes scored | 13 |
| Median MLX speedup over CPU | **13.88x** |
| Range of MLX speedup | 3.69x to 37.76x |
| Median MLX rate | 1.969 TFLOP/s |
| Best MLX rate | 3.757 TFLOP/s |

## History

Every recorded sweep. The data is in [../profiling/history.jsonl](../profiling/history.jsonl).
Print it with `.venv/bin/python3 scoreboard.py --show-history`.

| When | Commit | Label | Shapes | Median speedup | Median TFLOP/s |
|---|---|---|---:|---:|---:|
| 2026-08-29T11:11:06 | `b37a558*` | attempts 1-11, full sweep | 13 | 5.01x | 0.837 |
| 2026-08-29T11:39:59 | `6f6cfce*` | pad head_dim to 64 to reach the fused SDPA kernel | 13 | 4.76x | 0.797 |
| 2026-08-29T12:05:46 | `6f6cfce*` | pad head_dim to 64, gated to rho<=2 and S>=4D (shape 13 only) | 13 | 4.85x | 0.849 |
| 2026-08-29T12:57:29 | `9ddce6d*` | hoist MLX steel_attention as a custom kernel at head_dim 8 and 32 | 13 | 6.51x | 0.996 |
| 2026-08-29T13:18:56 | `92be893*` | re-measure after the steel_attention hoist, quieter machine | 13 | 5.61x | 1.108 |
| 2026-08-29T14:49:20 | `92be893*` | rows 27 and 28, ON BATTERY, do not compare: the CPU baseline ran 1.9x slow | 13 | 7.75x | 1.124 |
| 2026-08-29T15:00:25 | `92be893*` | rows 27 and 28, re-measured on AC power | 13 | 5.60x | 1.131 |
| 2026-08-29T15:13:00 | `92be893*` | row 29: mx.addmm for every projection | 13 | 6.08x | 1.247 |
| 2026-08-29T17:37:28 | `f9e3be7*` | row 31: single-pass LayerNorm kernel at d_model < 256 | 13 | 10.98x | 1.561 |
| 2026-08-29T18:06:35 | `7a97bbe*` | steel reads strided q,k,v and writes head-last: no copy | 13 | 13.88x | 1.969 |

A `*` on the commit means the working tree had uncommitted changes,
so that reading cannot be reproduced exactly.


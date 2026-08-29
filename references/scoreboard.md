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
- sweep took 8.4 minutes

MFU appears once, in its own section, and it is provisional. See
[../flops.py](../flops.py).

## Speedup against the CPU baseline

| # | Shape | CPU ms | MPS ms | MLX ms | MPS vs CPU | **MLX vs CPU** | MLX vs MPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 41.975 | 17.313 | 6.889 | 2.42x | **6.09x** | 2.51x |
| 2 | B1 D128 H4 S128 | 1.542 | 1.635 | 0.748 | 0.94x | **2.06x** | 2.19x |
| 3 | B4 D128 H4 S128 | 4.269 | 1.995 | 0.956 | 2.14x | **4.46x** | 2.09x |
| 4 | B16 D128 H4 S128 | 12.705 | 4.658 | 2.091 | 2.73x | **6.08x** | 2.23x |
| 5 | B128 D128 H4 S128 | 104.263 | 33.360 | 13.571 | 3.13x | **7.68x** | 2.46x |
| 6 | B10000 D128 H4 S128 | 13698.771 | 2680.809 | 1050.066 | 5.11x | **13.05x** | 2.55x |
| 7 | B64 D32 H4 S128 | 25.301 | 11.401 | 4.761 | 2.22x | **5.31x** | 2.39x |
| 8 | B64 D1024 H4 S128 | 471.858 | 165.513 | 128.463 | 2.85x | **3.67x** | 1.29x |
| 9 | B64 D128 H1 S128 | 31.898 | 7.953 | 6.268 | 4.01x | **5.09x** | 1.27x |
| 10 | B64 D128 H2 S128 | 40.476 | 12.919 | 6.151 | 3.13x | **6.58x** | 2.10x |
| 11 | B64 D128 H16 S128 | 125.794 | 41.952 | 7.061 | 3.00x | **17.82x** | 5.94x |
| 12 | B64 D128 H4 S32 | 11.052 | 3.232 | 2.007 | 3.42x | **5.51x** | 1.61x |
| 13 | B64 D128 H4 S1024 | 1858.001 | 564.827 | 69.288 | 3.29x | **26.82x** | 8.15x |

## Achieved arithmetic rate

Model FLOPs divided by measured time. Compare against the matmul rates
measured on this machine: MLX float32 reaches 4.06 TFLOP/s on a square
matmul, torch MPS 4.16, torch CPU 1.42.

| # | Shape | GFLOP | CPU TFLOP/s | MPS TFLOP/s | **MLX TFLOP/s** | MLX token/s | Plan chosen |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | B64 D128 H4 S128 | 8.59 | 0.205 | 0.496 | **1.247** | 1,189,178 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 2 | B1 D128 H4 S128 | 0.13 | 0.087 | 0.082 | **0.180** | 171,209 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 3 | B4 D128 H4 S128 | 0.54 | 0.126 | 0.269 | **0.561** | 535,320 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 4 | B16 D128 H4 S128 | 2.15 | 0.169 | 0.461 | **1.027** | 979,650 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 5 | B128 D128 H4 S128 | 17.18 | 0.165 | 0.515 | **1.266** | 1,207,284 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 6 | B10000 D128 H4 S128 | 1342.18 | 0.098 | 0.501 | **1.278** | 1,218,971 | fuse_qkv=True causal_block=full batch_chunk=1024 pad_head_dim=none steel=True |
| 7 | B64 D32 H4 S128 | 0.94 | 0.037 | 0.082 | **0.197** | 1,720,474 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 8 | B64 D1024 H4 S128 | 429.50 | 0.910 | 2.595 | **3.343** | 63,769 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False |
| 9 | B64 D128 H1 S128 | 8.59 | 0.269 | 1.080 | **1.370** | 1,306,904 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False |
| 10 | B64 D128 H2 S128 | 8.59 | 0.212 | 0.665 | **1.397** | 1,331,843 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False |
| 11 | B64 D128 H16 S128 | 8.59 | 0.068 | 0.205 | **1.217** | 1,160,158 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 12 | B64 D128 H4 S32 | 1.74 | 0.158 | 0.540 | **0.869** | 1,020,333 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |
| 13 | B64 D128 H4 S1024 | 188.98 | 0.102 | 0.335 | **2.727** | 945,854 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True |

## MFU — PROVISIONAL, do not quote without this note

> **PROVISIONAL: the MFU denominator (5.01 TFLOP/s) was asserted from memory, not verified. Every MFU below scales with it. The measured matmul rate of 4.06 TFLOP/s is the verified alternative.**

The numerator and the time are objective. The denominator is not:
5.01 TFLOP/s came from `14 cores x 128 ALUs x 2 x 1.398 GHz`,
and the clock and ALU count were never checked against a source. Every
number in this table scales linearly with it. Fix the constant in
`flops.py` before this table means anything.

| # | Shape | GFLOP | FLOP share | MLX MFU | MPS MFU |
|---:|---|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 8.59 | 0.4% | 24.9% | 9.9% |
| 2 | B1 D128 H4 S128 | 0.13 | 0.0% | 3.6% | 1.6% |
| 3 | B4 D128 H4 S128 | 0.54 | 0.0% | 11.2% | 5.4% |
| 4 | B16 D128 H4 S128 | 2.15 | 0.1% | 20.5% | 9.2% |
| 5 | B128 D128 H4 S128 | 17.18 | 0.9% | 25.3% | 10.3% |
| 6 | B10000 D128 H4 S128 | 1342.18 | 66.5% | 25.5% | 10.0% |
| 7 | B64 D32 H4 S128 | 0.94 | 0.0% | 3.9% | 1.6% |
| 8 | B64 D1024 H4 S128 | 429.50 | 21.3% | 66.7% | 51.8% |
| 9 | B64 D128 H1 S128 | 8.59 | 0.4% | 27.4% | 21.6% |
| 10 | B64 D128 H2 S128 | 8.59 | 0.4% | 27.9% | 13.3% |
| 11 | B64 D128 H16 S128 | 8.59 | 0.4% | 24.3% | 4.1% |
| 12 | B64 D128 H4 S32 | 1.74 | 0.1% | 17.4% | 10.8% |
| 13 | B64 D128 H4 S1024 | 188.98 | 9.4% | 54.4% | 6.7% |

- unweighted mean MLX MFU: **25.61%**
- FLOP-weighted mean MLX MFU: **36.98%**

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
| Median MLX speedup over CPU | **6.08x** |
| Range of MLX speedup | 2.06x to 26.82x |
| Median MLX rate | 1.247 TFLOP/s |
| Best MLX rate | 3.343 TFLOP/s |

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

A `*` on the commit means the working tree had uncommitted changes,
so that reading cannot be reproduced exactly.


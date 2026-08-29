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
- sweep took 3.2 minutes
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
| 1 | B64 D128 H4 S128 | 61.252 † | 17.309 | 3.788 | 3.54x | **16.17x** | 4.57x |
| 2 | B1 D128 H4 S128 | 2.571 † | 1.644 | 0.652 | 1.56x | **3.94x** | 2.52x |
| 3 | B4 D128 H4 S128 | 7.169 † | 2.024 | 0.757 | 3.54x | **9.47x** | 2.67x |
| 4 | B16 D128 H4 S128 | 19.373 † | 4.726 | 1.375 | 4.10x | **14.09x** | 3.44x |
| 5 | B128 D128 H4 S128 | 139.502 † | 33.535 | 7.359 | 4.16x | **18.96x** | 4.56x |
| 6 | B10000 D128 H4 S128 | 14639.960 † | 2669.151 | 567.842 | 5.48x | **25.78x** | 4.70x |
| 7 | B64 D32 H4 S128 | 25.959 † | 11.242 | 1.124 | 2.31x | **23.09x** | 10.00x |
| 8 | B64 D1024 H4 S128 | 470.854 † | 165.156 | 127.557 | 2.85x | **3.69x** | 1.29x |
| 9 | B64 D128 H1 S128 | 35.099 † | 7.932 | 3.923 | 4.43x | **8.95x** | 2.02x |
| 10 | B64 D128 H2 S128 | 46.106 † | 12.966 | 3.850 | 3.56x | **11.97x** | 3.37x |
| 11 | B64 D128 H16 S128 | 130.289 † | 41.966 | 3.850 | 3.10x | **33.84x** | 10.90x |
| 12 | B64 D128 H4 S32 | 12.224 † | 3.341 | 1.309 | 3.66x | **9.34x** | 2.55x |
| 13 | B64 D128 H4 S1024 | 1899.325 † | 562.926 | 45.231 | 3.37x | **41.99x** | 12.45x |

## Achieved arithmetic rate

Model FLOPs divided by measured time. Compare against the matmul rates
measured on this machine: MLX float32 reaches 4.06 TFLOP/s on a square
matmul, torch MPS 4.16, torch CPU 1.42.

| # | Shape | GFLOP | CPU TFLOP/s | MPS TFLOP/s | **MLX TFLOP/s** | MLX token/s | Plan chosen |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | B64 D128 H4 S128 | 8.59 | 0.140 | 0.496 | **2.267** | 2,162,381 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True |
| 2 | B1 D128 H4 S128 | 0.13 | 0.052 | 0.082 | **0.206** | 196,263 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True |
| 3 | B4 D128 H4 S128 | 0.54 | 0.075 | 0.265 | **0.709** | 676,168 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True |
| 4 | B16 D128 H4 S128 | 2.15 | 0.111 | 0.454 | **1.562** | 1,489,815 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True |
| 5 | B128 D128 H4 S128 | 17.18 | 0.123 | 0.512 | **2.335** | 2,226,528 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True |
| 6 | B10000 D128 H4 S128 | 1342.18 | 0.092 | 0.503 | **2.364** | 2,254,147 | fuse_qkv=True causal_block=full batch_chunk=1024 pad_head_dim=none steel=True fast_ln=True defer_bias=True |
| 7 | B64 D32 H4 S128 | 0.94 | 0.036 | 0.084 | **0.836** | 7,287,715 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True |
| 8 | B64 D1024 H4 S128 | 429.50 | 0.912 | 2.601 | **3.367** | 64,222 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=False defer_bias=False |
| 9 | B64 D128 H1 S128 | 8.59 | 0.245 | 1.083 | **2.190** | 2,088,098 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True defer_bias=True |
| 10 | B64 D128 H2 S128 | 8.59 | 0.186 | 0.662 | **2.231** | 2,127,631 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True defer_bias=True |
| 11 | B64 D128 H16 S128 | 8.59 | 0.066 | 0.205 | **2.231** | 2,127,608 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True |
| 12 | B64 D128 H4 S32 | 1.74 | 0.143 | 0.522 | **1.333** | 1,564,752 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True |
| 13 | B64 D128 H4 S1024 | 188.98 | 0.099 | 0.336 | **4.178** | 1,448,910 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True |

## MFU — PROVISIONAL, do not quote without this note

> **PROVISIONAL: the MFU denominator (5.01 TFLOP/s) was asserted from memory, not verified. Every MFU below scales with it. The measured matmul rate of 4.06 TFLOP/s is the verified alternative.**

The numerator and the time are objective. The denominator is not:
5.01 TFLOP/s came from `14 cores x 128 ALUs x 2 x 1.398 GHz`,
and the clock and ALU count were never checked against a source. Every
number in this table scales linearly with it. Fix the constant in
`flops.py` before this table means anything.

| # | Shape | GFLOP | FLOP share | MLX MFU | MPS MFU |
|---:|---|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 8.59 | 0.4% | 45.3% | 9.9% |
| 2 | B1 D128 H4 S128 | 0.13 | 0.0% | 4.1% | 1.6% |
| 3 | B4 D128 H4 S128 | 0.54 | 0.0% | 14.2% | 5.3% |
| 4 | B16 D128 H4 S128 | 2.15 | 0.1% | 31.2% | 9.1% |
| 5 | B128 D128 H4 S128 | 17.18 | 0.9% | 46.6% | 10.2% |
| 6 | B10000 D128 H4 S128 | 1342.18 | 66.5% | 47.2% | 10.0% |
| 7 | B64 D32 H4 S128 | 0.94 | 0.0% | 16.7% | 1.7% |
| 8 | B64 D1024 H4 S128 | 429.50 | 21.3% | 67.2% | 51.9% |
| 9 | B64 D128 H1 S128 | 8.59 | 0.4% | 43.7% | 21.6% |
| 10 | B64 D128 H2 S128 | 8.59 | 0.4% | 44.5% | 13.2% |
| 11 | B64 D128 H16 S128 | 8.59 | 0.4% | 44.5% | 4.1% |
| 12 | B64 D128 H4 S32 | 1.74 | 0.1% | 26.6% | 10.4% |
| 13 | B64 D128 H4 S1024 | 188.98 | 9.4% | 83.4% | 6.7% |

- unweighted mean MLX MFU: **39.63%**
- FLOP-weighted mean MLX MFU: **54.72%**

Shape 6 alone is two thirds of the FLOP weight, so a FLOP-weighted
score is mostly a score on shape 6.

## Accuracy

Against the CPU baseline, at `atol=0.002` and `rtol=0.02`.

| # | MPS | MLX |
|---:|---|---|
| 1 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.91e-06` (0/1048576 failed) |
| 2 | PASS `max_abs=1.07e-06` (0/16384 failed) | PASS `max_abs=1.19e-06` (0/16384 failed) |
| 3 | PASS `max_abs=1.19e-06` (0/65536 failed) | PASS `max_abs=1.19e-06` (0/65536 failed) |
| 4 | PASS `max_abs=1.19e-06` (0/262144 failed) | PASS `max_abs=1.43e-06` (0/262144 failed) |
| 5 | PASS `max_abs=1.43e-06` (0/2097152 failed) | PASS `max_abs=1.91e-06` (0/2097152 failed) |
| 6 | PASS `max_abs=1.91e-06` (0/163840000 failed) | PASS `max_abs=2.38e-06` (0/163840000 failed) |
| 7 | PASS `max_abs=1.19e-06` (0/262144 failed) | PASS `max_abs=1.19e-06` (0/262144 failed) |
| 8 | PASS `max_abs=2.86e-06` (0/8388608 failed) | PASS `max_abs=2.65e-06` (0/8388608 failed) |
| 9 | PASS `max_abs=1.67e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 10 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.67e-06` (0/1048576 failed) |
| 11 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 12 | PASS `max_abs=1.91e-06` (0/262144 failed) | PASS `max_abs=1.43e-06` (0/262144 failed) |
| 13 | PASS `max_abs=1.67e-06` (0/8388608 failed) | PASS `max_abs=1.67e-06` (0/8388608 failed) |

## Summary

| Metric | Value |
|---|---:|
| Shapes scored | 13 |
| Median MLX speedup over CPU | **14.09x** |
| Range of MLX speedup | 3.69x to 41.99x |
| Median MLX rate | 2.231 TFLOP/s |
| Best MLX rate | 4.178 TFLOP/s |

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
| 2026-08-29T18:41:52 | `6f510e7*` | row 36: defer the residual biases, fuse both residual adds into the GEMM C operand | 13 | 14.09x | 2.231 |

A `*` on the commit means the working tree had uncommitted changes,
so that reading cannot be reproduced exactly.


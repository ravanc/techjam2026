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
- sweep took 3.0 minutes
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
| 1 | B64 D128 H4 S128 | 61.252 † | 17.270 | 3.506 | 3.55x | **17.47x** | 4.93x |
| 2 | B1 D128 H4 S128 | 2.571 † | 1.617 | 0.651 | 1.59x | **3.95x** | 2.48x |
| 3 | B4 D128 H4 S128 | 7.169 † | 1.958 | 0.643 | 3.66x | **11.15x** | 3.04x |
| 4 | B16 D128 H4 S128 | 19.373 † | 4.529 | 1.174 | 4.28x | **16.50x** | 3.86x |
| 5 | B128 D128 H4 S128 | 139.502 † | 33.600 | 6.681 | 4.15x | **20.88x** | 5.03x |
| 6 | B10000 D128 H4 S128 | 15777.204 † | 2673.760 | 489.310 | 5.90x | **32.24x** | 5.46x |
| 7 | B64 D32 H4 S128 | 25.959 † | 11.404 | 1.055 | 2.28x | **24.59x** | 10.80x |
| 8 | B64 D1024 H4 S128 | 470.854 † | 165.374 | 124.911 | 2.85x | **3.77x** | 1.32x |
| 9 | B64 D128 H1 S128 | 35.099 † | 7.951 | 3.602 | 4.41x | **9.74x** | 2.21x |
| 10 | B64 D128 H2 S128 | 46.106 † | 12.975 | 3.592 | 3.55x | **12.84x** | 3.61x |
| 11 | B64 D128 H16 S128 | 130.289 † | 42.056 | 3.602 | 3.10x | **36.17x** | 11.68x |
| 12 | B64 D128 H4 S32 | 12.224 † | 3.265 | 1.115 | 3.74x | **10.96x** | 2.93x |
| 13 | B64 D128 H4 S1024 | 1899.325 † | 567.703 | 41.518 | 3.35x | **45.75x** | 13.67x |

## Achieved arithmetic rate

Model FLOPs divided by measured time. Compare against the matmul rates
measured on this machine: MLX float32 reaches 4.06 TFLOP/s on a square
matmul, torch MPS 4.16, torch CPU 1.42.

| # | Shape | GFLOP | CPU TFLOP/s | MPS TFLOP/s | **MLX TFLOP/s** | MLX token/s | Plan chosen |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | B64 D128 H4 S128 | 8.59 | 0.140 | 0.497 | **2.450** | 2,336,483 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 2 | B1 D128 H4 S128 | 0.13 | 0.052 | 0.083 | **0.206** | 196,652 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=none fuse_ln_qkv=none fuse_ln_ffn=none |
| 3 | B4 D128 H4 S128 | 0.54 | 0.075 | 0.274 | **0.835** | 795,932 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 4 | B16 D128 H4 S128 | 2.15 | 0.111 | 0.474 | **1.829** | 1,744,649 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 5 | B128 D128 H4 S128 | 17.18 | 0.123 | 0.511 | **2.572** | 2,452,465 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 6 | B10000 D128 H4 S128 | 1342.18 | 0.085 | 0.502 | **2.743** | 2,615,931 | fuse_qkv=True causal_block=full batch_chunk=1024 pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 7 | B64 D32 H4 S128 | 0.94 | 0.036 | 0.082 | **0.890** | 7,761,401 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=64x32x16x2x2 fuse_ln_qkv=64x32x16x2x2 fuse_ln_ffn=64x32x16x2x2 |
| 8 | B64 D1024 H4 S128 | 429.50 | 0.912 | 2.597 | **3.438** | 65,583 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=False defer_bias=False fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 9 | B64 D128 H1 S128 | 8.59 | 0.245 | 1.080 | **2.385** | 2,274,134 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 10 | B64 D128 H2 S128 | 8.59 | 0.186 | 0.662 | **2.392** | 2,280,743 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 11 | B64 D128 H16 S128 | 8.59 | 0.066 | 0.204 | **2.385** | 2,274,332 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 12 | B64 D128 H4 S32 | 1.74 | 0.143 | 0.534 | **1.565** | 1,836,943 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 13 | B64 D128 H4 S1024 | 188.98 | 0.099 | 0.333 | **4.552** | 1,578,491 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |

## MFU — PROVISIONAL, do not quote without this note

> **PROVISIONAL: the MFU denominator (5.01 TFLOP/s) was asserted from memory, not verified. Every MFU below scales with it. The measured matmul rate of 4.06 TFLOP/s is the verified alternative.**

The numerator and the time are objective. The denominator is not:
5.01 TFLOP/s came from `14 cores x 128 ALUs x 2 x 1.398 GHz`,
and the clock and ALU count were never checked against a source. Every
number in this table scales linearly with it. Fix the constant in
`flops.py` before this table means anything.

| # | Shape | GFLOP | FLOP share | MLX MFU | MPS MFU |
|---:|---|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 8.59 | 0.4% | 48.9% | 9.9% |
| 2 | B1 D128 H4 S128 | 0.13 | 0.0% | 4.1% | 1.7% |
| 3 | B4 D128 H4 S128 | 0.54 | 0.0% | 16.7% | 5.5% |
| 4 | B16 D128 H4 S128 | 2.15 | 0.1% | 36.5% | 9.5% |
| 5 | B128 D128 H4 S128 | 17.18 | 0.9% | 51.3% | 10.2% |
| 6 | B10000 D128 H4 S128 | 1342.18 | 66.5% | 54.8% | 10.0% |
| 7 | B64 D32 H4 S128 | 0.94 | 0.0% | 17.8% | 1.6% |
| 8 | B64 D1024 H4 S128 | 429.50 | 21.3% | 68.6% | 51.8% |
| 9 | B64 D128 H1 S128 | 8.59 | 0.4% | 47.6% | 21.6% |
| 10 | B64 D128 H2 S128 | 8.59 | 0.4% | 47.7% | 13.2% |
| 11 | B64 D128 H16 S128 | 8.59 | 0.4% | 47.6% | 4.1% |
| 12 | B64 D128 H4 S32 | 1.74 | 0.1% | 31.2% | 10.7% |
| 13 | B64 D128 H4 S1024 | 188.98 | 9.4% | 90.9% | 6.6% |

- unweighted mean MLX MFU: **43.36%**
- FLOP-weighted mean MLX MFU: **60.87%**

Shape 6 alone is two thirds of the FLOP weight, so a FLOP-weighted
score is mostly a score on shape 6.

## Accuracy

Against the CPU baseline, at `atol=0.002` and `rtol=0.02`.

| # | MPS | MLX |
|---:|---|---|
| 1 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.91e-06` (0/1048576 failed) |
| 2 | PASS `max_abs=1.07e-06` (0/16384 failed) | PASS `max_abs=1.19e-06` (0/16384 failed) |
| 3 | PASS `max_abs=1.19e-06` (0/65536 failed) | PASS `max_abs=1.43e-06` (0/65536 failed) |
| 4 | PASS `max_abs=1.19e-06` (0/262144 failed) | PASS `max_abs=1.43e-06` (0/262144 failed) |
| 5 | PASS `max_abs=1.43e-06` (0/2097152 failed) | PASS `max_abs=1.91e-06` (0/2097152 failed) |
| 6 | PASS `max_abs=1.91e-06` (0/163840000 failed) | PASS `max_abs=2.15e-06` (0/163840000 failed) |
| 7 | PASS `max_abs=1.19e-06` (0/262144 failed) | PASS `max_abs=1.07e-06` (0/262144 failed) |
| 8 | PASS `max_abs=2.86e-06` (0/8388608 failed) | PASS `max_abs=3.34e-06` (0/8388608 failed) |
| 9 | PASS `max_abs=1.67e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 10 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.67e-06` (0/1048576 failed) |
| 11 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 12 | PASS `max_abs=1.91e-06` (0/262144 failed) | PASS `max_abs=1.43e-06` (0/262144 failed) |
| 13 | PASS `max_abs=1.67e-06` (0/8388608 failed) | PASS `max_abs=1.91e-06` (0/8388608 failed) |

## Summary

| Metric | Value |
|---|---:|
| Shapes scored | 13 |
| Median MLX speedup over CPU | **16.50x** |
| Range of MLX speedup | 3.77x to 45.75x |
| Median MLX rate | 2.385 TFLOP/s |
| Best MLX rate | 4.552 TFLOP/s |

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
| 2026-08-29T19:19:40 | `43bbdae*` | row 33: fold GELU into the ffn_in GEMM epilogue | 13 | 14.50x | 2.288 |
| 2026-08-30T11:10:21 | `f43a374*` | row 46: absorb the LayerNorm into the GEMM weights, apply it in the epilogue | 13 | 16.50x | 2.385 |

A `*` on the commit means the working tree had uncommitted changes,
so that reading cannot be reproduced exactly.


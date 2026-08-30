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
| 1 | B64 D128 H4 S128 | 44.858 † | 17.769 | 3.592 | 2.52x | **12.49x** | 4.95x |
| 2 | B1 D128 H4 S128 | 1.603 † | 1.624 | 0.621 | 0.99x | **2.58x** | 2.61x |
| 3 | B4 D128 H4 S128 | 4.788 † | 2.108 | 0.750 | 2.27x | **6.38x** | 2.81x |
| 4 | B16 D128 H4 S128 | 12.964 † | 4.841 | 1.246 | 2.68x | **10.41x** | 3.89x |
| 5 | B128 D128 H4 S128 | 120.621 † | 34.414 | 6.944 | 3.51x | **17.37x** | 4.96x |
| 6 | B10000 D128 H4 S128 | 15777.204 † | 2747.871 | 479.310 | 5.74x | **32.92x** | 5.73x |
| 7 | B64 D32 H4 S128 | 27.904 † | 11.409 | 1.019 | 2.45x | **27.38x** | 11.19x |
| 8 | B64 D1024 H4 S128 | 470.782 † | 165.999 | 119.931 | 2.84x | **3.93x** | 1.38x |
| 9 | B64 D128 H1 S128 | 35.961 † | 7.982 | 3.657 | 4.51x | **9.83x** | 2.18x |
| 10 | B64 D128 H2 S128 | 51.488 † | 12.892 | 3.471 | 3.99x | **14.83x** | 3.71x |
| 11 | B64 D128 H16 S128 | 141.339 † | 42.267 | 3.578 | 3.34x | **39.50x** | 11.81x |
| 12 | B64 D128 H4 S32 | 12.794 † | 3.240 | 1.115 | 3.95x | **11.48x** | 2.91x |
| 13 | B64 D128 H4 S1024 | 1974.807 † | 584.371 | 42.995 | 3.38x | **45.93x** | 13.59x |

## Achieved arithmetic rate

Model FLOPs divided by measured time. Compare against the matmul rates
measured on this machine: MLX float32 reaches 4.06 TFLOP/s on a square
matmul, torch MPS 4.16, torch CPU 1.42.

| # | Shape | GFLOP | CPU TFLOP/s | MPS TFLOP/s | **MLX TFLOP/s** | MLX token/s | Plan chosen |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | B64 D128 H4 S128 | 8.59 | 0.191 | 0.483 | **2.392** | 2,280,716 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 2 | B1 D128 H4 S128 | 0.13 | 0.084 | 0.083 | **0.216** | 206,064 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=none fuse_ln_qkv=none fuse_ln_ffn=none |
| 3 | B4 D128 H4 S128 | 0.54 | 0.112 | 0.255 | **0.716** | 682,648 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 4 | B16 D128 H4 S128 | 2.15 | 0.166 | 0.444 | **1.724** | 1,643,769 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 5 | B128 D128 H4 S128 | 17.18 | 0.142 | 0.499 | **2.474** | 2,359,574 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 6 | B10000 D128 H4 S128 | 1342.18 | 0.085 | 0.488 | **2.800** | 2,670,507 | fuse_qkv=True causal_block=full batch_chunk=1024 pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 7 | B64 D32 H4 S128 | 0.94 | 0.034 | 0.082 | **0.922** | 8,037,779 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=64x32x16x2x2 fuse_ln_qkv=64x32x16x2x2 fuse_ln_ffn=64x32x16x2x2 |
| 8 | B64 D1024 H4 S128 | 429.50 | 0.912 | 2.587 | **3.581** | 68,306 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=False defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 9 | B64 D128 H1 S128 | 8.59 | 0.239 | 1.076 | **2.349** | 2,240,241 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 10 | B64 D128 H2 S128 | 8.59 | 0.167 | 0.666 | **2.475** | 2,360,155 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 11 | B64 D128 H16 S128 | 8.59 | 0.061 | 0.203 | **2.400** | 2,289,294 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 12 | B64 D128 H4 S32 | 1.74 | 0.136 | 0.539 | **1.565** | 1,837,046 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |
| 13 | B64 D128 H4 S1024 | 188.98 | 0.096 | 0.323 | **4.395** | 1,524,256 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 |

## MFU — PROVISIONAL, do not quote without this note

> **PROVISIONAL: the MFU denominator (5.01 TFLOP/s) was asserted from memory, not verified. Every MFU below scales with it. The measured matmul rate of 4.06 TFLOP/s is the verified alternative.**

The numerator and the time are objective. The denominator is not:
5.01 TFLOP/s came from `14 cores x 128 ALUs x 2 x 1.398 GHz`,
and the clock and ALU count were never checked against a source. Every
number in this table scales linearly with it. Fix the constant in
`flops.py` before this table means anything.

| # | Shape | GFLOP | FLOP share | MLX MFU | MPS MFU |
|---:|---|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 8.59 | 0.4% | 47.7% | 9.6% |
| 2 | B1 D128 H4 S128 | 0.13 | 0.0% | 4.3% | 1.6% |
| 3 | B4 D128 H4 S128 | 0.54 | 0.0% | 14.3% | 5.1% |
| 4 | B16 D128 H4 S128 | 2.15 | 0.1% | 34.4% | 8.9% |
| 5 | B128 D128 H4 S128 | 17.18 | 0.9% | 49.4% | 10.0% |
| 6 | B10000 D128 H4 S128 | 1342.18 | 66.5% | 55.9% | 9.7% |
| 7 | B64 D32 H4 S128 | 0.94 | 0.0% | 18.4% | 1.6% |
| 8 | B64 D1024 H4 S128 | 429.50 | 21.3% | 71.5% | 51.6% |
| 9 | B64 D128 H1 S128 | 8.59 | 0.4% | 46.9% | 21.5% |
| 10 | B64 D128 H2 S128 | 8.59 | 0.4% | 49.4% | 13.3% |
| 11 | B64 D128 H16 S128 | 8.59 | 0.4% | 47.9% | 4.1% |
| 12 | B64 D128 H4 S32 | 1.74 | 0.1% | 31.2% | 10.8% |
| 13 | B64 D128 H4 S1024 | 188.98 | 9.4% | 87.7% | 6.5% |

- unweighted mean MLX MFU: **43.01%**
- FLOP-weighted mean MLX MFU: **61.93%**

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
| 8 | PASS `max_abs=2.86e-06` (0/8388608 failed) | PASS `max_abs=3.22e-06` (0/8388608 failed) |
| 9 | PASS `max_abs=1.67e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 10 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.67e-06` (0/1048576 failed) |
| 11 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 12 | PASS `max_abs=1.91e-06` (0/262144 failed) | PASS `max_abs=1.43e-06` (0/262144 failed) |
| 13 | PASS `max_abs=1.67e-06` (0/8388608 failed) | PASS `max_abs=1.91e-06` (0/8388608 failed) |

## Summary

| Metric | Value |
|---|---:|
| Shapes scored | 13 |
| Median MLX speedup over CPU | **12.49x** |
| Range of MLX speedup | 2.58x to 45.93x |
| Median MLX rate | 2.392 TFLOP/s |
| Best MLX rate | 4.395 TFLOP/s |

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
| 2026-08-30T11:26:17 | `b352c4f*` | row 37 cheap form: defer the residual biases wherever row 46 covers both norms, so shape 8 defers too | 13 | 12.67x | 2.345 |
| 2026-08-30T11:30:16 | `b352c4f*` | row 37 cheap form, re-measured: shape 8 defers its residual biases | 13 | 12.49x | 2.392 |

A `*` on the commit means the working tree had uncommitted changes,
so that reading cannot be reproduced exactly.


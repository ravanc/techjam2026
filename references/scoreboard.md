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
- sweep took 4.3 minutes
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
| 1 | B64 D128 H4 S128 | 44.227 † | 17.373 | 3.306 | 2.55x | **13.38x** | 5.25x |
| 2 | B1 D128 H4 S128 | 1.695 † | 2.623 | 0.591 | 0.65x | **2.87x** | 4.43x |
| 3 | B4 D128 H4 S128 | 4.700 † | 2.015 | 0.629 | 2.33x | **7.47x** | 3.20x |
| 4 | B16 D128 H4 S128 | 13.364 † | 4.610 | 1.169 | 2.90x | **11.43x** | 3.94x |
| 5 | B128 D128 H4 S128 | 114.944 † | 33.662 | 6.034 | 3.41x | **19.05x** | 5.58x |
| 6 | B10000 D128 H4 S128 | 14946.867 † | 2751.531 | 447.041 | 5.43x | **33.44x** | 6.15x |
| 7 | B64 D32 H4 S128 | 27.267 | 11.504 | 1.004 | 2.37x | **27.17x** | 11.46x |
| 8 | B64 D1024 H4 S128 | 473.041 † | 165.559 | 117.968 | 2.86x | **4.01x** | 1.40x |
| 9 | B64 D128 H1 S128 | 32.895 | 8.093 | 3.442 | 4.06x | **9.56x** | 2.35x |
| 10 | B64 D128 H2 S128 | 42.969 | 13.004 | 3.319 | 3.30x | **12.94x** | 3.92x |
| 11 | B64 D128 H16 S128 | 133.328 | 42.273 | 3.427 | 3.15x | **38.91x** | 12.34x |
| 12 | B64 D128 H4 S32 | 10.042 | 3.390 | 1.132 | 2.96x | **8.87x** | 3.00x |
| 13 | B64 D128 H4 S1024 | 1918.386 | 569.696 | 39.702 | 3.37x | **48.32x** | 14.35x |

## Achieved arithmetic rate

Model FLOPs divided by measured time. Compare against the matmul rates
measured on this machine: MLX float32 reaches 4.06 TFLOP/s on a square
matmul, torch MPS 4.16, torch CPU 1.42.

| # | Shape | GFLOP | CPU TFLOP/s | MPS TFLOP/s | **MLX TFLOP/s** | MLX token/s | Plan chosen |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | B64 D128 H4 S128 | 8.59 | 0.194 | 0.494 | **2.598** | 2,477,826 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 2 | B1 D128 H4 S128 | 0.13 | 0.079 | 0.051 | **0.227** | 216,414 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=none fuse_ln_qkv=none fuse_ln_ffn=none stats_out=none stats_ffn=none final_ln=32x128x16x4x1 |
| 3 | B4 D128 H4 S128 | 0.54 | 0.114 | 0.266 | **0.853** | 813,560 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 4 | B16 D128 H4 S128 | 2.15 | 0.161 | 0.466 | **1.837** | 1,752,330 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 5 | B128 D128 H4 S128 | 17.18 | 0.149 | 0.510 | **2.847** | 2,715,111 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 6 | B10000 D128 H4 S128 | 1342.18 | 0.090 | 0.488 | **3.002** | 2,863,274 | fuse_qkv=True causal_block=full batch_chunk=1024 pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 7 | B64 D32 H4 S128 | 0.94 | 0.034 | 0.082 | **0.936** | 8,161,566 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=64x32x16x2x2 fuse_ln_qkv=64x32x16x2x2 fuse_ln_ffn=64x32x16x2x2 stats_out=64x32x16x2x2 stats_ffn=64x32x16x2x2 final_ln=32x32x16x4x1 |
| 8 | B64 D1024 H4 S128 | 429.50 | 0.908 | 2.594 | **3.641** | 69,442 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=False defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=none |
| 9 | B64 D128 H1 S128 | 8.59 | 0.261 | 1.061 | **2.496** | 2,380,113 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 10 | B64 D128 H2 S128 | 8.59 | 0.200 | 0.661 | **2.588** | 2,467,934 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 11 | B64 D128 H16 S128 | 8.59 | 0.064 | 0.203 | **2.507** | 2,390,473 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 12 | B64 D128 H4 S32 | 1.74 | 0.174 | 0.515 | **1.542** | 1,809,787 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 13 | B64 D128 H4 S1024 | 188.98 | 0.099 | 0.332 | **4.760** | 1,650,710 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |

## MFU

> **The MFU denominator is 4.946 TFLOP/s = 14 cores x 128 ALUs x 2 x 1.380 GHz. The core count and the 1380 MHz top DVFS state are read from this machine; the ALU count is bounded below at 105 by the measured matmul rate. The GPU does NOT hold 1380 MHz: a saturating loop sustains 3.92 TFLOP/s and a matmul reaches 4.06, both about 1.1 GHz. So 82% is the practical ceiling of this column.**

The numerator and the time are objective, and the denominator is now
sourced: 4.946 TFLOP/s is `14 cores x 128 ALUs x 2 x 1.380 GHz`.
The core count comes from `system_profiler`, and the 1380 MHz top
state from the GPU DVFS table `voltage-states9` in the pmgr device
tree. The ALU count is not published; the measured matmul rate bounds
it below at 105, so 128 is the only plausible width. See `flops.py`.

**Read this column against 82%, not 100%.** The GPU does not hold
1380 MHz. A saturating FMA loop sustains 3.92 TFLOP/s and a plain
matmul reaches 4.06, and both imply about 1.1 GHz. No kernel can
remove that gap.

**A causal shape can print more than 82%.** The FLOP model counts the
full `S x S` attention, because that is what the baseline computes,
while the optimized path skips the upper triangle. Shape 13 is
credited with 188.98 GFLOP and executes 120.33, so its 91.9% is 58.5%
on the work it really runs. The ceiling applies to executed work.

| # | Shape | GFLOP | FLOP share | MLX MFU | MPS MFU |
|---:|---|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 8.59 | 0.4% | 52.5% | 10.0% |
| 2 | B1 D128 H4 S128 | 0.13 | 0.0% | 4.6% | 1.0% |
| 3 | B4 D128 H4 S128 | 0.54 | 0.0% | 17.2% | 5.4% |
| 4 | B16 D128 H4 S128 | 2.15 | 0.1% | 37.2% | 9.4% |
| 5 | B128 D128 H4 S128 | 17.18 | 0.9% | 57.6% | 10.3% |
| 6 | B10000 D128 H4 S128 | 1342.18 | 66.5% | 60.7% | 9.9% |
| 7 | B64 D32 H4 S128 | 0.94 | 0.0% | 18.9% | 1.7% |
| 8 | B64 D1024 H4 S128 | 429.50 | 21.3% | 73.6% | 52.5% |
| 9 | B64 D128 H1 S128 | 8.59 | 0.4% | 50.5% | 21.5% |
| 10 | B64 D128 H2 S128 | 8.59 | 0.4% | 52.3% | 13.4% |
| 11 | B64 D128 H16 S128 | 8.59 | 0.4% | 50.7% | 4.1% |
| 12 | B64 D128 H4 S32 | 1.74 | 0.1% | 31.2% | 10.4% |
| 13 | B64 D128 H4 S1024 | 188.98 | 9.4% | 96.2% | 6.7% |

- unweighted mean MLX MFU: **46.40%**
- FLOP-weighted mean MLX MFU: **66.51%**

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
| 6 | PASS `max_abs=1.91e-06` (0/163840000 failed) | PASS `max_abs=1.91e-06` (0/163840000 failed) |
| 7 | PASS `max_abs=1.19e-06` (0/262144 failed) | PASS `max_abs=1.19e-06` (0/262144 failed) |
| 8 | PASS `max_abs=2.86e-06` (0/8388608 failed) | PASS `max_abs=3.46e-06` (0/8388608 failed) |
| 9 | PASS `max_abs=1.67e-06` (0/1048576 failed) | PASS `max_abs=1.67e-06` (0/1048576 failed) |
| 10 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.67e-06` (0/1048576 failed) |
| 11 | PASS `max_abs=1.43e-06` (0/1048576 failed) | PASS `max_abs=1.43e-06` (0/1048576 failed) |
| 12 | PASS `max_abs=1.91e-06` (0/262144 failed) | PASS `max_abs=1.43e-06` (0/262144 failed) |
| 13 | PASS `max_abs=1.67e-06` (0/8388608 failed) | PASS `max_abs=1.91e-06` (0/8388608 failed) |

## Summary

| Metric | Value |
|---|---:|
| Shapes scored | 13 |
| Median MLX speedup over CPU | **12.94x** |
| Range of MLX speedup | 2.87x to 48.32x |
| Median MLX rate | 2.507 TFLOP/s |
| Best MLX rate | 4.760 TFLOP/s |

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
| 2026-08-30T11:38:23 | `e2a8d22*` | clean baseline after rows 46 and 37, quiet machine | 13 | 12.97x | 2.342 |
| 2026-08-30T12:08:18 | `32a7aa7*` | MFU denominator corrected: 1.380 GHz from the GPU DVFS table, 5.01 -> 4.946 | 13 | 12.93x | 2.372 |
| 2026-08-30T16:39:54 | `32a7aa7*` | Row 47: take the LayerNorm statistics in the GEMM epilogue | 13 | 13.34x | 2.525 |
| 2026-08-31T01:36:59 | `e63aea5*` | profiling run at e63aea5, clean tree after row 48 revert | 13 | 13.13x | 2.484 |
| 2026-08-31T02:38:45 | `2a85fc5*` | Row 50: apply the final LayerNorm in the last ffn_out epilogue | 13 | 12.94x | 2.507 |

A `*` on the commit means the working tree had uncommitted changes,
so that reading cannot be reproduced exactly.


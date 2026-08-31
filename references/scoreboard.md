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
- sweep took 3.5 minutes
- **† marks a CPU reading that came from the cache, not from this
  sweep.** The earlier sweep ran under a different machine load, at
  a different chip temperature. The speedup beside a marked reading
  mixes two sweeps. Run without `--cpu-cache` before you report a
  number.

MFU appears once, in its own section, and it is provisional. See
[../flops.py](../flops.py).

**This table holds shapes 1 to 13.** Shape 14 is not missing by
accident: its baseline cannot run, so it has no CPU column and no
speedup. It runs under `shape14_harness.py`, and its reading lives in
`../profiling/results/shape14.json`. See `OPTIMIZATIONS.md` row 55
and [test-shapes.md](test-shapes.md).

## Speedup against the CPU baseline

| # | Shape | CPU ms | MPS ms | MLX ms | MPS vs CPU | **MLX vs CPU** | MLX vs MPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 44.227 † | 17.440 | 3.204 | 2.54x | **13.80x** | 5.44x |
| 2 | B1 D128 H4 S128 | 1.695 † | 1.526 | 0.610 | 1.11x | **2.78x** | 2.50x |
| 3 | B4 D128 H4 S128 | 4.700 † | 2.016 | 0.680 | 2.33x | **6.91x** | 2.96x |
| 4 | B16 D128 H4 S128 | 13.364 † | 4.703 | 1.301 | 2.84x | **10.27x** | 3.62x |
| 5 | B128 D128 H4 S128 | 114.944 † | 33.709 | 6.019 | 3.41x | **19.10x** | 5.60x |
| 6 | B10000 D128 H4 S128 | 14946.867 † | 2771.854 | 454.810 | 5.39x | **32.86x** | 6.09x |
| 7 | B64 D32 H4 S128 | 27.267 † | 11.520 | 1.012 | 2.37x | **26.95x** | 11.39x |
| 8 | B64 D1024 H4 S128 | 473.041 † | 167.571 | 119.196 | 2.82x | **3.97x** | 1.41x |
| 9 | B64 D128 H1 S128 | 32.895 † | 8.103 | 3.447 | 4.06x | **9.54x** | 2.35x |
| 10 | B64 D128 H2 S128 | 42.969 † | 13.095 | 3.374 | 3.28x | **12.74x** | 3.88x |
| 11 | B64 D128 H16 S128 | 133.328 † | 42.513 | 3.425 | 3.14x | **38.93x** | 12.41x |
| 12 | B64 D128 H4 S32 | 10.042 † | 3.467 | 1.183 | 2.90x | **8.49x** | 2.93x |
| 13 | B64 D128 H4 S1024 | 1918.386 † | 577.361 | 40.689 | 3.32x | **47.15x** | 14.19x |

## Achieved arithmetic rate

Model FLOPs divided by measured time. Compare against the matmul rates
measured on this machine: MLX float32 reaches 4.06 TFLOP/s on a square
matmul, torch MPS 4.16, torch CPU 1.42.

| # | Shape | GFLOP | CPU TFLOP/s | MPS TFLOP/s | **MLX TFLOP/s** | MLX token/s | Plan chosen |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | B64 D128 H4 S128 | 8.59 | 0.194 | 0.493 | **2.681** | 2,556,621 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 2 | B1 D128 H4 S128 | 0.13 | 0.079 | 0.088 | **0.220** | 209,671 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=none fuse_ln_qkv=none fuse_ln_ffn=none stats_out=none stats_ffn=none final_ln=32x128x16x4x1 |
| 3 | B4 D128 H4 S128 | 0.54 | 0.114 | 0.266 | **0.790** | 753,126 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 4 | B16 D128 H4 S128 | 2.15 | 0.161 | 0.457 | **1.651** | 1,574,451 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 5 | B128 D128 H4 S128 | 17.18 | 0.149 | 0.510 | **2.854** | 2,722,009 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 6 | B10000 D128 H4 S128 | 1342.18 | 0.090 | 0.484 | **2.951** | 2,814,361 | fuse_qkv=True causal_block=full batch_chunk=1024 pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 7 | B64 D32 H4 S128 | 0.94 | 0.034 | 0.082 | **0.929** | 8,098,030 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=64x32x16x2x2 fuse_ln_qkv=64x32x16x2x2 fuse_ln_ffn=64x32x16x2x2 stats_out=64x32x16x2x2 stats_ffn=64x32x16x2x2 final_ln=32x32x16x4x1 |
| 8 | B64 D1024 H4 S128 | 429.50 | 0.908 | 2.563 | **3.603** | 68,727 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=False defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=none |
| 9 | B64 D128 H1 S128 | 8.59 | 0.261 | 1.060 | **2.492** | 2,376,488 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 10 | B64 D128 H2 S128 | 8.59 | 0.200 | 0.656 | **2.546** | 2,428,218 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=False fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 11 | B64 D128 H16 S128 | 8.59 | 0.064 | 0.202 | **2.508** | 2,391,985 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 12 | B64 D128 H4 S32 | 1.74 | 0.174 | 0.503 | **1.475** | 1,731,070 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |
| 13 | B64 D128 H4 S1024 | 188.98 | 0.099 | 0.327 | **4.644** | 1,610,638 | fuse_qkv=True causal_block=full batch_chunk=none pad_head_dim=none steel=True fast_ln=True defer_bias=True fuse_gelu=32x64x16x2x2 fuse_ln_qkv=32x64x16x2x2 fuse_ln_ffn=32x64x16x2x2 stats_out=32x64x16x2x2 stats_ffn=32x64x16x2x2 final_ln=32x128x16x4x1 |

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
while the optimized path skips the upper triangle. So the table
carries BOTH: `counted` is the graded number, and `executed` divides
by the work the kernel really runs. **Read `executed` against 82%.**

The gap is not a constant factor. It is widest at shape 13, which executes 63.7% of what it is credited with, so its 93.9% is 59.8% on real work. A shape whose attention is a small share of the whole barely moves.

| # | Shape | GFLOP counted | GFLOP executed | FLOP share | MLX MFU counted | **MLX MFU executed** | MPS MFU counted |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 | 8.59 | 7.52 | 0.4% | 54.2% | **47.5%** | 10.0% |
| 2 | B1 D128 H4 S128 | 0.13 | 0.12 | 0.0% | 4.4% | **3.9%** | 1.8% |
| 3 | B4 D128 H4 S128 | 0.54 | 0.47 | 0.0% | 16.0% | **14.0%** | 5.4% |
| 4 | B16 D128 H4 S128 | 2.15 | 1.88 | 0.1% | 33.4% | **29.2%** | 9.2% |
| 5 | B128 D128 H4 S128 | 17.18 | 15.05 | 0.9% | 57.7% | **50.6%** | 10.3% |
| 6 | B10000 D128 H4 S128 | 1342.18 | 1175.72 | 66.5% | 59.7% | **52.3%** | 9.8% |
| 7 | B64 D32 H4 S128 | 0.94 | 0.67 | 0.0% | 18.8% | **13.5%** | 1.6% |
| 8 | B64 D1024 H4 S128 | 429.50 | 420.97 | 21.3% | 72.9% | **71.4%** | 51.8% |
| 9 | B64 D128 H1 S128 | 8.59 | 7.52 | 0.4% | 50.4% | **44.1%** | 21.4% |
| 10 | B64 D128 H2 S128 | 8.59 | 7.52 | 0.4% | 51.5% | **45.1%** | 13.3% |
| 11 | B64 D128 H16 S128 | 8.59 | 7.52 | 0.4% | 50.7% | **44.4%** | 4.1% |
| 12 | B64 D128 H4 S32 | 1.74 | 1.68 | 0.1% | 29.8% | **28.7%** | 10.2% |
| 13 | B64 D128 H4 S1024 | 188.98 | 120.33 | 9.4% | 93.9% | **59.8%** | 6.6% |

- unweighted mean MLX MFU: counted **45.64%**, executed **38.80%**
- FLOP-weighted mean MLX MFU: counted **65.44%**, executed **56.84%**

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
| Median MLX speedup over CPU | **12.74x** |
| Range of MLX speedup | 2.78x to 47.15x |
| Median MLX rate | 2.508 TFLOP/s |
| Best MLX rate | 4.644 TFLOP/s |


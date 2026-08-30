#!/usr/bin/env python3
"""
FLOP counting for the transformer of `torch_transformer_benchmark.py`.

This module holds one derived quantity and one set of measurements. Nothing
here depends on a number that was not either measured on this machine or
derived from the model definition.

Print the FLOP cost of every appendix shape:

    .venv/bin/python3 flops.py

Re-measure the matmul rates:

    .venv/bin/python3 flops.py --peak

## MFU, and where its denominator comes from

Model FLOPs Utilization divides the model rate by the hardware peak. Every
MFU number scales linearly with that peak, so the peak must be sourced.

An earlier version used 5.01 TFLOP/s from
`14 cores x 128 ALUs x 2 x 1.398 GHz`, and the clock and the ALU count were
asserted from memory. **Both are now checked on this machine.** See
`PEAK_TFLOPS` below and `references/machine.md`.
"""

from __future__ import annotations

from typing import Dict

from torch_transformer_benchmark import TransformerConfig

# Matmul rates MEASURED on this machine, in TFLOP/s. A square matmul at
# n=2048 and n=4096, median of 7 rounds, `mx.synchronize()` on both sides.
# These are observations, not specifications.
#
# Reproduce with `.venv/bin/python3 flops.py --peak`.
MEASURED_TFLOPS: Dict[str, float] = {
    "mlx_gpu_float32": 4.06,
    "mlx_gpu_float16": 4.61,
    "mlx_gpu_bfloat16": 4.61,
    "torch_mps_float32": 4.16,
    "torch_cpu_float32": 1.42,
}


def model_flops(config: TransformerConfig, causal_aware: bool = False) -> int:
    """
    Forward-pass matmul FLOPs of one call, for the whole batch.

    Per layer, per token, with D = d_model, F = ffn_dim, S = seq_len:

        q, k, v projections     3 * 2 * D * D
        scores  Q @ K^T             2 * S * D
        context probs @ V           2 * S * D
        output projection           2 * D * D
        ffn_in                      2 * D * F
        ffn_out                     2 * F * D
        -------------------------------------
        total                8*D*D + 4*S*D + 4*D*F

    Multiply by `batch_size * seq_len * num_layers`.

    Matmuls only. LayerNorm, softmax, GELU and the residual adds are not
    counted. That is the usual convention.

    `causal_aware=False` is the default: it counts the full S x S attention,
    which is what `BaselineSelfAttention` calculates. `causal_aware=True`
    counts only the lower triangle, which is what the blocked path of
    `UserOptimizedTransformer` actually calculates.
    """
    d_model = config.d_model
    tokens = config.batch_size * config.seq_len

    projection = 8 * d_model * d_model
    ffn = 4 * d_model * config.ffn_dim

    attention_span = config.seq_len
    if causal_aware:
        # Row i attends to i + 1 keys. The mean over rows is (S + 1) / 2.
        attention_span = (config.seq_len + 1) / 2.0
    attention = 4 * attention_span * d_model

    return int(tokens * config.num_layers * (projection + attention + ffn))


# ---------------------------------------------------------------------------
# The MFU denominator. Every term below is checked on this machine.
# ---------------------------------------------------------------------------
# peak = cores x ALUs per core x 2 flop per FMA x clock
#      = 14 x 128 x 2 x 1.380 GHz
#      = 4.946 TFLOP/s
#
# | term | value | how it was checked |
# |---|---|---|
# | cores | 14 | `system_profiler SPDisplaysDataType`, and `gpu-core-count` in the AGXAccelerator IORegistry node |
# | clock | 1.380 GHz | the GPU DVFS table `voltage-states9` in the pmgr device tree. Its top state is 1380 MHz |
# | ALUs per core | 128 | bounded by measurement, see below |
# | flop per FMA | 2 | a fused multiply-add is one multiply and one add |
#
# **The clock was 1.398 GHz and that was wrong.** The hardware's own table
# says 1380 MHz. Read it again with:
#
#     ioreg -lw0 -p IODeviceTree -n pmgr | grep -o '"voltage-states9" = <[0-9a-f]*>'
#
# It decodes as 14 pairs of little-endian uint32, `{frequency Hz, millivolts}`:
# 0, 338, 618, 796, 832, 924, 952, 1056, 1064, 1182, 1182, 1312, 1242, 1380 MHz.
#
# **The ALU count is bounded, not read.** No table on this machine states it.
# But a measured rate and a maximum clock give a lower bound: a plain matmul
# reaches 4.06 TFLOP/s (`flops.py --peak`), and the GPU cannot run above
# 1380 MHz, so
#
#     ALUs per core >= 4.06e12 / (14 x 2 x 1.380e9) = 105.1
#
# 64 is therefore impossible and 128 is the next width an Apple GPU core has.
# A pure FMA loop agrees: `profiling/alu_peak.py` reaches 3.92 TFLOP/s, which
# is 1.09 GHz at 128 ALUs, and 1.09 GHz sits on a real DVFS state.
#
# ---------------------------------------------------------------------------
# THE MACHINE DOES NOT HOLD 1380 MHz. Read this before you read an MFU.
# ---------------------------------------------------------------------------
# `PEAK_TFLOPS` is the top of the DVFS table. Nothing measured here has ever
# reached it. The two saturating measurements both imply about 1.1 GHz:
#
# | what | rate | implied clock at 128 ALUs |
# |---|---:|---:|
# | pure FMA loop, sustained (`profiling/alu_peak.py`) | 3.92 TFLOP/s | 1.09 GHz |
# | float32 matmul (`flops.py --peak`) | 4.06 TFLOP/s | 1.13 GHz |
#
# So an MFU against `PEAK_TFLOPS` carries an 18% penalty that no kernel can
# remove. Use `SUSTAINED_TFLOPS` when the question is "how good is this
# kernel", and `PEAK_TFLOPS` when the question is "what share of the chip".
#
# One exception runs the other way. `model_flops()` counts the full S x S
# attention, so a long causal sequence is credited with work it never runs
# and its MFU can pass 82%. Shape 13 counts 188.98 GFLOP, executes 120.33,
# and prints 91.9% where the executed work gives 58.5%. Pass
# `causal_aware=True` to compare against the work the kernel really does.
PEAK_TFLOPS = 4.946

# The best rate anything has reached on this machine, measured. It is the
# honest ceiling for a kernel, and it is 82% of PEAK_TFLOPS because the GPU
# runs at about 1.1 GHz under a sustained load, not at its 1.380 GHz top
# state.
SUSTAINED_TFLOPS = 4.06

# Kept so an old import does not break. It is no longer provisional.
PROVISIONAL_PEAK_TFLOPS = PEAK_TFLOPS

MFU_DISCLAIMER = (
    "The MFU denominator is 4.946 TFLOP/s = 14 cores x 128 ALUs x 2 x "
    "1.380 GHz. The core count and the 1380 MHz top DVFS state are read from "
    "this machine; the ALU count is bounded below at 105 by the measured "
    "matmul rate. The GPU does NOT hold 1380 MHz: a saturating loop sustains "
    "3.92 TFLOP/s and a matmul reaches 4.06, both about 1.1 GHz. So 82% is "
    "the practical ceiling of this column."
)


def provisional_mfu(flops: int, median_ms: float) -> float:
    """
    Model FLOPs Utilization against the top DVFS state of this GPU.

    The name says "provisional" for history. The denominator is now sourced:
    read the table above `PEAK_TFLOPS`. What it cannot remove is that the GPU
    does not hold its top clock, so 82% is the practical ceiling.
    """
    return achieved_tflops(flops, median_ms) / PEAK_TFLOPS


def sustained_mfu(flops: int, median_ms: float) -> float:
    """
    The same rate against the best rate this machine has ever reached.

    Use this to ask "how good is this kernel". Use `provisional_mfu` to ask
    "what share of the chip". `SUSTAINED_TFLOPS` is a stopwatch reading, so
    this number has no unverified term in it at all.
    """
    return achieved_tflops(flops, median_ms) / SUSTAINED_TFLOPS


def achieved_tflops(flops: int, median_ms: float) -> float:
    """
    Arithmetic rate of a measured run, in TFLOP/s.

    Both inputs are objective: `flops` comes from the model definition and
    `median_ms` from a stopwatch. Compare it against `MEASURED_TFLOPS` to see
    how a shape does against a plain matmul.
    """
    return flops / (median_ms / 1000.0) / 1e12


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="FLOP cost of the appendix shapes")
    parser.add_argument("--peak", action="store_true", help="re-measure matmul rates")
    args = parser.parse_args()

    if args.peak:
        _measure_peak()
        return 0

    from appendix_cases import APPENDIX_SHAPES

    print("measured matmul rates (TFLOP/s):")
    for name, rate in MEASURED_TFLOPS.items():
        print(f"  {name:<20} {rate:5.2f}")
    print()

    header = (
        f"{'#':>3} {'B':>6} {'D':>5} {'H':>3} {'S':>7} "
        f"{'GFLOP':>13} {'causal GFLOP':>13}"
    )
    print(header)
    print("-" * len(header))
    for shape in APPENDIX_SHAPES:
        config = shape.config()
        print(
            f"{shape.case_id:>3} {shape.batch_size:>6} {shape.d_model:>5} "
            f"{shape.num_heads:>3} {shape.seq_len:>7} "
            f"{model_flops(config) / 1e9:>13.2f} "
            f"{model_flops(config, causal_aware=True) / 1e9:>13.2f}"
        )
    return 0


def _measure_peak() -> None:
    """Re-measure the matmul rates. Prints the numbers behind MEASURED_TFLOPS."""
    import statistics
    import time

    import mlx.core as mx

    print("mlx matmul rate, median of 7 rounds, best of n=2048 and n=4096")
    for dtype, name in (
        (mx.float32, "float32"),
        (mx.float16, "float16"),
        (mx.bfloat16, "bfloat16"),
    ):
        best = 0.0
        for size in (2048, 4096):
            a = mx.random.normal((size, size)).astype(dtype)
            b = mx.random.normal((size, size)).astype(dtype)
            mx.eval(a, b)
            for _ in range(3):
                mx.eval([a @ b, a @ b])
                mx.synchronize()
            samples = []
            for _ in range(7):
                mx.synchronize()
                start = time.perf_counter()
                mx.eval([a @ b for _ in range(10)])
                mx.synchronize()
                samples.append((time.perf_counter() - start) / 10)
            best = max(best, 2.0 * size**3 / statistics.median(samples) / 1e12)
        print(f"  {name:>9}: {best:6.2f} TFLOP/s")


if __name__ == "__main__":
    raise SystemExit(main())

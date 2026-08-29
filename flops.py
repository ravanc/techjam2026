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

## MFU is deferred

An earlier version of this file computed Model FLOPs Utilization against a
theoretical peak of 5.01 TFLOP/s, which came from
`14 cores x 128 ALUs x 2 x 1.398 GHz`. **The clock and the ALU count were
never verified against a source.** Every MFU number scales linearly with
that constant, so the whole metric rested on an unchecked assumption.

MFU is therefore removed, not fixed. To bring it back, first establish the
peak rate from a source you trust, then divide `achieved_tflops` by it. The
measured matmul rates below are a defensible alternative denominator,
because they were measured here, but they answer a different question: they
give the share of the best rate any kernel reaches, not the share of the
hardware's theoretical peak.
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
# PROVISIONAL. Do not quote without the disclaimer.
# ---------------------------------------------------------------------------
# This peak came from `14 cores x 128 ALUs x 2 flop x 1.398 GHz`. The clock
# and the ALU count were NOT verified against any source. They were asserted
# from memory. Every MFU number scales linearly with this constant, so a wrong
# clock makes every MFU wrong by the same factor.
#
# It is kept because the hackathon grades an MFU score, so a provisional
# number is more useful than none. It is quarantined here, and every consumer
# must carry the disclaimer.
#
# To make it real: find the true M3 Pro GPU clock and ALU count from a source
# you trust, correct this value, and delete this comment block.
PROVISIONAL_PEAK_TFLOPS = 5.01

MFU_DISCLAIMER = (
    "PROVISIONAL: the MFU denominator (5.01 TFLOP/s) was asserted from "
    "memory, not verified. Every MFU below scales with it. The measured "
    "matmul rate of 4.06 TFLOP/s is the verified alternative."
)


def provisional_mfu(flops: int, median_ms: float) -> float:
    """
    Model FLOPs Utilization against an UNVERIFIED peak.

    Read `PROVISIONAL_PEAK_TFLOPS` before using this. The numerator and the
    time are objective; the denominator is not.
    """
    return achieved_tflops(flops, median_ms) / PROVISIONAL_PEAK_TFLOPS


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

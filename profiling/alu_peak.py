#!/usr/bin/env python3
"""
Measure the float32 FMA peak of this GPU, so the MFU denominator is a
stopwatch reading and not an assertion.

WHY THIS EXISTS

`flops.py` held `PROVISIONAL_PEAK_TFLOPS = 5.01`, from
`14 cores x 128 ALUs x 2 flop x 1.398 GHz`. Only the core count was ever
checked. MFU is the graded score and it scales linearly with this constant,
so a wrong clock makes every MFU wrong by the same factor.

This script does not look the clock up. It runs a kernel that does nothing
but fused multiply-add in registers, and it reports the rate that kernel
reaches. That rate is the roof for arithmetic on this machine.

HOW

Each thread holds `U` independent accumulators and runs

    acc[u] = fma(acc[u], m, b)

`ITER` times. There is no memory traffic in the loop, so the loop measures
the arithmetic units alone.

Four properties make the number trustworthy:

1. **The compiler cannot delete the work.** Each accumulator is a dependent
   chain, and float arithmetic is not associative, so no closed form exists.
   The thread writes the sum of its accumulators, so the chain is live.
2. **The compiler cannot fold the accumulators together.** Each one starts
   at a different value.
3. **`m` is 0.99999, so the chain converges** to `b / (1 - m)` instead of
   reaching infinity. A denormal or an infinity can change the throughput of
   an arithmetic unit, and this keeps the loop in normal floats.
4. **The seed comes from an input array**, so `m` and `b` are not compile
   time constants.

The script sweeps the unroll, the thread count and scalar against `float4`,
then reports the BEST rate. A peak is the best a machine reaches, so the
maximum is the correct summary, not the median.

Run:

    .venv/bin/python3 profiling/alu_peak.py
    .venv/bin/python3 profiling/alu_peak.py --json profiling/alu_peak.json

Read `references/machine.md` before you trust a number here. Check that no
other process holds the GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Dict, List, Optional

import mlx.core as mx

# The published shape of an Apple GPU core. The script does NOT use these to
# compute the peak. It uses them the other way round: it divides the MEASURED
# peak by them and prints the clock that the measurement implies, which is a
# check on the old 1.398 GHz assertion.
GPU_CORES = 14          # verified: `system_profiler SPDisplaysDataType`
ALUS_PER_CORE = 128     # NOT verified. Used for the implied-clock line only.

_SOURCE = """
    constexpr uint U = {unroll};
    constexpr uint ITER = {iters};

    uint tid = thread_position_in_grid.x;
    if (tid >= seed_shape[0]) {{
        return;
    }}

    // The seed comes from a buffer, so `m` and `b` are not compile time
    // constants and the loop cannot be folded.
    float s = seed[tid];
    float m = 0.99999f + s * 1e-9f;
    float b = 0.5f + s * 1e-9f;

    // Independent chains. One chain alone cannot fill the pipeline, because
    // each FMA waits for the one before it. `U` chains hide that latency.
    {vtype} acc[U];
    for (uint u = 0; u < U; ++u) {{
        acc[u] = ({vtype})(s + (float)u);
    }}

    for (uint i = 0; i < ITER; ++i) {{
        for (uint u = 0; u < U; ++u) {{
            acc[u] = metal::fma(acc[u], ({vtype})m, ({vtype})b);
        }}
    }}

    {vtype} total = ({vtype})0.0f;
    for (uint u = 0; u < U; ++u) {{
        total += acc[u];
    }}
    out[tid] = {reduce};
"""

_CACHE: Dict[tuple, object] = {}


def _kernel(unroll: int, iters: int, lanes: int):
    key = (unroll, iters, lanes)
    kernel = _CACHE.get(key)
    if kernel is None:
        vtype = "float" if lanes == 1 else "float%d" % lanes
        reduce_expr = "total" if lanes == 1 else " + ".join(
            "total[%d]" % i for i in range(lanes)
        )
        kernel = mx.fast.metal_kernel(
            name="techjam_alu_peak_u%d_i%d_v%d" % (unroll, iters, lanes),
            input_names=["seed"],
            output_names=["out"],
            source=_SOURCE.format(
                unroll=unroll, iters=iters, vtype=vtype, reduce=reduce_expr,
            ),
        )
        _CACHE[key] = kernel
    return kernel


def run_once(seed: mx.array, unroll: int, iters: int, lanes: int) -> mx.array:
    threads = seed.shape[0]
    return _kernel(unroll, iters, lanes)(
        inputs=[seed],
        grid=(threads, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(threads,)],
        output_dtypes=[mx.float32],
    )[0]


def time_config(threads: int, unroll: int, iters: int, lanes: int,
                repeats: int) -> Dict:
    """Return the achieved rate of one configuration, in TFLOP/s."""
    seed = mx.random.uniform(shape=(threads,)).astype(mx.float32)
    mx.eval(seed)
    mx.synchronize()

    # Two warm runs. The first one compiles the kernel.
    for _ in range(2):
        mx.eval(run_once(seed, unroll, iters, lanes))
        mx.synchronize()

    samples: List[float] = []
    for _ in range(repeats):
        mx.synchronize()
        start = time.perf_counter()
        mx.eval(run_once(seed, unroll, iters, lanes))
        mx.synchronize()
        samples.append(time.perf_counter() - start)

    median = statistics.median(samples)
    flops = 2.0 * threads * iters * unroll * lanes
    return {
        "threads": threads,
        "unroll": unroll,
        "iters": iters,
        "lanes": lanes,
        "ms": median * 1000.0,
        "tflops": flops / median / 1e12,
        "gflop_total": flops / 1e9,
    }



def time_sustained(config: Dict, launches: int, rounds: int) -> List[Dict]:
    """
    Run one configuration back to back, so the GPU clock ramps and holds.

    A short kernel measures whatever DVFS state the GPU happened to be in.
    This queues `launches` kernels into ONE eval, so the GPU has no gap to
    clock down in, and it repeats that for `rounds` rounds. The rate of the
    later rounds is the sustained rate.
    """
    threads = config["threads"]
    unroll = config["unroll"]
    lanes = config["lanes"]
    iters = config["iters"]

    seed = mx.random.uniform(shape=(threads,)).astype(mx.float32)
    mx.eval(seed)
    mx.synchronize()

    flops = 2.0 * threads * iters * unroll * lanes * launches

    rows: List[Dict] = []
    for _ in range(rounds):
        mx.synchronize()
        start = time.perf_counter()
        mx.eval([run_once(seed, unroll, iters, lanes) for _ in range(launches)])
        mx.synchronize()
        elapsed = time.perf_counter() - start
        rows.append({
            "threads": threads,
            "unroll": unroll,
            "iters": iters,
            "lanes": lanes,
            "launches": launches,
            "ms": elapsed * 1000.0,
            "tflops": flops / elapsed / 1e12,
            "gflop_total": flops / 1e9,
            "sustained": True,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="measured float32 FMA peak of this GPU")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--sustain-launches", type=int, default=24,
                        help="kernels queued into one eval, with no gap")
    parser.add_argument("--sustain-rounds", type=int, default=6)
    args = parser.parse_args()

    print("float32 FMA peak. Each row is one kernel configuration.")
    print("The loop touches no memory, so it measures the arithmetic units.")
    print("")
    print("%9s %7s %8s %6s %10s %12s" %
          ("threads", "unroll", "iters", "lanes", "ms", "TFLOP/s"))
    print("-" * 60)

    results: List[Dict] = []
    for lanes in (1, 4):
        for threads in (1 << 16, 1 << 18, 1 << 20):
            for unroll in (4, 8, 16):
                # Hold the work per launch near 30 GFLOP, so every
                # configuration runs long enough to clear the launch floor.
                iters = max(64, int(30e9 / (2.0 * threads * unroll * lanes)))
                row = time_config(threads, unroll, iters, lanes, args.repeats)
                results.append(row)
                print("%9d %7d %8d %6d %10.4f %12.3f" %
                      (row["threads"], row["unroll"], row["iters"],
                       row["lanes"], row["ms"], row["tflops"]))

    best = max(results, key=lambda r: r["tflops"])

    # The sweep above runs 8 ms bursts with a Python gap between them, and a
    # GPU does not hold its top clock through a gap. This phase runs the best
    # configuration back to back for seconds, so the clock ramps and stays
    # there. If the rate rises here, the sweep measured a DVFS state and not
    # the arithmetic units.
    print("")
    print("Sustained phase: the same configuration, %d launches back to back."
          % args.sustain_launches)
    sustained = time_sustained(best, args.sustain_launches, args.sustain_rounds)
    for i, row in enumerate(sustained):
        print("  round %d: %8.2f ms for %6.1f GFLOP -> %6.3f TFLOP/s"
              % (i + 1, row["ms"], row["gflop_total"], row["tflops"]))

    best_sustained = max(sustained, key=lambda r: r["tflops"])
    if best_sustained["tflops"] > best["tflops"]:
        best = best_sustained
    peak = best["tflops"]
    implied_clock = peak * 1e12 / (GPU_CORES * ALUS_PER_CORE * 2) / 1e9

    print("")
    print("BEST: %.3f TFLOP/s at threads=%d unroll=%d lanes=%d" %
          (peak, best["threads"], best["unroll"], best["lanes"]))
    print("")
    print("Implied clock, if the GPU really has %d cores x %d ALUs:" %
          (GPU_CORES, ALUS_PER_CORE))
    print("    %.3f TFLOP/s / (%d x %d x 2) = %.3f GHz" %
          (peak, GPU_CORES, ALUS_PER_CORE, implied_clock))
    print("")
    print("The core count is verified (system_profiler). The ALU count is")
    print("not, so read the clock line as a consistency check, not a fact.")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({
                "peak_tflops": peak,
                "best": best,
                "sustained": sustained,
                "gpu_cores": GPU_CORES,
                "alus_per_core_assumed": ALUS_PER_CORE,
                "implied_clock_ghz": implied_clock,
                "configurations": results,
            }, handle, indent=2)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

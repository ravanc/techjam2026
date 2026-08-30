#!/usr/bin/env python3
"""
Stage-by-stage roofline for the attention path of `UserOptimizedTransformer`.

The scoreboard gives one time for each shape. This script splits that time
into the stages of one transformer block, and it says what limits each stage:
the arithmetic units, the memory system, or the kernel launch.

For each stage it reports:

    ms          measured median time of the stage, alone, on the GPU
    GFLOP       matmul FLOPs of the stage (0 for a data movement stage)
    MiB         compulsory DRAM traffic: distinct inputs read + outputs
                written, once each. It is a LOWER BOUND. A kernel that
                spills, or that re-reads an operand, moves more.
    FLOP/B      arithmetic intensity, GFLOP / MiB. Compare against the ridge.
    GFLOP/s     achieved arithmetic rate
    GB/s        achieved bandwidth against the compulsory traffic
    %comp       achieved rate / measured matmul peak (4.06 TFLOP/s)
    %mem        achieved bandwidth / measured bandwidth (128 GB/s)
    limit       the larger of %comp and %mem names the limit. When both are
                small the stage is LAUNCH bound: it does not fill the GPU.

**%mem READS HIGH, AND CAN PASS 100%.** `ms` has FLOOR_MS subtracted, and
GB/s comes from `ms`. PEAK_GBPS does NOT have it subtracted: it is a raw
`x * 2.0` reading. So the two sides of the ratio are measured differently,
and %mem overstates by roughly `raw / (raw - FLOOR_MS)`.

Measured at the shape 6 `ln1`: `ms` 0.9477, `raw` 1.2492, both for 128 MiB.
The `ms` rate is 141.6 GB/s and %mem prints 110.6. The `raw` rate is
107.4 GB/s, and that agrees with the 109.0 GB/s that a 64 MiB `x * 2.0`
reaches in the table below, and with a standalone `fast_layernorm` at the
same size (1.305 ms, 107.5 GB/s).

So read %mem as a rank, not as a fraction of the roof. To compare a stage
against the roof, use `raw`. A reading above 100% is this bias, not a stage
that beat physics. The same bias inflates %comp, which is why shape 8
prints 100.4% on a GEMM.

The ridge point of this machine is 4.06e12 / 128e9 = 31.7 FLOP/byte. A stage
below the ridge cannot reach the arithmetic peak whatever the kernel does.

Run:

    .venv/bin/python3 profiling/stage_roofline.py --shapes 1,6,8,13
    .venv/bin/python3 profiling/stage_roofline.py --shapes 13 --json out.json

The script profiles ONE batch chunk, because that is what the GPU runs. The
chunk count is in the header of each shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as mlx_nn
import torch

from appendix_cases import APPENDIX_SHAPES as SHAPES
from torch_transformer_benchmark import (
    LAYER_NORM_EPS,
    UserOptimizedTransformer,
    _attention,
)
from fast_layernorm import layer_norm as fast_layer_norm
from steel_gemm import steel_addmm

MIB = 1024 * 1024
ITEM = 4  # float32

# Both roofs are MEASURED on this machine. They are stopwatch readings, not a
# specification.
#
# PEAK_TFLOPS is the square-matmul rate from `flops.py --peak`.
#
# PEAK_GBPS is the ACHIEVED streaming rate of `x * 2.0` at 1 GiB, not the
# 150 GB/s specification in references/machine.md. Measured here:
#
#     size MiB     1     4    16    64   256  1024
#     copy GB/s  7.1  34.9  72.9 109.0 125.6 128.1
#
# An array under 64 MiB does not reach the roof. That fact alone explains
# most of the small shapes.
PEAK_TFLOPS = 4.06
PEAK_GBPS = 128.0
RIDGE = PEAK_TFLOPS * 1e12 / (PEAK_GBPS * 1e9)

# The cost of one `mx.eval()` + `mx.synchronize()` round trip, with the graph
# build of a single small op. Every isolated stage time pays it once. The
# script measures it at start and subtracts it. Measured: about 0.13 ms.
FLOOR_MS = 0.0

# The smallest stage time the floor subtraction can resolve. Below it the
# remainder is the jitter of the round trip, not the work of the stage.
FLOOR_RESOLUTION_MS = 0.05


def measure_floor(repeats: int = 50) -> float:
    """
    Median cost of one `mx.eval()` + `mx.synchronize()` round trip.

    Every isolated stage measurement pays this once, so it must come off
    before a rate means anything. At shape 1 it is 0.13 ms against a 1.7 ms
    layer, so it changes every stage verdict.
    """
    a = mx.random.normal((4,))
    b = mx.random.normal((4,))
    mx.eval(a, b)
    mx.synchronize()
    for _ in range(5):
        mx.eval(a + b)
        mx.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(a + b)
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def timed(build, warmup: int = 3, repeats: int = 20) -> float:
    """
    Median wall time of one stage, in ms.

    `build` returns the output array of the stage. Its inputs are already
    evaluated, so the measurement holds the stage alone. `mx.eval` builds the
    graph and `mx.synchronize` waits for the GPU. `mx.eval` alone times the
    CPU graph build. See references/machine.md.
    """
    for _ in range(warmup):
        mx.eval(build())
    mx.synchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(build())
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def peak_delta_mib(build) -> float:
    """Peak GPU memory that one stage adds, in MiB. It shows a spill."""
    mx.eval(build())
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    out = build()
    mx.eval(out)
    mx.synchronize()
    peak = mx.get_peak_memory()
    del out
    return max(0.0, (peak - base) / MIB)


def classify(gflops: float, gbps: float) -> str:
    """Name the limit of a stage from its two achieved rates."""
    comp = gflops / (PEAK_TFLOPS * 1000.0)
    mem = gbps / PEAK_GBPS
    if comp < 0.05 and mem < 0.05:
        return "LAUNCH"
    return "COMPUTE" if comp >= mem else "IO"


def profile_shape(index: int, repeats: int) -> Optional[Dict]:
    """Time every stage of one transformer block, for one shape."""
    shape = next(s for s in SHAPES if s.case_id == index)
    if not shape.enabled:
        return None
    config = shape.config()

    model = UserOptimizedTransformer(config)
    model.eval()
    model._build_mlx_weights()
    plan = model.plan
    layer = model._mlx_layers[0]

    batch = config.batch_size
    chunk = plan.batch_chunk or batch
    chunk = min(chunk, batch)
    num_chunks = (batch + chunk - 1) // chunk

    seq = config.seq_len
    d_model = config.d_model
    ffn = config.ffn_dim
    heads_n = config.num_heads
    head_dim = d_model // heads_n
    width = plan.pad_head_dim or head_dim
    scale = head_dim**-0.5
    mask = "causal" if config.causal else None

    tokens = chunk * seq
    act = tokens * d_model * ITEM          # one B x S x D activation
    qkv_act = tokens * heads_n * width * ITEM

    # Profile the LayerNorm the PLAN selects, not always the MLX one. The
    # model chooses between them, so a fixed choice here measures the wrong
    # kernel and hides the effect of row 31.
    norm_kernel = fast_layer_norm if plan.fast_layer_norm else mx.fast.layer_norm
    # Row 36 defers the residual biases into the LayerNorm and gives the
    # residual add to the projection GEMM as its C operand. The stage list
    # must follow the block, or it measures kernels the model never runs.
    defer = plan.defer_bias
    norm_name = "ln%d (%s%s)" % (
        0, "fast_layernorm" if plan.fast_layer_norm else "mx.fast",
        ", +bias" if defer else "")

    x = mx.random.normal((chunk, seq, d_model)).astype(mx.float32)
    mx.eval(x)
    mx.synchronize()

    def heads_of(projection, last=width):
        return projection.reshape(chunk, seq, heads_n, last).transpose(0, 2, 1, 3)

    # ---- build the operands of each stage, evaluated -------------------
    # `carry` is the deferred bias vector of row 36. It is (d_model,), never
    # a full activation.
    carry = layer["ob"].astype(mx.float32) if defer else None

    def do_norm(value, wkey, bkey):
        if defer:
            return norm_kernel(value, layer[wkey], layer[bkey],
                               LAYER_NORM_EPS, pre_bias=carry)
        return norm_kernel(value, layer[wkey], layer[bkey], LAYER_NORM_EPS)

    h1 = do_norm(x, "n1w", "n1b")
    mx.eval(h1)
    fused = mx.addmm(layer["qkvb"], h1, layer["qkvw"])
    mx.eval(fused)
    q0, k0, v0 = (heads_of(p) for p in mx.split(fused, 3, axis=-1))
    mx.eval(q0, k0, v0)
    ctx = _attention(q0, k0, v0, scale, mask, plan.causal_block, plan.steel_attention)
    mx.eval(ctx)
    ctx_cut = ctx[..., :head_dim] if ctx.shape[-1] != head_dim else ctx
    merged = ctx_cut.transpose(0, 2, 1, 3).reshape(chunk, seq, d_model)
    mx.eval(merged)
    if defer:
        # The residual rides in the C operand of the projection GEMM.
        x1 = mx.addmm(x, merged, layer["ow"].T)
        attn_out = None
    else:
        attn_out = mx.addmm(layer["ob"], merged, layer["ow"].T)
        mx.eval(attn_out)
        x1 = x + attn_out
    mx.eval(x1)
    h2 = do_norm(x1, "n2w", "n2b")
    mx.eval(h2)
    hidden = mlx_nn.gelu(mx.addmm(layer["fib"], h2, layer["fiw"].T))
    mx.eval(hidden)
    mx.synchronize()

    # Score matrix traffic. The fused kernel never writes it. The fallback
    # writes it and reads it back, so it decides the SDPA traffic.
    score_bytes = chunk * heads_n * seq * seq * ITEM
    # Causal attention does half of the S x S work, plus the diagonal.
    sdpa_flops = 2.0 * chunk * heads_n * seq * (seq + 1) * width

    stages: List[Dict] = []

    def add(name, build, flops, byts, note=""):
        raw = timed(build, repeats=repeats)
        net = raw - FLOOR_MS
        # A stage that sits at the round trip carries no measurable GPU time.
        # The floor subtraction cannot resolve it, and a rate taken from the
        # remainder is noise. Report it as LAUNCH bound and give no rate.
        at_floor = net < FLOOR_RESOLUTION_MS
        ms = max(net, 0.0)
        seconds = max(net, FLOOR_RESOLUTION_MS) / 1000.0
        gflops = flops / 1e9 / seconds
        gbps = byts / 1e9 / seconds
        stages.append({
            "stage": name,
            "ms": ms,
            "ms_raw": raw,
            "at_floor": at_floor,
            "gflop": flops / 1e9,
            "mib": byts / MIB,
            "intensity": (flops / byts) if byts else 0.0,
            "gflops": None if at_floor else gflops,
            "gbps": None if at_floor else gbps,
            "pct_compute": None if at_floor else 100.0 * gflops / (PEAK_TFLOPS * 1000.0),
            "pct_memory": None if at_floor else 100.0 * gbps / PEAK_GBPS,
            "limit": "LAUNCH" if at_floor else classify(gflops, gbps),
            "note": note,
        })

    # 1. LayerNorm. No matmul. Reads x, writes h.
    add(norm_name.replace("ln0", "ln1"),
        lambda: do_norm(x, "n1w", "n1b"), 0.0, 2 * act)

    # 2. QKV projection. Reads h and the weight, writes 3 head-width acts.
    qkv_w = d_model * 3 * heads_n * width * ITEM
    add("qkv proj (addmm)",
        lambda: mx.addmm(layer["qkvb"], h1, layer["qkvw"]),
        2.0 * tokens * d_model * 3 * heads_n * width,
        act + qkv_w + 3 * qkv_act,
        "pad %d->%d" % (head_dim, width) if width != head_dim else "")

    # 3. Split and transpose into the head layout. Pure data movement.
    # Evaluate the three arrays separately. An earlier version concatenated
    # them to force the copy, and the concatenate added a whole extra pass
    # that the model never runs.
    def build_heads():
        return [heads_of(p) for p in mx.split(fused, 3, axis=-1)]
    add("split+transpose", build_heads, 0.0, 2 * 3 * qkv_act,
        "layout only")

    # 4. The attention core.
    path = "steel" if plan.steel_attention else (
        "fused" if width in (64, 72, 80, 96, 128) else "FALLBACK")
    add("sdpa (attention)",
        lambda: _attention(q0, k0, v0, scale, mask,
                           plan.causal_block, plan.steel_attention),
        sdpa_flops, 4 * qkv_act, path)

    # 5. Merge the heads back. Pure data movement, and FREE on the steel
    # path: row 34 makes the kernel write [B, S, H, D], so the model merges
    # with a plain reshape. An unconditional transpose here invents a copy
    # the model never runs. It cost 1.06 ms of a claimed 14.60 ms shape 6
    # layer, and it was most of the gap between the stage sum and the real
    # layer time. Follow `_mlx_transformer()`.
    if plan.steel_attention:
        merge_note = "free reshape (row 34)"
        merge_bytes = 0.0

        def build_merge():
            return ctx.reshape(chunk, seq, d_model)
    else:
        merge_note = "layout only"
        merge_bytes = 2 * act

        def build_merge():
            c = ctx[..., :head_dim] if ctx.shape[-1] != head_dim else ctx
            return c.transpose(0, 2, 1, 3).reshape(chunk, seq, d_model)

    add("merge heads", build_merge, 0.0, merge_bytes, merge_note)

    # 6. Output projection. Row 36 gives it the residual as its C operand,
    # so it reads one more activation and the separate add disappears.
    if defer:
        add("out proj (+residual)",
            lambda: mx.addmm(x, merged, layer["ow"].T),
            2.0 * tokens * d_model * d_model,
            3 * act + d_model * d_model * ITEM, "residual in C")
    else:
        add("out proj (addmm)",
            lambda: mx.addmm(layer["ob"], merged, layer["ow"].T),
            2.0 * tokens * d_model * d_model,
            2 * act + d_model * d_model * ITEM)

        # 7. Residual add. Row 36 removed this kernel.
        add("residual add", lambda: x + attn_out, 0.0, 3 * act)

    # The FFN, for the share it takes of the block.
    add(norm_name.replace("ln0", "ln2"),
        lambda: do_norm(x1, "n2w", "n2b"), 0.0, 2 * act)
    # Profile the FFN input the PLAN selects. Row 33 folds GELU into the
    # GEMM epilogue, so the fused path never writes the pre-activation to
    # DRAM. A fixed `mlx_nn.gelu(mx.addmm(...))` here measures a kernel the
    # model does not run.
    if plan.fuse_gelu is not None:
        bm, bn, bk, wm, wn = plan.fuse_gelu
        ffn_in_name = "ffn_in + gelu (fused)"

        def build_ffn_in():
            return steel_addmm(layer["fib"], h2, layer["fiw"], gelu=True,
                               bm=bm, bn=bn, bk=bk, wm=wm, wn=wn)
    else:
        ffn_in_name = "ffn_in + gelu"

        def build_ffn_in():
            return mlx_nn.gelu(mx.addmm(layer["fib"], h2, layer["fiw"].T))

    add(ffn_in_name, build_ffn_in,
        2.0 * tokens * d_model * ffn,
        act + d_model * ffn * ITEM + tokens * ffn * ITEM)
    if defer:
        add("ffn_out (+residual)",
            lambda: mx.addmm(x1, hidden, layer["fow"].T),
            2.0 * tokens * ffn * d_model,
            tokens * ffn * ITEM + ffn * d_model * ITEM + 2 * act,
            "residual in C")
    else:
        add("ffn_out (addmm)",
            lambda: mx.addmm(layer["fob"], hidden, layer["fow"].T),
            2.0 * tokens * ffn * d_model,
            tokens * ffn * ITEM + ffn * d_model * ITEM + act)
        add("residual add 2", lambda: x1 + attn_out, 0.0, 3 * act)

    # The SDPA peak memory says which kernel ran. A delta near the score
    # matrix size means the fallback materialized B x H x S x S.
    sdpa_peak = peak_delta_mib(
        lambda: _attention(q0, k0, v0, scale, mask,
                           plan.causal_block, plan.steel_attention)
    )
    # What the stage must allocate if it never materializes the scores: the
    # output alone, plus what the allocator holds for the operands.
    operand_only_mib = 4.0 * qkv_act / MIB

    # The head layout is a TRANSPOSE, and an MLX transpose is a free strided
    # view, not a copy. The layout therefore costs nothing as a stage. It
    # costs inside the attention kernel, which then reads q, k and v with a
    # stride instead of in order. This pair of numbers separates the two.
    contiguous = [mx.contiguous(a) for a in (q0, k0, v0)]
    mx.eval(contiguous)
    mx.synchronize()
    strided_ms = timed(
        lambda: _attention(q0, k0, v0, scale, mask,
                           plan.causal_block, plan.steel_attention),
        repeats=repeats) - FLOOR_MS
    contig_ms = timed(
        lambda: _attention(contiguous[0], contiguous[1], contiguous[2], scale,
                           mask, plan.causal_block, plan.steel_attention),
        repeats=repeats) - FLOOR_MS
    copy_ms = timed(lambda: [mx.contiguous(a) for a in (q0, k0, v0)],
                    repeats=repeats) - FLOOR_MS
    del contiguous

    total_ms = sum(s["ms"] for s in stages)
    # Stages up to and including the output projection. Row 36 removed the
    # separate residual add, so the count depends on the plan.
    attn_ms = sum(s["ms"] for s in stages[:6 if defer else 7])

    # The real model, for the same shape. The GPU overlaps the layers, so the
    # sum of the isolated stages is always larger. The gap is what the
    # pipeline buys.
    #
    # The gap is NOT fusion. `mx.compile` does not fuse the elementwise stages
    # of this block. Measured at the shape 6 chunk: `addmm` then `gelu` is
    # 2.4414 ms both eager and compiled, and add then LayerNorm is 2.8309 ms
    # eager against 2.8347 ms compiled. The byte count agrees: the unfused
    # pair moves 320 MiB, which is 2.71 ms at 124 GB/s. See WORKFLOW.md.
    real = torch.randn(batch, seq, d_model)
    with torch.no_grad():
        for _ in range(2):
            model(real)
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            model(real)
            samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    real_ms = samples[len(samples) // 2]
    per_unit = real_ms / (config.num_layers * num_chunks)

    return {
        "index": index,
        "label": "B%d D%d H%d S%d" % (batch, d_model, heads_n, seq),
        "config": {
            "batch": batch, "chunk": chunk, "num_chunks": num_chunks,
            "seq": seq, "d_model": d_model, "heads": heads_n,
            "head_dim": head_dim, "width": width, "ffn": ffn,
            "layers": config.num_layers,
        },
        "plan": plan.describe(),
        "stages": stages,
        "sdpa_peak_mib": sdpa_peak,
        "operand_only_mib": operand_only_mib,
        "sdpa_strided_ms": strided_ms,
        "sdpa_contiguous_ms": contig_ms,
        "operand_copy_ms": copy_ms,
        "score_matrix_mib": score_bytes / MIB,
        "sum_ms_one_layer_one_chunk": total_ms,
        "attention_ms_one_layer_one_chunk": attn_ms,
        "real_model_ms": real_ms,
        "real_ms_per_layer_per_chunk": per_unit,
        "floor_ms": FLOOR_MS,
    }


def print_report(result: Dict) -> None:
    cfg = result["config"]
    print("")
    print("=" * 100)
    print("Shape %d: %s   head_dim=%d  layers=%d"
          % (result["index"], result["label"], cfg["head_dim"], cfg["layers"]))
    print("plan: %s" % result["plan"])
    if cfg["num_chunks"] > 1:
        print("profiled ONE chunk of %d rows. The shape runs %d chunks x %d layers."
              % (cfg["chunk"], cfg["num_chunks"], cfg["layers"]))
    else:
        print("profiled one layer of %d. Batch is one chunk." % cfg["layers"])
    print("=" * 100)
    head = ("%-20s %9s %7s %9s %10s %8s %10s %9s %6s %6s  %-8s"
            % ("stage", "ms", "raw", "GFLOP", "MiB", "FLOP/B",
               "GFLOP/s", "GB/s", "%comp", "%mem", "limit"))
    print(head)
    print("-" * len(head))
    total = result["sum_ms_one_layer_one_chunk"]
    for s in result["stages"]:
        def num(value, fmt):
            return ("%%%s" % fmt) % value if value is not None else "-".rjust(
                int(fmt.split(".")[0]))
        print("%-20s %9s %7.4f %9.3f %10.1f %8.2f %10s %9s %6s %6s  %-8s %s"
              % (s["stage"],
                 num(s["ms"], "9.4f") if not s["at_floor"] else "at floor",
                 s["ms_raw"], s["gflop"], s["mib"], s["intensity"],
                 num(s["gflops"], "10.1f"), num(s["gbps"], "9.1f"),
                 num(s["pct_compute"], "6.1f"), num(s["pct_memory"], "6.1f"),
                 s["limit"], s["note"]))
    print("-" * len(head))
    print("%-20s %9.4f   (attention stages 1-7: %.4f ms, %.0f%%)"
          % ("sum of stages", total,
             result["attention_ms_one_layer_one_chunk"],
             100.0 * result["attention_ms_one_layer_one_chunk"] / total))
    print("%-20s %9.4f   (the real model, %s, divided by %d layers x %d chunks)"
          % ("real per layer", result["real_ms_per_layer_per_chunk"],
             "%.3f ms" % result["real_model_ms"],
             cfg["layers"], cfg["num_chunks"]))
    print("")
    print("`ms` has the %.4f ms eval+synchronize round trip removed; `raw` has not."
          % result["floor_ms"])
    print("The isolated sum is larger than the real layer because the GPU overlaps")
    print("the layers. mx.compile does NOT fuse the elementwise stages: measured")
    print("1.00x against eager for addmm+gelu and for add+LayerNorm. See WORKFLOW.md.")
    print("sdpa peak memory: %.1f MiB allocated. Operands+output alone would be %.1f MiB."
          % (result["sdpa_peak_mib"], result["operand_only_mib"]))
    print("   The full B x H x S x S score matrix is %.1f MiB. A delta near the"
          % result["score_matrix_mib"])
    print("   operand figure means the kernel never wrote the scores to DRAM.")
    print("")
    print("Strided operands. The head layout is a transpose, and an MLX transpose is")
    print("a free view. The attention kernel therefore reads q, k and v with a stride:")
    strided = result["sdpa_strided_ms"]
    contig = result["sdpa_contiguous_ms"]
    copy = result["operand_copy_ms"]
    if min(strided, contig) < FLOOR_RESOLUTION_MS:
        # Both sides sit at the round trip. The remainder is jitter, and a
        # ratio of two jitter values means nothing.
        print("   both sides are at the launch floor. This shape is too small to")
        print("   separate a strided read from a contiguous one.")
    else:
        print("   sdpa on the strided view   %7.4f ms   <- what the model runs"
              % strided)
        print("   sdpa on contiguous copies  %7.4f ms   (%.2fx faster)"
              % (contig, strided / contig))
        print("   cost to make them contiguous %5.4f ms  -> copy path total %7.4f ms"
              % (max(copy, 0.0), contig + max(copy, 0.0)))
    print("ridge point of this machine: %.1f FLOP/byte. A stage below it cannot"
          % RIDGE)
    print("reach the arithmetic peak, whatever the kernel does.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="1,6,8,13",
                        help="comma separated shape numbers")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    global FLOOR_MS
    torch.manual_seed(0)
    FLOOR_MS = measure_floor()
    print("eval+synchronize round trip floor: %.4f ms (subtracted from every stage)"
          % FLOOR_MS)
    print("roofs: %.2f TFLOP/s matmul, %.0f GB/s streaming copy. Ridge %.1f FLOP/byte."
          % (PEAK_TFLOPS, PEAK_GBPS, RIDGE))
    indices = [int(t) for t in args.shapes.split(",") if t.strip()]
    results = []
    for index in indices:
        result = profile_shape(index, args.repeats)
        if result is None:
            print("shape %d is disabled, skipped" % index)
            continue
        print_report(result)
        results.append(result)
        mx.clear_cache()

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"peak_tflops": PEAK_TFLOPS, "peak_gbps": PEAK_GBPS,
                       "ridge": RIDGE, "results": results}, handle, indent=2)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

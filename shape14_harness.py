#!/usr/bin/env python3
"""
An alternate harness for Appendix 3.7 shape 14 only.

    B = 32, D = 1024, H = 16, S = 100000, L = 2, causal, FFN = 1024

`torch_transformer_benchmark.py` cannot run this shape, and no machine can.
`BaselineSelfAttention` materializes a `B x H x S x S` score matrix, which is
18.6 TiB here. The baseline therefore gives no reference output and no
reference time. This file supplies both by other means. It does not change
the graded harness, and it does not change `BaselineTransformer`.

Three facts make the shape runnable at all:

1. `head_dim = 1024 / 16 = 64`, which is in `FUSED_HEAD_DIMS`. So
   `mx.fast.scaled_dot_product_attention` reaches the fused flash kernel.
   That kernel never materializes the scores, and it skips the causal
   triangle. Measured at S=100000: the call adds 390.6 MiB of peak memory,
   which is the output alone. On the fallback kernel one row of one layer
   needs 596 GiB, so the fused path is the only path.
2. `plan_kernels()` already selects `batch_chunk=1` for this shape. One
   sequence is 391 MiB, which fits the 12.0 GiB working set.
3. Every other field of the shape 14 plan is also selected by a shape that
   the graded harness runs. Only the value `batch_chunk=1` is new, and
   shape 6 runs the same loop at `batch_chunk=1024`. See `--coverage`.

## Why this file does not call `UserOptimizedTransformer.forward()`

`forward()` concatenates all 32 chunk outputs into one 12.2 GiB array, and
the harness holds a 12.2 GiB input beside it. That needs over 36 GiB on an
18 GiB machine. This file drives the chunk loop itself: it builds one row,
runs the full depth on it, reduces the result, and frees it. Peak live
memory stays near 2 GiB.

## How this file grades accuracy with no baseline

The model is causal, and LayerNorm, the FFN and the residual adds work on
one token at a time. So output row `i` depends only on input rows `0..i`.
Two checks follow from that.

**Check A, the prefix check.** Feed the *unmodified* `BaselineTransformer`
the first `n` tokens of the shape 14 input. That is a small problem, and the
CPU runs it. Its answer must equal the first `n` rows of the full shape 14
answer. This check uses the graded class and the graded `compare_outputs()`.
It agrees to `atol`, not bit exactly, because the two runs sum a different
number of keys.

**Check B, the tail check.** Check A cannot reach row 99999, which attends
every key. `frugal_forward()` repeats the baseline arithmetic with attention
in query blocks, so it never materializes the score matrix. `--validate`
proves it equals `BaselineTransformer` on the shapes that both can run.
Only then does `--tail` run it at S=100000, on one row.

`frugal_forward()` is a reference. It is NOT the graded baseline.

## What this file reports

The graded harness reports CPU, MPS, MLX and MFU. Two of those columns state
a fact here, not a number: the CPU and MPS baselines stop with an allocation
error before they compute anything, so they have no time. This file prints
the reason in their place. It prints no speedup, because a speedup against a
baseline that cannot exist is invented.

Read the MFU with the caution that `references/scoreboard.md` already gives
for shape 13, and read it harder. `flops.py` credits the full `S x S`,
because that is what the baseline computes, while the fused kernel runs the
triangle only. Attention is 97% of shape 14, so this column prints close to
twice the true utilization.

## Commands

    .venv/bin/python3 shape14_harness.py --coverage
    .venv/bin/python3 shape14_harness.py --ladder
    .venv/bin/python3 shape14_harness.py --validate
    .venv/bin/python3 shape14_harness.py --check
    .venv/bin/python3 shape14_harness.py --time
    .venv/bin/python3 shape14_harness.py --tail

`--time` takes about 10 minutes. `--tail` takes about 15 minutes on the CPU.
Read the timing rules in CLAUDE.md before you start either one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from flops import model_flops, provisional_mfu
from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    _to_mlx,
    _to_torch,
    compare_outputs,
    copy_model_weights,
    plan_kernels,
)

# Appendix 3.7 shape 14.
SHAPE14 = TransformerConfig(
    batch_size=32,
    d_model=1024,
    num_heads=16,
    seq_len=100000,
    num_layers=2,
    causal=True,
    ffn_dim=1024,
)

# The graded harness defaults. Do not raise them. See CLAUDE.md.
RTOL = 0.02
ATOL = 0.002
SEED = 1234

# The size the CPU and MPS baselines ask for at shape 14, and cannot get.
BASELINE_SCORE_BYTES = (
    SHAPE14.batch_size * SHAPE14.num_heads * SHAPE14.seq_len * SHAPE14.seq_len * 4
)

RESULT_PATH = "profiling/results/shape14.json"


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

def row_seed(row: int, seed: int = SEED) -> int:
    """
    The seed of one batch row.

    `generate_random_case()` draws the whole `[B, S, D]` tensor from one
    generator. That tensor is 12.2 GiB, so this file never builds it. It
    gives each row its own stream instead. The stream is reproducible, and
    it is the definition of the shape 14 input for every run of this file.
    """
    return seed * 1_000_003 + row


def make_row(
    config: TransformerConfig, row: int, seed: int = SEED
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One batch row of the input, as `[1, S, D]` on the CPU in float32."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(row_seed(row, seed))
    x = torch.randn(
        1, config.seq_len, config.d_model,
        generator=generator, device="cpu", dtype=torch.float32,
    )
    mask = torch.ones(1, config.seq_len, device="cpu", dtype=torch.bool)
    return x, mask


def build_models(
    config: TransformerConfig, seed: int = SEED
) -> Tuple[BaselineTransformer, UserOptimizedTransformer]:
    """
    Build both models with the same weights.

    `BaselineTransformer` holds no tensor that grows with the batch or the
    sequence, so it costs nothing to build it at shape 14. Only its
    `forward()` fails. This file calls that `forward()` on short inputs only.
    """
    torch.manual_seed(seed)
    baseline = BaselineTransformer(config).eval()
    optimized = UserOptimizedTransformer(config).eval()
    copy_model_weights(baseline, optimized)
    optimized._build_mlx_weights()
    return baseline, optimized


# --------------------------------------------------------------------------
# The MLX path, one row at a time
# --------------------------------------------------------------------------

@dataclass
class RowResult:
    """One row of the answer, reduced. The full row is 391 MiB, so it goes."""

    row: int
    compute_ms: float
    total_ms: float
    checksum: float
    sum_squares: float
    max_abs: float
    probes: Dict[int, np.ndarray] = field(default_factory=dict)
    head: Optional[torch.Tensor] = None


def run_row(
    optimized: UserOptimizedTransformer,
    config: TransformerConfig,
    row: int,
    seed: int = SEED,
    probe_positions: Tuple[int, ...] = (),
    head_tokens: int = 0,
) -> RowResult:
    """
    Run the whole depth on one batch row, then reduce the answer.

    `compute_ms` times the MLX call alone. `total_ms` adds the framework
    boundary, which `UserOptimizedTransformer.forward()` also pays: the copy
    of the input into MLX and the wrap of the output back into torch.
    `references/machine.md` measures that boundary at 3.4% of shape 6, so
    the two numbers are not the same and this file reports both.
    """
    x, mask = make_row(config, row, seed)

    t_total = time.perf_counter_ns()
    mlx_x = _to_mlx(x)
    mlx_mask = _to_mlx(mask)

    t_compute = time.perf_counter_ns()
    out = optimized._mlx_call[False](
        mlx_x, mlx_mask, optimized._mlx_layers, *optimized._mlx_final
    )
    mx.eval(out)
    mx.synchronize()
    compute_ms = (time.perf_counter_ns() - t_compute) / 1e6

    torch_out = _to_torch(out, dtype=torch.float32, device=torch.device("cpu"))
    total_ms = (time.perf_counter_ns() - t_total) / 1e6

    probes = {
        position: np.array(np.asarray(out[0, position]), copy=True)
        for position in probe_positions
    }
    head = torch_out[:, :head_tokens].clone() if head_tokens else None

    result = RowResult(
        row=row,
        compute_ms=compute_ms,
        total_ms=total_ms,
        checksum=float(mx.sum(out).item()),
        sum_squares=float(mx.sum(out.astype(mx.float32) ** 2).item()),
        max_abs=float(mx.max(mx.abs(out)).item()),
        probes=probes,
        head=head,
    )

    del out, mlx_x, mlx_mask, torch_out, x
    mx.clear_cache()
    return result


# --------------------------------------------------------------------------
# The frugal reference
# --------------------------------------------------------------------------

def frugal_forward(
    baseline: BaselineTransformer,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    q_block: int = 256,
) -> torch.Tensor:
    """
    The arithmetic of `BaselineTransformer.forward()`, in query blocks.

    The baseline builds the whole `B x H x S x S` score matrix and then sets
    the masked entries to `-inf`. Those entries leave the softmax as zero, so
    they add nothing. This function gives each query block only the keys it
    can see, so it never holds more than `H x q_block x S` scores.

    It is a REFERENCE, not the graded baseline. `--validate` proves it equal
    to `BaselineTransformer` on the shapes that both can run. Read that
    output before you trust any number this function produces.

    Every other operation copies the baseline exactly, in the same order and
    the same precision: the softmax accumulates in float32, and the two
    residual adds and both LayerNorms keep their place.
    """
    config = baseline.config
    causal = config.causal
    batch, seq_len, _ = x.shape

    for layer in baseline.layers:
        attention = layer.attention
        heads, head_dim, scale = attention.num_heads, attention.head_dim, attention.scale

        normed = layer.norm1(x)
        q = attention._split_heads(attention.q_proj(normed))
        k = attention._split_heads(attention.k_proj(normed))
        v = attention._split_heads(attention.v_proj(normed))
        del normed

        context = torch.empty(batch, heads, seq_len, head_dim, dtype=x.dtype)
        for start in range(0, seq_len, q_block):
            stop = min(start + q_block, seq_len)
            # A causal query cannot see a key past its own row, so the keys
            # beyond `stop` are all masked. Drop them instead of masking them.
            keys = stop if causal else seq_len
            scores = torch.matmul(
                q[:, :, start:stop], k[:, :, :keys].transpose(-2, -1)
            ) * scale

            if causal:
                rows = torch.arange(start, stop)[:, None]
                columns = torch.arange(keys)[None, :]
                scores = scores.masked_fill(columns > rows, float("-inf"))

            if valid_token_mask is not None:
                invalid = ~valid_token_mask[:, None, None, :keys]
                scores = scores.masked_fill(invalid, float("-inf"))

            probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
            context[:, :, start:stop] = torch.matmul(probs, v[:, :, :keys])
            del scores, probs

        del q, k, v
        context = (
            context.transpose(1, 2).contiguous().view(batch, seq_len, attention.d_model)
        )
        attended = attention.out_proj(context)
        del context
        if valid_token_mask is not None:
            attended = attended.masked_fill(~valid_token_mask[..., None], 0)

        x = x + attended
        del attended
        x = x + layer.ffn_out(F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none"))
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)

    x = baseline.final_norm(x)
    if valid_token_mask is not None:
        x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def mode_coverage() -> bool:
    """Show which kernel branches shape 14 selects that no other shape does."""
    from appendix_cases import APPENDIX_SHAPES

    plans = {}
    for shape in APPENDIX_SHAPES:
        config = TransformerConfig(
            batch_size=shape.batch_size, d_model=shape.d_model,
            num_heads=shape.num_heads, seq_len=shape.seq_len,
            num_layers=shape.num_layers, causal=shape.causal,
            ffn_dim=shape.ffn_dim,
        )
        plans[shape.case_id] = plan_kernels(config, 4).describe()

    print("=== Kernel branch coverage ===")
    print(f"shape 14 plan: {plans[14]}\n")
    unique = []
    for field_text in plans[14].split():
        covered = sorted(i for i in plans if i != 14 and field_text in plans[i].split())
        if covered:
            print(f"  {field_text:32s} also in shapes {covered}")
        else:
            unique.append(field_text)
            print(f"  {field_text:32s} UNIQUE TO SHAPE 14")
    print()
    if unique == ["batch_chunk=1"]:
        print("Only the VALUE of batch_chunk is new. Shape 6 runs the same loop")
        print("at batch_chunk=1024, so the code path itself is under test.")
        return True
    print(f"Branches with no cover: {unique}")
    return not unique


def mode_ladder() -> bool:
    """Prove the fused attention kernel survives S=100000 at head_dim=64."""
    heads = SHAPE14.num_heads
    head_dim = SHAPE14.d_model // SHAPE14.num_heads

    print("=== Attention feasibility ladder ===")
    print(f"H={heads} head_dim={head_dim} float32 mask=causal, one batch row")
    print(f"{'S':>8} {'qkv MiB':>9} {'scores GiB':>11} {'ms':>10} "
          f"{'call MiB':>9} {'TFLOP/s':>8} {'path':>9}")

    passed = True
    for seq_len in (4096, 16384, 65536, SHAPE14.seq_len):
        mx.clear_cache()
        q = mx.random.normal((1, heads, seq_len, head_dim), dtype=mx.float32)
        k = mx.random.normal((1, heads, seq_len, head_dim), dtype=mx.float32)
        v = mx.random.normal((1, heads, seq_len, head_dim), dtype=mx.float32)
        mx.eval(q, k, v)
        # `reset_peak_memory()` sets the peak to zero, so subtract the
        # memory q, k and v already hold. What is left is what the call adds.
        mx.reset_peak_memory()
        base = mx.get_active_memory()

        for _ in range(2):
            start = time.perf_counter_ns()
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=head_dim ** -0.5, mask="causal"
            )
            mx.eval(out)
            mx.synchronize()
            ms = (time.perf_counter_ns() - start) / 1e6

        peak_mib = (mx.get_peak_memory() - base) / 2**20
        scores_gib = heads * seq_len * seq_len * 4 / 2**30
        # The fused kernel holds no score matrix, so its peak is the output.
        fused = peak_mib < scores_gib * 1024 * 0.5
        passed &= fused
        counted = 4 * heads * seq_len * seq_len * head_dim
        print(f"{seq_len:>8} {3*heads*seq_len*head_dim*4/2**20:>9.0f} "
              f"{scores_gib:>11.1f} {ms:>10.2f} {peak_mib:>9.1f} "
              f"{counted/1e12/(ms/1e3):>8.3f} "
              f"{'fused' if fused else 'FALLBACK':>9}")
        del q, k, v, out

    print()
    print("TFLOP/s counts the full S x S, as flops.py does. The fused kernel")
    print("runs the triangle only, so the executed rate is about half of it.")
    return passed


def mode_validate(q_block: int) -> bool:
    """
    Prove `frugal_forward()` equals `BaselineTransformer` where both can run.

    Check B rests on this. If a row here fails, the tail check means nothing.

    The sweep must exercise the query loop. A `q_block` at or above `seq_len`
    gives one block, and one block is the unblocked baseline again. So every
    case runs at a `q_block` below its own `seq_len` as well. Shape 13 is
    here because `S=1024` is the longest sequence the baseline can hold, and
    it is the only case that tests a many-block loop against a real answer.
    """
    from appendix_cases import APPENDIX_SHAPES

    print("=== frugal_forward() against BaselineTransformer ===")
    print(f"criterion: abs_error <= {ATOL:g} OR relative_error <= {RTOL:.2%}")
    print(f"{'#':>3} {'shape':<26} {'pad':>4} {'q_block':>8} {'blocks':>7} "
          f"{'max_abs':>11} {'max_rel':>11} {'result':>7}")

    # Shape 6 is 10000 rows of the shape 5 sequence, and `frugal_forward()`
    # splits the sequence, not the batch. It adds no case. Shape 14 is the
    # shape that cannot run.
    cases = [s for s in APPENDIX_SHAPES if s.case_id not in (6, 14)]
    passed = True
    for shape in cases:
        config = TransformerConfig(
            batch_size=shape.batch_size, d_model=shape.d_model,
            num_heads=shape.num_heads, seq_len=shape.seq_len,
            num_layers=shape.num_layers, causal=shape.causal,
            ffn_dim=shape.ffn_dim,
        )
        torch.manual_seed(SEED)
        baseline = BaselineTransformer(config).eval()

        # One block, and several blocks. Both must agree with the baseline.
        blocks = sorted({max(8, config.seq_len // 8), q_block})

        for padding in (0.0, 0.3):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(SEED)
            x = torch.randn(
                config.batch_size, config.seq_len, config.d_model,
                generator=generator, dtype=torch.float32,
            )
            if padding > 0:
                low = max(1, int(round(config.seq_len * (1.0 - padding))))
                lengths = torch.randint(
                    low=low, high=config.seq_len + 1,
                    size=(config.batch_size,), generator=generator,
                )
                positions = torch.arange(config.seq_len)[None, :]
                mask = positions < lengths[:, None]
                x = x.masked_fill(~mask[..., None], 0)
            else:
                mask = torch.ones(
                    config.batch_size, config.seq_len, dtype=torch.bool
                )

            with torch.inference_mode():
                reference = baseline(x, mask)

            for block in blocks:
                with torch.inference_mode():
                    candidate = frugal_forward(baseline, x, mask, q_block=block)
                result = compare_outputs(
                    reference, candidate, rtol=RTOL, atol=ATOL
                )
                passed &= result.passed
                count = -(-config.seq_len // block)
                label = (f"B{shape.batch_size} D{shape.d_model} "
                         f"H{shape.num_heads} S{shape.seq_len}")
                print(f"{shape.case_id:>3} {label:<26} {padding:>4.1f} "
                      f"{block:>8} {count:>7} "
                      f"{result.max_abs_error:>11.3e} "
                      f"{result.max_relative_error:>11.3e} "
                      f"{'PASS' if result.passed else 'FAIL':>7}")
                del candidate
            del reference, x, mask

    print(f"\nsummary: {'PASS' if passed else 'FAIL'}")
    if passed:
        print("frugal_forward() is a valid reference. --tail may use it.")
    else:
        print("frugal_forward() is NOT valid. Do not trust --tail.")
    return passed


def mode_check(rows: int, prefix: int, seed: int) -> bool:
    """
    Check A. The prefix of the shape 14 answer, against the real baseline.

    The model is causal, so output row `i` uses input rows `0..i` only.
    Therefore the baseline on the first `prefix` tokens must reproduce the
    first `prefix` rows of the full answer. The baseline runs unmodified,
    and `compare_outputs()` runs unmodified.
    """
    print("=== Check A: causal prefix, against BaselineTransformer ===")
    print(f"criterion: abs_error <= {ATOL:g} OR relative_error <= {RTOL:.2%}")
    scores_mib = SHAPE14.num_heads * prefix * prefix * 4 / 2**20
    print(f"rows={rows} prefix={prefix} tokens "
          f"(baseline score matrix {scores_mib:.0f} MiB per row)")

    baseline, optimized = build_models(SHAPE14, seed)
    print(f"plan: {optimized.plan.describe()}\n")

    print(f"{'row':>4} {'max_abs':>11} {'max_rel':>11} {'failed':>16} "
          f"{'mlx s':>8} {'result':>7}")
    passed = True
    for row in range(rows):
        result_row = run_row(
            optimized, SHAPE14, row, seed=seed, head_tokens=prefix
        )
        x, mask = make_row(SHAPE14, row, seed)
        with torch.inference_mode():
            reference = baseline(x[:, :prefix], mask[:, :prefix])
        del x, mask

        result = compare_outputs(
            reference, result_row.head, rtol=RTOL, atol=ATOL
        )
        passed &= result.passed
        print(f"{row:>4} {result.max_abs_error:>11.3e} "
              f"{result.max_relative_error:>11.3e} "
              f"{result.failed_elements:>7}/{result.total_elements:<8} "
              f"{result_row.compute_ms/1000:>8.2f} "
              f"{'PASS' if result.passed else 'FAIL':>7}")
        del reference, result_row

    print(f"\nsummary: {'PASS' if passed else 'FAIL'}")
    print("This checks the first tokens of each row. Run --tail for the end.")
    return passed


def mode_tail(row: int, tail: int, q_block: int, seed: int) -> bool:
    """
    Check B. The end of one row, against the frugal reference.

    Row 99999 attends every key, so no truncation reaches it. This runs the
    whole sequence through `frugal_forward()` on the CPU. Run --validate
    first.
    """
    print("=== Check B: the tail of one row, against frugal_forward() ===")
    print(f"criterion: abs_error <= {ATOL:g} OR relative_error <= {RTOL:.2%}")
    print(f"row={row} tail={tail} tokens q_block={q_block}")
    print("frugal_forward() is a reference, not the graded baseline. "
          "See --validate.\n")

    baseline, optimized = build_models(SHAPE14, seed)
    positions = tuple(range(SHAPE14.seq_len - tail, SHAPE14.seq_len))

    start = time.perf_counter_ns()
    mlx_row = run_row(optimized, SHAPE14, row, seed=seed, probe_positions=positions)
    print(f"MLX row {row}: {mlx_row.compute_ms/1000:.2f} s")

    x, mask = make_row(SHAPE14, row, seed)
    start = time.perf_counter_ns()
    with torch.inference_mode():
        reference = frugal_forward(baseline, x, mask, q_block=q_block)
    print(f"frugal reference: {(time.perf_counter_ns()-start)/1e9/60:.1f} min\n")
    del x, mask

    candidate = torch.from_numpy(
        np.stack([mlx_row.probes[position] for position in positions])
    )[None]
    result = compare_outputs(
        reference[:, -tail:], candidate, rtol=RTOL, atol=ATOL
    )
    print(f"max_abs={result.max_abs_error:.6g} "
          f"max_rel={result.max_relative_error:.6g} "
          f"failed={result.failed_elements}/{result.total_elements}")
    print(f"\nsummary: {'PASS' if result.passed else 'FAIL'}")
    return result.passed


def mode_time(rows: int, warmup: bool, seed: int) -> bool:
    """
    Time the MLX path. There is no baseline time, so there is no speedup.

    One pass gives one sample for each batch row, and every row does the same
    work. So a single pass yields both the total and a spread.
    """
    print("=== Shape 14 timing ===")
    print("CPU  : cannot run. BaselineSelfAttention asks for "
          f"{BASELINE_SCORE_BYTES/2**40:.1f} TiB of scores.")
    print("MPS  : cannot run. Same reason.")
    print("MLX  : measured below. No speedup column: there is no baseline "
          "time to divide by.\n")

    baseline, optimized = build_models(SHAPE14, seed)
    print(f"plan: {optimized.plan.describe()}")
    print(f"rows={rows} of {SHAPE14.batch_size}, one row for each chunk\n")

    if warmup:
        print("warmup: one row, discarded")
        run_row(optimized, SHAPE14, 0, seed=seed)

    print(f"{'row':>4} {'compute s':>10} {'total s':>10} {'checksum':>16}")
    results: List[RowResult] = []
    for row in range(rows):
        result = run_row(optimized, SHAPE14, row, seed=seed)
        results.append(result)
        print(f"{row:>4} {result.compute_ms/1000:>10.3f} "
              f"{result.total_ms/1000:>10.3f} {result.checksum:>16.4f}")

    compute = sorted(r.compute_ms for r in results)
    total = sorted(r.total_ms for r in results)
    median_compute = compute[len(compute) // 2]
    median_total = total[len(total) // 2]

    counted = model_flops(SHAPE14)
    executed = model_flops(SHAPE14, causal_aware=True)
    per_row = counted / SHAPE14.batch_size
    full_compute_s = median_compute * SHAPE14.batch_size / 1000
    full_total_s = median_total * SHAPE14.batch_size / 1000

    print(f"\nper row : median compute {median_compute/1000:.3f} s "
          f"(min {compute[0]/1000:.3f}, max {compute[-1]/1000:.3f})")
    print(f"per row : median total   {median_total/1000:.3f} s")
    scope = "measured" if rows == SHAPE14.batch_size else \
        f"EXTRAPOLATED from {rows} of {SHAPE14.batch_size} rows"
    print(f"full B={SHAPE14.batch_size}: compute {full_compute_s:.1f} s "
          f"({full_compute_s/60:.2f} min), total {full_total_s:.1f} s "
          f"({full_total_s/60:.2f} min)  [{scope}]")
    print(f"boundary: {100*(median_total-median_compute)/median_total:.1f}% "
          "of the call is the framework copy")

    tflops = per_row / 1e12 / (median_compute / 1000)
    mfu = provisional_mfu(counted, full_compute_s * 1000)
    executed_tflops = executed / 1e12 / full_compute_s
    executed_mfu = provisional_mfu(executed, full_compute_s * 1000)

    print(f"\n{'':28s} {'counted':>10} {'executed':>10}")
    print(f"{'GFLOP':28s} {counted/1e9:>10,.0f} {executed/1e9:>10,.0f}")
    print(f"{'MLX TFLOP/s':28s} {tflops:>10.3f} {executed_tflops:>10.3f}")
    print(f"{'MFU (MLX)':28s} {mfu*100:>9.1f}% {executed_mfu*100:>9.1f}%")
    print(f"{'of the 4.06 TFLOP/s ceiling':28s} "
          f"{tflops/4.06*100:>9.1f}% {executed_tflops/4.06*100:>9.1f}%")
    print(f"\ntokens/s                    : "
          f"{SHAPE14.batch_size*SHAPE14.seq_len/full_compute_s:,.0f}")
    print("\n`counted` credits the full S x S, which is what the baseline")
    print("calculates and what the graded MFU uses. `executed` credits the")
    print("causal triangle, which is what the fused kernel runs. Attention")
    print("is 97% of this shape, so the two differ by almost 2x, and the")
    print("counted MFU passes 100%. The EXECUTED column is the one to read")
    print("against the 82% practical ceiling in CLAUDE.md.")

    record = {
        "commit": _commit(),
        "shape": 14,
        "config": {
            "batch_size": SHAPE14.batch_size, "d_model": SHAPE14.d_model,
            "num_heads": SHAPE14.num_heads, "seq_len": SHAPE14.seq_len,
            "num_layers": SHAPE14.num_layers, "ffn_dim": SHAPE14.ffn_dim,
        },
        "rows_measured": rows,
        "full_batch_is_extrapolated": rows != SHAPE14.batch_size,
        "cpu_ms": None,
        "mps_ms": None,
        "cannot_run_reason": (
            f"BaselineSelfAttention materializes {BASELINE_SCORE_BYTES/2**40:.1f} "
            "TiB of scores"
        ),
        "mlx_row_compute_ms_median": median_compute,
        "mlx_row_total_ms_median": median_total,
        "mlx_full_compute_s": full_compute_s,
        "mlx_full_total_s": full_total_s,
        "gflop_counted": counted / 1e9,
        "gflop_executed": executed / 1e9,
        "mlx_tflops": tflops,
        "mlx_tflops_executed": executed_tflops,
        "mfu": mfu,
        "mfu_executed": executed_mfu,
        "plan": optimized.plan.describe(),
        "row_compute_ms": [r.compute_ms for r in results],
        "checksums": [r.checksum for r in results],
    }
    with open(RESULT_PATH, "w") as handle:
        json.dump(record, handle, indent=2)
    print(f"\nwrote {RESULT_PATH}")
    return True


def _commit() -> str:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return head + ("*" if dirty.strip() else "")
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--coverage", action="store_true",
                        help="which kernel branches shape 14 selects")
    parser.add_argument("--ladder", action="store_true",
                        help="prove the fused kernel survives S=100000")
    parser.add_argument("--validate", action="store_true",
                        help="prove frugal_forward() equals the baseline")
    parser.add_argument("--check", action="store_true",
                        help="check A: the causal prefix")
    parser.add_argument("--tail", action="store_true",
                        help="check B: the end of one row. About 15 minutes.")
    parser.add_argument("--time", action="store_true",
                        help="time the MLX path. About 10 minutes.")

    parser.add_argument("--rows", type=int, default=2,
                        help="batch rows to run (--check, --time)")
    parser.add_argument("--prefix", type=int, default=1024,
                        help="tokens the baseline reproduces (--check)")
    parser.add_argument("--tail-row", type=int, default=0)
    parser.add_argument("--tail-tokens", type=int, default=8)
    parser.add_argument("--q-block", type=int, default=256)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    modes = (args.coverage, args.ladder, args.validate,
             args.check, args.tail, args.time)
    if not any(modes):
        parser.print_help()
        return 2

    passed = True
    if args.coverage:
        passed &= mode_coverage()
    if args.ladder:
        passed &= mode_ladder()
    if args.validate:
        passed &= mode_validate(args.q_block)
    if args.check:
        passed &= mode_check(args.rows, args.prefix, args.seed)
    if args.tail:
        passed &= mode_tail(args.tail_row, args.tail_tokens,
                            args.q_block, args.seed)
    if args.time:
        passed &= mode_time(args.rows, not args.no_warmup, args.seed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

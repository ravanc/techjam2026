#!/usr/bin/env python3
"""
Correctness test for the dispatch paths of `UserOptimizedTransformer`.

`plan_kernels()` picks a kernel path from the shape, and `_mlx_transformer()`
picks again from `padded` and the dtype. A wrong pick gives a wrong number
and no exception. Row 27 and row 51 of `OPTIMIZATIONS.md` are both of that
kind. This file covers the paths that the 14 appendix shapes reach.

Scope: the 14 appendix shapes, float32, causal. Shape 14 does not run on
this machine, so it gets a plan check only.

Three layers run:

**L1 plan.** Assert `plan_kernels()` against a golden table. It runs no
kernel and it takes under one second. It also asserts that three branches
stay unselected: `causal_block`, `pad_head_dim` and the unfused QKV path.
No appendix shape reaches them, so they carry no test. The assertion tells
you the day one of them becomes live.

**L2 A/B.** Run the real plan and an all-off plan on the same weights and
the same input, then compare. Rows 33, 36, 46, 47 and 50 each claim to be
exact against the all-off path, so this tests the claim directly. It uses
no torch baseline, so it is cheap at shape 6 and shape 8.

**L3 kernels.** Compare each hoisted kernel against a plain MLX expression,
at the sizes and the tiles that the 13 shapes select.

Run it:

    .venv/bin/python3 tests/test_paths.py                    # every layer, 18 s
    .venv/bin/python3 tests/test_paths.py --layer 1          # the plan alone
    .venv/bin/python3 tests/test_paths.py --layer 2 --shape 6  # one shape

WHAT L2 COVERS ON A PADDED BATCH

A padded batch sets `plain` in `_mlx_transformer()`, so every fusion turns
off and the real plan meets the reference plan on most of the graph. Two
things still differ, and both are the ones that failed before:

  * the steel attention gate. The reference never uses the steel kernel, so
    a real plan that takes it on an array mask disagrees. That is row 27.
  * `fast_layer_norm` against `mx.fast.layer_norm`.

A removed row 27 gate is measured to FAIL this test. See the run log below.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import fields, replace
from typing import Dict, List

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import mlx.nn as mlx_nn  # noqa: E402
import torch  # noqa: E402

from appendix_cases import APPENDIX_SHAPES, SHAPES_BY_ID  # noqa: E402
from fast_layernorm import layer_norm_stats  # noqa: E402
from steel_gemm import (  # noqa: E402
    choose_final_ln_tile,
    choose_tile,
    layer_norm_constants,
    row_stats_reduce,
    steel_addmm,
)
from torch_transformer_benchmark import (  # noqa: E402
    LAYER_NORM_EPS,
    BaselineTransformer,
    UserOptimizedTransformer,
    copy_model_weights,
    generate_random_case,
    KernelPlan,
    plan_kernels,
)

# ---------------------------------------------------------------- L1 plan

# The plan of each appendix shape in float32. Generated from the code, then
# read and frozen. Update it in the same change that moves a plan, and say
# in the commit why the plan moved.
GOLDEN_PLANS: Dict[int, dict] = {
    1: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=True, fast_layer_norm=True, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=(32, 128, 16, 4, 1)),
    2: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=True, fast_layer_norm=True, defer_bias=True, fuse_gelu=None, fuse_ln_qkv=None, fuse_ln_ffn=None, fuse_stats_out=None, fuse_stats_ffn=None, final_ln=(32, 128, 16, 4, 1)),
    3: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=True, fast_layer_norm=True, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=(32, 128, 16, 4, 1)),
    4: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=True, fast_layer_norm=True, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=(32, 128, 16, 4, 1)),
    5: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=True, fast_layer_norm=True, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=(32, 128, 16, 4, 1)),
    6: dict(fuse_qkv=True, causal_block=None, batch_chunk=1024, pad_head_dim=None, steel_attention=True, fast_layer_norm=True, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=(32, 128, 16, 4, 1)),
    7: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=True, fast_layer_norm=True, defer_bias=True, fuse_gelu=(64, 32, 16, 2, 2), fuse_ln_qkv=(64, 32, 16, 2, 2), fuse_ln_ffn=(64, 32, 16, 2, 2), fuse_stats_out=(64, 32, 16, 2, 2), fuse_stats_ffn=(64, 32, 16, 2, 2), final_ln=(32, 32, 16, 4, 1)),
    8: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=False, fast_layer_norm=False, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=None),
    9: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=False, fast_layer_norm=True, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=(32, 128, 16, 4, 1)),
    10: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=False, fast_layer_norm=True, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=(32, 128, 16, 4, 1)),
    11: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=True, fast_layer_norm=True, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=(32, 128, 16, 4, 1)),
    12: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=True, fast_layer_norm=True, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=(32, 128, 16, 4, 1)),
    13: dict(fuse_qkv=True, causal_block=None, batch_chunk=None, pad_head_dim=None, steel_attention=True, fast_layer_norm=True, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=(32, 128, 16, 4, 1)),
    14: dict(fuse_qkv=True, causal_block=None, batch_chunk=1, pad_head_dim=None, steel_attention=False, fast_layer_norm=False, defer_bias=True, fuse_gelu=(32, 64, 16, 2, 2), fuse_ln_qkv=(32, 64, 16, 2, 2), fuse_ln_ffn=(32, 64, 16, 2, 2), fuse_stats_out=(32, 64, 16, 2, 2), fuse_stats_ffn=(32, 64, 16, 2, 2), final_ln=None),
}

# No appendix shape selects these three branches. They therefore carry no
# test. Assert that they stay unselected. See the module docstring.
DEAD_BRANCHES = {
    "causal_block": None,
    "pad_head_dim": None,
    "fuse_qkv": True,
}

PLAN_FIELDS: List[str] = [f.name for f in fields(KernelPlan)]


def layer1(verbose: bool = True) -> bool:
    """Assert the plan of each appendix shape against the golden table."""
    passed = True
    if verbose:
        print("=== L1 plan ===")
    for shape in APPENDIX_SHAPES:
        plan = plan_kernels(shape.config(), 4)
        golden = GOLDEN_PLANS[shape.case_id]

        wrong = [
            name for name in PLAN_FIELDS
            if getattr(plan, name) != golden[name]
        ]
        # The three branches that no shape reaches.
        dead = [
            name for name, value in DEAD_BRANCHES.items()
            if getattr(plan, name) != value
        ]

        ok = not wrong and not dead
        passed &= ok
        if verbose:
            state = "PASS" if ok else "FAIL"
            print(f"shape {shape.case_id:>2}  {state}")
            for name in wrong:
                print(f"    {name}: golden={golden[name]!r} "
                      f"now={getattr(plan, name)!r}")
            for name in dead:
                print(f"    DEAD BRANCH IS NOW LIVE: {name} = "
                      f"{getattr(plan, name)!r}. It carries no test.")

    if verbose:
        print(f"L1 {'PASS' if passed else 'FAIL'}\n")
    return passed


# ---------------------------------------------------------------- L3 kernels

# The tolerance of a float32 GEMM. The steel kernel and MLX accumulate in a
# different order, so the two answers differ in the last bits. The threshold
# scales with the size of the reference, because the error of a K step dot
# product grows with the magnitude of the result.
GEMM_RTOL = 2e-5


def _mismatch(reference: mx.array, candidate: mx.array) -> tuple:
    """Return (max_abs_error, scaled_error). Both are floats."""
    error = mx.max(mx.abs(candidate - reference)).item()
    scale = max(mx.max(mx.abs(reference)).item(), 1e-6)
    return error, error / scale


def kernel_cases() -> List[dict]:
    """
    Every (epilogue, M, N, K, tile) that the 13 shapes run.

    The list comes from `APPENDIX_SHAPES` and `plan_kernels()`, not from a
    frozen table, so it follows the planner. L1 holds the planner itself.
    """
    cases = {}
    for shape in APPENDIX_SHAPES:
        if not shape.enabled:
            continue
        config = shape.config()
        plan = plan_kernels(config, 4)
        rows = (plan.batch_chunk or config.batch_size) * config.seq_len
        d_model, ffn_dim = config.d_model, config.ffn_dim
        table = (
            ("gelu", plan.fuse_gelu, ffn_dim, d_model, True),
            ("ln_qkv", plan.fuse_ln_qkv, 3 * d_model, d_model, False),
            ("ln_ffn", plan.fuse_ln_ffn, ffn_dim, d_model, True),
            ("st_out", plan.fuse_stats_out, d_model, d_model, True),
            ("st_ffn", plan.fuse_stats_ffn, d_model, ffn_dim, True),
            ("finln", plan.final_ln, d_model, ffn_dim, True),
        )
        for kind, tile, n, k, transpose_b in table:
            if tile is None:
                continue
            key = (kind, rows, n, k, tile, transpose_b)
            cases.setdefault(key, []).append(shape.case_id)
    return [
        dict(kind=key[0], m=key[1], n=key[2], k=key[3], tile=key[4],
             transpose_b=key[5], shapes=ids)
        for key, ids in sorted(cases.items())
    ]


def _run_case(case: dict, eps: float) -> tuple:
    """Run one kernel case. Return (ok, label, max_abs, scaled)."""
    kind, m, n, k = case["kind"], case["m"], case["n"], case["k"]
    bm, bn, bk, wm, wn = case["tile"]
    transpose_b = case["transpose_b"]

    mx.random.seed(1234)
    a = mx.random.normal((m, k))
    weight = mx.random.normal((n, k) if transpose_b else (k, n))
    effective = weight.T if transpose_b else weight
    proj_bias = mx.random.normal((n,))

    if kind in ("ln_qkv", "ln_ffn"):
        # Row 46. The GEMM absorbs the LayerNorm above it.
        gain = mx.random.normal((k,))
        ln_bias = mx.random.normal((k,))
        carry = mx.random.normal((k,))
        packed, lnc1, lnc2, c3 = layer_norm_constants(
            gain, ln_bias, weight, proj_bias, transpose_b=transpose_b,
            carry=carry)
        stats = layer_norm_stats(a, eps, pre_bias=carry)
        gelu = kind == "ln_ffn"
        got = steel_addmm(
            c3, a, packed, transpose_b=transpose_b, gelu=gelu,
            bm=bm, bn=bn, bk=bk, wm=wm, wn=wn,
            rowstat=stats, lnc1=lnc1, lnc2=lnc2)
        normed = mx.fast.layer_norm(a + carry, gain, ln_bias, eps)
        want = mx.addmm(proj_bias, normed, effective)
        if gelu:
            want = mlx_nn.gelu(want)

    elif kind in ("st_out", "st_ffn"):
        # Row 47. The GEMM takes the statistics of its own output.
        residual = mx.random.normal((m, n))
        carry = mx.random.normal((n,))
        got, partials = steel_addmm(
            residual, a, weight, transpose_b=transpose_b,
            bm=bm, bn=bn, bk=bk, wm=wm, wn=wn,
            row_stats=True, row_carry=carry)
        want = mx.addmm(residual, a, effective)
        # The statistics must match the separate kernel as well.
        got_stats = row_stats_reduce(partials, n, eps)
        want_stats = layer_norm_stats(want, eps, pre_bias=carry)
        mx.eval(got, want, got_stats, want_stats)
        mx.synchronize()
        out_error = _mismatch(want, got)
        stat_error = _mismatch(want_stats, got_stats)
        error = max(out_error, stat_error, key=lambda pair: pair[1])
        ok = out_error[1] <= GEMM_RTOL and stat_error[1] <= GEMM_RTOL
        label = (f"{kind:7s} M={m:<7} N={n:<5} K={k:<5} "
                 f"{bm}x{bn}x{bk} w{wm}{wn}")
        return ok, label, error[0], error[1]

    elif kind == "finln":
        # Row 50. The last `ffn_out` applies the final LayerNorm.
        residual = mx.random.normal((m, n))
        carry = mx.random.normal((n,))
        gain = mx.random.normal((n,))
        final_bias = mx.random.normal((n,))
        got = steel_addmm(
            residual, a, weight, transpose_b=transpose_b,
            bm=bm, bn=bn, bk=bk, wm=wm, wn=wn,
            final_gain=gain, final_bias=final_bias,
            row_carry=carry, final_eps=eps)
        want = mx.fast.layer_norm(
            mx.addmm(residual, a, effective) + carry, gain, final_bias, eps)

    elif kind == "gelu":
        # Row 33. The GEMM applies GELU in its epilogue.
        got = steel_addmm(
            proj_bias, a, weight, transpose_b=transpose_b, gelu=True,
            bm=bm, bn=bn, bk=bk, wm=wm, wn=wn)
        want = mlx_nn.gelu(mx.addmm(proj_bias, a, effective))

    else:
        raise ValueError(f"unknown kind {kind}")

    mx.eval(got, want)
    mx.synchronize()
    max_abs, scaled = _mismatch(want, got)
    label = f"{kind:7s} M={m:<7} N={n:<5} K={k:<5} {bm}x{bn}x{bk} w{wm}{wn}"
    return scaled <= GEMM_RTOL, label, max_abs, scaled


def _refusals() -> tuple:
    """
    Assert that the kernels REFUSE a geometry they cannot serve.

    A refusal that stops refusing is a silent error. Row 51 records the
    cause: MLX's `BlockLoader` truncates its integer division with no guard,
    so a tile with inexact thread geometry reads past the operand and gives
    a wrong answer.
    """
    checks = []

    # Row 51. `bn` 48 and 96 give inexact geometry. `choose_tile()` must not
    # return them, and `steel_addmm()` must raise on them.
    for bn in (48, 96):
        checks.append((
            f"steel_addmm refuses bn={bn}",
            lambda bn=bn: _raises(
                lambda: steel_addmm(
                    mx.zeros((bn,)), mx.zeros((64, 16)),
                    mx.zeros((bn, 16)), bm=32, bn=bn, bk=16, wm=2, wn=2))))

    # `choose_tile()` must return None when no tile divides the problem.
    checks.append((
        "choose_tile refuses M=33",
        lambda: choose_tile(33, 64, 64) is None))
    # `choose_final_ln_tile()` must refuse a width with inexact geometry.
    checks.append((
        "choose_final_ln_tile refuses N=96",
        lambda: choose_final_ln_tile(1024, 96, 128) is None))
    # An unaligned K must raise, not run the untested leftover loop.
    checks.append((
        "steel_addmm refuses K=24 against bk=16",
        lambda: _raises(
            lambda: steel_addmm(
                mx.zeros((32,)), mx.zeros((32, 24)), mx.zeros((32, 24)),
                bm=32, bn=32, bk=16, wm=2, wn=2))))

    passed = True
    lines = []
    for name, check in checks:
        ok = bool(check())
        passed &= ok
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return passed, lines


def _raises(call) -> bool:
    try:
        result = call()
        mx.eval(result)
    except ValueError:
        return True
    except Exception:
        return False
    return False


def layer3(verbose: bool = True) -> bool:
    """Compare each hoisted kernel against a plain MLX expression."""
    passed = True
    if verbose:
        print("=== L3 kernels ===")

    cases = kernel_cases()
    for case in cases:
        ok, label, max_abs, scaled = _run_case(case, LAYER_NORM_EPS)
        passed &= ok
        if verbose:
            state = "PASS" if ok else "FAIL"
            print(f"{state}  {label}  max_abs={max_abs:.3e} "
                  f"scaled={scaled:.2e}  shapes={case['shapes']}")

    if verbose:
        print(f"  {len(cases)} kernel cases")
    refused, lines = _refusals()
    passed &= refused
    if verbose:
        print("refusals:")
        for line in lines:
            print(line)
        print(f"L3 {'PASS' if passed else 'FAIL'}\n")
    return passed


# ---------------------------------------------------------------- L2 A/B

# The two paths run the same arithmetic in a different order, so they differ
# in the last bits. The threshold is on the error divided by the size of the
# reference. The harness thresholds are atol=0.002 and rtol=0.02, so this is
# far tighter than the test that grades the model.
AB_RTOL = 1e-4


def all_off_plan(plan: KernelPlan) -> KernelPlan:
    """
    The reference plan. Every fusion is off, so the model runs plain MLX
    operations: `mx.fast.layer_norm`, `mx.addmm`, `mlx_nn.gelu` and
    `mx.fast.scaled_dot_product_attention`.

    `batch_chunk` stays. It splits the batch and changes no arithmetic, and
    shape 6 does not fit the working set without it. `chunk_case()` tests
    the chunk loop on its own.
    """
    return KernelPlan(
        fuse_qkv=True,
        causal_block=None,
        batch_chunk=plan.batch_chunk,
        pad_head_dim=None,
        steel_attention=False,
        fast_layer_norm=False,
        defer_bias=False,
        fuse_gelu=None,
        fuse_ln_qkv=None,
        fuse_ln_ffn=None,
        fuse_stats_out=None,
        fuse_stats_ffn=None,
        final_ln=None,
    )


def _run_pair(config, plan_a, plan_b, seed: int, padding_ratio: float):
    """
    Run two plans on the same weights and the same input.

    Return (max_abs_error, scaled_error, has_nan).
    """
    device = torch.device("cpu")
    torch.manual_seed(seed)
    weights = BaselineTransformer(config)

    x, valid_mask = generate_random_case(
        config=config, device=device, dtype=torch.float32, seed=seed,
        padding_ratio=padding_ratio, input_scale=1.0,
    )

    outputs = []
    for plan in (plan_a, plan_b):
        model = UserOptimizedTransformer(config)
        model.plan_override = plan
        copy_model_weights(weights, model)
        with torch.inference_mode():
            outputs.append(model(x, valid_mask).clone())
        del model

    got, want = outputs
    error = (got - want).abs()
    max_abs = float(error.max().item())
    scale = max(float(want.abs().max().item()), 1e-6)
    has_nan = bool(torch.isnan(got).any() or torch.isnan(want).any())
    return max_abs, max_abs / scale, has_nan


def layer2(verbose: bool = True, seed: int = 1234,
           only: List[int] = None) -> bool:
    """Compare the real plan against the all-off plan, on every shape."""
    passed = True
    if verbose:
        print("=== L2 A/B ===")
        print(f"{'#':>3} {'shape':26s} {'padding':>8}  "
              f"{'max_abs':>10} {'scaled':>9}  state")

    for shape in APPENDIX_SHAPES:
        if not shape.enabled:
            continue
        if only and shape.case_id not in only:
            continue
        config = shape.config()
        real = plan_kernels(config, 4)
        reference = all_off_plan(real)

        for padding_ratio in (0.0, 0.3):
            max_abs, scaled, has_nan = _run_pair(
                config, real, reference, seed, padding_ratio)
            ok = scaled <= AB_RTOL and not has_nan
            passed &= ok
            if verbose:
                name = (f"B{shape.batch_size} D{shape.d_model} "
                        f"H{shape.num_heads} S{shape.seq_len}")
                state = "PASS" if ok else "FAIL"
                nan = "  NaN IN OUTPUT" if has_nan else ""
                print(f"{shape.case_id:>3} {name:26s} {padding_ratio:>8.1f}  "
                      f"{max_abs:>10.3e} {scaled:>9.2e}  {state}{nan}")
            gc.collect()

    # The chunk loop. It splits the batch, so it must change no number.
    if not only or 1 in (only or []):
        ok, max_abs, scaled = chunk_case(seed)
        passed &= ok
        if verbose:
            print(f"chunk loop, shape 1, chunk=16 against chunk=None: "
                  f"max_abs={max_abs:.3e} scaled={scaled:.2e} "
                  f"{'PASS' if ok else 'FAIL'}")

    if verbose:
        print(f"L2 {'PASS' if passed else 'FAIL'}\n")
    return passed


def chunk_case(seed: int = 1234):
    """
    Test `batch_chunk` on its own.

    Only shape 6 selects a chunk, and shape 6 does not fit the working set
    without one, so the chunk loop has no A/B partner there. Run shape 1
    with a forced chunk instead. The chunk splits the batch alone, so the
    output must be BIT EXACT.
    """
    config = SHAPES_BY_ID[1].config()
    plan = plan_kernels(config, 4)
    chunked = replace(plan, batch_chunk=16)
    max_abs, scaled, has_nan = _run_pair(config, chunked, plan, seed, 0.0)
    return max_abs == 0.0 and not has_nan, max_abs, scaled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", default="1,2,3",
                        help="which layers to run, for example 1,3")
    parser.add_argument("--shape", default="",
                        help="limit L2 to these appendix shapes")
    args = parser.parse_args()
    wanted = {int(part) for part in args.layer.split(",")}
    only = [int(p) for p in args.shape.split(",") if p.strip()]

    passed = True
    if 1 in wanted:
        passed &= layer1()
    if 2 in wanted:
        passed &= layer2(only=only)
    if 3 in wanted:
        passed &= layer3()
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

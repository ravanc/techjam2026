"""
Hoist MLX's own flash attention kernel out of its headers, and instantiate
it at a head_dim that MLX does not ship.

WHY THIS EXISTS

`mx.fast.scaled_dot_product_attention` reaches the fused `steel_attention`
kernel for five head_dim values only: 64, 72, 80, 96 and 128. Every other
width falls back to a kernel that materializes the whole B x H x S x S score
matrix. See `references/mlx-tensorops.md` section 0.

MLX gives no way to name a kernel and run it, so the fused kernel cannot be
called at head_dim 32 or 8 through the public API. But the wheel ships the
Metal SOURCE of that kernel:

    mlx/include/mlx/backend/metal/kernels/steel/attn/

The kernel is a C++ template. `BD` is the head width, and the template does
not restrict it beyond `BD % 8 == 0`. MLX compiled five values. Nothing
stops us from compiling another one.

HOW

`mx.fast.metal_kernel` JIT-compiles Metal source. Its JIT cannot `#include`
the steel headers, because MLX embeds a fixed header set in the binary and
the steel headers are not in it. So this module reads the headers off disk
and inlines them.

Three edits to the hoisted kernel, and no others:

1. `[[kernel]]` becomes a plain function. `mx.fast.metal_kernel` writes its
    own kernel signature, so the hoisted code becomes a callee.
2. The `[[function_constant]]` flags become compile-time `constexpr bool`.
    We know the shape when we compile, so a runtime constant is not needed.
3. `constant AttnParams*` becomes `thread AttnParams*`. The caller builds
    the parameters on the stack instead of in a buffer.

The arithmetic is untouched. This is Apple's kernel, at a new width.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

KERNELS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels",
)

# Dependency order. `attn.h` includes the first six, so inline them first.
_CHAIN = [
    "steel/defines.h",
    "steel/utils/type_traits.h",
    "steel/utils/integral_constant.h",
    "steel/utils.h",
    "steel/attn/transforms.h",
    "steel/attn/loader.h",
    "steel/attn/mma.h",
    "steel/attn/params.h",
    "steel/gemm/params.h",
]

_MLX_INCLUDE = re.compile(r'^\s*#include\s+"mlx/.*"\s*$', re.M)
_PRAGMA_ONCE = re.compile(r"^\s*#pragma once\s*$", re.M)


def _read(rel: str) -> str:
    with open(os.path.join(KERNELS, rel)) as handle:
        text = handle.read()
    text = _MLX_INCLUDE.sub("", text)
    text = _PRAGMA_ONCE.sub("", text)
    return text


# The kernel declaration in `steel_attention.h`, and what it becomes. The
# body between them is Apple's, unchanged.
_KERNEL_DECL = re.compile(
    r"\[\[kernel,\s*max_total_threads_per_threadgroup\(WM \* WN \* 32\)\]\]"
    r"\s*void attention\(.*?\)\s*\{",
    re.S,
)

_CALLEE_DECL = """METAL_FUNC void attention(
    const device T* Q,
    const device T* K,
    const device T* V,
    device T* O,
    const thread AttnParams* params,
    const thread AttnMaskParams* mask_params,
    const device MaskType* mask,
    const device T* sinks,
    uint simd_lane_id,
    uint simd_group_id,
    uint3 tid,
    uint3 lid,
    threadgroup T* Q_smem,
    threadgroup T* KV_smem) {"""

_FUNCTION_CONSTANTS = re.compile(
    r"constant bool (\w+) \[\[function_constant\(\d+\)\]\];")


def _hoist_kernel(align_q: bool, align_k: bool, causal: bool) -> str:
    """
    Read `steel_attention.h` and make it a callable device function.

    The three edits are listed in the module docstring. Nothing else in the
    file changes, so the arithmetic stays Apple's.
    """
    text = _read("steel/attn/kernels/steel_attention.h")

    flags = {
        "align_Q": align_q, "align_K": align_k,
        "has_mask": False, "do_causal": causal, "has_sinks": False,
    }

    def replace_flag(match: "re.Match") -> str:
        name = match.group(1)
        value = "true" if flags[name] else "false"
        return f"constexpr constant bool {name} = {value};"

    text = _FUNCTION_CONSTANTS.sub(replace_flag, text)
    if _KERNEL_DECL.search(text) is None:
        raise RuntimeError(
            "the kernel declaration in steel_attention.h did not match; "
            "MLX changed the file, so re-check this module")
    text = _KERNEL_DECL.sub(_CALLEE_DECL, text, count=1)

    # Metal rejects a threadgroup variable in a non-kernel function, so the
    # two declarations move to the caller and arrive as pointers. The sizes
    # are unchanged: `smem_floats()` recomputes what the kernel computed.
    declarations = (
        "  threadgroup T Q_smem[BQ * (BD + padQ)];\n"
        "  threadgroup T KV_smem[tgp_mem_s];")
    if declarations not in text:
        raise RuntimeError(
            "the threadgroup declarations in steel_attention.h did not "
            "match; MLX changed the file, so re-check this module")
    text = text.replace(declarations, "", 1)
    return text


def smem_floats(bq: int, bk: int, bd: int, itemsize: int = 4) -> Tuple[int, int]:
    """
    Repeat the threadgroup sizes that `steel_attention.h` calculates.

    Keep this in step with the `padQ`, `padK`, `padV`, `tgp_mem_0` and
    `tgp_mem_1` lines of that file.
    """
    pad = 16 // itemsize
    q_smem = bq * (bd + pad)
    kv_smem = max((bk + pad) * bd, bk * (bd + pad))
    return q_smem, kv_smem


def build_header(align_q: bool, align_k: bool, causal: bool) -> str:
    """Inline every steel header, then the hoisted kernel."""
    parts = [
        "#ifndef METAL_FUNC",
        "#define METAL_FUNC inline __attribute__((__always_inline__))",
        "#endif",
        "using namespace metal;",
    ]
    parts.extend(_read(name) for name in _CHAIN)
    parts.append("using namespace mlx::steel;")
    parts.append(_hoist_kernel(align_q, align_k, causal))
    return "\n".join(parts)


_CACHE: Dict[tuple, object] = {}


def _source(batch, heads, seq, head_dim, scale, bq, bk, wm, wn,
            out_strides) -> str:
    """
    The kernel body. It builds AttnParams on the stack, then calls the
    hoisted kernel.

    Every shape value is a literal, because the JIT compiles one kernel per
    shape anyway. A literal costs nothing and it lets the compiler fold the
    block counts.

    The INPUT strides are not literals. `mx.fast.metal_kernel` gives the
    kernel a `q_strides` buffer that holds the true strides of the array it
    binds, so the kernel reads whatever layout the caller passes. That is
    what lets `ensure_row_contiguous` stay False. See `steel_attention()`.

    The OUTPUT strides are literals, because this module chooses the output
    layout itself.
    """
    nq = -(-seq // bq)
    nk = -(-seq // bk)
    q_smem, kv_smem = smem_floats(bq, bk, head_dim)
    return f"""
  threadgroup float Q_smem[{q_smem}];
  threadgroup float KV_smem[{kv_smem}];

  AttnParams p;
  p.B = {batch};
  p.H = {heads};
  p.D = {head_dim};
  p.qL = {seq};
  p.kL = {seq};
  p.gqa_factor = 1;
  p.scale = {scale!r}f;
  p.NQ = {nq};
  p.NK = {nk};
  p.NQ_aligned = {seq // bq};
  p.NK_aligned = {seq // bk};
  p.qL_rem = {seq - (seq // bq) * bq};
  p.kL_rem = {seq - (seq // bk) * bk};
  p.qL_off = 0;

  // The batch, the head and the sequence strides of the array that MLX
  // bound. The kernel needs the last axis contiguous, and nothing else.
  p.Q_strides[0] = q_strides[0];
  p.Q_strides[1] = q_strides[1];
  p.Q_strides[2] = q_strides[2];
  p.K_strides[0] = k_strides[0];
  p.K_strides[1] = k_strides[1];
  p.K_strides[2] = k_strides[2];
  p.V_strides[0] = v_strides[0];
  p.V_strides[1] = v_strides[1];
  p.V_strides[2] = v_strides[2];
  p.O_strides[0] = {out_strides[0]};
  p.O_strides[1] = {out_strides[1]};
  p.O_strides[2] = {out_strides[2]};

  AttnMaskParams mp;
  mp.M_strides[0] = 0;
  mp.M_strides[1] = 0;
  mp.M_strides[2] = 0;

  attention<float, {bq}, {bk}, {head_dim}, {wm}, {wn}, float, float>(
      q, k, v, out, &p, &mp,
      (const device float*)nullptr, (const device float*)nullptr,
      thread_index_in_simdgroup,
      simdgroup_index_in_threadgroup,
      threadgroup_position_in_grid,
      thread_position_in_threadgroup,
      Q_smem, KV_smem);
"""


# Metal gives a threadgroup 32 KiB. `steel_attention.h` holds Q and one of
# K or V there, so a wide head with a wide K block does not fit. Check it
# before you compile, because the failure is a compile error at run time.
THREADGROUP_BYTES = 32 * 1024


def fits_threadgroup(bq: int, bk: int, bd: int, itemsize: int = 4) -> bool:
    """Return True when the two threadgroup buffers fit."""
    q_smem, kv_smem = smem_floats(bq, bk, bd, itemsize)
    return (q_smem + kv_smem) * itemsize <= THREADGROUP_BYTES


def supports(head_dim: int, bq: int = 32, bk: int = 32) -> bool:
    """
    Return True when this module can run the fused kernel at `head_dim`.

    The kernel needs `BD % 8 == 0`, because its MMA fragment is 8 wide. It
    also needs the threadgroup buffers to fit.
    """
    return head_dim % 8 == 0 and fits_threadgroup(bq, bk, head_dim)


def steel_attention(
    q: mx.array, k: mx.array, v: mx.array, scale: float,
    causal: bool = True, bq: int = 32, bk: int = 32,
    wm: int = 4, wn: int = 1, head_last: bool = False,
) -> mx.array:
    """
    Run MLX's flash attention kernel at ANY head_dim that is a multiple of 8.

    `q`, `k` and `v` are [B, H, S, D] float32. `S` must match between q and
    k. They do NOT have to be contiguous: the only requirement is that the
    last axis has stride 1. A strided view is the normal case, and it costs
    no copy. See "NO COPY" below.

    `head_last` picks the output layout:

      False  the output is [B, H, S, D], contiguous.
      True   the output is [B, S, H, D], contiguous.

    Use `head_last=True` when the caller merges the heads next. The kernel
    then writes the merged layout directly, so `reshape(B, S, H * D)` is a
    free view instead of a copy.

    NO COPY

    `ensure_row_contiguous` stays False. With it True, MLX copies q, k and v
    into fresh contiguous buffers before every launch. The model builds them
    as strided views of one fused QKV buffer, so that copy always ran, and it
    cost more than the kernel: at the shape 6 chunk the call took 5.364 ms on
    the views against 2.223 ms on ready-made contiguous arrays, and the copy
    alone was 3.328 ms.

    The copy buys nothing. `steel_attention.h` reads Q, K and V through
    `params->Q_strides`, so it already handles any layout with a contiguous
    last axis. `mx.fast.metal_kernel` passes the true strides of each bound
    array in a `q_strides` buffer, and `_source()` copies them into
    AttnParams. See OPTIMIZATIONS.md row 34.
    """
    batch, heads, seq, head_dim = q.shape
    if head_dim % 8:
        raise ValueError(f"head_dim {head_dim} must be a multiple of 8")
    if q.dtype != mx.float32:
        raise ValueError("float32 only")

    # The output is contiguous in the layout the caller asked for. These are
    # the (batch, head, sequence) strides that place element [b, h, s, d].
    if head_last:
        out_shape = (batch, seq, heads, head_dim)
        out_strides = (seq * heads * head_dim, head_dim, heads * head_dim)
    else:
        out_shape = (batch, heads, seq, head_dim)
        out_strides = (heads * seq * head_dim, seq * head_dim, head_dim)

    # `_source()` bakes the shape in as literals, so every one of them must
    # be in the key. Leaving `seq` out gave a silent wrong answer: an S=256
    # call reused the S=128 kernel and read the wrong strides. The INPUT
    # strides are not in the key, because they are not literals any more.
    key = (batch, heads, seq, head_dim, bq, bk, wm, wn, causal, head_last)
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"steel_attn_bd{head_dim}_bq{bq}_bk{bk}"
                 f"_b{batch}_h{heads}_s{seq}_hl{int(head_last)}",
            input_names=["q", "k", "v"],
            output_names=["out"],
            header=build_header(seq % bq == 0, seq % bk == 0, causal),
            source=_source(batch, heads, seq, head_dim, scale, bq, bk, wm, wn,
                           out_strides),
            ensure_row_contiguous=False,
        )
        _CACHE[key] = kernel

    nq = -(-seq // bq)
    outputs = kernel(
        inputs=[q, k, v],
        grid=(nq * 32, heads * wm, batch * wn),
        threadgroup=(32, wm, wn),
        output_shapes=[out_shape],
        output_dtypes=[q.dtype],
    )
    return outputs[0]

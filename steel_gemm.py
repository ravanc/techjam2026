"""
Hoist MLX's own steel GEMM kernel out of its headers, and give it an
epilogue that MLX does not expose.

WHY THIS EXISTS

`mlx_nn.gelu(mx.addmm(b, h, w.T))` runs two kernels. The GEMM writes the
whole activation to DRAM, and GELU reads it back and writes it again. That
second pass costs 0.977 ms at the shape 6 chunk, which is 7.1% of the layer.
`mx.compile` does not fuse it. See OPTIMIZATIONS.md row 33.

`steel/gemm/mma.h` holds two epilogue hooks. Both act on `Ctile`, the
accumulator in registers, before the kernel stores it:

    apply_epilogue(C, ldc, fdc, epilogue_op)   // binary. Adds the C operand
    apply_epilogue(epilogue_op)                // unary. One value in, one out

`mx.addmm` already uses the binary hook with `TransformAdd`. The unary hook
is the one MLX never exposes, and it is exactly where GELU belongs: after
the bias add, before the store. So the extra pass disappears.

HOW

The method is row 25's, applied to `steel/gemm/` in place of `steel/attn/`.
`mx.fast.metal_kernel` JIT-compiles Metal source, but its JIT cannot
`#include` the steel headers, because MLX embeds a fixed header set in the
binary and the steel headers are not in it. So this module reads the headers
off disk and inlines them.

Five edits to the hoisted kernel, and no others:

1. `[[kernel]]` becomes a plain function. `mx.fast.metal_kernel` writes its
    own kernel signature, so the hoisted code becomes a callee.
2. The `[[function_constant]]` flags become compile-time `constexpr bool`.
    We know the shape when we compile, so a runtime constant is not needed.
3. `constant GEMMParams*` becomes `thread GEMMParams*`. The caller builds
    the parameters on the stack instead of in a buffer.
4. The `threadgroup` arrays move to the caller. Metal rejects a threadgroup
    variable in a non-kernel function.
5. The `has_batch` branch goes. This module never batches, and Metal type
    checks the dead branch. See `_BATCH_BRANCH`.

The arithmetic is untouched. This is Apple's kernel, with one added
epilogue struct.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

import mlx.core as mx

KERNELS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels",
)

# Dependency order. `gemm.h` includes the first four of the gemm headers, so
# inline them first. The four `steel/` headers are the same ones that
# `steel_attention.py` inlines.
_CHAIN = [
    "steel/defines.h",
    "steel/utils/type_traits.h",
    "steel/utils/integral_constant.h",
    "steel/utils.h",
    "steel/gemm/transforms.h",
    "steel/gemm/params.h",
    "steel/gemm/loader.h",
    "steel/gemm/mma.h",
    "steel/gemm/gemm.h",
]

_MLX_INCLUDE = re.compile(r'^\s*#include\s+"mlx/.*"\s*$', re.M)
_PRAGMA_ONCE = re.compile(r"^\s*#pragma once\s*$", re.M)


def _read(rel: str) -> str:
    with open(os.path.join(KERNELS, rel)) as handle:
        text = handle.read()
    text = _MLX_INCLUDE.sub("", text)
    text = _PRAGMA_ONCE.sub("", text)
    return text


# The batch branch of `steel_gemm_fused.h`, and what replaces it.
#
# This module never batches: it flattens [..., M, K] to one [M, K] and runs
# a single GEMM, so `has_batch` is always false and `tid.z` is always 0.
# The branch cannot simply stay dead, because Metal type checks it: it reads
# `batch_strides` through `const constant auto*`, and a hoisted callee gets
# its parameters in the `thread` address space, not `constant`.
#
# So the branch goes, and the `else` body stays. That body is the identity
# at `tid.z == 0`, so the arithmetic does not change.
_BATCH_BRANCH = """  // Adjust for batch
  if (has_batch) {
    const constant auto* A_bstrides = batch_strides;
    const constant auto* B_bstrides = batch_strides + params->batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, A_bstrides, B_bstrides, params->batch_ndim);

    A += batch_offsets.x;
    B += batch_offsets.y;

    if (use_out_source) {
      const constant auto* C_bstrides = B_bstrides + params->batch_ndim;
      C += elem_to_loc(tid.z, batch_shape, C_bstrides, params->batch_ndim);
    }
  } else {
    A += params->batch_stride_a * tid.z;
    B += params->batch_stride_b * tid.z;

    if (use_out_source) {
      C += addmm_params->batch_stride_c * tid.z;
    }
  }"""

_BATCH_REPLACEMENT = """  // Adjust for batch. `has_batch` is always false here; see steel_gemm.py.
  A += params->batch_stride_a * tid.z;
  B += params->batch_stride_b * tid.z;

  if (use_out_source) {
    C += addmm_params->batch_stride_c * tid.z;
  }"""

# Metal has no `erf`. MLX ships its own in `erf.h`, and `unary_ops.h` calls
# it for `mx.erf`. Inline the same file, so this kernel and `mlx_nn.gelu`
# call the SAME approximation and agree bit for bit. `erf.h` needs
# `expm1f.h`, so that goes first.
_ERF_CHAIN = ["expm1f.h", "erf.h"]

# The GELU that goes in the unary hook. It must match `mlx_nn.gelu`, which
# is the exact erf form, not the tanh approximation:
#
#     gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
_GELU_TRANSFORM = """
template <typename OutT, typename InT>
struct TransformGelu {
  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(
        x * 0.5f * (1.0f + erf(x * 0.7071067811865475f)));
  }
  static METAL_FUNC OutT apply(InT x, OutT) {
    return apply(x);
  }
};
"""

# The kernel declaration in `steel_gemm_fused.h`, and what it becomes. The
# body between them is Apple's, unchanged.
_KERNEL_DECL = re.compile(
    r"\[\[kernel,\s*max_total_threads_per_threadgroup\(WM\s*\*\s*WN\s*\*\s*32\)\]\]"
    r"\s*void gemm\(.*?\)\s*\{",
    re.S,
)

_CALLEE_DECL = """METAL_FUNC void gemm(
    const device T* A,
    const device T* B,
    const device T* C,
    device T* D,
    const thread GEMMParams* params,
    const thread GEMMAddMMParams* addmm_params,
    uint simd_lane_id,
    uint simd_group_id,
    uint3 tid,
    uint3 lid,
    threadgroup T* As,
    threadgroup T* Bs) {"""

_FUNCTION_CONSTANTS = re.compile(
    r"constant bool (\w+) \[\[function_constant\(\d+\)\]\];")

# The two threadgroup declarations that move to the caller.
_TGP_DECLS = """  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];"""


def _hoist_kernel(align_m: bool, align_n: bool, align_k: bool,
                  gelu: bool) -> str:
    """
    Read `steel_gemm_fused.h` and make it a callable device function.

    The five edits are listed in the module docstring. Nothing else in the
    file changes, so the arithmetic stays Apple's.
    """
    text = _read("steel/gemm/kernels/steel_gemm_fused.h")

    flags = {
        "has_batch": False,
        # The bias always arrives as the C operand, exactly as `mx.addmm`
        # passes it. `do_axpby` stays false, so the epilogue is a plain add
        # and not the alpha/beta form.
        "use_out_source": True,
        "do_axpby": False,
        "align_M": align_m,
        "align_N": align_n,
        "align_K": align_k,
    }

    def replace_flag(match: "re.Match") -> str:
        name = match.group(1)
        value = "true" if flags[name] else "false"
        return f"constexpr constant bool {name} = {value};"

    text = _FUNCTION_CONSTANTS.sub(replace_flag, text)

    if _KERNEL_DECL.search(text) is None:
        raise RuntimeError(
            "the kernel declaration in steel_gemm_fused.h did not match; "
            "MLX changed the file, so re-check this module")
    text = _KERNEL_DECL.sub(_CALLEE_DECL, text, count=1)

    if _BATCH_BRANCH not in text:
        raise RuntimeError(
            "the batch branch in steel_gemm_fused.h did not match; "
            "MLX changed the file, so re-check this module")
    text = text.replace(_BATCH_BRANCH, _BATCH_REPLACEMENT, 1)

    if _TGP_DECLS not in text:
        raise RuntimeError(
            "the threadgroup declarations in steel_gemm_fused.h did not "
            "match; MLX changed the file, so re-check this module")
    text = text.replace(_TGP_DECLS, "", 1)

    if gelu:
        # The unary hook goes after the binary one and before the store, at
        # every one of the four exit paths. `store_result` and
        # `store_result_safe` each mark one.
        text = text.replace(
            "    // Store results to device memory\n"
            "    return mma_op.store_result(",
            "    mma_op.apply_epilogue(TransformGelu<AccumType, AccumType>{});\n"
            "    // Store results to device memory\n"
            "    return mma_op.store_result(")
        text = text.replace(
            "      // Store results to device memory\n"
            "      return mma_op.store_result_safe(",
            "      mma_op.apply_epilogue(TransformGelu<AccumType, AccumType>{});\n"
            "      // Store results to device memory\n"
            "      return mma_op.store_result_safe(")
    return text


def tgp_floats(bm: int, bn: int, bk: int, transpose_a: bool,
               transpose_b: bool, itemsize: int = 4) -> Tuple[int, int]:
    """
    Repeat the threadgroup sizes that `steel/gemm/gemm.h` calculates.

    Keep this in step with the `tgp_padding_a`, `tgp_padding_b`,
    `tgp_mem_size_a` and `tgp_mem_size_b` lines of that file.
    """
    pad = 16 // itemsize
    a = bk * (bm + pad) if transpose_a else bm * (bk + pad)
    b = bn * (bk + pad) if transpose_b else bk * (bn + pad)
    return a, b


THREADGROUP_BYTES = 32 * 1024


def fits_threadgroup(bm: int, bn: int, bk: int, transpose_a: bool,
                     transpose_b: bool, itemsize: int = 4) -> bool:
    """Return True when the two threadgroup buffers fit."""
    a, b = tgp_floats(bm, bn, bk, transpose_a, transpose_b, itemsize)
    return (a + b) * itemsize <= THREADGROUP_BYTES


# Tiles to try, in the order to prefer them. The first one that divides the
# problem wins, so the kernel always takes the aligned path. Measured at the
# shape 6 ffn_in size (M=131072, K=128, N=128), 100 repeats:
#
#     bm32 bn64  1.527 ms      bm64 bn32  1.531 ms     bm32 bn32  1.593 ms
#     bm64 bn64  1.648 ms      bm64 bn128 1.907 ms     bm128 bn64 1.947 ms
#
# against 2.300 ms for `mx.addmm` then `mlx_nn.gelu`.
_TILES = [
    (32, 64, 16, 2, 2),
    (64, 32, 16, 2, 2),
    (32, 32, 16, 2, 2),
    (64, 64, 16, 2, 2),
]


def choose_tile(m: int, n: int, k: int, transpose_a: bool = False,
                transpose_b: bool = True) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Return the first tile that divides (m, n, k) and fits the threadgroup.

    Return None when no tile divides the problem. The caller then keeps the
    MLX path. An unaligned tile still gives the right answer, because the
    kernel carries a safe path for it, but that path is slower and this
    module has no measurement of it. So refuse rather than guess.
    """
    for bm, bn, bk, wm, wn in _TILES:
        if m % bm or n % bn or k % bk:
            continue
        if not fits_threadgroup(bm, bn, bk, transpose_a, transpose_b):
            continue
        return bm, bn, bk, wm, wn
    return None


def build_header(align_m: bool, align_n: bool, align_k: bool,
                 gelu: bool) -> str:
    """Inline every steel header, then the hoisted kernel."""
    parts = [
        "#ifndef METAL_FUNC",
        "#define METAL_FUNC inline __attribute__((__always_inline__))",
        "#endif",
        "using namespace metal;",
    ]
    if gelu:
        parts.extend(_read(name) for name in _ERF_CHAIN)
    parts.extend(_read(name) for name in _CHAIN)
    parts.append("using namespace mlx::steel;")
    if gelu:
        parts.append(_GELU_TRANSFORM)
    parts.append(_hoist_kernel(align_m, align_n, align_k, gelu))
    return "\n".join(parts)


_CACHE: Dict[tuple, object] = {}


def _source(m: int, n: int, k: int, lda: int, ldb: int, ldd: int,
            ldc: int, fdc: int, bm: int, bn: int, bk: int, wm: int, wn: int,
            transpose_a: bool, transpose_b: bool) -> str:
    """
    The kernel body. It builds GEMMParams on the stack, then calls the
    hoisted kernel.

    Every shape value is a literal, because the JIT compiles one kernel per
    shape anyway. A literal costs nothing and it lets the compiler fold the
    tile counts.

    `GEMMParams` and `GEMMAddMMParams` hold `const` members, so they take a
    brace initializer. The field order is the order in `steel/gemm/params.h`.
    """
    tiles_m = -(-m // bm)
    tiles_n = -(-n // bn)
    a_floats, b_floats = tgp_floats(bm, bn, bk, transpose_a, transpose_b)
    return f"""
  threadgroup float As[{a_floats}];
  threadgroup float Bs[{b_floats}];

  GEMMParams p = {{
      {m}, {n}, {k},
      {lda}, {ldb}, {ldd},
      {tiles_n}, {tiles_m},
      0, 0, 0,
      0,
      {k // bk},
      0}};

  // `ldc = 0` broadcasts one bias row over every row of the output, and
  // `fdc = 1` steps along it by column. That is how `mx.addmm` gives a
  // (N,) bias to the same kernel.
  GEMMAddMMParams ap = {{{ldc}, {fdc}, 0, 1.0f, 1.0f}};

  gemm<float, {bm}, {bn}, {bk}, {wm}, {wn},
       {"true" if transpose_a else "false"},
       {"true" if transpose_b else "false"}, float>(
      a, b, c, out, &p, &ap,
      thread_index_in_simdgroup,
      simdgroup_index_in_threadgroup,
      threadgroup_position_in_grid,
      thread_position_in_threadgroup,
      As, Bs);
"""


def steel_addmm(
    bias: mx.array,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = True,
    gelu: bool = False,
    bm: int = 32,
    bn: int = 32,
    bk: int = 16,
    wm: int = 2,
    wn: int = 2,
) -> mx.array:
    """
    `bias + a @ b`, with GELU folded into the GEMM epilogue when `gelu`.

    This is `mx.addmm(bias, a, b)` when `gelu` is False, and
    `mlx_nn.gelu(mx.addmm(bias, a, b))` when it is True. The second form runs
    ONE kernel where MLX runs two.

    `a` is [..., M, K]. `bias` is (N,), broadcast over the rows.

    `transpose_b` names the layout of `b` as it is STORED, because the steel
    kernel takes the layout as a template argument and reads it in place:

      True   `b` is [N, K], and the result is `a @ b.T`. This is the torch
             Linear layout, so it costs no transpose.
      False  `b` is [K, N], and the result is `a @ b`. This is the fused
             QKV layout that `_build_mlx_weights()` produces.

    Pass the array in its own layout, NOT `b.T`. An MLX array carries no
    readable stride in Python, so this module cannot detect a transposed
    view. `ensure_row_contiguous` stays False, so a view would be read as
    though it were contiguous and the answer would be wrong.

    float32 only.
    """
    if a.dtype != mx.float32 or b.dtype != mx.float32:
        raise ValueError("float32 only")
    if b.ndim != 2:
        raise ValueError(f"b must be rank 2, got {b.shape}")

    k = a.shape[-1]
    if transpose_b:
        n, kb = b.shape
        ldb = k
    else:
        kb, n = b.shape
        ldb = n
    if kb != k:
        raise ValueError(f"shape mismatch: a {a.shape} against b {b.shape}")
    if bias.ndim != 1 or bias.shape[0] != n:
        raise ValueError(f"bias must be ({n},), got {bias.shape}")

    out_shape = tuple(a.shape[:-1]) + (n,)
    m = 1
    for dim in a.shape[:-1]:
        m *= dim

    transpose_a = False
    lda = k
    ldd = n

    if not fits_threadgroup(bm, bn, bk, transpose_a, transpose_b):
        raise ValueError(
            f"tile {bm}x{bn}x{bk} does not fit the 32 KiB threadgroup")

    align_m = m % bm == 0
    align_n = n % bn == 0
    align_k = k % bk == 0
    if not align_k:
        # The unaligned K path reads `gemm_k_iterations_aligned` from the
        # params, and `_source()` bakes `k // bk`. That is correct, but the
        # leftover loop is untested here, so refuse it rather than return a
        # wrong answer.
        raise ValueError(f"K={k} must be a multiple of bk={bk}")

    key = (m, n, k, lda, ldb, ldd, bm, bn, bk, wm, wn,
           transpose_a, transpose_b, gelu, align_m, align_n)
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"steel_addmm_m{m}_n{n}_k{k}"
                 f"_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}"
                 f"_tb{int(transpose_b)}_g{int(gelu)}",
            input_names=["a", "b", "c"],
            output_names=["out"],
            header=build_header(align_m, align_n, align_k, gelu),
            source=_source(m, n, k, lda, ldb, ldd, 0, 1,
                           bm, bn, bk, wm, wn, transpose_a, transpose_b),
            ensure_row_contiguous=False,
        )
        _CACHE[key] = kernel

    tiles_m = -(-m // bm)
    tiles_n = -(-n // bn)
    outputs = kernel(
        inputs=[a, b, bias],
        grid=(tiles_n * 32, tiles_m * wm, wn),
        threadgroup=(32, wm, wn),
        output_shapes=[out_shape],
        output_dtypes=[a.dtype],
    )
    return outputs[0]

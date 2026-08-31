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

import re
from typing import Dict, Optional, Tuple

import mlx.core as mx

from mlx_kernels import available as kernels_available
from mlx_kernels import read as kernels_read

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

# The anchor in `mma.h` that the LayerNorm epilogue goes in front of. It is
# the binary `apply_epilogue`, the one `mx.addmm` uses for its C operand.
_MMA_ANCHOR = """  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int ldc,
      const int fdc,
      thread const BinaryEpilogue& epilogue_op) thread {"""

# The LayerNorm epilogue of row 46. It is a new method on `BlockMMA`.
#
# MLX ships two epilogue hooks and neither one fits. The unary hook sees one
# accumulator value and no index. The binary hook sees ONE operand, indexed
# either by row (`fdc = 0`) or by column (`ldc = 0`), never both. Row 46
# needs a term that is a product of a row value and a column value:
#
#     out[i,n] = P_i * acc[i,n] - Q_i * c1[n] + c2[n]
#
# So this method reads two operands at once. It is Apple's binary
# `apply_epilogue` loop with a second pointer added, and the same indexing:
# the row is `sm + i * TM_stride`, and the column is `sn + j * TN_stride + k`.
#
#     rowstat   2 floats for each row, {rstd, rstd * mean}, from
#               `fast_layernorm.layer_norm_stats()`
#     lnc1      (N,), the column sum of the prepacked weight
#     lnc2      (N,), the LayerNorm bias through the weight, plus the
#               projection bias
#
# The caller adds the third constant, `c3`, through the ordinary C operand
# before this method runs, so it needs no pointer here.
#
# It has NO bounds check, so the caller must use an aligned tile.
# `steel_addmm()` refuses the unaligned case.
# `store_result_tgp` writes the accumulator to THREADGROUP memory, not
# device, so a second GEMM can read it as its A operand without a DRAM round
# trip. Apple's `store_result` writes to device only.
#
# **Nothing in the model uses this.** It exists for row 44, which chained
# `ffn_in` and `ffn_out` and LOST. `profiling/chain_probe.py` holds that
# kernel and its measurement. The method stays here because the patch belongs
# beside the other one, and because it is the piece a future attempt needs.
# See OPTIMIZATIONS.md row 44. This mirrors it exactly: the same
# epilogue pass, the same `sm * ld + sn` offset, and the same `MMATile::store`
# with the threadgroup overload. `LDH` is the leading dimension of the
# `hidden` tile, and it is a compile-time constant.
_TGP_STORE = """
  /* Store the accumulator to threadgroup memory. See OPTIMIZATIONS.md row 44. */
  template <short LDH>
  METAL_FUNC void store_result_tgp(threadgroup U* H) thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    H += sm * LDH + sn;
    Ctile.template store<U, WM, WN, LDH, 1>(H);
  }

"""

_LN_EPILOGUE = """
  /* LayerNorm epilogue. Added by steel_gemm.py. See OPTIMIZATIONS.md row 46. */
  METAL_FUNC void apply_layer_norm_epilogue(
      const device U* rowstat,
      const device U* lnc1,
      const device U* lnc2) thread {
    // Adjust for simdgroup and thread location, as apply_epilogue does.
    rowstat += 2 * (sm);
    lnc1 += (sn);
    lnc2 += (sn);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      // One row for the whole fragment row, so read the pair once.
      const U p = rowstat[2 * (i * TM_stride) + 0];
      const U q = rowstat[2 * (i * TM_stride) + 1];

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        thread auto& accum = Ctile.frag_at(i, j);
        const int offset = j * TN_stride;

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile)::kElemsPerFrag; k++) {
          accum[k] = p * accum[k] - q * lnc1[offset + k] + lnc2[offset + k];
        }
      }
    }
  }

"""

# The row statistics epilogue of row 47. It is a second new method on
# `BlockMMA`.
#
# Row 46 left one pass behind: `layer_norm_stats()` reads a whole activation
# to write two floats for each row. That pass runs at the memory roof, so
# only fewer bytes win it. Every activation it reads was written by a GEMM
# one stage earlier, and that GEMM holds the value in registers at the store.
# So this method takes the statistics there, and the read never happens.
#
# THE OBSTACLE, AND THE ANSWER
#
# A threadgroup owns one BN-wide tile of the row, not the whole row, so it
# cannot centre against a mean it does not have. It therefore writes the RAW
# sum and the RAW sum of squares, and a reduce kernel takes
# `var = Q/D - mean^2`.
#
# `fast_layernorm` refuses that uncentred form for a whole row, because it
# cancels when the row mean is large against the standard deviation.
# `profiling/ln_tiled_stats_probe.py` measures the drift of this model and it
# is 0.12 to 0.33, which is far below the cancellation regime. The probe
# gives the same projection error as today on shapes 6, 8 and 13.
#
# THE LANE REDUCTION
#
# One fragment row lives in four lanes. `BaseMMAFrag::get_coord` gives
# `fm = (qid & 4) + ((lane / 2) % 4)` with `qid = lane / 4`, so the four
# lanes of one row are `lane`, `lane ^ 1`, `lane ^ 8` and `lane ^ 9`. Two
# `simd_shuffle_xor` steps therefore reduce the row, with no threadgroup
# memory and no barrier. Row 44 measured that threadgroup memory is the thing
# to avoid here.
#
# The lane with `fn == 0` writes the pair. Two simdgroups cover each row of
# the tile, one for each `simd_group_id % WN`, so the partials buffer holds
# `WN * tiles_n` entries for each row. `row_stats_reduce()` sums them.
#
# The buffer is `[P][2][M]`, so the eight leader lanes of a simdgroup write
# eight adjacent floats. A `[M][P][2]` layout would scatter them.
#
# It has NO bounds check, so the caller must use an aligned tile.
_ROW_STATS_EPILOGUE = """
  /* Row statistics epilogue. Added by steel_gemm.py. See OPTIMIZATIONS.md
     row 47. */
  METAL_FUNC void write_row_stats(
      device U* part,
      const device U* carry,
      const int m_stride,
      ushort simd_lane_id) thread {
    // `carry` is the deferred residual bias of row 36. The statistics are
    // taken over `x + carry`, exactly as `layer_norm_stats(pre_bias=)` takes
    // them, and it costs one vector load for the tile.
    carry += (sn);

    // The four lanes of one fragment row are lane ^ 1 and lane ^ 8.
    const bool leader = ((simd_lane_id % 2) == 0) &&
                        (((simd_lane_id / 4) & 2) == 0);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      U total = U(0);
      U square = U(0);

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        thread auto& accum = Ctile.frag_at(i, j);
        const int offset = j * TN_stride;

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile)::kElemsPerFrag; k++) {
          const U v = accum[k] + carry[offset + k];
          total += v;
          square += v * v;
        }
      }

      total += simd_shuffle_xor(total, 1);
      total += simd_shuffle_xor(total, 8);
      square += simd_shuffle_xor(square, 1);
      square += simd_shuffle_xor(square, 8);

      if (leader) {
        const int row = sm + i * TM_stride;
        part[row] = total;
        part[m_stride + row] = square;
      }
    }
  }

"""


# The final LayerNorm epilogue of row 50. It is a third new method on
# `BlockMMA`.
#
# Row 46 folds a LayerNorm into the GEMM BELOW it. The final LayerNorm has no
# GEMM below it, so rows 37, 45, 46 and 47 all named it as the one that stays,
# and none of them measured it. It costs 1.2478 ms for each shape 6 chunk, at
# 100.2 GB/s, which is 2.8% of the shape.
#
# This method reaches it from the other side: the GEMM ABOVE it, which is the
# `ffn_out` of the last layer. That GEMM already holds the value in registers
# at the store, so the LayerNorm runs there and the activation never makes a
# second round trip.
#
# THE OBSTACLE, AND THE ANSWER
#
# Row 47 could take only the raw sums, because a threadgroup owns a BN wide
# piece of the row and cannot centre against a mean it does not have. To
# APPLY the LayerNorm the tile must own the WHOLE row. So this epilogue needs
# `bn == N`, and it needs `wn == 1` as well, so that one simdgroup owns the
# row and the two `simd_shuffle_xor` steps of row 47 reduce all of it.
#
# A full row tile is not free in general: at the shape 6 `ffn_in` size it
# costs 1.25x, because that GEMM is compute bound. It IS free on `ffn_out`,
# which takes the residual as a matrix C and is IO bound. See `_TILES`.
#
# `simd_shuffle_xor` is a butterfly, so every one of the four lanes of a
# fragment row ends with the total. No leader lane and no broadcast.
#
# The variance is the UNCENTRED form, as row 47 uses.
# `profiling/ln_tiled_stats_probe.py` measures that this model stays far
# below the cancellation regime.
#
# It reuses the pointer slots of rows 46 and 47, which are always free here:
# `ffn_out` absorbs no LayerNorm and takes no statistics.
#
#     gain     `lnc1`, the (N,) LayerNorm gain
#     lnbias   `lnc2`, the (N,) LayerNorm bias
#     carry    `rowcarry`, the (N,) deferred residual bias of row 36
#
# It has NO bounds check, so the caller must use an aligned tile.
_FINAL_LN_EPILOGUE = """
  /* Final LayerNorm epilogue. Added by steel_gemm.py. See OPTIMIZATIONS.md
     row 50. It needs BN == N and WN == 1, so that this simdgroup owns the
     whole output row. */
  METAL_FUNC void apply_final_layer_norm(
      const device U* gain,
      const device U* lnbias,
      const device U* carry,
      const U inv_d,
      const U eps) thread {
    gain += (sn);
    lnbias += (sn);
    carry += (sn);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      U total = U(0);
      U square = U(0);

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        thread auto& accum = Ctile.frag_at(i, j);
        const int offset = j * TN_stride;

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile)::kElemsPerFrag; k++) {
          const U v = accum[k] + carry[offset + k];
          total += v;
          square += v * v;
        }
      }

      // The four lanes of one fragment row are lane ^ 1 and lane ^ 8. The
      // xor shuffle is a butterfly, so all four end with the total.
      total += simd_shuffle_xor(total, 1);
      total += simd_shuffle_xor(total, 8);
      square += simd_shuffle_xor(square, 1);
      square += simd_shuffle_xor(square, 8);

      const U mean = total * inv_d;
      const U var = metal::fmax(square * inv_d - mean * mean, U(0));
      const U rstd = metal::rsqrt(var + eps);

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        thread auto& accum = Ctile.frag_at(i, j);
        const int offset = j * TN_stride;

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile)::kElemsPerFrag; k++) {
          const U v = accum[k] + carry[offset + k];
          accum[k] = (v - mean) * rstd * gain[offset + k]
              + lnbias[offset + k];
        }
      }
    }
  }

"""

def _read(rel: str) -> str:
    text = kernels_read(rel)
    text = _MLX_INCLUDE.sub("", text)
    text = _PRAGMA_ONCE.sub("", text)
    if rel == "steel/gemm/mma.h":
        if _MMA_ANCHOR not in text:
            raise RuntimeError(
                "the binary apply_epilogue in mma.h did not match; MLX "
                "changed the file, so re-check steel_gemm.py")
        text = text.replace(
            _MMA_ANCHOR,
            _TGP_STORE + _LN_EPILOGUE + _ROW_STATS_EPILOGUE
            + _FINAL_LN_EPILOGUE + _MMA_ANCHOR, 1)
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
    const device T* rowstat,
    const device T* lnc1,
    const device T* lnc2,
    device T* rowpart,
    const device T* rowcarry,
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


# The C tile offset in `steel_gemm_fused.h`, and what row 46 adds beside it.
# `rowstat` is indexed by row and the two constants by column, so they take
# the same `c_row` and `c_col` the kernel already computed.
_C_OFFSET = """  if (use_out_source) {
    C += c_row_long * addmm_params->ldc + c_col_long * addmm_params->fdc;
  }"""

_C_OFFSET_WITH_LN = _C_OFFSET + """

  // Row 46. See steel_gemm.py.
  rowstat += c_row_long * 2;
  lnc1 += c_col_long;
  lnc2 += c_col_long;"""

# Row 47. Each simdgroup owns one partial plane pair. `tid_x` is the N tile
# index, and the two simdgroups that cover one row of the tile differ in
# `simd_group_id % WN`. So the plane index is `WN * tid_x + simd_group_id % WN`
# and the buffer holds `WN * tiles_n` entries for each row.
_C_OFFSET_WITH_STATS = """

  // Row 47. See steel_gemm.py.
  rowpart += (WN * tid_x + int(simd_group_id % WN)) * 2 * params->M
      + c_row_long;
  rowcarry += c_col_long;"""

# Row 50. It reuses the `lnc1`, `lnc2` and `rowcarry` slots, which the
# `ffn_out` GEMM never uses. `c_col_long` is always 0 here, because the
# epilogue needs `bn == N` and so there is one N tile, but index it anyway.
_C_OFFSET_WITH_FINAL_LN = """

  // Row 50. See steel_gemm.py.
  lnc1 += c_col_long;
  lnc2 += c_col_long;
  rowcarry += c_col_long;"""

# The two store sites. The epilogue goes in front of each one, so it acts on
# the accumulator in registers and never on DRAM.
_STORE_SITES = (
    ("    // Store results to device memory\n"
     "    return mma_op.store_result(",
     "    "),
    ("      // Store results to device memory\n"
     "      return mma_op.store_result_safe(",
     "      "),
)


def _hoist_kernel(align_m: bool, align_n: bool, align_k: bool,
                  gelu: bool, layer_norm: bool = False,
                  row_stats: bool = False,
                  final_ln: Optional[Tuple[int, float]] = None) -> str:
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

    if layer_norm:
        # The three pointers are in the signature whatever the flag, so the
        # offsets need no compile-time guard: this block only reaches the
        # source when the epilogue is on.
        if _C_OFFSET not in text:
            raise RuntimeError(
                "the C tile offset in steel_gemm_fused.h did not match; "
                "MLX changed the file, so re-check this module")
        text = text.replace(_C_OFFSET, _C_OFFSET_WITH_LN, 1)

    if row_stats:
        if _C_OFFSET not in text:
            raise RuntimeError(
                "the C tile offset in steel_gemm_fused.h did not match; "
                "MLX changed the file, so re-check this module")
        text = text.replace(_C_OFFSET, _C_OFFSET + _C_OFFSET_WITH_STATS, 1)

    if final_ln is not None:
        if _C_OFFSET not in text:
            raise RuntimeError(
                "the C tile offset in steel_gemm_fused.h did not match; "
                "MLX changed the file, so re-check this module")
        text = text.replace(
            _C_OFFSET, _C_OFFSET + _C_OFFSET_WITH_FINAL_LN, 1)

    # Both epilogues go in front of the store, so they act on the accumulator
    # in registers. The LayerNorm goes FIRST: it finishes the projection, and
    # GELU then runs on the finished value. Inserting the LayerNorm first
    # leaves it above GELU, because each insertion goes in front of the same
    # anchor.
    for anchor, indent in _STORE_SITES:
        if layer_norm:
            text = text.replace(
                anchor,
                f"{indent}mma_op.apply_layer_norm_epilogue("
                f"rowstat, lnc1, lnc2);\n" + anchor)
        if gelu:
            text = text.replace(
                anchor,
                f"{indent}mma_op.apply_epilogue("
                f"TransformGelu<AccumType, AccumType>{{}});\n" + anchor)
        if final_ln is not None:
            width, eps = final_ln
            text = text.replace(
                anchor,
                f"{indent}mma_op.apply_final_layer_norm("
                f"lnc1, lnc2, rowcarry, "
                f"{1.0 / float(width)!r}f, {float(eps)!r}f);\n" + anchor)
        if row_stats:
            # LAST, so it sits directly above the store and reads the value
            # the kernel is about to write.
            text = text.replace(
                anchor,
                f"{indent}mma_op.write_row_stats("
                f"rowpart, rowcarry, params->M, simd_lane_id);\n" + anchor)
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


def _loader_ok(brows: int, bcols: int, tgp_size: int) -> bool:
    """
    Repeat the thread geometry that `steel/gemm/loader.h` calculates.

    `BlockLoader` takes these three as DEFAULT TEMPLATE ARGUMENTS, and every
    one of them is a truncating integer division with no guard:

        n_reads = (BCOLS * BROWS) / tgp_size
        TCOLS   = BCOLS / n_reads
        TROWS   = tgp_size / TCOLS

    Then `bi = thread_idx / TCOLS` and the load loop steps `i` from 0 to
    BROWS by TROWS. So the loader is correct only when the three divisions
    are exact and `TROWS` divides `BROWS`. Nothing in MLX checks this,
    because MLX only ever instantiates the tiles in its own dispatch table.

    Measured on a plain GEMM with no epilogue, M=1024 K=128, `bm32 bk16`,
    128 threads:

        BN  32  64 128  ->  max_abs 0.00e+00
        BN  48  96      ->  max_abs 5.2e+00, 5.2e+00   WRONG
        BN 160          ->  does not compile: TCOLS is 0

    At `BN = 96` this gives `n_reads = 12`, `TCOLS = 16 / 12 = 1` and
    `TROWS = 128`. The 128 threads then load 128 rows of a 96 row tile, so
    the loader reads 32 rows past the operand and writes past the threadgroup
    buffer. The answer is wrong and nothing reports it.

    Return True when the geometry is exact.
    """
    if brows <= 0 or bcols <= 0 or tgp_size <= 0:
        return False
    if (bcols * brows) % tgp_size:
        return False
    n_reads = (bcols * brows) // tgp_size
    if n_reads < 1 or bcols % n_reads:
        return False
    tcols = bcols // n_reads
    if tgp_size % tcols:
        return False
    trows = tgp_size // tcols
    return brows % trows == 0


def loader_geometry_ok(bm: int, bn: int, bk: int, wm: int, wn: int,
                       transpose_a: bool, transpose_b: bool) -> bool:
    """
    Return True when BOTH block loaders of this tile have exact geometry.

    A tile that fails this gives a WRONG ANSWER, not a slow one. See
    `_loader_ok()`. `choose_tile()` and `choose_final_ln_tile()` both apply
    it, and `steel_addmm()` refuses a tile that fails it.
    """
    tgp_size = wm * wn * 32
    a_rows, a_cols = (bk, bm) if transpose_a else (bm, bk)
    b_rows, b_cols = (bn, bk) if transpose_b else (bk, bn)
    return (_loader_ok(a_rows, a_cols, tgp_size)
            and _loader_ok(b_rows, b_cols, tgp_size))


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
#
# THAT ORDER HOLDS FOR `ffn_in` ONLY. `ffn_in` has a vector bias and a GELU,
# so it is compute bound and the tile shape matters. `ffn_out` and `out proj`
# take the residual as a MATRIX C, which makes them IO bound, and there the
# tile shape does not matter. Measured at the same M, K and N (131072, 128,
# 128) with a matrix C, 40 repeats:
#
#     bm32 bn64  1.8055 ms    bm32 bn128 1.8074 ms    bm64 bn128 1.8048 ms
#
# So a full row tile (bn = N) is FREE on those two GEMMs, and it is not free
# on `ffn_in`. A full row tile lets one threadgroup own a whole output row.
#
# ROW 54 RE-SWEPT THIS ORDER WITH THE ROW 46 AND ROW 47 EPILOGUES ON, and it
# holds. 129 tiles on each of the four shape 6 stages, each paired against the
# tile in use and alternated every repeat (`profiling/tile_resweep.py`). The
# best candidate is 0.997x on `qkv proj` and 1.007x on `ffn_in`, and the null
# control moves 1.5%. Do not re-sweep it again without a new epilogue.
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

    Return None as well when the MLX headers are absent. This module cannot
    build a kernel without them, so the caller keeps the MLX path and the
    run completes. See `mlx_kernels.py`.
    """
    if not kernels_available():
        return None
    for bm, bn, bk, wm, wn in _TILES:
        if m % bm or n % bn or k % bk:
            continue
        if not fits_threadgroup(bm, bn, bk, transpose_a, transpose_b):
            continue
        if not loader_geometry_ok(bm, bn, bk, wm, wn, transpose_a,
                                  transpose_b):
            # A wrong answer, not a slow one. See `_loader_ok()`.
            continue
        return bm, bn, bk, wm, wn
    return None


# Row 50. The final LayerNorm epilogue needs a FULL ROW tile: `bn == N`, so
# one threadgroup owns the whole output row, and `wn == 1`, so one simdgroup
# owns it and the two `simd_shuffle_xor` steps reduce all of it.
#
# `bm32 wm4` is the best of the six that build, at every size measured
# (`profiling/final_ln_probe.py`). `wm2` does not compile at `bn = 128`: 64
# threads cannot load that threadgroup tile.
_FINAL_LN_TILES = [(32, 16, 4), (64, 16, 4)]


def choose_final_ln_tile(
        m: int, n: int, k: int,
        transpose_b: bool = True) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Return a full row tile for the final LayerNorm epilogue, or None.

    None means the shape cannot take row 50. The usual reason is a wide
    `d_model`: `bn = N` puts N * (bk + 4) floats in the threadgroup, so
    `d_model = 1024` needs 80 KiB against the 32 KiB limit. Shape 8 therefore
    keeps the separate final LayerNorm.
    """
    if not kernels_available():
        return None
    if n % 8:
        # TN = BN / (WN * 8) must be a whole number of fragments.
        return None
    for bm, bk, wm in _FINAL_LN_TILES:
        if m % bm or k % bk or bm % (wm * 8):
            continue
        if not fits_threadgroup(bm, n, bk, False, transpose_b):
            continue
        if not loader_geometry_ok(bm, n, bk, wm, 1, False, transpose_b):
            # `bn = N` here, so N itself decides. With `bk16` and 128
            # threads only N in {8, 16, 32, 64, 128} has exact geometry, so
            # `d_model = 96` gets no tile and keeps its own final LayerNorm.
            continue
        return bm, n, bk, wm, 1
    return None


def build_header(align_m: bool, align_n: bool, align_k: bool,
                 gelu: bool, layer_norm: bool = False,
                 row_stats: bool = False,
                 final_ln: Optional[Tuple[int, float]] = None) -> str:
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
    parts.append(
        _hoist_kernel(align_m, align_n, align_k, gelu, layer_norm, row_stats,
                      final_ln))
    return "\n".join(parts)


_CACHE: Dict[tuple, object] = {}


def _source(m: int, n: int, k: int, lda: int, ldb: int, ldd: int,
            ldc: int, fdc: int, bm: int, bn: int, bk: int, wm: int, wn: int,
            transpose_a: bool, transpose_b: bool,
            layer_norm: bool = False, row_stats: bool = False,
            final_ln: bool = False) -> str:
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
    # The callee always takes the three LayerNorm pointers. When the epilogue
    # is off, nothing dereferences them and `a` stands in, so the kernel needs
    # no extra buffer.
    if layer_norm:
        ln_args = "rowstat, lnc1, lnc2"
    elif final_ln:
        # Row 50 reads the two constant slots and never `rowstat`.
        ln_args = "a, lnc1, lnc2"
    else:
        ln_args = "a, a, a"
    # `out` stands in for the partials buffer when row 47 is off. Nothing
    # dereferences it then, so the kernel needs no extra buffer.
    if row_stats:
        stat_args = "rowpart, rowcarry"
    elif final_ln:
        # Row 50 reads the carry slot and never writes partials.
        stat_args = "out, rowcarry"
    else:
        stat_args = "out, a"
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
      a, b, c, out, {ln_args}, {stat_args}, &p, &ap,
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
    rowstat: Optional[mx.array] = None,
    lnc1: Optional[mx.array] = None,
    lnc2: Optional[mx.array] = None,
    row_stats: bool = False,
    row_carry: Optional[mx.array] = None,
    final_gain: Optional[mx.array] = None,
    final_bias: Optional[mx.array] = None,
    final_eps: float = 1e-5,
):
    """
    `bias + a @ b`, with GELU folded into the GEMM epilogue when `gelu`.

    This is `mx.addmm(bias, a, b)` when `gelu` is False, and
    `mlx_nn.gelu(mx.addmm(bias, a, b))` when it is True. The second form runs
    ONE kernel where MLX runs two.

    `a` is [..., M, K]. `bias` is (N,), broadcast over the rows.

    THE MATRIX C OPERAND (row 47)

    `bias` may instead be a full [..., M, N] array. The result is then
    `bias + a @ b`, which is what `mx.addmm(x, context, w.T)` computes for
    the residual add of row 36. The steel kernel already carries this case:
    it reads C at `c_row * ldc + c_col * fdc`, so a vector passes `ldc = 0`
    and a matrix passes `ldc = N`. Nothing else changes.

    Row 47 needs this, because the two GEMMs that write the activation the
    statistics pass reads, `out proj` and `ffn_out`, both take the residual
    as a matrix C. They cannot reach a steel epilogue without it.

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

    THE LAYERNORM EPILOGUE (row 46)

    Pass `rowstat`, `lnc1` and `lnc2` together to fold the LayerNorm that
    precedes this GEMM into it. Then `a` is the RAW activation, `b` is the
    prepacked weight, and `bias` carries `c3`:

        out[i,n] = P_i * (a @ b + c3)[i,n] - Q_i * lnc1[n] + lnc2[n]

    which equals `layer_norm(a + carry) @ b + proj_bias`. Build the three
    constants with `layer_norm_constants()` and `rowstat` with
    `fast_layernorm.layer_norm_stats()`.

        rowstat  [M, 2], holding {rstd, rstd * mean} for each row
        lnc1     (N,), the column sum of the prepacked weight
        lnc2     (N,), the LayerNorm bias through the weight, plus the bias

    The epilogue has no bounds check, so this path needs an aligned tile.

    THE ROW STATISTICS EPILOGUE (row 47)

    Set `row_stats` to take the LayerNorm statistics of the OUTPUT of this
    GEMM, in the same epilogue that stores it. The call then returns the pair
    `(out, partials)` instead of `out`, and `row_stats_reduce()` turns the
    partials into the `[M, 2]` array that `rowstat` above takes.

    `row_carry` is the deferred residual bias of row 36, an (N,) vector. The
    statistics are taken over `out + row_carry`, which is what
    `fast_layernorm.layer_norm_stats(pre_bias=)` computes. Pass None when the
    block defers no bias.

    This epilogue has no bounds check either, so it needs an aligned tile.

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
    out_shape = tuple(a.shape[:-1]) + (n,)
    m = 1
    for dim in a.shape[:-1]:
        m *= dim

    # `ldc = 0` broadcasts one row of C over every output row. `ldc = n`
    # gives each output row its own row of C. See `_source()`.
    if bias.ndim == 1:
        if bias.shape[0] != n:
            raise ValueError(f"bias must be ({n},), got {bias.shape}")
        ldc = 0
    elif tuple(bias.shape) == out_shape:
        ldc = n
    else:
        raise ValueError(
            f"bias must be ({n},) or {out_shape}, got {tuple(bias.shape)}")

    transpose_a = False
    lda = k
    ldd = n

    if not fits_threadgroup(bm, bn, bk, transpose_a, transpose_b):
        raise ValueError(
            f"tile {bm}x{bn}x{bk} does not fit the 32 KiB threadgroup")

    if not loader_geometry_ok(bm, bn, bk, wm, wn, transpose_a, transpose_b):
        # This is a WRONG ANSWER, not a slow one, and MLX reports nothing.
        # See `_loader_ok()`.
        raise ValueError(
            f"tile {bm}x{bn}x{bk} wm{wm} wn{wn} gives the block loader "
            f"inexact thread geometry, so it would read past the operand")

    align_m = m % bm == 0
    align_n = n % bn == 0
    align_k = k % bk == 0
    if not align_k:
        # The unaligned K path reads `gemm_k_iterations_aligned` from the
        # params, and `_source()` bakes `k // bk`. That is correct, but the
        # leftover loop is untested here, so refuse it rather than return a
        # wrong answer.
        raise ValueError(f"K={k} must be a multiple of bk={bk}")

    ln_parts = (rowstat, lnc1, lnc2)
    layer_norm = any(part is not None for part in ln_parts)
    if layer_norm:
        if not all(part is not None for part in ln_parts):
            raise ValueError(
                "the LayerNorm epilogue needs rowstat, lnc1 and lnc2 "
                "together, or none of them")
        if not (align_m and align_n):
            # The epilogue reads `rowstat` and the two constants with no
            # bounds check, so a partial tile would read past the end.
            raise ValueError(
                f"the LayerNorm epilogue needs an aligned tile: M={m} must "
                f"divide by bm={bm} and N={n} by bn={bn}")
        if rowstat.shape != (m, 2):
            raise ValueError(
                f"rowstat must be ({m}, 2), got {rowstat.shape}")
        for name, vec in (("lnc1", lnc1), ("lnc2", lnc2)):
            if vec.ndim != 1 or vec.shape[0] != n:
                raise ValueError(f"{name} must be ({n},), got {vec.shape}")

    if row_stats:
        if not (align_m and align_n):
            # The epilogue writes `partials` with no bounds check.
            raise ValueError(
                f"the row statistics epilogue needs an aligned tile: M={m} "
                f"must divide by bm={bm} and N={n} by bn={bn}")
        if row_carry is None:
            row_carry = mx.zeros((n,), dtype=mx.float32)
        elif row_carry.ndim != 1 or row_carry.shape[0] != n:
            raise ValueError(
                f"row_carry must be ({n},), got {row_carry.shape}")

    final_ln = final_gain is not None or final_bias is not None
    if final_ln:
        if final_gain is None or final_bias is None:
            raise ValueError(
                "the final LayerNorm epilogue needs final_gain and "
                "final_bias together, or neither")
        if row_stats or layer_norm:
            raise ValueError(
                "the final LayerNorm epilogue does not share a kernel with "
                "row 46 or row 47; they use the same pointer slots")
        if bn != n or wn != 1:
            # One simdgroup must own the whole output row, or the two
            # `simd_shuffle_xor` steps reduce only part of it.
            raise ValueError(
                f"the final LayerNorm epilogue needs bn == N and wn == 1, "
                f"got bn={bn}, N={n}, wn={wn}")
        if not (align_m and align_n):
            # The epilogue reads the three vectors with no bounds check.
            raise ValueError(
                f"the final LayerNorm epilogue needs an aligned tile: M={m} "
                f"must divide by bm={bm} and N={n} by bn={bn}")
        for name, vec in (("final_gain", final_gain),
                          ("final_bias", final_bias)):
            if vec.ndim != 1 or vec.shape[0] != n:
                raise ValueError(f"{name} must be ({n},), got {vec.shape}")
        if row_carry is None:
            row_carry = mx.zeros((n,), dtype=mx.float32)
        elif row_carry.ndim != 1 or row_carry.shape[0] != n:
            raise ValueError(
                f"row_carry must be ({n},), got {row_carry.shape}")

    final_arg = (n, float(final_eps)) if final_ln else None
    key = (m, n, k, lda, ldb, ldd, ldc, bm, bn, bk, wm, wn,
           transpose_a, transpose_b, gelu, layer_norm, row_stats,
           align_m, align_n, final_arg)
    kernel = _CACHE.get(key)
    if kernel is None:
        names = ["a", "b", "c"]
        if layer_norm:
            names += ["rowstat", "lnc1", "lnc2"]
        if final_ln:
            names += ["lnc1", "lnc2"]
        if row_stats or final_ln:
            names += ["rowcarry"]
        kernel = mx.fast.metal_kernel(
            name=f"steel_addmm_m{m}_n{n}_k{k}"
                 f"_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}"
                 f"_tb{int(transpose_b)}_g{int(gelu)}_ln{int(layer_norm)}"
                 f"_mc{int(ldc != 0)}_rs{int(row_stats)}"
                 f"_fln{int(final_ln)}",
            input_names=names,
            output_names=["out", "rowpart"] if row_stats else ["out"],
            header=build_header(align_m, align_n, align_k, gelu, layer_norm,
                                row_stats, final_arg),
            source=_source(m, n, k, lda, ldb, ldd, ldc, 1,
                           bm, bn, bk, wm, wn, transpose_a, transpose_b,
                           layer_norm, row_stats, final_ln),
            ensure_row_contiguous=False,
        )
        _CACHE[key] = kernel

    tiles_m = -(-m // bm)
    tiles_n = -(-n // bn)
    operands = [a, b, bias]
    if layer_norm:
        operands += [rowstat, lnc1, lnc2]
    if final_ln:
        operands += [final_gain, final_bias]
    if row_stats or final_ln:
        operands += [row_carry]

    shapes = [out_shape]
    if row_stats:
        # `[P][2][M]`. Eight leader lanes of one simdgroup then write eight
        # adjacent floats. See `_ROW_STATS_EPILOGUE`.
        shapes.append((wn * tiles_n, 2, m))

    outputs = kernel(
        inputs=operands,
        grid=(tiles_n * 32, tiles_m * wm, wn),
        threadgroup=(32, wm, wn),
        output_shapes=shapes,
        output_dtypes=[a.dtype] * len(shapes),
    )
    return (outputs[0], outputs[1]) if row_stats else outputs[0]


# The reduce half of row 47. It sums the `P` partials of each row and writes
# the same `[M, 2]` pair that `fast_layernorm.layer_norm_stats()` writes, so
# the LayerNorm epilogue of row 46 takes it with no change.
#
# It reads `P * 2` floats for each row where `layer_norm_stats()` reads the
# whole activation: 4 MiB against 65 MiB at the shape 6 chunk.
#
# The variance is the UNCENTRED form, because a tile never sees a whole row.
# `profiling/ln_tiled_stats_probe.py` measures that this model never enters
# the cancellation regime.
_REDUCE_SOURCE = """
    constexpr uint P = {planes};
    constexpr float INV_D = {inv_d}f;
    constexpr float EPS = {eps}f;

    uint row = thread_position_in_grid.x;
    uint n_rows = {rows};
    if (row >= n_rows) {{
        return;
    }}

    // `[P][2][M]`, so neighbouring threads read neighbouring floats.
    float total = 0.0f;
    float square = 0.0f;
    for (uint p = 0; p < P; ++p) {{
        total += part[(2 * p + 0) * n_rows + row];
        square += part[(2 * p + 1) * n_rows + row];
    }}

    float mean = total * INV_D;
    float var = metal::fmax(square * INV_D - mean * mean, 0.0f);
    float rstd = metal::rsqrt(var + EPS);

    out[2 * row + 0] = rstd;
    out[2 * row + 1] = rstd * mean;
"""

_REDUCE_CACHE: Dict[tuple, object] = {}

# Threads per threadgroup for the reduce. One thread takes one row.
_REDUCE_GROUP = 256


def row_stats_reduce(partials: mx.array, width: int, eps: float) -> mx.array:
    """
    Turn the partials of `steel_addmm(row_stats=True)` into `[M, 2]`.

    The result holds `{rstd, rstd * mean}` for each row, which is what
    `fast_layernorm.layer_norm_stats()` returns and what the `rowstat`
    argument above takes.

    `width` is the row width the statistics run over, which is N of the GEMM
    that produced the partials.
    """
    if partials.ndim != 3 or partials.shape[1] != 2:
        raise ValueError(
            f"partials must be [P, 2, M], got {partials.shape}")
    planes, _, rows = partials.shape

    key = (planes, rows, width, float(eps))
    kernel = _REDUCE_CACHE.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"steel_row_stats_reduce_p{planes}_m{rows}_d{width}",
            input_names=["part"],
            output_names=["out"],
            source=_REDUCE_SOURCE.format(
                planes=planes, rows=rows,
                inv_d=repr(1.0 / float(width)), eps=float(eps)),
            ensure_row_contiguous=False,
        )
        _REDUCE_CACHE[key] = kernel

    groups = -(-rows // _REDUCE_GROUP)
    outputs = kernel(
        inputs=[partials],
        grid=(groups * _REDUCE_GROUP, 1, 1),
        threadgroup=(_REDUCE_GROUP, 1, 1),
        output_shapes=[(rows, 2)],
        output_dtypes=[partials.dtype],
    )
    return outputs[0]


def layer_norm_constants(
    gain: mx.array,
    ln_bias: mx.array,
    weight: mx.array,
    proj_bias: mx.array,
    transpose_b: bool = True,
    carry: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Build the four constants that let a GEMM absorb the LayerNorm above it.

    Return `(prepacked_weight, lnc1, lnc2, c3)`. Every one of them depends on
    the weights alone, so build them ONCE, at weight build time. That is the
    whole point of row 46: the LayerNorm gain, the LayerNorm bias and the
    normalization all become constants, and only two floats for each row stay
    at run time.

    The algebra, with `w` for the gain, `b` for the LayerNorm bias, `c` for
    the deferred residual bias of row 36, `m` and `r` for the row mean and
    rstd of `x + c`, and `B` for the GEMM weight:

        layer_norm(x + c)[i,j] = ((x[i,j] + c_j) - m_i) * r_i * w_j + b_j

        out[i,n] = sum_j layer_norm(x + c)[i,j] * B[j,n] + proj_bias[n]
                 = r_i * ((x @ Bw)[i,n] + c3[n] - m_i * lnc1[n]) + lnc2[n]

    so

        Bw[j,n] = w_j * B[j,n]
        lnc1[n] = sum_j Bw[j,n]
        lnc2[n] = sum_j b_j * B[j,n] + proj_bias[n]
        c3[n]   = sum_j c_j * Bw[j,n]

    `c3` is the C operand of the GEMM, so it reaches the accumulator through
    the machinery `mx.addmm` already uses, before the epilogue runs.

    `transpose_b` names the layout of `weight` as it is STORED, exactly as in
    `steel_addmm()`:

      True   `weight` is [N, K]. This is the torch Linear layout.
      False  `weight` is [K, N]. This is the fused QKV layout.

    The returned weight keeps the same layout as the one passed in.

    Pass `carry=None` when the block defers no bias. `c3` is then zero, and
    the C operand adds nothing.
    """
    gain = gain.astype(mx.float32)
    ln_bias = ln_bias.astype(mx.float32)

    if transpose_b:
        # [N, K]. The K axis is last, so the gain broadcasts along it.
        packed = weight * gain[None, :]
        lnc1 = mx.sum(packed, axis=1)
        lnc2 = weight @ ln_bias + proj_bias
        c3 = packed @ carry if carry is not None else mx.zeros_like(lnc1)
    else:
        # [K, N]. The K axis is first.
        packed = weight * gain[:, None]
        lnc1 = mx.sum(packed, axis=0)
        lnc2 = ln_bias @ weight + proj_bias
        c3 = carry @ packed if carry is not None else mx.zeros_like(lnc1)

    return packed, lnc1, lnc2, c3

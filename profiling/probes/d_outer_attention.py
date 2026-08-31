"""
A flash attention kernel that blocks the head dimension, so a wide head
fits the 32 KiB threadgroup.

WHY THIS EXISTS

`steel_attention.py` hoists Apple's kernel and compiles it at a head_dim
that MLX does not ship. It cannot reach `head_dim = 256`, which shape 8
uses. The block is threadgroup memory, and it is hard:

    Q_smem  = BQ * (BD + 4)                      = 32.5 KiB at BQ 32, BD 256
    KV_smem = max((BK + 4) * BD, BK * (BD + 4))  = 36.0 KiB at BK 32, BD 256

Together that is 68.5 KiB. This machine gives a threadgroup 32 KiB, and the
limit is hard: a kernel that asks for 33 KiB fails to load. Measured with
`profiling/tg_limit.py`. OPTIMIZATIONS.md rows 26 and 41 shrank the blocks
to fit, and every fitting block shape lost to the MLX fallback, because a
`BK = 8` block runs 16 K iterations with one or two warps.

HOW THIS ONE FITS

It blocks the head dimension of K and V, and it keeps Q whole.

    S = Q @ K.T   sums over D, so it accumulates over D chunks.
    O[:, d] += P @ V[:, d]   is independent for each D chunk.

So the K loop becomes:

    for each K block:
        S = 0
        for each D chunk:  load K[:, chunk],  S += Q[:, chunk] @ K[:, chunk].T
        softmax(S) -> P                    <- needs the whole row of S, and has it
        rescale O by the online factor
        for each D chunk:  load V[:, chunk],  O[:, chunk] += P @ V[:, chunk]

    Q_smem  = BQ * (BD + 4)                        = 16.25 KiB at BQ 16
    KV_smem = max((BK + 4) * BDC, BK * (BDC + 4))  =  9.00 KiB at BK 32, BDC 64

Together 25.25 KiB, and `BK` stays 32.

WHAT IT COSTS

Nothing in arithmetic and nothing in DRAM traffic. It reads each of Q, K
and V once, exactly as Apple's kernel does, and it runs the same FLOPs. It
pays two extra threadgroup barriers for each D chunk, and it holds Q whole
in threadgroup memory, which caps BQ at 16 and so caps the threadgroup at
two simdgroups.

This is NOT the `d_outer` of `metal-flash-attention`. That one blocks the D
axis of the O accumulator and spills O to device memory. This one keeps O
in registers and blocks the operands instead, so there is no spill.

WHAT IS FORKED

Apple's `steel_attention.h` is a template over `T, BQ, BK, BD, WM, WN`. The
D block is a structural change to two loops, to three block loaders and to
the threadgroup sizes, so this module does not edit Apple's text. It carries
its own kernel body, built from Apple's own `BlockLoaderT`, `MMATile` and
`tile_matmad`. The softmax, the online rescale and the causal mask are
copied from Apple's file line for line.

`upstream_moved()` hashes Apple's file against the version this fork was
read from. When MLX changes it, read the new file and check the softmax,
the online rescale and the causal mask below against it.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mlx_kernels import available as kernels_available
from mlx_kernels import read as kernels_read
from steel_attention import _CHAIN, _read

# The version of `steel_attention.h` this fork was read against. When this
# moves, read the new file and check the softmax, the causal mask and the
# online rescale below against it.
_UPSTREAM = "steel/attn/kernels/steel_attention.h"
_UPSTREAM_SHA = "b9283e574dc7fe1f58f2a55725a14fd6"

THREADGROUP_BYTES = 32 * 1024


def upstream_digest() -> str:
    """Return the MD5 of Apple's kernel file, or '' when it is absent."""
    if not kernels_available():
        return ""
    return hashlib.md5(kernels_read(_UPSTREAM).encode()).hexdigest()


def upstream_moved() -> bool:
    """
    Return True when Apple's kernel file changed since this fork was read.

    The fork copies Apple's softmax, online rescale and causal mask. When
    this returns True, read the new file against them before you trust a
    number from this module.
    """
    digest = upstream_digest()
    return bool(digest) and digest != _UPSTREAM_SHA


# The small reduction operators. Copied from `steel_attention.h`.
_OPS = """
struct DOMaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct DOSumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct DOMulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};
struct DOExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};
struct DODivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x / y; }
};
"""

# The kernel. `BDC` is the head-dimension chunk. Every other template
# parameter means what it means in Apple's file.
#
# This build handles the ALIGNED case only: `S % BQ == 0` and `S % BK == 0`.
# Shape 8 is S=128 with BQ=16 and BK=32, so both hold. `supports()` refuses
# anything else, and the caller keeps the MLX path.
_KERNEL = """
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int BDC,
    int WM,
    int WN,
    bool do_causal,
    typename AccumType = float>
METAL_FUNC void attention_d_blocked(
    const device T* Q,
    const device T* K,
    const device T* V,
    device T* O,
    const thread AttnParams* params,
    uint simd_lane_id,
    uint simd_group_id,
    uint3 tid,
    uint3 lid,
    threadgroup T* Q_smem,
    threadgroup T* KV_smem) {

  (void)lid;

  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] +
       tidl.y * params->Q_strides[1] +
       tidl.x * BQ * params->Q_strides[2];

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + kv_head_idx * params->K_strides[1];
  V += tidl.z * params->V_strides[0] + kv_head_idx * params->V_strides[1];

  O += tidl.z * params->O_strides[0] +
       tidl.y * params->O_strides[1] +
       tidl.x * BQ * params->O_strides[2];

  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BDC + padV;

  // The number of head-dimension chunks. This is the whole change.
  constexpr int TDB = BD / BDC;

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  constexpr short tgp_size = WM * WN * 32;

  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, tgp_size>;
  // K is loaded in transposed, one D chunk at a time.
  using KBlockLoader = BlockLoaderT<T, BK, BDC, 1, LDK_tgp, 0, tgp_size>;
  using VBlockLoader = BlockLoaderT<T, BK, BDC, LDV_tgp, 1, 0, tgp_size>;

  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;

  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");
  static_assert(BD % BDC == 0, "BDC must divide BD");
  static_assert(BDC % kFragSize == 0, "BDC must be a multiple of 8");

  constexpr int TQ = BQ / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;
  constexpr int TDC = BDC / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Q is loaded once, whole, and stays for the K loop.
  loader_q.load_unsafe();

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  const int k_row = params->K_strides[2];
  const int v_row = params->V_strides[2];

  for (int kb = 0; kb < kb_lim; kb++) {
    Stile.clear();

    // S = Q @ K.T, accumulated over the head-dimension chunks.
    STEEL_PRAGMA_UNROLL
    for (short c = 0; c < TDB; c++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      KBlockLoader loader_k(
          K + kb * BK * k_row + c * BDC, k_row, Ks,
          simd_group_id, simd_lane_id);
      loader_k.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TDC; dd++) {
        simdgroup_barrier(mem_flags::mem_none);

        Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
            &Qs[Qs_offset + (c * BDC + dd * kFragSize)]);
        Ktile.template load<T, 1, 1, LDK_tgp, 1>(
            &Ks[Ks_offset + dd * Ks_tile_stride]);

        simdgroup_barrier(mem_flags::mem_none);

        tile_matmad(Stile, Qtile, Ktile, Stile);
      }
    }

    // Apply scale in float32
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Do softmax
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<DOMaxOp>(new_max);
    Stile.template row_bin_op<DOExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<DOSumOp>(sum_score_tmp);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Rescale the running O by the online factor, before it takes this
    // block's contribution.
    Otile.template row_bin_op<DOMulOp>(factor);

    // O[:, chunk] += P @ V[:, chunk], one head-dimension chunk at a time.
    STEEL_PRAGMA_UNROLL
    for (short c = 0; c < TDB; c++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      VBlockLoader loader_v(
          V + kb * BK * v_row + c * BDC, v_row, Vs,
          simd_group_id, simd_lane_id);
      loader_v.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDC; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if (BDC >= 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if (BDC >= 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(0, c * TDC + id),
              Stile.frag_at(0, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(0, c * TDC + id));
        }
      }
    }
  }

  // Normalize output
  Otile.template row_bin_op<DODivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  O += (tm + sm) * params->O_strides[2] + sn;
  Otile.template store<T, 1, 1>(O, params->O_strides[2]);
}
"""


def smem_floats(bq: int, bk: int, bd: int, bdc: int,
                itemsize: int = 4) -> Tuple[int, int]:
    """Return the two threadgroup buffer sizes, in elements."""
    pad = 16 // itemsize
    q_smem = bq * (bd + pad)
    kv_smem = max((bk + pad) * bdc, bk * (bdc + pad))
    return q_smem, kv_smem


def fits_threadgroup(bq: int, bk: int, bd: int, bdc: int,
                     itemsize: int = 4) -> bool:
    q_smem, kv_smem = smem_floats(bq, bk, bd, bdc, itemsize)
    return (q_smem + kv_smem) * itemsize <= THREADGROUP_BYTES


def smem_kib(bq: int, bk: int, bd: int, bdc: int, itemsize: int = 4) -> float:
    q_smem, kv_smem = smem_floats(bq, bk, bd, bdc, itemsize)
    return (q_smem + kv_smem) * itemsize / 1024.0


def supports(seq: int, head_dim: int, bq: int, bk: int, bdc: int) -> bool:
    """
    Return True when this module can run the shape.

    The build handles the aligned case only, so `S` must divide by both
    block sizes. It also needs the MLX headers on disk.
    """
    if not kernels_available():
        return False
    if head_dim % 8 or head_dim % bdc or bdc % 8:
        return False
    if seq % bq or seq % bk:
        return False
    return fits_threadgroup(bq, bk, head_dim, bdc)


def build_header() -> str:
    parts = [
        "#ifndef METAL_FUNC",
        "#define METAL_FUNC inline __attribute__((__always_inline__))",
        "#endif",
        "using namespace metal;",
    ]
    parts.extend(_read(name) for name in _CHAIN)
    parts.append("using namespace mlx::steel;")
    parts.append(_OPS)
    parts.append(_KERNEL)
    return "\n".join(parts)


_CACHE: Dict[tuple, object] = {}


def _source(batch, heads, seq, head_dim, scale, bq, bk, bdc, wm, wn,
            causal, out_strides) -> str:
    nq = -(-seq // bq)
    nk = -(-seq // bk)
    q_smem, kv_smem = smem_floats(bq, bk, head_dim, bdc)
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

  attention_d_blocked<float, {bq}, {bk}, {head_dim}, {bdc}, {wm}, {wn},
                      {'true' if causal else 'false'}, float>(
      q, k, v, out, &p,
      thread_index_in_simdgroup,
      simdgroup_index_in_threadgroup,
      threadgroup_position_in_grid,
      thread_position_in_threadgroup,
      Q_smem, KV_smem);
"""


def d_blocked_attention(
    q: mx.array, k: mx.array, v: mx.array, scale: float,
    causal: bool = True, bq: int = 16, bk: int = 32, bdc: int = 64,
    head_last: bool = False,
) -> mx.array:
    """
    Run the D-blocked flash kernel. Same interface as `steel_attention()`.

    `wm` is not a free parameter here. `BQ` must equal `kNWarps * 8` with
    `TQ = 1`, so `wm = bq // 8` and `wn = 1`.
    """
    batch, heads, seq, head_dim = q.shape
    if q.dtype != mx.float32:
        raise ValueError("float32 only")
    if not supports(seq, head_dim, bq, bk, bdc):
        raise ValueError(
            f"unsupported: seq={seq} head_dim={head_dim} bq={bq} bk={bk} "
            f"bdc={bdc} smem={smem_kib(bq, bk, head_dim, bdc):.2f} KiB")

    wm = bq // 8
    wn = 1

    if head_last:
        out_shape = (batch, seq, heads, head_dim)
        out_strides = (seq * heads * head_dim, head_dim, heads * head_dim)
    else:
        out_shape = (batch, heads, seq, head_dim)
        out_strides = (heads * seq * head_dim, seq * head_dim, head_dim)

    key = (batch, heads, seq, head_dim, bq, bk, bdc, causal, head_last)
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"do_attn_bd{head_dim}_c{bdc}_bq{bq}_bk{bk}"
                 f"_b{batch}_h{heads}_s{seq}_hl{int(head_last)}",
            input_names=["q", "k", "v"],
            output_names=["out"],
            header=build_header(),
            source=_source(batch, heads, seq, head_dim, scale, bq, bk, bdc,
                           wm, wn, causal, out_strides),
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

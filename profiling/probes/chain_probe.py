"""
Row 44, measured and REVERTED: chain `ffn_in` and `ffn_out` into one kernel.

WHAT IT TRIED

`ffn_in` writes `hidden` to DRAM and `ffn_out` reads it straight back. At the
shape 6 chunk that round trip is 128 MiB, about 1.1 ms of a 12.0 ms layer.
Chaining the two GEMMs deletes it: `hidden` stays in threadgroup memory.

It is possible at all only because `appendix_cases.py` sets
`ffn_dim == d_model` on every shape, so there is no 4x FFN expansion and one
row block of `hidden` fits.

WHY IT LOST

One threadgroup must own `bm` rows and ALL of `ffn_dim`, so `bn1 = ffn_dim`
and there is one N tile. That is what makes the whole `hidden` row available.
It also makes the kernel threadgroup-hungry, and threadgroup memory is what
limits how many threadgroups stay resident on a core.

Best of 40 configurations is **0.969x**. The mechanism shows in a controlled
pair, at a fixed `bm = 32`:

    bk  8   24.0 KiB   best 0.996x
    bk 16   29.0 KiB   best 0.896x

Same tile, more threadgroup memory, worse time. The round trip saving is
real and the occupancy loss eats all of it.

Run it again:

    .venv/bin/python3 profiling/probes/chain_probe.py

Read OPTIMIZATIONS.md row 44 before you try this again.
"""

from __future__ import annotations

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx  # noqa: E402

from fast_layernorm import layer_norm_stats  # noqa: E402
from steel_gemm import _CHAIN, _ERF_CHAIN, _GELU_TRANSFORM  # noqa: E402
from steel_gemm import _read, kernels_available  # noqa: E402
from steel_gemm import layer_norm_constants, steel_addmm  # noqa: E402

LAYER_NORM_EPS = 1e-5
REPEATS = 50

_CACHE: dict = {}


# ---------------------------------------------------------------------------
# The kernel itself. Kept so the measurement can be repeated.
#
# `ffn_in` writes `hidden` to DRAM and `ffn_out` reads it straight back. At
# the shape 6 chunk that round trip is 128 MiB, about 1.1 ms of a 12.0 ms
# layer. Chaining deletes it: `hidden` stays in threadgroup memory.
#
# It is only possible because `appendix_cases.py` sets `ffn_dim == d_model`
# on every shape, so there is no 4x FFN expansion and one row block of
# `hidden` fits. A normal transformer with `ffn_dim = 4 * d_model` cannot do
# this.
#
# THE SHAPE OF THE KERNEL
#
# One threadgroup owns `bm` rows and ALL of `ffn_dim`, so `bn1 = ffn_dim` and
# there is one N tile. That is what makes the whole `hidden` row available to
# the second GEMM.
#
#   phase 1   hidden = gelu(layer_norm(x) @ W1 + b1)   -> threadgroup
#   phase 2   out    = residual + hidden @ W2          -> device
#
# THE THREADGROUP BUDGET, at shape 6 (bm 32, ffn 128, bk 16)
#
#   phase 1   As1 2.5 KiB + Bs1 10.0 KiB = 12.5 KiB
#   hidden    Hs 16.5 KiB, live across both phases
#   phase 2   Bs2 10.0 KiB, ALIASED onto the dead As1 and Bs1
#   peak      29.0 KiB of 32.0
#
# The aliasing is what makes it fit. Without it the total is 39.0 KiB.
_CHAIN_SOURCE = """
  // `Sc` is the scratch that phase 1 uses for As1 and Bs1, and phase 2
  // reuses for Bs2. The barrier after the `hidden` store separates them.
  threadgroup float Sc[{sc_floats}];
  threadgroup float Hs[{hs_floats}];

  threadgroup float* As1 = Sc;
  threadgroup float* Bs1 = Sc + {as1_floats};
  threadgroup float* Bs2 = Sc;

  constexpr int BM = {bm};
  constexpr int BN1 = {ffn};
  constexpr int BN2 = {n_out};
  constexpr int BK = {bk};
  constexpr int WM = {wm};
  constexpr int WN = {wn};
  constexpr short HS_LD = {hs_ld};

  using gemm1_t = mlx::steel::GEMMKernel<
      float, float, BM, BN1, BK, WM, WN, false, true, true, true, float>;
  using gemm2_t = mlx::steel::GEMMKernel<
      float, float, BM, BN2, BK, WM, WN, false, true, true, true, float>;

  // The second MMA reads its A operand from `Hs`, so its threadgroup leading
  // dimension is the `hidden` width and not `BK + padding`.
  using mma2_t = mlx::steel::BlockMMA<
      float, float, BM, BN2, BK, WM, WN, false, true, HS_LD, BK + 4, float>;

  const int tid_m = threadgroup_position_in_grid.y;
  const int c_row = tid_m * BM;

  const uint slid = thread_index_in_simdgroup;
  const uint sgid = simdgroup_index_in_threadgroup;

  // ---- phase 1 -----------------------------------------------------------
  thread typename gemm1_t::mma_t mma1(sgid, slid);
  thread typename gemm1_t::loader_a_t la(
      a + (ulong)c_row * {lda}, {lda}, As1, sgid, slid);
  thread typename gemm1_t::loader_b_t lb(w1, {ldb1}, Bs1, sgid, slid);

  for (int k = 0; k < {k_iters1}; ++k) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    la.load_unsafe();
    lb.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma1.mma(As1, Bs1);
    la.next();
    lb.next();
  }}

  // The C operand carries `c3`, the deferred residual bias through the
  // prepacked weight. `ldc = 0` broadcasts it over the rows.
  // `TransformAdd` takes (alpha, beta) and ignores both on the plain add
  // path. `mx.addmm` constructs it the same way.
  const mlx::steel::TransformAdd<float, float> add_op(1.0f, 1.0f);
  mma1.apply_epilogue(c3, 0, 1, add_op);
  mma1.apply_layer_norm_epilogue(
      rowstat + (ulong)c_row * 2, lnc1, lnc2);
  mma1.apply_epilogue(TransformGelu<float, float>{{}});
  mma1.store_result_tgp<HS_LD>(Hs);

  // `hidden` is complete, and As1 and Bs1 are dead. Phase 2 may reuse them.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- phase 2 -----------------------------------------------------------
  thread mma2_t mma2(sgid, slid);
  thread typename gemm2_t::loader_b_t lb2(w2, {ldb2}, Bs2, sgid, slid);

  for (int k = 0; k < {k_iters2}; ++k) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lb2.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // `Hs` holds the whole row, so step into it by the k block instead of
    // reloading. This is the round trip that row 44 deletes.
    mma2.mma(Hs + k * BK, Bs2);
    lb2.next();
  }}

  // The residual add, exactly as row 36 gives it to `mx.addmm`.
  mma2.apply_epilogue(
      resid + (ulong)c_row * {ldd}, {ldd}, 1, add_op);
  mma2.store_result(out + (ulong)c_row * {ldd}, {ldd});
"""


def _chain_header() -> str:
    """Inline the steel headers, plus the GELU transform the chain needs."""
    parts = [
        "#ifndef METAL_FUNC",
        "#define METAL_FUNC inline __attribute__((__always_inline__))",
        "#endif",
        "using namespace metal;",
    ]
    parts.extend(_read(name) for name in _ERF_CHAIN)
    parts.extend(_read(name) for name in _CHAIN)
    parts.append("using namespace mlx::steel;")
    parts.append(_GELU_TRANSFORM)
    return "\n".join(parts)


def chain_supported(rows: int, d_model: int, ffn: int, bm: int = 32,
                    bk: int = 16, wm: int = 2, wn: int = 2) -> bool:
    """
    Return True when the chained kernel can run this shape.

    Five conditions, and every one of them is a hard limit:

    1. The headers must be on disk.
    2. `bn1 = ffn` and `bn2 = d_model` must each be a multiple of `wn * 8`,
       or the MMA fragment count is zero.
    3. `bm` must be a multiple of `wm * 8`, and must divide the row count.
    4. `bk` must divide both `d_model` and `ffn`.
    5. The threadgroup must hold the scratch plus `hidden`.
    """
    if not kernels_available():
        return False
    if rows % bm or bm % (wm * 8):
        return False
    if ffn % (wn * 8) or d_model % (wn * 8):
        return False
    if d_model % bk or ffn % bk:
        return False
    return chain_threadgroup_kib(bm, d_model, ffn, bk) <= 32.0


def chain_threadgroup_kib(bm: int, d_model: int, ffn: int,
                          bk: int) -> float:
    """Return the peak threadgroup use of the chained kernel, in KiB."""
    pad = 4
    as1 = bm * (bk + pad)
    bs1 = ffn * (bk + pad)
    bs2 = d_model * (bk + pad)
    hs = bm * (ffn + pad)
    return (max(as1 + bs1, bs2) + hs) * 4 / 1024


def steel_ffn_chain(
    x: mx.array,
    w1: mx.array,
    c3: mx.array,
    lnc1: mx.array,
    lnc2: mx.array,
    rowstat: mx.array,
    w2: mx.array,
    residual: mx.array,
    bm: int = 32,
    bk: int = 16,
    wm: int = 2,
    wn: int = 2,
) -> mx.array:
    """
    One kernel for the whole FFN, LayerNorm included.

        out = residual + gelu(layer_norm(x) @ w1.T + b1) @ w2.T

    `w1` and `w2` are both [N, K], the torch Linear layout. `w1` is the
    prepacked weight of row 46 and `c3`, `lnc1`, `lnc2` and `rowstat` are its
    constants. See `layer_norm_constants()`.

    `hidden` never reaches DRAM. That is the whole point: see the row 44 note
    above.

    float32 only, and the caller must test `chain_supported()` first.
    """
    for name, arr in (("x", x), ("w1", w1), ("w2", w2),
                      ("residual", residual)):
        if arr.dtype != mx.float32:
            raise ValueError(f"{name} must be float32, got {arr.dtype}")

    d_model = x.shape[-1]
    ffn, k1 = w1.shape
    n_out, k2 = w2.shape
    if k1 != d_model:
        raise ValueError(f"w1 is {w1.shape}, but x has width {d_model}")
    if k2 != ffn:
        raise ValueError(f"w2 is {w2.shape}, but hidden has width {ffn}")

    rows = 1
    for dim in x.shape[:-1]:
        rows *= dim
    if not chain_supported(rows, n_out, ffn, bm, bk, wm, wn):
        raise ValueError(
            f"the chained kernel does not serve rows={rows} d_model={d_model} "
            f"ffn={ffn} at bm={bm} bk={bk}: "
            f"{chain_threadgroup_kib(bm, n_out, ffn, bk):.1f} KiB of 32")
    if rowstat.shape != (rows, 2):
        raise ValueError(f"rowstat must be ({rows}, 2), got {rowstat.shape}")

    hs_ld = ffn + 4
    key = (rows, d_model, ffn, n_out, bm, bk, wm, wn)
    kernel = _CACHE.get(key)
    if kernel is None:
        pad = 4
        as1 = bm * (bk + pad)
        bs1 = ffn * (bk + pad)
        bs2 = n_out * (bk + pad)
        kernel = mx.fast.metal_kernel(
            name=f"steel_ffn_chain_m{rows}_d{d_model}_f{ffn}"
                 f"_bm{bm}_bk{bk}_wm{wm}_wn{wn}",
            input_names=["a", "w1", "c3", "lnc1", "lnc2", "rowstat",
                         "w2", "resid"],
            output_names=["out"],
            header=_chain_header(),
            source=_CHAIN_SOURCE.format(
                sc_floats=max(as1 + bs1, bs2), hs_floats=bm * hs_ld,
                as1_floats=as1, bm=bm, ffn=ffn, n_out=n_out, bk=bk,
                wm=wm, wn=wn, hs_ld=hs_ld, lda=d_model, ldb1=d_model,
                ldb2=ffn, ldd=n_out,
                k_iters1=d_model // bk, k_iters2=ffn // bk,
            ),
            ensure_row_contiguous=False,
        )
        _CACHE[key] = kernel

    outputs = kernel(
        inputs=[x, w1, c3, lnc1, lnc2, rowstat, w2, residual],
        grid=(32, (rows // bm) * wm, wn),
        threadgroup=(32, wm, wn),
        output_shapes=[tuple(x.shape[:-1]) + (n_out,)],
        output_dtypes=[x.dtype],
    )
    return outputs[0]


def _timed(build) -> float:
    for _ in range(5):
        mx.eval(build())
    mx.synchronize()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        mx.eval(build())
        mx.synchronize()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


def main() -> None:
    """Repeat the sweep that reverted row 44."""
    mx.random.seed(0)
    rows, d_model, ffn = 131072, 128, 128     # the shape 6 chunk
    x = mx.random.normal((rows, d_model))
    resid = mx.random.normal((rows, d_model))
    gain = mx.random.normal((d_model,))
    ln_bias = mx.random.normal((d_model,))
    w1 = mx.random.normal((ffn, d_model))
    b1 = mx.random.normal((ffn,))
    w2 = mx.random.normal((d_model, ffn))
    carry = mx.random.normal((d_model,))
    mx.eval(x, resid, gain, ln_bias, w1, b1, w2, carry)
    packed, c1, c2, c3 = layer_norm_constants(
        gain, ln_bias, w1, b1, True, carry)
    mx.eval(packed, c1, c2, c3)

    def today():
        stats = layer_norm_stats(x, LAYER_NORM_EPS, pre_bias=carry)
        hidden = steel_addmm(c3, x, packed, transpose_b=True, gelu=True,
                             bm=32, bn=64, bk=16, wm=2, wn=2,
                             rowstat=stats, lnc1=c1, lnc2=c2)
        return mx.addmm(resid, hidden, w2.T)

    def chained(bm, bk, wm, wn):
        stats = layer_norm_stats(x, LAYER_NORM_EPS, pre_bias=carry)
        return steel_ffn_chain(x, packed, c3, c1, c2, stats, w2, resid,
                               bm=bm, bk=bk, wm=wm, wn=wn)

    print(f"rows={rows} d_model={d_model} ffn={ffn}, median of {REPEATS}\n")
    base = _timed(today)
    print(f"  today, 3 kernels (stats, ffn_in, ffn_out): {base:7.3f} ms\n")

    best = None
    for bm in (8, 16, 32):
        for bk in (8, 16, 32):
            for wm in (1, 2, 4):
                for wn in (1, 2, 4):
                    if not chain_supported(rows, d_model, ffn, bm, bk, wm, wn):
                        continue
                    if bm % (wm * 8):
                        continue
                    try:
                        ms = _timed(lambda: chained(bm, bk, wm, wn))
                    except Exception:
                        continue
                    ratio = base / ms
                    kib = chain_threadgroup_kib(bm, d_model, ffn, bk)
                    if best is None or ratio > best[0]:
                        best = (ratio, bm, bk, wm, wn)
                    print(f"  bm{bm:<3} bk{bk:<3} wm{wm} wn{wn}: {ms:7.3f} ms  "
                          f"{ratio:.3f}x   threadgroup {kib:5.1f} KiB")
    if best:
        print(f"\nbest {best[0]:.3f}x at bm{best[1]} bk{best[2]} "
              f"wm{best[3]} wn{best[4]}. Nothing wins, so row 44 is REVERTED.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Size the threadgroup memory of a fused attention -> out projection kernel.

This is a paper calculation, and it decides whether the kernel is worth
building. It writes no kernel and it runs no GPU work.

    .venv/bin/python3 profiling/probes/attn_out_budget.py

The fusion holds the attention output in threadgroup memory and projects it
there, so the `context` activation never reaches DRAM. One threadgroup must
therefore own `bq` query rows over the FULL `d_model`, because the out
projection mixes every head. That is the same structure as row 44, which
chained `ffn_in` and `ffn_out`, and row 44 measured what the threadgroup cost
does to the time:

    24.0 KiB -> 0.996x        29.0 KiB -> 0.896x

So the answer this script needs is one number: the peak threadgroup bytes.

The buffers, for one threadgroup:

  O_tgp     bq x (d_model + pad)      the concatenated attention output. It
                                      must live in threadgroup memory, because
                                      `BlockMMA::mma()` reads its A operand
                                      from there (row 44 checked this).
  Q_smem    bq x (head_dim + pad)     phase 1, one head at a time
  KV_smem   as `steel_attention.py`   phase 1, one head at a time
  Bs_o      bk_o x (d_model + pad)    phase 2, the out projection weight tile.
                                      `bn_o = d_model`, because the row is
                                      whole and there is one N tile.

Phase 2 does not read Q_smem or KV_smem, so `Bs_o` can alias them. Metal does
not alias two declarations, so that needs one flat buffer and manual offsets.
The report gives both the aliased peak and the plain sum.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from appendix_cases import APPENDIX_SHAPES
from steel_attention import THREADGROUP_BYTES, smem_floats

KIB = 1024.0

# Row 44 measured the same axis on the same machine, with the same tile shape
# and only `bk` changed. These two points are the whole basis of the decision.
ROW44 = ((24.0, 0.996), (29.0, 0.896))

# Shape 6 numbers that other rows measured. `CHUNK` comes from
# `CHUNK_ACTIVATION_BYTES`, and the layer time and the DRAM rate come from
# row 44 and row 42.
CHUNK = 1024
LAYER_MS = 12.0
DRAM_GBPS = 119.9


def budget(bq: int, bk: int, head_dim: int, d_model: int, bk_o: int,
           itemsize: int = 4) -> dict:
    """Return the threadgroup bytes of every buffer, and the two peaks."""
    pad = 16 // itemsize
    q_smem, kv_smem = smem_floats(bq, bk, head_dim, itemsize)

    o_tgp = bq * (d_model + pad)
    bs_o = bk_o * (d_model + pad)

    phase1 = (q_smem + kv_smem) * itemsize
    phase2 = bs_o * itemsize
    out = o_tgp * itemsize

    return {
        "O_tgp": out,
        "Q_smem": q_smem * itemsize,
        "KV_smem": kv_smem * itemsize,
        "Bs_o": phase2,
        "aliased": out + max(phase1, phase2),
        "plain": out + phase1 + phase2,
    }


def interpolate(kib: float) -> float:
    """Read a ratio off the row 44 line. It is two points, so say so."""
    (a, ra), (b, rb) = ROW44
    return ra + (rb - ra) * (kib - a) / (b - a)


def main() -> int:
    print("Fused attention -> out projection: the threadgroup budget")
    print(f"Metal gives one threadgroup {THREADGROUP_BYTES / KIB:.1f} KiB.")
    print()
    print("Today the steel attention kernel of shape 6 uses 9.00 KiB.")
    print("Row 44 measured 24.0 KiB -> 0.996x and 29.0 KiB -> 0.896x.")
    print()

    bq, bk = 32, 32          # the block row 38 measured as best on every shape
    print(f"{'shape':>5} {'d_model':>7} {'heads':>5} {'hd':>4} {'bk_o':>5} "
          f"{'O_tgp':>7} {'attn':>7} {'Bs_o':>7} {'aliased':>8} {'plain':>7} "
          f"{'fits':>5}")
    print("-" * 82)

    for shape in APPENDIX_SHAPES:
        if not shape.enabled:
            continue
        head_dim = shape.d_model // shape.num_heads
        for bk_o in (8, 16, 32):
            b = budget(bq, bk, head_dim, shape.d_model, bk_o)
            attn = b["Q_smem"] + b["KV_smem"]
            fits = "yes" if b["aliased"] <= THREADGROUP_BYTES else "NO"
            print(f"{shape.case_id:>5} {shape.d_model:>7} {shape.num_heads:>5} "
                  f"{head_dim:>4} {bk_o:>5} "
                  f"{b['O_tgp'] / KIB:>7.2f} {attn / KIB:>7.2f} "
                  f"{b['Bs_o'] / KIB:>7.2f} {b['aliased'] / KIB:>8.2f} "
                  f"{b['plain'] / KIB:>7.2f} {fits:>5}")
        print()

    # The prize. The fusion deletes the `context` activation, which the
    # attention writes and the out projection reads straight back.
    print("The prize, at the shape 6 chunk:")
    six = [s for s in APPENDIX_SHAPES if s.case_id == 6][0]
    rows = CHUNK * six.seq_len
    trip = 2 * rows * six.d_model * 4          # one write and one read
    print(f"  chunk {CHUNK} x seq {six.seq_len} = {rows} rows x "
          f"d_model {six.d_model}")
    print(f"  context is {trip / 2 / KIB / 1024:.0f} MiB, so the round trip is "
          f"{trip / KIB / 1024:.0f} MiB")
    print(f"  at {DRAM_GBPS:.0f} GB/s that is {trip / (DRAM_GBPS * 1e9) * 1e3:.3f} ms "
          f"of a {LAYER_MS:.1f} ms layer, or "
          f"{trip / (DRAM_GBPS * 1e9) * 1e3 / LAYER_MS * 100:.1f}%")
    print("  That is the SAME round trip size row 44 chased, and row 44 lost.")
    print()

    print("The decision, on shape 6, which carries 66.5% of the FLOP weight:")
    six = [s for s in APPENDIX_SHAPES if s.case_id == 6][0]
    hd = six.d_model // six.num_heads
    for bk_o in (8, 16, 32):
        b = budget(bq, bk, hd, six.d_model, bk_o)
        kib = b["aliased"] / KIB
        if kib > THREADGROUP_BYTES / KIB:
            print(f"  bk_o {bk_o:>2}: {kib:>5.2f} KiB  does not fit")
        else:
            print(f"  bk_o {bk_o:>2}: {kib:>5.2f} KiB aliased, "
                  f"{b['plain'] / KIB:>5.2f} KiB plain  -> row 44 reads "
                  f"about {interpolate(kib):.3f}x")
    print()
    print("Read that last ratio as an indication, not a prediction. Row 44")
    print("holds TWO points, and they come from a GEMM chain, not from")
    print("attention. What is not an indication: 25.50 KiB is 2.8x the 9.00")
    print("KiB the attention kernel uses today, it sits inside the band row 44")
    print("measured, and without the aliasing it is 29.62 KiB, which is row")
    print("44's own losing point of 29.0 KiB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

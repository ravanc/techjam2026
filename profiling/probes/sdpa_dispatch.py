"""
Find the head_dim range that reaches the fused SDPA kernel, and find the
point where a pad into that range starts to pay.

`mx.fast.scaled_dot_product_attention` accepts every head_dim. It dispatches
to a fused flash kernel for some of them, and to a fallback for the rest.
The fallback materializes the whole B x H x S x S score matrix. The fused
kernel does not. Peak GPU memory therefore separates the two paths exactly.

Two modes:

    path   Sweep head_dim. Classify each value by peak memory.
    pad    Time the fallback at the true head_dim against the fused kernel
           at a padded width. Report the crossover.

Run both with `.venv/bin/python3 profiling/probes/sdpa_dispatch.py --mode both`.

The pad is exact. A zero in q or k adds nothing to the dot product, a zero
in v adds nothing to the output, and `scale` stays at the true head_dim.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import mlx.core as mx

MIB = 1024 * 1024

# The head_dim values that reach the fused kernel in MLX 0.32.2. Measured by
# peak memory, over head_dim 1 to 288. The set does not change with the mask
# kind, the dtype, the sequence length or B*H. It is NOT the range 64..128.
FUSED_HEAD_DIMS = (64, 72, 80, 96, 128)


def _make(batch, heads, seq, head_dim, dtype, kv_seq=None):
    """Build q, k and v, and evaluate them before a measurement."""
    kv_seq = kv_seq or seq
    shape_q = (batch, heads, seq, head_dim)
    shape_kv = (batch, heads, kv_seq, head_dim)
    q = mx.random.normal(shape_q).astype(dtype)
    k = mx.random.normal(shape_kv).astype(dtype)
    v = mx.random.normal(shape_kv).astype(dtype)
    mx.eval(q, k, v)
    mx.synchronize()
    return q, k, v


def _build_mask(kind, batch, heads, seq):
    """Return the mask argument for one mask kind."""
    if kind == "none":
        return None
    if kind == "causal":
        return "causal"
    if kind == "bool":
        index = mx.arange(seq)
        keep = (index[:, None] >= index[None, :])
        mask = mx.broadcast_to(keep, (batch, 1, seq, seq))
        mx.eval(mask)
        mx.synchronize()
        return mask
    raise ValueError(kind)


def probe_path(batch, heads, seq, head_dim, dtype, mask_kind):
    """
    Classify one call as fused or fallback, by peak GPU memory.

    Returns the extra bytes that the call held, and the classification.
    """
    q, k, v = _make(batch, heads, seq, head_dim, dtype)
    mask = _build_mask(mask_kind, batch, heads, seq)
    scale = head_dim ** -0.5

    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    mx.eval(out)
    mx.synchronize()
    extra = mx.get_peak_memory() - base

    score_bytes = batch * heads * seq * seq * mx.zeros(1, dtype=dtype).itemsize
    del q, k, v, out, mask
    mx.clear_cache()
    return extra, score_bytes


def time_call(fn, warmup=3, repeats=20):
    """Time a call. Synchronize after every evaluation."""
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def pad_to(array, width):
    """Pad the last axis with zeros, up to `width`."""
    head_dim = array.shape[-1]
    if head_dim == width:
        return array
    zeros = mx.zeros(array.shape[:-1] + (width - head_dim,), dtype=array.dtype)
    return mx.concatenate([array, zeros], axis=-1)


def sweep_path(args):
    """
    Sweep head_dim and classify each value.

    A call is `fused` when the extra peak memory stays far under the size of
    the score matrix. It is `fallback` when the extra memory reaches that
    size.
    """
    dtypes = {"float32": mx.float32, "float16": mx.float16,
              "bfloat16": mx.bfloat16}
    widths = list(range(1, args.max_head_dim + 1))

    for name in args.dtypes:
        dtype = dtypes[name]
        for mask_kind in args.masks:
            print()
            print(f"=== dtype={name} mask={mask_kind} "
                  f"B={args.batch} H={args.heads} S={args.seq} ===")
            runs = []
            for head_dim in widths:
                extra, score_bytes = probe_path(
                    args.batch, args.heads, args.seq, head_dim,
                    dtype, mask_kind)
                fused = extra < 0.5 * score_bytes
                runs.append((head_dim, extra, fused))
            report_runs(runs, score_bytes)


def report_runs(runs, score_bytes):
    """Print the fused head_dim values as ranges, then the raw memory."""
    fused = [head_dim for head_dim, _, ok in runs if ok]
    print(f"score matrix = {score_bytes / MIB:.0f} MiB")
    print(f"fused head_dim values: {compress(fused)}")
    print(f"count fused = {len(fused)} of {len(runs)}")
    print()
    print("head_dim  extra MiB  path")
    for head_dim, extra, ok in runs:
        print(f"{head_dim:8d}  {extra / MIB:9.1f}  "
              f"{'FUSED' if ok else 'fallback'}")


def compress(values):
    """Write a sorted integer list as ranges."""
    if not values:
        return "(none)"
    parts = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = value
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def sweep_pad(args):
    """
    Time the two paths against each other, at one shape.

    Column `direct` calls SDPA at the true head_dim. Column `pad<w>` pads q,
    k and v to width `w` and calls SDPA there. The pad cost is included, so
    the number is what an attention-only change would give. A model that
    folds the pad into the QKV weight pays less than this.
    """
    for head_dim in args.head_dims:
        q, k, v = _make(args.batch, args.heads, args.seq, head_dim,
                        mx.float32)
        mask = _build_mask(args.mask, args.batch, args.heads, args.seq)
        scale = head_dim ** -0.5

        def direct():
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask)

        base = time_call(direct, repeats=args.repeats)
        row = {"direct": base}
        for width in args.pad_widths:
            if width <= head_dim:
                continue

            def padded(width=width):
                out = mx.fast.scaled_dot_product_attention(
                    pad_to(q, width), pad_to(k, width), pad_to(v, width),
                    scale=scale, mask=mask)
                return out[..., :head_dim]

            row[f"pad{width}"] = time_call(padded, repeats=args.repeats)

            # Check that the pad is exact, once per width.
            if args.check:
                a = direct()
                b = padded()
                mx.eval(a, b)
                mx.synchronize()
                diff = float(mx.max(mx.abs(a - b)))
                row[f"pad{width}_err"] = diff

        best = min((value, name) for name, value in row.items()
                   if not name.endswith("_err"))
        cells = "  ".join(
            f"{name}={value:8.3f}" for name, value in row.items()
            if not name.endswith("_err"))
        errs = "  ".join(
            f"{name}={value:.2e}" for name, value in row.items()
            if name.endswith("_err"))
        print(f"hd={head_dim:4d}  {cells}  best={best[1]} "
              f"({base / best[0]:.2f}x)  {errs}")
        del q, k, v, mask
        mx.clear_cache()


def sweep_signature(args):
    """
    Classify by a timing signature, not by memory.

    Peak memory cannot separate the two paths at a short sequence, because
    the score matrix and the output have the same size when S == head_dim.
    The two kernels react to a causal mask in opposite ways, and that
    difference holds at every sequence length:

        fused     `mask="causal"` is FASTER than `mask=None`, near 0.55x.
        fallback  `mask="causal"` is SLOWER, near 1.4x.

    The kernel skips the masked blocks in the fused path. It builds the full
    square and then masks it in the fallback path.
    """
    print("  S  head_dim   none ms  causal ms   ratio  path")
    for seq in args.seqs:
        for head_dim in args.head_dims:
            q, k, v = _make(args.batch, args.heads, seq, head_dim, mx.float32)
            scale = head_dim ** -0.5

            def call(mask):
                return lambda: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask)

            plain = time_call(call(None), repeats=args.repeats)
            causal = time_call(call("causal"), repeats=args.repeats)
            ratio = causal / plain
            path = "FUSED" if ratio < 0.9 else "fallback"
            print(f"{seq:5d}  {head_dim:8d}  {plain:8.3f}  {causal:9.3f}  "
                  f"{ratio:6.2f}  {path}")
            del q, k, v
            mx.clear_cache()
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["path", "pad", "sig", "both"],
                        default="both")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--max-head-dim", type=int, default=288)
    parser.add_argument("--dtypes", nargs="+", default=["float32"])
    parser.add_argument("--masks", nargs="+", default=["causal"])
    parser.add_argument("--mask", default="causal")
    parser.add_argument("--head-dims", nargs="+", type=int,
                        default=[8, 16, 24, 32, 40, 48, 56, 63])
    parser.add_argument("--pad-widths", nargs="+", type=int,
                        default=FUSED_HEAD_DIMS)
    parser.add_argument("--seqs", nargs="+", type=int,
                        default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.mode == "sig":
        sweep_signature(args)
        return
    if args.mode in ("path", "both"):
        sweep_path(args)
    if args.mode in ("pad", "both"):
        print()
        print(f"=== pad crossover  B={args.batch} H={args.heads} "
              f"S={args.seq} mask={args.mask}  median ms ===")
        sweep_pad(args)


if __name__ == "__main__":
    main()

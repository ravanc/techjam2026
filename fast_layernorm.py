"""
A single-pass LayerNorm kernel for a narrow row.

WHY THIS EXISTS

`mx.fast.layer_norm` loses throughput as the normalized row gets narrower.
Measured at a constant 64 MiB, in GB/s of the read plus the write:

    row width D     32    64    96   128   192   256   512  1024
    copy x * 2     108   109   108   109   108   106   108   110
    fast.rms_norm   74   106   107   107   105   107   107   109
    fast.layer_norm  5    12    21    33    87   105   107   107

At D >= 256 the MLX kernel reaches copy speed. Below 256 it degrades as
about 1/D. `mx.fast.rms_norm` holds full speed at every width, and
`mx.mean(x, axis=-1)` reaches 93 to 98 GB/s at every width, so neither the
memory system nor the reduction width is the limit.

Twelve of the fourteen appendix shapes use d_model 128 or 32. The two
LayerNorm calls of one block are 26% of the shape 6 layer time, and they do
zero matmul FLOPs. See `OPTIMIZATIONS.md` row 31.

HOW

One SIMD group of 32 lanes takes one row. Lane `l` reads elements
`l, l + 32, l + 64, ...` so the 32 lanes of one step read 32 adjacent floats.
That is a coalesced read. The lane keeps its values in registers.

The reduction then runs TWICE over the registers, not over DRAM:

    pass 1   sum, then simd_sum, gives the mean
    pass 2   sum of (v - mean)^2, then simd_sum, gives the variance

Two passes over registers cost a few instructions. The kernel is memory
bound, so they are free. This is the same arithmetic as the torch baseline,
which also centres before it squares. The one-pass `E[x^2] - mean^2` form
is cheaper still, but it cancels when the mean is large against the standard
deviation, so this module does not use it.

The kernel holds `ceil(D / 32)` floats per lane. At D = 248 that is 8. The
module refuses a width it cannot hold, and the caller then uses MLX.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import mlx.core as mx

SIMD_WIDTH = 32

# The row widths this kernel serves. At 256 and above `mx.fast.layer_norm`
# already reaches copy speed, so there is nothing to win.
MAX_WIDTH = 256

# Rows per threadgroup. 8 rows x 32 lanes = 256 threads.
ROWS_PER_GROUP = 8

_SOURCE = """
    #define PRE_BIAS {pre_bias}
    constexpr uint SIMD = {simd};
    constexpr uint D = {width};
    constexpr uint K = (D + SIMD - 1) / SIMD;
    constexpr float EPS = {eps};

    uint lane = thread_position_in_threadgroup.x;
    uint row = thread_position_in_grid.y;
    uint n_rows = x_shape[0];
    if (row >= n_rows) {{
        return;
    }}

    const device T* xr = x + (ulong)row * (ulong)D;
    device T* orow = out + (ulong)row * (ulong)D;

    // One coalesced read of the row. The values stay in registers.
    //
    // `pb` is the deferred bias. The block defers every residual bias into
    // this vector instead of adding it to the whole activation, so this
    // kernel adds it here. It is one 128 element vector load for the row,
    // and the kernel is memory bound on the row itself, so it is free.
    // See OPTIMIZATIONS.md row 36.
    float vals[K];
    float total = 0.0f;
    for (uint k = 0; k < K; ++k) {{
        uint j = lane + k * SIMD;
        float v = (j < D) ? (float)xr[j] : 0.0f;
#if PRE_BIAS
        if (j < D) {{
            v += (float)pb[j];
        }}
#endif
        vals[k] = v;
        total += v;
    }}
    float mean = simd_sum(total) / (float)D;

    // Pass 2, over the registers. A lane past the end of the row must not
    // add (0 - mean)^2, so the guard stays.
    float sq = 0.0f;
    for (uint k = 0; k < K; ++k) {{
        uint j = lane + k * SIMD;
        if (j < D) {{
            float d = vals[k] - mean;
            sq += d * d;
        }}
    }}
    float rstd = metal::rsqrt(simd_sum(sq) / (float)D + EPS);

    for (uint k = 0; k < K; ++k) {{
        uint j = lane + k * SIMD;
        if (j < D) {{
            float y = (vals[k] - mean) * rstd;
            orow[j] = (T)(y * (float)w[j] + (float)b[j]);
        }}
    }}
"""

# One compiled kernel for each (width, eps, pre_bias). Building it is not
# free, so the module keeps it. A model uses at most two entries per width:
# one with the deferred bias and one without.
_CACHE: Dict[Tuple[int, float, bool], object] = {}


def supports(width: int, dtype: mx.Dtype) -> bool:
    """
    Return True when this kernel serves the row width and the type.

    It refuses a width of 256 or more, because `mx.fast.layer_norm` already
    reaches copy speed there. It serves float32 only: the model casts to
    float32 before every LayerNorm, so no other type reaches this code.
    """
    return 0 < width < MAX_WIDTH and dtype == mx.float32


def _kernel(width: int, eps: float, pre_bias: bool):
    key = (width, eps, pre_bias)
    kernel = _CACHE.get(key)
    if kernel is None:
        names = ["x", "w", "b"] + (["pb"] if pre_bias else [])
        kernel = mx.fast.metal_kernel(
            name="techjam_layer_norm_w%d%s" % (width, "_pb" if pre_bias else ""),
            input_names=names,
            output_names=["out"],
            source=_SOURCE.format(
                simd=SIMD_WIDTH, width=width, eps=float(eps),
                pre_bias=1 if pre_bias else 0,
            ),
        )
        _CACHE[key] = kernel
    return kernel


def layer_norm(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    eps: float,
    pre_bias: Optional[mx.array] = None,
) -> mx.array:
    """
    LayerNorm over the last axis. It matches `mx.fast.layer_norm`.

    `pre_bias` is an optional `(D,)` vector that the kernel adds to the row
    BEFORE it normalizes. So `layer_norm(x, w, b, eps, pre_bias=c)` equals
    `layer_norm(x + c, w, b, eps)`, and it costs no extra pass over `x`.

    The block uses this to defer every residual bias. See OPTIMIZATIONS.md
    row 36. Adding `c` to the whole activation costs a full read and a full
    write of `x`. Adding it here costs one vector load for each row.

    The caller must test `supports()` first. This function does not fall
    back, so a wrong width raises instead of returning a wrong answer.
    """
    width = x.shape[-1]
    if not supports(width, x.dtype):
        raise ValueError(
            "fast_layernorm does not serve width %d at %s" % (width, x.dtype)
        )

    shape = x.shape
    flat = x.reshape(-1, width)
    n_rows = flat.shape[0]

    inputs = [flat, weight, bias]
    if pre_bias is not None:
        if pre_bias.shape != (width,):
            raise ValueError(
                "pre_bias must be (%d,), got %s" % (width, pre_bias.shape)
            )
        inputs.append(pre_bias)

    outputs = _kernel(width, eps, pre_bias is not None)(
        inputs=inputs,
        template=[("T", x.dtype)],
        grid=(SIMD_WIDTH, n_rows, 1),
        threadgroup=(SIMD_WIDTH, ROWS_PER_GROUP, 1),
        output_shapes=[flat.shape],
        output_dtypes=[x.dtype],
    )
    return outputs[0].reshape(shape)

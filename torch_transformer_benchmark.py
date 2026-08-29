#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every output element:
    abs(user - ref) <= atol
    OR
    abs(user - ref) <= rtol * abs(ref)

The default thresholds are atol=0.002 and rtol=0.02 (2%).
"""

from __future__ import annotations

import argparse
import copy
import math
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class BaselineSelfAttention(nn.Module):
    """Explicit multi-head self-attention implemented with native PyTorch ops."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            # Mask invalid key positions. Shape: [B, 1, 1, S].
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        # Computing softmax in fp32 provides a stable reference for fp16/bf16 tests.
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


LAYER_NORM_EPS = 1e-5


def _to_mlx(tensor: torch.Tensor) -> mx.array:
    """Convert a torch tensor into an MLX array. NumPy has no bfloat16 type."""
    if tensor.dtype == torch.bfloat16:
        return mx.array(tensor.detach().float().cpu().numpy()).astype(mx.bfloat16)
    return mx.array(tensor.detach().cpu().numpy())


def _to_torch(
    array: mx.array, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Wrap an MLX array as a torch tensor. Do not copy the bytes.

    MLX allocates every array in unified memory, so the CPU reads an MLX
    buffer with no transfer. `np.asarray` returns a view of that buffer.
    `np.array` returns a copy. The copy is pure overhead, and `forward()`
    runs it inside the timed region on every call.

    The result SHARES memory with `array`. A write to the tensor changes
    the MLX array. The harness only reads the result, so this is safe.
    The chain tensor -> ndarray -> memoryview -> array holds a reference,
    so MLX cannot free the buffer while the tensor lives.

    `.to()` copies when the dtype or the device changes. The bfloat16 path
    and a non-CPU device therefore return an independent tensor.

    The caller must evaluate `array` first. `forward()` does this.
    """
    if array.dtype == mx.bfloat16:
        array = array.astype(mx.float32)
        mx.eval(array)
    return torch.from_numpy(np.asarray(array)).to(device=device, dtype=dtype)


def _dtype_size(dtype: mx.Dtype) -> int:
    """Bytes of one element. float32 gives 4, float16 and bfloat16 give 2."""
    return 2 if dtype in (mx.float16, mx.bfloat16) else 4


@dataclass(frozen=True)
class KernelPlan:
    """
    Which MLX kernel path to use. Chosen from the shape, once per model.

    The Appendix 3.7 shapes span 7 orders of magnitude of work, from 0.13
    GFLOP (shape 2) to 2.7 PFLOP (shape 14). One kernel path is not correct
    across that range. Every threshold below comes from a measurement on
    this machine. See references/mlx-tensorops.md.
    """

    fuse_qkv: bool
    causal_block: Optional[int]
    batch_chunk: Optional[int]
    pad_head_dim: Optional[int] = None
    steel_attention: bool = False

    def describe(self) -> str:
        block = self.causal_block if self.causal_block else "full"
        chunk = self.batch_chunk if self.batch_chunk else "none"
        pad = self.pad_head_dim if self.pad_head_dim else "none"
        return (
            f"fuse_qkv={self.fuse_qkv} causal_block={block} "
            f"batch_chunk={chunk} pad_head_dim={pad} "
            f"steel={self.steel_attention}"
        )


# One activation of this size per batch chunk. 64 MiB keeps the live set of a
# chunk inside the GPU cache hierarchy and away from the 12 GiB working set
# limit of the M3 Pro. Measured at shape 6 (B=10000, S=128, D=128), one layer:
#   chunk=10000  284.1 ms  8.55 GiB peak
#   chunk= 4096  286.1 ms  6.71 GiB peak
#   chunk= 1024  271.9 ms  5.60 GiB peak   <- 64 MiB rule picks this
#   chunk=  256  271.4 ms  3.86 GiB peak
# Chunking is not slower. It is free insurance against the working set limit.
CHUNK_ACTIVATION_BYTES = 64 * 1024 * 1024

# Blocked causal attention needs enough independent (batch x head) work to pay
# for its extra kernel launches, and enough sequence to have a masked triangle
# worth skipping.
MIN_BLOCK_PARALLEL = 64
MIN_BLOCK_SEQ = 64

# `mx.fast.scaled_dot_product_attention` dispatches to its fused (flash)
# kernel ONLY for head_dim in [64, 128]. Outside that range it accepts the
# call, returns a correct answer, and silently uses a fallback that
# materializes the whole B x H x S x S score matrix.
#
# Measured by peak GPU memory at B=8, H=8, S=2048, where the score matrix
# would be 1024 MiB. `peak - base`, in MiB:
#
#   head_dim    8     16     32     48     64     72     96    128    256
#   peak MiB  1048   1068   1108   1148    128    144    192    256   1668
#   path      fall   fall   fall   fall  FUSED  FUSED  FUSED  FUSED   fall
#
# This is the cause of the 18x spread of SDPA efficiency against head_dim
# (78 GFLOP/s at 8, 270 at 32, 830 at 64, 1390 at 128, 760 at 256). It is
# not a curve. It is a cliff between two implementations.
#
# Padding head_dim up to 64 with zeros crosses onto the fused kernel. It is
# exact in exact arithmetic: zeros in q and k add nothing to the dot
# product, zeros in v add nothing to the output, the extra output columns
# are discarded, and `scale` stays at the TRUE head_dim so the softmax does
# not move. The padding is folded into the QKV weight, so the projection
# writes the wide layout directly and no extra pass over the data is needed.
#
# WHY IT WINS: not arithmetic. The padded path does MORE arithmetic. The
# fallback writes the B x H x S x S score matrix to DRAM and reads it back
# two or three times; the fused kernel keeps the score tile in registers
# and never writes it. At shape 13 that matrix is 1.000 GiB per layer
# against 96 MiB of q, k, v operands, a ratio of 10.7x. Four layers at two
# or three passes is 8 to 12 GiB, which is 57 to 86 ms at the 150 GB/s of
# this machine. The measured saving is 71.6 ms. The pad costs 8.6 ms of
# extra QKV projection. That trade is the whole optimization.
#
# The score matrix is S x S and the operands are S x head_dim, so the
# traffic removed scales with S / head_dim. That is why the gate needs a
# long sequence, and why head_dim = 8 cannot use it: reaching 64 from 8
# needs an 8x wider projection, which no sequence length repays.
#
# `OPTIMIZATIONS.md` attempt 11 padded 8 up to 32 and reverted it. 32 is
# still on the fallback path, which is why it failed. The target is 64.
#
# THE SET IS DISCRETE. It is not the range 64..128. MLX compiles one Metal
# template for each width, and it ships five that the dispatch reaches:
# `steel_attention_*_bd64_*`, `bd72`, `bd80`, `bd96` and `bd128`. A
# head_dim of 65, 100 or 127 sits inside the old range and takes the
# fallback. Measured over head_dim 1..288, by peak GPU memory. The set does
# not move with the mask kind, the dtype, the sequence length or B * H.
#
#     .venv/bin/python3 profiling/sdpa_dispatch.py --mode path --max-head-dim 288
#
# MLX also ships `bd192` and `bd256` and never calls them. That costs shape
# 8 (head_dim = 256) the fused path, and Python cannot reach around it.
# See `OPTIMIZATIONS.md` rows 20 and 21.
from steel_attention import steel_attention as steel_sdpa
from steel_attention import supports as steel_supports

SDPA_FUSED_HEAD_DIMS = (64, 72, 80, 96, 128)
SDPA_FUSED_MIN_HEAD_DIM = SDPA_FUSED_HEAD_DIMS[0]
SDPA_FUSED_MAX_HEAD_DIM = SDPA_FUSED_HEAD_DIMS[-1]


def sdpa_pad_width(head_dim: int) -> Optional[int]:
    """
    Return the width to pad `head_dim` up to, so the call reaches the fused
    kernel. Return None when the head already fits, or when it is too wide
    to reach any member of the set.

    Always take the SMALLEST member at or above `head_dim`. A wider target
    lost every row of the crossover sweep: the fused kernel costs time in
    proportion to the padded width, so extra width buys nothing. See
    `references/mlx-tensorops.md` section 3.
    """
    if head_dim in SDPA_FUSED_HEAD_DIMS:
        return None
    for width in SDPA_FUSED_HEAD_DIMS:
        if width > head_dim:
            return width
    return None


def plan_kernels(config: TransformerConfig, itemsize: int) -> KernelPlan:
    """
    Pick the kernel path for one shape.

    **Blocked causal attention.** `mx.fast.scaled_dot_product_attention` does
    not skip the masked triangle. In MLX 0.32.2 `mask="causal"` is *slower*
    than no mask at all. At B=64, H=4, S=1024, head_dim=32:

        mask=None      38.5 ms
        mask="causal"  52.7 ms
        array mask     55.1 ms

    Splitting the query into blocks and giving each block only the keys it
    can see does skip the triangle. It is bit exact: `max_abs_diff = 0.0`
    against the full call. Block size sweep, best of each row:

        B    H     S   hd | full   blk32   blk64 | best
        64   4   128   32 | 1.001  0.917   0.895 | blk64  1.12x
        64   4  1024   32 | 52.00  36.32  31.90  | blk64  1.63x
        64  16   128    8 | 3.408  2.649   2.709 | blk32  1.29x
        64   4   128    8 | 0.855  0.687   0.693 | blk32  1.24x
        64   2   128   64 | 0.194  0.310   0.281 | full   (blocking loses)
        64   1   128  128 | 0.208  0.318   0.303 | full   (blocking loses)
        64   4   128  256 | 2.549  3.726   3.154 | full   (blocking loses)
        1    4   128   32 | 0.050  0.129   0.074 | full   (blocking loses)
        4    4   128   32 | 0.088  0.134   0.093 | full   (blocking loses)
        16   4   128   32 | 0.267  0.257   0.247 | blk64  1.08x

    Three conditions decide it, and the three rules below match all ten rows:

    1. A wide head is already efficient, so blocking only adds launches. The
       fused kernel reaches 1390 GFLOP/s at head_dim=128 but only 270 at
       head_dim=32 and 78 at head_dim=8.
    2. A short sequence has no triangle worth skipping.
    3. A small batch cannot fill the GPU, so launch cost dominates.

    **Fused QKV.** One [D, 3D] matmul in place of three [D, D] matmuls. It
    removes two kernel launches per layer. This matters only where the
    launch is a large share of the call, which is the small-batch shapes:

        B=1,  S=128, D=128 | 0.0439 -> 0.0220 ms  1.99x
        B=64, S=128, D=128 | 0.6202 -> 0.5524 ms  1.12x
        B=64, S=128, D=1024| 14.30  -> 14.27  ms  1.00x

    It never loses outside measurement noise, so it is always on.

    **Batch chunking.** See `CHUNK_ACTIVATION_BYTES`.
    """
    head_dim = config.d_model // config.num_heads
    parallel = config.batch_size * config.num_heads

    # Cross onto the fused SDPA kernel when the head is too narrow for it.
    # See SDPA_FUSED_MIN_HEAD_DIM. Two conditions gate it, and a full sweep
    # measured both. A blanket `head_dim < 64` rule LOSES: it gave 0.983x
    # FLOP-weighted, because it pays the pad on shapes that cannot use it.
    #
    # 1. The pad factor `64 / head_dim` must be at most 2. The pad widens
    #    the QKV projection by that factor, and the projection is 6*D*D of
    #    the 12*D*D + 4*S*D per-token cost. At head_dim=8 the factor is 8
    #    and no attention rate recovers it: shape 11 measured 0.756x and
    #    shape 7 measured 0.894x.
    # 2. Attention must dominate the layer, which needs a long sequence
    #    against the model width. Attention is 4*S*D per token against
    #    6*D*D for the projection, so the ratio is 2*S/(3*D). Shapes with
    #    S == D measured 0.93x to 1.11x, which is inside the noise floor.
    #
    # The threshold sits at S >= 4*D, between shape 1 (S = D, no gain) and
    # shape 13 (S = 8*D, 1.65x). **It rests on one measured point.** Sweep
    # S at fixed D before you move it. See OPTIMIZATIONS.md row 17.
    # `steel_attention.py` compiles MLX's own fused kernel at the TRUE
    # head_dim, so it needs no pad and no blocking. Prefer it wherever it
    # runs: it removes the S x S score matrix without widening any matmul.
    # It supports a string causal mask only, so a padded batch keeps the
    # old path. `_attention()` applies that rule, because the shape alone
    # does not tell this function if the batch has padding. See
    # OPTIMIZATIONS.md rows 25 and 27.
    use_steel = (
        config.causal
        and head_dim not in SDPA_FUSED_HEAD_DIMS
        and steel_supports(head_dim)
    )

    pad_head_dim: Optional[int] = None
    target = sdpa_pad_width(head_dim)
    if (
        not use_steel
        and target is not None
        and target <= 2 * head_dim
        and config.seq_len >= 4 * config.d_model
    ):
        pad_head_dim = target
    effective_head_dim = pad_head_dim or head_dim

    causal_block: Optional[int] = None
    if (
        config.causal
        and not use_steel
        and effective_head_dim < 64
        and config.seq_len > MIN_BLOCK_SEQ
        and parallel >= MIN_BLOCK_PARALLEL
    ):
        causal_block = 32 if effective_head_dim <= 16 else 64

    per_sample = config.seq_len * config.d_model * itemsize
    chunk = max(1, CHUNK_ACTIVATION_BYTES // max(1, per_sample))
    batch_chunk = chunk if chunk < config.batch_size else None

    return KernelPlan(
        fuse_qkv=True,
        causal_block=causal_block,
        batch_chunk=batch_chunk,
        pad_head_dim=pad_head_dim,
        steel_attention=use_steel,
    )


def _attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    mask,
    block: Optional[int],
    steel: bool = False,
) -> mx.array:
    """
    Causal attention, with or without query blocking.

    Without a block size this is one fused call. With a block size the query
    splits into blocks of `block` rows, and block `i` receives only the keys
    up to its own last row. The masked triangle is then never calculated.

    MLX aligns a `"causal"` mask to the *end* of the key sequence, so a query
    block of length `stop - start` against keys `[0, stop)` gets exactly the
    rows it needs. Verified bit exact against the unblocked call.
    """
    if steel and isinstance(mask, str):
        # MLX's own fused kernel, compiled at a head_dim that MLX does not
        # ship. It handles the causal mask itself, so it never blocks. See
        # `steel_attention.py`.
        #
        # The kernel takes a string mask only. A padded batch gives an array
        # mask, and this test sends that batch to the SDPA call below. Do not
        # remove the test: `plan_kernels()` sets `steel` from the shape, and
        # the shape does not tell it if the batch has padding.
        return steel_sdpa(q, k, v, scale, causal=mask == "causal")

    if block is None:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    seq_len = q.shape[2]
    parts = []
    for start in range(0, seq_len, block):
        stop = min(start + block, seq_len)
        part_mask = mask if isinstance(mask, str) or mask is None else (
            mask[..., start:stop, :stop]
        )
        parts.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, start:stop],
                k[:, :, :stop],
                v[:, :, :stop],
                scale=scale,
                mask=part_mask,
            )
        )
    return mx.concatenate(parts, axis=2)


def _mlx_transformer(
    x: mx.array,
    valid_mask: mx.array,
    layers: List[Dict[str, mx.array]],
    final_weight: mx.array,
    final_bias: mx.array,
    num_heads: int,
    causal: bool,
    compute_dtype: mx.Dtype,
    padded: bool,
    plan: KernelPlan,
) -> mx.array:
    """
    The baseline forward pass, written with MLX operations. One batch chunk.

    The LayerNorm runs in float32 and then returns to the model type. The
    torch baseline accumulates its LayerNorm in float32 for every input
    type, so this agrees with it.

    Attention uses the fused MLX kernel. I also tested an explicit softmax
    in float32, to copy line 111 of the baseline. It gave no gain: float16
    went from 49 to 50 failures, and bfloat16 from 172424 to 173161. The
    fused kernel is simpler, so I kept it.

    The `padded` flag selects the mask form. An array mask is necessary only
    when the batch has padding.

    `plan` selects the kernel path for this shape. See `plan_kernels()`.
    """
    half = compute_dtype != mx.float32

    def norm(value, weight, bias):
        result = mx.fast.layer_norm(
            value.astype(mx.float32), weight, bias, LAYER_NORM_EPS
        )
        return result.astype(compute_dtype) if half else result

    batch, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    # The SDPA operand width. It is wider than head_dim when the plan pads
    # the head to reach the fused kernel. `scale` stays at the TRUE head_dim,
    # because the padded lanes are zero and must not enter the softmax scale.
    width = plan.pad_head_dim or head_dim
    scale = head_dim**-0.5

    # The mask form. True keeps the key position.
    if padded:
        keep = valid_mask[:, None, None, :]
        if causal:
            index = mx.arange(seq_len)
            keep = mx.logical_and(keep, (index[:, None] >= index[None, :]))
        mask = mx.broadcast_to(keep, (batch, 1, seq_len, seq_len))
    else:
        mask = "causal" if causal else None

    # The token mask. It clears the padded positions. It is necessary only
    # when the batch has padding, and `padded` is a compile-time flag, so the
    # unpadded graph holds no mask operation at all.
    valid_tokens = valid_mask[..., None] if padded else None

    def heads(projection: mx.array, last: int = width) -> mx.array:
        return projection.reshape(
            batch, seq_len, num_heads, last
        ).transpose(0, 2, 1, 3)

    # Every projection uses `mx.addmm`, not `h @ w + b`. `mx.addmm` gives the
    # bias to the matmul as its C operand, so the GPU adds it inside the
    # matmul kernel. `h @ w + b` starts a second kernel, which reads and
    # writes the whole output again. `mx.compile` does NOT fuse that add.
    # Measured on the qkv projection alone: 6.768 ms to 3.749 ms at the
    # shape 6 dimensions. See OPTIMIZATIONS.md row 29.
    for layer in layers:
        h = norm(x, layer["n1w"], layer["n1b"])

        if plan.fuse_qkv:
            fused = mx.addmm(layer["qkvb"], h, layer["qkvw"])
            q, k, v = (heads(part) for part in mx.split(fused, 3, axis=-1))
        else:
            # The unfused path never pads, so it uses the true head width.
            q = heads(mx.addmm(layer["qb"], h, layer["qw"].T), head_dim)
            k = heads(mx.addmm(layer["kb"], h, layer["kw"].T), head_dim)
            v = heads(mx.addmm(layer["vb"], h, layer["vw"].T), head_dim)

        # An explicit float32 softmax gave no accuracy gain here. Measured.
        context = _attention(
            q, k, v, scale, mask, plan.causal_block, plan.steel_attention
        )
        if context.shape[-1] != head_dim:
            # Drop the zero lanes that the padded projection produced.
            context = context[..., :head_dim]
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        attention = mx.addmm(layer["ob"], context, layer["ow"].T)
        # The baseline clears the padded rows of the attention output here.
        # This code does not, and the output stays bit exact. Attention is
        # the only operation that mixes the positions, and it runs above this
        # line. Every operation from here to the end of the block acts on one
        # position at a time, so a value at a padded position never reaches a
        # valid position. The mask at the end of the block removes it. A NaN
        # from a fully masked query row goes the same way, because `mx.where`
        # selects a value and does not calculate one.
        x = x + attention

        h = norm(x, layer["n2w"], layer["n2b"])
        h = mlx_nn.gelu(mx.addmm(layer["fib"], h, layer["fiw"].T))
        x = x + mx.addmm(layer["fob"], h, layer["fow"].T)
        if padded:
            x = mx.where(valid_tokens, x, 0)

    x = norm(x, final_weight, final_bias)
    # The final LayerNorm returns the bias at a zeroed position, not zero,
    # so this mask stays.
    return mx.where(valid_tokens, x, 0) if padded else x


class UserOptimizedTransformer(BaselineTransformer):
    """
    An MLX implementation behind the torch interface of the harness.

    The class keeps the torch parameters. Therefore load_state_dict(), .to()
    and .eval() operate with no change. forward() converts the input to MLX,
    calculates the result with MLX, and converts the result back to torch.

    The two conversions are inside the timed region. The input conversion
    copies, because MLX must own its memory. The output conversion does
    not: `_to_torch()` returns a view of the MLX buffer. See
    OPTIMIZATIONS.md row 23 for what the boundary costs.

    Accuracy at the harness defaults, atol=0.002 and rtol=0.02:

      float32  : PASS. max_abs=2.98e-06, which is 670 times inside atol.
      float16  : 49 of 1572864 elements fail (0.003%).
      bfloat16 : 172424 of 1572864 elements fail (11%).

    The two half types cannot pass, and no implementation can make them
    pass. One bfloat16 step at magnitude 1.0 is 0.0078. That is 4 times
    atol=0.002. Therefore the atol test cannot absorb even one different
    rounding, and the rtol test fails wherever the reference is near zero.

    Measured proof: torch.nn.functional.scaled_dot_product_attention, which
    line 192 of this file gives as an example optimization, fails 34687 of
    524288 elements in bfloat16 against this same baseline. MLX is not the
    cause. The thresholds are float32 thresholds.
    """

    use_mlx_compile: bool = True

    # Set this to a `KernelPlan` to override `plan_kernels()`. It is for
    # tuning: it lets a benchmark compare two paths on one shape.
    plan_override: Optional["KernelPlan"] = None

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self._mlx_layers: Optional[List[Dict[str, mx.array]]] = None
        self._mlx_final: Optional[Tuple[mx.array, mx.array]] = None
        self._mlx_call = None
        self.plan: Optional[KernelPlan] = None

    def _build_mlx_weights(self) -> None:
        """
        Copy the torch parameters into MLX arrays.

        The harness copies the weights and then moves the model. Both events
        occur after __init__. Therefore build this copy at the first call.
        """
        params = dict(self.named_parameters())
        names = {
            "n1w": "norm1.weight", "n1b": "norm1.bias",
            "qw": "attention.q_proj.weight", "qb": "attention.q_proj.bias",
            "kw": "attention.k_proj.weight", "kb": "attention.k_proj.bias",
            "vw": "attention.v_proj.weight", "vb": "attention.v_proj.bias",
            "ow": "attention.out_proj.weight", "ob": "attention.out_proj.bias",
            "n2w": "norm2.weight", "n2b": "norm2.bias",
            "fiw": "ffn_in.weight", "fib": "ffn_in.bias",
            "fow": "ffn_out.weight", "fob": "ffn_out.bias",
        }
        # The LayerNorm weights stay in float32, because the baseline
        # accumulates the LayerNorm in float32 for every input type.
        norm_keys = {"n1w", "n1b", "n2w", "n2b"}
        self._mlx_layers = [
            {
                key: (
                    _to_mlx(params[f"layers.{index}.{suffix}"]).astype(mx.float32)
                    if key in norm_keys
                    else _to_mlx(params[f"layers.{index}.{suffix}"])
                )
                for key, suffix in names.items()
            }
            for index in range(self.config.num_layers)
        ]
        self._mlx_final = (
            _to_mlx(params["final_norm.weight"]).astype(mx.float32),
            _to_mlx(params["final_norm.bias"]).astype(mx.float32),
        )

        num_heads = self.config.num_heads
        causal = self.config.causal
        compute_dtype = _to_mlx(params["final_norm.weight"]).dtype

        # The plan needs the run dtype, which is known only now. Set
        # `plan_override` before the first call to test another path.
        self.plan = self.plan_override or plan_kernels(
            self.config, _dtype_size(compute_dtype)
        )

        # One [D, 3D] weight in place of three [D, D] weights. The order is
        # q, k, v, so `mx.split(..., 3)` returns them in that order.
        #
        # When the plan pads the head, each head keeps its own columns and
        # gains `pad - head_dim` zero columns, giving [D, 3*H*pad]. The
        # projection then writes the wide head layout directly, so the pad
        # costs no separate pass over the activation. The zero columns make
        # q and k contribute nothing to the dot product and v contribute
        # nothing to the output, so the result is unchanged.
        pad = self.plan.pad_head_dim
        head_dim = self.config.d_model // num_heads
        for layer in self._mlx_layers:
            weight = mx.concatenate(
                [layer["qw"], layer["kw"], layer["vw"]], axis=0
            ).T
            bias = mx.concatenate(
                [layer["qb"], layer["kb"], layer["vb"]], axis=0
            )
            if pad is not None:
                d_model = self.config.d_model
                weight = mx.pad(
                    weight.reshape(d_model, 3, num_heads, head_dim),
                    [(0, 0), (0, 0), (0, 0), (0, pad - head_dim)],
                ).reshape(d_model, 3 * num_heads * pad)
                bias = mx.pad(
                    bias.reshape(3, num_heads, head_dim),
                    [(0, 0), (0, 0), (0, pad - head_dim)],
                ).reshape(3 * num_heads * pad)
            layer["qkvw"] = weight
            layer["qkvb"] = bias

        # One variant for each mask form. `padded` changes the graph, so it
        # cannot be a traced argument. Two variants compile at most.
        plan = self.plan

        def make_call(padded: bool):
            def call(x, valid_mask, layers, final_weight, final_bias):
                return _mlx_transformer(
                    x, valid_mask, layers, final_weight, final_bias,
                    num_heads, causal, compute_dtype, padded, plan,
                )
            return mx.compile(call) if self.use_mlx_compile else call

        self._mlx_call = {padded: make_call(padded) for padded in (False, True)}

        mx.eval([w for layer in self._mlx_layers for w in layer.values()])
        mx.eval(self._mlx_final)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ====================== your codes here ======================
        if self._mlx_layers is None:
            self._build_mlx_weights()

        if valid_token_mask is None:
            valid_token_mask = torch.ones(
                x.shape[0], x.shape[1], device=x.device, dtype=torch.bool
            )

        mlx_x = _to_mlx(x)
        mlx_mask = _to_mlx(valid_token_mask)

        padded = not bool(valid_token_mask.all())
        call = self._mlx_call[padded]

        chunk = self.plan.batch_chunk
        if chunk is None:
            output = call(mlx_x, mlx_mask, self._mlx_layers, *self._mlx_final)
        else:
            # A large batch does not fit the 12 GiB GPU working set in one
            # piece. Run the whole depth for one chunk, then the next. Only
            # one chunk of intermediates is live at a time.
            batch = mlx_x.shape[0]
            parts = []
            for start in range(0, batch, chunk):
                stop = min(start + chunk, batch)
                part = call(
                    mlx_x[start:stop], mlx_mask[start:stop],
                    self._mlx_layers, *self._mlx_final,
                )
                mx.eval(part)
                parts.append(part)
            output = mx.concatenate(parts, axis=0)
        mx.eval(output)

        return _to_torch(output, dtype=x.dtype, device=x.device)
        # ============================================================


def copy_model_weights(
    baseline: nn.Module, optimized: nn.Module, strict: bool = True
) -> None:
    """Copy identical weights into both implementations for a fair comparison."""
    state_dict = copy.deepcopy(baseline.state_dict())
    incompatible = optimized.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[warning] missing optimized keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warning] unexpected optimized keys: {incompatible.unexpected_keys}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )
    positions = torch.arange(config.seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask


@dataclass
class AccuracyResult:
    passed: bool
    total_elements: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    mean_abs_error: float
    failed_feature_dims: List[int]
    worst_index: Tuple[int, ...]
    reference_at_worst: float
    optimized_at_worst: float


def compare_outputs(
    reference: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float,
    atol: float,
) -> AccuracyResult:
    if reference.shape != optimized.shape:
        raise AssertionError(
            f"shape mismatch: baseline={tuple(reference.shape)}, "
            f"optimized={tuple(optimized.shape)}"
        )
    if reference.dtype != optimized.dtype:
        print(
            f"[warning] dtype mismatch: baseline={reference.dtype}, "
            f"optimized={optimized.dtype}"
        )

    ref = reference.detach().float()
    opt = optimized.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()

    # Exact interpretation of the requested OR condition. torch.isclose uses
    # atol + rtol * abs(ref), which is slightly more permissive and is not used.
    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed_mask = ~passed_mask
    failed_elements = int(failed_mask.sum().item())
    total_elements = reference.numel()

    flat_worst = int(abs_error.reshape(-1).argmax().item())
    worst_index_list = []
    remaining = flat_worst
    for size in reversed(reference.shape):
        worst_index_list.append(remaining % size)
        remaining //= size
    worst_index = tuple(reversed(worst_index_list))

    denominator = ref.abs().clamp_min(1e-12)
    relative_error = abs_error / denominator

    # Summarize failures by the last/output-feature dimension.
    if reference.ndim == 0:
        failed_feature_dims = [0] if failed_elements else []
    elif reference.ndim == 1:
        failed_feature_dims = torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
    else:
        reduce_dims = tuple(range(reference.ndim - 1))
        failed_by_feature = failed_mask.any(dim=reduce_dims)
        failed_feature_dims = (
            torch.nonzero(failed_by_feature, as_tuple=False).flatten().tolist()
        )

    return AccuracyResult(
        passed=failed_elements == 0,
        total_elements=total_elements,
        failed_elements=failed_elements,
        max_abs_error=float(abs_error.max().item()),
        max_relative_error=float(relative_error.max().item()),
        mean_abs_error=float(abs_error.mean().item()),
        failed_feature_dims=failed_feature_dims,
        worst_index=worst_index,
        reference_at_worst=float(ref[worst_index].item()),
        optimized_at_worst=float(opt[worst_index].item()),
    )


def run_accuracy_tests(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
) -> bool:
    print("\n=== Accuracy check ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")

    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=padding_ratio,
                input_scale=input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)

            all_passed &= result.passed
            global_max_abs = max(global_max_abs, result.max_abs_error)
            global_max_rel = max(global_max_rel, result.max_relative_error)
            total_failed += result.failed_elements
            total_elements += result.total_elements

            status = "PASS" if result.passed else "FAIL"
            print(
                f"trial {trial + 1:02d}/{trials}: {status} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"max_rel={result.max_relative_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

            if not result.passed:
                preview = result.failed_feature_dims[:16]
                suffix = "..." if len(result.failed_feature_dims) > len(preview) else ""
                print(
                    f"  worst_index={result.worst_index}, "
                    f"baseline={result.reference_at_worst:.8g}, "
                    f"optimized={result.optimized_at_worst:.8g}"
                )
                print(f"  failed output feature dims={preview}{suffix}")

    print(
        f"summary: {'PASS' if all_passed else 'FAIL'} | "
        f"max_abs={global_max_abs:.6g} | max_rel={global_max_rel:.6g} | "
        f"failed={total_failed}/{total_elements}"
    )
    return all_passed


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class TimingResult:
    samples_ms: List[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> List[float]:
    samples_ms: List[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

            torch.cuda.synchronize(device)
            for index in range(iterations):
                starts[index].record()
                model(x, valid_mask)
                ends[index].record()
            torch.cuda.synchronize(device)

            samples_ms.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
        else:
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, valid_mask)
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

    return samples_ms


def benchmark_models(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
) -> None:
    print("\n=== Performance benchmark ===")
    print("timing excludes random-data generation and uses a fixed input")
    if device.type == "cuda":
        print("CUDA latency is measured with torch.cuda.Event on the current stream")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    # Warm up both models before collecting any timing data.
    warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
        else:
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens_per_call = config.batch_size * config.seq_len
    baseline_tokens_per_second = tokens_per_call * 1000.0 / baseline_result.median_ms
    optimized_tokens_per_second = tokens_per_call * 1000.0 / optimized_result.median_ms

    print(
        f"baseline : median={baseline_result.median_ms:.4f} ms | "
        f"mean={baseline_result.mean_ms:.4f} ms | "
        f"p90={baseline_result.p90_ms:.4f} ms | "
        f"min={baseline_result.min_ms:.4f} ms | "
        f"throughput={baseline_tokens_per_second:.2f} token/s"
    )
    print(
        f"optimized: median={optimized_result.median_ms:.4f} ms | "
        f"mean={optimized_result.mean_ms:.4f} ms | "
        f"p90={optimized_result.p90_ms:.4f} ms | "
        f"min={optimized_result.min_ms:.4f} ms | "
        f"throughput={optimized_tokens_per_second:.2f} token/s"
    )
    print(f"speedup  : {speedup:.3f}x based on median latency")


def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and optimized PyTorch Transformer"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable TF32 on CUDA for both implementations",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()
    validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    # Compile only after model construction, weight copy, device transfer, and eval().
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    optimized = maybe_compile(optimized, args.compile_user, args.compile_mode)

    print("=== Configuration ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    accuracy_passed = run_accuracy_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol,
        atol=args.atol,
    )

    if not accuracy_passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        print("Use --benchmark-on-failure to benchmark an incorrect implementation anyway.")
        return 2

    benchmark_models(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
    )
    return 0 if accuracy_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

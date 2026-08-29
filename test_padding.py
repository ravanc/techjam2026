#!/usr/bin/env python3
"""
Correctness test for the padded batch. It is not a timing test.

Every sweep in this project runs at `--padding-ratio 0.0`, so the padded
graph of `UserOptimizedTransformer` has no coverage there. This file gives
it coverage. See OPTIMIZATIONS.md rows 27 and 28.

Two tests run:

1. **Accuracy.** The optimized model must agree with `BaselineTransformer`
   on a padded batch, at the thresholds the harness uses. This test caught
   the row 27 bug: `plan_kernels()` selects the steel kernel from the shape,
   the shape does not say if the batch has padding, and the kernel then
   dropped the token mask and the causal mask.
2. **Bit exactness of the mask removal.** Row 28 removes the mask on the
   attention output, and it removes all three masks from the unpadded graph.
   This test rebuilds the full mask discipline from the source of
   `_mlx_transformer()` and compares the two outputs with `torch.equal`.

Run it:

    .venv/bin/python3 test_padding.py
"""

from __future__ import annotations

import argparse
import inspect
import textwrap
from typing import Dict, List, Tuple

import torch

import torch_transformer_benchmark as tb
from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    compare_outputs,
    copy_model_weights,
)

# batch, seq_len, d_model, num_heads, ffn_dim, num_layers. The head widths
# are 16, 64 and 8, which cover the steel path, the fused SDPA path and the
# narrowest head the steel kernel takes.
SHAPES: List[Tuple[int, int, int, int, int, int]] = [
    (4, 64, 128, 8, 512, 4),
    (4, 64, 128, 2, 512, 4),
    (4, 128, 128, 16, 512, 3),
]


def make_masks(batch: int, seq_len: int) -> Dict[str, torch.Tensor]:
    """Three mask patterns. Each one holds a different risk."""
    full = torch.ones(batch, seq_len, dtype=torch.bool)

    ragged = full.clone()
    ragged[1, seq_len // 2:] = False
    ragged[batch - 1, 3:] = False

    # Sample 0 has no valid token. Its first query row then sees no key, the
    # softmax divides by zero, and the attention output holds a NaN. The mask
    # at the end of the block must remove it.
    empty = full.clone()
    empty[0, :] = False

    return {"all valid": full, "ragged": ragged, "one empty sample": empty}


def build_full_mask_variant() -> None:
    """
    Compile a copy of `_mlx_transformer()` that keeps all three masks.

    The copy is the code as it stood before row 28, with the row 27 fix in
    place. It goes into the module namespace as `_mlx_transformer_full`.
    """
    source = textwrap.dedent(inspect.getsource(tb._mlx_transformer))
    edits = [
        ("valid_tokens = valid_mask[..., None] if padded else None",
         "valid_tokens = valid_mask[..., None]"),
        # Row 36 moved the residual add into the `defer` branch. `defer` is
        # false whenever `padded` is true, so the padded path still runs this
        # line, and it is still the right place to mask the attention output.
        ('            x = x + mx.addmm(layer["ob"], context, layer["ow"].T)\n',
         '            x = x + mx.where(\n'
         '                valid_tokens,\n'
         '                mx.addmm(layer["ob"], context, layer["ow"].T),\n'
         '                0,\n'
         '            )\n'),
        ("        if padded:\n            x = mx.where(valid_tokens, x, 0)\n",
         "        x = mx.where(valid_tokens, x, 0)\n"),
        ("return mx.where(valid_tokens, x, 0) if padded else x",
         "return mx.where(valid_tokens, x, 0)"),
        ("def _mlx_transformer(", "def _mlx_transformer_full("),
    ]
    for old, new in edits:
        if old not in source:
            raise RuntimeError(
                f"_mlx_transformer() no longer holds this text, so the test "
                f"cannot rebuild the full mask variant: {old!r}"
            )
        source = source.replace(old, new, 1)
    exec(compile(source, "<full mask variant>", "exec"), tb.__dict__)


def run(rtol: float, atol: float, seed: int) -> bool:
    build_full_mask_variant()
    passed = True

    print("shape                  causal  mask              accuracy"
          "                 bit exact")
    for causal in (True, False):
        for batch, seq_len, d_model, heads, ffn_dim, layers in SHAPES:
            config = TransformerConfig(
                batch, seq_len, d_model, heads, ffn_dim, layers, causal
            )
            torch.manual_seed(seed)
            baseline = BaselineTransformer(config)
            data = torch.randn(batch, seq_len, d_model)

            for tag, mask in make_masks(batch, seq_len).items():
                # The harness zeroes the padded positions of the input. Do
                # the same, so this test uses the same inputs it does.
                x = data.masked_fill(~mask[..., None], 0)

                model = UserOptimizedTransformer(config)
                copy_model_weights(baseline, model)
                with torch.inference_mode():
                    reference = baseline(x, mask)
                    candidate = model(x, mask).clone()

                # The same comparison the harness makes.
                result = compare_outputs(
                    reference, candidate, rtol=rtol, atol=atol
                )

                # The full mask variant, for the bit exactness test.
                original = tb._mlx_transformer
                tb._mlx_transformer = tb._mlx_transformer_full
                full_model = UserOptimizedTransformer(config)
                copy_model_weights(baseline, full_model)
                with torch.inference_mode():
                    full = full_model(x, mask).clone()
                tb._mlx_transformer = original

                exact = torch.equal(candidate, full)
                nan = bool(torch.isnan(candidate).any())
                ok = result.passed and exact and not nan
                passed &= ok

                head_dim = d_model // heads
                shape = f"B{batch} S{seq_len} D{d_model} H{heads} hd{head_dim}"
                status = "PASS" if result.passed else "FAIL"
                print(
                    f"{shape:22s} {int(causal):^6d}  {tag:16s}  "
                    f"{status} max_abs={result.max_abs_error:.2e} "
                    f"failed={result.failed_elements:<8d}  "
                    f"{'yes' if exact else 'NO'}"
                    f"{'  NaN IN OUTPUT' if nan else ''}"
                )

    print()
    print("PASS" if passed else "FAIL")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    return 0 if run(args.rtol, args.atol, args.seed) else 1


if __name__ == "__main__":
    raise SystemExit(main())

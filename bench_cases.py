#!/usr/bin/env python3
"""
Deterministic test-case generation shared by every backend.

The harness helper `generate_random_case()` seeds a `torch.Generator` on the
target device. The CPU and MPS RNG streams are different, so the same seed
produces different inputs on each device:

    cpu  seed=1234 -> [ 0.0461,  0.4024, -1.0115,  0.2167]
    mps  seed=1234 -> [-0.1472, -0.4256,  0.6888, -1.1171]

Cross-device outputs therefore cannot be compared when each device draws its
own input. Every case here is drawn once on the CPU and then moved, so all
backends see identical bytes.

The tensors are drawn in float32 and cast afterwards. The harness draws
directly in the run dtype, which differs for float16/bfloat16 (a half-precision
normal draw is not a rounded float32 draw), but casting keeps a single stream
that every dtype can share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from torch_transformer_benchmark import TransformerConfig

# Offset the harness applies to the seed of the timing case, kept so the timing
# input here matches the one `benchmark_models()` would have built.
TIMING_SEED_OFFSET = 100000


@dataclass(frozen=True)
class TestCase:
    """One (input, mask) pair. Always held on the CPU in float32."""

    x: torch.Tensor
    valid_mask: torch.Tensor
    seed: int

    def to(self, device: torch.device, dtype: torch.dtype) -> "TestCase":
        return TestCase(
            x=self.x.to(device=device, dtype=dtype),
            valid_mask=self.valid_mask.to(device=device),
            seed=self.seed,
        )


def make_case(
    config: TransformerConfig,
    seed: int,
    padding_ratio: float = 0.0,
    input_scale: float = 1.0,
) -> TestCase:
    """Build a single CPU/float32 case, mirroring `generate_random_case()`."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_mask = torch.ones(
            config.batch_size, config.seq_len, device="cpu", dtype=torch.bool
        )
        return TestCase(x=x, valid_mask=valid_mask, seed=seed)

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device="cpu",
    )
    positions = torch.arange(config.seq_len, device="cpu")[None, :]
    valid_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_mask[..., None], 0)
    return TestCase(x=x, valid_mask=valid_mask, seed=seed)


def make_accuracy_cases(
    config: TransformerConfig,
    seed: int,
    trials: int,
    padding_ratio: float = 0.0,
    input_scale: float = 1.0,
) -> List[TestCase]:
    """The `trials` cases the harness would use for its accuracy check."""
    return [
        make_case(config, seed + trial, padding_ratio, input_scale)
        for trial in range(trials)
    ]


def make_timing_case(
    config: TransformerConfig,
    seed: int,
    padding_ratio: float = 0.0,
    input_scale: float = 1.0,
) -> TestCase:
    """The fixed case the harness would time against."""
    return make_case(config, seed + TIMING_SEED_OFFSET, padding_ratio, input_scale)

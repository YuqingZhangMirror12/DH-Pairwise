"""Symmetric whole-mask coarse scorer for Pairwise v0.2.

The historical MM/ECCV implementations concatenated ``feature_a`` and
``feature_b`` in input order, so A/B exchange invariance was not guaranteed.
This module uses only symmetric feature combinations.  It returns evidence;
the validation-frozen high-recall rejection policy lives in
``training.thresholds`` and is intentionally not embedded in the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CoarseOutput:
    """Whole-mask score and health state for each sample."""

    logit: Tensor
    probability: Tensor
    embedding_a: Tensor
    embedding_b: Tensor
    valid_problem: Tensor


class _ConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class SymmetricCoarseSiamese(nn.Module):
    """Small shared CNN with an exactly symmetric comparison head.

    Non-finite, out-of-range, empty-foreground and empty-background inputs
    invalidate only the affected sample.  Its returned probability is neutral
    (0.5), and downstream policy must pass it onward or abstain rather than
    treating it as a confident rejection.
    """

    def __init__(
        self,
        input_channels: int = 1,
        widths: Tuple[int, ...] = (16, 32, 64),
        embedding_dim: int = 96,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if not widths or min(widths) <= 0:
            raise ValueError("widths must contain positive values")
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        stages = []
        previous = input_channels
        for width in widths:
            stages.append(_ConvStage(previous, width))
            previous = width
        self.encoder = nn.Sequential(*stages)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(previous, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )
        # |a-b|, a*b, (a+b)/2 and elementwise max are all A/B symmetric.
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.input_channels = input_channels

    def _validate(self, a: Tensor, b: Tensor) -> Tensor:
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            raise TypeError("coarse masks must be torch.Tensor values")
        if a.ndim != 4 or b.ndim != 4:
            raise ValueError("coarse masks must have shape [B, C, H, W]")
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError("coarse A/B masks must have identical shapes")
        if a.device != b.device or a.dtype != b.dtype:
            raise ValueError("coarse A/B masks must share device and dtype")
        if a.shape[0] < 1 or a.shape[1] != self.input_channels:
            raise ValueError("coarse mask batch/channel shape is invalid")
        if min(a.shape[-2:]) < 2 ** len(self.encoder):
            raise ValueError("coarse masks are too small for the encoder")
        if not a.is_floating_point() or not b.is_floating_point():
            raise TypeError("coarse masks must use a floating-point dtype")
        finite_a = torch.isfinite(a).flatten(1).all(dim=1)
        finite_b = torch.isfinite(b).flatten(1).all(dim=1)
        safe_a = torch.where(finite_a[:, None, None, None], a, torch.zeros_like(a))
        safe_b = torch.where(finite_b[:, None, None, None], b, torch.zeros_like(b))

        def valid_mask(value: Tensor, finite: Tensor) -> Tensor:
            flattened = value.flatten(1)
            normalized = ((flattened >= 0.0) & (flattened <= 1.0)).all(dim=1)
            has_foreground = flattened.max(dim=1).values > 0.0
            has_background = flattened.min(dim=1).values < 1.0
            return finite & normalized & has_foreground & has_background

        return valid_mask(safe_a, finite_a) & valid_mask(safe_b, finite_b)

    def _encode(self, value: Tensor, valid: Tensor) -> Tensor:
        safe = torch.where(valid[:, None, None, None], value, torch.zeros_like(value))
        result = self.projection(self.encoder(safe))
        return torch.where(valid[:, None], result, torch.zeros_like(result))

    def forward(self, mask_a: Tensor, mask_b: Tensor) -> CoarseOutput:
        valid = self._validate(mask_a, mask_b)
        embedding_a = self._encode(mask_a, valid)
        embedding_b = self._encode(mask_b, valid)
        symmetric = torch.cat(
            [
                torch.abs(embedding_a - embedding_b),
                embedding_a * embedding_b,
                (embedding_a + embedding_b) * 0.5,
                torch.maximum(embedding_a, embedding_b),
            ],
            dim=1,
        )
        raw_logit = self.head(symmetric).squeeze(1)
        # Neutral score plus invalid flag: never a confident negative.
        logit = torch.where(valid, raw_logit, torch.zeros_like(raw_logit))
        return CoarseOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            embedding_a=embedding_a,
            embedding_b=embedding_b,
            valid_problem=valid,
        )


__all__ = ["CoarseOutput", "SymmetricCoarseSiamese"]

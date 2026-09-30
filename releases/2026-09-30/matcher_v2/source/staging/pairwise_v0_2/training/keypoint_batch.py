"""Minimal tensor bridge for the contour-keypoint local research arm.

This bridge intentionally contains no sampling, labels, losses, filesystem
access, or cache writes.  It pads already-selected keypoint candidates so the
existing :class:`OrderedLocalMatcher` can train with the same patch encoder and
either dual-softmax or dustbin Sinkhorn.  ``correspondence_mask`` is the only
new model input.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from staging.pairwise_v0_2.geometry.keypoint_candidates import KeypointPairResult
from staging.pairwise_v0_2.geometry.schema import DEFAULT_DIRECTION_ORDER


@dataclass(frozen=True)
class KeypointTensorBatch:
    local_a: Tensor
    local_b: Tensor
    token_mask_a: Tensor
    token_mask_b: Tensor
    correspondence_mask: Tensor
    sample_index: Tensor
    direction_index: Tensor
    candidate_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.local_a.ndim != 5 or self.local_b.ndim != 5:
            raise ValueError("local tensors must have shape [N, L, C, H, W]")
        count = int(self.local_a.shape[0])
        if int(self.local_b.shape[0]) != count:
            raise ValueError("A/B keypoint candidate counts differ")
        if self.local_a.shape[2:] != self.local_b.shape[2:]:
            raise ValueError("A/B keypoint patch shapes differ")
        if not self.local_a.is_floating_point() or not self.local_b.is_floating_point():
            raise TypeError("local keypoint tensors must be floating point")
        if self.token_mask_a.dtype != torch.bool or tuple(self.token_mask_a.shape) != (
            count,
            int(self.local_a.shape[1]),
        ):
            raise TypeError("token_mask_a must be bool [N, L_a]")
        if self.token_mask_b.dtype != torch.bool or tuple(self.token_mask_b.shape) != (
            count,
            int(self.local_b.shape[1]),
        ):
            raise TypeError("token_mask_b must be bool [N, L_b]")
        expected_pairs = (count, int(self.local_a.shape[1]), int(self.local_b.shape[1]))
        if (
            self.correspondence_mask.dtype != torch.bool
            or tuple(self.correspondence_mask.shape) != expected_pairs
        ):
            raise TypeError("correspondence_mask must be bool [N, L_a, L_b]")
        for name, value in (
            ("sample_index", self.sample_index),
            ("direction_index", self.direction_index),
        ):
            if value.dtype != torch.long or tuple(value.shape) != (count,):
                raise TypeError("{} must be int64 [N]".format(name))
        if len(self.candidate_ids) != count or len(set(self.candidate_ids)) != count:
            raise ValueError("candidate_ids must be unique and match N")
        if count:
            if not self.token_mask_a.any(dim=1).all().item():
                raise ValueError("every candidate needs at least one A token")
            if not self.token_mask_b.any(dim=1).all().item():
                raise ValueError("every candidate needs at least one B token")
            active = (
                self.correspondence_mask
                & self.token_mask_a[:, :, None]
                & self.token_mask_b[:, None, :]
            )
            if not active.flatten(1).any(dim=1).all().item():
                raise ValueError("every candidate needs an allowed keypoint pair")
            if ((self.direction_index < 0) | (self.direction_index >= 4)).any().item():
                raise ValueError("direction_index is out of range")

    @property
    def candidate_count(self) -> int:
        return int(self.local_a.shape[0])

    def model_inputs(self) -> Dict[str, Tensor]:
        return {
            "patches_a": self.local_a,
            "patches_b": self.local_b,
            "token_mask_a": self.token_mask_a,
            "token_mask_b": self.token_mask_b,
            "correspondence_mask": self.correspondence_mask,
        }

    def to(self, device: torch.device) -> "KeypointTensorBatch":
        return replace(
            self,
            local_a=self.local_a.to(device),
            local_b=self.local_b.to(device),
            token_mask_a=self.token_mask_a.to(device),
            token_mask_b=self.token_mask_b.to(device),
            correspondence_mask=self.correspondence_mask.to(device),
            sample_index=self.sample_index.to(device),
            direction_index=self.direction_index.to(device),
        )


def build_keypoint_tensor_batch(
    results: Sequence[KeypointPairResult],
) -> KeypointTensorBatch:
    """Pad one or more four-direction keypoint results for the local matcher."""

    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise TypeError("results must be a finite sequence")
    if not results:
        raise ValueError("results cannot be empty")
    if any(not isinstance(value, KeypointPairResult) for value in results):
        raise TypeError("every result must be KeypointPairResult")
    flattened = [
        (sample_index, candidate)
        for sample_index, result in enumerate(results)
        for candidate in result.candidates
    ]
    if not flattened:
        raise ValueError("keypoint results contain no usable candidates")
    patch_shape = flattened[0][1].patches_a.shape[1:]
    if any(
        candidate.patches_a.shape[1:] != patch_shape
        or candidate.patches_b.shape[1:] != patch_shape
        for _, candidate in flattened
    ):
        raise ValueError("keypoint candidates use different patch shapes")

    count = len(flattened)
    max_a = max(len(candidate.tokens_a) for _, candidate in flattened)
    max_b = max(len(candidate.tokens_b) for _, candidate in flattened)
    local_a = torch.zeros((count, max_a, *patch_shape), dtype=torch.float32)
    local_b = torch.zeros((count, max_b, *patch_shape), dtype=torch.float32)
    mask_a = torch.zeros((count, max_a), dtype=torch.bool)
    mask_b = torch.zeros((count, max_b), dtype=torch.bool)
    correspondence = torch.zeros((count, max_a, max_b), dtype=torch.bool)
    sample_indices = []
    direction_indices = []
    candidate_ids = []
    for index, (sample_index, candidate) in enumerate(flattened):
        length_a = len(candidate.tokens_a)
        length_b = len(candidate.tokens_b)
        local_a[index, :length_a].copy_(
            torch.from_numpy(np.asarray(candidate.patches_a, dtype=np.float32).copy())
        )
        local_b[index, :length_b].copy_(
            torch.from_numpy(np.asarray(candidate.patches_b, dtype=np.float32).copy())
        )
        mask_a[index, :length_a] = True
        mask_b[index, :length_b] = True
        correspondence[index, :length_a, :length_b].copy_(
            torch.from_numpy(
                np.asarray(candidate.correspondence_mask, dtype=np.bool_).copy()
            )
        )
        sample_indices.append(sample_index)
        direction_indices.append(DEFAULT_DIRECTION_ORDER.index(candidate.direction))
        candidate_ids.append(
            "sample/{:06d}:{}".format(sample_index, candidate.candidate_id)
        )
    return KeypointTensorBatch(
        local_a=local_a,
        local_b=local_b,
        token_mask_a=mask_a,
        token_mask_b=mask_b,
        correspondence_mask=correspondence,
        sample_index=torch.tensor(sample_indices, dtype=torch.long),
        direction_index=torch.tensor(direction_indices, dtype=torch.long),
        candidate_ids=tuple(candidate_ids),
    )


__all__ = ["KeypointTensorBatch", "build_keypoint_tensor_batch"]

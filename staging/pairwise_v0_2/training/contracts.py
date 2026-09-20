"""Tensor-only batch contract between geometry/data code and the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class PairwiseBatch:
    """One padded variable-length local-patch batch.

    Geometry code owns contour/candidate extraction.  This class deliberately
    knows only tensors, so archive/manifest implementations can change without
    coupling model code to paths or ZIP member names.
    """

    coarse_a: Tensor
    coarse_b: Tensor
    local_a: Tensor
    local_b: Tensor
    token_mask_a: Tensor
    token_mask_b: Tensor
    label: Tensor
    correspondence: Optional[Tensor] = None
    correspondence_mask: Optional[Tensor] = None
    sample_id: Optional[Any] = None
    cluster_id: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.coarse_a.ndim != 4 or self.coarse_b.ndim != 4:
            raise ValueError("coarse tensors must have shape [B, C, H, W]")
        if self.local_a.ndim != 5 or self.local_b.ndim != 5:
            raise ValueError("local tensors must have shape [B, L, C, H, W]")
        batch = self.coarse_a.shape[0]
        if batch < 1 or self.coarse_b.shape[0] != batch:
            raise ValueError("coarse A/B batch sizes are invalid")
        if self.local_a.shape[0] != batch or self.local_b.shape[0] != batch:
            raise ValueError("coarse and local batch sizes differ")
        if (
            not self.coarse_a.is_floating_point()
            or not self.coarse_b.is_floating_point()
        ):
            raise TypeError("coarse tensors must use floating-point dtypes")
        if not self.local_a.is_floating_point() or not self.local_b.is_floating_point():
            raise TypeError("local tensors must use floating-point dtypes")
        expected_a = (batch, self.local_a.shape[1])
        expected_b = (batch, self.local_b.shape[1])
        if (
            self.token_mask_a.dtype != torch.bool
            or tuple(self.token_mask_a.shape) != expected_a
        ):
            raise TypeError("token_mask_a must be bool with shape [B, L_a]")
        if (
            self.token_mask_b.dtype != torch.bool
            or tuple(self.token_mask_b.shape) != expected_b
        ):
            raise TypeError("token_mask_b must be bool with shape [B, L_b]")
        if self.label.dtype != torch.bool or tuple(self.label.shape) != (batch,):
            raise TypeError("label must be an explicit bool tensor with shape [B]")
        if self.correspondence is not None:
            expected = (batch, self.local_a.shape[1], self.local_b.shape[1])
            if tuple(self.correspondence.shape) != expected:
                raise ValueError("correspondence must have shape [B, L_a, L_b]")
            if not self.correspondence.is_floating_point():
                raise TypeError("correspondence must use a floating-point dtype")
            if self.correspondence_mask is None:
                raise ValueError("correspondence requires correspondence_mask")
        if self.correspondence_mask is not None:
            expected = (batch, self.local_a.shape[1], self.local_b.shape[1])
            if (
                self.correspondence_mask.dtype != torch.bool
                or tuple(self.correspondence_mask.shape) != expected
            ):
                raise TypeError("correspondence_mask must be bool [B, L_a, L_b]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PairwiseBatch":
        required = (
            "coarse_a",
            "coarse_b",
            "local_a",
            "local_b",
            "token_mask_a",
            "token_mask_b",
            "label",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError("batch mapping is missing {}".format(", ".join(missing)))
        return cls(
            coarse_a=value["coarse_a"],
            coarse_b=value["coarse_b"],
            local_a=value["local_a"],
            local_b=value["local_b"],
            token_mask_a=value["token_mask_a"],
            token_mask_b=value["token_mask_b"],
            label=value["label"],
            correspondence=value.get("correspondence"),
            correspondence_mask=value.get("correspondence_mask"),
            sample_id=value.get("sample_id"),
            cluster_id=value.get("cluster_id"),
        )

    def to(self, device: torch.device) -> "PairwiseBatch":
        return PairwiseBatch(
            coarse_a=self.coarse_a.to(device),
            coarse_b=self.coarse_b.to(device),
            local_a=self.local_a.to(device),
            local_b=self.local_b.to(device),
            token_mask_a=self.token_mask_a.to(device),
            token_mask_b=self.token_mask_b.to(device),
            label=self.label.to(device),
            correspondence=(
                None if self.correspondence is None else self.correspondence.to(device)
            ),
            correspondence_mask=(
                None
                if self.correspondence_mask is None
                else self.correspondence_mask.to(device)
            ),
            sample_id=self.sample_id,
            cluster_id=self.cluster_id,
        )

    def swapped(self) -> "PairwiseBatch":
        return PairwiseBatch(
            coarse_a=self.coarse_b,
            coarse_b=self.coarse_a,
            local_a=self.local_b,
            local_b=self.local_a,
            token_mask_a=self.token_mask_b,
            token_mask_b=self.token_mask_a,
            label=self.label,
            correspondence=(
                None
                if self.correspondence is None
                else self.correspondence.transpose(1, 2)
            ),
            correspondence_mask=(
                None
                if self.correspondence_mask is None
                else self.correspondence_mask.transpose(1, 2)
            ),
            sample_id=self.sample_id,
            cluster_id=self.cluster_id,
        )


__all__ = ["PairwiseBatch"]

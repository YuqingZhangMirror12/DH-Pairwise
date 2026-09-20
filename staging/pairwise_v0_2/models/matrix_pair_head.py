"""Shape-only pair classification from an unchanged soft matching matrix.

This is ShreddingNet-inspired matrix-pattern classification, not its RGB model
or its 0.006 dual-softmax threshold. Real Sinkhorn mass stays continuous and
padding stays explicit. There is no pose prediction, correspondence selection,
hard gate, or feedback into the matcher. The matrix-only head never consumes
coarse/local logits. A separate affine fusion consumes its detached logit and
the original coarse logit, so fusion training cannot alter matrix-only learning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MatrixPairHeadConfig:
    widths: Tuple[int, ...] = (16, 32, 64)
    affinity_scale: float = 8.0
    include_affinity: bool = True
    symmetric: bool = True

    def __post_init__(self):
        if not self.widths or any(type(x) is not int or x <= 0 for x in self.widths):
            raise ValueError("widths must be positive integers")
        if not 0 < self.affinity_scale < float("inf"):
            raise ValueError("affinity_scale must be finite and positive")


@dataclass(frozen=True)
class MatrixPairHeadOutput:
    matrix_logit: Tensor
    matrix_coarse_logit: Tensor
    existing_coarse_logit: Tensor
    existing_local_logit: Optional[Tensor]
    valid_problem: Tensor


class MatrixPairHead(nn.Module):
    """Three spatial CNN stages, valid-aware pooling, and swap symmetry.

    No fixed image resize is performed: rows/columns remain ordered contour
    indices. Inputs are detached at this boundary to enforce the frozen-matcher
    experiment even when the caller supplies online differentiable tensors.
    Optional points are accepted for the shared cache interface but NOT used in
    this first matrix-structure-only experiment. No target pose is an input.
    """

    def __init__(self, config: MatrixPairHeadConfig = MatrixPairHeadConfig()):
        super().__init__()
        self.config = config
        channels = 3 if config.include_affinity else 2
        layers = []
        for width in config.widths:
            layers.append(nn.Conv2d(channels, width, 3, padding=1))
            channels = width
        self.convolutions = nn.ModuleList(layers)
        self.classifier = nn.Linear(channels, 1)
        self.coarse_fusion = nn.Linear(2, 1)
        with torch.no_grad():
            self.coarse_fusion.weight.copy_(torch.tensor([[1.0, 0.0]]))
            self.coarse_fusion.bias.zero_()

    def matrix_channels(self, real_transport: Tensor, affinity: Tensor,
                        valid_a: Tensor, valid_b: Tensor):
        if real_transport.ndim != 3 or min(real_transport.shape) < 1:
            raise ValueError("real_transport must be nonempty [B,N,M]")
        if affinity.shape != real_transport.shape:
            raise ValueError("affinity must match real_transport")
        b, n, m = real_transport.shape
        if valid_a.shape != (b, n) or valid_b.shape != (b, m):
            raise ValueError("valid masks must match matrix axes")
        if valid_a.dtype != torch.bool or valid_b.dtype != torch.bool:
            raise ValueError("valid masks must be bool")
        mask = (valid_a[:, :, None] & valid_b[:, None, :]).detach()
        p = torch.where(mask, real_transport.detach().float(), 0.0)
        a = torch.where(mask, affinity.detach().float(), 0.0)
        if not torch.isfinite(p).all() or (p < 0).any():
            raise ValueError("valid transport must be finite and nonnegative")
        if not torch.isfinite(a).all():
            raise ValueError("valid affinity must be finite")
        # P is real_transport with per-point total mass approximately <= 1,
        # not an arbitrary probability rescaled to unit total matrix mass.
        # This monotonic log transform keeps weak evidence and handles changing
        # valid contour lengths; it is NOT a correspondence cutoff.
        scale = (valid_a.sum(1).float() * valid_b.sum(1).float()).sqrt().clamp_min(1)
        confidence = torch.log1p(p * scale[:, None, None]) / torch.log1p(scale)[:, None, None]
        channels = [confidence]
        if self.config.include_affinity:
            channels.append(torch.tanh(a / self.config.affinity_scale))
        channels.append(mask.float())
        return torch.stack(channels, dim=1), mask[:, None].float()

    def _encode(self, values: Tensor, mask: Tensor):
        for convolution in self.convolutions:
            values = F.gelu(convolution(values)) * mask
            # Pool features over valid cells only, avoiding padding dilution.
            denominator = F.avg_pool2d(mask, 2, ceil_mode=True, count_include_pad=False)
            values = F.avg_pool2d(values, 2, ceil_mode=True, count_include_pad=False)
            values = values / denominator.clamp_min(1e-8)
            mask = (denominator > 0).to(values.dtype)
            values = values * mask
        return (values * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)

    def forward(self, real_transport: Tensor, affinity: Tensor,
                valid_a: Tensor, valid_b: Tensor, coarse_logit: Tensor,
                local_logit: Optional[Tensor] = None,
                points_a_rc: Optional[Tensor] = None,
                points_b_rc: Optional[Tensor] = None) -> MatrixPairHeadOutput:
        values, mask = self.matrix_channels(real_transport, affinity, valid_a, valid_b)
        batch = values.shape[0]
        if coarse_logit.shape != (batch,) or not torch.isfinite(coarse_logit).all():
            raise ValueError("coarse_logit must be finite [B]")
        if local_logit is not None and local_logit.shape != (batch,):
            raise ValueError("local_logit must be [B]")
        features = self._encode(values, mask)
        if self.config.symmetric:
            features = (features + self._encode(values.transpose(2, 3), mask.transpose(2, 3))) * 0.5
        logit = self.classifier(features).squeeze(1)
        fused = self.coarse_fusion(torch.stack((logit.detach(), coarse_logit.detach().float()), dim=1)).squeeze(1)
        return MatrixPairHeadOutput(logit, fused, coarse_logit, local_logit,
                                   valid_a.any(1) & valid_b.any(1))

    def architecture_metadata(self):
        return dict(config=asdict(self.config), parameter_count=sum(p.numel() for p in self.parameters()),
                    input_channels=["log1p_soft_transport"] + (["tanh_affinity"] if self.config.include_affinity else []) + ["valid_pair_mask"],
                    matcher_detached=True, matrix_only_uses_coarse=False,
                    fusion="affine(detached_matrix_logit, original_coarse_logit)",
                    points_used=False, pose_used=False, correspondence_threshold=None)


def load_matrix_pair_head(path: str | Path, branch: str = "matrix_only",
                          device: str = "cpu") -> MatrixPairHead:
    """Load a validation-frozen branch winner from the independent runner."""
    if branch not in ("matrix_only", "matrix_coarse"):
        raise ValueError("unknown matrix head branch")
    receipt: Mapping = torch.load(path, map_location="cpu", weights_only=False)
    if receipt.get("schema_version") != "rachel-matrix-pair-head/v1" or receipt.get("status") != "complete":
        raise ValueError("head checkpoint must be a complete validation-frozen receipt")
    config = dict(receipt["model_config"])
    config["widths"] = tuple(config["widths"])
    model = MatrixPairHead(MatrixPairHeadConfig(**config))
    model.load_state_dict(receipt["selected_heads"][branch]["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)

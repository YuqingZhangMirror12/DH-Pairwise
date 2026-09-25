"""Finite directional damage compatibility, separate from zero-offset location.

This module never sees GT, recipe labels or source-anchor labels at inference.
Parameters must be bound to the TRAIN-only calibration receipt.
"""
from dataclasses import dataclass, fields
import math

import torch
from torch import Tensor

from .geometry import pair_frame


@dataclass(frozen=True)
class CompatibilityConfig:
    normal_reliability_min: float
    normal_sigma_per_spacing: float
    tangent_sigma_per_spacing: float
    fallback_sigma_per_spacing: float
    sigma_floor_px: float
    damage_normal_upper_px: float
    evidence_tangent_sigma_per_spacing: float = 0.5
    damage_normal_lower_px: float = 0.
    damage_tangent_offset_px: float = 0.
    unreliable_normal_damage_offset_px: float = 0.

    @classmethod
    def from_calibration(cls, record):
        if record.get('status') != 'complete' or record.get('schema') != 's7-consensus-train-geometry/2':
            raise ValueError('requires completed TRAIN-only geometry calibration')
        names = {f.name for f in fields(cls)}
        return cls(**{k:v for k,v in record['parameters'].items() if k in names})

    def __post_init__(self):
        if (not all(math.isfinite(getattr(self, f.name)) for f in fields(self)) or
                not 0 < self.normal_reliability_min <= 1 or self.damage_normal_upper_px <= 0
                or min(self.normal_sigma_per_spacing, self.tangent_sigma_per_spacing,
                       self.fallback_sigma_per_spacing, self.sigma_floor_px,
                       self.evidence_tangent_sigma_per_spacing) <= 0
                or self.damage_normal_lower_px != 0 or self.damage_tangent_offset_px != 0
                or self.unreliable_normal_damage_offset_px != 0):
            raise ValueError('invalid finite directional compatibility contract')


@dataclass(frozen=True)
class Compatibility:
    kernel: Tensor
    localization_kernel: Tensor
    normal_px: Tensor
    tangent_px: Tensor
    normal_reliability: Tensor
    reliable_normal: Tensor
    material_offset_rc: Tensor
    unexplained_residual_rc: Tensor


def damage_compatibility(residual_rc: Tensor, normal_a: Tensor, normal_b: Tensor,
                         reliability_a: Tensor, reliability_b: Tensor,
                         spacing_a_px: Tensor, spacing_b_px: Tensor,
                         config: CompatibilityConfig) -> Compatibility:
    """Compute exp(-distance-to-finite-D^2/2), preserving absolute Q elsewhere.

    Positive A-outward normal residual may represent material loss. Negative
    normal residual (penetration) and tangential slide are not damage offsets.
    The returned localization kernel does NOT excuse normal gaps. It is a
    measurement feature, not automatic proof of a precise/correct anchor.
    """
    normal, tangent, reliability = pair_frame(normal_a, normal_b, reliability_a, reliability_b)
    reliable = reliability >= config.normal_reliability_min
    rn = (residual_rc * normal).sum(-1)
    rt = (residual_rc * tangent).sum(-1)
    spacing = .5 * (spacing_a_px + spacing_b_px)
    sn = (config.normal_sigma_per_spacing * spacing).clamp_min(config.sigma_floor_px)
    st = (config.tangent_sigma_per_spacing * spacing).clamp_min(config.sigma_floor_px)
    se = (config.evidence_tangent_sigma_per_spacing * spacing).clamp_min(config.sigma_floor_px)
    sf = (config.fallback_sigma_per_spacing * spacing).clamp_min(config.sigma_floor_px)
    offset_n = torch.where(reliable, rn.clamp(0, config.damage_normal_upper_px), torch.zeros_like(rn))
    offset_rc = offset_n[..., None] * normal
    fallback_d2 = (residual_rc/sf[..., None]).square().sum(-1)
    compatibility_d2 = torch.where(reliable, ((rn-offset_n)/sn).square()+(rt/se).square(), fallback_d2)
    location_d2 = torch.where(reliable, (rn/sn).square()+(rt/st).square(), fallback_d2)
    return Compatibility(torch.exp(-.5*compatibility_d2), torch.exp(-.5*location_d2),
        rn, rt, reliability, reliable, offset_rc, residual_rc-offset_rc)

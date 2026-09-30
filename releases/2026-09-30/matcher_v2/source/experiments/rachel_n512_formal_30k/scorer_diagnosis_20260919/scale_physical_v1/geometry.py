"""Common-center physical downscaling for a frozen, fixed-token causal probe.

Pure tensor geometry: no contour extraction, model loading, GT, or clipping.
The default800 canvas has common center399.5 in both row and column. Output
pixels inverse-map into the original binary image with nearest/zero padding
and align_corners=True. Raster pixel counts are observations, not material loss.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch.nn import functional as F


def checked_scale(scale):
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise ValueError("scale must be a finite real number in (0,1]")
    value = float(scale)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("scale must be a finite real number in (0,1]")
    return value


def corrected_window_sizes(window_sizes, scale):
    """Preserve physical sampling spans, not nominal pixel-count multiplication.

    ContourPatchSampler uses halfspan=(w-1)/2, hence new w=1+s*(w-1).
    Fractional nominal sizes are intentional; no rounding is performed.
    """
    scale = checked_scale(scale)
    values = tuple(float(w) for w in window_sizes)
    if not values or any(not math.isfinite(w) or w < 1 for w in values):
        raise ValueError("window sizes must be finite and >=1")
    return tuple(1. + scale*(w-1.) for w in values)


def cloned_sampler(sampler, scale):
    """Clone a sampler and compensate its exact existing offsets, preserving source."""
    scale = checked_scale(scale)
    offsets = getattr(sampler, "offsets_rc", None)
    if (not isinstance(offsets, torch.Tensor) or not offsets.is_floating_point()
            or offsets.ndim < 2 or offsets.shape[-1] != 2 or not torch.isfinite(offsets).all()):
        raise ValueError("sampler must expose finite floating offsets_rc[...,2]")
    sizes = corrected_window_sizes(sampler.window_sizes_px, scale)
    if offsets.shape[0] != len(sizes):
        raise ValueError("window count differs from sampler offsets")
    result = deepcopy(sampler)
    with torch.no_grad():
        result.offsets_rc.mul_(scale)
    result.window_sizes_px = sizes
    return result


@dataclass(frozen=True)
class ScaledFragment:
    mask: torch.Tensor
    points_rc: torch.Tensor
    valid: torch.Tensor
    diagnostics: dict


@dataclass(frozen=True)
class ScaledPair:
    a: ScaledFragment
    b: ScaledFragment

    @property
    def inputs(self):
        return (self.a.mask, self.b.mask, self.a.points_rc, self.b.points_rc, self.a.valid, self.b.valid)

    @property
    def diagnostics(self):
        return dict(scale=self.a.diagnostics["scale"], center_rc=self.a.diagnostics["center_rc"],
                    a=self.a.diagnostics, b=self.b.diagnostics,
                    transform="same global center and scale for both fragments; no independent recentering")


def downscale_fragment(mask, points_rc, valid, scale, *, canvas_size=800):
    """Transform a batched binary mask and its EXISTING fixed contour tokens.

    Torch tensors or NumPy arrays are accepted; results are tensors. Mask dtype,
    point dtype and device are preserved. Valid coordinates must be on-canvas;
    invalid/padded coordinates are copied bit-for-bit, never reinterpreted or
    turned into valid contour points. Neither inputs nor token order are changed.
    """
    scale = checked_scale(scale)
    if type(canvas_size) is not int or canvas_size < 2:
        raise ValueError("canvas_size must be an integer >=2")
    mask, points_rc, valid = map(torch.as_tensor, (mask, points_rc, valid))
    if (mask.ndim != 4 or mask.shape[1:] != (1, canvas_size, canvas_size)
            or points_rc.ndim != 3 or points_rc.shape[-1] != 2
            or points_rc.shape[0] != mask.shape[0] or valid.shape != points_rc.shape[:2]
            or valid.dtype != torch.bool or not points_rc.is_floating_point()
            or mask.device != points_rc.device or mask.device != valid.device):
        raise ValueError("expected common-device binary Bx1xHxH, floating BxNx2, boolean BxN")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must already be binary0/1")
    active = points_rc[valid]
    if not torch.isfinite(active).all() or (active < 0).any() or (active > canvas_size-1).any():
        raise ValueError("valid points must be finite and inside the original canvas")
    center = (canvas_size-1.)/2.
    with torch.no_grad():
        if scale == 1.:
            # Exact identity also preserves non-floating binary mask dtypes.
            result_mask = mask.clone()
            result_points = points_rc.clone()
        else:
            axis = torch.arange(canvas_size, device=mask.device, dtype=torch.float32)
            inverse = center+(axis-center)/scale
            rr, cc = torch.meshgrid(inverse, inverse, indexing="ij")
            grid = torch.stack((2.*cc/(canvas_size-1.)-1., 2.*rr/(canvas_size-1.)-1.), dim=-1)
            grid = grid[None].expand(mask.shape[0], -1, -1, -1)
            result_mask = F.grid_sample(mask.to(torch.float32), grid, mode="nearest",
                padding_mode="zeros", align_corners=True).to(mask.dtype)
            transformed = center+scale*(points_rc-center)
            result_points = torch.where(valid[..., None], transformed, points_rc).clone()
        result_valid = valid.clone()
        before = (mask != 0).sum(dim=(1,2,3)).cpu().tolist()
        after = (result_mask != 0).sum(dim=(1,2,3)).cpu().tolist()
    diagnostics = dict(scale=scale, canvas_size=canvas_size, center_rc=[center, center],
        pixel_count_before=before, pixel_count_after=after,
        observed_pixel_count_ratio=[new/old if old else None for old,new in zip(before,after)],
        nominal_area_scale=scale*scale, token_count=points_rc.shape[1],
        valid_token_count=valid.sum(dim=1).cpu().tolist(), point_count_changed=False,
        contour_reextracted=False, clipping_applied=False,
        raster_note="nearest integer resampling; pixel count ratios need not equal s^2 and do not measure material loss")
    return ScaledFragment(result_mask, result_points, result_valid, diagnostics)


def downscale_pair(mask_a, mask_b, points_rc_a, points_rc_b, valid_a, valid_b, scale, *, canvas_size=800):
    """Apply exactly one shared transform to both sides of the six-input pair."""
    a = downscale_fragment(mask_a, points_rc_a, valid_a, scale, canvas_size=canvas_size)
    b = downscale_fragment(mask_b, points_rc_b, valid_b, scale, canvas_size=canvas_size)
    if a.mask.shape[0] != b.mask.shape[0]:
        raise ValueError("pair batch sizes differ")
    return ScaledPair(a, b)

"""Single-fragment inward edge-weathering prototype, not an MM reproduction.

No pair labels, other fragments, clean correspondences or pose targets are
accepted. The 800x800 coordinate frame is retained exactly. This changes the
contour, so a caller must re-extract model contours AND must not reuse old
token-index correspondence targets. No training targets are produced here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .rachel_preprocess import RachelPreprocessError, extract_ordered_outer_contour


@dataclass(frozen=True)
class EdgeWeatheringConfig:
    max_depth_px: float = 2.0
    correlation_length_range_px: tuple = (16.0, 64.0)
    coverage_fraction_range: tuple = (0.2, 0.5)
    max_area_loss_fraction: float = 0.03

    def __post_init__(self):
        if not np.isfinite(self.max_depth_px) or self.max_depth_px < 0:
            raise ValueError("max_depth_px must be finite and nonnegative")
        lo, hi = self.correlation_length_range_px
        if not np.isfinite((lo, hi)).all() or not 0 < lo <= hi:
            raise ValueError("correlation lengths must be finite, positive and ordered")
        lo, hi = self.coverage_fraction_range
        if not np.isfinite((lo, hi)).all() or not 0 <= lo <= hi <= 1:
            raise ValueError("coverage fractions must be ordered within [0,1]")
        if not np.isfinite(self.max_area_loss_fraction) or not 0 <= self.max_area_loss_fraction <= 1:
            raise ValueError("area-loss fraction must be within [0,1]")


_CROSS = ndimage.generate_binary_structure(2, 1)


def _topology_reason(mask):
    if not mask.any():
        return "empty_mask"
    if ndimage.label(mask, structure=_CROSS)[1] != 1:
        return "disconnected_mask"
    if not np.array_equal(ndimage.binary_fill_holes(mask, structure=_CROSS), mask):
        return "mask_has_holes"
    return None


def _arc_depth(points, rng, config):
    """C1 nonnegative compact bumps on the closed contour's actual arc length.

    Bump support lengths (not Gaussian sigma) define the correlation scale.
    Disjoint supports occupy the sampled target fraction of the full perimeter;
    random gaps, circular phase and amplitudes avoid a periodic tooth pattern.
    Very short contours may have support shorter than the nominal lower bound.
    """
    segment_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    perimeter = float(segment_lengths.sum())
    arc = np.r_[0., np.cumsum(segment_lengths[:-1])]
    coverage = float(rng.uniform(*config.coverage_fraction_range))
    support_total = coverage * perimeter
    depth = np.zeros(len(points), np.float64)
    support_width, count = 0., 0
    if support_total > 0:
        lo, hi = config.correlation_length_range_px
        min_count = max(1, int(np.ceil(support_total / hi)))
        max_count = max(min_count, int(np.floor(support_total / lo)))
        count = int(rng.integers(min_count, max_count + 1))
        support_width = support_total / count
        gaps = rng.dirichlet(np.full(count, 2.0)) * (perimeter - support_total)
        start = float(rng.uniform(0., perimeter))
        for gap in gaps:
            relative = np.mod(arc - start, perimeter)
            inside = relative < support_width
            amplitude = float(rng.uniform(.5, 1.)) * config.max_depth_px
            depth[inside] = amplitude * .5 * (1. - np.cos(2. * np.pi * relative[inside] / support_width))
            start = (start + support_width + float(gap)) % perimeter
    measured_coverage = float(segment_lengths[depth > 0].sum() / perimeter)
    return arc, depth, dict(perimeter_px=perimeter, requested_coverage_fraction=coverage,
        actual_coverage_fraction=measured_coverage, bump_count=count,
        correlation_support_px=support_width,
        tiny_support_clipped=bool(0 < support_width < config.correlation_length_range_px[0]),
        max_sampled_depth_px=float(depth.max(initial=0.)))


def weather_fragment_edges(mask, *, seed, fragment_id, config=EdgeWeatheringConfig(), return_depth=False):
    """Return ``(new_bool_mask, JSON-serializable_report)`` without input writes.

    Local random state is SHA-derived from seed and fragment_id. A pixel in the
    inward boundary band inherits its nearest original contour point's depth.
    Its inward distance uses padded-mask EDT minus 0.5 pixel (the pixel-cell
    boundary convention); only pixels inside the original mask can be deleted.

    Reject empty/disconnected/holed proposals or excessive area loss explicitly,
    returning a copy of the original. Never fill holes or keep only a main piece.
    return_depth adds JSON lists of original contour points/arclength/depth for
    visualization, not an N512 contour or valid training correspondences.
    """
    value = np.asarray(mask)
    if value.dtype != np.bool_ or value.shape != (800, 800):
        raise ValueError("mask must be bool [800,800] in the unchanged model frame")
    if type(seed) is not int or not isinstance(fragment_id, str) or not fragment_id:
        raise ValueError("integer seed and nonempty fragment_id are required")
    if not isinstance(config, EdgeWeatheringConfig):
        raise TypeError("config must be EdgeWeatheringConfig")
    original = value.copy()
    area = int(original.sum())
    report = dict(schema_version="rachel-edge-weathering-prototype/v1", config=asdict(config),
        seed=seed, fragment_id=fragment_id, status="identity", applied=False, skipped=False,
        skip_reason=None, identity_reason=None, original_area_px=area, proposed_removed_area_px=0,
        proposed_removed_fraction=0., removed_area_px=0, removed_fraction=0., contour_changed=False,
        frame_unchanged=True, pixels_added=0, topology_connectivity=4,
        method="local compact raised-cosine arc-depth bumps, inward deletion only; our proposal, not MM reproduction",
        correspondence_targets_generated=False, old_correspondence_targets_reusable=False)

    def skip(reason):
        report.update(status="skipped", skipped=True, skip_reason=reason)
        return original.copy(), report

    if config.max_depth_px == 0 or config.coverage_fraction_range[1] == 0:
        report["identity_reason"] = "zero_strength"
        return original, report
    reason = _topology_reason(original)
    if reason:
        return skip("input_" + reason)
    try:
        points, _ = extract_ordered_outer_contour(original, cap=original.size, smoothing_sigma=0.)
    except RachelPreprocessError:
        return skip("input_contour_too_small_or_invalid")
    digest = hashlib.sha256((str(seed) + "\0" + fragment_id).encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    arc, depth, diagnostics = _arc_depth(points, rng, config)
    report.update(diagnostics)
    if return_depth:
        report["depth_diagnostics"] = dict(contour_points_rc=points.tolist(), arclength_px=arc.tolist(),
                                           depth_px=depth.tolist())
    inward = ndimage.distance_transform_edt(np.pad(original, 1))[1:-1, 1:-1] - .5
    band_points = np.argwhere(original & (inward <= config.max_depth_px))
    proposal = original.copy()
    if len(band_points):
        nearest = cKDTree(points).query(band_points)[1]
        remove = inward[band_points[:, 0], band_points[:, 1]] <= depth[nearest]
        chosen = band_points[remove]
        proposal[chosen[:, 0], chosen[:, 1]] = False
    removed = int(area - proposal.sum())
    fraction = removed / max(1, area)
    report.update(proposed_removed_area_px=removed, proposed_removed_fraction=fraction)
    if not removed:
        report["identity_reason"] = "no_pixel_reached_by_depth"
        return original, report
    reason = _topology_reason(proposal)
    if reason:
        return skip("proposal_" + reason)
    if fraction > config.max_area_loss_fraction + 1e-12:
        return skip("area_loss_exceeds_limit")
    try:
        new_points, _ = extract_ordered_outer_contour(proposal, cap=proposal.size, smoothing_sigma=0.)
    except RachelPreprocessError:
        return skip("proposal_contour_too_small_or_invalid")
    report.update(status="applied", applied=True, removed_area_px=removed, removed_fraction=fraction,
                  contour_changed=not np.array_equal(points, new_points))
    return proposal, report


__all__ = ["EdgeWeatheringConfig", "weather_fragment_edges"]

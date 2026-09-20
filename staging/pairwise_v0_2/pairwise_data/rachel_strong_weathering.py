"""CPU-only 10--30 px inward weathering in the unchanged fragment pixel frame.

This is geometry, not training supervision. No pair label/translation is taken.
Wave descendants may be projected to original source arcs by the caller; new
local/seam notches MUST be ignored as positive correspondence ancestry. The
report supplies the affected source arcs/points for that separate operation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .rachel_preprocess import RachelPreprocessError, extract_ordered_outer_contour


@dataclass(frozen=True)
class StrongWeatheringConfig:
    depth_range_px: tuple = (10., 30.)
    wave_correlation_px_range: tuple = (30., 90.)
    local_notch_count_range: tuple = (1, 3)
    local_support_px_range: tuple = (30., 90.)
    gap_count_range: tuple = (1, 5)
    gap_support_px_range: tuple = (15., 50.)
    minimum_arc_separation_px: float = 5.
    seam_point_tolerance_px: float = 1.5
    seam_short_gap_bridge_px: float = 0.
    topology_connectivity: int = 4
    source_arc_guard_px: float = 2.
    raster_depth_tolerance_px: float = 1.5
    max_removed_fraction: float = .75
    min_remaining_area_px: int = 16
    max_attempts: int = 16

    def __post_init__(self):
        for name in ("depth_range_px", "wave_correlation_px_range", "local_support_px_range", "gap_support_px_range"):
            value = getattr(self, name)
            if len(value) != 2 or not np.isfinite(value).all() or not 0 < value[0] <= value[1]:
                raise ValueError(name + " must be finite, positive and ordered")
            object.__setattr__(self, name, tuple(float(x) for x in value))
        if not 10 <= self.depth_range_px[0] <= self.depth_range_px[1] <= 30:
            raise ValueError("strong depth must remain within 10--30 original-frame pixels")
        if not 15 <= self.gap_support_px_range[0] <= self.gap_support_px_range[1] <= 50:
            raise ValueError("seam gap supports must remain within 15--50 arc-length pixels")
        for name in ("local_notch_count_range", "gap_count_range"):
            value = getattr(self, name)
            if len(value) != 2 or any(type(x) is not int for x in value) or not 1 <= value[0] <= value[1] <= 5:
                raise ValueError(name + " must contain ordered integers within 1--5")
            object.__setattr__(self, name, tuple(value))
        for name in ("minimum_arc_separation_px", "seam_point_tolerance_px", "source_arc_guard_px", "raster_depth_tolerance_px", "seam_short_gap_bridge_px"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and nonnegative")
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.seam_short_gap_bridge_px > 8.:
            raise ValueError("only source-arc short gaps up to 8px may bridge augmentation placement")
        if type(self.topology_connectivity) is not int or self.topology_connectivity not in (4, 8):
            raise ValueError("topology_connectivity must be exactly 4 or 8")
        if self.raster_depth_tolerance_px > 1.5:
            raise ValueError("raster tolerance cannot be used to hide reduced strong depth")
        if not np.isfinite(self.max_removed_fraction) or not 0 < self.max_removed_fraction < 1:
            raise ValueError("max_removed_fraction must be within (0,1)")
        object.__setattr__(self, "max_removed_fraction", float(self.max_removed_fraction))
        if type(self.min_remaining_area_px) is not int or self.min_remaining_area_px < 1:
            raise ValueError("min_remaining_area_px must be a positive integer")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 64:
            raise ValueError("max_attempts must be an integer within 1--64")


_FOUR = ndimage.generate_binary_structure(2, 1)
_EIGHT = np.ones((3, 3), dtype=bool)


def _signed_arc_distance(arc, center, perimeter):
    return (arc - center + perimeter / 2.) % perimeter - perimeter / 2.


def _seam_eligible(points, seam_points_rc, seam_arc_indices, tolerance, max_gap_px=0.):
    """Optional bounded source-arc placement bridge, never correspondence labels.

    Bridge distance is between original supported source vertices BEFORE the
    Euclidean raster allowance, not the shorter gap after that allowance.
    Long gaps, other branches and another fragment's coordinates are not joined.
    """
    n = len(points)
    if seam_points_rc is not None and seam_arc_indices is not None:
        raise ValueError("provide seam_points_rc OR seam_arc_indices, not both")
    if seam_arc_indices is not None:
        indices = np.asarray(seam_arc_indices)
        if indices.dtype == np.bool_:
            if indices.shape != (n,):
                raise ValueError("boolean seam_arc_indices must match the dense original outer contour")
            supported = indices.copy()
            valid = indices.copy()
        else:
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("seam_arc_indices must be 1-D integer indices into the dense original outer contour")
            if np.any(indices < 0) or np.any(indices >= n) or len(np.unique(indices)) != len(indices):
                raise ValueError("seam_arc_indices must be unique and in bounds")
            valid = np.zeros(n, bool)
            valid[indices] = True
            supported = valid.copy()
    elif seam_points_rc is None:
        return np.zeros(n, bool)
    else:
        seam = np.asarray(seam_points_rc, dtype=float)
        if seam.ndim != 2 or seam.shape[1:] != (2,) or not np.isfinite(seam).all():
            raise ValueError("seam_points_rc must be finite [N,2] in this fragment's original pixel frame")
        if not len(seam):
            return np.zeros(n, bool)
        distance, nearest = cKDTree(points).query(seam)
        if np.any(distance > tolerance + 1e-9):
            raise ValueError("seam points are not on the supplied mask's original outer contour")
        valid = cKDTree(seam).query(points)[0] <= tolerance
        supported = np.zeros(n, bool); supported[nearest] = True
    if max_gap_px > 0:
        step = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        arc = np.r_[0., np.cumsum(step[:-1])]; perimeter = float(step.sum())
        ids = np.flatnonzero(supported)
        if len(ids) >= 2:
            for first, last in zip(ids, np.roll(ids, -1)):
                length = float((arc[last] - arc[first]) % perimeter)
                if length <= max_gap_px + 1e-8:
                    cursor = int(first)
                    while cursor != last:
                        valid[cursor] = True
                        cursor = (cursor + 1) % n
                    valid[last] = True
    return valid


def _eligible_runs(eligible, arc, edge_lengths):
    edges = eligible & np.roll(eligible, -1)
    perimeter = float(edge_lengths.sum())
    if edges.all():
        return [(0., perimeter, True)]
    runs = []
    for start in np.flatnonzero(edges & ~np.roll(edges, 1)):
        cursor, length = int(start), 0.
        while edges[cursor]:
            length += float(edge_lengths[cursor])
            cursor = (cursor + 1) % len(edges)
        runs.append((float(arc[start]), length, False))
    return runs


def _place_bumps(widths, runs, perimeter, rng, minimum_separation):
    centers = np.full(len(widths), np.nan)
    for i in np.argsort(-np.asarray(widths), kind="stable"):
        width = widths[i]
        choices = [(s, length, circular) for s, length, circular in runs if length >= width]
        if not choices:
            return None
        weights = np.array([length if circular else max(length - width, 1e-9) for _, length, circular in choices])
        weights /= weights.sum()
        accepted = False
        for _ in range(64):
            start, length, circular = choices[int(rng.choice(len(choices), p=weights))]
            center = ((start + rng.uniform(0., length)) if circular else
                      (start + width / 2. + rng.uniform(0., max(0., length - width)))) % perimeter
            placed = np.flatnonzero(np.isfinite(centers))
            if any(abs(float(_signed_arc_distance(center, centers[j], perimeter))) <
                   (width + widths[j]) / 2. + minimum_separation for j in placed):
                continue
            centers[i] = center
            accepted = True
            break
        if not accepted:
            return None
    return centers


def _bump_profile(arc, center, width, depth, perimeter):
    relative = np.abs(_signed_arc_distance(arc, center, perimeter)) / (width / 2.)
    # A smooth central plateau reaches the requested depth; both endpoints are C1.
    taper = np.clip((relative - .6) / .4, 0., 1.)
    value = depth * .5 * (1. + np.cos(np.pi * taper))
    value[relative >= 1.] = 0.
    return value


def _wave_profile(arc, perimeter, rng, config):
    count = max(8, int(np.ceil(perimeter)))
    spacing = perimeter / count
    correlation = float(rng.uniform(*config.wave_correlation_px_range))
    noise = ndimage.gaussian_filter1d(rng.normal(size=count), sigma=correlation / spacing, mode="wrap")
    amplitude = float(np.ptp(noise))
    if amplitude <= 1e-12:
        return None, correlation
    scaled = (noise - float(noise.min())) / amplitude
    grid = np.arange(count + 1) * spacing
    profile = np.interp(arc, grid, np.r_[scaled, scaled[0]])
    # Normalize at actual source vertices, retaining the requested 10--30 range.
    variation = float(np.ptp(profile))
    if variation <= 1e-12:
        return None, correlation
    lo, hi = config.depth_range_px
    return lo + (hi - lo) * (profile - profile.min()) / variation, correlation


def strong_weather_fragment(mask, rng, *, mode="mixed", seam_points_rc=None,
                            seam_arc_indices=None, config=StrongWeatheringConfig(), return_depth=False):
    """Return (new_bool_mask, JSON-safe report), without writes or target creation.

    H/W are not resized; production's 800x800 frame uses exactly the same pixels.
    ``mixed`` randomly chooses ONE available mode (wave/local, plus seam_gaps
    when a seam is supplied); it does not add depths or hide gaps under a wave.
    seam_points_rc must densely mark eligible original boundary locations. Sparse
    point sets are not joined by default. Explicit seam_short_gap_bridge_px<=8
    can bridge only short original source-arc gaps for placement, not targets.
    Alternatively seam_arc_indices refer to
    extract_ordered_outer_contour(mask, cap=mask.size, smoothing_sigma=0.).

    Request peak depths, support lengths and K once. Retries only change spatial
    placement/waveform. Rejected narrow geometry returns the unchanged mask;
    no adaptive depth reduction, connected-component cropping, or hole filling.
    return_depth adds JSON lists to report, never changes the two-item return.
    """
    original = np.asarray(mask)
    if original.dtype != np.bool_ or original.ndim != 2 or min(original.shape, default=0) < 1:
        raise ValueError("mask must be a nonempty-shaped bool [H,W] array")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if not isinstance(config, StrongWeatheringConfig):
        raise TypeError("config must be StrongWeatheringConfig")
    if mode not in ("wave", "local", "seam_gaps", "mixed"):
        raise ValueError("unknown strong-weathering mode")
    if seam_points_rc is not None and seam_arc_indices is not None:
        raise ValueError("provide only one seam representation")
    original = original.copy()
    area = int(original.sum())
    original_n4 = int(ndimage.label(original, structure=_FOUR)[1])
    original_n8 = int(ndimage.label(original, structure=_EIGHT)[1])
    original_components = original_n4 if config.topology_connectivity == 4 else original_n8
    available = ["wave", "local"] + (["seam_gaps"] if seam_points_rc is not None or seam_arc_indices is not None else [])
    resolved = available[int(rng.integers(len(available)))] if mode == "mixed" else mode
    if resolved == "wave" and config.depth_range_px[0] == config.depth_range_px[1]:
        raise ValueError("wave needs a nonzero depth range; a uniform erode is not this augmentation")
    report = dict(schema_version="rachel-strong-weathering/1", mode=mode, resolved_mode=resolved,
        config=asdict(config), status="skipped", applied=False, skipped=True, skip_reason=None,
        frame_shape=list(original.shape), frame_unchanged=True, original_area_px=area,
        original_component_count=original_components, component_count=original_components,
        original_component_count_4=original_n4, original_component_count_8=original_n8,
        component_count_4=original_n4, component_count_8=original_n8,
        removed_area_px=0, removed_fraction=0., pixels_added=0, attempts=0, rejection_counts={},
        requested_depth_range_px=list(config.depth_range_px), requested_peak_depths_px=[],
        applied_max_depth_px=0., gap_k_requested=0, gap_k_applied=0,
        local_notch_count_requested=0, local_notch_count_applied=0, applied_regions=[], ignore_source_regions=[],
        correspondence_targets_generated=False, old_correspondence_targets_reusable=False,
        ancestry_policy="project_wave_descendants_to_original_source_arcs" if resolved == "wave" else "ignore_new_notch_descendants_on_covered_source_arcs",
        topology_connectivity=config.topology_connectivity, distance_convention="padded exterior-background EDT minus 0.5 pixel; holes do not seed erosion",
        strength_reduced_on_retry=False)

    def skip(reason):
        report["skip_reason"] = reason
        return original.copy(), report

    if original_components != 1:
        return skip("input_empty_or_not_single_component")
    if area <= config.min_remaining_area_px:
        return skip("input_too_small_to_retain_minimum_area")
    try:
        points, _ = extract_ordered_outer_contour(original, cap=original.size, smoothing_sigma=0.)
    except RachelPreprocessError:
        return skip("input_contour_too_short_or_invalid")
    points = np.asarray(points, dtype=float)
    edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    perimeter = float(edge_lengths.sum())
    arc = np.r_[0., np.cumsum(edge_lengths[:-1])]
    report.update(source_arc_origin_rc=points[0].tolist(), source_perimeter_px=perimeter,
                  source_contour_point_count=len(points), source_contour_smoothing_sigma=0.)
    strict_eligible = _seam_eligible(points, seam_points_rc, seam_arc_indices, config.seam_point_tolerance_px) if resolved == "seam_gaps" else np.ones(len(points), bool)
    eligible = (_seam_eligible(points, seam_points_rc, seam_arc_indices, config.seam_point_tolerance_px,
                config.seam_short_gap_bridge_px) if resolved == "seam_gaps" else strict_eligible)
    report.update(seam_bridged_vertex_count=int(np.count_nonzero(eligible & ~strict_eligible)),
                  seam_bridge_scope="augmentation placement only; never new correspondence supervision")
    runs = _eligible_runs(eligible, arc, edge_lengths)
    report["eligible_source_arc_length_px"] = float(sum(length for _, length, _ in runs))
    if resolved == "seam_gaps" and not runs:
        return skip("no_eligible_seam_arc")
    if resolved == "wave":
        widths, peaks = [], [float(config.depth_range_px[1])]
    else:
        k_range = config.gap_count_range if resolved == "seam_gaps" else config.local_notch_count_range
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        width_range = config.gap_support_px_range if resolved == "seam_gaps" else config.local_support_px_range
        widths = rng.uniform(*width_range, size=k).tolist()
        peaks = rng.uniform(*config.depth_range_px, size=k).tolist()
        report["gap_k_requested" if resolved == "seam_gaps" else "local_notch_count_requested"] = k
        report["requested_support_lengths_px"] = widths
    report["requested_peak_depths_px"] = peaks
    # Internal holes do not act as alternate erosion sources. Fill is only for
    # measuring distance; neither this fill nor any added pixel enters output.
    filled = ndimage.binary_fill_holes(np.pad(original, 1), structure=_EIGHT)
    inward = ndimage.distance_transform_edt(filled)[1:-1, 1:-1] - .5
    if float(inward[original].max()) < max(peaks) - config.raster_depth_tolerance_px:
        return skip("geometry_too_narrow_to_reach_requested_depth")
    band = np.argwhere(original & (inward <= config.depth_range_px[1]))
    nearest = cKDTree(points).query(band)[1]
    band_distance = inward[band[:, 0], band[:, 1]]
    for attempt in range(1, config.max_attempts + 1):
        report["attempts"] = attempt
        if resolved == "wave":
            depth, correlation = _wave_profile(arc, perimeter, rng, config)
            if depth is None:
                reason = "wave_profile_degenerate_on_short_contour"
                centers, profiles = None, []
            else:
                reason = None
                centers, profiles = None, [depth]
        else:
            centers = _place_bumps(widths, runs, perimeter, rng, config.minimum_arc_separation_px)
            reason = "requested_notches_do_not_fit_eligible_arcs" if centers is None else None
            profiles = [] if centers is None else [_bump_profile(arc, c, w, p, perimeter) for c, w, p in zip(centers, widths, peaks)]
            depth = None if not profiles else np.maximum.reduce(profiles)
        if reason is None:
            removed_selector = band_distance <= depth[nearest]
            removed_pixels = band[removed_selector]
            proposal = original.copy()
            proposal[removed_pixels[:, 0], removed_pixels[:, 1]] = False
            removed = int(len(removed_pixels))
            proposal_n4 = int(ndimage.label(proposal, structure=_FOUR)[1])
            proposal_n8 = int(ndimage.label(proposal, structure=_EIGHT)[1])
            components = proposal_n4 if config.topology_connectivity == 4 else proposal_n8
            actual_peaks = []
            for profile in profiles:
                hit = removed_selector & (profile[nearest] > 0.)
                actual_peaks.append(float(band_distance[hit].max()) if hit.any() else 0.)
            report.update(last_proposed_removed_area_px=removed, last_proposed_component_count=components,
                          last_proposed_component_count_4=proposal_n4, last_proposed_component_count_8=proposal_n8,
                          last_proposed_peak_depths_px=actual_peaks)
            if components != 1:
                reason = "proposal_empty_or_disconnected"
            elif area - removed < config.min_remaining_area_px or removed / area > config.max_removed_fraction:
                reason = "proposal_retained_area_limit"
            elif any(actual < wanted - config.raster_depth_tolerance_px for actual, wanted in zip(actual_peaks, peaks)):
                reason = "proposal_did_not_reach_requested_depth"
            elif np.any(ndimage.binary_fill_holes(proposal, structure=_EIGHT) & ~proposal & original):
                reason = "proposal_created_enclosed_void"
            if reason is None:
                regions = []
                if centers is not None:
                    for center, width, peak, actual, profile in zip(centers, widths, peaks, actual_peaks, profiles):
                        covered = np.abs(_signed_arc_distance(arc, center, perimeter)) <= width / 2. + config.source_arc_guard_px
                        region_pixels = removed_selector & (profile[nearest] > 0.)
                        start = float((center - width / 2.) % perimeter)
                        regions.append(dict(center_arc_px=float(center), support_start_arc_px=start,
                            support_length_px=float(width), support_wraps=bool(start + width > perimeter),
                            requested_peak_depth_px=float(peak), applied_max_depth_px=actual,
                            removed_area_px=int(region_pixels.sum()), source_arc_guard_px=config.source_arc_guard_px,
                            ignore_start_arc_px=float((center - width / 2. - config.source_arc_guard_px) % perimeter),
                            ignore_length_px=float(min(perimeter, width + 2. * config.source_arc_guard_px)),
                            source_point_indices=np.flatnonzero(covered).tolist(),
                            source_points_rc=points[covered].tolist(), ancestry="ignore_new_notch_descendants"))
                report.update(status="applied", applied=True, skipped=False, skip_reason=None,
                    removed_area_px=removed, removed_fraction=removed / area, component_count=components,
                    component_count_4=proposal_n4, component_count_8=proposal_n8,
                    applied_max_depth_px=max(actual_peaks), applied_peak_depths_px=actual_peaks,
                    gap_k_applied=len(regions) if resolved == "seam_gaps" else 0,
                    local_notch_count_applied=len(regions) if resolved == "local" else 0,
                    applied_regions=regions, ignore_source_regions=regions,
                    actual_sampled_depth_min_px=float(depth.min()), actual_sampled_depth_max_px=float(depth.max()))
                if resolved == "wave":
                    report["wave_correlation_px"] = correlation
                if return_depth:
                    report["depth_diagnostics"] = dict(contour_points_rc=points.tolist(), arclength_px=arc.tolist(), depth_px=depth.tolist())
                return proposal, report
        report["rejection_counts"][reason] = report["rejection_counts"].get(reason, 0) + 1
    return skip("requested_geometry_rejected_after_bounded_retries")


__all__ = ["StrongWeatheringConfig", "strong_weather_fragment"]

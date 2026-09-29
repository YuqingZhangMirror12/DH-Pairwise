"""Conservative, training-only pixel-deletion proposals for size augmentation.

This primitive does not choose crops, rebalance labels, or enforce a desired
area ratio. The caller must restrict it to training. No crop is recentered,
resized or rotated: both translation conventions stay in the original frames.

Positive supervision is a subset of *original* dense MNN correspondences,
supported at BOTH endpoints by the same original reciprocal GT token edge.
After cropping, only a surviving continuous original seam can supervise new
tokens; new cut edges never create targets. These conservative geometric
constants are an unvalidated prototype, not a new global preprocessing policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from numbers import Real
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from .rachel_preprocess import (
    RachelPreprocessError,
    _accepted_continuous_seam,
    _dense_external_contour,
    extract_ordered_outer_contour,
    recover_mutual_contour_correspondences,
)
from .rachel_training_dataset import RachelPairSample, _readonly

SCHEMA_VERSION = "rachel-training-original-seam-size-crop/1"
_CONNECTIVITY = ndimage.generate_binary_structure(2, 1)


@dataclass(frozen=True)
class SizeCropConfig:
    contour_cap: int = 512
    min_retained_seam_length_px: float = 64.0
    min_matched_tokens: int = 8
    max_dense_to_token_distance_px: float = 3.0
    cut_edge_exclusion_px: float = 8.0
    original_gt_support_distance_px: float = 6.0
    dense_match_distance_px: float = 3.0
    max_token_residual_px: float = 3.0
    smoothing_sigma: float = 3.0

    def __post_init__(self):
        for name, minimum in (("contour_cap", 4), ("min_matched_tokens", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(name + " must be an integer >= " + str(minimum))
        if self.min_matched_tokens > self.contour_cap:
            raise ValueError("min_matched_tokens must not exceed contour_cap")
        for name in ("min_retained_seam_length_px", "max_dense_to_token_distance_px",
                     "cut_edge_exclusion_px", "original_gt_support_distance_px",
                     "dense_match_distance_px", "max_token_residual_px", "smoothing_sigma"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(name + " must be finite and positive")
            if not np.isfinite(value) or float(value) <= 0:
                raise ValueError(name + " must be finite and positive")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class SizeCropResult:
    accepted: bool
    sample: RachelPairSample
    reason: str
    diagnostics: Dict[str, object]


def _mask_2d(value):
    mask = np.asarray(value)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2 or not np.isfinite(mask).all() or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must be binary [H,W] or [1,H,W]")
    return np.asarray(mask, dtype=np.bool_)


def _updated_coarse(mask, original_coarse):
    """Mirror _load_mask's PIL nearest resize, retaining its existing shape.

    The loader's only coarse utility also opens a file; keep its tiny in-memory
    operation here instead of writing/encoding an image just to reopen it.
    This resize is only the cached coarse view, never the physical mask.
    """
    old = np.asarray(original_coarse)
    if old.ndim not in (2, 3) or (old.ndim == 3 and old.shape[0] != 1):
        raise ValueError("coarse mask must have shape [H,W] or [1,H,W]")
    h, w = old.shape[-2:]
    if min(h, w) <= 0:
        raise ValueError("coarse mask dimensions must be positive")
    nearest = getattr(Image, "Resampling", Image).NEAREST
    resized = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        (w, h), resample=nearest)) == 255
    return _readonly(resized[None] if old.ndim == 3 else resized, old.dtype)


def _original_gt_edges(sample):
    a, b = np.asarray(sample.points_rc_a), np.asarray(sample.points_rc_b)
    va, vb = np.asarray(sample.contour_valid_a), np.asarray(sample.contour_valid_b)
    ta, tb = np.asarray(sample.target_a), np.asarray(sample.target_b)
    for points, valid, targets in ((a, va, ta), (b, vb, tb)):
        if (points.ndim != 2 or points.shape[1:] != (2,) or valid.shape != (len(points),)
                or valid.dtype != np.bool_ or targets.shape != (len(points),)
                or not np.issubdtype(targets.dtype, np.integer)
                or not np.isfinite(points[valid]).all()):
            raise ValueError("original points, valid masks and targets have incompatible shapes or values")
    ia = np.flatnonzero(va & (ta >= 0) & (ta < len(b)))
    ib = ta[ia]
    keep = vb[ib] & (tb[ib] == ia)
    return np.column_stack((ia[keep], ib[keep])).astype(np.int64, copy=False)


def _gt_supported_dense(matches, dense_a, dense_b, sample, gt_edges, radius):
    """Require ONE original GT edge supporting both dense endpoints."""
    if not len(matches) or not len(gt_edges):
        return matches[:0]
    anchors_a = np.asarray(sample.points_rc_a, dtype=np.float64)[gt_edges[:, 0]]
    anchors_b = np.asarray(sample.points_rc_b, dtype=np.float64)[gt_edges[:, 1]]
    neighbors = cKDTree(anchors_a).query_ball_point(dense_a[matches[:, 0]], r=radius)
    keep = np.zeros(len(matches), dtype=np.bool_)
    for index, candidates in enumerate(neighbors):
        if candidates:
            delta = anchors_b[candidates] - dense_b[matches[index, 1]]
            keep[index] = bool(np.any(np.linalg.norm(delta, axis=1) <= radius))
    return matches[keep]


def _continuous_runs(matches, size_a, size_b):
    """Exact original index-adjacent runs, allowing either direction and wrap."""
    pairs = {tuple(map(int, pair)) for pair in matches}
    seen_runs = set()
    runs = []
    for direction in (-1, 1):
        remaining = set(pairs)
        starts = sorted(pair for pair in pairs
                        if ((pair[0] - 1) % size_a, (pair[1] - direction) % size_b) not in pairs)
        # The second pass also handles a completely closed cyclic run.
        for start in starts + sorted(pairs):
            if start not in remaining:
                continue
            run = []
            current = start
            while current in remaining:
                remaining.remove(current)
                run.append(current)
                current = ((current[0] + 1) % size_a, (current[1] + direction) % size_b)
            key = frozenset(run)
            if len(run) >= 4 and key not in seen_runs:
                seen_runs.add(key)
                runs.append(np.asarray(run, dtype=np.int64))
    return runs


def _run_lengths(run, dense_a, dense_b):
    # Count actual consecutive segments, not a chord over missing seam points
    # or a sum that joins separate retained sections.
    return tuple(float(np.linalg.norm(np.diff(dense[run[:, axis]], axis=0), axis=1).sum())
                 for axis, dense in enumerate((dense_a, dense_b)))


def _mapped_targets(points_a, points_b, dense_a, dense_b, run, translation,
                    cut_tree, side, config):
    da, ia = cKDTree(points_a).query(dense_a[run[:, 0]], k=1)
    db, ib = cKDTree(points_b).query(dense_b[run[:, 1]], k=1)
    residual = np.linalg.norm(points_a[ia] + translation - points_b[ib], axis=1)
    cut_points = points_a[ia] if side == "a" else points_b[ib]
    cut_distance = cut_tree.query(cut_points, k=1)[0]
    good = ((da <= config.max_dense_to_token_distance_px)
            & (db <= config.max_dense_to_token_distance_px)
            & (residual <= config.max_token_residual_px)
            & (cut_distance > config.cut_edge_exclusion_px))
    # As in the release, order by dense-to-token error and select one-to-one.
    order = sorted(np.flatnonzero(good), key=lambda k: (float(da[k] + db[k]), int(ia[k]), int(ib[k]), int(k)))
    target_a, target_b = np.full(len(points_a), -1, np.int64), np.full(len(points_b), -1, np.int64)
    selected = []
    for k in order:
        i, j = int(ia[k]), int(ib[k])
        if target_a[i] < 0 and target_b[j] < 0:
            target_a[i], target_b[j] = j, i
            selected.append(k)
    diagnostics = {"token_mapping_proposal_count": int(np.count_nonzero(good)),
                   "token_match_count": len(selected),
                   "max_selected_token_residual_px": None,
                   "max_selected_dense_to_token_distance_px": None,
                   "min_selected_token_cut_distance_px": None}
    if selected:
        diagnostics.update(max_selected_token_residual_px=float(np.max(residual[selected])),
            max_selected_dense_to_token_distance_px=float(np.max(np.maximum(da[selected], db[selected]))),
            min_selected_token_cut_distance_px=float(np.min(cut_distance[selected])))
    return _readonly(target_a, np.int64), _readonly(target_b, np.int64), diagnostics


def crop_training_pair(sample: RachelPairSample, *, side: str,
                       bounds_rc: Tuple[int, int, int, int],
                       config: SizeCropConfig = SizeCropConfig()) -> SizeCropResult:
    """Keep a half-open rectangle on ONE original canvas, or return sample.

    Rejection/no-op always returns the exact original sample, without modifying
    arrays or their writeability. Accepted outputs resample both contours, but
    only the cropped mask/coarse view changes. Fragment/pair identifiers and
    both translation fields are retained; cache invalidation belongs to the
    wrapper. Shape/config misuse raises ValueError, geometric rejection does not.
    """
    if not isinstance(sample, RachelPairSample):
        raise TypeError("sample must be RachelPairSample")
    if not isinstance(config, SizeCropConfig):
        raise TypeError("config must be SizeCropConfig")
    if side not in ("a", "b"):
        raise ValueError("side must be 'a' or 'b'")
    try:
        bounds_rc = tuple(bounds_rc)
    except TypeError as error:
        raise ValueError("bounds_rc must contain four integer pixel bounds") from error
    if len(bounds_rc) != 4 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in bounds_rc):
        raise ValueError("bounds_rc must contain four integer pixel bounds")
    bounds = tuple(map(int, bounds_rc))
    r0, r1, c0, c1 = bounds
    ma, mb = _mask_2d(sample.mask_a), _mask_2d(sample.mask_b)
    original = ma if side == "a" else mb
    if not (0 <= r0 < r1 <= original.shape[0] and 0 <= c0 < c1 <= original.shape[1]):
        raise ValueError("bounds_rc must be a nonempty half-open rectangle inside the original canvas")
    crop = np.zeros_like(original)
    crop[r0:r1, c0:c1] = original[r0:r1, c0:c1]
    new_a, new_b = (crop, mb) if side == "a" else (ma, crop)
    area_a, area_b = int(ma.sum()), int(mb.sum())
    post_a, post_b = int(new_a.sum()), int(new_b.sum())
    diagnostics = {"schema_version": SCHEMA_VERSION, "training_gt_used": bool(sample.label),
        "side": side, "bounds_rc": list(bounds), "config": asdict(config),
        "original_area_a_px": area_a, "original_area_b_px": area_b,
        "post_area_a_px": post_a, "post_area_b_px": post_b,
        "post_area_ratio": min(post_a, post_b) / max(post_a, post_b) if min(post_a, post_b) > 0 else None,
        "deleted_area_px": int(original.sum() - crop.sum()), "connectivity": 4,
        "original_gt_edge_count": 0, "retained_dense_match_count": 0,
        "retained_contiguous_seam_length_px": 0.0, "token_match_count": 0}

    def reject(reason):
        return SizeCropResult(False, sample, reason, diagnostics)

    if not diagnostics["deleted_area_px"]:
        return reject("no_op_crop")
    if not post_a or not post_b:
        return reject("empty_fragment")
    if any(ndimage.label(mask, structure=_CONNECTIVITY)[1] != 1 for mask in (ma, mb)):
        return reject("original_not_single_component")
    component_count = ndimage.label(crop, structure=_CONNECTIVITY)[1]
    diagnostics["post_component_count"] = int(component_count)
    if component_count != 1:
        return reject("disconnected_crop")
    if float(sample.label) not in (0., 1.):
        return reject("invalid_pair_label")
    if bool(sample.label) != bool(sample.translation_valid):
        return reject("invalid_translation_valid_flag")
    try:
        points_a, valid_a = extract_ordered_outer_contour(new_a, cap=config.contour_cap,
                                                        smoothing_sigma=config.smoothing_sigma)
        points_b, valid_b = extract_ordered_outer_contour(new_b, cap=config.contour_cap,
                                                        smoothing_sigma=config.smoothing_sigma)
    except RachelPreprocessError as error:
        diagnostics["contour_error"] = str(error)
        return reject("invalid_cropped_contour")
    target_a, target_b = np.full(len(points_a), -1, np.int64), np.full(len(points_b), -1, np.int64)
    if bool(sample.label):
        translation = np.asarray(sample.translation_a_to_b_rc, dtype=np.float64)
        xy = np.asarray(sample.translation_a_to_b_xy_cartesian, dtype=np.float64)
        if (translation.shape != (2,) or xy.shape != (2,) or not np.isfinite(translation).all()
                or not np.isfinite(xy).all()
                or not np.allclose(xy, (translation[1], -translation[0]), rtol=0, atol=1e-4)):
            return reject("invalid_original_translation")
        try:
            gt_edges = _original_gt_edges(sample)
        except ValueError as error:
            diagnostics["target_error"] = str(error)
            return reject("invalid_original_targets")
        diagnostics["original_gt_edge_count"] = len(gt_edges)
        if not len(gt_edges):
            return reject("no_original_gt_edges")
        try:
            dense_a, dense_b = _dense_external_contour(ma), _dense_external_contour(mb)
        except RachelPreprocessError as error:
            diagnostics["contour_error"] = str(error)
            return reject("invalid_original_contour")
        raw = recover_mutual_contour_correspondences(dense_a + translation, dense_b,
                                                      max_distance_px=config.dense_match_distance_px)
        original_seam = _accepted_continuous_seam(raw, dense_a + translation, dense_b)
        supported = _gt_supported_dense(original_seam, dense_a, dense_b, sample, gt_edges,
                                       config.original_gt_support_distance_px)
        diagnostics.update(original_dense_match_count=len(raw),
            original_dense_continuous_match_count=len(original_seam),
            original_gt_supported_dense_match_count=len(supported))
        if not len(supported):
            return reject("no_gt_supported_original_seam")
        # Only edges across deleted foreground are artificial. Existing natural
        # boundaries coinciding with a rectangle side are not automatically cuts.
        deleted = original & ~crop
        cut_pixels = np.argwhere(crop & ndimage.binary_dilation(deleted, structure=_CONNECTIVITY))
        diagnostics["artificial_cut_edge_pixel_count"] = len(cut_pixels)
        if not len(cut_pixels):
            return reject("no_supported_cut_boundary")
        cut_tree = cKDTree(cut_pixels)
        dense_side = dense_a[supported[:, 0]] if side == "a" else dense_b[supported[:, 1]]
        rc = dense_side.astype(np.int64)
        survives = crop[rc[:, 0], rc[:, 1]]
        away = cut_tree.query(dense_side, k=1)[0] > config.cut_edge_exclusion_px
        retained = supported[survives & away]
        runs = _continuous_runs(retained, len(dense_a), len(dense_b))
        diagnostics.update(retained_dense_match_count=len(retained), retained_contiguous_run_count=len(runs))
        if not runs:
            return reject("no_retained_contiguous_original_seam")
        # Supervise only the longest retained original run. Do not join separate
        # sections to clear the length threshold, or silently add a second seam.
        run = max(runs, key=lambda item: (min(_run_lengths(item, dense_a, dense_b)), len(item)))
        lengths = _run_lengths(run, dense_a, dense_b)
        diagnostics.update(retained_contiguous_seam_length_px=min(lengths),
            retained_contiguous_seam_length_a_px=lengths[0], retained_contiguous_seam_length_b_px=lengths[1],
            retained_contiguous_seam_pair_count=len(run))
        if min(lengths) < config.min_retained_seam_length_px:
            return reject("retained_seam_too_short")
        target_a, target_b, mapped = _mapped_targets(np.asarray(points_a, dtype=np.float64),
            np.asarray(points_b, dtype=np.float64), dense_a, dense_b, run, translation, cut_tree, side, config)
        diagnostics.update(mapped)
        if diagnostics["token_match_count"] < config.min_matched_tokens:
            return reject("too_few_retained_token_matches")
    raw_crop = np.array(getattr(sample, "mask_" + side), copy=True)
    raw_crop[...] = crop[None] if raw_crop.ndim == 3 else crop
    updates = {"mask_" + side: _readonly(raw_crop, raw_crop.dtype),
               "coarse_mask_" + side: _updated_coarse(crop, getattr(sample, "coarse_mask_" + side)),
               "points_rc_a": points_a, "points_rc_b": points_b,
               "contour_valid_a": valid_a, "contour_valid_b": valid_b,
               "target_a": _readonly(target_a, np.int64), "target_b": _readonly(target_b, np.int64)}
    diagnostics.update(valid_token_count_a=int(valid_a.sum()), valid_token_count_b=int(valid_b.sum()))
    return SizeCropResult(True, replace(sample, **updates), "accepted", diagnostics)


__all__ = ["SizeCropConfig", "SizeCropResult", "crop_training_pair", "SCHEMA_VERSION"]

"""S7 strong TRAIN damage with inherited, never nearest-neighbor pair labels."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .rachel_preprocess import extract_ordered_outer_contour
from .rachel_training_dataset import _readonly
from .rachel_weathered_dataset import (
    _FragmentView, _identity_view, inherit_pair_targets, source_arc_ancestry)
from .rachel_staged_damage_dataset import clean_report

SCHEMA = "rachel-s7-strong-source-arc/1"
RECIPE_PERCENT = dict(reference_e1=30, wave=15, local=10, seam_gaps=15,
                      partial_curve=20, gen5_partition=10)


def stable_rng(seed, *parts):
    token = json.dumps([SCHEMA, seed, parts], sort_keys=True, separators=(",", ":"))
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:16], "big"))


def recipe_schedule(group_count, seed=260915, *, protected_groups=()):
    """Paired positive/negative slots; preserve original tiny-union sources."""
    if group_count < 100 or group_count % 100:
        raise ValueError("paired group count must be a positive multiple of100")
    order = stable_rng(seed, "recipe-order").permutation(group_count).tolist()
    protected = set(protected_groups)
    gen5_count = group_count * RECIPE_PERCENT["gen5_partition"] // 100
    gen5 = [i for i in order if i not in protected][:gen5_count]
    if len(gen5) != gen5_count:
        raise ValueError("insufficient unprotected slots for Gen5 replacement")
    result = [None] * group_count
    for i in gen5:
        result[i] = "gen5_partition"
    remaining = [i for i in order if result[i] is None]
    offset = 0
    for name, percent in RECIPE_PERCENT.items():
        if name == "gen5_partition":
            continue
        count = group_count * percent // 100
        for i in remaining[offset:offset + count]:
            result[i] = name
        offset += count
    assert offset == len(remaining) and all(result)
    return tuple(result)


def changed_report(original, sample, recipe, *, detail=None, fallback_reason=None):
    report = clean_report(sample, epoch=1)
    report.update(schema_version=SCHEMA, fallback_reason=fallback_reason,
                  inheritance_rule="source-arc inherited GT; artificial-notch descendants ignored",
                  s7=dict(recipe=recipe, detail=detail or {}))
    for side in "ab":
        old, new = (np.asarray(getattr(s, "mask_" + side), bool) for s in (original, sample))
        changed = not np.array_equal(old, new)
        removed = int(np.count_nonzero(old & ~new))
        report["changed_" + side] = changed
        report["side_" + side].update(tier=recipe, applied=changed, attempted_applied=changed,
            effective_applied=changed, original_area_px=int(old.sum()), retained_area_px=int(new.sum()),
            removed_area_px=removed, effective_removed_area_px=removed,
            removed_fraction=removed / max(1, int(old.sum())))
    report["changed_pair"] = report["changed_a"] or report["changed_b"]
    # Same existing E1 loss rule: damaged boundaries do not teach their biased
    # all-correspondence displacement as if it were a clean rigid observation.
    report["pose_supervision_enabled"] = bool(sample.label) and not report["changed_pair"]
    return report


def _seam_points(sample, side):
    if not sample.label:
        return None
    from .rachel_partial_seam_dataset import source_seam_context, PartialSeamConfig
    context = source_seam_context(sample, side, PartialSeamConfig().crop_config)
    return context["points"] if context else np.empty((0, 2))


def _changed_view(sample, side, mask, geometry, recipe):
    old = np.asarray(getattr(sample, "mask_" + side)[0], bool)
    points, valid = getattr(sample, "points_rc_" + side), getattr(sample, "contour_valid_" + side)
    if np.array_equal(mask, old):
        return _identity_view(old, points, valid, recipe, geometry)
    new_points, new_valid = extract_ordered_outer_contour(mask, cap=512, smoothing_sigma=3.)
    ancestry, representatives, projection = source_arc_ancestry(old, points, valid, new_points, 30.)
    # A notch is a newly exposed cut, not a surviving complementary seam.
    # Wave recession remains source-arc supervision, as with historical E1.
    excluded = [point for region in geometry.get("ignore_source_regions", [])
                for point in region.get("source_points_rc", [])]
    if recipe in ("local", "seam_gaps") and not excluded:
        raise ValueError("changed strong notch geometry must identify ignored original source arcs")
    if recipe in ("local", "seam_gaps"):
        excluded = np.asarray(excluded, dtype=float)
        forbidden_old = cKDTree(excluded).query(points)[0] <= 8.
        representatives[forbidden_old] = -1
        ancestry[(ancestry >= 0) & forbidden_old[np.maximum(ancestry, 0)]] = -1
    return _FragmentView(_readonly(mask, bool), new_points, new_valid,
        _readonly(ancestry, np.int64), _readonly(representatives, np.int64),
        True, recipe, dict(geometry, source_arc_inheritance=projection))


def strong_pair(sample, mode, rng, *, endpoints=None, geometry_function=None):
    """Only augmentation geometry may consult TRAIN seam GT; inputs stay clean.

    Negative gap morphology is placed on its own contour, with the same depth,
    K and length generator; no negative correspondence/translation is invented.
    The caller couples acceptance of positive and negative members.
    """
    if mode not in ("wave", "local", "seam_gaps"):
        raise ValueError("unknown strong recipe")
    if geometry_function is None:
        from functools import partial
        from .rachel_strong_weathering import strong_weather_fragment, StrongWeatheringConfig
        geometry_function = partial(strong_weather_fragment, config=StrongWeatheringConfig(
            topology_connectivity=8, seam_short_gap_bridge_px=8.))
    if endpoints is None:
        endpoints = ("ab", "a", "b")[int(rng.integers(3))]
    views, details = {}, {}
    for side in "ab":
        mask = np.asarray(getattr(sample, "mask_" + side)[0], bool)
        if side in endpoints:
            if mode == "seam_gaps":
                # Negatives have no true seam. Their own complete source contour
                # supplies candidate notch locations, never positive match labels.
                seam = (_seam_points(sample, side) if sample.label else
                        extract_ordered_outer_contour(mask, cap=mask.size, smoothing_sigma=0.)[0])
            else:
                seam = None
            if seam is not None and not len(seam):
                return sample, changed_report(sample, sample, mode,
                    fallback_reason="no_original_seam_for_gap_placement")
            changed, geometry = geometry_function(mask, rng, mode=mode, seam_points_rc=seam)
        else:
            changed, geometry = mask, dict(applied=False, reason="unselected_endpoint")
        views[side] = _changed_view(sample, side, changed, geometry, mode)
        details[side] = views[side].report
    a, b = inherit_pair_targets(sample, views["a"], views["b"])
    if sample.label and int((a >= 0).sum()) < 4:
        return sample, changed_report(sample, sample, mode, detail=details,
            fallback_reason="fewer_than4_reliable_inherited_correspondences")
    updates = dict(target_a=_readonly(a, np.int64), target_b=_readonly(b, np.int64))
    for side, view in views.items():
        nearest = getattr(Image, "Resampling", Image).NEAREST
        coarse = np.asarray(Image.fromarray(view.mask.astype(np.uint8)*255).resize((128,128), nearest)) > 0
        updates.update({"mask_"+side:_readonly(view.mask[None], np.float32),
            "coarse_mask_"+side:_readonly(coarse[None], np.float32),
            "points_rc_"+side:view.points, "contour_valid_"+side:view.valid})
    result = replace(sample, **updates)
    report = changed_report(sample, result, mode, detail=details)
    for side in "ab":
        report["side_"+side].update(details[side], tier=mode,
            effective_applied=report["changed_"+side], effective_removed_area_px=
            report["side_"+side]["removed_area_px"] if report["changed_"+side] else 0)
    return result, report


def strong_group(positive, negative, mode, seed=260915):
    if not positive.label or negative.label:
        raise ValueError("requires one true pair and one negative")
    ids = (positive.pair_id, negative.pair_id)
    endpoints = ("ab", "a", "b")[int(stable_rng(seed, ids, mode, "endpoints").integers(3))]
    rows = [strong_pair(s, mode, stable_rng(seed, ids, mode, i), endpoints=endpoints)
            for i, s in enumerate((positive, negative))]
    if not all(report["changed_pair"] for _, report in rows):
        return tuple((s, changed_report(s, s, mode,
            detail={"attempted": report["s7"]["detail"]},
            fallback_reason="coupled_geometry_or_supervision_rejection"))
            for s, (_, report) in zip((positive, negative), rows))
    return tuple(rows)


__all__ = ["SCHEMA", "RECIPE_PERCENT", "stable_rng", "recipe_schedule", "changed_report", "strong_pair", "strong_group"]

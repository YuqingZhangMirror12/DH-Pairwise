"""TRAIN-source curves positioned by the original positive seam, not bbox luck.

Only the cut offset is solved from TRAIN GT. Original labels, translations,
source-arc correspondence rules, topology checks and the 8px new-edge ignore
remain unchanged. A native negative receives the same donor/oblique direction,
but an independently solved offset matching positive material retention. Both
members must pass before either is augmented. Equal application rates do not
prove that all remaining shape distributions are free of label shortcuts.

This wrapper is intended for the weathering stage; probability=1 requests a
curve in that stage, not in the separate clean training stage. Actual coverage
must be measured, including jointly rejected proposals.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from .rachel_curve_cut import OutlineBank
from .rachel_partial_seam_dataset import (
    PartialSeamConfig, PartialSeamDataset, _ignore_ambiguous_nonmatches,
    _mask_2d, crop_training_pair, source_seam_context,
)
from .rachel_preprocess import RachelPreprocessError


SCHEMA_VERSION = "rachel-guided-partial-source-seam/2"


def _curve_field(mask, profile, angle_deg, flip, reverse):
    mask = np.asarray(mask, dtype=bool)
    profile = np.asarray(profile, dtype=float)
    angle = float(angle_deg) % 180.
    if mask.ndim != 2 or not mask.any():
        raise ValueError("nonempty 2D mask required")
    if not np.isfinite(angle) or min(angle % 90., 90. - angle % 90.) < 15. - 1e-6:
        raise ValueError("cut direction must be oblique, at least 15 degrees from either axis")
    if profile.shape != (129,) or not np.isfinite(profile).all():
        raise ValueError("129 finite TRAIN contour-profile samples required")
    if reverse:
        profile = profile[::-1]
    if flip:
        profile = -profile
    pixels = np.argwhere(mask)
    lo, hi = pixels.min(0), pixels.max(0)
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]],
                        [hi[0], lo[1]], [hi[0], hi[1]]], dtype=float)
    theta = np.deg2rad(angle)
    tangent = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-tangent[1], tangent[0]])
    t_range, n_range = corners @ tangent, corners @ normal
    span = max(1., float(np.ptp(t_range)))
    grid = np.linspace(0., 1., 129)

    def scores(points):
        points = np.asarray(points, dtype=float)
        t = points[:, 0] * tangent[0] + points[:, 1] * tangent[1]
        n = points[:, 0] * normal[0] + points[:, 1] * normal[1]
        return n - np.interp((t - t_range.min()) / span, grid, profile) * span

    frame = dict(angle_deg=angle, tangent_span_px=span,
                 normal_min_px=float(n_range.min()), normal_span_px=float(np.ptp(n_range)))
    return mask, pixels, scores, frame


def _weighted_offset(values, weights, target_retention, keep_low):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if (values.ndim != 1 or weights.shape != values.shape or len(values) < 2
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not 0 < target_retention < 1):
        raise ValueError("finite source samples, positive weights and a partial target required")
    order = np.argsort(values, kind="stable")
    distinct, starts = np.unique(values[order], return_index=True)
    if len(distinct) < 2:
        raise ValueError("curve is tangent to all source samples; no partial quantile")
    cumulative = np.cumsum(np.add.reduceat(weights[order], starts))[:-1] / weights.sum()
    fractions = cumulative if keep_low else 1. - cumulative
    index = int(np.argmin(np.abs(fractions - target_retention)))
    # Place the boundary between unequal scores: tied raster samples stay whole.
    return float(.5 * (distinct[index] + distinct[index + 1]))


def _mask_at_offset(mask, pixels, scores, offset, keep_low):
    values = scores(pixels)
    survive = values <= offset if keep_low else values >= offset
    retained = np.zeros_like(mask)
    retained[pixels[survive, 0], pixels[survive, 1]] = True
    return retained


def _offset_report(frame, offset, target, actual, material):
    return dict(offset_px=float(offset),
        bbox_fraction=float((offset - frame["normal_min_px"]) / frame["normal_span_px"]),
        target_retention=float(target), actual_retention=float(actual),
        material_retention=float(material), **frame)


def guided_curve_mask(mask, profile, *, points, weights, target_retention,
                      angle_deg, keep_low, flip=False, reverse=False):
    """Solve a curved cut's normal offset using weighted GT seam quantiles.

    The old bbox-fraction 0.15..0.85 restriction is deliberately not imposed:
    a genuine source seam can lie near a bbox corner. Physical seam retention,
    material deletion and all downstream supervision checks still apply.
    """
    mask, pixels, scores, frame = _curve_field(mask, profile, angle_deg, flip, reverse)
    points, weights = np.asarray(points, float), np.asarray(weights, float)
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("finite source seam points required")
    source_pixels = np.rint(points).astype(np.int64)
    if (np.any(source_pixels < 0) or np.any(source_pixels >= np.array(mask.shape))
            or not mask[source_pixels[:, 0], source_pixels[:, 1]].all()):
        raise ValueError("source seam points must lie on original material")
    offset = _weighted_offset(scores(source_pixels), weights, target_retention, keep_low)
    retained = _mask_at_offset(mask, pixels, scores, offset, keep_low)
    actual = float(weights[retained[source_pixels[:, 0], source_pixels[:, 1]]].sum() / weights.sum())
    info = _offset_report(frame, offset, target_retention, actual, retained.sum() / len(pixels))
    info["offset_policy"] = "original_source_seam_weighted_quantile"
    return retained, info


def material_matched_curve_mask(mask, profile, *, target_retention, angle_deg,
                                keep_low, flip=False, reverse=False):
    """Independently position a negative's curve, matching retained foreground.

    This uses only its own silhouette and the paired positive's retained area
    fraction. No correspondence or valid translation is invented for negatives.
    """
    mask, pixels, scores, frame = _curve_field(mask, profile, angle_deg, flip, reverse)
    if len(pixels) < 2 or not np.isfinite(target_retention) or not 0 < target_retention < 1:
        raise ValueError("nontrivial original material and partial retention required")
    values = scores(pixels)
    low_fraction = target_retention if keep_low else 1. - target_retention
    count = max(1, min(len(values) - 1, int(round(low_fraction * len(values)))))
    boundary = np.partition(values, (count - 1, count))
    offset = float(.5 * (boundary[count - 1] + boundary[count]))
    survive = values <= offset if keep_low else values >= offset
    retained = np.zeros_like(mask)
    retained[pixels[survive, 0], pixels[survive, 1]] = True
    actual = float(survive.mean())
    info = _offset_report(frame, offset, target_retention, actual, actual)
    info["offset_policy"] = "independent_foreground_quantile_matching_positive_material_retention"
    return retained, info


class GuidedPartialDataset(PartialSeamDataset):
    def __init__(self, base, *, pair_metadata, bank, seed=260910, epoch=0,
                 config=PartialSeamConfig(probability=1., max_attempts=12)):
        if (config.crop_config.cut_edge_exclusion_px != 8.
                or config.crop_config.min_retained_seam_length_px < 32.
                or config.crop_config.min_matched_tokens < 4):
            raise ValueError("guided recipe keeps 8px ignore, at least 32px seam and 4 original matches")
        self.bank = bank if isinstance(bank, OutlineBank) else OutlineBank(bank)
        super().__init__(base, pair_metadata=pair_metadata, seed=seed, epoch=epoch, config=config)

    def proposal(self, ids, attempt):
        rng = self._rng(ids, "guided-curve-proposal-v1", attempt)
        return dict(donor_index=int(rng.integers(len(self.bank.profiles))),
            angle_deg=float(rng.uniform(15., 75.) + 90 * int(rng.integers(2))),
            target_retention=float(rng.uniform(self.config.retention_min, self.config.retention_max)),
            keep_low=bool(rng.integers(2)), flip=bool(rng.integers(2)), reverse=bool(rng.integers(2)))

    def _materialize(self, group_index):
        indices = self._groups[group_index]
        ids = tuple(self.pair_metadata[i][0] for i in indices)
        originals = tuple(self.base[i] for i in indices)
        if any(s.pair_id != self.pair_metadata[i][0] or int(s.label) != self.pair_metadata[i][1]
               for s, i in zip(originals, indices)):
            raise ValueError("reference TRAIN order/labels changed")
        coin = float(self._rng(ids, "guided-curve-coin-v1").random())
        requested = bool(coin < self.config.probability)
        group = dict(schema_version=SCHEMA_VERSION, seed=self.seed, epoch=self.epoch,
            group_pair_ids=list(ids), requested=requested, applied=False, reason="shared_coin_skipped",
            attempts=0, attempt_reasons={}, members=[{}, {}], hard_negative=dict(replaced=False),
            request_coin=coin, requested_probability=self.config.probability,
            acceptance_policy="paired_positive_and_native_negative_joint_acceptance")
        if not requested:
            return originals, group
        masks = [tuple(_mask_2d(getattr(s, "mask_" + side)) for side in "ab") for s in originals]
        sides = ["a" if m[0].sum() <= m[1].sum() else "b" for m in masks]
        selected = [m[0 if side == "a" else 1] for m, side in zip(masks, sides)]
        try:
            context = source_seam_context(originals[0], sides[0], self.config.crop_config)
        except (ValueError, RachelPreprocessError):
            context = None
        if context is None:
            group["reason"] = "original_source_seam_unavailable"
            return originals, group
        reasons = Counter()
        for attempt in range(1, self.config.max_attempts + 1):
            proposal = self.proposal(ids, attempt)
            group["attempts"] = attempt
            profile = self.bank.profiles[proposal["donor_index"]]
            params = {k: proposal[k] for k in ("angle_deg", "keep_low", "flip", "reverse")}
            try:
                pos_crop, pos_offset = guided_curve_mask(selected[0], profile,
                    points=context["points"], weights=context["weights"],
                    target_retention=proposal["target_retention"], **params)
            except ValueError:
                reasons["positive:unsolvable_source_quantile"] += 1
                continue
            retention = pos_offset["actual_retention"]
            if not self.config.retention_min <= retention <= self.config.retention_max:
                reasons["positive:source_seam_not_partial"] += 1
                continue
            pos_result = crop_training_pair(originals[0], side=sides[0],
                retained_mask=pos_crop, config=self.config.crop_config, topology_connectivity=2)
            if not pos_result.accepted:
                reasons["positive:" + pos_result.reason] += 1
                continue
            try:
                neg_crop, neg_offset = material_matched_curve_mask(selected[1], profile,
                    target_retention=pos_offset["material_retention"], **params)
            except ValueError:
                reasons["negative:unsolvable_material_quantile"] += 1
                continue
            neg_result = crop_training_pair(originals[1], side=sides[1],
                retained_mask=neg_crop, config=self.config.crop_config, topology_connectivity=2)
            if not neg_result.accepted:
                reasons["negative:" + neg_result.reason] += 1
                continue
            results, offsets = (pos_result, neg_result), (pos_offset, neg_offset)
            outputs = tuple(_ignore_ambiguous_nonmatches(old, result.sample, side, None)
                for old, result, side in zip(originals, results, sides))
            donor = self.bank.metadata["arcs"][proposal["donor_index"]]
            reports = []
            for old, result, side, offset, original_masks in zip(originals, results, sides, offsets, masks):
                areas = [int(m.sum()) for m in original_masks]
                reports.append(dict(side=side, proposal=proposal, offset_metadata=offset,
                    material_retention=offset["material_retention"],
                    donor_lineage=donor["lineage"], donor_family=donor["family"], donor_split="train",
                    cut_kind="oblique_real_outline_profile_source_seam_guided",
                    original_area_ratio=min(areas) / max(areas),
                    post_area_ratio=result.diagnostics["post_area_ratio"],
                    source_seam_length_px=context["length"] if old.label else None,
                    source_seam_retention=retention if old.label else None,
                    physically_retained_source_seam_length_px=retention * context["length"] if old.label else None,
                    retained_supervised_seam_length_px=result.diagnostics["retained_contiguous_seam_length_px"] if old.label else None,
                    supervised_token_count=result.diagnostics["token_match_count"],
                    artificial_cut_exclusion_px=8., geometry=result.diagnostics))
            group.update(applied=True, reason="coupled_accepted", proposal=proposal,
                         members=reports, attempt_reasons=dict(reasons))
            return outputs, group
        group.update(reason="no_jointly_accepted_attempt", attempt_reasons=dict(reasons))
        return originals, group


__all__ = ["GuidedPartialDataset", "guided_curve_mask", "material_matched_curve_mask"]

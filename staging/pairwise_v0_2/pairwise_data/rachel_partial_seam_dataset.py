"""TRAIN-only asymmetric partial-seam truncation in the original canvas.

The two labels in each deterministic group share proposal parameters AND
acceptance. Physical surviving source-seam coverage is checked independently
of the 8px artificial-edge supervision exclusion. No new-boundary geometric
match is ever promoted to positive GT. This is not proof of no shape shortcuts.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .rachel_size_crop import (SizeCropConfig, SizeCropResult, _mask_2d,
    _original_gt_edges, _gt_supported_dense, _updated_coarse)
from .rachel_size_crop_dataset import RachelSizeCropDataset, SizeCropDatasetConfig
from .rachel_preprocess import (_dense_external_contour, _accepted_continuous_seam,
    recover_mutual_contour_correspondences, RachelPreprocessError, extract_ordered_outer_contour)
from .rachel_training_dataset import _readonly
from .rachel_weathered_dataset import (source_arc_ancestry, inherit_pair_targets,
    _FragmentView, _identity_view)


SCHEMA_VERSION = "rachel-partial-source-seam/1"


@dataclass(frozen=True)
class PartialSeamConfig:
    probability: float = .50
    retention_min: float = .25
    retention_max: float = .75
    cut_fraction_min: float = .15
    cut_fraction_max: float = .85
    max_attempts: int = 6
    source_sampling_gap_allowance_px: float = 8.
    cache_groups: int = 8
    crop_config: SizeCropConfig = field(default_factory=lambda: SizeCropConfig(
        min_retained_seam_length_px=32., min_matched_tokens=4))

    def __post_init__(self):
        values = (self.probability, self.retention_min, self.retention_max,
                  self.cut_fraction_min, self.cut_fraction_max)
        if not np.isfinite(values).all() or not 0 <= self.probability <= 1:
            raise ValueError("invalid partial-seam probabilities")
        if not 0 < self.retention_min <= self.retention_max < 1:
            raise ValueError("physical source-seam retention must be strictly partial")
        if not 0 < self.cut_fraction_min < self.cut_fraction_max < 1:
            raise ValueError("cut fractions must be inside the original bbox")
        if any(type(x) is not int or x < 1 for x in (self.max_attempts, self.cache_groups)):
            raise ValueError("attempt/cache counts must be positive integers")
        if not isinstance(self.crop_config, SizeCropConfig):
            raise TypeError("crop_config must be SizeCropConfig")
        if self.source_sampling_gap_allowance_px != 8.:
            raise ValueError("this first controlled recipe fixes the source sampling gap at 8px")


class _TrainProxy:
    split = "train"
    def __init__(self, base):
        self.base, self.root = base, getattr(base, "root", None)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        return self.base[index]


def _train_base(base):
    if getattr(base, "split", None) == "train":
        return base
    # The composite adapter exposes TRAIN through its validated child loaders,
    # not .split; do not blindly treat an arbitrary missing split as TRAIN.
    from .rachel_composite_training import CompositeRachelPairDataset
    if (isinstance(base, CompositeRachelPairDataset) and base._sources
            and all(source.split == "train" for source in base._sources.values())):
        return _TrainProxy(base)
    raise ValueError("partial seam augmentation accepts only explicit TRAIN sources")


def source_seam_context(sample, side, config):
    """GT-supported original dense seam, before ANY truncation or prediction."""
    edges = _original_gt_edges(sample)
    if not len(edges):
        return None
    da, db = (_dense_external_contour(_mask_2d(getattr(sample, "mask_" + s))) for s in "ab")
    shift = np.asarray(sample.translation_a_to_b_rc, dtype=float)
    raw = recover_mutual_contour_correspondences(da + shift, db,
                                                 max_distance_px=config.dense_match_distance_px)
    seam = _accepted_continuous_seam(raw, da + shift, db)
    supported = _gt_supported_dense(seam, da, db, sample, edges,
                                    config.original_gt_support_distance_px)
    dense, indices = (da, supported[:, 0]) if side == "a" else (db, supported[:, 1])
    if not len(indices):
        return None
    indices = np.unique(indices)
    steps = np.linalg.norm(np.roll(dense, -1, axis=0) - dense, axis=1)
    weights = .5 * (steps[indices] + steps[(indices - 1) % len(dense)])
    starts = np.r_[0., np.cumsum(steps[:-1])]
    return dict(points=dense[indices], weights=weights, length=float(weights.sum()),
                arcs=starts[indices], perimeter=float(steps.sum()))


def retained_source_arc_length(context, crop, cut_tree):
    """Length counts observed source arc cells only, not the gaps between them.

    Consecutive supported source samples may be up to 8px apart. Missing pixels
    in the cropped mask or proximity to the new cut always break eligibility.
    This tolerates raster MNN skips without declaring a new geometric seam.
    """
    p = context["points"]
    pixel = np.rint(p).astype(int)
    keep = crop[pixel[:, 0], pixel[:, 1]] & (cut_tree.query(p)[0] > 8.)
    indices = np.flatnonzero(keep)
    if not len(indices):
        return 0.
    arcs, weights, perimeter = context["arcs"][indices], context["weights"][indices], context["perimeter"]
    gaps = (np.roll(arcs, -1) - arcs) % perimeter
    if np.all(gaps <= 8.):
        return float(weights.sum())
    start = (int(np.flatnonzero(gaps > 8.)[0]) + 1) % len(indices)
    current = best = 0.
    for offset in range(len(indices)):
        i = (start + offset) % len(indices)
        if offset and gaps[(i - 1) % len(indices)] > 8.:
            current = 0.
        current += weights[i]
        best = max(best, current)
    return float(best)


def crop_training_pair(sample, *, side, bounds_rc=None, retained_mask=None, config,
                       topology_connectivity=1):
    """Partial-specific crop: inherit old reciprocal token edges by source arc.

    Unlike the historical strict pixel-MNN crop prototype, no requirement says
    that both dense raster indices advance by exactly one at every step.
    """
    if topology_connectivity not in (1, 2):
        raise ValueError("topology connectivity must be 4-neighbor (1) or 8-neighbor (2)")
    ma, mb = (_mask_2d(getattr(sample, "mask_" + s)) for s in "ab")
    original = ma if side == "a" else mb
    if (bounds_rc is None) == (retained_mask is None):
        raise ValueError("provide exactly one crop geometry")
    if retained_mask is not None:
        crop = np.asarray(retained_mask, dtype=bool)
        if crop.shape != original.shape or np.any(crop & ~original):
            raise ValueError("partial geometry must only delete original material")
    else:
        r0, r1, c0, c1 = bounds_rc
        crop = np.zeros_like(original)
        crop[r0:r1, c0:c1] = original[r0:r1, c0:c1]
    areas = [int(ma.sum()), int(mb.sum())]
    areas[0 if side == "a" else 1] = int(crop.sum())
    diag = dict(post_area_ratio=min(areas) / max(areas) if min(areas) else None,
        token_match_count=0, retained_contiguous_seam_length_px=0.,
        correspondence_policy="original reciprocal tokens via within-fragment source arcs",
        source_sampling_gap_allowance_px=8., artificial_cut_exclusion_px=8.,
        topology_connectivity=4 if topology_connectivity == 1 else 8)
    def reject(reason):
        return SizeCropResult(False, sample, reason, diag)
    connection = ndimage.generate_binary_structure(2, 1)
    topology = ndimage.generate_binary_structure(2, topology_connectivity)
    if not min(areas):
        return reject("empty_fragment")
    if any(ndimage.label(mask, structure=topology)[1] != 1 for mask in (ma, mb)):
        return reject("original_not_single_component")
    if ndimage.label(crop, structure=topology)[1] != 1:
        return reject("disconnected_crop")
    if np.array_equal(original, crop):
        return reject("no_op_crop")
    if bool(sample.label) != bool(sample.translation_valid):
        return reject("invalid_translation_flag")
    cut_pixels = np.argwhere(crop & ndimage.binary_dilation(original & ~crop, structure=connection))
    if not len(cut_pixels):
        return reject("no_artificial_cut_boundary")
    cut_tree = cKDTree(cut_pixels)
    try:
        points, valid = extract_ordered_outer_contour(crop, cap=config.contour_cap,
                                                    smoothing_sigma=config.smoothing_sigma)
    except RachelPreprocessError:
        return reject("invalid_crop_contour")
    views = {}
    for endpoint, mask in zip("ab", (ma, mb)):
        old_points, old_valid = getattr(sample, "points_rc_" + endpoint), getattr(sample, "contour_valid_" + endpoint)
        if endpoint != side:
            views[endpoint] = _identity_view(mask, old_points, old_valid, "clean", {})
            continue
        ancestor, representatives, projection = source_arc_ancestry(mask, old_points, old_valid, points, 0.)
        forbidden = cut_tree.query(points)[0] <= config.cut_edge_exclusion_px
        representatives[(representatives >= 0) & forbidden[np.maximum(representatives, 0)]] = -1
        ancestor[forbidden] = -1
        views[endpoint] = _FragmentView(_readonly(crop, bool), points, valid,
            _readonly(ancestor, np.int64), _readonly(representatives, np.int64), True, "partial", projection)
    target_a, target_b = inherit_pair_targets(sample, views["a"], views["b"])
    if sample.label:
        # The retained seam is still intact: disallow source representatives
        # whose sampling displacement no longer fits the unchanged GT xy.
        ia = np.flatnonzero(target_a >= 0)
        ib = target_a[ia]
        residual = np.linalg.norm(views["a"].points[ia] + sample.translation_a_to_b_rc
                                  - views["b"].points[ib], axis=1)
        bad = residual > config.max_token_residual_px
        target_a[ia[bad]], target_b[ib[bad]] = -2, -2
        context = source_seam_context(sample, side, config)
        length = retained_source_arc_length(context, crop, cut_tree) if context else 0.
        diag["retained_contiguous_seam_length_px"] = length
        if length < config.min_retained_seam_length_px:
            return reject("retained_source_arc_too_short")
        diag["token_match_count"] = int((target_a >= 0).sum())
        if diag["token_match_count"] < config.min_matched_tokens:
            return reject("too_few_inherited_token_matches")
    updates = {"mask_" + side: _readonly(crop[None], np.float32),
        "coarse_mask_" + side: _updated_coarse(crop, getattr(sample, "coarse_mask_" + side)),
        "points_rc_" + side: points, "contour_valid_" + side: valid,
        "target_a": _readonly(target_a, np.int64), "target_b": _readonly(target_b, np.int64)}
    return SizeCropResult(True, replace(sample, **updates), "accepted", diag)


def halfplane_bounds(mask, *, axis, keep_low, fraction):
    rc = np.argwhere(mask)
    if not len(rc):
        return None
    minimum, maximum = rc[:, axis].min(), rc[:, axis].max()
    cut = int(round(minimum + fraction * (maximum - minimum + 1)))
    cut = max(int(minimum) + 1, min(int(maximum), cut))
    bounds = [0, mask.shape[0], 0, mask.shape[1]]
    bounds[axis * 2 + (1 if keep_low else 0)] = cut
    return tuple(bounds)


def _ignore_ambiguous_nonmatches(original, student, side, bounds):
    """Do not convert unrepresented old seam tokens or new cuts to dustbin GT."""
    updates = {}
    for endpoint in "ab":
        targets = np.array(getattr(student, "target_" + endpoint), copy=True)
        points = np.asarray(getattr(student, "points_rc_" + endpoint))
        old_targets = np.asarray(getattr(original, "target_" + endpoint))
        old_points = np.asarray(getattr(original, "points_rc_" + endpoint))
        original_seam = old_points[old_targets >= 0]
        if len(original_seam):
            near = cKDTree(original_seam).query(points)[0] <= 6.
            targets[(targets < 0) & near] = -2
        if endpoint == side:
            ma = _mask_2d(getattr(original, "mask_" + side))
            mb = _mask_2d(getattr(student, "mask_" + side))
            cut = np.argwhere(mb & ndimage.binary_dilation(ma & ~mb,
                structure=ndimage.generate_binary_structure(2, 1)))
            if len(cut):
                near = cKDTree(cut).query(points)[0] <= 8.
                targets[near & (targets < 0)] = -2
        updates["target_" + endpoint] = _readonly(targets, np.int64)
    return replace(student, **updates)


class PartialSeamDataset(RachelSizeCropDataset):
    """Return samples; diagnostics(index) returns the separate generation recipe.

    Set the epoch before constructing nonpersistent DataLoader workers. The
    proposal chooses the smaller endpoint by area (label blind), then samples
    a bbox-normalized axis/retained-half. The other endpoint is never cropped.
    """
    def __init__(self, base, seed=260910, epoch=0, *, pair_metadata,
                 config=PartialSeamConfig()):
        if not isinstance(config, PartialSeamConfig):
            raise TypeError("config must be PartialSeamConfig")
        super().__init__(_train_base(base), pair_metadata=pair_metadata, seed=seed,
            config=SizeCropDatasetConfig(probability=config.probability,
                max_attempts=config.max_attempts, cache_groups=config.cache_groups,
                crop_config=config.crop_config))
        self.config = config
        self.set_epoch(epoch)

    def _rng(self, pair_ids, tag, *parts):
        payload = json.dumps([SCHEMA_VERSION, self.seed, self.epoch, pair_ids, tag, parts],
                             sort_keys=True, separators=(",", ":"))
        seed = int.from_bytes(hashlib.sha256(payload.encode()).digest(), "big")
        return np.random.Generator(np.random.PCG64(seed))

    def _materialize(self, group_index):
        indices = self._groups[group_index]
        ids = tuple(self.pair_metadata[i][0] for i in indices)
        originals = tuple(self.base[i] for i in indices)
        if any(sample.pair_id != self.pair_metadata[i][0]
               or int(sample.label) != self.pair_metadata[i][1]
               for sample, i in zip(originals, indices)):
            raise ValueError("base order/labels differ from fixed TRAIN metadata")
        requested = self._rng(ids, "coin").random() < self.config.probability
        group = dict(schema_version=SCHEMA_VERSION, seed=self.seed, epoch=self.epoch,
            group_pair_ids=list(ids), requested=bool(requested), applied=False,
            reason="shared_coin_skipped", attempts=0, attempt_reasons={}, members=[{}, {}])
        if not requested:
            return originals, group
        masks = [tuple(_mask_2d(getattr(sample, "mask_" + s)) for s in "ab") for sample in originals]
        sides = ["a" if pair[0].sum() <= pair[1].sum() else "b" for pair in masks]
        try:
            context = source_seam_context(originals[0], sides[0], self.config.crop_config)
        except (ValueError, RachelPreprocessError):
            context = None
        if context is None:
            group["reason"] = "original_source_seam_unavailable"
            return originals, group
        reasons = Counter()
        for attempt in range(1, self.config.max_attempts + 1):
            rng = self._rng(ids, "attempt", attempt)
            axis, keep_low = int(rng.integers(2)), bool(rng.integers(2))
            fraction = float(rng.uniform(self.config.cut_fraction_min, self.config.cut_fraction_max))
            bounds = [halfplane_bounds(pair[0 if side == "a" else 1], axis=axis,
                keep_low=keep_low, fraction=fraction) for pair, side in zip(masks, sides)]
            group["attempts"] = attempt
            if any(bound is None for bound in bounds):
                reasons["empty_original"] += 1
                continue
            r0, r1, c0, c1 = bounds[0]
            points, weights = context["points"], context["weights"]
            survives = ((points[:, 0] >= r0) & (points[:, 0] < r1)
                        & (points[:, 1] >= c0) & (points[:, 1] < c1))
            retained = float(weights[survives].sum())
            retention = retained / context["length"]
            if not self.config.retention_min <= retention <= self.config.retention_max:
                reasons["physical_seam_not_partial"] += 1
                continue
            results = [crop_training_pair(sample, side=side, bounds_rc=bound,
                        config=self.config.crop_config)
                       for sample, side, bound in zip(originals, sides, bounds)]
            if not all(result.accepted for result in results):
                for label, result in zip(("positive", "negative"), results):
                    if not result.accepted:
                        reasons[label + ":" + result.reason] += 1
                continue
            outputs = tuple(_ignore_ambiguous_nonmatches(old, result.sample, side, bound)
                for old, result, side, bound in zip(originals, results, sides, bounds))
            member_reports = []
            for label, old, result, side, bound in zip((1, 0), originals, results, sides, bounds):
                areas = [int(np.count_nonzero(getattr(old, "mask_" + s))) for s in "ab"]
                member_reports.append(dict(side=side, bounds_rc=list(bound), axis=axis,
                    keep_low=keep_low, fraction=fraction,
                    original_area_ratio=min(areas) / max(areas),
                    post_area_ratio=result.diagnostics["post_area_ratio"],
                    source_seam_length_px=context["length"] if label else None,
                    physically_retained_source_seam_length_px=retained if label else None,
                    source_seam_retention=retention if label else None,
                    retained_supervised_seam_length_px=result.diagnostics["retained_contiguous_seam_length_px"] if label else None,
                    supervised_token_count=result.diagnostics["token_match_count"],
                    artificial_cut_exclusion_px=self.config.crop_config.cut_edge_exclusion_px,
                    geometry=result.diagnostics))
            group.update(applied=True, reason="coupled_accepted", members=member_reports,
                         attempt_reasons=dict(reasons))
            return outputs, group
        group.update(reason="no_jointly_accepted_attempt", attempt_reasons=dict(reasons))
        return originals, group

    def diagnostics(self, index):
        index = self._index(index)
        group_index = self._index_to_group[index]
        _, group = self._get_group(group_index)
        member = 0 if self._groups[group_index][0] == index else 1
        result = {key: deepcopy(value) for key, value in group.items() if key != "members"}
        result.update(deepcopy(group["members"][member]))
        result.update(pair_id=self.pair_metadata[index][0], label=bool(self.pair_metadata[index][1]))
        return result


__all__ = ["PartialSeamDataset", "PartialSeamConfig", "source_seam_context", "halfplane_bounds"]

"""E1 TRAIN-only weathering wrapper with source-arc correspondence inheritance.

The weathering/tier/cache path sees one fragment only, never pair labels. Target
inheritance subsequently uses the existing clean reciprocal assignments; it
never matches the two weathered contours by proximity. Original frame and GT
translation remain unchanged. An independent E1 loss must mask its translation
auxiliary on changed positive pairs using report['pose_supervision_enabled'].
This module does not alter the original dataset, collate, loss or evaluator.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .rachel_edge_weathering import EdgeWeatheringConfig, weather_fragment_edges
from .rachel_preprocess import extract_ordered_outer_contour
from .rachel_training_dataset import RachelPairSample, _readonly


SCHEMA = "rachel-weathered-source-arc-e1/v1"
SMOOTHING_ALLOWANCE_PX = 6.0  # Conservative allowance around the existing sigma3 curve.
_CACHE_SIZE = 128
_CACHE_PROTOCOL = dict(contour_cap=512, smoothing_sigma=3., smoothing_allowance_px=SMOOTHING_ALLOWANCE_PX,
                       projection_ambiguity_margin_px=.5, weathering_defaults=asdict(EdgeWeatheringConfig()))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _circular_distance(a, b, length):
    delta = np.abs(a - b)
    return np.minimum(delta, length - delta)


def _project_source_arc(points, source, radius, nonlocal_arc_gap):
    """Project onto local source line segments; reject nearby competing branches."""
    source = np.asarray(source, np.float64)
    points = np.asarray(points, np.float64)
    edge = np.roll(source, -1, axis=0) - source
    length = np.linalg.norm(edge, axis=1)
    if np.any(length <= 0):
        raise ValueError("source contour contains duplicate adjacent vertices")
    starts = np.r_[0., np.cumsum(length[:-1])]
    perimeter = float(length.sum())
    _, vertices = cKDTree(source).query(points, k=min(16, len(source)))
    vertices = np.asarray(vertices).reshape(len(points), -1)
    segments = np.concatenate((vertices, (vertices - 1) % len(source)), axis=1)
    delta = points[:, None] - source[segments]
    alpha = np.clip((delta * edge[segments]).sum(2) / length[segments] ** 2, 0., 1.)
    projections = source[segments] + alpha[..., None] * edge[segments]
    distances = np.linalg.norm(points[:, None] - projections, axis=2)
    arcs = (starts[segments] + alpha * length[segments]) % perimeter
    winner = distances.argmin(axis=1)
    index = np.arange(len(points))
    best_arc, best_distance = arcs[index, winner], distances[index, winner]
    separated = _circular_distance(arcs, best_arc[:, None], perimeter) > nonlocal_arc_gap
    ambiguous = np.any(separated & (distances <= best_distance[:, None] + .5), axis=1)
    trusted = (best_distance <= radius) & ~ambiguous
    # A new CCW walk must not jump backwards along the original source curve.
    if len(points) >= 4:
        advance = (np.roll(best_arc, -1) - best_arc + perimeter / 2) % perimeter - perimeter / 2
        backward = advance < -.5
        trusted &= ~(backward | np.roll(backward, 1))
    return best_arc, trusted, best_distance, perimeter


def source_arc_ancestry(clean_mask, clean_points, clean_valid, new_points, max_depth_px):
    """Return old_index per new token and one representative per clean token.

    Both token sets project to the same original dense contour. Arc-Voronoi
    ownership and representative choice use within-fragment arclength only.
    Additional descendants, ambiguous projections and near-tied cells are ignored.
    """
    dense, _ = extract_ordered_outer_contour(clean_mask, cap=clean_mask.size, smoothing_sigma=0.)
    old_ids = np.flatnonzero(clean_valid)
    old_points = np.asarray(clean_points)[old_ids]
    perimeter_guess = float(np.linalg.norm(np.roll(dense, -1, axis=0) - dense, axis=1).sum())
    separation = max(8., 2. * perimeter_guess / max(1, len(old_ids)))
    old_arc, old_trust, _, perimeter = _project_source_arc(old_points, dense, SMOOTHING_ALLOWANCE_PX, separation)
    new_arc, new_trust, distances, _ = _project_source_arc(new_points, dense,
        max_depth_px + SMOOTHING_ALLOWANCE_PX, separation)
    costs = _circular_distance(new_arc[:, None], old_arc[None], perimeter)
    # Do not let ignored/ambiguous old points gain ownership of another cell.
    nearest = costs.argmin(axis=1)
    trusted = new_trust & old_trust[nearest]
    if len(old_ids) > 1:
        two = np.partition(costs, 1, axis=1)[:, :2]
        trusted &= np.abs(two[:, 0] - two[:, 1]) > 1e-6
    ancestor = np.full(len(new_points), -1, np.int64)
    ancestor[trusted] = old_ids[nearest[trusted]]
    representatives = np.full(len(clean_points), -1, np.int64)
    for old_position, old_id in enumerate(old_ids):
        children = np.flatnonzero(ancestor == old_id)
        if not len(children):
            continue
        order = np.argsort(costs[children, old_position], kind="stable")
        chosen = children[order[0]]
        if len(order) > 1 and abs(costs[chosen, old_position] - costs[children[order[1]], old_position]) <= 1e-6:
            ancestor[children] = -1
            continue
        representatives[old_id] = chosen
        ancestor[children[children != chosen]] = -1
    return ancestor, representatives, dict(source_projection_radius_px=max_depth_px + SMOOTHING_ALLOWANCE_PX,
        smoothing_allowance_px=SMOOTHING_ALLOWANCE_PX, projection_ambiguity_margin_px=.5,
        projected_max_distance_px=float(distances.max(initial=0.)),
        trusted_representatives=int((representatives >= 0).sum()),
        ignored_new_ancestry_count=int((ancestor < 0).sum()), source_arc_perimeter_px=perimeter)


@dataclass(frozen=True)
class _FragmentView:
    mask: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    ancestor: np.ndarray
    representatives: np.ndarray
    changed: bool
    tier: str
    report: dict


def _identity_view(mask, points, valid, tier, report):
    indices = np.arange(len(points), dtype=np.int64)
    indices[~valid] = -1
    return _FragmentView(_readonly(mask, np.bool_), _readonly(points, np.float32), _readonly(valid, np.bool_),
        _readonly(indices, np.int64), _readonly(indices, np.int64), False, tier, report)


def inherit_pair_targets(sample, first, second):
    """Reciprocal clean edge (i,j) -> the two independently inherited tokens."""
    a, b = np.full(len(first.points), -2, np.int64), np.full(len(second.points), -2, np.int64)
    if not bool(sample.label):
        a[first.valid], b[second.valid] = -1, -1
        return a, b
    for result, view, old_targets, old_valid in ((a, first, sample.target_a, sample.contour_valid_a),
                                               (b, second, sample.target_b, sample.contour_valid_b)):
        old_ids = np.flatnonzero(old_valid)
        dustbin = np.asarray(old_targets)[old_ids] == -1
        # A changed token near a seam/nonseam boundary has uncertain dustbin GT.
        if view.changed:
            dustbin &= np.roll(dustbin, 1) & np.roll(dustbin, -1)
        for old_id in old_ids[dustbin]:
            representative = view.representatives[old_id]
            if representative >= 0:
                result[representative] = -1
    for i in np.flatnonzero(np.asarray(sample.target_a) >= 0):
        j = int(sample.target_a[i])
        if not 0 <= j < len(sample.target_b) or sample.target_b[j] != i:
            raise ValueError("original clean targets are not reciprocal")
        u, v = int(first.representatives[i]), int(second.representatives[j])
        if u >= 0 and v >= 0:
            a[u], b[v] = v, u
    return a, b


class RachelWeatheredDataset:
    """Return (RachelPairSample, report); use an E1-specific collate and loss.

    Tier probabilities are per FRAGMENT (default 70/25/5), not per pair. Disk
    caches, when requested, contain fragment-only geometry and no pair labels or
    targets. Recreate training workers/loaders after set_epoch; persistent worker
    copies do not automatically observe a main-process epoch change.
    """

    def __init__(self, base_dataset, seed=260909, epoch=0, cache_dir=None,
                 clean_probability=.70, mild_probability=.25, mild_depth_px=2., moderate_depth_px=4.):
        if type(seed) is not int or type(epoch) is not int or epoch < 0:
            raise ValueError("seed/epoch must be integers and epoch nonnegative")
        if (not np.isfinite((clean_probability, mild_probability)).all()
                or min(clean_probability, mild_probability) < 0 or clean_probability + mild_probability > 1):
            raise ValueError("clean/mild probabilities must be nonnegative with sum <=1")
        EdgeWeatheringConfig(max_depth_px=mild_depth_px)
        EdgeWeatheringConfig(max_depth_px=moderate_depth_px)
        self.base_dataset, self.seed, self.epoch = base_dataset, seed, epoch
        self.root, self.split = getattr(base_dataset, "root", None), getattr(base_dataset, "split", None)
        if self.split not in (None, "train"):
            raise ValueError("weathering wrapper is TRAIN-only; use original VAL/TEST datasets")
        self.contour_cap = 512
        self.parameters = dict(clean_probability=float(clean_probability), mild_probability=float(mild_probability),
                               mild_depth_px=float(mild_depth_px), moderate_depth_px=float(moderate_depth_px))
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache, self._hits, self._disk_hits, self._misses = OrderedDict(), 0, 0, 0

    def __len__(self):
        return len(self.base_dataset)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch
        self._cache.clear()

    def cache_info(self):
        return dict(size=len(self._cache), capacity=_CACHE_SIZE, hits=self._hits, disk_hits=self._disk_hits, misses=self._misses)

    def tier_for_fragment(self, fragment_id):
        draw = int(_digest([SCHEMA, self.seed, self.epoch, str(fragment_id), "tier"])[:16], 16) / float(2 ** 64)
        if draw < self.parameters["clean_probability"]:
            return "clean"
        return "mild" if draw < self.parameters["clean_probability"] + self.parameters["mild_probability"] else "moderate"

    def _disk_paths(self, key):
        directory = self.cache_dir / key[:2]
        return directory / (key + ".npz"), directory / (key + ".json")

    def _load_disk(self, key):
        arrays_path, report_path = self._disk_paths(key)
        if not arrays_path.exists() or not report_path.exists():
            return None
        with report_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("cache_key") != key or metadata.get("schema_version") != SCHEMA:
            raise ValueError("weathering cache identity differs")
        with np.load(arrays_path, allow_pickle=False) as archive:
            mask = np.unpackbits(archive["packed_mask"], axis=1).astype(bool)
            arrays = {name: archive[name] for name in ("points", "valid", "ancestor", "representatives")}
        if mask.shape != (800, 800) or arrays["points"].shape != (len(arrays["valid"]), 2):
            raise ValueError("weathering cache geometry shape differs")
        return _FragmentView(_readonly(mask, np.bool_), *[_readonly(arrays[name], dtype) for name, dtype in
            (("points", np.float32), ("valid", np.bool_), ("ancestor", np.int64), ("representatives", np.int64))],
            metadata["changed"], metadata["tier"], metadata["report"])

    def _save_disk(self, key, view):
        arrays_path, report_path = self._disk_paths(key)
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = "." + uuid.uuid4().hex + ".tmp"
        temp_arrays, temp_report = Path(str(arrays_path) + suffix), Path(str(report_path) + suffix)
        try:
            with temp_arrays.open("xb") as stream:
                np.savez_compressed(stream, packed_mask=np.packbits(view.mask, axis=1), points=view.points,
                                    valid=view.valid, ancestor=view.ancestor, representatives=view.representatives)
            with temp_report.open("x", encoding="utf-8") as stream:
                json.dump(dict(schema_version=SCHEMA, cache_key=key, changed=view.changed,
                               tier=view.tier, report=view.report), stream, allow_nan=False)
            os.replace(temp_arrays, arrays_path)
            os.replace(temp_report, report_path)  # Completion marker written last.
        finally:
            for path in (temp_arrays, temp_report):
                if path.exists():
                    path.unlink()

    def _fragment(self, token, raw_mask, points, valid):
        raw = np.asarray(raw_mask)
        if raw.shape != (1, 800, 800) or not np.isfinite(raw).all() or not np.all((raw == 0) | (raw == 1)):
            raise ValueError("base fragment must be binary [1,800,800]")
        mask = np.ascontiguousarray(raw[0], dtype=np.bool_)
        points, valid = np.asarray(points), np.asarray(valid)
        fingerprint = hashlib.sha256(mask.tobytes() + points.tobytes() + valid.tobytes()).hexdigest()
        key = _digest([SCHEMA, self.seed, self.epoch, token, self.parameters, _CACHE_PROTOCOL, fingerprint])
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        view = self._load_disk(key) if self.cache_dir is not None else None
        if view is not None:
            self._disk_hits += 1
        else:
            self._misses += 1
            tier = self.tier_for_fragment(token)
            depth = 0. if tier == "clean" else self.parameters[tier + "_depth_px"]
            weather_seed = int(_digest([SCHEMA, self.seed, self.epoch, "weather"] )[:16], 16)
            changed_mask, report = weather_fragment_edges(mask, seed=weather_seed, fragment_id=token,
                                                         config=EdgeWeatheringConfig(max_depth_px=depth))
            # Canonical JSON types keep fresh/in-memory/disk-hit reports identical.
            report = json.loads(json.dumps(dict(report, tier=tier), allow_nan=False))
            if np.array_equal(changed_mask, mask):
                view = _identity_view(mask, points, valid, tier, report)
            else:
                new_points, new_valid = extract_ordered_outer_contour(changed_mask, cap=512, smoothing_sigma=3.)
                ancestor, representatives, projection = source_arc_ancestry(mask, points, valid, new_points, depth)
                report["source_arc_inheritance"] = projection
                view = _FragmentView(_readonly(changed_mask, np.bool_), new_points, new_valid,
                    _readonly(ancestor, np.int64), _readonly(representatives, np.int64), True, tier, report)
            if self.cache_dir is not None:
                self._save_disk(key, view)
        self._cache[key] = view
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return view

    def __getitem__(self, index):
        clean = self.base_dataset[index]
        if not isinstance(clean, RachelPairSample) or float(clean.label) not in (0., 1.):
            raise TypeError("base dataset must return binary-labelled RachelPairSample")
        # Complete both fragment-only paths before looking at pair targets.
        a = self._fragment(clean.fragment_a_token, clean.mask_a, clean.points_rc_a, clean.contour_valid_a)
        b = self._fragment(clean.fragment_b_token, clean.mask_b, clean.points_rc_b, clean.contour_valid_b)
        changed_a, changed_b, fallback = a.changed, b.changed, None
        result = clean
        inherited = int(np.count_nonzero(clean.target_a >= 0))
        if changed_a or changed_b:
            target_a, target_b = inherit_pair_targets(clean, a, b)
            inherited = int(np.count_nonzero(target_a >= 0))
            if bool(clean.label) and not np.any(target_a >= 0):
                fallback = "no_reliable_inherited_positive_correspondence"
                changed_a = changed_b = False
            else:
                updates = dict(target_a=_readonly(target_a, np.int64), target_b=_readonly(target_b, np.int64))
                for side, view in (("a", a), ("b", b)):
                    if view.changed:
                        nearest = getattr(Image, "Resampling", Image).NEAREST
                        coarse = np.asarray(Image.fromarray(view.mask.astype(np.uint8) * 255).resize((128, 128), nearest)) > 0
                        updates.update({"mask_" + side: _readonly(view.mask[None], np.float32),
                            "coarse_mask_" + side: _readonly(coarse[None], np.float32),
                            "points_rc_" + side: view.points, "contour_valid_" + side: view.valid})
                result = replace(clean, **updates)
        report = dict(schema_version=SCHEMA, pair_id=clean.pair_id, epoch=self.epoch,
            tier=dict(a=a.tier, b=b.tier), changed_a=bool(changed_a), changed_b=bool(changed_b),
            changed_pair=bool(changed_a or changed_b), pose_supervision_enabled=bool(clean.label) and not (changed_a or changed_b),
            fallback_reason=fallback, inherited_match_count=inherited,
            effective_supervised_match_count=int(np.count_nonzero(result.target_a >= 0)),
            ignored_token_count=int(np.count_nonzero((result.target_a == -2) & result.contour_valid_a)
                                  + np.count_nonzero((result.target_b == -2) & result.contour_valid_b)),
            inheritance_rule="clean reciprocal targets via independent within-fragment source-arc ancestors",
            original_gt_translation_preserved=True, cross_fragment_geometry_used_for_targets=False)
        for side, view, changed in (("a", a, changed_a), ("b", b, changed_b)):
            report["side_" + side] = dict(view.report, attempted_applied=bool(view.report["applied"]),
                effective_applied=bool(changed), effective_removed_area_px=view.report["removed_area_px"] if changed else 0)
        return result, report


__all__ = ["RachelWeatheredDataset", "source_arc_ancestry", "inherit_pair_targets"]

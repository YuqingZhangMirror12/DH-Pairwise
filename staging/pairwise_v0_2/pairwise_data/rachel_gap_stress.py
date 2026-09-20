"""TEST-only inward-weathering stress inputs and metric-only direction evidence.

This does not subclass, disguise the split for, or change any training dataset.
The evaluation runner must enumerate the same original TEST release (3000 rows)
for every model and condition. Fragment geometry depends only on mask, ID, seed
and the fixed 0/2/4px condition. Zero returns the exact original sample object.
Nonzero conditions have all-ignore assignment targets and cannot be trained with
the old supervised loss. Original labels, 800px frame and placement GT survive.

Closing-direction bias is NOT a gap-width ground truth or a gap reconstruction
algorithm. Its subset/normals come exclusively from the CLEAN GT seam. Since
B's placement in A's frame is -t, (t_hat-t_GT) dot outward_normal_A > 0 means B
moved toward A (closing). Normal projection can cancel on a curved seam; the
separate positive/negative arc-weighted projections make this visible.
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

from .rachel_edge_weathering import EdgeWeatheringConfig, weather_fragment_edges
from .rachel_preprocess import extract_ordered_outer_contour
from .rachel_training_dataset import RachelPairSample, _readonly


SCHEMA = "rachel-test-gap-stress/v1"
METRIC_SCHEMA = "rachel-clean-gt-closing-direction-bias/v1"
_CACHE_CAPACITY = 128
_PROBE_DISTANCES_PX = (2., 4., 8.)
_MINIMUM_SUPPORTED_ARC_PX = 8.


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _mask_at(mask, points):
    """Nearest pixel centers; out-of-frame points are explicitly outside."""
    points = np.asarray(points, np.float64)
    indices = np.floor(points + .5).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(mask.shape)), axis=1)
    result = np.zeros(len(indices), dtype=np.bool_)
    result[valid] = mask[indices[valid, 0], indices[valid, 1]]
    return result


@dataclass(frozen=True)
class GapMetricSidecar:
    pair_id: str
    label: bool
    translation_gt_rc: object
    normals_rc: np.ndarray
    arc_weights_px: np.ndarray
    support: dict


def clean_gt_gap_metrics(clean):
    """Prepare prediction-independent, de-duplicated clean seam normal evidence.

    Each valid ordered A token owns half its preceding and following polygon
    edge. These arc cells do not overlap; a token is counted only once, regardless
    of patch overlap or correspondence multiplicity. We report the sampled-clean
    polygon arc coverage, NOT an exact dense physical seam length.
    """
    if not isinstance(clean, RachelPairSample):
        raise TypeError("clean source must be RachelPairSample")
    gt = _readonly(clean.translation_a_to_b_rc, np.float64) if bool(clean.translation_valid) else None
    support = dict(schema_version=METRIC_SCHEMA, valid=False, invalid_reason=None,
        subset_definition="clean reciprocal GT seam only; independent of erosion and prediction",
        arc_definition="non-overlapping half-adjacent-edge cells on clean ordered A contour",
        normal_definition="A outward normal, verified by clean A/B mask probes at original GT",
        probe_distances_px=list(_PROBE_DISTANCES_PX), minimum_supported_arc_px=_MINIMUM_SUPPORTED_ARC_PX,
        perimeter_px=0., seam_token_count=0, seam_arc_px=0., normal_count=0,
        supported_arc_px=0., coverage_fraction=0., normal_resultant_length=None,
        sign_convention="positive=(t_hat_rc-t_GT_rc) dot outward_normal_A; B placement=-t",
        gap_width_ground_truth=False)

    def result(normals=None, weights=None, invalid=None):
        if invalid is not None:
            support["invalid_reason"] = invalid
        return GapMetricSidecar(str(clean.pair_id), bool(clean.label), gt,
            _readonly(np.empty((0, 2)) if normals is None else normals, np.float64),
            _readonly(np.empty(0) if weights is None else weights, np.float64), support)

    if not bool(clean.label):
        return result(invalid="negative_pair_has_no_gt_seam")
    if gt is None or gt.shape != (2,) or not np.isfinite(gt).all():
        return result(invalid="missing_or_nonfinite_gt_translation")
    ids = np.flatnonzero(clean.contour_valid_a)
    points = np.asarray(clean.points_rc_a, np.float64)[ids]
    if len(points) < 4 or not np.isfinite(points).all():
        return result(invalid="invalid_clean_ordered_contour")
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    area = .5 * np.sum(points[:, 1] * np.roll(-points[:, 0], -1)
                        - np.roll(points[:, 1], -1) * -points[:, 0])
    if np.any(lengths <= 1e-8) or abs(area) <= 1e-8:
        return result(invalid="degenerate_clean_ordered_contour")
    weights = .5 * (lengths + np.roll(lengths, 1))
    support["perimeter_px"] = float(lengths.sum())
    targets = np.asarray(clean.target_a)[ids]
    seam = targets >= 0
    if np.any(seam):
        js = targets[seam]
        if (np.any(js >= len(clean.target_b)) or np.any(~np.asarray(clean.contour_valid_b)[js])
                or not np.array_equal(np.asarray(clean.target_b)[js], ids[seam])):
            raise ValueError("clean metric seam targets must be reciprocal and valid")
    support.update(seam_token_count=int(seam.sum()), seam_arc_px=float(weights[seam].sum()))
    if not np.any(seam):
        return result(invalid="no_clean_reciprocal_gt_seam")
    # Orientation comes only from A's closed polygon, never the model/prediction.
    tangent = edges / lengths[:, None] + np.roll(edges / lengths[:, None], 1, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    normals = np.stack((tangent[:, 1], -tangent[:, 0]), axis=1) * (1. if area > 0 else -1.)
    normals /= np.maximum(norm[:, None], 1e-12)
    a, b = np.asarray(clean.mask_a[0], bool), np.asarray(clean.mask_b[0], bool)
    supported = np.zeros(len(points), dtype=np.bool_)
    for distance in _PROBE_DISTANCES_PX:
        plus, minus = points + distance * normals, points - distance * normals
        # At the original placement, +n must go from A to B; require both local
        # inside/outside tests. A guessed normal is never flipped to fit B/GT.
        supported |= (~_mask_at(a, plus) & _mask_at(a, minus)
                      & _mask_at(b, plus + gt) & ~_mask_at(b, minus + gt))
    supported &= seam & (norm > 1e-8)
    accepted_normals, accepted_weights = normals[supported], weights[supported]
    arc = float(accepted_weights.sum())
    support.update(normal_count=int(supported.sum()), supported_arc_px=arc,
        coverage_fraction=arc / max(1e-12, support["seam_arc_px"]))
    if arc > 0:
        support["normal_resultant_length"] = float(np.linalg.norm(
            (accepted_normals * accepted_weights[:, None]).sum(0) / arc))
    if int(supported.sum()) < 2 or arc < _MINIMUM_SUPPORTED_ARC_PX:
        return result(accepted_normals, accepted_weights, "insufficient_gt_supported_normal_arc")
    support["valid"] = True
    return result(accepted_normals, accepted_weights)


def closing_direction_bias(t_hat_rc, metrics):
    """JSON diagnostics; positive signed projection means B moved toward A."""
    if not isinstance(metrics, GapMetricSidecar):
        raise TypeError("metrics must be GapMetricSidecar")
    output = dict(schema_version=METRIC_SCHEMA, pair_id=metrics.pair_id, valid=False,
        invalid_reason=metrics.support.get("invalid_reason"), signed_closing_bias_px=None,
        positive_closing_component_px=None, negative_opening_component_px=None,
        closing_arc_fraction=None, opening_arc_fraction=None, neutral_arc_fraction=None,
        translation_error_px=None, support=dict(metrics.support), gap_width_ground_truth=False)
    if not metrics.support["valid"]:
        return output
    if t_hat_rc is None:
        output["invalid_reason"] = "missing_predicted_translation"
        return output
    estimate = np.asarray(t_hat_rc, np.float64)
    if estimate.shape != (2,) or not np.isfinite(estimate).all():
        output["invalid_reason"] = "nonfinite_or_invalid_predicted_translation"
        return output
    error = estimate - metrics.translation_gt_rc
    projection = metrics.normals_rc @ error
    weights = metrics.arc_weights_px / metrics.arc_weights_px.sum()
    output.update(valid=True, invalid_reason=None,
        signed_closing_bias_px=float(weights @ projection),
        positive_closing_component_px=float(weights @ np.maximum(projection, 0.)),
        negative_opening_component_px=float(weights @ np.minimum(projection, 0.)),
        closing_arc_fraction=float(weights[projection > 1e-8].sum()),
        opening_arc_fraction=float(weights[projection < -1e-8].sum()),
        neutral_arc_fraction=float(weights[np.abs(projection) <= 1e-8].sum()),
        translation_error_px=float(np.linalg.norm(error)))
    return output


@dataclass(frozen=True)
class GapStressSample:
    student: RachelPairSample
    report: dict
    metrics: GapMetricSidecar


def gap_stress_model_inputs(student):
    """Only six unbatched arrays accepted by the existing model forward."""
    return (student.mask_a, student.mask_b, student.points_rc_a, student.points_rc_b,
            student.contour_valid_a, student.contour_valid_b)


class RachelGapStressDataset:
    """TEST-only wrapper. All rows retained; no positive-pair fallback/selection.

    Use the same original 3000-row TEST dataset and cache directory for each
    model. The caller records/enforces release identity and complete enumeration;
    short TEST fixtures are supported without disguising their split.
    """
    def __init__(self, base_test_dataset, max_depth_px=0, seed=260910, cache_dir=None):
        if getattr(base_test_dataset, "split", None) != "test":
            raise ValueError("gap stress is TEST-only; an explicit test split is required")
        if isinstance(max_depth_px, bool) or max_depth_px not in (0, 2, 4):
            raise ValueError("frozen gap stress depths are exactly 0/2/4px")
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        self.base_dataset, self.split = base_test_dataset, "test"
        self.root, self.contour_cap = getattr(base_test_dataset, "root", None), 512
        self.seed, self.max_depth_px = seed, float(max_depth_px)
        self.config = EdgeWeatheringConfig(max_depth_px=self.max_depth_px)
        self.parameters = dict(max_depth_px=self.max_depth_px, seed=seed, contour_cap=512,
            smoothing_sigma=3., canvas_size=800, weathering=asdict(self.config))
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache, self._hits, self._disk_hits, self._misses = OrderedDict(), 0, 0, 0

    def __len__(self):
        return len(self.base_dataset)

    def cache_info(self):
        return dict(size=len(self._cache), capacity=_CACHE_CAPACITY, hits=self._hits,
                    disk_hits=self._disk_hits, misses=self._misses)

    def _paths(self, key):
        directory = self.cache_dir / key[:2]
        return directory / (key + ".npz"), directory / (key + ".json")

    def _load(self, key):
        if self.cache_dir is None:
            return None
        data_path, report_path = self._paths(key)
        if not data_path.exists() or not report_path.exists():
            return None
        with report_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("schema_version") != SCHEMA or metadata.get("cache_key") != key:
            raise ValueError("gap stress cache identity differs")
        with np.load(data_path, allow_pickle=False) as archive:
            mask = np.unpackbits(archive["packed_mask"], axis=1).astype(bool)
            points, valid = archive["points"], archive["valid"]
        if mask.shape != (800, 800) or points.shape != (len(valid), 2):
            raise ValueError("gap stress cache shape differs")
        return (_readonly(mask, np.bool_), _readonly(points, np.float32),
                _readonly(valid, np.bool_), metadata["report"])

    def _save(self, key, value):
        if self.cache_dir is None:
            return
        data_path, report_path = self._paths(key)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = "." + uuid.uuid4().hex + ".tmp"
        temporary = (Path(str(data_path) + suffix), Path(str(report_path) + suffix))
        mask, points, valid, report = value
        try:
            with temporary[0].open("xb") as stream:
                np.savez_compressed(stream, packed_mask=np.packbits(mask, axis=1), points=points, valid=valid)
            with temporary[1].open("x", encoding="utf-8") as stream:
                json.dump(dict(schema_version=SCHEMA, cache_key=key, report=report), stream, allow_nan=False)
            os.replace(temporary[0], data_path)
            os.replace(temporary[1], report_path)
        finally:
            for path in temporary:
                if path.exists():
                    path.unlink()

    def _fragment(self, fragment_id, raw_mask):
        raw_mask = np.asarray(raw_mask)
        if raw_mask.shape != (1, 800, 800) or not np.all((raw_mask == 0) | (raw_mask == 1)):
            raise ValueError("gap stress requires binary [1,800,800] masks")
        mask = np.ascontiguousarray(raw_mask[0], dtype=np.bool_)
        key = _digest([SCHEMA, str(fragment_id), self.parameters, hashlib.sha256(mask.tobytes()).hexdigest()])
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        value = self._load(key)
        if value is not None:
            self._disk_hits += 1
        else:
            self._misses += 1
            changed, report = weather_fragment_edges(mask, seed=self.seed, fragment_id=str(fragment_id), config=self.config)
            report = json.loads(json.dumps(report, allow_nan=False))
            if report["applied"]:
                points, valid = extract_ordered_outer_contour(changed, cap=512, smoothing_sigma=3.)
            else:
                # On skips retain the caller's original token arrays exactly;
                # do not cache old tokens that could differ under the same mask.
                points, valid = np.empty((0, 2), np.float32), np.empty(0, np.bool_)
            value = (_readonly(changed, np.bool_), _readonly(points, np.float32),
                     _readonly(valid, np.bool_), report)
            self._save(key, value)
        self._cache[key] = value
        if len(self._cache) > _CACHE_CAPACITY:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index):
        clean = self.base_dataset[index]
        if not isinstance(clean, RachelPairSample) or float(clean.label) not in (0., 1.):
            raise TypeError("TEST source must return binary-labelled RachelPairSample")
        updates, details = {}, {}
        for side in "ab":
            mask, points, valid, report = self._fragment(getattr(clean, "fragment_" + side + "_token"),
                                                       getattr(clean, "mask_" + side))
            details[side] = dict(report, requested_attempt=self.max_depth_px > 0,
                                 actual_applied=bool(report["applied"]))
            if report["applied"]:
                nearest = getattr(Image, "Resampling", Image).NEAREST
                coarse = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((128, 128), nearest)) > 0
                updates.update({"mask_" + side: _readonly(mask[None], np.float32),
                    "coarse_mask_" + side: _readonly(coarse[None], np.float32),
                    "points_rc_" + side: points, "contour_valid_" + side: valid})
            if self.max_depth_px > 0:
                count = len(points) if report["applied"] else len(getattr(clean, "points_rc_" + side))
                updates["target_" + side] = _readonly(np.full(count, -2), np.int64)
        student = replace(clean, **updates) if updates else clean
        report = dict(schema_version=SCHEMA, pair_id=clean.pair_id, seed=self.seed,
            max_depth_px=self.max_depth_px, changed_a=details["a"]["actual_applied"],
            changed_b=details["b"]["actual_applied"],
            changed_pair=bool(details["a"]["actual_applied"] or details["b"]["actual_applied"]),
            side_a=details["a"], side_b=details["b"], frame_unchanged=True,
            original_gt_translation_preserved=True, assignment_supervision="original at zero; all ignore at nonzero",
            metric_sidecar_model_input=False, pair_fallback=False, source_split="test")
        return GapStressSample(student, report, clean_gt_gap_metrics(clean))


__all__ = ["RachelGapStressDataset", "GapStressSample", "GapMetricSidecar",
           "gap_stress_model_inputs", "clean_gt_gap_metrics", "closing_direction_bias"]

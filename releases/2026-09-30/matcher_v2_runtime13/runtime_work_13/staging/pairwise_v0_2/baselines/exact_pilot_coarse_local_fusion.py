#!/usr/bin/env python3
"""Synthetic-calibrated Exact/Siamese fusion for the exact-seam pilot.

The two source models are already frozen.  This adapter replays the selected
Exact Keypoint-Sinkhorn checkpoint on the exact pilot's synthetic validation
records, aligns those scores with the matched whole-mask Siamese validation
scores, and fits one bounded standardized-logit mixing weight plus an
intercept.  The frozen parameters are written before the real-Dunhuang score
document is opened.

Real labels are used only for post-freeze metrics.  No real score, label,
threshold, direction, RGB, text, bounding box, or rotation feature can affect
the fitted fusion parameters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from staging.pairwise_v0_2.baselines.exact_seam_post_pilot_real_evaluation import (
    LoadedPostPilotPair,
    load_exact_seam_pilot_pair,
)
from staging.pairwise_v0_2.training import exact_seam_pilot as exact_pilot
from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache
from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.training.short_ablation import (
    EvidenceMode,
    PredictionBatch,
)


EXACT_PILOT_FUSION_VERSION = "dunhuang-exact-pilot-coarse-local-fusion/0.1"
EXACT_VALIDATION_SCORE_VERSION = "dunhuang-exact-validation-scores/0.1"
EXACT_METHOD = "exact"
SIAMESE_METHOD = "siamese"
FUSION_METHOD = "fusion/exact-keypoint-sinkhorn+whole-mask-siamese/0.1"
STRICT_POPULATION = "strict547"
BALANCED_POPULATION = "balanced1016"
CONSTRUCTED_STRATUM = "constructed_distractor_not_gt_negative"
_EPSILON = 1e-6
_WEIGHT_GRID = np.linspace(0.0, 1.0, 101, dtype=np.float64)
_METHODS = (EXACT_METHOD, SIAMESE_METHOD, FUSION_METHOD)


class ExactPilotFusionError(RuntimeError):
    """The exact-pilot fusion contract cannot be met."""


def _readonly_float(values: Sequence[float]) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    output.setflags(write=False)
    return output


def _readonly_bool(values: Sequence[bool]) -> np.ndarray:
    output = np.asarray(values, dtype=np.bool_).reshape(-1).copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class AlignedSyntheticCalibration:
    """Pair-aligned source scores from the shared synthetic validation set."""

    pair_ids: Tuple[str, ...]
    cluster_ids: Tuple[str, ...]
    labels: np.ndarray
    exact_probability: np.ndarray
    exact_valid: np.ndarray
    siamese_probability: np.ndarray
    siamese_valid: np.ndarray

    def __post_init__(self) -> None:
        pair_ids = tuple(self.pair_ids)
        cluster_ids = tuple(self.cluster_ids)
        count = len(pair_ids)
        if count == 0 or len(set(pair_ids)) != count:
            raise ExactPilotFusionError("synthetic calibration pair IDs are invalid")
        if len(cluster_ids) != count or any(
            not value for value in pair_ids + cluster_ids
        ):
            raise ExactPilotFusionError("synthetic calibration identities are invalid")
        labels = _readonly_bool(self.labels)
        exact_probability = _readonly_float(self.exact_probability)
        exact_valid = _readonly_bool(self.exact_valid)
        siamese_probability = _readonly_float(self.siamese_probability)
        siamese_valid = _readonly_bool(self.siamese_valid)
        for name, values in (
            ("labels", labels),
            ("exact probability", exact_probability),
            ("exact validity", exact_valid),
            ("Siamese probability", siamese_probability),
            ("Siamese validity", siamese_valid),
        ):
            if values.shape != (count,):
                raise ExactPilotFusionError(name + " cardinality changed")
        for name, probability, valid in (
            (EXACT_METHOD, exact_probability, exact_valid),
            (SIAMESE_METHOD, siamese_probability, siamese_valid),
        ):
            if np.any(valid & (~np.isfinite(probability))):
                raise ExactPilotFusionError(name + " has a nonfinite valid score")
            if np.any(valid & ((probability < 0.0) | (probability > 1.0))):
                raise ExactPilotFusionError(name + " valid score is outside [0,1]")
        object.__setattr__(self, "pair_ids", pair_ids)
        object.__setattr__(self, "cluster_ids", cluster_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "exact_probability", exact_probability)
        object.__setattr__(self, "exact_valid", exact_valid)
        object.__setattr__(self, "siamese_probability", siamese_probability)
        object.__setattr__(self, "siamese_valid", siamese_valid)

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)

    @property
    def common_valid(self) -> np.ndarray:
        output = (
            self.exact_valid
            & self.siamese_valid
            & np.isfinite(self.exact_probability)
            & np.isfinite(self.siamese_probability)
        )
        output.setflags(write=False)
        return output


@dataclass(frozen=True)
class FrozenExactSiameseFusion:
    """Frozen parameters fitted exclusively on synthetic calibration rows."""

    exact_logit_mean: float
    exact_logit_scale: float
    siamese_logit_mean: float
    siamese_logit_scale: float
    siamese_weight: float
    intercept: float
    calibration_pair_count: int
    calibration_positive_count: int
    calibration_negative_count: int
    calibration_log_loss: float

    def __post_init__(self) -> None:
        numeric = (
            self.exact_logit_mean,
            self.exact_logit_scale,
            self.siamese_logit_mean,
            self.siamese_logit_scale,
            self.siamese_weight,
            self.intercept,
            self.calibration_log_loss,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ExactPilotFusionError("fusion parameter is nonfinite")
        if self.exact_logit_scale <= 0.0 or self.siamese_logit_scale <= 0.0:
            raise ExactPilotFusionError("fusion logit scale must be positive")
        if not 0.0 <= self.siamese_weight <= 1.0:
            raise ExactPilotFusionError("Siamese weight must be in [0,1]")
        if (
            self.calibration_pair_count <= 0
            or self.calibration_positive_count <= 0
            or self.calibration_negative_count <= 0
            or self.calibration_positive_count + self.calibration_negative_count
            != self.calibration_pair_count
        ):
            raise ExactPilotFusionError("fusion calibration counts are invalid")

    @property
    def exact_weight(self) -> float:
        return 1.0 - self.siamese_weight

    def portable_dict(self) -> Mapping[str, Any]:
        return {
            "method": FUSION_METHOD,
            "exact_method": EXACT_METHOD,
            "siamese_method": SIAMESE_METHOD,
            "formula": (
                "sigmoid(intercept+exact_weight*z_exact+"
                "siamese_weight*z_siamese)"
            ),
            "exact_logit_mean": self.exact_logit_mean,
            "exact_logit_scale": self.exact_logit_scale,
            "siamese_logit_mean": self.siamese_logit_mean,
            "siamese_logit_scale": self.siamese_logit_scale,
            "exact_weight": self.exact_weight,
            "siamese_weight": self.siamese_weight,
            "intercept": self.intercept,
            "calibration_pair_count": self.calibration_pair_count,
            "calibration_positive_count": self.calibration_positive_count,
            "calibration_negative_count": self.calibration_negative_count,
            "calibration_log_loss": self.calibration_log_loss,
            "fit_population": "shared_balanced_synthetic_validation_as_calibration",
            "fit_weighting": "equal_row",
            "fit_objective": "binary_log_loss",
            "weight_grid": {"minimum": 0.0, "maximum": 1.0, "count": 101},
            "tie_break": "lower_siamese_weight",
            "probability_logit_clip_epsilon": _EPSILON,
            "classification_threshold": 0.5,
            "classification_threshold_origin": "fixed_not_fitted",
            "real_labels_or_scores_used_for_fit": False,
        }


def _logit(probability: np.ndarray) -> np.ndarray:
    bounded = np.clip(
        np.asarray(probability, dtype=np.float64), _EPSILON, 1.0 - _EPSILON
    )
    return np.log(bounded) - np.log1p(-bounded)


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(logit, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-bounded))


def _standardize(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(np.mean(values))
    scale = float(np.sqrt(np.mean((values - mean) ** 2)))
    if scale < 1e-6:
        scale = 1.0
    return (values - mean) / scale, mean, scale


def _optimal_intercept(base: np.ndarray, labels: np.ndarray) -> float:
    target = labels.astype(np.float64)
    low = -40.0
    high = 40.0
    for _ in range(80):
        middle = (low + high) / 2.0
        derivative = float(np.sum(_sigmoid(base + middle) - target))
        if derivative > 0.0:
            high = middle
        else:
            low = middle
    return (low + high) / 2.0


def _log_loss(logit: np.ndarray, labels: np.ndarray) -> float:
    target = labels.astype(np.float64)
    return float(np.mean(np.logaddexp(0.0, logit) - target * logit))


def fit_exact_siamese_fusion(
    calibration: AlignedSyntheticCalibration,
) -> FrozenExactSiameseFusion:
    """Fit a one-dimensional convex mixing weight on synthetic rows only."""

    if not isinstance(calibration, AlignedSyntheticCalibration):
        raise TypeError("calibration must be AlignedSyntheticCalibration")
    usable = calibration.common_valid
    labels = calibration.labels[usable]
    if len(labels) == 0 or not labels.any() or labels.all():
        raise ExactPilotFusionError("calibration intersection lacks both classes")
    exact_z, exact_mean, exact_scale = _standardize(
        _logit(calibration.exact_probability[usable])
    )
    siamese_z, siamese_mean, siamese_scale = _standardize(
        _logit(calibration.siamese_probability[usable])
    )
    best: Optional[Tuple[Tuple[float, float], float, float, float]] = None
    for siamese_weight in _WEIGHT_GRID:
        base = (1.0 - siamese_weight) * exact_z + siamese_weight * siamese_z
        intercept = _optimal_intercept(base, labels)
        loss = _log_loss(base + intercept, labels)
        key = (loss, float(siamese_weight))
        if best is None or key < best[0]:
            best = (key, float(siamese_weight), intercept, loss)
    if best is None:  # pragma: no cover - fixed nonempty grid
        raise ExactPilotFusionError("fusion weight selection failed")
    _key, siamese_weight, intercept, loss = best
    return FrozenExactSiameseFusion(
        exact_logit_mean=exact_mean,
        exact_logit_scale=exact_scale,
        siamese_logit_mean=siamese_mean,
        siamese_logit_scale=siamese_scale,
        siamese_weight=siamese_weight,
        intercept=intercept,
        calibration_pair_count=len(labels),
        calibration_positive_count=int(labels.sum()),
        calibration_negative_count=int((~labels).sum()),
        calibration_log_loss=loss,
    )


def apply_exact_siamese_fusion(
    model: FrozenExactSiameseFusion,
    exact_probability: Sequence[float],
    siamese_probability: Sequence[float],
    valid: Sequence[bool],
) -> np.ndarray:
    """Apply frozen parameters; labels are deliberately absent from the API."""

    if not isinstance(model, FrozenExactSiameseFusion):
        raise TypeError("model must be FrozenExactSiameseFusion")
    exact = np.asarray(exact_probability, dtype=np.float64).reshape(-1)
    siamese = np.asarray(siamese_probability, dtype=np.float64).reshape(-1)
    usable = np.asarray(valid, dtype=np.bool_).reshape(-1)
    if exact.shape != siamese.shape or exact.shape != usable.shape:
        raise ExactPilotFusionError("fusion source score shapes differ")
    usable = usable & np.isfinite(exact) & np.isfinite(siamese)
    output = np.full(exact.shape, np.nan, dtype=np.float64)
    exact_z = (
        _logit(exact[usable]) - model.exact_logit_mean
    ) / model.exact_logit_scale
    siamese_z = (
        _logit(siamese[usable]) - model.siamese_logit_mean
    ) / model.siamese_logit_scale
    output[usable] = _sigmoid(
        model.intercept
        + model.exact_weight * exact_z
        + model.siamese_weight * siamese_z
    )
    output.setflags(write=False)
    return output


def _score_row(
    *,
    pair_id: str,
    component_id: str,
    label: bool,
    probability: float,
    valid: bool,
) -> Mapping[str, Any]:
    return {
        "pair_id": pair_id,
        "component_id": component_id,
        "label": bool(label),
        "probability": float(probability) if valid else None,
        "valid": bool(valid),
    }


def score_exact_synthetic_validation(
    *,
    records: Sequence[TrainingPairRecord],
    models: LoadedPostPilotPair,
    loader: Any,
    cache: GeometryArtifactCache,
    geometry_config: GeometryBatchConfig,
    batch_size: int,
) -> Mapping[str, Any]:
    """Replay the frozen Exact winner on the pilot validation sequence."""

    validation_records = tuple(records)
    if not validation_records or any(
        not isinstance(record, TrainingPairRecord) for record in validation_records
    ):
        raise TypeError("records must be a nonempty TrainingPairRecord sequence")
    if not isinstance(models, LoadedPostPilotPair):
        raise TypeError("models must be LoadedPostPilotPair")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    score_rows: List[Mapping[str, Any]] = []
    loaded = models.exact
    for rows in exact_pilot._chunks(validation_records, batch_size):  # noqa: SLF001
        prepared = exact_pilot._prepared_batch(  # noqa: SLF001
            rows,
            loader=loader,
            cache=cache,
            geometry_config=geometry_config,
        )
        prediction = loaded.session.predict_batch(
            prepared, evidence=EvidenceMode.LOCAL
        )
        if not isinstance(prediction, PredictionBatch) or tuple(
            prediction.probability.shape
        ) != (len(rows),):
            raise ExactPilotFusionError("Exact validation prediction shape changed")
        probability = prediction.probability.detach().to(torch.float64).cpu()
        valid = (
            prediction.valid.detach().cpu()
            & torch.isfinite(probability)
            & (probability >= 0.0)
            & (probability <= 1.0)
        )
        for record, score, usable in zip(rows, probability, valid):
            score_rows.append(
                _score_row(
                    pair_id=record.pair_id,
                    component_id=record.component_id,
                    label=record.label,
                    probability=float(score),
                    valid=bool(usable),
                )
            )
    if len(score_rows) != len(validation_records):
        raise ExactPilotFusionError("Exact validation replay changed row count")
    return {
        "schema_version": EXACT_VALIDATION_SCORE_VERSION,
        "status": "complete_frozen_exact_synthetic_validation_replay",
        "source_checkpoint_epoch": int(models.checkpoint_epochs[EXACT_METHOD]),
        "real_evaluation_accessed": False,
        "records": score_rows,
    }


def rebuild_targeted_validation_records(
    *,
    mask_root: Path,
    seed: int,
    siamese_document: Mapping[str, Any],
) -> Tuple[TrainingPairRecord, ...]:
    """Rebuild only the synthetic parent groups named by Siamese scores.

    The original population builder must scan training and validation groups
    to discover its quota.  At fusion time the matched Siamese score artifact
    already freezes the exact 1,394 calibration IDs, so replaying unrelated
    training groups is unnecessary.  This function loads each named parent
    group once and returns records in the artifact's existing order.
    """

    rows = siamese_document.get("records")
    if not isinstance(rows, list) or not rows:
        raise ExactPilotFusionError("Siamese validation scores lack records")
    root = Path(mask_root).resolve()
    expected: Dict[str, Mapping[str, Any]] = {}
    group_order: List[str] = []
    seen_groups = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ExactPilotFusionError("Siamese validation row is not an object")
        pair_id = row.get("pair_id")
        component_id = row.get("component_id")
        label = row.get("label")
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or pair_id in expected
            or not isinstance(component_id, str)
            or not component_id.startswith("synthetic-parent/")
            or not isinstance(label, bool)
        ):
            raise ExactPilotFusionError("Siamese validation identity is invalid")
        expected[pair_id] = row
        if component_id not in seen_groups:
            seen_groups.add(component_id)
            group_order.append(component_id)

    found: Dict[str, TrainingPairRecord] = {}
    for component_id in group_order:
        relative = component_id[len("synthetic-parent/") :]
        group = (root / relative).resolve()
        try:
            group.relative_to(root)
        except ValueError as exc:
            raise ExactPilotFusionError(
                "synthetic validation group escaped mask root"
            ) from exc
        if not group.is_dir():
            raise FileNotFoundError(group)
        for record in exact_pilot._records_for_group(  # noqa: SLF001
            root, group, split="val", seed=seed
        ):
            row = expected.get(record.pair_id)
            if row is None:
                continue
            if (
                record.component_id != row.get("component_id")
                or record.label is not row.get("label")
            ):
                raise ExactPilotFusionError(
                    "rebuilt synthetic validation metadata changed"
                )
            if record.pair_id in found:
                raise ExactPilotFusionError(
                    "rebuilt synthetic validation pair is duplicated"
                )
            found[record.pair_id] = record
    missing = [pair_id for pair_id in expected if pair_id not in found]
    if missing:
        raise ExactPilotFusionError(
            "{} synthetic validation pair IDs were not rebuilt".format(len(missing))
        )
    return tuple(found[str(row["pair_id"])] for row in rows)


def align_synthetic_calibration(
    exact_document: Mapping[str, Any],
    siamese_document: Mapping[str, Any],
) -> AlignedSyntheticCalibration:
    """Require the two source score artifacts to describe the same sequence."""

    exact_rows = exact_document.get("records")
    siamese_rows = siamese_document.get("records")
    if not isinstance(exact_rows, list) or not isinstance(siamese_rows, list):
        raise ExactPilotFusionError("synthetic score artifact lacks records")
    if not exact_rows or len(exact_rows) != len(siamese_rows):
        raise ExactPilotFusionError("synthetic source score counts differ")
    pair_ids: List[str] = []
    clusters: List[str] = []
    labels: List[bool] = []
    exact_probability: List[float] = []
    exact_valid: List[bool] = []
    siamese_probability: List[float] = []
    siamese_valid: List[bool] = []
    for index, (exact_row, siamese_row) in enumerate(
        zip(exact_rows, siamese_rows)
    ):
        if not isinstance(exact_row, Mapping) or not isinstance(
            siamese_row, Mapping
        ):
            raise ExactPilotFusionError("synthetic score row is not an object")
        identity = (
            exact_row.get("pair_id"),
            exact_row.get("component_id"),
            exact_row.get("label"),
        )
        if identity != (
            siamese_row.get("pair_id"),
            siamese_row.get("component_id"),
            siamese_row.get("label"),
        ):
            raise ExactPilotFusionError(
                "synthetic source row {} identity differs".format(index)
            )
        pair_id, component_id, label = identity
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or not isinstance(component_id, str)
            or not component_id
            or not isinstance(label, bool)
        ):
            raise ExactPilotFusionError("synthetic score identity is invalid")
        exact_is_valid = bool(exact_row.get("valid", True))
        siamese_is_valid = bool(siamese_row.get("valid", True))
        exact_score = exact_row.get("probability")
        siamese_score = siamese_row.get("probability")
        pair_ids.append(pair_id)
        clusters.append(component_id)
        labels.append(label)
        exact_probability.append(
            np.nan if exact_score is None else float(exact_score)
        )
        exact_valid.append(exact_is_valid)
        siamese_probability.append(
            np.nan if siamese_score is None else float(siamese_score)
        )
        siamese_valid.append(siamese_is_valid)
    return AlignedSyntheticCalibration(
        pair_ids=tuple(pair_ids),
        cluster_ids=tuple(clusters),
        labels=np.asarray(labels, dtype=np.bool_),
        exact_probability=np.asarray(exact_probability, dtype=np.float64),
        exact_valid=np.asarray(exact_valid, dtype=np.bool_),
        siamese_probability=np.asarray(siamese_probability, dtype=np.float64),
        siamese_valid=np.asarray(siamese_valid, dtype=np.bool_),
    )


def _weighted_ranking(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> Tuple[float, float]:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = weights > 0.0
    labels = labels[keep]
    scores = scores[keep]
    weights = weights[keep]
    positive_total = float(weights[labels].sum())
    negative_total = float(weights[~labels].sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise ExactPilotFusionError("weighted ranking requires both classes")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    weights = weights[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    positive_weight = np.add.reduceat(weights * labels, starts)
    negative_weight = np.add.reduceat(weights * ~labels, starts)
    true_positive = np.cumsum(positive_weight)
    false_positive = np.cumsum(negative_weight)
    true_positive_rate = true_positive / positive_total
    false_positive_rate = false_positive / negative_total
    previous_tpr = np.r_[0.0, true_positive_rate[:-1]]
    previous_fpr = np.r_[0.0, false_positive_rate[:-1]]
    auroc = np.sum(
        (false_positive_rate - previous_fpr)
        * (true_positive_rate + previous_tpr)
        * 0.5
    )
    precision = true_positive / (true_positive + false_positive)
    auprc = np.sum((true_positive_rate - previous_tpr) * precision)
    return float(auroc), float(auprc)


def _interval(values: Sequence[float], point: float) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ExactPilotFusionError("bootstrap produced no valid replicates")
    lower, upper = np.quantile(array, [0.025, 0.975])
    return {
        "point_estimate": float(point),
        "percentile_95_ci": [float(lower), float(upper)],
        "bootstrap_mean": float(array.mean()),
        "bootstrap_standard_error": (
            float(array.std(ddof=1)) if array.size > 1 else 0.0
        ),
        "probability_delta_gt_zero": float(np.mean(array > 0.0)),
        "valid_replicates": int(array.size),
    }


def _paired_cluster_bootstrap(
    *,
    labels: np.ndarray,
    clusters: np.ndarray,
    scores: Mapping[str, np.ndarray],
    replicates: int,
    rng: np.random.Generator,
) -> Mapping[str, Any]:
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    comparisons = {
        "fusion_minus_exact": (FUSION_METHOD, EXACT_METHOD),
        "fusion_minus_siamese": (FUSION_METHOD, SIAMESE_METHOD),
    }
    draws: Dict[str, Dict[str, List[float]]] = {
        name: {"auroc": [], "auprc": []} for name in comparisons
    }
    skipped = 0
    for _ in range(replicates):
        selected = rng.integers(
            0, len(unique_clusters), size=len(unique_clusters)
        )
        multiplicity = np.bincount(selected, minlength=len(unique_clusters))
        row_weight = multiplicity[inverse].astype(np.float64)
        if not np.any(row_weight[labels] > 0.0) or not np.any(
            row_weight[~labels] > 0.0
        ):
            skipped += 1
            continue
        metrics = {
            method: _weighted_ranking(labels, scores[method], row_weight)
            for method in _METHODS
        }
        for name, (left, right) in comparisons.items():
            draws[name]["auroc"].append(metrics[left][0] - metrics[right][0])
            draws[name]["auprc"].append(metrics[left][1] - metrics[right][1])
    point = {
        method: _weighted_ranking(
            labels, scores[method], np.ones(len(labels), dtype=np.float64)
        )
        for method in _METHODS
    }
    output = {}
    for name, (left, right) in comparisons.items():
        output[name] = {
            "auroc": _interval(
                draws[name]["auroc"], point[left][0] - point[right][0]
            ),
            "auprc": _interval(
                draws[name]["auprc"], point[left][1] - point[right][1]
            ),
        }
    return {
        "sampling_cluster_count": int(len(unique_clusters)),
        "replicates_requested": replicates,
        "skipped_single_class_replicates": skipped,
        "comparisons": output,
    }


def _real_source_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, np.ndarray]:
    probability: Dict[str, List[float]] = {
        EXACT_METHOD: [],
        SIAMESE_METHOD: [],
    }
    valid: Dict[str, List[bool]] = {EXACT_METHOD: [], SIAMESE_METHOD: []}
    for row in rows:
        methods = row.get("methods")
        if not isinstance(methods, Mapping):
            raise ExactPilotFusionError("real row lacks method scores")
        for method in (EXACT_METHOD, SIAMESE_METHOD):
            source = methods.get(method)
            if not isinstance(source, Mapping):
                raise ExactPilotFusionError("real row lacks " + method)
            usable = bool(source.get("valid"))
            value = source.get("probability")
            score = np.nan if value is None else float(value)
            valid[method].append(
                usable and math.isfinite(score) and 0.0 <= score <= 1.0
            )
            probability[method].append(score)
    return MappingProxyType(
        {
            EXACT_METHOD: np.asarray(probability[EXACT_METHOD], dtype=np.float64),
            SIAMESE_METHOD: np.asarray(
                probability[SIAMESE_METHOD], dtype=np.float64
            ),
            "exact_valid": np.asarray(valid[EXACT_METHOD], dtype=np.bool_),
            "siamese_valid": np.asarray(valid[SIAMESE_METHOD], dtype=np.bool_),
        }
    )


def _evaluate_population(
    *,
    name: str,
    indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    all_scores: Mapping[str, np.ndarray],
    common_valid: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> Mapping[str, Any]:
    selected = np.asarray(tuple(indices), dtype=np.int64)
    usable_indices = selected[common_valid[selected]]
    labels = np.asarray(
        [bool(rows[index]["label"]) for index in usable_indices], dtype=np.bool_
    )
    clusters = np.asarray(
        [str(rows[index]["cluster_id"]) for index in usable_indices], dtype=object
    )
    if len(labels) == 0 or not labels.any() or labels.all():
        raise ExactPilotFusionError(name + " common-valid rows lack both classes")
    scores = {
        method: np.asarray(all_scores[method][usable_indices], dtype=np.float64)
        for method in _METHODS
    }
    metrics = {
        method: evaluate_pairwise(
            probability=scores[method].tolist(),
            label=labels.tolist(),
            valid=np.ones(len(labels), dtype=np.bool_).tolist(),
            cluster_id=clusters.tolist(),
            threshold=0.5,
        )
        for method in _METHODS
    }
    return {
        "pair_count": int(len(selected)),
        "common_valid_count": int(len(usable_indices)),
        "common_valid_coverage": float(len(usable_indices) / len(selected)),
        "positive_count": int(
            sum(bool(rows[index]["label"]) for index in selected)
        ),
        "negative_count": int(
            sum(not bool(rows[index]["label"]) for index in selected)
        ),
        "primary_metric_population": "intersection_valid:exact,siamese,fusion",
        "methods": metrics,
        "paired_cluster_bootstrap": _paired_cluster_bootstrap(
            labels=labels,
            clusters=clusters,
            scores=scores,
            replicates=replicates,
            rng=rng,
        ),
    }


def evaluate_frozen_fusion_on_real(
    document: Mapping[str, Any],
    model: FrozenExactSiameseFusion,
    *,
    bootstrap_replicates: int = 20_000,
    bootstrap_seed: int = 260_832,
) -> Mapping[str, Any]:
    """Evaluate already-frozen parameters; real labels cannot alter scores."""

    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 1
    ):
        raise ValueError("bootstrap_replicates must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap_seed must be an integer")
    rows = document.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ExactPilotFusionError("real evaluation lacks pair rows")
    if document.get("evaluated_pair_count") != len(rows):
        raise ExactPilotFusionError("real evaluation pair count changed")
    pair_ids = []
    strict_indices = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ExactPilotFusionError("real pair row is not an object")
        pair_id = row.get("pair_id")
        cluster_id = row.get("cluster_id")
        label = row.get("label")
        stratum = row.get("stratum")
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or not isinstance(cluster_id, str)
            or not cluster_id
            or not isinstance(label, bool)
            or not isinstance(stratum, str)
        ):
            raise ExactPilotFusionError("real pair identity is invalid")
        pair_ids.append(pair_id)
        if stratum != CONSTRUCTED_STRATUM:
            strict_indices.append(index)
    if len(set(pair_ids)) != len(pair_ids) or not strict_indices:
        raise ExactPilotFusionError("real pair population is invalid")
    source = _real_source_arrays(rows)
    common_valid = source["exact_valid"] & source["siamese_valid"]
    fusion = apply_exact_siamese_fusion(
        model,
        source[EXACT_METHOD],
        source[SIAMESE_METHOD],
        common_valid,
    )
    all_scores = {
        EXACT_METHOD: source[EXACT_METHOD],
        SIAMESE_METHOD: source[SIAMESE_METHOD],
        FUSION_METHOD: fusion,
    }
    seed_sequence = np.random.SeedSequence(bootstrap_seed)
    strict_seed, balanced_seed = seed_sequence.spawn(2)
    populations = {
        STRICT_POPULATION: _evaluate_population(
            name=STRICT_POPULATION,
            indices=strict_indices,
            rows=rows,
            all_scores=all_scores,
            common_valid=common_valid,
            replicates=bootstrap_replicates,
            rng=np.random.default_rng(strict_seed),
        ),
        BALANCED_POPULATION: _evaluate_population(
            name=BALANCED_POPULATION,
            indices=range(len(rows)),
            rows=rows,
            all_scores=all_scores,
            common_valid=common_valid,
            replicates=bootstrap_replicates,
            rng=np.random.default_rng(balanced_seed),
        ),
    }
    pair_rows = []
    for index, row in enumerate(rows):
        usable = bool(common_valid[index])
        pair_rows.append(
            {
                "pair_id": row["pair_id"],
                "cluster_id": row["cluster_id"],
                "label": row["label"],
                "stratum": row["stratum"],
                "methods": {
                    EXACT_METHOD: {
                        "probability": (
                            float(source[EXACT_METHOD][index]) if usable else None
                        ),
                        "valid": usable,
                    },
                    SIAMESE_METHOD: {
                        "probability": (
                            float(source[SIAMESE_METHOD][index]) if usable else None
                        ),
                        "valid": usable,
                    },
                    FUSION_METHOD: {
                        "probability": float(fusion[index]) if usable else None,
                        "valid": usable,
                    },
                },
            }
        )
    return {
        "schema_version": EXACT_PILOT_FUSION_VERSION,
        "status": "complete_frozen_synthetic_calibration_real_evaluation",
        "evaluated_pair_count": len(rows),
        "methods": list(_METHODS),
        "no_leakage": {
            "fusion_parameters_frozen_before_real_document_opened": True,
            "real_labels_used_for": "metrics_after_parameters_frozen_only",
            "real_scores_used_for": "frozen_forward_application_only",
            "real_labels_or_scores_used_to_choose_weight": False,
            "real_labels_or_scores_used_to_choose_threshold": False,
            "classification_threshold": 0.5,
            "threshold_origin": "fixed_not_fitted",
            "real_score_restandardization": False,
        },
        "fusion_model": dict(model.portable_dict()),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "master_seed": bootstrap_seed,
            "sampling": (
                "paired uniform resampling of JSON cluster_id values with "
                "replacement; every method receives identical row multiplicities"
            ),
            "confidence_interval": "two-sided_95_percentile",
        },
        "populations": populations,
        "pairs": pair_rows,
    }


def _json_object(path: Path, name: str) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExactPilotFusionError(name + " is not readable JSON") from exc
    if not isinstance(value, Mapping):
        raise ExactPilotFusionError(name + " root must be an object")
    return value


def _write_fresh_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def freeze_parameters_then_open_real(
    *,
    calibration: AlignedSyntheticCalibration,
    parameter_path: Path,
    real_document_loader: Callable[[], Mapping[str, Any]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> Tuple[FrozenExactSiameseFusion, Mapping[str, Any]]:
    """Make the no-leakage file/open ordering explicit and unit-testable."""

    model = fit_exact_siamese_fusion(calibration)
    parameter_document = {
        "schema_version": EXACT_PILOT_FUSION_VERSION,
        "status": "frozen_before_real_evaluation_open",
        "real_evaluation_accessed_during_fit": False,
        "calibration": {
            "pair_count": calibration.pair_count,
            "common_valid_count": int(calibration.common_valid.sum()),
            "positive_count": int(calibration.labels[calibration.common_valid].sum()),
            "negative_count": int((~calibration.labels[calibration.common_valid]).sum()),
            "role": "fusion_fit_only_not_independent_validation_report",
        },
        "model": dict(model.portable_dict()),
    }
    _write_fresh_json(parameter_path, parameter_document)
    # This call is deliberately after the frozen parameter artifact exists.
    real_document = real_document_loader()
    result = evaluate_frozen_fusion_on_real(
        real_document,
        model,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    return model, result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--siamese-validation-scores", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--real-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=260_832)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    exact_scores_path = output_dir / "exact_validation_scores.json"
    parameter_path = output_dir / "fusion_parameters.json"
    result_path = output_dir / "fusion_real_evaluation.json"
    if any(path.exists() or path.is_symlink() for path in (
        exact_scores_path,
        parameter_path,
        result_path,
    )):
        raise FileExistsError("fusion output directory already contains a result")

    pilot_summary = _json_object(args.pilot_summary, "pilot summary")
    pilot_config = pilot_summary.get("config")
    if not isinstance(pilot_config, Mapping):
        raise ExactPilotFusionError("pilot summary lacks config")
    try:
        seed = int(pilot_config["seed"])
        planned_validation_count = int(pilot_config["validation_pairs"])
        default_batch_size = int(pilot_config["batch_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExactPilotFusionError("pilot summary config is invalid") from exc
    batch_size = default_batch_size if args.batch_size is None else args.batch_size
    siamese_document = _json_object(
        args.siamese_validation_scores, "Siamese validation scores"
    )
    validation_records = rebuild_targeted_validation_records(
        mask_root=args.mask_root,
        seed=seed,
        siamese_document=siamese_document,
    )
    if len(validation_records) != planned_validation_count:
        raise ExactPilotFusionError(
            "targeted validation count differs from the pilot summary"
        )
    loader = exact_pilot._DirectoryMaskLoader(args.mask_root)  # noqa: SLF001
    cache = GeometryArtifactCache(args.cache_root)
    geometry_config = GeometryBatchConfig()
    models = load_exact_seam_pilot_pair(args.pilot_summary, device=args.device)
    exact_document = score_exact_synthetic_validation(
        records=validation_records,
        models=models,
        loader=loader,
        cache=cache,
        geometry_config=geometry_config,
        batch_size=batch_size,
    )
    _write_fresh_json(exact_scores_path, exact_document)
    calibration = align_synthetic_calibration(exact_document, siamese_document)
    model, result = freeze_parameters_then_open_real(
        calibration=calibration,
        parameter_path=parameter_path,
        real_document_loader=lambda: _json_object(
            args.real_evaluation, "real evaluation"
        ),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_fresh_json(result_path, result)
    compact = {
        "status": result["status"],
        "output": str(result_path),
        "fusion_parameters": str(parameter_path),
        "siamese_weight": model.siamese_weight,
        "exact_weight": model.exact_weight,
        "populations": {
            name: {
                "common_valid_count": row["common_valid_count"],
                "methods": {
                    method: {
                        "row": metrics["row"],
                        "cluster_balanced": metrics["cluster_balanced"],
                    }
                    for method, metrics in row["methods"].items()
                },
                "paired_cluster_bootstrap": row["paired_cluster_bootstrap"],
            }
            for name, row in result["populations"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "AlignedSyntheticCalibration",
    "BALANCED_POPULATION",
    "EXACT_METHOD",
    "EXACT_PILOT_FUSION_VERSION",
    "ExactPilotFusionError",
    "FUSION_METHOD",
    "FrozenExactSiameseFusion",
    "SIAMESE_METHOD",
    "STRICT_POPULATION",
    "align_synthetic_calibration",
    "apply_exact_siamese_fusion",
    "evaluate_frozen_fusion_on_real",
    "fit_exact_siamese_fusion",
    "freeze_parameters_then_open_real",
    "main",
    "score_exact_synthetic_validation",
]


if __name__ == "__main__":
    raise SystemExit(main())

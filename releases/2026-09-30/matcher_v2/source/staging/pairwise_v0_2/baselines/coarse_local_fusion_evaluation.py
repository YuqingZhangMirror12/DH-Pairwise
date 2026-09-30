#!/usr/bin/env python3
"""Calibration-only fusion of matched whole-mask and LOCAL-Q1 scores.

For each of the four local contour arms, this evaluator fits one bounded
coarse mixing weight plus an intercept on ``validation_calibration`` only.
The inputs are standardized probability logits from that local arm and the
matched Route-A whole-mask MobileNetV2 Siamese.  Parameters are frozen before
``validation_report`` or the strict real-Dunhuang score document is opened for
metrics.  No direction, RGB, text, bounding-box, or rotation feature enters
the fusion.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np
import torch
from torch import nn

from staging.pairwise_v0_2.baselines.historical_mm_siamese import (
    load_historical_mm_checkpoint,
    score_historical_mm_mask_pairs,
)
from staging.pairwise_v0_2.baselines.matched_route_a_siamese import (
    MATCHED_ROUTE_A_SIAMESE_ID,
)
from staging.pairwise_v0_2.baselines.mm_validation_comparison import (
    FourArmRunWinnerLoader,
    LoadedArmWinner,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise
from staging.pairwise_v0_2.training.local_q1_cache_builder import reopen_local_q1_cache
from staging.pairwise_v0_2.training.local_q1_provider import (
    FrozenLocalQ1BatchPlan,
    LocalQ1ReadOnlyBatchProvider,
    reopen_local_q1_batch_plan,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    EvidenceMode,
    PredictionBatch,
    PreparedAblationBatch,
    record_sequence_fingerprint,
)


COARSE_LOCAL_FUSION_VERSION = "dunhuang-coarse-local-fusion-evaluation/0.1"
CALIBRATION_PHASE = "validation_calibration"
REPORT_PHASE = "validation_report"
REAL_PHASE = "real_dunhuang_strict_alpha"
_EPSILON = 1e-6
_WEIGHT_GRID = np.linspace(0.0, 1.0, 101, dtype=np.float64)
_FOUR_ARMS = (
    AblationArmName.LOCAL_DUAL_SOFTMAX,
    AblationArmName.LOCAL_DUSTBIN_SINKHORN,
    AblationArmName.KEYPOINT_DUAL_SOFTMAX,
    AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
)
LOCAL_METHOD_IDS = tuple(name.value for name in _FOUR_ARMS)


class CoarseLocalFusionError(ValueError):
    """Raised when the calibration-only fusion contract cannot be met."""


class _Plan(Protocol):
    def phase_batches(self, phase: str) -> Tuple[Mapping[str, Any], ...]: ...


class _Provider(Protocol):
    def planned_records(
        self, phase: str, batch_ordinal: int
    ) -> Tuple[TrainingPairRecord, ...]: ...

    def prepare(
        self,
        records: Sequence[TrainingPairRecord],
        *,
        arm: AblationArm,
        phase: str,
    ) -> PreparedAblationBatch: ...


@dataclass(frozen=True)
class PairScoreTable:
    """Aligned labels, identities, and native score vectors for one phase."""

    phase: str
    pair_ids: Tuple[str, ...]
    labels: np.ndarray
    cluster_ids: Tuple[str, ...]
    dataset_ids: Tuple[str, ...]
    probability: Mapping[str, np.ndarray]
    valid: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        pair_ids = tuple(self.pair_ids)
        cluster_ids = tuple(self.cluster_ids)
        dataset_ids = tuple(self.dataset_ids)
        count = len(pair_ids)
        if not self.phase or count == 0 or len(set(pair_ids)) != count:
            raise CoarseLocalFusionError("score-table pair identity is invalid")
        if len(cluster_ids) != count or len(dataset_ids) != count:
            raise CoarseLocalFusionError("score-table metadata cardinality changed")
        if any(not value for value in pair_ids + cluster_ids + dataset_ids):
            raise CoarseLocalFusionError("score-table identity contains an empty value")
        labels = np.asarray(self.labels, dtype=np.bool_).reshape(-1).copy()
        if labels.shape != (count,):
            raise CoarseLocalFusionError("score-table labels must be bool [N]")
        probability = {}
        valid = {}
        if not self.probability or set(self.probability) != set(self.valid):
            raise CoarseLocalFusionError("score-table method coverage differs")
        for method in self.probability:
            scores = np.asarray(self.probability[method], dtype=np.float64).reshape(-1)
            usable = np.asarray(self.valid[method], dtype=np.bool_).reshape(-1)
            if scores.shape != (count,) or usable.shape != (count,):
                raise CoarseLocalFusionError("score-table method shape changed")
            if np.any(usable & (~np.isfinite(scores))):
                raise CoarseLocalFusionError("valid score-table probability is nonfinite")
            if np.any(usable & ((scores < 0.0) | (scores > 1.0))):
                raise CoarseLocalFusionError("valid score-table probability is outside [0,1]")
            scores = scores.copy()
            usable = usable.copy()
            scores.setflags(write=False)
            usable.setflags(write=False)
            probability[str(method)] = scores
            valid[str(method)] = usable
        labels.setflags(write=False)
        object.__setattr__(self, "pair_ids", pair_ids)
        object.__setattr__(self, "cluster_ids", cluster_ids)
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "probability", MappingProxyType(probability))
        object.__setattr__(self, "valid", MappingProxyType(valid))

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)


@dataclass(frozen=True)
class CoarseLocalFusionModel:
    """Frozen calibration-only parameters for one local/coarse pair."""

    local_method: str
    fused_method: str
    local_logit_mean: float
    local_logit_scale: float
    coarse_logit_mean: float
    coarse_logit_scale: float
    coarse_weight: float
    intercept: float
    calibration_pair_count: int
    calibration_positive_count: int
    calibration_negative_count: int
    calibration_weighted_log_loss: float

    def __post_init__(self) -> None:
        numeric = (
            self.local_logit_mean,
            self.local_logit_scale,
            self.coarse_logit_mean,
            self.coarse_logit_scale,
            self.coarse_weight,
            self.intercept,
            self.calibration_weighted_log_loss,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise CoarseLocalFusionError("fusion parameter is nonfinite")
        if self.local_logit_scale <= 0.0 or self.coarse_logit_scale <= 0.0:
            raise CoarseLocalFusionError("fusion logit scale must be positive")
        if not 0.0 <= self.coarse_weight <= 1.0:
            raise CoarseLocalFusionError("coarse fusion weight must be in [0,1]")
        if (
            self.calibration_pair_count <= 0
            or self.calibration_positive_count <= 0
            or self.calibration_negative_count <= 0
            or self.calibration_positive_count + self.calibration_negative_count
            != self.calibration_pair_count
        ):
            raise CoarseLocalFusionError("fusion calibration counts are invalid")

    def portable_dict(self) -> Dict[str, Any]:
        return {
            "local_method": self.local_method,
            "coarse_method": MATCHED_ROUTE_A_SIAMESE_ID,
            "fused_method": self.fused_method,
            "formula": (
                "sigmoid(intercept+(1-coarse_weight)*z_local+"
                "coarse_weight*z_coarse)"
            ),
            "local_logit_mean": self.local_logit_mean,
            "local_logit_scale": self.local_logit_scale,
            "coarse_logit_mean": self.coarse_logit_mean,
            "coarse_logit_scale": self.coarse_logit_scale,
            "coarse_weight": self.coarse_weight,
            "local_weight": 1.0 - self.coarse_weight,
            "intercept": self.intercept,
            "calibration_pair_count": self.calibration_pair_count,
            "calibration_positive_count": self.calibration_positive_count,
            "calibration_negative_count": self.calibration_negative_count,
            "calibration_weighted_log_loss": self.calibration_weighted_log_loss,
            "fit_phase": CALIBRATION_PHASE,
            "fit_weighting": "equal_dataset_then_equal_cluster_then_equal_row",
            "fit_objective": "weighted_binary_log_loss",
            "tie_break": "lower_coarse_weight",
            "probability_logit_clip_epsilon": _EPSILON,
            "weight_grid": {
                "minimum": 0.0,
                "maximum": 1.0,
                "count": len(_WEIGHT_GRID),
            },
            "direction_features_used": False,
        }


def fused_method_id(local_method: str) -> str:
    if local_method not in LOCAL_METHOD_IDS:
        raise ValueError("unknown LOCAL-Q1 fusion arm")
    return "fusion/{}+matched-whole-mask/0.1".format(local_method)


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), _EPSILON, 1.0 - _EPSILON)
    return np.log(value) - np.log1p(-value)


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(logit, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-bounded))


def _fit_weights(table: PairScoreTable, usable: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(usable)
    datasets = np.asarray(table.dataset_ids, dtype=object)[indices]
    clusters = np.asarray(table.cluster_ids, dtype=object)[indices]
    weights = np.zeros(len(indices), dtype=np.float64)
    unique_datasets = sorted(set(str(value) for value in datasets))
    for dataset in unique_datasets:
        dataset_indices = np.flatnonzero(datasets == dataset)
        cluster_values = sorted(set(str(clusters[index]) for index in dataset_indices))
        for cluster in cluster_values:
            members = dataset_indices[clusters[dataset_indices] == cluster]
            weights[members] = (
                1.0 / len(unique_datasets) / len(cluster_values) / len(members)
            )
    if not np.all(weights > 0.0):
        raise CoarseLocalFusionError("calibration weighting left an unweighted row")
    return weights * (len(weights) / weights.sum())


def _weighted_standardize(
    values: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    total = float(weights.sum())
    mean = float(np.dot(weights, values) / total)
    variance = float(np.dot(weights, (values - mean) ** 2) / total)
    scale = math.sqrt(max(variance, 0.0))
    if scale < 1e-6:
        scale = 1.0
    return (values - mean) / scale, mean, scale


def _optimal_intercept(
    base_logit: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    low = -40.0
    high = 40.0
    target = labels.astype(np.float64)
    for _ in range(80):
        middle = (low + high) / 2.0
        derivative = float(np.dot(weights, _sigmoid(base_logit + middle) - target))
        if derivative > 0.0:
            high = middle
        else:
            low = middle
    return (low + high) / 2.0


def _weighted_log_loss(
    logit: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    target = labels.astype(np.float64)
    losses = np.logaddexp(0.0, logit) - target * logit
    return float(np.dot(weights, losses) / weights.sum())


def fit_coarse_local_fusions(
    calibration: PairScoreTable,
) -> Mapping[str, CoarseLocalFusionModel]:
    """Fit all four models using calibration rows and no later-phase values."""

    if calibration.phase != CALIBRATION_PHASE:
        raise CoarseLocalFusionError("fusion fitting requires validation_calibration")
    required = set(LOCAL_METHOD_IDS) | {MATCHED_ROUTE_A_SIAMESE_ID}
    if not required.issubset(calibration.probability):
        raise CoarseLocalFusionError("calibration score table lacks a fusion source")
    models = {}
    for local_method in LOCAL_METHOD_IDS:
        usable = (
            calibration.valid[local_method]
            & calibration.valid[MATCHED_ROUTE_A_SIAMESE_ID]
            & np.isfinite(calibration.probability[local_method])
            & np.isfinite(calibration.probability[MATCHED_ROUTE_A_SIAMESE_ID])
        )
        labels = calibration.labels[usable]
        if len(labels) == 0 or not labels.any() or labels.all():
            raise CoarseLocalFusionError(
                local_method + " calibration intersection lacks both classes"
            )
        weights = _fit_weights(calibration, usable)
        local_raw = _logit(calibration.probability[local_method][usable])
        coarse_raw = _logit(
            calibration.probability[MATCHED_ROUTE_A_SIAMESE_ID][usable]
        )
        local_z, local_mean, local_scale = _weighted_standardize(local_raw, weights)
        coarse_z, coarse_mean, coarse_scale = _weighted_standardize(
            coarse_raw, weights
        )
        best = None
        for coarse_weight in _WEIGHT_GRID:
            base = (1.0 - coarse_weight) * local_z + coarse_weight * coarse_z
            intercept = _optimal_intercept(base, labels, weights)
            loss = _weighted_log_loss(base + intercept, labels, weights)
            key = (loss, float(coarse_weight))
            if best is None or key < best[0]:
                best = (key, float(coarse_weight), intercept, loss)
        if best is None:  # pragma: no cover - nonempty fixed grid
            raise CoarseLocalFusionError("fusion weight selection failed")
        _key, coarse_weight, intercept, loss = best
        model = CoarseLocalFusionModel(
            local_method=local_method,
            fused_method=fused_method_id(local_method),
            local_logit_mean=local_mean,
            local_logit_scale=local_scale,
            coarse_logit_mean=coarse_mean,
            coarse_logit_scale=coarse_scale,
            coarse_weight=coarse_weight,
            intercept=intercept,
            calibration_pair_count=len(labels),
            calibration_positive_count=int(labels.sum()),
            calibration_negative_count=int((~labels).sum()),
            calibration_weighted_log_loss=loss,
        )
        models[local_method] = model
    return MappingProxyType(models)


def apply_coarse_local_fusions(
    table: PairScoreTable,
    models: Mapping[str, CoarseLocalFusionModel],
) -> Tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    """Apply already-frozen fusion parameters without reading table labels."""

    if set(models) != set(LOCAL_METHOD_IDS):
        raise CoarseLocalFusionError("exactly four frozen fusion models are required")
    probability = {}
    valid = {}
    for local_method in LOCAL_METHOD_IDS:
        model = models[local_method]
        if model.local_method != local_method:
            raise CoarseLocalFusionError("fusion model/local method identity changed")
        usable = (
            table.valid[local_method]
            & table.valid[MATCHED_ROUTE_A_SIAMESE_ID]
            & np.isfinite(table.probability[local_method])
            & np.isfinite(table.probability[MATCHED_ROUTE_A_SIAMESE_ID])
        )
        scores = np.full(table.pair_count, np.nan, dtype=np.float64)
        local_z = (
            _logit(table.probability[local_method][usable])
            - model.local_logit_mean
        ) / model.local_logit_scale
        coarse_z = (
            _logit(table.probability[MATCHED_ROUTE_A_SIAMESE_ID][usable])
            - model.coarse_logit_mean
        ) / model.coarse_logit_scale
        fused_logit = (
            model.intercept
            + (1.0 - model.coarse_weight) * local_z
            + model.coarse_weight * coarse_z
        )
        scores[usable] = _sigmoid(fused_logit)
        scores.setflags(write=False)
        usable = usable.copy()
        usable.setflags(write=False)
        probability[model.fused_method] = scores
        valid[model.fused_method] = usable
    return MappingProxyType(probability), MappingProxyType(valid)


def _metric_row(
    probability: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    clusters: Sequence[str],
) -> Mapping[str, Any]:
    serial = [float(value) if usable else 0.5 for value, usable in zip(probability, valid)]
    return evaluate_pairwise(
        probability=serial,
        label=labels.tolist(),
        valid=valid.tolist(),
        cluster_id=list(clusters),
        threshold=0.5,
    )


def evaluate_frozen_fusions(
    table: PairScoreTable,
    models: Mapping[str, CoarseLocalFusionModel],
) -> Mapping[str, Any]:
    """Evaluate source and fused methods after parameters are frozen."""

    fused_probability, fused_valid = apply_coarse_local_fusions(table, models)
    method_probability = {
        MATCHED_ROUTE_A_SIAMESE_ID: table.probability[MATCHED_ROUTE_A_SIAMESE_ID],
        **{method: table.probability[method] for method in LOCAL_METHOD_IDS},
        **dict(fused_probability),
    }
    method_valid = {
        MATCHED_ROUTE_A_SIAMESE_ID: table.valid[MATCHED_ROUTE_A_SIAMESE_ID],
        **{method: table.valid[method] for method in LOCAL_METHOD_IDS},
        **dict(fused_valid),
    }
    common_valid = np.ones(table.pair_count, dtype=np.bool_)
    for method in method_probability:
        common_valid &= method_valid[method] & np.isfinite(method_probability[method])
    common_labels = table.labels[common_valid]
    if len(common_labels) == 0 or not common_labels.any() or common_labels.all():
        raise CoarseLocalFusionError(
            table.phase + " common-valid population lacks both classes"
        )
    methods = {}
    fused_ids = {model.fused_method for model in models.values()}
    for method, scores in method_probability.items():
        usable = method_valid[method]
        methods[method] = {
            "kind": "calibration_frozen_fusion" if method in fused_ids else "source",
            "native_valid_count": int(usable.sum()),
            "native_metrics": _metric_row(
                scores, table.labels, usable, table.cluster_ids
            ),
            "common_valid_metrics": _metric_row(
                scores, table.labels, common_valid, table.cluster_ids
            ),
            "probability": [
                float(score) if valid else None
                for score, valid in zip(scores.tolist(), usable.tolist())
            ],
            "valid": usable.tolist(),
        }
    return {
        "phase": table.phase,
        "pair_count": table.pair_count,
        "positive_count": int(table.labels.sum()),
        "negative_count": int((~table.labels).sum()),
        "common_valid_count": int(common_valid.sum()),
        "pair_ids": list(table.pair_ids),
        "labels": table.labels.tolist(),
        "cluster_ids": list(table.cluster_ids),
        "dataset_ids": list(table.dataset_ids),
        "common_valid": common_valid.tolist(),
        "parameters_frozen_before_label_metrics": True,
        "methods": methods,
    }


def build_fusion_evaluation(
    *,
    calibration: PairScoreTable,
    models: Mapping[str, CoarseLocalFusionModel],
    report: PairScoreTable,
    real: PairScoreTable,
) -> Mapping[str, Any]:
    """Build final report without fitting from report or real data."""

    if calibration.phase != CALIBRATION_PHASE or report.phase != REPORT_PHASE:
        raise CoarseLocalFusionError("fusion phase roles are incorrect")
    if real.phase != REAL_PHASE:
        raise CoarseLocalFusionError("fusion real-test role is incorrect")
    if set(calibration.pair_ids) & set(report.pair_ids):
        raise CoarseLocalFusionError("calibration and report pair IDs overlap")
    return {
        "schema_version": COARSE_LOCAL_FUSION_VERSION,
        "status": "complete_calibration_frozen_report_and_real_fusion",
        "fusion_scope": {
            "coarse_method": MATCHED_ROUTE_A_SIAMESE_ID,
            "local_methods": list(LOCAL_METHOD_IDS),
            "direction_features_used": False,
            "rgb_text_bbox_rotation_features_used": False,
        },
        "no_leakage": {
            "parameter_fit_reads_only": CALIBRATION_PHASE,
            "validation_report_labels_used_for": "metrics_after_parameters_frozen_only",
            "real_labels_used_for": "metrics_after_parameters_frozen_only",
            "calibration_report_pair_ids_disjoint": True,
        },
        "calibration": {
            "phase": calibration.phase,
            "pair_count": calibration.pair_count,
            "models": {
                local: models[local].portable_dict() for local in LOCAL_METHOD_IDS
            },
        },
        "validation_report": evaluate_frozen_fusions(report, models),
        "real_dunhuang": evaluate_frozen_fusions(real, models),
    }


def fit_and_evaluate_coarse_local_fusions(
    *,
    calibration: PairScoreTable,
    report: PairScoreTable,
    real: PairScoreTable,
) -> Mapping[str, Any]:
    """Fit from calibration only, then evaluate two untouched score tables."""

    models = fit_coarse_local_fusions(calibration)
    return build_fusion_evaluation(
        calibration=calibration,
        models=models,
        report=report,
        real=real,
    )


def score_route_a_phase(
    *,
    phase: str,
    plan: _Plan,
    provider: _Provider,
    arm_winner_loader: Callable[[AblationArmName], LoadedArmWinner],
    matched_model: nn.Module,
    matched_mask_loader: Callable[[Any], np.ndarray],
    matched_batch_size: int = 256,
) -> PairScoreTable:
    """Replay one frozen phase through the four winners and matched coarse model."""

    if phase not in {CALIBRATION_PHASE, REPORT_PHASE}:
        raise ValueError("only calibration/report phases may be scored")
    entries = tuple(plan.phase_batches(phase))
    if not entries:
        raise CoarseLocalFusionError(phase + " has no frozen batches")
    batches = []
    records = []
    for expected_ordinal, entry in enumerate(entries):
        ordinal = int(entry.get("ordinal", -1))
        if ordinal != expected_ordinal:
            raise CoarseLocalFusionError(phase + " batch ordinals are not contiguous")
        batch = tuple(provider.planned_records(phase, ordinal))
        if not batch:
            raise CoarseLocalFusionError(phase + " contains an empty batch")
        batches.append(batch)
        records.extend(batch)
    population = tuple(records)
    pair_ids = tuple(record.pair_id for record in population)
    if len(set(pair_ids)) != len(pair_ids):
        raise CoarseLocalFusionError(phase + " repeats a pair ID")

    matched_probability = score_historical_mm_mask_pairs(
        matched_model,
        (
            (
                matched_mask_loader(record.fragment_a),
                matched_mask_loader(record.fragment_b),
            )
            for record in population
        ),
        device=(
            next(matched_model.parameters(), torch.empty(0)).device
            if isinstance(matched_model, nn.Module)
            else torch.device("cpu")
        ),
        batch_size=matched_batch_size,
    ).numpy()
    probability: Dict[str, np.ndarray] = {
        MATCHED_ROUTE_A_SIAMESE_ID: matched_probability
    }
    valid: Dict[str, np.ndarray] = {
        MATCHED_ROUTE_A_SIAMESE_ID: np.ones(len(population), dtype=np.bool_)
    }
    prepared_identity: Dict[Tuple[str, int], Tuple[str, str, str]] = {}
    for name in _FOUR_ARMS:
        loaded = arm_winner_loader(name)
        if not isinstance(loaded, LoadedArmWinner) or loaded.arm.name is not name:
            raise CoarseLocalFusionError("winner loader returned the wrong arm")
        arm_probability = []
        arm_valid = []
        for ordinal, batch in enumerate(batches):
            prepared = provider.prepare(batch, arm=loaded.arm, phase=phase)
            if not isinstance(prepared, PreparedAblationBatch):
                raise CoarseLocalFusionError("provider returned an invalid batch")
            if prepared.record_sequence_sha256 != record_sequence_fingerprint(batch):
                raise CoarseLocalFusionError("provider changed frozen phase order")
            identity = (
                prepared.record_sequence_sha256,
                prepared.prepared_input_sha256,
                str(prepared.local_candidate_sha256),
            )
            key = (prepared.candidate_representation, ordinal)
            previous = prepared_identity.setdefault(key, identity)
            if previous != identity:
                raise CoarseLocalFusionError(
                    "dual-softmax and Sinkhorn received different phase tensors"
                )
            prediction = loaded.session.predict_batch(
                prepared, evidence=EvidenceMode.LOCAL
            )
            if not isinstance(prediction, PredictionBatch) or tuple(
                prediction.probability.shape
            ) != (len(batch),):
                raise CoarseLocalFusionError("winner phase prediction shape changed")
            arm_probability.extend(
                float(value) for value in prediction.probability.detach().cpu()
            )
            arm_valid.extend(bool(value) for value in prediction.valid.detach().cpu())
        probability[name.value] = np.asarray(arm_probability, dtype=np.float64)
        valid[name.value] = np.asarray(arm_valid, dtype=np.bool_)
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return PairScoreTable(
        phase=phase,
        pair_ids=pair_ids,
        labels=np.asarray([record.label for record in population], dtype=np.bool_),
        cluster_ids=tuple(record.component_id for record in population),
        dataset_ids=tuple(record.dataset_id for record in population),
        probability=probability,
        valid=valid,
    )


def real_score_table(document: Mapping[str, Any]) -> PairScoreTable:
    """Adapt the existing real-Dunhuang row-score output without refitting."""

    pair_ids = document.get("pair_ids")
    labels = document.get("labels")
    clusters = document.get("cluster_ids")
    methods = document.get("methods")
    if not all(isinstance(value, list) for value in (pair_ids, labels, clusters)):
        raise CoarseLocalFusionError("real score document lacks aligned identities")
    if not isinstance(methods, Mapping):
        raise CoarseLocalFusionError("real score document lacks methods")
    count = len(pair_ids)
    if document.get("pair_count") != count or len(labels) != count or len(clusters) != count:
        raise CoarseLocalFusionError("real score document cardinality changed")
    probability = {}
    valid = {}
    for method in (MATCHED_ROUTE_A_SIAMESE_ID,) + LOCAL_METHOD_IDS:
        row = methods.get(method)
        if not isinstance(row, Mapping):
            raise CoarseLocalFusionError("real score document lacks " + method)
        scores = row.get("probability")
        usable = row.get("valid")
        if not isinstance(scores, list) or not isinstance(usable, list):
            raise CoarseLocalFusionError("real method row is incomplete")
        if len(scores) != count or len(usable) != count:
            raise CoarseLocalFusionError("real method row cardinality changed")
        probability[method] = np.asarray(
            [np.nan if value is None else float(value) for value in scores],
            dtype=np.float64,
        )
        valid[method] = np.asarray(usable, dtype=np.bool_)
    return PairScoreTable(
        phase=REAL_PHASE,
        pair_ids=tuple(str(value) for value in pair_ids),
        labels=np.asarray(labels, dtype=np.bool_),
        cluster_ids=tuple(str(value) for value in clusters),
        dataset_ids=(str(document.get("dataset_id", "real_dunhuang")),) * count,
        probability=probability,
        valid=valid,
    )


def _json_object(path: Path, name: str) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CoarseLocalFusionError(name + " is not readable JSON") from exc
    if not isinstance(value, Mapping):
        raise CoarseLocalFusionError(name + " root must be an object")
    return value


def write_fusion_evaluation(path: Path, result: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "route_a_freeze",
        "predecessor_freeze",
        "eligibility_index",
        "eligibility_receipt",
        "route_policy",
        "route_config",
        "mm_archive",
        "eccv_archive",
        "mm_fingerprint_cache",
        "eccv_fingerprint_cache",
        "historical_split",
        "synthetic_manifest",
        "synthetic_archive",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--mm-extracted-root", type=Path)
    parser.add_argument("--eccv-extracted-root", type=Path)
    parser.add_argument("--synthetic-extracted-root", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-plan", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--matched-checkpoint", type=Path, required=True)
    parser.add_argument("--matched-batch-size", type=int, default=256)
    parser.add_argument("--max-local-tensor-elements", type=int)
    parser.add_argument("--real-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from staging.pairwise_v0_2.training import local_q1_route_a_research as research

    args = _parser().parse_args(argv)
    inputs = research.RouteAResearchInputs(
        route_a_freeze=args.route_a_freeze,
        predecessor_freeze=args.predecessor_freeze,
        eligibility_index=args.eligibility_index,
        eligibility_receipt=args.eligibility_receipt,
        route_policy=args.route_policy,
        route_config=args.route_config,
        mm_archive=args.mm_archive,
        eccv_archive=args.eccv_archive,
        mm_fingerprint_cache=args.mm_fingerprint_cache,
        eccv_fingerprint_cache=args.eccv_fingerprint_cache,
        historical_split=args.historical_split,
        synthetic_manifest=args.synthetic_manifest,
        synthetic_archive=args.synthetic_archive,
        mm_extracted_root=args.mm_extracted_root,
        eccv_extracted_root=args.eccv_extracted_root,
        synthetic_extracted_root=args.synthetic_extracted_root,
    )
    population = research.rebuild_route_a_population(inputs)
    population = research.population_for_existing_cache(population, args.cache_dir)
    config = research.planning_config_with_local_tensor_bound(
        research._planning_config_from_cache(args.cache_dir),  # noqa: SLF001
        args.max_local_tensor_elements,
    )
    trust = research.automatic_cache_trust(population, args.cache_dir)
    loader_factory = research.research_loader_factory(inputs)
    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=loader_factory,
        output_dir=args.cache_dir,
        trust=trust,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
        replay_source_masks=False,
    )
    _value, plan_file, plan_content = research._load_receipt(  # noqa: SLF001
        args.batch_plan
    )
    external_locks = research._external_locks(  # noqa: SLF001
        trust,
        config,
        plan_file_sha256=plan_file,
        plan_content_sha256=plan_content,
    )
    plan: FrozenLocalQ1BatchPlan = reopen_local_q1_batch_plan(
        args.batch_plan, external_locks=external_locks
    )
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=loader_factory,
        plan=plan,
        geometry_config=config.geometry_batch_config,
        inventory_config=config.inventory_config,
        external_locks=external_locks,
    )
    winner_loader = FourArmRunWinnerLoader(
        args.run_directory, plan, device=args.device
    )
    matched_model = load_historical_mm_checkpoint(
        args.matched_checkpoint, device=args.device
    )
    mask_loader = loader_factory()
    try:
        # This ordering is deliberate: fusion parameters are frozen before the
        # report phase is scored or the real score document is opened.
        calibration = score_route_a_phase(
            phase=CALIBRATION_PHASE,
            plan=plan,
            provider=provider,
            arm_winner_loader=winner_loader,
            matched_model=matched_model,
            matched_mask_loader=mask_loader,
            matched_batch_size=args.matched_batch_size,
        )
        models = fit_coarse_local_fusions(calibration)
        report = score_route_a_phase(
            phase=REPORT_PHASE,
            plan=plan,
            provider=provider,
            arm_winner_loader=winner_loader,
            matched_model=matched_model,
            matched_mask_loader=mask_loader,
            matched_batch_size=args.matched_batch_size,
        )
        real = real_score_table(
            _json_object(args.real_evaluation, "real-Dunhuang evaluation")
        )
        result = build_fusion_evaluation(
            calibration=calibration,
            models=models,
            report=report,
            real=real,
        )
    finally:
        close = getattr(mask_loader, "close", None)
        if callable(close):
            close()
    write_fusion_evaluation(args.output, result)
    compact = {
        "status": result["status"],
        "output": str(args.output),
        "coarse_weights": {
            local: result["calibration"]["models"][local]["coarse_weight"]
            for local in LOCAL_METHOD_IDS
        },
        "validation_report": {
            model.fused_method: result["validation_report"]["methods"][
                model.fused_method
            ]["common_valid_metrics"]["row"]
            for model in models.values()
        },
        "real_dunhuang": {
            model.fused_method: result["real_dunhuang"]["methods"][
                model.fused_method
            ]["common_valid_metrics"]["row"]
            for model in models.values()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "CALIBRATION_PHASE",
    "COARSE_LOCAL_FUSION_VERSION",
    "CoarseLocalFusionError",
    "CoarseLocalFusionModel",
    "LOCAL_METHOD_IDS",
    "PairScoreTable",
    "REAL_PHASE",
    "REPORT_PHASE",
    "apply_coarse_local_fusions",
    "build_fusion_evaluation",
    "evaluate_frozen_fusions",
    "fit_and_evaluate_coarse_local_fusions",
    "fit_coarse_local_fusions",
    "fused_method_id",
    "main",
    "real_score_table",
    "score_route_a_phase",
    "write_fusion_evaluation",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""Post-pilot real evaluation of weak versus exact-seam Sinkhorn.

This is an inference-only companion to ``training.exact_seam_pilot``.  It is
deliberately independent from the frozen LOCAL-Q1 runner/provider and compares
the two pilot checkpoints.  Optionally, the same ordered PNG alpha-mask pairs
are also scored by an exact-pilot-matched whole-mask MobileNetV2 Siamese
checkpoint.  Every local batch is constructed once, then passed to weak and
exact models in sequence; the Siamese control receives only the corresponding
ordered pair of complete masks.

Real labels and bbox-derived directions live in an evaluation side table.  A
blinded copy of every record (negative label, unknown direction) is used to
construct model tensors, so neither label nor direction can affect geometry,
candidate filtering, the forward pass, or even the auxiliary eval loss.  The
side table is joined only after every requested model prediction has been
collected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from staging.pairwise_v0_2.baselines.historical_mm_siamese import (
    load_historical_mm_checkpoint,
)
from staging.pairwise_v0_2.baselines.mm_validation_comparison import LoadedArmWinner
from staging.pairwise_v0_2.baselines.real_dunhuang_balanced_distractors import (
    BALANCED_DISTRACTOR_LABEL_ORIGIN,
    build_balanced_real_pair_dataset,
)
from staging.pairwise_v0_2.baselines.real_dunhuang_evaluation import (
    DEFAULT_TARGET_LONG_SIDE,
    RealArmScorer,
    RealBatchPrediction,
    RealDunhuangEvaluationError,
    RealPairDataset,
    _DIRECTION_LABELS,
    _direction_metrics,
    _metric_row,
    load_strict_real_pair_dataset,
    prepare_real_geometry_batch,
    score_historical_mm_real_pairs,
    score_loaded_winner_direct,
    write_real_dunhuang_evaluation,
)
from staging.pairwise_v0_2.models.local_matcher import MatcherMode
from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.training.checkpoint import load_trusted_checkpoint
from staging.pairwise_v0_2.training.geometry_batch import (
    DATA_DIRECTION_TO_INDEX,
    GeometryBatchConfig,
    RaggedGeometryBatch,
)
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache
from staging.pairwise_v0_2.training.local_q1_backend import (
    ExactSeamStepConfig,
    LocalQ1Backend,
    LocalQ1BackendMode,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    EvidenceMode,
)


POST_PILOT_REAL_EVALUATION_VERSION = (
    "dunhuang-exact-seam-post-pilot-real-evaluation/0.2"
)
EXACT_SEAM_PILOT_V01 = "dunhuang-exact-seam-synthetic-pilot/0.1"
EXACT_SEAM_PILOT_V02 = "dunhuang-exact-seam-synthetic-pilot/0.2"
_V02_WINNER_POLICY = "synthetic_validation_auroc_then_auprc_then_earlier_epoch"
_V02_WINNER_POPULATION = "balanced_synthetic_validation_only"
STRICT_LABEL_ORIGIN = "real_test_v0_1_strict_manifest"
WEAK_METHOD = "weak"
EXACT_METHOD = "exact"
SIAMESE_METHOD = "siamese"
_METHODS = (WEAK_METHOD, EXACT_METHOD)
_METHOD_ARM = {
    WEAK_METHOD: AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
    EXACT_METHOD: AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM,
}


class PostPilotRealEvaluationError(RealDunhuangEvaluationError):
    """The weak/exact real-evaluation contract cannot be met."""


@dataclass(frozen=True)
class LoadedPostPilotPair:
    """The two trusted pilot checkpoints and their shared inference contract."""

    weak: LoadedArmWinner
    exact: LoadedArmWinner
    pilot_version: str
    seed: int
    exact_loss_weight: float
    checkpoint_epochs: Mapping[str, int]

    def method(self, name: str) -> LoadedArmWinner:
        if name == WEAK_METHOD:
            return self.weak
        if name == EXACT_METHOD:
            return self.exact
        raise KeyError(name)


def _json_object(path: Path) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PostPilotRealEvaluationError(
            "pilot summary is not readable JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise PostPilotRealEvaluationError("pilot summary root must be an object")
    return value


def _pilot_arm(backend: LocalQ1Backend, name: AblationArmName) -> AblationArm:
    if name not in set(_METHOD_ARM.values()):
        raise ValueError("post-pilot evaluator accepts only weak/exact arms")
    return AblationArm(
        name=name,
        evidence=EvidenceMode.LOCAL,
        matcher_mode=MatcherMode.DUSTBIN_SINKHORN.value,
        model_config=backend.model_config_for(name),
        optimizer_config=backend.optimizer_config,
        aggregation_config=backend.aggregation_config,
        arc_pooling=backend.model_template.arc_pooling,
    )


def _required_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PostPilotRealEvaluationError(name + " must be numeric")
    output = float(value)
    if not math.isfinite(output) or output <= 0.0:
        raise PostPilotRealEvaluationError(name + " must be finite and positive")
    return output


def _required_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PostPilotRealEvaluationError(name + " must be a non-negative integer")
    return int(value)


def _checkpoint_filename(row: Mapping[str, Any], method: str) -> str:
    value = row.get("path")
    if not isinstance(value, str) or not value:
        raise PostPilotRealEvaluationError(method + " checkpoint path is missing")
    name = Path(value).name
    if not name or name in {".", ".."}:
        raise PostPilotRealEvaluationError(method + " checkpoint filename is invalid")
    return name


@dataclass(frozen=True)
class _CheckpointSelection:
    expected_epochs: Mapping[str, int]
    expected_validation: Mapping[str, Optional[Mapping[str, Any]]]
    require_synthetic_validation_provenance: bool


def _metric_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PostPilotRealEvaluationError(name + " must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise PostPilotRealEvaluationError(name + " must be finite")
    return output


def _same_mapping(left: Any, right: Any) -> bool:
    return (
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and dict(left) == dict(right)
    )


def _v02_checkpoint_selection(
    summary: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    *,
    planned_epochs: int,
) -> _CheckpointSelection:
    selection = summary.get("winner_selection")
    epoch_rows = summary.get("epochs")
    epoch_checkpoints = summary.get("epoch_checkpoints")
    if (
        not isinstance(selection, Mapping)
        or not isinstance(epoch_rows, list)
        or not isinstance(epoch_checkpoints, Mapping)
    ):
        raise PostPilotRealEvaluationError("v0.2 pilot lacks winner-selection evidence")
    if selection.get("policy") != _V02_WINNER_POLICY:
        raise PostPilotRealEvaluationError("v0.2 winner policy is unsupported")
    if selection.get("population") != _V02_WINNER_POPULATION:
        raise PostPilotRealEvaluationError(
            "v0.2 winner was not selected on synthetic validation"
        )
    if selection.get("real_evaluation_accessed") is not False:
        raise PostPilotRealEvaluationError(
            "v0.2 winner selection accessed real evaluation"
        )
    winners = selection.get("winners")
    if not isinstance(winners, Mapping):
        raise PostPilotRealEvaluationError("v0.2 winners are missing")

    rows_by_epoch: Dict[int, Mapping[str, Any]] = {}
    for row in epoch_rows:
        if not isinstance(row, Mapping):
            raise PostPilotRealEvaluationError("v0.2 epoch row is invalid")
        epoch = _required_int(row.get("epoch"), "v0.2 epoch")
        if epoch < 1 or epoch > planned_epochs or epoch in rows_by_epoch:
            raise PostPilotRealEvaluationError(
                "v0.2 epoch rows are duplicate or out of range"
            )
        validation = row.get("validation")
        row_receipts = row.get("checkpoints")
        if not isinstance(validation, Mapping) or not isinstance(row_receipts, Mapping):
            raise PostPilotRealEvaluationError(
                "v0.2 epoch lacks validation or checkpoints"
            )
        for method in _METHODS:
            metrics = validation.get(method)
            receipt = row_receipts.get(method)
            if not isinstance(metrics, Mapping) or not isinstance(receipt, Mapping):
                raise PostPilotRealEvaluationError(
                    "v0.2 epoch arm evidence is incomplete"
                )
            _metric_number(metrics.get("auroc"), method + " validation AUROC")
            _metric_number(metrics.get("auprc"), method + " validation AUPRC")
            if receipt.get("epoch") != epoch:
                raise PostPilotRealEvaluationError(
                    "v0.2 epoch checkpoint receipt has the wrong epoch"
                )
            _checkpoint_filename(receipt, method)
        rows_by_epoch[epoch] = row
    if set(rows_by_epoch) != set(range(1, planned_epochs + 1)):
        raise PostPilotRealEvaluationError(
            "v0.2 epoch evidence does not cover the configured run"
        )

    for method in _METHODS:
        history = epoch_checkpoints.get(method)
        if not isinstance(history, list) or len(history) != planned_epochs:
            raise PostPilotRealEvaluationError(
                "v0.2 epoch checkpoint history is incomplete"
            )
        history_by_epoch: Dict[int, Mapping[str, Any]] = {}
        for receipt in history:
            if not isinstance(receipt, Mapping):
                raise PostPilotRealEvaluationError(
                    "v0.2 epoch checkpoint receipt is invalid"
                )
            epoch = _required_int(receipt.get("epoch"), method + " receipt epoch")
            if epoch in history_by_epoch:
                raise PostPilotRealEvaluationError(
                    "v0.2 epoch checkpoint history is duplicated"
                )
            history_by_epoch[epoch] = receipt
        if set(history_by_epoch) != set(rows_by_epoch):
            raise PostPilotRealEvaluationError(
                "v0.2 epoch checkpoint history has different epochs"
            )
        for epoch, row in rows_by_epoch.items():
            row_receipts = row["checkpoints"]
            if not _same_mapping(history_by_epoch[epoch], row_receipts[method]):
                raise PostPilotRealEvaluationError(
                    "v0.2 epoch checkpoint receipts disagree"
                )

    expected_epochs: Dict[str, int] = {}
    expected_validation: Dict[str, Optional[Mapping[str, Any]]] = {}
    ordered_rows = tuple(rows_by_epoch[index] for index in sorted(rows_by_epoch))
    for method in _METHODS:

        def rank(row: Mapping[str, Any]) -> Tuple[float, float, int]:
            metrics = row["validation"][method]
            epoch = int(row["epoch"])
            return (
                _metric_number(metrics["auroc"], method + " validation AUROC"),
                _metric_number(metrics["auprc"], method + " validation AUPRC"),
                -epoch,
            )

        computed = max(ordered_rows, key=rank)
        computed_epoch = int(computed["epoch"])
        computed_validation = computed["validation"][method]
        computed_receipt = computed["checkpoints"][method]
        winner = winners.get(method)
        top_receipt = checkpoints.get(method)
        if not isinstance(winner, Mapping) or not isinstance(top_receipt, Mapping):
            raise PostPilotRealEvaluationError("v0.2 selected checkpoint is missing")
        winner_epoch = _required_int(winner.get("epoch"), method + " winner epoch")
        if winner_epoch != computed_epoch:
            raise PostPilotRealEvaluationError(
                "v0.2 winner is not the synthetic-validation optimum"
            )
        winner_validation = winner.get("validation")
        winner_receipt = winner.get("checkpoint")
        if not _same_mapping(winner_validation, computed_validation):
            raise PostPilotRealEvaluationError(
                "v0.2 winner validation metrics disagree"
            )
        if not (
            _same_mapping(winner_receipt, computed_receipt)
            and _same_mapping(top_receipt, computed_receipt)
        ):
            raise PostPilotRealEvaluationError(
                "v0.2 winner path/receipt differs from selected checkpoint"
            )
        if computed_receipt.get("epoch") != winner_epoch:
            raise PostPilotRealEvaluationError("v0.2 winner receipt epoch disagrees")
        expected_epochs[method] = winner_epoch
        expected_validation[method] = dict(computed_validation)
    return _CheckpointSelection(
        expected_epochs=expected_epochs,
        expected_validation=expected_validation,
        require_synthetic_validation_provenance=True,
    )


def _checkpoint_selection(
    summary: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    *,
    pilot_version: str,
    planned_epochs: int,
) -> _CheckpointSelection:
    if pilot_version == EXACT_SEAM_PILOT_V01:
        return _CheckpointSelection(
            expected_epochs={method: planned_epochs for method in _METHODS},
            expected_validation={method: None for method in _METHODS},
            require_synthetic_validation_provenance=False,
        )
    if pilot_version == EXACT_SEAM_PILOT_V02:
        return _v02_checkpoint_selection(
            summary,
            checkpoints,
            planned_epochs=planned_epochs,
        )
    raise PostPilotRealEvaluationError("pilot summary version is unsupported")


def load_exact_seam_pilot_pair(
    summary_path: Path,
    *,
    device: Union[str, torch.device] = "cuda",
) -> LoadedPostPilotPair:
    """Safely load the two ``exact_seam_pilot`` checkpoints.

    The existing restricted ``weights_only=True`` loader is reused directly.
    No new authority, manifest, or hash wrapper is created.
    """

    summary_path = Path(summary_path)
    summary = _json_object(summary_path)
    pilot_version = summary.get("pilot_version")
    if not isinstance(pilot_version, str) or pilot_version not in {
        EXACT_SEAM_PILOT_V01,
        EXACT_SEAM_PILOT_V02,
    }:
        raise PostPilotRealEvaluationError("pilot summary version is unsupported")
    if summary.get("status") != "complete":
        raise PostPilotRealEvaluationError("pilot summary is not complete")
    config = summary.get("config")
    checkpoints = summary.get("checkpoints")
    if not isinstance(config, Mapping) or not isinstance(checkpoints, Mapping):
        raise PostPilotRealEvaluationError("pilot summary is incomplete")
    seed = _required_int(config.get("seed"), "pilot seed")
    planned_epochs = _required_int(config.get("epochs"), "pilot epochs")
    if planned_epochs < 1:
        raise PostPilotRealEvaluationError("pilot epochs must be positive")
    selection = _checkpoint_selection(
        summary,
        checkpoints,
        pilot_version=pilot_version,
        planned_epochs=planned_epochs,
    )
    exact_loss_weight = _required_number(
        config.get("exact_loss_weight"), "exact loss weight"
    )
    backend = LocalQ1Backend(
        device=device,
        mode=LocalQ1BackendMode.FORMAL,
        exact_seam_step_config=ExactSeamStepConfig(loss_weight=exact_loss_weight),
    )
    arms = {
        method: _pilot_arm(backend, arm_name)
        for method, arm_name in _METHOD_ARM.items()
    }
    sessions = {
        method: backend.create_session(arm, seed=seed) for method, arm in arms.items()
    }
    epochs: Dict[str, int] = {}
    for method in _METHODS:
        receipt = checkpoints.get(method)
        if not isinstance(receipt, Mapping):
            raise PostPilotRealEvaluationError(
                method + " checkpoint receipt is missing"
            )
        file_sha256 = receipt.get("file_sha256")
        content_sha256 = receipt.get("canonical_content_sha256")
        if not isinstance(file_sha256, str) or not isinstance(content_sha256, str):
            raise PostPilotRealEvaluationError(
                method + " checkpoint receipt lacks safe-loader identity"
            )
        checkpoint_path = summary_path.parent / _checkpoint_filename(receipt, method)
        loaded = load_trusted_checkpoint(
            checkpoint_path,
            sessions[method].model,
            expected_config={
                "pilot_version": pilot_version,
                "arm": dict(arms[method].model_config),
            },
            expected_file_sha256=file_sha256,
            expected_canonical_content_sha256=content_sha256,
            map_location="cpu",
            trusted=True,
        )
        epochs[method] = int(loaded["epoch"])
        if epochs[method] != selection.expected_epochs[method]:
            raise PostPilotRealEvaluationError(
                method + " loaded epoch differs from its selected checkpoint"
            )
        if selection.require_synthetic_validation_provenance:
            provenance = loaded.get("provenance")
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("synthetic_validation_only") is not True
                or provenance.get("real_evaluation_accessed") is not False
            ):
                raise PostPilotRealEvaluationError(
                    method + " winner lacks synthetic-only provenance"
                )
            if not _same_mapping(
                loaded.get("metrics"), selection.expected_validation[method]
            ):
                raise PostPilotRealEvaluationError(
                    method + " checkpoint metrics differ from winner validation"
                )
        sessions[method].model.eval()
    return LoadedPostPilotPair(
        weak=LoadedArmWinner(
            arm=arms[WEAK_METHOD],
            session=sessions[WEAK_METHOD],
            runner_mm_report={"pilot_method": WEAK_METHOD},
        ),
        exact=LoadedArmWinner(
            arm=arms[EXACT_METHOD],
            session=sessions[EXACT_METHOD],
            runner_mm_report={"pilot_method": EXACT_METHOD},
        ),
        pilot_version=pilot_version,
        seed=seed,
        exact_loss_weight=exact_loss_weight,
        checkpoint_epochs=epochs,
    )


def _blinded_record(record: TrainingPairRecord) -> TrainingPairRecord:
    """Remove evaluation supervision before any tensor/candidate construction."""

    return replace(
        record,
        label=False,
        direction_b_wrt_a=None,
        label_origin="post_pilot_real_inference_blinded",
        static_hard_negative_score=None,
        provenance={
            "alpha_mask_only": True,
            "evaluation_supervision_blinded_before_geometry": True,
        },
    )


def _chunks(values: Sequence[Any], size: int) -> Iterable[Tuple[Any, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def _model_precision(loaded: LoadedArmWinner) -> Tuple[str, str]:
    session = loaded.session
    model = getattr(session, "model", None)
    device = getattr(session, "device", None)
    if model is None or device is None:
        return "scorer-defined", "scorer-defined"
    dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    if len(dtypes) != 1:
        raise PostPilotRealEvaluationError("model uses mixed parameter precision")
    return next(iter(dtypes)), str(torch.device(device))


def _require_matched_inference_contract(
    models: LoadedPostPilotPair,
) -> Mapping[str, str]:
    weak = models.weak.arm
    exact = models.exact.arm
    if (
        weak.name is not _METHOD_ARM[WEAK_METHOD]
        or exact.name is not _METHOD_ARM[EXACT_METHOD]
    ):
        raise PostPilotRealEvaluationError("loaded pilot pair has the wrong arms")
    for key in ("architecture", "candidate_representation", "contour_keypoint_config"):
        if weak.model_config.get(key) != exact.model_config.get(key):
            raise PostPilotRealEvaluationError(
                "weak/exact inference contract differs at " + key
            )
    if (
        weak.matcher_mode != exact.matcher_mode
        or weak.aggregation_config != exact.aggregation_config
        or weak.arc_pooling != exact.arc_pooling
    ):
        raise PostPilotRealEvaluationError(
            "weak/exact matcher or aggregation contract differs"
        )
    weak_precision, weak_device = _model_precision(models.weak)
    exact_precision, exact_device = _model_precision(models.exact)
    if weak_precision != exact_precision or weak_device != exact_device:
        raise PostPilotRealEvaluationError(
            "weak/exact evaluation precision or device differs"
        )
    return {"precision": weak_precision, "device": weak_device}


def _ranking_metrics(
    probability: Sequence[float],
    labels: Sequence[bool],
    valid: Sequence[bool],
    clusters: Sequence[str],
) -> Optional[Mapping[str, Any]]:
    usable_labels = [label for label, keep in zip(labels, valid) if keep]
    if not usable_labels or not any(usable_labels) or all(usable_labels):
        return None
    metrics = _metric_row(probability, labels, valid, clusters)
    return {
        "sample_count": int(metrics["sample_count"]),
        "positive_count": int(metrics["positive_count"]),
        "negative_count": int(metrics["negative_count"]),
        "cluster_count": int(metrics["cluster_count"]),
        "row": {
            "auroc": float(metrics["row"]["auroc"]),
            "auprc": float(metrics["row"]["auprc"]),
        },
        "cluster_balanced": {
            "auroc": float(metrics["cluster_balanced"]["auroc"]),
            "auprc": float(metrics["cluster_balanced"]["auprc"]),
        },
    }


def _score_summary(
    probability: Sequence[float], valid: Sequence[bool]
) -> Mapping[str, Any]:
    values = np.asarray(
        [score for score, keep in zip(probability, valid) if keep], dtype=np.float64
    )
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _difference(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _method_delta(
    exact: Mapping[str, Any], weak: Mapping[str, Any]
) -> Mapping[str, Any]:
    exact_ranking = exact["common_valid_ranking"]
    weak_ranking = weak["common_valid_ranking"]
    ranking = None
    if exact_ranking is not None and weak_ranking is not None:
        ranking = {
            level: {
                metric: float(exact_ranking[level][metric])
                - float(weak_ranking[level][metric])
                for metric in ("auroc", "auprc")
            }
            for level in ("row", "cluster_balanced")
        }
    exact_direction = exact["direction_on_common_valid"]
    weak_direction = weak["direction_on_common_valid"]
    return {
        "native_coverage": float(exact["native_coverage"])
        - float(weak["native_coverage"]),
        "common_valid_ranking": ranking,
        "common_valid_mean_probability": _difference(
            exact["common_valid_score_summary"]["mean"],
            weak["common_valid_score_summary"]["mean"],
        ),
        "direction_accuracy_invalid_as_incorrect": _difference(
            None
            if exact_direction is None
            else exact_direction["accuracy_invalid_as_incorrect"],
            None
            if weak_direction is None
            else weak_direction["accuracy_invalid_as_incorrect"],
        ),
        "direction_accuracy_valid_only": _difference(
            None if exact_direction is None else exact_direction["accuracy_valid_only"],
            None if weak_direction is None else weak_direction["accuracy_valid_only"],
        ),
    }


def _population_metrics(
    indices: Sequence[int],
    *,
    labels: Sequence[bool],
    clusters: Sequence[str],
    direction_targets: Sequence[int],
    values: Mapping[str, Mapping[str, Sequence[Any]]],
    method_names: Sequence[str] = _METHODS,
    direction_methods: Sequence[str] = _METHODS,
) -> Mapping[str, Any]:
    selected = tuple(int(index) for index in indices)
    ordered_methods = tuple(method_names)
    if not ordered_methods or len(set(ordered_methods)) != len(ordered_methods):
        raise PostPilotRealEvaluationError("evaluation method names are invalid")
    if any(name not in values for name in ordered_methods):
        raise PostPilotRealEvaluationError("evaluation method values are missing")
    methods_with_direction = frozenset(direction_methods)
    if not methods_with_direction.issubset(ordered_methods):
        raise PostPilotRealEvaluationError("direction method names are invalid")
    selected_labels = [labels[index] for index in selected]
    selected_clusters = [clusters[index] for index in selected]
    selected_targets = [direction_targets[index] for index in selected]
    common_valid = [
        all(bool(values[name]["valid"][index]) for name in ordered_methods)
        for index in selected
    ]
    methods: Dict[str, Mapping[str, Any]] = {}
    for name in ordered_methods:
        probability = [float(values[name]["probability"][index]) for index in selected]
        native_valid = [bool(values[name]["valid"][index]) for index in selected]
        predicted = [int(values[name]["direction"][index]) for index in selected]
        if any(
            keep and not 0.0 <= score <= 1.0
            for score, keep in zip(probability, native_valid)
        ):
            raise PostPilotRealEvaluationError(
                "valid real probabilities must remain in [0, 1]"
            )
        methods[name] = {
            "native_valid_count": sum(native_valid),
            "native_coverage": sum(native_valid) / len(selected),
            "native_ranking": _ranking_metrics(
                probability,
                selected_labels,
                native_valid,
                selected_clusters,
            ),
            "common_valid_ranking": _ranking_metrics(
                probability,
                selected_labels,
                common_valid,
                selected_clusters,
            ),
            "common_valid_score_summary": _score_summary(probability, common_valid),
            "direction_on_common_valid": (
                _direction_metrics(
                    labels=selected_labels,
                    target=selected_targets,
                    valid=common_valid,
                    predicted=predicted,
                )
                if name in methods_with_direction and any(selected_labels)
                else None
            ),
        }
    result: Dict[str, Any] = {
        "pair_count": len(selected),
        "positive_count": sum(selected_labels),
        "negative_count": len(selected) - sum(selected_labels),
        "common_valid_count": sum(common_valid),
        "common_valid_coverage": sum(common_valid) / len(selected),
        "primary_metric_population": (
            "weak_intersection_exact_valid"
            if ordered_methods == _METHODS
            else "intersection_valid:" + ",".join(ordered_methods)
        ),
        "threshold_fitted_or_selected_on_real": False,
        "threshold_metrics_reported": False,
        "methods": methods,
        "exact_minus_weak": _method_delta(methods[EXACT_METHOD], methods[WEAK_METHOD]),
    }
    if SIAMESE_METHOD in methods:
        result["exact_minus_siamese"] = _method_delta(
            methods[EXACT_METHOD], methods[SIAMESE_METHOD]
        )
        result["siamese_minus_weak"] = _method_delta(
            methods[SIAMESE_METHOD], methods[WEAK_METHOD]
        )
    return result


def evaluate_exact_seam_post_pilot_real(
    *,
    dataset: RealPairDataset,
    geometry_config: GeometryBatchConfig,
    geometry_cache: GeometryArtifactCache,
    models: LoadedPostPilotPair,
    arm_scorer: RealArmScorer = score_loaded_winner_direct,
    batch_size: int = 1,
    siamese_model: Optional[nn.Module] = None,
    siamese_device: Union[str, torch.device] = "cpu",
    siamese_batch_size: int = 256,
    balanced_negative_semantics: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Score weak/exact and an optional Siamese on one ordered alpha stream."""

    if not isinstance(dataset, RealPairDataset):
        raise TypeError("dataset must be RealPairDataset")
    if not isinstance(geometry_config, GeometryBatchConfig):
        raise TypeError("geometry_config must be GeometryBatchConfig")
    if not isinstance(geometry_cache, GeometryArtifactCache):
        raise TypeError("geometry_cache must be GeometryArtifactCache")
    if not isinstance(models, LoadedPostPilotPair):
        raise TypeError("models must be LoadedPostPilotPair")
    if not callable(arm_scorer):
        raise TypeError("arm_scorer must be callable")
    if siamese_model is not None and not isinstance(siamese_model, nn.Module):
        raise TypeError("siamese_model must be a torch.nn.Module or None")
    if (
        isinstance(siamese_batch_size, bool)
        or not isinstance(siamese_batch_size, int)
        or siamese_batch_size < 1
    ):
        raise ValueError("siamese_batch_size must be a positive integer")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= geometry_config.max_batch_size
    ):
        raise ValueError("batch_size is outside the geometry config bound")
    inference = _require_matched_inference_contract(models)
    records = tuple(dataset.records)
    pair_ids = tuple(record.pair_id for record in records)
    if len(set(pair_ids)) != len(pair_ids):
        raise PostPilotRealEvaluationError("real pair IDs are not unique")
    blinded = tuple(_blinded_record(record) for record in records)
    if tuple(record.pair_id for record in blinded) != pair_ids:
        raise PostPilotRealEvaluationError("supervision blinding changed pair order")
    method_names = _METHODS + (
        (SIAMESE_METHOD,) if siamese_model is not None else tuple()
    )
    values: Dict[str, Dict[str, list]] = {
        name: {"probability": [], "valid": [], "direction": []} for name in method_names
    }
    shared_batch_count = 0
    for batch_records in _chunks(blinded, batch_size):
        prepared = prepare_real_geometry_batch(
            batch_records,
            mask_loader=dataset.mask_loader,
            geometry_config=geometry_config,
            geometry_cache=geometry_cache,
            arm=models.weak.arm,
        )
        payload = prepared.payload
        if not isinstance(payload, RaggedGeometryBatch):
            raise PostPilotRealEvaluationError("real payload is not ragged geometry")
        if (
            bool(payload.labels.any().item())
            or bool(payload.direction_target_valid.any().item())
            or payload.exact_loss_targets() is not None
        ):
            raise PostPilotRealEvaluationError(
                "evaluation supervision entered the inference payload"
            )
        if tuple(payload.sample_ids) != tuple(
            record.pair_id for record in batch_records
        ):
            raise PostPilotRealEvaluationError("prepared real pair order changed")
        # Deliberately paired: the exact same object and candidate order are
        # consumed by weak and exact before any label/direction side-table join.
        for name in _METHODS:
            prediction = arm_scorer(models.method(name), prepared)
            if not isinstance(prediction, RealBatchPrediction):
                raise PostPilotRealEvaluationError("real scorer returned wrong type")
            expected = (len(batch_records),)
            if tuple(prediction.probability.shape) != expected:
                raise PostPilotRealEvaluationError(
                    "real prediction cardinality changed"
                )
            finite = torch.isfinite(prediction.probability)
            usable = prediction.valid & finite
            best = torch.where(
                usable,
                prediction.best_direction_index,
                torch.full_like(prediction.best_direction_index, -1),
            )
            values[name]["probability"].extend(
                float(value) for value in prediction.probability
            )
            values[name]["valid"].extend(bool(value) for value in usable)
            values[name]["direction"].extend(int(value) for value in best)
        shared_batch_count += 1

    siamese_batch_count = 0
    if siamese_model is not None:
        # ``score_historical_mm_real_pairs`` exposes only an ordered binary-mask
        # stream to the model.  Dataset labels, bbox-derived directions, RGB and
        # contour-candidate tensors cannot enter through that scorer boundary.
        siamese_probability = score_historical_mm_real_pairs(
            siamese_model,
            dataset,
            device=siamese_device,
            batch_size=siamese_batch_size,
        )
        if siamese_probability.dtype != torch.float64 or tuple(
            siamese_probability.shape
        ) != (len(records),):
            raise PostPilotRealEvaluationError(
                "Siamese real prediction cardinality changed"
            )
        siamese_finite = torch.isfinite(siamese_probability)
        siamese_in_range = (siamese_probability >= 0.0) & (siamese_probability <= 1.0)
        siamese_usable = siamese_finite & siamese_in_range
        values[SIAMESE_METHOD]["probability"].extend(
            float(value) for value in siamese_probability
        )
        values[SIAMESE_METHOD]["valid"].extend(bool(value) for value in siamese_usable)
        values[SIAMESE_METHOD]["direction"].extend([-1] * len(records))
        siamese_batch_count = math.ceil(len(records) / siamese_batch_size)

    if any(len(values[name]["valid"]) != len(records) for name in method_names):
        raise PostPilotRealEvaluationError("real prediction count changed")

    # Evaluation side-table join starts here, after every paired forward.
    labels = tuple(record.label for record in records)
    clusters = tuple(record.component_id for record in records)
    true_directions = tuple(record.direction_b_wrt_a for record in records)
    direction_targets = tuple(
        -1 if value is None else int(DATA_DIRECTION_TO_INDEX[value])
        for value in true_directions
    )
    strict_indices = tuple(
        index
        for index, record in enumerate(records)
        if record.label_origin == STRICT_LABEL_ORIGIN
    )
    constructed_indices = tuple(
        index
        for index, record in enumerate(records)
        if record.label_origin == BALANCED_DISTRACTOR_LABEL_ORIGIN
    )
    if not strict_indices or len(strict_indices) + len(constructed_indices) != len(
        records
    ):
        raise PostPilotRealEvaluationError(
            "real population contains an unsupported label origin"
        )
    if constructed_indices and (
        strict_indices != tuple(range(len(strict_indices)))
        or constructed_indices != tuple(range(len(strict_indices), len(records)))
    ):
        raise PostPilotRealEvaluationError(
            "balanced population must retain strict records as its prefix"
        )
    strict_negative_indices = tuple(
        index for index in strict_indices if not labels[index]
    )
    populations: Dict[str, Mapping[str, Any]] = {
        "strict547": _population_metrics(
            strict_indices,
            labels=labels,
            clusters=clusters,
            direction_targets=direction_targets,
            values=values,
            method_names=method_names,
        )
    }
    if constructed_indices:
        populations["balanced1016"] = _population_metrics(
            tuple(range(len(records))),
            labels=labels,
            clusters=clusters,
            direction_targets=direction_targets,
            values=values,
            method_names=method_names,
        )
    strata: Dict[str, Mapping[str, Any]] = {
        "strict_manifest_negative_39": _population_metrics(
            strict_negative_indices,
            labels=labels,
            clusters=clusters,
            direction_targets=direction_targets,
            values=values,
            method_names=method_names,
        )
    }
    if constructed_indices:
        strata["constructed_distractor_469"] = _population_metrics(
            constructed_indices,
            labels=labels,
            clusters=clusters,
            direction_targets=direction_targets,
            values=values,
            method_names=method_names,
        )
    pair_rows = []
    constructed_index_set = set(constructed_indices)
    for index, record in enumerate(records):
        method_rows = {}
        for name in method_names:
            valid = bool(values[name]["valid"][index])
            direction = int(values[name]["direction"][index])
            method_rows[name] = {
                "probability": (
                    float(values[name]["probability"][index]) if valid else None
                ),
                "valid": valid,
                "predicted_direction": (
                    _DIRECTION_LABELS[direction]
                    if valid and 0 <= direction < len(_DIRECTION_LABELS)
                    else None
                ),
            }
        pair_rows.append(
            {
                "pair_id": record.pair_id,
                "label": record.label,
                "true_direction": record.direction_b_wrt_a,
                "cluster_id": record.component_id,
                "stratum": (
                    "constructed_distractor_not_gt_negative"
                    if index in constructed_index_set
                    else (
                        "strict_manifest_positive"
                        if record.label
                        else "strict_manifest_negative"
                    )
                ),
                "methods": method_rows,
                "exact_minus_weak_probability": (
                    method_rows[EXACT_METHOD]["probability"]
                    - method_rows[WEAK_METHOD]["probability"]
                    if method_rows[EXACT_METHOD]["probability"] is not None
                    and method_rows[WEAK_METHOD]["probability"] is not None
                    else None
                ),
                "exact_minus_siamese_probability": (
                    method_rows[EXACT_METHOD]["probability"]
                    - method_rows[SIAMESE_METHOD]["probability"]
                    if SIAMESE_METHOD in method_rows
                    and method_rows[EXACT_METHOD]["probability"] is not None
                    and method_rows[SIAMESE_METHOD]["probability"] is not None
                    else None
                ),
            }
        )
    result: Dict[str, Any] = {
        "schema_version": POST_PILOT_REAL_EVALUATION_VERSION,
        "status": (
            "complete_weak_vs_exact_vs_siamese_real_alpha_mask_only"
            if siamese_model is not None
            else "complete_weak_vs_exact_real_alpha_contour_only"
        ),
        "pilot": {
            "version": models.pilot_version,
            "seed": models.seed,
            "exact_loss_weight": models.exact_loss_weight,
            "checkpoint_epochs": dict(models.checkpoint_epochs),
        },
        "evaluated_pair_count": len(records),
        "evaluated_methods": list(method_names),
        "shared_prepared_batch_count": shared_batch_count,
        "siamese_inference_batch_count": siamese_batch_count,
        "geometry_config_fingerprint": geometry_config.fingerprint,
        "fairness": {
            "same_pair_order": True,
            "same_prepared_batch_object_per_pair_of_forwards": True,
            "same_tensor_and_candidate_order": True,
            "same_batch_size": True,
            "same_batch_size_scope": "weak_and_exact_local_models",
            "same_aggregation": True,
            "same_aggregation_scope": "weak_and_exact_local_models",
            "same_evaluation_precision": True,
            "same_evaluation_precision_scope": "weak_and_exact_local_models",
            "evaluation_precision": inference["precision"],
            "evaluation_device": inference["device"],
            "weak_then_exact_forward_within_each_batch": True,
            "siamese_receives_same_ordered_pair_ids": siamese_model is not None,
            "siamese_receives_same_ordered_alpha_masks": siamese_model is not None,
            "siamese_uses_local_candidate_tensor": False,
            "siamese_batch_size": (
                siamese_batch_size if siamese_model is not None else None
            ),
            "common_valid_metrics_intersect_all_evaluated_methods": True,
        },
        "preprocessing": {
            "pixel_source": "fragment_png_alpha_only_threshold_ge_128",
            "rgb_used": False,
            "text_or_ocr_used": False,
            "bbox_or_canvas_coordinate_model_input": False,
            "label_model_input_or_candidate_filter": False,
            "true_direction_model_input_or_candidate_filter": False,
            "bbox_used_after_inference_for": "positive_direction_target_only",
            "evaluation_supervision_blinded_before_geometry": True,
            "normalization": (
                "tight_alpha_crop_then_common_scale_within_case_from_alpha_"
                "fragment_dimensions_only"
            ),
            "target_long_side": dataset.target_long_side,
            "known_upright_orientation": True,
            "rotation_search": False,
            "exact_seam_targets_used_at_real_inference": False,
            "siamese_model_input": (
                "ordered_whole_tight_crop_case_common_scale_alpha_masks_"
                "PIL_bilinear_64x64_single_channel"
                if siamese_model is not None
                else None
            ),
            "siamese_bbox_direction_or_rgb_model_input": False,
        },
        "populations": populations,
        "strata": strata,
        "pairs": pair_rows,
    }
    if balanced_negative_semantics is not None:
        result["balanced_negative_semantics"] = dict(balanced_negative_semantics)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-path-receipt", type=Path, required=True)
    parser.add_argument("--main-root", type=Path)
    parser.add_argument("--supp-root", type=Path)
    parser.add_argument("--geometry-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--siamese-checkpoint",
        type=Path,
        help=(
            "optional tensor-only raw HistoricalMMSiamese state_dict trained "
            "on the exact-pilot population"
        ),
    )
    parser.add_argument("--siamese-batch-size", type=int, default=256)
    parser.add_argument(
        "--target-long-side", type=int, default=DEFAULT_TARGET_LONG_SIDE
    )
    parser.add_argument(
        "--include-balanced-1016",
        action="store_true",
        help=(
            "append 469 constructed cross-case distractors and derive both "
            "strict547 and balanced1016 results from one 1016-pair forward"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    strict = load_strict_real_pair_dataset(
        args.manifest,
        args.local_path_receipt,
        main_root=args.main_root,
        supp_root=args.supp_root,
        target_long_side=args.target_long_side,
    )
    try:
        dataset = strict
        negative_semantics = None
        if args.include_balanced_1016:
            balanced = build_balanced_real_pair_dataset(strict)
            dataset = balanced.dataset
            negative_semantics = balanced.receipt["negative_semantics"]
        models = load_exact_seam_pilot_pair(
            args.pilot_summary,
            device=args.device,
        )
        siamese_model = (
            None
            if args.siamese_checkpoint is None
            else load_historical_mm_checkpoint(
                args.siamese_checkpoint,
                device=args.device,
            )
        )
        result = evaluate_exact_seam_post_pilot_real(
            dataset=dataset,
            geometry_config=GeometryBatchConfig(),
            geometry_cache=GeometryArtifactCache(args.geometry_cache_dir),
            models=models,
            batch_size=args.batch_size,
            siamese_model=siamese_model,
            siamese_device=args.device,
            siamese_batch_size=args.siamese_batch_size,
            balanced_negative_semantics=negative_semantics,
        )
    finally:
        strict.mask_loader.close()
    write_real_dunhuang_evaluation(args.output, result)
    compact = {
        "status": result["status"],
        "output": str(args.output),
        "populations": {
            name: {
                "pair_count": row["pair_count"],
                "common_valid_count": row["common_valid_count"],
                "methods": {
                    method: {
                        "native": method_row["native_ranking"],
                        "common": method_row["common_valid_ranking"],
                    }
                    for method, method_row in row["methods"].items()
                },
                "exact_minus_weak": row["exact_minus_weak"],
                "exact_minus_siamese": row.get("exact_minus_siamese"),
            }
            for name, row in result["populations"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "EXACT_METHOD",
    "EXACT_SEAM_PILOT_V01",
    "EXACT_SEAM_PILOT_V02",
    "LoadedPostPilotPair",
    "POST_PILOT_REAL_EVALUATION_VERSION",
    "PostPilotRealEvaluationError",
    "SIAMESE_METHOD",
    "WEAK_METHOD",
    "evaluate_exact_seam_post_pilot_real",
    "load_exact_seam_pilot_pair",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

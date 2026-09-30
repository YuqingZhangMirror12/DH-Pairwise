"""Inference-only contour-corrosion follow-up for the exact-seam pilot.

The evaluator reconstructs the completed pilot's exact 200-pair gen4
validation population, then scores the weak and exact-seam Sinkhorn
checkpoints on seven fixed mask-only conditions:

* the clean masks;
* binary erosion with disk radii 2, 4, and 8 pixels; and
* one, two, and four deterministic radius-8 boundary bites.

The original pair label, parent group, and upright direction are retained in
an evaluation side table.  Blinded records are used for every geometry build,
and exact-seam targets are deliberately disabled at inference.  A clean
reproduction gate must match the final pilot AUROC/AUPRC before any result is
written.  No threshold is selected or fitted by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
import torch

from staging.pairwise_v0_1.baselines.b1_contours import (
    BoundarySaddlePolicy,
    ContourPreprocessConfig,
    ForegroundPolarity,
    preprocess_mask_contour_only,
)
from staging.pairwise_v0_2.baselines import (
    exact_seam_post_pilot_real_evaluation as post_pilot,
)
from staging.pairwise_v0_2.baselines.real_dunhuang_evaluation import (
    RealArmScorer,
    RealBatchPrediction,
    score_loaded_winner_direct,
    write_real_dunhuang_evaluation,
)
from staging.pairwise_v0_2.geometry import ContourKeypointConfig
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.exact_seam_pilot import (
    EXACT_SEAM_PILOT_VERSION,
    ExactSeamPilotConfig,
    ExactSeamPilotError,
    _DirectoryMaskLoader,
    _prepared_batch,
    build_exact_seam_pilot_population,
)
from staging.pairwise_v0_2.training.geometry_batch import (
    DATA_DIRECTION_TO_INDEX,
    KEYPOINT_REPRESENTATION,
    GeometryBatchConfig,
    GeometryBatchError,
    RaggedGeometryBatch,
    build_geometry_batch,
)
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache
from staging.pairwise_v0_2.training.local_q1_provider import (
    local_q1_prepared_digests,
)
from staging.pairwise_v0_2.training.metrics import binary_metrics
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    PreparedAblationBatch,
    record_sequence_fingerprint,
)


CORROSION_ROBUSTNESS_VERSION = "dunhuang-exact-seam-corrosion-robustness/0.1"
FIXED_PROFILE = "gen4-contour-corrosion-fixed-v1"
EXPECTED_VALIDATION_PAIR_COUNT = 200
FIXED_MAX_PAIR_COUNT = 1000
FIXED_TRAIN_PAIR_COUNT = 800
FIXED_VALIDATION_FRACTION = 0.2
WEAK_METHOD = post_pilot.WEAK_METHOD
EXACT_METHOD = post_pilot.EXACT_METHOD
_METHODS = (WEAK_METHOD, EXACT_METHOD)
_BREAK_OFFSETS = (0.0, 0.5, 0.25, 0.75)
_CONTOUR_CONFIG = ContourPreprocessConfig(
    polarity=ForegroundPolarity.BRIGHT,
    threshold=0.5,
    connectivity=4,
    saddle_policy=BoundarySaddlePolicy.FOREGROUND_4_BACKGROUND_8,
    min_component_pixels=1,
    min_contour_points=4,
)


class CorrosionRobustnessError(RuntimeError):
    """The fixed synthetic robustness comparison cannot be evaluated."""


@dataclass(frozen=True)
class CorrosionCondition:
    """One predeclared, immutable contour-corruption condition."""

    name: str
    family: str
    severity: int
    erosion_radius_px: int = 0
    break_count: int = 0
    break_radius_px: int = 0

    def __post_init__(self) -> None:
        if not self.name or self.family not in {"clean", "erosion", "break"}:
            raise ValueError("condition name/family is invalid")
        for name in (
            "severity",
            "erosion_radius_px",
            "break_count",
            "break_radius_px",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name + " must be a non-negative integer")
        if self.family == "clean":
            if (
                self.severity
                or self.erosion_radius_px
                or self.break_count
                or self.break_radius_px
            ):
                raise ValueError("clean condition cannot corrupt the mask")
        elif self.family == "erosion":
            if self.erosion_radius_px < 1 or self.break_count or self.break_radius_px:
                raise ValueError("erosion condition has inconsistent parameters")
        elif (
            self.break_count not in {1, 2, 4}
            or self.break_radius_px < 1
            or self.erosion_radius_px
        ):
            raise ValueError("break condition has inconsistent parameters")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


FIXED_CONDITIONS = (
    CorrosionCondition("clean", "clean", 0),
    CorrosionCondition("erosion_r2", "erosion", 2, erosion_radius_px=2),
    CorrosionCondition("erosion_r4", "erosion", 4, erosion_radius_px=4),
    CorrosionCondition("erosion_r8", "erosion", 8, erosion_radius_px=8),
    CorrosionCondition(
        "break_k1_r8", "break", 1, break_count=1, break_radius_px=8
    ),
    CorrosionCondition(
        "break_k2_r8", "break", 2, break_count=2, break_radius_px=8
    ),
    CorrosionCondition(
        "break_k4_r8", "break", 4, break_count=4, break_radius_px=8
    ),
)
_CONDITION_BY_NAME = {condition.name: condition for condition in FIXED_CONDITIONS}
_FAMILY_CONDITIONS = {
    "erosion": ("clean", "erosion_r2", "erosion_r4", "erosion_r8"),
    "break": ("clean", "break_k1_r8", "break_k2_r8", "break_k4_r8"),
}


@dataclass(frozen=True)
class CorrosionRobustnessConfig:
    """Paths and execution settings for the fixed post-pilot evaluation."""

    pilot_summary: Path
    mask_root: Path
    pilot_cache_dir: Path
    geometry_cache_dir: Path
    output: Path
    device: str = "cuda"
    batch_size: int = 1
    clean_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "pilot_summary",
            "mask_root",
            "pilot_cache_dir",
            "geometry_cache_dir",
            "output",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.batch_size != 1:
            raise ValueError(
                "fixed robustness evaluation requires batch_size=1 for "
                "pair-local failure accounting"
            )
        if (
            not math.isfinite(float(self.clean_tolerance))
            or not 0.0 < float(self.clean_tolerance) <= 1e-3
        ):
            raise ValueError("clean_tolerance must be finite in (0, 1e-3]")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_object(path: Path) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrosionRobustnessError("pilot summary is not readable JSON") from exc
    if not isinstance(value, Mapping):
        raise CorrosionRobustnessError("pilot summary root must be an object")
    return value


def _required_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CorrosionRobustnessError(
            "{} must be an integer >= {}".format(name, minimum)
        )
    return int(value)


def _required_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorrosionRobustnessError(name + " must be numeric")
    output = float(value)
    if not math.isfinite(output) or output <= 0.0:
        raise CorrosionRobustnessError(name + " must be finite and positive")
    return output


def _disk(radius: int) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 1, dtype=np.int64)
    rows, columns = np.meshgrid(coordinates, coordinates, indexing="ij")
    return rows**2 + columns**2 <= radius * radius


def _break_phase(fragment_id: str, seed: int) -> float:
    digest = hashlib.sha256(
        "{}\0{}\0{}".format(FIXED_PROFILE, seed, fragment_id).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _circular_distance(values: np.ndarray, target: float) -> np.ndarray:
    difference = np.abs(values - float(target)) % 1.0
    return np.minimum(difference, 1.0 - difference)


def _remove_boundary_bites(
    mask: np.ndarray,
    *,
    fragment_id: str,
    seed: int,
    count: int,
    radius: int,
) -> np.ndarray:
    result = preprocess_mask_contour_only(mask, _CONTOUR_CONFIG)
    if not result.ok or result.contour is None:
        raise CorrosionRobustnessError(
            "clean fragment cannot supply a canonical break contour: {}".format(
                result.failure_reason
            )
        )
    contour = result.contour
    arcs = np.asarray(contour.arc_fractions, dtype=np.float64)
    points = np.asarray(contour.points, dtype=np.float64)
    phase = _break_phase(fragment_id, seed)
    selected = []
    for offset in _BREAK_OFFSETS[:count]:
        target = (phase + offset) % 1.0
        selected.append(points[int(np.argmin(_circular_distance(arcs, target)))])
    output = np.asarray(mask, dtype=np.bool_).copy()
    rows, columns = output.shape
    for center_row, center_column in selected:
        row_start = max(0, int(math.floor(center_row - radius - 1)))
        row_stop = min(rows, int(math.ceil(center_row + radius + 1)))
        column_start = max(0, int(math.floor(center_column - radius - 1)))
        column_stop = min(columns, int(math.ceil(center_column + radius + 1)))
        if row_start >= row_stop or column_start >= column_stop:
            continue
        row_grid, column_grid = np.ogrid[
            row_start:row_stop, column_start:column_stop
        ]
        inside = (
            (row_grid.astype(np.float64) + 0.5 - center_row) ** 2
            + (column_grid.astype(np.float64) + 0.5 - center_column) ** 2
            <= radius * radius
        )
        view = output[row_start:row_stop, column_start:column_stop]
        view[inside] = False
    return np.ascontiguousarray(output, dtype=np.bool_)


def degrade_mask(
    mask: np.ndarray,
    *,
    fragment_id: str,
    condition: CorrosionCondition,
    seed: int,
) -> np.ndarray:
    """Apply one fixed, label-independent mask degradation."""

    value = np.asarray(mask)
    if value.ndim != 2 or value.size == 0 or value.dtype != np.bool_:
        raise TypeError("corrosion input must be a non-empty 2D bool mask")
    if not isinstance(fragment_id, str) or not fragment_id:
        raise ValueError("fragment_id is required")
    if not isinstance(condition, CorrosionCondition):
        raise TypeError("condition must be CorrosionCondition")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if condition.family == "clean":
        output = value.copy()
    elif condition.family == "erosion":
        output = ndimage.binary_erosion(
            value,
            structure=_disk(condition.erosion_radius_px),
            border_value=0,
        )
    else:
        output = _remove_boundary_bites(
            value,
            fragment_id=fragment_id,
            seed=seed,
            count=condition.break_count,
            radius=condition.break_radius_px,
        )
    output = np.ascontiguousarray(output, dtype=np.bool_)
    output.setflags(write=False)
    return output


class _ConditionMaskLoader:
    def __init__(
        self,
        base_loader: Callable[[MaskMemberRef], np.ndarray],
        *,
        condition: CorrosionCondition,
        seed: int,
    ) -> None:
        self.base_loader = base_loader
        self.condition = condition
        self.seed = seed
        self._memo: Dict[str, np.ndarray] = {}

    def __call__(self, reference: MaskMemberRef) -> np.ndarray:
        cached = self._memo.get(reference.fragment_id)
        if cached is not None:
            return cached
        output = degrade_mask(
            self.base_loader(reference),
            fragment_id=reference.fragment_id,
            condition=self.condition,
            seed=self.seed,
        )
        self._memo[reference.fragment_id] = output
        return output

    @property
    def cached_fragment_count(self) -> int:
        return len(self._memo)


def _blinded_record(record: TrainingPairRecord) -> TrainingPairRecord:
    return replace(
        record,
        label=False,
        direction_b_wrt_a=None,
        label_origin="exact_seam_corrosion_inference_blinded",
        static_hard_negative_score=None,
        provenance={
            "mask_only": True,
            "evaluation_supervision_blinded_before_geometry": True,
            "exact_seam_targets_at_inference": False,
        },
    )


def _prepare_inference_batch(
    records: Sequence[TrainingPairRecord],
    *,
    mask_loader: Callable[[MaskMemberRef], np.ndarray],
    geometry_config: GeometryBatchConfig,
    geometry_cache: GeometryArtifactCache,
    arm: AblationArm,
    condition: CorrosionCondition,
) -> PreparedAblationBatch:
    if arm.candidate_representation != KEYPOINT_REPRESENTATION:
        raise CorrosionRobustnessError(
            "robustness evaluator requires contour-keypoint checkpoints"
        )
    raw_keypoint_config = arm.model_config.get("contour_keypoint_config")
    if not isinstance(raw_keypoint_config, Mapping):
        raise CorrosionRobustnessError("pilot arm lacks keypoint configuration")
    try:
        keypoint_config = ContourKeypointConfig(**dict(raw_keypoint_config))
    except (TypeError, ValueError) as exc:
        raise CorrosionRobustnessError(
            "pilot keypoint configuration is invalid"
        ) from exc
    population = tuple(records)
    payload = build_geometry_batch(
        population,
        mask_loader,
        geometry_config,
        geometry_artifact_cache=geometry_cache,
        candidate_representation=KEYPOINT_REPRESENTATION,
        keypoint_config=keypoint_config,
        exact_seam_supervision=False,
    )
    if payload.exact_loss_targets() is not None:
        raise CorrosionRobustnessError(
            "exact seam targets entered corrosion inference"
        )
    prepared_sha, local_sha = local_q1_prepared_digests(payload)
    preprocessing_sha = _sha256(
        {
            "profile": FIXED_PROFILE,
            "condition": condition.to_dict(),
            "coarse_preprocess_mode": geometry_config.coarse_preprocess_mode,
            "coarse_output_size": list(geometry_config.coarse_output_size),
            "coarse_content_fraction": geometry_config.coarse_content_fraction,
        }
    )
    return PreparedAblationBatch(
        payload=payload,
        sample_count=len(population),
        record_sequence_sha256=record_sequence_fingerprint(population),
        prepared_input_sha256=prepared_sha,
        local_candidate_sha256=local_sha,
        coarse_preprocessing_sha256=preprocessing_sha,
        geometry_config_sha256=geometry_config.fingerprint,
        processing_counts={
            "mask_load_count": len(
                {
                    reference.fragment_id
                    for record in population
                    for reference in (record.fragment_a, record.fragment_b)
                }
            ),
            "coarse_preprocess_count": 2 * len(population),
            "geometry_build_count": 0,
            "geometry_cache_read_count": 0,
            "geometry_cache_write_count": 0,
            "local_candidate_count": payload.candidate_count,
        },
        candidate_representation=KEYPOINT_REPRESENTATION,
    )


def _invalid_prediction() -> RealBatchPrediction:
    return RealBatchPrediction(
        probability=torch.tensor([float("nan")], dtype=torch.float32),
        valid=torch.tensor([False], dtype=torch.bool),
        best_direction_index=torch.tensor([-1], dtype=torch.long),
    )


def _append_prediction(
    destination: Dict[str, list], prediction: RealBatchPrediction
) -> None:
    if tuple(prediction.probability.shape) != (1,):
        raise CorrosionRobustnessError("prediction cardinality changed")
    finite = torch.isfinite(prediction.probability)
    usable = prediction.valid & finite
    direction = torch.where(
        usable,
        prediction.best_direction_index,
        torch.full_like(prediction.best_direction_index, -1),
    )
    if usable.any().item():
        probability = float(prediction.probability.item())
        observed_direction = int(direction.item())
        if not 0.0 <= probability <= 1.0:
            raise CorrosionRobustnessError("valid probability is outside [0, 1]")
        if not 0 <= observed_direction < 4:
            raise CorrosionRobustnessError("valid direction index is outside [0, 3]")
    destination["probability"].append(float(prediction.probability.item()))
    destination["valid"].append(bool(usable.item()))
    destination["direction"].append(int(direction.item()))


def _ranking_metrics(
    probability: Sequence[float],
    labels: Sequence[bool],
    valid: Sequence[bool],
    clusters: Sequence[str],
) -> Optional[Mapping[str, Any]]:
    return post_pilot._ranking_metrics(probability, labels, valid, clusters)


def _direction_metrics(
    *,
    labels: Sequence[bool],
    targets: Sequence[int],
    valid: Sequence[bool],
    predicted: Sequence[int],
) -> Mapping[str, Any]:
    return post_pilot._direction_metrics(
        labels=labels,
        target=targets,
        valid=valid,
        predicted=predicted,
    )


def _score_summary(
    probability: Sequence[float], valid: Sequence[bool]
) -> Mapping[str, Any]:
    return post_pilot._score_summary(probability, valid)


def _operational_condition_metrics(
    condition_values: Mapping[str, Mapping[str, Sequence[Any]]],
    *,
    labels: Sequence[bool],
    clusters: Sequence[str],
    direction_targets: Sequence[int],
) -> Mapping[str, Any]:
    count = len(labels)
    common_valid = [
        all(bool(condition_values[name]["valid"][index]) for name in _METHODS)
        for index in range(count)
    ]
    methods: Dict[str, Mapping[str, Any]] = {}
    for name in _METHODS:
        values = condition_values[name]
        native_valid = [bool(value) for value in values["valid"]]
        probability = [float(value) for value in values["probability"]]
        predicted = [int(value) for value in values["direction"]]
        methods[name] = {
            "native_valid_count": sum(native_valid),
            "native_coverage": sum(native_valid) / count,
            "common_valid_ranking": _ranking_metrics(
                probability, labels, common_valid, clusters
            ),
            "common_valid_score_summary": _score_summary(
                probability, common_valid
            ),
            "direction_on_common_valid": _direction_metrics(
                labels=labels,
                targets=direction_targets,
                valid=common_valid,
                predicted=predicted,
            ),
        }
    return {
        "pair_count": count,
        "positive_count": sum(labels),
        "negative_count": count - sum(labels),
        "common_valid_count": sum(common_valid),
        "common_valid_coverage": sum(common_valid) / count,
        "primary_metric_population": "weak_intersection_exact_valid_at_condition",
        "threshold_fitted_or_selected": False,
        "threshold_metrics_reported": False,
        "methods": methods,
        "exact_minus_weak": post_pilot._method_delta(
            methods[EXACT_METHOD], methods[WEAK_METHOD]
        ),
    }


def _fixed_intersection_curve(
    family: str,
    values: Mapping[str, Mapping[str, Mapping[str, Sequence[Any]]]],
    *,
    labels: Sequence[bool],
    clusters: Sequence[str],
    direction_targets: Sequence[int],
) -> Mapping[str, Any]:
    names = _FAMILY_CONDITIONS[family]
    fixed = [
        all(
            bool(values[condition][method]["valid"][index])
            for condition in names
            for method in _METHODS
        )
        for index in range(len(labels))
    ]
    selected = [index for index, keep in enumerate(fixed) if keep]
    selected_labels = [labels[index] for index in selected]
    selected_clusters = [clusters[index] for index in selected]
    selected_targets = [direction_targets[index] for index in selected]
    points = []
    for condition_name in names:
        condition = _CONDITION_BY_NAME[condition_name]
        methods: Dict[str, Mapping[str, Any]] = {}
        for method in _METHODS:
            method_values = values[condition_name][method]
            probability = [
                float(method_values["probability"][index]) for index in selected
            ]
            predicted = [int(method_values["direction"][index]) for index in selected]
            all_valid = [True] * len(selected)
            methods[method] = {
                "ranking": (
                    _ranking_metrics(
                        probability,
                        selected_labels,
                        all_valid,
                        selected_clusters,
                    )
                    if selected
                    else None
                ),
                "direction": (
                    _direction_metrics(
                        labels=selected_labels,
                        targets=selected_targets,
                        valid=all_valid,
                        predicted=predicted,
                    )
                    if selected and any(selected_labels)
                    else None
                ),
                "score_summary": _score_summary(probability, all_valid),
            }
        weak_ranking = methods[WEAK_METHOD]["ranking"]
        exact_ranking = methods[EXACT_METHOD]["ranking"]
        delta = None
        if weak_ranking is not None and exact_ranking is not None:
            delta = {
                level: {
                    metric: float(exact_ranking[level][metric])
                    - float(weak_ranking[level][metric])
                    for metric in ("auroc", "auprc")
                }
                for level in ("row", "cluster_balanced")
            }
        points.append(
            {
                "condition": condition_name,
                "severity": condition.severity,
                "methods": methods,
                "exact_minus_weak_ranking": delta,
            }
        )
    return {
        "family": family,
        "condition_order": list(names),
        "fixed_intersection_definition": (
            "weak_and_exact_valid_at_every_condition_in_this_family"
        ),
        "fixed_pair_count": len(selected),
        "fixed_coverage": len(selected) / len(labels),
        "fixed_positive_count": sum(selected_labels),
        "fixed_negative_count": len(selected_labels) - sum(selected_labels),
        "points": points,
    }


def _native_clean_metrics(
    values: Mapping[str, Mapping[str, Sequence[Any]]], labels: Sequence[bool]
) -> Mapping[str, Mapping[str, float]]:
    label_tensor = torch.tensor(labels, dtype=torch.bool)
    output = {}
    for method in _METHODS:
        probability = torch.tensor(
            values[method]["probability"], dtype=torch.float64
        )
        valid = torch.tensor(values[method]["valid"], dtype=torch.bool)
        output[method] = binary_metrics(probability, label_tensor, valid)
    return output


def _clean_reproduction_gate(
    observed: Mapping[str, Mapping[str, float]],
    expected: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float,
) -> Mapping[str, Any]:
    compared = ("sample_count", "positive_count", "negative_count", "auroc", "auprc")
    differences: Dict[str, Dict[str, float]] = {}
    maximum = 0.0
    for method in _METHODS:
        expected_row = expected.get(method)
        if not isinstance(expected_row, Mapping):
            raise CorrosionRobustnessError(
                method + " clean pilot metrics are missing"
            )
        differences[method] = {}
        for metric in compared:
            expected_value = expected_row.get(metric)
            if isinstance(expected_value, bool) or not isinstance(
                expected_value, (int, float)
            ):
                raise CorrosionRobustnessError(
                    "pilot clean metric is missing: {}.{}".format(method, metric)
                )
            difference = abs(
                float(observed[method][metric]) - float(expected_value)
            )
            differences[method][metric] = difference
            maximum = max(maximum, difference)
    passed = maximum <= tolerance
    receipt = {
        "passed": passed,
        "tolerance": tolerance,
        "compared_metrics": list(compared),
        "maximum_absolute_difference": maximum,
        "absolute_differences": differences,
        "observed": {name: dict(observed[name]) for name in _METHODS},
        "expected": {name: dict(expected[name]) for name in _METHODS},
    }
    if not passed:
        raise CorrosionRobustnessError(
            "clean checkpoint validation did not reproduce the pilot summary: "
            "max_abs_difference={:.9g}".format(maximum)
        )
    return receipt


def evaluate_exact_seam_corrosion_robustness(
    *,
    records: Sequence[TrainingPairRecord],
    base_mask_loader: Callable[[MaskMemberRef], np.ndarray],
    geometry_config: GeometryBatchConfig,
    geometry_cache: GeometryArtifactCache,
    models: post_pilot.LoadedPostPilotPair,
    expected_clean_metrics: Mapping[str, Mapping[str, Any]],
    seed: int,
    arm_scorer: RealArmScorer = score_loaded_winner_direct,
    batch_size: int = 1,
    clean_tolerance: float = 1e-6,
    expected_pair_count: int = EXPECTED_VALIDATION_PAIR_COUNT,
) -> Mapping[str, Any]:
    """Score weak/exact checkpoints on the fixed seven-condition profile."""

    population = tuple(records)
    if not isinstance(geometry_config, GeometryBatchConfig):
        raise TypeError("geometry_config must be GeometryBatchConfig")
    if not isinstance(geometry_cache, GeometryArtifactCache):
        raise TypeError("geometry_cache must be GeometryArtifactCache")
    if not isinstance(models, post_pilot.LoadedPostPilotPair):
        raise TypeError("models must be LoadedPostPilotPair")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if seed != models.seed:
        raise CorrosionRobustnessError(
            "corruption seed must equal the completed pilot seed: {} != {}".format(
                seed, models.seed
            )
        )
    if len(population) != expected_pair_count:
        raise CorrosionRobustnessError(
            "validation pair count changed: {} != {}".format(
                len(population), expected_pair_count
            )
        )
    if expected_pair_count < 2 or batch_size != 1:
        raise ValueError("evaluation requires at least two pairs and batch_size=1")
    if not callable(base_mask_loader) or not callable(arm_scorer):
        raise TypeError("mask loader and arm scorer must be callable")
    if len({record.pair_id for record in population}) != len(population):
        raise CorrosionRobustnessError("validation pair IDs are not unique")
    labels = tuple(record.label for record in population)
    if set(labels) != {False, True}:
        raise CorrosionRobustnessError("validation population needs both classes")
    if expected_pair_count == EXPECTED_VALIDATION_PAIR_COUNT and (
        sum(labels) != EXPECTED_VALIDATION_PAIR_COUNT // 2
    ):
        raise CorrosionRobustnessError("fixed validation population is not 100/100")
    clusters = tuple(record.canonical_group_id for record in population)
    direction_targets = tuple(
        -1
        if record.direction_b_wrt_a is None
        else DATA_DIRECTION_TO_INDEX[record.direction_b_wrt_a]
        for record in population
    )
    blinded = tuple(_blinded_record(record) for record in population)
    if tuple(record.pair_id for record in blinded) != tuple(
        record.pair_id for record in population
    ):
        raise CorrosionRobustnessError("blinding changed validation pair order")
    inference = post_pilot._require_matched_inference_contract(models)
    values: Dict[str, Dict[str, Dict[str, list]]] = {}
    geometry_failures: Dict[str, Dict[str, int]] = {}
    cached_fragments: Dict[str, int] = {}
    shared_prepared_batch_count = 0

    def score_condition(condition: CorrosionCondition) -> None:
        nonlocal shared_prepared_batch_count
        condition_loader = _ConditionMaskLoader(
            base_mask_loader,
            condition=condition,
            seed=seed,
        )
        values[condition.name] = {
            method: {"probability": [], "valid": [], "direction": []}
            for method in _METHODS
        }
        failures: Dict[str, int] = {}
        for record in blinded:
            try:
                prepared = _prepare_inference_batch(
                    (record,),
                    mask_loader=condition_loader,
                    geometry_config=geometry_config,
                    geometry_cache=geometry_cache,
                    arm=models.weak.arm,
                    condition=condition,
                )
            except GeometryBatchError as exc:
                reason = str(exc) or type(exc).__name__
                failures[reason] = failures.get(reason, 0) + 1
                invalid = _invalid_prediction()
                for method in _METHODS:
                    _append_prediction(values[condition.name][method], invalid)
                continue
            payload = prepared.payload
            if not isinstance(payload, RaggedGeometryBatch):
                raise CorrosionRobustnessError(
                    "prepared corrosion payload is not ragged geometry"
                )
            if (
                bool(payload.labels.any().item())
                or bool(payload.direction_target_valid.any().item())
                or payload.exact_loss_targets() is not None
            ):
                raise CorrosionRobustnessError(
                    "evaluation supervision entered corrosion inference"
                )
            if tuple(payload.sample_ids) != (record.pair_id,):
                raise CorrosionRobustnessError("prepared pair order changed")
            for method in _METHODS:
                prediction = arm_scorer(models.method(method), prepared)
                if not isinstance(prediction, RealBatchPrediction):
                    raise CorrosionRobustnessError(
                        "corrosion scorer returned the wrong prediction type"
                    )
                _append_prediction(values[condition.name][method], prediction)
            shared_prepared_batch_count += 1
        geometry_failures[condition.name] = failures
        cached_fragments[condition.name] = condition_loader.cached_fragment_count

    # The clean stream is a fail-fast scientific control, not a post-hoc
    # receipt.  No corrupted mask is built and no corrupted forward is issued
    # unless the reloaded checkpoints reproduce the completed pilot metrics.
    score_condition(FIXED_CONDITIONS[0])
    observed_clean = _native_clean_metrics(values["clean"], labels)
    clean_gate = _clean_reproduction_gate(
        observed_clean,
        expected_clean_metrics,
        tolerance=clean_tolerance,
    )
    for condition in FIXED_CONDITIONS[1:]:
        score_condition(condition)
    condition_metrics = {
        condition.name: {
            "condition": condition.to_dict(),
            **_operational_condition_metrics(
                values[condition.name],
                labels=labels,
                clusters=clusters,
                direction_targets=direction_targets,
            ),
        }
        for condition in FIXED_CONDITIONS
    }
    family_curves = {
        family: _fixed_intersection_curve(
            family,
            values,
            labels=labels,
            clusters=clusters,
            direction_targets=direction_targets,
        )
        for family in _FAMILY_CONDITIONS
    }
    pair_rows = []
    for index, record in enumerate(population):
        condition_rows = {}
        for condition in FIXED_CONDITIONS:
            method_rows = {}
            for method in _METHODS:
                valid = bool(values[condition.name][method]["valid"][index])
                probability = float(
                    values[condition.name][method]["probability"][index]
                )
                direction = int(
                    values[condition.name][method]["direction"][index]
                )
                method_rows[method] = {
                    "probability": probability if valid else None,
                    "valid": valid,
                    "predicted_direction": (
                        post_pilot._DIRECTION_LABELS[direction]
                        if valid and 0 <= direction < 4
                        else None
                    ),
                }
            condition_rows[condition.name] = {
                "methods": method_rows,
                "exact_minus_weak_probability": (
                    method_rows[EXACT_METHOD]["probability"]
                    - method_rows[WEAK_METHOD]["probability"]
                    if method_rows[EXACT_METHOD]["probability"] is not None
                    and method_rows[WEAK_METHOD]["probability"] is not None
                    else None
                ),
            }
        pair_rows.append(
            {
                "pair_id": record.pair_id,
                "label": record.label,
                "true_direction": record.direction_b_wrt_a,
                "cluster_id": record.canonical_group_id,
                "conditions": condition_rows,
            }
        )
    return {
        "schema_version": CORROSION_ROBUSTNESS_VERSION,
        "status": "complete_fixed_weak_vs_exact_corrosion_robustness",
        "profile": FIXED_PROFILE,
        "pilot": {
            "version": EXACT_SEAM_PILOT_VERSION,
            "seed": models.seed,
            "corruption_seed": seed,
            "corruption_seed_equals_pilot_seed": True,
            "exact_loss_weight": models.exact_loss_weight,
            "checkpoint_epochs": dict(models.checkpoint_epochs),
        },
        "validation": {
            "pair_count": len(population),
            "positive_count": sum(labels),
            "negative_count": len(population) - sum(labels),
            "pair_order_sha256": record_sequence_fingerprint(population),
            "same_pairs_at_every_condition": True,
            "labels_and_directions_preserved_in_post_inference_side_table": True,
            "population_reconstruction": {
                "supported_run": "exact_seam_pilot_fixed_default_v0.1_only",
                "max_pairs": FIXED_MAX_PAIR_COUNT,
                "train_pairs": FIXED_TRAIN_PAIR_COUNT,
                "validation_pairs": EXPECTED_VALIDATION_PAIR_COUNT,
                "validation_fraction": FIXED_VALIDATION_FRACTION,
                "validation_fraction_source": (
                    "fixed_evaluator_constant_matching_remote_pilot_cli_not_"
                    "inferred_from_summary"
                ),
            },
        },
        "condition_order": [condition.name for condition in FIXED_CONDITIONS],
        "condition_definitions": [
            condition.to_dict() for condition in FIXED_CONDITIONS
        ],
        "clean_reproduction_gate": clean_gate,
        "fairness": {
            "same_prepared_batch_object_per_weak_exact_forward": True,
            "same_tensor_and_candidate_order_per_weak_exact_forward": True,
            "same_checkpoint_inference_contract": True,
            "same_evaluation_precision": True,
            "evaluation_precision": inference["precision"],
            "evaluation_device": inference["device"],
            "batch_size": batch_size,
            "geometry_failure_invalidates_both_methods_for_one_pair": True,
        },
        "preprocessing": {
            "mask_only": True,
            "rgb_text_or_ocr_used": False,
            "known_upright_orientation": True,
            "rotation_search": False,
            "full_canvas_shape_retained_after_corruption": True,
            "condition_depends_only_on_mask_fragment_id_and_seed": True,
            "corruption_seed": seed,
            "corruption_seed_equals_pilot_seed": True,
            "label_or_direction_used_for_corruption": False,
            "evaluation_supervision_blinded_before_geometry": True,
            "exact_seam_targets_used_at_inference": False,
            "corrosion_config_enabled": False,
            "corruption_is_applied_before_unchanged_geometry_config": True,
            "geometry_config_fingerprint": geometry_config.fingerprint,
        },
        "processing": {
            "pair_condition_count": len(population) * len(FIXED_CONDITIONS),
            "planned_weak_exact_pair_forwards": (
                2 * len(population) * len(FIXED_CONDITIONS)
            ),
            "completed_shared_prepared_batch_count": shared_prepared_batch_count,
            "cached_fragment_count_by_condition": cached_fragments,
            "geometry_failures_by_condition": geometry_failures,
        },
        "conditions": condition_metrics,
        "family_fixed_intersection_curves": family_curves,
        "pairs": pair_rows,
    }


def _pilot_summary_parts(
    summary: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    if summary.get("pilot_version") != EXACT_SEAM_PILOT_VERSION:
        raise CorrosionRobustnessError("pilot summary version is unsupported")
    if summary.get("status") != "complete":
        raise CorrosionRobustnessError("pilot summary is not complete")
    config = summary.get("config")
    epochs = summary.get("epochs")
    if not isinstance(config, Mapping) or not isinstance(epochs, list) or not epochs:
        raise CorrosionRobustnessError("pilot summary lacks config/epochs")
    final = epochs[-1]
    validation = final.get("validation") if isinstance(final, Mapping) else None
    if not isinstance(validation, Mapping):
        raise CorrosionRobustnessError("pilot final validation metrics are missing")
    expected = {}
    for method in _METHODS:
        row = validation.get(method)
        if not isinstance(row, Mapping):
            raise CorrosionRobustnessError(method + " final metrics are missing")
        expected[method] = row
    return config, expected


def _rebuild_validation_population(
    *,
    summary_path: Path,
    summary_config: Mapping[str, Any],
    mask_root: Path,
    pilot_cache: GeometryArtifactCache,
    geometry_config: GeometryBatchConfig,
) -> Tuple[Tuple[TrainingPairRecord, ...], _DirectoryMaskLoader]:
    max_pairs = _required_int(summary_config.get("max_pairs"), "max_pairs", minimum=500)
    train_pairs = _required_int(
        summary_config.get("train_pairs"),
        "train_pairs",
        minimum=2,
    )
    validation_pairs = _required_int(
        summary_config.get("validation_pairs"),
        "validation_pairs",
        minimum=2,
    )
    fixed_counts = (
        max_pairs == FIXED_MAX_PAIR_COUNT
        and train_pairs == FIXED_TRAIN_PAIR_COUNT
        and validation_pairs == EXPECTED_VALIDATION_PAIR_COUNT
    )
    if not fixed_counts:
        raise CorrosionRobustnessError(
            "corrosion evaluator v0.1 only supports the fixed default pilot "
            "run (max/train/validation=1000/800/200 and validation_fraction=0.2)"
        )
    pilot_config = ExactSeamPilotConfig(
        mask_root=mask_root,
        output_root=Path(summary_path).parent,
        cache_root=pilot_cache.root,
        max_pairs=max_pairs,
        epochs=_required_int(summary_config.get("epochs"), "epochs", minimum=1),
        batch_size=_required_int(
            summary_config.get("batch_size"), "batch_size", minimum=1
        ),
        # The remote command fixed this value explicitly.  Pilot summary v0.1
        # does not persist it, so do not infer it from any reported counts.
        validation_fraction=FIXED_VALIDATION_FRACTION,
        seed=_required_int(summary_config.get("seed"), "seed"),
        generator=str(summary_config.get("generator", "")),
        device="cuda",
        exact_loss_weight=_required_number(
            summary_config.get("exact_loss_weight"), "exact_loss_weight"
        ),
    )
    loader = _DirectoryMaskLoader(mask_root)

    def eligible(record: TrainingPairRecord) -> bool:
        try:
            _prepared_batch(
                (record,),
                loader=loader,
                cache=pilot_cache,
                geometry_config=geometry_config,
            )
        except (ExactSeamPilotError, GeometryBatchError):
            return False
        return True

    population = build_exact_seam_pilot_population(
        pilot_config,
        record_filter=eligible,
    )
    validation = tuple(population.validation_records)
    if len(validation) != EXPECTED_VALIDATION_PAIR_COUNT:
        raise CorrosionRobustnessError("reconstructed validation count changed")
    return validation, loader


def run_exact_seam_corrosion_robustness(
    config: CorrosionRobustnessConfig,
) -> Mapping[str, Any]:
    """Reconstruct the pilot population and execute the fixed GPU evaluation."""

    if not isinstance(config, CorrosionRobustnessConfig):
        raise TypeError("config must be CorrosionRobustnessConfig")
    summary = _json_object(config.pilot_summary)
    summary_config, expected_clean = _pilot_summary_parts(summary)
    models = post_pilot.load_exact_seam_pilot_pair(
        config.pilot_summary,
        device=config.device,
    )
    geometry_config = GeometryBatchConfig()
    records, loader = _rebuild_validation_population(
        summary_path=config.pilot_summary,
        summary_config=summary_config,
        mask_root=config.mask_root,
        pilot_cache=GeometryArtifactCache(config.pilot_cache_dir),
        geometry_config=geometry_config,
    )
    return evaluate_exact_seam_corrosion_robustness(
        records=records,
        base_mask_loader=loader,
        geometry_config=geometry_config,
        geometry_cache=GeometryArtifactCache(config.geometry_cache_dir),
        models=models,
        expected_clean_metrics=expected_clean,
        seed=models.seed,
        batch_size=config.batch_size,
        clean_tolerance=config.clean_tolerance,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--pilot-cache-dir", type=Path, required=True)
    parser.add_argument("--geometry-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--clean-tolerance", type=float, default=1e-6)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    config = CorrosionRobustnessConfig(
        pilot_summary=arguments.pilot_summary,
        mask_root=arguments.mask_root,
        pilot_cache_dir=arguments.pilot_cache_dir,
        geometry_cache_dir=arguments.geometry_cache_dir,
        output=arguments.output,
        device=arguments.device,
        batch_size=arguments.batch_size,
        clean_tolerance=arguments.clean_tolerance,
    )
    result = run_exact_seam_corrosion_robustness(config)
    write_real_dunhuang_evaluation(config.output, result)
    compact = {
        "status": result["status"],
        "output": str(config.output),
        "clean_reproduction_gate": result["clean_reproduction_gate"]["passed"],
        "conditions": {
            name: {
                "common_valid_count": row["common_valid_count"],
                "weak": row["methods"][WEAK_METHOD]["common_valid_ranking"],
                "exact": row["methods"][EXACT_METHOD]["common_valid_ranking"],
                "exact_minus_weak": row["exact_minus_weak"],
            }
            for name, row in result["conditions"].items()
        },
        "family_fixed_intersection_curves": result[
            "family_fixed_intersection_curves"
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "CORROSION_ROBUSTNESS_VERSION",
    "CorrosionCondition",
    "CorrosionRobustnessConfig",
    "CorrosionRobustnessError",
    "EXPECTED_VALIDATION_PAIR_COUNT",
    "FIXED_CONDITIONS",
    "FIXED_PROFILE",
    "degrade_mask",
    "evaluate_exact_seam_corrosion_robustness",
    "main",
    "run_exact_seam_corrosion_robustness",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""Pre-registered frozen Exact-Sinkhorn arclength-order readout.

``fit-synthetic`` replays the frozen Exact 5570/1394 population, verifies the
clockwise anti-monotone sign against synthetic exact correspondence targets,
extracts one fixed coherent-transport statistic, and fits only scalar beta and
intercept.  ``evaluate-real`` loads those frozen parameters before opening the
real dataset and compares baseline/readout once on strict547 and balanced1016
with paired component-cluster bootstrap intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from staging.pairwise_v0_2.baselines.exact_seam_post_pilot_real_evaluation import (
    _blinded_record,
    load_exact_seam_pilot_pair,
)
from staging.pairwise_v0_2.baselines.frozen_transport_readout import (
    FrozenTransportReadoutError,
    _concatenate,
    _exact_model,
    _real_method_metrics,
    _selected_exact_identity,
    _summary,
)
from staging.pairwise_v0_2.baselines.real_dunhuang_balanced_distractors import (
    build_balanced_real_pair_dataset,
)
from staging.pairwise_v0_2.baselines.real_dunhuang_evaluation import (
    DEFAULT_TARGET_LONG_SIDE,
    _DIRECTION_LABELS,
    load_strict_real_pair_dataset,
)
from staging.pairwise_v0_2.geometry import ContourKeypointConfig
from staging.pairwise_v0_2.models.frozen_transport_order_readout import (
    READOUT_ORDER_COHERENT,
    OrderCoordinateSidecar,
    build_order_coordinate_sidecar,
    exact_target_order_concordance,
    transport_pairwise_anti_order_mass,
)
from staging.pairwise_v0_2.models.frozen_transport_readout import (
    FrozenArcDataset,
    FrozenReadoutParameters,
    READOUT_BASELINE,
    READOUT_MASS_ONLY,
    aggregate_frozen_readout,
    fit_frozen_readout,
)
from staging.pairwise_v0_2.models.pairwise import PairwiseScoreSource
from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise
from staging.pairwise_v0_2.training.exact_seam_pilot import (
    ExactSeamPilotConfig,
    _DirectoryMaskLoader,
)
from staging.pairwise_v0_2.training.exact_seam_population_snapshot import (
    load_exact_seam_population_snapshot,
)
from staging.pairwise_v0_2.training.geometry_batch import (
    KEYPOINT_REPRESENTATION,
    GeometryBatchConfig,
    RaggedGeometryBatch,
    build_geometry_batch,
)
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache


ORDER_READOUT_VERSION = "dunhuang-frozen-transport-order-readout/0.1"
FIT_STATUS = "complete_synthetic_only_frozen_order_readout_fit"
REAL_STATUS = "complete_frozen_order_readout_real_evaluation"
READOUTS = (READOUT_BASELINE, READOUT_ORDER_COHERENT)
BOOTSTRAP_SEED = 260_833
BOOTSTRAP_REPLICATES = 20_000


def _chunks(values: Sequence[Any], size: int) -> Iterable[Tuple[Any, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def _winner_keypoint_config(models: Any) -> ContourKeypointConfig:
    arm = models.exact.arm
    if arm.candidate_representation != KEYPOINT_REPRESENTATION:
        raise FrozenTransportReadoutError(
            "Exact winner does not use contour-keypoint candidates"
        )
    keypoint_row = arm.model_config.get("contour_keypoint_config")
    if not isinstance(keypoint_row, Mapping):
        raise FrozenTransportReadoutError("Exact winner lacks keypoint config")
    try:
        config = ContourKeypointConfig(**dict(keypoint_row))
    except (TypeError, ValueError) as exc:
        raise FrozenTransportReadoutError(
            "Exact winner keypoint config is invalid"
        ) from exc
    if not config.same_scale_correspondence_only:
        raise FrozenTransportReadoutError(
            "pairwise order readout requires frozen same-scale correspondence"
        )
    return config


def _build_with_sidecar(
    records: Tuple[TrainingPairRecord, ...],
    *,
    loader: Callable,
    cache: GeometryArtifactCache,
    geometry_config: GeometryBatchConfig,
    keypoint_config: ContourKeypointConfig,
    exact_seam_supervision: bool,
) -> tuple[RaggedGeometryBatch, OrderCoordinateSidecar]:
    observed: List[Tuple[Any, ...]] = []
    payload = build_geometry_batch(
        records,
        loader,
        geometry_config,
        geometry_artifact_cache=cache,
        candidate_representation=KEYPOINT_REPRESENTATION,
        keypoint_config=keypoint_config,
        exact_seam_supervision=exact_seam_supervision,
        keypoint_candidate_observer=lambda values: observed.append(tuple(values)),
    )
    if len(observed) != 1 or len(observed[0]) != payload.candidate_count:
        raise FrozenTransportReadoutError("order-coordinate observer lost candidates")
    sidecar = build_order_coordinate_sidecar(
        observed[0],
        padded_length_a=int(payload.local_a.shape[1]),
        padded_length_b=int(payload.local_b.shape[1]),
    )
    if (
        tuple(sidecar.coordinate_a.shape) != tuple(payload.token_mask_a.shape)
        or tuple(sidecar.coordinate_b.shape) != tuple(payload.token_mask_b.shape)
    ):
        raise FrozenTransportReadoutError("order sidecar changed tensor alignment")
    return payload, sidecar


class _TargetAccumulator:
    def __init__(self) -> None:
        self.target_pair_count = [0] * 4
        self.total: Dict[int, Dict[int, List[int]]] = {
            index: {} for index in range(4)
        }
        self.off_target_groups = 0
        self.off_target_matched_edges = 0

    def add(
        self,
        payload: RaggedGeometryBatch,
        sidecar: OrderCoordinateSidecar,
    ) -> None:
        target = payload.exact_assignment_target_a
        if target is None:
            raise FrozenTransportReadoutError("synthetic sign check lacks exact targets")
        for direction in range(4):
            selected = (
                payload.labels
                & payload.direction_target_valid
                & (payload.direction_target == direction)
            )
            self.target_pair_count[direction] += int(selected.sum().item())
        values = exact_target_order_concordance(
            target,
            payload.token_mask_a,
            payload.token_mask_b,
            payload.correspondence_mask,
            sidecar,
        )
        if not values.candidate_index.numel():
            return
        candidate_direction = payload.direction_index.index_select(
            0, values.candidate_index
        )
        sample = payload.sample_index.index_select(0, values.candidate_index)
        target_direction = payload.direction_target.index_select(0, sample)
        target_valid = payload.direction_target_valid.index_select(0, sample)
        true_direction = target_valid & (candidate_direction == target_direction)
        self.off_target_groups += int((~true_direction).sum().item())
        self.off_target_matched_edges += int(
            values.matched_edge_count[~true_direction].sum().item()
        )
        for index in torch.nonzero(true_direction, as_tuple=False).flatten():
            direction = int(candidate_direction[index].item())
            scale = int(values.scale_index[index].item())
            row = self.total[direction].setdefault(scale, [0, 0, 0, 0])
            row[0] += int(values.matched_edge_count[index].item())
            row[1] += int(values.anti_pair_count[index].item())
            row[2] += int(values.monotone_pair_count[index].item())
            row[3] += int(values.tied_pair_count[index].item())

    def result(self) -> Mapping[str, Any]:
        rows: Dict[str, Any] = {}
        matched_edges = 0
        anti_sum = 0
        mono_sum = 0
        tied_sum = 0
        for direction, name in enumerate(_DIRECTION_LABELS):
            scale_rows = {}
            direction_edges = 0
            direction_anti = 0
            direction_mono = 0
            direction_tied = 0
            for scale, values in sorted(self.total[direction].items()):
                edge_count, anti, mono, tied = values
                comparable = anti + mono
                if comparable == 0 or anti <= mono:
                    raise FrozenTransportReadoutError(
                        "synthetic exact target direction/scale does not verify "
                        "pairwise anti-order: {}/{}".format(name, scale)
                    )
                scale_rows[str(scale)] = {
                    "matched_edges": edge_count,
                    "anti_pairs": anti,
                    "monotone_pairs": mono,
                    "tied_pairs": tied,
                    "anti_fraction_among_comparable": anti / comparable,
                }
                direction_edges += edge_count
                direction_anti += anti
                direction_mono += mono
                direction_tied += tied
            observed = self.target_pair_count[direction] > 0
            if observed and not scale_rows:
                raise FrozenTransportReadoutError(
                    "observed synthetic direction lacks comparable exact targets: "
                    + name
                )
            comparable = direction_anti + direction_mono
            rows[name] = {
                "observed_in_frozen_population": observed,
                "target_pairs": self.target_pair_count[direction],
                "matched_edges": direction_edges,
                "anti_pairs": direction_anti,
                "monotone_pairs": direction_mono,
                "tied_pairs": direction_tied,
                "anti_fraction_among_comparable": (
                    None if comparable == 0 else direction_anti / comparable
                ),
                "by_scale": scale_rows,
            }
            matched_edges += direction_edges
            anti_sum += direction_anti
            mono_sum += direction_mono
            tied_sum += direction_tied
        if matched_edges == 0 or anti_sum <= mono_sum:
            raise FrozenTransportReadoutError(
                "synthetic true-direction exact targets contradict pairwise anti-order"
            )
        return {
            "status": "passed_before_real_access",
            "expected_mapping": "pairwise_anti_concordant",
            "expected_mapping_origin": (
                "canonical image-clockwise traversal plus opposing facing sides"
            ),
            "target_filter": "candidate_direction_equals_synthetic_direction_target",
            "missing_direction_policy": "report_unobserved_without_imputed_evidence",
            "matched_edges": matched_edges,
            "anti_pairs": anti_sum,
            "monotone_pairs": mono_sum,
            "tied_pairs": tied_sum,
            "anti_fraction_among_comparable": anti_sum / (anti_sum + mono_sum),
            "off_target_exact_groups_excluded": self.off_target_groups,
            "off_target_matched_edges_excluded": self.off_target_matched_edges,
            "by_direction": rows,
        }


def _arc_batch(
    payload: RaggedGeometryBatch,
    sidecar: OrderCoordinateSidecar,
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> FrozenArcDataset:
    moved = payload.to(device)
    model_inputs = moved.model_inputs()
    with torch.inference_mode():
        output = model.forward_flat_candidates(
            **model_inputs, score_source=PairwiseScoreSource.LOCAL
        )
        candidate = output.arc_pairwise_output
        if candidate is None or candidate.local.transport is None:
            raise FrozenTransportReadoutError(
                "Exact winner did not expose dustbin transport"
            )
        local = candidate.local
        moved_sidecar = sidecar.to(device, dtype=local.assignment.dtype)
        quality = transport_pairwise_anti_order_mass(
            local.assignment,
            model_inputs["token_mask_a"],
            model_inputs["token_mask_b"],
            model_inputs["correspondence_mask"],
            moved_sidecar,
        )
        batch = FrozenArcDataset(
            arc_logit=output.arc_logits.detach(),
            mass_only=quality.detach(),
            entropy_aware=torch.zeros_like(quality),
            arc_valid=output.arc_valid.detach(),
            sample_index=output.sample_index.detach(),
            direction_index=output.direction_index.detach(),
            geometry_valid=moved.geometry_valid.detach(),
        )
        baseline = aggregate_frozen_readout(
            batch,
            readout_name=READOUT_BASELINE,
            arc_pooling=model.config.arc_pooling,
            direction_temperature=model.config.direction_aggregation_temperature,
        )
        expected = output.direction_output
        if not torch.equal(
            baseline.pair_valid, expected.pair_valid & moved.geometry_valid
        ) or not torch.allclose(
            baseline.pair_logit, expected.pair_logit, rtol=1e-5, atol=1e-6
        ):
            raise FrozenTransportReadoutError(
                "beta=0 order readout changed frozen Exact output"
            )
    return batch.to(torch.device("cpu"))


def _extract_population(
    records: Sequence[TrainingPairRecord],
    *,
    prepare: Callable[[Tuple[TrainingPairRecord, ...]], tuple[RaggedGeometryBatch, OrderCoordinateSidecar]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    labels: Optional[Tensor],
    target_accumulator: Optional[_TargetAccumulator] = None,
) -> FrozenArcDataset:
    batches = []
    for rows in _chunks(records, batch_size):
        payload, sidecar = prepare(rows)
        if target_accumulator is not None:
            target_accumulator.add(payload, sidecar)
        batches.append(_arc_batch(payload, sidecar, model, device=device))
    return _concatenate(batches, labels=labels)


def _internal_name(name: str) -> str:
    if name == READOUT_BASELINE:
        return READOUT_BASELINE
    if name == READOUT_ORDER_COHERENT:
        return READOUT_MASS_ONLY
    raise KeyError(name)


def _parameters_dict(
    external_name: str, parameters: FrozenReadoutParameters
) -> Mapping[str, Any]:
    value = dict(parameters.to_dict())
    value["name"] = external_name
    return value


def _prediction(
    dataset: FrozenArcDataset,
    *,
    name: str,
    beta: float,
    intercept: float,
    model: torch.nn.Module,
):
    return aggregate_frozen_readout(
        dataset,
        readout_name=_internal_name(name),
        beta=beta,
        intercept=intercept,
        arc_pooling=model.config.arc_pooling,
        direction_temperature=model.config.direction_aggregation_temperature,
    )


def _metrics(
    dataset: FrozenArcDataset,
    *,
    name: str,
    beta: float,
    intercept: float,
    model: torch.nn.Module,
    clusters: Sequence[str],
) -> Mapping[str, Any]:
    if dataset.label is None:
        raise ValueError("metrics require labels")
    output = _prediction(
        dataset,
        name=name,
        beta=beta,
        intercept=intercept,
        model=model,
    )
    return evaluate_pairwise(
        output.pair_probability.tolist(),
        dataset.label.tolist(),
        output.pair_valid.tolist(),
        clusters,
        threshold=0.5,
    )


def fit_synthetic(
    *,
    pilot_summary: Path,
    population_snapshot: Path,
    mask_root: Path,
    synthetic_cache_root: Path,
    output: Path,
    device: str,
    batch_size: int,
    fit_max_iterations: int,
    validation_fraction: float,
) -> Mapping[str, Any]:
    """Fit beta/intercept and freeze selection before any real access."""

    started = time.perf_counter()
    summary = _summary(pilot_summary)
    config_row = summary["config"]
    population_row = summary["population"]
    config = ExactSeamPilotConfig(
        mask_root=mask_root,
        output_root=Path(output).parent,
        cache_root=synthetic_cache_root,
        max_pairs=int(config_row["max_pairs"]),
        epochs=int(config_row["epochs"]),
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=int(config_row["seed"]),
        generator=str(config_row["generator"]),
        device=device,
        exact_loss_weight=float(config_row["exact_loss_weight"]),
        initialization_seed=int(
            config_row.get("initialization_seed", config_row["seed"])
        ),
    )
    population_started = time.perf_counter()
    population = load_exact_seam_population_snapshot(population_snapshot, config)
    population_seconds = time.perf_counter() - population_started
    if (
        len(population.training_records) != int(config_row["train_pairs"])
        or len(population.validation_records) != int(config_row["validation_pairs"])
        or population.consumed_group_count
        != int(population_row["consumed_group_count"])
    ):
        raise FrozenTransportReadoutError("population snapshot differs from Exact pilot")

    models = load_exact_seam_pilot_pair(pilot_summary, device=device)
    model = _exact_model(models)
    geometry_config = GeometryBatchConfig()
    keypoint_config = _winner_keypoint_config(models)
    cache = GeometryArtifactCache(synthetic_cache_root)
    loader = _DirectoryMaskLoader(mask_root)

    def prepare(rows: Tuple[TrainingPairRecord, ...]):
        return _build_with_sidecar(
            rows,
            loader=loader,
            cache=cache,
            geometry_config=geometry_config,
            keypoint_config=keypoint_config,
            exact_seam_supervision=True,
        )

    feature_started = time.perf_counter()
    target_accumulator = _TargetAccumulator()
    train = _extract_population(
        population.training_records,
        prepare=prepare,
        model=model,
        device=torch.device(device),
        batch_size=batch_size,
        labels=torch.tensor(
            [record.label for record in population.training_records], dtype=torch.bool
        ),
        target_accumulator=target_accumulator,
    )
    validation = _extract_population(
        population.validation_records,
        prepare=prepare,
        model=model,
        device=torch.device(device),
        batch_size=batch_size,
        labels=torch.tensor(
            [record.label for record in population.validation_records],
            dtype=torch.bool,
        ),
        target_accumulator=target_accumulator,
    )
    verification = target_accumulator.result()
    feature_seconds = time.perf_counter() - feature_started

    fit_started = time.perf_counter()
    fitted = {
        READOUT_BASELINE: fit_frozen_readout(
            train,
            readout_name=READOUT_BASELINE,
            arc_pooling=model.config.arc_pooling,
            direction_temperature=model.config.direction_aggregation_temperature,
            max_iterations=fit_max_iterations,
        ),
        READOUT_ORDER_COHERENT: fit_frozen_readout(
            train,
            readout_name=READOUT_MASS_ONLY,
            arc_pooling=model.config.arc_pooling,
            direction_temperature=model.config.direction_aggregation_temperature,
            max_iterations=fit_max_iterations,
        ),
    }
    train_clusters = [record.canonical_group_id for record in population.training_records]
    validation_clusters = [
        record.canonical_group_id for record in population.validation_records
    ]
    rows = {}
    for name in READOUTS:
        parameter = fitted[name]
        rows[name] = {
            "parameters": _parameters_dict(name, parameter),
            "train_metrics": _metrics(
                train,
                name=name,
                beta=parameter.beta,
                intercept=parameter.intercept,
                model=model,
                clusters=train_clusters,
            ),
            "validation_metrics": _metrics(
                validation,
                name=name,
                beta=parameter.beta,
                intercept=parameter.intercept,
                model=model,
                clusters=validation_clusters,
            ),
        }
    winner = max(
        READOUTS,
        key=lambda name: (
            float(rows[name]["validation_metrics"]["row"]["auroc"]),
            float(rows[name]["validation_metrics"]["row"]["auprc"]),
            int(name == READOUT_BASELINE),
        ),
    )
    fit_seconds = time.perf_counter() - fit_started
    result: Dict[str, Any] = {
        "schema_version": ORDER_READOUT_VERSION,
        "status": FIT_STATUS,
        "scope": {
            "mask_only": True,
            "known_upright_orientation": True,
            "rotation_search": False,
            "rgb_or_text_used": False,
            "backbone_and_existing_local_head_frozen": True,
            "fitted_parameters": ["beta", "intercept"],
            "real_data_opened": False,
            "real_labels_used_for_fit_or_selection": False,
        },
        "pilot": _selected_exact_identity(summary),
        "population": {
            "train_pairs": train.pair_count,
            "validation_pairs": validation.pair_count,
            "consumed_group_count": population.consumed_group_count,
            "source": "frozen_ordered_pair_id_snapshot",
            "same_population_as_exact_pilot": True,
        },
        "pre_registration": {
            "mapping": "pairwise_anti_concordant",
            "coordinate": (
                "real source contour arc fraction; deterministic largest-circular-"
                "gap unwrap and min-max per side and scale"
            ),
            "comparison": (
                "two transported edges are coherent iff their strict A/B "
                "arclength orders have opposite signs"
            ),
            "q": (
                "anti-concordant pairwise real-assignment mass divided by "
                "minimum within-scale unordered token-pair capacity"
            ),
            "subsequence_and_translation_invariant": True,
            "arc_formula": "frozen_arc_logit + beta * q + intercept",
            "aggregation": "unchanged_log_mean_exp_arc_then_direction",
            "feature_grid_searched": False,
        },
        "synthetic_exact_target_sign_verification": verification,
        "readouts": rows,
        "selection": {
            "source": "synthetic_validation_only",
            "policy": "row_auroc_then_row_auprc_then_frozen_baseline",
            "winner": winner,
            "real_evaluation_accessed": False,
        },
        "runtime_seconds": {
            "population_snapshot_load": population_seconds,
            "shared_frozen_feature_extraction_and_sign_check": feature_seconds,
            "scalar_fit_and_metrics": fit_seconds,
            "total": time.perf_counter() - started,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _load_fit(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != ORDER_READOUT_VERSION
        or value.get("status") != FIT_STATUS
    ):
        raise FrozenTransportReadoutError("order readout fit artifact is unsupported")
    scope = value.get("scope")
    selection = value.get("selection")
    verification = value.get("synthetic_exact_target_sign_verification")
    if (
        not isinstance(scope, Mapping)
        or scope.get("real_data_opened") is not False
        or scope.get("real_labels_used_for_fit_or_selection") is not False
        or not isinstance(selection, Mapping)
        or selection.get("source") != "synthetic_validation_only"
        or selection.get("real_evaluation_accessed") is not False
        or not isinstance(verification, Mapping)
        or verification.get("status") != "passed_before_real_access"
    ):
        raise FrozenTransportReadoutError("fit artifact is not synthetic-only frozen")
    return value


def _fit_parameters(fit: Mapping[str, Any]) -> Mapping[str, Tuple[float, float]]:
    rows = fit.get("readouts")
    if not isinstance(rows, Mapping) or set(rows) != set(READOUTS):
        raise FrozenTransportReadoutError("order readout parameters are incomplete")
    output = {}
    for name in READOUTS:
        values = rows[name].get("parameters")
        if not isinstance(values, Mapping) or values.get("name") != name:
            raise FrozenTransportReadoutError("order readout identity changed")
        beta = float(values["beta"])
        intercept = float(values["intercept"])
        if not math.isfinite(beta) or not math.isfinite(intercept):
            raise FrozenTransportReadoutError("order readout parameter is non-finite")
        output[name] = (beta, intercept)
    return output


def _weighted_ranking(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> Tuple[float, float]:
    keep = weights > 0.0
    labels, scores, weights = labels[keep], scores[keep], weights[keep]
    positive_total = float(weights[labels].sum())
    negative_total = float(weights[~labels].sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise ValueError("weighted ranking requires both classes")
    order = np.argsort(-scores, kind="mergesort")
    labels, scores, weights = labels[order], scores[order], weights[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    positive = np.add.reduceat(weights * labels, starts)
    negative = np.add.reduceat(weights * ~labels, starts)
    true_positive = np.cumsum(positive)
    false_positive = np.cumsum(negative)
    tpr, fpr = true_positive / positive_total, false_positive / negative_total
    previous_tpr, previous_fpr = np.r_[0.0, tpr[:-1]], np.r_[0.0, fpr[:-1]]
    auroc = np.sum((fpr - previous_fpr) * (tpr + previous_tpr) * 0.5)
    precision = true_positive / (true_positive + false_positive)
    auprc = np.sum((tpr - previous_tpr) * precision)
    return float(auroc), float(auprc)


def _paired_bootstrap(
    *,
    labels: np.ndarray,
    clusters: np.ndarray,
    baseline: np.ndarray,
    treatment: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> Mapping[str, Any]:
    unique, inverse = np.unique(clusters, return_inverse=True)
    draws = {"auroc": [], "auprc": []}
    skipped = 0
    for _ in range(replicates):
        selected = rng.integers(0, len(unique), size=len(unique))
        multiplicity = np.bincount(selected, minlength=len(unique))
        weight = multiplicity[inverse].astype(np.float64)
        if not np.any(weight[labels] > 0.0) or not np.any(weight[~labels] > 0.0):
            skipped += 1
            continue
        base_metric = _weighted_ranking(labels, baseline, weight)
        treatment_metric = _weighted_ranking(labels, treatment, weight)
        draws["auroc"].append(treatment_metric[0] - base_metric[0])
        draws["auprc"].append(treatment_metric[1] - base_metric[1])
    base_point = _weighted_ranking(labels, baseline, np.ones(len(labels)))
    treatment_point = _weighted_ranking(labels, treatment, np.ones(len(labels)))
    intervals = {}
    for index, name in enumerate(("auroc", "auprc")):
        values = np.asarray(draws[name], dtype=np.float64)
        lower, upper = np.quantile(values, [0.025, 0.975])
        point = treatment_point[index] - base_point[index]
        intervals[name] = {
            "point_estimate": float(point),
            "percentile_95_ci": [float(lower), float(upper)],
            "bootstrap_standard_error": float(values.std(ddof=1)),
            "probability_delta_gt_zero": float(np.mean(values > 0.0)),
            "valid_replicates": int(len(values)),
        }
    return {
        "sampling_cluster_count": int(len(unique)),
        "replicates_requested": replicates,
        "skipped_single_class_replicates": skipped,
        "comparison": "order_coherent_minus_frozen_exact_baseline",
        "metrics": intervals,
    }


def evaluate_real(
    *,
    pilot_summary: Path,
    readout_fit: Path,
    manifest: Path,
    local_path_receipt: Path,
    main_root: Path,
    supp_root: Path,
    real_cache_root: Path,
    output: Path,
    device: str,
    batch_size: int,
    target_long_side: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> Mapping[str, Any]:
    """Evaluate both locked scores once; real labels cannot alter scores."""

    started = time.perf_counter()
    fit = _load_fit(readout_fit)
    if fit["selection"].get("winner") != READOUT_ORDER_COHERENT:
        raise FrozenTransportReadoutError(
            "synthetic validation did not select the order readout; real access refused"
        )
    parameters = _fit_parameters(fit)
    summary = _summary(pilot_summary)
    if dict(fit["pilot"]) != dict(_selected_exact_identity(summary)):
        raise FrozenTransportReadoutError("fit/evaluation Exact winners differ")
    models = load_exact_seam_pilot_pair(pilot_summary, device=device)
    model = _exact_model(models)
    keypoint_config = _winner_keypoint_config(models)
    strict = load_strict_real_pair_dataset(
        manifest,
        local_path_receipt,
        main_root=main_root,
        supp_root=supp_root,
        target_long_side=target_long_side,
    )
    try:
        build = build_balanced_real_pair_dataset(strict)
        dataset = build.dataset
        records = tuple(dataset.records)
        blinded = tuple(_blinded_record(record) for record in records)
        if any(record.label for record in blinded) or any(
            record.direction_b_wrt_a is not None for record in blinded
        ):
            raise FrozenTransportReadoutError("real supervision was not blinded")
        cache = GeometryArtifactCache(real_cache_root)
        geometry_config = GeometryBatchConfig()

        def prepare(rows: Tuple[TrainingPairRecord, ...]):
            return _build_with_sidecar(
                rows,
                loader=dataset.mask_loader,
                cache=cache,
                geometry_config=geometry_config,
                keypoint_config=keypoint_config,
                exact_seam_supervision=False,
            )

        features = _extract_population(
            blinded,
            prepare=prepare,
            model=model,
            device=torch.device(device),
            batch_size=batch_size,
            labels=None,
        )
        labelled = replace(
            features,
            label=torch.tensor([record.label for record in records], dtype=torch.bool),
        )
        scores = {}
        for name in READOUTS:
            beta, intercept = parameters[name]
            prediction = _prediction(
                labelled,
                name=name,
                beta=beta,
                intercept=intercept,
                model=model,
            )
            scores[name] = {
                "probability": prediction.pair_probability.numpy(),
                "valid": prediction.pair_valid.numpy(),
                "direction": prediction.best_direction_index.numpy(),
            }
        if not np.array_equal(
            scores[READOUT_BASELINE]["valid"],
            scores[READOUT_ORDER_COHERENT]["valid"],
        ):
            raise FrozenTransportReadoutError("order readout changed pair validity")

        populations: Dict[str, Any] = {}
        strict_count = len(strict.records)
        slices = {"strict547": (0, strict_count), "balanced1016": (0, len(records))}
        seed_sequence = np.random.SeedSequence(bootstrap_seed)
        rngs = [np.random.default_rng(value) for value in seed_sequence.spawn(2)]
        for population_index, (population_name, (start, stop)) in enumerate(slices.items()):
            selected_records = records[start:stop]
            valid = scores[READOUT_BASELINE]["valid"][start:stop]
            labels = np.asarray(
                [record.label for record in selected_records], dtype=np.bool_
            )[valid]
            clusters = np.asarray(
                [record.component_id for record in selected_records], dtype=object
            )[valid]
            baseline_probability = scores[READOUT_BASELINE]["probability"][start:stop][valid]
            treatment_probability = scores[READOUT_ORDER_COHERENT]["probability"][start:stop][valid]
            methods = {}
            for name in READOUTS:
                methods[name] = _real_method_metrics(
                    probability=scores[name]["probability"][start:stop].tolist(),
                    valid=scores[name]["valid"][start:stop].tolist(),
                    best_direction=scores[name]["direction"][start:stop].tolist(),
                    records=selected_records,
                )
            populations[population_name] = {
                "pair_count": len(selected_records),
                "common_valid_count": int(valid.sum()),
                "positive_count": sum(record.label for record in selected_records),
                "negative_count": sum(not record.label for record in selected_records),
                "methods": methods,
                "paired_cluster_bootstrap": _paired_bootstrap(
                    labels=labels,
                    clusters=clusters,
                    baseline=baseline_probability,
                    treatment=treatment_probability,
                    replicates=bootstrap_replicates,
                    rng=rngs[population_index],
                ),
            }
        pair_rows = []
        for index, record in enumerate(records):
            pair_rows.append(
                {
                    "pair_id": record.pair_id,
                    "cluster_id": record.component_id,
                    "label": record.label,
                    "true_direction": record.direction_b_wrt_a,
                    "readouts": {
                        name: {
                            "probability": (
                                float(scores[name]["probability"][index])
                                if scores[name]["valid"][index]
                                else None
                            ),
                            "valid": bool(scores[name]["valid"][index]),
                            "predicted_direction": (
                                _DIRECTION_LABELS[int(scores[name]["direction"][index])]
                                if scores[name]["valid"][index]
                                else None
                            ),
                        }
                        for name in READOUTS
                    },
                }
            )
        result: Dict[str, Any] = {
            "schema_version": ORDER_READOUT_VERSION,
            "status": REAL_STATUS,
            "scope": {
                "mask_only": True,
                "known_upright_orientation": True,
                "rotation_search": False,
                "backbone_and_existing_local_head_frozen": True,
                "parameters_loaded_before_real_dataset": True,
                "real_labels_used_for_fit_or_selection": False,
                "real_supervision_blinded_before_geometry": True,
                "post_real_feature_or_parameter_tuning": False,
            },
            "pilot": _selected_exact_identity(summary),
            "synthetic_selection": dict(fit["selection"]),
            "readout_parameters": {
                name: {"beta": parameters[name][0], "intercept": parameters[name][1]}
                for name in READOUTS
            },
            "bootstrap": {
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "sampling": "paired component-cluster resampling with replacement",
                "confidence_interval": "two-sided_95_percentile",
            },
            "populations": populations,
            "balanced_negative_semantics": build.receipt["negative_semantics"],
            "pairs": pair_rows,
            "runtime_seconds": {"total": time.perf_counter() - started},
        }
    finally:
        strict.mask_loader.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit-synthetic", allow_abbrev=False)
    fit.add_argument("--pilot-summary", type=Path, required=True)
    fit.add_argument("--population-snapshot", type=Path, required=True)
    fit.add_argument("--mask-root", type=Path, required=True)
    fit.add_argument("--synthetic-cache-root", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--batch-size", type=_positive_int, default=8)
    fit.add_argument("--fit-max-iterations", type=_positive_int, default=64)
    fit.add_argument("--validation-fraction", type=float, default=0.2)
    real = commands.add_parser("evaluate-real", allow_abbrev=False)
    real.add_argument("--pilot-summary", type=Path, required=True)
    real.add_argument("--readout-fit", type=Path, required=True)
    real.add_argument("--manifest", type=Path, required=True)
    real.add_argument("--local-path-receipt", type=Path, required=True)
    real.add_argument("--main-root", type=Path, required=True)
    real.add_argument("--supp-root", type=Path, required=True)
    real.add_argument("--real-cache-root", type=Path, required=True)
    real.add_argument("--output", type=Path, required=True)
    real.add_argument("--device", default="cuda:0")
    real.add_argument("--batch-size", type=_positive_int, default=8)
    real.add_argument("--target-long-side", type=_positive_int, default=DEFAULT_TARGET_LONG_SIDE)
    real.add_argument("--bootstrap-replicates", type=_positive_int, default=BOOTSTRAP_REPLICATES)
    real.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "fit-synthetic":
        result = fit_synthetic(
            pilot_summary=arguments.pilot_summary,
            population_snapshot=arguments.population_snapshot,
            mask_root=arguments.mask_root,
            synthetic_cache_root=arguments.synthetic_cache_root,
            output=arguments.output,
            device=arguments.device,
            batch_size=arguments.batch_size,
            fit_max_iterations=arguments.fit_max_iterations,
            validation_fraction=arguments.validation_fraction,
        )
        compact = {
            "status": result["status"],
            "output": str(arguments.output),
            "sign_verification": result["synthetic_exact_target_sign_verification"],
            "selection": result["selection"],
            "validation": {
                name: {
                    "auroc": row["validation_metrics"]["row"]["auroc"],
                    "auprc": row["validation_metrics"]["row"]["auprc"],
                }
                for name, row in result["readouts"].items()
            },
            "runtime_seconds": result["runtime_seconds"],
        }
    else:
        result = evaluate_real(
            pilot_summary=arguments.pilot_summary,
            readout_fit=arguments.readout_fit,
            manifest=arguments.manifest,
            local_path_receipt=arguments.local_path_receipt,
            main_root=arguments.main_root,
            supp_root=arguments.supp_root,
            real_cache_root=arguments.real_cache_root,
            output=arguments.output,
            device=arguments.device,
            batch_size=arguments.batch_size,
            target_long_side=arguments.target_long_side,
            bootstrap_replicates=arguments.bootstrap_replicates,
            bootstrap_seed=arguments.bootstrap_seed,
        )
        compact = {
            "status": result["status"],
            "output": str(arguments.output),
            "synthetic_winner": result["synthetic_selection"]["winner"],
            "populations": result["populations"],
            "runtime_seconds": result["runtime_seconds"],
        }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

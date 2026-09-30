"""Frozen Exact-Sinkhorn transport-readout ablation.

``fit-synthetic`` reconstructs the same balanced/group-disjoint Exact pilot
population, performs one inference-only pass through the selected Exact winner,
and fits only ``beta`` and ``intercept`` for mass-only and entropy-aware scalar
arc residuals.  Selection uses synthetic validation AUROC/AUPRC only.

``evaluate-real`` loads the already-frozen scalar parameters before opening any
real dataset, blinds every real record before geometry construction, performs
one shared Exact-winner forward, and evaluates all readouts on strict547 and
balanced1016.  Real labels are never used for fitting or selection.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from staging.pairwise_v0_2.baselines.exact_seam_post_pilot_real_evaluation import (
    EXACT_METHOD,
    _blinded_record,
    load_exact_seam_pilot_pair,
)
from staging.pairwise_v0_2.baselines.real_dunhuang_balanced_distractors import (
    build_balanced_real_pair_dataset,
)
from staging.pairwise_v0_2.baselines.real_dunhuang_evaluation import (
    DEFAULT_TARGET_LONG_SIDE,
    _DIRECTION_LABELS,
    _direction_metrics,
    load_strict_real_pair_dataset,
    prepare_real_geometry_batch,
)
from staging.pairwise_v0_2.geometry import ContourKeypointConfig
from staging.pairwise_v0_2.models.frozen_transport_readout import (
    FrozenArcDataset,
    FrozenReadoutParameters,
    READOUT_BASELINE,
    READOUT_ENTROPY_AWARE,
    READOUT_MASS_ONLY,
    READOUT_NAMES,
    aggregate_frozen_readout,
    fit_frozen_readout,
    transport_quality_features,
)
from staging.pairwise_v0_2.models.pairwise import PairwiseScoreSource
from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise
from staging.pairwise_v0_2.training.exact_seam_pilot import (
    ExactSeamPilotConfig,
    _DirectoryMaskLoader,
    build_exact_seam_pilot_population,
    exact_seam_group_eligibility,
)
from staging.pairwise_v0_2.training.exact_seam_population_snapshot import (
    load_exact_seam_population_snapshot,
)
from staging.pairwise_v0_2.training.geometry_batch import (
    DATA_DIRECTION_TO_INDEX,
    KEYPOINT_REPRESENTATION,
    GeometryBatchConfig,
    RaggedGeometryBatch,
    build_geometry_batch,
)
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache


FROZEN_TRANSPORT_READOUT_VERSION = "dunhuang-frozen-transport-readout/0.1"
FIT_STATUS = "complete_synthetic_only_frozen_exact_readout_fit"
REAL_STATUS = "complete_frozen_readouts_real_evaluation"


class FrozenTransportReadoutError(RuntimeError):
    """The standalone frozen-readout contract could not be met."""


def _chunks(values: Sequence[Any], size: int) -> Iterable[Tuple[Any, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def _summary(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("status") != "complete":
        raise FrozenTransportReadoutError("pilot summary is not complete JSON")
    return value


def _selected_exact_identity(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = summary.get("winner_selection")
    if not isinstance(selection, Mapping):
        raise FrozenTransportReadoutError("pilot summary lacks winner selection")
    winners = selection.get("winners")
    if not isinstance(winners, Mapping) or not isinstance(
        winners.get(EXACT_METHOD), Mapping
    ):
        raise FrozenTransportReadoutError("pilot summary lacks Exact winner")
    winner = winners[EXACT_METHOD]
    checkpoint = winner.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise FrozenTransportReadoutError("Exact winner checkpoint is missing")
    return {
        "epoch": int(winner["epoch"]),
        "file_sha256": str(checkpoint["file_sha256"]),
        "model_state_sha256": str(checkpoint["model_state_sha256"]),
        "checkpoint_filename": Path(str(checkpoint["path"])).name,
    }


def _exact_model(models: Any) -> torch.nn.Module:
    session = models.exact.session
    model = getattr(session, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise FrozenTransportReadoutError("loaded Exact winner lacks a model")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise FrozenTransportReadoutError("Exact backbone did not freeze")
    return model


def _arc_batch(
    payload: RaggedGeometryBatch,
    model: torch.nn.Module,
    *,
    device: torch.device,
    top_k: int,
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
        quality = transport_quality_features(
            local.assignment,
            local.unmatched_a,
            local.unmatched_b,
            model_inputs["token_mask_a"],
            model_inputs["token_mask_b"],
            model_inputs.get("correspondence_mask"),
            top_k=top_k,
        )
        batch = FrozenArcDataset(
            arc_logit=output.arc_logits.detach(),
            mass_only=quality.mass_only.detach(),
            entropy_aware=quality.entropy_aware.detach(),
            arc_valid=output.arc_valid.detach(),
            sample_index=output.sample_index.detach(),
            direction_index=output.direction_index.detach(),
            geometry_valid=moved.geometry_valid.detach(),
            label=None,
        )
        baseline = aggregate_frozen_readout(
            batch,
            readout_name=READOUT_BASELINE,
            arc_pooling=model.config.arc_pooling,
            direction_temperature=model.config.direction_aggregation_temperature,
        )
        expected = output.direction_output
        if not torch.equal(baseline.pair_valid, expected.pair_valid & moved.geometry_valid):
            raise FrozenTransportReadoutError(
                "beta=0 readout changed frozen pair validity"
            )
        if not torch.allclose(
            baseline.pair_logit,
            expected.pair_logit,
            rtol=1e-5,
            atol=1e-6,
        ):
            difference = float(
                (baseline.pair_logit - expected.pair_logit).abs().max().cpu()
            )
            raise FrozenTransportReadoutError(
                "beta=0 readout changed frozen pair logits: {}".format(difference)
            )
    return batch.to(torch.device("cpu"))


def _concatenate(
    batches: Sequence[FrozenArcDataset],
    *,
    labels: Optional[Tensor],
) -> FrozenArcDataset:
    if not batches:
        raise ValueError("at least one feature batch is required")
    arc_logits: List[Tensor] = []
    mass: List[Tensor] = []
    entropy: List[Tensor] = []
    valid: List[Tensor] = []
    sample_indices: List[Tensor] = []
    directions: List[Tensor] = []
    geometry: List[Tensor] = []
    offset = 0
    for batch in batches:
        arc_logits.append(batch.arc_logit)
        mass.append(batch.mass_only)
        entropy.append(batch.entropy_aware)
        valid.append(batch.arc_valid)
        sample_indices.append(batch.sample_index + offset)
        directions.append(batch.direction_index)
        geometry.append(batch.geometry_valid)
        offset += batch.pair_count
    if labels is not None and tuple(labels.shape) != (offset,):
        raise ValueError("labels do not match concatenated pair count")
    return FrozenArcDataset(
        arc_logit=torch.cat(arc_logits),
        mass_only=torch.cat(mass),
        entropy_aware=torch.cat(entropy),
        arc_valid=torch.cat(valid),
        sample_index=torch.cat(sample_indices),
        direction_index=torch.cat(directions),
        geometry_valid=torch.cat(geometry),
        label=labels,
    )


def _extract_population(
    records: Sequence[TrainingPairRecord],
    *,
    prepare: Callable[[Tuple[TrainingPairRecord, ...]], RaggedGeometryBatch],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    top_k: int,
    labels: Optional[Tensor],
) -> FrozenArcDataset:
    batches = []
    for rows in _chunks(records, batch_size):
        batches.append(
            _arc_batch(prepare(rows), model, device=device, top_k=top_k)
        )
    return _concatenate(batches, labels=labels)


def _metrics(
    dataset: FrozenArcDataset,
    parameters: FrozenReadoutParameters,
    *,
    model: torch.nn.Module,
    clusters: Sequence[str],
) -> Mapping[str, object]:
    if dataset.label is None:
        raise ValueError("metrics require labels")
    output = aggregate_frozen_readout(
        dataset,
        readout_name=parameters.name,
        beta=parameters.beta,
        intercept=parameters.intercept,
        arc_pooling=model.config.arc_pooling,
        direction_temperature=model.config.direction_aggregation_temperature,
    )
    return evaluate_pairwise(
        output.pair_probability.tolist(),
        dataset.label.tolist(),
        output.pair_valid.tolist(),
        clusters,
        threshold=0.5,
    )


def _fit_rank(row: Mapping[str, Any]) -> Tuple[float, float, int]:
    validation = row["validation_metrics"]
    row_metrics = validation["row"]
    name = str(row["parameters"]["name"])
    # Exact ties prefer the simpler frozen baseline, then mass-only.
    simplicity = {
        READOUT_BASELINE: 2,
        READOUT_MASS_ONLY: 1,
        READOUT_ENTROPY_AWARE: 0,
    }[name]
    return float(row_metrics["auroc"]), float(row_metrics["auprc"]), simplicity


def fit_synthetic(
    *,
    pilot_summary: Path,
    mask_root: Path,
    synthetic_cache_root: Path,
    output: Path,
    device: str,
    batch_size: int,
    top_k: int,
    fit_max_iterations: int,
    validation_fraction: float,
    population_snapshot: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Fit the two scalar treatments on synthetic train only."""

    started = time.perf_counter()
    summary = _summary(pilot_summary)
    config_row = summary.get("config")
    population_row = summary.get("population")
    if not isinstance(config_row, Mapping) or not isinstance(population_row, Mapping):
        raise FrozenTransportReadoutError("pilot summary lacks population/config")
    max_pairs = int(config_row["max_pairs"])
    seed = int(config_row["seed"])
    generator = str(config_row["generator"])
    exact_weight = float(config_row["exact_loss_weight"])
    config = ExactSeamPilotConfig(
        mask_root=mask_root,
        output_root=Path(output).parent,
        cache_root=synthetic_cache_root,
        max_pairs=max_pairs,
        epochs=int(config_row["epochs"]),
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=seed,
        generator=generator,
        device=device,
        exact_loss_weight=exact_weight,
        initialization_seed=int(config_row.get("initialization_seed", seed)),
    )
    geometry_config = GeometryBatchConfig()
    cache = GeometryArtifactCache(synthetic_cache_root)
    loader = _DirectoryMaskLoader(mask_root)

    def eligible(rows: Sequence[TrainingPairRecord]) -> Sequence[bool]:
        return exact_seam_group_eligibility(
            rows,
            loader=loader,
            cache=cache,
            geometry_config=geometry_config,
        )

    population_started = time.perf_counter()
    if population_snapshot is None:
        population = build_exact_seam_pilot_population(
            config, group_record_filter=eligible
        )
        population_source = "deterministic_geometry_eligibility_replay"
    else:
        population = load_exact_seam_population_snapshot(
            population_snapshot, config
        )
        population_source = "frozen_ordered_pair_id_snapshot"
    population_seconds = time.perf_counter() - population_started
    if (
        len(population.training_records) != int(config_row["train_pairs"])
        or len(population.validation_records)
        != int(config_row["validation_pairs"])
        or population.consumed_group_count
        != int(population_row["consumed_group_count"])
    ):
        raise FrozenTransportReadoutError(
            "reconstructed population differs from the trained pilot"
        )
    models = load_exact_seam_pilot_pair(pilot_summary, device=device)
    model = _exact_model(models)
    torch_device = torch.device(device)

    def prepare(rows: Tuple[TrainingPairRecord, ...]) -> RaggedGeometryBatch:
        return build_geometry_batch(
            rows,
            loader,
            geometry_config,
            geometry_artifact_cache=cache,
            candidate_representation=KEYPOINT_REPRESENTATION,
            keypoint_config=ContourKeypointConfig(),
            exact_seam_supervision=False,
        )

    feature_started = time.perf_counter()
    train = _extract_population(
        population.training_records,
        prepare=prepare,
        model=model,
        device=torch_device,
        batch_size=batch_size,
        top_k=top_k,
        labels=torch.tensor(
            [record.label for record in population.training_records], dtype=torch.bool
        ),
    )
    validation = _extract_population(
        population.validation_records,
        prepare=prepare,
        model=model,
        device=torch_device,
        batch_size=batch_size,
        top_k=top_k,
        labels=torch.tensor(
            [record.label for record in population.validation_records], dtype=torch.bool
        ),
    )
    feature_seconds = time.perf_counter() - feature_started
    fit_started = time.perf_counter()
    parameters = {
        name: fit_frozen_readout(
            train,
            readout_name=name,
            arc_pooling=model.config.arc_pooling,
            direction_temperature=model.config.direction_aggregation_temperature,
            max_iterations=fit_max_iterations,
        )
        for name in READOUT_NAMES
    }
    rows = {}
    train_clusters = [
        record.canonical_group_id for record in population.training_records
    ]
    validation_clusters = [
        record.canonical_group_id for record in population.validation_records
    ]
    for name in READOUT_NAMES:
        rows[name] = {
            "parameters": parameters[name].to_dict(),
            "train_metrics": _metrics(
                train, parameters[name], model=model, clusters=train_clusters
            ),
            "validation_metrics": _metrics(
                validation,
                parameters[name],
                model=model,
                clusters=validation_clusters,
            ),
        }
    fit_seconds = time.perf_counter() - fit_started
    winner = max(rows.values(), key=_fit_rank)
    result: Dict[str, Any] = {
        "schema_version": FROZEN_TRANSPORT_READOUT_VERSION,
        "status": FIT_STATUS,
        "scope": {
            "mask_only": True,
            "known_upright_orientation": True,
            "rotation_search": False,
            "rgb_or_text_used": False,
            "backbone_frozen": True,
            "existing_local_head_frozen": True,
            "fitted_parameters_per_treatment": ["beta", "intercept"],
            "real_data_opened": False,
            "real_labels_used_for_fit_or_selection": False,
        },
        "pilot": _selected_exact_identity(summary),
        "population": {
            "train_pairs": train.pair_count,
            "validation_pairs": validation.pair_count,
            "consumed_group_count": population.consumed_group_count,
            "group_disjoint": True,
            "same_population_as_exact_pilot": True,
            "source": population_source,
        },
        "transport_quality": {
            "top_k": top_k,
            "mass_only": (
                "symmetric top-k mean real/non-dustbin token mass"
            ),
            "entropy_aware": (
                "symmetric top-k mean real mass times one-minus normalized "
                "row/column entropy including dustbin"
            ),
            "arc_formula": "frozen_arc_logit + beta * q + intercept",
            "aggregation": "unchanged_log_mean_exp_arc_then_direction",
        },
        "readouts": rows,
        "selection": {
            "source": "synthetic_validation_only",
            "policy": "row_auroc_then_row_auprc_then_simpler_readout",
            "winner": winner["parameters"]["name"],
            "real_evaluation_accessed": False,
        },
        "runtime_seconds": {
            "population_reconstruction": population_seconds,
            "frozen_feature_extraction": feature_seconds,
            "scalar_fit_and_metrics": fit_seconds,
            "total": time.perf_counter() - started,
        },
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _load_fit(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != FROZEN_TRANSPORT_READOUT_VERSION
        or value.get("status") != FIT_STATUS
    ):
        raise FrozenTransportReadoutError("readout fit artifact is unsupported")
    scope = value.get("scope")
    selection = value.get("selection")
    if (
        not isinstance(scope, Mapping)
        or scope.get("real_data_opened") is not False
        or scope.get("real_labels_used_for_fit_or_selection") is not False
        or not isinstance(selection, Mapping)
        or selection.get("source") != "synthetic_validation_only"
        or selection.get("real_evaluation_accessed") is not False
    ):
        raise FrozenTransportReadoutError("fit artifact is not synthetic-only")
    return value


def _fit_parameters(fit: Mapping[str, Any]) -> Mapping[str, FrozenReadoutParameters]:
    rows = fit.get("readouts")
    if not isinstance(rows, Mapping) or set(rows) != set(READOUT_NAMES):
        raise FrozenTransportReadoutError("fit artifact readouts are incomplete")
    output = {}
    for name in READOUT_NAMES:
        row = rows[name]
        if not isinstance(row, Mapping) or not isinstance(row.get("parameters"), Mapping):
            raise FrozenTransportReadoutError("fit parameters are missing")
        values = row["parameters"]
        output[name] = FrozenReadoutParameters(
            name=str(values["name"]),
            beta=float(values["beta"]),
            intercept=float(values["intercept"]),
            train_loss=float(values["train_loss"]),
            optimizer_steps=int(values["optimizer_steps"]),
        )
        if output[name].name != name:
            raise FrozenTransportReadoutError("fit readout name changed")
    return output


def _real_method_metrics(
    *,
    probability: Sequence[float],
    valid: Sequence[bool],
    best_direction: Sequence[int],
    records: Sequence[TrainingPairRecord],
) -> Mapping[str, Any]:
    labels = [record.label for record in records]
    clusters = [record.component_id for record in records]
    targets = [
        -1
        if record.direction_b_wrt_a is None
        else int(DATA_DIRECTION_TO_INDEX[record.direction_b_wrt_a])
        for record in records
    ]
    return {
        "valid_count": sum(valid),
        "coverage": sum(valid) / len(valid),
        "ranking": evaluate_pairwise(
            probability, labels, valid, clusters, threshold=0.5
        ),
        "direction": _direction_metrics(
            labels=labels,
            target=targets,
            valid=valid,
            predicted=best_direction,
        ),
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
    include_balanced_1016: bool,
) -> Mapping[str, Any]:
    """Evaluate pre-frozen scalar readouts; no real-label fitting is possible."""

    started = time.perf_counter()
    # Parameters and synthetic winner are fixed before the real loader is opened.
    fit = _load_fit(readout_fit)
    parameters = _fit_parameters(fit)
    top_k = int(fit["transport_quality"]["top_k"])
    summary = _summary(pilot_summary)
    if dict(fit["pilot"]) != dict(_selected_exact_identity(summary)):
        raise FrozenTransportReadoutError(
            "readout fit and evaluation use different Exact winners"
        )
    models = load_exact_seam_pilot_pair(pilot_summary, device=device)
    model = _exact_model(models)
    strict = load_strict_real_pair_dataset(
        manifest,
        local_path_receipt,
        main_root=main_root,
        supp_root=supp_root,
        target_long_side=target_long_side,
    )
    try:
        build = build_balanced_real_pair_dataset(strict) if include_balanced_1016 else None
        dataset = strict if build is None else build.dataset
        records = tuple(dataset.records)
        blinded = tuple(_blinded_record(record) for record in records)
        if any(record.label for record in blinded) or any(
            record.direction_b_wrt_a is not None for record in blinded
        ):
            raise FrozenTransportReadoutError("real supervision was not blinded")
        cache = GeometryArtifactCache(real_cache_root)
        geometry_config = GeometryBatchConfig()

        def prepare(rows: Tuple[TrainingPairRecord, ...]) -> RaggedGeometryBatch:
            prepared = prepare_real_geometry_batch(
                rows,
                mask_loader=dataset.mask_loader,
                geometry_config=geometry_config,
                geometry_cache=cache,
                arm=models.exact.arm,
            )
            payload = prepared.payload
            if not isinstance(payload, RaggedGeometryBatch):
                raise FrozenTransportReadoutError("real payload is not ragged geometry")
            return payload

        features = _extract_population(
            blinded,
            prepare=prepare,
            model=model,
            device=torch.device(device),
            batch_size=batch_size,
            top_k=top_k,
            labels=None,
        )
        # The side-table join starts only after all frozen inference is complete.
        labelled = replace(
            features,
            label=torch.tensor([record.label for record in records], dtype=torch.bool),
        )
        scores = {}
        for name in READOUT_NAMES:
            parameter = parameters[name]
            prediction = aggregate_frozen_readout(
                labelled,
                readout_name=name,
                beta=parameter.beta,
                intercept=parameter.intercept,
                arc_pooling=model.config.arc_pooling,
                direction_temperature=model.config.direction_aggregation_temperature,
            )
            scores[name] = {
                "probability": prediction.pair_probability.tolist(),
                "valid": prediction.pair_valid.tolist(),
                "direction": prediction.best_direction_index.tolist(),
            }
        validity = {tuple(scores[name]["valid"]) for name in READOUT_NAMES}
        if len(validity) != 1:
            raise FrozenTransportReadoutError("readout changed transport validity")
        populations: Dict[str, Any] = {}
        strict_count = len(strict.records)
        population_slices = {"strict547": (0, strict_count)}
        if include_balanced_1016:
            population_slices["balanced1016"] = (0, len(records))
        for population_name, (start, stop) in population_slices.items():
            selected_records = records[start:stop]
            populations[population_name] = {
                "pair_count": len(selected_records),
                "positive_count": sum(record.label for record in selected_records),
                "negative_count": sum(not record.label for record in selected_records),
                "methods": {
                    name: _real_method_metrics(
                        probability=scores[name]["probability"][start:stop],
                        valid=scores[name]["valid"][start:stop],
                        best_direction=scores[name]["direction"][start:stop],
                        records=selected_records,
                    )
                    for name in READOUT_NAMES
                },
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
                                _DIRECTION_LABELS[scores[name]["direction"][index]]
                                if scores[name]["valid"][index]
                                and 0 <= scores[name]["direction"][index] < 4
                                else None
                            ),
                        }
                        for name in READOUT_NAMES
                    },
                }
            )
        result: Dict[str, Any] = {
            "schema_version": FROZEN_TRANSPORT_READOUT_VERSION,
            "status": REAL_STATUS,
            "scope": {
                "mask_only": True,
                "known_upright_orientation": True,
                "rotation_search": False,
                "rgb_or_text_used": False,
                "backbone_frozen": True,
                "existing_local_head_frozen": True,
                "readout_parameters_loaded_before_real_dataset": True,
                "real_labels_used_for_fit_or_selection": False,
                "real_supervision_blinded_before_geometry": True,
            },
            "pilot": _selected_exact_identity(summary),
            "synthetic_selection": dict(fit["selection"]),
            "readout_parameters": {
                name: parameters[name].to_dict() for name in READOUT_NAMES
            },
            "shared_frozen_forward_for_all_readouts": True,
            "populations": populations,
            "balanced_negative_semantics": (
                None if build is None else build.receipt["negative_semantics"]
            ),
            "pairs": pair_rows,
            "runtime_seconds": {"total": time.perf_counter() - started},
        }
    finally:
        strict.mask_loader.close()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
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
    fit.add_argument("--mask-root", type=Path, required=True)
    fit.add_argument("--synthetic-cache-root", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--batch-size", type=_positive_int, default=8)
    fit.add_argument("--top-k", type=_positive_int, default=3)
    fit.add_argument("--fit-max-iterations", type=_positive_int, default=64)
    fit.add_argument("--validation-fraction", type=float, default=0.2)
    fit.add_argument("--population-snapshot", type=Path)

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
    real.add_argument("--include-balanced-1016", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "fit-synthetic":
        result = fit_synthetic(
            pilot_summary=arguments.pilot_summary,
            mask_root=arguments.mask_root,
            synthetic_cache_root=arguments.synthetic_cache_root,
            output=arguments.output,
            device=arguments.device,
            batch_size=arguments.batch_size,
            top_k=arguments.top_k,
            fit_max_iterations=arguments.fit_max_iterations,
            validation_fraction=arguments.validation_fraction,
            population_snapshot=arguments.population_snapshot,
        )
        compact = {
            "status": result["status"],
            "output": str(arguments.output),
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
            include_balanced_1016=arguments.include_balanced_1016,
        )
        compact = {
            "status": result["status"],
            "output": str(arguments.output),
            "synthetic_winner": result["synthetic_selection"]["winner"],
            "populations": {
                population: {
                    name: {
                        "auroc": row["ranking"]["row"]["auroc"],
                        "auprc": row["ranking"]["row"]["auprc"],
                        "direction_accuracy": row["direction"][
                            "accuracy_valid_only"
                        ],
                    }
                    for name, row in values["methods"].items()
                }
                for population, values in result["populations"].items()
            },
            "runtime_seconds": result["runtime_seconds"],
        }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "FROZEN_TRANSPORT_READOUT_VERSION",
    "FrozenTransportReadoutError",
    "evaluate_real",
    "fit_synthetic",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

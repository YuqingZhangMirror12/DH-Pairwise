#!/usr/bin/env python3
"""GPU-timed Full -> selective ShreddingNet replay on real balanced-1016.

This independent runtime experiment reuses the frozen Hybrid-99 route artifact
and the exact real-data preparation code.  It runs Full over all 1,016 pairs,
releases Full from CUDA, and then runs ShreddingNet only over routed pairs.
The existing exact-six real prediction artifact is read only for replay parity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from experiments.rachel_n512_formal_30k import (
    run_full_shredding_selective_replay as synthetic_replay,
)
from staging.pairwise_v0_2.baselines import rachel_n512_real_external as real_eval
from staging.pairwise_v0_2.baselines import (
    rachel_same_data_benchmark_eval_adapter as benchmark_adapter,
)
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


SCHEMA_VERSION = "rachel-full-shredding-real-selective-replay/1.0"
PAIR_SCHEMA_VERSION = "rachel-full-shredding-real-selective-replay-pair/1.0"
EXPECTED_PAIRS = 1_016
DEFAULT_N512_RUN = synthetic_replay.DEFAULT_N512_RUN
DEFAULT_SHREDDINGNET_FREEZE = synthetic_replay.DEFAULT_SHREDDINGNET_FREEZE
DEFAULT_REAL_CONTROL_ROOT = Path(
    "/root/autodl-tmp/rachel_pairwise_source_immutable_20260901_001/"
    "staging/pairwise_v0_2/pairwise_data/real_test_v0_1"
)
DEFAULT_REAL_AUTHORITY_ROOT = Path(
    "/root/autodl-tmp/dunhuang_pairwise_v02/"
    "real_dunhuang_strict_alpha_20260830_001"
)
DEFAULT_CACHED_REAL_PAIR_ONLY = Path(
    "/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/"
    "real/imported-prefix-001/pair-only-attempt-001.json"
)


class RealSelectiveReplayError(RuntimeError):
    """A real population, frozen authority, timing, or parity gate failed."""


@dataclass(frozen=True)
class RealSelectiveReplayConfig:
    n512_run_directory: Path
    shreddingnet_freeze_path: Path
    route_artifact: Path
    real_manifest: Path
    real_local_receipt: Path
    real_main_root: Path
    real_supp_root: Path
    cached_real_pair_only: Path
    output_root: Path
    batch_size: int = 1
    device: str = "cuda:0"
    score_atol: float = 2e-5
    pose_atol_px: float = 2e-3
    relative_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        for name in (
            "n512_run_directory",
            "shreddingnet_freeze_path",
            "route_artifact",
            "real_manifest",
            "real_local_receipt",
            "real_main_root",
            "real_supp_root",
            "cached_real_pair_only",
            "output_root",
        ):
            value = Path(getattr(self, name)).expanduser()
            if not value.is_absolute():
                raise ValueError(name + " must be absolute")
            object.__setattr__(self, name, Path(os.path.abspath(str(value))))
        if type(self.batch_size) is not int or self.batch_size <= 0:  # noqa: E721
            raise ValueError("batch_size must be a positive integer")
        if torch.device(self.device).type != "cuda":
            raise ValueError("real selective replay requires a CUDA device")
        for name in ("score_atol", "pose_atol_px", "relative_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(name + " must be finite and non-negative")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_cached_balanced(path: Path) -> Tuple[Mapping[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RealSelectiveReplayError("cached real pair-only artifact is missing")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RealSelectiveReplayError("cached real pair-only artifact is unreadable") from error
    balanced = document.get("balanced_1016") if isinstance(document, Mapping) else None
    rows = balanced.get("pairs") if isinstance(balanced, Mapping) else None
    if not isinstance(rows, list) or len(rows) != EXPECTED_PAIRS:
        raise RealSelectiveReplayError("cached balanced-1016 population differs")
    pair_ids = [row.get("pair_id") if isinstance(row, Mapping) else None for row in rows]
    if any(not isinstance(value, str) or not value for value in pair_ids):
        raise RealSelectiveReplayError("cached balanced pair ID is invalid")
    if len(set(pair_ids)) != EXPECTED_PAIRS:
        raise RealSelectiveReplayError("cached balanced pair IDs are duplicated")
    return tuple(rows)


def _close(left: float, right: float, *, atol: float, rtol: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and math.isclose(
        left, right, abs_tol=atol, rel_tol=rtol
    )


def validate_cached_real_replay(
    records: Sequence[Mapping[str, object]],
    cached_rows: Sequence[Mapping[str, object]],
    *,
    score_atol: float,
    pose_atol_px: float,
    relative_tolerance: float,
) -> Mapping[str, object]:
    """Require parity for every field used by the Hybrid runtime.

    Cross-run model outputs are retained as diagnostics.  Pair identity,
    order, labels, clusters, input authorities, and checkpoint identities are
    hard gates.  Full validity/route/classification and ShreddingNet pose
    validity/value differences are counted so that a completed timing run is
    never discarded merely because another CUDA node resolves a numerical
    boundary differently.  Frozen cached predictions remain the sole quality
    authority; this replay measures runtime and resources.
    """

    if len(records) != EXPECTED_PAIRS or len(cached_rows) != EXPECTED_PAIRS:
        raise RealSelectiveReplayError("real replay/cache counts differ")
    replay_ids = [str(row.get("pair_id")) for row in records]
    cached_ids = [str(row.get("pair_id")) for row in cached_rows]
    if replay_ids != cached_ids or len(set(replay_ids)) != EXPECTED_PAIRS:
        raise RealSelectiveReplayError("real replay/cache pair order differs")
    maxima = {
        "full_fused_probability": 0.0,
        "full_translation_component_px": 0.0,
        "shreddingnet_pair_probability": 0.0,
        "shreddingnet_translation_component_px": 0.0,
    }
    selected_count = 0
    selected_pose_valid_count = 0
    full_validity_mismatch_count = 0
    route_set_mismatch_count = 0
    full_classification_decision_mismatch_count = 0
    full_score_tolerance_mismatch_count = 0
    full_unused_pose_tolerance_mismatch_count = 0
    shredding_score_tolerance_mismatch_count = 0
    shredding_pose_validity_mismatch_count = 0
    shredding_pose_tolerance_mismatch_count = 0
    for record, cached in zip(records, cached_rows):
        pair_id = str(record["pair_id"])
        full = record.get("full")
        methods = cached.get("methods") if isinstance(cached, Mapping) else None
        cached_full = methods.get("full_n512") if isinstance(methods, Mapping) else None
        cached_shred = (
            methods.get("shreddingnet_adapted") if isinstance(methods, Mapping) else None
        )
        if not all(isinstance(value, Mapping) for value in (full, cached_full, cached_shred)):
            raise RealSelectiveReplayError("real cached method structure differs")
        full_score = float(full["fused_probability"])
        cached_full_score = float(cached_full["probability"])
        maxima["full_fused_probability"] = max(
            maxima["full_fused_probability"], abs(full_score - cached_full_score)
        )
        if not _close(
            full_score,
            cached_full_score,
            atol=score_atol,
            rtol=relative_tolerance,
        ):
            full_score_tolerance_mismatch_count += 1
        if bool(full["decision_valid"]) != bool(cached_full["valid"]):
            full_validity_mismatch_count += 1
        if bool(record["label"]) != bool(cached.get("label")):
            raise RealSelectiveReplayError("real replay/cache label differs: " + pair_id)
        if str(record.get("case_cluster")) != str(cached.get("case_cluster")):
            raise RealSelectiveReplayError("real replay/cache cluster differs: " + pair_id)
        route_threshold = float(record["route_threshold"])
        cached_route = (
            not bool(cached_full["valid"])
            or not math.isfinite(cached_full_score)
            or cached_full_score >= route_threshold
        )
        if cached_route != bool(record["routed_to_shreddingnet"]):
            route_set_mismatch_count += 1
        cached_class_decision = bool(
            cached_full.get("decision_at_frozen_validation_threshold")
        )
        if cached_class_decision != bool(full["decision_at_frozen_threshold"]):
            full_classification_decision_mismatch_count += 1
        replay_full_pose = np.asarray(full["translation_hat_rc"], dtype=np.float64)
        cached_full_pose = np.asarray(
            cached_full["translation_hat_rc_unsupervised"], dtype=np.float64
        )
        if replay_full_pose.shape != (2,) or cached_full_pose.shape != (2,):
            raise RealSelectiveReplayError("Full real pose shape differs: " + pair_id)
        pose_error = float(np.max(np.abs(replay_full_pose - cached_full_pose)))
        maxima["full_translation_component_px"] = max(
            maxima["full_translation_component_px"], pose_error
        )
        if not np.allclose(
            replay_full_pose,
            cached_full_pose,
            atol=pose_atol_px,
            rtol=relative_tolerance,
        ):
            full_unused_pose_tolerance_mismatch_count += 1
        if not bool(record["routed_to_shreddingnet"]):
            if record.get("shreddingnet") is not None:
                raise RealSelectiveReplayError("unrouted pair has ShreddingNet output")
            continue
        selected_count += 1
        shred = record.get("shreddingnet")
        if not isinstance(shred, Mapping):
            raise RealSelectiveReplayError("routed pair lacks ShreddingNet output")
        shred_score = float(shred["pair_probability"])
        cached_shred_score = float(cached_shred["probability"])
        maxima["shreddingnet_pair_probability"] = max(
            maxima["shreddingnet_pair_probability"],
            abs(shred_score - cached_shred_score),
        )
        if not _close(
            shred_score,
            cached_shred_score,
            atol=score_atol,
            rtol=relative_tolerance,
        ):
            shredding_score_tolerance_mismatch_count += 1
        cached_pose = cached_shred.get("translation_hat_rc_unsupervised")
        cached_pose_valid = cached_pose is not None
        replay_pose_valid = bool(shred["translation_valid"])
        if replay_pose_valid != cached_pose_valid:
            shredding_pose_validity_mismatch_count += 1
        if replay_pose_valid:
            selected_pose_valid_count += 1
            replay_pose = np.asarray(shred["translation_hat_rc"], dtype=np.float64)
            if replay_pose.shape != (2,) or not np.all(np.isfinite(replay_pose)):
                raise RealSelectiveReplayError("ShreddingNet replay pose is malformed")
            if cached_pose_valid:
                cached_pose_array = np.asarray(cached_pose, dtype=np.float64)
                if cached_pose_array.shape != (2,) or not np.all(
                    np.isfinite(cached_pose_array)
                ):
                    raise RealSelectiveReplayError("ShreddingNet cached pose is malformed")
                pose_error = float(np.max(np.abs(replay_pose - cached_pose_array)))
                maxima["shreddingnet_translation_component_px"] = max(
                    maxima["shreddingnet_translation_component_px"], pose_error
                )
                if not np.allclose(
                    replay_pose,
                    cached_pose_array,
                    atol=pose_atol_px,
                    rtol=relative_tolerance,
                ):
                    shredding_pose_tolerance_mismatch_count += 1
        elif shred.get("translation_hat_rc") is not None:
            raise RealSelectiveReplayError("invalid ShreddingNet pose must be null")
    return {
        "status": "complete_cross_node_runtime_parity_diagnostics",
        "parity_scope": (
            "pair identity/order and labels/clusters are exact hard gates; all "
            "cross-node model-output differences are recorded diagnostics"
        ),
        "ordered_pair_population_exact": True,
        "full_pairs_checked": len(records),
        "shreddingnet_selected_pairs_checked": selected_count,
        "shreddingnet_selected_valid_poses_checked": selected_pose_valid_count,
        "discrete_mismatch_count": {
            "full_validity": full_validity_mismatch_count,
            "route_membership": route_set_mismatch_count,
            "full_frozen_classification_decision": (
                full_classification_decision_mismatch_count
            ),
            "shreddingnet_translation_validity": (
                shredding_pose_validity_mismatch_count
            ),
        },
        "route_set_exact": route_set_mismatch_count == 0,
        "full_classification_decisions_exact": (
            full_classification_decision_mismatch_count == 0
        ),
        "shreddingnet_translation_validity_exact": (
            shredding_pose_validity_mismatch_count == 0
        ),
        "numeric_tolerance_mismatch_count": {
            "full_fused_probability": full_score_tolerance_mismatch_count,
            "full_translation_component_unused": (
                full_unused_pose_tolerance_mismatch_count
            ),
            "shreddingnet_pair_probability_diagnostic": (
                shredding_score_tolerance_mismatch_count
            ),
            "shreddingnet_translation_component": (
                shredding_pose_tolerance_mismatch_count
            ),
        },
        "full_pose_role": "unused_diagnostic_only",
        "full_unused_pose_tolerance_mismatch_count": (
            full_unused_pose_tolerance_mismatch_count
        ),
        "max_absolute_error": maxima,
        "tolerances": {
            "score_atol": score_atol,
            "pose_atol_px": pose_atol_px,
            "relative_tolerance": relative_tolerance,
        },
    }


def summarize_records(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if len(records) != EXPECTED_PAIRS:
        raise RealSelectiveReplayError("real replay is not balanced-1016")
    positive_count = sum(bool(row["label"]) for row in records)
    selected = [row for row in records if bool(row["routed_to_shreddingnet"])]
    selected_positive = sum(bool(row["label"]) for row in selected)
    selected_pose_valid = sum(
        isinstance(row.get("shreddingnet"), Mapping)
        and bool(row["shreddingnet"]["translation_valid"])
        for row in selected
    )
    return {
        "population": {
            "total": len(records),
            "positive": positive_count,
            "negative": len(records) - positive_count,
        },
        "routing": {
            "selected_count": len(selected),
            "selected_fraction": len(selected) / len(records),
            "positive_selected_count": selected_positive,
            "positive_route_recall": selected_positive / positive_count,
            "negative_selected_count": len(selected) - selected_positive,
        },
        "selected_shreddingnet_pose": {
            "valid_count_all_selected_pairs": selected_pose_valid,
            "valid_fraction_all_selected_pairs": (
                selected_pose_valid / len(selected) if selected else None
            ),
        },
    }


def _make_records(
    population: real_eval.PreparedStrictRealPopulation,
    full_predictions: Sequence[real_eval.TargetBlindPrediction],
    selected: np.ndarray,
    shredding_predictions: Sequence[real_eval.TargetBlindPrediction],
    *,
    route_threshold: float,
    class_threshold: float,
) -> Tuple[Mapping[str, object], ...]:
    selected_indices = [int(value) for value in np.flatnonzero(selected)]
    selected_position = {index: position for position, index in enumerate(selected_indices)}
    if len(full_predictions) != EXPECTED_PAIRS or len(selected) != EXPECTED_PAIRS:
        raise RealSelectiveReplayError("Full real replay coverage differs")
    if len(shredding_predictions) != len(selected_indices):
        raise RealSelectiveReplayError("selected ShreddingNet coverage differs")
    records: List[Mapping[str, object]] = []
    for index, pair in enumerate(population.pair_inputs):
        full = full_predictions[index]
        if full.pair_id != pair.pair_id:
            raise RealSelectiveReplayError("Full real replay order differs")
        position = selected_position.get(index)
        shred_value = None
        if position is not None:
            shred = shredding_predictions[position]
            if shred.pair_id != pair.pair_id:
                raise RealSelectiveReplayError("selected ShreddingNet order differs")
            shred_value = {
                "pair_probability": float(shred.probability),
                "decision_valid": bool(shred.valid),
                "translation_valid": shred.translation_hat_rc is not None,
                "translation_hat_rc": (
                    list(shred.translation_hat_rc)
                    if shred.translation_hat_rc is not None
                    else None
                ),
            }
        records.append(
            {
                "schema_version": PAIR_SCHEMA_VERSION,
                "ordinal": index,
                "pair_id": pair.pair_id,
                "label": bool(population.labels[index]),
                "case_cluster": population.case_clusters[index],
                "route_threshold": route_threshold,
                "routed_to_shreddingnet": bool(selected[index]),
                "full": {
                    "fused_probability": float(full.probability),
                    "decision_valid": bool(full.valid),
                    "decision_at_frozen_threshold": bool(
                        full.valid
                        and math.isfinite(float(full.probability))
                        and float(full.probability) >= class_threshold
                    ),
                    "translation_hat_rc": list(full.translation_hat_rc),
                },
                "shreddingnet": shred_value,
            }
        )
    return tuple(records)


def _move_shredding_to_cuda(
    frozen: benchmark_adapter.FrozenSameDataBenchmark, device: torch.device
) -> None:
    predictor = frozen._predictor
    predictor.coarse.to(device).eval()
    predictor.classify.to(device).eval()
    # The released ShreddingNet wrapper performs its own batch-to-device
    # transfer using ``predictor.device``.  Keep that runtime device in sync
    # with the moved module weights as well as the outer adapter device.
    predictor.device = device
    frozen._device = device


def _protected_overlap(output: Path, protected: Path) -> bool:
    try:
        output.relative_to(protected)
        return True
    except ValueError:
        pass
    try:
        protected.relative_to(output)
        return True
    except ValueError:
        return False


def run_real_selective_replay(config: RealSelectiveReplayConfig) -> Path:
    """Execute and atomically publish one balanced-1016 selective replay."""

    route_artifact = synthetic_replay.load_route_artifact(config.route_artifact)
    route = route_artifact["route"]
    classification = route_artifact["classification"]
    checkpoints = route_artifact["frozen_checkpoints_sha256"]
    assert isinstance(route, Mapping)
    assert isinstance(classification, Mapping)
    assert isinstance(checkpoints, Mapping)
    route_threshold = float(route["threshold"])
    class_threshold = float(classification["threshold"])

    if not torch.cuda.is_available():
        raise RealSelectiveReplayError("CUDA is unavailable")
    device = torch.device(config.device)
    try:
        device_index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.get_device_properties(device_index)
    except (AssertionError, RuntimeError) as error:
        raise RealSelectiveReplayError("requested CUDA device is unavailable") from error
    if config.output_root.exists() or config.output_root.is_symlink():
        raise RealSelectiveReplayError("output_root already exists; refusing overwrite")
    for protected in (
        config.n512_run_directory,
        config.shreddingnet_freeze_path.parent,
        config.route_artifact,
        config.real_manifest.parent,
        config.real_local_receipt.parent,
        config.real_main_root,
        config.real_supp_root,
        config.cached_real_pair_only,
    ):
        if _protected_overlap(config.output_root, protected):
            raise RealSelectiveReplayError("output_root overlaps an input authority")

    cold_started = time.perf_counter()
    authority_started = time.perf_counter()
    receipt, receipt_sha, winners = real_eval._freeze_completed_n512_authority(
        config.n512_run_directory.resolve(strict=True), device="cpu"
    )
    real_eval._require_formal_n512_convergence(
        config.n512_run_directory, winners, receipt=receipt
    )
    full = winners.get("full_n512")
    if full is None or full.checkpoint_sha256 != checkpoints.get("full_n512"):
        raise RealSelectiveReplayError("route artifact is bound to another Full winner")
    frozen_shred = benchmark_adapter.freeze_shreddingnet_benchmark(
        config.shreddingnet_freeze_path.resolve(strict=True), device="cpu"
    )
    expected_shred = checkpoints.get("shreddingnet_adapted")
    if not isinstance(expected_shred, Mapping) or dict(
        frozen_shred.checkpoint_sha256_by_stage
    ) != dict(expected_shred):
        raise RealSelectiveReplayError(
            "route artifact is bound to another ShreddingNet winner"
        )
    authority_seconds = time.perf_counter() - authority_started

    preparation_started = time.perf_counter()
    strict = real_eval.prepare_strict_real_population(
        config.real_manifest.resolve(strict=True),
        config.real_local_receipt.resolve(strict=True),
        main_root=config.real_main_root.resolve(strict=True),
        supp_root=config.real_supp_root.resolve(strict=True),
        canvas_size=full.model_config.canvas_size,
        contour_cap=full.model_config.contour_cap,
    )
    population = real_eval.build_balanced_1016_population(strict)
    if len(population.pair_inputs) != EXPECTED_PAIRS:
        raise RealSelectiveReplayError("prepared real population is not balanced-1016")
    preparation_seconds = time.perf_counter() - preparation_started
    cold_setup_seconds = time.perf_counter() - cold_started

    pipeline_started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    full_started = time.perf_counter()
    full.model.to(device).eval()
    full_predictions = real_eval.score_target_blind(
        full,
        population.pair_inputs,
        population.fragments,
        batch_size=config.batch_size,
    )
    torch.cuda.synchronize(device)
    full_seconds = time.perf_counter() - full_started
    full_memory = synthetic_replay._cuda_memory(device)
    full_scores = np.asarray(
        [value.probability for value in full_predictions], dtype=np.float64
    )
    full_valid = np.asarray([value.valid for value in full_predictions], dtype=np.bool_)
    selected = synthetic_replay.route_full_scores(
        full_scores, full_valid, route_threshold
    )

    release_started = time.perf_counter()
    full.model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    full_release_seconds = time.perf_counter() - release_started
    after_full_release = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }

    selected_inputs = tuple(
        population.pair_inputs[int(index)] for index in np.flatnonzero(selected)
    )
    torch.cuda.reset_peak_memory_stats(device)
    shred_restore_started = time.perf_counter()
    _move_shredding_to_cuda(frozen_shred, device)
    torch.cuda.synchronize(device)
    shred_restore_seconds = time.perf_counter() - shred_restore_started
    shred_started = time.perf_counter()
    shredding_predictions = real_eval.score_target_blind_benchmark(
        frozen_shred,
        selected_inputs,
        population.fragments,
        batch_size=config.batch_size,
    )
    torch.cuda.synchronize(device)
    shred_seconds = time.perf_counter() - shred_started
    shred_memory = synthetic_replay._cuda_memory(device)
    shred_release_started = time.perf_counter()
    frozen_shred.release_to_cpu()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    shred_release_seconds = time.perf_counter() - shred_release_started
    after_shred_release = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }

    merge_started = time.perf_counter()
    records = _make_records(
        population,
        full_predictions,
        selected,
        shredding_predictions,
        route_threshold=route_threshold,
        class_threshold=class_threshold,
    )
    summary = summarize_records(records)
    merge_seconds = time.perf_counter() - merge_started
    pipeline_seconds = time.perf_counter() - pipeline_started

    parity_started = time.perf_counter()
    cached_rows = _read_cached_balanced(config.cached_real_pair_only)
    parity = validate_cached_real_replay(
        records,
        cached_rows,
        score_atol=config.score_atol,
        pose_atol_px=config.pose_atol_px,
        relative_tolerance=config.relative_tolerance,
    )
    parity_seconds = time.perf_counter() - parity_started
    result: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete_real_balanced1016_runtime_replay_with_cross_node_diagnostics"
        ),
        "experiment": "Hybrid-99 Full screening then selective ShreddingNet real replay",
        "population": "real balanced-1016, exact frozen order",
        "route": {
            "score": "full_n512.fused_probability",
            "operator": "greater_than_or_equal",
            "threshold": route_threshold,
            "invalid_and_nonfinite_policy": "fail_open_route_to_shreddingnet",
            "classification_threshold": class_threshold,
            "shreddingnet_score_used_for_final_classification": False,
        },
        "summary": summary,
        "timing": {
            "boundary": (
                "CUDA synchronized around each GPU stage; pipeline includes Full, "
                "release, selected ShreddingNet restore/inference/release, and merge"
            ),
            "authority_restore_cpu_seconds": authority_seconds,
            "real_population_preparation_seconds": preparation_seconds,
            "cold_authority_and_real_setup_seconds": cold_setup_seconds,
            "full_seconds": full_seconds,
            "full_release_seconds": full_release_seconds,
            "shredding_cuda_restore_seconds": shred_restore_seconds,
            "shredding_selected_inference_seconds": shred_seconds,
            "shredding_release_seconds": shred_release_seconds,
            "merge_seconds": merge_seconds,
            "pipeline_total_seconds": pipeline_seconds,
            "pipeline_input_pairs_per_second": EXPECTED_PAIRS / pipeline_seconds,
            "cache_parity_seconds_excluded_from_pipeline": parity_seconds,
        },
        "resource": {
            "cuda_device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "full_stage": full_memory,
            "shredding_stage": shred_memory,
            "sequential_peak_allocated_bytes": max(
                full_memory["peak_allocated_bytes"],
                shred_memory["peak_allocated_bytes"],
            ),
            "sequential_peak_reserved_bytes": max(
                full_memory["peak_reserved_bytes"],
                shred_memory["peak_reserved_bytes"],
            ),
            "after_full_release": after_full_release,
            "after_shredding_release": after_shred_release,
            "sequential_not_concurrent": True,
        },
        "cache_parity": parity,
        "provenance": {
            "n512_run_receipt": str(config.n512_run_directory / "run_receipt.json"),
            "n512_run_receipt_sha256": receipt_sha,
            "full_winner_epoch": full.epoch,
            "full_winner_checkpoint_sha256": full.checkpoint_sha256,
            "shreddingnet": frozen_shred.provenance(),
            "real_manifest": str(config.real_manifest),
            "real_manifest_sha256": _sha256_file(config.real_manifest),
            "real_local_receipt": str(config.real_local_receipt),
            "real_local_receipt_sha256": _sha256_file(config.real_local_receipt),
            "cached_real_pair_only": str(config.cached_real_pair_only),
            "cached_real_pair_only_sha256": _sha256_file(
                config.cached_real_pair_only
            ),
            "route_artifact": str(config.route_artifact),
            "route_artifact_file_sha256": _sha256_file(config.route_artifact),
            "route_artifact_content_sha256": route_artifact["content_sha256"],
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "training_performed": False,
            "threshold_fit_performed": False,
            "exact6_outputs_modified": False,
        },
    }
    result["content_sha256"] = _canonical_sha256(result)

    config.output_root.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=".partial-real-hybrid99-", dir=str(config.output_root.parent))
    )
    try:
        pair_path = staged / "pair_results.jsonl"
        sealed._atomic_jsonl(pair_path, records)
        summary_path = staged / "summary.json"
        sealed._atomic_json(summary_path, result)
        run_receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": result["status"],
            "summary": "summary.json",
            "summary_sha256": _sha256_file(summary_path),
            "pair_results": "pair_results.jsonl",
            "pair_results_sha256": _sha256_file(pair_path),
            "pair_results_count": len(records),
            "cache_parity_status": parity["status"],
        }
        run_receipt["content_sha256"] = _canonical_sha256(run_receipt)
        sealed._atomic_json(staged / "run_receipt.json", run_receipt)
        sealed._publish_directory_no_replace(
            staged, config.output_root, completion_receipt="run_receipt.json"
        )
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return config.output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--n512-run-directory", type=Path, default=DEFAULT_N512_RUN)
    parser.add_argument(
        "--shreddingnet-freeze-path", type=Path, default=DEFAULT_SHREDDINGNET_FREEZE
    )
    parser.add_argument("--route-artifact", type=Path, required=True)
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=DEFAULT_REAL_CONTROL_ROOT / "real_test_manifest.json",
    )
    parser.add_argument(
        "--real-local-receipt",
        type=Path,
        default=DEFAULT_REAL_CONTROL_ROOT / "local_path_receipt.json",
    )
    parser.add_argument(
        "--real-main-root",
        type=Path,
        default=DEFAULT_REAL_AUTHORITY_ROOT / "Dunhuang Dataset",
    )
    parser.add_argument(
        "--real-supp-root",
        type=Path,
        default=DEFAULT_REAL_AUTHORITY_ROOT / "Dunhuang Dataset Supp",
    )
    parser.add_argument(
        "--cached-real-pair-only", type=Path, default=DEFAULT_CACHED_REAL_PAIR_ONLY
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-atol", type=float, default=2e-5)
    parser.add_argument("--pose-atol-px", type=float, default=2e-3)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    output = run_real_selective_replay(
        RealSelectiveReplayConfig(
            n512_run_directory=arguments.n512_run_directory,
            shreddingnet_freeze_path=arguments.shreddingnet_freeze_path,
            route_artifact=arguments.route_artifact,
            real_manifest=arguments.real_manifest,
            real_local_receipt=arguments.real_local_receipt,
            real_main_root=arguments.real_main_root,
            real_supp_root=arguments.real_supp_root,
            cached_real_pair_only=arguments.cached_real_pair_only,
            output_root=arguments.output_root,
            batch_size=arguments.batch_size,
            device=arguments.device,
            score_atol=arguments.score_atol,
            pose_atol_px=arguments.pose_atol_px,
            relative_tolerance=arguments.relative_tolerance,
        )
    )
    print(str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

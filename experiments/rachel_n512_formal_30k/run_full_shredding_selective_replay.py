"""GPU-timed Full -> selective ShreddingNet replay on Rachel synthetic test.

This is an independent post-evaluation runtime experiment.  It restores the
already frozen Full N=512 winner, screens all 3,000 synthetic-test pairs with
the fused probability, releases Full from CUDA, and runs the already frozen
ShreddingNet-adapted pose path only for routed pairs.  Existing exact-six
outputs are read-only numerical authorities; this runner never modifies them
and publishes to a new output directory without replacement.
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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import DataLoader, Subset

from staging.pairwise_v0_2.baselines import (
    rachel_same_data_benchmark_eval_adapter as benchmark_adapter,
)
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Pairwise
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import (
    RachelBatch,
    RachelPairDataset,
)
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


SCHEMA_VERSION = "rachel-full-shredding-selective-replay/1.0"
PAIR_SCHEMA_VERSION = "rachel-full-shredding-selective-replay-pair/1.0"
EXPECTED_SYNTHETIC_PAIRS = 3_000
DEFAULT_ROUTE_THRESHOLD = 0.011353014037013054
ROUTE_ARTIFACT_SCHEMA_VERSION = "rachel-full-shredding-hybrid-route/1.0"
ROUTE_ARTIFACT_STATUS = "frozen_validation_only_before_hybrid_evaluation"
DEFAULT_N512_RUN = Path(
    "/root/autodl-tmp/rachel_n512_convergence_20260901_001/"
    "run-convergence-50a8cffb0ae92614"
)
DEFAULT_SHREDDINGNET_FREEZE = Path(
    "/root/autodl-tmp/rachel_same_data_benchmark_direct_20260904_001/"
    "shreddingnet_train/train_val_freeze.json"
)
DEFAULT_DATASET_ROOT = Path("/root/autodl-tmp/dataset_rachel_pairwise_n512_v1")
DEFAULT_CACHED_SYNTHETIC_ROOT = Path(
    "/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/"
    "synthetic/imported-prefix-001/test-8f45e0f44c0ace48"
)


class SelectiveReplayError(RuntimeError):
    """A frozen authority, replay, timing, or cache-parity gate failed."""


@dataclass(frozen=True)
class SelectiveReplayConfig:
    n512_run_directory: Path
    shreddingnet_freeze_path: Path
    dataset_root: Path
    cached_synthetic_root: Path
    route_artifact: Path
    output_root: Path
    batch_size: int = 16
    num_workers: int = 8
    device: str = "cuda:0"
    score_atol: float = 2e-5
    pose_atol_px: float = 2e-3
    relative_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        for name in (
            "n512_run_directory",
            "shreddingnet_freeze_path",
            "dataset_root",
            "cached_synthetic_root",
            "route_artifact",
            "output_root",
        ):
            value = Path(getattr(self, name)).expanduser()
            if not value.is_absolute():
                raise ValueError(name + " must be absolute")
            object.__setattr__(self, name, Path(os.path.abspath(str(value))))
        if type(self.batch_size) is not int or self.batch_size <= 0:  # noqa: E721
            raise ValueError("batch_size must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers < 0:  # noqa: E721
            raise ValueError("num_workers must be a non-negative integer")
        for name in ("score_atol", "pose_atol_px", "relative_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(name + " must be finite and non-negative")
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("selective replay requires a CUDA device")


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


def load_route_artifact(path: Path) -> Mapping[str, object]:
    """Load a content-bound validation-only Hybrid route authority."""

    lexical_path = Path(path).expanduser()
    if lexical_path.is_symlink():
        raise SelectiveReplayError("route authority may not be a symlink")
    path = lexical_path.resolve(strict=True)
    if not path.is_file() or path.name != "route_artifact.json":
        raise SelectiveReplayError("route authority must be route_artifact.json")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectiveReplayError("route artifact is unreadable strict JSON") from error
    if not isinstance(artifact, Mapping):
        raise SelectiveReplayError("route artifact root is not an object")
    content_sha = artifact.get("content_sha256")
    canonical = dict(artifact)
    canonical.pop("content_sha256", None)
    if (
        not isinstance(content_sha, str)
        or len(content_sha) != 64
        or _canonical_sha256(canonical) != content_sha
    ):
        raise SelectiveReplayError("route artifact content SHA-256 differs")
    route = artifact.get("route")
    classification = artifact.get("classification")
    pose = artifact.get("pose")
    protocol = artifact.get("protocol")
    checkpoints = artifact.get("frozen_checkpoints_sha256")
    threshold = route.get("threshold") if isinstance(route, Mapping) else None
    class_threshold = (
        classification.get("threshold")
        if isinstance(classification, Mapping)
        else None
    )
    if (
        artifact.get("schema_version") != ROUTE_ARTIFACT_SCHEMA_VERSION
        or artifact.get("status") != ROUTE_ARTIFACT_STATUS
        or not isinstance(route, Mapping)
        or route.get("score") != "full_n512.fused_probability"
        or route.get("rule")
        != "fail_open_if_invalid_else_probability_greater_than_or_equal_to_threshold"
        or route.get("threshold_fit_source") != "validation_only"
        or route.get("test_or_real_parameter_fit") is not False
        or route.get("top_k_or_budget_cap") is not None
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
        or not isinstance(classification, Mapping)
        or classification.get("score") != "full_n512.fused_probability"
        or classification.get("shreddingnet_score_used") is not False
        or isinstance(class_threshold, bool)
        or not isinstance(class_threshold, (int, float))
        or not float(threshold) < float(class_threshold)
        or not isinstance(pose, Mapping)
        or pose.get("compute_pose_for_shreddingnet_rejected_pairs") is not True
        or pose.get("unrouted_pairs_are_unconditional_pose_failures") is not True
        or not isinstance(protocol, Mapping)
        or protocol.get("test_or_real_read_by_this_command") is not False
        or protocol.get("single_primary_operating_point") is not True
        or protocol.get("threshold_sweep_on_test_or_real_forbidden") is not True
        or not isinstance(checkpoints, Mapping)
        or not isinstance(checkpoints.get("full_n512"), str)
        or not isinstance(checkpoints.get("shreddingnet_adapted"), Mapping)
    ):
        raise SelectiveReplayError("route artifact protocol fields differ")
    return artifact


def route_full_scores(
    scores: Sequence[float], valid: Sequence[bool], threshold: float
) -> np.ndarray:
    """Apply inclusive Hybrid-99 routing with fail-open invalid/nonfinite scores."""

    probability = np.asarray(scores, dtype=np.float64)
    validity = np.asarray(valid, dtype=np.bool_)
    if probability.ndim != 1 or validity.shape != probability.shape:
        raise ValueError("scores and valid must be aligned one-dimensional arrays")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("threshold must be finite and in [0,1]")
    return (~validity) | (~np.isfinite(probability)) | (probability >= threshold)


def _read_jsonl_unique(path: Path, expected_count: int) -> Tuple[Mapping[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise SelectiveReplayError("cached JSONL is missing: " + str(path))
    rows: List[Mapping[str, object]] = []
    seen = set()
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                pair_id = value.get("pair_id") if isinstance(value, Mapping) else None
                if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
                    raise SelectiveReplayError(
                        "cached pair_id is missing/duplicated at line {}".format(
                            line_number
                        )
                    )
                seen.add(pair_id)
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise SelectiveReplayError("cached JSONL is unreadable") from error
    if len(rows) != expected_count:
        raise SelectiveReplayError(
            "cached JSONL count differs: expected {}, observed {}".format(
                expected_count, len(rows)
            )
        )
    return tuple(rows)


def _close(left: float, right: float, *, atol: float, rtol: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and math.isclose(
        left, right, rel_tol=rtol, abs_tol=atol
    )


def validate_cached_replay(
    records: Sequence[Mapping[str, object]],
    cached_full_rows: Sequence[Mapping[str, object]],
    cached_shredding_rows: Sequence[Mapping[str, object]],
    *,
    score_atol: float,
    pose_atol_px: float,
    relative_tolerance: float,
) -> Mapping[str, object]:
    """Require per-pair Full and selected-Shredding cache parity."""

    full_by_id = {str(row.get("pair_id")): row for row in cached_full_rows}
    shred_by_id = {str(row.get("pair_id")): row for row in cached_shredding_rows}
    replay_ids = [str(row.get("pair_id")) for row in records]
    if (
        len(full_by_id) != len(cached_full_rows)
        or len(shred_by_id) != len(cached_shredding_rows)
        or set(replay_ids) != set(full_by_id)
        or set(replay_ids) != set(shred_by_id)
    ):
        raise SelectiveReplayError("replay/cache pair_id populations differ")

    full_score_max = 0.0
    full_pose_max = 0.0
    shred_score_max = 0.0
    shred_pose_max = 0.0
    selected_count = 0
    selected_pose_valid_count = 0
    for record in records:
        pair_id = str(record["pair_id"])
        cached_full = full_by_id[pair_id]
        cached_shred = shred_by_id[pair_id]
        full = record["full"]
        assert isinstance(full, Mapping)
        cached_full_score = float(cached_full["scores"]["fused"]["probability"])
        score_error = abs(float(full["fused_probability"]) - cached_full_score)
        full_score_max = max(full_score_max, score_error)
        if not _close(
            float(full["fused_probability"]),
            cached_full_score,
            atol=score_atol,
            rtol=relative_tolerance,
        ):
            raise SelectiveReplayError("Full score cache mismatch: " + pair_id)
        if bool(full["decision_valid"]) != bool(cached_full["decision"]["valid"]):
            raise SelectiveReplayError("Full validity cache mismatch: " + pair_id)
        full_pose = np.asarray(full["translation_hat_rc"], dtype=np.float64)
        cached_full_pose = np.asarray(
            cached_full["geometry"]["translation_hat_rc"], dtype=np.float64
        )
        if full_pose.shape != (2,) or cached_full_pose.shape != (2,):
            raise SelectiveReplayError("Full cached pose shape differs: " + pair_id)
        pose_error = float(np.max(np.abs(full_pose - cached_full_pose)))
        full_pose_max = max(full_pose_max, pose_error)
        if not np.allclose(
            full_pose,
            cached_full_pose,
            rtol=relative_tolerance,
            atol=pose_atol_px,
        ):
            raise SelectiveReplayError("Full pose cache mismatch: " + pair_id)

        if not bool(record["routed_to_shreddingnet"]):
            if record["shreddingnet"] is not None:
                raise SelectiveReplayError("unrouted pair unexpectedly has ShreddingNet output")
            continue
        selected_count += 1
        replay_shred = record["shreddingnet"]
        if not isinstance(replay_shred, Mapping):
            raise SelectiveReplayError("routed pair lacks ShreddingNet output")
        cached_shred_score = float(
            cached_shred["scores"]["pair_probability"]["probability"]
        )
        score_error = abs(float(replay_shred["pair_probability"]) - cached_shred_score)
        shred_score_max = max(shred_score_max, score_error)
        if not _close(
            float(replay_shred["pair_probability"]),
            cached_shred_score,
            atol=score_atol,
            rtol=relative_tolerance,
        ):
            raise SelectiveReplayError("ShreddingNet score cache mismatch: " + pair_id)
        cached_geometry = cached_shred["geometry"]
        cached_pose_valid = bool(cached_geometry["translation_prediction_valid"])
        replay_pose_valid = bool(replay_shred["translation_valid"])
        if replay_pose_valid != cached_pose_valid:
            raise SelectiveReplayError("ShreddingNet pose validity mismatch: " + pair_id)
        if replay_pose_valid:
            selected_pose_valid_count += 1
            replay_pose = np.asarray(replay_shred["translation_hat_rc"], dtype=np.float64)
            cached_pose = np.asarray(cached_geometry["translation_hat_rc"], dtype=np.float64)
            if replay_pose.shape != (2,) or cached_pose.shape != (2,):
                raise SelectiveReplayError("ShreddingNet cached pose shape differs: " + pair_id)
            pose_error = float(np.max(np.abs(replay_pose - cached_pose)))
            shred_pose_max = max(shred_pose_max, pose_error)
            if not np.allclose(
                replay_pose,
                cached_pose,
                rtol=relative_tolerance,
                atol=pose_atol_px,
            ):
                raise SelectiveReplayError("ShreddingNet pose cache mismatch: " + pair_id)
        elif replay_shred["translation_hat_rc"] is not None:
            raise SelectiveReplayError("invalid ShreddingNet pose must be null")

    return {
        "status": "passed",
        "pair_id_population_exact": True,
        "full_pairs_checked": len(records),
        "shreddingnet_selected_pairs_checked": selected_count,
        "shreddingnet_selected_valid_poses_checked": selected_pose_valid_count,
        "max_absolute_error": {
            "full_fused_probability": full_score_max,
            "full_translation_component_px": full_pose_max,
            "shreddingnet_pair_probability": shred_score_max,
            "shreddingnet_translation_component_px": shred_pose_max,
        },
        "tolerances": {
            "score_atol": score_atol,
            "pose_atol_px": pose_atol_px,
            "relative_tolerance": relative_tolerance,
        },
    }


def _cuda_memory(device: torch.device) -> Mapping[str, object]:
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 2**30),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved(device) / 2**30),
    }


def _full_inputs(batch: RachelBatch, device: torch.device) -> Tuple[torch.Tensor, ...]:
    return (
        sealed._tensor(batch.mask_a, device, torch.float32),
        sealed._tensor(batch.mask_b, device, torch.float32),
        sealed._tensor(batch.points_rc_a, device, torch.float32),
        sealed._tensor(batch.points_rc_b, device, torch.float32),
        sealed._tensor(batch.contour_valid_a, device, torch.bool),
        sealed._tensor(batch.contour_valid_b, device, torch.bool),
    )


def _run_full_stage(
    winner: object,
    loader: Iterable[RachelBatch],
    manifest_rows: Sequence[object],
    *,
    device: torch.device,
    precision: str,
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    if getattr(winner, "arm", None) != "full_n512" or not isinstance(
        getattr(winner, "model", None), RachelN512Pairwise
    ):
        raise SelectiveReplayError("frozen Full winner is missing")
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = winner.model.to(device).eval()
    pair_ids: List[str] = []
    scores: List[float] = []
    valid: List[bool] = []
    translations: List[List[float]] = []
    batch_count = 0
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            count = len(batch.pair_ids)
            expected = manifest_rows[cursor : cursor + count]
            if tuple(row.pair_id for row in expected) != tuple(batch.pair_ids):
                raise SelectiveReplayError("Full loader order differs from manifest")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                output = model(*_full_inputs(batch, device))
            score_cpu = output.fused_probability.detach().float().cpu().numpy()
            valid_cpu = output.decision_valid.detach().cpu().numpy().astype(np.bool_)
            pose_cpu = output.translation_hat_rc.detach().float().cpu().numpy()
            pair_ids.extend(batch.pair_ids)
            scores.extend(float(value) for value in score_cpu)
            valid.extend(bool(value) for value in valid_cpu)
            translations.extend(
                [float(value[0]), float(value[1])] for value in pose_cpu
            )
            cursor += count
            batch_count += 1
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    memory = _cuda_memory(device)
    if cursor != len(manifest_rows) or len(set(pair_ids)) != len(manifest_rows):
        raise SelectiveReplayError("Full replay coverage is incomplete")
    values = {
        "pair_ids": tuple(pair_ids),
        "scores": np.asarray(scores, dtype=np.float64),
        "valid": np.asarray(valid, dtype=np.bool_),
        "translations": np.asarray(translations, dtype=np.float64),
    }
    timing = {
        "seconds": seconds,
        "batch_count": batch_count,
        "pairs_per_second": len(pair_ids) / seconds,
        "gpu_memory": memory,
    }
    return values, timing


def _release_full(winner: object, device: torch.device) -> Mapping[str, object]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    winner.model = winner.model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return {
        "seconds": time.perf_counter() - started,
        "allocated_bytes_after_release": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes_after_release": int(torch.cuda.memory_reserved(device)),
    }


def _run_shredding_stage(
    freeze_path: Path,
    loader: Iterable[RachelBatch],
    expected_pair_ids: Sequence[str],
    *,
    device: torch.device,
) -> Tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    stage_started = time.perf_counter()
    restore_started = time.perf_counter()
    frozen = benchmark_adapter.freeze_shreddingnet_benchmark(
        freeze_path, device=device
    )
    torch.cuda.synchronize(device)
    restore_seconds = time.perf_counter() - restore_started
    inference_started = time.perf_counter()
    pair_ids: List[str] = []
    scores: List[float] = []
    valid: List[bool] = []
    translations: List[Optional[List[float]]] = []
    errors: List[Optional[float]] = []
    labels: List[bool] = []
    cursor = 0
    batch_count = 0
    for batch in loader:
        prediction = frozen.predict_batch(batch, return_correspondence=False)
        count = len(prediction.pair_ids)
        expected = tuple(expected_pair_ids[cursor : cursor + count])
        if tuple(prediction.pair_ids) != expected or tuple(batch.pair_ids) != expected:
            raise SelectiveReplayError("selected ShreddingNet order differs")
        for index, pair_id in enumerate(prediction.pair_ids):
            pose_valid = bool(prediction.translation_valid[index])
            target_valid = bool(batch.translation_valid[index])
            pose = (
                [float(value) for value in prediction.translation_hat_rc[index]]
                if pose_valid
                else None
            )
            error = None
            if pose_valid and target_valid:
                error = float(
                    np.linalg.norm(
                        prediction.translation_hat_rc[index].astype(np.float64)
                        - batch.translation_a_to_b_rc[index].astype(np.float64)
                    )
                )
            pair_ids.append(pair_id)
            scores.append(float(prediction.pair_probability[index]))
            valid.append(pose_valid)
            translations.append(pose)
            errors.append(error)
            labels.append(target_valid)
        cursor += count
        batch_count += 1
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    stage_seconds = time.perf_counter() - stage_started
    memory = _cuda_memory(device)
    provenance = frozen.provenance()
    release_started = time.perf_counter()
    frozen.release_to_cpu()
    del frozen
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    release_finished = time.perf_counter()
    if cursor != len(expected_pair_ids) or tuple(pair_ids) != tuple(expected_pair_ids):
        raise SelectiveReplayError("selected ShreddingNet coverage is incomplete")
    values = {
        "pair_ids": tuple(pair_ids),
        "scores": np.asarray(scores, dtype=np.float64),
        "translation_valid": np.asarray(valid, dtype=np.bool_),
        "translations": tuple(translations),
        "translation_errors": tuple(errors),
        "labels": np.asarray(labels, dtype=np.bool_),
    }
    timing = {
        "restore_seconds": restore_seconds,
        "inference_seconds": inference_seconds,
        "restore_plus_inference_seconds": stage_seconds,
        "release_seconds": release_finished - release_started,
        "batch_count": batch_count,
        "routed_pairs_per_inference_second": (
            len(pair_ids) / inference_seconds if pair_ids else None
        ),
        "gpu_memory": memory,
        "allocated_bytes_after_release": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes_after_release": int(torch.cuda.memory_reserved(device)),
    }
    return values, timing, provenance


def _hybrid_summary(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    total = len(records)
    positives = sum(bool(row["label"]) for row in records)
    negatives = total - positives
    selected = [row for row in records if bool(row["routed_to_shreddingnet"])]
    selected_positive = sum(bool(row["label"]) for row in selected)
    selected_negative = len(selected) - selected_positive
    valid_positive_errors = [
        float(row["shreddingnet"]["translation_l2_px"])
        for row in selected
        if bool(row["label"])
        and isinstance(row.get("shreddingnet"), Mapping)
        and row["shreddingnet"].get("translation_l2_px") is not None
    ]
    return {
        "population": {
            "total": total,
            "positive": positives,
            "negative": negatives,
        },
        "routing": {
            "selected_count": len(selected),
            "selected_fraction": len(selected) / total,
            "positive_selected_count": selected_positive,
            "positive_route_recall": selected_positive / positives,
            "negative_selected_count": selected_negative,
            "negative_rejection_fraction": (negatives - selected_negative) / negatives,
            "selected_positive_fraction": (
                selected_positive / len(selected) if selected else None
            ),
        },
        "pose": {
            "valid_positive_count": len(valid_positive_errors),
            "valid_positive_fraction_unconditional": len(valid_positive_errors)
            / positives,
            "median_te_px_conditional_on_valid": (
                float(np.median(valid_positive_errors)) if valid_positive_errors else None
            ),
            "p90_te_px_conditional_on_valid": (
                float(np.quantile(valid_positive_errors, 0.9))
                if valid_positive_errors
                else None
            ),
            "unconditional_recall": {
                "at_{}px".format(tolerance): sum(
                    error <= tolerance for error in valid_positive_errors
                )
                / positives
                for tolerance in (2, 5, 8, 10)
            },
        },
    }


def _make_records(
    manifest_rows: Sequence[object],
    full: Mapping[str, object],
    selected: np.ndarray,
    shredding: Mapping[str, object],
    route_threshold: float,
) -> Tuple[Mapping[str, object], ...]:
    selected_indices = np.flatnonzero(selected)
    selected_position = {int(index): position for position, index in enumerate(selected_indices)}
    records: List[Mapping[str, object]] = []
    for index, row in enumerate(manifest_rows):
        position = selected_position.get(index)
        shred = None
        if position is not None:
            pose_valid = bool(shredding["translation_valid"][position])
            error = shredding["translation_errors"][position]
            shred = {
                "pair_probability": float(shredding["scores"][position]),
                "translation_valid": pose_valid,
                "translation_hat_rc": shredding["translations"][position],
                "translation_l2_px": None if error is None else float(error),
            }
        records.append(
            {
                "schema_version": PAIR_SCHEMA_VERSION,
                "ordinal": index,
                "pair_id": row.pair_id,
                "label": bool(row.label),
                "cluster_id": row.cluster_id,
                "route_threshold": route_threshold,
                "routed_to_shreddingnet": bool(selected[index]),
                "full": {
                    "fused_probability": float(full["scores"][index]),
                    "decision_valid": bool(full["valid"][index]),
                    "translation_hat_rc": [
                        float(value) for value in full["translations"][index]
                    ],
                },
                "shreddingnet": shred,
            }
        )
    return tuple(records)


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


def run_selective_replay(config: SelectiveReplayConfig) -> Path:
    """Run and atomically publish one synthetic Hybrid-99 selective replay."""

    # Authenticate the single validation-frozen operating point before CUDA work.
    route_artifact = load_route_artifact(config.route_artifact)
    route_authority = route_artifact["route"]
    assert isinstance(route_authority, Mapping)
    route_threshold = float(route_authority["threshold"])
    if not torch.cuda.is_available():
        raise SelectiveReplayError("CUDA is unavailable")
    device = torch.device(config.device)
    try:
        device_index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.get_device_properties(device_index)
    except (AssertionError, RuntimeError) as error:
        raise SelectiveReplayError("requested CUDA device is unavailable") from error
    output_root = config.output_root
    if output_root.exists() or output_root.is_symlink():
        raise SelectiveReplayError("output_root already exists; refusing overwrite")
    for protected in (
        config.n512_run_directory,
        config.shreddingnet_freeze_path.parent,
        config.dataset_root,
        config.cached_synthetic_root,
        config.route_artifact,
    ):
        if _protected_overlap(output_root, protected):
            raise SelectiveReplayError("output_root overlaps an input authority")

    cold_started = time.perf_counter()
    authority_started = time.perf_counter()
    receipt, receipt_sha256, winners = sealed._freeze_completed_winners(
        config.n512_run_directory.resolve(strict=True)
    )
    sealed._require_formal_convergence(receipt, winners)
    authority_seconds = time.perf_counter() - authority_started
    full_winners = [winner for winner in winners if winner.arm == "full_n512"]
    if len(full_winners) != 1:
        raise SelectiveReplayError("exactly one frozen Full winner is required")
    full_winner = full_winners[0]
    checkpoint_authority = route_artifact["frozen_checkpoints_sha256"]
    assert isinstance(checkpoint_authority, Mapping)
    if checkpoint_authority.get("full_n512") != full_winner.checkpoint_sha256:
        raise SelectiveReplayError("route artifact is bound to another Full winner")
    train_config = receipt.get("config")
    if not isinstance(train_config, Mapping):
        raise SelectiveReplayError("N512 training config is missing")
    precision = train_config.get("precision")
    seed = train_config.get("seed")
    if precision not in {"fp32", "bf16"} or type(seed) is not int:  # noqa: E721
        raise SelectiveReplayError("N512 precision/seed is invalid")
    sealed._set_determinism(seed)
    dataset_root = sealed._resolve_evaluation_dataset_root(
        receipt, config.dataset_root
    )
    manifest_path, manifest_rows = sealed._test_manifest_rows(dataset_root)
    if len(manifest_rows) != EXPECTED_SYNTHETIC_PAIRS:
        raise SelectiveReplayError("synthetic population is not exactly 3,000")
    dataset = RachelPairDataset(dataset_root, "test")
    full_loader = sealed._test_loader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=seed,
    )
    cold_setup_seconds = time.perf_counter() - cold_started

    pipeline_started = time.perf_counter()
    full, full_timing = _run_full_stage(
        full_winner,
        full_loader,
        manifest_rows,
        device=device,
        precision=precision,
    )
    selected = route_full_scores(
        full["scores"], full["valid"], route_threshold
    )
    full_release = _release_full(full_winner, device)

    loader_started = time.perf_counter()
    selected_indices = [int(value) for value in np.flatnonzero(selected)]
    selected_pair_ids = [manifest_rows[index].pair_id for index in selected_indices]
    selected_loader: DataLoader = sealed._test_loader(
        Subset(dataset, selected_indices),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=seed,
    )
    route_loader_seconds = time.perf_counter() - loader_started

    shredding, shredding_timing, shredding_provenance = _run_shredding_stage(
        config.shreddingnet_freeze_path,
        selected_loader,
        selected_pair_ids,
        device=device,
    )
    for split in ("train", "val"):
        observed = _sha256_file(dataset_root / "pairs" / (split + ".jsonl"))
        if shredding_provenance["training_manifest_sha256"][split] != observed:
            raise SelectiveReplayError(
                "ShreddingNet {} manifest differs from N512 dataset".format(split)
            )
    expected_shredding_hashes = checkpoint_authority.get("shreddingnet_adapted")
    assert isinstance(expected_shredding_hashes, Mapping)
    if dict(shredding_provenance["checkpoint_sha256_by_stage"]) != dict(
        expected_shredding_hashes
    ):
        raise SelectiveReplayError(
            "route artifact is bound to another ShreddingNet winner"
        )

    merge_started = time.perf_counter()
    records = _make_records(
        manifest_rows, full, selected, shredding, route_threshold
    )
    summary = _hybrid_summary(records)
    merge_seconds = time.perf_counter() - merge_started
    pipeline_seconds = time.perf_counter() - pipeline_started

    cache_started = time.perf_counter()
    cached_full_path = config.cached_synthetic_root / "full_n512" / "pair_scores.jsonl"
    cached_shred_path = (
        config.cached_synthetic_root / "shreddingnet_adapted" / "pair_scores.jsonl"
    )
    cached_full = _read_jsonl_unique(cached_full_path, EXPECTED_SYNTHETIC_PAIRS)
    cached_shred = _read_jsonl_unique(cached_shred_path, EXPECTED_SYNTHETIC_PAIRS)
    cache_parity = validate_cached_replay(
        records,
        cached_full,
        cached_shred,
        score_atol=config.score_atol,
        pose_atol_px=config.pose_atol_px,
        relative_tolerance=config.relative_tolerance,
    )
    cache_seconds = time.perf_counter() - cache_started

    timing = {
        "boundary": (
            "CUDA-synchronized before/after each GPU stage; T_total begins before "
            "Full CPU-to-GPU transfer and ends after result merge, excluding cached-output "
            "parity and disk serialization"
        ),
        "full_authority_restore_seconds": authority_seconds,
        "cold_authority_and_dataset_setup_seconds": cold_setup_seconds,
        "full_seconds": full_timing["seconds"],
        "full_release_seconds": full_release["seconds"],
        "route_subset_loader_build_seconds": route_loader_seconds,
        "shredding_restore_seconds": shredding_timing["restore_seconds"],
        "shredding_inference_seconds": shredding_timing["inference_seconds"],
        "shredding_restore_plus_inference_seconds": shredding_timing[
            "restore_plus_inference_seconds"
        ],
        "shredding_release_seconds": shredding_timing["release_seconds"],
        "merge_seconds": merge_seconds,
        "pipeline_total_seconds": pipeline_seconds,
        "pipeline_input_pairs_per_second": EXPECTED_SYNTHETIC_PAIRS
        / pipeline_seconds,
        "cache_parity_seconds_excluded_from_pipeline": cache_seconds,
        "full_batch_count": full_timing["batch_count"],
        "shredding_batch_count": shredding_timing["batch_count"],
    }
    resource = {
        "cuda_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "full_stage": full_timing["gpu_memory"],
        "shredding_stage": shredding_timing["gpu_memory"],
        "sequential_peak_allocated_bytes": max(
            full_timing["gpu_memory"]["peak_allocated_bytes"],
            shredding_timing["gpu_memory"]["peak_allocated_bytes"],
        ),
        "sequential_peak_reserved_bytes": max(
            full_timing["gpu_memory"]["peak_reserved_bytes"],
            shredding_timing["gpu_memory"]["peak_reserved_bytes"],
        ),
        "full_release": full_release,
        "shredding_release": {
            "allocated_bytes_after_release": shredding_timing[
                "allocated_bytes_after_release"
            ],
            "reserved_bytes_after_release": shredding_timing[
                "reserved_bytes_after_release"
            ],
        },
        "sequential_not_concurrent": True,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_verified_synthetic_selective_replay",
        "experiment": "Hybrid-99 Full fused screening then selective ShreddingNet pose",
        "population": "Rachel synthetic test 3000, original order",
        "route": {
            "score": "full_n512.fused_probability",
            "operator": "greater_than_or_equal",
            "threshold": route_threshold,
            "invalid_policy": "fail_open_route_to_shreddingnet",
            "nonfinite_policy": "fail_open_route_to_shreddingnet",
            "shreddingnet_pair_score_used_for_final_classification": False,
        },
        "summary": summary,
        "timing": timing,
        "resource": resource,
        "cache_parity": cache_parity,
        "provenance": {
            "n512_run_receipt": str(
                config.n512_run_directory / "run_receipt.json"
            ),
            "n512_run_receipt_sha256": receipt_sha256,
            "full_winner_epoch": full_winner.epoch,
            "full_winner_checkpoint": str(full_winner.checkpoint_path),
            "full_winner_checkpoint_sha256": full_winner.checkpoint_sha256,
            "shreddingnet": shredding_provenance,
            "dataset_root": str(dataset_root),
            "test_manifest": str(manifest_path),
            "test_manifest_sha256": _sha256_file(manifest_path),
            "cached_full_pair_scores": str(cached_full_path),
            "cached_full_pair_scores_sha256": _sha256_file(cached_full_path),
            "cached_shreddingnet_pair_scores": str(cached_shred_path),
            "cached_shreddingnet_pair_scores_sha256": _sha256_file(
                cached_shred_path
            ),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "route_artifact": str(config.route_artifact),
            "route_artifact_file_sha256": _sha256_file(config.route_artifact),
            "route_artifact_content_sha256": route_artifact["content_sha256"],
            "training_performed": False,
            "threshold_fit_performed": False,
            "exact6_outputs_modified": False,
        },
    }
    result["content_sha256"] = _canonical_sha256(result)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=".partial-hybrid99-", dir=str(output_root.parent))
    )
    try:
        pair_path = staged / "pair_results.jsonl"
        sealed._atomic_jsonl(pair_path, records)
        summary_path = staged / "summary.json"
        sealed._atomic_json(summary_path, result)
        run_receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_verified_synthetic_selective_replay",
            "summary": "summary.json",
            "summary_sha256": _sha256_file(summary_path),
            "pair_results": "pair_results.jsonl",
            "pair_results_sha256": _sha256_file(pair_path),
            "pair_results_count": len(records),
            "cache_parity_status": cache_parity["status"],
        }
        run_receipt["content_sha256"] = _canonical_sha256(run_receipt)
        sealed._atomic_json(staged / "run_receipt.json", run_receipt)
        sealed._publish_directory_no_replace(
            staged, output_root, completion_receipt="run_receipt.json"
        )
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--n512-run-directory", type=Path, default=DEFAULT_N512_RUN)
    parser.add_argument(
        "--shreddingnet-freeze-path",
        type=Path,
        default=DEFAULT_SHREDDINGNET_FREEZE,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--cached-synthetic-root", type=Path, default=DEFAULT_CACHED_SYNTHETIC_ROOT
    )
    parser.add_argument("--route-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-atol", type=float, default=2e-5)
    parser.add_argument("--pose-atol-px", type=float, default=2e-3)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    output = run_selective_replay(
        SelectiveReplayConfig(
            n512_run_directory=arguments.n512_run_directory,
            shreddingnet_freeze_path=arguments.shreddingnet_freeze_path,
            dataset_root=arguments.dataset_root,
            cached_synthetic_root=arguments.cached_synthetic_root,
            route_artifact=arguments.route_artifact,
            output_root=arguments.output_root,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
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


__all__ = [
    "DEFAULT_ROUTE_THRESHOLD",
    "SelectiveReplayConfig",
    "SelectiveReplayError",
    "main",
    "route_full_scores",
    "run_selective_replay",
    "validate_cached_replay",
]

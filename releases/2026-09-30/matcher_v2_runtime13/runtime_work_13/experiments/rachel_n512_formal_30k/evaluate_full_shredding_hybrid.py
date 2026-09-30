#!/usr/bin/env python3
"""Evaluate frozen Hybrid-99 from exact-six cached predictions.

The route threshold must come from the separately frozen validation-only
artifact.  This evaluator never fits or overrides a threshold and never
modifies the exact-six evidence it reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise


SCHEMA_VERSION = "rachel-full-shredding-hybrid99-cached-evaluation/1.0"
ROUTE_SCHEMA_VERSION = "rachel-full-shredding-hybrid-route/1.0"
METHOD_ID = "full_n512_screen__shreddingnet_adapted_pose_hybrid99"
FULL_CHECKPOINT_SHA256 = (
    "c8ffd9b53b359e86a9be6695127238ce94afe5b3af003a1b8908155a773a68e1"
)
SHREDDING_CHECKPOINT_SHA256 = {
    "coarse": "e8e114c8362f400ea69cf3a3ecb59cf1267c10aa73a6488672a9b089d6e2998f",
    "matching": "2ddbf7dfecda4f65bc4b45d4df2dfd1fdd18813fc9982f308f28ff01b45fc1b1",
    "classify": "e155f76fd2102c92f52a876edbb87d698f3b265d55234dad4fa08990d17340f3",
}
CONDITIONS = (
    "clean",
    "erosion_r2",
    "erosion_r4",
    "erosion_r8",
    "local_bites_k1_r8",
    "local_bites_k2_r8",
    "local_bites_k4_r8",
)
TOLERANCES = (2, 5, 8, 10)
ROUTE_THRESHOLD = 0.011353014037013054
FULL_CLASSIFICATION_THRESHOLD = 0.9967334270477295
SHREDDING_CLASSIFICATION_THRESHOLD = 0.666015625
# These are the completed exact-six, end-to0.0 synthetic end-to-end timings.
# The hybrid estimate is deliberately linear and is not a replay measurement.
FULL_SECONDS_PER_PAIR = 50.68685146421194 / 3000.0
SHREDDING_SECONDS_PER_PAIR = 1478.7883032094687 / 3000.0


class HybridEvaluationError(RuntimeError):
    """A frozen input, population, or metric invariant was violated."""


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


def _read_json(path: Path, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HybridEvaluationError(description + " is not readable strict JSON") from error
    if not isinstance(value, Mapping):
        raise HybridEvaluationError(description + " root is not an object")
    return value


def _read_jsonl(path: Path, description: str) -> Tuple[Mapping[str, object], ...]:
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
                    raise HybridEvaluationError(
                        "{} has invalid/duplicate pair_id at line {}".format(
                            description, line_number
                        )
                    )
                seen.add(pair_id)
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise HybridEvaluationError(description + " is not readable JSONL") from error
    if not rows:
        raise HybridEvaluationError(description + " is empty")
    return tuple(rows)


def _score(value: object) -> float:
    """Return a numeric score or NaN for the explicit invalid representation."""

    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HybridEvaluationError("probability is neither numeric nor null")
    return float(value)


def _explicit_bool(value: object, description: str) -> bool:
    if type(value) is not bool:  # noqa: E721 - reject 0/1 and truthy strings
        raise HybridEvaluationError(description + " must be an explicit bool")
    return value


def _route_artifact(path: Path) -> Tuple[Mapping[str, object], float, float]:
    value = dict(_read_json(path, "route artifact"))
    declared = value.pop("content_sha256", None)
    if not isinstance(declared, str) or _canonical_sha256(value) != declared:
        raise HybridEvaluationError("route artifact content SHA-256 differs")
    value["content_sha256"] = declared
    route = value.get("route")
    classification = value.get("classification")
    checkpoints = value.get("frozen_checkpoints_sha256")
    protocol = value.get("protocol")
    if (
        value.get("schema_version") != ROUTE_SCHEMA_VERSION
        or value.get("status") != "frozen_validation_only_before_hybrid_evaluation"
        or value.get("method_id") != METHOD_ID
        or not isinstance(route, Mapping)
        or route.get("threshold_fit_source") != "validation_only"
        or route.get("test_or_real_parameter_fit") is not False
        or route.get("rule")
        != "fail_open_if_invalid_else_probability_greater_than_or_equal_to_threshold"
        or not isinstance(classification, Mapping)
        or classification.get("shreddingnet_score_used") is not False
        or not isinstance(checkpoints, Mapping)
        or checkpoints.get("full_n512") != FULL_CHECKPOINT_SHA256
        or checkpoints.get("shreddingnet_adapted") != SHREDDING_CHECKPOINT_SHA256
        or not isinstance(protocol, Mapping)
        or protocol.get("test_or_real_read_by_this_command") is not False
        or protocol.get("threshold_sweep_on_test_or_real_forbidden") is not True
    ):
        raise HybridEvaluationError("route artifact identity/protocol differs")
    threshold = route.get("threshold")
    class_threshold = classification.get("threshold")
    for name, item in (("route", threshold), ("classification", class_threshold)):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 1.0
        ):
            raise HybridEvaluationError(name + " threshold is invalid")
    if not float(threshold) < float(class_threshold):
        raise HybridEvaluationError("route threshold must include every Full decision")
    if float(threshold) != ROUTE_THRESHOLD:
        raise HybridEvaluationError("route threshold differs from frozen Hybrid-99")
    if float(class_threshold) != FULL_CLASSIFICATION_THRESHOLD:
        raise HybridEvaluationError("Full classification threshold differs")
    return value, float(threshold), float(class_threshold)


def _route(scores: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    return (~valid) | (~np.isfinite(scores)) | (scores >= threshold)


def _weights(clusters: Sequence[str]) -> np.ndarray:
    values = np.asarray(clusters, dtype=object)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * counts[inverse].astype(np.float64))


def _route_metrics(
    selected: np.ndarray, labels: np.ndarray, clusters: Sequence[str]
) -> Mapping[str, object]:
    selected = np.asarray(selected, dtype=np.bool_)
    labels = np.asarray(labels, dtype=np.bool_)
    if selected.shape != labels.shape or selected.ndim != 1:
        raise HybridEvaluationError("route/label vectors differ")
    if not np.any(labels) or np.all(labels):
        raise HybridEvaluationError("routing metrics require both classes")
    weight = _weights(clusters)
    positive_weight = float(weight[labels].sum())
    negative_weight = float(weight[~labels].sum())
    selected_count = int(selected.sum())
    selected_positive = int((selected & labels).sum())
    selected_negative = int((selected & ~labels).sum())
    return {
        "sample_count": int(len(labels)),
        "selected_count": selected_count,
        "selected_fraction": float(selected.mean()),
        "positive_selected_count": selected_positive,
        "row_positive_route_recall": float(selected[labels].mean()),
        "cluster_balanced_positive_route_recall": float(
            weight[selected & labels].sum() / positive_weight
        ),
        "negative_selected_count": selected_negative,
        "row_negative_rejection": float((~selected[~labels]).mean()),
        "cluster_balanced_negative_rejection": float(
            weight[(~selected) & (~labels)].sum() / negative_weight
        ),
        "row_selected_precision": selected_positive / selected_count,
    }


def _pose(errors: np.ndarray, labels: np.ndarray) -> Mapping[str, object]:
    errors = np.asarray(errors, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)
    if errors.shape != labels.shape:
        raise HybridEvaluationError("pose error/label vectors differ")
    valid = labels & np.isfinite(errors)
    positive_count = int(labels.sum())
    values = errors[valid]
    return {
        "positive_count": positive_count,
        "valid_pose_count": int(valid.sum()),
        "valid_pose_fraction_unconditional": float(valid.sum() / positive_count),
        "median_te_px_conditional_on_valid": (
            float(np.median(values)) if len(values) else None
        ),
        "p90_te_px_conditional_on_valid": (
            float(np.quantile(values, 0.9)) if len(values) else None
        ),
        "unconditional_recall": {
            "at_{}px".format(tolerance): float(
                np.count_nonzero(valid & (errors <= tolerance)) / positive_count
            )
            for tolerance in TOLERANCES
        },
    }


def _linear_runtime_estimate(sample_count: int, selected_count: int) -> Mapping[str, object]:
    full_seconds = sample_count * FULL_SECONDS_PER_PAIR
    shredding_seconds = selected_count * SHREDDING_SECONDS_PER_PAIR
    hybrid_seconds = full_seconds + shredding_seconds
    all_pair_cascade_seconds = sample_count * (
        FULL_SECONDS_PER_PAIR + SHREDDING_SECONDS_PER_PAIR
    )
    return {
        "estimate_not_measured_replay": True,
        "source": "completed_exact_six_synthetic_3000_end_to_end_wallclock",
        "source_full_seconds_for_3000": 50.68685146421194,
        "source_shredding_seconds_for_3000": 1478.7883032094687,
        "full_all_pairs_seconds": full_seconds,
        "shredding_selected_pairs_seconds": shredding_seconds,
        "hybrid_total_seconds": hybrid_seconds,
        "full_plus_shredding_all_pairs_seconds": all_pair_cascade_seconds,
        "speedup_vs_full_plus_shredding_all_pairs": (
            all_pair_cascade_seconds / hybrid_seconds if hybrid_seconds else None
        ),
        "limitations": (
            "linear estimate from cached synthetic end-to-end timings; excludes "
            "selective batching, launch, I/O, and device-utilization effects"
        ),
    }


def _assembly(
    decisions: np.ndarray, errors: np.ndarray, labels: np.ndarray
) -> Mapping[str, object]:
    decisions = np.asarray(decisions, dtype=np.bool_)
    errors = np.asarray(errors, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)
    if decisions.shape != labels.shape or errors.shape != labels.shape:
        raise HybridEvaluationError("assembly vectors differ")
    predicted = int(decisions.sum())
    target = int(labels.sum())
    result = {}
    for tolerance in TOLERANCES:
        true_positive = int(
            np.count_nonzero(decisions & labels & np.isfinite(errors) & (errors <= tolerance))
        )
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / target if target else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result["at_{}px".format(tolerance)] = {
            "true_positive_count": true_positive,
            "predicted_count": predicted,
            "target_count": target,
            "false_positive_count": predicted - true_positive,
            "false_negative_count": target - true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


def _method_vectors_synthetic(
    rows: Sequence[Mapping[str, object]], method: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    labels = np.asarray(
        [_explicit_bool(row.get("label"), "synthetic label") for row in rows],
        dtype=np.bool_,
    )
    clusters = [str(row["cluster_id"]) for row in rows]
    pair_ids = [str(row["pair_id"]) for row in rows]
    if method == "full_n512":
        scores = np.asarray(
            [_score(row["scores"]["fused"].get("probability")) for row in rows]
        )
        valid = np.asarray(
            [
                _explicit_bool(
                    row["scores"]["fused"].get("valid"),
                    "synthetic Full score validity",
                )
                for row in rows
            ],
            dtype=np.bool_,
        )
        errors = np.asarray(
            [
                float(row["geometry"]["translation_l2_px"])
                if row["geometry"]["translation_l2_px"] is not None
                else math.nan
                for row in rows
            ]
        )
    else:
        scores = np.asarray(
            [
                _score(row["scores"]["pair_probability"].get("probability"))
                for row in rows
            ]
        )
        valid = np.asarray(
            [
                _explicit_bool(
                    row["scores"]["pair_probability"].get("valid"),
                    "synthetic Shredding score validity",
                )
                for row in rows
            ],
            dtype=np.bool_,
        )
        errors = np.asarray(
            [
                float(row["geometry"]["translation_l2_px"])
                if row["geometry"]["translation_l2_px"] is not None
                and bool(row["geometry"]["translation_prediction_valid"])
                else math.nan
                for row in rows
            ]
        )
    return scores, valid, errors, labels, clusters, pair_ids


def _population_result(
    *,
    full_scores: np.ndarray,
    full_valid: np.ndarray,
    full_errors: np.ndarray,
    shred_scores: np.ndarray,
    shred_valid: np.ndarray,
    shred_errors: np.ndarray,
    labels: np.ndarray,
    clusters: Sequence[str],
    route_threshold: float,
    class_threshold: float,
    shred_threshold: float,
) -> Mapping[str, object]:
    selected = _route(full_scores, full_valid, route_threshold)
    full_decision = full_valid & np.isfinite(full_scores) & (full_scores >= class_threshold)
    shred_decision = shred_valid & np.isfinite(shred_scores) & (shred_scores >= shred_threshold)
    if np.any(full_decision & ~selected):
        raise HybridEvaluationError("Full-positive decision was not routed")
    hybrid_errors = np.where(selected, shred_errors, math.nan)
    full_classification = evaluate_pairwise(
        full_scores, labels, full_valid, clusters, threshold=class_threshold
    )
    return {
        "routing": _route_metrics(selected, labels, clusters),
        "classification": {
            "semantics": "identical_to_full_n512_by_design",
            "hybrid": full_classification,
            "full_n512": full_classification,
            "shreddingnet_adapted": evaluate_pairwise(
                shred_scores, labels, shred_valid, clusters, threshold=shred_threshold
            ),
            "invariant_hybrid_equals_full": True,
        },
        "pose": {
            "hybrid99": _pose(hybrid_errors, labels),
            "full_n512": _pose(full_errors, labels),
            "shreddingnet_adapted_all_pairs": _pose(shred_errors, labels),
        },
        "joint_full_decision_and_shredding_pose_unconditional_recall": {
            "at_{}px".format(tolerance): float(
                np.count_nonzero(
                    full_decision
                    & labels
                    & np.isfinite(hybrid_errors)
                    & (hybrid_errors <= tolerance)
                )
                / np.count_nonzero(labels)
            )
            for tolerance in TOLERANCES
        },
        "assembly_edge": {
            "hybrid99_full_decision_shredding_pose": _assembly(
                full_decision, hybrid_errors, labels
            ),
            "full_n512": _assembly(full_decision, full_errors, labels),
            "shreddingnet_adapted_all_pairs": _assembly(
                shred_decision, shred_errors, labels
            ),
        },
        "runtime_linear_estimate": _linear_runtime_estimate(
            len(labels), int(selected.sum())
        ),
    }


def _synthetic(
    root: Path, route_threshold: float, class_threshold: float
) -> Mapping[str, object]:
    full_path = root / "full_n512" / "pair_scores.jsonl"
    shred_path = root / "shreddingnet_adapted" / "pair_scores.jsonl"
    full_rows = _read_jsonl(full_path, "synthetic Full pair scores")
    shred_rows = _read_jsonl(shred_path, "synthetic ShreddingNet pair scores")
    if len(full_rows) != 3000 or len(shred_rows) != 3000:
        raise HybridEvaluationError("synthetic population must contain 3000 pairs")
    fv = _method_vectors_synthetic(full_rows, "full_n512")
    sv = _method_vectors_synthetic(shred_rows, "shreddingnet_adapted")
    if fv[3:].__class__ is not tuple:  # pragma: no cover - defensive type anchor
        raise AssertionError
    if fv[5] != sv[5] or fv[4] != sv[4] or not np.array_equal(fv[3], sv[3]):
        raise HybridEvaluationError("synthetic Full/Shredding pair alignment differs")
    if int(fv[3].sum()) != 1500:
        raise HybridEvaluationError("synthetic class balance differs")
    result = _population_result(
        full_scores=fv[0],
        full_valid=fv[1],
        full_errors=fv[2],
        shred_scores=sv[0],
        shred_valid=sv[1],
        shred_errors=sv[2],
        labels=fv[3],
        clusters=fv[4],
        route_threshold=route_threshold,
        class_threshold=class_threshold,
        shred_threshold=0.666015625,
    )
    return {
        "population": "synthetic_test_3000",
        "result": result,
        "sources": {
            "full": {"path": str(full_path), "sha256": _sha256_file(full_path)},
            "shreddingnet": {
                "path": str(shred_path),
                "sha256": _sha256_file(shred_path),
            },
        },
    }


def _manifest_identity(row: Mapping[str, object]) -> Tuple[bool, str, Tuple[str, ...]]:
    label = _explicit_bool(row.get("label"), "test manifest label")
    first = row.get("fragment_a")
    second = row.get("fragment_b")
    first_unit = first.get("split_unit_id") if isinstance(first, Mapping) else None
    second_unit = second.get("split_unit_id") if isinstance(second, Mapping) else None
    if not isinstance(first_unit, str) or not first_unit:
        raise HybridEvaluationError("test manifest fragment_a split unit is invalid")
    if not isinstance(second_unit, str) or not second_unit:
        raise HybridEvaluationError("test manifest fragment_b split unit is invalid")
    ordered = sorted((first_unit, second_unit))
    cluster = (
        "unit:" + ordered[0]
        if ordered[0] == ordered[1]
        else "unit-pair:" + _canonical_sha256(ordered)
    )
    return label, cluster, tuple(sorted(set(ordered)))


def _positive_target(dataset_root: Path, value: object) -> Tuple[float, float]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise HybridEvaluationError("positive target path is invalid")
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.parts[:2] != ("targets", "pairs")
        or logical.suffix.casefold() != ".npz"
    ):
        raise HybridEvaluationError("positive target is not a release targets/pairs NPZ")
    current = dataset_root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise HybridEvaluationError("symlinked positive target is forbidden")
    try:
        target_path = current.resolve(strict=True)
        target_path.relative_to(dataset_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise HybridEvaluationError("positive target is missing or escapes release") from error
    try:
        with np.load(target_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "correspondence_indices",
                "translation_a_to_b_rc",
                "translation_a_to_b_xy_cartesian",
            }:
                raise HybridEvaluationError("positive target archive fields differ")
            rc = np.asarray(archive["translation_a_to_b_rc"], dtype=np.float64)
            xy = np.asarray(
                archive["translation_a_to_b_xy_cartesian"], dtype=np.float64
            )
    except (OSError, ValueError) as error:
        raise HybridEvaluationError("positive target archive is unreadable") from error
    if (
        rc.shape != (2,)
        or xy.shape != (2,)
        or not np.all(np.isfinite(rc))
        or not np.all(np.isfinite(xy))
        or not np.allclose(xy, (rc[1], -rc[0]), rtol=0.0, atol=1e-4)
    ):
        raise HybridEvaluationError("positive translation target is invalid")
    return float(rc[0]), float(rc[1])


def _test_manifest(path: Path) -> Mapping[str, Mapping[str, object]]:
    rows = _read_jsonl(path, "synthetic test manifest")
    if path.name != "test.jsonl" or path.parent.name != "pairs":
        raise HybridEvaluationError("test manifest must be pairs/test.jsonl")
    dataset_root = path.parent.parent.resolve(strict=True)
    output: Dict[str, Mapping[str, object]] = {}
    for row in rows:
        if row.get("split") != "test":
            raise HybridEvaluationError("test manifest contains a non-test row")
        label, cluster, source_units = _manifest_identity(row)
        target_value = row.get("correspondence_path")
        if not label and target_value is not None:
            raise HybridEvaluationError("negative test pair unexpectedly has a target")
        target = _positive_target(dataset_root, target_value) if label else None
        enriched = dict(row)
        enriched["_cluster_id"] = cluster
        enriched["_source_unit_ids"] = source_units
        enriched["_translation_a_to_b_rc"] = target
        output[str(row["pair_id"])] = enriched
    return output


def _translation_error(prediction: object, target: object, valid: bool) -> float:
    if not valid or prediction is None or target is None:
        return math.nan
    first = np.asarray(prediction, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    if first.shape != (2,) or second.shape != (2,) or not np.all(np.isfinite(first)):
        return math.nan
    return float(np.linalg.norm(first - second))


def _corrosion(
    root: Path,
    test_manifest_path: Path,
    route_threshold: float,
    class_threshold: float,
) -> Mapping[str, object]:
    manifest = _test_manifest(test_manifest_path)
    by_condition = {}
    sources = {}
    for condition in CONDITIONS:
        path = root / "conditions" / condition / "pair_scores.jsonl"
        all_rows = _read_jsonl(path, "corrosion " + condition)
        rows = [
            row
            for row in all_rows
            if bool(row.get("fixed_all_method_all_condition_common_valid"))
        ]
        if len(rows) != 1822 or sum(bool(row["label"]) for row in rows) != 923:
            raise HybridEvaluationError(condition + " fixed common-valid population differs")
        pair_ids = [str(row["pair_id"]) for row in rows]
        labels = np.asarray([bool(row["label"]) for row in rows], dtype=np.bool_)
        clusters = [str(row["cluster_id"]) for row in rows]
        if any(pair_id not in manifest for pair_id in pair_ids):
            raise HybridEvaluationError(condition + " pair is missing from test manifest")
        full = [row["methods"]["full_n512"] for row in rows]
        shred = [row["methods"]["shreddingnet_adapted"] for row in rows]
        full_scores = np.asarray([float(row["probability"]) for row in full])
        full_valid = np.asarray([bool(row["valid"]) for row in full], dtype=np.bool_)
        shred_scores = np.asarray([float(row["probability"]) for row in shred])
        shred_valid = np.asarray([bool(row["valid"]) for row in shred], dtype=np.bool_)
        full_errors = np.asarray(
            [
                _translation_error(
                    row["geometry"]["translation_hat_rc"],
                    manifest[pair_id].get("_translation_a_to_b_rc"),
                    bool(row["geometry"]["translation_valid"]),
                )
                for row, pair_id in zip(full, pair_ids)
            ]
        )
        shred_errors = np.asarray(
            [
                _translation_error(
                    row["geometry"]["translation_hat_rc"],
                    manifest[pair_id].get("_translation_a_to_b_rc"),
                    bool(row["geometry"]["translation_valid"]),
                )
                for row, pair_id in zip(shred, pair_ids)
            ]
        )
        by_condition[condition] = _population_result(
            full_scores=full_scores,
            full_valid=full_valid,
            full_errors=full_errors,
            shred_scores=shred_scores,
            shred_valid=shred_valid,
            shred_errors=shred_errors,
            labels=labels,
            clusters=clusters,
            route_threshold=route_threshold,
            class_threshold=class_threshold,
            shred_threshold=0.666015625,
        )
        sources[condition] = {"path": str(path), "sha256": _sha256_file(path)}
    return {
        "population": "fixed_all_method_all_condition_common_valid_1822",
        "conditions": by_condition,
        "sources": sources,
        "test_manifest": {
            "path": str(test_manifest_path),
            "sha256": _sha256_file(test_manifest_path),
        },
    }


def _real_method_arrays(
    rows: Sequence[Mapping[str, object]], method: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = [row["methods"][method] for row in rows]
    return (
        np.asarray([float(row["probability"]) for row in values]),
        np.asarray([bool(row["valid"]) for row in values], dtype=np.bool_),
        np.asarray(
            [bool(row["decision_at_frozen_validation_threshold"]) for row in values],
            dtype=np.bool_,
        ),
    )


def _real(
    pair_path: Path,
    translation_path: Path,
    route_threshold: float,
    class_threshold: float,
) -> Mapping[str, object]:
    pair_document = _read_json(pair_path, "real pair-only evaluation")
    translation_document = _read_json(translation_path, "real translation evaluation")
    balanced = pair_document.get("balanced_1016")
    strict = pair_document.get("strict_547")
    positive_rows = translation_document.get("positive_pairs")
    if (
        not isinstance(balanced, Mapping)
        or not isinstance(strict, Mapping)
        or not isinstance(positive_rows, list)
    ):
        raise HybridEvaluationError("real population structure differs")
    balanced_rows = balanced.get("pairs")
    strict_rows = strict.get("pairs")
    if (
        not isinstance(balanced_rows, list)
        or len(balanced_rows) != 1016
        or not isinstance(strict_rows, list)
        or len(strict_rows) != 547
        or len(positive_rows) != 508
    ):
        raise HybridEvaluationError("real population counts differ")
    positive_by_id = {str(row["pair_id"]): row for row in positive_rows}
    if len(positive_by_id) != 508:
        raise HybridEvaluationError("real positive pair IDs are duplicated")
    labels = np.asarray([bool(row["label"]) for row in balanced_rows], dtype=np.bool_)
    clusters = [str(row["case_cluster"]) for row in balanced_rows]
    pair_ids = [str(row["pair_id"]) for row in balanced_rows]
    full_scores, full_valid, full_decision_cached = _real_method_arrays(
        balanced_rows, "full_n512"
    )
    shred_scores, shred_valid, shred_decision = _real_method_arrays(
        balanced_rows, "shreddingnet_adapted"
    )
    full_decision = full_valid & np.isfinite(full_scores) & (full_scores >= class_threshold)
    if not np.array_equal(full_decision, full_decision_cached):
        raise HybridEvaluationError("real Full cached decisions disagree with threshold")
    full_errors = np.full(len(balanced_rows), math.nan, dtype=np.float64)
    shred_errors = np.full(len(balanced_rows), math.nan, dtype=np.float64)
    for index, (pair_id, label) in enumerate(zip(pair_ids, labels)):
        if not label:
            continue
        row = positive_by_id.get(pair_id)
        if row is None:
            raise HybridEvaluationError("real positive pair is missing translation GT")
        full = row["full_n512"]
        shred = row["shreddingnet_adapted"]
        if float(full["probability_frozen"]) != full_scores[index]:
            raise HybridEvaluationError("real Full probability differs across artifacts")
        if float(shred["probability_frozen"]) != shred_scores[index]:
            raise HybridEvaluationError("real Shredding probability differs across artifacts")
        if bool(full["translation_prediction_valid"]):
            full_errors[index] = float(full["translation_l2_error_px"])
        if bool(shred["translation_prediction_valid"]):
            shred_errors[index] = float(shred["translation_l2_error_px"])
    primary = _population_result(
        full_scores=full_scores,
        full_valid=full_valid,
        full_errors=full_errors,
        shred_scores=shred_scores,
        shred_valid=shred_valid,
        shred_errors=shred_errors,
        labels=labels,
        clusters=clusters,
        route_threshold=route_threshold,
        class_threshold=class_threshold,
        shred_threshold=0.666015625,
    )

    def assembly_view(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
        ids = [str(row["pair_id"]) for row in rows]
        row_labels = np.asarray([bool(row["label"]) for row in rows], dtype=np.bool_)
        fs, fv, f_cached = _real_method_arrays(rows, "full_n512")
        ss, sv, s_decision = _real_method_arrays(rows, "shreddingnet_adapted")
        f_decision = fv & np.isfinite(fs) & (fs >= class_threshold)
        if not np.array_equal(f_decision, f_cached):
            raise HybridEvaluationError("real assembly Full decision differs")
        selected = _route(fs, fv, route_threshold)
        if np.any(f_decision & ~selected):
            raise HybridEvaluationError("real Full-positive assembly edge was not routed")
        f_error = np.full(len(rows), math.nan)
        s_error = np.full(len(rows), math.nan)
        for index, (pair_id, label) in enumerate(zip(ids, row_labels)):
            if not label:
                continue
            positive = positive_by_id[pair_id]
            if bool(positive["full_n512"]["translation_prediction_valid"]):
                f_error[index] = float(positive["full_n512"]["translation_l2_error_px"])
            if bool(positive["shreddingnet_adapted"]["translation_prediction_valid"]):
                s_error[index] = float(
                    positive["shreddingnet_adapted"]["translation_l2_error_px"]
                )
        return {
            "hybrid99_full_decision_shredding_pose": _assembly(
                f_decision, np.where(selected, s_error, math.nan), row_labels
            ),
            "full_n512": _assembly(f_decision, f_error, row_labels),
            "shreddingnet_adapted_all_pairs": _assembly(
                s_decision, s_error, row_labels
            ),
        }

    return {
        "balanced_1016": primary,
        "assembly_views": {
            "balanced_1016_selected_list_diagnostic": assembly_view(balanced_rows),
            "strict_547_descriptive": assembly_view(strict_rows),
        },
        "sources": {
            "pair_only": {"path": str(pair_path), "sha256": _sha256_file(pair_path)},
            "translation_gt": {
                "path": str(translation_path),
                "sha256": _sha256_file(translation_path),
            },
        },
    }


def evaluate(
    *,
    route_artifact_path: Path,
    synthetic_root: Path,
    corrosion_root: Path,
    test_manifest_path: Path,
    real_pair_path: Path,
    real_translation_path: Path,
) -> Mapping[str, object]:
    route_artifact, route_threshold, class_threshold = _route_artifact(
        route_artifact_path
    )
    result: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_cached_exact_six_hybrid99_evaluation",
        "method_id": METHOD_ID,
        "protocol": {
            "route_threshold_fit_source": "validation_only",
            "test_or_real_parameter_fit": False,
            "single_primary_operating_point": True,
            "classification_score_and_decision": "unchanged_full_n512",
            "pose_source": "routed_shreddingnet_adapted",
            "cached_outputs_reused_for_quality": True,
            "training_performed": False,
            "model_forward_performed": False,
        },
        "route_artifact": {
            "path": str(route_artifact_path),
            "file_sha256": _sha256_file(route_artifact_path),
            "content_sha256": route_artifact["content_sha256"],
            "threshold": route_threshold,
            "classification_threshold": class_threshold,
        },
        "synthetic": _synthetic(synthetic_root, route_threshold, class_threshold),
        "corrosion": _corrosion(
            corrosion_root, test_manifest_path, route_threshold, class_threshold
        ),
        "real": _real(
            real_pair_path, real_translation_path, route_threshold, class_threshold
        ),
    }
    result["content_sha256"] = _canonical_sha256(result)
    return result


def _exclusive_json(path: Path, value: object) -> None:
    path = path.resolve()
    if path.exists():
        raise HybridEvaluationError("refusing to overwrite hybrid result")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise HybridEvaluationError("refusing to overwrite hybrid result") from error
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--route-artifact", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--corrosion-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--real-pair", type=Path, required=True)
    parser.add_argument("--real-translation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = evaluate(
        route_artifact_path=args.route_artifact.resolve(strict=True),
        synthetic_root=args.synthetic_root.resolve(strict=True),
        corrosion_root=args.corrosion_root.resolve(strict=True),
        test_manifest_path=args.test_manifest.resolve(strict=True),
        real_pair_path=args.real_pair.resolve(strict=True),
        real_translation_path=args.real_translation.resolve(strict=True),
    )
    _exclusive_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "content_sha256": result["content_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

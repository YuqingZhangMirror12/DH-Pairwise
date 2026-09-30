#!/usr/bin/env python3
"""Freeze the validation-only router for the Full -> ShreddingNet cascade.

This command deliberately accepts no test, corrosion, or real-data input.  It
selects the highest observed Full validation score whose inclusive gate keeps
at least 99% of positive rows and 99% of equal-cluster positive weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "rachel-full-shredding-hybrid-route/1.0"
EXPECTED_VALIDATION_SCORES_SHA256 = (
    "574f127762e59f81f6f53c400a398a04b1486fb9e94c20f51da4dea0d19176d9"
)
FULL_CHECKPOINT_SHA256 = (
    "c8ffd9b53b359e86a9be6695127238ce94afe5b3af003a1b8908155a773a68e1"
)
SHREDDING_CHECKPOINT_SHA256 = {
    "coarse": "e8e114c8362f400ea69cf3a3ecb59cf1267c10aa73a6488672a9b089d6e2998f",
    "matching": "2ddbf7dfecda4f65bc4b45d4df2dfd1fdd18813fc9982f308f28ff01b45fc1b1",
    "classify": "e155f76fd2102c92f52a876edbb87d698f3b265d55234dad4fa08990d17340f3",
}
FULL_CLASSIFICATION_THRESHOLD = 0.9967334270477295
TARGET_RECALL = 0.99
EXPECTED_ROUTE_THRESHOLD = 0.011353014037013054


class HybridRouteFreezeError(RuntimeError):
    """The validation-only route freeze contract was violated."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_validation(path: Path) -> Tuple[List[str], List[str], List[bool], List[float], List[bool]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HybridRouteFreezeError("validation scores are not readable strict JSON") from error
    if not isinstance(value, Mapping) or set(value) != {
        "clusters",
        "labels",
        "pair_ids",
        "probability",
        "valid",
    }:
        raise HybridRouteFreezeError("validation score schema differs")
    pair_ids = value["pair_ids"]
    clusters = value["clusters"]
    labels = value["labels"]
    scores = value["probability"]
    valid = value["valid"]
    if not all(isinstance(row, list) for row in (pair_ids, clusters, labels, scores, valid)):
        raise HybridRouteFreezeError("validation score columns must be arrays")
    count = len(pair_ids)
    if count != 3000 or any(len(row) != count for row in (clusters, labels, scores, valid)):
        raise HybridRouteFreezeError("validation population must contain 3000 aligned rows")
    if len(set(pair_ids)) != count or any(not isinstance(item, str) or not item for item in pair_ids):
        raise HybridRouteFreezeError("validation pair IDs are empty or duplicated")
    if any(not isinstance(item, str) or not item for item in clusters):
        raise HybridRouteFreezeError("validation cluster IDs are invalid")
    if any(type(item) is not bool for item in labels + valid):  # noqa: E721
        raise HybridRouteFreezeError("validation labels/valid flags are not booleans")
    if sum(labels) != 1500:
        raise HybridRouteFreezeError("validation class balance differs")
    numeric_scores: List[float] = []
    for item, is_valid in zip(scores, valid):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise HybridRouteFreezeError("validation score is not numeric")
        score = float(item)
        if is_valid and (not math.isfinite(score) or not 0.0 <= score <= 1.0):
            raise HybridRouteFreezeError("valid validation score is outside [0,1]")
        numeric_scores.append(score)
    return list(pair_ids), list(clusters), list(labels), numeric_scores, list(valid)


def _metrics(
    clusters: Sequence[str],
    labels: Sequence[bool],
    scores: Sequence[float],
    valid: Sequence[bool],
    threshold: float,
) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    for cluster in clusters:
        counts[cluster] = counts.get(cluster, 0) + 1
    cluster_count = len(counts)
    weights = [1.0 / (cluster_count * counts[cluster]) for cluster in clusters]
    routed = [
        (not is_valid) or (not math.isfinite(score)) or score >= threshold
        for score, is_valid in zip(scores, valid)
    ]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    positive_weight = sum(weight for weight, label in zip(weights, labels) if label)
    routed_positive_weight = sum(
        weight
        for weight, label, route in zip(weights, labels, routed)
        if label and route
    )
    routed_count = sum(routed)
    routed_positive = sum(label and route for label, route in zip(labels, routed))
    routed_negative = sum((not label) and route for label, route in zip(labels, routed))
    return {
        "sample_count": len(labels),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "cluster_count": cluster_count,
        "route_count": routed_count,
        "route_fraction": routed_count / len(labels),
        "routed_positive_count": routed_positive,
        "row_positive_route_recall": routed_positive / positive_count,
        "cluster_balanced_positive_route_recall": (
            routed_positive_weight / positive_weight
        ),
        "routed_negative_count": routed_negative,
        "row_negative_rejection": 1.0 - routed_negative / negative_count,
        "row_routed_precision": routed_positive / routed_count,
        "invalid_fail_open_count": sum(not flag for flag in valid),
    }


def freeze_route(validation_scores: Path) -> Dict[str, object]:
    validation_scores = validation_scores.resolve(strict=True)
    observed_sha256 = _sha256_file(validation_scores)
    if observed_sha256 != EXPECTED_VALIDATION_SCORES_SHA256:
        raise HybridRouteFreezeError("Full winner validation score SHA-256 differs")
    pair_ids, clusters, labels, scores, valid = _read_validation(validation_scores)
    del pair_ids
    candidates = sorted(
        {
            score
            for score, is_valid in zip(scores, valid)
            if is_valid and math.isfinite(score)
        },
        reverse=True,
    )
    selected = None
    selected_metrics = None
    for threshold in candidates:
        metrics = _metrics(clusters, labels, scores, valid, threshold)
        if (
            metrics["row_positive_route_recall"] + 1e-12 >= TARGET_RECALL
            and metrics["cluster_balanced_positive_route_recall"] + 1e-12
            >= TARGET_RECALL
        ):
            selected = threshold
            selected_metrics = metrics
            break
    if selected is None or selected_metrics is None:
        raise HybridRouteFreezeError("no validation route threshold meets recall targets")
    if selected != EXPECTED_ROUTE_THRESHOLD:
        raise HybridRouteFreezeError(
            "derived route threshold differs from the frozen protocol: {!r}".format(selected)
        )
    if not selected < FULL_CLASSIFICATION_THRESHOLD:
        raise HybridRouteFreezeError("route threshold must include every Full-positive decision")
    artifact: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_validation_only_before_hybrid_evaluation",
        "method_id": "full_n512_screen__shreddingnet_adapted_pose_hybrid99",
        "route": {
            "score": "full_n512.fused_probability",
            "rule": "fail_open_if_invalid_else_probability_greater_than_or_equal_to_threshold",
            "threshold": selected,
            "fit_method": "highest_observed_threshold_meeting_row_and_equal_cluster_positive_recall",
            "minimum_row_positive_recall": TARGET_RECALL,
            "minimum_cluster_balanced_positive_recall": TARGET_RECALL,
            "threshold_fit_source": "validation_only",
            "test_or_real_parameter_fit": False,
            "top_k_or_budget_cap": None,
        },
        "classification": {
            "score": "full_n512.fused_probability",
            "threshold": FULL_CLASSIFICATION_THRESHOLD,
            "shreddingnet_score_used": False,
        },
        "pose": {
            "source": "shreddingnet_adapted_for_routed_pairs_only",
            "compute_pose_for_shreddingnet_rejected_pairs": True,
            "unrouted_pairs_are_unconditional_pose_failures": True,
        },
        "validation": {
            "scores_filename": validation_scores.name,
            "scores_sha256": observed_sha256,
            "metrics": selected_metrics,
        },
        "frozen_checkpoints_sha256": {
            "full_n512": FULL_CHECKPOINT_SHA256,
            "shreddingnet_adapted": dict(SHREDDING_CHECKPOINT_SHA256),
        },
        "protocol": {
            "test_or_real_read_by_this_command": False,
            "single_primary_operating_point": True,
            "threshold_sweep_on_test_or_real_forbidden": True,
        },
    }
    artifact["content_sha256"] = hashlib.sha256(_canonical_bytes(artifact)).hexdigest()
    return artifact


def _exclusive_json(path: Path, value: object) -> None:
    path = path.resolve()
    if path.exists():
        raise HybridRouteFreezeError("refusing to overwrite existing route artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise HybridRouteFreezeError("refusing to overwrite existing route artifact") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    artifact = freeze_route(arguments.validation_scores)
    _exclusive_json(arguments.output, artifact)
    print(json.dumps({
        "status": artifact["status"],
        "output": str(arguments.output.resolve()),
        "threshold": artifact["route"]["threshold"],
        "content_sha256": artifact["content_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

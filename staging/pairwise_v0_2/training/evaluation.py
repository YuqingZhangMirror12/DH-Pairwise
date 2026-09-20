"""Validation-only Pairwise thresholding and cluster-aware evaluation.

Pair rows from one simulated source/component are correlated.  This module
therefore keeps two views separate:

* ``row``: every emitted pair has equal weight; and
* ``cluster_balanced``: every physical/source component has equal total weight.

The threshold artifact is fitted on validation only and is cryptographically
bound to the model/config/validation stream.  Sealed real-test records must
never be passed to ``fit_pairwise_threshold``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EVALUATION_VERSION = "dunhuang-pairwise-evaluation/0.2"


def _sha256_hex(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("{} must be a 64-character SHA-256".format(name))
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("{} must be hexadecimal".format(name)) from exc
    return value.casefold()


def _validation_split(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source_split is required")
    normalized = value.strip().casefold()
    allowed = normalized in {"val", "validation", "dev"} or (
        ("validation" in normalized or normalized.startswith("val_"))
        and "train" not in normalized
        and "test" not in normalized
    )
    if not allowed:
        raise ValueError("Pairwise threshold fitting is validation-only")
    return value


def _arrays(
    probability: Sequence[float],
    label: Sequence[bool],
    valid: Sequence[bool],
    cluster_id: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability_array = np.asarray(probability, dtype=np.float64)
    label_array = np.asarray(label)
    valid_array = np.asarray(valid)
    cluster_array = np.asarray(cluster_id, dtype=object)
    if probability_array.ndim != 1 or probability_array.size == 0:
        raise ValueError("probability must be a non-empty vector")
    expected = probability_array.shape
    if label_array.shape != expected or label_array.dtype != np.bool_:
        raise TypeError("label must be an explicit bool vector")
    if valid_array.shape != expected or valid_array.dtype != np.bool_:
        raise TypeError("valid must be an explicit bool vector")
    if cluster_array.shape != expected:
        raise ValueError("cluster_id shape differs from probability")
    usable = valid_array & np.isfinite(probability_array)
    if not np.any(usable):
        raise ValueError("no valid evaluation rows")
    probability_array = probability_array[usable]
    label_array = label_array[usable]
    cluster_array = cluster_array[usable]
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("valid probabilities must remain in [0, 1]")
    if any(not isinstance(value, str) or not value for value in cluster_array):
        raise ValueError("every usable row requires a non-empty cluster_id")
    if not np.any(label_array) or np.all(label_array):
        raise ValueError("evaluation requires both classes")
    return probability_array, label_array, cluster_array


def _weights(cluster_id: np.ndarray, cluster_balanced: bool) -> np.ndarray:
    if not cluster_balanced:
        return np.full(cluster_id.shape, 1.0 / len(cluster_id), dtype=np.float64)
    unique, inverse, counts = np.unique(cluster_id, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * counts[inverse].astype(np.float64))


def _confusion(
    probability: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    prediction = probability >= threshold
    tp = float(weight[prediction & label].sum())
    tn = float(weight[~prediction & ~label].sum())
    fp = float(weight[prediction & ~label].sum())
    fn = float(weight[~prediction & label].sum())
    precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0.0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": tp + tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "f1": f1,
        "tp_weight": tp,
        "tn_weight": tn,
        "fp_weight": fp,
        "fn_weight": fn,
    }


def _weighted_ranking(
    probability: np.ndarray, label: np.ndarray, weight: np.ndarray
) -> Tuple[float, float]:
    order = np.argsort(-probability, kind="mergesort")
    probability = probability[order]
    label = label[order]
    weight = weight[order]
    positive_total = float(weight[label].sum())
    negative_total = float(weight[~label].sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise ValueError("weighted ranking requires both classes")
    true_positive = 0.0
    false_positive = 0.0
    previous_tpr = 0.0
    previous_fpr = 0.0
    previous_recall = 0.0
    auroc = 0.0
    auprc = 0.0
    start = 0
    while start < len(probability):
        stop = start + 1
        while stop < len(probability) and probability[stop] == probability[start]:
            stop += 1
        group_label = label[start:stop]
        group_weight = weight[start:stop]
        true_positive += float(group_weight[group_label].sum())
        false_positive += float(group_weight[~group_label].sum())
        tpr = true_positive / positive_total
        fpr = false_positive / negative_total
        auroc += (fpr - previous_fpr) * (tpr + previous_tpr) * 0.5
        precision = true_positive / (true_positive + false_positive)
        auprc += (tpr - previous_recall) * precision
        previous_tpr = tpr
        previous_fpr = fpr
        previous_recall = tpr
        start = stop
    return float(auroc), float(auprc)


def evaluate_pairwise(
    probability: Sequence[float],
    label: Sequence[bool],
    valid: Sequence[bool],
    cluster_id: Sequence[str],
    *,
    threshold: float,
    ece_bins: int = 15,
) -> Dict[str, object]:
    """Return row-level and equal-cluster-weight Pairwise metrics."""

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    if isinstance(ece_bins, bool) or not isinstance(ece_bins, int) or ece_bins < 2:
        raise ValueError("ece_bins must be an integer >= 2")
    probability_array, label_array, cluster_array = _arrays(
        probability, label, valid, cluster_id
    )
    result: Dict[str, object] = {
        "schema_version": EVALUATION_VERSION,
        "sample_count": int(len(probability_array)),
        "positive_count": int(label_array.sum()),
        "negative_count": int((~label_array).sum()),
        "cluster_count": int(len(np.unique(cluster_array))),
        "threshold": float(threshold),
    }
    for name, balanced in (("row", False), ("cluster_balanced", True)):
        weight = _weights(cluster_array, balanced)
        metrics = _confusion(probability_array, label_array, weight, threshold)
        auroc, auprc = _weighted_ranking(probability_array, label_array, weight)
        target = label_array.astype(np.float64)
        metrics["auroc"] = auroc
        metrics["auprc"] = auprc
        metrics["brier"] = float(np.sum(weight * (probability_array - target) ** 2))
        ece = 0.0
        for index in range(ece_bins):
            lower = index / ece_bins
            upper = (index + 1) / ece_bins
            member = (probability_array >= lower) & (
                probability_array <= upper
                if index == ece_bins - 1
                else probability_array < upper
            )
            bin_weight = float(weight[member].sum())
            if bin_weight > 0.0:
                confidence = float(np.sum(weight[member] * probability_array[member]) / bin_weight)
                accuracy = float(np.sum(weight[member] * target[member]) / bin_weight)
                ece += bin_weight * abs(confidence - accuracy)
        metrics["ece"] = float(ece)
        result[name] = metrics
    return result


@dataclass(frozen=True)
class PairwiseThresholdArtifact:
    threshold: float
    fit_method: str
    source_split: str
    validation_fingerprint_sha256: str
    checkpoint_sha256: str
    model_config_sha256: str
    aggregation_config_sha256: str
    sample_count: int
    cluster_count: int
    achieved_cluster_balanced_f1: float
    achieved_cluster_balanced_precision: float
    achieved_cluster_balanced_recall: float
    schema_version: str = "dunhuang-pairwise-threshold/0.2"

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0, 1]")
        if self.fit_method != "maximize_cluster_balanced_f1":
            raise ValueError("unsupported Pairwise threshold method")
        _validation_split(self.source_split)
        for name in (
            "validation_fingerprint_sha256",
            "checkpoint_sha256",
            "model_config_sha256",
            "aggregation_config_sha256",
        ):
            _sha256_hex(getattr(self, name), name)
        if self.sample_count <= 0 or self.cluster_count <= 0:
            raise ValueError("threshold artifact counts must be positive")
        for name in (
            "achieved_cluster_balanced_f1",
            "achieved_cluster_balanced_precision",
            "achieved_cluster_balanced_recall",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("{} must be finite and in [0, 1]".format(name))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def fit_pairwise_threshold(
    probability: Sequence[float],
    label: Sequence[bool],
    valid: Sequence[bool],
    cluster_id: Sequence[str],
    *,
    source_split: str,
    validation_fingerprint_sha256: str,
    checkpoint_sha256: str,
    model_config_sha256: str,
    aggregation_config_sha256: str,
) -> PairwiseThresholdArtifact:
    """Fit a validation-only equal-cluster-weight maximum-F1 threshold."""

    source_split = _validation_split(source_split)
    probability_array, label_array, cluster_array = _arrays(
        probability, label, valid, cluster_id
    )
    weight = _weights(cluster_array, True)
    best: Optional[Tuple[Tuple[float, float, float, float], float, Dict[str, float]]] = None
    for threshold in sorted(set(float(value) for value in probability_array), reverse=True):
        metrics = _confusion(probability_array, label_array, weight, threshold)
        # Primary F1; ties prefer precision, recall, then the higher threshold.
        key = (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    if best is None:  # guarded by _arrays
        raise RuntimeError("no threshold candidates")
    metrics = best[2]
    return PairwiseThresholdArtifact(
        threshold=best[1],
        fit_method="maximize_cluster_balanced_f1",
        source_split=source_split,
        validation_fingerprint_sha256=_sha256_hex(
            validation_fingerprint_sha256, "validation_fingerprint_sha256"
        ),
        checkpoint_sha256=_sha256_hex(checkpoint_sha256, "checkpoint_sha256"),
        model_config_sha256=_sha256_hex(model_config_sha256, "model_config_sha256"),
        aggregation_config_sha256=_sha256_hex(
            aggregation_config_sha256, "aggregation_config_sha256"
        ),
        sample_count=int(len(probability_array)),
        cluster_count=int(len(np.unique(cluster_array))),
        achieved_cluster_balanced_f1=metrics["f1"],
        achieved_cluster_balanced_precision=metrics["precision"],
        achieved_cluster_balanced_recall=metrics["recall"],
    )


def recall_at_fixed_fpr(
    probability: Sequence[float],
    label: Sequence[bool],
    valid: Sequence[bool],
    cluster_id: Sequence[str],
    *,
    maximum_fpr: float,
) -> Dict[str, float]:
    """Best equal-cluster-weight recall under a frozen FPR ceiling."""

    if not math.isfinite(maximum_fpr) or not 0.0 <= maximum_fpr < 1.0:
        raise ValueError("maximum_fpr must be finite and in [0, 1)")
    probability_array, label_array, cluster_array = _arrays(
        probability, label, valid, cluster_id
    )
    weight = _weights(cluster_array, True)
    best: Optional[Tuple[Tuple[float, float, float], float, Dict[str, float]]] = None
    candidates = sorted(set(float(value) for value in probability_array), reverse=True)
    # A threshold immediately above the maximum is the legitimate all-negative
    # operating point.  Do not clip it to 1.0: saturated probability==1 rows
    # would otherwise be forced positive and could make a strict FPR ceiling
    # appear infeasible.
    candidates.insert(
        0, float(np.nextafter(max(probability_array), math.inf))
    )
    for threshold in candidates:
        metrics = _confusion(probability_array, label_array, weight, threshold)
        if metrics["false_positive_rate"] > maximum_fpr + 1e-12:
            continue
        key = (
            metrics["recall"],
            -metrics["false_positive_rate"],
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    if best is None:
        raise RuntimeError("no fixed-FPR operating point")
    return {
        "maximum_fpr": float(maximum_fpr),
        "threshold": float(best[1]),
        "recall": float(best[2]["recall"]),
        "observed_fpr": float(best[2]["false_positive_rate"]),
    }


def selective_risk_curve(
    probability: Sequence[float],
    label: Sequence[bool],
    valid: Sequence[bool],
    cluster_id: Sequence[str],
    *,
    coverages: Sequence[float] = (0.25, 0.5, 0.75, 0.9, 1.0),
    decision_threshold: float = 0.5,
) -> List[Dict[str, float]]:
    """Error risk for most-confident rows at predeclared coverage levels."""

    probability_array, label_array, cluster_array = _arrays(
        probability, label, valid, cluster_id
    )
    if not coverages:
        raise ValueError("coverages cannot be empty")
    if not math.isfinite(decision_threshold) or not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be finite and in [0, 1]")
    previous = 0.0
    for coverage in coverages:
        if (
            not math.isfinite(float(coverage))
            or not 0.0 < float(coverage) <= 1.0
            or float(coverage) <= previous
        ):
            raise ValueError("coverages must be finite, increasing, and in (0, 1]")
        previous = float(coverage)
    confidence = np.abs(probability_array - decision_threshold)
    order = np.argsort(-confidence, kind="mergesort")
    prediction = probability_array >= decision_threshold
    output = []
    for coverage in coverages:
        count = max(1, int(math.ceil(float(coverage) * len(order))))
        selected = order[:count]
        selected_clusters = cluster_array[selected]
        weight = _weights(selected_clusters, True)
        error = prediction[selected] != label_array[selected]
        output.append(
            {
                "requested_coverage": float(coverage),
                "observed_row_coverage": float(count / len(order)),
                "cluster_balanced_risk": float(weight[error].sum()),
                "accepted_count": float(count),
                "decision_threshold": float(decision_threshold),
            }
        )
    return output


def cluster_bootstrap_threshold_metrics(
    probability: Sequence[float],
    label: Sequence[bool],
    valid: Sequence[bool],
    cluster_id: Sequence[str],
    *,
    threshold: float,
    repetitions: int = 2000,
    seed: str = "pairwise-v0.2-cluster-bootstrap",
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Cluster bootstrap CIs for thresholded metrics and Brier score."""

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 100:
        raise ValueError("repetitions must be an integer >= 100")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a non-empty string")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    probability_array, label_array, cluster_array = _arrays(
        probability, label, valid, cluster_id
    )
    unique = sorted(set(cluster_array.tolist()))
    summaries = []
    for cluster in unique:
        member = cluster_array == cluster
        cluster_weight = np.full(int(member.sum()), 1.0 / int(member.sum()))
        metrics = _confusion(
            probability_array[member], label_array[member], cluster_weight, threshold
        )
        metrics["brier"] = float(
            np.mean((probability_array[member] - label_array[member].astype(float)) ** 2)
        )
        summaries.append(metrics)
    seed_value = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed_value)
    values = {name: [] for name in ("f1", "precision", "recall", "false_positive_rate", "brier")}
    for _ in range(repetitions):
        sample = [summaries[rng.randrange(len(summaries))] for _ in summaries]
        tp = sum(item["tp_weight"] for item in sample) / len(sample)
        fp = sum(item["fp_weight"] for item in sample) / len(sample)
        fn = sum(item["fn_weight"] for item in sample) / len(sample)
        tn = sum(item["tn_weight"] for item in sample) / len(sample)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        values["f1"].append(f1)
        values["precision"].append(precision)
        values["recall"].append(recall)
        values["false_positive_rate"].append(fpr)
        values["brier"].append(sum(item["brier"] for item in sample) / len(sample))
    alpha = (1.0 - confidence) / 2.0
    point = evaluate_pairwise(
        probability_array,
        label_array,
        np.ones(len(label_array), dtype=bool),
        cluster_array,
        threshold=threshold,
    )["cluster_balanced"]
    output: Dict[str, Dict[str, float]] = {}
    for name, samples in values.items():
        output[name] = {
            "estimate": float(point[name]),
            "lower": float(np.quantile(samples, alpha)),
            "upper": float(np.quantile(samples, 1.0 - alpha)),
        }
    return output


__all__ = [
    "EVALUATION_VERSION",
    "PairwiseThresholdArtifact",
    "cluster_bootstrap_threshold_metrics",
    "evaluate_pairwise",
    "fit_pairwise_threshold",
    "recall_at_fixed_fpr",
    "selective_risk_curve",
]

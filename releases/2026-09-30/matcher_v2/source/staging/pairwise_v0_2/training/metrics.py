"""Dependency-light binary metrics for train/validation receipts."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor


def _ranking_metrics(scores: List[float], labels: List[bool]) -> Tuple[float, float]:
    grouped = {}
    for score, label in zip(scores, labels):
        positive, negative = grouped.get(score, (0, 0))
        grouped[score] = (
            positive + int(label),
            negative + int(not label),
        )
    total_positive = sum(int(value) for value in labels)
    total_negative = len(labels) - total_positive
    tp = 0
    fp = 0
    previous_tpr = 0.0
    previous_fpr = 0.0
    auroc = 0.0
    average_precision = 0.0
    previous_recall = 0.0
    for score in sorted(grouped, reverse=True):
        positive, negative = grouped[score]
        tp += positive
        fp += negative
        tpr = tp / total_positive
        fpr = fp / total_negative
        auroc += (fpr - previous_fpr) * (tpr + previous_tpr) * 0.5
        recall = tpr
        precision = tp / (tp + fp)
        average_precision += (recall - previous_recall) * precision
        previous_tpr = tpr
        previous_fpr = fpr
        previous_recall = recall
    return auroc, average_precision


def binary_metrics(
    probability: Tensor,
    label: Tensor,
    valid: Tensor,
    *,
    threshold: float = 0.5,
    ece_bins: int = 15,
) -> Dict[str, float]:
    """Compute exact AUROC/AP plus thresholded and calibration metrics."""

    if probability.ndim != 1 or not probability.is_floating_point():
        raise TypeError("probability must be a floating-point [N] tensor")
    if label.dtype != torch.bool or tuple(label.shape) != tuple(probability.shape):
        raise TypeError("label must be bool and match probability")
    if valid.dtype != torch.bool or tuple(valid.shape) != tuple(probability.shape):
        raise TypeError("valid must be bool and match probability")
    if not 0.0 <= threshold <= 1.0 or ece_bins < 2:
        raise ValueError("threshold/ece_bins configuration is invalid")
    usable = valid & torch.isfinite(probability)
    probability = probability[usable].detach().to(torch.float64).cpu()
    label = label[usable].detach().cpu()
    if probability.numel() == 0:
        raise ValueError("no valid metric samples")
    if ((probability < 0.0) | (probability > 1.0)).any().item():
        raise ValueError("probabilities must remain in [0, 1]")
    positive = int(label.sum().item())
    negative = int(label.numel() - positive)
    if positive == 0 or negative == 0:
        raise ValueError("AUROC/AUPRC require both classes")
    prediction = probability >= threshold
    tp = int((prediction & label).sum().item())
    tn = int((~prediction & ~label).sum().item())
    fp = int((prediction & ~label).sum().item())
    fn = int((~prediction & label).sum().item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    auroc, auprc = _ranking_metrics(probability.tolist(), label.tolist())
    target = label.to(torch.float64)
    brier = float((probability - target).square().mean().item())
    ece = 0.0
    for index in range(ece_bins):
        lower = index / ece_bins
        upper = (index + 1) / ece_bins
        member = (probability >= lower) & (
            probability <= upper if index == ece_bins - 1 else probability < upper
        )
        if member.any().item():
            confidence = float(probability[member].mean().item())
            accuracy = float(target[member].mean().item())
            ece += float(member.to(torch.float64).mean().item()) * abs(
                confidence - accuracy
            )
    return {
        "sample_count": float(label.numel()),
        "positive_count": float(positive),
        "negative_count": float(negative),
        "accuracy": (tp + tn) / label.numel(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
    }


__all__ = ["binary_metrics"]

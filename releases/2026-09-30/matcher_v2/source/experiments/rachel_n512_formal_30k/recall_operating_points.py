"""Validation-only high-recall operating points; no pose gate or REAL fitting."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def _arrays(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels) or not np.isfinite(scores).all():
        raise ValueError("requires aligned finite 1D validation labels/scores")
    if not labels.any() or labels.all():
        raise ValueError("validation requires both classes")
    return labels, scores


def threshold_for_recall(labels, scores, target=.95):
    """Largest observed threshold attaining requested empirical positive recall.

    Accept score>=threshold. Ties stay together. The target is measured only on
    the calibration population; no guarantee transfers to another data domain.
    """
    labels, scores = _arrays(labels, scores)
    if not 0 < target <= 1:
        raise ValueError("recall target must lie in(0,1]")
    positives = np.sort(scores[labels])[::-1]
    required = int(np.ceil(float(target) * len(positives)))
    return float(positives[required - 1])


def recall_selection_key(labels, scores):
    labels, scores = _arrays(labels, scores)
    accepted = scores >= threshold_for_recall(labels, scores, .95)
    precision = float((labels & accepted).sum() / max(1, accepted.sum()))
    return precision, float(average_precision_score(labels, scores))


def fit_operating_points(labels, scores):
    from .run_layout_decoder_experiment import fit_threshold, classification
    labels, scores = _arrays(labels, scores)
    thresholds = {"max_f1": float(fit_threshold(labels, scores))}
    thresholds.update({"recall_%d" % int(target * 100): threshold_for_recall(labels, scores, target)
                       for target in (.90, .95, .98, .99)})
    # A requested95% VAL recall can be stricter than max-F1 when synthetic
    # separation is easy. The recall-first deployment candidate never narrows
    # the max-F1 acceptance set, and asks for99% empirical VAL recall.
    thresholds["recall_first"] = min(thresholds["max_f1"], thresholds["recall_99"])
    return dict(thresholds=thresholds,
        validation={name: classification(labels, scores, threshold) for name, threshold in thresholds.items()},
        selection_key=list(recall_selection_key(labels, scores)),
        rule="accept score>=threshold; recall targets empirical on calibration population only")

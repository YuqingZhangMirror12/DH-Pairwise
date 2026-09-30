import math

import numpy as np
import pytest

from staging.pairwise_v0_2.training.evaluation import (
    cluster_bootstrap_threshold_metrics,
    evaluate_pairwise,
    fit_pairwise_threshold,
    recall_at_fixed_fpr,
    selective_risk_curve,
)


SHA = "a" * 64


def _fixture():
    probability = np.asarray([0.95, 0.80, 0.60, 0.55, 0.45, 0.20, 0.10, 0.05])
    label = np.asarray([True, True, False, True, False, False, True, False])
    valid = np.ones(8, dtype=bool)
    # c1 has six rows, c2/c3 one each: row and cluster-balanced metrics differ.
    cluster = np.asarray(["c1", "c1", "c1", "c1", "c1", "c1", "c2", "c3"])
    return probability, label, valid, cluster


def test_row_and_cluster_balanced_metrics_are_separate():
    probability, label, valid, cluster = _fixture()
    result = evaluate_pairwise(
        probability, label, valid, cluster, threshold=0.5, ece_bins=5
    )
    assert result["sample_count"] == 8
    assert result["cluster_count"] == 3
    assert result["row"]["f1"] != result["cluster_balanced"]["f1"]
    for view in ("row", "cluster_balanced"):
        for name in ("auroc", "auprc", "f1", "brier", "ece"):
            assert 0.0 <= result[view][name] <= 1.0


def test_threshold_is_validation_only_and_hash_bound():
    probability, label, valid, cluster = _fixture()
    artifact = fit_pairwise_threshold(
        probability,
        label,
        valid,
        cluster,
        source_split="validation",
        validation_fingerprint_sha256=SHA,
        checkpoint_sha256="b" * 64,
        model_config_sha256="c" * 64,
        aggregation_config_sha256="d" * 64,
    )
    assert 0.0 <= artifact.threshold <= 1.0
    assert len(artifact.content_sha256) == 64
    assert artifact.sample_count == 8
    with pytest.raises(ValueError, match="validation-only"):
        fit_pairwise_threshold(
            probability,
            label,
            valid,
            cluster,
            source_split="test",
            validation_fingerprint_sha256=SHA,
            checkpoint_sha256="b" * 64,
            model_config_sha256="c" * 64,
            aggregation_config_sha256="d" * 64,
        )


def test_fixed_fpr_selective_curve_and_bootstrap_are_finite_deterministic():
    probability, label, valid, cluster = _fixture()
    point = recall_at_fixed_fpr(
        probability, label, valid, cluster, maximum_fpr=0.25
    )
    assert point["observed_fpr"] <= 0.25 + 1e-12
    curve = selective_risk_curve(
        probability, label, valid, cluster, coverages=(0.5, 1.0)
    )
    assert [row["requested_coverage"] for row in curve] == [0.5, 1.0]
    first = cluster_bootstrap_threshold_metrics(
        probability,
        label,
        valid,
        cluster,
        threshold=0.5,
        repetitions=200,
        seed="fixed",
    )
    second = cluster_bootstrap_threshold_metrics(
        probability,
        label,
        valid,
        cluster,
        threshold=0.5,
        repetitions=200,
        seed="fixed",
    )
    assert first == second
    for interval in first.values():
        assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0
        assert math.isfinite(interval["estimate"])


def test_fail_closed_input_contracts():
    probability, label, valid, cluster = _fixture()
    with pytest.raises(ValueError, match="both classes"):
        evaluate_pairwise(
            probability,
            np.ones_like(label),
            valid,
            cluster,
            threshold=0.5,
        )
    with pytest.raises(ValueError, match="cluster_id"):
        evaluate_pairwise(
            probability,
            label,
            valid,
            np.asarray([""] * len(cluster)),
            threshold=0.5,
        )
    with pytest.raises(ValueError, match="coverages"):
        selective_risk_curve(
            probability, label, valid, cluster, coverages=(0.8, 0.7)
        )


def test_saturated_scores_allow_all_negative_fixed_fpr_and_fitted_selective_threshold():
    probability = np.asarray([1.0, 1.0, 1.0, 1.0])
    label = np.asarray([True, False, True, False])
    valid = np.ones(4, dtype=bool)
    cluster = np.asarray(["a", "b", "c", "d"])

    point = recall_at_fixed_fpr(
        probability, label, valid, cluster, maximum_fpr=0.0
    )
    assert point["threshold"] > 1.0
    assert point["observed_fpr"] == 0.0
    curve = selective_risk_curve(
        probability,
        label,
        valid,
        cluster,
        coverages=(1.0,),
        decision_threshold=0.9,
    )
    assert curve[0]["decision_threshold"] == 0.9

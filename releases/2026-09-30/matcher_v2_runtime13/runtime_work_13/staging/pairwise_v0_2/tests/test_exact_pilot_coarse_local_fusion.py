from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from staging.pairwise_v0_2.baselines.exact_pilot_coarse_local_fusion import (
    BALANCED_POPULATION,
    EXACT_METHOD,
    FUSION_METHOD,
    SIAMESE_METHOD,
    STRICT_POPULATION,
    ExactPilotFusionError,
    align_synthetic_calibration,
    apply_exact_siamese_fusion,
    evaluate_frozen_fusion_on_real,
    fit_exact_siamese_fusion,
    freeze_parameters_then_open_real,
)


def _synthetic_documents():
    labels = [False, True, False, True, False, True, False, True]
    exact_scores = [0.5] * len(labels)
    siamese_scores = [0.05, 0.95, 0.10, 0.90, 0.15, 0.85, 0.20, 0.80]

    def rows(scores, *, include_valid):
        output = []
        for index, (label, score) in enumerate(zip(labels, scores)):
            row = {
                "pair_id": "synthetic-pair-{}".format(index),
                "component_id": "synthetic-parent-{}".format(index // 2),
                "label": label,
                "probability": score,
            }
            if include_valid:
                row["valid"] = True
            output.append(row)
        return {"records": output}

    return rows(exact_scores, include_valid=True), rows(
        siamese_scores, include_valid=False
    )


def _real_document():
    labels = [False, True, False, True, True, False, False, True]
    exact_scores = [0.15, 0.85, 0.25, 0.75, 0.70, 0.30, 0.35, 0.65]
    siamese_scores = [0.30, 0.70, 0.40, 0.60, 0.55, 0.45, 0.40, 0.60]
    rows = []
    for index, (label, exact, siamese) in enumerate(
        zip(labels, exact_scores, siamese_scores)
    ):
        exact_valid = index != 4
        rows.append(
            {
                "pair_id": "real-pair-{}".format(index),
                "cluster_id": "real-case-{}".format(index // 2),
                "label": label,
                "stratum": (
                    "strict_manifest_positive"
                    if index < 6 and label
                    else (
                        "strict_manifest_negative"
                        if index < 6
                        else "constructed_distractor_not_gt_negative"
                    )
                ),
                "methods": {
                    EXACT_METHOD: {
                        "probability": exact if exact_valid else None,
                        "valid": exact_valid,
                    },
                    SIAMESE_METHOD: {
                        "probability": siamese,
                        "valid": True,
                    },
                },
            }
        )
    return {"evaluated_pair_count": len(rows), "pairs": rows}


def test_alignment_and_equal_row_fit_select_informative_siamese_source():
    exact, siamese = _synthetic_documents()
    calibration = align_synthetic_calibration(exact, siamese)
    model = fit_exact_siamese_fusion(calibration)

    assert calibration.pair_count == 8
    assert model.calibration_positive_count == 4
    assert model.calibration_negative_count == 4
    assert model.siamese_weight == pytest.approx(1.0)
    assert model.portable_dict()["fit_weighting"] == "equal_row"
    assert model.portable_dict()["classification_threshold_origin"] == (
        "fixed_not_fitted"
    )


def test_alignment_rejects_any_pair_identity_or_order_change():
    exact, siamese = _synthetic_documents()
    siamese["records"][0], siamese["records"][1] = (
        siamese["records"][1],
        siamese["records"][0],
    )

    with pytest.raises(ExactPilotFusionError, match="identity differs"):
        align_synthetic_calibration(exact, siamese)


def test_frozen_real_scores_ignore_real_labels_and_bootstrap_is_deterministic():
    exact, siamese = _synthetic_documents()
    model = fit_exact_siamese_fusion(
        align_synthetic_calibration(exact, siamese)
    )
    document = _real_document()
    first = evaluate_frozen_fusion_on_real(
        document, model, bootstrap_replicates=50, bootstrap_seed=123
    )
    second = evaluate_frozen_fusion_on_real(
        document, model, bootstrap_replicates=50, bootstrap_seed=123
    )
    flipped = copy.deepcopy(document)
    for row in flipped["pairs"]:
        row["label"] = not row["label"]
    changed_labels = evaluate_frozen_fusion_on_real(
        flipped, model, bootstrap_replicates=50, bootstrap_seed=123
    )

    assert first["populations"] == second["populations"]
    assert first["populations"][STRICT_POPULATION]["pair_count"] == 6
    assert first["populations"][STRICT_POPULATION]["common_valid_count"] == 5
    assert first["populations"][BALANCED_POPULATION]["pair_count"] == 8
    assert first["populations"][BALANCED_POPULATION]["common_valid_count"] == 7
    for original, relabelled in zip(first["pairs"], changed_labels["pairs"]):
        assert original["methods"][FUSION_METHOD] == relabelled["methods"][
            FUSION_METHOD
        ]
    assert first["no_leakage"]["real_labels_or_scores_used_to_choose_weight"] is False
    assert first["no_leakage"]["real_labels_or_scores_used_to_choose_threshold"] is False


def test_parameter_artifact_exists_before_real_loader_is_called(tmp_path):
    exact, siamese = _synthetic_documents()
    calibration = align_synthetic_calibration(exact, siamese)
    parameters = tmp_path / "fusion_parameters.json"
    observations = []

    def load_real():
        observations.append(parameters.is_file())
        frozen = json.loads(parameters.read_text(encoding="utf-8"))
        assert frozen["real_evaluation_accessed_during_fit"] is False
        return _real_document()

    model, result = freeze_parameters_then_open_real(
        calibration=calibration,
        parameter_path=parameters,
        real_document_loader=load_real,
        bootstrap_replicates=10,
        bootstrap_seed=321,
    )

    assert observations == [True]
    assert parameters.is_file()
    assert model.siamese_weight == pytest.approx(1.0)
    assert result["status"] == (
        "complete_frozen_synthetic_calibration_real_evaluation"
    )


def test_application_has_no_label_argument_and_preserves_invalid_rows():
    exact, siamese = _synthetic_documents()
    model = fit_exact_siamese_fusion(
        align_synthetic_calibration(exact, siamese)
    )
    output = apply_exact_siamese_fusion(
        model,
        [0.2, 0.8, 0.6],
        [0.3, 0.7, 0.5],
        np.asarray([True, False, True], dtype=np.bool_),
    )

    assert np.isfinite(output[[0, 2]]).all()
    assert np.isnan(output[1])

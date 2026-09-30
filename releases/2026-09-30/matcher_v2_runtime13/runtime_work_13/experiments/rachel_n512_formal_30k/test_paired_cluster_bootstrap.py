from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).with_name("paired_cluster_bootstrap.py")
SPEC = importlib.util.spec_from_file_location("rachel_n512_paired_bootstrap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_method_order_requires_complete_same_data_benchmark_pair():
    complete = (
        "coarse_only",
        "full_n512",
        "matched_mm_converged",
        "matched_mm_same_exposure_epoch5",
        *MODULE.SAME_DATA_BENCHMARK_METHODS,
    )
    assert MODULE._method_order(complete) == complete
    with pytest.raises(MODULE.PairedBootstrapError, match="present together"):
        MODULE._method_order(complete[:-1])


@pytest.fixture(autouse=True)
def _small_formal_authorities(monkeypatch):
    """Keep fixtures small while exercising the production fail-closed gates."""

    constructed_pairs = [
        {"pair_id": "r6", "source_case_uids": ["case0", "case3"]},
        {"pair_id": "r7", "source_case_uids": ["case1", "case2"]},
    ]
    values = {
        "EXPECTED_SYNTHETIC_COUNT": 8,
        "EXPECTED_SYNTHETIC_POSITIVE": 4,
        "EXPECTED_SYNTHETIC_UNIT_COUNT": 3,
        "EXPECTED_SYNTHETIC_CLUSTER_COUNT": 4,
        "EXPECTED_SYNTHETIC_MANIFEST_SHA256": "9" * 64,
        "EXPECTED_SYNTHETIC_PAIR_ORDER_SHA256": hashlib.sha256(
            _canonical(["p{}".format(index) for index in range(8)])
        ).hexdigest(),
        "EXPECTED_REAL_MANIFEST_SHA256": "8" * 64,
        "EXPECTED_STRICT_COUNT": 6,
        "EXPECTED_STRICT_POSITIVE": 4,
        "EXPECTED_STRICT_CASE_COUNT": 3,
        "EXPECTED_STRICT_PAIR_ORDER_SHA256": hashlib.sha256(
            _canonical(["r{}".format(index) for index in range(6)])
        ).hexdigest(),
        "EXPECTED_BALANCED_COUNT": 8,
        "EXPECTED_BALANCED_POSITIVE": 4,
        "EXPECTED_CONSTRUCTED_COUNT": 2,
        "EXPECTED_CONSTRUCTED_SELECTION_SHA256": hashlib.sha256(
            _canonical(constructed_pairs)
        ).hexdigest(),
    }
    for name, value in values.items():
        monkeypatch.setattr(MODULE, name, value)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _threshold(arm: str, checkpoint: str) -> dict:
    return {
        "threshold": 0.5,
        "fit_method": "maximize_cluster_balanced_f1",
        "source_split": "val",
        "validation_fingerprint_sha256": "1" * 64,
        "checkpoint_sha256": checkpoint,
        "model_config_sha256": "2" * 64,
        "aggregation_config_sha256": "3" * 64,
        "sample_count": 8,
        "cluster_count": 4,
        "achieved_cluster_balanced_f1": 0.5,
        "achieved_cluster_balanced_precision": 0.5,
        "achieved_cluster_balanced_recall": 0.5,
        "schema_version": "dunhuang-pairwise-threshold/0.2",
    }


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _sealed_fixture(tmp_path: Path):
    root = tmp_path / "sealed"
    labels = [True, False, True, False, True, False, True, False]
    coarse_scores = [0.60, 0.50, 0.40, 0.20, 0.80, 0.70, 0.30, 0.10]
    full_scores = [0.90, 0.10, 0.85, 0.15, 0.80, 0.20, 0.75, 0.25]
    validity = {
        "coarse_only": [False, True, True, True, True, True, True, True],
        "full_n512": [True, True, True, True, True, True, True, False],
    }
    arm_results = []
    rows_by_arm = {}
    dependency_by_cluster = (
        ["u0"],
        ["u0", "u1"],
        ["u1", "u2"],
        ["u2"],
    )
    for arm, scores in (("coarse_only", coarse_scores), ("full_n512", full_scores)):
        checkpoint = ("a" if arm == "coarse_only" else "b") * 64
        threshold = _threshold(arm, checkpoint)
        rows = []
        for index, (label, score) in enumerate(zip(labels, scores)):
            valid = validity[arm][index]
            main = "coarse" if arm == "coarse_only" else "fused"
            row = {
                "schema_version": "rachel-n512-sealed-test-pair/1.0",
                "arm": arm,
                "pair_id": "p{}".format(index),
                "label": label,
                "cluster_id": "c{}".format(index // 2),
                "source_unit_ids": dependency_by_cluster[index // 2],
                "scores": {main: {"probability": score, "valid": valid}},
                "main_score": main,
                "decision": {
                    "validation_threshold": 0.5,
                    "valid": valid,
                    "predicted_label": score >= 0.5 if valid else None,
                },
            }
            if arm == "full_n512":
                registration_valid = label and valid
                translation_error = float(index + 1) if registration_valid else None
                e_rmse = float(index + 1) if registration_valid else None
                predicted_edge = valid and score >= 0.5
                row["geometry"] = {
                    "decision_valid": valid,
                    "translation_target_valid": label,
                    "translation_hat_rc": [float(index), -float(index)],
                    "translation_l2_px": translation_error,
                    "correspondence": {
                        "strict_true_positive": 2 if registration_valid else 0,
                        "strict_predicted_count": 2 if registration_valid else 0,
                        "mutual_top1_true_positive": 2 if registration_valid else 0,
                        "mutual_top1_predicted_count": 2 if registration_valid else 0,
                        "target_count": 2 if label else 0,
                        "dustbin_correct": 4 if valid else 0,
                        "dustbin_total": 4,
                    },
                    "pairingnet_style_registration": {
                        "target_valid": label,
                        "prediction_valid": valid,
                        "identity_fallback_used": (
                            not valid if label else None
                        ),
                        "e_rmse": e_rmse,
                        "registration_recall_lt4_success": (
                            e_rmse < 4.0 if registration_valid else False if label else None
                        ),
                        "symmetric_hausdorff_px": e_rmse,
                        "translation_l2_px": translation_error,
                        "normalized_translation_error": (
                            translation_error / 20.0
                            if registration_valid
                            else None
                        ),
                        "source_contour_area_px2": 10.0 if registration_valid else None,
                        "target_contour_area_px2": 10.0 if registration_valid else None,
                        "rotation_error": {
                            "status": "not_applicable_conditioned_upright_orientation",
                            "estimated_or_supervised": False,
                            "ground_truth_rotation_degrees": 0.0,
                        },
                    },
                    "assembly_edge": {
                        "target_edge": label,
                        "predicted_edge": predicted_edge,
                        "true_positive_by_tolerance": {
                            "at_{}".format(tolerance): bool(
                                label
                                and predicted_edge
                                and translation_error is not None
                                and translation_error <= tolerance
                            )
                            for tolerance in (2, 5, 8, 10, 100)
                        },
                    },
                }
            rows.append(row)
        score_path = root / arm / "pair_scores.jsonl"
        _jsonl(score_path, rows)
        rows_by_arm[arm] = rows
        arm_results.append(
            {
                "arm": arm,
                "winner_checkpoint_sha256": checkpoint,
                "validation_threshold": threshold,
                "validation_threshold_sha256": hashlib.sha256(
                    _canonical(threshold)
                ).hexdigest(),
                "pair_scores": str(score_path.relative_to(root)),
                "pair_scores_sha256": _sha(score_path),
                "pair_scores_count": len(rows),
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "threshold_fit_performed": False,
            }
        )
    pair_ids = ["p{}".format(index) for index in range(8)]
    receipt = {
        "schema_version": "rachel-n512-sealed-synthetic-test/1.0",
        "status": MODULE.COMPATIBILITY_SEALED_STATUS,
        "formal_evaluation": False,
        "compatibility_mode": True,
        "source_training_run": {
            "status_at_open": "complete_train_validation_only",
            "winners_frozen_before_test_open": True,
            "both_arms_validation_plateau_verified": True,
            "formal_config_verified": True,
            "receipt_sha256": "4" * 64,
        },
        "test_population": {
            "expected_count": 8,
            "observed_count": 8,
            "positive_count": 4,
            "negative_count": 4,
            "exact_one_to_one": True,
            "cluster_count": 4,
            "manifest_sha256": "9" * 64,
            "pair_ids_sha256": hashlib.sha256(_canonical(pair_ids)).hexdigest(),
        },
        "protocol": {
            "formal_evaluation": False,
            "compatibility_mode": True,
            "formal_exact_six_frozen_before_test_open": False,
            "formal_method_inventory": None,
            "main_threshold": "validation_fit_checkpoint_bound_artifact_only",
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_fit_performed": False,
            "cross_arm_winner_selected_on_test": False,
            "all_requested_winners_and_thresholds_frozen_before_current_test_open": True,
            "formal_config_verified": True,
            "both_arms_validation_plateau_verified": True,
            "evaluation_history_disclosure": {
                "prior_epoch5_synthetic_test_completed": True,
                "prior_epoch5_synthetic_test_human_visible_before_continuation": True,
                "prior_epoch5_synthetic_test_used_by_automated_checkpoint_selection": False,
                "prior_epoch5_synthetic_test_used_by_automated_threshold_fitting": False,
                "prior_epoch5_synthetic_test_used_by_automated_early_stopping": False,
                "prior_epoch5_synthetic_test_used_by_automated_scheduler": False,
                "claim_no_human_cognitive_influence": False,
                "prior_epoch5_real_evaluator_started_then_stopped": True,
                "prior_epoch5_real_result_formed_or_read": False,
                "prior_convergence_time_synthetic_test_mask_morphology_review": True,
                "prior_convergence_time_synthetic_test_mask_sample_count": 500,
                "prior_convergence_time_synthetic_test_labels_read": False,
                "prior_convergence_time_synthetic_test_model_scores_read": False,
                "prior_convergence_time_synthetic_test_activity_used_for_training_checkpoint_or_threshold_selection": False,
                "prior_convergence_time_synthetic_test_activity_influenced_fixed_corrosion_conditions": False,
                "prior_convergence_time_synthetic_test_activity_used_for_corrosion_runtime_and_representation_QA": True,
                "real_data_accessed_in_that_activity": False,
                "claim_of_pristine_first_project_test": False,
            },
        },
        "arm_results": arm_results,
        "test_accessed": True,
        "real_external_test_accessed": False,
    }
    receipt_path = root / "test_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return receipt_path, receipt, rows_by_arm


def _real_fixture(tmp_path: Path, *, balanced: bool = True, matched: bool = True):
    labels = [True, False, True, False, True, True, False, False]
    if not balanced:
        labels = labels[:6]
    methods = ["coarse_only", "full_n512"] + (["matched_mm"] if matched else [])
    thresholds = {
        arm: _threshold(arm, ("c" if arm == "coarse_only" else "d") * 64)
        for arm in ("coarse_only", "full_n512")
    }
    metadata = {
        arm: {
            "winner": {
                "checkpoint_sha256": thresholds[arm]["checkpoint_sha256"],
                "validation_threshold": thresholds[arm],
            }
        }
        for arm in ("coarse_only", "full_n512")
    }
    if matched:
        checkpoint = "e" * 64
        matched_threshold = _threshold("matched_mm", checkpoint)
        metadata["matched_mm"] = {
            "score_semantics": "historical_mm_probability",
            "winner": {
                "checkpoint_sha256": checkpoint,
                "validation_threshold": matched_threshold,
            },
        }
    rows = []
    for index, label in enumerate(labels):
        probabilities = {
            "coarse_only": [0.7, 0.6, 0.4, 0.3, 0.8, 0.5, 0.2, 0.1][index],
            "full_n512": [0.9, 0.1, 0.85, 0.15, 0.8, 0.2, 0.75, 0.25][index],
            "matched_mm": [0.8, 0.2, 0.6, 0.4, 0.55, 0.45, 0.35, 0.3][index],
        }
        method_rows = {}
        for method in methods:
            probability = probabilities[method]
            prediction = {"probability": probability, "valid": True}
            if method == "matched_mm":
                prediction["score_semantics"] = "historical_mm_probability"
            prediction["decision_at_frozen_validation_threshold"] = probability >= 0.5
            method_rows[method] = prediction
        if index < 6:
            case_cluster = "case{}".format(index // 2)
            source_cases = [case_cluster]
        else:
            source_cases = ["case0", "case3"] if index == 6 else ["case1", "case2"]
            case_cluster = "casepair{}".format(index)
        rows.append(
            {
                "pair_id": "r{}".format(index),
                "label": label,
                "case_cluster": case_cluster,
                "source_case_uids": source_cases,
                "coarse_full_common_valid": True,
                "methods": method_rows,
            }
        )
    status = (
        MODULE.COMPATIBILITY_REAL_BALANCED_STATUS
        if balanced
        else MODULE.COMPATIBILITY_REAL_STRICT_STATUS
    )
    dataset = {
        "pair_count": len(rows),
        "positive_count": sum(labels),
        "negative_count": len(rows) - sum(labels),
        "case_cluster_count": len({row["case_cluster"] for row in rows}),
        "manifest_sha256": "8" * 64,
        "pair_order_sha256": hashlib.sha256(
            _canonical([row["pair_id"] for row in rows])
        ).hexdigest(),
    }
    document = {
        "schema_version": "rachel-n512-real-external/1.0",
        "status": status,
        "dataset": dataset,
        "protocol": {
            "formal_evaluation": False,
            "compatibility_mode": True,
            "formal_exact_six_frozen_before_real_open": False,
            "formal_method_inventory": None,
            "winner_and_validation_threshold_frozen_before_real_open": True,
            "real_correspondence_or_translation_gt_read": False,
            "primary_metrics": "threshold_free_AUROC_AUPRC",
            "thresholded_metrics": "secondary_frozen_validation_threshold_only",
            "all_requested_winners_and_thresholds_frozen_before_current_real_open": True,
            "formal_config_verified": True,
            "both_arms_validation_plateau_verified": True,
            "positive_direction_derived_or_bbox_read": False,
            "evaluation_history_disclosure": {
                "prior_epoch5_synthetic_test_completed": True,
                "prior_epoch5_synthetic_test_human_visible_before_continuation": True,
                "prior_epoch5_synthetic_test_used_by_automated_checkpoint_selection": False,
                "prior_epoch5_synthetic_test_used_by_automated_threshold_fitting": False,
                "prior_epoch5_synthetic_test_used_by_automated_early_stopping": False,
                "prior_epoch5_synthetic_test_used_by_automated_scheduler": False,
                "claim_no_human_cognitive_influence": False,
                "prior_epoch5_real_evaluator_started_then_stopped": True,
                "prior_epoch5_real_result_formed_or_read": False,
                "prior_convergence_time_synthetic_test_mask_morphology_review": True,
                "prior_convergence_time_synthetic_test_mask_sample_count": 500,
                "prior_convergence_time_synthetic_test_labels_read": False,
                "prior_convergence_time_synthetic_test_model_scores_read": False,
                "prior_convergence_time_synthetic_test_activity_used_for_training_checkpoint_or_threshold_selection": False,
                "prior_convergence_time_synthetic_test_activity_influenced_fixed_corrosion_conditions": False,
                "prior_convergence_time_synthetic_test_activity_used_for_corrosion_runtime_and_representation_QA": True,
                "real_data_accessed_in_that_activity": False,
                "claim_of_project_first_real_access": False,
            },
        },
        "source_training_run": {
            "status_at_open": "complete_train_validation_only",
            "receipt_sha256": "6" * 64,
            "both_arms_validation_plateau_verified": True,
            "formal_config_verified": True,
        },
        "methods": metadata,
        "pairs": rows,
    }
    if balanced:
        dataset.update(
            {
                "strict_prefix_count": 6,
                "constructed_count": 2,
                "strict_prefix_preserved_exactly": True,
            }
        )
        document["negative_semantics"] = {
            "constructed_are_GT_negatives": False,
            "never_used_for_training_threshold_or_tuning": True,
        }
        construction = {
            "status": "complete_label_blind_constructed_distractor_plan",
            "constructed_selection_sha256": (
                MODULE.EXPECTED_CONSTRUCTED_SELECTION_SHA256
            ),
            "selection_uses_model_scores": False,
            "selection_uses_pair_labels": False,
            "selection_uses_gt_bbox_or_direction": False,
            "same_case_forbidden": True,
            "same_alpha_sha256_forbidden": True,
            "uses_per_fragment_occurrence": 1,
            "max_pairs_per_unordered_case_pair": 1,
            "negative_semantics": "constructed_not_GT_negative",
            "constructed_pairs": [
                {"pair_id": "r6", "source_case_uids": ["case0", "case3"]},
                {"pair_id": "r7", "source_case_uids": ["case1", "case2"]},
            ],
        }
        construction["content_sha256"] = hashlib.sha256(
            _canonical(construction)
        ).hexdigest()
        document["construction_receipt"] = construction
    if matched:
        document["source_matched_mm_training_run"] = {
            "status_at_open": "complete_train_validation_only",
            "receipt_sha256": "7" * 64,
            "converged_and_epoch5_winners_frozen_before_current_real_open": True,
            "formal_config_verified": True,
        }
    path = tmp_path / ("balanced.json" if balanced else "strict.json")
    path.write_text(json.dumps(document, sort_keys=True) + "\n")
    return path, document


def test_sealed_common_valid_paired_bootstrap_is_deterministic(tmp_path: Path):
    receipt, _, _ = _sealed_fixture(tmp_path)
    first = MODULE.analyze_synthetic(
        receipt, replicates=400, seed=19, compatibility_mode=True
    )
    second = MODULE.analyze_synthetic(
        receipt, replicates=400, seed=19, compatibility_mode=True
    )

    assert first["common_population"]["row_count"] == 6
    assert first["formal_evaluation"] is False
    assert first["protocol"]["formal_status_claimed"] is False
    assert first["coverage"]["all_method_common_valid"]["valid_fraction"] == 0.75
    assert (
        first["paired_endpoint_pigeonhole_bootstrap"]
        == second["paired_endpoint_pigeonhole_bootstrap"]
    )
    delta = first["paired_endpoint_pigeonhole_bootstrap"]["metrics"]["auroc"][
        "paired_deltas"
    ]["full_n512_minus_coarse_only"]
    assert delta["point_estimate"] > 0
    assert len(delta["percentile_95_ci"]) == 2
    disclosure = first["protocol"]["automated_no_test_or_real_tuning"]
    assert disclosure["automated_nonuse_verified"] is True
    assert disclosure["claim_no_human_cognitive_influence"] is False
    assert disclosure["evaluation_history_disclosure"][
        "prior_epoch5_synthetic_test_human_visible_before_continuation"
    ] is True
    assert len(first["common_population"]["semantic_sha256"]) == 64
    bootstrap = first["paired_endpoint_pigeonhole_bootstrap"]
    assert bootstrap["sampling_dependency_unit_count"] == 3
    assert bootstrap["sampling_pair_cluster_count_descriptive"] == 4
    assert bootstrap["bootstrap"] == "endpoint-unit_pigeonhole_product_multiplicity"
    geometry = first["synthetic_geometry"]
    assert geometry["shared_draws_across_all_direct_geometry_metrics"] is True
    assert geometry["full_n512_minus_coarse_only"]["status"] == "not_applicable"
    assert (
        geometry["full_n512"]["joint_success_at_validation_threshold"][
            "success_by_tolerance"
        ]["success_at_8px"]["valid_replicates"]
        > 0
    )
    direct = geometry["full_n512"]
    assert direct["pairingnet_style_registration"][
        "registration_recall_e_rmse_lt4"
    ]["point_estimate"] == pytest.approx(0.5)
    assert direct["pairingnet_style_registration"]["rotation_error"]["status"] == (
        "not_applicable_conditioned_upright_orientation"
    )
    assert direct["correspondence"]["strict_dustbin_aware"]["harmonic_f1"][
        "valid_replicates"
    ] > 0
    assert direct["assembly_edge_at_validation_threshold"]["by_tolerance"][
        "at_100"
    ]["harmonic_f1"]["point_estimate"] == pytest.approx(1.0)
    assert direct["translation_l2_px"]["p90"]["point_estimate"] == pytest.approx(
        6.4
    )


def test_stats_default_rejects_explicit_legacy_compatibility_inputs(tmp_path: Path):
    sealed_path, _, _ = _sealed_fixture(tmp_path)
    real_path, _ = _real_fixture(tmp_path, balanced=True, matched=True)
    with pytest.raises(MODULE.PairedBootstrapError, match="completed synthetic"):
        MODULE.load_sealed(sealed_path)
    with pytest.raises(
        MODULE.PairedBootstrapError,
        match="combined balanced1016 exact-prefix wrapper",
    ):
        MODULE.load_real(real_path)


def test_stats_formal_sealed_rejects_non_exact_six_inventory(tmp_path: Path):
    receipt_path, receipt, _ = _sealed_fixture(tmp_path)
    receipt["status"] = MODULE.FORMAL_SEALED_STATUS
    receipt["formal_evaluation"] = True
    receipt["compatibility_mode"] = False
    receipt["protocol"].update(
        {
            "formal_evaluation": True,
            "compatibility_mode": False,
            "formal_exact_six_frozen_before_test_open": True,
            "formal_method_inventory": [
                "coarse_only",
                "full_n512",
                "matched_mm_converged",
                "matched_mm_same_exposure_epoch5",
                *MODULE.SAME_DATA_BENCHMARK_METHODS,
            ],
        }
    )
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    with pytest.raises(MODULE.PairedBootstrapError, match="exact-six methods"):
        MODULE.load_sealed(receipt_path)


def test_stats_formal_real_rejects_non_exact_six_inventory(tmp_path: Path):
    path, balanced = _real_fixture(tmp_path, balanced=True, matched=False)
    formal_inventory = [
        "coarse_only",
        "full_n512",
        "matched_mm_converged",
        "matched_mm_same_exposure_epoch5",
        *MODULE.SAME_DATA_BENCHMARK_METHODS,
    ]
    balanced["status"] = "complete_balanced_1016_target_blind_external_test"
    balanced["protocol"].update(
        {
            "formal_evaluation": True,
            "compatibility_mode": False,
            "formal_exact_six_frozen_before_real_open": True,
            "balanced1016_single_forward_required": True,
            "formal_method_inventory": formal_inventory,
        }
    )
    strict = json.loads(json.dumps(balanced))
    strict["status"] = "complete_strict_547_target_blind_external_test"
    strict["pairs"] = strict["pairs"][:6]
    wrapper = {
        "schema_version": MODULE.REAL_SCHEMA_VERSION,
        "status": MODULE.FORMAL_REAL_COMBINED_STATUS,
        "forward_contract": {
            "forward_pair_count_per_arm": 8,
            "strict_547_derived_from_exact_prediction_prefix": True,
            "strict_pairs_forwarded_twice": False,
            "methods": ["coarse_only", "full_n512"],
            "all_methods_share_exact_strict_prediction_prefix": True,
        },
        "strict_547": strict,
        "balanced_1016": balanced,
    }
    path.write_text(json.dumps(wrapper, sort_keys=True) + "\n")
    with pytest.raises(MODULE.PairedBootstrapError, match="exact-six methods"):
        MODULE.load_real(path)


def test_stats_cli_defaults_to_formal_and_compatibility_is_explicit(tmp_path: Path):
    arguments = [
        "--synthetic-receipt",
        str(tmp_path / "receipt.json"),
        "--output",
        str(tmp_path / "result.json"),
    ]
    assert MODULE._parser().parse_args(arguments).compatibility_non_formal is False
    assert (
        MODULE._parser()
        .parse_args(arguments + ["--compatibility-non-formal"])
        .compatibility_non_formal
        is True
    )


def test_analyze_loaded_cannot_emit_formal_status_for_legacy_inventory(
    tmp_path: Path,
):
    loaded = MODULE.LoadedEvaluation(
        source_kind="sealed_synthetic",
        source_path=tmp_path / "unused.json",
        source_sha256="0" * 64,
        methods=("coarse_only", "full_n512"),
        rows=(),
        strict_prefix_count=None,
        validation_thresholds={"coarse_only": 0.5, "full_n512": 0.5},
        no_test_tuning_evidence={},
        input_files=(),
        formal_evaluation=True,
    )
    with pytest.raises(MODULE.PairedBootstrapError, match="exact-six methods"):
        MODULE.analyze_loaded(loaded, replicates=2, seed=1)


def test_pigeonhole_weights_use_shared_endpoint_unit_multiplicities():
    rows = tuple(
        MODULE.PairRow(
            pair_id="p{}".format(index),
            label=index % 2 == 0,
            cluster="c{}".format(index),
            dependency_units=units,
            stratum="fixture",
            probability={"coarse_only": 0.5, "full_n512": 0.5},
            valid={"coarse_only": True, "full_n512": True},
        )
        for index, units in enumerate((("u0",), ("u0", "u1"), ("u1", "u2"), ("u2",)))
    )
    units, first, second = MODULE._dependency_index(rows)

    class _FixedRng:
        @staticmethod
        def integers(low, high, size):
            assert (low, high, size) == (0, 3, 3)
            return MODULE.np.asarray([0, 0, 1], dtype=MODULE.np.int64)

    weights = MODULE._pigeonhole_weights(len(units), first, second, _FixedRng())
    assert weights.tolist() == [2.0, 2.0, 0.0, 0.0]


def test_sealed_discovers_both_matched_mm_controls_with_honest_score_semantics(
    tmp_path: Path,
):
    receipt_path, receipt, _ = _sealed_fixture(tmp_path)
    labels = [True, False, True, False, True, False, True, False]
    methods = (
        "matched_mm_converged",
        "matched_mm_same_exposure_epoch5",
    )
    dependency_by_cluster = (
        ["u0"],
        ["u0", "u1"],
        ["u1", "u2"],
        ["u2"],
    )
    for method_index, method in enumerate(methods):
        checkpoint = ("e" if method_index == 0 else "f") * 64
        threshold = _threshold(method, checkpoint)
        scores = [
            0.85,
            0.15,
            0.80,
            0.20,
            0.75,
            0.25,
            0.70,
            0.30,
        ]
        rows = [
            {
                "schema_version": "rachel-n512-sealed-test-pair/1.0",
                "arm": method,
                "pair_id": "p{}".format(index),
                "label": label,
                "cluster_id": "c{}".format(index // 2),
                "source_unit_ids": dependency_by_cluster[index // 2],
                "scores": {
                    "historical_mm_probability": {
                        "probability": score,
                        "valid": True,
                    }
                },
                "main_score": "historical_mm_probability",
                "decision": {
                    "validation_threshold": 0.5,
                    "valid": True,
                    "predicted_label": score >= 0.5,
                },
            }
            for index, (label, score) in enumerate(zip(labels, scores))
        ]
        score_path = receipt_path.parent / method / "pair_scores.jsonl"
        _jsonl(score_path, rows)
        receipt["arm_results"].append(
            {
                "arm": method,
                "score_semantics": "historical_mm_probability",
                "winner_checkpoint_sha256": checkpoint,
                "validation_threshold": threshold,
                "validation_threshold_sha256": hashlib.sha256(
                    _canonical(threshold)
                ).hexdigest(),
                "pair_scores": str(score_path.relative_to(receipt_path.parent)),
                "pair_scores_sha256": _sha(score_path),
                "pair_scores_count": len(rows),
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "threshold_fit_performed": False,
            }
        )
    receipt["source_matched_mm_training_run"] = {
        "status_at_open": "complete_train_validation_only",
        "receipt_sha256": "5" * 64,
        "converged_and_epoch5_winners_frozen_before_test_open": True,
        "formal_config_verified": True,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    loaded = MODULE.load_sealed(receipt_path, compatibility_mode=True)
    assert loaded.methods == (
        "coarse_only",
        "full_n512",
        "matched_mm_converged",
        "matched_mm_same_exposure_epoch5",
    )
    assert all(row.valid["matched_mm_converged"] for row in loaded.rows)
    result = MODULE.analyze_synthetic(
        receipt_path, replicates=20, seed=11, compatibility_mode=True
    )
    assert result["methods"] == list(loaded.methods)
    deltas = result["paired_endpoint_pigeonhole_bootstrap"]["metrics"]["auroc"][
        "paired_deltas"
    ]
    assert "full_n512_minus_matched_mm_converged" in deltas
    assert "full_n512_minus_matched_mm_same_exposure_epoch5" in deltas


def test_sealed_rejects_cross_arm_label_mismatch_even_with_updated_file_sha(
    tmp_path: Path,
):
    receipt_path, receipt, rows = _sealed_fixture(tmp_path)
    rows["full_n512"][2]["label"] = False
    score_path = receipt_path.parent / "full_n512" / "pair_scores.jsonl"
    _jsonl(score_path, rows["full_n512"])
    for arm in receipt["arm_results"]:
        if arm["arm"] == "full_n512":
            arm["pair_scores_sha256"] = _sha(score_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    with pytest.raises(
        MODULE.PairedBootstrapError, match="label differs for pair_id=p2"
    ):
        MODULE.load_sealed(receipt_path, compatibility_mode=True)


def test_sealed_rejects_test_time_threshold_fit(tmp_path: Path):
    receipt_path, receipt, _ = _sealed_fixture(tmp_path)
    receipt["protocol"]["threshold_fit_performed"] = True
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    with pytest.raises(MODULE.PairedBootstrapError, match="tuning"):
        MODULE.load_sealed(receipt_path, compatibility_mode=True)


def test_sealed_accepts_exact_float32_translation_rounding_boundary(tmp_path: Path):
    receipt_path, receipt, rows_by_arm = _sealed_fixture(tmp_path)
    rows = rows_by_arm["full_n512"]
    direct_translation = 26.689451217651367
    compatibility_translation = 26.68945232023151
    rows[0]["geometry"]["translation_l2_px"] = direct_translation
    registration = rows[0]["geometry"]["pairingnet_style_registration"]
    registration["translation_l2_px"] = compatibility_translation
    registration["normalized_translation_error"] = compatibility_translation / 20.0
    outcomes = rows[0]["geometry"]["assembly_edge"][
        "true_positive_by_tolerance"
    ]
    outcomes.update({"at_2": False, "at_5": False, "at_8": False, "at_10": False})
    assert MODULE._same_translation_metric(
        compatibility_translation, direct_translation
    )
    score_path = receipt_path.parent / "full_n512" / "pair_scores.jsonl"
    _jsonl(score_path, rows)
    result = next(
        row for row in receipt["arm_results"] if row["arm"] == "full_n512"
    )
    result["pair_scores_sha256"] = _sha(score_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    loaded = MODULE.load_sealed(receipt_path, compatibility_mode=True)
    assert len(loaded.rows) == 8


def test_sealed_rejects_translation_difference_beyond_float32_boundary(
    tmp_path: Path,
):
    receipt_path, receipt, rows_by_arm = _sealed_fixture(tmp_path)
    rows = rows_by_arm["full_n512"]
    registration = rows[0]["geometry"]["pairingnet_style_registration"]
    registration["translation_l2_px"] = 1.001
    registration["normalized_translation_error"] = 1.001 / 20.0
    score_path = receipt_path.parent / "full_n512" / "pair_scores.jsonl"
    _jsonl(score_path, rows)
    result = next(
        row for row in receipt["arm_results"] if row["arm"] == "full_n512"
    )
    result["pair_scores_sha256"] = _sha(score_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(MODULE.PairedBootstrapError, match="translation metrics disagree"):
        MODULE.load_sealed(receipt_path, compatibility_mode=True)


def test_balanced_real_bootstraps_all_methods_and_keeps_strict_descriptive(
    tmp_path: Path,
):
    path, _ = _real_fixture(tmp_path, balanced=True, matched=True)
    result = MODULE.analyze_real(
        path, replicates=300, seed=7, compatibility_mode=True
    )

    assert result["methods"] == ["coarse_only", "full_n512", "matched_mm"]
    paired = result["paired_endpoint_pigeonhole_bootstrap"]["metrics"]["auprc"][
        "paired_deltas"
    ]
    assert "full_n512_minus_coarse_only" in paired
    assert "full_n512_minus_matched_mm" in paired
    assert (
        result["strict547_descriptive"]["role"] == "descriptive_only_no_inferential_CI"
    )
    assert "paired_endpoint_pigeonhole_bootstrap" not in result["strict547_descriptive"]
    assert result["geometry"]["status"] == (
        "not_read_or_reported_by_this_target_blind_pair_ranking_artifact"
    )
    assert result["geometry"]["ground_truth_availability_claim_made"] is False
    assert "independent post-prediction evaluator" in result["geometry"]["reason"]


def test_real_loader_auto_discovers_converged_and_same_exposure_matched_methods(
    tmp_path: Path,
):
    path, document = _real_fixture(tmp_path, balanced=True, matched=True)
    old_metadata = document["methods"].pop("matched_mm")
    del old_metadata
    for method, token, threshold_value in (
        ("matched_mm_converged", "e", 0.55),
        ("matched_mm_same_exposure_epoch5", "f", 0.45),
    ):
        checkpoint = token * 64
        threshold = _threshold(method, checkpoint)
        threshold["threshold"] = threshold_value
        document["methods"][method] = {
            "score_semantics": "historical_mm_probability",
            "winner": {
                "checkpoint_sha256": checkpoint,
                "validation_threshold": threshold,
            },
        }
    for row in document["pairs"]:
        old = row["methods"].pop("matched_mm")
        for method in (
            "matched_mm_converged",
            "matched_mm_same_exposure_epoch5",
        ):
            threshold = document["methods"][method]["winner"]["validation_threshold"][
                "threshold"
            ]
            row["methods"][method] = {
                "probability": old["probability"],
                "valid": old["valid"],
                "score_semantics": "historical_mm_probability",
                "decision_at_frozen_validation_threshold": (
                    old["probability"] >= threshold
                ),
            }
    path.write_text(json.dumps(document, sort_keys=True) + "\n")

    loaded = MODULE.load_real(path, compatibility_mode=True)
    assert loaded.methods == (
        "coarse_only",
        "full_n512",
        "matched_mm_converged",
        "matched_mm_same_exposure_epoch5",
    )
    assert loaded.validation_thresholds["matched_mm_converged"] == 0.55
    assert loaded.validation_thresholds["matched_mm_same_exposure_epoch5"] == 0.45


def test_real_loader_accepts_verified_single_forward_wrapper(tmp_path: Path):
    path, balanced = _real_fixture(tmp_path, balanced=True, matched=True)
    strict = json.loads(json.dumps(balanced))
    strict["status"] = MODULE.COMPATIBILITY_REAL_STRICT_STATUS
    strict["pairs"] = strict["pairs"][:6]
    strict["dataset"] = dict(strict["dataset"])
    strict["dataset"].update(
        {
            "pair_count": 6,
            "positive_count": 4,
            "negative_count": 2,
            "case_cluster_count": 3,
            "pair_order_sha256": hashlib.sha256(
                _canonical(["r{}".format(index) for index in range(6)])
            ).hexdigest(),
        }
    )
    for key in (
        "population",
        "strict_prefix_count",
        "constructed_count",
        "strict_prefix_preserved_exactly",
    ):
        strict["dataset"].pop(key, None)
    strict.pop("construction_receipt", None)
    strict.pop("negative_semantics", None)
    wrapper = {
        "schema_version": "rachel-n512-real-external/1.0",
        "status": MODULE.COMPATIBILITY_REAL_COMBINED_STATUS,
        "forward_contract": {
            "forward_pair_count_per_arm": 8,
            "strict_547_derived_from_exact_prediction_prefix": True,
            "strict_pairs_forwarded_twice": False,
            "methods": list(balanced["methods"]),
            "all_methods_share_exact_strict_prediction_prefix": True,
        },
        "strict_547": strict,
        "balanced_1016": balanced,
    }
    path.write_text(json.dumps(wrapper, sort_keys=True) + "\n")
    loaded = MODULE.load_real(path, compatibility_mode=True)
    assert loaded.source_kind == "real_balanced1016"
    assert len(loaded.rows) == 8


def test_stats_rejects_matched_without_frozen_source_provenance(tmp_path: Path):
    receipt_path, receipt, _ = _sealed_fixture(tmp_path)
    receipt["arm_results"].append(dict(receipt["arm_results"][0], arm="matched_mm"))
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    with pytest.raises(MODULE.PairedBootstrapError, match="matched-MM source"):
        MODULE.load_sealed(receipt_path, compatibility_mode=True)


def test_stats_rejects_nonformal_population_even_if_internally_consistent(
    tmp_path: Path, monkeypatch
):
    receipt_path, _, _ = _sealed_fixture(tmp_path)
    monkeypatch.setattr(MODULE, "EXPECTED_SYNTHETIC_COUNT", 3000)
    with pytest.raises(MODULE.PairedBootstrapError, match="frozen formal 3000"):
        MODULE.load_sealed(receipt_path, compatibility_mode=True)


def test_strict_real_is_descriptive_only(tmp_path: Path):
    path, _ = _real_fixture(tmp_path, balanced=False, matched=False)
    result = MODULE.analyze_real(
        path, replicates=20, seed=3, compatibility_mode=True
    )
    assert result["inference"]["status"] == "not_run_by_design"
    assert "paired_endpoint_pigeonhole_bootstrap" not in result


def test_real_rejects_false_common_valid_flag(tmp_path: Path):
    path, document = _real_fixture(tmp_path, balanced=True, matched=False)
    document["pairs"][0]["methods"]["full_n512"]["valid"] = False
    document["pairs"][0]["methods"]["full_n512"][
        "decision_at_frozen_validation_threshold"
    ] = None
    path.write_text(json.dumps(document, sort_keys=True) + "\n")
    with pytest.raises(MODULE.PairedBootstrapError, match="common-valid flag differs"):
        MODULE.load_real(path, compatibility_mode=True)


def test_real_rejects_stratum_disagreement_inside_method_row(tmp_path: Path):
    path, document = _real_fixture(tmp_path, balanced=True, matched=True)
    document["pairs"][6]["methods"]["matched_mm"]["stratum"] = (
        "strict_manifest_negative"
    )
    path.write_text(json.dumps(document, sort_keys=True) + "\n")
    with pytest.raises(
        MODULE.PairedBootstrapError, match="stratum differs across methods"
    ):
        MODULE.load_real(path, compatibility_mode=True)


def test_balanced_real_binds_constructed_receipt_to_result_rows(tmp_path: Path):
    path, document = _real_fixture(tmp_path, balanced=True, matched=False)
    document["pairs"][6]["pair_id"] = "changed-constructed-pair"
    document["dataset"]["pair_order_sha256"] = hashlib.sha256(
        _canonical([row["pair_id"] for row in document["pairs"]])
    ).hexdigest()
    path.write_text(json.dumps(document, sort_keys=True) + "\n")

    with pytest.raises(
        MODULE.PairedBootstrapError,
        match="differ from constructed receipt",
    ):
        MODULE.load_real(path, compatibility_mode=True)


def test_balanced_real_binds_strict_prefix_to_frozen_pair_order(tmp_path: Path):
    path, document = _real_fixture(tmp_path, balanced=True, matched=False)
    document["pairs"][0], document["pairs"][2] = (
        document["pairs"][2],
        document["pairs"][0],
    )
    document["dataset"]["pair_order_sha256"] = hashlib.sha256(
        _canonical([row["pair_id"] for row in document["pairs"]])
    ).hexdigest()
    path.write_text(json.dumps(document, sort_keys=True) + "\n")

    with pytest.raises(MODULE.PairedBootstrapError, match="dataset receipt differs"):
        MODULE.load_real(path, compatibility_mode=True)


def test_sealed_rejects_pair_score_byte_tampering(tmp_path: Path):
    receipt_path, _, _ = _sealed_fixture(tmp_path)
    score_path = receipt_path.parent / "coarse_only" / "pair_scores.jsonl"
    score_path.write_bytes(score_path.read_bytes() + b"\n")
    with pytest.raises(MODULE.PairedBootstrapError, match="SHA-256 differs"):
        MODULE.load_sealed(receipt_path, compatibility_mode=True)


@pytest.mark.parametrize("tamper", ["assembly", "nte"])
def test_sealed_rejects_resigned_direct_geometry_tampering(
    tmp_path: Path, tamper: str
):
    receipt_path, receipt, rows_by_arm = _sealed_fixture(tmp_path)
    rows = rows_by_arm["full_n512"]
    if tamper == "assembly":
        value = rows[0]["geometry"]["assembly_edge"][
            "true_positive_by_tolerance"
        ]
        value["at_5"] = not value["at_5"]
    else:
        rows[0]["geometry"]["pairingnet_style_registration"][
            "normalized_translation_error"
        ] = 0.123
    score_path = receipt_path.parent / "full_n512" / "pair_scores.jsonl"
    _jsonl(score_path, rows)
    result = next(
        row for row in receipt["arm_results"] if row["arm"] == "full_n512"
    )
    result["pair_scores_sha256"] = _sha(score_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    with pytest.raises(
        MODULE.PairedBootstrapError,
        match="assembly-edge outcome differs|normalized translation differs",
    ):
        MODULE.load_sealed(receipt_path, compatibility_mode=True)


def test_assembly_edge_uses_translation_l2_not_pairingnet_ermse_and_is_inclusive(
    tmp_path: Path,
):
    receipt_path, receipt, rows_by_arm = _sealed_fixture(tmp_path)
    rows = rows_by_arm["full_n512"]
    first = rows[0]["geometry"]
    first_registration = first["pairingnet_style_registration"]
    first["translation_l2_px"] = 6.0
    first_registration["translation_l2_px"] = 6.0
    first_registration["normalized_translation_error"] = 0.3
    first_registration["e_rmse"] = 6.0**0.5
    first_registration["registration_recall_lt4_success"] = True
    first["assembly_edge"]["true_positive_by_tolerance"] = {
        "at_2": False,
        "at_5": False,
        "at_8": True,
        "at_10": True,
        "at_100": True,
    }

    boundary = rows[2]["geometry"]
    boundary_registration = boundary["pairingnet_style_registration"]
    boundary["translation_l2_px"] = 5.0
    boundary_registration["translation_l2_px"] = 5.0
    boundary_registration["normalized_translation_error"] = 0.25
    boundary_registration["e_rmse"] = 9.0
    boundary_registration["registration_recall_lt4_success"] = False
    boundary["assembly_edge"]["true_positive_by_tolerance"] = {
        "at_2": False,
        "at_5": True,
        "at_8": True,
        "at_10": True,
        "at_100": True,
    }

    score_path = receipt_path.parent / "full_n512" / "pair_scores.jsonl"
    _jsonl(score_path, rows)
    result = next(
        row for row in receipt["arm_results"] if row["arm"] == "full_n512"
    )
    result["pair_scores_sha256"] = _sha(score_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    loaded = MODULE.load_sealed(receipt_path, compatibility_mode=True)
    loaded_by_id = {row.pair_id: row for row in loaded.rows}
    assert loaded_by_id["p0"].geometry["assembly_edge"][
        "true_positive_by_tolerance"
    ]["at_5"] is False
    assert loaded_by_id["p2"].geometry["assembly_edge"][
        "true_positive_by_tolerance"
    ]["at_5"] is True


def test_pairingnet_compatibility_bootstrap_includes_invalid_positive_identity(
    tmp_path: Path,
):
    receipt_path, receipt, rows_by_arm = _sealed_fixture(tmp_path)
    rows = rows_by_arm["full_n512"]
    row = rows[0]
    row["scores"]["fused"]["valid"] = False
    row["decision"]["valid"] = False
    row["decision"]["predicted_label"] = None
    geometry = row["geometry"]
    geometry["decision_valid"] = False
    geometry["translation_l2_px"] = None
    geometry["correspondence"].update(
        {
            "strict_true_positive": 0,
            "strict_predicted_count": 0,
            "mutual_top1_true_positive": 0,
            "mutual_top1_predicted_count": 0,
            "dustbin_correct": 0,
        }
    )
    registration = geometry["pairingnet_style_registration"]
    registration.update(
        {
            "prediction_valid": False,
            "identity_fallback_used": True,
            "e_rmse": 0.0,
            "registration_recall_lt4_success": True,
            "symmetric_hausdorff_px": 0.0,
            "translation_l2_px": 0.0,
            "normalized_translation_error": 0.0,
            "source_contour_area_px2": 10.0,
            "target_contour_area_px2": 10.0,
        }
    )
    geometry["assembly_edge"]["predicted_edge"] = False
    geometry["assembly_edge"]["true_positive_by_tolerance"] = {
        "at_2": False,
        "at_5": False,
        "at_8": False,
        "at_10": False,
        "at_100": False,
    }

    score_path = receipt_path.parent / "full_n512" / "pair_scores.jsonl"
    _jsonl(score_path, rows)
    result = next(
        item for item in receipt["arm_results"] if item["arm"] == "full_n512"
    )
    result["pair_scores_sha256"] = _sha(score_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    report = MODULE.analyze_synthetic(
        receipt_path, replicates=40, seed=29, compatibility_mode=True
    )
    registration_report = report["synthetic_geometry"]["full_n512"][
        "pairingnet_style_registration"
    ]
    assert registration_report["eligible_positive_count"] == 4
    assert registration_report["evaluated_positive_count"] == 4
    assert registration_report["valid_pose_count"] == 3
    assert registration_report["identity_fallback_count"] == 1
    assert registration_report["registration_recall_e_rmse_lt4"][
        "point_estimate"
    ] == pytest.approx(0.5)
    assert registration_report["mean_e_rmse"]["point_estimate"] == pytest.approx(
        3.75
    )
    assert registration_report["mean_symmetric_hausdorff_px"][
        "point_estimate"
    ] == pytest.approx(3.75)
    assert registration_report["mean_normalized_translation_error"][
        "point_estimate"
    ] == pytest.approx(0.1875)


def test_cli_entrypoint_atomically_writes_portable_json(tmp_path: Path):
    receipt_path, _, _ = _sealed_fixture(tmp_path)
    output = tmp_path / "analysis" / "result.json"
    assert (
        MODULE.main(
            [
                "--synthetic-receipt",
                str(receipt_path),
                "--output",
                str(output),
                "--replicates",
                "20",
                "--seed",
                "9",
                "--compatibility-non-formal",
            ]
        )
        == 0
    )
    restored = json.loads(output.read_text())
    assert restored["status"] == "complete_non_formal_compatibility_analysis"
    assert restored["formal_evaluation"] is False
    assert restored["protocol"]["replicates"] == 20
    assert not list(output.parent.glob(".result.json.tmp-*"))
    with pytest.raises(MODULE.PairedBootstrapError, match="already exists"):
        MODULE.main(
            [
                "--synthetic-receipt",
                str(receipt_path),
                "--output",
                str(output),
                "--replicates",
                "2",
                "--compatibility-non-formal",
            ]
        )


def test_real_cli_temp_name_cannot_alias_or_overwrite_input(tmp_path: Path):
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    _, document = _real_fixture(fixture_root, balanced=False, matched=False)
    directory = tmp_path / "analysis"
    directory.mkdir()
    source = directory / ".result.json.tmp"
    source.write_text(json.dumps(document, sort_keys=True) + "\n")
    before = source.read_bytes()
    output = directory / "result.json"

    assert (
        MODULE.main(
            [
                "--real-json",
                str(source),
                "--output",
                str(output),
                "--replicates",
                "2",
                "--compatibility-non-formal",
            ]
        )
        == 0
    )
    assert source.read_bytes() == before
    assert output.is_file()

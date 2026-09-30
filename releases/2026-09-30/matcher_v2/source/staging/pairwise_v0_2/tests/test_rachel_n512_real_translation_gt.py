from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np
from PIL import Image
import pytest

from staging.pairwise_v0_2.baselines.rachel_matched_mm_evaluation import (
    MATCHED_METHODS,
)
import staging.pairwise_v0_2.baselines.rachel_n512_real_translation_gt as pose
from staging.pairwise_v0_2.baselines.rachel_n512_real_translation_gt import (
    RachelRealTranslationGTError,
    RealTranslationGTAuthority,
    evaluate_real_translation_gt_postprediction,
    freeze_real_prediction_result,
)
from staging.pairwise_v0_2.pairwise_data import real_dunhuang_representations as reps


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_rgba(path: Path, mask: np.ndarray, color: int) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = mask.astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def _fixture_authority(tmp_path: Path):
    input_root = tmp_path / "inputs"
    recorded_main = input_root / "recorded-main"
    recorded_supp = input_root / "recorded-supp"
    actual_main = input_root / "actual-main"
    actual_supp = input_root / "actual-supp"
    actual_supp.mkdir(parents=True)
    case_uid = "fixture-case"
    masks = [np.ones((10, 10), dtype=np.bool_) for _ in range(3)]
    boxes = [(0, 0, 10, 10), (10, 0, 20, 10), (20, 0, 30, 10)]
    fragments = []
    recorded_paths = []
    for index, (mask, bbox) in enumerate(zip(masks, boxes), start=1):
        relative = Path("Ground Truth Simple") / "1" / "{}.png".format(index)
        _write_rgba(actual_main / relative, mask, color=20 * index)
        recorded_paths.append(str(recorded_main / relative))
        fragments.append(
            {
                "fragment_id": index,
                "has_alpha": True,
                "alpha_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "bbox_xyxy": list(bbox),
                "size_wh": [10, 10],
            }
        )
    pair_values = ((1, 2, "positive"), (1, 3, "negative"), (2, 3, "positive"))
    pair_labels = [
        {
            "pair_uid": "fixture-source-pair-{}".format(index),
            "fragment_a": first,
            "fragment_b": second,
            "label": label,
            "label_source": "gt_numeric_bbox_plus_fragment_alpha",
            "reason": "fixture_geometry_label",
        }
        for index, (first, second, label) in enumerate(pair_values, start=1)
    ]
    case = {
        "case_uid": case_uid,
        "canonical_collection": "main",
        "canonical_category": "ground_truth_simple",
        "observed_categories": ["ground_truth_simple"],
        "disposition": "eligible",
        "numeric_metadata": {"canvas_wh": [40, 20]},
        "fragments": fragments,
        "pair_labels": pair_labels,
        "occurrences": [
            {
                "occurrence_uid": "fixture-occurrence",
                "collection": "main",
                "category": "ground_truth_simple",
            }
        ],
    }
    manifest_without_sha = {
        "schema_version": "pairwise-v0.2-real-external-test/0.1",
        "cases": [case],
    }
    manifest_sha = _canonical_sha(manifest_without_sha)
    manifest = {**manifest_without_sha, "manifest_sha256": manifest_sha}
    receipt = {
        "schema_version": "pairwise-v0.2-real-external-test-local-receipt/0.1",
        "portable_manifest_sha256": manifest_sha,
        "dataset_roots": {
            "main": str(recorded_main),
            "supp": str(recorded_supp),
        },
        "cases": {
            case_uid: {
                "occurrences": [
                    {
                        "occurrence_uid": "fixture-occurrence",
                        "collection": "main",
                        "category": "ground_truth_simple",
                        "fragment_paths": recorded_paths,
                    }
                ]
            }
        },
    }
    manifest_path = input_root / "authority" / "manifest.json"
    receipt_path = input_root / "authority" / "paths.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    pair_ids = tuple(
        reps._existing_evaluation_pair_id(case_uid, first, second)
        for first, second, _ in pair_values
    )
    authority = RealTranslationGTAuthority(
        manifest_sha256=manifest_sha,
        strict_pair_order_sha256=_canonical_sha(pair_ids),
        case_count=1,
        fragment_count=3,
        strict_pair_count=3,
        strict_positive_count=2,
        strict_negative_count=1,
        balanced_pair_count=3,
        canvas_size=32,
        contour_cap=8,
        require_matched_controls=True,
        enforce_formal_combined_gate=False,
    )
    return {
        "manifest": manifest_path,
        "receipt": receipt_path,
        "main_root": actual_main,
        "supp_root": actual_supp,
        "authority": authority,
        "pair_ids": pair_ids,
    }


def _method_summary(checkpoint: str, threshold: float = 0.5):
    return {
        "winner": {
            "epoch": 1,
            "checkpoint_sha256": checkpoint,
            "validation_threshold": {
                "threshold": threshold,
                "source_split": "val",
                "fit_method": "maximize_cluster_balanced_f1",
                "checkpoint_sha256": checkpoint,
            },
        }
    }


def _history_disclosure():
    return {
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
    }


def _prediction_result(path: Path, fixture, *, prefix_tamper: bool = False) -> Path:
    methods = {
        "coarse_only": _method_summary("1" * 64),
        "full_n512": _method_summary("2" * 64),
        MATCHED_METHODS[0]: _method_summary("3" * 64),
        MATCHED_METHODS[1]: _method_summary("4" * 64),
    }
    for method in MATCHED_METHODS:
        methods[method]["score_semantics"] = "historical_mm_probability"
    labels = (True, False, True)
    translations = ((0.0, -8.0), (0.0, -16.0), (0.0, 0.0))
    full_valid = (True, True, False)
    probabilities = (0.9, 0.2, 0.4)
    rows = []
    for index, pair_id in enumerate(fixture["pair_ids"]):
        per_method = {}
        for name in methods:
            probability = probabilities[index]
            valid = full_valid[index] if name == "full_n512" else True
            row = {
                "probability": probability,
                "valid": valid,
                "decision_at_frozen_validation_threshold": (
                    probability >= 0.5 if valid else None
                ),
            }
            if name == "full_n512":
                row["translation_hat_rc_unsupervised"] = list(translations[index])
            if name in MATCHED_METHODS:
                row["score_semantics"] = "historical_mm_probability"
            per_method[name] = row
        rows.append(
            {
                "pair_id": pair_id,
                "label": labels[index],
                "case_cluster": "fixture-case",
                "source_case_uids": ["fixture-case"],
                "coarse_full_common_valid": full_valid[index],
                "methods": per_method,
            }
        )
    balanced_rows = json.loads(json.dumps(rows))
    if prefix_tamper:
        balanced_rows[0]["methods"]["full_n512"]["probability"] = 0.1
    protocol = {
        "formal_evaluation": False,
        "compatibility_mode": True,
        "formal_exact_six_frozen_before_real_open": False,
        "formal_method_inventory": None,
        "winner_and_validation_threshold_frozen_before_real_open": True,
        "case_common_parent_canvas_scale": True,
        "tight_crop_then_centerpad": True,
        "formal_config_verified": True,
        "both_arms_validation_plateau_verified": True,
        "all_requested_winners_and_thresholds_frozen_before_current_real_open": True,
        "per_fragment_independent_resize": False,
        "rgb_used": False,
        "bbox_or_gt_canvas_origin_exposed_to_model": False,
        "real_correspondence_or_translation_gt_read": False,
        "positive_direction_derived_or_bbox_read": False,
        "canvas_size": 32,
        "contour_cap": 8,
        "pixel_source": "PNG_alpha_ge_128_only",
        "primary_metrics": "threshold_free_AUROC_AUPRC",
        "thresholded_metrics": "secondary_frozen_validation_threshold_only",
        "evaluation_history_disclosure": _history_disclosure(),
    }
    dataset = {
        "manifest_sha256": fixture["authority"].manifest_sha256,
        "pair_count": 3,
        "positive_count": 2,
        "negative_count": 1,
        "case_cluster_count": 1,
        "pair_order_sha256": fixture["authority"].strict_pair_order_sha256,
    }
    source_training = {
        "receipt_sha256": "5" * 64,
        "status_at_open": "complete_train_validation_only",
        "both_arms_validation_plateau_verified": True,
        "formal_config_verified": True,
    }
    source_matched = {
        "receipt_sha256": "6" * 64,
        "status_at_open": "complete_train_validation_only",
        "converged_and_epoch5_winners_frozen_before_current_real_open": True,
        "formal_config_verified": True,
    }
    strict = {
        "status": pose.real_eval.COMPATIBILITY_STRICT_STATUS,
        "dataset": dataset,
        "protocol": protocol,
        "methods": methods,
        "pairs": rows,
        "source_training_run": source_training,
        "source_matched_mm_training_run": source_matched,
    }
    balanced = {
        **strict,
        "status": pose.real_eval.COMPATIBILITY_BALANCED_STATUS,
        "dataset": {
            **dataset,
            "pair_count": 3,
            "strict_prefix_count": 3,
            "strict_prefix_preserved_exactly": True,
        },
        "pairs": balanced_rows,
    }
    result = {
        "schema_version": "rachel-n512-real-external/1.0",
        "status": pose.real_eval.COMPATIBILITY_COMBINED_STATUS,
        "forward_contract": {
            "forward_pair_count_per_arm": 3,
            "strict_547_derived_from_exact_prediction_prefix": True,
            "strict_pairs_forwarded_twice": False,
            "all_methods_share_exact_strict_prediction_prefix": True,
            "methods": list(pose.EXPECTED_METHODS),
        },
        "source_training_run": source_training,
        "source_matched_mm_training_run": source_matched,
        "strict_547": strict,
        "balanced_1016": balanced,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def _add_same_data_benchmarks(path: Path) -> Path:
    document = json.loads(path.read_text(encoding="utf-8"))
    document["status"] = pose.real_eval.FORMAL_COMBINED_STATUS
    translations = ((0.0, -8.0), (0.0, -16.0), (0.0, -8.0))
    probabilities = (0.9, 0.2, 0.9)
    provenance = {}
    for index, method in enumerate(pose.benchmark_adapter.BENCHMARK_METHODS):
        checkpoint = ("7" if index == 0 else "8") * 64
        threshold = {
            "threshold": 0.5,
            "source_split": "val",
            "fit_method": "maximize_cluster_balanced_f1",
            "checkpoint_sha256": checkpoint,
        }
        metadata = {
            "score_semantics": "adapted_pair_probability",
            "winner": {
                "method_id": "fixture-" + method,
                "checkpoint_sha256_by_stage": {"winner": checkpoint},
                "freeze_authority_sha256": ("9" if index == 0 else "a") * 64,
                "validation_threshold": threshold,
                "validation_threshold_sha256": "b" * 64,
            },
            "native_cm_fm_se_or_ga_claimed": False,
        }
        provenance[method] = {
            "method_key": method,
            "all_winners_and_validation_threshold_frozen": True,
            "sealed_synthetic_accessed_during_freeze": False,
            "real_data_accessed_during_freeze": False,
        }
        for population in ("strict_547", "balanced_1016"):
            document[population]["methods"][method] = metadata
            protocol = document[population]["protocol"]
            protocol.update(
                {
                    "formal_evaluation": True,
                    "compatibility_mode": False,
                    "formal_exact_six_frozen_before_real_open": True,
                    "formal_method_inventory": list(
                        pose.COMPLETE_BENCHMARK_METHODS
                    ),
                    "balanced1016_single_forward_required": True,
                    "same_data_benchmark_winners_and_validation_thresholds_frozen_before_current_real_open": True,
                    "same_data_benchmark_mask_only": True,
                    "same_data_benchmark_upright_translation_only_adaptation": True,
                    "same_data_benchmark_exact_reproduction_claimed": False,
                    "shreddingnet_balanced_selected_list_claimed_as_native_CM_FM_SE_or_GA": False,
                }
            )
            for row_index, row in enumerate(document[population]["pairs"]):
                row["methods"][method] = {
                    "probability": probabilities[row_index],
                    "valid": True,
                    "score_semantics": "adapted_pair_probability",
                    "decision_at_frozen_validation_threshold": (
                        probabilities[row_index] >= 0.5
                    ),
                    "translation_hat_rc_unsupervised": list(
                        translations[row_index]
                    ),
                }
    source = {
        "training_manifest_evidence": {"exact_manifest_bytes": True},
        "methods": provenance,
        "all_winners_and_validation_thresholds_frozen_before_current_real_open": True,
    }
    for population in ("strict_547", "balanced_1016"):
        document[population]["source_same_data_benchmark_training_runs"] = source
    document["strict_547"]["status"] = (
        "complete_strict_547_target_blind_external_test"
    )
    document["balanced_1016"]["status"] = (
        "complete_balanced_1016_target_blind_external_test"
    )
    document["source_same_data_benchmark_training_runs"] = source
    document["forward_contract"]["methods"] = list(
        pose.COMPLETE_BENCHMARK_METHODS
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_complete_postprediction_evaluation_recovers_sign_and_metrics(tmp_path):
    fixture = _fixture_authority(tmp_path)
    predictions = _prediction_result(tmp_path / "predictions" / "real.json", fixture)
    output = tmp_path / "outputs" / "pose.json"
    result = evaluate_real_translation_gt_postprediction(
        predictions,
        fixture["manifest"],
        fixture["receipt"],
        output_path=output,
        main_root=fixture["main_root"],
        supp_root=fixture["supp_root"],
        bootstrap_repetitions=100,
        authority=fixture["authority"],
        compatibility_mode=True,
    )

    assert output.is_file()
    assert result["status"] == (
        "complete_non_formal_compatibility_real_translation_gt"
    )
    assert result["formal_evaluation"] is False
    rows = result["positive_pairs"]
    assert len(rows) == 2
    assert rows[0]["translation_gt_a_to_b_rc"] == [0.0, -8.0]
    assert rows[0]["translation_gt_a_to_b_xy_cartesian"] == [-8.0, -0.0]
    assert rows[0]["placement_shift_b_into_a_rc"] == [-0.0, 8.0]
    assert rows[0]["full_n512"]["translation_l2_error_px"] == 0.0
    assert rows[1]["full_n512"]["translation_prediction_valid"] is False
    metrics = result["method_metrics"]["full_n512"]["case_bootstrap"]["metrics"]
    assert metrics["valid_translation_prediction_fraction"]["estimate"] == 0.5
    assert metrics["median_l2_px"]["estimate"] == 0.0
    assert metrics["recall_at_2px"]["estimate"] == 0.5
    assert (
        metrics["joint_frozen_threshold_and_translation_recall_at_2px"][
            "estimate"
        ]
        == 0.5
    )
    assert result["method_metrics"]["coarse_only"]["status"] == "not_applicable"
    assert result["method_metrics"][MATCHED_METHODS[0]]["status"] == "not_applicable"
    assert all(
        row["seam_and_correspondence_qa"]["status"] == "passed" for row in rows
    )
    disclosure = result["protocol"]["evaluation_history_disclosure"]
    assert disclosure["prior_epoch5_synthetic_test_human_visible_before_continuation"]
    assert not disclosure[
        "prior_epoch5_synthetic_test_used_by_automated_checkpoint_selection"
    ]
    assert not disclosure[
        "prior_epoch5_synthetic_test_used_by_automated_threshold_fitting"
    ]
    assert result["protocol"]["model_forward_or_gpu_work_performed_here"] is False
    endpoint_authority = result["label_authority_opened_after_prediction_freeze"][
        "ordered_endpoint_authority"
    ]
    assert endpoint_authority["translation_orientation"] == (
        "fragment_a_to_fragment_b"
    )
    assert endpoint_authority["rows"][0]["fragment_a_token"].endswith(
        "/fragment/1"
    )
    assert endpoint_authority["rows"][0]["fragment_b_token"].endswith(
        "/fragment/2"
    )
    assert endpoint_authority["ordered_pair_endpoint_sha256"] == _canonical_sha(
        endpoint_authority["rows"]
    )
    restored = json.loads(output.read_text(encoding="utf-8"))
    content_sha = restored.pop("content_sha256")
    assert content_sha == _canonical_sha(restored)


def test_complete_six_method_direct_metrics_are_common_and_non_native(tmp_path):
    fixture = _fixture_authority(tmp_path)
    predictions = _add_same_data_benchmarks(
        _prediction_result(tmp_path / "predictions" / "real-six.json", fixture)
    )
    result = evaluate_real_translation_gt_postprediction(
        predictions,
        fixture["manifest"],
        fixture["receipt"],
        output_path=tmp_path / "outputs" / "pose-six.json",
        main_root=fixture["main_root"],
        supp_root=fixture["supp_root"],
        bootstrap_repetitions=100,
        authority=fixture["authority"],
    )
    pairing = result["method_metrics"][
        pose.benchmark_adapter.PAIRINGNET_METHOD_KEY
    ]
    assert pairing["point_estimates"]["valid_translation_prediction_fraction"] == 1.0
    assert pairing["point_estimates"]["recall_at_2px"] == 1.0
    strict_edge = pairing["assembly_edge_strict_547"]["by_tolerance"]["at_2px"]
    assert strict_edge["precision"] == strict_edge["recall"] == 1.0
    assert pairing["pairingnet_style_registration"]["rr_lt4"] == 1.0
    assert pairing["pairingnet_style_registration"][
        "mean_normalized_translation_error"
    ] == 0.0
    assert pairing["correspondence"]["status"] == "not_applicable"
    shred = result["method_metrics"][
        pose.benchmark_adapter.SHREDDINGNET_METHOD_KEY
    ]
    assert shred["shreddingnet_native_CM_FM_SE"]["status"] == "not_reported"
    assert result["direct_pairwise_metric_contract"][
        "balanced_selected_list_diagnostics_claimed_as_native_shreddingnet_metrics"
    ] is False
    assert set(result["positive_pairs"][0]) >= set(
        pose.benchmark_adapter.BENCHMARK_METHODS
    )


def test_translation_gt_default_rejects_legacy_four_method_source(tmp_path):
    fixture = _fixture_authority(tmp_path)
    predictions = _prediction_result(
        tmp_path / "predictions" / "legacy-four.json", fixture
    )
    with pytest.raises(
        RachelRealTranslationGTError,
        match="formal/compatibility status differs",
    ):
        freeze_real_prediction_result(predictions, authority=fixture["authority"])


def test_translation_gt_formal_status_rejects_non_exact_six_inventory(tmp_path):
    fixture = _fixture_authority(tmp_path)
    predictions = _prediction_result(
        tmp_path / "predictions" / "false-formal-four.json", fixture
    )
    document = json.loads(predictions.read_text(encoding="utf-8"))
    document["status"] = pose.real_eval.FORMAL_COMBINED_STATUS
    for population, status in (
        ("strict_547", "complete_strict_547_target_blind_external_test"),
        ("balanced_1016", "complete_balanced_1016_target_blind_external_test"),
    ):
        document[population]["status"] = status
        document[population]["protocol"].update(
            {
                "formal_evaluation": True,
                "compatibility_mode": False,
                "formal_exact_six_frozen_before_real_open": True,
                "formal_method_inventory": list(
                    pose.COMPLETE_BENCHMARK_METHODS
                ),
                "balanced1016_single_forward_required": True,
            }
        )
    predictions.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        RachelRealTranslationGTError,
        match="formal strict result must contain exact-six",
    ):
        freeze_real_prediction_result(predictions, authority=fixture["authority"])


def test_translation_gt_compatibility_cannot_consume_formal_six_receipt(tmp_path):
    fixture = _fixture_authority(tmp_path)
    predictions = _add_same_data_benchmarks(
        _prediction_result(tmp_path / "predictions" / "formal-six.json", fixture)
    )
    with pytest.raises(
        RachelRealTranslationGTError,
        match="formal/compatibility status differs",
    ):
        freeze_real_prediction_result(
            predictions,
            authority=fixture["authority"],
            compatibility_mode=True,
        )


def test_translation_gt_cli_defaults_to_formal_and_compatibility_is_explicit(
    tmp_path,
):
    arguments = [
        "--pair-only-result",
        str(tmp_path / "predictions.json"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--local-path-receipt",
        str(tmp_path / "paths.json"),
        "--output",
        str(tmp_path / "result.json"),
    ]
    assert pose._parser().parse_args(arguments).compatibility_non_formal is False
    assert (
        pose._parser()
        .parse_args(arguments + ["--compatibility-non-formal"])
        .compatibility_non_formal
        is True
    )


def test_swap_exactly_negates_gt_and_assembly_shift(tmp_path):
    fixture = _fixture_authority(tmp_path)
    spec = reps.load_real_external_test_spec(
        fixture["manifest"],
        fixture["receipt"],
        main_root=fixture["main_root"],
        supp_root=fixture["supp_root"],
        expected_case_count=1,
        expected_fragment_count=3,
        expected_pair_count=3,
        expected_positive_count=2,
        expected_negative_count=1,
        derive_positive_direction=False,
    )
    first_spec, second_spec = spec.fragments[:2]
    masks = [pose.real_eval._load_strict_alpha(value) for value in spec.fragments[:2]]
    first = pose._build_fragment_gt(
        first_spec,
        mask=masks[0],
        bbox_xyxy=(0, 0, 10, 10),
        canvas_wh=(40, 20),
        canvas_size=32,
        contour_cap=8,
    )
    second = pose._build_fragment_gt(
        second_spec,
        mask=masks[1],
        bbox_xyxy=(10, 0, 20, 10),
        canvas_wh=(40, 20),
        canvas_size=32,
        contour_cap=8,
    )
    pair_authority = {
        "label_source": "gt_numeric_bbox_plus_fragment_alpha",
        "geometry_diagnostic_label": "positive",
    }
    forward = pose._positive_gt(first, second, "forward", pair_authority)
    reverse = pose._positive_gt(second, first, "reverse", pair_authority)
    assert np.allclose(
        reverse.translation_a_to_b_rc,
        -np.asarray(forward.translation_a_to_b_rc),
    )
    assert np.allclose(
        forward.placement_shift_b_into_a_rc,
        -np.asarray(forward.translation_a_to_b_rc),
    )


def test_curated_two_fragment_positive_keeps_label_when_alpha_gap_is_not_strong(
    tmp_path,
):
    fixture = _fixture_authority(tmp_path)
    spec = reps.load_real_external_test_spec(
        fixture["manifest"],
        fixture["receipt"],
        main_root=fixture["main_root"],
        supp_root=fixture["supp_root"],
        expected_case_count=1,
        expected_fragment_count=3,
        expected_pair_count=3,
        expected_positive_count=2,
        expected_negative_count=1,
        derive_positive_direction=False,
    )
    first_spec, second_spec = spec.fragments[:2]
    first_mask = pose.real_eval._load_strict_alpha(first_spec)
    second_mask = pose.real_eval._load_strict_alpha(second_spec)
    first = pose._build_fragment_gt(
        first_spec,
        mask=first_mask,
        bbox_xyxy=(0, 0, 10, 10),
        canvas_wh=(40, 20),
        canvas_size=32,
        contour_cap=8,
    )
    second = pose._build_fragment_gt(
        second_spec,
        mask=second_mask,
        bbox_xyxy=(25, 0, 35, 10),
        canvas_wh=(40, 20),
        canvas_size=32,
        contour_cap=8,
    )
    gt = pose._positive_gt(
        first,
        second,
        "curated",
        {
            "label_source": "curated_two_fragment_conjunction_category",
            "geometry_diagnostic_label": "negative",
        },
    )
    assert gt.translation_a_to_b_rc == (0.0, -20.0)
    assert not gt.seam_qa["strong_geometry_contact_contract_required"]
    assert not gt.seam_qa["strong_geometry_contact_contract_passed"]
    assert gt.seam_qa["model_frame_correspondence_residual_px"]["max"] > 2.0


def test_freeze_rejects_balanced_prefix_prediction_change(tmp_path):
    fixture = _fixture_authority(tmp_path)
    predictions = _prediction_result(
        tmp_path / "predictions" / "tampered.json", fixture, prefix_tamper=True
    )
    with pytest.raises(RachelRealTranslationGTError, match="single-forward prefix"):
        freeze_real_prediction_result(
            predictions,
            authority=fixture["authority"],
            compatibility_mode=True,
        )


def test_existing_output_stops_before_any_gt_open(tmp_path, monkeypatch):
    fixture = _fixture_authority(tmp_path)
    predictions = _prediction_result(tmp_path / "predictions" / "real.json", fixture)
    output = tmp_path / "outputs" / "pose.json"
    output.parent.mkdir(parents=True)
    output.write_text("user file", encoding="utf-8")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("GT was opened")

    monkeypatch.setattr(pose, "_load_gt_after_prediction_freeze", forbidden)
    with pytest.raises(RachelRealTranslationGTError, match="overwrite"):
        evaluate_real_translation_gt_postprediction(
            predictions,
            fixture["manifest"],
            fixture["receipt"],
            output_path=output,
            main_root=fixture["main_root"],
            supp_root=fixture["supp_root"],
            bootstrap_repetitions=10,
            authority=fixture["authority"],
            compatibility_mode=True,
        )
    assert called is False
    assert output.read_text(encoding="utf-8") == "user file"


def test_prediction_freeze_occurs_before_gt_loader(tmp_path, monkeypatch):
    fixture = _fixture_authority(tmp_path)
    predictions = _prediction_result(tmp_path / "predictions" / "real.json", fixture)
    events = []
    original_freeze = pose.freeze_real_prediction_result
    original_gt = pose._load_gt_after_prediction_freeze

    def freeze(*args, **kwargs):
        events.append("freeze")
        return original_freeze(*args, **kwargs)

    def gt(*args, **kwargs):
        events.append("gt")
        return original_gt(*args, **kwargs)

    monkeypatch.setattr(pose, "freeze_real_prediction_result", freeze)
    monkeypatch.setattr(pose, "_load_gt_after_prediction_freeze", gt)
    evaluate_real_translation_gt_postprediction(
        predictions,
        fixture["manifest"],
        fixture["receipt"],
        output_path=tmp_path / "outputs" / "pose.json",
        main_root=fixture["main_root"],
        supp_root=fixture["supp_root"],
        bootstrap_repetitions=10,
        authority=fixture["authority"],
        compatibility_mode=True,
    )
    assert events == ["freeze", "gt"]


def test_tampered_declared_alpha_is_rejected_after_prediction_freeze(tmp_path):
    fixture = _fixture_authority(tmp_path)
    predictions = _prediction_result(tmp_path / "predictions" / "real.json", fixture)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256")
    manifest["cases"][0]["fragments"][0]["alpha_mask_sha256"] = "0" * 64
    new_sha = _canonical_sha(manifest)
    manifest["manifest_sha256"] = new_sha
    fixture["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    receipt["portable_manifest_sha256"] = new_sha
    fixture["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    changed_authority = RealTranslationGTAuthority(
        manifest_sha256=new_sha,
        strict_pair_order_sha256=fixture["authority"].strict_pair_order_sha256,
        case_count=1,
        fragment_count=3,
        strict_pair_count=3,
        strict_positive_count=2,
        strict_negative_count=1,
        balanced_pair_count=3,
        canvas_size=32,
        contour_cap=8,
        require_matched_controls=True,
        enforce_formal_combined_gate=False,
    )
    # Rebind only the source dataset digest so prediction freeze remains valid.
    source = json.loads(predictions.read_text(encoding="utf-8"))
    for name in ("strict_547", "balanced_1016"):
        source[name]["dataset"]["manifest_sha256"] = new_sha
    predictions.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(RachelRealTranslationGTError, match="alpha-mask SHA-256"):
        evaluate_real_translation_gt_postprediction(
            predictions,
            fixture["manifest"],
            fixture["receipt"],
            output_path=tmp_path / "outputs" / "pose.json",
            main_root=fixture["main_root"],
            supp_root=fixture["supp_root"],
            bootstrap_repetitions=10,
            authority=changed_authority,
            compatibility_mode=True,
        )


def test_case_bootstrap_is_deterministic_and_keeps_case_cluster(tmp_path):
    del tmp_path
    errors = np.asarray([0.0, 4.0, 9.0], dtype=np.float64)
    valid = np.asarray([True, True, True])
    accepted = np.asarray([True, False, True])
    first = pose._case_bootstrap(
        errors=errors,
        valid=valid,
        threshold_positive=accepted,
        cases=("case-a", "case-a", "case-b"),
        repetitions=200,
        seed="fixture-seed",
    )
    second = pose._case_bootstrap(
        errors=errors,
        valid=valid,
        threshold_positive=accepted,
        cases=("case-a", "case-a", "case-b"),
        repetitions=200,
        seed="fixture-seed",
    )
    assert first == second
    assert first["case_count"] == 2
    assert first["same_case_pairs_keep_endpoints_together"] is True


def test_chunked_boundary_extraction_matches_builder_definition(monkeypatch):
    mask = np.asarray(
        [
            [0, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0],
        ],
        dtype=np.bool_,
    )
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    rows, columns = np.nonzero(mask & ~interior)
    expected = np.column_stack((rows + 20, columns + 10)).astype(np.float64)
    monkeypatch.setattr(pose, "BOUNDARY_WORKING_SET_PIXELS", 5)
    observed = pose._boundary_points_parent_rc(mask, (10, 20, 15, 25))
    assert np.array_equal(observed, expected)


def test_production_authority_invokes_formal_combined_gate_on_frozen_bytes(
    tmp_path, monkeypatch
):
    fixture = _fixture_authority(tmp_path)
    predictions = _add_same_data_benchmarks(
        _prediction_result(tmp_path / "predictions" / "real.json", fixture)
    )
    expected_payload = predictions.read_bytes()
    expected_sha = hashlib.sha256(expected_payload).hexdigest()
    calls = []

    def gate(payload, source_sha256):
        calls.append((payload, source_sha256))

    monkeypatch.setattr(pose, "_run_formal_combined_authority_gate", gate)
    authority = replace(
        fixture["authority"], enforce_formal_combined_gate=True
    )
    frozen = freeze_real_prediction_result(predictions, authority=authority)
    assert frozen.source_payload == expected_payload
    assert calls == [(expected_payload, expected_sha)]


def _add_known_matched_wrapper_alignment_omission(path: Path) -> Path:
    document = json.loads(path.read_text(encoding="utf-8"))
    evidence = {
        "claim_level": "fixture_exact_training_alignment",
        "train_manifest": {"content_sha256": "c" * 64},
    }
    for population in ("strict_547", "balanced_1016"):
        document[population]["source_matched_mm_training_run"][
            "training_alignment_hash_evidence"
        ] = evidence
    document["matched_training_alignment_hash_evidence"] = evidence
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_freeze_accepts_only_known_matched_wrapper_alignment_omission(tmp_path):
    fixture = _fixture_authority(tmp_path)
    predictions = _add_known_matched_wrapper_alignment_omission(
        _prediction_result(tmp_path / "predictions" / "real.json", fixture)
    )
    expected_payload = predictions.read_bytes()

    frozen = freeze_real_prediction_result(
        predictions,
        authority=fixture["authority"],
        compatibility_mode=True,
    )

    assert frozen.source_payload == expected_payload
    assert predictions.read_bytes() == expected_payload


@pytest.mark.parametrize(
    "tamper",
    ("balanced_evidence", "wrapper_receipt", "root_evidence"),
)
def test_freeze_rejects_other_matched_wrapper_authority_differences(
    tmp_path, tamper
):
    fixture = _fixture_authority(tmp_path)
    predictions = _add_known_matched_wrapper_alignment_omission(
        _prediction_result(tmp_path / "predictions" / "real.json", fixture)
    )
    document = json.loads(predictions.read_text(encoding="utf-8"))
    if tamper == "balanced_evidence":
        document["balanced_1016"]["source_matched_mm_training_run"][
            "training_alignment_hash_evidence"
        ]["claim_level"] = "different"
    elif tamper == "wrapper_receipt":
        document["source_matched_mm_training_run"]["receipt_sha256"] = "d" * 64
    else:
        document["matched_training_alignment_hash_evidence"][
            "claim_level"
        ] = "different"
    predictions.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        RachelRealTranslationGTError,
        match="strict/balanced/wrapper matched authority differs",
    ):
        freeze_real_prediction_result(
            predictions,
            authority=fixture["authority"],
            compatibility_mode=True,
        )


def test_formal_combined_gate_uses_immutable_snapshot_and_removes_it(
    tmp_path, monkeypatch
):
    del tmp_path
    payload = b'{"frozen":true}'
    source_sha = hashlib.sha256(payload).hexdigest()
    observed_snapshot = []

    def load_real(path):
        snapshot = Path(path)
        observed_snapshot.append((snapshot, snapshot.read_bytes()))
        return pose.formal_stats.LoadedEvaluation(
            source_kind="real_balanced1016",
            source_path=snapshot,
            source_sha256=source_sha,
            methods=pose.COMPLETE_BENCHMARK_METHODS,
            rows=tuple(None for _ in range(pose.formal_stats.EXPECTED_BALANCED_COUNT)),
            strict_prefix_count=pose.formal_stats.EXPECTED_STRICT_COUNT,
            validation_thresholds={
                method: 0.5 for method in pose.COMPLETE_BENCHMARK_METHODS
            },
            no_test_tuning_evidence={
                "winner_and_validation_threshold_frozen_before_real_open": True,
                "real_geometry_ground_truth_read": False,
            },
            input_files=(),
        )

    monkeypatch.setattr(pose.formal_stats, "load_real", load_real)
    pose._run_formal_combined_authority_gate(payload, source_sha)
    assert observed_snapshot[0][1] == payload
    assert not observed_snapshot[0][0].exists()


def test_freeze_rejects_nonfull_method_threshold_decision_tamper(tmp_path):
    fixture = _fixture_authority(tmp_path)
    predictions = _prediction_result(tmp_path / "predictions" / "real.json", fixture)
    document = json.loads(predictions.read_text(encoding="utf-8"))
    for population in ("strict_547", "balanced_1016"):
        document[population]["pairs"][0]["methods"]["coarse_only"][
            "decision_at_frozen_validation_threshold"
        ] = False
    predictions.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RachelRealTranslationGTError, match="stored decision"):
        freeze_real_prediction_result(
            predictions,
            authority=fixture["authority"],
            compatibility_mode=True,
        )


def test_freeze_rejects_matched_score_semantics_tamper(tmp_path):
    fixture = _fixture_authority(tmp_path)
    predictions = _prediction_result(tmp_path / "predictions" / "real.json", fixture)
    document = json.loads(predictions.read_text(encoding="utf-8"))
    for population in ("strict_547", "balanced_1016"):
        document[population]["methods"][MATCHED_METHODS[0]][
            "score_semantics"
        ] = "fused_probability"
    predictions.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RachelRealTranslationGTError, match="score semantics"):
        freeze_real_prediction_result(
            predictions,
            authority=fixture["authority"],
            compatibility_mode=True,
        )


def test_label_authority_frozen_bytes_survive_swap_then_recheck_blocks_publish(
    tmp_path, monkeypatch
):
    fixture = _fixture_authority(tmp_path)
    predictions = _prediction_result(tmp_path / "predictions" / "real.json", fixture)
    output = tmp_path / "outputs" / "pose.json"
    original_gt = pose._load_gt_after_prediction_freeze

    def swap_then_load(frozen, frozen_labels, **kwargs):
        fixture["manifest"].write_text("{}", encoding="utf-8")
        return original_gt(frozen, frozen_labels, **kwargs)

    monkeypatch.setattr(pose, "_load_gt_after_prediction_freeze", swap_then_load)
    with pytest.raises(RachelRealTranslationGTError, match="changed after"):
        evaluate_real_translation_gt_postprediction(
            predictions,
            fixture["manifest"],
            fixture["receipt"],
            output_path=output,
            main_root=fixture["main_root"],
            supp_root=fixture["supp_root"],
            bootstrap_repetitions=10,
            authority=fixture["authority"],
            compatibility_mode=True,
        )
    assert not output.exists()


def test_boundary_point_hard_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(pose, "MAX_BOUNDARY_POINTS_PER_FRAGMENT", 2)
    with pytest.raises(RachelRealTranslationGTError, match="fixed QA point limit"):
        pose._boundary_points_parent_rc(
            np.ones((5, 5), dtype=np.bool_), (0, 0, 5, 5)
        )


def test_atomic_publish_fsyncs_parent_directory(tmp_path, monkeypatch):
    target = tmp_path / "output" / "result.json"
    real_fsync = os.fsync
    fsync_modes = []

    def tracking_fsync(file_descriptor):
        fsync_modes.append(os.fstat(file_descriptor).st_mode)
        return real_fsync(file_descriptor)

    monkeypatch.setattr(pose.os, "fsync", tracking_fsync)
    pose._write_atomic_no_replace(target, {"status": "complete"})
    assert target.is_file()
    assert any(stat.S_ISDIR(mode) for mode in fsync_modes)

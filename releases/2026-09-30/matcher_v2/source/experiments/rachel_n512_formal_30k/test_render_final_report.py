import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest


MODULE = importlib.import_module(
    "experiments.rachel_n512_formal_30k.render_final_report"
)
N512_RECEIPT = "1" * 64
MATCHED_RECEIPT = "2" * 64
CHECKPOINTS = {
    "coarse_only": "3" * 64,
    "full_n512": "4" * 64,
    "matched_mm_converged": "5" * 64,
    "matched_mm_same_exposure_epoch5": "6" * 64,
    "pairingnet_adapted": "b" * 64,
    "shreddingnet_adapted": "c" * 64,
}
THRESHOLDS = {
    "coarse_only": "7" * 64,
    "full_n512": "8" * 64,
    "matched_mm_converged": "9" * 64,
    "matched_mm_same_exposure_epoch5": "a" * 64,
    "pairingnet_adapted": "d" * 64,
    "shreddingnet_adapted": "e" * 64,
}
BENCHMARK_STAGE_CHECKPOINTS = {
    "pairingnet_adapted": {"winner": CHECKPOINTS["pairingnet_adapted"]},
    "shreddingnet_adapted": {
        "screening": "8" * 64,
        "scorer": CHECKPOINTS["shreddingnet_adapted"],
        "transform": "9" * 64,
    },
}
BENCHMARK_FREEZE_SHA = {
    "pairingnet_adapted": "4" * 64,
    "shreddingnet_adapted": "5" * 64,
}


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


MODEL_CONFIG_SHA = hashlib.sha256(
    _canonical(MODULE.CANONICAL_RACHEL_N512_MODEL_CONFIG)
).hexdigest()
LOSS_CONFIG_SHA = hashlib.sha256(
    _canonical(MODULE.CANONICAL_RACHEL_N512_LOSS_CONFIG)
).hexdigest()


def _config_authority():
    return {
        "comparison": "complete_dataclass_field_mapping_exact_equality",
        "model_config": dict(MODULE.CANONICAL_RACHEL_N512_MODEL_CONFIG),
        "model_config_sha256": MODEL_CONFIG_SHA,
        "loss_config": dict(MODULE.CANONICAL_RACHEL_N512_LOSS_CONFIG),
        "loss_config_sha256": LOSS_CONFIG_SHA,
        "model_config_sha256_by_arm": {
            "coarse_only": MODEL_CONFIG_SHA,
            "full_n512": MODEL_CONFIG_SHA,
        },
        "loss_config_sha256_by_arm": {
            "coarse_only": LOSS_CONFIG_SHA,
            "full_n512": LOSS_CONFIG_SHA,
        },
        "all_winner_configs_exactly_equal": True,
    }


def _rachel_provenance():
    return {
        "sidecar_path": "/authority/rachel_data_provenance_pretest.json",
        "sidecar_sha256": "b" * 64,
        "schema_version": "rachel-n512-pretest-data-provenance/1.0",
        "status": "frozen_before_current_final_controller_test_open",
        "dataset_root": "/dataset/rachel",
        "source_authority": {
            "description": "Rachel RGB JPEG plus colocated label.csv only",
            "shredding_data_read": False,
        },
        "selection_contract": json.loads(
            json.dumps(MODULE.CANONICAL_RACHEL_SELECTION_CONTRACT)
        ),
        "files": {
            "preprocess_receipt": {
                "relative_path": "preprocess_receipt.json",
                "sha256": "9" * 64,
            },
            "preprocess_summary": {
                "relative_path": "qa/preprocess_summary.json",
                "sha256": "a" * 64,
            },
            "train_manifest": {
                "relative_path": "pairs/train.jsonl",
                "sha256": "d" * 64,
                "line_count": 24000,
            },
            "validation_manifest": {
                "relative_path": "pairs/val.jsonl",
                "sha256": "f" * 64,
                "line_count": 3000,
            },
        },
        "preprocess_receipt_schema_version": "rachel-pairwise-n512-preprocessing/1.0",
        "preprocess_receipt_status": "complete_rachel_pairwise_n512_30k",
        "preprocess_summary_exactly_embedded_in_receipt": True,
        "selection_summary_schema_version": "rachel-pairwise-30k-selection/1.0",
        "test_manifest_content_read_or_hashed_by_pretest_sidecar": False,
        "verified_before_current_test_or_real_open": True,
    }


def _with_content_sha(value):
    result = dict(value)
    result["content_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return result


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _benchmark_adaptation(method: str):
    common = {
        "claim": "same_data_method_adaptation_not_exact_reproduction",
        "mask_only": True,
    }
    if method == "pairingnet_adapted":
        return {
            **common,
            "primary_pose": "upright_translation_only_consensus",
            "pair_head_is_official_component": False,
            "dual_softmax_is_sinkhorn": False,
        }
    return {
        **common,
        "rachel_n512": True,
        "upright_known_translation_only_primary": True,
        "global_assembly_performed": False,
        "native_cm_fm_se_or_ga_claimed": False,
        "balanced_pair_list_diagnostics_are_native_cm_fm_se": False,
    }


def _direct_prf():
    return {
        "true_positive_count": 900,
        "predicted_count": 1000,
        "target_count": 1500,
        "false_positive_count": 100,
        "false_negative_count": 600,
        "precision": 0.9,
        "recall": 0.6,
        "f1": 0.72,
    }


def _benchmark_direct_metrics(method: str):
    return {
        "schema_version": "rachel-common-direct-pairwise-report/1.0",
        "status": "complete_frozen_direct_pairwise_evaluation",
        "method_key": method,
        "pair_count": 3000,
        "positive_count": 1500,
        "negative_count": 1500,
        "scope": {
            "same_data_method_adaptation_not_exact_reproduction": True,
            "global_assembly_performed": False,
            "native_cm_fm_se_or_ga_claimed": False,
        },
        "threshold": {
            "source": "frozen_validation_only_artifact",
            "fit_performed_here": False,
            "artifact_sha256": THRESHOLDS[method],
        },
        "translation": {
            "eligible_positive_count": 1500,
            "valid_prediction_count": 1350,
            "valid_prediction_fraction": 0.9,
            "te_px_conditioned_on_valid_pose": {
                "count": 1350,
                "median": 3.0,
                "p90": 8.0,
            },
            "unconditional_positive_recall": {
                "at_{}px".format(tolerance): 0.8
                for tolerance in MODULE.TOLERANCES
            },
        },
        "assembly_edge": {
            "predicted_edge_count": 1000,
            "target_edge_count": 1500,
            "by_tolerance": {
                "at_{}px".format(tolerance): _direct_prf()
                for tolerance in MODULE.TOLERANCES
            },
        },
        "pairingnet_style_registration": {
            "eligible_positive_count": 1500,
            "valid_pose_count": 1350,
            "identity_fallback_count": 150,
            "rotation_error": {"status": "not_applicable"},
            "rr_lt4": 0.8,
            "mean_e_rmse": 2.0,
            "mean_symmetric_hausdorff_px": 4.0,
            "mean_normalized_translation_error": 0.001,
        },
        "correspondence": {
            "status": "reported",
            "thresholded_exact": _direct_prf(),
            "reciprocal_top1_exact": _direct_prf(),
            "dustbin_aware": {"status": "not_applicable"},
        },
        "unavailable_or_not_applicable": {
            "native_global_assembly_GA": {"status": "not_applicable"},
            "shreddingnet_native_CM_FM_SE": {"status": "not_reported"},
        },
    }


def _sealed_receipt():
    rows = []
    for method in MODULE.METHODS:
        row = {"arm": method}
        if method in MODULE.BENCHMARK_METHODS:
            row.update(
                {
                    "validation_threshold_sha256": THRESHOLDS[method],
                    "adaptation_claim": (
                        "same_data_method_adaptation_not_exact_reproduction"
                    ),
                    "native_cm_fm_se_or_ga_claimed": False,
                    "metrics": _benchmark_direct_metrics(method),
                }
            )
        rows.append(row)
    return {
        "schema_version": "rachel-n512-sealed-synthetic-test/1.0",
        "status": "complete_frozen_synthetic_test_only",
        "formal_evaluation": True,
        "compatibility_mode": False,
        "arm_results": rows,
    }


def _minimal_completed_root(tmp_path: Path) -> Path:
    root = tmp_path / "final"
    root.mkdir()
    inventory = _with_content_sha(
        {
            "schema_version": MODULE.CONTROLLER_SCHEMA,
            "status": "complete_content_sha256_inventory_before_terminal_receipt",
            "root": str(root.resolve()),
            "excludes": list(MODULE.INVENTORY_EXCLUDES),
            "files": [],
            "file_count": 0,
        }
    )
    inventory_path = root / MODULE.INVENTORY_RELATIVE
    _write_json(inventory_path, inventory)
    terminal = _with_content_sha(
        {
            "schema_version": MODULE.CONTROLLER_SCHEMA,
            "status": "complete_final_pairwise_protocol",
            "root": str(root.resolve()),
            "stage_order": list(MODULE.STAGE_ORDER),
            "fresh_sibling_output_directories": [
                "sealed",
                "corrosion",
                "real",
                "stats_synthetic",
                "stats_real",
                "real_translation_gt",
            ],
            "logs_directory": "control/logs",
            "terminal_receipt_first_hard_link_wins": True,
            "formal_exact_six": True,
            "six_methods_verified_in_sealed_corrosion_real_statistics_and_real_translation_gt": True,
            "same_data_benchmark_adaptations_not_exact_reproductions": True,
            "native_pairingnet_or_shreddingnet_CM_FM_SE_GA_claimed": False,
            "automatic_performance_pass_fail_applied": False,
            "corrosion_scope": {
                "formal_exact_six": True,
                "method_inventory": list(MODULE.CORROSION_METHODS),
                "same_data_benchmark_methods_included": True,
                "benchmark_correspondence": (
                    "not_applicable_corrupted_contour_indices_have_no_current_condition_GT"
                ),
                "same_data_benchmark_adaptations_not_exact_reproductions": True,
            },
            "content_inventory": {
                "path": MODULE.INVENTORY_RELATIVE,
                "file_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                "inventory_file_count": 0,
            },
        }
    )
    _write_json(root / MODULE.TERMINAL_RELATIVE, terminal)
    return root


def test_module_imports_and_accepts_bound_receipts(tmp_path: Path):
    root = _minimal_completed_root(tmp_path)
    terminal, inventory = MODULE._load_terminal_and_inventory(root)
    assert terminal["status"] == "complete_final_pairwise_protocol"
    assert inventory["file_count"] == 0


def test_inventory_hash_tamper_fails_closed(tmp_path: Path):
    root = _minimal_completed_root(tmp_path)
    path = root / MODULE.INVENTORY_RELATIVE
    value = json.loads(path.read_text(encoding="utf-8"))
    value["file_count"] = 1
    _write_json(path, value)
    with pytest.raises(MODULE.FinalReportError, match="content SHA-256 differs"):
        MODULE._load_terminal_and_inventory(root)


def test_renderer_rejects_old_four_method_artifacts() -> None:
    values = {
        method: {"auroc": 0.8, "auprc": 0.8}
        for method in MODULE.BASE_METHODS
    }
    comparisons = {
        "full_n512_minus_" + method: {"auroc": 0.0, "auprc": 0.0}
        for method in MODULE.BASE_METHODS
        if method != "full_n512"
    }
    with pytest.raises(MODULE.FinalReportError, match="method set differs"):
        MODULE._point_metrics(
            {
                "row_count": 10,
                "positive_count": 5,
                "negative_count": 5,
                "methods": values,
                "comparisons": comparisons,
            },
            "old four-method ranking artifact",
        )

    sealed = _sealed_receipt()
    sealed["arm_results"] = sealed["arm_results"][:4]
    with pytest.raises(MODULE.FinalReportError, match="exact-six inventory differs"):
        MODULE._synthetic_direct_exact_six(sealed, {})

    translation = _real_translation_document("1" * 64)
    for method in MODULE.BENCHMARK_METHODS:
        del translation["method_metrics"][method]
    with pytest.raises(MODULE.FinalReportError, match="method set differs"):
        MODULE._real_translation(translation)


def _aggregate_completed_root(tmp_path: Path) -> Path:
    root = tmp_path / "complete"
    root.mkdir()
    sealed_receipt_path = root / "sealed/test-fixture/test_receipt.json"
    _write_json(sealed_receipt_path, _sealed_receipt())
    sealed_sha = hashlib.sha256(sealed_receipt_path.read_bytes()).hexdigest()

    synthetic = {
        "schema_version": "rachel-n512-paired-endpoint-bootstrap/1.1",
        "status": "complete_formal_exact_six",
        "formal_evaluation": True,
        "compatibility_mode": False,
        "methods": list(MODULE.METHODS),
        "input": {"source_kind": "sealed_synthetic", "source_sha256": sealed_sha},
    }
    real_source_path = root / MODULE.REAL_SOURCE_RELATIVE
    _write_json(real_source_path, {"fixture": "pair-only source is never parsed"})
    real_source_sha = hashlib.sha256(real_source_path.read_bytes()).hexdigest()
    real_stats = {
        "schema_version": "rachel-n512-paired-endpoint-bootstrap/1.1",
        "status": "complete_formal_exact_six",
        "formal_evaluation": True,
        "compatibility_mode": False,
        "methods": list(MODULE.METHODS),
        "input": {"source_kind": "real_balanced1016", "source_sha256": real_source_sha},
    }
    translation = _with_content_sha(
        {
            "schema_version": "rachel-n512-real-translation-gt-posteval/1.0",
            "status": "complete_postprediction_real_positive_translation_gt",
                "source_pair_only_evaluation": {
                    "file_sha256_frozen_before_gt_open": real_source_sha,
                    "source_training_receipt_sha256": N512_RECEIPT,
                    "source_matched_receipt_sha256": MATCHED_RECEIPT,
                    "formal_combined_authority_gate_passed": True,
                    "exact_method_order": list(MODULE.METHODS),
                    "checkpoint_sha256_by_method": dict(CHECKPOINTS),
                },
                "formal_evaluation": True,
                "compatibility_mode": False,
            }
        )
    corrosion_logical = "corrosion/corrosion-fixture/robustness_receipt.json"
    corrosion = {
        "schema_version": "rachel-n512-corrosion-robustness/2.0",
        "status": "complete_formal_exact_six_frozen_synthetic_corrosion_only",
        "formal_evaluation": True,
        "compatibility_mode": False,
        "clean_reference": {"receipt_sha256": sealed_sha},
        "source_training_runs": {
            "rachel_n512": {
                "receipt_sha256": N512_RECEIPT,
                "formal_plateau_verified_before_test_open": True,
            },
            "matched_mm": {
                "receipt_sha256": MATCHED_RECEIPT,
                "formal_plateau_verified_before_test_open": True,
            },
            "same_data_benchmarks": {
                "all_winners_and_validation_thresholds_frozen_before_test_open": True,
            },
        },
    }
    for logical, value in (
        (MODULE.SYNTHETIC_STATS_RELATIVE, synthetic),
        (MODULE.REAL_STATS_RELATIVE, real_stats),
        (MODULE.REAL_TRANSLATION_RELATIVE, translation),
        (corrosion_logical, corrosion),
    ):
        _write_json(root / logical, value)
    sealed_artifacts = []
    for method in MODULE.METHODS:
        raw_jsonl = root / "sealed/test-fixture" / method / "pair_scores.jsonl"
        raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
        raw_jsonl.write_text('{"must_not_be_opened":true}\n', encoding="utf-8")
        artifact = {
            "method": method,
            "path": method + "/pair_scores.jsonl",
            "sha256": hashlib.sha256(raw_jsonl.read_bytes()).hexdigest(),
            "checkpoint_sha256": CHECKPOINTS[method],
            "validation_threshold_sha256": THRESHOLDS[method],
        }
        if method in {"coarse_only", "full_n512"}:
            artifact.update(
                {
                    "model_config_sha256": MODEL_CONFIG_SHA,
                    "loss_config_sha256": LOSS_CONFIG_SHA,
                }
            )
        if method in MODULE.BENCHMARK_METHODS:
            artifact.update(
                {
                    "checkpoint_sha256_by_stage": BENCHMARK_STAGE_CHECKPOINTS[
                        method
                    ],
                    "freeze_authority_sha256": BENCHMARK_FREEZE_SHA[method],
                    "adaptation": _benchmark_adaptation(method),
                    "native_cm_fm_se_or_ga_claimed": False,
                }
            )
        sealed_artifacts.append(artifact)
    corrosion_artifacts = []
    for condition in MODULE.CONDITIONS:
        raw_jsonl = (
            root
            / "corrosion/corrosion-fixture/conditions"
            / condition
            / "pair_scores.jsonl"
        )
        raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
        raw_jsonl.write_text('{"must_not_be_opened":true}\n', encoding="utf-8")
        corrosion_artifacts.append(
            {
                "condition": condition,
                "path": "conditions/{}/pair_scores.jsonl".format(condition),
                "sha256": hashlib.sha256(raw_jsonl.read_bytes()).hexdigest(),
            }
        )

    artifact_sha = {
        logical: hashlib.sha256((root / logical).read_bytes()).hexdigest()
        for logical in (
            MODULE.SYNTHETIC_STATS_RELATIVE,
            MODULE.REAL_STATS_RELATIVE,
            MODULE.REAL_TRANSLATION_RELATIVE,
            corrosion_logical,
        )
    }
    integrity_flags = {
        "metric_artifacts_read_for_integrity_validation": True,
        "metric_values_used_for_adaptive_model_checkpoint_threshold_or_condition_selection": False,
        "metric_values_emitted_to_controller_stdout": False,
        "predeclared_integrity_gates_may_abort": True,
    }
    queue_fields = {
        "queue_terminal": {
            "path": "/benchmark/control/terminal_receipt.json",
            "file_sha256": "1" * 64,
        },
        "pairingnet": {
            "completion_receipt_sha256": BENCHMARK_FREEZE_SHA[
                "pairingnet_adapted"
            ],
            "winner_checkpoint_sha256": CHECKPOINTS["pairingnet_adapted"],
            "validation_threshold_artifact_sha256": THRESHOLDS[
                "pairingnet_adapted"
            ],
        },
        "shreddingnet": {
            "train_val_freeze_file_sha256": BENCHMARK_FREEZE_SHA[
                "shreddingnet_adapted"
            ],
            "winner_checkpoint_sha256_by_stage": BENCHMARK_STAGE_CHECKPOINTS[
                "shreddingnet_adapted"
            ],
            "validation_threshold_artifact_sha256": THRESHOLDS[
                "shreddingnet_adapted"
            ],
        },
        "queue_source_authority": {
            "reviewed_GO_markers": {
                name: {
                    "path": "/immutable/reviews/" + name + ".json",
                    "relative_path": "reviews/" + name + ".json",
                    "file_sha256": str(index + 1) * 64,
                    "schema_version": "fixture-review/1.0",
                    "status": "GO",
                }
                for index, name in enumerate(
                    (
                        "queue_controller",
                        "train_val_asset_freeze",
                        "benchmark_environment",
                        "pairingnet_runner",
                        "shreddingnet_runner",
                    )
                )
            }
        },
        "exact_two_benchmark_methods_verified": True,
        "sealed_test_or_real_opened": False,
    }
    queue_logical = "control/same_data_benchmark_queue_gate.json"
    _write_json(
        root / queue_logical,
        _with_content_sha(
            {
                "schema_version": MODULE.CONTROLLER_SCHEMA,
                "status": MODULE.GATE_STATUS[queue_logical],
                **integrity_flags,
                **queue_fields,
            }
        ),
    )
    queue_gate_sha = hashlib.sha256((root / queue_logical).read_bytes()).hexdigest()
    benchmark_methods = {
        method: {
            "freeze_authority_sha256": BENCHMARK_FREEZE_SHA[method],
            "checkpoint_sha256_by_stage": BENCHMARK_STAGE_CHECKPOINTS[method],
            "validation_threshold_sha256": THRESHOLDS[method],
            "validation_threshold_checkpoint_sha256": CHECKPOINTS[method],
            "validation_threshold": {
                "checkpoint_sha256": CHECKPOINTS[method]
            },
            "same_data_method_adaptation_not_exact_reproduction": True,
            "native_cm_fm_se_or_ga_claimed": False,
            "adaptation": _benchmark_adaptation(method),
        }
        for method in MODULE.BENCHMARK_METHODS
    }
    benchmark_freeze = {
        "queue_gate_file_sha256": queue_gate_sha,
        "methods": benchmark_methods,
        "all_winners_and_validation_thresholds_strict_loaded_cpu": True,
        "both_same_data_adaptations_not_exact_reproductions": True,
        "native_cm_fm_se_or_ga_claimed": False,
        "training_manifest_alignment": {
            "exact_manifest_bytes_equal_across_n512_pairingnet_and_shreddingnet": True,
            "manifest_content_sha256": {"train": "d" * 64, "val": "f" * 64},
            "methods": {
                method: {
                    "training_manifest_sha256": {
                        "train": "d" * 64,
                        "val": "f" * 64,
                    },
                    "checkpoint_sha256_by_stage": BENCHMARK_STAGE_CHECKPOINTS[
                        method
                    ],
                    "freeze_authority_sha256": BENCHMARK_FREEZE_SHA[method],
                    "validation_threshold_sha256": THRESHOLDS[method],
                    "adaptation": _benchmark_adaptation(method),
                }
                for method in MODULE.BENCHMARK_METHODS
            },
        },
    }
    gates = {
        queue_logical: queue_fields,
        "control/train_validation_freeze_gate.json": {
            "n512": {
                "receipt_sha256": N512_RECEIPT,
                "convergence_receipt_sha256": "b" * 64,
                "winner_checkpoint_sha256_by_arm": {
                    method: CHECKPOINTS[method]
                    for method in ("coarse_only", "full_n512")
                },
                "validation_threshold_sha256_by_arm": {
                    method: THRESHOLDS[method]
                    for method in ("coarse_only", "full_n512")
                },
                "canonical_model_and_loss_config_authority": _config_authority(),
                "both_arms_validation_plateau_verified": True,
            },
            "matched_mm": {
                "receipt_sha256": MATCHED_RECEIPT,
                "winner_checkpoint_sha256_by_method": {
                    method: CHECKPOINTS[method]
                    for method in (
                        "matched_mm_converged",
                        "matched_mm_same_exposure_epoch5",
                    )
                },
                "validation_threshold_sha256_by_method": {
                    method: THRESHOLDS[method]
                    for method in (
                        "matched_mm_converged",
                        "matched_mm_same_exposure_epoch5",
                    )
                },
                "converged_and_epoch5_validation_winners_frozen": True,
                "validation_plateau_verified": True,
            },
            "same_dataset_seed_population_and_exposure_alignment_verified": True,
            "same_data_benchmarks": benchmark_freeze,
            "formal_method_inventory": list(MODULE.METHODS),
            "formal_exact_six_winners_and_validation_thresholds_frozen_before_test_or_real_open": True,
            "alignment_hash_evidence": {
                "claim_level": "same_frozen_train_val_manifest_content_and_exact_validation_pair_order;per_epoch_training_presentation_order_not_provable",
                "canonical_dataset_root": "/dataset/rachel",
                "canonical_seed": 3407,
                "train_manifest": {
                    "content_sha256": "d" * 64,
                    "bound_to_both_training_fingerprints": True,
                    "pair_count": 24000,
                    "pair_order_fingerprint_sha256": "e" * 64,
                },
                "validation_manifest": {
                    "content_sha256": "f" * 64,
                    "bound_to_both_training_fingerprints": True,
                    "pair_count": 3000,
                    "pair_order_fingerprint_sha256": "0" * 64,
                    "exact_order_equal_to_all_four_validation_score_artifacts": True,
                },
                "training_fingerprints_recomputed": {
                    "rachel_n512": "1" * 64,
                    "matched_mm": "2" * 64,
                },
                "validation_pair_order": {
                    "exact_order_equal_across_four_frozen_thresholds": True,
                    "pair_count": 3000,
                    "pair_order_fingerprint_sha256": "0" * 64,
                    "sources": {
                        method: {
                            "artifact_sha256": "3" * 64,
                            "pair_count": 3000,
                            "pair_order_fingerprint_sha256": "0" * 64,
                            "threshold_bound_to_pair_order": True,
                            "threshold_checkpoint_sha256": CHECKPOINTS[method],
                        }
                        for method in MODULE.BASE_METHODS
                    },
                },
                "limitations": {
                    "exact_per_epoch_training_pair_presentation_order_saved": False,
                    "exact_per_epoch_training_pair_presentation_order_claimed_equal": False,
                },
            },
            "rachel_data_provenance": _rachel_provenance(),
            "test_accessed_by_controller_preflight": False,
            "real_external_test_accessed_by_controller_preflight": False,
        },
        "control/sealed_gate.json": {
            "result_directory": str((root / "sealed/test-fixture").resolve()),
            "receipt_sha256": sealed_sha,
            "pair_score_artifacts": sealed_artifacts,
            "formal_exact_six": True,
            "six_methods_verified": True,
            "method_inventory": list(MODULE.METHODS),
            "same_data_benchmark_adaptations_not_exact_reproductions": True,
            "native_cm_fm_se_or_ga_claimed": False,
            "pair_count": 3000,
        },
        "control/corrosion_gate.json": {
            "result_directory": str((root / "corrosion/corrosion-fixture").resolve()),
            "receipt_sha256": artifact_sha[corrosion_logical],
            "condition_pair_score_artifacts": corrosion_artifacts,
            "formal_exact_six": True,
            "six_methods_verified": True,
            "method_inventory": list(MODULE.CORROSION_METHODS),
            "same_data_benchmark_methods_included": True,
            "same_data_benchmark_adaptations_not_exact_reproductions": True,
            "benchmark_correspondence_not_applicable_under_corrosion": True,
            "clean_reference": {"receipt_sha256": sealed_sha},
        },
        "control/real_gate.json": {
            "result": str(real_source_path.resolve()),
            "result_sha256": real_source_sha,
        },
        "control/synthetic_stats_gate.json": {
            "result": str((root / MODULE.SYNTHETIC_STATS_RELATIVE).resolve()),
            "result_sha256": artifact_sha[MODULE.SYNTHETIC_STATS_RELATIVE],
            "source_receipt_sha256": sealed_sha,
        },
        "control/real_stats_gate.json": {
            "result": str((root / MODULE.REAL_STATS_RELATIVE).resolve()),
            "result_sha256": artifact_sha[MODULE.REAL_STATS_RELATIVE],
            "source_result_sha256": real_source_sha,
        },
        "control/real_translation_gate.json": {
            "result": str((root / MODULE.REAL_TRANSLATION_RELATIVE).resolve()),
            "result_sha256": artifact_sha[MODULE.REAL_TRANSLATION_RELATIVE],
            "source_pair_only_result_sha256": real_source_sha,
        },
        "control/source_code_freeze.json": {
            "source_root": "/immutable/source",
            "scope": "all regular .py/.sh under staging/pairwise_v0_2 and experiments/rachel_n512_formal_30k",
            "test_or_real_manifest_read": False,
            "files": [
                {"path": path, "size": 1, "sha256": "c" * 64}
                for path in (
                    "experiments/rachel_n512_formal_30k/paired_cluster_bootstrap.py",
                    "experiments/rachel_n512_formal_30k/run_final_pairwise_protocol.sh",
                    "staging/pairwise_v0_2/training/rachel_n512_sealed_test.py",
                    "staging/pairwise_v0_2/baselines/rachel_same_data_benchmark_eval_adapter.py",
                    "staging/pairwise_v0_2/baselines/rachel_pairingnet_benchmark.py",
                    "staging/pairwise_v0_2/baselines/rachel_shreddingnet_benchmark.py",
                    "staging/pairwise_v0_2/baselines/rachel_n512_corrosion_robustness.py",
                    "staging/pairwise_v0_2/baselines/rachel_n512_real_external.py",
                    "staging/pairwise_v0_2/baselines/rachel_n512_real_translation_gt.py",
                )
            ],
        },
    }
    gates["control/source_code_freeze.json"]["file_count"] = len(
        gates["control/source_code_freeze.json"]["files"]
    )
    for logical, fields in gates.items():
        integrity = (
            {}
            if logical == "control/source_code_freeze.json"
            else {
                "metric_artifacts_read_for_integrity_validation": True,
                "metric_values_used_for_adaptive_model_checkpoint_threshold_or_condition_selection": False,
                "metric_values_emitted_to_controller_stdout": False,
                "predeclared_integrity_gates_may_abort": True,
            }
        )
        _write_json(
            root / logical,
            _with_content_sha(
                {
                    "schema_version": MODULE.CONTROLLER_SCHEMA,
                    "status": MODULE.GATE_STATUS[logical],
                    **integrity,
                    **fields,
                }
            ),
        )

    members = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            payload = path.read_bytes()
            members.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    inventory = _with_content_sha(
        {
            "schema_version": MODULE.CONTROLLER_SCHEMA,
            "status": "complete_content_sha256_inventory_before_terminal_receipt",
            "root": str(root.resolve()),
            "excludes": list(MODULE.INVENTORY_EXCLUDES),
            "file_count": len(members),
            "files": members,
        }
    )
    inventory_path = root / MODULE.INVENTORY_RELATIVE
    _write_json(inventory_path, inventory)
    gate_rows = []
    for logical in MODULE.GATE_STATUS:
        gate_rows.append(
            {
                "path": logical,
                "file_sha256": hashlib.sha256((root / logical).read_bytes()).hexdigest(),
                "status": MODULE.GATE_STATUS[logical],
            }
        )
    terminal = _with_content_sha(
        {
            "schema_version": MODULE.CONTROLLER_SCHEMA,
            "status": "complete_final_pairwise_protocol",
            "root": str(root.resolve()),
            "terminal_receipt_first_hard_link_wins": True,
            "stage_order": list(MODULE.STAGE_ORDER),
            "fresh_sibling_output_directories": [
                "sealed",
                "corrosion",
                "real",
                "stats_synthetic",
                "stats_real",
                "real_translation_gt",
            ],
            "logs_directory": "control/logs",
            "formal_exact_six": True,
            "six_methods_verified_in_sealed_corrosion_real_statistics_and_real_translation_gt": True,
            "same_data_benchmark_adaptations_not_exact_reproductions": True,
            "native_pairingnet_or_shreddingnet_CM_FM_SE_GA_claimed": False,
            "automatic_performance_pass_fail_applied": False,
            "corrosion_scope": {
                "formal_exact_six": True,
                "method_inventory": list(MODULE.CORROSION_METHODS),
                "same_data_benchmark_methods_included": True,
                "benchmark_correspondence": (
                    "not_applicable_corrupted_contour_indices_have_no_current_condition_GT"
                ),
                "same_data_benchmark_adaptations_not_exact_reproductions": True,
            },
            "gate_receipts": gate_rows,
            "frozen_training_authority": {
                "n512_receipt_sha256": N512_RECEIPT,
                "matched_mm_receipt_sha256": MATCHED_RECEIPT,
                "same_data_benchmarks": benchmark_freeze,
                "formal_method_inventory": list(MODULE.METHODS),
                "formal_exact_six_winners_and_thresholds_frozen_before_test_or_real_open": True,
                "canonical_model_and_loss_config_authority": _config_authority(),
                "rachel_data_provenance": _rachel_provenance(),
                "receipts_winner_checkpoints_configs_and_data_provenance_reverified_after_all_evaluations": True,
            },
            "content_inventory": {
                "path": MODULE.INVENTORY_RELATIVE,
                "file_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                "inventory_file_count": len(members),
            },
        }
    )
    _write_json(root / MODULE.TERMINAL_RELATIVE, terminal)
    return root


def _rebind_inventory(root: Path, mutate):
    path = root / MODULE.INVENTORY_RELATIVE
    inventory = json.loads(path.read_text(encoding="utf-8"))
    inventory.pop("content_sha256")
    mutate(inventory)
    _write_json(path, _with_content_sha(inventory))
    terminal_path = root / MODULE.TERMINAL_RELATIVE
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal.pop("content_sha256")
    terminal["content_inventory"]["file_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    terminal["content_inventory"]["inventory_file_count"] = inventory["file_count"]
    _write_json(terminal_path, _with_content_sha(terminal))


def test_aggregate_loader_binds_exact_six_artifacts_without_opening_jsonl(
    tmp_path: Path, monkeypatch
):
    root = _aggregate_completed_root(tmp_path)
    original = Path.read_bytes

    def guarded(path):
        if path.suffix == ".jsonl":
            raise AssertionError("JSONL was opened")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    loaded = MODULE._load_aggregate_inputs(root)
    assert set(loaded.artifact_sha256) == {
        "synthetic_stats",
        "sealed_exact_six_receipt",
        "real_stats",
        "corrosion_receipt",
        "real_translation_gt",
        "real_pair_only_unopened",
    }
    assert len(loaded.terminal_file_sha256) == 64
    assert len(loaded.inventory_file_sha256) == 64
    assert set(loaded.member_lstat_snapshot) == set(loaded.inventory) | set(
        MODULE.INVENTORY_EXCLUDES
    )


@pytest.mark.parametrize("field", ["size", "sha256"])
def test_inventoried_aggregate_size_or_hash_tamper_fails(tmp_path: Path, field: str):
    root = _aggregate_completed_root(tmp_path)

    def mutate(inventory):
        row = next(
            item
            for item in inventory["files"]
            if item["path"] == MODULE.SYNTHETIC_STATS_RELATIVE
        )
        row[field] = row[field] + 1 if field == "size" else "0" * 64

    _rebind_inventory(root, mutate)
    with pytest.raises(
        MODULE.FinalReportError,
        match="size differs|content inventory|SHA-256 differs",
    ):
        MODULE._load_aggregate_inputs(root)


def test_duplicate_inventory_path_fails_closed(tmp_path: Path):
    root = _aggregate_completed_root(tmp_path)

    def mutate(inventory):
        inventory["files"].append(dict(inventory["files"][0]))
        inventory["file_count"] += 1

    _rebind_inventory(root, mutate)
    with pytest.raises(MODULE.FinalReportError, match="duplicate path"):
        MODULE._load_aggregate_inputs(root)


def test_unsafe_inventory_path_and_direct_jsonl_read_are_rejected(tmp_path: Path):
    root = _aggregate_completed_root(tmp_path)

    def mutate(inventory):
        inventory["files"][0]["path"] = "../escape.json"

    _rebind_inventory(root, mutate)
    with pytest.raises(MODULE.FinalReportError, match="path is unsafe"):
        MODULE._load_aggregate_inputs(root)
    with pytest.raises(MODULE.FinalReportError, match="must never be opened"):
        MODULE._read_json_member(
            root, "sealed/test-fixture/coarse_only/pair_scores.jsonl", "raw scores"
        )


def _e5_disclosure():
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
    }


def _interval(point, *, delta=False, nonnegative=False):
    low = point - 0.02
    high = point + 0.02
    if not delta:
        low = max(0.0, low)
        if not nonnegative:
            high = min(1.0, high)
    row = {
        "point_estimate": point,
        "percentile_95_ci": [low, high],
        "bootstrap_mean": point,
        "bootstrap_standard_error": 0.01,
        "valid_replicates": 19990,
    }
    if delta:
        row["probability_delta_gt_zero"] = 0.8 if point > 0 else 0.2
    return row


def _coverage(total, valid, positive, negative):
    def row(count):
        positive_valid = min(positive, count // 2)
        negative_valid = count - positive_valid
        if negative_valid > negative:
            negative_valid = negative
            positive_valid = count - negative_valid
        return {
            "valid_count": count,
            "population_count": total,
            "valid_fraction": count / total,
            "positive_valid_count": positive_valid,
            "positive_count": positive,
            "negative_valid_count": negative_valid,
            "negative_count": negative,
        }

    return {
        "native_by_method": {method: row(min(total, valid + 10)) for method in MODULE.METHODS},
        "all_method_common_valid": row(valid),
    }


def _point_metrics(valid, positive_valid, values):
    methods = {
        method: {"auroc": values[method][0], "auprc": values[method][1]}
        for method in MODULE.METHODS
    }
    comparisons = {
        "full_n512_minus_" + method: {
            metric: methods["full_n512"][metric] - methods[method][metric]
            for metric in ("auroc", "auprc")
        }
        for method in MODULE.COMPARATORS
    }
    return {
        "row_count": valid,
        "positive_count": positive_valid,
        "negative_count": valid - positive_valid,
        "methods": methods,
        "comparisons": comparisons,
    }


def _ranking_document(source_kind, source_sha, *, total, valid, positive, values, real):
    coverage = _coverage(total, valid, positive, total - positive)
    common_positive = coverage["all_method_common_valid"]["positive_valid_count"]
    points = _point_metrics(valid, common_positive, values)
    metrics = {}
    for metric in ("auroc", "auprc"):
        metrics[metric] = {
            "methods": {
                method: _interval(points["methods"][method][metric])
                for method in MODULE.METHODS
            },
            "paired_deltas": {
                name: _interval(row[metric], delta=True)
                for name, row in points["comparisons"].items()
            },
        }
    no_tuning = {
        "automated_nonuse_verified": True,
        "claim_no_human_cognitive_influence": False,
        "evaluation_history_disclosure": _e5_disclosure(),
    }
    if real:
        no_tuning.update(
            {
                "winner_and_validation_threshold_frozen_before_real_open": True,
                "threshold_fit_performed_on_real": False,
                "real_geometry_ground_truth_read": False,
                "constructed_negatives_used_for_tuning": False,
            }
        )
    else:
        no_tuning.update(
            {
                "winner_frozen_before_test_open": True,
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "threshold_fit_performed": False,
            }
        )
    return {
        "schema_version": "rachel-n512-paired-endpoint-bootstrap/1.1",
        "status": "complete_formal_exact_six",
        "formal_evaluation": True,
        "compatibility_mode": False,
        "input": {"source_kind": source_kind, "source_sha256": source_sha},
        "methods": list(MODULE.METHODS),
        "protocol": {
            "replicates": 20000,
            "seed": 20260901,
            "delta_direction": "full_n512 minus comparator",
            "automated_no_test_or_real_tuning": no_tuning,
        },
        "coverage": coverage,
        "common_population": {
            "row_count": valid,
            "positive_count": points["positive_count"],
            "negative_count": points["negative_count"],
            "semantic_sha256": "b" * 64,
        },
        "point_metrics": points,
        "paired_endpoint_pigeonhole_bootstrap": {
            "bootstrap": "endpoint-unit_pigeonhole_product_multiplicity",
            "replicates_requested": 20000,
            "valid_replicates": 19990,
            "skipped_single_class_replicates": 10,
            "metrics": metrics,
        },
    }


def _synthetic_geometry():
    success = {
        "success_at_{}px".format(tolerance): _interval(0.7 + tolerance / 100)
        for tolerance in MODULE.TOLERANCES
    }
    recall = {
        "recall_at_{}px".format(tolerance): _interval(
            success["success_at_{}px".format(tolerance)]["point_estimate"] * 0.9
        )
        for tolerance in MODULE.TOLERANCES
    }

    def prf(true_positive, predicted, target):
        precision = true_positive / predicted if predicted else 0.0
        recall_value = true_positive / target if target else 0.0
        harmonic = (
            2 * precision * recall_value / (precision + recall_value)
            if precision + recall_value
            else 0.0
        )
        return {
            "point_counts": {
                "true_positive": true_positive,
                "predicted_count": predicted,
                "target_count": target,
            },
            "precision": _interval(precision),
            "recall": _interval(recall_value),
            "harmonic_f1": _interval(harmonic),
        }

    assembly = {
        "at_{}".format(tolerance): prf(
            500 + index * 100, 1200, 1500
        )
        for index, tolerance in enumerate(MODULE.ASSEMBLY_EDGE_TOLERANCES)
    }
    return {
        "scope": "full_n512_geometry_on_entire_sealed_synthetic_population",
        "bootstrap": "endpoint-unit_pigeonhole_product_multiplicity",
        "sampling_dependency_unit_count": 44,
        "shared_draws_across_all_direct_geometry_metrics": True,
        "validation_threshold": 0.55,
        "threshold_source": "frozen_validation_checkpoint_bound_artifact",
        "full_n512": {
            "translation_l2_px": {
                "definition": "positive and full decision-valid; conditional on a finite translation",
                "eligible_positive_count": 1500,
                "valid_translation_count": 1350,
                "valid_translation_fraction": 0.9,
                "median": _interval(3.0, nonnegative=True),
                "p90": _interval(8.0, nonnegative=True),
                "recall_definition": (
                    "denominator is every positive pair; decision-invalid or "
                    "missing translation is a failure"
                ),
                "recall_by_tolerance": recall,
                "success_by_tolerance": success,
                "bootstrap_replicates_without_valid_translation": 10,
            },
            "correspondence": {
                "scope": (
                    "entire sealed population; invalid decisions preserve targets "
                    "and contribute no predicted matches/correct dustbins"
                ),
                "strict_dustbin_aware": prf(900, 1100, 1200),
                "mutual_top1": prf(950, 1150, 1200),
                "dustbin_accuracy": {
                    "point_counts": {"correct": 9000, "token_count": 10000},
                    "accuracy": _interval(0.9),
                },
            },
            "pairingnet_style_registration": {
                "compatibility_source": (
                    "PairingNet released matching_test.py, specialized to frozen "
                    "upright translation-only Dunhuang inputs"
                ),
                "eligible_positive_count": 1500,
                "evaluated_positive_count": 1500,
                "valid_pose_count": 1350,
                "identity_fallback_count": 150,
                "valid_pose_fraction": 0.9,
                "invalid_pose_compatibility_fallback": (
                    "identity_translation_for_unconditional_official_style_aggregation"
                ),
                "e_rmse_definition": (
                    "sqrt(mean(per_correspondence_euclidean_distance)); official "
                    "compatibility definition, not conventional RMSE"
                ),
                "registration_recall_definition": "e_rmse_strictly_less_than_4",
                "hausdorff_definition": (
                    "max(directed_HD(transformed_source_seam,target_seam),"
                    "directed_HD(target_seam,transformed_source_seam))"
                ),
                "normalized_translation_error_definition": (
                    "identity-fallback translation_l2_px divided by the sum of ordered "
                    "N512 contour polygon areas after PairingNet int32 quantization"
                ),
                "mean_e_rmse": _interval(2.5, nonnegative=True),
                "registration_recall_e_rmse_lt4": _interval(0.8),
                "mean_symmetric_hausdorff_px": _interval(
                    5.0, nonnegative=True
                ),
                "mean_normalized_translation_error": _interval(
                    0.002, nonnegative=True
                ),
                "bootstrap_replicates_without_evaluable_positive": 10,
                "rotation_error": {
                    "status": "not_applicable_conditioned_upright_orientation",
                    "estimated_or_supervised": False,
                },
            },
            "assembly_edge_at_validation_threshold": {
                "definition": (
                    "predicted edge requires decision-valid and frozen-threshold "
                    "pair acceptance; true positive additionally requires adjacent "
                    "GT and translation L2 error at or below the pixel tolerance"
                ),
                "predicted_edge_count": 1200,
                "target_edge_count": 1500,
                "by_tolerance": assembly,
            },
            "joint_success_at_validation_threshold": {
                "definition": (
                    "denominator is every positive pair; success requires full "
                    "decision-valid, score >= frozen validation threshold, and "
                    "translation error <= tolerance"
                ),
                "denominator_positive_count": 1500,
                "success_by_tolerance": {
                    name: _interval(value["point_estimate"] - 0.1)
                    for name, value in success.items()
                },
                "bootstrap_replicates_without_positive": 10,
            },
        },
        "full_n512_minus_coarse_only": {
            "status": "not_applicable",
            "reason": (
                "coarse_only emits pair scores but no correspondence or 2D "
                "translation, so translation and joint geometric deltas are undefined"
            ),
        },
    }


def _strict_descriptive(values):
    coverage = _coverage(547, 520, 508, 39)
    positive = coverage["all_method_common_valid"]["positive_valid_count"]
    return {
        "role": "descriptive_only_no_inferential_CI",
        "coverage": coverage,
        "common_population_sha256": "c" * 64,
        "point_metrics": _point_metrics(520, positive, values),
    }


def _corrosion_receipt(clean_receipt_sha):
    method_values = {
        "coarse_only": (0.70, 0.68),
        "full_n512": (0.82, 0.80),
        "matched_mm_converged": (0.77, 0.75),
        "matched_mm_same_exposure_epoch5": (0.73, 0.71),
        "pairingnet_adapted": (0.79, 0.78),
        "shreddingnet_adapted": (0.76, 0.74),
    }
    by_condition = {}
    clean_delta = {}
    comparator_delta = {}
    translation_names = {
        "median_l2_px",
        "p90_l2_px",
        "valid_pose_fraction",
        *("recall_at_{}px".format(item) for item in MODULE.TOLERANCES),
        *(
            "joint_frozen_threshold_and_translation_recall_at_{}px".format(item)
            for item in MODULE.TOLERANCES
        ),
        *("assembly_precision_at_{}px".format(item) for item in MODULE.TOLERANCES),
        *("assembly_recall_at_{}px".format(item) for item in MODULE.TOLERANCES),
        *("assembly_f1_at_{}px".format(item) for item in MODULE.TOLERANCES),
        "pairing_rr_lt4",
        "pairing_mean_e_rmse",
        "pairing_mean_symmetric_hausdorff_px",
        "pairing_mean_normalized_translation_error",
    }
    nonnegative_translation_names = {
        "median_l2_px",
        "p90_l2_px",
        "pairing_mean_e_rmse",
        "pairing_mean_symmetric_hausdorff_px",
        "pairing_mean_normalized_translation_error",
    }
    direct_by_method = {
        method: {
            "threshold": 0.55,
            "threshold_source": "frozen_validation_checkpoint_bound_artifact",
            "by_condition": {},
            "condition_minus_clean": {},
            "correspondence": {
                "status": "not_applicable",
                "reason": "current-condition corrupted contour indices have no GT",
            },
        }
        for method in MODULE.TRANSLATION_METHODS
    }

    def translation_point(name, condition_index, method_index):
        if name == "median_l2_px":
            return 3.0 + condition_index + 0.2 * method_index
        if name == "p90_l2_px":
            return 8.0 + condition_index + 0.3 * method_index
        if name == "pairing_mean_e_rmse":
            return 2.0 + 0.2 * condition_index + 0.1 * method_index
        if name == "pairing_mean_symmetric_hausdorff_px":
            return 4.0 + 0.5 * condition_index + 0.2 * method_index
        if name == "pairing_mean_normalized_translation_error":
            return 0.001 + 0.0001 * condition_index + 0.00005 * method_index
        return 0.75 - 0.01 * condition_index - 0.01 * method_index

    for index, condition in enumerate(MODULE.CONDITIONS):
        drop = index * 0.01
        by_condition[condition] = {
            method: {
                "auroc": _interval(values[0] - drop),
                "auprc": _interval(values[1] - drop),
            }
            for method, values in method_values.items()
        }
        clean_delta[condition] = {
            method: {
                "auroc": _interval(-drop, delta=True),
                "auprc": _interval(-drop, delta=True),
            }
            for method in MODULE.CORROSION_METHODS
        }
        comparator_delta[condition] = {
            method: {
                metric: _interval(
                    by_condition[condition]["full_n512"][metric]["point_estimate"]
                    - by_condition[condition][method][metric]["point_estimate"],
                    delta=True,
                )
                for metric in ("auroc", "auprc")
            }
            for method in MODULE.CORROSION_COMPARATORS
        }
        for method_index, method in enumerate(MODULE.TRANSLATION_METHODS):
            points = {
                name: _interval(
                    translation_point(name, index, method_index),
                    nonnegative=name in nonnegative_translation_names,
                )
                for name in translation_names
            }
            deltas = {
                name: _interval(
                    translation_point(name, index, method_index)
                    - translation_point(name, 0, method_index),
                    delta=True,
                )
                for name in translation_names
            }
            direct_by_method[method]["by_condition"][condition] = points
            direct_by_method[method]["condition_minus_clean"][condition] = deltas
    full_projection = direct_by_method["full_n512"]
    paired = {
        "bootstrap": "endpoint-unit_pigeonhole_product_multiplicity",
        "shared_draws_across_all_methods_conditions_and_geometry_metrics": True,
        "population": "fixed_all_method_all_condition_common_valid",
        "method_order": list(MODULE.CORROSION_METHODS),
        "formal_exact_six": True,
        "replicates_requested": 20000,
        "valid_replicates": 19990,
        "skipped_single_class_replicates": 10,
        "seed": 260901,
        "ranking": {
            "by_condition_method": by_condition,
            "condition_minus_clean_same_method": clean_delta,
            "full_n512_minus_comparator_within_condition": comparator_delta,
        },
        "positive_only_translation_gt_and_joint": {
            "negative_pairs_excluded": True,
            "threshold": 0.55,
            "threshold_source": "frozen_validation_checkpoint_bound_artifact",
            "by_condition": full_projection["by_condition"],
            "condition_minus_clean": full_projection["condition_minus_clean"],
        },
        "direct_geometry_by_method": direct_by_method,
    }
    return {
        "schema_version": "rachel-n512-corrosion-robustness/2.0",
        "status": "complete_formal_exact_six_frozen_synthetic_corrosion_only",
        "formal_evaluation": True,
        "compatibility_mode": False,
        "test_accessed": True,
        "real_external_test_accessed": False,
        "clean_reference": {"receipt_sha256": clean_receipt_sha},
        "source_training_runs": {
            "rachel_n512": {
                "receipt_sha256": N512_RECEIPT,
                "formal_plateau_verified_before_test_open": True,
            },
            "matched_mm": {
                "receipt_sha256": MATCHED_RECEIPT,
                "formal_plateau_verified_before_test_open": True,
            },
            "same_data_benchmarks": {
                "all_winners_and_validation_thresholds_frozen_before_test_open": True,
            },
        },
        "test_population": {
            "pair_count": 3000,
            "positive_count": 1500,
            "negative_count": 1500,
            "same_pair_ids_labels_clusters_source_units_and_order_at_every_condition": True,
        },
        "protocol": {
            "condition_order": list(MODULE.CONDITIONS),
            "mask_only": True,
            "rgb_text_or_ocr_used": False,
            "known_upright_orientation": True,
            "rotation_search": False,
            "formal_evaluation": True,
            "compatibility_mode": False,
            "formal_exact_six": True,
            "formal_method_inventory": list(MODULE.CORROSION_METHODS),
            "method_order": list(MODULE.CORROSION_METHODS),
            "same_data_benchmark_methods_included": True,
            "same_corrupted_masks_supplied_to_all_six_methods": True,
            "each_method_forwarded_each_pair_exactly_once_per_condition": True,
            "all_winners_and_validation_thresholds_frozen_before_first_test_path_access": True,
            "exact_train_validation_manifest_alignment_verified_before_first_test_path_access": True,
            "frozen_validation_thresholds_are_secondary_only": True,
            "threshold_fit_performed": False,
            "checkpoint_selection_performed": False,
            "training_performed": False,
            "real_external_test_accessed": False,
            "evaluation_history_disclosure": _e5_disclosure(),
        },
        "summary": {
            "condition_order": list(MODULE.CONDITIONS),
            "method_order": list(MODULE.CORROSION_METHODS),
            "formal_exact_six": True,
            "primary_population": {
                "definition": "all_six_methods_valid_at_all_seven_conditions",
                "pair_count": 2700,
                "positive_count": 1350,
                "negative_count": 1350,
            },
            "paired_endpoint_pigeonhole_bootstrap": paired,
        },
        "paired_endpoint_pigeonhole_bootstrap": paired,
        "condition_results": [
            {"condition": condition} for condition in MODULE.CONDITIONS
        ],
    }


def _case_interval(point, nonnegative=False):
    return {
        "estimate": point,
        "ci95_low": max(0.0, point - 0.02),
        "ci95_high": point + 0.02 if nonnegative else min(1.0, point + 0.02),
        "valid_bootstrap_replicates": 20000,
    }


def _real_translation_document(real_source_sha):
    names = {
        "valid_translation_prediction_fraction",
        "median_l2_px",
        "p90_l2_px",
        *("recall_at_{}px".format(item) for item in MODULE.TOLERANCES),
        *(
            "joint_frozen_threshold_and_translation_recall_at_{}px".format(item)
            for item in MODULE.TOLERANCES
        ),
    }
    metrics = {
        name: _case_interval(
            3.0 if name == "median_l2_px" else (5.0 if name == "p90_l2_px" else 0.8),
            nonnegative=name in {"median_l2_px", "p90_l2_px"},
        )
        for name in names
    }

    def evaluated(method):
        assembly_precision = 300 / 400
        assembly_recall = 300 / 508
        assembly_f1 = (
            2
            * assembly_precision
            * assembly_recall
            / (assembly_precision + assembly_recall)
        )
        assembly_prf = {
            "true_positive_count": 300,
            "predicted_count": 400,
            "target_count": 508,
            "false_positive_count": 100,
            "false_negative_count": 208,
            "precision": assembly_precision,
            "recall": assembly_recall,
            "f1": assembly_f1,
        }
        return {
            "status": "evaluated_positive_translation_gt",
            "eligible_positive_count": 508,
            "scope": "strict_547_positive_pairs_plus_strict_and_balanced_edge_views",
            "frozen_validation_threshold": 0.55,
            "threshold_fit_performed_here": False,
            "invalid_translation_predictions_count_as_recall_failures": True,
            "point_estimates": {
                name: interval["estimate"] for name, interval in metrics.items()
            },
            "case_bootstrap": {
                "schema": "case_cluster_percentile_bootstrap/1.0",
                "seed": MODULE.REAL_TRANSLATION_BOOTSTRAP_SEED,
                "repetitions": 20000,
                "sampling_unit": "authoritative_real_case_uid",
                "same_case_pairs_keep_endpoints_together": True,
                "metrics": metrics,
            },
            "assembly_edge_strict_547": {
                "predicted_edge_count": 400,
                "target_edge_count": 508,
                "by_tolerance": {
                    "at_{}px".format(tolerance): dict(assembly_prf)
                    for tolerance in MODULE.TOLERANCES
                }
            },
            "pairingnet_style_registration": {
                "rotation_error": {"status": "not_applicable"},
                "rr_lt4": 0.8,
                "valid_pose_count": 450,
                "identity_fallback_count": 58,
                "mean_e_rmse": 2.0,
                "mean_symmetric_hausdorff_px": 4.0,
                "mean_normalized_translation_error": 0.001,
            },
            "correspondence": {"status": "not_applicable"},
            "native_global_assembly_GA": {"status": "not_applicable"},
            "shreddingnet_native_CM_FM_SE": {"status": "not_reported"},
            "adaptation": (
                _benchmark_adaptation(method)
                if method in MODULE.BENCHMARK_METHODS
                else None
            ),
        }

    body = {
        "schema_version": "rachel-n512-real-translation-gt-posteval/1.0",
        "status": "complete_postprediction_real_positive_translation_gt",
        "formal_evaluation": True,
        "compatibility_mode": False,
        "source_pair_only_evaluation": {
            "file_sha256_frozen_before_gt_open": real_source_sha,
            "source_training_receipt_sha256": N512_RECEIPT,
            "source_matched_receipt_sha256": MATCHED_RECEIPT,
            "formal_combined_authority_gate_passed": True,
            "exact_method_order": list(MODULE.METHODS),
            "checkpoint_sha256_by_method": dict(CHECKPOINTS),
        },
        "population": {
            "strict_pair_count": 547,
            "positive_pair_count": 508,
            "negative_pairs_have_translation_gt": False,
            "balanced_constructed_pairs_used_for_translation_or_GT": False,
            "balanced_constructed_pairs_claimed_GT_negative": False,
        },
        "method_metrics": {
            method: (
                evaluated(method)
                if method in MODULE.TRANSLATION_METHODS
                else {
                    "status": "not_applicable",
                    "reason": "method_does_not_emit_a_supervised_2d_translation",
                    "translation_metrics": None,
                }
            )
            for method in MODULE.METHODS
        },
        "direct_pairwise_metric_contract": {
            "translation_tolerances_px": list(MODULE.TOLERANCES),
            "rotation_error": "not_applicable_known_upright",
            "same_positive_pairs_and_case_bootstrap_draws_across_methods": True,
            "balanced_selected_list_diagnostics_claimed_as_native_shreddingnet_metrics": False,
        },
        "positive_pairs": [{} for _ in range(508)],
        "seam_and_correspondence_qa": {
            "status": "passed_all_translation_transform_and_residual_checks",
            "positive_pair_count": 508,
            "maximum_pair_p95_model_frame_residual_px": 1.25,
        },
        "protocol": {
            "pair_only_prediction_file_frozen_before_any_current_gt_open": True,
            "model_forward_or_gpu_work_performed_here": False,
            "gt_used_for_training_checkpoint_threshold_or_model_selection": False,
            "threshold_fit_performed_here": False,
            "coarse_and_matched_translation_are_not_applicable": True,
            "formal_evaluation": True,
            "compatibility_mode": False,
            "formal_exact_six_required": True,
            "pairingnet_and_shreddingnet_adapted_translation_evaluated": True,
            "shreddingnet_selected_list_diagnostic_claimed_native_CM_FM_SE_GA": False,
            "evaluation_history_disclosure": _e5_disclosure(),
        },
    }
    return _with_content_sha(body)


def _reseal_complete_root(root: Path):
    inventory_path = root / MODULE.INVENTORY_RELATIVE
    terminal_path = root / MODULE.TERMINAL_RELATIVE
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal.pop("content_sha256", None)

    gate_updates = {
        "control/synthetic_stats_gate.json": (
            "result_sha256",
            MODULE.SYNTHETIC_STATS_RELATIVE,
        ),
        "control/real_stats_gate.json": ("result_sha256", MODULE.REAL_STATS_RELATIVE),
        "control/real_translation_gate.json": (
            "result_sha256",
            MODULE.REAL_TRANSLATION_RELATIVE,
        ),
        "control/corrosion_gate.json": (
            "receipt_sha256",
            "corrosion/corrosion-fixture/robustness_receipt.json",
        ),
    }
    for gate_logical, (field, artifact_logical) in gate_updates.items():
        gate_path = root / gate_logical
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate.pop("content_sha256", None)
        gate[field] = hashlib.sha256((root / artifact_logical).read_bytes()).hexdigest()
        _write_json(gate_path, _with_content_sha(gate))

    inventory_path.unlink()
    terminal_path.unlink()
    members = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            payload = path.read_bytes()
            members.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    inventory = _with_content_sha(
        {
            "schema_version": MODULE.CONTROLLER_SCHEMA,
            "status": "complete_content_sha256_inventory_before_terminal_receipt",
            "root": str(root.resolve()),
            "excludes": list(MODULE.INVENTORY_EXCLUDES),
            "file_count": len(members),
            "files": members,
        }
    )
    _write_json(inventory_path, inventory)
    terminal.update(
        {
            "training_performed": False,
            "checkpoint_or_threshold_selection_performed": False,
            "metric_values_used_for_adaptive_model_checkpoint_threshold_or_condition_selection": False,
            "all_commands_completed_without_overwrite": True,
            "gate_receipts": [
                {
                    "path": logical,
                    "file_sha256": hashlib.sha256((root / logical).read_bytes()).hexdigest(),
                    "status": MODULE.GATE_STATUS[logical],
                }
                for logical in MODULE.GATE_STATUS
            ],
            "content_inventory": {
                "path": MODULE.INVENTORY_RELATIVE,
                "file_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                "inventory_file_count": len(members),
            },
        }
    )
    _write_json(terminal_path, _with_content_sha(terminal))


def _normalization_root(tmp_path: Path):
    root = _aggregate_completed_root(tmp_path)
    real_source_sha = hashlib.sha256((root / MODULE.REAL_SOURCE_RELATIVE).read_bytes()).hexdigest()
    values = {
        "coarse_only": (0.70, 0.68),
        "full_n512": (0.82, 0.80),
        "matched_mm_converged": (0.77, 0.75),
        "matched_mm_same_exposure_epoch5": (0.73, 0.71),
        "pairingnet_adapted": (0.79, 0.78),
        "shreddingnet_adapted": (0.76, 0.74),
    }
    synthetic = _ranking_document(
        "sealed_synthetic",
        hashlib.sha256(
            (root / "sealed/test-fixture/test_receipt.json").read_bytes()
        ).hexdigest(),
        total=3000,
        valid=2800,
        positive=1500,
        values=values,
        real=False,
    )
    synthetic["synthetic_geometry"] = _synthetic_geometry()
    real = _ranking_document(
        "real_balanced1016",
        real_source_sha,
        total=1016,
        valid=980,
        positive=508,
        values=values,
        real=True,
    )
    real.update(
        {
            "strict547_descriptive": _strict_descriptive(values),
            "balanced1016_inference_role": (
                "primary external paired endpoint-case pigeonhole bootstrap; "
                "constructed negatives remain non-GT distractors"
            ),
            "geometry": {
                "status": "not_read_or_reported_by_this_target_blind_pair_ranking_artifact",
                "ground_truth_availability_claim_made": False,
            },
        }
    )
    _write_json(root / MODULE.SYNTHETIC_STATS_RELATIVE, synthetic)
    _write_json(root / MODULE.REAL_STATS_RELATIVE, real)
    _write_json(
        root / "corrosion/corrosion-fixture/robustness_receipt.json",
        _corrosion_receipt(
            hashlib.sha256(
                (root / "sealed/test-fixture/test_receipt.json").read_bytes()
            ).hexdigest()
        ),
    )
    _write_json(
        root / MODULE.REAL_TRANSLATION_RELATIVE,
        _real_translation_document(real_source_sha),
    )
    _reseal_complete_root(root)
    return root


def test_normalization_extracts_all_populations_conditions_and_disclosures(tmp_path: Path):
    root = _normalization_root(tmp_path)
    summary = MODULE._normalize_aggregates(MODULE._load_aggregate_inputs(root))
    assert summary["methods"] == list(MODULE.METHODS)
    assert summary["populations"]["synthetic_balanced"]["source_pair_count"] == 3000
    assert summary["populations"]["real_strict547_descriptive"][
        "inferential_confidence_intervals"
    ] is None
    assert summary["populations"]["real_balanced1016_inferential"][
        "constructed_negative_count"
    ] == 469
    assert set(summary["corrosion_seven_condition"]["conditions"]) == set(
        MODULE.CONDITIONS
    )
    assert summary["geometry_translation"]["real_strict547_positive_full_n512"][
        "eligible_positive_count"
    ] == 508
    assert summary["disclosures"]["e5_human_visible"][
        "prior_epoch5_synthetic_test_human_visible_before_continuation"
    ] is True
    assert summary["integrity"]["performance_pass_fail_threshold_applied"] is False
    assert summary["frozen_authority"]["winner_checkpoint_sha256_by_method"] == CHECKPOINTS


def test_normalization_rejects_missing_ci_field_even_when_resealed(tmp_path: Path):
    root = _normalization_root(tmp_path)
    path = root / MODULE.SYNTHETIC_STATS_RELATIVE
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["paired_endpoint_pigeonhole_bootstrap"]["metrics"]["auroc"][
        "methods"
    ]["full_n512"]["percentile_95_ci"]
    _write_json(path, document)
    _reseal_complete_root(root)
    with pytest.raises(MODULE.FinalReportError, match="interval fields differ"):
        MODULE._normalize_aggregates(MODULE._load_aggregate_inputs(root))


def test_normalization_rejects_wrong_condition_set_even_when_resealed(tmp_path: Path):
    root = _normalization_root(tmp_path)
    path = root / "corrosion/corrosion-fixture/robustness_receipt.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["protocol"]["condition_order"] = document["protocol"]["condition_order"][:-1]
    _write_json(path, document)
    _reseal_complete_root(root)
    with pytest.raises(MODULE.FinalReportError, match="corrosion protocol differs"):
        MODULE._normalize_aggregates(MODULE._load_aggregate_inputs(root))


def test_cli_hashes_raw_jsonl_without_parsing_and_renders_disclosures(
    tmp_path: Path, capsys
):
    root = _normalization_root(tmp_path)
    output = tmp_path / "report"
    assert (
        MODULE.main(
            ["--final-root", str(root), "--output-directory", str(output)]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(output)
    assert sorted(path.name for path in output.iterdir()) == [
        "completion_receipt.json",
        "final_report.md",
        "final_report_summary.json",
    ]
    markdown = (output / "final_report.md").read_text(encoding="utf-8")
    required = {
        "Synthetic balanced",
        "Real balanced1016 inferential",
        "Real strict547 descriptive",
        "469 constructed non-GT distractor negatives",
        "Seven-condition mask-corrosion robustness",
        "human-visible before continuation",
        "applied no performance pass/fail threshold",
        "pairwise relative pose only",
        "not global multi-fragment assembly",
        "PairingNet/ShreddingNet are not exact reproductions",
        "Exact-six synthetic direct placement readout",
        "Real strict547 exact-six direct placement/translation GT",
        "This is the formal exact-six corruption comparison",
        "Recall@2",
        "Recall@10",
        "Frozen-threshold assembly-edge precision/recall/F1",
        "P@2",
        "F1@10",
        "Pairing-compatible registration under corruption",
        "Mean eRMSE",
        "no native PairingNet GA or ShreddingNet CM/FM/SE/GA claim",
        "the added pair head is not an official component",
        "balanced selected-list diagnostics are not native CM/FM/SE",
        "a morphology-only review inspected 500 synthetic-test masks",
        "Frozen architecture and Rachel data provenance",
        "Rachel RGB JPEG plus colocated label.csv only",
        "formal test manifest content was not read or hashed",
    }
    assert all(text in markdown for text in required)
    assert hashlib.sha256(markdown.encode("utf-8")).hexdigest() == (
        "1e6d3a5f5a8347d452634aadba58b1f01930f03e68ea62841819b26240bbf86f"
    )
    summary = json.loads(
        (output / "final_report_summary.json").read_text(encoding="utf-8")
    )
    declared = summary.pop("content_sha256")
    assert declared == hashlib.sha256(_canonical(summary)).hexdigest()
    assert summary["integrity"][
        "raw_pair_score_jsonl_sha256_verified_without_parsing"
    ] is True
    assert summary["integrity"]["raw_pair_score_jsonl_parsed"] is False
    receipt = json.loads(
        (output / "completion_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "complete_verified_final_report_render"
    assert receipt["completion_receipt_published_last"] is True


def test_report_output_is_no_clobber(tmp_path: Path):
    root = _normalization_root(tmp_path)
    output = tmp_path / "report"
    inputs = MODULE._load_aggregate_inputs(root)
    MODULE._render_and_publish(inputs, output)
    report_before = (output / "final_report.md").read_bytes()
    with pytest.raises(MODULE.FinalReportError, match="fresh"):
        MODULE._render_and_publish(inputs, output)
    assert (output / "final_report.md").read_bytes() == report_before


def test_report_output_rejects_final_root_nesting_and_symlink_parent(tmp_path: Path):
    root = _normalization_root(tmp_path)
    inputs = MODULE._load_aggregate_inputs(root)
    with pytest.raises(MODULE.FinalReportError, match="nested in final root"):
        MODULE._render_and_publish(inputs, root / "report")
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    link_parent = tmp_path / "link-parent"
    try:
        link_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem does not support symlinks")
    with pytest.raises(MODULE.FinalReportError, match="non-symlink"):
        MODULE._render_and_publish(inputs, link_parent / "report")


def test_publication_failure_cleans_partial_and_new_output(tmp_path: Path, monkeypatch):
    output = tmp_path / "report"
    original = os.link
    calls = 0

    def fail_first_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected link failure")
        return original(source, destination)

    monkeypatch.setattr(os, "link", fail_first_link)
    with pytest.raises(OSError, match="injected"):
        MODULE._publish_report_directory(output, b"md", b"json", b"receipt")
    assert not output.exists()
    assert list(tmp_path.glob(".report.partial-*")) == []


def test_completion_receipt_is_linked_last(tmp_path: Path, monkeypatch):
    output = tmp_path / "report"
    original = os.link
    destinations = []

    def record(source, destination):
        destinations.append(Path(destination).name)
        return original(source, destination)

    monkeypatch.setattr(os, "link", record)
    MODULE._publish_report_directory(output, b"md", b"json", b"receipt")
    assert destinations == [
        "final_report.md",
        "final_report_summary.json",
        "completion_receipt.json",
    ]


def _resign_gate(root: Path, logical: str, mutate):
    path = root / logical
    gate = json.loads(path.read_text(encoding="utf-8"))
    gate.pop("content_sha256", None)
    mutate(gate)
    _write_json(path, _with_content_sha(gate))
    _reseal_complete_root(root)


def test_terminal_requires_exactly_all_nine_gate_bindings(tmp_path: Path):
    root = _aggregate_completed_root(tmp_path)
    terminal_path = root / MODULE.TERMINAL_RELATIVE
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal.pop("content_sha256")
    terminal["gate_receipts"] = terminal["gate_receipts"][:-1]
    _write_json(terminal_path, _with_content_sha(terminal))
    with pytest.raises(MODULE.FinalReportError, match="exact nine"):
        MODULE._load_aggregate_inputs(root)


def test_freeze_gate_authority_tamper_fails_even_when_resigned(tmp_path: Path):
    root = _aggregate_completed_root(tmp_path)
    _resign_gate(
        root,
        "control/train_validation_freeze_gate.json",
        lambda gate: gate["n512"].__setitem__("receipt_sha256", "f" * 64),
    )
    with pytest.raises(MODULE.FinalReportError, match="terminal frozen training authority"):
        MODULE._load_aggregate_inputs(root)


def test_complete_canonical_model_config_tamper_fails_when_resigned(tmp_path: Path):
    root = _aggregate_completed_root(tmp_path)

    def mutate(gate):
        gate["n512"]["canonical_model_and_loss_config_authority"][
            "model_config"
        ]["contour_cap"] = 511

    _resign_gate(root, "control/train_validation_freeze_gate.json", mutate)
    with pytest.raises(MODULE.FinalReportError, match="complete canonical authority"):
        MODULE._load_aggregate_inputs(root)


def test_rachel_selection_provenance_tamper_fails_when_resigned(tmp_path: Path):
    root = _aggregate_completed_root(tmp_path)

    def mutate(gate):
        gate["rachel_data_provenance"]["selection_contract"][
            "negative_origin_counts_per_split"
        ]["train"]["same_folder_hard"] = 5999

    _resign_gate(root, "control/train_validation_freeze_gate.json", mutate)
    with pytest.raises(MODULE.FinalReportError, match="provenance authority"):
        MODULE._load_aggregate_inputs(root)


def test_source_freeze_missing_required_runtime_file_fails_when_resigned(tmp_path: Path):
    root = _aggregate_completed_root(tmp_path)

    def mutate(gate):
        gate["files"] = gate["files"][1:]
        gate["file_count"] = len(gate["files"])

    _resign_gate(root, "control/source_code_freeze.json", mutate)
    with pytest.raises(MODULE.FinalReportError, match="source-code freeze authority"):
        MODULE._load_aggregate_inputs(root)


@pytest.mark.parametrize(
    "logical,disclosure_path",
    [
        (
            MODULE.SYNTHETIC_STATS_RELATIVE,
            ("protocol", "automated_no_test_or_real_tuning", "evaluation_history_disclosure"),
        ),
        (
            MODULE.REAL_STATS_RELATIVE,
            ("protocol", "automated_no_test_or_real_tuning", "evaluation_history_disclosure"),
        ),
        (
            "corrosion/corrosion-fixture/robustness_receipt.json",
            ("protocol", "evaluation_history_disclosure"),
        ),
        (
            MODULE.REAL_TRANSLATION_RELATIVE,
            ("protocol", "evaluation_history_disclosure"),
        ),
    ],
)
def test_each_aggregate_rejects_contradictory_morphology_review_when_resealed(
    tmp_path: Path, logical: str, disclosure_path
):
    root = _normalization_root(tmp_path)
    path = root / logical
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("content_sha256", None)
    disclosure = document
    for key in disclosure_path:
        disclosure = disclosure[key]
    disclosure["prior_convergence_time_synthetic_test_mask_sample_count"] = 499
    if logical == MODULE.REAL_TRANSLATION_RELATIVE:
        document = _with_content_sha(document)
    _write_json(path, document)
    _reseal_complete_root(root)
    with pytest.raises(MODULE.FinalReportError, match="mask_sample_count differs"):
        MODULE._normalize_aggregates(MODULE._load_aggregate_inputs(root))


def test_missing_morphology_disclosure_field_fails_closed(tmp_path: Path):
    root = _normalization_root(tmp_path)
    path = root / MODULE.SYNTHETIC_STATS_RELATIVE
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["protocol"]["automated_no_test_or_real_tuning"][
        "evaluation_history_disclosure"
    ]["prior_convergence_time_synthetic_test_labels_read"]
    _write_json(path, document)
    _reseal_complete_root(root)
    with pytest.raises(MODULE.FinalReportError, match="labels_read differs"):
        MODULE._normalize_aggregates(MODULE._load_aggregate_inputs(root))


def test_prepublication_reload_detects_authenticated_source_change(tmp_path: Path):
    root = _normalization_root(tmp_path)
    inputs = MODULE._load_aggregate_inputs(root)
    path = root / MODULE.SYNTHETIC_STATS_RELATIVE
    document = json.loads(path.read_text(encoding="utf-8"))
    document["protocol"]["automated_no_test_or_real_tuning"][
        "fixture_authenticated_extra"
    ] = "changed-after-first-load"
    _write_json(path, document)
    _reseal_complete_root(root)
    output = tmp_path / "must-not-publish"
    with pytest.raises(
        MODULE.FinalReportError,
        match="authority snapshot changed|SHA-256 differs",
    ):
        MODULE._render_and_publish(inputs, output)
    assert not output.exists()


def test_prepublication_rejects_resigned_gate_inventory_terminal_with_same_projection(
    tmp_path: Path,
):
    root = _normalization_root(tmp_path)
    inputs = MODULE._load_aggregate_inputs(root)
    first_summary = MODULE._normalize_aggregates(inputs)
    _resign_gate(
        root,
        "control/real_stats_gate.json",
        lambda gate: gate.__setitem__("authenticated_noop_extra", "changed"),
    )
    reloaded = MODULE._load_aggregate_inputs(root)
    second_summary = MODULE._normalize_aggregates(reloaded)
    first_without_snapshot = json.loads(json.dumps(first_summary))
    second_without_snapshot = json.loads(json.dumps(second_summary))
    del first_without_snapshot["integrity"]["complete_authority_snapshot_sha256"]
    del second_without_snapshot["integrity"]["complete_authority_snapshot_sha256"]
    assert first_without_snapshot == second_without_snapshot
    output = tmp_path / "must-not-publish-resigned"
    with pytest.raises(
        MODULE.FinalReportError,
        match="authority snapshot changed|SHA-256 differs",
    ):
        MODULE._render_and_publish(inputs, output)
    assert not output.exists()


def test_prepublication_rejects_same_size_in_place_jsonl_change(
    tmp_path: Path,
):
    root = _normalization_root(tmp_path)
    inputs = MODULE._load_aggregate_inputs(root)
    path = root / "sealed/test-fixture/coarse_only/pair_scores.jsonl"
    original = path.read_bytes()
    changed = original.replace(b"true", b"null")
    assert changed != original and len(changed) == len(original)
    before = path.stat()
    path.write_bytes(changed)
    os.utime(
        path,
        ns=(before.st_atime_ns, max(before.st_mtime_ns + 1_000_000, path.stat().st_mtime_ns)),
    )
    output = tmp_path / "must-not-publish-jsonl-change"
    with pytest.raises(
        MODULE.FinalReportError,
        match="authority snapshot changed|SHA-256 differs",
    ):
        MODULE._render_and_publish(inputs, output)
    assert not output.exists()


def test_initial_load_rejects_same_size_jsonl_tamper_against_inventory(
    tmp_path: Path,
):
    root = _normalization_root(tmp_path)
    path = root / "sealed/test-fixture/coarse_only/pair_scores.jsonl"
    original = path.read_bytes()
    changed = original.replace(b"true", b"null")
    assert changed != original and len(changed) == len(original)
    path.write_bytes(changed)
    with pytest.raises(MODULE.FinalReportError, match="SHA-256 differs"):
        MODULE._load_aggregate_inputs(root)


def test_unexpected_regular_file_fails_full_tree_coverage(tmp_path: Path):
    root = _aggregate_completed_root(tmp_path)
    (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(MODULE.FinalReportError, match="file set differs"):
        MODULE._load_aggregate_inputs(root)


def test_missing_inventory_row_fails_full_tree_coverage(tmp_path: Path):
    root = _aggregate_completed_root(tmp_path)

    def mutate(inventory):
        inventory["files"] = [
            row
            for row in inventory["files"]
            if row["path"]
            != "sealed/test-fixture/coarse_only/pair_scores.jsonl"
        ]
        inventory["file_count"] = len(inventory["files"])

    _rebind_inventory(root, mutate)
    with pytest.raises(MODULE.FinalReportError, match="file set differs"):
        MODULE._load_aggregate_inputs(root)


def test_symlink_and_special_file_fail_full_tree_coverage(tmp_path: Path):
    root = _aggregate_completed_root(tmp_path)
    target = root / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = root / "forbidden-link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("filesystem does not support symlinks")
    with pytest.raises(MODULE.FinalReportError, match="symlinks are forbidden"):
        MODULE._load_aggregate_inputs(root)
    link.unlink()
    target.unlink()
    fifo = root / "forbidden-fifo"
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError):
        pytest.skip("filesystem does not support FIFOs")
    with pytest.raises(MODULE.FinalReportError, match="special files are forbidden"):
        MODULE._load_aggregate_inputs(root)

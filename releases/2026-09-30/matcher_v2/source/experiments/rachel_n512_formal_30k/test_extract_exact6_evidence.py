from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.rachel_n512_formal_30k import extract_exact6_evidence as target


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "reports" / "pairwise_exact6_20260906"


def _load(name: str):
    path = EVIDENCE / name
    return json.loads(path.read_text(encoding="utf-8")), hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _coverage(total: int, positive: int):
    return {
        "record_count": total,
        "valid_count": total,
        "valid_fraction": 1.0,
        "positive_count": positive,
        "valid_positive_count": positive,
        "negative_count": total - positive,
        "valid_negative_count": total - positive,
    }


def _real_view(total: int, positive: int, base: float):
    ranking = {
        "row": {"auroc": base, "auprc": base - 0.01},
        "case_cluster_balanced": {"auroc": base + 0.01, "auprc": base},
    }
    thresholded = {
        "row": {"f1": base - 0.02, "recall": base - 0.03},
        "case_cluster_balanced": {"f1": base - 0.01, "recall": base - 0.02},
    }
    return {
        "coverage": _coverage(total, positive),
        "ranking_primary_threshold_free": ranking,
        "frozen_validation_threshold_secondary": thresholded,
    }


def _real_population(total: int, positive: int, status: str):
    methods = {}
    for index, method in enumerate(target.METHODS):
        base = 0.70 + index * 0.02
        native_key = (
            "pair_classification_diagnostic"
            if method in {"pairingnet_adapted", "shreddingnet_adapted"}
            else "native"
        )
        methods[method] = {
            native_key: _real_view(total, positive, base),
            "all_method_common_valid": _real_view(total, positive, base),
        }
    return {
        "schema_version": "rachel-n512-real-external/1.0",
        "status": status,
        "dataset": {
            "pair_count": total,
            "positive_count": positive,
            "negative_count": total - positive,
            **(
                {
                    "strict_prefix_count": 547,
                    "constructed_count": 469,
                    "strict_prefix_preserved_exactly": True,
                }
                if total == 1016
                else {}
            ),
        },
        "protocol": {
            "formal_evaluation": True,
            "compatibility_mode": False,
            "formal_method_inventory": list(target.METHODS),
            "same_data_benchmark_exact_reproduction_claimed": False,
        },
        "all_method_common_valid": _coverage(total, positive),
        "methods": methods,
    }


def _real_fixture():
    return {
        "schema_version": "rachel-n512-real-external/1.0",
        "status": "complete_strict_and_balanced_single_forward_population",
        "forward_contract": {
            "methods": list(target.METHODS),
            "forward_pair_count_per_arm": 1016,
            "strict_547_derived_from_exact_prediction_prefix": True,
            "strict_pairs_forwarded_twice": False,
        },
        "protocol": {
            "formal_evaluation": True,
            "compatibility_mode": False,
            "formal_exact_six_frozen_before_real_open": True,
            "formal_method_inventory": list(target.METHODS),
            "same_data_benchmark_exact_reproduction_claimed": False,
        },
        "strict_547": _real_population(
            547, 508, "complete_strict_547_target_blind_external_test"
        ),
        "balanced_1016": _real_population(
            1016, 508, "complete_balanced_1016_target_blind_external_test"
        ),
    }


def _interval(point: float, *, delta: bool = False):
    result = {
        "point_estimate": point,
        "percentile_95_ci": [point - 0.01, point + 0.01],
        "bootstrap_mean": point,
        "bootstrap_standard_error": 0.005,
        "valid_replicates": 20_000,
    }
    if delta:
        result["probability_delta_gt_zero"] = 0.99
    return result


def _bootstrap_fixture(source_sha: str, source_kind: str, total: int, positive: int):
    points = {
        method: {"auroc": 0.70 + index * 0.02, "auprc": 0.69 + index * 0.02}
        for index, method in enumerate(target.METHODS)
    }
    metrics = {}
    for metric in ("auroc", "auprc"):
        methods = {
            method: _interval(points[method][metric]) for method in target.METHODS
        }
        deltas = {}
        for method in target.METHODS:
            if method == "full_n512":
                continue
            point = points["full_n512"][metric] - points[method][metric]
            deltas["full_n512_minus_" + method] = _interval(point, delta=True)
        metrics[metric] = {"methods": methods, "paired_deltas": deltas}
    return {
        "schema_version": "rachel-n512-paired-endpoint-bootstrap/1.1",
        "status": "complete_formal_exact_six",
        "formal_evaluation": True,
        "compatibility_mode": False,
        "methods": list(target.METHODS),
        "input": {"source_kind": source_kind, "source_sha256": source_sha},
        "protocol": {"replicates": 20_000, "seed": 20_260_901},
        "coverage": {"all_method_common_valid": _coverage(total, positive)},
        "point_metrics": {
            "row_count": total,
            "positive_count": positive,
            "negative_count": total - positive,
            "methods": points,
        },
        "paired_endpoint_pigeonhole_bootstrap": {
            "replicates_requested": 20_000,
            "valid_replicates": 20_000,
            "skipped_single_class_replicates": 0,
            "metrics": metrics,
        },
    }


def _translation_fixture(real_sha: str):
    metric_names = {
        "valid_translation_prediction_fraction",
        "median_l2_px",
        "p90_l2_px",
        *("recall_at_{}px".format(value) for value in target.TOLERANCES),
        *(
            "joint_frozen_threshold_and_translation_recall_at_{}px".format(value)
            for value in target.TOLERANCES
        ),
    }
    methods = {}
    for method in target.METHODS:
        if method not in target.TRANSLATION_METHODS:
            methods[method] = {
                "status": "not_applicable",
                "translation_metrics": None,
            }
            continue
        estimates = {
            name: (4.0 if name in {"median_l2_px", "p90_l2_px"} else 0.8)
            for name in metric_names
        }
        intervals = {
            name: {
                "estimate": estimate,
                "ci95_low": estimate - 0.01,
                "ci95_high": estimate + 0.01,
                "valid_bootstrap_replicates": 20_000,
            }
            for name, estimate in estimates.items()
        }
        methods[method] = {
            "status": "evaluated_positive_translation_gt",
            "eligible_positive_count": 508,
            "point_estimates": estimates,
            "case_bootstrap": {
                "schema": "case_cluster_percentile_bootstrap/1.0",
                "repetitions": 20_000,
                "sampling_unit": "authoritative_real_case_uid",
                "metrics": intervals,
            },
            "assembly_edge_strict_547": {"by_tolerance": {}},
            "assembly_edge_balanced_1016_selected_list_diagnostic": {
                "by_tolerance": {}
            },
            "pairingnet_style_registration": {"rr_lt4": 0.8},
            "correspondence": {"status": "not_applicable"},
        }
    value = {
        "schema_version": "rachel-n512-real-translation-gt-posteval/1.0",
        "status": "complete_postprediction_real_positive_translation_gt",
        "formal_evaluation": True,
        "compatibility_mode": False,
        "source_pair_only_evaluation": {
            "file_sha256_frozen_before_gt_open": real_sha,
            "exact_method_order": list(target.METHODS),
        },
        "population": {"strict_pair_count": 547, "positive_pair_count": 508},
        "method_metrics": methods,
    }
    value["content_sha256"] = hashlib.sha256(target._canonical_bytes(value)).hexdigest()
    return value


def test_current_completed_stage_receipts_extract():
    synthetic, synthetic_sha = _load("synthetic_test_receipt.json")
    corrosion, _ = _load("corrosion_robustness_receipt.json")

    synthetic_out = target.extract_synthetic(synthetic)
    corrosion_out = target.extract_corrosion(corrosion, synthetic_sha)

    assert synthetic_out["population"]["common_valid"]["valid_count"] == 3000
    assert corrosion_out["population"]["valid_count"] == 1822
    assert corrosion_out["classification_bootstrap"]["valid_replicates"] == 19_999


def test_current_real_pair_only_receipt_extracts_on_common_valid_population():
    real, _ = _load("real_pair_only.json")

    output = target.extract_real(real)

    assert output["strict547_descriptive"]["common_valid"]["valid_count"] == 547
    assert output["balanced1016_inferential"]["common_valid"]["valid_count"] == 1016
    assert set(output["balanced1016_inferential"]["methods"]) == set(target.METHODS)


def test_current_resource_summary_extracts_with_source_hashes():
    resources, _ = _load("resource_summary.json")

    output = target.extract_resources(resources)

    assert set(output["methods"]) == set(target.TRANSLATION_METHODS)
    assert output["declared_source_sha256_count"] >= 3
    concurrency = output["concurrency_assessment"]
    assert concurrency["pairingnet_plus_shreddingnet"]["headroom_gibibytes"] == pytest.approx(
        11.005859375
    )
    assert concurrency["all_three"]["capacity_deficit_mebibytes"] == 144
    assert concurrency["all_three"]["verdict"] == "not_suitable"


def test_current_paired_bootstraps_extract_and_bind_exact_source_files():
    synthetic, synthetic_sha = _load("synthetic_test_receipt.json")
    real, real_sha = _load("real_pair_only.json")
    synthetic_stats, _ = _load("synthetic_paired_endpoint_bootstrap_20000.json")
    real_stats, _ = _load("real_paired_endpoint_bootstrap_20000.json")

    synthetic_output = target._extract_ranking_bootstrap(
        synthetic_stats, synthetic_sha, "sealed_synthetic", 3000
    )
    real_output = target._extract_ranking_bootstrap(
        real_stats, real_sha, "real_balanced1016", 1016
    )

    assert synthetic_output["coverage"] == target.extract_synthetic(synthetic)[
        "population"
    ]["common_valid"]
    assert real_output["coverage"] == target.extract_real(real)[
        "balanced1016_inferential"
    ]["common_valid"]
    assert synthetic_output["valid_replicates"] == 20_000
    assert real_output["valid_replicates"] == 20_000


def test_build_summary_accepts_bound_fixtures_and_hashes_content():
    synthetic, synthetic_sha = _load("synthetic_test_receipt.json")
    corrosion, corrosion_sha = _load("corrosion_robustness_receipt.json")
    real = _real_fixture()
    real_sha = "a" * 64
    translation = _translation_fixture(real_sha)
    translation_sha = hashlib.sha256(target._canonical_bytes(translation)).hexdigest()
    synthetic_stats = _bootstrap_fixture(
        synthetic_sha, "sealed_synthetic", 3000, 1500
    )
    real_stats = _bootstrap_fixture(real_sha, "real_balanced1016", 1016, 508)

    result = target.build_summary(
        synthetic=synthetic,
        synthetic_sha=synthetic_sha,
        corrosion=corrosion,
        corrosion_sha=corrosion_sha,
        real=real,
        real_sha=real_sha,
        translation=translation,
        translation_sha=translation_sha,
        synthetic_bootstrap=synthetic_stats,
        synthetic_bootstrap_sha="b" * 64,
        real_bootstrap=real_stats,
        real_bootstrap_sha="c" * 64,
    )

    content_sha = result["content_sha256"]
    body = dict(result)
    del body["content_sha256"]
    assert content_sha == hashlib.sha256(target._canonical_bytes(body)).hexdigest()
    assert result["method_display_names"]["pairingnet_adapted"] == "PairingNet-adapted"
    assert result["claim_limits"][
        "same_data_benchmarks_are_adaptations_not_exact_reproductions"
    ] is True


def test_rejects_common_valid_population_drift():
    synthetic, synthetic_sha = _load("synthetic_test_receipt.json")
    corrosion, _ = _load("corrosion_robustness_receipt.json")
    corrosion = copy.deepcopy(corrosion)
    corrosion["summary"]["conditions"]["clean"]["methods"]["full_n512"][
        "fixed_all_method_all_condition_common_valid"
    ]["coverage"]["valid_count"] = 1821

    with pytest.raises(target.EvidenceExtractionError, match="coverage differs"):
        target.extract_corrosion(corrosion, synthetic_sha)


def test_rejects_unadapted_method_alias():
    synthetic, _ = _load("synthetic_test_receipt.json")
    synthetic["protocol"]["formal_method_inventory"][4] = "pairingnet"

    with pytest.raises(target.EvidenceExtractionError, match="method order differs"):
        target.extract_synthetic(synthetic)


def test_rejects_translation_real_file_hash_mismatch():
    translation = _translation_fixture("a" * 64)

    with pytest.raises(target.EvidenceExtractionError, match="not bound"):
        target.extract_translation(translation, "d" * 64)


def test_read_json_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")

    with pytest.raises(target.EvidenceExtractionError, match="duplicate JSON key"):
        target._read_json(path, "duplicate fixture")

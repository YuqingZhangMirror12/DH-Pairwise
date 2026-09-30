from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "experiments"
    / "exact_seam_scale6964_results"
    / "cross_seed_paired_cluster_bootstrap.py"
)
SPEC = importlib.util.spec_from_file_location("cross_seed_bootstrap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _document(seed: str) -> dict:
    labels = [True, True, True, True, False, False, False, False]
    strata = [
        "strict_manifest_positive",
        "strict_manifest_positive",
        "strict_manifest_positive",
        "strict_manifest_positive",
        "strict_manifest_negative",
        "strict_manifest_negative",
        "constructed_distractor_not_gt_negative",
        "constructed_distractor_not_gt_negative",
    ]
    directions = ["right", "right", "below", "above", None, None, None, None]
    clusters = [
        "case-0",
        "case-0",
        "case-1",
        "case-1",
        "case-2",
        "case-2",
        "case-3",
        "case-3",
    ]
    if seed == "260830":
        scores = {
            "exact": [0.95, 0.90, 0.85, 0.80, 0.25, 0.20, 0.15, 0.10],
            "weak": [0.80, 0.75, 0.55, 0.45, 0.60, 0.40, 0.35, 0.30],
            "siamese": [0.70, 0.65, 0.40, 0.35, 0.75, 0.55, 0.50, 0.25],
        }
        predictions = {
            "exact": ["right", "right", "below", "above"],
            "weak": ["right", "right", "above", "below"],
        }
    else:
        scores = {
            "exact": [0.92, 0.87, 0.82, 0.70, 0.30, 0.22, 0.18, 0.12],
            "weak": [0.78, 0.67, 0.62, 0.48, 0.65, 0.52, 0.38, 0.28],
            "siamese": [0.68, 0.60, 0.52, 0.30, 0.72, 0.58, 0.44, 0.20],
        }
        predictions = {
            "exact": ["below", "right", "below", "right"],
            "weak": ["right", "below", "below", "below"],
        }

    rows = []
    for index, label in enumerate(labels):
        methods = {}
        for method in MODULE.METHODS:
            valid = not (
                (seed == "260831" and index == 0 and method == "exact")
                or (seed == "260830" and index == 7 and method == "siamese")
            )
            predicted = (
                predictions[method][index] if label and method in predictions else None
            )
            methods[method] = {
                "valid": valid,
                "probability": scores[method][index],
                "predicted_direction": predicted,
            }
        rows.append(
            {
                "pair_id": f"pair-{index}",
                "label": label,
                "cluster_id": clusters[index],
                "stratum": strata[index],
                "true_direction": directions[index],
                "methods": methods,
            }
        )
    return {
        "evaluated_methods": list(MODULE.METHODS),
        "evaluated_pair_count": len(rows),
        "pairs": rows,
    }


def _write(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_cross_seed_analysis_freezes_intersection_and_pairs_bootstrap(tmp_path: Path):
    first = _write(tmp_path / "seed260830.json", _document("260830"))
    second = _write(tmp_path / "seed260831.json", _document("260831"))

    result = MODULE.analyze(
        [("260830", first), ("260831", second)],
        replicates=500,
        master_seed=11,
    )

    assert result["alignment"]["global_common_valid_pair_ids"] == [
        "pair-1",
        "pair-2",
        "pair-3",
        "pair-4",
        "pair-5",
        "pair-6",
    ]
    assert (
        result["coverage"]["balanced1016"]["all_seed_all_method_common_valid_count"]
        == 6
    )
    balanced = result["balanced1016"]
    assert balanced["common_valid_point_metrics"]["positive_count"] == 3
    assert balanced["common_valid_point_metrics"]["negative_count"] == 3
    bootstrap = balanced["paired_cluster_bootstrap"]
    assert bootstrap["sampling_cluster_count"] == 4
    comparison = bootstrap["comparisons"]["exact_minus_siamese"]["auroc"]
    assert comparison["all_seed_point_deltas_positive"] is True
    assert comparison["cross_seed_mean"]["valid_replicates"] > 0

    direction = result["direction"]
    assert direction["row_count"] == 3
    assert direction["by_seed"]["260830"]["exact_accuracy"] == 1.0
    assert direction["by_seed"]["260831"]["exact_accuracy"] == pytest.approx(2 / 3)
    assert direction["all_seed_point_deltas_positive"] is True
    assert direction["cross_seed_prediction_agreement"]["exact"][
        "all_seed_prediction_agreement_fraction"
    ] == pytest.approx(2 / 3)
    assert result["strict547_descriptive"]["role"].startswith("descriptive")


def test_cross_seed_analysis_rejects_changed_pair_semantics(tmp_path: Path):
    first_document = _document("260830")
    second_document = deepcopy(_document("260831"))
    second_document["pairs"][3]["cluster_id"] = "different-case"
    first = _write(tmp_path / "first.json", first_document)
    second = _write(tmp_path / "second.json", second_document)

    with pytest.raises(ValueError, match="cluster_id differs for pair_id=pair-3"):
        MODULE.analyze([("260830", first), ("260831", second)], replicates=10)

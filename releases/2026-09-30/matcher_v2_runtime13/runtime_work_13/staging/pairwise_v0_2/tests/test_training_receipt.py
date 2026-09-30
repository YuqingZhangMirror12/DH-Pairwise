import json
from pathlib import Path

import pytest

from staging.pairwise_v0_2.pairwise_data.training_receipt import (
    TrainingReceiptError,
    _historical_split_audit,
    _nonportable_pointers,
    build_training_receipt,
)


STAGING = Path(__file__).resolve().parents[2]
V01_SPLIT = STAGING / "pairwise_v0_1/data_gate/split_candidate_70_15_15.json"
SYNTHETIC_DIR = (
    STAGING / "pairwise_v0_2/manifests/dunhuang_pairwise_mask_subset_v0_2/manifest"
)
PORTABLE_DIR = STAGING / "pairwise_v0_2/pairwise_data/train_val_v0_2"


def _load(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def test_committed_summary_has_real_cross_dataset_totals():
    summary = _load(PORTABLE_DIR / "data_summary.json")

    assert summary["by_split"]["train"] == {
        "component_count": 7697,
        "group_count": 19749,
        "negative_count": 981713,
        "positive_count": 310096,
        "usable_pair_count": 1291809,
    }
    assert summary["by_split"]["val"] == {
        "component_count": 588,
        "group_count": 3025,
        "negative_count": 209790,
        "positive_count": 63256,
        "usable_pair_count": 273046,
    }
    synthetic = summary["datasets"]["dunhuang_voronoi_masks_no_erode_v0_2"]
    assert synthetic["train"]["group_count"] == 4992
    assert synthetic["train"]["usable_pair_count"] == 17432
    assert synthetic["val"]["usable_pair_count"] == 0


def test_committed_receipt_authorizes_training_but_seals_external_test():
    receipt = _load(PORTABLE_DIR / "split_receipt.json")

    assert receipt["status"] == "authorized_ready_for_model_training"
    assert (
        receipt["authorization"]["status"]
        == "authorized_for_model_training_by_current_user_instruction"
    )
    assert (
        receipt["historical_split_audit"]["overlap_counts"][
            "train_val_component_overlap"
        ]
        == 0
    )
    assert receipt["synthetic_split_audit"]["counts"]["quarantine_group_count"] == 8
    assert (
        receipt["real_dunhuang_external_test"]["referenced_by_training_receipt"]
        is False
    )
    assert _nonportable_pointers(receipt) == []


def test_committed_stream_audit_matches_actual_emission_and_is_portable():
    audit = _load(PORTABLE_DIR / "stream_audit.json")

    assert audit["status"] == "pass"
    assert audit["splits"]["train"]["counts"] == {
        "emitted_component_count": 7694,
        "emitted_group_count": 19733,
        "negative_count": 981713,
        "pair_count": 1291809,
        "positive_count": 310096,
    }
    assert audit["splits"]["val"]["counts"] == {
        "emitted_component_count": 588,
        "emitted_group_count": 3024,
        "negative_count": 209790,
        "pair_count": 273046,
        "positive_count": 63256,
    }
    assert audit["splits"]["train"]["assigned_but_zero_emission_group_count"] == 16
    assert audit["splits"]["val"]["assigned_but_zero_emission_group_count"] == 1
    assert audit["checks"][-1] == {
        "check_id": "actual_train_val_component_overlap",
        "expected": 0,
        "observed": 0,
        "status": "pass",
    }
    assert audit["real_dunhuang_sealed_test_record_count"] == 0
    assert _nonportable_pointers(audit) == []


def test_receipt_rebuild_is_deterministic(tmp_path):
    receipt_path, summary_path = build_training_receipt(
        historical_split_path=V01_SPLIT,
        synthetic_manifest_path=SYNTHETIC_DIR / "synthetic_groups.jsonl",
        synthetic_summary_path=SYNTHETIC_DIR / "synthetic_summary.json",
        output_dir=tmp_path,
    )

    assert (
        receipt_path.read_bytes() == (PORTABLE_DIR / "split_receipt.json").read_bytes()
    )
    assert (
        summary_path.read_bytes() == (PORTABLE_DIR / "data_summary.json").read_bytes()
    )


def test_historical_audit_rejects_component_cross_split_mismatch():
    payload = {
        "schema_version": "pairwise-real-data-gate/0.1",
        "candidate_id": "fixture",
        "seed": "fixture",
        "ratios": {"train": 0.5, "val": 0.5, "test": 0},
        "status": "complete",
        "authorization": "fixture",
        "assignments": {
            "groups": {
                "mm/base/root/source/a": "train",
                "mm/base/root/source/b": "val",
            },
            "group_components": {
                "mm/base/root/source/a": "mm/source/source",
                "mm/base/root/source/b": "mm/source/source",
            },
            "components": {"mm/source/source": "train"},
        },
    }

    with pytest.raises(TrainingReceiptError, match="split mismatch"):
        _historical_split_audit(payload)

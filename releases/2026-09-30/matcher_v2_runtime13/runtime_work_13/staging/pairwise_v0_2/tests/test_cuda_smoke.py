import json
from copy import deepcopy

import pytest

from staging.pairwise_v0_2.training.cuda_smoke import (
    CudaSmokeError,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_MANIFEST_SHA256,
    SMOKE_SCHEMA_VERSION,
    _aggregate_transport,
    _collect_balanced_candidates,
    _summary,
    validate_receipt,
)


class _Binding:
    sha256 = EXPECTED_ARCHIVE_SHA256


class _Fragment:
    binding = _Binding()


class _Record:
    def __init__(self, label):
        self.label = label
        self.split = "train"
        self.dataset_id = "dunhuang_voronoi_masks_no_erode_v0_2"
        self.fragment_a = _Fragment()
        self.fragment_b = _Fragment()
        self.provenance = {
            "source_lineage_status": "unavailable_training_only",
            "real_dunhuang_sealed_test": False,
        }


def _safe_receipt():
    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "status": "passed",
        "scope": {
            "new_synthetic_train_only": True,
            "historical_data_used": False,
            "historical_test_used": False,
            "real_dunhuang_data_used": False,
            "checkpoint_written": False,
            "sample_identifiers_persisted": False,
            "pixels_persisted": False,
        },
        "hashes": {
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "source_tree_sha256": "a" * 64,
            "model_config_sha256": "b" * 64,
            "geometry_config_sha256": "c" * 64,
        },
        "batch_aggregate": {
            "pair_count": 2,
            "positive_count": 1,
            "negative_count": 1,
        },
        "execution": {
            "loss": {"initial": 1.0, "final": 0.5},
            "sinkhorn": {"all_converged": True, "row_residual_max": 1e-6},
        },
        "results": {
            "finite_loss": True,
            "finite_nonzero_gradients": True,
            "all_pair_outputs_valid": True,
            "final_loss_lower_than_initial": True,
        },
    }


def test_balanced_prefix_selection_is_deterministic_and_fail_closed():
    records = [_Record(True), _Record(True), _Record(False), _Record(False)]
    positive, negative = _collect_balanced_candidates(records, per_class_limit=2)
    assert len(positive) == 2
    assert len(negative) == 2
    assert all(record.label for record in positive)
    assert not any(record.label for record in negative)

    escaped = _Record(True)
    escaped.provenance = dict(escaped.provenance, real_dunhuang_sealed_test=True)
    with pytest.raises(CudaSmokeError, match="train-only"):
        _collect_balanced_candidates([escaped, _Record(False)], per_class_limit=1)


def test_numeric_summaries_require_finite_nonempty_observations():
    summary = _summary([1.0, 0.75, 0.25])
    assert summary["initial"] == 1.0
    assert summary["final"] == 0.25
    assert summary["minimum"] == 0.25
    with pytest.raises(CudaSmokeError, match="non-empty and finite"):
        _summary([])
    with pytest.raises(CudaSmokeError, match="non-empty and finite"):
        _summary([float("nan")])

    transport = _aggregate_transport(
        [
            {
                "transport_count": 8,
                "row_residual_max": 1e-7,
                "col_residual_max": 2e-7,
                "iteration_min": 50,
                "iteration_max": 50,
            },
            {
                "transport_count": 8,
                "row_residual_max": 3e-7,
                "col_residual_max": 1e-7,
                "iteration_min": 50,
                "iteration_max": 50,
            },
        ]
    )
    assert transport["all_converged"] is True
    assert transport["row_residual_max"] == pytest.approx(3e-7)


def test_receipt_is_json_safe_and_security_scan_rejects_sensitive_detail():
    receipt = _safe_receipt()
    validate_receipt(receipt)
    json.dumps(receipt, allow_nan=False)

    mutations = [
        ("runtime_path", "/root/private/input.zip"),
        ("sample_id", "synthetic/group/1"),
        ("archive_member", "output/mask.png"),
        ("secret", "never-write-this"),
        ("pixels", [0, 1, 1, 0]),
    ]
    for key, value in mutations:
        unsafe = deepcopy(receipt)
        unsafe[key] = value
        with pytest.raises(CudaSmokeError):
            validate_receipt(unsafe)


def test_receipt_requires_every_execution_gate_and_train_only_scope():
    unsafe = _safe_receipt()
    unsafe["results"]["finite_nonzero_gradients"] = False
    with pytest.raises(CudaSmokeError, match="smoke gates"):
        validate_receipt(unsafe)

    unsafe = _safe_receipt()
    unsafe["scope"]["historical_test_used"] = True
    with pytest.raises(CudaSmokeError, match="train-only isolation"):
        validate_receipt(unsafe)

    unsafe = _safe_receipt()
    unsafe["hashes"]["archive_sha256"] = "not-a-hash"
    with pytest.raises(CudaSmokeError, match="hashes"):
        validate_receipt(unsafe)

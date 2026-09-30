from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pytest

from staging.pairwise_v0_2.preflight import validate_c0_run_artifacts as validator
from staging.pairwise_v0_2.tests.test_c0_runner import _Fixture


_PROVIDER_LOGICAL_IDS_BY_ROLE = {
    "eccv_archive": "canonical://eccv_1113data/1113data",
    "mm_archive": "canonical://mm_augmented/dunhuang_augmented_data",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rehash_receipt(run_dir: Path, mutate) -> Dict[str, Any]:
    path = run_dir / "c0_run_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    mutate(receipt)
    receipt.pop("content_sha256", None)
    receipt["content_sha256"] = _content_sha256(receipt)
    _write_json(path, receipt)
    return receipt


def _patch_sidecar_paths(run_dir: Path) -> None:
    path = run_dir / "checkpoint_sidecar.local.json"
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    for row in sidecar["checkpoints"]:
        row["path"] = str((run_dir / Path(row["path"]).name).resolve())
    sidecar["winner_path"] = str(
        (run_dir / Path(sidecar["winner_path"]).name).resolve()
    )
    _write_json(path, sidecar)


def _prepare_fixture_artifacts(root: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    fixture = _Fixture(root)
    artifacts = fixture.run(root / "run")
    run_dir = artifacts.run_directory
    receipt_path = run_dir / "c0_run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    production_plan_path = (
        Path(__file__).resolve().parents[1] / "preflight" / "c0_run_plan.json"
    )
    plan = json.loads(production_plan_path.read_text(encoding="utf-8"))
    plan["contract"] = copy.deepcopy(receipt["contract"])
    plan.pop("content_sha256", None)
    plan["content_sha256"] = _content_sha256(plan)
    plan_path = root / "fixture_c0_run_plan.json"
    _write_json(plan_path, plan)
    plan_file_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    memo_bounds = {
        "max_batch_size": 2,
        "max_cache_bytes": 2 * 1024**2,
        "max_cached_fragments": 32,
    }
    counters = {
        "archive_decode_count": 16,
        "batch_count": 44,
        "coarse_preprocess_count": 16,
        "fragment_request_count": 176,
        "loader_cache_hit_count": 0,
        "mask_loader_call_count": 16,
        "memo_byte_count": 16 * 128 * 128 * 4,
        "memo_entry_count": 16,
        "memo_eviction_count": 0,
        "memo_hit_count": 160,
        "memo_miss_count": 16,
        "record_count": 88,
    }
    archive_roles = [
        row for row in plan["roles"] if row["role"] in {"mm_archive", "eccv_archive"}
    ]
    reconstructed_archive_locks = {
        "eccv_1113data": {
            "bytes": next(
                row["bytes"] for row in archive_roles if row["role"] == "eccv_archive"
            ),
            "format": "tar",
            "sha256": next(
                row["sha256"] for row in archive_roles if row["role"] == "eccv_archive"
            ),
        },
        "mm_augmented": {
            "bytes": next(
                row["bytes"] for row in archive_roles if row["role"] == "mm_archive"
            ),
            "format": "zip",
            "sha256": next(
                row["sha256"] for row in archive_roles if row["role"] == "mm_archive"
            ),
        },
    }
    receipt["locks"]["production_run_plan_file_sha256"] = plan_file_sha256
    receipt["locks"]["production_run_plan_content_sha256"] = plan["content_sha256"]
    freeze_role = next(row for row in plan["roles"] if row["role"] == "freeze_receipt")
    receipt["locks"]["freeze_file_sha256"] = freeze_role["sha256"]
    receipt["locks"]["freeze_content_sha256"] = plan["freeze_content_sha256"]
    receipt["provider"]["identity_index_content_sha256"] = plan["identity_index"][
        "content_sha256"
    ]
    receipt["provider"]["archive_locks_sha256"] = _content_sha256(
        reconstructed_archive_locks
    )
    receipt["provider"]["provider_version"] = validator.PROVIDER_ATTESTATION_VERSION
    receipt["provider"]["preprocessing_sha256"] = (
        validator.PRODUCTION_PREPROCESSING_SHA256
    )
    receipt["provider_final_evidence"] = {
        "archive_verification": [
            {
                "archive_format": "zip" if row["role"] == "mm_archive" else "tar",
                "byte_count": row["bytes"],
                "expected_sha256": row["sha256"],
                "logical_id": _PROVIDER_LOGICAL_IDS_BY_ROLE[row["role"]],
                "observed_sha256": row["sha256"],
                "verification_mode": "fixture_byte_hash",
            }
            for row in archive_roles
        ],
        "counters": counters,
        "freeze_content_sha256": plan["freeze_content_sha256"],
        "geometry_cache_local_sinkhorn_calls": 0,
        "historical_test_access": dict(validator._HISTORICAL_TEST_ACCESS),
        "identity_index_content_sha256": plan["identity_index"]["content_sha256"],
        "memo_bounds": memo_bounds,
        "preprocessing": copy.deepcopy(validator._PRODUCTION_PREPROCESSING),
        "provider_version": validator.PROVIDER_BASE_VERSION,
        "schema_version": validator.PROVIDER_RECEIPT_SCHEMA_VERSION,
        "sealed_real_read": False,
        "status": "archive_preflight_complete",
    }
    fixture_model_config = {"family": "tiny-c0-fixture", "width": 1}
    fixture_optimizer_config = {"kind": "none"}
    expected_checkpoint_config_hash = _content_sha256(
        {
            "runner": receipt["contract"],
            "provider": receipt["provider"],
            "backend": receipt["backend"],
            "environment": receipt["environment"],
            "model": fixture_model_config,
            "optimizer": fixture_optimizer_config,
        }
    )
    for epoch in receipt["epochs"]:
        epoch["checkpoint"]["config_hash"] = expected_checkpoint_config_hash
    receipt["winner"]["checkpoint"]["config_hash"] = expected_checkpoint_config_hash
    receipt.pop("content_sha256", None)
    receipt["content_sha256"] = _content_sha256(receipt)
    _write_json(receipt_path, receipt)

    expectations = {
        "expected_plan_file_sha256": plan_file_sha256,
        "expected_plan_content_sha256": plan["content_sha256"],
        "expected_contract": receipt["contract"],
        "expected_provider_counts": {
            "batch_count": 44,
            "record_count": 88,
            "fragment_request_count": 176,
        },
        "expected_memo_bounds": memo_bounds,
        "expected_backend_version": receipt["backend"]["backend_version"],
        "expected_backend_model_family": receipt["backend"]["model_family"],
        "expected_model_config": fixture_model_config,
        "expected_optimizer_config": fixture_optimizer_config,
        "expected_archive_role_to_provider_logical_id": dict(
            _PROVIDER_LOGICAL_IDS_BY_ROLE
        ),
        "expected_cluster_counts": {
            dataset: receipt["epochs"][0]["validation"]["by_dataset"][dataset][
                "cluster_count"
            ]
            for dataset in validator._DATASETS
        },
        "expected_train_order_commitment_sha256": receipt["metadata_attestation"][
            "train_order_commitment_sha256"
        ],
        "expected_validation_order_commitment_sha256": receipt["metadata_attestation"][
            "validation_order_commitment_sha256"
        ],
        "expected_preprocessing": copy.deepcopy(validator._PRODUCTION_PREPROCESSING),
        "expected_archive_evidence": {
            role: (
                "zip" if role == "mm_archive" else "tar",
                "fixture_byte_hash",
            )
            for role in _PROVIDER_LOGICAL_IDS_BY_ROLE
        },
    }
    return run_dir, plan_path, expectations


@pytest.fixture(scope="module")
def fixture_template(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("c0-artifact-template")
    return _prepare_fixture_artifacts(root)


@pytest.fixture
def fixture_run(
    tmp_path: Path,
    fixture_template: Tuple[Path, Path, Dict[str, Any]],
) -> Tuple[Path, Path, Dict[str, Any]]:
    template_run, template_plan, expectations = fixture_template
    run_dir = tmp_path / "run"
    shutil.copytree(template_run, run_dir)
    _patch_sidecar_paths(run_dir)
    plan_path = tmp_path / "fixture_c0_run_plan.json"
    shutil.copy2(template_plan, plan_path)
    return run_dir, plan_path, copy.deepcopy(expectations)


def _validate(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> Mapping[str, Any]:
    run_dir, plan_path, expectations = fixture_run
    return validator._validate_fixture_c0_run_artifacts(
        run_dir, plan_path, **expectations
    )


def test_fixture_mode_accepts_exact_artifacts_and_emits_portable_receipt(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    aggregate = _validate(fixture_run)
    assert aggregate["schema_version"] == (
        "dunhuang-pairwise-c0-artifact-validation/0.2"
    )
    assert aggregate["status"] == "pass_structural_artifact_consistency_only"
    assert aggregate["validation_mode"] == "fixture_non_production"
    assert aggregate["observed"] == {
        "batches_per_epoch": 4,
        "checkpoint_count": 5,
        "epoch_count": 5,
        "full_validation_replay_count": 6,
        "total_steps": 20,
    }
    assert aggregate["winner"]["epoch"] == 2
    assert aggregate["checkpoint_content_assurance"] == {
        "artifact_provenance_authenticated": False,
        "canonical_content_recomputed": False,
        "canonical_content_scope": (
            "receipt_sidecar_cross_consistency_only_not_checkpoint_self_proof"
        ),
        "checkpoint_payload_authenticated": False,
        "checkpoint_bytes_hashed": True,
        "deserialization_api_used": None,
        "filesystem_snapshot_scope": (
            "single_descriptor_reads_with_post_validation_identity_recheck"
        ),
        "raw_predictions_recomputed": False,
        "receipt_signature_verified": False,
        "result_validity_certified": False,
        "threshold_fit_recomputed": False,
    }
    assert aggregate["thresholds"] == {
        "fit_recomputed": False,
        "metrics_recomputed": False,
        "pooled_threshold": None,
        "production_threshold": None,
    }
    assert aggregate["scope"]["artifact_provenance_authenticated"] is False
    assert aggregate["scope"]["checkpoint_deserialized"] is False
    assert aggregate["scope"]["checkpoint_payload_authenticated"] is False
    assert aggregate["scope"]["raw_predictions_recomputed"] is False
    assert aggregate["scope"]["receipt_signature_verified"] is False
    assert aggregate["scope"]["result_validity_certified"] is False
    assert len(aggregate["artifacts"]) == 7
    unsigned = dict(aggregate)
    stored = unsigned.pop("content_sha256")
    assert stored == _content_sha256(unsigned)
    serialized = json.dumps(aggregate, sort_keys=True).casefold()
    assert str(fixture_run[0]).casefold() not in serialized
    assert "torch.load" not in serialized


def test_production_entry_has_no_fixture_override_and_rejects_fixture_plan(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, plan_path, _expectations = fixture_run
    with pytest.raises(
        validator.C0ArtifactValidationError,
        match="canonical run-plan file SHA-256 mismatch",
    ):
        validator.validate_c0_run_artifacts(run_dir, plan_path)


def test_production_policy_binds_the_canonical_plan_and_exact_schedule() -> None:
    plan_path = Path(__file__).resolve().parents[1] / "preflight" / "c0_run_plan.json"
    plan = validator._load_canonical_json(
        plan_path, max_bytes=16 * 1024 * 1024, name="run plan"
    )
    policy = validator._production_policy()
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == (
        validator.PRODUCTION_RUN_PLAN_FILE_SHA256
    )
    validator._validate_plan(plan, policy)
    assert policy.contract["epochs"] == 5
    assert policy.contract["total_steps"] == 320
    assert policy.provider_counts == {
        "batch_count": 6_722,
        "record_count": 1_720_196,
        "fragment_request_count": 3_440_392,
    }
    assert policy.memo_bounds == {
        "max_batch_size": 256,
        "max_cached_fragments": 131_072,
        "max_cache_bytes": 8 * 1024**3,
    }
    assert policy.cublas_workspace_config == ":4096:8"
    assert policy.expected_cluster_counts == {
        "mm_augmented": 55,
        "eccv_1113data": 533,
    }
    assert policy.expected_train_order_commitment_sha256 == (
        validator.PRODUCTION_TRAIN_ORDER_COMMITMENT_SHA256
    )
    assert policy.expected_validation_order_commitment_sha256 == (
        validator.PRODUCTION_VALIDATION_ORDER_COMMITMENT_SHA256
    )
    assert policy.require_optimizer_state_sha256 is True
    assert policy.expected_preprocessing == validator._PRODUCTION_PREPROCESSING
    assert policy.expected_backend_version == validator.BACKEND_ATTESTATION_VERSION
    assert policy.expected_backend_model_family == validator.BACKEND_MODEL_FAMILY
    assert policy.expected_model_config == validator._FROZEN_MODEL_CONFIG
    assert policy.expected_optimizer_config == validator._FROZEN_OPTIMIZER_CONFIG
    assert policy.expected_archive_role_to_provider_logical_id == {
        "eccv_archive": "canonical://eccv_1113data/1113data",
        "mm_archive": "canonical://mm_augmented/dunhuang_augmented_data",
    }
    assert policy.expected_archive_evidence == {
        "eccv_archive": ("tar", "observed_open_file_stream"),
        "mm_archive": ("zip", "observed_open_file_stream"),
    }


def test_provider_archive_genuine_distinct_logical_id_shape_is_accepted(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, plan_path, _expectations = fixture_run
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt = json.loads((run_dir / "c0_run_receipt.json").read_text(encoding="utf-8"))
    plan_by_role = {
        row["role"]: row
        for row in plan["roles"]
        if row["role"] in _PROVIDER_LOGICAL_IDS_BY_ROLE
    }
    provider_by_id = {
        row["logical_id"]: row
        for row in receipt["provider_final_evidence"]["archive_verification"]
    }
    for role, provider_id in _PROVIDER_LOGICAL_IDS_BY_ROLE.items():
        plan_row = plan_by_role[role]
        provider_row = provider_by_id[provider_id]
        assert provider_id != plan_row["logical_id"]
        assert provider_row["expected_sha256"] == plan_row["sha256"]
        assert provider_row["observed_sha256"] == plan_row["sha256"]
        assert provider_row["byte_count"] == plan_row["bytes"]
    assert _validate(fixture_run)["status"] == (
        "pass_structural_artifact_consistency_only"
    )


@pytest.mark.parametrize("mutation", ("swap", "wrong_id"))
def test_provider_archive_role_to_logical_id_mapping_rejects_swap_and_wrong_id(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
    mutation: str,
) -> None:
    run_dir, _plan_path, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        rows = receipt["provider_final_evidence"]["archive_verification"]
        if mutation == "swap":
            rows[0]["logical_id"], rows[1]["logical_id"] = (
                rows[1]["logical_id"],
                rows[0]["logical_id"],
            )
        else:
            rows[0]["logical_id"] = "canonical://eccv_1113data/archive"

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(
        validator.C0ArtifactValidationError,
        match="archive logical ID changed|archive evidence differs from plan",
    ):
        _validate(fixture_run)


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("backend_version", "forged-backend/999"),
        ("model_family", "Not-The-Frozen-Model"),
    ),
)
def test_backend_identity_is_bound_by_the_external_validation_policy(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
    field: str,
    changed: str,
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        receipt["backend"][field] = changed

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(
        validator.C0ArtifactValidationError,
        match="backend attestation identity/safety changed",
    ):
        _validate(fixture_run)


def test_checkpoint_config_hash_is_reconstructed_from_the_frozen_session_config(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        forged = "e" * 64
        for epoch in receipt["epochs"]:
            epoch["checkpoint"]["config_hash"] = forged
        receipt["winner"]["checkpoint"]["config_hash"] = forged

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(
        validator.C0ArtifactValidationError,
        match="reconstructed frozen session config",
    ):
        _validate(fixture_run)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra_test_flag", "key set differs"),
        ("malicious_fit_method", "threshold binding changed"),
        ("wrong_model_hash", "threshold binding changed"),
        ("wrong_sample_count", "threshold sample count"),
    ),
)
def test_diagnostic_threshold_schema_and_bindings_are_exact(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
    mutation: str,
    message: str,
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        row = receipt["thresholds"]["by_dataset"]["mm_augmented"]
        if mutation == "extra_test_flag":
            row["test_records_used"] = True
        elif mutation == "malicious_fit_method":
            row["fit_method"] = "use_test_labels"
        elif mutation == "wrong_model_hash":
            row["model_config_sha256"] = "f" * 64
        else:
            row["sample_count"] = 1

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(validator.C0ArtifactValidationError, match=message):
        _validate(fixture_run)


def test_receipt_portability_scan_rejects_embedded_local_or_sealed_path(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        receipt["thresholds"]["by_dataset"]["mm_augmented"]["path"] = (
            "/sealed/test/labels.json"
        )

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(validator.C0ArtifactValidationError, match="identity/path key"):
        _validate(fixture_run)


@pytest.mark.parametrize(
    "value",
    ("FILE:/sealed/test.json", " /root/private.json", "../private.json"),
)
def test_portability_normalization_rejects_case_whitespace_and_parent_traversal(
    value: str,
) -> None:
    with pytest.raises(
        validator.C0ArtifactValidationError, match="absolute/local path"
    ):
        validator._assert_portable({"opaque_note": value})


@pytest.mark.parametrize(
    "value",
    (
        "dunhuang-pairwise-threshold/0.2",
        "canonical://mm_augmented/archive",
        "SymmetricCoarseSiamese-C0-N-Q1",
    ),
)
def test_portability_normalization_preserves_valid_schema_and_opaque_strings(
    value: str,
) -> None:
    validator._assert_portable({"opaque_note": value})


def test_receipt_replacement_after_parse_is_detected_before_aggregate(
    fixture_run: Tuple[Path, Path, Dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _plan, _expectations = fixture_run
    original = validator._validate_provider

    def replace_after_validation(receipt, plan, policy) -> None:
        original(receipt, plan, policy)
        path = run_dir / "c0_run_receipt.json"
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)

    monkeypatch.setattr(validator, "_validate_provider", replace_after_validation)
    with pytest.raises(
        validator.C0ArtifactValidationError,
        match="run receipt changed during validation",
    ):
        _validate(fixture_run)


def test_checkpoint_replacement_after_hash_is_detected_before_aggregate(
    fixture_run: Tuple[Path, Path, Dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _plan, _expectations = fixture_run
    original = validator._validate_checkpoint_files_and_sidecar

    def replace_after_hash(run_directory, epochs, winner):
        result = original(run_directory, epochs, winner)
        path = run_dir / "checkpoint-epoch-01.pt"
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        return result

    monkeypatch.setattr(
        validator,
        "_validate_checkpoint_files_and_sidecar",
        replace_after_hash,
    )
    with pytest.raises(
        validator.C0ArtifactValidationError,
        match="checkpoint checkpoint-epoch-01.pt changed during validation",
    ):
        _validate(fixture_run)


def test_receipt_byte_tamper_fails_self_hash(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run
    path = run_dir / "c0_run_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["status"] = "tampered"
    _write_json(path, value)
    with pytest.raises(validator.C0ArtifactValidationError, match="self content hash"):
        _validate(fixture_run)


def test_wrong_winner_is_recomputed_and_rejected(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        receipt["winner"] = copy.deepcopy(receipt["winner"])
        receipt["winner"]["epoch"] = 1

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(
        validator.C0ArtifactValidationError, match="winner epoch differs"
    ):
        _validate(fixture_run)


def test_provider_count_tamper_is_rejected_even_with_valid_receipt_self_hash(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        receipt["provider_final_evidence"]["counters"]["record_count"] = 87

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(
        validator.C0ArtifactValidationError, match="provider record_count"
    ):
        _validate(fixture_run)


@pytest.mark.parametrize(
    ("target", "changed", "message"),
    (
        ("attestation", validator.PROVIDER_BASE_VERSION, "attestation version"),
        (
            "final",
            validator.PROVIDER_ATTESTATION_VERSION,
            "base/adapter version mapping",
        ),
    ),
)
def test_provider_versions_are_exact_and_the_adapter_mapping_is_bound(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
    target: str,
    changed: str,
    message: str,
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        if target == "attestation":
            receipt["provider"]["provider_version"] = changed
        else:
            receipt["provider_final_evidence"]["provider_version"] = changed

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(validator.C0ArtifactValidationError, match=message):
        _validate(fixture_run)


@pytest.mark.parametrize(
    ("counter", "changed", "message"),
    (
        ("memo_entry_count", 15, "entries must equal memo misses"),
        ("memo_byte_count", 16 * 65_536 - 1, "memo bytes differ"),
    ),
)
def test_zero_eviction_memo_has_exact_entry_and_tensor_byte_accounting(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
    counter: str,
    changed: int,
    message: str,
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        receipt["provider_final_evidence"]["counters"][counter] = changed

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(validator.C0ArtifactValidationError, match=message):
        _validate(fixture_run)


def test_cluster_count_cannot_collapse_while_sample_counts_stay_correct(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        receipt["epochs"][0]["validation"]["by_dataset"]["mm_augmented"][
            "cluster_count"
        ] = 1

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(validator.C0ArtifactValidationError, match="cluster count"):
        _validate(fixture_run)


def test_population_order_commitment_is_not_only_sha_shaped(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        receipt["metadata_attestation"]["validation_order_commitment_sha256"] = "f" * 64

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(
        validator.C0ArtifactValidationError,
        match="validation order commitment differs",
    ):
        _validate(fixture_run)


def test_production_checkpoint_requires_adamw_optimizer_state_claim(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run
    receipt = json.loads((run_dir / "c0_run_receipt.json").read_text(encoding="utf-8"))
    claim = receipt["epochs"][0]["checkpoint"]
    assert claim["optimizer_state_sha256"] is None  # Tiny runner has no optimizer.
    with pytest.raises(
        validator.C0ArtifactValidationError,
        match="required for production AdamW",
    ):
        validator._validate_checkpoint_claim(
            claim,
            1,
            "production checkpoint",
            optimizer_required=True,
        )


def test_self_consistent_full_canvas_preprocessing_is_still_rejected(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        preprocessing = receipt["provider_final_evidence"]["preprocessing"]
        preprocessing["mode"] = "full_canvas_stretch"
        preprocessing["spatial_contract"] = "legacy_full_canvas_position_preserved"
        preprocessing["geometry_config_sha256"] = "c" * 64
        payload = {
            key: value
            for key, value in preprocessing.items()
            if key not in {"geometry_config_sha256", "preprocessing_sha256"}
        }
        preprocessing["preprocessing_sha256"] = _content_sha256(payload)
        receipt["provider"]["preprocessing_sha256"] = preprocessing[
            "preprocessing_sha256"
        ]

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(
        validator.C0ArtifactValidationError,
        match="frozen tight-crop letterbox",
    ):
        _validate(fixture_run)


@pytest.mark.parametrize(
    ("field", "changed"),
    (("archive_format", "7z"), ("verification_mode", "metadata_only")),
)
def test_archive_format_and_open_stream_verification_mode_are_bound(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
    field: str,
    changed: str,
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        receipt["provider_final_evidence"]["archive_verification"][0][field] = changed

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(
        validator.C0ArtifactValidationError,
        match="format/verification mode changed",
    ):
        _validate(fixture_run)


def test_scope_tamper_is_rejected_even_with_valid_receipt_self_hash(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run

    def mutate(receipt: Dict[str, Any]) -> None:
        receipt["scope"]["sealed_real_records_accepted"] = True

    _rehash_receipt(run_dir, mutate)
    with pytest.raises(validator.C0ArtifactValidationError, match="scope admits"):
        _validate(fixture_run)


def test_checkpoint_file_hash_tamper_is_rejected(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run
    path = run_dir / "checkpoint-epoch-03.pt"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(
        validator.C0ArtifactValidationError, match="file SHA-256 mismatch"
    ):
        _validate(fixture_run)


def test_oversized_sparse_checkpoint_is_rejected_before_hashing(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run
    path = run_dir / "checkpoint-epoch-03.pt"
    with path.open("r+b") as stream:
        stream.truncate(validator._MAX_CHECKPOINT_BYTES + 1)
    with pytest.raises(
        validator.C0ArtifactValidationError, match="byte count is outside its bound"
    ):
        _validate(fixture_run)


def test_opaque_checkpoint_bytes_only_receive_structural_assurance(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run
    epoch = 5
    checkpoint_path = run_dir / "checkpoint-epoch-05.pt"
    checkpoint_path.write_bytes(b"opaque-bytes-that-are-not-a-checkpoint")
    replacement_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    def mutate(receipt: Dict[str, Any]) -> None:
        receipt["epochs"][epoch - 1]["checkpoint"]["file_sha256"] = replacement_sha256

    _rehash_receipt(run_dir, mutate)
    sidecar_path = run_dir / "checkpoint_sidecar.local.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoints"][epoch - 1]["file_sha256"] = replacement_sha256
    _write_json(sidecar_path, sidecar)

    aggregate = _validate(fixture_run)
    assert aggregate["status"] == "pass_structural_artifact_consistency_only"
    assert aggregate["scope"]["checkpoint_payload_authenticated"] is False
    assert aggregate["scope"]["result_validity_certified"] is False


def test_extra_run_artifact_is_rejected_before_receipt_trust(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run
    (run_dir / "notes.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(validator.C0ArtifactValidationError, match="allowlist differs"):
        _validate(fixture_run)


def test_duplicate_json_key_is_rejected(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run
    (run_dir / "c0_run_receipt.json").write_text(
        '{"status":"a","status":"b"}\n', encoding="utf-8"
    )
    with pytest.raises(validator.C0ArtifactValidationError, match="duplicate JSON key"):
        _validate(fixture_run)


def test_semantically_equal_but_noncanonical_json_is_rejected(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run
    path = run_dir / "c0_run_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    with pytest.raises(validator.C0ArtifactValidationError, match="not canonical"):
        _validate(fixture_run)


def test_sidecar_canonical_content_claim_must_match_epoch_receipt(
    fixture_run: Tuple[Path, Path, Dict[str, Any]],
) -> None:
    run_dir, _plan, _expectations = fixture_run
    path = run_dir / "checkpoint_sidecar.local.json"
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    sidecar["checkpoints"][0]["canonical_content_sha256"] = "f" * 64
    _write_json(path, sidecar)
    with pytest.raises(validator.C0ArtifactValidationError, match="inconsistent"):
        _validate(fixture_run)

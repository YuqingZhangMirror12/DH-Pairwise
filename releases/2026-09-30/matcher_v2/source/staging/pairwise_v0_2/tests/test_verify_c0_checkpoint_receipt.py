from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pytest

from staging.pairwise_v0_2.preflight import (
    verify_c0_checkpoint_receipt as verifier,
)
from staging.pairwise_v0_2.training.c0_coarse_backend import (
    C0_COARSE_SEED,
    C0CoarseBackend,
)
from staging.pairwise_v0_2.training.c0_runtime_adapter import c0_coarse_arm
from staging.pairwise_v0_2.training.checkpoint import save_checkpoint


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _pretty_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_token(namespace: str) -> str:
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()


def _session_config(receipt: Mapping[str, Any], session: Any) -> Mapping[str, Any]:
    return {
        "runner": receipt["contract"],
        "provider": receipt["provider"],
        "backend": receipt["backend"],
        "environment": receipt["environment"],
        "model": dict(session.model_config),
        "optimizer": dict(session.optimizer_config),
    }


def _provenance(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "freeze_file_sha256": receipt["locks"]["freeze_file_sha256"],
        "freeze_content_sha256": receipt["locks"]["freeze_content_sha256"],
        "source_lock_commitment_sha256": receipt["locks"][
            "source_lock_commitment_sha256"
        ],
        "environment_sha256": receipt["environment_sha256"],
        "train_order_commitment_sha256": receipt["metadata_attestation"][
            "train_order_commitment_sha256"
        ],
        "validation_order_commitment_sha256": receipt["metadata_attestation"][
            "validation_order_commitment_sha256"
        ],
    }


def _make_structural_acceptance(run_dir: Path, receipt: Mapping) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "schema_version": "fixture-structural-acceptance/0.1",
        "status": "pass_structural_artifact_consistency_only",
        "validation_mode": "fixture_non_production",
        "run_receipt": {
            "file_sha256": _sha256_file(run_dir / "c0_run_receipt.json"),
            "content_sha256": receipt["content_sha256"],
        },
    }
    value["content_sha256"] = _content_sha256(value)
    return value


def _build_fixture(run_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    run_dir.mkdir()
    receipt: Dict[str, Any] = {
        "schema_version": "fixture-c0-run/0.1",
        "contract": {
            "epochs": 5,
            "production": False,
            "seed_sha256": _hash_token("fixture-seed"),
        },
        "provider": {
            "provider_version": "fixture-c0-provider/0.1",
            "sealed_real_test_capability": False,
        },
        "backend": {
            "backend_version": "fixture-c0-backend/0.1",
            "device_type": "cuda",
            "model_family": "SymmetricCoarseSiamese-C0-N-Q1",
            "sealed_real_test_capability": False,
        },
        "environment": {
            "schema_version": "fixture-c0-environment/0.1",
            "source_device": "cuda",
        },
        "environment_sha256": _hash_token("fixture-environment"),
        "locks": {
            "freeze_file_sha256": _hash_token("freeze-file"),
            "freeze_content_sha256": _hash_token("freeze-content"),
            "source_lock_commitment_sha256": _hash_token("source-locks"),
        },
        "metadata_attestation": {
            "train_order_commitment_sha256": _hash_token("train-order"),
            "validation_order_commitment_sha256": _hash_token("validation-order"),
        },
    }
    session = C0CoarseBackend("cpu").create_session(
        c0_coarse_arm(), seed=C0_COARSE_SEED
    )
    config = _session_config(receipt, session)
    epoch_rows = []
    for epoch in range(1, 6):
        validation = {
            "equal_domain_macro_cluster": {
                "auroc": 0.5 + epoch / 100.0,
                "auprc": 0.4 + epoch / 100.0,
            },
            "prediction_commitment_sha256": _hash_token("prediction-{}".format(epoch)),
            "nested_replay_metric": {
                "dataset_a": epoch / 10.0,
                "dataset_b": [epoch, True],
            },
        }
        train = {"mean_loss": 1.0 / epoch, "step_count": epoch}
        metrics = {"epoch": epoch, "train": train, "validation": validation}
        checkpoint = save_checkpoint(
            run_dir / "checkpoint-epoch-{:02d}.pt".format(epoch),
            session.model,
            optimizer=session.optimizer,
            config=config,
            epoch=epoch,
            metrics=metrics,
            provenance=_provenance(receipt),
        )
        epoch_rows.append(
            {
                "epoch": epoch,
                "train": train,
                "validation": validation,
                "checkpoint": {
                    "epoch": epoch,
                    "file_sha256": checkpoint.file_sha256,
                    "canonical_content_sha256": (checkpoint.canonical_content_sha256),
                    "config_hash": checkpoint.config_hash,
                    "model_state_sha256": checkpoint.model_state_sha256,
                    "optimizer_state_sha256": checkpoint.optimizer_state_sha256,
                },
            }
        )
    receipt["epochs"] = epoch_rows
    receipt["winner"] = {
        "epoch": 5,
        "selection_policy": verifier._SELECTION_POLICY,
        "checkpoint": copy.deepcopy(epoch_rows[4]["checkpoint"]),
        "validation": copy.deepcopy(epoch_rows[4]["validation"]),
    }
    receipt["content_sha256"] = _content_sha256(receipt)
    (run_dir / "c0_run_receipt.json").write_bytes(_pretty_json(receipt))
    return run_dir, _make_structural_acceptance(run_dir, receipt)


@pytest.fixture(scope="module")
def fixture_template(tmp_path_factory: pytest.TempPathFactory):
    return _build_fixture(tmp_path_factory.mktemp("c0-content-template") / "run")


@pytest.fixture
def fixture_run(
    tmp_path: Path,
    fixture_template: Tuple[Path, Dict[str, Any]],
) -> Tuple[Path, Dict[str, Any]]:
    template_dir, acceptance = fixture_template
    run_dir = tmp_path / "run"
    shutil.copytree(template_dir, run_dir)
    return run_dir, copy.deepcopy(acceptance)


def _rewrite_receipt(
    run_dir: Path, acceptance: Dict[str, Any], mutate
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    path = run_dir / "c0_run_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    mutate(receipt)
    receipt.pop("content_sha256", None)
    receipt["content_sha256"] = _content_sha256(receipt)
    path.write_bytes(_pretty_json(receipt))
    acceptance["run_receipt"] = {
        "file_sha256": _sha256_file(path),
        "content_sha256": receipt["content_sha256"],
    }
    acceptance.pop("content_sha256", None)
    acceptance["content_sha256"] = _content_sha256(acceptance)
    return receipt, acceptance


def _verify(fixture_run: Tuple[Path, Dict[str, Any]]) -> Mapping[str, Any]:
    run_dir, acceptance = fixture_run
    return verifier._verify_fixture_c0_checkpoint_receipt(run_dir, acceptance)


def test_accepts_five_restricted_fresh_cpu_loads_and_reports_limitations(
    fixture_run: Tuple[Path, Dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    original = verifier.load_trusted_checkpoint

    def observed_loader(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(verifier, "load_trusted_checkpoint", observed_loader)
    aggregate = _verify(fixture_run)
    assert aggregate["status"] == (
        "pass_restricted_checkpoint_content_and_receipt_binding"
    )
    assert aggregate["checkpoint_content_assurance"]["checkpoint_count"] == 5
    assert aggregate["checkpoint_content_assurance"]["fresh_cpu_session_count"] == 5
    assert aggregate["winner_selection"]["recomputed_epoch"] == 5
    assert aggregate["winner_final_replay"]["all_fields_exact_match"] is True
    assert aggregate["scope"]["checkpoint_deserialized"] is True
    assert aggregate["scope"]["model_forward_executed"] is False
    assert aggregate["scope"]["raw_auroc_recomputed"] is False
    assert aggregate["scope"]["raw_auprc_recomputed"] is False
    assert aggregate["scope"]["threshold_fit_recomputed"] is False
    assert aggregate["scope"]["historical_test_read"] is False
    assert aggregate["scope"]["sealed_real_read"] is False
    assert aggregate["scope"]["unsafe_pickle_fallback_used"] is False
    assert len(calls) == 5
    assert all(call["trusted"] is True for call in calls)
    assert all(call["map_location"] == "cpu" for call in calls)
    assert all(call["optimizer"] is not None for call in calls)
    unsigned = dict(aggregate)
    stored = unsigned.pop("content_sha256")
    assert stored == _content_sha256(unsigned)


@pytest.mark.parametrize(
    "claim_field",
    [
        "file_sha256",
        "canonical_content_sha256",
        "model_state_sha256",
        "optimizer_state_sha256",
    ],
)
def test_self_consistently_rehashed_receipt_cannot_forge_checkpoint_claims(
    fixture_run: Tuple[Path, Dict[str, Any]], claim_field: str
) -> None:
    run_dir, acceptance = fixture_run

    def mutate(receipt):
        receipt["epochs"][0]["checkpoint"][claim_field] = _hash_token(
            "forged-" + claim_field
        )

    _rewrite_receipt(run_dir, acceptance, mutate)
    with pytest.raises(
        verifier.C0CheckpointContentVerificationError,
        match="epoch 1|restricted checkpoint load",
    ):
        _verify(fixture_run)


def test_self_consistently_rehashed_config_substitution_is_rejected(
    fixture_run: Tuple[Path, Dict[str, Any]],
) -> None:
    run_dir, acceptance = fixture_run

    def mutate(receipt):
        receipt["provider"]["provider_version"] = "forged-provider/9.9"

    _rewrite_receipt(run_dir, acceptance, mutate)
    with pytest.raises(
        verifier.C0CheckpointContentVerificationError,
        match="checkpoint config hash",
    ):
        _verify(fixture_run)


def test_checkpoint_metrics_are_bound_to_each_epoch_receipt(
    fixture_run: Tuple[Path, Dict[str, Any]],
) -> None:
    run_dir, acceptance = fixture_run

    def mutate(receipt):
        receipt["epochs"][0]["train"]["mean_loss"] = 0.123456

    _rewrite_receipt(run_dir, acceptance, mutate)
    with pytest.raises(
        verifier.C0CheckpointContentVerificationError,
        match="checkpoint metrics receipt binding",
    ):
        _verify(fixture_run)


def test_self_consistently_rehashed_run_locks_cannot_forge_provenance(
    fixture_run: Tuple[Path, Dict[str, Any]],
) -> None:
    run_dir, acceptance = fixture_run

    def mutate(receipt):
        receipt["metadata_attestation"]["validation_order_commitment_sha256"] = (
            _hash_token("forged-validation-order")
        )

    _rewrite_receipt(run_dir, acceptance, mutate)
    with pytest.raises(
        verifier.C0CheckpointContentVerificationError,
        match="checkpoint provenance receipt binding",
    ):
        _verify(fixture_run)


def test_winner_final_replay_requires_every_epoch_five_validation_field(
    fixture_run: Tuple[Path, Dict[str, Any]],
) -> None:
    run_dir, acceptance = fixture_run

    def mutate(receipt):
        receipt["winner"]["validation"]["nested_replay_metric"]["dataset_b"][0] = 4

    _rewrite_receipt(run_dir, acceptance, mutate)
    with pytest.raises(
        verifier.C0CheckpointContentVerificationError,
        match="winner final replay versus selected epoch validation",
    ):
        _verify(fixture_run)


def test_winner_policy_breaks_auroc_then_auprc_ties_by_earlier_epoch() -> None:
    epochs = [
        {
            "epoch": epoch,
            "validation": {
                "equal_domain_macro_cluster": {
                    "auroc": 0.9 if epoch in {2, 3, 4} else 0.5,
                    "auprc": 0.8 if epoch in {3, 4} else 0.7,
                }
            },
        }
        for epoch in range(1, 6)
    ]
    selected, _rankings = verifier._recompute_winner_from_aggregate_receipts(epochs)
    assert selected == 3


def test_public_boundary_requires_production_structural_acceptance(
    fixture_run: Tuple[Path, Dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, acceptance = fixture_run
    monkeypatch.setattr(
        verifier.structural,
        "validate_c0_run_artifacts",
        lambda *_args, **_kwargs: acceptance,
    )
    with pytest.raises(
        verifier.C0CheckpointContentVerificationError,
        match="production structural acceptance",
    ):
        verifier.verify_c0_checkpoint_receipt(run_dir, Path("unused-plan.json"))

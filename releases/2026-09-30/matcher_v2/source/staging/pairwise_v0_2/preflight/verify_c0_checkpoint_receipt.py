"""Restricted content verification for structurally accepted C0 checkpoints.

This is a second, read-only acceptance boundary.  It first reruns the frozen
production structural validator, then opens each of the five checkpoint files
with :func:`load_trusted_checkpoint`.  That loader requires PyTorch's
``weights_only=True`` restricted unpickler and has no unsafe-pickle fallback.

The verifier constructs a fresh, exact C0 coarse session on CPU for every
checkpoint.  It binds the loaded config, model state, optimizer state, epoch,
metrics, and provenance back to the portable run receipt.  It does not open
training/validation archives, read historical-test or sealed-real records, run
model inference, or claim to recompute AUROC/AUPRC/thresholds from raw rows.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from staging.pairwise_v0_2.preflight import validate_c0_run_artifacts as structural
from staging.pairwise_v0_2.training.c0_coarse_backend import (
    C0_COARSE_MODEL_CONFIG,
    C0_COARSE_OPTIMIZER_CONFIG,
    C0_COARSE_SEED,
    C0CoarseBackend,
)
from staging.pairwise_v0_2.training.c0_runtime_adapter import c0_coarse_arm
from staging.pairwise_v0_2.training.checkpoint import (
    canonical_config_hash,
    load_trusted_checkpoint,
)


CONTENT_ACCEPTANCE_SCHEMA_VERSION = (
    "dunhuang-pairwise-c0-checkpoint-content-validation/0.1"
)
_CHECKPOINT_COUNT = 5
_REQUIRED_PRODUCTION_WINNER_EPOCH = 5
_SELECTION_POLICY = "max_equal_domain_macro_cluster_auroc_then_auprc_then_earlier_epoch"
_MAX_CHECKPOINT_BYTES = 256 * 1024**2


class C0CheckpointContentVerificationError(RuntimeError):
    """Raised when checkpoint content cannot be bound to the C0 receipt."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise C0CheckpointContentVerificationError(
            "value is not canonical JSON"
        ) from exc


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise C0CheckpointContentVerificationError(name + " must be an object")
    return value


def _require_hash_equal(observed: Any, expected: Any, name: str) -> None:
    if (
        not isinstance(observed, str)
        or not isinstance(expected, str)
        or not hmac.compare_digest(observed, expected)
    ):
        raise C0CheckpointContentVerificationError(name + " differs")


def _assert_json_exact(observed: Any, expected: Any, name: str) -> None:
    """Compare every JSON field while reporting the first differing location."""

    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            raise C0CheckpointContentVerificationError(name + " type differs")
        if set(observed) != set(expected):
            raise C0CheckpointContentVerificationError(name + " key set differs")
        for key in sorted(expected):
            _assert_json_exact(observed[key], expected[key], name + "." + str(key))
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise C0CheckpointContentVerificationError(name + " list differs")
        for index, item in enumerate(expected):
            _assert_json_exact(observed[index], item, "{}[{}]".format(name, index))
        return
    if type(observed) is not type(expected) or observed != expected:  # noqa: E721
        raise C0CheckpointContentVerificationError(name + " value differs")


def _finite_metric(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise C0CheckpointContentVerificationError(name + " is invalid")
    return float(value)


def _recompute_winner_from_aggregate_receipts(
    epochs: Sequence[Mapping[str, Any]],
) -> Tuple[int, List[Mapping[str, Any]]]:
    """Apply the frozen policy to receipt aggregates, never to raw rows."""

    if len(epochs) != _CHECKPOINT_COUNT:
        raise C0CheckpointContentVerificationError(
            "winner recomputation requires exactly five epoch receipts"
        )
    ranking_rows: List[Mapping[str, Any]] = []
    for expected_epoch, row_value in enumerate(epochs, 1):
        row = _required_mapping(row_value, "epoch receipt")
        epoch = row.get("epoch")
        if type(epoch) is not int or epoch != expected_epoch:  # noqa: E721
            raise C0CheckpointContentVerificationError(
                "epoch receipt order/number differs"
            )
        validation = _required_mapping(row.get("validation"), "epoch validation")
        macro = _required_mapping(
            validation.get("equal_domain_macro_cluster"),
            "equal-domain macro cluster metrics",
        )
        if set(macro) != {"auprc", "auroc"}:
            raise C0CheckpointContentVerificationError(
                "equal-domain macro cluster metric keys differ"
            )
        ranking_rows.append(
            {
                "epoch": epoch,
                "auroc": _finite_metric(macro["auroc"], "aggregate AUROC"),
                "auprc": _finite_metric(macro["auprc"], "aggregate AUPRC"),
            }
        )
    selected = max(
        ranking_rows,
        key=lambda row: (row["auroc"], row["auprc"], -row["epoch"]),
    )
    return int(selected["epoch"]), ranking_rows


def _expected_checkpoint_config(receipt: Mapping[str, Any], session: Any) -> Mapping:
    return {
        "runner": dict(_required_mapping(receipt.get("contract"), "runner contract")),
        "provider": dict(
            _required_mapping(receipt.get("provider"), "provider attestation")
        ),
        "backend": dict(
            _required_mapping(receipt.get("backend"), "backend attestation")
        ),
        "environment": dict(
            _required_mapping(receipt.get("environment"), "runtime environment")
        ),
        "model": dict(session.model_config),
        "optimizer": dict(session.optimizer_config),
    }


def _expected_checkpoint_provenance(receipt: Mapping[str, Any]) -> Mapping:
    locks = _required_mapping(receipt.get("locks"), "run locks")
    metadata = _required_mapping(
        receipt.get("metadata_attestation"), "metadata attestation"
    )
    return {
        "freeze_file_sha256": locks.get("freeze_file_sha256"),
        "freeze_content_sha256": locks.get("freeze_content_sha256"),
        "source_lock_commitment_sha256": locks.get("source_lock_commitment_sha256"),
        "environment_sha256": receipt.get("environment_sha256"),
        "train_order_commitment_sha256": metadata.get("train_order_commitment_sha256"),
        "validation_order_commitment_sha256": metadata.get(
            "validation_order_commitment_sha256"
        ),
    }


def _assert_cpu_tree(value: Any, name: str) -> None:
    # Avoid importing or calling torch.load here.  Tensor values expose a
    # device attribute; primitive optimizer metadata is traversed unchanged.
    device = getattr(value, "device", None)
    if device is not None:
        if getattr(device, "type", None) != "cpu":
            raise C0CheckpointContentVerificationError(name + " is not on CPU")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_cpu_tree(item, "{}.{}".format(name, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_cpu_tree(item, "{}[{}]".format(name, index))


def _new_exact_c0_cpu_session() -> Any:
    backend = C0CoarseBackend("cpu")
    session = backend.create_session(c0_coarse_arm(), seed=C0_COARSE_SEED)
    if session.device.type != "cpu":
        raise C0CheckpointContentVerificationError(
            "fresh checkpoint verification session is not CPU-bound"
        )
    if canonical_config_hash(session.model_config) != canonical_config_hash(
        C0_COARSE_MODEL_CONFIG
    ):
        raise C0CheckpointContentVerificationError("C0 model config changed")
    if canonical_config_hash(session.optimizer_config) != canonical_config_hash(
        C0_COARSE_OPTIMIZER_CONFIG
    ):
        raise C0CheckpointContentVerificationError("C0 optimizer config changed")
    return session


def _receipt_snapshot(run_dir: Path) -> Tuple[Mapping[str, Any], Any]:
    try:
        return structural._load_canonical_json_snapshot(
            run_dir / "c0_run_receipt.json",
            max_bytes=32 * 1024 * 1024,
            name="C0 run receipt",
        )
    except structural.C0ArtifactValidationError as exc:
        raise C0CheckpointContentVerificationError(str(exc)) from exc


def _validate_structural_binding(
    receipt: Mapping[str, Any], receipt_snapshot: Any, acceptance: Mapping[str, Any]
) -> Tuple[str, str]:
    run_receipt = _required_mapping(
        acceptance.get("run_receipt"), "structural run-receipt binding"
    )
    _require_hash_equal(
        receipt_snapshot.sha256,
        run_receipt.get("file_sha256"),
        "run receipt file SHA-256 after structural acceptance",
    )
    stored_content_sha256 = receipt.get("content_sha256")
    unsigned = dict(receipt)
    unsigned.pop("content_sha256", None)
    recomputed_content_sha256 = _content_sha256(unsigned)
    _require_hash_equal(
        recomputed_content_sha256,
        stored_content_sha256,
        "run receipt self content SHA-256",
    )
    _require_hash_equal(
        recomputed_content_sha256,
        run_receipt.get("content_sha256"),
        "run receipt content SHA-256 after structural acceptance",
    )
    structural_unsigned = dict(acceptance)
    stored_structural_content = structural_unsigned.pop("content_sha256", None)
    _require_hash_equal(
        _content_sha256(structural_unsigned),
        stored_structural_content,
        "structural acceptance content SHA-256",
    )
    return receipt_snapshot.sha256, recomputed_content_sha256


def _verify_loaded_checkpoint(
    *,
    run_dir: Path,
    receipt: Mapping[str, Any],
    epoch_row: Mapping[str, Any],
    expected_epoch: int,
) -> Tuple[Mapping[str, Any], Any]:
    checkpoint = _required_mapping(
        epoch_row.get("checkpoint"), "epoch checkpoint receipt"
    )
    filename = "checkpoint-epoch-{:02d}.pt".format(expected_epoch)
    checkpoint_path = run_dir / filename
    try:
        snapshot = structural._hash_regular_file_snapshot(
            checkpoint_path,
            name="checkpoint " + filename,
            max_bytes=_MAX_CHECKPOINT_BYTES,
        )
    except structural.C0ArtifactValidationError as exc:
        raise C0CheckpointContentVerificationError(str(exc)) from exc
    _require_hash_equal(
        snapshot.sha256,
        checkpoint.get("file_sha256"),
        "epoch {} external file SHA-256".format(expected_epoch),
    )

    session = _new_exact_c0_cpu_session()
    expected_config = _expected_checkpoint_config(receipt, session)
    expected_config_hash = canonical_config_hash(expected_config)
    _require_hash_equal(
        expected_config_hash,
        checkpoint.get("config_hash"),
        "epoch {} checkpoint config hash".format(expected_epoch),
    )
    try:
        loaded = load_trusted_checkpoint(
            checkpoint_path,
            session.model,
            optimizer=session.optimizer,
            expected_config=expected_config,
            expected_file_sha256=str(checkpoint.get("file_sha256")),
            expected_canonical_content_sha256=str(
                checkpoint.get("canonical_content_sha256")
            ),
            map_location="cpu",
            trusted=True,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise C0CheckpointContentVerificationError(
            "epoch {} restricted checkpoint load failed: {}".format(expected_epoch, exc)
        ) from exc

    if loaded.get("epoch") != expected_epoch:
        raise C0CheckpointContentVerificationError(
            "loaded checkpoint epoch differs at epoch {}".format(expected_epoch)
        )
    for field in (
        "file_sha256",
        "canonical_content_sha256",
        "config_hash",
        "model_state_sha256",
        "optimizer_state_sha256",
    ):
        _require_hash_equal(
            loaded.get(field),
            checkpoint.get(field),
            "epoch {} loaded {}".format(expected_epoch, field),
        )
    if checkpoint.get("optimizer_state_sha256") is None:
        raise C0CheckpointContentVerificationError(
            "production C0 checkpoint lacks optimizer-state content"
        )
    _require_hash_equal(
        canonical_config_hash(loaded.get("config")),
        expected_config_hash,
        "epoch {} loaded config".format(expected_epoch),
    )
    expected_metrics = {
        "epoch": expected_epoch,
        "train": epoch_row.get("train"),
        "validation": epoch_row.get("validation"),
    }
    _assert_json_exact(
        loaded.get("metrics"),
        expected_metrics,
        "epoch {} checkpoint metrics receipt binding".format(expected_epoch),
    )
    _assert_json_exact(
        loaded.get("provenance"),
        _expected_checkpoint_provenance(receipt),
        "epoch {} checkpoint provenance receipt binding".format(expected_epoch),
    )
    for parameter in session.model.parameters():
        _assert_cpu_tree(parameter, "loaded model parameter")
    _assert_cpu_tree(session.optimizer.state_dict(), "loaded optimizer state")
    return (
        {
            "epoch": expected_epoch,
            "file": filename,
            "bytes": snapshot.byte_count,
            "file_sha256": loaded["file_sha256"],
            "canonical_content_sha256": loaded["canonical_content_sha256"],
            "config_hash": loaded["config_hash"],
            "model_state_sha256": loaded["model_state_sha256"],
            "optimizer_state_sha256": loaded["optimizer_state_sha256"],
            "checkpoint_metrics_receipt_match": True,
            "checkpoint_provenance_receipt_match": True,
            "fresh_session_device": "cpu",
        },
        snapshot,
    )


def _verify_after_structural_acceptance(
    run_dir: Path,
    acceptance: Mapping[str, Any],
    *,
    validation_mode: str,
    required_winner_epoch: int,
) -> Mapping[str, Any]:
    run_dir = Path(run_dir)
    receipt, receipt_snapshot = _receipt_snapshot(run_dir)
    receipt_file_sha256, receipt_content_sha256 = _validate_structural_binding(
        receipt, receipt_snapshot, acceptance
    )
    epochs_value = receipt.get("epochs")
    if not isinstance(epochs_value, list):
        raise C0CheckpointContentVerificationError("epochs must be a list")
    epochs = [_required_mapping(value, "epoch receipt") for value in epochs_value]

    checkpoint_rows: List[Mapping[str, Any]] = []
    checkpoint_snapshots = []
    for expected_epoch, epoch_row in enumerate(epochs, 1):
        row, snapshot = _verify_loaded_checkpoint(
            run_dir=run_dir,
            receipt=receipt,
            epoch_row=epoch_row,
            expected_epoch=expected_epoch,
        )
        checkpoint_rows.append(row)
        checkpoint_snapshots.append(snapshot)

    recomputed_winner_epoch, rankings = _recompute_winner_from_aggregate_receipts(
        epochs
    )
    winner = _required_mapping(receipt.get("winner"), "winner receipt")
    if winner.get("selection_policy") != _SELECTION_POLICY:
        raise C0CheckpointContentVerificationError("winner selection policy changed")
    if winner.get("epoch") != recomputed_winner_epoch:
        raise C0CheckpointContentVerificationError(
            "receipt winner differs from independently recomputed winner"
        )
    if recomputed_winner_epoch != required_winner_epoch:
        raise C0CheckpointContentVerificationError(
            "production C0 winner must be epoch {}".format(required_winner_epoch)
        )
    selected_epoch_row = epochs[recomputed_winner_epoch - 1]
    _assert_json_exact(
        winner.get("checkpoint"),
        selected_epoch_row.get("checkpoint"),
        "winner checkpoint versus selected epoch",
    )
    _assert_json_exact(
        winner.get("validation"),
        selected_epoch_row.get("validation"),
        "winner final replay versus selected epoch validation",
    )
    selected_validation = _required_mapping(
        selected_epoch_row.get("validation"), "selected epoch validation"
    )
    _require_hash_equal(
        _required_mapping(winner.get("validation"), "winner validation").get(
            "prediction_commitment_sha256"
        ),
        selected_validation.get("prediction_commitment_sha256"),
        "winner final replay prediction commitment",
    )
    validation_content_sha256 = _content_sha256(selected_validation)

    aggregate: Dict[str, Any] = {
        "schema_version": CONTENT_ACCEPTANCE_SCHEMA_VERSION,
        "status": "pass_restricted_checkpoint_content_and_receipt_binding",
        "validation_mode": validation_mode,
        "scope": {
            "artifact_provenance_authenticated": False,
            "checkpoint_deserialized": True,
            "checkpoint_payload_bound_to_receipt_hash_claims": True,
            "historical_test_read": False,
            "model_forward_executed": False,
            "raw_auprc_recomputed": False,
            "raw_auroc_recomputed": False,
            "raw_predictions_recomputed": False,
            "receipt_signature_verified": False,
            "result_validity_certified": False,
            "sealed_real_read": False,
            "threshold_fit_recomputed": False,
            "train_or_validation_archive_read": False,
            "unsafe_pickle_fallback_used": False,
        },
        "structural_acceptance": {
            "content_sha256": acceptance["content_sha256"],
            "schema_version": acceptance["schema_version"],
            "status": acceptance["status"],
            "revalidated_in_process": True,
        },
        "run_receipt": {
            "file_sha256": receipt_file_sha256,
            "content_sha256": receipt_content_sha256,
            "file_sha256_recomputed": True,
            "content_sha256_recomputed": True,
        },
        "checkpoint_content_assurance": {
            "canonical_content_recomputed": True,
            "checkpoint_count": len(checkpoint_rows),
            "config_recomputed_and_exact": True,
            "deserialization_api": (
                "training.checkpoint.load_trusted_checkpoint/weights_only_true"
            ),
            "external_file_sha256_recomputed": True,
            "fresh_cpu_session_count": len(checkpoint_rows),
            "model_state_recomputed_and_loaded_strict": True,
            "optimizer_state_recomputed_and_loaded": True,
        },
        "checkpoints": checkpoint_rows,
        "winner_selection": {
            "aggregate_receipt_policy_recomputed": True,
            "policy": _SELECTION_POLICY,
            "rankings": rankings,
            "receipt_epoch": winner["epoch"],
            "recomputed_epoch": recomputed_winner_epoch,
        },
        "winner_final_replay": {
            "epoch": recomputed_winner_epoch,
            "epoch_validation_content_sha256": validation_content_sha256,
            "winner_validation_content_sha256": _content_sha256(winner["validation"]),
            "all_fields_exact_match": True,
            "prediction_commitment_exact_match": True,
            "prediction_commitment_sha256": selected_validation[
                "prediction_commitment_sha256"
            ],
        },
        "limitations": {
            "aggregate_metric_source": (
                "portable_run_receipt_only_without_raw_validation_rows"
            ),
            "raw_auprc_recomputed": False,
            "raw_auroc_recomputed": False,
            "threshold_fit_recomputed": False,
        },
    }
    try:
        structural._assert_portable(aggregate)
    except structural.C0ArtifactValidationError as exc:
        raise C0CheckpointContentVerificationError(str(exc)) from exc
    aggregate["content_sha256"] = _content_sha256(aggregate)

    try:
        structural._assert_snapshot_current(receipt_snapshot, "C0 run receipt")
        for snapshot in checkpoint_snapshots:
            structural._assert_snapshot_current(
                snapshot, "checkpoint " + snapshot.path.name
            )
    except structural.C0ArtifactValidationError as exc:
        raise C0CheckpointContentVerificationError(str(exc)) from exc
    return aggregate


def verify_c0_checkpoint_receipt(
    run_dir: Path,
    canonical_run_plan_path: Path,
    *,
    aggregate_receipt_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Verify production C0 checkpoint contents after structural acceptance.

    The public boundary has no fixture/relaxed mode, alternate session factory,
    alternate winner, unsafe-loader option, or archive/data input.
    """

    try:
        acceptance = structural.validate_c0_run_artifacts(
            Path(run_dir), Path(canonical_run_plan_path)
        )
    except structural.C0ArtifactValidationError as exc:
        raise C0CheckpointContentVerificationError(
            "structural acceptance failed: " + str(exc)
        ) from exc
    if (
        acceptance.get("status") != "pass_structural_artifact_consistency_only"
        or acceptance.get("validation_mode") != "production"
    ):
        raise C0CheckpointContentVerificationError(
            "production structural acceptance was not established"
        )
    aggregate = _verify_after_structural_acceptance(
        Path(run_dir),
        acceptance,
        validation_mode="production",
        required_winner_epoch=_REQUIRED_PRODUCTION_WINNER_EPOCH,
    )
    if aggregate_receipt_path is not None:
        try:
            structural._write_aggregate_receipt(
                Path(aggregate_receipt_path), aggregate, run_dir=Path(run_dir)
            )
        except structural.C0ArtifactValidationError as exc:
            raise C0CheckpointContentVerificationError(str(exc)) from exc
    return aggregate


def _verify_fixture_c0_checkpoint_receipt(
    run_dir: Path, structural_acceptance: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Private unit-test boundary; it still uses exact C0 CPU sessions/loader."""

    return _verify_after_structural_acceptance(
        Path(run_dir),
        structural_acceptance,
        validation_mode="fixture_non_production",
        required_winner_epoch=_REQUIRED_PRODUCTION_WINNER_EPOCH,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Restricted, production-strict C0 checkpoint content verification")
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        receipt = verify_c0_checkpoint_receipt(
            args.run_dir,
            args.run_plan,
            aggregate_receipt_path=args.output,
        )
    except C0CheckpointContentVerificationError as exc:
        print(
            "C0 checkpoint content verification failed: {}".format(exc), file=sys.stderr
        )
        return 2
    if args.output is None:
        sys.stdout.buffer.write(structural._canonical_pretty_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "C0CheckpointContentVerificationError",
    "CONTENT_ACCEPTANCE_SCHEMA_VERSION",
    "verify_c0_checkpoint_receipt",
]

"""Sealed synthetic-test evaluation for completed Rachel N=512 runs.

This entry point is deliberately separate from training.  It validates and
strictly restores every validation-selected winner *before* it first opens the
Rachel ``test`` manifest.  It never trains, selects a checkpoint, or fits a
threshold; the only decision threshold it applies is the checkpoint-bound
validation artifact stored in the completed training receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import random
import shutil
import stat
import tempfile
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from staging.pairwise_v0_2.baselines.rachel_matched_mm_evaluation import (
    MATCHED_METHODS,
    MATCHED_SCORE,
    FrozenMatchedMMWinner,
    SyntheticMaskPair,
    freeze_matched_mm_winners,
    matched_metrics,
    safe_synthetic_model_mask_path,
    score_synthetic_mask_pairs,
    synthetic_pair_records,
)
from staging.pairwise_v0_2.baselines import (
    rachel_same_data_benchmark_eval_adapter as benchmark_adapter,
)
from staging.pairwise_v0_2.models.rachel_n512 import (
    RachelN512Config,
    RachelN512Pairwise,
)
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import (
    RachelBatch,
    RachelPairDataset,
    collate_rachel_pairs,
)
from staging.pairwise_v0_2.training.evaluation import (
    PairwiseThresholdArtifact,
    evaluate_pairwise,
)
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig


SCHEMA_VERSION = "rachel-n512-sealed-synthetic-test/1.0"
FORMAL_COMPLETE_STATUS = "complete_frozen_synthetic_test_only"
COMPATIBILITY_COMPLETE_STATUS = "complete_non_formal_compatibility_synthetic_evaluation"
TRAIN_SCHEMA_VERSION = "rachel-n512-train-run/1.0"
CHECKPOINT_SCHEMA_VERSION = "rachel-n512-checkpoint/1.0"
CONVERGENCE_SCHEMA_VERSION = "rachel-n512-convergence-extension/1.0"
EXPECTED_TEST_PAIRS = 3000
ARMS = ("coarse_only", "full_n512")
FORMAL_MAX_TOTAL_EPOCHS = 128
FORMAL_MIN_TOTAL_EPOCHS = 20
FORMAL_PATIENCE = 12
FORMAL_MIN_RELATIVE_PRIMARY_IMPROVEMENT = 0.005
FORMAL_ETA_MIN = 1e-6
FORMAL_TRAIN_PAIRS = 24_000
FORMAL_VAL_PAIRS = 3_000
FORMAL_BATCH_SIZE = 16
PAIRINGNET_REGISTRATION_RECALL_THRESHOLD = 4.0
ASSEMBLY_EDGE_TOLERANCES = (2, 5, 8, 10, 100)

_CONVERGENCE_POINTER_FIELDS = frozenset(
    {
        "schema_version",
        "receipt",
        "receipt_sha256",
        "source_receipt_sha256",
        "source_last_epoch",
        "max_total_epochs",
    }
)
_CONVERGENCE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "fingerprint_sha256",
        "source_run",
        "source_receipt_sha256",
        "policy",
        "config",
        "population",
        "arm_results",
        "test_accessed",
        "real_external_test_accessed",
    }
)
_CONVERGENCE_POLICY_FIELDS = frozenset(
    {
        "source_run",
        "output_root",
        "arms",
        "max_total_epochs",
        "min_total_epochs",
        "patience",
        "min_relative_primary_improvement",
        "eta_min",
        "device",
        "remaining_epochs",
    }
)


class RachelN512SealedTestError(RuntimeError):
    """A sealed-test provenance, population, or inference gate failed."""


@dataclass(frozen=True)
class RachelN512SealedTestConfig:
    run_directory: Path
    output_root: Path
    dataset_root: Optional[Path] = None
    matched_mm_run_directory: Optional[Path] = None
    pairingnet_run_directory: Optional[Path] = None
    shreddingnet_freeze_path: Optional[Path] = None
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda:0"
    compatibility_mode: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_directory", Path(self.run_directory).expanduser())
        object.__setattr__(self, "output_root", Path(self.output_root).expanduser())
        if self.dataset_root is not None:
            object.__setattr__(
                self, "dataset_root", Path(self.dataset_root).expanduser()
            )
        if self.matched_mm_run_directory is not None:
            object.__setattr__(
                self,
                "matched_mm_run_directory",
                Path(self.matched_mm_run_directory).expanduser(),
            )
        if self.pairingnet_run_directory is not None:
            object.__setattr__(
                self,
                "pairingnet_run_directory",
                Path(self.pairingnet_run_directory).expanduser(),
            )
        if self.shreddingnet_freeze_path is not None:
            object.__setattr__(
                self,
                "shreddingnet_freeze_path",
                Path(self.shreddingnet_freeze_path).expanduser(),
            )
        if (self.pairingnet_run_directory is None) != (
            self.shreddingnet_freeze_path is None
        ):
            raise ValueError(
                "PairingNet and ShreddingNet benchmark authorities must be supplied together"
            )
        if type(self.batch_size) is not int or self.batch_size <= 0:  # noqa: E721
            raise ValueError("batch_size must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers < 0:  # noqa: E721
            raise ValueError("num_workers must be a non-negative integer")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty torch device string")
        if type(self.compatibility_mode) is not bool:  # noqa: E721
            raise TypeError("compatibility_mode must be bool")


@dataclass(frozen=True)
class _ManifestRow:
    pair_id: str
    label: bool
    cluster_id: str
    source_unit_ids: Tuple[str, ...] = ()
    mask_a_path: Optional[Path] = None
    mask_b_path: Optional[Path] = None


@dataclass
class _FrozenWinner:
    arm: str
    epoch: int
    checkpoint_path: Path
    checkpoint_sha256: str
    model_config: RachelN512Config
    model_config_sha256: str
    loss_config: RachelN512LossConfig
    loss_config_sha256: str
    threshold: PairwiseThresholdArtifact
    model: nn.Module


def _canonical_rachel_model_config() -> RachelN512Config:
    """Return the only architecture accepted by the formal N=512 freeze."""

    return RachelN512Config(validate_runtime_inputs=False)


def _canonical_rachel_loss_config() -> RachelN512LossConfig:
    """Return the only target-loss configuration accepted by the freeze."""

    return RachelN512LossConfig(
        validate_runtime_targets=False,
        collect_cpu_diagnostics=False,
    )


def _canonical_config_authority(
    winners: Sequence[_FrozenWinner],
) -> Dict[str, object]:
    """Summarize exact per-field winner configuration equality and SHA binding."""

    expected_model = asdict(_canonical_rachel_model_config())
    expected_loss = asdict(_canonical_rachel_loss_config())
    model_sha = hashlib.sha256(_canonical_bytes(expected_model)).hexdigest()
    loss_sha = hashlib.sha256(_canonical_bytes(expected_loss)).hexdigest()
    if not winners or any(
        asdict(winner.model_config) != expected_model
        or winner.model_config_sha256 != model_sha
        or asdict(winner.loss_config) != expected_loss
        or winner.loss_config_sha256 != loss_sha
        for winner in winners
    ):
        raise RachelN512SealedTestError(
            "winner model/loss config differs from the complete canonical config"
        )
    return {
        "comparison": "complete_dataclass_field_mapping_exact_equality",
        "model_config": expected_model,
        "model_config_sha256": model_sha,
        "loss_config": expected_loss,
        "loss_config_sha256": loss_sha,
        "model_config_sha256_by_arm": {
            winner.arm: winner.model_config_sha256 for winner in winners
        },
        "loss_config_sha256_by_arm": {
            winner.arm: winner.loss_config_sha256 for winner in winners
        },
        "all_winner_configs_exactly_equal": True,
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    _publish_bytes_no_replace(path, _canonical_bytes(value) + b"\n")


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    _publish_bytes_no_replace(path, payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bytes_no_replace(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise RachelN512SealedTestError("refusing to overwrite sealed output file")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + target.name + ".tmp-", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise RachelN512SealedTestError(
                "refusing to overwrite sealed output file"
            ) from error
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_directory_no_replace(
    staged: Path, final: Path, *, completion_receipt: str
) -> None:
    """Publish a staged tree with the completion receipt linked strictly last."""

    staged = Path(staged)
    final = Path(final)
    receipt_source = staged / completion_receipt
    if not receipt_source.is_file() or receipt_source.is_symlink():
        raise RachelN512SealedTestError("sealed completion receipt is missing")
    try:
        final.mkdir()
    except FileExistsError as error:
        raise RachelN512SealedTestError(
            "sealed-test output already exists; preserve it rather than overwriting"
        ) from error
    published = False
    try:
        directories = sorted(
            (item for item in staged.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
        )
        for directory in directories:
            if directory.is_symlink():
                raise RachelN512SealedTestError(
                    "symlinks are forbidden in staged sealed output"
                )
            (final / directory.relative_to(staged)).mkdir()
        files = sorted(item for item in staged.rglob("*") if item.is_file())
        for source in files:
            if source == receipt_source:
                continue
            if source.is_symlink():
                raise RachelN512SealedTestError(
                    "symlinks are forbidden in staged sealed output"
                )
            os.link(source, final / source.relative_to(staged))
        for directory in sorted(
            (item for item in final.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        os.link(receipt_source, final / completion_receipt)
        _fsync_directory(final)
        _fsync_directory(final.parent)
        published = True
    except FileExistsError as error:
        raise RachelN512SealedTestError(
            "refusing to overwrite sealed output during publication"
        ) from error
    finally:
        if not published:
            shutil.rmtree(final, ignore_errors=True)
    shutil.rmtree(staged)


def _read_json_object(path: Path, name: str) -> Dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RachelN512SealedTestError("cannot read " + name) from error
    if not isinstance(value, dict):
        raise RachelN512SealedTestError(name + " must be a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_object_bytes(payload: bytes, name: str) -> Dict[str, object]:
    """Decode one finite JSON object while rejecting ambiguous encodings."""

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value

    def reject_constant(token: str):
        raise ValueError("non-finite JSON constant: " + token)

    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RachelN512SealedTestError(
            name + " must be strict finite UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise RachelN512SealedTestError(name + " must be a JSON object")
    return value


def _inode_identity(metadata: os.stat_result) -> Tuple[int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode)


def _file_identity(metadata: os.stat_result) -> Tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_descriptor_bytes(descriptor: int) -> bytes:
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _open_relative_component(
    parent_descriptor: int, component: str, *, directory: bool
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    metadata = os.stat(component, dir_fd=parent_descriptor, follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode):
        raise RachelN512SealedTestError(
            "N=512 convergence receipt path traverses a symlink"
        )
    descriptor = os.open(component, flags, dir_fd=parent_descriptor)
    opened = os.fstat(descriptor)
    expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_kind(opened.st_mode) or _inode_identity(opened) != _inode_identity(
        metadata
    ):
        os.close(descriptor)
        raise RachelN512SealedTestError(
            "N=512 convergence receipt path identity changed while opening"
        )
    return descriptor


def _read_frozen_run_json_member(
    run_directory: Path,
    relative_value: object,
    expected_sha256: object,
    description: str,
) -> Tuple[Dict[str, object], bytes]:
    """Read a SHA-bound run member once through no-follow directory handles."""

    if not _is_sha256(expected_sha256):
        raise RachelN512SealedTestError(description + " SHA-256 is invalid")
    if (
        not isinstance(relative_value, str)
        or not relative_value
        or "\\" in relative_value
        or "\x00" in relative_value
    ):
        raise RachelN512SealedTestError(description + " path is invalid")
    logical = PurePosixPath(relative_value)
    if (
        logical.is_absolute()
        or relative_value != logical.as_posix()
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.suffix != ".json"
    ):
        raise RachelN512SealedTestError(
            description + " path must be a canonical relative JSON path"
        )
    try:
        run = Path(run_directory).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RachelN512SealedTestError("N=512 run directory is unavailable") from error
    if not run.is_dir() or run.name.startswith(".partial-"):
        raise RachelN512SealedTestError(
            "N=512 convergence authority requires a finalized run"
        )

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors = []
    confirmation_descriptors = []
    try:
        root_descriptor = os.open(str(run), directory_flags)
        descriptors.append(root_descriptor)
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise RachelN512SealedTestError("N=512 run authority is not a directory")
        directory_identities = [_inode_identity(root_metadata)]
        parent = root_descriptor
        for component in logical.parts[:-1]:
            child = _open_relative_component(parent, component, directory=True)
            descriptors.append(child)
            directory_identities.append(_inode_identity(os.fstat(child)))
            parent = child
        member_descriptor = _open_relative_component(
            parent, logical.parts[-1], directory=False
        )
        descriptors.append(member_descriptor)
        before = os.fstat(member_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RachelN512SealedTestError(
                description + " must be a single-link regular file"
            )
        payload = _read_descriptor_bytes(member_descriptor)
        after = os.fstat(member_descriptor)
        if _file_identity(after) != _file_identity(before):
            raise RachelN512SealedTestError(
                description + " changed while its bytes were frozen"
            )
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if observed_sha256 != expected_sha256:
            raise RachelN512SealedTestError(description + " SHA-256 differs")

        # Confirm that the canonical run path still names the same directory
        # chain and file.  Parsing below consumes only ``payload``; no later
        # pathname read can substitute different JSON bytes.
        confirmed_root = os.open(str(run), directory_flags)
        confirmation_descriptors.append(confirmed_root)
        if _inode_identity(os.fstat(confirmed_root)) != directory_identities[0]:
            raise RachelN512SealedTestError(
                description + " run identity changed after its bytes were frozen"
            )
        confirmed_parent = confirmed_root
        for index, component in enumerate(logical.parts[:-1], 1):
            confirmed_child = _open_relative_component(
                confirmed_parent, component, directory=True
            )
            confirmation_descriptors.append(confirmed_child)
            if (
                _inode_identity(os.fstat(confirmed_child))
                != directory_identities[index]
            ):
                raise RachelN512SealedTestError(
                    description + " directory identity changed after read"
                )
            confirmed_parent = confirmed_child
        confirmed_member = _open_relative_component(
            confirmed_parent, logical.parts[-1], directory=False
        )
        confirmation_descriptors.append(confirmed_member)
        confirmed_metadata = os.fstat(confirmed_member)
        if confirmed_metadata.st_nlink != 1 or _file_identity(
            confirmed_metadata
        ) != _file_identity(after):
            raise RachelN512SealedTestError(
                description + " path identity changed after its bytes were frozen"
            )
    except RachelN512SealedTestError:
        raise
    except OSError as error:
        raise RachelN512SealedTestError(
            description + " is not a stable confined regular file"
        ) from error
    finally:
        for descriptor in reversed(confirmation_descriptors):
            os.close(descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return _strict_json_object_bytes(payload, description), payload


def _canonical_absolute_path_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RachelN512SealedTestError(description + " is invalid")
    path = Path(value)
    if not path.is_absolute() or str(path) != value or os.path.normpath(value) != value:
        raise RachelN512SealedTestError(
            description + " must be a canonical absolute path"
        )
    return value


def _json_values_exactly_equal(left: object, right: object) -> bool:
    try:
        return _canonical_bytes(left) == _canonical_bytes(right)
    except (TypeError, ValueError, UnicodeError):
        return False


def _load_bound_n512_convergence_receipt(
    n512_receipt: Mapping[str, object], n512_run_directory: Path
) -> Dict[str, object]:
    """Load and cross-check the sole authoritative N=512 continuation policy."""

    if (
        n512_receipt.get("schema_version") != TRAIN_SCHEMA_VERSION
        or n512_receipt.get("status") != "complete_train_validation_only"
    ):
        raise RachelN512SealedTestError(
            "N=512 top-level training schema/status is not production-complete"
        )
    _require_false(n512_receipt, "test_accessed", "N=512 training receipt")
    _require_false(
        n512_receipt,
        "real_external_test_accessed",
        "N=512 training receipt",
    )
    top_fingerprint = n512_receipt.get("fingerprint_sha256")
    if not _is_sha256(top_fingerprint):
        raise RachelN512SealedTestError(
            "N=512 top-level training fingerprint is invalid"
        )

    pointer = n512_receipt.get("convergence")
    if not isinstance(pointer, Mapping) or set(pointer) != set(
        _CONVERGENCE_POINTER_FIELDS
    ):
        raise RachelN512SealedTestError(
            "N=512 convergence receipt pointer fields differ"
        )
    if pointer.get("schema_version") != CONVERGENCE_SCHEMA_VERSION:
        raise RachelN512SealedTestError("N=512 convergence pointer schema differs")
    if (
        type(pointer.get("source_last_epoch")) is not int  # noqa: E721
        or type(pointer.get("max_total_epochs")) is not int  # noqa: E721
    ):
        raise RachelN512SealedTestError(
            "N=512 convergence pointer integer policy is malformed"
        )
    convergence_receipt, _ = _read_frozen_run_json_member(
        n512_run_directory,
        pointer.get("receipt"),
        pointer.get("receipt_sha256"),
        "N=512 convergence receipt",
    )
    if set(convergence_receipt) != set(_CONVERGENCE_RECEIPT_FIELDS):
        raise RachelN512SealedTestError("N=512 convergence receipt fields differ")
    if (
        convergence_receipt.get("schema_version") != CONVERGENCE_SCHEMA_VERSION
        or convergence_receipt.get("status") != "complete_train_validation_only"
    ):
        raise RachelN512SealedTestError(
            "N=512 convergence receipt schema/status differs"
        )
    _require_false(convergence_receipt, "test_accessed", "N=512 convergence receipt")
    _require_false(
        convergence_receipt,
        "real_external_test_accessed",
        "N=512 convergence receipt",
    )

    config = convergence_receipt.get("config")
    population = convergence_receipt.get("population")
    arm_results = convergence_receipt.get("arm_results")
    policy = convergence_receipt.get("policy")
    continuation = config.get("continuation") if isinstance(config, Mapping) else None
    if (
        not isinstance(config, Mapping)
        or not isinstance(population, Mapping)
        or not isinstance(arm_results, list)
        or not isinstance(policy, Mapping)
        or not isinstance(continuation, Mapping)
    ):
        raise RachelN512SealedTestError(
            "N=512 convergence config/population/arms/policy is malformed"
        )
    if set(policy) != set(_CONVERGENCE_POLICY_FIELDS):
        raise RachelN512SealedTestError("N=512 convergence policy fields differ")

    if (
        convergence_receipt.get("fingerprint_sha256") != top_fingerprint
        or not _json_values_exactly_equal(config, n512_receipt.get("config"))
        or not _json_values_exactly_equal(population, n512_receipt.get("population"))
        or not _json_values_exactly_equal(arm_results, n512_receipt.get("arm_results"))
    ):
        raise RachelN512SealedTestError(
            "N=512 convergence receipt differs from top-level frozen evidence"
        )
    source_receipt_sha256 = convergence_receipt.get("source_receipt_sha256")
    if (
        not _is_sha256(source_receipt_sha256)
        or source_receipt_sha256 != pointer.get("source_receipt_sha256")
        or source_receipt_sha256 != continuation.get("source_receipt_sha256")
    ):
        raise RachelN512SealedTestError(
            "N=512 convergence source receipt binding differs"
        )
    if (
        continuation.get("schema_version") != CONVERGENCE_SCHEMA_VERSION
        or continuation.get("source_last_epoch") != pointer.get("source_last_epoch")
        or pointer.get("source_last_epoch") != 5
    ):
        raise RachelN512SealedTestError(
            "N=512 convergence continuation source authority differs"
        )

    source_run = _canonical_absolute_path_string(
        convergence_receipt.get("source_run"), "N=512 convergence source_run"
    )
    policy_source = _canonical_absolute_path_string(
        policy.get("source_run"), "N=512 convergence policy source_run"
    )
    policy_output = _canonical_absolute_path_string(
        policy.get("output_root"), "N=512 convergence policy output_root"
    )
    configured_arms = config.get("arms")
    if (
        source_run != policy_source
        or source_run != continuation.get("source_run")
        or policy_output != config.get("output_root")
        or not isinstance(configured_arms, list)
        or policy.get("arms") != configured_arms
        or len(configured_arms) != len(ARMS)
        or set(configured_arms) != set(ARMS)
    ):
        raise RachelN512SealedTestError(
            "N=512 convergence policy source/output/arms differ"
        )

    expected_policy = {
        "max_total_epochs": FORMAL_MAX_TOTAL_EPOCHS,
        "min_total_epochs": FORMAL_MIN_TOTAL_EPOCHS,
        "patience": FORMAL_PATIENCE,
        "min_relative_primary_improvement": (FORMAL_MIN_RELATIVE_PRIMARY_IMPROVEMENT),
        "eta_min": FORMAL_ETA_MIN,
        "remaining_epochs": FORMAL_MAX_TOTAL_EPOCHS - 5,
    }
    if any(policy.get(key) != value for key, value in expected_policy.items()):
        raise RachelN512SealedTestError("N=512 convergence formal policy values differ")
    if any(
        type(policy.get(key)) is not int  # noqa: E721
        for key in (
            "max_total_epochs",
            "min_total_epochs",
            "patience",
            "remaining_epochs",
        )
    ) or any(
        type(policy.get(key)) is not float  # noqa: E721
        for key in ("min_relative_primary_improvement", "eta_min")
    ):
        raise RachelN512SealedTestError("N=512 convergence policy scalar types differ")
    if (
        pointer.get("max_total_epochs") != policy.get("max_total_epochs")
        or config.get("epochs") != policy.get("max_total_epochs")
        or continuation.get("scheduler_t_max") != policy.get("remaining_epochs")
        or continuation.get("min_total_epochs") != policy.get("min_total_epochs")
        or continuation.get("patience") != policy.get("patience")
        or continuation.get("min_relative_primary_improvement")
        != policy.get("min_relative_primary_improvement")
        or continuation.get("scheduler_eta_min") != policy.get("eta_min")
        or config.get("device") != policy.get("device")
        or not isinstance(policy.get("device"), str)
        or not str(policy["device"]).startswith("cuda")
    ):
        raise RachelN512SealedTestError(
            "N=512 convergence policy/config cross-binding differs"
        )
    if (
        any(
            type(config.get(key)) is not int  # noqa: E721
            for key in ("epochs", "batch_size", "seed")
        )
        or any(
            type(continuation.get(key)) is not int  # noqa: E721
            for key in (
                "source_last_epoch",
                "scheduler_t_max",
                "min_total_epochs",
                "patience",
            )
        )
        or any(
            type(continuation.get(key)) is not float  # noqa: E721
            for key in (
                "min_relative_primary_improvement",
                "scheduler_eta_min",
            )
        )
    ):
        raise RachelN512SealedTestError("N=512 convergence config scalar types differ")
    _require_false(continuation, "test_accessed", "N=512 continuation config")
    _require_false(
        continuation,
        "real_external_test_accessed",
        "N=512 continuation config",
    )

    expected_population = {
        "train_total": FORMAL_TRAIN_PAIRS,
        "val_total": FORMAL_VAL_PAIRS,
        "train_rows_per_epoch": FORMAL_TRAIN_PAIRS,
        "val_rows": FORMAL_VAL_PAIRS,
    }
    if set(population) != set(expected_population) or any(
        type(population.get(key)) is not int  # noqa: E721
        or population.get(key) != value
        for key, value in expected_population.items()
    ):
        raise RachelN512SealedTestError(
            "N=512 convergence population schema/types differ"
        )

    results_by_arm = {}
    for raw_result in arm_results:
        if not isinstance(raw_result, Mapping):
            raise RachelN512SealedTestError("N=512 convergence arm result is malformed")
        arm = raw_result.get("arm")
        if arm not in ARMS or arm in results_by_arm:
            raise RachelN512SealedTestError(
                "N=512 convergence arm result is missing or duplicated"
            )
        _require_false(raw_result, "test_accessed", str(arm) + " arm result")
        _require_false(
            raw_result,
            "real_external_test_accessed",
            str(arm) + " arm result",
        )
        results_by_arm[str(arm)] = raw_result
    if set(results_by_arm) != set(ARMS):
        raise RachelN512SealedTestError(
            "N=512 convergence arm result inventory differs"
        )
    return convergence_receipt


def _safe_winner_path(run_directory: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RachelN512SealedTestError("winner checkpoint path is invalid")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise RachelN512SealedTestError("winner checkpoint path is unsafe")
    unresolved = run_directory.joinpath(*logical.parts)
    current = run_directory
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise RachelN512SealedTestError("symlinked winner checkpoint is forbidden")
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(run_directory)
    except (OSError, RuntimeError, ValueError) as error:
        raise RachelN512SealedTestError(
            "winner checkpoint is missing or escapes run directory"
        ) from error
    if not resolved.is_file() or resolved.suffix != ".pt":
        raise RachelN512SealedTestError("winner checkpoint must be a .pt file")
    return resolved


def _safe_run_json_member(run_directory: Path, value: object, description: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RachelN512SealedTestError(description + " path is invalid")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise RachelN512SealedTestError(description + " path is unsafe")
    current = run_directory
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise RachelN512SealedTestError(description + " may not be symlinked")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(run_directory)
    except (OSError, RuntimeError, ValueError) as error:
        raise RachelN512SealedTestError(
            description + " is missing or escapes the run"
        ) from error
    if not resolved.is_file() or resolved.suffix != ".json":
        raise RachelN512SealedTestError(description + " must be a JSON file")
    return resolved


def _new_model(arm: str, config: RachelN512Config) -> nn.Module:
    container = RachelN512Pairwise(config)
    return container.coarse if arm == "coarse_only" else container


def _torch_load_checkpoint(path: Path) -> Mapping[str, object]:
    version = torch.__version__.split("+", 1)[0].split(".")
    try:
        major = int(version[0])
    except (IndexError, ValueError):
        major = 0
    try:
        # PyTorch 1.13 exposes ``weights_only`` but its restricted unpickler
        # cannot read even primitive state-dict archives produced by the same
        # runtime.  Those legacy checkpoints are still SHA-bound to the frozen
        # private-run receipt before this call; PyTorch >=2 uses safe loading.
        value = torch.load(
            path,
            map_location="cpu",
            **({"weights_only": True} if major >= 2 else {}),
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        raise RachelN512SealedTestError(
            "cannot safely load winner checkpoint"
        ) from error
    if not isinstance(value, Mapping):
        raise RachelN512SealedTestError("winner checkpoint must be a mapping")
    return value


def _require_false(mapping: Mapping[str, object], key: str, location: str) -> None:
    if mapping.get(key) is not False:
        raise RachelN512SealedTestError(location + "." + key + " must be false")


def _validate_threshold(
    value: object,
    *,
    arm: str,
    checkpoint_sha256: str,
    model_config_sha256: str,
) -> PairwiseThresholdArtifact:
    if not isinstance(value, Mapping):
        raise RachelN512SealedTestError("validation_threshold must be an object")
    try:
        artifact = PairwiseThresholdArtifact(**dict(value))
    except (TypeError, ValueError) as error:
        raise RachelN512SealedTestError(
            "validation_threshold failed schema validation"
        ) from error
    if artifact.schema_version != "dunhuang-pairwise-threshold/0.2":
        raise RachelN512SealedTestError("unsupported validation threshold schema")
    expected_aggregation = hashlib.sha256(
        _canonical_bytes({"pair_score": "coarse" if arm == "coarse_only" else "fused"})
    ).hexdigest()
    if artifact.checkpoint_sha256 != checkpoint_sha256:
        raise RachelN512SealedTestError("threshold is bound to a different checkpoint")
    if artifact.model_config_sha256 != model_config_sha256:
        raise RachelN512SealedTestError(
            "threshold is bound to a different model config"
        )
    if artifact.aggregation_config_sha256 != expected_aggregation:
        raise RachelN512SealedTestError("threshold is bound to a different pair score")
    return artifact


def _freeze_completed_winners(
    run_directory: Path,
) -> Tuple[Dict[str, object], str, Tuple[_FrozenWinner, ...]]:
    """Validate/strict-load winners without touching any test artefact."""

    run_directory = run_directory.resolve(strict=True)
    if not run_directory.is_dir() or run_directory.name.startswith(".partial-"):
        raise RachelN512SealedTestError("run_directory is not a finalized run")
    receipt_path = run_directory / "run_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RachelN512SealedTestError(
            "completed training receipt must be a regular non-symlink file"
        )
    receipt = _read_json_object(receipt_path, "completed training receipt")
    receipt_sha256 = _sha256_file(receipt_path)
    if receipt.get("schema_version") != TRAIN_SCHEMA_VERSION:
        raise RachelN512SealedTestError("unsupported training receipt schema")
    if receipt.get("status") != "complete_train_validation_only":
        raise RachelN512SealedTestError("training receipt is not complete/frozen")
    _require_false(receipt, "test_accessed", "training receipt")
    _require_false(receipt, "real_external_test_accessed", "training receipt")
    run_config = receipt.get("config")
    if not isinstance(run_config, Mapping):
        raise RachelN512SealedTestError("training receipt config is missing")
    configured_arms = run_config.get("arms")
    arm_results = receipt.get("arm_results")
    if (
        not isinstance(configured_arms, list)
        or not configured_arms
        or not isinstance(arm_results, list)
        or len(configured_arms) != len(arm_results)
        or len(set(configured_arms)) != len(configured_arms)
        or any(arm not in ARMS for arm in configured_arms)
    ):
        raise RachelN512SealedTestError("training arms are malformed or incomplete")

    results_by_arm: Dict[str, Mapping[str, object]] = {}
    for result in arm_results:
        if not isinstance(result, Mapping) or result.get("arm") in results_by_arm:
            raise RachelN512SealedTestError("arm result is malformed or duplicated")
        results_by_arm[str(result.get("arm"))] = result
    if set(results_by_arm) != set(configured_arms):
        raise RachelN512SealedTestError(
            "completed receipt lacks a configured arm result"
        )

    frozen = []
    for arm in configured_arms:
        result = results_by_arm[arm]
        _require_false(result, "test_accessed", arm + " arm result")
        _require_false(result, "real_external_test_accessed", arm + " arm result")
        epoch = result.get("winner_epoch")
        winner_path_value = result.get("winner_checkpoint")
        winner_sha = result.get("winner_checkpoint_sha256")
        if type(epoch) is not int or epoch <= 0:  # noqa: E721
            raise RachelN512SealedTestError("winner_epoch is invalid")
        if not isinstance(winner_sha, str) or len(winner_sha) != 64:
            raise RachelN512SealedTestError("winner checkpoint SHA-256 is invalid")
        winner_path = _safe_winner_path(run_directory, winner_path_value)
        observed_sha = _sha256_file(winner_path)
        if observed_sha != winner_sha:
            raise RachelN512SealedTestError("winner checkpoint SHA-256 mismatch")
        epochs = result.get("epochs")
        if not isinstance(epochs, list):
            raise RachelN512SealedTestError("arm epoch history is missing")
        selected = [
            row
            for row in epochs
            if isinstance(row, Mapping) and row.get("epoch") == epoch
        ]
        if (
            len(selected) != 1
            or selected[0].get("checkpoint") != winner_path_value
            or selected[0].get("checkpoint_sha256") != winner_sha
        ):
            raise RachelN512SealedTestError(
                "winner checkpoint does not match frozen epoch history"
            )

        checkpoint = _torch_load_checkpoint(winner_path)
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RachelN512SealedTestError("unsupported checkpoint schema")
        if checkpoint.get("arm") != arm or checkpoint.get("epoch") != epoch:
            raise RachelN512SealedTestError("checkpoint arm/epoch provenance mismatch")
        _require_false(checkpoint, "test_accessed", "checkpoint")
        _require_false(checkpoint, "real_external_test_accessed", "checkpoint")
        if checkpoint.get("run_config") != run_config:
            raise RachelN512SealedTestError(
                "checkpoint run config disagrees with receipt"
            )
        model_config_value = checkpoint.get("model_config")
        loss_config_value = checkpoint.get("loss_config")
        if not isinstance(model_config_value, Mapping) or not isinstance(
            loss_config_value, Mapping
        ):
            raise RachelN512SealedTestError("checkpoint model/loss config is missing")
        try:
            model_config = RachelN512Config(**dict(model_config_value))
            loss_config = RachelN512LossConfig(**dict(loss_config_value))
        except (TypeError, ValueError) as error:
            raise RachelN512SealedTestError(
                "checkpoint model/loss config is invalid"
            ) from error
        canonical_model_config = asdict(_canonical_rachel_model_config())
        canonical_loss_config = asdict(_canonical_rachel_loss_config())
        if (
            dict(model_config_value) != canonical_model_config
            or asdict(model_config) != canonical_model_config
        ):
            raise RachelN512SealedTestError(
                "checkpoint model config differs from the complete canonical config"
            )
        if (
            dict(loss_config_value) != canonical_loss_config
            or asdict(loss_config) != canonical_loss_config
        ):
            raise RachelN512SealedTestError(
                "checkpoint loss config differs from the complete canonical config"
            )
        model_config_sha = hashlib.sha256(
            _canonical_bytes(asdict(model_config))
        ).hexdigest()
        loss_config_sha = hashlib.sha256(
            _canonical_bytes(asdict(loss_config))
        ).hexdigest()
        threshold = _validate_threshold(
            result.get("validation_threshold"),
            arm=arm,
            checkpoint_sha256=winner_sha,
            model_config_sha256=model_config_sha,
        )
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise RachelN512SealedTestError("checkpoint model state is missing")
        model = _new_model(arm, model_config)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise RachelN512SealedTestError(
                "winner checkpoint failed strict model restore"
            ) from error
        frozen.append(
            _FrozenWinner(
                arm=arm,
                epoch=epoch,
                checkpoint_path=winner_path,
                checkpoint_sha256=winner_sha,
                model_config=model_config,
                model_config_sha256=model_config_sha,
                loss_config=loss_config,
                loss_config_sha256=loss_config_sha,
                threshold=threshold,
                model=model,
            )
        )
    winners = tuple(frozen)
    _canonical_config_authority(winners)
    return receipt, receipt_sha256, winners


def _require_formal_convergence(
    receipt: Mapping[str, object], winners: Sequence[_FrozenWinner]
) -> None:
    """Fail closed unless the exact formal run reached a validation plateau."""

    if {winner.arm for winner in winners} != set(ARMS):
        raise RachelN512SealedTestError(
            "combined formal evaluation requires both Rachel N=512 arms"
        )
    results = receipt.get("arm_results")
    if not isinstance(results, list):
        raise RachelN512SealedTestError("Rachel convergence results are missing")
    by_arm = {
        str(row.get("arm")): row
        for row in results
        if isinstance(row, Mapping) and row.get("arm") in ARMS
    }
    if set(by_arm) != set(ARMS):
        raise RachelN512SealedTestError("Rachel convergence arms are incomplete")
    for arm in ARMS:
        row = by_arm[arm]
        if (
            row.get("stop_reason") != "validation_early_stop"
            or row.get("convergence_claim") != "validation_plateau_under_declared_rule"
        ):
            raise RachelN512SealedTestError(
                arm + " has not established a validation convergence plateau"
            )
    config = receipt.get("config")
    population = receipt.get("population")
    convergence = receipt.get("convergence")
    if (
        not isinstance(config, Mapping)
        or not isinstance(population, Mapping)
        or not isinstance(convergence, Mapping)
    ):
        raise RachelN512SealedTestError(
            "formal Rachel config/population/convergence authority is missing"
        )
    continuation = config.get("continuation")
    if not isinstance(continuation, Mapping):
        raise RachelN512SealedTestError("formal Rachel continuation config is missing")
    exact_config = {
        "epochs": FORMAL_MAX_TOTAL_EPOCHS,
        "batch_size": FORMAL_BATCH_SIZE,
    }
    if any(config.get(key) != value for key, value in exact_config.items()):
        raise RachelN512SealedTestError(
            "formal Rachel total epochs/batch exposure config differs"
        )
    exact_continuation = {
        "source_last_epoch": 5,
        "scheduler": "CosineAnnealingLR",
        "scheduler_eta_min": FORMAL_ETA_MIN,
        "min_total_epochs": FORMAL_MIN_TOTAL_EPOCHS,
        "patience": FORMAL_PATIENCE,
        "min_relative_primary_improvement": (FORMAL_MIN_RELATIVE_PRIMARY_IMPROVEMENT),
        "early_stop_reads": ["val"],
        "checkpoint_selection_reads": ["val"],
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    if any(continuation.get(key) != value for key, value in exact_continuation.items()):
        raise RachelN512SealedTestError(
            "formal Rachel validation-only continuation policy differs"
        )
    exact_population = {
        "train_total": FORMAL_TRAIN_PAIRS,
        "val_total": FORMAL_VAL_PAIRS,
        "train_rows_per_epoch": FORMAL_TRAIN_PAIRS,
        "val_rows": FORMAL_VAL_PAIRS,
    }
    if any(population.get(key) != value for key, value in exact_population.items()):
        raise RachelN512SealedTestError(
            "formal Rachel train/validation exposure population differs"
        )
    if (
        convergence.get("max_total_epochs") != FORMAL_MAX_TOTAL_EPOCHS
        or convergence.get("source_last_epoch") != 5
    ):
        raise RachelN512SealedTestError(
            "formal Rachel convergence receipt policy differs"
        )


def _resolved_training_dataset_root(
    receipt: Mapping[str, object], *, description: str
) -> Path:
    config = receipt.get("config")
    value = config.get("dataset_root") if isinstance(config, Mapping) else None
    if not isinstance(value, str) or not value:
        raise RachelN512SealedTestError(description + " dataset_root is missing")
    try:
        root = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RachelN512SealedTestError(
            description + " dataset_root is unavailable"
        ) from error
    if not root.is_dir():
        raise RachelN512SealedTestError(
            description + " dataset_root is not a directory"
        )
    return root


def _require_matched_training_alignment(
    n512_receipt: Mapping[str, object], matched_receipt: Mapping[str, object]
) -> Path:
    """Verify same release, seed and epoch exposure before combined evaluation."""

    n512_config = n512_receipt.get("config")
    matched_config = matched_receipt.get("config")
    n512_population = n512_receipt.get("population")
    matched_population = matched_receipt.get("population")
    if (
        not isinstance(n512_config, Mapping)
        or not isinstance(matched_config, Mapping)
        or not isinstance(n512_population, Mapping)
        or not isinstance(matched_population, Mapping)
    ):
        raise RachelN512SealedTestError(
            "combined training config/population authority is missing"
        )
    n512_root = _resolved_training_dataset_root(
        n512_receipt, description="Rachel N=512"
    )
    matched_root = _resolved_training_dataset_root(
        matched_receipt, description="matched-MM"
    )
    if n512_root != matched_root:
        raise RachelN512SealedTestError(
            "Rachel N=512 and matched-MM resolved training releases differ"
        )
    if (
        type(n512_config.get("seed")) is not int  # noqa: E721
        or n512_config.get("seed") != matched_config.get("seed")
        or n512_config.get("batch_size") != FORMAL_BATCH_SIZE
        or matched_config.get("batch_size") != FORMAL_BATCH_SIZE
    ):
        raise RachelN512SealedTestError(
            "Rachel N=512 and matched-MM seed/batch exposure differ"
        )
    exact_matched_config = {
        "max_total_epochs": FORMAL_MAX_TOTAL_EPOCHS,
        "min_total_epochs": FORMAL_MIN_TOTAL_EPOCHS,
        "patience": FORMAL_PATIENCE,
        "min_relative_auroc_improvement": (FORMAL_MIN_RELATIVE_PRIMARY_IMPROVEMENT),
        "source_splits_opened": ["train", "val"],
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    if any(
        matched_config.get(key) != value for key, value in exact_matched_config.items()
    ):
        raise RachelN512SealedTestError(
            "matched-MM formal validation-only convergence config differs"
        )
    shared_population = {
        "train_total": FORMAL_TRAIN_PAIRS,
        "val_total": FORMAL_VAL_PAIRS,
    }
    if any(
        n512_population.get(key) != value or matched_population.get(key) != value
        for key, value in shared_population.items()
    ):
        raise RachelN512SealedTestError(
            "Rachel N=512 and matched-MM train/validation populations differ"
        )
    if (
        n512_population.get("train_rows_per_epoch") != FORMAL_TRAIN_PAIRS
        or n512_population.get("val_rows") != FORMAL_VAL_PAIRS
        or matched_population.get("train_positive") != FORMAL_TRAIN_PAIRS // 2
        or matched_population.get("val_positive") != FORMAL_VAL_PAIRS // 2
    ):
        raise RachelN512SealedTestError(
            "Rachel N=512 and matched-MM label/exposure contract differs"
        )
    return n512_root


def _validation_pair_order_evidence(
    run_directory: Path,
    *,
    artifact: object,
    artifact_sha256: object,
    threshold: object,
    description: str,
) -> Tuple[Tuple[str, ...], Mapping[str, object]]:
    path = _safe_run_json_member(run_directory, artifact, description)
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or _sha256_file(path) != artifact_sha256
    ):
        raise RachelN512SealedTestError(description + " SHA-256 differs")
    value = _read_json_object(path, description)
    raw_pair_ids = value.get("pair_ids")
    if (
        not isinstance(raw_pair_ids, list)
        or len(raw_pair_ids) != FORMAL_VAL_PAIRS
        or len(set(raw_pair_ids)) != FORMAL_VAL_PAIRS
        or any(not isinstance(pair_id, str) or not pair_id for pair_id in raw_pair_ids)
    ):
        raise RachelN512SealedTestError(description + " pair order is invalid")
    for field in ("labels", "clusters", "probability", "valid"):
        vector = value.get(field)
        if not isinstance(vector, list) or len(vector) != FORMAL_VAL_PAIRS:
            raise RachelN512SealedTestError(
                description + " score vectors are incomplete"
            )
    if not isinstance(threshold, Mapping):
        raise RachelN512SealedTestError(description + " threshold is missing")
    checkpoint_sha256 = threshold.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise RachelN512SealedTestError(
            description + " threshold checkpoint binding is missing"
        )
    fingerprint = hashlib.sha256(_canonical_bytes(raw_pair_ids)).hexdigest()
    if threshold.get("validation_fingerprint_sha256") != fingerprint:
        raise RachelN512SealedTestError(
            description + " threshold is not bound to its validation pair order"
        )
    return tuple(raw_pair_ids), {
        "artifact": str(path.relative_to(run_directory)),
        "artifact_sha256": artifact_sha256,
        "pair_count": len(raw_pair_ids),
        "pair_order_fingerprint_sha256": fingerprint,
        "threshold_bound_to_pair_order": True,
        "threshold_checkpoint_sha256": checkpoint_sha256,
    }


def _training_manifest_pair_ids(
    path: Path, *, split: str, expected_count: int
) -> Tuple[str, ...]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                value = json.loads(line)
                if (
                    not isinstance(value, Mapping)
                    or value.get("split") != split
                    or not isinstance(value.get("pair_id"), str)
                    or not value.get("pair_id")
                ):
                    raise RachelN512SealedTestError(
                        split
                        + " manifest pair identity is invalid at line "
                        + str(line_number)
                    )
                rows.append(str(value["pair_id"]))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RachelN512SealedTestError(
            split + " manifest cannot supply pair identity evidence"
        ) from error
    if len(rows) != expected_count or len(set(rows)) != expected_count:
        raise RachelN512SealedTestError(
            split + " manifest pair population is incomplete or duplicated"
        )
    return tuple(rows)


def _matched_training_hash_evidence(
    n512_receipt: Mapping[str, object],
    matched_receipt: Mapping[str, object],
    *,
    n512_run_directory: Path,
    matched_run_directory: Path,
) -> Mapping[str, object]:
    """Bind same-release exposure claims to frozen manifests and score artefacts."""

    dataset_root = _require_matched_training_alignment(n512_receipt, matched_receipt)
    manifests = {}
    manifest_hashes = {}
    for split in ("train", "val"):
        path = dataset_root / "pairs" / (split + ".jsonl")
        if path.is_symlink() or not path.is_file():
            raise RachelN512SealedTestError(
                "aligned " + split + " manifest must be a regular file"
            )
        observed = _sha256_file(path)
        manifests[split] = path
        manifest_hashes[split] = observed
    manifest_pair_ids = {
        "train": _training_manifest_pair_ids(
            manifests["train"], split="train", expected_count=FORMAL_TRAIN_PAIRS
        ),
        "val": _training_manifest_pair_ids(
            manifests["val"], split="val", expected_count=FORMAL_VAL_PAIRS
        ),
    }

    convergence_receipt = _load_bound_n512_convergence_receipt(
        n512_receipt, n512_run_directory
    )
    n512_policy = convergence_receipt["policy"]
    n512_config = convergence_receipt["config"]
    n512_fingerprint_payload = {
        "schema_version": convergence_receipt["schema_version"],
        "source_receipt_sha256": convergence_receipt["source_receipt_sha256"],
        "config": dict(n512_policy),
        "extension_run_config": dict(n512_config),
        "train_manifest_sha256": manifest_hashes["train"],
        "val_manifest_sha256": manifest_hashes["val"],
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    n512_fingerprint = hashlib.sha256(
        _canonical_bytes(n512_fingerprint_payload)
    ).hexdigest()
    if convergence_receipt.get("fingerprint_sha256") != n512_fingerprint:
        raise RachelN512SealedTestError(
            "N=512 training fingerprint is not bound to current train/val manifests"
        )

    matched_config = matched_receipt.get("config")
    matched_schema = matched_receipt.get("schema_version")
    if not isinstance(matched_config, Mapping) or not isinstance(matched_schema, str):
        raise RachelN512SealedTestError(
            "matched-MM receipt lacks hash-reconstructable training provenance"
        )
    matched_fingerprint_payload = {
        "schema_version": matched_schema,
        "config": dict(matched_config),
        "train_manifest_sha256": manifest_hashes["train"],
        "val_manifest_sha256": manifest_hashes["val"],
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    matched_fingerprint = hashlib.sha256(
        _canonical_bytes(matched_fingerprint_payload)
    ).hexdigest()
    if matched_receipt.get("fingerprint_sha256") != matched_fingerprint:
        raise RachelN512SealedTestError(
            "matched-MM training fingerprint is not bound to current train/val manifests"
        )

    n512_results = convergence_receipt["arm_results"]
    order_by_source: Dict[str, Tuple[str, ...]] = {}
    score_evidence: Dict[str, Mapping[str, object]] = {}
    for raw_result in n512_results:
        if not isinstance(raw_result, Mapping) or raw_result.get("arm") not in ARMS:
            raise RachelN512SealedTestError("N=512 validation evidence is malformed")
        arm = str(raw_result["arm"])
        order, evidence = _validation_pair_order_evidence(
            n512_run_directory,
            artifact=raw_result.get("winner_validation_scores"),
            artifact_sha256=raw_result.get("winner_validation_scores_sha256"),
            threshold=raw_result.get("validation_threshold"),
            description="N=512 " + arm + " winner validation scores",
        )
        order_by_source[arm] = order
        score_evidence[arm] = evidence

    epochs = matched_receipt.get("epochs")
    winner_epoch = matched_receipt.get("winner_epoch")
    if not isinstance(epochs, list) or type(winner_epoch) is not int:  # noqa: E721
        raise RachelN512SealedTestError("matched-MM epoch history is missing")
    winner_rows = [
        row
        for row in epochs
        if isinstance(row, Mapping) and row.get("epoch") == winner_epoch
    ]
    if len(winner_rows) != 1:
        raise RachelN512SealedTestError("matched-MM winner validation row is ambiguous")
    converged_order, converged_evidence = _validation_pair_order_evidence(
        matched_run_directory,
        artifact=winner_rows[0].get("validation_scores"),
        artifact_sha256=winner_rows[0].get("validation_scores_sha256"),
        threshold=matched_receipt.get("validation_threshold"),
        description="matched-MM converged validation scores",
    )
    same_exposure = matched_receipt.get("same_exposure_epoch5")
    if not isinstance(same_exposure, Mapping):
        raise RachelN512SealedTestError("matched-MM epoch-5 evidence is missing")
    epoch5_order, epoch5_evidence = _validation_pair_order_evidence(
        matched_run_directory,
        artifact=same_exposure.get("validation_scores"),
        artifact_sha256=same_exposure.get("validation_scores_sha256"),
        threshold=same_exposure.get("validation_threshold"),
        description="matched-MM epoch-5 validation scores",
    )
    order_by_source[MATCHED_METHODS[0]] = converged_order
    order_by_source[MATCHED_METHODS[1]] = epoch5_order
    score_evidence[MATCHED_METHODS[0]] = converged_evidence
    score_evidence[MATCHED_METHODS[1]] = epoch5_evidence
    first_order = next(iter(order_by_source.values()))
    if any(order != first_order for order in order_by_source.values()):
        raise RachelN512SealedTestError(
            "N=512 and matched-MM validation pair orders differ"
        )
    if first_order != manifest_pair_ids["val"]:
        raise RachelN512SealedTestError(
            "frozen validation score order differs from the bound validation manifest"
        )
    pair_order_fingerprint = hashlib.sha256(
        _canonical_bytes(list(first_order))
    ).hexdigest()
    return {
        "claim_level": (
            "same_frozen_train_val_manifest_content_and_exact_validation_pair_order;"
            "per_epoch_training_presentation_order_not_provable"
        ),
        "canonical_dataset_root": str(dataset_root),
        "canonical_seed": n512_config.get("seed"),
        "train_manifest": {
            "path": str(manifests["train"]),
            "content_sha256": manifest_hashes["train"],
            "bound_to_both_training_fingerprints": True,
            "pair_count": len(manifest_pair_ids["train"]),
            "pair_order_fingerprint_sha256": hashlib.sha256(
                _canonical_bytes(list(manifest_pair_ids["train"]))
            ).hexdigest(),
        },
        "validation_manifest": {
            "path": str(manifests["val"]),
            "content_sha256": manifest_hashes["val"],
            "bound_to_both_training_fingerprints": True,
            "pair_count": len(manifest_pair_ids["val"]),
            "pair_order_fingerprint_sha256": pair_order_fingerprint,
            "exact_order_equal_to_all_four_validation_score_artifacts": True,
        },
        "training_fingerprints_recomputed": {
            "rachel_n512": n512_fingerprint,
            "matched_mm": matched_fingerprint,
        },
        "validation_pair_order": {
            "exact_order_equal_across_four_frozen_thresholds": True,
            "pair_count": len(first_order),
            "pair_order_fingerprint_sha256": pair_order_fingerprint,
            "sources": score_evidence,
        },
        "limitations": {
            "exact_per_epoch_training_pair_presentation_order_saved": False,
            "exact_per_epoch_training_pair_presentation_order_claimed_equal": False,
        },
    }


def _resolve_evaluation_dataset_root(
    receipt: Mapping[str, object], override: Optional[Path]
) -> Path:
    receipt_root = _resolved_training_dataset_root(receipt, description="Rachel N=512")
    if override is None:
        return receipt_root
    try:
        selected = Path(override).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RachelN512SealedTestError("dataset root is unavailable") from error
    if not selected.is_dir():
        raise RachelN512SealedTestError("dataset root must be a directory")
    if selected != receipt_root:
        raise RachelN512SealedTestError(
            "dataset override must canonically equal the receipt dataset_root"
        )
    return selected


def _benchmark_training_manifest_evidence(
    dataset_root: Path,
    frozen: Mapping[str, benchmark_adapter.FrozenSameDataBenchmark],
) -> Mapping[str, object]:
    """Bind both benchmark winners to the exact N=512 train/val bytes."""

    if tuple(frozen) != benchmark_adapter.BENCHMARK_METHODS:
        raise RachelN512SealedTestError(
            "same-data benchmark winner inventory is incomplete"
        )
    observed = {}
    for split in ("train", "val"):
        path = dataset_root / "pairs" / (split + ".jsonl")
        if path.is_symlink() or not path.is_file():
            raise RachelN512SealedTestError(
                "same-data benchmark train/val manifest is missing"
            )
        observed[split] = _sha256_file(path)
    for method in benchmark_adapter.BENCHMARK_METHODS:
        if dict(frozen[method].training_manifest_sha256) != observed:
            raise RachelN512SealedTestError(
                method + " train/validation manifest differs from Rachel N=512"
            )
    return {
        "manifest_content_sha256": observed,
        "exact_manifest_bytes_equal_across_n512_pairingnet_and_shreddingnet": True,
        "methods": {
            method: frozen[method].provenance()
            for method in benchmark_adapter.BENCHMARK_METHODS
        },
    }


def _test_manifest_rows(
    dataset_root: Path,
    expected_count: int = EXPECTED_TEST_PAIRS,
    *,
    require_model_masks: bool = False,
) -> Tuple[Path, Tuple[_ManifestRow, ...]]:
    """Open and validate the sealed population after winners are frozen."""

    unresolved = dataset_root / "pairs" / "test.jsonl"
    current = dataset_root
    for part in ("pairs", "test.jsonl"):
        current = current / part
        if current.is_symlink():
            raise RachelN512SealedTestError("symlinked test manifest is forbidden")
    try:
        path = unresolved.resolve(strict=True)
        path.relative_to(dataset_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise RachelN512SealedTestError(
            "test manifest is missing or escapes dataset root"
        ) from error
    if not path.is_file():
        raise RachelN512SealedTestError("test manifest must be a regular file")
    rows = []
    seen = set()
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise RachelN512SealedTestError(
                        "test manifest line is not an object"
                    )
                pair_id = value.get("pair_id")
                label = value.get("label")
                first = value.get("fragment_a")
                second = value.get("fragment_b")
                first_unit = (
                    first.get("split_unit_id") if isinstance(first, Mapping) else None
                )
                second_unit = (
                    second.get("split_unit_id") if isinstance(second, Mapping) else None
                )
                first_mask = None
                second_mask = None
                if require_model_masks:
                    if not isinstance(first, Mapping) or not isinstance(
                        second, Mapping
                    ):
                        raise RachelN512SealedTestError(
                            "matched-MM test row lacks fragment endpoints"
                        )
                    try:
                        first_mask = safe_synthetic_model_mask_path(
                            dataset_root, first.get("model_mask_path")
                        )
                        second_mask = safe_synthetic_model_mask_path(
                            dataset_root, second.get("model_mask_path")
                        )
                    except RuntimeError as error:
                        raise RachelN512SealedTestError(
                            "matched-MM test model-mask path is invalid"
                        ) from error
                    if first_mask == second_mask:
                        raise RachelN512SealedTestError(
                            "matched-MM test pair endpoints must differ"
                        )
                if (
                    value.get("split") != "test"
                    or type(label) is not bool  # noqa: E721
                    or not isinstance(pair_id, str)
                    or not pair_id
                    or not isinstance(first_unit, str)
                    or not first_unit
                    or not isinstance(second_unit, str)
                    or not second_unit
                ):
                    raise RachelN512SealedTestError(
                        "malformed test manifest line {}".format(line_number)
                    )
                if pair_id in seen:
                    raise RachelN512SealedTestError(
                        "duplicate test pair_id: " + pair_id
                    )
                seen.add(pair_id)
                units = sorted((first_unit, second_unit))
                cluster = (
                    "unit:" + units[0]
                    if units[0] == units[1]
                    else "unit-pair:"
                    + hashlib.sha256(_canonical_bytes(units)).hexdigest()
                )
                rows.append(
                    _ManifestRow(
                        pair_id,
                        label,
                        cluster,
                        tuple(sorted({first_unit, second_unit})),
                        first_mask,
                        second_mask,
                    )
                )
    except json.JSONDecodeError as error:
        raise RachelN512SealedTestError(
            "test manifest contains invalid JSON"
        ) from error
    except OSError as error:
        raise RachelN512SealedTestError("cannot read test manifest") from error
    if len(rows) != expected_count:
        raise RachelN512SealedTestError(
            "test population must contain exactly {} pairs; observed {}".format(
                expected_count, len(rows)
            )
        )
    positive = sum(row.label for row in rows)
    if positive * 2 != len(rows):
        raise RachelN512SealedTestError(
            "test population must preserve exact 1:1 positive/negative labels"
        )
    return path, tuple(rows)


def _worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _test_loader(
    dataset: RachelPairDataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    options: Dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "collate_fn": collate_rachel_pairs,
        "worker_init_fn": _worker_seed,
        "generator": generator,
    }
    if num_workers > 0:
        options.update({"persistent_workers": False, "prefetch_factor": 2})
    return DataLoader(**options)


def _tensor(value: np.ndarray, device: torch.device, dtype=None) -> Tensor:
    result = torch.from_numpy(np.ascontiguousarray(value))
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result.to(device=device, non_blocking=False)


def _full_inputs_targets(
    batch: RachelBatch, device: torch.device
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    inputs = (
        _tensor(batch.mask_a, device, torch.float32),
        _tensor(batch.mask_b, device, torch.float32),
        _tensor(batch.points_rc_a, device, torch.float32),
        _tensor(batch.points_rc_b, device, torch.float32),
        _tensor(batch.contour_valid_a, device, torch.bool),
        _tensor(batch.contour_valid_b, device, torch.bool),
    )
    targets = (
        _tensor(batch.target_a, device, torch.long),
        _tensor(batch.target_b, device, torch.long),
        _tensor(batch.translation_a_to_b_rc, device, torch.float32),
        _tensor(batch.translation_valid, device, torch.bool),
    )
    return inputs, targets


def _coarse_inputs(batch: RachelBatch, device: torch.device) -> Tuple[Tensor, Tensor]:
    first = _tensor(batch.mask_a, device, torch.float32)
    second = _tensor(batch.mask_b, device, torch.float32)
    return (
        F.interpolate(first, size=(128, 128), mode="nearest"),
        F.interpolate(second, size=(128, 128), mode="nearest"),
    )


def _correspondence_per_sample(
    assignment: Tensor,
    unmatched_a: Tensor,
    unmatched_b: Tensor,
    target_a: Tensor,
    target_b: Tensor,
    sample_valid: Tensor,
) -> Dict[str, np.ndarray]:
    row_value, row_index = assignment.max(dim=2)
    col_value, col_index = assignment.max(dim=1)
    count_a = assignment.shape[1]
    indices_a = torch.arange(count_a, device=assignment.device)[None, :]
    reciprocal_a = col_index.gather(1, row_index) == indices_a
    reciprocal_real = (
        reciprocal_a
        & (row_value > unmatched_a)
        & (col_value.gather(1, row_index) > unmatched_b.gather(1, row_index))
    )
    valid_a = target_a != -2
    valid_b = target_b != -2
    predicted_match = reciprocal_real & valid_a
    predicted_match = predicted_match & sample_valid[:, None]
    true_match = target_a >= 0
    true_positive = predicted_match & (row_index == target_a)
    mutual_candidate = reciprocal_a & valid_a & sample_valid[:, None]
    mutual_true_positive = mutual_candidate & (row_index == target_a)
    predicted_dustbin_a = unmatched_a >= row_value
    col_best, _ = assignment.max(dim=1)
    predicted_dustbin_b = unmatched_b >= col_best
    dustbin_correct = (
        (predicted_dustbin_a == (target_a == -1)) & valid_a & sample_valid[:, None]
    ).sum(dim=1) + (
        (predicted_dustbin_b == (target_b == -1)) & valid_b & sample_valid[:, None]
    ).sum(dim=1)
    dustbin_total = valid_a.sum(dim=1) + valid_b.sum(dim=1)

    def numpy_counts(value: Tensor) -> np.ndarray:
        return value.detach().to(dtype=torch.int64).cpu().numpy()

    return {
        "strict_true_positive": numpy_counts(true_positive.sum(dim=1)),
        "strict_predicted_count": numpy_counts(predicted_match.sum(dim=1)),
        "mutual_top1_true_positive": numpy_counts(mutual_true_positive.sum(dim=1)),
        "mutual_top1_predicted_count": numpy_counts(mutual_candidate.sum(dim=1)),
        "target_count": numpy_counts(true_match.sum(dim=1)),
        "dustbin_correct": numpy_counts(dustbin_correct),
        "dustbin_total": numpy_counts(dustbin_total),
    }


def _ordered_polygon_area_rc(points_rc: np.ndarray) -> float:
    """PairingNet/OpenCV polygon area after the released int32 quantization."""

    raw_points = np.asarray(points_rc)
    points = np.asarray(raw_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
        raise RachelN512SealedTestError(
            "PairingNet-style contour area requires at least three ordered points"
        )
    if not np.isfinite(points).all():
        raise RachelN512SealedTestError(
            "PairingNet-style contour area received non-finite points"
        )
    # PairingNet's matching_test.py casts contours to np.int32 before calling
    # cv2.contourArea.  Preserve that quantization even though Rachel's
    # smoothed/resampled contour coordinates are fractional.
    quantized = points.astype(np.int32).astype(np.float64)
    row = quantized[:, 0]
    column = quantized[:, 1]
    return float(
        abs(np.dot(column, np.roll(row, -1)) - np.dot(row, np.roll(column, -1))) * 0.5
    )


def _symmetric_hausdorff(first: np.ndarray, second: np.ndarray) -> float:
    """Exact symmetric Hausdorff distance without adding a SciPy dependency."""

    source = np.asarray(first, dtype=np.float64)
    target = np.asarray(second, dtype=np.float64)
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1:] != (2,)
        or target.shape[1:] != (2,)
        or not len(source)
        or not len(target)
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise RachelN512SealedTestError(
            "PairingNet-style Hausdorff inputs are empty or malformed"
        )

    def directed(left: np.ndarray, right: np.ndarray) -> float:
        nearest = np.full(len(left), np.inf, dtype=np.float64)
        for start in range(0, len(right), 128):
            block = right[start : start + 128]
            distance = np.linalg.norm(left[:, None, :] - block[None, :, :], axis=2)
            nearest = np.minimum(nearest, distance.min(axis=1))
        return float(nearest.max())

    return max(directed(source, target), directed(target, source))


def _pairingnet_registration_per_sample(
    batch: RachelBatch,
    translation_prediction: np.ndarray,
    decision_valid: np.ndarray,
) -> Tuple[Dict[str, object], ...]:
    """Compute translation-only PairingNet matching metrics for every pair.

    The compatibility eRMSE intentionally mirrors PairingNet's released code:
    ``sqrt(mean(per-correspondence Euclidean distance))``.  It is not silently
    replaced by the conventional root-mean-square Euclidean error.
    """

    predictions = np.asarray(translation_prediction, dtype=np.float64)
    valid = np.asarray(decision_valid, dtype=np.bool_)
    if predictions.shape != (len(batch.pair_ids), 2) or valid.shape != (
        len(batch.pair_ids),
    ):
        raise RachelN512SealedTestError(
            "PairingNet-style registration prediction shape differs"
        )
    rows = []
    for index in range(len(batch.pair_ids)):
        target_valid = bool(batch.translation_valid[index])
        prediction_valid = bool(valid[index])
        result: Dict[str, object] = {
            "target_valid": target_valid,
            "prediction_valid": prediction_valid,
            "identity_fallback_used": (not prediction_valid if target_valid else None),
            "e_rmse": None,
            "registration_recall_lt4_success": (False if target_valid else None),
            "symmetric_hausdorff_px": None,
            "translation_l2_px": None,
            "normalized_translation_error": None,
            "source_contour_area_px2": None,
            "target_contour_area_px2": None,
            "rotation_error": {
                "status": "not_applicable_conditioned_upright_orientation",
                "estimated_or_supervised": False,
                "ground_truth_rotation_degrees": 0.0,
            },
        }
        if not target_valid:
            rows.append(result)
            continue
        matched_a = np.flatnonzero(batch.target_a[index] >= 0)
        if not len(matched_a):
            raise RachelN512SealedTestError(
                "positive PairingNet-style registration target has no seam points"
            )
        matched_b = batch.target_a[index, matched_a].astype(np.int64, copy=False)
        source = batch.points_rc_a[index, matched_a].astype(np.float64, copy=False)
        target = batch.points_rc_b[index, matched_b].astype(np.float64, copy=False)
        effective_prediction = (
            predictions[index] if prediction_valid else np.zeros(2, dtype=np.float64)
        )
        transformed = source + effective_prediction[None, :]
        residual = np.linalg.norm(transformed - target, axis=1)
        e_rmse = float(np.sqrt(residual.mean()))
        hausdorff = _symmetric_hausdorff(transformed, target)
        translation_l2 = float(
            np.linalg.norm(
                effective_prediction
                - batch.translation_a_to_b_rc[index].astype(np.float64)
            )
        )
        source_contour = batch.points_rc_a[index, batch.contour_valid_a[index]]
        target_contour = batch.points_rc_b[index, batch.contour_valid_b[index]]
        source_area = _ordered_polygon_area_rc(source_contour)
        target_area = _ordered_polygon_area_rc(target_contour)
        area_sum = source_area + target_area
        if area_sum <= 0.0:
            raise RachelN512SealedTestError(
                "PairingNet-style normalized translation area is non-positive"
            )
        values = (e_rmse, hausdorff, translation_l2, source_area, target_area)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise RachelN512SealedTestError(
                "PairingNet-style registration produced a non-finite metric"
            )
        result.update(
            {
                "e_rmse": e_rmse,
                "registration_recall_lt4_success": (
                    e_rmse < PAIRINGNET_REGISTRATION_RECALL_THRESHOLD
                ),
                "symmetric_hausdorff_px": hausdorff,
                "translation_l2_px": translation_l2,
                "normalized_translation_error": translation_l2 / area_sum,
                "source_contour_area_px2": source_area,
                "target_contour_area_px2": target_area,
            }
        )
        rows.append(result)
    return tuple(rows)


def _prf(true_positive: int, predicted: int, target: int) -> Dict[str, object]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / target if target else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "predicted_count": predicted,
        "target_count": target,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _coverage(valid: Sequence[bool], labels: Sequence[bool]) -> Dict[str, object]:
    valid_array = np.asarray(valid, dtype=np.bool_)
    label_array = np.asarray(labels, dtype=np.bool_)
    positive = label_array
    negative = ~label_array
    return {
        "valid_count": int(valid_array.sum()),
        "record_count": int(len(valid_array)),
        "valid_fraction": float(valid_array.mean()),
        "positive_valid_count": int((valid_array & positive).sum()),
        "positive_count": int(positive.sum()),
        "positive_valid_fraction": float(valid_array[positive].mean()),
        "negative_valid_count": int((valid_array & negative).sum()),
        "negative_count": int(negative.sum()),
        "negative_valid_fraction": float(valid_array[negative].mean()),
    }


def _ranking_only(
    probability: Sequence[float],
    labels: Sequence[bool],
    valid: Sequence[bool],
    clusters: Sequence[str],
) -> Dict[str, object]:
    value = evaluate_pairwise(probability, labels, valid, clusters, threshold=0.5)
    return {
        "sample_count": value["sample_count"],
        "positive_count": value["positive_count"],
        "negative_count": value["negative_count"],
        "cluster_count": value["cluster_count"],
        "row": {
            "auroc": value["row"]["auroc"],
            "auprc": value["row"]["auprc"],
        },
        "cluster_balanced": {
            "auroc": value["cluster_balanced"]["auroc"],
            "auprc": value["cluster_balanced"]["auprc"],
        },
        "coverage": _coverage(valid, labels),
        "thresholded_metrics_reported": False,
    }


def _evaluate_frozen_arm(
    winner: _FrozenWinner,
    loader: Iterable[RachelBatch],
    manifest_rows: Sequence[_ManifestRow],
    *,
    device: torch.device,
    precision: str,
) -> Tuple[Dict[str, object], Tuple[Dict[str, object], ...]]:
    model = winner.model.to(device)
    model.eval()
    probabilities: Dict[str, list] = {"coarse": [], "local": [], "fused": []}
    validities: Dict[str, list] = {"coarse": [], "local": [], "fused": []}
    labels: list = []
    clusters: list = []
    pair_ids: list = []
    records: list = []
    correspondence_totals = {
        "strict_true_positive": 0,
        "strict_predicted_count": 0,
        "mutual_top1_true_positive": 0,
        "mutual_top1_predicted_count": 0,
        "target_count": 0,
        "dustbin_correct": 0,
        "dustbin_total": 0,
    }
    translation_errors: list = []
    translation_error_by_pair: list = []
    translation_eligible_by_pair: list = []
    main_probability_by_pair: list = []
    main_valid_by_pair: list = []
    pairingnet_registration_by_pair: list = []
    cursor = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            count = len(batch.pair_ids)
            expected = manifest_rows[cursor : cursor + count]
            if tuple(row.pair_id for row in expected) != tuple(batch.pair_ids):
                raise RachelN512SealedTestError(
                    "test loader order/identity disagrees with frozen manifest"
                )
            expected_labels = np.asarray(
                [row.label for row in expected], dtype=np.float32
            )
            if not np.array_equal(batch.labels, expected_labels):
                raise RachelN512SealedTestError(
                    "test loader labels disagree with frozen manifest"
                )
            if not np.array_equal(
                batch.translation_valid,
                expected_labels.astype(np.bool_),
            ):
                raise RachelN512SealedTestError(
                    "test translation-valid flags disagree with positive labels"
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                if winner.arm == "coarse_only":
                    first, second = _coarse_inputs(batch, device)
                    output = model(first, second)
                    batch_probabilities = {"coarse": output.probability}
                    batch_validities = {"coarse": output.valid_problem}
                    main_probability = output.probability
                    main_valid = output.valid_problem
                    correspondence = None
                    translation_prediction = None
                    translation_error = None
                else:
                    assert isinstance(model, RachelN512Pairwise)
                    inputs, targets = _full_inputs_targets(batch, device)
                    output = model(*inputs)
                    batch_probabilities = {
                        "coarse": output.coarse_probability,
                        "local": output.local_probability,
                        "fused": output.fused_probability,
                    }
                    batch_validities = {
                        "coarse": output.coarse.valid_problem,
                        "local": output.decision_valid,
                        "fused": output.decision_valid,
                    }
                    main_probability = output.fused_probability
                    main_valid = output.decision_valid
                    correspondence = _correspondence_per_sample(
                        output.assignment,
                        output.unmatched_a,
                        output.unmatched_b,
                        targets[0],
                        targets[1],
                        main_valid,
                    )
                    translation_prediction = (
                        output.translation_hat_rc.detach().float().cpu().numpy()
                    )
                    translation_error = (
                        torch.linalg.vector_norm(
                            output.translation_hat_rc - targets[2], dim=1
                        )
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )
                    pairingnet_registration = _pairingnet_registration_per_sample(
                        batch,
                        translation_prediction,
                        main_valid.detach().cpu().numpy().astype(np.bool_, copy=False),
                    )

            cpu_probabilities = {
                name: value.detach().float().cpu().numpy()
                for name, value in batch_probabilities.items()
            }
            cpu_validities = {
                name: value.detach().cpu().numpy().astype(np.bool_, copy=False)
                for name, value in batch_validities.items()
            }
            main_probability_cpu = main_probability.detach().float().cpu().numpy()
            main_valid_cpu = (
                main_valid.detach().cpu().numpy().astype(np.bool_, copy=False)
            )
            for name, value in cpu_probabilities.items():
                probabilities[name].extend(float(item) for item in value)
                validities[name].extend(bool(item) for item in cpu_validities[name])

            for index, row in enumerate(expected):
                probability_value = float(main_probability_cpu[index])
                valid_value = bool(main_valid_cpu[index])
                prediction = (
                    probability_value >= winner.threshold.threshold
                    if valid_value and math.isfinite(probability_value)
                    else None
                )
                scores = {
                    name: {
                        "probability": float(cpu_probabilities[name][index]),
                        "valid": bool(cpu_validities[name][index]),
                    }
                    for name in batch_probabilities
                }
                record: Dict[str, object] = {
                    "schema_version": "rachel-n512-sealed-test-pair/1.0",
                    "arm": winner.arm,
                    "pair_id": row.pair_id,
                    "label": row.label,
                    "cluster_id": row.cluster_id,
                    "source_unit_ids": list(row.source_unit_ids),
                    "scores": scores,
                    "main_score": "coarse" if winner.arm == "coarse_only" else "fused",
                    "decision": {
                        "validation_threshold": winner.threshold.threshold,
                        "valid": valid_value,
                        "predicted_label": prediction,
                    },
                }
                eligible = bool(batch.translation_valid[index])
                error_value: Optional[float] = None
                if winner.arm == "full_n512":
                    assert correspondence is not None
                    assert translation_prediction is not None
                    assert translation_error is not None
                    for name in correspondence_totals:
                        correspondence_totals[name] += int(correspondence[name][index])
                    if eligible and valid_value:
                        error_value = float(translation_error[index])
                        if not math.isfinite(error_value):
                            raise RachelN512SealedTestError(
                                "valid translation error is non-finite"
                            )
                        translation_errors.append(error_value)
                    record["geometry"] = {
                        "decision_valid": valid_value,
                        "translation_target_valid": eligible,
                        "translation_hat_rc": [
                            float(item) for item in translation_prediction[index]
                        ],
                        "translation_l2_px": error_value,
                        "correspondence": {
                            name: int(correspondence[name][index])
                            for name in correspondence
                        },
                        "pairingnet_style_registration": dict(
                            pairingnet_registration[index]
                        ),
                        "assembly_edge": {
                            "target_edge": row.label,
                            "predicted_edge": bool(prediction),
                            "true_positive_by_tolerance": {
                                "at_{}".format(tolerance): bool(
                                    row.label
                                    and prediction
                                    and error_value is not None
                                    and error_value <= tolerance
                                )
                                for tolerance in ASSEMBLY_EDGE_TOLERANCES
                            },
                        },
                    }
                    pairingnet_registration_by_pair.append(
                        dict(pairingnet_registration[index])
                    )
                records.append(record)
                labels.append(row.label)
                clusters.append(row.cluster_id)
                pair_ids.append(row.pair_id)
                translation_error_by_pair.append(error_value)
                translation_eligible_by_pair.append(eligible)
                main_probability_by_pair.append(probability_value)
                main_valid_by_pair.append(valid_value)
            cursor += count

    if cursor != len(manifest_rows) or len(set(pair_ids)) != len(manifest_rows):
        raise RachelN512SealedTestError("test evaluation coverage is incomplete")
    methods = (
        ("coarse",)
        if winner.arm == "coarse_only"
        else (
            "coarse",
            "local",
            "fused",
        )
    )
    main_method = "coarse" if winner.arm == "coarse_only" else "fused"
    main_metrics = evaluate_pairwise(
        probabilities[main_method],
        labels,
        validities[main_method],
        clusters,
        threshold=winner.threshold.threshold,
    )
    main_metrics["coverage"] = _coverage(validities[main_method], labels)
    metrics: Dict[str, object] = {
        "sample_count": len(pair_ids),
        "pair_ids_sha256": hashlib.sha256(_canonical_bytes(pair_ids)).hexdigest(),
        "seconds": time.perf_counter() - started,
        "main_method": main_method,
        "main_pairwise": main_metrics,
        "method_ranking": {
            name: _ranking_only(probabilities[name], labels, validities[name], clusters)
            for name in methods
        },
        "threshold": {
            "value": winner.threshold.threshold,
            "source_split": winner.threshold.source_split,
            "artifact_sha256": winner.threshold.content_sha256,
            "fit_performed_during_test": False,
        },
    }
    if winner.arm == "full_n512":
        strict = _prf(
            correspondence_totals["strict_true_positive"],
            correspondence_totals["strict_predicted_count"],
            correspondence_totals["target_count"],
        )
        mutual = _prf(
            correspondence_totals["mutual_top1_true_positive"],
            correspondence_totals["mutual_top1_predicted_count"],
            correspondence_totals["target_count"],
        )
        errors = np.asarray(translation_errors, dtype=np.float64)
        eligible_array = np.asarray(translation_eligible_by_pair, dtype=np.bool_)
        main_valid_array = np.asarray(main_valid_by_pair, dtype=np.bool_)
        main_probability_array = np.asarray(main_probability_by_pair, dtype=np.float64)
        error_array = np.asarray(
            [np.nan if value is None else value for value in translation_error_by_pair],
            dtype=np.float64,
        )
        positive_total = int(eligible_array.sum())
        valid_positive = eligible_array & main_valid_array & np.isfinite(error_array)
        predicted_positive = (
            main_valid_array
            & np.isfinite(main_probability_array)
            & (main_probability_array >= winner.threshold.threshold)
        )
        tolerances = (2, 5, 8, 10)
        success = {
            "success_at_{}px".format(tolerance): (
                float(np.mean(errors <= tolerance)) if errors.size else None
            )
            for tolerance in tolerances
        }
        joint = {}
        for tolerance in tolerances:
            successful = (
                eligible_array
                & valid_positive
                & predicted_positive
                & (error_array <= tolerance)
            )
            joint["success_at_{}px".format(tolerance)] = (
                float(successful.sum() / positive_total) if positive_total else None
            )
            joint["success_count_at_{}px".format(tolerance)] = int(successful.sum())
        metrics["decision_coverage"] = _coverage(main_valid_by_pair, labels)
        metrics["correspondence"] = {
            "scope": (
                "entire_sealed_population; decision-invalid samples contribute no "
                "predicted matches/correct dustbins and retain their supervised targets"
            ),
            "strict_dustbin_aware": strict,
            "mutual_top1": mutual,
            "dustbin_accuracy": (
                correspondence_totals["dustbin_correct"]
                / correspondence_totals["dustbin_total"]
                if correspondence_totals["dustbin_total"]
                else 0.0
            ),
            "dustbin_correct": correspondence_totals["dustbin_correct"],
            "dustbin_token_count": correspondence_totals["dustbin_total"],
        }
        metrics["translation"] = {
            "scope": "positive_translation_targets_with_decision_valid",
            "eligible_positive_count": positive_total,
            "valid_count": int(errors.size),
            "valid_fraction": float(errors.size / positive_total)
            if positive_total
            else 0.0,
            "median_l2_px": float(np.median(errors)) if errors.size else None,
            "p90_l2_px": float(np.quantile(errors, 0.9)) if errors.size else None,
            **success,
        }
        metrics["joint_success"] = {
            "definition": (
                "positive_pair_and_decision_valid_and_pair_score_at_or_above_"
                "validation_threshold_and_translation_error_within_tolerance;_"
                "invalid_or_classification_rejected_pairs_count_as_failures"
            ),
            "denominator_positive_count": positive_total,
            **joint,
        }
        if len(pairingnet_registration_by_pair) != len(manifest_rows):
            raise RachelN512SealedTestError(
                "PairingNet-style registration coverage is incomplete"
            )
        predicted_pose_valid = np.asarray(
            [
                row["target_valid"] and row["prediction_valid"]
                for row in pairingnet_registration_by_pair
            ],
            dtype=np.bool_,
        )
        positive = np.asarray(labels, dtype=np.bool_)
        e_rmse = np.asarray(
            [
                np.nan if row["e_rmse"] is None else float(row["e_rmse"])
                for row in pairingnet_registration_by_pair
            ],
            dtype=np.float64,
        )
        hausdorff = np.asarray(
            [
                np.nan
                if row["symmetric_hausdorff_px"] is None
                else float(row["symmetric_hausdorff_px"])
                for row in pairingnet_registration_by_pair
            ],
            dtype=np.float64,
        )
        nte = np.asarray(
            [
                np.nan
                if row["normalized_translation_error"] is None
                else float(row["normalized_translation_error"])
                for row in pairingnet_registration_by_pair
            ],
            dtype=np.float64,
        )
        registration_evaluable = (
            positive & np.isfinite(e_rmse) & np.isfinite(hausdorff) & np.isfinite(nte)
        )
        if int(registration_evaluable.sum()) != int(positive.sum()):
            raise RachelN512SealedTestError(
                "PairingNet-style identity-fallback registration is not complete"
            )
        rr_success = positive & (e_rmse < PAIRINGNET_REGISTRATION_RECALL_THRESHOLD)
        metrics["pairingnet_style_registration"] = {
            "compatibility_source": (
                "PairingNet released matching_test.py; upright translation-only "
                "specialization"
            ),
            "e_rmse_definition": (
                "sqrt(mean(per_correspondence_euclidean_distance)); preserved "
                "verbatim for benchmark compatibility, not conventional RMSE"
            ),
            "registration_recall_definition": "e_rmse_strictly_less_than_4",
            "hausdorff_definition": (
                "max(directed_HD(transformed_source_seam,target_seam),"
                "directed_HD(target_seam,transformed_source_seam))"
            ),
            "normalized_translation_error_definition": (
                "identity-fallback translation_l2_px divided by the sum of ordered "
                "N512 contour polygon areas after PairingNet's int32 quantization"
            ),
            "invalid_pose_compatibility_fallback": (
                "identity_translation_for_unconditional_official_style_aggregation"
            ),
            "eligible_positive_count": int(positive.sum()),
            "evaluated_positive_count": int(registration_evaluable.sum()),
            "valid_pose_count": int(predicted_pose_valid.sum()),
            "identity_fallback_count": int(positive.sum() - predicted_pose_valid.sum()),
            "valid_pose_fraction": float(predicted_pose_valid.sum() / positive.sum()),
            "mean_e_rmse": float(e_rmse[registration_evaluable].mean()),
            "registration_recall_lt4": float(rr_success.sum() / positive.sum()),
            "registration_recall_lt4_success_count": int(rr_success.sum()),
            "mean_symmetric_hausdorff_px": float(
                hausdorff[registration_evaluable].mean()
            ),
            "mean_normalized_translation_error": float(
                nte[registration_evaluable].mean()
            ),
            "rotation_error": {
                "status": "not_applicable_conditioned_upright_orientation",
                "estimated_or_supervised": False,
                "reason": (
                    "Dunhuang inputs are pre-oriented; the model has no rotation "
                    "degree of freedom"
                ),
            },
        }
        predicted_edge = (
            main_valid_array
            & np.isfinite(main_probability_array)
            & (main_probability_array >= winner.threshold.threshold)
        )
        assembly_by_tolerance = {}
        for tolerance in ASSEMBLY_EDGE_TOLERANCES:
            true_positive = int(
                (
                    positive
                    & predicted_edge
                    & valid_positive
                    & (error_array <= tolerance)
                ).sum()
            )
            row = _prf(
                true_positive,
                int(predicted_edge.sum()),
                int(positive.sum()),
            )
            row["false_positive_count"] = row["predicted_count"] - true_positive
            row["false_negative_count"] = row["target_count"] - true_positive
            assembly_by_tolerance["at_{}".format(tolerance)] = row
        metrics["assembly_edge"] = {
            "definition": (
                "one predicted assembly edge is emitted only when decision-valid and "
                "above the frozen validation threshold; it is a true positive only "
                "when the pair is adjacent and translation L2 error is at or below "
                "the stated pixel tolerance. A positive wrong-pose prediction is both "
                "an unmatched predicted edge and a missed target edge"
            ),
            "validation_threshold": winner.threshold.threshold,
            "predicted_edge_count": int(predicted_edge.sum()),
            "target_edge_count": int(positive.sum()),
            "by_tolerance": assembly_by_tolerance,
        }
    return metrics, tuple(records)


def _evaluate_matched_winner(
    winner: FrozenMatchedMMWinner,
    manifest_rows: Sequence[_ManifestRow],
    *,
    batch_size: int,
) -> Tuple[Dict[str, object], Tuple[Mapping[str, object], ...]]:
    """Evaluate one matched control from model-mask paths only."""

    pairs = []
    for row in manifest_rows:
        if row.mask_a_path is None or row.mask_b_path is None:
            raise RachelN512SealedTestError(
                "matched-MM model-mask endpoints were not frozen from manifest"
            )
        pairs.append(SyntheticMaskPair(row.pair_id, row.mask_a_path, row.mask_b_path))
    started = time.perf_counter()
    predictions = score_synthetic_mask_pairs(winner, pairs, batch_size=batch_size)
    pair_ids = [row.pair_id for row in manifest_rows]
    labels = [row.label for row in manifest_rows]
    clusters = [row.cluster_id for row in manifest_rows]
    probability = [row.probability for row in predictions]
    valid = [row.valid for row in predictions]
    main = matched_metrics(
        predictions,
        labels,
        clusters,
        threshold=winner.threshold.threshold,
    )
    raw_records = synthetic_pair_records(winner, predictions, manifest_rows)
    records = tuple(
        {**dict(record), "source_unit_ids": list(row.source_unit_ids)}
        for record, row in zip(raw_records, manifest_rows)
    )
    metrics: Dict[str, object] = {
        "sample_count": len(pair_ids),
        "pair_ids_sha256": hashlib.sha256(_canonical_bytes(pair_ids)).hexdigest(),
        "seconds": time.perf_counter() - started,
        "main_method": MATCHED_SCORE,
        "main_pairwise": main,
        "method_ranking": {
            MATCHED_SCORE: _ranking_only(probability, labels, valid, clusters)
        },
        "threshold": {
            "value": winner.threshold.threshold,
            "source_split": winner.threshold.source_split,
            "artifact_sha256": winner.threshold.content_sha256,
            "fit_performed_during_test": False,
        },
        "input_contract": {
            "source": "model/masks_800_binary_0_255_only",
            "preprocess": "PIL_bilinear_64x64_single_channel",
            "rgb_used": False,
            "contour_or_geometry_used": False,
        },
    }
    return metrics, records


def _set_determinism(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RachelN512SealedTestError(
            "CUBLAS deterministic workspace must be configured before torch import"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_sealed_synthetic_test(config: RachelN512SealedTestConfig) -> Path:
    """Evaluate frozen validation winners under the fixed synthetic protocol."""

    # This is the critical ordering invariant: no test path is resolved/read
    # until all completed-run metadata and checkpoint states are frozen here.
    run_directory = config.run_directory.resolve(strict=True)
    receipt, receipt_sha256, winners = _freeze_completed_winners(run_directory)
    _require_formal_convergence(receipt, winners)
    formal_evaluation = not config.compatibility_mode
    if formal_evaluation and (
        config.matched_mm_run_directory is None
        or config.pairingnet_run_directory is None
        or config.shreddingnet_freeze_path is None
    ):
        raise RachelN512SealedTestError(
            "formal sealed evaluation requires exact-six frozen authorities"
        )
    canonical_config_authority = _canonical_config_authority(winners)
    train_config = receipt["config"]
    assert isinstance(train_config, Mapping)
    precision = train_config.get("precision")
    seed = train_config.get("seed")
    if precision not in {"fp32", "bf16"} or type(seed) is not int or seed < 0:  # noqa: E721
        raise RachelN512SealedTestError("training precision/seed is invalid")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RachelN512SealedTestError(
            "requested sealed-test CUDA device is unavailable"
        )
    if precision == "bf16" and device.type not in {"cuda", "cpu"}:
        raise RachelN512SealedTestError("bf16 evaluation device is unsupported")

    matched_receipt: Optional[Mapping[str, object]] = None
    matched_receipt_sha256: Optional[str] = None
    matched_winners: Mapping[str, FrozenMatchedMMWinner] = {}
    matched_run_directory: Optional[Path] = None
    matched_training_hash_evidence: Optional[Mapping[str, object]] = None
    if config.matched_mm_run_directory is not None:
        matched_run_directory = config.matched_mm_run_directory.resolve(strict=True)
        (
            matched_receipt,
            matched_receipt_sha256,
            matched_winners,
        ) = freeze_matched_mm_winners(matched_run_directory, device=device)
        if tuple(matched_winners) != MATCHED_METHODS:
            raise RachelN512SealedTestError(
                "matched-MM converged/same-exposure winners are incomplete"
            )
        _require_matched_training_alignment(receipt, matched_receipt)
        matched_training_hash_evidence = _matched_training_hash_evidence(
            receipt,
            matched_receipt,
            n512_run_directory=run_directory,
            matched_run_directory=matched_run_directory,
        )
    benchmark_winners: Mapping[str, benchmark_adapter.FrozenSameDataBenchmark] = {}
    if config.pairingnet_run_directory is not None:
        assert config.shreddingnet_freeze_path is not None
        try:
            benchmark_winners = benchmark_adapter.freeze_same_data_benchmarks(
                pairingnet_run_directory=config.pairingnet_run_directory,
                shreddingnet_freeze_path=config.shreddingnet_freeze_path,
                device=device,
            )
        except benchmark_adapter.RachelBenchmarkEvalAdapterError as error:
            raise RachelN512SealedTestError(
                "same-data benchmark freeze failed: " + str(error)
            ) from error
    if formal_evaluation and (
        tuple(matched_winners) != MATCHED_METHODS
        or tuple(benchmark_winners) != benchmark_adapter.BENCHMARK_METHODS
    ):
        raise RachelN512SealedTestError(
            "formal sealed evaluation requires exact-six frozen winners"
        )
    for winner in winners:
        model_config = winner.model_config
        if (
            model_config.canvas_size != 800
            or model_config.coarse_size != 128
            or model_config.contour_cap != 512
        ):
            raise RachelN512SealedTestError(
                "winner model does not match Rachel 800/coarse128/N512 release"
            )
    _set_determinism(seed)

    dataset_root = _resolve_evaluation_dataset_root(receipt, config.dataset_root)
    benchmark_training_evidence = (
        _benchmark_training_manifest_evidence(dataset_root, benchmark_winners)
        if benchmark_winners
        else None
    )
    output_root = config.output_root.resolve()
    protected_roots = [run_directory, dataset_root]
    if matched_run_directory is not None:
        protected_roots.append(matched_run_directory)
    if benchmark_winners:
        protected_roots.extend(
            (
                config.pairingnet_run_directory.resolve(strict=True),
                config.shreddingnet_freeze_path.resolve(strict=True).parent,
            )
        )
    for protected in protected_roots:
        try:
            output_root.relative_to(protected)
        except ValueError:
            continue
        raise RachelN512SealedTestError(
            "sealed-test output must not mutate a training run or dataset release"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    # First sealed-test access occurs here, after every winner was restored.
    test_manifest_path, manifest_rows = _test_manifest_rows(
        dataset_root, require_model_masks=bool(matched_winners)
    )
    test_manifest_sha256 = _sha256_file(test_manifest_path)
    test_dataset = RachelPairDataset(dataset_root, "test")
    if len(test_dataset) != len(manifest_rows):
        raise RachelN512SealedTestError("test loader/manifest population mismatch")
    test_loader = _test_loader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=seed + 202_609_01,
    )

    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "formal_evaluation": formal_evaluation,
        "compatibility_mode": config.compatibility_mode,
        "training_receipt_sha256": receipt_sha256,
        "training_fingerprint_sha256": receipt.get("fingerprint_sha256"),
        "test_manifest_sha256": test_manifest_sha256,
        "expected_test_pairs": EXPECTED_TEST_PAIRS,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "device": config.device,
        "precision": precision,
        "threshold_source": "completed_training_receipt_validation_threshold",
        "canonical_n512_config_authority_sha256": hashlib.sha256(
            _canonical_bytes(canonical_config_authority)
        ).hexdigest(),
        "matched_mm": (
            {
                "training_receipt_sha256": matched_receipt_sha256,
                "training_alignment_evidence_sha256": hashlib.sha256(
                    _canonical_bytes(matched_training_hash_evidence)
                ).hexdigest(),
                "methods": {
                    method: {
                        "checkpoint_sha256": matched_winners[method].checkpoint_sha256,
                        "validation_threshold_sha256": (
                            matched_winners[method].threshold.content_sha256
                        ),
                    }
                    for method in MATCHED_METHODS
                },
            }
            if matched_winners
            else None
        ),
        "same_data_benchmarks": (
            {
                "training_alignment_evidence_sha256": hashlib.sha256(
                    _canonical_bytes(benchmark_training_evidence)
                ).hexdigest(),
                "methods": {
                    method: {
                        "method_id": benchmark_winners[method].method_id,
                        "freeze_authority_sha256": (
                            benchmark_winners[method].freeze_authority_sha256
                        ),
                        "checkpoint_sha256_by_stage": dict(
                            benchmark_winners[method].checkpoint_sha256_by_stage
                        ),
                        "validation_threshold_sha256": (
                            benchmark_winners[method].threshold_artifact_sha256
                        ),
                    }
                    for method in benchmark_adapter.BENCHMARK_METHODS
                },
            }
            if benchmark_winners
            else None
        ),
    }
    fingerprint = hashlib.sha256(_canonical_bytes(fingerprint_payload)).hexdigest()
    final_directory = output_root / ("test-" + fingerprint[:16])
    partial_directory = output_root / (".partial-test-" + fingerprint[:16])
    if final_directory.exists() or partial_directory.exists():
        raise RachelN512SealedTestError(
            "sealed-test output already exists; preserve it rather than overwriting"
        )
    partial_directory.mkdir()
    _atomic_json(
        partial_directory / "evaluation_config.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "running_formal_exact_six_frozen_synthetic_test"
                if formal_evaluation
                else "running_non_formal_compatibility_synthetic_evaluation"
            ),
            "formal_evaluation": formal_evaluation,
            "compatibility_mode": config.compatibility_mode,
            "fingerprint_sha256": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "run_directory": str(run_directory),
            "matched_mm_run_directory": (
                str(matched_run_directory) if matched_run_directory else None
            ),
            "pairingnet_run_directory": (
                str(config.pairingnet_run_directory.resolve(strict=True))
                if benchmark_winners
                else None
            ),
            "shreddingnet_freeze_path": (
                str(config.shreddingnet_freeze_path.resolve(strict=True))
                if benchmark_winners
                else None
            ),
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "test_accessed": True,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_fit_performed": False,
            "formal_config_verified": True,
            "both_arms_validation_plateau_verified": True,
            "evaluation_history_disclosure": {
                "evaluation_kind": "fixed_protocol_repeat",
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
    )
    started = time.perf_counter()
    arm_results = []
    try:
        manifest_by_arm_sha = hashlib.sha256(
            _canonical_bytes([row.pair_id for row in manifest_rows])
        ).hexdigest()
        for winner in winners:
            metrics, records = _evaluate_frozen_arm(
                winner,
                test_loader,
                manifest_rows,
                device=device,
                precision=precision,
            )
            score_path = partial_directory / winner.arm / "pair_scores.jsonl"
            _atomic_jsonl(score_path, records)
            score_sha = _sha256_file(score_path)
            arm_result = {
                "arm": winner.arm,
                "winner_epoch": winner.epoch,
                "winner_checkpoint": str(winner.checkpoint_path),
                "winner_checkpoint_sha256": winner.checkpoint_sha256,
                "model_config": asdict(winner.model_config),
                "model_config_sha256": winner.model_config_sha256,
                "loss_config": asdict(winner.loss_config),
                "loss_config_sha256": winner.loss_config_sha256,
                "validation_threshold": winner.threshold.to_dict(),
                "validation_threshold_sha256": winner.threshold.content_sha256,
                "pair_scores": str(score_path.relative_to(partial_directory)),
                "pair_scores_sha256": score_sha,
                "pair_scores_count": len(records),
                "metrics": metrics,
                "direct_pairwise_metric_status": (
                    {
                        "status": "reported",
                        "metrics": [
                            "TE_px",
                            "assembly_edge_at_2_5_8_10px",
                            "PairingNet_RR_HD_NTE",
                            "correspondence",
                        ],
                        "rotation_error": "not_applicable_known_upright",
                    }
                    if winner.arm == "full_n512"
                    else {
                        "status": "not_applicable",
                        "reason": "coarse_only_method_emits_no_2d_translation_or_correspondence",
                        "rotation_error": "not_applicable_known_upright",
                    }
                ),
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "threshold_fit_performed": False,
            }
            _atomic_json(
                partial_directory / winner.arm / "test_result.json", arm_result
            )
            arm_results.append(arm_result)
            winner.model = winner.model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for method in MATCHED_METHODS:
            if method not in matched_winners:
                continue
            matched_winner = matched_winners[method]
            metrics, records = _evaluate_matched_winner(
                matched_winner,
                manifest_rows,
                batch_size=config.batch_size,
            )
            score_path = partial_directory / method / "pair_scores.jsonl"
            _atomic_jsonl(score_path, records)
            score_sha = _sha256_file(score_path)
            arm_result = {
                "arm": method,
                "model_family": "historical_mm_mobilenetv2_siamese",
                "score_semantics": MATCHED_SCORE,
                "same_exposure_epoch5": method == MATCHED_METHODS[1],
                "winner_epoch": matched_winner.epoch,
                "winner_checkpoint": str(matched_winner.checkpoint_path),
                "winner_checkpoint_sha256": matched_winner.checkpoint_sha256,
                "model_config": dict(matched_winner.model_config),
                "model_config_sha256": matched_winner.model_config_sha256,
                "validation_threshold": matched_winner.threshold.to_dict(),
                "validation_threshold_sha256": (
                    matched_winner.threshold.content_sha256
                ),
                "pair_scores": str(score_path.relative_to(partial_directory)),
                "pair_scores_sha256": score_sha,
                "pair_scores_count": len(records),
                "metrics": metrics,
                "direct_pairwise_metric_status": {
                    "status": "not_applicable",
                    "reason": "historical_matched_MM_emits_pairability_only",
                    "translation_TE_assembly_RR_HD_NTE_correspondence": None,
                    "rotation_error": "not_applicable_known_upright",
                },
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "threshold_fit_performed": False,
            }
            _atomic_json(partial_directory / method / "test_result.json", arm_result)
            arm_results.append(arm_result)
            matched_winner.model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if benchmark_winners:
            benchmark_manifest = tuple(
                benchmark_adapter.BenchmarkEvalManifestRow(
                    pair_id=row.pair_id,
                    label=row.label,
                    cluster_id=row.cluster_id,
                    source_unit_ids=row.source_unit_ids,
                )
                for row in manifest_rows
            )
            for method in benchmark_adapter.BENCHMARK_METHODS:
                benchmark = benchmark_winners[method]
                metrics, records = (
                    benchmark_adapter.evaluate_frozen_benchmark_synthetic(
                        benchmark,
                        test_loader,
                        benchmark_manifest,
                    )
                )
                score_path = partial_directory / method / "pair_scores.jsonl"
                _atomic_jsonl(score_path, records)
                score_sha = _sha256_file(score_path)
                threshold_checkpoint_sha = benchmark.threshold_artifact.get(
                    "checkpoint_sha256"
                )
                if not isinstance(threshold_checkpoint_sha, str):
                    raise RachelN512SealedTestError(
                        method + " frozen threshold checkpoint binding is missing"
                    )
                arm_result = {
                    "arm": method,
                    "method_id": benchmark.method_id,
                    "model_family": method,
                    "score_semantics": "adapted_pair_probability",
                    "adaptation_claim": (
                        "same_data_method_adaptation_not_exact_reproduction"
                    ),
                    "adaptation": dict(benchmark.adaptation_disclosure),
                    "winner_checkpoint": str(benchmark.freeze_authority_path),
                    "winner_checkpoint_sha256": threshold_checkpoint_sha,
                    "winner_checkpoint_sha256_by_stage": dict(
                        benchmark.checkpoint_sha256_by_stage
                    ),
                    "freeze_authority_sha256": benchmark.freeze_authority_sha256,
                    "validation_threshold": dict(benchmark.threshold_artifact),
                    "validation_threshold_sha256": (
                        benchmark.threshold_artifact_sha256
                    ),
                    "pair_scores": str(score_path.relative_to(partial_directory)),
                    "pair_scores_sha256": score_sha,
                    "pair_scores_count": len(records),
                    "metrics": metrics,
                    "direct_pairwise_metric_status": {
                        "status": "reported",
                        "metrics": [
                            "TE_px",
                            "assembly_edge_at_2_5_8_10px",
                            "PairingNet_RR_HD_NTE",
                            "correspondence_when_available",
                        ],
                        "rotation_error": "not_applicable_known_upright",
                        "native_global_assembly_GA": "not_applicable_pairwise_only",
                        "shreddingnet_native_CM_FM_SE": "not_reported",
                    },
                    "training_performed": False,
                    "checkpoint_selection_performed": False,
                    "threshold_fit_performed": False,
                    "sealed_population_forwarded_once": True,
                    "native_cm_fm_se_or_ga_claimed": False,
                }
                _atomic_json(
                    partial_directory / method / "test_result.json", arm_result
                )
                arm_results.append(arm_result)
                benchmark.release_to_cpu()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        receipt_out = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                FORMAL_COMPLETE_STATUS
                if formal_evaluation
                else COMPATIBILITY_COMPLETE_STATUS
            ),
            "formal_evaluation": formal_evaluation,
            "compatibility_mode": config.compatibility_mode,
            "fingerprint_sha256": fingerprint,
            "source_training_run": {
                "run_directory": str(run_directory),
                "receipt": "run_receipt.json",
                "receipt_sha256": receipt_sha256,
                "fingerprint_sha256": receipt.get("fingerprint_sha256"),
                "status_at_open": receipt.get("status"),
                "winners_frozen_before_test_open": True,
                "both_arms_validation_plateau_verified": True,
                "formal_config_verified": True,
                "canonical_model_and_loss_config_authority": (
                    canonical_config_authority
                ),
            },
            "source_matched_mm_training_run": (
                {
                    "run_directory": str(matched_run_directory),
                    "receipt": "run_receipt.json",
                    "receipt_sha256": matched_receipt_sha256,
                    "fingerprint_sha256": (
                        matched_receipt.get("fingerprint_sha256")
                        if matched_receipt is not None
                        else None
                    ),
                    "status_at_open": (
                        matched_receipt.get("status")
                        if matched_receipt is not None
                        else None
                    ),
                    "converged_and_epoch5_winners_frozen_before_test_open": True,
                    "formal_config_verified": True,
                    "training_alignment_hash_evidence": (
                        matched_training_hash_evidence
                    ),
                }
                if matched_winners
                else None
            ),
            "source_same_data_benchmark_training_runs": (
                {
                    "all_methods_frozen_before_test_open": True,
                    "training_manifest_alignment": benchmark_training_evidence,
                    "methods": {
                        method: benchmark_winners[method].provenance()
                        for method in benchmark_adapter.BENCHMARK_METHODS
                    },
                }
                if benchmark_winners
                else None
            ),
            "test_population": {
                "dataset_root": str(dataset_root),
                "manifest": str(test_manifest_path),
                "manifest_sha256": test_manifest_sha256,
                "pair_ids_sha256": manifest_by_arm_sha,
                "expected_count": EXPECTED_TEST_PAIRS,
                "observed_count": len(manifest_rows),
                "positive_count": sum(row.label for row in manifest_rows),
                "negative_count": sum(not row.label for row in manifest_rows),
                "exact_one_to_one": True,
                "cluster_count": len(set(row.cluster_id for row in manifest_rows)),
            },
            "protocol": {
                "formal_evaluation": formal_evaluation,
                "compatibility_mode": config.compatibility_mode,
                "formal_method_inventory": (
                    list(ARMS)
                    + list(MATCHED_METHODS)
                    + list(benchmark_adapter.BENCHMARK_METHODS)
                    if formal_evaluation
                    else None
                ),
                "formal_exact_six_frozen_before_test_open": (
                    True if formal_evaluation else False
                ),
                "precision": precision,
                "device": config.device,
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
                "main_threshold": "validation_fit_checkpoint_bound_artifact_only",
                "ranking_metrics": "row_and_equal_lineage_or_lineage_pair_cluster",
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "threshold_fit_performed": False,
                "cross_arm_winner_selected_on_test": False,
                "real_external_test_accessed": False,
                "formal_config_verified": True,
                "both_arms_validation_plateau_verified": True,
                "all_requested_winners_and_thresholds_frozen_before_current_test_open": True,
                "same_data_benchmark_methods_requested": bool(benchmark_winners),
                "same_data_benchmark_winners_and_thresholds_frozen_before_test_open": (
                    True if benchmark_winners else None
                ),
                "method_forward_population": {
                    "pair_count_per_method": len(manifest_rows),
                    "each_pair_forwarded_exactly_once_per_method": True,
                    "methods": list(ARMS)
                    + list(matched_winners)
                    + list(benchmark_winners),
                },
                "evaluation_history_disclosure": {
                    "evaluation_kind": "fixed_protocol_repeat",
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
            "seconds": time.perf_counter() - started,
            "test_accessed": True,
            "real_external_test_accessed": False,
        }
        _atomic_json(partial_directory / "test_receipt.json", receipt_out)
        for winner in winners:
            winner.model = winner.model.to("cpu")
        for matched_winner in matched_winners.values():
            matched_winner.model.to("cpu")
        for benchmark in benchmark_winners.values():
            benchmark.release_to_cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _publish_directory_no_replace(
            partial_directory,
            final_directory,
            completion_receipt="test_receipt.json",
        )
        return final_directory
    except BaseException as error:
        for benchmark in benchmark_winners.values():
            try:
                benchmark.release_to_cpu()
            except Exception:
                pass
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _atomic_json(
            partial_directory / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_after_sealed_test_open",
                "formal_evaluation": formal_evaluation,
                "compatibility_mode": config.compatibility_mode,
                "error_type": type(error).__name__,
                "error": str(error),
                "test_accessed": True,
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "threshold_fit_performed": False,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--matched-mm-run-directory", type=Path)
    parser.add_argument("--pairingnet-run-directory", type=Path)
    parser.add_argument("--shreddingnet-freeze-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compatibility-non-formal",
        action="store_true",
        help="permit legacy method inventories and emit only non-formal status",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    output = run_sealed_synthetic_test(
        RachelN512SealedTestConfig(
            run_directory=arguments.run_directory,
            output_root=arguments.output_root,
            dataset_root=arguments.dataset_root,
            matched_mm_run_directory=arguments.matched_mm_run_directory,
            pairingnet_run_directory=arguments.pairingnet_run_directory,
            shreddingnet_freeze_path=arguments.shreddingnet_freeze_path,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            device=arguments.device,
            compatibility_mode=arguments.compatibility_non_formal,
        )
    )
    print(str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_TEST_PAIRS",
    "RachelN512SealedTestConfig",
    "RachelN512SealedTestError",
    "main",
    "run_sealed_synthetic_test",
]

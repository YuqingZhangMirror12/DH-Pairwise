"""Validation-only convergence extension for the formal Rachel N=512 run.

This entry point resumes both Rachel arms from their hash-bound epoch-5
model *and* AdamW optimizer states.  It deliberately constructs only the
``train`` and ``val`` datasets; synthetic ``test`` and real Dunhuang data are
outside this module's authority.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from pathlib import PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import (
    RachelPairDataset,
)
from staging.pairwise_v0_2.training.evaluation import fit_pairwise_threshold
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from staging.pairwise_v0_2.training.rachel_n512_runner import (
    ARMS,
    SCHEMA_VERSION as TRAIN_SCHEMA_VERSION,
    _arm_selection_metrics,
    _atomic_json,
    _atomic_torch_save,
    _canonical_bytes,
    _loader,
    _new_model,
    _parameter_count,
    _read_manifest_rows,
    _set_determinism,
    _sha256_file,
    _train_epoch,
    _validation_report,
    epoch_indices,
    select_winner_epoch,
    stratified_subset_indices,
)


CONVERGENCE_SCHEMA_VERSION = "rachel-n512-convergence-extension/1.0"
SOURCE_LAST_EPOCH = 5


class RachelN512ConvergenceError(RuntimeError):
    """The continuation source or validation-only convergence contract failed."""


@dataclass(frozen=True)
class RachelN512ConvergenceConfig:
    source_run: Path
    output_root: Path
    arms: Tuple[str, ...] = ARMS
    max_total_epochs: int = 128
    min_total_epochs: int = 20
    patience: int = 12
    min_relative_primary_improvement: float = 0.005
    eta_min: float = 1e-6
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_run", Path(self.source_run).expanduser())
        object.__setattr__(self, "output_root", Path(self.output_root).expanduser())
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be a non-empty unique sequence")
        if any(arm not in ARMS for arm in self.arms):
            raise ValueError("unsupported Rachel convergence arm")
        for name in ("max_total_epochs", "min_total_epochs", "patience"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        if self.max_total_epochs <= SOURCE_LAST_EPOCH:
            raise ValueError("max_total_epochs must extend beyond epoch 5")
        if not SOURCE_LAST_EPOCH < self.min_total_epochs <= self.max_total_epochs:
            raise ValueError("min_total_epochs must be in (5, max_total_epochs]")
        relative = float(self.min_relative_primary_improvement)
        if not math.isfinite(relative) or relative < 0.0:
            raise ValueError("min_relative_primary_improvement must be non-negative")
        if not math.isfinite(float(self.eta_min)) or float(self.eta_min) <= 0.0:
            raise ValueError("eta_min must be finite and positive")
        if not self.device.startswith("cuda"):
            raise ValueError("formal convergence training requires a CUDA device")

    @property
    def remaining_epochs(self) -> int:
        return self.max_total_epochs - SOURCE_LAST_EPOCH

    def portable_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["source_run"] = str(self.source_run.resolve())
        value["output_root"] = str(self.output_root.resolve())
        value["arms"] = list(self.arms)
        value["remaining_epochs"] = self.remaining_epochs
        return value


@dataclass(frozen=True)
class ValidationPlateauState:
    anchor_primary: float
    anchor_epoch: int
    epochs_without_qualifying_improvement: int = 0


@dataclass(frozen=True)
class SourceArm:
    arm: str
    result: Mapping[str, object]
    epoch_rows: Tuple[Mapping[str, object], ...]
    checkpoint_paths: Tuple[Path, ...]
    epoch5_payload: Mapping[str, object]


@dataclass(frozen=True)
class SourceRun:
    path: Path
    receipt: Mapping[str, object]
    receipt_sha256: str
    run_config: Mapping[str, object]
    arms: Tuple[SourceArm, ...]


def _read_json_object(path: Path, name: str) -> Dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RachelN512ConvergenceError(name + " must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RachelN512ConvergenceError("cannot read " + name) from error
    if not isinstance(value, dict):
        raise RachelN512ConvergenceError(name + " must be a JSON object")
    return value


def _require_false(value: Mapping[str, object], key: str, location: str) -> None:
    if value.get(key) is not False:
        raise RachelN512ConvergenceError(location + "." + key + " must be false")


def _safe_relative_checkpoint(run: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RachelN512ConvergenceError("source checkpoint path is invalid")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise RachelN512ConvergenceError("source checkpoint path is unsafe")
    current = run
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise RachelN512ConvergenceError("symlinked source checkpoint is forbidden")
    try:
        path = run.joinpath(*logical.parts).resolve(strict=True)
        path.relative_to(run)
    except (OSError, RuntimeError, ValueError) as error:
        raise RachelN512ConvergenceError(
            "source checkpoint is missing or escapes source run"
        ) from error
    if not path.is_file() or path.suffix != ".pt":
        raise RachelN512ConvergenceError("source checkpoint must be a .pt file")
    return path


def _torch_load_hash_bound(path: Path, expected_sha256: str) -> Mapping[str, object]:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise RachelN512ConvergenceError("source checkpoint SHA-256 is invalid")
    if _sha256_file(path) != expected_sha256:
        raise RachelN512ConvergenceError("source checkpoint SHA-256 mismatch")
    major_text = torch.__version__.split("+", 1)[0].split(".", 1)[0]
    try:
        major = int(major_text)
    except ValueError:
        major = 0
    try:
        # Rachel formal checkpoints are private, receipt-hash-bound artifacts.
        # Torch 1.13 cannot restricted-load its own primitive checkpoint dict;
        # Torch >=2 must use the restricted weights-only loader.
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
        raise RachelN512ConvergenceError(
            "cannot load hash-bound source checkpoint"
        ) from error
    if not isinstance(value, Mapping):
        raise RachelN512ConvergenceError("source checkpoint must be a mapping")
    return value


def _validated_source_run(config: RachelN512ConvergenceConfig) -> SourceRun:
    """Validate the finalized five-epoch source without opening any dataset split."""

    try:
        run = config.source_run.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RachelN512ConvergenceError("source run is unavailable") from error
    if not run.is_dir() or run.name.startswith(".partial-"):
        raise RachelN512ConvergenceError("source run must be finalized")
    receipt_path = run / "run_receipt.json"
    receipt = _read_json_object(receipt_path, "source run receipt")
    if receipt.get("schema_version") != TRAIN_SCHEMA_VERSION:
        raise RachelN512ConvergenceError("unsupported source training receipt schema")
    if receipt.get("status") != "complete_train_validation_only":
        raise RachelN512ConvergenceError("source training run is not complete")
    _require_false(receipt, "test_accessed", "source receipt")
    _require_false(receipt, "real_external_test_accessed", "source receipt")
    run_config = receipt.get("config")
    if not isinstance(run_config, Mapping):
        raise RachelN512ConvergenceError("source run config is missing")
    if run_config.get("epochs") != SOURCE_LAST_EPOCH:
        raise RachelN512ConvergenceError("source run must end at epoch 5")
    source_arms = run_config.get("arms")
    results = receipt.get("arm_results")
    if (
        not isinstance(source_arms, list)
        or not isinstance(results, list)
        or len(source_arms) != len(results)
        or len(set(source_arms)) != len(source_arms)
    ):
        raise RachelN512ConvergenceError("source arms are malformed")
    result_by_arm = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise RachelN512ConvergenceError("source arm result is malformed")
        arm = result.get("arm")
        if arm in result_by_arm:
            raise RachelN512ConvergenceError("source arm result is duplicated")
        result_by_arm[arm] = result
    if set(result_by_arm) != set(source_arms):
        raise RachelN512ConvergenceError("source receipt lacks an arm result")
    if any(arm not in result_by_arm for arm in config.arms):
        raise RachelN512ConvergenceError("requested continuation arm is absent")

    validated_arms = []
    for arm in config.arms:
        result = result_by_arm[arm]
        _require_false(result, "test_accessed", str(arm) + " source result")
        _require_false(
            result,
            "real_external_test_accessed",
            str(arm) + " source result",
        )
        rows = result.get("epochs")
        if (
            not isinstance(rows, list)
            or [row.get("epoch") for row in rows if isinstance(row, Mapping)]
            != list(range(1, SOURCE_LAST_EPOCH + 1))
            or len(rows) != SOURCE_LAST_EPOCH
        ):
            raise RachelN512ConvergenceError(
                "source epoch history must be exactly 1..5"
            )
        paths: List[Path] = []
        epoch5_payload = None
        for epoch, row in enumerate(rows, 1):
            assert isinstance(row, Mapping)
            path = _safe_relative_checkpoint(run, row.get("checkpoint"))
            expected_sha = row.get("checkpoint_sha256")
            payload = _torch_load_hash_bound(path, expected_sha)
            if (
                payload.get("schema_version") != "rachel-n512-checkpoint/1.0"
                or payload.get("arm") != arm
                or payload.get("epoch") != epoch
                or payload.get("run_config") != run_config
            ):
                raise RachelN512ConvergenceError(
                    "source checkpoint provenance disagrees with receipt"
                )
            _require_false(payload, "test_accessed", "source checkpoint")
            _require_false(payload, "real_external_test_accessed", "source checkpoint")
            if not isinstance(payload.get("model_state_dict"), Mapping):
                raise RachelN512ConvergenceError("source model state is missing")
            if not isinstance(payload.get("optimizer_state_dict"), Mapping):
                raise RachelN512ConvergenceError("source optimizer state is missing")
            paths.append(path)
            if epoch == SOURCE_LAST_EPOCH:
                epoch5_payload = payload
        assert epoch5_payload is not None
        validated_arms.append(
            SourceArm(
                arm=str(arm),
                result=result,
                epoch_rows=tuple(rows),
                checkpoint_paths=tuple(paths),
                epoch5_payload=epoch5_payload,
            )
        )
    return SourceRun(
        path=run,
        receipt=receipt,
        receipt_sha256=_sha256_file(receipt_path),
        run_config=run_config,
        arms=tuple(validated_arms),
    )


def _strict_restore_epoch5(
    source: SourceArm,
    source_run_config: Mapping[str, object],
    device: torch.device,
) -> Tuple[nn.Module, torch.optim.AdamW, RachelN512Config, RachelN512LossConfig]:
    """Restore model and exact AdamW continuation state from epoch 5."""

    payload = source.epoch5_payload
    model_value = payload.get("model_config")
    loss_value = payload.get("loss_config")
    if not isinstance(model_value, Mapping) or not isinstance(loss_value, Mapping):
        raise RachelN512ConvergenceError("source model/loss config is missing")
    try:
        model_config = RachelN512Config(**dict(model_value))
        loss_config = RachelN512LossConfig(**dict(loss_value))
    except (TypeError, ValueError) as error:
        raise RachelN512ConvergenceError(
            "source model/loss config is invalid"
        ) from error
    learning_rate = source_run_config.get("learning_rate")
    weight_decay = source_run_config.get("weight_decay")
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
        or not isinstance(weight_decay, (int, float))
        or isinstance(weight_decay, bool)
        or not math.isfinite(float(weight_decay))
        or float(weight_decay) < 0.0
    ):
        raise RachelN512ConvergenceError("source AdamW hyperparameters are invalid")
    model = _new_model(source.arm, model_config, device)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise RachelN512ConvergenceError(
            "epoch-5 model failed strict restore"
        ) from error
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    state = payload["optimizer_state_dict"]
    groups = state.get("param_groups") if isinstance(state, Mapping) else None
    if not isinstance(groups, list) or len(groups) != 1:
        raise RachelN512ConvergenceError("epoch-5 AdamW state must have one group")
    group = groups[0]
    if (
        not isinstance(group, Mapping)
        or float(group.get("lr", math.nan)) != float(learning_rate)
        or float(group.get("weight_decay", math.nan)) != float(weight_decay)
        or tuple(group.get("betas", ())) != (0.9, 0.999)
        or float(group.get("eps", math.nan)) != 1e-8
        or group.get("amsgrad") is not False
    ):
        raise RachelN512ConvergenceError("epoch-5 AdamW hyperparameters changed")
    if len(group.get("params", ())) != sum(1 for _ in model.parameters()):
        raise RachelN512ConvergenceError("epoch-5 AdamW parameter binding changed")
    try:
        optimizer.load_state_dict(state)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise RachelN512ConvergenceError(
            "epoch-5 AdamW failed strict restore"
        ) from error
    restored = optimizer.param_groups[0]
    if (
        float(restored["lr"]) != float(learning_rate)
        or float(restored["weight_decay"]) != float(weight_decay)
        or tuple(restored["betas"]) != (0.9, 0.999)
        or float(restored["eps"]) != 1e-8
        or restored["amsgrad"] is not False
    ):
        raise RachelN512ConvergenceError("restored optimizer is not exact AdamW")
    return model, optimizer, model_config, loss_config


def update_validation_plateau(
    state: ValidationPlateauState,
    *,
    epoch: int,
    primary_score: float,
    full_coverage: bool,
    relative_improvement: float,
) -> Tuple[ValidationPlateauState, bool]:
    """Advance the early-stop counter using validation metrics only."""

    score = float(primary_score)
    if epoch <= state.anchor_epoch or not math.isfinite(score):
        raise ValueError("invalid validation plateau update")
    if not math.isfinite(relative_improvement) or relative_improvement < 0.0:
        raise ValueError("relative_improvement must be finite and non-negative")
    threshold = state.anchor_primary * (1.0 + relative_improvement)
    qualifying = bool(full_coverage) and (
        score > 0.0 if state.anchor_primary == 0.0 else score >= threshold
    )
    if qualifying:
        return ValidationPlateauState(score, epoch, 0), True
    return ValidationPlateauState(
        state.anchor_primary,
        state.anchor_epoch,
        state.epochs_without_qualifying_improvement + 1,
    ), False


def validation_early_stop_due(
    state: ValidationPlateauState,
    *,
    current_epoch: int,
    min_total_epochs: int,
    patience: int,
) -> bool:
    """Return the declared validation-only stop decision."""

    if current_epoch <= SOURCE_LAST_EPOCH:
        raise ValueError("current_epoch must be an extension epoch")
    if min_total_epochs <= SOURCE_LAST_EPOCH or patience <= 0:
        raise ValueError("invalid early-stop policy")
    return (
        current_epoch >= min_total_epochs
        and state.epochs_without_qualifying_improvement >= patience
    )


def build_remaining_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    config: RachelN512ConvergenceConfig,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Cosine schedule spanning exactly epochs 6..max_total_epochs."""

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.remaining_epochs,
        eta_min=config.eta_min,
    )


def _required_source_scalar(
    source: Mapping[str, object],
    name: str,
    expected_type,
):
    value = source.get(name)
    if type(value) is not expected_type:  # noqa: E721
        raise RachelN512ConvergenceError("source run config " + name + " is invalid")
    return value


def _extension_run_config(
    source: SourceRun,
    config: RachelN512ConvergenceConfig,
    *,
    dataset_root: Path,
    output_root: Path,
) -> Dict[str, object]:
    """Build the single config object embedded in receipt and every checkpoint."""

    result = dict(source.run_config)
    result.update(
        {
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "arms": list(config.arms),
            "epochs": config.max_total_epochs,
            "device": config.device,
            "continuation": {
                "schema_version": CONVERGENCE_SCHEMA_VERSION,
                "source_run": str(source.path),
                "source_receipt_sha256": source.receipt_sha256,
                "source_last_epoch": SOURCE_LAST_EPOCH,
                "scheduler": "CosineAnnealingLR",
                "scheduler_t_max": config.remaining_epochs,
                "scheduler_eta_min": config.eta_min,
                "min_total_epochs": config.min_total_epochs,
                "patience": config.patience,
                "min_relative_primary_improvement": (
                    config.min_relative_primary_improvement
                ),
                "early_stop_reads": ["val"],
                "checkpoint_selection_reads": ["val"],
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
        }
    )
    return result


def _repack_source_history(
    source: SourceArm,
    source_run: SourceRun,
    *,
    arm_directory: Path,
    partial_directory: Path,
    extension_run_config: Mapping[str, object],
) -> List[Dict[str, object]]:
    """Copy epochs 1..5 into the new immutable run with normalized run_config."""

    rows = []
    for epoch, (source_row, source_path) in enumerate(
        zip(source.epoch_rows, source.checkpoint_paths), 1
    ):
        payload = (
            source.epoch5_payload
            if epoch == SOURCE_LAST_EPOCH
            else _torch_load_hash_bound(
                source_path, str(source_row["checkpoint_sha256"])
            )
        )
        normalized = dict(payload)
        normalized["run_config"] = dict(extension_run_config)
        normalized["continuation_provenance"] = {
            "schema_version": CONVERGENCE_SCHEMA_VERSION,
            "origin": "hash_bound_source_epoch",
            "source_receipt_sha256": source_run.receipt_sha256,
            "source_checkpoint_sha256": source_row["checkpoint_sha256"],
            "source_epoch": epoch,
            "model_and_optimizer_state_unchanged": True,
        }
        normalized["test_accessed"] = False
        normalized["real_external_test_accessed"] = False
        output = arm_directory / ("epoch-{:03d}.pt".format(epoch))
        _atomic_torch_save(output, normalized)
        repacked = dict(source_row)
        repacked.update(
            {
                "checkpoint": str(output.relative_to(partial_directory)),
                "checkpoint_sha256": _sha256_file(output),
                "origin": "source_run_epoch_1_to_5",
                "source_checkpoint": str(source_path),
                "source_checkpoint_sha256": source_row["checkpoint_sha256"],
                "source_epoch_record_sha256": hashlib.sha256(
                    _canonical_bytes(source_row)
                ).hexdigest(),
                "validation_score_artifact_available": False,
            }
        )
        rows.append(repacked)
    return rows


def _initial_plateau(rows: Sequence[Mapping[str, object]]) -> ValidationPlateauState:
    candidates = []
    for row in rows:
        metrics = row.get("selection_metrics")
        epoch = row.get("epoch")
        if not isinstance(metrics, Mapping) or type(epoch) is not int:  # noqa: E721
            raise RachelN512ConvergenceError("source selection metrics are malformed")
        primary = float(metrics.get("primary_score", math.nan))
        coverage = float(metrics.get("coverage", math.nan))
        if not math.isfinite(primary) or not math.isfinite(coverage):
            raise RachelN512ConvergenceError("source selection metrics are non-finite")
        if coverage >= 1.0 - 1e-12:
            candidates.append((primary, -epoch, epoch))
    if not candidates:
        raise RachelN512ConvergenceError(
            "source lacks a full-coverage validation epoch"
        )
    primary, _, epoch = max(candidates)
    return ValidationPlateauState(anchor_primary=primary, anchor_epoch=epoch)


def _load_winner_for_validation(
    *,
    arm: str,
    row: Mapping[str, object],
    partial_directory: Path,
    extension_run_config: Mapping[str, object],
    device: torch.device,
) -> Tuple[nn.Module, RachelN512Config]:
    checkpoint_value = row.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise RachelN512ConvergenceError("winner checkpoint path is missing")
    checkpoint_path = partial_directory / checkpoint_value
    payload = _torch_load_hash_bound(checkpoint_path, str(row.get("checkpoint_sha256")))
    if (
        payload.get("schema_version") != "rachel-n512-checkpoint/1.0"
        or payload.get("arm") != arm
        or payload.get("epoch") != row.get("epoch")
        or payload.get("run_config") != extension_run_config
    ):
        raise RachelN512ConvergenceError("winner checkpoint provenance changed")
    model_value = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(model_value, Mapping) or not isinstance(state, Mapping):
        raise RachelN512ConvergenceError("winner model payload is incomplete")
    try:
        model_config = RachelN512Config(**dict(model_value))
    except (TypeError, ValueError) as error:
        raise RachelN512ConvergenceError("winner model config is invalid") from error
    model = _new_model(arm, model_config, device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RachelN512ConvergenceError(
            "winner model failed strict restore"
        ) from error
    return model, model_config


def run_convergence_extension(config: RachelN512ConvergenceConfig) -> Path:
    """Resume both arms from epoch 5 and stop only on validation or hard cap."""

    if not torch.cuda.is_available():
        raise RachelN512ConvergenceError("formal convergence training requires CUDA")
    source = _validated_source_run(config)
    device = torch.device(config.device)
    seed = _required_source_scalar(source.run_config, "seed", int)
    batch_size = _required_source_scalar(source.run_config, "batch_size", int)
    num_workers = _required_source_scalar(source.run_config, "num_workers", int)
    precision = _required_source_scalar(source.run_config, "precision", str)
    log_every_steps = _required_source_scalar(source.run_config, "log_every_steps", int)
    gradient_clip_norm = float(source.run_config.get("gradient_clip_norm", math.nan))
    max_train_pairs = source.run_config.get("max_train_pairs")
    max_val_pairs = source.run_config.get("max_val_pairs")
    if precision not in {"fp32", "bf16"}:
        raise RachelN512ConvergenceError("source precision is invalid")
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0.0:
        raise RachelN512ConvergenceError("source gradient clip is invalid")
    if max_train_pairs is not None and type(max_train_pairs) is not int:  # noqa: E721
        raise RachelN512ConvergenceError("source max_train_pairs is invalid")
    if max_val_pairs is not None and type(max_val_pairs) is not int:  # noqa: E721
        raise RachelN512ConvergenceError("source max_val_pairs is invalid")
    source_lr = float(source.run_config.get("learning_rate", math.nan))
    if not math.isfinite(source_lr) or config.eta_min >= source_lr:
        raise RachelN512ConvergenceError("eta_min must be below source learning rate")

    try:
        dataset_root = Path(str(source.run_config["dataset_root"])).resolve(strict=True)
    except (KeyError, OSError, RuntimeError) as error:
        raise RachelN512ConvergenceError(
            "source dataset root is unavailable"
        ) from error
    if not dataset_root.is_dir():
        raise RachelN512ConvergenceError("source dataset root is not a directory")
    train_manifest_path = dataset_root / "pairs" / "train.jsonl"
    val_manifest_path = dataset_root / "pairs" / "val.jsonl"
    for path in (train_manifest_path, val_manifest_path):
        if path.is_symlink() or not path.is_file():
            raise RachelN512ConvergenceError(
                "train/val manifest must be a regular file"
            )

    output_root = config.output_root.resolve()
    try:
        output_root.relative_to(source.path)
    except ValueError:
        pass
    else:
        raise RachelN512ConvergenceError("extension output must not mutate source run")
    output_root.mkdir(parents=True, exist_ok=True)
    extension_run_config = _extension_run_config(
        source,
        config,
        dataset_root=dataset_root,
        output_root=output_root,
    )
    fingerprint_payload = {
        "schema_version": CONVERGENCE_SCHEMA_VERSION,
        "source_receipt_sha256": source.receipt_sha256,
        "config": config.portable_dict(),
        "extension_run_config": extension_run_config,
        "train_manifest_sha256": _sha256_file(train_manifest_path),
        "val_manifest_sha256": _sha256_file(val_manifest_path),
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    fingerprint = hashlib.sha256(_canonical_bytes(fingerprint_payload)).hexdigest()
    final_directory = output_root / ("run-convergence-" + fingerprint[:16])
    partial_directory = output_root / (".partial-convergence-" + fingerprint[:16])
    if final_directory.exists() or partial_directory.exists():
        raise RachelN512ConvergenceError(
            "extension output already exists; preserve it rather than overwriting"
        )
    partial_directory.mkdir()
    _atomic_json(
        partial_directory / "run_config.json",
        {
            "schema_version": CONVERGENCE_SCHEMA_VERSION,
            "status": "running_train_validation_only",
            "fingerprint_sha256": fingerprint,
            "config": extension_run_config,
            "policy": config.portable_dict(),
            "test_accessed": False,
            "real_external_test_accessed": False,
        },
    )

    try:
        _set_determinism(seed)
        train_dataset = RachelPairDataset(dataset_root, "train")
        val_dataset = RachelPairDataset(dataset_root, "val")
        train_manifest = _read_manifest_rows(dataset_root, "train")
        val_manifest = _read_manifest_rows(dataset_root, "val")
        if len(train_dataset) != len(train_manifest) or len(val_dataset) != len(
            val_manifest
        ):
            raise RachelN512ConvergenceError("loader/manifest population mismatch")
        train_subset = stratified_subset_indices(
            [row.label for row in train_manifest], seed + 31_337, max_train_pairs
        )
        val_order = stratified_subset_indices(
            [row.label for row in val_manifest], seed + 91_733, max_val_pairs
        )
        val_rows = tuple(val_manifest[index] for index in val_order)
        val_by_id = {row.pair_id: row for row in val_rows}
        if len(val_by_id) != len(val_rows):
            raise RachelN512ConvergenceError("validation pair IDs are duplicated")
        val_loader = _loader(
            val_dataset,
            val_order,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed + 77,
        )

        arm_results = []
        for source_arm in source.arms:
            arm = source_arm.arm
            _set_determinism(seed)
            model, optimizer, model_config, loss_config = _strict_restore_epoch5(
                source_arm, source.run_config, device
            )
            scheduler = build_remaining_cosine_scheduler(optimizer, config)
            arm_directory = partial_directory / arm
            arm_directory.mkdir()
            epoch_rows = _repack_source_history(
                source_arm,
                source,
                arm_directory=arm_directory,
                partial_directory=partial_directory,
                extension_run_config=extension_run_config,
            )
            plateau = _initial_plateau(epoch_rows)
            stop_reason = "hard_cap_reached"

            for epoch in range(SOURCE_LAST_EPOCH + 1, config.max_total_epochs + 1):
                positions = epoch_indices(len(train_subset), seed, epoch, None)
                train_order = tuple(train_subset[position] for position in positions)
                train_loader = _loader(
                    train_dataset,
                    train_order,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    seed=seed + epoch,
                )
                learning_rate_used = float(optimizer.param_groups[0]["lr"])
                train_report = _train_epoch(
                    arm=arm,
                    model=model,
                    optimizer=optimizer,
                    loader=train_loader,
                    device=device,
                    precision=precision,
                    loss_config=loss_config,
                    gradient_clip_norm=gradient_clip_norm,
                    log_every_steps=log_every_steps,
                    epoch=epoch,
                )
                validation_report, scores = _validation_report(
                    arm=arm,
                    model=model,
                    loader=val_loader,
                    manifest_by_id=val_by_id,
                    device=device,
                    precision=precision,
                )
                selection_metrics = _arm_selection_metrics(arm, validation_report)
                score_path = arm_directory / ("val-scores-{:03d}.json".format(epoch))
                _atomic_json(score_path, scores)
                score_sha = _sha256_file(score_path)
                scheduler.step()
                learning_rate_next = float(optimizer.param_groups[0]["lr"])
                checkpoint_path = arm_directory / ("epoch-{:03d}.pt".format(epoch))
                checkpoint = {
                    "schema_version": "rachel-n512-checkpoint/1.0",
                    "arm": arm,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "model_config": asdict(model_config),
                    "loss_config": asdict(loss_config),
                    "run_config": extension_run_config,
                    "train_report": train_report,
                    "validation_report": validation_report,
                    "selection_metrics": selection_metrics,
                    "learning_rate_used": learning_rate_used,
                    "learning_rate_next": learning_rate_next,
                    "validation_score_sha256": score_sha,
                    "test_accessed": False,
                    "real_external_test_accessed": False,
                }
                _atomic_torch_save(checkpoint_path, checkpoint)
                checkpoint_sha = _sha256_file(checkpoint_path)
                epoch_row = {
                    "epoch": epoch,
                    "checkpoint": str(checkpoint_path.relative_to(partial_directory)),
                    "checkpoint_sha256": checkpoint_sha,
                    "validation_scores": str(score_path.relative_to(partial_directory)),
                    "validation_scores_sha256": score_sha,
                    "train": train_report,
                    "validation": validation_report,
                    "selection_metrics": selection_metrics,
                    "learning_rate_used": learning_rate_used,
                    "learning_rate_next": learning_rate_next,
                    "origin": "convergence_extension",
                }
                epoch_rows.append(epoch_row)
                plateau, qualifying = update_validation_plateau(
                    plateau,
                    epoch=epoch,
                    primary_score=float(selection_metrics["primary_score"]),
                    full_coverage=float(selection_metrics["coverage"]) >= 1.0 - 1e-12,
                    relative_improvement=config.min_relative_primary_improvement,
                )
                _atomic_json(arm_directory / "epochs.json", epoch_rows)
                print(
                    json.dumps(
                        {
                            "event": "convergence_epoch_complete",
                            "arm": arm,
                            "epoch": epoch,
                            "learning_rate_used": learning_rate_used,
                            "learning_rate_next": learning_rate_next,
                            "qualifying_primary_improvement": qualifying,
                            "epochs_without_qualifying_improvement": (
                                plateau.epochs_without_qualifying_improvement
                            ),
                            **selection_metrics,
                            "train_loss": train_report["mean_loss"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if validation_early_stop_due(
                    plateau,
                    current_epoch=epoch,
                    min_total_epochs=config.min_total_epochs,
                    patience=config.patience,
                ):
                    stop_reason = "validation_early_stop"
                    break

            winner_epoch = select_winner_epoch(epoch_rows)
            winner_row = next(row for row in epoch_rows if row["epoch"] == winner_epoch)
            winner_model, winner_model_config = _load_winner_for_validation(
                arm=arm,
                row=winner_row,
                partial_directory=partial_directory,
                extension_run_config=extension_run_config,
                device=device,
            )
            replay_report, winner_scores = _validation_report(
                arm=arm,
                model=winner_model,
                loader=val_loader,
                manifest_by_id=val_by_id,
                device=device,
                precision=precision,
            )
            replay_metrics = _arm_selection_metrics(arm, replay_report)
            if _canonical_bytes(replay_metrics) != _canonical_bytes(
                winner_row["selection_metrics"]
            ):
                raise RachelN512ConvergenceError(
                    "winner validation replay changed selection metrics"
                )
            winner_score_path = arm_directory / "winner-validation-scores.json"
            _atomic_json(winner_score_path, winner_scores)
            winner_score_sha = _sha256_file(winner_score_path)
            model_config_sha = hashlib.sha256(
                _canonical_bytes(asdict(winner_model_config))
            ).hexdigest()
            threshold = fit_pairwise_threshold(
                winner_scores["probability"],
                winner_scores["labels"],
                winner_scores["valid"],
                winner_scores["clusters"],
                source_split="val",
                validation_fingerprint_sha256=hashlib.sha256(
                    _canonical_bytes(winner_scores["pair_ids"])
                ).hexdigest(),
                checkpoint_sha256=winner_row["checkpoint_sha256"],
                model_config_sha256=model_config_sha,
                aggregation_config_sha256=hashlib.sha256(
                    _canonical_bytes(
                        {"pair_score": "coarse" if arm == "coarse_only" else "fused"}
                    )
                ).hexdigest(),
            )
            arm_result = {
                "arm": arm,
                "parameter_count": _parameter_count(winner_model),
                "epochs": epoch_rows,
                "winner_epoch": winner_epoch,
                "winner_checkpoint": winner_row["checkpoint"],
                "winner_checkpoint_sha256": winner_row["checkpoint_sha256"],
                "winner_selection_metrics": winner_row["selection_metrics"],
                "winner_validation_scores": str(
                    winner_score_path.relative_to(partial_directory)
                ),
                "winner_validation_scores_sha256": winner_score_sha,
                "validation_threshold": threshold.to_dict(),
                "stop_reason": stop_reason,
                "best_epoch": winner_epoch,
                "actual_stop_epoch": int(epoch_rows[-1]["epoch"]),
                "extension_epochs_completed": int(epoch_rows[-1]["epoch"])
                - SOURCE_LAST_EPOCH,
                "epochs_without_qualifying_improvement": (
                    plateau.epochs_without_qualifying_improvement
                ),
                "early_stop_anchor_epoch": plateau.anchor_epoch,
                "early_stop_anchor_primary": plateau.anchor_primary,
                "convergence_claim": (
                    "validation_plateau_under_declared_rule"
                    if stop_reason == "validation_early_stop"
                    else "not_established_before_hard_cap"
                ),
                "test_accessed": False,
                "real_external_test_accessed": False,
            }
            _atomic_json(arm_directory / "arm_result.json", arm_result)
            arm_results.append(arm_result)
            del scheduler, optimizer, model, winner_model
            torch.cuda.empty_cache()

        population = {
            "train_total": len(train_dataset),
            "val_total": len(val_dataset),
            "train_rows_per_epoch": len(train_subset),
            "val_rows": len(val_order),
        }
        convergence_receipt = {
            "schema_version": CONVERGENCE_SCHEMA_VERSION,
            "status": "complete_train_validation_only",
            "fingerprint_sha256": fingerprint,
            "source_run": str(source.path),
            "source_receipt_sha256": source.receipt_sha256,
            "policy": config.portable_dict(),
            "config": extension_run_config,
            "population": population,
            "arm_results": arm_results,
            "test_accessed": False,
            "real_external_test_accessed": False,
        }
        _atomic_json(
            partial_directory / "convergence_receipt.json", convergence_receipt
        )
        convergence_sha = _sha256_file(partial_directory / "convergence_receipt.json")
        run_receipt = {
            "schema_version": TRAIN_SCHEMA_VERSION,
            "status": "complete_train_validation_only",
            "fingerprint_sha256": fingerprint,
            "config": extension_run_config,
            "population": population,
            "arm_results": arm_results,
            "convergence": {
                "schema_version": CONVERGENCE_SCHEMA_VERSION,
                "receipt": "convergence_receipt.json",
                "receipt_sha256": convergence_sha,
                "source_receipt_sha256": source.receipt_sha256,
                "source_last_epoch": SOURCE_LAST_EPOCH,
                "max_total_epochs": config.max_total_epochs,
            },
            "test_accessed": False,
            "real_external_test_accessed": False,
        }
        _atomic_json(partial_directory / "run_receipt.json", run_receipt)
        os.replace(partial_directory, final_directory)
        return final_directory
    except BaseException as error:
        _atomic_json(
            partial_directory / "failure.json",
            {
                "schema_version": CONVERGENCE_SCHEMA_VERSION,
                "status": "failed_train_validation_only",
                "error_type": type(error).__name__,
                "error": str(error),
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--max-total-epochs", type=int, default=128)
    parser.add_argument("--min-total-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-relative-primary-improvement", type=float, default=0.005)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    run_directory = run_convergence_extension(
        RachelN512ConvergenceConfig(
            source_run=arguments.source_run,
            output_root=arguments.output_root,
            arms=tuple(arguments.arms),
            max_total_epochs=arguments.max_total_epochs,
            min_total_epochs=arguments.min_total_epochs,
            patience=arguments.patience,
            min_relative_primary_improvement=(
                arguments.min_relative_primary_improvement
            ),
            eta_min=arguments.eta_min,
            device=arguments.device,
        )
    )
    print(str(run_directory), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONVERGENCE_SCHEMA_VERSION",
    "RachelN512ConvergenceConfig",
    "RachelN512ConvergenceError",
    "SOURCE_LAST_EPOCH",
    "ValidationPlateauState",
    "build_remaining_cosine_scheduler",
    "main",
    "run_convergence_extension",
    "update_validation_plateau",
    "validation_early_stop_due",
]

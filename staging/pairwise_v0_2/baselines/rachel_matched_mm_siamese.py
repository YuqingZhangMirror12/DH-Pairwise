#!/usr/bin/env python3
"""Converged historical-MM Siamese control on frozen Rachel train/val rows.

Only model-facing binary-mask paths, labels, pair IDs and lineage IDs are
accepted from ``pairs/train.jsonl`` and ``pairs/val.jsonl``.  The adapter never
opens contour, correspondence, translation, RGB, synthetic-test, or real data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Strict CUDA determinism requires this to be set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v2

from staging.pairwise_v0_2.training.evaluation import (
    evaluate_pairwise,
    fit_pairwise_threshold,
)
from staging.pairwise_v0_2.training.rachel_n512_runner import epoch_indices


SCHEMA_VERSION = "rachel-matched-historical-mm-train/1.0"
CHECKPOINT_SCHEMA_VERSION = "rachel-matched-historical-mm-checkpoint/1.0"
DEFAULT_SEED = 260831
DEFAULT_MAX_TOTAL_EPOCHS = 128
MODEL_RECIPE = "historical-mm-mobilenetv2-focal-adadelta-steplr/1.0"


class RachelMatchedMMError(RuntimeError):
    """The matched historical-MM control violated its sealed contract."""


class HistoricalMMSiamese(nn.Module):
    """Exact historical ordered MobileNetV2 Siamese architecture.

    The definition is local because the general historical inference adapter
    imports the legacy v0.1 data stream, which is deliberately absent from the
    sealed Rachel training release.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v2(weights=None, dropout=0)
        backbone.features[0][0] = nn.Conv2d(
            1,
            32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.fc_in_features = backbone.last_channel
        self.mobilenet = nn.Sequential(*(list(backbone.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, value: Tensor) -> Tensor:
        output = self.mobilenet(value)
        output = F.adaptive_avg_pool2d(output, (1, 1))
        return output.reshape(output.shape[0], -1)

    def forward(self, input_a: Tensor, input_b: Tensor) -> Tensor:
        embedding_a = self.forward_once(input_a)
        embedding_b = self.forward_once(input_b)
        return self.sigmoid(self.fc(torch.cat((embedding_a, embedding_b), dim=1)))


def preprocess_historical_mm_mask(mask: np.ndarray) -> Tensor:
    """Exact historical Grayscale -> PIL bilinear 64x64 -> tensor recipe."""

    value = np.asarray(mask)
    if value.ndim != 2 or value.size == 0 or value.dtype != np.bool_:
        raise RachelMatchedMMError(
            "historical MM preprocessing requires a non-empty 2D bool mask"
        )
    source = Image.fromarray(value.astype(np.uint8) * 255, mode="L")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = source.resize((64, 64), resample=resampling)
    array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


class _ProbabilityFocalLoss(nn.Module):
    """Probability focal formula in the historical checkpoint companion."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        if not 0.0 < alpha <= 1.0 or not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError("invalid historical focal-loss parameters")
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, probability: Tensor, target: Tensor) -> Tensor:
        if probability.shape != target.shape:
            raise ValueError("focal probability and target shapes differ")
        bce = F.binary_cross_entropy(probability, target, reduction="none")
        probability_of_observed_class = torch.exp(-bce)
        return (
            self.alpha * (1.0 - probability_of_observed_class).pow(self.gamma) * bce
        ).mean()


def initialize_historical_mm_training_weights(model: HistoricalMMSiamese) -> None:
    """Exact explicit Linear initialization from the historical trainer."""

    if not isinstance(model, HistoricalMMSiamese):
        raise TypeError("model must be HistoricalMMSiamese")
    for module in tuple(model.mobilenet.modules()) + tuple(model.fc.modules()):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is None:
                raise RachelMatchedMMError("historical Linear unexpectedly lacks bias")
            nn.init.constant_(module.bias, 0.01)


@dataclass(frozen=True)
class RachelMMRunConfig:
    dataset_root: Path
    output_root: Path
    max_total_epochs: int = DEFAULT_MAX_TOTAL_EPOCHS
    min_total_epochs: int = 20
    patience: int = 12
    min_relative_auroc_improvement: float = 0.005
    batch_size: int = 16
    eval_batch_size: int = 128
    seed: int = DEFAULT_SEED
    device: str = "cuda:0"
    learning_rate: float = 1.0
    scheduler_gamma: float = 0.7
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).expanduser())
        object.__setattr__(self, "output_root", Path(self.output_root).expanduser())
        for name in ("max_total_epochs", "min_total_epochs", "patience"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        if not 5 <= self.min_total_epochs <= self.max_total_epochs:
            raise ValueError("min_total_epochs must be in [5, max_total_epochs]")
        if type(self.batch_size) is not int or self.batch_size <= 0:  # noqa: E721
            raise ValueError("batch_size must be a positive integer")
        if type(self.eval_batch_size) is not int or self.eval_batch_size <= 0:  # noqa: E721
            raise ValueError("eval_batch_size must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:  # noqa: E721
            raise ValueError("seed must be a non-negative integer")
        numeric = {
            "min_relative_auroc_improvement": self.min_relative_auroc_improvement,
            "learning_rate": self.learning_rate,
            "scheduler_gamma": self.scheduler_gamma,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
        }
        if any(not math.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("numeric hyperparameters must be finite")
        if self.min_relative_auroc_improvement < 0.0:
            raise ValueError("relative improvement must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.scheduler_gamma < 1.0:
            raise ValueError("scheduler_gamma must be in (0, 1)")
        if not 0.0 < self.focal_alpha <= 1.0 or self.focal_gamma < 0.0:
            raise ValueError("invalid focal parameters")
        if not self.device.startswith("cuda"):
            raise ValueError("formal matched control requires CUDA")

    def portable_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["dataset_root"] = str(self.dataset_root.resolve())
        result["output_root"] = str(self.output_root.resolve())
        result["model_recipe"] = MODEL_RECIPE
        result["source_splits_opened"] = ["train", "val"]
        result["test_accessed"] = False
        result["real_external_test_accessed"] = False
        return result


@dataclass(frozen=True)
class _PairRow:
    pair_id: str
    label: bool
    cluster_id: str
    unit_a: str
    unit_b: str
    mask_a: Path
    mask_b: Path


class _PairDataset(Dataset):
    def __init__(
        self, rows: Sequence[_PairRow], tensors: Mapping[Path, Tensor]
    ) -> None:
        self.rows = tuple(rows)
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        row = self.rows[index]
        return (
            self.tensors[row.mask_a],
            self.tensors[row.mask_b],
            torch.tensor(float(row.label), dtype=torch.float32),
        )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _seed(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RachelMatchedMMError("CUBLAS deterministic workspace changed")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _safe_model_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RachelMatchedMMError("model_mask_path is invalid")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise RachelMatchedMMError("model_mask_path is unsafe")
    if logical.parts[:2] != ("model", "masks_800") or logical.suffix.lower() != ".png":
        raise RachelMatchedMMError("only model/masks_800 PNG inputs are allowed")
    current = root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise RachelMatchedMMError("symlinked model mask is forbidden")
    try:
        resolved = root.joinpath(*logical.parts).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise RachelMatchedMMError(
            "model mask is missing or escapes the release"
        ) from error
    if not resolved.is_file():
        raise RachelMatchedMMError("model mask is not a regular file")
    return resolved


def _cluster_id(first: str, second: str) -> str:
    units = sorted((first, second))
    if units[0] == units[1]:
        return "unit:" + units[0]
    return "unit-pair:" + hashlib.sha256(_canonical_bytes(units)).hexdigest()


def _model_config(config: RachelMMRunConfig) -> Dict[str, object]:
    return {
        "architecture": "HistoricalMMSiamese",
        "recipe": MODEL_RECIPE,
        "input": "bool_mask_PIL_bilinear_64x64_single_channel",
        "ordered_pair": True,
        "focal_formula": "alpha_times_one_minus_p_observed_pow_gamma_times_bce",
        "focal_alpha": config.focal_alpha,
        "focal_gamma": config.focal_gamma,
        "optimizer": {
            "name": "Adadelta",
            "learning_rate": config.learning_rate,
            "rho": 0.9,
            "eps": 1e-6,
            "weight_decay": 0.0,
        },
        "scheduler": {
            "name": "StepLR",
            "step_size": 1,
            "gamma": config.scheduler_gamma,
        },
    }


def _read_rows(root: Path, split: str) -> Tuple[_PairRow, ...]:
    if split not in {"train", "val"}:
        raise RachelMatchedMMError("matched adapter permits train/val only")
    manifest = _safe_manifest(root, split)
    rows: List[_PairRow] = []
    seen = set()
    with manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RachelMatchedMMError("malformed manifest JSON") from error
            if not isinstance(value, Mapping) or value.get("split") != split:
                raise RachelMatchedMMError("manifest split mismatch")
            pair_id = value.get("pair_id")
            label = value.get("label")
            first = value.get("fragment_a")
            second = value.get("fragment_b")
            if (
                not isinstance(pair_id, str)
                or not pair_id
                or pair_id in seen
                or type(label) is not bool  # noqa: E721
                or not isinstance(first, Mapping)
                or not isinstance(second, Mapping)
            ):
                raise RachelMatchedMMError("malformed pair row {}".format(line_number))
            unit_a = first.get("split_unit_id")
            unit_b = second.get("split_unit_id")
            if not all(isinstance(unit, str) and unit for unit in (unit_a, unit_b)):
                raise RachelMatchedMMError("pair row lacks split lineage")
            rows.append(
                _PairRow(
                    pair_id=pair_id,
                    label=label,
                    cluster_id=_cluster_id(unit_a, unit_b),
                    unit_a=unit_a,
                    unit_b=unit_b,
                    mask_a=_safe_model_path(root, first.get("model_mask_path")),
                    mask_b=_safe_model_path(root, second.get("model_mask_path")),
                )
            )
            seen.add(pair_id)
    if not rows or sum(row.label for row in rows) * 2 != len(rows):
        raise RachelMatchedMMError(split + " must be non-empty and exactly 1:1")
    return tuple(rows)


def _safe_manifest(root: Path, split: str) -> Path:
    current = root
    for part in ("pairs", split + ".jsonl"):
        current = current / part
        if current.is_symlink():
            raise RachelMatchedMMError("symlinked manifest component is forbidden")
    try:
        manifest = current.resolve(strict=True)
        manifest.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise RachelMatchedMMError(
            "manifest is missing or escapes the release"
        ) from error
    if not manifest.is_file():
        raise RachelMatchedMMError("manifest must be a regular file")
    return manifest


def _load_mask(path: Path) -> Tensor:
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (800, 800):
                raise RachelMatchedMMError("model mask must be an 800x800 PNG")
            array = np.asarray(image)
    except (OSError, ValueError) as error:
        raise RachelMatchedMMError("cannot decode model mask") from error
    if array.ndim != 2 or array.dtype != np.uint8:
        raise RachelMatchedMMError("model mask must be single-channel uint8")
    unique = np.unique(array)
    if not set(int(value) for value in unique).issubset({0, 255}) or len(unique) < 2:
        raise RachelMatchedMMError("model mask must be non-constant binary 0/255")
    return preprocess_historical_mm_mask(array == 255).contiguous()


def _preload(rows: Sequence[_PairRow]) -> Dict[Path, Tensor]:
    paths = sorted({row.mask_a for row in rows} | {row.mask_b for row in rows})
    return {path: _load_mask(path) for path in paths}


def _initialize_model(seed: int, device: torch.device) -> HistoricalMMSiamese:
    _seed(seed)
    model = HistoricalMMSiamese()
    initialize_historical_mm_training_weights(model)
    return model.to(device)


def _loader(
    dataset: _PairDataset, indices: Sequence[int], *, batch_size: int
) -> DataLoader:
    return DataLoader(
        torch.utils.data.Subset(dataset, tuple(indices)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )


def _predict(
    model: nn.Module,
    dataset: _PairDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> List[float]:
    output: List[float] = []
    model.eval()
    with torch.inference_mode():
        for input_a, input_b, _ in _loader(
            dataset, range(len(dataset)), batch_size=batch_size
        ):
            probability = model(
                input_a.to(device, non_blocking=True),
                input_b.to(device, non_blocking=True),
            )[:, 0]
            if (
                not torch.isfinite(probability).all()
                or ((probability < 0.0) | (probability > 1.0)).any()
            ):
                raise RachelMatchedMMError("historical model returned invalid scores")
            output.extend(float(value) for value in probability.cpu())
    return output


def _validation(
    model: nn.Module,
    dataset: _PairDataset,
    rows: Sequence[_PairRow],
    *,
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, object], List[float]]:
    probability = _predict(model, dataset, device=device, batch_size=batch_size)
    report = evaluate_pairwise(
        probability,
        [row.label for row in rows],
        [True] * len(rows),
        [row.cluster_id for row in rows],
        threshold=0.5,
    )
    return report, probability


def _winner(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not rows:
        raise RachelMatchedMMError("winner selection requires epochs")
    return max(
        rows,
        key=lambda row: (
            float(row["selection_metrics"]["auroc"]),
            float(row["selection_metrics"]["auprc"]),
            -int(row["epoch"]),
        ),
    )


def _advance_plateau(
    *,
    anchor: float,
    anchor_epoch: int,
    count: int,
    epoch: int,
    auroc: float,
    relative_improvement: float,
) -> Tuple[float, int, int, bool]:
    if epoch <= anchor_epoch or not math.isfinite(auroc):
        raise ValueError("invalid plateau update")
    qualifying = anchor == -math.inf or auroc >= anchor * (1.0 + relative_improvement)
    if qualifying:
        return auroc, epoch, 0, True
    return anchor, anchor_epoch, count + 1, False


def _early_stop_due(epoch: int, count: int, config: RachelMMRunConfig) -> bool:
    return epoch >= config.min_total_epochs and count >= config.patience


def _require_disjoint_splits(
    train_rows: Sequence[_PairRow], val_rows: Sequence[_PairRow]
) -> None:
    train_pair_ids = {row.pair_id for row in train_rows}
    val_pair_ids = {row.pair_id for row in val_rows}
    train_units = {unit for row in train_rows for unit in (row.unit_a, row.unit_b)}
    val_units = {unit for row in val_rows for unit in (row.unit_a, row.unit_b)}
    train_masks = {path for row in train_rows for path in (row.mask_a, row.mask_b)}
    val_masks = {path for row in val_rows for path in (row.mask_a, row.mask_b)}
    if (
        train_pair_ids & val_pair_ids
        or train_units & val_units
        or train_masks & val_masks
    ):
        raise RachelMatchedMMError("train/val identities are not disjoint")


def _fit_validation_threshold(
    probability: Sequence[float],
    rows: Sequence[_PairRow],
    *,
    checkpoint_sha256: str,
    model_config: Mapping[str, object],
):
    return fit_pairwise_threshold(
        probability,
        [row.label for row in rows],
        [True] * len(rows),
        [row.cluster_id for row in rows],
        source_split="val",
        validation_fingerprint_sha256=hashlib.sha256(
            _canonical_bytes([row.pair_id for row in rows])
        ).hexdigest(),
        checkpoint_sha256=checkpoint_sha256,
        model_config_sha256=hashlib.sha256(_canonical_bytes(model_config)).hexdigest(),
        aggregation_config_sha256=hashlib.sha256(
            _canonical_bytes({"pair_score": "historical_mm_probability"})
        ).hexdigest(),
    )


def run_training(config: RachelMMRunConfig) -> Path:
    """Train the matched control to a validation-only convergence plateau."""

    if not torch.cuda.is_available():
        raise RachelMatchedMMError("formal matched control requires CUDA")
    root = config.dataset_root.resolve(strict=True)
    output_root = config.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    train_manifest = _safe_manifest(root, "train")
    val_manifest = _safe_manifest(root, "val")
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config.portable_dict(),
        "train_manifest_sha256": _sha256_file(train_manifest),
        "val_manifest_sha256": _sha256_file(val_manifest),
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    fingerprint = hashlib.sha256(_canonical_bytes(fingerprint_payload)).hexdigest()
    final_directory = output_root / ("run-" + fingerprint[:16])
    partial_directory = output_root / (".partial-run-" + fingerprint[:16])
    if final_directory.exists() or partial_directory.exists():
        raise RachelMatchedMMError("matched-control output already exists")
    partial_directory.mkdir()
    run_config = config.portable_dict()
    _atomic_json(
        partial_directory / "run_config.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running_train_validation_only",
            "fingerprint_sha256": fingerprint,
            "config": run_config,
            "test_accessed": False,
            "real_external_test_accessed": False,
        },
    )
    try:
        train_rows = _read_rows(root, "train")
        val_rows = _read_rows(root, "val")
        if len(train_rows) != 24000 or len(val_rows) != 3000:
            raise RachelMatchedMMError("formal population must be 24000/3000")
        _require_disjoint_splits(train_rows, val_rows)
        tensors = _preload(train_rows + val_rows)
        train_dataset = _PairDataset(train_rows, tensors)
        val_dataset = _PairDataset(val_rows, tensors)
        device = torch.device(config.device)
        model = _initialize_model(config.seed, device)
        criterion = _ProbabilityFocalLoss(config.focal_alpha, config.focal_gamma)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=config.scheduler_gamma
        )
        model_config = _model_config(config)
        epoch_rows: List[Dict[str, object]] = []
        anchor = -math.inf
        anchor_epoch = 0
        plateau = 0
        stop_reason = "hard_cap_reached"
        for epoch in range(1, config.max_total_epochs + 1):
            order = epoch_indices(len(train_dataset), config.seed, epoch, None)
            train_loader = _loader(train_dataset, order, batch_size=config.batch_size)
            model.train()
            loss_sum = 0.0
            presented = 0
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            for input_a, input_b, target in train_loader:
                input_a = input_a.to(device, non_blocking=True)
                input_b = input_b.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                probability = model(input_a, input_b)[:, 0]
                loss = criterion(probability, target)
                if not torch.isfinite(loss):
                    raise RachelMatchedMMError("training loss is non-finite")
                loss.backward()
                optimizer.step()
                count = len(target)
                loss_sum += float(loss.detach()) * count
                presented += count
            if presented != len(train_dataset):
                raise RachelMatchedMMError("epoch exposure is not exhaustive")
            validation, probability = _validation(
                model,
                val_dataset,
                val_rows,
                device=device,
                batch_size=config.eval_batch_size,
            )
            metrics = validation["cluster_balanced"]
            selection_metrics = {
                "primary_score": float(metrics["auroc"]),
                "auroc": float(metrics["auroc"]),
                "auprc": float(metrics["auprc"]),
                "coverage": 1.0,
                "weighting": "equal_lineage_or_lineage_pair_cluster",
            }
            score_path = partial_directory / ("val-scores-{:03d}.json".format(epoch))
            score_artifact = {
                "pair_ids": [row.pair_id for row in val_rows],
                "labels": [row.label for row in val_rows],
                "clusters": [row.cluster_id for row in val_rows],
                "probability": probability,
                "valid": [True] * len(val_rows),
            }
            _atomic_json(score_path, score_artifact)
            scheduler.step()
            checkpoint_path = partial_directory / ("epoch-{:03d}.pt".format(epoch))
            checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_config": model_config,
                "run_config": run_config,
                "train": {"mean_loss": loss_sum / presented, "sample_count": presented},
                "validation": validation,
                "selection_metrics": selection_metrics,
                "learning_rate_used": learning_rate_used,
                "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
                "test_accessed": False,
                "real_external_test_accessed": False,
            }
            _atomic_torch(checkpoint_path, checkpoint)
            row = {
                "epoch": epoch,
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "validation_scores": score_path.name,
                "validation_scores_sha256": _sha256_file(score_path),
                "train": checkpoint["train"],
                "validation": validation,
                "selection_metrics": selection_metrics,
                "learning_rate_used": learning_rate_used,
                "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
            }
            epoch_rows.append(row)
            anchor, anchor_epoch, plateau, qualifying = _advance_plateau(
                anchor=anchor,
                anchor_epoch=anchor_epoch,
                count=plateau,
                epoch=epoch,
                auroc=float(selection_metrics["auroc"]),
                relative_improvement=config.min_relative_auroc_improvement,
            )
            _atomic_json(partial_directory / "epochs.json", epoch_rows)
            print(
                json.dumps(
                    {
                        "event": "matched_mm_epoch_complete",
                        "epoch": epoch,
                        "train_loss": checkpoint["train"]["mean_loss"],
                        "qualifying_auroc_improvement": qualifying,
                        "epochs_without_qualifying_improvement": plateau,
                        **selection_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if _early_stop_due(epoch, plateau, config):
                stop_reason = "validation_early_stop"
                break

        winner_row = _winner(epoch_rows)
        winner_checkpoint = partial_directory / str(winner_row["checkpoint"])
        if _sha256_file(winner_checkpoint) != winner_row["checkpoint_sha256"]:
            raise RachelMatchedMMError("winner checkpoint SHA-256 changed")
        try:
            payload = torch.load(
                winner_checkpoint, map_location="cpu", weights_only=True
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RachelMatchedMMError("cannot load winner checkpoint") from error
        if payload.get("run_config") != run_config:
            raise RachelMatchedMMError("winner run config changed")
        winner_model = HistoricalMMSiamese().to(device)
        winner_model.load_state_dict(payload["model_state_dict"], strict=True)
        validation, probability = _validation(
            winner_model,
            val_dataset,
            val_rows,
            device=device,
            batch_size=config.eval_batch_size,
        )
        replay = {
            "primary_score": float(validation["cluster_balanced"]["auroc"]),
            "auroc": float(validation["cluster_balanced"]["auroc"]),
            "auprc": float(validation["cluster_balanced"]["auprc"]),
            "coverage": 1.0,
            "weighting": "equal_lineage_or_lineage_pair_cluster",
        }
        if _canonical_bytes(replay) != _canonical_bytes(
            winner_row["selection_metrics"]
        ):
            raise RachelMatchedMMError("winner validation replay changed")
        threshold = _fit_validation_threshold(
            probability,
            val_rows,
            checkpoint_sha256=str(winner_row["checkpoint_sha256"]),
            model_config=model_config,
        )
        epoch5_row = next(row for row in epoch_rows if row["epoch"] == 5)
        epoch5_checkpoint_path = partial_directory / str(epoch5_row["checkpoint"])
        if _sha256_file(epoch5_checkpoint_path) != epoch5_row["checkpoint_sha256"]:
            raise RachelMatchedMMError("epoch-5 checkpoint SHA-256 changed")
        epoch5_scores_path = partial_directory / str(epoch5_row["validation_scores"])
        if _sha256_file(epoch5_scores_path) != epoch5_row["validation_scores_sha256"]:
            raise RachelMatchedMMError("epoch-5 validation score SHA-256 changed")
        try:
            epoch5_scores = json.loads(epoch5_scores_path.read_text(encoding="utf-8"))
            epoch5_probability = epoch5_scores["probability"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RachelMatchedMMError(
                "cannot read epoch-5 validation scores"
            ) from error
        if (
            epoch5_scores.get("pair_ids") != [row.pair_id for row in val_rows]
            or epoch5_scores.get("labels") != [row.label for row in val_rows]
            or epoch5_scores.get("clusters") != [row.cluster_id for row in val_rows]
            or epoch5_scores.get("valid") != [True] * len(val_rows)
        ):
            raise RachelMatchedMMError("epoch-5 validation score semantics changed")
        epoch5_threshold = _fit_validation_threshold(
            epoch5_probability,
            val_rows,
            checkpoint_sha256=str(epoch5_row["checkpoint_sha256"]),
            model_config=model_config,
        )
        raw_winner = partial_directory / "winner_state_dict.pt"
        _atomic_torch(raw_winner, payload["model_state_dict"])
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_train_validation_only",
            "fingerprint_sha256": fingerprint,
            "config": run_config,
            "model_config": model_config,
            "population": {
                "train_total": len(train_rows),
                "val_total": len(val_rows),
                "train_positive": sum(row.label for row in train_rows),
                "val_positive": sum(row.label for row in val_rows),
            },
            "epochs": epoch_rows,
            "winner_epoch": int(winner_row["epoch"]),
            "winner_checkpoint": winner_row["checkpoint"],
            "winner_checkpoint_sha256": winner_row["checkpoint_sha256"],
            "winner_selection_metrics": winner_row["selection_metrics"],
            "winner_state_dict": raw_winner.name,
            "winner_state_dict_sha256": _sha256_file(raw_winner),
            "validation_threshold": threshold.to_dict(),
            "same_exposure_epoch5": {
                "checkpoint": epoch5_row["checkpoint"],
                "checkpoint_sha256": epoch5_row["checkpoint_sha256"],
                "selection_metrics": epoch5_row["selection_metrics"],
                "validation_scores": epoch5_row["validation_scores"],
                "validation_scores_sha256": epoch5_row["validation_scores_sha256"],
                "validation_threshold": epoch5_threshold.to_dict(),
                "selection_and_threshold_source": "val_only",
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
            "stop_reason": stop_reason,
            "actual_stop_epoch": int(epoch_rows[-1]["epoch"]),
            "early_stop_anchor_epoch": anchor_epoch,
            "early_stop_anchor_auroc": anchor,
            "epochs_without_qualifying_improvement": plateau,
            "convergence_claim": (
                "validation_plateau_under_declared_rule"
                if stop_reason == "validation_early_stop"
                else "not_established_before_hard_cap"
            ),
            "test_accessed": False,
            "real_external_test_accessed": False,
        }
        _atomic_json(partial_directory / "run_receipt.json", receipt)
        os.replace(partial_directory, final_directory)
        return final_directory
    except BaseException as error:
        _atomic_json(
            partial_directory / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_train_validation_only",
                "error_type": type(error).__name__,
                "error": str(error),
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-total-epochs", type=int, default=128)
    parser.add_argument("--min-total-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-relative-auroc-improvement", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    run_directory = run_training(
        RachelMMRunConfig(
            dataset_root=arguments.dataset_root,
            output_root=arguments.output_root,
            max_total_epochs=arguments.max_total_epochs,
            min_total_epochs=arguments.min_total_epochs,
            patience=arguments.patience,
            min_relative_auroc_improvement=arguments.min_relative_auroc_improvement,
            batch_size=arguments.batch_size,
            eval_batch_size=arguments.eval_batch_size,
            seed=arguments.seed,
            device=arguments.device,
        )
    )
    print(str(run_directory), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__: Tuple[str, ...] = (
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_MAX_TOTAL_EPOCHS",
    "DEFAULT_SEED",
    "RachelMMRunConfig",
    "RachelMatchedMMError",
    "SCHEMA_VERSION",
    "main",
    "run_training",
)

#!/usr/bin/env python3
"""Matched-data whole-mask Siamese control for the Route-A local experiment.

This is a deliberately small, independent baseline runner.  It reconstructs
the same frozen Route-A population used by the four local arms, consumes the
same train/select/calibration/report record ordinals from their batch plan,
and trains the historical ordered MobileNetV2 Siamese architecture from a
fixed random initialization.  Model inputs are only two binary masks resized
to 64 x 64; direction, RGB, text, bounding boxes, and rotation are unavailable
to the model.

The validation roles are kept strict: ``validation_select`` alone chooses the
epoch, ``validation_calibration`` alone chooses the threshold, and final
AUROC/AUPRC are computed only on ``validation_report``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from staging.pairwise_v0_2.baselines.historical_mm_siamese import (
    HISTORICAL_MM_INPUT_SIZE,
    HistoricalMMSiamese,
    preprocess_historical_mm_mask,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise
from staging.pairwise_v0_2.training.local_q1_cache_builder import LocalQ1Population
from staging.pairwise_v0_2.training.local_q1_route_a_research import (
    RouteAResearchInputs,
    rebuild_route_a_population,
    research_loader_factory,
)
from staging.pairwise_v0_2.training.short_ablation import (
    record_sequence_fingerprint,
)


MATCHED_SIAMESE_SCHEMA_VERSION = "dunhuang-route-a-matched-mm-siamese/0.1"
MATCHED_ROUTE_A_SIAMESE_ID = "route-a-matched-mm-mobilenetv2-siamese/0.1"
MATCHED_SIAMESE_RECIPE = (
    "historical-mm-mobilenetv2-best-effort-focal-adadelta/0.1"
)
_PHASES = (
    "train",
    "validation_select",
    "validation_calibration",
    "validation_report",
)
_VALIDATION_PHASES = _PHASES[1:]
_PLAN_SCHEMA_VERSION = "dunhuang-local-q1-batch-plan/0.4"


class MatchedSiameseError(RuntimeError):
    """The matched-data Siamese experiment contract was violated."""


def _canonical_json(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(child) for child in item]
        return item

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_content_sha256(value: Mapping[str, Any], name: str) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    observed = _sha256(_canonical_json(unsigned))
    if claimed != observed:
        raise MatchedSiameseError(name + " content hash is inconsistent")
    return observed


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    temporary.replace(target)


def _reference_key(reference: MaskMemberRef) -> Tuple[str, ...]:
    return (
        reference.binding.logical_id,
        reference.binding.sha256,
        reference.archive_member,
        reference.threshold_rule,
    )


@dataclass(frozen=True)
class MatchedPhasePopulation:
    """Exact record tuples selected by the shared frozen batch plan."""

    train: Tuple[TrainingPairRecord, ...]
    validation_select: Tuple[TrainingPairRecord, ...]
    validation_calibration: Tuple[TrainingPairRecord, ...]
    validation_report: Tuple[TrainingPairRecord, ...]
    plan_file_sha256: str
    plan_content_sha256: str

    def __post_init__(self) -> None:
        for phase in _PHASES:
            records = tuple(getattr(self, phase))
            if not records or any(
                not isinstance(record, TrainingPairRecord) for record in records
            ):
                raise MatchedSiameseError(phase + " records are invalid")
            expected_split = "train" if phase == "train" else "val"
            if any(record.split != expected_split for record in records):
                raise MatchedSiameseError(phase + " contains the wrong split")
            if len({record.pair_id for record in records}) != len(records):
                raise MatchedSiameseError(phase + " repeats a pair ID")
            object.__setattr__(self, phase, records)

    def records(self, phase: str) -> Tuple[TrainingPairRecord, ...]:
        if phase not in _PHASES:
            raise ValueError("unknown matched-Siamese phase")
        return tuple(getattr(self, phase))

    def summary(self) -> Dict[str, Any]:
        return {
            "batch_plan": {
                "file_sha256": self.plan_file_sha256,
                "content_sha256": self.plan_content_sha256,
            },
            "phases": {
                phase: {
                    "record_count": len(self.records(phase)),
                    "pair_sequence_sha256": record_sequence_fingerprint(
                        self.records(phase)
                    ),
                    "by_dataset": {
                        dataset: sum(
                            record.dataset_id == dataset
                            for record in self.records(phase)
                        )
                        for dataset in sorted(
                            {record.dataset_id for record in self.records(phase)}
                        )
                    },
                }
                for phase in _PHASES
            },
        }


def _phase_ordinals(
    document: Mapping[str, Any], phase: str
) -> Tuple[int, ...]:
    phases = document.get("phases")
    receipt = phases.get(phase) if isinstance(phases, Mapping) else None
    if not isinstance(receipt, Mapping):
        raise MatchedSiameseError("batch plan lacks phase " + phase)
    batches = receipt.get("batches")
    if not isinstance(batches, list) or not batches:
        raise MatchedSiameseError(phase + " has no frozen batches")
    output = []
    for expected_batch_ordinal, batch in enumerate(batches):
        if not isinstance(batch, Mapping) or batch.get("ordinal") != (
            expected_batch_ordinal
        ):
            raise MatchedSiameseError(phase + " batch ordinals are invalid")
        ordinals = batch.get("record_ordinals")
        if not isinstance(ordinals, list) or not ordinals:
            raise MatchedSiameseError(phase + " batch record ordinals are invalid")
        if any(type(value) is not int or value < 0 for value in ordinals):  # noqa: E721
            raise MatchedSiameseError(phase + " contains an invalid record ordinal")
        output.extend(ordinals)
    if (
        receipt.get("record_count") != len(output)
        or receipt.get("batch_count") != len(batches)
        or len(set(output)) != len(output)
    ):
        raise MatchedSiameseError(phase + " cardinality is inconsistent")
    return tuple(output)


def load_matched_phase_population(
    batch_plan_path: Path,
    population: LocalQ1Population,
) -> MatchedPhasePopulation:
    """Map the shared plan ordinals onto the exact reconstructed records."""

    if not isinstance(population, LocalQ1Population):
        raise TypeError("population must be LocalQ1Population")
    path = Path(batch_plan_path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MatchedSiameseError("batch plan is not JSON") from exc
    if not isinstance(document, Mapping):
        raise MatchedSiameseError("batch plan root must be an object")
    if document.get("schema_version") != _PLAN_SCHEMA_VERSION:
        raise MatchedSiameseError("unsupported batch-plan schema")
    content_sha256 = _self_content_sha256(document, "batch plan")
    external_locks = document.get("external_locks")
    if not isinstance(external_locks, Mapping) or (
        external_locks.get("freeze_file_sha256") != population.freeze_file_sha256
        or external_locks.get("freeze_content_sha256")
        != population.freeze_content_sha256
    ):
        raise MatchedSiameseError("batch plan and Route-A population differ")

    ordinals = {phase: _phase_ordinals(document, phase) for phase in _PHASES}
    if set(ordinals["train"]) != set(range(len(population.training_records))):
        raise MatchedSiameseError("train plan is not exhaustive")
    validation_ordinals = [
        ordinal
        for phase in _VALIDATION_PHASES
        for ordinal in ordinals[phase]
    ]
    if sorted(validation_ordinals) != list(range(len(population.validation_records))):
        raise MatchedSiameseError(
            "validation phases are not disjoint and exhaustive"
        )

    phase_records = {
        "train": tuple(population.training_records[index] for index in ordinals["train"]),
        **{
            phase: tuple(
                population.validation_records[index] for index in ordinals[phase]
            )
            for phase in _VALIDATION_PHASES
        },
    }
    validation_components = {
        phase: {record.component_id for record in phase_records[phase]}
        for phase in _VALIDATION_PHASES
    }
    if any(
        validation_components[left].intersection(validation_components[right])
        for index, left in enumerate(_VALIDATION_PHASES)
        for right in _VALIDATION_PHASES[index + 1 :]
    ):
        raise MatchedSiameseError("validation phases share a component")
    return MatchedPhasePopulation(
        **phase_records,
        plan_file_sha256=_sha256(payload),
        plan_content_sha256=content_sha256,
    )


class HistoricalProbabilityFocalLoss(nn.Module):
    """Probability focal formula present in the old checkpoint companion source."""

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
            self.alpha
            * (1.0 - probability_of_observed_class).pow(self.gamma)
            * bce
        ).mean()


def initialize_historical_mm_training_weights(model: HistoricalMMSiamese) -> None:
    """Reproduce the old training source's explicit Linear initialization."""

    if not isinstance(model, HistoricalMMSiamese):
        raise TypeError("model must be HistoricalMMSiamese")
    for module in tuple(model.mobilenet.modules()) + tuple(model.fc.modules()):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is None:
                raise MatchedSiameseError("historical Linear unexpectedly lacks bias")
            nn.init.constant_(module.bias, 0.01)


def _seed_execution(seed: int, device: torch.device) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise MatchedSiameseError("CUDA was requested but is unavailable")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def make_random_historical_mm_model(
    seed: int, *, device: torch.device
) -> HistoricalMMSiamese:
    """Create the historical architecture without pretrained/checkpoint weights."""

    _seed_execution(seed, device)
    model = HistoricalMMSiamese()
    initialize_historical_mm_training_weights(model)
    return model.to(device)


class PreprocessedMaskPairDataset(Dataset):
    """Pair dataset backed only by cached 64 x 64 single-channel mask tensors."""

    def __init__(
        self,
        records: Sequence[TrainingPairRecord],
        tensors: Mapping[Tuple[str, ...], Tensor],
    ) -> None:
        self.records = tuple(records)
        if not self.records:
            raise MatchedSiameseError("pair dataset is empty")
        self._a = tuple(tensors[_reference_key(row.fragment_a)] for row in self.records)
        self._b = tuple(tensors[_reference_key(row.fragment_b)] for row in self.records)
        self._label = torch.tensor(
            [float(row.label) for row in self.records], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self._a[index], self._b[index], self._label[index]


def _close_loader(loader: Any) -> None:
    close = getattr(loader, "close", None)
    if callable(close):
        close()


def preload_mask_tensors(
    records: Sequence[TrainingPairRecord],
    loader_factory: Callable[[], Callable[[MaskMemberRef], np.ndarray]],
    *,
    workers: int,
    existing: Optional[Mapping[Tuple[str, ...], Tensor]] = None,
) -> Dict[Tuple[str, ...], Tensor]:
    """Decode each required mask once and retain only the 64 x 64 mask tensor."""

    if not callable(loader_factory):
        raise TypeError("loader_factory must be callable")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    output = dict(existing or {})
    references: Dict[Tuple[str, ...], MaskMemberRef] = {}
    for record in records:
        if not isinstance(record, TrainingPairRecord):
            raise TypeError("records must contain TrainingPairRecord values")
        for reference in (record.fragment_a, record.fragment_b):
            key = _reference_key(reference)
            if key not in output:
                references.setdefault(key, reference)
    ordered = tuple(sorted(references.items()))
    if not ordered:
        return output

    local = threading.local()
    created_loaders = []
    loader_lock = threading.Lock()

    def load(
        item: Tuple[Tuple[str, ...], MaskMemberRef],
    ) -> Tuple[Tuple[str, ...], Tensor]:
        loader = getattr(local, "loader", None)
        if loader is None:
            loader = loader_factory()
            if not callable(loader):
                raise TypeError("loader_factory must return a callable")
            local.loader = loader
            with loader_lock:
                created_loaders.append(loader)
        key, reference = item
        mask = loader(reference)
        tensor = preprocess_historical_mm_mask(mask)
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != (
            1,
            HISTORICAL_MM_INPUT_SIZE[0],
            HISTORICAL_MM_INPUT_SIZE[1],
        ):
            raise MatchedSiameseError("historical preprocessing shape changed")
        return key, tensor.contiguous()

    try:
        if workers == 1:
            loaded = tuple(load(item) for item in ordered)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                loaded = tuple(executor.map(load, ordered))
    finally:
        for loader in created_loaders:
            _close_loader(loader)
    output.update(loaded)
    return output


def _predict(
    model: nn.Module,
    dataset: PreprocessedMaskPairDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    values = []
    model.eval()
    with torch.inference_mode():
        for input_a, input_b, _target in loader:
            output = model(
                input_a.to(device, non_blocking=device.type == "cuda"),
                input_b.to(device, non_blocking=device.type == "cuda"),
            )
            if not isinstance(output, Tensor) or tuple(output.shape) != (
                len(input_a),
                1,
            ):
                raise MatchedSiameseError("Siamese model output must be [B,1]")
            probability = output[:, 0]
            if not torch.isfinite(probability).all().item() or (
                (probability < 0.0) | (probability > 1.0)
            ).any().item():
                raise MatchedSiameseError("Siamese model returned invalid probability")
            values.append(probability.to(dtype=torch.float64, device="cpu"))
    return torch.cat(values, dim=0)


def evaluate_matched_records(
    records: Sequence[TrainingPairRecord],
    probability: Tensor,
    *,
    threshold: float,
) -> Mapping[str, Any]:
    """Return overall and per-source metrics in the four-arm report shape."""

    rows = tuple(records)
    if probability.dtype != torch.float64 or tuple(probability.shape) != (len(rows),):
        raise TypeError("probability must be float64 [N]")
    labels = [record.label for record in rows]
    clusters = [record.component_id for record in rows]
    valid = [True] * len(rows)
    overall = evaluate_pairwise(
        probability.tolist(), labels, valid, clusters, threshold=threshold
    )
    by_dataset = {}
    for dataset in sorted({record.dataset_id for record in rows}):
        indices = [
            index for index, record in enumerate(rows) if record.dataset_id == dataset
        ]
        by_dataset[dataset] = evaluate_pairwise(
            [float(probability[index]) for index in indices],
            [labels[index] for index in indices],
            [True] * len(indices),
            [clusters[index] for index in indices],
            threshold=threshold,
        )
    macro = {
        metric: float(
            np.mean(
                [
                    float(value["cluster_balanced"][metric])
                    for value in by_dataset.values()
                ]
            )
        )
        for metric in ("auroc", "auprc", "brier", "ece")
    }
    return MappingProxyType(
        {
            "threshold": float(threshold),
            "coverage": {
                "record_count": len(rows),
                "valid_count": len(rows),
                "valid_fraction": 1.0,
                "by_dataset": {
                    dataset: {
                        "record_count": sum(
                            record.dataset_id == dataset for record in rows
                        ),
                        "valid_count": sum(
                            record.dataset_id == dataset for record in rows
                        ),
                    }
                    for dataset in sorted(by_dataset)
                },
            },
            "overall": overall,
            "by_dataset": by_dataset,
            "equal_dataset_macro_cluster": macro,
            "record_sequence_sha256": record_sequence_fingerprint(rows),
            "direction": {
                "accuracy": None,
                "reason": "historical_whole_mask_siamese_has_no_direction_head",
            },
        }
    )


def fit_cluster_balanced_f1_threshold(
    records: Sequence[TrainingPairRecord], probability: Tensor
) -> Mapping[str, float]:
    """Fit the same equal-cluster maximum-F1 rule on calibration only."""

    rows = tuple(records)
    if probability.dtype != torch.float64 or tuple(probability.shape) != (len(rows),):
        raise TypeError("probability must be float64 [N]")
    scores = probability.numpy()
    labels = np.asarray([row.label for row in rows], dtype=np.bool_)
    clusters = np.asarray([row.component_id for row in rows], dtype=object)
    unique, inverse, counts = np.unique(clusters, return_inverse=True, return_counts=True)
    weights = 1.0 / (len(unique) * counts[inverse].astype(np.float64))
    best = None
    for threshold in sorted(set(float(value) for value in scores), reverse=True):
        prediction = scores >= threshold
        tp = float(weights[prediction & labels].sum())
        fp = float(weights[prediction & ~labels].sum())
        fn = float(weights[~prediction & labels].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        key = (f1, precision, recall, threshold)
        if best is None or key > best[0]:
            best = (key, threshold, precision, recall, f1)
    if best is None:
        raise MatchedSiameseError("calibration has no threshold candidate")
    return MappingProxyType(
        {
            "threshold": float(best[1]),
            "cluster_balanced_precision": float(best[2]),
            "cluster_balanced_recall": float(best[3]),
            "cluster_balanced_f1": float(best[4]),
        }
    )


@dataclass(frozen=True)
class MatchedSiameseContract:
    epochs: int = 5
    seed: int = 260829
    train_batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 1.0
    scheduler_gamma: float = 0.7
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    preprocess_workers: int = 8

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "train_batch_size",
            "eval_batch_size",
            "preprocess_workers",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.scheduler_gamma <= 1.0:
            raise ValueError("scheduler_gamma must be in (0,1]")

    def portable_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "seed": self.seed,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "preprocess_workers": self.preprocess_workers,
            "loss": {
                "name": "historical_probability_focal",
                "alpha": self.focal_alpha,
                "gamma": self.focal_gamma,
            },
            "optimizer": {
                "name": "Adadelta",
                "learning_rate": self.learning_rate,
                "rho": 0.9,
                "eps": 1e-6,
                "weight_decay": 0.0,
            },
            "scheduler": {
                "name": "StepLR",
                "step_size": 1,
                "gamma": self.scheduler_gamma,
            },
        }


def _cpu_state_dict(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def _save_raw_state_dict(path: Path, state: Mapping[str, Tensor]) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(dict(state), temporary)
    temporary.replace(target)


def run_matched_siamese_training(
    *,
    phases: MatchedPhasePopulation,
    loader_factory: Callable[[], Callable[[MaskMemberRef], np.ndarray]],
    output_dir: Path,
    contract: MatchedSiameseContract,
    device: torch.device,
    model_factory: Optional[Callable[[], nn.Module]] = None,
) -> Mapping[str, Any]:
    """Train, select, calibrate, and report the matched whole-mask control."""

    if not isinstance(phases, MatchedPhasePopulation):
        raise TypeError("phases must be MatchedPhasePopulation")
    if not isinstance(contract, MatchedSiameseContract):
        raise TypeError("contract must be MatchedSiameseContract")
    output_directory = Path(output_dir)
    if output_directory.exists() or output_directory.is_symlink():
        raise MatchedSiameseError("output directory already exists")
    output_directory.mkdir(parents=True)
    _seed_execution(contract.seed, device)
    if model_factory is None:
        model = make_random_historical_mm_model(contract.seed, device=device)
        model_semantics = (
            "HistoricalMMSiamese_random_init_old_linear_xavier_bias_0.01"
        )
    else:
        model = model_factory().to(device)
        model_semantics = "injected_test_model"
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return torch.nn.Module")

    tensors = preload_mask_tensors(
        phases.train + phases.validation_select,
        loader_factory,
        workers=contract.preprocess_workers,
    )
    train_dataset = PreprocessedMaskPairDataset(phases.train, tensors)
    select_dataset = PreprocessedMaskPairDataset(phases.validation_select, tensors)
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(contract.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=contract.train_batch_size,
        shuffle=True,
        generator=sampler_generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    criterion = HistoricalProbabilityFocalLoss(
        alpha=contract.focal_alpha, gamma=contract.focal_gamma
    )
    optimizer = torch.optim.Adadelta(model.parameters(), lr=contract.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=contract.scheduler_gamma
    )

    epochs = []
    best_key = None
    best_epoch = None
    best_state = None
    for epoch in range(1, contract.epochs + 1):
        model.train()
        loss_sum = 0.0
        presented = 0
        optimizer_steps = 0
        for input_a, input_b, target in train_loader:
            input_a = input_a.to(device, non_blocking=device.type == "cuda")
            input_b = input_b.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            model_output = model(input_a, input_b)
            if tuple(model_output.shape) != (len(target), 1):
                raise MatchedSiameseError("Siamese model output must be [B,1]")
            loss = criterion(model_output[:, 0], target)
            if not torch.isfinite(loss).item():
                raise MatchedSiameseError("training loss is non-finite")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(target)
            presented += len(target)
            optimizer_steps += 1
        if presented != len(train_dataset):
            raise MatchedSiameseError("one epoch did not present every train pair once")
        select_probability = _predict(
            model,
            select_dataset,
            device=device,
            batch_size=contract.eval_batch_size,
        )
        select_metrics = evaluate_matched_records(
            phases.validation_select, select_probability, threshold=0.5
        )
        key = (
            float(select_metrics["equal_dataset_macro_cluster"]["auroc"]),
            float(select_metrics["equal_dataset_macro_cluster"]["auprc"]),
            -epoch,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
        epoch_row = {
            "epoch": epoch,
            "train": {
                "mean_loss": loss_sum / presented,
                "presented_count": presented,
                "optimizer_steps": optimizer_steps,
                "learning_rate_used": float(scheduler.get_last_lr()[0]),
            },
            "validation_select": dict(select_metrics),
        }
        epochs.append(epoch_row)
        progress = {
            "schema_version": MATCHED_SIAMESE_SCHEMA_VERSION,
            "status": "training",
            "completed_epochs": epoch,
            "total_epochs": contract.epochs,
            "current_winner_epoch": best_epoch,
            "epochs": epochs,
        }
        _write_json(output_directory / "progress.json", progress)
        print(_canonical_json(epoch_row).decode("utf-8"), flush=True)
        scheduler.step()

    if best_state is None or best_epoch is None:
        raise MatchedSiameseError("winner selection failed")
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    winner_path = output_directory / "matched_historical_mm_siamese_winner.pt"
    _save_raw_state_dict(winner_path, best_state)

    # Validation roles are opened in order only after the winning epoch is fixed.
    select_replay = _predict(
        model,
        select_dataset,
        device=device,
        batch_size=contract.eval_batch_size,
    )
    select_replay_metrics = evaluate_matched_records(
        phases.validation_select, select_replay, threshold=0.5
    )

    tensors = preload_mask_tensors(
        phases.validation_calibration,
        loader_factory,
        workers=contract.preprocess_workers,
        existing=tensors,
    )
    calibration_dataset = PreprocessedMaskPairDataset(
        phases.validation_calibration, tensors
    )
    calibration_probability = _predict(
        model,
        calibration_dataset,
        device=device,
        batch_size=contract.eval_batch_size,
    )
    calibration_metrics = evaluate_matched_records(
        phases.validation_calibration, calibration_probability, threshold=0.5
    )
    threshold = fit_cluster_balanced_f1_threshold(
        phases.validation_calibration, calibration_probability
    )

    tensors = preload_mask_tensors(
        phases.validation_report,
        loader_factory,
        workers=contract.preprocess_workers,
        existing=tensors,
    )
    report_dataset = PreprocessedMaskPairDataset(phases.validation_report, tensors)
    report_probability = _predict(
        model,
        report_dataset,
        device=device,
        batch_size=contract.eval_batch_size,
    )
    report_metrics = evaluate_matched_records(
        phases.validation_report,
        report_probability,
        threshold=float(threshold["threshold"]),
    )
    score_rows = [
        {
            "pair_id": record.pair_id,
            "dataset_id": record.dataset_id,
            "component_id": record.component_id,
            "label": record.label,
            "probability": float(probability),
        }
        for record, probability in zip(phases.validation_report, report_probability)
    ]
    _write_json(
        output_directory / "validation_report_scores.json", {"records": score_rows}
    )
    checkpoint_sha256 = _sha256(winner_path.read_bytes())
    receipt = {
        "schema_version": MATCHED_SIAMESE_SCHEMA_VERSION,
        "status": "complete_matched_route_a_whole_mask_siamese",
        "recipe": MATCHED_SIAMESE_RECIPE,
        "scope": {
            "input_modality": "binary_mask_only",
            "rgb_used": False,
            "text_or_ocr_used": False,
            "bounding_box_used": False,
            "rotation_search_used": False,
            "direction_supervision_used": False,
            "real_dunhuang_used": False,
        },
        "fairness": {
            "same_route_a_train_record_ids": True,
            "same_batch_plan_validation_partition": True,
            "train_pair_presentations_per_epoch": 1,
            "total_presentations_per_train_pair": contract.epochs,
            "same_train_batch_order_as_local_arms": False,
            "same_optimizer_as_local_arms": False,
            "claim_boundary": (
                "matched_record_ids_exposure_count_and_validation_roles_not_"
                "matched_batch_order_or_optimizer"
            ),
            "winner_reads_only": "validation_select",
            "threshold_reads_only": "validation_calibration",
            "final_metrics_read_only": "validation_report",
            "architectural_difference": (
                "historical control sees whole 64x64 masks; local arms see contour patches"
            ),
            "direction_comparison": "not_applicable_no_direction_head",
        },
        "population": phases.summary(),
        "contract": contract.portable_dict(),
        "model": {
            "semantics": model_semantics,
            "architecture": "ordered_shared_MobileNetV2_then_concat_256_1_sigmoid",
            "pretrained_weights": False,
            "historical_checkpoint_initialization": False,
            "preprocessing": "bool_mask_PIL_bilinear_64x64_single_channel",
            "recipe_fidelity": {
                "architecture_and_preprocessing": "checkpoint_compatible_exact",
                "linear_initialization": "replayed_from_nearby_historical_training_source",
                "focal_formula": (
                    "replayed_from_historical_checkpoint_companion_source_"
                    "constant_alpha_0.25_for_both_labels"
                ),
                "optimizer_scheduler": (
                    "best_effort_from_nearby_historical_mobilenet_training_source"
                ),
                "exact_checkpoint_producing_training_script_identified": False,
            },
        },
        "training": {
            "epochs": epochs,
            "total_optimizer_steps": sum(row["train"]["optimizer_steps"] for row in epochs),
            "total_pair_presentations": len(phases.train) * contract.epochs,
        },
        "winner": {
            "epoch": best_epoch,
            "selection_policy": (
                "max_equal_dataset_macro_cluster_auroc_then_auprc_then_earlier_epoch"
            ),
            "checkpoint": {
                "path": winner_path.name,
                "file_sha256": checkpoint_sha256,
                "format": "tensor_only_raw_state_dict",
            },
            "validation_select_replay": dict(select_replay_metrics),
        },
        "calibration": {
            "fixed_threshold": dict(threshold),
            "metrics_at_0_5": dict(calibration_metrics),
        },
        "validation_report": dict(report_metrics),
        "score_artifact": "validation_report_scores.json",
    }
    receipt["content_sha256"] = _sha256(_canonical_json(receipt))
    _write_json(output_directory / "matched_siamese_receipt.json", receipt)
    return MappingProxyType(receipt)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "route_a_freeze",
        "predecessor_freeze",
        "eligibility_index",
        "eligibility_receipt",
        "route_policy",
        "route_config",
        "mm_archive",
        "eccv_archive",
        "mm_fingerprint_cache",
        "eccv_fingerprint_cache",
        "historical_split",
        "synthetic_manifest",
        "synthetic_archive",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--mm-extracted-root", type=Path)
    parser.add_argument("--eccv-extracted-root", type=Path)
    parser.add_argument("--synthetic-extracted-root", type=Path)
    parser.add_argument("--batch-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=_positive_int, default=5)
    parser.add_argument("--seed", type=int, default=260829)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=256)
    parser.add_argument("--preprocess-workers", type=_positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--scheduler-gamma", type=float, default=0.7)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    extracted = (
        args.mm_extracted_root,
        args.eccv_extracted_root,
        args.synthetic_extracted_root,
    )
    if any(value is not None for value in extracted) and not all(
        value is not None for value in extracted
    ):
        raise MatchedSiameseError(
            "all three extracted roots must be provided together"
        )
    inputs = RouteAResearchInputs(
        route_a_freeze=args.route_a_freeze,
        predecessor_freeze=args.predecessor_freeze,
        eligibility_index=args.eligibility_index,
        eligibility_receipt=args.eligibility_receipt,
        route_policy=args.route_policy,
        route_config=args.route_config,
        mm_archive=args.mm_archive,
        eccv_archive=args.eccv_archive,
        mm_fingerprint_cache=args.mm_fingerprint_cache,
        eccv_fingerprint_cache=args.eccv_fingerprint_cache,
        historical_split=args.historical_split,
        synthetic_manifest=args.synthetic_manifest,
        synthetic_archive=args.synthetic_archive,
        mm_extracted_root=args.mm_extracted_root,
        eccv_extracted_root=args.eccv_extracted_root,
        synthetic_extracted_root=args.synthetic_extracted_root,
    )
    population = rebuild_route_a_population(inputs)
    phases = load_matched_phase_population(args.batch_plan, population)
    receipt = run_matched_siamese_training(
        phases=phases,
        loader_factory=research_loader_factory(inputs),
        output_dir=args.output_dir,
        contract=MatchedSiameseContract(
            epochs=args.epochs,
            seed=args.seed,
            train_batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            scheduler_gamma=args.scheduler_gamma,
            preprocess_workers=args.preprocess_workers,
        ),
        device=torch.device(args.device),
    )
    print(_canonical_json(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MATCHED_ROUTE_A_SIAMESE_ID",
    "MATCHED_SIAMESE_RECIPE",
    "MATCHED_SIAMESE_SCHEMA_VERSION",
    "HistoricalProbabilityFocalLoss",
    "MatchedPhasePopulation",
    "MatchedSiameseContract",
    "MatchedSiameseError",
    "PreprocessedMaskPairDataset",
    "evaluate_matched_records",
    "fit_cluster_balanced_f1_threshold",
    "initialize_historical_mm_training_weights",
    "load_matched_phase_population",
    "main",
    "make_random_historical_mm_model",
    "preload_mask_tensors",
    "run_matched_siamese_training",
]

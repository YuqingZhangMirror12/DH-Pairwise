"""Deterministic train/validation runner for Rachel full-contour Pairwise.

The command intentionally opens only ``train`` and ``val``.  Rachel ``test``
and the real Dunhuang external set have separate, explicit evaluation entry
points so neither can influence checkpoint selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

# Strict deterministic CUDA GEMM must be configured before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

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
    evaluate_pairwise,
    fit_pairwise_threshold,
)
from staging.pairwise_v0_2.training.rachel_n512_loss import (
    RachelN512LossConfig,
    compute_rachel_n512_loss,
)


SCHEMA_VERSION = "rachel-n512-train-run/1.0"
ARMS = ("coarse_only", "full_n512")


class RachelN512RunnerError(RuntimeError):
    """A formal run violated its frozen train/validation contract."""


@dataclass(frozen=True)
class RachelN512RunConfig:
    dataset_root: Path
    output_root: Path
    arms: Tuple[str, ...] = ARMS
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 260831
    precision: str = "fp32"
    device: str = "cuda:0"
    num_workers: int = 4
    max_train_pairs: Optional[int] = None
    max_val_pairs: Optional[int] = None
    log_every_steps: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).expanduser())
        object.__setattr__(self, "output_root", Path(self.output_root).expanduser())
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be a non-empty unique sequence")
        if any(arm not in ARMS for arm in self.arms):
            raise ValueError("unsupported Rachel training arm")
        for name in ("epochs", "batch_size", "log_every_steps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers < 0:  # noqa: E721
            raise ValueError("num_workers must be a non-negative integer")
        if type(self.seed) is not int or self.seed < 0:  # noqa: E721
            raise ValueError("seed must be a non-negative integer")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(name + " must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        for name in ("max_train_pairs", "max_val_pairs"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):  # noqa: E721
                raise ValueError(name + " must be a positive integer or None")
            if value is not None and value % 2:
                raise ValueError(name + " must be even to preserve exact 1:1 labels")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if not self.device.startswith("cuda"):
            raise ValueError("formal Rachel training requires a CUDA device")

    def portable_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["dataset_root"] = str(self.dataset_root.resolve())
        value["output_root"] = str(self.output_root.resolve())
        value["arms"] = list(self.arms)
        return value


@dataclass(frozen=True)
class _ManifestRow:
    pair_id: str
    label: bool
    cluster_id: str


class _IndexView(Dataset):
    def __init__(self, source: RachelPairDataset, indices: Sequence[int]) -> None:
        self.source = source
        self.indices = tuple(int(value) for value in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.source[self.indices[index]]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def epoch_indices(
    length: int, seed: int, epoch: int, limit: Optional[int]
) -> Tuple[int, ...]:
    """Deterministic order shared by all arms; pilot limits are explicit."""

    if length <= 0 or seed < 0 or epoch <= 0:
        raise ValueError("invalid deterministic epoch order arguments")
    generator = np.random.default_rng(seed + 1_000_003 * epoch)
    order = generator.permutation(length)
    if limit is not None:
        order = order[: min(limit, length)]
    return tuple(int(value) for value in order)


def validation_indices(length: int, seed: int, limit: Optional[int]) -> Tuple[int, ...]:
    """One fixed validation subset; never reshuffled between epochs/arms."""

    return epoch_indices(length, seed + 91_733, 1, limit)


def stratified_subset_indices(
    labels: Sequence[bool], seed: int, limit: Optional[int]
) -> Tuple[int, ...]:
    """Freeze an exact 1:1 pilot subset before any epoch shuffling."""

    label_array = np.asarray(labels)
    if label_array.ndim != 1 or label_array.size == 0 or label_array.dtype != np.bool_:
        raise TypeError("labels must be a non-empty explicit bool vector")
    if limit is None:
        if int(label_array.sum()) * 2 != len(label_array):
            raise RachelN512RunnerError(
                "uncapped release split no longer satisfies exact 1:1 labels"
            )
        return tuple(range(len(label_array)))
    if type(limit) is not int or limit <= 0 or limit % 2:  # noqa: E721
        raise ValueError("stratified subset limit must be positive and even")
    per_class = limit // 2
    positive = np.flatnonzero(label_array)
    negative = np.flatnonzero(~label_array)
    if len(positive) < per_class or len(negative) < per_class:
        raise RachelN512RunnerError("requested pilot subset exceeds a label quota")
    generator = np.random.default_rng(seed)
    selected = np.concatenate(
        (
            generator.permutation(positive)[:per_class],
            generator.permutation(negative)[:per_class],
        )
    )
    selected = generator.permutation(selected)
    return tuple(int(value) for value in selected)


def select_winner_epoch(rows: Sequence[Mapping[str, object]]) -> int:
    """Apply the predeclared validation-only arm selection key."""

    if not rows:
        raise ValueError("winner selection requires epoch reports")
    best = None
    full_coverage_epoch_found = False
    for row in rows:
        epoch = row.get("epoch")
        metrics = row.get("selection_metrics")
        if type(epoch) is not int or epoch <= 0 or not isinstance(metrics, Mapping):  # noqa: E721
            raise ValueError("malformed epoch selection report")
        auroc = float(metrics.get("auroc", math.nan))
        auprc = float(metrics.get("auprc", math.nan))
        primary = float(metrics.get("primary_score", auroc))
        coverage = float(metrics.get("coverage", 1.0))
        translation = float(metrics.get("translation_success_at_8px", 0.0))
        correspondence = float(metrics.get("correspondence_f1", 0.0))
        if not all(
            math.isfinite(value)
            for value in (
                primary,
                coverage,
                auroc,
                auprc,
                translation,
                correspondence,
            )
        ):
            raise ValueError("winner metrics must be finite")
        full_coverage = coverage >= 1.0 - 1e-12
        if not full_coverage:
            continue
        full_coverage_epoch_found = True
        key = (
            primary,
            auroc,
            auprc,
            translation,
            correspondence,
            coverage,
            -epoch,
        )
        if best is None or key > best[0]:
            best = (key, epoch)
    if not full_coverage_epoch_found or best is None:
        raise ValueError("winner selection requires a full-coverage validation epoch")
    return best[1]


def _read_manifest_rows(root: Path, split: str) -> Tuple[_ManifestRow, ...]:
    path = root / "pairs" / (split + ".jsonl")
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if value.get("split") != split or type(value.get("label")) is not bool:  # noqa: E721
                raise RachelN512RunnerError(
                    "malformed {} manifest line {}".format(split, line_number)
                )
            pair_id = value.get("pair_id")
            first = value.get("fragment_a", {}).get("split_unit_id")
            second = value.get("fragment_b", {}).get("split_unit_id")
            if not all(
                isinstance(item, str) and item for item in (pair_id, first, second)
            ):
                raise RachelN512RunnerError("manifest lacks pair/lineage identity")
            units = sorted((first, second))
            if units[0] == units[1]:
                cluster_id = "unit:" + units[0]
            else:
                cluster_id = (
                    "unit-pair:" + hashlib.sha256(_canonical_bytes(units)).hexdigest()
                )
            rows.append(_ManifestRow(pair_id, value["label"], cluster_id))
    if not rows:
        raise RachelN512RunnerError(split + " manifest is empty")
    return tuple(rows)


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _loader(
    dataset: RachelPairDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    options = {
        "dataset": _IndexView(dataset, indices),
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


def _full_batch(
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
        _tensor(batch.labels, device, torch.float32),
        _tensor(batch.target_a, device, torch.long),
        _tensor(batch.target_b, device, torch.long),
        _tensor(batch.translation_a_to_b_rc, device, torch.float32),
        _tensor(batch.translation_valid, device, torch.bool),
    )
    return inputs, targets


def _coarse_batch(
    batch: RachelBatch, device: torch.device, coarse_size: int = 128
) -> Tuple[Tensor, Tensor, Tensor]:
    # Use the exact full-mask torch resize used inside RachelN512Pairwise.
    # PIL and torch nearest-neighbour coordinate conventions are not generally
    # byte-identical, so cached coarse masks cannot be mixed between arms.
    full_a = _tensor(batch.mask_a, device, torch.float32)
    full_b = _tensor(batch.mask_b, device, torch.float32)
    return (
        F.interpolate(full_a, size=(coarse_size, coarse_size), mode="nearest"),
        F.interpolate(full_b, size=(coarse_size, coarse_size), mode="nearest"),
        _tensor(batch.labels, device, torch.float32),
    )


def _method_scores(
    probability: Sequence[float],
    label: Sequence[bool],
    valid: Sequence[bool],
    clusters: Sequence[str],
) -> Dict[str, object]:
    return evaluate_pairwise(
        probability,
        label,
        valid,
        clusters,
        threshold=0.5,
    )


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _run_fingerprint(config: RachelN512RunConfig) -> str:
    train_manifest = config.dataset_root.resolve() / "pairs" / "train.jsonl"
    val_manifest = config.dataset_root.resolve() / "pairs" / "val.jsonl"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config.portable_dict(),
        "train_manifest_sha256": _sha256_file(train_manifest),
        "val_manifest_sha256": _sha256_file(val_manifest),
        "test_accessed": False,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _new_model(
    arm: str, model_config: RachelN512Config, device: torch.device
) -> nn.Module:
    # Construct the same full container first so coarse-only and full_n512 get
    # byte-identical shared-coarse initialization under the same seed.  The
    # Rachel container also replaces AdaptiveAvgPool with the deterministic
    # spatial mean required by the target CUDA runtime.
    container = RachelN512Pairwise(model_config)
    model: nn.Module = container.coarse if arm == "coarse_only" else container
    return model.to(device)


def _train_epoch(
    *,
    arm: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: Iterable[RachelBatch],
    device: torch.device,
    precision: str,
    loss_config: RachelN512LossConfig,
    gradient_clip_norm: float,
    log_every_steps: int,
    epoch: int,
) -> Dict[str, object]:
    model.train()
    loss_sum = torch.zeros((), device=device)
    component_sums: Dict[str, Tensor] = {}
    sample_count = 0
    started = time.perf_counter()
    for step, batch in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            if arm == "coarse_only":
                mask_a, mask_b, labels = _coarse_batch(batch, device)
                output = model(mask_a, mask_b)
                values = F.binary_cross_entropy_with_logits(
                    output.logit, labels, reduction="none"
                )
                valid = output.valid_problem
                loss = (values * valid.to(values.dtype)).sum() / valid.sum().clamp_min(
                    1
                )
                components = {"coarse_pair_bce": loss}
            else:
                inputs, targets = _full_batch(batch, device)
                output = model(*inputs)
                loss_output = compute_rachel_n512_loss(
                    output, *targets, config=loss_config
                )
                loss = loss_output.total
                components = {
                    "fused_pair_bce": loss_output.fused_pair_bce,
                    "coarse_pair_bce": loss_output.coarse_pair_bce,
                    "local_pair_bce": loss_output.local_pair_bce,
                    "assignment_nll": loss_output.assignment_nll,
                    "translation_smooth_l1": loss_output.translation_smooth_l1,
                    "sinkhorn_residual": loss_output.sinkhorn_residual,
                }
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        batch_count = len(batch.pair_ids)
        loss_sum = loss_sum + loss.detach() * batch_count
        for name, value in components.items():
            component_sums[name] = (
                component_sums.get(name, torch.zeros((), device=device))
                + value.detach() * batch_count
            )
        sample_count += batch_count
        if step % log_every_steps == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "arm": arm,
                        "epoch": epoch,
                        "step": step,
                        "samples": sample_count,
                        "mean_loss": float((loss_sum / sample_count).cpu().item()),
                        "gradient_norm": float(gradient_norm.detach().cpu().item()),
                        "elapsed_seconds": elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if sample_count <= 0:
        raise RachelN512RunnerError("training epoch had no samples")
    torch.cuda.synchronize(device)
    return {
        "sample_count": sample_count,
        "mean_loss": float((loss_sum / sample_count).cpu().item()),
        "mean_components": {
            name: float((value / sample_count).cpu().item())
            for name, value in sorted(component_sums.items())
        },
        "seconds": time.perf_counter() - started,
    }


def _correspondence_counts(
    assignment: Tensor,
    unmatched_a: Tensor,
    unmatched_b: Tensor,
    target_a: Tensor,
    target_b: Tensor,
    sample_valid: Tensor,
) -> Tuple[int, int, int, int, int, int, int]:
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
    valid_a = (target_a != -2) & sample_valid[:, None]
    predicted_match = reciprocal_real & valid_a
    true_match = (target_a >= 0) & sample_valid[:, None]
    true_positive = predicted_match & (row_index == target_a)
    # Threshold-free ranking diagnostic: a real mutual-top1 candidate is
    # counted even while the dustbin still owns most row mass.  Keep this
    # separate from the stricter dustbin-aware inference metric above.
    mutual_candidate = reciprocal_a & valid_a
    mutual_true_positive = mutual_candidate & (row_index == target_a)

    pred_dustbin_a = unmatched_a >= row_value
    col_best, _ = assignment.max(dim=1)
    pred_dustbin_b = unmatched_b >= col_best
    valid_b = (target_b != -2) & sample_valid[:, None]
    dustbin_correct = ((pred_dustbin_a == (target_a == -1)) & valid_a).sum() + (
        (pred_dustbin_b == (target_b == -1)) & valid_b
    ).sum()
    dustbin_total = valid_a.sum() + valid_b.sum()
    return (
        int(true_positive.sum().item()),
        int(predicted_match.sum().item()),
        int(mutual_true_positive.sum().item()),
        int(mutual_candidate.sum().item()),
        int(true_match.sum().item()),
        int(dustbin_correct.item()),
        int(dustbin_total.item()),
    )


def _validation_report(
    *,
    arm: str,
    model: nn.Module,
    loader: Iterable[RachelBatch],
    manifest_by_id: Mapping[str, _ManifestRow],
    device: torch.device,
    precision: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    model.eval()
    previous_model_config = None
    if arm == "full_n512":
        assert isinstance(model, RachelN512Pairwise)
        previous_model_config = model.config
        model.config = replace(model.config, validate_runtime_inputs=True)

    probabilities: Dict[str, list] = {
        "coarse": [],
        "fused": [],
        "local": [],
    }
    validities: Dict[str, list] = {name: [] for name in probabilities}
    labels = []
    clusters = []
    pair_ids = []
    correspondence_tp = 0
    correspondence_predicted = 0
    mutual_correspondence_tp = 0
    mutual_correspondence_predicted = 0
    correspondence_target = 0
    dustbin_correct = 0
    dustbin_total = 0
    translation_errors = []
    started = time.perf_counter()

    try:
        with torch.inference_mode():
            for batch in loader:
                batch_rows = []
                for pair_id in batch.pair_ids:
                    row = manifest_by_id.get(pair_id)
                    if row is None:
                        raise RachelN512RunnerError(
                            "validation batch pair_id is absent from manifest"
                        )
                    batch_rows.append(row)
                expected_label = np.asarray(
                    [row.label for row in batch_rows], dtype=np.float32
                )
                if not np.array_equal(batch.labels, expected_label):
                    raise RachelN512RunnerError(
                        "validation loader labels disagree with manifest"
                    )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16",
                ):
                    if arm == "coarse_only":
                        mask_a, mask_b, _ = _coarse_batch(batch, device)
                        output = model(mask_a, mask_b)
                        coarse_probability = output.probability
                        coarse_valid = output.valid_problem
                    else:
                        inputs, targets = _full_batch(batch, device)
                        output = model(*inputs)
                        coarse_probability = output.coarse_probability
                        coarse_valid = output.coarse.valid_problem
                probabilities["coarse"].extend(
                    coarse_probability.detach().float().cpu().tolist()
                )
                validities["coarse"].extend(coarse_valid.detach().cpu().tolist())
                if arm == "full_n512":
                    probabilities["fused"].extend(
                        output.fused_probability.detach().float().cpu().tolist()
                    )
                    probabilities["local"].extend(
                        output.local_probability.detach().float().cpu().tolist()
                    )
                    decision_valid = output.decision_valid
                    validities["fused"].extend(decision_valid.detach().cpu().tolist())
                    validities["local"].extend(decision_valid.detach().cpu().tolist())
                    target_a = targets[1]
                    target_b = targets[2]
                    counts = _correspondence_counts(
                        output.assignment,
                        output.unmatched_a,
                        output.unmatched_b,
                        target_a,
                        target_b,
                        decision_valid,
                    )
                    correspondence_tp += counts[0]
                    correspondence_predicted += counts[1]
                    mutual_correspondence_tp += counts[2]
                    mutual_correspondence_predicted += counts[3]
                    correspondence_target += counts[4]
                    dustbin_correct += counts[5]
                    dustbin_total += counts[6]
                    translation_mask = targets[4] & decision_valid
                    error = torch.linalg.vector_norm(
                        output.translation_hat_rc - targets[3], dim=1
                    )
                    translation_errors.extend(
                        error[translation_mask].detach().float().cpu().tolist()
                    )
                labels.extend(row.label for row in batch_rows)
                clusters.extend(row.cluster_id for row in batch_rows)
                pair_ids.extend(batch.pair_ids)
    finally:
        if previous_model_config is not None:
            assert isinstance(model, RachelN512Pairwise)
            model.config = previous_model_config

    methods = ("coarse",) if arm == "coarse_only" else ("coarse", "local", "fused")
    report: Dict[str, object] = {
        "sample_count": len(pair_ids),
        "pair_ids_sha256": hashlib.sha256(_canonical_bytes(pair_ids)).hexdigest(),
        "seconds": time.perf_counter() - started,
        "methods": {},
    }
    for method in methods:
        report["methods"][method] = _method_scores(
            probabilities[method], labels, validities[method], clusters
        )
        report["methods"][method]["coverage"] = {
            "valid_count": int(sum(bool(value) for value in validities[method])),
            "record_count": len(validities[method]),
            "valid_fraction": float(
                sum(bool(value) for value in validities[method])
                / len(validities[method])
            ),
        }
    if arm == "full_n512":
        precision = (
            correspondence_tp / correspondence_predicted
            if correspondence_predicted
            else 0.0
        )
        recall = (
            correspondence_tp / correspondence_target if correspondence_target else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        mutual_precision = (
            mutual_correspondence_tp / mutual_correspondence_predicted
            if mutual_correspondence_predicted
            else 0.0
        )
        mutual_recall = (
            mutual_correspondence_tp / correspondence_target
            if correspondence_target
            else 0.0
        )
        mutual_f1 = (
            2.0 * mutual_precision * mutual_recall / (mutual_precision + mutual_recall)
            if mutual_precision + mutual_recall
            else 0.0
        )
        errors = np.asarray(translation_errors, dtype=np.float64)
        report["correspondence"] = {
            "metric_boundary": {
                "strict": "mutual_top1_and_real_edge_probability_exceeds_dustbin",
                "mutual_top1": "threshold_free_real_edge_rank_diagnostic",
            },
            "true_positive": correspondence_tp,
            "predicted_count": correspondence_predicted,
            "target_count": correspondence_target,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mutual_top1_true_positive": mutual_correspondence_tp,
            "mutual_top1_predicted_count": mutual_correspondence_predicted,
            "mutual_top1_precision": mutual_precision,
            "mutual_top1_recall": mutual_recall,
            "mutual_top1_f1": mutual_f1,
            "dustbin_accuracy": (
                dustbin_correct / dustbin_total if dustbin_total else 0.0
            ),
            "dustbin_token_count": dustbin_total,
        }
        report["translation"] = {
            "valid_count": int(errors.size),
            "median_l2_px": float(np.median(errors)) if errors.size else None,
            "p90_l2_px": float(np.quantile(errors, 0.9)) if errors.size else None,
            "success_at_4px": float(np.mean(errors <= 4.0)) if errors.size else None,
            "success_at_8px": float(np.mean(errors <= 8.0)) if errors.size else None,
            "success_at_16px": float(np.mean(errors <= 16.0)) if errors.size else None,
        }
    score_artifact = {
        "pair_ids": pair_ids,
        "labels": labels,
        "clusters": clusters,
        "probability": probabilities["coarse" if arm == "coarse_only" else "fused"],
        "valid": validities["coarse" if arm == "coarse_only" else "fused"],
    }
    return report, score_artifact


def _arm_selection_metrics(arm: str, report: Mapping[str, object]) -> Dict[str, object]:
    method = "coarse" if arm == "coarse_only" else "fused"
    metrics = report["methods"][method]["cluster_balanced"]
    auroc = float(metrics["auroc"])
    auprc = float(metrics["auprc"])
    coverage = float(report["methods"][method]["coverage"]["valid_fraction"])
    if arm == "coarse_only":
        return {
            "primary_score": auroc,
            "auroc": auroc,
            "auprc": auprc,
            "coverage": coverage,
            "weighting": "equal_lineage_or_lineage_pair_cluster",
        }
    correspondence_f1 = float(report["correspondence"]["mutual_top1_f1"])
    translation_success = report["translation"]["success_at_8px"]
    translation_success = (
        float(translation_success) if translation_success is not None else 0.0
    )
    primary = (
        float(
            max(auroc, 0.0)
            * max(auprc, 0.0)
            * max(correspondence_f1, 0.0)
            * max(translation_success, 0.0)
        )
        ** 0.25
    )
    return {
        "primary_score": primary,
        "auroc": auroc,
        "auprc": auprc,
        "coverage": coverage,
        "correspondence_f1": correspondence_f1,
        "correspondence_metric": "mutual_top1_f1",
        "translation_success_at_8px": translation_success,
        "weighting": "equal_lineage_or_lineage_pair_cluster",
        "protocol": "geometric_mean_pair_auroc_pair_auprc_corr_f1_translation_at_8px",
    }


def run_training(config: RachelN512RunConfig) -> Path:
    """Train requested arms and select winners using validation only."""

    if not torch.cuda.is_available():
        raise RachelN512RunnerError("formal Rachel training requires CUDA")
    device = torch.device(config.device)
    _set_determinism(config.seed)
    dataset_root = config.dataset_root.resolve(strict=True)
    output_root = config.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _run_fingerprint(
        replace(config, dataset_root=dataset_root, output_root=output_root)
    )
    final_directory = output_root / ("run-" + fingerprint[:16])
    partial_directory = output_root / (".partial-run-" + fingerprint[:16])
    if final_directory.exists() or partial_directory.exists():
        raise RachelN512RunnerError(
            "run directory already exists; preserve it and choose a fresh output/config"
        )
    partial_directory.mkdir()
    _atomic_json(
        partial_directory / "run_config.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running_train_validation_only",
            "fingerprint_sha256": fingerprint,
            "config": config.portable_dict(),
            "test_accessed": False,
            "real_external_test_accessed": False,
        },
    )

    try:
        train_dataset = RachelPairDataset(dataset_root, "train")
        val_dataset = RachelPairDataset(dataset_root, "val")
        train_manifest = _read_manifest_rows(dataset_root, "train")
        val_manifest = _read_manifest_rows(dataset_root, "val")
        if len(train_dataset) != len(train_manifest) or len(val_dataset) != len(
            val_manifest
        ):
            raise RachelN512RunnerError("loader/manifest population count mismatch")
        train_subset = stratified_subset_indices(
            [row.label for row in train_manifest],
            config.seed + 31_337,
            config.max_train_pairs,
        )
        val_order = stratified_subset_indices(
            [row.label for row in val_manifest],
            config.seed + 91_733,
            config.max_val_pairs,
        )
        val_rows = tuple(val_manifest[index] for index in val_order)
        val_by_id = {row.pair_id: row for row in val_rows}
        if len(val_by_id) != len(val_rows):
            raise RachelN512RunnerError("validation subset contains duplicate pair IDs")
        val_loader = _loader(
            val_dataset,
            val_order,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed + 77,
        )

        arm_results = []
        for arm in config.arms:
            _set_determinism(config.seed)
            model_config = RachelN512Config(validate_runtime_inputs=False)
            model = _new_model(arm, model_config, device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            fast_loss_config = RachelN512LossConfig(
                validate_runtime_targets=False,
                collect_cpu_diagnostics=False,
            )
            arm_directory = partial_directory / arm
            arm_directory.mkdir()
            epoch_rows = []
            score_by_epoch = {}
            for epoch in range(1, config.epochs + 1):
                shuffled_positions = epoch_indices(
                    len(train_subset), config.seed, epoch, None
                )
                train_order = tuple(
                    train_subset[position] for position in shuffled_positions
                )
                train_loader = _loader(
                    train_dataset,
                    train_order,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    seed=config.seed + epoch,
                )
                train_report = _train_epoch(
                    arm=arm,
                    model=model,
                    optimizer=optimizer,
                    loader=train_loader,
                    device=device,
                    precision=config.precision,
                    loss_config=fast_loss_config,
                    gradient_clip_norm=config.gradient_clip_norm,
                    log_every_steps=config.log_every_steps,
                    epoch=epoch,
                )
                validation_report, scores = _validation_report(
                    arm=arm,
                    model=model,
                    loader=val_loader,
                    manifest_by_id=val_by_id,
                    device=device,
                    precision=config.precision,
                )
                selection_metrics = _arm_selection_metrics(arm, validation_report)
                checkpoint_path = arm_directory / ("epoch-{:03d}.pt".format(epoch))
                checkpoint = {
                    "schema_version": "rachel-n512-checkpoint/1.0",
                    "arm": arm,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": asdict(model_config),
                    "loss_config": asdict(fast_loss_config),
                    "run_config": config.portable_dict(),
                    "train_report": train_report,
                    "validation_report": validation_report,
                    "selection_metrics": selection_metrics,
                    "test_accessed": False,
                    "real_external_test_accessed": False,
                }
                _atomic_torch_save(checkpoint_path, checkpoint)
                checkpoint_sha = _sha256_file(checkpoint_path)
                epoch_row = {
                    "epoch": epoch,
                    "checkpoint": str(checkpoint_path.relative_to(partial_directory)),
                    "checkpoint_sha256": checkpoint_sha,
                    "train": train_report,
                    "validation": validation_report,
                    "selection_metrics": selection_metrics,
                }
                epoch_rows.append(epoch_row)
                score_by_epoch[epoch] = scores
                _atomic_json(arm_directory / "epochs.json", epoch_rows)
                print(
                    json.dumps(
                        {
                            "event": "epoch_complete",
                            "arm": arm,
                            "epoch": epoch,
                            **selection_metrics,
                            "train_loss": train_report["mean_loss"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            winner_epoch = select_winner_epoch(epoch_rows)
            winner_row = next(row for row in epoch_rows if row["epoch"] == winner_epoch)
            winner_scores = score_by_epoch[winner_epoch]
            model_config_sha = hashlib.sha256(
                _canonical_bytes(asdict(model_config))
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
                "parameter_count": _parameter_count(model),
                "epochs": epoch_rows,
                "winner_epoch": winner_epoch,
                "winner_checkpoint": winner_row["checkpoint"],
                "winner_checkpoint_sha256": winner_row["checkpoint_sha256"],
                "winner_selection_metrics": winner_row["selection_metrics"],
                "validation_threshold": threshold.to_dict(),
                "test_accessed": False,
                "real_external_test_accessed": False,
            }
            _atomic_json(arm_directory / "arm_result.json", arm_result)
            arm_results.append(arm_result)
            del optimizer, model
            torch.cuda.empty_cache()

        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_train_validation_only",
            "fingerprint_sha256": fingerprint,
            "config": config.portable_dict(),
            "population": {
                "train_total": len(train_dataset),
                "val_total": len(val_dataset),
                "train_rows_per_epoch": len(train_subset),
                "val_rows": len(val_order),
            },
            "arm_results": arm_results,
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
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=260831)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--max-val-pairs", type=int)
    parser.add_argument("--log-every-steps", type=int, default=50)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    run_directory = run_training(
        RachelN512RunConfig(
            dataset_root=arguments.dataset_root,
            output_root=arguments.output_root,
            arms=tuple(arguments.arms),
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            gradient_clip_norm=arguments.gradient_clip_norm,
            seed=arguments.seed,
            precision=arguments.precision,
            device=arguments.device,
            num_workers=arguments.num_workers,
            max_train_pairs=arguments.max_train_pairs,
            max_val_pairs=arguments.max_val_pairs,
            log_every_steps=arguments.log_every_steps,
        )
    )
    print(str(run_directory), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "RachelN512RunConfig",
    "RachelN512RunnerError",
    "epoch_indices",
    "main",
    "run_training",
    "select_winner_epoch",
    "validation_indices",
]

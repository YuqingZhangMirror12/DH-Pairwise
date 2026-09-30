"""Train/validation-only PairingNet-derived baseline for Rachel masks.

This is a documented task adaptation of PairingNet, not an unmodified-source
reproduction.  See ``PAIRINGNET_RACHEL_ADAPTATION.md`` beside this file.

The executable deliberately names and opens only the Rachel ``train`` and
``val`` manifests.  Sealed synthetic and real evaluation are separate protocol
steps owned by the final evaluation controller.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import subprocess
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import (
    RachelBatch,
    RachelDatasetConfig,
    RachelPairDataset,
)
from staging.pairwise_v0_2.baselines.rachel_materialized_training import (
    benchmark_manifest_lines, selected_manifest_path, materialized_train_dataset,
    training_hook_identity,
    collate_benchmark_pairs as collate_rachel_pairs,
)
from staging.pairwise_v0_2.training.rachel_n512_runner import epoch_indices
from staging.pairwise_v0_2.training.evaluation import (
    evaluate_pairwise,
    fit_pairwise_threshold,
)


SCHEMA_VERSION = "rachel-pairingnet-adapted-benchmark/1.0"
OFFICIAL_REPOSITORY = "https://github.com/zhourixin/PairingNet"
OFFICIAL_COMMIT = "e878b781b2b2065a4b7da09d2f639e8f0a35e97a"
OFFICIAL_PAPER = "https://arxiv.org/abs/2312.08704"
FORMAL_SEED = 260831
FORMAL_TRAIN_ROWS = 24_000
FORMAL_VAL_ROWS = 3_000
ASSEMBLY_TOLERANCES_PX = (2.0, 5.0, 8.0, 10.0)
PAIR_THRESHOLD = 0.5
METHOD_ID = "pairingnet_rachel_mask_n512_upright_translation_v1"
WINNER_CHECKPOINT_KIND = "frozen_train_val_winner"
LAST_CHECKPOINT_KIND = "last_resumable_completed_epoch"
ADAPTATION_CONTRACT_FILENAME = "PAIRINGNET_RACHEL_ADAPTATION.md"
OFFICIAL_KEY_FILE_SHA256 = {
    "PairingNet Code/run.py": "bac51a2eb26832a1f6819361ccdb3dfb823cbe071ae6e5fb12ca3a1b99c0eeee",
    "PairingNet Code/utils/config.py": "b0d74cd5d76feb1018eceea32e5e6c02acd8e2b25d1f9e8c6c44368f66ef25ac",
    "PairingNet Code/utils/pipeline.py": "2201b0f2345355fd529ceccdbb63e5891367f5633793d369fa83bba0a16766af",
    "PairingNet Code/utils/encoder.py": "277b29352a73eb0ab083c3ead814d26b359aec15531ec3e253cbe98f69845106",
    "PairingNet Code/utils/loss.py": "a084522f67409e4a755be78d1cc59c8c260302757c7f94d27008068db6d45e07",
    "PairingNet Code/utils/evaluation.py": "5e135d5875355af91250337850260251816d18907fd80a31348bd1d1414a70ac",
    "PairingNet Code/utils/ransac.py": "2a657035810c1a17f5983a0c079d9cbd4b561e9fbc5399531af83076f9494e0f",
    "PairingNet Code/PairingNet_train_val_test.py": "b8bd21d882bd2e61ff892f5024e2349ce852a53b77861ebfe1d52c105888ef82",
    "PairingNet Code/matching_test.py": "01a19d723a2a1643992e7be510899efdc153fe0bd04a95050e935679eba65bec",
}


class PairingNetBenchmarkError(RuntimeError):
    """A frozen benchmark invariant was violated."""


@dataclass(frozen=True)
class PairingNetRachelModelConfig:
    canvas_size: int = 800
    contour_cap: int = 512
    patch_size: int = 7
    feature_dim: int = 64
    resgcn_layers: int = 14
    contour_neighbor_radius: int = 8
    pair_hidden_dim: int = 32
    inference_epsilon: float = 0.006
    translation_inlier_px: float = 10.0
    translation_min_candidates: int = 4
    se2_ransac_min_candidates: int = 20

    def __post_init__(self) -> None:
        integer_fields = (
            "canvas_size",
            "contour_cap",
            "patch_size",
            "feature_dim",
            "resgcn_layers",
            "contour_neighbor_radius",
            "pair_hidden_dim",
            "translation_min_candidates",
            "se2_ransac_min_candidates",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        if self.patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        if self.feature_dim != 64:
            raise ValueError("the audited PairingNet local feature dimension is 64")
        for name in ("inference_epsilon", "translation_inlier_px"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(name + " must be finite and positive")


@dataclass(frozen=True)
class PairingNetRachelRunConfig:
    dataset_root: Path
    output_root: Path
    official_source_root: Path
    seed: int = FORMAL_SEED
    batch_size: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    min_epochs: int = 20
    max_epochs: int = 128
    patience: int = 12
    relative_improvement: float = 0.005
    eta_min: float = 1e-6
    num_workers: int = 4
    gradient_clip_norm: float = 5.0
    pair_loss_weight: float = 1.0
    matching_loss_weight: float = 1.0
    device: str = "cuda:0"
    precision: str = "fp32"
    log_every_steps: int = 50
    train_materialized_manifest: Optional[Path] = None
    experimental_data: bool = False
    selection_metric: str = "official"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).expanduser())
        object.__setattr__(self, "output_root", Path(self.output_root).expanduser())
        object.__setattr__(
            self,
            "official_source_root",
            Path(self.official_source_root).expanduser(),
        )
        if type(self.experimental_data) is not bool:
            raise TypeError("experimental_data must be bool")
        if self.train_materialized_manifest is not None:
            object.__setattr__(self, "train_materialized_manifest", Path(self.train_materialized_manifest).expanduser())
        if self.experimental_data != (self.train_materialized_manifest is not None):
            raise ValueError("experimental data requires an explicit materialized TRAIN manifest, and vice versa")
        if self.selection_metric not in {"official", "recall95_precision"}:
            raise ValueError("selection_metric must be official or recall95_precision")
        if self.selection_metric != "official" and not self.experimental_data:
            raise ValueError("recall selection requires explicit experimental data")
        if type(self.seed) is not int or self.seed <= 0:
            raise ValueError("seed must be a positive integer")
        if self.seed != FORMAL_SEED and not self.experimental_data:
            raise ValueError("formal PairingNet benchmark seed must be 260831")
        for name in (
            "batch_size",
            "min_epochs",
            "max_epochs",
            "patience",
            "log_every_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers < 0:  # noqa: E721
            raise ValueError("num_workers must be a non-negative integer")
        if not self.min_epochs <= self.max_epochs:
            raise ValueError("min_epochs must not exceed max_epochs")
        for name in (
            "learning_rate",
            "relative_improvement",
            "eta_min",
            "gradient_clip_norm",
            "pair_loss_weight",
            "matching_loss_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(name + " must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if not self.device.startswith("cuda"):
            raise ValueError("formal benchmark training requires a CUDA device")

    def portable_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["dataset_root"] = str(self.dataset_root.resolve())
        # Output identity is lexical.  Resolving it here would make an alias
        # through a symlink compare equal to an approved run-state path.
        value["output_root"] = str(_lexical_absolute_path(self.output_root))
        value["official_source_root"] = str(self.official_source_root.resolve())
        value["train_materialized_manifest"] = (str(self.train_materialized_manifest.resolve())
            if self.train_materialized_manifest is not None else None)
        return value


@dataclass(frozen=True)
class ManifestRow:
    pair_id: str
    label: bool
    split_units: Tuple[str, ...]
    cluster_id: str


@dataclass(frozen=True)
class PairingNetRachelOutput:
    pair_logit: Tensor
    pair_probability: Tensor
    similarity: Tensor
    fused_a: Tensor
    fused_b: Tensor
    valid_a: Tensor
    valid_b: Tensor


@dataclass(frozen=True)
class PairingNetLossOutput:
    total: Tensor
    matching_focal: Tensor
    pair_bce: Tensor


@dataclass(frozen=True)
class TranslationEstimate:
    valid: bool
    translation_rc: Optional[np.ndarray]
    candidate_count: int
    inlier_count: int


@dataclass(frozen=True)
class SE2Estimate:
    valid: bool
    matrix_xy: Optional[np.ndarray]
    candidate_count: int
    inlier_count: int


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


def _sha256_json_value(value: object) -> str:
    """Hash the exact canonical JSON-line representation written on disk."""

    return hashlib.sha256(_canonical_bytes(value) + b"\n").hexdigest()


def _lexical_absolute_path(path: Path) -> Path:
    """Return an absolute, normalized path without following any symlink."""

    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(os.fspath(expanded)))


def _lstat_optional(path: Path) -> Optional[os.stat_result]:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise PairingNetBenchmarkError("cannot lstat guarded path: " + str(path)) from error


def _reject_symlink_components(path: Path) -> Path:
    """Fail closed if any existing component is a symbolic link.

    ``Path.resolve`` is intentionally forbidden for this guard: resolving and
    then comparing would allow an attacker-controlled alias to become equal to
    an approved output, partial, failed, or resume directory.
    """

    absolute = _lexical_absolute_path(path)
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        metadata = _lstat_optional(current)
        if metadata is None:
            # Descendants cannot exist once an ancestor is absent.
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise PairingNetBenchmarkError(
                "symbolic-link path component is forbidden: " + str(current)
            )
    return absolute


def _require_guarded_regular_file(path: Path, label: str) -> Path:
    guarded = _reject_symlink_components(path)
    metadata = _lstat_optional(guarded)
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        raise PairingNetBenchmarkError(label + " must be a regular file")
    return guarded


def _validated_run_state_paths(
    output_root: Path, resume_from: Optional[Path]
) -> Tuple[Path, Path, Path, Optional[Path]]:
    """Validate lexical run-state identity and every existing path component."""

    output = _reject_symlink_components(output_root)
    partial = _reject_symlink_components(
        output.with_name("." + output.name + ".partial")
    )
    failed = _reject_symlink_components(
        output.with_name("." + output.name + ".failed")
    )
    parent_metadata = _lstat_optional(output.parent)
    if parent_metadata is None or not stat.S_ISDIR(parent_metadata.st_mode):
        raise PairingNetBenchmarkError(
            "output parent must already be a non-symlink directory"
        )
    if _lstat_optional(output) is not None:
        raise PairingNetBenchmarkError("completed output already exists")

    partial_metadata = _lstat_optional(partial)
    failed_metadata = _lstat_optional(failed)
    if resume_from is None:
        if partial_metadata is not None or failed_metadata is not None:
            raise PairingNetBenchmarkError(
                "partial/failed state exists; pass its exact path with --resume-from"
            )
        return output, partial, failed, None

    resume = _reject_symlink_components(resume_from)
    if resume != partial and resume != failed:
        raise PairingNetBenchmarkError(
            "--resume-from must be this output's lexical-exact .partial or .failed directory"
        )
    resume_metadata = _lstat_optional(resume)
    if resume_metadata is None or not stat.S_ISDIR(resume_metadata.st_mode):
        raise PairingNetBenchmarkError("--resume-from must be an existing directory")
    other_metadata = failed_metadata if resume == partial else partial_metadata
    if other_metadata is not None:
        raise PairingNetBenchmarkError("ambiguous partial and failed resume states")
    return output, partial, failed, resume


def _create_guarded_run_directory(path: Path) -> None:
    guarded = _reject_symlink_components(path)
    parent_metadata = _lstat_optional(guarded.parent)
    if parent_metadata is None or not stat.S_ISDIR(parent_metadata.st_mode):
        raise PairingNetBenchmarkError("run-state parent is not a directory")
    if _lstat_optional(guarded) is not None:
        raise PairingNetBenchmarkError("run-state destination already exists")
    guarded.mkdir()


def _move_guarded_run_directory(source: Path, destination: Path) -> None:
    source = _reject_symlink_components(source)
    destination = _reject_symlink_components(destination)
    if source.parent != destination.parent:
        raise PairingNetBenchmarkError("run-state move must remain within one parent")
    source_metadata = _lstat_optional(source)
    if source_metadata is None or not stat.S_ISDIR(source_metadata.st_mode):
        raise PairingNetBenchmarkError("run-state source is not a directory")
    if _lstat_optional(destination) is not None:
        raise PairingNetBenchmarkError("run-state destination already exists")
    os.replace(source, destination)


def _adapter_identity_hashes() -> Tuple[str, str]:
    source_path = Path(__file__).resolve(strict=True)
    contract_path = source_path.with_name(ADAPTATION_CONTRACT_FILENAME)
    return _sha256_file(source_path), _sha256_file(contract_path)


def _atomic_json(path: Path, value: object) -> None:
    path = _reject_symlink_components(path)
    temporary = _reject_symlink_components(path.with_name("." + path.name + ".tmp"))
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: object) -> None:
    path = _reject_symlink_components(path)
    temporary = _reject_symlink_components(path.with_name("." + path.name + ".tmp"))
    torch.save(value, temporary)
    os.replace(temporary, path)


def _capture_rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": tuple(torch.cuda.get_rng_state_all()),
    }


def _restore_rng_state(value: object) -> None:
    if not isinstance(value, Mapping):
        raise PairingNetBenchmarkError("resume checkpoint lacks RNG state")
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(value) != required:
        raise PairingNetBenchmarkError("resume RNG state fields changed")
    cuda_state = value["torch_cuda"]
    if not isinstance(cuda_state, (tuple, list)) or len(cuda_state) != torch.cuda.device_count():
        raise PairingNetBenchmarkError("resume CUDA RNG/device count mismatch")
    try:
        random.setstate(value["python"])
        np.random.set_state(value["numpy"])
        torch.set_rng_state(value["torch_cpu"].cpu())
        torch.cuda.set_rng_state_all(list(cuda_state))
    except (TypeError, ValueError, RuntimeError) as error:
        raise PairingNetBenchmarkError("cannot restore strict RNG state") from error


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # PairingNet's graph message aggregation is scatter/gather based on CUDA.
    # The official implementation does not claim bitwise determinism, and some
    # target torch/CUDA combinations reject its backward pass when strict
    # deterministic algorithms are forced.  We freeze every RNG/order control
    # but make no byte-for-byte CUDA claim.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _read_manifest(root: Path, split: str, train_materialized_manifest=None) -> Tuple[ManifestRow, ...]:
    if split not in {"train", "val"}:
        raise ValueError("benchmark manifest split must be train or val")
    rows: List[ManifestRow] = []
    seen = set()
    with benchmark_manifest_lines(root, split, train_materialized_manifest) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise PairingNetBenchmarkError(
                    "invalid {} manifest JSON line {}".format(split, line_number)
                ) from error
            if value.get("split") != split or type(value.get("label")) is not bool:  # noqa: E721
                raise PairingNetBenchmarkError(
                    "malformed {} manifest line {}".format(split, line_number)
                )
            pair_id = value.get("pair_id")
            first = value.get("fragment_a", {}).get("split_unit_id")
            second = value.get("fragment_b", {}).get("split_unit_id")
            if not all(
                isinstance(item, str) and item for item in (pair_id, first, second)
            ):
                raise PairingNetBenchmarkError(
                    "manifest lacks pair/parent-lineage identity"
                )
            if pair_id in seen:
                raise PairingNetBenchmarkError("duplicate pair_id: " + pair_id)
            seen.add(pair_id)
            ordered_units = tuple(sorted((first, second)))
            unique_units = tuple(sorted(set(ordered_units)))
            cluster_id = (
                "unit:" + ordered_units[0]
                if ordered_units[0] == ordered_units[1]
                else "unit-pair:"
                + hashlib.sha256(_canonical_bytes(list(ordered_units))).hexdigest()
            )
            rows.append(
                ManifestRow(
                    pair_id=pair_id,
                    label=value["label"],
                    split_units=unique_units,
                    cluster_id=cluster_id,
                )
            )
    if not rows:
        raise PairingNetBenchmarkError(split + " manifest is empty")
    return tuple(rows)


def _git_output(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PairingNetBenchmarkError("cannot audit official git checkout") from error
    if completed.returncode != 0:
        raise PairingNetBenchmarkError(
            "official git audit failed: " + completed.stderr.strip()
        )
    return completed.stdout.strip()


def audit_official_source(root: Path) -> Dict[str, object]:
    """Verify the exact clean PairingNet checkout used as adaptation authority."""

    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise PairingNetBenchmarkError("official source root is not a directory")
    top_level = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top_level != root:
        raise PairingNetBenchmarkError("official source root is not the git top level")
    head = _git_output(root, "rev-parse", "HEAD")
    if head != OFFICIAL_COMMIT:
        raise PairingNetBenchmarkError("official source commit mismatch")
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise PairingNetBenchmarkError("official source checkout is not clean")
    actual_hashes = {}
    for relative, expected in OFFICIAL_KEY_FILE_SHA256.items():
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise PairingNetBenchmarkError("official key file escapes checkout") from error
        if not path.is_file():
            raise PairingNetBenchmarkError("official key path is not a file: " + relative)
        actual = _sha256_file(path)
        if actual != expected:
            raise PairingNetBenchmarkError("official key file hash mismatch: " + relative)
        actual_hashes[relative] = actual
    origin = _git_output(root, "remote", "get-url", "origin")
    return {
        "schema_version": "pairingnet-official-source-audit/1.0",
        "status": "exact_clean_checkout_verified",
        "checkout_root": str(root),
        "repository": OFFICIAL_REPOSITORY,
        "origin_reported": origin,
        "commit": head,
        "clean": True,
        "key_file_sha256": actual_hashes,
    }


def audit_train_val_population(
    root: Path, *, require_formal_counts: bool = True, train_materialized_manifest=None
) -> Dict[str, object]:
    """Fail closed on population, balance and parent-lineage leakage."""

    root = Path(root).expanduser().resolve(strict=True)
    train_path = selected_manifest_path(root, "train", train_materialized_manifest)
    val_path = root / "pairs" / "val.jsonl"
    train = _read_manifest(root, "train", train_materialized_manifest)
    val = _read_manifest(root, "val")
    train_units = {unit for row in train for unit in row.split_units}
    val_units = {unit for row in val for unit in row.split_units}
    overlap = train_units & val_units
    if overlap:
        raise PairingNetBenchmarkError(
            "train/validation split_unit_id leakage: {} units".format(len(overlap))
        )
    if require_formal_counts and (
        len(train) != FORMAL_TRAIN_ROWS or len(val) != FORMAL_VAL_ROWS
    ):
        raise PairingNetBenchmarkError(
            "formal Rachel population must be exactly 24000 train / 3000 val"
        )
    counts: Dict[str, Dict[str, int]] = {}
    for split, rows in (("train", train), ("val", val)):
        positives = sum(row.label for row in rows)
        negatives = len(rows) - positives
        if positives != negatives:
            raise PairingNetBenchmarkError(split + " is no longer exact 1:1")
        counts[split] = {
            "rows": len(rows),
            "positive": positives,
            "negative": negatives,
            "split_unit_count": len(
                {unit for row in rows for unit in row.split_units}
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "train_val_population_audited",
        "official_reference": {
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "paper": OFFICIAL_PAPER,
        },
        "population": counts,
        "manifests": {
            "train_sha256": _sha256_file(train_path),
            "val_sha256": _sha256_file(val_path),
            **({"materialized_training_hook": training_hook_identity()}
               if train_materialized_manifest is not None else {}),
        },
        "parent_lineage_disjoint": True,
        "seed": FORMAL_SEED,
        "max_train_pairs": None,
        "max_val_pairs": None,
        "formal_population_required": require_formal_counts,
        "train_materialized_manifest": (str(Path(train_materialized_manifest).resolve())
            if train_materialized_manifest is not None else None),
        "sealed_synthetic_accessed": False,
        "real_data_accessed": False,
    }


def _worker_seed(worker_id: int) -> None:
    del worker_id
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


class _ConvBNReLU(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(value)))


def _sample_ordered_patches(
    image: Tensor, points_rc: Tensor, patch_size: int
) -> Tensor:
    """Deterministic integer patches matching PairingNet's direct indexing."""

    if image.ndim != 4 or points_rc.ndim != 3 or points_rc.shape[-1] != 2:
        raise ValueError("invalid image/point tensor shapes")
    batch, channels, height, width = image.shape
    if points_rc.shape[0] != batch:
        raise ValueError("image and point batch sizes disagree")
    radius = patch_size // 2
    offsets = torch.arange(
        -radius, radius + 1, device=points_rc.device, dtype=points_rc.dtype
    )
    offset_row, offset_col = torch.meshgrid(offsets, offsets, indexing="ij")
    offset = torch.stack((offset_row, offset_col), dim=-1)
    centre = torch.round(points_rc).to(torch.long)
    sample_rc = centre[:, :, None, None, :] + offset.to(torch.long)[
        None, None, :, :, :
    ]
    sample_row = sample_rc[..., 0].clamp(0, height - 1)
    sample_col = sample_rc[..., 1].clamp(0, width - 1)
    batch_index = torch.arange(batch, device=image.device)[:, None, None, None]
    image_nhwc = image.permute(0, 2, 3, 1)
    sampled = image_nhwc[batch_index, sample_row, sample_col]
    return sampled.permute(0, 1, 4, 2, 3).contiguous()


def _mask_boundary(mask: Tensor) -> Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B,1,H,W]")
    binary = (mask >= 0.5).to(mask.dtype)
    eroded = -F.max_pool2d(-binary, kernel_size=3, stride=1, padding=1)
    return ((binary > 0.5) & (eroded < 0.5)).to(mask.dtype)


class _ContourPatchEncoder(nn.Module):
    """PairingNet edge-only patch encoder, 1 x 7 x 7 to 64D."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.conv = _ConvBNReLU(1, feature_dim, kernel_size=3, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, patches: Tensor, valid: Tensor) -> Tensor:
        batch, count, channels, height, width = patches.shape
        value = patches.reshape(batch * count, channels, height, width)
        flat_valid = valid.reshape(batch * count)
        valid_index = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
        if valid_index.numel() == 0:
            raise ValueError("patch encoder requires a valid contour point")
        encoded = self.pool(self.conv(value.index_select(0, valid_index))).flatten(1)
        output = encoded.new_zeros((batch * count, encoded.shape[1]))
        output = output.index_copy(0, valid_index, encoded)
        return output.reshape(batch, count, -1)


class _MaskContextPatchEncoder(nn.Module):
    """Official texture-patch topology fed replicated mask occupancy."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.conv1 = _ConvBNReLU(3, 32, kernel_size=3, padding=1)
        self.conv2 = _ConvBNReLU(32, feature_dim, kernel_size=3, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, patches: Tensor, valid: Tensor) -> Tensor:
        batch, count, channels, height, width = patches.shape
        value = patches.reshape(batch * count, channels, height, width)
        flat_valid = valid.reshape(batch * count)
        valid_index = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
        if valid_index.numel() == 0:
            raise ValueError("context encoder requires a valid contour point")
        encoded = self.pool(
            self.conv2(self.conv1(value.index_select(0, valid_index)))
        ).flatten(1)
        output = encoded.new_zeros((batch * count, encoded.shape[1]))
        output = output.index_copy(0, valid_index, encoded)
        return output.reshape(batch, count, -1)


def _ring_neighbours(value: Tensor, valid: Tensor, radius: int) -> Tuple[Tensor, Tensor]:
    """Gather the ordered +/- radius cyclic neighbourhood for every node."""

    if value.ndim != 3 or valid.shape != value.shape[:2]:
        raise ValueError("ring graph value/valid shapes disagree")
    batch, count, channels = value.shape
    lengths = valid.sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("each ring graph must contain a valid node")
    expected_valid = (
        torch.arange(count, device=valid.device)[None, :] < lengths[:, None]
    )
    if not torch.equal(valid, expected_valid):
        raise ValueError("ring graph valid nodes must form a contiguous prefix")
    center = torch.arange(count, device=value.device)[None, :, None]
    offsets = torch.arange(-radius, radius + 1, device=value.device)[None, None, :]
    neighbour_index = (center + offsets) % lengths[:, None, None]
    gather_index = neighbour_index[..., None].expand(-1, -1, -1, channels)
    expanded = value[:, None, :, :].expand(-1, count, -1, -1)
    neighbours = torch.gather(expanded, 2, gather_index)
    neighbour_valid = valid[:, :, None].expand(batch, count, 2 * radius + 1)
    return neighbours, neighbour_valid


class _RingGATConv(nn.Module):
    """Single-head GATConv specialized to PairingNet's ordered ring graph."""

    def __init__(self, channels: int, radius: int) -> None:
        super().__init__()
        self.radius = radius
        self.projection = nn.Linear(channels, channels, bias=False)
        self.attention_source = nn.Parameter(torch.empty(channels))
        self.attention_target = nn.Parameter(torch.empty(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention_source[None, :])
        nn.init.xavier_uniform_(self.attention_target[None, :])
        nn.init.zeros_(self.bias)

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        projected = self.projection(value)
        neighbours, neighbour_valid = _ring_neighbours(
            projected, valid, self.radius
        )
        source_score = torch.einsum("bnkd,d->bnk", neighbours, self.attention_source)
        target_score = torch.einsum("bnd,d->bn", projected, self.attention_target)
        logits = F.leaky_relu(source_score + target_score[:, :, None], 0.2)
        # Give padded centre nodes one harmless finite softmax entry.  Their
        # output is zeroed below, while avoiding all--inf softmax NaNs that can
        # otherwise contaminate gradients through the residual graph.
        safe_neighbour_valid = neighbour_valid.clone()
        safe_neighbour_valid[:, :, self.radius] |= ~valid
        logits = logits.masked_fill(~safe_neighbour_valid, -torch.inf)
        weight = torch.softmax(logits, dim=-1)
        output = torch.sum(weight[..., None] * neighbours, dim=2) + self.bias
        return output * valid[:, :, None].to(output.dtype)


class _ResPlusGATLayer(nn.Module):
    """DeepGCN ``res+`` ordering: norm, activation, graph conv, residual."""

    def __init__(self, channels: int, radius: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU(inplace=True)
        self.graph = _RingGATConv(channels, radius)

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        batch, count, channels = value.shape
        flattened = value.reshape(batch * count, channels)
        flat_valid = valid.reshape(batch * count)
        valid_index = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
        if valid_index.numel() == 0:
            raise ValueError("graph normalization requires a valid contour point")
        normalized_valid = self.norm(flattened.index_select(0, valid_index))
        normalized = flattened.new_zeros(flattened.shape)
        normalized = normalized.index_copy(0, valid_index, normalized_valid)
        normalized = normalized.reshape(batch, count, channels)
        update = self.graph(self.activation(normalized), valid)
        return (value + update) * valid[:, :, None].to(value.dtype)


class _ResGCN(nn.Module):
    def __init__(self, channels: int, layers: int, radius: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_ResPlusGATLayer(channels, radius) for _ in range(layers)]
        )

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        for layer in self.layers:
            value = layer(value, valid)
        return value


class PairingNetRachelAdapted(nn.Module):
    """Mask-only PairingNet local matcher plus an explicit pairability head."""

    def __init__(
        self, config: PairingNetRachelModelConfig = PairingNetRachelModelConfig()
    ) -> None:
        super().__init__()
        self.config = config
        channels = config.feature_dim
        self.contour_patch = _ContourPatchEncoder(channels)
        self.context_patch = _MaskContextPatchEncoder(channels)
        self.contour_coordinate_projection = nn.Linear(channels + 2, channels)
        self.contour_gcn = _ResGCN(
            channels, config.resgcn_layers, config.contour_neighbor_radius
        )
        self.context_gcn = _ResGCN(
            channels, config.resgcn_layers, config.contour_neighbor_radius
        )
        self.gate1 = nn.Linear(2 * channels, channels)
        self.gate2 = nn.Linear(channels, channels)
        self.pair_head = nn.Sequential(
            nn.Linear(6, config.pair_hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(config.pair_hidden_dim, 1),
        )

    def _encode(
        self, mask: Tensor, points_rc: Tensor, valid: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        boundary = _mask_boundary(mask)
        contour_patches = _sample_ordered_patches(
            boundary, points_rc, self.config.patch_size
        )
        context_patches = _sample_ordered_patches(
            mask, points_rc, self.config.patch_size
        ).repeat(1, 1, 3, 1, 1)
        contour = self.contour_patch(contour_patches, valid)
        normalized_points = points_rc / (self.config.canvas_size / 2.0) - 1.0
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1).to(points_rc.dtype)
        mean = (
            normalized_points * valid[:, :, None].to(points_rc.dtype)
        ).sum(dim=1, keepdim=True) / denominator[:, :, None]
        centered = normalized_points - mean
        contour = self.contour_coordinate_projection(
            torch.cat((contour, centered), dim=-1)
        )
        contour = contour * valid[:, :, None].to(contour.dtype)
        context = self.context_patch(context_patches, valid)
        context = context * valid[:, :, None].to(context.dtype)
        contour = self.contour_gcn(contour, valid)
        context = self.context_gcn(context, valid)
        gate = torch.sigmoid(
            self.gate2(F.elu(self.gate1(torch.cat((contour, context), dim=-1))))
        )
        # This follows the released code's c*w + t*(1-w) ordering.
        fused = contour * gate + context * (1.0 - gate)
        return fused, contour, context

    @staticmethod
    def _dual_softmax(
        first: Tensor, second: Tensor, valid_a: Tensor, valid_b: Tensor
    ) -> Tensor:
        logits = torch.bmm(first, second.transpose(1, 2)) / math.sqrt(
            first.shape[-1]
        )
        pair_valid = valid_a[:, :, None] & valid_b[:, None, :]
        logits = logits.masked_fill(~pair_valid, -1e9)
        column_normalized = torch.softmax(logits, dim=1)
        row_normalized = torch.softmax(logits, dim=2)
        return column_normalized * row_normalized * pair_valid.to(logits.dtype)

    @staticmethod
    def _pair_evidence(similarity: Tensor, valid_a: Tensor, valid_b: Tensor) -> Tensor:
        pair_valid = valid_a[:, :, None] & valid_b[:, None, :]
        masked_similarity = similarity.masked_fill(~pair_valid, -torch.inf)
        row_max = torch.where(
            valid_a, masked_similarity.max(dim=2).values, 0.0
        )
        col_max = torch.where(
            valid_b, masked_similarity.max(dim=1).values, 0.0
        )
        valid_a_float = valid_a.to(similarity.dtype)
        valid_b_float = valid_b.to(similarity.dtype)
        count_a = valid_a_float.sum(dim=1).clamp_min(1.0)
        count_b = valid_b_float.sum(dim=1).clamp_min(1.0)
        mean_row = (row_max * valid_a_float).sum(dim=1) / count_a
        mean_col = (col_max * valid_b_float).sum(dim=1) / count_b
        centered_row = (row_max - mean_row[:, None]) * valid_a_float
        row_std = torch.sqrt(
            centered_row.square().sum(dim=1) / count_a + 1e-12
        )
        maximum = masked_similarity.amax(dim=(1, 2))
        flattened = masked_similarity.flatten(1)
        flattened_valid = pair_valid.flatten(1)
        top_count = min(32, flattened.shape[1])
        top_values = flattened.topk(top_count, dim=1).values
        valid_top_count = flattened_valid.sum(dim=1).clamp(min=1, max=top_count)
        top_rank_valid = (
            torch.arange(top_count, device=similarity.device)[None, :]
            < valid_top_count[:, None]
        )
        top_mean = torch.where(top_rank_valid, top_values, 0.0).sum(dim=1)
        top_mean = top_mean / valid_top_count.to(similarity.dtype)
        total_mass = similarity.sum(dim=(1, 2)) / torch.sqrt(count_a * count_b)
        return torch.stack(
            (maximum, mean_row, mean_col, row_std, top_mean, total_mass),
            dim=1,
        )

    def forward(
        self,
        mask_a: Tensor,
        mask_b: Tensor,
        points_rc_a: Tensor,
        points_rc_b: Tensor,
        valid_a: Tensor,
        valid_b: Tensor,
    ) -> PairingNetRachelOutput:
        fused_a, _, _ = self._encode(mask_a, points_rc_a, valid_a)
        fused_b, _, _ = self._encode(mask_b, points_rc_b, valid_b)
        similarity = self._dual_softmax(fused_a, fused_b, valid_a, valid_b)
        pair_logit = self.pair_head(
            self._pair_evidence(similarity, valid_a, valid_b)
        ).squeeze(1)
        return PairingNetRachelOutput(
            pair_logit=pair_logit,
            pair_probability=torch.sigmoid(pair_logit),
            similarity=similarity,
            fused_a=fused_a,
            fused_b=fused_b,
            valid_a=valid_a,
            valid_b=valid_b,
        )


def pairingnet_focal_loss(
    similarity: Tensor,
    target_a: Tensor,
    valid_a: Tensor,
    valid_b: Tensor,
    *,
    alpha: float = 0.55,
    gamma: float = 8.0,
    target_b: Optional[Tensor] = None,
) -> Tensor:
    """Released PairingNet focal form over valid matrix entries."""

    if similarity.ndim != 3 or target_a.shape != similarity.shape[:2]:
        raise ValueError("similarity/target shapes disagree")
    supervised_a = valid_a & (target_a != -2)
    supervised_b = valid_b if target_b is None else valid_b & (target_b != -2)
    pair_valid = supervised_a[:, :, None] & supervised_b[:, None, :]
    ground_truth = torch.zeros_like(pair_valid)
    matched = target_a >= 0
    if torch.any(matched):
        batch_index, source_index = torch.nonzero(matched, as_tuple=True)
        target_index = target_a[batch_index, source_index]
        if torch.any(target_index >= similarity.shape[2]):
            raise ValueError("correspondence target exceeds similarity matrix")
        ground_truth[batch_index, source_index, target_index] = True
    if torch.any(ground_truth & ~pair_valid):
        raise ValueError("ground truth references invalid contour points")
    positive_probability = similarity[ground_truth]
    negative_probability = similarity[pair_valid & ~ground_truth]
    positive_loss = -alpha * (1.0 - positive_probability).pow(gamma) * torch.log(
        positive_probability.clamp_min(1e-9)
    )
    negative_loss = -(1.0 - alpha) * negative_probability.pow(gamma) * torch.log(
        (1.0 - negative_probability).clamp_min(1e-9)
    )
    denominator = positive_loss.numel() + negative_loss.numel()
    if denominator <= 0:
        raise ValueError("focal loss received no valid similarity entries")
    # The factor 400 is present in the official released implementation.
    return 400.0 * (positive_loss.sum() + negative_loss.sum()) / denominator


def compute_pairingnet_loss(
    output: PairingNetRachelOutput,
    labels: Tensor,
    target_a: Tensor,
    *,
    pair_weight: float = 1.0,
    matching_weight: float = 1.0,
    target_b: Optional[Tensor] = None,
) -> PairingNetLossOutput:
    matching = pairingnet_focal_loss(
        output.similarity,
        target_a,
        output.valid_a,
        output.valid_b,
        target_b=target_b,
    )
    pair = F.binary_cross_entropy_with_logits(output.pair_logit, labels)
    total = matching_weight * matching + pair_weight * pair
    return PairingNetLossOutput(total=total, matching_focal=matching, pair_bce=pair)


def pairingnet_diagonal_morphology(similarity: np.ndarray) -> np.ndarray:
    """One erosion and dilation using the released diagonal kernels."""

    if similarity.ndim != 2 or not np.all(np.isfinite(similarity)):
        raise ValueError("similarity must be a finite matrix")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - formal runtime dependency
        raise PairingNetBenchmarkError("opencv-python is required") from error
    erosion_kernel = np.eye(3, dtype=np.uint8)
    erosion_kernel[1, 1] = 0
    erosion_kernel = np.rot90(erosion_kernel).copy()
    eroded = cv2.erode(
        np.asarray(similarity, dtype=np.float32),
        erosion_kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    dilation_kernel = erosion_kernel.copy()
    dilation_kernel[1, 1] = 1
    return cv2.dilate(
        eroded,
        dilation_kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _weighted_median(value: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(value, kind="mergesort")
    sorted_value = value[order]
    sorted_weight = weight[order]
    cutoff = 0.5 * float(sorted_weight.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weight), cutoff, side="left"))
    return float(sorted_value[min(index, len(sorted_value) - 1)])


def translation_only_consensus(
    similarity: np.ndarray,
    points_rc_a: np.ndarray,
    points_rc_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    config: PairingNetRachelModelConfig = PairingNetRachelModelConfig(),
) -> TranslationEstimate:
    """Fixed-threshold, robust translation estimate from local matches."""

    length_a = int(np.count_nonzero(valid_a))
    length_b = int(np.count_nonzero(valid_b))
    if length_a <= 0 or length_b <= 0:
        return TranslationEstimate(False, None, 0, 0)
    cropped = np.asarray(similarity[:length_a, :length_b], dtype=np.float32)
    enhanced = pairingnet_diagonal_morphology(cropped)
    source_index, target_index = np.where(enhanced > config.inference_epsilon)
    candidate_count = len(source_index)
    if candidate_count < config.translation_min_candidates:
        return TranslationEstimate(False, None, candidate_count, 0)
    delta = (
        np.asarray(points_rc_b[target_index], dtype=np.float64)
        - np.asarray(points_rc_a[source_index], dtype=np.float64)
    )
    weight = np.asarray(enhanced[source_index, target_index], dtype=np.float64)
    weight = np.maximum(weight, np.finfo(np.float64).tiny)
    initial = np.asarray(
        (
            _weighted_median(delta[:, 0], weight),
            _weighted_median(delta[:, 1], weight),
        ),
        dtype=np.float64,
    )
    residual = np.linalg.norm(delta - initial[None, :], axis=1)
    inlier = residual <= config.translation_inlier_px
    inlier_count = int(np.count_nonzero(inlier))
    if inlier_count < config.translation_min_candidates:
        return TranslationEstimate(False, None, candidate_count, inlier_count)
    refined = np.asarray(
        (
            _weighted_median(delta[inlier, 0], weight[inlier]),
            _weighted_median(delta[inlier, 1], weight[inlier]),
        ),
        dtype=np.float32,
    )
    return TranslationEstimate(True, refined, candidate_count, inlier_count)


def official_style_se2_ransac(
    similarity: np.ndarray,
    points_rc_a: np.ndarray,
    points_rc_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    config: PairingNetRachelModelConfig = PairingNetRachelModelConfig(),
) -> SE2Estimate:
    """Secondary OpenCV SE(2) RANSAC using PairingNet's fixed candidates.

    The candidate threshold and minimum count follow the official path.  The
    solver uses OpenCV partial-affine RANSAC for the inlier set, followed by a
    rigid Kabsch refit (no scale).  It is not a byte-for-byte port of the
    repository's custom Python RANSAC, so the receipt labels this diagnostic
    ``official_style`` rather than ``official_exact``.
    """

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - formal runtime dependency
        raise PairingNetBenchmarkError("opencv-python is required") from error
    length_a = int(np.count_nonzero(valid_a))
    length_b = int(np.count_nonzero(valid_b))
    enhanced = pairingnet_diagonal_morphology(
        np.asarray(similarity[:length_a, :length_b], dtype=np.float32)
    )
    source_index, target_index = np.where(enhanced > config.inference_epsilon)
    candidate_count = len(source_index)
    if candidate_count < config.se2_ransac_min_candidates:
        return SE2Estimate(False, None, candidate_count, 0)
    source_xy = np.asarray(points_rc_a[source_index, ::-1], dtype=np.float32)
    target_xy = np.asarray(points_rc_b[target_index, ::-1], dtype=np.float32)
    cv2.setRNGSeed(0)
    matrix, inlier = cv2.estimateAffinePartial2D(
        source_xy,
        target_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.translation_inlier_px,
        maxIters=4000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or not np.all(np.isfinite(matrix)):
        return SE2Estimate(False, None, candidate_count, 0)
    inlier_count = int(np.count_nonzero(inlier)) if inlier is not None else 0
    if inlier is not None and inlier_count >= 2:
        inlier_mask = np.asarray(inlier).reshape(-1).astype(np.bool_)
        source_inlier = source_xy[inlier_mask].astype(np.float64)
        target_inlier = target_xy[inlier_mask].astype(np.float64)
        source_center = source_inlier.mean(axis=0)
        target_center = target_inlier.mean(axis=0)
        covariance = (source_inlier - source_center).T @ (
            target_inlier - target_center
        )
        left, _, right_t = np.linalg.svd(covariance)
        rotation = right_t.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right_t[-1, :] *= -1.0
            rotation = right_t.T @ left.T
        translation = target_center - rotation @ source_center
        matrix = np.concatenate((rotation, translation[:, None]), axis=1)
    return SE2Estimate(
        True, np.asarray(matrix, dtype=np.float32), candidate_count, inlier_count
    )


def _polygon_area(points_rc: np.ndarray, valid: np.ndarray) -> float:
    # PairingNet's released NTE path casts the full ordered contour to int32
    # and calls cv2.contourArea.  Preserve that otherwise-surprising
    # quantization here so this secondary metric really is source-compatible.
    points = np.asarray(points_rc[valid], dtype=np.int32)
    if len(points) < 3:
        return 0.0
    try:
        import cv2

        return float(cv2.contourArea(points))
    except ImportError:  # pragma: no cover - OpenCV is a formal dependency
        row = points[:, 0].astype(np.float64)
        col = points[:, 1].astype(np.float64)
        return 0.5 * abs(
            float(
                np.dot(col, np.roll(row, -1))
                - np.dot(row, np.roll(col, -1))
            )
        )


def _symmetric_hausdorff(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) == 0 or len(second) == 0:
        return math.inf
    try:
        from scipy.spatial import cKDTree

        first_to_second = cKDTree(second).query(first, k=1)[0]
        second_to_first = cKDTree(first).query(second, k=1)[0]
        return float(max(np.max(first_to_second), np.max(second_to_first)))
    except ImportError:  # pragma: no cover - scipy is a formal dependency
        distance = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
        return float(max(distance.min(axis=1).max(), distance.min(axis=0).max()))


def _primary_geometry_record(
    *,
    estimate: TranslationEstimate,
    points_rc_a: np.ndarray,
    points_rc_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    target_a: np.ndarray,
    ground_truth_translation_rc: np.ndarray,
) -> Dict[str, object]:
    matched_source = np.flatnonzero(target_a >= 0)
    if len(matched_source) == 0:
        raise PairingNetBenchmarkError("positive pair has no seam correspondences")
    matched_target = target_a[matched_source]
    source_seam = np.asarray(points_rc_a[matched_source], dtype=np.float64)
    target_seam = np.asarray(points_rc_b[matched_target], dtype=np.float64)
    # The released PairingNet test path substitutes an identity transform when
    # RANSAC fails.  Translation-only identity is t=(0,0), retained here solely
    # for unconditional RR/HD/NTE compatibility aggregates.
    predicted = (
        np.asarray(estimate.translation_rc, dtype=np.float64)
        if estimate.valid and estimate.translation_rc is not None
        else np.zeros(2, dtype=np.float64)
    )
    transformed = source_seam + predicted[None, :]
    point_error = np.linalg.norm(transformed - target_seam, axis=1)
    e_rmse = math.sqrt(float(point_error.mean()))
    translation_error = float(
        np.linalg.norm(
            predicted - np.asarray(ground_truth_translation_rc, dtype=np.float64)
        )
    )
    area = _polygon_area(points_rc_a, valid_a) + _polygon_area(points_rc_b, valid_b)
    nte = translation_error / area if area > 0.0 else math.inf
    return {
        "pose_valid": bool(estimate.valid),
        "translation_rc": predicted.tolist() if estimate.valid else None,
        "translation_error_px": translation_error if estimate.valid else None,
        "compatibility_fallback_translation_error_px": translation_error,
        "pairingnet_e_rmse": e_rmse,
        "pairingnet_hd_px": _symmetric_hausdorff(transformed, target_seam),
        "pairingnet_nte": nte,
        "candidate_count": estimate.candidate_count,
        "inlier_count": estimate.inlier_count,
    }


def _secondary_geometry_record(
    *,
    estimate: SE2Estimate,
    points_rc_a: np.ndarray,
    points_rc_b: np.ndarray,
    target_a: np.ndarray,
    ground_truth_translation_rc: np.ndarray,
) -> Dict[str, object]:
    if not estimate.valid or estimate.matrix_xy is None:
        return {
            "pose_valid": False,
            "re_radians_conditioned": None,
            "translation_error_px_conditioned": None,
            "candidate_count": estimate.candidate_count,
            "inlier_count": estimate.inlier_count,
        }
    matrix = np.asarray(estimate.matrix_xy, dtype=np.float64)
    matched_source = np.flatnonzero(target_a >= 0)
    matched_target = target_a[matched_source]
    source_xy = np.asarray(points_rc_a[matched_source, ::-1], dtype=np.float64)
    target_rc = np.asarray(points_rc_b[matched_target], dtype=np.float64)
    transformed_xy = source_xy @ matrix[:, :2].T + matrix[:, 2][None, :]
    transformed_rc = transformed_xy[:, ::-1]
    rotation = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    rotation_error = abs(math.atan2(math.sin(rotation), math.cos(rotation)))
    predicted_translation_rc = np.asarray((matrix[1, 2], matrix[0, 2]))
    translation_error = float(
        np.linalg.norm(
            predicted_translation_rc
            - np.asarray(ground_truth_translation_rc, dtype=np.float64)
        )
    )
    point_error = np.linalg.norm(transformed_rc - target_rc, axis=1)
    return {
        "pose_valid": True,
        "re_radians_conditioned": rotation_error,
        "translation_error_px_conditioned": translation_error,
        "pairingnet_e_rmse_conditioned": math.sqrt(float(point_error.mean())),
        "pairingnet_hd_px_conditioned": _symmetric_hausdorff(
            transformed_rc, target_rc
        ),
        "candidate_count": estimate.candidate_count,
        "inlier_count": estimate.inlier_count,
    }


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        raise ValueError("AUROC requires both labels")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float(
        (ranks[labels].sum() - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_)
    positive = int(labels.sum())
    if positive <= 0:
        raise ValueError("average precision requires positives")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / positive)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_validation_records(
    records: Sequence[Mapping[str, object]],
    *,
    pair_threshold: float = PAIR_THRESHOLD,
) -> Dict[str, object]:
    if not records:
        raise ValueError("validation records cannot be empty")
    labels = np.asarray([bool(row["label"]) for row in records], dtype=np.bool_)
    scores = np.asarray(
        [float(row["pair_probability"]) for row in records], dtype=np.float64
    )
    clusters = tuple(
        str(row.get("cluster_id", "pair:" + str(row["pair_id"])))
        for row in records
    )
    if not np.all(np.isfinite(scores)):
        raise PairingNetBenchmarkError("validation pair scores are non-finite")
    predicted_pair = scores >= pair_threshold
    pair_tp = int(np.count_nonzero(predicted_pair & labels))
    pair_fp = int(np.count_nonzero(predicted_pair & ~labels))
    pair_fn = int(np.count_nonzero(~predicted_pair & labels))
    pair_precision = _safe_ratio(pair_tp, pair_tp + pair_fp)
    pair_recall = _safe_ratio(pair_tp, pair_tp + pair_fn)
    pair_f1 = (
        2.0 * pair_precision * pair_recall / (pair_precision + pair_recall)
        if pair_precision + pair_recall
        else 0.0
    )
    weighted_pair = evaluate_pairwise(
        scores,
        labels,
        np.ones(len(records), dtype=np.bool_),
        clusters,
        threshold=pair_threshold,
    )

    positive_rows = [row for row in records if bool(row["label"])]
    positive_count = len(positive_rows)
    pose_valid = np.asarray(
        [bool(row["primary_geometry"]["pose_valid"]) for row in positive_rows],
        dtype=np.bool_,
    )
    conditioned_te = np.asarray(
        [
            float(row["primary_geometry"]["translation_error_px"])
            for row in positive_rows
            if bool(row["primary_geometry"]["pose_valid"])
        ],
        dtype=np.float64,
    )
    assembly: Dict[str, object] = {}
    positive_translation_success: Dict[str, float] = {}
    for tolerance in ASSEMBLY_TOLERANCES_PX:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        positive_pose_success = 0
        for row in records:
            accepted = float(row["pair_probability"]) >= pair_threshold
            geometry = row.get("primary_geometry")
            pose = bool(geometry and geometry.get("pose_valid"))
            if bool(row["label"]):
                error_value = geometry.get("translation_error_px") if geometry else None
                within = pose and error_value is not None and float(error_value) <= tolerance
                if within:
                    positive_pose_success += 1
                if accepted and within:
                    true_positive += 1
                else:
                    false_negative += 1
                    if accepted and pose:
                        false_positive += 1
            elif accepted and pose:
                false_positive += 1
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        key = "{}px".format(int(tolerance))
        assembly[key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
        }
        positive_translation_success[key] = _safe_ratio(
            positive_pose_success, positive_count
        )

    compatibility_ermse = np.asarray(
        [float(row["primary_geometry"]["pairingnet_e_rmse"]) for row in positive_rows]
    )
    compatibility_hd = np.asarray(
        [float(row["primary_geometry"]["pairingnet_hd_px"]) for row in positive_rows]
    )
    compatibility_nte = np.asarray(
        [float(row["primary_geometry"]["pairingnet_nte"]) for row in positive_rows]
    )
    if not all(
        np.all(np.isfinite(value))
        for value in (compatibility_ermse, compatibility_hd, compatibility_nte)
    ):
        raise PairingNetBenchmarkError("positive compatibility geometry is non-finite")

    report: Dict[str, object] = {
        "record_count": len(records),
        "positive_count": positive_count,
        "negative_count": len(records) - positive_count,
        "pair_decision_secondary": {
            "threshold": pair_threshold,
            "auroc": weighted_pair["row"]["auroc"],
            "auprc": weighted_pair["row"]["auprc"],
            "precision": pair_precision,
            "recall": pair_recall,
            "f1": pair_f1,
            "row": weighted_pair["row"],
            "cluster_balanced": weighted_pair["cluster_balanced"],
            "cluster_count": weighted_pair["cluster_count"],
            "weighting": "equal_lineage_or_lineage_pair_cluster",
        },
        "primary_translation_only": {
            "orientation_condition": "upright_known_no_rotation_estimated",
            "rotation_error": {
                "status": "not_applicable",
                "reason": "orientation_is_conditioned_and_not_a_model_output",
            },
            "pose_coverage": _safe_ratio(int(pose_valid.sum()), positive_count),
            "pose_valid_count": int(pose_valid.sum()),
            "positive_count": positive_count,
            "te_px_conditioned_on_pose_valid": {
                "count": len(conditioned_te),
                "mean": float(conditioned_te.mean()) if len(conditioned_te) else None,
                "median": float(np.median(conditioned_te)) if len(conditioned_te) else None,
                "p95": float(np.quantile(conditioned_te, 0.95))
                if len(conditioned_te)
                else None,
            },
            "positive_translation_success_unconditional": positive_translation_success,
            "assembly_edge": assembly,
        },
        "pairingnet_compatible_secondary": {
            "rr_threshold_definition": "released_e_rmse_lt_4",
            "rr": float(np.mean(compatibility_ermse < 4.0)),
            "hd_px_mean": float(compatibility_hd.mean()),
            "nte_mean": float(compatibility_nte.mean()),
            "invalid_pose_fallback": "identity_translation_for_unconditional_compatibility",
            "re_primary": {
                "status": "not_applicable",
                "reason": "translation_only_orientation_conditioned",
            },
        },
    }

    secondary_rows = [
        row["secondary_se2"]
        for row in positive_rows
        if row.get("secondary_se2") is not None
    ]
    if secondary_rows:
        valid_secondary = [row for row in secondary_rows if row["pose_valid"]]
        report["official_style_se2_ransac_diagnostic"] = {
            "solver": "opencv_estimateAffinePartial2D_not_official_custom_ransac",
            "positive_count": positive_count,
            "pose_valid_count": len(valid_secondary),
            "pose_coverage": _safe_ratio(len(valid_secondary), positive_count),
            "re_radians_mean_conditioned": float(
                np.mean([row["re_radians_conditioned"] for row in valid_secondary])
            )
            if valid_secondary
            else None,
            "translation_error_px_mean_conditioned": float(
                np.mean(
                    [
                        row["translation_error_px_conditioned"]
                        for row in valid_secondary
                    ]
                )
            )
            if valid_secondary
            else None,
        }
    return report


def selection_key(report: Mapping[str, object], epoch: int, metric: str = "official") -> Tuple[float, ...]:
    if metric == "recall95_precision":
        return tuple(float(x) for x in report["recall_operating_points"]["selection_key"]) + (-float(epoch),)
    if metric != "official":
        raise ValueError("unknown selection metric")
    primary = report["primary_translation_only"]
    te = primary["te_px_conditioned_on_pose_valid"]
    median = te["median"]
    rr = report["pairingnet_compatible_secondary"]["rr"]
    translation_success = primary["positive_translation_success_unconditional"][
        "8px"
    ]
    pair = report["pair_decision_secondary"]["cluster_balanced"]
    auroc = float(pair["auroc"])
    auprc = float(pair["auprc"])
    joint = float(
        max(auroc, 0.0)
        * max(auprc, 0.0)
        * max(float(translation_success), 0.0)
        * max(float(rr), 0.0)
    ) ** 0.25
    return (
        joint,
        float(translation_success),
        float(rr),
        auroc,
        auprc,
        -float(median) if median is not None else -1e30,
        -float(epoch),
    )


def is_qualifying_improvement(
    best_primary: float, current_primary: float, relative_improvement: float
) -> bool:
    if not all(
        math.isfinite(value) for value in (current_primary, relative_improvement)
    ) or relative_improvement <= 0.0:
        raise ValueError("invalid improvement inputs")
    if not math.isfinite(best_primary):
        return True
    if best_primary <= 0.0:
        return current_primary > best_primary
    return current_primary >= best_primary * (1.0 + relative_improvement)


def frozen_inference_contract(
    validation_threshold: Optional[float] = None,
    threshold_artifact_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Machine-readable handoff contract for later sealed evaluators."""

    adapter_source_sha256, adaptation_contract_sha256 = _adapter_identity_hashes()
    return {
        "schema_version": "rachel-pairingnet-frozen-inference-contract/1.0",
        "method_id": METHOD_ID,
        "checkpoint_restore": {
            "function": "load_frozen_pairingnet_checkpoint",
            "strict_state_dict": True,
            "strict_formal_n512_config": True,
            "required_checkpoint_schema": SCHEMA_VERSION,
            "required_checkpoint_kind": WINNER_CHECKPOINT_KIND,
            "required_official_commit": OFFICIAL_COMMIT,
            "required_adapter_source_sha256": adapter_source_sha256,
            "required_adaptation_contract_sha256": adaptation_contract_sha256,
            "rejects_resumable_last_checkpoint": True,
        },
        "batch_inference": {
            "function": "infer_pairingnet_batch",
            "input": "RachelBatch(mask-only,N<=512)",
            "outputs": {
                "pair_probability": "float32[B]",
                "translation_a_to_b_rc": "float32[B,2], NaN when invalid",
                "translation_valid": "bool[B]",
                "correspondence_indices": "tuple of int64[K_i,2]",
                "correspondence_scores": "tuple of float32[K_i]",
            },
        },
        "orientation": "upright_known_translation_only_primary",
        "pair_decision": {
            "primary": "validation_fitted_maximize_cluster_balanced_f1",
            "threshold": validation_threshold,
            "threshold_artifact_sha256": threshold_artifact_sha256,
            "artifact_file": "validation_threshold.json",
            "artifact_loader": "load_frozen_validation_threshold",
            "status": "frozen"
            if validation_threshold is not None
            else "pending_winner_validation_fit",
            "adapter_native_secondary_threshold": PAIR_THRESHOLD,
        },
        "correspondence_threshold": 0.006,
        "sealed_synthetic_accessed": False,
        "real_data_accessed": False,
    }


def load_frozen_validation_threshold(
    path: Path, winner_checkpoint_path: Path
) -> float:
    """Restore a threshold only when its artifact and winner binding verify."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairingNetBenchmarkError("cannot read frozen threshold artifact") from error
    if not isinstance(value, Mapping):
        raise PairingNetBenchmarkError("frozen threshold artifact is not an object")
    artifact = value.get("artifact")
    if (
        value.get("schema_version")
        != "rachel-pairingnet-validation-threshold/1.0"
        or value.get("method_id") != METHOD_ID
        or not isinstance(artifact, Mapping)
        or artifact.get("fit_method") != "maximize_cluster_balanced_f1"
        or artifact.get("source_split") != "val"
    ):
        raise PairingNetBenchmarkError("frozen threshold identity mismatch")
    content_sha256 = hashlib.sha256(_canonical_bytes(artifact)).hexdigest()
    if (
        value.get("artifact_content_sha256") != content_sha256
        or value.get("winner_checkpoint_sha256")
        != _sha256_file(Path(winner_checkpoint_path))
        or artifact.get("checkpoint_sha256")
        != value.get("winner_checkpoint_sha256")
        or artifact.get("validation_fingerprint_sha256")
        != value.get("score_order_sha256")
    ):
        raise PairingNetBenchmarkError("frozen threshold binding mismatch")
    threshold = artifact.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise PairingNetBenchmarkError("frozen threshold value is invalid")
    return float(threshold)


def _batch_tensors(
    batch: RachelBatch, device: torch.device
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    for valid in (batch.contour_valid_a, batch.contour_valid_b):
        lengths = np.count_nonzero(valid, axis=1)
        expected = np.arange(valid.shape[1])[None, :] < lengths[:, None]
        if not np.array_equal(valid, expected):
            raise PairingNetBenchmarkError(
                "ordered contour validity must be one contiguous prefix"
            )
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
    )
    return inputs, targets


def _thresholded_correspondences(
    similarity: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    length_a = int(np.count_nonzero(valid_a))
    length_b = int(np.count_nonzero(valid_b))
    enhanced = pairingnet_diagonal_morphology(
        np.asarray(similarity[:length_a, :length_b], dtype=np.float32)
    )
    source, target = np.where(enhanced > epsilon)
    indices = np.stack((source, target), axis=1).astype(np.int64, copy=False)
    scores = enhanced[source, target].astype(np.float32, copy=False)
    return indices, scores


def infer_pairingnet_batch(
    model: PairingNetRachelAdapted,
    batch: RachelBatch,
    device: torch.device,
    *,
    precision: str = "fp32",
) -> Dict[str, object]:
    """Emit the fixed score/translation/correspondence interface."""

    inputs, _ = _batch_tensors(batch, device)
    enabled_bf16 = precision == "bf16"
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled_bf16,
    ):
        output = model(*inputs)
    probability = output.pair_probability.detach().float().cpu().numpy()
    similarity = output.similarity.detach().float().cpu().numpy()
    batch_size = len(batch.pair_ids)
    translation = np.full((batch_size, 2), np.nan, dtype=np.float32)
    translation_valid = np.zeros(batch_size, dtype=np.bool_)
    correspondence_indices = []
    correspondence_scores = []
    estimates = []
    for index in range(batch_size):
        indices, scores = _thresholded_correspondences(
            similarity[index],
            batch.contour_valid_a[index],
            batch.contour_valid_b[index],
            model.config.inference_epsilon,
        )
        estimate = translation_only_consensus(
            similarity[index],
            batch.points_rc_a[index],
            batch.points_rc_b[index],
            batch.contour_valid_a[index],
            batch.contour_valid_b[index],
            model.config,
        )
        if estimate.valid and estimate.translation_rc is not None:
            translation[index] = estimate.translation_rc
            translation_valid[index] = True
        correspondence_indices.append(indices)
        correspondence_scores.append(scores)
        estimates.append(estimate)
    return {
        "schema_version": "rachel-pairingnet-batch-inference/1.0",
        "method_id": METHOD_ID,
        "pair_ids": tuple(batch.pair_ids),
        "pair_probability": probability.astype(np.float32, copy=False),
        "translation_a_to_b_rc": translation,
        "translation_valid": translation_valid,
        "correspondence_indices": tuple(correspondence_indices),
        "correspondence_scores": tuple(correspondence_scores),
        "_translation_estimates": tuple(estimates),
        "_similarity": similarity,
    }


def load_frozen_pairingnet_checkpoint(
    path: Path,
    device: torch.device,
    *,
    require_formal_config: bool = True,
) -> PairingNetRachelAdapted:
    checkpoint_path = _require_guarded_regular_file(Path(path), "winner checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise PairingNetBenchmarkError("checkpoint is not a mapping")
    adapter_source_sha256, adaptation_contract_sha256 = _adapter_identity_hashes()
    if (
        checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("checkpoint_kind") != WINNER_CHECKPOINT_KIND
        or checkpoint.get("method_id") != METHOD_ID
        or checkpoint.get("official_commit") != OFFICIAL_COMMIT
        or checkpoint.get("adapter_source_sha256") != adapter_source_sha256
        or checkpoint.get("adaptation_contract_sha256")
        != adaptation_contract_sha256
    ):
        raise PairingNetBenchmarkError("checkpoint identity mismatch")
    model_value = checkpoint.get("model_config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(model_value, Mapping) or not isinstance(state, Mapping):
        raise PairingNetBenchmarkError("checkpoint lacks model config/state")
    if require_formal_config and model_value != asdict(PairingNetRachelModelConfig()):
        raise PairingNetBenchmarkError("checkpoint is not the frozen formal N512 config")
    model = PairingNetRachelAdapted(
        PairingNetRachelModelConfig(**model_value)
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _validation_records(
    *,
    model: PairingNetRachelAdapted,
    loader: Iterable[RachelBatch],
    manifest_by_id: Mapping[str, ManifestRow],
    device: torch.device,
    precision: str,
    include_secondary_se2: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    model.eval()
    records: List[Dict[str, object]] = []
    started = time.perf_counter()
    for batch in loader:
        inference = infer_pairingnet_batch(
            model, batch, device, precision=precision
        )
        probability = inference["pair_probability"]
        estimates = inference["_translation_estimates"]
        similarity = inference["_similarity"]
        for index, pair_id in enumerate(batch.pair_ids):
            manifest_row = manifest_by_id.get(pair_id)
            if manifest_row is None:
                raise PairingNetBenchmarkError(
                    "validation pair is absent from audited manifest"
                )
            label = bool(batch.labels[index])
            if label != manifest_row.label:
                raise PairingNetBenchmarkError(
                    "loader label disagrees with audited manifest"
                )
            estimate = estimates[index]
            if label:
                primary = _primary_geometry_record(
                    estimate=estimate,
                    points_rc_a=batch.points_rc_a[index],
                    points_rc_b=batch.points_rc_b[index],
                    valid_a=batch.contour_valid_a[index],
                    valid_b=batch.contour_valid_b[index],
                    target_a=batch.target_a[index],
                    ground_truth_translation_rc=batch.translation_a_to_b_rc[index],
                )
                secondary = None
                if include_secondary_se2:
                    secondary_estimate = official_style_se2_ransac(
                        similarity[index],
                        batch.points_rc_a[index],
                        batch.points_rc_b[index],
                        batch.contour_valid_a[index],
                        batch.contour_valid_b[index],
                        model.config,
                    )
                    secondary = _secondary_geometry_record(
                        estimate=secondary_estimate,
                        points_rc_a=batch.points_rc_a[index],
                        points_rc_b=batch.points_rc_b[index],
                        target_a=batch.target_a[index],
                        ground_truth_translation_rc=batch.translation_a_to_b_rc[index],
                    )
            else:
                primary = {
                    "pose_valid": bool(estimate.valid),
                    "translation_rc": estimate.translation_rc.tolist()
                    if estimate.valid and estimate.translation_rc is not None
                    else None,
                    "translation_error_px": None,
                    "candidate_count": estimate.candidate_count,
                    "inlier_count": estimate.inlier_count,
                }
                secondary = None
            records.append(
                {
                    "pair_id": pair_id,
                    "label": label,
                    "cluster_id": manifest_row.cluster_id,
                    "pair_probability": float(probability[index]),
                    "primary_geometry": primary,
                    "secondary_se2": secondary,
                }
            )
    report = summarize_validation_records(records)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["method_id"] = METHOD_ID
    report["secondary_se2_executed"] = include_secondary_se2
    return report, records


def _fit_frozen_validation_threshold(
    records: Sequence[Mapping[str, object]],
    *,
    winner_checkpoint_path: Path,
    model_config: PairingNetRachelModelConfig,
) -> Dict[str, object]:
    """Use the common Rachel cluster-balanced validation threshold fitter."""

    ordered_scores = [
        {
            "pair_id": str(row["pair_id"]),
            "label": bool(row["label"]),
            "cluster_id": str(row["cluster_id"]),
            "pair_probability": float(row["pair_probability"]),
            "valid": True,
        }
        for row in records
    ]
    validation_fingerprint = hashlib.sha256(
        _canonical_bytes(ordered_scores)
    ).hexdigest()
    aggregation_config = {
        "pair_score": "joint_binary_pair_head_probability",
        "validity": "all_finite_validation_scores",
        "cluster_id": "equal_lineage_or_lineage_pair_cluster",
        "objective": "maximize_cluster_balanced_f1",
        "tie_break": "precision_then_recall_then_higher_threshold",
    }
    model_config_sha256 = hashlib.sha256(
        _canonical_bytes(asdict(model_config))
    ).hexdigest()
    aggregation_config_sha256 = hashlib.sha256(
        _canonical_bytes(aggregation_config)
    ).hexdigest()
    checkpoint_sha256 = _sha256_file(winner_checkpoint_path)
    artifact = fit_pairwise_threshold(
        [row["pair_probability"] for row in ordered_scores],
        np.asarray([row["label"] for row in ordered_scores], dtype=np.bool_),
        np.ones(len(ordered_scores), dtype=np.bool_),
        [row["cluster_id"] for row in ordered_scores],
        source_split="val",
        validation_fingerprint_sha256=validation_fingerprint,
        checkpoint_sha256=checkpoint_sha256,
        model_config_sha256=model_config_sha256,
        aggregation_config_sha256=aggregation_config_sha256,
    )
    return {
        "schema_version": "rachel-pairingnet-validation-threshold/1.0",
        "method_id": METHOD_ID,
        "role": "primary_main_table_pair_operating_point",
        "artifact": artifact.to_dict(),
        "artifact_content_sha256": artifact.content_sha256,
        "score_order_sha256": validation_fingerprint,
        "winner_checkpoint_sha256": checkpoint_sha256,
        "model_config_sha256": model_config_sha256,
        "aggregation_config": aggregation_config,
        "aggregation_config_sha256": aggregation_config_sha256,
        "sample_count": len(ordered_scores),
        "sealed_synthetic_accessed": False,
        "real_data_accessed": False,
    }


def _train_epoch(
    *,
    model: PairingNetRachelAdapted,
    optimizer: torch.optim.Optimizer,
    loader: Iterable[RachelBatch],
    device: torch.device,
    config: PairingNetRachelRunConfig,
    epoch: int,
) -> Dict[str, object]:
    model.train()
    total_sum = 0.0
    matching_sum = 0.0
    pair_sum = 0.0
    sample_count = 0
    started = time.perf_counter()
    for step, batch in enumerate(loader, 1):
        inputs, targets = _batch_tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.precision == "bf16",
        ):
            output = model(*inputs)
            loss = compute_pairingnet_loss(
                output,
                targets[0],
                targets[1],
                pair_weight=config.pair_loss_weight,
                matching_weight=config.matching_loss_weight,
                target_b=_tensor(batch.target_b, device, torch.long),
            )
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        count = len(batch.pair_ids)
        total_sum += float(loss.total.detach().float().cpu().item()) * count
        matching_sum += (
            float(loss.matching_focal.detach().float().cpu().item()) * count
        )
        pair_sum += float(loss.pair_bce.detach().float().cpu().item()) * count
        sample_count += count
        if step % config.log_every_steps == 0:
            print(
                json.dumps(
                    {
                        "event": "pairingnet_train_progress",
                        "epoch": epoch,
                        "step": step,
                        "samples": sample_count,
                        "mean_total_loss": total_sum / sample_count,
                        "gradient_norm": float(
                            gradient_norm.detach().float().cpu().item()
                        ),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if sample_count <= 0:
        raise PairingNetBenchmarkError("training epoch had no samples")
    torch.cuda.synchronize(device)
    return {
        "sample_count": sample_count,
        "mean_total_loss": total_sum / sample_count,
        "mean_matching_focal": matching_sum / sample_count,
        "mean_pair_bce": pair_sum / sample_count,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _adaptation_disclosure() -> Dict[str, object]:
    return {
        "claim": "same_data_method_adaptation_not_exact_reproduction",
        "nonofficial_extensions": [
            "joint_binary_pair_head",
            "all_negative_correspondence_supervision_for_negative_pairs",
        ],
        "mask_only": True,
        "rgb_texture_branch_available": False,
        "texture_branch_adaptation": "three_replicated_binary_mask_occupancy_channels",
        "ordered_contour_cap": 512,
        "official_ordered_contour_cap": 2900,
        "primary_pose": "upright_translation_only_consensus",
        "primary_rotation_error": "not_applicable_conditioned_orientation",
        "secondary_pose": "official_style_se2_ransac_not_byte_exact",
        "negative_pair_extension": True,
        "negative_pair_extension_reason": "Rachel target includes pairability",
        "pair_head_is_official_component": False,
        "direct_translation_regression_loss": False,
        "padding_adaptation": "invalid_Rachel_nodes_are_masked_not_encoded_at_origin",
        "dual_softmax_is_sinkhorn": False,
        "paper_matching_batch_size": 20,
        "released_config_matching_batch_size": 25,
        "selected_matching_batch_size": 20,
        "seed_and_epoch_order_frozen": True,
        "bitwise_cuda_determinism_claimed": False,
    }


def _checkpoint_payload(
    *,
    model: PairingNetRachelAdapted,
    epoch: int,
    report: Mapping[str, object],
    run_config: PairingNetRachelRunConfig,
    population_audit: Mapping[str, object],
    official_source_audit: Mapping[str, object],
    adapter_source_sha256: str,
    adaptation_contract_sha256: str,
) -> Dict[str, object]:
    # state_dict() is a shallow view of live parameters.  Clone to CPU because
    # this payload is also retained inside last.pt across later optimizer
    # steps; otherwise the purported winner silently mutates with the model.
    frozen_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": WINNER_CHECKPOINT_KIND,
        "method_id": METHOD_ID,
        "official_commit": OFFICIAL_COMMIT,
        "adapter_source_sha256": adapter_source_sha256,
        "adaptation_contract_sha256": adaptation_contract_sha256,
        "epoch": epoch,
        "model_config": asdict(model.config),
        "model_state_dict": frozen_state,
        "selection_key": list(selection_key(report, epoch, run_config.selection_metric)),
        "run_config": run_config.portable_dict(),
        "manifest_sha256": population_audit["manifests"],
        "official_source_audit": dict(official_source_audit),
        "adaptation": _adaptation_disclosure(),
        "sealed_synthetic_accessed": False,
        "real_data_accessed": False,
    }


def _last_checkpoint_payload(
    *,
    model: PairingNetRachelAdapted,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    completed_epoch: int,
    epoch_rows: Sequence[Mapping[str, object]],
    best_key: Optional[Tuple[float, ...]],
    best_epoch: Optional[int],
    best_primary: float,
    plateau: int,
    winner_checkpoint: Optional[Mapping[str, object]],
    run_config: PairingNetRachelRunConfig,
    population_audit: Mapping[str, object],
    official_source_audit: Mapping[str, object],
    source_sha256: str,
    adaptation_contract_sha256: str,
    resume_count: int,
) -> Dict[str, object]:
    """Create the epoch-granular source of truth for interruption recovery."""

    history = [dict(row) for row in epoch_rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": LAST_CHECKPOINT_KIND,
        "method_id": METHOD_ID,
        "official_commit": OFFICIAL_COMMIT,
        "completed_epoch": completed_epoch,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "optimizer_name": "Adam",
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_name": "CosineAnnealingLR",
        "scheduler_state_dict": scheduler.state_dict(),
        "plateau": plateau,
        "best_key": list(best_key) if best_key is not None else None,
        "best_epoch": best_epoch,
        "best_primary": best_primary,
        # The exact winner is nested so a crash between winner.pt and last.pt
        # cannot leave an uncommitted winner visible after resume.
        "winner_checkpoint": winner_checkpoint,
        "epoch_history": history,
        "epoch_history_sha256": _sha256_json_value(history),
        "run_config": run_config.portable_dict(),
        "manifest_sha256": population_audit["manifests"],
        "official_source_audit": dict(official_source_audit),
        "adapter_source_sha256": source_sha256,
        "adaptation_contract_sha256": adaptation_contract_sha256,
        "rng_state_after_completed_epoch": _capture_rng_state(),
        "resume_count": resume_count,
        "partial_epoch_policy": "discard_and_replay_from_last_completed_epoch",
        "sealed_synthetic_accessed": False,
        "real_data_accessed": False,
    }


def _load_last_checkpoint(
    path: Path,
    *,
    device: torch.device,
    model_config: PairingNetRachelModelConfig,
    run_config: PairingNetRachelRunConfig,
    population_audit: Mapping[str, object],
    official_source_audit: Mapping[str, object],
    source_sha256: str,
    adaptation_contract_sha256: str,
) -> Dict[str, object]:
    """Load and fail closed on every identity needed for strict resume."""

    checkpoint_path = _require_guarded_regular_file(Path(path), "last checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise PairingNetBenchmarkError("last checkpoint is not a mapping")
    identity = (
        checkpoint.get("schema_version") == SCHEMA_VERSION
        and checkpoint.get("checkpoint_kind") == LAST_CHECKPOINT_KIND
        and checkpoint.get("method_id") == METHOD_ID
        and checkpoint.get("official_commit") == OFFICIAL_COMMIT
        and checkpoint.get("model_config") == asdict(model_config)
        and checkpoint.get("run_config") == run_config.portable_dict()
        and checkpoint.get("manifest_sha256") == population_audit["manifests"]
        and checkpoint.get("official_source_audit") == official_source_audit
        and checkpoint.get("adapter_source_sha256") == source_sha256
        and checkpoint.get("adaptation_contract_sha256")
        == adaptation_contract_sha256
    )
    if not identity:
        raise PairingNetBenchmarkError("strict resume identity mismatch")
    completed_epoch = checkpoint.get("completed_epoch")
    history = checkpoint.get("epoch_history")
    history_sha256 = checkpoint.get("epoch_history_sha256")
    if (
        type(completed_epoch) is not int  # noqa: E721
        or not 0 <= completed_epoch <= run_config.max_epochs
        or not isinstance(history, list)
        or history_sha256 != _sha256_json_value(history)
        or len(history) != completed_epoch
    ):
        raise PairingNetBenchmarkError("resume epoch history integrity failed")
    if history and any(
        type(row) is not dict or row.get("epoch") != index  # noqa: E721
        for index, row in enumerate(history, 1)
    ):
        raise PairingNetBenchmarkError("resume epoch history is not contiguous")
    plateau = checkpoint.get("plateau")
    resume_count = checkpoint.get("resume_count")
    best_primary = checkpoint.get("best_primary")
    best_epoch = checkpoint.get("best_epoch")
    best_key = checkpoint.get("best_key")
    winner = checkpoint.get("winner_checkpoint")
    if (
        type(plateau) is not int  # noqa: E721
        or plateau < 0
        or type(resume_count) is not int  # noqa: E721
        or resume_count < 0
        or not isinstance(best_primary, (int, float))
        or (
            completed_epoch > 0
            and not math.isfinite(float(best_primary))
        )
        or checkpoint.get("optimizer_name") != "Adam"
        or checkpoint.get("scheduler_name") != "CosineAnnealingLR"
    ):
        raise PairingNetBenchmarkError("resume stopping state is invalid")
    if completed_epoch == 0:
        if any(value is not None for value in (best_epoch, best_key, winner)):
            raise PairingNetBenchmarkError("epoch-zero resume unexpectedly has a winner")
    else:
        if (
            type(best_epoch) is not int  # noqa: E721
            or not 1 <= best_epoch <= completed_epoch
            or not isinstance(best_key, list)
            or len(best_key) != (3 if run_config.selection_metric == "recall95_precision" else 7)
            or not isinstance(winner, Mapping)
            or winner.get("epoch") != best_epoch
            or winner.get("schema_version") != SCHEMA_VERSION
            or winner.get("checkpoint_kind") != WINNER_CHECKPOINT_KIND
            or winner.get("method_id") != METHOD_ID
            or winner.get("official_commit") != OFFICIAL_COMMIT
            or winner.get("adapter_source_sha256") != source_sha256
            or winner.get("adaptation_contract_sha256")
            != adaptation_contract_sha256
            or winner.get("model_config") != asdict(model_config)
            or winner.get("run_config") != run_config.portable_dict()
            or winner.get("manifest_sha256") != population_audit["manifests"]
            or winner.get("official_source_audit") != official_source_audit
            or winner.get("selection_key") != best_key
            or not isinstance(winner.get("model_state_dict"), Mapping)
        ):
            raise PairingNetBenchmarkError("resume winner state is invalid")
    for name in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state_after_completed_epoch",
    ):
        if checkpoint.get(name) is None:
            raise PairingNetBenchmarkError("resume checkpoint lacks " + name)
    scheduler_state = checkpoint["scheduler_state_dict"]
    if (
        not isinstance(scheduler_state, Mapping)
        or scheduler_state.get("last_epoch") != completed_epoch
    ):
        raise PairingNetBenchmarkError("resume scheduler epoch is inconsistent")
    return dict(checkpoint)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = _reject_symlink_components(path)
    temporary = _reject_symlink_components(path.with_name("." + path.name + ".tmp"))
    with temporary.open("wb") as stream:
        for row in rows:
            stream.write(_canonical_bytes(row) + b"\n")
    os.replace(temporary, path)


def _validate_formal_protocol(
    run_config: PairingNetRachelRunConfig,
    model_config: PairingNetRachelModelConfig,
) -> None:
    if model_config != PairingNetRachelModelConfig():
        raise PairingNetBenchmarkError("formal method identity requires the frozen N512 model config")
    expected = {
        "batch_size": 20,
        "learning_rate": 1e-3,
        "weight_decay": 5e-4,
        "min_epochs": 20,
        "max_epochs": 128,
        "patience": 12,
        "relative_improvement": 0.005,
        "eta_min": 1e-6,
        "gradient_clip_norm": 5.0,
        "pair_loss_weight": 1.0,
        "matching_loss_weight": 1.0,
    }
    if run_config.experimental_data:
        # Dataset size, seed and epoch bounds are explicit experimental knobs;
        # every architecture, loss, optimizer and stopping-rule value stays fixed.
        expected.pop("min_epochs")
        expected.pop("max_epochs")
    if any(getattr(run_config, name) != value for name, value in expected.items()):
        raise PairingNetBenchmarkError("formal training hyperparameters changed")


def run_train_val_benchmark(
    run_config: PairingNetRachelRunConfig,
    model_config: PairingNetRachelModelConfig = PairingNetRachelModelConfig(),
    *,
    resume_from: Optional[Path] = None,
) -> Dict[str, object]:
    """Run the full Rachel train/validation benchmark without test discovery."""

    _validate_formal_protocol(run_config, model_config)
    dataset_root = run_config.dataset_root.resolve(strict=True)
    output_root, partial_root, failed_root, resume_source = (
        _validated_run_state_paths(run_config.output_root, resume_from)
    )
    if not torch.cuda.is_available():
        raise PairingNetBenchmarkError("formal PairingNet benchmark requires CUDA")
    device = torch.device(run_config.device)
    population = audit_train_val_population(dataset_root,
        require_formal_counts=not run_config.experimental_data,
        train_materialized_manifest=run_config.train_materialized_manifest)
    population["seed"] = run_config.seed
    population["training_input_contract"] = ("shared_materialized_TRAIN_original_targets_compact_valid_ignore_minus2"
        if run_config.experimental_data else "original_clean_Rachel_release")
    official_source_audit = audit_official_source(run_config.official_source_root)
    train_manifest = _read_manifest(dataset_root, "train", run_config.train_materialized_manifest)
    val_manifest = _read_manifest(dataset_root, "val")
    if ((not run_config.experimental_data and len(train_manifest) != FORMAL_TRAIN_ROWS)
            or len(val_manifest) != FORMAL_VAL_ROWS):
        raise PairingNetBenchmarkError("formal population changed after audit")
    source_sha256, adaptation_contract_sha256 = _adapter_identity_hashes()

    resume_checkpoint: Optional[Dict[str, object]] = None
    if resume_source is not None:
        config_path = _require_guarded_regular_file(
            resume_source / "run_config.json", "resume run config"
        )
        try:
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PairingNetBenchmarkError("cannot read strict resume run config") from error
        if (
            saved_config.get("schema_version") != SCHEMA_VERSION
            or saved_config.get("method_id") != METHOD_ID
            or saved_config.get("run_config") != run_config.portable_dict()
            or saved_config.get("model_config") != asdict(model_config)
            or saved_config.get("adapter_source_sha256") != source_sha256
            or saved_config.get("adaptation_contract_sha256")
            != adaptation_contract_sha256
        ):
            raise PairingNetBenchmarkError("strict resume run config mismatch")
        resume_checkpoint = _load_last_checkpoint(
            resume_source / "last.pt",
            device=device,
            model_config=model_config,
            run_config=run_config,
            population_audit=population,
            official_source_audit=official_source_audit,
            source_sha256=source_sha256,
            adaptation_contract_sha256=adaptation_contract_sha256,
        )
        if resume_source == failed_root:
            _move_guarded_run_directory(failed_root, partial_root)
    else:
        _create_guarded_run_directory(partial_root)

    try:
        if resume_checkpoint is None:
            _atomic_json(
                partial_root / "run_config.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "running_train_validation_only",
                    "method_id": METHOD_ID,
                    "official_reference": {
                        "repository": OFFICIAL_REPOSITORY,
                        "commit": OFFICIAL_COMMIT,
                        "paper": OFFICIAL_PAPER,
                    },
                    "run_config": run_config.portable_dict(),
                    "model_config": asdict(model_config),
                    "population_audit": population,
                    "official_source_audit": official_source_audit,
                    "adaptation": _adaptation_disclosure(),
                    "inference_contract": frozen_inference_contract(),
                    "resume_contract": {
                        "checkpoint": "last.pt",
                        "granularity": "completed_epoch",
                        "partial_epoch_policy": "discard_and_replay_from_last_completed_epoch",
                        "strict_cli": "--resume-from",
                    },
                    "adapter_source_sha256": source_sha256,
                    "adaptation_contract_sha256": adaptation_contract_sha256,
                    "sealed_synthetic_accessed": False,
                    "real_data_accessed": False,
                },
            )

        _set_determinism(run_config.seed)
        dataset_config = RachelDatasetConfig(
            mask_size=model_config.canvas_size,
            contour_cap=model_config.contour_cap,
        )
        train_dataset = (materialized_train_dataset(run_config.train_materialized_manifest)
            if run_config.train_materialized_manifest is not None
            else RachelPairDataset(dataset_root, "train", dataset_config))
        val_dataset = RachelPairDataset(dataset_root, "val", dataset_config)
        if len(train_dataset) != len(train_manifest) or len(val_dataset) != len(val_manifest):
            raise PairingNetBenchmarkError("loader/manifest population mismatch")
        val_order = tuple(range(len(val_dataset)))
        val_by_id = {row.pair_id: row for row in val_manifest}
        if len(val_by_id) != len(val_manifest):
            raise PairingNetBenchmarkError("validation pair IDs are duplicated")
        val_loader = _loader(
            val_dataset,
            val_order,
            batch_size=run_config.batch_size,
            num_workers=run_config.num_workers,
            seed=run_config.seed + 77,
        )

        _set_determinism(run_config.seed)
        model = PairingNetRachelAdapted(model_config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=run_config.learning_rate,
            weight_decay=run_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=run_config.max_epochs, eta_min=run_config.eta_min
        )
        epoch_rows: List[Dict[str, object]] = []
        best_key: Optional[Tuple[float, ...]] = None
        best_epoch: Optional[int] = None
        best_primary = -math.inf
        plateau = 0
        completed_epoch = 0
        resume_count = 0
        winner_payload: Optional[Mapping[str, object]] = None
        checkpoint_path = partial_root / "winner.pt"
        last_checkpoint_path = partial_root / "last.pt"

        if resume_checkpoint is not None:
            model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            completed_epoch = int(resume_checkpoint["completed_epoch"])
            epoch_rows = [dict(row) for row in resume_checkpoint["epoch_history"]]
            best_epoch_value = resume_checkpoint["best_epoch"]
            best_epoch = int(best_epoch_value) if best_epoch_value is not None else None
            best_key_value = resume_checkpoint["best_key"]
            best_key = (
                tuple(float(value) for value in best_key_value)
                if best_key_value is not None
                else None
            )
            best_primary = float(resume_checkpoint["best_primary"])
            plateau = int(resume_checkpoint["plateau"])
            winner_payload = resume_checkpoint["winner_checkpoint"]
            resume_count = int(resume_checkpoint["resume_count"]) + 1
            _restore_rng_state(resume_checkpoint["rng_state_after_completed_epoch"])
            if winner_payload is not None:
                _atomic_torch_save(checkpoint_path, winner_payload)
            _atomic_json(partial_root / "epochs.json", epoch_rows)
            _atomic_json(
                partial_root / "resume_state.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "method_id": METHOD_ID,
                    "resume_count": resume_count,
                    "resumed_from_completed_epoch": completed_epoch,
                    "next_epoch": completed_epoch + 1,
                    "partial_epoch_policy": "discard_and_replay_from_last_completed_epoch",
                },
            )

        _atomic_torch_save(
            last_checkpoint_path,
            _last_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                completed_epoch=completed_epoch,
                epoch_rows=epoch_rows,
                best_key=best_key,
                best_epoch=best_epoch,
                best_primary=best_primary,
                plateau=plateau,
                winner_checkpoint=winner_payload,
                run_config=run_config,
                population_audit=population,
                official_source_audit=official_source_audit,
                source_sha256=source_sha256,
                adaptation_contract_sha256=adaptation_contract_sha256,
                resume_count=resume_count,
            ),
        )

        stop_reason = "hard_cap_reached"
        if completed_epoch >= run_config.min_epochs and plateau >= run_config.patience:
            stop_reason = "validation_plateau"
        else:
            for epoch in range(completed_epoch + 1, run_config.max_epochs + 1):
                order = epoch_indices(len(train_dataset), run_config.seed, epoch, None)
                train_loader = _loader(
                    train_dataset,
                    order,
                    batch_size=run_config.batch_size,
                    num_workers=run_config.num_workers,
                    seed=run_config.seed + epoch,
                )
                learning_rate_used = float(optimizer.param_groups[0]["lr"])
                train_report = _train_epoch(
                    model=model,
                    optimizer=optimizer,
                    loader=train_loader,
                    device=device,
                    config=run_config,
                    epoch=epoch,
                )
                validation_report, validation_records = _validation_records(
                    model=model,
                    loader=val_loader,
                    manifest_by_id=val_by_id,
                    device=device,
                    precision=run_config.precision,
                    include_secondary_se2=False,
                )
                if run_config.selection_metric == "recall95_precision":
                    from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
                    validation_report["recall_operating_points"] = fit_operating_points(
                        [row["label"] for row in validation_records],
                        [row["pair_probability"] for row in validation_records])
                key = selection_key(validation_report, epoch, run_config.selection_metric)
                primary = float(key[0])
                qualifying = is_qualifying_improvement(
                    best_primary, primary, run_config.relative_improvement
                )
                if qualifying:
                    best_primary = primary
                    plateau = 0
                else:
                    plateau += 1
                winner_updated = best_key is None or key > best_key
                if winner_updated:
                    best_key = key
                    best_epoch = epoch
                    winner_payload = _checkpoint_payload(
                        model=model,
                        epoch=epoch,
                        report=validation_report,
                        run_config=run_config,
                        population_audit=population,
                        official_source_audit=official_source_audit,
                        adapter_source_sha256=source_sha256,
                        adaptation_contract_sha256=adaptation_contract_sha256,
                    )
                    _atomic_torch_save(checkpoint_path, winner_payload)
                epoch_row = {
                    "epoch": epoch,
                    "learning_rate_used": learning_rate_used,
                    "train": train_report,
                    "validation": validation_report,
                    "selection_key": list(key),
                    "qualifying_relative_improvement": qualifying,
                    "winner_updated": winner_updated,
                    "epochs_without_qualifying_improvement": plateau,
                }
                epoch_rows.append(epoch_row)
                scheduler.step()
                completed_epoch = epoch
                # last.pt is the atomic commit point.  epochs.json and
                # winner.pt are regenerated from it during strict resume.
                _atomic_torch_save(
                    last_checkpoint_path,
                    _last_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        completed_epoch=completed_epoch,
                        epoch_rows=epoch_rows,
                        best_key=best_key,
                        best_epoch=best_epoch,
                        best_primary=best_primary,
                        plateau=plateau,
                        winner_checkpoint=winner_payload,
                        run_config=run_config,
                        population_audit=population,
                        official_source_audit=official_source_audit,
                        source_sha256=source_sha256,
                        adaptation_contract_sha256=adaptation_contract_sha256,
                        resume_count=resume_count,
                    ),
                )
                _atomic_json(partial_root / "epochs.json", epoch_rows)
                print(
                    json.dumps(
                        {
                            "event": "pairingnet_epoch_complete",
                            "epoch": epoch,
                            "threshold_free_selection_primary": primary,
                            "rr": validation_report[
                                "pairingnet_compatible_secondary"
                            ]["rr"],
                            "plateau": plateau,
                            "winner_epoch": best_epoch,
                            "resumable_checkpoint": "last.pt",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if epoch >= run_config.min_epochs and plateau >= run_config.patience:
                    stop_reason = "validation_plateau"
                    break

        if best_epoch is None or best_key is None or not checkpoint_path.is_file():
            raise PairingNetBenchmarkError("training produced no winner checkpoint")
        winner = load_frozen_pairingnet_checkpoint(checkpoint_path, device)
        native_report, final_records = _validation_records(
            model=winner,
            loader=val_loader,
            manifest_by_id=val_by_id,
            device=device,
            precision=run_config.precision,
            include_secondary_se2=True,
        )
        if run_config.selection_metric == "recall95_precision":
            from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
            native_report["recall_operating_points"] = fit_operating_points(
                [row["label"] for row in final_records], [row["pair_probability"] for row in final_records])
        if selection_key(native_report, best_epoch, run_config.selection_metric)[:-1] != best_key[:-1]:
            raise PairingNetBenchmarkError(
                "restored winner metrics disagree with saved checkpoint"
            )
        threshold = _fit_frozen_validation_threshold(
            final_records,
            winner_checkpoint_path=checkpoint_path,
            model_config=model_config,
        )
        threshold_value = float(threshold["artifact"]["threshold"])
        primary_report = summarize_validation_records(
            final_records, pair_threshold=threshold_value
        )
        primary_report["method_id"] = METHOD_ID
        primary_report["secondary_se2_executed"] = True
        primary_report["pair_operating_point"] = {
            "role": "primary_main_table",
            "source": "validation_fitted_maximize_cluster_balanced_f1",
            "threshold": threshold_value,
            "threshold_artifact_content_sha256": threshold[
                "artifact_content_sha256"
            ],
        }
        native_report["pair_operating_point"] = {
            "role": "adapter_native_secondary_not_official_pairingnet_pair_head",
            "threshold": PAIR_THRESHOLD,
        }
        for row in final_records:
            score = float(row["pair_probability"])
            row["decision_at_frozen_validation_threshold"] = score >= threshold_value
            row["decision_at_adapter_native_0_5"] = score >= PAIR_THRESHOLD
        final_report = {
            "schema_version": "rachel-pairingnet-winner-validation/1.0",
            "method_id": METHOD_ID,
            "selection_threshold_used": run_config.selection_metric == "recall95_precision",
            "selection_key": list(best_key),
            "selection_metric": run_config.selection_metric,
            "recall_operating_points": native_report.get("recall_operating_points"),
            "primary_frozen_validation_threshold": primary_report,
            "adapter_native_0_5_secondary": native_report,
            "validation_threshold": threshold,
            "sealed_synthetic_accessed": False,
            "real_data_accessed": False,
        }
        _atomic_json(partial_root / "validation_threshold.json", threshold)
        _atomic_json(partial_root / "winner_validation_report.json", final_report)
        _write_jsonl(partial_root / "winner_validation_predictions.jsonl", final_records)
        final_inference_contract = frozen_inference_contract(
            threshold_value,
            str(threshold["artifact_content_sha256"]),
        )
        _atomic_json(partial_root / "inference_contract.json", final_inference_contract)
        last_checkpoint = _load_last_checkpoint(
            last_checkpoint_path,
            device=device,
            model_config=model_config,
            run_config=run_config,
            population_audit=population,
            official_source_audit=official_source_audit,
            source_sha256=source_sha256,
            adaptation_contract_sha256=adaptation_contract_sha256,
        )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "train_validation_complete",
            "method_id": METHOD_ID,
            "official_commit": OFFICIAL_COMMIT,
            "adaptation_claim": "same_data_method_adaptation_not_exact_reproduction",
            "adaptation": _adaptation_disclosure(),
            "stop_reason": stop_reason,
            "convergence_demonstrated": stop_reason == "validation_plateau",
            "epochs_completed": completed_epoch,
            "winner_epoch": best_epoch,
            "winner_checkpoint_kind": WINNER_CHECKPOINT_KIND,
            "adapter_source_sha256": source_sha256,
            "adaptation_contract_sha256": adaptation_contract_sha256,
            "selection_threshold_used": run_config.selection_metric == "recall95_precision",
            "selection_metric": run_config.selection_metric,
            "pose_used_for_checkpoint_selection": run_config.selection_metric == "official",
            "selection_key": list(best_key),
            "resume_count": resume_count,
            "resume_contract": {
                "checkpoint": "last.pt",
                "granularity": "completed_epoch",
                "partial_epoch_policy": "discard_and_replay_from_last_completed_epoch",
                "optimizer": "Adam",
                "scheduler": "CosineAnnealingLR",
            },
            "epoch_history_sha256": last_checkpoint["epoch_history_sha256"],
            "last_checkpoint_sha256": _sha256_file(last_checkpoint_path),
            "winner_checkpoint_sha256": _sha256_file(checkpoint_path),
            "validation_threshold": threshold,
            "validation_threshold_file_sha256": _sha256_file(
                partial_root / "validation_threshold.json"
            ),
            "inference_contract_sha256": _sha256_file(
                partial_root / "inference_contract.json"
            ),
            "winner_validation_report_sha256": _sha256_file(
                partial_root / "winner_validation_report.json"
            ),
            "winner_validation_predictions_sha256": _sha256_file(
                partial_root / "winner_validation_predictions.jsonl"
            ),
            "population_audit": population,
            "official_source_audit": official_source_audit,
            "sealed_synthetic_accessed": False,
            "real_data_accessed": False,
        }
        _atomic_json(partial_root / "completion_receipt.json", receipt)
        _move_guarded_run_directory(partial_root, output_root)
        return receipt
    except BaseException:
        partial_root = _reject_symlink_components(partial_root)
        failed_root = _reject_symlink_components(failed_root)
        partial_metadata = _lstat_optional(partial_root)
        failed_metadata = _lstat_optional(failed_root)
        if partial_metadata is not None and failed_metadata is None:
            _move_guarded_run_directory(partial_root, failed_root)
        raise


def _parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rachel same-data PairingNet-derived train/val benchmark"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="exact .<output>.partial or .<output>.failed directory to resume",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--train-materialized-manifest", type=Path)
    parser.add_argument("--experimental-data", action="store_true")
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=128)
    parser.add_argument("--selection-metric", choices=("official", "recall95_precision"), default="official")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parse_arguments(argv)
    if arguments.audit_only:
        if arguments.experimental_data != (arguments.train_materialized_manifest is not None):
            raise SystemExit("experimental data requires --train-materialized-manifest and --experimental-data together")
        if arguments.resume_from is not None:
            raise SystemExit("--resume-from cannot be combined with --audit-only")
        report = {
            "schema_version": "rachel-pairingnet-preflight-audit/1.0",
            "population_audit": audit_train_val_population(
                arguments.dataset_root, require_formal_counts=not arguments.experimental_data,
                train_materialized_manifest=arguments.train_materialized_manifest
            ),
            "official_source_audit": audit_official_source(
                arguments.official_source_root
            ),
        }
        print(json.dumps(report, sort_keys=True))
        return 0
    if arguments.output_root is None:
        raise SystemExit("--output-root is required unless --audit-only is used")
    receipt = run_train_val_benchmark(
        PairingNetRachelRunConfig(
            dataset_root=arguments.dataset_root,
            output_root=arguments.output_root,
            official_source_root=arguments.official_source_root,
            device=arguments.device,
            precision=arguments.precision,
            num_workers=arguments.num_workers,
            log_every_steps=arguments.log_every_steps,
            train_materialized_manifest=arguments.train_materialized_manifest,
            experimental_data=arguments.experimental_data,
            seed=arguments.seed,
            min_epochs=arguments.min_epochs,
            max_epochs=arguments.max_epochs,
            selection_metric=arguments.selection_metric,
        ),
        resume_from=arguments.resume_from,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

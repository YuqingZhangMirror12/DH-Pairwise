"""Train/validation-only Rachel adaptation of the official ShreddingNet release.

This module deliberately has no sealed-test or real-data entry point.  The
official checkout is treated as source authority, not imported at runtime: its
released implementation contains cwd-relative imports, hard-coded ``.cuda()``
calls, and dependencies that make direct composition unsafe.  The architecture
port below is hash-bound to that checkout and records every Rachel adaptation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import stat
import subprocess
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import (
    RachelBatch,
    RachelPairDataset,
)
from staging.pairwise_v0_2.baselines.rachel_materialized_training import (
    benchmark_manifest_lines, selected_manifest_path, materialized_train_dataset,
    training_hook_identity,
    supervised_contour_masks, collate_benchmark_pairs as collate_rachel_pairs,
)
from staging.pairwise_v0_2.training.evaluation import (
    PairwiseThresholdArtifact,
    evaluate_pairwise,
    fit_pairwise_threshold,
)


SCHEMA_VERSION = "rachel-shreddingnet-maskonly-benchmark/1.0"
FREEZE_SCHEMA_VERSION = "rachel-shreddingnet-train-val-freeze/1.0"
METHOD_ID = "rachel_shreddingnet_maskonly_n512_upright_release0ae3b544_v1"
OFFICIAL_COMMIT = "0ae3b544ca4e910732f3f459b39aa15cdc62dbcb"
PRIMARY_TRANSLATION_TOLERANCES_PX = (2.0, 5.0, 8.0, 10.0)
OFFICIAL_SECONDARY_ROTATION_DEG = 5.0
OFFICIAL_SECONDARY_TRANSLATION_PX = 100.0
MAX_POSE_CORRESPONDENCES = 8192
_CHECKPOINT_FLOAT_SENTINEL = "__rachel_shreddingnet_float64_tensor__"
ADAPTATION_DOC_FILENAME = "RACHEL_SHREDDINGNET_BENCHMARK.md"
FREEZE_CHECKPOINT_KIND = "rachel_shreddingnet_train_val_freeze"
COARSE_VALIDATION_EPOCH_SEED_MULTIPLIER = 2_000_033


def _stage_checkpoint_kind(stage: str, artifact: str) -> str:
    if stage not in {"coarse", "matching", "classify"} or artifact not in {
        "progress",
        "winner",
    }:
        raise ValueError("invalid stage checkpoint kind")
    return "rachel_shreddingnet_{}_{}".format(stage, artifact)

OFFICIAL_FILE_SHA256 = {
    "config/art_2192/pairing.yaml": "88bd381829fe4d0106504f7a657d5ee0d9670e5f1107146556d1491445d9efa2",
    "config/art_2192/matching.yaml": "4edeee07105b1cf2121b32eceb39afda8f2d33d4a6edf001c13d4227d8c1b1c8",
    "config/art_2192/classify.yaml": "87acb8bcae5ac68cbc34ea429220227d03ac20361e6eb6486ddfdc3725e2fdad",
    "config/default.py": "519f0067ff92069b7776462aee6242be3c4a259828e08269e024070117b51132",
    "nets/feature_extract.py": "3fb17f2473e4dc0eb3f797bbae1142f5476988b751898481012c50969d25800e",
    "nets/feature_fuse.py": "5f2808380f3edc8afc2d741e1b65cb9ff3f039bff2d50e4ac2ed922f237a64ed",
    "nets/matching.py": "462010262e82594f5baabde55b00cde386c0f7fdc222ac749bf9c5797845e2ec",
    "nets/pairing.py": "fcd10e99ef1c48e7d0bc00d544df1da50d7f0f27161d8a4e352591bc339da19b",
    "nets/score_evaluators.py": "21693f47ea448a1527696efcf5ae3fbc0560783f77adbdd43578005e104303b4",
    "nets/utils/focal_loss.py": "03de200b526c92899e2fb75ac5cfc360b799d4316b332dfb87028dcc6e9886a8",
    "nets/utils/infornce_loss.py": "6b48d6281b93f7c0364ff8a7e7367ebe3df8483542f7d401961ced300e97a6df",
    "nets/utils/gcn.py": "d75b408d0fd3df40c2030293df7d0e78bdf16968fc16a47717df8f1a43aaf9d4",
    "dataset/base_dataset.py": "0088c25587d4183ac0bce9dbb4f889aeac4d27bba9b1322ead604ea06b2eb23a",
    "dataset/pairing_dataset.py": "d8b851fae2aa517de1bc14fb200f6505b8414fabaffde211d3a6b53e8ea8fbd3",
    "dataset/matching_dataset.py": "f1f47525b032b23efe97510c6984021b69dc07c0fdd056ec5ebae4d8e4e04f6d",
    "dataset/classify_dataset.py": "4a5e2d65527ca0ea192f91bcc5bb6be549ad15e89a30a03d52d620da91a3c795",
    "dataset/scripts/encoders.py": "ac8bdf639a0ad5281c8b4a9c3fbac5298d0bd5f9dec09b4cc715927609d04f6b",
    "train.py": "5f812ec02e29bf17c34fdcd809465eada0134194185302a33957c3f40439b5c8",
    "trainers/base_trainers.py": "bcd19c07c49f2a6a382b10f02c08bb2c17234fa484ef58d1e8a613dde74e50d6",
    "trainers/classify_trainers.py": "dfa0448aa90cc43bf3a345e026ca5f7324df250287bad5d5338d31f9bc2f6f9b",
    "trainers/matching_trainers.py": "37b8285ede3a9e2e82520ca17a53e97ac18a0f261ce27d2af060d313f4ac94f9",
    "trainers/pairing_trainers.py": "421c3e4bc83bec1b8f9896e91c449dc49cd1e973f0145c3a1dbfc98ceaa17426",
    "metrics/metrics_tools.py": "5e47f7145bbfd88d95eed562174a9a1ef80a3bbe1bd89ab71d6f5d39cea20e6b",
    "visualize/ransac.py": "75c615abf69fa32f39bc3fed408e5fc881825f4533499cb53b7562e6720ea260",
}


class BenchmarkContractError(RuntimeError):
    """The official source, Rachel split, or frozen bundle violates contract."""


@dataclass(frozen=True)
class ReleaseRecipe:
    """Executable art_2192 recipe frozen by the official release YAML."""

    # Same-exposure Rachel comparison seed.  The release seed remains disclosed
    # separately and is not silently used for the primary benchmark.
    seed: int = 260831
    official_release_seed: int = 1024
    contour_cap: int = 512
    patch_size: int = 7
    feature_dim: int = 64
    resgcn_blocks: int = 14
    cycle_radius: int = 8
    decoder_blocks: int = 2
    attention_heads: int = 8
    coarse_epochs: int = 128
    matching_epochs: int = 128
    classify_epochs: int = 128
    coarse_batch_size: int = 54
    matching_batch_size: int = 36
    classify_batch_size: int = 20
    coarse_lr: float = 1e-4
    matching_lr: float = 1e-3
    classify_lr: float = 1e-4
    weight_decay: float = 5e-4
    infonce_temperature: float = 0.12
    focal_alpha: float = 0.55
    focal_gamma: float = 8.0
    correspondence_threshold: float = 0.006
    pair_score_threshold: float = 0.5
    official_top_k: int = 20
    rachel_top_k: int = 2
    experimental_data: bool = False
    selection_metric: str = "official"

    def __post_init__(self) -> None:
        if type(self.experimental_data) is not bool:
            raise TypeError("experimental_data must be bool")
        if self.selection_metric not in {"official", "recall95_precision"}:
            raise ValueError("selection_metric must be official or recall95_precision")
        if self.selection_metric != "official" and not self.experimental_data:
            raise ValueError("recall selection requires explicit experimental data")
        integer_names = (
            "seed",
            "official_release_seed",
            "contour_cap",
            "patch_size",
            "feature_dim",
            "resgcn_blocks",
            "cycle_radius",
            "decoder_blocks",
            "attention_heads",
            "coarse_epochs",
            "matching_epochs",
            "classify_epochs",
            "coarse_batch_size",
            "matching_batch_size",
            "classify_batch_size",
            "official_top_k",
            "rachel_top_k",
        )
        for name in integer_names:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        if self.patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        if self.feature_dim % self.attention_heads != 0:
            raise ValueError("feature_dim must be divisible by attention_heads")
        for name in (
            "coarse_lr",
            "matching_lr",
            "classify_lr",
            "weight_decay",
            "infonce_temperature",
            "focal_alpha",
            "focal_gamma",
            "correspondence_threshold",
            "pair_score_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(name + " must be finite and positive")


@dataclass(frozen=True)
class RuntimeBatchConfig:
    """GPU-memory adaptation while retaining release optimizer exposure."""

    coarse_microbatch: int = 1
    matching_microbatch: int = 1
    classify_microbatch: int = 1
    amp: bool = True

    def __post_init__(self) -> None:
        for name in (
            "coarse_microbatch",
            "matching_microbatch",
            "classify_microbatch",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        if type(self.amp) is not bool:  # noqa: E721
            raise TypeError("amp must be bool")

    def microbatch(self, stage: str) -> int:
        if stage not in {"coarse", "matching", "classify"}:
            raise ValueError("unknown stage")
        return int(getattr(self, stage + "_microbatch"))


def _require_executable_release_recipe(recipe: ReleaseRecipe) -> None:
    reference = ReleaseRecipe()
    checked = (replace(recipe, experimental_data=False, selection_metric="official", seed=reference.seed,
        coarse_epochs=reference.coarse_epochs, matching_epochs=reference.matching_epochs,
        classify_epochs=reference.classify_epochs) if recipe.experimental_data else recipe)
    if checked != reference:
        raise BenchmarkContractError(
            "formal runner recipe must equal the frozen release/Rachel contract"
        )


@dataclass(frozen=True)
class PairMetadata:
    split: str
    pair_id: str
    label: bool
    fragment_a_token: str
    fragment_b_token: str
    lineage_a: str
    lineage_b: str

    @property
    def cluster_id(self) -> str:
        """Physical-lineage cluster used by the shared validation fitter."""

        if self.lineage_a == self.lineage_b:
            return "unit:" + self.lineage_a
        payload = json.dumps(
            sorted((self.lineage_a, self.lineage_b)),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return "unit-pair:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TranslationEstimate:
    translation_rc: np.ndarray
    valid: bool
    correspondence_count: int
    inlier_count: int


@dataclass(frozen=True)
class SE2Estimate:
    rotation: np.ndarray
    translation_rc: np.ndarray
    rotation_degrees: float
    valid: bool
    correspondence_count: int
    inlier_count: int


@dataclass(frozen=True)
class FrozenInferenceOutput:
    method_id: str
    pair_ids: Tuple[str, ...]
    coarse_score: np.ndarray
    pair_score: np.ndarray
    translation_hat_rc: np.ndarray
    translation_valid: np.ndarray
    correspondence_count: np.ndarray
    translation_inlier_count: np.ndarray
    se2_translation_hat_rc: np.ndarray
    se2_rotation_degrees: np.ndarray
    se2_valid: np.ndarray
    se2_inlier_count: np.ndarray
    correspondence_probability: Optional[np.ndarray] = None
    correspondence_binary: Optional[np.ndarray] = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_absolute_path(value: Union[str, Path]) -> Path:
    """Return a normalized absolute path without resolving symlinks."""

    expanded = Path(value).expanduser()
    return Path(os.path.abspath(os.fspath(expanded)))


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _assert_no_symlink_components(path: Path, *, name: str) -> None:
    """Use lstat on every existing path component; never follow an alias."""

    absolute = _lexical_absolute_path(path)
    current = Path(absolute.anchor)
    components = absolute.parts[1:]
    for index, part in enumerate(components):
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            # Once a component is absent, descendants cannot exist without a
            # concurrent mutation; the caller rechecks after any mkdir/write.
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise BenchmarkContractError(name + " path contains a symlink: " + str(current))
        if index < len(components) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise BenchmarkContractError(
                name + " ancestor is not a directory: " + str(current)
            )


def _ensure_directory_tree(path: Path, *, name: str) -> None:
    """Create missing directories one component at a time without aliases."""

    absolute = _lexical_absolute_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current)
            except FileExistsError:
                pass
            metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise BenchmarkContractError(
                name + " directory component is unsafe: " + str(current)
            )


def _prepare_fresh_file_path(path: Path, *, name: str) -> Path:
    absolute = _lexical_absolute_path(path)
    _assert_no_symlink_components(absolute, name=name)
    if _path_lexists(absolute):
        raise BenchmarkContractError(name + " already exists; overwrite forbidden")
    return absolute


_TRAINING_ROOT_ALLOWED_ENTRIES = {
    "run_contract.json",
    "stages",
    "validation_threshold_scores.jsonl",
    "train_val_freeze.json",
    "validation_report.json",
}


def _inspect_training_output_root(path: Path) -> str:
    """Classify the lexical training root without following any alias."""

    root = _lexical_absolute_path(path)
    _assert_no_symlink_components(root, name="training output root")
    if not _path_lexists(root):
        return "absent"
    metadata = os.lstat(root)
    if not stat.S_ISDIR(metadata.st_mode):
        raise BenchmarkContractError("training output root is not a directory")
    entries: Dict[str, os.stat_result] = {}
    with os.scandir(root) as stream:
        for entry in stream:
            entry_metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(entry_metadata.st_mode):
                raise BenchmarkContractError(
                    "training output root entry is a symlink: " + entry.name
                )
            entries[entry.name] = entry_metadata
    if not entries:
        return "safe_empty_orphan"
    unknown = set(entries) - _TRAINING_ROOT_ALLOWED_ENTRIES
    if unknown:
        raise BenchmarkContractError(
            "training output root has unknown entries: " + repr(sorted(unknown))
        )
    contract = entries.get("run_contract.json")
    if contract is None or not stat.S_ISREG(contract.st_mode):
        raise BenchmarkContractError(
            "non-empty training output root lacks a regular run contract"
        )
    for name, entry_metadata in entries.items():
        if name == "stages":
            if not stat.S_ISDIR(entry_metadata.st_mode):
                raise BenchmarkContractError("training stages entry is not a directory")
        elif not stat.S_ISREG(entry_metadata.st_mode):
            raise BenchmarkContractError(
                "training output artifact is not a regular file: " + name
            )
    return "contract_only" if set(entries) == {"run_contract.json"} else "active_contract"


def _adapter_identity() -> Dict[str, str]:
    source = _lexical_absolute_path(__file__)
    document = source.with_name(ADAPTATION_DOC_FILENAME)
    for path, name in ((source, "adapter source"), (document, "adaptation document")):
        _assert_no_symlink_components(path, name=name)
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode):
            raise BenchmarkContractError(name + " is not a regular file")
    return {
        "adapter_source_filename": source.name,
        "adapter_source_sha256": _sha256_file(source),
        "adaptation_document_filename": document.name,
        "adaptation_document_sha256": _sha256_file(document),
        **{"dependency_" + name: digest for name, digest in training_hook_identity().items()},
    }


def _ordered_pair_ids_sha256(split: str, rows: Sequence["PairMetadata"]) -> str:
    return _canonical_sha256(
        {
            "schema_version": "rachel-shreddingnet-ordered-pair-ids/1.0",
            "split": split,
            "ordered_pair_ids": [row.pair_id for row in rows],
        }
    )


def _dataset_binding(dataset_audit: Mapping[str, object]) -> Dict[str, object]:
    manifest = dataset_audit.get("manifest_content_sha256")
    ordered = dataset_audit.get("ordered_pair_ids_sha256")
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {"train", "val"}
        or not all(isinstance(manifest[key], str) for key in ("train", "val"))
        or not isinstance(ordered, Mapping)
        or set(ordered) != {"train", "val"}
        or not all(isinstance(ordered[key], str) for key in ("train", "val"))
    ):
        raise BenchmarkContractError("Rachel manifest identity binding is malformed")
    return {
        "manifest_content_sha256": dict(manifest),
        "ordered_pair_ids_sha256": dict(ordered),
    }


def _train_val_provenance(
    adapter_identity: Mapping[str, str], dataset_binding: Mapping[str, object]
) -> Dict[str, str]:
    manifest = dataset_binding["manifest_content_sha256"]
    ordered = dataset_binding["ordered_pair_ids_sha256"]
    assert isinstance(manifest, Mapping) and isinstance(ordered, Mapping)
    return {
        "adapter_source_sha256": adapter_identity["adapter_source_sha256"],
        "adaptation_document_sha256": adapter_identity[
            "adaptation_document_sha256"
        ],
        "train_manifest_content_sha256": str(manifest["train"]),
        "val_manifest_content_sha256": str(manifest["val"]),
        "train_ordered_pair_ids_sha256": str(ordered["train"]),
        "val_ordered_pair_ids_sha256": str(ordered["val"]),
    }


def _canonical_sha256(value: Mapping[str, object]) -> str:
    copied = dict(value)
    copied.pop("content_sha256", None)
    payload = json.dumps(
        copied, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _open_exclusive_temporary(path: Path, *, name: str) -> int:
    try:
        return os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError as error:
        raise BenchmarkContractError(name + " temporary path already exists") from error


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path = _lexical_absolute_path(path)
    _assert_no_symlink_components(path, name="JSON output")
    _ensure_directory_tree(path.parent, name="JSON output parent")
    payload = dict(value)
    payload["content_sha256"] = _canonical_sha256(payload)
    temporary = path.with_name(
        path.name + ".{}.{}.tmp".format(os.getpid(), time.time_ns())
    )
    descriptor = _open_exclusive_temporary(temporary, name="JSON output")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
        _assert_no_symlink_components(path, name="JSON output")
        os.replace(temporary, path)
    finally:
        if _path_lexists(temporary):
            os.unlink(temporary)


def _write_json_new_atomic(path: Path, value: Mapping[str, object], *, name: str) -> None:
    """Publish a new JSON artifact atomically without an overwrite window."""

    path = _prepare_fresh_file_path(path, name=name)
    _ensure_directory_tree(path.parent, name=name + " parent")
    payload = dict(value)
    payload["content_sha256"] = _canonical_sha256(payload)
    temporary = path.with_name(
        path.name + ".{}.{}.tmp".format(os.getpid(), time.time_ns())
    )
    descriptor = _open_exclusive_temporary(temporary, name=name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
        _assert_no_symlink_components(path, name=name)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise BenchmarkContractError(name + " appeared during publish") from error
    finally:
        if _path_lexists(temporary):
            os.unlink(temporary)


def _git_output(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ("git", "-C", str(repo), *arguments),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise BenchmarkContractError("cannot inspect official git checkout") from error


def audit_official_source(repo: Union[str, Path]) -> Dict[str, object]:
    """Verify the exact official checkout without importing or modifying it."""

    root = Path(repo).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise BenchmarkContractError("official source root is not a directory")
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit != OFFICIAL_COMMIT:
        raise BenchmarkContractError(
            "official source commit differs: {} != {}".format(commit, OFFICIAL_COMMIT)
        )
    dirty = _git_output(root, "status", "--porcelain=v1")
    if dirty:
        raise BenchmarkContractError("official source checkout is dirty")
    observed: Dict[str, str] = {}
    for relative, expected in OFFICIAL_FILE_SHA256.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise BenchmarkContractError("official source file missing: " + relative)
        actual = _sha256_file(path)
        if actual != expected:
            raise BenchmarkContractError("official source hash differs: " + relative)
        observed[relative] = actual
    return {
        "authority": "official_shreddingnet_release_commit_and_art_2192_yaml",
        "root": root.as_posix(),
        "commit": commit,
        "clean_checkout": True,
        "file_sha256": observed,
        "release_yaml_executable_recipe": {
            "epochs": {"coarse": 128, "matching": 128, "classify": 128},
            "batch_size": {"coarse": 54, "matching": 36, "classify": 20},
            "validation_loader_shuffle": {
                "coarse": True,
                "matching": False,
                "classify": False,
            },
            "winner_semantics": {
                "coarse": "min_validation_infor_loss",
                "matching": "min_validation_positive_loss",
                "classify": "max_validation_accuracy",
            },
        },
        "released_architecture": {
            "coarse": {
                "input": "7x7_contour_and_rgb_patches_at_every_contour_point",
                "encoder": "two_nonshared_14_block_resplus_GAT_ResGCNs_dim64",
                "graph": "closed_contour_self_and_plusminus8_neighbors",
                "fusion": "bidirectional_cross_modal_attention_then_average",
                "descriptor": "flatten_all_contour_point_features",
                "objective": "multi_positive_InfoNCE_temperature_0.12",
            },
            "fine": {
                "backbone": "independently_trained_coarse_shape_backbone",
                "decoder": "two_stacked_bidirectional_cross_attention_blocks",
                "correspondence": "row_softmax_times_column_softmax",
                "objective": "focal_alpha_0.55_gamma_8_times_400",
            },
            "classifier": {
                "input": "float_erode_then_dilate_then_thresholded_correspondence",
                "network": "three_conv_CNN_adaptive_pool_sigmoid",
                "objective": "binary_cross_entropy",
                "released_state_semantics": (
                    "matcher_parameters_frozen_but_BatchNorm_buffers_update_in_train_mode"
                ),
            },
        },
        "paper_metric_definitions": {
            "CM": "coarse_candidate_edge_precision_and_recall",
            "FM": "fine_retained_edge_precision_and_recall",
            "SE": "score_evaluator_ROC_AUC_on_candidate_edges",
            "GA_precision": "true_edges_among_global_selected_assembly_edges",
            "GA_recall": (
                "ground_truth_edges_with_global_relative_pose_rotation_within_5deg_"
                "and_translation_within_100px"
            ),
            "pairwise_without_MST_must_not_be_called_GA": True,
            "adapter_balanced_pair_list_not_native_candidate_graph": True,
            "adapter_reports_native_CM_FM_SE": False,
        },
        "released_global_ransac": {
            "samples": 5,
            "correspondence_distance_px": 10,
            "maximum_iterations": 4000,
            "minimum_inliers_strictly_greater_than": 15,
            "maximum_validations": 20,
            "adapter_secondary_difference": (
                "deterministic_max_inlier_then_all_inlier_refit;headline_settings_retained;"
                "not_bitwise_release_RANSAC"
            ),
        },
        "paper_supplement_conflict_disclosed": {
            "supplement_epochs": {"coarse": 128, "matching": 128, "classify": 10},
            "supplement_batch_size": {"coarse": 75, "matching": 54, "classify": 20},
            "supplement_says_all_checkpoints_use_lowest_validation_loss": True,
        },
        "release_epoch_zero_best_checkpoint_bug_fixed_by_adapter": True,
    }


def _safe_manifest_artifact(value: object, prefix: Tuple[str, ...], suffix: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise BenchmarkContractError("manifest artifact path is invalid")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise BenchmarkContractError("manifest artifact path is unsafe: " + value)
    if tuple(logical.parts[: len(prefix)]) != prefix or logical.suffix.casefold() != suffix:
        raise BenchmarkContractError("manifest artifact class differs: " + value)


def _parse_pair_manifest(root: Path, split: str, train_materialized_manifest=None) -> Tuple[Tuple[PairMetadata, ...], str]:
    if split not in {"train", "val"}:
        raise BenchmarkContractError("benchmark adapter accepts only train and val")
    path = selected_manifest_path(root, split, train_materialized_manifest)
    if path.is_symlink() or not path.is_file():
        raise BenchmarkContractError("missing selected {} manifest".format(split))
    rows: List[PairMetadata] = []
    seen = set()
    with benchmark_manifest_lines(root, split, train_materialized_manifest) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise BenchmarkContractError(
                    "invalid {} JSON line {}".format(split, line_number)
                ) from error
            if not isinstance(value, Mapping) or value.get("split") != split:
                raise BenchmarkContractError("manifest row split differs")
            pair_id = value.get("pair_id")
            label = value.get("label")
            first = value.get("fragment_a")
            second = value.get("fragment_b")
            if (
                not isinstance(pair_id, str)
                or not pair_id
                or type(label) is not bool  # noqa: E721
                or not isinstance(first, Mapping)
                or not isinstance(second, Mapping)
            ):
                raise BenchmarkContractError("manifest row identity/label is invalid")
            if pair_id in seen:
                raise BenchmarkContractError("duplicate pair_id: " + pair_id)
            seen.add(pair_id)
            for fragment in (first, second):
                _safe_manifest_artifact(
                    fragment.get("model_mask_path"), ("model", "masks_800"), ".png"
                )
                _safe_manifest_artifact(
                    fragment.get("contour_path"), ("model", "contours_n512"), ".npz"
                )
            target = value.get("correspondence_path")
            if label:
                _safe_manifest_artifact(target, ("targets", "pairs"), ".npz")
            elif target is not None:
                raise BenchmarkContractError("negative row references correspondence GT")
            tokens = (first.get("fragment_token"), second.get("fragment_token"))
            lineages = (first.get("split_unit_id"), second.get("split_unit_id"))
            if any(not isinstance(item, str) or not item for item in tokens + lineages):
                raise BenchmarkContractError("fragment token/lineage is invalid")
            rows.append(
                PairMetadata(
                    split=split,
                    pair_id=pair_id,
                    label=label,
                    fragment_a_token=tokens[0],
                    fragment_b_token=tokens[1],
                    lineage_a=lineages[0],
                    lineage_b=lineages[1],
                )
            )
    if not rows:
        raise BenchmarkContractError(split + " manifest is empty")
    return tuple(rows), _sha256_file(path)


def audit_rachel_train_val(
    dataset_root: Union[str, Path], *, require_formal_counts: bool = True,
    train_materialized_manifest=None,
) -> Tuple[Dict[str, object], Tuple[PairMetadata, ...], Tuple[PairMetadata, ...]]:
    """Audit only the two authorized manifests and their lineage boundary."""

    root = Path(dataset_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise BenchmarkContractError("Rachel root is not a directory")
    train, train_sha = _parse_pair_manifest(root, "train", train_materialized_manifest)
    val, val_sha = _parse_pair_manifest(root, "val")
    train_lineages = {
        lineage for row in train for lineage in (row.lineage_a, row.lineage_b)
    }
    val_lineages = {lineage for row in val for lineage in (row.lineage_a, row.lineage_b)}
    overlap = train_lineages & val_lineages
    if overlap:
        raise BenchmarkContractError(
            "train/val source-manuscript lineage overlap: " + repr(sorted(overlap)[:5])
        )
    counts = {
        "train": {
            "rows": len(train),
            "positive": sum(row.label for row in train),
            "negative": sum(not row.label for row in train),
        },
        "val": {
            "rows": len(val),
            "positive": sum(row.label for row in val),
            "negative": sum(not row.label for row in val),
        },
    }
    expected = {
        "train": {"rows": 24_000, "positive": 12_000, "negative": 12_000},
        "val": {"rows": 3_000, "positive": 1_500, "negative": 1_500},
    }
    if require_formal_counts and counts != expected:
        raise BenchmarkContractError("Rachel formal train/val counts differ")
    if any(row["positive"] != row["negative"] for row in counts.values()):
        raise BenchmarkContractError("Rachel train/val must remain class balanced")
    audit = {
        "dataset_root": root.as_posix(),
        "manifest_sha256": {"train": train_sha, "val": val_sha},
        "manifest_content_sha256": {"train": train_sha, "val": val_sha},
        "ordered_pair_ids_sha256": {
            "train": _ordered_pair_ids_sha256("train", train),
            "val": _ordered_pair_ids_sha256("val", val),
        },
        "counts": counts,
        "formal_expected_counts": expected,
        "formal_counts_required": require_formal_counts,
        "train_materialized_manifest": (str(Path(train_materialized_manifest).resolve())
            if train_materialized_manifest is not None else None),
        "lineage_disjoint": True,
        "lineage_counts": {"train": len(train_lineages), "val": len(val_lineages)},
        "opened_manifests": [str(Path(train_materialized_manifest).resolve())
            if train_materialized_manifest is not None else "pairs/train.jsonl", "pairs/val.jsonl"],
        "sealed_test_manifest_opened": False,
        "real_data_opened": False,
        "rgb_opened": False,
    }
    return audit, train, val


def binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Exact Mann-Whitney AUROC with average ranks for ties."""

    y = np.asarray(labels, dtype=np.bool_)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not np.all(np.isfinite(s)):
        raise ValueError("labels/scores must be finite one-dimensional arrays")
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and s[order[end]] == s[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[y].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def edge_precision_recall_f1(
    labels: Sequence[bool], selected: Sequence[bool]
) -> Dict[str, Union[int, float]]:
    truth = np.asarray(labels, dtype=np.bool_)
    prediction = np.asarray(selected, dtype=np.bool_)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("edge arrays must be aligned and one-dimensional")
    tp = int(np.count_nonzero(truth & prediction))
    fp = int(np.count_nonzero(~truth & prediction))
    fn = int(np.count_nonzero(truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def topk_candidate_mask(
    rows: Sequence[PairMetadata], scores: Sequence[float], k: int
) -> np.ndarray:
    """Released per-fragment directed top-K selection, symmetrized by union."""

    if type(k) is not int or k <= 0:  # noqa: E721
        raise ValueError("k must be positive")
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(rows),) or not np.all(np.isfinite(values)):
        raise ValueError("coarse scores are invalid")
    incident: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        incident[row.fragment_a_token].append(index)
        incident[row.fragment_b_token].append(index)
    selected = np.zeros(len(rows), dtype=np.bool_)
    for indices in incident.values():
        ranked = sorted(indices, key=lambda i: (-values[i], rows[i].pair_id))
        selected[ranked[:k]] = True
    return selected


def cauchy_translation_consensus(
    source_rc: np.ndarray,
    target_rc: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    scale_px: float = 10.0,
    min_correspondences: int = 16,
    iterations: int = 20,
) -> TranslationEstimate:
    """Robust no-rotation translation estimate for upright fragments."""

    source = np.asarray(source_rc, dtype=np.float64)
    target = np.asarray(target_rc, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (2,) or target.shape != source.shape:
        raise ValueError("correspondence points must have aligned [N,2] shape")
    count = len(source)
    if weights is None:
        base_weight = np.ones(count, dtype=np.float64)
    else:
        base_weight = np.asarray(weights, dtype=np.float64)
        if base_weight.shape != (count,):
            raise ValueError("weights shape differs")
    finite = (
        np.all(np.isfinite(source), axis=1)
        & np.all(np.isfinite(target), axis=1)
        & np.isfinite(base_weight)
        & (base_weight > 0.0)
    )
    source = source[finite]
    target = target[finite]
    base_weight = base_weight[finite]
    count = len(source)
    if count < min_correspondences:
        return TranslationEstimate(np.zeros(2, dtype=np.float64), False, count, 0)
    displacement = target - source
    estimate = np.asarray(
        [
            _weighted_median(displacement[:, axis], base_weight)
            for axis in range(2)
        ],
        dtype=np.float64,
    )
    for _ in range(iterations):
        residual = np.linalg.norm(displacement - estimate[None, :], axis=1)
        robust = 1.0 / (1.0 + (residual / scale_px) ** 2)
        combined = base_weight * robust
        if not np.isfinite(combined.sum()) or combined.sum() <= 0.0:
            return TranslationEstimate(np.zeros(2), False, count, 0)
        updated = (displacement * combined[:, None]).sum(axis=0) / combined.sum()
        if np.linalg.norm(updated - estimate) < 1e-4:
            estimate = updated
            break
        estimate = updated
    residual = np.linalg.norm(displacement - estimate[None, :], axis=1)
    inlier_mask = residual <= scale_px
    inliers = int(np.count_nonzero(inlier_mask))
    if inliers >= min_correspondences:
        inlier_weight = base_weight[inlier_mask]
        estimate = (
            displacement[inlier_mask] * inlier_weight[:, None]
        ).sum(axis=0) / inlier_weight.sum()
        residual = np.linalg.norm(displacement - estimate[None, :], axis=1)
        inliers = int(np.count_nonzero(residual <= scale_px))
    valid = bool(inliers >= min_correspondences and np.all(np.isfinite(estimate)))
    return TranslationEstimate(estimate, valid, count, inliers)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _rigid_transform_2d(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid_source = source.mean(axis=0)
    centroid_target = target.mean(axis=0)
    centered_source = source - centroid_source
    centered_target = target - centroid_target
    u, _, vt = np.linalg.svd(centered_source.T @ centered_target)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = centroid_target - rotation @ centroid_source
    return rotation, translation


def deterministic_se2_ransac(
    source_rc: np.ndarray,
    target_rc: np.ndarray,
    *,
    seed: int,
    max_iterations: int = 4000,
    sample_size: int = 5,
    inlier_threshold_px: float = 10.0,
    min_inliers: int = 16,
) -> SE2Estimate:
    """Deterministic compatibility port of the release's SE(2) RANSAC settings."""

    source = np.asarray(source_rc, dtype=np.float64)
    target = np.asarray(target_rc, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (2,) or target.shape != source.shape:
        raise ValueError("RANSAC points must have aligned [N,2] shape")
    finite = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    source = source[finite]
    target = target[finite]
    count = len(source)
    if count < max(sample_size, min_inliers):
        return SE2Estimate(np.eye(2), np.zeros(2), 0.0, False, count, 0)
    rng = np.random.default_rng(seed)
    best_inliers: Optional[np.ndarray] = None
    best_error = math.inf
    for _ in range(max_iterations):
        proposal = rng.choice(count, size=sample_size, replace=False)
        try:
            rotation, translation = _rigid_transform_2d(source[proposal], target[proposal])
        except np.linalg.LinAlgError:
            continue
        transformed = source @ rotation.T + translation
        residual = np.linalg.norm(target - transformed, axis=1)
        inliers = residual <= inlier_threshold_px
        inlier_count = int(inliers.sum())
        if inlier_count < min_inliers:
            continue
        mean_error = float(residual[inliers].mean())
        if best_inliers is None or inlier_count > int(best_inliers.sum()) or (
            inlier_count == int(best_inliers.sum()) and mean_error < best_error
        ):
            best_inliers = inliers
            best_error = mean_error
            if inlier_count == count:
                break
    if best_inliers is None:
        return SE2Estimate(np.eye(2), np.zeros(2), 0.0, False, count, 0)
    rotation, translation = _rigid_transform_2d(source[best_inliers], target[best_inliers])
    angle = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return SE2Estimate(
        rotation=rotation,
        translation_rc=translation,
        rotation_degrees=angle,
        valid=True,
        correspondence_count=count,
        inlier_count=int(best_inliers.sum()),
    )


def _require_torch_geometric():
    try:
        from torch_geometric.nn import DeepGCNLayer, GATConv
    except ImportError as error:
        raise BenchmarkContractError(
            "training/inference requires torch-geometric compatible with installed torch"
        ) from error
    return DeepGCNLayer, GATConv


def _valid_prefix_lengths(valid: Tensor) -> Tensor:
    """Validate Rachel's prefix-padding contract and return per-row lengths."""

    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("valid must be a [B,N] bool tensor")
    lengths = valid.sum(dim=1, dtype=torch.long)
    expected = (
        torch.arange(valid.shape[1], device=valid.device)[None, :]
        < lengths[:, None]
    )
    if not torch.equal(valid, expected):
        raise BenchmarkContractError("contour valid mask must be a contiguous prefix")
    if bool(torch.any(lengths == 0)):
        raise BenchmarkContractError("every fragment needs at least one contour point")
    return lengths


def cycle_edge_index(valid: Tensor, radius: int = 8) -> Tensor:
    """Build compact per-fragment closed-contour +/-radius graphs.

    Nodes are numbered in the same compact order produced by gathering
    ``value[valid]``. Padded slots never become graph nodes and therefore never
    enter a ResGCN BatchNorm statistic.
    """

    lengths = _valid_prefix_lengths(valid)
    if type(radius) is not int or radius <= 0:  # noqa: E721
        raise ValueError("radius must be positive")
    edge_blocks = []
    device = valid.device
    compact_offset = 0
    for length in lengths.tolist():
        count = int(length)
        nodes = torch.arange(
            compact_offset, compact_offset + count, dtype=torch.long, device=device
        )
        for offset in range(-radius, radius + 1):
            source = nodes
            target = torch.roll(nodes, shifts=-offset)
            edge_blocks.append(torch.stack((source, target), dim=0))
        compact_offset += count
    return torch.cat(edge_blocks, dim=1).long()


def _sample_patches(image: Tensor, points_rc: Tensor, patch_size: int) -> Tensor:
    """Index local windows without materializing a full im2col tensor."""

    if image.ndim != 4 or points_rc.ndim != 3 or points_rc.shape[-1] != 2:
        raise ValueError("image/points shapes differ from [B,C,H,W]/[B,N,2]")
    batch_size, channels, height, width = image.shape
    if points_rc.shape[0] != batch_size:
        raise ValueError("image/points batch sizes differ")
    half = patch_size // 2
    center = points_rc.round().long()
    offsets = torch.arange(-half, half + 1, device=image.device)
    rows = center[:, :, 0, None, None] + offsets[None, None, :, None]
    cols = center[:, :, 1, None, None] + offsets[None, None, None, :]
    rows = rows.expand(-1, -1, patch_size, patch_size)
    cols = cols.expand(-1, -1, patch_size, patch_size)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows.clamp(0, height - 1)
    cols = cols.clamp(0, width - 1)
    batch = torch.arange(batch_size, device=image.device)[:, None, None, None]
    bhwc = image.permute(0, 2, 3, 1)
    sampled = bhwc[batch, rows, cols]
    sampled = sampled * inside[..., None].to(sampled.dtype)
    return sampled.permute(0, 1, 4, 2, 3).contiguous()


def mask_only_contour_support_patches(
    mask: Tensor, points_rc: Tensor, valid: Tensor, patch_size: int = 7
) -> Tuple[Tensor, Tensor]:
    """Create released-shape contour and three-channel mask-support patches."""

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B,1,H,W]")
    if valid.shape != points_rc.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid/points shapes differ")
    batch_size, _, height, width = mask.shape
    line_map = torch.zeros_like(mask)
    rounded = points_rc.round().long()
    rows = rounded[:, :, 0].clamp(0, height - 1)
    cols = rounded[:, :, 1].clamp(0, width - 1)
    batch = torch.arange(batch_size, device=mask.device)[:, None].expand_as(rows)
    line_map[batch[valid], 0, rows[valid], cols[valid]] = 1.0
    contour = _sample_patches(line_map, points_rc, patch_size).squeeze(2)
    support = _sample_patches(mask.expand(-1, 3, -1, -1), points_rc, patch_size)
    contour = contour * valid[:, :, None, None].to(contour.dtype)
    support = support * valid[:, :, None, None, None].to(support.dtype)
    return contour, support


def released_antidiagonal_morphology(probability: Tensor, threshold: float) -> Tensor:
    """Torch equivalent of released float morphology, then thresholding.

    The release applies OpenCV erosion/dilation to the dual-softmax float
    matrix and thresholds only afterwards.  Moving the threshold before the
    morphology is not equivalent and is deliberately avoided here.
    """

    if probability.ndim != 3:
        raise ValueError("probability must have shape [B,N,M]")
    if not probability.is_floating_point() or not math.isfinite(threshold):
        raise ValueError("probability/threshold types are invalid")
    padded = F.pad(probability, (1, 1, 1, 1), value=0.0)
    upper_right = padded[:, 0:-2, 2:]
    lower_left = padded[:, 2:, 0:-2]
    eroded = torch.minimum(upper_right, lower_left)
    padded_eroded = F.pad(eroded, (1, 1, 1, 1), value=0.0)
    dilated = torch.maximum(
        torch.maximum(
            padded_eroded[:, 0:-2, 2:],
            padded_eroded[:, 1:-1, 1:-1],
        ),
        padded_eroded[:, 2:, 0:-2],
    )
    return dilated > threshold


def released_dual_softmax(logits: Tensor, pair_valid: Tensor) -> Tensor:
    """Released dual softmax with an explicit FP32 probability boundary.

    AMP may supply FP16/BF16 logits. Masking FP16 with the release-style large
    negative sentinel overflows before softmax, so logits are promoted to FP32
    for masking and both normalizations. The returned probabilities stay FP32;
    autograd casts their gradients back through the promotion to the input
    dtype.
    """

    if logits.ndim != 3 or pair_valid.shape != logits.shape:
        raise ValueError("logits/pair_valid must have aligned [B,N,M] shape")
    if pair_valid.dtype != torch.bool:
        raise TypeError("pair_valid must be bool")
    masked = logits.to(torch.float32).masked_fill(~pair_valid, -1e9)
    probability = torch.softmax(masked, dim=1) * torch.softmax(masked, dim=-1)
    return probability * pair_valid.to(probability.dtype)


class _ConvBNReLU(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, *, padding: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(value)))


class ReleasedPatchEncoder(nn.Module):
    """Released patch encoders applied only to real contour points."""

    def __init__(self) -> None:
        super().__init__()
        self.contour = _ConvBNReLU(1, 64, padding=0)
        self.support1 = _ConvBNReLU(3, 32, padding=1)
        self.support2 = _ConvBNReLU(32, 64, padding=0)

    def forward(
        self, contour: Tensor, support: Tensor, valid: Tensor
    ) -> Tuple[Tensor, Tensor]:
        batch_size, count = contour.shape[:2]
        if (
            contour.shape != (batch_size, count, 7, 7)
            or support.shape != (batch_size, count, 3, 7, 7)
            or valid.shape != (batch_size, count)
        ):
            raise ValueError("patch encoder inputs violate the released 7x7 contract")
        _valid_prefix_lengths(valid)
        flat_valid = valid.reshape(-1)
        contour_compact = contour.reshape(batch_size * count, 1, 7, 7)[flat_valid]
        support_compact = support.reshape(batch_size * count, 3, 7, 7)[flat_valid]
        contour_compact = self.contour(contour_compact)
        contour_compact = F.adaptive_avg_pool2d(contour_compact, (1, 1)).flatten(1)
        support_compact = self.support1(support_compact)
        support_compact = self.support2(support_compact)
        support_compact = F.adaptive_avg_pool2d(support_compact, (1, 1)).flatten(1)
        contour_feature = contour_compact.new_zeros((batch_size * count, 64))
        support_feature = support_compact.new_zeros((batch_size * count, 64))
        contour_feature[flat_valid] = contour_compact
        support_feature[flat_valid] = support_compact
        return (
            contour_feature.reshape(batch_size, count, 64),
            support_feature.reshape(batch_size, count, 64),
        )


class ReleasedDeepResGCN(nn.Module):
    def __init__(self, recipe: ReleaseRecipe) -> None:
        super().__init__()
        DeepGCNLayer, GATConv = _require_torch_geometric()
        blocks = []
        for _ in range(recipe.resgcn_blocks):
            convolution = GATConv(
                recipe.feature_dim,
                recipe.feature_dim,
                heads=1,
                concat=False,
            )
            normalizer = nn.BatchNorm1d(recipe.feature_dim)
            activation = nn.ReLU(inplace=True)
            blocks.append(
                DeepGCNLayer(
                    convolution, normalizer, activation, block="res+"
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, value: Tensor, edge_index: Tensor) -> Tensor:
        for block in self.blocks:
            value = block(value, edge_index)
        return value


class ReleasedMaskOnlyFeatureExtractor(nn.Module):
    """Two non-shared ResGCNs; RGB branch is adapted to binary support."""

    def __init__(self, recipe: ReleaseRecipe) -> None:
        super().__init__()
        self.recipe = recipe
        self.patch_encoder = ReleasedPatchEncoder()
        self.coordinate_fuse = nn.Linear(recipe.feature_dim + 2, recipe.feature_dim)
        self.contour_gcn = ReleasedDeepResGCN(recipe)
        self.support_gcn = ReleasedDeepResGCN(recipe)

    def forward(
        self, mask: Tensor, points_rc: Tensor, valid: Tensor
    ) -> Tuple[Tensor, Tensor]:
        contour_patch, support_patch = mask_only_contour_support_patches(
            mask, points_rc, valid, self.recipe.patch_size
        )
        contour_feature, support_feature = self.patch_encoder(
            contour_patch, support_patch, valid
        )
        normalized = points_rc / 400.0 - 1.0
        # This preserves the release's effective ``pcd += 1; center; -= 1``.
        valid_float = valid[:, :, None].to(normalized.dtype)
        mean = (normalized * valid_float).sum(dim=1, keepdim=True) / valid_float.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        centered = (normalized - mean - 1.0) * valid_float
        batch_size, count, dim = contour_feature.shape
        flat_valid = valid.reshape(-1)
        contour_compact = contour_feature.reshape(batch_size * count, dim)[flat_valid]
        support_compact = support_feature.reshape(batch_size * count, dim)[flat_valid]
        centered_compact = centered.reshape(batch_size * count, 2)[flat_valid]
        contour_compact = self.coordinate_fuse(
            torch.cat((contour_compact, centered_compact), dim=-1)
        )
        edge_index = cycle_edge_index(valid, self.recipe.cycle_radius)
        contour_compact = self.contour_gcn(contour_compact, edge_index)
        support_compact = self.support_gcn(support_compact, edge_index)
        contour_output = contour_compact.new_zeros((batch_size * count, dim))
        support_output = support_compact.new_zeros((batch_size * count, dim))
        contour_output[flat_valid] = contour_compact
        support_output[flat_valid] = support_compact
        return (
            contour_output.reshape(batch_size, count, dim),
            support_output.reshape(batch_size, count, dim),
        )


class ReleasedAttentionBlock(nn.Module):
    """Released attention equation (heads affect scale; projections are not split)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_valid: Tensor,
        key_valid: Tensor,
    ) -> Tensor:
        residual = query
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~key_valid[:, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        output = self.norm1(residual + torch.matmul(weights, v))
        output = self.norm2(output + self.ffn(output))
        return output * query_valid[:, :, None].to(output.dtype)


class ReleasedCrossModalFuse(nn.Module):
    def __init__(self, recipe: ReleaseRecipe) -> None:
        super().__init__()
        self.first = ReleasedAttentionBlock(recipe.feature_dim, recipe.attention_heads)
        self.second = ReleasedAttentionBlock(recipe.feature_dim, recipe.attention_heads)

    def forward(self, first: Tensor, second: Tensor, valid: Tensor) -> Tensor:
        enhanced_first = self.first(first, second, second, valid, valid)
        enhanced_second = self.second(second, first, first, valid, valid)
        return 0.5 * (enhanced_first + enhanced_second)


class ReleasedFragmentEncoder(nn.Module):
    def __init__(self, recipe: ReleaseRecipe) -> None:
        super().__init__()
        self.extractor = ReleasedMaskOnlyFeatureExtractor(recipe)
        self.fuse = ReleasedCrossModalFuse(recipe)

    def local_features(self, mask: Tensor, points_rc: Tensor, valid: Tensor) -> Tensor:
        contour, support = self.extractor(mask, points_rc, valid)
        return self.fuse(contour, support, valid)

    def forward(self, mask: Tensor, points_rc: Tensor, valid: Tensor) -> Tensor:
        local = self.local_features(mask, points_rc, valid)
        return local.reshape(local.shape[0], -1)


class ReleasedFineMatcher(nn.Module):
    def __init__(self, recipe: ReleaseRecipe) -> None:
        super().__init__()
        self.encoder = ReleasedFragmentEncoder(recipe)
        blocks = []
        for _ in range(recipe.decoder_blocks):
            blocks.append(
                nn.ModuleList(
                    (
                        ReleasedAttentionBlock(
                            recipe.feature_dim, recipe.attention_heads
                        ),
                        ReleasedAttentionBlock(
                            recipe.feature_dim, recipe.attention_heads
                        ),
                    )
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        mask_a: Tensor,
        mask_b: Tensor,
        points_a: Tensor,
        points_b: Tensor,
        valid_a: Tensor,
        valid_b: Tensor,
    ) -> Tensor:
        first = self.encoder.local_features(mask_a, points_a, valid_a)
        second = self.encoder.local_features(mask_b, points_b, valid_b)
        for update_first, update_second in self.blocks:
            first = update_first(first, second, second, valid_a, valid_b)
            second = update_second(second, first, first, valid_b, valid_a)
        logits = torch.bmm(first, second.transpose(1, 2)) / math.sqrt(first.shape[-1])
        pair_valid = valid_a[:, :, None] & valid_b[:, None, :]
        return released_dual_softmax(logits, pair_valid)


class ReleasedCNNScoreEvaluator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, binary_correspondence: Tensor) -> Tensor:
        feature = self.cnn(binary_correspondence[:, None].to(torch.float32))
        return torch.sigmoid(self.fc(feature.flatten(1)).squeeze(1))


class ReleasedClassifyStage(nn.Module):
    """Released stage-three wrapper, including its frozen matcher buffers.

    The official code freezes matcher parameters, but calls ``model.train()``
    on the complete wrapper.  Consequently matcher BatchNorm running buffers
    evolve during classifier training.  Keeping both modules in this wrapper
    preserves that executable-release behavior and makes the winning state
    fully restorable.
    """

    def __init__(self, recipe: ReleaseRecipe) -> None:
        super().__init__()
        self.matcher = ReleasedFineMatcher(recipe)
        self.classifier = ReleasedCNNScoreEvaluator()
        self.matcher.requires_grad_(False)

    def forward(
        self,
        mask_a: Tensor,
        mask_b: Tensor,
        points_a: Tensor,
        points_b: Tensor,
        valid_a: Tensor,
        valid_b: Tensor,
        correspondence_threshold: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        probability = self.matcher(
            mask_a, mask_b, points_a, points_b, valid_a, valid_b
        )
        binary = released_antidiagonal_morphology(
            probability, correspondence_threshold
        )
        return self.classifier(binary), probability, binary


def multi_positive_infonce(
    query: Tensor,
    key: Tensor,
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
    temperature: float,
) -> Tensor:
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    logits = query @ key.T / temperature
    positive = torch.zeros_like(logits, dtype=torch.bool)
    targets_by_source: Dict[str, set] = defaultdict(set)
    for source, target in zip(source_tokens, target_tokens):
        targets_by_source[source].add(target)
    for row, source in enumerate(source_tokens):
        allowed_targets = targets_by_source[source]
        for column, target in enumerate(target_tokens):
            if target in allowed_targets:
                positive[row, column] = True
    positive.fill_diagonal_(True)
    log_probability = F.log_softmax(logits, dim=1)
    masked = torch.logsumexp(log_probability.masked_fill(~positive, -torch.inf), dim=1)
    return -masked.mean()


def released_coarse_loss(
    feature_a: Tensor,
    feature_b: Tensor,
    tokens_a: Sequence[str],
    tokens_b: Sequence[str],
    temperature: float,
) -> Tuple[Tensor, Tensor]:
    cross = multi_positive_infonce(feature_a, feature_b, tokens_a, tokens_b, temperature)
    same_a = multi_positive_infonce(feature_a, feature_a, tokens_a, tokens_a, temperature)
    same_b = multi_positive_infonce(feature_b, feature_b, tokens_b, tokens_b, temperature)
    total = cross + (0.5 * same_a + 0.5 * same_b) / 2.0
    return total, cross


def correspondence_target_matrix(batch: RachelBatch, device: torch.device) -> Tensor:
    target_a = torch.as_tensor(batch.target_a, dtype=torch.long, device=device)
    masks = supervised_contour_masks(batch)
    valid_a = torch.as_tensor(masks[0], dtype=torch.bool, device=device)
    valid_b = torch.as_tensor(masks[1], dtype=torch.bool, device=device)
    batch_size, cap = target_a.shape
    output = torch.zeros((batch_size, cap, cap), dtype=torch.bool, device=device)
    rows = torch.arange(cap, device=device)[None, :].expand(batch_size, -1)
    batch_index = torch.arange(batch_size, device=device)[:, None].expand_as(rows)
    selected = (target_a >= 0) & valid_a
    output[batch_index[selected], rows[selected], target_a[selected]] = True
    output &= valid_a[:, :, None] & valid_b[:, None, :]
    return output


def released_focal_correspondence_loss(
    probability: Tensor,
    ground_truth: Tensor,
    valid_pairs: Tensor,
    *,
    alpha: float,
    gamma: float,
) -> Tuple[Tensor, Tensor]:
    # Logarithms stay float32 even when the encoder runs under CUDA AMP. Keep
    # the release's additive epsilon inside each logarithm. In FP32,
    # ``1.0 - 1e-9`` rounds back to one, so an upper clamp cannot safely replace
    # ``log(1 - p + eps)`` for an exact-one negative probability.
    probability = probability.float()
    if not bool(torch.isfinite(probability).all()) or bool(
        torch.any((probability < 0.0) | (probability > 1.0))
    ):
        raise BenchmarkContractError("fine probabilities must be finite in [0,1]")
    positive_probability = probability[ground_truth]
    negative_mask = valid_pairs & ~ground_truth
    negative_probability = probability[negative_mask]
    if positive_probability.numel() == 0 or negative_probability.numel() == 0:
        raise BenchmarkContractError("fine batch lacks positive or negative cells")
    epsilon = probability.new_tensor(1e-9)
    positive_loss = -alpha * (1.0 - positive_probability).pow(gamma) * torch.log(
        positive_probability + epsilon
    )
    negative_loss = -(1.0 - alpha) * negative_probability.pow(gamma) * torch.log(
        1.0 - negative_probability + epsilon
    )
    return 400.0 * torch.cat((positive_loss, negative_loss)).mean(), positive_loss.mean()


class _IndexView(Dataset):
    def __init__(self, source: RachelPairDataset, indices: Sequence[int]) -> None:
        self.source = source
        self.indices = tuple(int(index) for index in indices)
        if not self.indices:
            raise BenchmarkContractError("stage dataset is empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.source[self.indices[index]]


def _loader(
    dataset: RachelPairDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _IndexView(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        collate_fn=collate_rachel_pairs,
    )


def _side_inputs(
    batch: RachelBatch, side: str, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor]:
    if side not in {"a", "b"}:
        raise ValueError("side must be a or b")
    mask = torch.as_tensor(
        getattr(batch, "mask_" + side), dtype=torch.float32, device=device
    )
    points = torch.as_tensor(
        getattr(batch, "points_rc_" + side), dtype=torch.float32, device=device
    )
    valid = torch.as_tensor(
        getattr(batch, "contour_valid_" + side), dtype=torch.bool, device=device
    )
    return mask, points, valid


def _matching_forward(
    model: ReleasedFineMatcher, batch: RachelBatch, device: torch.device
) -> Tensor:
    mask_a, points_a, valid_a = _side_inputs(batch, "a", device)
    mask_b, points_b, valid_b = _side_inputs(batch, "b", device)
    return model(mask_a, mask_b, points_a, points_b, valid_a, valid_b)


def _coarse_batch(
    model: ReleasedFragmentEncoder,
    batch: RachelBatch,
    device: torch.device,
    recipe: ReleaseRecipe,
) -> Tuple[Tensor, Dict[str, float]]:
    mask_a, points_a, valid_a = _side_inputs(batch, "a", device)
    mask_b, points_b, valid_b = _side_inputs(batch, "b", device)
    feature_a = model(mask_a, points_a, valid_a)
    feature_b = model(mask_b, points_b, valid_b)
    total, cross = released_coarse_loss(
        feature_a,
        feature_b,
        batch.fragment_a_tokens,
        batch.fragment_b_tokens,
        recipe.infonce_temperature,
    )
    normalized_a = F.normalize(feature_a.detach(), dim=-1)
    normalized_b = F.normalize(feature_b.detach(), dim=-1)
    rank = torch.argsort(normalized_a @ normalized_b.T, dim=1, descending=True)
    diagonal = torch.arange(len(batch.pair_ids), device=device)
    top1 = float((rank[:, 0] == diagonal).float().mean().cpu())
    top5 = float(
        (rank[:, : min(5, rank.shape[1])] == diagonal[:, None])
        .any(dim=1)
        .float()
        .mean()
        .cpu()
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "infor_loss": float(cross.detach().cpu()),
        "top1_recall": top1,
        "top5_recall": top5,
    }


def _matching_batch(
    model: ReleasedFineMatcher,
    batch: RachelBatch,
    device: torch.device,
    recipe: ReleaseRecipe,
) -> Tuple[Tensor, Dict[str, float]]:
    probability = _matching_forward(model, batch, device)
    ground_truth = correspondence_target_matrix(batch, device)
    masks = supervised_contour_masks(batch)
    valid_a = torch.as_tensor(masks[0], dtype=torch.bool, device=device)
    valid_b = torch.as_tensor(masks[1], dtype=torch.bool, device=device)
    valid_pairs = valid_a[:, :, None] & valid_b[:, None, :]
    total, positive = released_focal_correspondence_loss(
        probability,
        ground_truth,
        valid_pairs,
        alpha=recipe.focal_alpha,
        gamma=recipe.focal_gamma,
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "positive_loss": float(positive.detach().cpu()),
    }


def _classify_batch(
    model: ReleasedClassifyStage,
    batch: RachelBatch,
    device: torch.device,
    recipe: ReleaseRecipe,
) -> Tuple[Tensor, Dict[str, float]]:
    mask_a, points_a, valid_a = _side_inputs(batch, "a", device)
    mask_b, points_b, valid_b = _side_inputs(batch, "b", device)
    score, _, _ = model(
        mask_a,
        mask_b,
        points_a,
        points_b,
        valid_a,
        valid_b,
        recipe.correspondence_threshold,
    )
    labels = torch.as_tensor(batch.labels, dtype=torch.float32, device=device)
    loss = F.binary_cross_entropy(score, labels)
    predicted = score >= 0.5
    truth = labels >= 0.5
    tp = (predicted & truth).sum().item()
    fp = (predicted & ~truth).sum().item()
    fn = (~predicted & truth).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return loss, {
        "loss": float(loss.detach().cpu()),
        "accuracy": float((predicted == truth).float().mean().cpu()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _aggregate_epoch(values: Sequence[Tuple[int, Mapping[str, float]]]) -> Dict[str, float]:
    if not values:
        raise BenchmarkContractError("epoch produced no batches")
    keys = set(values[0][1])
    if any(set(row) != keys for _, row in values):
        raise BenchmarkContractError("epoch metric keys differ")
    return {
        # The released BaseTrainer averages one metric value per DataLoader
        # batch, including a possibly smaller final batch.
        key: sum(float(row[key]) for _, row in values) / len(values)
        for key in sorted(keys)
    }


def _encode_safe_checkpoint_value(value: Any) -> Any:
    """Avoid raw pickle floats unsupported by early weights-only loaders."""

    if type(value) is float:  # noqa: E721 - bool/int must remain unchanged
        return {_CHECKPOINT_FLOAT_SENTINEL: torch.tensor(value, dtype=torch.float64)}
    if isinstance(value, Mapping):
        return {
            key: _encode_safe_checkpoint_value(item) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_encode_safe_checkpoint_value(item) for item in value)
    if isinstance(value, list):
        return [_encode_safe_checkpoint_value(item) for item in value]
    return value


def _decode_safe_checkpoint_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {_CHECKPOINT_FLOAT_SENTINEL}:
            encoded = value[_CHECKPOINT_FLOAT_SENTINEL]
            if not isinstance(encoded, Tensor) or encoded.numel() != 1:
                raise BenchmarkContractError("checkpoint float encoding is invalid")
            return float(encoded.detach().cpu().item())
        return {
            key: _decode_safe_checkpoint_value(item) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_decode_safe_checkpoint_value(item) for item in value)
    if isinstance(value, list):
        return [_decode_safe_checkpoint_value(item) for item in value]
    return value


def _torch_save_atomic(path: Path, value: Mapping[str, object]) -> None:
    path = _lexical_absolute_path(path)
    _assert_no_symlink_components(path, name="checkpoint")
    _ensure_directory_tree(path.parent, name="checkpoint parent")
    temporary = path.with_name(
        path.name + ".{}.{}.tmp".format(os.getpid(), time.time_ns())
    )
    descriptor = _open_exclusive_temporary(temporary, name="checkpoint")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(_encode_safe_checkpoint_value(dict(value)), stream)
        _assert_no_symlink_components(path, name="checkpoint")
        os.replace(temporary, path)
    finally:
        if _path_lexists(temporary):
            os.unlink(temporary)


def _torch_load(path: Path, device: torch.device) -> Mapping[str, object]:
    path = _lexical_absolute_path(path)
    _assert_no_symlink_components(path, name="checkpoint")
    if path.is_symlink() or not path.is_file():
        raise BenchmarkContractError("checkpoint missing or symlinked: " + str(path))
    parameters = inspect.signature(torch.load).parameters
    kwargs: Dict[str, object] = {"map_location": device}
    if "weights_only" in parameters:
        kwargs["weights_only"] = True
    value = _decode_safe_checkpoint_value(torch.load(path, **kwargs))
    if not isinstance(value, Mapping):
        raise BenchmarkContractError("checkpoint root is not a mapping")
    return value


def _winner_checkpoint_payload(
    *,
    stage: str,
    epoch: int,
    settings: Mapping[str, object],
    selection_value: float,
    recipe: ReleaseRecipe,
    runtime: RuntimeBatchConfig,
    model_state_dict: Mapping[str, Tensor],
    matching_winner_sha256: Optional[str],
    adapter_identity: Mapping[str, str],
    dataset_binding: Mapping[str, object],
    validation_order_receipt: Mapping[str, object],
) -> Dict[str, object]:
    """Build the hash-bound stage winner saved by the formal runner."""

    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": _stage_checkpoint_kind(stage, "winner"),
        "method_id": METHOD_ID,
        "stage": stage,
        "epoch": int(epoch),
        "selection_metric": settings["selection_metric"],
        "selection_mode": settings["selection_mode"],
        "selection_value": float(selection_value),
        "recipe": asdict(recipe),
        "runtime_batch_config": asdict(runtime),
        "model_state_dict": model_state_dict,
        "matching_winner_sha256": matching_winner_sha256,
        "adapter_identity": dict(adapter_identity),
        "dataset_binding": dict(dataset_binding),
        "train_val_provenance": _train_val_provenance(
            adapter_identity, dataset_binding
        ),
        "validation_order_contract": _validation_order_contract(stage, recipe.seed),
        "winner_validation_order": dict(validation_order_receipt),
    }


def _winner_identity_matches(
    checkpoint: Mapping[str, object],
    *,
    stage: str,
    best_epoch: int,
    best_value: float,
    settings: Mapping[str, object],
    recipe: ReleaseRecipe,
    runtime: RuntimeBatchConfig,
    matching_winner_sha256: Optional[str],
    adapter_identity: Mapping[str, str],
    dataset_binding: Mapping[str, object],
    validation_order_receipt: Mapping[str, object],
) -> bool:
    required = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": _stage_checkpoint_kind(stage, "winner"),
        "method_id": METHOD_ID,
        "stage": stage,
        "epoch": int(best_epoch),
        "selection_metric": settings["selection_metric"],
        "selection_mode": settings["selection_mode"],
        "selection_value": float(best_value),
        "recipe": asdict(recipe),
        "runtime_batch_config": asdict(runtime),
        "matching_winner_sha256": matching_winner_sha256,
        "adapter_identity": dict(adapter_identity),
        "dataset_binding": dict(dataset_binding),
        "train_val_provenance": _train_val_provenance(
            adapter_identity, dataset_binding
        ),
        "validation_order_contract": _validation_order_contract(stage, recipe.seed),
        "winner_validation_order": dict(validation_order_receipt),
    }
    return all(checkpoint.get(key) == value for key, value in required.items()) and isinstance(
        checkpoint.get("model_state_dict"), Mapping
    )


def _repair_or_verify_progress_winner(
    *,
    stage: str,
    progress: Mapping[str, object],
    winner_path: Path,
    settings: Mapping[str, object],
    recipe: ReleaseRecipe,
    runtime: RuntimeBatchConfig,
    device: torch.device,
    matching_winner_sha256: Optional[str],
    adapter_identity: Mapping[str, str],
    dataset_binding: Mapping[str, object],
    validation_order_receipt: Mapping[str, object],
) -> bool:
    """Verify the winner, repairing only the safe progress-first commit gap.

    Epoch progress is atomically committed before a newly improved winner. A
    process failure in that narrow interval leaves the current epoch model in
    ``progress.pt`` and an absent or stale ``winner.pt``. Repair is safe only
    when the progress epoch itself is the recorded best epoch; an older best
    model cannot be reconstructed from the current progress state.

    Returns whether a repair was performed.
    """

    best_epoch = int(progress["best_epoch"])
    best_value = float(progress["best_value"])
    epoch_completed = int(progress["epoch_completed"])
    winner_matches = False
    if winner_path.is_file() and not winner_path.is_symlink():
        try:
            winner = _torch_load(winner_path, device)
        except (BenchmarkContractError, OSError, RuntimeError, ValueError, EOFError):
            winner = {}
        winner_matches = _winner_identity_matches(
            winner,
            stage=stage,
            best_epoch=best_epoch,
            best_value=best_value,
            settings=settings,
            recipe=recipe,
            runtime=runtime,
            matching_winner_sha256=matching_winner_sha256,
            adapter_identity=adapter_identity,
            dataset_binding=dataset_binding,
            validation_order_receipt=validation_order_receipt,
        )
    if winner_matches:
        return False
    if best_epoch != epoch_completed:
        raise BenchmarkContractError(
            stage + " winner differs and its older best state cannot be repaired"
        )
    model_state = progress.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise BenchmarkContractError(stage + " progress model state is invalid")
    _torch_save_atomic(
        winner_path,
        _winner_checkpoint_payload(
            stage=stage,
            epoch=best_epoch,
            settings=settings,
            selection_value=best_value,
            recipe=recipe,
            runtime=runtime,
            model_state_dict=model_state,
            matching_winner_sha256=matching_winner_sha256,
            adapter_identity=adapter_identity,
            dataset_binding=dataset_binding,
            validation_order_receipt=validation_order_receipt,
        ),
    )
    return True


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _stage_model(stage: str, recipe: ReleaseRecipe) -> nn.Module:
    if stage == "coarse":
        return ReleasedFragmentEncoder(recipe)
    if stage == "matching":
        return ReleasedFineMatcher(recipe)
    if stage == "classify":
        return ReleasedClassifyStage(recipe)
    raise ValueError("unknown stage")


def _stage_recipe(stage: str, recipe: ReleaseRecipe) -> Dict[str, object]:
    if stage == "coarse":
        return {
            "epochs": recipe.coarse_epochs,
            "batch_size": recipe.coarse_batch_size,
            "learning_rate": recipe.coarse_lr,
            "selection_metric": "infor_loss",
            "selection_mode": "min",
            "population": "positive_pairs_only",
            "official_validation_loader_shuffle": True,
        }
    if stage == "matching":
        return {
            "epochs": recipe.matching_epochs,
            "batch_size": recipe.matching_batch_size,
            "learning_rate": recipe.matching_lr,
            "selection_metric": "positive_loss",
            "selection_mode": "min",
            "population": "positive_pairs_only",
            "official_validation_loader_shuffle": False,
        }
    if stage == "classify":
        return {
            "epochs": recipe.classify_epochs,
            "batch_size": recipe.classify_batch_size,
            "learning_rate": recipe.classify_lr,
            "selection_metric": ("recall95_precision" if recipe.selection_metric == "recall95_precision" else "accuracy"),
            "selection_mode": "max",
            "population": "frozen_balanced_positive_negative_manifest",
            "official_validation_loader_shuffle": False,
        }
    raise ValueError("unknown stage")


def _selection_improved(value: float, best: float, mode: str) -> bool:
    return value < best if mode == "min" else value > best


def _recall_validation_metrics(labels, scores):
    from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
    operating = fit_operating_points(labels, scores)
    return dict(recall95_precision=float(operating["selection_key"][0]),
        recall95_auprc=float(operating["selection_key"][1]),
        recall95_threshold=float(operating["thresholds"]["recall_95"]),
        recall95_recall=float(operating["validation"]["recall_95"]["recall"]),
        max_f1=float(operating["validation"]["max_f1"]["f1"]))


def _recall_selection_improved(metrics, best_value, best_epoch, history):
    previous_ap = float(history[best_epoch]["val"]["recall95_auprc"]) if best_epoch >= 0 else -math.inf
    return (metrics["recall95_precision"], metrics["recall95_auprc"]) > (best_value, previous_ap)


def _slice_batch(batch: RachelBatch, start: int, stop: int) -> RachelBatch:
    if not 0 <= start < stop <= len(batch.pair_ids):
        raise ValueError("invalid RachelBatch slice")
    values: Dict[str, object] = {}
    for name in batch.__dataclass_fields__:
        value = getattr(batch, name)
        values[name] = value[start:stop]
    return RachelBatch(**values)


def _microbatches(batch: RachelBatch, size: int) -> Tuple[RachelBatch, ...]:
    return tuple(
        _slice_batch(batch, start, min(start + size, len(batch.pair_ids)))
        for start in range(0, len(batch.pair_ids), size)
    )


def _epoch_order_indices(
    population: Sequence[int], seed: int, epoch_one_based: int
) -> Tuple[int, ...]:
    """Use the formal Rachel epoch-order algorithm on a stage population."""

    if not population or seed < 0 or epoch_one_based <= 0:
        raise ValueError("invalid epoch-order arguments")
    generator = np.random.default_rng(seed + 1_000_003 * epoch_one_based)
    permutation = generator.permutation(len(population))
    return tuple(int(population[int(position)]) for position in permutation)


def _order_sha256(
    metadata: Sequence[PairMetadata], indices: Sequence[int]
) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(metadata[int(index)].pair_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validation_order_contract(stage: str, seed: int) -> Dict[str, object]:
    if stage not in {"coarse", "matching", "classify"} or seed < 0:
        raise ValueError("invalid validation-order contract arguments")
    shuffled = stage == "coarse"
    return {
        "schema_version": "rachel-shreddingnet-stage-validation-order/1.0",
        "stage": stage,
        "official_release_validation_loader_shuffle": shuffled,
        "adapter_order_algorithm": (
            "torch_randperm_isolated_epoch_generator"
            if shuffled
            else "manifest_order_identity"
        ),
        "formal_seed": seed,
        "epoch_seed_derivation": (
            "formal_seed_plus_2000033_times_epoch_one_based"
            if shuffled
            else None
        ),
        "drop_last": False,
    }


def _stage_validation_order(
    stage: str,
    population: Sequence[int],
    metadata: Sequence[PairMetadata],
    seed: int,
    epoch_one_based: int,
) -> Tuple[Tuple[int, ...], Dict[str, object]]:
    """Freeze the release loader's stage-specific validation ordering."""

    if not population or epoch_one_based <= 0:
        raise ValueError("invalid validation-order arguments")
    contract = _validation_order_contract(stage, seed)
    order_seed: Optional[int]
    if stage == "coarse":
        order_seed = seed + COARSE_VALIDATION_EPOCH_SEED_MULTIPLIER * epoch_one_based
        generator = torch.Generator()
        generator.manual_seed(order_seed)
        positions = tuple(
            int(value)
            for value in torch.randperm(len(population), generator=generator).tolist()
        )
    else:
        order_seed = None
        positions = tuple(range(len(population)))
    indices = tuple(int(population[position]) for position in positions)
    receipt = {
        "stage": stage,
        "epoch_one_based": epoch_one_based,
        "validation_loader_shuffle": stage == "coarse",
        "validation_order_seed": order_seed,
        "validation_pair_order_sha256": _order_sha256(metadata, indices),
        "validation_permutation_sha256": _canonical_sha256(
            {
                "schema_version": (
                    "rachel-shreddingnet-validation-permutation/1.0"
                ),
                "stage": stage,
                "epoch_one_based": epoch_one_based,
                "population_size": len(population),
                "order_seed": order_seed,
                "positions": list(positions),
            }
        ),
        "validation_order_contract_sha256": _canonical_sha256(contract),
    }
    return indices, receipt


_VALIDATION_ORDER_RECEIPT_FIELDS = (
    "stage",
    "epoch_one_based",
    "validation_loader_shuffle",
    "validation_order_seed",
    "validation_pair_order_sha256",
    "validation_permutation_sha256",
    "validation_order_contract_sha256",
)


def _validation_order_receipt_from_history(
    history: Sequence[Mapping[str, object]], epoch_zero_based: int
) -> Dict[str, object]:
    if not 0 <= epoch_zero_based < len(history):
        raise BenchmarkContractError("winner validation epoch is outside history")
    row = history[epoch_zero_based]
    if not isinstance(row, Mapping):
        raise BenchmarkContractError("stage history row is malformed")
    return {field: row.get(field) for field in _VALIDATION_ORDER_RECEIPT_FIELDS}


def _validation_order_receipt_structure_matches(
    *, stage: str, epoch_zero_based: int, recipe: ReleaseRecipe, receipt: object
) -> bool:
    if not isinstance(receipt, Mapping) or epoch_zero_based < 0:
        return False
    expected_seed = (
        recipe.seed
        + COARSE_VALIDATION_EPOCH_SEED_MULTIPLIER * (epoch_zero_based + 1)
        if stage == "coarse"
        else None
    )
    if (
        set(receipt) != set(_VALIDATION_ORDER_RECEIPT_FIELDS)
        or receipt.get("stage") != stage
        or receipt.get("epoch_one_based") != epoch_zero_based + 1
        or receipt.get("validation_loader_shuffle") != (stage == "coarse")
        or receipt.get("validation_order_seed") != expected_seed
        or receipt.get("validation_order_contract_sha256")
        != _canonical_sha256(_validation_order_contract(stage, recipe.seed))
    ):
        return False
    for field in (
        "validation_pair_order_sha256",
        "validation_permutation_sha256",
    ):
        value = receipt.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            return False
    return True


def _validate_stage_history_orders(
    *,
    stage: str,
    history: Sequence[Mapping[str, object]],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    train_metadata: Sequence[PairMetadata],
    val_metadata: Sequence[PairMetadata],
    seed: int,
) -> None:
    """Strictly replay all persisted train and validation permutations."""

    for epoch, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise BenchmarkContractError(stage + " history row is malformed")
        if row.get("epoch") != epoch or row.get("epoch_one_based") != epoch + 1:
            raise BenchmarkContractError(stage + " history epoch differs")
        expected_train = _epoch_order_indices(train_indices, seed, epoch + 1)
        if row.get("training_pair_order_sha256") != _order_sha256(
            train_metadata, expected_train
        ):
            raise BenchmarkContractError(stage + " training order binding differs")
        _, expected_validation = _stage_validation_order(
            stage, val_indices, val_metadata, seed, epoch + 1
        )
        observed_validation = {
            field: row.get(field) for field in _VALIDATION_ORDER_RECEIPT_FIELDS
        }
        if observed_validation != expected_validation:
            raise BenchmarkContractError(stage + " validation order binding differs")


def _amp_enabled(device: torch.device, runtime: RuntimeBatchConfig) -> bool:
    return bool(runtime.amp and device.type == "cuda")


def _buffer_snapshot(model: nn.Module) -> Dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def _restore_buffers(model: nn.Module, snapshot: Mapping[str, Tensor]) -> None:
    observed = dict(model.named_buffers())
    if set(observed) != set(snapshot):
        raise BenchmarkContractError("model buffer set changed during gradient cache")
    with torch.no_grad():
        for name, value in snapshot.items():
            observed[name].copy_(value)


def _coarse_features(
    model: ReleasedFragmentEncoder, batch: RachelBatch, device: torch.device
) -> Tuple[Tensor, Tensor]:
    mask_a, points_a, valid_a = _side_inputs(batch, "a", device)
    mask_b, points_b, valid_b = _side_inputs(batch, "b", device)
    return (
        model(mask_a, points_a, valid_a),
        model(mask_b, points_b, valid_b),
    )


def _coarse_metrics(
    feature_a: Tensor,
    feature_b: Tensor,
    batch: RachelBatch,
    recipe: ReleaseRecipe,
) -> Tuple[Tensor, Dict[str, float]]:
    total, cross = released_coarse_loss(
        feature_a.float(),
        feature_b.float(),
        batch.fragment_a_tokens,
        batch.fragment_b_tokens,
        recipe.infonce_temperature,
    )
    similarity = F.normalize(feature_a.detach().float(), dim=-1) @ F.normalize(
        feature_b.detach().float(), dim=-1
    ).T
    rank = torch.argsort(similarity, dim=1, descending=True)
    diagonal = torch.arange(len(batch.pair_ids), device=rank.device)
    top1 = float((rank[:, 0] == diagonal).float().mean().cpu())
    top5 = float(
        (rank[:, : min(5, rank.shape[1])] == diagonal[:, None])
        .any(dim=1)
        .float()
        .mean()
        .cpu()
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "infor_loss": float(cross.detach().cpu()),
        "top1_recall": top1,
        "top5_recall": top5,
    }


def _matching_cell_counts(batch: RachelBatch) -> Tuple[int, int]:
    valid_a, valid_b = supervised_contour_masks(batch)
    cells = int(np.sum(valid_a.sum(axis=1) * valid_b.sum(axis=1)))
    positives = int(np.count_nonzero(np.asarray(batch.target_a) >= 0))
    if cells <= 0 or positives <= 0 or positives > cells:
        raise BenchmarkContractError("fine effective batch target counts are invalid")
    return cells, positives


def _classification_metrics(score: Tensor, truth: Tensor) -> Dict[str, float]:
    predicted = score >= 0.5
    target = truth >= 0.5
    tp = int((predicted & target).sum().item())
    fp = int((predicted & ~target).sum().item())
    fn = int((~predicted & target).sum().item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    loss = F.binary_cross_entropy(score.float(), truth.float())
    return {
        "loss": float(loss.detach().cpu()),
        "accuracy": float((predicted == target).float().mean().cpu()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _classifier_outputs(
    model: ReleasedClassifyStage,
    batch: RachelBatch,
    device: torch.device,
    recipe: ReleaseRecipe,
) -> Tuple[Tensor, Tensor]:
    mask_a, points_a, valid_a = _side_inputs(batch, "a", device)
    mask_b, points_b, valid_b = _side_inputs(batch, "b", device)
    score, _, _ = model(
        mask_a,
        mask_b,
        points_a,
        points_b,
        valid_a,
        valid_b,
        recipe.correspondence_threshold,
    )
    truth = torch.as_tensor(batch.labels, dtype=torch.float32, device=device)
    return score, truth


def _train_effective_batch(
    *,
    stage: str,
    model: nn.Module,
    batch: RachelBatch,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    recipe: ReleaseRecipe,
    runtime: RuntimeBatchConfig,
    device: torch.device,
) -> Dict[str, float]:
    """One optimizer update at the official effective-batch exposure."""

    chunks = _microbatches(batch, runtime.microbatch(stage))
    amp = _amp_enabled(device, runtime)
    optimizer.zero_grad(set_to_none=True)
    if stage == "coarse":
        assert isinstance(model, ReleasedFragmentEncoder)
        # Gradient-cache: preserve the complete effective-batch InfoNCE
        # denominator without retaining encoder activations for all fragments.
        buffers = _buffer_snapshot(model)
        cached_a: List[Tensor] = []
        cached_b: List[Tensor] = []
        with torch.no_grad():
            for chunk in chunks:
                with torch.cuda.amp.autocast(enabled=amp):
                    first, second = _coarse_features(model, chunk, device)
                cached_a.append(first.detach())
                cached_b.append(second.detach())
        leaf_a = torch.cat(cached_a, dim=0).float().detach().requires_grad_(True)
        leaf_b = torch.cat(cached_b, dim=0).float().detach().requires_grad_(True)
        loss, metrics = _coarse_metrics(leaf_a, leaf_b, batch, recipe)
        scaler.scale(loss).backward()
        if leaf_a.grad is None or leaf_b.grad is None:
            raise BenchmarkContractError("coarse gradient cache produced no feature grad")
        gradients_a = leaf_a.grad.detach()
        gradients_b = leaf_b.grad.detach()
        _restore_buffers(model, buffers)
        offset = 0
        for chunk in chunks:
            with torch.cuda.amp.autocast(enabled=amp):
                first, second = _coarse_features(model, chunk, device)
            stop = offset + len(chunk.pair_ids)
            surrogate = (
                first.float() * gradients_a[offset:stop]
            ).sum() + (
                second.float() * gradients_b[offset:stop]
            ).sum()
            surrogate.backward()
            offset = stop
    elif stage == "matching":
        assert isinstance(model, ReleasedFineMatcher)
        counts = [_matching_cell_counts(chunk) for chunk in chunks]
        total_cells = sum(value[0] for value in counts)
        total_positives = sum(value[1] for value in counts)
        loss_value = 0.0
        positive_value = 0.0
        for chunk, (cells, positives) in zip(chunks, counts):
            with torch.cuda.amp.autocast(enabled=amp):
                loss, metrics = _matching_batch(model, chunk, device, recipe)
            scaler.scale(loss * (cells / total_cells)).backward()
            loss_value += float(metrics["loss"]) * cells / total_cells
            positive_value += (
                float(metrics["positive_loss"]) * positives / total_positives
            )
        metrics = {"loss": loss_value, "positive_loss": positive_value}
    else:
        assert stage == "classify" and isinstance(model, ReleasedClassifyStage)
        scores = []
        truths = []
        total = len(batch.pair_ids)
        for chunk in chunks:
            with torch.cuda.amp.autocast(enabled=amp):
                score, truth = _classifier_outputs(model, chunk, device, recipe)
            # BCE on probabilities is deliberately evaluated in float32 outside
            # autocast; PyTorch rejects BCELoss while autocast is active.
            loss = F.binary_cross_entropy(score.float(), truth.float())
            scaler.scale(loss * (len(chunk.pair_ids) / total)).backward()
            scores.append(score.detach())
            truths.append(truth.detach())
        metrics = _classification_metrics(torch.cat(scores), torch.cat(truths))
    scaler.step(optimizer)
    scaler.update()
    return metrics


def _validate_effective_batch(
    *,
    stage: str,
    model: nn.Module,
    batch: RachelBatch,
    recipe: ReleaseRecipe,
    runtime: RuntimeBatchConfig,
    device: torch.device,
) -> Dict[str, float]:
    chunks = _microbatches(batch, runtime.microbatch(stage))
    amp = _amp_enabled(device, runtime)
    if stage == "coarse":
        assert isinstance(model, ReleasedFragmentEncoder)
        first_all = []
        second_all = []
        for chunk in chunks:
            with torch.cuda.amp.autocast(enabled=amp):
                first, second = _coarse_features(model, chunk, device)
            first_all.append(first)
            second_all.append(second)
        _, metrics = _coarse_metrics(
            torch.cat(first_all), torch.cat(second_all), batch, recipe
        )
        return metrics
    if stage == "matching":
        assert isinstance(model, ReleasedFineMatcher)
        counts = [_matching_cell_counts(chunk) for chunk in chunks]
        total_cells = sum(value[0] for value in counts)
        total_positives = sum(value[1] for value in counts)
        loss_value = 0.0
        positive_value = 0.0
        for chunk, (cells, positives) in zip(chunks, counts):
            with torch.cuda.amp.autocast(enabled=amp):
                _, metrics = _matching_batch(model, chunk, device, recipe)
            loss_value += float(metrics["loss"]) * cells / total_cells
            positive_value += (
                float(metrics["positive_loss"]) * positives / total_positives
            )
        return {"loss": loss_value, "positive_loss": positive_value}
    assert stage == "classify" and isinstance(model, ReleasedClassifyStage)
    scores = []
    truths = []
    for chunk in chunks:
        with torch.cuda.amp.autocast(enabled=amp):
            score, truth = _classifier_outputs(model, chunk, device, recipe)
        scores.append(score)
        truths.append(truth)
    return _classification_metrics(torch.cat(scores), torch.cat(truths))


def _run_stage(
    *,
    stage: str,
    output_root: Path,
    train_dataset: RachelPairDataset,
    val_dataset: RachelPairDataset,
    train_metadata: Sequence[PairMetadata],
    val_metadata: Sequence[PairMetadata],
    recipe: ReleaseRecipe,
    runtime: RuntimeBatchConfig,
    device: torch.device,
    num_workers: int,
    resume: bool,
    adapter_identity: Mapping[str, str],
    dataset_binding: Mapping[str, object],
    matching_winner_path: Optional[Path] = None,
) -> Dict[str, object]:
    _seed_everything(recipe.seed)
    settings = _stage_recipe(stage, recipe)
    stage_root = output_root / "stages" / stage
    _assert_no_symlink_components(stage_root, name=stage + " stage")
    _ensure_directory_tree(stage_root, name=stage + " stage")
    _assert_no_symlink_components(stage_root, name=stage + " stage")
    progress_path = stage_root / "progress.pt"
    winner_path = stage_root / "winner.pt"
    completion_path = stage_root / "completion.json"
    for artifact_path, artifact_name in (
        (progress_path, stage + " progress"),
        (winner_path, stage + " winner"),
        (completion_path, stage + " completion"),
    ):
        _assert_no_symlink_components(artifact_path, name=artifact_name)
    if matching_winner_path is not None:
        _assert_no_symlink_components(
            matching_winner_path, name="matching winner upstream"
        )
    matching_winner_sha256 = (
        _sha256_file(matching_winner_path)
        if matching_winner_path is not None
        else None
    )
    train_indices = [
        index
        for index, row in enumerate(train_metadata)
        if row.label or stage == "classify"
    ]
    val_indices = [
        index for index, row in enumerate(val_metadata) if row.label or stage == "classify"
    ]
    if not train_indices or not val_indices:
        raise BenchmarkContractError(stage + " train/validation population is empty")
    validation_order_contract = _validation_order_contract(stage, recipe.seed)
    if completion_path.is_file():
        if not resume:
            raise BenchmarkContractError(stage + " completion appeared during fresh run")
        completion = _read_verified_json(completion_path, stage + " completion")
        if (
            completion.get("schema_version") != SCHEMA_VERSION
            or completion.get("checkpoint_kind")
            != "rachel_shreddingnet_{}_completion".format(stage)
            or completion.get("method_id") != METHOD_ID
            or completion.get("stage") != stage
            or completion.get("adapter_identity") != dict(adapter_identity)
            or completion.get("dataset_binding") != dict(dataset_binding)
            or completion.get("train_val_provenance")
            != _train_val_provenance(adapter_identity, dataset_binding)
            or completion.get("validation_order_contract")
            != validation_order_contract
        ):
            raise BenchmarkContractError(stage + " completion identity differs")
        if winner_path.is_symlink() or not winner_path.is_file():
            raise BenchmarkContractError(stage + " completed winner is missing")
        if _sha256_file(winner_path) != completion.get("winner_sha256"):
            raise BenchmarkContractError(stage + " completed winner hash differs")
        if completion.get("runtime_batch_config") != asdict(runtime):
            raise BenchmarkContractError(stage + " completed runtime batch differs")
        if completion.get("settings") != settings:
            raise BenchmarkContractError(stage + " completed release settings differ")
        completed_epoch = completion.get("winner_epoch")
        completed_value = completion.get("winner_value")
        if (
            not isinstance(completed_epoch, int)
            or isinstance(completed_value, bool)
            or not isinstance(completed_value, (int, float))
            or not math.isfinite(float(completed_value))
        ):
            raise BenchmarkContractError(stage + " completion winner fields differ")
        completed_history = completion.get("history")
        if (
            not isinstance(completed_history, list)
            or len(completed_history) != int(settings["epochs"])
        ):
            raise BenchmarkContractError(stage + " completion history differs")
        _validate_stage_history_orders(
            stage=stage,
            history=completed_history,
            train_indices=train_indices,
            val_indices=val_indices,
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            seed=recipe.seed,
        )
        winner_validation_order = _validation_order_receipt_from_history(
            completed_history, completed_epoch
        )
        if completion.get("winner_validation_order") != winner_validation_order:
            raise BenchmarkContractError(
                stage + " completion winner validation order differs"
            )
        winner_checkpoint = _torch_load(winner_path, device)
        if not _winner_identity_matches(
            winner_checkpoint,
            stage=stage,
            best_epoch=completed_epoch,
            best_value=float(completed_value),
            settings=settings,
            recipe=recipe,
            runtime=runtime,
            matching_winner_sha256=matching_winner_sha256,
            adapter_identity=adapter_identity,
            dataset_binding=dataset_binding,
            validation_order_receipt=winner_validation_order,
        ):
            raise BenchmarkContractError(stage + " completed winner identity differs")
        return completion
    if not resume and (
        _path_lexists(progress_path) or _path_lexists(winner_path)
    ):
        raise BenchmarkContractError(stage + " checkpoint appeared during fresh run")
    if resume and not _path_lexists(progress_path) and _path_lexists(winner_path):
        raise BenchmarkContractError(stage + " orphan winner exists without progress")

    model = _stage_model(stage, recipe).to(device)
    if stage == "classify":
        if matching_winner_path is None:
            raise BenchmarkContractError("classifier stage requires matching winner")
        assert isinstance(model, ReleasedClassifyStage)
        matching_checkpoint = _load_winner_state(
            matching_winner_path,
            stage="matching",
            recipe=recipe,
            runtime=runtime,
            adapter_identity=adapter_identity,
            dataset_binding=dataset_binding,
            device=device,
        )
        model.matcher.load_state_dict(
            matching_checkpoint["model_state_dict"], strict=True
        )
        model.matcher.requires_grad_(False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=recipe.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(settings["epochs"])
    )
    scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled(device, runtime))
    start_epoch = 0
    mode = str(settings["selection_mode"])
    best_value = math.inf if mode == "min" else -math.inf
    best_epoch = -1
    history: List[Dict[str, object]] = []
    if resume and progress_path.is_file():
        progress = _torch_load(progress_path, device)
        if (
            progress.get("schema_version") != SCHEMA_VERSION
            or progress.get("checkpoint_kind")
            != _stage_checkpoint_kind(stage, "progress")
            or progress.get("stage") != stage
            or progress.get("method_id") != METHOD_ID
        ):
            raise BenchmarkContractError(stage + " progress identity differs")
        if progress.get("recipe") != asdict(recipe):
            raise BenchmarkContractError(stage + " progress recipe differs")
        if progress.get("matching_winner_sha256") != matching_winner_sha256:
            raise BenchmarkContractError(stage + " progress upstream winner differs")
        if (
            progress.get("adapter_identity") != dict(adapter_identity)
            or progress.get("dataset_binding") != dict(dataset_binding)
            or progress.get("train_val_provenance")
            != _train_val_provenance(adapter_identity, dataset_binding)
            or progress.get("validation_order_contract")
            != validation_order_contract
        ):
            raise BenchmarkContractError(stage + " progress provenance differs")
        model.load_state_dict(progress["model_state_dict"], strict=True)
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        scheduler.load_state_dict(progress["scheduler_state_dict"])
        if progress.get("runtime_batch_config") != asdict(runtime):
            raise BenchmarkContractError(stage + " progress runtime batch differs")
        scaler.load_state_dict(progress.get("scaler_state_dict", {}))
        start_epoch = int(progress["epoch_completed"]) + 1
        best_value = float(progress["best_value"])
        best_epoch = int(progress["best_epoch"])
        persisted_history = progress.get("history")
        if not isinstance(persisted_history, list):
            raise BenchmarkContractError(stage + " progress history differs")
        history = list(persisted_history)
        if (
            best_epoch < 0
            or start_epoch < 1
            or len(history) != start_epoch
            or int(history[-1].get("epoch", -1)) != start_epoch - 1
        ):
            raise BenchmarkContractError(stage + " progress epoch/history differs")
        _validate_stage_history_orders(
            stage=stage,
            history=history,
            train_indices=train_indices,
            val_indices=val_indices,
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            seed=recipe.seed,
        )
        winner_validation_order = _validation_order_receipt_from_history(
            history, best_epoch
        )
        if progress.get("best_validation_order") != winner_validation_order:
            raise BenchmarkContractError(stage + " progress best order differs")
        repaired = _repair_or_verify_progress_winner(
            stage=stage,
            progress=progress,
            winner_path=winner_path,
            settings=settings,
            recipe=recipe,
            runtime=runtime,
            device=device,
            matching_winner_sha256=matching_winner_sha256,
            adapter_identity=adapter_identity,
            dataset_binding=dataset_binding,
            validation_order_receipt=winner_validation_order,
        )
        if repaired:
            print(
                json.dumps(
                    {
                        "event": "rachel_shreddingnet_winner_repaired",
                        "stage": stage,
                        "best_epoch": best_epoch,
                        "best_value": best_value,
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
    elif progress_path.exists():
        raise BenchmarkContractError(stage + " progress exists without --resume")

    for epoch in range(start_epoch, int(settings["epochs"])):
        epoch_train_indices = _epoch_order_indices(
            train_indices, recipe.seed, epoch + 1
        )
        epoch_val_indices, validation_order_receipt = _stage_validation_order(
            stage, val_indices, val_metadata, recipe.seed, epoch + 1
        )
        train_loader = _loader(
            train_dataset,
            epoch_train_indices,
            batch_size=int(settings["batch_size"]),
            shuffle=False,
            drop_last=False,
            seed=recipe.seed + epoch + 1,
            num_workers=num_workers,
            device=device,
        )
        val_loader = _loader(
            val_dataset,
            epoch_val_indices,
            batch_size=int(settings["batch_size"]),
            shuffle=False,
            drop_last=False,
            seed=recipe.seed,
            num_workers=num_workers,
            device=device,
        )
        model.train()
        train_values = []
        for batch in train_loader:
            metrics = _train_effective_batch(
                stage=stage,
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                recipe=recipe,
                runtime=runtime,
                device=device,
            )
            train_values.append((len(batch.pair_ids), metrics))
        scheduler.step()

        model.eval()
        val_values = []
        recall_scores, recall_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                metrics = _validate_effective_batch(
                    stage=stage,
                    model=model,
                    batch=batch,
                    recipe=recipe,
                    runtime=runtime,
                    device=device,
                )
                val_values.append((len(batch.pair_ids), metrics))
                if stage == "classify" and recipe.selection_metric == "recall95_precision":
                    # Full-VAL operating points cannot be averaged over batches.
                    # Run target-blind scoring at the registered microbatch size.
                    for start in range(0, len(batch.pair_ids), runtime.classify_microbatch):
                        chunk = _slice_batch(batch, start, start + runtime.classify_microbatch)
                        with torch.cuda.amp.autocast(enabled=_amp_enabled(device, runtime)):
                            score, _ = _classifier_outputs(model, chunk, device, recipe)
                        recall_scores.extend(score.detach().float().cpu().tolist())
                        recall_labels.extend(np.asarray(chunk.labels, bool).tolist())
        train_metrics = _aggregate_epoch(train_values)
        val_metrics = _aggregate_epoch(val_values)
        if recall_scores:
            val_metrics.update(_recall_validation_metrics(recall_labels, recall_scores))
        selected_value = float(val_metrics[str(settings["selection_metric"])])
        if not math.isfinite(selected_value):
            raise BenchmarkContractError(stage + " validation selection metric is non-finite")
        improved = _selection_improved(selected_value, best_value, mode)
        if recall_scores:
            improved = _recall_selection_improved(val_metrics, best_value, best_epoch, history)
        winner_payload: Optional[Dict[str, object]] = None
        if improved:
            best_value = selected_value
            best_epoch = epoch
            winner_payload = _winner_checkpoint_payload(
                stage=stage,
                epoch=epoch,
                settings=settings,
                selection_value=selected_value,
                recipe=recipe,
                runtime=runtime,
                model_state_dict=model.state_dict(),
                matching_winner_sha256=matching_winner_sha256,
                adapter_identity=adapter_identity,
                dataset_binding=dataset_binding,
                validation_order_receipt=validation_order_receipt,
            )
        epoch_row: Dict[str, object] = {
            "epoch": epoch,
            "epoch_one_based": epoch + 1,
            "training_pair_order_sha256": _order_sha256(
                train_metadata, epoch_train_indices
            ),
            **validation_order_receipt,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate_after_scheduler_step": scheduler.get_last_lr()[0],
            "best_epoch": best_epoch,
            "best_value": best_value,
        }
        history.append(epoch_row)
        _torch_save_atomic(
            progress_path,
            {
                "schema_version": SCHEMA_VERSION,
                "checkpoint_kind": _stage_checkpoint_kind(stage, "progress"),
                "method_id": METHOD_ID,
                "stage": stage,
                "epoch_completed": epoch,
                "best_epoch": best_epoch,
                "best_value": best_value,
                "history": history,
                "recipe": asdict(recipe),
                "matching_winner_sha256": matching_winner_sha256,
                "adapter_identity": dict(adapter_identity),
                "dataset_binding": dict(dataset_binding),
                "train_val_provenance": _train_val_provenance(
                    adapter_identity, dataset_binding
                ),
                "validation_order_contract": validation_order_contract,
                "best_validation_order": _validation_order_receipt_from_history(
                    history, best_epoch
                ),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "runtime_batch_config": asdict(runtime),
            },
        )
        # Progress is committed first. If the process fails before the new
        # winner rename, resume can reconstruct that winner from this exact
        # epoch state. The reverse ordering cannot reconstruct optimizer state
        # from a winner-only file.
        if winner_payload is not None:
            _torch_save_atomic(winner_path, winner_payload)
        print(
            json.dumps(
                {
                    "event": "rachel_shreddingnet_epoch",
                    "stage": stage,
                    **epoch_row,
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )

    if best_epoch < 0 or winner_path.is_symlink() or not winner_path.is_file():
        raise BenchmarkContractError(stage + " did not produce a validation winner")
    winner_validation_order = _validation_order_receipt_from_history(
        history, best_epoch
    )
    final_winner = _torch_load(winner_path, device)
    if not _winner_identity_matches(
        final_winner,
        stage=stage,
        best_epoch=best_epoch,
        best_value=best_value,
        settings=settings,
        recipe=recipe,
        runtime=runtime,
        matching_winner_sha256=matching_winner_sha256,
        adapter_identity=adapter_identity,
        dataset_binding=dataset_binding,
        validation_order_receipt=winner_validation_order,
    ):
        raise BenchmarkContractError(stage + " final winner identity differs")
    completion = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": "rachel_shreddingnet_{}_completion".format(stage),
        "status": "complete_train_val_stage",
        "method_id": METHOD_ID,
        "stage": stage,
        "settings": settings,
        "runtime_batch_config": asdict(runtime),
        "effective_batch_size": int(settings["batch_size"]),
        "gpu_microbatch_size": runtime.microbatch(stage),
        "nominal_gradient_accumulation_steps": int(
            math.ceil(int(settings["batch_size"]) / runtime.microbatch(stage))
        ),
        "optimizer_steps_use_release_effective_exposure": True,
        "coarse_full_effective_infonce_denominator_preserved": stage == "coarse",
        "microbatch_batchnorm_numerically_identical_to_monolithic_release_batch": (
            runtime.microbatch(stage) >= int(settings["batch_size"])
        ),
        "amp_enabled": _amp_enabled(device, runtime),
        "formal_exposure_seed": recipe.seed,
        "official_release_seed": recipe.official_release_seed,
        "winner_epoch": best_epoch,
        "winner_value": best_value,
        "winner_path": winner_path.relative_to(output_root).as_posix(),
        "winner_sha256": _sha256_file(winner_path),
        "completed_epochs": int(settings["epochs"]),
        "epoch_zero_checkpoint_omission_fixed": True,
        "history": history,
        "validation_order_contract": validation_order_contract,
        "winner_validation_order": winner_validation_order,
        "sealed_test_or_real_opened": False,
        "adapter_identity": dict(adapter_identity),
        "dataset_binding": dict(dataset_binding),
        "train_val_provenance": _train_val_provenance(
            adapter_identity, dataset_binding
        ),
        "dual_softmax_probability_compute_dtype": "float32",
        "focal_logarithm_epsilon_semantics": (
            "log_p_plus_1e-9_and_log_1_minus_p_plus_1e-9"
        ),
    }
    _write_json_new_atomic(
        completion_path, completion, name=stage + " completion receipt"
    )
    return json.loads(completion_path.read_text(encoding="utf-8"))


def _read_verified_json(path: Path, name: str) -> Dict[str, object]:
    path = _lexical_absolute_path(path)
    _assert_no_symlink_components(path, name=name)
    if path.is_symlink() or not path.is_file():
        raise BenchmarkContractError(name + " is missing or symlinked")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkContractError("cannot read " + name) from error
    if not isinstance(value, dict):
        raise BenchmarkContractError(name + " root must be an object")
    if value.get("content_sha256") != _canonical_sha256(value):
        raise BenchmarkContractError(name + " content hash differs")
    return value


def _canonical_jsonl_payload(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = _prepare_fresh_file_path(path, name="JSONL output")
    _ensure_directory_tree(path.parent, name="JSONL output parent")
    temporary = path.with_name(
        path.name + ".{}.{}.tmp".format(os.getpid(), time.time_ns())
    )
    descriptor = _open_exclusive_temporary(temporary, name="JSONL output")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_jsonl_payload(rows))
        _assert_no_symlink_components(path, name="JSONL output")
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise BenchmarkContractError("JSONL output appeared during publish") from error
    finally:
        if _path_lexists(temporary):
            os.unlink(temporary)


def _publish_or_reuse_exact_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    resume: bool,
    name: str,
) -> Dict[str, object]:
    """Publish once, or resume only from byte-exact recomputed JSONL."""

    path = _lexical_absolute_path(path)
    _assert_no_symlink_components(path, name=name)
    expected_payload = _canonical_jsonl_payload(rows)
    expected_sha256 = hashlib.sha256(expected_payload).hexdigest()
    if _path_lexists(path):
        if not resume:
            raise BenchmarkContractError(name + " already exists outside resume")
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode):
            raise BenchmarkContractError(name + " is not a regular file")
        observed_payload = path.read_bytes()
        observed_sha256 = hashlib.sha256(observed_payload).hexdigest()
        if observed_sha256 != expected_sha256 or observed_payload != expected_payload:
            raise BenchmarkContractError(
                name + " differs from current frozen-winner recomputation"
            )
        try:
            observed_rows = [
                json.loads(line)
                for line in observed_payload.decode("utf-8").splitlines()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BenchmarkContractError(name + " cannot be decoded completely") from error
        if observed_rows != [dict(row) for row in rows]:
            raise BenchmarkContractError(name + " row content differs")
        return {
            "sha256": observed_sha256,
            "row_count": len(observed_rows),
            "reused_after_verified_resume": True,
        }
    _write_jsonl_atomic(path, rows)
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise BenchmarkContractError(name + " post-publish SHA-256 differs")
    return {
        "sha256": observed_sha256,
        "row_count": len(rows),
        "reused_after_verified_resume": False,
    }


def _safe_bundle_file(
    root: Path, relative: object, expected_sha256: object, name: str
) -> Path:
    root = _lexical_absolute_path(root)
    _assert_no_symlink_components(root, name="bundle root")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise BenchmarkContractError(name + " relative path is invalid")
    logical = PurePosixPath(relative)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise BenchmarkContractError(name + " relative path is unsafe")
    candidate = root.joinpath(*logical.parts)
    current = root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise BenchmarkContractError(name + " path contains a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise BenchmarkContractError(name + " path is missing or escapes bundle") from error
    if not resolved.is_file() or _sha256_file(resolved) != expected_sha256:
        raise BenchmarkContractError(name + " SHA-256 differs")
    return resolved


def _load_winner_state(
    path: Path,
    *,
    stage: str,
    recipe: ReleaseRecipe,
    runtime: RuntimeBatchConfig,
    adapter_identity: Mapping[str, str],
    dataset_binding: Mapping[str, object],
    device: torch.device,
) -> Mapping[str, object]:
    checkpoint = _torch_load(path, device)
    settings = _stage_recipe(stage, recipe)
    epoch = checkpoint.get("epoch")
    if (
        checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("checkpoint_kind")
        != _stage_checkpoint_kind(stage, "winner")
        or checkpoint.get("method_id") != METHOD_ID
        or checkpoint.get("stage") != stage
        or checkpoint.get("recipe") != asdict(recipe)
        or checkpoint.get("runtime_batch_config") != asdict(runtime)
        or checkpoint.get("adapter_identity") != dict(adapter_identity)
        or checkpoint.get("dataset_binding") != dict(dataset_binding)
        or checkpoint.get("train_val_provenance")
        != _train_val_provenance(adapter_identity, dataset_binding)
        or checkpoint.get("selection_metric") != settings["selection_metric"]
        or checkpoint.get("selection_mode") != settings["selection_mode"]
        or type(epoch) is not int  # noqa: E721
        or isinstance(checkpoint.get("selection_value"), bool)
        or not isinstance(checkpoint.get("selection_value"), (int, float))
        or not math.isfinite(float(checkpoint["selection_value"]))
        or not isinstance(checkpoint.get("model_state_dict"), Mapping)
        or checkpoint.get("validation_order_contract")
        != _validation_order_contract(stage, recipe.seed)
        or not _validation_order_receipt_structure_matches(
            stage=stage,
            epoch_zero_based=epoch if type(epoch) is int else -1,  # noqa: E721
            recipe=recipe,
            receipt=checkpoint.get("winner_validation_order"),
        )
    ):
        raise BenchmarkContractError(stage + " winner contract differs")
    return checkpoint


def _restore_three_stage_models(
    *,
    output_root: Path,
    recipe: ReleaseRecipe,
    runtime: RuntimeBatchConfig,
    adapter_identity: Mapping[str, str],
    dataset_binding: Mapping[str, object],
    device: torch.device,
) -> Tuple[ReleasedFragmentEncoder, ReleasedClassifyStage]:
    coarse_path = output_root / "stages" / "coarse" / "winner.pt"
    matching_path = output_root / "stages" / "matching" / "winner.pt"
    classify_path = output_root / "stages" / "classify" / "winner.pt"
    coarse_checkpoint = _load_winner_state(
        coarse_path,
        stage="coarse",
        recipe=recipe,
        runtime=runtime,
        adapter_identity=adapter_identity,
        dataset_binding=dataset_binding,
        device=device,
    )
    matching_checkpoint = _load_winner_state(
        matching_path,
        stage="matching",
        recipe=recipe,
        runtime=runtime,
        adapter_identity=adapter_identity,
        dataset_binding=dataset_binding,
        device=device,
    )
    classify_checkpoint = _load_winner_state(
        classify_path,
        stage="classify",
        recipe=recipe,
        runtime=runtime,
        adapter_identity=adapter_identity,
        dataset_binding=dataset_binding,
        device=device,
    )
    if classify_checkpoint.get("matching_winner_sha256") != _sha256_file(
        matching_path
    ):
        raise BenchmarkContractError("classifier winner/matching winner binding differs")
    coarse = ReleasedFragmentEncoder(recipe).to(device)
    coarse.load_state_dict(coarse_checkpoint["model_state_dict"], strict=True)
    classify = ReleasedClassifyStage(recipe).to(device)
    # Restore stage two first, then the complete stage-three winner.  This
    # verifies both contracts while retaining released classifier-stage BN
    # buffer evolution from the stage-three checkpoint.
    classify.matcher.load_state_dict(
        matching_checkpoint["model_state_dict"], strict=True
    )
    classify.load_state_dict(classify_checkpoint["model_state_dict"], strict=True)
    coarse.requires_grad_(False).eval()
    classify.requires_grad_(False).eval()
    return coarse, classify


def _validation_fingerprint(
    rows: Sequence[PairMetadata], manifest_sha256: str
) -> str:
    value = {
        "schema_version": "rachel-shreddingnet-validation-order/1.0",
        "method_id": METHOD_ID,
        "manifest_sha256": manifest_sha256,
        "ordered_rows": [
            {
                "ordinal": index,
                "pair_id": row.pair_id,
                "label": row.label,
                "cluster_id": row.cluster_id,
            }
            for index, row in enumerate(rows)
        ],
    }
    return _canonical_sha256(value)


def _score_validation_for_threshold(
    *,
    dataset: RachelPairDataset,
    metadata: Sequence[PairMetadata],
    coarse: ReleasedFragmentEncoder,
    classify: ReleasedClassifyStage,
    recipe: ReleaseRecipe,
    runtime: RuntimeBatchConfig,
    device: torch.device,
    num_workers: int,
) -> Tuple[Dict[str, object], ...]:
    loader = _loader(
        dataset,
        tuple(range(len(metadata))),
        batch_size=recipe.classify_batch_size,
        shuffle=False,
        drop_last=False,
        seed=recipe.seed,
        num_workers=num_workers,
        device=device,
    )
    by_id = {row.pair_id: row for row in metadata}
    output: List[Dict[str, object]] = []
    amp = _amp_enabled(device, runtime)
    with torch.no_grad():
        for effective_batch in loader:
            for batch in _microbatches(
                effective_batch, runtime.classify_microbatch
            ):
                with torch.cuda.amp.autocast(enabled=amp):
                    first, second = _coarse_features(coarse, batch, device)
                    coarse_score = F.cosine_similarity(
                        first.float(), second.float(), dim=-1
                    )
                    mask_a, points_a, valid_a = _side_inputs(batch, "a", device)
                    mask_b, points_b, valid_b = _side_inputs(batch, "b", device)
                    pair_score, _, _ = classify(
                        mask_a,
                        mask_b,
                        points_a,
                        points_b,
                        valid_a,
                        valid_b,
                        recipe.correspondence_threshold,
                    )
                coarse_values = coarse_score.detach().cpu().double().numpy()
                pair_values = pair_score.detach().cpu().double().numpy()
                for pair_id, coarse_value, pair_value in zip(
                    batch.pair_ids, coarse_values, pair_values
                ):
                    row = by_id.get(pair_id)
                    if row is None:
                        raise BenchmarkContractError(
                            "validation batch contains an unknown pair_id"
                        )
                    output.append(
                        {
                            "schema_version": "rachel-shreddingnet-val-score/1.0",
                            "method_id": METHOD_ID,
                            "ordinal": len(output),
                            "pair_id": pair_id,
                            "label": row.label,
                            "cluster_id": row.cluster_id,
                            "coarse_score": float(coarse_value),
                            "pair_score": float(pair_value),
                        }
                    )
    expected_ids = [row.pair_id for row in metadata]
    if [str(row["pair_id"]) for row in output] != expected_ids:
        raise BenchmarkContractError("validation score order differs from manifest")
    return tuple(output)


def _threshold_and_freeze(
    *,
    output_root: Path,
    dataset_audit: Mapping[str, object],
    official_audit: Mapping[str, object],
    val_dataset: RachelPairDataset,
    val_metadata: Sequence[PairMetadata],
    stage_receipts: Mapping[str, Mapping[str, object]],
    recipe: ReleaseRecipe,
    runtime: RuntimeBatchConfig,
    adapter_identity: Mapping[str, str],
    dataset_binding: Mapping[str, object],
    device: torch.device,
    num_workers: int,
    resume: bool,
) -> Dict[str, object]:
    freeze_path = output_root / "train_val_freeze.json"
    if _path_lexists(freeze_path):
        raise BenchmarkContractError("train_val_freeze.json already exists")
    coarse, classify = _restore_three_stage_models(
        output_root=output_root,
        recipe=recipe,
        runtime=runtime,
        adapter_identity=adapter_identity,
        dataset_binding=dataset_binding,
        device=device,
    )
    rows = _score_validation_for_threshold(
        dataset=val_dataset,
        metadata=val_metadata,
        coarse=coarse,
        classify=classify,
        recipe=recipe,
        runtime=runtime,
        device=device,
        num_workers=num_workers,
    )
    score_path = output_root / "validation_threshold_scores.jsonl"
    score_receipt = _publish_or_reuse_exact_jsonl(
        score_path,
        rows,
        resume=resume,
        name="validation threshold scores",
    )
    manifest_sha = str(dataset_audit["manifest_sha256"]["val"])
    validation_fingerprint = _validation_fingerprint(val_metadata, manifest_sha)
    model_config_sha = _canonical_sha256(
        {
            "method_id": METHOD_ID,
            "official_commit": OFFICIAL_COMMIT,
            "recipe": asdict(recipe),
            "runtime_batch_config": asdict(runtime),
            "adapter_identity": dict(adapter_identity),
            "dataset_binding": dict(dataset_binding),
            "train_val_provenance": _train_val_provenance(
                adapter_identity, dataset_binding
            ),
        }
    )
    aggregation_config = {
        "fit_method": "maximize_cluster_balanced_f1",
        "same_shared_fitter_as_primary_rachel_arm": True,
        "within_lineage_cluster": "source_image_name_lineage",
        "cross_lineage_negative_cluster": "sha256_of_sorted_lineage_pair",
        "tie_break": "f1_then_precision_then_recall_then_higher_threshold",
    }
    aggregation_config_sha = _canonical_sha256(aggregation_config)
    classifier_sha = str(stage_receipts["classify"]["winner_sha256"])
    threshold = fit_pairwise_threshold(
        [float(row["pair_score"]) for row in rows],
        [bool(row["label"]) for row in rows],
        [True] * len(rows),
        [str(row["cluster_id"]) for row in rows],
        source_split="validation",
        validation_fingerprint_sha256=validation_fingerprint,
        checkpoint_sha256=classifier_sha,
        model_config_sha256=model_config_sha,
        aggregation_config_sha256=aggregation_config_sha,
    )
    checkpoint_rows = {}
    for stage in ("coarse", "matching", "classify"):
        receipt = stage_receipts[stage]
        checkpoint_rows[stage] = {
            "path": receipt["winner_path"],
            "sha256": receipt["winner_sha256"],
            "checkpoint_kind": _stage_checkpoint_kind(stage, "winner"),
            "winner_epoch_zero_based": receipt["winner_epoch"],
            "winner_epoch_one_based": int(receipt["winner_epoch"]) + 1,
            "selection_metric": receipt["settings"]["selection_metric"],
            "selection_mode": receipt["settings"]["selection_mode"],
        }
    freeze = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "checkpoint_kind": FREEZE_CHECKPOINT_KIND,
        "status": "complete_train_val_frozen_no_test_or_real",
        "method_id": METHOD_ID,
        "official_commit": OFFICIAL_COMMIT,
        "recipe": asdict(recipe),
        "runtime_batch_config": asdict(runtime),
        "adapter_identity": dict(adapter_identity),
        "dataset_binding": dict(dataset_binding),
        "train_val_provenance": _train_val_provenance(
            adapter_identity, dataset_binding
        ),
        "checkpoints": checkpoint_rows,
        "threshold": {
            **threshold.to_dict(),
            "content_sha256": threshold.content_sha256,
            "score_file": score_path.relative_to(output_root).as_posix(),
            "score_file_sha256": score_receipt["sha256"],
            "score_file_row_count": score_receipt["row_count"],
            "score_file_reused_after_verified_resume": score_receipt[
                "reused_after_verified_resume"
            ],
            "validation_order_sha256": validation_fingerprint,
            "aggregation_config": aggregation_config,
            "native_release_secondary_threshold": recipe.pair_score_threshold,
            "threshold_used_for_stage_winner_selection": False,
            "fit_after_all_three_winners_fixed": True,
        },
        "dataset_audit": dict(dataset_audit),
        "official_source_audit": dict(official_audit),
        "stage_completion_receipts": {
            stage: {
                "path": "stages/{}/completion.json".format(stage),
                "content_sha256": stage_receipts[stage]["content_sha256"],
            }
            for stage in ("coarse", "matching", "classify")
        },
        "inference_contract": {
            "loader": "load_frozen_inference",
            "method_id": METHOD_ID,
            "inputs": [
                "pair_ids",
                "mask_a",
                "mask_b",
                "points_rc_a",
                "points_rc_b",
                "contour_valid_a",
                "contour_valid_b",
            ],
            "forbidden_forward_fields": [
                "labels",
                "target_a",
                "target_b",
                "translation_a_to_b_rc",
                "translation_a_to_b_xy_cartesian",
                "translation_valid",
                "lineage",
            ],
            "outputs": [
                "coarse_score",
                "pair_score",
                "correspondence_probability_optional",
                "correspondence_binary_optional",
                "translation_hat_rc",
                "translation_valid",
                "se2_translation_hat_rc_secondary",
                "se2_rotation_degrees_secondary",
            ],
        },
        "scope": {
            "mask_only": True,
            "rachel_n512": True,
            "upright_known_orientation": True,
            "train_val_only": True,
            "sealed_test_or_real_opened": False,
            "original_shreddingnet_reproduction": False,
            "global_assembly_performed": False,
        },
        "numeric_contract": {
            "dual_softmax_probability_compute_dtype": "float32",
            "amp_logits_promoted_before_mask_and_softmax": True,
            "focal_logarithm_epsilon_semantics": (
                "log_p_plus_1e-9_and_log_1_minus_p_plus_1e-9"
            ),
        },
    }
    _write_json_new_atomic(freeze_path, freeze, name="train/val freeze")
    return _read_verified_json(freeze_path, "train/val freeze")


class RachelShreddingNetInference:
    """Hash-verified, label-free inference over Rachel model inputs only."""

    def __init__(
        self,
        *,
        coarse: ReleasedFragmentEncoder,
        classify: ReleasedClassifyStage,
        recipe: ReleaseRecipe,
        runtime: RuntimeBatchConfig,
        pair_threshold: float,
        device: torch.device,
    ) -> None:
        self.coarse = coarse
        self.classify = classify
        self.recipe = recipe
        self.runtime = runtime
        self.pair_threshold = float(pair_threshold)
        self.device = device

    @torch.no_grad()
    def predict_batch(
        self,
        batch: RachelBatch,
        *,
        return_correspondence: bool = False,
        compute_pose_for_rejected: bool = False,
    ) -> FrozenInferenceOutput:
        """Predict without consulting any supervision-bearing batch field."""

        pair_ids: List[str] = []
        coarse_values: List[np.ndarray] = []
        pair_values: List[np.ndarray] = []
        translations: List[np.ndarray] = []
        translation_valid: List[np.ndarray] = []
        correspondence_counts: List[np.ndarray] = []
        translation_inliers: List[np.ndarray] = []
        se2_translations: List[np.ndarray] = []
        se2_angles: List[np.ndarray] = []
        se2_valid: List[np.ndarray] = []
        se2_inliers: List[np.ndarray] = []
        probability_values: List[np.ndarray] = []
        binary_values: List[np.ndarray] = []
        amp = _amp_enabled(self.device, self.runtime)
        for chunk in _microbatches(batch, self.runtime.classify_microbatch):
            with torch.cuda.amp.autocast(enabled=amp):
                first, second = _coarse_features(self.coarse, chunk, self.device)
                coarse_score = F.cosine_similarity(
                    first.float(), second.float(), dim=-1
                )
                mask_a, points_a, valid_a = _side_inputs(chunk, "a", self.device)
                mask_b, points_b, valid_b = _side_inputs(chunk, "b", self.device)
                pair_score, probability, binary = self.classify(
                    mask_a,
                    mask_b,
                    points_a,
                    points_b,
                    valid_a,
                    valid_b,
                    self.recipe.correspondence_threshold,
                )
            probability_np = probability.detach().float().cpu().numpy()
            binary_np = binary.detach().cpu().numpy().astype(np.bool_, copy=False)
            score_np = pair_score.detach().float().cpu().numpy()
            coarse_np = coarse_score.detach().float().cpu().numpy()
            points_a_np = np.asarray(chunk.points_rc_a, dtype=np.float64)
            points_b_np = np.asarray(chunk.points_rc_b, dtype=np.float64)
            valid_a_np = np.asarray(chunk.contour_valid_a, dtype=np.bool_)
            valid_b_np = np.asarray(chunk.contour_valid_b, dtype=np.bool_)
            batch_translation = np.zeros((len(chunk.pair_ids), 2), dtype=np.float64)
            batch_translation_valid = np.zeros(len(chunk.pair_ids), dtype=np.bool_)
            batch_count = np.zeros(len(chunk.pair_ids), dtype=np.int64)
            batch_translation_inlier = np.zeros(len(chunk.pair_ids), dtype=np.int64)
            batch_se2_translation = np.zeros(
                (len(chunk.pair_ids), 2), dtype=np.float64
            )
            batch_se2_angle = np.zeros(len(chunk.pair_ids), dtype=np.float64)
            batch_se2_valid = np.zeros(len(chunk.pair_ids), dtype=np.bool_)
            batch_se2_inlier = np.zeros(len(chunk.pair_ids), dtype=np.int64)
            pose_gate = min(
                self.pair_threshold, self.recipe.pair_score_threshold
            )
            for index, pair_id in enumerate(chunk.pair_ids):
                matrix = binary_np[index]
                matrix &= valid_a_np[index, :, None]
                matrix &= valid_b_np[index, None, :]
                row_index, column_index = np.nonzero(matrix)
                weights = probability_np[index, row_index, column_index]
                if len(row_index) > MAX_POSE_CORRESPONDENCES:
                    order = np.lexsort((column_index, row_index, -weights))
                    keep = order[:MAX_POSE_CORRESPONDENCES]
                    row_index = row_index[keep]
                    column_index = column_index[keep]
                    weights = weights[keep]
                batch_count[index] = len(row_index)
                if not compute_pose_for_rejected and score_np[index] < pose_gate:
                    continue
                source = points_a_np[index, row_index]
                target = points_b_np[index, column_index]
                translation = cauchy_translation_consensus(
                    source,
                    target,
                    weights,
                    min_correspondences=16,
                )
                batch_translation[index] = translation.translation_rc
                batch_translation_valid[index] = translation.valid
                batch_translation_inlier[index] = translation.inlier_count
                ransac_seed = int.from_bytes(
                    hashlib.sha256(
                        (str(self.recipe.seed) + "\0" + pair_id).encode("utf-8")
                    ).digest()[:8],
                    "big",
                )
                se2 = deterministic_se2_ransac(
                    source,
                    target,
                    seed=ransac_seed,
                    max_iterations=4000,
                    sample_size=5,
                    inlier_threshold_px=10.0,
                    min_inliers=16,
                )
                batch_se2_translation[index] = se2.translation_rc
                batch_se2_angle[index] = se2.rotation_degrees
                batch_se2_valid[index] = se2.valid
                batch_se2_inlier[index] = se2.inlier_count
            pair_ids.extend(chunk.pair_ids)
            coarse_values.append(coarse_np)
            pair_values.append(score_np)
            translations.append(batch_translation)
            translation_valid.append(batch_translation_valid)
            correspondence_counts.append(batch_count)
            translation_inliers.append(batch_translation_inlier)
            se2_translations.append(batch_se2_translation)
            se2_angles.append(batch_se2_angle)
            se2_valid.append(batch_se2_valid)
            se2_inliers.append(batch_se2_inlier)
            if return_correspondence:
                probability_values.append(probability_np)
                binary_values.append(binary_np.copy())
        return FrozenInferenceOutput(
            method_id=METHOD_ID,
            pair_ids=tuple(pair_ids),
            coarse_score=np.concatenate(coarse_values),
            pair_score=np.concatenate(pair_values),
            translation_hat_rc=np.concatenate(translations),
            translation_valid=np.concatenate(translation_valid),
            correspondence_count=np.concatenate(correspondence_counts),
            translation_inlier_count=np.concatenate(translation_inliers),
            se2_translation_hat_rc=np.concatenate(se2_translations),
            se2_rotation_degrees=np.concatenate(se2_angles),
            se2_valid=np.concatenate(se2_valid),
            se2_inlier_count=np.concatenate(se2_inliers),
            correspondence_probability=(
                np.concatenate(probability_values)
                if return_correspondence
                else None
            ),
            correspondence_binary=(
                np.concatenate(binary_values) if return_correspondence else None
            ),
        )


def load_frozen_inference(
    freeze_path: Union[str, Path], *, device: Union[str, torch.device]
) -> RachelShreddingNetInference:
    """Verify a three-stage train/val bundle and restore its frozen interface."""

    path = _lexical_absolute_path(freeze_path)
    _assert_no_symlink_components(path, name="train/val freeze")
    freeze = _read_verified_json(path, "train/val freeze")
    adapter_identity = _adapter_identity()
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA_VERSION
        or freeze.get("checkpoint_kind") != FREEZE_CHECKPOINT_KIND
        or freeze.get("status") != "complete_train_val_frozen_no_test_or_real"
        or freeze.get("method_id") != METHOD_ID
        or freeze.get("official_commit") != OFFICIAL_COMMIT
        or freeze.get("adapter_identity") != adapter_identity
    ):
        raise BenchmarkContractError("freeze identity/status differs")
    try:
        recipe = ReleaseRecipe(**freeze["recipe"])
        runtime = RuntimeBatchConfig(**freeze["runtime_batch_config"])
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkContractError("freeze recipe/runtime differs") from error
    _require_executable_release_recipe(recipe)
    root = path.parent
    _assert_no_symlink_components(root, name="frozen bundle root")
    frozen_dataset = freeze.get("dataset_audit")
    if not isinstance(frozen_dataset, Mapping):
        raise BenchmarkContractError("freeze dataset audit is missing")
    dataset_binding = _dataset_binding(frozen_dataset)
    if (
        freeze.get("dataset_binding") != dataset_binding
        or freeze.get("train_val_provenance")
        != _train_val_provenance(adapter_identity, dataset_binding)
    ):
        raise BenchmarkContractError("freeze dataset binding differs")
    if freeze.get("numeric_contract") != {
        "dual_softmax_probability_compute_dtype": "float32",
        "amp_logits_promoted_before_mask_and_softmax": True,
        "focal_logarithm_epsilon_semantics": (
            "log_p_plus_1e-9_and_log_1_minus_p_plus_1e-9"
        ),
    }:
        raise BenchmarkContractError("freeze numeric contract differs")
    checkpoint_paths: Dict[str, Path] = {}
    checkpoints = freeze.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {
        "coarse",
        "matching",
        "classify",
    }:
        raise BenchmarkContractError("freeze checkpoint table differs")
    for stage in ("coarse", "matching", "classify"):
        row = checkpoints[stage]
        if not isinstance(row, Mapping):
            raise BenchmarkContractError(stage + " freeze checkpoint is malformed")
        if row.get("checkpoint_kind") != _stage_checkpoint_kind(stage, "winner"):
            raise BenchmarkContractError(stage + " freeze checkpoint kind differs")
        checkpoint_paths[stage] = _safe_bundle_file(
            root, row.get("path"), row.get("sha256"), stage + " winner"
        )
    threshold = freeze.get("threshold")
    if not isinstance(threshold, Mapping):
        raise BenchmarkContractError("freeze threshold is missing")
    if threshold.get("checkpoint_sha256") != checkpoints["classify"].get("sha256"):
        raise BenchmarkContractError("threshold/classifier binding differs")
    score_path = _safe_bundle_file(
        root,
        threshold.get("score_file"),
        threshold.get("score_file_sha256"),
        "validation threshold scores",
    )
    if score_path.parent != root:
        raise BenchmarkContractError("threshold score file location differs")
    score_row_count = threshold.get("score_file_row_count")
    score_reused = threshold.get("score_file_reused_after_verified_resume")
    if (
        type(score_row_count) is not int  # noqa: E721
        or score_row_count < 1
        or type(score_reused) is not bool  # noqa: E721
    ):
        raise BenchmarkContractError("threshold score transaction receipt differs")
    threshold_fields = {
        name: threshold.get(name)
        for name in PairwiseThresholdArtifact.__dataclass_fields__
    }
    try:
        threshold_artifact = PairwiseThresholdArtifact(**threshold_fields)
    except (TypeError, ValueError) as error:
        raise BenchmarkContractError("frozen threshold artifact differs") from error
    if threshold.get("content_sha256") != threshold_artifact.content_sha256:
        raise BenchmarkContractError("frozen threshold content SHA-256 differs")
    model_config_sha = _canonical_sha256(
        {
            "method_id": METHOD_ID,
            "official_commit": OFFICIAL_COMMIT,
            "recipe": asdict(recipe),
            "runtime_batch_config": asdict(runtime),
            "adapter_identity": adapter_identity,
            "dataset_binding": dataset_binding,
            "train_val_provenance": _train_val_provenance(
                adapter_identity, dataset_binding
            ),
        }
    )
    aggregation_config = threshold.get("aggregation_config")
    if (
        threshold_artifact.model_config_sha256 != model_config_sha
        or not isinstance(aggregation_config, Mapping)
        or threshold_artifact.aggregation_config_sha256
        != _canonical_sha256(dict(aggregation_config))
    ):
        raise BenchmarkContractError("threshold model/aggregation binding differs")
    score_rows: List[Dict[str, object]] = []
    with score_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BenchmarkContractError(
                    "invalid validation threshold score line"
                ) from error
            if (
                not isinstance(row, dict)
                or row.get("schema_version")
                != "rachel-shreddingnet-val-score/1.0"
                or row.get("method_id") != METHOD_ID
                or row.get("ordinal") != line_number - 1
            ):
                raise BenchmarkContractError("validation threshold score row differs")
            score_rows.append(row)
    if len(score_rows) != score_row_count:
        raise BenchmarkContractError("threshold score row count differs")
    manifest_table = frozen_dataset.get("manifest_sha256")
    if not isinstance(manifest_table, Mapping) or not isinstance(
        manifest_table.get("val"), str
    ):
        raise BenchmarkContractError("freeze validation manifest binding is missing")
    reconstructed_order = _canonical_sha256(
        {
            "schema_version": "rachel-shreddingnet-validation-order/1.0",
            "method_id": METHOD_ID,
            "manifest_sha256": manifest_table["val"],
            "ordered_rows": [
                {
                    "ordinal": row["ordinal"],
                    "pair_id": row["pair_id"],
                    "label": row["label"],
                    "cluster_id": row["cluster_id"],
                }
                for row in score_rows
            ],
        }
    )
    if (
        reconstructed_order != threshold_artifact.validation_fingerprint_sha256
        or reconstructed_order != threshold.get("validation_order_sha256")
    ):
        raise BenchmarkContractError("threshold validation order binding differs")
    try:
        replayed_threshold = fit_pairwise_threshold(
            [float(row["pair_score"]) for row in score_rows],
            [row["label"] for row in score_rows],
            [True] * len(score_rows),
            [str(row["cluster_id"]) for row in score_rows],
            source_split=threshold_artifact.source_split,
            validation_fingerprint_sha256=reconstructed_order,
            checkpoint_sha256=threshold_artifact.checkpoint_sha256,
            model_config_sha256=model_config_sha,
            aggregation_config_sha256=threshold_artifact.aggregation_config_sha256,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkContractError("cannot replay frozen threshold fit") from error
    if replayed_threshold.to_dict() != threshold_artifact.to_dict():
        raise BenchmarkContractError("replayed frozen threshold differs")
    device_value = torch.device(device)
    coarse_checkpoint = _load_winner_state(
        checkpoint_paths["coarse"],
        stage="coarse",
        recipe=recipe,
        runtime=runtime,
        adapter_identity=adapter_identity,
        dataset_binding=dataset_binding,
        device=device_value,
    )
    matching_checkpoint = _load_winner_state(
        checkpoint_paths["matching"],
        stage="matching",
        recipe=recipe,
        runtime=runtime,
        adapter_identity=adapter_identity,
        dataset_binding=dataset_binding,
        device=device_value,
    )
    classify_checkpoint = _load_winner_state(
        checkpoint_paths["classify"],
        stage="classify",
        recipe=recipe,
        runtime=runtime,
        adapter_identity=adapter_identity,
        dataset_binding=dataset_binding,
        device=device_value,
    )
    if classify_checkpoint.get("matching_winner_sha256") != checkpoints[
        "matching"
    ].get("sha256"):
        raise BenchmarkContractError("classifier/matching checkpoint binding differs")
    coarse = ReleasedFragmentEncoder(recipe).to(device_value)
    coarse.load_state_dict(coarse_checkpoint["model_state_dict"], strict=True)
    classify = ReleasedClassifyStage(recipe).to(device_value)
    classify.matcher.load_state_dict(
        matching_checkpoint["model_state_dict"], strict=True
    )
    classify.load_state_dict(classify_checkpoint["model_state_dict"], strict=True)
    coarse.requires_grad_(False).eval()
    classify.requires_grad_(False).eval()
    pair_threshold = threshold.get("threshold")
    if (
        isinstance(pair_threshold, bool)
        or not isinstance(pair_threshold, (int, float))
        or not math.isfinite(float(pair_threshold))
        or not 0.0 <= float(pair_threshold) <= 1.0
    ):
        raise BenchmarkContractError("frozen pair threshold is invalid")
    return RachelShreddingNetInference(
        coarse=coarse,
        classify=classify,
        recipe=recipe,
        runtime=runtime,
        pair_threshold=float(pair_threshold),
        device=device_value,
    )


def _selection_metric_views(
    labels: np.ndarray, selected: np.ndarray, clusters: Sequence[str]
) -> Dict[str, object]:
    ordinary = edge_precision_recall_f1(labels, selected)
    evaluated = evaluate_pairwise(
        selected.astype(np.float64),
        labels.astype(np.bool_),
        np.ones(len(labels), dtype=np.bool_),
        clusters,
        threshold=0.5,
    )
    return {
        "row": ordinary,
        "cluster_balanced": {
            key: evaluated["cluster_balanced"][key]
            for key in (
                "precision",
                "recall",
                "f1",
                "tp_weight",
                "fp_weight",
                "fn_weight",
            )
        },
    }


def _assembly_metric_views(
    *,
    labels: np.ndarray,
    selected: np.ndarray,
    pose_correct: np.ndarray,
    pose_valid: np.ndarray,
    clusters: Sequence[str],
) -> Dict[str, object]:
    if not (
        labels.shape
        == selected.shape
        == pose_correct.shape
        == pose_valid.shape
        == (len(clusters),)
    ):
        raise ValueError("assembly edge arrays are not aligned")
    predicted_transformed_edge = selected & pose_valid
    correct = labels & predicted_transformed_edge & pose_correct

    def view(weight: np.ndarray, *, integer: bool) -> Dict[str, object]:
        tp = float(weight[correct].sum())
        # An accepted edge with the wrong transform is a wrong predicted edge
        # and also leaves its GT edge unrecovered.
        fp = float(weight[predicted_transformed_edge & ~correct].sum())
        fn = float(weight[labels & ~correct].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        output: Dict[str, object] = {
            "true_positive": int(tp) if integer else tp,
            "false_positive": int(fp) if integer else fp,
            "false_negative": int(fn) if integer else fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if integer:
            output.update(
                {
                    "selected_edge_count": int(selected.sum()),
                    "predicted_transformed_edge_count": int(
                        predicted_transformed_edge.sum()
                    ),
                    "ground_truth_edge_count": int(labels.sum()),
                    "selected_invalid_pose_count": int(
                        np.count_nonzero(selected & ~pose_valid)
                    ),
                }
            )
        return output

    row_weight = np.ones(len(labels), dtype=np.float64)
    cluster_array = np.asarray(clusters, dtype=object)
    unique, inverse, counts = np.unique(
        cluster_array, return_inverse=True, return_counts=True
    )
    cluster_weight = 1.0 / (
        len(unique) * counts[inverse].astype(np.float64)
    )
    return {
        "row": view(row_weight, integer=True),
        "cluster_balanced": view(cluster_weight, integer=False),
    }


def _safe_candidate_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    candidate: np.ndarray,
    clusters: Sequence[str],
) -> Dict[str, object]:
    selected_labels = labels[candidate]
    if len(selected_labels) == 0 or np.all(selected_labels) or not np.any(
        selected_labels
    ):
        return {
            "computable": False,
            "reason": "candidate_population_lacks_both_classes",
            "candidate_count": int(candidate.sum()),
        }
    selected_clusters = np.asarray(clusters, dtype=object)[candidate].tolist()
    evaluated = evaluate_pairwise(
        scores[candidate],
        selected_labels,
        np.ones(int(candidate.sum()), dtype=np.bool_),
        selected_clusters,
        threshold=0.5,
    )
    return {
        "computable": True,
        "candidate_count": int(candidate.sum()),
        "row_auroc": binary_auroc(selected_labels, scores[candidate]),
        "cluster_balanced_auroc": evaluated["cluster_balanced"]["auroc"],
    }


def evaluate_validation(
    *,
    dataset_root: Union[str, Path],
    freeze_path: Union[str, Path],
    output_path: Union[str, Path],
    device: Union[str, torch.device],
    num_workers: int = 0,
) -> Dict[str, object]:
    """Descriptive val-only replay; never accepts a test or real split."""

    destination = _prepare_fresh_file_path(
        _lexical_absolute_path(output_path), name="validation report"
    )
    freeze_file = _lexical_absolute_path(freeze_path)
    _assert_no_symlink_components(freeze_file, name="train/val freeze")
    freeze = _read_verified_json(freeze_file, "train/val freeze")
    frozen_dataset = freeze.get("dataset_audit")
    materialized = frozen_dataset.get("train_materialized_manifest") if isinstance(frozen_dataset, Mapping) else None
    dataset_audit, _, val_metadata = audit_rachel_train_val(dataset_root,
        require_formal_counts=materialized is None, train_materialized_manifest=materialized)
    if (
        not isinstance(frozen_dataset, Mapping)
        or _dataset_binding(frozen_dataset) != _dataset_binding(dataset_audit)
    ):
        raise BenchmarkContractError("current Rachel manifests differ from freeze")
    inference = load_frozen_inference(freeze_file, device=device)
    val_dataset = RachelPairDataset(dataset_root, "val")
    loader = _loader(
        val_dataset,
        tuple(range(len(val_metadata))),
        batch_size=inference.recipe.classify_batch_size,
        shuffle=False,
        drop_last=False,
        seed=inference.recipe.seed,
        num_workers=num_workers,
        device=inference.device,
    )
    coarse_parts = []
    score_parts = []
    translation_parts = []
    translation_valid_parts = []
    correspondence_count_parts = []
    se2_translation_parts = []
    se2_angle_parts = []
    se2_valid_parts = []
    label_parts = []
    gt_translation_parts = []
    gt_valid_parts = []
    observed_ids: List[str] = []
    for batch in loader:
        prediction = inference.predict_batch(batch)
        observed_ids.extend(prediction.pair_ids)
        coarse_parts.append(prediction.coarse_score)
        score_parts.append(prediction.pair_score)
        translation_parts.append(prediction.translation_hat_rc)
        translation_valid_parts.append(prediction.translation_valid)
        correspondence_count_parts.append(prediction.correspondence_count)
        se2_translation_parts.append(prediction.se2_translation_hat_rc)
        se2_angle_parts.append(prediction.se2_rotation_degrees)
        se2_valid_parts.append(prediction.se2_valid)
        # Supervision is read only after the frozen forward has returned.
        label_parts.append(np.asarray(batch.labels >= 0.5, dtype=np.bool_))
        gt_translation_parts.append(
            np.asarray(batch.translation_a_to_b_rc, dtype=np.float64)
        )
        gt_valid_parts.append(
            np.asarray(batch.translation_valid, dtype=np.bool_)
        )
    if observed_ids != [row.pair_id for row in val_metadata]:
        raise BenchmarkContractError("validation inference order differs")
    labels = np.concatenate(label_parts)
    coarse_score = np.concatenate(coarse_parts).astype(np.float64)
    pair_score = np.concatenate(score_parts).astype(np.float64)
    translation = np.concatenate(translation_parts)
    translation_valid = np.concatenate(translation_valid_parts)
    correspondence_count = np.concatenate(correspondence_count_parts)
    se2_translation = np.concatenate(se2_translation_parts)
    se2_angle = np.concatenate(se2_angle_parts)
    se2_valid = np.concatenate(se2_valid_parts)
    gt_translation = np.concatenate(gt_translation_parts)
    gt_valid = np.concatenate(gt_valid_parts)
    if not np.array_equal(labels, gt_valid):
        raise BenchmarkContractError("Rachel positive/translation validity differs")
    clusters = [row.cluster_id for row in val_metadata]
    fitted_threshold = inference.pair_threshold
    native_threshold = inference.recipe.pair_score_threshold
    fitted_selected = pair_score >= fitted_threshold
    native_selected = pair_score >= native_threshold
    translation_error = np.full(len(labels), np.inf, dtype=np.float64)
    translation_usable = labels & translation_valid
    translation_error[translation_usable] = np.linalg.norm(
        translation[translation_usable] - gt_translation[translation_usable], axis=1
    )
    se2_translation_error = np.full(len(labels), np.inf, dtype=np.float64)
    se2_usable = labels & se2_valid
    se2_translation_error[se2_usable] = np.linalg.norm(
        se2_translation[se2_usable] - gt_translation[se2_usable], axis=1
    )
    assembly_primary = {}
    assembly_native = {}
    for tolerance in PRIMARY_TRANSLATION_TOLERANCES_PX:
        key = "@{}px".format(int(tolerance))
        pose_correct = translation_error <= tolerance
        assembly_primary[key] = _assembly_metric_views(
            labels=labels,
            selected=fitted_selected,
            pose_correct=pose_correct,
            pose_valid=translation_valid,
            clusters=clusters,
        )
        assembly_native[key] = _assembly_metric_views(
            labels=labels,
            selected=native_selected,
            pose_correct=pose_correct,
            pose_valid=translation_valid,
            clusters=clusters,
        )
    selected_pair_list_diagnostics = {}
    for name, k in (
        ("listed_pair_topk_2", inference.recipe.rachel_top_k),
        ("listed_pair_topk_20", inference.recipe.official_top_k),
    ):
        candidate = topk_candidate_mask(val_metadata, coarse_score, k)
        selected_pair_list_diagnostics[name] = {
            "k": k,
            "scope": "preselected_balanced_validation_pair_list_only",
            "native_shreddingnet_candidate_graph_metric": False,
            "topk_degenerates_to_all_listed_pairs": bool(np.all(candidate)),
            "coarse_topk_like_selection": _selection_metric_views(
                labels, candidate, clusters
            ),
            "score_filtered_topk_like_primary_fitted_threshold": _selection_metric_views(
                labels, candidate & fitted_selected, clusters
            ),
            "score_filtered_topk_like_native_0_5_threshold": _selection_metric_views(
                labels, candidate & native_selected, clusters
            ),
            "pair_score_auroc_within_topk_like_selection": _safe_candidate_auroc(
                labels, pair_score, candidate, clusters
            ),
        }
    secondary_pose_correct = (
        se2_valid
        & (np.abs(se2_angle) < OFFICIAL_SECONDARY_ROTATION_DEG)
        & (se2_translation_error < OFFICIAL_SECONDARY_TRANSLATION_PX)
    )
    report = {
        "schema_version": "rachel-shreddingnet-validation-report/1.0",
        "status": "complete_descriptive_validation_replay_no_test_or_real",
        "method_id": METHOD_ID,
        "freeze_content_sha256": freeze["content_sha256"],
        "adapter_identity": freeze["adapter_identity"],
        "dataset_binding": freeze["dataset_binding"],
        "train_val_provenance": freeze["train_val_provenance"],
        "validation_manifest_sha256": dataset_audit["manifest_sha256"]["val"],
        "validation_pair_order_sha256": _validation_fingerprint(
            val_metadata, str(dataset_audit["manifest_sha256"]["val"])
        ),
        "threshold_protocol": {
            "primary_threshold": fitted_threshold,
            "primary_origin": "validation_fitted_cluster_balanced_f1_after_winner_freeze",
            "native_secondary_threshold": native_threshold,
            "threshold_fit_and_descriptive_report_share_validation_population": True,
            "threshold_fitted_on_test_or_real": False,
        },
        "pair_classification_diagnostic": {
            "primary_fitted": evaluate_pairwise(
                pair_score,
                labels,
                np.ones(len(labels), dtype=np.bool_),
                clusters,
                threshold=fitted_threshold,
            ),
            "native_0_5": evaluate_pairwise(
                pair_score,
                labels,
                np.ones(len(labels), dtype=np.bool_),
                clusters,
                threshold=native_threshold,
            ),
        },
        "balanced_selected_pair_list_module_like_diagnostics": {
            "not_named_native_cm_fm_se": True,
            "reason": (
                "frozen_validation_manifest_is_a_preselected_1_to_1_balanced_pair_"
                "list_not_the_exhaustive_same_parent_candidate_graph"
            ),
            "views": selected_pair_list_diagnostics,
        },
        "primary_translation_only_pairwise_assembly_edge": assembly_primary,
        "secondary_native_0_5_translation_only_pairwise_assembly_edge": assembly_native,
        "secondary_official_se2_threshold_pairwise_edge": {
            "criterion": {
                "rotation_error_strictly_below_degrees": OFFICIAL_SECONDARY_ROTATION_DEG,
                "translation_error_strictly_below_px": OFFICIAL_SECONDARY_TRANSLATION_PX,
            },
            "primary_fitted_pair_threshold": _assembly_metric_views(
                labels=labels,
                selected=fitted_selected,
                pose_correct=secondary_pose_correct,
                pose_valid=se2_valid,
                clusters=clusters,
            ),
            "native_0_5_pair_threshold": _assembly_metric_views(
                labels=labels,
                selected=native_selected,
                pose_correct=secondary_pose_correct,
                pose_valid=se2_valid,
                clusters=clusters,
            ),
            "metric_named_ga": False,
            "reason": "no_maximum_spanning_tree_or_global_multifragment_placement",
        },
        "pose_diagnostics": {
            "accepted_primary_count": int(fitted_selected.sum()),
            "accepted_native_0_5_count": int(native_selected.sum()),
            "translation_valid_count": int(translation_valid.sum()),
            "se2_valid_count": int(se2_valid.sum()),
            "correspondence_count_median": float(np.median(correspondence_count)),
            "correspondence_count_p95": float(
                np.quantile(correspondence_count, 0.95)
            ),
        },
        "scope": {
            "validation_only": True,
            "sealed_test_or_real_opened": False,
            "global_assembly_performed": False,
            "pairwise_results_called_ga": False,
            "mask_only_rachel_n512_upright_adaptation": True,
            "headline_metric": "translation_L2_pairwise_assembly_edge_at_2_5_8_10px",
            "native_cm_fm_se_reported": False,
        },
    }
    _write_json_new_atomic(destination, report, name="validation report")
    return _read_verified_json(destination, "validation report")


def _device(value: Union[str, torch.device]) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BenchmarkContractError("CUDA was requested but is unavailable")
    return device


def _formal_runtime_preflight(
    *,
    dataset_root: Union[str, Path],
    dataset_audit: Mapping[str, object],
    train_dataset: RachelPairDataset,
    train_metadata: Sequence[PairMetadata],
    recipe: ReleaseRecipe,
    device: torch.device,
    train_materialized_manifest=None,
) -> Dict[str, object]:
    """Decode train probes and run all real stage forwards before any output write."""

    _require_torch_geometric()
    positive_index = next(
        (index for index, row in enumerate(train_metadata) if row.label), None
    )
    negative_index = next(
        (index for index, row in enumerate(train_metadata) if not row.label), None
    )
    if positive_index is None or negative_index is None:
        raise BenchmarkContractError("preflight requires train positive and negative probes")
    try:
        positive_batch = collate_rachel_pairs([train_dataset[positive_index]])
        classify_batch = collate_rachel_pairs(
            [train_dataset[positive_index], train_dataset[negative_index]]
        )
    except Exception as error:
        raise BenchmarkContractError("cannot decode Rachel train preflight probes") from error

    train_args = ((_lexical_absolute_path(dataset_root), "train", train_materialized_manifest)
                  if train_materialized_manifest is not None
                  else (_lexical_absolute_path(dataset_root), "train"))
    current_train, current_train_sha = _parse_pair_manifest(*train_args)
    current_val, current_val_sha = _parse_pair_manifest(
        _lexical_absolute_path(dataset_root), "val"
    )
    current_binding = {
        "manifest_content_sha256": {
            "train": current_train_sha,
            "val": current_val_sha,
        },
        "ordered_pair_ids_sha256": {
            "train": _ordered_pair_ids_sha256("train", current_train),
            "val": _ordered_pair_ids_sha256("val", current_val),
        },
    }
    if current_binding != _dataset_binding(dataset_audit):
        raise BenchmarkContractError("Rachel manifests changed during preflight")

    stage_shapes: Dict[str, object] = {}
    try:
        for stage in ("coarse", "matching", "classify"):
            _seed_everything(recipe.seed)
            model = _stage_model(stage, recipe).to(device).eval()
            with torch.no_grad():
                if stage == "coarse":
                    assert isinstance(model, ReleasedFragmentEncoder)
                    first, second = _coarse_features(model, positive_batch, device)
                    outputs = (first, second)
                elif stage == "matching":
                    assert isinstance(model, ReleasedFineMatcher)
                    mask_a, points_a, valid_a = _side_inputs(
                        positive_batch, "a", device
                    )
                    mask_b, points_b, valid_b = _side_inputs(
                        positive_batch, "b", device
                    )
                    outputs = (
                        model(
                            mask_a,
                            mask_b,
                            points_a,
                            points_b,
                            valid_a,
                            valid_b,
                        ),
                    )
                else:
                    assert isinstance(model, ReleasedClassifyStage)
                    mask_a, points_a, valid_a = _side_inputs(
                        classify_batch, "a", device
                    )
                    mask_b, points_b, valid_b = _side_inputs(
                        classify_batch, "b", device
                    )
                    outputs = model(
                        mask_a,
                        mask_b,
                        points_a,
                        points_b,
                        valid_a,
                        valid_b,
                        recipe.correspondence_threshold,
                    )
            if any(not bool(torch.isfinite(value).all()) for value in outputs):
                raise BenchmarkContractError(stage + " preflight produced non-finite output")
            stage_shapes[stage] = [list(value.shape) for value in outputs]
            del model, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except BenchmarkContractError:
        raise
    except Exception as error:
        raise BenchmarkContractError("real model construct/forward preflight failed") from error
    probe_ids = (
        train_metadata[positive_index].pair_id,
        train_metadata[negative_index].pair_id,
    )
    try:
        torch_geometric_version = importlib_metadata.version("torch-geometric")
    except importlib_metadata.PackageNotFoundError as error:
        raise BenchmarkContractError("torch-geometric distribution metadata missing") from error
    return {
        "preflight_kind": "rachel_shreddingnet_real_models_train_probe",
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric_version,
        "real_stage_model_constructed_and_forwarded": {
            "coarse": True,
            "matching": True,
            "classify": True,
        },
        "stage_output_shapes": stage_shapes,
        "train_probe_pair_ids_sha256": hashlib.sha256(
            ("\n".join(probe_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "train_positive_and_negative_artifacts_decoded": True,
        "train_val_manifests_reverified_after_decode": True,
        "output_created_or_written": False,
    }


def train_rachel_shreddingnet(
    *,
    dataset_root: Union[str, Path],
    official_repo: Union[str, Path],
    output_root: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    num_workers: int = 0,
    resume: bool = False,
    write_validation_report: bool = True,
    recipe: ReleaseRecipe = ReleaseRecipe(),
    runtime: RuntimeBatchConfig = RuntimeBatchConfig(),
    train_materialized_manifest=None,
) -> Dict[str, object]:
    """Run official-release three-stage training on authorized Rachel train/val."""

    if type(num_workers) is not int or num_workers < 0:  # noqa: E721
        raise ValueError("num_workers must be a non-negative integer")
    _require_executable_release_recipe(recipe)
    if recipe.experimental_data != (train_materialized_manifest is not None):
        raise BenchmarkContractError("experimental data requires an explicit materialized TRAIN manifest, and vice versa")
    if type(resume) is not bool or type(write_validation_report) is not bool:  # noqa: E721
        raise TypeError("resume/report flags must be bool")
    destination = _lexical_absolute_path(output_root)
    initial_root_state = _inspect_training_output_root(destination)
    if resume and initial_root_state == "absent":
        raise BenchmarkContractError(
            "--resume requires an existing lexical output directory"
        )
    if not resume and initial_root_state != "absent":
        raise BenchmarkContractError(
            "fresh output must be absent; use --resume only for a controlled "
            "existing training root"
        )
    dataset_audit, train_metadata, val_metadata = audit_rachel_train_val(
        dataset_root, require_formal_counts=not recipe.experimental_data,
        train_materialized_manifest=train_materialized_manifest,
    )
    if len(val_metadata) != 3000 and recipe.experimental_data:
        raise BenchmarkContractError("experimental TRAIN override must retain the complete clean 3000-row VAL")
    official_audit = audit_official_source(official_repo)
    adapter_identity = _adapter_identity()
    dataset_binding = _dataset_binding(dataset_audit)
    device_value = _device(device)
    _seed_everything(recipe.seed)
    try:
        train_dataset = (materialized_train_dataset(train_materialized_manifest)
            if train_materialized_manifest is not None else RachelPairDataset(dataset_root, "train"))
        val_dataset = RachelPairDataset(dataset_root, "val")
    except Exception as error:
        raise BenchmarkContractError("cannot construct Rachel train/val datasets") from error
    preflight = _formal_runtime_preflight(
        dataset_root=dataset_root,
        dataset_audit=dataset_audit,
        train_dataset=train_dataset,
        train_metadata=train_metadata,
        recipe=recipe,
        device=device_value,
        train_materialized_manifest=train_materialized_manifest,
    )
    run_contract = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": "rachel_shreddingnet_run_contract",
        "status": "initialized_train_val_only",
        "method_id": METHOD_ID,
        "official_commit": OFFICIAL_COMMIT,
        "output_root_lexical": destination.as_posix(),
        "recipe": asdict(recipe),
        "runtime_batch_config": asdict(runtime),
        "adapter_identity": adapter_identity,
        "dataset_binding": dataset_binding,
        "train_val_provenance": _train_val_provenance(
            adapter_identity, dataset_binding
        ),
        "dataset_audit": dataset_audit,
        "official_source_audit": official_audit,
        "runtime_preflight": preflight,
        "formal_exposure": {
            "seed": recipe.seed,
            "official_release_seed_secondary": recipe.official_release_seed,
            "training_epoch_order_algorithm": (
                "numpy_default_rng_seed_plus_1000003_times_epoch"
            ),
            "stage_validation_order_contract": {
                stage: _validation_order_contract(stage, recipe.seed)
                for stage in ("coarse", "matching", "classify")
            },
            "same_frozen_train_val_manifest_order": True,
            "per_stage_population": {
                "coarse": "positive_pairs_only",
                "matching": "positive_pairs_only",
                "classify": "all_frozen_balanced_manifest_rows",
            },
        },
        "microbatch_semantics": {
            "optimizer_step_effective_batch_sizes": {
                "coarse": recipe.coarse_batch_size,
                "matching": recipe.matching_batch_size,
                "classify": recipe.classify_batch_size,
            },
            "coarse_gradient_cache_preserves_full_effective_infonce_denominator": True,
            "matching_and_classifier_loss_weighted_across_microbatches": True,
            "batchnorm_uses_microbatch_statistics": True,
            "bitwise_equivalent_to_monolithic_release_batch": False,
            "dual_softmax_probability_compute_dtype": "float32",
            "amp_logits_promoted_before_mask_and_softmax": True,
            "focal_logarithm_epsilon_semantics": (
                "log_p_plus_1e-9_and_log_1_minus_p_plus_1e-9"
            ),
        },
        "scope": {
            "sealed_test_or_real_opened": False,
            "rgb_opened": False,
            "fresh_no_clobber": not resume,
            "resume_requested": resume,
            "root_state_before_invocation": initial_root_state,
            "recovered_safe_empty_orphan": (
                initial_root_state == "safe_empty_orphan"
            ),
        },
    }
    contract_path = destination / "run_contract.json"
    current_root_state = _inspect_training_output_root(destination)
    if current_root_state != initial_root_state:
        raise BenchmarkContractError("training output root changed during preflight")
    has_contract = initial_root_state in {"contract_only", "active_contract"}
    if has_contract:
        if not resume:
            raise BenchmarkContractError("fresh invocation cannot reuse a run contract")
        existing = _read_verified_json(contract_path, "run contract")
        expected = dict(run_contract)
        expected.pop("scope")
        observed = dict(existing)
        observed.pop("content_sha256", None)
        observed.pop("scope", None)
        if observed != expected:
            raise BenchmarkContractError("resume run contract differs")
    else:
        _ensure_directory_tree(destination.parent, name="training output parent")
        if initial_root_state == "absent":
            try:
                os.mkdir(destination)
            except FileExistsError as error:
                raise BenchmarkContractError(
                    "training output appeared before create"
                ) from error
        elif _inspect_training_output_root(destination) != "safe_empty_orphan":
            raise BenchmarkContractError("empty training root recovery state changed")
        _assert_no_symlink_components(destination, name="training output root")
        _write_json_new_atomic(contract_path, run_contract, name="run contract")
    stage_receipts: Dict[str, Mapping[str, object]] = {}
    for stage in ("coarse", "matching", "classify"):
        stage_receipts[stage] = _run_stage(
            stage=stage,
            output_root=destination,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            recipe=recipe,
            runtime=runtime,
            device=device_value,
            num_workers=num_workers,
            resume=resume,
            adapter_identity=adapter_identity,
            dataset_binding=dataset_binding,
            matching_winner_path=(
                destination / "stages" / "matching" / "winner.pt"
                if stage == "classify"
                else None
            ),
        )
    freeze_path = destination / "train_val_freeze.json"
    _assert_no_symlink_components(freeze_path, name="train/val freeze")
    if _path_lexists(freeze_path):
        if not resume:
            raise BenchmarkContractError("freeze appeared during fresh run")
        verified_inference = load_frozen_inference(freeze_path, device=device_value)
        del verified_inference
        freeze = _read_verified_json(freeze_path, "train/val freeze")
    else:
        freeze = _threshold_and_freeze(
            output_root=destination,
            dataset_audit=dataset_audit,
            official_audit=official_audit,
            val_dataset=val_dataset,
            val_metadata=val_metadata,
            stage_receipts=stage_receipts,
            recipe=recipe,
            runtime=runtime,
            adapter_identity=adapter_identity,
            dataset_binding=dataset_binding,
            device=device_value,
            num_workers=num_workers,
            resume=resume,
        )
    report_path = destination / "validation_report.json"
    report: Optional[Dict[str, object]] = None
    if write_validation_report:
        _assert_no_symlink_components(report_path, name="validation report")
        if _path_lexists(report_path):
            if not resume:
                raise BenchmarkContractError("validation report appeared during fresh run")
            report = _read_verified_json(report_path, "validation report")
            if report.get("freeze_content_sha256") != freeze.get("content_sha256"):
                raise BenchmarkContractError("validation report/freeze binding differs")
        else:
            report = evaluate_validation(
                dataset_root=dataset_root,
                freeze_path=freeze_path,
                output_path=report_path,
                device=device_value,
                num_workers=num_workers,
            )
    return {
        "status": "complete_train_val_only",
        "method_id": METHOD_ID,
        "output_root": destination.as_posix(),
        "freeze_path": freeze_path.as_posix(),
        "freeze_content_sha256": freeze["content_sha256"],
        "validation_report_path": report_path.as_posix() if report else None,
        "validation_report_content_sha256": (
            report["content_sha256"] if report else None
        ),
        "sealed_test_or_real_opened": False,
    }


def resource_envelope(
    *,
    recipe: ReleaseRecipe = ReleaseRecipe(),
    runtime: RuntimeBatchConfig = RuntimeBatchConfig(),
) -> Dict[str, object]:
    """Pre-smoke planning envelope; values are explicitly not measurements."""

    _require_executable_release_recipe(recipe)

    return {
        "schema_version": "rachel-shreddingnet-resource-envelope/1.0",
        "method_id": METHOD_ID,
        "status": "planning_prior_not_measured",
        "release_effective_batch": {
            "coarse": recipe.coarse_batch_size,
            "matching": recipe.matching_batch_size,
            "classify": recipe.classify_batch_size,
        },
        "gpu_microbatch": {
            "coarse": runtime.coarse_microbatch,
            "matching": runtime.matching_microbatch,
            "classify": runtime.classify_microbatch,
        },
        "nominal_accumulation_steps": {
            "coarse": math.ceil(
                recipe.coarse_batch_size / runtime.coarse_microbatch
            ),
            "matching": math.ceil(
                recipe.matching_batch_size / runtime.matching_microbatch
            ),
            "classify": math.ceil(
                recipe.classify_batch_size / runtime.classify_microbatch
            ),
        },
        "amp_requested": runtime.amp,
        "numeric_contract": {
            "dual_softmax_probability_compute_dtype": "float32",
            "amp_logits_promoted_before_mask_and_softmax": True,
            "focal_logarithm_epsilon_semantics": (
                "log_p_plus_1e-9_and_log_1_minus_p_plus_1e-9"
            ),
        },
        "pre_smoke_24gb_planning_range": {
            "microbatch_1_peak_allocated_gib": {
                "coarse": [4, 10],
                "matching": [6, 16],
                "classify": [7, 18],
            },
            "pair_throughput_per_second": {
                "coarse_gradient_cache": [0.3, 3.0],
                "matching": [0.2, 2.0],
                "classify": [0.2, 2.0],
            },
            "confidence": "low_until_gpu_smoke_receipt_exists",
        },
        "throughput_interpretation": {
            "coarse_encoder_passes_per_optimizer_step": (
                4 * recipe.coarse_batch_size
            ),
            "reason": "two fragments times cache_and_replay",
            "formal_eta_must_be_recomputed_from_measured_stage_smoke": True,
        },
        "recommendation": {
            "first_command": "gpu-smoke",
            "start_microbatch": 1,
            "num_workers": 0,
            "do_not_start_full_128_epoch_run_before_smoke": True,
        },
    }


def run_gpu_smoke(
    *,
    dataset_root: Union[str, Path],
    official_repo: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cuda",
    num_workers: int = 0,
    recipe: ReleaseRecipe = ReleaseRecipe(),
    runtime: RuntimeBatchConfig = RuntimeBatchConfig(),
) -> Dict[str, object]:
    """Measure one release-effective optimizer step per stage, train only."""

    _require_executable_release_recipe(recipe)
    destination = (
        _prepare_fresh_file_path(
            _lexical_absolute_path(output_path), name="GPU smoke output"
        )
        if output_path is not None
        else None
    )
    device_value = _device(device)
    if device_value.type != "cuda":
        raise BenchmarkContractError("gpu-smoke requires a CUDA device")
    dataset_audit, train_metadata, _ = audit_rachel_train_val(dataset_root)
    official_audit = audit_official_source(official_repo)
    adapter_identity = _adapter_identity()
    dataset_binding = _dataset_binding(dataset_audit)
    try:
        train_dataset = RachelPairDataset(dataset_root, "train")
    except Exception as error:
        raise BenchmarkContractError("cannot construct Rachel train dataset") from error
    preflight = _formal_runtime_preflight(
        dataset_root=dataset_root,
        dataset_audit=dataset_audit,
        train_dataset=train_dataset,
        train_metadata=train_metadata,
        recipe=recipe,
        device=device_value,
    )
    stage_rows = {}
    # Keep CUDA placement explicit while using the current-device form of the
    # memory-statistics APIs.  The target torch build accepts this form but can
    # reject an explicit ``torch.device`` argument for those APIs.
    torch.cuda.set_device(
        device_value.index
        if device_value.index is not None
        else torch.cuda.current_device()
    )
    for stage in ("coarse", "matching", "classify"):
        _seed_everything(recipe.seed)
        model = _stage_model(stage, recipe).to(device_value)
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(_stage_recipe(stage, recipe)["learning_rate"]),
            weight_decay=recipe.weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=_amp_enabled(device_value, runtime)
        )
        population = [
            index
            for index, row in enumerate(train_metadata)
            if row.label or stage == "classify"
        ]
        order = _epoch_order_indices(population, recipe.seed, 1)
        loader = _loader(
            train_dataset,
            order,
            batch_size=int(_stage_recipe(stage, recipe)["batch_size"]),
            shuffle=False,
            drop_last=False,
            seed=recipe.seed + 1,
            num_workers=num_workers,
            device=device_value,
        )
        batch = next(iter(loader))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(device_value)
        started = time.perf_counter()
        metrics = _train_effective_batch(
            stage=stage,
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=scaler,
            recipe=recipe,
            runtime=runtime,
            device=device_value,
        )
        torch.cuda.synchronize(device_value)
        elapsed = time.perf_counter() - started
        stage_rows[stage] = {
            "effective_batch_size": len(batch.pair_ids),
            "configured_release_effective_batch_size": int(
                _stage_recipe(stage, recipe)["batch_size"]
            ),
            "gpu_microbatch_size": runtime.microbatch(stage),
            "nominal_accumulation_steps": math.ceil(
                len(batch.pair_ids) / runtime.microbatch(stage)
            ),
            "amp_enabled": _amp_enabled(device_value, runtime),
            "elapsed_seconds": elapsed,
            "effective_pairs_per_second": len(batch.pair_ids) / elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "metrics": metrics,
        }
        del model, optimizer, scaler, loader, batch
        torch.cuda.empty_cache()
    receipt = {
        "schema_version": "rachel-shreddingnet-gpu-smoke/1.0",
        "status": "complete_train_only_release_effective_step_smoke",
        "method_id": METHOD_ID,
        "device": str(device_value),
        "device_name": torch.cuda.get_device_name(device_value),
        "recipe": asdict(recipe),
        "runtime_batch_config": asdict(runtime),
        "adapter_identity": adapter_identity,
        "dataset_binding": dataset_binding,
        "train_val_provenance": _train_val_provenance(
            adapter_identity, dataset_binding
        ),
        "runtime_preflight": preflight,
        "stages": stage_rows,
        "dataset_manifest_sha256": dataset_audit["manifest_sha256"],
        "official_commit": official_audit["commit"],
        "scope": {
            "train_manifest_only_for_tensor_loading": True,
            "validation_tensor_loading": False,
            "sealed_test_or_real_opened": False,
            "random_smoke_models_not_checkpoints": True,
            "formal_training_started": False,
        },
        "numeric_contract": {
            "dual_softmax_probability_compute_dtype": "float32",
            "amp_logits_promoted_before_mask_and_softmax": True,
            "focal_logarithm_epsilon_semantics": (
                "log_p_plus_1e-9_and_log_1_minus_p_plus_1e-9"
            ),
        },
    }
    if destination is not None:
        _write_json_new_atomic(destination, receipt, name="GPU smoke output")
        return _read_verified_json(destination, "GPU smoke receipt")
    receipt["content_sha256"] = _canonical_sha256(receipt)
    return receipt


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coarse-microbatch", type=int, default=1)
    parser.add_argument("--matching-microbatch", type=int, default=1)
    parser.add_argument("--classify-microbatch", type=int, default=1)
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="disable CUDA automatic mixed precision",
    )
    parser.set_defaults(amp=True)


def _runtime_from_arguments(arguments: argparse.Namespace) -> RuntimeBatchConfig:
    return RuntimeBatchConfig(
        coarse_microbatch=arguments.coarse_microbatch,
        matching_microbatch=arguments.matching_microbatch,
        classify_microbatch=arguments.classify_microbatch,
        amp=arguments.amp,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rachel train/val-only ShreddingNet release adaptation"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="audit official source and train/val manifests")
    audit.add_argument("--dataset-root", required=True)
    audit.add_argument("--official-repo", required=True)
    audit.add_argument("--output")
    audit.add_argument("--train-materialized-manifest", type=Path)
    audit.add_argument("--experimental-data", action="store_true")

    train = commands.add_parser("train", help="run or resume all three release stages")
    train.add_argument("--dataset-root", required=True)
    train.add_argument("--official-repo", required=True)
    train.add_argument("--output-root", required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--skip-validation-report", action="store_true")
    train.add_argument("--train-materialized-manifest", type=Path)
    train.add_argument("--experimental-data", action="store_true")
    train.add_argument("--seed", type=int, default=260831)
    train.add_argument("--coarse-epochs", type=int, default=128)
    train.add_argument("--matching-epochs", type=int, default=128)
    train.add_argument("--classify-epochs", type=int, default=128)
    train.add_argument("--selection-metric", choices=("official", "recall95_precision"), default="official")
    _add_runtime_arguments(train)

    evaluate = commands.add_parser(
        "evaluate-val", help="replay a frozen bundle on validation only"
    )
    evaluate.add_argument("--dataset-root", required=True)
    evaluate.add_argument("--freeze", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--num-workers", type=int, default=0)

    smoke = commands.add_parser(
        "gpu-smoke", help="one train-only effective optimizer step per stage"
    )
    smoke.add_argument("--dataset-root", required=True)
    smoke.add_argument("--official-repo", required=True)
    smoke.add_argument("--output")
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--num-workers", type=int, default=0)
    _add_runtime_arguments(smoke)

    resources = commands.add_parser(
        "resource-envelope", help="print the pre-smoke resource planning envelope"
    )
    _add_runtime_arguments(resources)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(arguments)
    result: Mapping[str, Any]
    if parsed.command == "audit":
        if parsed.experimental_data != (parsed.train_materialized_manifest is not None):
            parser.error("experimental data requires --train-materialized-manifest and --experimental-data together")
        dataset, _, _ = audit_rachel_train_val(parsed.dataset_root,
            require_formal_counts=not parsed.experimental_data,
            train_materialized_manifest=parsed.train_materialized_manifest)
        official = audit_official_source(parsed.official_repo)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "audit_complete_train_val_only",
            "method_id": METHOD_ID,
            "adapter_identity": _adapter_identity(),
            "dataset_binding": _dataset_binding(dataset),
            "dataset": dataset,
            "official_source": official,
        }
        if parsed.output:
            destination = _prepare_fresh_file_path(
                _lexical_absolute_path(parsed.output), name="audit output"
            )
            _write_json_new_atomic(destination, result, name="audit output")
            result = _read_verified_json(destination, "audit receipt")
    elif parsed.command == "train":
        result = train_rachel_shreddingnet(
            dataset_root=parsed.dataset_root,
            official_repo=parsed.official_repo,
            output_root=parsed.output_root,
            device=parsed.device,
            num_workers=parsed.num_workers,
            resume=parsed.resume,
            write_validation_report=not parsed.skip_validation_report,
            runtime=_runtime_from_arguments(parsed),
            train_materialized_manifest=parsed.train_materialized_manifest,
            recipe=ReleaseRecipe(seed=parsed.seed, experimental_data=parsed.experimental_data,
                coarse_epochs=parsed.coarse_epochs, matching_epochs=parsed.matching_epochs,
                classify_epochs=parsed.classify_epochs, selection_metric=parsed.selection_metric),
        )
    elif parsed.command == "evaluate-val":
        result = evaluate_validation(
            dataset_root=parsed.dataset_root,
            freeze_path=parsed.freeze,
            output_path=parsed.output,
            device=parsed.device,
            num_workers=parsed.num_workers,
        )
    elif parsed.command == "gpu-smoke":
        result = run_gpu_smoke(
            dataset_root=parsed.dataset_root,
            official_repo=parsed.official_repo,
            output_path=parsed.output,
            device=parsed.device,
            num_workers=parsed.num_workers,
            runtime=_runtime_from_arguments(parsed),
        )
    else:
        assert parsed.command == "resource-envelope"
        result = resource_envelope(runtime=_runtime_from_arguments(parsed))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproducible source-disjoint short-training harness for Pairwise v0.2.

This module is deliberately an orchestration boundary rather than an image
provider.  A caller supplies two small callbacks: a prepared-batch provider
(which may later use the fragment geometry cache) and an arm backend.  The
runner owns the safety-critical parts: data/source locks, deterministic
dataset-by-label/component sampling, checkpoint hashing, validation-only
threshold fitting, cluster-aware evaluation, and portable receipts.

The sealed real Dunhuang test is outside this API.  Only strict ``train`` and
``val`` :class:`TrainingPairRecord` values are accepted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np
import scipy
import torch
from torch import Tensor, nn

from staging.pairwise_v0_2.models.local_matcher import MatcherMode
from staging.pairwise_v0_2.models.pairwise import ArcPoolingConfig
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.checkpoint import (
    canonical_config_hash,
    save_checkpoint,
)
from staging.pairwise_v0_2.training.data_lock import (
    ExperimentDataLock,
    require_locked_artifacts,
)
from staging.pairwise_v0_2.training.evaluation import (
    cluster_bootstrap_threshold_metrics,
    evaluate_pairwise,
    fit_pairwise_threshold,
    recall_at_fixed_fpr,
    selective_risk_curve,
)
from staging.pairwise_v0_2.training.geometry_batch import (
    COARSE_NUMERIC_CONTRACT,
    KEYPOINT_REPRESENTATION,
    LOCAL_CANDIDATE_REPRESENTATIONS,
    MULTIRUN_REPRESENTATION,
    GeometryBatchConfig,
)


SHORT_ABLATION_SCHEMA_VERSION = "dunhuang-pairwise-short-ablation/0.2"
SOURCE_LOCK_SCHEMA_VERSION = "dunhuang-pairwise-source-lock/0.2"
STREAM_CONTRACT_SCHEMA_VERSION = "dunhuang-pairwise-stream-contract/0.2"
SELECTION_SCHEMA_VERSION = "dunhuang-pairwise-population-selection/0.2"
CHECKPOINT_EXTERNAL_RECEIPT_VERSION = (
    "dunhuang-pairwise-checkpoint-external-receipt/0.2"
)
APPROVED_COARSE_PREPROCESS_MODE = "tight_crop_letterbox"
_FORBIDDEN_COARSE_PREPROCESS_MODES = frozenset(
    (
        "full_canvas_stretch",
        "full_canvas_stretch_legacy",
        "legacy_full_canvas_stretch",
    )
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_REQUIRED_SOURCE_ROLES = frozenset(
    (
        "backend",
        "batch_provider",
        "candidate_builder",
        "checkpoint",
        "coarse_model",
        "data_lock",
        "data_stream",
        "evaluation",
        "geometry",
        "geometry_cache",
        "geometry_schema",
        "harness",
        "lazy_mask_loader",
        "local_matcher",
        "model",
        "optimal_transport",
        "sampler",
        "fragment_geometry_cache",
        "training_engine",
    )
)
_FORBIDDEN_RECEIPT_KEYS = frozenset(
    (
        "component_id",
        "device_id",
        "fragment_id",
        "group_id",
        "host_id",
        "hostname",
        "member_path",
        "pair_id",
        "password",
        "path",
        "sample_id",
        "secret",
        "ssh_password",
        "token",
    )
)
_PROCESSING_COUNT_KEYS = frozenset(
    (
        "coarse_preprocess_count",
        "geometry_build_count",
        "geometry_cache_read_count",
        "geometry_cache_write_count",
        "local_candidate_count",
        "mask_load_count",
    )
)
_COARSE_FORBIDDEN_PROCESSING_KEYS = frozenset(
    (
        "geometry_build_count",
        "geometry_cache_read_count",
        "geometry_cache_write_count",
        "local_candidate_count",
    )
)


class ShortAblationError(ValueError):
    """Raised before an unsafe, non-reproducible, or incomparable run."""


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ShortAblationError(
            "{} must be 64 lowercase hexadecimal characters".format(name)
        )
    return value


def _sha256_path(path: Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise ShortAblationError("source-lock input must be a regular file")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalized(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    # Dataclasses must precede the generic ``to_dict`` hook.  Several receipt
    # dataclasses implement ``to_dict`` in terms of this normalizer; reversing
    # the order would recurse through that convenience method forever.
    if hasattr(value, "__dataclass_fields__"):
        return _normalized(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _normalized(value.to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _normalized(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalized(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ShortAblationError("portable values cannot contain NaN/Inf")
        return value
    raise TypeError("unsupported portable value: {}".format(type(value).__name__))


def _deep_freeze_portable(value: Any) -> Any:
    """Copy portable configuration into recursively immutable containers."""

    normalized = _normalized(value)

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType(
                {str(key): freeze(child) for key, child in item.items()}
            )
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(normalized)


def _assert_portable_receipt(value: Any, location: str = "receipt") -> None:
    """Reject machine paths, credentials, and row-level identities."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
            if normalized_key in _FORBIDDEN_RECEIPT_KEYS:
                raise ShortAblationError(
                    "forbidden receipt key at {}.{}".format(location, key)
                )
            _assert_portable_receipt(item, "{}.{}".format(location, key))
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_portable_receipt(item, "{}[{}]".format(location, index))
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(
            ("/", "~/", "file:", "\\\\")
        ) or _WINDOWS_ABSOLUTE_RE.match(stripped):
            raise ShortAblationError(
                "machine-local path is forbidden at {}".format(location)
            )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    normalized = _normalized(value)
    _assert_portable_receipt(normalized)
    payload = (
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + destination.name + ".", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, order=True)
class SourceFileLock:
    """One portable role-to-content binding; the runtime path is excluded."""

    role: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not _ROLE_RE.fullmatch(self.role):
            raise ShortAblationError("source role is not portable")
        _require_sha256(self.sha256, "source SHA-256")

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "sha256": self.sha256}


@dataclass(frozen=True)
class SourceCodeLock:
    """Exact source contents used by model, provider, sampler, and runner."""

    files: Tuple[SourceFileLock, ...]
    schema_version: str = SOURCE_LOCK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_LOCK_SCHEMA_VERSION:
            raise ShortAblationError("unsupported source-lock schema")
        canonical = tuple(sorted(self.files))
        if not canonical or canonical != self.files:
            raise ShortAblationError("source-lock entries must be non-empty and sorted")
        roles = [item.role for item in canonical]
        if len(roles) != len(set(roles)):
            raise ShortAblationError("source-lock roles must be unique")
        missing = sorted(_REQUIRED_SOURCE_ROLES - set(roles))
        if missing:
            raise ShortAblationError(
                "source lock is missing required roles: {}".format(", ".join(missing))
            )

    @property
    def lock_sha256(self) -> str:
        return _content_sha256(
            {
                "schema_version": self.schema_version,
                "files": [item.to_dict() for item in self.files],
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "files": [item.to_dict() for item in self.files],
            "lock_sha256": self.lock_sha256,
        }


def build_source_code_lock(source_paths: Mapping[str, Path]) -> SourceCodeLock:
    """Observe source files for a separate preregistration/freeze step."""

    if not isinstance(source_paths, Mapping):
        raise TypeError("source_paths must be a mapping")
    entries = tuple(
        sorted(
            SourceFileLock(str(role), _sha256_path(Path(path)))
            for role, path in source_paths.items()
        )
    )
    return SourceCodeLock(entries)


def require_source_code_lock(
    required: SourceCodeLock, source_paths: Mapping[str, Path]
) -> SourceCodeLock:
    """Re-hash runtime sources and fail unless the frozen lock is exact."""

    if not isinstance(required, SourceCodeLock):
        raise TypeError("required source lock must be SourceCodeLock")
    observed = build_source_code_lock(source_paths)
    if not hmac.compare_digest(required.lock_sha256, observed.lock_sha256):
        raise ShortAblationError("runtime source code does not match source lock")
    if required.to_dict() != observed.to_dict():
        raise ShortAblationError("runtime source code does not match source lock")
    return observed


def _require_callback_source_binding(
    callback: Any, role: str, source_paths: Mapping[str, Path]
) -> None:
    """Bind a declared role to the module that implements its callback."""

    implementation = getattr(callback, "__func__", callback)
    module_name = getattr(implementation, "__module__", None)
    module = sys.modules.get(module_name) if isinstance(module_name, str) else None
    module_file = getattr(module, "__file__", None)
    if module_file is None or role not in source_paths:
        raise ShortAblationError("{} callback source cannot be bound".format(role))
    module_path = Path(module_file)
    if module_path.suffix in {".pyc", ".pyo"}:
        module_path = module_path.with_suffix(".py")
    if not hmac.compare_digest(
        _sha256_path(module_path), _sha256_path(Path(source_paths[role]))
    ):
        raise ShortAblationError(
            "{} callback implementation differs from source lock role".format(role)
        )


@dataclass(frozen=True)
class ApprovedCoarsePreprocessing:
    """Root-approved coarse preprocessing, checked before any data/model load."""

    mode: str
    output_size: Tuple[int, int]
    resize_mode: str
    content_fraction: float
    component_connectivity: int
    numeric_contract: str
    geometry_config_sha256: str
    preprocessing_sha256: str
    spatial_contract: str = (
        "single_fragment_foreground_bbox_then_aspect_preserving_centered_"
        "letterbox_no_common_canvas_position"
    )

    def __post_init__(self) -> None:
        if self.mode in _FORBIDDEN_COARSE_PREPROCESS_MODES:
            raise ShortAblationError(
                "legacy full-canvas stretch is forbidden for short training"
            )
        if self.mode != APPROVED_COARSE_PREPROCESS_MODE:
            raise ShortAblationError("only tight_crop_letterbox is approved")
        if len(self.output_size) != 2 or any(
            type(value) is not int or value < 8 for value in self.output_size
        ):
            raise ShortAblationError("coarse output size is invalid")
        if self.resize_mode not in {"nearest", "bilinear", "area"}:
            raise ShortAblationError("coarse resize mode is invalid")
        if not math.isfinite(self.content_fraction) or not (
            0.5 <= self.content_fraction <= 1.0
        ):
            raise ShortAblationError("coarse content fraction is invalid")
        if self.component_connectivity != 4:
            raise ShortAblationError(
                "only deterministic largest 4-connected component cropping is approved"
            )
        if self.numeric_contract != COARSE_NUMERIC_CONTRACT:
            raise ShortAblationError("coarse numeric contract is not approved")
        _require_sha256(self.geometry_config_sha256, "geometry config SHA-256")
        _require_sha256(self.preprocessing_sha256, "preprocessing SHA-256")
        expected = _content_sha256(self._preprocessing_payload())
        if not hmac.compare_digest(expected, self.preprocessing_sha256):
            raise ShortAblationError("coarse preprocessing digest is inconsistent")

    def _preprocessing_payload(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "output_size": list(self.output_size),
            "resize_mode": self.resize_mode,
            "content_fraction": self.content_fraction,
            "component_connectivity": self.component_connectivity,
            "numeric_contract": self.numeric_contract,
            "spatial_contract": self.spatial_contract,
        }

    @classmethod
    def from_geometry_config(
        cls, config: GeometryBatchConfig
    ) -> "ApprovedCoarsePreprocessing":
        if not isinstance(config, GeometryBatchConfig):
            raise TypeError("config must be GeometryBatchConfig")
        payload = {
            "mode": config.coarse_preprocess_mode,
            "output_size": list(config.coarse_output_size),
            "resize_mode": config.coarse_resize_mode,
            "content_fraction": config.coarse_content_fraction,
            "component_connectivity": config.coarse_component_connectivity,
            "numeric_contract": config.provenance_dict()["coarse_numeric_contract"],
            "spatial_contract": config.provenance_dict()["coarse_spatial_contract"],
        }
        return cls(
            mode=config.coarse_preprocess_mode,
            output_size=tuple(config.coarse_output_size),
            resize_mode=config.coarse_resize_mode,
            content_fraction=float(config.coarse_content_fraction),
            component_connectivity=config.coarse_component_connectivity,
            numeric_contract=str(payload["numeric_contract"]),
            geometry_config_sha256=config.fingerprint,
            preprocessing_sha256=_content_sha256(payload),
            spatial_contract=str(payload["spatial_contract"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        value = self._preprocessing_payload()
        value.update(
            {
                "geometry_config_sha256": self.geometry_config_sha256,
                "preprocessing_sha256": self.preprocessing_sha256,
            }
        )
        return value


@dataclass(frozen=True)
class StreamContract:
    """Frozen exact metadata stream identity, before bounded selection."""

    split: str
    sha256: str
    count: int
    positive_count: int
    negative_count: int
    component_count: int
    schema_version: str = STREAM_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STREAM_CONTRACT_SCHEMA_VERSION:
            raise ShortAblationError("unsupported stream-contract schema")
        if self.split not in {"train", "val"}:
            raise ShortAblationError("stream split must be train or val")
        _require_sha256(self.sha256, "stream SHA-256")
        for name in (
            "count",
            "positive_count",
            "negative_count",
            "component_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ShortAblationError("stream counts must be non-negative ints")
        if self.count <= 0 or self.component_count <= 0:
            raise ShortAblationError("stream must be non-empty")
        if self.positive_count + self.negative_count != self.count:
            raise ShortAblationError("stream class counts are inconsistent")
        if self.positive_count == 0 or self.negative_count == 0:
            raise ShortAblationError("stream must contain both classes")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, order=True)
class DatasetLabelQuota:
    """Explicit positive/negative population quota for one logical dataset."""

    dataset: str
    positive: int
    negative: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ShortAblationError("dataset quota requires a logical name")
        if self.dataset.startswith(("/", "file:")):
            raise ShortAblationError("dataset quota cannot be a local path")
        for value in (self.positive, self.negative):
            if type(value) is not int or value <= 0:
                raise ShortAblationError(
                    "every pilot dataset requires positive and negative quotas"
                )

    @property
    def total(self) -> int:
        return self.positive + self.negative

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PopulationSelectionMode(str, Enum):
    DATASET_LABEL_COMPONENT_QUOTA = "dataset_label_component_quota"
    FULL_FROZEN_STREAM = "full_frozen_stream"


@dataclass(frozen=True)
class PopulationSelectionConfig:
    """Deterministic dataset×label quotas with bounded component contribution."""

    split: str
    quotas: Tuple[DatasetLabelQuota, ...]
    seed: str
    mode: PopulationSelectionMode = (
        PopulationSelectionMode.DATASET_LABEL_COMPONENT_QUOTA
    )
    max_records_per_component_per_label: int = 2
    max_components_per_dataset_label: int = 10000

    def __post_init__(self) -> None:
        if self.split not in {"train", "val"}:
            raise ShortAblationError("selection split must be train or val")
        parsed_mode = PopulationSelectionMode(getattr(self.mode, "value", self.mode))
        object.__setattr__(self, "mode", parsed_mode)
        if not isinstance(self.seed, str) or not self.seed:
            raise ShortAblationError("selection seed is required")
        canonical = tuple(sorted(self.quotas))
        if not canonical or canonical != self.quotas:
            raise ShortAblationError("dataset quotas must be non-empty and sorted")
        datasets = [item.dataset for item in canonical]
        if len(datasets) != len(set(datasets)):
            raise ShortAblationError("dataset quotas must be unique")
        for name in (
            "max_records_per_component_per_label",
            "max_components_per_dataset_label",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ShortAblationError("selection bounds must be positive ints")

    @property
    def population_size(self) -> int:
        return sum(item.total for item in self.quotas)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "quotas": [item.to_dict() for item in self.quotas],
            "seed": self.seed,
            "mode": self.mode.value,
            "max_records_per_component_per_label": (
                self.max_records_per_component_per_label
            ),
            "max_components_per_dataset_label": (self.max_components_per_dataset_label),
        }


class EvidenceMode(str, Enum):
    COARSE = "coarse"
    LOCAL = "local"
    FUSED = "fused"


class ExecutionKind(str, Enum):
    SYNTHETIC_TRAIN_VALIDATION = "synthetic_train_validation"
    PROVIDER_FREE_DRY_RUN = "provider_free_contract_dry_run"


class AblationArmName(str, Enum):
    COARSE_ONLY = "coarse_only"
    LOCAL_DUAL_SOFTMAX = "local_dual_softmax"
    LOCAL_DUSTBIN_SINKHORN = "local_dustbin_sinkhorn"
    KEYPOINT_DUAL_SOFTMAX = "keypoint_dual_softmax"
    KEYPOINT_DUSTBIN_SINKHORN = "keypoint_dustbin_sinkhorn"
    KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM = (
        "keypoint_dustbin_sinkhorn_exact_seam"
    )
    FUSED = "fused"


def _named_config_values(value: Any, target: str) -> List[Any]:
    found: List[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) == target:
                found.append(item)
            found.extend(_named_config_values(item, target))
    elif isinstance(value, (tuple, list)):
        for item in value:
            found.extend(_named_config_values(item, target))
    return found


@dataclass(frozen=True)
class AblationArm:
    """One preregistered arm; the harness never ranks arms or chooses a winner."""

    name: AblationArmName
    evidence: EvidenceMode
    matcher_mode: Optional[str]
    model_config: Mapping[str, Any]
    optimizer_config: Mapping[str, Any]
    aggregation_config: Mapping[str, Any]
    arc_pooling: Optional[ArcPoolingConfig] = None

    def __post_init__(self) -> None:
        name = AblationArmName(getattr(self.name, "value", self.name))
        evidence = EvidenceMode(getattr(self.evidence, "value", self.evidence))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "evidence", evidence)
        expected = {
            AblationArmName.COARSE_ONLY: (EvidenceMode.COARSE, None),
            AblationArmName.LOCAL_DUAL_SOFTMAX: (
                EvidenceMode.LOCAL,
                MatcherMode.DUAL_SOFTMAX.value,
            ),
            AblationArmName.LOCAL_DUSTBIN_SINKHORN: (
                EvidenceMode.LOCAL,
                MatcherMode.DUSTBIN_SINKHORN.value,
            ),
            AblationArmName.KEYPOINT_DUAL_SOFTMAX: (
                EvidenceMode.LOCAL,
                MatcherMode.DUAL_SOFTMAX.value,
            ),
            AblationArmName.KEYPOINT_DUSTBIN_SINKHORN: (
                EvidenceMode.LOCAL,
                MatcherMode.DUSTBIN_SINKHORN.value,
            ),
            AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM: (
                EvidenceMode.LOCAL,
                MatcherMode.DUSTBIN_SINKHORN.value,
            ),
            AblationArmName.FUSED: (
                EvidenceMode.FUSED,
                MatcherMode.DUSTBIN_SINKHORN.value,
            ),
        }[name]
        if (evidence, self.matcher_mode) != expected:
            raise ShortAblationError("arm evidence/matcher contract is inconsistent")
        if name is AblationArmName.COARSE_ONLY:
            if self.arc_pooling is not None:
                raise ShortAblationError("coarse-only does not use arc pooling")
        elif not isinstance(self.arc_pooling, ArcPoolingConfig):
            raise ShortAblationError("local/fused arms require arc pooling config")
        for value, label in (
            (self.model_config, "model config"),
            (self.optimizer_config, "optimizer config"),
            (self.aggregation_config, "aggregation config"),
        ):
            if not isinstance(value, Mapping) or not value:
                raise ShortAblationError("{} must be a non-empty mapping".format(label))
            normalized = _normalized(value)
            _assert_portable_receipt(normalized, label)
        # ``frozen=True`` only protects dataclass fields, not caller-owned
        # nested dict/list objects.  Copy and recursively freeze every mapping
        # before any run/config fingerprint can be observed.
        object.__setattr__(
            self, "model_config", _deep_freeze_portable(self.model_config)
        )
        object.__setattr__(
            self,
            "optimizer_config",
            _deep_freeze_portable(self.optimizer_config),
        )
        object.__setattr__(
            self,
            "aggregation_config",
            _deep_freeze_portable(self.aggregation_config),
        )
        if name is not AblationArmName.COARSE_ONLY:
            declared_matchers = _named_config_values(
                _normalized(self.model_config), "matcher_mode"
            )
            if not declared_matchers or any(
                value != self.matcher_mode for value in declared_matchers
            ):
                raise ShortAblationError(
                    "model config matcher_mode differs from ablation arm"
                )
            declared_pooling = _named_config_values(
                _normalized(self.model_config), "arc_pooling"
            )
            expected_pooling = _normalized(self.arc_pooling)
            if not declared_pooling or any(
                _normalized(value) != expected_pooling for value in declared_pooling
            ):
                raise ShortAblationError(
                    "model config arc_pooling differs from ablation arm"
                )

    @property
    def model_config_sha256(self) -> str:
        return canonical_config_hash(self.model_config)

    @property
    def optimizer_config_sha256(self) -> str:
        return canonical_config_hash(self.optimizer_config)

    @property
    def aggregation_config_sha256(self) -> str:
        payload = {
            "evidence": self.evidence.value,
            "matcher_mode": self.matcher_mode,
            "candidate_representation": self.candidate_representation,
            "aggregation_config": _normalized(self.aggregation_config),
            "arc_pooling": (
                None if self.arc_pooling is None else _normalized(self.arc_pooling)
            ),
        }
        return canonical_config_hash(payload)

    @property
    def uses_local_geometry(self) -> bool:
        return self.evidence is not EvidenceMode.COARSE

    @property
    def candidate_representation(self) -> str:
        """Return the label-blind local-candidate representation for this arm."""

        if self.name in {
            AblationArmName.KEYPOINT_DUAL_SOFTMAX,
            AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
            AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM,
        }:
            return KEYPOINT_REPRESENTATION
        return MULTIRUN_REPRESENTATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name.value,
            "evidence": self.evidence.value,
            "matcher_mode": self.matcher_mode,
            "candidate_representation": self.candidate_representation,
            "model_config_sha256": self.model_config_sha256,
            "optimizer_config_sha256": self.optimizer_config_sha256,
            "aggregation_config_sha256": self.aggregation_config_sha256,
            "arc_pooling": (
                None if self.arc_pooling is None else _normalized(self.arc_pooling)
            ),
        }


@dataclass(frozen=True)
class PilotBudget:
    """Hard caps for a short GPU pilot; never an unbounded training plan."""

    epochs: int
    batch_size: int
    max_optimizer_steps_per_arm: int
    max_train_stream_rows: int
    max_validation_stream_rows: int
    bootstrap_repetitions: int = 200
    fixed_fpr: float = 0.05
    min_training_valid_fraction: float = 1.0
    min_validation_valid_fraction: float = 1.0
    require_equal_validation_validity_across_arms: bool = True
    gpu_minutes_lower_estimate: float = 0.0
    gpu_minutes_upper_estimate: float = 0.0
    estimate_basis: str = "preregistered_engineering_estimate_not_measurement"

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "batch_size",
            "max_optimizer_steps_per_arm",
            "max_train_stream_rows",
            "max_validation_stream_rows",
            "bootstrap_repetitions",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ShortAblationError("pilot bounds must be positive ints")
        if self.bootstrap_repetitions < 100:
            raise ShortAblationError("cluster bootstrap requires at least 100 repeats")
        if not math.isfinite(self.fixed_fpr) or not 0.0 <= self.fixed_fpr < 1.0:
            raise ShortAblationError("fixed_fpr must be in [0, 1)")
        for name in (
            "min_training_valid_fraction",
            "min_validation_valid_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ShortAblationError("coverage fractions must be in (0, 1]")
        if type(self.require_equal_validation_validity_across_arms) is not bool:
            raise TypeError("validation validity comparison policy must be bool")
        lower = float(self.gpu_minutes_lower_estimate)
        upper = float(self.gpu_minutes_upper_estimate)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ShortAblationError("GPU time estimate must be finite")
        if lower < 0.0 or upper < lower:
            raise ShortAblationError("GPU time estimate range is invalid")
        if not isinstance(self.estimate_basis, str) or not self.estimate_basis:
            raise ShortAblationError("GPU estimate basis is required")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShortAblationConfig:
    """Complete preregistered experiment configuration."""

    preprocessing: ApprovedCoarsePreprocessing
    train_selection: PopulationSelectionConfig
    validation_selection: PopulationSelectionConfig
    arms: Tuple[AblationArm, ...]
    budget: PilotBudget
    initialization_seed: str
    expected_train_selection_sha256: str
    expected_validation_selection_sha256: str
    schema_version: str = SHORT_ABLATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # This is intentionally the first substantive validation.  A legacy
        # preprocessing mode cannot survive config construction, so no runner
        # can touch manifests or instantiate a model first.
        if not isinstance(self.preprocessing, ApprovedCoarsePreprocessing):
            raise TypeError("preprocessing must be root-approved")
        if self.preprocessing.mode != APPROVED_COARSE_PREPROCESS_MODE:
            raise ShortAblationError("legacy coarse preprocessing is forbidden")
        if self.schema_version != SHORT_ABLATION_SCHEMA_VERSION:
            raise ShortAblationError("unsupported short-ablation schema")
        if self.train_selection.split != "train":
            raise ShortAblationError("train selection must target train")
        if self.validation_selection.split != "val":
            raise ShortAblationError("validation selection must target val")
        if (
            not isinstance(self.initialization_seed, str)
            or not self.initialization_seed
        ):
            raise ShortAblationError("initialization seed is required")
        if not self.arms:
            raise ShortAblationError("at least one ablation arm is required")
        names = [arm.name for arm in self.arms]
        if len(names) != len(set(names)):
            raise ShortAblationError("ablation arm names must be unique")
        order = list(AblationArmName)
        if names != sorted(names, key=order.index):
            raise ShortAblationError("ablation arms must use canonical order")
        _require_sha256(
            self.expected_train_selection_sha256,
            "expected train selection SHA-256",
        )
        _require_sha256(
            self.expected_validation_selection_sha256,
            "expected validation selection SHA-256",
        )
        train_steps = self.budget.epochs * int(
            math.ceil(self.train_selection.population_size / self.budget.batch_size)
        )
        if train_steps > self.budget.max_optimizer_steps_per_arm:
            raise ShortAblationError("configured training exceeds optimizer-step cap")
        if self.train_selection.population_size > self.budget.max_train_stream_rows:
            raise ShortAblationError("train selection exceeds stream cap")
        if (
            self.validation_selection.population_size
            > self.budget.max_validation_stream_rows
        ):
            raise ShortAblationError("validation selection exceeds stream cap")

    @property
    def config_sha256(self) -> str:
        return canonical_config_hash(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "preprocessing": self.preprocessing.to_dict(),
            "train_selection": self.train_selection.to_dict(),
            "validation_selection": self.validation_selection.to_dict(),
            "arms": [arm.to_dict() for arm in self.arms],
            "budget": self.budget.to_dict(),
            "initialization_seed": self.initialization_seed,
            "expected_train_selection_sha256": (self.expected_train_selection_sha256),
            "expected_validation_selection_sha256": (
                self.expected_validation_selection_sha256
            ),
            "winner_selection_policy": "none_short_pilot_reports_all_arms",
        }


@dataclass(frozen=True)
class RuntimeDataArtifacts:
    """Machine-local inputs used only for lock verification, never serialized."""

    required_lock: ExperimentDataLock
    split_receipt_path: Path
    stream_audit_path: Path
    synthetic_manifest_path: Path
    archive_bindings: Tuple[ArchiveBinding, ...]


@dataclass(frozen=True)
class PreparedAblationBatch:
    """Opaque provider payload plus hashes checked by the runner."""

    payload: Any
    sample_count: int
    record_sequence_sha256: str
    prepared_input_sha256: str
    local_candidate_sha256: Optional[str]
    coarse_preprocessing_sha256: str
    geometry_config_sha256: Optional[str]
    processing_counts: Mapping[str, int]
    candidate_representation: str = MULTIRUN_REPRESENTATION

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ShortAblationError("prepared batch sample_count must be positive")
        if self.candidate_representation not in LOCAL_CANDIDATE_REPRESENTATIONS:
            raise ShortAblationError("unsupported prepared candidate representation")
        _require_sha256(self.record_sequence_sha256, "batch sequence SHA-256")
        _require_sha256(self.prepared_input_sha256, "prepared input SHA-256")
        if self.local_candidate_sha256 is not None:
            _require_sha256(self.local_candidate_sha256, "local candidate SHA-256")
        _require_sha256(self.coarse_preprocessing_sha256, "batch preprocessing SHA-256")
        if self.geometry_config_sha256 is not None:
            _require_sha256(
                self.geometry_config_sha256, "batch geometry config SHA-256"
            )
        if not isinstance(self.processing_counts, Mapping):
            raise TypeError("processing_counts must be a mapping")
        if set(self.processing_counts) != set(_PROCESSING_COUNT_KEYS):
            raise ShortAblationError("processing_counts has missing/unexpected fields")
        for value in self.processing_counts.values():
            if type(value) is not int or value < 0:
                raise ShortAblationError(
                    "processing counts must be non-negative integers"
                )


@dataclass(frozen=True)
class BatchProviderContract:
    """Path-free declaration checked before streams or models are opened."""

    coarse_preprocess_mode: str
    coarse_preprocessing_sha256: str
    geometry_config_sha256: str
    cache_interface: str
    provider_version: str
    coarse_only_geometry_free: bool = True

    def __post_init__(self) -> None:
        if self.coarse_preprocess_mode in _FORBIDDEN_COARSE_PREPROCESS_MODES:
            raise ShortAblationError("legacy full-canvas preprocessing is forbidden")
        if self.coarse_preprocess_mode != APPROVED_COARSE_PREPROCESS_MODE:
            raise ShortAblationError("batch provider preprocessing is unapproved")
        _require_sha256(
            self.coarse_preprocessing_sha256,
            "provider preprocessing SHA-256",
        )
        _require_sha256(self.geometry_config_sha256, "provider geometry config SHA-256")
        for name in ("cache_interface", "provider_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ShortAblationError("provider contract strings are required")
            _assert_portable_receipt(value, "provider contract " + name)
        if type(self.coarse_only_geometry_free) is not bool:
            raise TypeError("coarse_only_geometry_free must be bool")
        if not self.coarse_only_geometry_free:
            raise ShortAblationError(
                "short-ablation provider must keep coarse-only geometry-free"
            )

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TrainBatchResult:
    loss: float
    valid_count: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.loss)):
            raise ShortAblationError("training loss is non-finite")
        if type(self.valid_count) is not int or self.valid_count < 0:
            raise ShortAblationError("training valid_count is invalid")
        normalized = _normalized(self.diagnostics)
        _assert_portable_receipt(normalized, "train diagnostics")


@dataclass(frozen=True)
class PredictionBatch:
    probability: Tensor
    valid: Tensor
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.probability.ndim != 1 or not self.probability.is_floating_point():
            raise ShortAblationError("prediction probability must be float [B]")
        if self.valid.dtype != torch.bool or tuple(self.valid.shape) != tuple(
            self.probability.shape
        ):
            raise ShortAblationError("prediction valid must be bool [B]")
        usable = self.valid
        if usable.any().item():
            values = self.probability[usable]
            if (
                (~torch.isfinite(values)).any().item()
                or (values < 0.0).any().item()
                or (values > 1.0).any().item()
            ):
                raise ShortAblationError(
                    "valid prediction probabilities must be finite in [0, 1]"
                )
        normalized = _normalized(self.diagnostics)
        _assert_portable_receipt(normalized, "prediction diagnostics")


class PreparedBatchProvider(Protocol):
    contract: BatchProviderContract

    def prepare(
        self,
        records: Sequence[TrainingPairRecord],
        *,
        arm: AblationArm,
        phase: str,
    ) -> PreparedAblationBatch:
        """Prepare one exact record sequence, optionally through a cache."""


@dataclass(frozen=True)
class BackendContract:
    """Declare real-training versus dry-run semantics before model creation."""

    execution_kind: ExecutionKind
    backend_version: str
    model_family: str
    device_type: str
    sealed_real_test_capability: bool = False

    def __post_init__(self) -> None:
        parsed = ExecutionKind(
            getattr(self.execution_kind, "value", self.execution_kind)
        )
        object.__setattr__(self, "execution_kind", parsed)
        for name in ("backend_version", "model_family"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ShortAblationError("backend contract strings are required")
            _assert_portable_receipt(value, "backend contract " + name)
        if self.device_type not in {"cpu", "cuda", "mps"}:
            raise ShortAblationError("backend device_type is unsupported")
        if type(self.sealed_real_test_capability) is not bool:
            raise TypeError("sealed_real_test_capability must be bool")
        if self.sealed_real_test_capability:
            raise ShortAblationError(
                "short-ablation backend cannot have sealed-real-test capability"
            )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["execution_kind"] = self.execution_kind.value
        return value


class ArmSession(Protocol):
    model: nn.Module
    optimizer: torch.optim.Optimizer
    model_config: Mapping[str, Any]
    optimizer_config: Mapping[str, Any]

    def train_batch(self, batch: PreparedAblationBatch) -> TrainBatchResult:
        """Run one optimizer step."""

    def predict_batch(
        self, batch: PreparedAblationBatch, *, evidence: EvidenceMode
    ) -> PredictionBatch:
        """Return one score and decision-valid bit per prepared record."""


class AblationBackend(Protocol):
    contract: BackendContract

    def create_session(self, arm: AblationArm, *, seed: int) -> ArmSession:
        """Create a fresh deterministic session for one arm."""


RecordFactory = Callable[[], Iterable[TrainingPairRecord]]


def _record_payload(record: TrainingPairRecord) -> Dict[str, Any]:
    def fragment(value: Any) -> Dict[str, Any]:
        return {
            "archive": {
                "logical_id": value.binding.logical_id,
                "format": value.binding.archive_format,
                "sha256": value.binding.sha256,
            },
            "archive_member": value.archive_member,
            "fragment_id": value.fragment_id,
            "content_sha256": value.content_sha256,
            "threshold_rule": value.threshold_rule,
        }

    return {
        "schema_version": record.schema_version,
        "pair_id": record.pair_id,
        "fragment_a": fragment(record.fragment_a),
        "fragment_b": fragment(record.fragment_b),
        "label": record.label,
        "direction_b_wrt_a": record.direction_b_wrt_a,
        "dataset_id": record.dataset_id,
        "canonical_group_id": record.canonical_group_id,
        "component_id": record.component_id,
        "split": record.split,
        "canonical_pair_key": list(record.canonical_pair_key),
        "label_origin": record.label_origin,
        "static_hard_negative_score": record.static_hard_negative_score,
        "provenance": _normalized(record.provenance),
    }


def _record_line(record: TrainingPairRecord) -> bytes:
    return _canonical_bytes(_record_payload(record)) + b"\n"


def _record_sequence_sha256(records: Sequence[TrainingPairRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_record_line(record))
    return digest.hexdigest()


def record_sequence_fingerprint(
    records: Sequence[TrainingPairRecord],
) -> str:
    """Hash one exact provider batch without exposing its row identities.

    Batch providers use this public helper in ``PreparedAblationBatch``.  The
    digest binds order, labels, fragment contents/references, split, component,
    and provenance; only the digest can enter a portable experiment receipt.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a finite sequence")
    if not records:
        raise ShortAblationError("record sequence cannot be empty")
    if not all(isinstance(record, TrainingPairRecord) for record in records):
        raise TypeError("record sequence must contain TrainingPairRecord values")
    return _record_sequence_sha256(records)


def _validate_record(
    record: TrainingPairRecord,
    split: str,
    allowed_datasets: frozenset,
    allowed_archive_bindings: frozenset,
) -> None:
    if not isinstance(record, TrainingPairRecord):
        raise TypeError("record factory must emit TrainingPairRecord")
    if record.split != split:
        raise ShortAblationError("record escaped its configured split")
    if record.dataset_id not in allowed_datasets:
        raise ShortAblationError("record dataset is absent from explicit quotas")
    if record.provenance.get("real_dunhuang_sealed_test") is not False:
        raise ShortAblationError("sealed/unknown test provenance is forbidden")
    for fragment in (record.fragment_a, record.fragment_b):
        binding_key = (
            fragment.binding.logical_id,
            fragment.binding.archive_format,
            fragment.binding.sha256,
        )
        if binding_key not in allowed_archive_bindings:
            raise ShortAblationError(
                "record archive binding is absent from the experiment data lock"
            )


def _allowed_binding_keys(
    bindings: Iterable[ArchiveBinding],
) -> frozenset:
    values = tuple(bindings)
    if not values or not all(isinstance(item, ArchiveBinding) for item in values):
        raise TypeError("allowed archive bindings must contain ArchiveBinding values")
    return frozenset(
        (item.logical_id, item.archive_format, item.sha256) for item in values
    )


def _fragment_reference_identity(fragment: Any) -> Tuple[str, ...]:
    return (
        fragment.binding.logical_id,
        fragment.binding.archive_format,
        fragment.binding.sha256,
        fragment.archive_member,
        fragment.threshold_rule,
    )


def _record_source_lineages(record: TrainingPairRecord) -> set:
    output = set()
    for key in ("source_id", "source_group_id"):
        value = record.provenance.get(key)
        if isinstance(value, str) and value.strip():
            output.add((key, value))
    return output


@dataclass
class _ObservedStream:
    contract: StreamContract
    components: set
    fragment_references: set
    fragment_content_sha256: set
    source_lineages: set
    records_by_cell: Mapping[
        Tuple[str, bool], Mapping[str, Tuple[Tuple[str, TrainingPairRecord], ...]]
    ]
    full_records: Optional[Tuple[TrainingPairRecord, ...]]


def _rank(seed: str, namespace: str, value: str) -> str:
    return hashlib.sha256(_canonical_bytes([seed, namespace, value])).hexdigest()


def _observe_and_bound_stream(
    factory: RecordFactory,
    selection: PopulationSelectionConfig,
    *,
    maximum_rows: int,
    allowed_archive_bindings: Iterable[ArchiveBinding],
) -> _ObservedStream:
    if not callable(factory):
        raise TypeError("record source must be a restartable factory")
    allowed = frozenset(item.dataset for item in selection.quotas)
    digest = hashlib.sha256()
    count = 0
    positives = 0
    components = set()
    fragment_references = set()
    fragment_content_sha256 = set()
    source_lineages = set()
    allowed_binding_keys = _allowed_binding_keys(allowed_archive_bindings)
    # Each cell stores at most R records per component.  This is bounded by
    # explicit component and R limits even if a source has millions of rows.
    mutable: Dict[Tuple[str, bool], Dict[str, List[Tuple[str, TrainingPairRecord]]]] = {
        (quota.dataset, label): {}
        for quota in selection.quotas
        for label in (False, True)
    }
    full_records: Optional[List[TrainingPairRecord]] = (
        [] if selection.mode is PopulationSelectionMode.FULL_FROZEN_STREAM else None
    )
    for record in factory():
        _validate_record(
            record,
            selection.split,
            allowed,
            allowed_binding_keys,
        )
        count += 1
        if count > maximum_rows:
            raise ShortAblationError("metadata stream exceeds configured row cap")
        positives += int(record.label)
        components.add(record.component_id)
        source_lineages.update(_record_source_lineages(record))
        for fragment in (record.fragment_a, record.fragment_b):
            fragment_references.add(_fragment_reference_identity(fragment))
            if fragment.content_sha256 is not None:
                fragment_content_sha256.add(fragment.content_sha256)
        digest.update(_record_line(record))
        if full_records is not None:
            full_records.append(record)
            continue
        cell = mutable[(record.dataset_id, record.label)]
        if record.component_id not in cell:
            if len(cell) >= selection.max_components_per_dataset_label:
                raise ShortAblationError(
                    "dataset/label cell exceeds component-memory bound"
                )
            cell[record.component_id] = []
        values = cell[record.component_id]
        if any(existing.pair_id == record.pair_id for _, existing in values):
            # The raw contract still records the duplicate row, but a bounded
            # pilot population never spends two component slots on one Pair.
            continue
        rank = _rank(
            selection.seed,
            "record/{}/{}/{}".format(
                selection.split, record.dataset_id, int(record.label)
            ),
            record.pair_id,
        )
        values.append((rank, record))
        values.sort(key=lambda item: item[0])
        del values[selection.max_records_per_component_per_label :]
    contract = StreamContract(
        split=selection.split,
        sha256=digest.hexdigest(),
        count=count,
        positive_count=positives,
        negative_count=count - positives,
        component_count=len(components),
    )
    frozen = {
        cell: {component: tuple(values) for component, values in components_map.items()}
        for cell, components_map in mutable.items()
    }
    return _ObservedStream(
        contract,
        components,
        fragment_references,
        fragment_content_sha256,
        source_lineages,
        frozen,
        None if full_records is None else tuple(full_records),
    )


def _source_disjoint_overlap_counts(
    train: _ObservedStream, validation: _ObservedStream
) -> Dict[str, int]:
    return {
        "component": len(train.components.intersection(validation.components)),
        "fragment_reference": len(
            train.fragment_references.intersection(validation.fragment_references)
        ),
        "fragment_content": len(
            train.fragment_content_sha256.intersection(
                validation.fragment_content_sha256
            )
        ),
        "source_lineage": len(
            train.source_lineages.intersection(validation.source_lineages)
        ),
    }


def freeze_stream_contract(
    factory: RecordFactory,
    selection: PopulationSelectionConfig,
    *,
    maximum_rows: int,
    allowed_archive_bindings: Iterable[ArchiveBinding],
) -> StreamContract:
    """Generate a stream contract during an explicit preregistration step."""

    return _observe_and_bound_stream(
        factory,
        selection,
        maximum_rows=maximum_rows,
        allowed_archive_bindings=allowed_archive_bindings,
    ).contract


@dataclass(frozen=True)
class PopulationSelectionReceipt:
    split: str
    selection_sha256: str
    selected_count: int
    positive_count: int
    negative_count: int
    component_count: int
    by_dataset: Mapping[str, Mapping[str, int]]
    max_selected_records_per_component_label: int
    config_sha256: str
    selection_mode: str
    order: str
    schema_version: str = SELECTION_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return _normalized(self)


def _select_population(
    observed: _ObservedStream,
    config: PopulationSelectionConfig,
) -> Tuple[Tuple[TrainingPairRecord, ...], PopulationSelectionReceipt]:
    if config.mode is PopulationSelectionMode.FULL_FROZEN_STREAM:
        if observed.full_records is None:
            raise RuntimeError("full frozen stream records were not retained")
        frozen = observed.full_records
        by_dataset: Dict[str, Dict[str, int]] = {}
        component_label_counts: Dict[Tuple[str, bool], int] = {}
        for quota in config.quotas:
            dataset_records = [
                record for record in frozen if record.dataset_id == quota.dataset
            ]
            positives = [record for record in dataset_records if record.label]
            negatives = [record for record in dataset_records if not record.label]
            if len(positives) != quota.positive or len(negatives) != quota.negative:
                raise ShortAblationError(
                    "full frozen stream dataset/class counts differ from contract"
                )
            for record in dataset_records:
                key = (record.component_id, record.label)
                component_label_counts[key] = component_label_counts.get(key, 0) + 1
            by_dataset[quota.dataset] = {
                "positive": len(positives),
                "negative": len(negatives),
                "positive_component_count": len(
                    {record.component_id for record in positives}
                ),
                "negative_component_count": len(
                    {record.component_id for record in negatives}
                ),
            }
        if len(frozen) != config.population_size:
            raise ShortAblationError("full frozen stream population size differs")
        receipt = PopulationSelectionReceipt(
            split=config.split,
            selection_sha256=_record_sequence_sha256(frozen),
            selected_count=len(frozen),
            positive_count=sum(record.label for record in frozen),
            negative_count=sum(not record.label for record in frozen),
            component_count=len({record.component_id for record in frozen}),
            by_dataset=by_dataset,
            max_selected_records_per_component_label=max(
                component_label_counts.values(), default=0
            ),
            config_sha256=canonical_config_hash(config.to_dict()),
            selection_mode=config.mode.value,
            order="frozen_source_order",
        )
        if receipt.selection_sha256 != observed.contract.sha256:
            raise ShortAblationError("full frozen selection/order digest changed")
        return frozen, receipt

    selected: List[TrainingPairRecord] = []
    by_dataset: Dict[str, Dict[str, int]] = {}
    selected_component_label_counts: Dict[Tuple[str, bool], int] = {}
    for quota in config.quotas:
        dataset_counts = {
            "positive": 0,
            "negative": 0,
            "positive_component_count": 0,
            "negative_component_count": 0,
        }
        for label, target, class_name in (
            (True, quota.positive, "positive"),
            (False, quota.negative, "negative"),
        ):
            cell = observed.records_by_cell[(quota.dataset, label)]
            component_order = sorted(
                cell,
                key=lambda component: _rank(
                    config.seed,
                    "component/{}/{}/{}".format(
                        config.split, quota.dataset, int(label)
                    ),
                    component,
                ),
            )
            cell_selected: List[TrainingPairRecord] = []
            # Round-robin over components prevents a prolific physical source
            # from exhausting a dataset/label quota before others contribute.
            for record_slot in range(config.max_records_per_component_per_label):
                for component in component_order:
                    values = cell[component]
                    if record_slot < len(values):
                        cell_selected.append(values[record_slot][1])
                        if len(cell_selected) == target:
                            break
                if len(cell_selected) == target:
                    break
            if len(cell_selected) != target:
                raise ShortAblationError(
                    "{} {} quota shortfall: requested {}, selected {}".format(
                        quota.dataset, class_name, target, len(cell_selected)
                    )
                )
            selected.extend(cell_selected)
            distinct = len({record.component_id for record in cell_selected})
            dataset_counts[class_name] = len(cell_selected)
            dataset_counts[class_name + "_component_count"] = distinct
            for record in cell_selected:
                key = (record.component_id, label)
                selected_component_label_counts[key] = (
                    selected_component_label_counts.get(key, 0) + 1
                )
        by_dataset[quota.dataset] = dataset_counts
    # A final deterministic shuffle removes dataset blocks while preserving an
    # exact, hash-bound order shared by every ablation arm.
    selected.sort(
        key=lambda record: _rank(
            config.seed,
            "selection-order/{}".format(config.split),
            record.pair_id,
        )
    )
    frozen = tuple(selected)
    receipt = PopulationSelectionReceipt(
        split=config.split,
        selection_sha256=_record_sequence_sha256(frozen),
        selected_count=len(frozen),
        positive_count=sum(record.label for record in frozen),
        negative_count=sum(not record.label for record in frozen),
        component_count=len({record.component_id for record in frozen}),
        by_dataset=by_dataset,
        max_selected_records_per_component_label=max(
            selected_component_label_counts.values(), default=0
        ),
        config_sha256=canonical_config_hash(config.to_dict()),
        selection_mode=config.mode.value,
        order="deterministic_hash_order",
    )
    return frozen, receipt


def freeze_population_selection(
    factory: RecordFactory,
    selection: PopulationSelectionConfig,
    *,
    maximum_rows: int,
    allowed_archive_bindings: Iterable[ArchiveBinding],
) -> Tuple[StreamContract, PopulationSelectionReceipt]:
    """Return raw/selection hashes for a separate freeze-before-training step."""

    observed = _observe_and_bound_stream(
        factory,
        selection,
        maximum_rows=maximum_rows,
        allowed_archive_bindings=allowed_archive_bindings,
    )
    _, receipt = _select_population(observed, selection)
    return observed.contract, receipt


@dataclass(frozen=True)
class FrozenSourceDisjointPopulations:
    """Portable result of a metadata-only freeze; contains no selected rows."""

    train_stream: StreamContract
    validation_stream: StreamContract
    train_selection: PopulationSelectionReceipt
    validation_selection: PopulationSelectionReceipt
    train_validation_component_overlap_count: int
    train_validation_fragment_reference_overlap_count: int
    train_validation_fragment_content_overlap_count: int
    train_validation_source_lineage_overlap_count: int

    def __post_init__(self) -> None:
        counts = (
            self.train_validation_component_overlap_count,
            self.train_validation_fragment_reference_overlap_count,
            self.train_validation_fragment_content_overlap_count,
            self.train_validation_source_lineage_overlap_count,
        )
        if any(value != 0 for value in counts):
            raise ShortAblationError("frozen train/validation physical sources overlap")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_stream": self.train_stream.to_dict(),
            "validation_stream": self.validation_stream.to_dict(),
            "train_selection": self.train_selection.to_dict(),
            "validation_selection": self.validation_selection.to_dict(),
            "train_validation_component_overlap_count": 0,
            "train_validation_fragment_reference_overlap_count": 0,
            "train_validation_fragment_content_overlap_count": 0,
            "train_validation_source_lineage_overlap_count": 0,
        }


def freeze_source_disjoint_populations(
    *,
    train_records: RecordFactory,
    validation_records: RecordFactory,
    train_selection: PopulationSelectionConfig,
    validation_selection: PopulationSelectionConfig,
    max_train_stream_rows: int,
    max_validation_stream_rows: int,
    allowed_archive_bindings: Iterable[ArchiveBinding],
) -> FrozenSourceDisjointPopulations:
    """Freeze exact raw/selected identities and prove component disjointness."""

    train_observed = _observe_and_bound_stream(
        train_records,
        train_selection,
        maximum_rows=max_train_stream_rows,
        allowed_archive_bindings=allowed_archive_bindings,
    )
    validation_observed = _observe_and_bound_stream(
        validation_records,
        validation_selection,
        maximum_rows=max_validation_stream_rows,
        allowed_archive_bindings=allowed_archive_bindings,
    )
    overlaps = _source_disjoint_overlap_counts(train_observed, validation_observed)
    if any(overlaps.values()):
        raise ShortAblationError("train/validation physical source identity overlaps")
    _, train_receipt = _select_population(train_observed, train_selection)
    _, validation_receipt = _select_population(
        validation_observed, validation_selection
    )
    return FrozenSourceDisjointPopulations(
        train_stream=train_observed.contract,
        validation_stream=validation_observed.contract,
        train_selection=train_receipt,
        validation_selection=validation_receipt,
        train_validation_component_overlap_count=0,
        train_validation_fragment_reference_overlap_count=0,
        train_validation_fragment_content_overlap_count=0,
        train_validation_source_lineage_overlap_count=0,
    )


def _require_stream_contract(
    required: StreamContract, observed: StreamContract
) -> None:
    if not isinstance(required, StreamContract):
        raise TypeError("required stream contract must be StreamContract")
    if required.to_dict() != observed.to_dict() or not hmac.compare_digest(
        required.sha256, observed.sha256
    ):
        raise ShortAblationError(
            "runtime metadata stream does not match frozen contract"
        )


def _batches(
    records: Sequence[TrainingPairRecord], batch_size: int
) -> Iterable[Tuple[TrainingPairRecord, ...]]:
    for start in range(0, len(records), batch_size):
        yield tuple(records[start : start + batch_size])


def _epoch_order(
    records: Sequence[TrainingPairRecord], seed: str, epoch: int
) -> Tuple[TrainingPairRecord, ...]:
    return tuple(
        sorted(
            records,
            key=lambda record: _rank(
                seed, "epoch/{:06d}".format(epoch), record.pair_id
            ),
        )
    )


def _seed_integer(seed: str) -> int:
    return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")


def _seed_runtime(seed: int) -> None:
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise ShortAblationError(
            "CUBLAS_WORKSPACE_CONFIG must use a deterministic CUDA setting"
        )
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


def _runtime_environment() -> Dict[str, Any]:
    """Portable dependency identity; deliberately excludes host/device IDs."""

    return {
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": getattr(torch.version, "cuda", None),
        "deterministic_algorithms_required": True,
    }


def _validate_session(
    session: ArmSession, arm: AblationArm, backend_contract: BackendContract
) -> None:
    if not isinstance(session.model, nn.Module):
        raise ShortAblationError("backend session must expose torch nn.Module")
    if not isinstance(session.optimizer, torch.optim.Optimizer):
        raise ShortAblationError("backend session must expose torch optimizer")
    if canonical_config_hash(session.model_config) != arm.model_config_sha256:
        raise ShortAblationError("backend model config differs from frozen arm")
    if canonical_config_hash(session.optimizer_config) != arm.optimizer_config_sha256:
        raise ShortAblationError("backend optimizer config differs from frozen arm")
    if not callable(session.train_batch) or not callable(session.predict_batch):
        raise ShortAblationError("backend session callbacks are incomplete")
    devices = {
        value.device.type
        for value in tuple(session.model.parameters()) + tuple(session.model.buffers())
    }
    if not devices:
        raise ShortAblationError("backend model must expose tensor state")
    if devices != {backend_contract.device_type}:
        raise ShortAblationError("backend model device differs from its contract")
    model_parameters = {id(value) for value in session.model.parameters()}
    optimizer_parameters = {
        id(value)
        for group in session.optimizer.param_groups
        for value in group["params"]
    }
    if not optimizer_parameters or not optimizer_parameters.issubset(model_parameters):
        raise ShortAblationError("optimizer parameters are not owned by the model")


def _prepare_checked(
    provider: PreparedBatchProvider,
    records: Sequence[TrainingPairRecord],
    arm: AblationArm,
    phase: str,
    preprocessing: ApprovedCoarsePreprocessing,
) -> PreparedAblationBatch:
    prepared = provider.prepare(records, arm=arm, phase=phase)
    if not isinstance(prepared, PreparedAblationBatch):
        raise ShortAblationError("provider must return PreparedAblationBatch")
    if prepared.sample_count != len(records):
        raise ShortAblationError("provider changed batch cardinality")
    expected_sequence = _record_sequence_sha256(records)
    if not hmac.compare_digest(prepared.record_sequence_sha256, expected_sequence):
        raise ShortAblationError("provider changed record identity/order")
    if not hmac.compare_digest(
        prepared.coarse_preprocessing_sha256,
        preprocessing.preprocessing_sha256,
    ):
        raise ShortAblationError("provider used unapproved coarse preprocessing")
    if arm.uses_local_geometry:
        if prepared.candidate_representation != arm.candidate_representation:
            raise ShortAblationError(
                "provider candidate representation differs from ablation arm"
            )
        if prepared.geometry_config_sha256 is None or not hmac.compare_digest(
            prepared.geometry_config_sha256,
            preprocessing.geometry_config_sha256,
        ):
            raise ShortAblationError("provider used unexpected local geometry config")
        if prepared.local_candidate_sha256 is None:
            raise ShortAblationError("local arm lacks candidate artifact digest")
    else:
        if prepared.local_candidate_sha256 is not None:
            raise ShortAblationError("coarse-only batch exposed local candidates")
        nonzero = {
            key: prepared.processing_counts[key]
            for key in _COARSE_FORBIDDEN_PROCESSING_KEYS
            if prepared.processing_counts[key] != 0
        }
        if nonzero:
            raise ShortAblationError(
                "coarse-only provider invoked geometry/cache/local processing"
            )
    return prepared


def _require_shared_prepared_artifacts(
    prepared: PreparedAblationBatch,
    *,
    phase: str,
    arm: AblationArm,
    prepared_inputs: Dict[Tuple[str, str, str], str],
    local_candidates: Dict[Tuple[str, str, str], str],
) -> None:
    # The record order is shared globally, while tensors/candidates are only
    # expected to be byte-identical between matchers using the same
    # representation.  Keypoint and multi-run are the intended ablation.
    key = (
        phase,
        prepared.record_sequence_sha256,
        prepared.candidate_representation,
    )
    existing = prepared_inputs.setdefault(key, prepared.prepared_input_sha256)
    if not hmac.compare_digest(existing, prepared.prepared_input_sha256):
        raise ShortAblationError(
            "prepared input tensors differ across arms within candidate representation"
        )
    if arm.uses_local_geometry:
        if prepared.local_candidate_sha256 is None:  # pragma: no cover - above
            raise ShortAblationError("local candidate digest is missing")
        candidate = local_candidates.setdefault(key, prepared.local_candidate_sha256)
        if not hmac.compare_digest(candidate, prepared.local_candidate_sha256):
            raise ShortAblationError(
                "local candidates differ within candidate representation"
            )


def _checkpoint_config(
    *,
    config: ShortAblationConfig,
    arm: AblationArm,
    data_lock: ExperimentDataLock,
    source_lock: SourceCodeLock,
    train_contract: StreamContract,
    validation_contract: StreamContract,
    train_selection: PopulationSelectionReceipt,
    validation_selection: PopulationSelectionReceipt,
    provider_contract: BatchProviderContract,
    backend_contract: BackendContract,
    runtime_environment: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "short_ablation_schema_version": config.schema_version,
        "experiment_config_sha256": config.config_sha256,
        "arm": arm.to_dict(),
        "model_config": _normalized(arm.model_config),
        "optimizer_config": _normalized(arm.optimizer_config),
        "data_lock_sha256": data_lock.lock_sha256,
        "source_lock_sha256": source_lock.lock_sha256,
        "train_stream_sha256": train_contract.sha256,
        "validation_stream_sha256": validation_contract.sha256,
        "train_selection_sha256": train_selection.selection_sha256,
        "validation_selection_sha256": validation_selection.selection_sha256,
        "preprocessing_sha256": config.preprocessing.preprocessing_sha256,
        "geometry_config_sha256": config.preprocessing.geometry_config_sha256,
        "batch_provider_contract": provider_contract.to_dict(),
        "backend_contract": backend_contract.to_dict(),
        "runtime_environment": _normalized(runtime_environment),
    }


def _dataset_macro(per_dataset: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    metric_names = ("auroc", "auprc", "f1", "brier", "ece")
    output: Dict[str, Any] = {
        "dataset_count": len(per_dataset),
        "weighting": "equal_dataset_macro",
    }
    for view in ("row", "cluster_balanced"):
        output[view] = {
            name: float(
                np.mean([float(result[view][name]) for result in per_dataset.values()])
            )
            for name in metric_names
        }
    return output


def _validation_coverage(
    records: Sequence[TrainingPairRecord], valid: Sequence[bool]
) -> Dict[str, Any]:
    if len(records) != len(valid):
        raise ShortAblationError("validation coverage cardinality changed")

    def summarize(indices: Sequence[int]) -> Dict[str, Any]:
        total = len(indices)
        usable = sum(bool(valid[index]) for index in indices)
        return {
            "total_count": total,
            "valid_count": usable,
            "invalid_count": total - usable,
            "valid_fraction": float(usable / total) if total else 0.0,
        }

    by_dataset: Dict[str, Any] = {}
    for dataset in sorted({record.dataset_id for record in records}):
        dataset_indices = [
            index
            for index, record in enumerate(records)
            if record.dataset_id == dataset
        ]
        by_dataset[dataset] = {
            "overall": summarize(dataset_indices),
            "positive": summarize(
                [index for index in dataset_indices if records[index].label]
            ),
            "negative": summarize(
                [index for index in dataset_indices if not records[index].label]
            ),
            "component_count": len(
                {records[index].component_id for index in dataset_indices}
            ),
            "valid_component_count": len(
                {
                    records[index].component_id
                    for index in dataset_indices
                    if valid[index]
                }
            ),
        }
    return {"overall": summarize(tuple(range(len(records)))), "by_dataset": by_dataset}


_LOCAL_DIAGNOSTIC_KEYS = frozenset(
    (
        "candidate_count",
        "decision_valid_count",
        "finite_problem_count",
        "matcher_mode",
    )
)
_SINKHORN_DIAGNOSTIC_KEYS = _LOCAL_DIAGNOSTIC_KEYS.union(
    (
        "col_residual_max",
        "converged_count",
        "nonconverged_finite_count",
        "row_residual_max",
    )
)
_COARSE_EXECUTION_DIAGNOSTIC_KEYS = frozenset(
    ("coarse_forward_count", "local_forward_count", "sinkhorn_call_count")
)


def _require_arm_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    arm: AblationArm,
    backend_contract: BackendContract,
    phase: str,
) -> None:
    if arm.name is AblationArmName.COARSE_ONLY:
        if not rows:
            raise ShortAblationError("{} diagnostics are missing".format(phase))
        for row in rows:
            missing = sorted(_COARSE_EXECUTION_DIAGNOSTIC_KEYS - set(row))
            if missing:
                raise ShortAblationError(
                    "{} coarse diagnostics lack {}".format(phase, ", ".join(missing))
                )
            if (
                type(row["coarse_forward_count"]) is not int
                or row["coarse_forward_count"] <= 0
            ):
                raise ShortAblationError("coarse-only did not invoke coarse forward")
            for key in ("local_forward_count", "sinkhorn_call_count"):
                if type(row[key]) is not int or row[key] != 0:
                    raise ShortAblationError(
                        "coarse-only invoked local/Sinkhorn computation"
                    )
        return
    if (
        backend_contract.execution_kind is ExecutionKind.PROVIDER_FREE_DRY_RUN
        or not arm.uses_local_geometry
    ):
        return
    if not rows:
        raise ShortAblationError("{} diagnostics are missing".format(phase))
    required = (
        _SINKHORN_DIAGNOSTIC_KEYS
        if arm.matcher_mode == MatcherMode.DUSTBIN_SINKHORN.value
        else _LOCAL_DIAGNOSTIC_KEYS
    )
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ShortAblationError(
                "{} diagnostics lack {}".format(phase, ", ".join(missing))
            )
        if row.get("matcher_mode") != arm.matcher_mode:
            raise ShortAblationError(
                "{} diagnostics matcher mode differs from arm".format(phase)
            )


def _flatten_diagnostics(value: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        name = str(key) if not prefix else prefix + "." + str(key)
        if isinstance(item, Mapping):
            output.update(_flatten_diagnostics(item, name))
        else:
            output[name] = item
    return output


def _summarize_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate numeric health signals without persisting per-batch rows."""

    flattened = [_flatten_diagnostics(row) for row in rows]
    keys = sorted({key for row in flattened for key in row})
    numeric: Dict[str, Any] = {}
    categorical: Dict[str, Any] = {}
    for key in keys:
        values = [row[key] for row in flattened if key in row and row[key] is not None]
        number_values = [
            float(value)
            for value in values
            if type(value) in {int, float} and math.isfinite(float(value))
        ]
        if len(number_values) == len(values) and values:
            numeric[key] = {
                "observation_count": len(number_values),
                "minimum": min(number_values),
                "maximum": max(number_values),
                "mean": float(np.mean(number_values)),
                "sum": float(np.sum(number_values)),
            }
        else:
            categorical[key] = sorted(
                {str(value) for value in values}, key=str.casefold
            )
    return {
        "batch_count": len(rows),
        "numeric": numeric,
        "categorical": categorical,
    }


def _evaluate_predictions(
    records: Sequence[TrainingPairRecord],
    probabilities: Sequence[float],
    valid: Sequence[bool],
    *,
    arm: AblationArm,
    checkpoint_sha256: str,
    validation_fingerprint: str,
    budget: PilotBudget,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    labels = [record.label for record in records]
    clusters = [record.component_id for record in records]
    threshold = fit_pairwise_threshold(
        probabilities,
        labels,
        valid,
        clusters,
        source_split="validation",
        validation_fingerprint_sha256=validation_fingerprint,
        checkpoint_sha256=checkpoint_sha256,
        model_config_sha256=arm.model_config_sha256,
        aggregation_config_sha256=arm.aggregation_config_sha256,
    )
    overall = evaluate_pairwise(
        probabilities,
        labels,
        valid,
        clusters,
        threshold=threshold.threshold,
    )
    by_dataset: Dict[str, Mapping[str, Any]] = {}
    for dataset in sorted({record.dataset_id for record in records}):
        index = [
            position
            for position, record in enumerate(records)
            if record.dataset_id == dataset
        ]
        by_dataset[dataset] = evaluate_pairwise(
            [probabilities[position] for position in index],
            [labels[position] for position in index],
            [valid[position] for position in index],
            [clusters[position] for position in index],
            threshold=threshold.threshold,
        )
    report = {
        "coverage": _validation_coverage(records, valid),
        "overall": overall,
        "by_dataset": by_dataset,
        "dataset_macro": _dataset_macro(by_dataset),
        "recall_at_fixed_fpr": recall_at_fixed_fpr(
            probabilities,
            labels,
            valid,
            clusters,
            maximum_fpr=budget.fixed_fpr,
        ),
        "selective_risk": selective_risk_curve(
            probabilities,
            labels,
            valid,
            clusters,
            decision_threshold=threshold.threshold,
        ),
        "cluster_bootstrap": cluster_bootstrap_threshold_metrics(
            probabilities,
            labels,
            valid,
            clusters,
            threshold=threshold.threshold,
            repetitions=budget.bootstrap_repetitions,
            seed="{}:{}".format(validation_fingerprint, arm.name.value),
        ),
        "invalid_prediction_count": int(sum(not bool(value) for value in valid)),
    }
    return report, threshold.to_dict()


@dataclass(frozen=True)
class ShortAblationArtifacts:
    """Runtime-only paths are intentionally separate from portable receipts."""

    run_directory: Path
    run_receipt_path: Path
    checkpoint_paths: Mapping[str, Path]
    checkpoint_receipt_paths: Mapping[str, Path]
    threshold_paths: Mapping[str, Path]
    receipt: Mapping[str, Any]


def run_short_ablation(
    *,
    config: ShortAblationConfig,
    data_artifacts: RuntimeDataArtifacts,
    required_train_stream: StreamContract,
    required_validation_stream: StreamContract,
    required_source_lock: SourceCodeLock,
    source_paths: Mapping[str, Path],
    train_records: RecordFactory,
    validation_records: RecordFactory,
    batch_provider: PreparedBatchProvider,
    backend: AblationBackend,
    output_root: Path,
) -> ShortAblationArtifacts:
    """Run a bounded synthetic train/validation ablation and write receipts.

    The call order is security-significant: approved preprocessing and source/
    data locks are checked, both metadata streams and their source-disjointness
    are verified, and both selected populations match frozen hashes *before*
    ``backend.create_session`` can instantiate or deserialize any model.
    """

    if not isinstance(config, ShortAblationConfig):
        raise TypeError("config must be ShortAblationConfig")
    if config.preprocessing.mode != APPROVED_COARSE_PREPROCESS_MODE:
        raise ShortAblationError("legacy coarse preprocessing is forbidden")
    if not isinstance(data_artifacts, RuntimeDataArtifacts):
        raise TypeError("data_artifacts must be RuntimeDataArtifacts")
    if not callable(getattr(batch_provider, "prepare", None)):
        raise TypeError("batch_provider is incomplete")
    if not callable(getattr(backend, "create_session", None)):
        raise TypeError("backend is incomplete")

    # Provider policy is inspected before any data artifact or model session.
    provider_contract = getattr(batch_provider, "contract", None)
    if not isinstance(provider_contract, BatchProviderContract):
        raise ShortAblationError("batch provider lacks a frozen contract")
    if provider_contract.coarse_preprocess_mode != config.preprocessing.mode:
        raise ShortAblationError("batch provider preprocessing mode differs")
    if not hmac.compare_digest(
        provider_contract.coarse_preprocessing_sha256,
        config.preprocessing.preprocessing_sha256,
    ):
        raise ShortAblationError("batch provider preprocessing hash differs")
    if not hmac.compare_digest(
        provider_contract.geometry_config_sha256,
        config.preprocessing.geometry_config_sha256,
    ):
        raise ShortAblationError("batch provider geometry hash differs")
    backend_contract = getattr(backend, "contract", None)
    if not isinstance(backend_contract, BackendContract):
        raise ShortAblationError("backend lacks a frozen contract")

    _require_callback_source_binding(
        batch_provider.prepare, "batch_provider", source_paths
    )
    _require_callback_source_binding(backend.create_session, "backend", source_paths)
    source_lock = require_source_code_lock(required_source_lock, source_paths)
    data_lock = require_locked_artifacts(
        required_lock=data_artifacts.required_lock,
        split_receipt_path=data_artifacts.split_receipt_path,
        stream_audit_path=data_artifacts.stream_audit_path,
        synthetic_manifest_path=data_artifacts.synthetic_manifest_path,
        archive_bindings=data_artifacts.archive_bindings,
    )
    train_observed = _observe_and_bound_stream(
        train_records,
        config.train_selection,
        maximum_rows=config.budget.max_train_stream_rows,
        allowed_archive_bindings=data_artifacts.archive_bindings,
    )
    validation_observed = _observe_and_bound_stream(
        validation_records,
        config.validation_selection,
        maximum_rows=config.budget.max_validation_stream_rows,
        allowed_archive_bindings=data_artifacts.archive_bindings,
    )
    _require_stream_contract(required_train_stream, train_observed.contract)
    _require_stream_contract(required_validation_stream, validation_observed.contract)
    source_overlap_counts = _source_disjoint_overlap_counts(
        train_observed, validation_observed
    )
    if any(source_overlap_counts.values()):
        raise ShortAblationError("train/validation physical source identity overlaps")
    train_population, train_selection = _select_population(
        train_observed, config.train_selection
    )
    validation_population, validation_selection = _select_population(
        validation_observed, config.validation_selection
    )
    if not hmac.compare_digest(
        train_selection.selection_sha256,
        config.expected_train_selection_sha256,
    ):
        raise ShortAblationError("train selection differs from frozen fingerprint")
    if not hmac.compare_digest(
        validation_selection.selection_sha256,
        config.expected_validation_selection_sha256,
    ):
        raise ShortAblationError("validation selection differs from frozen fingerprint")

    runtime_environment = _runtime_environment()
    run_binding = {
        "schema_version": config.schema_version,
        "config_sha256": config.config_sha256,
        "data_lock_sha256": data_lock.lock_sha256,
        "source_lock_sha256": source_lock.lock_sha256,
        "train_stream_sha256": train_observed.contract.sha256,
        "validation_stream_sha256": validation_observed.contract.sha256,
        "train_selection_sha256": train_selection.selection_sha256,
        "validation_selection_sha256": validation_selection.selection_sha256,
        "batch_provider_contract_sha256": _content_sha256(provider_contract.to_dict()),
        "backend_contract_sha256": _content_sha256(backend_contract.to_dict()),
        "runtime_environment_sha256": _content_sha256(runtime_environment),
    }
    run_fingerprint = _content_sha256(run_binding)
    destination = Path(output_root) / ("run-" + run_fingerprint[:16])
    if destination.exists():
        raise ShortAblationError("run destination already exists; overwrite forbidden")
    Path(output_root).mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=".partial-short-ablation-", dir=str(output_root))
    )
    checkpoint_paths: Dict[str, Path] = {}
    checkpoint_receipt_paths: Dict[str, Path] = {}
    threshold_paths: Dict[str, Path] = {}
    arm_receipts = []
    shared_validation_validity_sha256: Optional[str] = None
    shared_prepared_inputs: Dict[Tuple[str, str, str], str] = {}
    shared_local_candidates: Dict[Tuple[str, str, str], str] = {}
    try:
        common_seed = _seed_integer(config.initialization_seed)
        for arm in config.arms:
            _seed_runtime(common_seed)
            session = backend.create_session(arm, seed=common_seed)
            _validate_session(session, arm, backend_contract)
            losses: List[float] = []
            valid_counts = 0
            optimizer_steps = 0
            training_diagnostics: List[Mapping[str, Any]] = []
            training_processing: List[Mapping[str, Any]] = []
            for epoch in range(config.budget.epochs):
                ordered = _epoch_order(
                    train_population, config.initialization_seed, epoch
                )
                for records in _batches(ordered, config.budget.batch_size):
                    prepared = _prepare_checked(
                        batch_provider,
                        records,
                        arm,
                        "train",
                        config.preprocessing,
                    )
                    _require_shared_prepared_artifacts(
                        prepared,
                        phase="train",
                        arm=arm,
                        prepared_inputs=shared_prepared_inputs,
                        local_candidates=shared_local_candidates,
                    )
                    result = session.train_batch(prepared)
                    training_processing.append(dict(prepared.processing_counts))
                    if not isinstance(result, TrainBatchResult):
                        raise ShortAblationError(
                            "session train_batch must return TrainBatchResult"
                        )
                    losses.append(float(result.loss))
                    if result.valid_count > len(records):
                        raise ShortAblationError(
                            "training valid_count exceeds batch cardinality"
                        )
                    valid_counts += result.valid_count
                    training_diagnostics.append(dict(result.diagnostics))
                    optimizer_steps += 1
                    if optimizer_steps > config.budget.max_optimizer_steps_per_arm:
                        raise ShortAblationError("optimizer-step cap exceeded")

            training_presented_count = config.budget.epochs * len(train_population)
            training_valid_fraction = valid_counts / training_presented_count
            if (
                valid_counts <= 0
                or training_valid_fraction < config.budget.min_training_valid_fraction
            ):
                raise ShortAblationError(
                    "training valid coverage is below the preregistered minimum"
                )

            checkpoint_config = _checkpoint_config(
                config=config,
                arm=arm,
                data_lock=data_lock,
                source_lock=source_lock,
                train_contract=train_observed.contract,
                validation_contract=validation_observed.contract,
                train_selection=train_selection,
                validation_selection=validation_selection,
                provider_contract=provider_contract,
                backend_contract=backend_contract,
                runtime_environment=runtime_environment,
            )
            checkpoint_path = partial / (arm.name.value + ".pt")
            checkpoint = save_checkpoint(
                checkpoint_path,
                session.model,
                config=checkpoint_config,
                epoch=config.budget.epochs,
                optimizer=session.optimizer,
                metrics={
                    "optimizer_steps": optimizer_steps,
                    "mean_training_loss": float(np.mean(losses)),
                    "last_training_loss": float(losses[-1]),
                    "training_valid_count": valid_counts,
                    "training_diagnostics": _summarize_diagnostics(
                        training_diagnostics
                    ),
                },
                provenance={
                    "run_fingerprint_sha256": run_fingerprint,
                    "source_lock_sha256": source_lock.lock_sha256,
                    "data_lock_sha256": data_lock.lock_sha256,
                    "sealed_real_test_accessed": False,
                    "execution_kind": backend_contract.execution_kind.value,
                },
            )
            external_checkpoint = {
                "schema_version": CHECKPOINT_EXTERNAL_RECEIPT_VERSION,
                "arm": arm.name.value,
                "checkpoint_file_sha256": checkpoint.file_sha256,
                "checkpoint_config_sha256": checkpoint.config_hash,
                "model_config_sha256": arm.model_config_sha256,
                "optimizer_config_sha256": arm.optimizer_config_sha256,
                "source_lock_sha256": source_lock.lock_sha256,
                "data_lock_sha256": data_lock.lock_sha256,
                "train_selection_sha256": train_selection.selection_sha256,
                "epoch_count": config.budget.epochs,
                "optimizer_steps": optimizer_steps,
                "execution_kind": backend_contract.execution_kind.value,
            }
            external_checkpoint["content_sha256"] = _content_sha256(external_checkpoint)
            checkpoint_receipt_path = partial / (arm.name.value + ".checkpoint.json")
            checkpoint_receipt_file_sha = _atomic_json(
                checkpoint_receipt_path, external_checkpoint
            )

            probabilities: List[float] = []
            validity: List[bool] = []
            validation_diagnostics: List[Mapping[str, Any]] = []
            validation_processing: List[Mapping[str, Any]] = []
            for records in _batches(validation_population, config.budget.batch_size):
                prepared = _prepare_checked(
                    batch_provider,
                    records,
                    arm,
                    "validation",
                    config.preprocessing,
                )
                _require_shared_prepared_artifacts(
                    prepared,
                    phase="validation",
                    arm=arm,
                    prepared_inputs=shared_prepared_inputs,
                    local_candidates=shared_local_candidates,
                )
                prediction = session.predict_batch(prepared, evidence=arm.evidence)
                validation_processing.append(dict(prepared.processing_counts))
                if not isinstance(prediction, PredictionBatch):
                    raise ShortAblationError(
                        "session predict_batch must return PredictionBatch"
                    )
                if prediction.probability.shape[0] != len(records):
                    raise ShortAblationError("prediction cardinality changed")
                probabilities.extend(
                    float(value)
                    for value in prediction.probability.detach().cpu().tolist()
                )
                validity.extend(
                    bool(value) for value in prediction.valid.detach().cpu().tolist()
                )
                validation_diagnostics.append(dict(prediction.diagnostics))
            validation_valid_count = sum(validity)
            validation_valid_fraction = validation_valid_count / len(validity)
            if (
                validation_valid_count <= 0
                or validation_valid_fraction
                < config.budget.min_validation_valid_fraction
            ):
                raise ShortAblationError(
                    "validation valid coverage is below the preregistered minimum"
                )
            validity_sha256 = _content_sha256([bool(value) for value in validity])
            if shared_validation_validity_sha256 is None:
                shared_validation_validity_sha256 = validity_sha256
            elif (
                config.budget.require_equal_validation_validity_across_arms
                and not hmac.compare_digest(
                    validity_sha256, shared_validation_validity_sha256
                )
            ):
                raise ShortAblationError(
                    "validation validity differs across ablation arms"
                )
            _require_arm_diagnostics(
                training_diagnostics,
                arm,
                backend_contract,
                "training",
            )
            _require_arm_diagnostics(
                validation_diagnostics,
                arm,
                backend_contract,
                "validation",
            )
            evaluation, threshold = _evaluate_predictions(
                validation_population,
                probabilities,
                validity,
                arm=arm,
                checkpoint_sha256=checkpoint.file_sha256,
                validation_fingerprint=validation_selection.selection_sha256,
                budget=config.budget,
            )
            threshold_path = partial / (arm.name.value + ".threshold.json")
            threshold_file_sha = _atomic_json(threshold_path, threshold)
            arm_receipts.append(
                {
                    "arm": arm.to_dict(),
                    "optimizer_steps": optimizer_steps,
                    "mean_training_loss": float(np.mean(losses)),
                    "last_training_loss": float(losses[-1]),
                    "training_valid_count": valid_counts,
                    "training_presented_count": training_presented_count,
                    "training_valid_fraction": training_valid_fraction,
                    "validation_validity_sha256": validity_sha256,
                    "training_diagnostics": _summarize_diagnostics(
                        training_diagnostics
                    ),
                    "validation_diagnostics": _summarize_diagnostics(
                        validation_diagnostics
                    ),
                    "training_processing_counts": _summarize_diagnostics(
                        training_processing
                    ),
                    "validation_processing_counts": _summarize_diagnostics(
                        validation_processing
                    ),
                    "checkpoint": {
                        "file_sha256": checkpoint.file_sha256,
                        "config_sha256": checkpoint.config_hash,
                        "external_receipt_content_sha256": external_checkpoint[
                            "content_sha256"
                        ],
                        "external_receipt_file_sha256": (checkpoint_receipt_file_sha),
                    },
                    "threshold": {
                        "content_sha256": _content_sha256(threshold),
                        "file_sha256": threshold_file_sha,
                        "artifact": threshold,
                    },
                    "evaluation": evaluation,
                    "selection_status": "not_selected_no_winner_in_short_pilot",
                }
            )
            checkpoint_paths[arm.name.value] = checkpoint_path
            checkpoint_receipt_paths[arm.name.value] = checkpoint_receipt_path
            threshold_paths[arm.name.value] = threshold_path

        receipt: Dict[str, Any] = {
            "schema_version": config.schema_version,
            "status": (
                "complete_provider_free_contract_dry_run"
                if backend_contract.execution_kind
                is ExecutionKind.PROVIDER_FREE_DRY_RUN
                else "complete_short_synthetic_validation_only"
            ),
            "scope": (
                "provider_free_contract_smoke_no_mask_pixels_no_real_test"
                if backend_contract.execution_kind
                is ExecutionKind.PROVIDER_FREE_DRY_RUN
                else "pairwise_mask_only_train_validation_no_real_test"
            ),
            "run_fingerprint_sha256": run_fingerprint,
            "experiment_config_sha256": config.config_sha256,
            "data_lock_sha256": data_lock.lock_sha256,
            "source_lock": source_lock.to_dict(),
            "preprocessing": config.preprocessing.to_dict(),
            "batch_provider_contract": provider_contract.to_dict(),
            "backend_contract": backend_contract.to_dict(),
            "runtime_environment": runtime_environment,
            "streams": {
                "train": train_observed.contract.to_dict(),
                "validation": validation_observed.contract.to_dict(),
                "train_validation_component_overlap_count": 0,
                "train_validation_fragment_reference_overlap_count": 0,
                "train_validation_fragment_content_overlap_count": 0,
                "train_validation_source_lineage_overlap_count": 0,
            },
            "selections": {
                "train": train_selection.to_dict(),
                "validation": validation_selection.to_dict(),
            },
            "budget": config.budget.to_dict(),
            "arm_results": arm_receipts,
            "comparison_policy": {
                "same_physical_population_all_arms": True,
                "same_initialization_seed_all_arms": True,
                "same_preprocessing_contract_all_arms": True,
                "provider_attested_record_sequence_equal_all_arms": True,
                "provider_attested_prepared_input_digest_equal_within_representation": True,
                "provider_attested_local_candidate_digest_equal_within_representation": True,
                "equal_validation_validity_required": (
                    config.budget.require_equal_validation_validity_across_arms
                ),
                "row_and_component_metrics_reported": True,
                "dataset_macro_reported": True,
                "winner_selected": False,
            },
            "real_dunhuang_sealed_test": {
                "record_count": 0,
                "accessed": False,
                "uploaded": False,
                "used_for_threshold": False,
            },
            "portable_receipt": {
                "contains_machine_paths": False,
                "contains_row_level_identifiers": False,
                "contains_physical_source_identifiers": False,
            },
        }
        receipt["content_sha256"] = _content_sha256(receipt)
        run_receipt_path = partial / "run_receipt.json"
        _atomic_json(run_receipt_path, receipt)
        os.replace(str(partial), str(destination))
        return ShortAblationArtifacts(
            run_directory=destination,
            run_receipt_path=destination / run_receipt_path.name,
            checkpoint_paths={
                name: destination / path.name for name, path in checkpoint_paths.items()
            },
            checkpoint_receipt_paths={
                name: destination / path.name
                for name, path in checkpoint_receipt_paths.items()
            },
            threshold_paths={
                name: destination / path.name for name, path in threshold_paths.items()
            },
            receipt=receipt,
        )
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


__all__ = [
    "APPROVED_COARSE_PREPROCESS_MODE",
    "AblationArm",
    "AblationArmName",
    "AblationBackend",
    "ApprovedCoarsePreprocessing",
    "BackendContract",
    "BatchProviderContract",
    "DatasetLabelQuota",
    "EvidenceMode",
    "ExecutionKind",
    "FrozenSourceDisjointPopulations",
    "PilotBudget",
    "PopulationSelectionConfig",
    "PopulationSelectionMode",
    "PopulationSelectionReceipt",
    "PredictionBatch",
    "PreparedAblationBatch",
    "PreparedBatchProvider",
    "RuntimeDataArtifacts",
    "SHORT_ABLATION_SCHEMA_VERSION",
    "ShortAblationArtifacts",
    "ShortAblationConfig",
    "ShortAblationError",
    "SourceCodeLock",
    "SourceFileLock",
    "StreamContract",
    "TrainBatchResult",
    "build_source_code_lock",
    "freeze_population_selection",
    "freeze_source_disjoint_populations",
    "freeze_stream_contract",
    "record_sequence_fingerprint",
    "require_source_code_lock",
    "run_short_ablation",
]

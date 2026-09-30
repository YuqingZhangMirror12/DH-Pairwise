"""Archive-attested, geometry-free coarse provider for C0-N-Q1.

The provider is deliberately narrower than the generic short-ablation bridge:
it accepts historical MM/ECCV records only, re-attests every endpoint through
``HistoricalIdentityIndex``, loads masks only from a fully observed canonical
archive pair, and prepares the coarse Siamese tensors.  Candidate geometry,
geometry caches, local patches and optimal transport are not reachable here.

Construction is a data preflight.  It must happen before a model is created:
``LazyMaskArchiveLoader.verify_all_sources`` is completed and compared with
the archive locks in the frozen C0-N-Q1 receipt while the loader still has zero
member requests, archive opens and decoded masks.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
from scipy import ndimage
from torch import Tensor

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    FREEZE_SCHEMA_VERSION,
    HISTORICAL_TEST_ACCESS_EVIDENCE,
    FragmentIdentity,
    HistoricalIdentityError,
    HistoricalIdentityIndex,
    VerifiedPair,
    historical_identity_index_content_sha256,
)
from staging.pairwise_v0_2.pairwise_data.lazy_mask_loader import (
    ArchiveVerificationReceipt,
    LazyMaskArchiveLoader,
    LazyMaskLoaderError,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.c0_coarse_backend import (
    C0_COARSE_MODEL_CONFIG,
    C0_COARSE_OPTIMIZER_CONFIG,
    C0CoarsePayload,
    c0_coarse_payload_content_sha256,
)
from staging.pairwise_v0_2.training.checkpoint import canonical_config_hash
from staging.pairwise_v0_2.training.geometry_batch import (
    GeometryBatchConfig,
    GeometryBatchError,
    preprocess_coarse_mask,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    ApprovedCoarsePreprocessing,
    BatchProviderContract,
    EvidenceMode,
    PreparedAblationBatch,
    record_sequence_fingerprint,
)


C0_COARSE_PROVIDER_VERSION = "c0-n-q1-coarse-provider/0.5"
C0_COARSE_PROVIDER_RECEIPT_VERSION = "c0-n-q1-coarse-provider-receipt/0.5"
C0_COARSE_TENSOR_BYTES = 128 * 128 * 4
C0_PRODUCTION_MAX_CACHED_FRAGMENTS = 131_072
C0_PRODUCTION_MAX_CACHE_BYTES = 8 * 1024**3
_C0_DATASETS = ("mm_augmented", "eccv_1113data")
_C0_MAX_INPUT_PIXELS = 100_000_000
_C0_PHASE_TO_SPLIT = MappingProxyType({"train": "train", "validation": "val"})
_ZERO_LOCAL_PROCESSING = MappingProxyType(
    {
        "geometry_build_count": 0,
        "geometry_cache_read_count": 0,
        "geometry_cache_write_count": 0,
        "local_candidate_count": 0,
    }
)


class C0CoarseProviderError(ValueError):
    """Raised when C0 data cannot be proven safe before tensor delivery."""


@dataclass(frozen=True)
class C0CoarseProviderConfig:
    """Fixed-size host-memory bounds for the coarse tensor LRU."""

    max_batch_size: int = 256
    max_cached_fragments: int = C0_PRODUCTION_MAX_CACHED_FRAGMENTS
    max_cache_bytes: int = C0_PRODUCTION_MAX_CACHE_BYTES

    def __post_init__(self) -> None:
        if type(self.max_batch_size) is not int or self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        for name in ("max_cached_fragments", "max_cache_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("{} must be a non-negative integer".format(name))
        if (self.max_cached_fragments == 0) != (self.max_cache_bytes == 0):
            raise ValueError("both coarse memo limits must be zero or positive")


@dataclass(frozen=True)
class C0CoarseProviderStats:
    """Portable cumulative counters plus the current bounded memo footprint."""

    batch_count: int
    record_count: int
    fragment_request_count: int
    memo_hit_count: int
    memo_miss_count: int
    mask_loader_call_count: int
    archive_decode_count: int
    loader_cache_hit_count: int
    coarse_preprocess_count: int
    memo_eviction_count: int
    memo_entry_count: int
    memo_byte_count: int

    def to_dict(self) -> Dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class C0RecordOrderAttestation:
    """Physical/content-bound digest for one exact finite record sequence."""

    phase: str
    count: int
    record_sequence_sha256: str
    verified_pairs: Tuple[VerifiedPair, ...]

    def __post_init__(self) -> None:
        if self.phase not in _C0_PHASE_TO_SPLIT:
            raise C0CoarseProviderError("record attestation phase is invalid")
        if self.count != len(self.verified_pairs) or self.count <= 0:
            raise C0CoarseProviderError("record attestation count is inconsistent")
        _require_sha256(self.record_sequence_sha256, "record sequence digest")


@dataclass(frozen=True)
class _FragmentMemoKey:
    dataset_id: str
    archive_sha256: str
    archive_member: str
    content_sha256: str
    threshold_rule: str
    preprocessing_sha256: str


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise C0CoarseProviderError("{} must be a lowercase SHA-256".format(name))
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_bindings() -> Mapping[str, Any]:
    # Kept as a function so tests can replace the two canonical module globals
    # without adding a production injection point for weaker bindings.
    return {
        "mm_augmented": MM_CANONICAL_BINDING,
        "eccv_1113data": ECCV_CANONICAL_BINDING,
    }


def _validate_freeze_receipt(
    receipt: Mapping[str, Any],
    expected_content_sha256: str,
) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(receipt, Mapping):
        raise TypeError("freeze_receipt must be a mapping")
    expected_content_sha = _require_sha256(
        expected_content_sha256, "expected freeze receipt digest"
    )
    if receipt.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise C0CoarseProviderError("unsupported C0 freeze receipt schema")
    if receipt.get("status") != "pass_metadata_only_no_model_execution":
        raise C0CoarseProviderError("C0 freeze receipt has not passed")

    content_sha = _require_sha256(
        receipt.get("content_sha256"), "freeze receipt digest"
    )
    if not hmac.compare_digest(content_sha, expected_content_sha):
        raise C0CoarseProviderError(
            "C0 freeze receipt differs from the run-plan binding"
        )
    unsigned = dict(receipt)
    unsigned.pop("content_sha256", None)
    observed_content_sha = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if not hmac.compare_digest(content_sha, observed_content_sha):
        raise C0CoarseProviderError("C0 freeze receipt content digest changed")

    scope = receipt.get("scope")
    if not isinstance(scope, Mapping):
        raise C0CoarseProviderError("C0 freeze receipt lacks scope")
    if set(scope) != {
        "experiment",
        "datasets",
        "pair_stream_splits_read",
        "historical_test_access",
        "sealed_real_read",
        "mask_pixels_decoded",
        "model_executed",
    }:
        raise C0CoarseProviderError("C0 freeze receipt scope schema changed")
    if set(scope.get("datasets", ())) != set(_C0_DATASETS):
        raise C0CoarseProviderError("C0 freeze receipt dataset scope changed")
    if set(scope.get("pair_stream_splits_read", ())) != {"train", "val"}:
        raise C0CoarseProviderError("C0 freeze receipt split scope changed")
    if scope.get("historical_test_access") != dict(HISTORICAL_TEST_ACCESS_EVIDENCE):
        raise C0CoarseProviderError("C0 historical-test access evidence changed")
    for name in ("sealed_real_read", "mask_pixels_decoded", "model_executed"):
        if scope.get(name) is not False:
            raise C0CoarseProviderError("C0 freeze receipt scope is unsafe")

    locks = receipt.get("locks")
    archives = locks.get("archives") if isinstance(locks, Mapping) else None
    if not isinstance(archives, Mapping) or set(archives) != set(_C0_DATASETS):
        raise C0CoarseProviderError("C0 freeze receipt archive locks changed")
    canonical = _canonical_bindings()
    normalized: Dict[str, Mapping[str, Any]] = {}
    for dataset_id in _C0_DATASETS:
        lock = archives.get(dataset_id)
        binding = canonical[dataset_id]
        if not isinstance(lock, Mapping):
            raise C0CoarseProviderError("C0 archive lock is invalid")
        if (
            lock.get("format") != binding.archive_format
            or lock.get("sha256") != binding.sha256
            or type(lock.get("bytes")) is not int
            or int(lock["bytes"]) <= 0
        ):
            raise C0CoarseProviderError("C0 archive lock is not canonical")
        normalized[dataset_id] = MappingProxyType(
            {
                "format": binding.archive_format,
                "sha256": binding.sha256,
                "bytes": int(lock["bytes"]),
            }
        )
    return MappingProxyType(normalized)


def _validate_identity_index_locks(
    identity_index: HistoricalIdentityIndex,
    freeze_receipt: Mapping[str, Any],
) -> None:
    """Bind the in-memory identity join to the receipt's frozen inputs.

    The index intentionally retains no runtime paths.  Its only portable
    construction evidence is therefore the exact SHA/size map produced by
    ``HistoricalIdentityIndex.from_files`` plus hashes of the parsed split
    candidate and seed.  All of them are required before archive preflight.
    """

    locks = freeze_receipt.get("locks")
    if not isinstance(locks, Mapping):
        raise C0CoarseProviderError("C0 freeze receipt lacks identity locks")
    expected_artifacts = locks.get("metadata_artifacts")
    observed_artifacts = identity_index.artifact_locks
    if not isinstance(expected_artifacts, Mapping) or not isinstance(
        observed_artifacts, Mapping
    ):
        raise C0CoarseProviderError("historical identity artifact locks are invalid")
    if set(observed_artifacts) != set(expected_artifacts):
        raise C0CoarseProviderError(
            "historical identity artifact inventory differs from the freeze"
        )
    for role in sorted(expected_artifacts):
        expected = expected_artifacts[role]
        observed = observed_artifacts[role]
        if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
            raise C0CoarseProviderError("historical identity artifact lock is invalid")
        if set(expected) != {"bytes", "sha256"} or set(observed) != {
            "bytes",
            "sha256",
        }:
            raise C0CoarseProviderError(
                "historical identity artifact lock fields changed"
            )
        if (
            type(expected.get("bytes")) is not int
            or int(expected["bytes"]) <= 0
            or type(observed.get("bytes")) is not int
            or int(observed["bytes"]) != int(expected["bytes"])
        ):
            raise C0CoarseProviderError(
                "historical identity artifact byte count changed"
            )
        expected_sha = _require_sha256(
            expected.get("sha256"), "frozen identity artifact digest"
        )
        observed_sha = _require_sha256(
            observed.get("sha256"), "observed identity artifact digest"
        )
        if not hmac.compare_digest(expected_sha, observed_sha):
            raise C0CoarseProviderError("historical identity artifact digest changed")

    split_bindings = (
        (
            "split_candidate_sha256",
            identity_index.split_candidate_id,
            "historical split candidate",
        ),
        (
            "split_seed_sha256",
            identity_index.split_seed,
            "historical split seed",
        ),
    )
    for receipt_key, value, label in split_bindings:
        if not isinstance(value, str) or not value:
            raise C0CoarseProviderError("{} is absent".format(label))
        expected_sha = _require_sha256(
            locks.get(receipt_key), "frozen {} digest".format(label)
        )
        observed_sha = hashlib.sha256(value.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(expected_sha, observed_sha):
            raise C0CoarseProviderError("{} differs from the freeze".format(label))


def _validate_identity_index_content(
    identity_index: HistoricalIdentityIndex,
    expected_content_sha256: str,
) -> str:
    """Recompute and externally anchor the complete private identity mapping."""

    expected = _require_sha256(
        expected_content_sha256, "expected historical identity index digest"
    )
    try:
        observed = historical_identity_index_content_sha256(identity_index)
    except (HistoricalIdentityError, TypeError, ValueError) as exc:
        raise C0CoarseProviderError(
            "historical identity index content is invalid"
        ) from exc
    reported = _require_sha256(
        identity_index.content_sha256, "reported historical identity index digest"
    )
    if not hmac.compare_digest(reported, observed):
        raise C0CoarseProviderError(
            "historical identity index self-reported digest is inconsistent"
        )
    if not hmac.compare_digest(observed, expected):
        raise C0CoarseProviderError(
            "historical identity index content differs from the run plan"
        )
    return observed


def _exact_preprocessing() -> Tuple[GeometryBatchConfig, ApprovedCoarsePreprocessing]:
    config = GeometryBatchConfig(
        coarse_output_size=(128, 128),
        coarse_resize_mode="bilinear",
        coarse_preprocess_mode="tight_crop_letterbox",
        coarse_content_fraction=0.875,
        coarse_component_connectivity=4,
        max_batch_size=256,
    )
    approved = ApprovedCoarsePreprocessing.from_geometry_config(config)
    if (
        approved.mode != "tight_crop_letterbox"
        or approved.output_size != (128, 128)
        or approved.resize_mode != "bilinear"
        or approved.content_fraction != 0.875
        or approved.component_connectivity != 4
    ):
        raise C0CoarseProviderError("C0 coarse preprocessing contract changed")
    return config, approved


def _verify_archive_preflight(
    loader: LazyMaskArchiveLoader,
    archive_locks: Mapping[str, Mapping[str, Any]],
) -> Tuple[ArchiveVerificationReceipt, ...]:
    if not isinstance(loader, LazyMaskArchiveLoader):
        raise TypeError("mask_loader must be LazyMaskArchiveLoader")
    before = loader.stats.to_dict()
    for name in ("requests", "archive_opens", "decoded_masks", "decoded_bytes"):
        if before[name] != 0:
            raise C0CoarseProviderError(
                "archive preflight must precede every member read/decode"
            )
    canonical = _canonical_bindings()
    expected_by_logical_id = {
        binding.logical_id: (dataset_id, binding)
        for dataset_id, binding in canonical.items()
    }
    registered_sources = getattr(loader, "_sources", None)
    if not isinstance(registered_sources, Mapping) or set(registered_sources) != set(
        expected_by_logical_id
    ):
        raise C0CoarseProviderError("registered archive source set is not canonical")
    for logical_id, (_dataset_id, binding) in expected_by_logical_id.items():
        spec = registered_sources[logical_id]
        if getattr(spec, "binding", None) != binding:
            raise C0CoarseProviderError(
                "registered archive binding format/identity is not canonical"
            )

    try:
        receipts = loader.verify_all_sources()
    except LazyMaskLoaderError as exc:
        raise C0CoarseProviderError("canonical archive verification failed") from exc
    after = loader.stats.to_dict()
    for name in ("requests", "archive_opens", "decoded_masks", "decoded_bytes"):
        if after[name] != 0:
            raise C0CoarseProviderError("archive preflight decoded or opened a member")

    if {item.logical_id for item in receipts} != set(expected_by_logical_id):
        raise C0CoarseProviderError("archive preflight source set is not canonical")
    if len(receipts) != len(expected_by_logical_id):
        raise C0CoarseProviderError("archive preflight contains duplicate sources")
    for item in receipts:
        dataset_id, binding = expected_by_logical_id[item.logical_id]
        lock = archive_locks[dataset_id]
        if (
            item.expected_sha256 != binding.sha256
            or item.observed_sha256 != binding.sha256
            or item.byte_count != lock["bytes"]
        ):
            raise C0CoarseProviderError(
                "observed archive SHA/size differs from the C0 freeze"
            )
    return tuple(receipts)


def _same_config(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return hmac.compare_digest(
        canonical_config_hash(observed), canonical_config_hash(expected)
    )


def _require_coarse_arm(arm: AblationArm) -> None:
    if not isinstance(arm, AblationArm):
        raise TypeError("arm must be AblationArm")
    if (
        arm.name is not AblationArmName.COARSE_ONLY
        or arm.evidence is not EvidenceMode.COARSE
        or arm.matcher_mode is not None
        or arm.arc_pooling is not None
        or arm.uses_local_geometry
    ):
        raise C0CoarseProviderError("C0 provider accepts only coarse-only evidence")
    if not _same_config(arm.model_config, C0_COARSE_MODEL_CONFIG):
        raise C0CoarseProviderError("C0 provider model config changed")
    if not _same_config(arm.optimizer_config, C0_COARSE_OPTIMIZER_CONFIG):
        raise C0CoarseProviderError("C0 provider optimizer config changed")


def _select_largest_four_connected(mask: np.ndarray) -> np.ndarray:
    """Select by area, then lexicographic half-open bounding box."""

    structure = ndimage.generate_binary_structure(2, 1)
    labels, count = ndimage.label(mask, structure=structure)
    if count <= 0:
        raise C0CoarseProviderError("historical mask is empty")
    areas = np.bincount(labels.reshape(-1), minlength=count + 1)
    objects = ndimage.find_objects(labels, max_label=count)
    candidates = []
    for label in range(1, count + 1):
        component_slice = objects[label - 1]
        if component_slice is None:
            continue
        bbox = (
            int(component_slice[0].start),
            int(component_slice[1].start),
            int(component_slice[0].stop),
            int(component_slice[1].stop),
        )
        candidates.append((-int(areas[label]), bbox, label))
    if not candidates:
        raise C0CoarseProviderError("largest-component selection failed")
    selected = min(candidates)[2]
    output = np.ascontiguousarray(labels == selected, dtype=np.bool_)
    if not output.any():
        raise C0CoarseProviderError("largest component is empty")
    return output


def _validate_loader_mask(mask: Any, identity: FragmentIdentity) -> np.ndarray:
    if identity.dataset_id not in _C0_DATASETS:
        raise C0CoarseProviderError("C0 mask belongs to an unapproved dataset")
    value = np.asarray(mask)
    if value.ndim != 2 or value.size == 0 or value.dtype != np.bool_:
        raise C0CoarseProviderError("loader mask must be a non-empty 2D bool array")
    if int(value.size) > _C0_MAX_INPUT_PIXELS:
        raise C0CoarseProviderError("loader mask exceeds the frozen pixel bound")
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value, dtype=np.bool_)
    # Historical brighter-foreground masks must contain both semantic values.
    # Uniform arrays cannot prove foreground polarity and are never valid C0
    # fragments, even though the generic loader can represent them.
    if not value.any():
        raise C0CoarseProviderError("historical mask is empty")
    if value.all():
        raise C0CoarseProviderError("historical mask polarity is not identifiable")
    return value


def _expected_c0_letterbox_frame(
    selected_component: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Recompute the centered resize frame from the tight component bbox."""

    component = np.asarray(selected_component)
    if component.ndim != 2 or component.dtype != np.bool_ or component.size == 0:
        raise C0CoarseProviderError("selected component contract changed")
    foreground = np.argwhere(component)
    if foreground.size == 0:
        raise C0CoarseProviderError("selected component is empty")
    row_min, column_min = foreground.min(axis=0)
    row_max, column_max = foreground.max(axis=0) + 1
    crop_rows = int(row_max - row_min)
    crop_columns = int(column_max - column_min)
    content_rows = 112
    content_columns = 112
    scale = min(content_rows / crop_rows, content_columns / crop_columns)
    resized_rows = max(1, min(content_rows, int(round(crop_rows * scale))))
    resized_columns = max(
        1,
        min(content_columns, int(round(crop_columns * scale))),
    )
    if max(resized_rows, resized_columns) != 112:
        raise C0CoarseProviderError("C0 letterbox frame extent changed")
    top = (128 - resized_rows) // 2
    left = (128 - resized_columns) // 2
    return top, top + resized_rows, left, left + resized_columns


def _validate_coarse_tensor(value: Tensor, selected_component: np.ndarray) -> Tensor:
    if not isinstance(value, Tensor):
        raise C0CoarseProviderError("coarse preprocessing did not return a tensor")
    if (
        value.dtype != torch.float32
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or tuple(value.shape) != (1, 128, 128)
    ):
        raise C0CoarseProviderError("coarse tensor shape/dtype/device changed")
    output = value.detach().contiguous()
    if (
        not torch.isfinite(output).all().item()
        or (output < 0.0).any().item()
        or (output > 1.0).any().item()
    ):
        raise C0CoarseProviderError("coarse tensor values violate [0,1]")

    top, bottom, left, right = _expected_c0_letterbox_frame(selected_component)
    plane = output[0]
    outside_nonzero = any(
        torch.count_nonzero(region).item() != 0
        for region in (
            plane[:top, :],
            plane[bottom:, :],
            plane[top:bottom, :left],
            plane[top:bottom, right:],
        )
    )
    if outside_nonzero:
        raise C0CoarseProviderError(
            "coarse tensor has values outside its centered letterbox frame"
        )
    if not (plane[top:bottom, left:right] > 0.0).any().item():
        raise C0CoarseProviderError("coarse tensor lost all foreground")
    return output


class C0CoarseProvider:
    """Prepare only physical-identity-attested C0 coarse batches."""

    def __init__(
        self,
        *,
        identity_index: HistoricalIdentityIndex,
        mask_loader: LazyMaskArchiveLoader,
        freeze_receipt: Mapping[str, Any],
        expected_freeze_content_sha256: str,
        expected_identity_index_content_sha256: str,
        config: C0CoarseProviderConfig = C0CoarseProviderConfig(),
    ) -> None:
        if type(identity_index) is not HistoricalIdentityIndex:  # noqa: E721
            raise TypeError("identity_index must be HistoricalIdentityIndex")
        if not isinstance(config, C0CoarseProviderConfig):
            raise TypeError("config must be C0CoarseProviderConfig")
        self.mask_loader = mask_loader
        self.config = config
        self.expected_freeze_content_sha256 = _require_sha256(
            expected_freeze_content_sha256, "expected freeze receipt digest"
        )
        self.archive_locks = _validate_freeze_receipt(
            freeze_receipt, self.expected_freeze_content_sha256
        )
        _validate_identity_index_locks(identity_index, freeze_receipt)
        self.expected_identity_index_content_sha256 = _validate_identity_index_content(
            identity_index, expected_identity_index_content_sha256
        )
        # The index owns a deep, primitive, read-only mapping.  Keep that exact
        # externally anchored object rather than allocating a second full copy.
        self.identity_index = identity_index
        self.archive_verification_receipts = _verify_archive_preflight(
            mask_loader, self.archive_locks
        )
        self._geometry_config, self.preprocessing = _exact_preprocessing()
        if self.config.max_batch_size > self._geometry_config.max_batch_size:
            raise C0CoarseProviderError("provider batch bound exceeds frozen C0 bound")
        self.contract = BatchProviderContract(
            coarse_preprocess_mode=self.preprocessing.mode,
            coarse_preprocessing_sha256=self.preprocessing.preprocessing_sha256,
            geometry_config_sha256=self.preprocessing.geometry_config_sha256,
            cache_interface="bounded_host_coarse_tensor_lru_no_geometry_cache",
            provider_version=C0_COARSE_PROVIDER_VERSION,
            coarse_only_geometry_free=True,
        )

        self._memo: "OrderedDict[_FragmentMemoKey, Tensor]" = OrderedDict()
        self._memo_bytes = 0
        self._batch_count = 0
        self._record_count = 0
        self._fragment_request_count = 0
        self._memo_hit_count = 0
        self._memo_miss_count = 0
        self._mask_loader_call_count = 0
        self._archive_decode_count = 0
        self._loader_cache_hit_count = 0
        self._coarse_preprocess_count = 0
        self._memo_eviction_count = 0
        self._provider_opened_archives = set()
        self._last_loader_stats = dict(mask_loader.stats.to_dict())
        self._lock = threading.Lock()
        self._failed = False

    def _assert_loader_exclusive(self) -> None:
        if self.mask_loader.stats.to_dict() != self._last_loader_stats:
            self._failed = True
            raise C0CoarseProviderError("mask loader changed outside the C0 provider")

    @staticmethod
    def _loader_delta(
        before: Mapping[str, int], after: Mapping[str, int]
    ) -> Dict[str, int]:
        if set(before) != set(after):
            raise C0CoarseProviderError("mask loader statistics schema changed")
        delta = {}
        for name in before:
            previous = before[name]
            current = after[name]
            if type(previous) is not int or type(current) is not int:
                raise C0CoarseProviderError("mask loader statistics are invalid")
            difference = current - previous
            if difference < 0:
                raise C0CoarseProviderError("mask loader statistics moved backwards")
            delta[name] = difference
        return delta

    @staticmethod
    def _zero_loader_delta(stats: Mapping[str, int]) -> Dict[str, int]:
        return {name: 0 for name in stats}

    def _validate_owned_loader_call(
        self,
        *,
        reference: Any,
        mask: np.ndarray,
        delta: Mapping[str, int],
    ) -> None:
        required = {
            "requests",
            "cache_hits",
            "cache_misses",
            "archive_opens",
            "decoded_masks",
            "decoded_bytes",
            "decoded_pixels",
            "evictions",
            "archive_verifications",
            "archive_verified_bytes",
            "archive_hash_failures",
        }
        if set(delta) != required:
            raise C0CoarseProviderError("mask loader statistics schema changed")
        if delta["requests"] != 1:
            raise C0CoarseProviderError("external mask loader request raced C0")
        if delta["cache_hits"] + delta["cache_misses"] != 1:
            raise C0CoarseProviderError("mask loader cache accounting changed")
        if delta["decoded_masks"] != delta["cache_misses"]:
            raise C0CoarseProviderError("mask loader decode accounting changed")
        if delta["cache_misses"] == 0:
            if delta["decoded_bytes"] != 0 or delta["decoded_pixels"] != 0:
                raise C0CoarseProviderError("mask loader cache hit decoded content")
        else:
            if delta["decoded_bytes"] <= 0 or delta["decoded_pixels"] != int(mask.size):
                raise C0CoarseProviderError(
                    "mask loader decoded size accounting changed"
                )
        logical_id = reference.binding.logical_id
        expected_open = int(logical_id not in self._provider_opened_archives)
        if delta["archive_opens"] != expected_open:
            raise C0CoarseProviderError("mask loader archive-open accounting changed")
        if (
            delta["archive_verifications"] != 0
            or delta["archive_verified_bytes"] != 0
            or delta["archive_hash_failures"] != 0
        ):
            raise C0CoarseProviderError(
                "archive verification changed during C0 data use"
            )
        # A single owned cache insertion may evict zero or more entries under
        # the loader's independently bounded pixel quota.  Exact batch-delta
        # reconciliation below still ensures those evictions cannot absorb an
        # external request.
        if delta["evictions"] < 0:  # pragma: no cover - guarded by _loader_delta
            raise C0CoarseProviderError("mask loader eviction accounting changed")
        self._provider_opened_archives.add(logical_id)

    def _verify_input(
        self, value: Union[TrainingPairRecord, VerifiedPair], *, phase: str
    ) -> VerifiedPair:
        if isinstance(value, VerifiedPair):
            supplied = value
            record = supplied.record
        elif isinstance(value, TrainingPairRecord):
            supplied = None
            record = value
        else:
            raise TypeError("C0 records must be TrainingPairRecord or VerifiedPair")
        try:
            verified = self.identity_index.verify_record(record)
        except HistoricalIdentityError as exc:
            raise C0CoarseProviderError("historical physical identity failed") from exc
        if supplied is not None and (
            supplied.fragment_a != verified.fragment_a
            or supplied.fragment_b != verified.fragment_b
            or supplied.record != verified.record
        ):
            raise C0CoarseProviderError("supplied VerifiedPair is not canonical")
        expected_split = _C0_PHASE_TO_SPLIT[phase]
        if verified.record.split != expected_split:
            raise C0CoarseProviderError("C0 record escaped its requested phase")
        if verified.record.dataset_id not in _C0_DATASETS:
            raise C0CoarseProviderError("C0 record dataset is not historical")
        if verified.record.provenance.get("real_dunhuang_sealed_test") is not False:
            raise C0CoarseProviderError(
                "sealed/unknown real-test provenance is forbidden"
            )
        canonical = _canonical_bindings()[verified.record.dataset_id]
        for reference, identity in (
            (verified.record.fragment_a, verified.fragment_a),
            (verified.record.fragment_b, verified.fragment_b),
        ):
            if (
                reference.binding != canonical
                or identity.binding != canonical
                or reference.threshold_rule != "binary_brighter_value"
                or reference.content_sha256 != identity.content_sha256
                or reference.archive_member != identity.archive_member
                or reference.dataset_id != identity.dataset_id
            ):
                raise C0CoarseProviderError(
                    "historical mask identity/polarity contract changed"
                )
            _require_sha256(identity.content_sha256, "fragment content digest")
        return verified

    def attest_record_order(
        self,
        records: Sequence[Union[TrainingPairRecord, VerifiedPair]],
        *,
        phase: str,
    ) -> C0RecordOrderAttestation:
        """Verify first, then freeze an enriched physical/content-bound order."""

        if phase not in _C0_PHASE_TO_SPLIT:
            raise C0CoarseProviderError("phase must be train or validation")
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("records must be a finite sequence")
        if not records:
            raise C0CoarseProviderError("C0 batch cannot be empty")
        if len(records) > self.config.max_batch_size:
            raise C0CoarseProviderError("C0 batch exceeds the frozen batch bound")
        verified = tuple(self._verify_input(item, phase=phase) for item in records)
        enriched_records = tuple(item.record for item in verified)
        digest = record_sequence_fingerprint(enriched_records)
        return C0RecordOrderAttestation(
            phase=phase,
            count=len(verified),
            record_sequence_sha256=digest,
            verified_pairs=verified,
        )

    @staticmethod
    def _memo_key(
        identity: FragmentIdentity, threshold_rule: str, preprocessing_sha256: str
    ) -> _FragmentMemoKey:
        return _FragmentMemoKey(
            dataset_id=identity.dataset_id,
            archive_sha256=identity.binding.sha256,
            archive_member=identity.archive_member,
            content_sha256=identity.content_sha256,
            threshold_rule=threshold_rule,
            preprocessing_sha256=preprocessing_sha256,
        )

    def _memo_put(self, key: _FragmentMemoKey, value: Tensor) -> None:
        if self.config.max_cached_fragments == 0:
            return
        byte_count = int(value.numel() * value.element_size())
        if byte_count > self.config.max_cache_bytes:
            return
        previous = self._memo.pop(key, None)
        if previous is not None:
            self._memo_bytes -= int(previous.numel() * previous.element_size())
        self._memo[key] = value
        self._memo.move_to_end(key)
        self._memo_bytes += byte_count
        while (
            len(self._memo) > self.config.max_cached_fragments
            or self._memo_bytes > self.config.max_cache_bytes
        ):
            _, removed = self._memo.popitem(last=False)
            self._memo_bytes -= int(removed.numel() * removed.element_size())
            self._memo_eviction_count += 1

    def _coarse_fragment(
        self, identity: FragmentIdentity, reference: Any
    ) -> Tuple[Tensor, bool, Mapping[str, int]]:
        self._fragment_request_count += 1
        key = self._memo_key(
            identity,
            reference.threshold_rule,
            self.preprocessing.preprocessing_sha256,
        )
        cached = self._memo.get(key)
        if cached is not None:
            self._memo.move_to_end(key)
            self._memo_hit_count += 1
            return cached, True, self._zero_loader_delta(self._last_loader_stats)

        self._memo_miss_count += 1
        before = self.mask_loader.stats.to_dict()
        try:
            mask = self.mask_loader(reference)
        except (LazyMaskLoaderError, TypeError, ValueError) as exc:
            raise C0CoarseProviderError("historical mask load failed") from exc
        after = self.mask_loader.stats.to_dict()
        delta = self._loader_delta(before, after)
        self._validate_owned_loader_call(
            reference=reference,
            mask=np.asarray(mask),
            delta=delta,
        )
        decoded = delta["decoded_masks"]
        loader_hits = delta["cache_hits"]
        self._mask_loader_call_count += 1
        self._archive_decode_count += decoded
        self._loader_cache_hit_count += loader_hits

        canonical_mask = _validate_loader_mask(mask, identity)
        selected = _select_largest_four_connected(canonical_mask)
        try:
            coarse = preprocess_coarse_mask(selected, self._geometry_config)
        except (GeometryBatchError, TypeError, ValueError, RuntimeError) as exc:
            raise C0CoarseProviderError("coarse preprocessing failed") from exc
        coarse = _validate_coarse_tensor(coarse, selected)
        self._coarse_preprocess_count += 1
        self._memo_put(key, coarse)
        return coarse, False, delta

    def prepare(
        self,
        records: Sequence[Union[TrainingPairRecord, VerifiedPair]],
        *,
        arm: AblationArm,
        phase: str,
    ) -> PreparedAblationBatch:
        """Prepare one bounded batch without exposing any local evidence path."""

        if not self._lock.acquire(blocking=False):
            raise C0CoarseProviderError("concurrent C0 provider access is forbidden")
        try:
            if self._failed:
                raise C0CoarseProviderError(
                    "C0 provider is tainted by an earlier failure"
                )
            self._assert_loader_exclusive()
            _require_coarse_arm(arm)
            attestation = self.attest_record_order(records, phase=phase)
            batch_loader_start = dict(self.mask_loader.stats.to_dict())
            owned_loader_delta = self._zero_loader_delta(batch_loader_start)

            batch_mask_loads = 0
            batch_preprocesses = 0
            coarse_a = []
            coarse_b = []
            try:
                for pair in attestation.verified_pairs:
                    first, first_hit, first_delta = self._coarse_fragment(
                        pair.fragment_a, pair.record.fragment_a
                    )
                    second, second_hit, second_delta = self._coarse_fragment(
                        pair.fragment_b, pair.record.fragment_b
                    )
                    for name in owned_loader_delta:
                        owned_loader_delta[name] += (
                            first_delta[name] + second_delta[name]
                        )
                    coarse_a.append(first)
                    coarse_b.append(second)
                    batch_mask_loads += int(not first_hit) + int(not second_hit)
                    batch_preprocesses += int(not first_hit) + int(not second_hit)
            except BaseException:
                self._last_loader_stats = dict(self.mask_loader.stats.to_dict())
                self._failed = True
                raise

            tensor_a = torch.stack(coarse_a, dim=0).contiguous()
            tensor_b = torch.stack(coarse_b, dim=0).contiguous()
            labels = torch.tensor(
                [float(item.record.label) for item in attestation.verified_pairs],
                dtype=torch.float32,
            )
            payload = C0CoarsePayload.from_tensors(tensor_a, tensor_b, labels)
            observed_payload_sha = c0_coarse_payload_content_sha256(
                payload.coarse_a, payload.coarse_b, payload.labels
            )
            if not hmac.compare_digest(
                observed_payload_sha, payload.payload_content_sha256
            ):
                raise C0CoarseProviderError("C0 payload digest changed during creation")

            counts = {
                "mask_load_count": batch_mask_loads,
                "coarse_preprocess_count": batch_preprocesses,
                **dict(_ZERO_LOCAL_PROCESSING),
            }
            prepared = PreparedAblationBatch(
                payload=payload,
                sample_count=attestation.count,
                record_sequence_sha256=attestation.record_sequence_sha256,
                prepared_input_sha256=payload.payload_content_sha256,
                local_candidate_sha256=None,
                coarse_preprocessing_sha256=(self.preprocessing.preprocessing_sha256),
                geometry_config_sha256=None,
                processing_counts=counts,
            )
            if not hmac.compare_digest(
                prepared.prepared_input_sha256, payload.payload_content_sha256
            ):
                raise C0CoarseProviderError("prepared/payload digest mismatch")

            batch_loader_end = dict(self.mask_loader.stats.to_dict())
            try:
                observed_loader_delta = self._loader_delta(
                    batch_loader_start, batch_loader_end
                )
            except C0CoarseProviderError:
                self._last_loader_stats = batch_loader_end
                self._failed = True
                raise
            if observed_loader_delta != owned_loader_delta:
                self._last_loader_stats = batch_loader_end
                self._failed = True
                raise C0CoarseProviderError(
                    "external mask loader activity raced the C0 batch"
                )

            self._batch_count += 1
            self._record_count += attestation.count
            # Bind the exact already-reconciled snapshot.  If an external call
            # lands after this snapshot, the next prepare/receipt detects it.
            self._last_loader_stats = batch_loader_end
            return prepared
        except BaseException:
            # Metadata/config failures before member access are retry-safe.  A
            # member/preprocessing failure marks the provider in the inner
            # block above so partially consumed archive state is never reused.
            raise
        finally:
            self._lock.release()

    def _stats_snapshot_unlocked(self) -> C0CoarseProviderStats:
        return C0CoarseProviderStats(
            batch_count=self._batch_count,
            record_count=self._record_count,
            fragment_request_count=self._fragment_request_count,
            memo_hit_count=self._memo_hit_count,
            memo_miss_count=self._memo_miss_count,
            mask_loader_call_count=self._mask_loader_call_count,
            archive_decode_count=self._archive_decode_count,
            loader_cache_hit_count=self._loader_cache_hit_count,
            coarse_preprocess_count=self._coarse_preprocess_count,
            memo_eviction_count=self._memo_eviction_count,
            memo_entry_count=len(self._memo),
            memo_byte_count=self._memo_bytes,
        )

    def stats_snapshot(self) -> C0CoarseProviderStats:
        if not self._lock.acquire(blocking=False):
            raise C0CoarseProviderError(
                "cannot snapshot C0 provider statistics during a batch"
            )
        try:
            self._assert_loader_exclusive()
            return self._stats_snapshot_unlocked()
        finally:
            self._lock.release()

    def receipt(self) -> Mapping[str, Any]:
        """Return portable provider evidence; runtime paths/row IDs are absent."""

        if not self._lock.acquire(blocking=False):
            raise C0CoarseProviderError("cannot emit a C0 receipt during a batch")
        try:
            if self._failed:
                raise C0CoarseProviderError(
                    "tainted C0 provider cannot emit a passing receipt"
                )
            self._assert_loader_exclusive()
            _validate_identity_index_content(
                self.identity_index, self.expected_identity_index_content_sha256
            )
            formats = {
                binding.logical_id: binding.archive_format
                for binding in _canonical_bindings().values()
            }
            archive_verification = []
            for item in self.archive_verification_receipts:
                value = item.to_dict()
                value["archive_format"] = formats[item.logical_id]
                archive_verification.append(value)
            return {
                "schema_version": C0_COARSE_PROVIDER_RECEIPT_VERSION,
                "provider_version": C0_COARSE_PROVIDER_VERSION,
                "status": "archive_preflight_complete",
                "freeze_content_sha256": self.expected_freeze_content_sha256,
                "identity_index_content_sha256": (
                    self.expected_identity_index_content_sha256
                ),
                "archive_verification": archive_verification,
                "preprocessing": self.preprocessing.to_dict(),
                "memo_bounds": {
                    "max_batch_size": self.config.max_batch_size,
                    "max_cached_fragments": self.config.max_cached_fragments,
                    "max_cache_bytes": self.config.max_cache_bytes,
                },
                "counters": self._stats_snapshot_unlocked().to_dict(),
                "geometry_cache_local_sinkhorn_calls": 0,
                "historical_test_access": dict(HISTORICAL_TEST_ACCESS_EVIDENCE),
                "sealed_real_read": False,
            }
        finally:
            self._lock.release()


__all__ = [
    "C0_COARSE_PROVIDER_RECEIPT_VERSION",
    "C0_COARSE_PROVIDER_VERSION",
    "C0_COARSE_TENSOR_BYTES",
    "C0_PRODUCTION_MAX_CACHE_BYTES",
    "C0_PRODUCTION_MAX_CACHED_FRAGMENTS",
    "C0CoarseProvider",
    "C0CoarseProviderConfig",
    "C0CoarseProviderError",
    "C0CoarseProviderStats",
    "C0RecordOrderAttestation",
]

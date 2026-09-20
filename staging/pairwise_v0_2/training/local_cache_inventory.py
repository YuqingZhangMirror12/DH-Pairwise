"""Label-blind inventory and freeze boundary for local geometry qualification.

The local matcher must compare assignment rules on exactly the same contour
windows.  This module materializes each unique fragment at most once, combines
the cached role-neutral artifacts into all four upright directions, and emits
an aggregate portable receipt.  Pair labels and historical direction labels
are deliberately never read while building the inventory.

The generic :mod:`geometry_cache` receipt remains the authority for individual
artifact bytes.  The inventory adds a population/order commitment and bounded
geometry/candidate census without serializing archive paths, member names,
fragment IDs, pair IDs, or labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from staging.pairwise_v0_2.geometry import (
    DEFAULT_DIRECTION_ORDER,
    GEOMETRY_VERSION,
    CandidateBuilderConfig,
    GeometryStatus,
    combine_fragment_results,
    geometry_config_fingerprint,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.fragment_geometry_cache import (
    FragmentGeometryCacheLookup,
    load_or_build_fragment_geometry,
)
from staging.pairwise_v0_2.training.geometry_cache import (
    CACHE_RECIPE_VERSION,
    CACHE_SCHEMA_VERSION,
    FrozenReceiptTrust,
    GeometryArtifactCache,
    GeometryCacheLimits,
)


LOCAL_CACHE_INVENTORY_SCHEMA_VERSION = "dunhuang-local-cache-inventory/0.1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = frozenset(
    {
        "archive_member",
        "fragment_id",
        "group_id",
        "label",
        "member_path",
        "pair_id",
        "path",
        "sample_id",
    }
)


class LocalCacheInventoryError(RuntimeError):
    """The local qualification inventory is unsafe or inconsistent."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _portable(value: Any, location: str = "receipt") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and (
            value.startswith("/")
            or value.startswith("file://")
            or re.match(r"^[A-Za-z]:[\\/]", value)
        ):
            raise LocalCacheInventoryError(
                "{} contains a machine-local path".format(location)
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LocalCacheInventoryError("{} contains NaN/Inf".format(location))
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LocalCacheInventoryError(
                    "{} contains a non-string key".format(location)
                )
            if key.casefold() in _FORBIDDEN_KEYS:
                raise LocalCacheInventoryError(
                    "{} exposes a row-level identity".format(location)
                )
            result[key] = _portable(item, "{}.{}".format(location, key))
        return result
    if isinstance(value, (list, tuple)):
        return [
            _portable(item, "{}[{}]".format(location, index))
            for index, item in enumerate(value)
        ]
    raise LocalCacheInventoryError(
        "{} is not JSON-portable: {}".format(location, type(value).__name__)
    )


def _positive_edges(value: Sequence[int], name: str) -> Tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("{} must be a finite sequence".format(name))
    edges = tuple(value)
    if (
        not edges
        or any(isinstance(item, bool) or not isinstance(item, int) for item in edges)
        or any(item <= 0 for item in edges)
        or tuple(sorted(set(edges))) != edges
    ):
        raise ValueError("{} must be unique increasing positive integers".format(name))
    return edges


@dataclass(frozen=True)
class LocalCacheInventoryConfig:
    geometry: CandidateBuilderConfig = field(default_factory=CandidateBuilderConfig)
    sequence_length_bucket_edges: Tuple[int, ...] = (16, 32, 64, 128, 256, 512)
    candidate_count_bucket_edges: Tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256, 512)
    max_records: int = 20_000
    max_unique_references: int = 40_000
    max_mask_pixels: int = 100_000_000
    require_all_fragments_ok: bool = True
    require_all_pairs_ok: bool = True
    require_train_val_fragment_disjoint: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, CandidateBuilderConfig):
            raise TypeError("geometry must be CandidateBuilderConfig")
        object.__setattr__(
            self,
            "sequence_length_bucket_edges",
            _positive_edges(
                self.sequence_length_bucket_edges, "sequence_length_bucket_edges"
            ),
        )
        object.__setattr__(
            self,
            "candidate_count_bucket_edges",
            _positive_edges(
                self.candidate_count_bucket_edges, "candidate_count_bucket_edges"
            ),
        )
        for name in ("max_records", "max_unique_references", "max_mask_pixels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        for name in (
            "require_all_fragments_ok",
            "require_all_pairs_ok",
            "require_train_val_fragment_disjoint",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("{} must be bool".format(name))

    def portable_dict(self) -> Dict[str, Any]:
        return {
            "geometry": asdict(self.geometry),
            "sequence_length_bucket_edges": list(self.sequence_length_bucket_edges),
            "candidate_count_bucket_edges": list(self.candidate_count_bucket_edges),
            "max_records": self.max_records,
            "max_unique_references": self.max_unique_references,
            "max_mask_pixels": self.max_mask_pixels,
            "require_all_fragments_ok": self.require_all_fragments_ok,
            "require_all_pairs_ok": self.require_all_pairs_ok,
            "require_train_val_fragment_disjoint": (
                self.require_train_val_fragment_disjoint
            ),
        }


@dataclass(frozen=True)
class LocalCacheInventoryResult:
    receipt: Mapping[str, Any]
    cache_receipt: Mapping[str, Any]
    semantic_commitment_sha256: str
    cache_receipt_canonical_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "semantic_commitment_sha256",
            "cache_receipt_canonical_sha256",
        ):
            if not _SHA256_RE.fullmatch(getattr(self, name)):
                raise ValueError("{} must be lowercase SHA-256".format(name))

    def portable_json(self) -> str:
        return (
            json.dumps(
                _portable(dict(self.receipt)),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )


def _reference_key(reference: MaskMemberRef) -> Tuple[str, ...]:
    """Internal-only physical lookup key; never serialized."""

    return (
        reference.binding.logical_id,
        reference.binding.sha256,
        reference.archive_member,
        reference.threshold_rule,
    )


def _increment(target: Dict[str, int], key: str, amount: int = 1) -> None:
    target[key] = target.get(key, 0) + amount


def _bucket(value: int, edges: Sequence[int]) -> str:
    for edge in edges:
        if value <= edge:
            return "le_{}".format(edge)
    return "gt_{}".format(edges[-1])


def _status_counts(values: Sequence[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for value in values:
        _increment(result, value)
    return dict(sorted(result.items()))


def _summary(values: Sequence[int], edges: Sequence[int]) -> Dict[str, Any]:
    buckets: Dict[str, int] = {}
    for value in values:
        _increment(buckets, _bucket(value, edges))
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "sum": sum(values),
        "buckets": dict(sorted(buckets.items())),
    }


def _cache_inventory_commitment(cache_receipt: Mapping[str, Any]) -> str:
    """Bind artifact semantics while allowing writable→frozen mode transition."""

    required = (
        "schema_version",
        "cache_schema_version",
        "typed_payload_version",
        "artifact_count",
        "artifacts",
        "path_policy",
    )
    try:
        payload = {name: cache_receipt[name] for name in required}
    except KeyError as exc:  # pragma: no cover - generic cache owns this schema
        raise LocalCacheInventoryError("cache receipt is incomplete") from exc
    return _sha256(_canonical_json(payload))


def _record_order_commitments(
    records: Sequence[TrainingPairRecord],
    identity_by_reference: Mapping[Tuple[str, ...], str],
) -> Tuple[str, str]:
    rows = []
    for record in records:
        rows.append(
            {
                "dataset": record.dataset_id,
                "split": record.split,
                "fragment_a_cache_key": identity_by_reference[
                    _reference_key(record.fragment_a)
                ],
                "fragment_b_cache_key": identity_by_reference[
                    _reference_key(record.fragment_b)
                ],
            }
        )
    order_hash = _sha256(_canonical_json(rows))
    set_hash = _sha256(
        _canonical_json(sorted(_sha256(_canonical_json(row)) for row in rows))
    )
    return order_hash, set_hash


def build_local_cache_inventory(
    records: Sequence[TrainingPairRecord],
    mask_loader: Callable[[MaskMemberRef], np.ndarray],
    cache: GeometryArtifactCache,
    config: Optional[LocalCacheInventoryConfig] = None,
) -> LocalCacheInventoryResult:
    """Build an exact fragment cache and aggregate all-direction census.

    Supervision fields are intentionally never read.  Candidate combination
    always receives ``direction_b_wrt_a=None`` and therefore enumerates the
    same four upright directions at training and inference time.
    """

    settings = config or LocalCacheInventoryConfig()
    if not isinstance(settings, LocalCacheInventoryConfig):
        raise TypeError("config must be LocalCacheInventoryConfig")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a finite sequence")
    population = tuple(records)
    if not population:
        raise LocalCacheInventoryError("records cannot be empty")
    if len(population) > settings.max_records:
        raise LocalCacheInventoryError("record population exceeds configured bound")
    if not callable(mask_loader):
        raise TypeError("mask_loader must be callable")
    if not isinstance(cache, GeometryArtifactCache):
        raise TypeError("cache must be GeometryArtifactCache")

    references: Dict[Tuple[str, ...], MaskMemberRef] = {}
    reference_dataset_splits: Dict[Tuple[str, ...], set] = {}
    population_counts: Dict[str, int] = {}
    endpoint_counts: Dict[str, int] = {}
    for record in population:
        if not isinstance(record, TrainingPairRecord):
            raise TypeError("every record must be TrainingPairRecord")
        if record.split not in {"train", "val"}:
            raise LocalCacheInventoryError("only train/val records are permitted")
        stratum = "{}:{}".format(record.split, record.dataset_id)
        _increment(population_counts, stratum)
        _increment(endpoint_counts, stratum, 2)
        for reference in (record.fragment_a, record.fragment_b):
            reference_key = _reference_key(reference)
            references.setdefault(reference_key, reference)
            reference_dataset_splits.setdefault(reference_key, set()).add(stratum)
    if len(references) > settings.max_unique_references:
        raise LocalCacheInventoryError("unique reference population exceeds bound")

    lookup_by_identity: Dict[str, FragmentGeometryCacheLookup] = {}
    identity_by_reference: Dict[Tuple[str, ...], str] = {}
    fragment_dataset_splits: Dict[str, set] = {}
    mask_load_count = 0
    canonical_duplicate_reference_count = 0
    cache_hit_count = 0
    cache_miss_count = 0
    for key, reference in references.items():
        value = np.asarray(mask_loader(reference))
        mask_load_count += 1
        if value.ndim != 2 or value.size == 0 or value.dtype != np.bool_:
            raise LocalCacheInventoryError(
                "mask loader must return a non-empty 2D bool array"
            )
        if int(value.size) > settings.max_mask_pixels:
            raise LocalCacheInventoryError("decoded mask exceeds configured bound")
        lookup = load_or_build_fragment_geometry(
            np.ascontiguousarray(value, dtype=np.bool_),
            reference.threshold_rule,
            settings.geometry,
            cache,
        )
        identity_key = lookup.identity.key
        identity_by_reference[key] = identity_key
        if identity_key in lookup_by_identity:
            canonical_duplicate_reference_count += 1
            if (
                lookup_by_identity[identity_key].logical_payload_sha256
                != lookup.logical_payload_sha256
            ):
                raise LocalCacheInventoryError(
                    "one canonical fragment resolved to different payloads"
                )
        else:
            lookup_by_identity[identity_key] = lookup
            if lookup.cache_hit:
                cache_hit_count += 1
            else:
                cache_miss_count += 1
        fragment_dataset_splits.setdefault(identity_key, set()).update(
            reference_dataset_splits[key]
        )

    train_val_overlap_count = sum(
        1
        for strata in fragment_dataset_splits.values()
        if any(value.startswith("train:") for value in strata)
        and any(value.startswith("val:") for value in strata)
    )
    if settings.require_train_val_fragment_disjoint and train_val_overlap_count:
        raise LocalCacheInventoryError(
            "canonical fragment content overlaps train and validation"
        )

    identities = tuple(
        lookup_by_identity[key].identity for key in sorted(lookup_by_identity)
    )
    cache_receipt = cache.portable_receipt(identities)
    cache_receipt_hash = _sha256(_canonical_json(cache_receipt))
    cache_inventory_commitment = _cache_inventory_commitment(cache_receipt)

    fragment_statuses = []
    fragment_failure_reasons = []
    fragment_sequence_counts = []
    fragment_patch_counts = []
    sequence_lengths = []
    sequence_side_counts: Dict[str, int] = {}
    fragment_counts_by_population: Dict[str, int] = {}
    for identity_key in sorted(lookup_by_identity):
        lookup = lookup_by_identity[identity_key]
        result = lookup.result
        fragment_statuses.append(result.status.value)
        for stratum in fragment_dataset_splits[identity_key]:
            _increment(fragment_counts_by_population, stratum)
        if result.status is not GeometryStatus.OK or result.artifact is None:
            fragment_failure_reasons.append(result.failure_reason or "unknown")
            continue
        lengths = [sequence.length for sequence in result.artifact.sequences]
        fragment_sequence_counts.append(len(lengths))
        fragment_patch_counts.append(sum(lengths))
        sequence_lengths.extend(lengths)
        for sequence in result.artifact.sequences:
            _increment(sequence_side_counts, sequence.side.value)

    if settings.require_all_fragments_ok and any(
        value != GeometryStatus.OK.value for value in fragment_statuses
    ):
        raise LocalCacheInventoryError(
            "fragment qualification requires every cached fragment to succeed"
        )

    pair_statuses = []
    pair_failure_reasons = []
    candidate_counts = []
    emitted_direction_counts = []
    all_direction_pair_count = 0
    pair_counts_by_population: Dict[str, Dict[str, int]] = {}
    for record in population:
        result_a = lookup_by_identity[
            identity_by_reference[_reference_key(record.fragment_a)]
        ].result
        result_b = lookup_by_identity[
            identity_by_reference[_reference_key(record.fragment_b)]
        ].result
        pair_result = combine_fragment_results(
            result_a,
            result_b,
            direction_b_wrt_a=None,
            config=settings.geometry,
        )
        pair_statuses.append(pair_result.status.value)
        stratum = "{}:{}".format(record.split, record.dataset_id)
        stratum_counts = pair_counts_by_population.setdefault(stratum, {})
        _increment(stratum_counts, pair_result.status.value)
        if pair_result.status is not GeometryStatus.OK:
            pair_failure_reasons.append(pair_result.failure_reason or "unknown")
            continue
        candidate_counts.append(len(pair_result.candidates))
        direction_count = len(pair_result.direction_groups)
        emitted_direction_counts.append(direction_count)
        if direction_count == len(DEFAULT_DIRECTION_ORDER):
            all_direction_pair_count += 1

    if settings.require_all_pairs_ok and any(
        value != GeometryStatus.OK.value for value in pair_statuses
    ):
        raise LocalCacheInventoryError(
            "pair qualification requires every record to emit candidates"
        )

    order_hash, set_hash = _record_order_commitments(population, identity_by_reference)
    semantic = {
        "schema_version": LOCAL_CACHE_INVENTORY_SCHEMA_VERSION,
        "inventory_config_sha256": _sha256(_canonical_json(settings.portable_dict())),
        "geometry_version": GEOMETRY_VERSION,
        "geometry_config_sha256": geometry_config_fingerprint(settings.geometry),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_recipe_version": CACHE_RECIPE_VERSION,
        "cache_inventory_commitment_sha256": cache_inventory_commitment,
        "cache_artifact_count": cache_receipt["artifact_count"],
        "record_geometry_order_sha256": order_hash,
        "record_geometry_set_sha256": set_hash,
        "population_record_counts": dict(sorted(population_counts.items())),
        "population_endpoint_counts": dict(sorted(endpoint_counts.items())),
        "fragment_counts_by_population": dict(
            sorted(fragment_counts_by_population.items())
        ),
        "train_val_canonical_fragment_overlap_count": train_val_overlap_count,
        "fragment_status_counts": _status_counts(fragment_statuses),
        "fragment_failure_reason_counts": _status_counts(fragment_failure_reasons),
        "fragment_sequence_count_summary": _summary(
            fragment_sequence_counts, settings.candidate_count_bucket_edges
        ),
        "fragment_patch_count_summary": _summary(
            fragment_patch_counts, settings.sequence_length_bucket_edges
        ),
        "sequence_length_summary": _summary(
            sequence_lengths, settings.sequence_length_bucket_edges
        ),
        "sequence_side_counts": dict(sorted(sequence_side_counts.items())),
        "pair_status_counts": _status_counts(pair_statuses),
        "pair_failure_reason_counts": _status_counts(pair_failure_reasons),
        "pair_candidate_count_summary": _summary(
            candidate_counts, settings.candidate_count_bucket_edges
        ),
        "pair_emitted_direction_count_summary": _summary(
            emitted_direction_counts, (1, 2, 3, 4)
        ),
        "all_four_direction_pair_count": all_direction_pair_count,
        "pair_counts_by_population": {
            key: dict(sorted(value.items()))
            for key, value in sorted(pair_counts_by_population.items())
        },
    }
    semantic_commitment = _sha256(_canonical_json(semantic))
    receipt = {
        "schema_version": LOCAL_CACHE_INVENTORY_SCHEMA_VERSION,
        "status": "qualified" if not pair_failure_reasons else "failed_closed",
        "semantic_commitment_sha256": semantic_commitment,
        "semantic_inventory": semantic,
        "config": settings.portable_dict(),
        "operational_counts": {
            "record_count": len(population),
            "endpoint_occurrence_count": 2 * len(population),
            "unique_physical_reference_count": len(references),
            "unique_canonical_fragment_count": len(lookup_by_identity),
            "canonical_duplicate_reference_count": canonical_duplicate_reference_count,
            "mask_load_count": mask_load_count,
            "cache_hit_count": cache_hit_count,
            "cache_miss_count": cache_miss_count,
        },
        "cache_receipt_canonical_sha256": cache_receipt_hash,
        "generation_contract": {
            "input_modality": "canonical_bool_mask_only",
            "candidate_generation_direction_argument": None,
            "candidate_generation_directions": [
                direction.value for direction in DEFAULT_DIRECTION_ORDER
            ],
            "supervision_fields_read": [],
            "cache_identity_scope": "single_fragment_role_neutral",
            "path_policy": "no_paths_member_names_fragment_pair_or_sample_ids",
            "test_or_sealed_inputs_permitted": False,
        },
    }
    receipt["content_sha256"] = _sha256(_canonical_json(receipt))
    portable_receipt = _portable(receipt)
    return LocalCacheInventoryResult(
        receipt=portable_receipt,
        cache_receipt=_portable(cache_receipt),
        semantic_commitment_sha256=semantic_commitment,
        cache_receipt_canonical_sha256=cache_receipt_hash,
    )


def reopen_and_verify_local_cache_inventory(
    *,
    root: Path,
    cache_receipt: Mapping[str, Any],
    cache_trust: FrozenReceiptTrust,
    records: Sequence[TrainingPairRecord],
    mask_loader: Callable[[MaskMemberRef], np.ndarray],
    expected_semantic_commitment_sha256: str,
    config: Optional[LocalCacheInventoryConfig] = None,
    limits: Optional[GeometryCacheLimits] = None,
) -> LocalCacheInventoryResult:
    """Eagerly verify a frozen cache and replay the label-blind census.

    The expected semantic commitment must come from the experiment lock, not
    from the inventory being replayed.  Every required artifact is verified by
    :meth:`GeometryArtifactCache.from_frozen_receipt` before any result is
    returned, and the replay must perform zero cache writes/misses.
    """

    if not isinstance(
        expected_semantic_commitment_sha256, str
    ) or not _SHA256_RE.fullmatch(expected_semantic_commitment_sha256):
        raise ValueError("expected semantic commitment must be lowercase SHA-256")
    cache = GeometryArtifactCache.from_frozen_receipt(
        root,
        cache_receipt,
        limits=limits,
        trust=cache_trust,
    )
    result = build_local_cache_inventory(records, mask_loader, cache, config)
    if result.semantic_commitment_sha256 != expected_semantic_commitment_sha256:
        raise LocalCacheInventoryError(
            "frozen local cache semantic commitment differs from experiment lock"
        )
    operational = result.receipt["operational_counts"]
    if operational["cache_miss_count"] != 0:
        raise LocalCacheInventoryError("frozen local cache replay produced a miss")
    if operational["cache_hit_count"] != operational["unique_canonical_fragment_count"]:
        raise LocalCacheInventoryError(
            "frozen local cache replay did not hit every canonical fragment"
        )
    return result


def write_portable_inventory(path: Path, receipt: Mapping[str, Any]) -> str:
    """Write canonical JSON and return its exact file SHA-256."""

    target = Path(path)
    if target.exists():
        raise LocalCacheInventoryError("inventory output already exists")
    payload = _canonical_json(_portable(dict(receipt)))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return _sha256(payload)


__all__ = [
    "LOCAL_CACHE_INVENTORY_SCHEMA_VERSION",
    "LocalCacheInventoryConfig",
    "LocalCacheInventoryError",
    "LocalCacheInventoryResult",
    "build_local_cache_inventory",
    "reopen_and_verify_local_cache_inventory",
    "write_portable_inventory",
]

"""Minimal canonical-MM 30k/6k selection and exact local-geometry cache.

The selection is fixed-seed and origin-by-label stratified.  A small ranked
reserve is geometry-qualified with the same four-direction LOCAL-Q1 geometry
configuration; unused reserve artifacts are removed from the freshly created
cache before its normal zero-miss read-only freeze.  The resulting sibling
selection JSON is the shared population input for coarse Siamese, local
dual-softmax, and dustbin Sinkhorn.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from staging.pairwise_v0_2.pairwise_data.lazy_mask_loader import (
    ArchiveSourceSpec,
    LazyMaskArchiveLoader,
)
from staging.pairwise_v0_2.pairwise_data.mm30k import (
    MM30K_SELECTION_SCHEMA_VERSION,
    MM30KSelectionConfig,
    MM30KSelectionError,
    mm30k_stratum_counts,
    select_mm30k_qualified,
    select_mm30k_ranked_reserve,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MM_CANONICAL_BINDING,
    MaskMemberRef,
    TrainingPairRecord,
    iter_historical_pair_records,
)
from staging.pairwise_v0_2.training.fragment_geometry_cache import (
    FragmentGeometryCacheLookup,
    load_or_build_fragment_geometry,
)
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.geometry_cache import (
    GeometryArtifactCache,
    GeometryCacheLimits,
)
from staging.pairwise_v0_2.training.local_cache_inventory import (
    LocalCacheInventoryConfig,
)
from staging.pairwise_v0_2.training.local_q1_cache_builder import (
    CACHE_DIRECTORY_NAME,
    LocalQ1CacheBuildArtifacts,
    LocalQ1CacheBuilderError,
    LocalQ1FileLock,
    LocalQ1Population,
    LocalQ1PrecacheAuthority,
    build_local_q1_cache,
    prebuild_local_q1_fragment_reserve,
)
from staging.pairwise_v0_2.training.local_q1_pair_qualification import (
    qualify_local_q1_pair_geometry,
)


MM30K_SELECTION_SUFFIX = ".mm30k_selection.json"


class MM30KResearchError(RuntimeError):
    """The direct MM30K cache/population path failed closed."""


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


def _read(path: Path) -> bytes:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise MM30KResearchError("required MM30K input is not a regular file")
    return target.read_bytes()


def _file_lock(path: Path) -> LocalQ1FileLock:
    payload = _read(path)
    return LocalQ1FileLock(byte_count=len(payload), sha256=_sha256(payload))


def mm30k_selection_path(cache_dir: Path) -> Path:
    target = Path(cache_dir)
    return target.parent / (target.name + MM30K_SELECTION_SUFFIX)


@dataclass(frozen=True)
class MM30KMaskLoaderFactory:
    mm_archive: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "mm_archive", Path(self.mm_archive))

    def __call__(self) -> LazyMaskArchiveLoader:
        return LazyMaskArchiveLoader(
            {
                MM_CANONICAL_BINDING.logical_id: ArchiveSourceSpec(
                    binding=MM_CANONICAL_BINDING,
                    source=self.mm_archive,
                )
            }
        )


def iter_canonical_mm_records(
    *, mm_archive: Path, historical_split: Path, split: str
) -> Iterable[TrainingPairRecord]:
    return iter_historical_pair_records(
        split_manifest=historical_split,
        split=split,
        mm_archive=mm_archive,
        eccv_archive=None,
    )


def _reference_key(reference: MaskMemberRef) -> Tuple[str, ...]:
    return (
        reference.binding.logical_id,
        reference.binding.sha256,
        reference.archive_member,
        reference.threshold_rule,
    )


def _reserve_lookups(
    *,
    records: Sequence[TrainingPairRecord],
    loader_factory: MM30KMaskLoaderFactory,
    cache_root: Path,
    inventory_config: LocalCacheInventoryConfig,
    cache_limits: GeometryCacheLimits,
) -> Mapping[Tuple[str, ...], FragmentGeometryCacheLookup]:
    references: Dict[Tuple[str, ...], MaskMemberRef] = {}
    for record in records:
        for reference in (record.fragment_a, record.fragment_b):
            references.setdefault(_reference_key(reference), reference)
    cache = GeometryArtifactCache(cache_root, limits=cache_limits)
    loader = loader_factory()
    lookups: Dict[Tuple[str, ...], FragmentGeometryCacheLookup] = {}
    try:
        for key in sorted(references):
            reference = references[key]
            mask = np.asarray(loader(reference))
            if mask.ndim != 2 or mask.size == 0 or mask.dtype != np.bool_:
                raise MM30KResearchError("MM loader did not return a 2D bool mask")
            if int(mask.size) > inventory_config.max_mask_pixels:
                raise MM30KResearchError("MM mask exceeds geometry pixel bound")
            lookup = load_or_build_fragment_geometry(
                np.ascontiguousarray(mask, dtype=np.bool_),
                reference.threshold_rule,
                inventory_config.geometry,
                cache,
            )
            if not lookup.cache_hit:
                raise MM30KResearchError("reserve replay unexpectedly missed cache")
            lookups[key] = lookup
    finally:
        loader.close()
    return MappingProxyType(lookups)


def _qualification(
    records: Sequence[TrainingPairRecord],
    *,
    lookups: Mapping[Tuple[str, ...], FragmentGeometryCacheLookup],
    geometry_config: GeometryBatchConfig,
) -> Tuple[Mapping[str, bool], Mapping[str, int]]:
    eligible: Dict[str, bool] = {}
    failures: Dict[str, int] = {}
    for record in records:
        result = qualify_local_q1_pair_geometry(
            lookups[_reference_key(record.fragment_a)].result,
            lookups[_reference_key(record.fragment_b)].result,
            pair_id=record.pair_id,
            geometry_config=geometry_config,
            require_candidates=True,
        )
        eligible[record.pair_id] = result.eligible
        if not result.eligible:
            stage = result.failure_stage or "unknown"
            failures[stage] = failures.get(stage, 0) + 1
    return MappingProxyType(eligible), MappingProxyType(dict(sorted(failures.items())))


def _selection_receipt(
    *,
    training: Sequence[TrainingPairRecord],
    validation: Sequence[TrainingPairRecord],
    config: MM30KSelectionConfig,
    failure_counts: Mapping[str, int],
) -> Mapping[str, Any]:
    train_tokens = [record.pair_id for record in training]
    validation_tokens = [record.pair_id for record in validation]
    identity_payload = {
        "training_record_tokens": train_tokens,
        "validation_record_tokens": validation_tokens,
    }
    train_components = {record.component_id for record in training}
    validation_components = {record.component_id for record in validation}
    overlap = train_components.intersection(validation_components)
    if overlap:
        raise MM30KResearchError("MM train/validation components overlap")
    receipt: Dict[str, Any] = {
        "schema_version": MM30K_SELECTION_SCHEMA_VERSION,
        "status": "mm30k_geometry_qualified_fixed_seed_no_test",
        "scope": {
            "experiment": "MM30K-PAIRWISE",
            "datasets": ["mm_augmented"],
            "pair_stream_splits_read": ["train", "val"],
            "historical_test_read": False,
            "sealed_real_read": False,
            "archive_mask_members_opened": True,
            "mask_pixels_decoded": True,
            "model_imported": False,
            "model_executed": False,
            "geometry_qualification_executed": True,
            "fixture_scale_test_only": config.fixture_scale_test_only,
        },
        "locks": {
            "population_identity": {
                "member_count": len(training) + len(validation),
                "content_sha256": _sha256(_canonical_json(identity_payload)),
            }
        },
        "selection": {
            "algorithm": "sha256_origin_label_ranked_reserve_then_geometry_qualified_v1",
            "seed_sha256": _sha256(config.seed.encode("utf-8")),
            "train_quotas": dict(sorted(config.train_quotas.items())),
            "validation_quotas": dict(sorted(config.validation_quotas.items())),
            "all_selected_pairs_geometry_qualified": True,
            "candidate_truncation_used": False,
            "train_validation_component_overlap_count": 0,
            "geometry_failure_counts_in_ranked_reserves": dict(failure_counts),
        },
        "training": {
            "population": {"count": len(training)},
            "record_tokens": train_tokens,
        },
        "validation": {
            "population": {"count": len(validation)},
            "record_tokens": validation_tokens,
        },
    }
    receipt["content_sha256"] = _sha256(_canonical_json(receipt))
    return MappingProxyType(receipt)


def _write_selection(path: Path, receipt: Mapping[str, Any]) -> bytes:
    payload = _canonical_json(receipt)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise MM30KResearchError("MM30K selection output already exists") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def _load_selection(path: Path) -> Tuple[Mapping[str, Any], bytes]:
    payload = _read(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MM30KResearchError("MM30K selection is not JSON") from exc
    if not isinstance(value, Mapping) or payload != _canonical_json(value):
        raise MM30KResearchError("MM30K selection must be canonical JSON")
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    if claimed != _sha256(_canonical_json(unsigned)):
        raise MM30KResearchError("MM30K selection content hash changed")
    if (
        value.get("schema_version") != MM30K_SELECTION_SCHEMA_VERSION
        or value.get("status") != "mm30k_geometry_qualified_fixed_seed_no_test"
    ):
        raise MM30KResearchError("MM30K selection schema/status changed")
    return MappingProxyType(dict(value)), payload


def _replay_tokens(
    records: Iterable[TrainingPairRecord], tokens: Sequence[str]
) -> Tuple[TrainingPairRecord, ...]:
    desired = set(tokens)
    if len(desired) != len(tokens):
        raise MM30KResearchError("MM30K selection repeats a record token")
    found: Dict[str, TrainingPairRecord] = {}
    for record in records:
        token = record.pair_id
        if token in desired:
            if token in found:
                raise MM30KResearchError("canonical MM replay repeated a selected pair")
            found[token] = record
    if set(found) != desired:
        raise MM30KResearchError("canonical MM replay is missing selected pairs")
    return tuple(found[token] for token in tokens)


def _population(
    *,
    training: Sequence[TrainingPairRecord],
    validation: Sequence[TrainingPairRecord],
    receipt: Mapping[str, Any],
    receipt_payload: bytes,
    selection_path: Path,
    mm_archive: Path,
    historical_split: Path,
) -> LocalQ1Population:
    if dict(mm30k_stratum_counts(training, split="train")) != receipt["selection"][
        "train_quotas"
    ]:
        raise MM30KResearchError("MM30K training strata differ from selection")
    if dict(mm30k_stratum_counts(validation, split="val")) != receipt["selection"][
        "validation_quotas"
    ]:
        raise MM30KResearchError("MM30K validation strata differ from selection")
    if {record.component_id for record in training}.intersection(
        record.component_id for record in validation
    ):
        raise MM30KResearchError("MM30K train/validation components overlap")
    file_sha = _sha256(receipt_payload)
    roles = {
        "freeze_receipt": LocalQ1FileLock(len(receipt_payload), file_sha),
        "mm_archive": _file_lock(mm_archive),
        "historical_split": _file_lock(historical_split),
    }
    authority = _sha256(
        _canonical_json(
            {
                "selection_file_sha256": file_sha,
                "mm_archive_sha256": roles["mm_archive"].sha256,
                "historical_split_sha256": roles["historical_split"].sha256,
            }
        )
    )
    del selection_path
    return LocalQ1Population(
        training_records=tuple(training),
        validation_records=tuple(validation),
        freeze_receipt=receipt,
        freeze_file_sha256=file_sha,
        freeze_content_sha256=receipt["content_sha256"],
        input_role_locks=roles,
        precache_authority=LocalQ1PrecacheAuthority(
            run_plan_file_sha256=authority,
            run_plan_content_sha256=authority,
            source_bundle_manifest_sha256=_sha256(Path(__file__).read_bytes()),
        ),
    )


def rebuild_mm30k_population(
    *, mm_archive: Path, historical_split: Path, selection_path: Path
) -> LocalQ1Population:
    """Replay the exact 30k/6k token list from canonical MM metadata."""

    receipt, payload = _load_selection(selection_path)
    try:
        train_tokens = tuple(receipt["training"]["record_tokens"])
        validation_tokens = tuple(receipt["validation"]["record_tokens"])
    except (KeyError, TypeError) as exc:
        raise MM30KResearchError("MM30K selection record tokens are missing") from exc
    training = _replay_tokens(
        iter_canonical_mm_records(
            mm_archive=mm_archive,
            historical_split=historical_split,
            split="train",
        ),
        train_tokens,
    )
    validation = _replay_tokens(
        iter_canonical_mm_records(
            mm_archive=mm_archive,
            historical_split=historical_split,
            split="val",
        ),
        validation_tokens,
    )
    return _population(
        training=training,
        validation=validation,
        receipt=receipt,
        receipt_payload=payload,
        selection_path=selection_path,
        mm_archive=mm_archive,
        historical_split=historical_split,
    )


@dataclass(frozen=True)
class MM30KPreparedCache:
    population: LocalQ1Population
    selection_path: Path
    cache_artifacts: LocalQ1CacheBuildArtifacts
    reserve_record_count: int
    pruned_artifact_count: int


def prepare_mm30k_cache(
    *,
    mm_archive: Path,
    historical_split: Path,
    output_dir: Path,
    selection_path: Path,
    inventory_config: LocalCacheInventoryConfig,
    geometry_config: GeometryBatchConfig,
    cache_limits: GeometryCacheLimits,
    producer_workers: int,
    selection_config: MM30KSelectionConfig | None = None,
) -> MM30KPreparedCache:
    """Select, qualify, prune, freeze, and zero-miss replay MM30K once."""

    config = selection_config or MM30KSelectionConfig()
    if inventory_config.geometry != geometry_config.geometry:
        raise MM30KResearchError("inventory and tensor geometry configs differ")
    if Path(selection_path).exists() or Path(selection_path).is_symlink():
        raise MM30KResearchError("MM30K selection output must be fresh")
    train_reserve = select_mm30k_ranked_reserve(
        iter_canonical_mm_records(
            mm_archive=mm_archive,
            historical_split=historical_split,
            split="train",
        ),
        split="train",
        config=config,
    )
    validation_reserve = select_mm30k_ranked_reserve(
        iter_canonical_mm_records(
            mm_archive=mm_archive,
            historical_split=historical_split,
            split="val",
        ),
        split="val",
        config=config,
    )
    reserve = train_reserve + validation_reserve
    loader_factory = MM30KMaskLoaderFactory(mm_archive)
    stats = dict(
        prebuild_local_q1_fragment_reserve(
            records=reserve,
            loader_factory=loader_factory,
            output_dir=output_dir,
            inventory_config=inventory_config,
            cache_limits=cache_limits,
            producer_workers=producer_workers,
        )
    )
    cache_root = Path(output_dir) / CACHE_DIRECTORY_NAME
    lookups = _reserve_lookups(
        records=reserve,
        loader_factory=loader_factory,
        cache_root=cache_root,
        inventory_config=inventory_config,
        cache_limits=cache_limits,
    )
    train_eligible, train_failures = _qualification(
        train_reserve, lookups=lookups, geometry_config=geometry_config
    )
    validation_eligible, validation_failures = _qualification(
        validation_reserve, lookups=lookups, geometry_config=geometry_config
    )
    training = select_mm30k_qualified(
        train_reserve,
        split="train",
        quotas=config.train_quotas,
        eligible_tokens=train_eligible,
        seed=config.seed,
    )
    validation = select_mm30k_qualified(
        validation_reserve,
        split="val",
        quotas=config.validation_quotas,
        eligible_tokens=validation_eligible,
        seed=config.seed,
    )
    selected_identity_keys = {
        lookups[_reference_key(reference)].identity.key
        for record in training + validation
        for reference in (record.fragment_a, record.fragment_b)
    }
    artifact_paths = tuple(cache_root.rglob("*.npz"))
    if len(artifact_paths) != stats["producer_cache_miss_count"]:
        raise MM30KResearchError("reserve cache artifact count differs from producer")
    for path in artifact_paths:
        if path.stem not in selected_identity_keys:
            path.unlink()
    remaining = tuple(cache_root.rglob("*.npz"))
    if {path.stem for path in remaining} != selected_identity_keys:
        raise MM30KResearchError("pruned MM30K cache differs from selected identities")
    stats["pruned_artifact_count"] = len(artifact_paths) - len(remaining)
    # Reconcile the quota ledger after removing only fresh, unselected reserve files.
    GeometryArtifactCache(cache_root, limits=cache_limits)
    failures: Dict[str, int] = {}
    for source in (train_failures, validation_failures):
        for stage, count in source.items():
            failures[stage] = failures.get(stage, 0) + count
    receipt = _selection_receipt(
        training=training,
        validation=validation,
        config=config,
        failure_counts=failures,
    )
    receipt_payload = _write_selection(selection_path, receipt)
    population = _population(
        training=training,
        validation=validation,
        receipt=receipt,
        receipt_payload=receipt_payload,
        selection_path=selection_path,
        mm_archive=mm_archive,
        historical_split=historical_split,
    )
    try:
        artifacts = build_local_q1_cache(
            population=population,
            loader_factory=loader_factory,
            output_dir=output_dir,
            inventory_config=inventory_config,
            cache_limits=cache_limits,
            producer_workers=producer_workers,
            fresh_prebuilt_producer_stats=stats,
        )
    except (LocalQ1CacheBuilderError, MM30KSelectionError) as exc:
        raise MM30KResearchError("MM30K exact cache freeze failed") from exc
    return MM30KPreparedCache(
        population=population,
        selection_path=Path(selection_path),
        cache_artifacts=artifacts,
        reserve_record_count=len(reserve),
        pruned_artifact_count=stats["pruned_artifact_count"],
    )


__all__ = [
    "MM30KMaskLoaderFactory",
    "MM30KPreparedCache",
    "MM30KResearchError",
    "MM30K_SELECTION_SUFFIX",
    "iter_canonical_mm_records",
    "mm30k_selection_path",
    "prepare_mm30k_cache",
    "rebuild_mm30k_population",
]

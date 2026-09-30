#!/usr/bin/env python3
"""Shortest Route-A cache -> batch plan -> CUDA dual/dustbin research path.

This is intentionally a research launcher, not a new authority framework.  It
replays the already frozen Route-A population, uses the existing cache/provider
contracts, and derives the bookkeeping hashes required by those APIs directly
from the supplied files.  It never accepts a test or sealed-real-data input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from staging.pairwise_v0_2.geometry import CandidateBuilderConfig, CorrosionConfig
from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    HistoricalIdentityIndex,
    historical_identity_index_content_sha256,
)
from staging.pairwise_v0_2.pairwise_data.lazy_mask_loader import (
    ArchiveSourceSpec,
    LazyMaskArchiveLoader,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    MaskMemberRef,
    iter_historical_pair_records,
    iter_synthetic_pair_records_payload,
)
from staging.pairwise_v0_2.preflight.freeze_local_q1 import (
    ROUTE_A_SCHEMA_VERSION,
    SYNTHETIC_ARCHIVE_BINDING,
    freeze_local_q1,
)
from staging.pairwise_v0_2.preflight.local_q1_geometry_eligibility_authority import (
    LocalQ1EligibilityTrust,
    LocalQ1ExternalFileLock,
    load_local_q1_geometry_eligibility_authority_payloads_v2,
    load_local_q1_route_a_config_authority_payload_contract,
)
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.geometry_cache import GeometryCacheLimits
from staging.pairwise_v0_2.training.local_cache_inventory import (
    LocalCacheInventoryConfig,
)
from staging.pairwise_v0_2.training.local_q1_backend import (
    LocalQ1Backend,
    LocalQ1BackendMode,
)
from staging.pairwise_v0_2.training.local_q1_cache_builder import (
    BUILD_RECEIPT_NAME,
    CACHE_RECEIPT_NAME,
    INVENTORY_RECEIPT_NAME,
    PLANNING_CONFIG_NAME,
    LocalQ1CacheTrust,
    LocalQ1FileLock,
    LocalQ1Population,
    LocalQ1PrecacheAuthority,
    attest_rebuilt_local_q1_population,
    build_local_q1_cache,
    reopen_local_q1_cache,
)
from staging.pairwise_v0_2.training.local_q1_plan import (
    LOCAL_Q1_PLANNING_CONFIG_SCHEMA_VERSION,
    ExplicitLocalQ1PlanningConfig,
    freeze_local_q1_batch_plan,
    load_explicit_local_q1_planning_config,
)
from staging.pairwise_v0_2.training.local_q1_provider import (
    LocalQ1BatchPlanConfig,
    LocalQ1ExternalLocks,
    LocalQ1ReadOnlyBatchProvider,
    reopen_local_q1_batch_plan,
)
from staging.pairwise_v0_2.training.local_q1_runner import (
    LocalQ1RunnerContract,
    run_local_q1,
)
from staging.pairwise_v0_2.training.mm30k_research import (
    MM30KMaskLoaderFactory,
    mm30k_selection_path,
    prepare_mm30k_cache,
    rebuild_mm30k_population,
)


RESEARCH_BRIDGE_VERSION = "dunhuang-local-q1-route-a-research-bridge/0.1"
_FORMAL_FOUR_ARMS = (
    "local_dual_softmax",
    "local_dustbin_sinkhorn",
    "keypoint_dual_softmax",
    "keypoint_dustbin_sinkhorn",
)
RESEARCH_ARM_MATRIX = MappingProxyType(
    {
        # Backward-compatible CLI aliases.  The formal runner intentionally
        # executes the complete 2x2 comparison for either spelling.
        "multirun": _FORMAL_FOUR_ARMS,
        "keypoint": _FORMAL_FOUR_ARMS,
    }
)
_ROLE_NAMES = (
    "freeze_receipt",
    "mm_archive",
    "eccv_archive",
    "mm_fingerprint_cache",
    "eccv_fingerprint_cache",
    "historical_split",
    "synthetic_manifest",
    "synthetic_archive",
)
_LARGE_ARCHIVE_ROLES = frozenset(
    ("mm_archive", "eccv_archive", "synthetic_archive")
)


class RouteAResearchError(RuntimeError):
    """The direct Route-A research path is incomplete or inconsistent."""


def representation_contract(representation: str) -> Mapping[str, Any]:
    """Return the integrated four-arm tensor contract used by the runner."""

    if representation not in RESEARCH_ARM_MATRIX:
        raise RouteAResearchError("unsupported local representation")
    return MappingProxyType(
        {
            "representation": "formal_multirun_keypoint_2x2",
            "cli_representation_alias": representation,
            "arms": RESEARCH_ARM_MATRIX[representation],
            "correspondence_mask": (
                "explicit_sparse_bool_N_La_Lb_for_keypoint_and_"
                "implicit_full_cartesian_for_multirun"
            ),
            "tensor_bridge": "training.geometry_batch.RaggedGeometryBatch",
            "training_status": "launchable_now",
        }
    )


def _require_launchable_representation(representation: str) -> None:
    contract = representation_contract(representation)
    if contract["training_status"] != "launchable_now":
        raise RouteAResearchError("local four-arm representation is not launchable")


def _canonical_json(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: normalize(child) for key, child in item.items()}
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


def _read(path: Path) -> bytes:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RouteAResearchError("required input is not a regular file: " + str(path))
    return target.read_bytes()


def _json(payload: bytes, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RouteAResearchError(name + " is not JSON") from exc
    if not isinstance(value, Mapping):
        raise RouteAResearchError(name + " root must be an object")
    return value


def _content_sha256(value: Mapping[str, Any], name: str) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    observed = _sha256(_canonical_json(unsigned))
    if claimed != observed:
        raise RouteAResearchError(name + " content hash is inconsistent")
    return observed


def _file_lock(path: Path) -> LocalQ1FileLock:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RouteAResearchError("required input is not a regular file: " + str(path))
    digest = hashlib.sha256()
    byte_count = 0
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return LocalQ1FileLock(byte_count=byte_count, sha256=digest.hexdigest())


@dataclass(frozen=True)
class RouteAResearchInputs:
    route_a_freeze: Path
    predecessor_freeze: Path
    eligibility_index: Path
    eligibility_receipt: Path
    route_policy: Path
    route_config: Path
    mm_archive: Path
    eccv_archive: Path
    mm_fingerprint_cache: Path
    eccv_fingerprint_cache: Path
    historical_split: Path
    synthetic_manifest: Path
    synthetic_archive: Path
    mm_extracted_root: Optional[Path] = None
    eccv_extracted_root: Optional[Path] = None
    synthetic_extracted_root: Optional[Path] = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else Path(value))
        extracted = (
            self.mm_extracted_root,
            self.eccv_extracted_root,
            self.synthetic_extracted_root,
        )
        if any(value is not None for value in extracted) and not all(
            value is not None for value in extracted
        ):
            raise RouteAResearchError(
                "MM, ECCV, and synthetic extracted roots must be supplied together"
            )

    @property
    def role_paths(self) -> Mapping[str, Path]:
        return MappingProxyType(
            {
                "freeze_receipt": self.route_a_freeze,
                "mm_archive": self.mm_archive,
                "eccv_archive": self.eccv_archive,
                "mm_fingerprint_cache": self.mm_fingerprint_cache,
                "eccv_fingerprint_cache": self.eccv_fingerprint_cache,
                "historical_split": self.historical_split,
                "synthetic_manifest": self.synthetic_manifest,
                "synthetic_archive": self.synthetic_archive,
            }
        )


@dataclass(frozen=True)
class MM30KResearchInputs:
    route_config: Path
    mm_archive: Path
    historical_split: Path

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, Path(getattr(self, name)))


def _eligibility_trust(value: Mapping[str, Any]) -> LocalQ1EligibilityTrust:
    roles = value.get("input_role_locks")
    if not isinstance(roles, Mapping) or set(roles) != set(_ROLE_NAMES):
        raise RouteAResearchError("Route-A freeze lacks eligibility input locks")
    try:
        return LocalQ1EligibilityTrust(
            predecessor_freeze_file_sha256=value["predecessor_freeze_file_sha256"],
            predecessor_freeze_content_sha256=value[
                "predecessor_freeze_content_sha256"
            ],
            input_role_locks={
                role: LocalQ1ExternalFileLock(
                    bytes=roles[role]["bytes"], sha256=roles[role]["sha256"]
                )
                for role in _ROLE_NAMES
            },
            source_bundle_manifest_sha256=value["source_bundle_manifest_sha256"],
            geometry_batch_config_sha256=value["geometry_batch_config_sha256"],
            planning_guard_config_sha256=value["planning_guard_config_sha256"],
            eligibility_policy_file_sha256=value["eligibility_policy_file_sha256"],
            eligibility_policy_content_sha256=value[
                "eligibility_policy_content_sha256"
            ],
            config_authority_file_sha256=value["config_authority_file_sha256"],
            config_authority_content_sha256=value["config_authority_content_sha256"],
            eligibility_index_file_sha256=value["eligibility_index_file_sha256"],
            eligibility_index_content_sha256=value["eligibility_index_content_sha256"],
            eligibility_receipt_file_sha256=value["eligibility_receipt_file_sha256"],
            eligibility_receipt_content_sha256=value[
                "eligibility_receipt_content_sha256"
            ],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RouteAResearchError("Route-A eligibility locks are incomplete") from exc


def rebuild_route_a_population(inputs: RouteAResearchInputs) -> LocalQ1Population:
    """Replay the existing Route-A decisions and return the exact 6,964 pairs."""

    if not isinstance(inputs, RouteAResearchInputs):
        raise TypeError("inputs must be RouteAResearchInputs")
    freeze_payload = _read(inputs.route_a_freeze)
    freeze_receipt = _json(freeze_payload, "Route-A freeze")
    freeze_content = _content_sha256(freeze_receipt, "Route-A freeze")
    if (
        freeze_receipt.get("schema_version") != ROUTE_A_SCHEMA_VERSION
        or freeze_receipt.get("status")
        != "pass_route_a_geometry_qualified_population_refrozen_no_model_no_test"
    ):
        raise RouteAResearchError("input is not the production Route-A freeze")
    authority_row = freeze_receipt.get("geometry_eligibility_authority")
    if not isinstance(authority_row, Mapping):
        raise RouteAResearchError("Route-A freeze lacks eligibility authority")
    trust = _eligibility_trust(authority_row.get("external_locks", {}))

    predecessor_payload = _read(inputs.predecessor_freeze)
    if (
        len(predecessor_payload) != trust.input_role_locks["freeze_receipt"].bytes
        or _sha256(predecessor_payload) != trust.predecessor_freeze_file_sha256
    ):
        raise RouteAResearchError("predecessor freeze differs from Route-A lock")
    # When complete extracted roots are supplied, they are both the mask source
    # and an archive-shaped metadata source.  The exact replay below remains
    # the population check; rereading multi-GB archive containers merely to
    # rediscover their already-frozen locks would add no training signal.
    extracted_metadata = inputs.mm_extracted_root is not None
    role_locks = {"freeze_receipt": _file_lock(inputs.route_a_freeze)}
    for role in _ROLE_NAMES:
        if role == "freeze_receipt":
            continue
        lock = trust.input_role_locks[role]
        if extracted_metadata and role in _LARGE_ARCHIVE_ROLES:
            observed = LocalQ1FileLock(byte_count=lock.bytes, sha256=lock.sha256)
        else:
            observed = _file_lock(inputs.role_paths[role])
        role_locks[role] = observed
        if observed.byte_count != lock.bytes or observed.sha256 != lock.sha256:
            raise RouteAResearchError("Route-A input role changed: " + role)

    index_payload = _read(inputs.eligibility_index)
    receipt_payload = _read(inputs.eligibility_receipt)
    policy_payload = _read(inputs.route_policy)
    config_payload = _read(inputs.route_config)
    eligibility = load_local_q1_geometry_eligibility_authority_payloads_v2(
        index_payload=index_payload,
        receipt_payload=receipt_payload,
        policy_payload=policy_payload,
        config_authority_payload=config_payload,
        trust=trust,
        expected_index_bytes=len(index_payload),
        expected_index_file_sha256=trust.eligibility_index_file_sha256,
        expected_index_content_sha256=trust.eligibility_index_content_sha256,
        expected_receipt_bytes=len(receipt_payload),
        expected_receipt_file_sha256=trust.eligibility_receipt_file_sha256,
        expected_receipt_content_sha256=trust.eligibility_receipt_content_sha256,
        expected_policy_bytes=len(policy_payload),
        expected_policy_file_sha256=trust.eligibility_policy_file_sha256,
        expected_policy_content_sha256=trust.eligibility_policy_content_sha256,
        expected_config_authority_bytes=len(config_payload),
        expected_config_authority_file_sha256=trust.config_authority_file_sha256,
        expected_config_authority_content_sha256=trust.config_authority_content_sha256,
    )

    mm_cache = _read(inputs.mm_fingerprint_cache)
    eccv_cache = _read(inputs.eccv_fingerprint_cache)
    split_payload = _read(inputs.historical_split)
    identity_index = HistoricalIdentityIndex.from_payloads(
        mm_cache_payload=mm_cache,
        eccv_cache_payload=eccv_cache,
        split_payload=split_payload,
    )
    expected_identity = freeze_receipt.get("locks", {}).get("historical_identity_index")
    if expected_identity != {
        "member_count": identity_index.identity_count,
        "content_sha256": historical_identity_index_content_sha256(identity_index),
    }:
        raise RouteAResearchError("historical identity index differs from freeze")
    split_document = _json(split_payload, "historical split")
    mm_metadata_source = (
        inputs.mm_extracted_root
        if extracted_metadata
        else inputs.mm_archive
    )
    eccv_metadata_source = (
        inputs.eccv_extracted_root
        if extracted_metadata
        else inputs.eccv_archive
    )

    def historical(split: str):
        return iter_historical_pair_records(
            split_manifest=split_document,
            split=split,
            mm_archive=mm_metadata_source,
            eccv_archive=eccv_metadata_source,
        )

    rebuilt = freeze_local_q1(
        identity_index=identity_index,
        validation_records=historical("val"),
        historical_training_records=historical("train"),
        synthetic_training_records=iter_synthetic_pair_records_payload(
            _read(inputs.synthetic_manifest)
        ),
        historical_archive_locks={
            "mm_augmented": {
                "format": MM_CANONICAL_BINDING.archive_format,
                "logical_id": MM_CANONICAL_BINDING.logical_id,
                "bytes": trust.input_role_locks["mm_archive"].bytes,
                "sha256": trust.input_role_locks["mm_archive"].sha256,
            },
            "eccv_1113data": {
                "format": ECCV_CANONICAL_BINDING.archive_format,
                "logical_id": ECCV_CANONICAL_BINDING.logical_id,
                "bytes": trust.input_role_locks["eccv_archive"].bytes,
                "sha256": trust.input_role_locks["eccv_archive"].sha256,
            },
        },
        synthetic_manifest_lock=trust.input_role_locks[
            "synthetic_manifest"
        ].portable_dict(),
        geometry_eligibility=eligibility,
        predecessor_freeze_file_bytes=predecessor_payload,
        expected_predecessor_freeze_bytes=len(predecessor_payload),
        expected_predecessor_freeze_file_sha256=(trust.predecessor_freeze_file_sha256),
        expected_predecessor_freeze_content_sha256=(
            trust.predecessor_freeze_content_sha256
        ),
        synthetic_archive_lock=trust.input_role_locks[
            "synthetic_archive"
        ].portable_dict(),
        expected_source_bundle_manifest_sha256=trust.source_bundle_manifest_sha256,
        expected_geometry_batch_config_sha256=trust.geometry_batch_config_sha256,
        expected_planning_guard_config_sha256=trust.planning_guard_config_sha256,
    )
    if _canonical_json(rebuilt.receipt) != _canonical_json(freeze_receipt):
        raise RouteAResearchError("Route-A replay differs from frozen population")

    research_binding = {
        "bridge_version": RESEARCH_BRIDGE_VERSION,
        "freeze_file_sha256": _sha256(freeze_payload),
        "freeze_content_sha256": freeze_content,
        "roles": {
            role: role_locks[role].portable_dict() for role in sorted(role_locks)
        },
    }
    run_identity = _sha256(_canonical_json(research_binding))
    source_identity = _sha256(_read(Path(__file__)))
    return attest_rebuilt_local_q1_population(
        canonical_freeze_file_bytes=freeze_payload,
        expected_freeze_file_sha256=_sha256(freeze_payload),
        expected_freeze_content_sha256=freeze_content,
        rebuilt=rebuilt,
        input_role_locks=role_locks,
        precache_authority=LocalQ1PrecacheAuthority(
            run_plan_file_sha256=run_identity,
            run_plan_content_sha256=run_identity,
            source_bundle_manifest_sha256=source_identity,
        ),
    )


def _candidate_config(value: Mapping[str, Any]) -> CandidateBuilderConfig:
    data = dict(value)
    data["corrosion"] = CorrosionConfig(**dict(data["corrosion"]))
    data["window_scale_fractions"] = tuple(data["window_scale_fractions"])
    data["output_size"] = tuple(data["output_size"])
    return CandidateBuilderConfig(**data)


def route_a_planning_config(route_config: Path) -> ExplicitLocalQ1PlanningConfig:
    payload = _read(route_config)
    document = _json(payload, "Route-A config")
    content = _content_sha256(document, "Route-A config")
    authority = load_local_q1_route_a_config_authority_payload_contract(
        payload=payload,
        expected_file_sha256=_sha256(payload),
        expected_content_sha256=content,
    )
    document = authority.document
    geometry_row = document["geometry_batch_config"]
    bounds = geometry_row["bounds"]
    geometry = _candidate_config(geometry_row["geometry"])
    geometry_config = GeometryBatchConfig(
        geometry=geometry,
        coarse_output_size=tuple(geometry_row["coarse_output_size"]),
        coarse_resize_mode=geometry_row["coarse_resize_mode"],
        coarse_preprocess_mode=geometry_row["coarse_preprocess_mode"],
        coarse_content_fraction=geometry_row["coarse_content_fraction"],
        coarse_component_connectivity=geometry_row["coarse_component_connectivity"],
        **dict(bounds),
    )
    guard = document["planning_guard_config"]
    inventory_row = dict(guard["inventory_config"])
    inventory_row["geometry"] = _candidate_config(inventory_row["geometry"])
    inventory_row["sequence_length_bucket_edges"] = tuple(
        inventory_row["sequence_length_bucket_edges"]
    )
    inventory_row["candidate_count_bucket_edges"] = tuple(
        inventory_row["candidate_count_bucket_edges"]
    )
    inventory_config = LocalCacheInventoryConfig(**inventory_row)
    batch_plan_config = LocalQ1BatchPlanConfig(**dict(guard["batch_plan_config"]))
    cache_limits = GeometryCacheLimits()
    if geometry_config.fingerprint != document["geometry_batch_config_sha256"]:
        raise RouteAResearchError("Route-A geometry config reconstruction changed")
    component_hashes = document["component_config_hashes"]
    if (
        _sha256(_canonical_json(inventory_config.portable_dict()))
        != component_hashes["local_cache_inventory_config_sha256"]
        or _sha256(_canonical_json(batch_plan_config.portable_dict()))
        != component_hashes["local_q1_batch_plan_config_sha256"]
    ):
        raise RouteAResearchError("Route-A planning guard reconstruction changed")
    receipt: Dict[str, Any] = {
        "schema_version": LOCAL_Q1_PLANNING_CONFIG_SCHEMA_VERSION,
        "status": "frozen_explicit_all_fields_no_defaults",
        "inventory_config": asdict(inventory_config),
        "geometry_batch_config": asdict(geometry_config),
        "batch_plan_config": asdict(batch_plan_config),
        "cache_limits": asdict(cache_limits),
        "authority_boundary": {
            "hash_arguments_source": "self_provided_by_enclosing_cli_caller",
            "independent_authority_verified_by_this_receipt": False,
            "enclosing_authority_status": "pending",
            "result_bearing_authorized": False,
            "promotion_requirement": (
                "later_independent_run_authority_must_lock_batch_plan_file_and_"
                "content_sha256_which_transitively_bind_this_config_before_any_"
                "result_bearing_execution"
            ),
        },
    }
    receipt["content_sha256"] = _sha256(_canonical_json(receipt))
    payload = _canonical_json(receipt)
    return ExplicitLocalQ1PlanningConfig(
        inventory_config=inventory_config,
        geometry_batch_config=geometry_config,
        batch_plan_config=batch_plan_config,
        cache_limits=cache_limits,
        receipt=receipt,
        canonical_file_sha256=_sha256(payload),
        content_sha256=receipt["content_sha256"],
    )


def mm30k_planning_config(route_config: Path) -> ExplicitLocalQ1PlanningConfig:
    """Reuse Route-A tensor geometry with only population-size bounds enlarged."""

    base = route_a_planning_config(route_config)
    inventory_config = replace(
        base.inventory_config,
        max_records=40_000,
        max_unique_references=80_000,
    )
    receipt: Dict[str, Any] = {
        "schema_version": LOCAL_Q1_PLANNING_CONFIG_SCHEMA_VERSION,
        "status": "frozen_explicit_all_fields_no_defaults",
        "inventory_config": asdict(inventory_config),
        "geometry_batch_config": asdict(base.geometry_batch_config),
        "batch_plan_config": asdict(base.batch_plan_config),
        "cache_limits": asdict(base.cache_limits),
        "authority_boundary": {
            "hash_arguments_source": "self_provided_by_enclosing_cli_caller",
            "independent_authority_verified_by_this_receipt": False,
            "enclosing_authority_status": "pending",
            "result_bearing_authorized": False,
            "promotion_requirement": (
                "later_independent_run_authority_must_lock_batch_plan_file_and_"
                "content_sha256_which_transitively_bind_this_config_before_any_"
                "result_bearing_execution"
            ),
        },
    }
    receipt["content_sha256"] = _sha256(_canonical_json(receipt))
    payload = _canonical_json(receipt)
    return ExplicitLocalQ1PlanningConfig(
        inventory_config=inventory_config,
        geometry_batch_config=base.geometry_batch_config,
        batch_plan_config=base.batch_plan_config,
        cache_limits=base.cache_limits,
        receipt=receipt,
        canonical_file_sha256=_sha256(payload),
        content_sha256=receipt["content_sha256"],
    )


def write_route_a_planning_config(
    path: Path, config: ExplicitLocalQ1PlanningConfig
) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise RouteAResearchError("planning-config output already exists")
    target.write_bytes(_canonical_json(config.receipt))


def planning_config_with_local_tensor_bound(
    config: ExplicitLocalQ1PlanningConfig,
    max_local_tensor_elements: Optional[int],
) -> ExplicitLocalQ1PlanningConfig:
    """Shrink only the research batch-packing bound and re-sign it in memory."""

    if max_local_tensor_elements is None:
        return config
    if (
        isinstance(max_local_tensor_elements, bool)
        or not isinstance(max_local_tensor_elements, int)
        or max_local_tensor_elements <= 0
    ):
        raise RouteAResearchError(
            "max local tensor elements must be a positive integer"
        )
    current = config.geometry_batch_config.max_local_tensor_elements
    if max_local_tensor_elements > current:
        raise RouteAResearchError(
            "research max local tensor elements may only shrink the frozen bound"
        )
    if max_local_tensor_elements == current:
        return config

    geometry_config = replace(
        config.geometry_batch_config,
        max_local_tensor_elements=max_local_tensor_elements,
    )
    receipt = dict(config.receipt)
    receipt["geometry_batch_config"] = asdict(geometry_config)
    receipt.pop("content_sha256", None)
    receipt["content_sha256"] = _sha256(_canonical_json(receipt))
    payload = _canonical_json(receipt)
    return replace(
        config,
        geometry_batch_config=geometry_config,
        receipt=receipt,
        canonical_file_sha256=_sha256(payload),
        content_sha256=receipt["content_sha256"],
    )


@dataclass(frozen=True)
class _ResearchMaskLoaderFactory:
    mm_archive: Path
    eccv_archive: Path
    synthetic_archive: Path
    mm_extracted_root: Optional[Path] = None
    eccv_extracted_root: Optional[Path] = None
    synthetic_extracted_root: Optional[Path] = None

    def __post_init__(self) -> None:
        for name in (
            "mm_archive",
            "eccv_archive",
            "synthetic_archive",
            "mm_extracted_root",
            "eccv_extracted_root",
            "synthetic_extracted_root",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else Path(value))
        extracted = (
            self.mm_extracted_root,
            self.eccv_extracted_root,
            self.synthetic_extracted_root,
        )
        if any(value is not None for value in extracted) and not all(
            value is not None for value in extracted
        ):
            raise RouteAResearchError(
                "MM, ECCV, and synthetic extracted roots must be supplied together"
            )

    def __call__(self) -> Any:
        if self.mm_extracted_root is not None:
            return _ExtractedResearchMaskLoader(
                {
                    MM_CANONICAL_BINDING.logical_id: (
                        MM_CANONICAL_BINDING,
                        self.mm_extracted_root,
                    ),
                    ECCV_CANONICAL_BINDING.logical_id: (
                        ECCV_CANONICAL_BINDING,
                        self.eccv_extracted_root,
                    ),
                    SYNTHETIC_ARCHIVE_BINDING.logical_id: (
                        SYNTHETIC_ARCHIVE_BINDING,
                        self.synthetic_extracted_root,
                    ),
                }
            )
        return LazyMaskArchiveLoader(
            {
                MM_CANONICAL_BINDING.logical_id: ArchiveSourceSpec(
                    binding=MM_CANONICAL_BINDING, source=self.mm_archive
                ),
                ECCV_CANONICAL_BINDING.logical_id: ArchiveSourceSpec(
                    binding=ECCV_CANONICAL_BINDING, source=self.eccv_archive
                ),
                SYNTHETIC_ARCHIVE_BINDING.logical_id: ArchiveSourceSpec(
                    binding=SYNTHETIC_ARCHIVE_BINDING,
                    source=self.synthetic_archive,
                ),
            }
        )


class _ExtractedResearchMaskLoader:
    """Read already-extracted PNG members directly without rescanning archives."""

    def __init__(self, sources: Mapping[str, Tuple[Any, Path]]) -> None:
        self._sources = {
            logical_id: (binding, Path(root))
            for logical_id, (binding, root) in sources.items()
        }

    def __call__(self, reference: MaskMemberRef) -> np.ndarray:
        if not isinstance(reference, MaskMemberRef):
            raise TypeError("mask loader requires MaskMemberRef")
        try:
            binding, root = self._sources[reference.binding.logical_id]
        except KeyError as exc:
            raise RouteAResearchError("no extracted root for mask reference") from exc
        if binding != reference.binding:
            raise RouteAResearchError("extracted-root binding disagrees with reference")
        path = root.joinpath(*reference.archive_member.split("/"))
        if not path.is_file():
            raise RouteAResearchError("extracted mask member is missing: " + str(path))
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise RouteAResearchError("extracted mask member must be PNG")
                image.load()
                if reference.threshold_rule == "grayscale_uint8_gt_127":
                    mask = np.asarray(image.convert("L"), dtype=np.uint8) > 127
                elif reference.threshold_rule == "binary_brighter_value":
                    values = np.asarray(image)
                    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
                        raise RouteAResearchError(
                            "historical masks must be scalar numeric PNGs"
                        )
                    if not np.all(np.isfinite(values)) or np.any(values < 0):
                        raise RouteAResearchError("historical mask values are invalid")
                    unique = np.unique(values)
                    if len(unique) > 2:
                        raise RouteAResearchError(
                            "historical mask has more than two scalar values"
                        )
                    if len(unique) == 1:
                        mask = np.full(values.shape, bool(unique[0] > 0), dtype=bool)
                    else:
                        mask = values == unique[-1]
                else:
                    raise RouteAResearchError("unsupported mask threshold rule")
        except RouteAResearchError:
            raise
        except (OSError, UnidentifiedImageError, TypeError, ValueError) as exc:
            raise RouteAResearchError("cannot decode extracted mask PNG") from exc
        output = np.ascontiguousarray(mask, dtype=bool)
        output.setflags(write=False)
        return output


def research_loader_factory(inputs: RouteAResearchInputs) -> _ResearchMaskLoaderFactory:
    return _ResearchMaskLoaderFactory(
        mm_archive=inputs.mm_archive,
        eccv_archive=inputs.eccv_archive,
        synthetic_archive=inputs.synthetic_archive,
        mm_extracted_root=inputs.mm_extracted_root,
        eccv_extracted_root=inputs.eccv_extracted_root,
        synthetic_extracted_root=inputs.synthetic_extracted_root,
    )


def _load_receipt(path: Path) -> Tuple[Mapping[str, Any], str, str]:
    payload = _read(path)
    value = _json(payload, path.name)
    return value, _sha256(payload), _content_sha256(value, path.name)


def automatic_cache_trust(
    population: LocalQ1Population, cache_dir: Path
) -> LocalQ1CacheTrust:
    cache, cache_file, cache_content = _load_receipt(
        Path(cache_dir) / CACHE_RECEIPT_NAME
    )
    inventory, inventory_file, inventory_content = _load_receipt(
        Path(cache_dir) / INVENTORY_RECEIPT_NAME
    )
    build, build_file, build_content = _load_receipt(
        Path(cache_dir) / BUILD_RECEIPT_NAME
    )
    del cache, build
    return LocalQ1CacheTrust(
        expected_freeze_file_sha256=population.freeze_file_sha256,
        expected_freeze_content_sha256=population.freeze_content_sha256,
        expected_run_plan_file_sha256=(
            population.precache_authority.run_plan_file_sha256
        ),
        expected_run_plan_content_sha256=(
            population.precache_authority.run_plan_content_sha256
        ),
        expected_source_bundle_manifest_sha256=(
            population.precache_authority.source_bundle_manifest_sha256
        ),
        expected_cache_receipt_file_sha256=cache_file,
        expected_cache_receipt_content_sha256=cache_content,
        expected_inventory_receipt_file_sha256=inventory_file,
        expected_inventory_receipt_content_sha256=inventory_content,
        expected_inventory_semantic_sha256=inventory["semantic_commitment_sha256"],
        expected_build_receipt_file_sha256=build_file,
        expected_build_receipt_content_sha256=build_content,
    )


def population_for_existing_cache(
    population: LocalQ1Population, cache_dir: Path
) -> LocalQ1Population:
    """Keep the frozen data population while adopting its cache-build identity.

    Research launcher fixes must not force a multi-hour geometry-cache rebuild.
    The completed build receipt already records the run/source bookkeeping used
    when those exact tensors were created.  Reuse only that bookkeeping row;
    the cache reopen still checks the same population freeze and every input
    role against the rebuilt records.
    """

    build, _file_sha256, _content_sha256_value = _load_receipt(
        Path(cache_dir) / BUILD_RECEIPT_NAME
    )
    row = build.get("precache_authority")
    if not isinstance(row, Mapping) or set(row) != {
        "run_plan_file_sha256",
        "run_plan_content_sha256",
        "source_bundle_manifest_sha256",
    }:
        raise RouteAResearchError("cache build lacks its research identity")
    population_lock = build.get("population_lock")
    if not isinstance(population_lock, Mapping) or (
        population_lock.get("freeze_file_sha256") != population.freeze_file_sha256
        or population_lock.get("freeze_content_sha256")
        != population.freeze_content_sha256
    ):
        raise RouteAResearchError("cache build belongs to a different population")
    try:
        authority = LocalQ1PrecacheAuthority(**dict(row))
    except (TypeError, ValueError) as exc:
        raise RouteAResearchError("cache build research identity is invalid") from exc
    return replace(population, precache_authority=authority)


def _external_locks(
    trust: LocalQ1CacheTrust,
    config: ExplicitLocalQ1PlanningConfig,
    *,
    plan_file_sha256: Optional[str] = None,
    plan_content_sha256: Optional[str] = None,
) -> LocalQ1ExternalLocks:
    return LocalQ1ExternalLocks(
        run_plan_file_sha256=trust.expected_run_plan_file_sha256,
        run_plan_content_sha256=trust.expected_run_plan_content_sha256,
        source_bundle_manifest_sha256=trust.expected_source_bundle_manifest_sha256,
        planning_config_receipt_file_sha256=config.canonical_file_sha256,
        planning_config_receipt_content_sha256=config.content_sha256,
        freeze_file_sha256=trust.expected_freeze_file_sha256,
        freeze_content_sha256=trust.expected_freeze_content_sha256,
        cache_receipt_file_sha256=trust.expected_cache_receipt_file_sha256,
        cache_receipt_content_sha256=trust.expected_cache_receipt_content_sha256,
        inventory_receipt_file_sha256=trust.expected_inventory_receipt_file_sha256,
        inventory_receipt_content_sha256=(
            trust.expected_inventory_receipt_content_sha256
        ),
        inventory_semantic_sha256=trust.expected_inventory_semantic_sha256,
        build_receipt_file_sha256=trust.expected_build_receipt_file_sha256,
        build_receipt_content_sha256=trust.expected_build_receipt_content_sha256,
        batch_plan_file_sha256=plan_file_sha256,
        batch_plan_content_sha256=plan_content_sha256,
    )


def build_route_a_cache(
    *,
    inputs: RouteAResearchInputs,
    output_dir: Path,
    producer_workers: int,
) -> Mapping[str, Any]:
    population = rebuild_route_a_population(inputs)
    config = route_a_planning_config(inputs.route_config)
    artifacts = build_local_q1_cache(
        population=population,
        loader_factory=research_loader_factory(inputs),
        output_dir=output_dir,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
        producer_workers=producer_workers,
    )
    config_path = Path(output_dir) / PLANNING_CONFIG_NAME
    write_route_a_planning_config(config_path, config)
    return {
        "status": "route_a_fresh_cache_complete_zero_miss_read_only",
        "cache_dir": str(Path(output_dir)),
        "planning_config": str(config_path),
        "artifact_count": artifacts.cache_receipt["artifact_count"],
        "training_count": len(population.training_records),
        "validation_count": len(population.validation_records),
    }


def build_route_a_batch_plan(
    *,
    inputs: RouteAResearchInputs,
    cache_dir: Path,
    output_path: Path,
    representation: str = "multirun",
    max_local_tensor_elements: Optional[int] = None,
) -> Mapping[str, Any]:
    _require_launchable_representation(representation)
    population = rebuild_route_a_population(inputs)
    population = population_for_existing_cache(population, cache_dir)
    config_path = Path(cache_dir) / PLANNING_CONFIG_NAME
    config_payload = _read(config_path)
    config_value = _json(config_payload, "planning config")
    config = load_explicit_local_q1_planning_config(
        config_path,
        expected_file_sha256=_sha256(config_payload),
        expected_content_sha256=_content_sha256(config_value, "planning config"),
    )
    config = planning_config_with_local_tensor_bound(config, max_local_tensor_elements)
    trust = automatic_cache_trust(population, cache_dir)
    loader_factory = research_loader_factory(inputs)
    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=loader_factory,
        output_dir=cache_dir,
        trust=trust,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
        replay_source_masks=False,
    )
    plan = freeze_local_q1_batch_plan(
        opened=opened,
        loader_factory=loader_factory,
        output_path=output_path,
        trust=trust,
        config=config,
    )
    return {
        "status": "route_a_four_phase_batch_plan_complete",
        "representation": dict(representation_contract(representation)),
        "batch_plan": str(Path(output_path)),
        "batch_plan_file_sha256": plan.canonical_file_sha256,
        "batch_plan_content_sha256": plan.content_sha256,
        "phase_batch_counts": {
            phase: len(plan.phase_batches(phase))
            for phase in (
                "train",
                "validation_select",
                "validation_calibration",
                "validation_report",
            )
        },
    }


def train_route_a_dual_and_dustbin(
    *,
    inputs: RouteAResearchInputs,
    cache_dir: Path,
    batch_plan: Path,
    output_root: Path,
    epochs: int,
    initialization_seed: int,
    representation: str = "multirun",
    max_local_tensor_elements: Optional[int] = None,
) -> Mapping[str, Any]:
    _require_launchable_representation(representation)
    population = rebuild_route_a_population(inputs)
    population = population_for_existing_cache(population, cache_dir)
    config_path = Path(cache_dir) / PLANNING_CONFIG_NAME
    config_payload = _read(config_path)
    config_value = _json(config_payload, "planning config")
    config = load_explicit_local_q1_planning_config(
        config_path,
        expected_file_sha256=_sha256(config_payload),
        expected_content_sha256=_content_sha256(config_value, "planning config"),
    )
    config = planning_config_with_local_tensor_bound(config, max_local_tensor_elements)
    trust = automatic_cache_trust(population, cache_dir)
    loader_factory = research_loader_factory(inputs)
    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=loader_factory,
        output_dir=cache_dir,
        trust=trust,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
        replay_source_masks=False,
    )
    plan_value, plan_file, plan_content = _load_receipt(batch_plan)
    del plan_value
    external_locks = _external_locks(
        trust,
        config,
        plan_file_sha256=plan_file,
        plan_content_sha256=plan_content,
    )
    plan = reopen_local_q1_batch_plan(batch_plan, external_locks=external_locks)
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=loader_factory,
        plan=plan,
        geometry_config=config.geometry_batch_config,
        inventory_config=config.inventory_config,
        external_locks=external_locks,
    )
    contract = LocalQ1RunnerContract(
        epochs=epochs,
        initialization_seed=initialization_seed,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=True,
    )

    def backend_factory() -> LocalQ1Backend:
        return LocalQ1Backend(device="cuda", mode=LocalQ1BackendMode.FORMAL)

    artifacts = run_local_q1(
        contract=contract,
        plan=plan,
        train_records=population.training_records,
        validation_records=None,
        provider=provider,
        backend_factory=backend_factory,
        output_root=output_root,
    )
    return {
        "status": "cuda_multirun_keypoint_four_arm_complete",
        "run_directory": str(artifacts.run_directory),
        "receipt": str(artifacts.receipt_path),
        "representation": dict(representation_contract(representation)),
        "arms": list(RESEARCH_ARM_MATRIX[representation]),
    }


def run_route_a_pipeline(
    *,
    inputs: RouteAResearchInputs,
    cache_dir: Path,
    batch_plan: Path,
    output_root: Path,
    producer_workers: int,
    epochs: int,
    initialization_seed: int,
    representation: str = "multirun",
    max_local_tensor_elements: Optional[int] = None,
) -> Mapping[str, Any]:
    """Run the shortest resumable fresh-cache -> plan -> CUDA baseline chain."""

    _require_launchable_representation(representation)
    population = rebuild_route_a_population(inputs)
    config = route_a_planning_config(inputs.route_config)
    cache_artifacts = build_local_q1_cache(
        population=population,
        loader_factory=research_loader_factory(inputs),
        output_dir=cache_dir,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
        producer_workers=producer_workers,
    )
    config_path = Path(cache_dir) / PLANNING_CONFIG_NAME
    write_route_a_planning_config(config_path, config)
    config = planning_config_with_local_tensor_bound(config, max_local_tensor_elements)
    trust = automatic_cache_trust(population, cache_dir)
    loader_factory = research_loader_factory(inputs)
    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=loader_factory,
        output_dir=cache_dir,
        trust=trust,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
        replay_source_masks=False,
    )
    plan = freeze_local_q1_batch_plan(
        opened=opened,
        loader_factory=loader_factory,
        output_path=batch_plan,
        trust=trust,
        config=config,
    )
    external_locks = _external_locks(
        trust,
        config,
        plan_file_sha256=plan.canonical_file_sha256,
        plan_content_sha256=plan.content_sha256,
    )
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=loader_factory,
        plan=plan,
        geometry_config=config.geometry_batch_config,
        inventory_config=config.inventory_config,
        external_locks=external_locks,
    )
    contract = LocalQ1RunnerContract(
        epochs=epochs,
        initialization_seed=initialization_seed,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=True,
    )

    def backend_factory() -> LocalQ1Backend:
        return LocalQ1Backend(device="cuda", mode=LocalQ1BackendMode.FORMAL)

    training = run_local_q1(
        contract=contract,
        plan=plan,
        train_records=population.training_records,
        validation_records=None,
        provider=provider,
        backend_factory=backend_factory,
        output_root=output_root,
    )
    return {
        "status": "route_a_cache_plan_cuda_multirun_keypoint_four_arm_complete",
        "cache_dir": str(Path(cache_dir)),
        "cache_artifact_count": cache_artifacts.cache_receipt["artifact_count"],
        "batch_plan": str(Path(batch_plan)),
        "phase_batch_counts": {
            phase: len(plan.phase_batches(phase))
            for phase in (
                "train",
                "validation_select",
                "validation_calibration",
                "validation_report",
            )
        },
        "run_directory": str(training.run_directory),
        "receipt": str(training.receipt_path),
        "representation": dict(representation_contract(representation)),
    }


def _planning_config_from_cache(cache_dir: Path) -> ExplicitLocalQ1PlanningConfig:
    config_path = Path(cache_dir) / PLANNING_CONFIG_NAME
    payload = _read(config_path)
    value = _json(payload, "planning config")
    return load_explicit_local_q1_planning_config(
        config_path,
        expected_file_sha256=_sha256(payload),
        expected_content_sha256=_content_sha256(value, "planning config"),
    )


def resume_route_a_plan_and_train(
    *,
    inputs: RouteAResearchInputs,
    cache_dir: Path,
    batch_plan: Path,
    output_root: Path,
    epochs: int,
    initialization_seed: int,
    representation: str = "multirun",
    max_local_tensor_elements: Optional[int] = None,
) -> Mapping[str, Any]:
    """Resume an existing Route-A cache with one reopen, then plan and train."""

    _require_launchable_representation(representation)
    population = rebuild_route_a_population(inputs)
    population = population_for_existing_cache(population, cache_dir)
    config = planning_config_with_local_tensor_bound(
        _planning_config_from_cache(cache_dir), max_local_tensor_elements
    )
    trust = automatic_cache_trust(population, cache_dir)
    loader_factory = research_loader_factory(inputs)
    # This is deliberately the only full cache reopen/replay in the resume
    # path.  The same in-memory opened cache flows directly into planning and
    # the provider used by all four formal arms.
    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=loader_factory,
        output_dir=cache_dir,
        trust=trust,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
        replay_source_masks=False,
    )
    plan = freeze_local_q1_batch_plan(
        opened=opened,
        loader_factory=loader_factory,
        output_path=batch_plan,
        trust=trust,
        config=config,
    )
    external_locks = _external_locks(
        trust,
        config,
        plan_file_sha256=plan.canonical_file_sha256,
        plan_content_sha256=plan.content_sha256,
    )
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=loader_factory,
        plan=plan,
        geometry_config=config.geometry_batch_config,
        inventory_config=config.inventory_config,
        external_locks=external_locks,
    )
    contract = LocalQ1RunnerContract(
        epochs=epochs,
        initialization_seed=initialization_seed,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=True,
    )

    def backend_factory() -> LocalQ1Backend:
        return LocalQ1Backend(device="cuda", mode=LocalQ1BackendMode.FORMAL)

    training = run_local_q1(
        contract=contract,
        plan=plan,
        train_records=population.training_records,
        validation_records=None,
        provider=provider,
        backend_factory=backend_factory,
        output_root=output_root,
    )
    return {
        "status": "route_a_existing_cache_single_reopen_plan_four_arm_train_complete",
        "cache_dir": str(Path(cache_dir)),
        "cache_reopen_count": 1,
        "batch_plan": str(Path(batch_plan)),
        "phase_batch_counts": {
            phase: len(plan.phase_batches(phase))
            for phase in (
                "train",
                "validation_select",
                "validation_calibration",
                "validation_report",
            )
        },
        "run_directory": str(training.run_directory),
        "receipt": str(training.receipt_path),
        "representation": dict(representation_contract(representation)),
        "arms": list(RESEARCH_ARM_MATRIX[representation]),
    }


def build_mm30k_cache(
    *, inputs: MM30KResearchInputs, output_dir: Path, producer_workers: int
) -> Mapping[str, Any]:
    config = mm30k_planning_config(inputs.route_config)
    selection_path = mm30k_selection_path(output_dir)
    prepared = prepare_mm30k_cache(
        mm_archive=inputs.mm_archive,
        historical_split=inputs.historical_split,
        output_dir=output_dir,
        selection_path=selection_path,
        inventory_config=config.inventory_config,
        geometry_config=config.geometry_batch_config,
        cache_limits=config.cache_limits,
        producer_workers=producer_workers,
    )
    config_path = Path(output_dir) / PLANNING_CONFIG_NAME
    write_route_a_planning_config(config_path, config)
    return {
        "status": "mm30k_fresh_cache_complete_zero_miss_read_only",
        "cache_dir": str(Path(output_dir)),
        "selection": str(selection_path),
        "planning_config": str(config_path),
        "artifact_count": prepared.cache_artifacts.cache_receipt["artifact_count"],
        "training_count": len(prepared.population.training_records),
        "validation_count": len(prepared.population.validation_records),
        "ranked_reserve_record_count": prepared.reserve_record_count,
        "pruned_reserve_artifact_count": prepared.pruned_artifact_count,
    }


def _mm30k_population_and_loader(
    inputs: MM30KResearchInputs, cache_dir: Path
) -> Tuple[LocalQ1Population, MM30KMaskLoaderFactory]:
    population = rebuild_mm30k_population(
        mm_archive=inputs.mm_archive,
        historical_split=inputs.historical_split,
        selection_path=mm30k_selection_path(cache_dir),
    )
    return population, MM30KMaskLoaderFactory(inputs.mm_archive)


def build_mm30k_batch_plan(
    *,
    inputs: MM30KResearchInputs,
    cache_dir: Path,
    output_path: Path,
    representation: str = "multirun",
    max_local_tensor_elements: Optional[int] = None,
) -> Mapping[str, Any]:
    _require_launchable_representation(representation)
    population, loader_factory = _mm30k_population_and_loader(inputs, cache_dir)
    config = planning_config_with_local_tensor_bound(
        _planning_config_from_cache(cache_dir), max_local_tensor_elements
    )
    trust = automatic_cache_trust(population, cache_dir)
    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=loader_factory,
        output_dir=cache_dir,
        trust=trust,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
        replay_source_masks=False,
    )
    plan = freeze_local_q1_batch_plan(
        opened=opened,
        loader_factory=loader_factory,
        output_path=output_path,
        trust=trust,
        config=config,
    )
    return {
        "status": "mm30k_four_phase_batch_plan_complete",
        "representation": dict(representation_contract(representation)),
        "batch_plan": str(Path(output_path)),
        "batch_plan_file_sha256": plan.canonical_file_sha256,
        "batch_plan_content_sha256": plan.content_sha256,
        "phase_batch_counts": {
            phase: len(plan.phase_batches(phase))
            for phase in (
                "train",
                "validation_select",
                "validation_calibration",
                "validation_report",
            )
        },
    }


def train_mm30k_dual_and_dustbin(
    *,
    inputs: MM30KResearchInputs,
    cache_dir: Path,
    batch_plan: Path,
    output_root: Path,
    epochs: int,
    initialization_seed: int,
    representation: str = "multirun",
    max_local_tensor_elements: Optional[int] = None,
) -> Mapping[str, Any]:
    _require_launchable_representation(representation)
    population, loader_factory = _mm30k_population_and_loader(inputs, cache_dir)
    config = planning_config_with_local_tensor_bound(
        _planning_config_from_cache(cache_dir), max_local_tensor_elements
    )
    trust = automatic_cache_trust(population, cache_dir)
    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=loader_factory,
        output_dir=cache_dir,
        trust=trust,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
        replay_source_masks=False,
    )
    _plan_value, plan_file, plan_content = _load_receipt(batch_plan)
    external_locks = _external_locks(
        trust,
        config,
        plan_file_sha256=plan_file,
        plan_content_sha256=plan_content,
    )
    plan = reopen_local_q1_batch_plan(batch_plan, external_locks=external_locks)
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=loader_factory,
        plan=plan,
        geometry_config=config.geometry_batch_config,
        inventory_config=config.inventory_config,
        external_locks=external_locks,
    )
    contract = LocalQ1RunnerContract(
        epochs=epochs,
        initialization_seed=initialization_seed,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=True,
    )

    def backend_factory() -> LocalQ1Backend:
        return LocalQ1Backend(device="cuda", mode=LocalQ1BackendMode.FORMAL)

    artifacts = run_local_q1(
        contract=contract,
        plan=plan,
        train_records=population.training_records,
        validation_records=None,
        provider=provider,
        backend_factory=backend_factory,
        output_root=output_root,
    )
    return {
        "status": "mm30k_cuda_multirun_keypoint_four_arm_complete",
        "selection": str(mm30k_selection_path(cache_dir)),
        "run_directory": str(artifacts.run_directory),
        "receipt": str(artifacts.receipt_path),
        "representation": dict(representation_contract(representation)),
        "arms": list(RESEARCH_ARM_MATRIX[representation]),
    }


def run_mm30k_pipeline(
    *,
    inputs: MM30KResearchInputs,
    cache_dir: Path,
    batch_plan: Path,
    output_root: Path,
    producer_workers: int,
    epochs: int,
    initialization_seed: int,
    representation: str = "multirun",
    max_local_tensor_elements: Optional[int] = None,
) -> Mapping[str, Any]:
    """Run MM30K cache, plan, and the four local arms without Route-A inputs."""

    _require_launchable_representation(representation)
    config = mm30k_planning_config(inputs.route_config)
    selection_path = mm30k_selection_path(cache_dir)
    prepared = prepare_mm30k_cache(
        mm_archive=inputs.mm_archive,
        historical_split=inputs.historical_split,
        output_dir=cache_dir,
        selection_path=selection_path,
        inventory_config=config.inventory_config,
        geometry_config=config.geometry_batch_config,
        cache_limits=config.cache_limits,
        producer_workers=producer_workers,
    )
    config_path = Path(cache_dir) / PLANNING_CONFIG_NAME
    write_route_a_planning_config(config_path, config)
    config = planning_config_with_local_tensor_bound(config, max_local_tensor_elements)
    population = prepared.population
    loader_factory = MM30KMaskLoaderFactory(inputs.mm_archive)
    trust = automatic_cache_trust(population, cache_dir)
    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=loader_factory,
        output_dir=cache_dir,
        trust=trust,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
        replay_source_masks=False,
    )
    plan = freeze_local_q1_batch_plan(
        opened=opened,
        loader_factory=loader_factory,
        output_path=batch_plan,
        trust=trust,
        config=config,
    )
    external_locks = _external_locks(
        trust,
        config,
        plan_file_sha256=plan.canonical_file_sha256,
        plan_content_sha256=plan.content_sha256,
    )
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=loader_factory,
        plan=plan,
        geometry_config=config.geometry_batch_config,
        inventory_config=config.inventory_config,
        external_locks=external_locks,
    )
    contract = LocalQ1RunnerContract(
        epochs=epochs,
        initialization_seed=initialization_seed,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=True,
    )

    def backend_factory() -> LocalQ1Backend:
        return LocalQ1Backend(device="cuda", mode=LocalQ1BackendMode.FORMAL)

    training = run_local_q1(
        contract=contract,
        plan=plan,
        train_records=population.training_records,
        validation_records=None,
        provider=provider,
        backend_factory=backend_factory,
        output_root=output_root,
    )
    return {
        "status": "mm30k_cache_plan_cuda_multirun_keypoint_four_arm_complete",
        "selection": str(selection_path),
        "training_count": len(population.training_records),
        "validation_count": len(population.validation_records),
        "cache_dir": str(Path(cache_dir)),
        "cache_artifact_count": prepared.cache_artifacts.cache_receipt[
            "artifact_count"
        ],
        "batch_plan": str(Path(batch_plan)),
        "phase_batch_counts": {
            phase: len(plan.phase_batches(phase))
            for phase in (
                "train",
                "validation_select",
                "validation_calibration",
                "validation_report",
            )
        },
        "run_directory": str(training.run_directory),
        "receipt": str(training.receipt_path),
        "representation": dict(representation_contract(representation)),
    }


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--population", choices=("route_a", "mm30k"), default="route_a")
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--mm-archive", type=Path, required=True)
    parser.add_argument("--historical-split", type=Path, required=True)
    parser.add_argument("--route-a-freeze", type=Path)
    parser.add_argument("--predecessor-freeze", type=Path)
    parser.add_argument("--eligibility-index", type=Path)
    parser.add_argument("--eligibility-receipt", type=Path)
    parser.add_argument("--route-policy", type=Path)
    for role in (
        "eccv_archive",
        "mm_fingerprint_cache",
        "eccv_fingerprint_cache",
        "synthetic_manifest",
        "synthetic_archive",
    ):
        parser.add_argument("--" + role.replace("_", "-"), type=Path)
    parser.add_argument("--mm-extracted-root", type=Path)
    parser.add_argument("--eccv-extracted-root", type=Path)
    parser.add_argument("--synthetic-extracted-root", type=Path)


def _inputs(args: argparse.Namespace) -> RouteAResearchInputs:
    required = (
        "route_a_freeze",
        "predecessor_freeze",
        "eligibility_index",
        "eligibility_receipt",
        "route_policy",
        "eccv_archive",
        "mm_fingerprint_cache",
        "eccv_fingerprint_cache",
        "synthetic_manifest",
        "synthetic_archive",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise RouteAResearchError(
            "--population route_a requires: "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    return RouteAResearchInputs(
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


def _mm30k_inputs(args: argparse.Namespace) -> MM30KResearchInputs:
    return MM30KResearchInputs(
        route_config=args.route_config,
        mm_archive=args.mm_archive,
        historical_split=args.historical_split,
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _add_local_tensor_bound(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-local-tensor-elements",
        type=_positive_int,
        help=(
            "shrink the in-memory batch-packing bound; plan and train must use "
            "the same value"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", allow_abbrev=False)
    _add_inputs(cache)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--producer-workers", type=int, default=16)
    plan = commands.add_parser("plan", allow_abbrev=False)
    _add_inputs(plan)
    plan.add_argument("--cache-dir", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument(
        "--representation", choices=tuple(RESEARCH_ARM_MATRIX), default="multirun"
    )
    _add_local_tensor_bound(plan)
    train = commands.add_parser("train", allow_abbrev=False)
    _add_inputs(train)
    train.add_argument("--cache-dir", type=Path, required=True)
    train.add_argument("--batch-plan", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--initialization-seed", type=int, default=260829)
    train.add_argument(
        "--representation", choices=tuple(RESEARCH_ARM_MATRIX), default="multirun"
    )
    _add_local_tensor_bound(train)
    resume = commands.add_parser("resume", allow_abbrev=False)
    _add_inputs(resume)
    resume.add_argument("--cache-dir", type=Path, required=True)
    resume.add_argument("--batch-plan", type=Path, required=True)
    resume.add_argument("--output-root", type=Path, required=True)
    resume.add_argument("--epochs", type=int, default=5)
    resume.add_argument("--initialization-seed", type=int, default=260829)
    resume.add_argument(
        "--representation", choices=tuple(RESEARCH_ARM_MATRIX), default="multirun"
    )
    _add_local_tensor_bound(resume)
    all_stages = commands.add_parser("all", allow_abbrev=False)
    _add_inputs(all_stages)
    all_stages.add_argument("--cache-dir", type=Path, required=True)
    all_stages.add_argument("--batch-plan", type=Path, required=True)
    all_stages.add_argument("--output-root", type=Path, required=True)
    all_stages.add_argument("--producer-workers", type=int, default=16)
    all_stages.add_argument("--epochs", type=int, default=5)
    all_stages.add_argument("--initialization-seed", type=int, default=260829)
    all_stages.add_argument(
        "--representation", choices=tuple(RESEARCH_ARM_MATRIX), default="multirun"
    )
    _add_local_tensor_bound(all_stages)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    inputs = _mm30k_inputs(args) if args.population == "mm30k" else _inputs(args)
    if args.command == "cache":
        if isinstance(inputs, MM30KResearchInputs):
            result = build_mm30k_cache(
                inputs=inputs,
                output_dir=args.output_dir,
                producer_workers=args.producer_workers,
            )
        else:
            result = build_route_a_cache(
                inputs=inputs,
                output_dir=args.output_dir,
                producer_workers=args.producer_workers,
            )
    elif args.command == "plan":
        if isinstance(inputs, MM30KResearchInputs):
            result = build_mm30k_batch_plan(
                inputs=inputs,
                cache_dir=args.cache_dir,
                output_path=args.output,
                representation=args.representation,
                max_local_tensor_elements=args.max_local_tensor_elements,
            )
        else:
            result = build_route_a_batch_plan(
                inputs=inputs,
                cache_dir=args.cache_dir,
                output_path=args.output,
                representation=args.representation,
                max_local_tensor_elements=args.max_local_tensor_elements,
            )
    elif args.command == "train":
        if isinstance(inputs, MM30KResearchInputs):
            result = train_mm30k_dual_and_dustbin(
                inputs=inputs,
                cache_dir=args.cache_dir,
                batch_plan=args.batch_plan,
                output_root=args.output_root,
                epochs=args.epochs,
                initialization_seed=args.initialization_seed,
                representation=args.representation,
                max_local_tensor_elements=args.max_local_tensor_elements,
            )
        else:
            result = train_route_a_dual_and_dustbin(
                inputs=inputs,
                cache_dir=args.cache_dir,
                batch_plan=args.batch_plan,
                output_root=args.output_root,
                epochs=args.epochs,
                initialization_seed=args.initialization_seed,
                representation=args.representation,
                max_local_tensor_elements=args.max_local_tensor_elements,
            )
    elif args.command == "resume":
        if isinstance(inputs, MM30KResearchInputs):
            raise RouteAResearchError(
                "resume currently supports --population route_a only"
            )
        result = resume_route_a_plan_and_train(
            inputs=inputs,
            cache_dir=args.cache_dir,
            batch_plan=args.batch_plan,
            output_root=args.output_root,
            epochs=args.epochs,
            initialization_seed=args.initialization_seed,
            representation=args.representation,
            max_local_tensor_elements=args.max_local_tensor_elements,
        )
    else:
        if isinstance(inputs, MM30KResearchInputs):
            result = run_mm30k_pipeline(
                inputs=inputs,
                cache_dir=args.cache_dir,
                batch_plan=args.batch_plan,
                output_root=args.output_root,
                producer_workers=args.producer_workers,
                epochs=args.epochs,
                initialization_seed=args.initialization_seed,
                representation=args.representation,
                max_local_tensor_elements=args.max_local_tensor_elements,
            )
        else:
            result = run_route_a_pipeline(
                inputs=inputs,
                cache_dir=args.cache_dir,
                batch_plan=args.batch_plan,
                output_root=args.output_root,
                producer_workers=args.producer_workers,
                epochs=args.epochs,
                initialization_seed=args.initialization_seed,
                representation=args.representation,
                max_local_tensor_elements=args.max_local_tensor_elements,
            )
    print(_canonical_json(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PLANNING_CONFIG_NAME",
    "RESEARCH_ARM_MATRIX",
    "RESEARCH_BRIDGE_VERSION",
    "MM30KResearchInputs",
    "RouteAResearchError",
    "RouteAResearchInputs",
    "automatic_cache_trust",
    "build_route_a_batch_plan",
    "build_route_a_cache",
    "build_mm30k_batch_plan",
    "build_mm30k_cache",
    "main",
    "planning_config_with_local_tensor_bound",
    "rebuild_route_a_population",
    "research_loader_factory",
    "representation_contract",
    "resume_route_a_plan_and_train",
    "run_mm30k_pipeline",
    "run_route_a_pipeline",
    "route_a_planning_config",
    "mm30k_planning_config",
    "train_mm30k_dual_and_dustbin",
    "train_route_a_dual_and_dustbin",
    "write_route_a_planning_config",
]

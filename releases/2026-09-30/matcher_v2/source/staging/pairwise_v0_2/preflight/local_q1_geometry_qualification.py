"""Externally bound Route-A geometry eligibility and lazy qualifier support.

The canonical eligibility *index* stores only opaque pair commitments and
supervision-blind geometry/resource outcomes.  A separate portable receipt
binds that index to the predecessor population, all eight input roles, the
source bundle, frozen geometry/planning configs, and the corrected Route-A
policy.  Loading requires caller-supplied file/content digests for both files;
neither document is allowed to authorize itself.

The real qualification run is intentionally not performed by this module at
import time.  ``LazyLocalQ1GeometryQualifier`` can be driven by the complete
metadata frontier and only decodes/builds geometry for the exact prefix that
precedes each selected winner.  It receives endpoint references but never a
TrainingPairRecord, label, direction, label origin, or hard-negative field.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from staging.pairwise_v0_2.geometry import (
    CandidateBuilderConfig,
    CorrosionConfig,
    geometry_config_fingerprint,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import MaskMemberRef
from staging.pairwise_v0_2.training.fragment_geometry_cache import (
    FragmentGeometryCacheLookup,
    load_or_build_fragment_geometry,
)
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.geometry_cache import (
    GeometryArtifactCache,
    GeometryCacheLimits,
)
from staging.pairwise_v0_2.training.local_q1_pair_qualification import (
    qualification_contract,
    qualify_local_q1_pair_geometry,
)
from staging.pairwise_v0_2.training.local_cache_inventory import (
    LocalCacheInventoryConfig,
)


LOCAL_Q1_ELIGIBILITY_INDEX_SCHEMA_VERSION = (
    "dunhuang-local-q1-geometry-eligibility-index/0.1"
)
LOCAL_Q1_ELIGIBILITY_RECEIPT_SCHEMA_VERSION = (
    "dunhuang-local-q1-geometry-eligibility-receipt/0.1"
)
ROUTE_A_POLICY_SCHEMA_VERSION = "dunhuang-local-q1-refreeze-route-decision/0.2"
ROUTE_A_POLICY_FILE_SHA256 = (
    "3cf8781b65ad0d49fd12f47bb8c8f02f2052ee9f11b178b9c1e3001d3e52b5e3"
)
ROUTE_A_POLICY_CONTENT_SHA256 = (
    "be35c86a2504c4904106b0421f1456f8a7d64844d2135f084075107c4423543a"
)
CANONICAL_ROUTE_A_POLICY_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments/local_q1_refreeze_decision/route_a_policy_v0_2.json"
)
ROUTE_A_CONFIG_AUTHORITY_SCHEMA_VERSION = (
    "dunhuang-local-q1-route-a-config-authority/0.1"
)
ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256 = (
    "a7b238d7913be08d988277ba98dd21c170669e97002f28afe1b9112bae802379"
)
ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256 = (
    "200f8bfeadf957d2eec5e75d1448609433fab5a33fc33aa69723d4ae97b37041"
)
ROUTE_A_GEOMETRY_BATCH_CONFIG_SHA256 = (
    "dc994bf15596eea1b3d9245ac41505b1e72739eb49a91fef7a02012c744dcfd3"
)
ROUTE_A_PLANNING_GUARD_CONFIG_SHA256 = (
    "60d3f585f6f5ac27fdb3d2dbad29b8f52c9adb1d1955e88bfaa956ba51798d56"
)
ROUTE_A_INVENTORY_CONFIG_SHA256 = (
    "0270d5b7d7f042943cd52be28d6665ada11a037880fe81abf38f6d7fd008c123"
)
ROUTE_A_BATCH_PLAN_CONFIG_SHA256 = (
    "c27d3bc0629f65f33cecfb339b4b29e241a7f1c84c35e06d91eb140a9b4476b3"
)
ROUTE_A_CANDIDATE_BUILDER_CONFIG_SHA256 = (
    "1c5b6e4352bdbf686106e705989a734aa274d8fbff4b75854ffe092572fb7db9"
)
CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments/local_q1_refreeze_decision/route_a_config_authority_v0_1.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INPUT_ROLE_NAMES = frozenset(
    {
        "freeze_receipt",
        "mm_archive",
        "eccv_archive",
        "mm_fingerprint_cache",
        "eccv_fingerprint_cache",
        "historical_split",
        "synthetic_manifest",
        "synthetic_archive",
    }
)
_FORBIDDEN_PORTABLE_KEYS = frozenset(
    {
        "absolute_path",
        "archive_member",
        "canonical_group_id",
        "component_id",
        "fragment_id",
        "group_id",
        "label",
        "local_path",
        "member_path",
        "pair_id",
        "path",
        "secret",
    }
)


def _snapshot_locked_regular_file(
    path: Path, expected_file_sha256: str, name: str
) -> bytes:
    _require_sha256(expected_file_sha256, name + " file hash")
    target = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = None
    try:
        descriptor = os.open(str(target), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise LocalQ1GeometryQualificationError(
                "{} is not a non-empty regular file".format(name)
            )
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise LocalQ1GeometryQualificationError(
                    "{} changed while being read".format(name)
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise LocalQ1GeometryQualificationError(
                "{} grew while being read".format(name)
            )
        after = os.fstat(descriptor)
        path_state = os.stat(str(target), follow_symlinks=False)
        if (
            not stat.S_ISREG(path_state.st_mode)
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino) != (path_state.st_dev, path_state.st_ino)
        ):
            raise LocalQ1GeometryQualificationError(
                "{} path or bytes changed during snapshot".format(name)
            )
        payload = b"".join(chunks)
    except OSError as exc:
        raise LocalQ1GeometryQualificationError(
            "cannot snapshot {} without following links".format(name)
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not hmac.compare_digest(_sha256(payload), expected_file_sha256):
        raise LocalQ1GeometryQualificationError(
            "{} external file hash mismatch".format(name)
        )
    return payload


@dataclass(frozen=True)
class LocalQ1RouteAConfigAuthority:
    """Typed, same-fd snapshot of the preregistered production configs."""

    document: Mapping[str, Any]
    geometry_config: GeometryBatchConfig
    inventory_config: LocalCacheInventoryConfig
    batch_plan_config: Mapping[str, Any]
    file_sha256: str = ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256
    content_sha256: str = ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256

    def __post_init__(self) -> None:
        if not isinstance(self.geometry_config, GeometryBatchConfig):
            raise TypeError("config authority geometry_config is invalid")
        if not isinstance(self.inventory_config, LocalCacheInventoryConfig):
            raise TypeError("config authority inventory_config is invalid")
        if (
            self.file_sha256 != ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256
            or self.content_sha256 != ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256
        ):
            raise LocalQ1GeometryQualificationError(
                "Route-A config authority external locks changed"
            )
        frozen_document = _deep_freeze(
            _portable(dict(self.document), "route_a_config_authority")
        )
        if _content_sha256(frozen_document, "Route-A config authority") != (
            self.content_sha256
        ):
            raise LocalQ1GeometryQualificationError(
                "Route-A config authority changed before deep freeze"
            )
        object.__setattr__(self, "document", frozen_document)
        object.__setattr__(
            self,
            "batch_plan_config",
            _deep_freeze(dict(self.batch_plan_config)),
        )


def _exact_dataclass_dict(value: Any, cls: Any, name: str) -> Dict[str, Any]:
    expected = {field.name for field in fields(cls)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LocalQ1GeometryQualificationError(
            "{} must contain every {} field exactly".format(name, cls.__name__)
        )
    return dict(value)


def _candidate_config_from_authority(value: Any) -> CandidateBuilderConfig:
    values = _exact_dataclass_dict(
        value, CandidateBuilderConfig, "Route-A candidate config"
    )
    corrosion = _exact_dataclass_dict(
        values["corrosion"], CorrosionConfig, "Route-A corrosion config"
    )
    values["corrosion"] = CorrosionConfig(**corrosion)
    values["window_scale_fractions"] = tuple(values["window_scale_fractions"])
    values["output_size"] = tuple(values["output_size"])
    try:
        return CandidateBuilderConfig(**values)
    except (TypeError, ValueError) as exc:
        raise LocalQ1GeometryQualificationError(
            "Route-A candidate config is invalid"
        ) from exc


def load_local_q1_route_a_config_authority_payload(
    *,
    payload: bytes,
    expected_file_sha256: str = ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256,
    expected_content_sha256: str = ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256,
) -> LocalQ1RouteAConfigAuthority:
    """Validate captured immutable bytes and recompute every config commitment."""

    contract_authority = load_local_q1_route_a_config_authority_payload_contract(
        payload=payload,
        expected_file_sha256=expected_file_sha256,
        expected_content_sha256=expected_content_sha256,
    )
    if (
        expected_file_sha256 != ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256
        or expected_content_sha256 != ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256
    ):
        raise LocalQ1GeometryQualificationError(
            "only the frozen Route-A config authority may be loaded"
        )
    if type(payload) is not bytes:  # noqa: E721
        raise LocalQ1GeometryQualificationError(
            "Route-A config authority payload must be immutable bytes"
        )
    if not hmac.compare_digest(_sha256(payload), expected_file_sha256):
        raise LocalQ1GeometryQualificationError(
            "Route-A config authority external file hash mismatch"
        )
    document = _parse_strict_json(payload, "Route-A config authority")
    expected_root = {
        "schema_version",
        "status",
        "scope",
        "authority_contract",
        "hash_contract",
        "route_a_policy_lock",
        "geometry_batch_config",
        "geometry_batch_config_sha256",
        "planning_guard_config",
        "planning_guard_config_sha256",
        "component_config_hashes",
        "production_population_contract",
        "content_sha256",
    }
    if (
        set(document) != expected_root
        or document.get("schema_version") != ROUTE_A_CONFIG_AUTHORITY_SCHEMA_VERSION
        or document.get("status")
        != "preregistered_exact_configs_and_population_before_real_feasibility_result"
        or _content_sha256(document, "Route-A config authority")
        != expected_content_sha256
    ):
        raise LocalQ1GeometryQualificationError(
            "Route-A config authority schema/status/content changed"
        )
    if document.get("scope") != {
        "decision_made_before_real_route_a_feasibility_result": True,
        "historical_test_pair_stream_read": False,
        "mask_pixels_decoded": False,
        "model_or_gpu_used": False,
        "sealed_real_read": False,
    } or document.get("authority_contract") != {
        "config_values_caller_substitutable": False,
        "fixture_scale_positive_values_authorize_real_scan": False,
        "production_driver_must_bind_this_file_and_content_sha256_externally": True,
        "production_driver_must_recompute_all_component_and_combined_hashes": True,
    }:
        raise LocalQ1GeometryQualificationError(
            "Route-A config authority scope/contract changed"
        )
    if document.get("hash_contract") != {
        "algorithm": "sha256",
        "canonicalization": (
            "json_utf8_ensure_ascii_false_sort_keys_true_separators_comma_colon_"
            "allow_nan_false_no_trailing_newline"
        ),
        "excluded_top_level_fields": ["content_sha256"],
    } or document.get("route_a_policy_lock") != {
        "schema_version": ROUTE_A_POLICY_SCHEMA_VERSION,
        "file_sha256": ROUTE_A_POLICY_FILE_SHA256,
        "content_sha256": ROUTE_A_POLICY_CONTENT_SHA256,
    }:
        raise LocalQ1GeometryQualificationError(
            "Route-A config authority hash/policy contract changed"
        )

    geometry_receipt = document.get("geometry_batch_config")
    planning = document.get("planning_guard_config")
    if not isinstance(geometry_receipt, Mapping) or not isinstance(planning, Mapping):
        raise LocalQ1GeometryQualificationError(
            "Route-A config authority embedded configs are missing"
        )
    try:
        candidate_config = _candidate_config_from_authority(
            geometry_receipt["geometry"]
        )
        bounds = geometry_receipt["bounds"]
        expected_bounds = {
            "max_batch_size",
            "max_input_pixels_per_mask",
            "max_candidates_per_sample",
            "max_candidates_per_batch",
            "max_sequence_length",
            "max_local_tensor_elements",
            "max_attention_score_elements_per_candidate",
            "max_attention_score_elements_per_batch",
            "max_affinity_elements_per_candidate",
            "max_affinity_elements_per_batch",
            "max_sinkhorn_elements_per_candidate",
            "max_sinkhorn_elements_per_batch",
            "max_candidate_id_bytes",
        }
        if not isinstance(bounds, Mapping) or set(bounds) != expected_bounds:
            raise LocalQ1GeometryQualificationError("Route-A geometry bounds changed")
        geometry_config = GeometryBatchConfig(
            geometry=candidate_config,
            coarse_output_size=tuple(geometry_receipt["coarse_output_size"]),
            coarse_resize_mode=geometry_receipt["coarse_resize_mode"],
            coarse_preprocess_mode=geometry_receipt["coarse_preprocess_mode"],
            coarse_content_fraction=geometry_receipt["coarse_content_fraction"],
            coarse_component_connectivity=geometry_receipt[
                "coarse_component_connectivity"
            ],
            **dict(bounds),
        )
        if _canonical_json(geometry_config.provenance_dict()) != _canonical_json(
            geometry_receipt
        ):
            raise LocalQ1GeometryQualificationError(
                "Route-A geometry typed reconstruction changed"
            )

        if set(planning) != {"inventory_config", "batch_plan_config"}:
            raise LocalQ1GeometryQualificationError(
                "Route-A planning guard schema changed"
            )
        inventory_values = _exact_dataclass_dict(
            planning["inventory_config"],
            LocalCacheInventoryConfig,
            "Route-A inventory config",
        )
        inventory_values["geometry"] = _candidate_config_from_authority(
            inventory_values["geometry"]
        )
        inventory_values["sequence_length_bucket_edges"] = tuple(
            inventory_values["sequence_length_bucket_edges"]
        )
        inventory_values["candidate_count_bucket_edges"] = tuple(
            inventory_values["candidate_count_bucket_edges"]
        )
        inventory_config = LocalCacheInventoryConfig(**inventory_values)
        batch_plan_config = planning["batch_plan_config"]
        if (
            not isinstance(batch_plan_config, Mapping)
            or set(batch_plan_config)
            != {
                "packing_max_records",
                "attention_head_count",
                "score_element_bytes",
                "planning_elements_per_second",
                "time_safety_factor",
            }
            or any(
                isinstance(batch_plan_config[name], bool)
                or not isinstance(batch_plan_config[name], int)
                or batch_plan_config[name] <= 0
                for name in (
                    "packing_max_records",
                    "attention_head_count",
                    "score_element_bytes",
                    "planning_elements_per_second",
                )
            )
            or isinstance(batch_plan_config["time_safety_factor"], bool)
            or not isinstance(batch_plan_config["time_safety_factor"], (int, float))
            or not math.isfinite(batch_plan_config["time_safety_factor"])
            or batch_plan_config["time_safety_factor"] < 1.0
        ):
            raise LocalQ1GeometryQualificationError(
                "Route-A batch-plan config is invalid"
            )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, LocalQ1GeometryQualificationError):
            raise
        raise LocalQ1GeometryQualificationError(
            "Route-A config typed reconstruction failed"
        ) from exc

    component_hashes = document.get("component_config_hashes")
    observed_geometry_hash = _sha256(_canonical_json(geometry_receipt))
    observed_planning_hash = _sha256(_canonical_json(planning))
    observed_inventory_hash = _sha256(_canonical_json(inventory_config.portable_dict()))
    observed_batch_plan_hash = _sha256(_canonical_json(batch_plan_config))
    observed_candidate_hash = geometry_config_fingerprint(candidate_config)
    if (
        observed_geometry_hash != ROUTE_A_GEOMETRY_BATCH_CONFIG_SHA256
        or geometry_config.fingerprint != observed_geometry_hash
        or document.get("geometry_batch_config_sha256") != observed_geometry_hash
        or observed_planning_hash != ROUTE_A_PLANNING_GUARD_CONFIG_SHA256
        or document.get("planning_guard_config_sha256") != observed_planning_hash
        or observed_inventory_hash != ROUTE_A_INVENTORY_CONFIG_SHA256
        or observed_batch_plan_hash != ROUTE_A_BATCH_PLAN_CONFIG_SHA256
        or observed_candidate_hash != ROUTE_A_CANDIDATE_BUILDER_CONFIG_SHA256
        or component_hashes
        != {
            "candidate_builder_config_sha256": observed_candidate_hash,
            "local_cache_inventory_config_sha256": observed_inventory_hash,
            "local_q1_batch_plan_config_sha256": observed_batch_plan_hash,
        }
        or inventory_config.geometry != candidate_config
    ):
        raise LocalQ1GeometryQualificationError(
            "Route-A embedded config commitment recomputation failed"
        )
    if document.get("production_population_contract") != {
        "selection_seed": "local-q1-260828",
        "training_dataset_order": [
            "mm_augmented",
            "eccv_1113data",
            "canonical_new",
        ],
        "training_label_order": ["negative", "positive"],
        "training_target_per_dataset_label": 512,
        "training_total": 3072,
        "validation_caps": {"mm_augmented": 32, "eccv_1113data": 4},
        "validation_component_counts": {
            "mm_augmented": 55,
            "eccv_1113data": 533,
        },
        "validation_total": 3892,
        "selected_pair_total": 6964,
        "required_all_four_direction_pair_count": 6964,
        "required_fragment_pair_direction_resource_failure_count": 0,
        "required_train_validation_component_member_content_overlap_count": 0,
    }:
        raise LocalQ1GeometryQualificationError(
            "Route-A production population contract changed"
        )
    authority = LocalQ1RouteAConfigAuthority(
        document=document,
        geometry_config=geometry_config,
        inventory_config=inventory_config,
        batch_plan_config=batch_plan_config,
    )
    if (
        _canonical_json(authority.document)
        != _canonical_json(contract_authority.document)
        or authority.geometry_config.fingerprint
        != contract_authority.geometry_batch_config_sha256
        or authority.document["planning_guard_config_sha256"]
        != contract_authority.planning_guard_config_sha256
        or authority.file_sha256 != contract_authority.file_sha256
        or authority.content_sha256 != contract_authority.content_sha256
    ):
        raise LocalQ1GeometryQualificationError(
            "typed and dependency-light Route-A config authorities differ"
        )
    return authority


def load_local_q1_route_a_config_authority(
    *,
    path: Path = CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH,
    expected_file_sha256: str = ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256,
    expected_content_sha256: str = ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256,
) -> LocalQ1RouteAConfigAuthority:
    """Snapshot a file once, then delegate to the immutable-payload validator."""

    payload = _snapshot_locked_regular_file(
        path, expected_file_sha256, "Route-A config authority"
    )
    return load_local_q1_route_a_config_authority_payload(
        payload=payload,
        expected_file_sha256=expected_file_sha256,
        expected_content_sha256=expected_content_sha256,
    )


def load_local_q1_geometry_eligibility_authority_payloads(
    *,
    index_payload: bytes,
    receipt_payload: bytes,
    trust: LocalQ1EligibilityTrust,
    expected_index_bytes: int,
    expected_index_file_sha256: str,
    expected_index_content_sha256: str,
    expected_receipt_bytes: int,
    expected_receipt_file_sha256: str,
    expected_receipt_content_sha256: str,
    policy_path: Path = CANONICAL_ROUTE_A_POLICY_PATH,
    config_authority_path: Path = CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH,
    _fixture_scale_test_only: bool = False,
) -> LocalQ1GeometryEligibilityAuthority:
    """Compatibility entry point for the original two-payload public API.

    New authority consumers must capture all four inputs themselves and call
    ``load_local_q1_geometry_eligibility_authority_payloads_v2`` instead.
    """

    if not isinstance(trust, LocalQ1EligibilityTrust):
        raise TypeError("trust must be LocalQ1EligibilityTrust")
    if type(_fixture_scale_test_only) is not bool:  # noqa: E721
        raise TypeError("_fixture_scale_test_only must be bool")
    config_authority_payload = _snapshot_locked_regular_file(
        config_authority_path,
        trust.config_authority_file_sha256,
        "Route-A config authority",
    )
    policy_payload = _snapshot_locked_regular_file(
        policy_path,
        trust.eligibility_policy_file_sha256,
        "corrected Route-A policy",
    )
    return load_local_q1_geometry_eligibility_authority_payloads_v2(
        index_payload=index_payload,
        receipt_payload=receipt_payload,
        policy_payload=policy_payload,
        config_authority_payload=config_authority_payload,
        trust=trust,
        expected_index_bytes=expected_index_bytes,
        expected_index_file_sha256=expected_index_file_sha256,
        expected_index_content_sha256=expected_index_content_sha256,
        expected_receipt_bytes=expected_receipt_bytes,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        expected_receipt_content_sha256=expected_receipt_content_sha256,
        expected_policy_bytes=len(policy_payload),
        expected_policy_file_sha256=trust.eligibility_policy_file_sha256,
        expected_policy_content_sha256=trust.eligibility_policy_content_sha256,
        expected_config_authority_bytes=len(config_authority_payload),
        expected_config_authority_file_sha256=trust.config_authority_file_sha256,
        expected_config_authority_content_sha256=(
            trust.config_authority_content_sha256
        ),
        _fixture_scale_test_only=_fixture_scale_test_only,
    )


def load_local_q1_geometry_eligibility_authority(
    *,
    index_path: Path,
    receipt_path: Path,
    trust: LocalQ1EligibilityTrust,
    policy_path: Path = CANONICAL_ROUTE_A_POLICY_PATH,
    config_authority_path: Path = CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH,
    _fixture_scale_test_only: bool = False,
) -> LocalQ1GeometryEligibilityAuthority:
    """Snapshot each eligibility file once, then validate those exact bytes."""

    if not isinstance(trust, LocalQ1EligibilityTrust):
        raise TypeError("trust must be LocalQ1EligibilityTrust")
    if type(_fixture_scale_test_only) is not bool:  # noqa: E721
        raise TypeError("_fixture_scale_test_only must be bool")
    config_authority_payload = _snapshot_locked_regular_file(
        config_authority_path,
        trust.config_authority_file_sha256,
        "Route-A config authority",
    )
    policy_payload = _snapshot_locked_regular_file(
        policy_path,
        trust.eligibility_policy_file_sha256,
        "corrected Route-A policy",
    )
    index_payload = _snapshot_locked_regular_file(
        index_path, trust.eligibility_index_file_sha256, "eligibility index"
    )
    receipt_payload = _snapshot_locked_regular_file(
        receipt_path, trust.eligibility_receipt_file_sha256, "eligibility receipt"
    )
    return load_local_q1_geometry_eligibility_authority_payloads_v2(
        index_payload=index_payload,
        receipt_payload=receipt_payload,
        policy_payload=policy_payload,
        config_authority_payload=config_authority_payload,
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
        expected_config_authority_bytes=len(config_authority_payload),
        expected_config_authority_file_sha256=trust.config_authority_file_sha256,
        expected_config_authority_content_sha256=(
            trust.config_authority_content_sha256
        ),
        _fixture_scale_test_only=_fixture_scale_test_only,
    )


class LazyLocalQ1GeometryQualifier:
    """Decode/cache fragments and assess only frontier pairs requested by caller."""

    def __init__(
        self,
        *,
        mask_loader: Callable[[MaskMemberRef], np.ndarray],
        cache_root: Path,
        geometry_config: GeometryBatchConfig,
        cache_limits: Optional[GeometryCacheLimits] = None,
        config_authority: Optional[LocalQ1RouteAConfigAuthority] = None,
    ) -> None:
        if not callable(mask_loader):
            raise TypeError("mask_loader must be callable")
        if not isinstance(geometry_config, GeometryBatchConfig):
            raise TypeError("geometry_config must be GeometryBatchConfig")
        if cache_limits is not None and not isinstance(
            cache_limits, GeometryCacheLimits
        ):
            raise TypeError("cache_limits must be GeometryCacheLimits")
        if config_authority is not None and not isinstance(
            config_authority, LocalQ1RouteAConfigAuthority
        ):
            raise TypeError("config_authority is invalid")
        if config_authority is not None and (
            geometry_config.fingerprint != config_authority.geometry_config.fingerprint
        ):
            raise LocalQ1GeometryQualificationError(
                "qualification geometry differs from config authority"
            )
        root = Path(cache_root)
        if root.exists() or root.is_symlink():
            raise LocalQ1GeometryQualificationError(
                "eligibility scratch cache must be a newly claimed path"
            )
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            root.mkdir()
        except FileExistsError as exc:
            raise LocalQ1GeometryQualificationError(
                "eligibility scratch cache path was concurrently claimed"
            ) from exc
        if root.is_symlink() or any(root.iterdir()):
            raise LocalQ1GeometryQualificationError(
                "new eligibility scratch cache root is not empty/regular"
            )
        cache = GeometryArtifactCache(root, limits=cache_limits)
        if any(root.rglob("*.npz")):
            raise LocalQ1GeometryQualificationError(
                "eligibility scratch cache contained a preexisting artifact"
            )
        self._loader = mask_loader
        self._cache = cache
        self._cache_root = root
        self._config = geometry_config
        self._config_authority = config_authority
        self._reference_memo = {}  # type: Dict[Tuple[str, ...], FragmentGeometryCacheLookup]
        self._reference_mask_pixels = {}  # type: Dict[Tuple[str, ...], int]
        self._decisions = {}  # type: Dict[str, LocalQ1EligibilityDecision]
        self.mask_decode_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self._sealed_evidence = None  # type: Optional[Mapping[str, Any]]

    @staticmethod
    def _reference_key(reference: MaskMemberRef) -> Tuple[str, ...]:
        return (
            reference.binding.logical_id,
            reference.binding.sha256,
            reference.archive_member,
            reference.threshold_rule,
        )

    def _fragment(self, reference: MaskMemberRef) -> FragmentGeometryCacheLookup:
        key = self._reference_key(reference)
        existing = self._reference_memo.get(key)
        if existing is not None:
            return existing
        value = np.asarray(self._loader(reference))
        self.mask_decode_count += 1
        if value.ndim != 2 or value.size == 0 or value.dtype != np.bool_:
            raise LocalQ1GeometryQualificationError(
                "qualification loader must return non-empty 2D bool masks"
            )
        if value.size > self._config.max_input_pixels_per_mask:
            raise LocalQ1GeometryQualificationError(
                "qualification mask exceeds frozen input-pixel bound"
            )
        lookup = load_or_build_fragment_geometry(
            np.ascontiguousarray(value, dtype=np.bool_),
            reference.threshold_rule,
            self._config.geometry,
            self._cache,
        )
        if lookup.cache_hit:
            self.cache_hit_count += 1
        else:
            self.cache_miss_count += 1
        self._reference_memo[key] = lookup
        self._reference_mask_pixels[key] = int(value.size)
        return lookup

    def __call__(self, item: LocalQ1PairGeometryInput) -> LocalQ1EligibilityDecision:
        if not isinstance(item, LocalQ1PairGeometryInput):
            raise TypeError("lazy qualifier accepts LocalQ1PairGeometryInput only")
        previous = self._decisions.get(item.pair_identity_sha256)
        if previous is not None:
            return previous
        if self._sealed_evidence is not None:
            raise LocalQ1GeometryQualificationError(
                "qualification scratch cache was already sealed"
            )
        first = self._fragment(item.fragment_a)
        second = self._fragment(item.fragment_b)
        decision = LocalQ1EligibilityDecision(
            pair_identity_sha256=item.pair_identity_sha256,
            assessment=qualify_local_q1_pair_geometry(
                first.result,
                second.result,
                pair_id=item.pair_id,
                geometry_config=self._config,
                require_candidates=True,
            ),
        )
        self._decisions[item.pair_identity_sha256] = decision
        return decision

    @property
    def decisions(self) -> Tuple[LocalQ1EligibilityDecision, ...]:
        return tuple(self._decisions.values())

    def execution_evidence(
        self, *, selected_inputs: Sequence[LocalQ1PairGeometryInput]
    ) -> Mapping[str, Any]:
        """Return identity-free evidence for the externally locked receipt."""

        if self._sealed_evidence is not None:
            return self._sealed_evidence

        if self._config_authority is None:
            raise LocalQ1GeometryQualificationError(
                "selected-population evidence requires the config authority"
            )
        selected = tuple(selected_inputs)
        if not selected or any(
            not isinstance(item, LocalQ1PairGeometryInput) for item in selected
        ):
            raise LocalQ1GeometryQualificationError(
                "selected geometry inputs are required before cache sealing"
            )
        selected_tokens = [item.pair_identity_sha256 for item in selected]
        if len(selected_tokens) != len(set(selected_tokens)):
            raise LocalQ1GeometryQualificationError(
                "selected population contains duplicate pair commitments"
            )
        selected_reference_keys = set()
        identity_splits = {}  # type: Dict[str, set]
        selected_max_mask_pixels = 0
        for item in selected:
            decision = self._decisions.get(item.pair_identity_sha256)
            if decision is None or not decision.assessment.eligible:
                raise LocalQ1GeometryQualificationError(
                    "selected pair lacks an eligible geometry decision"
                )
            for reference in (item.fragment_a, item.fragment_b):
                reference_key = self._reference_key(reference)
                lookup = self._reference_memo.get(reference_key)
                pixels = self._reference_mask_pixels.get(reference_key)
                if lookup is None or pixels is None:
                    raise LocalQ1GeometryQualificationError(
                        "selected reference was not decoded by the lazy frontier"
                    )
                selected_reference_keys.add(reference_key)
                selected_max_mask_pixels = max(selected_max_mask_pixels, pixels)
                identity_splits.setdefault(lookup.identity.key, set()).add(
                    reference.split
                )
        train_val_overlap = sum(
            1
            for splits in identity_splits.values()
            if "train" in splits and "val" in splits
        )
        inventory_config = self._config_authority.inventory_config
        if (
            len(selected) > inventory_config.max_records
            or len(selected_reference_keys) > inventory_config.max_unique_references
            or selected_max_mask_pixels > inventory_config.max_mask_pixels
            or (
                inventory_config.require_train_val_fragment_disjoint
                and train_val_overlap
            )
        ):
            raise LocalQ1GeometryQualificationError(
                "selected population exceeds preregistered inventory guards"
            )

        artifact_rows = []
        total_file_bytes = 0
        try:
            with self._cache._commit_lock():  # serialize against cache commits
                root_entries = sorted(self._cache_root.iterdir(), key=lambda p: p.name)
                for entry in root_entries:
                    state = entry.lstat()
                    if entry.name in {".commit.lock", ".quota-ledger.json"}:
                        if (
                            not stat.S_ISREG(state.st_mode)
                            or stat.S_ISLNK(state.st_mode)
                            or state.st_nlink != 1
                        ):
                            raise LocalQ1GeometryQualificationError(
                                "qualification cache control file is not private/regular"
                            )
                        continue
                    if entry.name == ".key-locks":
                        if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(
                            state.st_mode
                        ):
                            raise LocalQ1GeometryQualificationError(
                                "qualification cache key-lock root is not regular"
                            )
                        for lock_path in sorted(
                            entry.iterdir(), key=lambda path: path.name
                        ):
                            lock_state = lock_path.lstat()
                            if (
                                not re.fullmatch(r"[0-9a-f]{2}\.lock", lock_path.name)
                                or not stat.S_ISREG(lock_state.st_mode)
                                or stat.S_ISLNK(lock_state.st_mode)
                                or lock_state.st_nlink != 1
                            ):
                                raise LocalQ1GeometryQualificationError(
                                    "qualification cache key-lock file is non-canonical"
                                )
                        continue
                    if (
                        not re.fullmatch(r"[0-9a-f]{2}", entry.name)
                        or not stat.S_ISDIR(state.st_mode)
                        or stat.S_ISLNK(state.st_mode)
                    ):
                        raise LocalQ1GeometryQualificationError(
                            "qualification cache contains a non-canonical entry"
                        )
                    for artifact_path in sorted(entry.iterdir(), key=lambda p: p.name):
                        key = artifact_path.stem
                        if (
                            not _SHA256_RE.fullmatch(key)
                            or artifact_path.name != key + ".npz"
                            or entry.name != key[:2]
                        ):
                            raise LocalQ1GeometryQualificationError(
                                "qualification cache artifact path is non-canonical"
                            )
                        flags = os.O_RDONLY
                        if hasattr(os, "O_CLOEXEC"):
                            flags |= os.O_CLOEXEC
                        if hasattr(os, "O_NOFOLLOW"):
                            flags |= os.O_NOFOLLOW
                        descriptor = os.open(str(artifact_path), flags)
                        try:
                            before = os.fstat(descriptor)
                            if (
                                not stat.S_ISREG(before.st_mode)
                                or before.st_nlink != 1
                                or before.st_size <= 0
                                or before.st_size
                                > self._cache.limits.max_artifact_file_bytes
                            ):
                                raise LocalQ1GeometryQualificationError(
                                    "qualification cache artifact is not private/bounded"
                                )
                            digest = hashlib.sha256()
                            remaining = before.st_size
                            while remaining:
                                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                                if not chunk:
                                    raise LocalQ1GeometryQualificationError(
                                        "qualification cache artifact changed while read"
                                    )
                                digest.update(chunk)
                                remaining -= len(chunk)
                            if os.read(descriptor, 1):
                                raise LocalQ1GeometryQualificationError(
                                    "qualification cache artifact grew while read"
                                )
                            after = os.fstat(descriptor)
                            path_state = artifact_path.lstat()
                            if (
                                before.st_dev,
                                before.st_ino,
                                before.st_size,
                                before.st_mtime_ns,
                            ) != (
                                after.st_dev,
                                after.st_ino,
                                after.st_size,
                                after.st_mtime_ns,
                            ) or (after.st_dev, after.st_ino) != (
                                path_state.st_dev,
                                path_state.st_ino,
                            ):
                                raise LocalQ1GeometryQualificationError(
                                    "qualification cache artifact/path changed during seal"
                                )
                            artifact_rows.append(
                                [key, before.st_size, digest.hexdigest()]
                            )
                            total_file_bytes += before.st_size
                        finally:
                            os.close(descriptor)
                if total_file_bytes > self._cache.limits.max_cache_file_bytes:
                    raise LocalQ1GeometryQualificationError(
                        "qualification cache inventory exceeds total byte bound"
                    )
        except OSError as exc:
            raise LocalQ1GeometryQualificationError(
                "cannot seal qualification scratch-cache inventory"
            ) from exc

        if (
            self.cache_hit_count + self.cache_miss_count != self.mask_decode_count
            or len(artifact_rows) != self.cache_miss_count
        ):
            raise LocalQ1GeometryQualificationError(
                "qualification scratch-cache execution arithmetic changed"
            )
        evidence = {
            "cache_role": "new_local_scratch_output_never_membership_input",
            "fresh_path_atomically_claimed": True,
            "initial_artifact_count": 0,
            "external_preexisting_artifact_count": 0,
            "predecessor_partial_cache_role_count": 0,
            "predecessor_partial_cache_tree_accessed": False,
            "mask_decode_count": self.mask_decode_count,
            "scratch_cache_hit_count": self.cache_hit_count,
            "scratch_cache_miss_count": self.cache_miss_count,
            "final_artifact_count": len(artifact_rows),
            "final_artifact_file_bytes": total_file_bytes,
            "scratch_cache_artifact_set_sha256": _sha256(
                _canonical_json(artifact_rows)
            ),
            "selected_population_guard": {
                "inventory_config_sha256": ROUTE_A_INVENTORY_CONFIG_SHA256,
                "selected_record_count": len(selected),
                "selected_pair_commitment_order_sha256": _commit_order(selected_tokens),
                "selected_pair_commitment_set_sha256": _commit_set(selected_tokens),
                "selected_unique_reference_count": len(selected_reference_keys),
                "selected_max_mask_pixels_per_reference": selected_max_mask_pixels,
                "max_records_bound": inventory_config.max_records,
                "max_unique_references_bound": (inventory_config.max_unique_references),
                "max_mask_pixels_per_reference_bound": (
                    inventory_config.max_mask_pixels
                ),
                "train_val_canonical_fragment_overlap_count": train_val_overlap,
                "all_selected_decisions_eligible": True,
                "additional_mask_decode_count": 0,
                "path_or_fragment_identifiers_present": False,
            },
            "decision_count": len(self._decisions),
            "path_or_fragment_identifiers_present": False,
        }
        self._sealed_evidence = _deep_freeze(evidence)
        return self._sealed_evidence


def build_untrusted_eligibility_index(
    decisions: Sequence[LocalQ1EligibilityDecision],
    *,
    geometry_config: GeometryBatchConfig,
) -> Mapping[str, Any]:
    """Build canonical index content; a later external authority must hash-lock it."""

    ordered = tuple(decisions)
    tokens = [decision.pair_identity_sha256 for decision in ordered]
    if len(tokens) != len(set(tokens)):
        raise LocalQ1GeometryQualificationError("eligibility decisions are duplicated")
    index: Dict[str, Any] = {
        "schema_version": LOCAL_Q1_ELIGIBILITY_INDEX_SCHEMA_VERSION,
        "status": "complete_lazy_frontier_decisions_label_direction_blind",
        "qualification_contract": qualification_contract(geometry_config),
        "decision_count": len(ordered),
        "decision_order_sha256": _sha256(_canonical_json(tokens)),
        "decision_set_sha256": _sha256(_canonical_json(sorted(tokens))),
        "decisions": [decision.portable_dict() for decision in ordered],
    }
    index["content_sha256"] = _sha256(_canonical_json(index))
    return _portable(index, "eligibility_index")


def build_untrusted_eligibility_receipt(
    *,
    index: Mapping[str, Any],
    selection_proof: Mapping[str, Any],
    predecessor_freeze_file_sha256: str,
    predecessor_freeze_content_sha256: str,
    input_role_locks: Mapping[str, LocalQ1ExternalFileLock],
    source_bundle_manifest_sha256: str,
    config_authority: LocalQ1RouteAConfigAuthority,
    qualification_scratch_cache: Mapping[str, Any],
    _fixture_scale_test_only: bool = False,
) -> Mapping[str, Any]:
    """Build a portable receipt that remains untrusted until independently locked."""

    if not isinstance(config_authority, LocalQ1RouteAConfigAuthority):
        raise TypeError(
            "config_authority must be a loaded LocalQ1RouteAConfigAuthority"
        )
    if type(_fixture_scale_test_only) is not bool:  # noqa: E721
        raise TypeError("_fixture_scale_test_only must be bool")
    geometry_batch_config_sha256 = config_authority.geometry_config.fingerprint
    planning_guard_config_sha256 = config_authority.document[
        "planning_guard_config_sha256"
    ]
    for name, value in (
        ("predecessor freeze file", predecessor_freeze_file_sha256),
        ("predecessor freeze content", predecessor_freeze_content_sha256),
        ("source bundle manifest", source_bundle_manifest_sha256),
        ("geometry batch config", geometry_batch_config_sha256),
        ("planning guard config", planning_guard_config_sha256),
    ):
        _require_sha256(value, name)
    if not isinstance(input_role_locks, Mapping) or set(input_role_locks) != (
        _INPUT_ROLE_NAMES
    ):
        raise LocalQ1GeometryQualificationError(
            "untrusted receipt still requires all eight exact input roles"
        )
    role_locks = {}
    for role in sorted(_INPUT_ROLE_NAMES):
        lock = input_role_locks[role]
        if not isinstance(lock, LocalQ1ExternalFileLock):
            raise TypeError("input role locks must be LocalQ1ExternalFileLock")
        role_locks[role] = lock.portable_dict()
    if role_locks["freeze_receipt"]["sha256"] != (predecessor_freeze_file_sha256):
        raise LocalQ1GeometryQualificationError(
            "predecessor freeze and input role lock disagree"
        )
    portable_index = _portable(dict(index), "eligibility_index")
    if portable_index.get("schema_version") != (
        LOCAL_Q1_ELIGIBILITY_INDEX_SCHEMA_VERSION
    ):
        raise LocalQ1GeometryQualificationError("eligibility index version changed")
    if _content_sha256(portable_index, "eligibility index") != portable_index.get(
        "content_sha256"
    ):
        raise LocalQ1GeometryQualificationError("eligibility index content changed")
    _validate_qualification_contract(
        portable_index.get("qualification_contract"),
        expected_geometry_config_sha256=geometry_batch_config_sha256,
    )
    decisions = {}
    order = []
    for row in portable_index.get("decisions", []):
        if not isinstance(row, Mapping):
            raise LocalQ1GeometryQualificationError("eligibility decision is invalid")
        token = _require_sha256(row.get("pair_identity_sha256"), "decision token")
        if token in decisions:
            raise LocalQ1GeometryQualificationError(
                "eligibility decision is duplicated"
            )
        decisions[token] = LocalQ1EligibilityDecision(
            pair_identity_sha256=token,
            assessment=_parse_assessment(row.get("assessment")),
        )
        order.append(token)
    if (
        portable_index.get("decision_count") != len(order)
        or portable_index.get("decision_order_sha256")
        != _sha256(_canonical_json(order))
        or portable_index.get("decision_set_sha256")
        != _sha256(_canonical_json(sorted(order)))
    ):
        raise LocalQ1GeometryQualificationError(
            "eligibility index decision commitments changed"
        )
    portable_proof = _portable(dict(selection_proof), "selection_proof")
    _validate_selection_proof(
        portable_proof,
        decision_count=len(order),
        require_production=not _fixture_scale_test_only,
    )
    stage = Counter(
        decision.assessment.failure_stage or "eligible"
        for decision in decisions.values()
    )
    stage_reason = Counter(
        "{}|{}".format(
            decision.assessment.failure_stage or "eligible",
            decision.assessment.failure_reason or "eligible",
        )
        for decision in decisions.values()
    )
    eligible_count = sum(
        decision.assessment.eligible for decision in decisions.values()
    )
    decision_census = {
        "decision_count": len(decisions),
        "eligible_count": eligible_count,
        "ineligible_count": len(decisions) - eligible_count,
        "by_failure_stage": dict(sorted(stage.items())),
        "by_failure_stage_reason": dict(sorted(stage_reason.items())),
    }
    scratch_evidence = _portable(
        dict(qualification_scratch_cache), "qualification_scratch_cache"
    )
    _validate_scratch_cache_evidence(
        scratch_evidence,
        decision_count=len(decisions),
        selected_record_count=(
            portable_proof["selected_training_count"]
            + portable_proof["selected_validation_count"]
        ),
        selected_pair_commitment_order_sha256=portable_proof[
            "selected_pair_commitment_order_sha256"
        ],
        selected_pair_commitment_set_sha256=portable_proof[
            "selected_pair_commitment_set_sha256"
        ],
    )
    index_bytes = _canonical_json(portable_index)
    receipt: Dict[str, Any] = {
        "schema_version": LOCAL_Q1_ELIGIBILITY_RECEIPT_SCHEMA_VERSION,
        "status": (
            "fixture_scale_test_only_not_authorized_for_real_scan"
            if _fixture_scale_test_only
            else "qualified_lazy_complete_frontier_no_model_no_test"
        ),
        "external_locks": {
            "predecessor_freeze_file_sha256": predecessor_freeze_file_sha256,
            "predecessor_freeze_content_sha256": predecessor_freeze_content_sha256,
            "input_role_locks": role_locks,
            "source_bundle_manifest_sha256": source_bundle_manifest_sha256,
            "geometry_batch_config_sha256": geometry_batch_config_sha256,
            "planning_guard_config_sha256": planning_guard_config_sha256,
            "eligibility_policy_file_sha256": ROUTE_A_POLICY_FILE_SHA256,
            "eligibility_policy_content_sha256": ROUTE_A_POLICY_CONTENT_SHA256,
            "config_authority_file_sha256": config_authority.file_sha256,
            "config_authority_content_sha256": config_authority.content_sha256,
        },
        "index_lock": {
            "bytes": len(index_bytes),
            "file_sha256": _sha256(index_bytes),
            "content_sha256": portable_index["content_sha256"],
            "decision_count": len(order),
            "decision_order_sha256": portable_index["decision_order_sha256"],
            "decision_set_sha256": portable_index["decision_set_sha256"],
        },
        "frontier_contract": {
            "policy": "complete_deterministic_frontier_consumed_lazily",
            "arbitrary_fixed_reserve_cutoff_used": False,
            "manual_exclusions_used": False,
            "unexplored_higher_priority_record_count": 0,
            "decision_requirement": (
                "every_candidate_preceding_or_equal_to_each_selected_frontier_slot"
            ),
        },
        "selection_proof": portable_proof,
        "decision_census": decision_census,
        "qualification_scratch_cache": scratch_evidence,
        "partial_cache_exclusion": {
            "partial_cache_as_membership_input_permitted": False,
            "partial_cache_role_count": 0,
            "partial_cache_tree_accessed": False,
        },
        "scope": {
            "input_modality": "canonical_bool_mask_only",
            "pair_stream_splits_read": ["train", "val"],
            "historical_test_pair_stream_read": False,
            "sealed_real_read": False,
            "model_imported": False,
            "model_executed": False,
            "supervision_fields_read_by_geometry_qualification": [],
            "fixture_scale_test_only": _fixture_scale_test_only,
        },
        "portable_privacy": {
            "absolute_paths_present": False,
            "member_group_component_fragment_or_pair_identifiers_present": False,
            "supervision_labels_or_historical_directions_present_in_index": False,
            "opaque_pair_commitments_present": True,
            "aggregate_population_difference_only": True,
            "secrets_present": False,
        },
    }
    receipt["content_sha256"] = _sha256(_canonical_json(receipt))
    return _portable(receipt, "eligibility_receipt")


def canonical_eligibility_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize an untrusted document for independent file/content locking."""

    return _canonical_json(_portable(dict(value)))


# Public authority values and the four-captured-payload consumer are defined
# once in the dependency-light module.  Producer-side geometry helpers in this
# module deliberately re-export those exact identities.
from staging.pairwise_v0_2.preflight.local_q1_geometry_eligibility_authority import (  # noqa: E402
    _canonical_json,
    _commit_order,
    _commit_set,
    _content_sha256,
    _deep_freeze,
    _parse_assessment,
    _parse_strict_json,
    _portable,
    _require_sha256,
    _sha256,
    _validate_qualification_contract,
    _validate_scratch_cache_evidence,
    _validate_selection_proof,
    LocalQ1EligibilityDecision,
    LocalQ1EligibilityTrust,
    LocalQ1ExternalFileLock,
    LocalQ1GeometryEligibilityAuthority,
    LocalQ1GeometryQualificationError,
    LocalQ1PairGeometryInput,
    LocalQ1RouteAConfigContractAuthority,
    load_local_q1_geometry_eligibility_authority_payloads_v2,
    load_local_q1_route_a_config_authority_payload_contract,
    load_local_q1_route_a_policy_payload,
)


__all__ = [
    "CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH",
    "CANONICAL_ROUTE_A_POLICY_PATH",
    "LOCAL_Q1_ELIGIBILITY_INDEX_SCHEMA_VERSION",
    "LOCAL_Q1_ELIGIBILITY_RECEIPT_SCHEMA_VERSION",
    "ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256",
    "ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256",
    "ROUTE_A_GEOMETRY_BATCH_CONFIG_SHA256",
    "ROUTE_A_PLANNING_GUARD_CONFIG_SHA256",
    "ROUTE_A_POLICY_CONTENT_SHA256",
    "ROUTE_A_POLICY_FILE_SHA256",
    "LazyLocalQ1GeometryQualifier",
    "LocalQ1EligibilityDecision",
    "LocalQ1EligibilityTrust",
    "LocalQ1ExternalFileLock",
    "LocalQ1GeometryEligibilityAuthority",
    "LocalQ1GeometryQualificationError",
    "LocalQ1PairGeometryInput",
    "LocalQ1RouteAConfigAuthority",
    "LocalQ1RouteAConfigContractAuthority",
    "build_untrusted_eligibility_index",
    "build_untrusted_eligibility_receipt",
    "canonical_eligibility_bytes",
    "load_local_q1_geometry_eligibility_authority",
    "load_local_q1_geometry_eligibility_authority_payloads",
    "load_local_q1_geometry_eligibility_authority_payloads_v2",
    "load_local_q1_route_a_policy_payload",
    "load_local_q1_route_a_config_authority",
    "load_local_q1_route_a_config_authority_payload",
    "load_local_q1_route_a_config_authority_payload_contract",
]

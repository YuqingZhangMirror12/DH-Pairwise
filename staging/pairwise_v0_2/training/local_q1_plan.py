#!/usr/bin/env python3
"""Freeze the formal four-phase LOCAL-Q1 batch plan from locked inputs.

This entry point never builds a cache and imports no model/backend.  It opens
the already frozen production cache under external run-plan, source, freeze,
cache, inventory, and build locks; loads one externally file/content-locked
configuration receipt with every dataclass field explicit; builds the
label-blind component-partitioned plan; writes it once; and reopens it under
its newly established file/content locks.

The caller must freeze the planning-config receipt before this command and
supply both hashes.  This CLI can verify equality to those caller-supplied
arguments, but it cannot establish that they came from an independent
authority.  The structural plan remains non-result-bearing until a later
independent run authority locks its file/content hashes, which transitively
bind the planning-config receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type

from staging.pairwise_v0_2.geometry import CandidateBuilderConfig, CorrosionConfig
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.geometry_cache import GeometryCacheLimits
from staging.pairwise_v0_2.training.local_cache_inventory import (
    LocalCacheInventoryConfig,
)
from staging.pairwise_v0_2.training.local_q1_cache_builder import (
    LOCAL_Q1_RUN_PLAN_ROLE_SPECS,
    LocalQ1CacheTrust,
    LocalQ1ProductionBinding,
    OpenedLocalQ1Cache,
    ProductionMaskLoaderFactory,
    reopen_production_local_q1_cache,
)
from staging.pairwise_v0_2.training.local_q1_provider import (
    FrozenLocalQ1BatchPlan,
    LocalQ1BatchPlanConfig,
    LocalQ1ExternalLocks,
    build_local_q1_batch_plan,
    reopen_local_q1_batch_plan,
    write_local_q1_batch_plan,
)


LOCAL_Q1_PLANNING_CONFIG_SCHEMA_VERSION = "dunhuang-local-q1-planning-config/0.2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "absolute_path",
        "archive_member",
        "component_id",
        "fragment_id",
        "member_path",
        "pair_id",
        "path",
        "secret",
    }
)


class LocalQ1PlanError(RuntimeError):
    """The formal planning boundary was incomplete or inconsistent."""


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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise LocalQ1PlanError(name + " must be lowercase SHA-256")
    return value


def _portable(value: Any, location: str = "planning_config") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and (
            value.startswith(("/", "~/", "file://"))
            or re.match(r"^[A-Za-z]:[\\/]", value)
        ):
            raise LocalQ1PlanError(location + " contains a machine-local path")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LocalQ1PlanError(location + " contains NaN/Inf")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LocalQ1PlanError(location + " contains a non-string key")
            if key.casefold() in _FORBIDDEN_CONFIG_KEYS:
                raise LocalQ1PlanError(location + " exposes a local or row identity")
            result[key] = _portable(item, location + "." + key)
        return result
    if isinstance(value, (tuple, list)):
        return [
            _portable(item, "{}[{}]".format(location, index))
            for index, item in enumerate(value)
        ]
    raise LocalQ1PlanError(
        "{} is not JSON-portable: {}".format(location, type(value).__name__)
    )


def _strict_json_loads(payload: bytes) -> Mapping[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise LocalQ1PlanError("planning config receipt is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise LocalQ1PlanError("planning config receipt root must be an object")
    return value


def _exact_fields(value: Any, cls: Type[Any], name: str) -> Dict[str, Any]:
    expected = {field.name for field in fields(cls)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LocalQ1PlanError(
            "{} must explicitly contain every {} field exactly".format(
                name, cls.__name__
            )
        )
    return dict(value)


def _candidate_config(value: Any) -> CandidateBuilderConfig:
    values = _exact_fields(value, CandidateBuilderConfig, "candidate config")
    corrosion = _exact_fields(
        values["corrosion"], CorrosionConfig, "candidate corrosion config"
    )
    values["corrosion"] = CorrosionConfig(**corrosion)
    values["window_scale_fractions"] = tuple(values["window_scale_fractions"])
    values["output_size"] = tuple(values["output_size"])
    return CandidateBuilderConfig(**values)


def _inventory_config(value: Any) -> LocalCacheInventoryConfig:
    values = _exact_fields(value, LocalCacheInventoryConfig, "cache inventory config")
    values["geometry"] = _candidate_config(values["geometry"])
    values["sequence_length_bucket_edges"] = tuple(
        values["sequence_length_bucket_edges"]
    )
    values["candidate_count_bucket_edges"] = tuple(
        values["candidate_count_bucket_edges"]
    )
    return LocalCacheInventoryConfig(**values)


def _geometry_batch_config(value: Any) -> GeometryBatchConfig:
    values = _exact_fields(value, GeometryBatchConfig, "geometry batch config")
    values["geometry"] = _candidate_config(values["geometry"])
    values["coarse_output_size"] = tuple(values["coarse_output_size"])
    return GeometryBatchConfig(**values)


def _batch_plan_config(value: Any) -> LocalQ1BatchPlanConfig:
    return LocalQ1BatchPlanConfig(
        **_exact_fields(value, LocalQ1BatchPlanConfig, "batch plan config")
    )


def _cache_limits(value: Any) -> GeometryCacheLimits:
    return GeometryCacheLimits(
        **_exact_fields(value, GeometryCacheLimits, "geometry cache limits")
    )


@dataclass(frozen=True)
class ExplicitLocalQ1PlanningConfig:
    inventory_config: LocalCacheInventoryConfig
    geometry_batch_config: GeometryBatchConfig
    batch_plan_config: LocalQ1BatchPlanConfig
    cache_limits: GeometryCacheLimits
    receipt: Mapping[str, Any]
    canonical_file_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.inventory_config, LocalCacheInventoryConfig):
            raise TypeError("inventory_config must be LocalCacheInventoryConfig")
        if not isinstance(self.geometry_batch_config, GeometryBatchConfig):
            raise TypeError("geometry_batch_config must be GeometryBatchConfig")
        if not isinstance(self.batch_plan_config, LocalQ1BatchPlanConfig):
            raise TypeError("batch_plan_config must be LocalQ1BatchPlanConfig")
        if not isinstance(self.cache_limits, GeometryCacheLimits):
            raise TypeError("cache_limits must be GeometryCacheLimits")
        _require_sha256(self.canonical_file_sha256, "planning config file hash")
        _require_sha256(self.content_sha256, "planning config content hash")
        portable = _portable(dict(self.receipt))
        if _sha256(_canonical_json(portable)) != self.canonical_file_sha256:
            raise LocalQ1PlanError("planning config canonical file hash mismatch")
        unsigned = dict(portable)
        claimed = unsigned.pop("content_sha256", None)
        if (
            claimed != self.content_sha256
            or _sha256(_canonical_json(unsigned)) != self.content_sha256
        ):
            raise LocalQ1PlanError("planning config content hash mismatch")
        if self.geometry_batch_config.geometry != self.inventory_config.geometry:
            raise LocalQ1PlanError(
                "geometry batch and inventory candidate configs differ"
            )


def load_explicit_local_q1_planning_config(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_content_sha256: str,
) -> ExplicitLocalQ1PlanningConfig:
    """Load one canonical, caller-hash-locked, explicit no-default config."""

    expected_file = _require_sha256(
        expected_file_sha256, "expected planning config file hash"
    )
    expected_content = _require_sha256(
        expected_content_sha256, "expected planning config content hash"
    )
    target = Path(path)
    try:
        if target.is_symlink() or not target.is_file():
            raise LocalQ1PlanError("planning config receipt is missing")
        if target.stat().st_size > _MAX_CONFIG_BYTES:
            raise LocalQ1PlanError("planning config receipt exceeds byte bound")
        payload = target.read_bytes()
    except OSError as exc:
        raise LocalQ1PlanError("cannot read planning config receipt") from exc
    if not hmac.compare_digest(_sha256(payload), expected_file):
        raise LocalQ1PlanError("planning config external file hash mismatch")
    value = _strict_json_loads(payload)
    if payload != _canonical_json(value):
        raise LocalQ1PlanError("planning config receipt is not canonical JSON")
    expected_root = {
        "schema_version",
        "status",
        "inventory_config",
        "geometry_batch_config",
        "batch_plan_config",
        "cache_limits",
        "authority_boundary",
        "content_sha256",
    }
    if set(value) != expected_root:
        raise LocalQ1PlanError("planning config receipt has missing or extra fields")
    if (
        value["schema_version"] != LOCAL_Q1_PLANNING_CONFIG_SCHEMA_VERSION
        or value["status"] != "frozen_explicit_all_fields_no_defaults"
    ):
        raise LocalQ1PlanError("planning config receipt contract changed")
    if value["authority_boundary"] != {
        "hash_arguments_source": "self_provided_by_enclosing_cli_caller",
        "independent_authority_verified_by_this_receipt": False,
        "enclosing_authority_status": "pending",
        "result_bearing_authorized": False,
        "promotion_requirement": (
            "later_independent_run_authority_must_lock_batch_plan_file_and_"
            "content_sha256_which_transitively_bind_this_config_before_any_"
            "result_bearing_execution"
        ),
    }:
        raise LocalQ1PlanError("planning config authority boundary changed")
    unsigned = dict(value)
    claimed_content = unsigned.pop("content_sha256")
    if claimed_content != expected_content or not hmac.compare_digest(
        _sha256(_canonical_json(unsigned)), expected_content
    ):
        raise LocalQ1PlanError("planning config external content hash mismatch")
    portable = _portable(dict(value))
    return ExplicitLocalQ1PlanningConfig(
        inventory_config=_inventory_config(portable["inventory_config"]),
        geometry_batch_config=_geometry_batch_config(portable["geometry_batch_config"]),
        batch_plan_config=_batch_plan_config(portable["batch_plan_config"]),
        cache_limits=_cache_limits(portable["cache_limits"]),
        receipt=portable,
        canonical_file_sha256=expected_file,
        content_sha256=expected_content,
    )


def _assert_zero_miss_read_only(opened: OpenedLocalQ1Cache) -> None:
    if not isinstance(opened, OpenedLocalQ1Cache):
        raise TypeError("opened must be OpenedLocalQ1Cache")
    replay = opened.inventory_replay.receipt.get("operational_counts", {})
    if (
        not opened.cache.read_only
        or opened.cache.frozen_receipt_trust is None
        or replay.get("cache_miss_count") != 0
        or replay.get("cache_hit_count")
        != replay.get("unique_canonical_fragment_count")
    ):
        raise LocalQ1PlanError(
            "planning requires an externally locked read-only zero-miss cache"
        )


def freeze_production_local_q1_batch_plan(
    *,
    binding: LocalQ1ProductionBinding,
    cache_dir: Path,
    output_path: Path,
    trust: LocalQ1CacheTrust,
    config: ExplicitLocalQ1PlanningConfig,
) -> FrozenLocalQ1BatchPlan:
    """Reopen cache, build four phases, write once, and structurally reopen.

    The reopen under just-computed plan hashes detects write corruption; it is
    not an independent authorization of the plan for result-bearing execution.
    """

    if not isinstance(binding, LocalQ1ProductionBinding):
        raise TypeError("binding must be LocalQ1ProductionBinding")
    if not isinstance(trust, LocalQ1CacheTrust):
        raise TypeError("trust must be LocalQ1CacheTrust")
    if not isinstance(config, ExplicitLocalQ1PlanningConfig):
        raise TypeError("config must be ExplicitLocalQ1PlanningConfig")
    opened = reopen_production_local_q1_cache(
        binding=binding,
        output_dir=Path(cache_dir),
        trust=trust,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
    )
    loader_factory = ProductionMaskLoaderFactory(
        mm_archive=binding.role_paths["mm_archive"],
        eccv_archive=binding.role_paths["eccv_archive"],
        synthetic_archive=binding.role_paths["synthetic_archive"],
        source_bundle=binding.source_bundle,
        expected_source_bundle_manifest_sha256=(
            trust.expected_source_bundle_manifest_sha256
        ),
    )
    return freeze_local_q1_batch_plan(
        opened=opened,
        loader_factory=loader_factory,
        output_path=output_path,
        trust=trust,
        config=config,
    )


def freeze_local_q1_batch_plan(
    *,
    opened: OpenedLocalQ1Cache,
    loader_factory: Any,
    output_path: Path,
    trust: LocalQ1CacheTrust,
    config: ExplicitLocalQ1PlanningConfig,
) -> FrozenLocalQ1BatchPlan:
    """Freeze four phases from any externally locked exact LOCAL-Q1 population.

    This is the Route-A bridge: population reconstruction and cache reopening
    happen before this boundary.  The planner therefore depends only on the
    already opened, read-only, zero-miss cache and its exact external locks; it
    does not fall back to the predecessor production-freeze constants.
    """

    if not isinstance(opened, OpenedLocalQ1Cache):
        raise TypeError("opened must be OpenedLocalQ1Cache")
    if not callable(loader_factory):
        raise TypeError("loader_factory must be callable")
    if not isinstance(trust, LocalQ1CacheTrust):
        raise TypeError("trust must be LocalQ1CacheTrust")
    if not isinstance(config, ExplicitLocalQ1PlanningConfig):
        raise TypeError("config must be ExplicitLocalQ1PlanningConfig")
    target = Path(output_path)
    if target.exists() or target.is_symlink():
        raise LocalQ1PlanError("batch-plan output already exists")
    population = opened.population
    if (
        trust.expected_freeze_file_sha256 != population.freeze_file_sha256
        or trust.expected_freeze_content_sha256 != population.freeze_content_sha256
        or trust.expected_run_plan_file_sha256
        != population.precache_authority.run_plan_file_sha256
        or trust.expected_run_plan_content_sha256
        != population.precache_authority.run_plan_content_sha256
        or trust.expected_source_bundle_manifest_sha256
        != population.precache_authority.source_bundle_manifest_sha256
    ):
        raise LocalQ1PlanError("planning trust differs from opened population")
    _assert_zero_miss_read_only(opened)
    external_locks = LocalQ1ExternalLocks(
        run_plan_file_sha256=trust.expected_run_plan_file_sha256,
        run_plan_content_sha256=trust.expected_run_plan_content_sha256,
        source_bundle_manifest_sha256=(trust.expected_source_bundle_manifest_sha256),
        planning_config_receipt_file_sha256=config.canonical_file_sha256,
        planning_config_receipt_content_sha256=config.content_sha256,
        freeze_file_sha256=trust.expected_freeze_file_sha256,
        freeze_content_sha256=trust.expected_freeze_content_sha256,
        cache_receipt_file_sha256=trust.expected_cache_receipt_file_sha256,
        cache_receipt_content_sha256=(trust.expected_cache_receipt_content_sha256),
        inventory_receipt_file_sha256=(trust.expected_inventory_receipt_file_sha256),
        inventory_receipt_content_sha256=(
            trust.expected_inventory_receipt_content_sha256
        ),
        inventory_semantic_sha256=trust.expected_inventory_semantic_sha256,
        build_receipt_file_sha256=trust.expected_build_receipt_file_sha256,
        build_receipt_content_sha256=trust.expected_build_receipt_content_sha256,
    )
    plan = build_local_q1_batch_plan(
        opened=opened,
        loader_factory=loader_factory,
        geometry_config=config.geometry_batch_config,
        inventory_config=config.inventory_config,
        external_locks=external_locks,
        plan_config=config.batch_plan_config,
    )
    counts = plan.receipt.get("operational_counts", {})
    if (
        counts.get("planning_cache_miss_count") != 0
        or counts.get("planning_cache_write_count") != 0
        or counts.get("planning_geometry_build_count") != 0
    ):
        raise LocalQ1PlanError("batch planning was not zero-miss/zero-write")
    observed_file = write_local_q1_batch_plan(target, plan)
    if not hmac.compare_digest(observed_file, plan.canonical_file_sha256):
        raise LocalQ1PlanError("written batch-plan file hash changed")
    trusted_plan_locks = replace(
        external_locks,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
    )
    reopened = reopen_local_q1_batch_plan(target, external_locks=trusted_plan_locks)
    if (
        reopened.canonical_file_sha256 != plan.canonical_file_sha256
        or reopened.content_sha256 != plan.content_sha256
        or _canonical_json(reopened.receipt) != _canonical_json(plan.receipt)
    ):
        raise LocalQ1PlanError("written batch plan failed exact external reopen")
    return reopened


def _sha_argument(value: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("value must be lowercase SHA-256")
    return value


def _parse_binding(value: str) -> Tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or role not in LOCAL_Q1_RUN_PLAN_ROLE_SPECS or not path:
        raise argparse.ArgumentTypeError("binding must be canonical ROLE=PATH")
    return role, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", action="append", type=_parse_binding, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planning-config-receipt", type=Path, required=True)
    for name in (
        "run-plan-file-sha256",
        "run-plan-content-sha256",
        "source-bundle-manifest-sha256",
        "planning-config-receipt-file-sha256",
        "planning-config-receipt-content-sha256",
        "freeze-file-sha256",
        "freeze-content-sha256",
        "cache-receipt-file-sha256",
        "cache-receipt-content-sha256",
        "inventory-receipt-file-sha256",
        "inventory-receipt-content-sha256",
        "inventory-semantic-sha256",
        "build-receipt-file-sha256",
        "build-receipt-content-sha256",
    ):
        parser.add_argument("--" + name, type=_sha_argument, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    pairs = list(args.bind)
    role_paths = dict(pairs)
    if len(role_paths) != len(pairs) or set(role_paths) != set(
        LOCAL_Q1_RUN_PLAN_ROLE_SPECS
    ):
        raise LocalQ1PlanError(
            "CLI requires every canonical LOCAL-Q1 role exactly once"
        )
    config = load_explicit_local_q1_planning_config(
        args.planning_config_receipt,
        expected_file_sha256=args.planning_config_receipt_file_sha256,
        expected_content_sha256=args.planning_config_receipt_content_sha256,
    )
    trust = LocalQ1CacheTrust(
        expected_freeze_file_sha256=args.freeze_file_sha256,
        expected_freeze_content_sha256=args.freeze_content_sha256,
        expected_run_plan_file_sha256=args.run_plan_file_sha256,
        expected_run_plan_content_sha256=args.run_plan_content_sha256,
        expected_source_bundle_manifest_sha256=(args.source_bundle_manifest_sha256),
        expected_cache_receipt_file_sha256=args.cache_receipt_file_sha256,
        expected_cache_receipt_content_sha256=(args.cache_receipt_content_sha256),
        expected_inventory_receipt_file_sha256=(args.inventory_receipt_file_sha256),
        expected_inventory_receipt_content_sha256=(
            args.inventory_receipt_content_sha256
        ),
        expected_inventory_semantic_sha256=args.inventory_semantic_sha256,
        expected_build_receipt_file_sha256=args.build_receipt_file_sha256,
        expected_build_receipt_content_sha256=(args.build_receipt_content_sha256),
    )
    plan = freeze_production_local_q1_batch_plan(
        binding=LocalQ1ProductionBinding(
            plan_path=args.run_plan,
            source_bundle=args.source_bundle,
            role_paths=role_paths,
        ),
        cache_dir=args.cache_dir,
        output_path=args.output,
        trust=trust,
        config=config,
    )
    print(
        _canonical_json(
            {
                "status": (
                    "structural_four_phase_batch_plan_written_and_reopened_"
                    "pending_independent_run_authority"
                ),
                "batch_plan_file_sha256": plan.canonical_file_sha256,
                "batch_plan_content_sha256": plan.content_sha256,
                "result_bearing_authorized": False,
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LOCAL_Q1_PLANNING_CONFIG_SCHEMA_VERSION",
    "ExplicitLocalQ1PlanningConfig",
    "LocalQ1PlanError",
    "freeze_local_q1_batch_plan",
    "freeze_production_local_q1_batch_plan",
    "load_explicit_local_q1_planning_config",
    "main",
]

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

import staging.pairwise_v0_2.training.local_q1_plan as plan_module
from staging.pairwise_v0_2.training.geometry_cache import GeometryCacheLimits
from staging.pairwise_v0_2.training.local_q1_cache_builder import (
    LOCAL_Q1_RUN_PLAN_ROLE_SPECS,
    LocalQ1ProductionBinding,
)
from staging.pairwise_v0_2.training.local_q1_plan import (
    LOCAL_Q1_PLANNING_CONFIG_SCHEMA_VERSION,
    LocalQ1PlanError,
    freeze_production_local_q1_batch_plan,
    load_explicit_local_q1_planning_config,
    main,
)
from staging.pairwise_v0_2.training.local_q1_provider import (
    LocalQ1BatchPlanConfig,
)
from staging.pairwise_v0_2.tests.test_local_q1_cache_builder import _trust
from staging.pairwise_v0_2.tests.test_local_q1_provider import _geometry


pytest_plugins = ("staging.pairwise_v0_2.tests.test_local_q1_provider",)


def _canonical(value) -> bytes:
    def normalize(item):
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


def _write_config_receipt(
    path,
    *,
    inventory_config,
    geometry_config=None,
    plan_config=None,
    cache_limits=None,
    mutate=None,
):
    receipt = {
        "schema_version": LOCAL_Q1_PLANNING_CONFIG_SCHEMA_VERSION,
        "status": "frozen_explicit_all_fields_no_defaults",
        "inventory_config": asdict(inventory_config),
        "geometry_batch_config": asdict(geometry_config or _geometry()),
        "batch_plan_config": asdict(
            plan_config or LocalQ1BatchPlanConfig(packing_max_records=2)
        ),
        "cache_limits": asdict(cache_limits or GeometryCacheLimits()),
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
    if mutate is not None:
        mutate(receipt)
    receipt["content_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    payload = _canonical(receipt)
    path.write_bytes(payload)
    return receipt, hashlib.sha256(payload).hexdigest()


def _fixture_trust(population, artifacts):
    return replace(
        _trust(artifacts),
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
    )


def _fixture_binding(tmp_path):
    return LocalQ1ProductionBinding(
        plan_path=tmp_path / "formal-precache-plan.json",
        source_bundle=tmp_path / "source-bundle",
        role_paths={
            role: tmp_path / "roles" / (role + ".bin")
            for role in LOCAL_Q1_RUN_PLAN_ROLE_SPECS
        },
    )


def test_explicit_config_receipt_is_canonical_locked_and_uses_no_defaults(
    tmp_path, provider_case
):
    inventory = provider_case[2]
    path = tmp_path / "planning-config.json"
    receipt, file_sha = _write_config_receipt(path, inventory_config=inventory)
    loaded = load_explicit_local_q1_planning_config(
        path,
        expected_file_sha256=file_sha,
        expected_content_sha256=receipt["content_sha256"],
    )
    assert loaded.inventory_config == inventory
    assert loaded.geometry_batch_config == _geometry()
    assert loaded.batch_plan_config == LocalQ1BatchPlanConfig(packing_max_records=2)
    assert loaded.canonical_file_sha256 == file_sha
    assert loaded.content_sha256 == receipt["content_sha256"]
    assert loaded.receipt["authority_boundary"]["enclosing_authority_status"] == (
        "pending"
    )
    assert loaded.receipt["authority_boundary"]["result_bearing_authorized"] is False

    missing = tmp_path / "missing-field.json"
    missing_receipt, missing_file_sha = _write_config_receipt(
        missing,
        inventory_config=inventory,
        mutate=lambda value: value["geometry_batch_config"].pop(
            "max_candidate_id_bytes"
        ),
    )
    with pytest.raises(LocalQ1PlanError, match="every GeometryBatchConfig field"):
        load_explicit_local_q1_planning_config(
            missing,
            expected_file_sha256=missing_file_sha,
            expected_content_sha256=missing_receipt["content_sha256"],
        )


def test_config_receipt_rejects_wrong_hash_noncanonical_and_paths(
    tmp_path, provider_case
):
    inventory = provider_case[2]
    path = tmp_path / "planning-config.json"
    receipt, file_sha = _write_config_receipt(path, inventory_config=inventory)
    with pytest.raises(LocalQ1PlanError, match="external file hash mismatch"):
        load_explicit_local_q1_planning_config(
            path,
            expected_file_sha256="f" * 64,
            expected_content_sha256=receipt["content_sha256"],
        )

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(LocalQ1PlanError, match="not canonical JSON"):
        load_explicit_local_q1_planning_config(
            noncanonical,
            expected_file_sha256=hashlib.sha256(noncanonical.read_bytes()).hexdigest(),
            expected_content_sha256=receipt["content_sha256"],
        )

    path_receipt = copy.deepcopy(receipt)
    path_receipt["batch_plan_config"]["unexpected_path"] = "/private/data"
    path_receipt.pop("content_sha256")
    path_receipt["content_sha256"] = hashlib.sha256(
        _canonical(path_receipt)
    ).hexdigest()
    path_payload = _canonical(path_receipt)
    path_file = tmp_path / "path.json"
    path_file.write_bytes(path_payload)
    with pytest.raises(LocalQ1PlanError, match="machine-local path|every.*field"):
        load_explicit_local_q1_planning_config(
            path_file,
            expected_file_sha256=hashlib.sha256(path_payload).hexdigest(),
            expected_content_sha256=path_receipt["content_sha256"],
        )
    assert file_sha == hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_receipt_self_consistent_resign_cannot_forge_authority(
    tmp_path, provider_case
):
    inventory = provider_case[2]
    cases = (
        lambda value: value["authority_boundary"].__setitem__(
            "independent_authority_verified_by_this_receipt", True
        ),
        lambda value: value["authority_boundary"].__setitem__(
            "enclosing_authority_status", "verified"
        ),
        lambda value: value["authority_boundary"].__setitem__(
            "result_bearing_authorized", True
        ),
        lambda value: value["authority_boundary"].__setitem__(
            "hash_arguments_source", "independent_registry"
        ),
    )
    for index, mutate in enumerate(cases):
        path = tmp_path / "forged-authority-{}.json".format(index)
        receipt, file_sha = _write_config_receipt(
            path,
            inventory_config=inventory,
            mutate=mutate,
        )
        with pytest.raises(LocalQ1PlanError, match="authority boundary"):
            load_explicit_local_q1_planning_config(
                path,
                expected_file_sha256=file_sha,
                expected_content_sha256=receipt["content_sha256"],
            )


def test_formal_orchestrator_reopens_builds_writes_and_reopens_exactly(
    tmp_path, monkeypatch, provider_case
):
    population, factory, inventory, artifacts, opened, _locks = provider_case
    config_path = tmp_path / "planning-config.json"
    receipt, file_sha = _write_config_receipt(config_path, inventory_config=inventory)
    config = load_explicit_local_q1_planning_config(
        config_path,
        expected_file_sha256=file_sha,
        expected_content_sha256=receipt["content_sha256"],
    )
    calls = {"reopen_cache": 0, "loader_factory": 0}

    def reopen_cache(**kwargs):
        calls["reopen_cache"] += 1
        assert kwargs["inventory_config"] == inventory
        assert kwargs["cache_limits"] == config.cache_limits
        return opened

    def loader_factory(**kwargs):
        calls["loader_factory"] += 1
        assert kwargs["expected_source_bundle_manifest_sha256"] == (
            population.precache_authority.source_bundle_manifest_sha256
        )
        return factory

    monkeypatch.setattr(plan_module, "reopen_production_local_q1_cache", reopen_cache)
    monkeypatch.setattr(plan_module, "ProductionMaskLoaderFactory", loader_factory)
    output = tmp_path / "formal-batch-plan.json"
    plan = freeze_production_local_q1_batch_plan(
        binding=_fixture_binding(tmp_path),
        cache_dir=tmp_path / "cache",
        output_path=output,
        trust=_fixture_trust(population, artifacts),
        config=config,
    )
    assert calls == {"reopen_cache": 1, "loader_factory": 1}
    assert output.read_bytes() == _canonical(plan.receipt)
    assert set(plan.receipt["phases"]) == {
        "train",
        "validation_select",
        "validation_calibration",
        "validation_report",
    }
    assert (
        plan.receipt["external_locks"]["planning_config_receipt_file_sha256"]
        == file_sha
    )
    assert (
        plan.receipt["external_locks"]["planning_config_receipt_content_sha256"]
        == receipt["content_sha256"]
    )
    assert plan.receipt["operational_counts"]["planning_cache_miss_count"] == 0
    assert plan.receipt["operational_counts"]["planning_cache_write_count"] == 0
    assert plan.receipt["authority_boundary"][
        "enclosing_planning_authority_status"
    ] == ("pending")
    assert plan.receipt["authority_boundary"]["result_bearing_authorized"] is False
    encoded = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in encoded
    assert all(
        record.component_id not in encoded for record in population.validation_records
    )
    with pytest.raises(LocalQ1PlanError, match="output already exists"):
        freeze_production_local_q1_batch_plan(
            binding=_fixture_binding(tmp_path),
            cache_dir=tmp_path / "cache",
            output_path=output,
            trust=_fixture_trust(population, artifacts),
            config=config,
        )


def test_cli_requires_every_explicit_argument_and_emits_hashes_only(
    tmp_path, monkeypatch, capsys, provider_case
):
    population, _factory, inventory, artifacts, _opened, _locks = provider_case
    config_path = tmp_path / "planning-config.json"
    receipt, file_sha = _write_config_receipt(config_path, inventory_config=inventory)
    trust = _fixture_trust(population, artifacts)
    expected_plan = SimpleNamespace(
        canonical_file_sha256="a" * 64,
        content_sha256="b" * 64,
    )
    monkeypatch.setattr(
        plan_module,
        "freeze_production_local_q1_batch_plan",
        lambda **_kwargs: expected_plan,
    )
    role_args = [
        value
        for role in LOCAL_Q1_RUN_PLAN_ROLE_SPECS
        for value in ("--bind", "{}={}".format(role, tmp_path / (role + ".bin")))
    ]
    hash_args = {
        "run-plan-file-sha256": trust.expected_run_plan_file_sha256,
        "run-plan-content-sha256": trust.expected_run_plan_content_sha256,
        "source-bundle-manifest-sha256": (trust.expected_source_bundle_manifest_sha256),
        "planning-config-receipt-file-sha256": file_sha,
        "planning-config-receipt-content-sha256": receipt["content_sha256"],
        "freeze-file-sha256": trust.expected_freeze_file_sha256,
        "freeze-content-sha256": trust.expected_freeze_content_sha256,
        "cache-receipt-file-sha256": trust.expected_cache_receipt_file_sha256,
        "cache-receipt-content-sha256": (trust.expected_cache_receipt_content_sha256),
        "inventory-receipt-file-sha256": (trust.expected_inventory_receipt_file_sha256),
        "inventory-receipt-content-sha256": (
            trust.expected_inventory_receipt_content_sha256
        ),
        "inventory-semantic-sha256": trust.expected_inventory_semantic_sha256,
        "build-receipt-file-sha256": trust.expected_build_receipt_file_sha256,
        "build-receipt-content-sha256": (trust.expected_build_receipt_content_sha256),
    }
    args = role_args + [
        "--run-plan",
        str(tmp_path / "run-plan.json"),
        "--source-bundle",
        str(tmp_path / "source"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--output",
        str(tmp_path / "batch-plan.json"),
        "--planning-config-receipt",
        str(config_path),
    ]
    for name, value in hash_args.items():
        args.extend(("--" + name, value))
    assert main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "status": (
            "structural_four_phase_batch_plan_written_and_reopened_"
            "pending_independent_run_authority"
        ),
        "batch_plan_file_sha256": "a" * 64,
        "batch_plan_content_sha256": "b" * 64,
        "result_bearing_authorized": False,
    }
    assert str(tmp_path) not in _canonical(output).decode("utf-8")

    with pytest.raises(SystemExit):
        main([])

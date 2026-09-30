from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import staging.pairwise_v0_2.training.local_q1_route_a_research as research_module
from staging.pairwise_v0_2.pairwise_data.training_stream import ArchiveBinding
from staging.pairwise_v0_2.preflight.freeze_local_q1 import ROUTE_A_SCHEMA_VERSION
from staging.pairwise_v0_2.tests.test_local_q1_cache_builder import (
    _MaskLoaderFactory,
    _authority,
    _config,
    _masks,
    _records,
    _role_locks,
    _trust,
)
from staging.pairwise_v0_2.tests.test_local_q1_provider import (
    _external_locks,
    _formal_population,
    _geometry,
)
from staging.pairwise_v0_2.training.local_q1_cache_builder import (
    LocalQ1CacheBuilderError,
    LocalQ1Population,
    LocalQ1PrecacheAuthority,
    build_local_q1_cache,
    reopen_local_q1_cache,
)
from staging.pairwise_v0_2.training.local_q1_provider import (
    LocalQ1BatchPlanConfig,
    LocalQ1ProviderError,
    LocalQ1ReadOnlyBatchProvider,
    build_local_q1_batch_plan,
)
from staging.pairwise_v0_2.training.local_q1_route_a_research import (
    RouteAResearchError,
    RouteAResearchInputs,
    planning_config_with_local_tensor_bound,
    population_for_existing_cache,
    representation_contract,
    resume_route_a_plan_and_train,
    route_a_planning_config,
)


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _route_a_receipt(train_count: int, validation_count: int):
    proof = {
        "selected_training_count": train_count,
        "selected_validation_count": validation_count,
        "unexplored_higher_priority_record_count": 0,
        "arbitrary_fixed_reserve_cutoff_used": False,
        "selected_failure_counts": {
            "fragment": 0,
            "pair": 0,
            "direction_coverage": 0,
            "resource": 0,
        },
    }
    receipt = {
        "schema_version": ROUTE_A_SCHEMA_VERSION,
        "status": (
            "pass_route_a_geometry_qualified_population_refrozen_no_model_no_test"
        ),
        "scope": {
            "experiment": "LOCAL-Q1",
            "pair_stream_splits_read": ["train", "val"],
            "sealed_real_read": False,
            "archive_mask_members_opened": False,
            "mask_pixels_decoded": False,
            "model_imported": False,
            "model_executed": False,
            "geometry_eligibility_authority_consumed": True,
            "geometry_qualification_executed_by_refreeze": False,
            "partial_cache_tree_accessed_by_refreeze": False,
            "fixture_scale_test_only": False,
        },
        "locks": {
            "historical_identity_index": {
                "member_count": 6,
                "content_sha256": hashlib.sha256(b"identity").hexdigest(),
            }
        },
        "training": {"population": {"count": train_count}},
        "validation": {"population": {"count": validation_count}},
        "geometry_eligibility_authority": {
            "production_authorized": True,
            "selection_replayed_exactly": True,
            "partial_cache_as_membership_input_permitted": False,
            "partial_cache_role_count": 0,
            "partial_cache_tree_accessed": False,
        },
        "route_a_frontier_selection_proof": proof,
    }
    receipt["content_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return receipt


def _route_a_population() -> LocalQ1Population:
    train, validation = _records()
    receipt = _route_a_receipt(len(train), len(validation))
    freeze_file = hashlib.sha256(b"route-a-freeze").hexdigest()
    return LocalQ1Population(
        training_records=train,
        validation_records=validation,
        freeze_receipt=receipt,
        freeze_file_sha256=freeze_file,
        freeze_content_sha256=receipt["content_sha256"],
        input_role_locks=_role_locks(freeze_file),
        precache_authority=_authority(),
    )


def test_population_accepts_exact_route_a_freeze_and_rejects_missing_proof():
    population = _route_a_population()
    train, validation = _records()
    receipt = dict(population.freeze_receipt)
    freeze_file = population.freeze_file_sha256
    assert population.records == train + validation

    changed = copy.deepcopy(receipt)
    changed["route_a_frontier_selection_proof"][
        "unexplored_higher_priority_record_count"
    ] = 1
    changed["content_sha256"] = hashlib.sha256(
        _canonical(
            {key: value for key, value in changed.items() if key != "content_sha256"}
        )
    ).hexdigest()
    with pytest.raises(LocalQ1CacheBuilderError, match="complete-frontier"):
        LocalQ1Population(
            training_records=train,
            validation_records=validation,
            freeze_receipt=changed,
            freeze_file_sha256=freeze_file,
            freeze_content_sha256=changed["content_sha256"],
            input_role_locks=_role_locks(freeze_file),
            precache_authority=_authority(),
        )


def test_route_a_population_builds_a_fresh_exact_cache(tmp_path):
    artifacts = build_local_q1_cache(
        population=_route_a_population(),
        loader_factory=_MaskLoaderFactory(_masks()),
        output_dir=tmp_path / "route-a-cache",
        inventory_config=_config(),
        producer_workers=1,
    )
    assert artifacts.cache_receipt["artifact_count"] == 5
    assert artifacts.build_receipt["status"] == (
        "complete_geometry_cache_frozen_zero_miss_no_model_no_test"
    )
    assert (
        artifacts.build_receipt["inventory"]["read_only_replay_cache_miss_count"] == 0
    )

    changed_runtime = replace(
        _route_a_population(),
        precache_authority=LocalQ1PrecacheAuthority(
            run_plan_file_sha256="f" * 64,
            run_plan_content_sha256="f" * 64,
            source_bundle_manifest_sha256="f" * 64,
        ),
    )
    rebound = population_for_existing_cache(changed_runtime, artifacts.output_dir)
    assert (
        rebound.precache_authority.portable_dict()
        == artifacts.build_receipt["precache_authority"]
    )
    assert rebound.training_records == changed_runtime.training_records


def test_route_a_config_maps_directly_to_cache_and_plan_configs():
    config = route_a_planning_config(
        "experiments/local_q1_refreeze_decision/route_a_config_authority_v0_1.json"
    )
    assert config.geometry_batch_config.fingerprint == (
        "dc994bf15596eea1b3d9245ac41505b1e72739eb49a91fef7a02012c744dcfd3"
    )
    assert config.geometry_batch_config.max_candidates_per_sample == 512
    assert config.geometry_batch_config.max_candidates_per_batch == 2048
    assert config.batch_plan_config.packing_max_records == 256
    assert config.inventory_config.require_all_pairs_ok is True


def test_research_local_tensor_bound_only_shrinks_and_resigns_in_memory():
    config = route_a_planning_config(
        "experiments/local_q1_refreeze_decision/route_a_config_authority_v0_1.json"
    )
    bounded = planning_config_with_local_tensor_bound(config, 32_000_000)

    assert config.geometry_batch_config.max_local_tensor_elements == 64_000_000
    assert bounded.geometry_batch_config.max_local_tensor_elements == 32_000_000
    assert (
        bounded.geometry_batch_config.geometry is config.geometry_batch_config.geometry
    )
    assert bounded.inventory_config is config.inventory_config
    assert bounded.batch_plan_config is config.batch_plan_config
    assert bounded.cache_limits is config.cache_limits
    assert bounded.canonical_file_sha256 != config.canonical_file_sha256
    assert bounded.content_sha256 != config.content_sha256
    assert planning_config_with_local_tensor_bound(config, None) is config

    with pytest.raises(RouteAResearchError, match="positive integer"):
        planning_config_with_local_tensor_bound(config, 0)
    with pytest.raises(RouteAResearchError, match="only shrink"):
        planning_config_with_local_tensor_bound(config, 64_000_001)


def test_keypoint_alias_exposes_integrated_four_arm_contract():
    contract = representation_contract("keypoint")
    assert contract["representation"] == "formal_multirun_keypoint_2x2"
    assert contract["cli_representation_alias"] == "keypoint"
    assert contract["arms"] == (
        "local_dual_softmax",
        "local_dustbin_sinkhorn",
        "keypoint_dual_softmax",
        "keypoint_dustbin_sinkhorn",
    )
    assert contract["correspondence_mask"] == (
        "explicit_sparse_bool_N_La_Lb_for_keypoint_and_"
        "implicit_full_cartesian_for_multirun"
    )
    assert contract["tensor_bridge"] == ("training.geometry_batch.RaggedGeometryBatch")
    assert contract["training_status"] == "launchable_now"

    multirun_contract = representation_contract("multirun")
    assert multirun_contract["arms"] == contract["arms"]
    assert multirun_contract["cli_representation_alias"] == "multirun"


def test_extracted_loader_matches_archive_masks_and_plan_bytes(monkeypatch, tmp_path):
    masks = _masks()
    for index in range(3):
        masks["fragment/d_phase_{}".format(index)] = masks["fragment/d"].copy()
        masks["fragment/e_phase_{}".format(index)] = masks["fragment/e"].copy()
    population = _formal_population()
    member_payloads = {}
    for record in population.records:
        for reference in (record.fragment_a, record.fragment_b):
            buffer = io.BytesIO()
            Image.fromarray(
                masks[reference.fragment_id].astype(np.uint8) * 255, mode="L"
            ).save(buffer, format="PNG")
            member_payloads[reference.archive_member] = buffer.getvalue()

    archive_path = tmp_path / "masks.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        for member, payload in sorted(member_payloads.items()):
            archive.writestr(member, payload)
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    binding = ArchiveBinding(
        logical_id="fixture://route-a/extracted-mm",
        archive_format="zip",
        sha256=archive_sha256,
    )
    eccv_binding = replace(binding, logical_id="fixture://route-a/extracted-eccv")
    synthetic_binding = replace(
        binding, logical_id="fixture://route-a/extracted-synthetic"
    )
    monkeypatch.setattr(research_module, "MM_CANONICAL_BINDING", binding)
    monkeypatch.setattr(research_module, "ECCV_CANONICAL_BINDING", eccv_binding)
    monkeypatch.setattr(research_module, "SYNTHETIC_ARCHIVE_BINDING", synthetic_binding)

    extracted_root = tmp_path / "extracted"
    for member, payload in member_payloads.items():
        target = extracted_root / member
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    def bind_record(record):
        first_payload = member_payloads[record.fragment_a.archive_member]
        second_payload = member_payloads[record.fragment_b.archive_member]
        first = replace(
            record.fragment_a,
            binding=binding,
            content_sha256=hashlib.sha256(first_payload).hexdigest(),
        )
        second = replace(
            record.fragment_b,
            binding=binding,
            content_sha256=hashlib.sha256(second_payload).hexdigest(),
        )
        return replace(record, fragment_a=first, fragment_b=second)

    population = replace(
        population,
        training_records=tuple(bind_record(row) for row in population.training_records),
        validation_records=tuple(
            bind_record(row) for row in population.validation_records
        ),
    )
    archive_factory = research_module._ResearchMaskLoaderFactory(
        archive_path, archive_path, archive_path
    )
    extracted_factory = research_module._ResearchMaskLoaderFactory(
        archive_path,
        archive_path,
        archive_path,
        extracted_root,
        extracted_root,
        extracted_root,
    )

    archive_loader = archive_factory()
    extracted_loader = extracted_factory()
    try:
        for record in population.records:
            for reference in (record.fragment_a, record.fragment_b):
                assert np.array_equal(
                    archive_loader(reference), extracted_loader(reference)
                )
    finally:
        archive_loader.close()

    artifacts = build_local_q1_cache(
        population=population,
        loader_factory=archive_factory,
        output_dir=tmp_path / "cache",
        inventory_config=_config(),
        producer_workers=1,
    )
    trust = replace(
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
    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=archive_factory,
        output_dir=artifacts.output_dir,
        trust=trust,
        inventory_config=_config(),
        replay_source_masks=False,
    )
    locks = _external_locks(population, artifacts)
    plan_kwargs = {
        "opened": opened,
        "geometry_config": _geometry(),
        "inventory_config": _config(),
        "external_locks": locks,
        "plan_config": LocalQ1BatchPlanConfig(packing_max_records=2),
    }
    archive_plan = build_local_q1_batch_plan(
        loader_factory=archive_factory, **plan_kwargs
    )
    extracted_plan = build_local_q1_batch_plan(
        loader_factory=extracted_factory, **plan_kwargs
    )
    assert archive_plan.receipt == extracted_plan.receipt
    assert archive_plan.canonical_file_sha256 == extracted_plan.canonical_file_sha256

    bounded_geometry = replace(
        plan_kwargs["geometry_config"], max_local_tensor_elements=100_000
    )
    bounded_plan = build_local_q1_batch_plan(
        loader_factory=extracted_factory,
        **{**plan_kwargs, "geometry_config": bounded_geometry},
    )
    assert (
        bounded_plan.receipt["totals"]["batch_count"]
        > archive_plan.receipt["totals"]["batch_count"]
    )

    def record_commitments(plan):
        output = {}
        for phase, phase_receipt in plan.receipt["phases"].items():
            for batch in phase_receipt["batches"]:
                for index, ordinal in enumerate(batch["record_ordinals"]):
                    output[(phase, ordinal)] = (
                        batch["record_geometry_sha256"][index],
                        batch["record_candidate_sha256"][index],
                        batch["record_candidate_counts"][index],
                        batch["record_candidate_sequence_lengths"][index],
                    )
        return output

    assert record_commitments(bounded_plan) == record_commitments(archive_plan)
    assert (
        bounded_plan.receipt["totals"]["record_count"]
        == archive_plan.receipt["totals"]["record_count"]
    )
    assert (
        bounded_plan.receipt["totals"]["candidate_count"]
        == archive_plan.receipt["totals"]["candidate_count"]
    )

    bounded_locks = replace(
        locks,
        batch_plan_file_sha256=bounded_plan.canonical_file_sha256,
        batch_plan_content_sha256=bounded_plan.content_sha256,
    )
    LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=extracted_factory,
        plan=bounded_plan,
        geometry_config=bounded_geometry,
        inventory_config=_config(),
        external_locks=bounded_locks,
    )
    with pytest.raises(LocalQ1ProviderError, match="config/plan"):
        LocalQ1ReadOnlyBatchProvider(
            opened=opened,
            loader_factory=extracted_factory,
            plan=bounded_plan,
            geometry_config=plan_kwargs["geometry_config"],
            inventory_config=_config(),
            external_locks=bounded_locks,
        )


def test_resume_reopens_existing_cache_once_then_runs_all_four_arms(
    monkeypatch, tmp_path
):
    inputs = RouteAResearchInputs(
        **{name: tmp_path / name for name in RouteAResearchInputs.__dataclass_fields__}
    )
    population = SimpleNamespace(training_records=("train-record",))
    config = SimpleNamespace(
        inventory_config="inventory-config",
        geometry_batch_config="geometry-config",
        cache_limits="cache-limits",
        canonical_file_sha256="a" * 64,
        content_sha256="b" * 64,
    )
    opened = object()
    trust = object()
    loader_factory = object()
    external_locks = object()
    plan = SimpleNamespace(
        canonical_file_sha256="c" * 64,
        content_sha256="d" * 64,
        phase_batches=lambda phase: (phase,),
    )
    calls = {"reopen": 0, "runner": 0}
    provider_arguments = {}

    monkeypatch.setattr(
        research_module, "rebuild_route_a_population", lambda value: population
    )
    monkeypatch.setattr(
        research_module,
        "population_for_existing_cache",
        lambda value, cache_dir: value,
    )
    monkeypatch.setattr(
        research_module, "_planning_config_from_cache", lambda value: config
    )
    monkeypatch.setattr(research_module, "automatic_cache_trust", lambda *args: trust)
    monkeypatch.setattr(
        research_module, "research_loader_factory", lambda value: loader_factory
    )

    def reopen(**kwargs):
        calls["reopen"] += 1
        assert kwargs["output_dir"] == tmp_path / "cache"
        assert kwargs["replay_source_masks"] is False
        return opened

    monkeypatch.setattr(research_module, "reopen_local_q1_cache", reopen)
    monkeypatch.setattr(
        research_module, "freeze_local_q1_batch_plan", lambda **kwargs: plan
    )
    monkeypatch.setattr(
        research_module, "_external_locks", lambda *args, **kwargs: external_locks
    )

    def provider(**kwargs):
        provider_arguments.update(kwargs)
        return "provider"

    monkeypatch.setattr(research_module, "LocalQ1ReadOnlyBatchProvider", provider)
    monkeypatch.setattr(
        research_module, "LocalQ1RunnerContract", lambda **kwargs: kwargs
    )

    def run(**kwargs):
        calls["runner"] += 1
        assert kwargs["provider"] == "provider"
        return SimpleNamespace(
            run_directory=tmp_path / "run",
            receipt_path=tmp_path / "run" / "receipt.json",
        )

    monkeypatch.setattr(research_module, "run_local_q1", run)
    result = resume_route_a_plan_and_train(
        inputs=inputs,
        cache_dir=tmp_path / "cache",
        batch_plan=tmp_path / "batch-plan.json",
        output_root=tmp_path / "outputs",
        epochs=5,
        initialization_seed=260829,
    )

    assert calls == {"reopen": 1, "runner": 1}
    assert provider_arguments["opened"] is opened
    assert result["cache_reopen_count"] == 1
    assert result["arms"] == [
        "local_dual_softmax",
        "local_dustbin_sinkhorn",
        "keypoint_dual_softmax",
        "keypoint_dustbin_sinkhorn",
    ]

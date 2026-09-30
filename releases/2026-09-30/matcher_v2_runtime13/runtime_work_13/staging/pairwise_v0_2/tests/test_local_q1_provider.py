from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import pytest

from staging.pairwise_v0_2.training import local_q1_provider as provider_module
from staging.pairwise_v0_2.models.local_matcher import MatcherMode
from staging.pairwise_v0_2.models.pairwise import ArcPoolingConfig
from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.local_q1_cache_builder import (
    LocalQ1FileLock,
    LocalQ1Population,
    build_local_q1_cache,
    reopen_local_q1_cache,
)
from staging.pairwise_v0_2.training.local_q1_provider import (
    LOCAL_Q1_BATCH_PLAN_SCHEMA_VERSION,
    LOCAL_Q1_BATCH_PROVIDER_VERSION,
    LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION,
    LOCAL_Q1_PROVENANCE_SAFETY_MAX_DEPTH,
    LOCAL_Q1_PROVENANCE_SAFETY_MAX_NODES,
    LOCAL_Q1_PROVIDER_RUNTIME_SCHEMA_VERSION,
    LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE,
    LOCAL_Q1_VALIDATION_ASSIGNMENT_SCHEMA_VERSION,
    FrozenLocalQ1BatchPlan,
    FrozenLocalQ1PlannedSafetyMetadata,
    LocalQ1BatchPlanConfig,
    LocalQ1ExternalLocks,
    LocalQ1ProviderError,
    LocalQ1ReadOnlyBatchProvider,
    build_local_q1_batch_plan,
    freeze_local_q1_validation_assignment,
    local_q1_prepared_digests,
    local_q1_provenance_safety_policy,
    reopen_local_q1_batch_plan,
    scan_local_q1_provenance_safety,
    write_local_q1_batch_plan,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArmName,
    EvidenceMode,
)
from staging.pairwise_v0_2.training.short_ablation_fixture import _arm
from staging.pairwise_v0_2.tests.test_local_q1_cache_builder import (
    _MaskLoaderFactory,
    _config,
    _masks,
    _population,
    _trust,
)


def _canonical(value):
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


def _portable_policy():
    return json.loads(
        _canonical(dict(local_q1_provenance_safety_policy())).decode("utf-8")
    )


def _geometry(**kwargs):
    values = {
        "geometry": _config().geometry,
        "coarse_output_size": (32, 32),
        "max_batch_size": 8,
        "max_input_pixels_per_mask": 72 * 88,
        "max_candidates_per_sample": 512,
        "max_candidates_per_batch": 2048,
        "max_sequence_length": 512,
        "max_local_tensor_elements": 64_000_000,
        "max_attention_score_elements_per_candidate": 1_048_576,
        "max_attention_score_elements_per_batch": 16_777_216,
        "max_affinity_elements_per_candidate": 262_144,
        "max_affinity_elements_per_batch": 8_388_608,
        "max_sinkhorn_elements_per_candidate": 263_169,
        "max_sinkhorn_elements_per_batch": 8_421_408,
    }
    values.update(kwargs)
    return GeometryBatchConfig(**values)


def _external_locks(population, artifacts):
    return LocalQ1ExternalLocks(
        run_plan_file_sha256=population.precache_authority.run_plan_file_sha256,
        run_plan_content_sha256=(population.precache_authority.run_plan_content_sha256),
        source_bundle_manifest_sha256=(
            population.precache_authority.source_bundle_manifest_sha256
        ),
        planning_config_receipt_file_sha256=hashlib.sha256(
            b"fixture-planning-config-file"
        ).hexdigest(),
        planning_config_receipt_content_sha256=hashlib.sha256(
            b"fixture-planning-config-content"
        ).hexdigest(),
        freeze_file_sha256=population.freeze_file_sha256,
        freeze_content_sha256=population.freeze_content_sha256,
        cache_receipt_file_sha256=artifacts.cache_receipt_file_sha256,
        cache_receipt_content_sha256=artifacts.cache_receipt["content_sha256"],
        inventory_receipt_file_sha256=artifacts.inventory_receipt_file_sha256,
        inventory_receipt_content_sha256=artifacts.inventory_receipt["content_sha256"],
        inventory_semantic_sha256=artifacts.inventory_receipt[
            "semantic_commitment_sha256"
        ],
        build_receipt_file_sha256=artifacts.build_receipt_file_sha256,
        build_receipt_content_sha256=artifacts.build_receipt["content_sha256"],
    )


@pytest.fixture(scope="module")
def provider_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("local-q1-provider") / "cache"
    population = _formal_population()
    masks = _masks()
    for index in range(3):
        masks["fragment/d_phase_{}".format(index)] = masks["fragment/d"].copy()
        masks["fragment/e_phase_{}".format(index)] = masks["fragment/e"].copy()
    factory = _MaskLoaderFactory(masks)
    inventory_config = _config()
    artifacts = build_local_q1_cache(
        population=population,
        loader_factory=factory,
        output_dir=root,
        inventory_config=inventory_config,
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
        loader_factory=factory,
        output_dir=root,
        trust=trust,
        inventory_config=inventory_config,
    )
    locks = _external_locks(population, artifacts)
    return population, factory, inventory_config, artifacts, opened, locks


def _formal_population():
    population = _population()
    template = population.validation_records[0]
    validation = []
    for index in range(3):
        component = "component/val/formal-phase-{}".format(index)
        group = "group/val/formal-phase-{}".format(index)
        first = replace(
            template.fragment_a,
            archive_member="masks/d_phase_{}.png".format(index),
            fragment_id="fragment/d_phase_{}".format(index),
            canonical_group_id=group,
            component_id=component,
            content_sha256=hashlib.sha256(
                "fixture/d_phase_{}".format(index).encode("utf-8")
            ).hexdigest(),
        )
        second = replace(
            template.fragment_b,
            archive_member="masks/e_phase_{}.png".format(index),
            fragment_id="fragment/e_phase_{}".format(index),
            canonical_group_id=group,
            component_id=component,
            content_sha256=hashlib.sha256(
                "fixture/e_phase_{}".format(index).encode("utf-8")
            ).hexdigest(),
        )
        validation.append(
            replace(
                template,
                fragment_a=first,
                fragment_b=second,
                canonical_group_id=group,
                component_id=component,
                canonical_pair_key=tuple(
                    sorted((first.fragment_id, second.fragment_id))
                ),
            )
        )
    receipt = copy.deepcopy(dict(population.freeze_receipt))
    receipt["validation"]["population"]["count"] = len(validation)
    receipt.pop("content_sha256")
    receipt["content_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    freeze_file_sha256 = hashlib.sha256(
        b"formal-provider-fixture\0" + _canonical(receipt)
    ).hexdigest()
    roles = dict(population.input_role_locks)
    roles["freeze_receipt"] = LocalQ1FileLock(
        byte_count=len(_canonical(receipt)),
        sha256=freeze_file_sha256,
    )
    return LocalQ1Population(
        training_records=population.training_records,
        validation_records=tuple(validation),
        freeze_receipt=receipt,
        freeze_file_sha256=freeze_file_sha256,
        freeze_content_sha256=receipt["content_sha256"],
        input_role_locks=roles,
        precache_authority=population.precache_authority,
    )


def _build_plan(provider_case, geometry=None, packing=2, opened=None):
    _population_value, factory, inventory, _artifacts, default_opened, locks = (
        provider_case
    )
    return build_local_q1_batch_plan(
        opened=opened or default_opened,
        loader_factory=factory,
        geometry_config=geometry or _geometry(),
        inventory_config=inventory,
        external_locks=locks,
        plan_config=LocalQ1BatchPlanConfig(packing_max_records=packing),
    )


def _plan_locks(locks, plan):
    return replace(
        locks,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
    )


def _local_arms():
    pooling = ArcPoolingConfig()
    return (
        _arm(
            AblationArmName.LOCAL_DUAL_SOFTMAX,
            EvidenceMode.LOCAL,
            MatcherMode.DUAL_SOFTMAX.value,
            pooling,
        ),
        _arm(
            AblationArmName.LOCAL_DUSTBIN_SINKHORN,
            EvidenceMode.LOCAL,
            MatcherMode.DUSTBIN_SINKHORN.value,
            pooling,
        ),
        _arm(
            AblationArmName.KEYPOINT_DUAL_SOFTMAX,
            EvidenceMode.LOCAL,
            MatcherMode.DUAL_SOFTMAX.value,
            pooling,
        ),
        _arm(
            AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
            EvidenceMode.LOCAL,
            MatcherMode.DUSTBIN_SINKHORN.value,
            pooling,
        ),
        _arm(
            AblationArmName.FUSED,
            EvidenceMode.FUSED,
            MatcherMode.DUSTBIN_SINKHORN.value,
            pooling,
        ),
    )


def _planned_records(population, plan, phase="train", batch_index=0):
    source = (
        population.training_records
        if phase == "train"
        else population.validation_records
    )
    entry = plan.receipt["phases"][phase]["batches"][batch_index]
    return tuple(source[index] for index in entry["record_ordinals"])


def test_plan_is_deterministic_supervision_blind_and_path_free(provider_case):
    population, _factory, _inventory, _artifacts, opened, _locks = provider_case
    first = _build_plan(provider_case)
    second = _build_plan(provider_case)
    assert first.receipt == second.receipt
    assert first.canonical_file_sha256 == second.canonical_file_sha256
    assert LOCAL_Q1_BATCH_PLAN_SCHEMA_VERSION == "dunhuang-local-q1-batch-plan/0.4"
    assert LOCAL_Q1_BATCH_PROVIDER_VERSION == (
        "dunhuang-local-q1-read-only-provider/0.7"
    )
    assert first.receipt["schema_version"] == LOCAL_Q1_BATCH_PLAN_SCHEMA_VERSION
    assert first.receipt["provider_version"] == LOCAL_Q1_BATCH_PROVIDER_VERSION

    changed_train = (
        replace(population.training_records[0], label=False, direction_b_wrt_a=None),
        replace(population.training_records[1], label=True, direction_b_wrt_a="above"),
    )
    changed_val = tuple(
        replace(record, label=False, direction_b_wrt_a=None)
        for record in population.validation_records
    )
    changed_population = replace(
        population,
        training_records=changed_train,
        validation_records=changed_val,
    )
    changed_opened = replace(opened, population=changed_population)
    changed = _build_plan(provider_case, opened=changed_opened)
    assert changed.receipt == first.receipt

    encoded = _canonical(first.receipt).decode("utf-8")
    for forbidden in (
        "fragment/a",
        "masks/a.png",
        '"pair_id"',
        '"fragment_id"',
        '"dataset_id"',
        "/tmp/",
        "/Users/",
    ):
        assert forbidden not in encoded
    assert first.receipt["ordering_contract"]["supervision_fields_read"] == []
    assert first.receipt["ordering_contract"]["cross_arm_shared"] == [
        "local_dual_softmax",
        "local_dustbin_sinkhorn",
        "keypoint_dual_softmax",
        "keypoint_dustbin_sinkhorn",
        "fused",
    ]
    assert first.receipt["portable_privacy"] == {
        "raw_pair_component_member_or_path_identifiers_present": False,
        "anonymous_per_record_fields_present": [
            "ordinal",
            "geometry_commitment",
            "candidate_commitment",
            "candidate_count",
            "candidate_sequence_lengths",
            "max_sequence_a",
            "max_sequence_b",
            "bucket",
            "exact_costs",
        ],
        "anonymous_complexity_may_form_a_fingerprint": True,
        "cryptographic_privacy_claimed": False,
    }
    assert first.receipt["authority_boundary"] == {
        "planning_config_hash_arguments_source": (
            "self_provided_by_enclosing_cli_caller"
        ),
        "planning_config_independent_authority_verified_here": False,
        "enclosing_planning_authority_status": "pending",
        "result_bearing_authorized": False,
        "promotion_requirement": (
            "later_independent_run_authority_must_lock_batch_plan_file_and_"
            "content_sha256_before_any_result_bearing_execution"
        ),
    }
    first_batch = first.receipt["phases"]["train"]["batches"][0]
    assert first_batch["record_candidate_sequence_lengths"]
    assert first_batch["record_exact_costs"]


def test_plan_prefetches_unique_masks_in_stable_order_without_changing_bytes(
    tmp_path, provider_case, monkeypatch
):
    population, factory, inventory, artifacts, opened, locks = provider_case
    shuffled_population = replace(
        population,
        training_records=tuple(reversed(population.training_records)),
    )
    shuffled_opened = replace(opened, population=shuffled_population)

    def key(reference):
        return (
            reference.binding.logical_id,
            reference.binding.sha256,
            reference.archive_member,
            reference.threshold_rule,
        )

    expected_order = sorted(
        {
            key(reference)
            for record in shuffled_population.records
            for reference in (record.fragment_a, record.fragment_b)
        }
    )
    expected_legacy_order = list(
        dict.fromkeys(
            key(reference)
            for record in shuffled_population.records
            for reference in (record.fragment_a, record.fragment_b)
        )
    )

    class TracingFactory:
        def __init__(self):
            self.calls = []

        def __call__(self):
            raw = factory()

            def load(reference):
                self.calls.append(key(reference))
                return raw(reference)

            load.close = raw.close
            return load

    def build(tracing_factory):
        case = (
            shuffled_population,
            tracing_factory,
            inventory,
            artifacts,
            shuffled_opened,
            locks,
        )
        return _build_plan(case, opened=shuffled_opened)

    legacy_factory = TracingFactory()
    with monkeypatch.context() as patch:
        patch.setattr(
            provider_module, "_prefetch_planning_masks", lambda *_a, **_k: None
        )
        legacy = build(legacy_factory)
    assert legacy_factory.calls == expected_legacy_order
    assert expected_legacy_order != expected_order

    ordered_factory = TracingFactory()
    ordered = build(ordered_factory)
    assert ordered_factory.calls == expected_order
    assert len(ordered_factory.calls) == len(set(ordered_factory.calls))
    assert ordered.receipt == legacy.receipt
    assert ordered.content_sha256 == legacy.content_sha256
    assert ordered.canonical_file_sha256 == legacy.canonical_file_sha256
    legacy_path = tmp_path / "legacy-plan.json"
    ordered_path = tmp_path / "ordered-plan.json"
    write_local_q1_batch_plan(legacy_path, legacy)
    write_local_q1_batch_plan(ordered_path, ordered)
    assert ordered_path.read_bytes() == legacy_path.read_bytes()


def test_validation_assignment_is_label_blind_component_disjoint_and_private(
    provider_case,
):
    population = provider_case[0]
    assignment = freeze_local_q1_validation_assignment(population.validation_records)
    changed = tuple(
        replace(record, label=False, direction_b_wrt_a=None)
        for record in population.validation_records
    )
    changed_assignment = freeze_local_q1_validation_assignment(changed)
    assert assignment.receipt == changed_assignment.receipt
    assert assignment.phase_by_component == changed_assignment.phase_by_component
    assert assignment.receipt["schema_version"] == (
        LOCAL_Q1_VALIDATION_ASSIGNMENT_SCHEMA_VERSION
    )
    assert assignment.receipt["namespace"] == (LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE)
    assert assignment.receipt["supervision_fields_read"] == []
    assert set(assignment.record_ordinals_by_phase) == {
        "validation_select",
        "validation_calibration",
        "validation_report",
    }
    component_sets = {
        phase: {population.validation_records[index].component_id for index in ordinals}
        for phase, ordinals in assignment.record_ordinals_by_phase.items()
    }
    assert all(component_sets.values())
    assert not (
        component_sets["validation_select"] & component_sets["validation_calibration"]
    )
    assert not (
        component_sets["validation_select"] & component_sets["validation_report"]
    )
    assert not (
        component_sets["validation_calibration"] & component_sets["validation_report"]
    )
    assert set().union(*component_sets.values()) == {
        record.component_id for record in population.validation_records
    }
    encoded = _canonical(assignment.receipt).decode("utf-8")
    assert all(
        record.component_id not in encoded for record in population.validation_records
    )
    with pytest.raises(LocalQ1ProviderError, match="at least three components"):
        freeze_local_q1_validation_assignment(population.validation_records[:2])


def test_formal_plan_packs_each_validation_phase_independently(provider_case):
    population = provider_case[0]
    plan = _build_plan(provider_case, packing=2)
    assignment = freeze_local_q1_validation_assignment(population.validation_records)
    assert set(plan.receipt["phases"]) == {
        "train",
        "validation_select",
        "validation_calibration",
        "validation_report",
    }
    observed_ordinals = []
    for phase in (
        "validation_select",
        "validation_calibration",
        "validation_report",
    ):
        phase_receipt = plan.receipt["phases"][phase]
        assert phase_receipt["record_count"] > 0
        assert phase_receipt["batch_count"] > 0
        expected = set(assignment.record_ordinals_by_phase[phase])
        observed = {
            ordinal
            for batch in phase_receipt["batches"]
            for ordinal in batch["record_ordinals"]
        }
        assert observed == expected
        observed_ordinals.extend(observed)
        assert all(
            {
                assignment.phase_by_component[
                    population.validation_records[ordinal].component_id
                ]
                for ordinal in batch["record_ordinals"]
            }
            == {phase}
            for batch in phase_receipt["batches"]
        )
    assert sorted(observed_ordinals) == list(range(len(population.validation_records)))


def test_provider_exposes_only_complete_authoritative_phase_batches(provider_case):
    population, factory, inventory, _artifacts, opened, locks = provider_case
    geometry = _geometry()
    plan = _build_plan(provider_case, geometry=geometry)
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=factory,
        plan=plan,
        geometry_config=geometry,
        inventory_config=inventory,
        external_locks=_plan_locks(locks, plan),
    )
    arm = _local_arms()[0]
    for phase in (
        "validation_select",
        "validation_calibration",
        "validation_report",
    ):
        records = provider.planned_records(phase, 0)
        assert records == _planned_records(population, plan, phase=phase)
        prepared = provider.prepare(records, arm=arm, phase=phase)
        assert prepared.sample_count == len(records)
    wrong = provider.planned_records("validation_report", 0)
    with pytest.raises(LocalQ1ProviderError, match="crosses frozen component phases"):
        provider.prepare(wrong, arm=arm, phase="validation_select")
    with pytest.raises(LocalQ1ProviderError, match="four formal phases"):
        provider.prepare(wrong, arm=arm, phase="validation")


def test_planned_safety_metadata_is_typed_immutable_supervision_blind_and_lazy(
    provider_case,
):
    population, factory, inventory, _artifacts, opened, locks = provider_case
    geometry = _geometry()
    plan = _build_plan(provider_case, geometry=geometry)

    def build_provider(opened_value):
        return LocalQ1ReadOnlyBatchProvider(
            opened=opened_value,
            loader_factory=factory,
            plan=plan,
            geometry_config=geometry,
            inventory_config=inventory,
            external_locks=_plan_locks(locks, plan),
        )

    provider = build_provider(opened)

    def fail_planned_records(*_args, **_kwargs):
        raise AssertionError("planned_safety_metadata called planned_records")

    provider.planned_records = fail_planned_records

    def collect(value):
        return tuple(
            value.planned_safety_metadata(phase, ordinal)
            for phase in (
                "train",
                "validation_select",
                "validation_calibration",
                "validation_report",
            )
            for ordinal in range(plan.receipt["phases"][phase]["batch_count"])
        )

    original = collect(provider)
    assert original
    assert all(
        isinstance(value, FrozenLocalQ1PlannedSafetyMetadata) for value in original
    )
    assert all(
        value.receipt["schema_version"]
        == LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION
        for value in original
    )
    assert all(
        _canonical(value.receipt["privacy"]["provenance_safety_scan"])
        == _canonical(_portable_policy())
        and value.receipt["privacy"]["raw_provenance_keys_or_values_present"] is False
        and value.receipt["privacy"]["provenance_shape_or_scan_counts_present"] is False
        for value in original
    )
    assert all(
        type(row.sealed_real_test_marker_explicit_false) is bool
        and not row.sealed_scope_marker
        and not row.historical_test_marker
        for value in original
        for row in value.records
    )
    assert all(
        tuple(row.population_ordinal for row in value.records)
        == tuple(
            plan.receipt["phases"][value.receipt["phase"]]["batches"][
                value.receipt["batch_ordinal"]
            ]["record_ordinals"]
        )
        for value in original
    )
    assert all(
        value.receipt["batch_plan_file_sha256"] == plan.canonical_file_sha256
        and value.receipt["batch_plan_content_sha256"] == plan.content_sha256
        and value.receipt["validation_assignment_sha256"]
        == plan.validation_assignment["assignment_sha256"]
        for value in original
    )
    assert all(
        value.receipt["assignment_phase_component_set_sha256"]
        == plan.validation_assignment["phases"][value.receipt["phase"]][
            "component_set_sha256"
        ]
        == value.receipt["phase_component_set_sha256"]
        for value in original
        if value.receipt["phase"] != "train"
    )
    runtime = provider.portable_receipt()
    assert runtime["counts"]["planned_safety_metadata_call_count"] == len(original)
    assert runtime["counts"]["planned_records_call_count"] == 0
    assert runtime["safety_metadata_api"]["calls_planned_records"] is False

    first = original[0]
    with pytest.raises(TypeError):
        first.receipt["phase"] = "validation_report"
    with pytest.raises(TypeError):
        first.receipt["records"][0]["expected_split"] = "val"
    with pytest.raises(AttributeError):
        first.records[0].expected_split = "val"

    changed_train = (
        replace(population.training_records[0], label=False, direction_b_wrt_a=None),
        replace(population.training_records[1], label=True, direction_b_wrt_a="above"),
    )
    changed_validation = tuple(
        replace(record, label=False, direction_b_wrt_a=None)
        for record in population.validation_records
    )
    changed_provider = build_provider(
        replace(
            opened,
            population=replace(
                population,
                training_records=changed_train,
                validation_records=changed_validation,
            ),
        )
    )
    changed = collect(changed_provider)
    assert tuple(_canonical(value.receipt) for value in changed) == tuple(
        _canonical(value.receipt) for value in original
    )

    class GuardedTrainingPairRecord(TrainingPairRecord):
        def __getattribute__(self, name):
            forbidden = {
                "label",
                "direction_b_wrt_a",
                "fragment_a",
                "fragment_b",
                "pair_id",
                "canonical_pair_key",
                "dataset_id",
                "canonical_group_id",
                "label_origin",
                "static_hard_negative_score",
            }
            state = object.__getattribute__(self, "__dict__")
            if state.get("_forbid_safety_metadata_access") and name in forbidden:
                raise AssertionError("forbidden full-record field was accessed")
            return super().__getattribute__(name)

    guarded_ordinal = plan.receipt["phases"]["train"]["batches"][0]["record_ordinals"][
        0
    ]
    guarded_source = population.training_records[guarded_ordinal]
    guarded = GuardedTrainingPairRecord(**dict(guarded_source.__dict__))
    object.__setattr__(guarded, "_forbid_safety_metadata_access", True)
    guarded_train = list(population.training_records)
    guarded_train[guarded_ordinal] = guarded
    guarded_provider = build_provider(
        replace(
            opened,
            population=replace(
                population,
                training_records=tuple(guarded_train),
            ),
        )
    )
    guarded_provider.planned_safety_metadata("train", 0)


@pytest.mark.parametrize(
    ("nested", "sealed", "historical"),
    (
        ({"real_dunhuang_sealed_test": True}, True, False),
        ([{" Sealed_Holdout ": True}], True, False),
        ({"Historical_Test": True}, False, True),
        ({"historical_eval_test": 1}, False, True),
        ([{"source_split": "Historical_Test"}], False, True),
        ({"withheld_test": "0"}, False, True),
        ({"Sealed_Holdout": " NO ", "Historical_Test": 0}, False, False),
        ({"real_dunhuang_sealed_test": False}, False, False),
    ),
)
def test_shared_recursive_provenance_scanner_normalizes_nested_markers(
    nested,
    sealed,
    historical,
):
    markers = scan_local_q1_provenance_safety(
        {
            "real_dunhuang_sealed_test": False,
            "nested": nested,
        }
    )
    assert markers.top_level_sealed_marker_explicit_false is True
    assert markers.sealed_scope_marker is sealed
    assert markers.historical_test_marker is historical

    missing = scan_local_q1_provenance_safety({"nested": {"safe": False}})
    assert missing.top_level_sealed_marker_explicit_false is False
    assert missing.sealed_scope_marker is False


def test_shared_recursive_provenance_scanner_budgets_and_types_fail_closed():
    cycle = {}
    cycle["child"] = cycle
    cycle_list = []
    cycle_list.append(cycle_list)

    too_deep = False
    for _ in range(LOCAL_Q1_PROVENANCE_SAFETY_MAX_DEPTH + 2):
        too_deep = {"child": too_deep}

    too_many = [None] * LOCAL_Q1_PROVENANCE_SAFETY_MAX_NODES
    invalid = (
        {"real_dunhuang_sealed_test": False, "nested": cycle},
        {"real_dunhuang_sealed_test": False, "nested": cycle_list},
        {"real_dunhuang_sealed_test": False, "nested": too_deep},
        {"real_dunhuang_sealed_test": False, "nested": too_many},
        {"real_dunhuang_sealed_test": False, "nested": {"unsupported"}},
        {"real_dunhuang_sealed_test": False, 3: "non-string-key"},
        {"real_dunhuang_sealed_test": False, "value": float("nan")},
        {"real_dunhuang_sealed_test": False, "value": object()},
    )
    for provenance in invalid:
        with pytest.raises(LocalQ1ProviderError, match="provenance safety scan"):
            scan_local_q1_provenance_safety(provenance)


def test_safety_metadata_changes_for_scope_fields_and_detects_component_overlap(
    provider_case,
):
    population, factory, inventory, _artifacts, opened, locks = provider_case
    geometry = _geometry()
    plan = _build_plan(provider_case, geometry=geometry)

    def build_provider(population_value):
        return LocalQ1ReadOnlyBatchProvider(
            opened=replace(opened, population=population_value),
            loader_factory=factory,
            plan=plan,
            geometry_config=geometry,
            inventory_config=inventory,
            external_locks=_plan_locks(locks, plan),
        )

    original_provider = build_provider(population)
    train_entry = plan.receipt["phases"]["train"]["batches"][0]
    target_ordinal = train_entry["record_ordinals"][0]
    original_metadata = original_provider.planned_safety_metadata("train", 0)
    target = population.training_records[target_ordinal]
    validation_component = population.validation_records[0].component_id
    changed_record = replace(
        target,
        fragment_a=replace(target.fragment_a, component_id=validation_component),
        fragment_b=replace(target.fragment_b, component_id=validation_component),
        component_id=validation_component,
    )
    changed_train = list(population.training_records)
    changed_train[target_ordinal] = changed_record
    component_provider = build_provider(
        replace(population, training_records=tuple(changed_train))
    )
    component_metadata = component_provider.planned_safety_metadata("train", 0)
    assert component_metadata.content_sha256 != original_metadata.content_sha256

    tokens_by_phase = {
        phase: {
            row.component_token_sha256
            for ordinal in range(plan.receipt["phases"][phase]["batch_count"])
            for row in component_provider.planned_safety_metadata(
                phase, ordinal
            ).records
        }
        for phase in (
            "train",
            "validation_select",
            "validation_calibration",
            "validation_report",
        )
    }
    validation_tokens = set().union(
        tokens_by_phase["validation_select"],
        tokens_by_phase["validation_calibration"],
        tokens_by_phase["validation_report"],
    )
    assert tokens_by_phase["train"] & validation_tokens
    assert not (
        tokens_by_phase["validation_select"] & tokens_by_phase["validation_calibration"]
    )
    assert not (
        tokens_by_phase["validation_select"] & tokens_by_phase["validation_report"]
    )
    assert not (
        tokens_by_phase["validation_calibration"] & tokens_by_phase["validation_report"]
    )

    changed_provenance = dict(target.provenance)
    changed_provenance["real_dunhuang_sealed_test"] = False
    changed_provenance["source_split"] = "historical_test"
    changed_provenance["nested_scope"] = [{"Sealed_Holdout": True}]
    provenance_train = list(population.training_records)
    provenance_train[target_ordinal] = replace(target, provenance=changed_provenance)
    provenance_metadata = build_provider(
        replace(population, training_records=tuple(provenance_train))
    ).planned_safety_metadata("train", 0)
    provenance_row = next(
        row
        for row in provenance_metadata.records
        if row.population_ordinal == target_ordinal
    )
    assert provenance_row.sealed_real_test_marker_explicit_false is True
    assert provenance_row.sealed_scope_marker is True
    assert provenance_row.historical_test_marker is True
    assert provenance_metadata.content_sha256 != original_metadata.content_sha256

    ignored_provenance = dict(target.provenance)
    ignored_provenance.update(
        {
            "private_nested": [
                {"pair_id": "raw-pair-id", "member_path": "/tmp/private-mask.png"}
            ]
        }
    )
    ignored_train = list(population.training_records)
    ignored_train[target_ordinal] = replace(target, provenance=ignored_provenance)
    ignored_metadata = build_provider(
        replace(population, training_records=tuple(ignored_train))
    ).planned_safety_metadata("train", 0)
    assert _canonical(ignored_metadata.receipt) == _canonical(original_metadata.receipt)
    encoded = _canonical(ignored_metadata.receipt).decode("utf-8")
    assert "raw-pair-id" not in encoded
    assert "/tmp/private-mask.png" not in encoded
    assert validation_component not in encoded


def test_planned_safety_metadata_tamper_and_extra_fields_fail_closed(provider_case):
    population, factory, inventory, _artifacts, opened, locks = provider_case
    geometry = _geometry()
    plan = _build_plan(provider_case, geometry=geometry)
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=factory,
        plan=plan,
        geometry_config=geometry,
        inventory_config=inventory,
        external_locks=_plan_locks(locks, plan),
    )
    metadata = provider.planned_safety_metadata("train", 0)

    def resign(value):
        value.pop("content_sha256", None)
        value["content_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
        return FrozenLocalQ1PlannedSafetyMetadata(
            receipt=value,
            content_sha256=value["content_sha256"],
        )

    tampered = json.loads(_canonical(metadata.receipt))
    tampered["records"][0]["population_ordinal"] += 1
    with pytest.raises(LocalQ1ProviderError, match="order/component commitments"):
        resign(tampered)

    extra = json.loads(_canonical(metadata.receipt))
    extra["records"][0]["unexpected"] = 0
    with pytest.raises(LocalQ1ProviderError, match="missing or extra"):
        resign(extra)

    identity = json.loads(_canonical(metadata.receipt))
    identity["records"][0]["pair_id"] = "raw-pair-id"
    with pytest.raises(LocalQ1ProviderError, match="identity"):
        resign(identity)

    scanner_policy = json.loads(_canonical(metadata.receipt))
    scanner_policy["privacy"]["provenance_safety_scan"]["max_depth"] += 1
    with pytest.raises(LocalQ1ProviderError, match="privacy changed"):
        resign(scanner_policy)


def test_padding_aware_packing_respects_exact_boundary_and_never_truncates(
    provider_case,
):
    one_per_batch = _build_plan(provider_case, packing=1)
    train_batches = one_per_batch.receipt["phases"]["train"]["batches"]
    assert len(train_batches) == 2
    individual_attention = [
        batch["complexity"]["attention_score_elements_per_head"]
        for batch in train_batches
    ]
    exact_boundary = max(individual_attention)
    bounded_geometry = _geometry(max_attention_score_elements_per_batch=exact_boundary)
    bounded = _build_plan(provider_case, geometry=bounded_geometry, packing=2)
    assert bounded.receipt["phases"]["train"]["batch_count"] == 2
    assert (
        bounded.receipt["totals"]["candidate_count"]
        == (one_per_batch.receipt["totals"]["candidate_count"])
    )
    assert all(
        batch["complexity"]["attention_score_elements_per_head"] <= exact_boundary
        for phase in (
            "train",
            "validation_select",
            "validation_calibration",
            "validation_report",
        )
        for batch in bounded.receipt["phases"][phase]["batches"]
    )

    with pytest.raises(LocalQ1ProviderError, match="cannot fit"):
        _build_plan(
            provider_case,
            geometry=_geometry(
                max_attention_score_elements_per_batch=exact_boundary - 1
            ),
            packing=2,
        )
    maximum_candidates = max(
        max(batch["record_candidate_counts"])
        for phase in (
            "train",
            "validation_select",
            "validation_calibration",
            "validation_report",
        )
        for batch in one_per_batch.receipt["phases"][phase]["batches"]
    )
    with pytest.raises(LocalQ1ProviderError, match="max_candidates_per_sample"):
        _build_plan(
            provider_case,
            geometry=_geometry(max_candidates_per_sample=maximum_candidates - 1),
            packing=1,
        )


def test_provider_reuses_plan_with_shared_tensors_within_each_representation(
    provider_case,
):
    population, factory, inventory, _artifacts, opened, locks = provider_case
    geometry = _geometry()
    plan = _build_plan(provider_case, geometry=geometry)
    trusted = _plan_locks(locks, plan)
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=factory,
        plan=plan,
        geometry_config=geometry,
        inventory_config=inventory,
        external_locks=trusted,
    )
    records = _planned_records(population, plan)
    prepared = [
        provider.prepare(records, arm=arm, phase="train") for arm in _local_arms()
    ]
    by_representation = {}
    for value in prepared:
        by_representation.setdefault(value.candidate_representation, []).append(value)
    assert set(by_representation) == {
        "multirun_sliding_window",
        "contour_keypoint",
    }
    assert all(
        len({value.prepared_input_sha256 for value in values}) == 1
        and len({value.local_candidate_sha256 for value in values}) == 1
        and len({tuple(value.payload.candidate_ids) for value in values}) == 1
        for values in by_representation.values()
    )
    assert (
        by_representation["multirun_sliding_window"][0].prepared_input_sha256
        != by_representation["contour_keypoint"][0].prepared_input_sha256
    )
    keypoint = by_representation["contour_keypoint"][0].payload
    assert (
        keypoint.model_inputs()["correspondence_mask"] is keypoint.correspondence_mask
    )
    assert keypoint.correspondence_mask.any().item()
    assert (~keypoint.correspondence_mask).any().item()
    assert all(
        int((keypoint.sample_index == index).sum().item()) == 4
        for index in range(keypoint.batch_size)
    )
    assert all(
        local_q1_prepared_digests(value.payload)
        == (value.prepared_input_sha256, value.local_candidate_sha256)
        for value in prepared
    )
    assert all(
        value.processing_counts["geometry_build_count"] == 0 for value in prepared
    )
    assert all(
        value.processing_counts["geometry_cache_write_count"] == 0 for value in prepared
    )
    assert all(
        value.processing_counts["geometry_cache_read_count"] > 0 for value in prepared
    )
    receipt = provider.portable_receipt()
    assert LOCAL_Q1_PROVIDER_RUNTIME_SCHEMA_VERSION == (
        "dunhuang-local-q1-provider-runtime/0.7"
    )
    assert receipt["schema_version"] == LOCAL_Q1_PROVIDER_RUNTIME_SCHEMA_VERSION
    assert receipt["provider_version"] == LOCAL_Q1_BATCH_PROVIDER_VERSION
    assert receipt["counts"]["prepare_call_count"] == 5
    assert receipt["counts"]["cache_miss_count"] == 0
    assert receipt["counts"]["cache_write_count"] == 0
    assert receipt["counts"]["geometry_build_count"] == 0
    assert receipt["counts"]["planned_safety_metadata_call_count"] == 0
    assert receipt["counts"]["planned_records_call_count"] == 0
    assert receipt["safety_metadata_api"] == {
        "schema_version": LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION,
        "method": "planned_safety_metadata",
        "return_type": "FrozenLocalQ1PlannedSafetyMetadata",
        "source": "plan_ordinals_and_authoritative_population_safety_fields",
        "calls_planned_records": False,
        "materializes_masks_or_batches": False,
        "provenance_safety_scan": _portable_policy(),
    }
    assert set(receipt["arm_prepare_call_counts"]) == {
        "local_dual_softmax",
        "local_dustbin_sinkhorn",
        "keypoint_dual_softmax",
        "keypoint_dustbin_sinkhorn",
        "fused",
    }

    reversed_records = tuple(reversed(records))
    with pytest.raises(LocalQ1ProviderError, match="frozen batch"):
        provider.prepare(reversed_records, arm=_local_arms()[0], phase="train")


def test_candidate_digest_is_supervision_blind_but_prepared_digest_is_not(
    provider_case,
):
    population, factory, inventory, _artifacts, opened, locks = provider_case
    geometry = _geometry()
    plan = _build_plan(provider_case, geometry=geometry)
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=factory,
        plan=plan,
        geometry_config=geometry,
        inventory_config=inventory,
        external_locks=_plan_locks(locks, plan),
    )
    records = _planned_records(population, plan)
    arm = _local_arms()[0]
    original = provider.prepare(records, arm=arm, phase="train")
    changed_records = tuple(
        replace(record, label=False, direction_b_wrt_a=None)
        if record.label
        else replace(record, label=True, direction_b_wrt_a="above")
        for record in records
    )
    changed = provider.prepare(changed_records, arm=arm, phase="train")

    # Population planning and local geometry remain supervision-blind, while
    # the complete prepared payload binds labels and direction supervision.
    assert original.local_candidate_sha256 == changed.local_candidate_sha256
    assert original.prepared_input_sha256 != changed.prepared_input_sha256
    assert original.record_sequence_sha256 != changed.record_sequence_sha256
    assert plan.receipt == _build_plan(provider_case, geometry=geometry).receipt


@pytest.mark.parametrize(
    "field_name,index",
    (
        ("candidate_valid", (0,)),
        ("direction_slot_valid", (0, 0)),
        ("geometry_valid", (0,)),
    ),
)
def test_candidate_and_prepared_digests_bind_model_validity(
    provider_case, field_name, index
):
    population, factory, inventory, _artifacts, opened, locks = provider_case
    geometry = _geometry()
    plan = _build_plan(provider_case, geometry=geometry)
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=factory,
        plan=plan,
        geometry_config=geometry,
        inventory_config=inventory,
        external_locks=_plan_locks(locks, plan),
    )
    prepared = provider.prepare(
        _planned_records(population, plan),
        arm=_local_arms()[0],
        phase="train",
    )
    batch = prepared.payload
    fields = dict(batch.__dict__)
    changed_validity = getattr(batch, field_name).clone()
    changed_validity[index] = ~changed_validity[index]
    fields[field_name] = changed_validity
    tampered = SimpleNamespace(**fields)

    changed_prepared_sha, changed_local_sha = local_q1_prepared_digests(tampered)
    assert changed_local_sha != prepared.local_candidate_sha256
    assert changed_prepared_sha != prepared.prepared_input_sha256


def _resigned(receipt):
    value = copy.deepcopy(dict(receipt))
    value.pop("content_sha256", None)
    value["content_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return FrozenLocalQ1BatchPlan(
        receipt=value,
        content_sha256=value["content_sha256"],
        canonical_file_sha256=hashlib.sha256(_canonical(value)).hexdigest(),
    )


def test_plan_recomputes_complexity_resources_and_totals_after_resigning(
    provider_case,
):
    plan = _build_plan(provider_case)
    tampered = copy.deepcopy(dict(plan.receipt))
    batch = tampered["phases"]["train"]["batches"][0]

    # Make the forged resource hierarchy internally self-consistent.  The
    # validator must still derive the original tensor/work values from the
    # trusted configs and anonymous candidate sequence lengths.
    batch["complexity"]["local_tensor_elements"] += 1
    resource = batch["resource_estimate"]
    resource["tensor_bytes"] += 4
    resource["conservative_peak_bytes"] += 4
    resource["work_elements"] += 1
    rate = tampered["resource_estimate"]["planning_elements_per_second"]
    safety = tampered["resource_estimate"]["time_safety_factor"]
    resource["estimated_seconds_nominal"] = resource["work_elements"] / float(rate)
    resource["estimated_seconds_upper"] = resource["estimated_seconds_nominal"] * safety
    resources = [
        value["resource_estimate"]
        for phase in tampered["phases"].values()
        for value in phase["batches"]
    ]
    root = tampered["resource_estimate"]
    root["estimated_peak_bytes"] = max(
        value["conservative_peak_bytes"] for value in resources
    )
    root["total_work_elements"] = sum(value["work_elements"] for value in resources)
    root["estimated_seconds_nominal"] = sum(
        value["estimated_seconds_nominal"] for value in resources
    )
    root["estimated_seconds_upper"] = sum(
        value["estimated_seconds_upper"] for value in resources
    )

    with pytest.raises(LocalQ1ProviderError, match="exact recomputation"):
        _resigned(tampered)


def test_plan_rejects_nested_unknown_negative_and_wrong_type_after_resigning(
    provider_case,
):
    plan = _build_plan(provider_case)
    cases = (
        (
            ("phases", "train", "batches", 0, "complexity", "unexpected"),
            0,
        ),
        (
            (
                "phases",
                "train",
                "batches",
                0,
                "complexity",
                "candidate_count",
            ),
            -1,
        ),
        (
            (
                "phases",
                "train",
                "batches",
                0,
                "complexity",
                "record_count",
            ),
            True,
        ),
        (
            (
                "phases",
                "train",
                "batches",
                0,
                "resource_estimate",
                "unexpected",
            ),
            0,
        ),
        (
            (
                "phases",
                "train",
                "batches",
                0,
                "resource_estimate",
                "work_elements",
            ),
            -1,
        ),
        (
            (
                "phases",
                "train",
                "batches",
                0,
                "resource_estimate",
                "estimated_seconds_nominal",
            ),
            0,
        ),
        (("resource_estimate", "unexpected"), 0),
        (("resource_estimate", "kind"), "measured_runtime"),
        (("resource_estimate", "estimated_peak_bytes"), False),
        (("resource_estimate", "estimated_seconds_upper"), -1.0),
        (("totals", "batch_count"), True),
    )
    for path, replacement in cases:
        tampered = copy.deepcopy(dict(plan.receipt))
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        with pytest.raises(LocalQ1ProviderError):
            _resigned(tampered)


def test_plan_rejects_forged_shapes_costs_slices_and_identity_injection(
    provider_case,
):
    plan = _build_plan(provider_case)

    for replacement in (-1, "/tmp/private-mask.png"):
        tampered = copy.deepcopy(dict(plan.receipt))
        batch = tampered["phases"]["train"]["batches"][0]
        batch["record_candidate_sequence_lengths"][0][0][0] = replacement
        with pytest.raises(LocalQ1ProviderError):
            _resigned(tampered)

    tampered = copy.deepcopy(dict(plan.receipt))
    batch = tampered["phases"]["train"]["batches"][0]
    batch["record_exact_costs"][0]["attention_elements"] += 1
    with pytest.raises(LocalQ1ProviderError, match="complexity was not recomputed"):
        _resigned(tampered)

    slice_cases = (
        ("candidate_slice", "candidate_count"),
        ("sequence_a_token_slice", "sequence_a_token_count"),
        ("sequence_b_token_slice", "sequence_b_token_count"),
    )
    for slice_name, total_name in slice_cases:
        tampered = copy.deepcopy(dict(plan.receipt))
        phase = tampered["phases"]["train"]
        batch = phase["batches"][0]
        batch[slice_name][1] += 1
        phase[slice_name][1] += 1
        tampered["totals"][total_name] += 1
        with pytest.raises(LocalQ1ProviderError, match="slice"):
            _resigned(tampered)

    for injected_key, injected_value in (
        ("path", "/tmp/private-mask.png"),
        ("pair_id", "raw-pair-id"),
        ("component_id", "raw-component-id"),
    ):
        tampered = copy.deepcopy(dict(plan.receipt))
        batch = tampered["phases"]["train"]["batches"][0]
        batch["resource_estimate"][injected_key] = injected_value
        with pytest.raises(LocalQ1ProviderError, match="path|identity"):
            _resigned(tampered)


def test_plan_external_reopen_and_tamper_missing_extra_fail_closed(
    tmp_path, provider_case
):
    _population_value, _factory, _inventory, _artifacts, _opened, locks = provider_case
    plan = _build_plan(provider_case)
    trusted = _plan_locks(locks, plan)
    path = tmp_path / "batch-plan.json"
    assert write_local_q1_batch_plan(path, plan) == plan.canonical_file_sha256
    reopened = reopen_local_q1_batch_plan(path, external_locks=trusted)
    assert reopened.receipt == plan.receipt

    tampered = bytearray(path.read_bytes())
    tampered[-1] ^= 1
    path.write_bytes(tampered)
    with pytest.raises(LocalQ1ProviderError, match="file hash mismatch"):
        reopen_local_q1_batch_plan(path, external_locks=trusted)

    missing = copy.deepcopy(dict(plan.receipt))
    missing["phases"]["train"]["batches"][0]["record_ordinals"].pop()
    with pytest.raises(LocalQ1ProviderError, match="record vectors|missing/extra"):
        _resigned(missing)

    extra = copy.deepcopy(dict(plan.receipt))
    extra["phases"]["train"]["batches"][0]["record_ordinals"].append(999)
    with pytest.raises(LocalQ1ProviderError, match="record vectors|missing/extra"):
        _resigned(extra)

    forged_assignment = copy.deepcopy(dict(plan.receipt))
    forged_assignment["validation_assignment"]["assignment_sha256"] = "f" * 64
    resigned_assignment = _resigned(forged_assignment)
    with pytest.raises(LocalQ1ProviderError, match="validation assignment"):
        LocalQ1ReadOnlyBatchProvider(
            opened=provider_case[4],
            loader_factory=provider_case[1],
            plan=resigned_assignment,
            geometry_config=_geometry(),
            inventory_config=provider_case[2],
            external_locks=_plan_locks(locks, resigned_assignment),
        )

    with pytest.raises(LocalQ1ProviderError, match="external lock mismatch"):
        build_local_q1_batch_plan(
            opened=provider_case[4],
            loader_factory=provider_case[1],
            geometry_config=_geometry(),
            inventory_config=provider_case[2],
            external_locks=replace(locks, freeze_file_sha256="f" * 64),
            plan_config=LocalQ1BatchPlanConfig(packing_max_records=2),
        )

    changed_source_locks = replace(locks, source_bundle_manifest_sha256="f" * 64)
    with pytest.raises(LocalQ1ProviderError, match="external lock mismatch"):
        build_local_q1_batch_plan(
            opened=provider_case[4],
            loader_factory=provider_case[1],
            geometry_config=_geometry(),
            inventory_config=provider_case[2],
            external_locks=changed_source_locks,
            plan_config=LocalQ1BatchPlanConfig(packing_max_records=2),
        )

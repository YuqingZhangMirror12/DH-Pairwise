from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from staging.pairwise_v0_2.geometry import (
    DEFAULT_DIRECTION_ORDER,
    CandidateBuilderConfig,
    build_fragment_geometry,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.preflight.local_q1_geometry_qualification import (
    CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH,
    CANONICAL_ROUTE_A_POLICY_PATH,
    ROUTE_A_POLICY_CONTENT_SHA256,
    ROUTE_A_POLICY_FILE_SHA256,
    ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256,
    ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256,
    LazyLocalQ1GeometryQualifier,
    LocalQ1EligibilityDecision,
    LocalQ1EligibilityTrust,
    LocalQ1ExternalFileLock,
    LocalQ1GeometryQualificationError,
    LocalQ1PairGeometryInput,
    build_untrusted_eligibility_index,
    build_untrusted_eligibility_receipt,
    canonical_eligibility_bytes,
    load_local_q1_geometry_eligibility_authority,
    load_local_q1_geometry_eligibility_authority_payloads as load_local_q1_geometry_eligibility_authority_legacy_payloads,
    load_local_q1_geometry_eligibility_authority_payloads_v2 as load_local_q1_geometry_eligibility_authority_payloads,
    load_local_q1_route_a_config_authority,
    load_local_q1_route_a_config_authority_payload,
    load_local_q1_route_a_policy_payload,
)
from staging.pairwise_v0_2.preflight import (
    local_q1_geometry_eligibility_authority as lightweight_authority,
)
from staging.pairwise_v0_2.preflight.local_q1_selection_frontier import (
    LocalQ1FrontierCandidate,
    LocalQ1SelectionFrontier,
)
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.local_q1_pair_qualification import (
    LocalQ1PairQualification,
    qualify_local_q1_pair_geometry,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _geometry() -> CandidateBuilderConfig:
    return CandidateBuilderConfig(
        window_scale_fractions=(0.16,),
        window_min_px=2.0,
        window_max_px=64.0,
        output_size=(10, 12),
        min_run_length_fraction=0.0,
        min_run_length_px=2.0,
        side_resample_count=20,
    )


def _mask(left: int, *, notch: bool = False) -> np.ndarray:
    value = np.zeros((96, 112), dtype=bool)
    value[12:78, left : left + 38] = True
    if notch:
        value[33:53, left + 27 : left + 38] = False
    return value


def _eligible_assessment() -> LocalQ1PairQualification:
    return LocalQ1PairQualification(
        eligible=True,
        failure_stage=None,
        failure_reason=None,
        fragment_statuses=("ok", "ok"),
        pair_status="ok",
        emitted_directions=tuple(
            direction.value for direction in DEFAULT_DIRECTION_ORDER
        ),
        candidate_count=4,
        max_sequence_a=8,
        max_sequence_b=8,
        local_tensor_elements=1,
        attention_elements_per_head=1,
        affinity_elements=1,
        sinkhorn_elements=1,
        candidate_semantic_sha256="a" * 64,
    )


def _ineligible_assessment() -> LocalQ1PairQualification:
    return LocalQ1PairQualification(
        eligible=False,
        failure_stage="fragment",
        failure_reason="fragment_geometry_not_ok:largest_component_below_minimum_pixels",
        fragment_statuses=("invalid_contour", "ok"),
        pair_status="not_evaluated",
        emitted_directions=(),
        candidate_count=0,
        max_sequence_a=0,
        max_sequence_b=0,
        local_tensor_elements=0,
        attention_elements_per_head=0,
        affinity_elements=0,
        sinkhorn_elements=0,
        candidate_semantic_sha256="b" * 64,
    )


def _record(
    token: str,
    *,
    dataset: str,
    split: str,
    component: str,
    label: bool,
    direction=None,
) -> TrainingPairRecord:
    binding = ArchiveBinding(
        logical_id="fixture://route-a/{}".format(dataset),
        archive_format="zip",
        sha256=_sha("binding-" + dataset),
    )

    def reference(endpoint: str) -> MaskMemberRef:
        return MaskMemberRef(
            binding=binding,
            archive_member="{}/{}_{}.png".format(split, token, endpoint),
            fragment_id="fragment/{}/{}_{}".format(split, token, endpoint),
            dataset_id=dataset,
            canonical_group_id="group/{}".format(component),
            component_id=component,
            split=split,
            threshold_rule="grayscale_uint8_gt_127",
            content_sha256=_sha("content-{}-{}".format(token, endpoint)),
        )

    first = reference("a")
    second = reference("b")
    return TrainingPairRecord(
        fragment_a=first,
        fragment_b=second,
        label=label,
        direction_b_wrt_a=direction,
        dataset_id=dataset,
        canonical_group_id="group/{}".format(component),
        component_id=component,
        split=split,
        canonical_pair_key=tuple(sorted((first.fragment_id, second.fragment_id))),
        label_origin="fixture",
        provenance={"real_dunhuang_sealed_test": False},
    )


def _candidate(
    record: TrainingPairRecord,
    *,
    dataset_name: str,
    component_rank: str,
    record_rank: str,
    ordinal: int,
) -> LocalQ1FrontierCandidate:
    token = _sha("pair-token-" + record.pair_id)
    return LocalQ1FrontierCandidate(
        record=record,
        dataset_name=dataset_name,
        component_token=_sha("component-" + record.component_id),
        pair_identity_token=token,
        record_token=_sha("record-{}-{}".format(record.pair_id, record.label)),
        record_rank=record_rank,
        component_rank=component_rank,
        source_ordinal=ordinal,
    )


def _frontier_result(tmp_path, *, relabel_validation: bool = False):
    database = tmp_path / "frontier.sqlite3"
    ineligible_tokens = set()
    ordinal = 0
    with LocalQ1SelectionFrontier(database) as frontier:
        for dataset_name, record_dataset in (
            ("mm_augmented", "mm_augmented"),
            ("eccv_1113data", "eccv_1113data"),
            ("canonical_new", "dunhuang_voronoi_masks_no_erode_v0_2"),
        ):
            for label in (False, True):
                component = "train/{}/{}".format(dataset_name, int(label))
                first = _candidate(
                    _record(
                        "{}-{}-first".format(dataset_name, int(label)),
                        dataset=record_dataset,
                        split="train",
                        component=component,
                        label=label,
                        direction="right" if label else None,
                    ),
                    dataset_name=dataset_name,
                    component_rank="1" * 64,
                    record_rank="0" * 64,
                    ordinal=ordinal,
                )
                ordinal += 1
                second = _candidate(
                    _record(
                        "{}-{}-second".format(dataset_name, int(label)),
                        dataset=record_dataset,
                        split="train",
                        component=component,
                        label=label,
                        direction="above" if label else None,
                    ),
                    dataset_name=dataset_name,
                    component_rank="1" * 64,
                    record_rank="1" * 64,
                    ordinal=ordinal,
                )
                ordinal += 1
                frontier.add(first)
                frontier.add(second)
                if dataset_name == "mm_augmented" and not label:
                    ineligible_tokens.add(first.pair_identity_token)

        for dataset_name in ("mm_augmented", "eccv_1113data"):
            component = "val/{}".format(dataset_name)
            first_record = _record(
                "{}-val-first".format(dataset_name),
                dataset=dataset_name,
                split="val",
                component=component,
                label=False,
            )
            second_record = _record(
                "{}-val-second".format(dataset_name),
                dataset=dataset_name,
                split="val",
                component=component,
                label=True,
                direction="left",
            )
            if relabel_validation:
                first_record = replace(
                    first_record,
                    label=True,
                    direction_b_wrt_a="above",
                )
                second_record = replace(
                    second_record,
                    label=False,
                    direction_b_wrt_a=None,
                )
            first = _candidate(
                first_record,
                dataset_name=dataset_name,
                component_rank="2" * 64,
                record_rank="0" * 64,
                ordinal=ordinal,
            )
            ordinal += 1
            second = _candidate(
                second_record,
                dataset_name=dataset_name,
                component_rank="2" * 64,
                record_rank="1" * 64,
                ordinal=ordinal,
            )
            ordinal += 1
            frontier.add(first)
            frontier.add(second)
            if dataset_name == "mm_augmented":
                ineligible_tokens.add(first.pair_identity_token)

        def decide(item):
            return LocalQ1EligibilityDecision(
                pair_identity_sha256=item.pair_identity_sha256,
                assessment=(
                    _ineligible_assessment()
                    if item.pair_identity_sha256 in ineligible_tokens
                    else _eligible_assessment()
                ),
            )

        result = frontier.select(
            decide=decide,
            train_datasets=("mm_augmented", "eccv_1113data", "canonical_new"),
            train_target_per_dataset_label=1,
            validation_caps={"mm_augmented": 1, "eccv_1113data": 1},
        )
    return result


def _documents(tmp_path):
    result = _frontier_result(tmp_path)
    config_authority = load_local_q1_route_a_config_authority()
    config = config_authority.geometry_config
    index = build_untrusted_eligibility_index(result.decisions, geometry_config=config)
    index_bytes = canonical_eligibility_bytes(index)
    index_path = tmp_path / "eligibility_index.json"
    index_path.write_bytes(index_bytes)
    predecessor_file = "1" * 64
    predecessor_content = "2" * 64
    roles = {
        role: LocalQ1ExternalFileLock(
            bytes=index + 1,
            sha256=predecessor_file
            if role == "freeze_receipt"
            else str(index + 2) * 64,
        )
        for index, role in enumerate(
            (
                "freeze_receipt",
                "mm_archive",
                "eccv_archive",
                "mm_fingerprint_cache",
                "eccv_fingerprint_cache",
                "historical_split",
                "synthetic_manifest",
                "synthetic_archive",
            )
        )
    }
    geometry_hash = index["qualification_contract"]["geometry_batch_config_sha256"]
    receipt = build_untrusted_eligibility_receipt(
        index=index,
        selection_proof=result.selection_proof,
        predecessor_freeze_file_sha256=predecessor_file,
        predecessor_freeze_content_sha256=predecessor_content,
        input_role_locks=roles,
        source_bundle_manifest_sha256="c" * 64,
        config_authority=config_authority,
        qualification_scratch_cache={
            "cache_role": "new_local_scratch_output_never_membership_input",
            "fresh_path_atomically_claimed": True,
            "initial_artifact_count": 0,
            "external_preexisting_artifact_count": 0,
            "predecessor_partial_cache_role_count": 0,
            "predecessor_partial_cache_tree_accessed": False,
            "mask_decode_count": len(result.decisions) * 2,
            "scratch_cache_hit_count": 0,
            "scratch_cache_miss_count": len(result.decisions) * 2,
            "final_artifact_count": len(result.decisions) * 2,
            "final_artifact_file_bytes": len(result.decisions) * 2,
            "scratch_cache_artifact_set_sha256": "f" * 64,
            "selected_population_guard": {
                "inventory_config_sha256": (
                    "0270d5b7d7f042943cd52be28d6665ada11a037880fe81abf38f6d7fd008c123"
                ),
                "selected_record_count": (
                    len(result.training_records) + len(result.validation_records)
                ),
                "selected_pair_commitment_order_sha256": result.selection_proof[
                    "selected_pair_commitment_order_sha256"
                ],
                "selected_pair_commitment_set_sha256": result.selection_proof[
                    "selected_pair_commitment_set_sha256"
                ],
                "selected_unique_reference_count": 1,
                "selected_max_mask_pixels_per_reference": 1,
                "max_records_bound": 20_000,
                "max_unique_references_bound": 40_000,
                "max_mask_pixels_per_reference_bound": 100_000_000,
                "train_val_canonical_fragment_overlap_count": 0,
                "all_selected_decisions_eligible": True,
                "additional_mask_decode_count": 0,
                "path_or_fragment_identifiers_present": False,
            },
            "decision_count": len(result.decisions),
            "path_or_fragment_identifiers_present": False,
        },
        _fixture_scale_test_only=True,
    )
    receipt_bytes = canonical_eligibility_bytes(receipt)
    receipt_path = tmp_path / "eligibility_receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    trust = LocalQ1EligibilityTrust(
        predecessor_freeze_file_sha256=predecessor_file,
        predecessor_freeze_content_sha256=predecessor_content,
        input_role_locks=roles,
        source_bundle_manifest_sha256="c" * 64,
        geometry_batch_config_sha256=geometry_hash,
        planning_guard_config_sha256=config_authority.document[
            "planning_guard_config_sha256"
        ],
        eligibility_policy_file_sha256=ROUTE_A_POLICY_FILE_SHA256,
        eligibility_policy_content_sha256=ROUTE_A_POLICY_CONTENT_SHA256,
        config_authority_file_sha256=ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256,
        config_authority_content_sha256=ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256,
        eligibility_index_file_sha256=hashlib.sha256(index_bytes).hexdigest(),
        eligibility_index_content_sha256=index["content_sha256"],
        eligibility_receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        eligibility_receipt_content_sha256=receipt["content_sha256"],
    )
    return index_path, receipt_path, trust, result


def _eligibility_payload_arguments(index_path, receipt_path, trust):
    index_payload = index_path.read_bytes()
    receipt_payload = receipt_path.read_bytes()
    policy_payload = CANONICAL_ROUTE_A_POLICY_PATH.read_bytes()
    config_authority_payload = CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH.read_bytes()
    return {
        "index_payload": index_payload,
        "receipt_payload": receipt_payload,
        "policy_payload": policy_payload,
        "config_authority_payload": config_authority_payload,
        "trust": trust,
        "expected_index_bytes": len(index_payload),
        "expected_index_file_sha256": trust.eligibility_index_file_sha256,
        "expected_index_content_sha256": trust.eligibility_index_content_sha256,
        "expected_receipt_bytes": len(receipt_payload),
        "expected_receipt_file_sha256": trust.eligibility_receipt_file_sha256,
        "expected_receipt_content_sha256": (trust.eligibility_receipt_content_sha256),
        "expected_policy_bytes": len(policy_payload),
        "expected_policy_file_sha256": trust.eligibility_policy_file_sha256,
        "expected_policy_content_sha256": (trust.eligibility_policy_content_sha256),
        "expected_config_authority_bytes": len(config_authority_payload),
        "expected_config_authority_file_sha256": (trust.config_authority_file_sha256),
        "expected_config_authority_content_sha256": (
            trust.config_authority_content_sha256
        ),
        "_fixture_scale_test_only": True,
    }


def test_shared_pair_qualification_is_eligible_and_resource_fail_closed() -> None:
    geometry = _geometry()
    first = build_fragment_geometry(_mask(9, notch=True), geometry)
    second = build_fragment_geometry(_mask(34), geometry)
    pair_id = "pair/sha256/" + "1" * 64

    eligible = qualify_local_q1_pair_geometry(
        first,
        second,
        pair_id=pair_id,
        geometry_config=GeometryBatchConfig(geometry=geometry),
    )
    assert eligible.eligible is True
    assert set(eligible.emitted_directions) == {
        direction.value for direction in DEFAULT_DIRECTION_ORDER
    }
    assert eligible.candidate_count > 0

    rejected = qualify_local_q1_pair_geometry(
        first,
        second,
        pair_id=pair_id,
        geometry_config=GeometryBatchConfig(
            geometry=geometry, max_candidates_per_sample=1
        ),
    )
    assert rejected.eligible is False
    assert rejected.failure_stage == "resource"
    assert rejected.failure_reason == "max_candidates_per_sample_exceeded"


def test_complete_frontier_refills_without_reserve_and_proves_prefix(tmp_path) -> None:
    result = _frontier_result(tmp_path)
    proof = result.selection_proof
    assert len(result.training_records) == 6
    assert len(result.validation_records) == 2
    assert proof["arbitrary_fixed_reserve_cutoff_used"] is False
    assert proof["unexplored_higher_priority_record_count"] == 0
    assert (
        proof["training"]["mm_augmented|negative"]["ineligible_predecessor_count"] == 1
    )
    assert proof["validation"]["mm_augmented"]["ineligible_predecessor_count"] == 1
    assert (
        proof["old_to_new_population_difference"]["training_by_dataset_label"][
            "mm_augmented|negative"
        ]["replacement_count"]
        == 1
    )


def test_fixed_pair_input_and_validation_requests_are_supervision_invariant(
    tmp_path,
) -> None:
    original = _frontier_result(tmp_path / "original")
    relabeled = _frontier_result(tmp_path / "relabeled", relabel_validation=True)
    assert [item.pair_identity_sha256 for item in original.decisions] == [
        item.pair_identity_sha256 for item in relabeled.decisions
    ]
    assert [record.pair_id for record in original.validation_records] == [
        record.pair_id for record in relabeled.validation_records
    ]

    base = _record(
        "fixed-input",
        dataset="mm_augmented",
        split="val",
        component="fixed-component",
        label=False,
    )
    changed = replace(base, label=True, direction_b_wrt_a="right")
    reversed_record = replace(
        changed,
        fragment_a=changed.fragment_b,
        fragment_b=changed.fragment_a,
    )
    inputs = [
        _candidate(
            record,
            dataset_name="mm_augmented",
            component_rank="1" * 64,
            record_rank="2" * 64,
            ordinal=index,
        ).geometry_input()
        for index, record in enumerate((base, changed, reversed_record))
    ]
    assert all(isinstance(item, LocalQ1PairGeometryInput) for item in inputs)
    assert set(inputs[0].__dataclass_fields__) == {
        "fragment_a",
        "fragment_b",
        "pair_id",
        "pair_identity_sha256",
    }
    assert inputs[0] == inputs[1] == inputs[2]


def test_preregistered_config_authority_recomputes_exact_production_contract(
    tmp_path,
) -> None:
    authority = load_local_q1_route_a_config_authority()
    assert authority.geometry_config.fingerprint == (
        "dc994bf15596eea1b3d9245ac41505b1e72739eb49a91fef7a02012c744dcfd3"
    )
    assert authority.document["planning_guard_config_sha256"] == (
        "60d3f585f6f5ac27fdb3d2dbad29b8f52c9adb1d1955e88bfaa956ba51798d56"
    )
    assert (
        authority.document["production_population_contract"]["selected_pair_total"]
        == 6964
    )

    link = tmp_path / "config_authority_link.json"
    link.symlink_to(
        __import__(
            "staging.pairwise_v0_2.preflight.local_q1_geometry_qualification",
            fromlist=["CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH"],
        ).CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH
    )
    with pytest.raises(LocalQ1GeometryQualificationError):
        load_local_q1_route_a_config_authority(path=link)


def test_preregistered_config_authority_accepts_only_captured_immutable_bytes() -> None:
    module = __import__(
        "staging.pairwise_v0_2.preflight.local_q1_geometry_qualification",
        fromlist=["CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH"],
    )
    payload = module.CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH.read_bytes()
    from_path = load_local_q1_route_a_config_authority()
    from_payload = load_local_q1_route_a_config_authority_payload(payload=payload)
    assert from_payload == from_path

    with pytest.raises(LocalQ1GeometryQualificationError, match="immutable bytes"):
        load_local_q1_route_a_config_authority_payload(payload=bytearray(payload))

    class EvilBytes(bytes):
        def decode(self, *_args, **_kwargs):
            return "{}"

    with pytest.raises(LocalQ1GeometryQualificationError, match="immutable bytes"):
        load_local_q1_route_a_config_authority_payload(payload=EvilBytes(payload))


def test_dependency_light_config_contract_matches_typed_authority_exactly() -> None:
    payload = CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH.read_bytes()
    lightweight = (
        lightweight_authority.load_local_q1_route_a_config_authority_payload_contract(
            payload=payload
        )
    )
    typed = load_local_q1_route_a_config_authority_payload(payload=payload)
    assert dict(lightweight.document) == dict(typed.document)
    assert lightweight.geometry_batch_config_sha256 == typed.geometry_config.fingerprint
    assert lightweight.planning_guard_config_sha256 == (
        typed.document["planning_guard_config_sha256"]
    )
    assert lightweight.file_sha256 == typed.file_sha256
    assert lightweight.content_sha256 == typed.content_sha256


def test_heavy_module_reexports_exact_dependency_light_authority_identities() -> None:
    import staging.pairwise_v0_2.preflight.local_q1_geometry_qualification as heavy

    for name in (
        "LocalQ1EligibilityDecision",
        "LocalQ1EligibilityTrust",
        "LocalQ1ExternalFileLock",
        "LocalQ1GeometryEligibilityAuthority",
        "LocalQ1GeometryQualificationError",
        "LocalQ1PairGeometryInput",
    ):
        assert getattr(heavy, name) is getattr(lightweight_authority, name)


def test_dependency_light_authority_import_excludes_geometry_execution_stack() -> None:
    code = r"""
import importlib
import sys

module = importlib.import_module(
    "staging.pairwise_v0_2.preflight.local_q1_geometry_eligibility_authority"
)
assert module.LocalQ1GeometryEligibilityAuthority.__module__ == module.__name__
for prefix in (
    "scipy",
    "torch",
    "staging.pairwise_v0_2.geometry.candidate_builder",
    "staging.pairwise_v0_2.training.fragment_geometry_cache",
    "staging.pairwise_v0_2.training.geometry_batch",
    "staging.pairwise_v0_2.training.geometry_cache",
    "staging.pairwise_v0_2.training.local_cache_inventory",
    "staging.pairwise_v0_2.models",
):
    assert not any(
        name == prefix or name.startswith(prefix + ".") for name in sys.modules
    ), prefix
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_corrected_route_a_policy_accepts_only_frozen_payload() -> None:
    payload = CANONICAL_ROUTE_A_POLICY_PATH.read_bytes()
    policy = load_local_q1_route_a_policy_payload(
        payload=payload,
        expected_bytes=len(payload),
    )
    assert policy["schema_version"] == ("dunhuang-local-q1-refreeze-route-decision/0.2")
    assert policy["content_sha256"] == ROUTE_A_POLICY_CONTENT_SHA256

    with pytest.raises(LocalQ1GeometryQualificationError, match="immutable bytes"):
        load_local_q1_route_a_policy_payload(
            payload=bytearray(payload),
            expected_bytes=len(payload),
        )


def test_lazy_qualifier_seals_fresh_cache_and_selected_population_guards(
    tmp_path,
) -> None:
    config_authority = load_local_q1_route_a_config_authority()
    record = _record(
        "live-evidence",
        dataset="mm_augmented",
        split="train",
        component="live-component",
        label=False,
    )
    candidate = _candidate(
        record,
        dataset_name="mm_augmented",
        component_rank="1" * 64,
        record_rank="2" * 64,
        ordinal=0,
    )
    geometry_input = candidate.geometry_input()

    def loader(reference):
        return _mask(
            9 if reference.archive_member.endswith("_a.png") else 34,
            notch=reference.archive_member.endswith("_a.png"),
        )

    qualifier = LazyLocalQ1GeometryQualifier(
        mask_loader=loader,
        cache_root=tmp_path / "fresh_qualification_cache",
        geometry_config=config_authority.geometry_config,
        config_authority=config_authority,
    )
    decision = qualifier(geometry_input)
    assert decision.assessment.eligible is True
    evidence = qualifier.execution_evidence(selected_inputs=(geometry_input,))
    assert (
        evidence["scratch_cache_hit_count"] + evidence["scratch_cache_miss_count"]
        == evidence["mask_decode_count"]
    )
    assert evidence["final_artifact_count"] == evidence["scratch_cache_miss_count"]
    assert evidence["final_artifact_file_bytes"] > 0
    assert evidence["selected_population_guard"] == {
        "inventory_config_sha256": (
            "0270d5b7d7f042943cd52be28d6665ada11a037880fe81abf38f6d7fd008c123"
        ),
        "selected_record_count": 1,
        "selected_pair_commitment_order_sha256": hashlib.sha256(
            (geometry_input.pair_identity_sha256 + "\n").encode("utf-8")
        ).hexdigest(),
        "selected_pair_commitment_set_sha256": hashlib.sha256(
            (geometry_input.pair_identity_sha256 + "\n").encode("utf-8")
        ).hexdigest(),
        "selected_unique_reference_count": 2,
        "selected_max_mask_pixels_per_reference": 96 * 112,
        "max_records_bound": 20_000,
        "max_unique_references_bound": 40_000,
        "max_mask_pixels_per_reference_bound": 100_000_000,
        "train_val_canonical_fragment_overlap_count": 0,
        "all_selected_decisions_eligible": True,
        "additional_mask_decode_count": 0,
        "path_or_fragment_identifiers_present": False,
    }


def test_preflight_import_keeps_model_stack_lazy_in_fresh_process() -> None:
    code = r"""
import sys
import staging.pairwise_v0_2.training as training
assert "torch" not in sys.modules
assert not any(name.startswith("staging.pairwise_v0_2.models") for name in sys.modules)
import staging.pairwise_v0_2.preflight.local_q1_geometry_qualification
for forbidden in (
    "staging.pairwise_v0_2.training.checkpoint",
    "staging.pairwise_v0_2.training.engine",
    "staging.pairwise_v0_2.training.losses",
):
    assert forbidden not in sys.modules
assert not any(name.startswith("staging.pairwise_v0_2.models") for name in sys.modules)
from staging.pairwise_v0_2.training import PairwiseBatch
assert PairwiseBatch.__name__ == "PairwiseBatch"
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_external_dual_hash_authority_is_deeply_immutable_and_cross_bound(
    tmp_path,
) -> None:
    index_path, receipt_path, trust, result = _documents(tmp_path)
    with pytest.raises(LocalQ1GeometryQualificationError, match="receipt contract"):
        load_local_q1_geometry_eligibility_authority(
            index_path=index_path,
            receipt_path=receipt_path,
            trust=trust,
        )
    authority = load_local_q1_geometry_eligibility_authority(
        index_path=index_path,
        receipt_path=receipt_path,
        trust=trust,
        _fixture_scale_test_only=True,
    )
    assert authority.decision_order == tuple(
        decision.pair_identity_sha256 for decision in result.decisions
    )
    authority.assert_selection_proof(result.selection_proof)
    with pytest.raises(TypeError):
        authority.receipt["selection_proof"]["decision_count"] = 0

    with pytest.raises(
        LocalQ1GeometryQualificationError, match="preregistered Route-A production"
    ):
        replace(
            trust,
            geometry_batch_config_sha256="e" * 64,
        )


def test_eligibility_payload_authority_matches_path_authority_field_by_field(
    tmp_path,
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    from_path = load_local_q1_geometry_eligibility_authority(
        index_path=index_path,
        receipt_path=receipt_path,
        trust=trust,
        _fixture_scale_test_only=True,
    )
    from_payloads = load_local_q1_geometry_eligibility_authority_payloads(
        **_eligibility_payload_arguments(index_path, receipt_path, trust)
    )
    assert type(from_payloads) is type(from_path)
    for field in fields(from_path):
        assert getattr(from_payloads, field.name) == getattr(from_path, field.name)


def test_original_two_payload_api_remains_compatible(tmp_path) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    arguments = _eligibility_payload_arguments(index_path, receipt_path, trust)
    legacy = load_local_q1_geometry_eligibility_authority_legacy_payloads(
        index_payload=arguments["index_payload"],
        receipt_payload=arguments["receipt_payload"],
        trust=trust,
        expected_index_bytes=arguments["expected_index_bytes"],
        expected_index_file_sha256=arguments["expected_index_file_sha256"],
        expected_index_content_sha256=arguments["expected_index_content_sha256"],
        expected_receipt_bytes=arguments["expected_receipt_bytes"],
        expected_receipt_file_sha256=arguments["expected_receipt_file_sha256"],
        expected_receipt_content_sha256=(arguments["expected_receipt_content_sha256"]),
        _fixture_scale_test_only=True,
    )
    full = load_local_q1_geometry_eligibility_authority_payloads(**arguments)
    for field in fields(full):
        assert getattr(legacy, field.name) == getattr(full, field.name)


def test_eligibility_path_authority_snapshots_each_of_four_files_once(
    tmp_path, monkeypatch
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    module = __import__(
        "staging.pairwise_v0_2.preflight.local_q1_geometry_qualification",
        fromlist=["_snapshot_locked_regular_file"],
    )
    original = module._snapshot_locked_regular_file
    calls = []

    def observed(path, expected_file_sha256, name):
        calls.append(name)
        return original(path, expected_file_sha256, name)

    monkeypatch.setattr(module, "_snapshot_locked_regular_file", observed)
    authority = load_local_q1_geometry_eligibility_authority(
        index_path=index_path,
        receipt_path=receipt_path,
        trust=trust,
        _fixture_scale_test_only=True,
    )
    assert authority.production_authorized is False
    assert calls == [
        "Route-A config authority",
        "corrected Route-A policy",
        "eligibility index",
        "eligibility receipt",
    ]


def test_eligibility_payload_authority_never_uses_path_open_or_read_bytes(
    tmp_path, monkeypatch
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    arguments = _eligibility_payload_arguments(index_path, receipt_path, trust)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("payload loader attempted a Path byte read")

    module = __import__(
        "staging.pairwise_v0_2.preflight.local_q1_geometry_qualification",
        fromlist=["_snapshot_locked_regular_file"],
    )
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(module, "_snapshot_locked_regular_file", forbidden)
    authority = load_local_q1_geometry_eligibility_authority_payloads(**arguments)
    assert authority.production_authorized is False


@pytest.mark.parametrize(
    "payload_name",
    [
        "index_payload",
        "receipt_payload",
        "policy_payload",
        "config_authority_payload",
    ],
)
def test_eligibility_payload_authority_rejects_bytes_subclasses(
    tmp_path, payload_name
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    arguments = _eligibility_payload_arguments(index_path, receipt_path, trust)

    class EvilBytes(bytes):
        def decode(self, *_args, **_kwargs):
            return "{}"

    arguments[payload_name] = EvilBytes(arguments[payload_name])
    with pytest.raises(LocalQ1GeometryQualificationError, match="immutable bytes"):
        load_local_q1_geometry_eligibility_authority_payloads(**arguments)


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("expected_index_bytes", 1, "byte length mismatch"),
        ("expected_receipt_bytes", True, "positive integer"),
        ("expected_policy_bytes", 1, "byte length mismatch"),
        ("expected_config_authority_bytes", True, "positive integer"),
        ("expected_index_file_sha256", "f" * 64, "locks differ from trust"),
        ("expected_index_content_sha256", "e" * 64, "locks differ from trust"),
        ("expected_receipt_file_sha256", "f" * 64, "locks differ from trust"),
        ("expected_receipt_content_sha256", "e" * 64, "locks differ from trust"),
        ("expected_policy_file_sha256", "f" * 64, "locks differ from trust"),
        ("expected_policy_content_sha256", "e" * 64, "locks differ from trust"),
        (
            "expected_config_authority_file_sha256",
            "f" * 64,
            "locks differ from trust",
        ),
        (
            "expected_config_authority_content_sha256",
            "e" * 64,
            "locks differ from trust",
        ),
    ],
)
def test_eligibility_payload_authority_rejects_external_length_or_hash_changes(
    tmp_path, field_name, replacement, message
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    arguments = _eligibility_payload_arguments(index_path, receipt_path, trust)
    arguments[field_name] = replacement
    with pytest.raises(LocalQ1GeometryQualificationError, match=message):
        load_local_q1_geometry_eligibility_authority_payloads(**arguments)


@pytest.mark.parametrize(
    "payload_name",
    [
        "index_payload",
        "receipt_payload",
        "policy_payload",
        "config_authority_payload",
    ],
)
def test_eligibility_payload_authority_rejects_byte_tampering(
    tmp_path, payload_name
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    arguments = _eligibility_payload_arguments(index_path, receipt_path, trust)
    payload = arguments[payload_name]
    arguments[payload_name] = bytes([payload[0] ^ 1]) + payload[1:]
    with pytest.raises(LocalQ1GeometryQualificationError, match="file hash mismatch"):
        load_local_q1_geometry_eligibility_authority_payloads(**arguments)


def test_eligibility_payload_authority_rejects_internal_content_hash_change(
    tmp_path,
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    arguments = _eligibility_payload_arguments(index_path, receipt_path, trust)
    index = json.loads(arguments["index_payload"].decode("utf-8"))
    index["content_sha256"] = "f" * 64
    payload = canonical_eligibility_bytes(index)
    changed_trust = replace(
        trust,
        eligibility_index_file_sha256=hashlib.sha256(payload).hexdigest(),
    )
    arguments.update(
        index_payload=payload,
        trust=changed_trust,
        expected_index_bytes=len(payload),
        expected_index_file_sha256=changed_trust.eligibility_index_file_sha256,
    )
    with pytest.raises(
        LocalQ1GeometryQualificationError, match="content hash mismatch"
    ):
        load_local_q1_geometry_eligibility_authority_payloads(**arguments)


@pytest.mark.parametrize(
    "payload_name",
    [
        "index_payload",
        "receipt_payload",
        "policy_payload",
        "config_authority_payload",
    ],
)
def test_eligibility_payload_authority_rejects_self_resigned_documents(
    tmp_path, payload_name
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    arguments = _eligibility_payload_arguments(index_path, receipt_path, trust)
    document = json.loads(arguments[payload_name].decode("utf-8"))
    document["status"] = "attacker_self_resigned"
    unsigned = dict(document)
    unsigned.pop("content_sha256")
    document["content_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    resigned_payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    arguments[payload_name] = resigned_payload
    arguments[
        {
            "index_payload": "expected_index_bytes",
            "receipt_payload": "expected_receipt_bytes",
            "policy_payload": "expected_policy_bytes",
            "config_authority_payload": "expected_config_authority_bytes",
        }[payload_name]
    ] = len(resigned_payload)
    with pytest.raises(LocalQ1GeometryQualificationError, match="file hash mismatch"):
        load_local_q1_geometry_eligibility_authority_payloads(**arguments)


@pytest.mark.parametrize("field_to_change", ["extra", "missing"])
def test_eligibility_payload_authority_rejects_resigned_extra_or_missing_fields(
    tmp_path, field_to_change
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    arguments = _eligibility_payload_arguments(index_path, receipt_path, trust)
    index = json.loads(arguments["index_payload"].decode("utf-8"))
    if field_to_change == "extra":
        index["unexpected"] = True
    else:
        index.pop("status")
    unsigned = dict(index)
    unsigned.pop("content_sha256")
    index["content_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payload = canonical_eligibility_bytes(index)
    changed_trust = replace(
        trust,
        eligibility_index_file_sha256=hashlib.sha256(payload).hexdigest(),
        eligibility_index_content_sha256=index["content_sha256"],
    )
    arguments.update(
        index_payload=payload,
        trust=changed_trust,
        expected_index_bytes=len(payload),
        expected_index_file_sha256=changed_trust.eligibility_index_file_sha256,
        expected_index_content_sha256=(changed_trust.eligibility_index_content_sha256),
    )
    with pytest.raises(LocalQ1GeometryQualificationError, match="contract changed"):
        load_local_q1_geometry_eligibility_authority_payloads(**arguments)


def test_eligibility_payload_authority_rejects_self_resigned_reordering(
    tmp_path,
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    arguments = _eligibility_payload_arguments(index_path, receipt_path, trust)
    index = json.loads(arguments["index_payload"].decode("utf-8"))
    index["decisions"].reverse()
    order = [row["pair_identity_sha256"] for row in index["decisions"]]
    index["decision_order_sha256"] = hashlib.sha256(
        json.dumps(
            order,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    unsigned = dict(index)
    unsigned.pop("content_sha256")
    index["content_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payload = canonical_eligibility_bytes(index)
    changed_trust = replace(
        trust,
        eligibility_index_file_sha256=hashlib.sha256(payload).hexdigest(),
        eligibility_index_content_sha256=index["content_sha256"],
    )
    arguments.update(
        index_payload=payload,
        trust=changed_trust,
        expected_index_bytes=len(payload),
        expected_index_file_sha256=changed_trust.eligibility_index_file_sha256,
        expected_index_content_sha256=(changed_trust.eligibility_index_content_sha256),
    )
    with pytest.raises(LocalQ1GeometryQualificationError, match="cross-lock mismatch"):
        load_local_q1_geometry_eligibility_authority_payloads(**arguments)


@pytest.mark.parametrize("encoding", ["duplicate", "noncanonical"])
def test_eligibility_payload_authority_rejects_duplicate_or_noncanonical_json(
    tmp_path, encoding
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    arguments = _eligibility_payload_arguments(index_path, receipt_path, trust)
    original = arguments["index_payload"]
    if encoding == "duplicate":
        payload = b'{"schema_version":"duplicate",' + original[1:]
    else:
        payload = json.dumps(
            json.loads(original.decode("utf-8")),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
    changed_trust = replace(
        trust,
        eligibility_index_file_sha256=hashlib.sha256(payload).hexdigest(),
    )
    arguments.update(
        index_payload=payload,
        trust=changed_trust,
        expected_index_bytes=len(payload),
        expected_index_file_sha256=changed_trust.eligibility_index_file_sha256,
    )
    with pytest.raises(
        LocalQ1GeometryQualificationError, match="strict JSON|canonical"
    ):
        load_local_q1_geometry_eligibility_authority_payloads(**arguments)


def test_authority_rejects_symlink_and_semantically_malformed_eligible_row(
    tmp_path,
) -> None:
    index_path, receipt_path, trust, _result = _documents(tmp_path)
    symlink = tmp_path / "index_link.json"
    symlink.symlink_to(index_path)
    with pytest.raises(LocalQ1GeometryQualificationError):
        load_local_q1_geometry_eligibility_authority(
            index_path=symlink,
            receipt_path=receipt_path,
            trust=trust,
            _fixture_scale_test_only=True,
        )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    eligible_row = next(
        row for row in index["decisions"] if row["assessment"]["eligible"]
    )
    eligible_row["assessment"]["emitted_directions"] = []
    unsigned = dict(index)
    unsigned.pop("content_sha256")
    index["content_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    malformed = canonical_eligibility_bytes(index)
    malformed_path = tmp_path / "malformed_index.json"
    malformed_path.write_bytes(malformed)
    malformed_trust = replace(
        trust,
        eligibility_index_file_sha256=hashlib.sha256(malformed).hexdigest(),
        eligibility_index_content_sha256=index["content_sha256"],
    )
    # Receipt still binds the original index, so either semantic validation or
    # receipt/index cross-lock must reject the independently locked mutation.
    with pytest.raises(LocalQ1GeometryQualificationError):
        load_local_q1_geometry_eligibility_authority(
            index_path=malformed_path,
            receipt_path=receipt_path,
            trust=malformed_trust,
            _fixture_scale_test_only=True,
        )


def test_lazy_qualifier_rejects_nonempty_partial_like_cache_before_decode(
    tmp_path,
) -> None:
    partial = tmp_path / "failed_partial_cache"
    partial.mkdir()
    (partial / "aa").mkdir()
    (partial / "aa" / ("a" * 64 + ".npz")).write_bytes(b"partial")
    calls = []

    def loader(_reference):
        calls.append("decoded")
        return _mask(9)

    with pytest.raises(LocalQ1GeometryQualificationError, match="newly claimed path"):
        LazyLocalQ1GeometryQualifier(
            mask_loader=loader,
            cache_root=partial,
            geometry_config=GeometryBatchConfig(geometry=_geometry()),
        )
    assert calls == []

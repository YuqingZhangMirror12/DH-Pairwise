from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    FragmentIdentity,
    HistoricalIdentityIndex,
)
from staging.pairwise_v0_2.pairwise_data.sampling import (
    validation_stream_fingerprint,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.preflight.freeze_local_q1 import (
    ROUTE_A_SCHEMA_VERSION,
    SYNTHETIC_ARCHIVE_BINDING,
    SYNTHETIC_DATASET_ID,
    LocalQ1FreezeError,
    LocalQ1RouteAFrontierSpool,
    assert_portable_local_q1_receipt,
    freeze_local_q1,
    _priority,
    _qualify_historical,
    _qualify_synthetic,
)
from staging.pairwise_v0_2.preflight.local_q1_geometry_qualification import (
    ROUTE_A_POLICY_CONTENT_SHA256,
    ROUTE_A_POLICY_FILE_SHA256,
    ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256,
    ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256,
    LocalQ1EligibilityDecision,
    LocalQ1EligibilityTrust,
    LocalQ1ExternalFileLock,
    LocalQ1GeometryQualificationError,
    build_untrusted_eligibility_index,
    build_untrusted_eligibility_receipt,
    canonical_eligibility_bytes,
    load_local_q1_geometry_eligibility_authority,
    load_local_q1_route_a_config_authority,
)
from staging.pairwise_v0_2.preflight.local_q1_selection_frontier import (
    LocalQ1FrontierCandidate,
    LocalQ1SelectionFrontier,
)
from staging.pairwise_v0_2.geometry import DEFAULT_DIRECTION_ORDER
from staging.pairwise_v0_2.training.local_q1_pair_qualification import (
    LocalQ1PairQualification,
)


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _historical_record(
    *,
    dataset: str,
    binding,
    split: str,
    component: str,
    group: str,
    fragments,
    left: int,
    right: int,
    label: bool,
) -> TrainingPairRecord:
    def ref(index: int) -> MaskMemberRef:
        member, fragment_id, _content = fragments[index]
        return MaskMemberRef(
            binding=binding,
            archive_member=member,
            fragment_id=fragment_id,
            dataset_id=dataset,
            canonical_group_id=group,
            component_id=component,
            split=split,
            threshold_rule="binary_brighter_value",
        )

    fragment_a = ref(left)
    fragment_b = ref(right)
    return TrainingPairRecord(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        label=label,
        direction_b_wrt_a=None,
        dataset_id=dataset,
        canonical_group_id=group,
        component_id=component,
        split=split,
        canonical_pair_key=tuple(
            sorted((fragment_a.fragment_id, fragment_b.fragment_id))
        ),
        label_origin="local-q1-fixture",
    )


def _synthetic_record(component_index: int, label: bool) -> TrainingPairRecord:
    component = "synthetic/component/{}".format(component_index)
    group = "synthetic/group/{}".format(component_index)

    def ref(endpoint: int) -> MaskMemberRef:
        return MaskMemberRef(
            binding=SYNTHETIC_ARCHIVE_BINDING,
            archive_member="synthetic/{}/{}.png".format(component_index, endpoint),
            fragment_id="{}/{}".format(group, endpoint),
            dataset_id=SYNTHETIC_DATASET_ID,
            canonical_group_id=group,
            component_id=component,
            split="train",
            threshold_rule="grayscale_uint8_gt_127",
            content_sha256=_sha(
                "synthetic-content-{}-{}".format(component_index, endpoint)
            ),
        )

    fragment_a = ref(0)
    fragment_b = ref(1)
    return TrainingPairRecord(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        label=label,
        direction_b_wrt_a=None,
        dataset_id=SYNTHETIC_DATASET_ID,
        canonical_group_id=group,
        component_id=component,
        split="train",
        canonical_pair_key=tuple(
            sorted((fragment_a.fragment_id, fragment_b.fragment_id))
        ),
        label_origin="local-q1-synthetic-fixture",
    )


def _fixture():
    identities = {}
    validation = []
    historical_train = []
    expected_validation_components = {}

    for dataset_index, (dataset, binding) in enumerate(
        (
            ("mm_augmented", MM_CANONICAL_BINDING),
            ("eccv_1113data", ECCV_CANONICAL_BINDING),
        )
    ):
        expected_validation_components[dataset] = 2
        for split, component_count in (("val", 2), ("train", 3)):
            for component_index in range(component_count):
                component = "{}/fixture/{}/component/{}".format(
                    dataset, split, component_index
                )
                group = "{}/fixture/{}/group/{}".format(dataset, split, component_index)
                fragments = []
                for endpoint in range(3):
                    member = "{}/{}/{}/{}.png".format(
                        dataset, split, component_index, endpoint
                    )
                    fragment_id = "{}/{}/{}/{}".format(
                        dataset, split, component_index, endpoint
                    )
                    content = _sha(
                        "historical-content-{}-{}-{}-{}".format(
                            dataset_index, split, component_index, endpoint
                        )
                    )
                    fragments.append((member, fragment_id, content))
                    identities[(dataset, member)] = FragmentIdentity(
                        dataset_id=dataset,
                        binding=binding,
                        archive_member=member,
                        fragment_id=fragment_id,
                        pair_group_id=group,
                        component_id=component,
                        split=split,
                        content_sha256=content,
                    )
                if split == "val":
                    validation.extend(
                        _historical_record(
                            dataset=dataset,
                            binding=binding,
                            split=split,
                            component=component,
                            group=group,
                            fragments=fragments,
                            left=left,
                            right=right,
                            label=label,
                        )
                        for left, right, label in (
                            (0, 1, False),
                            (0, 2, True),
                            (1, 2, False),
                        )
                    )
                else:
                    historical_train.extend(
                        _historical_record(
                            dataset=dataset,
                            binding=binding,
                            split=split,
                            component=component,
                            group=group,
                            fragments=fragments,
                            left=left,
                            right=right,
                            label=label,
                        )
                        for left, right, label in (
                            (0, 1, False),
                            (1, 2, True),
                        )
                    )

    index = HistoricalIdentityIndex(
        by_member=identities,
        split_candidate_id="fixture-candidate",
        split_seed="fixture-seed",
        artifact_locks={
            "mm_fingerprint_cache": {"bytes": 1, "sha256": "1" * 64},
            "eccv_fingerprint_cache": {"bytes": 1, "sha256": "2" * 64},
            "historical_split": {"bytes": 1, "sha256": "3" * 64},
        },
    )
    synthetic_train = [
        _synthetic_record(index, label)
        for index, label in enumerate((False, False, True, True))
    ]
    expected_validation_counts = dict(
        __import__("collections").Counter(
            (record.dataset_id, record.label) for record in validation
        )
    )
    expected_train_counts = dict(
        __import__("collections").Counter(
            (record.dataset_id, record.label)
            for record in historical_train + synthetic_train
        )
    )
    archive_locks = {
        "mm_augmented": {
            "format": MM_CANONICAL_BINDING.archive_format,
            "logical_id": MM_CANONICAL_BINDING.logical_id,
            "bytes": 11,
            "sha256": MM_CANONICAL_BINDING.sha256,
        },
        "eccv_1113data": {
            "format": ECCV_CANONICAL_BINDING.archive_format,
            "logical_id": ECCV_CANONICAL_BINDING.logical_id,
            "bytes": 12,
            "sha256": ECCV_CANONICAL_BINDING.sha256,
        },
    }
    return {
        "index": index,
        "validation": validation,
        "historical_train": historical_train,
        "synthetic_train": synthetic_train,
        "expected_validation_counts": expected_validation_counts,
        "expected_train_counts": expected_train_counts,
        "expected_validation_components": expected_validation_components,
        "archive_locks": archive_locks,
    }


def _freeze(
    fixture,
    *,
    validation=None,
    synthetic_train=None,
    target=2,
    geometry_eligibility=None,
    frontier_database=None,
    route_a_hashes=None,
):
    validation = fixture["validation"] if validation is None else validation
    synthetic_train = (
        fixture["synthetic_train"] if synthetic_train is None else synthetic_train
    )
    expected_validation_counts = dict(
        __import__("collections").Counter(
            (record.dataset_id, record.label) for record in validation
        )
    )
    expected_train_counts = dict(
        __import__("collections").Counter(
            (record.dataset_id, record.label)
            for record in fixture["historical_train"] + list(synthetic_train)
        )
    )
    route_a_hashes = route_a_hashes or {}
    return freeze_local_q1(
        identity_index=fixture["index"],
        validation_records=validation,
        historical_training_records=fixture["historical_train"],
        synthetic_training_records=synthetic_train,
        historical_archive_locks=fixture["archive_locks"],
        synthetic_manifest_lock={"bytes": 99, "sha256": "a" * 64},
        seed="fixture-local-q1",
        target_per_dataset_label=target,
        validation_caps={"mm_augmented": 2, "eccv_1113data": 1},
        expected_validation_counts=expected_validation_counts,
        expected_validation_fingerprint=validation_stream_fingerprint(validation)[
            "sha256"
        ],
        expected_validation_component_counts=fixture["expected_validation_components"],
        expected_train_input_counts=expected_train_counts,
        expected_synthetic_manifest_sha256="a" * 64,
        geometry_eligibility=geometry_eligibility,
        frontier_database=frontier_database,
        **route_a_hashes,
    )


def _route_a_assessment(eligible):
    if eligible:
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
            candidate_semantic_sha256="b" * 64,
        )
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
        candidate_semantic_sha256="c" * 64,
    )


def _route_a_authority(fixture, tmp_path):
    seed = "fixture-local-q1"
    predecessor_result = _freeze(fixture)
    predecessor_bytes = (
        json.dumps(
            predecessor_result.receipt,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    frontier_path = tmp_path / "qualification_frontier.sqlite3"
    train_candidates = []
    validation_candidates = []
    with LocalQ1SelectionFrontier(frontier_path) as frontier:
        for ordinal, record in enumerate(fixture["validation"]):
            qualified = _qualify_historical(fixture["index"].verify_record(record))
            rank = _priority(
                seed,
                "validation-record-label-blind|{}".format(
                    qualified.dataset_name
                ),
                qualified.pair_identity_token,
            )
            candidate = LocalQ1FrontierCandidate(
                record=qualified.record,
                dataset_name=qualified.dataset_name,
                component_token=qualified.component_token,
                pair_identity_token=qualified.pair_identity_token,
                record_token=qualified.record_token,
                record_rank=rank,
                component_rank=_priority(
                    seed,
                    "validation-component|{}".format(qualified.dataset_name),
                    qualified.component_token,
                ),
                source_ordinal=ordinal,
            )
            frontier.add(candidate)
            validation_candidates.append(candidate)

        ordinal = 0
        for record in fixture["historical_train"]:
            qualified = _qualify_historical(fixture["index"].verify_record(record))
            label_value = int(record.label)
            candidate = LocalQ1FrontierCandidate(
                record=qualified.record,
                dataset_name=qualified.dataset_name,
                component_token=qualified.component_token,
                pair_identity_token=qualified.pair_identity_token,
                record_token=qualified.record_token,
                record_rank=_priority(
                    seed,
                    "train-record|{}|{}".format(
                        qualified.dataset_name, label_value
                    ),
                    qualified.pair_identity_token,
                ),
                component_rank=_priority(
                    seed,
                    "train-component|{}|{}".format(
                        qualified.dataset_name, label_value
                    ),
                    qualified.component_token,
                ),
                source_ordinal=ordinal,
            )
            ordinal += 1
            frontier.add(candidate)
            train_candidates.append(candidate)
        for record in fixture["synthetic_train"]:
            qualified = _qualify_synthetic(record)
            label_value = int(record.label)
            candidate = LocalQ1FrontierCandidate(
                record=record,
                dataset_name=qualified.dataset_name,
                component_token=qualified.component_token,
                pair_identity_token=qualified.pair_identity_token,
                record_token=qualified.record_token,
                record_rank=_priority(
                    seed,
                    "train-record|{}|{}".format(
                        qualified.dataset_name, label_value
                    ),
                    qualified.pair_identity_token,
                ),
                component_rank=_priority(
                    seed,
                    "train-component|{}|{}".format(
                        qualified.dataset_name, label_value
                    ),
                    qualified.component_token,
                ),
                source_ordinal=ordinal,
            )
            ordinal += 1
            frontier.add(candidate)
            train_candidates.append(candidate)

        mm_negative = [
            candidate
            for candidate in train_candidates
            if candidate.dataset_name == "mm_augmented"
            and not candidate.record.label
        ]
        rejected_train = min(
            mm_negative,
            key=lambda candidate: (
                candidate.component_rank,
                candidate.component_token,
                candidate.record_rank,
            ),
        ).pair_identity_token
        first_mm_component = min(
            {
                candidate.component_token
                for candidate in validation_candidates
                if candidate.dataset_name == "mm_augmented"
            }
        )
        rejected_validation = min(
            (
                candidate
                for candidate in validation_candidates
                if candidate.dataset_name == "mm_augmented"
                and candidate.component_token == first_mm_component
            ),
            key=lambda candidate: candidate.record_rank,
        ).pair_identity_token
        rejected = {rejected_train, rejected_validation}

        def decide(item):
            return LocalQ1EligibilityDecision(
                pair_identity_sha256=item.pair_identity_sha256,
                assessment=_route_a_assessment(
                    item.pair_identity_sha256 not in rejected
                ),
            )

        result = frontier.select(
            decide=decide,
            train_datasets=("mm_augmented", "eccv_1113data", "canonical_new"),
            train_target_per_dataset_label=2,
            validation_caps={"mm_augmented": 2, "eccv_1113data": 1},
        )

    config_authority = load_local_q1_route_a_config_authority()
    geometry_config = config_authority.geometry_config
    index = build_untrusted_eligibility_index(
        result.decisions, geometry_config=geometry_config
    )
    index_bytes = canonical_eligibility_bytes(index)
    index_path = tmp_path / "eligibility_index.json"
    index_path.write_bytes(index_bytes)
    predecessor_file = hashlib.sha256(predecessor_bytes).hexdigest()
    predecessor_content = predecessor_result.receipt["content_sha256"]
    roles = {
        "freeze_receipt": LocalQ1ExternalFileLock(
            len(predecessor_bytes), predecessor_file
        ),
        "mm_archive": LocalQ1ExternalFileLock(
            fixture["archive_locks"]["mm_augmented"]["bytes"],
            fixture["archive_locks"]["mm_augmented"]["sha256"],
        ),
        "eccv_archive": LocalQ1ExternalFileLock(
            fixture["archive_locks"]["eccv_1113data"]["bytes"],
            fixture["archive_locks"]["eccv_1113data"]["sha256"],
        ),
        "mm_fingerprint_cache": LocalQ1ExternalFileLock(1, "1" * 64),
        "eccv_fingerprint_cache": LocalQ1ExternalFileLock(1, "2" * 64),
        "historical_split": LocalQ1ExternalFileLock(1, "3" * 64),
        "synthetic_manifest": LocalQ1ExternalFileLock(99, "a" * 64),
        "synthetic_archive": LocalQ1ExternalFileLock(
            100, SYNTHETIC_ARCHIVE_BINDING.sha256
        ),
    }
    source_hash = "d" * 64
    geometry_hash = index["qualification_contract"][
        "geometry_batch_config_sha256"
    ]
    planning_hash = config_authority.document["planning_guard_config_sha256"]
    receipt = build_untrusted_eligibility_receipt(
        index=index,
        selection_proof=result.selection_proof,
        predecessor_freeze_file_sha256=predecessor_file,
        predecessor_freeze_content_sha256=predecessor_content,
        input_role_locks=roles,
        source_bundle_manifest_sha256=source_hash,
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
            "scratch_cache_artifact_set_sha256": "8" * 64,
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
        source_bundle_manifest_sha256=source_hash,
        geometry_batch_config_sha256=geometry_hash,
        planning_guard_config_sha256=planning_hash,
        eligibility_policy_file_sha256=ROUTE_A_POLICY_FILE_SHA256,
        eligibility_policy_content_sha256=ROUTE_A_POLICY_CONTENT_SHA256,
        config_authority_file_sha256=ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256,
        config_authority_content_sha256=ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256,
        eligibility_index_file_sha256=hashlib.sha256(index_bytes).hexdigest(),
        eligibility_index_content_sha256=index["content_sha256"],
        eligibility_receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        eligibility_receipt_content_sha256=receipt["content_sha256"],
    )
    authority = load_local_q1_geometry_eligibility_authority(
        index_path=index_path,
        receipt_path=receipt_path,
        trust=trust,
        _fixture_scale_test_only=True,
    )
    route_hashes = {
        "expected_predecessor_freeze_file_sha256": predecessor_file,
        "expected_predecessor_freeze_content_sha256": predecessor_content,
        "expected_predecessor_freeze_bytes": len(predecessor_bytes),
        "predecessor_freeze_file_bytes": predecessor_bytes,
        "synthetic_archive_lock": {
            "bytes": 100,
            "sha256": SYNTHETIC_ARCHIVE_BINDING.sha256,
        },
        "expected_source_bundle_manifest_sha256": source_hash,
        "expected_geometry_batch_config_sha256": geometry_hash,
        "expected_planning_guard_config_sha256": planning_hash,
        "_fixture_scale_test_only": True,
    }
    return authority, route_hashes, result


def test_freeze_is_balanced_label_blind_ordered_and_portable() -> None:
    fixture = _fixture()
    result = _freeze(fixture)
    repeated = _freeze(fixture)

    assert len(result.training_records) == 12
    assert repeated.receipt == result.receipt
    assert repeated.training_records == result.training_records
    assert repeated.validation_records == result.validation_records
    train_counts = __import__("collections").Counter(
        (
            "canonical_new"
            if record.dataset_id == SYNTHETIC_DATASET_ID
            else record.dataset_id,
            record.label,
        )
        for record in result.training_records
    )
    assert set(train_counts.values()) == {2}
    assert len(result.validation_records) == 6
    source_positions = {
        record.pair_id: index for index, record in enumerate(fixture["validation"])
    }
    selected_positions = [
        source_positions[record.pair_id] for record in result.validation_records
    ]
    assert selected_positions == sorted(selected_positions)
    assert result.receipt["validation"]["selection_label_blind"] is True
    assert (
        result.receipt["validation"]["by_dataset"]["mm_augmented"][
            "selected_max_per_component"
        ]
        == 2
    )
    assert (
        result.receipt["validation"]["by_dataset"]["eccv_1113data"][
            "selected_max_per_component"
        ]
        == 1
    )
    assert result.receipt[
        "selected_train_vs_complete_validation_identity_universe_overlap"
    ] == {"component": 0, "physical_member": 0, "content_sha256": 0}
    assert result.receipt["scope"]["sealed_real_read"] is False
    assert result.receipt["scope"]["archive_mask_members_opened"] is False
    assert_portable_local_q1_receipt(result.receipt)


def test_validation_membership_is_independent_of_labels() -> None:
    fixture = _fixture()
    original = _freeze(fixture)
    relabeled = [
        replace(record, label=not record.label, direction_b_wrt_a=None)
        for record in fixture["validation"]
    ]
    changed = _freeze(fixture, validation=relabeled)
    assert [record.pair_id for record in original.validation_records] == [
        record.pair_id for record in changed.validation_records
    ]


def test_content_overlap_is_quarantined_before_quota_check() -> None:
    fixture = _fixture()
    val_content = next(
        identity.content_sha256
        for identity in fixture["index"].identities_for_split("val")
    )
    synthetic = list(fixture["synthetic_train"])
    first = synthetic[0]
    synthetic[0] = replace(
        first,
        fragment_a=replace(first.fragment_a, content_sha256=val_content),
    )
    with pytest.raises(LocalQ1FreezeError, match="insufficient eligible canonical_new"):
        _freeze(fixture, synthetic_train=synthetic)


def test_unreachable_quota_fails_closed() -> None:
    fixture = _fixture()
    with pytest.raises(LocalQ1FreezeError, match="insufficient eligible"):
        _freeze(fixture, target=3)


def test_route_a_refreeze_replays_external_decisions_and_refills_exactly(
    tmp_path,
) -> None:
    fixture = _fixture()
    authority, route_hashes, qualification = _route_a_authority(
        fixture, tmp_path
    )
    result = _freeze(
        fixture,
        geometry_eligibility=authority,
        frontier_database=tmp_path / "refreeze_frontier.sqlite3",
        route_a_hashes=route_hashes,
    )

    assert result.receipt["schema_version"] == ROUTE_A_SCHEMA_VERSION
    assert result.receipt["status"] == (
        "fixture_scale_test_only_not_authorized_for_real_scan"
    )
    assert result.receipt["geometry_eligibility_authority"][
        "production_authorized"
    ] is False
    assert result.training_records == qualification.training_records
    assert result.validation_records == qualification.validation_records
    assert len(result.training_records) == 12
    assert len(result.validation_records) == 6
    assert result.receipt["training"]["population"]["by_dataset_label"] == {
        "canonical_new|negative": 2,
        "canonical_new|positive": 2,
        "eccv_1113data|negative": 2,
        "eccv_1113data|positive": 2,
        "mm_augmented|negative": 2,
        "mm_augmented|positive": 2,
    }
    assert result.receipt[
        "selected_train_vs_complete_validation_identity_universe_overlap"
    ] == {"component": 0, "physical_member": 0, "content_sha256": 0}
    proof = result.receipt["route_a_frontier_selection_proof"]
    assert proof["arbitrary_fixed_reserve_cutoff_used"] is False
    assert proof["unexplored_higher_priority_record_count"] == 0
    assert proof["old_to_new_population_difference"][
        "training_by_dataset_label"
    ]["mm_augmented|negative"]["replacement_count"] == 1
    assert result.receipt["geometry_eligibility_authority"][
        "partial_cache_role_count"
    ] == 0
    assert result.receipt["scope"]["sealed_real_read"] is False
    assert_portable_local_q1_receipt(result.receipt)


def test_public_route_a_frontier_spool_matches_locked_legacy_population(
    tmp_path,
) -> None:
    fixture = _fixture()
    validation = fixture["validation"]
    with LocalQ1RouteAFrontierSpool(
        database=tmp_path / "public_frontier.sqlite3",
        identity_index=fixture["index"],
        seed="fixture-local-q1",
        target_per_dataset_label=2,
        validation_caps={"mm_augmented": 2, "eccv_1113data": 1},
        expected_validation_counts=fixture["expected_validation_counts"],
        expected_validation_fingerprint=validation_stream_fingerprint(validation)[
            "sha256"
        ],
        expected_validation_component_counts=fixture[
            "expected_validation_components"
        ],
        expected_train_input_counts=fixture["expected_train_counts"],
        _fixture_scale_test_only=True,
    ) as spool:
        for record in validation:
            spool.add_validation(record)
        spool.seal_validation()
        for record in fixture["historical_train"]:
            spool.add_historical_training(record)
        for record in fixture["synthetic_train"]:
            spool.add_synthetic_training(record)
        result = spool.select(
            decide=lambda item: LocalQ1EligibilityDecision(
                pair_identity_sha256=item.pair_identity_sha256,
                assessment=_route_a_assessment(True),
            )
        )

    legacy = _freeze(fixture)
    assert result.training_records == legacy.training_records
    assert result.validation_records == legacy.validation_records
    predecessor = result.selection_proof["predecessor_population"]
    assert predecessor == {
        "training_count": legacy.receipt["training"]["population"]["count"],
        "validation_count": legacy.receipt["validation"]["population"]["count"],
        "training_record_order_sha256": legacy.receipt["training"]["population"][
            "record_order_commitment_sha256"
        ],
        "training_record_set_sha256": legacy.receipt["training"]["population"][
            "record_set_commitment_sha256"
        ],
        "validation_record_order_sha256": legacy.receipt["validation"][
            "population"
        ]["record_order_commitment_sha256"],
        "validation_record_set_sha256": legacy.receipt["validation"][
            "population"
        ]["record_set_commitment_sha256"],
    }

    with pytest.raises(LocalQ1FreezeError, match="preregistered population"):
        LocalQ1RouteAFrontierSpool(
            database=tmp_path / "wrong_production_fingerprint.sqlite3",
            identity_index=fixture["index"],
            expected_validation_fingerprint="0" * 64,
        )


def test_route_a_refreeze_rejects_missing_decision_or_upstream_lock(tmp_path) -> None:
    fixture = _fixture()
    authority, route_hashes, _qualification = _route_a_authority(
        fixture, tmp_path
    )
    wrong_hashes = dict(route_hashes)
    wrong_hashes["expected_source_bundle_manifest_sha256"] = "0" * 64
    with pytest.raises(LocalQ1FreezeError, match="authority differs"):
        _freeze(
            fixture,
            geometry_eligibility=authority,
            frontier_database=tmp_path / "wrong_lock_frontier.sqlite3",
            route_a_hashes=wrong_hashes,
        )

    wrong_predecessor = dict(route_hashes)
    wrong_predecessor["predecessor_freeze_file_bytes"] += b" "
    with pytest.raises(LocalQ1FreezeError, match="file byte lock"):
        _freeze(
            fixture,
            geometry_eligibility=authority,
            frontier_database=tmp_path / "wrong_predecessor_frontier.sqlite3",
            route_a_hashes=wrong_predecessor,
        )

    wrong_scale = dict(route_hashes)
    wrong_scale.pop("_fixture_scale_test_only")
    with pytest.raises(LocalQ1FreezeError, match="cannot be confused"):
        _freeze(
            fixture,
            geometry_eligibility=authority,
            frontier_database=tmp_path / "wrong_scale_frontier.sqlite3",
            route_a_hashes=wrong_scale,
        )

    # Removing even one consumed prefix decision is rejected before a
    # population can be promoted.
    missing_token = authority.decision_order[0]
    reduced = dict(authority.decisions)
    reduced.pop(missing_token)
    with pytest.raises(LocalQ1GeometryQualificationError):
        replace(authority, decisions=reduced)

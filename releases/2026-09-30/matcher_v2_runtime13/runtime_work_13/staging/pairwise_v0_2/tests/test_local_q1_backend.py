from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from staging.pairwise_v0_2.geometry import ContourKeypointConfig
from staging.pairwise_v0_2.models.coarse import SymmetricCoarseSiamese
from staging.pairwise_v0_2.models.local_matcher import MatcherMode
from staging.pairwise_v0_2.models.pairwise import (
    CoarseModelConfig,
    DunhuangPairwiseV02,
    LocalModelConfig,
    PairwiseModelConfig,
)
from staging.pairwise_v0_2.training.checkpoint import (
    canonical_config_hash,
    canonical_tensor_tree_sha256,
    save_checkpoint,
)
from staging.pairwise_v0_2.training.geometry_batch import (
    KEYPOINT_REPRESENTATION,
    BatchComplexityReceipt,
    GeometrySampleReceipt,
    RaggedGeometryBatch,
)
from staging.pairwise_v0_2.training.local_q1_backend import (
    CheckpointModelScope,
    ExactSeamStepConfig,
    FusedCheckpointBindings,
    LOCAL_Q1_AUTHORITY_KIND,
    LOCAL_Q1_AUTHORITY_STATUS,
    LOCAL_Q1_COARSE_INPUT_SHAPE,
    LocalQ1Backend,
    LocalQ1BackendError,
    LocalQ1BackendMode,
    LocalQ1StepConfig,
    TrustedCheckpointBinding,
    checkpoint_authority_identity_sha256,
    checkpoint_semantic_projection,
)
from staging.pairwise_v0_2.training.local_q1_provider import (
    local_q1_prepared_digests,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    EvidenceMode,
    PreparedAblationBatch,
)


def _template(*, iterations: int = 5, tolerance: float = 1.0) -> PairwiseModelConfig:
    return PairwiseModelConfig(
        coarse=CoarseModelConfig(
            input_channels=1,
            widths=(4, 8),
            embedding_dim=8,
            hidden_dim=8,
        ),
        local=LocalModelConfig(
            input_channels=3,
            feature_dim=8,
            num_heads=2,
            ff_dim=16,
            matcher_mode=MatcherMode.DUSTBIN_SINKHORN.value,
            matcher_temperature=0.25,
            sinkhorn_iterations=iterations,
            sinkhorn_tolerance=tolerance,
            require_sinkhorn_convergence=True,
            sinkhorn_training_policy="finite_with_residual",
            dropout=0.0,
        ),
        fusion_hidden_dim=8,
        direction_aggregation_temperature=0.25,
    )


def _arm(backend: LocalQ1Backend, name: AblationArmName) -> AblationArm:
    evidence, matcher = {
        AblationArmName.LOCAL_DUAL_SOFTMAX: (
            EvidenceMode.LOCAL,
            MatcherMode.DUAL_SOFTMAX.value,
        ),
        AblationArmName.LOCAL_DUSTBIN_SINKHORN: (
            EvidenceMode.LOCAL,
            MatcherMode.DUSTBIN_SINKHORN.value,
        ),
        AblationArmName.KEYPOINT_DUAL_SOFTMAX: (
            EvidenceMode.LOCAL,
            MatcherMode.DUAL_SOFTMAX.value,
        ),
        AblationArmName.KEYPOINT_DUSTBIN_SINKHORN: (
            EvidenceMode.LOCAL,
            MatcherMode.DUSTBIN_SINKHORN.value,
        ),
        AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM: (
            EvidenceMode.LOCAL,
            MatcherMode.DUSTBIN_SINKHORN.value,
        ),
        AblationArmName.FUSED: (
            EvidenceMode.FUSED,
            MatcherMode.DUSTBIN_SINKHORN.value,
        ),
    }[name]
    return AblationArm(
        name=name,
        evidence=evidence,
        matcher_mode=matcher,
        model_config=backend.model_config_for(name),
        optimizer_config=backend.optimizer_config,
        aggregation_config=backend.aggregation_config,
        arc_pooling=backend.model_template.arc_pooling,
    )


def _config_fingerprint(provenance):
    raw = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _payload() -> RaggedGeometryBatch:
    generator = torch.Generator().manual_seed(90210)
    batch_size = 2
    candidate_count = 8
    coarse_a = torch.zeros((batch_size, 1, 16, 16), dtype=torch.float32)
    coarse_b = torch.zeros_like(coarse_a)
    coarse_a[0, 0, 2:12, 3:9] = 1.0
    coarse_a[1, 0, 3:13, 5:11] = 1.0
    coarse_b[0, 0, 4:14, 6:12] = 1.0
    coarse_b[1, 0, 2:10, 2:9] = 1.0
    local_a = torch.rand((candidate_count, 2, 3, 8, 8), generator=generator)
    local_b = torch.rand((candidate_count, 2, 3, 8, 8), generator=generator)
    sample_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    direction_index = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    provenance = {
        "fixture": "LOCAL-Q1 typed backend",
        "patch_shape": [3, 8, 8],
        "known_orientation": True,
    }
    fingerprint = _config_fingerprint(provenance)
    receipts = tuple(
        GeometrySampleReceipt(
            pair_id="fixture-pair-{}".format(index),
            geometry_cache_key="geometry/sha256/" + str(index) * 64,
            dataset_id="fixture",
            canonical_group_id="fixture/group/{}".format(index),
            component_id="fixture/component/{}".format(index),
            geometry_status="ok",
            geometry_failure_reason=None,
            candidate_count=4,
            direction_slot_valid=(True, True, True, True),
            emitted_directions=(
                "b_left_of_a",
                "b_right_of_a",
                "b_above_a",
                "b_below_a",
            ),
            geometry_quality={"fixture": True},
        )
        for index in range(batch_size)
    )
    return RaggedGeometryBatch(
        coarse_a=coarse_a,
        coarse_b=coarse_b,
        local_a=local_a,
        local_b=local_b,
        token_mask_a=torch.ones((candidate_count, 2), dtype=torch.bool),
        token_mask_b=torch.ones((candidate_count, 2), dtype=torch.bool),
        sample_index=sample_index,
        direction_index=direction_index,
        candidate_valid=torch.ones(candidate_count, dtype=torch.bool),
        labels=torch.tensor([True, False], dtype=torch.bool),
        direction_target=torch.tensor([0, -1], dtype=torch.long),
        direction_target_valid=torch.tensor([True, False], dtype=torch.bool),
        direction_slot_valid=torch.ones((batch_size, 4), dtype=torch.bool),
        geometry_valid=torch.ones(batch_size, dtype=torch.bool),
        candidate_ids=tuple(
            "fixture:{}:{}".format(int(sample_index[index]), index)
            for index in range(candidate_count)
        ),
        sample_ids=("fixture/sample/0", "fixture/sample/1"),
        geometry_cache_keys=("geometry/key/0", "geometry/key/1"),
        sample_receipts=receipts,
        config_fingerprint=fingerprint,
        config_provenance=provenance,
        complexity_receipt=BatchComplexityReceipt(
            candidate_count=candidate_count,
            padded_sequence_a=2,
            padded_sequence_b=2,
            local_tensor_elements=2 * candidate_count * 2 * 3 * 8 * 8,
            attention_score_elements_per_head=candidate_count * 16,
            affinity_elements=candidate_count * 4,
            sinkhorn_elements=candidate_count * 9,
            sequence_bucket_counts={"le_0008": candidate_count},
        ),
        sequence_length_buckets=("le_0008",) * candidate_count,
    )


def _prepared(
    payload: Optional[RaggedGeometryBatch] = None,
) -> PreparedAblationBatch:
    value = payload or _payload()
    prepared_sha, local_sha = local_q1_prepared_digests(value)
    return PreparedAblationBatch(
        payload=value,
        sample_count=value.batch_size,
        record_sequence_sha256=hashlib.sha256(
            b"fixture-record-sequence-with-supervision"
        ).hexdigest(),
        prepared_input_sha256=prepared_sha,
        local_candidate_sha256=local_sha,
        coarse_preprocessing_sha256=hashlib.sha256(
            b"fixture-coarse-preprocessing"
        ).hexdigest(),
        geometry_config_sha256=str(
            value.config_provenance.get(
                "base_geometry_config_sha256", value.config_fingerprint
            )
        ),
        processing_counts={
            "mask_load_count": 4,
            "coarse_preprocess_count": 4,
            "geometry_build_count": 0,
            "geometry_cache_read_count": 4,
            "geometry_cache_write_count": 0,
            "local_candidate_count": value.candidate_count,
        },
        candidate_representation=value.candidate_representation,
    )


def _keypoint_payload() -> RaggedGeometryBatch:
    base = _payload()
    correspondence = torch.zeros(
        (base.candidate_count, base.local_a.shape[1], base.local_b.shape[1]),
        dtype=torch.bool,
    )
    correspondence[:, 0, 0] = True
    correspondence[:, 1, 1] = True
    provenance = {
        **dict(base.config_provenance),
        "base_geometry_config_sha256": base.config_fingerprint,
        "local_candidate_representation": KEYPOINT_REPRESENTATION,
        "contour_keypoint_config": asdict(ContourKeypointConfig()),
    }
    return replace(
        base,
        candidate_representation=KEYPOINT_REPRESENTATION,
        correspondence_mask=correspondence,
        config_provenance=provenance,
        config_fingerprint=_config_fingerprint(provenance),
    )


def _exact_keypoint_payload() -> RaggedGeometryBatch:
    base = _keypoint_payload()
    target_a = torch.full(base.token_mask_a.shape, -2, dtype=torch.long)
    target_b = torch.full(base.token_mask_b.shape, -2, dtype=torch.long)
    # Positive sample, correct direction: one reciprocal match and one
    # confidently non-seam token on each side.
    target_a[0] = torch.tensor([0, -1])
    target_b[0] = torch.tensor([0, -1])
    # Positive wrong-direction candidates remain ignored.
    # Negative sample: every real token is explicitly assigned to dustbin.
    target_a[4:] = -1
    target_b[4:] = -1
    return replace(
        base,
        exact_assignment_target_a=target_a,
        exact_assignment_target_b=target_b,
        exact_supervision_config={
            "schema_version": "exact-seam-ragged-supervision/v0.1",
            "targets_are_model_inputs": False,
        },
    )


def _component_hash(module: torch.nn.Module) -> str:
    return canonical_tensor_tree_sha256(module.state_dict())


def test_local_arms_share_initialization_and_bypass_coarse_fusion() -> None:
    backend = LocalQ1Backend(
        mode=LocalQ1BackendMode.FIXTURE_NON_RESULT,
        model_template=_template(),
    )
    dual_arm = _arm(backend, AblationArmName.LOCAL_DUAL_SOFTMAX)
    dust_arm = _arm(backend, AblationArmName.LOCAL_DUSTBIN_SINKHORN)
    dual = backend.create_session(dual_arm, seed=81)
    dust = backend.create_session(dust_arm, seed=81)

    assert dual.initial_model_state_sha256 == dust.initial_model_state_sha256
    assert dual.score_source.value == dust.score_source.value == "local"
    assert all(
        name.startswith("local_model.") for name in dual.trainable_parameter_names
    )
    assert all(
        name.startswith("local_model.") for name in dust.trainable_parameter_names
    )
    dual_config = json.loads(json.dumps(dual.model_config))
    dust_config = json.loads(json.dumps(dust.model_config))
    dual_config["architecture"]["local"]["matcher_mode"] = "dustbin_sinkhorn"
    assert dual_config == dust_config

    batch = _prepared()
    coarse_before = _component_hash(dual.model.coarse_model)
    fusion_before = _component_hash(dual.model.fusion)
    local_before = _component_hash(dual.model.local_model)
    trained = dual.train_batch(batch)
    assert trained.loss == pytest.approx(trained.loss)
    assert trained.valid_count == batch.sample_count
    assert trained.diagnostics["score_source"] == "local"
    assert trained.diagnostics["coarse_forward_count"] == 0
    assert trained.diagnostics["fusion_forward_count"] == 0
    assert trained.diagnostics["dustbin_local_forward_count"] == 0
    assert trained.diagnostics["sinkhorn_problem_count"] == 0
    assert trained.diagnostics["result_eligible"] is False
    assert _component_hash(dual.model.coarse_model) == coarse_before
    assert _component_hash(dual.model.fusion) == fusion_before
    assert _component_hash(dual.model.local_model) != local_before

    predicted = dust.predict_batch(batch, evidence=EvidenceMode.LOCAL)
    assert torch.isfinite(predicted.probability).all().item()
    assert predicted.valid.tolist() == [True, True]
    assert predicted.diagnostics["score_source"] == "local"
    assert predicted.diagnostics["matcher_mode"] == "dustbin_sinkhorn"
    assert predicted.diagnostics["coarse_forward_count"] == 0
    assert predicted.diagnostics["fusion_forward_count"] == 0
    assert predicted.diagnostics["dustbin_local_forward_count"] == 2
    assert predicted.diagnostics["sinkhorn_problem_count"] == 16
    assert predicted.diagnostics["sinkhorn_provider_valid_problem_count"] == 16
    assert predicted.diagnostics["provider_valid_local_problem_count"] == 16
    assert predicted.diagnostics["provider_candidate_count_per_forward"] == 8
    assert predicted.diagnostics["provider_candidate_valid_count_per_forward"] == 8
    assert predicted.diagnostics["provider_geometry_valid_sample_count"] == 2
    assert (
        predicted.diagnostics["effective_training_valid_candidate_count_per_forward"]
        == 8
    )
    assert (
        predicted.diagnostics["effective_decision_valid_candidate_count_per_forward"]
        == 8
    )
    assert predicted.diagnostics["converged_count"] == batch.payload.candidate_count
    assert predicted.diagnostics["row_residual_max"] is not None
    assert predicted.diagnostics["row_residual_p99_9"] is not None
    assert predicted.diagnostics["ab_swap_logit_max_abs_error"] <= 1e-5
    assert predicted.diagnostics["ab_swap_probability_max_abs_error"] <= 1e-5


def test_keypoint_dual_and_sinkhorn_train_from_same_masked_batch() -> None:
    backend = LocalQ1Backend(
        mode=LocalQ1BackendMode.FIXTURE_NON_RESULT,
        model_template=_template(),
    )
    dual_arm = _arm(backend, AblationArmName.KEYPOINT_DUAL_SOFTMAX)
    dust_arm = _arm(backend, AblationArmName.KEYPOINT_DUSTBIN_SINKHORN)
    dual = backend.create_session(dual_arm, seed=83)
    dust = backend.create_session(dust_arm, seed=83)
    batch = _prepared(_keypoint_payload())

    assert dual.initial_model_state_sha256 == dust.initial_model_state_sha256
    assert dual.model_config["candidate_representation"] == KEYPOINT_REPRESENTATION
    assert dust.model_config["candidate_representation"] == KEYPOINT_REPRESENTATION
    assert batch.payload.model_inputs()["correspondence_mask"] is (
        batch.payload.correspondence_mask
    )
    assert (~batch.payload.correspondence_mask).any().item()

    dual_result = dual.train_batch(batch)
    dust_result = dust.train_batch(batch)
    assert math.isfinite(dual_result.loss)
    assert math.isfinite(dust_result.loss)
    assert dual_result.valid_count == dust_result.valid_count == batch.sample_count
    assert (
        dual_result.diagnostics["candidate_representation"]
        == dust_result.diagnostics["candidate_representation"]
        == KEYPOINT_REPRESENTATION
    )


def test_exact_fifth_arm_uses_same_inputs_and_initialization_but_adds_nll() -> None:
    backend = LocalQ1Backend(
        mode=LocalQ1BackendMode.FIXTURE_NON_RESULT,
        model_template=_template(),
        exact_seam_step_config=ExactSeamStepConfig(loss_weight=0.25),
    )
    weak = backend.create_session(
        _arm(backend, AblationArmName.KEYPOINT_DUSTBIN_SINKHORN), seed=87
    )
    exact = backend.create_session(
        _arm(
            backend,
            AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM,
        ),
        seed=87,
    )
    batch = _prepared(_exact_keypoint_payload())

    assert weak.initial_model_state_sha256 == exact.initial_model_state_sha256
    assert not any(name.startswith("exact_") for name in batch.payload.model_inputs())
    assert batch.payload.exact_loss_targets() is not None

    weak_result = weak.train_batch(batch)
    exact_result = exact.train_batch(batch)

    assert weak_result.diagnostics["exact_assignment_supervision_enabled"] is False
    assert weak_result.diagnostics["exact_assignment_loss"] == 0.0
    assert exact_result.diagnostics["exact_assignment_supervision_enabled"] is True
    assert exact_result.diagnostics["exact_assignment_loss"] > 0.0
    assert exact_result.diagnostics["exact_supervised_match_count"] == 1
    assert exact_result.diagnostics["exact_supervised_dustbin_a_count"] > 0
    assert exact_result.diagnostics["exact_supervised_dustbin_b_count"] > 0
    assert math.isfinite(exact_result.loss)


def test_exact_arm_inference_does_not_require_ground_truth_targets() -> None:
    backend = LocalQ1Backend(
        mode=LocalQ1BackendMode.FIXTURE_NON_RESULT,
        model_template=_template(),
    )
    exact = backend.create_session(
        _arm(
            backend,
            AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM,
        ),
        seed=89,
    )
    targetless = _prepared(_keypoint_payload())

    prediction = exact.predict_batch(targetless, evidence=EvidenceMode.LOCAL)
    assert prediction.probability.shape == (2,)
    with pytest.raises(LocalQ1BackendError, match="training arm requires"):
        exact.train_batch(targetless)


def test_exact_targets_change_prepared_digest_not_candidate_digest() -> None:
    weak_payload = _keypoint_payload()
    exact_payload = _exact_keypoint_payload()
    weak_prepared, weak_candidate = local_q1_prepared_digests(weak_payload)
    exact_prepared, exact_candidate = local_q1_prepared_digests(exact_payload)

    assert weak_candidate == exact_candidate
    assert weak_prepared != exact_prepared


def test_ragged_batch_rejects_exact_match_on_disabled_model_edge() -> None:
    payload = _exact_keypoint_payload()
    target_a = payload.exact_assignment_target_a.clone()
    target_b = payload.exact_assignment_target_b.clone()
    target_a[0] = torch.tensor([1, -2])
    target_b[0] = torch.tensor([-2, 0])

    with pytest.raises(ValueError, match="disabled correspondence edge"):
        replace(
            payload,
            exact_assignment_target_a=target_a,
            exact_assignment_target_b=target_b,
        )


def test_payload_digests_fail_before_model_or_optimizer_mutation() -> None:
    backend = LocalQ1Backend(
        mode=LocalQ1BackendMode.FIXTURE_NON_RESULT,
        model_template=_template(),
    )
    session = backend.create_session(
        _arm(backend, AblationArmName.LOCAL_DUAL_SOFTMAX), seed=91
    )
    batch = _prepared()
    state_before = canonical_tensor_tree_sha256(session.model.state_dict())
    assert not session.optimizer.state
    batch.payload.labels[0] = ~batch.payload.labels[0]

    with pytest.raises(LocalQ1BackendError, match="prepared input digest"):
        session.train_batch(batch)

    assert canonical_tensor_tree_sha256(session.model.state_dict()) == state_before
    assert not session.optimizer.state
    assert session.is_tainted is False
    assert session._forward_counts == {"coarse": 0, "local": 0, "fusion": 0}
    batch.payload.labels[0] = ~batch.payload.labels[0]
    clean_prediction = session.predict_batch(
        _prepared(batch.payload), evidence=EvidenceMode.LOCAL
    )
    assert clean_prediction.probability.shape == (2,)


def test_stale_local_digest_and_cross_arm_population_substitution_are_rejected() -> (
    None
):
    backend = LocalQ1Backend(
        mode=LocalQ1BackendMode.FIXTURE_NON_RESULT,
        model_template=_template(),
    )
    session = backend.create_session(
        _arm(backend, AblationArmName.LOCAL_DUAL_SOFTMAX), seed=101
    )
    original = _prepared()
    changed_payload = replace(
        original.payload,
        local_a=(original.payload.local_a + 1e-3).contiguous(),
    )
    changed_prepared, changed_local = local_q1_prepared_digests(changed_payload)
    stale_local = replace(
        original,
        payload=changed_payload,
        prepared_input_sha256=changed_prepared,
    )
    with pytest.raises(LocalQ1BackendError, match="local candidate digest"):
        session.predict_batch(stale_local, evidence=EvidenceMode.LOCAL)

    session.predict_batch(original, evidence=EvidenceMode.LOCAL)
    fully_rehashed = replace(
        original,
        payload=changed_payload,
        prepared_input_sha256=changed_prepared,
        local_candidate_sha256=changed_local,
    )
    with pytest.raises(LocalQ1BackendError, match="differ across LOCAL-Q1 arms"):
        session.predict_batch(fully_rehashed, evidence=EvidenceMode.LOCAL)


def _coarse_checkpoint_config(template: PairwiseModelConfig):
    return {
        "model": {
            "name": "SymmetricCoarseSiamese",
            "input_channels": template.coarse.input_channels,
            "widths": template.coarse.widths,
            "embedding_dim": template.coarse.embedding_dim,
            "hidden_dim": template.coarse.hidden_dim,
            "input_shape": LOCAL_Q1_COARSE_INPUT_SHAPE,
        }
    }


def _local_checkpoint_config(template: PairwiseModelConfig):
    return {
        "model_config": {
            "architecture": {"local": asdict(template.local)},
            "training": {
                "score_source": "local",
                "trainable_components": ["local_model"],
            },
        }
    }


def _checkpoint_binding(path: Path, receipt, config, scope):
    semantic_sha = canonical_config_hash(checkpoint_semantic_projection(config, scope))
    authority_file_sha = hashlib.sha256(
        ("authority-file:" + str(path.name)).encode("utf-8")
    ).hexdigest()
    authority_content_sha = hashlib.sha256(
        ("authority-content:" + str(path.name)).encode("utf-8")
    ).hexdigest()
    authority_identity = checkpoint_authority_identity_sha256(
        model_scope=scope,
        checkpoint_file_sha256=receipt.file_sha256,
        checkpoint_canonical_content_sha256=receipt.canonical_content_sha256,
        checkpoint_config_sha256=receipt.config_hash,
        semantic_projection_sha256=semantic_sha,
        authority_kind=LOCAL_Q1_AUTHORITY_KIND,
        authority_status=LOCAL_Q1_AUTHORITY_STATUS,
        authority_receipt_file_sha256=authority_file_sha,
        authority_receipt_content_sha256=authority_content_sha,
    )
    return TrustedCheckpointBinding(
        path=path,
        file_sha256=receipt.file_sha256,
        canonical_content_sha256=receipt.canonical_content_sha256,
        config_sha256=receipt.config_hash,
        expected_config=config,
        model_scope=scope,
        semantic_projection_sha256=semantic_sha,
        authority_kind=LOCAL_Q1_AUTHORITY_KIND,
        authority_status=LOCAL_Q1_AUTHORITY_STATUS,
        authority_receipt_file_sha256=authority_file_sha,
        authority_receipt_content_sha256=authority_content_sha,
        authority_identity_sha256=authority_identity,
    )


def test_formal_fused_loads_both_locked_checkpoints_and_trains_only_fusion(
    tmp_path: Path,
) -> None:
    template = _template(iterations=100, tolerance=1e-3)
    torch.manual_seed(301)
    coarse_source = SymmetricCoarseSiamese(
        input_channels=template.coarse.input_channels,
        widths=template.coarse.widths,
        embedding_dim=template.coarse.embedding_dim,
        hidden_dim=template.coarse.hidden_dim,
    )
    coarse_config = _coarse_checkpoint_config(template)
    coarse_path = tmp_path / "c0.pt"
    coarse_receipt = save_checkpoint(
        coarse_path,
        coarse_source,
        config=coarse_config,
        epoch=5,
        provenance={"sealed_real_test_accessed": False},
    )

    torch.manual_seed(302)
    local_source = DunhuangPairwiseV02(template)
    local_config = _local_checkpoint_config(template)
    local_path = tmp_path / "local.pt"
    local_receipt = save_checkpoint(
        local_path,
        local_source,
        config=local_config,
        epoch=3,
        provenance={"sealed_real_test_accessed": False},
    )
    bindings = FusedCheckpointBindings(
        coarse=_checkpoint_binding(
            coarse_path,
            coarse_receipt,
            coarse_config,
            CheckpointModelScope.COARSE_MODEL,
        ),
        local=_checkpoint_binding(
            local_path,
            local_receipt,
            local_config,
            CheckpointModelScope.PAIRWISE_MODEL,
        ),
    )
    backend = LocalQ1Backend(
        mode=LocalQ1BackendMode.FORMAL,
        model_template=template,
        fused_checkpoint_bindings=bindings,
    )
    session = backend.create_session(_arm(backend, AblationArmName.FUSED), seed=303)
    assert session.initialization_authority == (
        "trusted_external_c0_and_local_checkpoints"
    )
    assert all(name.startswith("fusion.") for name in session.trainable_parameter_names)
    assert _component_hash(session.model.coarse_model) == _component_hash(coarse_source)
    assert _component_hash(session.model.local_model) == _component_hash(
        local_source.local_model
    )

    coarse_before = _component_hash(session.model.coarse_model)
    local_before = _component_hash(session.model.local_model)
    fusion_before = _component_hash(session.model.fusion)
    result = session.train_batch(_prepared())
    assert result.valid_count == 2
    assert result.diagnostics["score_source"] == "fused"
    assert result.diagnostics["coarse_forward_count"] == 2
    assert result.diagnostics["local_forward_count"] == 2
    assert result.diagnostics["fusion_forward_count"] == 2
    assert result.diagnostics["dustbin_local_forward_count"] == 2
    assert result.diagnostics["sinkhorn_problem_count"] == 16
    assert result.diagnostics["result_eligible"] is True
    assert result.diagnostics["authority_receipt_opened_by_backend"] is False
    assert _component_hash(session.model.coarse_model) == coarse_before
    assert _component_hash(session.model.local_model) == local_before
    assert _component_hash(session.model.fusion) != fusion_before


def test_formal_mode_rejects_unstable_sinkhorn_and_random_fused() -> None:
    with pytest.raises(LocalQ1BackendError, match="temperature=0.25"):
        LocalQ1Backend(
            mode=LocalQ1BackendMode.FORMAL,
            model_template=_template(iterations=50, tolerance=1e-3),
        )

    backend = LocalQ1Backend(
        mode=LocalQ1BackendMode.FORMAL,
        model_template=_template(iterations=100, tolerance=1e-3),
    )
    with pytest.raises(LocalQ1BackendError, match="trusted C0/local checkpoints"):
        backend.model_config_for(AblationArmName.FUSED)


def test_checkpoint_binding_rejects_unlocked_config_hash(tmp_path: Path) -> None:
    with pytest.raises(LocalQ1BackendError, match="config hash"):
        TrustedCheckpointBinding(
            path=tmp_path / "not-opened.pt",
            file_sha256="a" * 64,
            canonical_content_sha256="b" * 64,
            config_sha256="c" * 64,
            expected_config={"different": "config"},
            model_scope=CheckpointModelScope.COARSE_MODEL,
            semantic_projection_sha256="d" * 64,
            authority_kind=LOCAL_Q1_AUTHORITY_KIND,
            authority_status=LOCAL_Q1_AUTHORITY_STATUS,
            authority_receipt_file_sha256="e" * 64,
            authority_receipt_content_sha256="f" * 64,
            authority_identity_sha256="0" * 64,
        )


def _fake_checkpoint_receipt(label: str, config):
    return SimpleNamespace(
        file_sha256=hashlib.sha256((label + ":file").encode("utf-8")).hexdigest(),
        canonical_content_sha256=hashlib.sha256(
            (label + ":content").encode("utf-8")
        ).hexdigest(),
        config_hash=canonical_config_hash(config),
    )


def test_semantic_projection_and_authority_identity_fail_closed(
    tmp_path: Path,
) -> None:
    template = _template(iterations=100, tolerance=1e-3)
    config = _coarse_checkpoint_config(template)
    receipt = _fake_checkpoint_receipt("semantic-authority", config)
    binding = _checkpoint_binding(
        tmp_path / "semantic-authority.pt",
        receipt,
        config,
        CheckpointModelScope.COARSE_MODEL,
    )
    portable = binding.portable_identity()
    assert portable["semantic_projection_sha256"] == (
        binding.semantic_projection_sha256
    )
    assert portable["external_authority"]["receipt_opened_by_backend"] is False
    assert (
        "future_runner_must_validate"
        in portable["external_authority"]["validation_boundary"]
    )

    with pytest.raises(LocalQ1BackendError, match="semantic projection hash"):
        replace(binding, semantic_projection_sha256="0" * 64)
    with pytest.raises(LocalQ1BackendError, match="authority identity"):
        replace(binding, authority_identity_sha256="0" * 64)
    with pytest.raises(LocalQ1BackendError, match="lowercase SHA-256"):
        replace(binding, authority_receipt_file_sha256=None)
    with pytest.raises(LocalQ1BackendError, match="authority status"):
        replace(binding, authority_status="claimed_but_not_validated")

    local_config_missing_fixed_path = {
        "model_config": {"architecture": {"local": asdict(template.local)}}
    }
    missing_receipt = _fake_checkpoint_receipt(
        "missing-local-training-scope", local_config_missing_fixed_path
    )
    with pytest.raises(LocalQ1BackendError, match="training must be a mapping"):
        _checkpoint_binding(
            tmp_path / "missing-local-training-scope.pt",
            missing_receipt,
            local_config_missing_fixed_path,
            CheckpointModelScope.PAIRWISE_MODEL,
        )


def test_formal_fused_rejects_shape_compatible_but_semantically_wrong_local(
    tmp_path: Path,
) -> None:
    target = _template(iterations=100, tolerance=1e-3)
    wrong = replace(
        target,
        local=replace(
            target.local,
            matcher_temperature=0.5,
            sinkhorn_iterations=7,
            sinkhorn_tolerance=0.25,
        ),
    )
    coarse_config = _coarse_checkpoint_config(target)
    wrong_local_config = _local_checkpoint_config(wrong)
    bindings = FusedCheckpointBindings(
        coarse=_checkpoint_binding(
            tmp_path / "semantic-c0.pt",
            _fake_checkpoint_receipt("semantic-c0", coarse_config),
            coarse_config,
            CheckpointModelScope.COARSE_MODEL,
        ),
        local=_checkpoint_binding(
            tmp_path / "semantic-wrong-local.pt",
            _fake_checkpoint_receipt("semantic-wrong-local", wrong_local_config),
            wrong_local_config,
            CheckpointModelScope.PAIRWISE_MODEL,
        ),
    )
    backend = LocalQ1Backend(
        mode=LocalQ1BackendMode.FORMAL,
        model_template=target,
        fused_checkpoint_bindings=bindings,
    )
    arm = _arm(backend, AblationArmName.FUSED)
    with pytest.raises(LocalQ1BackendError, match="projection differs from target"):
        backend.create_session(arm, seed=808)


def test_late_execution_failure_permanently_taints_session(monkeypatch) -> None:
    backend = LocalQ1Backend(
        mode=LocalQ1BackendMode.FIXTURE_NON_RESULT,
        model_template=_template(),
    )
    session = backend.create_session(
        _arm(backend, AblationArmName.LOCAL_DUAL_SOFTMAX), seed=404
    )
    before = canonical_tensor_tree_sha256(session.model.state_dict())

    def fail_late(*_arguments, **_keywords):
        raise LocalQ1BackendError("late swap audit failure")

    monkeypatch.setattr(session, "_swap_errors", fail_late)
    with pytest.raises(LocalQ1BackendError, match="late swap audit failure"):
        session.train_batch(_prepared())
    assert canonical_tensor_tree_sha256(session.model.state_dict()) != before
    assert session.is_tainted is True
    assert session.taint_reason == "train_batch:LocalQ1BackendError"
    with pytest.raises(LocalQ1BackendError, match="permanently tainted"):
        session.predict_batch(_prepared(), evidence=EvidenceMode.LOCAL)


def test_step_config_is_frozen_and_rechecked_before_every_batch() -> None:
    with pytest.raises(LocalQ1BackendError, match="frozen residual/swap"):
        LocalQ1Backend(
            mode=LocalQ1BackendMode.FORMAL,
            model_template=_template(iterations=100, tolerance=1e-3),
            step_config=replace(LocalQ1StepConfig(), sinkhorn_residual_weight=0.0),
        )
    with pytest.raises(LocalQ1BackendError, match="frozen residual/swap"):
        LocalQ1Backend(
            mode=LocalQ1BackendMode.FORMAL,
            model_template=_template(iterations=100, tolerance=1e-3),
            step_config=replace(LocalQ1StepConfig(), swap_tolerance=999.0),
        )

    backend = LocalQ1Backend(
        mode=LocalQ1BackendMode.FIXTURE_NON_RESULT,
        model_template=_template(),
    )
    session = backend.create_session(
        _arm(backend, AblationArmName.LOCAL_DUAL_SOFTMAX), seed=505
    )
    with pytest.raises(AttributeError):
        session.step_config = replace(LocalQ1StepConfig(), sinkhorn_residual_weight=0.0)
    session._step_config = replace(LocalQ1StepConfig(), sinkhorn_residual_weight=0.0)
    with pytest.raises(LocalQ1BackendError, match="changed after construction"):
        session.predict_batch(_prepared(), evidence=EvidenceMode.LOCAL)
    assert session.is_tainted is False
    assert session._forward_counts == {"coarse": 0, "local": 0, "fusion": 0}

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from staging.pairwise_v0_2.baselines.rachel_pairingnet_benchmark import (
    LAST_CHECKPOINT_KIND,
    METHOD_ID,
    OFFICIAL_COMMIT,
    WINNER_CHECKPOINT_KIND,
    PairingNetRachelRunConfig,
    PairingNetBenchmarkError,
    PairingNetRachelAdapted,
    PairingNetRachelModelConfig,
    SE2Estimate,
    TranslationEstimate,
    _adapter_identity_hashes,
    _fit_frozen_validation_threshold,
    _last_checkpoint_payload,
    _load_last_checkpoint,
    _primary_geometry_record,
    _ring_neighbours,
    _sample_ordered_patches,
    _validated_run_state_paths,
    audit_official_source,
    audit_train_val_population,
    compute_pairingnet_loss,
    frozen_inference_contract,
    infer_pairingnet_batch,
    is_qualifying_improvement,
    load_frozen_pairingnet_checkpoint,
    load_frozen_validation_threshold,
    official_style_se2_ransac,
    pairingnet_diagonal_morphology,
    pairingnet_focal_loss,
    selection_key,
    summarize_validation_records,
    translation_only_consensus,
)
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelBatch


def _row(pair_id, split, label, unit_a, unit_b=None):
    if unit_b is None:
        unit_b = unit_a
    return {
        "pair_id": pair_id,
        "split": split,
        "label": label,
        "fragment_a": {"split_unit_id": unit_a},
        "fragment_b": {"split_unit_id": unit_b},
    }


def _write_manifest(root: Path, split: str, rows):
    pairs = root / "pairs"
    pairs.mkdir(parents=True, exist_ok=True)
    (pairs / (split + ".jsonl")).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _small_config():
    return PairingNetRachelModelConfig(
        canvas_size=32,
        contour_cap=8,
        resgcn_layers=1,
        contour_neighbor_radius=2,
        pair_hidden_dim=8,
        translation_min_candidates=2,
        se2_ransac_min_candidates=4,
    )


def test_population_audit_is_train_val_only_balanced_and_lineage_disjoint(tmp_path):
    _write_manifest(
        tmp_path,
        "train",
        [_row("t-pos", "train", True, "train-a"), _row("t-neg", "train", False, "train-b", "train-c")],
    )
    _write_manifest(
        tmp_path,
        "val",
        [_row("v-pos", "val", True, "val-a"), _row("v-neg", "val", False, "val-b", "val-c")],
    )
    # Deliberately malformed and unread: a successful audit proves that this
    # path is neither required nor discovered.
    (tmp_path / "pairs" / "test.jsonl").write_text("not-json\n", encoding="utf-8")
    report = audit_train_val_population(tmp_path, require_formal_counts=False)
    assert report["population"]["train"]["rows"] == 2
    assert report["population"]["val"]["rows"] == 2
    assert report["parent_lineage_disjoint"] is True
    assert report["sealed_synthetic_accessed"] is False
    assert report["official_reference"]["commit"] == OFFICIAL_COMMIT


def test_population_audit_rejects_parent_lineage_overlap(tmp_path):
    _write_manifest(
        tmp_path,
        "train",
        [_row("t-pos", "train", True, "shared"), _row("t-neg", "train", False, "train-b")],
    )
    _write_manifest(
        tmp_path,
        "val",
        [_row("v-pos", "val", True, "shared"), _row("v-neg", "val", False, "val-b")],
    )
    with pytest.raises(PairingNetBenchmarkError, match="leakage"):
        audit_train_val_population(tmp_path, require_formal_counts=False)


def test_population_audit_locks_formal_24000_3000_counts(tmp_path):
    _write_manifest(
        tmp_path,
        "train",
        [_row("t-pos", "train", True, "train-a"), _row("t-neg", "train", False, "train-b")],
    )
    _write_manifest(
        tmp_path,
        "val",
        [_row("v-pos", "val", True, "val-a"), _row("v-neg", "val", False, "val-b")],
    )
    with pytest.raises(PairingNetBenchmarkError, match="24000 train / 3000 val"):
        audit_train_val_population(tmp_path)


def test_integer_patch_sampler_reads_ordered_centres():
    image = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    points = torch.tensor([[[2.0, 2.0], [1.0, 3.0]]])
    patches = _sample_ordered_patches(image, points, 3)
    assert patches.shape == (1, 2, 1, 3, 3)
    torch.testing.assert_close(patches[0, 0, 0], image[0, 0, 1:4, 1:4])
    assert patches[0, 1, 0, 1, 1].item() == image[0, 0, 1, 3].item()


def test_model_dual_softmax_masks_padding_and_backpropagates():
    torch.manual_seed(7)
    model = PairingNetRachelAdapted(_small_config())
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, 5:26, 6:27] = 1.0
    points = torch.tensor(
        [
            [
                [5, 6],
                [5, 12],
                [5, 18],
                [5, 26],
                [12, 26],
                [25, 26],
                [25, 12],
                [25, 6],
            ]
        ]
        * 2,
        dtype=torch.float32,
    )
    valid = torch.tensor([[True] * 8, [True] * 6 + [False] * 2])
    output = model(mask, mask, points, points, valid, valid)
    assert output.similarity.shape == (2, 8, 8)
    assert torch.count_nonzero(output.similarity[1, 6:, :]) == 0
    assert torch.count_nonzero(output.similarity[1, :, 6:]) == 0
    target = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [-1, -1, -1, -1, -1, -1, -2, -2]]
    )
    loss = compute_pairingnet_loss(output, torch.tensor([1.0, 0.0]), target)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_train_mode_valid_features_and_bn_stats_are_cap_invariant():
    """Padded nodes must not enter either patch BN or ResGCN BN statistics."""

    def config(cap):
        return PairingNetRachelModelConfig(
            canvas_size=32,
            contour_cap=cap,
            resgcn_layers=1,
            contour_neighbor_radius=2,
            pair_hidden_dim=8,
            translation_min_candidates=2,
            se2_ransac_min_candidates=4,
        )

    torch.manual_seed(17)
    model_cap4 = PairingNetRachelAdapted(config(4))
    model_cap8 = PairingNetRachelAdapted(config(8))
    model_cap8.load_state_dict(model_cap4.state_dict(), strict=True)
    model_cap4.train()
    model_cap8.train()

    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 5:27, 6:26] = 1.0
    points4 = torch.tensor(
        [[[5.0, 6.0], [5.0, 25.0], [26.0, 25.0], [26.0, 6.0]]]
    )
    points8 = torch.cat(
        (
            points4,
            torch.tensor(
                [[[0.0, 0.0], [31.0, 31.0], [0.0, 31.0], [31.0, 0.0]]]
            ),
        ),
        dim=1,
    )
    valid4 = torch.ones(1, 4, dtype=torch.bool)
    valid8 = torch.tensor([[True, True, True, True, False, False, False, False]])

    output4 = model_cap4(mask, mask, points4, points4, valid4, valid4)
    output8 = model_cap8(mask, mask, points8, points8, valid8, valid8)

    torch.testing.assert_close(output4.fused_a, output8.fused_a[:, :4], atol=1e-6, rtol=0)
    torch.testing.assert_close(output4.fused_b, output8.fused_b[:, :4], atol=1e-6, rtol=0)
    torch.testing.assert_close(
        output4.similarity, output8.similarity[:, :4, :4], atol=1e-7, rtol=0
    )
    torch.testing.assert_close(
        output4.pair_probability, output8.pair_probability, atol=1e-7, rtol=0
    )

    bn4 = {
        name: module
        for name, module in model_cap4.named_modules()
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
    }
    bn8 = {
        name: module
        for name, module in model_cap8.named_modules()
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
    }
    assert bn4.keys() == bn8.keys()
    for name in bn4:
        torch.testing.assert_close(bn4[name].running_mean, bn8[name].running_mean)
        torch.testing.assert_close(bn4[name].running_var, bn8[name].running_var)
        torch.testing.assert_close(
            bn4[name].num_batches_tracked, bn8[name].num_batches_tracked
        )


def test_ring_neighbourhood_uses_valid_length_not_tensor_cap():
    values4 = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    values8 = torch.cat(
        (values4, torch.full((1, 4, 1), 10_000.0)), dim=1
    )
    valid4 = torch.ones(1, 4, dtype=torch.bool)
    valid8 = torch.tensor([[True, True, True, True, False, False, False, False]])
    neighbours4, neighbour_valid4 = _ring_neighbours(values4, valid4, radius=2)
    neighbours8, neighbour_valid8 = _ring_neighbours(values8, valid8, radius=2)
    torch.testing.assert_close(neighbours4, neighbours8[:, :4])
    assert torch.equal(neighbour_valid4, neighbour_valid8[:, :4])
    with pytest.raises(ValueError, match="contiguous prefix"):
        _ring_neighbours(
            values4,
            torch.tensor([[True, False, True, True]]),
            radius=2,
        )


def test_focal_loss_matches_released_formula_on_tiny_matrix():
    similarity = torch.tensor([[[0.8, 0.1], [0.2, 0.7]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    target = torch.tensor([[0, 1]])
    actual = pairingnet_focal_loss(similarity, target, valid, valid)
    positive = torch.tensor([0.8, 0.7])
    negative = torch.tensor([0.1, 0.2])
    expected = 400 * (
        (-0.55 * (1 - positive) ** 8 * torch.log(positive)).sum()
        + (-(1 - 0.55) * negative**8 * torch.log(1 - negative)).sum()
    ) / 4
    torch.testing.assert_close(actual, expected)


def _translated_reverse_contours(count=24, translation=(3.0, 7.0)):
    source = np.stack(
        (np.arange(count, dtype=np.float32), np.arange(count, dtype=np.float32) * 2),
        axis=1,
    )
    target = np.zeros_like(source)
    for source_index in range(count):
        target[count - 1 - source_index] = source[source_index] + np.asarray(translation)
    similarity = np.zeros((count, count), dtype=np.float32)
    for source_index in range(count):
        similarity[source_index, count - 1 - source_index] = 0.1
    valid = np.ones(count, dtype=np.bool_)
    return source, target, similarity, valid


def test_diagonal_morphology_and_translation_consensus_recover_known_shift():
    source, target, similarity, valid = _translated_reverse_contours()
    enhanced = pairingnet_diagonal_morphology(similarity)
    assert np.count_nonzero(enhanced > 0.006) >= 4
    estimate = translation_only_consensus(
        similarity,
        source,
        target,
        valid,
        valid,
        PairingNetRachelModelConfig(
            canvas_size=64,
            contour_cap=24,
            resgcn_layers=1,
            contour_neighbor_radius=2,
            translation_min_candidates=4,
            se2_ransac_min_candidates=4,
        ),
    )
    assert estimate.valid
    np.testing.assert_allclose(estimate.translation_rc, [3.0, 7.0], atol=1e-6)


def test_official_style_ransac_is_explicitly_secondary_and_recovers_translation():
    source, target, similarity, valid = _translated_reverse_contours()
    config = PairingNetRachelModelConfig(
        canvas_size=64,
        contour_cap=24,
        resgcn_layers=1,
        contour_neighbor_radius=2,
        translation_min_candidates=4,
        se2_ransac_min_candidates=4,
    )
    estimate = official_style_se2_ransac(
        similarity, source, target, valid, valid, config
    )
    assert isinstance(estimate, SE2Estimate)
    assert estimate.valid
    assert estimate.matrix_xy is not None
    assert estimate.matrix_xy[0, 2] == pytest.approx(7.0, abs=1e-3)
    assert estimate.matrix_xy[1, 2] == pytest.approx(3.0, abs=1e-3)


def _metric_record(pair_id, label, score, pose, error=None, ermse=2.0, hd=3.0, nte=0.1):
    geometry = {
        "pose_valid": pose,
        "translation_error_px": error,
        "pairingnet_e_rmse": ermse,
        "pairingnet_hd_px": hd,
        "pairingnet_nte": nte,
    }
    return {
        "pair_id": pair_id,
        "label": label,
        "pair_probability": score,
        "primary_geometry": geometry,
        "secondary_se2": None,
    }


def test_assembly_edge_metrics_jointly_score_pair_and_translation():
    records = [
        _metric_record("p1", True, 0.9, True, 4.0),
        _metric_record("p2", True, 0.9, True, 9.0),
        _metric_record("n1", False, 0.9, True),
        _metric_record("n2", False, 0.1, False),
    ]
    report = summarize_validation_records(records)
    at5 = report["primary_translation_only"]["assembly_edge"]["5px"]
    assert (at5["tp"], at5["fp"], at5["fn"]) == (1, 2, 1)
    assert at5["precision"] == pytest.approx(1 / 3)
    assert at5["recall"] == pytest.approx(1 / 2)
    assert at5["f1"] == pytest.approx(0.4)
    at10 = report["primary_translation_only"]["assembly_edge"]["10px"]
    assert (at10["tp"], at10["fp"], at10["fn"]) == (2, 1, 0)
    assert at10["f1"] == pytest.approx(0.8)
    rotation = report["primary_translation_only"]["rotation_error"]
    assert rotation["status"] == "not_applicable"


def test_pairingnet_rr_uses_released_ermse_lt_four_threshold():
    records = [
        _metric_record("p1", True, 0.9, True, 1.0, ermse=3.999),
        _metric_record("p2", True, 0.9, True, 1.0, ermse=4.0),
        _metric_record("n1", False, 0.1, False),
        _metric_record("n2", False, 0.1, False),
    ]
    report = summarize_validation_records(records)
    assert report["pairingnet_compatible_secondary"]["rr"] == 0.5


def test_primary_geometry_keeps_re_not_applicable_and_uses_identity_fallback():
    points = np.asarray([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.float32)
    target = points + np.asarray([2, 5], dtype=np.float32)
    valid = np.ones(4, dtype=np.bool_)
    target_index = np.arange(4, dtype=np.int64)
    record = _primary_geometry_record(
        estimate=TranslationEstimate(False, None, 0, 0),
        points_rc_a=points,
        points_rc_b=target,
        valid_a=valid,
        valid_b=valid,
        target_a=target_index,
        ground_truth_translation_rc=np.asarray([2, 5], dtype=np.float32),
    )
    assert record["pose_valid"] is False
    assert record["translation_error_px"] is None
    assert record["compatibility_fallback_translation_error_px"] == pytest.approx(
        math.sqrt(29)
    )
    assert record["pairingnet_e_rmse"] == pytest.approx(29 ** 0.25)


def test_selection_key_is_threshold_free_joint_pair_and_geometry():
    records = [
        _metric_record("p1", True, 0.9, True, 1.0),
        _metric_record("p2", True, 0.9, True, 1.0),
        _metric_record("n1", False, 0.1, False),
        _metric_record("n2", False, 0.1, False),
    ]
    report = summarize_validation_records(records)
    key = selection_key(report, 3)
    pair = report["pair_decision_secondary"]["cluster_balanced"]
    translation = report["primary_translation_only"][
        "positive_translation_success_unconditional"
    ]["8px"]
    rr = report["pairingnet_compatible_secondary"]["rr"]
    assert key[0] == pytest.approx(
        (pair["auroc"] * pair["auprc"] * translation * rr) ** 0.25
    )
    assert key[-1] == -3.0


def test_selection_is_independent_of_pair_operating_threshold():
    records = [
        _metric_record("p1", True, 0.7, True, 1.0),
        _metric_record("p2", True, 0.4, True, 9.0, ermse=5.0),
        _metric_record("n1", False, 0.6, True),
        _metric_record("n2", False, 0.1, False),
    ]
    low = summarize_validation_records(records, pair_threshold=0.2)
    high = summarize_validation_records(records, pair_threshold=0.8)
    assert selection_key(low, 7) == selection_key(high, 7)


def test_plateau_qualification_uses_half_percent_relative_improvement():
    assert is_qualifying_improvement(-math.inf, 0.0, 0.005)
    assert not is_qualifying_improvement(0.0, 0.0, 0.005)
    assert is_qualifying_improvement(0.0, 0.01, 0.005)
    assert not is_qualifying_improvement(0.8, 0.8039, 0.005)
    assert is_qualifying_improvement(0.8, 0.804, 0.005)


def test_frozen_inference_contract_names_score_translation_and_correspondence():
    contract = frozen_inference_contract()
    assert contract["method_id"] == METHOD_ID
    assert contract["checkpoint_restore"]["required_official_commit"] == OFFICIAL_COMMIT
    assert (
        contract["checkpoint_restore"]["required_checkpoint_kind"]
        == WINNER_CHECKPOINT_KIND
    )
    assert contract["checkpoint_restore"]["rejects_resumable_last_checkpoint"] is True
    outputs = contract["batch_inference"]["outputs"]
    assert set(outputs) == {
        "pair_probability",
        "translation_a_to_b_rc",
        "translation_valid",
        "correspondence_indices",
        "correspondence_scores",
    }
    assert contract["sealed_synthetic_accessed"] is False
    assert contract["pair_decision"]["status"] == "pending_winner_validation_fit"


def test_common_cluster_threshold_is_bound_to_score_order_and_winner(tmp_path):
    winner = tmp_path / "winner.pt"
    winner.write_bytes(b"winner-checkpoint")
    records = [
        {**_metric_record("p1", True, 0.9, True, 1.0), "cluster_id": "c1"},
        {**_metric_record("n1", False, 0.8, False), "cluster_id": "c1"},
        {**_metric_record("p2", True, 0.4, True, 1.0), "cluster_id": "c2"},
        {**_metric_record("n2", False, 0.1, False), "cluster_id": "c2"},
    ]
    artifact = _fit_frozen_validation_threshold(
        records,
        winner_checkpoint_path=winner,
        model_config=_small_config(),
    )
    reversed_artifact = _fit_frozen_validation_threshold(
        list(reversed(records)),
        winner_checkpoint_path=winner,
        model_config=_small_config(),
    )
    assert artifact["artifact"]["fit_method"] == "maximize_cluster_balanced_f1"
    assert artifact["sample_count"] == 4
    assert artifact["score_order_sha256"] != reversed_artifact["score_order_sha256"]
    path = tmp_path / "validation_threshold.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    threshold = load_frozen_validation_threshold(path, winner)
    assert threshold == artifact["artifact"]["threshold"]
    winner.write_bytes(b"changed")
    with pytest.raises(PairingNetBenchmarkError, match="binding"):
        load_frozen_validation_threshold(path, winner)


def test_official_checkout_commit_cleanliness_and_key_hashes_are_verified():
    workspace = Path(__file__).resolve().parents[3]
    report = audit_official_source(workspace / "tmp" / "benchmarks" / "PairingNet")
    assert report["commit"] == OFFICIAL_COMMIT
    assert report["clean"] is True
    assert "PairingNet Code/utils/loss.py" in report["key_file_sha256"]


def test_batch_inference_emits_frozen_score_translation_correspondence_shapes():
    config = _small_config()
    model = PairingNetRachelAdapted(config).eval()
    mask = np.zeros((1, 1, 32, 32), dtype=np.float32)
    mask[:, :, 5:26, 6:27] = 1.0
    points = np.asarray(
        [[[5, 6], [5, 12], [5, 18], [5, 26], [12, 26], [25, 26], [25, 12], [25, 6]]],
        dtype=np.float32,
    )
    valid = np.ones((1, 8), dtype=np.bool_)
    batch = RachelBatch(
        pair_ids=("pair-1",),
        fragment_a_tokens=("a",),
        fragment_b_tokens=("b",),
        mask_a=mask,
        mask_b=mask.copy(),
        coarse_mask_a=np.zeros((1, 1, 8, 8), dtype=np.float32),
        coarse_mask_b=np.zeros((1, 1, 8, 8), dtype=np.float32),
        points_rc_a=points,
        points_rc_b=points.copy(),
        contour_valid_a=valid,
        contour_valid_b=valid.copy(),
        target_a=np.arange(8, dtype=np.int64)[None, :],
        target_b=np.arange(8, dtype=np.int64)[None, :],
        labels=np.asarray([1.0], dtype=np.float32),
        translation_a_to_b_rc=np.zeros((1, 2), dtype=np.float32),
        translation_a_to_b_xy_cartesian=np.zeros((1, 2), dtype=np.float32),
        translation_valid=np.ones(1, dtype=np.bool_),
    )
    result = infer_pairingnet_batch(model, batch, torch.device("cpu"))
    assert result["method_id"] == METHOD_ID
    assert result["pair_probability"].shape == (1,)
    assert result["translation_a_to_b_rc"].shape == (1, 2)
    assert result["translation_valid"].shape == (1,)
    assert result["correspondence_indices"][0].shape[1:] == (2,)
    assert result["correspondence_scores"][0].ndim == 1


def test_checkpoint_restore_is_strict_and_identity_bound(tmp_path):
    torch.manual_seed(3)
    config = _small_config()
    source = PairingNetRachelAdapted(config)
    adapter_sha256, contract_sha256 = _adapter_identity_hashes()
    checkpoint = {
        "schema_version": "rachel-pairingnet-adapted-benchmark/1.0",
        "checkpoint_kind": WINNER_CHECKPOINT_KIND,
        "method_id": METHOD_ID,
        "official_commit": OFFICIAL_COMMIT,
        "adapter_source_sha256": adapter_sha256,
        "adaptation_contract_sha256": contract_sha256,
        "model_config": config.__dict__,
        "model_state_dict": source.state_dict(),
    }
    path = tmp_path / "winner.pt"
    torch.save(checkpoint, path)
    restored = load_frozen_pairingnet_checkpoint(
        path, torch.device("cpu"), require_formal_config=False
    )
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])
    for field in (
        "official_commit",
        "adapter_source_sha256",
        "adaptation_contract_sha256",
    ):
        tampered = dict(checkpoint)
        tampered[field] = "wrong"
        torch.save(tampered, path)
        with pytest.raises(PairingNetBenchmarkError, match="identity"):
            load_frozen_pairingnet_checkpoint(
                path, torch.device("cpu"), require_formal_config=False
            )


@pytest.mark.parametrize("state_name", ("run", ".run.partial", ".run.failed"))
def test_run_state_rejects_symlink_at_output_partial_or_failed(tmp_path, state_name):
    parent = tmp_path / "runs"
    parent.mkdir()
    target = tmp_path / "attacker-controlled"
    target.mkdir()
    (parent / state_name).symlink_to(target, target_is_directory=True)
    with pytest.raises(PairingNetBenchmarkError, match="symbolic-link"):
        _validated_run_state_paths(parent / "run", None)


def test_run_state_rejects_symlink_parent_and_resolved_resume_alias(tmp_path):
    real_parent = tmp_path / "real-runs"
    real_parent.mkdir()
    alias_parent = tmp_path / "aliased-runs"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(PairingNetBenchmarkError, match="symbolic-link"):
        _validated_run_state_paths(alias_parent / "run", None)

    partial = real_parent / ".run.partial"
    partial.mkdir()
    resume_alias = real_parent / "resume-alias"
    resume_alias.symlink_to(partial, target_is_directory=True)
    # A resolve-then-compare guard would incorrectly accept this alias.
    with pytest.raises(PairingNetBenchmarkError, match="symbolic-link"):
        _validated_run_state_paths(real_parent / "run", resume_alias)


def test_run_state_accepts_only_lexical_exact_partial_or_failed(tmp_path):
    parent = tmp_path / "runs"
    parent.mkdir()
    output = parent / "nested" / ".." / "run"
    partial = parent / ".run.partial"
    partial.mkdir()
    paths = _validated_run_state_paths(output, partial)
    assert paths == (parent / "run", partial, parent / ".run.failed", partial)

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(PairingNetBenchmarkError, match="lexical-exact"):
        _validated_run_state_paths(output, outside)


def test_last_checkpoint_restores_adam_cosine_stopping_and_history(tmp_path):
    config = _small_config()
    run_config = PairingNetRachelRunConfig(
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "output",
        official_source_root=tmp_path / "official",
    )
    model = PairingNetRachelAdapted(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=run_config.max_epochs, eta_min=run_config.eta_min
    )
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    scheduler.step()
    report = summarize_validation_records(
        [
            _metric_record("p1", True, 0.9, True, 1.0),
            _metric_record("p2", True, 0.8, True, 2.0),
            _metric_record("n1", False, 0.2, False),
            _metric_record("n2", False, 0.1, False),
        ]
    )
    key = selection_key(report, 1)
    population = {
        "manifests": {"train_sha256": "train", "val_sha256": "val"}
    }
    official_audit = {"commit": OFFICIAL_COMMIT, "clean": True}
    adapter_sha256, contract_sha256 = _adapter_identity_hashes()
    winner = {
        "schema_version": "rachel-pairingnet-adapted-benchmark/1.0",
        "checkpoint_kind": WINNER_CHECKPOINT_KIND,
        "method_id": METHOD_ID,
        "official_commit": OFFICIAL_COMMIT,
        "adapter_source_sha256": adapter_sha256,
        "adaptation_contract_sha256": contract_sha256,
        "epoch": 1,
        "model_config": config.__dict__,
        "model_state_dict": model.state_dict(),
        "selection_key": list(key),
        "run_config": run_config.portable_dict(),
        "manifest_sha256": population["manifests"],
        "official_source_audit": official_audit,
    }
    history = [{"epoch": 1, "validation": report}]
    payload = _last_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_epoch=1,
        epoch_rows=history,
        best_key=key,
        best_epoch=1,
        best_primary=key[0],
        plateau=0,
        winner_checkpoint=winner,
        run_config=run_config,
        population_audit=population,
        official_source_audit=official_audit,
        source_sha256=adapter_sha256,
        adaptation_contract_sha256=contract_sha256,
        resume_count=0,
    )
    path = tmp_path / "last.pt"
    torch.save(payload, path)
    loaded = _load_last_checkpoint(
        path,
        device=torch.device("cpu"),
        model_config=config,
        run_config=run_config,
        population_audit=population,
        official_source_audit=official_audit,
        source_sha256=adapter_sha256,
        adaptation_contract_sha256=contract_sha256,
    )
    assert loaded["completed_epoch"] == 1
    assert loaded["scheduler_state_dict"]["last_epoch"] == 1
    assert loaded["optimizer_name"] == "Adam"
    assert loaded["partial_epoch_policy"].startswith("discard_and_replay")
    assert loaded["checkpoint_kind"] == LAST_CHECKPOINT_KIND
    with pytest.raises(PairingNetBenchmarkError, match="identity"):
        load_frozen_pairingnet_checkpoint(
            path, torch.device("cpu"), require_formal_config=False
        )

    payload["epoch_history"][0]["epoch"] = 2
    torch.save(payload, path)
    with pytest.raises(PairingNetBenchmarkError, match="history"):
        _load_last_checkpoint(
            path,
            device=torch.device("cpu"),
            model_config=config,
            run_config=run_config,
            population_audit=population,
            official_source_audit=official_audit,
            source_sha256=adapter_sha256,
            adaptation_contract_sha256=contract_sha256,
        )

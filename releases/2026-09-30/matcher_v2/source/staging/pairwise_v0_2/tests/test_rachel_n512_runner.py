import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.training.rachel_n512_runner import (
    RachelN512RunConfig,
    RachelN512RunnerError,
    _arm_selection_metrics,
    _coarse_batch,
    _correspondence_counts,
    _new_model,
    _read_manifest_rows,
    epoch_indices,
    select_winner_epoch,
    stratified_subset_indices,
    validation_indices,
)


def test_epoch_order_is_deterministic_unique_and_shared_by_arms():
    first = epoch_indices(100, 260831, 1, 32)
    second = epoch_indices(100, 260831, 1, 32)
    next_epoch = epoch_indices(100, 260831, 2, 32)
    assert first == second
    assert len(first) == len(set(first)) == 32
    assert first != next_epoch
    assert validation_indices(50, 260831, 20) == validation_indices(50, 260831, 20)


def test_winner_prefers_auroc_then_auprc_then_earlier_epoch():
    rows = [
        {"epoch": 1, "selection_metrics": {"auroc": 0.8, "auprc": 0.7}},
        {"epoch": 2, "selection_metrics": {"auroc": 0.8, "auprc": 0.8}},
        {"epoch": 3, "selection_metrics": {"auroc": 0.8, "auprc": 0.8}},
    ]
    assert select_winner_epoch(rows) == 2
    rows.append({"epoch": 4, "selection_metrics": {"auroc": 0.81, "auprc": 0.1}})
    assert select_winner_epoch(rows) == 4


def test_full_winner_uses_cluster_pair_geometry_and_full_coverage():
    report = {
        "methods": {
            "fused": {
                "row": {"auroc": 0.99, "auprc": 0.99},
                "cluster_balanced": {"auroc": 0.8, "auprc": 0.7},
                "coverage": {"valid_fraction": 1.0},
            }
        },
        "correspondence": {"f1": 0.0, "mutual_top1_f1": 0.6},
        "translation": {"success_at_8px": 0.5},
    }
    metrics = _arm_selection_metrics("full_n512", report)
    assert metrics["auroc"] == 0.8
    assert metrics["auprc"] == 0.7
    assert metrics["correspondence_f1"] == 0.6
    assert metrics["translation_success_at_8px"] == 0.5
    assert metrics["primary_score"] == pytest.approx((0.8 * 0.7 * 0.6 * 0.5) ** 0.25)
    rows = [
        {
            "epoch": 1,
            "selection_metrics": {
                "primary_score": 0.99,
                "coverage": 0.9,
                "auroc": 1.0,
                "auprc": 1.0,
            },
        },
        {
            "epoch": 2,
            "selection_metrics": {
                "primary_score": 0.5,
                "coverage": 1.0,
                "auroc": 0.8,
                "auprc": 0.8,
                "correspondence_f1": 0.6,
                "translation_success_at_8px": 0.6,
            },
        },
    ]
    assert select_winner_epoch(rows) == 2
    with pytest.raises(ValueError, match="full-coverage"):
        select_winner_epoch(rows[:1])


def test_stratified_pilot_subset_is_exactly_balanced_and_fixed():
    labels = np.asarray([True] * 70 + [False] * 30, dtype=np.bool_)
    first = stratified_subset_indices(labels, 17, 40)
    second = stratified_subset_indices(labels, 17, 40)
    assert first == second
    assert len(first) == len(set(first)) == 40
    assert int(labels[list(first)].sum()) == 20
    with pytest.raises(RachelN512RunnerError, match="exact 1:1"):
        stratified_subset_indices(labels, 17, None)


def test_coarse_arm_uses_the_same_torch_resize_as_full_model():
    mask = np.zeros((1, 1, 33, 33), dtype=np.float32)
    mask[:, :, 3:29, 7:24] = 1.0
    batch = SimpleNamespace(
        mask_a=mask,
        mask_b=mask[:, :, ::-1].copy(),
        labels=np.asarray([1.0], dtype=np.float32),
    )
    coarse_a, coarse_b, _ = _coarse_batch(batch, torch.device("cpu"), coarse_size=16)
    torch.testing.assert_close(
        coarse_a,
        F.interpolate(torch.from_numpy(mask), size=(16, 16), mode="nearest"),
    )
    torch.testing.assert_close(
        coarse_b,
        F.interpolate(torch.from_numpy(batch.mask_b), size=(16, 16), mode="nearest"),
    )


def test_correspondence_metrics_exclude_invalid_transport_samples():
    assignment = torch.zeros((2, 2, 2))
    assignment[:, torch.arange(2), torch.arange(2)] = 1.0
    unmatched = torch.zeros((2, 2))
    targets = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    counts = _correspondence_counts(
        assignment,
        unmatched,
        unmatched,
        targets,
        targets,
        torch.tensor([True, False]),
    )
    assert counts[:3] == (2, 2, 2)
    assert counts[4] == 2
    assert counts[6] == 4


def test_train_contract_does_not_require_or_read_test_manifest(tmp_path):
    root = tmp_path / "release"
    pairs = root / "pairs"
    pairs.mkdir(parents=True)
    row = {
        "pair_id": "pair-1",
        "label": True,
        "split": "val",
        "fragment_a": {"split_unit_id": "manuscript-a"},
        "fragment_b": {"split_unit_id": "manuscript-a"},
    }
    (pairs / "val.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    # There is deliberately no pairs/test.jsonl.  Train/validation metadata
    # construction remains complete and sealed-test independent.
    rows = _read_manifest_rows(root, "val")
    assert rows[0].pair_id == "pair-1"
    assert rows[0].cluster_id == "unit:manuscript-a"
    config = RachelN512RunConfig(
        dataset_root=root,
        output_root=tmp_path / "output",
        arms=("full_n512",),
        epochs=1,
        num_workers=0,
        max_train_pairs=16,
        max_val_pairs=8,
    )
    assert "test" not in config.portable_dict()


def test_shared_coarse_initialization_is_identical_between_arms():
    config = RachelN512Config(
        canvas_size=32,
        coarse_size=16,
        contour_cap=8,
        window_sizes_px=(4.0, 8.0),
        patch_size=8,
        feature_dim=16,
        num_heads=4,
        landmark_count=4,
        context_layers=1,
        evidence_dim=8,
        sinkhorn_iterations=5,
        activation_checkpointing=False,
    )
    torch.manual_seed(19)
    coarse = _new_model("coarse_only", config, torch.device("cpu"))
    torch.manual_seed(19)
    full = _new_model("full_n512", config, torch.device("cpu"))
    for name, value in coarse.state_dict().items():
        torch.testing.assert_close(value, full.coarse.state_dict()[name])


@pytest.mark.parametrize("arm", ("coarse_only", "full_n512"))
def test_checkpoint_model_and_optimizer_round_trip_strictly(arm):
    config = RachelN512Config(
        canvas_size=32,
        coarse_size=16,
        contour_cap=8,
        window_sizes_px=(4.0, 8.0),
        patch_size=8,
        feature_dim=16,
        num_heads=4,
        landmark_count=4,
        context_layers=1,
        evidence_dim=8,
        sinkhorn_iterations=5,
        activation_checkpointing=False,
    )
    torch.manual_seed(23)
    source = _new_model(arm, config, torch.device("cpu"))
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-4)
    checkpoint = {
        "model_config": config.__dict__,
        "model_state_dict": source.state_dict(),
        "optimizer_state_dict": source_optimizer.state_dict(),
    }
    restored_config = RachelN512Config(**checkpoint["model_config"])
    restored = _new_model(arm, restored_config, torch.device("cpu"))
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])


def test_runner_config_rejects_unknown_arms(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        RachelN512RunConfig(
            dataset_root=tmp_path,
            output_root=tmp_path / "out",
            arms=("not-a-model",),
        )

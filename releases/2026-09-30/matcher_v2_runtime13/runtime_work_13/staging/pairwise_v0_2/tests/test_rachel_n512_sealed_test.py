import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelBatch
from staging.pairwise_v0_2.training.evaluation import PairwiseThresholdArtifact
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
import staging.pairwise_v0_2.training.rachel_n512_sealed_test as sealed


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _small_config() -> RachelN512Config:
    return RachelN512Config(
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


def _completed_run(tmp_path: Path, arm: str = "coarse_only"):
    run = tmp_path / "run-fixture"
    run.mkdir()
    config = _small_config()
    run_config = {
        "dataset_root": str((tmp_path / "release").resolve()),
        "output_root": str((tmp_path / "train-output").resolve()),
        "arms": [arm],
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 5.0,
        "seed": 17,
        "precision": "fp32",
        "device": "cuda:0",
        "num_workers": 0,
        "max_train_pairs": None,
        "max_val_pairs": None,
        "log_every_steps": 50,
    }
    model = sealed._new_model(arm, config)
    checkpoint = {
        "schema_version": sealed.CHECKPOINT_SCHEMA_VERSION,
        "arm": arm,
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "model_config": sealed.asdict(config),
        "loss_config": sealed.asdict(sealed._canonical_rachel_loss_config()),
        "run_config": run_config,
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    checkpoint_path = run / arm / "epoch-001.pt"
    checkpoint_path.parent.mkdir()
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = _sha(checkpoint_path)
    model_config_sha = hashlib.sha256(
        sealed._canonical_bytes(sealed.asdict(config))
    ).hexdigest()
    aggregation_sha = hashlib.sha256(
        sealed._canonical_bytes(
            {"pair_score": "coarse" if arm == "coarse_only" else "fused"}
        )
    ).hexdigest()
    threshold = PairwiseThresholdArtifact(
        threshold=0.5,
        fit_method="maximize_cluster_balanced_f1",
        source_split="val",
        validation_fingerprint_sha256="1" * 64,
        checkpoint_sha256=checkpoint_sha,
        model_config_sha256=model_config_sha,
        aggregation_config_sha256=aggregation_sha,
        sample_count=4,
        cluster_count=2,
        achieved_cluster_balanced_f1=0.5,
        achieved_cluster_balanced_precision=0.5,
        achieved_cluster_balanced_recall=0.5,
    )
    relative = str(checkpoint_path.relative_to(run))
    arm_result = {
        "arm": arm,
        "epochs": [
            {
                "epoch": 1,
                "checkpoint": relative,
                "checkpoint_sha256": checkpoint_sha,
            }
        ],
        "winner_epoch": 1,
        "winner_checkpoint": relative,
        "winner_checkpoint_sha256": checkpoint_sha,
        "validation_threshold": threshold.to_dict(),
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    receipt = {
        "schema_version": sealed.TRAIN_SCHEMA_VERSION,
        "status": "complete_train_validation_only",
        "fingerprint_sha256": "2" * 64,
        "config": run_config,
        "arm_results": [arm_result],
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    _write_json(run / "run_receipt.json", receipt)
    return run, checkpoint, receipt


def test_completed_winner_is_checkpoint_and_threshold_bound_and_strictly_loaded(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sealed, "_canonical_rachel_model_config", _small_config)
    run, _, _ = _completed_run(tmp_path)
    receipt, receipt_sha, winners = sealed._freeze_completed_winners(run)
    assert receipt["status"] == "complete_train_validation_only"
    assert receipt_sha == _sha(run / "run_receipt.json")
    assert len(winners) == 1
    assert winners[0].arm == "coarse_only"
    assert winners[0].threshold.checkpoint_sha256 == winners[0].checkpoint_sha256
    authority = sealed._canonical_config_authority(winners)
    assert authority["all_winner_configs_exactly_equal"] is True
    assert authority["model_config_sha256"] == winners[0].model_config_sha256
    assert authority["loss_config_sha256"] == winners[0].loss_config_sha256


def test_freeze_rejects_any_noncanonical_model_config_field(tmp_path, monkeypatch):
    monkeypatch.setattr(sealed, "_canonical_rachel_model_config", _small_config)
    run, checkpoint, _ = _completed_run(tmp_path)
    broken = dict(checkpoint)
    broken["model_config"] = dict(checkpoint["model_config"])
    broken["model_config"]["translation_consensus_iterations"] = 3
    monkeypatch.setattr(sealed, "_torch_load_checkpoint", lambda path: broken)
    with pytest.raises(
        sealed.RachelN512SealedTestError,
        match="model config differs from the complete canonical config",
    ):
        sealed._freeze_completed_winners(run)


def test_freeze_rejects_any_noncanonical_loss_config_field(tmp_path, monkeypatch):
    monkeypatch.setattr(sealed, "_canonical_rachel_model_config", _small_config)
    run, checkpoint, _ = _completed_run(tmp_path)
    broken = dict(checkpoint)
    broken["loss_config"] = dict(checkpoint["loss_config"])
    broken["loss_config"]["translation_weight"] = 0.25
    monkeypatch.setattr(sealed, "_torch_load_checkpoint", lambda path: broken)
    with pytest.raises(
        sealed.RachelN512SealedTestError,
        match="loss config differs from the complete canonical config",
    ):
        sealed._freeze_completed_winners(run)


def test_strict_restore_rejects_missing_model_state_key(tmp_path, monkeypatch):
    monkeypatch.setattr(sealed, "_canonical_rachel_model_config", _small_config)
    run, checkpoint, _ = _completed_run(tmp_path)
    broken = dict(checkpoint)
    broken_state = dict(checkpoint["model_state_dict"])
    broken_state.pop(next(iter(broken_state)))
    broken["model_state_dict"] = broken_state
    monkeypatch.setattr(sealed, "_torch_load_checkpoint", lambda path: broken)
    with pytest.raises(sealed.RachelN512SealedTestError, match="strict model restore"):
        sealed._freeze_completed_winners(run)


def test_incomplete_run_fails_before_test_manifest_can_be_opened(tmp_path, monkeypatch):
    run = tmp_path / "run-incomplete"
    run.mkdir()
    _write_json(
        run / "run_receipt.json",
        {
            "schema_version": sealed.TRAIN_SCHEMA_VERSION,
            "status": "running_train_validation_only",
            "test_accessed": False,
            "real_external_test_accessed": False,
        },
    )
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("test manifest was opened")

    monkeypatch.setattr(sealed, "_test_manifest_rows", forbidden)
    with pytest.raises(sealed.RachelN512SealedTestError, match="not complete"):
        sealed.run_sealed_synthetic_test(
            sealed.RachelN512SealedTestConfig(
                run_directory=run,
                output_root=tmp_path / "out",
                device="cpu",
                num_workers=0,
            )
        )
    assert called is False


def _manifest_row(index: int, label: bool):
    unit_a = "unit-{:04d}".format(index // 3)
    unit_b = unit_a if index % 2 else "other-{:04d}".format(index // 5)
    return {
        "pair_id": "pair-{:04d}".format(index),
        "split": "test",
        "label": label,
        "fragment_a": {"split_unit_id": unit_a},
        "fragment_b": {"split_unit_id": unit_b},
    }


def test_test_manifest_gate_requires_exact_3000_and_one_to_one(tmp_path):
    root = tmp_path / "release"
    manifest = root / "pairs" / "test.jsonl"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", encoding="utf-8") as stream:
        for index in range(3000):
            stream.write(json.dumps(_manifest_row(index, index < 1500)) + "\n")
    path, rows = sealed._test_manifest_rows(root.resolve())
    assert path == manifest.resolve()
    assert len(rows) == 3000
    assert sum(row.label for row in rows) == 1500
    assert all(row.cluster_id.startswith(("unit:", "unit-pair:")) for row in rows)
    assert all(
        row.source_unit_ids == tuple(sorted(set(row.source_unit_ids)))
        and 1 <= len(row.source_unit_ids) <= 2
        for row in rows
    )

    with manifest.open("w", encoding="utf-8") as stream:
        for index in range(3000):
            stream.write(json.dumps(_manifest_row(index, index < 1499)) + "\n")
    with pytest.raises(sealed.RachelN512SealedTestError, match="exact 1:1"):
        sealed._test_manifest_rows(root.resolve())


class _FakeFull(torch.nn.Module):
    def forward(self, *inputs):
        del inputs
        probability = torch.tensor([0.9, 0.6, 0.2, 0.8])
        valid = torch.ones(4, dtype=torch.bool)
        assignment = torch.eye(4)[None].repeat(4, 1, 1)
        unmatched = torch.zeros(4, 4)
        translation = torch.tensor([[5.0, 0.0], [6.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        return SimpleNamespace(
            coarse_probability=probability,
            local_probability=probability,
            fused_probability=probability,
            decision_valid=valid,
            assignment=assignment,
            unmatched_a=unmatched,
            unmatched_b=unmatched,
            translation_hat_rc=translation,
            coarse=SimpleNamespace(valid_problem=valid),
        )


def _metric_batch() -> RachelBatch:
    count = 4
    mask = np.zeros((count, 1, 8, 8), dtype=np.float32)
    mask[:, :, 2:6, 2:6] = 1.0
    square = np.asarray(
        [[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
        dtype=np.float32,
    )
    points = np.repeat(square[None], count, axis=0)
    valid = np.ones((count, 4), dtype=np.bool_)
    target = np.asarray(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=np.int64,
    )
    return RachelBatch(
        pair_ids=("p0", "p1", "p2", "p3"),
        fragment_a_tokens=("a",) * count,
        fragment_b_tokens=("b",) * count,
        mask_a=mask,
        mask_b=mask.copy(),
        coarse_mask_a=mask,
        coarse_mask_b=mask.copy(),
        points_rc_a=points,
        points_rc_b=points.copy(),
        contour_valid_a=valid,
        contour_valid_b=valid.copy(),
        target_a=target,
        target_b=target.copy(),
        labels=np.asarray([1, 1, 0, 0], dtype=np.float32),
        translation_a_to_b_rc=np.zeros((count, 2), dtype=np.float32),
        translation_a_to_b_xy_cartesian=np.zeros((count, 2), dtype=np.float32),
        translation_valid=np.asarray([True, True, False, False]),
    )


def test_full_metrics_separate_translation_and_joint_success(monkeypatch):
    monkeypatch.setattr(sealed, "RachelN512Pairwise", _FakeFull)
    threshold = PairwiseThresholdArtifact(
        threshold=0.5,
        fit_method="maximize_cluster_balanced_f1",
        source_split="val",
        validation_fingerprint_sha256="1" * 64,
        checkpoint_sha256="2" * 64,
        model_config_sha256="3" * 64,
        aggregation_config_sha256="4" * 64,
        sample_count=4,
        cluster_count=4,
        achieved_cluster_balanced_f1=0.5,
        achieved_cluster_balanced_precision=0.5,
        achieved_cluster_balanced_recall=0.5,
    )
    winner = sealed._FrozenWinner(
        arm="full_n512",
        epoch=1,
        checkpoint_path=Path("checkpoint.pt"),
        checkpoint_sha256="2" * 64,
        model_config=_small_config(),
        model_config_sha256="3" * 64,
        loss_config=RachelN512LossConfig(
            validate_runtime_targets=False,
            collect_cpu_diagnostics=False,
        ),
        loss_config_sha256="5" * 64,
        threshold=threshold,
        model=_FakeFull(),
    )
    rows = tuple(
        sealed._ManifestRow(
            "p{}".format(index),
            index < 2,
            "c{}".format(index),
            ("unit-{}".format(index),),
        )
        for index in range(4)
    )
    metrics, records = sealed._evaluate_frozen_arm(
        winner,
        [_metric_batch()],
        rows,
        device=torch.device("cpu"),
        precision="fp32",
    )
    assert len(records) == 4
    assert [record["source_unit_ids"] for record in records] == [
        ["unit-{}".format(index)] for index in range(4)
    ]
    assert metrics["translation"]["median_l2_px"] == pytest.approx(5.5)
    assert metrics["translation"]["success_at_2px"] == pytest.approx(0.0)
    assert metrics["translation"]["success_at_8px"] == pytest.approx(1.0)
    assert metrics["joint_success"]["success_at_8px"] == pytest.approx(1.0)
    assert metrics["correspondence"]["strict_dustbin_aware"]["f1"] == pytest.approx(
        2.0 / 3.0
    )
    assert metrics["method_ranking"]["fused"]["thresholded_metrics_reported"] is False
    assert metrics["pairingnet_style_registration"][
        "registration_recall_lt4"
    ] == pytest.approx(1.0)
    assert metrics["assembly_edge"]["by_tolerance"]["at_5"]["true_positive"] == 1
    assert metrics["assembly_edge"]["by_tolerance"]["at_5"]["f1"] == (
        pytest.approx(0.4)
    )
    assert metrics["assembly_edge"]["by_tolerance"]["at_100"]["f1"] == (
        pytest.approx(0.8)
    )
    assert records[1]["geometry"]["pairingnet_style_registration"]["e_rmse"] < 5
    assert records[1]["geometry"]["translation_l2_px"] == pytest.approx(6.0)
    assert (
        records[1]["geometry"]["assembly_edge"]["true_positive_by_tolerance"]["at_5"]
        is False
    )
    assert (
        records[0]["geometry"]["pairingnet_style_registration"]["rotation_error"][
            "status"
        ]
        == "not_applicable_conditioned_upright_orientation"
    )


def test_pairingnet_compatibility_uses_identity_fallback_for_invalid_pose():
    batch = _metric_batch()
    rows = sealed._pairingnet_registration_per_sample(
        batch,
        np.asarray([[5.0, 0.0], [99.0, 99.0], [0.0, 0.0], [0.0, 0.0]]),
        np.asarray([True, False, True, True]),
    )
    assert rows[0]["identity_fallback_used"] is False
    assert rows[1]["prediction_valid"] is False
    assert rows[1]["identity_fallback_used"] is True
    assert rows[1]["e_rmse"] == pytest.approx(0.0)
    assert rows[1]["symmetric_hausdorff_px"] == pytest.approx(0.0)
    assert rows[1]["translation_l2_px"] == pytest.approx(0.0)
    assert rows[1]["registration_recall_lt4_success"] is True
    assert rows[2]["identity_fallback_used"] is None
    assert rows[2]["e_rmse"] is None


def test_pairingnet_contour_area_uses_released_int32_quantization():
    fractional_square = np.asarray(
        [[0.9, 0.9], [0.9, 5.5], [5.5, 5.5], [5.5, 0.9]],
        dtype=np.float64,
    )
    assert sealed._ordered_polygon_area_rc(fractional_square) == pytest.approx(25.0)


def test_matched_pair_rows_share_source_unit_dependency_contract(monkeypatch):
    rows = tuple(
        sealed._ManifestRow(
            "p{}".format(index),
            index == 0,
            "cluster-{}".format(index),
            ("unit-a-{}".format(index), "unit-b-{}".format(index)),
            Path("a-{}.png".format(index)),
            Path("b-{}.png".format(index)),
        )
        for index in range(2)
    )
    predictions = tuple(
        SimpleNamespace(pair_id=row.pair_id, probability=0.8 - 0.6 * index, valid=True)
        for index, row in enumerate(rows)
    )
    monkeypatch.setattr(
        sealed, "score_synthetic_mask_pairs", lambda *args, **kwargs: predictions
    )
    monkeypatch.setattr(sealed, "matched_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(sealed, "_ranking_only", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        sealed,
        "synthetic_pair_records",
        lambda winner, values, manifest_rows: tuple(
            {"pair_id": row.pair_id, "arm": winner.method} for row in manifest_rows
        ),
    )
    winner = SimpleNamespace(
        method="matched_mm_converged",
        threshold=SimpleNamespace(
            threshold=0.5, source_split="val", content_sha256="1" * 64
        ),
    )
    _, records = sealed._evaluate_matched_winner(winner, rows, batch_size=2)
    assert [record["source_unit_ids"] for record in records] == [
        list(row.source_unit_ids) for row in rows
    ]


def test_module_does_not_import_training_or_threshold_fitting():
    source = Path(sealed.__file__).read_text(encoding="utf-8")
    assert "fit_pairwise_threshold" not in source
    assert "torch.optim" not in source
    assert "compute_rachel_n512_loss" not in source


def test_sealed_disclosure_is_truthful_about_human_visible_epoch5_result():
    source = Path(sealed.__file__).read_text(encoding="utf-8")
    assert (
        '"prior_epoch5_synthetic_test_human_visible_before_continuation": True'
        in source
    )
    assert '"claim_no_human_cognitive_influence": False' in source
    assert (
        "prior_epoch5_synthetic_test_used_for_continuation_or_selection" not in source
    )
    assert "prior_epoch5_real_activity_used_for_training_or_selection" not in source


def _combined_receipt(tmp_path, *, converged=True):
    claim = (
        "validation_plateau_under_declared_rule"
        if converged
        else "not_established_before_hard_cap"
    )
    stop = "validation_early_stop" if converged else "hard_cap_reached"
    return {
        "config": {
            "dataset_root": str(tmp_path / "release"),
            "arms": list(sealed.ARMS),
            "precision": "fp32",
            "seed": 17,
            "epochs": 128,
            "batch_size": 16,
            "continuation": {
                "source_last_epoch": 5,
                "scheduler": "CosineAnnealingLR",
                "scheduler_eta_min": 1e-6,
                "min_total_epochs": 20,
                "patience": 12,
                "min_relative_primary_improvement": 0.005,
                "early_stop_reads": ["val"],
                "checkpoint_selection_reads": ["val"],
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
        },
        "population": {
            "train_total": 24_000,
            "val_total": 3_000,
            "train_rows_per_epoch": 24_000,
            "val_rows": 3_000,
        },
        "convergence": {"max_total_epochs": 128, "source_last_epoch": 5},
        "arm_results": [
            {"arm": arm, "stop_reason": stop, "convergence_claim": claim}
            for arm in sealed.ARMS
        ],
    }


def _bind_production_convergence_receipt(
    run_directory, n512_receipt, manifest_hashes=None
):
    """Give a compact receipt fixture the exact production continuation shape."""

    if manifest_hashes is None:
        manifest_hashes = {"train": "1" * 64, "val": "2" * 64}
    source_run = str((run_directory.parent / "source-run").resolve())
    output_root = str((run_directory.parent / "n512-output").resolve())
    source_receipt_sha256 = "a" * 64
    config = n512_receipt["config"]
    config.update({"output_root": output_root, "device": "cuda:0"})
    config["continuation"].update(
        {
            "schema_version": sealed.CONVERGENCE_SCHEMA_VERSION,
            "source_run": source_run,
            "source_receipt_sha256": source_receipt_sha256,
            "scheduler_t_max": 123,
        }
    )
    for result in n512_receipt["arm_results"]:
        result["test_accessed"] = False
        result["real_external_test_accessed"] = False
    policy = {
        "source_run": source_run,
        "output_root": output_root,
        "arms": list(config["arms"]),
        "max_total_epochs": 128,
        "min_total_epochs": 20,
        "patience": 12,
        "min_relative_primary_improvement": 0.005,
        "eta_min": 1e-6,
        "device": "cuda:0",
        "remaining_epochs": 123,
    }
    fingerprint_payload = {
        "schema_version": sealed.CONVERGENCE_SCHEMA_VERSION,
        "source_receipt_sha256": source_receipt_sha256,
        "config": policy,
        "extension_run_config": config,
        "train_manifest_sha256": manifest_hashes["train"],
        "val_manifest_sha256": manifest_hashes["val"],
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    fingerprint = hashlib.sha256(
        sealed._canonical_bytes(fingerprint_payload)
    ).hexdigest()
    n512_receipt.update(
        {
            "schema_version": sealed.TRAIN_SCHEMA_VERSION,
            "status": "complete_train_validation_only",
            "fingerprint_sha256": fingerprint,
            "test_accessed": False,
            "real_external_test_accessed": False,
        }
    )
    convergence_receipt = {
        "schema_version": sealed.CONVERGENCE_SCHEMA_VERSION,
        "status": "complete_train_validation_only",
        "fingerprint_sha256": fingerprint,
        "source_run": source_run,
        "source_receipt_sha256": source_receipt_sha256,
        "policy": policy,
        "config": config,
        "population": n512_receipt["population"],
        "arm_results": n512_receipt["arm_results"],
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    receipt_path = run_directory / "convergence_receipt.json"
    _write_json(receipt_path, convergence_receipt)
    n512_receipt["convergence"] = {
        "schema_version": sealed.CONVERGENCE_SCHEMA_VERSION,
        "receipt": "convergence_receipt.json",
        "receipt_sha256": _sha(receipt_path),
        "source_receipt_sha256": source_receipt_sha256,
        "source_last_epoch": 5,
        "max_total_epochs": 128,
    }
    return convergence_receipt, receipt_path


def _rewrite_bound_convergence_receipt(n512_receipt, receipt_path, value):
    _write_json(receipt_path, value)
    n512_receipt["convergence"]["receipt_sha256"] = _sha(receipt_path)


def test_combined_gate_freezes_every_model_family_before_test_open(
    tmp_path, monkeypatch
):
    n512_run = tmp_path / "n512"
    matched_run = tmp_path / "matched"
    n512_run.mkdir()
    matched_run.mkdir()
    receipt = _combined_receipt(tmp_path)
    winners = tuple(SimpleNamespace(arm=arm) for arm in sealed.ARMS)
    monkeypatch.setattr(
        sealed,
        "_freeze_completed_winners",
        lambda path: (receipt, "1" * 64, winners),
    )
    monkeypatch.setattr(sealed, "_canonical_config_authority", lambda value: {})
    monkeypatch.setattr(
        sealed,
        "freeze_matched_mm_winners",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            sealed.RachelN512SealedTestError("matched freeze failed")
        ),
    )
    opened = False

    def forbidden_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("test manifest opened before all freezes")

    monkeypatch.setattr(sealed, "_test_manifest_rows", forbidden_open)
    with pytest.raises(sealed.RachelN512SealedTestError, match="matched freeze failed"):
        sealed.run_sealed_synthetic_test(
            sealed.RachelN512SealedTestConfig(
                run_directory=n512_run,
                matched_mm_run_directory=matched_run,
                output_root=tmp_path / "out",
                device="cpu",
                num_workers=0,
                compatibility_mode=True,
            )
        )
    assert opened is False


def test_combined_gate_rejects_n512_hard_cap_before_matched_or_test_open(
    tmp_path, monkeypatch
):
    n512_run = tmp_path / "n512"
    matched_run = tmp_path / "matched"
    n512_run.mkdir()
    matched_run.mkdir()
    receipt = _combined_receipt(tmp_path, converged=False)
    winners = tuple(SimpleNamespace(arm=arm) for arm in sealed.ARMS)
    monkeypatch.setattr(
        sealed,
        "_freeze_completed_winners",
        lambda path: (receipt, "1" * 64, winners),
    )
    matched_called = False
    test_called = False

    def matched_forbidden(*args, **kwargs):
        nonlocal matched_called
        matched_called = True

    def test_forbidden(*args, **kwargs):
        nonlocal test_called
        test_called = True

    monkeypatch.setattr(sealed, "freeze_matched_mm_winners", matched_forbidden)
    monkeypatch.setattr(sealed, "_test_manifest_rows", test_forbidden)
    with pytest.raises(sealed.RachelN512SealedTestError, match="convergence plateau"):
        sealed.run_sealed_synthetic_test(
            sealed.RachelN512SealedTestConfig(
                run_directory=n512_run,
                matched_mm_run_directory=matched_run,
                output_root=tmp_path / "out",
                device="cpu",
                num_workers=0,
            )
        )
    assert matched_called is False
    assert test_called is False


def test_benchmark_freeze_failure_precedes_any_test_or_dataset_open(
    tmp_path, monkeypatch
):
    n512_run = tmp_path / "n512"
    n512_run.mkdir()
    pairing = tmp_path / "pairing"
    pairing.mkdir()
    shredding = tmp_path / "shredding" / "train_val_freeze.json"
    shredding.parent.mkdir()
    shredding.write_text("{}\n", encoding="utf-8")
    receipt = _combined_receipt(tmp_path)
    config = SimpleNamespace(canvas_size=800, coarse_size=128, contour_cap=512)
    winners = tuple(
        SimpleNamespace(arm=arm, model_config=config) for arm in sealed.ARMS
    )
    monkeypatch.setattr(
        sealed,
        "_freeze_completed_winners",
        lambda path: (receipt, "1" * 64, winners),
    )
    monkeypatch.setattr(sealed, "_canonical_config_authority", lambda value: {})
    monkeypatch.setattr(
        sealed.benchmark_adapter,
        "freeze_same_data_benchmarks",
        lambda **kwargs: (_ for _ in ()).throw(
            sealed.benchmark_adapter.RachelBenchmarkEvalAdapterError(
                "fixture benchmark freeze failed"
            )
        ),
    )
    opened = []

    def forbidden(*args, **kwargs):
        opened.append(True)
        raise AssertionError("test/dataset opened before benchmark freeze")

    monkeypatch.setattr(sealed, "_resolve_evaluation_dataset_root", forbidden)
    monkeypatch.setattr(sealed, "_test_manifest_rows", forbidden)
    with pytest.raises(
        sealed.RachelN512SealedTestError, match="benchmark freeze failed"
    ):
        sealed.run_sealed_synthetic_test(
            sealed.RachelN512SealedTestConfig(
                run_directory=n512_run,
                pairingnet_run_directory=pairing,
                shreddingnet_freeze_path=shredding,
                output_root=tmp_path / "out",
                device="cpu",
                num_workers=0,
                compatibility_mode=True,
            )
        )
    assert opened == []


def test_formal_exact_six_authorities_are_required_before_any_test_path_open(
    tmp_path, monkeypatch
):
    n512_run = tmp_path / "n512"
    n512_run.mkdir()
    receipt = _combined_receipt(tmp_path)
    winners = tuple(SimpleNamespace(arm=arm) for arm in sealed.ARMS)
    monkeypatch.setattr(
        sealed,
        "_freeze_completed_winners",
        lambda path: (receipt, "1" * 64, winners),
    )
    monkeypatch.setattr(sealed, "_canonical_config_authority", lambda value: {})
    opened = []

    def forbidden(*args, **kwargs):
        opened.append(True)
        raise AssertionError("test path opened before exact-six authority gate")

    monkeypatch.setattr(sealed, "_resolve_evaluation_dataset_root", forbidden)
    monkeypatch.setattr(sealed, "_test_manifest_rows", forbidden)
    with pytest.raises(
        sealed.RachelN512SealedTestError,
        match="requires exact-six frozen authorities",
    ):
        sealed.run_sealed_synthetic_test(
            sealed.RachelN512SealedTestConfig(
                run_directory=n512_run,
                output_root=tmp_path / "out",
                device="cpu",
                num_workers=0,
            )
        )
    assert opened == []


def test_sealed_cli_defaults_to_formal_and_compatibility_is_explicit(tmp_path):
    formal = sealed._parser().parse_args(
        [
            "--run-directory",
            str(tmp_path / "run"),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    compatibility = sealed._parser().parse_args(
        [
            "--run-directory",
            str(tmp_path / "run"),
            "--output-root",
            str(tmp_path / "out"),
            "--compatibility-non-formal",
        ]
    )
    assert formal.compatibility_non_formal is False
    assert compatibility.compatibility_non_formal is True


def test_benchmark_authorities_must_be_supplied_as_one_pair(tmp_path):
    with pytest.raises(ValueError, match="supplied together"):
        sealed.RachelN512SealedTestConfig(
            run_directory=tmp_path / "run",
            output_root=tmp_path / "out",
            pairingnet_run_directory=tmp_path / "pairing",
        )


def test_benchmark_training_alignment_requires_exact_manifest_bytes(tmp_path):
    dataset = tmp_path / "release"
    pairs = dataset / "pairs"
    pairs.mkdir(parents=True)
    for split in ("train", "val"):
        (pairs / (split + ".jsonl")).write_text(split + "-fixture\n", encoding="utf-8")
    hashes = {split: _sha(pairs / (split + ".jsonl")) for split in ("train", "val")}

    class Frozen:
        def __init__(self, manifest):
            self.training_manifest_sha256 = manifest

        def provenance(self):
            return {"frozen": True}

    frozen = {
        method: Frozen(hashes.copy())
        for method in sealed.benchmark_adapter.BENCHMARK_METHODS
    }
    evidence = sealed._benchmark_training_manifest_evidence(dataset, frozen)
    assert evidence["manifest_content_sha256"] == hashes
    frozen[sealed.benchmark_adapter.SHREDDINGNET_METHOD_KEY].training_manifest_sha256[
        "val"
    ] = "0" * 64
    with pytest.raises(sealed.RachelN512SealedTestError, match="differs"):
        sealed._benchmark_training_manifest_evidence(dataset, frozen)


def test_sealed_gate_requires_plateau_without_matched_control(tmp_path, monkeypatch):
    run = tmp_path / "n512"
    run.mkdir()
    receipt = _combined_receipt(tmp_path, converged=False)
    winners = tuple(SimpleNamespace(arm=arm) for arm in sealed.ARMS)
    monkeypatch.setattr(
        sealed,
        "_freeze_completed_winners",
        lambda path: (receipt, "1" * 64, winners),
    )
    opened = False

    def forbidden_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("test manifest opened")

    monkeypatch.setattr(sealed, "_test_manifest_rows", forbidden_open)
    with pytest.raises(sealed.RachelN512SealedTestError, match="convergence plateau"):
        sealed.run_sealed_synthetic_test(
            sealed.RachelN512SealedTestConfig(
                run_directory=run,
                output_root=tmp_path / "out",
                device="cpu",
                num_workers=0,
            )
        )
    assert opened is False


def test_formal_gate_rejects_changed_continuation_policy(tmp_path):
    receipt = _combined_receipt(tmp_path)
    receipt["config"]["continuation"]["patience"] = 11
    winners = tuple(SimpleNamespace(arm=arm) for arm in sealed.ARMS)
    with pytest.raises(
        sealed.RachelN512SealedTestError,
        match="continuation policy differs",
    ):
        sealed._require_formal_convergence(receipt, winners)


def _matched_receipt(tmp_path, dataset_root):
    return {
        "config": {
            "dataset_root": str(dataset_root),
            "seed": 17,
            "batch_size": 16,
            "max_total_epochs": 128,
            "min_total_epochs": 20,
            "patience": 12,
            "min_relative_auroc_improvement": 0.005,
            "source_splits_opened": ["train", "val"],
            "test_accessed": False,
            "real_external_test_accessed": False,
        },
        "population": {
            "train_total": 24_000,
            "val_total": 3_000,
            "train_positive": 12_000,
            "val_positive": 1_500,
        },
    }


def test_matched_alignment_uses_canonical_root_and_exact_exposure(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    alias = tmp_path / "release-alias"
    alias.symlink_to(release, target_is_directory=True)
    n512 = _combined_receipt(tmp_path)
    n512["config"]["dataset_root"] = str(alias)
    matched = _matched_receipt(tmp_path, release)
    assert (
        sealed._require_matched_training_alignment(n512, matched) == release.resolve()
    )

    matched["config"]["min_total_epochs"] = 19
    with pytest.raises(
        sealed.RachelN512SealedTestError, match="formal validation-only"
    ):
        sealed._require_matched_training_alignment(n512, matched)


def test_matched_alignment_hash_evidence_binds_manifests_and_validation_order(
    tmp_path,
):
    release = tmp_path / "release"
    pair_root = release / "pairs"
    pair_root.mkdir(parents=True)
    pair_ids = ["val-{:04d}".format(index) for index in range(3_000)]
    train_pair_ids = ["train-{:05d}".format(index) for index in range(24_000)]
    for split, values in (("train", train_pair_ids), ("val", pair_ids)):
        (pair_root / (split + ".jsonl")).write_text(
            "".join(
                json.dumps({"split": split, "pair_id": pair_id}, sort_keys=True) + "\n"
                for pair_id in values
            ),
            encoding="utf-8",
        )
    score_artifact = {
        "pair_ids": pair_ids,
        "labels": [index % 2 == 0 for index in range(3_000)],
        "clusters": ["c-{:04d}".format(index) for index in range(3_000)],
        "probability": [0.5] * 3_000,
        "valid": [True] * 3_000,
    }
    order_sha = hashlib.sha256(sealed._canonical_bytes(pair_ids)).hexdigest()
    n512_run = tmp_path / "n512"
    matched_run = tmp_path / "matched"
    n512_run.mkdir()
    matched_run.mkdir()
    for arm in sealed.ARMS:
        _write_json(n512_run / (arm + ".json"), score_artifact)
    _write_json(matched_run / "converged.json", score_artifact)
    _write_json(matched_run / "epoch5.json", score_artifact)

    n512 = _combined_receipt(tmp_path)
    for result in n512["arm_results"]:
        artifact = result["arm"] + ".json"
        result.update(
            {
                "winner_validation_scores": artifact,
                "winner_validation_scores_sha256": _sha(n512_run / artifact),
                "validation_threshold": {
                    "validation_fingerprint_sha256": order_sha,
                    "checkpoint_sha256": "b" * 64,
                },
            }
        )
    manifest_hashes = {
        split: _sha(pair_root / (split + ".jsonl")) for split in ("train", "val")
    }
    _, convergence_path = _bind_production_convergence_receipt(
        n512_run, n512, manifest_hashes
    )
    assert "policy" not in n512

    matched = _matched_receipt(tmp_path, release)
    matched["schema_version"] = "rachel-matched-historical-mm-train/1.0"
    matched["winner_epoch"] = 23
    matched["validation_threshold"] = {
        "validation_fingerprint_sha256": order_sha,
        "checkpoint_sha256": "c" * 64,
    }
    matched["epochs"] = [
        {
            "epoch": 23,
            "validation_scores": "converged.json",
            "validation_scores_sha256": _sha(matched_run / "converged.json"),
        }
    ]
    matched["same_exposure_epoch5"] = {
        "validation_scores": "epoch5.json",
        "validation_scores_sha256": _sha(matched_run / "epoch5.json"),
        "validation_threshold": {
            "validation_fingerprint_sha256": order_sha,
            "checkpoint_sha256": "d" * 64,
        },
    }
    matched_payload = {
        "schema_version": matched["schema_version"],
        "config": matched["config"],
        "train_manifest_sha256": manifest_hashes["train"],
        "val_manifest_sha256": manifest_hashes["val"],
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    matched["fingerprint_sha256"] = hashlib.sha256(
        sealed._canonical_bytes(matched_payload)
    ).hexdigest()

    evidence = sealed._matched_training_hash_evidence(
        n512,
        matched,
        n512_run_directory=n512_run.resolve(),
        matched_run_directory=matched_run.resolve(),
    )
    assert evidence["train_manifest"]["content_sha256"] == manifest_hashes["train"]
    assert (
        evidence["validation_pair_order"][
            "exact_order_equal_across_four_frozen_thresholds"
        ]
        is True
    )
    assert (
        evidence["limitations"][
            "exact_per_epoch_training_pair_presentation_order_claimed_equal"
        ]
        is False
    )

    matched["validation_threshold"]["validation_fingerprint_sha256"] = "0" * 64
    with pytest.raises(
        sealed.RachelN512SealedTestError,
        match="not bound to its validation pair order",
    ):
        sealed._matched_training_hash_evidence(
            n512,
            matched,
            n512_run_directory=n512_run.resolve(),
            matched_run_directory=matched_run.resolve(),
        )

    matched["validation_threshold"]["validation_fingerprint_sha256"] = order_sha
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    convergence["source_receipt_sha256"] = "e" * 64
    convergence["config"]["continuation"]["source_receipt_sha256"] = "e" * 64
    n512["config"]["continuation"]["source_receipt_sha256"] = "e" * 64
    n512["convergence"]["source_receipt_sha256"] = "e" * 64
    _rewrite_bound_convergence_receipt(n512, convergence_path, convergence)
    with pytest.raises(
        sealed.RachelN512SealedTestError,
        match="fingerprint is not bound",
    ):
        sealed._matched_training_hash_evidence(
            n512,
            matched,
            n512_run_directory=n512_run.resolve(),
            matched_run_directory=matched_run.resolve(),
        )


def _production_convergence_fixture(tmp_path, run_name="run-convergence-fixture"):
    run = tmp_path / run_name
    run.mkdir()
    receipt = _combined_receipt(tmp_path)
    convergence, convergence_path = _bind_production_convergence_receipt(run, receipt)
    return run, receipt, convergence, convergence_path


def test_bound_convergence_loader_accepts_production_shape_without_top_policy(
    tmp_path,
):
    run, receipt, convergence, _ = _production_convergence_fixture(tmp_path)
    assert "policy" not in receipt
    assert (
        sealed._load_bound_n512_convergence_receipt(receipt, run.resolve())
        == convergence
    )


def test_bound_convergence_loader_never_falls_back_to_fake_top_policy(tmp_path):
    run, receipt, convergence, convergence_path = _production_convergence_fixture(
        tmp_path
    )
    receipt["policy"] = convergence["policy"]
    convergence_path.unlink()
    with pytest.raises(
        sealed.RachelN512SealedTestError, match="stable confined regular file"
    ):
        sealed._load_bound_n512_convergence_receipt(receipt, run.resolve())


@pytest.mark.parametrize(
    "relative",
    [
        "../outside.json",
        "/outside.json",
        "nested//convergence_receipt.json",
        "convergence_receipt.json/",
    ],
)
def test_bound_convergence_loader_rejects_noncanonical_or_external_path(
    tmp_path, relative
):
    run, receipt, _, _ = _production_convergence_fixture(tmp_path)
    receipt["convergence"]["receipt"] = relative
    with pytest.raises(
        sealed.RachelN512SealedTestError, match="canonical relative JSON path"
    ):
        sealed._load_bound_n512_convergence_receipt(receipt, run.resolve())


def test_bound_convergence_loader_rejects_wrong_hash_and_pointer_schema(tmp_path):
    run, receipt, _, _ = _production_convergence_fixture(tmp_path)
    receipt["convergence"]["receipt_sha256"] = "0" * 64
    with pytest.raises(sealed.RachelN512SealedTestError, match="SHA-256 differs"):
        sealed._load_bound_n512_convergence_receipt(receipt, run.resolve())

    receipt["convergence"]["schema_version"] = "wrong-schema"
    with pytest.raises(
        sealed.RachelN512SealedTestError, match="pointer schema differs"
    ):
        sealed._load_bound_n512_convergence_receipt(receipt, run.resolve())


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "status",
        "fingerprint",
        "config",
        "population",
        "arm_results",
        "source_receipt",
        "policy",
        "receipt_test_flag",
        "receipt_real_flag",
        "arm_test_flag",
        "continuation_test_flag",
        "policy_scalar_type",
        "config_scalar_type",
        "population_scalar_type",
        "unexpected_field",
    ],
)
def test_bound_convergence_loader_rejects_receipt_cross_binding_mutations(
    tmp_path, case
):
    run, receipt, convergence, convergence_path = _production_convergence_fixture(
        tmp_path
    )
    mutated = json.loads(json.dumps(convergence))
    if case == "schema":
        mutated["schema_version"] = "wrong-schema"
    elif case == "status":
        mutated["status"] = "running_train_validation_only"
    elif case == "fingerprint":
        mutated["fingerprint_sha256"] = "f" * 64
    elif case == "config":
        mutated["config"]["seed"] = 999
    elif case == "population":
        mutated["population"]["train_total"] = 23_999
    elif case == "arm_results":
        mutated["arm_results"][0]["arm"] = "unknown"
    elif case == "source_receipt":
        mutated["source_receipt_sha256"] = "e" * 64
    elif case == "policy":
        mutated["policy"]["patience"] = 11
    elif case == "receipt_test_flag":
        mutated["test_accessed"] = True
    elif case == "receipt_real_flag":
        mutated["real_external_test_accessed"] = True
    elif case == "arm_test_flag":
        mutated["arm_results"][0]["test_accessed"] = True
        receipt["arm_results"][0]["test_accessed"] = True
    elif case == "continuation_test_flag":
        mutated["config"]["continuation"]["test_accessed"] = True
        receipt["config"]["continuation"]["test_accessed"] = True
    elif case == "policy_scalar_type":
        mutated["policy"]["max_total_epochs"] = 128.0
    elif case == "config_scalar_type":
        mutated["config"]["batch_size"] = 16.0
        receipt["config"]["batch_size"] = 16.0
    elif case == "population_scalar_type":
        mutated["population"]["train_total"] = 24_000.0
        receipt["population"]["train_total"] = 24_000.0
    elif case == "unexpected_field":
        mutated["unbound_extension"] = True
    else:  # pragma: no cover - parameter inventory is closed above
        raise AssertionError(case)
    _rewrite_bound_convergence_receipt(receipt, convergence_path, mutated)
    with pytest.raises(sealed.RachelN512SealedTestError):
        sealed._load_bound_n512_convergence_receipt(receipt, run.resolve())


@pytest.mark.parametrize("payload_case", ["duplicate", "nan", "utf8"])
def test_bound_convergence_loader_rejects_ambiguous_json(tmp_path, payload_case):
    run, receipt, convergence, convergence_path = _production_convergence_fixture(
        tmp_path
    )
    if payload_case == "duplicate":
        payload = b'{"schema_version":"first","schema_version":"second"}'
    elif payload_case == "nan":
        payload = sealed._canonical_bytes(convergence)
        needle = b'"eta_min":1e-06'
        assert needle in payload
        payload = payload.replace(needle, b'"eta_min":NaN', 1)
    else:
        payload = b'{"schema_version":"' + bytes([0xFF]) + b'"}'
    convergence_path.write_bytes(payload)
    receipt["convergence"]["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(
        sealed.RachelN512SealedTestError, match="strict finite UTF-8 JSON"
    ):
        sealed._load_bound_n512_convergence_receipt(receipt, run.resolve())


def test_bound_convergence_loader_rejects_symlink_components_and_hardlinks(
    tmp_path,
):
    run, receipt, _, convergence_path = _production_convergence_fixture(tmp_path)
    real = run / "real"
    real.mkdir()
    relocated = real / convergence_path.name
    convergence_path.replace(relocated)
    (run / "linked").symlink_to(real, target_is_directory=True)
    receipt["convergence"]["receipt"] = "linked/convergence_receipt.json"
    with pytest.raises(sealed.RachelN512SealedTestError, match="symlink"):
        sealed._load_bound_n512_convergence_receipt(receipt, run.resolve())

    (run / "linked").unlink()
    relocated.replace(convergence_path)
    receipt["convergence"]["receipt"] = "convergence_receipt.json"
    hardlink = tmp_path / "convergence-hardlink.json"
    os.link(convergence_path, hardlink)
    with pytest.raises(
        sealed.RachelN512SealedTestError, match="single-link regular file"
    ):
        sealed._load_bound_n512_convergence_receipt(receipt, run.resolve())


def test_bound_convergence_loader_rejects_in_read_toctou_tamper(tmp_path, monkeypatch):
    run, receipt, _, convergence_path = _production_convergence_fixture(tmp_path)
    original = sealed._read_descriptor_bytes

    def read_then_tamper(descriptor):
        payload = original(descriptor)
        convergence_path.write_text('{"tampered":true}\n', encoding="utf-8")
        return payload

    monkeypatch.setattr(sealed, "_read_descriptor_bytes", read_then_tamper)
    with pytest.raises(
        sealed.RachelN512SealedTestError, match="changed while its bytes were frozen"
    ):
        sealed._load_bound_n512_convergence_receipt(receipt, run.resolve())


def test_bound_convergence_loader_rejects_partial_run(tmp_path):
    run, receipt, _, _ = _production_convergence_fixture(
        tmp_path, run_name=".partial-convergence-fixture"
    )
    with pytest.raises(sealed.RachelN512SealedTestError, match="finalized run"):
        sealed._load_bound_n512_convergence_receipt(receipt, run.resolve())


def test_no_replace_publish_rejects_existing_file_and_directory(tmp_path):
    target = tmp_path / "one.json"
    sealed._atomic_json(target, {"first": True})
    with pytest.raises(sealed.RachelN512SealedTestError, match="overwrite"):
        sealed._atomic_json(target, {"second": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"first": True}

    staged = tmp_path / "staged"
    staged.mkdir()
    sealed._atomic_json(staged / "test_receipt.json", {"complete": True})
    final = tmp_path / "final"
    final.mkdir()
    with pytest.raises(sealed.RachelN512SealedTestError, match="already exists"):
        sealed._publish_directory_no_replace(
            staged, final, completion_receipt="test_receipt.json"
        )


def test_dataset_override_must_canonically_match_receipt(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(release, target_is_directory=True)
    receipt = _combined_receipt(tmp_path)
    receipt["config"]["dataset_root"] = str(release)
    assert sealed._resolve_evaluation_dataset_root(receipt, alias) == release.resolve()

    different = tmp_path / "different-release"
    different.mkdir()
    with pytest.raises(sealed.RachelN512SealedTestError, match="canonically equal"):
        sealed._resolve_evaluation_dataset_root(receipt, different)

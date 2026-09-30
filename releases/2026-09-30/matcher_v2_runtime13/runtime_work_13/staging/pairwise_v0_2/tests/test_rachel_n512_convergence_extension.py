import math
from dataclasses import asdict
import hashlib
import json

import pytest
import torch

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from staging.pairwise_v0_2.training.rachel_n512_convergence_extension import (
    RachelN512ConvergenceConfig,
    RachelN512ConvergenceError,
    ValidationPlateauState,
    _extension_run_config,
    _repack_source_history,
    _strict_restore_epoch5,
    _validated_source_run,
    build_remaining_cosine_scheduler,
    update_validation_plateau,
    validation_early_stop_due,
)
from staging.pairwise_v0_2.training.evaluation import PairwiseThresholdArtifact
from staging.pairwise_v0_2.training.rachel_n512_runner import _new_model
import staging.pairwise_v0_2.training.rachel_n512_sealed_test as sealed


def _config(tmp_path, **overrides):
    values = {
        "source_run": tmp_path / "source",
        "output_root": tmp_path / "output",
    }
    values.update(overrides)
    return RachelN512ConvergenceConfig(**values)


def test_default_convergence_contract_extends_epoch_five_to_128(tmp_path):
    config = _config(tmp_path)

    assert config.max_total_epochs == 128
    assert config.min_total_epochs == 20
    assert config.patience == 12
    assert config.min_relative_primary_improvement == pytest.approx(0.005)
    assert config.eta_min == pytest.approx(1e-6)
    assert config.remaining_epochs == 123
    assert config.portable_dict()["remaining_epochs"] == 123


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"arms": ()}, "arms must be a non-empty unique sequence"),
        (
            {"arms": ("full_n512", "full_n512")},
            "arms must be a non-empty unique sequence",
        ),
        ({"arms": ("unknown",)}, "unsupported Rachel convergence arm"),
        ({"max_total_epochs": 5}, "max_total_epochs must extend beyond epoch 5"),
        ({"max_total_epochs": True}, "max_total_epochs must be a positive integer"),
        ({"min_total_epochs": 5}, "min_total_epochs must be in"),
        (
            {"max_total_epochs": 12, "min_total_epochs": 13},
            "min_total_epochs must be in",
        ),
        ({"patience": 0}, "patience must be a positive integer"),
        (
            {"min_relative_primary_improvement": -0.001},
            "min_relative_primary_improvement must be non-negative",
        ),
        (
            {"min_relative_primary_improvement": math.inf},
            "min_relative_primary_improvement must be non-negative",
        ),
        ({"eta_min": 0.0}, "eta_min must be finite and positive"),
        ({"eta_min": math.nan}, "eta_min must be finite and positive"),
        ({"device": "cpu"}, "formal convergence training requires a CUDA device"),
    ],
)
def test_invalid_convergence_config_is_rejected(tmp_path, overrides, error):
    with pytest.raises(ValueError, match=error):
        _config(tmp_path, **overrides)


def test_validation_plateau_uses_half_percent_relative_threshold_and_resets():
    state = ValidationPlateauState(anchor_primary=0.2, anchor_epoch=5)

    state, qualified = update_validation_plateau(
        state,
        epoch=6,
        primary_score=0.20099,
        full_coverage=True,
        relative_improvement=0.005,
    )
    assert not qualified
    assert state == ValidationPlateauState(0.2, 5, 1)

    # Even a larger score cannot move the validation anchor without full coverage.
    state, qualified = update_validation_plateau(
        state,
        epoch=7,
        primary_score=0.25,
        full_coverage=False,
        relative_improvement=0.005,
    )
    assert not qualified
    assert state == ValidationPlateauState(0.2, 5, 2)

    # The exact 0.5% boundary is qualifying and resets the patience counter.
    state, qualified = update_validation_plateau(
        state,
        epoch=8,
        primary_score=0.201,
        full_coverage=True,
        relative_improvement=0.005,
    )
    assert qualified
    assert state == ValidationPlateauState(0.201, 8, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epoch": 5, "primary_score": 0.3, "full_coverage": True},
        {"epoch": 6, "primary_score": math.nan, "full_coverage": True},
    ],
)
def test_validation_plateau_rejects_invalid_updates(kwargs):
    with pytest.raises(ValueError, match="invalid validation plateau update"):
        update_validation_plateau(
            ValidationPlateauState(anchor_primary=0.2, anchor_epoch=5),
            relative_improvement=0.005,
            **kwargs,
        )


def test_remaining_cosine_scheduler_reaches_eta_min_after_123_steps(tmp_path):
    config = _config(tmp_path)
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = build_remaining_cosine_scheduler(optimizer, config)

    assert scheduler.T_max == 123
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)

    for _ in range(config.remaining_epochs):
        optimizer.step()
        scheduler.step()

    assert scheduler.last_epoch == 123
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6, abs=1e-12)


def test_validation_stop_requires_both_minimum_epoch_and_patience():
    state = ValidationPlateauState(0.2, 5, 12)
    assert not validation_early_stop_due(
        state, current_epoch=19, min_total_epochs=20, patience=12
    )
    assert validation_early_stop_due(
        state, current_epoch=20, min_total_epochs=20, patience=12
    )
    assert not validation_early_stop_due(
        ValidationPlateauState(0.2, 8, 11),
        current_epoch=20,
        min_total_epochs=20,
        patience=12,
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fixture(tmp_path):
    source = tmp_path / "run-source"
    source.mkdir()
    release = tmp_path / "release-with-no-test"
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
        validate_runtime_inputs=False,
    )
    run_config = {
        "dataset_root": str(release),
        "output_root": str(tmp_path / "old-output"),
        "arms": ["coarse_only", "full_n512"],
        "epochs": 5,
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
    arm_results = []
    for arm in run_config["arms"]:
        torch.manual_seed(23)
        model = _new_model(arm, config, torch.device("cpu"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        rows = []
        for epoch in range(1, 6):
            checkpoint = {
                "schema_version": "rachel-n512-checkpoint/1.0",
                "arm": arm,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": asdict(config),
                "loss_config": asdict(
                    RachelN512LossConfig(
                        validate_runtime_targets=False,
                        collect_cpu_diagnostics=False,
                    )
                ),
                "run_config": run_config,
                "test_accessed": False,
                "real_external_test_accessed": False,
            }
            path = source / arm / ("epoch-{:03d}.pt".format(epoch))
            path.parent.mkdir(exist_ok=True)
            torch.save(checkpoint, path)
            rows.append(
                {
                    "epoch": epoch,
                    "checkpoint": str(path.relative_to(source)),
                    "checkpoint_sha256": _sha(path),
                    "selection_metrics": {
                        "primary_score": epoch / 10.0,
                        "coverage": 1.0,
                        "auroc": epoch / 10.0,
                        "auprc": epoch / 10.0,
                    },
                }
            )
        arm_results.append(
            {
                "arm": arm,
                "epochs": rows,
                "test_accessed": False,
                "real_external_test_accessed": False,
            }
        )
    receipt = {
        "schema_version": "rachel-n512-train-run/1.0",
        "status": "complete_train_validation_only",
        "config": run_config,
        "arm_results": arm_results,
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    (source / "run_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return source


def test_source_list_receipt_and_epoch5_model_adamw_restore_without_test(tmp_path):
    source = _source_fixture(tmp_path)
    config = RachelN512ConvergenceConfig(
        source_run=source,
        output_root=tmp_path / "extension",
    )
    validated = _validated_source_run(config)
    assert tuple(arm.arm for arm in validated.arms) == (
        "coarse_only",
        "full_n512",
    )
    assert not (tmp_path / "release-with-no-test" / "pairs" / "test.jsonl").exists()
    for arm in validated.arms:
        model, optimizer, model_config, loss_config = _strict_restore_epoch5(
            arm, validated.run_config, torch.device("cpu")
        )
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        assert model_config.canvas_size == 32
        assert isinstance(loss_config, RachelN512LossConfig)
        assert len(model.state_dict()) > 0


def test_source_checkpoint_hash_mismatch_is_rejected(tmp_path):
    source = _source_fixture(tmp_path)
    receipt_path = source / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["arm_results"][0]["epochs"][4]["checkpoint_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    with pytest.raises(RachelN512ConvergenceError, match="SHA-256 mismatch"):
        _validated_source_run(
            RachelN512ConvergenceConfig(
                source_run=source,
                output_root=tmp_path / "extension",
            )
        )


def test_normalized_history_winner_is_accepted_by_sealed_evaluator(
    tmp_path, monkeypatch
):
    source_path = _source_fixture(tmp_path)
    (tmp_path / "release-with-no-test").mkdir()
    policy = RachelN512ConvergenceConfig(
        source_run=source_path,
        output_root=tmp_path / "extension-output",
    )
    source = _validated_source_run(policy)
    fixture_model_config = RachelN512Config(
        **dict(source.arms[0].epoch5_payload["model_config"])
    )
    fixture_loss_config = RachelN512LossConfig(
        **dict(source.arms[0].epoch5_payload["loss_config"])
    )
    monkeypatch.setattr(
        sealed, "_canonical_rachel_model_config", lambda: fixture_model_config
    )
    monkeypatch.setattr(
        sealed, "_canonical_rachel_loss_config", lambda: fixture_loss_config
    )
    run = tmp_path / "normalized-run"
    run.mkdir()
    run_config = _extension_run_config(
        source,
        policy,
        dataset_root=(tmp_path / "release-with-no-test").resolve(),
        output_root=(tmp_path / "extension-output").resolve(),
    )
    results = []
    for source_arm in source.arms:
        arm_directory = run / source_arm.arm
        arm_directory.mkdir()
        rows = _repack_source_history(
            source_arm,
            source,
            arm_directory=arm_directory,
            partial_directory=run,
            extension_run_config=run_config,
        )
        winner = rows[-1]
        model_config = source_arm.epoch5_payload["model_config"]
        model_sha = hashlib.sha256(
            json.dumps(
                model_config, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        aggregation_sha = hashlib.sha256(
            json.dumps(
                {
                    "pair_score": (
                        "coarse" if source_arm.arm == "coarse_only" else "fused"
                    )
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        threshold = PairwiseThresholdArtifact(
            threshold=0.5,
            fit_method="maximize_cluster_balanced_f1",
            source_split="val",
            validation_fingerprint_sha256="1" * 64,
            checkpoint_sha256=winner["checkpoint_sha256"],
            model_config_sha256=model_sha,
            aggregation_config_sha256=aggregation_sha,
            sample_count=4,
            cluster_count=2,
            achieved_cluster_balanced_f1=0.5,
            achieved_cluster_balanced_precision=0.5,
            achieved_cluster_balanced_recall=0.5,
        )
        results.append(
            {
                "arm": source_arm.arm,
                "epochs": rows,
                "winner_epoch": 5,
                "winner_checkpoint": winner["checkpoint"],
                "winner_checkpoint_sha256": winner["checkpoint_sha256"],
                "validation_threshold": threshold.to_dict(),
                "test_accessed": False,
                "real_external_test_accessed": False,
            }
        )
    (run / "run_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "rachel-n512-train-run/1.0",
                "status": "complete_train_validation_only",
                "fingerprint_sha256": "2" * 64,
                "config": run_config,
                "arm_results": results,
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt, _, winners = sealed._freeze_completed_winners(run)
    assert receipt["config"] == run_config
    assert tuple(winner.arm for winner in winners) == (
        "coarse_only",
        "full_n512",
    )

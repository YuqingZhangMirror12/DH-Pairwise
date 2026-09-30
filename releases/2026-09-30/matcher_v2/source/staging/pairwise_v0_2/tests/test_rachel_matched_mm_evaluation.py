import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from staging.pairwise_v0_2.baselines import rachel_matched_mm_evaluation as evaluation
from staging.pairwise_v0_2.baselines import rachel_matched_mm_siamese as training
from staging.pairwise_v0_2.training.evaluation import PairwiseThresholdArtifact


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _threshold(checkpoint_sha: str, model_config: object, value: float):
    return PairwiseThresholdArtifact(
        threshold=value,
        fit_method="maximize_cluster_balanced_f1",
        source_split="val",
        validation_fingerprint_sha256="1" * 64,
        checkpoint_sha256=checkpoint_sha,
        model_config_sha256=_canonical_sha(model_config),
        aggregation_config_sha256=_canonical_sha(
            {"pair_score": evaluation.MATCHED_SCORE}
        ),
        sample_count=3_000,
        cluster_count=100,
        achieved_cluster_balanced_f1=0.7,
        achieved_cluster_balanced_precision=0.7,
        achieved_cluster_balanced_recall=0.7,
    )


def _completed_matched_run(tmp_path: Path) -> Path:
    run = tmp_path / "run-matched"
    run.mkdir(parents=True)
    run_config = {
        "dataset_root": str((tmp_path / "release").resolve()),
        "output_root": str((tmp_path / "output").resolve()),
        "max_total_epochs": 128,
        "min_total_epochs": 20,
        "patience": 12,
        "min_relative_auroc_improvement": 0.005,
        "batch_size": 16,
        "eval_batch_size": 128,
        "seed": 260831,
        "device": "cuda:0",
        "learning_rate": 1.0,
        "scheduler_gamma": 0.7,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "model_recipe": training.MODEL_RECIPE,
        "source_splits_opened": ["train", "val"],
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    model_config = {
        "architecture": "HistoricalMMSiamese",
        "recipe": training.MODEL_RECIPE,
        "input": "bool_mask_PIL_bilinear_64x64_single_channel",
        "ordered_pair": True,
        "focal_formula": "alpha_times_one_minus_p_observed_pow_gamma_times_bce",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "optimizer": {
            "name": "Adadelta",
            "learning_rate": 1.0,
            "rho": 0.9,
            "eps": 1e-6,
            "weight_decay": 0.0,
        },
        "scheduler": {"name": "StepLR", "step_size": 1, "gamma": 0.7},
    }
    torch.manual_seed(9)
    model5 = training.HistoricalMMSiamese()
    torch.manual_seed(10)
    model20 = training.HistoricalMMSiamese()
    epoch_rows = []
    checkpoints = {}
    for epoch, model in ((5, model5), (20, model20)):
        checkpoint_path = run / "epoch-{:03d}.pt".format(epoch)
        torch.save(
            {
                "schema_version": training.CHECKPOINT_SCHEMA_VERSION,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "scheduler_state_dict": {},
                "model_config": model_config,
                "run_config": run_config,
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
            checkpoint_path,
        )
        checkpoint_sha = _file_sha(checkpoint_path)
        score_path = run / "val-scores-{:03d}.json".format(epoch)
        _write_json(score_path, {"probability": [0.2, 0.8]})
        row = {
            "epoch": epoch,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "validation_scores": score_path.name,
            "validation_scores_sha256": _file_sha(score_path),
            "selection_metrics": {
                "primary_score": 0.7 + epoch / 100.0,
                "auroc": 0.7 + epoch / 100.0,
                "auprc": 0.6 + epoch / 100.0,
                "coverage": 1.0,
                "weighting": "equal_lineage_or_lineage_pair_cluster",
            },
        }
        epoch_rows.append(row)
        checkpoints[epoch] = (checkpoint_path, checkpoint_sha, row)
    raw_state_path = run / "winner_state_dict.pt"
    torch.save(model20.state_dict(), raw_state_path)
    epoch5_path, epoch5_sha, epoch5_row = checkpoints[5]
    winner_path, winner_sha, winner_row = checkpoints[20]
    receipt = {
        "schema_version": training.SCHEMA_VERSION,
        "status": "complete_train_validation_only",
        "fingerprint_sha256": "2" * 64,
        "config": run_config,
        "model_config": model_config,
        "population": {
            "train_total": 24_000,
            "val_total": 3_000,
            "train_positive": 12_000,
            "val_positive": 1_500,
        },
        "epochs": epoch_rows,
        "winner_epoch": 20,
        "winner_checkpoint": winner_path.name,
        "winner_checkpoint_sha256": winner_sha,
        "winner_selection_metrics": winner_row["selection_metrics"],
        "winner_state_dict": raw_state_path.name,
        "winner_state_dict_sha256": _file_sha(raw_state_path),
        "validation_threshold": _threshold(
            winner_sha, model_config, 0.55
        ).to_dict(),
        "same_exposure_epoch5": {
            "checkpoint": epoch5_path.name,
            "checkpoint_sha256": epoch5_sha,
            "selection_metrics": epoch5_row["selection_metrics"],
            "validation_scores": epoch5_row["validation_scores"],
            "validation_scores_sha256": epoch5_row["validation_scores_sha256"],
            "validation_threshold": _threshold(
                epoch5_sha, model_config, 0.45
            ).to_dict(),
            "selection_and_threshold_source": "val_only",
            "test_accessed": False,
            "real_external_test_accessed": False,
        },
        "stop_reason": "validation_early_stop",
        "actual_stop_epoch": 20,
        "early_stop_anchor_epoch": 8,
        "early_stop_anchor_auroc": 0.9,
        "epochs_without_qualifying_improvement": 12,
        "convergence_claim": "validation_plateau_under_declared_rule",
        "test_accessed": False,
        "real_external_test_accessed": False,
    }
    _write_json(run / "run_receipt.json", receipt)
    return run


def test_freezes_converged_and_same_exposure_before_any_dataset_path(tmp_path):
    run = _completed_matched_run(tmp_path)

    receipt, receipt_sha, winners = evaluation.freeze_matched_mm_winners(run)

    assert receipt_sha == _file_sha(run / "run_receipt.json")
    assert tuple(winners) == evaluation.MATCHED_METHODS
    assert winners[evaluation.CONVERGED_METHOD].epoch == 20
    assert winners[evaluation.SAME_EXPOSURE_METHOD].epoch == 5
    assert winners[evaluation.CONVERGED_METHOD].threshold.threshold == 0.55
    assert winners[evaluation.SAME_EXPOSURE_METHOD].threshold.threshold == 0.45
    assert receipt["population"]["train_positive"] == 12_000
    assert all(not value.model.training for value in winners.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_state_dict_equality_is_independent_of_tensor_device():
    cpu_state = {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    cuda_state = {"weight": cpu_state["weight"].cuda()}

    assert evaluation._state_dicts_equal(cpu_state, cuda_state)

    cuda_state["weight"][1] = 3.0
    assert not evaluation._state_dicts_equal(cpu_state, cuda_state)


def test_freeze_rejects_unconverged_or_wrong_population_and_threshold_binding(tmp_path):
    for mutation, message in (
        (("stop_reason", "hard_cap_reached"), "validation plateau"),
        (("population", {"train_total": 1}), "population"),
    ):
        run = _completed_matched_run(tmp_path / mutation[0])
        receipt_path = run / "run_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt[mutation[0]] = mutation[1]
        _write_json(receipt_path, receipt)
        with pytest.raises(evaluation.RachelMatchedMMEvaluationError, match=message):
            evaluation.freeze_matched_mm_winners(run)

    run = _completed_matched_run(tmp_path / "threshold")
    receipt_path = run / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["validation_threshold"]["checkpoint_sha256"] = "f" * 64
    _write_json(receipt_path, receipt)
    with pytest.raises(evaluation.RachelMatchedMMEvaluationError, match="different checkpoint"):
        evaluation.freeze_matched_mm_winners(run)


class _FakeHistorical(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.shapes = []

    def forward(self, first, second):
        self.shapes.append((tuple(first.shape), tuple(second.shape)))
        probability = (first.mean(dim=(1, 2, 3)) + second.mean(dim=(1, 2, 3))) / 2
        return probability[:, None]


def _fake_winner(model: nn.Module):
    config = {"architecture": "HistoricalMMSiamese"}
    return evaluation.FrozenMatchedMMWinner(
        method=evaluation.CONVERGED_METHOD,
        epoch=20,
        checkpoint_path=Path("unused.pt"),
        checkpoint_sha256="2" * 64,
        model_config=config,
        model_config_sha256=_canonical_sha(config),
        threshold=PairwiseThresholdArtifact(
            threshold=0.5,
            fit_method="maximize_cluster_balanced_f1",
            source_split="val",
            validation_fingerprint_sha256="1" * 64,
            checkpoint_sha256="2" * 64,
            model_config_sha256=_canonical_sha(config),
            aggregation_config_sha256=_canonical_sha(
                {"pair_score": evaluation.MATCHED_SCORE}
            ),
            sample_count=4,
            cluster_count=2,
            achieved_cluster_balanced_f1=0.5,
            achieved_cluster_balanced_precision=0.5,
            achieved_cluster_balanced_recall=0.5,
        ),
        model=model,
    )


def _mask(path: Path, shift: int) -> None:
    value = np.zeros((800, 800), dtype=np.uint8)
    value[100 + shift : 500 + shift, 150:650] = 255
    Image.fromarray(value, mode="L").save(path)


def test_synthetic_scoring_opens_only_binary_model_masks_and_emits_honest_semantics(
    tmp_path,
):
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    _mask(first, 0)
    _mask(second, 20)
    model = _FakeHistorical()
    winner = _fake_winner(model)
    pairs = (evaluation.SyntheticMaskPair("p0", first, second),)

    predictions = evaluation.score_synthetic_mask_pairs(winner, pairs)
    rows = (
        SimpleNamespace(pair_id="p0", label=True, cluster_id="case-0"),
    )
    records = evaluation.synthetic_pair_records(winner, predictions, rows)

    assert model.shapes == [((1, 1, 64, 64), (1, 1, 64, 64))]
    assert records[0]["main_score"] == "historical_mm_probability"
    assert set(records[0]["scores"]) == {"historical_mm_probability"}
    assert records[0]["decision"]["validation_threshold"] == 0.5


def test_real_scoring_uses_only_alpha_derived_bool_mask_and_rejects_duplicate_pairs():
    first = np.zeros((800, 800), dtype=np.bool_)
    second = np.zeros((800, 800), dtype=np.bool_)
    first[100:400, 200:600] = True
    second[120:420, 180:580] = True
    fragments = {
        "f0": SimpleNamespace(mask=first, points_rc="must-not-be-read"),
        "f1": SimpleNamespace(mask=second, points_rc="must-not-be-read"),
    }
    pair = SimpleNamespace(
        pair_id="p0", fragment_a_id="f0", fragment_b_id="f1"
    )
    model = _FakeHistorical()
    predictions = evaluation.score_alpha_derived_mask_pairs(
        _fake_winner(model), (pair,), fragments
    )
    assert predictions[0].valid is True
    assert model.shapes == [((1, 1, 64, 64), (1, 1, 64, 64))]

    with pytest.raises(evaluation.RachelMatchedMMEvaluationError, match="duplicated"):
        evaluation.score_alpha_derived_mask_pairs(
            _fake_winner(_FakeHistorical()), (pair, pair), fragments
        )


def test_cublas_configuration_precedes_torch_import_and_adapter_has_no_legacy_import():
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    assert source.index('os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG"') < source.index(
        "import torch"
    )
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "pairwise_v0_1" not in source
    assert "historical_mm_siamese import" not in source
    assert "matched_route_a_siamese" not in source

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from staging.pairwise_v0_2.baselines import (
    exact_seam_post_pilot_real_evaluation as post_pilot,
)
from staging.pairwise_v0_2.baselines.mm_validation_comparison import LoadedArmWinner
from staging.pairwise_v0_2.baselines.real_dunhuang_balanced_distractors import (
    BALANCED_DISTRACTOR_LABEL_ORIGIN,
)
from staging.pairwise_v0_2.baselines.real_dunhuang_evaluation import (
    RealBatchPrediction,
    RealPairDataset,
    load_strict_real_pair_dataset,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.tests.test_real_dunhuang_evaluation import (
    _fixture_documents,
    _geometry_config,
)
from staging.pairwise_v0_2.training.geometry_batch import DATA_DIRECTION_TO_INDEX
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache
from staging.pairwise_v0_2.training.local_q1_backend import (
    ExactSeamStepConfig,
    LocalQ1Backend,
    LocalQ1BackendMode,
)


def _fake_loaded_pair() -> post_pilot.LoadedPostPilotPair:
    backend = LocalQ1Backend(
        device="cpu",
        mode=LocalQ1BackendMode.FORMAL,
        exact_seam_step_config=ExactSeamStepConfig(loss_weight=0.25),
    )
    weak = post_pilot._pilot_arm(
        backend, post_pilot._METHOD_ARM[post_pilot.WEAK_METHOD]
    )
    exact = post_pilot._pilot_arm(
        backend, post_pilot._METHOD_ARM[post_pilot.EXACT_METHOD]
    )
    return post_pilot.LoadedPostPilotPair(
        weak=LoadedArmWinner(weak, object(), {}),
        exact=LoadedArmWinner(exact, object(), {}),
        pilot_version=post_pilot.EXACT_SEAM_PILOT_V02,
        seed=17,
        exact_loss_weight=0.25,
        checkpoint_epochs={"weak": 3, "exact": 3},
    )


def test_post_pilot_real_pairs_share_blinded_batch_then_join_side_table(tmp_path):
    manifest, receipt = _fixture_documents(tmp_path)
    dataset = load_strict_real_pair_dataset(
        manifest,
        receipt,
        target_long_side=96,
        expected_case_count=2,
        expected_pair_count=4,
        expected_positive_count=3,
        expected_negative_count=1,
    )
    labels = {record.pair_id: record.label for record in dataset.records}
    directions = {
        record.pair_id: (
            -1
            if record.direction_b_wrt_a is None
            else DATA_DIRECTION_TO_INDEX[record.direction_b_wrt_a]
        )
        for record in dataset.records
    }
    calls = []

    def perfect_scorer(loaded, prepared):
        payload = prepared.payload
        calls.append(
            {
                "arm": loaded.arm.name.value,
                "prepared_id": id(prepared),
                "sample_ids": tuple(payload.sample_ids),
                "labels": payload.labels.clone(),
                "direction_valid": payload.direction_target_valid.clone(),
                "has_exact_targets": payload.exact_loss_targets() is not None,
            }
        )
        is_exact = loaded.arm.name is post_pilot._METHOD_ARM[post_pilot.EXACT_METHOD]
        positive_probability = 0.95 if is_exact else 0.9
        negative_probability = 0.05 if is_exact else 0.1
        probability = torch.tensor(
            [
                positive_probability if labels[pair_id] else negative_probability
                for pair_id in payload.sample_ids
            ],
            dtype=torch.float32,
        )
        direction = torch.tensor(
            [max(0, directions[pair_id]) for pair_id in payload.sample_ids],
            dtype=torch.long,
        )
        return RealBatchPrediction(
            probability=probability,
            valid=torch.ones(len(payload.sample_ids), dtype=torch.bool),
            best_direction_index=direction,
        )

    try:
        result = post_pilot.evaluate_exact_seam_post_pilot_real(
            dataset=dataset,
            geometry_config=_geometry_config(),
            geometry_cache=GeometryArtifactCache(tmp_path / "post-real-cache"),
            models=_fake_loaded_pair(),
            arm_scorer=perfect_scorer,
            batch_size=2,
        )
    finally:
        dataset.mask_loader.close()

    assert len(calls) == 4
    for weak_call, exact_call in zip(calls[::2], calls[1::2]):
        assert weak_call["prepared_id"] == exact_call["prepared_id"]
        assert weak_call["sample_ids"] == exact_call["sample_ids"]
        assert not weak_call["labels"].any()
        assert not exact_call["labels"].any()
        assert not weak_call["direction_valid"].any()
        assert not exact_call["direction_valid"].any()
        assert weak_call["has_exact_targets"] is False
        assert exact_call["has_exact_targets"] is False

    strict = result["populations"]["strict547"]
    assert strict["pair_count"] == 4
    assert strict["common_valid_count"] == 4
    assert strict["threshold_fitted_or_selected_on_real"] is False
    assert strict["threshold_metrics_reported"] is False
    for method in (post_pilot.WEAK_METHOD, post_pilot.EXACT_METHOD):
        row = strict["methods"][method]
        assert row["native_coverage"] == pytest.approx(1.0)
        assert row["common_valid_ranking"]["row"]["auroc"] == pytest.approx(1.0)
        assert row["common_valid_ranking"]["row"]["auprc"] == pytest.approx(1.0)
        assert row["direction_on_common_valid"][
            "accuracy_invalid_as_incorrect"
        ] == pytest.approx(1.0)
    assert strict["exact_minus_weak"]["common_valid_ranking"]["row"] == {
        "auroc": pytest.approx(0.0),
        "auprc": pytest.approx(0.0),
    }
    negative = result["strata"]["strict_manifest_negative_39"]
    assert negative["pair_count"] == 1
    assert negative["methods"]["weak"]["common_valid_ranking"] is None
    assert negative["methods"]["weak"]["direction_on_common_valid"] is None
    assert result["fairness"]["same_prepared_batch_object_per_pair_of_forwards"]
    assert result["preprocessing"]["rgb_used"] is False
    assert result["preprocessing"]["text_or_ocr_used"] is False
    assert result["preprocessing"]["bbox_or_canvas_coordinate_model_input"] is False
    assert [row["pair_id"] for row in result["pairs"]] == [
        record.pair_id for record in dataset.records
    ]
    assert all(row["methods"]["weak"]["valid"] for row in result["pairs"])
    assert all(
        row["methods"]["weak"]["predicted_direction"] is not None
        for row in result["pairs"]
        if row["label"]
    )


def test_optional_matched_siamese_uses_same_ordered_alpha_pairs_and_metrics(
    tmp_path,
):
    manifest, receipt = _fixture_documents(tmp_path)
    dataset = load_strict_real_pair_dataset(
        manifest,
        receipt,
        target_long_side=96,
        expected_case_count=2,
        expected_pair_count=4,
        expected_positive_count=3,
        expected_negative_count=1,
    )
    labels = {record.pair_id: record.label for record in dataset.records}
    directions = {
        record.pair_id: (
            -1
            if record.direction_b_wrt_a is None
            else DATA_DIRECTION_TO_INDEX[record.direction_b_wrt_a]
        )
        for record in dataset.records
    }

    def local_scorer(loaded, prepared):
        is_exact = loaded.arm.name is post_pilot._METHOD_ARM[post_pilot.EXACT_METHOD]
        probability = torch.tensor(
            [
                (0.85 if labels[pair_id] else 0.15)
                if is_exact
                else (0.75 if labels[pair_id] else 0.25)
                for pair_id in prepared.payload.sample_ids
            ],
            dtype=torch.float32,
        )
        direction = torch.tensor(
            [max(0, directions[pair_id]) for pair_id in prepared.payload.sample_ids],
            dtype=torch.long,
        )
        return RealBatchPrediction(
            probability=probability,
            valid=torch.ones(len(probability), dtype=torch.bool),
            best_direction_index=direction,
        )

    observed_siamese_batches = []

    class ConstantWholeMaskSiamese(torch.nn.Module):
        def forward(self, input_a, input_b):
            assert input_a.ndim == 4 and input_b.ndim == 4
            assert tuple(input_a.shape[1:]) == (1, 64, 64)
            assert tuple(input_b.shape[1:]) == (1, 64, 64)
            observed_siamese_batches.append(len(input_a))
            return torch.full(
                (len(input_a), 1),
                0.5,
                dtype=input_a.dtype,
                device=input_a.device,
            )

    try:
        result = post_pilot.evaluate_exact_seam_post_pilot_real(
            dataset=dataset,
            geometry_config=_geometry_config(),
            geometry_cache=GeometryArtifactCache(tmp_path / "siamese-real-cache"),
            models=_fake_loaded_pair(),
            arm_scorer=local_scorer,
            batch_size=2,
            siamese_model=ConstantWholeMaskSiamese(),
            siamese_device="cpu",
            siamese_batch_size=2,
        )
    finally:
        dataset.mask_loader.close()

    assert observed_siamese_batches == [2, 2]
    assert result["evaluated_methods"] == ["weak", "exact", "siamese"]
    assert result["siamese_inference_batch_count"] == 2
    strict = result["populations"]["strict547"]
    assert strict["common_valid_count"] == 4
    assert strict["primary_metric_population"] == (
        "intersection_valid:weak,exact,siamese"
    )
    for method in ("weak", "exact", "siamese"):
        assert strict["methods"][method]["native_valid_count"] == 4
        assert strict["methods"][method]["native_ranking"] is not None
        assert strict["methods"][method]["common_valid_ranking"] is not None
    assert strict["methods"]["siamese"]["native_ranking"]["row"][
        "auroc"
    ] == pytest.approx(0.5)
    assert strict["methods"]["siamese"]["direction_on_common_valid"] is None
    assert strict["exact_minus_siamese"]["common_valid_ranking"] is not None
    assert all(
        row["methods"]["siamese"]
        == {
            "probability": pytest.approx(0.5),
            "valid": True,
            "predicted_direction": None,
        }
        for row in result["pairs"]
    )
    assert result["fairness"]["siamese_receives_same_ordered_pair_ids"] is True
    assert result["fairness"]["siamese_uses_local_candidate_tensor"] is False
    assert result["preprocessing"]["siamese_bbox_direction_or_rgb_model_input"] is False


def test_pilot_loader_reuses_restricted_checkpoint_loader_and_remote_basename(
    tmp_path, monkeypatch
):
    _patch_lightweight_loader_backend(monkeypatch)
    summary_path = tmp_path / "exact_seam_pilot_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "pilot_version": post_pilot.EXACT_SEAM_PILOT_V01,
                "status": "complete",
                "config": {"seed": 31, "epochs": 3, "exact_loss_weight": 0.25},
                "checkpoints": {
                    "weak": {
                        "path": "/remote/run/weak_keypoint_sinkhorn.pt",
                        "file_sha256": "a" * 64,
                        "canonical_content_sha256": "b" * 64,
                    },
                    "exact": {
                        "path": "/remote/run/exact_keypoint_sinkhorn.pt",
                        "file_sha256": "c" * 64,
                        "canonical_content_sha256": "d" * 64,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def restricted_loader(path, model, **kwargs):
        del model
        calls.append((Path(path), kwargs))
        return {"epoch": 3}

    monkeypatch.setattr(post_pilot, "load_trusted_checkpoint", restricted_loader)
    loaded = post_pilot.load_exact_seam_pilot_pair(summary_path, device="cpu")

    assert loaded.seed == 31
    assert loaded.checkpoint_epochs == {"weak": 3, "exact": 3}
    assert [path for path, _ in calls] == [
        tmp_path / "weak_keypoint_sinkhorn.pt",
        tmp_path / "exact_keypoint_sinkhorn.pt",
    ]
    assert all(kwargs["trusted"] is True for _, kwargs in calls)
    assert all(kwargs["map_location"] == "cpu" for _, kwargs in calls)
    assert calls[0][1]["expected_file_sha256"] == "a" * 64
    assert calls[1][1]["expected_canonical_content_sha256"] == "d" * 64
    assert (
        calls[0][1]["expected_config"]["pilot_version"]
        == post_pilot.EXACT_SEAM_PILOT_V01
    )
    assert calls[0][1]["expected_config"]["arm"]["training"][
        "exact_assignment_supervision"
    ] == {"enabled": False}
    assert (
        calls[1][1]["expected_config"]["arm"]["training"][
            "exact_assignment_supervision"
        ]["enabled"]
        is True
    )

    def wrong_final_epoch(path, model, **kwargs):
        del model, kwargs
        epoch = 2 if Path(path).name.startswith("exact_") else 3
        return {"epoch": epoch}

    monkeypatch.setattr(post_pilot, "load_trusted_checkpoint", wrong_final_epoch)
    with pytest.raises(post_pilot.PostPilotRealEvaluationError, match="loaded epoch"):
        post_pilot.load_exact_seam_pilot_pair(summary_path, device="cpu")


class _FakeCheckpointModel:
    def eval(self):
        return self


class _FakeLoaderBackend:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def create_session(self, arm, *, seed):
        del arm, seed
        return SimpleNamespace(model=_FakeCheckpointModel())


def _patch_lightweight_loader_backend(monkeypatch):
    monkeypatch.setattr(post_pilot, "LocalQ1Backend", _FakeLoaderBackend)

    def arm(_backend, name):
        exact = name is post_pilot._METHOD_ARM[post_pilot.EXACT_METHOD]
        return SimpleNamespace(
            name=name,
            model_config={
                "training": {
                    "exact_assignment_supervision": {"enabled": exact},
                }
            },
        )

    monkeypatch.setattr(post_pilot, "_pilot_arm", arm)


def _v02_summary():
    validation = {
        1: {
            "weak": {"auroc": 0.80, "auprc": 0.70},
            "exact": {"auroc": 0.70, "auprc": 0.65},
        },
        2: {
            "weak": {"auroc": 0.91, "auprc": 0.82},
            "exact": {"auroc": 0.82, "auprc": 0.80},
        },
        3: {
            "weak": {"auroc": 0.88, "auprc": 0.90},
            "exact": {"auroc": 0.93, "auprc": 0.84},
        },
    }

    def receipt(method, epoch):
        token = "a" if method == "weak" else "b"
        return {
            "path": "/remote/run/{}_keypoint_sinkhorn_epoch_{:03d}.pt".format(
                method, epoch
            ),
            "file_sha256": token * 64,
            "canonical_content_sha256": ("c" if method == "weak" else "d") * 64,
            "epoch": epoch,
        }

    epoch_rows = []
    histories = {method: [] for method in ("weak", "exact")}
    for epoch in (1, 2, 3):
        receipts = {method: receipt(method, epoch) for method in ("weak", "exact")}
        epoch_rows.append(
            {
                "epoch": epoch,
                "validation": validation[epoch],
                "checkpoints": receipts,
            }
        )
        for method in histories:
            histories[method].append(receipts[method])
    winner_epochs = {"weak": 2, "exact": 3}
    winners = {
        method: {
            "epoch": epoch,
            "validation": validation[epoch][method],
            "checkpoint": dict(histories[method][epoch - 1]),
        }
        for method, epoch in winner_epochs.items()
    }
    return {
        "pilot_version": post_pilot.EXACT_SEAM_PILOT_V02,
        "status": "complete",
        "config": {"seed": 31, "epochs": 3, "exact_loss_weight": 0.25},
        "winner_selection": {
            "policy": post_pilot._V02_WINNER_POLICY,
            "population": post_pilot._V02_WINNER_POPULATION,
            "real_evaluation_accessed": False,
            "winners": winners,
        },
        "epochs": epoch_rows,
        "epoch_checkpoints": histories,
        "checkpoints": {
            method: dict(histories[method][epoch - 1])
            for method, epoch in winner_epochs.items()
        },
    }


def test_v02_loader_accepts_distinct_synthetic_validation_winner_epochs(
    tmp_path, monkeypatch
):
    _patch_lightweight_loader_backend(monkeypatch)
    summary = _v02_summary()
    summary_path = tmp_path / "exact_seam_pilot_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    validation = {
        (method, int(row["epoch"])): row["validation"][method]
        for row in summary["epochs"]
        for method in ("weak", "exact")
    }

    def restricted_loader(path, model, **kwargs):
        del model, kwargs
        method = "weak" if Path(path).name.startswith("weak_") else "exact"
        epoch = int(Path(path).stem.rsplit("_", 1)[1])
        return {
            "epoch": epoch,
            "metrics": validation[(method, epoch)],
            "provenance": {
                "synthetic_validation_only": True,
                "real_evaluation_accessed": False,
            },
        }

    monkeypatch.setattr(post_pilot, "load_trusted_checkpoint", restricted_loader)
    loaded = post_pilot.load_exact_seam_pilot_pair(summary_path, device="cpu")

    assert loaded.pilot_version == post_pilot.EXACT_SEAM_PILOT_V02
    assert loaded.checkpoint_epochs == {"weak": 2, "exact": 3}


@pytest.mark.parametrize("mutation", ("real_access", "winner_receipt"))
def test_v02_rejects_non_synthetic_or_inconsistent_winner(mutation):
    summary = copy.deepcopy(_v02_summary())
    if mutation == "real_access":
        summary["winner_selection"]["real_evaluation_accessed"] = True
        match = "accessed real evaluation"
    else:
        summary["winner_selection"]["winners"]["weak"]["checkpoint"]["path"] = (
            "/remote/run/different.pt"
        )
        match = "path/receipt"
    with pytest.raises(post_pilot.PostPilotRealEvaluationError, match=match):
        post_pilot._checkpoint_selection(
            summary,
            summary["checkpoints"],
            pilot_version=post_pilot.EXACT_SEAM_PILOT_V02,
            planned_epochs=3,
        )


@pytest.mark.parametrize("failure", ("loaded_epoch", "checkpoint_provenance"))
def test_v02_loader_rejects_wrong_loaded_epoch_or_real_provenance(
    tmp_path, monkeypatch, failure
):
    _patch_lightweight_loader_backend(monkeypatch)
    summary = _v02_summary()
    summary_path = tmp_path / "exact_seam_pilot_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    def restricted_loader(path, model, **kwargs):
        del model, kwargs
        method = "weak" if Path(path).name.startswith("weak_") else "exact"
        epoch = int(Path(path).stem.rsplit("_", 1)[1])
        metrics = next(
            row["validation"][method]
            for row in summary["epochs"]
            if row["epoch"] == epoch
        )
        return {
            "epoch": epoch - 1 if failure == "loaded_epoch" else epoch,
            "metrics": metrics,
            "provenance": {
                "synthetic_validation_only": True,
                "real_evaluation_accessed": failure == "checkpoint_provenance",
            },
        }

    monkeypatch.setattr(post_pilot, "load_trusted_checkpoint", restricted_loader)
    match = "loaded epoch" if failure == "loaded_epoch" else "synthetic-only"
    with pytest.raises(post_pilot.PostPilotRealEvaluationError, match=match):
        post_pilot.load_exact_seam_pilot_pair(summary_path, device="cpu")


def test_balanced_suffix_and_invalid_prediction_use_common_valid_without_fill(
    tmp_path,
):
    manifest, receipt = _fixture_documents(tmp_path)
    strict = load_strict_real_pair_dataset(
        manifest,
        receipt,
        target_long_side=96,
        expected_case_count=2,
        expected_pair_count=4,
        expected_positive_count=3,
        expected_negative_count=1,
    )
    first = strict.records[0].fragment_a
    second = strict.records[1].fragment_a
    constructed_cluster = "constructed-fixture-cluster"
    first = replace(
        first,
        canonical_group_id=constructed_cluster,
        component_id=constructed_cluster,
    )
    second = replace(
        second,
        canonical_group_id=constructed_cluster,
        component_id=constructed_cluster,
    )
    constructed = TrainingPairRecord(
        fragment_a=first,
        fragment_b=second,
        label=False,
        direction_b_wrt_a=None,
        dataset_id=first.dataset_id,
        canonical_group_id=constructed_cluster,
        component_id=constructed_cluster,
        split="val",
        canonical_pair_key=tuple(sorted((first.fragment_id, second.fragment_id))),
        label_origin=BALANCED_DISTRACTOR_LABEL_ORIGIN,
        provenance={"constructed_is_ground_truth_negative": False},
    )
    dataset = RealPairDataset(
        records=tuple(strict.records) + (constructed,),
        mask_loader=strict.mask_loader,
        manifest_sha256=strict.manifest_sha256,
        case_count=strict.case_count,
        positive_count=3,
        negative_count=2,
        target_long_side=96,
    )
    label_by_id = {record.pair_id: record.label for record in dataset.records}
    direction_by_id = {
        record.pair_id: (
            -1
            if record.direction_b_wrt_a is None
            else DATA_DIRECTION_TO_INDEX[record.direction_b_wrt_a]
        )
        for record in dataset.records
    }

    def scorer(loaded, prepared):
        is_exact = loaded.arm.name is post_pilot._METHOD_ARM[post_pilot.EXACT_METHOD]
        probabilities = []
        valid = []
        directions = []
        for pair_id in prepared.payload.sample_ids:
            invalid = is_exact and pair_id == constructed.pair_id
            probabilities.append(
                float("nan") if invalid else (0.9 if label_by_id[pair_id] else 0.1)
            )
            valid.append(not invalid)
            directions.append(max(0, direction_by_id[pair_id]))
        return RealBatchPrediction(
            probability=torch.tensor(probabilities, dtype=torch.float32),
            valid=torch.tensor(valid, dtype=torch.bool),
            best_direction_index=torch.tensor(directions, dtype=torch.long),
        )

    try:
        result = post_pilot.evaluate_exact_seam_post_pilot_real(
            dataset=dataset,
            geometry_config=_geometry_config(),
            geometry_cache=GeometryArtifactCache(tmp_path / "balanced-cache"),
            models=_fake_loaded_pair(),
            arm_scorer=scorer,
            batch_size=2,
        )
    finally:
        dataset.mask_loader.close()

    assert result["populations"]["strict547"]["pair_count"] == 4
    balanced = result["populations"]["balanced1016"]
    assert balanced["pair_count"] == 5
    assert balanced["common_valid_count"] == 4
    assert balanced["methods"]["weak"]["native_coverage"] == pytest.approx(1.0)
    assert balanced["methods"]["exact"]["native_coverage"] == pytest.approx(0.8)
    assert balanced["methods"]["weak"]["common_valid_ranking"]["row"][
        "auroc"
    ] == pytest.approx(1.0)
    constructed_stratum = result["strata"]["constructed_distractor_469"]
    assert constructed_stratum["pair_count"] == 1
    assert constructed_stratum["common_valid_count"] == 0
    assert constructed_stratum["methods"]["exact"]["native_coverage"] == 0.0
    pair = result["pairs"][-1]
    assert pair["pair_id"] == constructed.pair_id
    assert pair["methods"]["weak"]["probability"] == pytest.approx(0.1)
    assert pair["methods"]["exact"] == {
        "probability": None,
        "valid": False,
        "predicted_direction": None,
    }
    assert pair["exact_minus_weak_probability"] is None


def test_cli_exposes_single_pass_balanced_extension():
    parsed = post_pilot._parser().parse_args(
        [
            "--pilot-summary",
            "pilot/exact_seam_pilot_summary.json",
            "--manifest",
            "real/manifest.json",
            "--local-path-receipt",
            "real/paths.json",
            "--geometry-cache-dir",
            "cache",
            "--output",
            "result.json",
            "--include-balanced-1016",
        ]
    )
    assert parsed.pilot_summary == Path("pilot/exact_seam_pilot_summary.json")
    assert parsed.include_balanced_1016 is True
    assert parsed.device == "cuda"
    assert parsed.batch_size == 1
    assert parsed.siamese_checkpoint is None
    assert parsed.siamese_batch_size == 256

    with_siamese = post_pilot._parser().parse_args(
        [
            "--pilot-summary",
            "pilot/exact_seam_pilot_summary.json",
            "--manifest",
            "real/manifest.json",
            "--local-path-receipt",
            "real/paths.json",
            "--geometry-cache-dir",
            "cache",
            "--output",
            "result.json",
            "--siamese-checkpoint",
            "control/exact_pilot_matched_siamese_winner.pt",
            "--siamese-batch-size",
            "64",
        ]
    )
    assert with_siamese.siamese_checkpoint == Path(
        "control/exact_pilot_matched_siamese_winner.pt"
    )
    assert with_siamese.siamese_batch_size == 64

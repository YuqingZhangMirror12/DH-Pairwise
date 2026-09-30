from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from staging.pairwise_v0_2.baselines import (
    rachel_same_data_benchmark_eval_adapter as subject,
)
from staging.pairwise_v0_2.training.evaluation import PairwiseThresholdArtifact
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelBatch


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _pairing_bundle(root: Path, *, converged: bool = True) -> Path:
    root.mkdir()
    artifact = {
        "schema_version": "dunhuang-pairwise-threshold/0.2",
        "threshold": 0.4,
        "fit_method": "maximize_cluster_balanced_f1",
        "source_split": "validation",
        "validation_fingerprint_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "model_config_sha256": "3" * 64,
        "aggregation_config_sha256": "4" * 64,
        "sample_count": 3000,
        "cluster_count": 130,
        "achieved_cluster_balanced_f1": 0.5,
        "achieved_cluster_balanced_precision": 0.5,
        "achieved_cluster_balanced_recall": 0.5,
    }
    artifact_sha = subject._canonical_sha256(artifact)
    threshold = {
        "schema_version": "rachel-pairingnet-validation-threshold/1.0",
        "method_id": subject.pairing.METHOD_ID,
        "artifact": artifact,
        "artifact_content_sha256": artifact_sha,
        "winner_checkpoint_sha256": "ignored-by-mocked-loader",
    }
    _write_json(root / "validation_threshold.json", threshold)
    contract = {
        "schema_version": "rachel-pairingnet-frozen-inference-contract/1.0",
        "method_id": subject.pairing.METHOD_ID,
        "orientation": "upright_known_translation_only_primary",
        "pair_decision": {
            "status": "frozen",
            "threshold": 0.4,
            "threshold_artifact_sha256": artifact_sha,
        },
        "sealed_synthetic_accessed": False,
        "real_data_accessed": False,
    }
    _write_json(root / "inference_contract.json", contract)
    for name in (
        "winner.pt",
        "last.pt",
        "winner_validation_report.json",
        "winner_validation_predictions.jsonl",
    ):
        (root / name).write_bytes((name + "\n").encode("utf-8"))
    receipt = {
        "schema_version": subject.pairing.SCHEMA_VERSION,
        "status": "train_validation_complete",
        "method_id": subject.pairing.METHOD_ID,
        "official_commit": subject.pairing.OFFICIAL_COMMIT,
        "winner_checkpoint_kind": subject.pairing.WINNER_CHECKPOINT_KIND,
        "adaptation_claim": "same_data_method_adaptation_not_exact_reproduction",
        "selection_threshold_used": False,
        "sealed_synthetic_accessed": False,
        "real_data_accessed": False,
        "convergence_demonstrated": converged,
        "stop_reason": "validation_plateau" if converged else "max_epochs",
        "population_audit": {
            "formal_population_required": True,
            "parent_lineage_disjoint": True,
            "sealed_synthetic_accessed": False,
            "real_data_accessed": False,
            "population": {
                "train": {"rows": subject.pairing.FORMAL_TRAIN_ROWS},
                "val": {"rows": subject.pairing.FORMAL_VAL_ROWS},
            },
            "manifests": {
                "train_sha256": "d" * 64,
                "val_sha256": "e" * 64,
            },
        },
        "adaptation": {
            "mask_only": True,
            "primary_pose": "upright_translation_only_consensus",
            "pair_head_is_official_component": False,
        },
        "winner_checkpoint_sha256": _sha(root / "winner.pt"),
        "last_checkpoint_sha256": _sha(root / "last.pt"),
        "validation_threshold": threshold,
        "validation_threshold_file_sha256": _sha(
            root / "validation_threshold.json"
        ),
        "inference_contract_sha256": _sha(root / "inference_contract.json"),
        "winner_validation_report_sha256": _sha(
            root / "winner_validation_report.json"
        ),
        "winner_validation_predictions_sha256": _sha(
            root / "winner_validation_predictions.jsonl"
        ),
    }
    _write_json(root / "completion_receipt.json", receipt)
    return root


def _threshold() -> PairwiseThresholdArtifact:
    return PairwiseThresholdArtifact(
        threshold=0.6,
        fit_method="maximize_cluster_balanced_f1",
        source_split="validation",
        validation_fingerprint_sha256="1" * 64,
        checkpoint_sha256="c" * 64,
        model_config_sha256="2" * 64,
        aggregation_config_sha256="3" * 64,
        sample_count=3000,
        cluster_count=130,
        achieved_cluster_balanced_f1=0.7,
        achieved_cluster_balanced_precision=0.8,
        achieved_cluster_balanced_recall=0.65,
    )


def _shredding_bundle(root: Path) -> Path:
    root.mkdir()
    artifact = _threshold()
    checkpoints = {
        stage: {"sha256": digest}
        for stage, digest in (
            ("coarse", "a" * 64),
            ("matching", "b" * 64),
            ("classify", "c" * 64),
        )
    }
    value = {
        "schema_version": subject.shredding.FREEZE_SCHEMA_VERSION,
        "checkpoint_kind": subject.shredding.FREEZE_CHECKPOINT_KIND,
        "status": "complete_train_val_frozen_no_test_or_real",
        "method_id": subject.shredding.METHOD_ID,
        "official_commit": subject.shredding.OFFICIAL_COMMIT,
        "scope": {
            "mask_only": True,
            "rachel_n512": True,
            "upright_known_orientation": True,
            "train_val_only": True,
            "sealed_test_or_real_opened": False,
            "original_shreddingnet_reproduction": False,
            "global_assembly_performed": False,
        },
        "threshold": {
            **artifact.to_dict(),
            "content_sha256": artifact.content_sha256,
            "threshold_used_for_stage_winner_selection": False,
            "fit_after_all_three_winners_fixed": True,
        },
        "checkpoints": checkpoints,
        "dataset_binding": {
            "manifest_content_sha256": {
                "train": "d" * 64,
                "val": "e" * 64,
            }
        },
    }
    value["content_sha256"] = subject._canonical_sha256(value)
    path = root / "train_val_freeze.json"
    _write_json(path, value)
    return path


def test_common_prediction_rejects_non_nan_invalid_translation():
    with pytest.raises(ValueError, match="NaN"):
        subject.CommonBatchPrediction(
            schema_version=subject.SCHEMA_VERSION,
            method_key=subject.PAIRINGNET_METHOD_KEY,
            method_id="method",
            pair_ids=("p",),
            pair_probability=np.asarray([0.5], dtype=np.float32),
            decision_valid=np.asarray([True], dtype=np.bool_),
            translation_hat_rc=np.asarray([[0.0, 0.0]], dtype=np.float32),
            translation_valid=np.asarray([False], dtype=np.bool_),
            correspondence_indices=None,
            correspondence_scores=None,
            correspondence_semantics=None,
            auxiliary_scores={},
        )


def test_pairing_freeze_verifies_completion_hashes_and_disclosure(
    tmp_path, monkeypatch
):
    root = _pairing_bundle(tmp_path / "pairing")
    sentinel = object()
    monkeypatch.setattr(
        subject.pairing,
        "load_frozen_validation_threshold",
        lambda path, winner: 0.4,
    )
    monkeypatch.setattr(
        subject.pairing,
        "load_frozen_pairingnet_checkpoint",
        lambda path, device: sentinel,
    )
    frozen = subject.freeze_pairingnet_benchmark(root, device="cpu")
    assert frozen.method_key == subject.PAIRINGNET_METHOD_KEY
    assert frozen.threshold == 0.4
    assert frozen._predictor is sentinel
    assert frozen.provenance()["all_winners_and_validation_threshold_frozen"] is True
    assert frozen.provenance()["adaptation"]["pair_head_is_official_component"] is False


def test_pairing_freeze_rejects_unconverged_completion(tmp_path, monkeypatch):
    root = _pairing_bundle(tmp_path / "pairing", converged=False)
    monkeypatch.setattr(
        subject.pairing,
        "load_frozen_validation_threshold",
        lambda path, winner: 0.4,
    )
    monkeypatch.setattr(
        subject.pairing,
        "load_frozen_pairingnet_checkpoint",
        lambda path, device: object(),
    )
    with pytest.raises(subject.RachelBenchmarkEvalAdapterError, match="convergence"):
        subject.freeze_pairingnet_benchmark(root, device="cpu")


def test_pairing_freeze_rejects_post_receipt_file_tamper(tmp_path, monkeypatch):
    root = _pairing_bundle(tmp_path / "pairing")
    (root / "winner_validation_predictions.jsonl").write_text(
        "tampered\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        subject.pairing,
        "load_frozen_validation_threshold",
        lambda path, winner: 0.4,
    )
    monkeypatch.setattr(
        subject.pairing,
        "load_frozen_pairingnet_checkpoint",
        lambda path, device: object(),
    )
    with pytest.raises(subject.RachelBenchmarkEvalAdapterError, match="SHA-256"):
        subject.freeze_pairingnet_benchmark(root, device="cpu")


def test_shredding_freeze_verifies_content_scope_and_three_stage_binding(
    tmp_path, monkeypatch
):
    path = _shredding_bundle(tmp_path / "shredding")
    sentinel = SimpleNamespace()
    monkeypatch.setattr(
        subject.shredding, "load_frozen_inference", lambda path, device: sentinel
    )
    frozen = subject.freeze_shreddingnet_benchmark(path, device=torch.device("cpu"))
    assert frozen.method_key == subject.SHREDDINGNET_METHOD_KEY
    assert frozen.threshold == 0.6
    assert tuple(frozen.checkpoint_sha256_by_stage) == (
        "coarse",
        "matching",
        "classify",
    )
    assert frozen.adaptation_disclosure["native_cm_fm_se_or_ga_claimed"] is False


def test_shredding_freeze_rejects_content_tamper_before_loader(tmp_path, monkeypatch):
    path = _shredding_bundle(tmp_path / "shredding")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["scope"]["global_assembly_performed"] = True
    _write_json(path, value)
    called = False

    def loader(path, device):
        nonlocal called
        called = True
        return SimpleNamespace()

    monkeypatch.setattr(subject.shredding, "load_frozen_inference", loader)
    with pytest.raises(subject.RachelBenchmarkEvalAdapterError, match="content SHA"):
        subject.freeze_shreddingnet_benchmark(path, device="cpu")
    assert called is False


def test_dual_freeze_is_exact_and_ordered(monkeypatch):
    calls = []
    manifests = {"train": "d" * 64, "val": "e" * 64}
    first = SimpleNamespace(
        method_key=subject.PAIRINGNET_METHOD_KEY,
        training_manifest_sha256=manifests,
    )
    second = SimpleNamespace(
        method_key=subject.SHREDDINGNET_METHOD_KEY,
        training_manifest_sha256=manifests,
    )

    def pairing_freeze(path, *, device):
        calls.append(("pairing", path, str(device)))
        return first

    def shredding_freeze(path, *, device):
        calls.append(("shredding", path, str(device)))
        return second

    monkeypatch.setattr(subject, "freeze_pairingnet_benchmark", pairing_freeze)
    monkeypatch.setattr(subject, "freeze_shreddingnet_benchmark", shredding_freeze)
    frozen = subject.freeze_same_data_benchmarks(
        pairingnet_run_directory="pairing",
        shreddingnet_freeze_path="shredding/train_val_freeze.json",
        device="cpu",
    )
    assert tuple(frozen) == subject.BENCHMARK_METHODS
    assert calls == [
        ("pairing", "pairing", "cpu"),
        ("shredding", "shredding/train_val_freeze.json", "cpu"),
    ]


def _synthetic_batch() -> RachelBatch:
    count = 2
    masks = np.zeros((count, 1, 800, 800), dtype=np.float32)
    masks[:, :, 20:40, 30:50] = 1.0
    coarse = np.zeros((count, 1, 128, 128), dtype=np.float32)
    coarse[:, :, 3:7, 5:9] = 1.0
    points_a = np.zeros((count, 512, 2), dtype=np.float32)
    points_b = np.zeros((count, 512, 2), dtype=np.float32)
    square = np.asarray(
        [[10.0, 10.0], [10.0, 20.0], [20.0, 20.0], [20.0, 10.0]],
        dtype=np.float32,
    )
    points_a[:, :4] = square
    points_b[0, :4] = square + np.asarray([5.0, 0.0], dtype=np.float32)
    points_b[1, :4] = square
    valid = np.zeros((count, 512), dtype=np.bool_)
    valid[:, :4] = True
    target_a = np.full((count, 512), -2, dtype=np.int64)
    target_b = np.full((count, 512), -2, dtype=np.int64)
    target_a[0, :4] = np.arange(4)
    target_b[0, :4] = np.arange(4)
    target_a[1, :4] = -1
    target_b[1, :4] = -1
    return RachelBatch(
        pair_ids=("positive", "negative"),
        fragment_a_tokens=("a1", "a2"),
        fragment_b_tokens=("b1", "b2"),
        mask_a=masks,
        mask_b=masks.copy(),
        coarse_mask_a=coarse,
        coarse_mask_b=coarse.copy(),
        points_rc_a=points_a,
        points_rc_b=points_b,
        contour_valid_a=valid,
        contour_valid_b=valid.copy(),
        target_a=target_a,
        target_b=target_b,
        labels=np.asarray([1.0, 0.0], dtype=np.float32),
        translation_a_to_b_rc=np.asarray([[5.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        translation_a_to_b_xy_cartesian=np.asarray(
            [[0.0, -5.0], [0.0, 0.0]], dtype=np.float32
        ),
        translation_valid=np.asarray([True, False], dtype=np.bool_),
    )


class _FakeFrozen:
    method_key = subject.PAIRINGNET_METHOD_KEY
    method_id = "fixture-method"
    threshold = 0.5
    threshold_artifact_sha256 = "f" * 64

    def __init__(self, *, valid_pose=True):
        self.calls = 0
        self.valid_pose = valid_pose

    def predict_batch(self, batch, *, return_correspondence=False):
        self.calls += 1
        assert return_correspondence is True
        translation_valid = np.asarray(
            [self.valid_pose, self.valid_pose], dtype=np.bool_
        )
        translation = np.asarray([[5.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        translation[~translation_valid] = np.nan
        return subject.CommonBatchPrediction(
            schema_version=subject.SCHEMA_VERSION,
            method_key=self.method_key,
            method_id=self.method_id,
            pair_ids=tuple(batch.pair_ids),
            pair_probability=np.asarray([0.9, 0.8], dtype=np.float32),
            decision_valid=np.asarray([True, True], dtype=np.bool_),
            translation_hat_rc=translation,
            translation_valid=translation_valid,
            correspondence_indices=(
                np.asarray([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64),
                np.asarray([[0, 0]], dtype=np.int64),
            ),
            correspondence_scores=(
                np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
                np.asarray([0.5], dtype=np.float32),
            ),
            correspondence_semantics="fixture_sparse_threshold",
            auxiliary_scores={},
        )


def _manifest():
    return (
        subject.BenchmarkEvalManifestRow(
            "positive", True, "cluster-positive", ("unit-a",)
        ),
        subject.BenchmarkEvalManifestRow(
            "negative", False, "cluster-negative", ("unit-b", "unit-c")
        ),
    )


def test_synthetic_common_metrics_are_direct_and_single_forward():
    frozen = _FakeFrozen()
    report, records = subject.evaluate_frozen_benchmark_synthetic(
        frozen, [_synthetic_batch()], _manifest()
    )
    assert frozen.calls == 1
    assert len(records) == 2
    assert report["single_forward_population_contract"] == {
        "forward_pair_count": 2,
        "unique_pair_count": 2,
        "each_pair_forwarded_exactly_once": True,
        "forward_batch_count": 1,
        "elapsed_seconds": report["single_forward_population_contract"][
            "elapsed_seconds"
        ],
    }
    assert report["translation"]["te_px_conditioned_on_valid_pose"]["median"] == 0.0
    assert report["translation"]["te_px_conditioned_on_valid_pose"]["p90"] == 0.0
    assert report["translation"]["unconditional_positive_recall"] == {
        "at_2px": 1.0,
        "at_5px": 1.0,
        "at_8px": 1.0,
        "at_10px": 1.0,
    }
    assembly = report["assembly_edge"]["by_tolerance"]["at_2px"]
    assert assembly["true_positive_count"] == 1
    assert assembly["false_positive_count"] == 1
    assert assembly["false_negative_count"] == 0
    assert assembly["precision"] == 0.5
    assert assembly["recall"] == 1.0
    pairing = report["pairingnet_style_registration"]
    assert pairing["rr_lt4"] == 1.0
    assert pairing["mean_e_rmse"] == 0.0
    assert pairing["mean_symmetric_hausdorff_px"] == 0.0
    assert pairing["mean_normalized_translation_error"] == 0.0
    correspondence = report["correspondence"]
    assert correspondence["status"] == "reported"
    assert correspondence["thresholded_exact"]["true_positive_count"] == 4
    assert correspondence["thresholded_exact"]["predicted_count"] == 5
    assert correspondence["thresholded_exact"]["target_count"] == 4
    assert correspondence["thresholded_exact"]["precision"] == 0.8
    assert correspondence["dustbin_aware"]["status"] == "not_applicable"
    assert report["unavailable_or_not_applicable"]["native_global_assembly_GA"][
        "status"
    ] == "not_applicable"
    assert report["unavailable_or_not_applicable"]["shreddingnet_native_CM_FM_SE"][
        "status"
    ] == "not_reported"


def test_invalid_pose_is_unconditional_translation_failure_and_identity_fallback():
    frozen = _FakeFrozen(valid_pose=False)
    report, records = subject.evaluate_frozen_benchmark_synthetic(
        frozen, [_synthetic_batch()], _manifest()
    )
    assert report["translation"]["valid_prediction_count"] == 0
    assert set(report["translation"]["unconditional_positive_recall"].values()) == {
        0.0
    }
    assembly = report["assembly_edge"]["by_tolerance"]["at_10px"]
    assert assembly["true_positive_count"] == 0
    assert assembly["false_positive_count"] == 2
    assert assembly["false_negative_count"] == 1
    registration = records[0]["geometry"]["pairingnet_style_registration"]
    assert registration["identity_fallback_used"] is True
    assert registration["translation_l2_px"] == 5.0
    assert report["pairingnet_style_registration"]["identity_fallback_count"] == 1


def test_synthetic_evaluator_rejects_prediction_order_change():
    class WrongOrder(_FakeFrozen):
        def predict_batch(self, batch, *, return_correspondence=False):
            value = super().predict_batch(
                batch, return_correspondence=return_correspondence
            )
            object.__setattr__(value, "pair_ids", tuple(reversed(value.pair_ids)))
            return value

    with pytest.raises(subject.RachelBenchmarkEvalAdapterError, match="order"):
        subject.evaluate_frozen_benchmark_synthetic(
            WrongOrder(), [_synthetic_batch()], _manifest()
        )


def test_target_blind_batch_contains_only_fixed_non_authoritative_sentinels():
    source = _synthetic_batch()
    batch = subject.build_target_blind_rachel_batch(
        pair_ids=source.pair_ids,
        fragment_a_tokens=source.fragment_a_tokens,
        fragment_b_tokens=source.fragment_b_tokens,
        masks_a=[value[0].astype(np.bool_) for value in source.mask_a],
        masks_b=[value[0].astype(np.bool_) for value in source.mask_b],
        points_rc_a=list(source.points_rc_a),
        points_rc_b=list(source.points_rc_b),
        contour_valid_a=list(source.contour_valid_a),
        contour_valid_b=list(source.contour_valid_b),
    )
    subject._validate_model_input_batch(batch)
    assert batch.mask_a.shape == (2, 1, 800, 800)
    assert batch.coarse_mask_a.shape == (2, 1, 128, 128)
    assert not bool(batch.labels.any())
    assert not bool(batch.translation_valid.any())
    assert not bool(batch.translation_a_to_b_rc.any())
    assert np.all(batch.target_a[batch.contour_valid_a] == -1)
    assert np.all(batch.target_a[~batch.contour_valid_a] == -2)

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

import staging.pairwise_v0_2.baselines.rachel_n512_real_external as external
from staging.pairwise_v0_2.baselines.rachel_n512_real_external import (
    FrozenRachelWinner,
    PreparedRealFragment,
    PreparedStrictRealPopulation,
    RachelRealExternalError,
    StrictRealPairInput,
    TargetBlindPrediction,
    build_balanced_1016_population,
    freeze_completed_winners,
    prepare_case_alpha_masks,
    score_target_blind,
    summarize_strict_real_predictions,
    write_strict_real_evaluation,
)
from staging.pairwise_v0_2.models.rachel_n512 import (
    RachelN512Config,
    RachelN512Pairwise,
)
from staging.pairwise_v0_2.training.evaluation import PairwiseThresholdArtifact
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig


def _canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _small_config():
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
        validate_runtime_inputs=False,
    )


def _threshold(checkpoint_sha, config, arm):
    return PairwiseThresholdArtifact(
        threshold=0.4,
        fit_method="maximize_cluster_balanced_f1",
        source_split="val",
        validation_fingerprint_sha256="1" * 64,
        checkpoint_sha256=checkpoint_sha,
        model_config_sha256=_canonical_sha(asdict(config)),
        aggregation_config_sha256=_canonical_sha(
            {"pair_score": "coarse" if arm == "coarse_only" else "fused"}
        ),
        sample_count=20,
        cluster_count=4,
        achieved_cluster_balanced_f1=0.8,
        achieved_cluster_balanced_precision=0.8,
        achieved_cluster_balanced_recall=0.8,
    )


def test_completed_winner_and_validation_threshold_freeze_strictly(
    tmp_path, monkeypatch
):
    config = _small_config()
    loss_config = RachelN512LossConfig(
        validate_runtime_targets=False,
        collect_cpu_diagnostics=False,
    )
    monkeypatch.setattr(
        external.sealed_authority,
        "_canonical_rachel_model_config",
        lambda: config,
    )
    monkeypatch.setattr(
        external.sealed_authority,
        "_canonical_rachel_loss_config",
        lambda: loss_config,
    )
    run_config = {
        "dataset_root": str((tmp_path / "release").resolve()),
        "arms": ["coarse_only"],
        "precision": "fp32",
        "seed": 7,
    }
    torch.manual_seed(7)
    model = RachelN512Pairwise(config).coarse
    checkpoint_path = tmp_path / "coarse_only" / "epoch-001.pt"
    checkpoint_path.parent.mkdir()
    torch.save(
        {
            "schema_version": "rachel-n512-checkpoint/1.0",
            "arm": "coarse_only",
            "epoch": 1,
            "model_config": asdict(config),
            "loss_config": asdict(loss_config),
            "model_state_dict": model.state_dict(),
            "run_config": run_config,
            "test_accessed": False,
            "real_external_test_accessed": False,
        },
        checkpoint_path,
    )
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    threshold = _threshold(checkpoint_sha, config, "coarse_only")
    (tmp_path / "run_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "rachel-n512-train-run/1.0",
                "status": "complete_train_validation_only",
                "config": run_config,
                "arm_results": [
                    {
                        "arm": "coarse_only",
                        "winner_epoch": 1,
                        "winner_checkpoint": "coarse_only/epoch-001.pt",
                        "winner_checkpoint_sha256": checkpoint_sha,
                        "validation_threshold": threshold.to_dict(),
                        "epochs": [
                            {
                                "epoch": 1,
                                "checkpoint": "coarse_only/epoch-001.pt",
                                "checkpoint_sha256": checkpoint_sha,
                            }
                        ],
                        "test_accessed": False,
                        "real_external_test_accessed": False,
                    }
                ],
                "test_accessed": False,
                "real_external_test_accessed": False,
            }
        ),
        encoding="utf-8",
    )

    # The deployment runtime is torch 2.5.1, whose restricted unpickler reads
    # primitive float metadata.  The local torch 2.0 restricted unpickler does
    # not support BINFLOAT; wrap it only to assert the production call remains
    # weights_only=True while letting this old test runtime decode the fixture.
    original_load = torch.load
    observed = {}

    def compatible_load(path, *, map_location, weights_only=None):
        observed["weights_only"] = weights_only
        return original_load(path, map_location=map_location, weights_only=False)

    monkeypatch.setattr(torch, "load", compatible_load)
    winner = freeze_completed_winners(tmp_path)["coarse_only"]
    expected_weights_only = (
        True if int(torch.__version__.split(".", 1)[0]) >= 2 else None
    )
    assert observed == {"weights_only": expected_weights_only}
    assert winner.checkpoint_sha256 == checkpoint_sha
    assert winner.threshold.threshold == 0.4
    assert not winner.model.training


def test_winner_freeze_rejects_threshold_bound_to_another_checkpoint(tmp_path):
    (tmp_path / "run_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "rachel-n512-train-run/1.0",
                "status": "complete_train_validation_only",
                "arm_results": [],
                "test_accessed": False,
                "real_external_test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RachelRealExternalError, match="authority freeze failed"):
        freeze_completed_winners(tmp_path)


def test_case_common_scale_is_shared_before_tight_crop_and_centerpad():
    first = np.zeros((20, 10), dtype=np.bool_)
    first[2:18, 1:9] = True
    second = np.zeros((12, 24), dtype=np.bool_)
    second[1:11, 2:22] = True
    prepared = prepare_case_alpha_masks(
        {"case/fragment/1": first, "case/fragment/2": second},
        case_uid="case",
        canvas_wh=(64, 32),
        canvas_size=32,
        contour_cap=8,
    )
    a = prepared["case/fragment/1"]
    b = prepared["case/fragment/2"]
    assert a.shared_case_scale == b.shared_case_scale == 0.5
    assert a.tight_crop_hw == (8, 4)
    assert b.tight_crop_hw == (5, 10)
    for value in (a, b):
        assert value.mask.shape == (32, 32)
        assert value.points_rc.shape == (8, 2)
        assert value.contour_valid.shape == (8,)
        rows, columns = np.nonzero(value.mask)
        assert abs((rows.min() + rows.max()) - 31) <= 1
        assert abs((columns.min() + columns.max()) - 31) <= 1


def test_case_common_scale_can_upscale_without_fragment_specific_resize():
    first = np.zeros((4, 6), dtype=np.bool_)
    first[1:3, 1:5] = True
    second = np.zeros((6, 3), dtype=np.bool_)
    second[1:5, 1:2] = True
    prepared = prepare_case_alpha_masks(
        {"case/fragment/1": first, "case/fragment/2": second},
        case_uid="case",
        canvas_wh=(16, 8),
        canvas_size=32,
        contour_cap=8,
    )
    assert {value.shared_case_scale for value in prepared.values()} == {2.0}
    assert prepared["case/fragment/1"].tight_crop_hw == (4, 8)
    assert prepared["case/fragment/2"].tight_crop_hw == (8, 2)


class _FakeCoarse(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen = None

    def forward(self, first, second):
        self.seen = (tuple(first.shape), tuple(second.shape))
        count = first.shape[0]
        return SimpleNamespace(
            probability=torch.full((count,), 0.7, device=first.device),
            valid_problem=torch.ones((count,), dtype=torch.bool, device=first.device),
        )


class _FakeFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.argument_count = None

    def forward(self, *arguments):
        self.argument_count = len(arguments)
        count = arguments[0].shape[0]
        device = arguments[0].device
        return SimpleNamespace(
            fused_probability=torch.full((count,), 0.6, device=device),
            decision_valid=torch.ones((count,), dtype=torch.bool, device=device),
            coarse_probability=torch.full((count,), 0.55, device=device),
            coarse=SimpleNamespace(
                valid_problem=torch.ones((count,), dtype=torch.bool, device=device)
            ),
            translation_hat_rc=torch.tensor([[1.0, -2.0]], device=device).repeat(
                count, 1
            ),
            translation_dispersion_px=torch.full((count,), 3.0, device=device),
        )


def _prepared_fragment_stub(fragment_id: str, case_uid: str) -> PreparedRealFragment:
    mask = np.zeros((32, 32), dtype=np.bool_)
    mask[8:24, 8:24] = True
    return PreparedRealFragment(
        fragment_id=fragment_id,
        case_uid=case_uid,
        mask=mask,
        points_rc=np.zeros((8, 2), dtype=np.float32),
        contour_valid=np.ones((8,), dtype=np.bool_),
        shared_case_scale=1.0,
        tight_crop_hw=(16, 16),
        alpha_sha256=hashlib.sha256(fragment_id.encode()).hexdigest(),
        selection_height=16,
        selection_width=16,
        selection_foreground_area=256,
    )


@pytest.mark.parametrize("arm", ("coarse_only", "full_n512"))
def test_inference_path_is_target_blind_and_emits_unsupervised_translation(arm):
    mask_a = np.zeros((12, 12), dtype=np.bool_)
    mask_a[2:10, 2:8] = True
    mask_b = np.zeros((12, 12), dtype=np.bool_)
    mask_b[1:9, 4:11] = True
    fragments = prepare_case_alpha_masks(
        {"case/fragment/1": mask_a, "case/fragment/2": mask_b},
        case_uid="case",
        canvas_wh=(32, 32),
        canvas_size=32,
        contour_cap=8,
    )
    model = _FakeCoarse() if arm == "coarse_only" else _FakeFull()
    config = _small_config()
    winner = FrozenRachelWinner(
        arm=arm,
        epoch=1,
        checkpoint_path=Path("unused"),
        checkpoint_sha256="2" * 64,
        threshold=_threshold("2" * 64, config, arm),
        model_config=config,
        model=model,
    )
    pairs = (
        StrictRealPairInput(
            "pair-1", "case", "case/fragment/1", "case/fragment/2"
        ),
    )
    prediction = score_target_blind(winner, pairs, fragments)[0]
    assert prediction.pair_id == "pair-1"
    assert prediction.valid
    if arm == "coarse_only":
        assert model.seen == ((1, 1, 16, 16), (1, 1, 16, 16))
        assert prediction.translation_hat_rc is None
    else:
        assert model.argument_count == 6
        assert prediction.translation_hat_rc == (1.0, -2.0)
        assert prediction.translation_dispersion_px == 3.0


def test_summary_separates_native_common_ranking_and_frozen_threshold(tmp_path):
    config = _small_config()
    models = {"coarse_only": _FakeCoarse(), "full_n512": _FakeFull()}
    winners = {
        arm: FrozenRachelWinner(
            arm=arm,
            epoch=1,
            checkpoint_path=Path("unused"),
            checkpoint_sha256=("2" if arm == "coarse_only" else "3") * 64,
            threshold=_threshold(
                ("2" if arm == "coarse_only" else "3") * 64, config, arm
            ),
            model_config=config,
            model=models[arm],
        )
        for arm in ("coarse_only", "full_n512")
    }
    pair_inputs = tuple(
        StrictRealPairInput(
            "pair-{}".format(index),
            "case-{}".format(index // 2),
            "a-{}".format(index),
            "b-{}".format(index),
        )
        for index in range(4)
    )
    fragments = {
        fragment_id: _prepared_fragment_stub(fragment_id, "case-{}".format(index // 2))
        for index in range(4)
        for fragment_id in ("a-{}".format(index), "b-{}".format(index))
    }
    population = PreparedStrictRealPopulation(
        manifest_sha256="4" * 64,
        fragments=fragments,
        pair_inputs=pair_inputs,
        labels=(True, True, False, False),
        case_clusters=("case-0", "case-0", "case-1", "case-1"),
    )

    def predictions(probability, validity, translation):
        return tuple(
            TargetBlindPrediction(
                pair_id="pair-{}".format(index),
                probability=value,
                valid=validity[index],
                coarse_probability=value,
                coarse_valid=True,
                translation_hat_rc=(1.0, 2.0) if translation else None,
                translation_dispersion_px=3.0 if translation else None,
            )
            for index, value in enumerate(probability)
        )

    predictions_by_arm = {
        "coarse_only": predictions((0.9, 0.8, 0.2, 0.1), (True,) * 4, False),
        "full_n512": predictions(
            (0.9, 0.85, 0.25, 0.15), (False, True, True, True), True
        ),
    }
    result = summarize_strict_real_predictions(
        population, winners, predictions_by_arm
    )
    coarse = result["methods"]["coarse_only"]
    full = result["methods"]["full_n512"]
    assert coarse["native"]["coverage"]["valid_fraction"] == 1.0
    assert full["native"]["coverage"]["valid_fraction"] == 0.75
    assert (
        coarse["coarse_full_common_valid"]["coverage"]["valid_fraction"]
        == 0.75
    )
    assert coarse["native"]["ranking_primary_threshold_free"]["row"]["auroc"] == 1.0
    assert (
        coarse["native"]["frozen_validation_threshold_secondary"]["threshold"]
        == 0.4
    )
    diagnostic = full["translation_unsupervised_diagnostic"]
    assert diagnostic["used_for_accuracy_or_model_selection"] is False
    assert diagnostic["translation_gt_read_or_reported_here"] is False
    assert diagnostic["translation_gt_availability_claim"] == "none"
    assert diagnostic["translation_gt_evaluation"] == (
        "separate_independent_post_prediction_real_translation_evaluator"
    )
    assert result["pairs"][0]["source_case_uids"] == ["case-0"]

    output = tmp_path / "result.json"
    write_strict_real_evaluation(output, result)
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored["dataset"]["pair_count"] == 4
    assert not (tmp_path / ".result.json.tmp").exists()
    with pytest.raises(RachelRealExternalError, match="refusing to overwrite"):
        write_strict_real_evaluation(output, result)


def test_real_writer_rejects_training_or_authority_root(tmp_path):
    training_root = tmp_path / "training-run"
    training_root.mkdir()
    with pytest.raises(RachelRealExternalError, match="authority roots"):
        write_strict_real_evaluation(
            training_root / "result.json",
            {"status": "unused"},
            forbidden_roots=(training_root,),
        )
    assert not (training_root / "result.json").exists()


def test_strict_alpha_is_bound_to_declared_bool_mask_sha256(tmp_path):
    alpha = np.zeros((6, 8), dtype=np.uint8)
    alpha[1:5, 2:7] = 255
    source = tmp_path / "fragment.png"
    value = np.zeros((6, 8, 4), dtype=np.uint8)
    value[..., 3] = alpha
    Image.fromarray(value, mode="RGBA").save(source)
    mask = np.ascontiguousarray(alpha >= external.ALPHA_THRESHOLD, dtype=np.bool_)
    fragment = external.RealFragmentSpec(
        case_uid="case",
        fragment_id=1,
        collection="main",
        category="small",
        source_path=source,
        relative_path="case/1.png",
        canvas_wh=(8, 6),
        expected_size_wh=(8, 6),
        declared_alpha_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
    )
    assert np.array_equal(external._load_strict_alpha(fragment), mask)
    broken = external.RealFragmentSpec(
        **{**asdict(fragment), "declared_alpha_mask_sha256": "0" * 64}
    )
    with pytest.raises(RachelRealExternalError, match="SHA-256 differs"):
        external._load_strict_alpha(broken)


def test_real_pair_only_preparation_explicitly_disables_direction_derivation(
    tmp_path, monkeypatch
):
    observed = {}

    def fake_loader(*args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(fragments=(), pairs=(), manifest_sha256="4" * 64)

    monkeypatch.setattr(external, "load_real_external_test_spec", fake_loader)
    population = external.prepare_strict_real_population(
        tmp_path / "manifest.json", tmp_path / "paths.json"
    )
    assert observed["derive_positive_direction"] is False
    assert population.pair_inputs == ()


def test_dependency_free_balanced_builder_is_deterministic_and_preserves_prefix(
    monkeypatch,
):
    mask = np.zeros((32, 32), dtype=np.bool_)
    mask[8:24, 8:24] = True
    points = np.zeros((8, 2), dtype=np.float32)
    valid = np.ones((8,), dtype=np.bool_)
    fragments = {}
    for index in range(8):
        fragment_id = "case-{0}/fragment/{0}".format(index)
        fragments[fragment_id] = PreparedRealFragment(
                fragment_id=fragment_id,
                case_uid="case-{}".format(index // 2),
            mask=mask,
            points_rc=points,
            contour_valid=valid,
            shared_case_scale=1.0,
            tight_crop_hw=(16, 16),
            alpha_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
            selection_height=24 + index % 3,
            selection_width=22 + index % 4,
            selection_foreground_area=400 + index * 3,
        )
    pair_inputs = tuple(
        StrictRealPairInput(
            "strict-pair-{}".format(index),
            "strict-case-{}".format(index),
            "case-{0}/fragment/{0}".format(index * 2),
            "case-{0}/fragment/{0}".format(index * 2 + 1),
        )
        for index in range(4)
    )
    strict = PreparedStrictRealPopulation(
        manifest_sha256="5" * 64,
        fragments=fragments,
        pair_inputs=pair_inputs,
        labels=(True,) * 4,
        case_clusters=tuple(row.case_uid for row in pair_inputs),
        strata=("strict_manifest_positive",) * 4,
    )
    monkeypatch.setattr(external, "EXPECTED_STRICT_PAIR_COUNT", 4)
    monkeypatch.setattr(external, "EXPECTED_STRICT_POSITIVE_COUNT", 4)
    monkeypatch.setattr(external, "EXPECTED_STRICT_NEGATIVE_COUNT", 0)
    monkeypatch.setattr(external, "EXPECTED_REAL_FRAGMENT_COUNT", 8)
    monkeypatch.setattr(external, "EXPECTED_CONSTRUCTED_COUNT", 4)
    monkeypatch.setattr(external, "EXPECTED_BALANCED_PAIR_COUNT", 8)
    monkeypatch.setattr(external, "EXPECTED_BALANCED_NEGATIVE_COUNT", 4)
    monkeypatch.setattr(external, "EXPECTED_CONSTRUCTED_SELECTION_SHA256", "0" * 64)
    with pytest.raises(RachelRealExternalError, match="differs") as caught:
        build_balanced_1016_population(strict)
    observed_sha = str(caught.value).rsplit(" ", 1)[-1]
    assert len(observed_sha) == 64
    monkeypatch.setattr(
        external, "EXPECTED_CONSTRUCTED_SELECTION_SHA256", observed_sha
    )
    first = build_balanced_1016_population(strict)
    second = build_balanced_1016_population(strict)
    assert first.pair_inputs[:4] == strict.pair_inputs
    assert first.pair_inputs == second.pair_inputs
    assert first.labels == (True,) * 4 + (False,) * 4
    assert set(first.strata[4:]) == {"constructed_not_GT_negative"}
    constructed = first.pair_inputs[4:]
    assert len(
        {
            tuple(sorted((fragments[row.fragment_a_id].case_uid, fragments[row.fragment_b_id].case_uid)))
            for row in constructed
        }
    ) == 4
    assert Counter(
        fragment_id
        for row in constructed
        for fragment_id in (row.fragment_a_id, row.fragment_b_id)
    ) == Counter({fragment_id: 1 for fragment_id in fragments})
    config = _small_config()
    winners = {
        arm: FrozenRachelWinner(
            arm=arm,
            epoch=20,
            checkpoint_path=Path("unused"),
            checkpoint_sha256=("2" if arm == "coarse_only" else "3") * 64,
            threshold=_threshold(
                ("2" if arm == "coarse_only" else "3") * 64, config, arm
            ),
            model_config=config,
            model=_FakeCoarse() if arm == "coarse_only" else _FakeFull(),
        )
        for arm in external.SUPPORTED_ARMS
    }
    predictions = {
        arm: tuple(
            TargetBlindPrediction(
                pair_id=row.pair_id,
                probability=0.9 if first.labels[index] else 0.1,
                valid=True,
                coarse_probability=0.9 if first.labels[index] else 0.1,
                coarse_valid=True,
                translation_hat_rc=(0.0, 0.0) if arm == "full_n512" else None,
                translation_dispersion_px=0.0 if arm == "full_n512" else None,
            )
            for index, row in enumerate(first.pair_inputs)
        )
        for arm in external.SUPPORTED_ARMS
    }
    summarized = summarize_strict_real_predictions(first, winners, predictions)
    assert len(summarized["pairs"][0]["source_case_uids"]) == 1
    assert all(
        len(row["source_case_uids"]) == 2
        for row in summarized["pairs"][4:]
    )


def _matched_threshold(checkpoint_sha, model_config, value):
    return PairwiseThresholdArtifact(
        threshold=value,
        fit_method="maximize_cluster_balanced_f1",
        source_split="val",
        validation_fingerprint_sha256="1" * 64,
        checkpoint_sha256=checkpoint_sha,
        model_config_sha256=_canonical_sha(model_config),
        aggregation_config_sha256=_canonical_sha(
            {"pair_score": external.MATCHED_SCORE}
        ),
        sample_count=3_000,
        cluster_count=100,
        achieved_cluster_balanced_f1=0.8,
        achieved_cluster_balanced_precision=0.8,
        achieved_cluster_balanced_recall=0.8,
    )


def _matched_winners():
    model_config = {"architecture": "HistoricalMMSiamese"}
    return {
        method: external.FrozenMatchedMMWinner(
            method=method,
            epoch=20 if index == 0 else 5,
            checkpoint_path=Path("unused-{}.pt".format(index)),
            checkpoint_sha256=("6" if index == 0 else "7") * 64,
            model_config=model_config,
            model_config_sha256=_canonical_sha(model_config),
            threshold=_matched_threshold(
                ("6" if index == 0 else "7") * 64,
                model_config,
                0.55 if index == 0 else 0.45,
            ),
            model=_FakeCoarse(),
        )
        for index, method in enumerate(external.MATCHED_METHODS)
    }


def test_matched_real_augmentation_keeps_ranking_primary_and_threshold_secondary():
    config = _small_config()
    n512_winners = {
        arm: FrozenRachelWinner(
            arm=arm,
            epoch=1,
            checkpoint_path=Path("unused"),
            checkpoint_sha256=("2" if arm == "coarse_only" else "3") * 64,
            threshold=_threshold(
                ("2" if arm == "coarse_only" else "3") * 64, config, arm
            ),
            model_config=config,
            model=_FakeCoarse() if arm == "coarse_only" else _FakeFull(),
        )
        for arm in external.SUPPORTED_ARMS
    }
    population = PreparedStrictRealPopulation(
        manifest_sha256="4" * 64,
        fragments={
            fragment_id: _prepared_fragment_stub(
                fragment_id, "case-{}".format(index // 2)
            )
            for index in range(4)
            for fragment_id in ("a-{}".format(index), "b-{}".format(index))
        },
        pair_inputs=tuple(
            StrictRealPairInput(
                "pair-{}".format(index),
                "case-{}".format(index // 2),
                "a-{}".format(index),
                "b-{}".format(index),
            )
            for index in range(4)
        ),
        labels=(True, True, False, False),
        case_clusters=("case-0", "case-0", "case-1", "case-1"),
    )

    def n512_predictions(values, full):
        return tuple(
            TargetBlindPrediction(
                pair_id="pair-{}".format(index),
                probability=value,
                valid=True,
                coarse_probability=value,
                coarse_valid=True,
                translation_hat_rc=(1.0, 2.0) if full else None,
                translation_dispersion_px=3.0 if full else None,
            )
            for index, value in enumerate(values)
        )

    base = summarize_strict_real_predictions(
        population,
        n512_winners,
        {
            "coarse_only": n512_predictions((0.8, 0.7, 0.3, 0.2), False),
            "full_n512": n512_predictions((0.9, 0.8, 0.2, 0.1), True),
        },
    )
    matched_winners = _matched_winners()
    matched_predictions = {
        method: tuple(
            external.MatchedMMPrediction(
                pair_id="pair-{}".format(index), probability=value
            )
            for index, value in enumerate(
                (0.85, 0.75, 0.25, 0.15)
                if method == external.MATCHED_METHODS[0]
                else (0.7, 0.6, 0.4, 0.3)
            )
        )
        for method in external.MATCHED_METHODS
    }
    result = external._augment_with_matched_mm(
        base,
        population,
        matched_winners,
        matched_predictions,
        matched_run_directory=Path("/frozen/matched"),
        matched_receipt={"status": "complete_train_validation_only"},
        matched_receipt_sha256="8" * 64,
    )
    assert set(result["methods"]) == set(external.SUPPORTED_ARMS) | set(
        external.MATCHED_METHODS
    )
    converged = result["methods"][external.MATCHED_METHODS[0]]
    assert converged["score_semantics"] == "historical_mm_probability"
    assert converged["native"]["ranking_primary_threshold_free"]["row"]["auroc"] == 1.0
    assert (
        converged["native"]["frozen_validation_threshold_secondary"]["threshold"]
        == 0.55
    )
    assert result["all_method_common_valid"]["valid_fraction"] == 1.0
    assert set(result["pairs"][0]["methods"]) == set(result["methods"])
    disclosure = result["protocol"]["evaluation_history_disclosure"]
    assert disclosure["prior_epoch5_real_result_formed_or_read"] is False
    assert disclosure["claim_of_project_first_real_access"] is False
    assert disclosure[
        "prior_epoch5_synthetic_test_human_visible_before_continuation"
    ] is True
    assert disclosure["claim_no_human_cognitive_influence"] is False
    assert "prior_epoch5_synthetic_test_used_for_continuation_or_selection" not in disclosure
    assert "prior_epoch5_real_activity_used_for_training_or_selection" not in disclosure


def test_combined_real_freezes_matched_before_opening_real_population(
    tmp_path, monkeypatch
):
    config = _small_config()
    n512_winners = {
        arm: FrozenRachelWinner(
            arm=arm,
            epoch=1,
            checkpoint_path=Path("unused"),
            checkpoint_sha256=("2" if arm == "coarse_only" else "3") * 64,
            threshold=_threshold(
                ("2" if arm == "coarse_only" else "3") * 64, config, arm
            ),
            model_config=config,
            model=_FakeCoarse() if arm == "coarse_only" else _FakeFull(),
        )
        for arm in external.SUPPORTED_ARMS
    }
    monkeypatch.setattr(
        external,
        "_freeze_completed_n512_authority",
        lambda *a, **k: (
            {"status": "complete_train_validation_only"},
            "9" * 64,
            n512_winners,
        ),
    )
    monkeypatch.setattr(
        external,
        "_require_formal_n512_convergence",
        lambda *a, **k: {
            "config": {"dataset_root": "/same", "seed": 260831}
        },
    )
    monkeypatch.setattr(
        external,
        "freeze_matched_mm_winners",
        lambda *a, **k: (_ for _ in ()).throw(
            RachelRealExternalError("matched freeze failed")
        ),
    )
    opened = False

    def forbidden_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("real population opened before all freezes")

    monkeypatch.setattr(external, "prepare_strict_real_population", forbidden_open)
    matched_run = tmp_path / "matched"
    matched_run.mkdir()
    with pytest.raises(RachelRealExternalError, match="matched freeze failed"):
        external.evaluate_strict_real_external(
            tmp_path,
            tmp_path / "must-not-open.json",
            tmp_path / "must-not-open-receipt.json",
            matched_mm_run_directory=matched_run,
            device="cpu",
            compatibility_mode=True,
        )
    assert opened is False


def test_real_requires_formal_plateau_even_without_matched_control(
    tmp_path, monkeypatch
):
    config = _small_config()
    winners = {
        arm: FrozenRachelWinner(
            arm=arm,
            epoch=1,
            checkpoint_path=Path("unused"),
            checkpoint_sha256=("2" if arm == "coarse_only" else "3") * 64,
            threshold=_threshold(
                ("2" if arm == "coarse_only" else "3") * 64, config, arm
            ),
            model_config=config,
            model=_FakeCoarse() if arm == "coarse_only" else _FakeFull(),
        )
        for arm in external.SUPPORTED_ARMS
    }
    monkeypatch.setattr(
        external,
        "_freeze_completed_n512_authority",
        lambda *a, **k: (
            {"status": "complete_train_validation_only"},
            "9" * 64,
            winners,
        ),
    )
    monkeypatch.setattr(
        external,
        "_require_formal_n512_convergence",
        lambda *a, **k: (_ for _ in ()).throw(
            RachelRealExternalError("validation convergence plateau missing")
        ),
    )
    opened = False

    def forbidden_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("real authority opened")

    monkeypatch.setattr(external, "prepare_strict_real_population", forbidden_open)
    with pytest.raises(RachelRealExternalError, match="plateau missing"):
        external.evaluate_strict_real_external(
            tmp_path,
            tmp_path / "manifest.json",
            tmp_path / "paths.json",
            device="cpu",
        )
    assert opened is False


@pytest.mark.parametrize(
    ("authority_kwargs", "include_balanced", "message"),
    [
        ({}, True, "requires exact-six frozen authorities"),
        (
            {
                "matched_mm_run_directory": Path("matched"),
                "pairingnet_run_directory": Path("pairing"),
                "shreddingnet_freeze_path": Path("shredding.json"),
            },
            False,
            "requires one balanced1016 forward population",
        ),
    ],
)
def test_formal_real_gate_rejects_before_any_real_path_open(
    tmp_path, monkeypatch, authority_kwargs, include_balanced, message
):
    config = _small_config()
    winners = {
        arm: FrozenRachelWinner(
            arm=arm,
            epoch=20,
            checkpoint_path=Path("unused"),
            checkpoint_sha256=("2" if arm == "coarse_only" else "3") * 64,
            threshold=_threshold(
                ("2" if arm == "coarse_only" else "3") * 64, config, arm
            ),
            model_config=config,
            model=_FakeCoarse() if arm == "coarse_only" else _FakeFull(),
        )
        for arm in external.SUPPORTED_ARMS
    }
    monkeypatch.setattr(
        external,
        "_freeze_completed_n512_authority",
        lambda *a, **k: (
            {"status": "complete_train_validation_only"},
            "9" * 64,
            winners,
        ),
    )
    monkeypatch.setattr(
        external,
        "_require_formal_n512_convergence",
        lambda *a, **k: {"status": "complete_train_validation_only"},
    )
    opened = []

    def forbidden(*args, **kwargs):
        opened.append(True)
        raise AssertionError("real path opened before formal-six gate")

    monkeypatch.setattr(external, "prepare_strict_real_population", forbidden)
    with pytest.raises(RachelRealExternalError, match=message):
        external.evaluate_strict_real_external(
            tmp_path,
            tmp_path / "manifest-must-not-open.json",
            tmp_path / "receipt-must-not-open.json",
            include_balanced_1016=include_balanced,
            **authority_kwargs,
        )
    assert opened == []


def test_explicit_real_compatibility_receipt_cannot_claim_formal_status():
    stamped = external._stamp_real_evaluation_mode(
        {
            "schema_version": external.SCHEMA_VERSION,
            "status": "complete_strict_547_target_blind_external_test",
            "protocol": {},
        },
        formal_evaluation=False,
    )
    assert stamped["status"] == external.COMPATIBILITY_STRICT_STATUS
    assert stamped["protocol"]["formal_evaluation"] is False
    assert stamped["protocol"]["compatibility_mode"] is True
    assert stamped["protocol"]["formal_method_inventory"] is None


def test_real_cli_defaults_to_formal_and_compatibility_is_explicit(tmp_path):
    arguments = [
        "--run-directory",
        str(tmp_path / "run"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--local-path-receipt",
        str(tmp_path / "paths.json"),
        "--output",
        str(tmp_path / "result.json"),
    ]
    assert external._parser().parse_args(arguments).compatibility_non_formal is False
    assert (
        external._parser()
        .parse_args(arguments + ["--compatibility-non-formal"])
        .compatibility_non_formal
        is True
    )


def test_combined_real_rejects_matched_formal_config_before_real_open(
    tmp_path, monkeypatch
):
    config = _small_config()
    winners = {
        arm: FrozenRachelWinner(
            arm=arm,
            epoch=20,
            checkpoint_path=Path("unused"),
            checkpoint_sha256=("2" if arm == "coarse_only" else "3") * 64,
            threshold=_threshold(
                ("2" if arm == "coarse_only" else "3") * 64, config, arm
            ),
            model_config=config,
            model=_FakeCoarse() if arm == "coarse_only" else _FakeFull(),
        )
        for arm in external.SUPPORTED_ARMS
    }
    n512_receipt = {"status": "complete_train_validation_only"}
    monkeypatch.setattr(
        external,
        "_freeze_completed_n512_authority",
        lambda *a, **k: (n512_receipt, "9" * 64, winners),
    )
    monkeypatch.setattr(
        external,
        "_require_formal_n512_convergence",
        lambda *a, **k: n512_receipt,
    )
    monkeypatch.setattr(
        external,
        "freeze_matched_mm_winners",
        lambda *a, **k: (
            {"status": "complete_train_validation_only"},
            "8" * 64,
            _matched_winners(),
        ),
    )
    monkeypatch.setattr(
        external.sealed_authority,
        "_require_matched_training_alignment",
        lambda *a, **k: (_ for _ in ()).throw(
            external.sealed_authority.RachelN512SealedTestError(
                "matched-MM formal validation-only convergence config differs"
            )
        ),
    )
    opened = False

    def forbidden_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("real authority opened")

    monkeypatch.setattr(external, "prepare_strict_real_population", forbidden_open)
    matched_run = tmp_path / "matched"
    matched_run.mkdir()
    with pytest.raises(RachelRealExternalError, match="formal validation-only"):
        external.evaluate_strict_real_external(
            tmp_path,
            tmp_path / "manifest.json",
            tmp_path / "paths.json",
            matched_mm_run_directory=matched_run,
            device="cpu",
            compatibility_mode=True,
        )
    assert opened is False


def test_balanced_combined_flow_forwards_once_and_slices_exact_strict_prefix(
    tmp_path, monkeypatch
):
    config = _small_config()
    n512_winners = {
        arm: FrozenRachelWinner(
            arm=arm,
            epoch=1,
            checkpoint_path=Path("unused"),
            checkpoint_sha256=("2" if arm == "coarse_only" else "3") * 64,
            threshold=_threshold(
                ("2" if arm == "coarse_only" else "3") * 64, config, arm
            ),
            model_config=config,
            model=_FakeCoarse() if arm == "coarse_only" else _FakeFull(),
        )
        for arm in external.SUPPORTED_ARMS
    }
    matched_winners = _matched_winners()
    strict_inputs = tuple(
        StrictRealPairInput("p{}".format(index), "c", "a", "b")
        for index in range(2)
    )
    balanced_inputs = strict_inputs + tuple(
        StrictRealPairInput("p{}".format(index), "c", "a", "b")
        for index in range(2, 4)
    )
    strict = SimpleNamespace(pair_inputs=strict_inputs, fragments={})
    balanced = SimpleNamespace(pair_inputs=balanced_inputs, fragments={})
    monkeypatch.setattr(
        external,
        "_freeze_completed_n512_authority",
        lambda *a, **k: (
            {"status": "complete_train_validation_only"},
            "9" * 64,
            n512_winners,
        ),
    )
    monkeypatch.setattr(
        external,
        "_require_formal_n512_convergence",
        lambda *a, **k: {
            "config": {"dataset_root": "/same", "seed": 260831}
        },
    )
    monkeypatch.setattr(
        external,
        "freeze_matched_mm_winners",
        lambda *a, **k: (
            {
                "status": "complete_train_validation_only",
                "config": {"dataset_root": "/same", "seed": 260831},
            },
            "8" * 64,
            matched_winners,
        ),
    )
    monkeypatch.setattr(
        external.sealed_authority,
        "_require_matched_training_alignment",
        lambda *a, **k: Path("/same"),
    )
    monkeypatch.setattr(
        external.sealed_authority,
        "_matched_training_hash_evidence",
        lambda *a, **k: {"claim_level": "fixture_hash_evidence"},
    )
    monkeypatch.setattr(external, "prepare_strict_real_population", lambda *a, **k: strict)
    monkeypatch.setattr(external, "build_balanced_1016_population", lambda value: balanced)
    monkeypatch.setattr(external, "EXPECTED_STRICT_PAIR_COUNT", 2)
    monkeypatch.setattr(external, "EXPECTED_BALANCED_PAIR_COUNT", 4)
    calls = {method: [] for method in tuple(external.SUPPORTED_ARMS) + external.MATCHED_METHODS}

    def n512_score(winner, pair_inputs, fragments, *, batch_size):
        calls[winner.arm].append(len(pair_inputs))
        return tuple(
            TargetBlindPrediction(
                pair_id=row.pair_id,
                probability=0.5,
                valid=True,
                coarse_probability=0.5,
                coarse_valid=True,
                translation_hat_rc=None,
                translation_dispersion_px=None,
            )
            for row in pair_inputs
        )

    def matched_score(winner, pair_inputs, fragments, *, batch_size):
        calls[winner.method].append(len(pair_inputs))
        return tuple(
            external.MatchedMMPrediction(row.pair_id, 0.5) for row in pair_inputs
        )

    monkeypatch.setattr(external, "score_target_blind", n512_score)
    monkeypatch.setattr(external, "score_alpha_derived_mask_pairs", matched_score)
    monkeypatch.setattr(
        external,
        "summarize_strict_real_predictions",
        lambda *a, **k: {"kind": "strict"},
    )
    monkeypatch.setattr(
        external,
        "summarize_balanced_1016_predictions",
        lambda *a, **k: {"kind": "balanced"},
    )
    augmented_lengths = []

    def augment(base, population, winners, predictions, **kwargs):
        augmented_lengths.append(
            (base["kind"], {key: len(value) for key, value in predictions.items()})
        )
        return base

    monkeypatch.setattr(external, "_augment_with_matched_mm", augment)
    matched_run = tmp_path / "matched"
    matched_run.mkdir()
    result = external.evaluate_strict_real_external(
        tmp_path,
        tmp_path / "manifest.json",
        tmp_path / "paths.json",
        matched_mm_run_directory=matched_run,
        device="cpu",
        include_balanced_1016=True,
        compatibility_mode=True,
    )
    assert all(value == [4] for value in calls.values())
    assert augmented_lengths[0] == (
        "strict",
        {method: 2 for method in external.MATCHED_METHODS},
    )
    assert augmented_lengths[1] == (
        "balanced",
        {method: 4 for method in external.MATCHED_METHODS},
    )
    assert result["forward_contract"]["strict_pairs_forwarded_twice"] is False
    assert result["forward_contract"][
        "all_methods_share_exact_strict_prediction_prefix"
    ] is True
    assert result["source_training_run"]["receipt_sha256"] == "9" * 64
    assert result["source_training_run"][
        "both_arms_validation_plateau_verified"
    ] is True
    assert result["protocol"]["formal_config_verified"] is True
    assert result["protocol"][
        "all_requested_winners_and_thresholds_frozen_before_current_real_open"
    ] is True


def _benchmark_frozen_fixture(method, threshold=0.5):
    return SimpleNamespace(
        method_key=method,
        method_id="fixture-" + method,
        threshold=threshold,
        threshold_artifact={"threshold": threshold},
        threshold_artifact_sha256="a" * 64,
        checkpoint_sha256_by_stage={"winner": "b" * 64},
        freeze_authority_sha256="c" * 64,
        adaptation_disclosure={
            "mask_only": True,
            "upright_translation_only": True,
        },
        provenance=lambda: {"method_key": method},
        release_to_cpu=lambda: None,
    )


def _benchmark_fragment_fixture(fragment_id):
    mask = np.zeros((800, 800), dtype=np.bool_)
    mask[300:500, 300:500] = True
    points = np.zeros((512, 2), dtype=np.float32)
    points[:4] = np.asarray(
        [[300.0, 300.0], [300.0, 499.0], [499.0, 499.0], [499.0, 300.0]],
        dtype=np.float32,
    )
    valid = np.zeros(512, dtype=np.bool_)
    valid[:4] = True
    return PreparedRealFragment(
        fragment_id=fragment_id,
        case_uid="case",
        mask=mask,
        points_rc=points,
        contour_valid=valid,
        shared_case_scale=1.0,
        tight_crop_hw=(200, 200),
        alpha_sha256=hashlib.sha256(fragment_id.encode()).hexdigest(),
        selection_height=200,
        selection_width=200,
        selection_foreground_area=40_000,
    )


def test_same_data_benchmark_real_bridge_is_target_blind_and_order_exact():
    fragments = {
        "a": _benchmark_fragment_fixture("a"),
        "b": _benchmark_fragment_fixture("b"),
    }
    pairs = tuple(
        StrictRealPairInput("pair-{}".format(index), "case", "a", "b")
        for index in range(3)
    )
    calls = []

    class Frozen:
        def predict_batch(self, batch, *, return_correspondence):
            calls.append((tuple(batch.pair_ids), return_correspondence))
            count = len(batch.pair_ids)
            assert np.all(np.asarray(batch.labels) == 0.0)
            return external.benchmark_adapter.CommonBatchPrediction(
                schema_version=external.benchmark_adapter.SCHEMA_VERSION,
                method_key=external.benchmark_adapter.PAIRINGNET_METHOD_KEY,
                method_id="fixture",
                pair_ids=tuple(batch.pair_ids),
                pair_probability=np.full(count, 0.75, dtype=np.float32),
                decision_valid=np.ones(count, dtype=np.bool_),
                translation_hat_rc=np.tile(
                    np.asarray([[1.0, -2.0]], dtype=np.float32), (count, 1)
                ),
                translation_valid=np.ones(count, dtype=np.bool_),
                correspondence_indices=None,
                correspondence_scores=None,
                correspondence_semantics=None,
                auxiliary_scores={},
            )

    predictions = external.score_target_blind_benchmark(
        Frozen(), pairs, fragments, batch_size=2
    )
    assert [value.pair_id for value in predictions] == [
        "pair-0",
        "pair-1",
        "pair-2",
    ]
    assert calls == [(('pair-0', 'pair-1'), False), (('pair-2',), False)]
    assert all(value.translation_hat_rc == (1.0, -2.0) for value in predictions)


def test_same_data_benchmark_freeze_failure_precedes_any_real_open(
    tmp_path, monkeypatch
):
    config = _small_config()
    winners = {
        arm: FrozenRachelWinner(
            arm=arm,
            epoch=20,
            checkpoint_path=Path("unused"),
            checkpoint_sha256=("2" if arm == "coarse_only" else "3") * 64,
            threshold=_threshold(
                ("2" if arm == "coarse_only" else "3") * 64, config, arm
            ),
            model_config=config,
            model=_FakeCoarse() if arm == "coarse_only" else _FakeFull(),
        )
        for arm in external.SUPPORTED_ARMS
    }
    monkeypatch.setattr(
        external,
        "_freeze_completed_n512_authority",
        lambda *a, **k: (
            {"status": "complete_train_validation_only"},
            "9" * 64,
            winners,
        ),
    )
    monkeypatch.setattr(
        external,
        "_require_formal_n512_convergence",
        lambda *a, **k: {"status": "complete_train_validation_only"},
    )
    monkeypatch.setattr(
        external.benchmark_adapter,
        "freeze_same_data_benchmarks",
        lambda **kwargs: (_ for _ in ()).throw(
            external.benchmark_adapter.RachelBenchmarkEvalAdapterError(
                "benchmark freeze failed"
            )
        ),
    )
    opened = False

    def forbidden_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("real authority opened before benchmark freeze")

    monkeypatch.setattr(external, "prepare_strict_real_population", forbidden_open)
    with pytest.raises(RachelRealExternalError, match="benchmark freeze failed"):
        external.evaluate_strict_real_external(
            tmp_path,
            tmp_path / "manifest.json",
            tmp_path / "receipt.json",
            pairingnet_run_directory=tmp_path / "pairing",
            shreddingnet_freeze_path=tmp_path / "shredding.json",
            compatibility_mode=True,
        )
    assert opened is False


def test_same_data_benchmark_augmentation_discloses_non_native_shred_metrics():
    config = _small_config()
    n512_winners = {
        arm: FrozenRachelWinner(
            arm=arm,
            epoch=20,
            checkpoint_path=Path("unused"),
            checkpoint_sha256=("2" if arm == "coarse_only" else "3") * 64,
            threshold=_threshold(
                ("2" if arm == "coarse_only" else "3") * 64, config, arm
            ),
            model_config=config,
            model=_FakeCoarse() if arm == "coarse_only" else _FakeFull(),
        )
        for arm in external.SUPPORTED_ARMS
    }
    population = PreparedStrictRealPopulation(
        manifest_sha256="4" * 64,
        fragments={
            "a": _prepared_fragment_stub("a", "case"),
            "b": _prepared_fragment_stub("b", "case"),
        },
        pair_inputs=tuple(
            StrictRealPairInput("pair-{}".format(index), "case", "a", "b")
            for index in range(4)
        ),
        labels=(True, True, False, False),
        case_clusters=("c0", "c0", "c1", "c1"),
    )
    base_predictions = {
        arm: tuple(
            TargetBlindPrediction(
                pair_id=row.pair_id,
                probability=0.8 if population.labels[index] else 0.2,
                valid=True,
                coarse_probability=0.8 if population.labels[index] else 0.2,
                coarse_valid=True,
                translation_hat_rc=(0.0, 0.0) if arm == "full_n512" else None,
                translation_dispersion_px=0.0 if arm == "full_n512" else None,
            )
            for index, row in enumerate(population.pair_inputs)
        )
        for arm in external.SUPPORTED_ARMS
    }
    base = summarize_strict_real_predictions(
        population, n512_winners, base_predictions
    )
    frozen = {
        method: _benchmark_frozen_fixture(method)
        for method in external.benchmark_adapter.BENCHMARK_METHODS
    }
    predictions = {
        method: tuple(
            TargetBlindPrediction(
                pair_id=row.pair_id,
                probability=0.9 if population.labels[index] else 0.1,
                valid=True,
                coarse_probability=0.5,
                coarse_valid=True,
                translation_hat_rc=(1.0, 2.0),
                translation_dispersion_px=None,
            )
            for index, row in enumerate(population.pair_inputs)
        )
        for method in external.benchmark_adapter.BENCHMARK_METHODS
    }
    result = external._augment_with_same_data_benchmarks(
        base,
        population,
        frozen,
        predictions,
        training_manifest_evidence={"exact_bytes": True},
    )
    shred = result["methods"][
        external.benchmark_adapter.SHREDDINGNET_METHOD_KEY
    ]
    assert shred["native_cm_fm_se_or_ga_claimed"] is False
    assert shred["direct_metric_status"]["shreddingnet_native_CM_FM_SE"] == (
        "not_reported"
    )
    assert "native_CM" not in shred
    assert result["pairs"][0]["methods"][
        external.benchmark_adapter.PAIRINGNET_METHOD_KEY
    ]["translation_hat_rc_unsupervised"] == (1.0, 2.0)
    assert set(result["methods"]) == set(external.SUPPORTED_ARMS) | set(
        external.benchmark_adapter.BENCHMARK_METHODS
    )


def test_balanced_same_data_benchmarks_forward_once_then_slice_strict(
    tmp_path, monkeypatch
):
    config = _small_config()
    n512_winners = {
        arm: FrozenRachelWinner(
            arm=arm,
            epoch=20,
            checkpoint_path=Path("unused"),
            checkpoint_sha256=("2" if arm == "coarse_only" else "3") * 64,
            threshold=_threshold(
                ("2" if arm == "coarse_only" else "3") * 64, config, arm
            ),
            model_config=config,
            model=_FakeCoarse() if arm == "coarse_only" else _FakeFull(),
        )
        for arm in external.SUPPORTED_ARMS
    }
    frozen = {
        method: _benchmark_frozen_fixture(method)
        for method in external.benchmark_adapter.BENCHMARK_METHODS
    }
    strict_inputs = tuple(
        StrictRealPairInput("p{}".format(index), "c", "a", "b")
        for index in range(2)
    )
    balanced_inputs = strict_inputs + tuple(
        StrictRealPairInput("p{}".format(index), "c", "a", "b")
        for index in range(2, 4)
    )
    strict = SimpleNamespace(pair_inputs=strict_inputs, fragments={})
    balanced = SimpleNamespace(pair_inputs=balanced_inputs, fragments={})
    monkeypatch.setattr(
        external,
        "_freeze_completed_n512_authority",
        lambda *a, **k: (
            {
                "status": "complete_train_validation_only",
                "config": {"dataset_root": "/fixture"},
            },
            "9" * 64,
            n512_winners,
        ),
    )
    monkeypatch.setattr(
        external,
        "_require_formal_n512_convergence",
        lambda *a, **k: a[2] if len(a) > 2 else k.get("receipt", {}),
    )
    monkeypatch.setattr(
        external.benchmark_adapter,
        "freeze_same_data_benchmarks",
        lambda **kwargs: frozen,
    )
    monkeypatch.setattr(
        external.sealed_authority,
        "_resolved_training_dataset_root",
        lambda *a, **k: Path("/fixture"),
    )
    monkeypatch.setattr(
        external.sealed_authority,
        "_benchmark_training_manifest_evidence",
        lambda *a, **k: {"exact": True},
    )
    monkeypatch.setattr(external, "prepare_strict_real_population", lambda *a, **k: strict)
    monkeypatch.setattr(external, "build_balanced_1016_population", lambda value: balanced)
    monkeypatch.setattr(external, "EXPECTED_STRICT_PAIR_COUNT", 2)
    monkeypatch.setattr(external, "EXPECTED_BALANCED_PAIR_COUNT", 4)
    monkeypatch.setattr(
        external,
        "score_target_blind",
        lambda winner, pair_inputs, fragments, **kwargs: tuple(
            TargetBlindPrediction(
                row.pair_id, 0.5, True, 0.5, True, None, None
            )
            for row in pair_inputs
        ),
    )
    calls = {method: [] for method in external.benchmark_adapter.BENCHMARK_METHODS}

    def benchmark_score(winner, pair_inputs, fragments, **kwargs):
        calls[winner.method_key].append(len(pair_inputs))
        return tuple(
            TargetBlindPrediction(
                row.pair_id, 0.5, True, 0.5, True, (0.0, 0.0), None
            )
            for row in pair_inputs
        )

    monkeypatch.setattr(external, "score_target_blind_benchmark", benchmark_score)
    monkeypatch.setattr(
        external,
        "summarize_strict_real_predictions",
        lambda *a, **k: {"kind": "strict"},
    )
    monkeypatch.setattr(
        external,
        "summarize_balanced_1016_predictions",
        lambda *a, **k: {"kind": "balanced"},
    )
    augmented = []

    def augment(base, population, winners, predictions, **kwargs):
        augmented.append(
            (base["kind"], {key: len(value) for key, value in predictions.items()})
        )
        return base

    monkeypatch.setattr(external, "_augment_with_same_data_benchmarks", augment)
    result = external.evaluate_strict_real_external(
        tmp_path,
        tmp_path / "manifest.json",
        tmp_path / "paths.json",
        pairingnet_run_directory=tmp_path / "pairing",
        shreddingnet_freeze_path=tmp_path / "shredding.json",
        include_balanced_1016=True,
        compatibility_mode=True,
    )
    assert all(value == [4] for value in calls.values())
    assert augmented == [
        (
            "strict",
            {
                method: 2
                for method in external.benchmark_adapter.BENCHMARK_METHODS
            },
        ),
        (
            "balanced",
            {
                method: 4
                for method in external.benchmark_adapter.BENCHMARK_METHODS
            },
        ),
    ]
    assert result["forward_contract"]["strict_pairs_forwarded_twice"] is False

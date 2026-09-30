import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from staging.pairwise_v0_2.baselines import rachel_matched_mm_siamese as matched
from staging.pairwise_v0_2.baselines.historical_mm_siamese import (
    HistoricalMMSiamese as ReferenceHistoricalMMSiamese,
    preprocess_historical_mm_mask,
)
from staging.pairwise_v0_2.baselines.matched_route_a_siamese import (
    initialize_historical_mm_training_weights as initialize_reference_weights,
)


def _mask(path, *, shift=0, constant=False):
    value = np.zeros((800, 800), dtype=np.uint8)
    if constant:
        value.fill(255)
    else:
        value[100 + shift : 500 + shift, 150:650] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="L").save(path)
    return value


def _fragment(path, unit):
    return {
        "split_unit_id": unit,
        "model_mask_path": path,
        "contour_path": "must/not/be/opened.npz",
        "metadata": {"translation_a_to_b_rc": [99, 99]},
    }


def _release(tmp_path):
    root = tmp_path / "release"
    first = "model/masks_800/g/0.png"
    second = "model/masks_800/g/1.png"
    _mask(root / first)
    _mask(root / second, shift=20)
    (root / "pairs").mkdir()
    for split in ("train", "val"):
        rows = [
            {
                "pair_id": split + "-positive",
                "split": split,
                "label": True,
                "fragment_a": _fragment(first, "source-a"),
                "fragment_b": _fragment(second, "source-a"),
                "correspondence_path": "must/not/be/opened.npz",
                "translation_a_to_b_rc": [1, 2],
            },
            {
                "pair_id": split + "-negative",
                "split": split,
                "label": False,
                "fragment_a": _fragment(first, "source-a"),
                "fragment_b": _fragment(second, "source-b"),
                "correspondence_path": None,
                "translation_a_to_b_rc": None,
            },
        ]
        (root / "pairs" / (split + ".jsonl")).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return root


def test_config_freezes_convergence_and_historical_recipe(tmp_path):
    config = matched.RachelMMRunConfig(tmp_path / "data", tmp_path / "out")

    assert config.max_total_epochs == 128
    assert config.min_total_epochs == 20
    assert config.patience == 12
    assert config.min_relative_auroc_improvement == pytest.approx(0.005)
    assert config.learning_rate == pytest.approx(1.0)
    assert config.scheduler_gamma == pytest.approx(0.7)
    assert config.focal_alpha == pytest.approx(0.25)
    assert config.focal_gamma == pytest.approx(2.0)
    assert config.portable_dict()["source_splits_opened"] == ["train", "val"]
    assert config.portable_dict()["test_accessed"] is False


def test_train_val_adapter_reads_only_model_masks_and_preserves_cluster_semantics(
    tmp_path,
):
    root = _release(tmp_path)

    rows = matched._read_rows(root.resolve(), "train")

    assert [row.label for row in rows] == [True, False]
    assert rows[0].cluster_id == "unit:source-a"
    assert rows[1].cluster_id.startswith("unit-pair:")
    assert rows[0].mask_a.parts[-3:-1] == ("masks_800", "g")
    with pytest.raises(matched.RachelMatchedMMError, match="train/val only"):
        matched._read_rows(root.resolve(), "test")


def test_model_mask_path_rejects_target_paths_and_symlinks(tmp_path):
    root = _release(tmp_path).resolve()

    with pytest.raises(matched.RachelMatchedMMError, match="only model/masks_800"):
        matched._safe_model_path(root, "model/contours_n512/g/0.npz")

    target = root / "model" / "masks_800" / "g" / "0.png"
    link = root / "model" / "masks_800" / "g" / "link.png"
    link.symlink_to(target)
    with pytest.raises(matched.RachelMatchedMMError, match="symlinked"):
        matched._safe_model_path(root, "model/masks_800/g/link.png")


def test_manifest_parent_symlink_and_cross_split_identities_are_rejected(tmp_path):
    root = _release(tmp_path).resolve()
    train = matched._read_rows(root, "train")
    val = matched._read_rows(root, "val")
    with pytest.raises(matched.RachelMatchedMMError, match="not disjoint"):
        matched._require_disjoint_splits(train, val)

    separated = tuple(
        replace(
            row,
            unit_a="val-" + row.unit_a,
            unit_b="val-" + row.unit_b,
            mask_a=Path("/val") / row.mask_a.name,
            mask_b=Path("/val") / ("b-" + row.mask_b.name),
        )
        for row in val
    )
    matched._require_disjoint_splits(train, separated)

    real_pairs = root / "real-pairs"
    (root / "pairs").rename(real_pairs)
    (root / "pairs").symlink_to(real_pairs, target_is_directory=True)
    with pytest.raises(matched.RachelMatchedMMError, match="symlinked manifest"):
        matched._read_rows(root, "train")


def test_mask_decode_is_exact_historical_pil_bilinear(tmp_path):
    path = tmp_path / "mask.png"
    array = _mask(path)

    observed = matched._load_mask(path)
    expected = preprocess_historical_mm_mask(array == 255)

    assert torch.equal(observed, expected)
    assert observed.shape == (1, 64, 64)
    assert observed.dtype == torch.float32

    constant = tmp_path / "constant.png"
    _mask(constant, constant=True)
    with pytest.raises(matched.RachelMatchedMMError, match="non-constant binary"):
        matched._load_mask(constant)


def test_historical_focal_formula_matches_companion_source():
    probability = torch.tensor([0.1, 0.8, 0.4], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    observed = matched._ProbabilityFocalLoss(0.25, 2.0)(probability, target)
    bce = torch.nn.functional.binary_cross_entropy(
        probability, target, reduction="none"
    )
    expected = (0.25 * (1.0 - torch.exp(-bce)).square() * bce).mean()
    assert observed == pytest.approx(expected)


def test_winner_and_plateau_rules_are_validation_only(tmp_path):
    rows = [
        {"epoch": 1, "selection_metrics": {"auroc": 0.8, "auprc": 0.7}},
        {"epoch": 2, "selection_metrics": {"auroc": 0.8, "auprc": 0.8}},
        {"epoch": 3, "selection_metrics": {"auroc": 0.8, "auprc": 0.8}},
    ]
    assert matched._winner(rows)["epoch"] == 2

    state = matched._advance_plateau(
        anchor=-float("inf"),
        anchor_epoch=0,
        count=0,
        epoch=1,
        auroc=0.8,
        relative_improvement=0.005,
    )
    assert state == (0.8, 1, 0, True)
    state = matched._advance_plateau(
        anchor=state[0],
        anchor_epoch=state[1],
        count=state[2],
        epoch=2,
        auroc=0.803,
        relative_improvement=0.005,
    )
    assert state == (0.8, 1, 1, False)
    config = matched.RachelMMRunConfig(tmp_path, tmp_path / "out")
    assert not matched._early_stop_due(19, 99, config)
    assert matched._early_stop_due(20, 12, config)


def test_cublas_determinism_is_configured_before_training_import():
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_model_initialization_is_fixed_and_historical(tmp_path):
    first = matched._initialize_model(260831, torch.device("cpu"))
    second = matched._initialize_model(260831, torch.device("cpu"))

    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)
    linear_biases = [
        module.bias for module in first.modules() if isinstance(module, torch.nn.Linear)
    ]
    assert linear_biases
    assert all(
        torch.equal(value, torch.full_like(value, 0.01)) for value in linear_biases
    )


def test_self_contained_recipe_is_tensor_identical_to_reference():
    matched._seed(260831)
    observed = matched.HistoricalMMSiamese()
    matched.initialize_historical_mm_training_weights(observed)

    matched._seed(260831)
    reference = ReferenceHistoricalMMSiamese()
    initialize_reference_weights(reference)

    observed_state = observed.state_dict()
    reference_state = reference.state_dict()
    assert observed_state.keys() == reference_state.keys()
    assert all(
        torch.equal(observed_state[key], reference_state[key]) for key in observed_state
    )

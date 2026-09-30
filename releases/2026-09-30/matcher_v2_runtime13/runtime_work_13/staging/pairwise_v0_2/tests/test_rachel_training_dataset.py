from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import (
    RachelDatasetError,
    RachelPairDataset,
    collate_rachel_pairs,
)


def _write_fragment(root: Path, name: str, points: np.ndarray) -> dict:
    mask_path = root / "model" / "masks_800" / "fixture" / (name + ".png")
    contour_path = root / "model" / "contours_n512" / "fixture" / (name + ".npz")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    contour_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((800, 800), dtype=np.uint8)
    mask[260:540, 280:520] = 255
    Image.fromarray(mask, mode="L").save(mask_path)
    np.savez_compressed(
        contour_path,
        points_rc=np.asarray(points, dtype=np.float32),
        valid=np.ones(len(points), dtype=np.bool_),
    )
    return {
        "fragment_token": "rachel/fixture/" + name,
        "parent_group_id": "rachel/fixture",  # safe lineage provenance
        "model_mask_path": mask_path.relative_to(root).as_posix(),
        "contour_path": contour_path.relative_to(root).as_posix(),
    }


def _release(root: Path) -> tuple:
    points_a = np.asarray(
        [[260, 280], [539, 280], [539, 519], [260, 519]], dtype=np.float32
    )
    points_b = np.concatenate((points_a, [[400, 280]]), axis=0)
    points_c = np.asarray(
        [[260, 280], [400, 280], [539, 280], [539, 519], [400, 519], [260, 519]],
        dtype=np.float32,
    )
    first = _write_fragment(root, "a", points_a)
    second = _write_fragment(root, "b", points_b)
    third = _write_fragment(root, "c", points_c)
    target = root / "targets" / "pairs" / "fixture" / "positive.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        correspondence_indices=np.asarray(
            [[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64
        ),
        translation_a_to_b_rc=np.asarray([0.0, 0.0], dtype=np.float32),
        translation_a_to_b_xy_cartesian=np.asarray([0.0, -0.0], dtype=np.float32),
    )
    positive = {
        "pair_id": "positive",
        "split": "train",
        "label": True,
        "correspondence_path": target.relative_to(root).as_posix(),
        # Deliberately wrong manifest copy: the NPZ is the sole target authority.
        "translation_a_to_b_rc": [999.0, 999.0],
        "fragment_a": first,
        "fragment_b": second,
    }
    negative = {
        "pair_id": "negative",
        "split": "train",
        "label": False,
        "negative_origin": "same_folder_hard",
        "correspondence_path": None,
        "translation_a_to_b_rc": None,
        "fragment_a": first,
        "fragment_b": third,
    }
    manifest = root / "pairs" / "train.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in (positive, negative)),
        encoding="utf-8",
    )
    return positive, negative, manifest, target


def test_selected_loader_and_collate_use_compact_targets_and_dustbin(
    tmp_path: Path,
) -> None:
    _release(tmp_path)
    dataset = RachelPairDataset(tmp_path, "train")

    positive = dataset[0]
    negative = dataset[1]
    assert positive.mask_a.shape == (1, 800, 800)
    assert positive.coarse_mask_a.shape == (1, 128, 128)
    assert positive.mask_a.dtype == np.float32
    assert set(np.unique(positive.mask_a)) == {0.0, 1.0}
    assert set(np.unique(positive.coarse_mask_a)) == {0.0, 1.0}
    assert positive.target_a.tolist() == [0, 1, 2, 3]
    assert positive.target_b.tolist() == [0, 1, 2, 3, -1]
    assert positive.translation_a_to_b_rc.tolist() == [0.0, 0.0]
    assert bool(positive.translation_valid)
    assert np.all(negative.target_a == -1)
    assert np.all(negative.target_b == -1)
    assert not bool(negative.translation_valid)

    batch = collate_rachel_pairs((positive, negative))
    assert batch.mask_a.shape == (2, 1, 800, 800)
    assert batch.coarse_mask_b.shape == (2, 1, 128, 128)
    assert batch.points_rc_a.shape == (2, 512, 2)
    assert batch.contour_valid_a.shape == (2, 512)
    assert batch.target_a.shape == (2, 512)
    assert np.all(batch.target_a[0, 4:] == -2)
    assert np.all(batch.target_b[1, :6] == -1)
    assert np.all(batch.target_b[1, 6:] == -2)
    assert batch.labels.tolist() == [1.0, 0.0]
    assert batch.translation_valid.tolist() == [True, False]


def test_loader_rejects_recursive_parent_audit_field(tmp_path: Path) -> None:
    positive, _, manifest, _ = _release(tmp_path)
    positive["fragment_a"]["metadata"] = {
        "target_audit": {"parent_to_model_offset_rc": [1, 2]}
    }
    manifest.write_text(json.dumps(positive) + "\n", encoding="utf-8")

    with pytest.raises(RachelDatasetError, match="forbidden parent/RGB audit field"):
        RachelPairDataset(tmp_path, "train")


def test_loader_rejects_symlinked_model_input(tmp_path: Path) -> None:
    positive, _, manifest, _ = _release(tmp_path)
    real_mask = tmp_path / positive["fragment_a"]["model_mask_path"]
    link = real_mask.with_name("linked.png")
    try:
        link.symlink_to(real_mask)
    except OSError:
        pytest.skip("symlinks unavailable")
    positive["fragment_a"]["model_mask_path"] = link.relative_to(tmp_path).as_posix()
    manifest.write_text(json.dumps(positive) + "\n", encoding="utf-8")

    with pytest.raises(RachelDatasetError, match="symlinked release artefact"):
        RachelPairDataset(tmp_path, "train")


def test_positive_geometry_residual_gate_fails_closed(tmp_path: Path) -> None:
    _release(tmp_path)
    target = tmp_path / "targets" / "pairs" / "fixture" / "positive.npz"
    np.savez_compressed(
        target,
        correspondence_indices=np.asarray(
            [[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64
        ),
        translation_a_to_b_rc=np.asarray([20.0, 0.0], dtype=np.float32),
        translation_a_to_b_xy_cartesian=np.asarray([0.0, -20.0], dtype=np.float32),
    )

    with pytest.raises(RachelDatasetError, match="residual gate failed"):
        RachelPairDataset(tmp_path, "train")[0]

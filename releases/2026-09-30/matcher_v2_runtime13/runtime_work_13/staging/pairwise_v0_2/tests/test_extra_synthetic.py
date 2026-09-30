import csv
from pathlib import Path

import cv2
import numpy as np

from staging.pairwise_v0_2.pairwise_data.extra_synthetic import (
    iter_mask_pairs,
    iter_rgb_attachments,
)


def _write(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), value)


def test_mask_pairs_and_rgb_are_separate_views(tmp_path: Path) -> None:
    mask_root = tmp_path / "masks"
    group = mask_root / "gen2voronoi_1" / "no_erode" / "7"
    first = np.zeros((32, 32), dtype=np.uint8)
    second = np.zeros_like(first)
    first[:, :16] = 255
    second[:, 16:] = 255
    _write(group / "0.png", first)
    _write(group / "1.png", second)

    pairs = tuple(iter_mask_pairs(mask_root, max_groups=1))
    assert len(pairs) == 1
    assert pairs[0].label is True
    assert pairs[0].fragment_a_id == "0"
    assert pairs[0].fragment_b_id == "1"

    rgb_group = (
        tmp_path / "datasets" / "gen2voronoi_1" / "no_erode" / "7"
    )
    _write(rgb_group / "0.jpg", np.dstack([first] * 3))
    _write(rgb_group / "1.jpg", np.dstack([second] * 3))
    with (rgb_group / "label.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "neighbors"])
        writer.writerow([0, 1])
        writer.writerow([1, 0])

    attachments = tuple(iter_rgb_attachments(tmp_path / "datasets", mask_root))
    assert len(attachments) == 1
    assert attachments[0].group_key == "gen2voronoi_1/no_erode/7"
    assert attachments[0].fragment_ids == ("0", "1")


def test_non_neighbor_pair(tmp_path: Path) -> None:
    group = tmp_path / "masks" / "gen2voronoi_1" / "no_erode" / "0"
    first = np.zeros((128, 128), dtype=np.uint8)
    second = np.zeros_like(first)
    first[5:15, 5:15] = 255
    second[100:110, 100:110] = 255
    _write(group / "0.png", first)
    _write(group / "1.png", second)

    (pair,) = tuple(iter_mask_pairs(tmp_path / "masks"))
    assert pair.label is False
    assert pair.dilated_overlap_pixels == 0

from __future__ import annotations

import ast
import csv
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from staging.pairwise_v0_2.pairwise_data import rachel_preprocess as rp


def _write_rgb_fragment(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[mask] = np.array([225, 185, 135], dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(
        path,
        format="JPEG",
        quality=100,
        subsampling=0,
    )


def _write_label_csv(
    path: Path,
    *,
    masks: tuple,
    edges: tuple,
    image_name: str,
) -> None:
    fragment_count = len(masks)
    neighbors = {str(index): set() for index in range(fragment_count)}
    for first, second in edges:
        neighbors[str(first)].add(str(second))
        neighbors[str(second)].add(str(first))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "id",
                "mask_id",
                "image_name",
                "center_x",
                "center_y",
                "width",
                "height",
                "neighbors",
            ]
        )
        for fragment_id in sorted(neighbors, key=int):
            rows, columns = np.nonzero(masks[int(fragment_id)])
            width = int(columns.max() - columns.min() + 1)
            height = int(rows.max() - rows.min() + 1)
            writer.writerow(
                [
                    fragment_id,
                    fragment_id,
                    image_name,
                    float(columns.mean()),
                    float(rows.mean()),
                    width,
                    height,
                    ";".join(sorted(neighbors[fragment_id], key=int)),
                ]
            )


def _write_raw_label_csv(
    path: Path,
    *,
    masks: tuple,
    neighbor_cells: tuple,
    image_names: tuple,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "id",
                "mask_id",
                "image_name",
                "center_x",
                "center_y",
                "width",
                "height",
                "neighbors",
            ]
        )
        for fragment_id, (mask, neighbor_cell, image_name) in enumerate(
            zip(masks, neighbor_cells, image_names)
        ):
            rows, columns = np.nonzero(mask)
            writer.writerow(
                [
                    fragment_id,
                    fragment_id,
                    image_name,
                    float(columns.mean()),
                    float(rows.mean()),
                    int(columns.max() - columns.min() + 1),
                    int(rows.max() - rows.min() + 1),
                    neighbor_cell,
                ]
            )


def _write_group(
    root: Path,
    *,
    generator: str,
    group_id: str,
    masks: tuple,
    edges: tuple,
    image_name: str,
) -> Path:
    group = root / generator / "no_erode" / group_id
    group.mkdir(parents=True)
    for fragment_id, mask in enumerate(masks):
        _write_rgb_fragment(group / (str(fragment_id) + ".jpg"), mask)
    _write_label_csv(
        group / "label.csv",
        masks=masks,
        edges=edges,
        image_name=image_name,
    )
    return group


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Return an 8-neighbour, one-pixel internal boundary without SciPy."""

    padded = np.pad(np.asarray(mask, dtype=np.bool_), 1, constant_values=False)
    interior = np.ones(mask.shape, dtype=np.bool_)
    for row_delta in range(3):
        for column_delta in range(3):
            interior &= padded[
                row_delta : row_delta + mask.shape[0],
                column_delta : column_delta + mask.shape[1],
            ]
    return np.asarray(mask, dtype=np.bool_) & ~interior


def _shoelace_xy(points_rc: np.ndarray) -> float:
    """Signed area using the declared Cartesian convention x=col, y=-row."""

    rows = np.asarray(points_rc[:, 0], dtype=np.float64)
    columns = np.asarray(points_rc[:, 1], dtype=np.float64)
    x = columns
    y = -rows
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _within_boundary_tolerance(
    boundary: np.ndarray,
    points_rc: np.ndarray,
    *,
    tolerance: int,
) -> bool:
    boundary_points = np.argwhere(boundary).astype(np.float64)
    for point in np.asarray(points_rc, dtype=np.float64):
        squared_distance = np.sum((boundary_points - point[None, :]) ** 2, axis=1)
        if float(np.min(squared_distance)) > float(tolerance * tolerance):
            return False
    return True


def _rectangle_contour_ccw(
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> np.ndarray:
    """Dense closed raster contour, CCW for Cartesian x=column,y=-row."""

    points = []
    points.extend((row, left) for row in range(top, bottom + 1))
    points.extend((bottom, column) for column in range(left + 1, right + 1))
    points.extend((row, right) for row in range(bottom - 1, top - 1, -1))
    points.extend((top, column) for column in range(right - 1, left, -1))
    return np.asarray(points, dtype=np.float32)


def test_rgb_to_mask_is_boolean_thresholded_and_cleans_only_local_artifacts() -> None:
    config = rp.RachelPreprocessConfig(
        canvas_size=48,
        threshold=8,
        contour_cap=512,
        minimum_positive_seam=8,
        seam_bridge_radius=2,
    )
    rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    rgb[6:42, 7:40] = np.array([170, 120, 80], dtype=np.uint8)

    # JPEG-like black ink/pinholes inside the paper support must not survive in
    # the filled silhouette used by the mask-only model.
    rgb[20, 20] = 0
    rgb[21, 20] = 0

    # Threshold is strict: an isolated pixel at exactly 8 is background; a
    # brighter disconnected codec speck is removed with the non-main component.
    rgb[0, 0] = 8
    rgb[0, 2] = 9

    # A substantial exterior notch is real outline geometry, not a JPEG hole.
    rgb[6:16, 7:14] = 0

    mask = rp.rgb_to_binary_mask(rgb, config)

    assert mask.dtype == np.bool_
    assert mask.shape == (48, 48)
    assert mask.flags.c_contiguous
    assert mask[20, 20] and mask[21, 20]
    assert not mask[0, 0]
    assert not mask[0, 2]
    assert not mask[8, 9]
    assert mask[30, 30]


def test_rgb_to_mask_does_not_use_luminance_that_drops_single_channel_paper() -> None:
    config = rp.RachelPreprocessConfig(
        canvas_size=24,
        threshold=8,
        minimum_positive_seam=4,
        seam_bridge_radius=2,
    )
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[4:20, 5:19, 0] = 12

    mask = rp.rgb_to_binary_mask(rgb, config)

    # The preprocessing protocol is max(R,G,B)>threshold.  A luminance rule
    # would incorrectly erase this low-red synthetic fixture.
    assert mask[10, 10]
    assert int(mask.sum()) == 16 * 14


def test_ordered_outer_contour_is_dense_external_ccw_and_capped_at_512() -> None:
    mask = np.zeros((800, 800), dtype=np.bool_)
    mask[50:750, 80:720] = True
    mask[300:500, 300:500] = False  # internal hole must not become input contour

    first_points, first_valid = rp.extract_ordered_outer_contour(mask, cap=512)
    second_points, second_valid = rp.extract_ordered_outer_contour(mask, cap=512)

    assert first_points.shape == (512, 2)
    assert first_valid.shape == (512,)
    assert first_valid.dtype == np.bool_
    assert first_valid.all()
    assert np.array_equal(first_points, second_points)
    assert np.array_equal(first_valid, second_valid)
    assert len(np.unique(first_points, axis=0)) == 512

    rounded = np.rint(first_points).astype(np.int64)
    assert ((rounded[:, 0] >= 0) & (rounded[:, 0] < 800)).all()
    assert ((rounded[:, 1] >= 0) & (rounded[:, 1] < 800)).all()
    # PairingNet/ShreddingNet-style contour-coordinate smoothing may move a
    # corner a few pixels off the raw raster boundary, but may not create an
    # unrelated interior curve.
    assert _within_boundary_tolerance(
        _boundary(mask),
        first_points,
        tolerance=3,
    )
    assert not (
        (rounded[:, 0] >= 299)
        & (rounded[:, 0] <= 500)
        & (rounded[:, 1] >= 299)
        & (rounded[:, 1] <= 500)
    ).any()
    assert _shoelace_xy(first_points) > 0.0

    step = np.linalg.norm(first_points - np.roll(first_points, -1, axis=0), axis=1)
    assert float(np.percentile(step, 95)) <= 2.0 * float(np.percentile(step, 5))
    assert (first_points[:, 0] < 100).any()
    assert (first_points[:, 0] > 700).any()
    assert (first_points[:, 1] < 130).any()
    assert (first_points[:, 1] > 670).any()


def test_short_contour_is_not_repeated_or_padded_to_cap() -> None:
    mask = np.zeros((64, 64), dtype=np.bool_)
    mask[20:30, 22:34] = True

    points, valid = rp.extract_ordered_outer_contour(mask, cap=512)

    assert 4 <= len(points) < 512
    assert points.shape == (len(points), 2)
    assert valid.shape == (len(points),)
    assert valid.all()
    assert len(np.unique(points, axis=0)) == len(points)
    assert _shoelace_xy(points) > 0.0


def test_reciprocal_correspondence_rejects_one_way_and_over_3px_matches() -> None:
    contour_a = np.asarray(
        [[0.0, 0.0], [0.0, 2.0], [0.0, 2.8], [0.0, 10.0], [30.0, 30.0]],
        dtype=np.float32,
    )
    contour_b = np.asarray(
        [[0.0, 0.2], [0.0, 3.0], [0.0, 13.0], [40.0, 40.0]],
        dtype=np.float32,
    )

    matches = rp.recover_mutual_contour_correspondences(
        contour_a,
        contour_b,
        max_distance_px=3.0,
    )

    assert matches.dtype == np.int64
    assert matches.shape == (3, 2)
    assert matches.tolist() == [[0, 0], [2, 1], [3, 2]]

    # A[1] selects B[1], but B[1] selects the closer A[2], so A[1] is not a
    # reciprocal correspondence.  The far final points also cannot match.
    assert [1, 1] not in matches.tolist()
    assert [4, 3] not in matches.tolist()


def test_reciprocal_correspondence_is_full_contour_and_not_cardinal_gated() -> None:
    coordinate = np.arange(80, dtype=np.float32)
    contour_a = np.column_stack((coordinate, coordinate))
    # Non-integer oblique displacement avoids nearest-neighbour ties while
    # exercising a clearly non-cardinal seam.
    contour_b = contour_a + np.asarray([0.75, 1.5], dtype=np.float32)

    matches = rp.recover_mutual_contour_correspondences(
        contour_a,
        contour_b,
        max_distance_px=3.0,
    )

    # An oblique local seam remains matchable.  A four-cardinal-side gate would
    # incorrectly split or discard this diagonal run.
    assert len(matches) >= 78
    assert np.all(np.diff(matches[:, 0]) >= 0)
    assert matches[:, 0].min() >= 0 and matches[:, 0].max() < len(contour_a)
    assert matches[:, 1].min() >= 0 and matches[:, 1].max() < len(contour_b)


def test_reciprocal_correspondence_empty_result_has_stable_matrix_shape() -> None:
    contour_a = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    contour_b = np.asarray([[20.0, 20.0], [20.0, 21.0]], dtype=np.float32)

    matches = rp.recover_mutual_contour_correspondences(contour_a, contour_b)

    assert matches.dtype == np.int64
    assert matches.shape == (0, 2)


def test_closed_complementary_seam_is_reverse_ordered_and_swap_symmetric() -> None:
    contour_a = _rectangle_contour_ccw(0, 0, 40, 20)
    contour_b = _rectangle_contour_ccw(0, 22, 40, 42)
    # Move the B seam across the cyclic origin to ensure matching does not rely
    # on both contour arrays beginning at corresponding points.
    contour_b = np.roll(contour_b, 13, axis=0)

    a_to_b = rp.recover_mutual_contour_correspondences(
        contour_a,
        contour_b,
        max_distance_px=3.0,
    )
    b_to_a = rp.recover_mutual_contour_correspondences(
        contour_b,
        contour_a,
        max_distance_px=3.0,
    )

    assert {tuple(pair) for pair in a_to_b.tolist()} == {
        (second, first) for first, second in b_to_a.tolist()
    }
    seam_pairs = np.asarray(
        [
            pair
            for pair in a_to_b
            if contour_a[pair[0], 1] == 20.0
            and contour_b[pair[1], 1] == 22.0
        ],
        dtype=np.int64,
    )
    assert seam_pairs.shape == (41, 2)
    assert np.all(np.diff(seam_pairs[:, 0]) > 0)

    # The target traversal is reverse-monotone modulo its cyclic origin.
    target_indices = seam_pairs[:, 1]
    modular_step = np.mod(np.diff(target_indices), len(contour_b))
    assert set(modular_step.tolist()) == {len(contour_b) - 1}


def test_centerpad_removes_parent_origin_without_resizing_and_translation_sign() -> None:
    first_parent = np.zeros((64, 64), dtype=np.bool_)
    second_parent = np.zeros((64, 64), dtype=np.bool_)
    first_parent[6:26, 4:16] = True
    second_parent[31:51, 39:51] = True

    first = rp.centerpad_mask_with_transform(first_parent, canvas_size=64)
    second = rp.centerpad_mask_with_transform(second_parent, canvas_size=64)

    assert first.model_mask.dtype == np.bool_
    assert second.model_mask.dtype == np.bool_
    assert first.model_mask.shape == second.model_mask.shape == (64, 64)
    assert np.array_equal(first.model_mask, second.model_mask)
    assert int(first.model_mask.sum()) == int(first_parent.sum())
    assert int(second.model_mask.sum()) == int(second_parent.sum())

    assert tuple(first.bbox_min_rc) == (6, 4)
    assert tuple(second.bbox_min_rc) == (31, 39)
    assert tuple(first.pad_start_rc) == tuple(second.pad_start_rc) == (22, 26)
    assert tuple(first.parent_to_model_offset_rc) == (16, 22)
    assert tuple(second.parent_to_model_offset_rc) == (-9, -13)

    a_to_b = rp.translation_a_to_b(first, second)
    b_to_a = rp.translation_a_to_b(second, first)
    assert np.asarray(a_to_b).dtype == np.float64
    assert np.allclose(a_to_b, [-25.0, -35.0])
    assert np.allclose(b_to_a, [25.0, 35.0])
    assert np.allclose(np.asarray(a_to_b) + np.asarray(b_to_a), 0.0)

    # For any physical parent point s, m_i=s+offset_i.  The declared A->B
    # target must therefore map its A-frame coordinate into the B frame.
    parent_point = np.asarray([10.5, 12.5], dtype=np.float64)
    model_a = parent_point + np.asarray(first.parent_to_model_offset_rc)
    model_b = parent_point + np.asarray(second.parent_to_model_offset_rc)
    assert np.allclose(model_a + np.asarray(a_to_b), model_b)

    first_contour = rp.extract_ordered_outer_contour(first.model_mask, cap=512)
    second_contour = rp.extract_ordered_outer_contour(second.model_mask, cap=512)
    assert np.array_equal(first_contour[0], second_contour[0])
    assert np.array_equal(first_contour[1], second_contour[1])


def test_lineage_assignment_is_order_independent_and_deduplicates_image_name() -> None:
    repeated = [
        "folio-A.jpg",
        "folio-B.jpg",
        "folio-A.jpg",
        "folio-C.jpg",
        "folio-B.jpg",
    ]
    first = rp.assign_lineage_splits(repeated, seed="rachel-fixture")
    second = rp.assign_lineage_splits(reversed(repeated), seed="rachel-fixture")

    assert first == second
    assert set(first) == {"folio-A.jpg", "folio-B.jpg", "folio-C.jpg"}
    assert set(first.values()).issubset({"train", "val", "test"})


def test_lineage_assignment_is_exact_8_1_1_and_python_hash_seed_independent() -> None:
    image_names = ["folio-{:03d}.jpg".format(index) for index in range(100)]
    mapping = rp.assign_lineage_splits(image_names, seed="rachel-fixture")
    assert Counter(mapping.values()) == {"train": 80, "val": 10, "test": 10}

    script = (
        "import json; "
        "from staging.pairwise_v0_2.pairwise_data.rachel_preprocess "
        "import assign_lineage_splits; "
        "names=['folio-%03d.jpg'%i for i in range(100)]; "
        "print(json.dumps(assign_lineage_splits(names, seed='rachel-fixture'), "
        "sort_keys=True))"
    )
    outputs = []
    for hash_seed in ("1", "987654"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(json.loads(completed.stdout))
    assert outputs[0] == outputs[1] == mapping


def test_group_reader_quarantines_only_csv_positive_without_local_match(
    tmp_path: Path,
) -> None:
    first = np.zeros((96, 96), dtype=np.bool_)
    second = np.zeros((96, 96), dtype=np.bool_)
    first[10:86, 5:25] = True
    second[10:86, 65:85] = True
    group = _write_group(
        tmp_path,
        generator="gen2voronoi_1",
        group_id="far-positive",
        masks=(first, second),
        edges=((0, 1),),
        image_name="folio-A.jpg",
    )
    config = rp.RachelPreprocessConfig(
        canvas_size=96,
        minimum_positive_seam=64,
        seam_bridge_radius=2,
    )

    result = rp.read_rachel_group(
        group,
        generator="gen2voronoi_1",
        config=config,
    )

    pair = result.pairs[0]
    assert pair.label is True
    assert pair.status == "quarantined"
    assert "positive" in pair.quarantine_reason
    assert "correspondence" in pair.quarantine_reason


def test_group_reader_quarantines_only_csv_negative_with_accepted_local_seam(
    tmp_path: Path,
) -> None:
    first = np.zeros((96, 96), dtype=np.bool_)
    second = np.zeros((96, 96), dtype=np.bool_)
    first[10:86, 5:35] = True
    second[10:86, 37:67] = True  # two missing columns; contour distance is 3
    group = _write_group(
        tmp_path,
        generator="gen2voronoi_1",
        group_id="touching-negative",
        masks=(first, second),
        edges=(),
        image_name="folio-A.jpg",
    )
    config = rp.RachelPreprocessConfig(
        canvas_size=96,
        minimum_positive_seam=64,
        seam_bridge_radius=2,
    )

    result = rp.read_rachel_group(
        group,
        generator="gen2voronoi_1",
        config=config,
    )

    pair = result.pairs[0]
    assert pair.label is False
    assert pair.status == "quarantined"
    assert "negative" in pair.quarantine_reason
    assert "correspondence" in pair.quarantine_reason


def test_corner_only_raw_match_does_not_turn_diagonal_negative_into_conflict(
    tmp_path: Path,
) -> None:
    masks = []
    for row_start, column_start in ((5, 5), (5, 46), (46, 5), (46, 46)):
        mask = np.zeros((96, 96), dtype=np.bool_)
        mask[row_start : row_start + 40, column_start : column_start + 40] = True
        masks.append(mask)
    group = _write_group(
        tmp_path,
        generator="gen4voronoi",
        group_id="four-way-junction",
        masks=tuple(masks),
        edges=((0, 1), (0, 2), (1, 3), (2, 3)),
        image_name="folio-Q.jpg",
    )

    result = rp.read_rachel_group(
        group,
        generator="gen4voronoi",
        config=rp.RachelPreprocessConfig(
            canvas_size=96,
            minimum_positive_seam=32,
            seam_bridge_radius=2,
        ),
    )

    assert result.image_name == "folio-Q.jpg"
    # Diagonal fragments have reciprocal corner points within 3px but no
    # continuous, tangent-consistent seam.  They must remain valid negatives.
    pair_by_ids = {
        frozenset((str(pair.fragment_a_id), str(pair.fragment_b_id))): pair
        for pair in result.pairs
    }
    for ids in (("0", "3"), ("1", "2")):
        pair = pair_by_ids[frozenset(ids)]
        assert pair.label is False
        assert pair.accepted_seam_match_count == 0


@pytest.mark.parametrize(
    "seam_pixels,expected_eligible",
    [(63, False), (64, True)],
)
def test_dense_seam_63_64_eligibility_is_invariant_to_model_contour_cap(
    tmp_path: Path,
    seam_pixels: int,
    expected_eligible: bool,
) -> None:
    first = np.zeros((96, 96), dtype=np.bool_)
    second = np.zeros((96, 96), dtype=np.bool_)
    first[8 : 8 + seam_pixels, 5:35] = True
    second[8 : 8 + seam_pixels, 37:67] = True
    group = _write_group(
        tmp_path,
        generator="gen2voronoi_1",
        group_id="seam-{}".format(seam_pixels),
        masks=(first, second),
        edges=((0, 1),),
        image_name="folio-S.jpg",
    )

    pairs = []
    for cap in (64, 512):
        result = rp.read_rachel_group(
            group,
            generator="gen2voronoi_1",
            config=rp.RachelPreprocessConfig(
                canvas_size=96,
                contour_cap=cap,
                minimum_positive_seam=64,
                seam_bridge_radius=2,
            ),
        )
        pairs.append(result.pairs[0])

    assert pairs[0].seam_length_px == pytest.approx(float(seam_pixels))
    assert pairs[1].seam_length_px == pytest.approx(float(seam_pixels))
    assert pairs[0].main_training_eligible is expected_eligible
    assert pairs[1].main_training_eligible is expected_eligible
    assert pairs[0].label is pairs[1].label is True
    if not expected_eligible:
        assert pairs[0].selection_exclusion_reason == (
            "positive_seam_shorter_than_64_pixels"
        )


def test_group_reader_normalizes_supported_neighbor_serializations(
    tmp_path: Path,
) -> None:
    masks = []
    for first_column in (1, 27, 53):
        mask = np.zeros((96, 96), dtype=np.bool_)
        mask[10:86, first_column : first_column + 24] = True
        masks.append(mask)
    group = tmp_path / "gen3voronoi" / "no_erode" / "mixed-neighbors"
    group.mkdir(parents=True)
    for fragment_id, mask in enumerate(masks):
        _write_rgb_fragment(group / (str(fragment_id) + ".jpg"), mask)
    _write_raw_label_csv(
        group / "label.csv",
        masks=tuple(masks),
        neighbor_cells=("[1]", "0,2", "1;"),
        image_names=("folio-M.jpg",) * 3,
    )
    config = rp.RachelPreprocessConfig(
        canvas_size=96,
        minimum_positive_seam=64,
        seam_bridge_radius=2,
    )

    result = rp.read_rachel_group(
        group,
        generator="gen3voronoi",
        config=config,
    )

    assert result.image_name == "folio-M.jpg"
    assert {
        tuple(str(item) for item in edge) for edge in result.neighbor_edges
    } == {("0", "1"), ("1", "2")}


def test_csv_id_mask_id_and_jpeg_stem_are_mapped_without_row_order_assumption(
    tmp_path: Path,
) -> None:
    first = np.zeros((96, 96), dtype=np.bool_)
    second = np.zeros((96, 96), dtype=np.bool_)
    first[10:86, 5:35] = True
    second[10:86, 37:67] = True
    group = tmp_path / "gen2voronoi_1" / "no_erode" / "id-mapping"
    group.mkdir(parents=True)
    _write_rgb_fragment(group / "0.jpg", first)
    _write_rgb_fragment(group / "1.jpg", second)
    with (group / "label.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "id",
                "mask_id",
                "image_name",
                "center_x",
                "center_y",
                "width",
                "height",
                "neighbors",
            ]
        )
        # Deliberately reverse row order and separate semantic ID from the JPG
        # stem.  Neighbours live in the semantic `id` namespace.
        writer.writerow([20, 1, "folio-I.jpg", 0, 0, 30, 76, "10"])
        writer.writerow([10, 0, "folio-I.jpg", 0, 0, 30, 76, "20"])

    result = rp.read_rachel_group(
        group,
        generator="gen2voronoi_1",
        config=rp.RachelPreprocessConfig(
            canvas_size=96,
            minimum_positive_seam=64,
            seam_bridge_radius=2,
        ),
    )

    assert {
        tuple(str(item) for item in edge) for edge in result.neighbor_edges
    } == {("10", "20")}
    identity = {
        str(fragment.fragment_id): (
            str(fragment.mask_id),
            Path(fragment.jpeg_path).name,
        )
        for fragment in result.fragments
    }
    assert identity == {"10": ("0", "0.jpg"), "20": ("1", "1.jpg")}


@pytest.mark.parametrize("fault", ["missing_jpeg", "extra_jpeg", "duplicate_mask_id"])
def test_group_reader_rejects_jpeg_inventory_or_mask_id_mismatch(
    tmp_path: Path,
    fault: str,
) -> None:
    masks = []
    for first_column in (5, 37):
        mask = np.zeros((96, 96), dtype=np.bool_)
        mask[10:86, first_column : first_column + 30] = True
        masks.append(mask)
    group = tmp_path / "gen2voronoi_1" / "no_erode" / fault
    group.mkdir(parents=True)
    _write_rgb_fragment(group / "0.jpg", masks[0])
    if fault != "missing_jpeg":
        _write_rgb_fragment(group / "1.jpg", masks[1])
    if fault == "extra_jpeg":
        _write_rgb_fragment(group / "2.jpg", masks[1])

    with (group / "label.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "id",
                "mask_id",
                "image_name",
                "center_x",
                "center_y",
                "width",
                "height",
                "neighbors",
            ]
        )
        second_mask_id = 0 if fault == "duplicate_mask_id" else 1
        writer.writerow([10, 0, "folio-J.jpg", 0, 0, 30, 76, "20"])
        writer.writerow([20, second_mask_id, "folio-J.jpg", 0, 0, 30, 76, "10"])

    with pytest.raises(rp.RachelPreprocessError, match="(?i)(jpeg|mask_id)"):
        rp.read_rachel_group(
            group,
            generator="gen2voronoi_1",
            config=rp.RachelPreprocessConfig(
                canvas_size=96,
                minimum_positive_seam=64,
                seam_bridge_radius=2,
            ),
        )


def test_group_reader_rejects_missing_or_inconsistent_image_name(
    tmp_path: Path,
) -> None:
    first = np.zeros((96, 96), dtype=np.bool_)
    second = np.zeros((96, 96), dtype=np.bool_)
    first[10:86, 5:35] = True
    second[10:86, 37:67] = True
    group = tmp_path / "gen2voronoi_1" / "no_erode" / "bad-lineage"
    group.mkdir(parents=True)
    _write_rgb_fragment(group / "0.jpg", first)
    _write_rgb_fragment(group / "1.jpg", second)
    _write_raw_label_csv(
        group / "label.csv",
        masks=(first, second),
        neighbor_cells=("1", "0"),
        image_names=("folio-A.jpg", "folio-B.jpg"),
    )

    with pytest.raises(rp.RachelPreprocessError, match="image_name"):
        rp.read_rachel_group(
            group,
            generator="gen2voronoi_1",
            config=rp.RachelPreprocessConfig(
                canvas_size=96,
                minimum_positive_seam=64,
                seam_bridge_radius=2,
            ),
        )


def test_group_reader_does_not_silently_union_one_sided_neighbor_label(
    tmp_path: Path,
) -> None:
    first = np.zeros((96, 96), dtype=np.bool_)
    second = np.zeros((96, 96), dtype=np.bool_)
    first[10:86, 5:35] = True
    second[10:86, 37:67] = True
    group = tmp_path / "gen2voronoi_1" / "no_erode" / "asymmetric-label"
    group.mkdir(parents=True)
    _write_rgb_fragment(group / "0.jpg", first)
    _write_rgb_fragment(group / "1.jpg", second)
    _write_raw_label_csv(
        group / "label.csv",
        masks=(first, second),
        neighbor_cells=("1", ""),
        image_names=("folio-Y.jpg", "folio-Y.jpg"),
    )

    with pytest.raises(rp.RachelPreprocessError, match="asymmetric.*neighbor"):
        rp.read_rachel_group(
            group,
            generator="gen2voronoi_1",
            config=rp.RachelPreprocessConfig(
                canvas_size=96,
                minimum_positive_seam=64,
                seam_bridge_radius=2,
            ),
        )


@pytest.mark.parametrize("invalid_neighbor", ["0", "99", "[99]"])
def test_group_reader_rejects_self_or_unknown_neighbor_ids(
    tmp_path: Path,
    invalid_neighbor: str,
) -> None:
    first = np.zeros((64, 64), dtype=np.bool_)
    second = np.zeros((64, 64), dtype=np.bool_)
    first[5:45, 3:20] = True
    second[5:45, 40:57] = True
    group = tmp_path / "gen2voronoi_1" / "no_erode" / invalid_neighbor.strip("[]")
    group.mkdir(parents=True)
    _write_rgb_fragment(group / "0.jpg", first)
    _write_rgb_fragment(group / "1.jpg", second)
    _write_raw_label_csv(
        group / "label.csv",
        masks=(first, second),
        neighbor_cells=(invalid_neighbor, ""),
        image_names=("folio-Z.jpg", "folio-Z.jpg"),
    )

    with pytest.raises(rp.RachelPreprocessError, match="neighbor"):
        rp.read_rachel_group(
            group,
            generator="gen2voronoi_1",
            config=rp.RachelPreprocessConfig(
                canvas_size=64,
                minimum_positive_seam=32,
                seam_bridge_radius=2,
            ),
        )


def test_rachel_module_has_no_shredding_preprocessor_dependency() -> None:
    source_path = Path(rp.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert all("shredding" not in name.casefold() for name in imported_modules)

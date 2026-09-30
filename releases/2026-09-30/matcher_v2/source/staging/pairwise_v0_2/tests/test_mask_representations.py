from __future__ import annotations

import numpy as np
import pytest

from staging.pairwise_v0_2.pairwise_data.mask_representations import (
    MaskRepresentationError,
    prepare_parent_mask_representations,
)


def _solid(canvas_shape, box, *, dtype=np.bool_):
    mask = np.zeros(canvas_shape, dtype=dtype)
    mask[box] = True if dtype == np.bool_ else 255
    return mask


def _foreground_box_shape(mask):
    rows, columns = np.nonzero(mask)
    return (
        int(rows.max()) - int(rows.min()) + 1,
        int(columns.max()) - int(columns.min()) + 1,
    )


def _foreground_center(mask):
    rows, columns = np.nonzero(mask)
    return float(rows.mean()), float(columns.mean())


def test_tight_crop_discards_origin_and_shared_scale_preserves_relative_size() -> None:
    result = prepare_parent_mask_representations(
        {
            "large": _solid((80, 90), (slice(17, 37), slice(23, 33))),
            "small": _solid((80, 90), (slice(2, 12), slice(61, 66))),
        },
        target_long_side=180,
        contour_width=2,
    )

    assert result.filled["large"].shape == (180, 180)
    assert result.filled["small"].shape == (180, 180)
    assert _foreground_box_shape(result.filled["large"]) == (40, 20)
    assert _foreground_box_shape(result.filled["small"]) == (20, 10)
    assert result.filled["large"].sum() == 40 * 20
    assert result.filled["small"].sum() == 20 * 10
    assert _foreground_center(result.filled["large"]) == (89.5, 89.5)
    assert _foreground_center(result.filled["small"]) == (89.5, 89.5)


def test_parent_translation_is_not_visible_after_tight_crop_and_center_pad() -> None:
    first = prepare_parent_mask_representations(
        {
            "a": _solid((800, 800), (slice(20, 120), slice(30, 90))),
            "b": _solid((800, 800), (slice(300, 380), slice(500, 540))),
        }
    )
    translated = prepare_parent_mask_representations(
        {
            "a": _solid((800, 800), (slice(520, 620), slice(630, 690))),
            "b": _solid((800, 800), (slice(40, 120), slice(100, 140))),
        }
    )

    for key in ("a", "b"):
        assert first.filled[key].shape == (800, 800)
        assert first.contour[key].shape == (800, 800)
        assert np.array_equal(first.filled[key], translated.filled[key])
        assert np.array_equal(first.contour[key], translated.contour[key])


def test_uint8_0_255_becomes_contiguous_readonly_bool_with_matched_views() -> None:
    result = prepare_parent_mask_representations(
        {
            "a": _solid((20, 20), (slice(2, 18), slice(3, 17)), dtype=np.uint8),
            "b": _solid((20, 20), (slice(4, 16), slice(5, 15)), dtype=np.uint8),
        },
        target_long_side=32,
        contour_width=3,
    )

    assert tuple(result.filled) == tuple(result.contour) == ("a", "b")
    for key in result.filled:
        assert result.filled[key].shape == result.contour[key].shape
        for value in (result.filled[key], result.contour[key]):
            assert value.dtype == np.bool_
            assert value.flags.c_contiguous
            assert not value.flags.writeable


def test_contour_is_requested_width_inside_filled_mask() -> None:
    solid = np.ones((31, 31), dtype=np.bool_)
    result = prepare_parent_mask_representations(
        {"a": solid, "b": solid}, target_long_side=31, contour_width=10
    )
    contour = result.contour["a"]

    assert contour[:10].all()
    assert contour[:, :10].all()
    assert not contour[10:21, 10:21].any()
    assert np.all(contour <= result.filled["a"])


def test_result_is_deterministic() -> None:
    masks = {
        "a": _solid((13, 19), (slice(1, 12), slice(2, 14))),
        "b": _solid((13, 19), (slice(3, 10), slice(5, 18))),
    }
    first = prepare_parent_mask_representations(masks, target_long_side=41)
    second = prepare_parent_mask_representations(masks, target_long_side=41)

    for key in masks:
        assert np.array_equal(first.filled[key], second.filled[key])
        assert np.array_equal(first.contour[key], second.contour[key])


def test_fragment_keys_are_canonical_and_parent_size_is_not_artificially_capped() -> (
    None
):
    masks = {
        key: _solid(
            (16, 16),
            (slice(1, 4 + index), slice(2, 6 + index)),
        )
        for index, key in enumerate(("f", "e", "d", "c", "b", "a"))
    }
    result = prepare_parent_mask_representations(masks, target_long_side=32)

    assert tuple(result.filled) == ("a", "b", "c", "d", "e", "f")
    assert tuple(result.contour) == tuple(result.filled)


def test_mismatched_parent_canvas_shapes_are_rejected() -> None:
    with pytest.raises(MaskRepresentationError, match="share a canvas shape"):
        prepare_parent_mask_representations(
            {
                "a": np.ones((8, 8), dtype=np.bool_),
                "b": np.ones((8, 9), dtype=np.bool_),
            }
        )


def test_empty_mask_is_rejected() -> None:
    with pytest.raises(MaskRepresentationError, match="no foreground"):
        prepare_parent_mask_representations(
            {"empty": np.zeros((8, 8), dtype=np.bool_), "valid": np.ones((8, 8), bool)}
        )


def test_non_string_fragment_identifier_is_rejected_cleanly() -> None:
    with pytest.raises(MaskRepresentationError, match="fragment identifiers"):
        prepare_parent_mask_representations(
            {1: np.ones((8, 8), dtype=np.bool_), "valid": np.ones((8, 8), bool)}
        )


@pytest.mark.parametrize(
    "bad_masks, message",
    [
        (
            {"a": np.ones((3, 3, 1), bool), "b": np.ones((3, 3), bool)},
            "two-dimensional",
        ),
        ({"a": np.full((3, 3), 7, np.uint8), "b": np.ones((3, 3), bool)}, "0 and 255"),
    ],
)
def test_invalid_shape_and_nonbinary_dtype_are_rejected(bad_masks, message) -> None:
    with pytest.raises(MaskRepresentationError, match=message):
        prepare_parent_mask_representations(bad_masks)


@pytest.mark.parametrize("width", [0, -1, 1.5, True])
def test_contour_width_must_be_a_positive_integer(width) -> None:
    with pytest.raises(MaskRepresentationError, match="positive integer"):
        prepare_parent_mask_representations(
            {"a": np.ones((3, 3), bool), "b": np.ones((3, 3), bool)},
            contour_width=width,
        )

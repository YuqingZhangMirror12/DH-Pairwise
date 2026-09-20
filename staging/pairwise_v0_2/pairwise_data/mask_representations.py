"""Leakage-safe mask and contour views at one shared parent scale.

All fragment masks from a simulated parent share one source canvas.  The
canvas is resized once, identically for every fragment, before each fragment
is tight-cropped and centred on a square model canvas.  Cropping removes the
ground-truth parent-canvas origin; common scaling preserves relative physical
fragment size.  No fragment is independently resized.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

import numpy as np
from scipy import ndimage


class MaskRepresentationError(ValueError):
    """Raised when a parent group cannot yield comparable representations."""


@dataclass(frozen=True)
class ParentMaskRepresentations:
    """Read-only representations keyed by the caller's fragment identifiers."""

    filled: Mapping[str, np.ndarray]
    contour: Mapping[str, np.ndarray]


def _binary_mask(value: np.ndarray, key: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise MaskRepresentationError("mask {!r} must be two-dimensional".format(key))
    if array.dtype == np.bool_:
        binary = array
    elif np.issubdtype(array.dtype, np.integer):
        if not np.logical_or(array == 0, array == 255).all():
            raise MaskRepresentationError(
                "integer mask {!r} must contain only 0 and 255".format(key)
            )
        binary = array == 255
    else:
        raise MaskRepresentationError(
            "mask {!r} must have bool or integer 0/255 dtype".format(key)
        )
    if not binary.any():
        raise MaskRepresentationError("mask {!r} has no foreground".format(key))
    return binary


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    return mask[
        int(rows.min()) : int(rows.max()) + 1,
        int(columns.min()) : int(columns.max()) + 1,
    ]


def _nearest_resize(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    output_height, output_width = shape
    input_height, input_width = mask.shape
    row_index = np.minimum(
        ((np.arange(output_height) + 0.5) * input_height / output_height).astype(int),
        input_height - 1,
    )
    column_index = np.minimum(
        ((np.arange(output_width) + 0.5) * input_width / output_width).astype(int),
        input_width - 1,
    )
    return mask[row_index[:, None], column_index[None, :]]


def _readonly_bool(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.bool_, order="C", copy=True)
    result.setflags(write=False)
    return result


def _center_pad(mask: np.ndarray, size: int) -> np.ndarray:
    """Centre one tight crop without resizing it or retaining its old origin."""

    height, width = mask.shape
    if height > size or width > size:
        raise MaskRepresentationError(
            "tight crop exceeds the requested model canvas"
        )
    row_start = (size - height) // 2
    column_start = (size - width) // 2
    result = np.zeros((size, size), dtype=np.bool_)
    result[row_start : row_start + height, column_start : column_start + width] = mask
    return result


def prepare_parent_mask_representations(
    masks: Mapping[str, np.ndarray],
    *,
    target_long_side: int = 800,
    contour_width: int = 10,
) -> ParentMaskRepresentations:
    """Create matched filled and internal-boundary model views for one parent.

    ``contour`` is exactly ``M & ~binary_erosion(M, 3x3, iterations=width,
    border_value=0)`` after shared-canvas nearest-neighbour resizing.  Each
    tight crop is centre-padded to ``[target_long_side, target_long_side]``.
    The caller must pass aligned masks from the same parent canvas.
    """

    if not isinstance(masks, Mapping):
        raise TypeError("masks must be a mapping keyed by fragment identifier")
    if len(masks) < 2:
        raise MaskRepresentationError("a parent group must contain at least 2 masks")
    if type(target_long_side) is not int or target_long_side <= 0:  # noqa: E721
        raise MaskRepresentationError("target_long_side must be a positive integer")
    if type(contour_width) is not int or contour_width <= 0:  # noqa: E721
        raise MaskRepresentationError("contour_width must be a positive integer")

    keys = tuple(masks)
    if any(not isinstance(key, str) or not key for key in keys):
        raise MaskRepresentationError("fragment identifiers must be non-empty strings")

    binary_masks: Dict[str, np.ndarray] = {}
    for key in sorted(keys):
        value = masks[key]
        binary_masks[key] = _binary_mask(value, key)

    canvas_shapes = {mask.shape for mask in binary_masks.values()}
    if len(canvas_shapes) != 1:
        raise MaskRepresentationError(
            "fragment masks from one parent must share a canvas shape"
        )
    canvas_shape = next(iter(canvas_shapes))
    canvas_scale = target_long_side / float(max(canvas_shape))
    normalized_canvas_shape = tuple(
        max(1, int(round(length * canvas_scale))) for length in canvas_shape
    )

    structure = np.ones((3, 3), dtype=np.bool_)
    filled: Dict[str, np.ndarray] = {}
    contour: Dict[str, np.ndarray] = {}
    for key, mask in binary_masks.items():
        resized_canvas = _nearest_resize(mask, normalized_canvas_shape)
        tight_crop = _tight_crop(resized_canvas)
        erosion = ndimage.binary_erosion(
            tight_crop,
            structure=structure,
            iterations=contour_width,
            border_value=0,
        )
        filled[key] = _readonly_bool(_center_pad(tight_crop, target_long_side))
        contour[key] = _readonly_bool(
            _center_pad(tight_crop & ~erosion, target_long_side)
        )

    return ParentMaskRepresentations(
        filled=MappingProxyType(filled),
        contour=MappingProxyType(contour),
    )


__all__ = [
    "MaskRepresentationError",
    "ParentMaskRepresentations",
    "prepare_parent_mask_representations",
]

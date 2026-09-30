#!/usr/bin/env python3
"""Read-only candidate-arc coverage preflight for Pairwise v0.2.

The audit uses only the frozen, training-only no-erosion Voronoi masks.  For
each deterministically selected positive pair it derives a *weak contact
proxy* from the masks' shared synthetic canvas, then compares that proxy with
the label-blind output of ``build_pair_candidates(..., direction=None)``.

This is deliberately not seam ground truth.  Ten-pixel dilation says only
that two raster masks are close enough under the legacy synthetic neighbor
rule; it does not recover a physical tear correspondence.  Raw overlaps of
1--32 pixels are retained as their own diagnostic slices and never silently
treated as clean physical seams.

The script is fail-closed to the canonical synthetic manifest/archive hashes,
does not know how to open the sealed real-world test manifest, never executes
a model, and emits no filesystem paths, group IDs, fragment IDs, or member
names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from staging.pairwise_v0_2.geometry import (
    CandidateBuilderConfig,
    PairCandidateResult,
    PairDirection,
    build_pair_candidates,
    geometry_config_fingerprint,
)
from staging.pairwise_v0_2.pairwise_data.lazy_mask_loader import (
    ArchiveSourceSpec,
    LazyMaskArchiveLoader,
    LazyMaskLoaderConfig,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    SYNTHETIC_MANIFEST_SCHEMA_VERSION,
    ArchiveBinding,
    MaskMemberRef,
)


SCHEMA_VERSION = "dunhuang-pairwise-candidate-arc-recall-audit/0.2"
AUDIT_VERSION = "candidate-arc-recall-audit-v0.2.0"
CANONICAL_DATASET_ID = "dunhuang_voronoi_masks_no_erode_v0_2"
CANONICAL_ARCHIVE_LOGICAL_ID = "local_asset://pairwise_mask_subset_v0_2"
CANONICAL_ARCHIVE_SHA256 = (
    "e44e4c0e5825d8577861d79eaf4063888a349c6ba3e531be6e41f2b7dde505df"
)
CANONICAL_ARCHIVE_BYTES = 161_535_087
CANONICAL_MANIFEST_SHA256 = (
    "e9772ec8e074873e4343ca42906a4056ea382536d2cbdf6ec881c21891587326"
)
LEGACY_DILATION_ITERATIONS = 10
LEGACY_MINIMUM_OVERLAP_PIXELS = 30
DEFAULT_PER_STRATUM_CAP = 4
ROUGH_DIRECTION_DOMINANCE_RATIO = 1.25
PATCH_RASTER_TOLERANCE_PX = math.sqrt(0.5)
OVERLAP_SLICE_ORDER = ("0", "1", "2-4", "5-8", "9-16", "17-32")

# Frozen before the first canonical run.  These gates concern only whether the
# candidate generator is safe to hand to a local matcher; they make no claim
# about learned matching accuracy.
GATE_GEOMETRY_SUCCESS_MIN = 0.99
GATE_ANY_DIRECTION_PATCH_PAIR_HIT_MIN = 0.95
GATE_ALL_DIRECTION_CONTACT_RECALL_MIN = 0.90
GATE_REPORTABLE_SLICE_MIN_PAIRS = 4
GATE_REPORTABLE_SLICE_PATCH_PAIR_HIT_MIN = 0.80


class CandidateArcAuditError(ValueError):
    """Raised when a frozen input or audit invariant fails closed."""


@dataclass(frozen=True)
class WeakContactProxy:
    """Mask-derived contact support, explicitly weaker than physical seam GT."""

    support_a_rc: np.ndarray
    support_b_rc: np.ndarray
    raw_overlap_pixels: int
    legacy_dilated_overlap_pixels: int
    contact_mode: str
    rough_direction: Optional[PairDirection]
    rough_direction_status: str
    rough_direction_source: str
    rough_vector_row_col: Tuple[float, float]
    rough_dominance_ratio: Optional[float]

    def __post_init__(self) -> None:
        for value in (self.support_a_rc, self.support_b_rc):
            if value.ndim != 2 or value.shape[1:] != (2,):
                raise ValueError("contact support must have shape [N, 2]")
            if value.dtype != np.float64 or value.flags.writeable:
                raise TypeError("contact support must be read-only float64")
            if not np.all(np.isfinite(value)):
                raise ValueError("contact support contains non-finite coordinates")
        if self.raw_overlap_pixels < 0 or self.legacy_dilated_overlap_pixels < 0:
            raise ValueError("contact counts cannot be negative")


@dataclass(frozen=True)
class _SelectedPair:
    group_id: str
    generator_family: str
    fragment_count: int
    overlap_slice: str
    fragment_a: MaskMemberRef
    fragment_b: MaskMemberRef
    expected_raw_overlap_pixels: int
    expected_legacy_dilated_overlap_pixels: int
    rank_token: str


def _readonly_points(mask: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    output = np.column_stack(
        (rows.astype(np.float64) + 0.5, columns.astype(np.float64) + 0.5)
    )
    output.setflags(write=False)
    return output


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _rank_token(group_id: str, fragment_a: int, fragment_b: int) -> str:
    payload = "candidate-arc-v0.2\0{}\0{}\0{}".format(
        group_id, fragment_a, fragment_b
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selection_fingerprint(tokens: Iterable[str]) -> str:
    values = sorted(tokens)
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    digest.update("count={}".format(len(values)).encode("ascii"))
    return digest.hexdigest()


def overlap_slice(raw_overlap_pixels: int) -> str:
    """Return the frozen 0/1/2--4/5--8/9--16/17--32 diagnostic slice."""

    if type(raw_overlap_pixels) is not int or raw_overlap_pixels < 0:  # noqa: E721
        raise ValueError("raw_overlap_pixels must be a non-negative int")
    if raw_overlap_pixels == 0:
        return "0"
    if raw_overlap_pixels == 1:
        return "1"
    if raw_overlap_pixels <= 4:
        return "2-4"
    if raw_overlap_pixels <= 8:
        return "5-8"
    if raw_overlap_pixels <= 16:
        return "9-16"
    if raw_overlap_pixels <= 32:
        return "17-32"
    raise ValueError("canonical no-erosion audit supports raw overlap only through 32")


def _rough_direction(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    support_a: np.ndarray,
    support_b: np.ndarray,
) -> Tuple[
    Optional[PairDirection],
    str,
    str,
    Tuple[float, float],
    Optional[float],
]:
    """Derive a model-independent, deliberately rough B-w.r.t.-A direction."""

    distance_to_b, nearest_b = ndimage.distance_transform_edt(
        ~mask_b, return_indices=True
    )
    distance_to_a, nearest_a = ndimage.distance_transform_edt(
        ~mask_a, return_indices=True
    )
    index_a = np.floor(support_a).astype(np.int64)
    index_b = np.floor(support_b).astype(np.int64)
    vectors: List[np.ndarray] = []
    if len(index_a):
        nearest = nearest_b[:, index_a[:, 0], index_a[:, 1]].T
        vectors.append(nearest.astype(np.float64) - index_a.astype(np.float64))
    if len(index_b):
        nearest = nearest_a[:, index_b[:, 0], index_b[:, 1]].T
        vectors.append(index_b.astype(np.float64) - nearest.astype(np.float64))
    nonzero = np.empty((0, 2), dtype=np.float64)
    if vectors:
        stacked = np.concatenate(vectors, axis=0)
        finite = np.all(np.isfinite(stacked), axis=1)
        nonzero = stacked[finite & (np.linalg.norm(stacked, axis=1) > 0.0)]

    source = "nearest_contact_vector_median"
    if len(nonzero):
        vector = np.median(nonzero, axis=0)
    else:
        source = "foreground_centroid_fallback"
        vector = np.argwhere(mask_b).mean(axis=0) - np.argwhere(mask_a).mean(axis=0)

    row_value = float(vector[0])
    column_value = float(vector[1])
    if not np.isfinite(row_value) or not np.isfinite(column_value):
        return None, "degenerate", source, (row_value, column_value), None
    absolute_row = abs(row_value)
    absolute_column = abs(column_value)
    primary = max(absolute_row, absolute_column)
    secondary = min(absolute_row, absolute_column)
    if primary <= 1e-9:
        return None, "degenerate", source, (row_value, column_value), None
    dominance = primary / max(secondary, 1e-9)
    if absolute_column >= absolute_row:
        direction = (
            PairDirection.B_RIGHT_OF_A
            if column_value >= 0.0
            else PairDirection.B_LEFT_OF_A
        )
    else:
        direction = (
            PairDirection.B_BELOW_A
            if row_value >= 0.0
            else PairDirection.B_ABOVE_A
        )
    status = (
        "unambiguous"
        if dominance >= ROUGH_DIRECTION_DOMINANCE_RATIO
        else "diagonal_ambiguous"
    )
    # Keep the EDT arrays live through vector extraction; their values also
    # make the intended nearest-contact construction explicit to readers.
    del distance_to_a, distance_to_b
    return direction, status, source, (row_value, column_value), float(dominance)


def derive_weak_contact_proxy(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    dilation_iterations: int = LEGACY_DILATION_ITERATIONS,
) -> WeakContactProxy:
    """Derive symmetric boundary support under the legacy dilation radius.

    The legacy label itself dilates A only.  The audit recomputes that exact
    count for integrity, but uses symmetric A/B boundary supports so recall is
    not an artifact of pair ordering.
    """

    left = np.asarray(mask_a)
    right = np.asarray(mask_b)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape != right.shape
        or left.dtype != np.bool_
        or right.dtype != np.bool_
        or left.size == 0
    ):
        raise ValueError("contact proxy requires same-shape non-empty 2D bool masks")
    if not np.any(left) or not np.any(right):
        raise ValueError("contact proxy cannot use an empty foreground")
    if type(dilation_iterations) is not int or dilation_iterations < 1:  # noqa: E721
        raise ValueError("dilation_iterations must be a positive int")

    structure = np.ones((3, 3), dtype=bool)
    raw_overlap = int(np.count_nonzero(left & right))
    if raw_overlap > 32:
        raise ValueError("raw overlap exceeds the frozen canonical 1--32 range")
    dilated_a = ndimage.binary_dilation(
        left, structure=structure, iterations=dilation_iterations
    )
    dilated_b = ndimage.binary_dilation(
        right, structure=structure, iterations=dilation_iterations
    )
    legacy_overlap = int(np.count_nonzero(dilated_a & right))
    boundary_a = left & ~ndimage.binary_erosion(
        left, structure=structure, border_value=0
    )
    boundary_b = right & ~ndimage.binary_erosion(
        right, structure=structure, border_value=0
    )
    support_a_mask = boundary_a & dilated_b
    support_b_mask = boundary_b & dilated_a
    support_a = _readonly_points(support_a_mask)
    support_b = _readonly_points(support_b_mask)
    if not len(support_a) or not len(support_b):
        raise ValueError("legacy-positive pair has empty symmetric contact support")
    direction, status, source, vector, dominance = _rough_direction(
        left, right, support_a, support_b
    )
    return WeakContactProxy(
        support_a_rc=support_a,
        support_b_rc=support_b,
        raw_overlap_pixels=raw_overlap,
        legacy_dilated_overlap_pixels=legacy_overlap,
        contact_mode=(
            "disjoint_dilation_proxy"
            if raw_overlap == 0
            else "raw_overlap_1_32_diagnostic_proxy"
        ),
        rough_direction=direction,
        rough_direction_status=status,
        rough_direction_source=source,
        rough_vector_row_col=vector,
        rough_dominance_ratio=dominance,
    )


def diagnose_pixel_cell_topology(mask: np.ndarray) -> Mapping[str, Any]:
    """Count raster saddles that can branch the B1 pixel-cell edge graph.

    B1 stores one incoming/outgoing edge per grid vertex.  A 2x2 checkerboard
    saddle has two foreground and two background cells meeting at one vertex,
    so that representation becomes non-manifold.  This diagnostic selects the
    same largest 4-connected component and returns aggregate topology only.
    """

    value = np.asarray(mask)
    if value.ndim != 2 or value.dtype != np.bool_ or not np.any(value):
        raise ValueError("topology diagnostic requires a non-empty 2D bool mask")
    structure4 = ndimage.generate_binary_structure(2, 1)
    structure8 = ndimage.generate_binary_structure(2, 2)
    labels, component_count = ndimage.label(value, structure=structure4)
    counts = np.bincount(labels.reshape(-1), minlength=component_count + 1)
    objects = ndimage.find_objects(labels, max_label=component_count)
    candidates = []
    for label_index in range(1, component_count + 1):
        component_slice = objects[label_index - 1]
        if component_slice is None:
            continue
        bbox = (
            int(component_slice[0].start),
            int(component_slice[1].start),
            int(component_slice[0].stop),
            int(component_slice[1].stop),
        )
        candidates.append((-int(counts[label_index]), bbox, label_index))
    if not candidates:  # pragma: no cover - guarded by non-empty input
        raise RuntimeError("component labeling lost non-empty foreground")
    _, _, selected_label = min(candidates)
    component = labels == selected_label

    top_left = component[:-1, :-1]
    top_right = component[:-1, 1:]
    bottom_left = component[1:, :-1]
    bottom_right = component[1:, 1:]
    diagonal_main = top_left & bottom_right & ~top_right & ~bottom_left
    diagonal_cross = top_right & bottom_left & ~top_left & ~bottom_right
    saddle_kind = np.zeros(diagonal_main.shape, dtype=np.uint8)
    saddle_kind[diagonal_main] = 1
    saddle_kind[diagonal_cross] = 2
    saddle_positions = np.argwhere(saddle_kind > 0)

    background = ~component
    background4, background_count4 = ndimage.label(background, structure=structure4)
    background8, background_count8 = ndimage.label(background, structure=structure8)

    def exterior_labels(labeled: np.ndarray) -> set:
        border = np.concatenate(
            (
                labeled[0, :],
                labeled[-1, :],
                labeled[1:-1, 0],
                labeled[1:-1, -1],
            )
        )
        return {int(item) for item in np.unique(border) if int(item) != 0}

    exterior4 = exterior_labels(background4)
    exterior8 = exterior_labels(background8)
    context_counts = Counter()
    for row, column in saddle_positions:
        row_i = int(row)
        column_i = int(column)
        if saddle_kind[row_i, column_i] == 1:
            background_cells = (
                (row_i, column_i + 1),
                (row_i + 1, column_i),
            )
        else:
            background_cells = (
                (row_i, column_i),
                (row_i + 1, column_i + 1),
            )
        exterior_flags = [
            int(background4[cell]) in exterior4 for cell in background_cells
        ]
        if all(exterior_flags):
            context_counts["two_exterior_background_channels"] += 1
        elif any(exterior_flags):
            context_counts["exterior_hole_corner_contact"] += 1
        else:
            context_counts["interior_hole_channels"] += 1

    return {
        "foreground_component_count_4": int(component_count),
        "largest_component_pixels": int(np.count_nonzero(component)),
        "checkerboard_saddle_vertex_count": int(len(saddle_positions)),
        "checkerboard_saddle_context_counts": dict(sorted(context_counts.items())),
        "background_component_count_4": int(background_count4),
        "interior_hole_component_count_4": int(background_count4 - len(exterior4)),
        "background_component_count_8": int(background_count8),
        "interior_hole_component_count_8": int(background_count8 - len(exterior8)),
    }


def patch_sequence_contact_coverage(
    sequence: Any,
    support_rc: np.ndarray,
    *,
    raster_tolerance_px: float = PATCH_RASTER_TOLERANCE_PX,
) -> np.ndarray:
    """Return contact-support pixels covered by any emitted patch footprint."""

    points = np.asarray(support_rc, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
        raise ValueError("support_rc must be a non-empty [N, 2] array")
    if not np.isfinite(raster_tolerance_px) or raster_tolerance_px < 0.0:
        raise ValueError("raster_tolerance_px must be finite and non-negative")
    covered = np.zeros(len(points), dtype=bool)
    half_window = float(sequence.resolved_window_px) / 2.0 + raster_tolerance_px
    for patch in sequence.patches:
        center = np.asarray(patch.center_row_col, dtype=np.float64)
        tangent = np.asarray(patch.tangent_row_col, dtype=np.float64)
        inward = np.asarray(patch.inward_normal_row_col, dtype=np.float64)
        delta = points - center[None, :]
        covered |= (
            (np.abs(delta @ tangent) <= half_window)
            & (np.abs(delta @ inward) <= half_window)
        )
    covered.setflags(write=False)
    return covered


def _direction_metrics(
    result: PairCandidateResult,
    direction: PairDirection,
    support_a: np.ndarray,
    support_b: np.ndarray,
    cache_a: Dict[Tuple[str, int, int], np.ndarray],
    cache_b: Dict[Tuple[str, int, int], np.ndarray],
) -> Optional[Mapping[str, Any]]:
    candidates = result.candidates_for_direction(direction)
    if not candidates:
        return None

    def coverage_a(sequence: Any) -> np.ndarray:
        key = (sequence.side.value, sequence.run_index, sequence.scale_index)
        if key not in cache_a:
            cache_a[key] = patch_sequence_contact_coverage(sequence, support_a)
        return cache_a[key]

    def coverage_b(sequence: Any) -> np.ndarray:
        key = (sequence.side.value, sequence.run_index, sequence.scale_index)
        if key not in cache_b:
            cache_b[key] = patch_sequence_contact_coverage(sequence, support_b)
        return cache_b[key]

    union_a = np.zeros(len(support_a), dtype=bool)
    union_b = np.zeros(len(support_b), dtype=bool)
    run_cover_a: Dict[int, np.ndarray] = {}
    run_cover_b: Dict[int, np.ndarray] = {}
    scale_cover_a: Dict[int, np.ndarray] = {}
    scale_cover_b: Dict[int, np.ndarray] = {}
    scale_pair_hit: Dict[int, bool] = defaultdict(bool)
    candidate_pair_hit = False
    for candidate in candidates:
        cover_a = coverage_a(candidate.sequence_a)
        cover_b = coverage_b(candidate.sequence_b)
        union_a |= cover_a
        union_b |= cover_b
        run_cover_a.setdefault(
            candidate.sequence_a.run_index, np.zeros(len(support_a), dtype=bool)
        )
        run_cover_a[candidate.sequence_a.run_index] |= cover_a
        run_cover_b.setdefault(
            candidate.sequence_b.run_index, np.zeros(len(support_b), dtype=bool)
        )
        run_cover_b[candidate.sequence_b.run_index] |= cover_b
        scale = candidate.sequence_a.scale_index
        scale_cover_a.setdefault(scale, np.zeros(len(support_a), dtype=bool))
        scale_cover_b.setdefault(scale, np.zeros(len(support_b), dtype=bool))
        scale_cover_a[scale] |= cover_a
        scale_cover_b[scale] |= cover_b
        hit = bool(np.any(cover_a) and np.any(cover_b))
        scale_pair_hit[scale] = scale_pair_hit[scale] or hit
        candidate_pair_hit = candidate_pair_hit or hit

    best_run_a = max((int(value.sum()) for value in run_cover_a.values()), default=0)
    best_run_b = max((int(value.sum()) for value in run_cover_b.values()), default=0)
    scales = {}
    for scale in sorted(set(scale_cover_a) | set(scale_cover_b)):
        cover_a = scale_cover_a.get(scale, np.zeros(len(support_a), dtype=bool))
        cover_b = scale_cover_b.get(scale, np.zeros(len(support_b), dtype=bool))
        scales[str(scale)] = {
            "candidate_patch_pair_hit": bool(scale_pair_hit.get(scale, False)),
            "covered_pixels_a": int(cover_a.sum()),
            "covered_pixels_b": int(cover_b.sum()),
        }
    return {
        "candidate_patch_pair_hit": candidate_pair_hit,
        "facing_side_run_pair_hit": bool(
            any(np.any(value) for value in run_cover_a.values())
            and any(np.any(value) for value in run_cover_b.values())
        ),
        "covered_a": union_a,
        "covered_b": union_b,
        "best_single_run_pixels_a": best_run_a,
        "best_single_run_pixels_b": best_run_b,
        "contact_hit_run_count_a": sum(np.any(value) for value in run_cover_a.values()),
        "contact_hit_run_count_b": sum(np.any(value) for value in run_cover_b.values()),
        "scales": scales,
    }


def evaluate_candidate_output(
    result: PairCandidateResult,
    contact: WeakContactProxy,
    config: CandidateBuilderConfig,
) -> Mapping[str, Any]:
    """Evaluate one label-blind candidate output against one weak proxy."""

    support_a = contact.support_a_rc
    support_b = contact.support_b_rc
    zeros_a = np.zeros(len(support_a), dtype=bool)
    zeros_b = np.zeros(len(support_b), dtype=bool)
    scale_count = len(config.window_scale_fractions)
    if not result.ok:
        return {
            "geometry_success": False,
            "geometry_status": result.status.value,
            "geometry_failure_reason": result.failure_reason,
            "all_four_directions_emitted": False,
            "rough_direction_eligible": contact.rough_direction is not None,
            "rough_direction_unambiguous": (
                contact.rough_direction_status == "unambiguous"
            ),
            "rough_primary_direction_emitted": False,
            "rough_primary_facing_side_run_pair_hit": False,
            "rough_primary_candidate_patch_pair_hit": False,
            "any_direction_facing_side_run_pair_hit": False,
            "any_direction_candidate_patch_pair_hit": False,
            "covered_all_a": zeros_a,
            "covered_all_b": zeros_b,
            "covered_primary_a": zeros_a,
            "covered_primary_b": zeros_b,
            "best_single_run_pixels_a": 0,
            "best_single_run_pixels_b": 0,
            "contact_hit_run_count_a": 0,
            "contact_hit_run_count_b": 0,
            "scale_covered_a": {
                str(index): zeros_a.copy() for index in range(scale_count)
            },
            "scale_covered_b": {
                str(index): zeros_b.copy() for index in range(scale_count)
            },
            "scale_candidate_patch_pair_hit": {
                str(index): False for index in range(scale_count)
            },
        }

    expected = tuple(direction.value for direction in PairDirection)
    if result.quality.requested_directions != expected:
        raise CandidateArcAuditError(
            "candidate builder was not called with direction=None"
        )
    if result.quality.rotation_search_performed:
        raise CandidateArcAuditError("candidate audit unexpectedly searched rotation")

    cache_a: Dict[Tuple[str, int, int], np.ndarray] = {}
    cache_b: Dict[Tuple[str, int, int], np.ndarray] = {}
    details = {
        direction: _direction_metrics(
            result, direction, support_a, support_b, cache_a, cache_b
        )
        for direction in PairDirection
    }
    emitted = {direction for direction, detail in details.items() if detail is not None}
    covered_all_a = np.zeros(len(support_a), dtype=bool)
    covered_all_b = np.zeros(len(support_b), dtype=bool)
    best_single_a = 0
    best_single_b = 0
    hit_runs_a = 0
    hit_runs_b = 0
    any_run_pair_hit = False
    any_patch_pair_hit = False
    scale_covered_a = {
        str(index): np.zeros(len(support_a), dtype=bool)
        for index in range(scale_count)
    }
    scale_covered_b = {
        str(index): np.zeros(len(support_b), dtype=bool)
        for index in range(scale_count)
    }
    scale_pair_hit = {str(index): False for index in range(scale_count)}
    for detail in details.values():
        if detail is None:
            continue
        covered_all_a |= detail["covered_a"]
        covered_all_b |= detail["covered_b"]
        best_single_a = max(best_single_a, detail["best_single_run_pixels_a"])
        best_single_b = max(best_single_b, detail["best_single_run_pixels_b"])
        hit_runs_a += int(detail["contact_hit_run_count_a"])
        hit_runs_b += int(detail["contact_hit_run_count_b"])
        any_run_pair_hit = any_run_pair_hit or bool(
            detail["facing_side_run_pair_hit"]
        )
        any_patch_pair_hit = any_patch_pair_hit or bool(
            detail["candidate_patch_pair_hit"]
        )
        for scale, values in detail["scales"].items():
            # The per-direction arrays are recoverable from the shared cache;
            # union the sequences for the requested scale directly.
            for candidate in result.candidates_for_direction(
                next(
                    direction
                    for direction, candidate_detail in details.items()
                    if candidate_detail is detail
                )
            ):
                if str(candidate.sequence_a.scale_index) != scale:
                    continue
                key_a = (
                    candidate.sequence_a.side.value,
                    candidate.sequence_a.run_index,
                    candidate.sequence_a.scale_index,
                )
                key_b = (
                    candidate.sequence_b.side.value,
                    candidate.sequence_b.run_index,
                    candidate.sequence_b.scale_index,
                )
                scale_covered_a[scale] |= cache_a[key_a]
                scale_covered_b[scale] |= cache_b[key_b]
            scale_pair_hit[scale] = scale_pair_hit[scale] or bool(
                values["candidate_patch_pair_hit"]
            )

    primary = (
        None
        if contact.rough_direction is None
        else details.get(contact.rough_direction)
    )
    return {
        "geometry_success": True,
        "geometry_status": result.status.value,
        "geometry_failure_reason": None,
        "all_four_directions_emitted": emitted == set(PairDirection),
        "rough_direction_eligible": contact.rough_direction is not None,
        "rough_direction_unambiguous": (
            contact.rough_direction_status == "unambiguous"
        ),
        "rough_primary_direction_emitted": primary is not None,
        "rough_primary_facing_side_run_pair_hit": bool(
            primary is not None and primary["facing_side_run_pair_hit"]
        ),
        "rough_primary_candidate_patch_pair_hit": bool(
            primary is not None and primary["candidate_patch_pair_hit"]
        ),
        "any_direction_facing_side_run_pair_hit": any_run_pair_hit,
        "any_direction_candidate_patch_pair_hit": any_patch_pair_hit,
        "covered_all_a": covered_all_a,
        "covered_all_b": covered_all_b,
        "covered_primary_a": (
            zeros_a if primary is None else primary["covered_a"]
        ),
        "covered_primary_b": (
            zeros_b if primary is None else primary["covered_b"]
        ),
        "best_single_run_pixels_a": best_single_a,
        "best_single_run_pixels_b": best_single_b,
        "contact_hit_run_count_a": hit_runs_a,
        "contact_hit_run_count_b": hit_runs_b,
        "scale_covered_a": scale_covered_a,
        "scale_covered_b": scale_covered_b,
        "scale_candidate_patch_pair_hit": scale_pair_hit,
    }


def _member_reference(
    binding: ArchiveBinding,
    group: Mapping[str, Any],
    member: Mapping[str, Any],
) -> MaskMemberRef:
    group_id = str(group["group_id"])
    fragment_id = int(member["fragment_id"])
    component_digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    return MaskMemberRef(
        binding=binding,
        archive_member=str(member["archive_member"]),
        fragment_id="{}/fragment/{}".format(group_id, fragment_id),
        dataset_id=CANONICAL_DATASET_ID,
        canonical_group_id=group_id,
        component_id="synthetic/group-sha256/{}".format(component_digest),
        split="train",
        threshold_rule=str(member["threshold_rule"]),
        content_sha256=str(member["content_sha256"]),
    )


def _load_frozen_manifest(
    manifest_path: Path,
) -> Tuple[ArchiveBinding, List[Mapping[str, Any]]]:
    observed_hash = _sha256_path(manifest_path)
    if observed_hash != CANONICAL_MANIFEST_SHA256:
        raise CandidateArcAuditError("synthetic manifest SHA-256 mismatch")
    groups: List[Mapping[str, Any]] = []
    archive_binding: Optional[ArchiveBinding] = None
    seen_group_ids = set()
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                group = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CandidateArcAuditError(
                    "invalid JSONL at line {}".format(line_number)
                ) from exc
            if not isinstance(group, Mapping):
                raise CandidateArcAuditError("manifest row must be an object")
            if group.get("schema_version") != SYNTHETIC_MANIFEST_SCHEMA_VERSION:
                raise CandidateArcAuditError("unsupported synthetic manifest schema")
            if group.get("dataset_id") != CANONICAL_DATASET_ID:
                raise CandidateArcAuditError("unexpected synthetic dataset identity")
            if group.get("no_erode") is not True:
                raise CandidateArcAuditError("candidate audit requires no_erode=true")
            group_id = group.get("group_id")
            if not isinstance(group_id, str) or not group_id or group_id in seen_group_ids:
                raise CandidateArcAuditError("invalid or duplicate synthetic group ID")
            seen_group_ids.add(group_id)
            archive = group.get("archive")
            if not isinstance(archive, Mapping):
                raise CandidateArcAuditError("synthetic archive binding is missing")
            try:
                binding = ArchiveBinding(
                    logical_id=str(archive["logical_id"]),
                    archive_format=str(archive["format"]),
                    sha256=str(archive["sha256"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise CandidateArcAuditError("invalid synthetic archive binding") from exc
            if (
                binding.logical_id != CANONICAL_ARCHIVE_LOGICAL_ID
                or binding.archive_format != "zip"
                or binding.sha256 != CANONICAL_ARCHIVE_SHA256
                or int(archive.get("bytes", -1)) != CANONICAL_ARCHIVE_BYTES
            ):
                raise CandidateArcAuditError("manifest is not bound to canonical subset")
            if archive_binding is None:
                archive_binding = binding
            elif binding != archive_binding:
                raise CandidateArcAuditError("archive binding changes within manifest")
            groups.append(group)
    if archive_binding is None or len(groups) != 5000:
        raise CandidateArcAuditError("frozen manifest must contain exactly 5,000 groups")
    return archive_binding, groups


def _select_pairs(
    groups: Sequence[Mapping[str, Any]],
    binding: ArchiveBinding,
    per_stratum_cap: int,
) -> Tuple[Tuple[_SelectedPair, ...], Mapping[str, Any]]:
    if type(per_stratum_cap) is not int or per_stratum_cap < 1:  # noqa: E721
        raise ValueError("per_stratum_cap must be a positive int")
    eligible: List[_SelectedPair] = []
    eligible_groups = set()
    quarantined_groups = 0
    for group in groups:
        quarantine = group.get("quarantine")
        if not isinstance(quarantine, Mapping):
            raise CandidateArcAuditError("group lacks quarantine status")
        status = quarantine.get("status")
        if status == "quarantined":
            quarantined_groups += 1
            continue
        if status != "retained":
            raise CandidateArcAuditError("unsupported quarantine status")
        group_id = str(group["group_id"])
        eligible_groups.add(group_id)
        members = group.get("members")
        measurements = group.get("pair_measurements")
        if not isinstance(members, list) or not isinstance(measurements, list):
            raise CandidateArcAuditError("group members/measurements are invalid")
        by_id = {int(member["fragment_id"]): member for member in members}
        fragment_count = int(group.get("fragment_count", -1))
        if len(by_id) != fragment_count or not 2 <= fragment_count <= 5:
            raise CandidateArcAuditError("synthetic group fragment count is invalid")
        generator = str(group.get("generator_family", ""))
        if not generator:
            raise CandidateArcAuditError("generator family is missing")
        for measurement in measurements:
            if measurement.get("is_neighbor") is not True:
                continue
            first = int(measurement["fragment_a"])
            second = int(measurement["fragment_b"])
            raw_overlap = int(measurement["raw_overlap_pixels"])
            legacy_overlap = int(measurement["legacy_dilated_overlap_pixels"])
            if first not in by_id or second not in by_id or first >= second:
                raise CandidateArcAuditError("positive measurement identity is invalid")
            if legacy_overlap < LEGACY_MINIMUM_OVERLAP_PIXELS:
                raise CandidateArcAuditError("positive label violates legacy threshold")
            token = _rank_token(group_id, first, second)
            eligible.append(
                _SelectedPair(
                    group_id=group_id,
                    generator_family=generator,
                    fragment_count=fragment_count,
                    overlap_slice=overlap_slice(raw_overlap),
                    fragment_a=_member_reference(binding, group, by_id[first]),
                    fragment_b=_member_reference(binding, group, by_id[second]),
                    expected_raw_overlap_pixels=raw_overlap,
                    expected_legacy_dilated_overlap_pixels=legacy_overlap,
                    rank_token=token,
                )
            )

    strata: Dict[Tuple[str, int, str], List[_SelectedPair]] = defaultdict(list)
    for record in eligible:
        strata[
            (
                record.generator_family,
                record.fragment_count,
                record.overlap_slice,
            )
        ].append(record)
    selected: List[_SelectedPair] = []
    eligible_strata_counts = {}
    selected_strata_counts = {}
    for key in sorted(strata):
        ordered = sorted(strata[key], key=lambda record: record.rank_token)
        retained = ordered[:per_stratum_cap]
        selected.extend(retained)
        label = "{}|{}|{}".format(*key)
        eligible_strata_counts[label] = len(ordered)
        selected_strata_counts[label] = len(retained)
    selected.sort(key=lambda record: record.rank_token)
    return tuple(selected), {
        "eligible_positive_pairs": len(eligible),
        "eligible_retained_groups": len(eligible_groups),
        "quarantined_groups_excluded": quarantined_groups,
        "eligible_stratum_counts": eligible_strata_counts,
        "selected_stratum_counts": selected_strata_counts,
    }


def _equal_group_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["_group_id"]) for row in rows)
    group_count = len(counts)
    return np.asarray(
        [1.0 / (group_count * counts[str(row["_group_id"])]) for row in rows],
        dtype=np.float64,
    )


def _rate_summary(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    eligibility_field: Optional[str] = None,
) -> Mapping[str, Any]:
    selected = [
        row
        for row in rows
        if eligibility_field is None or bool(row[eligibility_field])
    ]
    if not selected:
        return {"denominator_pairs": 0, "row_rate": None, "equal_group_rate": None}
    values = np.asarray([bool(row[field]) for row in selected], dtype=np.float64)
    weights = _equal_group_weights(selected)
    return {
        "denominator_pairs": len(selected),
        "denominator_groups": len({str(row["_group_id"]) for row in selected}),
        "numerator_pairs": int(values.sum()),
        "row_rate": float(values.mean()),
        "equal_group_rate": float(np.sum(values * weights)),
    }


def _mean_summary(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    eligibility_field: Optional[str] = None,
) -> Mapping[str, Any]:
    selected = [
        row
        for row in rows
        if eligibility_field is None or bool(row[eligibility_field])
    ]
    if not selected:
        return {"denominator_pairs": 0, "row_mean": None, "equal_group_mean": None}
    values = np.asarray([float(row[field]) for row in selected], dtype=np.float64)
    weights = _equal_group_weights(selected)
    return {
        "denominator_pairs": len(selected),
        "denominator_groups": len({str(row["_group_id"]) for row in selected}),
        "row_mean": float(values.mean()),
        "equal_group_mean": float(np.sum(values * weights)),
        "minimum": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], scale_count: int) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty candidate audit slice")
    boolean_metrics = (
        "geometry_success",
        "all_four_directions_emitted",
        "rough_primary_direction_emitted",
        "rough_primary_facing_side_run_pair_hit",
        "rough_primary_candidate_patch_pair_hit",
        "any_direction_facing_side_run_pair_hit",
        "any_direction_candidate_patch_pair_hit",
        "contact_spans_multiple_hit_runs",
    )
    topology_diagnostics = [
        row["_non_manifold_topology_diagnostic"]
        for row in rows
        if row.get("_non_manifold_topology_diagnostic") is not None
    ]
    topology_context_counts = Counter()
    for diagnostic in topology_diagnostics:
        topology_context_counts.update(
            diagnostic["checkerboard_saddle_context_counts"]
        )
    saddle_counts = [
        int(item["checkerboard_saddle_vertex_count"])
        for item in topology_diagnostics
    ]
    output: Dict[str, Any] = {
        "denominators": {
            "pairs": len(rows),
            "groups": len({str(row["_group_id"]) for row in rows}),
            "contact_support_pixels_a": sum(int(row["support_pixels_a"]) for row in rows),
            "contact_support_pixels_b": sum(int(row["support_pixels_b"]) for row in rows),
            "rough_direction_eligible_pairs": sum(
                bool(row["rough_direction_eligible"]) for row in rows
            ),
            "rough_direction_unambiguous_pairs": sum(
                bool(row["rough_direction_unambiguous"]) for row in rows
            ),
        },
        "rates": {},
        "contact_arc_recall": {},
        "scale_patch_coverage": {},
        "geometry_status_counts": dict(
            sorted(Counter(str(row["geometry_status"]) for row in rows).items())
        ),
        "geometry_failure_reason_counts": dict(
            sorted(
                Counter(
                    str(row["geometry_failure_reason"])
                    for row in rows
                    if row["geometry_failure_reason"] is not None
                ).items()
            )
        ),
        "non_manifold_boundary_topology_diagnostic": {
            "denominator_failed_fragments": len(topology_diagnostics),
            "failed_fragments_with_checkerboard_saddle": sum(
                value > 0 for value in saddle_counts
            ),
            "checkerboard_saddle_vertices_total": sum(saddle_counts),
            "checkerboard_saddle_vertices_minimum": (
                min(saddle_counts) if saddle_counts else None
            ),
            "checkerboard_saddle_vertices_median": (
                float(np.median(saddle_counts)) if saddle_counts else None
            ),
            "checkerboard_saddle_vertices_maximum": (
                max(saddle_counts) if saddle_counts else None
            ),
            "checkerboard_saddle_context_counts": dict(
                sorted(topology_context_counts.items())
            ),
            "failed_fragments_with_4_connected_interior_hole": sum(
                int(item["interior_hole_component_count_4"]) > 0
                for item in topology_diagnostics
            ),
            "failed_fragments_with_8_connected_interior_hole": sum(
                int(item["interior_hole_component_count_8"]) > 0
                for item in topology_diagnostics
            ),
        },
    }
    for metric in boolean_metrics:
        eligibility = (
            "rough_direction_eligible"
            if metric.startswith("rough_primary")
            else None
        )
        output["rates"][metric] = _rate_summary(
            rows, metric, eligibility_field=eligibility
        )
    for metric in (
        "all_direction_contact_recall_combined",
        "all_direction_contact_recall_min_fragment",
        "rough_primary_contact_recall_combined",
        "best_single_run_contact_recall_combined",
        "multi_run_union_gain_combined",
    ):
        eligibility = (
            "rough_direction_eligible"
            if metric.startswith("rough_primary")
            else None
        )
        output["contact_arc_recall"][metric] = _mean_summary(
            rows, metric, eligibility_field=eligibility
        )
    total_support = sum(
        int(row["support_pixels_a"]) + int(row["support_pixels_b"]) for row in rows
    )
    output["contact_arc_recall"]["all_direction_pixel_micro"] = {
        "numerator_covered_pixels": sum(
            int(row["covered_all_pixels_a"]) + int(row["covered_all_pixels_b"])
            for row in rows
        ),
        "denominator_contact_support_pixels": total_support,
    }
    micro = output["contact_arc_recall"]["all_direction_pixel_micro"]
    micro["recall"] = (
        float(micro["numerator_covered_pixels"] / total_support)
        if total_support
        else None
    )
    for scale in range(scale_count):
        output["scale_patch_coverage"][str(scale)] = {
            "candidate_patch_pair_hit": _rate_summary(
                rows, "scale_{}_candidate_patch_pair_hit".format(scale)
            ),
            "contact_recall_combined": _mean_summary(
                rows, "scale_{}_contact_recall_combined".format(scale)
            ),
        }
    successful = [row for row in rows if bool(row["geometry_success"])]
    output["conditional_on_geometry_success"] = {
        "warning": (
            "diagnostic only; headline gate retains geometry failures as zero "
            "coverage to avoid survivorship bias"
        ),
        "denominator_pairs": len(successful),
        "any_direction_candidate_patch_pair_hit": (
            _rate_summary(successful, "any_direction_candidate_patch_pair_hit")
            if successful
            else None
        ),
        "all_four_directions_emitted": (
            _rate_summary(successful, "all_four_directions_emitted")
            if successful
            else None
        ),
        "all_direction_contact_recall_combined": (
            _mean_summary(successful, "all_direction_contact_recall_combined")
            if successful
            else None
        ),
    }
    return output


def _counts_by(rows: Sequence[Mapping[str, Any]], field: str) -> Mapping[str, int]:
    counts = Counter(str(row[field]) for row in rows)
    return dict(sorted(counts.items()))


def _slice_results(
    rows: Sequence[Mapping[str, Any]], field: str, scale_count: int
) -> Mapping[str, Any]:
    buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[field])].append(row)
    return {
        name: _aggregate(values, scale_count)
        for name, values in sorted(buckets.items())
    }


def _pair_row(
    selected: _SelectedPair,
    contact: WeakContactProxy,
    evaluation: Mapping[str, Any],
    config: CandidateBuilderConfig,
) -> Mapping[str, Any]:
    support_a = len(contact.support_a_rc)
    support_b = len(contact.support_b_rc)
    covered_all_a = int(np.count_nonzero(evaluation["covered_all_a"]))
    covered_all_b = int(np.count_nonzero(evaluation["covered_all_b"]))
    covered_primary_a = int(np.count_nonzero(evaluation["covered_primary_a"]))
    covered_primary_b = int(np.count_nonzero(evaluation["covered_primary_b"]))
    combined_denominator = support_a + support_b
    best_single = int(evaluation["best_single_run_pixels_a"]) + int(
        evaluation["best_single_run_pixels_b"]
    )
    row: Dict[str, Any] = {
        "_group_id": selected.group_id,
        "_non_manifold_topology_diagnostic": evaluation.get(
            "non_manifold_topology_diagnostic"
        ),
        "generator_family": selected.generator_family,
        "fragment_count": selected.fragment_count,
        "overlap_slice": selected.overlap_slice,
        "support_pixels_a": support_a,
        "support_pixels_b": support_b,
        "covered_all_pixels_a": covered_all_a,
        "covered_all_pixels_b": covered_all_b,
        "geometry_success": bool(evaluation["geometry_success"]),
        "geometry_status": str(evaluation["geometry_status"]),
        "geometry_failure_reason": evaluation["geometry_failure_reason"],
        "all_four_directions_emitted": bool(
            evaluation["all_four_directions_emitted"]
        ),
        "rough_direction_eligible": bool(evaluation["rough_direction_eligible"]),
        "rough_direction_unambiguous": bool(
            evaluation["rough_direction_unambiguous"]
        ),
        "rough_direction_status": contact.rough_direction_status,
        "rough_direction_source": contact.rough_direction_source,
        "rough_direction": (
            "none"
            if contact.rough_direction is None
            else contact.rough_direction.short_name
        ),
        "rough_primary_direction_emitted": bool(
            evaluation["rough_primary_direction_emitted"]
        ),
        "rough_primary_facing_side_run_pair_hit": bool(
            evaluation["rough_primary_facing_side_run_pair_hit"]
        ),
        "rough_primary_candidate_patch_pair_hit": bool(
            evaluation["rough_primary_candidate_patch_pair_hit"]
        ),
        "any_direction_facing_side_run_pair_hit": bool(
            evaluation["any_direction_facing_side_run_pair_hit"]
        ),
        "any_direction_candidate_patch_pair_hit": bool(
            evaluation["any_direction_candidate_patch_pair_hit"]
        ),
        "contact_spans_multiple_hit_runs": bool(
            int(evaluation["contact_hit_run_count_a"]) > 1
            or int(evaluation["contact_hit_run_count_b"]) > 1
        ),
        "all_direction_contact_recall_combined": (
            covered_all_a + covered_all_b
        )
        / combined_denominator,
        "all_direction_contact_recall_min_fragment": min(
            covered_all_a / support_a, covered_all_b / support_b
        ),
        "rough_primary_contact_recall_combined": (
            covered_primary_a + covered_primary_b
        )
        / combined_denominator,
        "best_single_run_contact_recall_combined": best_single
        / combined_denominator,
        "multi_run_union_gain_combined": (
            covered_all_a + covered_all_b - best_single
        )
        / combined_denominator,
    }
    for scale_index in range(len(config.window_scale_fractions)):
        key = str(scale_index)
        scale_a = int(np.count_nonzero(evaluation["scale_covered_a"][key]))
        scale_b = int(np.count_nonzero(evaluation["scale_covered_b"][key]))
        row["scale_{}_candidate_patch_pair_hit".format(scale_index)] = bool(
            evaluation["scale_candidate_patch_pair_hit"][key]
        )
        row["scale_{}_contact_recall_combined".format(scale_index)] = (
            scale_a + scale_b
        ) / combined_denominator
    return row


def _gate_decision(
    overall: Mapping[str, Any], by_overlap: Mapping[str, Any]
) -> Mapping[str, Any]:
    geometry = overall["rates"]["geometry_success"]
    patch_hit = overall["rates"]["any_direction_candidate_patch_pair_hit"]
    arc_recall = overall["contact_arc_recall"][
        "all_direction_contact_recall_combined"
    ]
    checks = [
        {
            "check": "geometry_success_conservative_rate",
            "threshold_min": GATE_GEOMETRY_SUCCESS_MIN,
            "observed": min(geometry["row_rate"], geometry["equal_group_rate"]),
        },
        {
            "check": "any_direction_same_candidate_patch_pair_hit_conservative_rate",
            "threshold_min": GATE_ANY_DIRECTION_PATCH_PAIR_HIT_MIN,
            "observed": min(patch_hit["row_rate"], patch_hit["equal_group_rate"]),
        },
        {
            "check": "all_direction_contact_arc_recall_conservative_mean",
            "threshold_min": GATE_ALL_DIRECTION_CONTACT_RECALL_MIN,
            "observed": min(arc_recall["row_mean"], arc_recall["equal_group_mean"]),
        },
    ]
    reportable_slices = []
    for name in OVERLAP_SLICE_ORDER:
        values = by_overlap.get(name)
        if values is None or values["denominators"]["pairs"] < GATE_REPORTABLE_SLICE_MIN_PAIRS:
            continue
        metric = values["rates"]["any_direction_candidate_patch_pair_hit"]
        reportable_slices.append(
            {
                "slice": name,
                "pairs": values["denominators"]["pairs"],
                "observed": min(metric["row_rate"], metric["equal_group_rate"]),
            }
        )
    slice_observed = min(
        (item["observed"] for item in reportable_slices), default=0.0
    )
    checks.append(
        {
            "check": "every_reportable_overlap_slice_patch_pair_hit_rate",
            "threshold_min": GATE_REPORTABLE_SLICE_PATCH_PAIR_HIT_MIN,
            "minimum_slice_pairs": GATE_REPORTABLE_SLICE_MIN_PAIRS,
            "observed": slice_observed,
            "reportable_slices": reportable_slices,
        }
    )
    for check in checks:
        check["pass"] = bool(check["observed"] >= check["threshold_min"])
    passed = all(check["pass"] for check in checks)
    return {
        "decision": "go_for_local_matcher_training" if passed else "no_go",
        "all_checks_pass": passed,
        "checks": checks,
        "scope": (
            "candidate-generation readiness on canonical synthetic masks only; "
            "not Pairwise accuracy and not real-world seam validity"
        ),
    }


def run_audit(
    *,
    manifest_path: Path,
    archive_path: Path,
    per_stratum_cap: int = DEFAULT_PER_STRATUM_CAP,
    geometry_config: CandidateBuilderConfig = CandidateBuilderConfig(),
) -> Mapping[str, Any]:
    """Run the frozen audit and return one portable aggregate receipt."""

    manifest = Path(manifest_path)
    archive = Path(archive_path)
    if not manifest.is_file() or not archive.is_file():
        raise CandidateArcAuditError("manifest and archive must be regular files")
    if archive.stat().st_size != CANONICAL_ARCHIVE_BYTES:
        raise CandidateArcAuditError("canonical archive byte count mismatch")
    binding, groups = _load_frozen_manifest(manifest)
    selected, selection_profile = _select_pairs(
        groups, binding, per_stratum_cap
    )
    if not selected:
        raise CandidateArcAuditError("deterministic candidate audit selection is empty")

    loader_config = LazyMaskLoaderConfig(
        cache_size=max(2, min(256, len(selected) * 2)),
        max_cache_pixels=max(1_280_000, min(163_840_000, len(selected) * 1_280_000)),
    )
    rows: List[Mapping[str, Any]] = []
    integrity_failures = Counter()
    with LazyMaskArchiveLoader(
        {
            binding.logical_id: ArchiveSourceSpec(
                binding=binding,
                source=archive,
            )
        },
        loader_config,
    ) as loader:
        for record in selected:
            mask_a = loader(record.fragment_a)
            mask_b = loader(record.fragment_b)
            contact = derive_weak_contact_proxy(mask_a, mask_b)
            if contact.raw_overlap_pixels != record.expected_raw_overlap_pixels:
                integrity_failures["raw_overlap_disagrees_with_manifest"] += 1
                raise CandidateArcAuditError("raw overlap disagrees with manifest")
            if (
                contact.legacy_dilated_overlap_pixels
                != record.expected_legacy_dilated_overlap_pixels
            ):
                integrity_failures["legacy_dilation_disagrees_with_manifest"] += 1
                raise CandidateArcAuditError("legacy dilation disagrees with manifest")
            # Critical label-blind boundary: no direction or label enters the
            # candidate builder.  Supervision is consulted only by this audit.
            result = build_pair_candidates(
                mask_a,
                mask_b,
                direction_b_wrt_a=None,
                config=geometry_config,
            )
            evaluation = dict(
                evaluate_candidate_output(result, contact, geometry_config)
            )
            topology_diagnostic = None
            if result.failure_reason == "a:non_manifold_boundary":
                topology_diagnostic = diagnose_pixel_cell_topology(mask_a)
            elif result.failure_reason == "b:non_manifold_boundary":
                topology_diagnostic = diagnose_pixel_cell_topology(mask_b)
            evaluation["non_manifold_topology_diagnostic"] = topology_diagnostic
            rows.append(_pair_row(record, contact, evaluation, geometry_config))
        loader_provenance = loader.provenance()

    scale_count = len(geometry_config.window_scale_fractions)
    overall = _aggregate(rows, scale_count)
    by_generator = _slice_results(rows, "generator_family", scale_count)
    by_fragment_count = _slice_results(rows, "fragment_count", scale_count)
    by_overlap = _slice_results(rows, "overlap_slice", scale_count)
    group_pair_counts = Counter(str(row["_group_id"]) for row in rows)
    pairs_per_group_histogram = Counter()
    for count in group_pair_counts.values():
        bucket = "1" if count == 1 else ("2-3" if count <= 3 else "4+")
        pairs_per_group_histogram[bucket] += 1
    failure_counts = Counter(
        str(row["geometry_failure_reason"])
        for row in rows
        if row["geometry_failure_reason"] is not None
    )
    status_counts = Counter(str(row["geometry_status"]) for row in rows)

    receipt: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit_version": AUDIT_VERSION,
        "status": "complete",
        "implementation": {
            "filename": Path(__file__).name,
            "sha256": _sha256_path(Path(__file__)),
        },
        "scope": {
            "purpose": "candidate-arc recall preflight before local matcher training",
            "data_role": "training-only synthetic no-erosion masks",
            "analysis_grain": "unordered positive fragment pair",
            "cluster_grain": "synthetic group",
            "candidate_builder_direction_argument": None,
            "candidate_generation_is_label_blind": True,
            "model_executed": False,
            "checkpoint_loaded": False,
            "real_dunhuang_test_opened": False,
            "production_or_mcp_modified": False,
        },
        "inputs": {
            "manifest": {
                "filename": manifest.name,
                "sha256": CANONICAL_MANIFEST_SHA256,
                "schema_version": SYNTHETIC_MANIFEST_SCHEMA_VERSION,
                "group_count": len(groups),
            },
            "archive": {
                "logical_id": binding.logical_id,
                "filename": archive.name,
                "format": binding.archive_format,
                "expected_sha256": binding.sha256,
                "bytes": CANONICAL_ARCHIVE_BYTES,
            },
        },
        "selection": {
            "strategy": (
                "within generator_family x fragment_count x raw_overlap_slice, "
                "take smallest sha256(candidate-arc-v0.2\\0group\\0a\\0b)"
            ),
            "selection_uses_positive_label": True,
            "selection_label_is_never_passed_to_candidate_builder": True,
            "per_stratum_cap": per_stratum_cap,
            "selected_pairs": len(rows),
            "selected_groups": len(group_pair_counts),
            "selection_fingerprint": _selection_fingerprint(
                record.rank_token for record in selected
            ),
            "counts_by_generator_family": _counts_by(rows, "generator_family"),
            "counts_by_fragment_count": _counts_by(rows, "fragment_count"),
            "counts_by_raw_overlap_slice": _counts_by(rows, "overlap_slice"),
            "pairs_per_group_histogram": dict(sorted(pairs_per_group_histogram.items())),
            **selection_profile,
        },
        "definitions": {
            "legacy_positive_rule": {
                "dilate_fragment": "a_only_for_manifest_integrity_reproduction",
                "iterations": LEGACY_DILATION_ITERATIONS,
                "kernel": "3x3_all_ones",
                "minimum_overlap_pixels_inclusive": LEGACY_MINIMUM_OVERLAP_PIXELS,
            },
            "weak_contact_proxy": (
                "for each side, 8-neighborhood one-pixel interior boundary pixels "
                "within symmetric Chebyshev radius 10 of the other raw mask"
            ),
            "raw_overlap_handling": (
                "0 is disjoint_dilation_proxy; 1-32 is retained and reported as "
                "raw_overlap_1_32_diagnostic_proxy, never asserted as a clean seam"
            ),
            "rough_direction_proxy": (
                "dominant axis of median nonzero nearest-contact B-minus-A vectors; "
                "foreground-centroid fallback only when overlap removes every "
                "nonzero vector; dominance below 1.25 is diagonal_ambiguous"
            ),
            "patch_coverage": (
                "contact pixel center lies inside any emitted oriented square patch "
                "footprint, expanded by sqrt(0.5) pixels for raster center/corner tolerance"
            ),
            "facing_side_run_pair_hit": (
                "at least one emitted run on each complementary facing side has a "
                "patch footprint intersecting its fragment's contact support"
            ),
            "candidate_patch_pair_hit": (
                "one same direction/run-pair/scale candidate has at least one contact "
                "support hit on both fragments; no descriptor correspondence is asserted"
            ),
            "contact_arc_recall": (
                "fraction of weak contact-support pixels covered by emitted patch footprints"
            ),
            "equal_group_aggregation": (
                "each synthetic group has equal total weight; pairs within group share it"
            ),
        },
        "geometry_contract": {
            "config": asdict(geometry_config),
            "config_fingerprint": geometry_config_fingerprint(geometry_config),
            "upright_orientation_assumed": True,
            "rotation_search_performed": False,
            "window_scale_fractions": list(geometry_config.window_scale_fractions),
        },
        "integrity": {
            "selected_pairs_recomputed_against_manifest": len(rows),
            "raw_overlap_mismatch_count": integrity_failures[
                "raw_overlap_disagrees_with_manifest"
            ],
            "legacy_dilation_mismatch_count": integrity_failures[
                "legacy_dilation_disagrees_with_manifest"
            ],
            "archive_sha256_verified_by_lazy_loader": True,
            "portable_output_contains_pair_rows": False,
            "portable_output_contains_group_or_fragment_ids": False,
            "portable_output_contains_archive_members": False,
        },
        "results": {
            "overall": overall,
            "by_generator_family": by_generator,
            "by_fragment_count": by_fragment_count,
            "by_raw_overlap_slice": by_overlap,
            "geometry_status_counts": dict(sorted(status_counts.items())),
            "geometry_failure_reason_counts": dict(sorted(failure_counts.items())),
            "rough_direction_status_counts": _counts_by(
                rows, "rough_direction_status"
            ),
            "rough_direction_source_counts": _counts_by(
                rows, "rough_direction_source"
            ),
            "rough_direction_counts": _counts_by(rows, "rough_direction"),
        },
        "decision": _gate_decision(overall, by_overlap),
        "loader_provenance": loader_provenance,
        "claim_boundary": [
            "This audit measures candidate coverage, not learned Pairwise accuracy.",
            "The mask-derived contact support is weak synthetic adjacency evidence, not a physical tear seam or correspondence map.",
            "A patch-pair hit proves only that a candidate survives preprocessing; it does not prove Sinkhorn can match it.",
            "Raw overlaps of 1-32 pixels may be raster/generator artifacts and are reported separately.",
            "The canonical 5k collection lacks trustworthy source-image lineage and remains training-only.",
            "No real Dunhuang image, historical validation image, model, checkpoint, or MCP was accessed by this audit.",
        ],
        "minimum_safe_remediation_if_non_manifold_boundary_is_observed": [
            "Repair the B1 pixel-cell boundary tracer at degree-four checkerboard saddle vertices; do not discard failed pairs or remove them from recall denominators.",
            "Preserve 4-connected foreground component selection and use a deterministic 4-foreground/8-background saddle-pairing rule (or an equivalently topology-tested marching-squares contour) to recover the external loop.",
            "Do not switch blindly to 8-connected foreground: that can merge diagonal islands and changes the frozen mask semantics.",
            "Do not treat hole removal alone as sufficient unless the aggregate saddle-context diagnostic shows failures are exclusively interior-hole artifacts.",
            "Require byte-stable manifold-mask outputs plus regression fixtures for exterior/exterior, exterior/hole, and interior-hole saddle contexts, then rerun this identical selection fingerprint before local matcher training.",
        ],
    }
    encoded = json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False)
    forbidden = (
        "/Users/",
        "/root/",
        "Dunhuang Dataset",
        "output/voronoi_masks/",
        '"fragment_id"',
        '"group_id"',
    )
    if any(token in encoded for token in forbidden):
        raise CandidateArcAuditError("portable receipt contains forbidden identity data")
    return receipt


def _default_paths() -> Tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[3]
    base = root / "staging/pairwise_v0_2"
    manifest_root = base / "manifests/dunhuang_pairwise_mask_subset_v0_2"
    return (
        manifest_root / "manifest/synthetic_groups.jsonl",
        manifest_root / "pairwise_mask_subset.zip",
        base / "preflight/candidate_arc_recall_audit_v0_2.json",
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    handle, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(destination.name), dir=str(destination.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    default_manifest, default_archive, default_output = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--archive", type=Path, default=default_archive)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--per-stratum-cap", type=int, default=DEFAULT_PER_STRATUM_CAP
    )
    args = parser.parse_args(argv)
    receipt = run_audit(
        manifest_path=args.manifest,
        archive_path=args.archive,
        per_stratum_cap=args.per_stratum_cap,
    )
    _write_json_atomic(args.output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "decision": receipt["decision"]["decision"],
                "selected_pairs": receipt["selection"]["selected_pairs"],
                "output_filename": args.output.name,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

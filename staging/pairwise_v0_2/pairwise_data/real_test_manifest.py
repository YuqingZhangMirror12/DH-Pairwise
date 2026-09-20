"""Build a fail-closed external-test manifest for real Dunhuang fragments.

The builder is deliberately read-only with respect to the supplied image roots.
It writes two portable JSON files plus a separate machine-local path receipt.
Portable records never contain absolute paths or the opaque PSD/source line from
``meta_data.txt``.

This module derives labels only from the curator category and the supplied
ground-truth placement (numeric bounding boxes plus fragment alpha masks).  It
does not inspect manuscript text, RGB similarity, or any model output.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


SCHEMA_VERSION = "pairwise-v0.2-real-external-test/0.1"
BUILDER_VERSION = "real-test-manifest-builder/0.1"
COLLECTION_NAMES = ("main", "supp")
CATEGORY_DIRECTORY_TO_ID = {
    "Ground Truth Simple": "ground_truth_simple",
    "Small": "small",
    "Issue": "issue",
    "No Conjunction": "no_conjunction",
}
CATEGORY_SEMANTICS = {
    "ground_truth_simple": "conjunction_positive",
    "small": "conjunction_positive",
    "issue": "review",
    "no_conjunction": "conjunction_negative_excluded",
}
CATEGORY_PRIORITY = {
    "ground_truth_simple": 0,
    "small": 1,
    "issue": 2,
    "no_conjunction": 3,
}

# These thresholds are label-construction constants, not tuned model thresholds.
STRONG_MAX_GAP_PX = 2.0
REVIEW_MAX_GAP_PX = 12.0
MIN_CONTACT_PIXELS = 8
MIN_CONTACT_BOUNDARY_FRACTION = 0.002
MAX_STRONG_OVERLAP_FRACTION = 0.01
ALPHA_FOREGROUND_THRESHOLD = 128
# The supplied collection contains legitimate very large plates.  Keep a finite
# Pillow safety ceiling, but place it above the audited 271,259,300-pixel
# maximum while still rejecting unexpectedly larger inputs.
TRUSTED_COLLECTION_MAX_IMAGE_PIXELS = 300_000_000
Image.MAX_IMAGE_PIXELS = TRUSTED_COLLECTION_MAX_IMAGE_PIXELS


class ManifestBuildError(ValueError):
    """Raised for invalid builder arguments, never for a single bad case."""


class ImagePixelLimitError(ValueError):
    """Raised when an image exceeds the explicit trusted-collection ceiling."""


@dataclass(frozen=True)
class NumericMetadata:
    canvas_wh: Tuple[int, int]
    bboxes_xyxy: Tuple[Tuple[int, int, int, int], ...]
    opaque_tail_line_count: int


@dataclass
class FragmentGeometry:
    fragment_id: int
    mask: np.ndarray
    boundary_xy: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    area_px: int


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _stable_uid(namespace: str, value: Any, length: int = 24) -> str:
    return "{}-{}".format(namespace, _sha256_bytes(_canonical_json(value))[:length])


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_int_fields(line: str, expected: int) -> Optional[Tuple[int, ...]]:
    fields = line.strip().split()
    if len(fields) != expected:
        return None
    try:
        return tuple(int(field) for field in fields)
    except ValueError:
        return None


def _parse_numeric_metadata(path: Path, fragment_count: int) -> NumericMetadata:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("metadata_not_utf8") from error
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 1 + fragment_count:
        raise ValueError("metadata_too_short")
    canvas = _parse_int_fields(lines[0], 2)
    if canvas is None or canvas[0] <= 0 or canvas[1] <= 0:
        raise ValueError("invalid_canvas")
    bboxes: List[Tuple[int, int, int, int]] = []
    for index in range(fragment_count):
        parsed = _parse_int_fields(lines[1 + index], 4)
        if parsed is None:
            raise ValueError("invalid_bbox_row")
        x1, y1, x2, y2 = parsed
        if x2 <= x1 or y2 <= y1:
            raise ValueError("nonpositive_bbox")
        bboxes.append((x1, y1, x2, y2))
    # The remaining line(s) are intentionally treated as opaque and never emitted.
    return NumericMetadata(
        canvas_wh=(canvas[0], canvas[1]),
        bboxes_xyxy=tuple(bboxes),
        opaque_tail_line_count=max(0, len(lines) - 1 - fragment_count),
    )


def _image_alpha(image: Image.Image) -> Tuple[bool, Optional[np.ndarray]]:
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if not has_alpha:
        return False, None
    if "A" in image.getbands():
        alpha_image = image.getchannel("A")
    else:
        alpha_image = image.convert("RGBA").getchannel("A")
    alpha = np.array(alpha_image, dtype=np.uint8, copy=True)
    return True, alpha >= ALPHA_FOREGROUND_THRESHOLD


def _boundary_points_xy(mask: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
    if mask.ndim != 2 or not bool(mask.any()):
        return np.empty((0, 2), dtype=np.int32)
    interior = np.zeros_like(mask, dtype=bool)
    if mask.shape[0] >= 3 and mask.shape[1] >= 3:
        interior[1:-1, 1:-1] = (
            mask[1:-1, 1:-1]
            & mask[:-2, 1:-1]
            & mask[2:, 1:-1]
            & mask[1:-1, :-2]
            & mask[1:-1, 2:]
        )
    rows, cols = np.nonzero(mask & ~interior)
    return np.column_stack((cols + int(x_offset), rows + int(y_offset))).astype(
        np.int32, copy=False
    )


def _overlap_pixels(a: FragmentGeometry, b: FragmentGeometry) -> int:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0
    a_crop = a.mask[y1 - ay1 : y2 - ay1, x1 - ax1 : x2 - ax1]
    b_crop = b.mask[y1 - by1 : y2 - by1, x1 - bx1 : x2 - bx1]
    return int(np.count_nonzero(a_crop & b_crop))


def _bbox_point_distance_lower_bound(a: np.ndarray, b: np.ndarray) -> float:
    amin, amax = a.min(axis=0), a.max(axis=0)
    bmin, bmax = b.min(axis=0), b.max(axis=0)
    delta = np.maximum(np.maximum(bmin - amax, amin - bmax), 0)
    return float(np.hypot(float(delta[0]), float(delta[1])))


def _pair_geometry_label(a: FragmentGeometry, b: FragmentGeometry) -> Dict[str, Any]:
    min_boundary = min(len(a.boundary_xy), len(b.boundary_xy))
    required_contact = max(
        MIN_CONTACT_PIXELS,
        int(math.ceil(MIN_CONTACT_BOUNDARY_FRACTION * min_boundary)),
    )
    overlap_px = _overlap_pixels(a, b)
    overlap_fraction = overlap_px / float(max(1, min(a.area_px, b.area_px)))
    lower_bound = _bbox_point_distance_lower_bound(a.boundary_xy, b.boundary_xy)

    min_gap: Optional[float]
    strong_mutual = 0
    review_mutual = 0
    if lower_bound > REVIEW_MAX_GAP_PX:
        min_gap = None
    else:
        tree_a = cKDTree(a.boundary_xy)
        tree_b = cKDTree(b.boundary_xy)
        distance_a, _ = tree_b.query(a.boundary_xy, k=1)
        distance_b, _ = tree_a.query(b.boundary_xy, k=1)
        min_gap = float(min(float(distance_a.min()), float(distance_b.min())))
        strong_mutual = int(
            min(
                np.count_nonzero(distance_a <= STRONG_MAX_GAP_PX),
                np.count_nonzero(distance_b <= STRONG_MAX_GAP_PX),
            )
        )
        review_mutual = int(
            min(
                np.count_nonzero(distance_a <= REVIEW_MAX_GAP_PX),
                np.count_nonzero(distance_b <= REVIEW_MAX_GAP_PX),
            )
        )

    large_overlap = overlap_fraction > MAX_STRONG_OVERLAP_FRACTION
    strong = (
        min_gap is not None
        and min_gap <= STRONG_MAX_GAP_PX
        and strong_mutual >= required_contact
        and not large_overlap
    )
    ambiguous = overlap_px > 0 or (
        min_gap is not None
        and min_gap <= REVIEW_MAX_GAP_PX
        and (review_mutual >= required_contact or min_gap <= STRONG_MAX_GAP_PX)
    )
    if strong:
        label, reason = "positive", "strong_alpha_boundary_contact"
    elif ambiguous:
        label, reason = (
            "review",
            (
                "overlap_exceeds_strong_contract"
                if large_overlap
                else "near_boundary_contact_ambiguous"
            ),
        )
    else:
        label, reason = "negative", "alpha_boundaries_separated"
    return {
        "label": label,
        "label_source": "gt_numeric_bbox_plus_fragment_alpha",
        "reason": reason,
        "diagnostics": {
            "bbox_boundary_distance_lower_bound_px": round(lower_bound, 4),
            "min_boundary_gap_px": None if min_gap is None else round(min_gap, 4),
            "mutual_boundary_pixels_within_strong_gap": strong_mutual,
            "mutual_boundary_pixels_within_review_gap": review_mutual,
            "required_contact_pixels": required_contact,
            "overlap_pixels": overlap_px,
            "overlap_fraction_of_smaller_fragment": round(overlap_fraction, 8),
        },
    }


def _is_connected(
    fragment_ids: Sequence[int], edges: Iterable[Tuple[int, int]]
) -> bool:
    ids = set(fragment_ids)
    if not ids:
        return False
    adjacency: Dict[int, set[int]] = {fragment_id: set() for fragment_id in ids}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    reached = {next(iter(ids))}
    queue = deque(reached)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            queue.append(neighbor)
    return reached == ids


def _safe_file_hash(path: Path) -> Optional[str]:
    try:
        return _sha256_file(path)
    except OSError:
        return None


def _inspect_occurrence(
    collection: str,
    category: str,
    group_dir: Path,
) -> Dict[str, Any]:
    errors: List[str] = []
    fragment_paths = sorted(
        (
            path
            for path in group_dir.glob("*.png")
            if path.name != "gt.png" and path.stem.isdigit()
        ),
        key=lambda path: int(path.stem),
    )
    fragment_ids = [int(path.stem) for path in fragment_paths]
    if fragment_ids != list(range(1, len(fragment_ids) + 1)):
        errors.append("fragment_ids_not_consecutive_from_one")
    if not fragment_paths:
        errors.append("no_numeric_fragment_png")

    metadata_path = group_dir / "meta_data.txt"
    gt_path = group_dir / "gt.png"
    metadata: Optional[NumericMetadata] = None
    if not metadata_path.is_file():
        errors.append("missing_metadata")
    else:
        try:
            metadata = _parse_numeric_metadata(metadata_path, len(fragment_paths))
        except (OSError, ValueError) as error:
            errors.append("metadata_error:{}".format(str(error)))

    image_records: List[Dict[str, Any]] = []
    geometries: List[FragmentGeometry] = []
    for ordinal, path in enumerate(fragment_paths):
        fragment_id = int(path.stem)
        content_sha256 = _safe_file_hash(path)
        record: Dict[str, Any] = {
            "fragment_id": fragment_id,
            "content_sha256": content_sha256,
        }
        try:
            with Image.open(path) as image:
                if image.size[0] * image.size[1] > TRUSTED_COLLECTION_MAX_IMAGE_PIXELS:
                    raise ImagePixelLimitError("image_pixel_limit_exceeded")
                image.load()
                record.update(
                    {
                        "format": image.format,
                        "mode": image.mode,
                        "size_wh": [int(image.size[0]), int(image.size[1])],
                    }
                )
                has_alpha, mask = _image_alpha(image)
            record["has_alpha"] = has_alpha
            if not has_alpha or mask is None:
                errors.append("fragment_{}_missing_alpha".format(fragment_id))
            elif not bool(mask.any()):
                errors.append("fragment_{}_empty_alpha".format(fragment_id))
            elif metadata is not None and ordinal < len(metadata.bboxes_xyxy):
                bbox = metadata.bboxes_xyxy[ordinal]
                expected_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                if tuple(record["size_wh"]) != expected_size:
                    errors.append("fragment_{}_bbox_size_mismatch".format(fragment_id))
                else:
                    boundary = _boundary_points_xy(mask, bbox[0], bbox[1])
                    if not len(boundary):
                        errors.append("fragment_{}_empty_boundary".format(fragment_id))
                    else:
                        record.update(
                            {
                                "alpha_mask_sha256": _sha256_bytes(mask.tobytes()),
                                "alpha_foreground_pixels": int(np.count_nonzero(mask)),
                                "bbox_xyxy": list(bbox),
                            }
                        )
                        geometries.append(
                            FragmentGeometry(
                                fragment_id=fragment_id,
                                mask=mask,
                                boundary_xy=boundary,
                                bbox_xyxy=bbox,
                                area_px=int(np.count_nonzero(mask)),
                            )
                        )
        except ImagePixelLimitError:
            errors.append("fragment_{}_image_pixel_limit_exceeded".format(fragment_id))
        except Exception as error:  # Pillow exposes several corruption exception types.
            errors.append(
                "fragment_{}_decode_error:{}".format(fragment_id, type(error).__name__)
            )
        image_records.append(record)

    gt_record: Dict[str, Any] = {"content_sha256": _safe_file_hash(gt_path)}
    if not gt_path.is_file():
        errors.append("missing_gt")
    else:
        try:
            with Image.open(gt_path) as gt:
                if gt.size[0] * gt.size[1] > TRUSTED_COLLECTION_MAX_IMAGE_PIXELS:
                    raise ImagePixelLimitError("image_pixel_limit_exceeded")
                gt_format = gt.format
                gt_mode = gt.mode
                gt_size = (int(gt.size[0]), int(gt.size[1]))
                gt.verify()
                gt_record.update(
                    {
                        "format": gt_format,
                        "mode": gt_mode,
                        "size_wh": [gt_size[0], gt_size[1]],
                    }
                )
            if (
                metadata is not None
                and tuple(gt_record["size_wh"]) != metadata.canvas_wh
            ):
                errors.append("gt_canvas_size_mismatch")
        except ImagePixelLimitError:
            errors.append("gt_image_pixel_limit_exceeded")
        except Exception as error:
            errors.append("gt_decode_error:{}".format(type(error).__name__))

    derived_pair_geometry: List[Dict[str, Any]] = []
    if (
        not errors
        and 2 <= len(fragment_paths) <= 5
        and len(geometries) == len(fragment_paths)
        and CATEGORY_SEMANTICS[category] == "conjunction_positive"
    ):
        geometry_by_id = {item.fragment_id: item for item in geometries}
        for fragment_a, fragment_b in itertools.combinations(sorted(geometry_by_id), 2):
            derived_pair_geometry.append(
                {
                    "fragment_a": fragment_a,
                    "fragment_b": fragment_b,
                    **_pair_geometry_label(
                        geometry_by_id[fragment_a], geometry_by_id[fragment_b]
                    ),
                }
            )

    numeric = None
    if metadata is not None:
        numeric = {
            "canvas_wh": list(metadata.canvas_wh),
            "bboxes_xyxy": [list(bbox) for bbox in metadata.bboxes_xyxy],
        }
    identity_payload = {
        "numeric_metadata": numeric,
        "fragment_content_sha256": [
            record.get("content_sha256") for record in image_records
        ],
        "gt_content_sha256": gt_record.get("content_sha256"),
    }
    exact_identity_evidence_complete = (
        numeric is not None
        and bool(image_records)
        and all(record.get("content_sha256") for record in image_records)
        and bool(gt_record.get("content_sha256"))
    )
    if errors and not exact_identity_evidence_complete:
        # Missing evidence must not make unrelated broken directories look like
        # exact duplicates.  Complete byte hashes remain safe to deduplicate even
        # when a PNG cannot decode or lacks alpha; incomplete cases get a
        # non-path, anonymous quarantine salt.
        identity_payload["invalid_occurrence_scope_sha256"] = _sha256_bytes(
            _canonical_json(
                {
                    "collection": collection,
                    "category": category,
                    "directory_name": group_dir.name,
                }
            )
        )
    case_uid = _stable_uid("dhcase", identity_payload)
    occurrence_uid = _stable_uid(
        "occ",
        {
            "collection": collection,
            "category": category,
            "directory_name": group_dir.name,
        },
    )
    return {
        "case_uid": case_uid,
        "content_signature_sha256": _sha256_bytes(_canonical_json(identity_payload)),
        "occurrence_uid": occurrence_uid,
        "collection": collection,
        "category": category,
        "category_semantics": CATEGORY_SEMANTICS[category],
        "group_dir": group_dir,
        "fragment_paths": fragment_paths,
        "metadata_path": metadata_path,
        "gt_path": gt_path,
        "fragment_count": len(fragment_paths),
        "numeric_metadata": numeric,
        "metadata_opaque_tail_line_count": (
            None if metadata is None else metadata.opaque_tail_line_count
        ),
        "fragments": image_records,
        "gt": gt_record,
        "derived_pair_geometry": derived_pair_geometry,
        "errors": sorted(set(errors)),
    }


def _portable_occurrence(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "occurrence_uid": record["occurrence_uid"],
        "collection": record["collection"],
        "category": record["category"],
        "category_semantics": record["category_semantics"],
    }


def _case_record(occurrences: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(
        occurrences,
        key=lambda item: (
            CATEGORY_PRIORITY[item["category"]],
            COLLECTION_NAMES.index(item["collection"]),
            item["occurrence_uid"],
        ),
    )
    canonical = ordered[0]
    categories = sorted({item["category"] for item in ordered})
    semantics = sorted({item["category_semantics"] for item in ordered})
    errors = sorted({error for item in ordered for error in item["errors"]})
    reasons: List[str] = []

    positive_category = "conjunction_positive" in semantics
    negative_category = "conjunction_negative_excluded" in semantics
    label_conflict = positive_category and negative_category
    if label_conflict:
        reasons.append("conflicting_positive_and_no_conjunction_labels")
    if negative_category:
        reasons.append("no_conjunction_excluded")
    if "review" in semantics:
        reasons.append("issue_category_requires_review")
    fragment_count = int(canonical["fragment_count"])
    if not 2 <= fragment_count <= 5:
        reasons.append("fragment_count_outside_2_to_5")
    if errors:
        reasons.append("invalid_or_incomplete_case")

    pair_labels: List[Dict[str, Any]] = []
    graph_summary: Dict[str, Any] = {
        "derivation_attempted": False,
        "positive_connected": False,
        "positive_or_review_connected": False,
    }
    expected_pair_count = fragment_count * (fragment_count - 1) // 2
    can_derive = (
        not errors and len(canonical["derived_pair_geometry"]) == expected_pair_count
    )
    if (
        can_derive
        and 2 <= fragment_count <= 5
        and positive_category
        and not negative_category
    ):
        graph_summary["derivation_attempted"] = True
        fragment_ids = [item["fragment_id"] for item in canonical["fragments"]]
        for derived in canonical["derived_pair_geometry"]:
            fragment_a = int(derived["fragment_a"])
            fragment_b = int(derived["fragment_b"])
            geometry_result = {
                key: value
                for key, value in derived.items()
                if key not in {"fragment_a", "fragment_b"}
            }
            if fragment_count == 2:
                geometry_result["geometry_diagnostic_label"] = geometry_result["label"]
                geometry_result["label"] = "positive"
                geometry_result["label_source"] = (
                    "curated_two_fragment_conjunction_category"
                )
                geometry_result["reason"] = (
                    "two_fragment_case_explicit_positive_contract"
                )
            pair_labels.append(
                {
                    "pair_uid": _stable_uid(
                        "pair", [canonical["case_uid"], fragment_a, fragment_b]
                    ),
                    "fragment_a": fragment_a,
                    "fragment_b": fragment_b,
                    **geometry_result,
                }
            )
        ids = sorted(fragment_ids)
        positive_edges = [
            (item["fragment_a"], item["fragment_b"])
            for item in pair_labels
            if item["label"] == "positive"
        ]
        review_edges = [
            (item["fragment_a"], item["fragment_b"])
            for item in pair_labels
            if item["label"] == "review"
        ]
        graph_summary.update(
            {
                "pair_count": len(pair_labels),
                "label_counts": dict(
                    sorted(Counter(item["label"] for item in pair_labels).items())
                ),
                "positive_connected": _is_connected(ids, positive_edges),
                "positive_or_review_connected": _is_connected(
                    ids, positive_edges + review_edges
                ),
            }
        )
        if fragment_count > 2:
            if review_edges:
                reasons.append("pair_contact_ambiguity_requires_review")
            if not graph_summary["positive_connected"]:
                reasons.append("strong_contact_graph_not_connected")

    if label_conflict or negative_category or errors or not 2 <= fragment_count <= 5:
        disposition = "excluded"
    elif "review" in semantics or any(
        reason in reasons
        for reason in (
            "pair_contact_ambiguity_requires_review",
            "strong_contact_graph_not_connected",
        )
    ):
        disposition = "review"
    elif positive_category and pair_labels:
        disposition = "eligible"
    else:
        disposition = "review"
        reasons.append("no_releasable_pair_labels")

    fragments = list(canonical["fragments"])
    gt = dict(canonical["gt"])
    return {
        "case_uid": canonical["case_uid"],
        "content_signature_sha256": canonical["content_signature_sha256"],
        "occurrences": [_portable_occurrence(item) for item in ordered],
        "canonical_collection": canonical["collection"],
        "canonical_category": canonical["category"],
        "observed_categories": categories,
        "fragment_count": fragment_count,
        "numeric_metadata": canonical["numeric_metadata"],
        "fragments": fragments,
        "gt": gt,
        "data_quality_errors": errors,
        "label_conflict": label_conflict,
        "disposition": disposition,
        "disposition_reasons": sorted(set(reasons)),
        "pair_labels": pair_labels,
        "contact_graph": graph_summary,
    }


def _portable_statistics(
    cases: Sequence[Mapping[str, Any]], occurrence_count: int
) -> Dict[str, Any]:
    by_disposition = Counter(item["disposition"] for item in cases)
    by_fragment_count = Counter(str(item["fragment_count"]) for item in cases)
    by_category: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    pair_counts_by_disposition: Dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        for occurrence in case["occurrences"]:
            by_category[occurrence["category"]] += 1
        labels = [item["label"] for item in case["pair_labels"]]
        pair_counts.update(labels)
        pair_counts_by_disposition[case["disposition"]].update(labels)
    return {
        "raw_occurrence_count": occurrence_count,
        "unique_case_count": len(cases),
        "deduplicated_occurrence_count": occurrence_count - len(cases),
        "case_count_by_disposition": dict(sorted(by_disposition.items())),
        "case_count_by_fragment_count": dict(sorted(by_fragment_count.items())),
        "occurrence_count_by_category": dict(sorted(by_category.items())),
        "derived_pair_label_counts": dict(sorted(pair_counts.items())),
        "derived_pair_label_counts_by_case_disposition": {
            disposition: dict(sorted(counts.items()))
            for disposition, counts in sorted(pair_counts_by_disposition.items())
        },
        "eligible_pair_label_counts": dict(
            sorted(pair_counts_by_disposition.get("eligible", Counter()).items())
        ),
        "eligible_case_count": by_disposition.get("eligible", 0),
        "review_case_count": by_disposition.get("review", 0),
        "excluded_case_count": by_disposition.get("excluded", 0),
        "label_conflict_case_count": sum(
            bool(item["label_conflict"]) for item in cases
        ),
        "invalid_case_count": sum(bool(item["data_quality_errors"]) for item in cases),
    }


def build_real_test_manifest(
    main_root: Path,
    supp_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    roots = {"main": Path(main_root), "supp": Path(supp_root)}
    for collection, root in roots.items():
        if not root.is_dir():
            raise ManifestBuildError("missing {} dataset root".format(collection))

    occurrences: List[Dict[str, Any]] = []
    for collection in COLLECTION_NAMES:
        root = roots[collection]
        for directory_name, category in CATEGORY_DIRECTORY_TO_ID.items():
            category_root = root / directory_name
            if not category_root.is_dir():
                continue
            for group_dir in sorted(
                (path for path in category_root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            ):
                occurrences.append(_inspect_occurrence(collection, category, group_dir))

    by_case: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for occurrence in occurrences:
        by_case[occurrence["case_uid"]].append(occurrence)
    cases = [_case_record(by_case[case_uid]) for case_uid in sorted(by_case)]
    statistics = _portable_statistics(cases, len(occurrences))
    policy = {
        "intended_use": "frozen_external_pairwise_test_only",
        "forbidden_uses": ["training", "model_selection", "threshold_tuning"],
        "sealed_test": {
            "status": "sealed_external_test",
            "eligible_records_exposed_to_model_training": False,
            "eligible_records_exposed_to_threshold_or_hyperparameter_selection": False,
            "builder_access_scope": "data_quality_audit_and_label_construction_only",
        },
        "deduplication": "exact_file_content_sha256_plus_numeric_metadata; opaque_psd_source_line_excluded",
        "category_policy": {
            "ground_truth_simple": "candidate_positive",
            "small": "candidate_positive_size_slice",
            "issue": "review_only",
            "no_conjunction": "always_excluded",
            "positive_plus_no_conjunction_exact_duplicate": "label_conflict_excluded",
        },
        "fragment_count": {"minimum": 2, "maximum": 5},
        "alpha_policy": {
            "required": True,
            "foreground_threshold": ALPHA_FOREGROUND_THRESHOLD,
            "missing_empty_or_undecodable": "excluded_fail_closed",
            "trusted_collection_max_image_pixels": TRUSTED_COLLECTION_MAX_IMAGE_PIXELS,
        },
        "pair_label_policy": {
            "two_fragment_conjunction_case": "explicit_positive",
            "multi_fragment_case": "derive_each_pair_from_numeric_bbox_placement_and_alpha_masks",
            "never_assume_all_within_group_pairs_positive": True,
            "uncertain_geometry": "review",
        },
        "contact_thresholds": {
            "strong_max_gap_px": STRONG_MAX_GAP_PX,
            "review_max_gap_px": REVIEW_MAX_GAP_PX,
            "minimum_contact_pixels": MIN_CONTACT_PIXELS,
            "minimum_contact_boundary_fraction": MIN_CONTACT_BOUNDARY_FRACTION,
            "maximum_strong_overlap_fraction_of_smaller_fragment": MAX_STRONG_OVERLAP_FRACTION,
        },
        "information_used_for_pair_labels": [
            "curator_category",
            "numeric_canvas_and_bboxes_from_meta_data",
            "fragment_alpha_masks",
        ],
        "information_not_used_for_pair_labels": [
            "opaque_psd_source_line",
            "rgb_similarity",
            "manuscript_text",
            "model_outputs",
        ],
    }
    manifest_without_digest = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "policy": policy,
        "statistics": statistics,
        "cases": cases,
    }
    digest = _sha256_bytes(_canonical_json(manifest_without_digest))
    manifest = {**manifest_without_digest, "manifest_sha256": digest}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": digest,
        "statistics": statistics,
        "sealed_test": policy["sealed_test"],
        "release_decision": (
            "review_required_before_external_test"
            if statistics["review_case_count"]
            or statistics["label_conflict_case_count"]
            else "eligible_records_frozen"
        ),
        "high_severity_findings": {
            "label_conflict_case_count": statistics["label_conflict_case_count"],
            "invalid_case_count": statistics["invalid_case_count"],
            "no_conjunction_in_eligible_count": sum(
                item["disposition"] == "eligible"
                and "no_conjunction" in item["observed_categories"]
                for item in cases
            ),
        },
    }

    receipt_cases: Dict[str, Any] = {}
    for case_uid in sorted(by_case):
        receipt_cases[case_uid] = {
            "occurrences": [
                {
                    "occurrence_uid": item["occurrence_uid"],
                    "collection": item["collection"],
                    "category": item["category"],
                    "group_directory": str(item["group_dir"].resolve()),
                    "metadata_path": str(item["metadata_path"].resolve()),
                    "gt_path": str(item["gt_path"].resolve()),
                    "fragment_paths": [
                        str(path.resolve()) for path in item["fragment_paths"]
                    ],
                    "metadata_file_sha256": _safe_file_hash(item["metadata_path"]),
                }
                for item in sorted(
                    by_case[case_uid], key=lambda row: row["occurrence_uid"]
                )
            ]
        }
    receipt = {
        "schema_version": "pairwise-v0.2-real-external-test-local-receipt/0.1",
        "portable_manifest_sha256": digest,
        "dataset_roots": {key: str(value.resolve()) for key, value in roots.items()},
        "warning": "machine-local_only_contains_absolute_paths; no PSD/source-line text is stored",
        "cases": receipt_cases,
    }

    _write_json(output_dir / "real_test_manifest.json", manifest)
    _write_json(output_dir / "real_test_summary.json", summary)
    _write_json(output_dir / "local_path_receipt.json", receipt)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--supp-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_real_test_manifest(args.main_root, args.supp_root, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Materialize mask-only views for the sealed real-Dunhuang external test.

The portable ``real_test_v0_1`` manifest remains the authority for case
selection, pair order, pair IDs, and labels.  This module only transports the
alpha foreground into two matched representations:

* ``filled``: the binary fragment foreground; and
* ``contour10``: the ten-pixel internal boundary of that same foreground.

Every fragment in a case uses one scale derived from the case's original
canvas.  Absolute bounding-box origins and the RGB channels are never used to
construct either representation.  The output is an inference-only
``external_test`` package; it is deliberately not adapted to the historical
``TrainingPairRecord(split="val")`` compatibility shim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy import ndimage


SCHEMA_VERSION = "dunhuang-real-external-test-representations/0.1"
REAL_TEST_SCHEMA_VERSION = "pairwise-v0.2-real-external-test/0.1"
REAL_PATH_RECEIPT_SCHEMA_VERSION = "pairwise-v0.2-real-external-test-local-receipt/0.1"
TARGET_LONG_SIDE = 800
CONTOUR_WIDTH = 10
ALPHA_THRESHOLD = 128
EXPECTED_CASE_COUNT = 445
EXPECTED_FRAGMENT_COUNT = 938
EXPECTED_PAIR_COUNT = 547
EXPECTED_POSITIVE_COUNT = 508
EXPECTED_NEGATIVE_COUNT = 39
_MAX_TRUSTED_IMAGE_PIXELS = 300_000_000
_REAL_DATASET_ID = "real_dunhuang_strict_alpha_v0_1"
_ALLOWED_COLLECTION_CATEGORIES = frozenset(
    {
        ("main", "ground_truth_simple"),
        ("main", "small"),
        ("supp", "ground_truth_simple"),
    }
)

# The collection contains several legitimately large plates.  Decoding still
# has an explicit hard bound below; this merely aligns Pillow's warning limit.
Image.MAX_IMAGE_PIXELS = _MAX_TRUSTED_IMAGE_PIXELS


class RealDunhuangRepresentationError(ValueError):
    """The strict external-test package cannot be materialized safely."""


@dataclass(frozen=True)
class RealFragmentSpec:
    """One selected fragment and its portable output identity."""

    case_uid: str
    fragment_id: int
    collection: str
    category: str
    source_path: Path
    relative_path: str
    canvas_wh: Tuple[int, int]
    expected_size_wh: Tuple[int, int]
    declared_alpha_mask_sha256: str


@dataclass(frozen=True)
class RealPairSpec:
    """One immutable pair row in source-manifest traversal order."""

    pair_id: str
    source_pair_uid: str
    case_uid: str
    collection: str
    category: str
    fragment_a_id: int
    fragment_b_id: int
    fragment_a_relative_path: str
    fragment_b_relative_path: str
    label: bool
    direction_b_wrt_a: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "pair_id": self.pair_id,
            "source_pair_uid": self.source_pair_uid,
            "split": "external_test",
            "label": self.label,
            "direction_b_wrt_a": self.direction_b_wrt_a,
            "case_uid": self.case_uid,
            "canonical_collection": self.collection,
            "canonical_category": self.category,
            "fragment_a": {
                "fragment_id": self.fragment_a_id,
                "relative_path": self.fragment_a_relative_path,
            },
            "fragment_b": {
                "fragment_id": self.fragment_b_id,
                "relative_path": self.fragment_b_relative_path,
            },
            "label_origin": "real_test_v0_1_strict_manifest",
            "provenance": {
                "real_dunhuang_external_test": True,
                "alpha_mask_only": True,
                "rgb_used": False,
                "bbox_origin_exposed_to_representation": False,
                "gt_composite_exposed_to_representation": False,
            },
        }


@dataclass(frozen=True)
class RealExternalTestSpec:
    """Selected fragments and pairs before any source image is decoded."""

    manifest_sha256: str
    fragments: Tuple[RealFragmentSpec, ...]
    pairs: Tuple[RealPairSpec, ...]
    case_count: int
    positive_count: int
    negative_count: int


def _read_json_object(path: Path, description: str) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RealDunhuangRepresentationError(
            description + " is not readable JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise RealDunhuangRepresentationError(description + " root must be an object")
    return value


def _require_sha256(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RealDunhuangRepresentationError(
            description + " must be lowercase SHA-256"
        )
    return value


def _positive_int_pair(value: Any, description: str) -> Tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)  # noqa: E721
    ):
        raise RealDunhuangRepresentationError(
            description + " must contain two positive integers"
        )
    return int(value[0]), int(value[1])


def _selected_local_occurrence(
    case: Mapping[str, Any], local_case: Mapping[str, Any]
) -> Mapping[str, Any]:
    collection = case.get("canonical_collection")
    category = case.get("canonical_category")
    portable = case.get("occurrences")
    local = local_case.get("occurrences")
    if not isinstance(portable, list) or not isinstance(local, list):
        raise RealDunhuangRepresentationError("real case occurrences are incomplete")
    accepted_uids = {
        row.get("occurrence_uid")
        for row in portable
        if isinstance(row, Mapping)
        and row.get("collection") == collection
        and row.get("category") == category
    }
    matches = [
        row
        for row in local
        if isinstance(row, Mapping)
        and row.get("occurrence_uid") in accepted_uids
        and row.get("collection") == collection
        and row.get("category") == category
    ]
    if not matches:
        raise RealDunhuangRepresentationError(
            "strict case does not have a canonical local occurrence"
        )
    return sorted(matches, key=lambda row: str(row.get("occurrence_uid")))[0]


def _rebase_path(
    value: Any,
    *,
    collection: str,
    receipt_roots: Mapping[str, Any],
    root_overrides: Mapping[str, Optional[Path]],
) -> Path:
    if not isinstance(value, str) or not value:
        raise RealDunhuangRepresentationError("strict fragment path is missing")
    original_root_value = receipt_roots.get(collection)
    if not isinstance(original_root_value, str) or not original_root_value:
        raise RealDunhuangRepresentationError("local receipt dataset root is missing")
    original_root = Path(original_root_value)
    original_path = Path(value)
    try:
        relative = original_path.relative_to(original_root)
    except ValueError as error:
        raise RealDunhuangRepresentationError(
            "strict fragment path escapes its recorded dataset root"
        ) from error
    if relative.is_absolute() or ".." in relative.parts:
        raise RealDunhuangRepresentationError("strict fragment relative path is unsafe")
    selected_root = root_overrides.get(collection) or original_root
    return Path(selected_root).joinpath(relative)


def _bbox_direction(
    fragment_a: Mapping[str, Any], fragment_b: Mapping[str, Any]
) -> str:
    """Return B-with-respect-to-A for supervision, never representation input."""

    try:
        ax1, ay1, ax2, ay2 = (int(item) for item in fragment_a["bbox_xyxy"])
        bx1, by1, bx2, by2 = (int(item) for item in fragment_b["bbox_xyxy"])
    except (KeyError, TypeError, ValueError) as error:
        raise RealDunhuangRepresentationError(
            "strict positive fragment lacks a numeric bbox"
        ) from error
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        raise RealDunhuangRepresentationError(
            "strict positive fragment bbox is nonpositive"
        )
    dx2 = (bx1 + bx2) - (ax1 + ax2)
    dy2 = (by1 + by2) - (ay1 + ay2)
    if dx2 == 0 and dy2 == 0:
        raise RealDunhuangRepresentationError(
            "positive pair has no directional centre displacement"
        )
    if abs(dx2) >= abs(dy2):
        return "right" if dx2 > 0 else "left"
    return "below" if dy2 > 0 else "above"


def _safe_fragment_relative_path(case_uid: str, fragment_id: int) -> str:
    path = PurePosixPath(case_uid, "{}.png".format(fragment_id))
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RealDunhuangRepresentationError("unsafe output fragment path")
    return path.as_posix()


def _existing_evaluation_pair_id(
    case_uid: str, fragment_a_id: int, fragment_b_id: int
) -> str:
    """Reproduce the stable ID emitted by the existing strict evaluator.

    The old evaluator had to label these records as ``val`` only because the
    training dataclass did not admit an external-test split.  Split is not part
    of its pair-ID payload, so the identity can be retained without carrying
    that compatibility fiction into the new package.
    """

    canonical_pair_key = sorted(
        (
            "{}/fragment/{}".format(case_uid, fragment_a_id),
            "{}/fragment/{}".format(case_uid, fragment_b_id),
        )
    )
    payload = json.dumps(
        [_REAL_DATASET_ID, canonical_pair_key],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "pair/sha256/" + hashlib.sha256(payload).hexdigest()


def _check_expected_count(
    observed: int, expected: Optional[int], description: str
) -> None:
    if expected is not None and observed != expected:
        raise RealDunhuangRepresentationError(
            "strict real {} count changed: {} != {}".format(
                description, observed, expected
            )
        )


def load_real_external_test_spec(
    manifest_path: Path,
    local_path_receipt_path: Path,
    *,
    main_root: Optional[Path] = None,
    supp_root: Optional[Path] = None,
    expected_case_count: Optional[int] = EXPECTED_CASE_COUNT,
    expected_fragment_count: Optional[int] = EXPECTED_FRAGMENT_COUNT,
    expected_pair_count: Optional[int] = EXPECTED_PAIR_COUNT,
    expected_positive_count: Optional[int] = EXPECTED_POSITIVE_COUNT,
    expected_negative_count: Optional[int] = EXPECTED_NEGATIVE_COUNT,
    derive_positive_direction: bool = True,
) -> RealExternalTestSpec:
    """Select the sealed strict population without decoding any image or RGB."""

    if type(derive_positive_direction) is not bool:  # noqa: E721
        raise TypeError("derive_positive_direction must be bool")

    manifest = _read_json_object(manifest_path, "real-test manifest")
    local_receipt = _read_json_object(local_path_receipt_path, "real-test path receipt")
    if manifest.get("schema_version") != REAL_TEST_SCHEMA_VERSION:
        raise RealDunhuangRepresentationError("unsupported real-test manifest schema")
    if local_receipt.get("schema_version") != REAL_PATH_RECEIPT_SCHEMA_VERSION:
        raise RealDunhuangRepresentationError(
            "unsupported real-test path receipt schema"
        )
    manifest_sha256 = _require_sha256(
        manifest.get("manifest_sha256"), "real-test manifest SHA-256"
    )
    manifest_without_digest = dict(manifest)
    manifest_without_digest.pop("manifest_sha256", None)
    observed_manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest_without_digest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if observed_manifest_sha256 != manifest_sha256:
        raise RealDunhuangRepresentationError(
            "real-test portable manifest canonical SHA-256 differs"
        )
    if local_receipt.get("portable_manifest_sha256") != manifest_sha256:
        raise RealDunhuangRepresentationError("real manifest and local paths differ")

    cases = manifest.get("cases")
    local_cases = local_receipt.get("cases")
    receipt_roots = local_receipt.get("dataset_roots")
    if (
        not isinstance(cases, list)
        or not isinstance(local_cases, Mapping)
        or not isinstance(receipt_roots, Mapping)
    ):
        raise RealDunhuangRepresentationError(
            "real-test manifest structure is incomplete"
        )
    overrides = {
        "main": None if main_root is None else Path(main_root),
        "supp": None if supp_root is None else Path(supp_root),
    }

    fragments_out: List[RealFragmentSpec] = []
    pairs_out: List[RealPairSpec] = []
    seen_pair_ids = set()
    seen_source_pair_uids = set()
    selected_cases = 0
    positive_count = 0
    negative_count = 0

    for case in cases:
        if not isinstance(case, Mapping):
            raise RealDunhuangRepresentationError("real-test case is invalid")
        collection = case.get("canonical_collection")
        category = case.get("canonical_category")
        if (
            case.get("disposition") != "eligible"
            or (
                collection,
                category,
            )
            not in _ALLOWED_COLLECTION_CATEGORIES
        ):
            continue
        observed_categories = case.get("observed_categories")
        if isinstance(observed_categories, list) and (
            "no_conjunction" in observed_categories or "issue" in observed_categories
        ):
            raise RealDunhuangRepresentationError(
                "strict eligible case conflicts with an excluded category"
            )
        case_uid = case.get("case_uid")
        if not isinstance(case_uid, str) or not case_uid:
            raise RealDunhuangRepresentationError("strict case UID is invalid")
        canvas_metadata = case.get("numeric_metadata")
        if not isinstance(canvas_metadata, Mapping):
            raise RealDunhuangRepresentationError(
                "strict case lacks numeric canvas metadata"
            )
        canvas_wh = _positive_int_pair(
            canvas_metadata.get("canvas_wh"), "strict case canvas_wh"
        )
        local_case = local_cases.get(case_uid)
        if not isinstance(local_case, Mapping):
            raise RealDunhuangRepresentationError("strict case lacks local paths")
        occurrence = _selected_local_occurrence(case, local_case)
        fragment_paths = occurrence.get("fragment_paths")
        fragments = case.get("fragments")
        pair_labels = case.get("pair_labels")
        if (
            not isinstance(fragment_paths, list)
            or not isinstance(fragments, list)
            or not isinstance(pair_labels, list)
        ):
            raise RealDunhuangRepresentationError("strict case payload is incomplete")

        source_path_by_id: Dict[int, Path] = {}
        for value in fragment_paths:
            if not isinstance(value, str) or not Path(value).stem.isdigit():
                raise RealDunhuangRepresentationError(
                    "strict fragment filename is invalid"
                )
            fragment_id = int(Path(value).stem)
            if fragment_id in source_path_by_id:
                raise RealDunhuangRepresentationError("duplicate strict fragment path")
            source_path_by_id[fragment_id] = _rebase_path(
                value,
                collection=str(collection),
                receipt_roots=receipt_roots,
                root_overrides=overrides,
            )

        fragment_by_id: Dict[int, Mapping[str, Any]] = {}
        relative_path_by_id: Dict[int, str] = {}
        case_fragments: List[RealFragmentSpec] = []
        for fragment in fragments:
            if (
                not isinstance(fragment, Mapping)
                or type(fragment.get("fragment_id")) is not int  # noqa: E721
            ):
                raise RealDunhuangRepresentationError(
                    "strict fragment record is invalid"
                )
            fragment_id = int(fragment["fragment_id"])
            if (
                fragment_id in fragment_by_id
                or fragment_id not in source_path_by_id
                or fragment.get("has_alpha") is not True
            ):
                raise RealDunhuangRepresentationError(
                    "strict fragment/path/alpha identity differs"
                )
            declared_alpha_mask_sha256 = _require_sha256(
                fragment.get("alpha_mask_sha256"),
                "strict fragment alpha-mask SHA-256",
            )
            expected_size_wh = _positive_int_pair(
                fragment.get("size_wh"), "strict fragment size_wh"
            )
            relative_path = _safe_fragment_relative_path(case_uid, fragment_id)
            fragment_by_id[fragment_id] = fragment
            relative_path_by_id[fragment_id] = relative_path
            case_fragments.append(
                RealFragmentSpec(
                    case_uid=case_uid,
                    fragment_id=fragment_id,
                    collection=str(collection),
                    category=str(category),
                    source_path=source_path_by_id[fragment_id],
                    relative_path=relative_path,
                    canvas_wh=canvas_wh,
                    expected_size_wh=expected_size_wh,
                    declared_alpha_mask_sha256=declared_alpha_mask_sha256,
                )
            )
        if set(fragment_by_id) != set(source_path_by_id):
            raise RealDunhuangRepresentationError(
                "strict fragment/path coverage differs"
            )
        expected_pairs = len(fragment_by_id) * (len(fragment_by_id) - 1) // 2
        if len(pair_labels) != expected_pairs:
            raise RealDunhuangRepresentationError(
                "strict case pair labels are not exhaustive"
            )

        seen_case_pairs = set()
        for pair in pair_labels:
            if not isinstance(pair, Mapping):
                raise RealDunhuangRepresentationError("strict pair label is invalid")
            if (
                type(pair.get("fragment_a")) is not int  # noqa: E721
                or type(pair.get("fragment_b")) is not int  # noqa: E721
            ):
                raise RealDunhuangRepresentationError("strict pair endpoint is invalid")
            first_id = int(pair["fragment_a"])
            second_id = int(pair["fragment_b"])
            unordered = tuple(sorted((first_id, second_id)))
            if (
                first_id == second_id
                or unordered in seen_case_pairs
                or first_id not in fragment_by_id
                or second_id not in fragment_by_id
            ):
                raise RealDunhuangRepresentationError(
                    "strict pair is duplicate, self, or has an unknown endpoint"
                )
            seen_case_pairs.add(unordered)
            source_pair_uid = pair.get("pair_uid")
            if (
                not isinstance(source_pair_uid, str)
                or not source_pair_uid
                or source_pair_uid in seen_source_pair_uids
            ):
                raise RealDunhuangRepresentationError(
                    "strict pair UID is missing or duplicated"
                )
            seen_source_pair_uids.add(source_pair_uid)
            pair_id = _existing_evaluation_pair_id(case_uid, first_id, second_id)
            if pair_id in seen_pair_ids:
                raise RealDunhuangRepresentationError(
                    "strict evaluator pair ID is duplicated"
                )
            seen_pair_ids.add(pair_id)
            label_value = pair.get("label")
            if label_value not in {"positive", "negative"}:
                raise RealDunhuangRepresentationError(
                    "strict pair has a review/unknown label"
                )
            label = label_value == "positive"
            direction = (
                _bbox_direction(fragment_by_id[first_id], fragment_by_id[second_id])
                if label and derive_positive_direction
                else None
            )
            pairs_out.append(
                RealPairSpec(
                    pair_id=pair_id,
                    source_pair_uid=source_pair_uid,
                    case_uid=case_uid,
                    collection=str(collection),
                    category=str(category),
                    fragment_a_id=first_id,
                    fragment_b_id=second_id,
                    fragment_a_relative_path=relative_path_by_id[first_id],
                    fragment_b_relative_path=relative_path_by_id[second_id],
                    label=label,
                    direction_b_wrt_a=direction,
                )
            )
            if label:
                positive_count += 1
            else:
                negative_count += 1

        fragments_out.extend(case_fragments)
        selected_cases += 1

    _check_expected_count(selected_cases, expected_case_count, "case")
    _check_expected_count(len(fragments_out), expected_fragment_count, "fragment")
    _check_expected_count(len(pairs_out), expected_pair_count, "pair")
    _check_expected_count(positive_count, expected_positive_count, "positive")
    _check_expected_count(negative_count, expected_negative_count, "negative")
    if len({fragment.relative_path for fragment in fragments_out}) != len(
        fragments_out
    ):
        raise RealDunhuangRepresentationError(
            "strict output fragment paths are not unique"
        )
    return RealExternalTestSpec(
        manifest_sha256=manifest_sha256,
        fragments=tuple(fragments_out),
        pairs=tuple(pairs_out),
        case_count=selected_cases,
        positive_count=positive_count,
        negative_count=negative_count,
    )


def _load_alpha_mask(fragment: RealFragmentSpec) -> np.ndarray:
    if not fragment.source_path.is_file():
        raise RealDunhuangRepresentationError(
            "strict real fragment file is missing: " + str(fragment.source_path)
        )
    try:
        with Image.open(fragment.source_path) as image:
            if image.format != "PNG":
                raise RealDunhuangRepresentationError(
                    "strict real fragment must be PNG"
                )
            width, height = image.size
            if (
                (width, height) != fragment.expected_size_wh
                or width <= 0
                or height <= 0
                or width * height > _MAX_TRUSTED_IMAGE_PIXELS
            ):
                raise RealDunhuangRepresentationError(
                    "strict real fragment dimensions differ from manifest"
                )
            if "A" not in image.getbands():
                raise RealDunhuangRepresentationError(
                    "strict real fragment lacks an alpha channel"
                )
            # Deliberately decode only alpha; RGB or luminance never enters the
            # representation computation.
            alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    except RealDunhuangRepresentationError:
        raise
    except (OSError, UnidentifiedImageError, ValueError, TypeError) as error:
        raise RealDunhuangRepresentationError(
            "cannot decode strict real fragment alpha"
        ) from error
    mask = np.ascontiguousarray(alpha >= ALPHA_THRESHOLD, dtype=np.bool_)
    if not mask.any():
        raise RealDunhuangRepresentationError(
            "strict real fragment alpha foreground is empty"
        )
    observed_sha256 = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
    if observed_sha256 != fragment.declared_alpha_mask_sha256:
        raise RealDunhuangRepresentationError(
            "strict real fragment alpha-mask SHA-256 differs from authority"
        )
    return mask


def _resize_by_case_canvas(
    mask: np.ndarray, *, canvas_wh: Tuple[int, int], target_long_side: int
) -> np.ndarray:
    canvas_width, canvas_height = canvas_wh
    scale = target_long_side / float(max(canvas_width, canvas_height))
    height, width = mask.shape
    output_width = max(1, int(round(width * scale)))
    output_height = max(1, int(round(height * scale)))
    if (output_height, output_width) == (height, width):
        output = mask
    else:
        source = Image.fromarray(mask.astype(np.uint8, copy=False) * 255, mode="L")
        output = (
            np.asarray(
                source.resize(
                    (output_width, output_height),
                    resample=Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
            >= ALPHA_THRESHOLD
        )
    if not output.any():
        raise RealDunhuangRepresentationError(
            "case-canvas resize removed all alpha foreground"
        )
    return np.ascontiguousarray(output, dtype=np.bool_)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    return np.ascontiguousarray(
        mask[
            int(rows.min()) : int(rows.max()) + 1,
            int(columns.min()) : int(columns.max()) + 1,
        ],
        dtype=np.bool_,
    )


def _internal_contour(mask: np.ndarray, width: int) -> np.ndarray:
    eroded = ndimage.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=np.bool_),
        iterations=width,
        border_value=0,
    )
    return np.ascontiguousarray(mask & ~eroded, dtype=np.bool_)


def _write_binary_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(mask, dtype=np.uint8) * np.uint8(255)
    Image.fromarray(pixels, mode="L").save(path, format="PNG")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
            )
            stream.write("\n")


def _receipt(
    spec: RealExternalTestSpec, *, target_long_side: int, contour_width: int
) -> Dict[str, object]:
    category_counts = Counter(
        "{}/{}".format(fragment.collection, fragment.category)
        for fragment in spec.fragments
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": spec.manifest_sha256,
        "split": "external_test",
        "counts": {
            "cases": spec.case_count,
            "fragments": len(spec.fragments),
            "pairs": len(spec.pairs),
            "positive_pairs": spec.positive_count,
            "negative_pairs": spec.negative_count,
            "filled_png": len(spec.fragments),
            "contour10_png": len(spec.fragments),
        },
        "fragment_counts_by_source": dict(sorted(category_counts.items())),
        "outputs": {
            "pairs": "external_test/pairs.jsonl",
            "filled": "representations/filled",
            "contour10": "representations/contour10",
        },
        "selection": {
            "included": [
                "main/Ground Truth Simple",
                "main/Small",
                "supp/Ground Truth Simple",
            ],
            "excluded": ["No Conjunction", "Issue"],
            "pair_id_source": "existing strict real_dunhuang_evaluation identity",
            "source_pair_uid_field": "real_test_v0_1 pair_uid",
            "pair_order_source": "real_test_v0_1 eligible case/pair traversal",
            "same_case_combinations_relabelled": False,
        },
        "representation": {
            "input": "PNG alpha >= 128 only",
            "output": "PNG mode L with canonical pixels 0/255",
            "target_long_side": target_long_side,
            "shared_scale_reference": "original case canvas_wh",
            "tight_crop_after_shared_scale": True,
            "contour_width_pixels": contour_width,
            "contour_definition": (
                "M & ~binary_erosion(M, 3x3, iterations=10, border_value=0)"
            ),
            "rgb_used": False,
            "bbox_origin_used": False,
            "gt_composite_used": False,
        },
        "integrity_work": {
            "source_file_hashes_computed": False,
            "archive_hashes_computed": False,
            "manifest_pair_ids_order_labels_modified": False,
            "existing_evaluator_pair_ids_modified": False,
        },
    }


def materialize_real_dunhuang_representations(
    manifest_path: Path,
    local_path_receipt_path: Path,
    *,
    output_root: Path,
    main_root: Optional[Path] = None,
    supp_root: Optional[Path] = None,
    target_long_side: int = TARGET_LONG_SIDE,
    contour_width: int = CONTOUR_WIDTH,
    expected_case_count: Optional[int] = EXPECTED_CASE_COUNT,
    expected_fragment_count: Optional[int] = EXPECTED_FRAGMENT_COUNT,
    expected_pair_count: Optional[int] = EXPECTED_PAIR_COUNT,
    expected_positive_count: Optional[int] = EXPECTED_POSITIVE_COUNT,
    expected_negative_count: Optional[int] = EXPECTED_NEGATIVE_COUNT,
) -> Path:
    """Create a fresh filled/contour10 real external-test package.

    ``output_root`` must not exist.  Any failure removes only the newly-created
    destination, never the source data or manifest.
    """

    if type(target_long_side) is not int or target_long_side <= 0:  # noqa: E721
        raise RealDunhuangRepresentationError(
            "target_long_side must be a positive integer"
        )
    if type(contour_width) is not int or contour_width <= 0:  # noqa: E721
        raise RealDunhuangRepresentationError(
            "contour_width must be a positive integer"
        )
    if contour_width != CONTOUR_WIDTH:
        raise RealDunhuangRepresentationError(
            "real external-test contour width must be exactly 10 pixels"
        )
    destination = Path(output_root)
    if destination.exists():
        raise RealDunhuangRepresentationError(
            "refusing to overwrite existing output_root: " + destination.as_posix()
        )

    spec = load_real_external_test_spec(
        manifest_path,
        local_path_receipt_path,
        main_root=main_root,
        supp_root=supp_root,
        expected_case_count=expected_case_count,
        expected_fragment_count=expected_fragment_count,
        expected_pair_count=expected_pair_count,
        expected_positive_count=expected_positive_count,
        expected_negative_count=expected_negative_count,
    )
    protected_roots = {
        Path(manifest_path).resolve(strict=True).parent,
        Path(local_path_receipt_path).resolve(strict=True).parent,
    }
    for configured_root in (main_root, supp_root):
        if configured_root is not None:
            protected_roots.add(Path(configured_root).resolve(strict=True))
    for collection in {fragment.collection for fragment in spec.fragments}:
        collection_paths = [
            str(fragment.source_path.resolve(strict=True))
            for fragment in spec.fragments
            if fragment.collection == collection
        ]
        common = Path(os.path.commonpath(collection_paths))
        protected_roots.add(common if common.is_dir() else common.parent)
    resolved_destination = destination.resolve()
    for protected_root in protected_roots:
        try:
            resolved_destination.relative_to(protected_root)
        except ValueError:
            continue
        raise RealDunhuangRepresentationError(
            "output_root may not mutate a manifest, receipt, or source-data root"
        )
    source_pairs = tuple(
        (
            pair.pair_id,
            pair.source_pair_uid,
            pair.label,
            pair.fragment_a_id,
            pair.fragment_b_id,
        )
        for pair in spec.pairs
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise RealDunhuangRepresentationError(
            "refusing to overwrite existing output_root: " + destination.as_posix()
        ) from error

    filled_root = destination / "representations" / "filled"
    contour_root = destination / "representations" / "contour10"
    try:
        for fragment in spec.fragments:
            alpha_mask = _load_alpha_mask(fragment)
            scaled = _resize_by_case_canvas(
                alpha_mask,
                canvas_wh=fragment.canvas_wh,
                target_long_side=target_long_side,
            )
            filled = _tight_crop(scaled)
            contour = _internal_contour(filled, contour_width)
            if filled.shape != contour.shape or np.any(contour & ~filled):
                raise RealDunhuangRepresentationError(
                    "filled/contour10 representation invariant failed"
                )
            _write_binary_png(filled_root / fragment.relative_path, filled)
            _write_binary_png(contour_root / fragment.relative_path, contour)

        pair_rows = tuple(pair.to_dict() for pair in spec.pairs)
        output_pairs = tuple(
            (
                str(row["pair_id"]),
                str(row["source_pair_uid"]),
                bool(row["label"]),
                int(row["fragment_a"]["fragment_id"]),  # type: ignore[index]
                int(row["fragment_b"]["fragment_id"]),  # type: ignore[index]
            )
            for row in pair_rows
        )
        if output_pairs != source_pairs:
            raise RealDunhuangRepresentationError(
                "external-test pair IDs, source UIDs, order, labels, or endpoints changed"
            )
        pair_path = destination / "external_test" / "pairs.jsonl"
        _write_jsonl(pair_path, pair_rows)
        try:
            written_rows = tuple(
                json.loads(line)
                for line in pair_path.read_text(encoding="utf-8").splitlines()
            )
            written_pairs = tuple(
                (
                    str(row["pair_id"]),
                    str(row["source_pair_uid"]),
                    bool(row["label"]),
                    int(row["fragment_a"]["fragment_id"]),
                    int(row["fragment_b"]["fragment_id"]),
                )
                for row in written_rows
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise RealDunhuangRepresentationError(
                "written external-test pair JSONL is unreadable"
            ) from error
        if written_pairs != source_pairs:
            raise RealDunhuangRepresentationError(
                "written pair IDs, source UIDs, order, labels, or endpoints changed"
            )
        receipt = _receipt(
            spec,
            target_long_side=target_long_side,
            contour_width=contour_width,
        )
        (destination / "external_test" / "receipt.json").write_text(
            json.dumps(
                receipt,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        for fragment in spec.fragments:
            if (
                not (filled_root / fragment.relative_path).is_file()
                or not (contour_root / fragment.relative_path).is_file()
            ):
                raise RealDunhuangRepresentationError(
                    "selected fragment representation is missing"
                )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-path-receipt", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--supp-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-long-side", type=int, default=TARGET_LONG_SIDE)
    parser.add_argument("--contour-width", type=int, default=CONTOUR_WIDTH)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    output = materialize_real_dunhuang_representations(
        arguments.manifest,
        arguments.local_path_receipt,
        output_root=arguments.output_root,
        main_root=arguments.main_root,
        supp_root=arguments.supp_root,
        target_long_side=arguments.target_long_side,
        contour_width=arguments.contour_width,
    )
    receipt = json.loads(
        (output / "external_test" / "receipt.json").read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "status": "real_dunhuang_external_test_prepared",
                "output_root": output.as_posix(),
                "counts": receipt["counts"],
                "representations": {
                    "filled": (output / "representations" / "filled").as_posix(),
                    "contour10": (output / "representations" / "contour10").as_posix(),
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALPHA_THRESHOLD",
    "CONTOUR_WIDTH",
    "EXPECTED_CASE_COUNT",
    "EXPECTED_FRAGMENT_COUNT",
    "EXPECTED_NEGATIVE_COUNT",
    "EXPECTED_PAIR_COUNT",
    "EXPECTED_POSITIVE_COUNT",
    "RealDunhuangRepresentationError",
    "RealExternalTestSpec",
    "RealFragmentSpec",
    "RealPairSpec",
    "SCHEMA_VERSION",
    "TARGET_LONG_SIDE",
    "load_real_external_test_spec",
    "main",
    "materialize_real_dunhuang_representations",
]

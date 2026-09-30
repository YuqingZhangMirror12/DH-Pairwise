"""Evaluate LOCAL-Q1 winners on the strict real-Dunhuang pairs.

This module is deliberately separate from the training runner and provider.
It consumes the already-built ``real_test_v0_1`` manifest, loads fragment
foreground from PNG alpha only, builds the existing four-direction geometry
contract, and performs a direct inference-only forward pass through each
winning LOCAL-Q1 checkpoint.

When requested, the same ordered alpha-mask pairs are also passed independently
to the historical 30k and matched Route-A MobileNetV2 Siamese checkpoints.
Their join-probability metrics share the four arms' pair IDs, labels, and
common-valid population; neither Siamese control has a direction output.

Ground-truth bounding boxes are used only to derive the positive direction
target.  The model receives a tight alpha mask with no canvas origin, no
ground-truth composite, and no RGB texture.  ``No Conjunction`` and ``Issue``
cases are never admitted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor, nn

from staging.pairwise_v0_2.baselines.historical_mm_siamese import (
    HISTORICAL_MM_BASELINE_ID,
    load_historical_mm_checkpoint,
    score_historical_mm_mask_pairs,
)
from staging.pairwise_v0_2.baselines.matched_route_a_siamese import (
    MATCHED_ROUTE_A_SIAMESE_ID,
)

from staging.pairwise_v0_2.baselines.mm_validation_comparison import (
    FourArmRunWinnerLoader,
    LoadedArmWinner,
)
from staging.pairwise_v0_2.geometry import ContourKeypointConfig
from staging.pairwise_v0_2.models.pairwise import PairwiseScoreSource
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.engine import eval_directional_step
from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise
from staging.pairwise_v0_2.training.geometry_batch import (
    DATA_DIRECTION_TO_INDEX,
    KEYPOINT_REPRESENTATION,
    GeometryBatchConfig,
    RaggedGeometryBatch,
    build_geometry_batch,
)
from staging.pairwise_v0_2.training.geometry_cache import (
    GeometryArtifactCache,
    GeometryCacheLimits,
)
from staging.pairwise_v0_2.training.local_q1_provider import (
    local_q1_prepared_digests,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    PreparedAblationBatch,
    record_sequence_fingerprint,
)


REAL_DUNHUANG_EVALUATION_VERSION = "dunhuang-real-pairwise-evaluation/0.1"
REAL_DUNHUANG_DATASET_ID = "real_dunhuang_strict_alpha_v0_1"
REAL_TEST_SCHEMA_VERSION = "pairwise-v0.2-real-external-test/0.1"
REAL_PATH_RECEIPT_SCHEMA_VERSION = "pairwise-v0.2-real-external-test-local-receipt/0.1"
STRICT_CATEGORIES = frozenset(("ground_truth_simple", "small"))
EXPECTED_STRICT_CASE_COUNT = 445
EXPECTED_STRICT_PAIR_COUNT = 547
EXPECTED_STRICT_POSITIVE_COUNT = 508
EXPECTED_STRICT_NEGATIVE_COUNT = 39
DEFAULT_TARGET_LONG_SIDE = 800
_MAX_TRUSTED_IMAGE_PIXELS = 300_000_000
# The strict collection contains a few legitimately very large PNGs.  Match
# Pillow's warning threshold to the explicit hard bound enforced on every
# decoded fragment below.
Image.MAX_IMAGE_PIXELS = _MAX_TRUSTED_IMAGE_PIXELS
_DIRECTION_LABELS = ("left", "right", "above", "below")
_FOUR_ARMS = (
    AblationArmName.LOCAL_DUAL_SOFTMAX,
    AblationArmName.LOCAL_DUSTBIN_SINKHORN,
    AblationArmName.KEYPOINT_DUAL_SOFTMAX,
    AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
)
_REAL_ALPHA_PREPROCESS_SHA256 = hashlib.sha256(
    b"real-dunhuang-alpha-only-tight-crop-case-common-scale-v1"
).hexdigest()


class RealDunhuangEvaluationError(ValueError):
    """Raised when the strict real-pair inference contract cannot be met."""


def _json_object(path: Path, name: str) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RealDunhuangEvaluationError(name + " is not readable JSON") from exc
    if not isinstance(value, Mapping):
        raise RealDunhuangEvaluationError(name + " root must be an object")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RealDunhuangEvaluationError(name + " must be lowercase SHA-256")
    return value


def _bbox_direction(
    fragment_a: Mapping[str, Any], fragment_b: Mapping[str, Any]
) -> str:
    """Return B-with-respect-to-A from the dominant GT-centre displacement.

    The result is supervision only.  Neither this function's inputs nor its
    displacement are retained by the alpha loader or geometry batch.
    """

    try:
        ax1, ay1, ax2, ay2 = (int(value) for value in fragment_a["bbox_xyxy"])
        bx1, by1, bx2, by2 = (int(value) for value in fragment_b["bbox_xyxy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RealDunhuangEvaluationError(
            "strict positive fragment lacks a numeric bbox"
        ) from exc
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        raise RealDunhuangEvaluationError("strict positive bbox is nonpositive")
    dx2 = (bx1 + bx2) - (ax1 + ax2)
    dy2 = (by1 + by2) - (ay1 + ay2)
    if dx2 == 0 and dy2 == 0:
        raise RealDunhuangEvaluationError(
            "positive pair has no directional centre displacement"
        )
    if abs(dx2) >= abs(dy2):
        return "right" if dx2 > 0 else "left"
    return "below" if dy2 > 0 else "above"


@dataclass(frozen=True)
class _AlphaFragmentSource:
    archive_member: str
    case_uid: str
    fragment_id: int
    path: Path
    expected_alpha_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        _require_sha256(self.expected_alpha_sha256, "alpha mask SHA-256")


class RealAlphaMaskLoader:
    """Load, verify, tight-crop and case-scale PNG alpha foreground only."""

    def __init__(
        self,
        sources: Mapping[str, _AlphaFragmentSource],
        *,
        target_long_side: int = DEFAULT_TARGET_LONG_SIDE,
        max_cached_cases: int = 8,
    ) -> None:
        if not sources:
            raise ValueError("real alpha loader requires fragment sources")
        if (
            isinstance(target_long_side, bool)
            or not isinstance(target_long_side, int)
            or target_long_side < 32
            or target_long_side > 4096
        ):
            raise ValueError("target_long_side must be an integer in [32, 4096]")
        if (
            isinstance(max_cached_cases, bool)
            or not isinstance(max_cached_cases, int)
            or max_cached_cases < 1
        ):
            raise ValueError("max_cached_cases must be a positive integer")
        self._sources = dict(sources)
        self.target_long_side = target_long_side
        self.max_cached_cases = max_cached_cases
        by_case: Dict[str, Dict[str, _AlphaFragmentSource]] = {}
        for member, source in self._sources.items():
            if member != source.archive_member:
                raise ValueError("alpha source key differs from archive_member")
            case = by_case.setdefault(source.case_uid, {})
            if member in case:
                raise ValueError("duplicate alpha fragment member")
            case[member] = source
        self._by_case = by_case
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self._closed = False

    @staticmethod
    def _raw_alpha_mask(source: _AlphaFragmentSource) -> np.ndarray:
        if not source.path.is_file():
            raise RealDunhuangEvaluationError(
                "strict real fragment file is missing: " + str(source.path)
            )
        try:
            with Image.open(source.path) as image:
                if image.format != "PNG":
                    raise RealDunhuangEvaluationError(
                        "strict real fragment must be PNG"
                    )
                width, height = image.size
                if (
                    width <= 0
                    or height <= 0
                    or width * height > _MAX_TRUSTED_IMAGE_PIXELS
                ):
                    raise RealDunhuangEvaluationError(
                        "strict real fragment dimensions are invalid"
                    )
                image.load()
                if "A" in image.getbands():
                    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
                elif "transparency" in image.info:
                    alpha = np.asarray(
                        image.convert("RGBA").getchannel("A"), dtype=np.uint8
                    )
                else:
                    raise RealDunhuangEvaluationError(
                        "strict real fragment lacks alpha"
                    )
        except RealDunhuangEvaluationError:
            raise
        except (OSError, UnidentifiedImageError, ValueError, TypeError) as exc:
            raise RealDunhuangEvaluationError(
                "cannot decode strict real fragment alpha"
            ) from exc
        mask = np.ascontiguousarray(alpha >= 128, dtype=np.bool_)
        if not mask.any():
            raise RealDunhuangEvaluationError("strict real fragment alpha is empty")
        if _sha256(mask.tobytes(order="C")) != source.expected_alpha_sha256:
            raise RealDunhuangEvaluationError(
                "strict real fragment alpha differs from manifest"
            )
        return mask

    @staticmethod
    def _tight_crop(mask: np.ndarray) -> np.ndarray:
        rows, columns = np.nonzero(mask)
        return np.ascontiguousarray(
            mask[
                int(rows.min()) : int(rows.max()) + 1,
                int(columns.min()) : int(columns.max()) + 1,
            ],
            dtype=np.bool_,
        )

    @staticmethod
    def _resize(mask: np.ndarray, scale: float) -> np.ndarray:
        height, width = mask.shape
        output_width = max(1, int(round(width * scale)))
        output_height = max(1, int(round(height * scale)))
        if (output_height, output_width) == (height, width):
            output = np.ascontiguousarray(mask, dtype=np.bool_)
        else:
            source = Image.fromarray(mask.astype(np.uint8, copy=False) * 255, mode="L")
            output = (
                np.asarray(
                    source.resize(
                        (output_width, output_height), resample=Image.Resampling.NEAREST
                    ),
                    dtype=np.uint8,
                )
                > 127
            )
            output = np.ascontiguousarray(output, dtype=np.bool_)
        if not output.any():
            raise RealDunhuangEvaluationError(
                "case-common alpha resize removed all foreground"
            )
        output.setflags(write=False)
        return output

    def _load_case(self, case_uid: str) -> Dict[str, np.ndarray]:
        sources = self._by_case[case_uid]
        cropped = {
            member: self._tight_crop(self._raw_alpha_mask(source))
            for member, source in sources.items()
        }
        longest = max(max(mask.shape) for mask in cropped.values())
        if longest <= 0:  # pragma: no cover - nonempty alpha already guarantees this
            raise RealDunhuangEvaluationError("case alpha scale is invalid")
        scale = self.target_long_side / float(longest)
        return {member: self._resize(mask, scale) for member, mask in cropped.items()}

    def __call__(self, reference: Any) -> np.ndarray:
        if self._closed:
            raise RealDunhuangEvaluationError("real alpha loader is closed")
        if not isinstance(reference, MaskMemberRef):
            raise TypeError("real alpha loader requires MaskMemberRef")
        try:
            source = self._sources[reference.archive_member]
        except KeyError as exc:
            raise RealDunhuangEvaluationError(
                "real alpha reference is not in the strict manifest"
            ) from exc
        case = self._cache.get(source.case_uid)
        if case is None:
            case = self._load_case(source.case_uid)
            self._cache[source.case_uid] = case
        self._cache.move_to_end(source.case_uid)
        while len(self._cache) > self.max_cached_cases:
            self._cache.popitem(last=False)
        return case[reference.archive_member]

    def close(self) -> None:
        self._cache.clear()
        self._closed = True


@dataclass(frozen=True)
class RealPairDataset:
    records: Tuple[TrainingPairRecord, ...]
    mask_loader: RealAlphaMaskLoader
    manifest_sha256: str
    case_count: int
    positive_count: int
    negative_count: int
    target_long_side: int

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("real pair dataset cannot be empty")
        if self.positive_count + self.negative_count != len(self.records):
            raise ValueError("real pair dataset label counts are inconsistent")


def _selected_local_occurrence(
    case: Mapping[str, Any], local_case: Mapping[str, Any]
) -> Mapping[str, Any]:
    collection = case.get("canonical_collection")
    category = case.get("canonical_category")
    portable = case.get("occurrences")
    local = local_case.get("occurrences")
    if not isinstance(portable, list) or not isinstance(local, list):
        raise RealDunhuangEvaluationError("real case occurrences are incomplete")
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
        raise RealDunhuangEvaluationError(
            "strict case does not have a canonical local occurrence"
        )
    # Exact duplicates can occur more than once inside the same collection and
    # curator category.  Their bytes and numeric metadata are already unified
    # by ``case_uid``; choose one path deterministically for alpha transport.
    return sorted(matches, key=lambda row: str(row["occurrence_uid"]))[0]


def _rebase_path(
    value: Any,
    *,
    collection: str,
    receipt_roots: Mapping[str, Any],
    root_overrides: Mapping[str, Optional[Path]],
) -> Path:
    if not isinstance(value, str) or not value:
        raise RealDunhuangEvaluationError("strict fragment path is missing")
    original_root_value = receipt_roots.get(collection)
    if not isinstance(original_root_value, str) or not original_root_value:
        raise RealDunhuangEvaluationError("local receipt dataset root is missing")
    original_root = Path(original_root_value)
    original_path = Path(value)
    try:
        relative = original_path.relative_to(original_root)
    except ValueError as exc:
        raise RealDunhuangEvaluationError(
            "strict fragment path escapes its recorded dataset root"
        ) from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise RealDunhuangEvaluationError("strict fragment relative path is unsafe")
    selected_root = root_overrides.get(collection) or original_root
    return Path(selected_root).joinpath(relative)


def load_strict_real_pair_dataset(
    manifest_path: Path,
    local_path_receipt_path: Path,
    *,
    main_root: Optional[Path] = None,
    supp_root: Optional[Path] = None,
    target_long_side: int = DEFAULT_TARGET_LONG_SIDE,
    expected_case_count: Optional[int] = EXPECTED_STRICT_CASE_COUNT,
    expected_pair_count: Optional[int] = EXPECTED_STRICT_PAIR_COUNT,
    expected_positive_count: Optional[int] = EXPECTED_STRICT_POSITIVE_COUNT,
    expected_negative_count: Optional[int] = EXPECTED_STRICT_NEGATIVE_COUNT,
) -> RealPairDataset:
    """Adapt eligible GTS/Small cases to the existing Pairwise record contract."""

    manifest = _json_object(manifest_path, "real-test manifest")
    local_receipt = _json_object(local_path_receipt_path, "real-test path receipt")
    if manifest.get("schema_version") != REAL_TEST_SCHEMA_VERSION:
        raise RealDunhuangEvaluationError("unsupported real-test manifest schema")
    if local_receipt.get("schema_version") != REAL_PATH_RECEIPT_SCHEMA_VERSION:
        raise RealDunhuangEvaluationError("unsupported real-test path receipt schema")
    manifest_sha256 = _require_sha256(
        manifest.get("manifest_sha256"), "real-test manifest SHA-256"
    )
    if local_receipt.get("portable_manifest_sha256") != manifest_sha256:
        raise RealDunhuangEvaluationError("real manifest and local paths differ")
    cases = manifest.get("cases")
    local_cases = local_receipt.get("cases")
    receipt_roots = local_receipt.get("dataset_roots")
    if (
        not isinstance(cases, list)
        or not isinstance(local_cases, Mapping)
        or not isinstance(receipt_roots, Mapping)
    ):
        raise RealDunhuangEvaluationError("real-test manifest structure is incomplete")
    binding = ArchiveBinding(
        logical_id="local_asset://dunhuang_real_strict_alpha_v0_1",
        archive_format="zip",
        sha256=manifest_sha256,
    )
    overrides = {
        "main": None if main_root is None else Path(main_root),
        "supp": None if supp_root is None else Path(supp_root),
    }
    records = []
    sources: Dict[str, _AlphaFragmentSource] = {}
    selected_case_count = 0
    positive_count = 0
    negative_count = 0
    for case in cases:
        if not isinstance(case, Mapping):
            raise RealDunhuangEvaluationError("real-test case is invalid")
        category = case.get("canonical_category")
        if case.get("disposition") != "eligible" or category not in STRICT_CATEGORIES:
            continue
        if category in {"no_conjunction", "issue"}:  # defensive explicit boundary
            raise RealDunhuangEvaluationError("forbidden real-test category admitted")
        observed = case.get("observed_categories")
        if isinstance(observed, list) and "no_conjunction" in observed:
            raise RealDunhuangEvaluationError(
                "strict case conflicts with No Conjunction"
            )
        case_uid = case.get("case_uid")
        collection = case.get("canonical_collection")
        if (
            not isinstance(case_uid, str)
            or not case_uid
            or collection not in {"main", "supp"}
        ):
            raise RealDunhuangEvaluationError("strict case identity is invalid")
        local_case = local_cases.get(case_uid)
        if not isinstance(local_case, Mapping):
            raise RealDunhuangEvaluationError("strict case lacks local paths")
        occurrence = _selected_local_occurrence(case, local_case)
        fragment_paths = occurrence.get("fragment_paths")
        fragments = case.get("fragments")
        pair_labels = case.get("pair_labels")
        if (
            not isinstance(fragment_paths, list)
            or not isinstance(fragments, list)
            or not isinstance(pair_labels, list)
        ):
            raise RealDunhuangEvaluationError("strict case payload is incomplete")
        path_by_id = {}
        for value in fragment_paths:
            if not isinstance(value, str) or not Path(value).stem.isdigit():
                raise RealDunhuangEvaluationError("strict fragment filename is invalid")
            fragment_id = int(Path(value).stem)
            if fragment_id in path_by_id:
                raise RealDunhuangEvaluationError("duplicate strict fragment path")
            path_by_id[fragment_id] = _rebase_path(
                value,
                collection=collection,
                receipt_roots=receipt_roots,
                root_overrides=overrides,
            )
        fragment_by_id: Dict[int, Mapping[str, Any]] = {}
        reference_by_id: Dict[int, MaskMemberRef] = {}
        for fragment in fragments:
            if (
                not isinstance(fragment, Mapping)
                or type(fragment.get("fragment_id")) is not int
            ):
                raise RealDunhuangEvaluationError("strict fragment record is invalid")
            fragment_id = int(fragment["fragment_id"])
            if fragment_id in fragment_by_id or fragment_id not in path_by_id:
                raise RealDunhuangEvaluationError("strict fragment/path IDs differ")
            if fragment.get("has_alpha") is not True:
                raise RealDunhuangEvaluationError("strict fragment lacks alpha")
            alpha_sha = _require_sha256(
                fragment.get("alpha_mask_sha256"), "strict fragment alpha SHA-256"
            )
            member = "real/{}/{}.png".format(case_uid, fragment_id)
            reference = MaskMemberRef(
                binding=binding,
                archive_member=member,
                fragment_id="{}/fragment/{}".format(case_uid, fragment_id),
                dataset_id=REAL_DUNHUANG_DATASET_ID,
                canonical_group_id=case_uid,
                component_id=case_uid,
                split="val",
                threshold_rule="grayscale_uint8_gt_127",
                content_sha256=alpha_sha,
            )
            fragment_by_id[fragment_id] = fragment
            reference_by_id[fragment_id] = reference
            sources[member] = _AlphaFragmentSource(
                archive_member=member,
                case_uid=case_uid,
                fragment_id=fragment_id,
                path=path_by_id[fragment_id],
                expected_alpha_sha256=alpha_sha,
            )
        if set(fragment_by_id) != set(path_by_id):
            raise RealDunhuangEvaluationError("strict fragment/path coverage differs")
        expected_pairs = len(fragment_by_id) * (len(fragment_by_id) - 1) // 2
        if len(pair_labels) != expected_pairs:
            raise RealDunhuangEvaluationError(
                "strict case pair labels are not exhaustive"
            )
        seen_pairs = set()
        for pair in pair_labels:
            if not isinstance(pair, Mapping):
                raise RealDunhuangEvaluationError("strict pair label is invalid")
            if (
                type(pair.get("fragment_a")) is not int
                or type(pair.get("fragment_b")) is not int
            ):
                raise RealDunhuangEvaluationError("strict pair endpoint is invalid")
            first_id = int(pair["fragment_a"])
            second_id = int(pair["fragment_b"])
            key = tuple(sorted((first_id, second_id)))
            if first_id == second_id or key in seen_pairs:
                raise RealDunhuangEvaluationError("strict pair is duplicate or self")
            seen_pairs.add(key)
            try:
                first = reference_by_id[first_id]
                second = reference_by_id[second_id]
            except KeyError as exc:
                raise RealDunhuangEvaluationError(
                    "strict pair refers to an unknown fragment"
                ) from exc
            label_value = pair.get("label")
            if label_value not in {"positive", "negative"}:
                raise RealDunhuangEvaluationError(
                    "strict pair has a review/unknown label"
                )
            label = label_value == "positive"
            direction = (
                _bbox_direction(fragment_by_id[first_id], fragment_by_id[second_id])
                if label
                else None
            )
            records.append(
                TrainingPairRecord(
                    fragment_a=first,
                    fragment_b=second,
                    label=label,
                    direction_b_wrt_a=direction,
                    dataset_id=REAL_DUNHUANG_DATASET_ID,
                    canonical_group_id=case_uid,
                    component_id=case_uid,
                    split="val",
                    canonical_pair_key=tuple(
                        sorted((first.fragment_id, second.fragment_id))
                    ),
                    label_origin="real_test_v0_1_strict_manifest",
                    provenance={
                        "real_dunhuang_external_test": True,
                        "alpha_mask_only": True,
                        "bbox_origin_exposed_to_model": False,
                        "gt_composite_exposed_to_model": False,
                        "canonical_category": category,
                    },
                )
            )
            if label:
                positive_count += 1
            else:
                negative_count += 1
        selected_case_count += 1
    if len({record.pair_id for record in records}) != len(records):
        raise RealDunhuangEvaluationError("strict real pair IDs are not unique")
    expected_values = (
        (selected_case_count, expected_case_count, "case"),
        (len(records), expected_pair_count, "pair"),
        (positive_count, expected_positive_count, "positive"),
        (negative_count, expected_negative_count, "negative"),
    )
    for observed, expected, name in expected_values:
        if expected is not None and observed != expected:
            raise RealDunhuangEvaluationError(
                "strict real {} count changed: {} != {}".format(
                    name, observed, expected
                )
            )
    loader = RealAlphaMaskLoader(
        sources,
        target_long_side=target_long_side,
    )
    return RealPairDataset(
        records=tuple(records),
        mask_loader=loader,
        manifest_sha256=manifest_sha256,
        case_count=selected_case_count,
        positive_count=positive_count,
        negative_count=negative_count,
        target_long_side=target_long_side,
    )


def prepare_real_geometry_batch(
    records: Sequence[TrainingPairRecord],
    *,
    mask_loader: Callable[[MaskMemberRef], np.ndarray],
    geometry_config: GeometryBatchConfig,
    geometry_cache: GeometryArtifactCache,
    arm: AblationArm,
) -> PreparedAblationBatch:
    """Build one inference batch with the exact LOCAL-Q1 tensor contract."""

    if not isinstance(arm, AblationArm):
        raise TypeError("arm must be AblationArm")
    population = tuple(records)
    if not population:
        raise ValueError("real geometry batch cannot be empty")
    keypoint_config = None
    if arm.candidate_representation == KEYPOINT_REPRESENTATION:
        value = arm.model_config.get("contour_keypoint_config")
        if not isinstance(value, Mapping):
            raise RealDunhuangEvaluationError(
                "keypoint winner lacks contour keypoint configuration"
            )
        try:
            keypoint_config = ContourKeypointConfig(**dict(value))
        except (TypeError, ValueError) as exc:
            raise RealDunhuangEvaluationError(
                "keypoint winner configuration is invalid"
            ) from exc
    payload = build_geometry_batch(
        population,
        mask_loader,
        geometry_config,
        geometry_artifact_cache=geometry_cache,
        candidate_representation=arm.candidate_representation,
        keypoint_config=keypoint_config,
    )
    prepared_sha, local_sha = local_q1_prepared_digests(payload)
    return PreparedAblationBatch(
        payload=payload,
        sample_count=len(population),
        record_sequence_sha256=record_sequence_fingerprint(population),
        prepared_input_sha256=prepared_sha,
        local_candidate_sha256=local_sha,
        coarse_preprocessing_sha256=_REAL_ALPHA_PREPROCESS_SHA256,
        geometry_config_sha256=geometry_config.fingerprint,
        processing_counts={
            "coarse_preprocess_count": 2 * len(population),
            "geometry_build_count": 0,
            "geometry_cache_read_count": 0,
            "geometry_cache_write_count": 0,
            "local_candidate_count": payload.candidate_count,
            "mask_load_count": len(
                {
                    reference
                    for record in population
                    for reference in (record.fragment_a, record.fragment_b)
                }
            ),
        },
        candidate_representation=payload.candidate_representation,
    )


@dataclass(frozen=True)
class RealBatchPrediction:
    probability: Tensor
    valid: Tensor
    best_direction_index: Tensor

    def __post_init__(self) -> None:
        if self.probability.ndim != 1 or not self.probability.is_floating_point():
            raise TypeError("real probability must be float [B]")
        expected = tuple(self.probability.shape)
        if self.valid.dtype != torch.bool or tuple(self.valid.shape) != expected:
            raise TypeError("real validity must be bool [B]")
        if (
            self.best_direction_index.dtype != torch.long
            or tuple(self.best_direction_index.shape) != expected
        ):
            raise TypeError("best_direction_index must be int64 [B]")


RealArmScorer = Callable[[LoadedArmWinner, PreparedAblationBatch], RealBatchPrediction]


def score_loaded_winner_direct(
    loaded: LoadedArmWinner, prepared: PreparedAblationBatch
) -> RealBatchPrediction:
    """Perform one direct inference-only forward and retain direction output."""

    if not isinstance(loaded, LoadedArmWinner):
        raise TypeError("loaded winner must be LoadedArmWinner")
    payload = prepared.payload
    if not isinstance(payload, RaggedGeometryBatch):
        raise TypeError("prepared real payload must be RaggedGeometryBatch")
    if payload.candidate_representation != loaded.arm.candidate_representation:
        raise RealDunhuangEvaluationError(
            "real payload representation differs from winner"
        )
    session = loaded.session
    model = getattr(session, "model", None)
    device = getattr(session, "device", None)
    step_config = getattr(session, "step_config", None)
    if model is None or device is None or step_config is None:
        raise RealDunhuangEvaluationError(
            "winner session lacks direct-inference attributes"
        )
    moved = payload.to(torch.device(device))
    result = eval_directional_step(
        model,
        moved.model_inputs(),
        moved.labels,
        moved.direction_target,
        moved.direction_target_valid,
        score_source=PairwiseScoreSource.LOCAL,
        coarse_loss_weight=float(step_config.coarse_loss_weight),
        direction_loss_weight=float(step_config.direction_loss_weight),
        sinkhorn_residual_weight=float(step_config.sinkhorn_residual_weight),
        sinkhorn_residual_target=float(step_config.sinkhorn_residual_target),
    )
    direction = result.output.direction_output
    valid = (direction.pair_valid & moved.geometry_valid).detach().cpu()
    best = direction.best_direction_index.detach().cpu().to(torch.long)
    best = torch.where(valid, best, torch.full_like(best, -1))
    return RealBatchPrediction(
        probability=direction.pair_probability.detach().cpu(),
        valid=valid,
        best_direction_index=best,
    )


def _direction_metrics(
    *,
    labels: Sequence[bool],
    target: Sequence[int],
    valid: Sequence[bool],
    predicted: Sequence[int],
) -> Mapping[str, Any]:
    if not (len(labels) == len(target) == len(valid) == len(predicted)):
        raise ValueError("direction metric vectors differ in length")
    support = Counter()
    correct = Counter()
    confusion = {name: Counter() for name in _DIRECTION_LABELS}
    positive_count = 0
    valid_positive_count = 0
    correct_count = 0
    for label, expected, usable, observed in zip(labels, target, valid, predicted):
        if not label:
            continue
        positive_count += 1
        if expected < 0 or expected >= len(_DIRECTION_LABELS):
            raise RealDunhuangEvaluationError(
                "positive real pair lacks a four-direction target"
            )
        expected_name = _DIRECTION_LABELS[expected]
        support[expected_name] += 1
        if usable and 0 <= observed < len(_DIRECTION_LABELS):
            valid_positive_count += 1
            observed_name = _DIRECTION_LABELS[observed]
            confusion[expected_name][observed_name] += 1
            if observed == expected:
                correct_count += 1
                correct[expected_name] += 1
    per_direction = {
        name: {
            "support": support[name],
            "correct": correct[name],
            "accuracy_invalid_as_incorrect": (
                correct[name] / support[name] if support[name] else None
            ),
            "predicted_counts_on_valid": {
                predicted_name: confusion[name][predicted_name]
                for predicted_name in _DIRECTION_LABELS
            },
        }
        for name in _DIRECTION_LABELS
    }
    return {
        "positive_count": positive_count,
        "valid_positive_count": valid_positive_count,
        "correct_count": correct_count,
        "coverage": valid_positive_count / positive_count,
        "accuracy_invalid_as_incorrect": correct_count / positive_count,
        "accuracy_valid_only": (
            correct_count / valid_positive_count if valid_positive_count else None
        ),
        "per_direction": per_direction,
    }


def _metric_row(
    probability: Sequence[float],
    labels: Sequence[bool],
    valid: Sequence[bool],
    clusters: Sequence[str],
) -> Mapping[str, Any]:
    return evaluate_pairwise(
        probability,
        labels,
        valid,
        clusters,
        threshold=0.5,
    )


def score_historical_mm_real_pairs(
    model: nn.Module,
    dataset: RealPairDataset,
    *,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 256,
) -> Tensor:
    """Score the strict ordered real pairs from their alpha masks only."""

    records = dataset.records
    if any(
        record.dataset_id != REAL_DUNHUANG_DATASET_ID or record.split != "val"
        for record in records
    ):
        raise RealDunhuangEvaluationError(
            "historical comparison received a non-strict-real record"
        )
    probability = score_historical_mm_mask_pairs(
        model,
        (
            (
                dataset.mask_loader(record.fragment_a),
                dataset.mask_loader(record.fragment_b),
            )
            for record in records
        ),
        device=device,
        batch_size=batch_size,
    )
    if probability.dtype != torch.float64 or tuple(probability.shape) != (
        len(records),
    ):
        raise RealDunhuangEvaluationError(
            "historical MM real prediction cardinality changed"
        )
    return probability


def evaluate_real_dunhuang_four_arms(
    *,
    dataset: RealPairDataset,
    geometry_config: GeometryBatchConfig,
    geometry_cache: GeometryArtifactCache,
    arm_winner_loader: Callable[[AblationArmName], LoadedArmWinner],
    arm_scorer: RealArmScorer = score_loaded_winner_direct,
    batch_size: int = 1,
    historical_model: Optional[nn.Module] = None,
    historical_device: Union[str, torch.device] = "cpu",
    historical_batch_size: int = 256,
    matched_model: Optional[nn.Module] = None,
    matched_device: Union[str, torch.device] = "cpu",
    matched_batch_size: int = 256,
) -> Mapping[str, Any]:
    """Score four winners and zero, one, or two whole-mask controls."""

    if not callable(arm_winner_loader) or not callable(arm_scorer):
        raise TypeError("winner loader and scorer must be callable")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or batch_size > geometry_config.max_batch_size
    ):
        raise ValueError("batch_size is outside the geometry config bound")
    records = dataset.records
    batches = tuple(
        records[start : start + batch_size]
        for start in range(0, len(records), batch_size)
    )
    labels = tuple(record.label for record in records)
    clusters = tuple(record.component_id for record in records)
    targets = tuple(
        -1
        if record.direction_b_wrt_a is None
        else int(DATA_DIRECTION_TO_INDEX[record.direction_b_wrt_a])
        for record in records
    )
    pair_ids = tuple(record.pair_id for record in records)
    prepared_identity: Dict[Tuple[str, int], Tuple[str, str, str]] = {}
    arm_values = {}
    for name in _FOUR_ARMS:
        loaded = arm_winner_loader(name)
        if not isinstance(loaded, LoadedArmWinner) or loaded.arm.name is not name:
            raise RealDunhuangEvaluationError(
                "winner loader returned the wrong real-evaluation arm"
            )
        probabilities = []
        valid_values = []
        best_values = []
        for ordinal, batch_records in enumerate(batches):
            prepared = prepare_real_geometry_batch(
                batch_records,
                mask_loader=dataset.mask_loader,
                geometry_config=geometry_config,
                geometry_cache=geometry_cache,
                arm=loaded.arm,
            )
            identity = (
                prepared.record_sequence_sha256,
                prepared.prepared_input_sha256,
                str(prepared.local_candidate_sha256),
            )
            key = (prepared.candidate_representation, ordinal)
            previous = prepared_identity.setdefault(key, identity)
            if previous != identity:
                raise RealDunhuangEvaluationError(
                    "dual-softmax and Sinkhorn received different real tensors"
                )
            prediction = arm_scorer(loaded, prepared)
            if not isinstance(prediction, RealBatchPrediction) or tuple(
                prediction.probability.shape
            ) != (len(batch_records),):
                raise RealDunhuangEvaluationError(
                    "real winner prediction cardinality changed"
                )
            probabilities.extend(float(value) for value in prediction.probability)
            valid_values.extend(bool(value) for value in prediction.valid)
            best_values.extend(int(value) for value in prediction.best_direction_index)
        if len(probabilities) != len(records):
            raise RealDunhuangEvaluationError("real winner pair count changed")
        arm_values[name.value] = {
            "probability": probabilities,
            "valid": valid_values,
            "best_direction_index": best_values,
        }
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    siamese_controls = []
    if historical_model is not None:
        siamese_controls.append(
            (
                HISTORICAL_MM_BASELINE_ID,
                historical_model,
                historical_device,
                historical_batch_size,
                "historical_mm_30k_checkpoint",
            )
        )
    if matched_model is not None:
        siamese_controls.append(
            (
                MATCHED_ROUTE_A_SIAMESE_ID,
                matched_model,
                matched_device,
                matched_batch_size,
                "route_a_matched_training_winner",
            )
        )
    method_names = [name.value for name in _FOUR_ARMS]
    for method_name, model, device, control_batch_size, _role in siamese_controls:
        probability = score_historical_mm_real_pairs(
            model,
            dataset,
            device=device,
            batch_size=control_batch_size,
        )
        arm_values[method_name] = {
            "probability": probability.tolist(),
            "valid": [True] * len(records),
        }
        method_names.append(method_name)
    common_valid = [
        all(
            bool(arm_values[name]["valid"][index])
            and math.isfinite(float(arm_values[name]["probability"][index]))
            for name in method_names
        )
        for index in range(len(records))
    ]
    common_labels = [label for label, valid in zip(labels, common_valid) if valid]
    if not common_labels or not any(common_labels) or all(common_labels):
        raise RealDunhuangEvaluationError(
            "selected methods have no common-valid real population with both classes"
        )
    methods = {}
    for name in _FOUR_ARMS:
        values = arm_values[name.value]
        native_metrics = _metric_row(
            values["probability"], labels, values["valid"], clusters
        )
        common_metrics = _metric_row(
            values["probability"], labels, common_valid, clusters
        )
        methods[name.value] = {
            "native_valid_count": sum(values["valid"]),
            "native_metrics": native_metrics,
            "common_valid_metrics": common_metrics,
            "direction_native": _direction_metrics(
                labels=labels,
                target=targets,
                valid=values["valid"],
                predicted=values["best_direction_index"],
            ),
            "direction_common_valid": _direction_metrics(
                labels=labels,
                target=targets,
                valid=common_valid,
                predicted=values["best_direction_index"],
            ),
            "probability": [
                float(score) if valid else None
                for score, valid in zip(values["probability"], values["valid"])
            ],
            "valid": list(values["valid"]),
            "best_direction": [
                _DIRECTION_LABELS[index] if 0 <= index < 4 and valid else None
                for index, valid in zip(values["best_direction_index"], values["valid"])
            ],
        }
    for method_name, _model, _device, _batch_size, role in siamese_controls:
        values = arm_values[method_name]
        methods[method_name] = {
            "native_valid_count": len(records),
            "native_metrics": _metric_row(
                values["probability"], labels, values["valid"], clusters
            ),
            "common_valid_metrics": _metric_row(
                values["probability"], labels, common_valid, clusters
            ),
            "direction_native": None,
            "direction_common_valid": None,
            "probability": list(values["probability"]),
            "valid": list(values["valid"]),
            "best_direction": [None] * len(records),
            "score_semantics": (
                "ordered_A_then_B_historical_checkpoint_join_probability"
                if method_name == HISTORICAL_MM_BASELINE_ID
                else "ordered_A_then_B_route_a_matched_checkpoint_join_probability"
            ),
            "checkpoint_role": role,
            "model_input": "same_strict_real_alpha_masks_as_four_local_arms",
            "model_preprocessing": (
                "bool_alpha_mask_PIL_bilinear_64x64_to_single_channel_tensor"
            ),
            "direction_head_available": False,
        }
    result = {
        "schema_version": REAL_DUNHUANG_EVALUATION_VERSION,
        "status": (
            "complete_four_winners_plus_two_siamese_controls_strict_real_alpha_only"
            if len(siamese_controls) == 2
            else (
                "complete_four_winners_plus_historical_mm_strict_real_alpha_only"
                if historical_model is not None
                else (
                    "complete_four_winners_plus_matched_siamese_strict_real_alpha_only"
                    if matched_model is not None
                    else "complete_four_winners_strict_real_alpha_only"
                )
            )
        ),
        "dataset_id": REAL_DUNHUANG_DATASET_ID,
        "manifest_sha256": dataset.manifest_sha256,
        "case_count": dataset.case_count,
        "pair_count": len(records),
        "positive_count": dataset.positive_count,
        "negative_count": dataset.negative_count,
        "common_valid_count": sum(common_valid),
        "same_pair_ids_all_arms": True,
        "same_prepared_tensors_within_representation": True,
        "pair_ids": list(pair_ids),
        "labels": list(labels),
        "cluster_ids": list(clusters),
        "direction_target": [
            _DIRECTION_LABELS[index] if index >= 0 else None for index in targets
        ],
        "common_valid": common_valid,
        "preprocessing": {
            "pixel_source": "fragment_png_alpha_only_threshold_ge_128",
            "rgb_used": False,
            "gt_composite_used": False,
            "bbox_origin_or_canvas_coordinate_exposed_to_model": False,
            "bbox_used_for": "positive_direction_target_only",
            "normalization": (
                "tight_alpha_crop_then_common_scale_within_case_from_alpha_"
                "fragment_dimensions_only"
            ),
            "target_long_side": dataset.target_long_side,
            "rotation_search": False,
        },
        "excluded_categories": ["issue", "no_conjunction"],
        "methods": methods,
    }
    if siamese_controls:
        count_word = {5: "five", 6: "six"}[len(method_names)]
        result.update(
            {
                "same_pair_ids_all_methods": True,
                "same_labels_all_methods": True,
                "primary_metric_population": (
                    "intersection_valid_across_all_{}_methods".format(count_word)
                ),
                "historical_mm_checkpoint_evaluated": historical_model is not None,
                "matched_route_a_siamese_checkpoint_evaluated": (
                    matched_model is not None
                ),
            }
        )
    return result


@dataclass(frozen=True)
class _RunPlanIdentity:
    canonical_file_sha256: str
    content_sha256: str


def _winner_loader_from_run(
    run_directory: Path, *, device: Union[str, torch.device]
) -> FourArmRunWinnerLoader:
    receipt = _json_object(
        Path(run_directory) / "local_q1_run_receipt.json", "LOCAL-Q1 run receipt"
    )
    batch_plan = receipt.get("batch_plan")
    if not isinstance(batch_plan, Mapping):
        raise RealDunhuangEvaluationError("run receipt lacks batch-plan identity")
    identity = _RunPlanIdentity(
        canonical_file_sha256=_require_sha256(
            batch_plan.get("file_sha256"), "run batch-plan file SHA-256"
        ),
        content_sha256=_require_sha256(
            batch_plan.get("content_sha256"), "run batch-plan content SHA-256"
        ),
    )
    return FourArmRunWinnerLoader(
        Path(run_directory),
        identity,
        device=device,  # type: ignore[arg-type]
    )


def write_real_dunhuang_evaluation(path: Path, result: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-path-receipt", type=Path, required=True)
    parser.add_argument("--main-root", type=Path)
    parser.add_argument("--supp-root", type=Path)
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--geometry-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--historical-checkpoint", type=Path)
    parser.add_argument("--historical-batch-size", type=int, default=256)
    parser.add_argument("--matched-checkpoint", type=Path)
    parser.add_argument("--matched-batch-size", type=int, default=256)
    parser.add_argument(
        "--target-long-side", type=int, default=DEFAULT_TARGET_LONG_SIDE
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from staging.pairwise_v0_2.training.local_q1_route_a_research import (
        route_a_planning_config,
    )

    args = _parser().parse_args(argv)
    dataset = load_strict_real_pair_dataset(
        args.manifest,
        args.local_path_receipt,
        main_root=args.main_root,
        supp_root=args.supp_root,
        target_long_side=args.target_long_side,
    )
    try:
        config = route_a_planning_config(args.route_config)
        cache = GeometryArtifactCache(
            args.geometry_cache_dir, limits=GeometryCacheLimits()
        )
        loader = _winner_loader_from_run(args.run_directory, device=args.device)
        historical_model = (
            None
            if args.historical_checkpoint is None
            else load_historical_mm_checkpoint(
                args.historical_checkpoint, device=args.device
            )
        )
        matched_model = (
            None
            if args.matched_checkpoint is None
            else load_historical_mm_checkpoint(
                args.matched_checkpoint, device=args.device
            )
        )
        result = evaluate_real_dunhuang_four_arms(
            dataset=dataset,
            geometry_config=config.geometry_batch_config,
            geometry_cache=cache,
            arm_winner_loader=loader,
            batch_size=args.batch_size,
            historical_model=historical_model,
            historical_device=args.device,
            historical_batch_size=args.historical_batch_size,
            matched_model=matched_model,
            matched_device=args.device,
            matched_batch_size=args.matched_batch_size,
        )
    finally:
        dataset.mask_loader.close()
    write_real_dunhuang_evaluation(args.output, result)
    compact = {
        "status": result["status"],
        "pair_count": result["pair_count"],
        "common_valid_count": result["common_valid_count"],
        "output": str(args.output),
        "metrics": {
            method: {
                "auroc": row["common_valid_metrics"]["row"]["auroc"],
                "auprc": row["common_valid_metrics"]["row"]["auprc"],
                "direction_accuracy": (
                    None
                    if row["direction_common_valid"] is None
                    else row["direction_common_valid"][
                        "accuracy_invalid_as_incorrect"
                    ]
                ),
            }
            for method, row in result["methods"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "DEFAULT_TARGET_LONG_SIDE",
    "EXPECTED_STRICT_CASE_COUNT",
    "EXPECTED_STRICT_NEGATIVE_COUNT",
    "EXPECTED_STRICT_PAIR_COUNT",
    "EXPECTED_STRICT_POSITIVE_COUNT",
    "MATCHED_ROUTE_A_SIAMESE_ID",
    "REAL_DUNHUANG_DATASET_ID",
    "REAL_DUNHUANG_EVALUATION_VERSION",
    "RealAlphaMaskLoader",
    "RealBatchPrediction",
    "RealDunhuangEvaluationError",
    "RealPairDataset",
    "evaluate_real_dunhuang_four_arms",
    "load_strict_real_pair_dataset",
    "prepare_real_geometry_batch",
    "score_historical_mm_real_pairs",
    "score_loaded_winner_direct",
    "write_real_dunhuang_evaluation",
]


if __name__ == "__main__":
    raise SystemExit(main())

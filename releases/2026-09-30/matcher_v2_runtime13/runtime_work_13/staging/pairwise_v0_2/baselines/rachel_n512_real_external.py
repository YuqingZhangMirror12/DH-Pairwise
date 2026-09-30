"""Target-blind strict/balanced real inference for Rachel N=512 Pairwise.

This module is intentionally separate from the Rachel train/validation runner.
Winner checkpoints and their validation-fitted thresholds are frozen first;
only then may callers open the sealed real-Dunhuang manifest.  Model inputs are
derived from PNG alpha only.  A single scale is computed from each case's
parent-canvas dimensions, applied to every fragment in that case, followed by
a tight crop and centre padding to 800.  Per-fragment resizing, RGB, GT canvas
origins, correspondence targets and translation targets are never inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.optimize import linear_sum_assignment

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from torch import nn

from staging.pairwise_v0_2.baselines.rachel_matched_mm_evaluation import (
    MATCHED_METHODS,
    MATCHED_SCORE,
    FrozenMatchedMMWinner,
    MatchedMMPrediction,
    freeze_matched_mm_winners,
    score_alpha_derived_mask_pairs,
)
from staging.pairwise_v0_2.baselines import (
    rachel_same_data_benchmark_eval_adapter as benchmark_adapter,
)
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import (
    extract_ordered_outer_contour,
)
from staging.pairwise_v0_2.pairwise_data.real_dunhuang_representations import (
    RealFragmentSpec,
    load_real_external_test_spec,
)
from staging.pairwise_v0_2.training.evaluation import (
    PairwiseThresholdArtifact,
    evaluate_pairwise,
)
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed_authority


SCHEMA_VERSION = "rachel-n512-real-external/1.0"
FORMAL_COMBINED_STATUS = (
    "complete_strict_and_balanced_single_forward_population"
)
COMPATIBILITY_COMBINED_STATUS = (
    "complete_non_formal_compatibility_strict_and_balanced_single_forward_population"
)
COMPATIBILITY_STRICT_STATUS = (
    "complete_non_formal_compatibility_strict_target_blind_external_test"
)
COMPATIBILITY_BALANCED_STATUS = (
    "complete_non_formal_compatibility_balanced_target_blind_external_test"
)
RUN_SCHEMA_VERSION = "rachel-n512-train-run/1.0"
CHECKPOINT_SCHEMA_VERSION = "rachel-n512-checkpoint/1.0"
SUPPORTED_ARMS = ("coarse_only", "full_n512")
ALPHA_THRESHOLD = 128
MAX_TRUSTED_IMAGE_PIXELS = 300_000_000
REAL_DATASET_ID = "real_dunhuang_strict_alpha_v0_1"
BALANCED_SEED = "real-dunhuang-balanced-distractors-v1-fixed-20260830"
EXPECTED_STRICT_PAIR_COUNT = 547
EXPECTED_STRICT_POSITIVE_COUNT = 508
EXPECTED_STRICT_NEGATIVE_COUNT = 39
EXPECTED_REAL_FRAGMENT_COUNT = 938
EXPECTED_CONSTRUCTED_COUNT = 469
EXPECTED_BALANCED_PAIR_COUNT = 1016
EXPECTED_BALANCED_NEGATIVE_COUNT = 508
EXPECTED_CONSTRUCTED_SELECTION_SHA256 = (
    "ba469ad0bbc1f9f61c8e74bf6159b8846c1e37853724379e5cfaa1b42d60db6b"
)
Image.MAX_IMAGE_PIXELS = MAX_TRUSTED_IMAGE_PIXELS


class RachelRealExternalError(RuntimeError):
    """The frozen-winner or alpha-only external-test contract was violated."""


@dataclass(frozen=True)
class FrozenRachelWinner:
    """One validation-selected model, frozen before real data are opened."""

    arm: str
    epoch: int
    checkpoint_path: Path
    checkpoint_sha256: str
    threshold: PairwiseThresholdArtifact
    model_config: RachelN512Config
    model: nn.Module


@dataclass(frozen=True)
class PreparedRealFragment:
    """Leakage-safe 800-square mask and complete ordered contour."""

    fragment_id: str
    case_uid: str
    mask: np.ndarray
    points_rc: np.ndarray
    contour_valid: np.ndarray
    shared_case_scale: float
    tight_crop_hw: Tuple[int, int]
    alpha_sha256: str
    selection_height: int
    selection_width: int
    selection_foreground_area: int


@dataclass(frozen=True)
class StrictRealPairInput:
    """Target-blind endpoint identity in authoritative strict-manifest order."""

    pair_id: str
    case_uid: str
    fragment_a_id: str
    fragment_b_id: str


@dataclass(frozen=True)
class PreparedStrictRealPopulation:
    """Prepared inputs plus sealed labels kept outside the model call path."""

    manifest_sha256: str
    fragments: Mapping[str, PreparedRealFragment]
    pair_inputs: Tuple[StrictRealPairInput, ...]
    labels: Tuple[bool, ...]
    case_clusters: Tuple[str, ...]
    strata: Tuple[str, ...] = ()
    construction_receipt: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class TargetBlindPrediction:
    pair_id: str
    probability: float
    valid: bool
    coarse_probability: float
    coarse_valid: bool
    translation_hat_rc: Optional[Tuple[float, float]]
    translation_dispersion_px: Optional[float]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_object(path: Path, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RachelRealExternalError(description + " is not readable JSON") from error
    if not isinstance(value, Mapping):
        raise RachelRealExternalError(description + " root must be an object")
    return value


def _load_strict_alpha(fragment: RealFragmentSpec) -> np.ndarray:
    """Decode only verified-spec PNG alpha; RGB bytes are never materialized."""

    path = Path(fragment.source_path)
    if not path.is_file():
        raise RachelRealExternalError("strict real fragment file is missing")
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise RachelRealExternalError("strict real fragment must be PNG")
            width, height = image.size
            if (
                (width, height) != fragment.expected_size_wh
                or width <= 0
                or height <= 0
                or width * height > MAX_TRUSTED_IMAGE_PIXELS
            ):
                raise RachelRealExternalError(
                    "strict real fragment dimensions differ from authority"
                )
            if "A" not in image.getbands():
                raise RachelRealExternalError("strict real fragment lacks alpha")
            alpha = np.array(image.getchannel("A"), dtype=np.uint8, copy=True)
    except RachelRealExternalError:
        raise
    except (OSError, UnidentifiedImageError, ValueError, TypeError) as error:
        raise RachelRealExternalError(
            "cannot decode strict real fragment alpha"
        ) from error
    mask = np.ascontiguousarray(alpha >= ALPHA_THRESHOLD, dtype=np.bool_)
    if not bool(mask.any()):
        raise RachelRealExternalError("strict real fragment alpha is empty")
    observed_sha256 = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
    if observed_sha256 != fragment.declared_alpha_mask_sha256:
        raise RachelRealExternalError(
            "strict real fragment alpha-mask SHA-256 differs from authority"
        )
    return mask


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    return np.ascontiguousarray(
        mask[
            int(rows.min()) : int(rows.max()) + 1,
            int(columns.min()) : int(columns.max()) + 1,
        ],
        dtype=np.bool_,
    )


def _resize_bool_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    height, width = mask.shape
    output_height = max(1, int(round(height * scale)))
    output_width = max(1, int(round(width * scale)))
    if (output_height, output_width) == (height, width):
        return np.ascontiguousarray(mask, dtype=np.bool_)
    output = (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                (output_width, output_height), resample=Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        >= ALPHA_THRESHOLD
    )
    if not bool(output.any()):
        raise RachelRealExternalError("shared-scale resize removed alpha foreground")
    return np.ascontiguousarray(output, dtype=np.bool_)


def _safe_run_member(run_directory: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise RachelRealExternalError("winner checkpoint path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RachelRealExternalError("winner checkpoint path escapes the run")
    root = Path(run_directory).resolve(strict=True)
    target = (root / relative).resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise RachelRealExternalError(
            "winner checkpoint path escapes the run"
        ) from error
    if not target.is_file():
        raise RachelRealExternalError("winner checkpoint is not a file")
    return target


def freeze_completed_winners(
    run_directory: Path,
    *,
    device: Union[str, torch.device] = "cpu",
) -> Mapping[str, FrozenRachelWinner]:
    """Load winners via the sealed synthetic evaluator's pure authority gate."""

    _, _, winners = _freeze_completed_n512_authority(run_directory, device=device)
    return winners


def _freeze_completed_n512_authority(
    run_directory: Path,
    *,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Mapping[str, object], str, Mapping[str, FrozenRachelWinner]]:
    """Freeze receipt/checkpoints without accepting or opening a real path."""

    try:
        receipt, receipt_sha256, raw_winners = (
            sealed_authority._freeze_completed_winners(Path(run_directory))
        )
    except (OSError, RuntimeError, sealed_authority.RachelN512SealedTestError) as error:
        raise RachelRealExternalError(
            "Rachel winner authority freeze failed: " + str(error)
        ) from error
    target_device = torch.device(device)
    winners: Dict[str, FrozenRachelWinner] = {}
    for raw in raw_winners:
        raw.model.to(target_device).eval()
        winners[raw.arm] = FrozenRachelWinner(
            arm=raw.arm,
            epoch=raw.epoch,
            checkpoint_path=raw.checkpoint_path,
            checkpoint_sha256=raw.checkpoint_sha256,
            threshold=raw.threshold,
            model_config=raw.model_config,
            model=raw.model,
        )
    return receipt, receipt_sha256, winners


def _require_formal_n512_convergence(
    run_directory: Path,
    winners: Mapping[str, FrozenRachelWinner],
    *,
    receipt: Optional[Mapping[str, object]] = None,
) -> Mapping[str, object]:
    """Require exact N=512 architecture/config and a validation plateau."""

    root = Path(run_directory).resolve(strict=True)
    receipt_path = root / "run_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RachelRealExternalError("formal Rachel receipt is missing or symlinked")
    observed_receipt = (
        _json_object(receipt_path, "formal Rachel run receipt")
        if receipt is None
        else receipt
    )
    if set(winners) != set(SUPPORTED_ARMS):
        raise RachelRealExternalError(
            "formal real evaluation requires both Rachel arms"
        )
    if any(
        winner.model_config.canvas_size != 800
        or winner.model_config.coarse_size != 128
        or winner.model_config.contour_cap != 512
        for winner in winners.values()
    ):
        raise RachelRealExternalError(
            "formal Rachel winners must be exact 800/coarse128/N512 models"
        )
    try:
        sealed_authority._require_formal_convergence(
            observed_receipt,
            tuple(winners.values()),
        )
    except sealed_authority.RachelN512SealedTestError as error:
        raise RachelRealExternalError(str(error)) from error
    return observed_receipt


def prepare_case_alpha_masks(
    raw_masks: Mapping[str, np.ndarray],
    *,
    case_uid: str,
    canvas_wh: Tuple[int, int],
    canvas_size: int = 800,
    contour_cap: int = 512,
) -> Mapping[str, PreparedRealFragment]:
    """Apply one parent-canvas scale to every alpha mask in a real case."""

    if not raw_masks or not isinstance(case_uid, str) or not case_uid:
        raise ValueError("case_uid and raw_masks are required")
    if (
        len(canvas_wh) != 2
        or any(type(value) is not int or value <= 0 for value in canvas_wh)  # noqa: E721
        or type(canvas_size) is not int
        or canvas_size < 32
        or type(contour_cap) is not int
        or contour_cap < 4
    ):
        raise ValueError("case canvas/preparation configuration is invalid")
    parent_long_side = max(canvas_wh)
    shared_scale = canvas_size / float(parent_long_side)
    normalized_masks: Dict[str, np.ndarray] = {}
    for fragment_id, raw in raw_masks.items():
        mask = np.asarray(raw)
        if mask.ndim != 2 or mask.dtype != np.bool_ or not bool(mask.any()):
            raise RachelRealExternalError("raw alpha mask must be nonempty bool [H,W]")
        key = str(fragment_id)
        if key in normalized_masks:
            raise RachelRealExternalError("fragment IDs collide after normalization")
        normalized_masks[key] = mask
    raw_tight = {
        fragment_id: _tight_crop_mask(mask)
        for fragment_id, mask in normalized_masks.items()
    }
    selection_longest = max(max(mask.shape) for mask in raw_tight.values())
    selection_scale = canvas_size / float(selection_longest)
    selection_masks = {
        fragment_id: _resize_bool_mask(mask, selection_scale)
        for fragment_id, mask in raw_tight.items()
    }
    prepared: Dict[str, PreparedRealFragment] = {}
    for fragment_id, mask in normalized_masks.items():
        scaled = _resize_bool_mask(mask, shared_scale)
        tight = _tight_crop_mask(scaled)
        tight_height, tight_width = tight.shape
        if tight_height > canvas_size or tight_width > canvas_size:
            raise RachelRealExternalError(
                "shared-scale fragment cannot fit the frozen model canvas"
            )
        start_row = (canvas_size - tight_height) // 2
        start_column = (canvas_size - tight_width) // 2
        model_mask = np.zeros((canvas_size, canvas_size), dtype=np.bool_)
        model_mask[
            start_row : start_row + tight_height,
            start_column : start_column + tight_width,
        ] = tight
        points, valid = extract_ordered_outer_contour(model_mask, cap=contour_cap)
        padded_points = np.zeros((contour_cap, 2), dtype=np.float32)
        padded_valid = np.zeros((contour_cap,), dtype=np.bool_)
        padded_points[: len(points)] = points
        padded_valid[: len(points)] = valid
        model_mask.setflags(write=False)
        padded_points.setflags(write=False)
        padded_valid.setflags(write=False)
        selection_mask = selection_masks[fragment_id]
        prepared[fragment_id] = PreparedRealFragment(
            fragment_id=fragment_id,
            case_uid=case_uid,
            mask=model_mask,
            points_rc=padded_points,
            contour_valid=padded_valid,
            shared_case_scale=shared_scale,
            tight_crop_hw=(tight_height, tight_width),
            alpha_sha256=hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
            selection_height=int(selection_mask.shape[0]),
            selection_width=int(selection_mask.shape[1]),
            selection_foreground_area=int(np.count_nonzero(selection_mask)),
        )
    return prepared


def prepare_strict_real_population(
    manifest_path: Path,
    local_path_receipt_path: Path,
    *,
    main_root: Optional[Path] = None,
    supp_root: Optional[Path] = None,
    canvas_size: int = 800,
    contour_cap: int = 512,
) -> PreparedStrictRealPopulation:
    """Open and prepare the authoritative 547-pair strict population."""

    spec = load_real_external_test_spec(
        manifest_path,
        local_path_receipt_path,
        main_root=main_root,
        supp_root=supp_root,
        derive_positive_direction=False,
    )
    by_case: Dict[str, list] = {}
    for fragment in spec.fragments:
        by_case.setdefault(fragment.case_uid, []).append(fragment)
    fragments: Dict[str, PreparedRealFragment] = {}
    for case_uid, rows in by_case.items():
        canvas_values = {row.canvas_wh for row in rows}
        if len(canvas_values) != 1:
            raise RachelRealExternalError("case fragments disagree on parent canvas")
        raw_masks = {
            "{}/fragment/{}".format(case_uid, row.fragment_id): _load_strict_alpha(row)
            for row in rows
        }
        fragments.update(
            prepare_case_alpha_masks(
                raw_masks,
                case_uid=case_uid,
                canvas_wh=next(iter(canvas_values)),
                canvas_size=canvas_size,
                contour_cap=contour_cap,
            )
        )

    pair_inputs = tuple(
        StrictRealPairInput(
            pair_id=pair.pair_id,
            case_uid=pair.case_uid,
            fragment_a_id="{}/fragment/{}".format(pair.case_uid, pair.fragment_a_id),
            fragment_b_id="{}/fragment/{}".format(pair.case_uid, pair.fragment_b_id),
        )
        for pair in spec.pairs
    )
    return PreparedStrictRealPopulation(
        manifest_sha256=spec.manifest_sha256,
        fragments=fragments,
        pair_inputs=pair_inputs,
        labels=tuple(pair.label for pair in spec.pairs),
        case_clusters=tuple(pair.case_uid for pair in spec.pairs),
        strata=tuple(
            "strict_manifest_positive" if pair.label else "strict_manifest_negative"
            for pair in spec.pairs
        ),
    )


@dataclass(frozen=True)
class _BalancedDescriptor:
    fragment: PreparedRealFragment
    feature: Tuple[float, float, float]

    @property
    def fragment_id(self) -> str:
        return self.fragment.fragment_id

    @property
    def case_uid(self) -> str:
        return self.fragment.case_uid


def _seeded_integer(seed: str, *parts: str) -> int:
    payload = json.dumps(
        [seed, *parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _balanced_descriptors(
    fragments: Mapping[str, PreparedRealFragment],
    target_size: int,
) -> Tuple[_BalancedDescriptor, ...]:
    output = []
    for fragment_id in sorted(fragments):
        fragment = fragments[fragment_id]
        feature = (
            math.log(fragment.selection_height / float(target_size)),
            math.log(fragment.selection_width / float(target_size)),
            math.log(
                fragment.selection_foreground_area / float(target_size * target_size)
            ),
        )
        if not all(math.isfinite(value) for value in feature):
            raise RachelRealExternalError("balanced scale feature is non-finite")
        output.append(_BalancedDescriptor(fragment=fragment, feature=feature))
    return tuple(output)


def _balanced_case_partition(
    descriptors: Sequence[_BalancedDescriptor], seed: str
) -> Tuple[Tuple[_BalancedDescriptor, ...], Tuple[_BalancedDescriptor, ...]]:
    by_case: Dict[str, list] = defaultdict(list)
    for value in descriptors:
        by_case[value.case_uid].append(value)
    target_left = len(descriptors) // 2
    base_left = sum(len(values) // 2 for values in by_case.values())
    odd_cases = sorted(
        (case_uid for case_uid, values in by_case.items() if len(values) % 2),
        key=lambda case_uid: (
            _seeded_integer(seed, "odd-case-side", case_uid),
            case_uid,
        ),
    )
    extra_left = target_left - base_left
    if extra_left < 0 or extra_left > len(odd_cases):
        raise RachelRealExternalError("case-balanced bipartition is infeasible")
    left_heavy = set(odd_cases[:extra_left])
    left_quota = {
        case_uid: len(values) // 2 + int(case_uid in left_heavy)
        for case_uid, values in by_case.items()
    }
    right_quota = {
        case_uid: len(values) - left_quota[case_uid]
        for case_uid, values in by_case.items()
    }
    left = []
    right = []
    used_left: Counter = Counter()
    used_right: Counter = Counter()
    ordered = sorted(
        descriptors,
        key=lambda value: (
            value.feature[2],
            value.feature[0],
            value.feature[1],
            _seeded_integer(seed, "fragment-side-tie", value.fragment_id),
            value.fragment_id,
        ),
    )
    for value in ordered:
        case_uid = value.case_uid
        left_available = used_left[case_uid] < left_quota[case_uid]
        right_available = used_right[case_uid] < right_quota[case_uid]
        if not left_available and not right_available:
            raise RachelRealExternalError("case partition quota was exceeded")
        if left_available and right_available:
            choose_left = (
                len(left) < len(right)
                if len(left) != len(right)
                else _seeded_integer(seed, "fragment-side", value.fragment_id) % 2 == 0
            )
        else:
            choose_left = left_available
        if choose_left:
            left.append(value)
            used_left[case_uid] += 1
        else:
            right.append(value)
            used_right[case_uid] += 1
    left.sort(key=lambda value: value.fragment_id)
    right.sort(key=lambda value: value.fragment_id)
    if len(left) != len(right) or len(left) + len(right) != len(descriptors):
        raise RachelRealExternalError("balanced fragment partition changed size")
    return tuple(left), tuple(right)


def _scale_distance(first: _BalancedDescriptor, second: _BalancedDescriptor) -> float:
    difference = np.abs(
        np.asarray(first.feature, dtype=np.float64)
        - np.asarray(second.feature, dtype=np.float64)
    )
    return float(np.mean(difference))


def _optimization_cost(
    first: _BalancedDescriptor, second: _BalancedDescriptor
) -> float:
    difference = np.abs(
        np.asarray(first.feature, dtype=np.float64)
        - np.asarray(second.feature, dtype=np.float64)
    )
    return float(np.mean(difference) + np.max(difference) ** 2)


def _allowed_constructed_pair(
    first: _BalancedDescriptor, second: _BalancedDescriptor
) -> bool:
    return (
        first.case_uid != second.case_uid
        and first.fragment.alpha_sha256 != second.fragment.alpha_sha256
    )


def _case_pair_key(
    first: _BalancedDescriptor, second: _BalancedDescriptor
) -> Tuple[str, str]:
    return tuple(sorted((first.case_uid, second.case_uid)))


def _duplicate_case_pair_excess(
    left: Sequence[_BalancedDescriptor],
    right: Sequence[_BalancedDescriptor],
    assignment: Sequence[int],
) -> Tuple[int, Counter]:
    counts = Counter(
        _case_pair_key(left[index], right[right_index])
        for index, right_index in enumerate(assignment)
    )
    return sum(max(0, count - 1) for count in counts.values()), counts


def _repair_case_pairs(
    *,
    left: Sequence[_BalancedDescriptor],
    right: Sequence[_BalancedDescriptor],
    assignment: Sequence[int],
    optimization_cost: np.ndarray,
    seed: str,
) -> Tuple[Tuple[int, ...], int]:
    selected = list(int(value) for value in assignment)
    repair_count = 0
    while True:
        current_excess, counts = _duplicate_case_pair_excess(left, right, selected)
        if current_excess == 0:
            return tuple(selected), repair_count
        duplicate_indices = [
            index
            for index, right_index in enumerate(selected)
            if counts[_case_pair_key(left[index], right[right_index])] > 1
        ]
        best = None
        for first in duplicate_indices:
            first_right = selected[first]
            for second in range(len(selected)):
                if first == second:
                    continue
                second_right = selected[second]
                if not _allowed_constructed_pair(left[first], right[second_right]):
                    continue
                if not _allowed_constructed_pair(left[second], right[first_right]):
                    continue
                candidate = list(selected)
                candidate[first], candidate[second] = second_right, first_right
                new_excess, _ = _duplicate_case_pair_excess(left, right, candidate)
                if new_excess >= current_excess:
                    continue
                cost_delta = float(
                    optimization_cost[first, second_right]
                    + optimization_cost[second, first_right]
                    - optimization_cost[first, first_right]
                    - optimization_cost[second, second_right]
                )
                tie = _seeded_integer(
                    seed,
                    "case-pair-repair",
                    left[first].fragment_id,
                    right[first_right].fragment_id,
                    left[second].fragment_id,
                    right[second_right].fragment_id,
                )
                key = (new_excess, cost_delta, tie, first, second)
                if best is None or key < best[0]:
                    best = (key, candidate)
        if best is None:
            raise RachelRealExternalError(
                "cannot remove repeated constructed case pairs"
            )
        selected = best[1]
        repair_count += 1
        if repair_count > len(selected):
            raise RachelRealExternalError("case-pair repair did not converge")


def _minimum_scale_matching(
    descriptors: Sequence[_BalancedDescriptor], seed: str
) -> Tuple[Tuple[Tuple[_BalancedDescriptor, _BalancedDescriptor, float], ...], int]:
    left, right = _balanced_case_partition(descriptors, seed)
    count = len(left)
    distance = np.empty((count, count), dtype=np.float64)
    optimization_cost = np.empty((count, count), dtype=np.float64)
    allowed = np.empty((count, count), dtype=np.bool_)
    selection_cost = np.empty((count, count), dtype=np.float64)
    finite_costs = []
    for row, first in enumerate(left):
        for column, second in enumerate(right):
            distance[row, column] = _scale_distance(first, second)
            robust_value = _optimization_cost(first, second)
            optimization_cost[row, column] = robust_value
            valid = _allowed_constructed_pair(first, second)
            allowed[row, column] = valid
            if valid:
                finite_costs.append(robust_value)
                jitter = (
                    _seeded_integer(
                        seed, "hungarian-tie", first.fragment_id, second.fragment_id
                    )
                    / float(2**256)
                ) * 1e-10
                selection_cost[row, column] = robust_value + jitter
            else:
                selection_cost[row, column] = np.nan
    if not finite_costs:
        raise RachelRealExternalError("no cross-case scale matches are available")
    selection_cost[~allowed] = max(finite_costs) + 1_000_000.0
    row_index, column_index = linear_sum_assignment(selection_cost)
    if not np.array_equal(row_index, np.arange(count, dtype=row_index.dtype)):
        raise RachelRealExternalError("Hungarian matching did not cover left side")
    assignment = tuple(int(value) for value in column_index)
    if any(not allowed[index, value] for index, value in enumerate(assignment)):
        raise RachelRealExternalError("forbidden constructed edge entered matching")
    repaired, repair_count = _repair_case_pairs(
        left=left,
        right=right,
        assignment=assignment,
        optimization_cost=optimization_cost,
        seed=seed,
    )
    return (
        tuple(
            (left[index], right[right_index], float(distance[index, right_index]))
            for index, right_index in enumerate(repaired)
        ),
        repair_count,
    )


def _constructed_cluster_id(first_case: str, second_case: str) -> str:
    return "constructed-casepair/sha256/" + _canonical_sha256(
        sorted((first_case, second_case))
    )


def _real_pair_id(first_fragment: str, second_fragment: str) -> str:
    payload = json.dumps(
        [REAL_DATASET_ID, sorted((first_fragment, second_fragment))],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "pair/sha256/" + hashlib.sha256(payload).hexdigest()


def _descriptor_receipt(value: _BalancedDescriptor) -> Mapping[str, object]:
    return {
        "fragment_id": value.fragment_id,
        "source_case_uid": value.case_uid,
        "alpha_sha256": value.fragment.alpha_sha256,
        "alpha_height": value.fragment.selection_height,
        "alpha_width": value.fragment.selection_width,
        "alpha_foreground_area": value.fragment.selection_foreground_area,
    }


def _quantiles(values: Sequence[float]) -> Mapping[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def build_balanced_1016_population(
    strict: PreparedStrictRealPopulation,
    *,
    seed: str = BALANCED_SEED,
) -> PreparedStrictRealPopulation:
    """Append one deterministic cross-case distractor per fragment occurrence."""

    if (
        len(strict.pair_inputs) != EXPECTED_STRICT_PAIR_COUNT
        or len(strict.fragments) != EXPECTED_REAL_FRAGMENT_COUNT
        or sum(strict.labels) != EXPECTED_STRICT_POSITIVE_COUNT
        or len(strict.labels) - sum(strict.labels) != EXPECTED_STRICT_NEGATIVE_COUNT
    ):
        raise RachelRealExternalError("balanced construction requires strict 547")
    descriptors = _balanced_descriptors(
        strict.fragments,
        next(iter(strict.fragments.values())).mask.shape[0],
    )
    selected, repair_count = _minimum_scale_matching(descriptors, seed)
    rows_and_inputs = []
    for first, second, distance in selected:
        if (
            _seeded_integer(
                seed,
                "pair-orientation",
                *sorted((first.fragment_id, second.fragment_id)),
            )
            % 2
        ):
            first, second = second, first
        canonical = sorted((first.fragment_id, second.fragment_id))
        cluster = _constructed_cluster_id(first.case_uid, second.case_uid)
        height_ratio = max(
            first.fragment.selection_height, second.fragment.selection_height
        ) / min(first.fragment.selection_height, second.fragment.selection_height)
        width_ratio = max(
            first.fragment.selection_width, second.fragment.selection_width
        ) / min(first.fragment.selection_width, second.fragment.selection_width)
        area_ratio = max(
            first.fragment.selection_foreground_area,
            second.fragment.selection_foreground_area,
        ) / min(
            first.fragment.selection_foreground_area,
            second.fragment.selection_foreground_area,
        )
        pair_id = _real_pair_id(first.fragment_id, second.fragment_id)
        row = {
            "pair_id": pair_id,
            "canonical_pair_key": canonical,
            "constructed_is_ground_truth_negative": False,
            "case_pair_cluster_id": cluster,
            "fragment_a": _descriptor_receipt(first),
            "fragment_b": _descriptor_receipt(second),
            "alpha_scale_distance": float(distance),
            "alpha_height_ratio": float(height_ratio),
            "alpha_width_ratio": float(width_ratio),
            "alpha_foreground_area_ratio": float(area_ratio),
        }
        pair = StrictRealPairInput(
            pair_id=pair_id,
            case_uid=cluster,
            fragment_a_id=first.fragment_id,
            fragment_b_id=second.fragment_id,
        )
        rows_and_inputs.append((canonical, pair, row))
    rows_and_inputs.sort(key=lambda value: value[0])
    constructed_inputs = tuple(value[1] for value in rows_and_inputs)
    constructed_rows = tuple(value[2] for value in rows_and_inputs)
    selection_sha = _canonical_sha256(constructed_rows)
    if selection_sha != EXPECTED_CONSTRUCTED_SELECTION_SHA256:
        raise RachelRealExternalError(
            "constructed selection differs from frozen balanced authority: "
            + selection_sha
        )
    fragment_use = Counter(
        fragment_id
        for pair in constructed_inputs
        for fragment_id in (pair.fragment_a_id, pair.fragment_b_id)
    )
    case_pairs = Counter(pair.case_uid for pair in constructed_inputs)
    ratios = tuple(
        float(row[name])
        for row in constructed_rows
        for name in (
            "alpha_height_ratio",
            "alpha_width_ratio",
            "alpha_foreground_area_ratio",
        )
    )
    if (
        len(constructed_inputs) != EXPECTED_CONSTRUCTED_COUNT
        or set(fragment_use) != set(strict.fragments)
        or set(fragment_use.values()) != {1}
        or max(case_pairs.values(), default=0) != 1
        or max(ratios) > 2.0
    ):
        raise RachelRealExternalError("constructed balanced constraints changed")
    distances = [float(row["alpha_scale_distance"]) for row in constructed_rows]
    receipt = {
        "status": "complete_label_blind_constructed_distractor_plan",
        "seed": seed,
        "constructed_selection_sha256": selection_sha,
        "selection_uses_model_scores": False,
        "selection_uses_pair_labels": False,
        "selection_uses_gt_bbox_or_direction": False,
        "same_case_forbidden": True,
        "same_alpha_sha256_forbidden": True,
        "uses_per_fragment_occurrence": 1,
        "max_pairs_per_unordered_case_pair": 1,
        "case_pair_2opt_repair_count": repair_count,
        "scale_distance": _quantiles(distances),
        "constructed_pairs": constructed_rows,
        "negative_semantics": "constructed_not_GT_negative",
    }
    receipt["content_sha256"] = _canonical_sha256(receipt)
    combined_inputs = strict.pair_inputs + constructed_inputs
    combined_labels = strict.labels + (False,) * len(constructed_inputs)
    if (
        len(combined_inputs) != EXPECTED_BALANCED_PAIR_COUNT
        or sum(combined_labels) != EXPECTED_STRICT_POSITIVE_COUNT
        or len(combined_labels) - sum(combined_labels)
        != EXPECTED_BALANCED_NEGATIVE_COUNT
    ):
        raise RachelRealExternalError("balanced 1016 label cardinality changed")
    return PreparedStrictRealPopulation(
        manifest_sha256=strict.manifest_sha256,
        fragments=strict.fragments,
        pair_inputs=combined_inputs,
        labels=combined_labels,
        case_clusters=strict.case_clusters
        + tuple(pair.case_uid for pair in constructed_inputs),
        strata=strict.strata
        + ("constructed_not_GT_negative",) * len(constructed_inputs),
        construction_receipt=receipt,
    )


def score_target_blind(
    winner: FrozenRachelWinner,
    pair_inputs: Sequence[StrictRealPairInput],
    fragments: Mapping[str, PreparedRealFragment],
    *,
    batch_size: int = 1,
) -> Tuple[TargetBlindPrediction, ...]:
    """Score only masks/contours/endpoints; labels and GT never enter forward."""

    if type(batch_size) is not int or batch_size <= 0:  # noqa: E721
        raise ValueError("batch_size must be a positive integer")
    device = next(winner.model.parameters()).device
    predictions = []
    winner.model.eval()
    with torch.inference_mode():
        for start in range(0, len(pair_inputs), batch_size):
            rows = tuple(pair_inputs[start : start + batch_size])
            first = [fragments[row.fragment_a_id] for row in rows]
            second = [fragments[row.fragment_b_id] for row in rows]
            mask_a = torch.from_numpy(
                np.stack([value.mask for value in first])[:, None].copy()
            ).to(device=device, dtype=torch.float32)
            mask_b = torch.from_numpy(
                np.stack([value.mask for value in second])[:, None].copy()
            ).to(device=device, dtype=torch.float32)
            if winner.arm == "coarse_only":
                size = winner.model_config.coarse_size
                output = winner.model(
                    F.interpolate(mask_a, size=(size, size), mode="nearest"),
                    F.interpolate(mask_b, size=(size, size), mode="nearest"),
                )
                probability = output.probability
                valid = output.valid_problem
                coarse_probability = probability
                coarse_valid = valid
                translations = [None] * len(rows)
                dispersions = [None] * len(rows)
            else:
                points_a = torch.from_numpy(
                    np.stack([value.points_rc for value in first]).copy()
                ).to(device=device, dtype=torch.float32)
                points_b = torch.from_numpy(
                    np.stack([value.points_rc for value in second]).copy()
                ).to(device=device, dtype=torch.float32)
                valid_a = torch.from_numpy(
                    np.stack([value.contour_valid for value in first]).copy()
                ).to(device=device, dtype=torch.bool)
                valid_b = torch.from_numpy(
                    np.stack([value.contour_valid for value in second]).copy()
                ).to(device=device, dtype=torch.bool)
                output = winner.model(
                    mask_a, mask_b, points_a, points_b, valid_a, valid_b
                )
                probability = output.fused_probability
                valid = output.decision_valid
                coarse_probability = output.coarse_probability
                coarse_valid = output.coarse.valid_problem
                translations = [
                    tuple(float(item) for item in value)
                    for value in output.translation_hat_rc.cpu().tolist()
                ]
                dispersions = [
                    float(value) for value in output.translation_dispersion_px.cpu()
                ]
            for index, row in enumerate(rows):
                predictions.append(
                    TargetBlindPrediction(
                        pair_id=row.pair_id,
                        probability=float(probability[index].cpu()),
                        valid=bool(valid[index].cpu()),
                        coarse_probability=float(coarse_probability[index].cpu()),
                        coarse_valid=bool(coarse_valid[index].cpu()),
                        translation_hat_rc=translations[index],
                        translation_dispersion_px=dispersions[index],
                    )
                )
    return tuple(predictions)


def score_target_blind_benchmark(
    frozen: benchmark_adapter.FrozenSameDataBenchmark,
    pair_inputs: Sequence[StrictRealPairInput],
    fragments: Mapping[str, PreparedRealFragment],
    *,
    batch_size: int = 1,
) -> Tuple[TargetBlindPrediction, ...]:
    """Score one adapted benchmark from mask/contour inputs only.

    This bridge intentionally asks for no correspondence matrix on real data:
    the independent post-prediction GT evaluator can score the returned pose,
    while keeping the target-blind artifact compact.  Every input pair appears
    in exactly one batch and therefore in exactly one benchmark forward.
    """

    if type(batch_size) is not int or batch_size <= 0:  # noqa: E721
        raise ValueError("batch_size must be a positive integer")
    values = tuple(pair_inputs)
    if not values or len({row.pair_id for row in values}) != len(values):
        raise RachelRealExternalError(
            "benchmark real population is empty or has duplicate pair IDs"
        )
    predictions = []
    for start in range(0, len(values), batch_size):
        rows = values[start : start + batch_size]
        try:
            first = tuple(fragments[row.fragment_a_id] for row in rows)
            second = tuple(fragments[row.fragment_b_id] for row in rows)
        except KeyError as error:
            raise RachelRealExternalError(
                "benchmark real endpoint is missing from prepared fragments"
            ) from error
        batch = benchmark_adapter.build_target_blind_rachel_batch(
            pair_ids=tuple(row.pair_id for row in rows),
            fragment_a_tokens=tuple(row.fragment_a_id for row in rows),
            fragment_b_tokens=tuple(row.fragment_b_id for row in rows),
            masks_a=tuple(value.mask for value in first),
            masks_b=tuple(value.mask for value in second),
            points_rc_a=tuple(value.points_rc for value in first),
            points_rc_b=tuple(value.points_rc for value in second),
            contour_valid_a=tuple(value.contour_valid for value in first),
            contour_valid_b=tuple(value.contour_valid for value in second),
        )
        common = frozen.predict_batch(batch, return_correspondence=False)
        expected_ids = tuple(row.pair_id for row in rows)
        if common.pair_ids != expected_ids:
            raise RachelRealExternalError(
                "benchmark real prediction order differs from authority"
            )
        auxiliary = common.auxiliary_scores.get("coarse_cosine_similarity")
        if auxiliary is None:
            auxiliary = common.pair_probability
        for index, pair_id in enumerate(common.pair_ids):
            pose_valid = bool(common.translation_valid[index])
            translation = (
                tuple(float(item) for item in common.translation_hat_rc[index])
                if pose_valid
                else None
            )
            predictions.append(
                TargetBlindPrediction(
                    pair_id=pair_id,
                    probability=float(common.pair_probability[index]),
                    valid=bool(common.decision_valid[index]),
                    coarse_probability=float(auxiliary[index]),
                    coarse_valid=True,
                    translation_hat_rc=translation,
                    translation_dispersion_px=None,
                )
            )
    if tuple(value.pair_id for value in predictions) != tuple(
        row.pair_id for row in values
    ):
        raise RachelRealExternalError(
            "benchmark real single-forward coverage differs from authority"
        )
    return tuple(predictions)


def _coverage(valid: Sequence[bool], labels: Sequence[bool]) -> Mapping[str, object]:
    valid_array = np.asarray(valid, dtype=np.bool_)
    label_array = np.asarray(labels, dtype=np.bool_)
    if valid_array.shape != label_array.shape or valid_array.ndim != 1:
        raise ValueError("coverage vectors differ")
    positive = label_array
    negative = ~label_array
    return {
        "record_count": int(len(valid_array)),
        "valid_count": int(valid_array.sum()),
        "valid_fraction": float(valid_array.mean()),
        "positive_count": int(positive.sum()),
        "valid_positive_count": int((valid_array & positive).sum()),
        "positive_coverage": float((valid_array & positive).sum() / positive.sum()),
        "negative_count": int(negative.sum()),
        "valid_negative_count": int((valid_array & negative).sum()),
        "negative_coverage": float((valid_array & negative).sum() / negative.sum()),
    }


def _metric_view(
    probability: Sequence[float],
    labels: Sequence[bool],
    valid: Sequence[bool],
    clusters: Sequence[str],
    *,
    threshold: float,
) -> Mapping[str, object]:
    """Separate threshold-free ranking from frozen-threshold confusion."""

    coverage = _coverage(valid, labels)
    try:
        raw = evaluate_pairwise(
            probability,
            labels,
            valid,
            clusters,
            threshold=threshold,
        )
    except ValueError as error:
        return {
            "coverage": coverage,
            "ranking_primary_threshold_free": None,
            "frozen_validation_threshold_secondary": None,
            "unavailable_reason": str(error),
        }
    confusion_names = (
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "false_positive_rate",
        "f1",
        "tp_weight",
        "tn_weight",
        "fp_weight",
        "fn_weight",
    )
    return {
        "coverage": coverage,
        "ranking_primary_threshold_free": {
            "row": {
                "auroc": raw["row"]["auroc"],
                "auprc": raw["row"]["auprc"],
            },
            "case_cluster_balanced": {
                "auroc": raw["cluster_balanced"]["auroc"],
                "auprc": raw["cluster_balanced"]["auprc"],
            },
        },
        "frozen_validation_threshold_secondary": {
            "threshold": float(threshold),
            "threshold_source": "winner_validation_only_never_real_test",
            "row": {name: raw["row"][name] for name in confusion_names},
            "case_cluster_balanced": {
                name: raw["cluster_balanced"][name] for name in confusion_names
            },
        },
    }


def _translation_diagnostic(
    predictions: Sequence[TargetBlindPrediction],
) -> Mapping[str, object]:
    vectors = []
    dispersions = []
    for prediction in predictions:
        if prediction.translation_hat_rc is None:
            continue
        vector = np.asarray(prediction.translation_hat_rc, dtype=np.float64)
        if vector.shape == (2,) and np.isfinite(vector).all():
            vectors.append(float(np.linalg.norm(vector)))
            if prediction.translation_dispersion_px is not None:
                dispersion = float(prediction.translation_dispersion_px)
                if np.isfinite(dispersion):
                    dispersions.append(dispersion)

    def summary(values: Sequence[float]) -> Optional[Mapping[str, float]]:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.9)),
            "maximum": float(np.max(array)),
        }

    return {
        "supervision": "none_in_this_target_blind_pair_ranking_artifact",
        "translation_gt_read_or_reported_here": False,
        "translation_gt_evaluation": (
            "separate_independent_post_prediction_real_translation_evaluator"
        ),
        "translation_gt_availability_claim": "none",
        "used_for_accuracy_or_model_selection": False,
        "finite_estimate_count": len(vectors),
        "finite_dispersion_count": len(dispersions),
        "translation_norm_rc_px": summary(vectors),
        "transport_dispersion_px": summary(dispersions),
    }


def summarize_strict_real_predictions(
    population: PreparedStrictRealPopulation,
    winners: Mapping[str, FrozenRachelWinner],
    predictions_by_arm: Mapping[str, Sequence[TargetBlindPrediction]],
) -> Mapping[str, object]:
    """Build strict-only native/common-valid metrics in authoritative order."""

    if set(winners) != set(SUPPORTED_ARMS) or set(predictions_by_arm) != set(
        SUPPORTED_ARMS
    ):
        raise RachelRealExternalError(
            "strict comparison requires coarse_only and full_n512 winners"
        )
    pair_ids = tuple(row.pair_id for row in population.pair_inputs)
    if not pair_ids or len(pair_ids) != len(population.labels):
        raise RachelRealExternalError("strict prepared population is malformed")
    normalized: Dict[str, Tuple[TargetBlindPrediction, ...]] = {}
    for arm in SUPPORTED_ARMS:
        values = tuple(predictions_by_arm[arm])
        if tuple(value.pair_id for value in values) != pair_ids:
            raise RachelRealExternalError(
                "prediction order differs from strict authority"
            )
        if any(
            not _is_probability(value.probability)
            or not _is_probability(value.coarse_probability)
            for value in values
        ):
            raise RachelRealExternalError(
                "real prediction is non-finite or out of range"
            )
        normalized[arm] = values

    common_valid = tuple(
        all(normalized[arm][index].valid for arm in SUPPORTED_ARMS)
        for index in range(len(pair_ids))
    )
    methods: Dict[str, object] = {}
    for arm in SUPPORTED_ARMS:
        values = normalized[arm]
        probability = tuple(value.probability for value in values)
        native_valid = tuple(value.valid for value in values)
        threshold = winners[arm].threshold.threshold
        methods[arm] = {
            "native": _metric_view(
                probability,
                population.labels,
                native_valid,
                population.case_clusters,
                threshold=threshold,
            ),
            "coarse_full_common_valid": _metric_view(
                probability,
                population.labels,
                common_valid,
                population.case_clusters,
                threshold=threshold,
            ),
            "winner": {
                "epoch": winners[arm].epoch,
                "checkpoint_sha256": winners[arm].checkpoint_sha256,
                "validation_threshold": winners[arm].threshold.to_dict(),
            },
        }
        if arm == "full_n512":
            methods[arm]["translation_unsupervised_diagnostic"] = (
                _translation_diagnostic(values)
            )

    rows = []
    for index, pair in enumerate(population.pair_inputs):
        try:
            source_case_uids = sorted(
                {
                    population.fragments[pair.fragment_a_id].case_uid,
                    population.fragments[pair.fragment_b_id].case_uid,
                }
            )
        except KeyError as error:
            raise RachelRealExternalError(
                "real pair endpoint is missing from prepared fragment authority"
            ) from error
        method_rows = {}
        for arm in SUPPORTED_ARMS:
            prediction = normalized[arm][index]
            method_rows[arm] = {
                "probability": prediction.probability,
                "valid": prediction.valid,
                "decision_at_frozen_validation_threshold": (
                    prediction.probability >= winners[arm].threshold.threshold
                    if prediction.valid
                    else None
                ),
                "coarse_probability": prediction.coarse_probability,
                "coarse_valid": prediction.coarse_valid,
                "translation_hat_rc_unsupervised": prediction.translation_hat_rc,
                "translation_dispersion_px_unsupervised": (
                    prediction.translation_dispersion_px
                ),
            }
        rows.append(
            {
                "pair_id": pair.pair_id,
                "label": population.labels[index],
                "case_cluster": population.case_clusters[index],
                "source_case_uids": source_case_uids,
                "coarse_full_common_valid": common_valid[index],
                "methods": method_rows,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_strict_547_target_blind_external_test",
        "dataset": {
            "manifest_sha256": population.manifest_sha256,
            "pair_count": len(pair_ids),
            "positive_count": int(sum(population.labels)),
            "negative_count": int(len(pair_ids) - sum(population.labels)),
            "case_cluster_count": len(set(population.case_clusters)),
            "pair_order_sha256": _canonical_sha256(pair_ids),
        },
        "protocol": {
            "winner_and_validation_threshold_frozen_before_real_open": True,
            "pixel_source": "PNG_alpha_ge_128_only",
            "case_common_parent_canvas_scale": True,
            "per_fragment_independent_resize": False,
            "tight_crop_then_centerpad": True,
            "canvas_size": next(iter(winners.values())).model_config.canvas_size,
            "contour_cap": next(iter(winners.values())).model_config.contour_cap,
            "rgb_used": False,
            "bbox_or_gt_canvas_origin_exposed_to_model": False,
            "real_correspondence_or_translation_gt_read": False,
            "positive_direction_derived_or_bbox_read": False,
            "primary_metrics": "threshold_free_AUROC_AUPRC",
            "thresholded_metrics": "secondary_frozen_validation_threshold_only",
            "evaluation_history_disclosure": {
                "prior_epoch5_synthetic_test_completed": True,
                "prior_epoch5_synthetic_test_human_visible_before_continuation": True,
                "prior_epoch5_synthetic_test_used_by_automated_checkpoint_selection": False,
                "prior_epoch5_synthetic_test_used_by_automated_threshold_fitting": False,
                "prior_epoch5_synthetic_test_used_by_automated_early_stopping": False,
                "prior_epoch5_synthetic_test_used_by_automated_scheduler": False,
                "claim_no_human_cognitive_influence": False,
                "prior_epoch5_real_evaluator_started_then_stopped": True,
                "prior_epoch5_real_result_formed_or_read": False,
                "prior_convergence_time_synthetic_test_mask_morphology_review": True,
                "prior_convergence_time_synthetic_test_mask_sample_count": 500,
                "prior_convergence_time_synthetic_test_labels_read": False,
                "prior_convergence_time_synthetic_test_model_scores_read": False,
                "prior_convergence_time_synthetic_test_activity_used_for_training_checkpoint_or_threshold_selection": False,
                "prior_convergence_time_synthetic_test_activity_influenced_fixed_corrosion_conditions": False,
                "prior_convergence_time_synthetic_test_activity_used_for_corrosion_runtime_and_representation_QA": True,
                "real_data_accessed_in_that_activity": False,
                "claim_of_project_first_real_access": False,
            },
        },
        "common_valid": _coverage(common_valid, population.labels),
        "methods": methods,
        "pairs": rows,
    }


def _stratum_metric_view(
    probability: Sequence[float],
    labels: Sequence[bool],
    valid: Sequence[bool],
    clusters: Sequence[str],
    selected: Sequence[bool],
    *,
    threshold: float,
) -> Mapping[str, object]:
    indices = [index for index, keep in enumerate(selected) if keep]
    selected_probability = tuple(probability[index] for index in indices)
    selected_labels = tuple(labels[index] for index in indices)
    selected_valid = tuple(valid[index] for index in indices)
    selected_clusters = tuple(clusters[index] for index in indices)
    usable_probability = [
        selected_probability[index]
        for index, keep in enumerate(selected_valid)
        if keep and _is_probability(selected_probability[index])
    ]
    row = {
        "count": len(indices),
        "valid_count": len(usable_probability),
        "coverage": (len(usable_probability) / len(indices) if indices else None),
        "mean_probability": (
            float(np.mean(usable_probability)) if usable_probability else None
        ),
        "predicted_positive_rate_at_frozen_validation_threshold": (
            float(
                np.mean(np.asarray(usable_probability, dtype=np.float64) >= threshold)
            )
            if usable_probability
            else None
        ),
    }
    if selected_labels and any(selected_labels) and not all(selected_labels):
        row["two_class_metrics"] = _metric_view(
            selected_probability,
            selected_labels,
            selected_valid,
            selected_clusters,
            threshold=threshold,
        )
    else:
        row["two_class_metrics"] = None
        row["ranking_unavailable_reason"] = "single_class_stratum"
    return row


def summarize_balanced_1016_predictions(
    population: PreparedStrictRealPopulation,
    winners: Mapping[str, FrozenRachelWinner],
    predictions_by_arm: Mapping[str, Sequence[TargetBlindPrediction]],
) -> Mapping[str, object]:
    """Summarize balanced 1016 while preserving constructed-negative semantics."""

    if (
        len(population.pair_inputs) != EXPECTED_BALANCED_PAIR_COUNT
        or population.construction_receipt is None
        or len(population.strata) != EXPECTED_BALANCED_PAIR_COUNT
    ):
        raise RachelRealExternalError("balanced result requires frozen 1016 population")
    result = dict(
        summarize_strict_real_predictions(population, winners, predictions_by_arm)
    )
    result["status"] = "complete_balanced_1016_target_blind_external_test"
    result["dataset"] = dict(result["dataset"])
    result["dataset"].update(
        {
            "population": "strict_547_plus_469_constructed_not_GT_negative",
            "strict_prefix_count": EXPECTED_STRICT_PAIR_COUNT,
            "constructed_count": EXPECTED_CONSTRUCTED_COUNT,
            "strict_prefix_preserved_exactly": True,
        }
    )
    result["construction_receipt"] = population.construction_receipt
    result["negative_semantics"] = {
        "strict_manifest_negative_count": EXPECTED_STRICT_NEGATIVE_COUNT,
        "strict_manifest_negatives_are_GT_labelled": True,
        "constructed_count": EXPECTED_CONSTRUCTED_COUNT,
        "constructed_are_GT_negatives": False,
        "required_label": "constructed_not_GT_negative",
        "never_used_for_training_threshold_or_tuning": True,
    }
    strict_selected = tuple(
        index < EXPECTED_STRICT_PAIR_COUNT
        for index in range(EXPECTED_BALANCED_PAIR_COUNT)
    )
    constructed_selected = tuple(
        value == "constructed_not_GT_negative" for value in population.strata
    )
    common_valid = tuple(
        all(predictions_by_arm[arm][index].valid for arm in SUPPORTED_ARMS)
        for index in range(EXPECTED_BALANCED_PAIR_COUNT)
    )
    methods = dict(result["methods"])
    for arm in SUPPORTED_ARMS:
        method = dict(methods[arm])
        values = tuple(predictions_by_arm[arm])
        probability = tuple(value.probability for value in values)
        native_valid = tuple(value.valid for value in values)
        threshold = winners[arm].threshold.threshold
        method["origin_strata_native"] = {
            "strict_manifest_547": _stratum_metric_view(
                probability,
                population.labels,
                native_valid,
                population.case_clusters,
                strict_selected,
                threshold=threshold,
            ),
            "constructed_not_GT_negative": _stratum_metric_view(
                probability,
                population.labels,
                native_valid,
                population.case_clusters,
                constructed_selected,
                threshold=threshold,
            ),
        }
        method["origin_strata_coarse_full_common_valid"] = {
            "strict_manifest_547": _stratum_metric_view(
                probability,
                population.labels,
                common_valid,
                population.case_clusters,
                strict_selected,
                threshold=threshold,
            ),
            "constructed_not_GT_negative": _stratum_metric_view(
                probability,
                population.labels,
                common_valid,
                population.case_clusters,
                constructed_selected,
                threshold=threshold,
            ),
        }
        methods[arm] = method
    result["methods"] = methods
    return result


def _augment_with_matched_mm(
    base_result: Mapping[str, object],
    population: PreparedStrictRealPopulation,
    winners: Mapping[str, FrozenMatchedMMWinner],
    predictions_by_method: Mapping[str, Sequence[MatchedMMPrediction]],
    *,
    matched_run_directory: Path,
    matched_receipt: Mapping[str, object],
    matched_receipt_sha256: str,
) -> Mapping[str, object]:
    """Add both matched controls without changing the existing two-arm semantics."""

    if (
        tuple(winners) != MATCHED_METHODS
        or tuple(predictions_by_method) != MATCHED_METHODS
    ):
        raise RachelRealExternalError(
            "matched-MM converged/epoch-5 controls are incomplete"
        )
    pair_ids = tuple(row.pair_id for row in population.pair_inputs)
    normalized: Dict[str, Tuple[MatchedMMPrediction, ...]] = {}
    for method in MATCHED_METHODS:
        values = tuple(predictions_by_method[method])
        if tuple(value.pair_id for value in values) != pair_ids:
            raise RachelRealExternalError("matched-MM prediction order differs")
        if any(
            not _is_probability(value.probability) or type(value.valid) is not bool
            for value in values
        ):
            raise RachelRealExternalError("matched-MM real prediction is invalid")
        normalized[method] = values

    raw_pairs = base_result.get("pairs")
    raw_methods = base_result.get("methods")
    raw_protocol = base_result.get("protocol")
    if (
        not isinstance(raw_pairs, list)
        or len(raw_pairs) != len(pair_ids)
        or not isinstance(raw_methods, Mapping)
        or set(SUPPORTED_ARMS) - set(raw_methods)
        or not isinstance(raw_protocol, Mapping)
    ):
        raise RachelRealExternalError("base real result cannot accept matched controls")
    pairs = [dict(row) for row in raw_pairs]
    coarse_full_common = []
    for index, row in enumerate(pairs):
        if row.get("pair_id") != pair_ids[index]:
            raise RachelRealExternalError("base/matched real pair order differs")
        value = row.get("coarse_full_common_valid")
        if type(value) is not bool:  # noqa: E721
            raise RachelRealExternalError("base common-valid flag is malformed")
        coarse_full_common.append(bool(value))
    all_method_common = tuple(
        coarse_full_common[index]
        and all(normalized[method][index].valid for method in MATCHED_METHODS)
        for index in range(len(pair_ids))
    )

    methods: Dict[str, object] = {
        str(name): dict(value)
        for name, value in raw_methods.items()
        if isinstance(value, Mapping)
    }
    for arm in SUPPORTED_ARMS:
        probability = tuple(float(row["methods"][arm]["probability"]) for row in pairs)
        threshold = float(methods[arm]["winner"]["validation_threshold"]["threshold"])
        methods[arm]["all_method_common_valid"] = _metric_view(
            probability,
            population.labels,
            all_method_common,
            population.case_clusters,
            threshold=threshold,
        )

    for method in MATCHED_METHODS:
        winner = winners[method]
        values = normalized[method]
        probability = tuple(value.probability for value in values)
        native_valid = tuple(value.valid for value in values)
        threshold = winner.threshold.threshold
        methods[method] = {
            "score_semantics": MATCHED_SCORE,
            "same_exposure_epoch5": method == MATCHED_METHODS[1],
            "native": _metric_view(
                probability,
                population.labels,
                native_valid,
                population.case_clusters,
                threshold=threshold,
            ),
            "coarse_full_common_valid": _metric_view(
                probability,
                population.labels,
                coarse_full_common,
                population.case_clusters,
                threshold=threshold,
            ),
            "all_method_common_valid": _metric_view(
                probability,
                population.labels,
                all_method_common,
                population.case_clusters,
                threshold=threshold,
            ),
            "winner": {
                "epoch": winner.epoch,
                "checkpoint_sha256": winner.checkpoint_sha256,
                "model_config_sha256": winner.model_config_sha256,
                "validation_threshold": winner.threshold.to_dict(),
            },
            "input_contract": {
                "pixel_source": "prepared_PNG_alpha_ge_128_binary_mask_only",
                "preprocess": "historical_PIL_bilinear_64x64_single_channel",
                "rgb_used": False,
                "contour_or_geometry_used": False,
            },
        }
        for index, row in enumerate(pairs):
            method_rows = row.get("methods")
            if not isinstance(method_rows, Mapping):
                raise RachelRealExternalError("base real pair methods are malformed")
            method_rows = dict(method_rows)
            prediction = values[index]
            method_row = {
                "probability": prediction.probability,
                "valid": prediction.valid,
                "score_semantics": MATCHED_SCORE,
                "decision_at_frozen_validation_threshold": (
                    prediction.probability >= threshold if prediction.valid else None
                ),
            }
            if population.strata:
                method_row["stratum"] = population.strata[index]
            method_rows[method] = method_row
            row["methods"] = method_rows

    if population.strata:
        strict_selected = tuple(
            value != "constructed_not_GT_negative" for value in population.strata
        )
        constructed_selected = tuple(
            value == "constructed_not_GT_negative" for value in population.strata
        )
        for method in tuple(SUPPORTED_ARMS) + MATCHED_METHODS:
            probability = tuple(
                float(row["methods"][method]["probability"]) for row in pairs
            )
            native_valid = tuple(bool(row["methods"][method]["valid"]) for row in pairs)
            threshold = float(
                methods[method]["winner"]["validation_threshold"]["threshold"]
            )

            def views(valid: Sequence[bool]) -> Mapping[str, object]:
                return {
                    "strict_manifest_547": _stratum_metric_view(
                        probability,
                        population.labels,
                        valid,
                        population.case_clusters,
                        strict_selected,
                        threshold=threshold,
                    ),
                    "constructed_not_GT_negative": _stratum_metric_view(
                        probability,
                        population.labels,
                        valid,
                        population.case_clusters,
                        constructed_selected,
                        threshold=threshold,
                    ),
                }

            if method in MATCHED_METHODS:
                methods[method]["origin_strata_native"] = views(native_valid)
                methods[method]["origin_strata_coarse_full_common_valid"] = views(
                    coarse_full_common
                )
            methods[method]["origin_strata_all_method_common_valid"] = views(
                all_method_common
            )

    for index, row in enumerate(pairs):
        row["all_method_common_valid"] = all_method_common[index]
    protocol = dict(raw_protocol)
    protocol.update(
        {
            "all_requested_winners_and_thresholds_frozen_before_current_real_open": True,
            "matched_mm_pixel_source": "prepared_PNG_alpha_ge_128_binary_mask_only",
            "matched_mm_preprocess": "historical_PIL_bilinear_64x64_single_channel",
            "evaluation_history_disclosure": {
                "prior_epoch5_synthetic_test_completed": True,
                "prior_epoch5_synthetic_test_human_visible_before_continuation": True,
                "prior_epoch5_synthetic_test_used_by_automated_checkpoint_selection": False,
                "prior_epoch5_synthetic_test_used_by_automated_threshold_fitting": False,
                "prior_epoch5_synthetic_test_used_by_automated_early_stopping": False,
                "prior_epoch5_synthetic_test_used_by_automated_scheduler": False,
                "claim_no_human_cognitive_influence": False,
                "prior_epoch5_real_evaluator_started_then_stopped": True,
                "prior_epoch5_real_result_formed_or_read": False,
                "prior_convergence_time_synthetic_test_mask_morphology_review": True,
                "prior_convergence_time_synthetic_test_mask_sample_count": 500,
                "prior_convergence_time_synthetic_test_labels_read": False,
                "prior_convergence_time_synthetic_test_model_scores_read": False,
                "prior_convergence_time_synthetic_test_activity_used_for_training_checkpoint_or_threshold_selection": False,
                "prior_convergence_time_synthetic_test_activity_influenced_fixed_corrosion_conditions": False,
                "prior_convergence_time_synthetic_test_activity_used_for_corrosion_runtime_and_representation_QA": True,
                "real_data_accessed_in_that_activity": False,
                "claim_of_project_first_real_access": False,
            },
        }
    )
    result = dict(base_result)
    result["protocol"] = protocol
    result["methods"] = methods
    result["pairs"] = pairs
    result["all_method_common_valid"] = _coverage(all_method_common, population.labels)
    result["source_matched_mm_training_run"] = {
        "run_directory": str(matched_run_directory),
        "receipt_sha256": matched_receipt_sha256,
        "fingerprint_sha256": matched_receipt.get("fingerprint_sha256"),
        "status_at_open": matched_receipt.get("status"),
        "converged_and_epoch5_winners_frozen_before_current_real_open": True,
        "formal_config_verified": True,
    }
    return result


def _augment_with_same_data_benchmarks(
    base_result: Mapping[str, object],
    population: PreparedStrictRealPopulation,
    frozen: Mapping[str, benchmark_adapter.FrozenSameDataBenchmark],
    predictions_by_method: Mapping[str, Sequence[TargetBlindPrediction]],
    *,
    training_manifest_evidence: Mapping[str, object],
) -> Mapping[str, object]:
    """Add frozen PairingNet/ShreddingNet adaptations to one real population."""

    if (
        tuple(frozen) != benchmark_adapter.BENCHMARK_METHODS
        or tuple(predictions_by_method) != benchmark_adapter.BENCHMARK_METHODS
    ):
        raise RachelRealExternalError(
            "same-data PairingNet/ShreddingNet winner inventory is incomplete"
        )
    pair_ids = tuple(row.pair_id for row in population.pair_inputs)
    normalized: Dict[str, Tuple[TargetBlindPrediction, ...]] = {}
    for method in benchmark_adapter.BENCHMARK_METHODS:
        values = tuple(predictions_by_method[method])
        if tuple(value.pair_id for value in values) != pair_ids:
            raise RachelRealExternalError(
                "same-data benchmark real prediction order differs"
            )
        if any(
            not _is_probability(value.probability)
            or type(value.valid) is not bool  # noqa: E721
            or (
                value.translation_hat_rc is not None
                and (
                    len(value.translation_hat_rc) != 2
                    or not all(math.isfinite(item) for item in value.translation_hat_rc)
                )
            )
            for value in values
        ):
            raise RachelRealExternalError(
                "same-data benchmark real prediction is malformed"
            )
        normalized[method] = values

    raw_pairs = base_result.get("pairs")
    raw_methods = base_result.get("methods")
    raw_protocol = base_result.get("protocol")
    if (
        not isinstance(raw_pairs, list)
        or len(raw_pairs) != len(pair_ids)
        or not isinstance(raw_methods, Mapping)
        or not isinstance(raw_protocol, Mapping)
    ):
        raise RachelRealExternalError(
            "base real result cannot accept same-data benchmarks"
        )
    pairs = [dict(row) for row in raw_pairs]
    existing_methods = tuple(str(name) for name in raw_methods)
    if set(benchmark_adapter.BENCHMARK_METHODS) & set(existing_methods):
        raise RachelRealExternalError("same-data benchmark method key already exists")
    methods: Dict[str, object] = {
        str(name): dict(value)
        for name, value in raw_methods.items()
        if isinstance(value, Mapping)
    }
    if tuple(methods) != existing_methods:
        raise RachelRealExternalError("base real method metadata is malformed")

    for index, row in enumerate(pairs):
        if row.get("pair_id") != pair_ids[index]:
            raise RachelRealExternalError(
                "base/same-data benchmark real pair order differs"
            )
        method_rows_value = row.get("methods")
        if not isinstance(method_rows_value, Mapping) or set(method_rows_value) != set(
            existing_methods
        ):
            raise RachelRealExternalError("base real pair method map is malformed")
        method_rows = dict(method_rows_value)
        for method in benchmark_adapter.BENCHMARK_METHODS:
            value = normalized[method][index]
            threshold = frozen[method].threshold
            method_rows[method] = {
                "probability": value.probability,
                "valid": value.valid,
                "score_semantics": "adapted_pair_probability",
                "decision_at_frozen_validation_threshold": (
                    value.probability >= threshold if value.valid else None
                ),
                "coarse_probability": value.coarse_probability,
                "coarse_valid": value.coarse_valid,
                "translation_hat_rc_unsupervised": value.translation_hat_rc,
                "translation_dispersion_px_unsupervised": None,
                "correspondence_output": {
                    "status": "not_stored_in_target_blind_real_artifact",
                    "reason": "compact_single_forward_real_prediction_contract",
                },
            }
            if population.strata:
                method_rows[method]["stratum"] = population.strata[index]
        row["methods"] = method_rows

    all_methods = existing_methods + benchmark_adapter.BENCHMARK_METHODS
    all_method_common = tuple(
        all(bool(pairs[index]["methods"][method]["valid"]) for method in all_methods)
        for index in range(len(pairs))
    )
    coarse_full_common = tuple(
        bool(row.get("coarse_full_common_valid")) for row in pairs
    )

    def threshold_for(method: str) -> float:
        if method in benchmark_adapter.BENCHMARK_METHODS:
            return frozen[method].threshold
        winner = methods[method].get("winner")
        artifact = winner.get("validation_threshold") if isinstance(winner, Mapping) else None
        threshold = artifact.get("threshold") if isinstance(artifact, Mapping) else None
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise RachelRealExternalError(
                "base real method validation threshold is malformed"
            )
        return float(threshold)

    for method in existing_methods:
        probability = tuple(
            float(row["methods"][method]["probability"]) for row in pairs
        )
        methods[method]["all_method_common_valid"] = _metric_view(
            probability,
            population.labels,
            all_method_common,
            population.case_clusters,
            threshold=threshold_for(method),
        )

    for method in benchmark_adapter.BENCHMARK_METHODS:
        winner = frozen[method]
        values = normalized[method]
        probability = tuple(value.probability for value in values)
        native_valid = tuple(value.valid for value in values)
        methods[method] = {
            "score_semantics": "adapted_pair_probability",
            "pair_classification_diagnostic": _metric_view(
                probability,
                population.labels,
                native_valid,
                population.case_clusters,
                threshold=winner.threshold,
            ),
            "coarse_full_common_valid": _metric_view(
                probability,
                population.labels,
                coarse_full_common,
                population.case_clusters,
                threshold=winner.threshold,
            ),
            "all_method_common_valid": _metric_view(
                probability,
                population.labels,
                all_method_common,
                population.case_clusters,
                threshold=winner.threshold,
            ),
            "translation_unsupervised_diagnostic": _translation_diagnostic(values),
            "winner": {
                "method_id": winner.method_id,
                "checkpoint_sha256_by_stage": dict(
                    winner.checkpoint_sha256_by_stage
                ),
                "freeze_authority_sha256": winner.freeze_authority_sha256,
                "validation_threshold": dict(winner.threshold_artifact),
                "validation_threshold_sha256": (
                    winner.threshold_artifact_sha256
                ),
            },
            "input_contract": {
                "pixel_source": "prepared_PNG_alpha_ge_128_binary_mask_only",
                "canvas": "shared_case_scale_tight_crop_centerpad_800",
                "ordered_contour": "upright_padded_N512",
                "rgb_used": False,
                "rotation_estimated": False,
            },
            "adaptation": dict(winner.adaptation_disclosure),
            "direct_metric_status": {
                "translation_and_assembly": (
                    "deferred_to_independent_post_prediction_real_GT_evaluator"
                ),
                "rotation_error": "not_applicable_known_upright",
                "correspondence": "not_available_real_target_blind_artifact",
                "native_global_assembly_GA": "not_applicable_pairwise_only",
                "shreddingnet_native_CM_FM_SE": "not_reported",
            },
            "single_forward_population_contract": {
                "forward_pair_count": len(pair_ids),
                "unique_pair_count": len(set(pair_ids)),
                "each_pair_forwarded_exactly_once": True,
            },
            "native_cm_fm_se_or_ga_claimed": False,
        }

    if population.strata:
        strict_selected = tuple(
            value != "constructed_not_GT_negative" for value in population.strata
        )
        constructed_selected = tuple(
            value == "constructed_not_GT_negative" for value in population.strata
        )

        def views(
            method: str, valid: Sequence[bool]
        ) -> Mapping[str, object]:
            probability = tuple(
                float(row["methods"][method]["probability"]) for row in pairs
            )
            return {
                "strict_manifest_547": _stratum_metric_view(
                    probability,
                    population.labels,
                    valid,
                    population.case_clusters,
                    strict_selected,
                    threshold=threshold_for(method),
                ),
                "constructed_not_GT_negative": _stratum_metric_view(
                    probability,
                    population.labels,
                    valid,
                    population.case_clusters,
                    constructed_selected,
                    threshold=threshold_for(method),
                ),
            }

        for method in all_methods:
            native_valid = tuple(
                bool(row["methods"][method]["valid"]) for row in pairs
            )
            if method in benchmark_adapter.BENCHMARK_METHODS:
                methods[method]["origin_strata_pair_classification_diagnostic"] = (
                    views(method, native_valid)
                )
                methods[method]["origin_strata_coarse_full_common_valid"] = views(
                    method, coarse_full_common
                )
            methods[method]["origin_strata_all_method_common_valid"] = views(
                method, all_method_common
            )

    for index, row in enumerate(pairs):
        row["all_method_common_valid"] = all_method_common[index]
    protocol = dict(raw_protocol)
    protocol.update(
        {
            "same_data_benchmark_winners_and_validation_thresholds_frozen_before_current_real_open": True,
            "same_data_benchmark_mask_only": True,
            "same_data_benchmark_upright_translation_only_adaptation": True,
            "same_data_benchmark_exact_reproduction_claimed": False,
            "shreddingnet_balanced_selected_list_claimed_as_native_CM_FM_SE_or_GA": False,
        }
    )
    result = dict(base_result)
    result["protocol"] = protocol
    result["methods"] = methods
    result["pairs"] = pairs
    result["all_method_common_valid"] = _coverage(
        all_method_common, population.labels
    )
    result["source_same_data_benchmark_training_runs"] = {
        "training_manifest_evidence": dict(training_manifest_evidence),
        "methods": {
            method: frozen[method].provenance()
            for method in benchmark_adapter.BENCHMARK_METHODS
        },
        "all_winners_and_validation_thresholds_frozen_before_current_real_open": True,
    }
    return result


def _is_probability(value: float) -> bool:
    return bool(np.isfinite(value) and 0.0 <= float(value) <= 1.0)


def write_strict_real_evaluation(
    path: Path,
    result: Mapping[str, object],
    *,
    forbidden_roots: Sequence[Path] = (),
) -> None:
    """Atomically publish one finite, portable strict-real JSON result."""

    target = Path(path)
    if target.exists() or target.is_symlink():
        raise RachelRealExternalError(
            "refusing to overwrite existing strict-real output"
        )
    resolved_target = target.resolve()
    for value in forbidden_roots:
        try:
            root = Path(value).expanduser().resolve(strict=True)
            resolved_target.relative_to(root)
        except ValueError:
            continue
        except (OSError, RuntimeError) as error:
            raise RachelRealExternalError(
                "cannot validate a strict-real forbidden output root"
            ) from error
        raise RachelRealExternalError(
            "strict-real output may not be inside training or real authority roots"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + target.name + ".tmp-", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise RachelRealExternalError(
                "refusing to overwrite existing strict-real output"
            ) from error
        parent_descriptor = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _attach_n512_training_authority(
    result: Mapping[str, object],
    *,
    run_directory: Path,
    receipt: Mapping[str, object],
    receipt_sha256: str,
) -> Mapping[str, object]:
    output = dict(result)
    output["source_training_run"] = {
        "run_directory": str(Path(run_directory).resolve(strict=True)),
        "receipt": "run_receipt.json",
        "receipt_sha256": receipt_sha256,
        "fingerprint_sha256": receipt.get("fingerprint_sha256"),
        "status_at_open": receipt.get("status"),
        "winners_frozen_before_real_open": True,
        "both_arms_validation_plateau_verified": True,
        "formal_config_verified": True,
    }
    protocol_value = output.get("protocol")
    protocol = dict(protocol_value) if isinstance(protocol_value, Mapping) else {}
    protocol.update(
        {
            "formal_config_verified": True,
            "both_arms_validation_plateau_verified": True,
            "positive_direction_derived_or_bbox_read": False,
            "all_requested_winners_and_thresholds_frozen_before_current_real_open": True,
        }
    )
    output["protocol"] = protocol
    return output


def _stamp_real_evaluation_mode(
    result: Mapping[str, object], *, formal_evaluation: bool
) -> Mapping[str, object]:
    """Make formal-six and explicit compatibility receipts non-confusable."""

    expected = tuple(SUPPORTED_ARMS) + MATCHED_METHODS + benchmark_adapter.BENCHMARK_METHODS
    output = dict(result)
    is_wrapper = output.get("status") == FORMAL_COMBINED_STATUS
    if formal_evaluation:
        forward = output.get("forward_contract")
        if (
            not is_wrapper
            or not isinstance(forward, Mapping)
            or tuple(forward.get("methods", ())) != expected
        ):
            raise RachelRealExternalError(
                "formal real receipt is not exact-six balanced1016 single-forward"
            )
        for population_name in ("strict_547", "balanced_1016"):
            population = output.get(population_name)
            methods = population.get("methods") if isinstance(population, Mapping) else None
            if not isinstance(methods, Mapping) or tuple(methods) != expected:
                raise RachelRealExternalError(
                    "formal real nested method inventory is not exact-six"
                )
            stamped = dict(population)
            protocol = dict(stamped.get("protocol", {}))
            protocol.update(
                {
                    "formal_evaluation": True,
                    "compatibility_mode": False,
                    "formal_exact_six_frozen_before_real_open": True,
                    "formal_method_inventory": list(expected),
                    "balanced1016_single_forward_required": True,
                }
            )
            stamped["protocol"] = protocol
            output[population_name] = stamped
        root_protocol = dict(output.get("protocol", {}))
        root_protocol.update(
            {
                "formal_evaluation": True,
                "compatibility_mode": False,
                "formal_exact_six_frozen_before_real_open": True,
                "formal_method_inventory": list(expected),
                "balanced1016_single_forward_required": True,
            }
        )
        output["protocol"] = root_protocol
        return output

    def non_formal_population(value: Mapping[str, object], status: str) -> Dict[str, object]:
        stamped = dict(value)
        stamped["status"] = status
        protocol = dict(stamped.get("protocol", {}))
        protocol.update(
            {
                "formal_evaluation": False,
                "compatibility_mode": True,
                "formal_exact_six_frozen_before_real_open": False,
                "formal_method_inventory": None,
            }
        )
        stamped["protocol"] = protocol
        return stamped

    if is_wrapper:
        output["status"] = COMPATIBILITY_COMBINED_STATUS
        output["strict_547"] = non_formal_population(
            output["strict_547"], COMPATIBILITY_STRICT_STATUS
        )
        output["balanced_1016"] = non_formal_population(
            output["balanced_1016"], COMPATIBILITY_BALANCED_STATUS
        )
    else:
        output = non_formal_population(output, COMPATIBILITY_STRICT_STATUS)
    root_protocol = dict(output.get("protocol", {}))
    root_protocol.update(
        {
            "formal_evaluation": False,
            "compatibility_mode": True,
            "formal_exact_six_frozen_before_real_open": False,
            "formal_method_inventory": None,
        }
    )
    output["protocol"] = root_protocol
    return output


def evaluate_strict_real_external(
    run_directory: Path,
    manifest_path: Path,
    local_path_receipt_path: Path,
    *,
    matched_mm_run_directory: Optional[Path] = None,
    pairingnet_run_directory: Optional[Path] = None,
    shreddingnet_freeze_path: Optional[Path] = None,
    main_root: Optional[Path] = None,
    supp_root: Optional[Path] = None,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 1,
    output_path: Optional[Path] = None,
    include_balanced_1016: bool = False,
    compatibility_mode: bool = False,
) -> Mapping[str, object]:
    """Freeze every requested winner first, then open and score the real set."""

    # This ordering is a protocol boundary: none of the calls above this line
    # accepts a real manifest/root, and no real path is opened before every
    # winner and validation threshold has been restored and verified.
    n512_receipt, n512_receipt_sha256, winners = _freeze_completed_n512_authority(
        run_directory, device=device
    )
    n512_receipt = _require_formal_n512_convergence(
        run_directory, winners, receipt=n512_receipt
    )
    if type(include_balanced_1016) is not bool:  # noqa: E721
        raise TypeError("include_balanced_1016 must be bool")
    if type(compatibility_mode) is not bool:  # noqa: E721
        raise TypeError("compatibility_mode must be bool")
    formal_evaluation = not compatibility_mode
    if formal_evaluation and (
        matched_mm_run_directory is None
        or pairingnet_run_directory is None
        or shreddingnet_freeze_path is None
    ):
        raise RachelRealExternalError(
            "formal real evaluation requires exact-six frozen authorities"
        )
    if formal_evaluation and not include_balanced_1016:
        raise RachelRealExternalError(
            "formal real evaluation requires one balanced1016 forward population"
        )
    matched_winners: Mapping[str, FrozenMatchedMMWinner] = {}
    matched_receipt: Optional[Mapping[str, object]] = None
    matched_receipt_sha256: Optional[str] = None
    matched_root: Optional[Path] = None
    matched_training_hash_evidence: Optional[Mapping[str, object]] = None
    if matched_mm_run_directory is not None:
        matched_root = Path(matched_mm_run_directory).resolve(strict=True)
        (
            matched_receipt,
            matched_receipt_sha256,
            matched_winners,
        ) = freeze_matched_mm_winners(matched_root, device=device)
        if tuple(matched_winners) != MATCHED_METHODS:
            raise RachelRealExternalError(
                "matched-MM converged/same-exposure winners are incomplete"
            )
        try:
            sealed_authority._require_matched_training_alignment(
                n512_receipt, matched_receipt
            )
            matched_training_hash_evidence = (
                sealed_authority._matched_training_hash_evidence(
                    n512_receipt,
                    matched_receipt,
                    n512_run_directory=Path(run_directory).resolve(strict=True),
                    matched_run_directory=matched_root,
                )
            )
        except sealed_authority.RachelN512SealedTestError as error:
            raise RachelRealExternalError(str(error)) from error
    if (pairingnet_run_directory is None) != (shreddingnet_freeze_path is None):
        raise RachelRealExternalError(
            "PairingNet and ShreddingNet freeze authorities must be supplied together"
        )
    benchmark_winners: Mapping[
        str, benchmark_adapter.FrozenSameDataBenchmark
    ] = {}
    benchmark_training_manifest_evidence: Optional[Mapping[str, object]] = None
    if pairingnet_run_directory is not None:
        assert shreddingnet_freeze_path is not None
        try:
            benchmark_winners = benchmark_adapter.freeze_same_data_benchmarks(
                pairingnet_run_directory=pairingnet_run_directory,
                shreddingnet_freeze_path=shreddingnet_freeze_path,
                device=device,
            )
            dataset_root = sealed_authority._resolved_training_dataset_root(
                n512_receipt, description="Rachel N=512"
            )
            benchmark_training_manifest_evidence = (
                sealed_authority._benchmark_training_manifest_evidence(
                    dataset_root, benchmark_winners
                )
            )
        except (
            benchmark_adapter.RachelBenchmarkEvalAdapterError,
            sealed_authority.RachelN512SealedTestError,
        ) as error:
            raise RachelRealExternalError(str(error)) from error
    if formal_evaluation and (
        tuple(matched_winners) != MATCHED_METHODS
        or tuple(benchmark_winners) != benchmark_adapter.BENCHMARK_METHODS
    ):
        raise RachelRealExternalError(
            "formal real evaluation requires exact-six frozen winners"
        )
    reference_config = winners["full_n512"].model_config
    coarse_config = winners["coarse_only"].model_config
    if (
        coarse_config.canvas_size != reference_config.canvas_size
        or coarse_config.contour_cap != reference_config.contour_cap
        or coarse_config.coarse_size != reference_config.coarse_size
    ):
        raise RachelRealExternalError("winner preprocessing configurations differ")

    population = prepare_strict_real_population(
        manifest_path,
        local_path_receipt_path,
        main_root=main_root,
        supp_root=supp_root,
        canvas_size=reference_config.canvas_size,
        contour_cap=reference_config.contour_cap,
    )
    if include_balanced_1016:
        balanced = build_balanced_1016_population(population)
        balanced_predictions = {
            arm: score_target_blind(
                winners[arm],
                balanced.pair_inputs,
                balanced.fragments,
                batch_size=batch_size,
            )
            for arm in SUPPORTED_ARMS
        }
        strict_predictions = {
            arm: values[:EXPECTED_STRICT_PAIR_COUNT]
            for arm, values in balanced_predictions.items()
        }
        strict_result = summarize_strict_real_predictions(
            population, winners, strict_predictions
        )
        balanced_result = summarize_balanced_1016_predictions(
            balanced, winners, balanced_predictions
        )
        if matched_winners:
            assert matched_root is not None
            assert matched_receipt is not None
            assert matched_receipt_sha256 is not None
            matched_balanced_predictions = {
                method: score_alpha_derived_mask_pairs(
                    matched_winners[method],
                    balanced.pair_inputs,
                    balanced.fragments,
                    batch_size=batch_size,
                )
                for method in MATCHED_METHODS
            }
            matched_strict_predictions = {
                method: values[:EXPECTED_STRICT_PAIR_COUNT]
                for method, values in matched_balanced_predictions.items()
            }
            strict_result = _augment_with_matched_mm(
                strict_result,
                population,
                matched_winners,
                matched_strict_predictions,
                matched_run_directory=matched_root,
                matched_receipt=matched_receipt,
                matched_receipt_sha256=matched_receipt_sha256,
            )
            balanced_result = _augment_with_matched_mm(
                balanced_result,
                balanced,
                matched_winners,
                matched_balanced_predictions,
                matched_run_directory=matched_root,
                matched_receipt=matched_receipt,
                matched_receipt_sha256=matched_receipt_sha256,
            )
        if benchmark_winners:
            assert benchmark_training_manifest_evidence is not None
            benchmark_balanced_predictions = {
                method: score_target_blind_benchmark(
                    benchmark_winners[method],
                    balanced.pair_inputs,
                    balanced.fragments,
                    batch_size=batch_size,
                )
                for method in benchmark_adapter.BENCHMARK_METHODS
            }
            benchmark_strict_predictions = {
                method: values[:EXPECTED_STRICT_PAIR_COUNT]
                for method, values in benchmark_balanced_predictions.items()
            }
            strict_result = _augment_with_same_data_benchmarks(
                strict_result,
                population,
                benchmark_winners,
                benchmark_strict_predictions,
                training_manifest_evidence=(
                    benchmark_training_manifest_evidence
                ),
            )
            balanced_result = _augment_with_same_data_benchmarks(
                balanced_result,
                balanced,
                benchmark_winners,
                benchmark_balanced_predictions,
                training_manifest_evidence=(
                    benchmark_training_manifest_evidence
                ),
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_strict_and_balanced_single_forward_population",
            "forward_contract": {
                "forward_pair_count_per_arm": EXPECTED_BALANCED_PAIR_COUNT,
                "strict_547_derived_from_exact_prediction_prefix": True,
                "strict_pairs_forwarded_twice": False,
                "methods": list(SUPPORTED_ARMS)
                + (list(MATCHED_METHODS) if matched_winners else [])
                + (
                    list(benchmark_adapter.BENCHMARK_METHODS)
                    if benchmark_winners
                    else []
                ),
                "all_methods_share_exact_strict_prediction_prefix": True,
            },
            "strict_547": strict_result,
            "balanced_1016": balanced_result,
        }
    else:
        predictions = {
            arm: score_target_blind(
                winners[arm],
                population.pair_inputs,
                population.fragments,
                batch_size=batch_size,
            )
            for arm in SUPPORTED_ARMS
        }
        result = summarize_strict_real_predictions(population, winners, predictions)
        if matched_winners:
            assert matched_root is not None
            assert matched_receipt is not None
            assert matched_receipt_sha256 is not None
            matched_predictions = {
                method: score_alpha_derived_mask_pairs(
                    matched_winners[method],
                    population.pair_inputs,
                    population.fragments,
                    batch_size=batch_size,
                )
                for method in MATCHED_METHODS
            }
            result = _augment_with_matched_mm(
                result,
                population,
                matched_winners,
                matched_predictions,
                matched_run_directory=matched_root,
                matched_receipt=matched_receipt,
                matched_receipt_sha256=matched_receipt_sha256,
            )
        if benchmark_winners:
            assert benchmark_training_manifest_evidence is not None
            benchmark_predictions = {
                method: score_target_blind_benchmark(
                    benchmark_winners[method],
                    population.pair_inputs,
                    population.fragments,
                    batch_size=batch_size,
                )
                for method in benchmark_adapter.BENCHMARK_METHODS
            }
            result = _augment_with_same_data_benchmarks(
                result,
                population,
                benchmark_winners,
                benchmark_predictions,
                training_manifest_evidence=benchmark_training_manifest_evidence,
            )
    if result.get("status") == "complete_strict_and_balanced_single_forward_population":
        result = dict(result)
        result["strict_547"] = _attach_n512_training_authority(
            result["strict_547"],
            run_directory=run_directory,
            receipt=n512_receipt,
            receipt_sha256=n512_receipt_sha256,
        )
        result["balanced_1016"] = _attach_n512_training_authority(
            result["balanced_1016"],
            run_directory=run_directory,
            receipt=n512_receipt,
            receipt_sha256=n512_receipt_sha256,
        )
        result["source_training_run"] = dict(
            result["strict_547"]["source_training_run"]
        )
        matched_source = result["strict_547"].get("source_matched_mm_training_run")
        if isinstance(matched_source, Mapping):
            result["source_matched_mm_training_run"] = dict(matched_source)
        benchmark_source = result["strict_547"].get(
            "source_same_data_benchmark_training_runs"
        )
        if isinstance(benchmark_source, Mapping):
            result["source_same_data_benchmark_training_runs"] = dict(
                benchmark_source
            )
        result["protocol"] = {
            "formal_config_verified": True,
            "both_arms_validation_plateau_verified": True,
            "positive_direction_derived_or_bbox_read": False,
            "all_requested_winners_and_thresholds_frozen_before_current_real_open": True,
            "same_data_benchmark_methods_requested": bool(benchmark_winners),
            "same_data_benchmark_winners_and_validation_thresholds_frozen_before_current_real_open": (
                True if benchmark_winners else None
            ),
            "same_data_benchmark_mask_only": (
                True if benchmark_winners else None
            ),
            "same_data_benchmark_upright_translation_only_adaptation": (
                True if benchmark_winners else None
            ),
            "same_data_benchmark_exact_reproduction_claimed": False,
            "shreddingnet_balanced_selected_list_claimed_as_native_CM_FM_SE_or_GA": False,
        }
    else:
        result = _attach_n512_training_authority(
            result,
            run_directory=run_directory,
            receipt=n512_receipt,
            receipt_sha256=n512_receipt_sha256,
        )
    if matched_training_hash_evidence is not None:
        result = dict(result)
        result["matched_training_alignment_hash_evidence"] = (
            matched_training_hash_evidence
        )
        if result.get("status") == "complete_strict_and_balanced_single_forward_population":
            for population_name in ("strict_547", "balanced_1016"):
                population_result = dict(result[population_name])
                raw_source = population_result.get(
                    "source_matched_mm_training_run"
                )
                if not isinstance(raw_source, Mapping):
                    continue
                source = dict(raw_source)
                source["training_alignment_hash_evidence"] = (
                    matched_training_hash_evidence
                )
                population_result["source_matched_mm_training_run"] = source
                result[population_name] = population_result
        else:
            source = dict(result["source_matched_mm_training_run"])
            source["training_alignment_hash_evidence"] = (
                matched_training_hash_evidence
            )
            result["source_matched_mm_training_run"] = source
    result = _stamp_real_evaluation_mode(
        result, formal_evaluation=formal_evaluation
    )
    for winner in winners.values():
        winner.model.to("cpu")
    for winner in matched_winners.values():
        winner.model.to("cpu")
    for winner in benchmark_winners.values():
        winner.release_to_cpu()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    if output_path is not None:
        forbidden_roots = [
            Path(run_directory),
            Path(manifest_path).parent,
            Path(local_path_receipt_path).parent,
        ]
        if matched_root is not None:
            forbidden_roots.append(matched_root)
        for winner in benchmark_winners.values():
            authority = winner.freeze_authority_path
            forbidden_roots.append(
                authority if authority.is_dir() else authority.parent
            )
        forbidden_roots.extend(
            root for root in (main_root, supp_root) if root is not None
        )
        write_strict_real_evaluation(
            output_path, result, forbidden_roots=forbidden_roots
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--matched-mm-run-directory", type=Path)
    parser.add_argument("--pairingnet-run-directory", type=Path)
    parser.add_argument("--shreddingnet-freeze-path", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-path-receipt", type=Path, required=True)
    parser.add_argument("--main-root", type=Path)
    parser.add_argument("--supp-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--include-balanced-1016", action="store_true")
    parser.add_argument(
        "--compatibility-non-formal",
        action="store_true",
        help="permit legacy/non-balanced evaluation and emit only non-formal status",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    evaluate_strict_real_external(
        arguments.run_directory,
        arguments.manifest,
        arguments.local_path_receipt,
        matched_mm_run_directory=arguments.matched_mm_run_directory,
        pairingnet_run_directory=arguments.pairingnet_run_directory,
        shreddingnet_freeze_path=arguments.shreddingnet_freeze_path,
        main_root=arguments.main_root,
        supp_root=arguments.supp_root,
        device=arguments.device,
        batch_size=arguments.batch_size,
        output_path=arguments.output,
        include_balanced_1016=arguments.include_balanced_1016,
        compatibility_mode=arguments.compatibility_non_formal,
    )
    print(str(arguments.output), flush=True)
    return 0


__all__ = [
    "build_balanced_1016_population",
    "FrozenRachelWinner",
    "PreparedRealFragment",
    "PreparedStrictRealPopulation",
    "RachelRealExternalError",
    "SCHEMA_VERSION",
    "StrictRealPairInput",
    "TargetBlindPrediction",
    "evaluate_strict_real_external",
    "freeze_completed_winners",
    "main",
    "prepare_case_alpha_masks",
    "prepare_strict_real_population",
    "score_target_blind",
    "score_target_blind_benchmark",
    "summarize_balanced_1016_predictions",
    "summarize_strict_real_predictions",
    "write_strict_real_evaluation",
]


if __name__ == "__main__":
    raise SystemExit(main())

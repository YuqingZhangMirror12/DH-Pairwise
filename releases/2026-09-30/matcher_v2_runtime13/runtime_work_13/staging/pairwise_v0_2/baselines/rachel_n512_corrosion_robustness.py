"""Inference-only severe contour-corrosion evaluation for Rachel Pairwise.

The evaluator is a post-training, synthetic-test-only protocol.  Its formal
default freezes the validation-selected Rachel N=512 winners, the matched
historical-MM converged/epoch-5 winners, and the same-data PairingNet and
ShreddingNet adaptations.  It verifies every formal convergence and exact
train/validation-manifest alignment claim before it opens the already-
completed exact-six sealed synthetic result or its fixed 3,000-pair manifest.
The former four-method protocol is available only through an explicit
non-formal compatibility flag.

Every condition is applied independently to each fragment's 800x800 binary
model mask.  Corruption never sees a pair label or the other endpoint.  The
full model contour is extracted again from the corrupted mask as a complete
CCW contour capped at 512 tokens; clean contour archives are never opened and
correspondence arrays are never materialized.  After every target-blind
prediction is complete, positive target archives are opened only for their
2-D translation arrays.
Ranking is reported on one all-method, all-condition fixed common-valid
population.  Frozen validation thresholds are secondary operating points
only.  Correspondence accuracy is not reported because corrosion changes
contour-token indices; translation uses the unchanged centered-800 frame.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy import ndimage

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from torch import Tensor

from staging.pairwise_v0_2.baselines.rachel_matched_mm_evaluation import (
    MATCHED_METHODS,
    MATCHED_SCORE,
    FrozenMatchedMMWinner,
    freeze_matched_mm_winners,
)
from staging.pairwise_v0_2.baselines.rachel_matched_mm_siamese import (
    preprocess_historical_mm_mask,
)
from staging.pairwise_v0_2.baselines import (
    rachel_same_data_benchmark_eval_adapter as benchmark_adapter,
)
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Pairwise
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import (
    RachelPreprocessError,
    extract_ordered_outer_contour,
)
from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


SCHEMA_VERSION = "rachel-n512-corrosion-robustness/2.0"
PAIR_SCHEMA_VERSION = "rachel-n512-corrosion-pair/2.0"
FIXED_PROFILE = "rachel-n512-severe-contour-corrosion-fixed-v1"
EXPECTED_TEST_PAIRS = 3_000
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_BOOTSTRAP_SEED = 260_901
N512_METHODS = ("coarse_only", "full_n512")
LEGACY_METHODS = N512_METHODS + MATCHED_METHODS
BENCHMARK_METHODS = benchmark_adapter.BENCHMARK_METHODS
METHODS = LEGACY_METHODS + BENCHMARK_METHODS
TRANSLATION_METHODS = (
    "full_n512",
    benchmark_adapter.PAIRINGNET_METHOD_KEY,
    benchmark_adapter.SHREDDINGNET_METHOD_KEY,
)
MAIN_SCORE_BY_METHOD = {
    "coarse_only": "coarse",
    "full_n512": "fused",
    MATCHED_METHODS[0]: MATCHED_SCORE,
    MATCHED_METHODS[1]: MATCHED_SCORE,
    benchmark_adapter.PAIRINGNET_METHOD_KEY: "pair_probability",
    benchmark_adapter.SHREDDINGNET_METHOD_KEY: "pair_probability",
}
_BREAK_OFFSETS = (0.0, 0.5, 0.25, 0.75)


class RachelN512CorrosionError(RuntimeError):
    """A provenance, corruption, parity, or evaluation gate failed."""


@dataclass(frozen=True)
class CorrosionCondition:
    """One immutable, preregistered fragment-level mask condition."""

    name: str
    family: str
    severity: int
    erosion_radius_px: int = 0
    bite_count: int = 0
    bite_radius_px: int = 0

    def __post_init__(self) -> None:
        if not self.name or self.family not in {"clean", "erosion", "local_bites"}:
            raise ValueError("invalid corrosion condition identity")
        for field in (
            "severity",
            "erosion_radius_px",
            "bite_count",
            "bite_radius_px",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:  # noqa: E721
                raise ValueError(field + " must be a non-negative integer")
        if self.family == "clean":
            if any(
                (
                    self.severity,
                    self.erosion_radius_px,
                    self.bite_count,
                    self.bite_radius_px,
                )
            ):
                raise ValueError("clean condition cannot alter a mask")
        elif self.family == "erosion":
            if (
                self.erosion_radius_px < 1
                or self.bite_count
                or self.bite_radius_px
            ):
                raise ValueError("erosion condition parameters disagree")
        elif (
            self.bite_count not in {1, 2, 4}
            or self.bite_radius_px < 1
            or self.erosion_radius_px
        ):
            raise ValueError("local-bite condition parameters disagree")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


FIXED_CONDITIONS = (
    CorrosionCondition("clean", "clean", 0),
    CorrosionCondition("erosion_r2", "erosion", 2, erosion_radius_px=2),
    CorrosionCondition("erosion_r4", "erosion", 4, erosion_radius_px=4),
    CorrosionCondition("erosion_r8", "erosion", 8, erosion_radius_px=8),
    CorrosionCondition(
        "local_bites_k1_r8",
        "local_bites",
        1,
        bite_count=1,
        bite_radius_px=8,
    ),
    CorrosionCondition(
        "local_bites_k2_r8",
        "local_bites",
        2,
        bite_count=2,
        bite_radius_px=8,
    ),
    CorrosionCondition(
        "local_bites_k4_r8",
        "local_bites",
        4,
        bite_count=4,
        bite_radius_px=8,
    ),
)


@dataclass(frozen=True)
class RachelN512CorrosionConfig:
    """Source runs, clean reference, output, and fixed inference settings."""

    run_directory: Path
    matched_mm_run_directory: Path
    sealed_test_directory: Path
    output_root: Path
    pairingnet_run_directory: Optional[Path] = None
    shreddingnet_freeze_path: Optional[Path] = None
    dataset_root: Optional[Path] = None
    device: str = "cuda:0"
    batch_size: int = 16
    fragment_cache_size: int = 384
    clean_tolerance: float = 1e-6
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    compatibility_mode: bool = False

    def __post_init__(self) -> None:
        for field in (
            "run_directory",
            "matched_mm_run_directory",
            "sealed_test_directory",
            "output_root",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)).expanduser())
        for field in ("pairingnet_run_directory", "shreddingnet_freeze_path"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, Path(value).expanduser())
        if self.dataset_root is not None:
            object.__setattr__(
                self, "dataset_root", Path(self.dataset_root).expanduser()
            )
        for field in ("batch_size", "fragment_cache_size", "bootstrap_replicates"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(field + " must be a positive integer")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:  # noqa: E721
            raise ValueError("bootstrap_seed must be a non-negative integer")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty torch device string")
        if (
            not math.isfinite(float(self.clean_tolerance))
            or not 0.0 < float(self.clean_tolerance) <= 1e-4
        ):
            raise ValueError("clean_tolerance must be finite in (0, 1e-4]")
        if type(self.compatibility_mode) is not bool:  # noqa: E721
            raise TypeError("compatibility_mode must be bool")
        benchmark_paths = (
            self.pairingnet_run_directory,
            self.shreddingnet_freeze_path,
        )
        if self.compatibility_mode:
            if any(value is not None for value in benchmark_paths):
                raise ValueError(
                    "legacy four-method compatibility forbids benchmark authorities"
                )
        elif any(value is None for value in benchmark_paths):
            raise ValueError(
                "formal corrosion evaluation requires PairingNet and ShreddingNet authorities"
            )


@dataclass(frozen=True)
class BlindSyntheticPair:
    """Only endpoint identity and paths; no label or target can reach scoring."""

    pair_id: str
    fragment_a_id: str
    fragment_b_id: str
    mask_a_path: Path
    mask_b_path: Path


@dataclass(frozen=True)
class CorruptedFragment:
    """One corrupted mask and its newly extracted, padded N=512 contour."""

    fragment_id: str
    mask: np.ndarray
    points_rc: np.ndarray
    contour_valid: np.ndarray
    geometry_valid: bool
    geometry_failure: Optional[str]
    benchmark_points_rc: np.ndarray
    benchmark_contour_valid: np.ndarray
    benchmark_geometry_valid: bool
    benchmark_geometry_failure: Optional[str]
    component_count_8_connected: int
    historical_mask: Tensor
    mask_sha256: str


@dataclass(frozen=True)
class CorrosionPrediction:
    """One target-blind method output."""

    pair_id: str
    probability: float
    valid: bool
    translation_hat_rc: Optional[Tuple[float, float]] = None
    translation_dispersion_px: Optional[float] = None
    translation_valid: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.translation_valid is None:
            object.__setattr__(
                self,
                "translation_valid",
                self.valid and self.translation_hat_rc is not None,
            )
        elif type(self.translation_valid) is not bool:  # noqa: E721
            raise TypeError("translation_valid must be bool")
        if self.translation_valid and self.translation_hat_rc is None:
            raise ValueError("valid translation prediction is missing")


@dataclass(frozen=True)
class _ExpectedCleanPrediction:
    pair_id: str
    label: bool
    cluster_id: str
    probability: float
    valid: bool
    translation_hat_rc: Optional[Tuple[float, float]] = None
    translation_valid: Optional[bool] = None
    source_unit_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.translation_valid is None:
            object.__setattr__(
                self,
                "translation_valid",
                self.valid and self.translation_hat_rc is not None,
            )


@dataclass(frozen=True)
class _SealedReference:
    directory: Path
    receipt_sha256: str
    manifest_sha256: str
    pair_ids_sha256: str
    precision: str
    batch_size: int
    expected: Mapping[str, Tuple[_ExpectedCleanPrediction, ...]]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RachelN512CorrosionError(description + " is not readable JSON") from error
    if not isinstance(value, Mapping):
        raise RachelN512CorrosionError(description + " must be a JSON object")
    return value


def _atomic_json(path: Path, value: object) -> None:
    _publish_bytes_no_replace(path, _canonical_bytes(value) + b"\n")


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    _publish_bytes_no_replace(path, payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bytes_no_replace(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise RachelN512CorrosionError("refusing to overwrite robustness output file")
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
            raise RachelN512CorrosionError(
                "refusing to overwrite robustness output file"
            ) from error
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_directory_no_replace(
    staged: Path, final: Path, *, completion_receipt: str
) -> None:
    """Hard-link a staged tree, publishing its completion receipt last."""

    staged = Path(staged)
    final = Path(final)
    receipt_source = staged / completion_receipt
    if not receipt_source.is_file() or receipt_source.is_symlink():
        raise RachelN512CorrosionError("robustness completion receipt is missing")
    try:
        final.mkdir()
    except FileExistsError as error:
        raise RachelN512CorrosionError(
            "robustness output already exists; preserve it rather than overwrite"
        ) from error
    published = False
    try:
        directories = sorted(
            (item for item in staged.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
        )
        for directory in directories:
            if directory.is_symlink():
                raise RachelN512CorrosionError(
                    "symlinks are forbidden in staged robustness output"
                )
            (final / directory.relative_to(staged)).mkdir()
        files = sorted(item for item in staged.rglob("*") if item.is_file())
        for source in files:
            if source == receipt_source:
                continue
            if source.is_symlink():
                raise RachelN512CorrosionError(
                    "symlinks are forbidden in staged robustness output"
                )
            os.link(source, final / source.relative_to(staged))
        for directory in sorted(
            (item for item in final.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        os.link(receipt_source, final / completion_receipt)
        _fsync_directory(final)
        _fsync_directory(final.parent)
        published = True
    except FileExistsError as error:
        raise RachelN512CorrosionError(
            "refusing to overwrite robustness output during publication"
        ) from error
    finally:
        if not published:
            shutil.rmtree(final, ignore_errors=True)
    shutil.rmtree(staged)


def _safe_result_member(root: Path, value: object, description: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RachelN512CorrosionError(description + " path is invalid")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise RachelN512CorrosionError(description + " path is unsafe")
    current = root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise RachelN512CorrosionError(description + " may not be symlinked")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise RachelN512CorrosionError(description + " is missing or escapes result") from error
    if not resolved.is_file():
        raise RachelN512CorrosionError(description + " is not a regular file")
    return resolved


def _disk(radius: int) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 1, dtype=np.int64)
    rows, columns = np.meshgrid(coordinates, coordinates, indexing="ij")
    return rows**2 + columns**2 <= radius * radius


def _condition_phase(fragment_id: str, condition: CorrosionCondition, seed: int) -> float:
    # Bite locations are nested across k=1/2/4: the condition's count selects
    # a prefix of the preregistered offsets around one fragment-specific phase.
    # This avoids confounding severity with a new random boundary location.
    phase_condition = "{}:radius{}".format(
        condition.family, condition.bite_radius_px
    )
    digest = hashlib.sha256(
        "{}\0{}\0{}\0{}".format(
            FIXED_PROFILE, seed, fragment_id, phase_condition
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _circular_distance(values: np.ndarray, target: float) -> np.ndarray:
    difference = np.abs(values - float(target)) % 1.0
    return np.minimum(difference, 1.0 - difference)


def _remove_local_bites(
    mask: np.ndarray,
    *,
    fragment_id: str,
    condition: CorrosionCondition,
    seed: int,
) -> np.ndarray:
    try:
        points, _ = extract_ordered_outer_contour(
            mask, cap=512, smoothing_sigma=0.0
        )
    except (RachelPreprocessError, ValueError) as error:
        raise RachelN512CorrosionError(
            "clean fragment cannot supply a bite boundary"
        ) from error
    value = np.asarray(points, dtype=np.float64)
    next_value = np.roll(value, -1, axis=0)
    lengths = np.linalg.norm(next_value - value, axis=1)
    total = float(lengths.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise RachelN512CorrosionError("bite boundary has zero arc length")
    arc = np.concatenate(([0.0], np.cumsum(lengths[:-1]))) / total
    phase = _condition_phase(fragment_id, condition, seed)
    centers = []
    for offset in _BREAK_OFFSETS[: condition.bite_count]:
        target = (phase + offset) % 1.0
        centers.append(value[int(np.argmin(_circular_distance(arc, target)))])
    output = np.asarray(mask, dtype=np.bool_).copy()
    height, width = output.shape
    radius = condition.bite_radius_px
    for center_row, center_column in centers:
        row_start = max(0, int(math.floor(center_row - radius - 1)))
        row_stop = min(height, int(math.ceil(center_row + radius + 1)))
        column_start = max(0, int(math.floor(center_column - radius - 1)))
        column_stop = min(width, int(math.ceil(center_column + radius + 1)))
        if row_start >= row_stop or column_start >= column_stop:
            continue
        row_grid, column_grid = np.ogrid[
            row_start:row_stop, column_start:column_stop
        ]
        inside = (
            (row_grid.astype(np.float64) + 0.5 - center_row) ** 2
            + (column_grid.astype(np.float64) + 0.5 - center_column) ** 2
            <= radius * radius
        )
        view = output[row_start:row_stop, column_start:column_stop]
        view[inside] = False
    return np.ascontiguousarray(output, dtype=np.bool_)


def degrade_mask(
    mask: np.ndarray,
    *,
    fragment_id: str,
    condition: CorrosionCondition,
    seed: int,
) -> np.ndarray:
    """Apply one deterministic, pair- and label-blind binary degradation."""

    value = np.asarray(mask)
    if value.ndim != 2 or value.size == 0 or value.dtype != np.bool_:
        raise TypeError("corrosion input must be a non-empty 2D bool mask")
    if not isinstance(fragment_id, str) or not fragment_id:
        raise ValueError("fragment_id is required")
    if not isinstance(condition, CorrosionCondition):
        raise TypeError("condition must be CorrosionCondition")
    if type(seed) is not int or seed < 0:  # noqa: E721
        raise ValueError("seed must be a non-negative integer")
    if condition.family == "clean":
        output = value.copy()
    elif condition.family == "erosion":
        output = ndimage.binary_erosion(
            value,
            structure=_disk(condition.erosion_radius_px),
            border_value=0,
        )
    else:
        output = _remove_local_bites(
            value,
            fragment_id=fragment_id,
            condition=condition,
            seed=seed,
        )
    result = np.ascontiguousarray(output, dtype=np.bool_)
    result.setflags(write=False)
    return result


def _load_binary_model_mask(path: Path, canvas_size: int) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (canvas_size, canvas_size):
                raise RachelN512CorrosionError(
                    "synthetic input must be an exact model-canvas PNG"
                )
            pixels = np.asarray(image)
    except RachelN512CorrosionError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise RachelN512CorrosionError("cannot decode synthetic model mask") from error
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise RachelN512CorrosionError("synthetic model mask must be uint8 grayscale")
    unique = set(int(item) for item in np.unique(pixels))
    if unique != {0, 255}:
        raise RachelN512CorrosionError(
            "clean synthetic model mask must contain exactly binary 0/255"
        )
    output = np.ascontiguousarray(pixels == 255, dtype=np.bool_)
    output.setflags(write=False)
    return output


class _CleanSourceAuthority:
    """Share immutable clean PNG/semantic hashes across every condition cache."""

    def __init__(self) -> None:
        self._records: Dict[str, Tuple[Path, str, str]] = {}
        self.verification_count = 0

    def load(self, fragment_id: str, path: Path, canvas_size: int) -> np.ndarray:
        file_sha256 = _sha256_file(path)
        mask = _load_binary_model_mask(path, canvas_size)
        if _sha256_file(path) != file_sha256:
            raise RachelN512CorrosionError(
                "clean source PNG changed while its semantic mask was decoded"
            )
        semantic_sha256 = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
        observed = (path, file_sha256, semantic_sha256)
        expected = self._records.setdefault(fragment_id, observed)
        if expected != observed:
            raise RachelN512CorrosionError(
                "clean source PNG bytes or decoded semantic mask changed across conditions"
            )
        self.verification_count += 1
        return mask

    def receipt(self) -> Mapping[str, object]:
        rows = self.rows()
        return {
            "unique_fragment_count": len(rows),
            "cross_condition_and_reload_verification_count": self.verification_count,
            "clean_source_authority_sha256": _canonical_sha256(rows),
            "raw_png_bytes_and_decoded_bool_semantics_both_frozen": True,
        }

    def rows(self) -> Tuple[Mapping[str, object], ...]:
        return tuple(
            {
                "fragment_id": fragment_id,
                "source_png_sha256": value[1],
                "clean_semantic_mask_sha256": value[2],
            }
            for fragment_id, value in sorted(self._records.items())
        )


def _ordered_polygon_area_rc(points_rc: np.ndarray) -> float:
    """PairingNet-compatible OpenCV-style int32 contour area."""

    points = np.asarray(points_rc, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or len(points) < 3
        or not np.all(np.isfinite(points))
    ):
        raise RachelN512CorrosionError(
            "PairingNet-compatible contour area input differs"
        )
    quantized = points.astype(np.int32).astype(np.float64)
    rows = quantized[:, 0]
    columns = quantized[:, 1]
    area = 0.5 * abs(
        float(np.dot(columns, np.roll(rows, -1)))
        - float(np.dot(rows, np.roll(columns, -1)))
    )
    if not math.isfinite(area) or area <= 0.0:
        raise RachelN512CorrosionError(
            "PairingNet-compatible contour area is non-positive"
        )
    return area


class _ConditionFragmentCache:
    """Bounded cache with reproducibility checks across any evictions."""

    def __init__(
        self,
        *,
        condition: CorrosionCondition,
        seed: int,
        canvas_size: int,
        contour_cap: int,
        capacity: int,
        clean_source_authority: Optional[_CleanSourceAuthority] = None,
    ) -> None:
        self.condition = condition
        self.seed = seed
        self.canvas_size = canvas_size
        self.contour_cap = contour_cap
        self.capacity = capacity
        self.clean_source_authority = (
            clean_source_authority
            if clean_source_authority is not None
            else _CleanSourceAuthority()
        )
        self._cache: OrderedDict[str, CorruptedFragment] = OrderedDict()
        self._paths: Dict[str, Path] = {}
        self._digests: Dict[str, str] = {}
        self._benchmark_areas: Dict[str, float] = {}
        self._geometry: Dict[
            str, Tuple[bool, Optional[str], int, int, bool, Optional[str]]
        ] = {}
        self.load_count = 0
        self.hit_count = 0

    def get(self, fragment_id: str, path: Path) -> CorruptedFragment:
        registered = self._paths.get(fragment_id)
        if registered is not None and registered != path:
            raise RachelN512CorrosionError(
                "one fragment_id resolved to different mask paths"
            )
        self._paths[fragment_id] = path
        cached = self._cache.pop(fragment_id, None)
        if cached is not None:
            self.hit_count += 1
            self._cache[fragment_id] = cached
            return cached
        clean = self.clean_source_authority.load(
            fragment_id, path, self.canvas_size
        )
        mask = degrade_mask(
            clean,
            fragment_id=fragment_id,
            condition=self.condition,
            seed=self.seed,
        )
        digest = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
        previous = self._digests.setdefault(fragment_id, digest)
        if previous != digest:
            raise RachelN512CorrosionError(
                "fragment corruption changed after a cache eviction"
            )
        points = np.zeros((self.contour_cap, 2), dtype=np.float32)
        valid = np.zeros((self.contour_cap,), dtype=np.bool_)
        benchmark_points = np.zeros((self.contour_cap, 2), dtype=np.float32)
        benchmark_valid = np.zeros((self.contour_cap,), dtype=np.bool_)
        geometry_valid = False
        failure: Optional[str] = None
        benchmark_geometry_valid = False
        benchmark_failure: Optional[str] = None
        _, component_count = ndimage.label(
            mask, structure=np.ones((3, 3), dtype=np.bool_)
        )
        component_count = int(component_count)
        try:
            extracted, extracted_valid = extract_ordered_outer_contour(
                mask,
                cap=self.contour_cap,
                smoothing_sigma=3.0,
            )
            if len(extracted) < 4 or not np.all(extracted_valid):
                raise RachelPreprocessError("extracted contour is incomplete")
            benchmark_points[: len(extracted)] = extracted
            benchmark_valid[: len(extracted)] = extracted_valid
            benchmark_geometry_valid = True
            benchmark_area = _ordered_polygon_area_rc(extracted)
        except (
            RachelPreprocessError,
            RachelN512CorrosionError,
            ValueError,
        ) as error:
            benchmark_failure = type(error).__name__ + ":" + str(error)
            benchmark_area = float("nan")
        if component_count != 1:
            failure = "component_count_8_connected_{}_not_exactly_one".format(
                component_count
            )
        elif benchmark_geometry_valid:
            points[:] = benchmark_points
            valid[:] = benchmark_valid
            geometry_valid = True
        else:
            failure = benchmark_failure
        points.setflags(write=False)
        valid.setflags(write=False)
        benchmark_points.setflags(write=False)
        benchmark_valid.setflags(write=False)
        historical = preprocess_historical_mm_mask(mask).contiguous()
        fragment = CorruptedFragment(
            fragment_id=fragment_id,
            mask=mask,
            points_rc=points,
            contour_valid=valid,
            geometry_valid=geometry_valid,
            geometry_failure=failure,
            benchmark_points_rc=benchmark_points,
            benchmark_contour_valid=benchmark_valid,
            benchmark_geometry_valid=benchmark_geometry_valid,
            benchmark_geometry_failure=benchmark_failure,
            component_count_8_connected=component_count,
            historical_mask=historical,
            mask_sha256=digest,
        )
        state = (
            geometry_valid,
            failure,
            int(np.count_nonzero(mask)),
            component_count,
            benchmark_geometry_valid,
            benchmark_failure,
        )
        previous_state = self._geometry.setdefault(fragment_id, state)
        if previous_state != state:
            raise RachelN512CorrosionError(
                "fragment geometry changed after a cache eviction"
            )
        previous_area = self._benchmark_areas.setdefault(fragment_id, benchmark_area)
        if not (
            (math.isnan(previous_area) and math.isnan(benchmark_area))
            or previous_area == benchmark_area
        ):
            raise RachelN512CorrosionError(
                "fragment benchmark contour area changed after a cache eviction"
            )
        self._cache[fragment_id] = fragment
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        self.load_count += 1
        return fragment

    def receipt(self) -> Mapping[str, object]:
        failures: Dict[str, int] = {}
        component_counts: Dict[str, int] = {}
        benchmark_failures: Dict[str, int] = {}
        for (
            valid,
            reason,
            _,
            component_count,
            benchmark_geometry_valid,
            benchmark_reason,
        ) in self._geometry.values():
            if not valid:
                key = reason or "unknown_geometry_failure"
                failures[key] = failures.get(key, 0) + 1
            if not benchmark_geometry_valid:
                key = benchmark_reason or "unknown_benchmark_geometry_failure"
                benchmark_failures[key] = benchmark_failures.get(key, 0) + 1
            count_key = str(component_count)
            component_counts[count_key] = component_counts.get(count_key, 0) + 1
        manifest = [
            {
                "fragment_id": fragment_id,
                "mask_sha256": self._digests[fragment_id],
                "geometry_valid": self._geometry[fragment_id][0],
                "foreground_area": self._geometry[fragment_id][2],
                "component_count_8_connected": self._geometry[fragment_id][3],
                "benchmark_geometry_valid": self._geometry[fragment_id][4],
                "benchmark_contour_area_px2": (
                    self._benchmark_areas[fragment_id]
                    if math.isfinite(self._benchmark_areas[fragment_id])
                    else None
                ),
            }
            for fragment_id in sorted(self._digests)
        ]
        return {
            "unique_fragment_count": len(self._digests),
            "decode_and_corrupt_count_including_evictions": self.load_count,
            "cache_hit_count": self.hit_count,
            "cache_capacity": self.capacity,
            "geometry_valid_fragment_count": sum(
                row[0] for row in self._geometry.values()
            ),
            "geometry_failures": failures,
            "benchmark_geometry_valid_fragment_count": sum(
                row[4] for row in self._geometry.values()
            ),
            "benchmark_geometry_failures": benchmark_failures,
            "component_count_8_connected_distribution": component_counts,
            "n512_geometry_requires_exactly_one_8_connected_component": True,
            "coarse_and_matched_receive_entire_corrupted_bool_mask": True,
            "pairingnet_and_shreddingnet_receive_entire_corrupted_bool_mask": True,
            "benchmark_contour_uses_outer_contour_without_component_rejection": True,
            "fragment_output_manifest_sha256": _canonical_sha256(manifest),
        }

    def fragment_audit(self) -> Mapping[str, Mapping[str, object]]:
        return {
            fragment_id: {
                "component_count_8_connected": state[3],
                "n512_geometry_valid": state[0],
                "n512_geometry_failure": state[1],
                "foreground_area": state[2],
                "benchmark_geometry_valid": state[4],
                "benchmark_geometry_failure": state[5],
                "benchmark_contour_area_px2": (
                    self._benchmark_areas[fragment_id]
                    if math.isfinite(self._benchmark_areas[fragment_id])
                    else None
                ),
                "corrupted_mask_sha256": self._digests[fragment_id],
            }
            for fragment_id, state in self._geometry.items()
        }


def _validated_probability(value: Tensor, valid: Tensor, index: int) -> Tuple[float, bool]:
    probability = float(value[index].detach().float().cpu())
    usable = bool(valid[index].detach().cpu()) and math.isfinite(probability)
    if usable and not 0.0 <= probability <= 1.0:
        raise RachelN512CorrosionError("model returned probability outside [0, 1]")
    return probability, usable


def _score_condition(
    pair_inputs: Sequence[BlindSyntheticPair],
    *,
    fragment_cache: _ConditionFragmentCache,
    n512_winners: Mapping[str, object],
    matched_winners: Mapping[str, FrozenMatchedMMWinner],
    benchmark_winners: Mapping[str, benchmark_adapter.FrozenSameDataBenchmark],
    methods: Sequence[str],
    device: torch.device,
    precision: str,
    batch_size: int,
) -> Tuple[
    Mapping[str, Tuple[CorrosionPrediction, ...]], Mapping[str, object]
]:
    """Score masks/endpoints only; labels and GT are absent from this API."""

    pairs = tuple(pair_inputs)
    if not pairs or len({row.pair_id for row in pairs}) != len(pairs):
        raise RachelN512CorrosionError("blind synthetic pair IDs are malformed")
    method_order = tuple(methods)
    if method_order not in {METHODS, LEGACY_METHODS}:
        raise RachelN512CorrosionError("corrosion method inventory is invalid")
    if set(n512_winners) != set(N512_METHODS) or tuple(matched_winners) != MATCHED_METHODS:
        raise RachelN512CorrosionError("frozen method set is incomplete")
    if method_order == METHODS:
        if tuple(benchmark_winners) != BENCHMARK_METHODS:
            raise RachelN512CorrosionError("frozen exact-six benchmark set is incomplete")
    elif benchmark_winners:
        raise RachelN512CorrosionError(
            "legacy four-method compatibility cannot score benchmark methods"
        )
    for winner in n512_winners.values():
        winner.model.to(device).eval()
    for winner in matched_winners.values():
        winner.model.to(device).eval()
    predictions: Dict[str, list] = {method: [] for method in method_order}
    batch_forward_count = {method: 0 for method in method_order}
    pair_forward_count = {method: 0 for method in method_order}
    full_model = n512_winners["full_n512"].model
    if not isinstance(full_model, RachelN512Pairwise):
        raise RachelN512CorrosionError("full_n512 winner restored wrong model type")

    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            rows = pairs[start : start + batch_size]
            first = [
                fragment_cache.get(row.fragment_a_id, row.mask_a_path) for row in rows
            ]
            second = [
                fragment_cache.get(row.fragment_b_id, row.mask_b_path) for row in rows
            ]
            mask_a = torch.from_numpy(
                np.stack([value.mask for value in first])[:, None].copy()
            ).to(device=device, dtype=torch.float32)
            mask_b = torch.from_numpy(
                np.stack([value.mask for value in second])[:, None].copy()
            ).to(device=device, dtype=torch.float32)
            coarse_winner = n512_winners["coarse_only"]
            coarse_size = coarse_winner.model_config.coarse_size
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                coarse_output = coarse_winner.model(
                    F.interpolate(mask_a, size=(coarse_size, coarse_size), mode="nearest"),
                    F.interpolate(mask_b, size=(coarse_size, coarse_size), mode="nearest"),
                )
            batch_forward_count["coarse_only"] += 1
            pair_forward_count["coarse_only"] += len(rows)
            for index, row in enumerate(rows):
                probability, valid = _validated_probability(
                    coarse_output.probability, coarse_output.valid_problem, index
                )
                predictions["coarse_only"].append(
                    CorrosionPrediction(row.pair_id, probability, valid)
                )

            if any(
                not value.benchmark_geometry_valid for value in first + second
            ):
                raise RachelN512CorrosionError(
                    "a corrupted fragment cannot supply the common benchmark contour"
                )
            points_a_np = np.stack(
                [value.benchmark_points_rc for value in first]
            ).copy()
            points_b_np = np.stack(
                [value.benchmark_points_rc for value in second]
            ).copy()
            valid_a_np = np.stack(
                [value.benchmark_contour_valid for value in first]
            ).copy()
            valid_b_np = np.stack(
                [value.benchmark_contour_valid for value in second]
            ).copy()
            points_a = torch.from_numpy(points_a_np).to(
                device=device, dtype=torch.float32
            )
            points_b = torch.from_numpy(points_b_np).to(
                device=device, dtype=torch.float32
            )
            valid_a = torch.from_numpy(valid_a_np).to(
                device=device, dtype=torch.bool
            )
            valid_b = torch.from_numpy(valid_b_np).to(
                device=device, dtype=torch.bool
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                full_output = full_model(
                    mask_a,
                    mask_b,
                    points_a,
                    points_b,
                    valid_a,
                    valid_b,
                )
            batch_forward_count["full_n512"] += 1
            pair_forward_count["full_n512"] += len(rows)
            translations = full_output.translation_hat_rc.detach().float().cpu()
            dispersions = full_output.translation_dispersion_px.detach().float().cpu()
            for index, row in enumerate(rows):
                probability, model_valid = _validated_probability(
                    full_output.fused_probability,
                    full_output.decision_valid,
                    index,
                )
                exact_component_valid = (
                    first[index].geometry_valid and second[index].geometry_valid
                )
                valid = model_valid and exact_component_valid
                translation = None
                dispersion = None
                if valid:
                    translation_value = translations[index]
                    dispersion_value = float(dispersions[index])
                    if not torch.isfinite(translation_value).all() or not math.isfinite(
                        dispersion_value
                    ):
                        raise RachelN512CorrosionError(
                            "valid full geometry output is non-finite"
                        )
                    translation = tuple(float(item) for item in translation_value)
                    dispersion = dispersion_value
                predictions["full_n512"].append(
                    CorrosionPrediction(
                        row.pair_id,
                        probability if valid else float("nan"),
                        valid,
                        translation,
                        dispersion,
                        valid,
                    )
                )

            historical_a = torch.stack([value.historical_mask for value in first]).to(
                device=device, dtype=torch.float32
            )
            historical_b = torch.stack([value.historical_mask for value in second]).to(
                device=device, dtype=torch.float32
            )
            for method in MATCHED_METHODS:
                winner = matched_winners[method]
                probability = winner.model(historical_a, historical_b)[:, 0]
                batch_forward_count[method] += 1
                pair_forward_count[method] += len(rows)
                finite = torch.isfinite(probability)
                if not finite.all() or ((probability < 0.0) | (probability > 1.0)).any():
                    raise RachelN512CorrosionError(
                        method + " returned an invalid historical-MM probability"
                    )
                for index, row in enumerate(rows):
                    predictions[method].append(
                        CorrosionPrediction(
                            row.pair_id,
                            float(probability[index].detach().float().cpu()),
                            True,
                        )
                    )
            if method_order == METHODS:
                benchmark_batch = benchmark_adapter.build_target_blind_rachel_batch(
                    pair_ids=tuple(row.pair_id for row in rows),
                    fragment_a_tokens=tuple(row.fragment_a_id for row in rows),
                    fragment_b_tokens=tuple(row.fragment_b_id for row in rows),
                    masks_a=tuple(value.mask for value in first),
                    masks_b=tuple(value.mask for value in second),
                    points_rc_a=tuple(value.benchmark_points_rc for value in first),
                    points_rc_b=tuple(value.benchmark_points_rc for value in second),
                    contour_valid_a=tuple(
                        value.benchmark_contour_valid for value in first
                    ),
                    contour_valid_b=tuple(
                        value.benchmark_contour_valid for value in second
                    ),
                )
                expected_mask_a = np.stack([value.mask for value in first])[:, None]
                expected_mask_b = np.stack([value.mask for value in second])[:, None]
                if not np.array_equal(benchmark_batch.mask_a, expected_mask_a) or not np.array_equal(
                    benchmark_batch.mask_b, expected_mask_b
                ):
                    raise RachelN512CorrosionError(
                        "benchmark adapter changed the common corrupted masks"
                    )
                for method in BENCHMARK_METHODS:
                    output = benchmark_winners[method].predict_batch(
                        benchmark_batch, return_correspondence=False
                    )
                    batch_forward_count[method] += 1
                    pair_forward_count[method] += len(rows)
                    if (
                        output.method_key != method
                        or output.pair_ids != tuple(row.pair_id for row in rows)
                    ):
                        raise RachelN512CorrosionError(
                            method + " corrosion prediction identity/order changed"
                        )
                    for index, row in enumerate(rows):
                        valid = bool(output.decision_valid[index])
                        probability = float(output.pair_probability[index])
                        translation_valid = bool(output.translation_valid[index])
                        translation = (
                            tuple(
                                float(item)
                                for item in output.translation_hat_rc[index]
                            )
                            if translation_valid
                            else None
                        )
                        predictions[method].append(
                            CorrosionPrediction(
                                pair_id=row.pair_id,
                                probability=probability,
                                valid=valid,
                                translation_hat_rc=translation,
                                translation_valid=translation_valid,
                            )
                        )
    expected_ids = tuple(row.pair_id for row in pairs)
    for method in method_order:
        observed_ids = tuple(row.pair_id for row in predictions[method])
        if observed_ids != expected_ids:
            raise RachelN512CorrosionError(method + " prediction order changed")
        if pair_forward_count[method] != len(pairs):
            raise RachelN512CorrosionError(method + " did not forward each pair exactly once")
    return (
        {method: tuple(predictions[method]) for method in method_order},
        {
            "method_order": list(method_order),
            "pair_count": len(pairs),
            "pair_forward_count_by_method": pair_forward_count,
            "batch_forward_count_by_method": batch_forward_count,
            "each_method_forwarded_each_pair_exactly_once": True,
            "same_common_corrupted_mask_arrays_used_by_all_methods": True,
            "benchmark_correspondence_output_requested": False,
        },
    )


def _parse_expected_pair_scores(
    path: Path,
    *,
    method: str,
    expected_sha256: object,
    expected_count: int,
) -> Tuple[_ExpectedCleanPrediction, ...]:
    if method not in MAIN_SCORE_BY_METHOD:
        raise RachelN512CorrosionError("unknown clean-reference method")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RachelN512CorrosionError(method + " score SHA-256 is malformed")
    if _sha256_file(path) != expected_sha256:
        raise RachelN512CorrosionError(method + " sealed score SHA-256 differs")
    rows = []
    seen = set()
    main_score = MAIN_SCORE_BY_METHOD[method]
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise RachelN512CorrosionError(
                        method + " sealed score line is not an object"
                    )
                pair_id = value.get("pair_id")
                label = value.get("label")
                cluster_id = value.get("cluster_id")
                scores = value.get("scores")
                score = scores.get(main_score) if isinstance(scores, Mapping) else None
                decision = value.get("decision")
                raw_source_units = value.get("source_unit_ids")
                source_unit_ids = (
                    tuple(raw_source_units)
                    if isinstance(raw_source_units, list)
                    else ()
                )
                if (
                    value.get("arm") != method
                    or value.get("main_score") != main_score
                    or not isinstance(pair_id, str)
                    or not pair_id
                    or type(label) is not bool  # noqa: E721
                    or not isinstance(cluster_id, str)
                    or not cluster_id
                    or not isinstance(score, Mapping)
                    or type(score.get("valid")) is not bool  # noqa: E721
                    or not isinstance(decision, Mapping)
                    or not 1 <= len(source_unit_ids) <= 2
                    or len(set(source_unit_ids)) != len(source_unit_ids)
                    or any(
                        not isinstance(item, str) or not item
                        for item in source_unit_ids
                    )
                ):
                    raise RachelN512CorrosionError(
                        "malformed {} sealed score line {}".format(method, line_number)
                    )
                if pair_id in seen:
                    raise RachelN512CorrosionError(
                        method + " sealed pair_id is duplicated"
                    )
                seen.add(pair_id)
                probability_value = score.get("probability")
                if isinstance(probability_value, bool) or not isinstance(
                    probability_value, (int, float)
                ):
                    raise RachelN512CorrosionError(
                        method + " sealed probability is non-numeric"
                    )
                probability = float(probability_value)
                valid = bool(score["valid"])
                if valid and (
                    not math.isfinite(probability) or not 0.0 <= probability <= 1.0
                ):
                    raise RachelN512CorrosionError(
                        method + " sealed valid probability is invalid"
                    )
                translation = None
                translation_valid = False
                if method == "full_n512" and valid:
                    geometry = value.get("geometry")
                    translation_value = (
                        geometry.get("translation_hat_rc")
                        if isinstance(geometry, Mapping)
                        else None
                    )
                    if (
                        not isinstance(translation_value, list)
                        or len(translation_value) != 2
                        or any(
                            isinstance(item, bool)
                            or not isinstance(item, (int, float))
                            or not math.isfinite(float(item))
                            for item in translation_value
                        )
                    ):
                        raise RachelN512CorrosionError(
                            "full_n512 sealed translation output is invalid"
                        )
                    translation = tuple(float(item) for item in translation_value)
                    translation_valid = True
                elif method in BENCHMARK_METHODS:
                    geometry = value.get("geometry")
                    if not isinstance(geometry, Mapping) or type(
                        geometry.get("translation_prediction_valid")
                    ) is not bool:  # noqa: E721
                        raise RachelN512CorrosionError(
                            method + " sealed translation validity is invalid"
                        )
                    translation_valid = bool(
                        geometry["translation_prediction_valid"]
                    )
                    translation_value = geometry.get("translation_hat_rc")
                    if translation_valid:
                        if (
                            not isinstance(translation_value, list)
                            or len(translation_value) != 2
                            or any(
                                isinstance(item, bool)
                                or not isinstance(item, (int, float))
                                or not math.isfinite(float(item))
                                for item in translation_value
                            )
                        ):
                            raise RachelN512CorrosionError(
                                method + " sealed translation output is invalid"
                            )
                        translation = tuple(
                            float(item) for item in translation_value
                        )
                    elif translation_value is not None:
                        raise RachelN512CorrosionError(
                            method + " invalid sealed translation must be null"
                        )
                rows.append(
                    _ExpectedCleanPrediction(
                        pair_id=pair_id,
                        label=label,
                        cluster_id=cluster_id,
                        probability=probability,
                        valid=valid,
                        translation_hat_rc=translation,
                        translation_valid=translation_valid,
                        source_unit_ids=source_unit_ids,
                    )
                )
    except json.JSONDecodeError as error:
        raise RachelN512CorrosionError(
            method + " sealed score JSONL is invalid"
        ) from error
    except OSError as error:
        raise RachelN512CorrosionError(
            method + " sealed score JSONL is unreadable"
        ) from error
    if len(rows) != expected_count:
        raise RachelN512CorrosionError(
            "{} sealed score count changed: {} != {}".format(
                method, len(rows), expected_count
            )
        )
    return tuple(rows)


def _load_sealed_reference_after_freeze(
    sealed_test_directory: Path,
    *,
    n512_receipt_sha256: str,
    n512_winners: Mapping[str, object],
    matched_receipt_sha256: str,
    matched_winners: Mapping[str, FrozenMatchedMMWinner],
    benchmark_winners: Mapping[str, benchmark_adapter.FrozenSameDataBenchmark],
    formal_evaluation: bool,
    methods: Sequence[str],
) -> _SealedReference:
    """Open the combined sealed result only after every winner is frozen."""

    try:
        root = Path(sealed_test_directory).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RachelN512CorrosionError(
            "combined sealed-test directory is missing"
        ) from error
    if not root.is_dir() or root.name.startswith(".partial-"):
        raise RachelN512CorrosionError(
            "combined sealed-test directory is not finalized"
        )
    receipt_path = root / "test_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RachelN512CorrosionError(
            "combined sealed-test receipt must be a regular file"
        )
    method_order = tuple(methods)
    expected_order = METHODS if formal_evaluation else LEGACY_METHODS
    if method_order != expected_order:
        raise RachelN512CorrosionError("clean-reference method inventory differs")
    receipt = _read_json_object(receipt_path, "combined sealed-test receipt")
    if (
        receipt.get("schema_version") != sealed.SCHEMA_VERSION
        or receipt.get("status")
        not in {
            sealed.FORMAL_COMPLETE_STATUS,
            sealed.COMPATIBILITY_COMPLETE_STATUS,
        }
        or receipt.get("test_accessed") is not True
        or receipt.get("real_external_test_accessed") is not False
    ):
        raise RachelN512CorrosionError("combined sealed-test receipt is incomplete")
    source = receipt.get("source_training_run")
    matched_source = receipt.get("source_matched_mm_training_run")
    benchmark_source = receipt.get("source_same_data_benchmark_training_runs")
    if (
        not isinstance(source, Mapping)
        or source.get("receipt_sha256") != n512_receipt_sha256
        or source.get("winners_frozen_before_test_open") is not True
        or not isinstance(matched_source, Mapping)
        or matched_source.get("receipt_sha256") != matched_receipt_sha256
        or matched_source.get(
            "converged_and_epoch5_winners_frozen_before_test_open"
        )
        is not True
    ):
        raise RachelN512CorrosionError(
            "combined sealed result is bound to different training runs"
        )
    protocol = receipt.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("threshold_fit_performed") is not False
        or protocol.get("checkpoint_selection_performed") is not False
        or protocol.get("formal_config_verified") is not True
        or protocol.get("both_arms_validation_plateau_verified") is not True
        or protocol.get(
            "all_requested_winners_and_thresholds_frozen_before_current_test_open"
        )
        is not True
    ):
        raise RachelN512CorrosionError(
            "combined sealed result lacks the frozen-inference protocol"
        )
    if formal_evaluation:
        benchmark_methods = (
            benchmark_source.get("methods")
            if isinstance(benchmark_source, Mapping)
            else None
        )
        if (
            receipt.get("formal_evaluation") is not True
            or receipt.get("compatibility_mode") is not False
            or receipt.get("status") != sealed.FORMAL_COMPLETE_STATUS
            or protocol.get("formal_evaluation") is not True
            or protocol.get("compatibility_mode") is not False
            or protocol.get("formal_method_inventory") != list(METHODS)
            or protocol.get("formal_exact_six_frozen_before_test_open") is not True
            or protocol.get(
                "same_data_benchmark_winners_and_thresholds_frozen_before_test_open"
            )
            is not True
            or not isinstance(benchmark_source, Mapping)
            or benchmark_source.get("all_methods_frozen_before_test_open") is not True
            or not isinstance(benchmark_methods, Mapping)
            or tuple(benchmark_methods) != BENCHMARK_METHODS
        ):
            raise RachelN512CorrosionError(
                "formal corrosion clean reference is not exact-six frozen authority"
            )
        for method in BENCHMARK_METHODS:
            provenance = benchmark_methods.get(method)
            winner = benchmark_winners.get(method)
            if (
                not isinstance(provenance, Mapping)
                or winner is None
                or provenance.get("freeze_authority_sha256")
                != winner.freeze_authority_sha256
                or provenance.get("validation_threshold_sha256")
                != winner.threshold_artifact_sha256
                or provenance.get("checkpoint_sha256_by_stage")
                != dict(winner.checkpoint_sha256_by_stage)
                or provenance.get("all_winners_and_validation_threshold_frozen")
                is not True
            ):
                raise RachelN512CorrosionError(
                    method + " clean-reference frozen authority differs"
                )
    elif (
        receipt.get("formal_evaluation") is not False
        or receipt.get("compatibility_mode") is not True
        or protocol.get("formal_evaluation") is not False
        or protocol.get("compatibility_mode") is not True
    ):
        raise RachelN512CorrosionError(
            "legacy clean reference must be explicitly non-formal compatibility"
        )
    precision = protocol.get("precision")
    batch_size = protocol.get("batch_size")
    if precision not in {"fp32", "bf16"} or type(batch_size) is not int or batch_size <= 0:  # noqa: E721
        raise RachelN512CorrosionError("combined sealed inference settings are invalid")
    population = receipt.get("test_population")
    if (
        not isinstance(population, Mapping)
        or population.get("expected_count") != EXPECTED_TEST_PAIRS
        or population.get("observed_count") != EXPECTED_TEST_PAIRS
        or population.get("positive_count") != EXPECTED_TEST_PAIRS // 2
        or population.get("negative_count") != EXPECTED_TEST_PAIRS // 2
        or population.get("exact_one_to_one") is not True
    ):
        raise RachelN512CorrosionError(
            "combined sealed population is not the fixed balanced 3,000"
        )
    manifest_sha = population.get("manifest_sha256")
    pair_ids_sha = population.get("pair_ids_sha256")
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in (manifest_sha, pair_ids_sha)
    ):
        raise RachelN512CorrosionError("combined sealed population hashes are invalid")
    arm_results = receipt.get("arm_results")
    if not isinstance(arm_results, list):
        raise RachelN512CorrosionError("combined sealed arm results are missing")
    by_method = {
        str(row["arm"]): row
        for row in arm_results
        if isinstance(row, Mapping) and "arm" in row
    }
    if (
        len(arm_results) != len(method_order)
        or len(by_method) != len(method_order)
        or set(by_method) != set(method_order)
    ):
        raise RachelN512CorrosionError(
            "combined sealed result method inventory differs"
        )
    expected: Dict[str, Tuple[_ExpectedCleanPrediction, ...]] = {}
    for method in method_order:
        row = by_method[method]
        if method in N512_METHODS:
            winner = n512_winners[method]
            authority_matches = (
                row.get("winner_checkpoint_sha256") == winner.checkpoint_sha256
                and row.get("validation_threshold_sha256")
                == winner.threshold.content_sha256
            )
        elif method in MATCHED_METHODS:
            winner = matched_winners[method]
            authority_matches = (
                row.get("winner_checkpoint_sha256") == winner.checkpoint_sha256
                and row.get("validation_threshold_sha256")
                == winner.threshold.content_sha256
            )
        else:
            benchmark = benchmark_winners[method]
            authority_matches = (
                row.get("freeze_authority_sha256")
                == benchmark.freeze_authority_sha256
                and row.get("winner_checkpoint_sha256_by_stage")
                == dict(benchmark.checkpoint_sha256_by_stage)
                and row.get("validation_threshold_sha256")
                == benchmark.threshold_artifact_sha256
                and row.get("adaptation_claim")
                == "same_data_method_adaptation_not_exact_reproduction"
            )
        if not authority_matches or row.get("pair_scores_count") != EXPECTED_TEST_PAIRS:
            raise RachelN512CorrosionError(
                method + " sealed result differs from the frozen winner/threshold"
            )
        score_path = _safe_result_member(
            root, row.get("pair_scores"), method + " pair scores"
        )
        expected[method] = _parse_expected_pair_scores(
            score_path,
            method=method,
            expected_sha256=row.get("pair_scores_sha256"),
            expected_count=EXPECTED_TEST_PAIRS,
        )
    reference = expected[method_order[0]]
    reference_identity = tuple(
        (row.pair_id, row.label, row.cluster_id, row.source_unit_ids)
        for row in reference
    )
    for method in method_order[1:]:
        identity = tuple(
            (row.pair_id, row.label, row.cluster_id, row.source_unit_ids)
            for row in expected[method]
        )
        if identity != reference_identity:
            raise RachelN512CorrosionError(
                "combined sealed method populations or order differ"
            )
    if _canonical_sha256([row.pair_id for row in reference]) != pair_ids_sha:
        raise RachelN512CorrosionError(
            "combined sealed pair_ids_sha256 does not bind pair order"
        )
    return _SealedReference(
        directory=root,
        receipt_sha256=_sha256_file(receipt_path),
        manifest_sha256=str(manifest_sha),
        pair_ids_sha256=str(pair_ids_sha),
        precision=str(precision),
        batch_size=int(batch_size),
        expected=expected,
    )


def _clean_parity_gate(
    observed: Mapping[str, Sequence[CorrosionPrediction]],
    expected: Mapping[str, Sequence[_ExpectedCleanPrediction]],
    *,
    tolerance: float,
) -> Mapping[str, object]:
    """Require exact identity/validity and strict score/translation parity."""

    method_order = tuple(expected)
    if method_order not in {METHODS, LEGACY_METHODS} or tuple(observed) != method_order:
        raise RachelN512CorrosionError("clean parity method inventory differs")
    methods: Dict[str, object] = {}
    global_max = 0.0
    for method in method_order:
        actual = tuple(observed[method])
        reference = tuple(expected[method])
        if len(actual) != len(reference):
            raise RachelN512CorrosionError(method + " clean parity count differs")
        probability_max = 0.0
        translation_max = 0.0
        for prediction, target in zip(actual, reference):
            if (
                prediction.pair_id != target.pair_id
                or prediction.valid != target.valid
                or prediction.translation_valid != target.translation_valid
            ):
                raise RachelN512CorrosionError(
                    "clean inference did not reproduce "
                    + method
                    + " identity, decision validity, or translation validity"
                )
            if target.valid:
                difference = abs(prediction.probability - target.probability)
                probability_max = max(probability_max, difference)
            if method in TRANSLATION_METHODS and target.translation_valid:
                if (
                    prediction.translation_hat_rc is None
                    or target.translation_hat_rc is None
                ):
                    raise RachelN512CorrosionError(
                        method + " clean translation parity is incomplete"
                    )
                translation_max = max(
                    translation_max,
                    max(
                        abs(first - second)
                        for first, second in zip(
                            prediction.translation_hat_rc,
                            target.translation_hat_rc,
                        )
                    ),
                )
        maximum = max(probability_max, translation_max)
        global_max = max(global_max, maximum)
        methods[method] = {
            "pair_count": len(actual),
            "validity_exact_match": True,
            "maximum_probability_absolute_difference": probability_max,
            "maximum_translation_component_absolute_difference": (
                translation_max if method in TRANSLATION_METHODS else None
            ),
            "passed": maximum <= tolerance,
        }
    passed = global_max <= tolerance
    receipt = {
        "passed": passed,
        "tolerance": tolerance,
        "maximum_absolute_difference": global_max,
        "methods": methods,
        "reference": "completed_combined_sealed_synthetic_test_pair_scores",
    }
    if not passed:
        raise RachelN512CorrosionError(
            "clean inference did not reproduce combined sealed scores: "
            "max_abs_difference={:.9g}".format(global_max)
        )
    return receipt


def _coverage(valid: Sequence[bool], labels: Sequence[bool]) -> Mapping[str, object]:
    selected = np.asarray(valid, dtype=np.bool_)
    target = np.asarray(labels, dtype=np.bool_)
    if selected.ndim != 1 or target.shape != selected.shape or not len(selected):
        raise ValueError("coverage vectors are empty or misaligned")
    return {
        "record_count": int(len(selected)),
        "valid_count": int(selected.sum()),
        "valid_fraction": float(selected.mean()),
        "positive_count": int(target.sum()),
        "positive_valid_count": int((selected & target).sum()),
        "positive_valid_fraction": float(selected[target].mean()),
        "negative_count": int((~target).sum()),
        "negative_valid_count": int((selected & ~target).sum()),
        "negative_valid_fraction": float(selected[~target].mean()),
    }


def _metric_view(
    predictions: Sequence[CorrosionPrediction],
    *,
    labels: Sequence[bool],
    clusters: Sequence[str],
    selected: Sequence[bool],
    threshold: float,
) -> Mapping[str, object]:
    values = tuple(predictions)
    if len(values) != len(labels) or len(values) != len(selected):
        raise ValueError("metric vectors are misaligned")
    valid = np.asarray(
        [
            bool(keep)
            and row.valid
            and math.isfinite(row.probability)
            for keep, row in zip(selected, values)
        ],
        dtype=np.bool_,
    )
    coverage = _coverage(valid, labels)
    selected_labels = np.asarray(labels, dtype=np.bool_)[valid]
    if not len(selected_labels) or len(np.unique(selected_labels)) != 2:
        return {
            "coverage": coverage,
            "primary_threshold_free_ranking": None,
            "secondary_frozen_validation_threshold": None,
            "ranking_available": False,
        }
    report = evaluate_pairwise(
        [row.probability for row in values],
        labels,
        valid,
        clusters,
        threshold=threshold,
    )
    ranking = {
        "sample_count": report["sample_count"],
        "positive_count": report["positive_count"],
        "negative_count": report["negative_count"],
        "cluster_count": report["cluster_count"],
        "row": {
            "auroc": report["row"]["auroc"],
            "auprc": report["row"]["auprc"],
        },
        "cluster_balanced": {
            "auroc": report["cluster_balanced"]["auroc"],
            "auprc": report["cluster_balanced"]["auprc"],
        },
    }
    secondary: Dict[str, object] = {
        "threshold": threshold,
        "threshold_source": "frozen_validation_artifact",
        "fit_performed_during_robustness_evaluation": False,
    }
    for level in ("row", "cluster_balanced"):
        secondary[level] = {
            key: report[level][key]
            for key in (
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "false_positive_rate",
                "f1",
                "brier",
                "ece",
            )
        }
    return {
        "coverage": coverage,
        "primary_threshold_free_ranking": ranking,
        "secondary_frozen_validation_threshold": secondary,
        "ranking_available": True,
    }


def _drop_value(clean: float, observed: float) -> Mapping[str, float]:
    absolute = float(clean - observed)
    relative = float(absolute / abs(clean)) if clean != 0.0 else 0.0
    return {
        "absolute_drop_positive_means_worse": absolute,
        "relative_drop_fraction_positive_means_worse": relative,
    }


def _relative_clean_drop(
    clean: Mapping[str, object], current: Mapping[str, object]
) -> Optional[Mapping[str, object]]:
    clean_ranking = clean.get("primary_threshold_free_ranking")
    current_ranking = current.get("primary_threshold_free_ranking")
    if not isinstance(clean_ranking, Mapping) or not isinstance(
        current_ranking, Mapping
    ):
        return None
    output: Dict[str, object] = {}
    for level in ("row", "cluster_balanced"):
        clean_level = clean_ranking[level]
        current_level = current_ranking[level]
        assert isinstance(clean_level, Mapping)
        assert isinstance(current_level, Mapping)
        output[level] = {
            metric: _drop_value(
                float(clean_level[metric]), float(current_level[metric])
            )
            for metric in ("auroc", "auprc")
        }
    clean_coverage = clean["coverage"]
    current_coverage = current["coverage"]
    assert isinstance(clean_coverage, Mapping)
    assert isinstance(current_coverage, Mapping)
    output["coverage"] = _drop_value(
        float(clean_coverage["valid_fraction"]),
        float(current_coverage["valid_fraction"]),
    )
    return output


def _translation_stability(
    clean: Sequence[CorrosionPrediction],
    current: Sequence[CorrosionPrediction],
    *,
    labels: Sequence[bool],
    selected: Sequence[bool],
) -> Mapping[str, object]:
    distances = []
    for keep, label, clean_row, current_row in zip(
        selected, labels, clean, current
    ):
        if (
            not keep
            or not label
            or not clean_row.valid
            or not current_row.valid
            or clean_row.translation_hat_rc is None
            or current_row.translation_hat_rc is None
        ):
            continue
        distances.append(
            math.hypot(
                current_row.translation_hat_rc[0] - clean_row.translation_hat_rc[0],
                current_row.translation_hat_rc[1] - clean_row.translation_hat_rc[1],
            )
        )
    value = np.asarray(distances, dtype=np.float64)
    return {
        "interpretation": (
            "secondary positive-only prediction stability relative to the clean "
            "full_n512 translation; negatives are excluded and this is not GT "
            "translation accuracy"
        ),
        "secondary_analysis": True,
        "positive_pairs_only_negative_pairs_excluded": True,
        "selected_pair_count": int(len(value)),
        "median_l2_change_px": float(np.median(value)) if len(value) else None,
        "p90_l2_change_px": float(np.quantile(value, 0.9)) if len(value) else None,
        "mean_l2_change_px": float(value.mean()) if len(value) else None,
        **{
            "fraction_within_{}px".format(tolerance): (
                float(np.mean(value <= tolerance)) if len(value) else None
            )
            for tolerance in (2, 5, 8, 10)
        },
    }


def summarize_corrosion_predictions(
    predictions_by_condition: Mapping[
        str, Mapping[str, Sequence[CorrosionPrediction]]
    ],
    *,
    labels: Sequence[bool],
    clusters: Sequence[str],
    thresholds: Mapping[str, float],
    pair_ids: Sequence[str],
) -> Mapping[str, object]:
    """Build fixed-common primary ranking and clean-relative robustness curves."""

    condition_names = tuple(condition.name for condition in FIXED_CONDITIONS)
    if tuple(predictions_by_condition) != condition_names:
        raise ValueError("conditions must be supplied in fixed preregistered order")
    count = len(pair_ids)
    method_order = tuple(thresholds)
    if (
        not count
        or len(labels) != count
        or len(clusters) != count
        or method_order not in {METHODS, LEGACY_METHODS}
    ):
        raise ValueError("summary side-table vectors are incomplete")
    for condition in condition_names:
        if tuple(predictions_by_condition[condition]) != method_order:
            raise ValueError("one condition lacks a method")
        for method in method_order:
            rows = tuple(predictions_by_condition[condition][method])
            if len(rows) != count or tuple(row.pair_id for row in rows) != tuple(
                pair_ids
            ):
                raise ValueError("prediction population/order changed")
    fixed_common = tuple(
        all(
            predictions_by_condition[condition][method][index].valid
            and math.isfinite(
                predictions_by_condition[condition][method][index].probability
            )
            for condition in condition_names
            for method in method_order
        )
        for index in range(count)
    )
    native_views: Dict[str, Dict[str, Mapping[str, object]]] = {}
    fixed_views: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for condition in condition_names:
        native_views[condition] = {}
        fixed_views[condition] = {}
        for method in method_order:
            values = predictions_by_condition[condition][method]
            native_views[condition][method] = _metric_view(
                values,
                labels=labels,
                clusters=clusters,
                selected=[True] * count,
                threshold=float(thresholds[method]),
            )
            fixed_views[condition][method] = _metric_view(
                values,
                labels=labels,
                clusters=clusters,
                selected=fixed_common,
                threshold=float(thresholds[method]),
            )
    conditions: Dict[str, object] = {}
    for condition in FIXED_CONDITIONS:
        methods = {}
        for method in method_order:
            methods[method] = {
                "native_valid": native_views[condition.name][method],
                "fixed_all_method_all_condition_common_valid": fixed_views[
                    condition.name
                ][method],
                "relative_to_clean": {
                    "native_valid": _relative_clean_drop(
                        native_views["clean"][method],
                        native_views[condition.name][method],
                    ),
                    "fixed_all_method_all_condition_common_valid": (
                        _relative_clean_drop(
                            fixed_views["clean"][method],
                            fixed_views[condition.name][method],
                        )
                    ),
                },
            }
        translation_stability = {
            method: _translation_stability(
                predictions_by_condition["clean"][method],
                predictions_by_condition[condition.name][method],
                labels=labels,
                selected=fixed_common,
            )
            for method in TRANSLATION_METHODS
            if method in method_order
        }
        conditions[condition.name] = {
            "condition": condition.to_dict(),
            "methods": methods,
            "translation_stability_by_method": translation_stability,
            "full_n512_translation_stability": translation_stability["full_n512"],
        }
    selected_ids = [pair_id for pair_id, keep in zip(pair_ids, fixed_common) if keep]
    return {
        "primary_population": {
            "definition": (
                "all_six_methods_valid_at_all_seven_conditions"
                if method_order == METHODS
                else "legacy_all_four_methods_valid_at_all_seven_conditions_nonformal"
            ),
            "pair_count": sum(fixed_common),
            "coverage": sum(fixed_common) / count,
            "positive_count": sum(
                label for label, keep in zip(labels, fixed_common) if keep
            ),
            "negative_count": sum(
                not label for label, keep in zip(labels, fixed_common) if keep
            ),
            "pair_ids_sha256": _canonical_sha256(selected_ids),
        },
        "condition_order": list(condition_names),
        "method_order": list(method_order),
        "formal_exact_six": method_order == METHODS,
        "conditions": conditions,
        "geometry_interpretation": {
            "correspondence_accuracy_reported": False,
            "reason": (
                "corrupted contours are independently resampled, so clean/GT "
                "correspondence token indices are not reusable"
            ),
            "gt_translation_accuracy_reported": False,
            "translation_reported": "prediction_stability_relative_to_clean_only",
            "benchmark_correspondence": {
                method: {
                    "status": "not_applicable",
                    "reason": (
                        "corruption reextracts contour indices and no current-condition "
                        "correspondence ground truth exists"
                    ),
                }
                for method in BENCHMARK_METHODS
                if method in method_order
            },
        },
    }


def _pair_inputs_from_manifest(
    dataset_root: Path, rows: Sequence[object]
) -> Tuple[BlindSyntheticPair, ...]:
    pairs = []
    for row in rows:
        if row.mask_a_path is None or row.mask_b_path is None:
            raise RachelN512CorrosionError(
                "test manifest did not freeze both model-mask endpoints"
            )
        try:
            first_id = row.mask_a_path.relative_to(dataset_root).as_posix()
            second_id = row.mask_b_path.relative_to(dataset_root).as_posix()
        except ValueError as error:
            raise RachelN512CorrosionError(
                "test model mask escapes the Rachel release"
            ) from error
        if first_id == second_id:
            raise RachelN512CorrosionError("test pair endpoints must differ")
        pairs.append(
            BlindSyntheticPair(
                pair_id=row.pair_id,
                fragment_a_id=first_id,
                fragment_b_id=second_id,
                mask_a_path=row.mask_a_path,
                mask_b_path=row.mask_b_path,
            )
        )
    if len({row.pair_id for row in pairs}) != len(pairs):
        raise RachelN512CorrosionError("blind test pair IDs are duplicated")
    return tuple(pairs)


def _safe_positive_target_path(dataset_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RachelN512CorrosionError("positive target path is invalid")
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.parts[:2] != ("targets", "pairs")
        or logical.suffix.casefold() != ".npz"
    ):
        raise RachelN512CorrosionError(
            "positive target must be a targets/pairs NPZ release member"
        )
    current = dataset_root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise RachelN512CorrosionError("symlinked positive target is forbidden")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(dataset_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise RachelN512CorrosionError(
            "positive target is missing or escapes the release"
        ) from error
    if not resolved.is_file():
        raise RachelN512CorrosionError("positive target is not a regular file")
    return resolved


def _load_positive_translation_targets_after_predictions(
    dataset_root: Path,
    test_manifest_path: Path,
    *,
    pair_ids: Sequence[str],
    labels: Sequence[bool],
    expected_manifest_sha256: Optional[str] = None,
) -> Tuple[Tuple[Optional[Tuple[float, float]], ...], Mapping[str, object]]:
    """Open raw positive target NPZs only after every model forward completed."""

    manifest_sha_before = _sha256_file(test_manifest_path)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha_before != expected_manifest_sha256
    ):
        raise RachelN512CorrosionError(
            "test manifest changed before post-prediction target opening"
        )
    rows = []
    target_authority = []
    try:
        with Path(test_manifest_path).open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise RachelN512CorrosionError(
                            "test target manifest line is not an object"
                        )
                    rows.append(value)
    except json.JSONDecodeError as error:
        raise RachelN512CorrosionError(
            "test target manifest contains invalid JSON"
        ) from error
    except OSError as error:
        raise RachelN512CorrosionError("test target manifest is unreadable") from error
    if len(rows) != len(pair_ids):
        raise RachelN512CorrosionError("test target manifest count changed")
    if _sha256_file(test_manifest_path) != manifest_sha_before:
        raise RachelN512CorrosionError(
            "test manifest changed during post-prediction target opening"
        )
    translations = []
    for index, (row, pair_id, label) in enumerate(zip(rows, pair_ids, labels)):
        if (
            row.get("pair_id") != pair_id
            or row.get("split") != "test"
            or type(row.get("label")) is not bool  # noqa: E721
            or row.get("label") != label
        ):
            raise RachelN512CorrosionError(
                "test target manifest identity changed at row {}".format(index)
            )
        target_value = row.get("correspondence_path")
        if not label:
            if target_value is not None:
                raise RachelN512CorrosionError(
                    "negative test pair unexpectedly references a target archive"
                )
            translations.append(None)
            continue
        target_path = _safe_positive_target_path(dataset_root, target_value)
        target_sha_before = _sha256_file(target_path)
        try:
            with np.load(target_path, allow_pickle=False) as archive:
                if set(archive.files) != {
                    "correspondence_indices",
                    "translation_a_to_b_rc",
                    "translation_a_to_b_xy_cartesian",
                }:
                    raise RachelN512CorrosionError(
                        "positive target archive fields differ"
                    )
                translation_rc = np.asarray(
                    archive["translation_a_to_b_rc"], dtype=np.float64
                )
                translation_xy = np.asarray(
                    archive["translation_a_to_b_xy_cartesian"], dtype=np.float64
                )
        except RachelN512CorrosionError:
            raise
        except (OSError, ValueError) as error:
            raise RachelN512CorrosionError(
                "positive translation target cannot be decoded"
            ) from error
        if (
            translation_rc.shape != (2,)
            or translation_xy.shape != (2,)
            or not np.all(np.isfinite(translation_rc))
            or not np.all(np.isfinite(translation_xy))
            or not np.allclose(
                translation_xy,
                (translation_rc[1], -translation_rc[0]),
                rtol=0.0,
                atol=1e-4,
            )
        ):
            raise RachelN512CorrosionError(
                "positive RC/Cartesian translation target is invalid"
            )
        translations.append((float(translation_rc[0]), float(translation_rc[1])))
        target_authority.append(
            {
                "pair_id": pair_id,
                "target_path": Path(target_value).as_posix(),
                "target_archive_sha256": target_sha_before,
            }
        )
        if _sha256_file(target_path) != target_sha_before:
            raise RachelN512CorrosionError(
                "positive target archive changed while translation was read"
            )
    if len(target_authority) != sum(labels):
        raise RachelN512CorrosionError(
            "positive target archive count differs from positive labels"
        )
    return tuple(translations), {
        "raw_positive_target_archive_count": len(target_authority),
        "negative_target_archive_count": 0,
        "target_authority_sha256": _canonical_sha256(target_authority),
        "opened_after_all_target_blind_predictions": True,
        "used_for_model_forward_checkpoint_selection_or_threshold_fit": False,
        "coordinate_frame": (
            "original_centered_800_model_frame_preserved_by_mask_only_corruption"
        ),
    }


def _positive_translation_metric_view(
    predictions: Sequence[CorrosionPrediction],
    *,
    labels: Sequence[bool],
    targets: Sequence[Optional[Tuple[float, float]]],
    selected: Sequence[bool],
    threshold: float,
) -> Mapping[str, object]:
    if not (
        len(predictions) == len(labels) == len(targets) == len(selected)
    ):
        raise ValueError("positive translation metric vectors differ")
    eligible_count = 0
    valid_count = 0
    errors = []
    threshold_positive = []
    for prediction, label, target, keep in zip(
        predictions, labels, targets, selected
    ):
        if not label:
            if target is not None:
                raise ValueError("negative pair must not carry translation GT")
            continue
        if target is None:
            raise ValueError("positive pair lacks translation GT")
        if not keep:
            continue
        eligible_count += 1
        if not prediction.valid or prediction.translation_hat_rc is None:
            continue
        error = math.hypot(
            prediction.translation_hat_rc[0] - target[0],
            prediction.translation_hat_rc[1] - target[1],
        )
        if not math.isfinite(error):
            raise RachelN512CorrosionError("valid translation error is non-finite")
        valid_count += 1
        errors.append(error)
        threshold_positive.append(prediction.probability >= threshold)
    value = np.asarray(errors, dtype=np.float64)
    accepted = np.asarray(threshold_positive, dtype=np.bool_)
    output: Dict[str, object] = {
        "scope": "positive_pairs_only_negative_pairs_excluded",
        "coordinate_frame": (
            "original_centered_800_model_frame_preserved_by_corruption"
        ),
        "eligible_positive_count": eligible_count,
        "valid_translation_prediction_count": valid_count,
        "valid_translation_prediction_fraction": (
            valid_count / eligible_count if eligible_count else 0.0
        ),
        "median_l2_px": float(np.median(value)) if len(value) else None,
        "p90_l2_px": float(np.quantile(value, 0.9)) if len(value) else None,
        "frozen_validation_threshold": threshold,
        "threshold_fit_performed": False,
    }
    for tolerance in (2, 5, 8, 10):
        success = value <= tolerance
        output["recall_at_{}px".format(tolerance)] = (
            float(success.sum() / eligible_count) if eligible_count else None
        )
        output[
            "joint_frozen_threshold_and_translation_recall_at_{}px".format(
                tolerance
            )
        ] = (
            float((success & accepted).sum() / eligible_count)
            if eligible_count
            else None
        )
    return output


def _prf(true_positive: int, predicted: int, target: int) -> Mapping[str, object]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / target if target else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive_count": int(true_positive),
        "predicted_count": int(predicted),
        "target_count": int(target),
        "false_positive_count": int(predicted - true_positive),
        "false_negative_count": int(target - true_positive),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _direct_geometry_metric_view(
    predictions: Sequence[CorrosionPrediction],
    *,
    labels: Sequence[bool],
    targets: Sequence[Optional[Tuple[float, float]]],
    selected: Sequence[bool],
    threshold: float,
    contour_area_sums: Sequence[float],
) -> Mapping[str, object]:
    """Direct TE/assembly and Pairing-compatible registration on positives."""

    if not (
        len(predictions)
        == len(labels)
        == len(targets)
        == len(selected)
        == len(contour_area_sums)
    ):
        raise ValueError("direct geometry metric vectors differ")
    eligible = 0
    valid_pose = 0
    errors = []
    e_rmse = []
    hausdorff = []
    nte = []
    rr_success = 0
    predicted_edges = 0
    true_positive_by_tolerance = {value: 0 for value in (2, 5, 8, 10)}
    for prediction, label, target, keep, area_sum in zip(
        predictions, labels, targets, selected, contour_area_sums
    ):
        if not keep:
            continue
        predicted_edge = (
            prediction.valid
            and math.isfinite(prediction.probability)
            and prediction.probability >= threshold
        )
        predicted_edges += int(predicted_edge)
        if not label:
            if target is not None:
                raise ValueError("negative pair must not carry translation GT")
            continue
        if target is None:
            raise ValueError("positive pair lacks translation GT")
        if not math.isfinite(float(area_sum)) or float(area_sum) <= 0.0:
            raise RachelN512CorrosionError(
                "positive PairingNet-compatible contour area sum is invalid"
            )
        eligible += 1
        pose_valid = (
            prediction.translation_valid
            and prediction.translation_hat_rc is not None
        )
        effective = prediction.translation_hat_rc if pose_valid else (0.0, 0.0)
        error = math.hypot(effective[0] - target[0], effective[1] - target[1])
        if not math.isfinite(error):
            raise RachelN512CorrosionError("direct translation error is non-finite")
        registration_e_rmse = math.sqrt(error)
        valid_pose += int(pose_valid)
        if pose_valid:
            errors.append(error)
        e_rmse.append(registration_e_rmse)
        hausdorff.append(error)
        nte.append(error / float(area_sum))
        rr_success += int(registration_e_rmse < 4.0)
        for tolerance in true_positive_by_tolerance:
            true_positive_by_tolerance[tolerance] += int(
                predicted_edge and pose_valid and error <= tolerance
            )
    conditioned = np.asarray(errors, dtype=np.float64)
    translation = {
        "scope": "positive_pairs_only_invalid_pose_counts_as_recall_failure",
        "eligible_positive_count": eligible,
        "valid_translation_prediction_count": valid_pose,
        "valid_translation_prediction_fraction": (
            valid_pose / eligible if eligible else 0.0
        ),
        "te_px_conditioned_on_valid_pose": {
            "count": len(errors),
            "median": float(np.median(conditioned)) if len(conditioned) else None,
            "p90": (
                float(np.quantile(conditioned, 0.9)) if len(conditioned) else None
            ),
        },
        "unconditional_positive_recall": {
            "at_{}px".format(tolerance): (
                sum(value <= tolerance for value in errors) / eligible
                if eligible
                else None
            )
            for tolerance in (2, 5, 8, 10)
        },
    }
    return {
        "translation": translation,
        "assembly_edge": {
            "definition": (
                "threshold-accepted edge is correct only for a positive pair with "
                "valid translation within tolerance; wrong or invalid pose is FP+FN"
            ),
            "frozen_validation_threshold": threshold,
            "predicted_edge_count": predicted_edges,
            "target_edge_count": eligible,
            "by_tolerance": {
                "at_{}px".format(tolerance): _prf(
                    true_positive_by_tolerance[tolerance],
                    predicted_edges,
                    eligible,
                )
                for tolerance in (2, 5, 8, 10)
            },
        },
        "pairingnet_style_registration": {
            "status": "reported",
            "compatibility_source": (
                "PairingNet released matching_test.py upright translation-only "
                "specialization"
            ),
            "derivation_under_translation_only_corrosion": (
                "e_rmse=sqrt(translation_l2_px), symmetric_HD=translation_l2_px; "
                "NTE uses PairingNet int32-quantized current corrupted contour areas"
            ),
            "eligible_positive_count": eligible,
            "valid_pose_count": valid_pose,
            "identity_fallback_count": eligible - valid_pose,
            "rr_lt4": rr_success / eligible if eligible else None,
            "mean_e_rmse": float(np.mean(e_rmse)) if e_rmse else None,
            "mean_symmetric_hausdorff_px": (
                float(np.mean(hausdorff)) if hausdorff else None
            ),
            "mean_normalized_translation_error": (
                float(np.mean(nte)) if nte else None
            ),
            "rotation_error": {
                "status": "not_applicable",
                "reason": "upright_orientation_is_conditioned",
            },
        },
        "correspondence": {
            "status": "not_applicable",
            "reason": (
                "current-condition corrupted contours have new token indices and no "
                "current-condition correspondence ground truth"
            ),
            "metrics": None,
        },
    }


def _attach_positive_translation_metrics(
    summary: Dict[str, object],
    predictions_by_condition: Mapping[
        str, Mapping[str, Sequence[CorrosionPrediction]]
    ],
    *,
    labels: Sequence[bool],
    targets: Sequence[Optional[Tuple[float, float]]],
    threshold: float,
    fixed_common: Sequence[bool],
    thresholds: Optional[Mapping[str, float]] = None,
    contour_area_sums_by_condition: Optional[
        Mapping[str, Sequence[float]]
    ] = None,
) -> None:
    conditions = summary.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("corrosion summary conditions are missing")
    interpretation = summary.get("geometry_interpretation")
    if not isinstance(interpretation, dict):
        raise ValueError("corrosion geometry interpretation is missing")
    interpretation.update(
        {
            "gt_translation_accuracy_reported": True,
            "gt_translation_scope": (
                "positive-only original centered-800 targets opened after all predictions"
            ),
            "translation_reported": (
                "positive-only_GT_accuracy_plus_secondary_positive-only_clean_stability"
            ),
        }
    )
    for condition in FIXED_CONDITIONS:
        row = conditions[condition.name]
        if not isinstance(row, dict):
            raise ValueError("corrosion condition summary is malformed")
        values = predictions_by_condition[condition.name]["full_n512"]
        row["full_n512_gt_translation_positive_only"] = {
            "native_positive_population": _positive_translation_metric_view(
                values,
                labels=labels,
                targets=targets,
                selected=[True] * len(labels),
                threshold=threshold,
            ),
            "fixed_all_method_all_condition_common_valid_positive_population": (
                _positive_translation_metric_view(
                    values,
                    labels=labels,
                    targets=targets,
                    selected=fixed_common,
                    threshold=threshold,
                )
            ),
        }
        if thresholds is not None and contour_area_sums_by_condition is not None:
            direct_by_method = {}
            for method in TRANSLATION_METHODS:
                if method not in predictions_by_condition[condition.name]:
                    continue
                direct_by_method[method] = {
                    "native_positive_population": _direct_geometry_metric_view(
                        predictions_by_condition[condition.name][method],
                        labels=labels,
                        targets=targets,
                        selected=[True] * len(labels),
                        threshold=float(thresholds[method]),
                        contour_area_sums=contour_area_sums_by_condition[
                            condition.name
                        ],
                    ),
                    "fixed_all_method_all_condition_common_valid_positive_population": (
                        _direct_geometry_metric_view(
                            predictions_by_condition[condition.name][method],
                            labels=labels,
                            targets=targets,
                            selected=fixed_common,
                            threshold=float(thresholds[method]),
                            contour_area_sums=contour_area_sums_by_condition[
                                condition.name
                            ],
                        )
                    ),
                }
            row["direct_geometry_by_method"] = direct_by_method


@dataclass(frozen=True)
class _BootstrapRankingPlan:
    labels: np.ndarray
    order: np.ndarray
    starts: np.ndarray

    @classmethod
    def build(
        cls, labels: np.ndarray, scores: np.ndarray
    ) -> "_BootstrapRankingPlan":
        order = np.argsort(-scores, kind="mergesort")
        ordered_scores = scores[order]
        starts = np.r_[
            0,
            np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1,
        ]
        return cls(labels=labels[order], order=order, starts=starts)

    def evaluate(self, weights: np.ndarray) -> Tuple[float, float]:
        ordered_weights = weights[self.order]
        positive_total = float(ordered_weights[self.labels].sum())
        negative_total = float(ordered_weights[~self.labels].sum())
        if positive_total <= 0.0 or negative_total <= 0.0:
            raise RachelN512CorrosionError(
                "bootstrap ranking draw must contain both classes"
            )
        positive = np.add.reduceat(ordered_weights * self.labels, self.starts)
        negative = np.add.reduceat(ordered_weights * ~self.labels, self.starts)
        true_positive = np.cumsum(positive)
        false_positive = np.cumsum(negative)
        tpr = true_positive / positive_total
        fpr = false_positive / negative_total
        previous_tpr = np.r_[0.0, tpr[:-1]]
        previous_fpr = np.r_[0.0, fpr[:-1]]
        auroc = np.sum((fpr - previous_fpr) * (tpr + previous_tpr) * 0.5)
        precision = np.divide(
            true_positive,
            true_positive + false_positive,
            out=np.zeros_like(true_positive),
            where=(true_positive + false_positive) > 0.0,
        )
        auprc = np.sum((tpr - previous_tpr) * precision)
        return float(auroc), float(auprc)


def _bootstrap_interval(
    values: Sequence[float], point: float, *, delta: bool = False
) -> Mapping[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise RachelN512CorrosionError(
            "no finite endpoint-bootstrap distribution is available"
        )
    lower, upper = np.quantile(array, (0.025, 0.975))
    output: Dict[str, object] = {
        "point_estimate": float(point),
        "percentile_95_ci": [float(lower), float(upper)],
        "bootstrap_mean": float(array.mean()),
        "bootstrap_standard_error": (
            float(array.std(ddof=1)) if array.size > 1 else 0.0
        ),
        "valid_replicates": int(array.size),
    }
    if delta:
        output["probability_delta_gt_zero"] = float(np.mean(array > 0.0))
    return output


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if (
        value_array.ndim != 1
        or weight_array.shape != value_array.shape
        or not value_array.size
        or not np.isfinite(value_array).all()
        or not np.isfinite(weight_array).all()
        or np.any(weight_array < 0.0)
        or not np.equal(weight_array, np.floor(weight_array)).all()
        or not 0.0 <= float(quantile) <= 1.0
    ):
        raise RachelN512CorrosionError(
            "weighted quantile requires finite values and non-negative integer weights"
        )
    integer_weights = weight_array.astype(np.int64)
    keep = integer_weights > 0
    if not np.any(keep):
        raise RachelN512CorrosionError("weighted quantile has no positive weight")
    selected_values = value_array[keep]
    selected_weights = integer_weights[keep]
    order = np.argsort(selected_values, kind="mergesort")
    ordered_values = selected_values[order]
    cumulative = np.cumsum(selected_weights[order], dtype=np.int64)
    expanded_count = int(cumulative[-1])
    # NumPy's default ``method="linear"`` uses the two zero-based ranks around
    # q * (n - 1).  Looking those ranks up in cumulative integer multiplicities
    # is exactly equivalent to ``np.quantile(np.repeat(values, weights), q)``
    # without allocating the expanded sample.
    position = float(quantile) * float(expanded_count - 1)
    lower_rank = int(math.floor(position))
    upper_rank = int(math.ceil(position))
    fraction = position - lower_rank
    lower_index = int(np.searchsorted(cumulative, lower_rank, side="right"))
    upper_index = int(np.searchsorted(cumulative, upper_rank, side="right"))
    lower = float(ordered_values[lower_index])
    upper = float(ordered_values[upper_index])
    return lower + (upper - lower) * fraction


def _endpoint_dependency_index(
    source_unit_ids: Sequence[Sequence[str]], selected: np.ndarray
) -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    selected_units = [
        tuple(units) for units, keep in zip(source_unit_ids, selected) if keep
    ]
    if any(
        not 1 <= len(units) <= 2
        or len(set(units)) != len(units)
        or tuple(sorted(units)) != units
        or any(not unit for unit in units)
        for units in selected_units
    ):
        raise RachelN512CorrosionError("bootstrap source_unit_ids are invalid")
    units = tuple(sorted({unit for row in selected_units for unit in row}))
    if len(units) < 2:
        raise RachelN512CorrosionError(
            "endpoint bootstrap requires at least two source units"
        )
    lookup = {unit: index for index, unit in enumerate(units)}
    first = np.asarray([lookup[row[0]] for row in selected_units], dtype=np.int64)
    second = np.asarray(
        [lookup[row[1]] if len(row) == 2 else -1 for row in selected_units],
        dtype=np.int64,
    )
    return units, first, second


def _pigeonhole_pair_weights(
    unit_count: int,
    first: np.ndarray,
    second: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = rng.integers(0, unit_count, size=unit_count)
    multiplicity = np.bincount(sampled, minlength=unit_count).astype(np.float64)
    weights = multiplicity[first].copy()
    paired = second >= 0
    weights[paired] *= multiplicity[second[paired]]
    return weights


def _direct_geometry_statistics(
    pose_errors: np.ndarray,
    pose_valid: np.ndarray,
    fallback_errors: np.ndarray,
    accepted: np.ndarray,
    positive: np.ndarray,
    contour_area_sums: np.ndarray,
    weights: np.ndarray,
) -> Mapping[str, float]:
    positive_weight = float(weights[positive].sum())
    if positive_weight <= 0.0:
        raise RachelN512CorrosionError("direct-geometry draw has no positive weight")
    valid_positive = positive & pose_valid
    valid_pose_weight = float(weights[valid_positive].sum())
    if valid_pose_weight <= 0.0:
        raise RachelN512CorrosionError(
            "direct-geometry draw has no valid positive pose"
        )
    output = {
        "median_l2_px": _weighted_quantile(
            pose_errors[valid_positive], weights[valid_positive], 0.5
        ),
        "p90_l2_px": _weighted_quantile(
            pose_errors[valid_positive], weights[valid_positive], 0.9
        ),
        "valid_pose_fraction": valid_pose_weight / positive_weight,
    }
    for tolerance in (2, 5, 8, 10):
        successful = valid_positive & (pose_errors <= tolerance)
        output["recall_at_{}px".format(tolerance)] = float(
            weights[successful].sum() / positive_weight
        )
        true_positive_weight = float(weights[successful & accepted].sum())
        predicted_weight = float(weights[accepted].sum())
        precision = true_positive_weight / predicted_weight if predicted_weight else 0.0
        recall = true_positive_weight / positive_weight
        output["assembly_precision_at_{}px".format(tolerance)] = precision
        output["assembly_recall_at_{}px".format(tolerance)] = recall
        output[
            "joint_frozen_threshold_and_translation_recall_at_{}px".format(
                tolerance
            )
        ] = recall
        output["assembly_f1_at_{}px".format(tolerance)] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    registration_errors = fallback_errors[positive]
    registration_weights = weights[positive]
    area = contour_area_sums[positive]
    if (
        not np.isfinite(registration_errors).all()
        or not np.isfinite(area).all()
        or np.any(area <= 0.0)
    ):
        raise RachelN512CorrosionError(
            "PairingNet-compatible bootstrap geometry is invalid"
        )
    output.update(
        {
            "pairing_rr_lt4": float(
                registration_weights[np.sqrt(registration_errors) < 4.0].sum()
                / positive_weight
            ),
            "pairing_mean_e_rmse": float(
                np.sum(np.sqrt(registration_errors) * registration_weights)
                / positive_weight
            ),
            "pairing_mean_symmetric_hausdorff_px": float(
                np.sum(registration_errors * registration_weights)
                / positive_weight
            ),
            "pairing_mean_normalized_translation_error": float(
                np.sum((registration_errors / area) * registration_weights)
                / positive_weight
            ),
        }
    )
    return output


def endpoint_pigeonhole_corrosion_bootstrap(
    predictions_by_condition: Mapping[
        str, Mapping[str, Sequence[CorrosionPrediction]]
    ],
    *,
    labels: Sequence[bool],
    source_unit_ids: Sequence[Sequence[str]],
    translation_targets: Sequence[Optional[Tuple[float, float]]],
    full_threshold: Optional[float] = None,
    thresholds: Optional[Mapping[str, float]] = None,
    contour_area_sums_by_condition: Optional[
        Mapping[str, Sequence[float]]
    ] = None,
    fixed_common: Sequence[bool],
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Mapping[str, object]:
    """Shared-draw endpoint-pigeonhole inference on the fixed common population."""

    if type(replicates) is not int or replicates <= 0:  # noqa: E721
        raise ValueError("replicates must be a positive integer")
    if type(seed) is not int or seed < 0:  # noqa: E721
        raise ValueError("seed must be a non-negative integer")
    condition_names = tuple(condition.name for condition in FIXED_CONDITIONS)
    if tuple(predictions_by_condition) != condition_names:
        raise ValueError("bootstrap conditions differ from fixed order")
    count = len(labels)
    if not (
        count
        == len(source_unit_ids)
        == len(translation_targets)
        == len(fixed_common)
    ):
        raise ValueError("bootstrap side-table vectors differ")
    selected = np.asarray(fixed_common, dtype=np.bool_)
    method_order = tuple(predictions_by_condition[condition_names[0]])
    if method_order not in {METHODS, LEGACY_METHODS}:
        raise ValueError("bootstrap method inventory differs")
    translation_methods = tuple(
        method for method in TRANSLATION_METHODS if method in method_order
    )
    if thresholds is None:
        if full_threshold is None:
            raise ValueError("bootstrap frozen thresholds are missing")
        threshold_by_method = {
            method: float(full_threshold) for method in translation_methods
        }
    else:
        if tuple(thresholds) != method_order:
            raise ValueError("bootstrap threshold inventory differs")
        threshold_by_method = {
            method: float(thresholds[method]) for method in translation_methods
        }
    if contour_area_sums_by_condition is None:
        area_by_condition = {
            condition: np.ones(count, dtype=np.float64)
            for condition in condition_names
        }
    else:
        if tuple(contour_area_sums_by_condition) != condition_names:
            raise ValueError("bootstrap contour-area condition inventory differs")
        area_by_condition = {
            condition: np.asarray(
                contour_area_sums_by_condition[condition], dtype=np.float64
            )
            for condition in condition_names
        }
        if any(value.shape != (count,) for value in area_by_condition.values()):
            raise ValueError("bootstrap contour-area vectors differ")
    selected_labels = np.asarray(labels, dtype=np.bool_)[selected]
    if not selected_labels.any() or selected_labels.all():
        raise RachelN512CorrosionError(
            "fixed common bootstrap population must contain both classes"
        )
    dependency_units, first_unit, second_unit = _endpoint_dependency_index(
        source_unit_ids, selected
    )
    plans: Dict[Tuple[str, str], _BootstrapRankingPlan] = {}
    for condition in condition_names:
        if tuple(predictions_by_condition[condition]) != method_order:
            raise ValueError("bootstrap condition lacks a method")
        for method in method_order:
            rows = predictions_by_condition[condition][method]
            if len(rows) != count:
                raise ValueError("bootstrap prediction count differs")
            scores = np.asarray(
                [row.probability for row, keep in zip(rows, selected) if keep],
                dtype=np.float64,
            )
            if not np.isfinite(scores).all():
                raise RachelN512CorrosionError(
                    "fixed common bootstrap scores must be finite"
                )
            plans[(condition, method)] = _BootstrapRankingPlan.build(
                selected_labels, scores
            )
    positive = selected_labels.copy()
    pose_errors: Dict[Tuple[str, str], np.ndarray] = {}
    pose_valid: Dict[Tuple[str, str], np.ndarray] = {}
    fallback_errors: Dict[Tuple[str, str], np.ndarray] = {}
    translation_accepted: Dict[Tuple[str, str], np.ndarray] = {}
    selected_targets = [
        target for target, keep in zip(translation_targets, selected) if keep
    ]
    for condition in condition_names:
        for method in translation_methods:
            key = (condition, method)
            rows = [
                row
                for row, keep in zip(
                    predictions_by_condition[condition][method], selected
                )
                if keep
            ]
            errors = np.full(len(rows), np.nan, dtype=np.float64)
            valid = np.zeros(len(rows), dtype=np.bool_)
            identity_fallback = np.full(len(rows), np.nan, dtype=np.float64)
            for index, (row, label, target) in enumerate(
                zip(rows, selected_labels, selected_targets)
            ):
                if not label:
                    if target is not None:
                        raise ValueError("negative pair carries translation GT")
                    continue
                if target is None:
                    raise RachelN512CorrosionError(
                        "fixed common positive lacks translation GT"
                    )
                translation_valid = (
                    row.translation_valid and row.translation_hat_rc is not None
                )
                effective = (
                    row.translation_hat_rc if translation_valid else (0.0, 0.0)
                )
                error = math.hypot(
                    effective[0] - target[0], effective[1] - target[1]
                )
                identity_fallback[index] = error
                if translation_valid:
                    valid[index] = True
                    errors[index] = error
            if not np.isfinite(identity_fallback[positive]).all():
                raise RachelN512CorrosionError(
                    "positive PairingNet-compatible fallback errors must be finite"
                )
            pose_errors[key] = errors
            pose_valid[key] = valid
            fallback_errors[key] = identity_fallback
            translation_accepted[key] = np.asarray(
                [
                    row.valid
                    and math.isfinite(row.probability)
                    and row.probability >= threshold_by_method[method]
                    for row in rows
                ],
                dtype=np.bool_,
            )

    unit_weights = np.ones(len(selected_labels), dtype=np.float64)
    ranking_points = {
        key: plan.evaluate(unit_weights) for key, plan in plans.items()
    }
    direct_points = {
        (condition, method): _direct_geometry_statistics(
            pose_errors[(condition, method)],
            pose_valid[(condition, method)],
            fallback_errors[(condition, method)],
            translation_accepted[(condition, method)],
            positive,
            area_by_condition[condition][selected],
            unit_weights,
        )
        for condition in condition_names
        for method in translation_methods
    }
    ranking_draws: Dict[Tuple[str, str], Tuple[list, list]] = {
        key: ([], []) for key in plans
    }
    direct_draws: Dict[Tuple[str, str], Dict[str, list]] = {
        key: {metric: [] for metric in point}
        for key, point in direct_points.items()
    }
    skipped = 0
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        weights = _pigeonhole_pair_weights(
            len(dependency_units), first_unit, second_unit, rng
        )
        if (
            not np.any(weights[selected_labels] > 0.0)
            or not np.any(weights[~selected_labels] > 0.0)
            or any(
                not np.any(weights[positive & pose_valid[key]] > 0.0)
                for key in pose_valid
            )
        ):
            skipped += 1
            continue
        for key, plan in plans.items():
            auroc, auprc = plan.evaluate(weights)
            ranking_draws[key][0].append(auroc)
            ranking_draws[key][1].append(auprc)
        for condition in condition_names:
            for method in translation_methods:
                key = (condition, method)
                statistics = _direct_geometry_statistics(
                    pose_errors[key],
                    pose_valid[key],
                    fallback_errors[key],
                    translation_accepted[key],
                    positive,
                    area_by_condition[condition][selected],
                    weights,
                )
                for metric, value in statistics.items():
                    direct_draws[key][metric].append(value)
    if replicates == skipped:
        raise RachelN512CorrosionError(
            "all endpoint-bootstrap replicates were single-class"
        )

    by_condition_method: Dict[str, object] = {}
    condition_minus_clean: Dict[str, object] = {}
    full_minus_comparator: Dict[str, object] = {}
    for condition in condition_names:
        by_condition_method[condition] = {}
        condition_minus_clean[condition] = {}
        full_minus_comparator[condition] = {}
        for method in method_order:
            key = (condition, method)
            by_condition_method[condition][method] = {
                metric: _bootstrap_interval(
                    ranking_draws[key][index], ranking_points[key][index]
                )
                for index, metric in enumerate(("auroc", "auprc"))
            }
            clean_key = ("clean", method)
            condition_minus_clean[condition][method] = {
                metric: _bootstrap_interval(
                    np.asarray(ranking_draws[key][index])
                    - np.asarray(ranking_draws[clean_key][index]),
                    ranking_points[key][index] - ranking_points[clean_key][index],
                    delta=True,
                )
                for index, metric in enumerate(("auroc", "auprc"))
            }
        for comparator in method_order:
            if comparator == "full_n512":
                continue
            full_key = (condition, "full_n512")
            comparator_key = (condition, comparator)
            full_minus_comparator[condition][comparator] = {
                metric: _bootstrap_interval(
                    np.asarray(ranking_draws[full_key][index])
                    - np.asarray(ranking_draws[comparator_key][index]),
                    ranking_points[full_key][index]
                    - ranking_points[comparator_key][index],
                    delta=True,
                )
                for index, metric in enumerate(("auroc", "auprc"))
            }

    direct_by_method: Dict[str, object] = {}
    for method in translation_methods:
        by_condition: Dict[str, object] = {}
        minus_clean: Dict[str, object] = {}
        for condition in condition_names:
            key = (condition, method)
            clean_key = ("clean", method)
            by_condition[condition] = {
                metric: _bootstrap_interval(direct_draws[key][metric], point)
                for metric, point in direct_points[key].items()
            }
            minus_clean[condition] = {
                metric: _bootstrap_interval(
                    np.asarray(direct_draws[key][metric])
                    - np.asarray(direct_draws[clean_key][metric]),
                    point - direct_points[clean_key][metric],
                    delta=True,
                )
                for metric, point in direct_points[key].items()
            }
        direct_by_method[method] = {
            "threshold": threshold_by_method[method],
            "threshold_source": "frozen_validation_checkpoint_bound_artifact",
            "by_condition": by_condition,
            "condition_minus_clean": minus_clean,
            "correspondence": {
                "status": "not_applicable",
                "reason": "current-condition corrupted contour indices have no GT",
            },
        }
    full_projection = direct_by_method["full_n512"]
    return {
        "bootstrap": "endpoint-unit_pigeonhole_product_multiplicity",
        "shared_draws_across_all_methods_conditions_and_geometry_metrics": True,
        "population": "fixed_all_method_all_condition_common_valid",
        "method_order": list(method_order),
        "formal_exact_six": method_order == METHODS,
        "sampling_dependency_unit_count": len(dependency_units),
        "replicates_requested": replicates,
        "valid_replicates": replicates - skipped,
        "skipped_single_class_replicates": skipped,
        "seed": seed,
        "confidence_interval": "two-sided_95_percentile",
        "ranking": {
            "by_condition_method": by_condition_method,
            "condition_minus_clean_same_method": condition_minus_clean,
            "full_n512_minus_comparator_within_condition": full_minus_comparator,
        },
        "positive_only_translation_gt_and_joint": {
            "negative_pairs_excluded": True,
            "threshold": threshold_by_method["full_n512"],
            "threshold_source": "frozen_validation_checkpoint_bound_artifact",
            "by_condition": full_projection["by_condition"],
            "condition_minus_clean": full_projection["condition_minus_clean"],
        },
        "direct_geometry_by_method": direct_by_method,
    }


def _condition_pair_rows(
    condition: CorrosionCondition,
    predictions: Mapping[str, Sequence[CorrosionPrediction]],
    *,
    manifest_rows: Sequence[object],
    pair_inputs: Sequence[BlindSyntheticPair],
    fragment_audit: Mapping[str, Mapping[str, object]],
    fixed_common: Sequence[bool],
    thresholds: Mapping[str, float],
) -> Tuple[Mapping[str, object], ...]:
    if len(manifest_rows) != len(pair_inputs):
        raise RachelN512CorrosionError("blind/side-table pair counts differ")
    method_order = tuple(thresholds)
    if method_order not in {METHODS, LEGACY_METHODS} or tuple(predictions) != method_order:
        raise RachelN512CorrosionError("pair-row method inventory differs")
    output = []
    for index, (manifest, pair_input) in enumerate(zip(manifest_rows, pair_inputs)):
        if pair_input.pair_id != manifest.pair_id:
            raise RachelN512CorrosionError("blind/side-table pair order differs")
        methods: Dict[str, object] = {}
        for method in method_order:
            prediction = predictions[method][index]
            if prediction.pair_id != manifest.pair_id:
                raise RachelN512CorrosionError("pair score order changed before write")
            method_row: Dict[str, object] = {
                "main_score": MAIN_SCORE_BY_METHOD[method],
                "probability": (
                    prediction.probability if prediction.valid else None
                ),
                "valid": prediction.valid,
                "decision_at_frozen_validation_threshold": (
                    prediction.probability >= thresholds[method]
                    if prediction.valid
                    else None
                ),
                "validation_threshold": thresholds[method],
            }
            if method in TRANSLATION_METHODS:
                method_row["geometry"] = {
                    "translation_hat_rc": (
                        list(prediction.translation_hat_rc)
                        if prediction.translation_hat_rc is not None
                        else None
                    ),
                    "translation_dispersion_px": (
                        prediction.translation_dispersion_px
                    ),
                    "translation_valid": prediction.translation_valid,
                    "correspondence": {
                        "status": "not_applicable",
                        "reason": (
                            "corrupted contour indices have no current-condition GT"
                        ),
                    },
                }
            methods[method] = method_row
        output.append(
            {
                "schema_version": PAIR_SCHEMA_VERSION,
                "profile": FIXED_PROFILE,
                "condition": condition.name,
                "pair_id": manifest.pair_id,
                "label": manifest.label,
                "cluster_id": manifest.cluster_id,
                "source_unit_ids": list(manifest.source_unit_ids),
                "fragment_morphology": {
                    "a": dict(fragment_audit[pair_input.fragment_a_id]),
                    "b": dict(fragment_audit[pair_input.fragment_b_id]),
                },
                "fixed_all_method_all_condition_common_valid": bool(
                    fixed_common[index]
                ),
                "methods": methods,
            }
        )
    return tuple(output)


def _source_path(configured: Optional[Path], value: object) -> Path:
    if configured is not None:
        candidate = configured
    elif isinstance(value, str) and value:
        candidate = Path(value)
    else:
        raise RachelN512CorrosionError("Rachel dataset root is missing")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RachelN512CorrosionError("Rachel dataset root is unavailable") from error
    if not resolved.is_dir():
        raise RachelN512CorrosionError("Rachel dataset root must be a directory")
    return resolved


def _reject_output_inside_sources(output_root: Path, sources: Sequence[Path]) -> None:
    resolved = output_root.resolve()
    for source in sources:
        try:
            resolved.relative_to(source)
        except ValueError:
            continue
        raise RachelN512CorrosionError(
            "robustness output may not mutate a training, dataset, or sealed result"
        )


def _set_determinism(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RachelN512CorrosionError(
            "CUBLAS deterministic workspace must be configured before torch import"
        )
    if type(seed) is not int or seed < 0:  # noqa: E721
        raise RachelN512CorrosionError("training seed is invalid")
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_rachel_n512_corrosion_robustness(
    config: RachelN512CorrosionConfig,
) -> Path:
    """Execute the frozen exact-six seven-condition synthetic protocol."""

    if not isinstance(config, RachelN512CorrosionConfig):
        raise TypeError("config must be RachelN512CorrosionConfig")
    formal_evaluation = not config.compatibility_mode
    method_order = METHODS if formal_evaluation else LEGACY_METHODS
    try:
        run_directory = config.run_directory.resolve(strict=True)
        matched_run_directory = config.matched_mm_run_directory.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RachelN512CorrosionError("one training run directory is missing") from error

    # No synthetic test artefact/path is accepted by these freeze calls.  The
    # formal N=512 plateau gate is also checked before the combined clean
    # reference or the test manifest can be opened.
    n512_receipt, n512_receipt_sha, n512_winner_sequence = (
        sealed._freeze_completed_winners(run_directory)
    )
    sealed._require_formal_convergence(n512_receipt, n512_winner_sequence)
    n512_winners = {winner.arm: winner for winner in n512_winner_sequence}
    if set(n512_winners) != set(N512_METHODS):
        raise RachelN512CorrosionError("formal N=512 winners are incomplete")
    train_config = n512_receipt.get("config")
    if not isinstance(train_config, Mapping):
        raise RachelN512CorrosionError("formal N=512 config is missing")
    precision = train_config.get("precision")
    seed = train_config.get("seed")
    if precision not in {"fp32", "bf16"} or type(seed) is not int or seed < 0:  # noqa: E721
        raise RachelN512CorrosionError("formal N=512 precision/seed is invalid")

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RachelN512CorrosionError("requested CUDA device is unavailable")
    matched_receipt, matched_receipt_sha, matched_winners = (
        freeze_matched_mm_winners(matched_run_directory, device=device)
    )
    if tuple(matched_winners) != MATCHED_METHODS:
        raise RachelN512CorrosionError(
            "matched-MM converged and epoch-5 winners are incomplete"
        )
    matched_config = matched_receipt.get("config")
    if not isinstance(matched_config, Mapping):
        raise RachelN512CorrosionError("matched-MM formal config is missing")
    try:
        dataset_root = sealed._require_matched_training_alignment(
            n512_receipt, matched_receipt
        )
        matched_training_hash_evidence = sealed._matched_training_hash_evidence(
            n512_receipt,
            matched_receipt,
            n512_run_directory=run_directory,
            matched_run_directory=matched_run_directory,
        )
    except sealed.RachelN512SealedTestError as error:
        raise RachelN512CorrosionError(str(error)) from error
    if config.dataset_root is not None:
        configured_dataset_root = _source_path(config.dataset_root, None)
        if configured_dataset_root != dataset_root:
            raise RachelN512CorrosionError(
                "configured dataset root differs from formal aligned release"
            )
    benchmark_winners: Mapping[
        str, benchmark_adapter.FrozenSameDataBenchmark
    ] = {}
    benchmark_training_evidence: Optional[Mapping[str, object]] = None
    pairingnet_run_directory: Optional[Path] = None
    shreddingnet_freeze_path: Optional[Path] = None
    if formal_evaluation:
        assert config.pairingnet_run_directory is not None
        assert config.shreddingnet_freeze_path is not None
        try:
            pairingnet_run_directory = (
                config.pairingnet_run_directory.resolve(strict=True)
            )
            shreddingnet_freeze_path = (
                config.shreddingnet_freeze_path.resolve(strict=True)
            )
            benchmark_winners = benchmark_adapter.freeze_same_data_benchmarks(
                pairingnet_run_directory=pairingnet_run_directory,
                shreddingnet_freeze_path=shreddingnet_freeze_path,
                device=device,
            )
            benchmark_training_evidence = (
                sealed._benchmark_training_manifest_evidence(
                    dataset_root, benchmark_winners
                )
            )
        except (
            OSError,
            RuntimeError,
            benchmark_adapter.RachelBenchmarkEvalAdapterError,
            sealed.RachelN512SealedTestError,
        ) as error:
            raise RachelN512CorrosionError(
                "same-data benchmark pre-test freeze/alignment failed: " + str(error)
            ) from error
        if tuple(benchmark_winners) != BENCHMARK_METHODS:
            raise RachelN512CorrosionError(
                "formal exact-six benchmark winner inventory is incomplete"
            )
    reference_config = n512_winners["full_n512"].model_config
    coarse_config = n512_winners["coarse_only"].model_config
    if (
        reference_config.canvas_size != 800
        or reference_config.coarse_size != 128
        or reference_config.contour_cap != 512
        or coarse_config.canvas_size != reference_config.canvas_size
        or coarse_config.coarse_size != reference_config.coarse_size
        or coarse_config.contour_cap != reference_config.contour_cap
    ):
        raise RachelN512CorrosionError(
            "frozen winners do not share the Rachel 800/coarse128/N512 contract"
        )
    _set_determinism(seed)

    try:
        sealed_test_directory = config.sealed_test_directory.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RachelN512CorrosionError(
            "combined sealed-test directory is missing"
        ) from error
    output_root = config.output_root.resolve()
    protected_sources = [
        run_directory,
        matched_run_directory,
        dataset_root,
        sealed_test_directory,
    ]
    if pairingnet_run_directory is not None and shreddingnet_freeze_path is not None:
        protected_sources.extend(
            (pairingnet_run_directory, shreddingnet_freeze_path.parent)
        )
    _reject_output_inside_sources(output_root, protected_sources)
    # Output-path rejection is intentionally before the first read of a
    # synthetic-test receipt, pair-score file, or test manifest.
    output_root.mkdir(parents=True, exist_ok=True)

    # First test-related access: the completed combined result supplies the
    # mandatory clean reproduction reference.  It is opened only after the
    # complete formal exact-six checkpoint/threshold and manifest authority.
    sealed_reference = _load_sealed_reference_after_freeze(
        sealed_test_directory,
        n512_receipt_sha256=n512_receipt_sha,
        n512_winners=n512_winners,
        matched_receipt_sha256=matched_receipt_sha,
        matched_winners=matched_winners,
        benchmark_winners=benchmark_winners,
        formal_evaluation=formal_evaluation,
        methods=method_order,
    )
    if sealed_reference.precision != precision:
        raise RachelN512CorrosionError(
            "clean reference precision differs from frozen N=512 precision"
        )
    if sealed_reference.batch_size != config.batch_size:
        raise RachelN512CorrosionError(
            "batch_size must exactly match the combined sealed clean reference"
        )

    test_manifest_path, manifest_rows = sealed._test_manifest_rows(
        dataset_root,
        expected_count=EXPECTED_TEST_PAIRS,
        require_model_masks=True,
    )
    manifest_sha = _sha256_file(test_manifest_path)
    if manifest_sha != sealed_reference.manifest_sha256:
        raise RachelN512CorrosionError(
            "current fixed test manifest differs from combined sealed reference"
        )
    pair_ids = tuple(row.pair_id for row in manifest_rows)
    labels = tuple(row.label for row in manifest_rows)
    clusters = tuple(row.cluster_id for row in manifest_rows)
    reference_identity = tuple(
        (row.pair_id, row.label, row.cluster_id, row.source_unit_ids)
        for row in sealed_reference.expected[method_order[0]]
    )
    if tuple(
        (row.pair_id, row.label, row.cluster_id, row.source_unit_ids)
        for row in manifest_rows
    ) != reference_identity:
        raise RachelN512CorrosionError(
            "test manifest identity/labels/clusters differ from sealed reference"
        )
    if _canonical_sha256(list(pair_ids)) != sealed_reference.pair_ids_sha256:
        raise RachelN512CorrosionError("fixed test pair order hash differs")
    pair_inputs = _pair_inputs_from_manifest(dataset_root, manifest_rows)

    thresholds: Dict[str, float] = {}
    for method in method_order:
        if method in N512_METHODS:
            thresholds[method] = float(n512_winners[method].threshold.threshold)
        elif method in MATCHED_METHODS:
            thresholds[method] = float(
                matched_winners[method].threshold.threshold
            )
        else:
            thresholds[method] = float(benchmark_winners[method].threshold)
    method_authority: Dict[str, Mapping[str, object]] = {}
    for method in method_order:
        if method in N512_METHODS:
            method_authority[method] = {
                "checkpoint_sha256": n512_winners[method].checkpoint_sha256,
                "threshold_sha256": n512_winners[method].threshold.content_sha256,
            }
        elif method in MATCHED_METHODS:
            method_authority[method] = {
                "checkpoint_sha256": matched_winners[method].checkpoint_sha256,
                "threshold_sha256": (
                    matched_winners[method].threshold.content_sha256
                ),
            }
        else:
            method_authority[method] = {
                "freeze_authority_sha256": (
                    benchmark_winners[method].freeze_authority_sha256
                ),
                "checkpoint_sha256_by_stage": dict(
                    benchmark_winners[method].checkpoint_sha256_by_stage
                ),
                "threshold_sha256": (
                    benchmark_winners[method].threshold_artifact_sha256
                ),
            }
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "profile": FIXED_PROFILE,
        "formal_evaluation": formal_evaluation,
        "compatibility_mode": config.compatibility_mode,
        "method_order": list(method_order),
        "n512_training_receipt_sha256": n512_receipt_sha,
        "matched_mm_training_receipt_sha256": matched_receipt_sha,
        "training_alignment_evidence_sha256": _canonical_sha256(
            matched_training_hash_evidence
        ),
        "benchmark_training_alignment_evidence_sha256": (
            _canonical_sha256(benchmark_training_evidence)
            if benchmark_training_evidence is not None
            else None
        ),
        "combined_sealed_receipt_sha256": sealed_reference.receipt_sha256,
        "test_manifest_sha256": manifest_sha,
        "pair_ids_sha256": sealed_reference.pair_ids_sha256,
        "conditions": [condition.to_dict() for condition in FIXED_CONDITIONS],
        "corruption_seed": seed,
        "canvas_size": reference_config.canvas_size,
        "contour_cap": reference_config.contour_cap,
        "contour_smoothing_sigma": 3.0,
        "batch_size": config.batch_size,
        "precision": precision,
        "device": config.device,
        "clean_tolerance": config.clean_tolerance,
        "bootstrap_replicates": config.bootstrap_replicates,
        "bootstrap_seed": config.bootstrap_seed,
        "methods": method_authority,
    }
    fingerprint = _canonical_sha256(fingerprint_payload)
    final_directory = output_root / ("corrosion-" + fingerprint[:16])
    partial_directory = output_root / (".partial-corrosion-" + fingerprint[:16])
    if final_directory.exists() or partial_directory.exists():
        raise RachelN512CorrosionError(
            "robustness output already exists; preserve it rather than overwrite"
        )
    partial_directory.mkdir()
    _atomic_json(
        partial_directory / "evaluation_config.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "running_formal_exact_six_frozen_synthetic_corrosion_only"
                if formal_evaluation
                else "running_non_formal_legacy_four_corrosion_compatibility"
            ),
            "formal_evaluation": formal_evaluation,
            "compatibility_mode": config.compatibility_mode,
            "fingerprint_sha256": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "run_directory": str(run_directory),
            "matched_mm_run_directory": str(matched_run_directory),
            "pairingnet_run_directory": (
                str(pairingnet_run_directory)
                if pairingnet_run_directory is not None
                else None
            ),
            "shreddingnet_freeze_path": (
                str(shreddingnet_freeze_path)
                if shreddingnet_freeze_path is not None
                else None
            ),
            "sealed_test_directory": str(sealed_reference.directory),
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "test_accessed": True,
            "real_external_test_accessed": False,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_fit_performed": False,
            "formal_config_verified": True,
            "both_arms_validation_plateau_verified": True,
            "matched_training_alignment_verified": True,
            "matched_training_alignment_hash_evidence": (
                matched_training_hash_evidence
            ),
            "benchmark_training_alignment_evidence": benchmark_training_evidence,
            "all_six_winners_and_validation_thresholds_frozen_before_test_open": (
                True if formal_evaluation else False
            ),
            "raw_target_archives_opened": False,
            "raw_target_archives_planned_only_after_all_predictions": True,
            "evaluation_history_disclosure": {
                "evaluation_kind": "fixed_protocol_repeat",
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
                "claim_of_pristine_first_project_test": False,
                "combined_sealed_clean_reference_gt_derived_fields_parsed": True,
                "combined_sealed_clean_reference_gt_derived_fields_used_for_model_or_selection": False,
            },
        },
    )

    started = time.perf_counter()
    predictions_by_condition: Dict[
        str, Mapping[str, Tuple[CorrosionPrediction, ...]]
    ] = {}
    processing: Dict[str, Mapping[str, object]] = {}
    morphology_by_condition: Dict[
        str, Mapping[str, Mapping[str, object]]
    ] = {}
    clean_source_authority = _CleanSourceAuthority()
    clean_gate: Optional[Mapping[str, object]] = None
    raw_target_access_started = False
    target_receipt: Optional[Mapping[str, object]] = None
    try:
        for condition in FIXED_CONDITIONS:
            cache = _ConditionFragmentCache(
                condition=condition,
                seed=seed,
                canvas_size=reference_config.canvas_size,
                contour_cap=reference_config.contour_cap,
                capacity=config.fragment_cache_size,
                clean_source_authority=clean_source_authority,
            )
            condition_predictions, forward_audit = _score_condition(
                pair_inputs,
                fragment_cache=cache,
                n512_winners=n512_winners,
                matched_winners=matched_winners,
                benchmark_winners=benchmark_winners,
                methods=method_order,
                device=device,
                precision=precision,
                batch_size=config.batch_size,
            )
            predictions_by_condition[condition.name] = condition_predictions
            processing[condition.name] = {
                **dict(cache.receipt()),
                "forward_audit": forward_audit,
            }
            morphology_by_condition[condition.name] = cache.fragment_audit()
            if condition.family == "clean":
                clean_gate = _clean_parity_gate(
                    condition_predictions,
                    sealed_reference.expected,
                    tolerance=config.clean_tolerance,
                )
                _atomic_json(
                    partial_directory / "clean_reproduction_gate.json", clean_gate
                )
        assert clean_gate is not None
        fixed_common = tuple(
            all(
                predictions_by_condition[condition.name][method][index].valid
                and math.isfinite(
                    predictions_by_condition[condition.name][method][index].probability
                )
                for condition in FIXED_CONDITIONS
                for method in method_order
            )
            for index in range(len(pair_ids))
        )
        # This is the sole raw-target opening point and it is deliberately
        # after every exact-six (or explicit legacy-four) condition forward.
        contour_area_sums_by_condition = {}
        for condition in FIXED_CONDITIONS:
            audit = morphology_by_condition[condition.name]
            sums = []
            for pair in pair_inputs:
                first_area = audit[pair.fragment_a_id].get(
                    "benchmark_contour_area_px2"
                )
                second_area = audit[pair.fragment_b_id].get(
                    "benchmark_contour_area_px2"
                )
                if (
                    isinstance(first_area, bool)
                    or not isinstance(first_area, (int, float))
                    or isinstance(second_area, bool)
                    or not isinstance(second_area, (int, float))
                    or not math.isfinite(float(first_area))
                    or not math.isfinite(float(second_area))
                    or float(first_area) + float(second_area) <= 0.0
                ):
                    raise RachelN512CorrosionError(
                        "current-condition PairingNet contour area is unavailable"
                    )
                sums.append(float(first_area) + float(second_area))
            contour_area_sums_by_condition[condition.name] = tuple(sums)
        raw_target_access_started = True
        translation_targets, target_receipt = (
            _load_positive_translation_targets_after_predictions(
                dataset_root,
                test_manifest_path,
                pair_ids=pair_ids,
                labels=labels,
                expected_manifest_sha256=manifest_sha,
            )
        )
        summary = summarize_corrosion_predictions(
            predictions_by_condition,
            labels=labels,
            clusters=clusters,
            thresholds=thresholds,
            pair_ids=pair_ids,
        )
        assert isinstance(summary, dict)
        _attach_positive_translation_metrics(
            summary,
            predictions_by_condition,
            labels=labels,
            targets=translation_targets,
            threshold=thresholds["full_n512"],
            fixed_common=fixed_common,
            thresholds=thresholds,
            contour_area_sums_by_condition=contour_area_sums_by_condition,
        )
        paired_bootstrap = endpoint_pigeonhole_corrosion_bootstrap(
            predictions_by_condition,
            labels=labels,
            source_unit_ids=tuple(row.source_unit_ids for row in manifest_rows),
            translation_targets=translation_targets,
            thresholds=thresholds,
            contour_area_sums_by_condition=contour_area_sums_by_condition,
            fixed_common=fixed_common,
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed,
        )
        summary["paired_endpoint_pigeonhole_bootstrap"] = paired_bootstrap
        primary = summary["primary_population"]
        assert isinstance(primary, Mapping)
        fixed_ids_sha = primary["pair_ids_sha256"]
        condition_results = []
        for condition in FIXED_CONDITIONS:
            rows = _condition_pair_rows(
                condition,
                predictions_by_condition[condition.name],
                manifest_rows=manifest_rows,
                pair_inputs=pair_inputs,
                fragment_audit=morphology_by_condition[condition.name],
                fixed_common=fixed_common,
                thresholds=thresholds,
            )
            score_path = partial_directory / "conditions" / condition.name / "pair_scores.jsonl"
            _atomic_jsonl(score_path, rows)
            condition_results.append(
                {
                    "condition": condition.name,
                    "pair_scores": str(score_path.relative_to(partial_directory)),
                    "pair_scores_sha256": _sha256_file(score_path),
                    "pair_scores_count": len(rows),
                    "processing": processing[condition.name],
                    "metrics": summary["conditions"][condition.name],
                }
            )
        clean_source_path = partial_directory / "clean_source_authority.jsonl"
        clean_source_rows = clean_source_authority.rows()
        _atomic_jsonl(clean_source_path, clean_source_rows)
        clean_source_receipt = {
            **dict(clean_source_authority.receipt()),
            "map": str(clean_source_path.relative_to(partial_directory)),
            "map_sha256": _sha256_file(clean_source_path),
            "map_count": len(clean_source_rows),
        }
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "complete_formal_exact_six_frozen_synthetic_corrosion_only"
                if formal_evaluation
                else "complete_non_formal_legacy_four_corrosion_compatibility"
            ),
            "formal_evaluation": formal_evaluation,
            "compatibility_mode": config.compatibility_mode,
            "fingerprint_sha256": fingerprint,
            "source_training_runs": {
                "rachel_n512": {
                    "run_directory": str(run_directory),
                    "receipt_sha256": n512_receipt_sha,
                    "formal_plateau_verified_before_test_open": True,
                    "both_winners_and_thresholds_frozen_before_test_open": True,
                },
                "matched_mm": {
                    "run_directory": str(matched_run_directory),
                    "receipt_sha256": matched_receipt_sha,
                    "formal_plateau_verified_before_test_open": True,
                    "converged_and_epoch5_winners_and_thresholds_frozen_before_test_open": True,
                },
                "same_data_benchmarks": (
                    {
                        "all_winners_and_validation_thresholds_frozen_before_test_open": True,
                        "exact_train_validation_manifest_alignment": (
                            benchmark_training_evidence
                        ),
                        "methods": {
                            method: benchmark_winners[method].provenance()
                            for method in BENCHMARK_METHODS
                        },
                    }
                    if formal_evaluation
                    else None
                ),
                "alignment_hash_evidence": matched_training_hash_evidence,
            },
            "clean_reference": {
                "sealed_test_directory": str(sealed_reference.directory),
                "receipt_sha256": sealed_reference.receipt_sha256,
                "gate": clean_gate,
                "sealed_gt_derived_fields_parsed": True,
                "sealed_gt_derived_fields_used_for_model_or_selection": False,
            },
            "clean_source_authority": clean_source_receipt,
            "post_prediction_positive_translation_targets": target_receipt,
            "paired_endpoint_pigeonhole_bootstrap": paired_bootstrap,
            "test_population": {
                "dataset_root": str(dataset_root),
                "manifest": str(test_manifest_path),
                "manifest_sha256": manifest_sha,
                "pair_ids_sha256": sealed_reference.pair_ids_sha256,
                "pair_count": len(pair_ids),
                "positive_count": sum(labels),
                "negative_count": len(labels) - sum(labels),
                "same_pair_ids_labels_clusters_source_units_and_order_at_every_condition": True,
                "fixed_all_method_all_condition_common_valid_pair_ids_sha256": fixed_ids_sha,
                "one_common_corrupted_mask_pair_population_per_condition": True,
            },
            "protocol": {
                "profile": FIXED_PROFILE,
                "formal_evaluation": formal_evaluation,
                "compatibility_mode": config.compatibility_mode,
                "formal_exact_six": formal_evaluation,
                "formal_method_inventory": (
                    list(METHODS) if formal_evaluation else None
                ),
                "method_order": list(method_order),
                "same_data_benchmark_methods_included": formal_evaluation,
                "all_winners_and_validation_thresholds_frozen_before_first_test_path_access": (
                    True if formal_evaluation else False
                ),
                "exact_train_validation_manifest_alignment_verified_before_first_test_path_access": (
                    True if formal_evaluation else False
                ),
                "condition_order": [
                    condition.name for condition in FIXED_CONDITIONS
                ],
                "condition_definitions": [
                    condition.to_dict() for condition in FIXED_CONDITIONS
                ],
                "mask_only": True,
                "rgb_text_or_ocr_used": False,
                "known_upright_orientation": True,
                "rotation_search": False,
                "corruption_is_fragment_level_pair_and_label_blind": True,
                "corruption_identity_inputs": (
                    "fixed_profile+training_seed+fragment_id+fixed_condition_parameters"
                ),
                "same_corrupted_masks_supplied_to_all_methods": True,
                "same_corrupted_masks_supplied_to_all_six_methods": (
                    True if formal_evaluation else False
                ),
                "contour_reextracted_from_each_corrupted_800_bool_mask": True,
                "n512_geometry_requires_exactly_one_8_connected_component": True,
                "coarse_and_matched_keep_entire_corrupted_bool_mask_when_split": True,
                "pairingnet_and_shreddingnet_keep_entire_corrupted_bool_mask_when_split": (
                    True if formal_evaluation else None
                ),
                "each_method_forwarded_each_pair_exactly_once_per_condition": True,
                "clean_contour_archives_opened": False,
                "raw_target_archives_opened_only_after_all_predictions": True,
                "raw_target_archives_used_only_for_positive_translation_metrics": True,
                "correspondence_indices_materialized_or_used": False,
                "translation_targets_used_for_model_or_selection": False,
                "real_external_test_accessed": False,
                "primary_metrics": (
                    "all-method ranking plus direct TE/assembly/Pairing RR-HD-NTE "
                    "on fixed all-method all-condition common-valid population"
                ),
                "benchmark_correspondence_metrics": "not_applicable_truthfully",
                "same_data_benchmark_adaptations_not_exact_reproductions": (
                    True if formal_evaluation else None
                ),
                "native_pairingnet_or_shreddingnet_global_metrics_claimed": False,
                "frozen_validation_thresholds_are_secondary_only": True,
                "threshold_fit_performed": False,
                "checkpoint_selection_performed": False,
                "training_performed": False,
                "precision": precision,
                "device": config.device,
                "batch_size": config.batch_size,
                "formal_config_verified": True,
                "both_arms_validation_plateau_verified": True,
                "matched_training_alignment_verified": True,
                "evaluation_history_disclosure": {
                    "evaluation_kind": "fixed_protocol_repeat",
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
                    "claim_of_pristine_first_project_test": False,
                    "combined_sealed_clean_reference_gt_derived_fields_parsed": True,
                    "combined_sealed_clean_reference_gt_derived_fields_used_for_model_or_selection": False,
                    "raw_positive_target_archives_opened_after_all_target_blind_predictions": True,
                    "raw_positive_targets_used_for_model_checkpoint_threshold_or_condition_selection": False,
                },
            },
            "summary": summary,
            "condition_results": condition_results,
            "seconds": time.perf_counter() - started,
            "test_accessed": True,
            "real_external_test_accessed": False,
        }
        _atomic_json(partial_directory / "robustness_receipt.json", receipt)
        for winner in n512_winners.values():
            winner.model.to("cpu")
        for winner in matched_winners.values():
            winner.model.to("cpu")
        for winner in benchmark_winners.values():
            winner.release_to_cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _publish_directory_no_replace(
            partial_directory,
            final_directory,
            completion_receipt="robustness_receipt.json",
        )
        return final_directory
    except BaseException as error:
        for winner in benchmark_winners.values():
            try:
                winner.release_to_cpu()
            except Exception:
                pass
        _atomic_json(
            partial_directory / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_after_synthetic_test_open",
                "formal_evaluation": formal_evaluation,
                "compatibility_mode": config.compatibility_mode,
                "error_type": type(error).__name__,
                "error": str(error),
                "clean_reproduction_gate_completed": clean_gate is not None,
                "raw_positive_target_access_started_after_all_predictions": (
                    raw_target_access_started
                ),
                "raw_positive_target_metrics_completed": target_receipt is not None,
                "test_accessed": True,
                "real_external_test_accessed": False,
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "threshold_fit_performed": False,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--matched-mm-run-directory", type=Path, required=True)
    parser.add_argument("--pairingnet-run-directory", type=Path)
    parser.add_argument("--shreddingnet-freeze-path", type=Path)
    parser.add_argument("--sealed-test-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fragment-cache-size", type=int, default=384)
    parser.add_argument("--clean-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--compatibility-non-formal",
        action="store_true",
        help="permit only the legacy four-method corrosion inventory as non-formal",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    output = run_rachel_n512_corrosion_robustness(
        RachelN512CorrosionConfig(
            run_directory=arguments.run_directory,
            matched_mm_run_directory=arguments.matched_mm_run_directory,
            pairingnet_run_directory=arguments.pairingnet_run_directory,
            shreddingnet_freeze_path=arguments.shreddingnet_freeze_path,
            sealed_test_directory=arguments.sealed_test_directory,
            output_root=arguments.output_root,
            dataset_root=arguments.dataset_root,
            device=arguments.device,
            batch_size=arguments.batch_size,
            fragment_cache_size=arguments.fragment_cache_size,
            clean_tolerance=arguments.clean_tolerance,
            bootstrap_replicates=arguments.bootstrap_replicates,
            bootstrap_seed=arguments.bootstrap_seed,
            compatibility_mode=arguments.compatibility_non_formal,
        )
    )
    receipt = _read_json_object(
        output / "robustness_receipt.json", "completed robustness receipt"
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output_directory": str(output),
                "clean_reproduction_gate": receipt["clean_reference"]["gate"][
                    "passed"
                ],
                "primary_population": receipt["summary"]["primary_population"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


__all__ = (
    "CorrosionCondition",
    "CorrosionPrediction",
    "FIXED_CONDITIONS",
    "FIXED_PROFILE",
    "LEGACY_METHODS",
    "BENCHMARK_METHODS",
    "METHODS",
    "RachelN512CorrosionConfig",
    "RachelN512CorrosionError",
    "degrade_mask",
    "main",
    "run_rachel_n512_corrosion_robustness",
    "summarize_corrosion_predictions",
)


if __name__ == "__main__":
    raise SystemExit(main())

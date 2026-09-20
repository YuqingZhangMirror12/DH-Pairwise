"""Leakage-safe runtime loader for the Rachel N=512 Pairwise release.

Only the selected split manifest, model-facing binary masks/contours, and the
positive pair target archive are readable through this module.  RGB images and
parent-canvas audit artefacts are deliberately outside its path vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError


SCHEMA_VERSION = "rachel-pairwise-runtime-loader/1.0"
VALID_SPLITS = frozenset(("train", "val", "test"))
_BANNED_KEYS = frozenset(
    (
        "target_audit",
        "parent_mask_path",
        "parent_to_model_offset_rc",
        "bbox_min_rc",
        "pad_start_rc",
        "rgb",
        "rgb_path",
        "source_rgb_path",
        "jpeg_path",
    )
)


class RachelDatasetError(ValueError):
    """A selected row or an on-disk artefact violates the release contract."""


@dataclass(frozen=True)
class RachelDatasetConfig:
    mask_size: int = 800
    coarse_size: int = 128
    contour_cap: int = 512
    residual_median_limit_px: float = 2.0
    residual_p95_limit_px: float = 4.0

    def __post_init__(self) -> None:
        for name in ("mask_size", "coarse_size", "contour_cap"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        for name in ("residual_median_limit_px", "residual_p95_limit_px"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(name + " must be finite and positive")


@dataclass(frozen=True)
class _FragmentInput:
    token: str
    mask_path: Path
    contour_path: Path


@dataclass(frozen=True)
class _SelectedPair:
    pair_id: str
    label: bool
    fragment_a: _FragmentInput
    fragment_b: _FragmentInput
    target_path: Union[Path, None]


@dataclass(frozen=True)
class _FragmentArrays:
    mask: np.ndarray
    coarse_mask: np.ndarray
    points_rc: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class RachelPairSample:
    """One unpadded pair.  It contains no filesystem or parent-frame fields."""

    pair_id: str
    fragment_a_token: str
    fragment_b_token: str
    mask_a: np.ndarray
    mask_b: np.ndarray
    coarse_mask_a: np.ndarray
    coarse_mask_b: np.ndarray
    points_rc_a: np.ndarray
    points_rc_b: np.ndarray
    contour_valid_a: np.ndarray
    contour_valid_b: np.ndarray
    target_a: np.ndarray
    target_b: np.ndarray
    label: np.float32
    translation_a_to_b_rc: np.ndarray
    translation_a_to_b_xy_cartesian: np.ndarray
    translation_valid: np.bool_


@dataclass(frozen=True)
class RachelBatch:
    """Padded, framework-neutral batch ready for NumPy-to-torch conversion."""

    pair_ids: Tuple[str, ...]
    fragment_a_tokens: Tuple[str, ...]
    fragment_b_tokens: Tuple[str, ...]
    mask_a: np.ndarray
    mask_b: np.ndarray
    coarse_mask_a: np.ndarray
    coarse_mask_b: np.ndarray
    points_rc_a: np.ndarray
    points_rc_b: np.ndarray
    contour_valid_a: np.ndarray
    contour_valid_b: np.ndarray
    target_a: np.ndarray
    target_b: np.ndarray
    labels: np.ndarray
    translation_a_to_b_rc: np.ndarray
    translation_a_to_b_xy_cartesian: np.ndarray
    translation_valid: np.ndarray

    def as_dict(self) -> Dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    output = np.ascontiguousarray(value, dtype=dtype)
    output.setflags(write=False)
    return output


def _reject_leakage_keys(value: object, location: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).casefold()
            if key_text in _BANNED_KEYS or key_text.startswith("target_audit_"):
                raise RachelDatasetError(
                    "forbidden parent/RGB audit field at {}: {}".format(location, key)
                )
            _reject_leakage_keys(nested, location + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_leakage_keys(nested, location + "[{}]".format(index))


def _safe_release_path(
    root: Path,
    value: object,
    *,
    prefix: Tuple[str, ...],
    suffix: str,
) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RachelDatasetError("release path must be a non-empty POSIX relative path")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise RachelDatasetError("unsafe release path: " + value)
    if (
        tuple(logical.parts[: len(prefix)]) != prefix
        or logical.suffix.casefold() != suffix
    ):
        raise RachelDatasetError(
            "release path is outside the allowed artefact class: " + value
        )
    unresolved = root.joinpath(*logical.parts)
    current = root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise RachelDatasetError(
                "symlinked release artefact is forbidden: " + value
            )
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise RachelDatasetError(
            "release artefact is missing or escapes root: " + value
        ) from error
    if not resolved.is_file():
        raise RachelDatasetError("release artefact is not a regular file: " + value)
    return resolved


def _fragment_input(root: Path, value: object, name: str) -> _FragmentInput:
    if not isinstance(value, Mapping):
        raise RachelDatasetError(name + " must be an object")
    token = value.get("fragment_token")
    if not isinstance(token, str) or not token:
        raise RachelDatasetError(name + ".fragment_token is required")
    return _FragmentInput(
        token=token,
        mask_path=_safe_release_path(
            root,
            value.get("model_mask_path"),
            prefix=("model", "masks_800"),
            suffix=".png",
        ),
        contour_path=_safe_release_path(
            root,
            value.get("contour_path"),
            prefix=("model", "contours_n512"),
            suffix=".npz",
        ),
    )


def _read_selected_pair(
    root: Path, split: str, row: object, line_number: int
) -> _SelectedPair:
    if not isinstance(row, Mapping):
        raise RachelDatasetError(
            "selected line {} is not an object".format(line_number)
        )
    _reject_leakage_keys(row, "selected line {}".format(line_number))
    if row.get("split") != split:
        raise RachelDatasetError("selected row split disagrees with requested split")
    pair_id = row.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise RachelDatasetError("selected pair_id is required")
    label = row.get("label")
    if type(label) is not bool:  # noqa: E721
        raise RachelDatasetError("selected label must be an explicit bool")
    target_value = row.get("correspondence_path")
    if label:
        target_path = _safe_release_path(
            root,
            target_value,
            prefix=("targets", "pairs"),
            suffix=".npz",
        )
    else:
        if target_value is not None:
            raise RachelDatasetError(
                "negative selected pair must not reference a target"
            )
        for name in ("translation_a_to_b_rc", "translation_a_to_b_xy_cartesian"):
            if row.get(name) is not None:
                raise RachelDatasetError(
                    "negative selected pair must not carry translation"
                )
        target_path = None
    return _SelectedPair(
        pair_id=pair_id,
        label=label,
        fragment_a=_fragment_input(root, row.get("fragment_a"), "fragment_a"),
        fragment_b=_fragment_input(root, row.get("fragment_b"), "fragment_b"),
        target_path=target_path,
    )


def _load_mask(
    path: Path, config: RachelDatasetConfig
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image)
    except (OSError, UnidentifiedImageError) as error:
        raise RachelDatasetError("cannot decode model mask") from error
    expected = (config.mask_size, config.mask_size)
    if pixels.ndim != 2 or pixels.shape != expected or pixels.dtype != np.uint8:
        raise RachelDatasetError(
            "model mask must be uint8 grayscale {}x{}".format(*expected)
        )
    unique = set(int(value) for value in np.unique(pixels))
    if not unique.issubset({0, 255}) or unique != {0, 255}:
        raise RachelDatasetError(
            "model mask must contain exactly binary disk values 0 and 255"
        )
    mask = pixels == 255
    nearest = getattr(Image, "Resampling", Image).NEAREST
    coarse = np.asarray(
        Image.fromarray(pixels, mode="L").resize(
            (config.coarse_size, config.coarse_size), resample=nearest
        ),
        dtype=np.uint8,
    )
    if not set(int(value) for value in np.unique(coarse)).issubset({0, 255}):
        raise RachelDatasetError("nearest coarse mask unexpectedly became non-binary")
    return (
        _readonly(mask[None, :, :], np.float32),
        _readonly((coarse == 255)[None, :, :], np.float32),
    )


def _load_contour(
    path: Path, config: RachelDatasetConfig
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"points_rc", "valid"}:
                raise RachelDatasetError("contour archive has unexpected fields")
            points = np.asarray(archive["points_rc"])
            valid = np.asarray(archive["valid"])
    except (OSError, ValueError) as error:
        if isinstance(error, RachelDatasetError):
            raise
        raise RachelDatasetError("cannot decode contour archive") from error
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or not 4 <= len(points) <= config.contour_cap
    ):
        raise RachelDatasetError("contour points must have shape [N,2], 4<=N<=cap")
    if (
        valid.shape != (len(points),)
        or valid.dtype != np.bool_
        or np.count_nonzero(valid) < 4
    ):
        raise RachelDatasetError("contour valid mask is invalid")
    if not np.all(np.isfinite(points[valid])):
        raise RachelDatasetError("valid contour points must be finite")
    if np.any(points[valid] < 0.0) or np.any(points[valid] > config.mask_size - 1):
        raise RachelDatasetError("valid contour points fall outside model canvas")
    return _readonly(points, np.float32), _readonly(valid, np.bool_)


def _load_positive_target(
    path: Path,
    first: _FragmentArrays,
    second: _FragmentArrays,
    config: RachelDatasetConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "correspondence_indices",
        "translation_a_to_b_rc",
        "translation_a_to_b_xy_cartesian",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise RachelDatasetError(
                    "positive target archive has unexpected fields"
                )
            pairs = np.asarray(archive["correspondence_indices"])
            translation_rc = np.asarray(
                archive["translation_a_to_b_rc"], dtype=np.float64
            )
            translation_xy = np.asarray(
                archive["translation_a_to_b_xy_cartesian"], dtype=np.float64
            )
    except (OSError, ValueError) as error:
        if isinstance(error, RachelDatasetError):
            raise
        raise RachelDatasetError("cannot decode positive target archive") from error
    if pairs.ndim != 2 or pairs.shape[1:] != (2,) or len(pairs) == 0:
        raise RachelDatasetError(
            "positive correspondence_indices must have non-empty [K,2] shape"
        )
    if not np.issubdtype(pairs.dtype, np.integer):
        raise RachelDatasetError("positive correspondence indices must be integers")
    pairs = np.asarray(pairs, dtype=np.int64)
    if len(np.unique(pairs[:, 0])) != len(pairs) or len(np.unique(pairs[:, 1])) != len(
        pairs
    ):
        raise RachelDatasetError("positive correspondences must be one-to-one")
    if (
        np.any(pairs[:, 0] < 0)
        or np.any(pairs[:, 0] >= len(first.points_rc))
        or np.any(pairs[:, 1] < 0)
        or np.any(pairs[:, 1] >= len(second.points_rc))
        or not np.all(first.valid[pairs[:, 0]])
        or not np.all(second.valid[pairs[:, 1]])
    ):
        raise RachelDatasetError(
            "positive correspondence references invalid contour token"
        )
    if translation_rc.shape != (2,) or translation_xy.shape != (2,):
        raise RachelDatasetError("positive translation targets must have shape [2]")
    if not np.all(np.isfinite(translation_rc)) or not np.all(
        np.isfinite(translation_xy)
    ):
        raise RachelDatasetError("positive translation targets must be finite")
    expected_xy = np.asarray((translation_rc[1], -translation_rc[0]))
    if not np.allclose(translation_xy, expected_xy, rtol=0.0, atol=1e-4):
        raise RachelDatasetError("RC and Cartesian translation targets disagree")
    residual = (
        second.points_rc[pairs[:, 1]].astype(np.float64)
        - first.points_rc[pairs[:, 0]].astype(np.float64)
        - translation_rc[None, :]
    )
    distances = np.linalg.norm(residual, axis=1)
    median = float(np.median(distances))
    p95 = float(np.quantile(distances, 0.95))
    if (
        not np.all(np.isfinite(distances))
        or median > config.residual_median_limit_px
        or p95 > config.residual_p95_limit_px
    ):
        raise RachelDatasetError(
            "positive contour/translation residual gate failed: median={:.6g}, p95={:.6g}".format(
                median, p95
            )
        )
    target_a = np.full(len(first.points_rc), -2, dtype=np.int64)
    target_b = np.full(len(second.points_rc), -2, dtype=np.int64)
    target_a[first.valid] = -1
    target_b[second.valid] = -1
    target_a[pairs[:, 0]] = pairs[:, 1]
    target_b[pairs[:, 1]] = pairs[:, 0]
    return (
        _readonly(target_a, np.int64),
        _readonly(target_b, np.int64),
        _readonly(translation_rc, np.float32),
        _readonly(translation_xy, np.float32),
    )


class RachelPairDataset:
    """Lazy selected-split dataset for the Rachel release."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        config: RachelDatasetConfig = RachelDatasetConfig(),
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError("split must be train, val, or test")
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise RachelDatasetError("dataset root must be a directory")
        self.split = split
        self.config = config
        manifest = _safe_release_path(
            self.root,
            "pairs/{}.jsonl".format(split),
            prefix=("pairs",),
            suffix=".jsonl",
        )
        rows = []
        seen = set()
        try:
            with manifest.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise RachelDatasetError(
                            "invalid JSON at selected line {}".format(line_number)
                        ) from error
                    pair = _read_selected_pair(self.root, split, value, line_number)
                    if pair.pair_id in seen:
                        raise RachelDatasetError(
                            "duplicate selected pair_id: " + pair.pair_id
                        )
                    seen.add(pair.pair_id)
                    rows.append(pair)
        except OSError as error:
            raise RachelDatasetError("cannot read selected split manifest") from error
        if not rows:
            raise RachelDatasetError("selected split manifest is empty")
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def _load_fragment(self, value: _FragmentInput) -> _FragmentArrays:
        mask, coarse = _load_mask(value.mask_path, self.config)
        points, valid = _load_contour(value.contour_path, self.config)
        return _FragmentArrays(mask, coarse, points, valid)

    def __getitem__(self, index: int) -> RachelPairSample:
        if type(index) is not int:  # noqa: E721
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self._rows)
        if not 0 <= index < len(self._rows):
            raise IndexError(index)
        row = self._rows[index]
        first = self._load_fragment(row.fragment_a)
        second = self._load_fragment(row.fragment_b)
        if row.label:
            assert row.target_path is not None
            target_a, target_b, translation_rc, translation_xy = _load_positive_target(
                row.target_path, first, second, self.config
            )
            translation_valid = np.bool_(True)
        else:
            target_a_value = np.full(len(first.points_rc), -2, dtype=np.int64)
            target_b_value = np.full(len(second.points_rc), -2, dtype=np.int64)
            target_a_value[first.valid] = -1
            target_b_value[second.valid] = -1
            target_a = _readonly(target_a_value, np.int64)
            target_b = _readonly(target_b_value, np.int64)
            translation_rc = _readonly(np.zeros(2), np.float32)
            translation_xy = _readonly(np.zeros(2), np.float32)
            translation_valid = np.bool_(False)
        return RachelPairSample(
            pair_id=row.pair_id,
            fragment_a_token=row.fragment_a.token,
            fragment_b_token=row.fragment_b.token,
            mask_a=first.mask,
            mask_b=second.mask,
            coarse_mask_a=first.coarse_mask,
            coarse_mask_b=second.coarse_mask,
            points_rc_a=first.points_rc,
            points_rc_b=second.points_rc,
            contour_valid_a=first.valid,
            contour_valid_b=second.valid,
            target_a=target_a,
            target_b=target_b,
            label=np.float32(row.label),
            translation_a_to_b_rc=translation_rc,
            translation_a_to_b_xy_cartesian=translation_xy,
            translation_valid=translation_valid,
        )


def collate_rachel_pairs(
    samples: Sequence[RachelPairSample], contour_cap: int = 512
) -> RachelBatch:
    """Pad variable contour lengths; ``-1`` is dustbin and ``-2`` is ignore."""

    rows = tuple(samples)
    if not rows:
        raise ValueError("cannot collate an empty Rachel batch")
    if type(contour_cap) is not int or contour_cap <= 0:  # noqa: E721
        raise ValueError("contour_cap must be a positive integer")
    if any(
        len(row.points_rc_a) > contour_cap or len(row.points_rc_b) > contour_cap
        for row in rows
    ):
        raise RachelDatasetError("sample contour exceeds batch cap")
    batch_size = len(rows)

    def points(side: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        output = np.zeros((batch_size, contour_cap, 2), dtype=np.float32)
        valid = np.zeros((batch_size, contour_cap), dtype=np.bool_)
        target = np.full((batch_size, contour_cap), -2, dtype=np.int64)
        for batch_index, row in enumerate(rows):
            value = getattr(row, "points_rc_" + side)
            mask = getattr(row, "contour_valid_" + side)
            assignment = getattr(row, "target_" + side)
            length = len(value)
            if mask.shape != (length,) or assignment.shape != (length,):
                raise RachelDatasetError("sample contour arrays disagree in length")
            output[batch_index, :length] = value
            valid[batch_index, :length] = mask
            target[batch_index, :length] = assignment
        return output, valid, target

    points_a, valid_a, target_a = points("a")
    points_b, valid_b, target_b = points("b")
    return RachelBatch(
        pair_ids=tuple(row.pair_id for row in rows),
        fragment_a_tokens=tuple(row.fragment_a_token for row in rows),
        fragment_b_tokens=tuple(row.fragment_b_token for row in rows),
        mask_a=np.stack([row.mask_a for row in rows]),
        mask_b=np.stack([row.mask_b for row in rows]),
        coarse_mask_a=np.stack([row.coarse_mask_a for row in rows]),
        coarse_mask_b=np.stack([row.coarse_mask_b for row in rows]),
        points_rc_a=points_a,
        points_rc_b=points_b,
        contour_valid_a=valid_a,
        contour_valid_b=valid_b,
        target_a=target_a,
        target_b=target_b,
        labels=np.asarray([row.label for row in rows], dtype=np.float32),
        translation_a_to_b_rc=np.stack(
            [row.translation_a_to_b_rc for row in rows]
        ).astype(np.float32, copy=False),
        translation_a_to_b_xy_cartesian=np.stack(
            [row.translation_a_to_b_xy_cartesian for row in rows]
        ).astype(np.float32, copy=False),
        translation_valid=np.asarray(
            [row.translation_valid for row in rows], dtype=np.bool_
        ),
    )


__all__ = [
    "RachelBatch",
    "RachelDatasetConfig",
    "RachelDatasetError",
    "RachelPairDataset",
    "RachelPairSample",
    "SCHEMA_VERSION",
    "collate_rachel_pairs",
]

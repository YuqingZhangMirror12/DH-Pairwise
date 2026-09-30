"""Freeze every Rachel train/validation byte consumed by pairwise benchmarks.

This module is deliberately narrower than the Rachel runtime loader.  It may
open the two selected training manifests, their referenced model assets, and a
small, fixed set of preprocessing/selection authority JSON files.  It has no
code path that opens ``pairs/test.jsonl`` or real-world evaluation data.

The public workflow is::

    python -m experiments.rachel_n512_formal_30k.rachel_train_val_asset_freeze \
        create --dataset-root DATASET --output-dir FREEZE
    python -m experiments.rachel_n512_formal_30k.rachel_train_val_asset_freeze \
        verify --dataset-root DATASET --freeze-dir FREEZE

``create`` also materializes a self-contained, read-only ``frozen_data`` view;
benchmark runners must use that directory as their dataset root.  ``verify``
reconstructs both the source and frozen-view semantic inventories and rehashes
every byte.  A manifest-preserving mutation of a mask, contour, target, or
authority file is therefore a hard failure.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence
from typing import Set, Tuple, Union
import uuid


INVENTORY_SCHEMA_VERSION = "rachel-train-val-semantic-asset-inventory/3.0"
RECEIPT_SCHEMA_VERSION = "rachel-train-val-semantic-asset-freeze/3.0"
VERIFICATION_SCHEMA_VERSION = "rachel-train-val-semantic-asset-verification/3.0"
FROZEN_VIEW_SCHEMA_VERSION = "rachel-frozen-train-val-dataset-view/2.0"
PREPROCESS_SCHEMA_VERSION = "rachel-pairwise-n512-preprocessing/1.0"
SELECTION_SCHEMA_VERSION = "rachel-pairwise-30k-selection/1.0"
PREPROCESS_STATUS = "complete_rachel_pairwise_n512_30k"
SELECTION_SEED = "rachel-pairwise-n512-v1"

INVENTORY_FILENAME = "train_val_semantic_asset_inventory.json"
RECEIPT_FILENAME = "train_val_semantic_asset_freeze_receipt.json"
FROZEN_DATA_DIRNAME = "frozen_data"

FORMAL_COUNTS = {
    "train": {"rows": 24_000, "positive": 12_000, "negative": 12_000},
    "val": {"rows": 3_000, "positive": 1_500, "negative": 1_500},
}
FORMAL_SELECTED_COUNTS = {
    "train": {
        "positive": 12_000,
        "same_folder_hard": 6_000,
        "cross_folder_scale_matched": 6_000,
    },
    "val": {
        "positive": 1_500,
        "same_folder_hard": 750,
        "cross_folder_scale_matched": 750,
    },
    "test": {
        "positive": 1_500,
        "same_folder_hard": 750,
        "cross_folder_scale_matched": 750,
    },
}
FORMAL_SPLIT_QUOTAS = {
    split: {
        **counts,
        "negative": counts["same_folder_hard"]
        + counts["cross_folder_scale_matched"],
        "total": 2 * counts["positive"],
    }
    for split, counts in FORMAL_SELECTED_COUNTS.items()
}

_MANIFEST_PATHS = {
    "train": "pairs/train.jsonl",
    "val": "pairs/val.jsonl",
}
_AUTHORITY_PATHS = {
    "preprocess_run_config": "run_config.json",
    "preprocess_receipt": "preprocess_receipt.json",
    "preprocess_summary": "qa/preprocess_summary.json",
    "selection_summary": "pairs/summary.json",
    "selection_lineage_assignments": "pairs/lineage_splits.json",
}
_EXPECTED_RECEIPT_PATHS = {
    "model_masks": "model/masks_800",
    "model_contours": "model/contours_n512",
    "target_parent_masks": "targets/parent_masks_800",
    "target_pair_geometry": "targets/pairs",
    "fragments_manifest": "manifests/fragments.jsonl",
    "within_candidates_manifest": "manifests/within_candidates.jsonl",
    "selected_pairs": "pairs",
}
_BANNED_MANIFEST_KEYS = frozenset(
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
_ASSET_CLASSES = {
    "model_mask": (("model", "masks_800"), ".png"),
    "contour_n512": (("model", "contours_n512"), ".npz"),
    "positive_correspondence_target": (("targets", "pairs"), ".npz"),
}
_JSON_ROLES = frozenset(_AUTHORITY_PATHS)
_ALLOWED_DATASET_STATIC_PATHS = frozenset(
    tuple(_MANIFEST_PATHS.values()) + tuple(_AUTHORITY_PATHS.values())
)


class RachelTrainValAssetFreezeError(ValueError):
    """The dataset or a freeze artifact violates the fail-closed contract."""


def _fail(message: str) -> None:
    raise RachelTrainValAssetFreezeError(message)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RachelTrainValAssetFreezeError(
            "value is not canonical finite JSON"
        ) from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _duplicate_rejecting_object(pairs: Sequence[Tuple[str, object]]) -> object:
    output: Dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            _fail("duplicate JSON key: " + key)
        output[key] = value
    return output


def _reject_json_constant(value: str) -> object:
    _fail("non-finite JSON constant is forbidden: " + value)
    raise AssertionError("unreachable")


def _strict_json_bytes(payload: bytes, description: str) -> object:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RachelTrainValAssetFreezeError(
            description + " is not strict UTF-8 JSON"
        ) from error


def _lexical_absolute(path: Union[str, Path]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _path_components(path: Path) -> Iterable[Path]:
    if not path.is_absolute():
        _fail("internal path must be absolute")
    current = Path(path.anchor)
    yield current
    for part in path.parts[1:]:
        current = current / part
        yield current


def _assert_no_symlink_components(
    path: Path, *, description: str, allow_missing_leaf: bool = False
) -> None:
    parts = tuple(_path_components(path))
    for index, component in enumerate(parts):
        try:
            metadata = os.lstat(component)
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            _fail(description + " has a missing path component: " + str(component))
        except OSError as error:
            raise RachelTrainValAssetFreezeError(
                description + " cannot be inspected: " + str(component)
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            _fail(description + " traverses a symlink: " + str(component))


def _validated_dataset_root(value: Union[str, Path]) -> Path:
    root = _lexical_absolute(value)
    _assert_no_symlink_components(root, description="dataset root")
    try:
        metadata = os.lstat(root)
    except OSError as error:
        raise RachelTrainValAssetFreezeError("dataset root is unreadable") from error
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("dataset root is not a directory")
    return root


def _race_test_hook(event: str, context: Mapping[str, object]) -> None:
    """No-op hook used only for deterministic TOCTOU regression injection."""

    del event, context


def _entry_identity(metadata: os.stat_result) -> Tuple[int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
    )


def _stable_file_identity(metadata: os.stat_result) -> Tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(metadata.st_size),
        int(
            getattr(
                metadata,
                "st_mtime_ns",
                int(metadata.st_mtime * 1_000_000_000),
            )
        ),
        int(
            getattr(
                metadata,
                "st_ctime_ns",
                int(metadata.st_ctime * 1_000_000_000),
            )
        ),
    )


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_REGULAR_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _open_absolute_directory_chain(
    path: Path, description: str
) -> Tuple[int, Tuple[Tuple[int, int, int], ...]]:
    """Open an absolute directory one no-follow component at a time."""

    if not path.is_absolute():
        _fail(description + " must be absolute")
    descriptor = -1
    identities: List[Tuple[int, int, int]] = []
    try:
        descriptor = os.open(path.anchor, _DIRECTORY_OPEN_FLAGS)
        root_metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            _fail(description + " filesystem root is not a directory")
        identities.append(_entry_identity(root_metadata))
        for component in path.parts[1:]:
            entry_metadata = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(entry_metadata.st_mode):
                _fail(description + " traverses a symlink component")
            if not stat.S_ISDIR(entry_metadata.st_mode):
                _fail(description + " component is not a directory")
            child = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=descriptor,
            )
            child_metadata = os.fstat(child)
            if (
                not stat.S_ISDIR(child_metadata.st_mode)
                or _entry_identity(entry_metadata) != _entry_identity(child_metadata)
            ):
                os.close(child)
                _fail(description + " directory entry identity changed while opening")
            os.close(descriptor)
            descriptor = child
            identities.append(_entry_identity(child_metadata))
        return descriptor, tuple(identities)
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise RachelTrainValAssetFreezeError(
            description + " cannot be opened component-wise without symlinks"
        ) from error
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _revalidate_absolute_directory_chain(
    path: Path,
    expected: Sequence[Tuple[int, int, int]],
    description: str,
) -> None:
    descriptor, observed = _open_absolute_directory_chain(path, description)
    os.close(descriptor)
    if tuple(observed) != tuple(expected):
        _fail(description + " was renamed or replaced after it was anchored")


def _open_relative_parent_chain(
    root_descriptor: int,
    directory_parts: Sequence[str],
    description: str,
) -> Tuple[int, Tuple[Tuple[int, int, int], ...]]:
    descriptor = os.dup(root_descriptor)
    identities: List[Tuple[int, int, int]] = []
    try:
        for component in directory_parts:
            entry_metadata = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(entry_metadata.st_mode):
                _fail(description + " parent traverses a symlink component")
            if not stat.S_ISDIR(entry_metadata.st_mode):
                _fail(description + " parent component is not a directory")
            child = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=descriptor,
            )
            child_metadata = os.fstat(child)
            if (
                not stat.S_ISDIR(child_metadata.st_mode)
                or _entry_identity(entry_metadata) != _entry_identity(child_metadata)
            ):
                os.close(child)
                _fail(description + " parent entry identity changed while opening")
            os.close(descriptor)
            descriptor = child
            identities.append(_entry_identity(child_metadata))
        return descriptor, tuple(identities)
    except OSError as error:
        os.close(descriptor)
        raise RachelTrainValAssetFreezeError(
            description + " parent cannot be opened component-wise"
        ) from error
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_relative_parent_chain(
    root_descriptor: int,
    directory_parts: Sequence[str],
    expected: Sequence[Tuple[int, int, int]],
    description: str,
) -> None:
    descriptor, observed = _open_relative_parent_chain(
        root_descriptor, directory_parts, description
    )
    os.close(descriptor)
    if tuple(observed) != tuple(expected):
        _fail(description + " parent was renamed or replaced while reading")


class _AnchoredDatasetRoot:
    """Race-resistant dataset reader rooted at a persistent directory FD."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._parent_anchor: Optional["_AnchoredDatasetRoot"] = None
        self._relative_parts: Tuple[str, ...] = ()
        self._relative_chain: Tuple[Tuple[int, int, int], ...] = ()
        self._description = "dataset root"
        self.descriptor, self._absolute_chain = _open_absolute_directory_chain(
            root, "dataset root"
        )
        self._root_identity = _entry_identity(os.fstat(self.descriptor))
        self.revalidate()

    @classmethod
    def from_anchored_parent(
        cls,
        parent: "_AnchoredDatasetRoot",
        relative: str,
        description: str,
    ) -> "_AnchoredDatasetRoot":
        """Derive a child root only through an already anchored parent FD."""

        logical = _safe_logical_path(relative, description=description)
        parts = tuple(PurePosixPath(logical).parts)
        descriptor, chain = _open_relative_parent_chain(
            parent.descriptor, parts, description
        )
        child = cls.__new__(cls)
        child.root = parent.root.joinpath(*parts)
        child._parent_anchor = parent
        child._relative_parts = parts
        child._relative_chain = chain
        child._description = description
        child.descriptor = descriptor
        child._absolute_chain = ()
        try:
            child._root_identity = _entry_identity(os.fstat(descriptor))
            child.revalidate()
        except BaseException:
            child.close()
            raise
        return child

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "_AnchoredDatasetRoot":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def revalidate(self) -> None:
        if self.descriptor < 0:
            _fail("dataset root anchor is closed")
        if _entry_identity(os.fstat(self.descriptor)) != self._root_identity:
            _fail("anchored dataset root descriptor identity changed")
        if self._parent_anchor is None:
            _revalidate_absolute_directory_chain(
                self.root, self._absolute_chain, self._description
            )
        else:
            self._parent_anchor.revalidate()
            _revalidate_relative_parent_chain(
                self._parent_anchor.descriptor,
                self._relative_parts,
                self._relative_chain,
                self._description,
            )

    def read(
        self, relative: str, description: str, *, retain_bytes: bool
    ) -> Tuple[Optional[bytes], Dict[str, object]]:
        logical = _safe_logical_path(relative, description=description)
        if logical.casefold() == "pairs/test.jsonl":
            _fail("sealed test manifest access is forbidden")
        parts = PurePosixPath(logical).parts
        parent_parts = parts[:-1]
        leaf = parts[-1]
        parent_descriptor, parent_chain = _open_relative_parent_chain(
            self.descriptor, parent_parts, description
        )
        file_descriptor = -1
        try:
            entry_before = os.stat(
                leaf,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(entry_before.st_mode):
                _fail(description + " is a forbidden symlink entry")
            if not stat.S_ISREG(entry_before.st_mode):
                _fail(description + " is not a regular file")
            file_descriptor = os.open(
                leaf,
                _REGULAR_OPEN_FLAGS,
                dir_fd=parent_descriptor,
            )
            before = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or _entry_identity(entry_before) != _entry_identity(before)
            ):
                _fail(description + " is not one stable regular-file entry")
            digest = hashlib.sha256()
            total = 0
            chunks: Optional[List[bytes]] = [] if retain_bytes else None
            while True:
                chunk = os.read(file_descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after = os.fstat(file_descriptor)
            _race_test_hook(
                "dataset_after_file_hash_before_entry_revalidation",
                {"dataset_root": self.root, "relative_path": logical},
            )
            entry_after = os.stat(
                leaf,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _stable_file_identity(before) != _stable_file_identity(after)
                or _entry_identity(entry_before) != _entry_identity(entry_after)
                or _entry_identity(after) != _entry_identity(entry_after)
                or total != int(before.st_size)
            ):
                _fail(description + " entry mutated or was replaced while hashing")
            _revalidate_relative_parent_chain(
                self.descriptor,
                parent_parts,
                parent_chain,
                description,
            )
            self.revalidate()
            return (
                b"".join(chunks) if chunks is not None else None,
                {
                    "sha256": digest.hexdigest(),
                    "size_bytes": total,
                    "device": int(before.st_dev),
                    "inode": int(before.st_ino),
                },
            )
        except OSError as error:
            raise RachelTrainValAssetFreezeError(
                description + " changed or became unreadable while hashing"
            ) from error
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            os.close(parent_descriptor)


def _safe_logical_path(
    value: object,
    *,
    description: str,
    prefix: Optional[Tuple[str, ...]] = None,
    suffix: Optional[str] = None,
) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(description + " must be a non-empty POSIX relative path")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(
        part in {"", ".", ".."} for part in logical.parts
    ):
        _fail(description + " is unsafe: " + value)
    # ``PurePosixPath`` normalizes repeated separators and dot components.
    # Accepting the normalized spelling would create two lexical names for one
    # artifact, so require the source string itself to be canonical.
    if logical.as_posix() != value:
        _fail(description + " is a non-canonical path alias: " + value)
    if prefix is not None and tuple(logical.parts[: len(prefix)]) != prefix:
        _fail(description + " is outside its allowed artifact class: " + value)
    if suffix is not None and logical.suffix.casefold() != suffix:
        _fail(description + " has the wrong suffix: " + value)
    normalized = logical.as_posix()
    if normalized.casefold() == "pairs/test.jsonl":
        _fail("sealed test manifest access is forbidden")
    return normalized


class _AssetCollector:
    """Deduplicate exact paths while rejecting distinct hard-link aliases."""

    def __init__(
        self,
        root: Path,
        *,
        anchor: Optional[_AnchoredDatasetRoot] = None,
    ) -> None:
        self.root = root
        self.anchor = anchor if anchor is not None else _AnchoredDatasetRoot(root)
        self._owns_anchor = anchor is None
        if self.anchor.root != root:
            _fail("asset collector root differs from its anchored root")
        self._assets: Dict[str, MutableMapping[str, object]] = {}
        self._physical_paths: Dict[Tuple[int, int], str] = {}

    def close(self) -> None:
        if self._owns_anchor:
            self.anchor.close()

    def __enter__(self) -> "_AssetCollector":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def add(
        self,
        relative: str,
        role: str,
        *,
        retain_bytes: bool = False,
    ) -> Optional[bytes]:
        if relative.casefold() == "pairs/test.jsonl":
            _fail("sealed test manifest access is forbidden")
        existing = self._assets.get(relative)
        if existing is not None:
            roles = existing["roles"]
            assert isinstance(roles, set)
            roles.add(role)
            existing["reference_count"] = int(existing["reference_count"]) + 1
            if retain_bytes:
                payload, observed = self.anchor.read(
                    relative, role, retain_bytes=True
                )
                if (
                    observed["sha256"] != existing["sha256"]
                    or observed["size_bytes"] != existing["size_bytes"]
                ):
                    _fail(role + " changed between duplicate references")
                return payload
            return None
        payload, observed = self.anchor.read(
            relative, role, retain_bytes=retain_bytes
        )
        physical = (int(observed["device"]), int(observed["inode"]))
        alias = self._physical_paths.get(physical)
        if alias is not None and alias != relative:
            _fail(
                "distinct dataset paths alias the same physical file: {} and {}".format(
                    alias, relative
                )
            )
        self._physical_paths[physical] = relative
        self._assets[relative] = {
            "relative_path": relative,
            "roles": {role},
            "reference_count": 1,
            "sha256": observed["sha256"],
            "size_bytes": observed["size_bytes"],
        }
        return payload

    def inventory(self) -> List[Dict[str, object]]:
        output = []
        for relative in sorted(self._assets):
            row = dict(self._assets[relative])
            roles = row["roles"]
            assert isinstance(roles, set)
            row["roles"] = sorted(roles)
            output.append(row)
        return output


def _reject_leakage_keys(value: object, location: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            folded = str(key).casefold()
            if folded in _BANNED_MANIFEST_KEYS or folded.startswith("target_audit_"):
                _fail("forbidden parent/RGB audit field at {}: {}".format(location, key))
            _reject_leakage_keys(nested, location + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_leakage_keys(nested, location + "[{}]".format(index))


def _required_text(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(description + " must be a non-empty string")
    return value


def _fragment_fields(value: object, description: str) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        _fail(description + " must be an object")
    token = _required_text(value.get("fragment_token"), description + ".fragment_token")
    lineage = _required_text(value.get("split_unit_id"), description + ".split_unit_id")
    image_name = value.get("image_name")
    if image_name is not None and image_name != lineage:
        _fail(description + ".image_name disagrees with split_unit_id")
    parent_group = _required_text(
        value.get("parent_group_id"), description + ".parent_group_id"
    )
    mask_prefix, mask_suffix = _ASSET_CLASSES["model_mask"]
    contour_prefix, contour_suffix = _ASSET_CLASSES["contour_n512"]
    mask = _safe_logical_path(
        value.get("model_mask_path"),
        description=description + ".model_mask_path",
        prefix=mask_prefix,
        suffix=mask_suffix,
    )
    contour = _safe_logical_path(
        value.get("contour_path"),
        description=description + ".contour_path",
        prefix=contour_prefix,
        suffix=contour_suffix,
    )
    return {
        "token": token,
        "lineage": lineage,
        "parent_group_id": parent_group,
        "model_mask_path": mask,
        "contour_path": contour,
    }


def _ordered_pair_ids_sha256(split: str, pair_ids: Sequence[str]) -> str:
    # This is byte-compatible with the ShreddingNet Rachel adapter.
    return _canonical_sha256(
        {
            "schema_version": "rachel-shreddingnet-ordered-pair-ids/1.0",
            "split": split,
            "ordered_pair_ids": list(pair_ids),
        }
    )


def _canonical_ordered_pairs_sha256(
    split: str, rows: Sequence[Mapping[str, object]]
) -> str:
    canonical_ids = []
    for row in rows:
        canonical_ids.append(
            _canonical_sha256(
                {
                    "split": split,
                    "pair_id": row["pair_id"],
                    "label": row["label"],
                    "fragment_a_token": row["fragment_a_token"],
                    "fragment_b_token": row["fragment_b_token"],
                }
            )
        )
    return _canonical_sha256(
        {
            "schema_version": "rachel-canonical-ordered-pairs/1.0",
            "split": split,
            "canonical_ordered_pair_ids": canonical_ids,
        }
    )


def _parse_manifest(
    *,
    split: str,
    payload: bytes,
    collector: _AssetCollector,
    globally_seen_pair_ids: Set[str],
    fragment_bindings: MutableMapping[str, Tuple[str, str, str, str]],
    mask_path_owners: MutableMapping[str, str],
    contour_path_owners: MutableMapping[str, str],
) -> Tuple[List[Dict[str, object]], Dict[str, int], Set[str]]:
    if split not in _MANIFEST_PATHS:
        _fail("only train and val manifests are authorized")
    rows: List[Dict[str, object]] = []
    pair_ids: Set[str] = set()
    lineages: Set[str] = set()
    origins: Counter = Counter()
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise RachelTrainValAssetFreezeError(
            split + " manifest is not UTF-8"
        ) from error
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_duplicate_rejecting_object,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as error:
            raise RachelTrainValAssetFreezeError(
                "invalid {} manifest JSON line {}".format(split, line_number)
            ) from error
        if not isinstance(value, Mapping):
            _fail("{} manifest line {} is not an object".format(split, line_number))
        _reject_leakage_keys(value, "{} line {}".format(split, line_number))
        if value.get("split") != split:
            _fail("manifest row split differs from authorized split")
        pair_id = _required_text(value.get("pair_id"), "pair_id")
        if pair_id in pair_ids or pair_id in globally_seen_pair_ids:
            _fail("duplicate train/val pair_id: " + pair_id)
        pair_ids.add(pair_id)
        globally_seen_pair_ids.add(pair_id)
        label = value.get("label")
        if type(label) is not bool:  # noqa: E721
            _fail("manifest label must be an explicit bool")
        first = _fragment_fields(
            value.get("fragment_a"),
            "{} line {} fragment_a".format(split, line_number),
        )
        second = _fragment_fields(
            value.get("fragment_b"),
            "{} line {} fragment_b".format(split, line_number),
        )
        if first["token"] == second["token"]:
            _fail("self-pair is forbidden: " + pair_id)
        if (
            value.get("fragment_a_token") != first["token"]
            or value.get("fragment_b_token") != second["token"]
        ):
            _fail("top-level and nested fragment tokens disagree")
        if value.get("main_training_eligible") is not True:
            _fail("selected manifest row is not marked training eligible")
        if value.get("selection_exclusion_reason") is not None:
            _fail("selected manifest row carries an exclusion reason")
        for fragment in (first, second):
            binding = (
                fragment["lineage"],
                fragment["parent_group_id"],
                fragment["model_mask_path"],
                fragment["contour_path"],
            )
            previous = fragment_bindings.get(fragment["token"])
            if previous is not None and previous != binding:
                _fail("fragment token has inconsistent lineage/assets: " + fragment["token"])
            fragment_bindings[fragment["token"]] = binding
            for path_key, owners, class_name in (
                ("model_mask_path", mask_path_owners, "model mask"),
                ("contour_path", contour_path_owners, "contour"),
            ):
                relative = fragment[path_key]
                owner = owners.get(relative)
                if owner is not None and owner != fragment["token"]:
                    _fail(
                        "distinct fragment tokens alias one {} path: {} and {}".format(
                            class_name, owner, fragment["token"]
                        )
                    )
                owners[relative] = fragment["token"]
            lineages.add(fragment["lineage"])
            collector.add(fragment["model_mask_path"], "model_mask")
            collector.add(fragment["contour_path"], "contour_n512")
        target = value.get("correspondence_path")
        if label:
            target_prefix, target_suffix = _ASSET_CLASSES[
                "positive_correspondence_target"
            ]
            target_path = _safe_logical_path(
                target,
                description="positive correspondence_path",
                prefix=target_prefix,
                suffix=target_suffix,
            )
            collector.add(target_path, "positive_correspondence_target")
            if value.get("negative_origin") is not None:
                _fail("positive pair carries negative_origin")
            translation = value.get("translation_a_to_b_rc")
            if (
                not isinstance(translation, list)
                or len(translation) != 2
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in translation
                )
            ):
                _fail("positive pair lacks a finite two-value translation target")
            if (
                first["lineage"] != second["lineage"]
                or first["parent_group_id"] != second["parent_group_id"]
            ):
                _fail("positive pair crosses its Rachel source group")
            origins["positive"] += 1
        else:
            if target is not None:
                _fail("negative pair references correspondence GT")
            for translation_key in (
                "translation_a_to_b_rc",
                "translation_a_to_b_xy_cartesian",
            ):
                if value.get(translation_key) is not None:
                    _fail("negative pair carries translation target")
            negative_origin = value.get("negative_origin")
            if negative_origin not in {
                "same_folder_hard",
                "cross_folder_scale_matched",
            }:
                _fail("negative pair has an unknown negative_origin")
            if negative_origin == "same_folder_hard" and (
                first["lineage"] != second["lineage"]
                or first["parent_group_id"] != second["parent_group_id"]
            ):
                _fail("same-folder hard negative crosses its Rachel source group")
            if negative_origin == "cross_folder_scale_matched" and (
                first["lineage"] == second["lineage"]
                or first["parent_group_id"] == second["parent_group_id"]
            ):
                _fail("cross-folder negative does not cross source lineages/groups")
            origins[str(negative_origin)] += 1
        _required_text(value.get("label_origin"), "label_origin")
        rows.append(
            {
                "pair_id": pair_id,
                "label": label,
                "fragment_a_token": first["token"],
                "fragment_b_token": second["token"],
            }
        )
    if not rows:
        _fail(split + " manifest is empty")
    counts = {
        "rows": len(rows),
        "positive": int(origins["positive"]),
        "negative": int(
            origins["same_folder_hard"]
            + origins["cross_folder_scale_matched"]
        ),
        "same_folder_hard": int(origins["same_folder_hard"]),
        "cross_folder_scale_matched": int(origins["cross_folder_scale_matched"]),
    }
    if counts["positive"] != counts["negative"]:
        _fail(split + " manifest is not exact 1:1 positive/negative")
    return rows, counts, lineages


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(description + " must be an object")
    return value


def _validate_authority_cross_binding(
    *,
    run_config: object,
    preprocess_receipt: object,
    preprocess_summary: object,
    selection_summary: object,
    lineage_assignments: object,
    manifest_counts: Mapping[str, Mapping[str, int]],
    manifest_lineages: Mapping[str, Set[str]],
    require_formal_counts: bool,
) -> Dict[str, object]:
    run = _mapping(run_config, "preprocess run config")
    receipt = _mapping(preprocess_receipt, "preprocess receipt")
    summary = _mapping(preprocess_summary, "preprocess summary")
    selection = _mapping(selection_summary, "selection summary")
    assignments = _mapping(lineage_assignments, "selection lineage assignments")
    if set(receipt) != {
        "schema_version",
        "status",
        "source_authority",
        "shredding_data_read",
        "preprocess",
        "selection",
        "paths",
    }:
        _fail("preprocess receipt fields differ from schema 1.0")
    if set(run) != {
        "schema_version",
        "source_root",
        "output_root",
        "seed",
        "generators",
        "preprocess_config",
        "limit_groups_per_generator",
        "split_unit",
        "model_input_contract",
    }:
        _fail("preprocess run-config fields differ from schema 1.0")
    if set(summary) != {
        "group_counts",
        "fragment_count",
        "fragment_counts_by_generator",
        "candidate_count",
        "candidate_counts_by_generator_label_eligibility",
        "positive_seam_length_px_quantiles",
        "contour_n512",
    }:
        _fail("preprocess summary fields differ from schema 1.0")
    if set(selection) != {
        "schema_version",
        "seed",
        "total_pairs",
        "split_quotas",
        "lineage_counts",
        "selected_counts",
    }:
        _fail("selection summary fields differ from schema 1.0")
    if receipt.get("schema_version") != PREPROCESS_SCHEMA_VERSION:
        _fail("preprocess receipt schema differs")
    if receipt.get("status") != PREPROCESS_STATUS:
        _fail("preprocess receipt is not a complete 30k release")
    if receipt.get("source_authority") != "Rachel RGB JPEG plus colocated label.csv only":
        _fail("preprocess source authority differs")
    if receipt.get("shredding_data_read") is not False:
        _fail("preprocess receipt does not prove independent Rachel processing")
    if receipt.get("preprocess") != summary:
        _fail("preprocess summary is not exactly embedded in receipt")
    if receipt.get("selection") != selection:
        _fail("selection summary is not exactly embedded in receipt")
    if receipt.get("paths") != _EXPECTED_RECEIPT_PATHS:
        _fail("preprocess receipt path authority differs")
    if run.get("schema_version") != PREPROCESS_SCHEMA_VERSION:
        _fail("preprocess run-config schema differs")
    if run.get("seed") != selection.get("seed"):
        _fail("run-config and selection seed differ")
    if run.get("limit_groups_per_generator") is not None:
        _fail("pilot-limited preprocessing cannot back the formal release")
    model_contract = _mapping(run.get("model_input_contract"), "model input contract")
    if set(model_contract) != {
        "rgb_used",
        "binary_values_in_memory",
        "binary_values_on_disk",
        "canvas_shape",
        "parent_origin_exposed_to_model",
        "contour",
    }:
        _fail("preprocess model-input contract fields differ")
    if (
        model_contract.get("rgb_used") is not False
        or model_contract.get("binary_values_in_memory") != [0, 1]
        or model_contract.get("binary_values_on_disk") != [0, 255]
        or model_contract.get("canvas_shape") != [800, 800]
        or model_contract.get("parent_origin_exposed_to_model") is not False
    ):
        _fail("preprocess model-input contract differs")
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        _fail("selection summary schema differs")
    if not isinstance(selection.get("seed"), str) or not selection["seed"]:
        _fail("selection seed is missing")
    if require_formal_counts and selection.get("seed") != SELECTION_SEED:
        _fail("formal selection seed differs")
    if require_formal_counts:
        if selection.get("total_pairs") != 30_000:
            _fail("formal selection total differs")
        if selection.get("selected_counts") != FORMAL_SELECTED_COUNTS:
            _fail("formal selected-count authority differs")
        if selection.get("split_quotas") != FORMAL_SPLIT_QUOTAS:
            _fail("formal split-quota authority differs")
    selected_counts = _mapping(selection.get("selected_counts"), "selected counts")
    for split in ("train", "val"):
        declared = _mapping(selected_counts.get(split), split + " selected counts")
        observed = manifest_counts[split]
        expected = {
            "positive": observed["positive"],
            "same_folder_hard": observed["same_folder_hard"],
            "cross_folder_scale_matched": observed[
                "cross_folder_scale_matched"
            ],
        }
        if dict(declared) != expected:
            _fail(split + " manifest counts disagree with selection summary")
    assignment_counts: Counter = Counter()
    for lineage, split_value in assignments.items():
        _required_text(lineage, "lineage assignment key")
        if split_value not in {"train", "val", "test"}:
            _fail("lineage assignment has an unknown split")
        assignment_counts[str(split_value)] += 1
    for split in ("train", "val"):
        for lineage in manifest_lineages[split]:
            if assignments.get(lineage) != split:
                _fail(split + " manifest lineage disagrees with lineage authority")
    if manifest_lineages["train"] & manifest_lineages["val"]:
        _fail("train/validation split_unit_id leakage")
    declared_lineage_counts = _mapping(
        selection.get("lineage_counts"), "selection lineage counts"
    )
    if dict(declared_lineage_counts) != {
        split: int(assignment_counts[split]) for split in ("train", "val", "test")
    }:
        _fail("lineage assignments disagree with selection summary")
    return {
        "preprocess_receipt_schema_version": PREPROCESS_SCHEMA_VERSION,
        "preprocess_receipt_status": PREPROCESS_STATUS,
        "selection_summary_schema_version": SELECTION_SCHEMA_VERSION,
        "selection_seed": str(selection["seed"]),
        "preprocess_summary_exactly_embedded_in_receipt": True,
        "selection_summary_exactly_embedded_in_receipt": True,
        "run_config_seed_matches_selection": True,
        "lineage_assignments_match_train_val_manifests": True,
        "lineage_assignment_counts": {
            split: int(assignment_counts[split])
            for split in ("train", "val", "test")
        },
        "source_authority": str(receipt["source_authority"]),
        "shredding_data_read": False,
    }


def _role_counts(assets: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    counts: Counter = Counter()
    for row in assets:
        roles = row.get("roles")
        if not isinstance(roles, list):
            _fail("internal asset role inventory is malformed")
        for role in roles:
            counts[str(role)] += 1
    return dict(sorted(counts.items()))


def _build_inventory_with_collector(
    collector: _AssetCollector, require_formal_counts: bool
) -> Dict[str, object]:
    authority_values: Dict[str, object] = {}
    for role, relative in _AUTHORITY_PATHS.items():
        if relative not in _ALLOWED_DATASET_STATIC_PATHS:
            _fail("internal authority allowlist violation")
        payload = collector.add(relative, role, retain_bytes=True)
        assert payload is not None
        authority_values[role] = _strict_json_bytes(payload, role)
    manifest_payloads: Dict[str, bytes] = {}
    for split, relative in _MANIFEST_PATHS.items():
        if relative not in _ALLOWED_DATASET_STATIC_PATHS:
            _fail("internal manifest allowlist violation")
        payload = collector.add(
            relative, split + "_pair_manifest", retain_bytes=True
        )
        assert payload is not None
        manifest_payloads[split] = payload
    globally_seen_pair_ids: Set[str] = set()
    fragment_bindings: Dict[str, Tuple[str, str, str, str]] = {}
    mask_path_owners: Dict[str, str] = {}
    contour_path_owners: Dict[str, str] = {}
    manifest_rows: Dict[str, List[Dict[str, object]]] = {}
    manifest_counts: Dict[str, Dict[str, int]] = {}
    manifest_lineages: Dict[str, Set[str]] = {}
    for split in ("train", "val"):
        rows, counts, lineages = _parse_manifest(
            split=split,
            payload=manifest_payloads[split],
            collector=collector,
            globally_seen_pair_ids=globally_seen_pair_ids,
            fragment_bindings=fragment_bindings,
            mask_path_owners=mask_path_owners,
            contour_path_owners=contour_path_owners,
        )
        manifest_rows[split] = rows
        manifest_counts[split] = counts
        manifest_lineages[split] = lineages
    if manifest_lineages["train"] & manifest_lineages["val"]:
        _fail("train/validation split_unit_id leakage")
    if require_formal_counts:
        observed_formal = {
            split: {
                key: manifest_counts[split][key]
                for key in ("rows", "positive", "negative")
            }
            for split in ("train", "val")
        }
        if observed_formal != FORMAL_COUNTS:
            _fail("formal Rachel population must be exactly 24000/3000 and 1:1")
    cross_binding = _validate_authority_cross_binding(
        run_config=authority_values["preprocess_run_config"],
        preprocess_receipt=authority_values["preprocess_receipt"],
        preprocess_summary=authority_values["preprocess_summary"],
        selection_summary=authority_values["selection_summary"],
        lineage_assignments=authority_values["selection_lineage_assignments"],
        manifest_counts=manifest_counts,
        manifest_lineages=manifest_lineages,
        require_formal_counts=require_formal_counts,
    )
    assets = collector.inventory()
    manifest_content_sha256 = {
        split: next(
            str(row["sha256"])
            for row in assets
            if row["relative_path"] == _MANIFEST_PATHS[split]
        )
        for split in ("train", "val")
    }
    ordered_pair_ids_sha256 = {
        split: _ordered_pair_ids_sha256(
            split, [str(row["pair_id"]) for row in manifest_rows[split]]
        )
        for split in ("train", "val")
    }
    canonical_ordered_pairs_sha256 = {
        split: _canonical_ordered_pairs_sha256(split, manifest_rows[split])
        for split in ("train", "val")
    }
    semantic = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "status": "complete_train_val_semantic_asset_inventory",
        "dataset_scope": {
            "opened_pair_manifests": [
                _MANIFEST_PATHS["train"],
                _MANIFEST_PATHS["val"],
            ],
            "opened_authority_json": [
                _AUTHORITY_PATHS[key] for key in sorted(_AUTHORITY_PATHS)
            ],
            "sealed_test_manifest_opened": False,
            "sealed_test_assets_opened": False,
            "real_data_opened": False,
            "rgb_opened": False,
        },
        "formal_counts_required": require_formal_counts,
        "counts": {
            split: dict(manifest_counts[split]) for split in ("train", "val")
        },
        "lineage_counts": {
            split: len(manifest_lineages[split]) for split in ("train", "val")
        },
        "lineage_disjoint": True,
        "manifest_content_sha256": manifest_content_sha256,
        "ordered_pair_ids_sha256": ordered_pair_ids_sha256,
        "canonical_ordered_pairs_sha256": canonical_ordered_pairs_sha256,
        "authority_cross_binding": cross_binding,
        "asset_file_count": len(assets),
        "asset_total_size_bytes": sum(int(row["size_bytes"]) for row in assets),
        "unique_files_by_role": _role_counts(assets),
        "assets": assets,
    }
    return {
        "semantic_inventory": semantic,
        "semantic_inventory_content_sha256": _canonical_sha256(semantic),
    }


def build_train_val_semantic_inventory(
    dataset_root: Union[str, Path], *, require_formal_counts: bool = True
) -> Dict[str, object]:
    """Build an in-memory inventory using one anchored dataset-root FD.

    Only ``pairs/train.jsonl``, ``pairs/val.jsonl``, their referenced training
    assets, and ``_AUTHORITY_PATHS`` are opened.  Every component is traversed
    with ``openat``/``O_NOFOLLOW`` and revalidated after hashing.
    """

    if type(require_formal_counts) is not bool:  # noqa: E721
        raise TypeError("require_formal_counts must be bool")
    root = _validated_dataset_root(dataset_root)
    with _AssetCollector(root) as collector:
        result = _build_inventory_with_collector(collector, require_formal_counts)
        collector.anchor.revalidate()
        return result


class _AnchoredFreshOutput:
    """Fresh output directory published only through anchored directory FDs."""

    def __init__(self, value: Union[str, Path]) -> None:
        self.path = _lexical_absolute(value)
        if self.path.name in {"", ".", ".."}:
            _fail("freeze output directory name is unsafe")
        self.parent_path = self.path.parent
        self.parent_descriptor, self._parent_chain = _open_absolute_directory_chain(
            self.parent_path, "freeze output parent"
        )
        self._parent_identity = _entry_identity(os.fstat(self.parent_descriptor))
        self.descriptor = -1
        try:
            try:
                os.stat(
                    self.path.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                _fail("freeze output directory already exists; overwrite is forbidden")
            os.mkdir(self.path.name, 0o755, dir_fd=self.parent_descriptor)
            entry = os.stat(
                self.path.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(entry.st_mode) or stat.S_ISLNK(entry.st_mode):
                _fail("fresh freeze output entry is not a real directory")
            self.descriptor = os.open(
                self.path.name,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=self.parent_descriptor,
            )
            opened = os.fstat(self.descriptor)
            if _entry_identity(entry) != _entry_identity(opened):
                _fail("freeze output entry changed between mkdir and open")
            self._output_identity = _entry_identity(opened)
            _race_test_hook(
                "output_after_directory_anchor_before_publication",
                {"output_dir": self.path},
            )
            self.revalidate()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1
        if self.parent_descriptor >= 0:
            os.close(self.parent_descriptor)
            self.parent_descriptor = -1

    def __enter__(self) -> "_AnchoredFreshOutput":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def revalidate(self) -> None:
        if self.descriptor < 0 or self.parent_descriptor < 0:
            _fail("freeze output anchor is closed")
        if _entry_identity(os.fstat(self.parent_descriptor)) != self._parent_identity:
            _fail("freeze output parent descriptor identity changed")
        _revalidate_absolute_directory_chain(
            self.parent_path,
            self._parent_chain,
            "freeze output parent",
        )
        try:
            entry = os.stat(
                self.path.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise RachelTrainValAssetFreezeError(
                "freeze output directory was removed or replaced"
            ) from error
        if (
            stat.S_ISLNK(entry.st_mode)
            or not stat.S_ISDIR(entry.st_mode)
            or _entry_identity(entry) != self._output_identity
            or _entry_identity(os.fstat(self.descriptor)) != self._output_identity
        ):
            _fail("freeze output directory was renamed, replaced, or symlinked")

    def open_directory(
        self, parts: Sequence[str], *, create: bool
    ) -> Tuple[int, Tuple[Tuple[int, int, int], ...]]:
        self.revalidate()
        descriptor = os.dup(self.descriptor)
        identities: List[Tuple[int, int, int]] = []
        try:
            for component in parts:
                if component in {"", ".", ".."} or "/" in component:
                    _fail("output relative directory component is unsafe")
                if create:
                    try:
                        os.mkdir(component, 0o755, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                entry = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
                    _fail("output subtree contains an unsafe directory entry")
                child = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
                opened = os.fstat(child)
                if _entry_identity(entry) != _entry_identity(opened):
                    os.close(child)
                    _fail("output subtree directory identity changed")
                os.close(descriptor)
                descriptor = child
                identities.append(_entry_identity(opened))
            self.revalidate()
            return descriptor, tuple(identities)
        except BaseException:
            os.close(descriptor)
            raise

    def write_file(
        self, relative: str, payload: bytes, *, mode: int = 0o444
    ) -> Dict[str, object]:
        logical = _safe_logical_path(relative, description="freeze output file")
        parts = PurePosixPath(logical).parts
        parent_descriptor, parent_chain = self.open_directory(
            parts[:-1], create=True
        )
        leaf = parts[-1]
        temporary = "." + leaf + ".tmp-" + uuid.uuid4().hex
        descriptor = -1
        try:
            self.revalidate()
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=parent_descriptor,
            )
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    _fail("short write while materializing frozen dataset")
                offset += written
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            written_metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(written_metadata.st_mode)
                or int(written_metadata.st_size) != len(payload)
            ):
                _fail("published temporary file identity/size differs")
            os.close(descriptor)
            descriptor = -1
            _race_test_hook(
                "output_after_temp_fsync_before_link",
                {"output_dir": self.path, "relative_path": logical},
            )
            self.revalidate()
            reopened_parent, observed_chain = self.open_directory(
                parts[:-1], create=False
            )
            os.close(reopened_parent)
            if observed_chain != parent_chain:
                _fail("output file parent was renamed or replaced")
            os.link(
                temporary,
                leaf,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            final_entry = os.stat(
                leaf,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            temporary_entry = os.stat(
                temporary,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _entry_identity(final_entry) != _entry_identity(temporary_entry)
                or stat.S_IMODE(final_entry.st_mode) != mode
            ):
                _fail("atomic output link identity or mode differs")
            os.unlink(temporary, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            self.revalidate()
            digest = hashlib.sha256(payload).hexdigest()
            return {
                "sha256": digest,
                "size_bytes": len(payload),
                "device": int(final_entry.st_dev),
                "inode": int(final_entry.st_ino),
            }
        except FileExistsError as error:
            raise RachelTrainValAssetFreezeError(
                "freeze artifact already exists; overwrite is forbidden"
            ) from error
        except OSError as error:
            raise RachelTrainValAssetFreezeError(
                "cannot atomically publish freeze artifact through anchored FDs"
            ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            os.close(parent_descriptor)

    def seal_directories(self, relative_directories: Iterable[str]) -> None:
        directories = sorted(
            set(relative_directories),
            key=lambda value: (len(PurePosixPath(value).parts), value),
            reverse=True,
        )
        for relative in directories:
            logical = _safe_logical_path(relative, description="freeze directory")
            descriptor, _ = self.open_directory(
                PurePosixPath(logical).parts, create=False
            )
            try:
                os.fchmod(descriptor, 0o555)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        self.revalidate()

    def seal_output_root(self) -> None:
        self.revalidate()
        os.fchmod(self.descriptor, 0o555)
        os.fsync(self.descriptor)
        os.fsync(self.parent_descriptor)
        self.revalidate()


def _json_file_bytes(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _canonicalize_selected_manifest(payload: bytes, split: str) -> bytes:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise RachelTrainValAssetFreezeError(
            split + " manifest cannot be rewritten as UTF-8"
        ) from error
    rows: List[bytes] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = _strict_json_bytes(
            line.encode("utf-8"),
            "{} manifest line {}".format(split, line_number),
        )
        if not isinstance(value, Mapping) or value.get("split") != split:
            _fail("rewritten manifest row split differs")
        rows.append(_canonical_bytes(value) + b"\n")
    if not rows:
        _fail("cannot rewrite an empty " + split + " manifest")
    return b"".join(rows)


def _materialize_frozen_dataset_view(
    *,
    collector: _AssetCollector,
    source_inventory: Mapping[str, object],
    output: _AnchoredFreshOutput,
) -> Dict[str, object]:
    semantic = _mapping(
        source_inventory.get("semantic_inventory"), "source semantic inventory"
    )
    source_assets = semantic.get("assets")
    if not isinstance(source_assets, list) or not source_assets:
        _fail("source semantic asset list is missing")
    frozen_root_descriptor, _ = output.open_directory(
        (FROZEN_DATA_DIRNAME,), create=True
    )
    frozen_root_metadata = os.fstat(frozen_root_descriptor)
    os.close(frozen_root_descriptor)
    directories: Set[str] = {FROZEN_DATA_DIRNAME}
    frozen_files: List[Dict[str, object]] = []
    destination_identities: Set[Tuple[int, int]] = set()
    for source_row_value in source_assets:
        source_row = _mapping(source_row_value, "source asset row")
        if set(source_row) != {
            "relative_path",
            "roles",
            "reference_count",
            "sha256",
            "size_bytes",
        }:
            _fail("source asset row fields differ before materialization")
        relative = _safe_logical_path(
            source_row.get("relative_path"), description="source asset"
        )
        payload, observed = collector.anchor.read(
            relative, "frozen view source " + relative, retain_bytes=True
        )
        assert payload is not None
        if (
            observed["sha256"] != source_row.get("sha256")
            or observed["size_bytes"] != source_row.get("size_bytes")
        ):
            _fail("source asset changed between inventory and frozen-view copy")
        materialization = "byte_copy"
        view_payload = payload
        for split, manifest_relative in _MANIFEST_PATHS.items():
            if relative == manifest_relative:
                view_payload = _canonicalize_selected_manifest(payload, split)
                materialization = "strict_parse_canonical_manifest_rewrite"
                break
        destination_relative = FROZEN_DATA_DIRNAME + "/" + relative
        destination = output.write_file(destination_relative, view_payload, mode=0o444)
        source_physical = (int(observed["device"]), int(observed["inode"]))
        destination_physical = (
            int(destination["device"]),
            int(destination["inode"]),
        )
        if destination_physical == source_physical:
            _fail("frozen view unexpectedly hard-links a source asset")
        if destination_physical in destination_identities:
            _fail("two frozen-view paths alias one physical file")
        destination_identities.add(destination_physical)
        parent = PurePosixPath(destination_relative).parent
        while parent.as_posix() not in {".", ""}:
            directories.add(parent.as_posix())
            parent = parent.parent
        roles = source_row.get("roles")
        if not isinstance(roles, list):
            _fail("source asset roles are malformed")
        frozen_files.append(
            {
                "relative_path": relative,
                "roles": list(roles),
                "reference_count": int(source_row["reference_count"]),
                "source_sha256": str(source_row["sha256"]),
                "source_size_bytes": int(source_row["size_bytes"]),
                "source_device": int(observed["device"]),
                "source_inode": int(observed["inode"]),
                "sha256": str(destination["sha256"]),
                "size_bytes": int(destination["size_bytes"]),
                "device": int(destination["device"]),
                "inode": int(destination["inode"]),
                "materialization": materialization,
                "file_mode_octal": "0444",
            }
        )
    output.seal_directories(directories)
    frozen_by_path = {str(row["relative_path"]): row for row in frozen_files}
    manifest_hashes = {
        split: str(frozen_by_path[path]["sha256"])
        for split, path in _MANIFEST_PATHS.items()
    }
    return {
        "schema_version": FROZEN_VIEW_SCHEMA_VERSION,
        "status": "complete_self_contained_read_only_train_val_view",
        "relative_root": FROZEN_DATA_DIRNAME,
        "root_device": int(frozen_root_metadata.st_dev),
        "root_inode": int(frozen_root_metadata.st_ino),
        "runner_dataset_root_must_equal_this_view": True,
        "source_hardlinks_used": False,
        "materialization": "independent_byte_copy_via_anchored_source_and_output_fds",
        "selected_manifests_rewritten": True,
        "selected_manifest_rewrite": "strict_parse_canonical_jsonl_same_relative_asset_paths",
        "contains_test_manifest": False,
        "contains_test_assets": False,
        "contains_real_or_rgb_data": False,
        "file_mode_octal": "0444",
        "directory_mode_octal": "0555",
        "file_count": len(frozen_files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in frozen_files),
        "manifest_content_sha256": manifest_hashes,
        "ordered_pair_ids_sha256": dict(semantic["ordered_pair_ids_sha256"]),
        "canonical_ordered_pairs_sha256": dict(
            semantic["canonical_ordered_pairs_sha256"]
        ),
        "files": frozen_files,
    }


def publish_train_val_semantic_asset_freeze(
    dataset_root: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    require_formal_counts: bool = True,
) -> Dict[str, object]:
    """Create a fresh, no-clobber inventory directory and receipt.

    The receipt is published last and is the commit marker.  The directory
    itself is claimed with exclusive ``mkdir``; existing directories, even
    empty ones, are never reused or replaced.
    """

    if type(require_formal_counts) is not bool:  # noqa: E721
        raise TypeError("require_formal_counts must be bool")
    root = _validated_dataset_root(dataset_root)
    output_path = _lexical_absolute(output_dir)
    with _AssetCollector(root) as collector:
        source_inventory = _build_inventory_with_collector(
            collector, require_formal_counts
        )
        with _AnchoredFreshOutput(output_path) as output:
            frozen_view = _materialize_frozen_dataset_view(
                collector=collector,
                source_inventory=source_inventory,
                output=output,
            )
            source_semantic = _mapping(
                source_inventory.get("semantic_inventory"),
                "source semantic inventory",
            )
            semantic = dict(source_semantic)
            semantic["frozen_view"] = frozen_view
            inventory = {
                "semantic_inventory": semantic,
                "semantic_inventory_content_sha256": _canonical_sha256(semantic),
            }
            inventory_payload = _json_file_bytes(inventory)
            inventory_written = output.write_file(
                INVENTORY_FILENAME, inventory_payload, mode=0o444
            )
            freeze_root_metadata = os.fstat(output.descriptor)
            frozen_dataset = {
                "relative_path": FROZEN_DATA_DIRNAME,
                "absolute_path": (output_path / FROZEN_DATA_DIRNAME).as_posix(),
                "runner_dataset_root_required": True,
                "freeze_root_device": int(freeze_root_metadata.st_dev),
                "freeze_root_inode": int(freeze_root_metadata.st_ino),
                "view_root_device": int(frozen_view["root_device"]),
                "view_root_inode": int(frozen_view["root_inode"]),
                "manifest_content_sha256": dict(
                    frozen_view["manifest_content_sha256"]
                ),
                "ordered_pair_ids_sha256": dict(
                    frozen_view["ordered_pair_ids_sha256"]
                ),
                "canonical_ordered_pairs_sha256": dict(
                    frozen_view["canonical_ordered_pairs_sha256"]
                ),
                "file_count": int(frozen_view["file_count"]),
                "total_size_bytes": int(frozen_view["total_size_bytes"]),
                "file_mode_octal": "0444",
                "directory_mode_octal": "0555",
            }
            receipt = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "status": "complete_self_contained_train_val_semantic_asset_freeze",
                "dataset_root": root.as_posix(),
                "frozen_dataset": frozen_dataset,
                "inventory": {
                    "filename": INVENTORY_FILENAME,
                    "sha256": str(inventory_written["sha256"]),
                    "size_bytes": int(inventory_written["size_bytes"]),
                    "semantic_inventory_content_sha256": inventory[
                        "semantic_inventory_content_sha256"
                    ],
                },
                "publication_contract": {
                    "fresh_directory_required": True,
                    "all_output_operations_anchored_by_parent_and_output_dir_fds": True,
                    "atomic_file_publication": (
                        "same_parent_dir_fd_temp_fsync_then_hard_link_no_replace"
                    ),
                    "source_assets_materialized_by_independent_byte_copy": True,
                    "source_hardlinks_used": False,
                    "frozen_view_read_only": True,
                    "overwrite_allowed": False,
                    "receipt_published_last_as_commit_marker": True,
                },
                "scope": {
                    "opened_pair_manifests": [
                        _MANIFEST_PATHS["train"],
                        _MANIFEST_PATHS["val"],
                    ],
                    "frozen_runner_dataset_root": frozen_dataset["absolute_path"],
                    "sealed_test_manifest_opened": False,
                    "sealed_test_assets_opened": False,
                    "real_data_opened": False,
                    "rgb_opened": False,
                },
            }
            receipt_payload = _json_file_bytes(receipt)
            output.write_file(RECEIPT_FILENAME, receipt_payload, mode=0o444)
            output.seal_output_root()
            published_files, published_directories = _walk_anchored_tree(output)
            expected_published_files = {
                INVENTORY_FILENAME,
                RECEIPT_FILENAME,
                *(
                    FROZEN_DATA_DIRNAME + "/" + str(row["relative_path"])
                    for row in frozen_view["files"]
                ),
            }
            if set(published_files) != expected_published_files:
                _fail("fresh freeze output tree changed before create committed")
            published_view_root = published_directories.get(FROZEN_DATA_DIRNAME)
            if published_view_root is None or (
                published_view_root["device"] != frozen_view["root_device"]
                or published_view_root["inode"] != frozen_view["root_inode"]
                or _entry_identity(os.fstat(output.descriptor))
                != _entry_identity(freeze_root_metadata)
            ):
                _fail("fresh freeze root identity changed before create committed")
            collector.anchor.revalidate()
            output.revalidate()
            return receipt


def _walk_anchored_tree(
    root: _AnchoredDatasetRoot,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    files: Dict[str, Dict[str, int]] = {}
    directories: Dict[str, Dict[str, int]] = {}

    def walk(descriptor: int, prefix: Tuple[str, ...]) -> None:
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as error:
            raise RachelTrainValAssetFreezeError(
                "freeze tree directory cannot be enumerated"
            ) from error
        if len(names) != len(set(names)):
            _fail("freeze tree directory reports duplicate entries")
        for name in names:
            if name in {"", ".", ".."} or "/" in name or "\x00" in name:
                _fail("freeze tree contains an unsafe entry name")
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = "/".join(prefix + (name,))
            if stat.S_ISLNK(metadata.st_mode):
                _fail("freeze tree contains a symlink: " + relative)
            row = {
                "device": int(metadata.st_dev),
                "inode": int(metadata.st_ino),
                "link_count": int(metadata.st_nlink),
                "mode": int(stat.S_IMODE(metadata.st_mode)),
                "size_bytes": int(metadata.st_size),
            }
            if stat.S_ISREG(metadata.st_mode):
                if prefix and prefix[0] == FROZEN_DATA_DIRNAME:
                    if int(metadata.st_nlink) != 1:
                        _fail("frozen view regular file has external hard links")
                files[relative] = row
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                _fail("freeze tree contains a special entry: " + relative)
            child = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                if _entry_identity(metadata) != _entry_identity(opened):
                    _fail("freeze tree directory changed while walking")
                directories[relative] = row
                walk(child, prefix + (name,))
                entry_after = os.stat(
                    name, dir_fd=descriptor, follow_symlinks=False
                )
                if _entry_identity(entry_after) != _entry_identity(opened):
                    _fail("freeze tree directory was replaced while walking")
            finally:
                os.close(child)

    walk(root.descriptor, ())
    root.revalidate()
    return files, directories


def _verify_train_val_semantic_asset_freeze_anchored(
    root: Path,
    frozen: Path,
    freeze_anchor: _AnchoredDatasetRoot,
    *,
    require_formal_counts: bool,
    expected_freeze_receipt_sha256: Optional[str] = None,
    expected_semantic_inventory_content_sha256: Optional[str] = None,
) -> Dict[str, object]:
    freeze_root_metadata = os.fstat(freeze_anchor.descriptor)
    freeze_root_identity = _entry_identity(freeze_root_metadata)
    if stat.S_IMODE(freeze_root_metadata.st_mode) != 0o555:
        _fail("freeze directory root is not read-only mode 0555")
    receipt_payload_optional, receipt_digest = freeze_anchor.read(
        RECEIPT_FILENAME, "freeze receipt", retain_bytes=True
    )
    inventory_payload_optional, inventory_digest = freeze_anchor.read(
        INVENTORY_FILENAME, "semantic asset inventory", retain_bytes=True
    )
    assert receipt_payload_optional is not None
    assert inventory_payload_optional is not None
    receipt_payload = receipt_payload_optional
    inventory_payload = inventory_payload_optional
    tree_files, tree_directories = _walk_anchored_tree(freeze_anchor)
    _race_test_hook(
        "freeze_after_initial_tree_walk",
        {"freeze_dir": frozen, "freeze_root_fd": freeze_anchor.descriptor},
    )
    freeze_anchor.revalidate()
    if (
        expected_freeze_receipt_sha256 is not None
        and receipt_digest["sha256"] != expected_freeze_receipt_sha256
    ):
        _fail("freeze receipt differs from the externally pinned SHA-256")
    receipt = _strict_json_bytes(receipt_payload, "freeze receipt")
    inventory = _strict_json_bytes(inventory_payload, "semantic asset inventory")
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_version",
        "status",
        "dataset_root",
        "frozen_dataset",
        "inventory",
        "publication_contract",
        "scope",
    }:
        _fail("freeze receipt fields differ")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status")
        != "complete_self_contained_train_val_semantic_asset_freeze"
        or receipt.get("dataset_root") != root.as_posix()
    ):
        _fail("freeze receipt identity differs")
    if receipt.get("publication_contract") != {
        "fresh_directory_required": True,
        "all_output_operations_anchored_by_parent_and_output_dir_fds": True,
        "atomic_file_publication": (
            "same_parent_dir_fd_temp_fsync_then_hard_link_no_replace"
        ),
        "source_assets_materialized_by_independent_byte_copy": True,
        "source_hardlinks_used": False,
        "frozen_view_read_only": True,
        "overwrite_allowed": False,
        "receipt_published_last_as_commit_marker": True,
    }:
        _fail("freeze receipt publication contract differs")
    frozen_dataset = _mapping(receipt.get("frozen_dataset"), "frozen dataset")
    expected_frozen_root = (frozen / FROZEN_DATA_DIRNAME).as_posix()
    if receipt.get("scope") != {
        "opened_pair_manifests": [
            _MANIFEST_PATHS["train"],
            _MANIFEST_PATHS["val"],
        ],
        "frozen_runner_dataset_root": expected_frozen_root,
        "sealed_test_manifest_opened": False,
        "sealed_test_assets_opened": False,
        "real_data_opened": False,
        "rgb_opened": False,
    }:
        _fail("freeze receipt scope differs")
    inventory_binding = _mapping(receipt.get("inventory"), "receipt inventory binding")
    if set(inventory_binding) != {
        "filename",
        "sha256",
        "size_bytes",
        "semantic_inventory_content_sha256",
    }:
        _fail("receipt inventory binding fields differ")
    if (
        inventory_binding.get("filename") != INVENTORY_FILENAME
        or inventory_binding.get("sha256") != inventory_digest["sha256"]
        or inventory_binding.get("size_bytes") != inventory_digest["size_bytes"]
    ):
        _fail("inventory file bytes differ from freeze receipt")
    if not isinstance(inventory, Mapping) or set(inventory) != {
        "semantic_inventory",
        "semantic_inventory_content_sha256",
    }:
        _fail("semantic inventory envelope fields differ")
    semantic = inventory.get("semantic_inventory")
    semantic_sha = inventory.get("semantic_inventory_content_sha256")
    if not isinstance(semantic, Mapping) or not _is_sha256(semantic_sha):
        _fail("semantic inventory envelope is malformed")
    if _canonical_sha256(semantic) != semantic_sha:
        _fail("semantic inventory content SHA-256 differs")
    if (
        expected_semantic_inventory_content_sha256 is not None
        and semantic_sha != expected_semantic_inventory_content_sha256
    ):
        _fail("semantic inventory differs from the externally pinned SHA-256")
    if inventory_binding.get("semantic_inventory_content_sha256") != semantic_sha:
        _fail("receipt semantic inventory binding differs")
    frozen_view = _mapping(semantic.get("frozen_view"), "frozen view inventory")
    if set(frozen_view) != {
        "schema_version",
        "status",
        "relative_root",
        "root_device",
        "root_inode",
        "runner_dataset_root_must_equal_this_view",
        "source_hardlinks_used",
        "materialization",
        "selected_manifests_rewritten",
        "selected_manifest_rewrite",
        "contains_test_manifest",
        "contains_test_assets",
        "contains_real_or_rgb_data",
        "file_mode_octal",
        "directory_mode_octal",
        "file_count",
        "total_size_bytes",
        "manifest_content_sha256",
        "ordered_pair_ids_sha256",
        "canonical_ordered_pairs_sha256",
        "files",
    }:
        _fail("frozen view inventory fields differ")
    if (
        frozen_view.get("schema_version")
        != FROZEN_VIEW_SCHEMA_VERSION
        or frozen_view.get("status")
        != "complete_self_contained_read_only_train_val_view"
        or frozen_view.get("relative_root") != FROZEN_DATA_DIRNAME
        or frozen_view.get("runner_dataset_root_must_equal_this_view") is not True
        or frozen_view.get("source_hardlinks_used") is not False
        or frozen_view.get("selected_manifests_rewritten") is not True
        or frozen_view.get("contains_test_manifest") is not False
        or frozen_view.get("contains_test_assets") is not False
        or frozen_view.get("contains_real_or_rgb_data") is not False
        or frozen_view.get("file_mode_octal") != "0444"
        or frozen_view.get("directory_mode_octal") != "0555"
    ):
        _fail("frozen view identity/scope differs")
    observed_view_root = tree_directories.get(FROZEN_DATA_DIRNAME)
    if observed_view_root is None or (
        observed_view_root["device"] != frozen_view.get("root_device")
        or observed_view_root["inode"] != frozen_view.get("root_inode")
    ):
        _fail("frozen view root physical identity differs")
    frozen_file_rows = frozen_view.get("files")
    if not isinstance(frozen_file_rows, list) or not frozen_file_rows:
        _fail("frozen view file inventory is missing")
    expected_tree_files = {INVENTORY_FILENAME, RECEIPT_FILENAME}
    expected_tree_directories: Set[str] = {FROZEN_DATA_DIRNAME}
    frozen_rows_by_path: Dict[str, Mapping[str, object]] = {}
    frozen_physical: Set[Tuple[int, int]] = set()
    for row_value in frozen_file_rows:
        row = _mapping(row_value, "frozen view file row")
        if set(row) != {
            "relative_path",
            "roles",
            "reference_count",
            "source_sha256",
            "source_size_bytes",
            "source_device",
            "source_inode",
            "sha256",
            "size_bytes",
            "device",
            "inode",
            "materialization",
            "file_mode_octal",
        }:
            _fail("frozen view file row fields differ")
        relative = _safe_logical_path(
            row.get("relative_path"), description="frozen view file"
        )
        if relative in frozen_rows_by_path:
            _fail("duplicate frozen view relative path")
        frozen_rows_by_path[relative] = row
        tree_relative = FROZEN_DATA_DIRNAME + "/" + relative
        expected_tree_files.add(tree_relative)
        parent = PurePosixPath(tree_relative).parent
        while parent.as_posix() not in {".", ""}:
            expected_tree_directories.add(parent.as_posix())
            parent = parent.parent
        observed_tree = tree_files.get(tree_relative)
        if observed_tree is None:
            _fail("frozen view file is missing from output tree")
        if (
            observed_tree["mode"] != 0o444
            or observed_tree["link_count"] != 1
            or observed_tree["size_bytes"] != row.get("size_bytes")
            or observed_tree["device"] != row.get("device")
            or observed_tree["inode"] != row.get("inode")
            or row.get("file_mode_octal") != "0444"
            or not _is_sha256(row.get("sha256"))
            or not _is_sha256(row.get("source_sha256"))
        ):
            _fail("frozen view file mode/size/identity differs")
        destination_identity = (
            int(observed_tree["device"]),
            int(observed_tree["inode"]),
        )
        source_identity = (int(row["source_device"]), int(row["source_inode"]))
        if destination_identity == source_identity:
            _fail("frozen view file is hard-linked to its source")
        if destination_identity in frozen_physical:
            _fail("frozen view files contain a physical alias")
        frozen_physical.add(destination_identity)
    if set(tree_files) != expected_tree_files:
        _fail("freeze output file tree differs from exact inventory")
    if set(tree_directories) != expected_tree_directories:
        _fail("freeze output directory tree differs from exact inventory")
    if any(row["mode"] != 0o555 for row in tree_directories.values()):
        _fail("frozen view contains a writable directory")
    if (
        tree_files[INVENTORY_FILENAME]["mode"] != 0o444
        or tree_files[RECEIPT_FILENAME]["mode"] != 0o444
    ):
        _fail("freeze receipt/inventory files are not read-only")
    if frozen_view.get("file_count") != len(frozen_rows_by_path) or frozen_view.get(
        "total_size_bytes"
    ) != sum(int(row["size_bytes"]) for row in frozen_rows_by_path.values()):
        _fail("frozen view aggregate file counts/sizes differ")
    source_semantic = dict(semantic)
    source_semantic.pop("frozen_view", None)
    with _AssetCollector(root) as source_collector:
        current_source = _build_inventory_with_collector(
            source_collector, require_formal_counts
        )
        if current_source.get("semantic_inventory") != source_semantic:
            _fail("Rachel train/val semantic assets changed after freeze")
        for relative, row in frozen_rows_by_path.items():
            _, observed = source_collector.anchor.read(
                relative, "source revalidation " + relative, retain_bytes=False
            )
            if (
                observed["sha256"] != row.get("source_sha256")
                or observed["size_bytes"] != row.get("source_size_bytes")
                or observed["device"] != row.get("source_device")
                or observed["inode"] != row.get("source_inode")
            ):
                _fail("source asset physical/content identity changed after freeze")
        source_collector.anchor.revalidate()
    frozen_dataset_root = frozen / FROZEN_DATA_DIRNAME
    with _AnchoredDatasetRoot.from_anchored_parent(
        freeze_anchor,
        FROZEN_DATA_DIRNAME,
        "frozen dataset view root",
    ) as frozen_view_anchor:
        view_root_metadata = os.fstat(frozen_view_anchor.descriptor)
        view_root_identity = _entry_identity(view_root_metadata)
        if (
            int(view_root_metadata.st_dev) != frozen_view.get("root_device")
            or int(view_root_metadata.st_ino) != frozen_view.get("root_inode")
        ):
            _fail("anchored frozen view root physical identity differs")
        with _AssetCollector(
            frozen_dataset_root, anchor=frozen_view_anchor
        ) as view_collector:
            current_view = _build_inventory_with_collector(
                view_collector, require_formal_counts
            )
            for relative, row in frozen_rows_by_path.items():
                _, observed = view_collector.anchor.read(
                    relative,
                    "frozen view revalidation " + relative,
                    retain_bytes=False,
                )
                if (
                    observed["sha256"] != row.get("sha256")
                    or observed["size_bytes"] != row.get("size_bytes")
                    or observed["device"] != row.get("device")
                    or observed["inode"] != row.get("inode")
                ):
                    _fail(
                        "frozen view physical/content identity changed after freeze"
                    )
            view_collector.anchor.revalidate()
        if _entry_identity(os.fstat(frozen_view_anchor.descriptor)) != view_root_identity:
            _fail("frozen view root descriptor identity changed")
        frozen_view_anchor.revalidate()
    current_view_semantic = _mapping(
        current_view.get("semantic_inventory"), "current frozen view semantic inventory"
    )
    current_view_assets = current_view_semantic.get("assets")
    if not isinstance(current_view_assets, list):
        _fail("current frozen view asset list is missing")
    for current_value in current_view_assets:
        current_row = _mapping(current_value, "current frozen view asset")
        relative = str(current_row.get("relative_path"))
        expected_row = frozen_rows_by_path.get(relative)
        if expected_row is None or (
            current_row.get("roles") != expected_row.get("roles")
            or current_row.get("reference_count")
            != expected_row.get("reference_count")
            or current_row.get("sha256") != expected_row.get("sha256")
            or current_row.get("size_bytes") != expected_row.get("size_bytes")
        ):
            _fail("current frozen view semantic assets differ from inventory")
    if len(current_view_assets) != len(frozen_rows_by_path):
        _fail("current frozen view asset count differs")
    for key in (
        "manifest_content_sha256",
        "ordered_pair_ids_sha256",
        "canonical_ordered_pairs_sha256",
    ):
        if current_view_semantic.get(key) != frozen_view.get(key):
            _fail("frozen view " + key + " differs")
    expected_frozen_dataset = {
        "relative_path": FROZEN_DATA_DIRNAME,
        "absolute_path": expected_frozen_root,
        "runner_dataset_root_required": True,
        "freeze_root_device": int(freeze_root_metadata.st_dev),
        "freeze_root_inode": int(freeze_root_metadata.st_ino),
        "view_root_device": int(frozen_view["root_device"]),
        "view_root_inode": int(frozen_view["root_inode"]),
        "manifest_content_sha256": dict(frozen_view["manifest_content_sha256"]),
        "ordered_pair_ids_sha256": dict(frozen_view["ordered_pair_ids_sha256"]),
        "canonical_ordered_pairs_sha256": dict(
            frozen_view["canonical_ordered_pairs_sha256"]
        ),
        "file_count": int(frozen_view["file_count"]),
        "total_size_bytes": int(frozen_view["total_size_bytes"]),
        "file_mode_octal": "0444",
        "directory_mode_octal": "0555",
    }
    if dict(frozen_dataset) != expected_frozen_dataset:
        _fail("receipt frozen dataset binding differs")
    final_tree_files, final_tree_directories = _walk_anchored_tree(freeze_anchor)
    if final_tree_files != tree_files or final_tree_directories != tree_directories:
        _fail("freeze output tree changed during verification")
    final_receipt_payload, final_receipt_digest = freeze_anchor.read(
        RECEIPT_FILENAME, "final freeze receipt", retain_bytes=True
    )
    final_inventory_payload, final_inventory_digest = freeze_anchor.read(
        INVENTORY_FILENAME, "final semantic asset inventory", retain_bytes=True
    )
    if (
        final_receipt_payload != receipt_payload
        or final_receipt_digest != receipt_digest
        or final_inventory_payload != inventory_payload
        or final_inventory_digest != inventory_digest
    ):
        _fail("freeze receipt or inventory changed during verification")
    final_freeze_root_metadata = os.fstat(freeze_anchor.descriptor)
    if (
        _entry_identity(final_freeze_root_metadata) != freeze_root_identity
        or stat.S_IMODE(final_freeze_root_metadata.st_mode) != 0o555
    ):
        _fail("freeze root descriptor identity changed during verification")
    freeze_anchor.revalidate()
    # The receipt itself is also fully read and hashed above.  Return its hash
    # so queue controllers can bind an exact verified freeze artifact.
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_train_val_semantic_assets_unchanged",
        "dataset_root": root.as_posix(),
        "freeze_dir": frozen.as_posix(),
        "frozen_dataset_root": expected_frozen_root,
        "freeze_root_device": int(freeze_root_metadata.st_dev),
        "freeze_root_inode": int(freeze_root_metadata.st_ino),
        "view_root_device": int(frozen_view["root_device"]),
        "view_root_inode": int(frozen_view["root_inode"]),
        "freeze_receipt_sha256": receipt_digest["sha256"],
        "freeze_receipt_size_bytes": receipt_digest["size_bytes"],
        "inventory_file_sha256": inventory_digest["sha256"],
        "inventory_file_size_bytes": inventory_digest["size_bytes"],
        "semantic_inventory_content_sha256": semantic_sha,
        "source_asset_file_count": semantic["asset_file_count"],
        "source_asset_total_size_bytes": semantic["asset_total_size_bytes"],
        "frozen_asset_file_count": frozen_view["file_count"],
        "frozen_asset_total_size_bytes": frozen_view["total_size_bytes"],
        "manifest_content_sha256": frozen_view["manifest_content_sha256"],
        "ordered_pair_ids_sha256": frozen_view["ordered_pair_ids_sha256"],
        "canonical_ordered_pairs_sha256": frozen_view[
            "canonical_ordered_pairs_sha256"
        ],
        "rehash_all_frozen_dataset_bytes": True,
        "same_freeze_root_fd_held_for_full_verification": True,
        "final_complete_tree_revalidated": True,
        "frozen_view_regular_file_link_count_required": 1,
        "source_snapshot_reverified": True,
        "runner_dataset_root_is_self_contained_frozen_view": True,
        "sealed_test_manifest_opened": False,
        "sealed_test_assets_opened": False,
        "real_data_opened": False,
        "rgb_opened": False,
    }


def verify_train_val_semantic_asset_freeze(
    dataset_root: Union[str, Path],
    freeze_dir: Union[str, Path],
    *,
    require_formal_counts: bool = True,
    expected_freeze_receipt_sha256: Optional[str] = None,
    expected_semantic_inventory_content_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Rehash source/view bytes while retaining one freeze-root FD throughout.

    Queue controllers should pin both optional expected hashes when rechecking
    the same freeze between methods.  The receipt, inventory, complete tree,
    and frozen dataset view are all opened from one persistent freeze anchor.
    """

    if type(require_formal_counts) is not bool:  # noqa: E721
        raise TypeError("require_formal_counts must be bool")
    for value, description in (
        (expected_freeze_receipt_sha256, "expected freeze receipt SHA-256"),
        (
            expected_semantic_inventory_content_sha256,
            "expected semantic inventory content SHA-256",
        ),
    ):
        if value is not None and not _is_sha256(value):
            _fail(description + " is malformed")

    root = _validated_dataset_root(dataset_root)
    frozen = _validated_dataset_root(freeze_dir)
    with _AnchoredDatasetRoot(frozen) as freeze_anchor:
        return _verify_train_val_semantic_asset_freeze_anchored(
            root,
            frozen,
            freeze_anchor,
            require_formal_counts=require_formal_counts,
            expected_freeze_receipt_sha256=expected_freeze_receipt_sha256,
            expected_semantic_inventory_content_sha256=(
                expected_semantic_inventory_content_sha256
            ),
        )


# Short aliases intended for queue/controller integration.
freeze_train_val_assets = publish_train_val_semantic_asset_freeze
verify_train_val_assets = verify_train_val_semantic_asset_freeze


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--dataset-root", type=Path, required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--dataset-root", type=Path, required=True)
    verify.add_argument("--freeze-dir", type=Path, required=True)
    verify.add_argument("--expected-freeze-receipt-sha256")
    verify.add_argument("--expected-semantic-inventory-content-sha256")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "create":
        result = publish_train_val_semantic_asset_freeze(
            arguments.dataset_root,
            arguments.output_dir,
            require_formal_counts=True,
        )
    elif arguments.command == "verify":
        result = verify_train_val_semantic_asset_freeze(
            arguments.dataset_root,
            arguments.freeze_dir,
            require_formal_counts=True,
            expected_freeze_receipt_sha256=(
                arguments.expected_freeze_receipt_sha256
            ),
            expected_semantic_inventory_content_sha256=(
                arguments.expected_semantic_inventory_content_sha256
            ),
        )
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FROZEN_DATA_DIRNAME",
    "FROZEN_VIEW_SCHEMA_VERSION",
    "INVENTORY_FILENAME",
    "INVENTORY_SCHEMA_VERSION",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "RachelTrainValAssetFreezeError",
    "build_train_val_semantic_inventory",
    "freeze_train_val_assets",
    "main",
    "publish_train_val_semantic_asset_freeze",
    "verify_train_val_assets",
    "verify_train_val_semantic_asset_freeze",
]

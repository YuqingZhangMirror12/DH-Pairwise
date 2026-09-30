"""Bounded, content-addressed offline cache for expensive geometry artifacts.

The key is per *fragment*, never per pair: canonical mask content, the full
geometry configuration (including polarity), threshold rule, and recipe/code
versions are hashed together.  A single-fragment artifact can therefore be
reused by every pair containing that fragment.

Each entry is one atomically replaced NPZ container.  Readers never observe a
partially written payload; they also verify the embedded manifest and every
array hash, so corruption/tampering fails closed.  Concurrent writers may do
duplicate work, but their commits are serialized and readers remain safe.
"""

from __future__ import annotations

import ast
import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from io import BufferedIOBase
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from staging.pairwise_v0_2.geometry import (
    CHANNEL_ORDER,
    FRAGMENT_GEOMETRY_ARTIFACT_VERSION,
    GEOMETRY_VERSION,
    WINDOW_NORMALIZATION,
    CandidateBuilderConfig,
)


CACHE_SCHEMA_VERSION = "dunhuang-fragment-geometry-cache/0.3"
CACHE_RECIPE_VERSION = "upright-role-neutral-fragment-patches/0.3"
CACHE_TYPED_PAYLOAD_VERSION = "fragment-geometry-payload/0.2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_NAME = "__manifest__"
_COMMIT_LOCK_NAME = ".commit.lock"
_QUOTA_LEDGER_NAME = ".quota-ledger.json"
_QUOTA_LEDGER_VERSION = "dunhuang-fragment-cache-quota/0.3"
_RECEIPT_VERSION = "dunhuang-fragment-cache-receipt/0.4"
_RECEIPT_FILE_ENCODING = "canonical_json_utf8_v1"
_NPY_MAGIC = b"\x93NUMPY"
_MAX_NPY_HEADER_BYTES = 65_536


class GeometryCacheError(RuntimeError):
    """Base class for cache contract failures."""


class GeometryCacheCorruptionError(GeometryCacheError):
    """An entry exists but its container, manifest, or payload is invalid."""


class GeometryCacheConflictError(GeometryCacheError):
    """A write attempted to replace an immutable key with another payload."""


@dataclass(frozen=True)
class GeometryCacheLimits:
    max_artifact_file_bytes: int = 1_000_000_000
    max_cache_file_bytes: int = 100_000_000_000
    max_array_count: int = 8192
    max_array_elements: int = 250_000_000
    max_total_array_bytes: int = 2_000_000_000
    max_metadata_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))


@dataclass(frozen=True)
class FrozenReceiptTrust:
    """External hashes that authorize one reviewed frozen-cache receipt.

    These values must come from the experiment/data lock, not from the receipt
    being opened.  ``expected_file_sha256`` binds the canonical UTF-8 JSON
    serialization including ``content_sha256``; ``expected_content_sha256``
    independently binds the receipt payload excluding that self-hash field.
    """

    expected_content_sha256: str
    expected_file_sha256: str
    file_encoding: str = _RECEIPT_FILE_ENCODING

    def __post_init__(self) -> None:
        for name in ("expected_content_sha256", "expected_file_sha256"):
            if not isinstance(getattr(self, name), str) or not SHA256_RE.fullmatch(
                getattr(self, name)
            ):
                raise ValueError("{} must be lowercase SHA-256".format(name))
        if self.file_encoding != _RECEIPT_FILE_ENCODING:
            raise ValueError("unsupported frozen receipt file encoding")


@dataclass(frozen=True)
class FragmentCacheIdentity:
    """Portable identity; deliberately excludes paths, pair IDs, and labels."""

    canonical_mask_sha256: str
    threshold_rule: str
    geometry_config_sha256: str
    key: str
    cache_schema_version: str = CACHE_SCHEMA_VERSION
    recipe_version: str = CACHE_RECIPE_VERSION
    geometry_version: str = GEOMETRY_VERSION

    def __post_init__(self) -> None:
        for name in ("canonical_mask_sha256", "geometry_config_sha256", "key"):
            if not SHA256_RE.fullmatch(getattr(self, name)):
                raise ValueError("{} must be lowercase SHA-256".format(name))
        if not self.threshold_rule.strip():
            raise ValueError("threshold_rule is required")
        if (
            self.cache_schema_version != CACHE_SCHEMA_VERSION
            or self.recipe_version != CACHE_RECIPE_VERSION
            or self.geometry_version != GEOMETRY_VERSION
        ):
            raise ValueError("fragment cache identity version mismatch")
        if self.key != _identity_key_sha256(self):
            raise ValueError("fragment cache identity key mismatch")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _FrozenCacheBinding:
    expected_artifacts: Mapping[str, Mapping[str, Any]]
    trust: FrozenReceiptTrust

    def __post_init__(self) -> None:
        copied = {}
        for key, value in self.expected_artifacts.items():
            copied[key] = MappingProxyType(dict(value))
        object.__setattr__(self, "expected_artifacts", MappingProxyType(copied))


@dataclass(frozen=True)
class GeometryCacheArtifact:
    identity: FragmentCacheIdentity
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    logical_payload_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, FragmentCacheIdentity):
            raise TypeError("identity must be FragmentCacheIdentity")
        if not isinstance(self.arrays, Mapping):
            raise TypeError("arrays must be a mapping")
        immutable_arrays = {
            name: _permanently_read_only_array(value)
            for name, value in self.arrays.items()
        }
        portable_metadata = _portable(self.metadata)
        object.__setattr__(self, "arrays", MappingProxyType(immutable_arrays))
        object.__setattr__(self, "metadata", _deep_freeze_json(portable_metadata))
        self.verify_logical_payload()

    def verify_logical_payload(self) -> None:
        """Recompute the logical digest from current identity, arrays and metadata."""

        if not isinstance(self.logical_payload_sha256, str) or not SHA256_RE.fullmatch(
            self.logical_payload_sha256
        ):
            raise GeometryCacheCorruptionError(
                "artifact logical payload hash is invalid"
            )
        observed = _logical_payload_sha256(self.identity, self.arrays, self.metadata)
        if not hmac.compare_digest(observed, self.logical_payload_sha256):
            raise GeometryCacheCorruptionError("artifact logical payload hash mismatch")

    def metadata_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-serializable copy of frozen metadata."""

        return _deep_thaw_json(self.metadata)


@dataclass(frozen=True)
class GeometryCacheLookup:
    artifact: GeometryCacheArtifact
    cache_hit: bool


@dataclass(frozen=True)
class GeometryCostEstimate:
    pair_count: int
    epochs: int
    seconds_per_pair_low: float
    seconds_per_pair_high: float
    uncached_days_low: float
    uncached_days_high: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_loads(value: str) -> Any:
    def reject_duplicates(pairs):
        result = {}
        for name, item in pairs:
            if name in result:
                raise ValueError("duplicate JSON key")
            result[name] = item
        return result

    return json.loads(value, object_pairs_hook=reject_duplicates)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_mask_sha256(mask: np.ndarray) -> str:
    value = np.asarray(mask)
    if value.ndim != 2 or value.size == 0 or value.dtype != np.bool_:
        raise ValueError("canonical cache masks must be non-empty 2D bool arrays")
    header = _canonical_json({"dtype": "bool", "shape": list(value.shape)})
    packed = np.packbits(np.ascontiguousarray(value).reshape(-1), bitorder="little")
    return _sha256(header + b"\0" + packed.tobytes())


def fragment_cache_identity(
    mask: np.ndarray,
    threshold_rule: str,
    geometry_config: CandidateBuilderConfig,
) -> FragmentCacheIdentity:
    if not isinstance(geometry_config, CandidateBuilderConfig):
        raise TypeError("geometry_config must be CandidateBuilderConfig")
    if not isinstance(threshold_rule, str) or not threshold_rule.strip():
        raise ValueError("threshold_rule is required")
    config_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "recipe_version": CACHE_RECIPE_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "typed_payload_version": CACHE_TYPED_PAYLOAD_VERSION,
        "fragment_artifact_version": FRAGMENT_GEOMETRY_ARTIFACT_VERSION,
        "channel_order": list(CHANNEL_ORDER),
        "window_normalization": WINDOW_NORMALIZATION,
        "geometry_config": asdict(geometry_config),
    }
    config_sha = _sha256(_canonical_json(config_payload))
    mask_sha = canonical_mask_sha256(mask)
    key_payload = {
        "canonical_mask_sha256": mask_sha,
        "threshold_rule": threshold_rule,
        "geometry_config_sha256": config_sha,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "recipe_version": CACHE_RECIPE_VERSION,
        "geometry_version": GEOMETRY_VERSION,
    }
    return FragmentCacheIdentity(
        canonical_mask_sha256=mask_sha,
        threshold_rule=threshold_rule,
        geometry_config_sha256=config_sha,
        key=_sha256(_canonical_json(key_payload)),
    )


def estimate_uncached_geometry_cost(
    pair_count: int,
    epochs: int,
    seconds_per_pair: Tuple[float, float] = (3.0, 5.0),
) -> GeometryCostEstimate:
    if pair_count <= 0 or epochs <= 0:
        raise ValueError("pair_count and epochs must be positive")
    low, high = map(float, seconds_per_pair)
    if not (0.0 < low <= high and math.isfinite(low) and math.isfinite(high)):
        raise ValueError("seconds_per_pair must be a finite positive range")
    scale = pair_count * epochs / 86_400.0
    return GeometryCostEstimate(
        pair_count=pair_count,
        epochs=epochs,
        seconds_per_pair_low=low,
        seconds_per_pair_high=high,
        uncached_days_low=scale * low,
        uncached_days_high=scale * high,
    )


def _portable(value: Any, *, key_name: Optional[str] = None) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str):
            if value.startswith(("/", "~/", "file://")) or re.match(
                r"^[A-Za-z]:[\\/]", value
            ):
                raise ValueError("portable metadata cannot contain local paths")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("portable metadata cannot contain non-finite values")
        return value
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, Mapping):
        result = {}
        for name, item in value.items():
            if not isinstance(name, str):
                raise TypeError("metadata keys must be strings")
            normalized = name.casefold()
            if normalized in {"path", "local_path", "absolute_path", "cwd"}:
                raise ValueError("portable metadata cannot contain path fields")
            result[name] = _portable(item, key_name=name)
        return result
    raise TypeError("metadata contains a non-JSON value")


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {name: _deep_freeze_json(item) for name, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _deep_thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _deep_thaw_json(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_thaw_json(item) for item in value]
    return value


def _array_bytes(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(value).view(np.uint8).tobytes()


def _has_immutable_buffer(value: np.ndarray) -> bool:
    current: Any = value
    seen = set()
    while isinstance(current, np.ndarray) and id(current) not in seen:
        seen.add(id(current))
        current = current.base
    if isinstance(current, bytes):
        return True
    if isinstance(current, memoryview):
        return current.readonly
    return False


def _permanently_read_only_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype.hasobject
        or array.dtype.fields is not None
        or array.dtype.itemsize <= 0
    ):
        raise TypeError("object, structured, and zero-width arrays are forbidden")
    if (
        array.flags.c_contiguous
        and not array.flags.writeable
        and _has_immutable_buffer(array)
    ):
        return array
    contiguous = np.ascontiguousarray(array)
    immutable = bytes(contiguous.view(np.uint8))
    output = np.frombuffer(immutable, dtype=contiguous.dtype).reshape(contiguous.shape)
    if output.flags.writeable:  # pragma: no cover - bytes-backed invariant
        raise RuntimeError("immutable array buffer unexpectedly became writable")
    return output


def _array_specs(
    arrays: Mapping[str, np.ndarray],
) -> Tuple[Sequence[Dict[str, Any]], int]:
    specs = []
    total_bytes = 0
    for name in sorted(arrays):
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name)
            or name == _MANIFEST_NAME
        ):
            raise GeometryCacheCorruptionError("artifact array name is invalid")
        value = np.asarray(arrays[name])
        if (
            value.dtype.hasobject
            or value.dtype.fields is not None
            or value.dtype.itemsize <= 0
        ):
            raise GeometryCacheCorruptionError("artifact array dtype is invalid")
        total_bytes += value.nbytes
        specs.append(
            {
                "name": name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "nbytes": value.nbytes,
                "sha256": _sha256(_array_bytes(value)),
            }
        )
    return specs, total_bytes


def _logical_payload_sha256(
    identity: FragmentCacheIdentity,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> str:
    specs, total_bytes = _array_specs(arrays)
    portable_metadata = _portable(metadata)
    if not isinstance(portable_metadata, dict):
        raise GeometryCacheCorruptionError("artifact metadata must be an object")
    return _sha256(
        _canonical_json(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "identity": identity.to_dict(),
                "arrays": specs,
                "metadata": portable_metadata,
                "total_array_bytes": total_bytes,
            }
        )
    )


def _identity_key_sha256(identity: FragmentCacheIdentity) -> str:
    return _sha256(
        _canonical_json(
            {
                "canonical_mask_sha256": identity.canonical_mask_sha256,
                "threshold_rule": identity.threshold_rule,
                "geometry_config_sha256": identity.geometry_config_sha256,
                "cache_schema_version": identity.cache_schema_version,
                "recipe_version": identity.recipe_version,
                "geometry_version": identity.geometry_version,
            }
        )
    )


def canonical_receipt_file_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash the canonical receipt file bytes for an external experiment lock.

    This is a freeze-time helper only.  A loader must receive the resulting
    digest through :class:`FrozenReceiptTrust`; deriving it from the untrusted
    receipt at load time would recreate a self-signing trust boundary.
    """

    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    portable = _portable(dict(receipt))
    if not isinstance(portable, dict):  # pragma: no cover - dict normalization
        raise TypeError("receipt must normalize to an object")
    return _sha256(_canonical_json(portable))


@dataclass(frozen=True)
class _NpyMemberHeader:
    info: ZipInfo
    array_name: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    header_bytes: int
    payload_bytes: int


def _read_exact(stream: BufferedIOBase, size: int, description: str) -> bytes:
    if size < 0:
        raise GeometryCacheCorruptionError("{} size is invalid".format(description))
    value = stream.read(size)
    if len(value) != size:
        raise GeometryCacheCorruptionError("{} is truncated".format(description))
    return value


def _inspect_npy_member(
    archive: ZipFile,
    info: ZipInfo,
    *,
    max_header_bytes: int,
) -> _NpyMemberHeader:
    with archive.open(info, "r") as stream:
        if _read_exact(stream, len(_NPY_MAGIC), "NPY magic") != _NPY_MAGIC:
            raise GeometryCacheCorruptionError("cache member is not an NPY array")
        version = tuple(_read_exact(stream, 2, "NPY version"))
        if version == (1, 0):
            length_size = 2
        elif version in {(2, 0), (3, 0)}:
            length_size = 4
        else:
            raise GeometryCacheCorruptionError("unsupported NPY format version")
        length_bytes = _read_exact(stream, length_size, "NPY header length")
        header_length = int.from_bytes(length_bytes, "little", signed=False)
        if header_length <= 0 or header_length > max_header_bytes:
            raise GeometryCacheCorruptionError("NPY header exceeds metadata bound")
        header_bytes = _read_exact(stream, header_length, "NPY header")
        encoding = "latin1" if version <= (2, 0) else "utf-8"
        try:
            header = ast.literal_eval(header_bytes.decode(encoding).strip())
        except (SyntaxError, UnicodeError, ValueError) as exc:
            raise GeometryCacheCorruptionError("NPY header is invalid") from exc
        if not isinstance(header, dict) or set(header) != {
            "descr",
            "fortran_order",
            "shape",
        }:
            raise GeometryCacheCorruptionError("NPY header fields are invalid")
        shape = header["shape"]
        if (
            not isinstance(shape, tuple)
            or len(shape) > 32
            or any(
                isinstance(part, bool) or not isinstance(part, int) or part < 0
                for part in shape
            )
        ):
            raise GeometryCacheCorruptionError("NPY shape is invalid")
        if header["fortran_order"] is not False:
            raise GeometryCacheCorruptionError(
                "Fortran-order cache arrays are forbidden"
            )
        try:
            dtype = np.dtype(header["descr"])
        except (TypeError, ValueError) as exc:
            raise GeometryCacheCorruptionError("NPY dtype is invalid") from exc
        if dtype.hasobject or dtype.fields is not None or dtype.itemsize <= 0:
            raise GeometryCacheCorruptionError("NPY dtype is forbidden")
        elements = math.prod(shape) if shape else 1
        payload_bytes = elements * dtype.itemsize
        header_total = len(_NPY_MAGIC) + 2 + length_size + header_length
        if header_total + payload_bytes != info.file_size:
            raise GeometryCacheCorruptionError("NPY entry size disagrees with header")
    return _NpyMemberHeader(
        info=info,
        array_name=info.filename[:-4],
        dtype=dtype,
        shape=shape,
        header_bytes=header_total,
        payload_bytes=payload_bytes,
    )


def _read_npy_payload(archive: ZipFile, expected: _NpyMemberHeader) -> np.ndarray:
    observed = _inspect_npy_member(
        archive,
        expected.info,
        max_header_bytes=_MAX_NPY_HEADER_BYTES,
    )
    if (
        observed.array_name != expected.array_name
        or observed.dtype != expected.dtype
        or observed.shape != expected.shape
        or observed.header_bytes != expected.header_bytes
        or observed.payload_bytes != expected.payload_bytes
    ):
        raise GeometryCacheCorruptionError("NPY header changed during read")
    with archive.open(expected.info, "r") as stream:
        _read_exact(stream, expected.header_bytes, "NPY header")
        payload = _read_exact(stream, expected.payload_bytes, "NPY payload")
        if stream.read(1):
            raise GeometryCacheCorruptionError("NPY payload has trailing bytes")
    return np.frombuffer(payload, dtype=expected.dtype).reshape(expected.shape)


class GeometryArtifactCache:
    """Single-file atomic fragment cache with integrity and allocation guards."""

    def __init__(
        self,
        root: Path,
        limits: Optional[GeometryCacheLimits] = None,
        *,
        read_only: bool = False,
        _frozen_binding: Optional[_FrozenCacheBinding] = None,
    ) -> None:
        self.root = Path(root)
        if limits is not None and not isinstance(limits, GeometryCacheLimits):
            raise TypeError("limits must be GeometryCacheLimits")
        self.limits = limits or GeometryCacheLimits()
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be bool")
        self.read_only = read_only
        if _frozen_binding is not None and not isinstance(
            _frozen_binding, _FrozenCacheBinding
        ):
            raise TypeError("internal frozen binding is invalid")
        if _frozen_binding is not None and not self.read_only:
            raise GeometryCacheError("frozen receipt bindings require read-only mode")
        self._frozen_binding = _frozen_binding
        self._frozen_receipt_verified = False
        self._accounted_total_bytes = 0
        self._accounted_entry_count = 0
        if self.read_only:
            if not self.root.is_dir():
                raise GeometryCacheError("read-only cache root does not exist")
            total_bytes, entry_count = self._scan_cache_unlocked()
            self._accounted_total_bytes = total_bytes
            self._accounted_entry_count = entry_count
            if self._frozen_binding is not None:
                observed = set()
                for item in self.root.rglob("*.npz"):
                    if not item.is_file():
                        continue
                    key = item.stem
                    if (
                        not SHA256_RE.fullmatch(key)
                        or item.parent != self.root / key[:2]
                        or item.name != key + ".npz"
                    ):
                        raise GeometryCacheError(
                            "frozen cache contains a non-canonical entry path"
                        )
                    if key in observed:
                        raise GeometryCacheError(
                            "frozen cache contains duplicate cache keys"
                        )
                    observed.add(key)
                if observed != set(self._frozen_binding.expected_artifacts):
                    raise GeometryCacheError(
                        "frozen cache inventory differs from reviewed receipt"
                    )
            return
        self.root.mkdir(parents=True, exist_ok=True)
        # Reconcile once per cache object.  Subsequent commits update the small
        # ledger in O(1), avoiding an O(N) tree walk for every fragment write.
        # The file lock also makes construction safe when several workers open
        # the same cache concurrently.
        with self._commit_lock():
            total_bytes, entry_count = self._scan_cache_unlocked()
            self._write_quota_ledger_unlocked(total_bytes, entry_count)
            self._accounted_total_bytes = total_bytes
            self._accounted_entry_count = entry_count

    @property
    def frozen_receipt_trust(self) -> Optional[FrozenReceiptTrust]:
        if self._frozen_binding is None or not self._frozen_receipt_verified:
            return None
        return self._frozen_binding.trust

    def _entry_path(self, identity: FragmentCacheIdentity) -> Path:
        return self.root / identity.key[:2] / (identity.key + ".npz")

    def _commit_lock(self):
        """Return an opened, exclusively locked commit file context manager."""

        class _Lock:
            def __init__(self, path: Path) -> None:
                self.path = path
                self.handle = None

            def __enter__(self):
                self.handle = self.path.open("a+b")
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
                return self.handle

            def __exit__(self, exc_type, exc, traceback) -> None:
                assert self.handle is not None
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                self.handle.close()

        return _Lock(self.root / _COMMIT_LOCK_NAME)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(str(path), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            # Some filesystems do not expose directory fsync.  Atomic rename
            # still holds; the caller can record this as a durability caveat.
            pass
        finally:
            os.close(descriptor)

    def _scan_cache_unlocked(self) -> Tuple[int, int]:
        total_bytes = 0
        entry_count = 0
        for item in self.root.rglob("*.npz"):
            if not item.is_file():
                continue
            try:
                file_bytes = item.stat().st_size
            except OSError as exc:
                raise GeometryCacheError("cannot inspect cache entry size") from exc
            if file_bytes > self.limits.max_artifact_file_bytes:
                raise GeometryCacheError("existing cache artifact exceeds file bound")
            total_bytes += file_bytes
            if total_bytes > self.limits.max_cache_file_bytes:
                raise GeometryCacheError("existing cache exceeds total file quota")
            entry_count += 1
        return total_bytes, entry_count

    def _write_quota_ledger_unlocked(self, total_bytes: int, entry_count: int) -> None:
        payload = _canonical_json(
            {
                "schema_version": _QUOTA_LEDGER_VERSION,
                "total_file_bytes": total_bytes,
                "entry_count": entry_count,
            }
        )
        fd, temporary_name = tempfile.mkstemp(
            prefix=".quota-ledger.", suffix=".tmp", dir=str(self.root)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.root / _QUOTA_LEDGER_NAME)
            self._fsync_directory(self.root)
        finally:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass

    def _read_quota_ledger_unlocked(self) -> Tuple[int, int]:
        path = self.root / _QUOTA_LEDGER_NAME
        try:
            value = _strict_json_loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or set(value) != {"schema_version", "total_file_bytes", "entry_count"}
                or value["schema_version"] != _QUOTA_LEDGER_VERSION
            ):
                raise ValueError("quota ledger schema mismatch")
            total_bytes = value["total_file_bytes"]
            entry_count = value["entry_count"]
            if (
                isinstance(total_bytes, bool)
                or not isinstance(total_bytes, int)
                or total_bytes < 0
                or isinstance(entry_count, bool)
                or not isinstance(entry_count, int)
                or entry_count < 0
            ):
                raise ValueError("quota ledger values are invalid")
            adjusted = False
            if total_bytes < self._accounted_total_bytes:
                total_bytes = self._accounted_total_bytes
                adjusted = True
            if entry_count < self._accounted_entry_count:
                entry_count = self._accounted_entry_count
                adjusted = True
            if total_bytes > self.limits.max_cache_file_bytes:
                raise GeometryCacheError("cache quota ledger exceeds total file quota")
            if adjusted:
                self._write_quota_ledger_unlocked(total_bytes, entry_count)
            self._accounted_total_bytes = total_bytes
            self._accounted_entry_count = entry_count
            return total_bytes, entry_count
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            total_bytes, entry_count = self._scan_cache_unlocked()
            self._write_quota_ledger_unlocked(total_bytes, entry_count)
            self._accounted_total_bytes = total_bytes
            self._accounted_entry_count = entry_count
            return total_bytes, entry_count

    def _validate_arrays(
        self, arrays: Mapping[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Sequence[Dict[str, Any]], int]:
        if len(arrays) > self.limits.max_array_count:
            raise GeometryCacheError("artifact exceeds array-count bound")
        normalized: Dict[str, np.ndarray] = {}
        specs = []
        total_bytes = 0
        for name in sorted(arrays):
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name):
                raise ValueError("array names must be portable identifiers")
            if name == _MANIFEST_NAME:
                raise ValueError("reserved array name")
            value = np.asarray(arrays[name])
            if (
                value.dtype.hasobject
                or value.dtype.fields is not None
                or value.dtype.itemsize <= 0
            ):
                raise TypeError(
                    "object, structured, and zero-width arrays are forbidden"
                )
            if value.size > self.limits.max_array_elements:
                raise GeometryCacheError("array exceeds element bound")
            value = np.ascontiguousarray(value)
            value.setflags(write=False)
            total_bytes += value.nbytes
            if total_bytes > self.limits.max_total_array_bytes:
                raise GeometryCacheError("artifact exceeds total array-byte bound")
            digest = _sha256(_array_bytes(value))
            specs.append(
                {
                    "name": name,
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "nbytes": value.nbytes,
                    "sha256": digest,
                }
            )
            normalized[name] = value
        return normalized, specs, total_bytes

    def put(
        self,
        identity: FragmentCacheIdentity,
        arrays: Mapping[str, np.ndarray],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> GeometryCacheArtifact:
        if self.read_only:
            raise GeometryCacheError("cannot write to a read-only cache")
        normalized, specs, total_bytes = self._validate_arrays(arrays)
        portable_metadata = _portable(dict(metadata or {}))
        if len(_canonical_json(portable_metadata)) > self.limits.max_metadata_bytes:
            raise GeometryCacheError("artifact metadata exceeds byte bound")
        logical = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "identity": identity.to_dict(),
            "arrays": specs,
            "metadata": portable_metadata,
            "total_array_bytes": total_bytes,
        }
        logical_sha = _sha256(_canonical_json(logical))
        manifest = dict(logical)
        manifest["logical_payload_sha256"] = logical_sha
        manifest_bytes = _canonical_json(manifest)
        if len(manifest_bytes) > self.limits.max_metadata_bytes:
            raise GeometryCacheError("artifact manifest exceeds metadata byte bound")
        payload = dict(normalized)
        payload[_MANIFEST_NAME] = np.frombuffer(manifest_bytes, dtype=np.uint8)

        target = self._entry_path(identity)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Cache entries are immutable.  This turns the public content hashes
        # into a write-once integrity boundary rather than allowing a caller to
        # silently re-sign arbitrary tensors under an existing identity.
        if target.exists():
            existing = self.get(identity)
            if existing.logical_payload_sha256 == logical_sha:
                return existing
            raise GeometryCacheConflictError(
                "cache key already exists with a different logical payload"
            )

        fd, temporary_name = tempfile.mkstemp(
            prefix="." + identity.key + ".", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary = Path(temporary_name)
            file_bytes = temporary.stat().st_size
            if file_bytes > self.limits.max_artifact_file_bytes:
                raise GeometryCacheError("compressed artifact exceeds file bound")
            with self._commit_lock():
                if target.exists():
                    existing_artifact = self.get(identity)
                    if existing_artifact.logical_payload_sha256 != logical_sha:
                        raise GeometryCacheConflictError(
                            "cache key already exists with a different logical payload"
                        )
                    return existing_artifact
                current, entry_count = self._read_quota_ledger_unlocked()
                if current + file_bytes > self.limits.max_cache_file_bytes:
                    raise GeometryCacheError("cache quota would be exceeded")
                # Reserve quota durably *before* publishing the entry.  A crash
                # may leave a conservative over-count, which is reconciled by
                # the next cache open; it can never leave an on-disk artifact
                # absent from the ledger and permit an over-quota next write.
                reserved_total = current + file_bytes
                reserved_count = entry_count + 1
                self._write_quota_ledger_unlocked(reserved_total, reserved_count)
                self._accounted_total_bytes = reserved_total
                self._accounted_entry_count = reserved_count
                os.replace(str(temporary), str(target))
                self._fsync_directory(target.parent)
        finally:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        return self.get(identity)

    def get(self, identity: FragmentCacheIdentity) -> GeometryCacheArtifact:
        if not isinstance(identity, FragmentCacheIdentity):
            raise TypeError("identity must be FragmentCacheIdentity")
        path = self._entry_path(identity)
        if not path.is_file():
            raise KeyError(identity.key)
        file_bytes = path.stat().st_size
        if file_bytes > self.limits.max_artifact_file_bytes:
            raise GeometryCacheCorruptionError("cache file exceeds bound")
        try:
            with ZipFile(path) as archive:
                members = archive.infolist()
                member_names = [item.filename for item in members]
                if len(member_names) != len(set(member_names)):
                    raise GeometryCacheCorruptionError(
                        "cache contains duplicate members"
                    )
                if len(members) > self.limits.max_array_count + 1:
                    raise GeometryCacheCorruptionError(
                        "cache member count exceeds bound"
                    )
                manifest_member_name = _MANIFEST_NAME + ".npy"
                if member_names.count(manifest_member_name) != 1:
                    raise GeometryCacheCorruptionError("cache manifest is missing")
                info_by_array_name = {}
                central_array_bytes = 0
                for info in members:
                    if (
                        info.is_dir()
                        or info.compress_type not in {ZIP_STORED, ZIP_DEFLATED}
                        or info.flag_bits & ~0x800
                    ):
                        raise GeometryCacheCorruptionError(
                            "cache ZIP member policy is invalid"
                        )
                    if info.filename == manifest_member_name:
                        array_name = _MANIFEST_NAME
                        if info.file_size > (
                            self.limits.max_metadata_bytes
                            + min(
                                _MAX_NPY_HEADER_BYTES,
                                self.limits.max_metadata_bytes,
                            )
                        ):
                            raise GeometryCacheCorruptionError(
                                "cache manifest ZIP entry exceeds metadata byte bound"
                            )
                    else:
                        match = re.fullmatch(
                            r"([A-Za-z][A-Za-z0-9_]{0,127})\.npy",
                            info.filename,
                        )
                        if match is None:
                            raise GeometryCacheCorruptionError(
                                "cache ZIP member name is invalid"
                            )
                        array_name = match.group(1)
                        central_array_bytes += info.file_size
                    info_by_array_name[array_name] = info
                if central_array_bytes > (
                    self.limits.max_total_array_bytes + self.limits.max_metadata_bytes
                ):
                    raise GeometryCacheCorruptionError(
                        "cache array ZIP entries exceed expansion bound"
                    )

                # Inspect every NPY declaration before reading any member body.
                # Central-directory sizes alone do not prevent a small header
                # from declaring an unsafe shape, or a manifest member from
                # borrowing the much larger array expansion budget.
                inspected = {}
                header_bytes = 0
                declared_array_bytes = 0
                for array_name in sorted(info_by_array_name):
                    header = _inspect_npy_member(
                        archive,
                        info_by_array_name[array_name],
                        max_header_bytes=min(
                            _MAX_NPY_HEADER_BYTES,
                            self.limits.max_metadata_bytes,
                        ),
                    )
                    inspected[array_name] = header
                    header_bytes += header.header_bytes
                    if header_bytes > self.limits.max_metadata_bytes:
                        raise GeometryCacheCorruptionError(
                            "cache NPY headers exceed metadata bound"
                        )
                    elements = math.prod(header.shape) if header.shape else 1
                    if array_name == _MANIFEST_NAME:
                        if (
                            header.dtype != np.dtype(np.uint8)
                            or len(header.shape) != 1
                            or header.payload_bytes > self.limits.max_metadata_bytes
                        ):
                            raise GeometryCacheCorruptionError(
                                "cache manifest NPY declaration is invalid"
                            )
                    else:
                        if elements > self.limits.max_array_elements:
                            raise GeometryCacheCorruptionError(
                                "cached array exceeds element bound"
                            )
                        declared_array_bytes += header.payload_bytes
                        if declared_array_bytes > self.limits.max_total_array_bytes:
                            raise GeometryCacheCorruptionError(
                                "cached arrays exceed total byte bound"
                            )

                raw_manifest = _read_npy_payload(archive, inspected[_MANIFEST_NAME])
                if raw_manifest.dtype != np.uint8 or raw_manifest.ndim != 1:
                    raise GeometryCacheCorruptionError("cache manifest type is invalid")
                if raw_manifest.nbytes > self.limits.max_metadata_bytes:
                    raise GeometryCacheCorruptionError(
                        "cache manifest exceeds metadata byte bound"
                    )
                manifest = _strict_json_loads(raw_manifest.tobytes().decode("utf-8"))
                if not isinstance(manifest, dict) or set(manifest) != {
                    "schema_version",
                    "identity",
                    "arrays",
                    "metadata",
                    "total_array_bytes",
                    "logical_payload_sha256",
                }:
                    raise GeometryCacheCorruptionError(
                        "cache manifest fields are invalid"
                    )
                logical_sha = manifest.get("logical_payload_sha256")
                if not isinstance(logical_sha, str) or not SHA256_RE.fullmatch(
                    logical_sha
                ):
                    raise GeometryCacheCorruptionError(
                        "logical payload hash is invalid"
                    )
                logical_manifest = {
                    name: item
                    for name, item in manifest.items()
                    if name != "logical_payload_sha256"
                }
                if logical_sha != _sha256(_canonical_json(logical_manifest)):
                    raise GeometryCacheCorruptionError("manifest hash mismatch")
                expected_entry = (
                    self._frozen_binding.expected_artifacts.get(identity.key)
                    if self._frozen_binding is not None
                    else None
                )
                if self._frozen_binding is not None and expected_entry is None:
                    raise GeometryCacheCorruptionError(
                        "cache key is absent from frozen receipt"
                    )
                expected_logical_sha = (
                    expected_entry["logical_payload_sha256"]
                    if expected_entry is not None
                    else None
                )
                if expected_logical_sha is not None and not hmac.compare_digest(
                    logical_sha, expected_logical_sha
                ):
                    raise GeometryCacheCorruptionError(
                        "cache payload differs from frozen receipt"
                    )
                if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
                    raise GeometryCacheCorruptionError("cache schema mismatch")
                if manifest.get("identity") != identity.to_dict():
                    raise GeometryCacheCorruptionError("cache identity mismatch")
                metadata = manifest.get("metadata")
                if not isinstance(metadata, dict):
                    raise GeometryCacheCorruptionError(
                        "cache metadata must be an object"
                    )
                try:
                    portable_metadata = _portable(metadata)
                except (TypeError, ValueError) as exc:
                    raise GeometryCacheCorruptionError(
                        "cache metadata is not portable"
                    ) from exc
                if portable_metadata != metadata:
                    raise GeometryCacheCorruptionError(
                        "cache metadata changed under portability normalization"
                    )
                if len(_canonical_json(metadata)) > self.limits.max_metadata_bytes:
                    raise GeometryCacheCorruptionError(
                        "cache metadata exceeds byte bound"
                    )
                specs = manifest.get("arrays")
                if (
                    not isinstance(specs, list)
                    or len(specs) > self.limits.max_array_count
                ):
                    raise GeometryCacheCorruptionError("array manifest is invalid")
                expected_names = set()
                validated_specs = []
                declared_total = 0
                for spec in specs:
                    if not isinstance(spec, dict) or set(spec) != {
                        "name",
                        "dtype",
                        "shape",
                        "nbytes",
                        "sha256",
                    }:
                        raise GeometryCacheCorruptionError(
                            "array specification is invalid"
                        )
                    name = spec["name"]
                    if (
                        not isinstance(name, str)
                        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name)
                        or name == _MANIFEST_NAME
                        or name in expected_names
                    ):
                        raise GeometryCacheCorruptionError(
                            "array name is invalid or duplicated"
                        )
                    dtype_text = spec["dtype"]
                    try:
                        dtype = np.dtype(dtype_text)
                    except (TypeError, ValueError) as exc:
                        raise GeometryCacheCorruptionError(
                            "array dtype is invalid"
                        ) from exc
                    if dtype.hasobject or dtype.fields is not None:
                        raise GeometryCacheCorruptionError(
                            "object or structured cache arrays are forbidden"
                        )
                    shape = spec["shape"]
                    if (
                        not isinstance(shape, list)
                        or len(shape) > 32
                        or any(
                            isinstance(part, bool)
                            or not isinstance(part, int)
                            or part < 0
                            for part in shape
                        )
                    ):
                        raise GeometryCacheCorruptionError("array shape is invalid")
                    elements = math.prod(shape) if shape else 1
                    if elements > self.limits.max_array_elements:
                        raise GeometryCacheCorruptionError(
                            "cached array exceeds element bound"
                        )
                    nbytes = spec["nbytes"]
                    if (
                        isinstance(nbytes, bool)
                        or not isinstance(nbytes, int)
                        or nbytes != elements * dtype.itemsize
                    ):
                        raise GeometryCacheCorruptionError(
                            "array byte declaration is invalid"
                        )
                    digest = spec["sha256"]
                    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                        raise GeometryCacheCorruptionError(
                            "array hash declaration is invalid"
                        )
                    declared_total += nbytes
                    if declared_total > self.limits.max_total_array_bytes:
                        raise GeometryCacheCorruptionError(
                            "cached arrays exceed total byte bound"
                        )
                    expected_names.add(name)
                    validated_specs.append(spec)
                if manifest.get("total_array_bytes") != declared_total:
                    raise GeometryCacheCorruptionError(
                        "cached byte declaration mismatch"
                    )
                if set(inspected) != expected_names | {_MANIFEST_NAME}:
                    raise GeometryCacheCorruptionError(
                        "cache members mismatch manifest"
                    )
                for spec in validated_specs:
                    header = inspected[spec["name"]]
                    if (
                        header.dtype.str != spec["dtype"]
                        or list(header.shape) != spec["shape"]
                        or header.payload_bytes != spec["nbytes"]
                    ):
                        raise GeometryCacheCorruptionError(
                            "NPY header disagrees with manifest"
                        )
                arrays = {}
                total = 0
                for spec in validated_specs:
                    value = _read_npy_payload(archive, inspected[spec["name"]])
                    if (
                        value.dtype.str != spec["dtype"]
                        or list(value.shape) != spec["shape"]
                        or value.nbytes != spec["nbytes"]
                        or _sha256(_array_bytes(value)) != spec["sha256"]
                    ):
                        raise GeometryCacheCorruptionError("cached array hash mismatch")
                    total += value.nbytes
                    if (
                        value.size > self.limits.max_array_elements
                        or total > self.limits.max_total_array_bytes
                    ):
                        raise GeometryCacheCorruptionError(
                            "loaded cache arrays exceed read bounds"
                        )
                    arrays[spec["name"]] = value
                if total != manifest.get("total_array_bytes"):
                    raise GeometryCacheCorruptionError("cached byte total mismatch")
                if expected_entry is not None and (
                    len(arrays) != expected_entry["array_count"]
                    or total != expected_entry["array_bytes"]
                ):
                    raise GeometryCacheCorruptionError(
                        "cache array counts or bytes differ from frozen receipt"
                    )
        except GeometryCacheCorruptionError:
            raise
        except (
            BadZipFile,
            EOFError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
            OSError,
            UnicodeError,
        ) as exc:
            raise GeometryCacheCorruptionError("cache container is invalid") from exc
        return GeometryCacheArtifact(
            identity=identity,
            arrays=arrays,
            metadata=metadata,
            logical_payload_sha256=logical_sha,
        )

    def get_or_compute(
        self,
        identity: FragmentCacheIdentity,
        producer: Callable[[], Tuple[Mapping[str, np.ndarray], Mapping[str, Any]]],
    ) -> GeometryCacheLookup:
        try:
            return GeometryCacheLookup(self.get(identity), True)
        except KeyError:
            if self.read_only:
                raise GeometryCacheError(
                    "required fragment is missing from the read-only cache"
                )
            # A small shard-lock set avoids one lock file per fragment while
            # ensuring the same expensive fragment is produced once even when
            # several workers encounter its first miss together.
            lock_root = self.root / ".key-locks"
            lock_root.mkdir(parents=True, exist_ok=True)
            lock_path = lock_root / (identity.key[:2] + ".lock")
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    return GeometryCacheLookup(self.get(identity), True)
                except KeyError:
                    arrays, metadata = producer()
                    return GeometryCacheLookup(
                        self.put(identity, arrays, metadata), False
                    )
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def portable_receipt(
        self, identities: Sequence[FragmentCacheIdentity]
    ) -> Dict[str, Any]:
        if isinstance(identities, (str, bytes)) or not isinstance(identities, Sequence):
            raise TypeError("identities must be a finite sequence")
        by_key = {}
        for identity in identities:
            if not isinstance(identity, FragmentCacheIdentity):
                raise TypeError("every identity must be FragmentCacheIdentity")
            if identity.key in by_key:
                raise GeometryCacheError(
                    "portable receipt cannot repeat a fragment identity"
                )
            by_key[identity.key] = identity
        if self._frozen_binding is not None and set(by_key) != set(
            self._frozen_binding.expected_artifacts
        ):
            raise GeometryCacheError(
                "frozen portable receipt requires the exact reviewed inventory"
            )
        entries = []
        for identity in (by_key[key] for key in sorted(by_key)):
            artifact = self.get(identity)
            entries.append(
                {
                    "key": identity.key,
                    "canonical_mask_sha256": identity.canonical_mask_sha256,
                    "geometry_config_sha256": identity.geometry_config_sha256,
                    "threshold_rule": identity.threshold_rule,
                    "logical_payload_sha256": artifact.logical_payload_sha256,
                    "array_count": len(artifact.arrays),
                    "array_bytes": sum(
                        item.nbytes for item in artifact.arrays.values()
                    ),
                }
            )
        receipt = {
            "schema_version": _RECEIPT_VERSION,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "typed_payload_version": CACHE_TYPED_PAYLOAD_VERSION,
            "cache_mode": (
                "read_only_frozen_bound"
                if self.read_only and self._frozen_binding is not None
                else "read_only_unbound"
                if self.read_only
                else "writable_build"
            ),
            "artifact_count": len(entries),
            "artifacts": entries,
            "path_policy": "no_local_paths_pair_ids_or_labels",
        }
        receipt["content_sha256"] = _sha256(_canonical_json(receipt))
        return receipt

    @classmethod
    def from_frozen_receipt(
        cls,
        root: Path,
        receipt: Mapping[str, Any],
        limits: Optional[GeometryCacheLimits] = None,
        *,
        trust: FrozenReceiptTrust,
    ) -> "GeometryArtifactCache":
        """Open an exact reviewed inventory in immutable read-only mode.

        ``trust`` must be supplied from the experiment/data lock.  The loader
        never derives trust from the receipt itself.  Before returning, it
        proves the exact on-disk inventory and fully reads, hashes, and checks
        every artifact named by that externally authorized receipt.
        """

        if not isinstance(receipt, Mapping):
            raise TypeError("frozen cache receipt must be a mapping")
        if not isinstance(trust, FrozenReceiptTrust):
            raise TypeError("trust must be FrozenReceiptTrust")
        if limits is not None and not isinstance(limits, GeometryCacheLimits):
            raise TypeError("limits must be GeometryCacheLimits")
        effective_limits = limits or GeometryCacheLimits()
        portable_receipt = _portable(dict(receipt))
        if not isinstance(portable_receipt, dict):  # pragma: no cover
            raise GeometryCacheError("frozen cache receipt must be an object")
        observed_file_sha = _sha256(_canonical_json(portable_receipt))
        if not hmac.compare_digest(observed_file_sha, trust.expected_file_sha256):
            raise GeometryCacheError("frozen cache receipt file hash mismatch")
        value = dict(portable_receipt)
        content_sha = value.pop("content_sha256", None)
        if not isinstance(content_sha, str) or not SHA256_RE.fullmatch(content_sha):
            raise GeometryCacheError("frozen cache receipt hash is invalid")
        if not hmac.compare_digest(content_sha, trust.expected_content_sha256):
            raise GeometryCacheError("frozen cache external content hash mismatch")
        if not hmac.compare_digest(content_sha, _sha256(_canonical_json(value))):
            raise GeometryCacheError("frozen cache receipt hash mismatch")
        if set(value) != {
            "schema_version",
            "cache_schema_version",
            "typed_payload_version",
            "cache_mode",
            "artifact_count",
            "artifacts",
            "path_policy",
        }:
            raise GeometryCacheError("frozen cache receipt fields are invalid")
        if (
            value["schema_version"] != _RECEIPT_VERSION
            or value["cache_schema_version"] != CACHE_SCHEMA_VERSION
            or value["typed_payload_version"] != CACHE_TYPED_PAYLOAD_VERSION
            or value["cache_mode"] not in {"writable_build", "read_only_frozen_bound"}
            or value["path_policy"] != "no_local_paths_pair_ids_or_labels"
        ):
            raise GeometryCacheError("frozen cache receipt version mismatch")
        artifacts = value["artifacts"]
        artifact_count = value["artifact_count"]
        if (
            not isinstance(artifacts, list)
            or isinstance(artifact_count, bool)
            or not isinstance(artifact_count, int)
            or artifact_count < 0
            or artifact_count != len(artifacts)
        ):
            raise GeometryCacheError("frozen cache receipt count mismatch")
        expected: Dict[str, Mapping[str, Any]] = {}
        artifact_fields = {
            "key",
            "canonical_mask_sha256",
            "geometry_config_sha256",
            "threshold_rule",
            "logical_payload_sha256",
            "array_count",
            "array_bytes",
        }
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or set(artifact) != artifact_fields:
                raise GeometryCacheError("frozen cache artifact fields are invalid")
            key = artifact["key"]
            for name in (
                "key",
                "canonical_mask_sha256",
                "geometry_config_sha256",
                "logical_payload_sha256",
            ):
                if not isinstance(artifact[name], str) or not SHA256_RE.fullmatch(
                    artifact[name]
                ):
                    raise GeometryCacheError("frozen cache artifact hashes are invalid")
            if (
                not isinstance(artifact["threshold_rule"], str)
                or not artifact["threshold_rule"].strip()
                or isinstance(artifact["array_count"], bool)
                or not isinstance(artifact["array_count"], int)
                or artifact["array_count"] < 0
                or isinstance(artifact["array_bytes"], bool)
                or not isinstance(artifact["array_bytes"], int)
                or artifact["array_bytes"] < 0
            ):
                raise GeometryCacheError("frozen cache artifact metadata is invalid")
            if key in expected:
                raise GeometryCacheError("frozen cache receipt repeats a key")
            if artifact["array_count"] > effective_limits.max_array_count:
                raise GeometryCacheError(
                    "frozen cache artifact array count exceeds bound"
                )
            if artifact["array_bytes"] > effective_limits.max_total_array_bytes:
                raise GeometryCacheError(
                    "frozen cache artifact array bytes exceed bound"
                )
            try:
                FragmentCacheIdentity(
                    canonical_mask_sha256=artifact["canonical_mask_sha256"],
                    threshold_rule=artifact["threshold_rule"],
                    geometry_config_sha256=artifact["geometry_config_sha256"],
                    key=key,
                )
            except (TypeError, ValueError) as exc:
                raise GeometryCacheError(
                    "frozen cache artifact identity is inconsistent"
                ) from exc
            expected[key] = dict(artifact)
        if [artifact["key"] for artifact in artifacts] != sorted(expected):
            raise GeometryCacheError(
                "frozen cache artifacts are not in canonical key order"
            )
        binding = _FrozenCacheBinding(expected_artifacts=expected, trust=trust)
        cache = cls(
            root,
            limits=effective_limits,
            read_only=True,
            _frozen_binding=binding,
        )
        # Eager verification is part of construction: no caller can receive a
        # cache that has merely compared filenames or deferred payload checks.
        for key in sorted(expected):
            entry = expected[key]
            identity = FragmentCacheIdentity(
                canonical_mask_sha256=entry["canonical_mask_sha256"],
                threshold_rule=entry["threshold_rule"],
                geometry_config_sha256=entry["geometry_config_sha256"],
                key=key,
            )
            artifact = cache.get(identity)
            if (
                artifact.logical_payload_sha256 != entry["logical_payload_sha256"]
                or len(artifact.arrays) != entry["array_count"]
                or sum(value.nbytes for value in artifact.arrays.values())
                != entry["array_bytes"]
            ):
                raise GeometryCacheCorruptionError(
                    "frozen cache artifact differs from reviewed receipt"
                )
        cache._frozen_receipt_verified = True
        return cache


__all__ = [
    "CACHE_RECIPE_VERSION",
    "CACHE_SCHEMA_VERSION",
    "CACHE_TYPED_PAYLOAD_VERSION",
    "FragmentCacheIdentity",
    "FrozenReceiptTrust",
    "GeometryArtifactCache",
    "GeometryCacheArtifact",
    "GeometryCacheConflictError",
    "GeometryCacheCorruptionError",
    "GeometryCacheError",
    "GeometryCacheLimits",
    "GeometryCacheLookup",
    "GeometryCostEstimate",
    "canonical_mask_sha256",
    "canonical_receipt_file_sha256",
    "estimate_uncached_geometry_cost",
    "fragment_cache_identity",
]

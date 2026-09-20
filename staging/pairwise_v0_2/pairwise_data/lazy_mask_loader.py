"""Bounded lazy scalar-mask loading for Pairwise v0.2 archive references.

Archive handles are opened on first use, members are read directly, and no
archive member is extracted.  Synthetic members are checked against their
manifest content hash before decoding.  Historical members use the frozen
two-valued brighter-foreground rule from Pairwise v0.1.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from .training_stream import ArchiveBinding, MaskMemberRef


MASK_LOADER_VERSION = "dunhuang-pairwise-lazy-mask-loader/0.2"
ArchiveSource = Union[str, Path, bytes, bytearray, BinaryIO]


class LazyMaskLoaderError(ValueError):
    """Raised when a lazy member cannot be verified and decoded safely."""


@dataclass(frozen=True)
class ArchiveSourceSpec:
    """Runtime-only map from portable archive identity to a local source.

    ``sha256_verified`` is retained solely so old v0.2 smoke invocations do not
    break.  It is deliberately ignored: the loader hashes the exact registered
    source itself before opening or decoding any member.
    """

    binding: ArchiveBinding
    source: ArchiveSource
    # Deprecated compatibility input.  Never consulted as evidence.
    sha256_verified: bool = False

    def __post_init__(self) -> None:
        if type(self.sha256_verified) is not bool:  # noqa: E721
            raise TypeError("sha256_verified compatibility field must be bool")


@dataclass(frozen=True)
class _ArchiveVerification:
    observed_sha256: str
    verification_mode: str
    byte_count: int


@dataclass(frozen=True)
class ArchiveVerificationReceipt:
    """Portable proof that one registered archive was observed in full.

    The receipt deliberately contains no runtime path.  Callers can therefore
    bind it into a preregistered experiment receipt before constructing a
    model without disclosing where the archive is mounted.
    """

    logical_id: str
    expected_sha256: str
    observed_sha256: str
    byte_count: int
    verification_mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.logical_id, str) or not self.logical_id:
            raise ValueError("archive verification logical_id is required")
        for name in ("expected_sha256", "observed_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError("archive verification SHA-256 is invalid")
        if self.expected_sha256 != self.observed_sha256:
            raise ValueError("archive verification receipt cannot record a mismatch")
        if type(self.byte_count) is not int or self.byte_count <= 0:  # noqa: E721
            raise ValueError("archive verification byte_count must be positive")
        if not isinstance(self.verification_mode, str) or not self.verification_mode:
            raise ValueError("archive verification mode is required")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_id": self.logical_id,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "byte_count": self.byte_count,
            "verification_mode": self.verification_mode,
        }


@dataclass(frozen=True)
class LazyMaskLoaderConfig:
    cache_size: int = 128
    max_cache_pixels: int = 50_000_000
    max_member_bytes: int = 32 * 1024 * 1024
    max_decoded_pixels: int = 100_000_000

    def __post_init__(self) -> None:
        if self.cache_size < 0 or self.max_cache_pixels < 0:
            raise ValueError("cache limits cannot be negative")
        if self.max_member_bytes <= 0 or self.max_decoded_pixels <= 0:
            raise ValueError("decode limits must be positive")


@dataclass
class LazyMaskLoaderStats:
    requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    archive_opens: int = 0
    decoded_masks: int = 0
    decoded_bytes: int = 0
    decoded_pixels: int = 0
    evictions: int = 0
    archive_verifications: int = 0
    archive_verified_bytes: int = 0
    archive_hash_failures: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "requests": int(self.requests),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "archive_opens": int(self.archive_opens),
            "decoded_masks": int(self.decoded_masks),
            "decoded_bytes": int(self.decoded_bytes),
            "decoded_pixels": int(self.decoded_pixels),
            "evictions": int(self.evictions),
            "archive_verifications": int(self.archive_verifications),
            "archive_verified_bytes": int(self.archive_verified_bytes),
            "archive_hash_failures": int(self.archive_hash_failures),
        }


class LazyMaskArchiveLoader:
    """Resolve ``MaskMemberRef`` objects without extracting archive members."""

    def __init__(
        self,
        sources: Mapping[str, ArchiveSourceSpec],
        config: LazyMaskLoaderConfig = LazyMaskLoaderConfig(),
    ) -> None:
        if not sources:
            raise ValueError("at least one archive source is required")
        self.config = config
        self._sources = dict(sources)
        for logical_id, spec in self._sources.items():
            if logical_id != spec.binding.logical_id:
                raise ValueError("source key must equal binding logical_id")
        self._handles: Dict[str, Union[zipfile.ZipFile, tarfile.TarFile]] = {}
        self._buffers: Dict[str, io.BytesIO] = {}
        self._owned_streams: Dict[str, BinaryIO] = {}
        self._prepared_sources: Dict[str, BinaryIO] = {}
        self._verifications: Dict[str, _ArchiveVerification] = {}
        self._member_hashes: Dict[Tuple[str, str, str], str] = {}
        self._cache: "OrderedDict[Tuple[str, str, str, str, str], np.ndarray]" = (
            OrderedDict()
        )
        self._cache_pixels = 0
        self._closed = False
        self.stats = LazyMaskLoaderStats()

    def __enter__(self) -> "LazyMaskArchiveLoader":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        for stream in self._owned_streams.values():
            stream.close()
        self._owned_streams.clear()
        for buffer in self._buffers.values():
            buffer.close()
        self._buffers.clear()
        self._prepared_sources.clear()
        self._cache.clear()
        self._cache_pixels = 0
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise LazyMaskLoaderError("mask loader is closed")

    @staticmethod
    def _hash_seekable_stream(stream: BinaryIO) -> Tuple[str, int]:
        """Hash a seekable binary stream from byte zero and rewind it."""

        try:
            if not stream.seekable() or not stream.readable():
                raise LazyMaskLoaderError(
                    "archive stream must be readable and seekable for verification"
                )
            stream.seek(0, os.SEEK_SET)
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise LazyMaskLoaderError(
                "archive stream is not verifiably seekable"
            ) from exc
        digest = hashlib.sha256()
        byte_count = 0
        try:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise LazyMaskLoaderError("archive stream must yield bytes")
                digest.update(chunk)
                byte_count += len(chunk)
            stream.seek(0, os.SEEK_SET)
        except (OSError, TypeError, ValueError) as exc:
            raise LazyMaskLoaderError("cannot hash registered archive source") from exc
        return digest.hexdigest(), byte_count

    def _verified_source(self, logical_id: str) -> BinaryIO:
        """Return the exact source stream after one observed SHA-256 check."""

        prepared = self._prepared_sources.get(logical_id)
        if prepared is not None:
            return prepared
        spec = self._sources[logical_id]
        source = spec.source
        stream: Optional[BinaryIO] = None
        mode: str
        try:
            if isinstance(source, (bytes, bytearray)):
                # Snapshot mutable bytearrays so verification and archive reads
                # are performed against exactly the same bytes.
                buffer = io.BytesIO(bytes(source))
                self._buffers[logical_id] = buffer
                stream = buffer
                mode = "observed_in_memory_bytes"
            elif isinstance(source, (str, Path)):
                path = Path(source)
                if not path.is_file():
                    raise LazyMaskLoaderError(
                        "registered archive path must be a regular file"
                    )
                stream = path.open("rb")
                self._owned_streams[logical_id] = stream
                mode = "observed_open_file_stream"
            else:
                stream = source
                mode = "observed_seekable_binary_stream"
            observed, byte_count = self._hash_seekable_stream(stream)
        except (LazyMaskLoaderError, OSError, TypeError, ValueError):
            owned = self._owned_streams.pop(logical_id, None)
            if owned is not None:
                owned.close()
            buffer = self._buffers.pop(logical_id, None)
            if buffer is not None:
                buffer.close()
            raise
        if observed != spec.binding.sha256:
            self.stats.archive_hash_failures += 1
            owned = self._owned_streams.pop(logical_id, None)
            if owned is not None:
                owned.close()
            buffer = self._buffers.pop(logical_id, None)
            if buffer is not None:
                buffer.close()
            raise LazyMaskLoaderError("registered archive SHA-256 mismatch")
        verification = _ArchiveVerification(observed, mode, byte_count)
        self._verifications[logical_id] = verification
        self._prepared_sources[logical_id] = stream
        self.stats.archive_verifications += 1
        self.stats.archive_verified_bytes += byte_count
        return stream

    def verify_all_sources(self) -> Tuple[ArchiveVerificationReceipt, ...]:
        """Hash every registered source before any model is constructed.

        This is an explicit preflight API.  It does not open archive members,
        decode masks, or populate the pixel cache.  Subsequent member reads use
        the exact already-verified open streams, so verification is not merely
        a caller assertion.
        """

        self._ensure_open()
        receipts = []
        for logical_id in sorted(self._sources):
            self._verified_source(logical_id)
            verification = self._verifications[logical_id]
            binding = self._sources[logical_id].binding
            receipts.append(
                ArchiveVerificationReceipt(
                    logical_id=logical_id,
                    expected_sha256=binding.sha256,
                    observed_sha256=verification.observed_sha256,
                    byte_count=verification.byte_count,
                    verification_mode=verification.verification_mode,
                )
            )
        return tuple(receipts)

    def _archive(
        self, reference: MaskMemberRef
    ) -> Union[zipfile.ZipFile, tarfile.TarFile]:
        logical_id = reference.binding.logical_id
        try:
            spec = self._sources[logical_id]
        except KeyError as exc:
            raise LazyMaskLoaderError(
                "no runtime source registered for {!r}".format(logical_id)
            ) from exc
        if spec.binding != reference.binding:
            raise LazyMaskLoaderError(
                "runtime archive binding disagrees with reference"
            )
        handle = self._handles.get(logical_id)
        if handle is not None:
            return handle
        source = self._verified_source(logical_id)
        try:
            if reference.binding.archive_format == "zip":
                handle = zipfile.ZipFile(source, mode="r")
            else:
                handle = tarfile.open(fileobj=source, mode="r:*")
        except (
            OSError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
            tarfile.TarError,
        ) as exc:
            raise LazyMaskLoaderError("cannot open registered archive") from exc
        self._handles[logical_id] = handle
        self.stats.archive_opens += 1
        return handle

    def _read_member(self, reference: MaskMemberRef) -> Tuple[bytes, str]:
        archive = self._archive(reference)
        member = reference.archive_member
        if isinstance(archive, zipfile.ZipFile):
            try:
                info = archive.getinfo(member)
            except KeyError as exc:
                raise LazyMaskLoaderError("ZIP member is missing") from exc
            size = info.file_size
            if info.is_dir():
                raise LazyMaskLoaderError("mask member cannot be a directory")
            if size <= 0 or size > self.config.max_member_bytes:
                raise LazyMaskLoaderError("mask member violates byte limit")
            try:
                with archive.open(info, mode="r") as stream:
                    payload = stream.read(self.config.max_member_bytes + 1)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise LazyMaskLoaderError("cannot read ZIP member") from exc
        else:
            try:
                info = archive.getmember(member)
            except KeyError as exc:
                raise LazyMaskLoaderError("TAR member is missing") from exc
            size = info.size
            if not info.isfile() or size <= 0 or size > self.config.max_member_bytes:
                raise LazyMaskLoaderError("mask member violates byte limit")
            stream = archive.extractfile(info)
            if stream is None:
                raise LazyMaskLoaderError("cannot open TAR member")
            try:
                with stream:
                    payload = stream.read(self.config.max_member_bytes + 1)
            except (OSError, tarfile.TarError) as exc:
                raise LazyMaskLoaderError("cannot read TAR member") from exc
        if len(payload) != size or len(payload) > self.config.max_member_bytes:
            raise LazyMaskLoaderError("mask member read is truncated or oversized")
        observed = hashlib.sha256(payload).hexdigest()
        if (
            reference.content_sha256 is not None
            and observed != reference.content_sha256
        ):
            raise LazyMaskLoaderError("mask member content SHA-256 mismatch")
        identity = (
            reference.binding.logical_id,
            reference.binding.sha256,
            reference.archive_member,
        )
        previous = self._member_hashes.get(identity)
        if previous is not None and previous != observed:
            raise LazyMaskLoaderError("archive member changed after verification")
        self._member_hashes[identity] = observed
        return payload, observed

    def _decode(self, payload: bytes, reference: MaskMemberRef) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "PNG":
                    raise LazyMaskLoaderError("mask member content must be PNG")
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise LazyMaskLoaderError("decoded dimensions are invalid")
                if width * height > self.config.max_decoded_pixels:
                    raise LazyMaskLoaderError("decoded mask exceeds pixel limit")
                image.load()
                if reference.threshold_rule == "grayscale_uint8_gt_127":
                    gray = np.asarray(image.convert("L"), dtype=np.uint8)
                    mask = gray > 127
                elif reference.threshold_rule == "binary_brighter_value":
                    values = np.asarray(image)
                    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
                        raise LazyMaskLoaderError(
                            "historical masks must be scalar numeric PNGs"
                        )
                    if not np.all(np.isfinite(values)) or np.any(values < 0):
                        raise LazyMaskLoaderError("historical mask values are invalid")
                    unique = np.unique(values)
                    if len(unique) > 2:
                        raise LazyMaskLoaderError(
                            "historical mask has more than two scalar values"
                        )
                    if len(unique) == 1:
                        mask = np.full(values.shape, bool(unique[0] > 0), dtype=bool)
                    else:
                        mask = values == unique[-1]
                else:  # guarded by MaskMemberRef but retained fail-closed
                    raise LazyMaskLoaderError("unsupported threshold rule")
        except LazyMaskLoaderError:
            raise
        except (OSError, UnidentifiedImageError, ValueError, TypeError) as exc:
            raise LazyMaskLoaderError("cannot decode mask PNG") from exc
        output = np.ascontiguousarray(mask, dtype=bool)
        output.setflags(write=False)
        return output

    def _cache_put(self, key: Tuple[str, str, str, str, str], mask: np.ndarray) -> None:
        if self.config.cache_size == 0 or self.config.max_cache_pixels == 0:
            return
        if int(mask.size) > self.config.max_cache_pixels:
            return
        previous = self._cache.pop(key, None)
        if previous is not None:
            self._cache_pixels -= int(previous.size)
        self._cache[key] = mask
        self._cache.move_to_end(key)
        self._cache_pixels += int(mask.size)
        while (
            len(self._cache) > self.config.cache_size
            or self._cache_pixels > self.config.max_cache_pixels
        ):
            _, removed = self._cache.popitem(last=False)
            self._cache_pixels -= int(removed.size)
            self.stats.evictions += 1

    def __call__(self, reference: Any) -> np.ndarray:
        self._ensure_open()
        if not isinstance(reference, MaskMemberRef):
            raise TypeError("mask loader requires MaskMemberRef")
        self.stats.requests += 1
        logical_id = reference.binding.logical_id
        try:
            spec = self._sources[logical_id]
        except KeyError as exc:
            raise LazyMaskLoaderError(
                "no runtime source registered for {!r}".format(logical_id)
            ) from exc
        if spec.binding != reference.binding:
            raise LazyMaskLoaderError(
                "runtime archive binding disagrees with reference"
            )
        # Verification is mandatory even on a potential cache path.
        self._verified_source(logical_id)
        member_identity = (
            logical_id,
            reference.binding.sha256,
            reference.archive_member,
        )
        known_member_sha = self._member_hashes.get(member_identity)
        if (
            known_member_sha is not None
            and reference.content_sha256 is not None
            and known_member_sha != reference.content_sha256
        ):
            raise LazyMaskLoaderError("mask member content SHA-256 mismatch")
        key = None
        if known_member_sha is not None:
            key = (
                logical_id,
                reference.binding.sha256,
                reference.archive_member,
                known_member_sha,
                reference.threshold_rule,
            )
        cached = None if key is None else self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats.cache_hits += 1
            return cached
        self.stats.cache_misses += 1
        payload, observed_member_sha = self._read_member(reference)
        mask = self._decode(payload, reference)
        self.stats.decoded_masks += 1
        self.stats.decoded_bytes += len(payload)
        self.stats.decoded_pixels += int(mask.size)
        key = (
            logical_id,
            reference.binding.sha256,
            reference.archive_member,
            observed_member_sha,
            reference.threshold_rule,
        )
        self._cache_put(key, mask)
        return mask

    def provenance(self) -> Mapping[str, Any]:
        return {
            "loader_version": MASK_LOADER_VERSION,
            "archive_member_access": "lazy_read_only_no_extraction",
            "registered_archives": [
                {
                    "logical_id": spec.binding.logical_id,
                    "format": spec.binding.archive_format,
                    "expected_sha256": spec.binding.sha256,
                    "observed_sha256": (
                        None
                        if spec.binding.logical_id not in self._verifications
                        else self._verifications[
                            spec.binding.logical_id
                        ].observed_sha256
                    ),
                    "verification_mode": (
                        "not_yet_observed"
                        if spec.binding.logical_id not in self._verifications
                        else self._verifications[
                            spec.binding.logical_id
                        ].verification_mode
                    ),
                    "verified_byte_count": (
                        0
                        if spec.binding.logical_id not in self._verifications
                        else self._verifications[spec.binding.logical_id].byte_count
                    ),
                }
                for spec in sorted(
                    self._sources.values(), key=lambda item: item.binding.logical_id
                )
            ],
            "stats": self.stats.to_dict(),
        }


__all__ = [
    "ArchiveSourceSpec",
    "ArchiveVerificationReceipt",
    "LazyMaskArchiveLoader",
    "LazyMaskLoaderConfig",
    "LazyMaskLoaderError",
    "LazyMaskLoaderStats",
    "MASK_LOADER_VERSION",
]

"""Create a deterministic, CRC-verified Pairwise mask subset ZIP.

The full ``dunhuang_datas.zip`` is only a source container: unrelated
``full_images`` members can be corrupt and the external-volume full-file hash
has been observed to be unstable.  This builder therefore reads only the
Pairwise v0.2 assets, verifies every selected source member through
``ZipExtFile`` CRC checks, and writes a canonical subset on a stable local
filesystem with fixed ZIP metadata.

No member is extracted to the filesystem.  At most one streaming copy buffer
is held in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, Mapping, Optional, Sequence


SCHEMA_VERSION = "dunhuang-pairwise-canonical-mask-subset/0.2"
BUILDER_VERSION = "canonical-mask-subset-builder/0.2"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FIXED_FILE_MODE = 0o100644
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class SyntheticSubsetError(RuntimeError):
    """Raised when a canonical subset cannot be proven complete and stable."""


def _selected_scope(name: str) -> Optional[str]:
    if name.startswith("output/voronoi_masks/"):
        return "voronoi_masks"
    if name.startswith("output/resize_edges/"):
        return "resize_edges"
    if name.startswith("src/"):
        return "source_code"
    if name.startswith("output/datasets/") and name.endswith("/label.csv"):
        return "legacy_labels"
    return None


def _safe_member(name: str) -> bool:
    member = PurePosixPath(name)
    return (
        bool(name)
        and not member.is_absolute()
        and ".." not in member.parts
        and "\\" not in name
        and not WINDOWS_ABSOLUTE_RE.match(name)
    )


def _sha256_stream(stream: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _copy_member(
    source: zipfile.ZipFile,
    source_info: zipfile.ZipInfo,
    target: zipfile.ZipFile,
    *,
    chunk_size: int,
) -> Dict[str, Any]:
    target_info = zipfile.ZipInfo(source_info.filename, date_time=FIXED_ZIP_TIMESTAMP)
    target_info.compress_type = zipfile.ZIP_DEFLATED
    target_info.create_system = 3
    target_info.external_attr = FIXED_FILE_MODE << 16
    target_info.internal_attr = 0
    target_info.comment = b""
    target_info.extra = b""

    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open(source_info, "r") as reader, target.open(
            target_info, "w", force_zip64=True
        ) as writer:
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                writer.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
    except (zipfile.BadZipFile, EOFError, OSError, RuntimeError) as error:
        raise SyntheticSubsetError(
            "selected source member failed CRC/decompression: {} ({})".format(
                source_info.filename, error
            )
        ) from error
    if copied != source_info.file_size:
        raise SyntheticSubsetError(
            "selected source member length mismatch: {} expected={} copied={}".format(
                source_info.filename, source_info.file_size, copied
            )
        )
    return {
        "archive_member": source_info.filename,
        "bytes": copied,
        "sha256": digest.hexdigest(),
        "source_zip_crc32": "{:08x}".format(source_info.CRC),
    }


def _verify_subset_members(path: Path) -> Dict[str, Any]:
    count = 0
    total_bytes = 0
    with zipfile.ZipFile(path, "r") as archive:
        if archive.comment:
            raise SyntheticSubsetError("canonical subset ZIP comment must be empty")
        names = archive.namelist()
        if names != sorted(names):
            raise SyntheticSubsetError("canonical subset members are not sorted")
        if len(names) != len(set(names)):
            raise SyntheticSubsetError("canonical subset has duplicate member names")
        for info in archive.infolist():
            if info.is_dir() or _selected_scope(info.filename) is None:
                raise SyntheticSubsetError(
                    "canonical subset contains an out-of-scope member: {}".format(
                        info.filename
                    )
                )
            if info.date_time != FIXED_ZIP_TIMESTAMP:
                raise SyntheticSubsetError(
                    "canonical member timestamp is not fixed: {}".format(info.filename)
                )
            if (info.external_attr >> 16) != FIXED_FILE_MODE:
                raise SyntheticSubsetError(
                    "canonical member mode is not fixed: {}".format(info.filename)
                )
            try:
                with archive.open(info, "r") as stream:
                    while stream.read(1024 * 1024):
                        pass
            except (zipfile.BadZipFile, EOFError, OSError, RuntimeError) as error:
                raise SyntheticSubsetError(
                    "canonical member failed CRC/decompression: {} ({})".format(
                        info.filename, error
                    )
                ) from error
            count += 1
            total_bytes += info.file_size
    return {
        "member_count": count,
        "uncompressed_bytes": total_bytes,
        "all_members_crc_verified": True,
    }


def build_canonical_mask_subset(
    *,
    source_archive_path: Path,
    output_zip_path: Path,
    summary_path: Optional[Path] = None,
    local_receipt_path: Optional[Path] = None,
    logical_source_id: str = "local_asset://dunhuang_datas_source_container",
    logical_subset_id: str = "local_asset://pairwise_mask_subset_v0_2",
    compression_level: int = 9,
    chunk_size: int = 1024 * 1024,
) -> Mapping[str, Any]:
    """Build and verify a deterministic subset; fail closed on any bad member."""

    source_archive_path = Path(source_archive_path).expanduser().resolve(strict=True)
    output_zip_path = Path(output_zip_path).expanduser().resolve()
    summary_path = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else output_zip_path.with_suffix(".summary.json")
    )
    local_receipt_path = (
        Path(local_receipt_path).expanduser().resolve()
        if local_receipt_path is not None
        else output_zip_path.with_suffix(".local_path_receipt.json")
    )
    if not zipfile.is_zipfile(source_archive_path):
        raise SyntheticSubsetError("source is not a valid ZIP container")
    if source_archive_path == output_zip_path:
        raise ValueError("source and output ZIP paths must differ")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for logical_id in (logical_source_id, logical_subset_id):
        if "://" not in logical_id or logical_id.startswith(("/", "file://")):
            raise ValueError("logical ids must be portable non-file locators")

    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix="." + output_zip_path.name + ".", dir=str(output_zip_path.parent)
    )
    os.close(fd)
    os.unlink(tmp_name)
    source_counts: Counter[str] = Counter()
    source_uncompressed_bytes = 0
    selected_member_count = 0
    try:
        with zipfile.ZipFile(source_archive_path, "r") as source:
            selected = [
                info
                for info in source.infolist()
                if not info.is_dir() and _selected_scope(info.filename) is not None
            ]
            if not selected:
                raise SyntheticSubsetError("source container has no selected members")
            if any(not _safe_member(info.filename) for info in selected):
                raise SyntheticSubsetError("unsafe selected ZIP member path")
            names = [info.filename for info in selected]
            if len(names) != len(set(names)):
                raise SyntheticSubsetError("duplicate selected ZIP member names")
            selected.sort(key=lambda info: info.filename)
            with zipfile.ZipFile(
                tmp_name,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=compression_level,
                strict_timestamps=True,
            ) as target:
                target.comment = b""
                for info in selected:
                    scope = _selected_scope(info.filename)
                    assert scope is not None
                    _copy_member(
                        source,
                        info,
                        target,
                        chunk_size=chunk_size,
                    )
                    source_counts[scope] += 1
                    source_uncompressed_bytes += info.file_size
                    selected_member_count += 1

        verification = _verify_subset_members(Path(tmp_name))
        subset_hash_observations = [
            _sha256_path(Path(tmp_name)),
            _sha256_path(Path(tmp_name)),
        ]
        if len(set(subset_hash_observations)) != 1:
            raise SyntheticSubsetError(
                "canonical subset SHA-256 is unstable across two reads: {}".format(
                    subset_hash_observations
                )
            )
        os.replace(tmp_name, output_zip_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    subset_sha256 = subset_hash_observations[0]
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "status": "pass",
        "source_container": {
            "logical_id": logical_source_id,
            "filename": source_archive_path.name,
            "bytes": source_archive_path.stat().st_size,
            "canonical": False,
            "full_container_sha256_status": "excluded_unstable_and_unrelated_member_crc_failures",
            "full_container_sha256": None,
        },
        "canonical_subset": {
            "logical_id": logical_subset_id,
            "filename": output_zip_path.name,
            "bytes": output_zip_path.stat().st_size,
            "sha256": subset_sha256,
            "sha256_complete_read_passes": 2,
            "sha256_stable": True,
            "zip_comment_empty": True,
            "fixed_member_timestamp": list(FIXED_ZIP_TIMESTAMP),
            "fixed_member_mode_octal": "100644",
            "compression": "deflate",
            "compression_level": compression_level,
        },
        "selected_scopes": {
            "output/voronoi_masks/": "all_regular_members",
            "output/datasets/**/label.csv": "label_csv_only",
            "output/resize_edges/": "all_regular_members",
            "src/": "all_regular_members",
        },
        "member_count_by_scope": dict(sorted(source_counts.items())),
        "selected_member_count": selected_member_count,
        "selected_uncompressed_bytes": source_uncompressed_bytes,
        "verification": verification,
        "processing_contract": {
            "source_open_mode": "read_only",
            "source_members_extracted": False,
            "copy_mode": "streaming_one_member_buffer",
            "selected_source_members_crc_verified_while_copying": True,
            "canonical_members_crc_verified_after_write": True,
            "unselected_source_corruption_ignored": True,
        },
    }
    _atomic_write(summary_path, _json_bytes(summary))
    receipt = {
        "schema_version": "dunhuang-pairwise-subset-local-path-receipt/0.2",
        "portable": False,
        "source_container_absolute_path": str(source_archive_path),
        "canonical_subset_absolute_path": str(output_zip_path),
        "portable_summary_absolute_path": str(summary_path),
        "canonical_subset_sha256": subset_sha256,
        "portable_summary_sha256": _sha256_path(summary_path),
        "source_container_modified": False,
    }
    _atomic_write(local_receipt_path, _json_bytes(receipt))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--local-receipt", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_canonical_mask_subset(
        source_archive_path=args.source_archive,
        output_zip_path=args.output_zip,
        summary_path=args.summary,
        local_receipt_path=args.local_receipt,
    )
    print(json.dumps(summary["canonical_subset"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from staging.pairwise_v0_2.pairwise_data.synthetic_subset import (
    FIXED_FILE_MODE,
    FIXED_ZIP_TIMESTAMP,
    SyntheticSubsetError,
    build_canonical_mask_subset,
)


def _source_zip(path: Path, *, compression: int = zipfile.ZIP_DEFLATED) -> None:
    members = {
        "full_images/unrelated.jpg": b"unselected",
        "output/datasets/gen2/no_erode/0/label.csv": b"id,neighbors\n0,1\n1,0\n",
        "output/resize_edges/a.png": b"edge",
        "output/voronoi_masks/gen2/no_erode/0/0.png": b"mask-zero",
        "output/voronoi_masks/gen2/no_erode/0/1.png": b"mask-one",
        "src/generator.py": b"print('source')\n",
    }
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name in reversed(list(members)):
            archive.writestr(name, members[name])


def test_subset_is_deterministic_sorted_fixed_and_excludes_unrelated_members(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    _source_zip(source)
    first = tmp_path / "first" / "pairwise_mask_subset.zip"
    second = tmp_path / "second" / "pairwise_mask_subset.zip"
    first_summary = build_canonical_mask_subset(
        source_archive_path=source,
        output_zip_path=first,
    )
    second_summary = build_canonical_mask_subset(
        source_archive_path=source,
        output_zip_path=second,
    )

    assert first.read_bytes() == second.read_bytes()
    assert (
        first_summary["canonical_subset"]["sha256"]
        == hashlib.sha256(first.read_bytes()).hexdigest()
    )
    assert first_summary["canonical_subset"] == second_summary["canonical_subset"]
    assert first_summary["source_container"]["full_container_sha256"] is None
    assert first_summary["verification"]["all_members_crc_verified"] is True
    assert first_summary["member_count_by_scope"] == {
        "legacy_labels": 1,
        "resize_edges": 1,
        "source_code": 1,
        "voronoi_masks": 2,
    }

    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert "full_images/unrelated.jpg" not in names
        assert all(info.date_time == FIXED_ZIP_TIMESTAMP for info in archive.infolist())
        assert all(
            info.external_attr >> 16 == FIXED_FILE_MODE for info in archive.infolist()
        )
        assert archive.testzip() is None


def test_subset_summary_is_portable_and_local_paths_are_receipt_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "private" / "source.zip"
    source.parent.mkdir()
    _source_zip(source)
    output = tmp_path / "out" / "pairwise_mask_subset.zip"
    summary_path = tmp_path / "out" / "subset_summary.json"
    receipt_path = tmp_path / "out" / "subset_receipt.json"
    build_canonical_mask_subset(
        source_archive_path=source,
        output_zip_path=output,
        summary_path=summary_path,
        local_receipt_path=receipt_path,
    )

    summary_text = summary_path.read_text(encoding="utf-8")
    receipt_text = receipt_path.read_text(encoding="utf-8")
    assert str(source.resolve()) not in summary_text
    assert str(output.resolve()) not in summary_text
    assert str(source.resolve()) in receipt_text
    assert str(output.resolve()) in receipt_text
    assert json.loads(summary_text)["status"] == "pass"


def test_selected_member_crc_failure_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.zip"
    _source_zip(source, compression=zipfile.ZIP_STORED)
    with zipfile.ZipFile(source) as archive:
        info = archive.getinfo("output/voronoi_masks/gen2/no_erode/0/0.png")
        filename_bytes = info.filename.encode("utf-8")
        data_offset = info.header_offset + 30 + len(filename_bytes) + len(info.extra)
    raw = bytearray(source.read_bytes())
    raw[data_offset] ^= 0x01
    source.write_bytes(raw)

    output = tmp_path / "pairwise_mask_subset.zip"
    with pytest.raises(SyntheticSubsetError, match="CRC/decompression"):
        build_canonical_mask_subset(
            source_archive_path=source,
            output_zip_path=output,
        )
    assert not output.exists()

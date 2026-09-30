import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from staging.pairwise_v0_2.pairwise_data.synthetic_manifest import (
    ArchiveExpectations,
    build_synthetic_manifest,
)


def _png(mask: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(stream, format="PNG")
    return stream.getvalue()


def _label(edges: list, fragment_count: int) -> bytes:
    neighbors = {index: set() for index in range(fragment_count)}
    for first, second in edges:
        neighbors[first].add(second)
        neighbors[second].add(first)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["id", "center_x", "center_y", "width", "height", "neighbors"])
    for fragment_id in range(fragment_count):
        writer.writerow(
            [
                fragment_id,
                0,
                0,
                48,
                48,
                ";".join(str(item) for item in sorted(neighbors[fragment_id])),
            ]
        )
    return stream.getvalue().encode("utf-8")


def _fixture_zip(path: Path) -> None:
    connected_a = np.zeros((48, 48), dtype=bool)
    connected_a[8:40, 2:20] = True
    connected_b = np.zeros((48, 48), dtype=bool)
    connected_b[8:40, 21:39] = True

    overlap_a = np.zeros((48, 48), dtype=bool)
    overlap_a[5:35, 5:25] = True
    overlap_b = np.zeros((48, 48), dtype=bool)
    overlap_b[10:40, 24:44] = True

    disconnected_a = np.zeros((48, 48), dtype=bool)
    disconnected_a[2:12, 2:12] = True
    disconnected_b = np.zeros((48, 48), dtype=bool)
    disconnected_b[34:44, 34:44] = True

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        groups = {
            ("gen2", "0"): (connected_a, connected_b),
            ("gen2", "1"): (overlap_a, overlap_b),
            ("gen2", "2"): (disconnected_a, disconnected_b),
        }
        for (family, group_id), masks in groups.items():
            for fragment_id, mask in enumerate(masks):
                archive.writestr(
                    "output/voronoi_masks/{}/no_erode/{}/{}.png".format(
                        family, group_id, fragment_id
                    ),
                    _png(mask),
                )
        archive.writestr(
            "output/datasets/gen2/no_erode/0/label.csv", _label([[0, 1]], 2)
        )
        archive.writestr(
            "output/datasets/gen2/no_erode/1/label.csv", _label([[0, 1]], 2)
        )


def _read_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_small_zip_records_hash_dimensions_overlap_connectivity_and_quarantine(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "source.zip"
    _fixture_zip(archive_path)
    source_hash_before = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    result = build_synthetic_manifest(
        archive_path=archive_path,
        output_dir=tmp_path / "out",
        dataset_id="fixture_v0_2",
        logical_archive_id="local_asset://fixture_zip",
        expectations=ArchiveExpectations(
            group_count=3,
            mask_member_count=6,
            no_erode_group_count=3,
            disconnected_group_count=1,
            quarantine_group_count=1,
            raw_overlap_group_count=1,
            legacy_label_group_count=2,
            legacy_label_mismatch_count=0,
        ),
    )

    assert result.summary["status"] == "pass"
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == source_hash_before
    records = _read_jsonl(result.manifest_path)
    assert len(records) == 3
    assert all(record["fragment_count"] == 2 for record in records)
    assert all(record["no_erode"] is True for record in records)
    assert all(
        member["width"] == 48
        and member["height"] == 48
        and len(member["content_sha256"]) == 64
        for record in records
        for member in record["members"]
    )

    overlap = next(record for record in records if record["source_group_id"] == "1")
    assert overlap["raw_overlap"]["maximum_pairwise_pixels"] == 25
    assert overlap["adjacency_edges"] == [[0, 1]]

    disconnected = next(
        record for record in records if record["source_group_id"] == "2"
    )
    assert disconnected["connectivity"] == {
        "component_count": 2,
        "components": [[0], [1]],
        "connected": False,
        "rule_id": "legacy_neighbor_rule_v1",
    }
    assert disconnected["quarantine"] == {
        "status": "quarantined",
        "reasons": ["disconnected_legacy_adjacency_graph"],
    }


def test_portable_outputs_never_contain_the_absolute_source_path(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "private" / "source.zip"
    archive_path.parent.mkdir()
    _fixture_zip(archive_path)
    result = build_synthetic_manifest(
        archive_path=archive_path,
        output_dir=tmp_path / "out",
        dataset_id="fixture_v0_2",
        logical_archive_id="local_asset://fixture_zip",
        expectations=ArchiveExpectations(legacy_label_mismatch_count=0),
    )

    absolute_source = str(archive_path.resolve())
    assert absolute_source not in result.manifest_path.read_text(encoding="utf-8")
    assert absolute_source not in result.summary_path.read_text(encoding="utf-8")
    assert absolute_source in result.local_receipt_path.read_text(encoding="utf-8")


def test_legacy_label_mismatch_is_quarantined_and_fails_zero_mismatch_gate(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "source.zip"
    _fixture_zip(archive_path)
    rewritten = tmp_path / "mismatch.zip"
    with zipfile.ZipFile(archive_path) as source, zipfile.ZipFile(
        rewritten, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            payload = source.read(info)
            if info.filename == "output/datasets/gen2/no_erode/0/label.csv":
                payload = _label([], 2)
            target.writestr(info.filename, payload)

    result = build_synthetic_manifest(
        archive_path=rewritten,
        output_dir=tmp_path / "out",
        dataset_id="fixture_v0_2",
        logical_archive_id="local_asset://fixture_zip",
        expectations=ArchiveExpectations(legacy_label_mismatch_count=0),
    )
    assert result.summary["status"] == "fail"
    first = _read_jsonl(result.manifest_path)[0]
    assert first["legacy_label_regression"]["matches_legacy_rule"] is False
    assert "legacy_label_regression_mismatch" in first["quarantine"]["reasons"]

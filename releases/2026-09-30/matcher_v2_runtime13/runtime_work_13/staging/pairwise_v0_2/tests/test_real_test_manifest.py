import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from staging.pairwise_v0_2.pairwise_data.real_test_manifest import (
    build_real_test_manifest,
)


CATEGORIES = ("Ground Truth Simple", "Small", "Issue", "No Conjunction")


def _roots(tmp_path: Path):
    main = tmp_path / "main"
    supp = tmp_path / "supp"
    for root in (main, supp):
        for category in CATEGORIES:
            (root / category).mkdir(parents=True)
    return main, supp


def _rgba_mask(width: int, height: int, inset: int = 0) -> Image.Image:
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[..., :3] = 120
    pixels[inset : height - inset or None, inset : width - inset or None, 3] = 255
    return Image.fromarray(pixels, mode="RGBA")


def _write_case(
    root: Path,
    category: str,
    name: str,
    bboxes,
    *,
    rgb_fragment: Optional[int] = None,
    corrupt_fragment: Optional[int] = None,
):
    group = root / category / name
    group.mkdir()
    max_x = max(box[2] for box in bboxes) + 3
    max_y = max(box[3] for box in bboxes) + 3
    lines = [f"{max_x} {max_y}"]
    for fragment_id, (x1, y1, x2, y2) in enumerate(bboxes, 1):
        width, height = x2 - x1, y2 - y1
        path = group / f"{fragment_id}.png"
        if corrupt_fragment == fragment_id:
            path.write_bytes(b"not-a-png")
        elif rgb_fragment == fragment_id:
            Image.new("RGB", (width, height), "white").save(path)
        else:
            _rgba_mask(width, height).save(path)
        lines.append(f"{x1} {y1} {x2} {y2}")
    lines.append(r"E:\private\opaque source.psd")
    (group / "meta_data.txt").write_text("\n".join(lines), encoding="utf-8")
    Image.new("RGB", (max_x, max_y), "white").save(group / "gt.png")
    return group


def _build(tmp_path: Path, main: Path, supp: Path):
    output = tmp_path / "output"
    build_real_test_manifest(main, supp, output)
    manifest = json.loads((output / "real_test_manifest.json").read_text())
    summary = json.loads((output / "real_test_summary.json").read_text())
    receipt = json.loads((output / "local_path_receipt.json").read_text())
    return output, manifest, summary, receipt


def test_two_fragment_case_is_explicit_positive_and_portable(tmp_path):
    main, supp = _roots(tmp_path)
    _write_case(
        main, "Ground Truth Simple", "case-a", [(0, 0, 10, 20), (10, 0, 20, 20)]
    )
    output, manifest, summary, receipt = _build(tmp_path, main, supp)

    assert summary["statistics"]["eligible_case_count"] == 1
    case = manifest["cases"][0]
    assert case["disposition"] == "eligible"
    assert len(case["pair_labels"]) == 1
    assert case["pair_labels"][0]["label"] == "positive"
    assert (
        case["pair_labels"][0]["label_source"]
        == "curated_two_fragment_conjunction_category"
    )
    assert manifest["policy"]["sealed_test"]["status"] == "sealed_external_test"
    assert (
        manifest["policy"]["sealed_test"]["eligible_records_exposed_to_model_training"]
        is False
    )

    portable = (output / "real_test_manifest.json").read_text()
    portable += (output / "real_test_summary.json").read_text()
    assert str(tmp_path) not in portable
    assert "opaque source.psd" not in portable
    receipt_text = (output / "local_path_receipt.json").read_text()
    assert str(tmp_path) in receipt_text
    assert "opaque source.psd" not in receipt_text
    assert receipt["portable_manifest_sha256"] == manifest["manifest_sha256"]


def test_multifragment_labels_only_actual_contacts(tmp_path):
    main, supp = _roots(tmp_path)
    _write_case(
        main,
        "Ground Truth Simple",
        "chain",
        [(0, 0, 20, 20), (20, 0, 40, 20), (40, 0, 60, 20)],
    )
    _, manifest, _, _ = _build(tmp_path, main, supp)
    case = manifest["cases"][0]
    labels = {
        (item["fragment_a"], item["fragment_b"]): item["label"]
        for item in case["pair_labels"]
    }
    assert labels == {(1, 2): "positive", (1, 3): "negative", (2, 3): "positive"}
    assert case["contact_graph"]["positive_connected"] is True
    assert case["disposition"] == "eligible"


def test_exact_content_deduplicates_and_label_conflict_is_isolated(tmp_path):
    main, supp = _roots(tmp_path)
    source = _write_case(
        main, "Ground Truth Simple", "same-a", [(0, 0, 10, 20), (10, 0, 20, 20)]
    )
    duplicate = supp / "No Conjunction" / "same-b"
    duplicate.mkdir()
    for path in source.iterdir():
        (duplicate / path.name).write_bytes(path.read_bytes())

    _, manifest, summary, _ = _build(tmp_path, main, supp)
    assert summary["statistics"]["raw_occurrence_count"] == 2
    assert summary["statistics"]["unique_case_count"] == 1
    assert summary["statistics"]["label_conflict_case_count"] == 1
    case = manifest["cases"][0]
    assert case["label_conflict"] is True
    assert case["disposition"] == "excluded"
    assert {item["collection"] for item in case["occurrences"]} == {"main", "supp"}
    assert {item["category"] for item in case["occurrences"]} == {
        "ground_truth_simple",
        "no_conjunction",
    }


def test_issue_and_missing_alpha_fail_closed(tmp_path):
    main, supp = _roots(tmp_path)
    _write_case(main, "Issue", "issue", [(0, 0, 10, 20), (10, 0, 20, 20)])
    _write_case(
        supp,
        "Ground Truth Simple",
        "no-alpha",
        [(0, 0, 10, 20), (10, 0, 20, 20)],
        rgb_fragment=2,
    )
    _, manifest, summary, _ = _build(tmp_path, main, supp)
    by_category = {case["canonical_category"]: case for case in manifest["cases"]}
    assert by_category["issue"]["disposition"] == "review"
    assert by_category["issue"]["pair_labels"] == []
    assert by_category["ground_truth_simple"]["disposition"] == "excluded"
    assert any(
        "missing_alpha" in error
        for error in by_category["ground_truth_simple"]["data_quality_errors"]
    )
    assert summary["statistics"]["invalid_case_count"] == 1


def test_unrelated_incomplete_directories_are_not_deduplicated(tmp_path):
    main, supp = _roots(tmp_path)
    (main / "Issue" / "empty-a").mkdir()
    (supp / "Issue" / "empty-b").mkdir()
    _, _, summary, _ = _build(tmp_path, main, supp)
    assert summary["statistics"]["raw_occurrence_count"] == 2
    assert summary["statistics"]["unique_case_count"] == 2
    assert summary["statistics"]["invalid_case_count"] == 2

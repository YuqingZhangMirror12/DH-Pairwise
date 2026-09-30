from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from scipy import ndimage

from staging.pairwise_v0_2.pairwise_data import real_dunhuang_representations
from staging.pairwise_v0_2.baselines.real_dunhuang_evaluation import (
    load_strict_real_pair_dataset,
)
from staging.pairwise_v0_2.pairwise_data.real_dunhuang_representations import (
    RealDunhuangRepresentationError,
    load_real_external_test_spec,
    materialize_real_dunhuang_representations,
)


def _alpha(shape: tuple[int, int], inset: int = 0) -> np.ndarray:
    value = np.zeros(shape, dtype=np.uint8)
    value[inset : shape[0] - inset, inset : shape[1] - inset] = 255
    return value


def _write_rgba(path: Path, alpha: np.ndarray, color: tuple[int, int, int]) -> None:
    value = np.empty((*alpha.shape, 4), dtype=np.uint8)
    value[..., :3] = color
    value[..., 3] = alpha
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="RGBA").save(path)


def _write_la(path: Path, alpha: np.ndarray, luminance: int) -> None:
    value = np.empty((*alpha.shape, 2), dtype=np.uint8)
    value[..., 0] = luminance
    value[..., 1] = alpha
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="LA").save(path)


def _fixture_documents(tmp_path: Path):
    tmp_path = tmp_path / "inputs"
    tmp_path.mkdir(parents=True, exist_ok=True)
    recorded_main = tmp_path / "recorded-main"
    recorded_supp = tmp_path / "recorded-supp"
    actual_main = tmp_path / "actual-main"
    actual_supp = tmp_path / "actual-supp"
    cases = []
    local_cases = {}
    pair_counter = 0

    def add_case(
        *,
        uid: str,
        collection: str,
        category: str,
        group: str,
        alphas,
        boxes,
        labels,
        modes=None,
        disposition: str = "eligible",
    ) -> None:
        nonlocal pair_counter
        directory = {
            "ground_truth_simple": "Ground Truth Simple",
            "small": "Small",
            "issue": "Issue",
            "no_conjunction": "No Conjunction",
        }[category]
        actual_root = actual_main if collection == "main" else actual_supp
        recorded_root = recorded_main if collection == "main" else recorded_supp
        fragment_rows = []
        recorded_paths = []
        if modes is None:
            modes = ("RGBA",) * len(alphas)
        for fragment_id, (alpha, box, mode) in enumerate(
            zip(alphas, boxes, modes), start=1
        ):
            relative = Path(directory) / group / "{}.png".format(fragment_id)
            actual_path = actual_root / relative
            if mode == "LA":
                _write_la(actual_path, alpha, luminance=15 + fragment_id)
            else:
                _write_rgba(
                    actual_path,
                    alpha,
                    color=(10 * fragment_id, 40, 200 - 10 * fragment_id),
                )
            recorded_paths.append(str(recorded_root / relative))
            fragment_rows.append(
                {
                    "fragment_id": fragment_id,
                    "has_alpha": True,
                    "alpha_mask_sha256": hashlib.sha256(
                        np.ascontiguousarray(alpha >= 128, dtype=np.bool_).tobytes()
                    ).hexdigest(),
                    "bbox_xyxy": list(box),
                    "size_wh": [alpha.shape[1], alpha.shape[0]],
                }
            )
        occurrence_uid = "occ-" + uid
        pair_rows = []
        for fragment_a, fragment_b, label in labels:
            pair_counter += 1
            pair_rows.append(
                {
                    "pair_uid": "pair-fixture-{:03d}".format(pair_counter),
                    "fragment_a": fragment_a,
                    "fragment_b": fragment_b,
                    "label": label,
                }
            )
        canvas_width = max(int(row[2]) for row in boxes) + 10
        canvas_height = max(int(row[3]) for row in boxes) + 10
        cases.append(
            {
                "case_uid": uid,
                "canonical_collection": collection,
                "canonical_category": category,
                "observed_categories": [category],
                "disposition": disposition,
                "numeric_metadata": {"canvas_wh": [canvas_width, canvas_height]},
                "fragments": fragment_rows,
                "pair_labels": pair_rows,
                "occurrences": [
                    {
                        "occurrence_uid": occurrence_uid,
                        "collection": collection,
                        "category": category,
                    }
                ],
            }
        )
        local_cases[uid] = {
            "occurrences": [
                {
                    "occurrence_uid": occurrence_uid,
                    "collection": collection,
                    "category": category,
                    "fragment_paths": recorded_paths,
                }
            ]
        }

    common_alpha = _alpha((20, 40), inset=2)
    add_case(
        uid="dhcase-main-gts",
        collection="main",
        category="ground_truth_simple",
        group="1",
        alphas=(common_alpha, common_alpha.copy()),
        boxes=((0, 0, 40, 20), (50, 0, 90, 20)),
        labels=((1, 2, "positive"),),
    )
    add_case(
        uid="dhcase-main-small",
        collection="main",
        category="small",
        group="2",
        alphas=(_alpha((20, 40)), _alpha((10, 20)), _alpha((16, 30))),
        boxes=((0, 0, 40, 20), (50, 0, 70, 10), (50, 30, 80, 46)),
        labels=(
            (1, 2, "positive"),
            (1, 3, "negative"),
            (2, 3, "positive"),
        ),
    )
    add_case(
        uid="dhcase-supp-gts",
        collection="supp",
        category="ground_truth_simple",
        group="3",
        alphas=(_alpha((12, 24)), _alpha((12, 24))),
        boxes=((0, 0, 24, 12), (30, 0, 54, 12)),
        labels=((1, 2, "positive"),),
        modes=("LA", "LA"),
    )
    add_case(
        uid="dhcase-main-issue",
        collection="main",
        category="issue",
        group="4",
        alphas=(_alpha((8, 8)), _alpha((8, 8))),
        boxes=((0, 0, 8, 8), (10, 0, 18, 8)),
        labels=((1, 2, "positive"),),
    )
    add_case(
        uid="dhcase-main-no-conjunction",
        collection="main",
        category="no_conjunction",
        group="5",
        alphas=(_alpha((8, 8)), _alpha((8, 8))),
        boxes=((0, 0, 8, 8), (10, 0, 18, 8)),
        labels=((1, 2, "negative"),),
    )
    # Supp/Small is outside the explicit three-source strict population.
    add_case(
        uid="dhcase-supp-small",
        collection="supp",
        category="small",
        group="6",
        alphas=(_alpha((8, 8)), _alpha((8, 8))),
        boxes=((0, 0, 8, 8), (10, 0, 18, 8)),
        labels=((1, 2, "positive"),),
    )

    manifest_without_digest = {
        "schema_version": "pairwise-v0.2-real-external-test/0.1",
        "cases": cases,
    }
    manifest_sha = hashlib.sha256(
        json.dumps(
            manifest_without_digest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {**manifest_without_digest, "manifest_sha256": manifest_sha}
    receipt = {
        "schema_version": "pairwise-v0.2-real-external-test-local-receipt/0.1",
        "portable_manifest_sha256": manifest_sha,
        "dataset_roots": {
            "main": str(recorded_main),
            "supp": str(recorded_supp),
        },
        "cases": local_cases,
    }
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "paths.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return manifest_path, receipt_path, actual_main, actual_supp


def _materialize_fixture(tmp_path: Path) -> Path:
    manifest, receipt, main_root, supp_root = _fixture_documents(tmp_path)
    return materialize_real_dunhuang_representations(
        manifest,
        receipt,
        output_root=tmp_path / "output",
        main_root=main_root,
        supp_root=supp_root,
        target_long_side=100,
        expected_case_count=3,
        expected_fragment_count=7,
        expected_pair_count=5,
        expected_positive_count=4,
        expected_negative_count=1,
    )


def _read_bool(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        assert image.mode == "L"
        value = np.asarray(image)
    assert set(np.unique(value)).issubset({0, 255})
    return value == 255


def test_fixture_materializes_alpha_only_external_test_with_manifest_pair_order(
    tmp_path: Path,
) -> None:
    output = _materialize_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in (output / "external_test" / "pairs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert [row["source_pair_uid"] for row in rows] == [
        "pair-fixture-001",
        "pair-fixture-002",
        "pair-fixture-003",
        "pair-fixture-004",
        "pair-fixture-005",
    ]
    assert all(row["pair_id"].startswith("pair/sha256/") for row in rows)
    assert [row["label"] for row in rows] == [True, True, False, True, True]
    assert {row["split"] for row in rows} == {"external_test"}
    assert {row["canonical_category"] for row in rows} == {
        "ground_truth_simple",
        "small",
    }
    assert not any(
        "issue" in row["case_uid"] or "no-conjunction" in row["case_uid"]
        for row in rows
    )
    assert rows[2]["direction_b_wrt_a"] is None
    assert "bbox_xyxy" not in json.dumps(rows)

    # The two source PNGs have different RGB values but identical alpha.
    first = _read_bool(output / "representations/filled/dhcase-main-gts/1.png")
    second = _read_bool(output / "representations/filled/dhcase-main-gts/2.png")
    assert np.array_equal(first, second)


def test_case_canvas_scale_is_shared_before_tight_crop_and_contour_is_internal(
    tmp_path: Path,
) -> None:
    output = _materialize_fixture(tmp_path)
    filled_root = output / "representations" / "filled"
    contour_root = output / "representations" / "contour10"

    # main/Small canvas is 90x56, so target=100 gives one shared scale of
    # 100/90.  It is not a separate long-side normalization per fragment.
    large = _read_bool(filled_root / "dhcase-main-small/1.png")
    small = _read_bool(filled_root / "dhcase-main-small/2.png")
    assert large.shape == (22, 44)
    assert small.shape == (11, 22)

    for path in filled_root.rglob("*.png"):
        relative = path.relative_to(filled_root)
        filled = _read_bool(path)
        contour = _read_bool(contour_root / relative)
        expected = filled & ~ndimage.binary_erosion(
            filled,
            structure=np.ones((3, 3), dtype=np.bool_),
            iterations=10,
            border_value=0,
        )
        assert filled.shape == contour.shape
        assert np.array_equal(contour, expected)
        assert np.all(contour <= filled)


def test_receipt_declares_exact_population_and_no_rgb_or_large_hash_work(
    tmp_path: Path,
) -> None:
    output = _materialize_fixture(tmp_path)
    receipt = json.loads(
        (output / "external_test" / "receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["split"] == "external_test"
    assert receipt["counts"] == {
        "cases": 3,
        "fragments": 7,
        "pairs": 5,
        "positive_pairs": 4,
        "negative_pairs": 1,
        "filled_png": 7,
        "contour10_png": 7,
    }
    assert receipt["representation"]["rgb_used"] is False
    assert receipt["representation"]["bbox_origin_used"] is False
    assert receipt["representation"]["shared_scale_reference"] == (
        "original case canvas_wh"
    )
    assert receipt["selection"]["same_case_combinations_relabelled"] is False
    assert receipt["integrity_work"]["source_file_hashes_computed"] is False
    assert receipt["integrity_work"]["archive_hashes_computed"] is False


def test_existing_output_is_rejected_without_touching_user_file(tmp_path: Path) -> None:
    manifest, receipt, main_root, supp_root = _fixture_documents(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "user.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(RealDunhuangRepresentationError, match="refusing to overwrite"):
        materialize_real_dunhuang_representations(
            manifest,
            receipt,
            output_root=output,
            main_root=main_root,
            supp_root=supp_root,
            expected_case_count=3,
            expected_fragment_count=7,
            expected_pair_count=5,
            expected_positive_count=4,
            expected_negative_count=1,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output / "external_test").exists()


def test_non_alpha_source_fails_closed_and_removes_partial_output(
    tmp_path: Path,
) -> None:
    manifest, receipt, main_root, supp_root = _fixture_documents(tmp_path)
    broken = main_root / "Ground Truth Simple/1/2.png"
    Image.fromarray(np.zeros((20, 40, 3), dtype=np.uint8), mode="RGB").save(broken)
    output = tmp_path / "output"

    with pytest.raises(RealDunhuangRepresentationError, match="lacks an alpha"):
        materialize_real_dunhuang_representations(
            manifest,
            receipt,
            output_root=output,
            main_root=main_root,
            supp_root=supp_root,
            expected_case_count=3,
            expected_fragment_count=7,
            expected_pair_count=5,
            expected_positive_count=4,
            expected_negative_count=1,
        )

    assert not output.exists()


def test_materializer_recomputes_declared_alpha_mask_digest(tmp_path: Path) -> None:
    manifest, receipt, main_root, supp_root = _fixture_documents(tmp_path)
    changed = main_root / "Ground Truth Simple/1/2.png"
    with Image.open(changed) as image:
        pixels = np.array(image.convert("RGBA"), dtype=np.uint8, copy=True)
    pixels[0, 0, 3] = 0 if pixels[0, 0, 3] >= 128 else 255
    Image.fromarray(pixels, mode="RGBA").save(changed)
    output = tmp_path / "output"

    with pytest.raises(
        RealDunhuangRepresentationError, match="alpha-mask SHA-256 differs"
    ):
        materialize_real_dunhuang_representations(
            manifest,
            receipt,
            output_root=output,
            main_root=main_root,
            supp_root=supp_root,
            expected_case_count=3,
            expected_fragment_count=7,
            expected_pair_count=5,
            expected_positive_count=4,
            expected_negative_count=1,
        )

    assert not output.exists()


def test_portable_manifest_canonical_digest_is_recomputed(tmp_path: Path) -> None:
    manifest_path, receipt_path, main_root, supp_root = _fixture_documents(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["canonical_category"] = "small"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        RealDunhuangRepresentationError, match="canonical SHA-256 differs"
    ):
        load_real_external_test_spec(
            manifest_path,
            receipt_path,
            main_root=main_root,
            supp_root=supp_root,
            expected_case_count=3,
            expected_fragment_count=7,
            expected_pair_count=5,
            expected_positive_count=4,
            expected_negative_count=1,
        )


def test_pair_only_spec_does_not_read_positive_bbox_direction(tmp_path: Path) -> None:
    manifest_path, receipt_path, main_root, supp_root = _fixture_documents(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        for fragment in case["fragments"]:
            fragment.pop("bbox_xyxy", None)
    without_digest = dict(manifest)
    without_digest.pop("manifest_sha256")
    digest = hashlib.sha256(
        json.dumps(
            without_digest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest["manifest_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["portable_manifest_sha256"] = digest
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    spec = load_real_external_test_spec(
        manifest_path,
        receipt_path,
        main_root=main_root,
        supp_root=supp_root,
        expected_case_count=3,
        expected_fragment_count=7,
        expected_pair_count=5,
        expected_positive_count=4,
        expected_negative_count=1,
        derive_positive_direction=False,
    )
    assert all(pair.direction_b_wrt_a is None for pair in spec.pairs)
    with pytest.raises(RealDunhuangRepresentationError, match="lacks a numeric bbox"):
        load_real_external_test_spec(
            manifest_path,
            receipt_path,
            main_root=main_root,
            supp_root=supp_root,
            expected_case_count=3,
            expected_fragment_count=7,
            expected_pair_count=5,
            expected_positive_count=4,
            expected_negative_count=1,
        )


def test_committed_manifest_preserves_all_547_pair_ids_order_and_labels() -> None:
    root = Path(__file__).resolve().parents[1] / "pairwise_data" / "real_test_v0_1"
    manifest_path = root / "real_test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = []
    allowed = {
        ("main", "ground_truth_simple"),
        ("main", "small"),
        ("supp", "ground_truth_simple"),
    }
    for case in manifest["cases"]:
        if (
            case["disposition"] != "eligible"
            or (
                case["canonical_collection"],
                case["canonical_category"],
            )
            not in allowed
        ):
            continue
        expected.extend(
            (pair["pair_uid"], pair["label"] == "positive")
            for pair in case["pair_labels"]
        )

    spec = load_real_external_test_spec(
        manifest_path,
        root / "local_path_receipt.json",
    )
    observed_source = [(pair.source_pair_uid, pair.label) for pair in spec.pairs]

    assert observed_source == expected
    assert len(observed_source) == 547
    assert sum(label for _, label in observed_source) == 508
    assert sum(not label for _, label in observed_source) == 39
    assert all(pair.to_dict()["split"] == "external_test" for pair in spec.pairs)
    assert {(pair.collection, pair.category) for pair in spec.pairs} == {
        ("main", "ground_truth_simple"),
        ("main", "small"),
        ("supp", "ground_truth_simple"),
    }

    # Preserve the pair identity already emitted by real_dunhuang_evaluation,
    # without inheriting its TrainingPairRecord(split="val") compatibility shim.
    existing = load_strict_real_pair_dataset(
        manifest_path,
        root / "local_path_receipt.json",
    )
    try:
        assert [(pair.pair_id, pair.label) for pair in spec.pairs] == [
            (record.pair_id, record.label) for record in existing.records
        ]
    finally:
        existing.mask_loader.close()


def test_cli_forwards_remote_paths_and_reports_materialized_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    receipt = tmp_path / "receipt.json"
    main_root = tmp_path / "Dunhuang Dataset"
    supp_root = tmp_path / "Dunhuang Dataset Supp"
    output_root = tmp_path / "output"
    observed = {}

    def fake_materialize(manifest_path, receipt_path, **kwargs):
        observed.update(
            {
                "manifest": manifest_path,
                "receipt": receipt_path,
                **kwargs,
            }
        )
        summary = output_root / "external_test" / "receipt.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(
            json.dumps(
                {
                    "counts": {
                        "cases": 445,
                        "fragments": 938,
                        "pairs": 547,
                        "positive_pairs": 508,
                        "negative_pairs": 39,
                    }
                }
            ),
            encoding="utf-8",
        )
        return output_root

    monkeypatch.setattr(
        real_dunhuang_representations,
        "materialize_real_dunhuang_representations",
        fake_materialize,
    )
    status = real_dunhuang_representations.main(
        [
            "--manifest",
            str(manifest),
            "--local-path-receipt",
            str(receipt),
            "--main-root",
            str(main_root),
            "--supp-root",
            str(supp_root),
            "--output-root",
            str(output_root),
        ]
    )

    assert status == 0
    assert observed == {
        "manifest": manifest,
        "receipt": receipt,
        "output_root": output_root,
        "main_root": main_root,
        "supp_root": supp_root,
        "target_long_side": 800,
        "contour_width": 10,
    }
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "real_dunhuang_external_test_prepared"
    assert printed["output_root"] == output_root.as_posix()
    assert printed["counts"]["pairs"] == 547

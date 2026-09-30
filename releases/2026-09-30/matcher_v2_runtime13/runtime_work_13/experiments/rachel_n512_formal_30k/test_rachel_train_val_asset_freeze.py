from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import pytest
import numpy as np
from PIL import Image

from experiments.rachel_n512_formal_30k import rachel_train_val_asset_freeze as freeze
from staging.pairwise_v0_2.baselines import rachel_pairingnet_benchmark as pairingnet
from staging.pairwise_v0_2.baselines import rachel_shreddingnet_benchmark as shreddingnet


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _fragment(token: str, lineage: str, group: str) -> Dict[str, object]:
    return {
        "fragment_token": token,
        "parent_group_id": group,
        "split_unit_id": lineage,
        "image_name": lineage,
        "generator": "gen1",
        "fragment_id": token,
        "foreground_area": 100.0,
        "bbox_aspect_ratio": 1.0,
        "model_mask_path": "model/masks_800/{}.png".format(token),
        "contour_path": "model/contours_n512/{}.npz".format(token),
        "metadata": {},
    }


def _pair(
    split: str,
    index: int,
    first: Mapping[str, object],
    second: Mapping[str, object],
    *,
    label: bool,
    negative_origin: Optional[str] = None,
) -> Dict[str, object]:
    pair_id = "{}-pair-{}".format(split, index)
    return {
        "pair_id": pair_id,
        "fragment_a_token": first["fragment_token"],
        "fragment_b_token": second["fragment_token"],
        "label": label,
        "label_origin": "fixture-positive" if label else "fixture-negative",
        "negative_origin": None if label else negative_origin,
        "seam_length_px": 80.0 if label else None,
        "main_training_eligible": True,
        "selection_exclusion_reason": None,
        "translation_a_to_b_rc": [0.0, 0.0] if label else None,
        "correspondence_path": (
            "targets/pairs/{}.npz".format(pair_id) if label else None
        ),
        "metadata": {},
        "split": split,
        "fragment_a": dict(first),
        "fragment_b": dict(second),
    }


def _write_jsonl(path: Path, rows: List[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _make_dataset(root: Path) -> Dict[str, object]:
    fragments = {
        "ta": _fragment("ta", "train-lineage-1", "train-group-1"),
        "tb": _fragment("tb", "train-lineage-1", "train-group-1"),
        "tc": _fragment("tc", "train-lineage-1", "train-group-1"),
        "td": _fragment("td", "train-lineage-2", "train-group-2"),
        "va": _fragment("va", "val-lineage-1", "val-group-1"),
        "vb": _fragment("vb", "val-lineage-1", "val-group-1"),
        "vc": _fragment("vc", "val-lineage-2", "val-group-2"),
    }
    train = [
        _pair("train", 0, fragments["ta"], fragments["tb"], label=True),
        _pair("train", 1, fragments["ta"], fragments["tc"], label=True),
        _pair(
            "train",
            2,
            fragments["tb"],
            fragments["tc"],
            label=False,
            negative_origin="same_folder_hard",
        ),
        _pair(
            "train",
            3,
            fragments["tc"],
            fragments["td"],
            label=False,
            negative_origin="cross_folder_scale_matched",
        ),
    ]
    val = [
        _pair("val", 0, fragments["va"], fragments["vb"], label=True),
        _pair(
            "val",
            1,
            fragments["va"],
            fragments["vc"],
            label=False,
            negative_origin="cross_folder_scale_matched",
        ),
    ]
    for token, row in fragments.items():
        mask = root / str(row["model_mask_path"])
        contour = root / str(row["contour_path"])
        mask.parent.mkdir(parents=True, exist_ok=True)
        contour.parent.mkdir(parents=True, exist_ok=True)
        mask.write_bytes(("mask:" + token).encode("ascii"))
        contour.write_bytes(("contour:" + token).encode("ascii"))
    for row in train + val:
        if row["label"]:
            target = root / str(row["correspondence_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(("target:" + str(row["pair_id"])).encode("ascii"))
    _write_jsonl(root / "pairs/train.jsonl", train)
    _write_jsonl(root / "pairs/val.jsonl", val)
    # It exists to make accidental test access observable.  Its contents are
    # neither parsed nor inventoried by the module under test.
    (root / "pairs/test.jsonl").write_text("DO NOT OPEN\n", encoding="utf-8")
    selected_counts = {
        "train": {
            "positive": 2,
            "same_folder_hard": 1,
            "cross_folder_scale_matched": 1,
        },
        "val": {
            "positive": 1,
            "same_folder_hard": 0,
            "cross_folder_scale_matched": 1,
        },
        "test": {
            "positive": 1,
            "same_folder_hard": 1,
            "cross_folder_scale_matched": 0,
        },
    }
    split_quotas = {
        split: {
            **counts,
            "negative": counts["same_folder_hard"]
            + counts["cross_folder_scale_matched"],
            "total": 2 * counts["positive"],
        }
        for split, counts in selected_counts.items()
    }
    lineage_assignments = {
        "train-lineage-1": "train",
        "train-lineage-2": "train",
        "val-lineage-1": "val",
        "val-lineage-2": "val",
        "test-lineage-1": "test",
    }
    selection_summary = {
        "schema_version": freeze.SELECTION_SCHEMA_VERSION,
        "seed": "fixture-seed",
        "total_pairs": 8,
        "split_quotas": split_quotas,
        "lineage_counts": {"train": 2, "val": 2, "test": 1},
        "selected_counts": selected_counts,
    }
    preprocess_summary = {
        "group_counts": {"processed_group": 5},
        "fragment_count": len(fragments),
        "fragment_counts_by_generator": {"gen1": len(fragments)},
        "candidate_count": 8,
        "candidate_counts_by_generator_label_eligibility": {},
        "positive_seam_length_px_quantiles": {},
        "contour_n512": {
            "count": len(fragments),
            "capped_fraction": 0.0,
            "min": 4,
            "max": 4,
        },
    }
    run_config = {
        "schema_version": freeze.PREPROCESS_SCHEMA_VERSION,
        "source_root": "/fixture/source",
        "output_root": str(root),
        "seed": "fixture-seed",
        "generators": ["gen1"],
        "preprocess_config": {},
        "limit_groups_per_generator": None,
        "split_unit": "CSV image_name source-manuscript lineage",
        "model_input_contract": {
            "rgb_used": False,
            "binary_values_in_memory": [0, 1],
            "binary_values_on_disk": [0, 255],
            "canvas_shape": [800, 800],
            "parent_origin_exposed_to_model": False,
            "contour": "fixture",
        },
    }
    preprocess_receipt = {
        "schema_version": freeze.PREPROCESS_SCHEMA_VERSION,
        "status": freeze.PREPROCESS_STATUS,
        "source_authority": "Rachel RGB JPEG plus colocated label.csv only",
        "shredding_data_read": False,
        "preprocess": preprocess_summary,
        "selection": selection_summary,
        "paths": dict(freeze._EXPECTED_RECEIPT_PATHS),
    }
    _write_json(root / "run_config.json", run_config)
    _write_json(root / "preprocess_receipt.json", preprocess_receipt)
    _write_json(root / "qa/preprocess_summary.json", preprocess_summary)
    _write_json(root / "pairs/summary.json", selection_summary)
    _write_json(root / "pairs/lineage_splits.json", lineage_assignments)
    return {
        "fragments": fragments,
        "train": train,
        "val": val,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_fixture_assets_loader_valid(
    dataset: Path, fixture: Mapping[str, object]
) -> None:
    fragments = fixture["fragments"]
    assert isinstance(fragments, dict)
    points = np.asarray(
        [[100.0, 100.0], [100.0, 200.0], [200.0, 200.0], [200.0, 100.0]],
        dtype=np.float32,
    )
    valid = np.ones(4, dtype=np.bool_)
    for fragment in fragments.values():
        assert isinstance(fragment, dict)
        mask = np.zeros((800, 800), dtype=np.uint8)
        mask[100:301, 100:301] = 255
        Image.fromarray(mask, mode="L").save(
            dataset / str(fragment["model_mask_path"])
        )
        np.savez_compressed(
            dataset / str(fragment["contour_path"]),
            points_rc=points,
            valid=valid,
        )
    for row in list(fixture["train"]) + list(fixture["val"]):
        assert isinstance(row, dict)
        if row["label"]:
            np.savez_compressed(
                dataset / str(row["correspondence_path"]),
                correspondence_indices=np.asarray(
                    [[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64
                ),
                translation_a_to_b_rc=np.asarray([0.0, 0.0], dtype=np.float32),
                translation_a_to_b_xy_cartesian=np.asarray(
                    [0.0, 0.0], dtype=np.float32
                ),
            )


def _promote_fixture_to_formal(
    dataset: Path, fixture: Mapping[str, object]
) -> None:
    fragments = fixture["fragments"]
    assert isinstance(fragments, dict)

    def write_formal_split(split: str) -> None:
        path = dataset / "pairs" / (split + ".jsonl")
        with path.open("w", encoding="utf-8") as stream:
            if split == "train":
                cells = (
                    (12_000, True, None, "ta", "tb"),
                    (6_000, False, "same_folder_hard", "ta", "tb"),
                    (
                        6_000,
                        False,
                        "cross_folder_scale_matched",
                        "ta",
                        "td",
                    ),
                )
            else:
                cells = (
                    (1_500, True, None, "va", "vb"),
                    (750, False, "same_folder_hard", "va", "vb"),
                    (
                        750,
                        False,
                        "cross_folder_scale_matched",
                        "va",
                        "vc",
                    ),
                )
            index = 0
            for count, label, origin, first, second in cells:
                for _ in range(count):
                    row = _pair(
                        split,
                        index,
                        fragments[first],
                        fragments[second],
                        label=label,
                        negative_origin=origin,
                    )
                    if label:
                        row["correspondence_path"] = (
                            "targets/pairs/formal-shared-{}.npz".format(split)
                        )
                    stream.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    index += 1

    write_formal_split("train")
    write_formal_split("val")
    for split in ("train", "val"):
        target = dataset / "targets/pairs/formal-shared-{}.npz".format(split)
        target.write_bytes(("formal-target:" + split).encode("ascii"))
    selection = {
        "schema_version": freeze.SELECTION_SCHEMA_VERSION,
        "seed": freeze.SELECTION_SEED,
        "total_pairs": 30_000,
        "split_quotas": freeze.FORMAL_SPLIT_QUOTAS,
        "lineage_counts": {"train": 2, "val": 2, "test": 1},
        "selected_counts": freeze.FORMAL_SELECTED_COUNTS,
    }
    _write_json(dataset / "pairs/summary.json", selection)
    receipt_path = dataset / "preprocess_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["selection"] = selection
    _write_json(receipt_path, receipt)
    run_path = dataset / "run_config.json"
    run_config = json.loads(run_path.read_text(encoding="utf-8"))
    run_config["seed"] = freeze.SELECTION_SEED
    _write_json(run_path, run_config)


def test_round_trip_binds_all_roles_and_pair_order(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    output = tmp_path / "freeze"
    receipt = freeze.publish_train_val_semantic_asset_freeze(
        dataset, output, require_formal_counts=False
    )
    verified = freeze.verify_train_val_semantic_asset_freeze(
        dataset, output, require_formal_counts=False
    )
    inventory = json.loads(
        (output / freeze.INVENTORY_FILENAME).read_text(encoding="utf-8")
    )
    semantic = inventory["semantic_inventory"]
    assert receipt["schema_version"] == freeze.RECEIPT_SCHEMA_VERSION
    assert verified["status"] == "verified_train_val_semantic_assets_unchanged"
    assert verified["rehash_all_frozen_dataset_bytes"] is True
    assert verified["same_freeze_root_fd_held_for_full_verification"] is True
    assert verified["final_complete_tree_revalidated"] is True
    assert verified["frozen_view_regular_file_link_count_required"] == 1
    for field in (
        "freeze_root_device",
        "freeze_root_inode",
        "view_root_device",
        "view_root_inode",
    ):
        assert receipt["frozen_dataset"][field] == verified[field]
    assert semantic["dataset_scope"]["opened_pair_manifests"] == [
        "pairs/train.jsonl",
        "pairs/val.jsonl",
    ]
    assert semantic["dataset_scope"]["sealed_test_manifest_opened"] is False
    assert set(semantic["unique_files_by_role"]) == {
        "preprocess_run_config",
        "preprocess_receipt",
        "preprocess_summary",
        "selection_summary",
        "selection_lineage_assignments",
        "train_pair_manifest",
        "val_pair_manifest",
        "model_mask",
        "contour_n512",
        "positive_correspondence_target",
    }
    assert set(semantic["ordered_pair_ids_sha256"]) == {"train", "val"}
    assert set(semantic["canonical_ordered_pairs_sha256"]) == {"train", "val"}
    assert all(
        len(value) == 64
        for value in (
            *semantic["ordered_pair_ids_sha256"].values(),
            *semantic["canonical_ordered_pairs_sha256"].values(),
        )
    )


def test_exact_formal_24000_3000_population_can_be_frozen(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    fixture = _make_dataset(dataset)
    _promote_fixture_to_formal(dataset, fixture)
    inventory = freeze.build_train_val_semantic_inventory(dataset)
    semantic = inventory["semantic_inventory"]
    assert semantic["formal_counts_required"] is True
    assert {
        split: {
            key: semantic["counts"][split][key]
            for key in ("rows", "positive", "negative")
        }
        for split in ("train", "val")
    } == freeze.FORMAL_COUNTS


@pytest.mark.parametrize(
    "relative",
    [
        "model/masks_800/ta.png",
        "model/contours_n512/ta.npz",
        "targets/pairs/train-pair-0.npz",
    ],
)
def test_same_manifest_mutated_training_asset_is_rejected(
    tmp_path: Path, relative: str
) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    output = tmp_path / "freeze"
    freeze.publish_train_val_semantic_asset_freeze(
        dataset, output, require_formal_counts=False
    )
    train_before = _sha(dataset / "pairs/train.jsonl")
    val_before = _sha(dataset / "pairs/val.jsonl")
    target = dataset / relative
    original = target.read_bytes()
    replacement = bytes((value ^ 1) for value in original)
    assert len(replacement) == len(original)
    target.write_bytes(replacement)
    assert _sha(dataset / "pairs/train.jsonl") == train_before
    assert _sha(dataset / "pairs/val.jsonl") == val_before
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError,
        match="semantic assets changed",
    ):
        freeze.verify_train_val_semantic_asset_freeze(
            dataset, output, require_formal_counts=False
        )


def test_neither_create_nor_verify_opens_test_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    output = tmp_path / "freeze"
    real_read = freeze._AnchoredDatasetRoot.read
    opened: List[str] = []

    def guarded_read(self, relative, description, *, retain_bytes):
        opened.append(relative)
        if relative.casefold() == "pairs/test.jsonl":
            raise AssertionError("sealed test manifest was opened")
        return real_read(
            self,
            relative,
            description,
            retain_bytes=retain_bytes,
        )

    monkeypatch.setattr(freeze._AnchoredDatasetRoot, "read", guarded_read)
    freeze.publish_train_val_semantic_asset_freeze(
        dataset, output, require_formal_counts=False
    )
    freeze.verify_train_val_semantic_asset_freeze(
        dataset, output, require_formal_counts=False
    )
    assert not any(path.casefold() == "pairs/test.jsonl" for path in opened)


def test_dataset_leaf_replacement_race_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    target_relative = "model/masks_800/ta.png"
    triggered = False

    def inject(event: str, context: Mapping[str, object]) -> None:
        nonlocal triggered
        if (
            not triggered
            and event == "dataset_after_file_hash_before_entry_revalidation"
            and context.get("relative_path") == target_relative
        ):
            triggered = True
            target = dataset / target_relative
            payload = target.read_bytes()
            target.rename(target.with_suffix(".old"))
            target.write_bytes(payload)

    monkeypatch.setattr(freeze, "_race_test_hook", inject)
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError,
        match="replaced|mutated|identity",
    ):
        freeze.build_train_val_semantic_inventory(
            dataset, require_formal_counts=False
        )
    assert triggered is True


def test_dataset_parent_symlink_replacement_race_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    target_relative = "model/masks_800/ta.png"
    triggered = False

    def inject(event: str, context: Mapping[str, object]) -> None:
        nonlocal triggered
        if (
            not triggered
            and event == "dataset_after_file_hash_before_entry_revalidation"
            and context.get("relative_path") == target_relative
        ):
            triggered = True
            parent = dataset / "model/masks_800"
            moved = dataset / "model/masks_800.original"
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)

    monkeypatch.setattr(freeze, "_race_test_hook", inject)
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError,
        match="symlink|replaced|renamed",
    ):
        freeze.build_train_val_semantic_inventory(
            dataset, require_formal_counts=False
        )
    assert triggered is True


def test_output_rename_symlink_race_never_writes_through_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    output = tmp_path / "freeze"
    moved = tmp_path / "freeze.original"
    outside = tmp_path / "outside"
    outside.mkdir()
    triggered = False

    def inject(event: str, context: Mapping[str, object]) -> None:
        nonlocal triggered
        if not triggered and event == "output_after_directory_anchor_before_publication":
            triggered = True
            assert Path(context["output_dir"]) == output
            output.rename(moved)
            output.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(freeze, "_race_test_hook", inject)
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError,
        match="renamed|replaced|symlink",
    ):
        freeze.publish_train_val_semantic_asset_freeze(
            dataset, output, require_formal_counts=False
        )
    assert triggered is True
    assert list(outside.iterdir()) == []
    assert list(moved.iterdir()) == []


def test_verify_rejects_freeze_root_swap_after_initial_tree_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    output = tmp_path / "freeze"
    freeze.publish_train_val_semantic_asset_freeze(
        dataset, output, require_formal_counts=False
    )
    old_output = tmp_path / "freeze.old"
    triggered = False

    def inject(event: str, context: Mapping[str, object]) -> None:
        nonlocal triggered
        if triggered or event != "freeze_after_initial_tree_walk":
            return
        triggered = True
        assert Path(context["freeze_dir"]) == output
        output.chmod(0o755)
        output.rename(old_output)
        replacement_view = output / freeze.FROZEN_DATA_DIRNAME
        replacement_view.mkdir(parents=True)
        old_view = old_output / freeze.FROZEN_DATA_DIRNAME
        for source in sorted(old_view.rglob("*")):
            relative = source.relative_to(old_view)
            destination = replacement_view / relative
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, destination)
        injected_test = replacement_view / "pairs/test.jsonl"
        injected_test.write_text("DO NOT OPEN\n", encoding="utf-8")
        injected_test.chmod(0o444)
        directories = sorted(
            (path for path in output.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            directory.chmod(0o555)
        output.chmod(0o555)
        assert not (output / freeze.RECEIPT_FILENAME).exists()
        assert not (output / freeze.INVENTORY_FILENAME).exists()

    monkeypatch.setattr(freeze, "_race_test_hook", inject)
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError,
        match="renamed|replaced|identity",
    ):
        freeze.verify_train_val_semantic_asset_freeze(
            dataset, output, require_formal_counts=False
        )
    assert triggered is True


def test_verify_rejects_external_hardlink_to_frozen_view_file(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    output = tmp_path / "freeze"
    freeze.publish_train_val_semantic_asset_freeze(
        dataset, output, require_formal_counts=False
    )
    frozen_file = output / freeze.FROZEN_DATA_DIRNAME / "model/masks_800/ta.png"
    outside_alias = tmp_path / "outside-mask-hardlink.png"
    os.link(frozen_file, outside_alias)
    assert frozen_file.stat().st_nlink == 2
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError,
        match="external hard links",
    ):
        freeze.verify_train_val_semantic_asset_freeze(
            dataset, output, require_formal_counts=False
        )


def test_pairingnet_and_shreddingnet_load_only_materialized_view(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    fixture = _make_dataset(dataset)
    _make_fixture_assets_loader_valid(dataset, fixture)
    output = tmp_path / "freeze"
    receipt = freeze.publish_train_val_semantic_asset_freeze(
        dataset, output, require_formal_counts=False
    )
    view = Path(receipt["frozen_dataset"]["absolute_path"])
    assert view == output / freeze.FROZEN_DATA_DIRNAME
    assert not (view / "pairs/test.jsonl").exists()
    assert all(
        path.stat().st_nlink == 1
        for path in view.rglob("*")
        if path.is_file()
    )
    (dataset / "model").rename(dataset / "model.source-removed")
    (dataset / "targets").rename(dataset / "targets.source-removed")
    pair_audit = pairingnet.audit_train_val_population(
        view, require_formal_counts=False
    )
    shred_audit, _, _ = shreddingnet.audit_rachel_train_val(
        view, require_formal_counts=False
    )
    assert pair_audit["sealed_synthetic_accessed"] is False
    assert shred_audit["sealed_test_manifest_opened"] is False
    pairing_sample = pairingnet.RachelPairDataset(view, "train")[0]
    shredding_sample = shreddingnet.RachelPairDataset(view, "train")[0]
    assert pairing_sample.pair_id == "train-pair-0"
    assert shredding_sample.pair_id == pairing_sample.pair_id
    assert pairing_sample.mask_a.shape == (1, 800, 800)
    assert shredding_sample.points_rc_a.shape == (4, 2)


def test_symlinked_training_asset_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    mask = dataset / "model/masks_800/ta.png"
    outside = tmp_path / "outside.png"
    outside.write_bytes(mask.read_bytes())
    mask.unlink()
    mask.symlink_to(outside)
    with pytest.raises(freeze.RachelTrainValAssetFreezeError, match="symlink"):
        freeze.build_train_val_semantic_inventory(
            dataset, require_formal_counts=False
        )


def test_symlinked_intermediate_asset_directory_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    contour_directory = dataset / "model/contours_n512"
    outside = tmp_path / "outside-contours"
    contour_directory.rename(outside)
    contour_directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(freeze.RachelTrainValAssetFreezeError, match="symlink"):
        freeze.build_train_val_semantic_inventory(
            dataset, require_formal_counts=False
        )


def test_manifest_path_escape_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    fixture = _make_dataset(dataset)
    train = fixture["train"]
    assert isinstance(train, list)
    train[0]["fragment_a"]["model_mask_path"] = (
        "model/masks_800/../../../../outside.png"
    )
    _write_jsonl(dataset / "pairs/train.jsonl", train)
    with pytest.raises(freeze.RachelTrainValAssetFreezeError, match="unsafe"):
        freeze.build_train_val_semantic_inventory(
            dataset, require_formal_counts=False
        )


def test_noncanonical_lexical_path_alias_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    fixture = _make_dataset(dataset)
    train = fixture["train"]
    assert isinstance(train, list)
    train[0]["fragment_a"]["model_mask_path"] = "model/masks_800//ta.png"
    _write_jsonl(dataset / "pairs/train.jsonl", train)
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError, match="non-canonical path alias"
    ):
        freeze.build_train_val_semantic_inventory(
            dataset, require_formal_counts=False
        )


def test_distinct_hardlink_aliases_are_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    fixture = _make_dataset(dataset)
    train = fixture["train"]
    assert isinstance(train, list)
    alias = dataset / "model/masks_800/alias.png"
    os.link(dataset / "model/masks_800/ta.png", alias)
    train[0]["fragment_b"]["model_mask_path"] = "model/masks_800/alias.png"
    _write_jsonl(dataset / "pairs/train.jsonl", train)
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError,
        match="alias the same physical file",
    ):
        freeze.build_train_val_semantic_inventory(
            dataset, require_formal_counts=False
        )


def test_distinct_fragment_tokens_cannot_share_one_asset_path(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    fixture = _make_dataset(dataset)
    train = fixture["train"]
    assert isinstance(train, list)
    train[0]["fragment_b"]["model_mask_path"] = "model/masks_800/ta.png"
    _write_jsonl(dataset / "pairs/train.jsonl", train)
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError,
        match="distinct fragment tokens alias one model mask path",
    ):
        freeze.build_train_val_semantic_inventory(
            dataset, require_formal_counts=False
        )


def test_publish_is_no_clobber(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    output = tmp_path / "freeze"
    freeze.publish_train_val_semantic_asset_freeze(
        dataset, output, require_formal_counts=False
    )
    before = {
        name: _sha(output / name)
        for name in (freeze.INVENTORY_FILENAME, freeze.RECEIPT_FILENAME)
    }
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError, match="already exists"
    ):
        freeze.publish_train_val_semantic_asset_freeze(
            dataset, output, require_formal_counts=False
        )
    after = {name: _sha(output / name) for name in before}
    assert after == before


def test_verify_accepts_and_enforces_external_freeze_hash_pins(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    output = tmp_path / "freeze"
    receipt = freeze.publish_train_val_semantic_asset_freeze(
        dataset, output, require_formal_counts=False
    )
    receipt_sha = _sha(output / freeze.RECEIPT_FILENAME)
    semantic_sha = receipt["inventory"]["semantic_inventory_content_sha256"]
    verified = freeze.verify_train_val_semantic_asset_freeze(
        dataset,
        output,
        require_formal_counts=False,
        expected_freeze_receipt_sha256=receipt_sha,
        expected_semantic_inventory_content_sha256=semantic_sha,
    )
    assert verified["freeze_receipt_sha256"] == receipt_sha
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError,
        match="externally pinned SHA-256",
    ):
        freeze.verify_train_val_semantic_asset_freeze(
            dataset,
            output,
            require_formal_counts=False,
            expected_freeze_receipt_sha256="0" * 64,
            expected_semantic_inventory_content_sha256=semantic_sha,
        )


def test_train_validation_lineage_leakage_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    fixture = _make_dataset(dataset)
    val = fixture["val"]
    assert isinstance(val, list)
    for row in val:
        for side in ("fragment_a", "fragment_b"):
            if row[side]["split_unit_id"] == "val-lineage-1":
                row[side]["split_unit_id"] = "train-lineage-1"
                row[side]["image_name"] = "train-lineage-1"
    _write_jsonl(dataset / "pairs/val.jsonl", val)
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError, match="lineage|leakage"
    ):
        freeze.build_train_val_semantic_inventory(
            dataset, require_formal_counts=False
        )


def test_special_file_is_rejected_without_opening_it(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO is unavailable")
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    mask = dataset / "model/masks_800/ta.png"
    mask.unlink()
    os.mkfifo(mask)
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError, match="not a regular file"
    ):
        freeze.build_train_val_semantic_inventory(
            dataset, require_formal_counts=False
        )


def test_preprocess_selection_cross_binding_is_required(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    summary_path = dataset / "pairs/summary.json"
    selection = json.loads(summary_path.read_text(encoding="utf-8"))
    selection["seed"] = "mutated-selection-seed"
    _write_json(summary_path, selection)
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError,
        match="selection summary is not exactly embedded",
    ):
        freeze.build_train_val_semantic_inventory(
            dataset, require_formal_counts=False
        )


def test_formal_counts_are_default_and_fail_closed(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    with pytest.raises(
        freeze.RachelTrainValAssetFreezeError, match="24000/3000"
    ):
        freeze.build_train_val_semantic_inventory(dataset)


def test_verify_rejects_tampered_receipt_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    output = tmp_path / "freeze"
    freeze.publish_train_val_semantic_asset_freeze(
        dataset, output, require_formal_counts=False
    )
    receipt_path = output / freeze.RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["scope"]["sealed_test_manifest_opened"] = True
    receipt_path.chmod(0o644)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)
    with pytest.raises(freeze.RachelTrainValAssetFreezeError):
        freeze.verify_train_val_semantic_asset_freeze(
            dataset, output, require_formal_counts=False
        )

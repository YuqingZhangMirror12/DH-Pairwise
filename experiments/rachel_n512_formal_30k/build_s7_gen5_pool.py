"""CPU-only materialized S7 Gen5 partition pool from canonical TRAIN sources.

The output directory must be new. No released files, GT or splits are changed.
The existing bounded JPEG boundary-ownership normalization is explicitly applied
to derivative masks (2% overlap / 3px second-owner depth; total support fixed).
Original CSV edges label within-source pairs. If their negatives are
insufficient, distinct frozen TRAIN lineages supply explicitly cross-source
negatives, with dustbin targets and no layout GT. Insufficient unique geometry
is a reported shortfall, never a reason to duplicate examples or alter masks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from staging.pairwise_v0_2.pairwise_data.rachel_gen5_partition import (
    GENERATOR, SCHEMA_VERSION as PARTITION_SCHEMA, build_gen5_partition,
    enumerate_gen5_partition_plans, unique_partition_pairs,
)
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import SCHEMA, load_sample, save_sample
from staging.pairwise_v0_2.pairwise_data.rachel_staged_damage_dataset import clean_report
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample, _readonly
from staging.pairwise_v0_2.pairwise_data.rachel_union_augmentation import (
    UnionAugmentationError, UnionConfig, UnionGeometryRejected, _path, normalize_boundary_ownership,
)


POOL_SCHEMA = "rachel-s7-gen5-partition-pool/1"
MANIFEST_NAME = "train_gen5_partition.json"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _rank(seed, value):
    return _digest([seed, value])


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def discover_train_groups(dataset_root, seed):
    """Read only source metadata; never inspect held-out mask/GT artifacts."""
    root = Path(dataset_root).resolve(strict=True)
    splits_path, manifest_path = root / "pairs/lineage_splits.json", root / "manifests/fragments.jsonl"
    splits = json.loads(splits_path.read_text())
    grouped = defaultdict(list)
    with manifest_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("generator") == GENERATOR and splits.get(row.get("split_unit_id")) == "train":
                grouped[(str(row["group_id"]), row["split_unit_id"])].append(row)
    groups = [dict(group_id=key[0], lineage_id=key[1], fragments=grouped[key])
              for key in sorted(grouped, key=lambda x: _rank(seed, x))]
    source = dict(source_root=str(root), source_family="canonical_rachel_release",
        source_fragment_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        lineage_splits_sha256=hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        train_group_count=len(groups), held_out_masks_or_targets_read=False)
    return groups, splits, source


def _sample_geometry(sample):
    masks = []
    for side in "ab":
        mask = np.ascontiguousarray(getattr(sample, "mask_" + side), dtype=np.uint8)
        masks.append(dict(shape=list(mask.shape), sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
    shift = sample.translation_a_to_b_rc.astype(float).tolist() if sample.translation_valid else None
    reverse = [-v if v else 0. for v in shift] if shift is not None else None
    return min(_digest(dict(masks=masks, translation_rc=shift, label=bool(sample.label))),
               _digest(dict(masks=masks[::-1], translation_rc=reverse, label=bool(sample.label))))


def candidate_sample(candidate):
    """Convert true parent-derived targets to the existing unpadded sample API."""
    pair, a, b = candidate.pair, candidate.fragment_a, candidate.fragment_b
    if not pair.main_training_eligible or pair.status != "accepted":
        raise UnionAugmentationError("cannot materialize an ineligible partition pair")
    matches = pair.token_correspondences
    if pair.label:
        distance = np.linalg.norm(b.model_contour[matches[:, 1]].astype(float)
            - a.model_contour[matches[:, 0]].astype(float) - np.asarray(pair.translation_a_to_b_rc), axis=1)
        if not len(distance) or not np.all(np.isfinite(distance)) or np.median(distance) > 2. or np.quantile(distance, .95) > 4.:
            raise UnionGeometryRejected("runtime_contour_translation_residual_gate")
    values = {}
    for side, fragment in (("a", a), ("b", b)):
        mask = np.asarray(fragment.model.model_mask, bool)
        coarse = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(
            (128, 128), getattr(Image, "Resampling", Image).NEAREST)) > 0
        target = np.full(len(fragment.model_contour), -2, np.int64)
        target[fragment.contour_valid] = -1
        if pair.label:
            source, other = (0, 1) if side == "a" else (1, 0)
            target[matches[:, source]] = matches[:, other]
        values.update({"fragment_" + side + "_token": fragment.fragment_id,
            "mask_" + side: _readonly(mask[None], np.float32),
            "coarse_mask_" + side: _readonly(coarse[None], np.float32),
            "points_rc_" + side: _readonly(fragment.model_contour, np.float32),
            "contour_valid_" + side: _readonly(fragment.contour_valid, bool),
            "target_" + side: _readonly(target, np.int64)})
    return RachelPairSample(pair_id=candidate.metadata["pair_id"], label=np.float32(pair.label),
        translation_a_to_b_rc=_readonly(pair.translation_a_to_b_rc if pair.label else np.zeros(2), np.float32),
        translation_a_to_b_xy_cartesian=_readonly(pair.translation_a_to_b_xy if pair.label else np.zeros(2), np.float32),
        translation_valid=np.bool_(pair.label), **values)


def _save_record(output, sample, source_row, provenance, stratum, source_root):
    token = hashlib.sha256(sample.pair_id.encode()).hexdigest()
    relative = "samples/%s/%s.npz" % (token[:2], token)
    report = clean_report(sample, epoch=1)
    report.update(schema_version=POOL_SCHEMA, gen5_partition=provenance,
        inheritance_rule=("new group GT reconstructed from original CSV adjacency and parent-frame contours"
                          if sample.label else "source-labelled negative; valid tokens dustbin, padded tokens ignored"),
        cross_fragment_geometry_used_for_targets=bool(sample.label),
        source_grouping_changed=True, physical_damage_applied=False,
        gt_layout_available=bool(sample.translation_valid))
    save_sample(output / relative, sample, report)
    ratio = min(source_row["fragment_" + s]["foreground_area"] for s in "ab") / max(
        source_row["fragment_" + s]["foreground_area"] for s in "ab")
    return dict(pair_id=sample.pair_id, label=bool(sample.label), artifact_path=relative,
        source_row=source_row, source_root=source_root, source_stratum=stratum,
        area_ratio_band="tiny_lt025" if ratio < .25 else "other", changed_pair=False,
        surviving_correspondences=int((sample.target_a >= 0).sum()),
        inherited_match_count=int((sample.target_a >= 0).sum()),
        geometry_signature=_sample_geometry(sample), partition_provenance=provenance)


def _process_group(job):
    root, output, spec, config, seed = job
    root, output, config = Path(root).resolve(strict=True), Path(output).resolve(strict=True), UnionConfig(**config)
    lineage, group_id, rows = spec["lineage_id"], spec["group_id"], spec["fragments"]
    reasons = Counter()
    if len(rows) != 5 or len({str(row["fragment_id"]) for row in rows}) != 5:
        return [], dict(incomplete_source_group=1), 0
    marker_path = _path(root, "groups/%s/%s.json" % (GENERATOR, group_id))
    marker = json.loads(marker_path.read_text())
    if marker.get("status") != "processed_group" or marker.get("image_name") != lineage:
        raise UnionAugmentationError("group marker disagrees with frozen TRAIN lineage")
    token_to_id = {row["fragment_token"]: str(row["fragment_id"]) for row in rows}
    if len(token_to_id) != 5:
        raise UnionAugmentationError("duplicate original fragment tokens")
    edges = []
    for candidate in marker["candidates"]:
        if candidate["label"]:
            edges.append((token_to_id[candidate["fragment_a_token"]], token_to_id[candidate["fragment_b_token"]]))
    masks = {}
    for row in rows:
        with Image.open(_path(root, row["target_audit"]["parent_mask_path"])) as image:
            values = np.asarray(image.convert("L"))
        if not np.all((values == 0) | (values == 255)):
            raise UnionAugmentationError("canonical audit mask is not binary; no new thresholding is permitted")
        masks[str(row["fragment_id"])] = values > 0
    try:
        masks, normalization = normalize_boundary_ownership(masks)
    except UnionGeometryRejected as error:
        return [], {error.reason: 1}, 0
    partitions = []
    for plan in enumerate_gen5_partition_plans(tuple(masks)):
        try:
            partition = build_gen5_partition(masks, group_id=group_id, lineage_id=lineage,
                lineage_splits={lineage: "train"}, neighbor_edges=edges, plan=plan, config=config)
        except UnionGeometryRejected as error:
            reasons[error.reason] += 1
            continue
        eligible = []
        for item in partition.pairs:
            if item.pair.main_training_eligible and item.pair.status == "accepted":
                eligible.append(item)
            else:
                reasons[item.pair.selection_exclusion_reason or item.pair.status] += 1
        partitions.append(replace(partition, pairs=tuple(eligible)))
    unique = unique_partition_pairs(partitions)
    entries = []
    for item in sorted(unique, key=lambda x: _rank(seed, x.metadata["pair_id"])):
        try:
            sample = candidate_sample(item)
        except UnionGeometryRejected as error:
            reasons[error.reason] += 1
            continue
        endpoints = {}
        for side, fragment in (("a", item.fragment_a), ("b", item.fragment_b)):
            endpoints["fragment_" + side] = dict(fragment_token=fragment.fragment_id,
                foreground_area=fragment.foreground_area, bbox_aspect_ratio=fragment.bbox_aspect_ratio,
                split_unit_id=lineage)
        row = dict(pair_id=sample.pair_id, split="train", label=bool(sample.label), **endpoints,
                   label_origin="gen5_partition_original_csv", generator=GENERATOR, group_id=group_id)
        provenance = dict(item.metadata, boundary_ownership_normalization=normalization,
            boundary_ownership_changed=normalization["applied"],
            original_csv_adjacency_unchanged=True)
        entries.append(_save_record(output, sample, row, provenance,
            "gen5_partition_positive" if sample.label else "gen5_partition_within_negative", str(root)))
    return entries, dict(reasons), len(partitions)


def _cross_negative(first, first_side, second, second_side, *, pair_id):
    values = {}
    for out, sample, side in (("a", first, first_side), ("b", second, second_side)):
        for name in ("mask", "coarse_mask", "points_rc", "contour_valid"):
            values[name + "_" + out] = getattr(sample, name + "_" + side)
        values["fragment_" + out + "_token"] = getattr(sample, "fragment_" + side + "_token")
        target = np.full(len(values["points_rc_" + out]), -2, np.int64)
        target[values["contour_valid_" + out]] = -1
        values["target_" + out] = _readonly(target, np.int64)
    return RachelPairSample(pair_id=pair_id, label=np.float32(0), translation_valid=np.bool_(False),
        translation_a_to_b_rc=_readonly(np.zeros(2), np.float32),
        translation_a_to_b_xy_cartesian=_readonly(np.zeros(2), np.float32), **values)


def fill_cross_negatives(positive_entries, negative_entries, *, output, count, seed, splits, source_root):
    """Bounded round-robin across distinct TRAIN lineages; no same-source relabel."""
    donors = {}
    for entry in positive_entries:
        for side in "ab":
            endpoint = entry["source_row"]["fragment_" + side]
            lineage = endpoint["split_unit_id"]
            if splits.get(lineage) != "train":
                raise UnionAugmentationError("cross-negative donor is outside frozen TRAIN")
            donors.setdefault(endpoint["fragment_token"], (entry, side, endpoint))
    donors = [donors[key] for key in sorted(donors, key=lambda x: _rank(seed, x))]
    existing = {e["geometry_signature"] for e in negative_entries}
    output, result, seen_pairs, reasons = Path(output), list(negative_entries), set(), Counter()
    attempts, maximum_attempts = 0, max(100, (count - len(result)) * 50, len(donors) * 4)

    @lru_cache(maxsize=16)
    def cached(relative):
        return load_sample(output / relative)[0]

    for offset in range(1, len(donors)):
        for index, (left, ls, la) in enumerate(donors):
            if len(result) >= count or attempts >= maximum_attempts:
                return result, dict(reasons), attempts
            right, rs, rb = donors[(index + offset) % len(donors)]
            key = tuple(sorted((la["fragment_token"], rb["fragment_token"])))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            attempts += 1
            if la["split_unit_id"] == rb["split_unit_id"]:
                reasons["cross_negative_same_lineage_skipped"] += 1
                continue
            pair_id = "rachel-gen5-cross-negative-" + _digest(key)[:24]
            sample = _cross_negative(cached(left["artifact_path"]), ls, cached(right["artifact_path"]), rs,
                                     pair_id=pair_id)
            signature = _sample_geometry(sample)
            if signature in existing:
                reasons["duplicate_cross_negative_geometry"] += 1
                continue
            provenance = dict(schema_version=PARTITION_SCHEMA, split="train", pair_id=pair_id,
                variant="cross_lineage_partition_negative", label=False,
                label_source="different original frozen TRAIN lineage IDs, not parent-frame geometric noncontact",
                source_lineage_ids=[la["split_unit_id"], rb["split_unit_id"]],
                source_partitions=[dict(selected_side=side, partition=entry["partition_provenance"])
                                   for entry, side in ((left, ls), (right, rs))],
                original_gt_layout_available=False, translation_inferred=False,
                new_pixels_added=0, holes_filled=False, resized=False, rotation_degrees=0)
            row = dict(pair_id=pair_id, split="train", label=False, fragment_a=la, fragment_b=rb,
                       label_origin="different_frozen_train_lineages", generator=GENERATOR)
            result.append(_save_record(output, sample, row, provenance,
                                       "gen5_partition_cross_lineage_negative", source_root))
            existing.add(signature)
    return result, dict(reasons), attempts


def build_pool(dataset_root, output_root, *, pairs_per_class=1200, workers=2, seed=260915,
               config=UnionConfig()):
    if type(pairs_per_class) is not int or pairs_per_class < 2 or type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("pairs_per_class>=2 for both partition modes, and workers in1..8 are required")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    root, output = Path(dataset_root).resolve(strict=True), Path(output_root).resolve()
    if output == root or root in output.parents:
        raise ValueError("output must not modify the canonical dataset tree")
    groups, splits, source = discover_train_groups(root, seed)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = dict(schema_version=POOL_SCHEMA, partition_schema=PARTITION_SCHEMA, **source,
        seed=seed, pairs_per_class=pairs_per_class, workers=workers, config=asdict(config),
        ordered_patterns=[[1,2,2],[2,1,2],[3,1,1]], equivalent_orderings_deduplicated=True,
        source_files_modified=False, masks_filled_resized_or_rotated=False,
        derivative_boundary_ownership=dict(method="existing normalize_boundary_ownership",
            maximum_overlap_fraction=.02, maximum_second_depth_px=3., total_support_preserved=True),
        required_positive_partition_modes=["1-2-2", "3-1-1"],
        pattern_coverage="reserve at least one unique positive for each mode; 2/1/2 aliases 1/2/2",
        positive_selection="seeded TRAIN group order then seeded unique eligible partition-pair order",
        negative_selection="eligible CSV negatives first; bounded cross-TRAIN-lineage partition fragments if short",
        no_model_or_held_out_selection=True)
    _write_json(output / "protocol.json", protocol)
    positives, negatives, donors, seen, reasons = [], [], [], set(), Counter()
    positive_modes = Counter()
    processed, valid_partitions = 0, 0

    def consume(result):
        nonlocal processed, valid_partitions
        entries, rejected, partition_count = result
        processed += 1
        valid_partitions += partition_count
        reasons.update(rejected)
        for entry in entries:
            if entry["geometry_signature"] in seen:
                reasons["duplicate_pool_geometry"] += 1
                continue
            seen.add(entry["geometry_signature"])
            bucket = positives if entry["label"] else negatives
            if entry["label"]:
                donors.append(entry)
                pattern = sorted(entry["partition_provenance"]["ordered_pattern"])
                mode = "1-2-2" if pattern == [1, 2, 2] else "3-1-1"
                missing = 2 - len(positive_modes)
                if positive_modes[mode] and len(positives) >= pairs_per_class - missing:
                    continue
            if len(bucket) < pairs_per_class:
                bucket.append(entry)
                if entry["label"]:
                    positive_modes[mode] += 1

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(0, len(groups), workers):
            batch = groups[start:start + workers]
            jobs = [(str(root), str(output), spec, asdict(config), seed) for spec in batch]
            for result in executor.map(_process_group, jobs) if executor else map(_process_group, jobs):
                consume(result)
            status = dict(status="running", stage="collect_gen5_train_partitions", processed_groups=processed,
                positive_count=len(positives), within_negative_count=len(negatives), elapsed_s=time.monotonic()-started)
            _write_json(output / "status.json", status)
            print(json.dumps(status), flush=True)
            donor_lineages = {e["source_row"]["fragment_a"]["split_unit_id"] for e in donors}
            if len(positives) >= pairs_per_class and (len(negatives) >= pairs_per_class or len(donor_lineages) >= 2):
                break
    finally:
        if executor:
            executor.shutdown(wait=True)
    negatives, cross_reasons, cross_attempts = fill_cross_negatives(donors, negatives, output=output,
        count=pairs_per_class, seed=seed, splits=splits, source_root=str(root))
    reasons.update(cross_reasons)
    entries = sorted(positives + negatives, key=lambda x: _rank(seed + 1, x["pair_id"]))
    stats = dict(sample_count=len(entries), positive_count=len(positives), negative_count=len(negatives),
        changed_pair_count=0, positives_with_zero_surviving_correspondences=0,
        source_strata=dict(Counter(e["source_stratum"] for e in entries)),
        selected_positive_partition_modes=dict(positive_modes),
        selected_within_ordered_patterns=dict(Counter(
            "-".join(map(str, e["partition_provenance"]["ordered_pattern"])) for e in entries
            if "ordered_pattern" in e["partition_provenance"])),
        selected_within_pattern_aliases=dict(Counter(
            "-".join(map(str, p)) for e in entries for p in e["partition_provenance"].get("ordered_patterns", []))),
        selected_parent_lineages=len({e["source_row"]["fragment_" + s]["split_unit_id"] for e in entries for s in "ab"}))
    complete = (len(positives) == len(negatives) == pairs_per_class and
                all(positive_modes[mode] >= 1 for mode in ("1-2-2", "3-1-1")))
    manifest = dict(schema_version=SCHEMA, split="train", artifact_root=str(output), entries=entries,
                    stats=stats, protocol=protocol, preparation_status="complete" if complete else "shortfall")
    _write_json(output / MANIFEST_NAME, manifest)
    with (output / "provenance.jsonl").open("x", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(dict(pair_id=entry["pair_id"], provenance=entry["partition_provenance"]),
                                    sort_keys=True, allow_nan=False) + "\n")
    summary = dict(schema_version=POOL_SCHEMA, status="complete" if complete else "shortfall",
        manifest=str(output / MANIFEST_NAME), pairs_per_class=pairs_per_class, **stats,
        processed_groups=processed, available_train_groups=len(groups), valid_partitions=valid_partitions,
        rejection_counts=dict(reasons), cross_negative_attempts=cross_attempts,
        validation_or_test_used=False, new_independent_source_manuscripts=False,
        unused_candidate_artifacts_may_exist=True, elapsed_s=time.monotonic()-started)
    _write_json(output / "summary.json", summary)
    _write_json(output / "status.json", summary)
    return summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pairs-per-class", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260915)
    return parser


if __name__ == "__main__":
    result = build_pool(**vars(make_parser().parse_args()))
    print(json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(0 if result["status"] == "complete" else 2)

"""TRAIN-only Gen4/5 unions, with parent-frame seam and translation targets.

This is a new data builder, not a change to a released loader or dataset.
Members are OR-ed without filling holes, closing gaps, erosion, rescaling, or
rotation. Invalid geometry is rejected, never silently repaired. Only the
canonical Rachel release and its existing TRAIN lineages are accepted by the
CLI; external archives and potentially duplicated source families are not read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .rachel_preprocess import (
    RachelFragment, RachelPair, RachelPreprocessConfig, RachelPreprocessError,
    _dense_external_contour, _make_pair, centerpad_mask_with_transform,
    extract_ordered_outer_contour,
)


SCHEMA_VERSION = "rachel-train-union-augmentation/1"
GENERATORS = {"gen4voronoi": 4, "gen4voronoi_1_3": 4, "gen5voronoi_1_1_3": 5}
CROSS = ndimage.generate_binary_structure(2, 1)


class UnionAugmentationError(ValueError):
    """Invalid input or a source outside the frozen TRAIN population."""


class UnionGeometryRejected(UnionAugmentationError):
    """An explicit geometry filter; ``reason`` is suitable for a counter."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class UnionConfig:
    canvas_size: int = 800
    contour_cap: int = 512
    smoothing_sigma: float = 3.0
    minimum_seam_px: int = 64
    minimum_token_matches: int = 8

    def __post_init__(self):
        for name in ("canvas_size", "contour_cap", "minimum_seam_px", "minimum_token_matches"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise UnionAugmentationError(name + " must be a positive integer")
        if self.contour_cap < 4 or not math.isfinite(self.smoothing_sigma) or self.smoothing_sigma < 0:
            raise UnionAugmentationError("invalid contour settings")


@dataclass(frozen=True)
class UnionPlan:
    merged_ids: Tuple[str, ...]
    singleton_id: str
    omitted_ids: Tuple[str, ...] = ()

    @property
    def variant(self):
        return "gen5_omit1_merge3_plus1" if self.omitted_ids else "merge%d_plus1" % len(self.merged_ids)


@dataclass(frozen=True)
class UnionCandidate:
    metadata: dict
    fragment_a: RachelFragment
    fragment_b: RachelFragment
    pair: RachelPair
    missing_parent_mask: np.ndarray


def enumerate_union_plans(fragment_ids: Sequence[str]):
    ids = tuple(sorted(str(x) for x in fragment_ids))
    if len(ids) not in (4, 5) or len(set(ids)) != len(ids):
        raise UnionAugmentationError("exactly four or five distinct source fragments are required")
    plans = [UnionPlan(tuple(x for x in ids if x != singleton), singleton) for singleton in ids]
    if len(ids) == 5:
        plans.extend(UnionPlan(tuple(x for x in ids if x not in (omitted, singleton)), singleton, (omitted,))
                     for omitted in ids for singleton in ids if singleton != omitted)
    return tuple(plans)


def _binary(value, shape, name):
    mask = np.asarray(value)
    if mask.shape != shape or not np.all((mask == 0) | (mask == 1)) or not mask.any():
        raise UnionAugmentationError(name + " must be a nonempty binary parent-canvas mask")
    return np.ascontiguousarray(mask, dtype=bool)


def _simple_mask(mask, name):
    if ndimage.label(mask, structure=CROSS)[1] != 1:
        raise UnionGeometryRejected(name + "_disconnected")
    # Explicitly test holes; do NOT replace mask with the filled result.
    if not np.array_equal(ndimage.binary_fill_holes(mask, structure=CROSS), mask):
        raise UnionGeometryRejected(name + "_has_hole")


def _touches_exterior(fragment, parent):
    exterior = ~ndimage.binary_fill_holes(parent, structure=CROSS)
    return bool(np.any(fragment & ndimage.binary_dilation(exterior, structure=CROSS, border_value=1)))


def normalize_boundary_ownership(parent_masks, *, maximum_overlap_fraction=.02, maximum_second_depth_px=3.0):
    """Assign only multiply-owned boundary pixels; preserve total support.

    This explicit JPEG-mask normalization is NOT erosion, gap repair, or an
    alteration to the source release. Exclusive pixels are unchanged. Stable
    lexicographic fragment IDs break equal interior-distance ties.
    """
    ids = sorted(parent_masks)
    stack = np.stack([np.asarray(parent_masks[x], bool) for x in ids])
    overlap, support = stack.sum(0) > 1, stack.any(0)
    fraction = float(overlap.sum() / max(1, support.sum()))
    if fraction > maximum_overlap_fraction:
        raise UnionGeometryRejected("ownership_overlap_fraction_exceeds_2_percent")
    maximum_depth = 0.0
    output = stack.copy()
    if overlap.any():
        depths = np.stack([ndimage.distance_transform_edt(mask) for mask in stack])
        maximum_depth = float(np.sort(depths[:, overlap], axis=0)[-2].max())
        if maximum_depth > maximum_second_depth_px:
            raise UnionGeometryRejected("ownership_second_depth_exceeds_3px")
        winner = depths[:, overlap].argmax(0)
        output[:, overlap] = np.arange(len(ids))[:, None] == winner[None, :]
    if not np.array_equal(output.any(0), support) or np.any(output.sum(0) > 1):
        raise UnionAugmentationError("boundary ownership did not preserve support")
    offset_deltas = {}
    for i, key in enumerate(ids):
        before = centerpad_mask_with_transform(stack[i], canvas_size=stack.shape[1])
        after = centerpad_mask_with_transform(output[i], canvas_size=stack.shape[1])
        offset_deltas[key] = (np.asarray(after.parent_to_model_offset_rc)
                             - np.asarray(before.parent_to_model_offset_rc)).tolist()
    return {key: output[i] for i, key in enumerate(ids)}, dict(
        method="max_interior_distance_overlap_only_stable_id_tie",
        applied=bool(overlap.any()), overlap_px=int(overlap.sum()), overlap_fraction=fraction,
        second_owner_depth_max_px=maximum_depth, total_support_unchanged=True,
        exclusive_pixels_unchanged=True, new_pixels_added=0, erosion_applied=False,
        parent_coordinate_frame_unchanged=True,
        fragment_centerpad_offset_delta_rc=offset_deltas,
        target_translation="rederived_from_new_union_and_singleton_centerpad_offsets")


def _fragment(identifier, parent, config):
    model = centerpad_mask_with_transform(parent, canvas_size=config.canvas_size)
    points, valid = extract_ordered_outer_contour(model.model_mask, cap=config.contour_cap,
                                                 smoothing_sigma=config.smoothing_sigma)
    coordinates = np.argwhere(parent)
    height, width = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
    return RachelFragment(identifier, identifier, Path("generated-union-no-rgb"), parent, model,
                          _dense_external_contour(parent), points, valid, int(parent.sum()), float(width / height))


def build_union_candidate(parent_masks: Mapping[str, np.ndarray], *, generator: str, group_id: str,
                          lineage_id: str, lineage_splits: Mapping[str, str], neighbor_edges,
                          plan: UnionPlan, config: UnionConfig = UnionConfig(), scale: float = 1.0):
    """Build one positive A=union, B=singleton; never read or infer held-out GT.

    ``neighbor_edges`` are the original same-group positive CSV edges, not
    predictions. A surviving original adjacency AND a recovered >=64px seam
    are required. Translation is derived after independently centre-padding A
    and B, not fitted from potentially ambiguous local matches.
    """
    if lineage_splits.get(lineage_id) != "train":
        raise UnionAugmentationError("source lineage is not in the existing frozen TRAIN split")
    if generator not in GENERATORS or len(parent_masks) != GENERATORS[generator]:
        raise UnionAugmentationError("source generator and fragment count disagree")
    if scale != 1.0:
        raise UnionAugmentationError("this union builder performs no resizing; scale must be 1.0")
    ids = set(parent_masks)
    used = tuple(plan.merged_ids) + (plan.singleton_id,) + tuple(plan.omitted_ids)
    if set(used) != ids or len(used) != len(ids) or plan not in enumerate_union_plans(tuple(ids)):
        raise UnionAugmentationError("union plan must partition all original members exactly once")
    edges = {frozenset((str(a), str(b))) for a, b in neighbor_edges}
    if any(len(edge) != 2 or not edge <= ids for edge in edges):
        raise UnionAugmentationError("neighbor edge is not a distinct same-group pair")
    if not any(frozenset((member, plan.singleton_id)) in edges for member in plan.merged_ids):
        raise UnionGeometryRejected("no_surviving_original_adjacency")
    masks = {key: _binary(value, (config.canvas_size,) * 2, key) for key, value in parent_masks.items()}
    ownership = np.stack(tuple(masks.values())).sum(axis=0)
    if np.any(ownership > 1):
        raise UnionGeometryRejected("source_fragment_overlap")
    parent = ownership > 0
    merged = np.logical_or.reduce([masks[x] for x in plan.merged_ids])
    single = masks[plan.singleton_id].copy()
    missing = np.logical_or.reduce([masks[x] for x in plan.omitted_ids]) if plan.omitted_ids else np.zeros_like(parent)
    _simple_mask(parent, "original_parent")
    _simple_mask(merged, "merged")
    _simple_mask(single, "singleton")
    _simple_mask(merged | single, "retained_parent")
    if not _touches_exterior(single, parent):
        raise UnionGeometryRejected("singleton_not_on_original_outer_boundary")
    if single.sum() >= merged.sum():
        raise UnionGeometryRejected("singleton_is_not_smaller_than_union")
    if plan.omitted_ids and not _touches_exterior(missing, parent):
        raise UnionGeometryRejected("omitted_region_is_an_internal_hole_not_an_open_gap")
    if np.any(missing & (merged | single)):
        raise UnionGeometryRejected("omitted_pixels_were_not_preserved_as_missing")
    a, b = _fragment("union", merged, config), _fragment("singleton", single, config)
    preprocessing = RachelPreprocessConfig(canvas_size=config.canvas_size, contour_cap=config.contour_cap,
        minimum_positive_seam=config.minimum_seam_px, contour_smoothing_sigma=config.smoothing_sigma)
    pair = _make_pair(a, b, label=True, config=preprocessing)
    if not pair.main_training_eligible:
        raise UnionGeometryRejected(pair.selection_exclusion_reason or "unreliable_seam")
    if len(pair.token_correspondences) < config.minimum_token_matches:
        raise UnionGeometryRejected("too_few_true_seam_token_matches")
    matches = pair.token_correspondences
    distances = np.linalg.norm(b.model_contour[matches[:, 1]] - a.model_contour[matches[:, 0]]
                               - np.asarray(pair.translation_a_to_b_rc), axis=1)
    residual_median, residual_p95 = float(np.median(distances)), float(np.quantile(distances, .95))
    if residual_median > 2 or residual_p95 > 4:
        raise UnionGeometryRejected("runtime_contour_translation_residual_gate")
    identity = dict(generator=generator, group_id=str(group_id), lineage_id=lineage_id,
                    merged_ids=list(plan.merged_ids), singleton_id=plan.singleton_id,
                    omitted_ids=list(plan.omitted_ids), scale=scale)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    signature = hashlib.sha256(a.model.model_mask.tobytes() + b.model.model_mask.tobytes()
        + np.asarray(pair.translation_a_to_b_rc, dtype="<f8").tobytes()).hexdigest()
    area_a, area_b = int(merged.sum()), int(single.sum())
    metadata = dict(schema_version=SCHEMA_VERSION, pair_id="rachel-union-" + digest, split="train", label=True,
        source_family="canonical_rachel_release", source_identity=identity, variant=plan.variant,
        parent_members=sorted(ids), merged_parent_members=list(plan.merged_ids),
        singleton_parent_member=plan.singleton_id, omitted_parent_members=list(plan.omitted_ids),
        lineage_id=lineage_id, generator=generator, group_id=str(group_id), scale=scale,
        rotation_degrees=0, resized=False, holes_filled=False, gap_closing_applied=False,
        area_a_px=area_a, area_b_px=area_b, min_area_px=area_b, max_area_px=area_a,
        area_ratio=area_b / area_a, tiny_area_ratio=area_b / area_a < .25,
        original_parent_area_px=int(parent.sum()), omitted_area_px=int(missing.sum()),
        model_offset_a_rc=list(a.model.parent_to_model_offset_rc), model_offset_b_rc=list(b.model.parent_to_model_offset_rc),
        translation_a_to_b_rc=list(pair.translation_a_to_b_rc),
        translation_a_to_b_xy_cartesian=list(pair.translation_a_to_b_xy),
        seam_length_px=pair.seam_length_px, dense_seam_match_count=pair.accepted_seam_match_count,
        token_translation_residual_median_px=residual_median, token_translation_residual_p95_px=residual_p95,
        token_match_count=len(pair.token_correspondences), geometry_signature=signature,
        true_seam_source="surviving_original_CSV_adjacency_and_parent_mask_contours",
        model_inputs_contain_parent_origin=False)
    return UnionCandidate(metadata, a, b, pair, missing)


def select_union_quota(metadata_rows, *, count: int, tiny_count: int, seed: int = 260909):
    """Select distinct geometries, report shortages; do not duplicate to fill.

    For 30,000 training positives with no tiny existing positive, the requested
    minimum is ceil(.15*30000)=4500. That is a measured-area quota, not a claim
    that merging a fixed number of pieces automatically achieves it.
    """
    if type(count) is not int or type(tiny_count) is not int or not 0 <= tiny_count <= count:
        raise UnionAugmentationError("invalid total/tiny quotas")
    ranked = sorted(metadata_rows, key=lambda row: hashlib.sha256((str(seed) + row["pair_id"]).encode()).digest())
    unique, seen = [], set()
    for row in ranked:
        if row.get("split") != "train" or row.get("label") is not True:
            raise UnionAugmentationError("quota input must contain TRAIN positives only")
        if row["geometry_signature"] not in seen:
            seen.add(row["geometry_signature"])
            unique.append(row)
    tiny = [row for row in unique if row["area_ratio"] < .25]
    chosen = tiny[:tiny_count]
    selected_ids = {row["pair_id"] for row in chosen}
    chosen += [row for row in unique if row["pair_id"] not in selected_ids][:count - len(chosen)]
    achieved_tiny = sum(row["area_ratio"] < .25 for row in chosen)
    return dict(status="complete" if len(chosen) == count and achieved_tiny >= tiny_count else "shortfall",
                requested=count, requested_tiny=tiny_count, selected_count=len(chosen),
                selected_tiny=achieved_tiny, available_unique=len(unique), available_tiny=len(tiny),
                pair_ids=[row["pair_id"] for row in chosen],
                selection="seeded rank with measured min/max<.25 quota; no replacement")


def _path(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise UnionAugmentationError("release artifact escapes its canonical root")
    return path


def _save_candidate(destination, candidate):
    identity = candidate.metadata["pair_id"]
    fragments = {}
    for side, fragment in (("a", candidate.fragment_a), ("b", candidate.fragment_b)):
        mask_path = Path("model/masks_800") / identity / (side + ".png")
        contour_path = Path("model/contours_n512") / identity / (side + ".npz")
        for relative in (mask_path, contour_path):
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fragment.model.model_mask.astype(np.uint8) * 255).save(destination / mask_path)
        np.savez_compressed(destination / contour_path, points_rc=fragment.model_contour, valid=fragment.contour_valid)
        fragments["fragment_" + side] = dict(fragment_token=identity + "/" + side,
            model_mask_path=mask_path.as_posix(), contour_path=contour_path.as_posix(),
            foreground_area=fragment.foreground_area, bbox_aspect_ratio=fragment.bbox_aspect_ratio,
            split_unit_id=candidate.metadata["lineage_id"])
    target_path = Path("targets/pairs") / (identity + ".npz")
    (destination / target_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / target_path, correspondence_indices=candidate.pair.token_correspondences,
                        translation_a_to_b_rc=np.asarray(candidate.pair.translation_a_to_b_rc, np.float32),
                        translation_a_to_b_xy_cartesian=np.asarray(candidate.pair.translation_a_to_b_xy, np.float32))
    # Parent origins and omission provenance remain in candidates.jsonl only.
    return dict(pair_id=identity, split="train", label=True, **fragments,
                correspondence_path=target_path.as_posix(), label_origin="train_union_true_seam")


def build_from_release(release_root, output_root, *, max_groups=32, seed=260909, config=UnionConfig(),
                       save_assets=False, tiny_only=False, boundary_ownership=False,
                       shard_index=0, shard_count=1):
    """CPU-only bounded source probe; default32 TRAIN groups, no model work.

    All input file writes are prohibited. The output root must be brand new.
    JSONL metadata is inspected to select TRAIN groups before any mask is read.
    No held-out masks, models, external archives, or source-wide hashes are read.
    """
    root, destination = Path(release_root).resolve(), Path(output_root).resolve()
    if type(max_groups) is not int or max_groups <= 0:
        raise UnionAugmentationError("max_groups must be a positive explicit bound")
    if type(shard_count) is not int or type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise UnionAugmentationError("invalid disjoint group shard")
    if destination == root or root in destination.parents:
        raise UnionAugmentationError("new output must not modify the released dataset tree")
    with (root / "pairs/lineage_splits.json").open() as stream:
        splits = json.load(stream)
    grouped = defaultdict(list)
    with (root / "manifests/fragments.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["generator"] in GENERATORS and splits.get(row["split_unit_id"]) == "train":
                grouped[(row["generator"], str(row["group_id"]), row["split_unit_id"])].append(row)
    keys = sorted(grouped, key=lambda key: hashlib.sha256((str(seed) + repr(key)).encode()).digest())[:max_groups]
    keys = keys[shard_index::shard_count]
    destination.mkdir(parents=True, exist_ok=False)
    model_rows = []
    counts, reasons, rows, seen = Counter(), Counter(), [], set()
    with (destination / "candidates.jsonl").open("x", encoding="utf-8") as output:
        for index, (generator, group_id, lineage) in enumerate(keys, 1):
            fragments = grouped[(generator, group_id, lineage)]
            if len(fragments) != GENERATORS[generator]:
                reasons["incomplete_source_group"] += 1
                continue
            with _path(root, "groups/%s/%s.json" % (generator, group_id)).open() as stream:
                marker = json.load(stream)
            if marker.get("status") != "processed_group" or marker.get("image_name") != lineage:
                raise UnionAugmentationError("group marker disagrees with TRAIN fragment metadata")
            token_to_id = {row["fragment_token"]: str(row["fragment_id"]) for row in fragments}
            edges = [(token_to_id[row["fragment_a_token"]], token_to_id[row["fragment_b_token"]])
                     for row in marker["candidates"] if row["label"]]
            masks = {}
            for row in fragments:
                path = _path(root, row["target_audit"]["parent_mask_path"])
                with Image.open(path) as image:
                    masks[str(row["fragment_id"])] = np.asarray(image.convert("L")) > 127
            normalization = dict(method="none", applied=False)
            if boundary_ownership:
                try:
                    masks, normalization = normalize_boundary_ownership(masks)
                except UnionGeometryRejected as error:
                    reasons[error.reason] += 1
                    continue
            for plan in enumerate_union_plans(tuple(masks)):
                counts["proposed"] += 1
                if tiny_only:
                    merged = np.logical_or.reduce([masks[x] for x in plan.merged_ids])
                    if masks[plan.singleton_id].sum() / max(1, merged.sum()) >= .25:
                        reasons["not_tiny_area_ratio"] += 1
                        continue
                try:
                    candidate = build_union_candidate(masks, generator=generator, group_id=group_id,
                        lineage_id=lineage, lineage_splits=splits, neighbor_edges=edges, plan=plan, config=config)
                except UnionGeometryRejected as error:
                    reasons[error.reason] += 1
                    continue
                if candidate.metadata["geometry_signature"] in seen:
                    reasons["duplicate_generated_pair_geometry"] += 1
                    continue
                seen.add(candidate.metadata["geometry_signature"])
                row = candidate.metadata
                row["boundary_ownership_normalization"] = normalization
                if save_assets:
                    model_rows.append(_save_candidate(destination, candidate))
                rows.append(row)
                counts["accepted"] += 1
                counts["tiny"] += int(row["area_ratio"] < .25)
                counts[row["variant"]] += 1
                output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            print(json.dumps(dict(event="union_group_complete", processed_groups=index, total_groups=len(keys),
                                  counts=dict(counts), rejection_counts=dict(reasons))), flush=True)
    if save_assets:
        (destination / "pairs").mkdir()
        with (destination / "pairs/train.jsonl").open("x", encoding="utf-8") as stream:
            for row in model_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = dict(schema_version=SCHEMA_VERSION, status="complete_bounded_probe", source_root=str(root),
        source_family="canonical_rachel_release", split="train", validation_or_test_modified=False,
        available_train_groups=len(grouped), processed_train_groups=len(keys), seed=seed, max_groups=max_groups,
        shard_index=shard_index, shard_count=shard_count,
        save_assets=save_assets, tiny_only=tiny_only, boundary_ownership=boundary_ownership,
        counts=dict(counts), rejection_counts=dict(reasons),
        unique_lineages=len({row["lineage_id"] for row in rows}),
        scale=1.0, erosion_applied=False, new_independent_source_manuscripts=False,
        required_tiny_for_30k_positive=4500,
        meets_15_percent_of_30k_positive=counts["tiny"] >= 4500,
        scope="A bounded feasibility probe, not a completed60k training dataset")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260909)
    parser.add_argument("--save-assets", action="store_true")
    parser.add_argument("--tiny-only", action="store_true")
    parser.add_argument("--boundary-ownership", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build_from_release(**vars(args)), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""TRAIN-only, lossless three-group partitions of five existing Gen5 pieces.

This builder does not read files, normalize overlap ownership, or alter source
masks. Each output group is exactly an OR of its members in the parent frame.
Labels are induced solely by the supplied original CSV adjacency graph. The
existing preprocessing code supplies contours, correspondence and translations;
its rejected/quarantined pairs remain rejected, never relabelled as negatives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import combinations
import json
from typing import Mapping, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .rachel_preprocess import RachelFragment, RachelPair, RachelPreprocessConfig, _make_pair
from .rachel_union_augmentation import (
    CROSS, UnionAugmentationError, UnionConfig, UnionGeometryRejected, _binary, _fragment,
)


SCHEMA_VERSION = "rachel-train-gen5-partition/1"
GENERATOR = "gen5voronoi_1_1_3"
ORDERED_PATTERNS = ((1, 2, 2), (2, 1, 2), (3, 1, 1))


@dataclass(frozen=True)
class Gen5PartitionPlan:
    groups: Tuple[Tuple[str, ...], ...]
    ordered_patterns: Tuple[Tuple[int, int, int], ...] = ()

    @property
    def canonical_groups(self):
        return tuple(sorted(tuple(sorted(group)) for group in self.groups))


@dataclass(frozen=True)
class Gen5PartitionPair:
    fragment_a: RachelFragment
    fragment_b: RachelFragment
    pair: RachelPair
    metadata: dict


@dataclass(frozen=True)
class Gen5Partition:
    metadata: dict
    fragments: Tuple[RachelFragment, ...]
    pairs: Tuple[Gen5PartitionPair, ...]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _ids(fragment_ids):
    ids = tuple(fragment_ids)
    if len(ids) != 5 or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != 5:
        raise UnionAugmentationError("exactly five distinct nonempty string fragment IDs are required")
    return tuple(sorted(ids))


def _patterns(ordered_patterns):
    patterns = tuple(dict.fromkeys(tuple(pattern) for pattern in ordered_patterns))
    if not patterns or any(pattern not in ORDERED_PATTERNS or
                           any(type(n) is not int for n in pattern) for pattern in patterns):
        raise UnionAugmentationError("only ordered patterns 1/2/2, 2/1/2 and 3/1/1 are registered")
    return patterns


def enumerate_gen5_partition_plans(fragment_ids: Sequence[str], ordered_patterns=ORDERED_PATTERNS):
    """Return unique unordered partitions, retaining requested order aliases.

    Defaults give 15 partitions of type {1,2,2} and 10 of type {3,1,1}, NOT
    separate new geometry for the equivalent requested 1/2/2 and 2/1/2 forms.
    ``groups`` retain the first requested representative ordering.
    """
    ids, patterns = _ids(fragment_ids), _patterns(ordered_patterns)
    plans = {}
    for pattern in patterns:
        for first in combinations(ids, pattern[0]):
            remaining = tuple(x for x in ids if x not in first)
            for second in combinations(remaining, pattern[1]):
                third = tuple(x for x in remaining if x not in second)
                groups = (first, second, third)
                key = tuple(sorted(groups))
                if key not in plans:
                    plans[key] = Gen5PartitionPlan(groups, (pattern,))
                elif pattern not in plans[key].ordered_patterns:
                    plans[key] = replace(plans[key], ordered_patterns=plans[key].ordered_patterns + (pattern,))
    return tuple(plans[key] for key in sorted(plans))


def _validated_plan(plan, ids):
    if not isinstance(plan, Gen5PartitionPlan) or len(plan.groups) != 3:
        raise UnionAugmentationError("a Gen5 plan must contain exactly three groups")
    flat = tuple(x for group in plan.groups for x in group)
    if len(flat) != 5 or set(flat) != set(ids):
        raise UnionAugmentationError("partition must use all five source members exactly once")
    pattern = tuple(len(group) for group in plan.groups)
    if pattern not in ORDERED_PATTERNS:
        raise UnionAugmentationError("representative group order is not a registered pattern")
    aliases = _patterns(plan.ordered_patterns or (pattern,))
    if pattern not in aliases or any(sorted(alias) != sorted(pattern) for alias in aliases):
        raise UnionAugmentationError("ordered pattern aliases must describe the same partition")
    return Gen5PartitionPlan(tuple(tuple(sorted(group)) for group in plan.groups), aliases)


def _geometry_signature(a, b, pair):
    """A/B-order-independent model-mask geometry, with positive GT placement.

    Negative examples have no layout GT; they are deduplicated by their two
    input masks, not by an invented negative displacement. Source lineage and
    the complete partition remain in provenance, not in this geometry key.
    """
    masks = []
    for fragment in (a, b):
        mask = np.ascontiguousarray(fragment.model.model_mask, dtype=np.uint8)
        masks.append(_digest(dict(shape=list(mask.shape), sha256=hashlib.sha256(mask.tobytes()).hexdigest())))
    shift = list(pair.translation_a_to_b_rc) if pair.translation_a_to_b_rc is not None else None
    reverse_shift = [-x if x else 0.0 for x in shift] if shift is not None else None
    direct = _digest(dict(masks=masks, translation_rc=shift, label=bool(pair.label)))
    reverse = _digest(dict(masks=masks[::-1], translation_rc=reverse_shift, label=bool(pair.label)))
    return min(direct, reverse)


def build_gen5_partition(parent_masks: Mapping[str, np.ndarray], *, group_id: str,
                         lineage_id: str, lineage_splits: Mapping[str, str], neighbor_edges,
                         plan: Gen5PartitionPlan, config: UnionConfig = UnionConfig(),
                         generator: str = GENERATOR, scale: float = 1.0):
    """Build the three exact unions and their three CSV-labelled pair examples.

    The TRAIN guard is checked before geometry. No masks or source dictionaries
    are mutated. All three group masks must be 4-connected; holes, if present,
    are left intact. Positive seam quality is still governed by ``_make_pair``
    and the configured minimum token count, not by changing its CSV label.
    """
    if not lineage_id or lineage_splits.get(lineage_id) != "train":
        raise UnionAugmentationError("source lineage is not in the existing frozen TRAIN split")
    if generator != GENERATOR:
        raise UnionAugmentationError("only the existing five-fragment Gen5 generator is allowed")
    if isinstance(scale, bool) or scale != 1.0:
        raise UnionAugmentationError("Gen5 partition performs no resizing; scale must be 1.0")
    ids = _ids(parent_masks)
    plan = _validated_plan(plan, ids)
    edges = {tuple(sorted((a, b))) for a, b in neighbor_edges}
    if any(a == b or a not in ids or b not in ids for a, b in edges):
        raise UnionAugmentationError("neighbor edge must be an original distinct same-group CSV pair")
    masks = {key: _binary(parent_masks[key], (config.canvas_size,) * 2, key) for key in ids}
    ownership = np.stack([masks[key] for key in ids]).sum(0)
    if np.any(ownership > 1):
        raise UnionGeometryRejected("source_fragment_overlap")
    identity = dict(generator=generator, group_id=str(group_id), lineage_id=lineage_id,
                    canonical_groups=[list(group) for group in plan.canonical_groups])
    partition_id = "rachel-gen5-partition-" + _digest(identity)[:24]
    fragments = []
    for members in plan.groups:
        mask = np.logical_or.reduce([masks[key] for key in members])
        if ndimage.label(mask, structure=CROSS)[1] != 1:
            raise UnionGeometryRejected("partition_group_disconnected:" + ",".join(members))
        identifier = "gen5-group-" + _digest(dict(source=identity, members=list(members)))[:24]
        fragments.append(_fragment(identifier, mask, config))
    union_masks = np.stack([fragment.parent_mask for fragment in fragments])
    if np.any(union_masks.sum(0) > 1) or not np.array_equal(union_masks.any(0), ownership > 0):
        raise UnionAugmentationError("partition did not exactly conserve source support")
    groups = [dict(index=i, fragment_id=fragment.fragment_id, member_ids=list(members),
                   area_px=fragment.foreground_area)
              for i, (fragment, members) in enumerate(zip(fragments, plan.groups))]
    metadata = dict(schema_version=SCHEMA_VERSION, partition_id=partition_id, split="train",
        source_family="canonical_rachel_release", source_identity=identity, generator=generator,
        group_id=str(group_id), lineage_id=lineage_id, parent_members=list(ids), groups=groups,
        ordered_pattern=[len(group) for group in plan.groups],
        ordered_patterns=[list(pattern) for pattern in plan.ordered_patterns],
        original_neighbor_edges=[list(edge) for edge in sorted(edges)], config=asdict(config),
        scale=1.0, resized=False, rotation_degrees=0, holes_filled=False,
        gap_closing_applied=False, boundary_ownership_changed=False,
        original_parent_area_px=int((ownership > 0).sum()), parent_support_exactly_preserved=True,
        group_overlap_px=0, omitted_parent_members=[], group_connectivity=4,
        label_source="any original CSV adjacency edge crossing the two selected member groups",
        model_inputs_contain_parent_origin=False)
    preprocessing = RachelPreprocessConfig(canvas_size=config.canvas_size, contour_cap=config.contour_cap,
        minimum_positive_seam=config.minimum_seam_px, contour_smoothing_sigma=config.smoothing_sigma)
    results = []
    for i, j in combinations(range(3), 2):
        a, b = fragments[i], fragments[j]
        crossed = [edge for edge in sorted(edges) if
                   (edge[0] in plan.groups[i] and edge[1] in plan.groups[j]) or
                   (edge[1] in plan.groups[i] and edge[0] in plan.groups[j])]
        pair = _make_pair(a, b, label=bool(crossed), config=preprocessing)
        if pair.label and pair.main_training_eligible and len(pair.token_correspondences) < config.minimum_token_matches:
            pair = replace(pair, main_training_eligible=False,
                           selection_exclusion_reason="too_few_true_seam_token_matches")
        signature = _geometry_signature(a, b, pair)
        selected_members = tuple(sorted((tuple(plan.groups[i]), tuple(plan.groups[j]))))
        pair_id = "rachel-gen5-pair-" + _digest(dict(source=identity, selected_groups=selected_members))[:24]
        third = next(k for k in range(3) if k not in (i, j))
        info = dict(metadata, pair_id=pair_id, label=bool(pair.label), selected_group_indices=[i, j],
            selected_group_members=[list(plan.groups[i]), list(plan.groups[j])],
            third_group=groups[third], third_group_not_in_pair=True, crossing_neighbor_edges=[list(x) for x in crossed],
            geometry_signature=signature, pair_status=pair.status,
            main_training_eligible=pair.main_training_eligible,
            selection_exclusion_reason=pair.selection_exclusion_reason, quarantine_reason=pair.quarantine_reason,
            area_a_px=a.foreground_area, area_b_px=b.foreground_area,
            area_ratio=min(a.foreground_area, b.foreground_area) / max(a.foreground_area, b.foreground_area),
            seam_length_px=pair.seam_length_px, token_match_count=len(pair.token_correspondences),
            model_offset_a_rc=list(a.model.parent_to_model_offset_rc),
            model_offset_b_rc=list(b.model.parent_to_model_offset_rc),
            translation_a_to_b_rc=list(pair.translation_a_to_b_rc) if pair.translation_a_to_b_rc is not None else None,
            translation_a_to_b_xy_cartesian=list(pair.translation_a_to_b_xy) if pair.translation_a_to_b_xy is not None else None)
        results.append(Gen5PartitionPair(a, b, pair, info))
    return Gen5Partition(metadata, tuple(fragments), tuple(results))


def unique_partition_pairs(partitions: Sequence[Gen5Partition]):
    """Deduplicate geometry without replacing failed examples with valid ones.

    Caller should filter ``pair.main_training_eligible`` explicitly. Every
    alternate complete partition/third-group provenance is retained in metadata.
    """
    unique = {}
    for partition in partitions:
        for item in partition.pairs:
            if item.metadata.get("split") != "train" or item.metadata.get("schema_version") != SCHEMA_VERSION:
                raise UnionAugmentationError("deduplication only accepts this TRAIN partition protocol")
            key = _geometry_signature(item.fragment_a, item.fragment_b, item.pair)
            if key != item.metadata.get("geometry_signature"):
                raise UnionAugmentationError("pair geometry signature does not match its fragments/GT")
            provenance = {name: item.metadata[name] for name in
                ("pair_id", "partition_id", "source_identity", "groups", "selected_group_indices",
                 "selected_group_members", "third_group", "ordered_pattern", "ordered_patterns")}
            if key not in unique:
                unique[key] = replace(item, metadata=dict(item.metadata, geometry_provenance=[provenance]))
            else:
                old = unique[key]
                if (old.pair.main_training_eligible, old.pair.status, old.pair.selection_exclusion_reason) != (
                        item.pair.main_training_eligible, item.pair.status, item.pair.selection_exclusion_reason):
                    raise UnionAugmentationError("duplicate geometry has inconsistent preprocessing eligibility")
                alternatives = old.metadata["geometry_provenance"]
                if provenance not in alternatives:
                    unique[key] = replace(old, metadata=dict(old.metadata, geometry_provenance=alternatives + [provenance]))
    return tuple(unique[key] for key in sorted(unique))


__all__ = ["SCHEMA_VERSION", "GENERATOR", "ORDERED_PATTERNS", "Gen5PartitionPlan",
           "Gen5PartitionPair", "Gen5Partition", "enumerate_gen5_partition_plans",
           "build_gen5_partition", "unique_partition_pairs"]

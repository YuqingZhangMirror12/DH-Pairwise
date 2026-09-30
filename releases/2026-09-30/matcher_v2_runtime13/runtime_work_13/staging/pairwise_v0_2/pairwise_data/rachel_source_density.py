"""Prototype dense TRAIN supervision from shared clean source-cell ownership.

No target translation or cross-fragment weathered nearest-neighbour match is
accepted. Shared pixel-cell edges in the known clean parent segmentation give
seam IDs and a continuous seam coordinate. Each observed contour independently
projects to its own clean boundary, then correspondence is inherited through
that common source coordinate. Fresh 512/1024 contours come from the same
fixed masks. Artificial/unsupported edits are rejected by exact E1 replay.

This is a bounded preparation prototype, NOT a formal24K materialization.
"""
from dataclasses import dataclass, replace
from collections import defaultdict
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree

from .rachel_preprocess import (_pixel_cell_boundary_edges, _trace_boundary_loops,
    _cartesian_signed_area, extract_ordered_outer_contour)
from .rachel_training_dataset import _readonly
from .rachel_edge_weathering import EdgeWeatheringConfig, weather_fragment_edges

SCHEMA = "rachel-source-cell-density-prototype/1"


@dataclass(frozen=True)
class SourceBoundary:
    starts: np.ndarray
    ends: np.ndarray
    keys: tuple


def _key(a, b):
    return tuple(sorted((tuple(int(v) for v in a), tuple(int(v) for v in b))))


def source_boundary(mask):
    edges = tuple(sorted(_pixel_cell_boundary_edges(np.asarray(mask, bool))))
    loops = _trace_boundary_loops(edges)
    outer = max(loops, key=lambda loop: abs(_cartesian_signed_area(
        np.asarray([edges[int(i)][0] for i in loop], float))))
    selected = [edges[int(i)] for i in outer]
    return SourceBoundary(np.asarray([e[0] for e in selected], float),
        np.asarray([e[1] for e in selected], float), tuple(_key(*e) for e in selected))


def shared_source_seams(first, second, minimum_edges=4):
    """Exact shared grid edges, not close contours or a fitted translation."""
    shared = set(first.keys) & set(second.keys)
    incident = defaultdict(set)
    for edge in shared:
        for vertex in edge:
            incident[vertex].add(edge)
    unused = set(shared)
    mapping, components, rejected = {}, [], []
    while unused:
        seed = min(unused)
        component, frontier = set(), [seed]
        while frontier:
            edge = frontier.pop()
            if edge in component:
                continue
            component.add(edge)
            for vertex in edge:
                frontier.extend(incident[vertex] - component)
        unused -= component
        vertices = {v for e in component for v in e}
        degree = {v: len(incident[v] & component) for v in vertices}
        if len(component) < minimum_edges or max(degree.values()) > 2:
            rejected.append(dict(edge_count=len(component), reason="short_or_branched_source_interface"))
            continue
        endpoints = sorted(v for v in vertices if degree[v] == 1)
        current = endpoints[0] if endpoints else min(vertices)
        ordered, left = [], set(component)
        while left:
            choices = sorted(incident[current] & left)
            if not choices:
                raise ValueError("disconnected source seam walk")
            edge = choices[0]
            other = edge[0] if current == edge[1] else edge[1]
            ordered.append((current, other))
            current = other
            left.remove(edge)
        cid = len(components)
        for position, oriented in enumerate(ordered):
            mapping[_key(*oriented)] = (cid, float(position), np.asarray(oriented[0], float), np.asarray(oriented[1], float))
        components.append(dict(id=cid, length_px=len(ordered), closed=not bool(endpoints)))
    return mapping, components, dict(shared_cell_edges=len(shared), accepted_edges=len(mapping),
                                     rejected_components=rejected)


def confirm_fixed_e1_mask(clean_mask, fixed_mask, side_report, origin_receipt=None):
    """Replay original recorded erosion, never write or replace physical masks.

    A straight/partial new cut with no exact recorded E1 origin cannot silently
    acquire a seam label. Full partial-seam augmentation needs its own lineage.
    """
    clean = np.asarray(clean_mask, bool).squeeze()
    fixed = np.asarray(fixed_mask, bool).squeeze()
    if clean.shape != (800, 800) or fixed.shape != clean.shape:
        raise ValueError("expected canonical800 masks")
    if not side_report["effective_applied"]:
        if not np.array_equal(clean, fixed):
            raise ValueError("unrecorded/artificial mask edit: unchanged E1 side differs")
        return dict(exact_replay=True, exact_replay_runtime="local_identity",
                    local_replay_differing_pixels=0, max_depth_px=0., removed_pixels=0)
    proposal, replay = weather_fragment_edges(clean, seed=side_report["seed"],
        fragment_id=side_report["fragment_id"], config=EdgeWeatheringConfig(**side_report["config"]))
    local_exact = bool(np.array_equal(proposal, fixed))
    if not local_exact:
        # A recorded read-only replay in the original materialization runtime
        # can prove exact origin when local NumPy/SciPy tie handling differs.
        # The fixed input is NEVER replaced by this local proposed mask.
        if not origin_receipt or not origin_receipt.get("exact_replay"):
            raise ValueError("unsupported/artificial new edge: no exact recorded E1 origin")
        expected = dict(clean_mask_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
            fixed_mask_sha256=hashlib.sha256(fixed.tobytes()).hexdigest(),
            side_report_sha256=hashlib.sha256(json.dumps(side_report, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest())
        if any(origin_receipt.get(k) != v for k, v in expected.items()):
            raise ValueError("original-runtime replay receipt does not identify these exact masks/report")
    return dict(exact_replay=True, exact_replay_runtime="local" if local_exact else "original_remote_runtime",
                local_replay_differing_pixels=int(np.count_nonzero(proposal != fixed)),
                max_depth_px=float(side_report["config"]["max_depth_px"]),
                removed_pixels=int(clean.sum()-fixed.sum()))


def project_to_source(points_model, parent_to_model_offset, boundary, seams,
                      *, max_depth_px, ownership_allowance_px=0.):
    """One fragment only: label-free projection to its full original boundary."""
    points = np.asarray(points_model, float) + .5 - np.asarray(parent_to_model_offset, float)
    starts, ends = boundary.starts, boundary.ends
    edges = ends - starts
    _, neighbors = cKDTree((starts + ends) / 2).query(points, k=min(24, len(starts)))
    neighbors = np.asarray(neighbors).reshape(len(points), -1)
    delta = points[:, None] - starts[neighbors]
    alpha = np.clip(np.sum(delta * edges[neighbors], 2), 0, 1)
    projected = starts[neighbors] + alpha[..., None] * edges[neighbors]
    distances = np.linalg.norm(points[:, None] - projected, axis=2)
    winner = distances.argmin(1)
    ids = neighbors[np.arange(len(points)), winner]
    best = distances[np.arange(len(points)), winner]
    arc = ids + alpha[np.arange(len(points)), winner]
    candidate_arc = neighbors + alpha
    separation = np.abs(candidate_arc - arc[:, None])
    separation = np.minimum(separation, len(starts) - separation)
    nonlocal_gap = max(8., 2. * len(starts) / max(1, len(points)))
    ambiguous = np.any((separation > nonlocal_gap) & (distances <= best[:, None] + .5), axis=1)
    radius = 6. + max_depth_px + ownership_allowance_px
    trusted = (best <= radius) & ~ambiguous
    component = np.full(len(points), -1, np.int64)
    seam_s = np.full(len(points), np.nan)
    for i, edge_id in enumerate(ids):
        source = seams.get(boundary.keys[int(edge_id)])
        if not trusted[i] or source is None:
            continue
        cid, position, begin, end = source
        point_on_edge = projected[i, winner[i]]
        component[i] = cid
        seam_s[i] = position + np.clip(np.dot(point_on_edge - begin, end - begin), 0, 1)
    # Source non-seam cells next to seam endpoints remain ambiguous dustbin
    # supervision; ignored cells are never manufactured into positive matches.
    shared_ids = np.array([i for i, key in enumerate(boundary.keys) if key in seams], int)
    near_seam = np.zeros(len(points), bool)
    if len(shared_ids):
        gap = np.abs(ids[:, None] - shared_ids[None])
        near_seam = np.minimum(gap, len(starts) - gap).min(1) <= 2
    nonseam = trusted & (component < 0) & ~near_seam
    step = float(np.linalg.norm(np.roll(points_model, -1, axis=0) - points_model, axis=1).sum() / len(points_model))
    return dict(component=component, seam_s=seam_s, trusted=trusted, nonseam=nonseam,
        source_edge_index=ids, distance_px=best, mean_step_px=step,
        projection_radius_px=radius, ambiguous_count=int(ambiguous.sum()))


def inherit_dense_source_targets(first, second, components):
    """Shared source-cell IDs independently choose descendants on each side.

    Each original cell may have at most one nearest descendant per side in
    seam arclength, within0.75 sampling step. Greedy one-to-one selection uses
    source arclength error only. Weathered A/B spatial proximity is not used.
    """
    targets = [np.full(len(x["component"]), -2, np.int64) for x in (first, second)]
    for target, view in zip(targets, (first, second)):
        target[view["nonseam"]] = -1
    proposals = {}
    for component in components:
        cid = component["id"]
        source_cells = np.arange(component["length_px"], dtype=float) + .5
        descendants = []
        for view in (first, second):
            ids = np.flatnonzero(view["component"] == cid)
            if not len(ids):
                descendants.append(None)
                continue
            distance = np.abs(source_cells[:, None] - view["seam_s"][ids][None])
            if component["closed"]:
                distance = np.minimum(distance, component["length_px"] - distance)
            choice = distance.argmin(1)
            best = distance[np.arange(len(source_cells)), choice]
            trusted = best <= .75 * view["mean_step_px"]
            descendants.append((ids[choice], best, trusted))
        if any(x is None for x in descendants):
            continue
        a, b = descendants
        for cell in np.flatnonzero(a[2] & b[2]):
            key = (int(a[0][cell]), int(b[0][cell]))
            cost = float(a[1][cell] + b[1][cell])
            candidate = (cost, cid, int(cell))
            if key not in proposals or candidate < proposals[key]:
                proposals[key] = candidate
    chosen = []
    for (i, j), (cost, cid, cell) in sorted(proposals.items(), key=lambda item: (item[1], item[0])):
        if targets[0][i] >= 0 or targets[1][j] >= 0:
            continue
        targets[0][i], targets[1][j] = j, i
        chosen.append((i, j, cid, cell, cost))
    return targets[0], targets[1], chosen


def resample_source_density(sample, report, source_masks, offsets, clean_model_masks,
                            *, cap, ownership_allowance_px=0., origin_receipts=None):
    if cap not in (512, 1024):
        raise ValueError("bounded prototype supports source512/1024 only")
    schema=report.get('schema_version')
    if schema not in ("rachel-weathered-source-arc-e1/v1","rachel-clean-source-density-eval/1"):
        raise ValueError("unknown augmentation; explicit new-edge lineage is required")
    if schema=='rachel-clean-source-density-eval/1' and (report.get('changed_pair') or
            any(report['side_'+s]['effective_applied'] for s in 'ab')):
        raise ValueError('clean evaluation source cannot claim weathering')
    if sample.label and np.any(np.asarray(source_masks['a'],bool) & np.asarray(source_masks['b'],bool)):
        raise ValueError("source-cell ownership must be exclusive before interface construction")
    replay = {s: confirm_fixed_e1_mask(clean_model_masks[s], getattr(sample, "mask_" + s),
        report["side_" + s], (origin_receipts or {}).get(s)) for s in "ab"}
    boundaries = {s: source_boundary(source_masks[s]) for s in "ab"}
    if sample.label:
        seams, components, topology = shared_source_seams(boundaries["a"], boundaries["b"])
    else:
        # Original pair-negative identity is established by the resolver.
        # Unrelated parent frames must never be intersected to invent seams.
        seams, components = {}, []
        topology = dict(shared_cell_edges=0, accepted_edges=0,
            rejected_components=[], source_interface_not_computed_for_negative=True)
    if sample.label and not components:
        raise ValueError("no exact continuous shared source-cell interface; do not invent target matches")
    updates, views = {}, {}
    for side in "ab":
        points, valid = extract_ordered_outer_contour(np.asarray(getattr(sample, "mask_" + side)).squeeze().astype(bool),
                                                    cap=cap, smoothing_sigma=3.)
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("duplicate contour points must not impersonate density")
        updates["points_rc_" + side], updates["contour_valid_" + side] = points, valid
        views[side] = project_to_source(points, offsets[side], boundaries[side], seams,
            max_depth_px=replay[side]["max_depth_px"], ownership_allowance_px=ownership_allowance_px)
    if sample.label:
        ta, tb, chosen = inherit_dense_source_targets(views["a"], views["b"], components)
    else:
        ta = np.full(len(views['a']['component']),-1,np.int64)
        tb = np.full(len(views['b']['component']),-1,np.int64)
        chosen = []
    updates.update(target_a=_readonly(ta, np.int64), target_b=_readonly(tb, np.int64))
    result = replace(sample, **updates)
    detail = dict(schema_version=SCHEMA, cap=cap, physical_masks_unchanged=True,
        gt_translation_used_for_matching=False, weathered_cross_fragment_nearest_neighbor=False,
        target_source=("exact clean ownership cell edges plus independent within-fragment ancestry"
            if sample.label else "existing frozen TRAIN nonpair identity; all real contour points dustbin"),
        topology=topology, components=components, replay=replay,
        new_match_count=len(chosen), old_match_count=int(np.count_nonzero(sample.target_a >= 0)),
        chosen_source_cells=[list(x) for x in chosen],
        sides={s:dict(points=len(views[s]["component"]), trusted=int(views[s]["trusted"].sum()),
                     ignored=int(np.count_nonzero(getattr(result,"target_"+s)==-2)),
                     source_seam_tokens=int(np.count_nonzero(views[s]["component"]>=0)),
                     mean_step_px=views[s]["mean_step_px"], projection_radius_px=views[s]["projection_radius_px"],
                     ambiguous_count=views[s]["ambiguous_count"]) for s in "ab"})
    fields = ('effective_supervised_match_count','inherited_match_count','ignored_token_count','inheritance_rule')
    new_report = dict(report, density=detail,
        original_e1_target_summary={k:report[k] for k in fields},
        effective_supervised_match_count=len(chosen), inherited_match_count=len(chosen),
        ignored_token_count=int(np.count_nonzero(ta==-2)+np.count_nonzero(tb==-2)),
        inheritance_rule='exact clean shared source-cell IDs via independent own-boundary projection')
    return result, new_report, views

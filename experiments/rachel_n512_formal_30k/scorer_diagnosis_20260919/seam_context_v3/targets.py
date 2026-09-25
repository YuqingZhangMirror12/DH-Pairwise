"""Reliable source-derived support/links. Unknown is never a negative label."""
import numpy as np
import torch
from .valid_contour import remap_target


def visible_components(target, valid):
    """Components of OBSERVED GT-positive samples, not invented intervening GT."""
    ids = np.flatnonzero(valid)
    good = np.asarray(target)[ids] >= 0
    component = np.full(len(target), -1, np.int64)
    stop = np.full(len(target), -1, np.int64)
    if not good.any():
        return component, stop
    starts = good & ~np.roll(good, 1)
    if not starts.any():
        component[ids] = 0
        stop[ids] = 0
        return component, stop
    start = int(np.flatnonzero(starts)[0])
    sequence = np.roll(ids, -start)
    k, previous = -1, False
    for i in sequence:
        yes = target[i] >= 0
        if yes and not previous:
            k += 1
        if yes:
            component[i] = k
        previous = yes
    for rank, i in enumerate(ids):
        if target[i] < 0:
            continue
        left, right = ids[(rank-1) % len(ids)], ids[(rank+1) % len(ids)]
        if target[left] == -1 or target[right] == -1:
            stop[i] = 1
        elif target[left] >= 0 and target[right] >= 0:
            stop[i] = 0
    return component, stop


def base_target_metadata(sample):
    result = {}
    for side in 'ab':
        target, valid = getattr(sample, 'target_'+side), getattr(sample, 'contour_valid_'+side)
        component, stop = visible_components(target, valid)
        result['component_'+side], result['stop_'+side] = component, stop
        result['gap_'+side] = np.zeros((len(target), len(target)), bool)
    return result


def known_gap_links(sample, clean_sample, report):
    """Recover links only across documented removed source arcs on a GT run.

    New cut points themselves are not assigned positive correspondence labels.
    Projection is within each fragment to its OWN source contour, never A-to-B
    nearest-neighbor matching. Incomplete source GT leaves the link unknown.
    """
    from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import source_arc_ancestry
    from scipy.spatial import cKDTree
    result = base_target_metadata(sample)
    for side in 'ab':
        if not report.get('changed_'+side, False):
            continue
        info = report.get('side_'+side, {})
        regions = info.get('ignore_source_regions', [])
        if not regions:
            continue
        p = getattr(clean_sample, 'points_rc_'+side)
        v = getattr(clean_sample, 'contour_valid_'+side)
        target = getattr(clean_sample, 'target_'+side)
        current = getattr(sample, 'points_rc_'+side)
        current_target = getattr(sample, 'target_'+side)
        source_mask = getattr(clean_sample, 'mask_'+side)[0]
        ancestry, _, _ = source_arc_ancestry(source_mask, p, v, current, 30.)
        removed = np.zeros(len(p), bool)
        for region in regions:
            xy = np.asarray(region.get('source_points_rc', []), float).reshape(-1, 2)
            if len(xy):
                removed |= cKDTree(xy).query(p)[0] <= 8.
        source_ids = np.flatnonzero(v)
        inverse = {int(i): k for k, i in enumerate(source_ids)}
        lengths = np.linalg.norm(np.roll(p[source_ids], -1, axis=0)-p[source_ids], axis=1)
        good = np.flatnonzero((current_target >= 0) & (ancestry >= 0))
        # Sparse supervised endpoints only; no full ground-truth seam filling.
        for i in good:
            for j in good:
                if i >= j:
                    continue
                start, finish = inverse.get(int(ancestry[i])), inverse.get(int(ancestry[j]))
                if start is None or finish is None or start == finish:
                    continue
                for u, w in ((start, finish), (finish, start)):
                    ranks = (np.arange((w-u) % len(source_ids)+1)+u) % len(source_ids)
                    path = source_ids[ranks]
                    if (lengths[ranks[:-1]].sum() <= 128. and
                        (target[path] >= 0).all() and removed[path].any()):
                        result['gap_'+side][i, j] = result['gap_'+side][j, i] = True
    return result


def compact_targets(batch, output):
    result = {'a': remap_target(batch['target_a'], output.ga, output.gb),
              'b': remap_target(batch['target_b'], output.gb, output.ga)}
    for side, g in (('a', output.ga), ('b', output.gb)):
        for key in ('component', 'stop'):
            result[key+'_'+side] = batch[key+'_'+side].gather(1, g.index)
        gap = batch['gap_'+side]
        result['gap_'+side] = gap.gather(1, g.index[:, :, None].expand(-1, -1, gap.shape[2])).gather(
            2, g.index[:, None].expand(-1, len(g.index[0]), -1))
    return result


def graph_targets(record, targets, b):
    i, j = record.edges.unbind(-1)
    ta, tb = targets['a'][b, i], targets['b'][b, j]
    correct = (ta == j) & (tb == i)
    known = (ta != -2) & (tb != -2)
    ni, nj = record.edges[record.neighbors].unbind(-1)
    continuation_a = ((targets['component_a'][b, i, None] == targets['component_a'][b, ni]) &
                      (targets['component_a'][b, i, None] >= 0))
    continuation_b = ((targets['component_b'][b, j, None] == targets['component_b'][b, nj]) &
                      (targets['component_b'][b, j, None] >= 0))
    gap_a = targets['gap_a'][b, i[:, None], ni]
    gap_b = targets['gap_b'][b, j[:, None], nj]
    both = correct[:, None] & correct[record.neighbors]
    reliable_link = both & (continuation_a | gap_a) & (continuation_b | gap_b)
    false_edge = known & ~correct
    conflict = (false_edge[:, None] | false_edge[record.neighbors]) & known[:, None] & known[record.neighbors]
    relation = torch.full_like(record.neighbors, -1)
    relation[conflict] = 2
    relation[reliable_link] = (gap_a | gap_b)[reliable_link].long()
    relation[(i[:, None] == ni) & (j[:, None] == nj)] = -1
    relation[~record.neighbor_valid] = -1
    stop_a, stop_b = targets['stop_a'][b, i], targets['stop_b'][b, j]
    stop = torch.where(correct & (stop_a >= 0) & (stop_b >= 0), torch.maximum(stop_a, stop_b), -1)
    return correct, known, relation, stop

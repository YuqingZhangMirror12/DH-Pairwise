"""Target-blind discrete translation modes and finite gappy ordered beam paths.

Only indices are detached. Verifier/refinement gather live model values later.
No GT, no minimum seam-length acceptance, and no cross-mode pose averaging.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from .beam_kernel import beam_paths


@dataclass
class Candidate:
    edge_ids: np.ndarray
    translation: np.ndarray
    structural_score: float
    teacher: bool = False


def _sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -40, 40)))


def _propose_one(record, q, ga, gb, batch_index, cfg):
    edges = record.edges.detach().cpu().numpy()
    if not len(edges):
        return []
    i, j = edges.T
    a, b = ga.points[batch_index].cpu().numpy(), gb.points[batch_index].cpu().numpy()
    aa, ab = ga.arc[batch_index].cpu().numpy(), gb.arc[batch_index].cpu().numpy()
    pa, pb = float(ga.perimeter[batch_index]), float(gb.perimeter[batch_index])
    ca, cb = ga.cell[batch_index].cpu().numpy(), gb.cell[batch_index].cpu().numpy()
    delta = b[j]-a[i]
    mass = q.detach().cpu().numpy()[i, j]
    support = _sigmoid(record.support.detach().float().cpu().numpy())
    stop = _sigmoid(record.stop.detach().float().cpu().numpy())
    neighbors = record.neighbors.cpu().numpy()
    nv = record.neighbor_valid.cpu().numpy()
    lp = record.links.detach().float().softmax(-1).cpu().numpy()
    weight = mass*support
    # Neighboring bins compete by actual transport support; shifted grids reduce
    # mode boundary artifacts. Every broad edge votes, not only the best512.
    seeds = []
    for offset in (0., .5):
        bins = np.floor(delta/24.+offset).astype(np.int64)
        _, inverse = np.unique(bins, axis=0, return_inverse=True)
        scores = np.bincount(inverse, weights=weight)
        sx = np.bincount(inverse, weights=weight*delta[:, 0])
        sy = np.bincount(inverse, weights=weight*delta[:, 1])
        for ix in np.argsort(-scores, kind='stable')[:24]:
            if scores[ix] > 0:
                seeds.append((scores[ix], np.array([sx[ix], sy[ix]])/scores[ix]))
    modes = []
    for score, seed in sorted(seeds, key=lambda x: -x[0]):
        if any(np.linalg.norm(seed-t) < 20 for t in modes):
            continue
        modes.append(seed)
        if len(modes) == 16:
            break
    paths = []
    for seed in modes:
        member = np.flatnonzero(np.linalg.norm(delta-seed, axis=1) <= cfg.mode_radius_px)
        if not len(member):
            continue
        estimate = seed.copy()
        for _ in range(3):
            w = weight[member]/(1+(np.linalg.norm(delta[member]-estimate, axis=1)/16.)**2)
            estimate = (w[:, None]*delta[member]).sum(0)/max(w.sum(), 1e-20)
        member = member[np.linalg.norm(delta[member]-estimate, axis=1) <= cfg.mode_radius_px]
        if not len(member):
            continue
        # Try origins after the largest uncovered arcs on BOTH fragments, so
        # A/B exchange does not change which unwrappings are explored.
        cuts = []
        for s, perimeter in ((aa[i[member]], pa), (ab[j[member]], pb)):
            unique = np.unique(s)
            gaps = (np.roll(unique, -1)-unique) % perimeter
            cuts.append(unique[(np.argsort(-gaps, kind='stable')[:3]+1) % len(unique)])
        # Inverse B winding is used for monotone seam continuation.
        mapping = np.full(len(edges), -1, np.int64)
        mapping[member] = np.arange(len(member))
        local_neighbors = mapping[neighbors[member]]
        local_valid = nv[member] & (local_neighbors >= 0)
        local_neighbors = np.maximum(local_neighbors, 0)
        local_b_arc = ab[j[member]]
        db_all = (local_b_arc[local_neighbors]-local_b_arc[:, None]) % pb
        local_lp = lp[member]
        for origin_a in cuts[0]:
            ua = (aa[i[member]]-origin_a) % pa
            ordered = np.argsort(ua, kind='stable')
            da_all = ua[:, None]-ua[local_neighbors]
            admissible = (local_valid & (da_all > 0) & (da_all <= cfg.max_gap_px) &
                (db_all > 0) & (db_all <= cfg.max_gap_px) &
                (np.maximum(da_all, db_all)/np.maximum(1., np.minimum(da_all, db_all)) <= 8))
            node = np.log1p(mass[member]/max(float(np.quantile(mass[member], .75)), 1e-12)) + np.log(support[member]+1e-8)
            # Approximate independent arc cells instead of duplicate Patch votes.
            node *= np.minimum(np.minimum(ca[i[member]], cb[j[member]])/8., 4.)
            finals = beam_paths(ordered,local_neighbors,admissible,local_lp,da_all,db_all,node,stop[member],pb,cfg.beam_width)
            for score, chain in finals:
                ids = member[np.asarray(chain, np.int64)]
                w = weight[ids]
                t = (delta[ids]*w[:, None]).sum(0)/max(float(w.sum()), 1e-20)
                paths.append(Candidate(ids, t.astype(np.float32), float(score)))
    # Remove same-mode/same-arc near duplicates, retain different seam hypotheses.
    output = []
    for candidate in sorted(paths, key=lambda x: -x.structural_score):
        ids = set(candidate.edge_ids.tolist())
        duplicate = False
        for previous in output:
            other = set(previous.edge_ids.tolist())
            overlap = len(ids & other)/max(1, min(len(ids), len(other)))
            if np.linalg.norm(candidate.translation-previous.translation) < 12 and overlap > .5:
                duplicate = True
                break
        if not duplicate:
            output.append(candidate)
        if len(output) >= cfg.max_candidates:
            break
    return output


def propose(record, q, ga, gb, batch_index, cfg):
    first = _propose_one(record, q, ga, gb, batch_index, cfg)
    swapped = SimpleNamespace(**vars(record))
    swapped.edges = record.edges.flip(-1)
    second = _propose_one(swapped, q.T, gb, ga, batch_index, cfg)
    second = [Candidate(c.edge_ids, -c.translation, c.structural_score) for c in second]
    # Both orientations are explored; under A/B exchange this union is unchanged.
    combined = first+second
    combined.sort(key=lambda c: (-c.structural_score, tuple(sorted(c.edge_ids.tolist()))))
    result = []
    for candidate in combined:
        a = set(candidate.edge_ids.tolist())
        if any(np.linalg.norm(candidate.translation-old.translation) < 12 and
               len(a & set(old.edge_ids.tolist()))/max(1, min(len(a), len(old.edge_ids))) > .5 for old in result):
            continue
        result.append(candidate)
        if len(result) == cfg.max_candidates: break
    return result


def open_arc_indices(arc, perimeter, support_ids, extension=64.):
    """Smallest directed covering arc; retain every intermediate observation."""
    s = np.unique(np.asarray(arc)[np.asarray(support_ids, np.int64)])
    if not len(s):
        return np.empty(0, np.int64), np.empty(0, np.float32)
    gaps = (np.roll(s, -1)-s) % perimeter
    k = int(np.argmax(gaps))
    start = float(s[(k+1) % len(s)])
    length = 0. if len(s) == 1 else float(perimeter-gaps[k])
    origin = (start-extension) % perimeter
    u = (np.asarray(arc)-origin) % perimeter
    keep = u <= min(perimeter, length+2*extension)+1e-5
    ids = np.flatnonzero(keep)
    ids = ids[np.argsort(u[ids], kind='stable')]
    return ids, ((u[ids]-extension)/max(length, 1.)).astype(np.float32)

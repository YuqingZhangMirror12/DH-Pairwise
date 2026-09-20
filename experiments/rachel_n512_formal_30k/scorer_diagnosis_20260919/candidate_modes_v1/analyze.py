"""Target-blind secondary-mode diagnostic on already saved candidate edges.

Not a deployed decoder, trained scorer, new benchmark, or GT-selected layout.
The original first mode must replay before interpreting the secondary modes.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def propose_modes(delta, mass, *, radius=10., limit=5, min_inliers=3):
    """Same seed support/tie cost/3 refinements as mode_consensus; then NMS.

    delta/mass are in the original production candidate order. Neither GT nor
    pair labels enter the proposals. Original evidence is never removed: later
    modes use all candidates, but seeds/centers within 2 radii are suppressed.
    """
    delta, mass = np.asarray(delta, np.float64), np.asarray(mass, np.float64)
    if (delta.ndim != 2 or delta.shape[1] != 2 or mass.shape != (len(delta),)
            or not np.isfinite(delta).all() or not np.isfinite(mass).all()
            or (mass <= 0).any() or radius <= 0 or limit < 1):
        raise ValueError('finite ordered offsets and positive masses required')
    if not len(delta):
        return []
    weight = mass / mass.max()
    squared = ((delta[:, None] - delta[None, :]) ** 2).sum(2)
    support_mask = squared <= radius ** 2
    support = support_mask @ weight
    available = np.ones(len(delta), bool)
    modes = []
    while available.any() and len(modes) < limit:
        ids = np.flatnonzero(available)
        tied = ids[np.isclose(support[ids], support[ids].max(), rtol=1e-12, atol=1e-14)]
        cost = ((squared[tied] * support_mask[tied]) @ weight) / support[tied]
        tied = tied[np.isclose(cost, cost.min(), rtol=1e-12, atol=1e-14)]
        seed = int(tied[0])
        center = delta[seed].copy()
        for _ in range(3):
            inside = np.linalg.norm(delta - center, axis=1) <= radius
            updated = (delta[inside] * weight[inside, None]).sum(0) / weight[inside].sum()
            change = np.linalg.norm(updated - center)
            center = updated
            if change < 1e-8:
                break
        distance = np.linalg.norm(delta - center, axis=1)
        inside = distance <= radius
        available[seed] = False
        available[distance <= 2 * radius] = False
        if (int(inside.sum()) < min_inliers or
                any(np.linalg.norm(center - np.array(m['translation_rc'])) <= 2 * radius for m in modes)):
            continue
        modes.append(dict(rank=len(modes) + 1, seed_index=seed,
            translation_rc=center.tolist(), candidate_inlier_ids=np.flatnonzero(inside).tolist(),
            inlier_count=int(inside.sum()), seed_support_normalized=float(support[seed]),
            raw_q_mass=float(mass[inside].sum()),
            residual_px=float(np.sqrt((weight[inside] * distance[inside] ** 2).sum() / weight[inside].sum()))))
    return modes


def run():
    selection = json.loads((ROOT / 'selected_cases.json').read_text())
    selection = {(r['dataset'], r['pair_id']): r for r in selection}
    rows, sources = [], []
    for model in ('s4', 's6', 's6_depth4', 's7'):
        source = ROOT / 'heatmaps_v1' / model / 'cases.json'
        sources.append(dict(model=model, path=str(source.relative_to(ROOT)),
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        for case in json.loads(source.read_text()):
            layout = case['layout']
            indices = np.asarray(layout['candidate_indices'], np.int64)
            a, b = np.asarray(case['points_rc_a']), np.asarray(case['points_rc_b'])
            delta = b[indices[:, 1]] - a[indices[:, 0]]
            mass = np.asarray(layout['candidate_sinkhorn_mass'])
            modes = propose_modes(delta, mass)
            if not layout['valid']:
                raise ValueError('this snapshot contains invalid/ambiguous baseline; extend replay explicitly')
            if not modes:
                raise ValueError('valid baseline without a replayed mode')
            error = float(np.linalg.norm(np.array(modes[0]['translation_rc']) - layout['t_a_to_b_rc']))
            same_inliers = modes[0]['candidate_inlier_ids'] == np.flatnonzero(layout['inlier_mask']).tolist()
            if error > 1e-7 or not same_inliers:
                raise ValueError('first-mode replay failed: ' + case['pair_id'])
            # Only AFTER proposals are fixed do we read GT for evaluation.
            gt = case.get('target_translation_rc')
            gt_available = bool(case['label'] and case.get('layout_gt_available') and gt is not None)
            errors = [float(np.linalg.norm(np.array(m['translation_rc']) - gt)) for m in modes] if gt_available else None
            selected = selection[(case['dataset'], case['pair_id'])]
            rows.append(dict(model=model, dataset=case['dataset'], pair_id=case['pair_id'],
                name=selected.get('name'), stratum=selected['stratum'], label=case['label'],
                layout_gt_available=gt_available, original_score=case['score'],
                first_mode_replay_l2_px=error, first_mode_inliers_equal=same_inliers,
                modes=modes, gt_mode_errors_px=errors,
                oracle_coverage20_at_k={str(k): any(e <= 20 for e in errors[:k]) for k in (1, 3, 5)} if gt_available else None))
    if any(hashlib.sha256((ROOT / s['path']).read_bytes()).hexdigest() != s['sha256'] for s in sources):
        raise RuntimeError('source changed during diagnostic')
    output = dict(schema_version='saved-candidate-modes-diagnostic/1', sources=sources,
        selection_sha256=hashlib.sha256((ROOT / 'selected_cases.json').read_bytes()).hexdigest(),
        sources_unchanged=True, inference_performed=False, training_performed=False,
        thresholds_fitted=False, gt_used_to_propose_modes=False,
        proposal=dict(radius_px=10, refinement_iterations=3, nms_distance_px=20, max_modes=5,
            input='original Top2 union capped512; no new correspondence candidates'),
        caveats=['Fixed illustrative cases, not a representative benchmark.',
            'Oracle candidate coverage is not a trained ranking result or achieved layout accuracy.',
            'No coverage means no correct proposal here, not no support in the full Sinkhorn matrix.',
            'OOD/negative cases have no layout GT; proposals cannot be labelled correct.',
            'Later modes are seed-support ordered, not necessarily refined-support ordered.',
            'The reference first-mode replay is validated for this snapshot; this is not a replacement decoder.'],
        rows=rows)
    path = Path(__file__).with_name('results.json')
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(cases=len(rows), max_first_mode_error=max(r['first_mode_replay_l2_px'] for r in rows))))
    for r in rows:
        if r['model'] == 's6' and r['stratum'] == 'real_TP_bad':
            print(json.dumps({k:r[k] for k in ('name','gt_mode_errors_px','oracle_coverage20_at_k')}, ensure_ascii=False))


if __name__ == '__main__':
    run()

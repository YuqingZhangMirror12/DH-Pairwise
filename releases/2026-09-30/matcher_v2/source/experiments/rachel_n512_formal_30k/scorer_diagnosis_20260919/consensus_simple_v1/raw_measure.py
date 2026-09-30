"""Raw EdgeCloud distributions requested in simplify.md, before builder design."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
import torch

from modes import weighted_modes, unique_modes

RADII = (8., 10., 12., 16.)
BINS = np.arange(0, 40.100001, .1)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def q(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), mean=float(a.mean()), **{
        'p'+str(p): float(np.percentile(a, p)) for p in (0, 1, 5, 10, 50, 90, 95, 99, 100)})


def spacing_bin(x):
    return '<=3' if x <= 3 else '3-4' if x <= 4 else '4-5' if x <= 5 else '5-6' if x <= 6 else '>6'


def init(root, out):
    global ROOT, OUT
    ROOT, OUT = Path(root), Path(out)
    torch.set_num_threads(1)


def worker(task):
    split, i = task
    row = json.loads((ROOT/split/f'{i:05d}.json').read_text())
    raw = torch.load(ROOT/row['evidence_file'], map_location='cpu', weights_only=False)
    c = raw['proposals'].cloud
    x = c.displacement.double().numpy()
    w = (c.q.double()*c.arc_weight.double()).numpy()
    spacing = float((.5*(c.spacing_a+c.spacing_b)).median()) if len(x) else 0.
    truth = row['target_translation_rc']
    use_gt = bool(row['label'] and row['gt_known'] and not row['gt_excluded'])
    r = dict(pair_id=row['pair_id'], index=i, split=split, label=row['label'],
        gt_known=row['gt_known'], gt_excluded=row['gt_excluded'], usable_gt=use_gt,
        cloud_edges=len(x), spacing=spacing, spacing_bin=spacing_bin(spacing),
        source_family=row.get('source_family'), base_pair_id=row.get('base_pair_id'),
        evidence_file=row['evidence_file'], evidence_sha256=row['evidence_sha256'])
    correct = np.linalg.norm(x - np.asarray(truth), axis=1) <= 20 if use_gt else None
    if use_gt:
        a = pdist(x[correct])
        hist = np.histogram(a, BINS)[0]
        assert int(hist.sum()) == len(a), 'correct edge distances must be <=40px'
        r.update(correct_edges=int(correct.sum()), wrong_edges=int((~correct).sum()),
            correct_pair_distance=q(a), correct_pair_distance_hist=hist.tolist(),
            correct_pair_distance_over_spacing=q(a/max(spacing, 1e-9)))
    r['radii'] = {}
    for radius in RADII:
        allm = unique_modes(weighted_modes(x, w, radius), radius)
        centers = np.asarray([m['center'] for m in allm]).reshape(-1, 2)
        masses = [m['mass'] for m in allm]
        s = dict(all_mode_count=len(allm), all_mode_spacing=q(pdist(centers)),
            all_mode_masses=masses, all_mode_centers=centers.tolist(),
            all_mode_mass=q(masses), total_edge_mass=float(w.sum()))
        if use_gt:
            cm = unique_modes(weighted_modes(x[correct], w[correct], radius), radius)
            wm = unique_modes(weighted_modes(x[~correct], w[~correct], radius), radius)
            cc = np.asarray([m['center'] for m in cm]).reshape(-1, 2)
            wc = np.asarray([m['center'] for m in wm]).reshape(-1, 2)
            near = cKDTree(x[~correct]).query(cc)[0] if len(cc) and (~correct).any() else np.array([])
            wd = cKDTree(cc).query(wc)[0] if len(cc) and len(wc) else np.array([])
            s.update(correct_mode_count=len(cm), correct_mode_nearest_wrong_edge=near.tolist(),
                correct_mode_nearest_wrong_edge_quantiles=q(near),
                wrong_mode_count=len(wm), wrong_mode_masses=[m['mass'] for m in wm],
                wrong_mode_distance_to_correct=wd.tolist(),
                wrong_mode_mass_fraction=[m['mass']/w.sum() for m in wm])
        r['radii'][str(int(radius))] = s
    save(OUT/split/f'{i:05d}.json', r)
    return r


def aggregate(rows):
    result = {}
    for split in sorted(set(r['split'] for r in rows)):
        result[split] = {}
        for binname in ['all','<=3','3-4','4-5','5-6','>6']:
            group = [r for r in rows if r['split'] == split and (binname == 'all' or r['spacing_bin'] == binname)]
            gt = [r for r in group if r['usable_gt']]
            hist = np.sum([r['correct_pair_distance_hist'] for r in gt], axis=0) if gt else np.zeros(len(BINS)-1, dtype=int)
            def histq(p):
                return float(BINS[min(np.searchsorted(np.cumsum(hist), p/100*sum(hist)), len(hist)-1)]+.05) if sum(hist) else None
            a = dict(pairs=len(group), gt_pairs=len(gt), spacing=q([r['spacing'] for r in group]),
                correct_pair_distance_pooled_hist=hist.tolist(), histogram_width_px=.1,
                correct_pair_distance_pooled={f'p{p}':histq(p) for p in (50,90,95,99)},
                per_pair_p95_correct_distance=q([r['correct_pair_distance']['p95'] for r in gt if r['correct_pair_distance']['n']]),
                per_pair_p95_over_spacing=q([r['correct_pair_distance_over_spacing']['p95'] for r in gt if r['correct_pair_distance_over_spacing']['n']]), radii={})
            for radius in ('8','10','12','16'):
                def values(key, pop):
                    return [v for r in pop for v in r['radii'][radius].get(key, [])]
                a['radii'][radius] = dict(
                    nearest_wrong_edge=q(values('correct_mode_nearest_wrong_edge',gt)),
                    wrong_mode_mass=q(values('wrong_mode_masses',gt)),
                    wrong_mode_distance=q(values('wrong_mode_distance_to_correct',gt)),
                    all_mode_mass=q(values('all_mode_masses',group)),
                    per_pair_mode_spacing_p50=q([r['radii'][radius]['all_mode_spacing']['p50'] for r in group if r['radii'][radius]['all_mode_spacing']['n']]),
                    mode_count=q([r['radii'][radius]['all_mode_count'] for r in group]))
            result[split][binname] = a
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase1', required=True); p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=16); p.add_argument('--limit', type=int)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    save(out/'protocol.json', dict(schema='raw-edge-distribution/1', radii=RADII,
        correct_edge='raw displacement within20px GT; GT diagnostics only',
        cloud='exact existing EdgeCloud, zero additional filtering',
        modes='Q*arc weighted flat mean shift; every unique displacement initialized',
        labelled_modes='fit separately to correct/wrong edges ONLY for descriptive analysis, never inference',
        hist='edge-pair-weighted pooled0.1px histogram; additionally report pair-weighted quantiles',
        real_used_for_scale_selection=False, cpu_workers=args.workers))
    tasks = [(s,i) for s,n in [('sim_select',1500),('dunhuang_cv',803),('turufan',602)] for i in range(min(n,args.limit or n))]
    start=time.time(); rows=[]
    try:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(args.phase1,args.out)) as pool:
            for row in pool.map(worker,tasks):
                rows.append(row)
                if len(rows)%100==0:
                    save(out/'status.json',dict(status='measuring',completed=len(rows),total=len(tasks),seconds=time.time()-start))
        save(out/'summary.json',aggregate(rows))
        save(out/'complete.json',dict(status='complete',pairs=len(rows),seconds=time.time()-start,limited=args.limit))
    except Exception as e:
        save(out/'failure.json',dict(error=repr(e),completed=len(rows)))
        raise


if __name__ == '__main__': main()

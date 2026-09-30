"""Read existing M12 Q/native hypotheses and fixed16 unions; NO inference.

Separate count, per-edge Q, and observation width before learned Scorer gates.
GT residual<=20 is a diagnostic proxy, not a human per-correspondence label.
Synthetic population here is the original1500 SELECT, explicitly NOT TEST3000.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')


def describe(ids, cloud, q, truth):
    index = {tuple(e): i for i, e in enumerate(cloud.ids.tolist())}
    edges = sorted(set(map(tuple, ids)))
    ii = torch.tensor([index[e] for e in edges], dtype=torch.long)
    qs = cloud.q[ii].double().numpy()
    width = cloud.arc_weight[ii].double().numpy()
    displacement = cloud.displacement[ii].double().numpy()
    if edges:
        actual = q[tuple(torch.tensor(edges).T)].double().numpy()
        assert np.array_equal(actual, qs)
    good = np.linalg.norm(displacement-np.asarray(truth), axis=1)<=20 if truth is not None else None
    n = len(edges); total = float(qs.sum()); mass = float((qs*width).sum())
    return dict(n=n, endpoints_a=len({a for a,b in edges}), endpoints_b=len({b for a,b in edges}),
        q_sum=total, q_mean=total/n if n else None, q_median=float(np.median(qs)) if n else None,
        q_max=float(qs.max()) if n else None, mass=mass,
        width_mean=float(width.mean()) if n else None,
        width_q_weighted=mass/total if total else None,
        effective_edges=total**2/float((qs*qs).sum()) if n and total else 0.,
        gt_residual20_edges=int(good.sum()) if good is not None else None,
        gt_residual20_mass=float((qs[good]*width[good]).sum()) if good is not None else None,
        edge_ids=[list(e) for e in edges], q_values=qs.tolist(), widths=width.tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase1',required=True)
    p.add_argument('--replay',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only stored-array analysis')
    torch.set_num_threads(1)
    phase, replay, out = map(Path, (args.phase1,args.replay,args.out))
    if read(replay/'complete.json').get('pairs')!=2905:raise ValueError('complete fixed16 replay required')
    out.mkdir(parents=True,exist_ok=False)
    rows=[];start=time.time();sources={};done=0
    for split,count in (('sim_select',1500),('dunhuang_cv',803),('turufan',602)):
        for i in range(count):
            rp=phase/split/f'{i:05d}.json';pp=replay/split/f'{i:05d}.json'
            row, prior=read(rp),read(pp)
            if row['pair_id']!=prior['pair_id']:raise ValueError('source/replay pairing differs')
            path=phase/row['evidence_file']
            if sha(path)!=row['evidence_sha256']:raise ValueError('raw Q cache changed')
            raw=torch.load(path,map_location='cpu',weights_only=False)
            cloud=raw['proposals'].cloud;hypotheses=raw['proposals'].hypotheses
            truth=row['target_translation_rc'] if row['label'] and row['gt_known'] and not row['gt_excluded'] else None
            kept=[c for c in prior['threshold16']['clusters'] if c['retained']]
            cc=[]
            for c in kept:
                ids=set().union(*(set(map(tuple,hypotheses[h].edge_ids.tolist())) for h in c['members']))
                features=describe(ids,cloud,raw['q'],truth)
                if features['n']!=c['union_edge_count'] or abs(features['mass']-c['raw_q_arc_mass_px'])>1e-4:
                    raise ValueError('deduplicated union/mass differs from stored replay')
                cc.append(dict(cluster_id=c['index'],members=c['members'],proposal_translation=c['translation_rc'],
                    proposal_gt_error=c['gt_error_px'],**features))
            raw_stats=describe(cloud.ids.tolist(),cloud,raw['q'],truth)
            for k in ('edge_ids','q_values','widths'):raw_stats.pop(k)
            rows.append(dict(split=split,index=i,pair_id=row['pair_id'],label=bool(row['label']),
                usable_gt=truth is not None,gt_excluded=row['gt_excluded'],numeric_valid=row['numeric_valid'],
                target=truth,raw_cloud=raw_stats,clusters=cc,
                points_a=raw['geometry_a'].points[0,:raw['q'].shape[0]].tolist(),
                points_b=raw['geometry_b'].points[0,:raw['q'].shape[1]].tolist(),
                evidence_sha256=row['evidence_sha256']))
            sources[str(rp)]=sha(rp);sources[str(pp)]=sha(pp)
            done+=1
            if done%100==0:save(out/'status.json',dict(status='reading_saved_arrays',pairs=done,elapsed_seconds=time.time()-start))
        save(out/(split+'.json'),[r for r in rows if r['split']==split])
    save(out/'complete.json',dict(status='complete',pairs=done,elapsed_seconds=time.time()-start,
        matcher_forward=False,scorer_forward=False,builder_rerun=False,optimizer_updates=0,gpu_used=False,
        raw_source_hashes_checked=True,sources=sources,
        diagnostic_gt_definition='raw displacement residual<=20px, not human edge correspondence GT',
        synthetic_population='SIM SELECT1500 from original phase1; not TEST3000'))
    print(json.dumps(dict(status='complete',pairs=done,elapsed_seconds=time.time()-start)))


if __name__=='__main__':main()

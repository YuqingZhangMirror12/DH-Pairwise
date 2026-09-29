"""Restore phase1 HYPOTHESIS-level metrics, alongside stricter raw-edge ones.

The main replay saved raw-edge contamination, which is NOT the original
hypothesis-level contamination definition. This audit reconstructs mode poses
on frozen clouds (no Matcher/GPU), selects on SELECT before reading real
performance, and preserves that earlier stricter selection.json unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from modes import weighted_modes
from raw_measure import save
from simple_builder import SimplePoseBuilder,SimplePolicy,canonical
from measure import CompatibilityConfig,ProposalConfig

def init(root,phase1):
    global ROOT,P1,GEO,CFG
    ROOT,P1=Path(root),Path(phase1);torch.set_num_threads(1)
    p=json.loads((P1/'protocol.json').read_text());GEO=CompatibilityConfig(**p['geometry']);CFG=ProposalConfig(**p['proposal_config'])

def statistic(poses,groups,truth,valid=None):
    valid=set(range(len(poses))) if valid is None else set(valid)
    correct={i for i,p in enumerate(poses) if i in valid and np.linalg.norm(p-truth)<=20}
    wrong40={i for i,p in enumerate(poses) if i in valid and np.linalg.norm(p-truth)>40}
    wrong20=valid-correct
    touching=[g for g in groups if g['retained'] and set(g['mode_ids'])&correct]
    complete=bool(correct and len(touching)==1 and correct<=set(touching[0]['mode_ids']) and touching[0]['gt_error_px']<=20)
    return dict(correct_modes=len(correct),single_complete=complete,
        mixed20_40=sum(bool(set(g['mode_ids'])&correct and set(g['mode_ids'])&wrong40) for g in groups),
        mixed20_20=sum(bool(set(g['mode_ids'])&correct and set(g['mode_ids'])&wrong20) for g in groups))

@torch.no_grad()
def worker(task):
    split,i=task
    row=json.loads((P1/split/f'{i:05d}.json').read_text())
    if not(row['label'] and row['gt_known'] and not row['gt_excluded']):return None
    raw=torch.load(P1/row['evidence_file'],map_location='cpu',weights_only=False)
    truth=np.asarray(row['target_translation_rc'])
    cloud=canonical(raw['proposals'].cloud)
    old=json.loads((ROOT/'baseline'/split/f'{i:05d}.json').read_text())
    oldposes=np.asarray([h.translation.numpy() for h in raw['proposals'].hypotheses])
    result={'baseline':statistic(oldposes,old['clusters'],truth,[i for i,h in enumerate(raw['proposals'].hypotheses) if len(h.edge_ids)])}
    for radius in [8,10,12,16]:
        b=SimplePoseBuilder(GEO,CFG,SimplePolicy(radius))
        modes=weighted_modes(cloud.displacement.double().numpy(),(cloud.q.double()*cloud.arc_weight.double()).numpy(),radius)
        centers=torch.tensor(np.asarray([m['center'] for m in modes]).reshape(-1,2),dtype=cloud.displacement.dtype)
        # cells/overlap do not enter the fit/contact pose; no need to load masks.
        fitted=[b._make(cloud,m['members'],centers[j],(j,),centers[j:j+1],None,None) for j,m in enumerate(modes)]
        records=json.loads((ROOT/f's{radius}'/split/f'{i:05d}.json').read_text())
        poses=np.asarray([c.translation.numpy() for c in fitted])
        result[f's{radius}']=statistic(poses,records['clusters'],truth)
    save(ROOT/'native_mode_audit'/split/f'{i:05d}.json',dict(pair_id=row['pair_id'],index=i,results=result))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase1',required=True);p.add_argument('--workers',type=int,default=48)
    a=p.parse_args();r=Path(a.root)
    assert (r/'complete.json').exists() and not (r/'native_mode_audit/complete.json').exists()
    save(r/'native_mode_audit/protocol.json',dict(reason='phase1 contamination and complete metrics operate on native hypotheses, not raw edges',
        original_raw_selection_preserved=True,gt_never_used_by_builder=True,real_performance_not_used_for_selection=True))
    controls=json.loads((r/'controls.json').read_text());sums={};start=time.time()
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(a.root,a.phase1)) as pool:
        for split,n in [('sim_select',1500),('dunhuang_cv',803)]:
            results=[x for x in pool.map(worker,[(split,i) for i in range(n)]) if x is not None]
            sums[split]={name:dict(pairs=len(results),single_complete=sum(x[name]['single_complete'] for x in results),
                fraction=sum(x[name]['single_complete'] for x in results)/len(results),
                mixed20_40=sum(x[name]['mixed20_40'] for x in results),mixed20_20=sum(x[name]['mixed20_20'] for x in results)) for name in results[0]}
            save(r/'native_mode_audit/summary.json',sums)
            if split=='sim_select':
                # No change of builder, radius set, data, GT20, or test controls.
                eligible=[name for name in controls if sums[split][name]['mixed20_40']==0 and all(c['passed'] for c in controls[name]) and sums[split][name]['fraction']>sums[split]['baseline']['fraction']]
                eligible.sort(key=lambda k:(-sums[split][k]['fraction'],int(k[1:])))
                save(r/'native_mode_audit/selection.json',dict(selected=eligible[0] if eligible else None,eligible=eligible,
                    selected_on='SIM SELECT native hypothesis metrics, phase1 definition',real_read_for_selection=False,
                    original_raw_selection='selection.json retained as additional stricter raw-edge check',recorded_unix=time.time()))
    save(r/'native_mode_audit/complete.json',dict(status='complete',seconds=time.time()-start,pairs=1042))
    print(json.dumps(sums))

if __name__=='__main__':main()

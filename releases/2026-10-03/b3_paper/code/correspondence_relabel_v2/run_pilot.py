"""CPU-only pilot for the new policy; no original archive or training writes."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from relabel import run, EvidenceError, POLICY, state_counts

def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def save_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def one(case, evidence, out):
    start=time.monotonic(); a=case['admission']; pid=a['pair_id']
    files={}
    for k,b in case['files'].items():
        p=evidence/b['path'];need=hashlib.sha256(p.read_bytes()).hexdigest()
        if need!=b['sha256']:raise ValueError('pilot input changed:'+str(p))
        files[k]=read_npz(p)
    z=files['sample']
    try:
        arrays,info=run(z,files.get('proof'),case['record'],files.get('preweather'),files.get('latent'))
    except EvidenceError as exc:
        # Fail closed and expose the exact evidence issue. Never automatically
        # relax a threshold, reconstruct missing pixels, or use old matches.
        arrays={k+'_'+s:np.full_like(z['target_'+s],-2) for k in ('full','tight') for s in 'ab'}
        info=dict(status='evidence_quarantined_all_ignore',error=str(exc),pairs=[],training_admitted=False)
    dest=out/pid;dest.mkdir()
    np.savez_compressed(dest/'labels.npz',**arrays)
    for mode in ('full','tight'):
        changed=dict(z,target_a=arrays[mode+'_a'],target_b=arrays[mode+'_b'])
        np.savez_compressed(dest/(mode+'.npz'),**changed)
        with np.load(dest/(mode+'.npz'),allow_pickle=False) as new:
            assert set(new.files)==set(z)
            assert all(np.array_equal(new[k],z[k]) for k in z if k not in ('target_a','target_b'))
            if not bool(z['label']):assert all(np.array_equal(new[k],z[k]) for k in z)
    metrics={}
    from relabel import dense,sample_mask
    from scipy.spatial import cKDTree
    t=z['translation_a_to_b_rc'].astype(float)
    db=cKDTree(dense(sample_mask(z,'b'))[0]).query(z['points_rc_a'].astype(float)+t)[0]
    for mode in ('old','full','tight'):
        ta=z['target_a'] if mode=='old' else arrays[mode+'_a'];tb=z['target_b'] if mode=='old' else arrays[mode+'_b']
        ii=np.flatnonzero(ta>=0); jj=ta[ii]
        dd=np.linalg.norm(z['points_rc_a'][ii].astype(float)+t-z['points_rc_b'][jj],axis=1)
        v=z['contour_valid_a'];near=v&(db<=3)
        metrics[mode]=dict(a=state_counts(ta,v),b=state_counts(tb,z['contour_valid_b']),
            near_le3=dict(match=int((near&(ta>=0)).sum()),ignore=int((near&(ta==-2)).sum()),unmatched=int((near&(ta==-1)).sum())),
            distances=dd.tolist())
    info.update(pair_id=pid,stage=a['stage'],recipe=case['record']['recipe'],label=a['label'],
                v14_fallback=bool(case['record'].get('v14_fallback')),metrics=metrics,
                seconds=time.monotonic()-start,all_nonlabel_arrays_identical=True)
    save_json(dest/'audit.json',info)
    return {k:v for k,v in info.items() if k not in ('pairs','policy','counts')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.evidence/'pilot.json').read_bytes());cases=manifest['cases']
    if args.limit:cases=cases[:args.limit]
    started=time.time();results=[]
    for i,c in enumerate(cases):
        results.append(one(c,args.evidence,args.out))
        if (i+1)%10==0:print(json.dumps(dict(done=i+1,total=len(cases),seconds=time.time()-started)),flush=True)
    save_json(args.out/'summary.json',dict(status='cpu_label_review_complete',cases=results,
              training_admitted=False,model_forwards=0,optimizer_updates=0,seconds=time.time()-started))

if __name__=='__main__':main()

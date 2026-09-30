"""Prespecified common-scale inference sensitivity; not per-case scale selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from experiments.rachel_n512_formal_30k.probe_local_score_domains import (
    Probe,FIELDS,save,load_winner,sealed,RachelPairDataset,make_ablation_loader)
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour


def scaled(arr,s):
    if s==1.: return [a.copy() for a in arr]
    dst=[a.copy() for a in arr];n=int(800*s);offset=(800-n)//2
    for side in (0,1):
        old=arr[side][0].squeeze().astype(np.uint8)
        mask=np.zeros((800,800),np.uint8)
        mask[offset:offset+n,offset:offset+n]=cv2.resize(old,(n,n),interpolation=cv2.INTER_NEAREST)
        pts,valid=extract_ordered_outer_contour(mask.astype(bool),cap=512,smoothing_sigma=3.)
        dst[side]=mask[None,None].astype(np.float32)
        dst[2+side]=np.zeros((1,512,2),np.float32);dst[4+side]=np.zeros((1,512),bool)
        dst[2+side][0,:len(pts)]=pts;dst[4+side][0,:len(valid)]=valid
    return dst


def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);sealed._set_determinism(260912)
    model,identity,thresholds=load_winner(args.training_run)
    device=torch.device('cuda:0');model=model.to(device).eval().requires_grad_(False)
    p=Probe(model,device,dest);items=[]
    manifest=[json.loads(x) for x in (Path(args.dataset)/'pairs/test.jsonl').read_text().splitlines() if x]
    key=lambda i:hashlib.sha256(manifest[i]['pair_id'].encode()).hexdigest()
    ix=[]
    for positive in (True,False): ix+=sorted([i for i,r in enumerate(manifest) if bool(r['label'])==positive],key=key)[:16]
    ds=RachelPairDataset(args.dataset,'test')
    for batch in make_ablation_loader(ds,ix,batch_size=4,num_workers=2,seed=260912,contour_cap=512):
        for i,k in enumerate(batch.pair_ids):
            items.append(('sim_scale',k,[getattr(batch,n)[i:i+1] for n in FIELDS],bool(batch.labels[i]),
                          batch.translation_a_to_b_rc[i] if batch.translation_valid[i] else None))
    root=Path(args.ood);meta=json.loads((root/'manifest.json').read_text())
    lookup={k:i for i,k in enumerate(meta['fragment_ids'])}
    with np.load(root/'inputs.npz') as f:
        masks,points,valid=f['packed_masks'],f['points'],f['valid']
    pairs=sorted(meta['pairs'],key=lambda r:hashlib.sha256(r['pair_id'].encode()).hexdigest())[:32]
    for r in pairs:
        a,b=lookup[r['fragment_a_id']],lookup[r['fragment_b_id']]
        items.append(('ood_scale',r['pair_id'],[np.unpackbits(masks[a],axis=1)[None,None].astype(np.float32),
            np.unpackbits(masks[b],axis=1)[None,None].astype(np.float32),points[a:a+1],points[b:b+1],valid[a:a+1],valid[b:b+1]],True,None))
    protocol=dict(status='running',model=identity,thresholds=thresholds,scales=[1.,.75,.5],
        counts={'ood':32,'sim_positive':16,'sim_negative':16},selection='sha256(pair_id) order; not score-selected',
        both_fragments_share_scale=True,individual_fragment_resize=False,training=False,
        no_per_case_best_scale_selection=True,canonical_results_unchanged=True)
    save(dest/'protocol.json',protocol);start=time.monotonic()
    for domain,k,arr,label,gt in items:
        for s in (1.,.75,.5):
            p.batch(domain,[k+'__scale'+str(s)],scaled(arr,s),[label],[None if gt is None else gt*s],
                    [dict(base_pair_id=k,scale=s)])
    save(dest/'results.json',dict(rows=p.rows))
    protocol.update(status='complete',elapsed_seconds=time.monotonic()-start)
    save(dest/'protocol.json',protocol)
    print(json.dumps(dict(status='complete',rows=len(p.rows),elapsed=protocol['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser()
    for name in ('training-run','dataset','ood','output'): a.add_argument('--'+name,required=True)
    run(a.parse_args())

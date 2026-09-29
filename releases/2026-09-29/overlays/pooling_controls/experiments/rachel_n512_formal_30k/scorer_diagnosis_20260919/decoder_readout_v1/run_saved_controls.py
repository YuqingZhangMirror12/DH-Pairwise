"""CPU frozen-readout screening on ALL saved failures and existing controls.

Not a full-population accuracy estimate. No Matcher/optimizer/threshold fit.
Results explicitly separate original-strict replay and sensitive rows.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, indent=2, allow_nan=False)); tmp.replace(path)


def outcome(logits, poses, threshold, gt, label):
    if not len(logits):
        return dict(winner=-1, accepted=False, correct=None if gt is None else False,
                    classification_correct=not label, score=0.)
    winner = max(range(len(logits)), key=lambda i: logits[i])
    score = 1 / (1 + math.exp(-max(-700, min(700, float(logits[winner])))))
    accepted = score >= threshold
    correct = None if gt is None else bool(np.linalg.norm(np.asarray(poses[winner]) - gt) <= 20)
    return dict(winner=winner, accepted=accepted, correct=correct,
                classification_correct=bool(accepted == label), score=score)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--audit-root', required=True); p.add_argument('--out', required=True)
    a=p.parse_args(); root=Path(a.audit_root); out=Path(a.out)
    if out.exists(): raise RuntimeError('immutable new results path required')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    all_rows=[]; sources={}
    for model in ('threshold_scratch_fixed','binary_patch','binary_stats','aggressive_binary_patch'):
        statepath=root/model/'head_state.pt'; sources[str(statepath)]=sha(statepath)
        state=torch.load(statepath, map_location='cpu', weights_only=False)
        layer=None
        if model != 'threshold_scratch_fixed':
            width=16 if model=='binary_stats' else 80
            layer=nn.Sequential(nn.Linear(width,64), nn.GELU(), nn.Linear(64,32),nn.GELU(),nn.Linear(32,1)).eval()
            layer.load_state_dict({k[len('cluster_mlp.'):]:v for k,v in state.items() if k.startswith('cluster_mlp.')})
        paths=[('all_failures',x) for x in sorted((root/model/'cases').glob('*/result.json'))]
        paths += [('matched_success_controls',x) for x in sorted((root/'controls'/model/'cases').glob('*/result.json'))]
        for population, path in paths:
            r=read(path); original=[c['logit'] for c in r['retained']]
            poses=[c['final_pose'] for c in r['retained']]
            variants={'baseline':original}; raw_max=[]
            if layer is None:
                cs=r['head']['clusters']
                variants.update(no_overlap=[x['logit']+x['overlap_penalty'] for x in cs],
                    no_conflict=[x['logit']+x['conflict_penalty'] for x in cs],
                    no_conflict_or_overlap=[x['logit']+x['conflict_penalty']+x['overlap_penalty'] for x in cs])
            else:
                for name in ['no_overlap','zero_mean','zero_max','zero_mean_max']:
                    if width==16 and name!='no_overlap': continue
                    variants[name]=[]
            assert sha(path.parent/'arrays.npz')==r['arrays_sha256']
            with np.load(path.parent/'arrays.npz') as data, torch.no_grad():
                for i in range(len(original)):
                    ids=data[f'c{i}/ids']; raw_max.append(float(data['q'][ids[:,0],ids[:,1]].max()))
                    if layer is None: continue
                    value=torch.from_numpy(data[f'c{i}/cluster_input'].copy())
                    assert abs(float(layer(value).squeeze())-original[i])<1e-5
                    for name in list(variants)[1:]:
                        v=value.clone()
                        if name=='no_overlap': v[-3]=0. # 13 of 16 statistics; no change to geometry
                        elif name=='zero_mean': v[:32]=0.
                        elif name=='zero_max': v[32:64]=0.
                        elif name=='zero_mean_max': v[:64]=0.
                        variants[name].append(float(layer(v).squeeze()))
            results={name:outcome(v,poses,r['threshold'],r['gt'],r['label']) for name,v in variants.items()}
            # Only change winner selection; preserve its ORIGINAL trained score.
            for name, values in [('raw_sum_q_rank',[c['sum_q'] for c in r['retained']]),('raw_max_q_rank',raw_max)]:
                if not values: results[name]=results['baseline']; continue
                j=max(range(len(values)),key=lambda i: values[i])
                z=outcome([original[j]],[poses[j]],r['threshold'],r['gt'],r['label']); z['winner']=j
                results[name]=z
            all_rows.append(dict(model=model,population=population,pair_id=r['pair_id'],name=r['name'],
                split=r['split'],kind=r['kind'],label=r['label'],layout_gt=r['gt'] is not None,
                original_strict=r['equivalence']['passed'],result_sha256=sha(path),
                arrays_sha256=r['arrays_sha256'],results=results))
    save(out,dict(status='complete',rows=all_rows,head_sha256=sources,
        scope='ALL previously enumerated failures plus saved matched controls; not population accuracy',
        weights_unchanged=True,training_performed=False,gt_used_to_predict=False,
        caveat='Zeroing a trained input block is an off-training-distribution intervention, not a trained replacement head. Raw-Q rank uses original neural score of its winner, not a calibrated Q classifier.'))
    print(json.dumps(dict(rows=len(all_rows),out=str(out))))


if __name__=='__main__': main()

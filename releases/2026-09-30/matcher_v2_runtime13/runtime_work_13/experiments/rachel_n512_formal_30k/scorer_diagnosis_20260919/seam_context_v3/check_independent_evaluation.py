"""Small CPU-only initial-checkpoint parity; never evaluates adapted epochs."""
import argparse
from dataclasses import fields
from pathlib import Path
import time
import torch
import numpy as np
from .config import Config
from .model import SeamContextModel
from .evaluate_external import load_inputs, NAMES
from .evaluate_independent import load_winner, setup, ARMS
from .prepare import save,sha


def values(model,inputs):
    with torch.no_grad():
        o=model(*inputs,decode=True,verify=True);v=o.verified[0]
        return dict(score=v.score.detach().clone(),has_candidate=v.has_candidate,
            translations=v.translations.detach().clone(),logits=v.logits.detach().clone(),
            q1=o.ot1.real_transport.detach().clone(),edges=o.records[0].edges.detach().clone(),
            proposals=[(tuple(c.edge_ids),c.translation.copy()) for c in o.candidates[0]])


def run(a):
    setup(260923);t0=time.time();root=Path(a.root)
    cp=torch.load(a.base,map_location='cpu',weights_only=False)
    cfg=Config(**{f.name:cp['config'][f.name] for f in fields(Config)})
    base=SeamContextModel(cfg);base.load_state_dict(cp['model'],strict=True);base.eval().requires_grad_(False)
    inputs=[]
    for split in ('dunhuang_cv','turufan'):
        _,batches,_,_=load_inputs(split,1);needed={True,False}
        for pairs,batch in batches:
            label=bool(pairs[0]['label'])
            if label not in needed:continue
            needed.remove(label)
            tensor=[torch.as_tensor(np.array(batch[k],copy=True),dtype=torch.bool if k.startswith('contour_valid') else torch.float32) for k in NAMES]
            inputs.append((pairs[0]['pair_id'],tensor))
            if not needed:break
    reference=[values(base,x) for _,x in inputs];del base
    result=dict(device='cpu',gpu_used=False,initial_epoch_only=True,base_sha256=sha(a.base),arms={},
        purpose='adapter regression check; no threshold fitting, no post-training real metrics')
    for arm in ARMS:
        model,state,selection=load_winner(root/arm,a.base,initial_parity=True);diffs=[]
        for (pid,x),ref in zip(inputs,reference):
            got=values(model,x)
            if got['has_candidate']!=ref['has_candidate'] or not torch.equal(got['edges'],ref['edges']):
                raise AssertionError('candidate identity changed at epoch0')
            if len(got['proposals'])!=len(ref['proposals']) or any(u[0]!=v[0] or not np.array_equal(u[1],v[1]) for u,v in zip(got['proposals'],ref['proposals'])):
                raise AssertionError('proposal set changed at epoch0')
            delta={k:float((got[k]-ref[k]).abs().max()) if ref[k].numel() else 0. for k in ('score','translations','logits','q1')}
            if delta['score']>2e-5 or delta['translations']>1e-3 or delta['logits']>2e-4 or delta['q1']>1e-6:
                raise AssertionError(str(delta))
            diffs.append(dict(pair_id=pid,delta=delta))
        result['arms'][arm]=dict(selected_epoch=selection['epoch'],checkpoint_sha256=sha(root/arm/'best.pt'),pairs=diffs)
        del model
        print(arm,'CPU epoch0 parity passed',flush=True)
    result.update(passed=True,seconds=time.time()-t0);save(a.out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);p.add_argument('--out',required=True)
    run(p.parse_args())

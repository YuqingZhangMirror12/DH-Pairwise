"""TRAIN-only gradient conflict measurement on the frozen B22 parameter state.

No optimizer updates. A negative cosine is a local optimization observation,
not proof that unshared feature extraction will improve held-out performance.
"""
import argparse
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
import time
import numpy as np
import torch
from .config import Config
from .model import SeamContextModel
from .data import Dataset,collate,to_device,INPUTS
from .losses import compute_loss
from .prepare import read,save,sha


def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=Config(**{f.name:cp['config'][f.name] for f in fields(Config)})
    model=SeamContextModel(cfg).cuda().eval();model.load_state_dict(cp['model'],strict=True)
    ds=Dataset(a.manifest);strata=defaultdict(list)
    for i,e in enumerate(ds.entries):strata[(e['recipe'],bool(e['label']))].append(i)
    rng=np.random.default_rng(260923)
    chosen=[]
    for key,indices in sorted(strata.items()):
        chosen.extend(int(i) for i in rng.choice(indices,min(5,len(indices)),replace=False))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    save(out/'protocol.json',dict(checkpoint_sha256=sha(a.checkpoint),sample_indices=chosen,
        sample_rule='up to5 per S7 recipe/label stratum, seed260923; TRAIN only',
        model_mode='eval for fixed deterministic activations, gradients enabled',optimizer_steps=0,
        matcher_objective='.5 OT + .5 support + .25 link + .05 balance',
        scorer_objective='candidate + .5 rank + .25 pose',teacher_candidates=False))
    named=[(n,p) for n,p in model.named_parameters() if n.startswith(('patch_encoder.','scale_gate.','arc_context.'))]
    params=[p for _,p in named];records=[];started=time.time()
    for index in chosen:
        batch=to_device(collate([ds[index]]),'cuda')
        o=model(*(batch[k] for k in INPUTS),decode=True,verify=True)
        _,c,counts=compute_loss(model,o,batch,'B',teacher_weight=0.)
        ml=.5*c['ot']+.5*c['support']+.25*c['link']+.05*c['balance']
        sl=c['candidate']+.5*c['rank']+.25*c['pose']
        mg=torch.autograd.grad(ml,params,retain_graph=True,allow_unused=True)
        sg=torch.autograd.grad(sl,params,allow_unused=True)
        stats={}
        for name,prefix in [('encoder',('patch_encoder.','scale_gate.')),('context',('arc_context.',)),('shared_total',('',))]:
            dot=nm=ns=0.
            for (n,p),u,v in zip(named,mg,sg):
                if not n.startswith(prefix):continue
                if u is not None:nm+=float(u.float().square().sum())
                if v is not None:ns+=float(v.float().square().sum())
                if u is not None and v is not None:dot+=float((u.float()*v.float()).sum())
            stats[name]=dict(matcher_norm=nm**.5,scorer_norm=ns**.5,cosine=dot/(nm*ns)**.5 if nm>1e-20 and ns>1e-20 else None)
        records.append(dict(pair_id=ds.entries[index]['pair_id'],recipe=ds.entries[index]['recipe'],label=bool(batch['labels'][0]),
            matcher_loss=float(ml),scorer_loss=float(sl),score=float(o.verified[0].score),counts=counts,gradients=stats))
        save(out/'records.json',records)
        print(len(records),len(chosen),records[-1]['recipe'],stats['encoder'],flush=True)
        del o,batch,mg,sg,ml,sl,c
    summary={}
    for name in ('encoder','context','shared_total'):
        values=[r['gradients'][name]['cosine'] for r in records if r['gradients'][name]['cosine'] is not None]
        summary[name]=dict(valid=len(values),negative_count=sum(v<0 for v in values),below_minus01=sum(v<-.1 for v in values),
            quantiles=np.quantile(values,[.1,.25,.5,.75,.9]).tolist() if values else None)
    save(out/'summary.json',dict(status='complete',n=len(records),seconds=time.time()-started,gradients=summary,
        optimizer_steps=0,parameters_changed=False,interpretation='local gradient conflict diagnostic, not generalization proof'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    run(p.parse_args())

"""Standalone original-source S7 M12 + matched C16 output/attention export.

Run with the old frozen source paths; never substitutes current model code.
The scorer and matcher remain frozen and receive only the original six tensors.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time


def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,allow_nan=False)+'\n')


def run(a):
    reg=read(a.registry)['old_matched_tokens'];sys.path[:0]=reg['pythonpath'].split(':')
    import numpy as np
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    # Exactly the historical sealed evaluator runtime, including convolution
    # TF32. Changing only matmul precision leaves patch features different.
    import random
    random.seed(260913);np.random.seed(260913);torch.manual_seed(260913);torch.cuda.manual_seed_all(260913)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    module=importlib.import_module(reg['module'])
    model,receipt=module.load_frozen_model(Path(reg['training_run']),'fixed_epoch',budget=16)
    assert receipt['checkpoint_sha256']==reg['checkpoint_sha256']
    model=model.cuda().eval().requires_grad_(False)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    save(out/'receipt.json',receipt)
    requests=read(a.cases)['cases'];wanted={r['pair_id'] for r in requests}
    captures={};active=[False];handles=[]
    def ahook(name):
        def hook(m,args,kwargs,output):
            if not active[0]:return
            q,k,v=args[:3];d=m.embed_dim;h=m.num_heads;hd=d//h
            q=torch.nn.functional.linear(q,m.in_proj_weight[:d],m.in_proj_bias[:d])
            k=torch.nn.functional.linear(k,m.in_proj_weight[d:2*d],m.in_proj_bias[d:2*d])
            q=q.reshape(1,-1,h,hd).transpose(1,2);k=k.reshape(1,-1,h,hd).transpose(1,2)
            w=((q.float()@k.float().transpose(-1,-2))/math.sqrt(hd)).softmax(-1).mean(1)[0]
            captures.setdefault(name,[]).append(w.cpu().numpy())
        return hook
    for name,m in model.score_head.named_modules():
        if isinstance(m,torch.nn.MultiheadAttention):handles.append(m.register_forward_hook(ahook(name),with_kwargs=True))
    def poolhook(m,args,output):
        if active[0]:captures.setdefault('pool',[]).append(output.squeeze(-1).softmax(0).cpu().numpy()[None])
    handles.append(model.score_head.head.pool_gate.register_forward_hook(poolhook))
    records=[];probe_records=[];started=time.time()
    base=Path('/root/autodl-tmp/rachel_score_design_20260913_001/real_domain_calibration_v1_20260921')
    real=Path('/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared')
    def down(m):
        ii=np.array_split(np.arange(len(m)),min(96,len(m)));jj=np.array_split(np.arange(m.shape[1]),min(96,m.shape[1]))
        return [[float(m[np.ix_(i,j)].max()) for j in jj] for i in ii]
    for split in ('dunhuang','dunhuang_cv','turufan'):
        path=real/'manifest.json' if split=='dunhuang' else base/('real' if split=='dunhuang_cv' else 'ood')/'manifest.json'
        meta=read(path);prepared=Path(meta.get('prepared',real))
        with np.load(prepared/'inputs.npz',allow_pickle=False) as z:arrays={k:z[k] for k in ('packed_masks','points','valid')}
        index={x:i for i,x in enumerate(meta['fragment_ids'])}
        # Original-only historical eight; all current CV benchmark pairs.
        pairs=[p for p in meta['pairs'] if p['pair_id'] in wanted] if split=='dunhuang' or a.probes_only else meta['pairs']
        for p in pairs:
            if any(r['pair_id']==p['pair_id'] for r in records):continue
            active[0]=p['pair_id'] in wanted;captures.clear()
            ix=[index[p['fragment_'+s+'_id']] for s in 'ab']
            inputs=[]
            for key in ('packed_masks','points','valid'):
                for i in ix:
                    v=arrays[key][i:i+1]
                    if key=='packed_masks':v=np.unpackbits(v,axis=-1)[:,None].astype(np.float32)
                    inputs.append(torch.as_tensor(v,device='cuda',dtype=torch.bool if key=='valid' else torch.float32))
            with torch.inference_mode():o=model(*inputs)
            sel=o.score_details['selection'];valid=bool(o.decision_valid[0]) if hasattr(o,'decision_valid') else bool(o.training_valid[0])
            s=float(o.fused_probability[0]);t=sel.translation_a_to_b_rc[0].cpu().numpy()
            row=dict(pair_id=p['pair_id'],split=split,score=s,decision_valid=valid,layout_valid=bool(sel.layout_valid[0]),
                translation=t.tolist() if np.isfinite(t).all() else None,
                endpoints_a=int(sel.mask_a[0].sum()),endpoints_b=int(sel.mask_b[0].sum()),
                inlier_count=int(sel.candidate_inliers[0].sum()))
            records.append(row)
            if active[0]:
                va,vb=arrays['valid'][ix[0]].astype(bool),arrays['valid'][ix[1]].astype(bool)
                q=o.assignment[0].cpu().numpy();pa,pb=arrays['points'][ix[0]],arrays['points'][ix[1]]
                data=dict(q=q[np.ix_(va,vb)],points_a=pa[va],points_b=pb[vb],features_a=o.token_features_a[0].cpu().numpy()[va],
                    features_b=o.token_features_b[0].cpu().numpy()[vb],selected_a=sel.mask_a[0].cpu().numpy(),selected_b=sel.mask_b[0].cpu().numpy(),
                    candidate_indices=sel.candidate_indices[0].cpu().numpy(),candidate_valid=sel.candidate_valid[0].cpu().numpy(),
                    candidate_inliers=sel.candidate_inliers[0].cpu().numpy())
                maps={'q':down(data['q'])}
                for key,vv in captures.items():
                    for direction,v in enumerate(vv):
                        data[key+'_'+str(direction)]=v
                        if direction==0 or key=='pool':maps[key+'_'+str(direction)]=down(v)
                file=hashlib.sha256(p['pair_id'].encode()).hexdigest()+'.npz';np.savez_compressed(out/file,**data)
                probe_records.append(dict(**row,heatmaps=maps,array_file=file))
            if len(records)%64==0:
                save(out/'status.json',dict(status='inference',processed=len(records),seconds=time.time()-started))
                print(len(records),round(time.time()-started,1),flush=True)
        save(out/'predictions.json',records);save(out/'probe_records.json',probe_records)
    for h in handles:h.remove()
    save(out/'status.json',dict(status='complete',count=len(records),probes=len(probe_records),seconds=time.time()-started,
        training=False,checkpoint_sha256=receipt['checkpoint_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registry',required=True);p.add_argument('--cases',required=True);p.add_argument('--out',required=True)
    p.add_argument('--probes-only',action='store_true')
    run(p.parse_args())

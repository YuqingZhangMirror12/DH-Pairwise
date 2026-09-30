"""Read-only frozen-v3 layer/transport probes; no training or target input.

Targets are joined only after forward and geometry-only diagnostic interventions.
Attention weights are observations, not causal explanations of model decisions.
"""
import argparse
import hashlib
from dataclasses import fields
import math
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .model import SeamContextModel
from .arc_context import Attention
from .evaluate_external import load_inputs, describe_prediction, attach_targets, NAMES, array
from .prepare import read, save, sha
from .seam_proposals import open_arc_indices


def downsample(m, bins=96):
    # Exact max within non-overlapping index bins; never interpret as new Q.
    m=np.asarray(m)
    ii=np.array_split(np.arange(m.shape[0]),min(bins,m.shape[0]))
    jj=np.array_split(np.arange(m.shape[1]),min(bins,m.shape[1]))
    return [[float(m[np.ix_(i,j)].max()) for j in jj] for i in ii]


def geometry_proxy(q, pa, pb, target):
    if target is None:return None
    error=np.linalg.norm(pb[None]-pa[:,None]-np.asarray(target),axis=-1)
    compatible=error<=20
    available=compatible.any(1)
    best=q.argmax(1)
    return dict(proxy='translation-compatible within20px, not annotated seam correspondence',
        eligible_a=int(available.sum()), mass_fraction=float(q[compatible].sum()/max(q.sum(),1e-12)),
        argmax_correct_a=int((compatible[np.arange(len(pa)),best]&available).sum()),
        any_compatible=bool(compatible.any()))


def attention_role(name):
    """Name each feature path explicitly when a Scorer-only clone is present."""
    if name in ('arc_context.blocks.0.cross_attn','arc_context.blocks.3.cross_attn'):
        return 'matcher_context'
    if name in ('scorer_features.arc_context.blocks.0.cross_attn',
                'scorer_features.arc_context.blocks.3.cross_attn'):
        return 'scorer_context'
    if name.startswith('verifier.blocks') and name.endswith('cross_attn') or name=='verifier.pool':
        return 'verifier'
    return None


def run(a, *, supplied_model=None, supplied_config=None, provenance=None):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if (supplied_model is None)!=(supplied_config is None):
        raise ValueError('supplied model and config must be paired')
    if supplied_model is None:
        cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        cfg=Config(**{f.name:cp['config'][f.name] for f in fields(Config)})
        model=SeamContextModel(cfg)
        model.load_state_dict(cp['model'],strict=True)
    else:
        model,cfg=supplied_model,supplied_config
    model=model.cuda().eval().requires_grad_(False)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    requested=read(a.cases)['cases'];records=[];capture={};handles=[]
    def attn_hook(name):
        def hook(module,args,kwargs,output):
            query,source,qv,sv=args[:4]
            bias=args[4] if len(args)>4 else kwargs.get('bias')
            b,n,_=query.shape
            q=module.q(query).reshape(b,n,module.heads,module.d).transpose(1,2)
            k=module.k(source).reshape(b,-1,module.heads,module.d).transpose(1,2)
            logits=(q.float()@k.float().transpose(-1,-2))/math.sqrt(module.d)
            if bias is not None:logits=logits+bias.float()
            safe=sv.clone();empty=~safe.any(1);safe[:,0]|=empty
            w=logits.masked_fill(~safe[:,None,None],-torch.inf).softmax(-1)
            w=w*qv[:,None,:,None]*(~empty)[:,None,None,None]
            capture.setdefault(name,[]).append(array(w.mean(1)))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,Attention) and attention_role(name):
            handles.append(module.register_forward_hook(attn_hook(name),with_kwargs=True))
    for name,module in [('quality_input',model.verifier.quality),('arc_outputs',model.arc_context),
                        ('edge_layer0',model.correspondence_context.blocks[0]),('edge_layer1',model.correspondence_context.blocks[1])]:
        def hook(m,args,output,name=name):
            capture[name]=args[0].detach().clone() if name=='quality_input' else output
        handles.append(module.register_forward_hook(hook))
    try:
        for split in sorted({r['split'] for r in requested}):
            meta,batches,source,dataset=load_inputs(split,1)
            wanted={r['pair_id']:r for r in requested if r['split']==split}
            reference={r['pair_id']:r for r in [__import__('json').loads(x) for x in (Path(a.reference)/split/'case_diagnostics.jsonl').read_text().splitlines()]}
            for pairs,batch in batches:
                pair=pairs[0]
                if pair['pair_id'] not in wanted:continue
                capture.clear()
                tensors=[torch.as_tensor(np.array(batch[k],copy=True),device='cuda',dtype=torch.bool if k.startswith('contour_valid') else torch.float32) for k in NAMES]
                with torch.no_grad():
                    o=model(*tensors,decode=True,verify=True)
                    pred=describe_prediction(o,0,pair,cfg)
                    variants={}
                    if pred['has_candidate']:
                        combined=capture['quality_input']
                        for variant in ('residual_half','residual_zero','overlap_zero','dustbin_zero'):
                            x=combined.clone()
                            if variant=='residual_half':x[:,cfg.dim+2:cfg.dim+4]=torch.asinh(torch.sinh(x[:,cfg.dim+2:cfg.dim+4])*.5)
                            if variant=='residual_zero':x[:,cfg.dim+2:cfg.dim+4]=0
                            if variant=='overlap_zero':x[:,cfg.dim+5:cfg.dim+8]=0
                            if variant=='dustbin_zero':x[:,cfg.dim+1]=0
                            quality=model.verifier.quality(x).squeeze(-1)
                            null=model.verifier.null(x.mean(0,keepdim=True)).squeeze()
                            variants[variant]=float((quality[pred['winner_index']]-null).sigmoid())
                # GT remains outside all network and intervention calls.
                row=attach_targets([pred],dict(meta,pairs=pairs),split,dataset=None)[0]
                old=reference[row['pair_id']]
                delta=abs(row['score']-old['score'])
                pose_delta=float(np.linalg.norm(np.asarray(row['translation'])-old['translation'])) if row['has_candidate'] and old['has_candidate'] else None
                n,m=int(o.ga.counts[0]),int(o.gb.counts[0]);rec=o.records[0]
                pa,pb=array(o.ga.points[0,:n]),array(o.gb.points[0,:m])
                arrays=dict(points_a=pa,points_b=pb,q0=array(o.ot0.real_transport[0,:n,:m]),q1=array(o.ot1.real_transport[0,:n,:m]),
                    affinity0=array(o.s0[0,:n,:m]),affinity1=array(o.s1[0,:n,:m]),
                    raw_a=array(o.fa[0,:n]),raw_b=array(o.fb[0,:m]),context_a=array(o.ha[0,:n]),context_b=array(o.hb[0,:m]),
                    edges=array(rec.edges).astype(np.int32),support=array(rec.support.sigmoid()),stop=array(rec.stop.sigmoid()),
                    edge_delta=array(rec.delta),edge_features=array(rec.features),edge_layer0=array(capture['edge_layer0']),
                    edge_layer1=array(capture['edge_layer1']),dustbin_a=array(o.ot1.dustbin_col[0,:n]),dustbin_b=array(o.ot1.dustbin_row[0,:m]))
                if hasattr(o,'scorer_fa'):
                    arrays.update(scorer_raw_a=array(o.scorer_fa[0,:n]),scorer_raw_b=array(o.scorer_fb[0,:m]),
                        scorer_context_a=array(o.scorer_ha[0,:n]),scorer_context_b=array(o.scorer_hb[0,:m]))
                maps={}
                for key in list(capture):
                    if not isinstance(capture[key],list):continue
                    values=capture[key]
                    for direction,v in enumerate(values):
                        ix=row['winner_index'] if key.startswith('verifier') else 0
                        if ix<0:continue
                        mm=v[ix]
                        if attention_role(key) in ('matcher_context','scorer_context'):
                            mm=mm[:n,:m] if direction==0 else mm[:m,:n]
                        elif row['has_candidate']:
                            ca,cb=row['candidates'][ix]['arc_point_counts']
                            if key.endswith('pool'):mm=mm[:,:ca if direction==0 else cb]
                            else:mm=mm[:ca,:cb] if direction==0 else mm[:cb,:ca]
                        arrays[key+'_'+str(direction)]=mm
                        if direction==0 or key.endswith('pool'):maps[key+'_'+str(direction)]=downsample(mm)
                pooling=[]
                if row['has_candidate']:
                    wi=row['winner_index'];edge_ids=o.candidates[0][wi].edge_ids
                    selected=array(rec.edges).astype(np.int64)[edge_ids]
                    for side,g in enumerate((o.ga,o.gb)):
                        count=int(g.counts[0]);arc=array(g.arc[0,:count]);perimeter=float(g.perimeter[0])
                        ids,u=open_arc_indices(arc,perimeter,selected[:,side],extension=256.)
                        support=np.isin(ids,selected[:,side]);internal=(u>=0)&(u<=1)
                        w=capture['verifier.pool'][side][wi,0,:len(ids)]
                        w=w/max(float(w.sum()),1e-12)
                        supp_arc=np.unique(arc[selected[:,side]])
                        span=perimeter-np.max((np.roll(supp_arc,-1)-supp_arc)%perimeter) if len(supp_arc)>1 else 0.
                        distance=u*max(span,1.);cs=np.r_[0,np.cumsum(w)];width=perimeter
                        for start in range(len(w)):
                            end=int(np.searchsorted(cs,cs[start]+.8,side='left'))-1
                            if start<=end<len(w):width=min(width,float(distance[end]-distance[start]))
                        pooling.append(dict(side='ab'[side],point_count=len(ids),support_points=int(support.sum()),
                            support_arc_span_px=float(span),perimeter_px=perimeter,
                            support_mass=float(w[support].sum()),internal_gap_mass=float(w[internal&~support].sum()),
                            exterior_context_mass=float(w[~internal].sum()),attention80_span_px=width,
                            effective_points=float(np.exp(-(w*np.log(np.maximum(w,1e-12))).sum())),
                            points_rc=array(g.points[0,ids]).tolist(),pool_weights=w.tolist(),roles=np.where(support,0,np.where(internal,1,2)).tolist()))
                file=hashlib.sha256(row['pair_id'].encode()).hexdigest()
                np.savez_compressed(out/(file+'.npz'),**arrays)
                target=row['target_translation_rc']
                proxies={k:geometry_proxy(arrays[k],pa,pb,target) for k in ('q0','q1')}
                retained=None
                if target is not None:
                    edges=arrays['edges'];err=np.linalg.norm(pb[edges[:,1]]-pa[edges[:,0]]-target,axis=1)
                    retained=dict(compatible_broad_edges=int((err<=20).sum()),broad_edge_count=len(edges),
                        compatible_support_mean=float(arrays['support'][err<=20].mean()) if (err<=20).any() else None,
                        incompatible_support_mean=float(arrays['support'][err>20].mean()) if (err>20).any() else None)
                record=dict(pair_id=row['pair_id'],split=split,reason=wanted[row['pair_id']]['reason'],
                    prediction=row,parity=dict(score_delta=delta,translation_delta_px=pose_delta),
                    heatmaps=dict(q0=downsample(arrays['q0']),q1=downsample(arrays['q1']),**maps),
                    proxies=proxies,broad_edges=retained,diagnostic_interventions=variants,
                    pooling=pooling,
                    heatmap_roles={key:attention_role(key.rsplit('_',1)[0]) for key in maps},
                    raw_array_file=file+'.npz',raw_array_sha256=sha(out/(file+'.npz')))
                records.append(record);save(out/'records.json',records)
                print(split,len(records),record['reason'],'parity',delta,flush=True)
        save(out/'status.json',dict(status='complete',count=len(records),checkpoint_sha256=sha(a.checkpoint),
            weights_changed=False,training=False,attention_is_not_causal_attribution=True,
            model_provenance=provenance,
            intervention_note='fixed original candidate and latent; diagnostic only, not test-performance variants'))
    finally:
        for h in handles:h.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--cases',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--out',required=True);run(p.parse_args())

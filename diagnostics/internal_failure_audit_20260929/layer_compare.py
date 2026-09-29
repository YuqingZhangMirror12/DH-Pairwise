"""Replay saved head layers and matched-control input-block contrasts only."""
import argparse
import hashlib
import itertools
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from audit import read,save

def network(state,prefix):
 indices=sorted({int(k.split('.')[1]) for k in state if k.startswith(prefix+'.')})
 modules=[]
 for i in range(max(indices)+1+(1 if prefix=='edge_mlp' else 0)):
  key=f'{prefix}.{i}.weight'
  modules.append(nn.Linear(state[key].shape[1],state[key].shape[0]) if key in state else nn.GELU())
 m=nn.Sequential(*modules);m.load_state_dict({k[len(prefix)+1:]:v for k,v in state.items() if k.startswith(prefix+'.')});m.eval().requires_grad_(False);return m

def contrast(net,start,end,patch):
 groups=({'mean_pool':list(range(32)),'max_pool':list(range(32,64)),'statistics':list(range(64,80))}
  if patch else {'quantity_coverage':list(range(8)),'residual_overlap':[8,9,10,13,14],'dustbin_outside_members':[11,12,15]})
 names=list(groups);values={}
 for flags in itertools.product((0,1),repeat=3):
  x=start.clone()
  for f,n in zip(flags,names):
   if f:x[groups[n]]=end[groups[n]]
  values[flags]=float(net(x).squeeze())
 shapley={}
 for k,n in enumerate(names):
  total=0.
  for flags in itertools.product((0,1),repeat=3):
   if flags[k]:continue
   changed=list(flags);changed[k]=1;count=sum(flags)
   total+=math.factorial(count)*math.factorial(2-count)/6*(values[tuple(changed)]-values[flags])
  shapley[n]=total
 delta=values[(1,1,1)]-values[(0,0,0)];assert abs(sum(shapley.values())-delta)<1e-5
 nodes,weights=np.polynomial.legendre.leggauss(128)
 alpha=torch.tensor((nodes+1)/2,dtype=start.dtype)[:,None]
 x=(start[None]+alpha*(end-start)[None]).requires_grad_(True)
 gradient=torch.autograd.grad(net(x).sum(),x)[0]
 ig=(end-start)*(gradient*torch.tensor(weights/2,dtype=start.dtype)[:,None]).sum(0)
 return dict(start_logit=values[(0,0,0)],end_logit=values[(1,1,1)],delta=delta,block_shapley=shapley,
  integrated_input_contributions=ig.tolist(),ig_completeness_error=float(ig.sum())-delta,
  caveat='Frozen-network sensitivity between two observed cluster inputs; matched only on point count and sumQ, not a causal identification.')

def case(root,pid):
 path=Path(root)/'cases'/hashlib.sha256(pid.encode()).hexdigest()[:24]
 return path,read(path/'result.json')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
 torch.set_num_threads(1);root=Path(a.root);plan=read(a.plan);out=[];details=[]
 for model in plan['models']:
  state=torch.load(root/model/'head_state.pt',map_location='cpu',weights_only=True);net=network(state,'cluster_mlp');patch=model!='binary_stats'
  edge=network(state,'edge_mlp') if patch else None
  for match in (m for m in plan['control_selection'] if m['model']==model):
   fp,f=case(root/model,match['failure']);cp,c=case(root/'controls'/model,match['control'])
   with np.load(fp/'arrays.npz') as z:fx=torch.from_numpy(z[f'c{f["selected"]}/cluster_input'].copy())
   with np.load(cp/'arrays.npz') as z:cx=torch.from_numpy(z[f'c{c["selected"]}/cluster_input'].copy())
   value=contrast(net,fx,cx,patch)
   assert abs(value['start_logit']-f['head']['clusters'][f['selected']]['logit'])<1e-5
   assert abs(value['end_logit']-c['head']['clusters'][c['selected']]['logit'])<1e-5
   out.append(dict(**match,failure_name=f['name'],control_name=c['name'],failure_score=f['score'],control_score=c['score'],
    both_reproduced=f['equivalence']['passed'] and c['equivalence']['passed'],comparison=value))
  if model=='aggressive_binary_patch':
   for path in (root/model/'cases').glob('*/result.json'):
    r=read(path)
    if not any('/'+x+' ' in r['name'] for x in ['85','414']):continue
    with np.load(path.parent/'arrays.npz') as z:arrays={k:z[k].copy() for k in z.files}
    summary=[]
    for i,c in enumerate(r['retained']):
     raw=torch.from_numpy(np.concatenate([arrays[f'c{i}/patch_context'],arrays[f'c{i}/edge_geometry']],axis=1))
     h=edge(raw);mx,index=h.max(0);pooled=arrays[f'c{i}/cluster_input'];assert np.allclose(mx.numpy(),pooled[32:64],atol=2e-5)
     ids=arrays[f'c{i}/ids'];q=arrays['q'][ids[:,0],ids[:,1]]
     source,count=np.unique(index.numpy(),return_counts=True)
     summary.append(dict(cluster=i,maximum_feature_source_distinct_pairs=len(source),
      top_source_pairs=[dict(pair=ids[k].tolist(),q=float(q[k]),max_dimensions=int(n)) for k,n in sorted(zip(source,count),key=lambda x:-x[1])],
      total_cluster_pairs=len(ids),score=c['score'],error_px=c['error_px']))
    correct_ids=[i for i,h in enumerate(r['hypotheses']) if h['error_px']<=20]
    union=np.unique(np.concatenate([arrays[f'h{i}/ids'] for i in correct_ids]),axis=0)
    best=max((c for c in r['retained'] if c['error_px']<=20),key=lambda c:c['score'])
    selected_ids=arrays[f'c{best["index"]}/ids']
    ids_set={tuple(x) for x in selected_ids};missing=[x for x in union if tuple(x) not in ids_set]
    details.append(dict(name=r['name'],pool_sources=summary,GT_near_hypotheses=correct_ids,
     oracle_union_pairs=len(union),oracle_union_q=float(arrays['q'][union[:,0],union[:,1]].sum()),
     best_correct_cluster=best['index'],correct_hypothesis_edges_outside_best_correct=len(missing),
     caveat='GT-near union is an oracle diagnostic, not a legal16px cluster or production proposal.'))
 save(root/'matched_control_layers.json',dict(status='complete',rows=out,examples=details,networks_read_only=True))
 print(dict(matched_failures=len(out),controls=sum(len(s['cases']) for s in plan['models'].values()),examples=len(details)))

if __name__=='__main__':main()

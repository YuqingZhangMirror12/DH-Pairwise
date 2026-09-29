"""Inspect the untruncated vote bins using saved Q, no Matcher reinference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import audit

p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
spec=audit.read(a.plan)['models'][a.model];model,_,_,common=audit.load(spec,a.model)
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.geometry import compact_contour,pair_frame
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import PairEvidence
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.legacy_pose_consensus import _cloud,_pose_key
out=Path(a.out);rows=[];builder=model.builder
for path in sorted((out/'cases').glob('*/result.json')):
 r=audit.read(path)
 if r['kind']!='no_correct_candidate':continue
 with np.load(path.parent/'arrays.npz') as z:data={k:z[k].copy() for k in ('q','points_a','points_b','unmatched_a','unmatched_b','cloud_ids','seeds')}
 t={k:torch.from_numpy(v) for k,v in data.items()};q=t['q'];ga=compact_contour(t['points_a'][None],torch.ones((1,len(q)),dtype=torch.bool));gb=compact_contour(t['points_b'][None],torch.ones((1,q.shape[1]),dtype=torch.bool))
 fa=torch.zeros((q.shape[0],96));fb=torch.zeros((q.shape[1],96))
 pair=PairEvidence(fa,fb,fa,fb,q,t['unmatched_a'],t['unmatched_b'],ga,gb,torch.arange(len(q)),torch.arange(q.shape[1]))
 cloud=_cloud(pair,builder.config);assert torch.equal(cloud.ids,t['cloud_ids'])
 actual=builder._seeds(cloud);assert torch.equal(actual,t['seeds'])
 n,_,rel=pair_frame(cloud.normal_a,cloud.normal_b,cloud.reliability_a,cloud.reliability_b)
 fractions=torch.tensor(builder.config.normal_vote_fractions,dtype=n.dtype)
 allowance=builder.geometry.damage_normal_upper_px*(rel>=builder.geometry.normal_reliability_min)
 votes=cloud.displacement[:,None]-n[:,None]*allowance[:,None,None]*fractions[None,:,None]
 weights=(cloud.q*cloud.arc_weight)[:,None].expand(-1,len(fractions))/len(fractions)
 bw=2*builder._local_scale(cloud);xy=votes.reshape(-1,2).numpy();w=weights.reshape(-1).numpy()
 bins=np.floor(xy/bw+.5).astype(np.int64);unique,inverse=np.unique(bins,axis=0,return_inverse=True)
 mass=np.bincount(inverse,weights=w,minlength=len(unique));sums=np.stack([np.bincount(inverse,weights=w*xy[:,d],minlength=len(unique)) for d in range(2)],-1)
 centers=sums/np.maximum(mass[:,None],1e-20);order=sorted(range(len(unique)),key=lambda k:(-mass[k],_pose_key(centers[k])))
 distances=np.linalg.norm(centers-np.array(r['gt']),axis=1);good=[k for k in order if distances[k]<=20]
 best=None if not good else good[0];rank=None if best is None else order.index(best)+1
 row=dict(pair_id=r['pair_id'],name=r['name'],strict_original_reproduced=r['equivalence']['passed'],
     bins=len(unique),bin_width_px=bw,gt_near_bins=len(good),best_gt_near_bin_mass_rank=rank,
     best_gt_near_bin_mass=None if best is None else float(mass[best]),top_bin_mass=float(mass[order[0]]),
     minimum_bin_gt_error_px=float(distances.min()),selected_near_GT20=r['stages']['seeds_near20'],
     top128_contains_GT20=bool(rank is not None and rank<=128),
     source_Q_sha256=audit.sha(path.parent/'arrays.npz'),GT_for_analysis_only=True,no_forward_or_seed_policy_change=True)
 rows.append(row)
audit.save(out/'seed_audit.json',dict(status='complete',rows=rows,semantics='All pre-truncation vote bins are read, actual16 seeds are exactly reproduced from saved CPU Q. GT only labels bins afterward.'))
print(json.dumps(dict(model=a.model,rows=len(rows),top128_near=sum(x['top128_contains_GT20'] for x in rows))))

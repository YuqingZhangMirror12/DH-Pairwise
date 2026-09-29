"""Independent count/identity/pose/membership checks on completed replay."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from raw_measure import save

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase1',required=True)
a=p.parse_args();r=Path(a.root);p1=Path(a.phase1)
assert json.loads((r/'complete.json').read_text())['pairs']==2905
assert not (r/'failure.json').exists()
summary=json.loads((r/'summary.json').read_text());checks=[];total=0
for split,n in [('sim_select',1500),('dunhuang_cv',803),('turufan',602)]:
 for variant in ['baseline','s8','s10','s12','s16']:
  rows=[json.loads(p.read_text()) for p in sorted((r/variant/split).glob('*.json'))]
  assert len(rows)==n and len({x['pair_id'] for x in rows})==n
  total+=n
  good=[x for x in rows if x['usable_gt']]
  assert sum(x['top_correct'] for x in good)==summary[variant][split]['top_correct']
  for row in rows:
   old=json.loads((p1/split/f"{row['index']:05d}.json").read_text())
   assert row['pair_id']==old['pair_id'] and row['label']==old['label'] and row['gt_excluded']==old['gt_excluded']
   assert row['retained']==min(8,row['prebudget']) and len(row['clusters'])==row['prebudget']
   arcs=[x['independent_arc_px'] for x in row['clusters']]
   if variant!='baseline':
    assert all(arcs[i]>=arcs[i+1] for i in range(len(arcs)-1))
    assert all(c['overlap']['fraction_sum_area']<.10 for c in row['clusters'] if c['overlap'].get('available'))
   if row['usable_gt']:
    for c in row['clusters']:
     assert abs(np.linalg.norm(np.asarray(c['pose'])-old['target_translation_rc'])-c['gt_error_px'])<1e-5
   if row['index']%97==0:
    raw=torch.load(p1/old['evidence_file'],map_location='cpu',weights_only=False)
    cloud=raw['proposals'].cloud;idx={tuple(x):i for i,x in enumerate(cloud.ids.tolist())}
    for c in row['clusters']:
     ids=[tuple(x) for x in c['member_edge_ids']];assert len(ids)==len(set(ids))
     indices=[idx[e] for e in ids]
     mass=float((cloud.q[indices]*cloud.arc_weight[indices]).sum())
     assert abs(mass-c['raw_q_arc_mass_px'])<1e-5
     if row['usable_gt']:
      err=np.linalg.norm(cloud.displacement[indices].numpy()-old['target_translation_rc'],axis=1)
      assert int((err<=20).sum())==c['correct_edge_count']
      assert bool((err<=20).any() and (err>40).any())==c['raw_edge_mixed20_40']
    checks.append([variant,split,row['index']])
assert summary['baseline']['dunhuang_cv']['top_correct']==216
assert summary['baseline']['dunhuang_cv']['budget_loss']==4
save(r/'verification.json',dict(passed=True,records=total,all_identity_denominator_pose_ranking_checked=True,
    sampled_raw_memberships=checks,baseline_top216_and_budget4_reproduced=True))
print(json.dumps(dict(passed=True,records=total,raw_checks=len(checks))))

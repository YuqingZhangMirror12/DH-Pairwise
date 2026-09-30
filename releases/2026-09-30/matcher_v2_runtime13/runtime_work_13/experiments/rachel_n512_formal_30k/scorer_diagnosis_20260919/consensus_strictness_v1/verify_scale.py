"""Independent row/union/pose checks and sampled raw mass/kernel recomputation."""
import argparse
import json
from pathlib import Path
import math

import numpy as np
import torch

from measure import read, save, sha, CompatibilityConfig, pair_frame


def run(root, phase1):
    root=Path(root);phase1=Path(phase1)
    protocol=read(root/'protocol.json');done=read(root/'complete.json');selection=read(root/'selection.json')
    assert done['status']=='complete' and done['summary_sha256']==sha(root/'summary.json')
    assert done['selection_sha256']==sha(root/'selection.json')
    assert selection['selected_before_real'] and selection['real_pairs_measured']==0
    p1=read(phase1/'protocol.json');geometry=CompatibilityConfig(**p1['geometry'])
    fixed={c['pair_id'] for c in p1['cases']['cases']}
    totals={name:{} for name in protocol['variants']}; sampled=0; checked=0
    for split,n in p1['expected'].items():
        rows={name:[] for name in protocol['variants']}
        for i in range(n):
            old=read(phase1/split/(str(i).zfill(5)+'.json'))
            hypothesis={h['index']:h for h in old['hypotheses']}
            correct={j for j,h in hypothesis.items() if h['edge_ids'] and h['gt_error_px'] is not None and h['gt_error_px']<=20}
            ref=set().union(*(set(map(tuple,hypothesis[j]['edge_ids'])) for j in correct)) if correct else set()
            is_sample=old['pair_id'] in fixed or i%100==0
            if is_sample:
                rawpath=phase1/old['evidence_file']
                assert sha(rawpath)==old['evidence_sha256']
                raw=torch.load(rawpath,map_location='cpu',weights_only=False);cloud=raw['proposals'].cloud
                normal,tangent,reliability=pair_frame(cloud.normal_a,cloud.normal_b,cloud.reliability_a,cloud.reliability_b)
                normal=normal.numpy(); reliable=reliability.numpy()>=geometry.normal_reliability_min
                spacing=.5*(cloud.spacing_a.numpy()+cloud.spacing_b.numpy())
                sn=np.maximum(spacing*geometry.normal_sigma_per_spacing,geometry.sigma_floor_px)
                qarc=(cloud.q*cloud.arc_weight).numpy()
                allids=list(map(tuple,cloud.ids.tolist()))
                refs={j:qarc*cloud.compatibility(raw['proposals'].hypotheses[j].translation,geometry).kernel.numpy()*
                    np.array([x in set(map(tuple,hypothesis[j]['edge_ids'])) for x in allids]) for j in hypothesis if hypothesis[j]['edge_ids']}
                sampled+=1
            for name,policy in protocol['variants'].items():
                row=read(root/name/split/(str(i).zfill(5)+'.json'))
                assert (row['pair_id'],row['label'],row['gt_excluded'])==(old['pair_id'],old['label'],old['gt_excluded'])
                assert row['correct_hypothesis_ids']==sorted(correct)
                radius=max(10,3*old['median_edgecloud_spacing_px']) if policy['adaptive'] else policy['radius_px']
                assert abs(radius-row['audit']['radius_px'])<1e-5
                assigned=[]
                for c in row['clusters']:
                    members=c['hypothesis_ids'];assigned.extend(members)
                    union=set().union(*(set(map(tuple,hypothesis[j]['edge_ids'])) for j in members))
                    assert union==set(map(tuple,c['original_union_ids']))
                    assert len(c['original_union_ids'])==len(union)
                    pose=np.array(c['translation_rc'])
                    distance=max(float(np.linalg.norm(np.array(hypothesis[j]['translation_rc'])-pose)) for j in members)
                    assert abs(distance-c['center_distance_max_px'])<1e-4
                    assert distance<=radius+1e-4
                    if old['gt_known']:
                        err=np.linalg.norm(pose-np.array(old['target_translation_rc']))
                        assert abs(err-c['gt_error_px'])<1e-7
                    else:
                        assert c['gt_error_px'] is None
                    if len(members)>1:
                        assert not c['overlap'].get('available') or c['overlap']['fraction_sum_area']<.1
                    if is_sample:
                        residual=cloud.displacement.numpy()-pose
                        rn=(residual*normal).sum(1)
                        offset=np.where(reliable,np.clip(rn,0,geometry.damage_normal_upper_px),0)
                        unexplained=residual-offset[:,None]*normal
                        member=(np.linalg.norm(unexplained,axis=1)<=radius+1e-6)&~(reliable&(rn < -3*sn))
                        loss=max(float(refs[j][~member].sum()/max(1e-30,float(refs[j].sum()))) for j in members)
                        assert abs(loss-c['mass_loss_guard'])<1e-4
                        if len(members)>1:
                            assert loss<=.05001
                        kernel=cloud.compatibility(torch.tensor(c['translation_rc']),geometry).kernel
                        expected={allids[k] for k in (kernel>=math.exp(-4.5)).nonzero().flatten().tolist()}
                        assert expected==set(map(tuple,c['edge_ids']))
                assert len(assigned)==len(set(assigned))
                assert set(assigned)=={j for j,h in hypothesis.items() if h['edge_ids']}
                assert row['retained_cluster_count']==min(len(row['clusters']),8)
                kept=[c for c in row['clusters'] if c['retained']]
                touching=[c for c in kept if set(c['hypothesis_ids'])&correct]
                complete=bool(correct and len(touching)==1 and correct<=set(touching[0]['hypothesis_ids'])
                    and touching[0]['gt_error_px']<=20 and not touching[0]['mixed20_40'])
                assert row['complete_correct_cluster']==complete
                rows[name].append(row);checked+=1
        for name,records in rows.items():
            valid=[r for r in records if r['gt_known'] and r['label'] and not r['gt_excluded']]
            totals[name][split]=dict(pairs=n,positive=sum(r['label'] for r in records),
                gt_valid=len(valid),complete_correct=sum(r['complete_correct_cluster'] for r in valid),
                top_correct=sum(r['top_cluster_correct'] for r in valid),mixed=sum(r['mixed20_40'] for r in valid))
            if split=='sim_select':assert len(valid)==750
            elif split=='dunhuang_cv':assert len(valid)==292 and sum(r['gt_excluded'] for r in records)==3
            else:assert len(valid)==0
    summary=read(root/'summary.json')
    for name,by_split in totals.items():
        for split,values in by_split.items():
            expected=summary['groups'][name][split]
            for a,b in [('complete_correct','complete_correct_count'),('top_correct','top_cluster_correct'),('mixed','mixed20_40')]:
                assert values[a]==expected[b]
    passing=[name for name in protocol['variants'] if selection['candidates'][name]['user_criteria_pass']]
    assert passing==selection['passing']
    assert selection['selected']==(passing[0] if passing else None)
    record=dict(status='passed',variant_pair_rows=checked,raw_pairs_sampled=sampled,
        checks=['all pair IDs/labels/GT exclusions','all centers/GT distances','all exact unique unions and partitioned original hypotheses',
            'all 8-candidate budgets','all complete-correct counts and selection rules','sampled raw95%mass and unchanged directional membership'],
        totals=totals,script_sha256=sha(__file__),no_claim_of_scorer_improvement=True)
    save(root/'verification.json',record);print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--phase1',default='/root/autodl-tmp/consensus_strictness_20260925/results_phase1')
    a=p.parse_args();torch.set_num_threads(1);run(a.root,a.phase1)

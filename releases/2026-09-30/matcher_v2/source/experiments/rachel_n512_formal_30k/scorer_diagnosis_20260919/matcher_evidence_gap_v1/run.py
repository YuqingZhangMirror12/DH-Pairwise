"""CPU-only paired geometry and16/20px candidate replay; no network forwards."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from geometry_diagnostics import measure, quant
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import observed_arc_cells
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.threshold_builder import ThresholdPoseBuilder, ThresholdPolicy


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');os.replace(tmp,p)


def init(phase, prior, helper):
    global ROOT, PRIOR, REPLAY
    torch.set_num_threads(1);cv2.setNumThreads(1)
    ROOT, PRIOR=Path(phase),Path(prior)
    spec=importlib.util.spec_from_file_location('frozen_threshold_replay',helper)
    REPLAY=importlib.util.module_from_spec(spec);spec.loader.exec_module(REPLAY)
    REPLAY.init(phase,'/nonexistent-read-only-helper-output')


def overlap_factory(masks):
    area=[float(m.sum()) for m in masks]
    def overlap(pose):
        dr,dc=map(int,pose.round().tolist());aa,bb=masks
        r0,c0=max(0,-dr),max(0,-dc);r1,c1=min(aa.shape[0],bb.shape[0]-dr),min(aa.shape[1],bb.shape[1]-dc)
        n=float((aa[r0:r1,c0:c1]*bb[r0+dr:r1+dr,c0+dc:c1+dc]).sum()) if r1>r0 and c1>c0 else 0.
        return dict(available=True,intersection_px=n,fraction_min_area=n/max(1.,min(area)),fraction_sum_area=n/max(1.,sum(area)))
    return overlap


def union_stats(ids,cloud,truth):
    ids=sorted(set(map(tuple,ids)));lookup={tuple(e):i for i,e in enumerate(cloud.ids.tolist())}
    ii=torch.tensor([lookup[e] for e in ids],dtype=torch.long)
    q=cloud.q[ii].double().numpy();w=cloud.arc_weight[ii].double().numpy()
    gt_good=np.linalg.norm(cloud.displacement[ii].double().numpy()-np.asarray(truth),axis=1)<=20 if truth is not None else None
    ua=len({a for a,b in ids});ub=len({b for a,b in ids})
    return dict(n=len(ids),endpoints_mean=(ua+ub)/2,q_sum=float(q.sum()),
                q_mean=float(q.mean()) if len(q) else None,mass=float((q*w).sum()),
                gt20_edges=int(gt_good.sum()) if gt_good is not None else None,
                gt20_mass=float((q[gt_good]*w[gt_good]).sum()) if gt_good is not None else None)


def augment_description(description,raw,truth):
    for c in description['clusters']:
        ids=set().union(*(set(map(tuple,raw['proposals'].hypotheses[i].edge_ids.tolist())) for i in c['members']))
        c['evidence']=union_stats(ids,raw['proposals'].cloud,truth)
        assert c['evidence']['n']==c['union_edge_count']
    good=[c for c in description['clusters'] if c['retained'] and c['gt_error_px'] is not None and c['gt_error_px']<=20]
    description['best_correct']=good[0]['evidence'] if good else None
    return description


@torch.no_grad()
def worker(task):
    split,i=task;row=read(ROOT/split/('%05d.json'%i));old=read(PRIOR/split/('%05d.json'%i))
    assert row['pair_id']==old['pair_id']
    path=ROOT/row['evidence_file'];assert sha(path)==row['evidence_sha256']
    raw=torch.load(path,map_location='cpu',weights_only=False)
    masks=REPLAY.masks_for(split,row);native=raw['proposals']
    cells=tuple(observed_arc_cells(g,REPLAY.CONFIG.observation_radius_px)[0] for g in (raw['geometry_a'],raw['geometry_b']))
    builder=ThresholdPoseBuilder(REPLAY.GEOMETRY,REPLAY.CONFIG,ThresholdPolicy(pose_diameter_px=20.))
    out=builder.build_from_hypotheses(native.cloud,native.hypotheses,native.seeds,overlap_fn=overlap_factory(masks),cells=cells)
    for c in builder.all_clusters:
        assert c.actual_diameter_px<=20.0001
        assert bool(((c.translation-c.member_translations_rc).norm(dim=1)<=20.0001).all())
    # Labels joined after both policies' clustering; no label-based admission.
    truth=row['target_translation_rc'] if row['label'] and row['gt_known'] and not row['gt_excluded'] else None
    new=REPLAY.describe(builder.all_clusters,len(out.clusters),raw,row,True)
    pair=dict(split=split,index=i,pair_id=row['pair_id'],label=bool(row['label']),usable_gt=truth is not None,
              gt_excluded=row['gt_excluded'],recipe=row['recipe'],source_family=row.get('source_family'),
              raw_q_sha256=row['evidence_sha256'],baseline_sha256=sha(PRIOR/split/('%05d.json'%i)),
              threshold16=augment_description(old['threshold16'],raw,truth),
              threshold20=augment_description(new,raw,truth))
    if truth is not None:
        cloud=native.cloud;hs=native.hypotheses
        good=[h for h in hs if len(h.edge_ids) and np.linalg.norm(h.translation.numpy()-truth)<=20]
        native_ids=set().union(*(set(map(tuple,h.edge_ids.tolist())) for h in good)) if good else set()
        pair['raw_cloud']=union_stats(cloud.ids.tolist(),cloud,truth)
        pair['native_correct_union']=union_stats(native_ids,cloud,truth)
        a,b=[raw['geometry_'+s].points[0,:raw['q'].shape[k]].numpy() for k,s in enumerate('ab')]
        pair['geometry']=measure(masks[0].numpy(),masks[1].numpy(),truth,a,b)
        if split=='sim_select':
            sample,report,entry=REPLAY.DATASET[i]
            pair['sim_metadata']=dict(latent_seam=report.get('latent_seam'),
                quota_length_px=report.get('compound',{}).get('quota_length_px'),
                inherited_match_count=report.get('inherited_match_count'),
                source_pair_id=report.get('source_pair_id'),
                source_family=entry.get('source_row',{}).get('fragment_a',{}).get('split_unit_id'))
            pair['source_family']=pair['sim_metadata']['source_family']
        for policy in ('threshold16','threshold20'):
            s=pair[policy]['best_correct'];g=pair['geometry']
            pair[policy]['correct_n_zero_filled']=s['n'] if s else 0
            pair[policy]['correct_mass_zero_filled']=s['mass'] if s else 0
            pair[policy]['correct_edges_per100px_L40']=(100*(s['n'] if s else 0)/g['length_40_px']) if g['length_40_px'] else None
            pair[policy]['correct_endpoints_over_potential20']=(s['endpoints_mean'] if s else 0)/g['potential_tokens_20_mean'] if g['potential_tokens_20_mean'] else None
    return pair


def summarize(rows):
    result={}
    for split in sorted({r['split'] for r in rows}):
        allr=[r for r in rows if r['split']==split];pos=[r for r in allr if r['usable_gt']];neg=[r for r in allr if not r['label']]
        s=dict(pairs=len(allr),positive_with_gt=len(pos),negative=len(neg),excluded=sum(r['gt_excluded'] for r in allr),geometry={},policies={})
        for key in pos[0]['geometry'] if pos else []:
            if isinstance(pos[0]['geometry'][key],(int,float)) or pos[0]['geometry'][key] is None:
                s['geometry'][key]=quant([r['geometry'][key] for r in pos])
        for key in ('raw_cloud','native_correct_union'):
            s[key]={m:quant([r[key][m] for r in pos]) for m in ('n','endpoints_mean','mass','gt20_edges','gt20_mass','q_mean')}
        for policy in ('threshold16','threshold20'):
            s['policies'][policy]={k:sum(r[policy][k] for r in pos) for k in ('top_correct','coverage_prebudget','coverage_retained','budget_loss','single_complete_correct_native','native_mixed20_40','native_mixed20_20')}
            s['policies'][policy].update({k:quant([r[policy][k] for r in pos]) for k in ('correct_n_zero_filled','correct_mass_zero_filled','correct_edges_per100px_L40','correct_endpoints_over_potential20')})
            s['policies'][policy]['negative_max_mass']=quant([r[policy]['max_cluster_raw_q_arc_mass_px'] for r in neg])
        both=[r for r in pos if r['threshold16']['best_correct'] and r['threshold20']['best_correct']]
        s['paired_n_change_both_correct']=quant([r['threshold20']['best_correct']['n']-r['threshold16']['best_correct']['n'] for r in both])
        s['paired_n_increased']=sum(r['threshold20']['best_correct']['n']>r['threshold16']['best_correct']['n'] for r in both)
        s['both_correct_pairs']=len(both)
        s['length_strata']=[]
        for lo,hi in ((0,256),(256,512),(512,768),(768,1024),(1024,float('inf'))):
            rr=[r for r in pos if lo<=r['geometry']['length_40_px']<hi]
            s['length_strata'].append(dict(lower=lo,upper=hi if np.isfinite(hi) else None,pairs=len(rr),
                n16=quant([r['threshold16']['correct_n_zero_filled'] for r in rr]),
                raw_gt20=quant([r['raw_cloud']['gt20_edges'] for r in rr]),
                density16=quant([r['threshold16']['correct_edges_per100px_L40'] for r in rr]),
                length10_over40=quant([r['geometry']['length10_over40'] for r in rr]),
                model_spacing=quant([r['geometry']['model_spacing_mean_px'] for r in rr])))
        result[split]=s
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase1',required=True);p.add_argument('--prior',required=True)
    p.add_argument('--helper',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    root=Path(a.out);root.mkdir(parents=True,exist_ok=False);start=time.time()
    save(root/'protocol.json',dict(schema='matcher-evidence-three-factors/1',phase1=a.phase1,prior=a.prior,
        phase1_protocol_sha256=sha(Path(a.phase1)/'protocol.json'),prior_complete_sha256=sha(Path(a.prior)/'complete.json'),
        helper_sha256=sha(a.helper),script_sha256=sha(__file__),geometry_script_sha256=sha(Path(__file__).with_name('geometry_diagnostics.py')),
        policies=[16,20],threshold_semantics='maximum pairwise original native-pose diameter',
        candidate_budget=8,overlap_limit=.10,gt_correctness_px=20,matcher_forward=False,scorer_forward=False,
        training_updates=0,gpu_used=False,test_accessed=False,diagnostic_not_new_production_threshold=True,
        geometry='observed GT proximity band; no semantic/original seam GT in real data',limit=a.limit))
    rows=[]
    try:
        tasks=[(s,i) for s,n in [('sim_select',1500),('dunhuang_cv',803)] for i in range(min(a.limit or n,n))]
        with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(a.phase1,a.prior,a.helper)) as pool:
            for r in pool.map(worker,tasks,chunksize=1):
                save(root/r['split']/('%05d.json'%r['index']),r);rows.append(r)
                if len(rows)%100==0:save(root/'status.json',dict(status='diagnosing',pairs=len(rows),expected=len(tasks),elapsed=time.time()-start))
        save(root/'summary.json',summarize(rows))
        save(root/'complete.json',dict(status='complete',pairs=len(rows),expected=len(tasks),limit=a.limit,elapsed_seconds=time.time()-start,
            full_population=not a.limit,matcher_forward=False,scorer_forward=False,training_updates=0))
    except Exception as e:
        save(root/'failure.json',dict(error=repr(e),pairs=len(rows)));raise


if __name__=='__main__':main()

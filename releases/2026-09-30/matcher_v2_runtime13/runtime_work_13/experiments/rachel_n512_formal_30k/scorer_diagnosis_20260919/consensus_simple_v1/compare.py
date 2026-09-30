"""Same frozen phase1 population, new raw-mode builder; SELECT precedes real."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch

from raw_measure import save,q
from simple_builder import SimplePoseBuilder,SimplePolicy
# Reuse phase1 data loader/identities and masks, never rerun the Matcher.
from measure import Dataset,CompatibilityConfig,ProposalConfig,PairEvidence,observed_arc_cells,CV,make_cloud
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import material_overlap

RADII=(8,10,12,16)


def init(root,out):
    global ROOT,OUT,DS,CFG,GEOM,REAL
    ROOT,OUT=Path(root),Path(out)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    GEOM=CompatibilityConfig(**protocol['geometry']);CFG=ProposalConfig(**protocol['proposal_config'])
    DS=Dataset(protocol['select_spec']['path'],protocol['select_spec']['sha256']);REAL={}


def masks_for(split,row):
    if split=='sim_select':
        sample,_,_=DS[row['index']]
        return [torch.from_numpy(np.array(m.squeeze(),dtype=np.float32,copy=True)) for m in (sample.mask_a,sample.mask_b)]
    if split not in REAL:
        meta=json.loads((CV/('real' if split=='dunhuang_cv' else 'ood')/'manifest.json').read_text())
        with np.load(Path(meta['prepared'])/'inputs.npz',allow_pickle=False) as z: masks=z['packed_masks']
        REAL[split]=(meta,masks,{f:i for i,f in enumerate(meta['fragment_ids'])})
    meta,masks,index=REAL[split];item=meta['pairs'][row['index']]
    assert item['pair_id']==row['pair_id']
    return [torch.from_numpy(np.unpackbits(masks[index[item['fragment_'+s+'_id']]],axis=-1).astype(np.float32)) for s in ('a','b')]


def edge_set(x): return set(map(tuple,x.tolist()))


def measure_clusters(row,raw,clusters,retained,pair,geometry,policy,baseline=False):
    cloud=raw['proposals'].cloud; x=cloud.displacement.numpy(); truth=row['target_translation_rc']
    use_gt=bool(row['label'] and row['gt_known'] and not row['gt_excluded'])
    labels=np.linalg.norm(x-np.asarray(truth),axis=1) if use_gt else None
    ids=edge_set(cloud.ids)
    id_to_index={tuple(e):i for i,e in enumerate(cloud.ids.tolist())}
    correct_ids={e for e,i in id_to_index.items() if labels[i]<=20} if use_gt else set()
    old_h=raw['proposals'].hypotheses
    old_good=[h for h in old_h if use_gt and len(h.edge_ids) and float((h.translation-torch.tensor(truth)).norm())<=20]
    old_ref=set().union(*(edge_set(h.edge_ids) for h in old_good)) if old_good else set()
    cr=[]
    for i,c in enumerate(clusters):
        e=edge_set(c.edge_ids);ci=[id_to_index[k] for k in e]
        d=None if not use_gt else float(np.linalg.norm(c.translation.numpy()-np.asarray(truth)))
        rawmass=float((cloud.q[ci]*cloud.arc_weight[ci]).sum()) if ci else 0.
        near=0 if not use_gt else int(sum(labels[j]<=20 for j in ci))
        far20=0 if not use_gt else int(sum(labels[j]>20 for j in ci))
        far40=0 if not use_gt else int(sum(labels[j]>40 for j in ci))
        cr.append(dict(index=i,retained=i<retained,pose=c.translation.tolist(),gt_error_px=d,
            member_edge_ids=sorted(e),mode_ids=list(c.merged_hypothesis_ids),
            independent_arc_px=getattr(c,'independent_arc_px',None),raw_q_arc_mass_px=rawmass,
            directional_mass_px=c.absolute_support_mass_px,
            correct_edge_count=near,wrong20_edge_count=far20,wrong40_edge_count=far40,
            raw_edge_mixed20_40=bool(near and far40),raw_edge_mixed20_20=bool(near and far20),
            correct_raw_edge_coverage=len(e&correct_ids)/len(correct_ids) if correct_ids else None,
            old_reference_union_coverage=len(e&old_ref)/len(old_ref) if old_ref else None,
            contact_shift_px=getattr(c,'contact_shift_px',None),overlap=c.overlap))
    kept=cr[:retained]
    good=[c for c in kept if c['gt_error_px'] is not None and c['gt_error_px']<=20]
    precorrect=[c for c in cr if c['gt_error_px'] is not None and c['gt_error_px']<=20]
    largestgood=good[0] if good else None
    directional_coverage=None
    if old_ref:
        directional_coverage=0.
        if largestgood is not None:
            ker=cloud.compatibility(torch.tensor(largestgood['pose']),geometry).kernel.numpy()
            directional={e for e,j in id_to_index.items() if ker[j]>=math.exp(-4.5)}
            directional_coverage=len(directional&old_ref)/len(old_ref)
    targets=[c for c in kept if c['correct_edge_count']]
    jaccard=0
    for i,a in enumerate(kept):
        ea=set(map(tuple,a['member_edge_ids']))
        for b in kept[i+1:]:
            eb=set(map(tuple,b['member_edge_ids']));u=ea|eb
            jaccard+=bool(u and len(ea&eb)/len(u)>=.9)
    closest=[np.linalg.norm(np.asarray(a['pose'])-b['pose']) for i,a in enumerate(kept) for b in kept[i+1:]]
    maxmass=max([c['raw_q_arc_mass_px'] for c in kept],default=0.)
    # Full-Q diagnostic uses the ranking winner. No Scorer or accept decision.
    fullmass=None
    if kept:
        pose=torch.tensor(kept[0]['pose'])
        kernel=torch.cat([pair.compatibility(pose,geometry,s,min(s+64,len(pair.local_a)))[1].kernel for s in range(0,len(pair.local_a),64)])
        fullmass=float((pair.q*kernel).sum())
    return dict(pair_id=row['pair_id'],index=row['index'],split=row['split'],label=row['label'],
        gt_excluded=row['gt_excluded'],usable_gt=use_gt,recipe=row['recipe'],
        source_family=row.get('source_family'),base_pair_id=row.get('base_pair_id'),
        cloud_edges=len(cloud.ids),reference_correct_raw_edges=len(correct_ids),old_reference_edges=len(old_ref),
        prebudget=len(cr),retained=len(kept),top_correct=bool(kept and kept[0]['gt_error_px'] is not None and kept[0]['gt_error_px']<=20),
        coverage_prebudget=bool(precorrect),coverage_retained=bool(good),
        complete_correct_raw_support=bool(correct_ids and any(c['correct_raw_edge_coverage']==1. for c in good)),
        single_complete_correct_raw_support=bool(correct_ids and len(targets)==1 and targets[0]['gt_error_px']<=20 and targets[0]['correct_raw_edge_coverage']==1.),
        raw_edge_mixed20_40=sum(c['raw_edge_mixed20_40'] for c in cr),
        raw_edge_mixed20_20=sum(c['raw_edge_mixed20_20'] for c in cr),
        retained_raw_edge_mixed20_40=sum(c['raw_edge_mixed20_40'] for c in kept),
        retained_raw_edge_mixed20_20=sum(c['raw_edge_mixed20_20'] for c in kept),
        old_reference_union_coverage=(largestgood['old_reference_union_coverage'] if largestgood else 0.) if old_ref else None,
        old_reference_directional_coverage=directional_coverage,
        top_correct_raw_coverage=(largestgood['correct_raw_edge_coverage'] if largestgood else 0.) if correct_ids else None,
        jaccard090_pairs=jaccard,min_pose_separation_px=min(closest) if closest else None,
        maximum_cluster_mass_px=maxmass,maximum_mass_share=maxmass/sum(c['raw_q_arc_mass_px'] for c in kept) if maxmass else None,
        ranking_winner_fullq_mass=fullmass,clusters=cr,ranking='legacy directional mass' if baseline else 'independent arc')


@torch.no_grad()
def worker(task):
    split,i=task
    row=json.loads((ROOT/split/f'{i:05d}.json').read_text())
    path=ROOT/row['evidence_file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['evidence_sha256']: raise ValueError('frozen evidence changed')
    raw=torch.load(path,map_location='cpu',weights_only=False)
    masks=masks_for(split,row);na,nb=raw['q'].shape
    a=torch.zeros((na,1));b=torch.zeros((nb,1))
    pair=PairEvidence(a,b,a,b,raw['q'],raw['unmatched_a'],raw['unmatched_b'],raw['geometry_a'],raw['geometry_b'],
        torch.arange(na),torch.arange(nb),*masks,row['numeric_valid'])
    cells=tuple(observed_arc_cells(g,CFG.observation_radius_px)[0] for g in (pair.ga,pair.gb))
    # Exact old raster overlap, but avoid recomputing constant image areas.
    area_a,area_b=map(lambda m:float(m.sum()),masks)
    def overlap(t):
        dr,dc=map(int,t.round().tolist());aa,bb=masks
        r0,c0=max(0,-dr),max(0,-dc);r1,c1=min(aa.shape[0],bb.shape[0]-dr),min(aa.shape[1],bb.shape[1]-dc)
        n=float((aa[r0:r1,c0:c1]*bb[r0+dr:r1+dr,c0+dc:c1+dc]).sum()) if r1>r0 and c1>c0 else 0.
        return dict(available=True,intersection_px=n,fraction_min_area=n/max(1.,min(area_a,area_b)),fraction_sum_area=n/max(1.,area_a+area_b))
    answer={}
    allc=raw['prebudget_clusters'];n=len(raw['proposals'].clusters)
    base=measure_clusters(row,raw,allc,n,pair,GEOM,None,True)
    save(OUT/'baseline'/split/f'{i:05d}.json',base)
    answer['baseline']={k:v for k,v in base.items() if k!='clusters'}
    for radius in RADII:
        builder=SimplePoseBuilder(GEOM,CFG,SimplePolicy(radius))
        p=builder.build_from_cloud(raw['proposals'].cloud,overlap_fn=overlap,cells=cells)
        result=measure_clusters(row,raw,builder.all_clusters,len(p.clusters),pair,GEOM,builder.policy)
        result['audit']=builder.audit
        save(OUT/f's{radius}'/split/f'{i:05d}.json',result)
        answer[f's{radius}']={k:v for k,v in result.items() if k!='clusters'}
    return answer


def summarize(rows):
    gt=[r for r in rows if r['usable_gt']];neg=[r for r in rows if not r['label']]
    r=dict(pairs=len(rows),gt_positive_pairs=len(gt),negative_pairs=len(neg),
        top_correct=sum(x['top_correct'] for x in gt),coverage_prebudget=sum(x['coverage_prebudget'] for x in gt),
        coverage_retained=sum(x['coverage_retained'] for x in gt),budget_loss=sum(x['coverage_prebudget'] and not x['coverage_retained'] for x in gt),
        complete_raw=sum(x['complete_correct_raw_support'] for x in gt),single_complete_raw=sum(x['single_complete_correct_raw_support'] for x in gt),
        complete_raw_fraction=sum(x['complete_correct_raw_support'] for x in gt)/len(gt) if gt else None,
        single_complete_raw_fraction=sum(x['single_complete_correct_raw_support'] for x in gt)/len(gt) if gt else None,
        raw_edge_mixed20_40=sum(x['raw_edge_mixed20_40'] for x in gt),raw_edge_mixed20_20=sum(x['raw_edge_mixed20_20'] for x in gt),
        retained_raw_edge_mixed20_40=sum(x['retained_raw_edge_mixed20_40'] for x in gt),retained_raw_edge_mixed20_20=sum(x['retained_raw_edge_mixed20_20'] for x in gt),
        jaccard090_pairs=sum(x['jaccard090_pairs']>0 for x in rows),retained_count=q([x['retained'] for x in rows]),
        negatives={key:q([x[key] for x in neg if x[key] is not None]) for key in ['maximum_cluster_mass_px','maximum_mass_share','ranking_winner_fullq_mass','retained','prebudget']})
    for key in ['old_reference_union_coverage','old_reference_directional_coverage','top_correct_raw_coverage']:
        r[key]=q([x[key] for x in gt if x[key] is not None])
    return r


def controls(geometry,radius):
    results=[]
    for name,centers in [('separated60',[0,60]),('drift80',[0,20,40,60,80]),('four_plus_distractor',[0,4,10,15,200])]:
        for spacing in (3.,4.,4.6,5.):
            c=make_cloud(centers);c=replace(c,spacing_a=torch.ones(len(c.ids))*spacing,spacing_b=torch.ones(len(c.ids))*spacing)
            b=SimplePoseBuilder(geometry,policy=SimplePolicy(radius));p=b.build_from_cloud(c)
            groups=[sorted(set((cl.edge_ids[:,0]//4).tolist())) for cl in p.clusters]
            passed=all(len(g)==1 for g in groups) if name=='separated60' else all(len(g)<5 for g in groups) if name=='drift80' else all(not (4 in g and len(g)>1) for g in groups)
            results.append(dict(name=name,spacing=spacing,groups=groups,passed=passed,
                complete_four=any(g==[0,1,2,3] for g in groups) if name=='four_plus_distractor' else None))
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase1',required=True);p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--limit',type=int)
    a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    rawroot=out.parent/'raw_results'
    assert json.loads((rawroot/'verification.json').read_text())['passed']
    save(out/'protocol.json',dict(schema='simple-pose-comparison/1',radii=RADII,budget=8,physical_overlap=.10,
        selection='SIM severe raw20/40 contamination0; controls pass; complete_raw exceeds same-metric baseline, then highest complete_raw, smaller radius ties',
        additional_previous_request_gate='single_complete_raw>=.90 reported separately; definition differs from old hypothesis metric',
        scale_form='fixedpx: no monotone proportional spacing dependence in raw report; no sigma derivation',
        selection_before_real=True,raw_summary_sha256=hashlib.sha256((rawroot/'summary.json').read_bytes()).hexdigest(),
        numeric_backend='CPU FP32 geometry; FP64 mean shift',no_training_changes=True))
    protocol=json.loads((Path(a.phase1)/'protocol.json').read_text());geometry=CompatibilityConfig(**protocol['geometry'])
    control={f's{r}':controls(geometry,r) for r in RADII};save(out/'controls.json',control)
    rows={n:{} for n in ['baseline']+[f's{r}' for r in RADII]};summary={n:{} for n in rows};start=time.time();count=0
    try:
        with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(a.phase1,a.out)) as pool:
            for split,n in [('sim_select',1500),('dunhuang_cv',803),('turufan',602)]:
                for name in rows:rows[name][split]=[]
                for answer in pool.map(worker,[(split,i) for i in range(min(n,a.limit or n))]):
                    for name,r in answer.items():rows[name][split].append(r)
                    count+=1
                    if count%100==0:save(out/'status.json',dict(status='measuring',pairs=count,split=split,seconds=time.time()-start))
                for name in rows:summary[name][split]=summarize(rows[name][split])
                save(out/'summary.json',summary)
                if split=='sim_select':
                    eligible=[];gates={}
                    for name in control:
                        r=summary[name][split]
                        g=dict(no_severe_contamination=r['raw_edge_mixed20_40']==0,controls=all(x['passed'] for x in control[name]),
                            completeness_improved=r['complete_raw_fraction']>summary['baseline'][split]['complete_raw_fraction'],
                            extra_single_complete90=r['single_complete_raw_fraction']>=.9,
                            full_four=all(x['complete_four'] for x in control[name] if x['complete_four'] is not None))
                        gates[name]=g
                        if g['no_severe_contamination'] and g['controls'] and g['completeness_improved']:eligible.append(name)
                    key=lambda name:(-summary[name][split]['complete_raw_fraction'],int(name[1:]))
                    chosen=sorted(eligible,key=key)[0] if eligible else None
                    save(out/'selection.json',dict(selected=chosen,eligible=eligible,gates=gates,
                        diagnostic_only=chosen is None,diagnostic_variant=chosen or sorted(control,key=key)[0],
                        selected_on='SIM SELECT only',real_evaluated_at_selection=False,recorded_unix=time.time()))
        save(out/'pair_summary.json',rows)
        save(out/'complete.json',dict(status='complete',pairs=count,variant_records=count*5,seconds=time.time()-start,limited=a.limit))
    except Exception as e:
        save(out/'failure.json',dict(error=repr(e),pairs=count));raise

if __name__=='__main__':main()

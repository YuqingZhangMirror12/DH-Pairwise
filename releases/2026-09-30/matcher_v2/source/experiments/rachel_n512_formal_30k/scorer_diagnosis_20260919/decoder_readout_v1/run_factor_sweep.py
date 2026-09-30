"""Frozen CPU mechanism controls on full SELECT and REAL-CAL/SELECT only.

Same checkpoint/Q within each pair. No optimizer, GT seeds, threshold refit,
TEST opening, or edits to any training source. Pilot output is never a result
for the full population. All interventions are preregistered mechanisms.
"""
import argparse
from dataclasses import asdict
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

from decoder import CONTROLS,copy_with_search


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sigmoid(x):return 1/(1+math.exp(-max(-700,min(700,float(x)))))


def readouts(model,pair,pred,is_binary):
    """All variants keep original poses, pair identities and frozen weights."""
    import torch
    import torch.nn.functional as F
    rows=[];variants={'baseline':[]}
    if is_binary:
        names=['zero_overlap_feature']+([] if model.head.variant=='stats' else ['zero_mean','zero_max','zero_mean_max'])
    else:names=['no_overlap_penalty','no_conflict_penalty','no_conflict_or_overlap']
    variants.update({k:[] for k in names})
    for c in pred.clusters:
        r=c.readout;ids=torch.unique(c.proposal.edge_ids.long().to(pair.q.device),dim=0)
        q=pair.q[ids[:,0],ids[:,1]]
        rows.append(dict(translation=c.translation.detach().cpu().tolist(),logit=float(r.logit),score=float(r.score),
            pairs=len(ids),sum_q=float(q.sum()),max_q=float(q.max()),members=list(c.proposal.merged_hypothesis_ids)))
        variants['baseline'].append(float(r.logit))
        if is_binary:
            value=(r.inputs.statistics if r.pooled_features is None else torch.cat((r.pooled_features,r.inputs.statistics)))
            for name in names:
                x=value.clone()
                if name=='zero_overlap_feature':x[-3]=0.
                elif name=='zero_mean':x[:32]=0.
                elif name=='zero_max':x[32:64]=0.
                else:x[:64]=0.
                variants[name].append(float(model.head.cluster_mlp(x).squeeze()))
        else:
            head=model.head;scale=float(head.length_scale_px)
            positive=float(F.softplus(head.positive_scale))*math.log1p(float(r.positive_evidence_px)/scale)
            conflict=float(F.softplus(head.conflict_scale))*math.log1p(float(r.conflict_evidence_px)/scale)
            overlap=float(head.bias)+positive-conflict-float(r.logit)
            variants['no_overlap_penalty'].append(float(r.logit)+overlap)
            variants['no_conflict_penalty'].append(float(r.logit)+conflict)
            variants['no_conflict_or_overlap'].append(float(r.logit)+conflict+overlap)
    result={}
    for key,logits in variants.items():
        winner=max(range(len(logits)),key=logits.__getitem__) if logits else -1
        result[key]=dict(winner=winner,score=sigmoid(logits[winner]) if winner>=0 else 0.,logits=logits)
    for key,field in [('raw_sum_q_rank','sum_q'),('raw_max_q_rank','max_q')]:
        winner=max(range(len(rows)),key=lambda i:rows[i][field]) if rows else -1
        result[key]=dict(winner=winner,score=rows[winner]['score'] if winner>=0 else 0.,
                        classification_score='original trained score of Q-ranked winner, not calibrated Q')
    return rows,result


def outcome(base,variant,candidates,target,threshold,valid):
    import numpy as np
    win=variant['winner'];pose=candidates[win]['translation'] if win>=0 else None
    known=target is not None
    errors=[float(np.linalg.norm(np.asarray(c['translation'])-target)) for c in candidates] if known else []
    error=errors[win] if known and win>=0 else None
    return dict(base,has_candidate=win>=0,numeric_valid=valid,score=variant['score'],translation=pose,
        selected_cluster_id=win,error_px=error,gt_known=known,
        layout20=bool(known and valid and error is not None and error<=20),
        candidate_coverage=bool(known and any(e<=20 for e in errors)),
        accepted=bool(win>=0 and valid and variant['score']>=threshold))


def run(args):
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(26092406)
    sys.path.insert(0,str(Path(args.audit_root).resolve()))
    import audit
    spec=read(Path(args.audit_root)/'plan.json')['models'][args.model]
    model,contract,provenance,common=audit.load(spec,args.model)
    original_state=common.state_digest(model);model.eval().requires_grad_(False)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    plan_path=spec['config'].get('real-plan','/root/autodl-tmp/binary_micro32_20260928/real_split.json')
    role_plan=read(plan_path)
    threshold=float(provenance.get('threshold',provenance.get('thresholds',{}).get('dunhuang_cv',-1)))
    if not .2<=threshold<=.8:raise ValueError('original frozen CAL threshold required')
    geometry_source=Path(importlib.import_module(model.builder.__module__).__file__)
    metadata=dict(model=args.model,checkpoint_sha256=provenance['checkpoint_sha256'],original_model_sha256=original_state,
        source_plan_sha256=sha(Path(args.audit_root)/'plan.json'),real_roles_sha256=sha(plan_path),
        script_sha256=sha(__file__),decoder_sha256=sha(Path(__file__).with_name('decoder.py')),
        geometry_source_sha256=sha(geometry_source),search={k:asdict(v) for k,v in CONTROLS.items()},
        threshold=threshold,threshold_refitted=False,training_performed=False,gpu_used=False,test_used=False,
        real_development_only=True,pilot_limit=args.pilot_limit,
        caveat='zeroed pooling/features are frozen-weight diagnostics, not retrained architectures; raw-Q modes change ranking only')
    save(out/'protocol.json',metadata);start=time.time();n=0;population={};metric_rows={}
    builder=model.builder
    dmod=importlib.import_module('experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.data')
    archived={}
    for split in ('dunhuang_cv','turufan'):
        path=Path(spec['config']['out'].replace('turufan',split))/'case_diagnostics.jsonl'
        archived[split]={r['pair_id']:r for r in map(json.loads,path.read_text().splitlines())}
    def populations():
        # REAL fold0 is excluded BEFORE decoding. It cannot influence choices.
        for split in ('dunhuang_cv','turufan'):
            meta,batches,source,dataset=common.load_population(split,contract,1)
            roles={pid:role for role in ('real_cal','real_select') for pid in role_plan['datasets'][split]['roles'][role]['pair_ids']}
            gt=({r['pair_id']:r for r in read(role_plan['gt_path'])['positive_pairs']} if split=='dunhuang_cv' else {})
            for items,batch in batches:
                item=items[0];pid=item['pair_id']
                if pid not in roles:continue
                yield split,roles[pid],item,batch,(lambda item=item,gt=gt: np.asarray(gt[item['pair_id']]['translation_gt_a_to_b_rc']) if item['label'] and item['pair_id'] in gt else None)
        cfg=contract['validation']['select_mixed'];ds=dmod.Dataset(cfg['path'],cfg['sha256'])
        if len(ds)!=1500:raise ValueError('full registered SELECT required')
        for j in range(len(ds)):
            sample=ds[j];item=ds.entries[j];batch=dmod.collate([sample])
            yield 'sim_select','sim_select',item,batch,(lambda sample=sample: np.asarray(sample[0].translation_a_to_b_rc) if sample[0].label and sample[0].translation_valid else None)
    try:
        with (out/'records.jsonl').open('x') as stream,torch.no_grad():
            for split,role,item,batch,target_after in populations():
                if args.pilot_limit and n>=args.pilot_limit:break
                inputs=common.tensor_inputs(batch,'cpu');evidence=model.matcher(*(inputs[k] for k in common.INPUTS))
                pair=common.PairEvidence.from_matcher(evidence,0,inputs['mask_a'],inputs['mask_b'])
                predictions={}
                for name,policy in CONTROLS.items():
                    controlled=copy_with_search(builder,policy);begin=time.time();proposals=controlled(pair)
                    pred=model.score_pair(pair,threshold=threshold,proposals=proposals)
                    candidates,variants=readouts(model,pair,pred,args.model!='threshold_scratch_fixed')
                    predictions[name]=dict(candidates=candidates,readouts=variants,search_audit=controlled.search_audit,
                        seed_count=len(proposals.seeds),hypotheses_count=len(proposals.hypotheses),seconds=time.time()-begin,
                        numeric_valid=bool(pred.numeric_valid))
                    if n==0 and name=='baseline':
                        old=model.score_pair(pair,threshold=threshold)
                        if not torch.equal(pred.score,old.score) or pred.selected_cluster_id!=old.selected_cluster_id:
                            raise AssertionError('copied baseline search did not replay original bit-exactly')
                # Only now join labels and GT; none has reached search/scoring.
                target=target_after();base=dict(pair_id=item['pair_id'],label=bool(item['label']))
                for search,record in predictions.items():
                    for intervention,value in record['readouts'].items():
                        row=outcome(base,value,record['candidates'],target,threshold,record['numeric_valid'])
                        metric_rows.setdefault((split,role,search,intervention),[]).append(row)
                comparison=None
                if split in archived:
                    old=archived[split][item['pair_id']];actual=predictions['baseline'];c=actual['candidates'];v=actual['readouts']['baseline']
                    aligned=len(c)==len(old['candidates'])
                    max_pose=max((float(np.linalg.norm(np.asarray(x['translation'])-y['refined_translation'])) for x,y in zip(c,old['candidates'])),default=0.)
                    max_score=max((abs(x['score']-y['score']) for x,y in zip(c,old['candidates'])),default=0.)
                    comparison=dict(strict=aligned and max_pose<=.05 and max_score<=5e-4 and v['winner']==old['selected_cluster_id'],
                        candidate_count_matches=aligned,max_pose_delta_px=max_pose,max_score_delta=max_score,
                        winner_matches=v['winner']==old['selected_cluster_id'],note='all interventions still compare against the same CPU baseline')
                row=dict(pair_id=item['pair_id'],split=split,role=role,label=base['label'],reference_equivalence=comparison,
                    target=None if target is None else target.tolist(),predictions=predictions)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();n+=1
                population[split]=population.get(split,0)+1
                save(out/'status.json',dict(status='running',model=args.model,pairs=n,population=population,
                    elapsed_seconds=time.time()-start,last_pair=item['pair_id']))
            stream.flush();os.fsync(stream.fileno())
        if common.state_digest(model)!=original_state:raise ValueError('frozen model changed')
        if not args.pilot_limit and population!=dict(dunhuang_cv=639,turufan=480,sim_select=1500):
            raise ValueError('incomplete development population:'+str(population))
        summary=[]
        for (split,role,search,intervention),rows in metric_rows.items():
            scores=common.population_summary(rows,threshold)
            if split=='turufan':
                for k in list(scores):
                    if k.startswith('joint_') or k in ('layout20','layout20_count','candidate_coverage','candidate_coverage_count','covered_but_winner_wrong','winner_correct_but_rejected','positive_no_correct_candidate','wrong_pose_accepted'):scores[k]=None
            baseline=metric_rows[(split,role,'baseline','baseline')]
            byid={r['pair_id']:r for r in baseline};transitions=dict(classification_gained=0,classification_lost=0,layout_gained=0,layout_lost=0)
            for r in rows:
                b=byid[r['pair_id']];was=b['accepted']==b['label'];now=r['accepted']==r['label']
                transitions['classification_gained']+=int(now and not was);transitions['classification_lost']+=int(was and not now)
                if r['gt_known']:
                    transitions['layout_gained']+=int(r['layout20'] and not b['layout20']);transitions['layout_lost']+=int(b['layout20'] and not r['layout20'])
            summary.append(dict(split=split,role=role,search=search,readout=intervention,metrics=scores,paired_transitions=transitions))
        save(out/'summary.json',dict(status='pilot_complete' if args.pilot_limit else 'complete',protocol=metadata,population=population,rows=summary,
            predictions_sha256=sha(out/'records.jsonl'),model_unchanged=True,elapsed_seconds=time.time()-start))
        save(out/'complete.json',dict(status='pilot_complete' if args.pilot_limit else 'complete',pairs=n,population=population,
            model_unchanged=True,summary_sha256=sha(out/'summary.json'),records_sha256=sha(out/'records.jsonl')))
        print(json.dumps(dict(status='pilot_complete' if args.pilot_limit else 'complete',pairs=n,population=population)))
    except BaseException as e:
        save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),pairs=n));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit-root',required=True);p.add_argument('--model',required=True)
    p.add_argument('--out',required=True);p.add_argument('--pilot-limit',type=int)
    a=p.parse_args();run(a)

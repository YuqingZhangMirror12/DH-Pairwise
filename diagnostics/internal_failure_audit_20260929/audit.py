"""CPU-only, fixed-checkpoint audit of a pre-enumerated complete failure set.

Never writes training roots, uses an optimizer, changes a threshold, or injects
GT into ordinary inference. GT-seeded probes run AFTER the saved prediction
and are explicitly labeled oracle diagnostics, never reported as performance.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import types

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,allow_nan=False,indent=2));temp.replace(p)
def module_file(path):
    spec=importlib.util.spec_from_file_location('diagnostic_bound_entry',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def alias(name,path):
    m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m


def load(spec, key):
    a=spec['config'];root=Path(a['root']);common_path=a['common']
    if key=='aggressive_binary_patch':
        entry=module_file(a['entry']);training,helper=entry.bootstrap(root/'source',common_path,a['binary-source'])
        loader=importlib.import_module('consensus_aggressive_eval_adapter.loading')
        model,contract,provenance=loader.load_selected(root,a['reference'],'sim',a['real-plan'],helper)
    elif key.startswith('binary_'):
        entry=module_file(a['entry']);training,helper=entry.bootstrap(root/'source',common_path)
        loader=importlib.import_module('consensus_binary_eval_adapter.loading')
        model,contract,provenance=loader.load_selected(root,a['variant'],a['reference'],'sim',a['real-plan'],helper)
    else:
        sys.path.insert(0,str(root/'source'));alias('consensus_binary_eval_common',common_path)
        loader=importlib.import_module('consensus_binary_eval_common.frozen')
        model,contract,provenance=loader.load_selected(root,'scratch_fixed',a['reference'],completed_arm_only=True)
    common=importlib.import_module('consensus_binary_eval_common.evaluate')
    return model,contract,provenance,common


def arr(t):return t.detach().cpu().numpy()
def plain(t):return arr(t).tolist()
def error(t,gt):return None if gt is None else float((t.detach().cpu().double()-gt).norm())
def sigmoid(x):return 1/(1+math.exp(-max(-700,min(700,x))))


def cluster_record(p,pair,gt):
    ids=p.edge_ids.long().to(pair.q.device);q=pair.q[ids[:,0],ids[:,1]]
    return dict(pose=plain(p.translation),error_px=error(p.translation,gt),pairs=len(ids),
        sum_q=float(q.sum()),mean_q=float(q.mean()) if len(q) else 0.,
        mass_px=float(p.absolute_support_mass_px),members=list(p.merged_hypothesis_ids),
        original_diameter_px=getattr(p,'actual_diameter_px',None),overlap=p.overlap,
        endpoint_count_a=int(ids[:,0].unique().numel()),endpoint_count_b=int(ids[:,1].unique().numel()))


def binary_interventions(head,pred,gt,arrays):
    import torch
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.binary_scorer_v1.head import SCALAR_NAMES
    results=[];vectors=[]
    for i,c in enumerate(pred.clusters):
        r=c.readout;x=r.inputs
        inp=x.statistics if r.pooled_features is None else torch.cat((r.pooled_features,x.statistics))
        vectors.append(inp.detach())
        arrays[f'c{i}/statistics']=arr(x.statistics);arrays[f'c{i}/edge_geometry']=arr(x.edge_geometry)
        arrays[f'c{i}/pool_weights']=arr(x.normalized_weights)
        arrays[f'c{i}/cluster_input']=arr(inp)
        if x.patch_context is not None:arrays[f'c{i}/patch_context']=arr(x.patch_context)
        h=inp
        layers=[]
        for k,layer in enumerate(head.cluster_mlp):
            h=layer(h);arrays[f'c{i}/cluster_layer_{k}']=arr(h)
            layers.append(dict(layer=k,min=float(h.min()),max=float(h.max()),norm=float(h.norm())))
        assert abs(float(h.squeeze())-float(r.logit))<1e-5
        # Exact final-layer contribution, not a claim about feature causality.
        activation=torch.from_numpy(arrays[f'c{i}/cluster_layer_3'])
        contribution=activation*head.cluster_mlp[-1].weight[0]
        arrays[f'c{i}/last_layer_contributions']=arr(contribution)
        results.append(dict(cluster=i,statistics=dict(zip(SCALAR_NAMES,plain(x.statistics))),
            logit=float(r.logit),score=float(r.score),layer_summaries=layers,
            final_bias=float(head.cluster_mlp[-1].bias[0]),contribution_sum=float(contribution.sum())))
    counter=None
    if gt is not None and len(pred.clusters)>1:
        good=[i for i,c in enumerate(pred.clusters) if error(c.translation,gt)<=20]
        bad=[i for i,c in enumerate(pred.clusters) if error(c.translation,gt)>20]
        if good and bad:
            gi=max(good,key=lambda i:float(pred.clusters[i].readout.logit))
            bi=max(bad,key=lambda i:float(pred.clusters[i].readout.logit))
            g,b=vectors[gi],vectors[bi]
            groups=({'mean_pool':list(range(32)),'max_pool':list(range(32,64)),'statistics':list(range(64,80))}
                    if head.variant=='patch' else {'quantity_coverage':list(range(8)),
                        'residual_overlap':[8,9,10,13,14],'dustbin_outside_members':[11,12,15]})
            names=list(groups);values={};hybrids=[]
            for flags in itertools.product((0,1),repeat=3):
                v=g.clone()
                for flag,name in zip(flags,names):
                    if flag:v[groups[name]]=b[groups[name]]
                value=float(head.cluster_mlp(v).squeeze());values[flags]=value
                hybrids.append(dict(wrong_candidate_blocks=[n for f,n in zip(flags,names) if f],logit=value,score=sigmoid(value)))
            shapley={}
            for k,name in enumerate(names):
                value=0.
                for flags in itertools.product((0,1),repeat=3):
                    if flags[k]:continue
                    count=sum(flags);changed=list(flags);changed[k]=1
                    weight=math.factorial(count)*math.factorial(2-count)/6
                    value+=weight*(values[tuple(changed)]-values[flags])
                shapley[name]=value
            delta=values[(1,1,1)]-values[(0,0,0)]
            assert abs(sum(shapley.values())-delta)<1e-5
            counter=dict(correct=gi,wrong=bi,wrong_minus_correct_logit=delta,
                block_shapley_logit=shapley,hybrids=hybrids,
                caveat='Exact frozen-MLP input-block substitution, possibly off-manifold; not retraining or proof of upstream causality.')
    return dict(clusters=results,ranking_block_intervention=counter)


def complex_interventions(model,pred,pair,gt,arrays):
    import torch
    import torch.nn.functional as F
    head=model.head;rows=[]
    a,b,c=map(float,(F.softplus(head.positive_scale),F.softplus(head.conflict_scale),F.softplus(head.overlap_scale)))
    length=float(head.length_scale_px);bias=float(head.bias)
    for i,item in enumerate(pred.clusters):
        r=item.readout;p=float(r.positive_evidence_px);n=float(r.conflict_evidence_px)
        # Recover the actual overlap penalty by subtracting known exact terms.
        positive=a*math.log1p(p/length);conflict=b*math.log1p(n/length)
        overlap=bias+positive-conflict-float(r.logit)
        data=dict(cluster=i,logit=float(r.logit),score=float(r.score),bias=bias,
            positive_term=positive,conflict_penalty=conflict,overlap_penalty=overlap,
            no_conflict_logit=float(r.logit)+conflict,
            all_support_logit=bias+a*math.log1p(float(r.observed_mass_length_px)/length)-overlap)
        for stage,encoded in [('initial',item.initial_encoded),('final',item.encoded)]:
            evidence=encoded.evidence;statistics={}
            for side in 'ab':
                z=getattr(encoded,side);ev=getattr(evidence,side);w=ev.mass*ev.observed_arc_px
                arrays[f'c{i}/{stage}/{side}/probabilities']=arr(z.local_probabilities)
                arrays[f'c{i}/{stage}/{side}/state']=arr(z.state)
                arrays[f'c{i}/{stage}/{side}/mass']=arr(w)
                statistics[side]=dict(mass=float(w.sum()),support=float((w*z.local_probabilities[:,0]).sum()),
                    unknown=float((w*z.local_probabilities[:,1]).sum()),conflict=float((w*z.local_probabilities[:,2]).sum()),
                    nonzero_endpoints=int((w>0).sum()))
            data[stage]=statistics
        rows.append(data)
    return dict(clusters=rows,readout_parameters=dict(positive=a,conflict=b,overlap=c,length=length,bias=bias))


def diagnose(model,pair,target,out,common,is_binary):
    import numpy as np
    import torch
    builder=model.builder;recorded={};original=builder._combine
    def traced(*args,**kwargs):
        result=original(*args,**kwargs);recorded[tuple(result.merged_hypothesis_ids)]=result;return result
    builder._combine=traced
    try:
        with torch.no_grad():pred=model.score_pair(pair,threshold=target['threshold'],capture_diagnostics=True)
    finally:builder._combine=original
    # GT is first accessed after the complete normal forward above.
    gt=None if target['gt'] is None else torch.tensor(target['gt'],dtype=torch.float64)
    props=pred.proposals;expected=target['expected'];actual=[];errors=[]
    for i,c in enumerate(pred.clusters):
        x=cluster_record(c.proposal,pair,gt)
        x.update(index=i,final_pose=plain(c.translation),final_error_px=error(c.translation,gt),
            score=float(c.readout.score),logit=float(c.readout.logit))
        actual.append(x)
    if len(actual)!=len(expected['candidates']):errors.append('candidate_count_changed')
    max_score=max_pose=0.
    for c,old in zip(actual,expected['candidates']):
        max_score=max(max_score,abs(c['score']-old['score']))
        max_pose=max(max_pose,float(np.linalg.norm(np.array(c['final_pose'])-old['refined_translation'])))
    if max_score>5e-4:errors.append('cpu_gpu_score_delta_over_5e-4')
    if max_pose>.05:errors.append('cpu_gpu_pose_delta_over_0.05px')
    if pred.selected_cluster_id!=expected['selected_cluster_id']:errors.append('winner_changed')
    if pred.accepted!=bool(expected['score']>=target['threshold'] and expected['numeric_valid'] and expected['has_candidate']):errors.append('acceptance_changed')
    before=list((i,) for i,h in enumerate(props.hypotheses) if len(h.edge_ids))
    for merge in props.merge_trace:
        members=tuple(merge['hypothesis_ids']);s=set(members)
        before=[part for part in before if not set(part).issubset(s)]+[members]
    prephysical=[recorded[k] for k in before]
    hypotheses=[cluster_record(p,pair,gt) for p in props.hypotheses]
    physical=[cluster_record(p,pair,gt) for p in prephysical]
    prebudget=[cluster_record(p,pair,gt) for p in builder.all_clusters]
    arrays=dict(q=arr(pair.q),points_a=arr(pair.points_a),points_b=arr(pair.points_b),
        mask_a=arr(pair.mask_a),mask_b=arr(pair.mask_b),unmatched_a=arr(pair.unmatched_a),unmatched_b=arr(pair.unmatched_b),
        cloud_ids=arr(props.cloud.ids),cloud_displacement=arr(props.cloud.displacement),seeds=arr(props.seeds))
    for i,h in enumerate(props.hypotheses):arrays[f'h{i}/ids']=arr(h.edge_ids)
    for i,c in enumerate(pred.clusters):arrays[f'c{i}/ids']=arr(c.proposal.edge_ids)
    with torch.no_grad():
        head=binary_interventions(model.head,pred,gt,arrays) if is_binary else complex_interventions(model,pred,pair,gt,arrays)
    oracle=None;support=None
    if gt is not None:
        delta=pair.points_b[None,:,:]-pair.points_a[:,None,:]-gt.to(pair.q)
        near=delta.norm(dim=-1)<=20;nonzero=pair.q>=builder.config.minimum_absolute_q
        ci=props.cloud.ids;cs=(props.cloud.displacement-gt.to(props.cloud.displacement)).norm(dim=-1)<=20
        support=dict(full_Q_near_GT20_pairs=int((near&nonzero).sum()),full_Q_near_GT20_sum=float(pair.q[near].sum()),
            cloud_near_GT20_pairs=int(cs.sum()),cloud_near_GT20_sum=float(props.cloud.q[cs].sum()),
            total_Q=float(pair.q.sum()),cloud_total_pairs=len(ci))
        if target['kind']=='no_correct_candidate':
            from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import material_overlap,observed_arc_cells
            overlap_fn=lambda t:material_overlap(pair,t)
            with torch.no_grad():
                initial=gt.to(props.cloud.displacement)
                hypothesis=builder._hypothesis(props.cloud,initial,(0,),initial[None],(0,),overlap_fn)
                if len(hypothesis.edge_ids):
                    cells=tuple(observed_arc_cells(g,builder.config.observation_radius_px)[0].detach().cpu() for g in (pair.ga,pair.gb))
                    fitted=builder._combine(props.cloud,(hypothesis,),(0,),cells,overlap_fn)
                    oracle=dict(hypothesis=cluster_record(hypothesis,pair,gt),joint=cluster_record(fitted,pair,gt),
                        GT_injected=True,diagnostic_only=True,actual_candidate_set_unchanged=True)
                else:oracle=dict(empty=True,GT_injected=True,diagnostic_only=True)
    correct=lambda rows:sum(x['error_px'] is not None and x['error_px']<=20 for x in rows)
    seed_errors=[] if gt is None else plain((props.seeds.double()-gt).norm(dim=-1))
    stage=dict(seeds_near20=sum(x<=20 for x in seed_errors),seed_errors=seed_errors,
        native_hypotheses_correct=correct(hypotheses),post_joint_before_physical_correct=correct(physical),
        prebudget_correct=correct(prebudget),retained_correct=sum(x['error_px'] is not None and x['error_px']<=20 for x in actual),
        post_neural_refine_correct=sum(x['final_error_px'] is not None and x['final_error_px']<=20 for x in actual)) if gt is not None else None
    result=dict(pair_id=target['pair_id'],name=target['name'],split=target['split'],kind=target['kind'],
        label=target['label'],gt=target['gt'],threshold=target['threshold'],selected=pred.selected_cluster_id,
        accepted=pred.accepted,score=float(pred.score),stages=stage,raw_support=support,
        hypotheses=hypotheses,post_joint_before_physical=physical,prebudget=prebudget,
        retained=actual,merge_trace=props.merge_trace,builder_audit=builder.audit,head=head,oracle=oracle,
        equivalence=dict(passed=not errors,errors=errors,max_score_delta=max_score,max_pose_delta_px=max_pose),
        semantics='GT-free actual path first; CPU-vs-original-GPU comparison; block interventions/oracle probes are diagnostics, not new model results.')
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'arrays.npz',**arrays)
    result['arrays_sha256']=sha(out/'arrays.npz');save(out/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--model',required=True)
    p.add_argument('--out',required=True);p.add_argument('--limit',type=int);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);plan=read(args.plan);spec=plan['models'][args.model]
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(26092406)
    model,contract,provenance,common=load(spec,args.model)
    initial=common.state_digest(model);matcher_digest=common.state_digest(model.matcher)
    identity=dict(model=args.model,checkpoint_sha256=provenance['checkpoint_sha256'],model_digest=initial,
        matcher_tensor_digest=matcher_digest,plan_sha256=sha(args.plan),script_sha256=sha(__file__),provenance=provenance)
    if (out/'identity.json').exists():assert read(out/'identity.json')==identity,'do not reuse changed audit implementation'
    else:save(out/'identity.json',identity)
    torch.save(model.head.state_dict(),out/'head_state.pt')
    cases=spec['cases'];cases=sorted(cases,key=lambda r:(not any('/'+x+' ' in r['name'] for x in ['414','85']),r['split'],r['pair_id']))
    wanted={x['pair_id']:x for x in cases[:args.limit] if args.limit} if args.limit else {x['pair_id']:x for x in cases}
    completed=[];start=time.time()
    for split in ['dunhuang_cv','turufan']:
        selected={k:v for k,v in wanted.items() if v['split']==split}
        if not selected:continue
        meta,batches,source,dataset=common.load_population(split,contract,1)
        for items,batch in batches:
            item=items[0];pid=item['pair_id']
            if pid not in selected:continue
            target=selected[pid];assert target['checkpoint_sha256']==provenance['checkpoint_sha256']
            case=out/'cases'/hashlib.sha256(pid.encode()).hexdigest()[:24]
            if (case/'result.json').exists():
                result=read(case/'result.json');assert sha(case/'arrays.npz')==result['arrays_sha256']
            else:
                inputs=common.tensor_inputs(batch,'cpu')
                with torch.no_grad():
                    evidence=model.matcher(*(inputs[k] for k in common.INPUTS))
                    pair=common.PairEvidence.from_matcher(evidence,0,inputs['mask_a'],inputs['mask_b'])
                result=diagnose(model,pair,target,case,common,args.model!='threshold_scratch_fixed')
            completed.append(dict(pair_id=pid,case=str(case),equivalent=result['equivalence']['passed']))
            save(out/'status.json',dict(status='running',completed=len(completed),requested=len(wanted),seconds=time.time()-start,last=target['name']))
            print(json.dumps(dict(model=args.model,completed=len(completed),requested=len(wanted),name=target['name'],equivalence=result['equivalence'])),flush=True)
    assert len(completed)==len(wanted)
    assert common.state_digest(model)==initial
    save(out/('pilot_complete.json' if args.limit else 'complete.json'),dict(status='complete',cases=len(completed),
        original_model_unchanged=True,all_reproduced=all(x['equivalent'] for x in completed),rows=completed,seconds=time.time()-start,identity=identity))
    save(out/'status.json',dict(status='pilot_complete' if args.limit else 'complete',completed=len(completed),requested=len(wanted),seconds=time.time()-start))


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();sys.exit(1)

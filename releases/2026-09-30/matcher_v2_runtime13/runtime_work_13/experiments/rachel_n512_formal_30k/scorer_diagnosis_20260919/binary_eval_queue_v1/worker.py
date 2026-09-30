"""Isolated CPU selection/audit workers and disposable device tracing gate."""
import argparse
from dataclasses import fields,is_dataclass,replace
import importlib
from pathlib import Path
import sys
import tempfile
from queue_contracts import evaluation_args,read,save,sha,hashes,verify_result

def bind(args):
    a=evaluation_args(args.root,args.prepared,args.variant)
    sys.path.insert(0,str(Path(args.prepared)/'binary_eval_v1'))
    entry=importlib.import_module('entry');receipt=entry.validate_preparation(a)
    training,helper=entry.bootstrap(Path(args.root)/'source',a.common_source)
    return a,receipt,training,helper

def move(value,device):
    import torch
    if isinstance(value,torch.Tensor):return value.to(device)
    if is_dataclass(value):return replace(value,**{f.name:move(getattr(value,f.name),device) for f in fields(value)})
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    return value

def trace_check(training,variant,device):
    """Actual production-size head on synthetic correspondences, not real data."""
    import torch
    common=importlib.import_module('consensus_binary_eval_common.evaluate')
    diagnostic=importlib.import_module('consensus_binary_eval_adapter.snapshot')
    tracer=importlib.import_module('consensus_binary_eval_adapter.trace')
    fixture=importlib.import_module(training.__package__.rsplit('.',1)[0]+'.binary_scorer_v1.test_binary').fixture
    torch.manual_seed(training.TrainingConfig(scorer_variant=variant).head_seed)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    original,pair,proposals,_=fixture(variant)
    pair=replace(pair,**{name:getattr(pair,name).repeat(1,24)
                        for name in ('local_a','local_b','context_a','context_b')})
    model=training.S7Consensus(None,original.geometry,head=training.fresh_head(training.TrainingConfig().head_seed,variant))
    model.to(device).eval().requires_grad_(False);pair=move(pair,device);proposals=move(proposals,device)
    state=common.state_digest(model);cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if str(device).startswith('cuda') else None
    with torch.no_grad():
        baseline=model.score_pair(pair,proposals=proposals,threshold=.3)
        with tracer.MLPTrace(model.head) as trace:actual=model.score_pair(pair,proposals=proposals,threshold=.3)
        same=(torch.equal(baseline.score,actual.score) and torch.equal(baseline.translation_a_to_b_rc,actual.translation_a_to_b_rc)
              and baseline.selected_cluster_id==actual.selected_cluster_id
              and all(torch.equal(a.readout.logit,b.readout.logit) for a,b in zip(baseline.clusters,actual.clusters)))
        meta,arrays=diagnostic.snapshot_prediction('synthetic-device-gate',pair,actual,threshold=.3,
            provenance=dict(variant='binary_'+variant,checkpoint='synthetic fresh head; not a trained checkpoint'),trace=trace)
    with tempfile.TemporaryDirectory() as t:
        directory=Path(t)/'case';diagnostic.write_snapshot(directory,meta,arrays)
        audit=diagnostic.audit_snapshot(directory/'evidence.json')
    unchanged=common.state_digest(model)==state
    rng_same=torch.equal(torch.get_rng_state(),cpu_rng) and (cuda_rng is None or torch.equal(torch.cuda.get_rng_state(),cuda_rng))
    passed=same and unchanged and rng_same and audit['status']=='passed' and not audit['errors']
    return dict(schema='binary-trace-device-gate/1',status='passed' if passed else 'failed',variant=variant,
        device=str(device),head_parameters=sum(p.numel() for p in model.head.parameters()),
        parameters_unchanged=unchanged,rng_unchanged=rng_same,capture_bitwise_equal=same,
        numeric_replay_passed=audit['status']=='passed' and not audit['errors'],numeric_audit=audit,
        real_inference_performed=False,trained_checkpoint_opened=False,optimizer_updates=0,
        scope='production-size binary head and evidence recorder on synthetic inputs; not Matcher GPU forward validation')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('selected','trace','verify-job'))
    for key in ('root','prepared','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--variant',required=True,choices=('patch','stats'));p.add_argument('--device',default='cpu',choices=('cpu','cuda:0'))
    p.add_argument('--selection',choices=('sim','real'));p.add_argument('--split',choices=('sim_test_v14','dunhuang_cv','turufan'))
    p.add_argument('--job');p.add_argument('--selected-gate');args=p.parse_args()
    if Path(args.out).exists():raise ValueError('preserve existing gate/audit output')
    a,receipt,training,helper=bind(args)
    bindings={k:receipt[k] for k in ('adapter_python_sha256','common_python_sha256','training_source_sha256','real_plan_sha256','fixed_case_plan_sha256')}
    if args.operation=='selected':
        if args.device!='cpu':raise ValueError('selected model inspection is CPU-only')
        loader=importlib.import_module('consensus_binary_eval_adapter.loading');selected={}
        for choice in ('sim','real'):
            model,_,selected[choice]=loader.load_selected(a.root,args.variant,a.reference,choice,a.real_plan,helper)
            del model
        result=dict(schema='binary-frozen-selection-gate/1',status='passed',variant=args.variant,
            selected=selected,real_inference_performed=False,trained_checkpoints_inspected=True)
    elif args.operation=='trace':
        result=trace_check(training,args.variant,args.device)
    else:
        if not args.selection or not args.split or not args.job or not args.selected_gate or args.device!='cpu':
            raise ValueError('CPU result verification requires job/selection/split/gate')
        selected=read(args.selected_gate)['selected'][args.selection]
        audit=importlib.import_module('consensus_binary_eval_adapter.snapshot').audit_snapshot
        result=verify_result(args.job,args.variant,args.selection,args.split,selected,read(a.real_plan),read(a.case_plan),audit)
    result.update(source_bindings=bindings,queue_python_sha256=hashes(Path(__file__).parent))
    save(args.out,result)
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()

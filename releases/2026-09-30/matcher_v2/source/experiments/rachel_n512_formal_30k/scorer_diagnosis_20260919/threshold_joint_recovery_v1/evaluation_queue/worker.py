"""Fresh-process loaders, actual tracing, and prediction reuse; never training."""
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import sys
from contracts import (POST,PREPARED,FORMAL,CONTROL,REL,args_for,read,save,sha,inventory,
    same_model,accepted,verify_result)

def bind(kind,out,operation='evaluate'):
    a=args_for(kind,out,operation=operation)
    sys.path.insert(0,str(POST/'threshold_joint_eval_v1'))
    entry=importlib.import_module('entry');receipt=entry.validate_preparation(a)
    training,helper=entry.bootstrap(Path(a.root)/'source',a.common_source,a.joint_source)
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return a,entry,training,helper,receipt

def selected(kind,a,helper):
    loader=importlib.import_module('consensus_joint_eval_adapter.loading')
    common=importlib.import_module('consensus_joint_eval_common.evaluate')
    choices=('sim','real') if kind=='joint' else ('real',)
    results={}
    for choice in choices:
        if kind=='joint':model,_,proof=loader.load_joint(a.root,a.reference,choice,a.real_plan,helper)
        else:model,_,proof=loader.load_frozen_real(a.root,a.reference,a.real_selection,a.real_plan,helper)
        proof['model_state_sha256']=common.state_digest(model);results[choice]=proof;del model
    return dict(status='passed',kind=kind,selected=results,real_inference_performed=False,
        trained_checkpoints_inspected=True,selection_on_test=False)

def reuse(source,out,source_proof,destination_proof,split,evaluation,audit):
    """Only metadata/threshold decisions change; all scores/layouts/arrays stay.

    This is recorded as reuse, not as a second network forward. Exact complete
    model tensor identity and source/data bindings are required, not epoch alone.
    """
    if not same_model(source_proof,destination_proof):raise ValueError('cannot reuse a different model')
    source=Path(source);out=Path(out)
    if out.exists():raise ValueError('preserve existing output')
    frozen=read(source/'prediction_complete.json')
    if ((source/'failure.json').exists() or read(source/'status.json').get('status')!='complete'
            or frozen.get('status')!='all_predictions_frozen' or not frozen.get('model_state_unchanged')
            or frozen.get('checkpoint_sha256')!=source_proof['checkpoint_sha256']
            or frozen.get('split')!=split or frozen.get('sha256')!=sha(source/'pair_predictions.jsonl')):
        raise ValueError('reuse source not frozen/hash-bound')
    roles=read(PREPARED/'real_split.json')
    source_files={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
    out.mkdir(parents=True)
    threshold=destination_proof['thresholds'][split]
    provenance=dict(read(source/'protocol.json'))
    provenance.pop('status',None);provenance.update(destination_proof,split=split,threshold=threshold,
        threshold_origin='source-isolated REAL-CAL at REAL-SELECT epoch' if split!='sim_test_v14' else 'synthetic CAL at selected epoch',
        prediction_reused=True,forward_evaluation_source=str(source),threshold_refitting=False)
    for name in ('pair_predictions.jsonl','case_diagnostics.jsonl'):
        rows=[json.loads(s) for s in (source/name).read_text().splitlines()]
        for row in rows:row['accepted']=accepted(row,threshold)
        with (out/name).open('x') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    labeled=rows
    cases=deepcopy(read(source/'diagnostic_index.json')['cases'])
    for case in cases:
        old=source/case['evidence'];new=out/case['evidence']
        shutil.copytree(old.parent,new.parent)
        meta=read(new);meta.update(threshold=threshold,accepted=accepted(meta,threshold),
            provenance=evaluation.snapshot_provenance(provenance))
        save(new,meta);result=audit(new)
        if result.get('status')!='passed':raise ValueError('reused numeric evidence differs')
        save(out/case['numerical_audit'],result)
    summary=evaluation.make_summary(labeled,split,roles,provenance);summary['diagnostic_cases']=cases
    save(out/'summary.json',summary)
    save(out/'diagnostic_index.json',dict(cases=cases,selected_by_new_results=False))
    save(out/'protocol.json',dict(status='complete',**provenance))
    save(out/'prediction_complete.json',dict(status='all_predictions_frozen',pairs=len(labeled),
        sha256=sha(out/'pair_predictions.jsonl'),model_state_unchanged=True,**provenance))
    save(out/'prediction_reuse.json',dict(status='complete',source=str(source),model_state_sha256=source_proof['model_state_sha256'],
        original_threshold=source_proof['thresholds'][split],new_threshold=threshold,source_files_sha256=source_files,
        network_forward_calls=0,original_outputs_unchanged=True))
    if any(sha(p)!=h for p,h in source_files.items()):raise ValueError('reuse source changed')
    save(out/'status.json',dict(status='complete',pairs=len(labeled),prediction_reused=True))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('selected','trace','reselect','verify','reuse'))
    p.add_argument('--kind',required=True,choices=('joint','frozen'));p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--choice',choices=('sim','real'))
    p.add_argument('--split');p.add_argument('--selected-gate');p.add_argument('--job');p.add_argument('--source')
    a=p.parse_args()
    if Path(a.out).exists():raise ValueError('preserve prior worker output')
    args,entry,training,helper,receipt=bind(a.kind,a.out,'reselect-control' if a.operation=='reselect' else 'evaluate')
    if a.operation=='reselect':
        if a.kind!='frozen':raise ValueError('only frozen control needs retrospective selection')
        args.device=a.device
        importlib.import_module('consensus_joint_eval_adapter.reselect').run(args,helper)
        return
    if a.operation=='selected':
        if a.device!='cpu':raise ValueError('selection gate is CPU-only')
        result=selected(a.kind,args,helper)
    elif a.operation=='trace':
        result=importlib.import_module('consensus_joint_eval_common.attention_gate').run(a.device)
        result['kind']=a.kind
    else:
        if a.device!='cpu' or not a.choice or not a.split or not a.selected_gate:raise ValueError('CPU verification/reuse inputs required')
        gate=read(a.selected_gate);proof=gate['selected'][a.choice]
        common=importlib.import_module('consensus_joint_eval_common.evaluate')
        if a.operation=='reuse':
            if a.kind!='joint' or a.choice!='real':raise ValueError('reuse is joint SIM to REAL only')
            # Source outputs must pass full original-choice verification first.
            verify_result(a.source,'joint','sim',a.split,gate['selected']['sim'],read(args.real_plan),read(args.case_plan),common.audit_snapshot)
            reuse(a.source,a.out,gate['selected']['sim'],proof,a.split,
                  importlib.import_module('consensus_joint_eval_adapter.evaluate'),common.audit_snapshot)
            return
        result=verify_result(a.job,a.kind,a.choice,a.split,proof,read(args.real_plan),read(args.case_plan),common.audit_snapshot)
    result.update(preparation_sha256=sha(args.preparation),queue_source_sha256=inventory(Path(__file__).parent))
    save(a.out,result)
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()

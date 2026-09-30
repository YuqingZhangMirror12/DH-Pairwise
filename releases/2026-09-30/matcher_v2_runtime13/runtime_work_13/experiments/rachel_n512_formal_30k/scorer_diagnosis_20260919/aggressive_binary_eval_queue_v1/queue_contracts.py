"""External six-job frozen evaluation contracts; no model or CUDA import."""
import hashlib
import json
import math
import os
from pathlib import Path

EXPECTED={'sim_test_aggressive':3000,'dunhuang_cv':803,'turufan':602}
TASKS=[(choice,split) for split in EXPECTED for choice in ('sim','real')]
FORMAL_ROOT=Path('/root/autodl-tmp/s7_aggressive_binary_v17_20260928')
PREPARED=Path('/root/autodl-tmp/aggressive_binary_20260927/evaluation_recovery_02')
REFERENCE='/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt'

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def hashes(directory):return {p.name:sha(p) for p in sorted(Path(directory).glob('*.py'))}
def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('x') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def identity(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime=int(fields[19]),cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
def evaluation_args(root,prepared,variant,reference=REFERENCE):
    from types import SimpleNamespace
    root=Path(root);prepared=Path(prepared)
    return SimpleNamespace(root=root,variant=variant,reference=reference,
        preparation=prepared/'evaluation_preparation_remote.json',common_source=prepared/'s7_consensus_eval_v14',binary_source=prepared/'binary_eval_v1',
        real_plan=prepared/'real_split.json',case_plan=prepared/'s7_consensus_eval_v14/case_plan.json')
def evaluator_command(python,root,prepared,out,variant,choice,split):
    if variant != 'patch' or (choice,split) not in TASKS:raise ValueError('unregistered evaluation task')
    a=evaluation_args(root,prepared,variant)
    command=[python,str(Path(prepared)/'aggressive_binary_eval_v1/entry.py')]
    for name in ('root','reference','preparation','common_source','real_plan','case_plan','binary_source'):
        command+=['--'+name.replace('_','-'),str(getattr(a,name))]
    return command+['--out',str(out),'--selection',choice,'--split',split,'--device','cuda:0']
def safe_child(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('evidence path escapes job directory')
    return path
def validate_selected_gate(gate,variant,root):
    if (gate.get('schema')!='aggressive-binary-frozen-selection-gate/1' or gate.get('status')!='passed'
            or gate.get('variant')!=variant or gate.get('real_inference_performed') is not False
            or set(gate.get('selected',{}))!={'sim','real'}):raise ValueError('terminal CPU model gate incomplete')
    for choice,p in gate['selected'].items():
        stage=Path(root)/('formal_scratch_aggressive')/'scorer'
        cp=stage/('best_joint.pt' if choice=='sim' else 'best_real.pt')
        if (Path(p['checkpoint']).resolve()!=cp.resolve() or p.get('checkpoint_sha256')!=sha(cp)
                or p.get('selection_kind')!=choice or p.get('variant')!='binary_'+variant
                or p.get('matcher_updated_during_scorer_training') is not False
                or p.get('selection_sha256')!=sha(stage/'selection.json')
                or p.get('terminal_receipt_sha256')!=sha(stage.parent/'training_complete.json')):
            raise ValueError('selected trained binary model changed')
    return gate
def validate_gpu_gate(gate,variant,bindings):
    if (gate.get('schema')!='binary-trace-device-gate/1' or gate.get('status')!='passed'
            or gate.get('variant')!=variant or gate.get('device')!='cuda:0'
            or gate.get('parameters_unchanged') is not True or gate.get('rng_unchanged') is not True
            or gate.get('capture_bitwise_equal') is not True or gate.get('numeric_replay_passed') is not True
            or gate.get('real_inference_performed') is not False or gate.get('source_bindings')!=bindings
            or gate.get('head_parameters')!=(34529 if variant=='patch' else 3201)
            or gate.get('optimizer_updates')!=0 or gate.get('trained_checkpoint_opened') is not False):
        raise ValueError('actual source-bound CUDA trace gate required')

def verify_result(directory,variant,choice,split,selected,role_plan,case_plan,audit):
    """Verify existing frozen outputs and rerun C10 numeric audits, no inference."""
    directory=Path(directory)
    if (directory/'failure.json').exists():raise ValueError('evaluation failure precedes stale summary')
    status=read(directory/'status.json');summary=read(directory/'summary.json')
    protocol=read(directory/'protocol.json');frozen=read(directory/'prediction_complete.json')
    cases=read(directory/'diagnostic_index.json')
    expected=dict(variant='binary_'+variant,selection_kind=choice,split=split,total_pairs=EXPECTED[split],
        selected_epoch=selected['selected_epoch'],checkpoint_sha256=selected['checkpoint_sha256'],
        threshold=selected['thresholds'][split])
    if (status.get('status')!='complete' or status.get('pairs')!=EXPECTED[split]
            or summary.get('status')!='complete' or protocol.get('status')!='complete'
            or frozen.get('status')!='all_predictions_frozen' or frozen.get('pairs')!=EXPECTED[split]
            or frozen.get('model_state_unchanged') is not True
            or frozen.get('sha256')!=sha(directory/'pair_predictions.jsonl')
            or any(any(r.get(k)!=v for k,v in expected.items()) for r in (summary,protocol,frozen))):
        raise ValueError('frozen result identity/count/hash differs')
    rows=[json.loads(line) for line in (directory/'pair_predictions.jsonl').read_text().splitlines()]
    ids=[r['pair_id'] for r in rows]
    if len(rows)!=EXPECTED[split] or len(set(ids))!=len(ids):raise ValueError('missing/duplicate prediction population')
    if any(not isinstance(r.get('score'),(int,float)) or not math.isfinite(r['score'])
           or any(k in r for k in ('label','target_translation_rc','layout20','gt_known')) for r in rows):
        raise ValueError('raw frozen predictions contain targets or invalid scores')
    if split=='sim_test_aggressive':
        if summary.get('main_group')!='all' or summary['groups']['all']['primary']['pairs']!=3000:
            raise ValueError('SIM TEST summary population differs')
    else:
        spec=role_plan['datasets'][split];expected_ids=set(spec['excluded_gt_pair_ids'])
        for name in ('real_cal','real_select','real_test'):
            expected_ids.update(spec['roles'][name]['pair_ids'])
            if summary['groups'][name]['primary']['pairs']!=len(spec['roles'][name]['pair_ids']):
                raise ValueError('real role counts differ')
        if (set(ids)!=expected_ids or summary.get('main_group')!='real_test'
                or summary.get('real_test_is_historically_unseen') is not False):
            raise ValueError('source-isolated development population differs')
        if split=='dunhuang_cv' and summary['groups']['gt_corrected_800_development_context']['primary']['pairs']!=800:
            raise ValueError('corrected Dunhuang count differs')
        if split=='turufan':
            if summary.get('layout_gt_available') is not False:raise ValueError('Turufan has no Layout GT')
            for group in summary['groups'].values():
                for report in group.values():
                    for key in ('layout20','joint_f1','joint_fp','candidate_coverage'):
                        if report.get(key) is not None:raise ValueError('invented Turufan layout metric')
    wanted={c['pair_id'] for c in case_plan['cases'] if c['split']==split}
    actual=cases.get('cases',[])
    if (cases.get('selected_by_new_results') is not False or len(actual)!=len(wanted)
            or {c['pair_id'] for c in actual}!=wanted or summary.get('diagnostic_cases')!=actual):
        raise ValueError('fixed case population differs')
    audited=[]
    for case in actual:
        evidence=safe_child(directory,case['evidence']);meta=read(evidence)
        sidecar=safe_child(evidence.parent,meta['sidecar']['path'])
        if (meta.get('pair_id')!=case['pair_id'] or meta.get('variant')!=variant
                or meta.get('provenance',{}).get('checkpoint_sha256')!=selected['checkpoint_sha256']
                or meta['sidecar']['sha256']!=sha(sidecar) or case.get('sidecar_sha256')!=sha(sidecar)
                or case.get('numerical_audit_status')!='passed'
                or read(safe_child(directory,case['numerical_audit'])).get('status')!='passed'):
            raise ValueError('fixed case evidence identity differs')
        result=audit(evidence)
        if result.get('status')!='passed' or result.get('errors')!=[]:raise ValueError('independent case numeric audit failed')
        audited.append(dict(pair_id=case['pair_id'],evidence_sha256=sha(evidence),sidecar_sha256=sha(sidecar)))
    return dict(status='passed',variant=variant,selection_kind=choice,split=split,pairs=len(rows),
        predictions_sha256=frozen['sha256'],summary_sha256=sha(directory/'summary.json'),
        checkpoint_sha256=selected['checkpoint_sha256'],selected_epoch=selected['selected_epoch'],
        fixed_cases=audited,model_state_unchanged=True,real_inference_performed=False)

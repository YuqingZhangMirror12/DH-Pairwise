"""Queue/terminal contracts only. No training or device import."""
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

PREPARED=Path('/root/autodl-tmp/consensus_threshold_joint_20260927')
POST=PREPARED/'evaluation_source_02'
FORMAL=Path('/root/autodl-tmp/s7_consensus_threshold_joint_e32_20260927')
CONTROL=Path('/root/autodl-tmp/s7_consensus_threshold_v1_20260925')
QUEUE=Path('/root/autodl-tmp/scorer_queue_20260928')
REL=Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1')
REFERENCE='/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt'
SPLITS={'sim_test_v14':3000,'dunhuang_cv':803,'turufan':602}
TASKS=[('joint',c,s) for s in SPLITS for c in ('sim','real')]+[('frozen','real',s) for s in SPLITS]

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def inventory(p):return {f.name:sha(f) for f in sorted(Path(p).glob('*.py'))}
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def args_for(kind,out,operation='evaluate',choice='real',split=None):
    if kind not in ('joint','frozen'):raise ValueError('unknown model kind')
    return SimpleNamespace(root=FORMAL if kind=='joint' else CONTROL,reference=REFERENCE,out=Path(out),
        preparation=POST/'preparation.json',common_source=POST/'s7_consensus_eval_v14',
        joint_source=PREPARED/'training_source_02',real_plan=PREPARED/'real_split.json',
        case_plan=POST/'s7_consensus_eval_v14/case_plan.json',model_kind=kind,selection=choice,split=split,
        operation=operation,device='cpu',real_selection=FORMAL/'postprocess_joint_01/control_reselection/real_selection.json')
def evaluator_command(python,kind,choice,split,out):
    if (kind,choice,split) not in TASKS:raise ValueError('unregistered evaluation')
    a=args_for(kind,out,choice=choice,split=split)
    cmd=[python,str(POST/'threshold_joint_eval_v1/entry.py'),'evaluate']
    for key in ('root','reference','out','preparation','common_source','joint_source','real_plan',
                'model_kind','selection','split','case_plan'):
        cmd+=['--'+key.replace('_','-'),str(getattr(a,key))]
    if kind=='frozen':cmd+=['--real-selection',str(a.real_selection)]
    return cmd+['--device','cuda:0']
def accepted(row,threshold):
    return bool(row['has_candidate'] and row['numeric_valid'] and row['score']>=threshold)
def same_model(a,b):
    return all(bool(a.get(k)) and a.get(k)==b.get(k) for k in (
        'model_state_sha256','training_implementation_sha256','data_contract_sha256','geometry_calibration_sha256'))
def safe_child(root,name):
    root=Path(root).resolve();p=(root/name).resolve()
    if not p.is_relative_to(root):raise ValueError('diagnostic path escapes output')
    return p

def verify_result(out,kind,choice,split,selected,roles,cases,audit):
    out=Path(out)
    if (out/'failure.json').exists():raise ValueError('failure overrides stale complete')
    status,summary,protocol,frozen=[read(out/name) for name in ('status.json','summary.json','protocol.json','prediction_complete.json')]
    expected=dict(variant='threshold_joint' if kind=='joint' else 'threshold',selection_kind=choice,
        selected_epoch=selected['selected_epoch'],checkpoint_sha256=selected['checkpoint_sha256'],
        split=split,total_pairs=SPLITS[split],threshold=selected['thresholds'][split])
    if (status.get('status')!='complete' or status.get('pairs')!=SPLITS[split]
            or summary.get('status')!='complete' or protocol.get('status')!='complete'
            or frozen.get('status')!='all_predictions_frozen' or frozen.get('pairs')!=SPLITS[split]
            or frozen.get('model_state_unchanged') is not True or frozen.get('sha256')!=sha(out/'pair_predictions.jsonl')
            or any(any(r.get(k)!=v for k,v in expected.items()) for r in (summary,protocol,frozen))):
        raise ValueError('frozen output binding/terminal/hash differs')
    rows=[json.loads(s) for s in (out/'pair_predictions.jsonl').read_text().splitlines()]
    ids=[r['pair_id'] for r in rows]
    if len(rows)!=SPLITS[split] or len(set(ids))!=len(ids):raise ValueError('missing/duplicate predictions')
    for r in rows:
        if (not isinstance(r.get('score'),(int,float)) or not math.isfinite(r['score'])
                or any(k in r for k in ('label','target_translation_rc','gt_known','layout20'))
                or r.get('accepted')!=accepted(r,expected['threshold'])):
            raise ValueError('invalid or target-leaking raw prediction')
    if split=='sim_test_v14':
        if summary.get('main_group')!='all' or summary['groups']['all']['primary']['pairs']!=3000:
            raise ValueError('SIM population differs')
    else:
        spec=roles['datasets'][split];wanted=set(spec['excluded_gt_pair_ids'])
        for role in ('real_cal','real_select','real_test'):
            wanted.update(spec['roles'][role]['pair_ids'])
            if summary['groups'][role]['primary']['pairs']!=len(spec['roles'][role]['pair_ids']):
                raise ValueError('source-isolated role counts differ')
        if (set(ids)!=wanted or summary.get('main_group')!='real_test'
                or summary.get('real_test_is_historically_unseen') is not False):
            raise ValueError('source-isolated populations differ')
        if split=='dunhuang_cv' and summary['groups']['gt_corrected_800_development_context']['primary']['pairs']!=800:
            raise ValueError('corrected Dunhuang population differs')
        if split=='turufan':
            if summary.get('layout_gt_available') is not False:raise ValueError('Turufan has no Layout GT')
            for group in summary['groups'].values():
                for report in group.values():
                    for key in ('layout20','joint_f1','joint_fp','candidate_coverage'):
                        if report.get(key) is not None:raise ValueError('invented Turufan layout metric')
    diagnostic=read(out/'diagnostic_index.json');wanted={r['pair_id'] for r in cases['cases'] if r['split']==split}
    actual=diagnostic.get('cases',[])
    if (len(actual)!=len(wanted) or {r['pair_id'] for r in actual}!=wanted
            or diagnostic.get('selected_by_new_results') is not False or actual!=summary.get('diagnostic_cases')):
        raise ValueError('fixed diagnostic population differs')
    for item in actual:
        path=safe_child(out,item['evidence']);meta=read(path);side=safe_child(path.parent,meta['sidecar']['path'])
        if (meta['pair_id']!=item['pair_id'] or meta['provenance']['checkpoint_sha256']!=selected['checkpoint_sha256']
                or meta['semantics']['variant']!='threshold' or meta['semantics']['evidence_mode']!='exact_union_q'
                or sha(side)!=meta['sidecar']['sha256'] or sha(side)!=item['sidecar_sha256']
                or item.get('numerical_audit_status')!='passed'
                or read(safe_child(out,item['numerical_audit'])).get('status')!='passed'):
            raise ValueError('fixed diagnostic identity differs')
        if audit(path).get('status')!='passed':raise ValueError('numeric diagnostic audit failed')
    return dict(status='passed',kind=kind,selection_kind=choice,split=split,pairs=len(rows),
        predictions_sha256=frozen['sha256'],summary_sha256=sha(out/'summary.json'),fixed_cases=len(actual),
        checkpoint_sha256=selected['checkpoint_sha256'],model_state_unchanged=True,
        reused_predictions=(out/'prediction_reuse.json').exists())

def validate_trace(gate,kind):
    if (gate.get('status')!='passed' or gate.get('kind')!=kind or gate.get('device')!='cuda:0'
            or gate.get('parameters_unchanged') is not True or gate.get('rng_unchanged') is not True
            or gate.get('trained_checkpoint_or_dataset_opened') is not False
            or gate.get('training_started') is not False or gate.get('protocol',{}).get('variant')!='threshold'
            or gate.get('protocol',{}).get('evidence_mode')!='exact_union_q'
            or not gate.get('synthetic_model_cases')
            or any(not r.get('bitwise_outputs_equal') or r.get('audit')!='passed' for r in gate['synthetic_model_cases'])):
        raise ValueError('actual source-bound CUDA Attention gate required')

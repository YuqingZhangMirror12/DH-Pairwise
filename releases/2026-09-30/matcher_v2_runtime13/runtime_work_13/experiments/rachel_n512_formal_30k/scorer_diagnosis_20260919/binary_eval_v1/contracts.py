"""Terminal/source/selection checks, with no optimizer or heldout inference."""
import hashlib
import json
import math
import os
from pathlib import Path

E32_SHA='80cac47d5bc5340df35a7a7c36ab4a3580a9eea8c99adf797ff5744cd2068b17'
PLAN_SHA='0f8c13bbc4277052ecd6d21e335e97bf58dc8522174e2a11341c3221d50450f6'
DOMAINS=('dunhuang_cv','turufan')
STOPS=('simulation_plateau_after_lr_reductions','budget_limit_not_claimed_converged')

def read(path):return json.loads(Path(path).read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def inventory(path,recursive=False):
    path=Path(path);files=path.rglob('*.py') if recursive else path.glob('*.py')
    return {str(p.relative_to(path)):sha(p) for p in sorted(files)}

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('x') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def validate_threshold(v):
    if type(v) not in (int,float) or not math.isfinite(v) or not .2<=v<=.8 or abs(v*100-round(v*100))>1e-8:
        raise ValueError('threshold outside registered CAL grid')

def metric_identity(value):
    result={k:v for k,v in value.items() if k!='elapsed_seconds'}
    if isinstance(result.get('real_development'),dict):
        result['real_development']=metric_identity(result['real_development'])
    return result

def validate_budget(record):
    e=record.get('actual_epochs')
    if (type(e) is not int or not 16<=e<=48 or e%2 or record.get('updates')!=e*750
            or record.get('exposures')!=e*24000 or record.get('stop_reason') not in STOPS):
        raise ValueError('terminal training budget differs')
    return e

def validate_real_report(r):
    if (r.get('status')!='development_selection' or r.get('test_used') is not False
            or r.get('real_used') is not True or r.get('gradients_used') is not False
            or r.get('development_evaluation') is not True or set(r.get('thresholds',{}))!=set(DOMAINS)
            or len(r.get('key',[]))!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in r['key'])):
        raise ValueError('REAL selection is not development-only')
    for value in r['thresholds'].values():validate_threshold(value)

def validate_terminal(cp,selection,complete,terminal,config,variant,choice):
    if variant not in ('patch','stats') or choice not in ('sim','real'):
        raise ValueError('explicit binary variant and selection required')
    b=cp.get('binding',{})
    if (cp.get('stage')!='scorer' or b.get('arm')!='scratch_fixed' or b.get('formal_training') is not True
            or b.get('preflight_steps')!=0 or b.get('config')!=config
            or config.get('schema')!='binary-cluster-scorer/1' or config.get('scorer_variant')!=variant):
        raise ValueError('not the registered binary formal checkpoint')
    if (selection.get('status')!='selected' or complete.get('status')!='stage_complete'
            or terminal.get('status')!='training_complete' or terminal.get('arm')!='scratch_fixed'
            or terminal.get('stages')!=['scorer'] or terminal.get('last_stage')!=complete
            or any(r.get('binding')!=b for r in (selection,complete,terminal))
            or {k:v for k,v in selection.items() if k!='status'}!={k:v for k,v in complete.items() if k!='status'}):
        raise ValueError('binary terminal identities differ')
    epochs=validate_budget(selection)
    if (selection.get('matcher_unchanged') is not True or selection.get('selection_on_real') is not True
            or selection.get('test_used') is not False or selection.get('migration_origin') is not None):
        raise ValueError('binary head must preserve E32 and independent REAL selection')
    chosen=selection['best' if choice=='sim' else 'best_real'];e=chosen['epoch']
    if (type(e) is not int or e%2 or not (0 if choice=='sim' else 2)<=e<=epochs
            or cp.get('epoch')!=e or cp.get('metrics',{}).get('key')!=chosen['key']):
        raise ValueError('checkpoint is not selected epoch')
    if choice=='sim':
        validate_threshold(chosen['threshold'])
        if cp.get('threshold')!=chosen['threshold'] or cp['metrics'].get('threshold')!=chosen['threshold']:
            raise ValueError('SIM CAL threshold differs')
    else:
        validate_real_report(cp['metrics'])
        if cp.get('thresholds')!=chosen['thresholds'] or cp['metrics']['thresholds']!=chosen['thresholds']:
            raise ValueError('REAL CAL thresholds differ')
    return b

def verify_selection_curve(rows,selection):
    """Recheck the recorded selection, without fitting any new threshold."""
    epochs=validate_budget(selection)
    if [r.get('epoch') for r in rows]!=list(range(0,epochs+1,2)):
        raise ValueError('validation curve incomplete or reordered')
    for r in rows:
        if r.get('updates')!=r['epoch']*750 or r.get('exposures')!=r['epoch']*24000:
            raise ValueError('validation update identity differs')
        validate_threshold(r['threshold']);validate_real_report(r['real_development'])
        if not r.get('key') or any(type(v) not in (float,int) or not math.isfinite(v) for v in r['key']):
            raise ValueError('nonfinite SIM selection key')
    sim=max(rows,key=lambda r:tuple(r['key']))
    real=max(rows[1:],key=lambda r:tuple(r['real_development']['key']))
    if selection['best']!={'epoch':sim['epoch'],'key':sim['key'],'threshold':sim['threshold']}:
        raise ValueError('SIM winner differs from full recorded curve')
    if selection['best_real']!={'epoch':real['epoch'],'key':real['real_development']['key'],
                              'thresholds':real['real_development']['thresholds']}:
        raise ValueError('REAL winner differs from full recorded curve')

def partition_real(rows,plan,domain):
    spec=plan['datasets'][domain];by_id={r['pair_id']:r for r in rows}
    roles=('real_cal','real_select','real_test');expected=set(spec['excluded_gt_pair_ids'])
    for role in roles:expected.update(spec['roles'][role]['pair_ids'])
    if len(by_id)!=len(rows) or set(by_id)!=expected:raise ValueError('real population differs from source roles')
    out={'all_development_context':rows}
    for role in roles:out[role]=[by_id[i] for i in spec['roles'][role]['pair_ids']]
    if domain=='dunhuang_cv':
        out['gt_corrected_800_development_context']=[r for r in rows if r['pair_id'] not in spec['excluded_gt_pair_ids']]
    return out

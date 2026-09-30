"""One explicit CPU-only recovery for the observed 5px roundoff audit defect.

The bound generator/source/config and committed examples are never modified.
Only the independent auditor's lower comparison gets a 1e-9px numerical
tolerance. Old failure receipts remain in history. No automatic retry.
"""
import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

BASE=Path('/root/autodl-tmp/aggressive_data_v17_full30k_20260927')
DATA=BASE/'dataset_03'
SOURCE=BASE/'source_03'
ROOT=BASE/'recovery_numeric_01'
REL=Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/aggressive_data_full_v17')
PACKAGE='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_data_full_v17'
SOURCE_SHA='5c083e589287ad4c76820ccab78ffb5e1262b16a2c2eead5e4e2a1371046f05c'
CONFIG_SHA='24a2e5df2e1e96b9c51e77b8b48364b6006eed4daaea51a7920a3dc864896900'
EXPECTED_ERROR="AssertionError('final PRIMARY footprint peak5–35px including weak')"
REGISTER=Path('/root/autodl-tmp/aggressive_binary_20260927/queued_source_02/register.py')
ADMISSION=Path('/root/autodl-tmp/scorer_queue_20260928/admissions/aggressive_scratch.json')
OLD='if has_primary:need(5<=max(active_gaps)<=35+1e-4'
NEW='if has_primary:need(5-1e-9<=max(active_gaps)<=35+1e-4'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def inventory(p):return {f.name:sha(f) for f in sorted(Path(p).glob('*.py'))}
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def identity(pid):
    p=Path('/proc')/str(pid);a=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime=int(a[19]),state=a[0],cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
def live(record):
    try:a=identity(record['pid'])
    except (FileNotFoundError,ProcessLookupError):return False
    return a['state'] not in ('Z','X') and a['starttime']==record['starttime']
def environment():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
def verify_text(original,corrected):
    if original.count(OLD)!=1 or original.replace(OLD,NEW)!=corrected:
        raise ValueError('auditor must differ by only the documented 1e-9 lower-bound tolerance')
def validate_quiescent():
    if sha(DATA/'config.json')!=CONFIG_SHA or sha(SOURCE/'source_binding.json')!=SOURCE_SHA:
        raise ValueError('bound generation config/source changed')
    for path,digest in read(SOURCE/'source_binding.json').items():
        if sha(SOURCE/path)!=digest:raise ValueError('original generation source modified')
    config=read(DATA/'config.json')
    if config['source']!=str(SOURCE) or config['out']!=str(DATA):raise ValueError('wrong dataset')
    if (DATA/'pipeline_complete.json').exists() or ADMISSION.exists():raise ValueError('already completed/admitted; do not recover')
    if read(DATA/'pipeline_failure.json').get('error')!=EXPECTED_ERROR:raise ValueError('unrelated failure; no recovery')
    if live(read(DATA/'pipeline_launch.json')):raise ValueError('original pipeline still running')
    for p in DATA.glob('pipeline_resume_*.json'):
        if live(read(p)):raise ValueError('another resume still running')
    verify_text((SOURCE/REL/'audit.py').read_text(),(Path(__file__).with_name('audit.py')).read_text())
    return config
def corrected_auditor():
    name=PACKAGE+'.audit_numeric_recovery'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('audit.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
def files_to_preserve():
    return {str(p.relative_to(DATA)):sha(p) for split in ('train','cal','select','test')
        for folder in ('groups','audits') for p in sorted((DATA/split/folder).glob('*.json'))}
def check_preserved(before):
    for rel,digest in before.items():
        if sha(DATA/rel)!=digest:raise ValueError('pre-existing committed group/audit changed:'+rel)
def preflight():
    config=validate_quiescent()
    if (ROOT/'preflight.json').exists():raise ValueError('existing regression receipt; do not repeat')
    sys.path.insert(0,str(SOURCE))
    source=importlib.import_module(PACKAGE+'.source');source.initialize(config,'cal')
    original=importlib.import_module(PACKAGE+'.audit');corrected=corrected_auditor()
    group=read(DATA/'cal/groups/00285.json');record=group['records'][0]
    before={p:sha(p) for p in (record['sample_path'],record['proof_path'],record['target_metadata'])}
    try:original.audit_record(record,source.STATE['root'])
    except AssertionError as error:
        if str(error)!='final PRIMARY footprint peak5–35px including weak':raise
    else:raise ValueError('observed legacy defect no longer reproducible')
    result=corrected.audit_record(record,source.STATE['root'])
    actual=result['paired_gap']['new_primary_gap_peak_px']
    if not (5-1e-9<=actual<5):raise ValueError('not the observed lower-bound roundoff')
    if before!={p:sha(p) for p in before}:raise ValueError('regression changed a sample')
    save(ROOT/'preflight.json',dict(status='passed',record_id=record['id'],legacy_failure=EXPECTED_ERROR,
        recomputed_primary_peak_px=actual,stored_primary_peak_px=record['detail']['gap']['primary_gap_peak_px'],
        corrected_all_record_checks=result,source_sha256=inventory(Path(__file__).parent),
        immutable_sample_sha256=before,config_sha256=CONFIG_SHA,original_source_binding_sha256=SOURCE_SHA,
        only_change='lower audit bound numerical tolerance1e-9px; generator and all physical rules unchanged',
        gpu_used=False,gpu_tasks_started=False))
    print(json.dumps(dict(status='passed',actual_peak=actual,samples_unchanged=True)))
def worker_initialize(config,split):
    source=importlib.import_module(PACKAGE+'.source');source.initialize(config,split)
    pipeline=importlib.import_module(PACKAGE+'.pipeline');pipeline.audit_record=corrected_auditor().audit_record
def recovery_finish(config):
    original=importlib.import_module(PACKAGE+'.finish');result=original.finish(config)
    check_preserved(read(ROOT/'preexisting.json'))
    provenance=dict(schema='independent-audit-numeric-recovery/1',tolerance_lower_px=1e-9,
        physical_range_px=[5,35],upper_audit_tolerance_px=1e-4,
        original_source_binding_sha256=SOURCE_SHA,original_audit_sha256=sha(SOURCE/REL/'audit.py'),
        corrected_audit_sha256=sha(Path(__file__).with_name('audit.py')),
        recovery_source_sha256=inventory(Path(__file__).parent),preflight_sha256=sha(ROOT/'preflight.json'),
        cpu_tests_sha256=sha(ROOT/'cpu_tests_remote.json'),preserved_failure_sha256=sha(ROOT/'history/pipeline_failure.json'),
        preserved_group_and_audit_receipts_sha256=sha(ROOT/'preexisting.json'),
        previously_passed_stricter_audits_reused=True,committed_examples_changed=False)
    audit=read(DATA/'full_audit.json');audit['numerical_audit_recovery']=provenance;save(DATA/'full_audit.json',audit)
    contract=read(DATA/'data_contract.json');contract['aggressive_full_audit']['sha256']=sha(DATA/'full_audit.json')
    contract['numerical_audit_recovery']=provenance;save(DATA/'data_contract.json',contract)
    save(ROOT/'data_preservation.json',dict(status='passed',committed_examples_changed=False,source_modified=False,provenance=provenance))
    return result
def validate_cpu():
    r=read(ROOT/'cpu_tests_remote.json')
    if r.get('status')!='passed' or r.get('tests',0)<8 or any(r.get(k)!=0 for k in ('errors','failures','skipped')):
        raise ValueError('CPU recovery tests missing/failed')
    if r['source_sha256']!=inventory(Path(__file__).parent):raise ValueError('recovery source changed after tests')
    p=read(ROOT/'preflight.json')
    if p.get('status')!='passed' or p.get('source_sha256')!=r['source_sha256']:raise ValueError('actual sample regression not bound')
def driver():
    with (ROOT/'recovery.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate_cpu();validate_quiescent()
        if (ROOT/'history').exists():raise ValueError('already attempted; no automatic retry')
        save(ROOT/'preexisting.json',files_to_preserve())
        (ROOT/'history').mkdir()
        for name in ('pipeline_failure.json','pipeline_status.json','pipeline_launch.json','detached_launch.json'):
            shutil.copy2(DATA/name,ROOT/'history'/name)
        # Preserve failure bytes before clearing the current failure marker for
        # this explicitly verified recovery; history remains immutable.
        if sha(ROOT/'history/pipeline_failure.json')!=sha(DATA/'pipeline_failure.json'):raise ValueError('failure archive mismatch')
        (DATA/'pipeline_failure.json').unlink()
        sys.path.insert(0,str(SOURCE));pipeline=importlib.import_module(PACKAGE+'.pipeline')
        pipeline.audit_record=corrected_auditor().audit_record
        pipeline.initialize=worker_initialize;pipeline.finish=recovery_finish
        save(ROOT/'status.json',dict(status='resuming_unfinished_data_phases',controller=identity(os.getpid()),
            gpu_used=False,automatic_retries=0,config_changed=False,generator_source_changed=False))
        sys.argv=['audited-numeric-recovery','--out',str(DATA),'--resume'];pipeline.main()
        complete=read(DATA/'pipeline_complete.json')
        if complete.get('status')!='complete' or complete.get('pairs')!=30000:
            raise ValueError('no complete30K; cannot register GPU experiment')
        check_preserved(read(ROOT/'preexisting.json'))
        save(ROOT/'status.json',dict(status='full_data_complete_registering',gpu_used=False,automatic_retries=0))
        if ADMISSION.exists():raise ValueError('another owner registered; inspect rather than overwrite')
        command=[sys.executable,str(REGISTER)]
        with (ROOT/'registration.log').open('xb') as log:
            job=subprocess.Popen(command,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            save(ROOT/'registration_launch.json',dict(controller=identity(job.pid),command=command,started_unix=time.time()))
            code=job.wait()
        save(ROOT/'registration_exit.json',dict(returncode=code,finished_unix=time.time()))
        if code!=0 or not ADMISSION.exists():raise ValueError('registration failed; no automatic retry')
        value=dict(status='complete',data_complete_sha256=sha(DATA/'pipeline_complete.json'),
            admission_sha256=sha(ADMISSION),gpu_used=False,gpu_tasks_started=False,
            generator_source_changed=False,committed_examples_changed=False,automatic_retries=0,finished_unix=time.time())
        save(ROOT/'complete.json',value);save(ROOT/'status.json',value)
def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--driver',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only recovery')
    if a.preflight:preflight();return
    if a.driver:
        try:driver()
        except BaseException as error:
            value=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retries=0,gpu_tasks_started=False)
            save(ROOT/'failure.json',value);save(ROOT/'status.json',value);raise
        return
    validate_cpu();validate_quiescent()
    if any((ROOT/name).exists() for name in ('controller_launch.json','failure.json','complete.json')):
        raise ValueError('recovery already attempted; no repeat launch')
    command=[sys.executable,str(Path(__file__).resolve()),'--driver']
    with (ROOT/'controller.log').open('xb') as log:
        job=subprocess.Popen(command,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(controller=identity(job.pid),command=command,created_unix=time.time(),automatic_retries=0,gpu_used=False)
    save(ROOT/'controller_launch.json',launch);print(json.dumps(launch))
if __name__=='__main__':main()

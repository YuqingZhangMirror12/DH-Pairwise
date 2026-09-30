"""Finalize fully audited retained-v14 data, preserving existing repetitions."""
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
ROOT=BASE/'retention_finish_01'
PKG='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_data_full_v17'
REGISTER=Path('/root/autodl-tmp/aggressive_binary_20260927/queued_source_02/register.py')
ADMISSION=Path('/root/autodl-tmp/scorer_queue_20260928/admissions/aggressive_scratch.json')


def load(path,name):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def numeric():return load(BASE/'recovery_numeric_source_01/recover.py','retention_numeric_runtime')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory():return {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def validate():
    old=numeric();old.validate_cpu()
    if sha(DATA/'config.json')!=old.CONFIG_SHA or sha(old.SOURCE/'source_binding.json')!=old.SOURCE_SHA:
        raise ValueError('immutable source/config binding changed')
    for name,digest in read(old.SOURCE/'source_binding.json').items():
        if sha(old.SOURCE/name)!=digest:raise ValueError('generator source modified')
    for path in [DATA/'pipeline_launch.json']+list(DATA.glob('pipeline_resume_*.json')):
        if old.live(read(path)):raise ValueError('another data owner still running')
    if (DATA/'pipeline_complete.json').exists() or ADMISSION.exists():raise ValueError('already finalized/admitted')
    if read(DATA/'pipeline_failure.json').get('error')!="ValueError('duplicate numerical sample content')":
        raise ValueError('unrelated failure; inspect rather than resume')
    if read(BASE/'recovery_audit_fallback_01/failure.json').get('error')!="ValueError('duplicate numerical sample content')":
        raise ValueError('latest failed controller differs')
    for split,count in [('train',12000),('cal',750),('select',750),('test',1500)]:
        for phase in ('full_generate','full_audit'):
            receipt=read(DATA/'phase_receipts'/f'{phase}_{split}.json')
            if receipt.get('status')!='passed' or receipt.get('groups')!=count:
                raise ValueError('full generation and pixel audit required before finalization')
    sys.path.insert(0,str(old.SOURCE))
    return read(DATA/'config.json')


def finalizer():return load(Path(__file__).with_name('finish.py'),PKG+'.retention_finish')


def preserve_check():
    for path,digest in read(ROOT/'preexisting.json').items():
        if sha(DATA/path)!=digest:raise ValueError('finalization modified sample group/audit: '+path)


def finish(config):
    # Nothing is regenerated; the only policy amendment recognizes repetitions
    # which already exist in the retained original v14 and original split.
    result=finalizer().finish(config)
    preserve_check()
    replaced={str(p.relative_to(BASE/'recovery_audit_fallback_01')):dict(read(p),receipt_sha256=sha(p))
              for p in (BASE/'recovery_audit_fallback_01/replacements').glob('*/*.json')}
    for record in replaced.values():
        history=Path(record['history'])
        if sha(history/'preserved.json')!=record['history_manifest_sha256']:raise ValueError('rejected archive manifest changed')
        for path,digest in read(history/'preserved.json').items():
            if sha(history/path)!=digest:raise ValueError('rejected artifact changed')
    provenance=dict(schema='retained-v14-finalization/1',source_sha256=inventory(),
        generator_source_binding_sha256=numeric().SOURCE_SHA,source_config_sha256=numeric().CONFIG_SHA,
        source_or_sample_files_modified=False,real_or_test_results_not_used=True,
        existing_numeric_auditor_sha256=sha(BASE/'recovery_numeric_source_01/audit.py'),
        numeric_audit_cpu_sha256=sha(BASE/'recovery_numeric_01/cpu_tests_remote.json'),
        post_audit_same_pair_fallback=replaced,
        fallback_preflight_sha256=sha(BASE/'recovery_audit_fallback_01/preflight.json'),
        cpu_test_sha256=sha(ROOT/'cpu_tests_remote.json'))
    a=read(DATA/'full_audit.json');a['retention_finalization']=provenance;save(DATA/'full_audit.json',a)
    c=read(DATA/'data_contract.json');c['aggressive_full_audit']['sha256']=sha(DATA/'full_audit.json')
    c['retention_finalization']=provenance;save(DATA/'data_contract.json',c)
    save(ROOT/'retention_audit.json',dict(status='passed',duplicate_count=a['exact_numerical_duplicate_count'],
         unique_numeric_samples=a['unique_numerical_sample_count'],summaries=result,**provenance))
    return result


def validate_tests():
    t=read(ROOT/'cpu_tests_remote.json')
    if t.get('status')!='passed' or t.get('tests',0)<8 or any(t.get(k)!=0 for k in ('errors','failures','skipped')) or t['source_sha256']!=inventory():
        raise ValueError('finalization CPU test binding differs')


def driver():
    with (ROOT/'finish.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);validate_tests();config=validate()
        if (ROOT/'history').exists():raise ValueError('already attempted; no implicit retry')
        save(ROOT/'preexisting.json',numeric().files_to_preserve());(ROOT/'history').mkdir()
        for name in ('pipeline_failure.json','pipeline_status.json'):
            shutil.copy2(DATA/name,ROOT/'history'/name)
            if sha(DATA/name)!=sha(ROOT/'history'/name):raise ValueError('failed history copy mismatch')
        (DATA/'pipeline_failure.json').unlink()
        pipeline=importlib.import_module(PKG+'.pipeline');pipeline.finish=finish
        save(ROOT/'status.json',dict(status='finalizing_full_audited_30k',controller=numeric().identity(os.getpid()),gpu_used=False))
        sys.argv=['retention-finalizer','--out',str(DATA),'--resume'];pipeline.main();preserve_check()
        done=read(DATA/'pipeline_complete.json')
        if done.get('status')!='complete' or done.get('pairs')!=30000:raise ValueError('missing complete30K')
        save(ROOT/'status.json',dict(status='registering_completed_data',gpu_used=False))
        if ADMISSION.exists():raise ValueError('admission already registered by another owner')
        with (ROOT/'registration.log').open('xb') as log:
            process=subprocess.Popen([sys.executable,str(REGISTER)],env=numeric().environment(),stdout=log,stderr=subprocess.STDOUT)
            save(ROOT/'registration_launch.json',dict(controller=numeric().identity(process.pid)));code=process.wait()
        save(ROOT/'registration_exit.json',dict(returncode=code,finished_unix=time.time()))
        if code or not ADMISSION.exists():raise ValueError('data registration failed; no automatic retry')
        result=dict(status='complete',finished_unix=time.time(),gpu_used=False,gpu_tasks_started=False,
             data_complete_sha256=sha(DATA/'pipeline_complete.json'),admission_sha256=sha(ADMISSION))
        save(ROOT/'complete.json',result);save(ROOT/'status.json',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--driver',action='store_true');args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only finalization')
    if args.driver:
        try:driver()
        except BaseException as error:
            result=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retries=0)
            save(ROOT/'failure.json',result);save(ROOT/'status.json',result);raise
    else:
        validate_tests();validate()
        if any((ROOT/p).exists() for p in ('controller_launch.json','failure.json','complete.json')):raise ValueError('already registered; no duplicate launch')
        command=[sys.executable,str(Path(__file__).resolve()),'--driver']
        with (ROOT/'controller.log').open('xb') as log:
            process=subprocess.Popen(command,env=numeric().environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result=dict(controller=numeric().identity(process.pid),command=command,created_unix=time.time(),automatic_retries=0)
        save(ROOT/'controller_launch.json',result);print(json.dumps(result))


if __name__=='__main__':main()

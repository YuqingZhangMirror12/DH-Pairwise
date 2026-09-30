"""CPU event bridge: the existing full-data job may admit experiment3 once.

No dataset generation, GPU query, model launch, retry, or remote polling. The
already running server dispatcher alone decides when a released lane is used.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PREPARED=Path('/root/autodl-tmp/aggressive_binary_20260927')
ROOT=PREPARED/'data_admission_01'
DATA=Path('/root/autodl-tmp/aggressive_data_v17_full30k_20260927/dataset_03')
SOURCE=DATA.parent/'source_03'
SOURCE_BINDING_SHA='5c083e589287ad4c76820ccab78ffb5e1262b16a2c2eead5e4e2a1371046f05c'
QUEUE=Path('/root/autodl-tmp/scorer_queue_20260928')
REGISTER=PREPARED/'queued_source_02/register.py'
ADMISSION=QUEUE/'admissions/aggressive_scratch.json'
PYTHON='/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def inventory(path):return {p.name:sha(p) for p in sorted(Path(path).glob('*.py'))}
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def identity(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime=int(fields[19]),state=fields[0],
        cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
def live(record):
    try:actual=identity(record['pid'])
    except (FileNotFoundError,ProcessLookupError):return False
    return actual['state'] not in ('Z','X') and all(actual[k]==record[k] for k in ('starttime','cmdline'))
def environment():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def validate_preparation():
    receipt=read(PREPARED/'data_admission_preparation_remote_01.json')
    if (receipt.get('status')!='passed' or receipt.get('tests',0)<9
            or any(receipt.get(k)!=0 for k in ('errors','failures','skipped'))
            or receipt.get('source_sha256')!=inventory(Path(__file__).parent)
            or receipt.get('gpu_tasks_started') is not False):raise ValueError('event bridge CPU tests/binding missing')
    wrapper=read(PREPARED/'queued_preparation_remote_02.json')
    if (wrapper.get('status')!='passed' or wrapper.get('tests')!=5
            or any(wrapper.get(k)!=0 for k in ('errors','failures','skipped'))
            or wrapper.get('source_sha256')!=inventory(REGISTER.parent)):
        raise ValueError('tested experiment3 registration code changed')
    return receipt,wrapper

def data_binding():
    config=read(DATA/'config.json');launch=read(DATA/'pipeline_launch.json')
    if (Path(config.get('out',''))!=DATA or Path(config.get('source',''))!=SOURCE
            or config.get('source_binding_sha256')!=SOURCE_BINDING_SHA
            or sha(SOURCE/'source_binding.json')!=SOURCE_BINDING_SHA
            or launch.get('config_sha256')!=sha(DATA/'config.json')
            or launch.get('gpu_used') is not False or launch.get('automatic_retries')!=0
            or 'aggressive_data_full_v17.pipeline' not in launch.get('cmdline','')
            or str(DATA) not in launch.get('cmdline','') or '--pilot-only' in launch.get('cmdline','')):
        raise ValueError('must attach only the existing approved full dataset03 process')
    return dict(dataset=str(DATA),source=str(SOURCE),launch_sha256=sha(DATA/'pipeline_launch.json'),
        config_sha256=sha(DATA/'config.json'),source_binding_sha256=SOURCE_BINDING_SHA,controller=launch)

def state(binding):
    # Failure has priority, even if an old terminal file also exists.
    if (DATA/'pipeline_failure.json').exists():raise RuntimeError('data generation/audit failed; preserve output, no retry')
    if sha(DATA/'pipeline_launch.json')!=binding['launch_sha256'] or sha(DATA/'config.json')!=binding['config_sha256']:
        raise ValueError('data process/config replaced; explicit recovery required')
    if (DATA/'pipeline_complete.json').exists():return 'data_terminal'
    if live(binding['controller']):return 'waiting_for_live_data_pipeline'
    # Closing race: a producer can finish between the first file check and /proc.
    if (DATA/'pipeline_failure.json').exists():raise RuntimeError('data failure')
    if (DATA/'pipeline_complete.json').exists():return 'data_terminal'
    raise RuntimeError('bound data process exited without completion; no automatic restart')

def verify_admission():
    record=read(ADMISSION)
    expected=[PYTHON,str(REGISTER.with_name('run_queued.py')),'--gpus','{gpus}',
        '--release','{release}','--work',str(QUEUE/'work/aggressive_scratch')]
    if (record.get('schema')!='verified-future-gpu-task/1' or record.get('status')!='ready'
            or record.get('task')!='aggressive_scratch' or record.get('command')!=expected
            or record.get('work')!=str(QUEUE/'work/aggressive_scratch')
            or record.get('gpu_jobs_started') is not False or not record.get('data_admission')):
        raise ValueError('unexpected existing admission')
    required={str(DATA/name) for name in ('pipeline_complete.json','data_contract.json','human_approval.json',
        'geometry_calibration_v2/geometry_calibration.json')}
    if not required<=set(record.get('files_sha256',{})):raise ValueError('data admission omits terminal/calibration binding')
    for path,digest in record['files_sha256'].items():
        if sha(path)!=digest:raise ValueError('registered experiment3 source/data changed')
    return record

def admit():
    if ADMISSION.exists():
        verify_admission();return dict(status='already_admitted',new_registration=False)
    # This validated registrar opens the entire contract/audits/new calibration,
    # never just trusts pipeline_complete or a filename to stand for 30K quality.
    command=[PYTHON,str(REGISTER)]
    with (ROOT/'registration.log').open('xb') as log:
        child=subprocess.Popen(command,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        launch=dict(command=command,controller=identity(child.pid),started_unix=time.time(),gpu_started=False)
        save(ROOT/'registration_launch.json',launch);code=child.wait()
    save(ROOT/'registration_exit.json',dict(returncode=code,finished_unix=time.time()))
    if code!=0:raise RuntimeError('data admission refused; no automatic retry')
    verify_admission();return dict(status='admitted',new_registration=True)

def driver():
    with (ROOT/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate_preparation();plan=read(ROOT/'plan.json')
        if plan['source_sha256']!=inventory(Path(__file__).parent):raise ValueError('event source changed')
        binding=plan['data_binding']
        while True:
            value=state(binding)
            save(ROOT/'status.json',dict(status=value,controller=identity(os.getpid()),
                data_controller=binding['controller'],updated_unix=time.time(),gpu_used=False,poll_seconds=60,
                desktop_or_ssh_polling=False,automatic_retries=0))
            if value=='data_terminal':break
            time.sleep(60)
        validate_preparation()
        if plan['registration_source_sha256']!=inventory(REGISTER.parent):
            raise ValueError('registration source changed while data was running')
        result=admit()
        complete=dict(status='complete',result=result,admission=str(ADMISSION),admission_sha256=sha(ADMISSION),
            data_complete_sha256=sha(DATA/'pipeline_complete.json'),gpu_tasks_started=False,
            scheduler_dispatch_still_required=True,automatic_retries=0,finished_unix=time.time())
        save(ROOT/'complete.json',complete);save(ROOT/'status.json',complete)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--driver',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only admission bridge')
    if a.driver:
        try:driver()
        except BaseException as error:
            failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retries=0,gpu_tasks_started=False)
            save(ROOT/'failure.json',failure);save(ROOT/'status.json',failure);raise
        return
    validate_preparation();binding=data_binding();current=state(binding)
    ROOT.mkdir(parents=True,exist_ok=False)
    save(ROOT/'plan.json',dict(schema='full-data-to-experiment3-admission/1',data_binding=binding,
        source_sha256=inventory(Path(__file__).parent),registration_source_sha256=inventory(REGISTER.parent),
        preparation_sha256=sha(PREPARED/'data_admission_preparation_remote_01.json'),
        gpu_tasks_started=False,check_seconds=60,automatic_retries=0,initial_state=current))
    command=[sys.executable,str(Path(__file__).resolve()),'--driver']
    with (ROOT/'controller.log').open('xb') as log:
        child=subprocess.Popen(command,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(controller=identity(child.pid),command=command,created_unix=time.time(),gpu_tasks_started=False)
    save(ROOT/'controller_launch.json',record);print(json.dumps(record))

if __name__=='__main__':main()

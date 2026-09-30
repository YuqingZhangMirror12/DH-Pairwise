"""Scheduling only: no training-source edits, signal delivery or automatic retry."""
import hashlib,json,os,subprocess,time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/scorer_queue_20260928')
PYTHON='/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python'
REFERENCE='/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt'
BINARY=Path('/root/autodl-tmp/binary_scorer_20260927')
BINARY_ROOT=Path('/root/autodl-tmp/s7_binary_scorer_v1_20260927')
ORDER=['binary_patch','binary_stats','aggressive_scratch','joint_e32']
LANES=[dict(id='01',gpus='0,1',variant='simple',arm='m12'),
       dict(id='23',gpus='2,3',variant='threshold',arm='scratch_fixed'),
       dict(id='45',gpus='4,5',variant='simple',arm='scratch_fixed')]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('x') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def identity(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime=int(fields[19]),state=fields[0],cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
def live(record):
    try:a=identity(record['pid'])
    except (FileNotFoundError,ProcessLookupError):return False
    return a['state'] not in ('Z','X') and all(a[k]==record[k] for k in ('starttime','cmdline'))
def free(gpus):
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=15)
    ids={int(row.split(',')[0]):row.split(',')[1].strip() for row in inventory.splitlines() if row.strip()}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=15)
    occupied={row.split(',')[0].strip() for row in apps.splitlines() if row.strip()}
    return all(int(i) in ids and ids[int(i)] not in occupied for i in gpus.split(','))
def old_root(lane):return Path('/root/autodl-tmp/s7_consensus_'+lane['variant']+'_v1_20260925')
def env(gpu='',pythonpath=None):
    value=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if pythonpath:value['PYTHONPATH']=pythonpath
    return value
def launch_wait(command,log,environment):
    with Path(log).open('xb') as f:
        p=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        record=identity(p.pid);save(str(log)+'.launch.json',dict(command=command,identity=record,started_unix=time.time()))
        code=p.wait()
    save(str(log)+'.exit.json',dict(returncode=code,identity=record,finished_unix=time.time()))
    if code:raise RuntimeError('child failed; preserve output, no retry: '+str(log))
def wait_terminal(path,record,failures,expected):
    while True:
        if any(Path(p).exists() for p in failures):raise RuntimeError('failure receipt precedes stale running state')
        if Path(path).exists():
            value=read(path)
            if value.get('status')==expected:return value
        if not live(record):raise RuntimeError('registered process exited without terminal receipt: '+str(path))
        time.sleep(30)


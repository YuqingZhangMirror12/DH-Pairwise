"""One finite two-GPU inference experiment; no training or automatic retries."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from .io import read,sha,ref,save

GPU_UUIDS={0:'GPU-9e93c7ab-397b-0bcb-5c6d-9ea6b9b53c85',1:'GPU-faabfde3-405e-1c7f-9b19-8e538a248eb5'}


def admit(gpus):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    devices={int(v[0].strip()):(v[1].strip(),int(v[2])) for line in text.splitlines() if (v:=line.split(','))}
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    for gpu in gpus:
        if devices[gpu][0]!=GPU_UUIDS[gpu] or devices[gpu][1]>128 or GPU_UUIDS[gpu] in processes:raise ValueError('designated inference GPU is not free: '+str(gpu))
    return dict(devices={str(g):devices[g] for g in gpus},admitted_unix=time.time())


def launch(root,matcher,phase,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUIDS[gpu],CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','rotation_ensemble_v1.worker','--root',str(root),'--matcher',matcher,'--phase',phase]
    log=root/(phase+'_'+matcher+'.log');f=log.open('x')
    process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/(phase+'_'+matcher+'_launch.json'),dict(pid=process.pid,start_ticks=Path('/proc/'+str(process.pid)+'/stat').read_text().split()[21],
        command=command,gpu=gpu,uuid=GPU_UUIDS[gpu],launched_unix=time.time(),log=str(log)))
    return process,f


def returned(root,matcher,phase,process,log):
    code=process.wait();log.close();out=root/phase/matcher
    save(out/'actual_return.json',dict(pid=process.pid,returncode=code,returned_unix=time.time(),
        launch=ref(root/(phase+'_'+matcher+'_launch.json'))))
    if code!=0 or not (out/'complete.json').exists() or (out/'failure.json').exists():raise RuntimeError('inference failed: '+matcher+'/'+phase)


def run(root):
    if (root/'controller_launch.json').exists():raise ValueError('controller already launched; no retry')
    protocol=read(root/'protocol.json');admission=admit([0,1])
    save(root/'controller_launch.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        started_unix=time.time(),protocol=ref(root/'protocol.json'),admission=admission,command=sys.argv,no_training=True))
    jobs={k:launch(root,k,'development',i) for i,k in enumerate(('matcher_sim','matcher_endpoint'))}
    while not all((root/'development'/k/'smoke.json').exists() for k in jobs):
        for k,(p,_) in jobs.items():
            if p.poll() is not None:raise RuntimeError('worker exited before smoke admission: '+k)
        time.sleep(2)
    save(root/'smoke_admitted.json',dict(smokes={k:ref(root/'development'/k/'smoke.json') for k in jobs},admitted_unix=time.time()))
    for k,(p,log) in jobs.items():returned(root,k,'development',p,log)
    save(root/'controller_status.json',dict(state='cpu_development_comparison',updated_unix=time.time()))
    # A separate process exits before TEST admission. No shared live model,
    # target-dependent candidate mutation or test-based threshold refitting.
    subprocess.run([sys.executable,'-m','rotation_ensemble_v1.report','--root',str(root),'--phase','development'],check=True)
    default=read(root/'default.json');admission=admit([0]);save(root/'test_admission.json',dict(admission=admission,default=ref(root/'default.json')))
    matcher=default['setting']['matcher'];p,log=launch(root,matcher,'test',0);returned(root,matcher,'test',p,log)
    subprocess.run([sys.executable,'-m','rotation_ensemble_v1.report','--root',str(root),'--phase','test'],check=True)
    save(root/'complete.json',dict(status='complete',protocol=ref(root/'protocol.json'),development=ref(root/'development_report.json'),
        default=ref(root/'default.json'),test=ref(root/'test_report.json'),finished_unix=time.time(),
        original_models_datasets_training_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:run(a.root)
    except BaseException as e:
        save(a.root/'controller_failure.json',dict(error=str(e),type=type(e).__name__,traceback=traceback.format_exc(),time_unix=time.time()))
        raise

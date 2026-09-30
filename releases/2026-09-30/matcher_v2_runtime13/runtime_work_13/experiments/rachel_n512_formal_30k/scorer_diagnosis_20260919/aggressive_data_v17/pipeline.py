"""Probe + independent audit gate, then bounded review, audit, actual figures."""
import argparse,json,os,subprocess,sys,time,traceback,hashlib
from pathlib import Path

PACKAGE='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_data_v17'
def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--previous',required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();source=Path(a.source).resolve()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(source))
    if a.launch:
        if root.exists():raise ValueError('unique new pipeline root required')
        binding=json.loads((source/'source_binding.json').read_text())
        for path,expected in binding.items():
            if hashlib.sha256((source/path).read_bytes()).hexdigest()!=expected:raise ValueError('bound source changed: '+path)
        root.mkdir(parents=True)
        command=[sys.executable,'-u','-m',PACKAGE+'.pipeline','--source',str(source),'--root',str(root),
                 '--baseline',a.baseline,'--previous',a.previous]
        with (root/'pipeline.log').open('x') as log:process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=process.pid,starttime=Path(f'/proc/{process.pid}/stat').read_text().split()[21],command=command,
            source=str(source),time_unix=time.time(),gpu_used=False,automatic_retries=0,
            source_binding_sha256=hashlib.sha256((source/'source_binding.json').read_bytes()).hexdigest())
        save(root/'pipeline_launch.json',record);print(json.dumps(record));return
    common=['--baseline',a.baseline,'--previous',a.previous]
    stages=[('probe',[PACKAGE+'.run',*common,'--out',str(root/'probe_01'),'--probe']),
        ('probe_audit',[PACKAGE+'.parallel_audit','--pilot',str(root/'probe_01'),'--out',str(root/'probe_audit_01')]),
        ('generation',[PACKAGE+'.run',*common,'--out',str(root/'pilot_01')]),
        ('pixel_audit',[PACKAGE+'.parallel_audit','--pilot',str(root/'pilot_01'),'--out',str(root/'review_bundle_01')]),
        ('render',[PACKAGE+'.render_review','--bundle',str(root/'review_bundle_01'),'--out',str(root/'rendered_01')])]
    results=[]
    try:
        for stage,args in stages:
            command=[sys.executable,'-u','-m',*args]
            with (root/(stage+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                save(root/(stage+'_launch.json'),dict(pid=child.pid,starttime=Path(f'/proc/{child.pid}/stat').read_text().split()[21],command=command,time_unix=time.time()))
                save(root/'pipeline_status.json',dict(status='running',stage=stage,results=results,training_started=False))
                returncode=child.wait()
            results.append(dict(stage=stage,returncode=returncode,finished_unix=time.time()))
            if returncode:raise RuntimeError(stage+' returned '+str(returncode))
        save(root/'pipeline_complete.json',dict(status='cpu_review_artifacts_complete',results=results,training_started=False,full_generation_authorized=False))
        save(root/'pipeline_status.json',dict(status='complete',results=results,training_started=False))
    except BaseException as error:
        save(root/'pipeline_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),results=results,automatic_retry=False,training_started=False));raise

if __name__=='__main__':main()

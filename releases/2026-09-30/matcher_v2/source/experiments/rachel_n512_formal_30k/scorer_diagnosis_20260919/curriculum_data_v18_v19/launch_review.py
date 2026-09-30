"""External CPU-only launcher; no retry or training-controller mutation."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
 p.add_argument('--probe',type=int,default=0);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 source=Path(a.source).resolve();out=Path(a.out).resolve();receipt=out.with_suffix('.launch.json')
 if out.exists() or receipt.exists():raise ValueError('output/launch already exists; no duplicate start')
 binding=json.loads((source/'source_binding.json').read_text())
 for name,sha in binding.items():
  if hashlib.sha256((source/name).read_bytes()).hexdigest()!=sha:raise ValueError('bound source changed '+name)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 python='/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python'
 cmd=[python,'-m','experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.curriculum_data_v18_v19.run','--out',str(out),'--workers',str(a.workers)]
 if a.probe:cmd+=['--probe',str(a.probe)]
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.with_suffix('.log').open('xb') as log:
  child=subprocess.Popen(cmd,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 stat=(Path('/proc')/str(child.pid)/'stat').read_text();starttime=stat.rsplit(')',1)[1].split()[19]
 value=dict(pid=child.pid,proc_starttime=starttime,started_unix=time.time(),command=cmd,cwd=str(source),
  source_binding_sha256=hashlib.sha256((source/'source_binding.json').read_bytes()).hexdigest(),
  out=str(out),gpu_used=False,automatic_retry=False,full_generation_authorized=False)
 receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()

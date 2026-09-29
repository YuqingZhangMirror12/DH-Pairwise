"""External one-shot CPU launcher; source matched successful probe required."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--probe-root',required=True);p.add_argument('--v175-pairs',type=int,required=True)
    p.add_argument('--admission-root',required=True)
    p.add_argument('--v18-pairs',type=int,required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();source=Path(a.source).resolve();out=Path(a.out).resolve();receipt=out.with_suffix('.launch.json')
    if out.exists() or receipt.exists():raise ValueError('output or launch exists; do not duplicate')
    if not 1<=a.workers<=32:raise ValueError('maximum32 single-thread CPU workers')
    for name,sha in json.loads((source/'source_binding.json').read_text()).items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=sha:raise ValueError('bound source changed')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    python='/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python'
    cmd=[python,'-m','experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.curriculum_data_v18_v19.full',
        '--out',str(out),'--probe-root',str(Path(a.probe_root).resolve()),'--v175-pairs',str(a.v175_pairs),
        '--v18-pairs',str(a.v18_pairs),'--workers',str(a.workers),'--admission-root',str(Path(a.admission_root).resolve())]
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.log').open('xb') as log:
        child=subprocess.Popen(cmd,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    stat=(Path('/proc')/str(child.pid)/'stat').read_text();start=stat.rsplit(')',1)[1].split()[19]
    value=dict(pid=child.pid,proc_starttime=start,started_unix=time.time(),command=cmd,cwd=str(source),
        source_binding_sha256=hashlib.sha256((source/'source_binding.json').read_bytes()).hexdigest(),
        out=str(out),gpu_used=False,automatic_retry=False,full_generation_authorized=True,
        targets={'v17.5':a.v175_pairs,'v18':a.v18_pairs},workers=a.workers)
    receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()

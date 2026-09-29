import json,os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
if not (r/'simple_results/complete.json').exists() or (r/'native_audit_launch.json').exists():
    raise SystemExit('require complete comparison, no duplicate native audit')
env=os.environ.copy();env.update(PYTHONPATH='/root/autodl-tmp/s7_consensus_layered_v14_mergefix_20260925/source',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
cmd=[sys.executable,str(r/'audit_native_modes.py'),'--root',str(r/'simple_results'),
    '--phase1','/root/autodl-tmp/consensus_strictness_20260925/results_phase1','--workers','48']
with (r/'native_audit.log').open('xb') as f:p=subprocess.Popen(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
s=Path('/proc')/str(p.pid)
v=dict(pid=p.pid,starttime=int(s.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]),command=cmd)
(r/'native_audit_launch.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))

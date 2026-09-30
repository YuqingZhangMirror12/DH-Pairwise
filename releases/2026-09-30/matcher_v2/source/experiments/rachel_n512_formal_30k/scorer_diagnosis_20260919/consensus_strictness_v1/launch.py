"""Launch one CPU-only diagnostic without touching the training controller."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
out=root/'results_phase1'
launch=root/'launch.json'
if launch.exists() or out.exists():
    raise SystemExit('An existing diagnostic launch/output must be inspected; no automatic restart.')
formal=Path('/root/autodl-tmp/s7_consensus_layered_v14_mergefix_20260925')
env=os.environ.copy()
env.update(PYTHONPATH=str(formal/'source'),PYTHONDONTWRITEBYTECODE='1',
    CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
command=[sys.executable,str(root/'measure.py'),'--out',str(out),'--workers','16',
    '--case-plan',str(root/'case_plan.json'),'--guide-sha',
    hashlib.sha256((root/'CC_to_CodeX.md').read_bytes()).hexdigest()]
with (root/'measure.log').open('xb') as log:
    process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
identity=Path('/proc')/str(process.pid)
record=dict(pid=process.pid,command=command,cpu_workers=16,cuda_visible_devices='',
    starttime=identity.joinpath('stat').read_text().split()[21],
    cmdline=identity.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(),
    output=str(out),no_training_changes=True)
launch.write_text(json.dumps(record,indent=2))
print(json.dumps(record))

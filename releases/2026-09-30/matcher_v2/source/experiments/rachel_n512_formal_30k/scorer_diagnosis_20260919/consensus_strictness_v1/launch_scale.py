"""One immutable CPU sweep; never stop/restart or import into training."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
out = root/'results'
launch = root/'launch.json'
if out.exists() or launch.exists():
    raise SystemExit('existing run must be inspected, no automatic restart')
formal = Path('/root/autodl-tmp/s7_consensus_layered_v14_mergefix_20260925')
env = os.environ.copy()
env.update(PYTHONPATH=str(formal/'source'), PYTHONDONTWRITEBYTECODE='1',
    CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
command = [sys.executable, str(root/'scale_sweep.py'), '--out', str(out), '--workers', '16']
with (root/'sweep.log').open('xb') as log:
    proc = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
p = Path('/proc')/str(proc.pid)
record = dict(pid=proc.pid, starttime=int(p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]),
    cmdline=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(), command=command,
    cpu_workers=16, cuda_visible_devices='', output=str(out), no_training_changes=True)
launch.write_text(json.dumps(record, indent=2))
print(json.dumps(record))

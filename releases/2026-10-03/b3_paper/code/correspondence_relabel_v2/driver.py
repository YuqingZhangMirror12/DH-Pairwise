"""Single-use relabel process driver with real child return, no retries/monitor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')

def identity(pid):
    p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(s[19]),cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--launch',action='store_true');p.add_argument('--run',action='store_true')
    args=p.parse_args();src=Path(__file__).resolve().parent;root=args.root
    if args.launch:
        root.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                 PYTHONPATH=str(src)+':/root/autodl-tmp/matcher_v2_20260930/runtime_work_13')
        command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--run']
        with (root/'driver.log').open('xb') as f:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        launch=dict(command=command,process=identity(child.pid),started_unix=time.time(),gpu=False,automatic_retry=False,
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('*.py')})
        write(root/'launch.json',launch);print(json.dumps(launch));return
    if not args.run:raise ValueError('choose launch or run')
    command=['nice','-n','10',sys.executable,str(src/'build_full.py'),'--out-new',str(root/'build'),'--workers','6']
    with (root/'build.log').open('xb') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(root/'build_launch.json',dict(command=command,process=identity(child.pid),started_unix=time.time()))
        code=child.wait()
    write(root/'actual_return.json',dict(returncode=code,ended_unix=time.time(),command=command))
    if code:write(root/'failure.json',dict(returncode=code,no_automatic_retry=True))
    sys.exit(code)

if __name__=='__main__':main()

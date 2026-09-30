"""Preview or explicitly launch the isolated Full24 paired-density CPU job.

Default is preview-only. --launch is required to start the detached job.
Existing source_selection triggers exact-identity resume; original live data
and live training source are not modified. A local launch receipt records PID,
command and log. Completion is determined by data/run_state.json, not PID exit.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(args):
    root=Path(args.production_root).resolve(strict=True);source=root/'source';output=root/'data'
    if not (source/'experiments/rachel_n512_formal_30k/materialize_source_density.py').exists():
        raise ValueError('isolated production source snapshot missing')
    command=[args.python,'-m','experiments.rachel_n512_formal_30k.materialize_source_density',
        '--source-manifest',args.source_manifest,'--canonical-root',args.canonical_root,'--output',str(output)]
    if (output/'source_selection.json').exists():command.append('--resume')
    record=dict(preview_only=not args.launch,command=command,cwd=str(source),output=str(output),
        resume='--resume' in command,gpu_used=False)
    if not args.launch:print(json.dumps(record,indent=2));return 0
    if shutil.disk_usage(root).free<args.minimum_free_gb*1e9:raise ValueError('insufficient requested free disk reserve')
    state=output/'run_state.json'
    if state.exists() and json.loads(state.read_text()).get('status')=='complete':
        raise ValueError('full selected materialization already complete; refusing duplicate launch')
    previous=root/'launch.json'
    if previous.exists():
        old=json.loads(previous.read_text());pid=old.get('pid')
        proc=Path('/proc')/str(pid)/'cmdline'
        if proc.exists():
            cmd=proc.read_bytes().decode(errors='replace')
            if 'materialize_source_density' in cmd and str(output) in cmd:
                raise ValueError('same isolated materialization is already running, PID '+str(pid))
    log=root/('materialize_'+str(time.time_ns())+'.log')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    prefix=[shutil.which('nice'),'-n','10'] if shutil.which('nice') else []
    with log.open('ab') as stream:
        child=subprocess.Popen(prefix+command,cwd=source,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record.update(preview_only=False,pid=child.pid,log=str(log),started_at_unix=time.time(),
        completion_source=str(state),cpu_priority_nice=10 if prefix else None)
    temporary=previous.with_suffix('.json.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');os.replace(temporary,previous)
    print(json.dumps(record,indent=2));return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-root',required=True);parser.add_argument('--source-manifest',required=True)
    parser.add_argument('--canonical-root',required=True);parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--minimum-free-gb',type=float,default=5.)
    parser.add_argument('--launch',action='store_true')
    sys.exit(run(parser.parse_args()))

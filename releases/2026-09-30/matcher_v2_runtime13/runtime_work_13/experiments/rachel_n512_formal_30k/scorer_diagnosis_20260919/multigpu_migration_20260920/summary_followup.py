"""Run the existing CPU endpoint summary once lanes1/2/3 finish.

This replaces the one CPU leaf omitted when the old serial supervisor was split.
No training, evaluation, model selection, inference or recurring user messages.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time
from lane_runner import read, require_subset, save


def source_stage(priority):
    stages=[s for s in priority['stages'] if s['name']=='summarize_new_seven_arms']
    if len(stages)!=1: raise ValueError('expected exactly one original CPU summary')
    stage=stages[0]
    if stage['command'][1:3]!=['-m','matched_only.summarize_priority']:
        raise ValueError('only the original read-only summary is allowed')
    return stage


def readiness(migration_root):
    states={name:read(Path(migration_root)/name/'status.json') for name in ('lane1','lane2','lane3')}
    failed=[name for name,state in states.items() if state['status']=='failed']
    if failed: raise RuntimeError('producer lanes failed: '+','.join(failed))
    return all(state['status']=='complete' for state in states.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--priority-plan',required=True)
    p.add_argument('--migration-root',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    stage=source_stage(read(args.priority_plan))
    root=Path(args.migration_root)/'summary_followup'
    if not args.execute:
        print(stage['command']);return
    root.mkdir(exist_ok=False)
    state=dict(status='waiting',pid=os.getpid(),waiting_for=['lane1','lane2','lane3'],
        stage=stage,started_at_unix=time.time(),cuda_used=False)
    save(root/'status.json',state)
    try:
        while not readiness(args.migration_root): time.sleep(300)
        env=dict(os.environ);env.update(stage.get('env',{}))
        env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (root/'summary.log').open('x') as stream:
            child=subprocess.Popen(stage['command'],cwd=stage['cwd'],env=env,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',active_pid=child.pid);save(root/'status.json',state)
            code=child.wait()
        if code: raise RuntimeError('summary exit '+str(code))
        require_subset(read(stage['completion']),stage['completion_expect'])
        state.update(status='complete',finished_at_unix=time.time(),active_pid=None)
        save(root/'status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),finished_at_unix=time.time())
        save(root/'status.json',state)
        raise


if __name__=='__main__':main()

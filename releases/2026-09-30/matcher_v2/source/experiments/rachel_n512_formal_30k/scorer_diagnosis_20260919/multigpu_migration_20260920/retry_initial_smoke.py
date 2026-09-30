"""Explicitly retry only a failed first disposable smoke with zero completed steps."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from lane_runner import read, validate_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--lane',required=True)
    p.add_argument('--reason',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    plan=validate_plan(read(args.plan))
    lane=next(x for x in plan['lanes'] if x['name']==args.lane)
    root=Path(plan['output_root'])
    state=read(root/args.lane/'status.json')
    if state['status']!='failed' or state['completed_stages']!=0 or len(state['stages'])!=1:
        raise ValueError('only a failed initial smoke is eligible')
    stage=lane['stages'][0]
    if not stage['name'].endswith('_discard32'):
        raise ValueError('first stage is not disposable smoke')
    for pid in (state['pid'],state['active_pid']):
        try:
            os.kill(pid,0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('a recorded process is still alive')
    command=stage['command']
    output=Path(stage.get('output') or command[command.index('--output')+1])
    protocol=read(output/'protocol.json')
    if protocol.get('status')!='failed' or protocol.get('completed_disposable_steps')!=0:
        raise ValueError('failed smoke must have zero completed steps')
    if protocol.get('formal_pair_exposures')!=0 or protocol.get('formal_optimizer_updates')!=0:
        raise ValueError('formal training may not be retried with this utility')
    archive=root/'failed_initial_smokes'/(args.lane+'_attempt1')
    if archive.exists(): raise ValueError('one explicit retry only')
    command=[sys.executable,str(Path(__file__).with_name('lane_runner.py')),
        '--plan',args.plan,'--lane',args.lane,'--execute']
    if not args.execute:
        print(json.dumps(dict(archive=str(archive),command=command,reason=args.reason)))
        return
    archive.mkdir(parents=True)
    os.rename(root/args.lane,archive/'lane_state')
    os.rename(output,archive/'smoke_output')
    log=root/(args.lane+'_retry1_runner.log')
    with log.open('x') as stream:
        process=subprocess.Popen(command,cwd=Path(__file__).parent,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(reason=args.reason,pid=process.pid,lane=args.lane,command=command,
                 archived_output=str(archive),time_unix=time.time(),formal_progress_replayed=False)
    with (archive/'retry.json').open('x') as handle: json.dump(receipt,handle,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__': main()

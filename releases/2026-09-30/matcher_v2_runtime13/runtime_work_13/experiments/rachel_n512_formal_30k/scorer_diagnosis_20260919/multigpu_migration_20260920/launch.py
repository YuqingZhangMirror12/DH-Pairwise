"""Launch each registered lane once, detached from the SSH connection."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from lane_runner import read, validate_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = validate_plan(read(args.plan))
    if plan['hostname'] != socket.gethostname():
        raise ValueError('plan belongs to a different host')
    root = Path(plan['output_root'])
    path = root/'launch.json'
    commands = [[sys.executable, str(Path(__file__).with_name('lane_runner.py')),
        '--plan', args.plan, '--lane', lane['name'], '--execute'] for lane in plan['lanes']]
    if not args.execute:
        print(json.dumps(commands, indent=2))
        return
    # Claim the launch before spawning. Never double-launch on an uncertain SSH result.
    with path.open('x') as handle:
        json.dump(dict(status='launching', started_at_unix=time.time()), handle)
    records=[]
    for lane, command in zip(plan['lanes'], commands):
        log_path = root/(lane['name']+'_runner.log')
        with log_path.open('x') as stream:
            process = subprocess.Popen(command, cwd=Path(__file__).parent,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
        records.append(dict(lane=lane['name'], gpu_uuid=lane['gpu_uuid'], pid=process.pid,
            command=command, log=str(log_path)))
    with path.open('w') as handle:
        json.dump(dict(status='dispatched', launched_at_unix=time.time(), lanes=records), handle, indent=2)
    print(json.dumps(records, indent=2))


if __name__=='__main__':
    main()

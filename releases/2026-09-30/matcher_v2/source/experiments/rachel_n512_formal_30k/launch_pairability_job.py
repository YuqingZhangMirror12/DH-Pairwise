"""Launch one named experiment with explicit PID, command and exit receipt."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    root = Path(a.root).resolve()
    state_path = root / (a.name + '_state.json')
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        raise ValueError('explicit command required')
    if not a.worker:
        with state_path.open('x') as f:
            json.dump(dict(status='starting', command=command), f)
        with (root / (a.name + '_controller.log')).open('x') as out:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                '--name', a.name, '--worker', '--'] + command, cwd=root / 'source',
                start_new_session=True, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        print(json.dumps(dict(controller_pid=child.pid, state=str(state_path))), flush=True)
        return
    state = dict(status='running', command=command, controller_pid=os.getpid(), started_unix=time.time(),
                 log=str(root / (a.name + '.log')))
    with Path(state['log']).open('x') as out:
        child = subprocess.Popen(command, cwd=root / 'source', stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=subprocess.STDOUT)
        state['pid'] = child.pid
        state_path.write_text(json.dumps(state, indent=2))
        code = child.wait()
    state.update(status='complete' if code == 0 else 'failed', exit_code=code, ended_unix=time.time())
    state_path.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()

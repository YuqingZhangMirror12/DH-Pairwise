"""Hold only unstarted downstream observers for the user's S3/S4/S5 insertion.

Never signals S2, its first-five queue, or any active child. Original configs
and state files are retained verbatim; a separate receipt records the hold.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def command(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return ''


def run(root):
    root = Path(root).resolve()
    receipt = root / 'new_s345_20260914' / 'downstream_hold.json'
    if receipt.exists():
        raise RuntimeError('hold already recorded; inspect receipt rather than repeat signals')
    names = ('after050_inputs020_v4', 'after020_s3_to050', 'continuation_to020')
    observations = []
    for name in names:
        path = root / 'queues' / name
        state = json.loads((path / 'queue_state.json').read_text())
        pid = state['pid']
        cmd = command(pid)
        if str(path / 'config.json') not in cmd:
            raise RuntimeError('expected observer not live: ' + name)
        if state['status'] != 'waiting_for_dependency' or state.get('active_child'):
            raise RuntimeError('observer is not safely waiting: ' + name)
        if any(stage['status'] != 'queued' for stage in state['stages']):
            raise RuntimeError('downstream stage already started: ' + name)
        observations.append(dict(name=name, pid=pid, command=cmd,
                                 config=state['config'], state_before=state))
    first5 = json.loads((root / 'queues/candidates5/queue_state.json').read_text())
    child = first5.get('active_child', {})
    if not child or child['marker'] not in command(child['pid']):
        raise RuntimeError('S2/current first-five child must remain live during insertion')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    record = dict(status='holding', reason='User requested new S3/S4/S5 immediately after S2',
                  protected_first5_pid=first5['pid'], protected_child=child,
                  observers=observations, stopped=[], unix_time=time.time())
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    for item in observations:
        # Recheck each exact handle immediately before the scoped signal.
        state = json.loads((root / 'queues' / item['name'] / 'queue_state.json').read_text())
        if state['status'] != 'waiting_for_dependency' or state.get('active_child'):
            raise RuntimeError('observer advanced during hold: ' + item['name'])
        if command(item['pid']) != item['command']:
            raise RuntimeError('observer handle changed')
        os.kill(item['pid'], signal.SIGTERM)
        record['stopped'].append(item['name'])
        receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    deadline = time.monotonic() + 10
    while any(command(item['pid']) for item in observations):
        if time.monotonic() > deadline:
            raise RuntimeError('observer still live after scoped hold; inspect before proceeding')
        time.sleep(.2)
    if child['marker'] not in command(child['pid']):
        raise RuntimeError('first-five child changed; inspect current queue')
    record.update(status='held_for_user_requested_insertion', protected_child_still_live=True)
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: record[key] for key in ('status', 'stopped', 'protected_child')}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    run(p.parse_args().root)

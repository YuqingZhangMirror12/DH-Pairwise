"""CPU-only size-stratified readout after all fixed data-arm evaluations finish."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

from experiments.rachel_n512_formal_30k.run_pairability_stage1_followup import completed


def reporting_plan(root):
    root = Path(root).resolve()
    dependency = root / 'stage2_state.json'
    commands = []
    for arm in ('original24k', 'matched24k', 'realism60k'):
        for split in ('test', 'real'):
            commands.append((arm + '_' + split, [sys.executable, '-m',
                'experiments.rachel_n512_formal_30k.report_data_arm_area_strata',
                '--root', str(root), '--arm', arm, '--split', split,
                '--output', str(root / 'size_strata/data_arms' / arm / split)]))
    return dependency, commands


def run(root):
    root = Path(root).resolve()
    dependency, commands = reporting_plan(root)
    completed(dependency)
    rows = []
    for name, command in commands:
        row = dict(name=name, command=command, status='running', started_unix=time.time())
        rows.append(row)
        with (root / ('data_area_' + name + '.log')).open('x') as stream:
            child = subprocess.Popen(command, cwd=root / 'source', stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            (root / 'data_area_reports_steps.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
            code = child.wait()
        row.update(status='complete' if code == 0 else 'failed', exit_code=code, ended_unix=time.time())
        (root / 'data_area_reports_steps.json').write_text(json.dumps(rows, indent=2))
        if code:
            raise RuntimeError('data-arm size reporting failed: ' + name)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    run(p.parse_args().root)

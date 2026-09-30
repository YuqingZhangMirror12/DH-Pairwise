"""Build the new-host plan from unchanged experiment leaf specifications."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time

from lane_runner import validate_plan
from legacy_lanes import derive_legacy_lanes
from new_lanes import derive_lanes, architecture_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    d = Path(args.diagnosis_root).resolve()
    out = Path(args.output_root).resolve()
    if out.exists():
        raise ValueError('new migration directory required; never overwrite an active plan')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
        '--format=csv,noheader,nounits'], text=True)
    gpus = []
    for line in inventory.strip().splitlines():
        index, uuid, name, memory = [part.strip() for part in line.split(',')]
        gpus.append(dict(index=int(index), uuid=uuid, name=name, memory_mib=int(memory)))
    gpus.sort(key=lambda x: x['index'])
    if len(gpus) != 7 or [g['index'] for g in gpus] != list(range(7)):
        raise ValueError('expected the user-authorized seven-GPU host')
    legacy = derive_legacy_lanes(d/'followup_v1/launch_plan.json',
        d/'spectral_training_v1/gpu_v1/launch_plan.json',
        d/'spectral_training_v1/cache_v1/status.json')
    fresh = derive_lanes(json.loads((d/'priority_s7_matched_g_v1/launch_plan.json').read_text()),
        json.loads((d/'matcher_followup_v1/launch_plan.json').read_text()))
    titles = ['C2 resumed capacity control', 'S7 direct five scorer inputs',
              'G0 frozen independent feature branch', 'G1 trainable independent feature branch',
              'D1 and D4 continued attention controls', 'Three spectral controls',
              'S7 Matcher M16/M20 and their independent scorers']
    lanes = []
    for index in range(7):
        name = 'lane'+str(index)
        lane = fresh[name] if name in fresh else dict(name=name, stages=legacy[name])
        lane.update(gpu_uuid=gpus[index]['uuid'], gpu_index=index, title=titles[index])
        lanes.append(lane)
    plan = dict(schema='seven-gpu-migration/1', created_at_unix=time.time(),
        hostname=socket.gethostname(), ssh_port=22962, retired_ssh_port=42993,
        output_root=str(out), device_wrapper=str(Path(__file__).with_name('device_runtime.py')),
        inventory=gpus, lanes=lanes, architecture=architecture_summary(),
        policy='Seven independent lanes; original experiment leaf math/budget preserved; SIMVAL selection only')
    validate_plan(plan)
    out.mkdir(parents=True)
    with (out/'plan.json').open('x') as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
    print(json.dumps(dict(plan=str(out/'plan.json'), lanes={l['name']:len(l['stages']) for l in lanes})))


if __name__ == '__main__':
    main()

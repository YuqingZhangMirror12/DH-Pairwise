"""Finite, isolated four-head inference suite; never trains or edits checkpoints."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/rachel_score_design_20260913_001')
MODELS = {
    's4': ROOT / 'new_s345_20260914/s4_cross_attention/training',
    's6': ROOT / 'attention_depth_20260915/s4_cross_attention_depth2/training',
    's6_depth4': ROOT / 'attention_depth_20260915/s4_cross_attention_depth4/training',
    's7': ROOT / 's6_s7_20260915/priority_after_s5/s7_augmented_full24/training',
}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main(args):
    source = Path(args.source_root).resolve(strict=True)
    selection = Path(args.selection_json).resolve(strict=True)
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    status = dict(schema='scorer-heatmap-suite/1', status='running', pid=os.getpid(),
                  started_at_unix=started, models=args.models, source_root=str(source),
                  selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(), stages=[])
    save(output / 'status.json', status)
    lock = (output.parent / 'heatmap-gpu.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        for model in args.models:
            destination = output / model
            command = [sys.executable, '-m',
                'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.heatmap_probe',
                '--training-run', str(MODELS[model]), '--selection', 'fixed_epoch',
                '--selection-json', str(selection), '--output', str(destination),
                '--prepared-cache', '/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared',
                '--ood-prepared', '/root/autodl-tmp/turufan_ood_pairwise_20260912_001/prepared',
                '--dataset', '/root/autodl-tmp/dataset_rachel_pairwise_n512_v1',
                '--translation-gt-json', '/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json',
                '--device', args.device]
            stage = dict(model=model, command=command, status='running', started_at_unix=time.time())
            status['stages'].append(stage)
            with (output / (model + '.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, PYTHONUNBUFFERED='1'))
                stage['pid'] = child.pid
                save(output / 'status.json', status)
                code = child.wait()
            stage.update(returncode=code, finished_at_unix=time.time())
            if code:
                stage['status'] = 'failed'
                raise RuntimeError('%s probe failed with code %s; see its log' % (model, code))
            protocol = json.loads((destination / 'protocol.json').read_text())
            if protocol.get('status') != 'complete' or protocol['completed_count'] != len(json.loads(selection.read_text())):
                raise RuntimeError('%s process exited without complete selected population' % model)
            stage.update(status='complete', completed_count=protocol['completed_count'])
            save(output / 'status.json', status)
        status['status'] = 'complete'
    except Exception as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status.update(finished_at_unix=time.time(), elapsed_s=time.time()-started)
        save(output / 'status.json', status)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True)
    p.add_argument('--selection-json', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())

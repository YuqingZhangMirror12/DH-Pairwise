"""CPU-only targeted tests, source delta and inherited evaluation re-binding.

Old 116 training and full evaluation suites are preserved and reused by hash;
they are not re-labelled as new GPU/CPU runs. Only the naming path and recovery
controller are newly exercised here. No source is edited by this verifier.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unittest

from recovery_contract import REL, REVISION, inventory, read, sha, verify_source_delta


def dump(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def import_check(entry, training, common, joint):
    code = """import importlib.util, json, pathlib, sys
path, training, common, joint = sys.argv[1:]
spec=importlib.util.spec_from_file_location('repair_eval_import',path)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
model,helper=entry.bootstrap(training,common,joint)
assert pathlib.Path(model.__file__).resolve()==(pathlib.Path(training)/entry.RELATIVE/'train.py').resolve()
print(json.dumps({'status':'passed','training_import':str(model.__file__),'helper_import':str(helper.__file__)}))
"""
    result = subprocess.run([sys.executable, '-c', code, str(entry), str(training), str(common), str(joint)],
        cwd=training, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU only; no accidental GPU allocation')
    prepared = Path(args.prepared).resolve(); code = Path(__file__).resolve().parent
    out = prepared / 'training_preparation_v03_remote.json'
    if out.exists(): raise ValueError('preserve prior preparation; no repeat')
    old, new = prepared/'training_source_02', prepared/'training_source_03'
    baseline = prepared/'training_preparation_v02_remote.json'
    old_training = read(baseline)
    if (old_training.get('status') != 'cpu_preparation_passed'
            or inventory(old) != old_training['source_inventory_sha256']):
        raise ValueError('original CPU/source proof changed')
    delta = verify_source_delta(old, new)
    os.environ['JOINT_REPAIR_SOURCE'] = str(new)
    suite = unittest.defaultTestLoader.discover(str(code), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)
    old_eval = prepared/'evaluation_source_02'; destination = prepared/'evaluation_source_03'
    previous = read(old_eval/'preparation.json')
    if (previous['status'] != 'cpu_preparation_passed' or previous.get('errors') or previous.get('failures')
            or previous['implementations']['joint'] != {p.name:sha(p) for p in (old/REL).glob('*.py')}):
        raise ValueError('original evaluation preparation does not bind source02')
    for folder, field in (('threshold_joint_eval_v1', 'adapter_python_sha256'),
                          ('s7_consensus_eval_v14', 'common_python_sha256')):
        if {p.name:sha(p) for p in (old_eval/folder).glob('*.py')} != previous[field]:
            raise ValueError('evaluation implementation changed')
    if destination.exists(): raise ValueError('preserve existing evaluation revision')
    destination.mkdir()
    for folder in ('threshold_joint_eval_v1', 's7_consensus_eval_v14'):
        shutil.copytree(old_eval/folder, destination/folder,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    entry = destination/'threshold_joint_eval_v1/entry.py'
    checks = {kind:import_check(entry, root, destination/'s7_consensus_eval_v14', new)
              for kind,root in (('joint',new),('frozen',Path('/root/autodl-tmp/s7_consensus_threshold_v1_20260925/source')))}
    updated = copy.deepcopy(previous)
    updated['implementations']['joint'] = {p.name:sha(p) for p in (new/REL).glob('*.py')}
    updated['repair_rebinding'] = dict(revision=REVISION, original_receipt_sha256=sha(old_eval/'preparation.json'),
        original_evaluation_cpu_results_reused_without_rerun=True, source_delta=delta['changed'],
        fresh_import_checks=checks, fresh_training_regression_tests=result.testsRun,
        real_inference_performed=False, recorded_unix=time.time())
    dump(destination/'preparation.json', updated)
    record = dict(status='cpu_preparation_passed', revision=REVISION,
        tests=result.testsRun, errors=len(result.errors), failures=len(result.failures),
        old_failure_reproduced=True, actual_run_stage_cpu_update_verified=True,
        test_scope='6 targeted startup/delta tests + 4 recovery routing + 7 inherited gate contracts on repaired controller',
        inherited_training_cpu_tests=old_training['tests'], inherited_training_cpu_tests_rerun=False,
        inherited_preparation_sha256=sha(baseline), source_delta=delta,
        source_inventory_sha256=inventory(new), recovery_python_sha256=inventory(code),
        evaluation_preparation_sha256=sha(destination/'preparation.json'),
        formal_training_started=False, gpu_preflight=False, recorded_unix=time.time())
    dump(out, record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('source_delta','source_inventory_sha256','recovery_python_sha256')}, indent=2))


if __name__ == '__main__': main()

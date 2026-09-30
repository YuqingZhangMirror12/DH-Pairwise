"""Offline CPU verification of this code-only release; never launches a queue."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT/'runtime_work_13'
PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def inventory():
    expected = json.loads((ROOT/'code_binding.json').read_text())
    actual = {str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob('*.py')
              if p != Path(__file__).resolve()}
    if actual != expected:
        raise ValueError('code-only inventory differs from the public binding')
    return actual


def unit_child(suite, output):
    # These two suites have no separate receipt-writing entry point. Run their
    # synthetic fixtures in fresh processes to avoid import-alias contamination.
    definitions = {
        'diagnostics': ('diagnostics_source_03', [
            'test_diagnostic_metrics', 'test_freeze_development_groups',
            'test_mask_geometry', 'test_ridge_runner']),
        'native_budget': ('cal_budget_source_01', ['test_score_native']),
    }
    directory, names = definitions[suite]
    sys.path.insert(0, str(RUNTIME))
    sys.path.insert(0, str(ROOT/directory))
    import torch
    torch.set_num_threads(1)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames(names))
    record = dict(status='passed' if result.wasSuccessful() and not result.skipped else 'failed',
                  tests=result.testsRun, errors=len(result.errors), failures=len(result.failures),
                  skipped=len(result.skipped), cuda_initialized=torch.cuda.is_initialized(),
                  test_log=stream.getvalue())
    save(output, record)
    print(stream.getvalue(), end='')
    return 0 if record['status'] == 'passed' and not record['cuda_initialized'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-new', type=Path, required=True,
                        help='New receipt directory OUTSIDE this source capsule')
    parser.add_argument('--unit-child', choices=('diagnostics', 'native_budget'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' before running offline verification")
    if args.unit_child:
        return unit_child(args.unit_child, args.output_new)
    output = args.output_new.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('keep verification receipts outside the immutable source capsule')
    before = inventory()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(RUNTIME), CURRICULUM_BASELINE_SOURCE=str(RUNTIME))
    python = sys.executable
    # These are TEST entry points only. No dispatch/main, GPU launcher, remote
    # preflight, private data path or real checkpoint is invoked here.
    suites = [
        ('runtime', 194, RUNTIME,
         [python, '-m', PACKAGE+'.matcher_v2_v1.verify_runtime_cpu', '--source-root', str(RUNTIME), '--output-new']),
        ('diagnostics', 36, ROOT,
         [python, str(Path(__file__).resolve()), '--unit-child', 'diagnostics', '--output-new']),
        ('native_budget', 5, ROOT,
         [python, str(Path(__file__).resolve()), '--unit-child', 'native_budget', '--output-new']),
        ('reference_native', 23, ROOT/'reference_select_source_01',
         [python, 'run_cpu_tests.py', '--source-root', str(RUNTIME), '--out']),
        ('reference_scorer', 26, ROOT/'reference_scorer_source_01',
         [python, 'verify_cpu.py', '--source-root', str(RUNTIME),
          '--reference-source', str(ROOT/'reference_select_source_01'), '--out']),
        ('control_queue', 28, ROOT/'control_queue_source_01',
         [python, 'verify_cpu.py', '--source-root', str(RUNTIME), '--out']),
        ('supplemental_queue', 27, ROOT/'supplemental_queue_source_01',
         [python, 'verify_cpu.py', '--artifact-root', str(ROOT), '--out']),
        ('scorer_cal_budget', 38, ROOT/'scorer_cal_budget_source_03',
         [python, 'verify_cpu.py', '--out']),
    ]
    records = []
    began = time.time()
    for name, count, cwd, command in suites:
        path = output/(name+'.json')
        with (output/(name+'.log')).open('x') as log:
            returned = subprocess.run(command+[str(path)], cwd=cwd, env=env,
                                      stdout=log, stderr=subprocess.STDOUT, timeout=900)
        proof = json.loads(path.read_text()) if path.is_file() else {}
        valid = (returned.returncode == 0 and proof.get('status') == 'passed'
                 and proof.get('tests') == count
                 and all(proof.get(k) == 0 for k in ('errors', 'failures', 'skipped'))
                 and not proof.get('cuda_initialized', proof.get('gpu_used', False)))
        records.append(dict(suite=name, expected_tests=count, tests=proof.get('tests'),
                            returncode=returned.returncode, passed=valid,
                            receipt_sha256=sha(path) if path.is_file() else None))
        print(json.dumps(records[-1]), flush=True)
        if not valid:
            save(output/'failure.json', dict(status='failed', suites=records))
            raise RuntimeError('See '+str(output/(name+'.log')))
    unchanged = before == inventory()
    if not unchanged:
        raise ValueError('source changed during CPU verification')
    save(output/'complete.json', dict(schema='matcher-v2-runtime13-public-cpu/1', status='passed',
        tests=sum(row['tests'] for row in records), suites=records, errors=0, failures=0, skipped=0,
        source_unchanged=True, code_binding_sha256=sha(ROOT/'code_binding.json'),
        verifier_sha256=sha(__file__), elapsed_seconds=time.time()-began,
        gpu_used=False, private_inputs_opened=False, formal_training_started=False,
        accuracy_or_convergence_claim=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

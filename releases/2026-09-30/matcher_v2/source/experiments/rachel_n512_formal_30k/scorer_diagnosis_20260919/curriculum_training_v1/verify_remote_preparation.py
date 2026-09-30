"""Verify only new CPU components on an isolated, immutable remote package.

The earlier75 tests are reused by source identity, not rerun. This does not
admit actual data, choose a budget, calibrate the real TRAIN pool, or use CUDA.
"""
import argparse
import io
import json
import os
from pathlib import Path
import time
import unittest

from .checkpoint_io import file_sha
from .exposure import digest
from .verify_validation_preparation import bind_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior-remote', type=Path, required=True)
    parser.add_argument('--local-receipt', type=Path, required=True)
    parser.add_argument('--expected-package-sha256', required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('hide all CUDA devices for CPU preparation')
    prior = json.loads(args.prior_remote.read_text()); local = json.loads(args.local_receipt.read_text())
    if (prior['status'] != 'passed' or prior['tests'] != 27 or prior['unchanged_prior_tests_reused'] != 48
            or local['status'] != 'passed' or local['tests'] != 15 or local['unchanged_prior_tests_reused'] != 199):
        raise ValueError('preceding remote75 and local214 evidence required')
    source = Path(__file__).parent
    before = {p.name:file_sha(p) for p in source.glob('*.py')}
    if digest(before) != args.expected_package_sha256:
        raise ValueError('isolated deployed package differs from local copy')
    for receipt in (prior, local):
        if any(before.get(name) != sha for name, sha in receipt['source_sha256'].items()):
            raise ValueError('previously verified component changed')
    baseline = args.baseline_source.resolve()
    baseline_before = {str(p.relative_to(baseline)):file_sha(p) for p in baseline.rglob('*.py')}
    if baseline_before != local['baseline_python_sha256']:
        raise ValueError('remote immutable baseline differs from local test source')
    bind_baseline(baseline); args.out.mkdir(parents=True, exist_ok=False)
    names = ['test_validation_adapter', 'test_matcher_diagnostics', 'test_data_admission',
             'test_model_adapter', 'test_runtime_io', 'test_execution', 'test_train_geometry']
    suite = unittest.defaultTestLoader.loadTestsFromNames([__package__ + '.' + name for name in names])
    if suite.countTestCases() != 139:
        raise ValueError('unexpected new test set; do not recount earlier75 as new evidence')
    stream = io.StringIO(); start = time.time()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.out / 'tests.log').write_text(stream.getvalue())
    if (before != {p.name:file_sha(p) for p in source.glob('*.py')}
            or baseline_before != {str(p.relative_to(baseline)):file_sha(p) for p in baseline.rglob('*.py')}):
        raise ValueError('source changed during CPU preparation')
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        unchanged_prior_tests_reused=75, distinct_tests_total=214,
        prior_remote_receipt_sha256=file_sha(args.prior_remote), local_receipt_sha256=file_sha(args.local_receipt),
        source_sha256=before, source_manifest_sha256=digest(before), baseline_source=str(baseline),
        baseline_python_sha256=baseline_before, test_log_sha256=file_sha(args.out / 'tests.log'),
        elapsed_seconds=time.time() - start, gpu_used=False, full_data_admission=False,
        full_data_calibration=False, actual_protocol_budget_locked=False, formal_training_started=False,
        training_launcher_ready=False, actual_real_inference=False,
        scope='New139 CPU checks: synthetic receipt/loader/selection fixtures, actual96-width constructors, inherited-edge geometry, tiny optimizer/export and two-process Gloo. No full N512 GPU gate or real performance.')
    (args.out / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({name:receipt[name] for name in ('status', 'tests', 'failures', 'errors',
        'unchanged_prior_tests_reused', 'distinct_tests_total', 'gpu_used', 'formal_training_started')}))
    if not result.wasSuccessful():
        print(stream.getvalue()); raise SystemExit(1)


if __name__ == '__main__':
    main()

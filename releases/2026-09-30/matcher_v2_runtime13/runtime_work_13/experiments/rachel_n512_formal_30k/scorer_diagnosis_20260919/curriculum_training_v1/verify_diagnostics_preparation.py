"""Verify Matcher-only diagnostics against existing bound union feature code."""
import argparse
import io
import json
import os
from pathlib import Path
import time
import unittest

from .checkpoint_io import file_sha
from .verify_validation_preparation import bind_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior-receipt', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('CPU diagnostics preparation must hide GPUs')
    prior = json.loads(args.prior_receipt.read_text()); source = Path(__file__).parent
    if (prior['status'] != 'passed' or prior['tests'] != 24
            or prior['unchanged_prior_tests_reused'] != 75 or prior['gpu_used'] is not False):
        raise ValueError('preceding validation preparation must have passed')
    before = {p.name: file_sha(p) for p in source.glob('*.py')}
    if any(before.get(name) != signature for name, signature in prior['source_sha256'].items()):
        raise ValueError('previously tested source changed; cannot reuse evidence')
    baseline = args.baseline_source.resolve()
    baseline_before = {str(p.relative_to(baseline)): file_sha(p) for p in baseline.rglob('*.py')}
    if baseline_before != prior['baseline_python_sha256']:
        raise ValueError('immutable baseline differs from the prior preparation')
    bind_baseline(baseline); args.out.mkdir(parents=True, exist_ok=False)
    begin = time.time(); stream = io.StringIO()
    tests = unittest.defaultTestLoader.loadTestsFromName(__package__ + '.test_matcher_diagnostics')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    (args.out / 'tests.log').write_text(stream.getvalue())
    if (before != {p.name: file_sha(p) for p in source.glob('*.py')}
            or baseline_before != {str(p.relative_to(baseline)): file_sha(p) for p in baseline.rglob('*.py')}):
        raise ValueError('source changed during diagnostics checks')
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        scope='Scorer-free native union/Q/Q*arc winners, full quantiles and denominators, known/unknown GT, prebudget loss and image edge provenance; synthetic actual threshold union fixtures',
        unchanged_prior_tests_reused=99, prior_receipt_sha256=file_sha(args.prior_receipt),
        prior_components_all_unchanged=True, source_sha256=before,
        baseline_source=str(baseline), baseline_python_sha256=baseline_before, baseline_source_unchanged=True,
        test_log_sha256=file_sha(args.out / 'tests.log'), elapsed_seconds=time.time() - begin,
        gpu_used=False, formal_training_started=False, actual_real_inference=False,
        training_launcher_ready=False, actual_protocol_budget_locked=False, full_data_admission=False)
    (args.out / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'tests', 'failures', 'errors',
        'unchanged_prior_tests_reused', 'gpu_used', 'formal_training_started')}))
    if not result.wasSuccessful():
        print(stream.getvalue()); raise SystemExit(1)


if __name__ == '__main__':
    main()

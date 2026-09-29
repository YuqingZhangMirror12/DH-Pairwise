"""Local/remote reusable check of new model/loader/export adapters, CPU only."""
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
        raise ValueError('CPU preparation must hide GPUs')
    prior = json.loads(args.prior_receipt.read_text()); source = Path(__file__).parent
    if prior['status'] != 'passed' or prior['tests'] != 20 or prior['unchanged_prior_tests_reused'] != 120:
        raise ValueError('preceding admission preparation must pass')
    before = {p.name: file_sha(p) for p in source.glob('*.py')}
    if any(before.get(name) != sha for name, sha in prior['source_sha256'].items()):
        raise ValueError('prior code changed; earlier evidence cannot be reused')
    baseline = args.baseline_source.resolve()
    baseline_before = {str(p.relative_to(baseline)): file_sha(p) for p in baseline.rglob('*.py')}
    if baseline_before != prior['baseline_python_sha256']:
        raise ValueError('immutable baseline changed')
    bind_baseline(baseline); args.out.mkdir(parents=True, exist_ok=False)
    stream = io.StringIO(); begin = time.time()
    tests = unittest.defaultTestLoader.loadTestsFromNames([
        __package__ + '.test_runtime_io', __package__ + '.test_model_adapter'])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    (args.out / 'tests.log').write_text(stream.getvalue())
    if (before != {p.name: file_sha(p) for p in source.glob('*.py')}
            or baseline_before != {str(p.relative_to(baseline)): file_sha(p) for p in baseline.rglob('*.py')}):
        raise ValueError('source changed during model/export checks')
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        scope='Tiny CPU optimizer -> committed checkpoints -> selected tensor export; immutable validation attempts; actual96-width model initialization/freeze and synthetic admitted loader fixtures',
        unchanged_prior_tests_reused=140, prior_receipt_sha256=file_sha(args.prior_receipt),
        prior_components_all_unchanged=True, source_sha256=before,
        baseline_source=str(baseline), baseline_python_sha256=baseline_before,
        test_log_sha256=file_sha(args.out / 'tests.log'), elapsed_seconds=time.time() - begin,
        gpu_used=False, full_N512_training=False, formal_training_started=False,
        actual_protocol_budget_locked=False, training_launcher_ready=False,
        actual_real_inference=False, full_data_admission=False)
    (args.out / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'tests', 'failures', 'errors',
        'unchanged_prior_tests_reused', 'gpu_used', 'formal_training_started')}))
    if not result.wasSuccessful():
        print(stream.getvalue()); raise SystemExit(1)


if __name__ == '__main__':
    main()

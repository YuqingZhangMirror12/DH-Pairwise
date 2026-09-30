"""Local CPU checks of shared TRAIN calibration; no actual full calibration."""
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
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('CPU preparation must hide GPUs')
    source = Path(__file__).parent; prior = json.loads(args.prior_receipt.read_text())
    if prior['status'] != 'passed' or prior['tests'] != 27 or prior['unchanged_prior_tests_reused'] != 172:
        raise ValueError('preceding execution preparation must pass')
    before = {p.name:file_sha(p) for p in source.glob('*.py')}
    if any(before.get(name) != sha for name, sha in prior['source_sha256'].items()):
        raise ValueError('prior components changed; cannot reuse tests')
    baseline = Path(prior['baseline_source'])
    if prior['baseline_python_sha256'] != {str(p.relative_to(baseline)):file_sha(p) for p in baseline.rglob('*.py')}:
        raise ValueError('immutable baseline changed')
    bind_baseline(baseline); args.out.mkdir(parents=True, exist_ok=False); stream = io.StringIO(); start = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromName(__package__ + '.test_train_geometry')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.out / 'tests.log').write_text(stream.getvalue())
    if before != {p.name:file_sha(p) for p in source.glob('*.py')}:
        raise ValueError('source changed during geometry checks')
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        unchanged_prior_tests_reused=199, prior_receipt_sha256=file_sha(args.prior_receipt),
        source_sha256=before, baseline_source=str(baseline), baseline_python_sha256=prior['baseline_python_sha256'],
        scope='Synthetic TRAIN catalog/provenance, actual inherited-edge geometry, unchanged combined-pool formulas and output bindings',
        test_log_sha256=file_sha(args.out / 'tests.log'), elapsed_seconds=time.time() - start,
        gpu_used=False, full_data_calibration=False, formal_training_started=False, actual_protocol_budget_locked=False,
        training_launcher_ready=False, full_data_admission=False, actual_real_inference=False)
    (args.out / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key:receipt[key] for key in ('status', 'tests', 'failures', 'errors',
        'unchanged_prior_tests_reused', 'gpu_used', 'formal_training_started')}))
    if not result.wasSuccessful():
        print(stream.getvalue()); raise SystemExit(1)


if __name__ == '__main__':
    main()

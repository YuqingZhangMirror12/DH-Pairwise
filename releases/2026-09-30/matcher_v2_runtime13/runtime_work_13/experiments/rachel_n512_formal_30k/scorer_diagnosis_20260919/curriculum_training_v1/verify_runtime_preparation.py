"""Verify only new checkpoint/plan components, reusing unchanged prior evidence."""
import argparse
import io
import json
import os
from pathlib import Path
import time
import unittest

from .checkpoint_io import file_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior-receipt', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('hide GPUs for CPU-only preparation')
    prior = json.loads(args.prior_receipt.read_text()); source = Path(__file__).parent
    if prior['status'] != 'passed' or prior['tests'] != 48 or prior['gpu_used'] is not False:
        raise ValueError('expected completed earlier 48-test CPU preparation')
    before = {p.name: file_sha(p) for p in source.glob('*.py')}
    if any(before.get(name) != signature for name, signature in prior['source_sha256'].items()):
        raise ValueError('prior tested component changed; cannot reuse unchanged-test evidence')
    args.out.mkdir(parents=True, exist_ok=False); begin = time.time(); stream = io.StringIO()
    tests = unittest.defaultTestLoader.loadTestsFromNames([
        __package__ + '.test_checkpoint_io', __package__ + '.test_runtime_plan'])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    (args.out / 'tests.log').write_text(stream.getvalue())
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        scope='new complete-rank disk checkpoints, disk optimizer/RNG resume, explicit common runtime schedule and paired bindings; synthetic CPU fixtures',
        unchanged_prior_tests_reused=prior['tests'], prior_receipt_sha256=file_sha(args.prior_receipt),
        prior_components_all_unchanged=True, source_sha256=before,
        test_log_sha256=file_sha(args.out / 'tests.log'), elapsed_seconds=time.time() - begin,
        gpu_used=False, formal_training_started=False, training_launcher_ready=False,
        actual_protocol_budget_locked=False, full_data_admission=False)
    if before != {p.name: file_sha(p) for p in source.glob('*.py')}:
        raise ValueError('preparation source changed while checking')
    (args.out / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'tests', 'failures', 'errors',
          'unchanged_prior_tests_reused', 'gpu_used', 'formal_training_started')}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()

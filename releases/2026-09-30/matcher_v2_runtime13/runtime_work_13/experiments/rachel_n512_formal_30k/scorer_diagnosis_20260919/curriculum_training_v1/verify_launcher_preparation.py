"""Only new controller CPU checks; does not acquire GPUs or start experiments."""
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
        raise ValueError('CPU preparation must hide GPUs')
    source = Path(__file__).parent; prior = json.loads(args.prior_receipt.read_text())
    if prior['status'] != 'passed' or prior['tests'] != 139 or prior['distinct_tests_total'] != 214:
        raise ValueError('preceding214 CPU evidence required')
    before = {p.name:file_sha(p) for p in source.glob('*.py')}
    if any(before.get(name) != sha for name, sha in prior['source_sha256'].items()):
        raise ValueError('earlier components changed; cannot reuse evidence')
    args.out.mkdir(parents=True, exist_ok=False); stream = io.StringIO(); began = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromName(__package__ + '.test_launcher')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.out / 'tests.log').write_text(stream.getvalue())
    if before != {p.name:file_sha(p) for p in source.glob('*.py')}:
        raise ValueError('controller source changed during checks')
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        unchanged_prior_tests_reused=214, prior_receipt_sha256=file_sha(args.prior_receipt),
        source_sha256=before, baseline_python_sha256=prior['baseline_python_sha256'],
        test_log_sha256=file_sha(args.out / 'tests.log'), elapsed_seconds=time.time()-began,
        scope='CPU command/assignment protocol, actual short CPU child return, mocked gate sequence and tiny actual optimizer/selected-export verification',
        gpu_used=False, genuine_gpu_gate=False, formal_training_started=False,
        actual_protocol_budget_locked=False, queue_registered=False, actual_full_training=False,
        training_launcher_ready=False)
    (args.out / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key:receipt[key] for key in ('status', 'tests', 'failures', 'errors', 'unchanged_prior_tests_reused', 'gpu_used')}))
    if not result.wasSuccessful():
        print(stream.getvalue()); raise SystemExit(1)


if __name__ == '__main__':
    main()

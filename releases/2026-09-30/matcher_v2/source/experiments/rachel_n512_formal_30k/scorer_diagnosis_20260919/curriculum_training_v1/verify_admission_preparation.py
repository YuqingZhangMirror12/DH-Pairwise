"""Check new admission code; preserve and reuse unchanged earlier evidence."""
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
    prior = json.loads(args.prior_receipt.read_text()); source = Path(__file__).parent
    if (prior['status'] != 'passed' or prior['tests'] != 21
            or prior['unchanged_prior_tests_reused'] != 99 or prior['gpu_used'] is not False):
        raise ValueError('preceding diagnostics preparation must have passed')
    before = {p.name: file_sha(p) for p in source.glob('*.py')}
    if any(before.get(name) != signature for name, signature in prior['source_sha256'].items()):
        raise ValueError('prior source changed; do not reuse earlier evidence')
    baseline = Path(prior['baseline_source'])
    actual = {str(p.relative_to(baseline)): file_sha(p) for p in baseline.rglob('*.py')}
    if actual != prior['baseline_python_sha256']:
        raise ValueError('bound baseline source changed')
    args.out.mkdir(parents=True, exist_ok=False)
    begin = time.time(); stream = io.StringIO()
    tests = unittest.defaultTestLoader.loadTestsFromName(__package__ + '.test_data_admission')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    (args.out / 'tests.log').write_text(stream.getvalue())
    if before != {p.name: file_sha(p) for p in source.glob('*.py')}:
        raise ValueError('source changed during preparation')
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        scope='Synthetic disk final-release/group-audit/reference chain, actual input vs effective supervision dedup, immutable source accounting; not full-data admission',
        unchanged_prior_tests_reused=120, prior_receipt_sha256=file_sha(args.prior_receipt),
        prior_components_all_unchanged=True, source_sha256=before,
        baseline_source=str(baseline), baseline_python_sha256=actual,
        test_log_sha256=file_sha(args.out / 'tests.log'), elapsed_seconds=time.time() - begin,
        gpu_used=False, formal_training_started=False, full_data_admission=False,
        actual_protocol_budget_locked=False, training_launcher_ready=False)
    (args.out / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'tests', 'failures', 'errors',
        'unchanged_prior_tests_reused', 'gpu_used', 'formal_training_started')}))
    if not result.wasSuccessful():
        print(stream.getvalue()); raise SystemExit(1)


if __name__ == '__main__':
    main()

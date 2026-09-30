"""Small, deterministic CPU tests; no model loading or new predictions."""
import argparse
import io
import os
from pathlib import Path
import time
import unittest

import score_scorer as adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    adapter.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not args.out.exists(), 'CPU-only fresh receipt required')
    source = Path(__file__).parent; dependency = source.parent/'diagnostics_source_03/diagnostic_metrics.py'
    adapter.require(adapter.sha(dependency) == adapter.METRIC_SHA, 'bound metric code changed')
    before = adapter.inventory(source); started = time.time(); stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName('test_scorer_budgets')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    same = before == adapter.inventory(source)
    receipt = dict(status='passed' if result.wasSuccessful() and not result.skipped and same else 'failed',
        tests=result.testsRun, errors=len(result.errors), failures=len(result.failures), skipped=len(result.skipped),
        source_unchanged=same, source_sha256=before, metric_sha256=adapter.sha(dependency),
        gpu_used=False, trained_weights_or_real_images_opened=False, inference_performed=False,
        elapsed_seconds=time.time()-started, test_log=stream.getvalue())
    adapter.save(args.out, receipt); print(stream.getvalue(), end='')
    adapter.require(receipt['status'] == 'passed', 'Scorer budget tests failed')


if __name__ == '__main__':main()

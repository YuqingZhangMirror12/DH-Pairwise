"""CPU fixture tests for independent B1/B2 resource and lifecycle routing."""
import argparse
import io
import os
from pathlib import Path
import sys
import time
import unittest

import dispatch_controls as queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-runtime', action='store_true', help='Remote-only real imports and bound-plan checks; no GPU query')
    args = parser.parse_args()
    queue.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only fixtures required')
    queue.require(not args.out.exists(), 'new immutable CPU receipt required')
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch
    torch.set_num_threads(1)
    before = queue.own_code(); started = time.time(); log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName('test_dispatch_controls')
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    preflight = None
    if result.wasSuccessful() and args.verify_runtime:
        queue.check_bindings()
        preflight = queue.runtime_preflight(queue.load_modules())
    changed = before != queue.own_code()
    receipt = dict(status='passed' if result.wasSuccessful() and not changed else 'failed', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_sha256=before, source_unchanged=not changed, cuda_initialized=torch.cuda.is_initialized(),
        runtime_preflight=preflight,
        elapsed_seconds=time.time()-started, test_log=log.getvalue(),
        scope='controller/resource/model-origin synthetic fixtures; no GPU checks, child training, or claimed accuracy')
    queue.save(args.out, receipt); print(log.getvalue(), end='')
    queue.require(receipt['status'] == 'passed' and not receipt['cuda_initialized'], 'CPU control queue tests failed')


if __name__ == '__main__':
    main()

"""Reproducible isolated CPU companion tests; never a training/GPU gate."""
import argparse
import io
import os
from pathlib import Path
import sys
import time
import unittest

from admit_reference_select import require, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only tests required')
    require(not args.out.exists(), 'new CPU receipt path required')
    root = Path(__file__).parent
    before = {p.name: sha(p) for p in root.glob('*.py')}
    sys.path.insert(0, str(args.source_root.resolve()))
    begin = time.time(); stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName('test_reference_select')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    import torch
    unchanged = before == {p.name: sha(p) for p in root.glob('*.py')}
    receipt = dict(status='passed' if result.wasSuccessful() and unchanged else 'failed', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_sha256=before, source_unchanged=unchanged, cuda_initialized=torch.cuda.is_initialized(),
        elapsed_seconds=time.time()-begin, test_log=stream.getvalue(),
        scope='synthetic fixtures plus actual archive loader; no external samples, model training, or terminal-result claim')
    save(args.out, receipt)
    print(stream.getvalue(), end='')
    require(receipt['status'] == 'passed' and not receipt['cuda_initialized'], 'CPU tests failed')


if __name__ == '__main__':
    main()

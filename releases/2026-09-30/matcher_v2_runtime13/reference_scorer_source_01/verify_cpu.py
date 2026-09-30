"""Synthetic CPU checks of supplemental frozen-head evaluation; no data/weights."""
import argparse
import io
import os
from pathlib import Path
import sys
import time
import unittest

import evaluate_reference_scorer as adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'reference-source', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    adapter.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only tests required')
    adapter.require(not args.out.exists(), 'exclusive preparation receipt required')
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch
    torch.set_num_threads(1)
    readout = args.source_root/adapter.PACKAGE.replace('.', '/')/'curriculum_scorer_eval_v1'
    adapter.verify_code(readout, adapter.READOUT_CODE)
    adapter.verify_code(args.reference_source, adapter.REFERENCE_CODE)
    def inventory():
        return {p.name:adapter.sha(p) for p in Path(__file__).parent.glob('*.py')}
    before = inventory(); began = time.time(); stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName('test_reference_scorer')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    same = before == inventory()
    adapter.verify_code(readout, adapter.READOUT_CODE)
    adapter.verify_code(args.reference_source, adapter.REFERENCE_CODE)
    receipt = dict(status='passed' if result.wasSuccessful() and same and not result.skipped else 'failed',
        tests=result.testsRun, errors=len(result.errors), failures=len(result.failures), skipped=len(result.skipped),
        source_unchanged=same, source_sha256=before, reference_code_sha256=adapter.REFERENCE_CODE,
        readout_code_sha256=adapter.READOUT_CODE, cuda_initialized=torch.cuda.is_initialized(),
        actual_patch_stats_readout_and_numeric_artifact_audit_tested=True,
        elapsed_seconds=time.time()-began, test_log=stream.getvalue(),
        trained_weights_or_real_images_opened=False, formal_training_started=False)
    adapter.save(args.out, receipt); print(stream.getvalue(), end='')
    adapter.require(receipt['status']=='passed' and not receipt['cuda_initialized'], 'supplemental Scorer CPU tests failed')


if __name__ == '__main__':
    main()

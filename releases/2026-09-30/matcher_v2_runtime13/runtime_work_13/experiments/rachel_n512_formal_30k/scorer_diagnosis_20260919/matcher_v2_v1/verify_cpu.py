"""Verify an immutable source snapshot and save a non-GPU preparation receipt."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True, type=Path)
    parser.add_argument('--output-new', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU verification requires CUDA_VISIBLE_DEVICES empty before startup')
    if args.output_new.exists():
        raise FileExistsError(args.output_new)
    source = args.binding.resolve().parent
    raw = args.binding.read_bytes()
    files = json.loads(raw)
    for relative, expected in files.items():
        path = (source / relative).resolve(strict=True)
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('source binding mismatch: ' + relative)
    import torch
    import numpy
    torch.set_num_threads(1)
    package = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_v2_v1'
    names = [package + '.' + n for n in ('test_network', 'test_additive_exposure',
                                        'test_source_audit', 'test_seam_supervision')]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    log = io.StringIO()
    start = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    record = dict(schema='matcher-v2-cpu-preparation/1', status='passed' if result.wasSuccessful() else 'failed',
                  tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), elapsed_seconds=time.monotonic() - start,
                  source_root=str(source), binding_sha256=hashlib.sha256(raw).hexdigest(),
                  source_files=len(files), python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__,
                  cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], cuda_initialized=torch.cuda.is_initialized(),
                  gpu_gate_passed=False, formal_training_started=False, stdout=log.getvalue())
    args.output_new.parent.mkdir(parents=True, exist_ok=True)
    with args.output_new.open('x') as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'stdout'}, sort_keys=True))
    if not result.wasSuccessful():
        print(log.getvalue())
    return 0 if result.wasSuccessful() else 2


if __name__ == '__main__':
    raise SystemExit(main())

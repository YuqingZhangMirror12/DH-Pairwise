"""Bind all adapter sources only after complete CPU guard/equivalence tests."""
import argparse
import os
from pathlib import Path
import sys
import unittest

from common import source_map, save, require


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    require(not args.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='','new CPU-only preparation required')
    before=source_map()
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_repair.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    import torch
    ok=result.wasSuccessful() and not result.skipped and not torch.cuda.is_initialized() and before==source_map()
    receipt=dict(schema='matcher-v2-terminal-repair-cpu/1',status='passed' if ok else 'failed',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        cuda_initialized=torch.cuda.is_initialized(),source_files=before,source_files_unchanged=before==source_map())
    save(args.out,receipt)
    if not ok:sys.exit(1)


if __name__=='__main__':main()

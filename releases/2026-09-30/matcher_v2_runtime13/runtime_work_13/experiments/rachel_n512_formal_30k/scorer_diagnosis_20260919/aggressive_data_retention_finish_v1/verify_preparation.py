import argparse
import os
from pathlib import Path
import unittest
from recover import inventory,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU fixture tests only')
    out=Path(args.out)
    if out.exists():raise ValueError('preserve prior test receipt')
    before=inventory();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_finish'))
    same=before==inventory();value=dict(status='passed' if result.wasSuccessful() and same else 'failed',tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),source_sha256=before,source_unchanged=same,gpu_used=False)
    save(out,value)
    if value['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()

"""CPU fixture verification only; no SSH, real inputs, checkpoints, or GPUs."""
import argparse
import os
from pathlib import Path
import unittest
from contracts import inventory,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only fixture test')
    out=Path(a.out)
    if out.exists():raise ValueError('preserve previous receipt')
    before=inventory(Path(__file__).parent)
    suite=unittest.defaultTestLoader.loadTestsFromName('test_queue')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    unchanged=before==inventory(Path(__file__).parent)
    record=dict(status='passed' if result.wasSuccessful() and unchanged else 'failed',tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),source_sha256=before,
        source_unchanged=unchanged,gpu_tasks_started=False,real_inference_performed=False,real_checkpoints_opened=False)
    save(out,record)
    if record['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()

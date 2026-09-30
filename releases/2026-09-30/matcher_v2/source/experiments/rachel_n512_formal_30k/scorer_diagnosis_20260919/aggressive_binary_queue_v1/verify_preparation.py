"""CPU-only queue adapter checks; no remote or training process is started."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('preserve earlier verification')
    def inventory():return {x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')}
    before=inventory();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_queued'))
    same=inventory()==before
    record=dict(status='passed' if same and r.wasSuccessful() else 'failed',tests=r.testsRun,
        errors=len(r.errors),failures=len(r.failures),skipped=len(r.skipped),source_sha256=before,
        source_unchanged=same,gpu_tasks_started=False,real_inference_performed=False)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2)+'\n')
    if record['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()

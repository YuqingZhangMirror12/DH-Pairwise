import argparse
import io
import json
from pathlib import Path
import unittest
from launch_priority import sha, save

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
log=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('test_launcher')
r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
save(a.out,dict(status='passed' if r.wasSuccessful() else 'failed',tests=r.testsRun,errors=len(r.errors),
    failures=len(r.failures),skipped=len(r.skipped),sha256=sha(Path(__file__).with_name('launch_priority.py')),
    pause_source_sha256=sha(Path(__file__).with_name('pause_joint.py')),gpu_used=False,tests_log=log.getvalue()))
print(json.dumps(dict(status='passed' if r.wasSuccessful() else 'failed',tests=r.testsRun)))
if not r.wasSuccessful():print(log.getvalue());raise SystemExit(1)

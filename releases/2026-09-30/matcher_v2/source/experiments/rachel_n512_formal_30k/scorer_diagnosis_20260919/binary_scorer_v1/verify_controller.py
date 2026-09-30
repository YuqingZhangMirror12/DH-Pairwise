"""CPU-only mocked controller checks, saved separately from bound training code."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import unittest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only verification required')
    output=Path(a.out)
    if output.exists():raise ValueError('preserve previous CPU receipt')
    here=Path(__file__).resolve().parent
    def hashes():return {name:hashlib.sha256((here/name).read_bytes()).hexdigest()
                         for name in ('launch_training.py','test_launch_training.py','verify_controller.py')}
    before=hashes();spec=importlib.util.spec_from_file_location('binary_launcher_test',here/'test_launch_training.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    unchanged=hashes()==before
    record=dict(schema='binary-controller-preparation/1',status='passed' if result.wasSuccessful() and unchanged else 'failed',
        tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),
        source_sha256=before,source_files_unchanged=unchanged,all_process_and_gpu_queries_mocked=True,
        gpu_gate_started=False,formal_training_started=False,time_unix=time.time())
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('status','tests','errors','failures')}))
    if record['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()

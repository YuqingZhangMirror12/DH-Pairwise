"""Test only new scheduling/verification glue; do not rerun existing evaluations."""
import argparse
import io
import os
from pathlib import Path
import sys
import time
import unittest

import contracts as c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--remote-preflight',action='store_true')
    args=parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only preparation required')
    c.require(not args.out.exists(),'exclusive new CPU proof required')
    sys.path.insert(0,str(args.artifact_root/'runtime_work_13'))
    import torch
    torch.set_num_threads(1)
    before=c.own_code();deps=c.dependency_code(args.artifact_root);stream=io.StringIO();began=time.time()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_queue'))
    preflight=dict(status='not_requested',gpu_probed=False)
    if result.wasSuccessful() and args.remote_preflight:
        c.require(args.artifact_root.resolve()==c.ROOT,'registered remote artifact root required')
        c.check_static();runtime=c.load_runtime()
        preflight=dict(status='passed',gpu_probed=False,
            runtime_imports={k:str(Path(v.__file__).resolve()) for k,v in runtime.items()},
            source_binding=c.bound(c.SOURCE/'source_binding.json'),runtime_preparation=c.bound(c.PREPARATION),
            reference_admission=c.bound(c.ADMISSION/'complete.json'),strata=c.bound(c.STRATA),
            matcher_executions={arm:c.bound(c.paths(arm,'matcher')[0]) for arm in c.ARMS},
            model_loaded=False,training_or_gpu_status_polled=False)
    same=before==c.own_code() and deps==c.dependency_code(args.artifact_root)
    proof=dict(status='passed' if result.wasSuccessful() and not result.skipped and same else 'failed',
        tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),
        source_unchanged=same,source_sha256=before,dependencies_sha256=deps,
        cuda_initialized=torch.cuda.is_initialized(),runtime_preflight=preflight,
        actual_success_failure_child_returns_tested=True,elapsed_seconds=time.time()-began,test_log=stream.getvalue())
    c.save(args.out,proof);print(stream.getvalue(),end='')
    c.require(proof['status']=='passed' and not proof['cuda_initialized'],'new CPU queue tests failed')


if __name__=='__main__':main()

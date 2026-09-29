"""CPU-only comparison tests; retain earlier291 checks by exact source SHA."""
import argparse
import io
import json
import os
from pathlib import Path
import time
import unittest

from .checkpoint_io import file_sha
from .verify_validation_preparation import bind_baseline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline-source','prior-receipt','out'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('CPU preparation must hide GPUs')
    prior=json.loads(args.prior_receipt.read_text());source=Path(__file__).parent
    if prior['status']!='passed' or prior['tests']!=28 or prior['distinct_tests_total']!=291:
        raise ValueError('prior291 CPU coverage required')
    before={p.name:file_sha(p) for p in source.glob('*.py')}
    if any(before.get(name)!=sha for name,sha in prior['source_sha256'].items()):
        raise ValueError('prior tested source changed')
    baseline=args.baseline_source.resolve();base_sha={str(p.relative_to(baseline)):file_sha(p) for p in baseline.rglob('*.py')}
    if base_sha!=prior['baseline_python_sha256']:raise ValueError('immutable baseline changed')
    bind_baseline(baseline);args.out.mkdir(parents=True,exist_ok=False)
    started=time.time();stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromName(__package__+'.test_matcher_comparison')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (args.out/'tests.log').write_text(stream.getvalue())
    if before!={p.name:file_sha(p) for p in source.glob('*.py')} or base_sha!={str(p.relative_to(baseline)):file_sha(p) for p in baseline.rglob('*.py')}:
        raise ValueError('source changed during tests')
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        unchanged_prior_tests_reused=291,distinct_tests_total=291+result.testsRun,
        prior_receipt_sha256=file_sha(args.prior_receipt),source_sha256=before,baseline_python_sha256=base_sha,
        test_log_sha256=file_sha(args.out/'tests.log'),elapsed_seconds=time.time()-started,
        scope='synthetic completed controller/process-return/file integrity/population/paired-comparison tests; no heldout neural inference',
        population_runner_tested=True,population_controller_tested=True,population_comparison_tested=True,
        gpu_used=False,genuine_gpu_gate=False,formal_training_started=False,queue_registered=False,
        full_population_inference_completed=False,actual_protocol_budget_locked=False,full_data_admission=False)
    (args.out/'preparation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','tests','failures','errors','distinct_tests_total','gpu_used')}))
    if not result.wasSuccessful():print(stream.getvalue());raise SystemExit(1)


if __name__=='__main__':main()

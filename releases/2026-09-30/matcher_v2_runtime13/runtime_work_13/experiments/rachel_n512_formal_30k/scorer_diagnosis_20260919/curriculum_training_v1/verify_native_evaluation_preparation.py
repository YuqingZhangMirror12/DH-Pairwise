"""Test new native evaluation interface plus its changed diagnostic dependency.

Runs only CPU synthetic inputs/exports. This is not evidence of full TEST or
real inference, a deployed evaluator/controller, admitted data or a GPU gate.
"""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-source',type=Path,required=True)
    parser.add_argument('--prior-receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):
        raise ValueError('explicitly hide GPUs for CPU preparation')
    prior = json.loads(args.prior_receipt.read_text()); source = Path(__file__).parent
    if prior['status']!='passed' or prior['tests']!=15 or prior['unchanged_prior_tests_reused']!=214:
        raise ValueError('preceding229 CPU preparation required')
    before = {p.name:file_sha(p) for p in source.glob('*.py')}
    changed = {name for name,sha in prior['source_sha256'].items() if before.get(name)!=sha}
    if changed != {'matcher_diagnostics.py'}:
        raise ValueError('expected only the unlabelled diagnostic extension; otherwise audit test reuse')
    baseline = args.baseline_source.resolve()
    base_sha = {str(p.relative_to(baseline)):file_sha(p) for p in baseline.rglob('*.py')}
    if base_sha != prior['baseline_python_sha256']:
        raise ValueError('immutable baseline changed')
    bind_baseline(baseline); args.out.mkdir(parents=True,exist_ok=False)
    start = time.time(); stream = io.StringIO()
    names = ['test_matcher_diagnostics','test_matcher_evaluation']
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(__package__+'.'+name) for name in names)
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (args.out/'tests.log').write_text(stream.getvalue())
    if before!={p.name:file_sha(p) for p in source.glob('*.py')} or base_sha!={str(p.relative_to(baseline)):file_sha(p) for p in baseline.rglob('*.py')}:
        raise ValueError('source changed while checking')
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        prior_receipt_sha256=file_sha(args.prior_receipt),changed_prior_components=sorted(changed),
        unchanged_prior_tests_reused=208,affected_prior_tests_rerun=21,
        distinct_tests_total=208+result.testsRun,source_sha256=before,baseline_python_sha256=base_sha,
        test_log_sha256=file_sha(args.out/'tests.log'),elapsed_seconds=time.time()-start,
        scope='actual small-canvas width96 Matcher/one Sinkhorn, native union fixtures, posthoc GT and paired-order comparison, actual tiny CPU optimizer exports with synthetic controller receipts; no heldout inference',
        gpu_used=False,genuine_gpu_gate=False,formal_training_started=False,
        actual_protocol_budget_locked=False,full_data_admission=False,
        evaluation_interfaces_tested=True,evaluation_controller_ready=False,
        full_population_inference_completed=False,training_launcher_ready=False)
    (args.out/'preparation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','tests','failures','errors','distinct_tests_total','gpu_used')}))
    if not result.wasSuccessful():
        print(stream.getvalue());raise SystemExit(1)


if __name__=='__main__':
    main()

"""Check new validation/selection integration without rerunning unchanged tests."""
import argparse
import importlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

from .checkpoint_io import file_sha


def bind_baseline(root):
    root = root.resolve()
    parent = importlib.import_module(__package__.rsplit('.', 1)[0])
    relative = Path(*parent.__name__.split('.'))
    if not (root / relative / 's7_consensus_v1/evaluation.py').is_file():
        raise ValueError('complete immutable baseline source required')
    if any(name == 'staging' or name.startswith('staging.') or
           name.startswith(parent.__name__ + '.s7_consensus_v1') for name in sys.modules):
        raise ValueError('run in a fresh process before baseline modules are imported')
    # The isolated curriculum distribution contains only this nested package.
    # The architecture loader also imports siblings such as
    # experiments.rachel_n512_formal_30k.train_score_decoupled. Extending only
    # scorer_diagnosis_20260919 works in a full checkout but fails in deployment.
    # Extend every existing ancestor namespace; retain the already-imported
    # curriculum package so its bound implementation is never substituted.
    names = parent.__name__.split('.')
    for depth in range(1, len(names) + 1):
        package = importlib.import_module('.'.join(names[:depth]))
        location = str(root.joinpath(*names[:depth]))
        package.__path__ = [location, *[p for p in package.__path__ if p != location]]
    spec = importlib.machinery.ModuleSpec('staging', loader=None, is_package=True)
    module = importlib.util.module_from_spec(spec); module.__path__ = [str(root / 'staging')]
    sys.modules['staging'] = module
    os.environ['CURRICULUM_BASELINE_SOURCE'] = str(root)


def reference_architecture(path):
    """Read the original architecture without leaking legacy sys.path edits.

    The historic checkpoint loader prepends its checkout to sys.path. Keeping
    that side effect would hide this isolated curriculum package in spawned
    workers. No reference weights are returned or used for initialization.
    """
    original = list(sys.path)
    try:
        api = importlib.import_module(__package__.rsplit('.', 1)[0] + '.s7_consensus_v1.matcher')
        reference = api.S7MatcherAdapter.from_s7_m12(str(path))
        return reference.base.config
    finally:
        sys.path[:] = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior-receipt', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('explicitly hide GPUs for this CPU-only preparation')
    source = Path(__file__).parent
    prior = json.loads(args.prior_receipt.read_text())
    if (prior['status'] != 'passed' or prior['tests'] != 27
            or prior['unchanged_prior_tests_reused'] != 48 or prior['gpu_used'] is not False):
        raise ValueError('completed preceding 27+48 CPU preparation required')
    before = {p.name: file_sha(p) for p in source.glob('*.py')}
    if any(before.get(name) != signature for name, signature in prior['source_sha256'].items()):
        raise ValueError('previously tested component changed; old evidence cannot be reused')
    baseline = args.baseline_source.resolve()
    baseline_before = {str(p.relative_to(baseline)): file_sha(p) for p in baseline.rglob('*.py')}
    bind_baseline(baseline)
    args.out.mkdir(parents=True, exist_ok=False)
    begin = time.time(); stream = io.StringIO()
    tests = unittest.defaultTestLoader.loadTestsFromName(__package__ + '.test_validation_adapter')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    (args.out / 'tests.log').write_text(stream.getvalue())
    if (before != {p.name: file_sha(p) for p in source.glob('*.py')}
            or baseline_before != {str(p.relative_to(baseline)): file_sha(p) for p in baseline.rglob('*.py')}):
        raise ValueError('preparation or immutable baseline source changed while testing')
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        scope='locked-update validation/selection, canonical rank reports, CPU optimizer/RNG resume, actual baseline metric functions on synthetic rows; evaluator dispatch mocked, no neural inference',
        unchanged_prior_tests_reused=75, prior_receipt_sha256=file_sha(args.prior_receipt),
        prior_components_all_unchanged=True, source_sha256=before, baseline_source=str(baseline),
        baseline_python_sha256=baseline_before, baseline_source_unchanged=True,
        test_log_sha256=file_sha(args.out / 'tests.log'), elapsed_seconds=time.time() - begin,
        gpu_used=False, distributed_gpu_test=False, formal_training_started=False,
        training_launcher_ready=False, actual_protocol_budget_locked=False, full_data_admission=False)
    (args.out / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'tests', 'failures', 'errors',
        'unchanged_prior_tests_reused', 'gpu_used', 'formal_training_started')}))
    if not result.wasSuccessful():
        print(stream.getvalue()); raise SystemExit(1)


if __name__ == '__main__':
    main()

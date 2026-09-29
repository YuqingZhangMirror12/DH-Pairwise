"""Bound, CPU-only preparation receipt; never launches or resumes training."""
import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import time
import unittest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--baseline-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('hide GPUs explicitly for CPU-only preparation')
    import torch
    torch.set_num_threads(2)
    source = Path(__file__).parent.resolve()
    baseline = args.baseline_source.resolve()
    prior = json.loads(args.baseline_receipt.read_text())
    before, baseline_before = inventory(source), inventory(baseline)
    if prior['status'] != 'passed' or prior['baseline_python_sha256'] != baseline_before:
        raise ValueError('not the independently verified immutable baseline')
    curriculum = importlib.import_module(__package__.rsplit('.', 1)[0] + '.curriculum_training_v1.verify_validation_preparation')
    shared = Path(curriculum.__file__).parent.resolve()
    shared_before = inventory(shared)
    if prior['source_sha256'] != shared_before:
        raise ValueError('shared curriculum helpers differ from the complete CPU receipt')
    # The residual package is already anchored by this entry point; binding
    # changes only ancestor lookup so the formal baseline's S7 modules are used.
    curriculum.bind_baseline(baseline)
    head_module = importlib.import_module(__package__ + '.head')
    baseline_head = importlib.import_module(__package__.rsplit('.', 1)[0] + '.binary_scorer_v1.head')
    if baseline not in Path(baseline_head.__file__).resolve().parents:
        raise ValueError('baseline evidence/head implementation escaped its binding')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    suite = unittest.TestSuite()
    for name in ('test_head', 'test_snapshot', 'test_training'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(__package__ + '.' + name))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.out / 'tests.log').write_text(stream.getvalue())
    if (before != inventory(source) or baseline_before != inventory(baseline)
            or shared_before != inventory(shared)):
        raise ValueError('prepared or immutable source changed during testing')
    head = head_module.ResidualClusterHead()
    old_count = sum(p.numel() for p in baseline_head.BinaryClusterHead('patch').parameters())
    record = head_module.architecture_record(head)
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        scope='residual head, exact union/Q invariants, same-forward skip traces, independent arithmetic audit, synthetic CPU Matcher/head and serialized update1 recovery',
        architecture=record, baseline_head_parameters=old_count,
        added_parameters=record['parameter_count'] - old_count,
        source_sha256=before, baseline_source=str(baseline), baseline_python_sha256=baseline_before,
        baseline_receipt_sha256=digest(args.baseline_receipt), baseline_unchanged=True,
        shared_curriculum_source_sha256=shared_before, shared_curriculum_source_unchanged=True,
        reused_curriculum_binder_sha256=digest(curriculum.__file__),
        reused_training_core_sha256=digest(Path(curriculum.__file__).parent / 'training_core.py'),
        synthetic_canvas_px=32, synthetic_feature_dim=96, synthetic_contour_cap=16,
        test_log_sha256=digest(args.out / 'tests.log'), elapsed_seconds=time.time() - started,
        gpu_used=False, actual_data_used=False, formal_training_started=False,
        formal_launcher_registered=False, gpu_gate_passed=False, evaluation_controller_ready=False,
        remaining=['dedicated execution and selection binding', 'frozen evaluator/export registration',
                   'queue dependency registration after curriculum heads/evaluation', 'actual GPU gate before formal launch'])
    (args.out / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('status', 'tests', 'failures', 'errors', 'skipped',
                                                   'added_parameters', 'formal_training_started')}))
    if not result.wasSuccessful():
        print(stream.getvalue()); raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Verify composed production-baseline + v2 runtime without touching CUDA."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import unittest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output-new', type=Path, required=True)
    args = p.parse_args(); root = args.source_root.resolve()
    if root != Path(__file__).resolve().parents[4] or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('run the bound composed source with CUDA disabled')
    if args.output_new.exists():raise FileExistsError(args.output_new)
    binding = json.loads((root/'source_binding.json').read_text())
    composition = json.loads((root/'baseline_composition.json').read_text())
    assets = json.loads((root/'asset_binding.json').read_text())
    for name, expected in binding.items():
        path = (root/name).resolve(strict=True)
        if not path.is_relative_to(root) or sha(path) != expected:raise ValueError('source binding changed: '+name)
    for name, expected in composition['python_sha256'].items():
        if binding.get(name) != expected:raise ValueError('original production baseline was changed: '+name)
    for name, expected in assets.items():
        path = (root/name).resolve(strict=True)
        if not path.is_relative_to(root) or sha(path) != expected:raise ValueError('bound asset changed: '+name)
    os.environ['CURRICULUM_BASELINE_SOURCE'] = str(root)
    import torch
    torch.set_num_threads(1)
    base = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'
    from ..curriculum_scorer_eval_v1.entry import bind_evaluation
    package = root/Path(*base.rstrip('.').split('.'))
    bind_evaluation(package/'s7_consensus_eval_v14', package/'binary_eval_v1')
    names = [base+'matcher_v2_v1.'+n for n in ('test_network', 'test_additive_exposure',
        'test_source_audit', 'test_seam_supervision', 'test_runtime_schedule', 'test_model_runtime',
        'test_validation', 'test_runtime_pipeline', 'test_runtime_inputs', 'test_gradient_gate',
        'test_data_runtime', 'test_launcher', 'test_terminal', 'test_population',
        'test_compile_execution', 'test_evaluation_controller', 'test_pipeline', 'test_joint_priority')]
    names += [base+'curriculum_training_v1.'+n for n in ('test_runtime_io', 'test_model_adapter',
                                                        'test_model_integration', 'test_validation_adapter')]
    names += [base+'curriculum_scorer_eval_v1.'+n for n in ('test_evaluate', 'test_audit_controller')]
    start = time.monotonic(); log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    if torch.cuda.is_initialized():raise AssertionError('CPU verifier unexpectedly initialized CUDA')
    current = {str(p.relative_to(root)):sha(p) for p in root.rglob('*.py')}
    if current != binding or any(sha(root/name) != expected for name, expected in assets.items()):
        raise ValueError('bound source or asset changed while testing')
    record = dict(schema='matcher-v2-runtime-cpu-preparation/1', status='passed' if result.wasSuccessful() else 'failed',
        tests=result.testsRun, errors=len(result.errors), failures=len(result.failures), skipped=len(result.skipped),
        seconds=time.monotonic()-start, source_root=str(root), source_files=len(binding),
        binding_sha256=sha(root/'source_binding.json'), baseline_composition_sha256=sha(root/'baseline_composition.json'),
        asset_binding_sha256=sha(root/'asset_binding.json'), source_files_unchanged=True,
        original_baseline_files=len(composition['python_sha256']), original_baseline_unchanged=True,
        torch_version=torch.__version__, cuda_initialized=False, gpu_gate_passed=False,
        actual_new_formal_updates=0, stdout=log.getvalue())
    args.output_new.parent.mkdir(parents=True, exist_ok=True)
    with args.output_new.open('x') as out:json.dump(record, out, indent=2); out.write('\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'stdout'}))
    if not result.wasSuccessful():print(log.getvalue())
    return 0 if result.wasSuccessful() else 2


if __name__ == '__main__':raise SystemExit(main())

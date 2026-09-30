"""Check the isolated threshold source; no GPU or training launch."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import unittest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--baseline', required=True, help='Previously prepared SIMPLE source')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run this CPU preparation with CUDA_VISIBLE_DEVICES empty')
    source, baseline = Path(args.source).resolve(), Path(args.baseline).resolve()
    relative = Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1')
    package = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    changed = {str(p.relative_to(source)): sha(p) for p in source.rglob('*.py')
               if not (baseline / p.relative_to(source)).exists()
               or sha(p) != sha(baseline / p.relative_to(source))}
    allowed = {'config.py', 'pose_consensus.py', 'train.py', 'model.py', 'losses.py',
               'threshold_builder.py', 'threshold_evidence.py', 'test_threshold.py',
               'test_threshold_joint.py', 'test_frozen_start.py'}
    assert set(changed) == {str(relative / p) for p in allowed}, changed
    unchanged = ['matcher.py', 'consensus_head.py', 'evidence.py', 'compatibility.py',
                 'pose_refinement.py', 'data.py', 'targets.py', 'metrics.py',
                 'geometry.py', 'frozen_start.py', 'legacy_pose_consensus.py']
    for name in unchanged:
        assert sha(source / relative / name) == sha(baseline / relative / name), name
    sys.path.insert(0, str(source))
    module = importlib.import_module(package + '.train')
    assert Path(module.__file__).is_relative_to(source), module.__file__
    names = ['test_threshold', 'test_threshold_joint', 'test_frozen_start',
             'test_geometry', 'test_compatibility', 'test_matcher', 'test_evidence',
             'test_targets_losses', 'test_loss_numerics', 'test_training_contracts',
             'test_validation_protocol', 'test_metrics_cache', 'test_pipeline_contracts',
             'test_head_model.HeadModelTests.test_exchange_invariance_and_vector_translation_flip',
             'test_head_model.HeadModelTests.test_packed_candidate_batch_matches_dense_outputs_and_gradients']
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(package + '.' + n) for n in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = dict(status='cpu_preparation_passed' if result.wasSuccessful() else 'failed',
                   tests=result.testsRun, errors=len(result.errors), failures=len(result.failures),
                   tested_modules=names, changed_vs_simple_source_sha256=changed,
                   unchanged_source_files=unchanged,
                   source_package_sha256={p.name: sha(p) for p in (source / relative).glob('*.py')},
                   loss_coefficients_and_gt20_unchanged=True,
                   evidence_admission_changed=True, builder_only_ablation=False,
                   gpu_preflight=False, formal_training_started=False,
                   current_training_modified=False, recorded_unix=time.time())
    target = Path(args.out)
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if 'sha256' not in k}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""CPU-only version-specific source/gradient/selection preparation receipt."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True)
    parser.add_argument('--baseline',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation only')
    source=Path(args.source).resolve();baseline=Path(args.baseline).resolve()
    relative=Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1')
    package='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p.relative_to(baseline)):sha(p) for p in baseline.rglob('*.py')}
    inventory={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}
    if set(before)-set(inventory):raise ValueError('baseline source files missing')
    changed={k:v for k,v in inventory.items() if v!=before.get(k)}
    allowed={'config.py','train.py','launch.py','joint_policy.py','real_development.py','test_joint_policy.py',
             'evaluation_checkpoint.py','test_evaluation_checkpoint.py'}
    if set(changed)!={str(relative/n) for n in allowed}:raise ValueError('unreviewed source changes: '+repr(changed))
    sys.path.insert(0,str(source))
    train=importlib.import_module(package+'.train')
    if not Path(train.__file__).is_relative_to(source):raise ValueError('wrong source imported')
    # Same registered threshold-version107 as the frozen control, plus new4.
    # Inherited tests for mergefix/simple APIs are NOT this version's suite.
    names=['test_threshold','test_threshold_joint','test_frozen_start','test_geometry',
           'test_compatibility','test_matcher','test_evidence','test_targets_losses',
           'test_loss_numerics','test_training_contracts','test_validation_protocol',
           'test_metrics_cache','test_pipeline_contracts',
           'test_head_model.HeadModelTests.test_exchange_invariance_and_vector_translation_flip',
           'test_head_model.HeadModelTests.test_packed_candidate_batch_matches_dense_outputs_and_gradients',
           'test_joint_policy']
    names += ['test_evaluation_checkpoint.EvaluationCheckpointTests.'+name for name in (
        'test_immutable_copy_and_identical_replay', 'test_timing_only_change_keeps_original_bytes',
        'test_different_weights_or_selection_cannot_overwrite',
        'test_nested_development_timing_replay_preserves_archive',
        'test_nested_development_metric_change_is_not_timing')]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(package+'.'+n) for n in names)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    # Load only the external controller/split fixtures without confusing them
    # with the independently copied training package's experiments namespace.
    external='joint_external_cpu_tests'
    namespace=types.ModuleType(external);namespace.__path__=[str(Path(__file__).parent)]
    sys.modules[external]=namespace
    extra_suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(external+'.'+n)
        for n in ('test_launch_training','test_real_split'))
    extra=unittest.TextTestRunner(verbosity=1).run(extra_suite)
    assert before=={str(p.relative_to(baseline)):sha(p) for p in baseline.rglob('*.py')}
    receipt=dict(status='cpu_preparation_passed' if result.wasSuccessful() and extra.wasSuccessful() else 'failed',tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),tested_modules=names,
        source_inventory_sha256=inventory,changed_vs_frozen_threshold=changed,
        baseline_python_sources_unchanged=True,matcher_head_loss_builder_evidence_sources_unchanged=True,
        joint_optimizer_and_real_selection_new=True,gpu_preflight=False,formal_training_started=False,
        dedicated_launcher_pending=False,external_control_and_split_tests=extra.testsRun,
        external_errors=len(extra.errors),external_failures=len(extra.failures),
        external_python_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        recorded_unix=time.time(),
        preparation_revision='joint-source02-nested-timing/1',
        test_scope='registered threshold107 plus joint4 plus validation-archive5; inherited mergefix-policy binding test remains unchanged and is not the threshold protocol')
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_inventory_sha256','changed_vs_frozen_threshold')}))
    if not result.wasSuccessful() or not extra.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()

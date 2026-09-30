"""CPU/source checks only; never starts or interferes with GPU training."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import unittest

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--baseline',required=True);p.add_argument('--out',required=True)
a=p.parse_args();source=Path(a.source);base=Path(a.baseline)
relative=Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1')
package='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1'
sys.path.insert(0,str(source.resolve()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
changed={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')
    if not (base/p.relative_to(source)).exists() or sha(p)!=sha(base/p.relative_to(source))}
allowed={'config.py','pose_consensus.py','train.py','simple_builder.py','simple_modes.py','frozen_start.py',
    'test_simple.py','test_simple_modes.py','test_simple_joint_path.py','test_frozen_start.py'}
assert {str(relative/p) for p in allowed}==set(changed),changed
for name in ['matcher.py','model.py','losses.py','consensus_head.py','evidence.py','compatibility.py','pose_refinement.py','data.py']:
    assert sha(source/relative/name)==sha(base/relative/name),name
# Force import of actual training entry, not merely the offline implementation.
importlib.import_module(package+'.train')
suite=unittest.TestSuite()
for name in ['test_simple_modes','test_simple','test_simple_joint_path','test_frozen_start']:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(package+'.'+name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
record=dict(status='cpu_preparation_passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
    errors=len(result.errors),failures=len(result.failures),changed_source_sha256=changed,
    source_package_sha256={p.name:sha(p) for p in (source/relative).glob('*.py')},
    matcher_head_loss_evidence_and_data_unchanged=True,gpu_preflight=False,formal_training_started=False,
    current_training_modified=False,recorded_unix=time.time())
target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in record.items() if 'sha256' not in k}))
if not result.wasSuccessful():sys.exit(1)

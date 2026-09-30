"""CPU-only full regression and scope audit of the isolated repair source."""
import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time
import unittest


PACKAGE='experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1'
MODULE=PACKAGE.replace('/','.')


def file_map(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*.py') if '__pycache__' not in p.parts}


def verify(root,out):
    root,out=Path(root).resolve(),Path(out)
    old=root/'source';new=root/'source_mergefix1'
    if Path.cwd().resolve()!=new:
        raise ValueError('run from the isolated repair source, not active training source')
    cfg=importlib.import_module(MODULE+'.config')
    if Path(cfg.__file__).resolve().parent != new/PACKAGE:
        raise ValueError('hybrid import path would invalidate verification')
    a,b=file_map(old),file_map(new)
    removed=sorted(set(a)-set(b));added=sorted(set(b)-set(a))
    changed=sorted(k for k in set(a)&set(b) if a[k]!=b[k])
    expected_changed={PACKAGE+'/'+x for x in ('pose_consensus.py','config.py','train.py','preflight_consensus.py')}
    expected_added={PACKAGE+'/'+x for x in ('legacy_pose_consensus.py','pose_consensus_repair.py',
        'test_consensus_repair.py','evaluation_checkpoint.py','test_evaluation_checkpoint.py')}
    if removed or set(changed)!=expected_changed or set(added)!=expected_added:
        raise ValueError('source diff exceeds declared merge repair/checkpoint retention scope')
    if a[PACKAGE+'/pose_consensus.py']!=b[PACKAGE+'/legacy_pose_consensus.py']:
        raise ValueError('legacy seed/fit implementation was unexpectedly changed')
    formal=json.loads((root/'formal_m12/CONFIG.json').read_text())
    for name,expected in formal['implementation_sha256'].items():
        if a[PACKAGE+'/'+name]!=expected:
            raise ValueError('active formal source changed: '+name)
    new_config=json.loads(json.dumps(cfg.TrainingConfig().record()))
    old_config=formal['config']
    extras={'proposal_revision','merge_repair_policy','validation_checkpoint_archive'}
    if {k:v for k,v in new_config.items() if k not in extras}!=old_config:
        raise ValueError('non-repair training settings changed')
    before=dict(b);started=time.time()
    suite=unittest.defaultTestLoader.discover(str(new/PACKAGE),pattern='test_*.py',top_level_dir=str(new))
    stream=io.StringIO()
    results=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if file_map(new)!=before:
        raise ValueError('source changed during verification')
    record=dict(schema='s7-merge-source-verification/1',status='passed' if results.wasSuccessful() else 'failed',
        scope='CPU-only isolated-source integration; no formal training launch',
        test_count=results.testsRun,seconds=time.time()-started,failures=len(results.failures),errors=len(results.errors),
        changed_existing=changed,added=added,source_sha256=b,
        training_settings_unchanged_except=sorted(extras),new_settings={k:new_config[k] for k in extras},
        test_output=stream.getvalue())
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(record,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in record.items() if k not in ('source_sha256','test_output')}))
    if not results.wasSuccessful() or results.testsRun<144:
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();verify(a.root,a.out)

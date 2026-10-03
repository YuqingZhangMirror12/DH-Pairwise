"""Verify this public handoff offline with Python's standard library.

Checks bundled code hashes and aggregate arithmetic. Does NOT run a model,
recover private data, or claim to reproduce the underlying experiments.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text())


def main():
    manifest = read('code_manifest.json')
    for item in manifest['files']:
        p = ROOT/item['path']
        assert ROOT in p.resolve().parents
        content = p.read_bytes()
        assert hashlib.sha256(content).hexdigest()==item['sha256'], p
        assert len(content)==item['bytes'], p
        if p.suffix=='.py':
            ast.parse(content, filename=str(p))
    n = 0
    for name in ['simulation_data_ablation.json','endpoint_full_descriptive.json']:
        for row in read('evidence/'+name)['rows']:
            p, q, t, f = [row[k] for k in ['positives','negatives','tp','fp']]
            assert row['pairs']==p+q and row['fn']==p-t
            assert math.isclose(row['accuracy'],(t+q-f)/(p+q),abs_tol=1e-12)
            assert math.isclose(row['f1'],2*t/max(1,p+t+f),abs_tol=1e-12)
            j = row.get('joint_tp')
            if j is not None:
                assert math.isclose(row['joint_f1'],2*j/max(1,p+j+f+row['wrong_pose_accepted']),abs_tol=1e-12)
            n += 1
    t = read('evidence/training_design.json')
    assert sum(t['unique_rows'].values())==27179
    assert sum(t['positive_rows'].values())==12179
    assert sum(t['original_stage_updates'])+sum(t['additive_stage_updates'])==31667
    assert t['completed_exposures']==32*t['completed_updates']
    print(json.dumps(dict(code_files=len(manifest['files']),aggregate_rows=n,
                         status='passed',model_forward_calls=0),ensure_ascii=False))


if __name__=='__main__':
    main()

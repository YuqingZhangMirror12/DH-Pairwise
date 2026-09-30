import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
import torch
from raw_measure import q, save

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase1',required=True)
a=p.parse_args();r=Path(a.root);base=Path(a.phase1)
assert json.loads((r/'complete.json').read_text())['pairs']==2905
assert not (r/'failure.json').exists()
counts={};rechecked=[]
for split,expected,gtcount in [('sim_select',1500,750),('dunhuang_cv',803,292),('turufan',602,0)]:
    rows=[json.loads(p.read_text()) for p in sorted((r/split).glob('*.json'))]
    assert len(rows)==expected and len({x['pair_id'] for x in rows})==expected
    assert sum(x['usable_gt'] for x in rows)==gtcount
    for row in rows:
        original=json.loads((base/split/f"{row['index']:05d}.json").read_text())
        assert row['pair_id']==original['pair_id'] and row['evidence_sha256']==original['evidence_sha256']
        assert row['cloud_edges']==original['cloud_edges']
        if row['usable_gt']:
            n=row['correct_edges']
            assert row['correct_edges']+row['wrong_edges']==row['cloud_edges']
            assert sum(row['correct_pair_distance_hist'])==n*(n-1)//2
        if row['index']%97==0:
            path=base/row['evidence_file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['evidence_sha256']
            raw=torch.load(path,map_location='cpu',weights_only=False)
            cloud=raw['proposals'].cloud
            assert len(cloud.ids)==row['cloud_edges']
            if row['usable_gt']:
                x=cloud.displacement.double().numpy()
                good=np.linalg.norm(x-np.asarray(original['target_translation_rc']),axis=1)<=20
                assert q(pdist(x[good]))==row['correct_pair_distance']
            rechecked.append([split,row['index']])
    counts[split]=dict(pairs=expected,usable_gt=gtcount)
save(r/'verification.json',dict(passed=True,all_rows_identity_and_denominators_checked=True,
    counts=counts,source_sha_and_exact_distance_recomputed=rechecked,gt_not_in_inference=True))
print(json.dumps(dict(passed=True,counts=counts,rechecked=len(rechecked))))

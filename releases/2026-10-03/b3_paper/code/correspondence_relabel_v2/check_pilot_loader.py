"""Exercise actual CPU archive loader, overlay loader and target adapters."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from overlay_dataset import ReviewOverlayDataset
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.data import collate
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.check_data_compatibility import check_batch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=False);torch.set_num_threads(1);start=time.time()
    cases=json.loads((a.evidence/'pilot.json').read_bytes())['cases'];entries=[]
    for c in cases:
        old=(a.evidence/c['files']['sample']['path']).resolve();pid=c['admission']['pair_id'];overlay=(a.pilot/pid/'labels.npz').resolve()
        entries.append(dict(c['admission'],recipe=c['record']['recipe'],sample_path=str(old),sample_sha256=sha(old),label_overlay=dict(path=str(overlay),sha256=sha(overlay))))
    totals={};low={}
    for mode in ('full','tight'):
        manifest=a.out/('manifest_'+mode+'.json');write(manifest,dict(mode=mode,training_admitted=False,entries=entries))
        try:ReviewOverlayDataset(manifest)
        except PermissionError:pass
        else:raise AssertionError('Training guard did not refuse implicit admission')
        data=ReviewOverlayDataset(manifest,review_only=True);rows=[]
        for i in range(len(data)):
            item=data[i];pid=item[0].pair_id
            archive,report=load_sample(a.pilot/pid/(mode+'.npz'))
            for k in archive.__dataclass_fields__:
                assert np.array_equal(getattr(archive,k),getattr(item[0],k)),k
            rows.extend(check_batch([item],collate([item])))
        totals[mode]=len(rows);low[mode]=[r['pair_id'] for r in rows if r['label'] and r['inherited_pairs']<4]
    write(a.out/'complete.json',dict(status='cpu_load_collate_target_adapter_passed',cases=len(cases),modes=totals,
        low_support_positive_ids=low,training_admitted=False,training_min4_gate_not_waived=True,
        default_training_use_refused=True,model_forwards=0,optimizer_updates=0,seconds=time.time()-start))
    print(json.dumps(dict(passed=len(cases),modes=totals,low_support_counts={k:len(v) for k,v in low.items()})))
if __name__=='__main__':main()

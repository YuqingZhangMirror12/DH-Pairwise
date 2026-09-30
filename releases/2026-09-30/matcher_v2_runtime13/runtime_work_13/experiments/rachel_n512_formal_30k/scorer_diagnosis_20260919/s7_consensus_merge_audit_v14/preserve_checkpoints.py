"""Preserve available checkpoints byte-for-byte, without touching training.

Files are read via a single open handle, so the trainer's atomic rename cannot
produce a mix of two states. This tool does not claim to recover overwritten
historical epochs and never starts/stops/rebinds an experiment.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import torch


def preserve(root, out):
    root, out = Path(root), Path(out)
    binding = json.loads((root/'formal_scratch/CONFIG.json').read_text())
    if binding.get('arm') != 'scratch' or not binding.get('formal_training'):
        raise ValueError('formal scratch binding required')
    stage = root/'formal_scratch/matcher'
    snapshots=[]
    for name in ('best_joint.pt','best_layout.pt','last.pt'):
        source=stage/name
        raw=source.read_bytes()
        checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        if checkpoint.get('binding') != binding or checkpoint.get('stage') != 'matcher':
            raise ValueError('checkpoint binding or stage mismatch')
        snapshots.append((name,raw,dict(source=str(source),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
            epoch=checkpoint['epoch'],offset=checkpoint.get('offset'),updates=checkpoint.get('updates'),
            exposures=checkpoint.get('exposures'),plateau=checkpoint.get('plateau'),
            role='resumable optimizer/RNG state' if name=='last.pt' else 'selected evaluation weights')))
    out.mkdir(parents=True,exist_ok=False)
    for name,raw,metadata in snapshots:
        with (out/name).open('xb') as f:
            f.write(raw)
    record=dict(schema='s7-repair-preserved-matcher/1',created_unix=time.time(),
        scope='available states only; no reconstruction of missing historical checkpoints',
        original_training_unchanged=True,binding=binding,
        checkpoints={n:m for n,_,m in snapshots})
    (out/'preservation.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    print(json.dumps({n:m for n,_,m in snapshots}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(2);preserve(a.root,a.out)

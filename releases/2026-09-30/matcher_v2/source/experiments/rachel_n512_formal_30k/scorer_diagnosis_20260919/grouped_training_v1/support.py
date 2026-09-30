"""Shared fixed-S7 grouped-training protocol. No REAL inputs during fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only import cache, data

R = Path('/root/autodl-tmp/rachel_score_design_20260913_001')
ROOT = R/'grouped_training_v1_20260921'
SOURCE = R/'local_evidence_v2_20260921/source'
TRAIN_MANIFEST = R/'s6_s7_20260915/preparation/data/train_s7_24k.json'
VAL_CACHE = R/'scorer_diagnosis_20260919/matched_only_cache_v1/formal_v1/val'
SCHEMA = 's7-same-anchor-grouped-scorer/1'
SEED, EPOCHS, GROUP_SIZE, GROUP_BATCH = 260921, 16, 4, 12
ARMS = ('gs0_pair_bce', 'gs1_group_contrast')
GPUS = ('GPU-faabfde3-405e-1c7f-9b19-8e538a248eb5', 'GPU-bb55b6a9-367d-3705-1ae5-7b3042012e86')

def read(p): return json.loads(Path(p).read_text())
def save(p, x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

class GroupCache:
    """New schema, original tensor contract. Never bypass old cache validation."""
    batch = data.FormalCache.batch
    def __init__(self, root, split, require_complete=True):
        self.root=Path(root)/'cache'/split; self.split=split
        p=read(self.root/'protocol.json')
        if p['schema']!=SCHEMA or p['matcher_sha']!=cache.SOURCE_SHA:
            raise ValueError('not this fixed Matcher group cache')
        self.records=read(Path(root)/'groups'/f'{split}_pairs.json')
        self.arrays={k:np.load(self.root/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in cache.ARRAYS}
        n=len(self.records)
        for k,(d,shape) in cache.ARRAYS.items():
            if self.arrays[k].shape!=(n,*shape) or str(self.arrays[k].dtype)!=d:
                raise ValueError('cache shape/dtype differs: '+k)
        if require_complete and not self.arrays['ready'].all(): raise ValueError('incomplete group cache')
        self.groups=np.load(Path(root)/'groups'/f'{split}_groups.npy')
        if self.groups.shape!=(n//4,4) or not np.array_equal(np.sort(self.groups.ravel()),np.arange(n)):
            raise ValueError('groups must partition pairs exactly once')
        if not np.all(self.arrays['label'][self.groups].sum(1)==1): raise ValueError('one true neighbour per group')
        self.binding=dict(root=str(self.root),split=split,count=n,groups=len(self.groups),matcher_sha=p['matcher_sha'],
                          manifest_sha=data.sha(Path(root)/'groups'/f'{split}_pairs.json'))
    def __len__(self): return len(self.records)

def group_loss(logits, labels, valid, arm):
    """Class-balanced BCE plus optional within-anchor listwise CE.

    Both arms see exactly the same groups, order, batch, and class weights.
    Softmax is only a training objective, never the deployed pair probability.
    Invalid or zero-support positive candidates are not relabelled negative.
    """
    from torch.nn import functional as F
    z,y,v=(x.reshape(-1,GROUP_SIZE) for x in (logits,labels,valid))
    if not torch.all((y==0)|(y==1)) or not torch.all(y.sum(1)==1): raise ValueError('malformed group labels')
    raw=F.binary_cross_entropy_with_logits(z,y,reduction='none')
    weights=torch.where(y.bool(),.5, .5/(GROUP_SIZE-1))*v
    bce=(raw*weights).sum(1).mean()
    eligible=v.all(1)
    # Average over every group, with invalid groups contributing zero; avoid
    # positive-dependent reweighting or selecting only easy correct layouts.
    rank=(F.cross_entropy(z,y.argmax(1),reduction='none')*eligible).mean()
    loss=bce + (.5*rank if arm=='gs1_group_contrast' else 0.)
    if arm not in ARMS: raise ValueError(arm)
    return loss,bce,rank

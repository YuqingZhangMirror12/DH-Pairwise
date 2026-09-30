"""Immutable materialized pairs; no implicit second mirror/erosion transform."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs
from .targets import PairLabels


class Dataset:
    def __init__(self,path,expected_sha256=None):
        self.path=Path(path)
        raw=self.path.read_bytes()
        self.sha256=hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and self.sha256!=expected_sha256:
            raise ValueError('dataset manifest changed')
        record=json.loads(raw)
        self.entries=record['entries']
        self.artifact_root=Path(record.get('artifact_root',self.path.parent))
        if len({e['pair_id'] for e in self.entries})!=len(self.entries):
            raise ValueError('duplicate Pair IDs')

    def __len__(self):
        return len(self.entries)

    def __getitem__(self,index):
        entry=self.entries[index]
        path=entry.get('sample_path') or str(self.artifact_root/entry['artifact_path'])
        sample,report=load_sample(path)
        if sample.pair_id!=entry['pair_id'] or int(sample.label)!=int(entry['label']):
            raise ValueError('materialized sample identity mismatch')
        return sample,report,entry


def recipe_for(entry,report):
    value=entry.get('recipe',entry.get('s7_recipe',report.get('recipe')))
    if value is not None:
        return value
    if entry.get('view')=='clean':
        if report['changed_pair']:
            raise ValueError('clean validation view actually changed')
        return 'clean'
    if entry.get('view')=='hard':
        return 'hard_damage_unspecified' if report['changed_pair'] else 'clean'
    raise ValueError('missing explicit corrosion/clean identity')


def collate(items):
    samples,reports,entries=zip(*items)
    values=collate_rachel_pairs(samples).as_dict()
    out={k:torch.from_numpy(np.array(v,copy=True)) for k,v in values.items() if isinstance(v,np.ndarray)}
    out['pair_ids']=list(values['pair_ids'])
    out['recipes']=[recipe_for(e,r) for e,r in zip(entries,reports)]
    out['precise_recipe']=torch.tensor([x=='clean' for x in out['recipes']],dtype=torch.bool)
    # Legacy reports may disable their exact translation auxiliary on damage.
    # Global candidate GT remains available separately as translation_valid.
    out['pose_enabled']=out['precise_recipe']&out['translation_valid']
    return out


def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def pair_labels(batch):
    return [PairLabels(bool(batch['labels'][i]),bool(batch['translation_valid'][i]),
        batch['translation_a_to_b_rc'][i],batch['target_a'][i],batch['target_b'][i],
        (batch['target_a'][i]>=0)&batch['precise_recipe'][i],
        (batch['target_b'][i]>=0)&batch['precise_recipe'][i]) for i in range(len(batch['labels']))]

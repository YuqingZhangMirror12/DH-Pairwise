"""Existing, actually-applied S7 damage only; paired mirror augmentation."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs
from ..seam_context_v3.augmentation import paired_mirror, mirror_schedule
from ..seam_context_v3.prepare import read, save, sha, sources

RECIPES=('local','seam_gaps','partial_curve')
INPUTS=('mask_a','mask_b','points_rc_a','points_rc_b','contour_valid_a','contour_valid_b')
SEED=260923


def select_effective_groups(entries):
    groups=defaultdict(list)
    for entry in entries:
        if entry['s7_recipe'] in RECIPES:groups[entry['s7_group_index']].append(entry)
    selected=[]
    for key,group in sorted(groups.items()):
        if len(group)!=2 or sorted(int(e['label']) for e in group)!=[0,1]:
            raise ValueError('requires coupled positive/negative S7 groups')
        if len({e['s7_recipe'] for e in group})!=1:
            raise ValueError('coupled recipe differs')
        if all(e['changed_pair'] for e in group):selected.extend(group)
    return selected


def prepare(source,validation,out):
    source,validation,out=Path(source),Path(validation),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    original=read(source);selected=select_effective_groups(original['entries'])
    training_sources=set().union(*(sources(e['source_row']) for e in original['entries']))
    entries=[dict(pair_id=e['pair_id'],sample_path=str(Path(original['artifact_root'])/e['artifact_path']),
        label=int(e['label']),recipe=e['s7_recipe'],group=e['s7_group_index'],changed=True,
        sources=sorted(sources(e['source_row']))) for e in selected]
    if not entries or len({e['pair_id'] for e in entries})!=len(entries):raise ValueError('empty/duplicate hard population')
    manifests={}
    for name in ('cal_clean','cal_hard','select_clean','select_hard'):
        record=read(validation/(name+'.json'))
        src=set().union(*(set(e['sources']) for e in record['entries']))
        if src&training_sources:raise ValueError('validation overlaps original S7 pretraining sources')
        save(out/(name+'.json'),record);manifests[name]=dict(count=len(record['entries']),sha256=sha(out/(name+'.json')),sources=sorted(src))
    if set(manifests['cal_clean']['sources'])&set(manifests['select_clean']['sources']):raise ValueError('CAL/SELECT source overlap')
    save(out/'train.json',dict(entries=entries,source=str(source),source_sha256=sha(source),split='train'))
    save(out/'protocol.json',dict(status='ready',training_count=len(entries),positive_count=sum(e['label'] for e in entries),
        recipe_counts=dict(Counter(e['recipe'] for e in entries)),source=str(source),source_sha256=sha(source),
        hard_definition='local/seam_gaps/partial_curve, both coupled labels actually changed; no clean fallback',
        new_physical_damage_generated=False,existing_S7_subset=True,mirror_probability=.30,mirror_axes=['horizontal','vertical'],
        mirror_coupled_by_positive_negative_group=True,validation=manifests,real_or_test_used=False,
        validation_hard_caveat='Reuse frozen v3 hard views, including documented geometry fallbacks; not all are modified.',
        ancestry='Existing reciprocal source-arc labels; artificial new boundaries retain ignore labels; no new GT invention'))
    print(read(out/'protocol.json'),flush=True)


class Dataset:
    def __init__(self,path,train=False):
        self.entries=read(path)['entries'];self.train=train
        groups=sorted({e.get('group',i) for i,e in enumerate(self.entries)})
        lookup={g:i for i,g in enumerate(groups)}
        self.groups=np.array([lookup[e.get('group',i)] for i,e in enumerate(self.entries)])
        self.group_count=len(groups);self.set_epoch(0)

    def set_epoch(self,epoch):
        self.mirror_codes=mirror_schedule(self.group_count,.30 if self.train else 0,SEED,epoch)[self.groups]

    def __len__(self):return len(self.entries)

    def __getitem__(self,i):
        entry=self.entries[i];sample,report=load_sample(entry['sample_path'])
        if sample.pair_id!=entry['pair_id'] or int(sample.label)!=entry['label']:raise ValueError('sample identity mismatch')
        if self.train and (not report['changed_pair'] or entry['recipe'] not in RECIPES):raise ValueError('clean fallback in hard training')
        code=int(self.mirror_codes[i])
        if code:sample=paired_mirror(sample,'horizontal' if code==1 else 'vertical')
        return sample,report,entry


def collate(items):
    samples,reports,entries=zip(*items)
    values=collate_rachel_pairs(samples).as_dict()
    batch={k:torch.from_numpy(np.array(v,copy=True)) for k,v in values.items() if isinstance(v,np.ndarray)}
    batch['pair_ids']=values['pair_ids'];batch['pose_enabled']=torch.tensor([r['pose_supervision_enabled'] for r in reports],dtype=torch.bool)
    return batch


def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','validation','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a.source,a.validation,a.out)

import json
from pathlib import Path
import numpy as np
import torch
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs
from .targets import base_target_metadata
from .augmentation import paired_mirror, mirror_schedule


INPUTS = ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b')


class Dataset:
    def __init__(self, manifest, indices=None, *, train_mirror_probability=0., seed=260921):
        self.path = Path(manifest)
        self.record = json.loads(self.path.read_text())
        self.entries = self.record['entries']
        if indices is not None:
            self.entries = [self.entries[int(i)] for i in indices]
        if train_mirror_probability and self.path.stem != 'train':
            raise ValueError('paired mirror augmentation is TRAIN only')
        self.mirror_probability = train_mirror_probability
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.mirror_codes = mirror_schedule(len(self.entries),self.mirror_probability,self.seed,epoch)

    def augmentation_counts(self):
        return {name:int((self.mirror_codes==k).sum()) for k,name in
                enumerate(('unchanged','horizontal','vertical'))}

    def __len__(self): return len(self.entries)

    def __getitem__(self, i):
        entry = self.entries[int(i)]
        sample, report = load_sample(entry['sample_path'])
        if 'target_metadata' in entry:
            with np.load(entry['target_metadata'], allow_pickle=False) as z:
                target_metadata = {k: z[k].copy() for k in z.files}
        else:
            target_metadata = base_target_metadata(sample)
        code = int(self.mirror_codes[int(i)])
        if code:
            axis = 'horizontal' if code==1 else 'vertical'
            sample = paired_mirror(sample,axis)
            report = dict(report,v3_paired_mirror=axis)
            entry = dict(entry,v3_paired_mirror=axis)
        return sample, report, target_metadata, entry


def collate(items):
    samples, reports, extras, entries = zip(*items)
    arrays = collate_rachel_pairs(samples).as_dict()
    result = {k: torch.from_numpy(np.array(v, copy=True)) for k, v in arrays.items() if isinstance(v, np.ndarray)}
    result['pair_ids'] = arrays['pair_ids']
    result['pose_enabled'] = torch.tensor([r['pose_supervision_enabled'] for r in reports], dtype=torch.bool)
    for side in 'ab':
        for key in ('component', 'stop'):
            out = np.full((len(items), 512), -1, np.int64)
            for b, values in enumerate(extras):
                value = values[key+'_'+side]
                out[b, :len(value)] = value
            result[key+'_'+side] = torch.from_numpy(out)
        gap = np.zeros((len(items), 512, 512), bool)
        for b, values in enumerate(extras):
            value = values['gap_'+side]
            gap[b, :len(value), :len(value)] = value
        result['gap_'+side] = torch.from_numpy(gap)
    result['entries'] = entries
    return result


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

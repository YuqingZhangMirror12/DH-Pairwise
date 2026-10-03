"""Explicit review loader. Refuses implicit use as a training admission."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample

class ReviewOverlayDataset:
    def __init__(self,manifest,*,review_only=False):
        if review_only is not True:
            raise PermissionError('Human visual approval and a NEW training admission are still required')
        self.manifest=Path(manifest);self.record=json.loads(self.manifest.read_bytes())
        if self.record['training_admitted'] is not False:raise ValueError('unexpected admission mutation')
        self.entries=self.record['entries'];self.mode=self.record['mode']
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        e=self.entries[i];path=Path(e['sample_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=e['sample_sha256']:raise ValueError('old sample changed')
        sample,report=load_sample(path)
        if e.get('label_overlay'):
            overlay=Path(e['label_overlay']['path'])
            if hashlib.sha256(overlay.read_bytes()).hexdigest()!=e['label_overlay']['sha256']:raise ValueError('label overlay changed')
            with np.load(overlay,allow_pickle=False) as z:
                sample=replace(sample,target_a=z[self.mode+'_a'].copy(),target_b=z[self.mode+'_b'].copy())
        return sample,report,e

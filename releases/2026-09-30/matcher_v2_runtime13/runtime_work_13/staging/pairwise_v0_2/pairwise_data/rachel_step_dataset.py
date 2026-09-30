"""Strict S5 materialized reader; the paired512 arm reuses frozen v4 files."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .rachel_materialized_dataset import load_sample
from .rachel_paired_density_dataset import file_sha256,validate_density_sample
from .rachel_step_density import SAMPLING_SCHEMA, TARGET_SCHEMA, CAP

SCHEMA='rachel-step-source-dataset/1'
PAIR_SCHEMA='rachel-step-source-pair/1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_step_sample(sample,report,identity):
    d=report.get('density',{});step=report.get('step_sampling',{})
    if d.get('schema_version')!=TARGET_SCHEMA or d.get('sampling')!=step:
        raise ValueError('missing S5 target/sampling receipt')
    if report.get('step_density',{}).get('identity_sha256')!=identity:
        raise ValueError('S5 archive identity differs')
    if step.get('schema_version')!=SAMPLING_SCHEMA or step.get('contour_cap')!=CAP or step.get('target_step_px')!=3.:
        raise ValueError('S5 sampling configuration differs')
    if d.get('physical_masks_unchanged') is not True or d.get('gt_translation_used_for_matching') is not False:
        raise ValueError('S5 target-blind source contract differs')
    counts=[]
    for side in 'ab':
        p=getattr(sample,'points_rc_'+side);v=getattr(sample,'contour_valid_'+side);t=getattr(sample,'target_'+side)
        if not 4<=len(p)<=CAP or p.shape!=(len(p),2) or not np.isfinite(p).all():raise ValueError('bad S5 contour')
        if v.shape!=(len(p),) or not v.all() or len(np.unique(p,axis=0))!=len(p):raise ValueError('fake S5 points')
        if step['sides'][side]['count']!=len(p):raise ValueError('S5 point count differs from receipt')
        if t.shape!=(len(p),) or np.any(t< -2):raise ValueError('bad S5 targets')
        counts.append(int(np.count_nonzero(t>=0)))
        if sample.label and not counts[-1]:raise ValueError('positive became all-ignore/dustbin')
        if not sample.label and np.any(t!=-1):raise ValueError('negative non-dustbin targets')
    if counts[0]!=counts[1] or counts[0]!=d['new_match_count'] or counts[0]!=report['effective_supervised_match_count']:
        raise ValueError('S5 reciprocal match count mismatch')


class StepSourceDataset:
    """Read one complete TRAIN24K / VAL3K / TEST3K, never a smoke subset.

    Both arms use the SAME manifest and population. ``paired512`` reads the
    referenced original v4 archive, never an S5-downsampled approximation.
    """
    prototype_only=False

    def __init__(self,manifest_path,sampling='step3'):
        if sampling not in ('step3','paired512'):raise ValueError('S5 sampling arm must be step3 or paired512')
        self.manifest_path=Path(manifest_path).resolve(strict=True)
        record=json.loads(self.manifest_path.read_text())
        if record.get('schema_version')!=SCHEMA or record.get('split') not in ('train','val','test'):
            raise ValueError('requires dedicated S5 manifest')
        expected=24000 if record['split']=='train' else 3000
        if (record.get('status')!='complete' or record.get('full_split') is not True or
                any(record.get(k)!=expected for k in ('original_count','selected_count','completed_count')) or
                record.get('failed_count')!=0 or record.get('failures')):
            raise ValueError('refusing incomplete/smoke/noncanonical S5 population')
        self.root=Path(record['artifact_root']).resolve(strict=True)
        self.split=record['split'];self.sampling=sampling
        self.contour_cap=CAP if sampling=='step3' else 512
        base=record['protocol'];self.selection_identity=record['identity_sha256']
        if digest(base)!=self.selection_identity:raise ValueError('S5 protocol digest mismatch')
        if (base.get('source_density_version')!='v4' or base.get('ownership_protocol')!='density-local-unknown-ownership/4'
                or base.get('sampling_mode')!='paired_shared_arc_step' or base.get('step_px')!=3.
                or base.get('contour_cap')!=CAP or base.get('full_split') is not True
                or base.get('split')!=self.split):raise ValueError('S5 source/sampling protocol mismatch')
        control=base.get('control512',{})
        if control.get('contour_cap')!=512 or control.get('source_density_version')!='v4':
            raise ValueError('S5 control must be existing source-rebuilt v4 N512')
        if control.get('source_manifest_sha256')!=base.get('source_manifest_sha256'):
            raise ValueError('S5 control and step source manifests differ')
        self.entries=record['entries'];ids=[e['pair_id'] for e in self.entries]
        if (len(ids)!=expected or len(set(ids))!=expected or ids!=base['selected_pair_ids']
                or digest(ids)!=base['source_selection_sha256']):raise ValueError('S5 membership/order differs')
        self.rows=[e['source_row'] for e in self.entries]
        if any(r['split']!=self.split or r['pair_id']!=e['pair_id'] or bool(r['label'])!=bool(e['label'])
               for r,e in zip(self.rows,self.entries)):raise ValueError('S5 cross-split/label membership')
        self.stats=record['stats']
        if self.stats!={'positive':sum(bool(e['label']) for e in self.entries),
                        'negative':sum(not e['label'] for e in self.entries)}:
            raise ValueError('S5 class counts differ')
        self.protocol=dict(base,sampling=sampling,actual_contour_cap=self.contour_cap,
            actual_sampling_mode='paired_shared_arc_step' if sampling=='step3' else 'fixed_cap512_v4',
            common_manifest_identity_sha256=self.selection_identity)
        self.identity=digest(self.protocol)
        self._verified=set()
        for entry in self.entries:
            p=(self.root/entry['artifact_path']).resolve()
            if self.root not in p.parents:raise ValueError('S5 archive escapes root')
            control_path=Path(entry['control_artifact_path']).resolve()
            control_root=Path(control['artifact_root']).resolve()
            if control_root not in control_path.parents:raise ValueError('N512 archive escapes control root')

    def __len__(self):return len(self.entries)

    def weathered(self,index):
        entry=self.entries[index]
        if self.sampling=='step3':
            path=self.root/entry['artifact_path'];sha=entry['artifact_sha256']
        else:
            path=Path(entry['control_artifact_path']);sha=entry['control_artifact_sha256']
        if index not in self._verified:
            if file_sha256(path)!=sha:raise ValueError('S5/control archive SHA mismatch')
            self._verified.add(index)
        sample,report=load_sample(path)
        if sample.pair_id!=entry['pair_id'] or bool(sample.label)!=bool(entry['label']):
            raise ValueError('S5/control pair label differs')
        if self.sampling=='step3':validate_step_sample(sample,report,self.selection_identity)
        else:
            control=self.protocol['control512']
            validate_density_sample(sample,report,512,control['identity_sha256'])
        return sample,report

    def __getitem__(self,index):return self.weathered(index)[0]
    def get_report(self,index):return self.weathered(index)[1]
    def set_epoch(self,epoch):pass

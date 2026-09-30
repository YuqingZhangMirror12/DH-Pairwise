"""Cap-aware reader for independently materialized paired source-density data.

The old N512 materialized dataset is not modified. These manifests require a
complete fixed selection, one common source/pipeline identity across both caps,
and per-pair completion records. Model samples never contain source geometry.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .rachel_materialized_dataset import load_sample

SCHEMA = 'rachel-paired-source-density-train/1'
PAIR_SCHEMA = 'rachel-paired-source-density-pair/1'


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def validate_density_sample(sample,report,cap,identity):
    d=report.get('density',{})
    origin=report.get('paired_density',{})
    if d.get('cap')!=cap or origin.get('identity_sha256')!=identity:
        raise ValueError('archive density cap / paired identity differs')
    if d.get('physical_masks_unchanged') is not True or d.get('gt_translation_used_for_matching') is not False:
        raise ValueError('archive source-density supervision contract differs')
    counts=[]
    for side in 'ab':
        points=getattr(sample,'points_rc_'+side)
        valid=getattr(sample,'contour_valid_'+side)
        target=getattr(sample,'target_'+side)
        if not (4<=len(points)<=cap) or points.shape!=(len(points),2) or not np.isfinite(points).all():
            raise ValueError('invalid unpadded density contour')
        if valid.shape!=(len(points),) or not valid.all() or len(np.unique(points,axis=0))!=len(points):
            raise ValueError('fake / duplicated / padded density points')
        count=int(np.count_nonzero(target>=0));counts.append(count)
        if sample.label:
            if not count:raise ValueError('positive source cannot become all-ignore/dustbin')
        elif np.any(target!=-1):
            raise ValueError('negative real contour points must all be dustbin')
    if counts[0]!=counts[1] or counts[0]!=d['new_match_count'] or counts[0]!=report['effective_supervised_match_count']:
        raise ValueError('source-density target count differs from sidecar')


class PairedSourceDensityDataset:
    """RachelPairSample-compatible cap-aware derivative with fixed membership."""
    def __init__(self,manifest_path):
        self.manifest_path=Path(manifest_path).resolve(strict=True)
        record=json.loads(self.manifest_path.read_text())
        if record.get('schema_version')!=SCHEMA or record.get('split')!='train':
            raise ValueError('requires a paired-source-density TRAIN manifest')
        if record.get('status')!='complete' or record.get('failed_pair_count')!=0:
            raise ValueError('refusing incomplete / failed paired-density population')
        self.contour_cap=record['contour_cap']
        if self.contour_cap not in (512,1024):raise ValueError('unsupported paired cap')
        self.root=Path(record['artifact_root']).resolve(strict=True)
        self.identity=record['identity_sha256']
        run=json.loads((self.root/'run_state.json').read_text())
        if run.get('status')!='complete' or run.get('identity_sha256')!=self.identity:
            raise ValueError('paired materialization is not completely committed')
        selection=json.loads((self.root/'source_selection.json').read_text())
        if selection['identity_sha256']!=self.identity:
            raise ValueError('selection identity differs')
        self.entries=record['entries']
        selected=selection['selected_pair_ids']
        if len(set(selected))!=len(selected) or not selected or [e['pair_id'] for e in self.entries]!=selected:
            raise ValueError('derivative membership / order differs from fixed selection')
        if record['selected_pair_count']!=len(selected) or record['completed_pair_count']!=len(selected):
            raise ValueError('population completion count differs')
        self.rows=[entry['source_row'] for entry in self.entries]
        if any(row['split']!='train' for row in self.rows):raise ValueError('held-out source in TRAIN derivative')
        self.stats=record['stats'];self.protocol=record['protocol'];self.split='train'
        for entry in self.entries:
            path=(self.root/entry['artifact_path']).resolve()
            if self.root not in path.parents:raise ValueError('artifact outside derivative root')

    def __len__(self):return len(self.entries)

    def weathered(self,index):
        entry=self.entries[index];path=self.root/entry['artifact_path']
        complete=json.loads((path.parent/'complete.json').read_text())
        if complete.get('schema_version')!=PAIR_SCHEMA or complete.get('status')!='complete':
            raise ValueError('missing paired completion marker')
        if complete.get('identity_sha256')!=self.identity or complete.get('pair_id')!=entry['pair_id']:
            raise ValueError('pair completion identity differs')
        if set(complete.get('caps',{}))!={'512','1024'}:
            raise ValueError('pair completion does not contain both densities')
        receipt=complete['caps'][str(self.contour_cap)]
        if receipt['artifact_path']!=entry['artifact_path'] or receipt['sha256']!=entry['artifact_sha256']:
            raise ValueError('manifest and paired completion disagree')
        sample,report=load_sample(path)
        if sample.pair_id!=entry['pair_id'] or bool(sample.label)!=bool(entry['label']):
            raise ValueError('sample identity / label differs')
        validate_density_sample(sample,report,self.contour_cap,self.identity)
        return sample,report

    def __getitem__(self,index):return self.weathered(index)[0]
    def get_report(self,index):return self.weathered(index)[1]
    def set_epoch(self,epoch):
        """No augmentation redraw: both arms use the same fixed masks."""


class PairedSourceDensityWeatheredDataset(PairedSourceDensityDataset):
    def __getitem__(self,index):return self.weathered(index)

"""Resolve exact fixed E1 entries to original dense source geometry.

This bounded-pilot adapter reads the REAL Full24 manifest layout (not a
hand-authored two-row lookup). It never changes manifests or source images.
Production 24K preparation is deliberately not started by importing it.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .rachel_materialized_dataset import load_sample
from .rachel_preprocess import centerpad_mask_with_transform
from .rachel_union_augmentation import normalize_boundary_ownership
from .rachel_source_density import resample_source_density


def _mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L')) > 0


class SourceDensityRecordResolver:
    """Lazy original-source resolver; not a formal cap-specific TRAIN manifest."""
    prototype_only = True

    def __init__(self, manifest, canonical_root):
        self.manifest_path = Path(manifest).resolve(strict=True)
        record = json.loads(self.manifest_path.read_text())
        if record.get('schema_version') != 'rachel-materialized-e1-train/1' or record.get('split') != 'train':
            raise ValueError('requires actual fixed materialized E1 TRAIN manifest')
        self.entries = record['entries']
        self.artifact_root = Path(record['artifact_root']).resolve(strict=True)
        self.canonical_root = Path(canonical_root).resolve(strict=True)
        self._groups, self._unions = {}, {}

    def _group(self, generator, group_id):
        key = (generator,group_id)
        if key not in self._groups:
            path = self.canonical_root/'groups'/generator/(group_id+'.json')
            group = json.loads(path.read_text())
            if group.get('status') != 'processed_group':
                raise ValueError('source group is not processed')
            self._groups[key] = (group,path)
        return self._groups[key]

    def _parents(self, group):
        return {f['fragment_id']:_mask(self.canonical_root/f['target_audit']['parent_mask_path'])
                for f in group['fragments']}

    def _union(self, root, pair_id):
        root = Path(root)
        if root not in self._unions:
            records = {}
            with (root/'candidates.jsonl').open() as stream:
                for line in stream:
                    record = json.loads(line)
                    if record['pair_id'] in records:
                        raise ValueError('duplicate union provenance')
                    records[record['pair_id']] = record
            self._unions[root] = records
        return self._unions[root][pair_id]

    @staticmethod
    def _verify_centerpad(parent, clean, offset):
        centered = centerpad_mask_with_transform(parent)
        if not np.array_equal(centered.model_mask,clean):
            raise ValueError('source parent/union does not reproduce exact clean physical input')
        if tuple(centered.parent_to_model_offset_rc) != tuple(offset):
            raise ValueError('recorded source-to-model offset differs from reconstruction')

    def resolve(self, index):
        entry = self.entries[index];row=entry['source_row']
        if row['split'] != 'train' or bool(row['label']) != bool(entry['label']):
            raise ValueError('TRAIN entry label/provenance conflict')
        artifact = (self.artifact_root/entry['artifact_path']).resolve(strict=True)
        if self.artifact_root not in artifact.parents:
            raise ValueError('artifact outside fixed root')
        sample,report=load_sample(artifact)
        if sample.pair_id != entry['pair_id'] or bool(sample.label) != bool(row['label']):
            raise ValueError('fixed archive identity differs')
        root=Path(entry['source_root'])
        clean={s:_mask(root/row['fragment_'+s]['model_mask_path']) for s in 'ab'}
        provenance=dict(index=int(index),pair_id=sample.pair_id,label=bool(sample.label),
            source_stratum=entry['source_stratum'],fixed_artifact=str(artifact),
            source_root=str(root),groups=[],ownership_reference_only=False)
        if entry['source_stratum']=='union_positive_tiny':
            if not sample.label:
                raise ValueError('unexpected negative union family')
            u=self._union(root,sample.pair_id)
            if u['split']!='train' or u['resized'] or u['rotation_degrees'] or u['scale']!=1:
                raise ValueError('unsupported union source transform')
            group,path=self._group(u['generator'],u['group_id'])
            parent=self._parents(group)
            normalized,ownership=normalize_boundary_ownership(parent)
            source=dict(a=np.logical_or.reduce([normalized[k] for k in u['merged_parent_members']]),
                        b=normalized[u['singleton_parent_member']])
            allowed={frozenset((c['fragment_a_token'].split('/')[-1],c['fragment_b_token'].split('/')[-1]))
                     for c in group['candidates'] if c['label']}
            if not any(frozenset((m,u['singleton_parent_member'])) in allowed for m in u['merged_parent_members']):
                raise ValueError('union lacks original surviving CSV adjacency')
            offsets={s:u['model_offset_'+s+'_rc'] for s in 'ab'}
            for s in 'ab':self._verify_centerpad(source[s],clean[s],offsets[s])
            provenance.update(groups=[str(path)],union_members=u['merged_parent_members'],
                singleton_member=u['singleton_parent_member'],omitted_members=u['omitted_parent_members'],
                union_signature=u['geometry_signature'],ownership_normalization=ownership)
            allowance=0.
        else:
            fragments,groups,parents,source,offsets={},{},{},{},{}
            for s in 'ab':
                token=row['fragment_'+s]['fragment_token']
                parts=token.split('/')
                if len(parts)!=4 or parts[0]!='rachel':
                    raise ValueError('unknown native fragment-token layout')
                group,path=self._group(parts[1],parts[2]);groups[s]=group
                fragment=next(f for f in group['fragments'] if f['fragment_token']==token)
                if fragment['split_unit_id']!=row['fragment_'+s]['split_unit_id']:
                    raise ValueError('source TRAIN lineage changed')
                fragments[s]=fragment
                parents[s]=_mask(self.canonical_root/fragment['target_audit']['parent_mask_path'])
                offsets[s]=fragment['target_audit']['parent_to_model_offset_rc']
                self._verify_centerpad(parents[s],clean[s],offsets[s])
                provenance['groups'].append(str(path))
            same_group=provenance['groups'][0]==provenance['groups'][1]
            if same_group:
                candidates=[c for c in groups['a']['candidates'] if c['pair_id']==sample.pair_id]
                if len(candidates)!=1 or bool(candidates[0]['label'])!=bool(sample.label):
                    raise ValueError('original same-group pair/nonpair identity unavailable')
            elif sample.label or row['fragment_a']['split_unit_id']==row['fragment_b']['split_unit_id']:
                raise ValueError('cross negative requires distinct original frozen TRAIN lineage')
            if sample.label:
                normalized,ownership=normalize_boundary_ownership(self._parents(groups['a']))
                source={s:normalized[fragments[s]['fragment_id']] for s in 'ab'}
                provenance.update(ownership_reference_only=True,ownership_normalization=ownership)
                allowance=3. if ownership['applied'] else 0.
            else:
                source=parents
                allowance=0.
                provenance['negative_origin']=row.get('negative_origin')
        provenance['parent_to_model_offsets']=offsets
        return sample,report,source,offsets,clean,allowance,provenance

    def sample_at(self,index,cap):
        sample,report,source,offsets,clean,allowance,provenance=self.resolve(index)
        result,derived,views=resample_source_density(sample,report,source,offsets,clean,
            cap=cap,ownership_allowance_px=allowance)
        derived['source_resolution']=provenance
        return result,derived,views

    def stratified_indices(self,per_family_scan=128):
        """Read only bounded report metadata; never rank by model performance."""
        if not 1 <= per_family_scan <= 256:
            raise ValueError('pilot per-family scan must be1..256')
        def family(entry):
            s=entry['source_stratum']
            return 'cross_negative' if s.startswith('cross_negative_') else s
        selected,examined={},{}
        families={'native_positive','union_positive_tiny','native_hard_negative','cross_negative'}
        for index,entry in enumerate(self.entries):
            f=family(entry)
            if f not in families or examined.get(f,0)>=per_family_scan or all((f,d) in selected for d in (0.,2.,4.)):
                continue
            examined[f]=examined.get(f,0)+1
            with np.load(self.artifact_root/entry['artifact_path'],allow_pickle=False) as archive:
                report=json.loads(str(archive['report_json'].item()))
            depth=max(float(report['side_'+s]['config']['max_depth_px']) if report['side_'+s]['effective_applied'] else 0.
                      for s in 'ab')
            if depth not in (0.,2.,4.):raise ValueError('unexpected Full24 effective damage strength')
            selected.setdefault((f,depth),index)
        return [dict(family=f,effective_max_depth_px=d,index=i) for (f,d),i in sorted(selected.items())],examined

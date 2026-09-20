"""Real-label clean SIM VAL/TEST with cap-specific dense source supervision.

This is an evaluation dataset, not training augmentation. Every sample keeps
the original split label, physical masks and GT pose. Source-cell ancestry is
used offline for assignment targets; it never enters model input fields.
Optional separate disk caches avoid recomputing source targets every epoch.
"""
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import scipy

from .rachel_training_dataset import RachelPairDataset
from .rachel_source_density_records import SourceDensityRecordResolver,_mask
from .rachel_source_density import resample_source_density
from .rachel_union_augmentation import normalize_boundary_ownership
from .rachel_materialized_dataset import load_sample,save_sample
from .rachel_paired_density_dataset import file_sha256,validate_density_sample

SCHEMA='rachel-clean-source-density-eval/1'


class CleanSourceDensityDataset(SourceDensityRecordResolver):
    """RachelPairSample-compatible, real-label cap-aware VAL or TEST only."""
    prototype_only=False

    def __init__(self,root,split,contour_cap=512,cache_dir=None):
        if split not in ('val','test'):raise ValueError('clean density accepts only original SIM val/test')
        if contour_cap not in (512,1024):raise ValueError('clean density cap must be512 or1024')
        self.root=Path(root).resolve(strict=True);self.canonical_root=self.root
        self.split=split;self.contour_cap=contour_cap
        self.base=RachelPairDataset(self.root,split)
        self.manifest_path=self.root/'pairs'/(split+'.jsonl')
        self.rows=[json.loads(line) for line in self.manifest_path.read_text().splitlines() if line.strip()]
        if len(self.rows)!=len(self.base) or any(r['split']!=split for r in self.rows):
            raise ValueError('source split population differs from original runtime loader')
        self.entries=self.rows;self._groups={};self._unions={}
        from . import rachel_source_density,rachel_source_density_records,rachel_preprocess,rachel_union_augmentation
        modules=(rachel_source_density,rachel_source_density_records,rachel_preprocess,rachel_union_augmentation)
        identity=dict(schema_version=SCHEMA,root=str(self.root),split=split,contour_cap=contour_cap,
            python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            source_manifest_sha256=file_sha256(self.manifest_path),source_pair_count=len(self.rows),
            code_sha256={m.__name__:file_sha256(m.__file__) for m in modules},reader_sha256=file_sha256(__file__),
            pair_labels='unchanged original split labels',pose_targets='unchanged original split GT',
            assignment_targets='fresh exact clean source-cell ancestry, not old512 token indices',
            checkpoint_threshold_selection_permitted=(split=='val'),test_used_for_selection=False,
            smoothing_sigma=3.,weathering_applied=False)
        self.identity=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.protocol=dict(identity,identity_sha256=self.identity)
        self.stats=dict(positive=sum(bool(r['label']) for r in self.rows),negative=sum(not r['label'] for r in self.rows))
        self.cache_dir=Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir:
            if self.cache_dir==self.root or self.root in self.cache_dir.parents:
                raise ValueError('evaluation cache must not modify the original release')
            self.cache_dir.mkdir(parents=True,exist_ok=True)
            path=self.cache_dir/'cache_identity.json'
            if path.exists():
                if json.loads(path.read_text())!=self.protocol:raise ValueError('cache split/cap/source identity differs')
            else:
                temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(self.protocol,indent=2)+'\n')
                os.replace(temporary,path)

    def __len__(self):return len(self.rows)

    def _resolve_clean(self,index):
        row=self.rows[index];sample=self.base[index]
        if sample.pair_id!=row['pair_id'] or bool(sample.label)!=bool(row['label']):
            raise ValueError('original evaluation label/identity changed')
        clean={s:np.asarray(getattr(sample,'mask_'+s),bool).squeeze() for s in 'ab'}
        groups,fragments,parents,offsets={},{},{},{}
        paths=[]
        for s in 'ab':
            fragment=row['fragment_'+s];parts=fragment['fragment_token'].split('/')
            if len(parts)!=4 or parts[0]!='rachel':raise ValueError('unknown clean source token')
            group,path=self._group(parts[1],parts[2]);groups[s]=group;paths.append(str(path))
            origin=next(f for f in group['fragments'] if f['fragment_token']==fragment['fragment_token'])
            if origin['split_unit_id']!=fragment['split_unit_id']:raise ValueError('source lineage differs')
            fragments[s]=origin
            parents[s]=_mask(self.root/origin['target_audit']['parent_mask_path'])
            offsets[s]=origin['target_audit']['parent_to_model_offset_rc']
            self._verify_centerpad(parents[s],clean[s],offsets[s])
        if paths[0]==paths[1]:
            candidates=[c for c in groups['a']['candidates'] if c['pair_id']==sample.pair_id]
            if len(candidates)!=1 or bool(candidates[0]['label'])!=bool(sample.label):
                raise ValueError('original clean neighbor/nonneighbor label not backed by source group')
        elif sample.label or row['fragment_a']['split_unit_id']==row['fragment_b']['split_unit_id']:
            raise ValueError('cross negative must have different source lineage')
        if sample.label:
            normalized,ownership=normalize_boundary_ownership(self._parents(groups['a']))
            source={s:normalized[fragments[s]['fragment_id']] for s in 'ab'}
            allowance=3. if ownership['applied'] else 0.
        else:
            source=parents;ownership=None;allowance=0.
        count=int(np.count_nonzero(sample.target_a>=0))
        report=dict(schema_version=SCHEMA,pair_id=sample.pair_id,source_split=self.split,
            changed_pair=False,changed_a=False,changed_b=False,weathering_applied=False,
            side_a={'effective_applied':False},side_b={'effective_applied':False},
            pose_supervision_enabled=bool(sample.label),original_gt_translation_preserved=True,
            effective_supervised_match_count=count,inherited_match_count=count,
            ignored_token_count=int(np.count_nonzero(sample.target_a==-2)+np.count_nonzero(sample.target_b==-2)),
            inheritance_rule='original clean512 targets used only as an audit count; never reindexed',
            source_resolution=dict(groups=paths,parent_to_model_offsets=offsets,
                ownership_reference_only=True,ownership_normalization=ownership,
                label_origin=row['label_origin'],negative_origin=row.get('negative_origin')))
        return sample,report,source,offsets,clean,allowance

    def weathered(self,index):
        """Compatibility accessor; returned report explicitly says no weathering."""
        row=self.rows[index]
        path=(self.cache_dir/(hashlib.sha256(row['pair_id'].encode()).hexdigest()+'.npz')) if self.cache_dir else None
        if path and path.exists():
            result,report=load_sample(path)
            if result.pair_id!=row['pair_id'] or bool(result.label)!=bool(row['label']):
                raise ValueError('cached evaluation labels changed')
            validate_density_sample(result,report,self.contour_cap,self.identity)
            return result,report
        original,report,source,offsets,clean,allowance=self._resolve_clean(index)
        result,derived,_=resample_source_density(original,report,source,offsets,clean,
            cap=self.contour_cap,ownership_allowance_px=allowance)
        derived['paired_density']=dict(identity_sha256=self.identity,source_pair_id=result.pair_id,
                                       source_manifest=str(self.manifest_path),source_split=self.split)
        for field in ('mask_a','mask_b','coarse_mask_a','coarse_mask_b','label','translation_a_to_b_rc',
                      'translation_a_to_b_xy_cartesian','translation_valid'):
            if not np.array_equal(getattr(original,field),getattr(result,field),equal_nan=True):
                raise ValueError('clean original field changed: '+field)
        validate_density_sample(result,derived,self.contour_cap,self.identity)
        if path:save_sample(path,result,derived)
        return result,derived

    def __getitem__(self,index):return self.weathered(index)[0]
    def get_report(self,index):return self.weathered(index)[1]
    def set_epoch(self,epoch):
        """Deterministic clean evaluation, no augmentation redrawing."""

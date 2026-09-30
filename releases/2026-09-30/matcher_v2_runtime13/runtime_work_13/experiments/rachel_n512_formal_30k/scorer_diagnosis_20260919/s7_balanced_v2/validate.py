"""Validate realized quotas, negative lineage labels and both actual loaders."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedWeatheredDataset
from ..seam_context_v3.data import Dataset,collate
from ..s7_compound_v1.materialize import read,save_json
from .geometry import CORROSION_PERCENT,EIGHT


def validate(root):
    root=Path(root);status=read(root/'status.json')
    if status['status'] not in ('complete','pilot_complete'):
        raise ValueError('unfinished data cannot be validated as ready')
    old=MaterializedWeatheredDataset(status['manifest']);new=Dataset(root/'train.json')
    conservative=old.protocol.get('distribution_revision',{}).get('conservative_v12',False)
    layered=old.protocol.get('distribution_revision',{}).get('layered_damage_exclusive',False)
    n=len(old)//2;assert len(old)==len(new)==status['sample_count']
    assert Counter(e['label'] for e in old.entries)=={True:n,False:n}
    for label in (True,False):
        ee=[e for e in old.entries if e['label']==label]
        assert Counter(e['corrosion_recipe'] for e in ee)=={k:n*p//100 for k,p in old.protocol['recipe_percent'].items()}
        assert Counter(e['corrosion_category'] for e in ee)=={k:n*v//100 for k,v in old.protocol['corrosion_categories'].items()}
        assert sum(e['partial_applied'] for e in ee)==n*old.protocol.get('partial_percent',70)//100
        if layered:
            assert all(e['partial_applied']==(e['corrosion_recipe']=='partial') for e in ee)
            assert sum(e['corrosion_recipe']=='clean' for e in ee)==n*15//100
            assert sum(e['corrosion_recipe'] not in ('clean','partial') for e in ee)==n*60//100
        assert Counter(e['offline_paired_mirror'] for e in ee)=={None:n*85//100,'horizontal':n*75//1000,'vertical':n*75//1000}
    negative=[e for e in old.entries if not e['label']]
    assert len({e['source_pair_id'] for e in negative})==n
    if layered and n==400:
        assert Counter(e['negative_kind'] for e in negative)=={'same_parent_nonadjacent':120,'cross_parent_same_gen':140,'cross_gen':140}
        groups=Counter(e['anchor_group_id'] for e in negative if e['anchor_group_id'])
        assert len(groups)==66 and set(groups.values())=={3}
        assert Counter(e['source_stratum'] for e in old.entries if e['label'])=={
            'native_positive':300,'gen5_partition_positive':40,'union_positive_tiny':60}
        assert sum(e['source_stratum'].startswith('gen5_partition') for e in negative)==40
    if n==12000:
        assert Counter(e['negative_kind'] for e in negative)=={'same_parent_nonadjacent':3600,'cross_parent_same_gen':4200,'cross_gen':4200}
        groups=Counter(e['anchor_group_id'] for e in negative if e['anchor_group_id'])
        assert len(groups)==2000 and set(groups.values())=={3}
        for group in groups:
            rr=[e for e in negative if e['anchor_group_id']==group]
            assert len({e['anchor_fragment_token'] for e in rr})==1
            assert len({e['source_row']['fragment_b']['fragment_token'] for e in rr})==3
        expected=read(old.protocol['options']['sources'])['stats']['positive_strata']
        assert Counter(e['source_stratum'] for e in old.entries if e['label'])==expected
        assert sum(e['source_stratum'].startswith('gen5_partition') for e in negative)==1200
    checked=[];buckets={}
    for i,e in enumerate(old.entries):
        buckets.setdefault((e['corrosion_recipe'],e['partial_applied'],e['label'],e['offline_paired_mirror']),i)
    for i in buckets.values():
        sample,report=old[i];copy,report2,metadata,entry=new[i]
        assert sample.pair_id==copy.pair_id==old.entries[i]['pair_id'] and report==report2
        detail=report['compound'];recipe=old.entries[i]['corrosion_recipe']
        if conservative:
            from .conservative_weather import parts
            major,weak=parts(recipe)
            assert detail['corrosion_stage_count']==int(major is not None)+int(weak)+int(layered and recipe=='partial')
            assert detail['major_corrosion_count']<=1
        else:
            assert detail['corrosion_stage_count']==(0 if recipe=='clean' else 3 if recipe=='wave_local_gaps' else 2 if recipe in ('wave_gaps','wave_local') else 1)
        if recipe in ('clean','partial'):
            assert not detail['damage']
            if recipe=='partial':
                assert layered and detail['partial']['applied'] and not detail['weak_overlay']
        else:
            assert sum(d.get('removed_area_px',0) for d in detail['damage'].values())>0
        for side in 'ab':
            target=getattr(sample,'target_'+side);points=getattr(sample,'points_rc_'+side)
            assert np.array_equal(points,getattr(copy,'points_rc_'+side)) and np.isfinite(points).all()
            assert ndimage.label(getattr(sample,'mask_'+side)[0],EIGHT)[1]==1
            assert (target>=0).sum()>=4 if sample.label else not (target>=0).any()
            assert np.all(metadata['component_'+side][target<0]==-1)
            gap=metadata['gap_'+side];assert np.array_equal(gap,gap.T) and not gap[target<0].any()
        assert collate([new[i]])['gap_a'].shape==(1,512,512)
        checked.append(sample.pair_id)
    rows=[json.loads(l) for l in (root/'pair_metrics.jsonl').read_text().splitlines()]
    pos=[r for r in rows if r['label']]
    length_key='quota_length_px' if old.protocol.get('length_quota_basis')=='pre_weather_near_length' else 'd20_length_px'
    assert len(rows)==2*n and all(np.isfinite(r[length_key]) and r[length_key]>0 for r in pos)
    if old.protocol.get('length_quota_enforced',True):assert all(32<=r[length_key]<=800 for r in pos)
    if old.protocol.get('distribution_revision',{}).get('record_latent_seam'):
        assert all(r['latent_seam'] is not None for r in pos)
        if old.protocol['distribution_revision'].get('heterogeneous_strong'):
            assert all(r['latent_seam']['has_near_and_far'] for r in pos if 'wave' in r['recipe'])
        assert all((root/e['latent_seam_artifact']).is_file() for e in old.entries if e['label'])
    if old.protocol.get('length_quota_enforced',True):
        assert Counter(r['actual_length_bin'] for r in pos)=={'short':n//4,'medium':n//2,'long':n//4}
    elif old.protocol.get('length_quota_scope')=='nonpartial_only':
        ordinary=[r for r in pos if not r['partial']]
        assert all(32<=r[length_key]<=800 and r['actual_length_bin']==r['target_length_bin'] for r in ordinary)
        assert Counter(r['actual_length_bin'] for r in ordinary)==old.protocol['nonpartial_length_counts']
        assert all(r['target_length_bin'] is None for r in pos if r['partial'])
    if conservative:
        for r in rows:
            dd=[d for d in r['augmentation']['damage'].values() if d.get('applied')]
            assert len(dd)==(r['recipe'] not in ('clean','partial'))
            for d in dd:
                assert 0<d['applied_max_depth_px']<=9.
                assert 0<d['affected_fraction']<=.5+1e-9
                assert d['no_major_corrosion_combination']
                if 'gaps' in r['recipe']:assert 1<=d['notch_count']<=3
                if d['weak_applied']:
                    assert 1<=d['weak_peak_px']<=4 and d['weak_independently_removed_pixels']>=4
            assert r['augmentation']['major_corrosion_count']<=1
        assert all((root/e['weather_artifact']).is_file() for e in old.entries)
    if layered:
        for r in rows:
            layers=r['augmentation_layers']
            assert layers['order']==['source_structure','common_scale','paired_mirror','exclusive_damage']
            assert layers['mirror_before_damage'] and layers['no_primary_weather_after_partial']
            assert layers['corrosion']['partial']==r['partial']==(r['recipe']=='partial')
            assert layers['fragments']['offline_paired_mirror']==r['mirror']
            if r['partial']:
                assert not r['augmentation']['damage'] and r['augmentation']['corrosion_types']==['partial']
                assert not r['augmentation']['weak_overlay'] and r['augmentation']['major_corrosion_count']==0
                if old.protocol['distribution_revision'].get('partial_min_smaller_perimeter_fraction'):
                    detail=r['augmentation']['partial']
                    assert detail['mode'] in ('end','middle')
                    assert detail['support_constraint']['common_over_smaller_perimeter']>=.15
                    if detail['mode']=='middle':assert min(detail['retained_flanks_px'])>=16.
            for side in 'ab':
                assert r['fragment_stage_area_px'][side]>=r['damage_stage_area_px'][side]
            micro=r['background_degradation']
            assert bool(micro)==(r['recipe']!='clean')==layers['corrosion']['background_light']
            if r['recipe']=='clean':
                assert r['fragment_stage_area_px']==r['damage_stage_area_px']
            else:
                for d in micro.values():
                    assert abs(d['actual_affected_fraction']-.70)<=.02
                    assert 0<d['applied_max_depth_px']<=3.
                    assert not d['gt_used_for_placement']
                    assert d['minimum_ratio_relative']==.95
                    assert d['pair_area_ratio_after']>=d['pair_area_ratio_floor']
                assert set(micro)==set('ab')
        for e in old.entries:
            with np.load(root/e['weather_artifact'],allow_pickle=False) as z:
                assert str(z['coordinate_frame'])=='post_fragment_pre_damage_800px'
                if e['partial_applied']:assert not any(s+'_total' in z for s in 'ab')
            if e['corrosion_recipe']=='clean':assert not e.get('background_artifact')
            else:assert (root/e['background_artifact']).is_file()
        modes=old.protocol['distribution_revision'].get('partial_mode_percent')
        if modes:
            for label in (True,False):
                actual=Counter(r['augmentation']['partial']['mode'] for r in rows if r['partial'] and r['label']==label)
                assert actual=={k:n*.25*v/100 for k,v in modes.items()}
    result=dict(status='passed',pairs=2*n,positives=n,negatives=n,
        actual_loader_combinations_checked=len(checked),checked_pair_ids=checked,
        all_archives_roundtrip_checked_during_generation=True,
        quotas_exact=True,source_identity_and_anchor_quotas_checked=n==12000 or layered,
        original_data_or_training_modified=False,training_started=False,
        layered_damage_exclusive_checked=layered,mirror_before_damage_checked=layered)
    save_json(root/'validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checked_pair_ids'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    validate(p.parse_args().root)

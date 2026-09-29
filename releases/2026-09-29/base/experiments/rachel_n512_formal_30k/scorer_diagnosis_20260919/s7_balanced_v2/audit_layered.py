"""Independent raster reconstruction of every pilot pair and both damage stages."""
import argparse
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from ..s7_compound_v1.materialize import read,save_json

EIGHT=np.ones((3,3),bool)


def archived_depth(data,side):
    """Use the exact rasterizing field, never add an audit epsilon.

    Historical archives stored only float32 display fields. New archives also
    preserve the original float64 raster field. Keeping the old array unchanged
    avoids changing downstream eligibility decisions or the approved samples.
    """
    shown=data[side+'_total']
    if side+'_total_exact' not in data:
        return shown
    exact=data[side+'_total_exact']
    if (exact.dtype!=np.dtype('float64') or exact.shape!=shown.shape
            or not np.isfinite(exact).all()
            or not np.array_equal(exact.astype(shown.dtype),shown)):
        raise ValueError('stored exact depth field does not match its display field')
    return exact


def reconstruct(mask,points,field,maximum):
    filled=ndimage.binary_fill_holes(np.pad(mask,1),structure=EIGHT)
    depth=ndimage.distance_transform_edt(filled)[1:-1,1:-1]-.5
    band=np.argwhere(mask&(depth<=maximum));index=cKDTree(points).query(band)[1]
    removed=depth[tuple(band.T)]<=field[index]
    result=mask.copy();result[tuple(band[removed].T)]=False
    return result


def one(task):
    root,entry=task;root=Path(root);sample,report=load_sample(root/entry['artifact_path'])
    recipe=entry['corrosion_recipe'];coverage=[];material=0;area_before=[];area_after=[]
    with np.load(root/entry['weather_artifact'],allow_pickle=False) as data:
        main={k:data[k] for k in data.files}
    assert str(main['coordinate_frame'])=='post_fragment_pre_damage_800px'
    light={}
    if entry.get('background_artifact'):
        with np.load(root/entry['background_artifact'],allow_pickle=False) as data:
            light={k:data[k] for k in data.files}
        assert str(light['coordinate_frame'])=='post_primary_pre_background_800px'
    assert bool(light)==(recipe!='clean')
    for side in 'ab':
        original=np.unpackbits(main['packed_preweather_'+side],axis=1).astype(bool)
        final=getattr(sample,'mask_'+side)[0].astype(bool)
        middle=np.unpackbits(light['packed_before_'+side],axis=1).astype(bool) if light else final
        area_before.append(int(middle.sum()));area_after.append(int(final.sum()))
        assert not np.any(final&~middle) and not np.any(middle&~original)
        if recipe=='partial':
            assert side+'_total' not in main and not report['compound']['damage']
        elif side+'_total' in main:
            predicted=reconstruct(original,main[side+'_points'],archived_depth(main,side),9.)
            assert np.array_equal(predicted,middle),('primary raster mismatch',entry['pair_id'],side)
        else:assert np.array_equal(original,middle)
        if light:
            field=archived_depth(light,side);points=light[side+'_points'];eligible=light[side+'_eligible'];edge=light[side+'_edge']
            assert np.all(np.isfinite(field)) and 0<float(field.max())<=3.
            assert not np.any(field[~eligible])
            predicted=reconstruct(middle,points,field,3.)
            assert np.array_equal(predicted,final),('light raster mismatch',entry['pair_id'],side)
            ip=np.rint(points).astype(int);removed=middle[tuple(ip.T)]&~final[tuple(ip.T)]
            valid=eligible&np.roll(eligible,-1)
            ratio=float(edge[valid&removed&np.roll(removed,-1)].sum()/edge[valid].sum())
            assert abs(ratio-.70)<=.02001
            assert abs(ratio-report['background_degradation'][side]['actual_affected_fraction'])<1e-5
            # Eligible points must survive unchanged from the original boundary.
            from .conservative_weather import contour
            original_points=contour(original)[0]
            assert np.all(cKDTree(original_points).query(points[eligible])[0]<.25)
            if side+'_total' in main:
                distance,index=cKDTree(main[side+'_points']).query(points[eligible])
                assert not np.any((distance<.25)&(main[side+'_total'][index]>0.))
            coverage.append(ratio)
        assert ndimage.label(final,EIGHT)[1]==1
        assert not np.any(ndimage.binary_fill_holes(final,structure=EIGHT)&~final&original)
        material+=int(original.sum()-final.sum())
    if recipe=='clean':assert material==0 and not report['background_degradation']
    partial_constraint=None
    if recipe=='partial' and 'partial_original_points_rc_a' in main:
        from .partial_v14 import retained_support
        updates={name:main['partial_original_'+name] for name in (
            'points_rc_a','points_rc_b','contour_valid_a','contour_valid_b','target_a','target_b')}
        updates.update({f'mask_{s}':np.unpackbits(main['packed_preweather_'+s],axis=1).astype(np.float32)[None] for s in 'ab'})
        original=replace(sample,**updates)
        cropped=replace(sample,**{f'mask_{s}':np.unpackbits(light['packed_before_'+s],axis=1).astype(np.float32)[None] for s in 'ab'})
        detail=report['compound']['partial']
        if detail['mode']=='middle':
            side=detail['sides'][0 if sample.label else 1]
            predicted=reconstruct(getattr(original,'mask_'+side)[0].astype(bool),
                main['partial_cut_points'],main['partial_cut_field'],float(main['partial_cut_field'].max()))
            assert np.array_equal(predicted,getattr(cropped,'mask_'+side)[0].astype(bool))
            assert min(detail['retained_flanks_px'])>=16. and detail['removed_middle_arc_px']>0
        if sample.label:
            partial_constraint,_=retained_support(original,cropped)
            assert partial_constraint['common_over_smaller_perimeter']>=.15
            for key in ('common_retained_length_px','common_over_smaller_perimeter'):
                assert abs(partial_constraint[key]-detail['support_constraint'][key])<1e-6
            if detail['mode']=='middle':
                from .latent_seam import source_band
                points,edge,eligible=source_band(original,bridge=0.)[side]
                arc=np.r_[0.,np.cumsum(edge[:-1])];perimeter=float(edge.sum())
                proposal=detail['proposal'];length=detail['selected_run_length_px']
                inside=eligible&((arc-proposal['run_start_arc_px'])%perimeter<length)
                relative=(arc-proposal['center_arc_px']+perimeter/2)%perimeter-perimeter/2
                pixels=np.rint(points).astype(int)
                survive=getattr(cropped,'mask_'+side)[0][tuple(pixels.T)]>.5
                flanks=[float(edge[inside&survive&(relative<-proposal['width_px']/2)].sum()),
                        float(edge[inside&survive&(relative>proposal['width_px']/2)].sum())]
                removed=float(edge[inside&~survive].sum())
                assert np.allclose(flanks,detail['retained_flanks_px'],atol=1e-6)
                assert abs(removed-detail['removed_middle_arc_px'])<1e-6
                assert min(flanks)>=max(16.,.08*length) and removed>=.20*length
    if light:
        r0=min(area_before)/max(area_before);r1=min(area_after)/max(area_after)
        floor=min(.125,r0)*.95
        assert r1>=floor
        for d in report['background_degradation'].values():
            assert abs(d['pair_area_ratio_before']-r0)<1e-9
            assert abs(d['pair_area_ratio_after']-r1)<1e-9
            assert abs(d['pair_area_ratio_floor']-floor)<1e-9
    assert (sample.target_a>=0).sum()>=4 if sample.label else not (sample.target_a>=0).any()
    return dict(pair_id=entry['pair_id'],recipe=recipe,coverage=coverage,removed_pixels=material,
                partial_constraint=partial_constraint,
                raster_depth_lossless=all(side+'_total_exact' in data
                    for data in (main,light) for side in 'ab' if side+'_total' in data))


def run(root,workers):
    root=Path(root);manifest=read(root/'train_s7b_24k.json')
    if not manifest['protocol']['distribution_revision'].get('layered_damage_exclusive'):
        raise ValueError('layered data required')
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(one,[(str(root),e) for e in manifest['entries']],chunksize=4))
    coverage=[v for r in rows for v in r['coverage']]
    partial=[r['partial_constraint'] for r in rows if r['partial_constraint'] is not None]
    result=dict(status='passed',pairs=len(rows),all_actual_masks_reconstructed=True,
        background_fragments=len(coverage),actual_coverage_min=min(coverage),actual_coverage_max=max(coverage),
        actual_coverage_mean=float(np.mean(coverage)),primary_and_background_separately_reconstructed=True,
        clean_unchanged=True,partial_no_primary_weather=True,partial_new_cuts_excluded_from_light=True,
        topology_checked_all=True,no_new_positive_labels_invented_by_audit=True,
        raster_depth_lossless_all=all(r['raster_depth_lossless'] for r in rows),
        label_check='positive>=4 inherited correspondences; negative has no positive targets',
        partial_constraint_count=len(partial),
        partial_smaller_perimeter_ratio_min=min((r['common_over_smaller_perimeter'] for r in partial),default=None),
        rows=rows)
    save_json(root/'independent_pixel_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();run(a.root,a.workers)

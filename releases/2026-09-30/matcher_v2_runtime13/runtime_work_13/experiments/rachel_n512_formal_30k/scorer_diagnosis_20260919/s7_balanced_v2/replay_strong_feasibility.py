"""Replay exact data-slot attempts without exporting training examples.

Record old-seam placement capacity and requested notch plans. This diagnostic
does not change supervision, masks on disk, source plans, or validation rules.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
from . import materialize as m
from . import geometry as g
from ..s7_compound_v1 import geometry as weather
from ..s7_compound_v1.materialize import read, save_json


def initialize(options):
    m.initialize(options)


def replay(job):
    index,attempt,source_id=job
    s=m.STATE;o=s['options'];wanted=s['bins'][index]
    negative,_=m.negative_source(s['plan']['negative'][index])
    positive=m.clean_positive(source_id)
    arng=g.rng_for(o['seed'],g.SCHEMA,'fixed-area',index)
    q=s['profile']['conditional_area_deciles_px2'][wanted]
    area=float(np.interp(arng.random(),np.linspace(0,1,len(q)),q))
    seed=int(g.rng_for(o['seed'],g.SCHEMA,index,attempt).integers(0,2**31))
    trace=[];original=g.damaged_pair

    def observed(sample,rng,plan,eligible_fractions=None):
        record=dict(label=bool(sample.label),endpoints=plan['endpoints'],sides={},
                    preweather_targets=int((sample.target_a>=0).sum()),ancestry={})
        if sample.label:
            for side in plan['endpoints']:
                mask=getattr(sample,'mask_'+side)[0]
                dense,_=weather.extract_ordered_outer_contour(mask,cap=mask.size,smoothing_sigma=0.)
                edge=np.linalg.norm(np.roll(dense,-1,axis=0)-dense,axis=1)
                arc=np.r_[0.,np.cumsum(edge[:-1])]
                bits=weather.eligible_seam(sample,side,dense,plan.get('dense_weather_scope',False),
                    cut_guard=plan.get('positive_cut_guards',{}).get(side),
                    cut_guard_radius=plan.get('positive_cut_guard_radius',8.),
                    contact_tolerance=plan.get('positive_contact_tolerance',3.),
                    bridge_short_gaps_px=plan.get('bridge_short_gaps_px',0.))
                runs=weather._eligible_runs(bits,arc,edge)
                widths=plan['widths'][side]
                record['sides'][side]=dict(eligible_arc_px=float(edge[bits].sum()),
                    run_lengths_px=[float(r[1]) for r in runs],widths_px=widths,peaks_px=plan['peaks'][side],
                    total_requested_width_px=float(sum(widths)),
                    impossible_single_width=any(w>max([r[1] for r in runs],default=0.) for w in widths))
        changed_view=weather._changed_view
        def observe_view(old,side,mask,info,recipe):
            view=changed_view(old,side,mask,info,recipe)
            targets=getattr(old,'target_'+side)
            ids=np.flatnonzero(targets>=0)
            mapped=view.representatives[ids]>=0
            row=dict(old_matched=len(ids),matched_representatives_after_notches=int(mapped.sum()),
                     matched_ids=ids.tolist(),retained_matched_ids=ids[mapped].tolist(),
                     trusted_representatives=int((view.representatives>=0).sum()))
            if view.changed:
                from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import source_arc_ancestry
                _,reps,projection=source_arc_ancestry(getattr(old,'mask_'+side)[0],
                    getattr(old,'points_rc_'+side),getattr(old,'contour_valid_'+side),view.points,30.)
                row.update(matched_representatives_before_notches=int((reps[ids]>=0).sum()),
                           before_notch_ids=ids[reps[ids]>=0].tolist(),projection=projection)
            record['ancestry'][side]=row
            return view
        weather._changed_view=observe_view
        try:output=original(sample,rng,plan,eligible_fractions)
        finally:weather._changed_view=changed_view
        record['accepted']=output[0] is not None
        record['reason']=None if output[0] is not None else output[1]
        trace.append(record)
        return output

    g.damaged_pair=observed
    try:
        rows,detail=g.augment_group(positive,negative,s['recipes'][index],s['partial'][index],
            s['bank'],seed,s['profile'],area,quota_bin=wanted)
    finally:g.damaged_pair=original
    result=dict(slot=index,attempt=attempt,source_pair_id=source_id,recipe=s['recipes'][index],
        partial=s['partial'][index],length_bin=wanted,target_area=area,trace=trace,accepted=rows is not None)
    if rows is None:result['rejection']=detail
    else:
        sample,report,before=rows[0]
        result.update(inherited=int((sample.target_a>=0).sum()),
            weather=report['compound']['damage'],
            weather_plan_selection=report['compound'].get('weather_plan_selection'),
            latent_seam=report.get('latent_seam'),
            negative_inherited=int((rows[1][0].target_a>=0).sum()),
            components=[int(weather.ndimage.label(mask,weather.EIGHT)[1])
                        for row in rows for mask in m.masks(row[0])],
            preweather_length=m.pair_metrics(*[m.outline(x) for x in m.masks(before)],before.translation_a_to_b_rc)['d20_length_px'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--slots',default='93')
    p.add_argument('--attempt-start',type=int,default=0);p.add_argument('--attempt-count',type=int,default=128)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--profile');p.add_argument('--out',required=True)
    a=p.parse_args();o=read(Path(a.root)/'protocol.json')['options']
    if a.profile:o['profile']=a.profile
    plan=read(o['sources']);jobs=[]
    for index in map(int,a.slots.split(',')):
        original=plan['positive'][index]
        alternatives=[x for x in plan['positive'] if x['source_stratum']==original['source_stratum']]
        rng=g.rng_for(o['seed'],g.SCHEMA,'replacement',index)
        for attempt in range(a.attempt_start+a.attempt_count):
            source=original if attempt<4 else alternatives[int(rng.integers(len(alternatives)))]
            if attempt>=a.attempt_start:jobs.append((index,attempt,source['pair_id']))
    output=Path(a.out)
    if output.exists():raise ValueError('do not overwrite diagnostic evidence')
    with ProcessPoolExecutor(a.workers,initializer=initialize,initargs=(o,)) as pool:
        results=list(pool.map(replay,jobs))
    save_json(output,dict(root=a.root,profile=o.get('profile'),training_exported=False,results=results))
    print(json.dumps(dict(attempts=len(results),accepted=sum(x['accepted'] for x in results),
        weather_reached=sum(bool(x['trace']) for x in results),output=str(output))))


if __name__=='__main__':main()

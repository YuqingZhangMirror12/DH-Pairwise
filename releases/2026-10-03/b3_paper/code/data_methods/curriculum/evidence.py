"""Observed GT evidence islands, never a prediction from the current Matcher."""
import numpy as np
from scipy.spatial import cKDTree
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import source_arc_ancestry


def components(points,weights,keep):
    if not len(points) or not keep.any():return []
    distances=np.linalg.norm(points-np.roll(points,1,axis=0),axis=1)
    # Start after the largest source-arc discontinuity, preserving cyclic order.
    order=np.roll(np.arange(len(points)),-int(np.argmax(distances)))
    result=[];current=[]
    for i in order:
        if (not keep[i]) or (current and distances[i]>3.):
            if current:result.append(current);current=[]
        if keep[i]:current.append(int(i))
    if current:result.append(current)
    return [dict(indices=ids,length_px=float(weights[ids].sum())) for ids in result]


def source_islands(base,final,gap):
    sides={}
    for side in 'ab':
        p=gap[side+'_source_points'];w=gap[side+'_source_weight']
        keep=~gap[side+'_primary_affected'] & gap[side+'_source_valid']
        raw=components(p,w,keep)
        ancestry,_,_=source_arc_ancestry(getattr(base,'mask_'+side)[0],
            getattr(base,'points_rc_'+side),getattr(base,'contour_valid_'+side),
            getattr(final,'points_rc_'+side),30.)
        ids=np.flatnonzero((getattr(final,'target_'+side)>=0)&(ancestry>=0))
        source_points=getattr(base,'points_rc_'+side)[ancestry[ids]]
        distance,nearest=cKDTree(p).query(source_points)
        observations=[]
        for c in raw:
            member=(distance<=3.) & np.isin(nearest,c['indices'])
            c.update(inherited_target_indices=ids[member].tolist(),inherited_matches=int(member.sum()),
                     source_points_rc=p[c['indices']].tolist())
            c['qualifies']=c['length_px']>=8. and c['inherited_matches']>=1
            observations.append(c)
        sides[side]=dict(count=sum(c['qualifies'] for c in observations),segments=observations)
    return dict(definition='separated original GT-supported non-primary-damaged arcs >=8px; each retains >=1 inherited positive target; no model predictions',
                sides=sides,two_to_four_bilateral=all(2<=sides[s]['count']<=4 for s in 'ab'))

"""Read/replay failed data slots; no sample export, model load, or training."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from . import materialize as m
from .geometry import SCHEMA, augment_group
from ..s7_compound_v1.geometry import rng_for, masks
from ..s7_compound_v1.materialize import read, length_bin
from ..distribution_audit_20260923.measure import outline, pair_metrics


def inspect_slot(index):
    s=m.STATE;o=s['options'];wanted=s['bins'][index]
    negative,_=m.negative_source(s['plan']['negative'][index])
    original=s['positive'][index];alternatives=s['buckets'][original['source_stratum']]
    rng=rng_for(o['seed'],SCHEMA,'replacement',index)
    arng=rng_for(o['seed'],SCHEMA,'fixed-area',index)
    q=s['profile']['conditional_area_deciles_px2'][wanted]
    area=float(np.interp(arng.random(),np.linspace(0,1,len(q)),q))
    counts=Counter();nested=Counter();lengths=[]
    for attempt in range(o['diagnose_attempts']):
        source=original if attempt<4 else alternatives[int(rng.integers(len(alternatives)))]
        positive=m.clean_positive(source['pair_id'])
        seed=int(rng_for(o['seed'],SCHEMA,index,attempt).integers(0,2**31))
        rows,detail=augment_group(positive,negative,s['recipes'][index],s['partial'][index],
            s['bank'],seed,s['profile'],area)
        if rows is None:
            stage=detail.get('stage','');d=detail.get('detail',{})
            counts[stage+':'+str(d.get('reason'))]+=1
            d=d.get('detail',d)
            nested.update({stage+':'+k:v for k,v in d.get('attempt_reasons',{}).items()})
        else:
            pm=pair_metrics(*[outline(x) for x in masks(rows[0][0])],rows[0][0].translation_a_to_b_rc)
            length=pm['d20_length_px'];lengths.append(length)
            counts['success' if 32<=length<=800 and length_bin(length)==wanted else 'length_quota']+=1
    return dict(slot=index,recipe=s['recipes'][index],partial=s['partial'][index],
        length_bin=wanted,area=area,stratum=original['source_stratum'],
        counts=dict(counts),nested=dict(nested),accepted_weather_lengths=lengths)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--attempts',type=int,default=64)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--profile');p.add_argument('--slots');a=p.parse_args()
    root=Path(a.root);o=read(root/'protocol.json')['options'];o['diagnose_attempts']=a.attempts
    if a.profile:o['profile']=a.profile
    missing=[i for i in range(o['groups']) if not (root/'groups'/('%05d.json'%i)).exists()]
    if a.slots:missing=[int(x) for x in a.slots.split(',')]
    with ProcessPoolExecutor(a.workers,initializer=m.initialize,initargs=(o,)) as pool:
        for result in pool.map(inspect_slot,missing):print(json.dumps(result),flush=True)


if __name__=='__main__':main()

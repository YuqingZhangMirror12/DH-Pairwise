"""Bounded real-source gate for both Partial modes, separate from the800 pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
from . import materialize as m
from .audit_layered import one
from ..s7_compound_v1.materialize import save_json


def check(index):
    result=m.slot(index)
    audited=[one((m.STATE['options']['out'],e)) for e in result['entries']]
    constraint=audited[0]['partial_constraint']
    return dict(index=index,mode=m.STATE['partial_modes'][index],length_bin=m.STATE['bins'][index],
        ratio=constraint['common_over_smaller_perimeter'],length=constraint['common_retained_length_px'],
        pixel_audit_passed=True,attempts=result['attempts'] if 'attempts' in result else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--seed',type=int,default=26092431)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    options=dict(sources=str(Path(a.root)/'sources_v2.json'),out=str(out),groups=400,
                 workers=6,seed=a.seed,attempts=128,profile=a.profile)
    m.initialize(options);selected={}
    for index,mode in enumerate(m.STATE['partial_modes']):
        if mode:selected.setdefault((mode,m.STATE['bins'][index]),index)
    assert len(selected)==6
    results=[]
    with ProcessPoolExecutor(6,initializer=m.initialize,initargs=(options,)) as pool:
        for future in as_completed([pool.submit(check,i) for i in selected.values()]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    save_json(out/'gate.json',dict(status='passed',groups=6,pairs=12,not_pilot=True,rows=results))


if __name__=='__main__':main()

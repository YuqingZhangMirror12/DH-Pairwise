"""Summarize actual common scale and window footprints of a completed export."""
import argparse
import json
from pathlib import Path
from .measure import read,save,quantile


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.root)
    if read(root/'status.json')['status'] not in ('complete','pilot_complete'):
        raise ValueError('do not turn a completed subset into a full distribution')
    rows=[json.loads(x) for x in (root/'pair_metrics.jsonl').read_text().splitlines()]
    report={}
    for label in (True,False):
        rr=[r['pair_shared_scale'] for r in rows if r['label']==label]
        report['positive' if label else 'negative']=dict(pairs=len(rr),
            common_scale=quantile([r['common_scale'] for r in rr]),
            requested_area=quantile([r['requested_mean_area_px2'] for r in rr]),
            clipped_fraction=sum(r['clipped'] for r in rr)/len(rr),
            topology_backoff_fraction=sum(r['topology_backoff_used'] for r in rr)/len(rr),
            windows_original_simulation_px={str(w):quantile([w/r['common_scale'] for r in rr]) for w in (7,16,32,64)})
    save(a.out,dict(source=str(root/'pair_metrics.jsonl'),
        definition='width in pre-scale synthetic canonical coordinates; not paper millimetres',**report))


if __name__=='__main__':main()

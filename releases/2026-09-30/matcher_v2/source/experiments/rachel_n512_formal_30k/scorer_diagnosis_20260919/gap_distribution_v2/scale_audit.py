"""Resolve each retained GT endpoint to its original scan and shared scale."""
import argparse
import json
from pathlib import Path
from .measure import read,save,quantile


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-manifest',required=True)
    p.add_argument('--pairs',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();cases={c['case_uid']:c for c in read(a.raw_manifest)['cases']}
    rows=[]
    for pair in [json.loads(l) for l in Path(a.pairs).read_text().splitlines()]:
        for side in 'ab':
            token=pair['fragment_'+side+'_id'];case,frag=token.split('/fragment/')
            c=cases[case];scale=800/max(c['numeric_metadata']['canvas_wh'])
            original=next(f for f in c['fragments'] if str(f['fragment_id'])==frag)
            rows.append(dict(pair_id=pair['pair_id'],fragment_id=token,case_uid=case,
                raw_area_px2=original['alpha_foreground_pixels'],model_area_px2=pair['area_px_'+side],
                shared_scale=scale,canvas_wh=c['numeric_metadata']['canvas_wh'],
                **{'window%d_raw_px'%w:w/scale for w in (7,16,32,64)}))
    save(a.out,dict(source=a.raw_manifest,rows=rows,endpoint_occurrences=len(rows),
        unique_fragments=len({r['fragment_id'] for r in rows}),
        weighting='one occurrence of each endpoint in every retained positive pair',
        model_to_scan_is_known_but_physical_DPI_is_not=True,
        summary={k:quantile([r[k] for r in rows]) for k in
                 ('shared_scale','raw_area_px2','model_area_px2','window7_raw_px','window16_raw_px','window32_raw_px','window64_raw_px')}))


if __name__=='__main__':main()

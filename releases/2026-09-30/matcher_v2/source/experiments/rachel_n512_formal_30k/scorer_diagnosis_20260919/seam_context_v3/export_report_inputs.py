"""Export original cached masks and stricter source exclusions; CPU only."""
import argparse
import base64
from io import BytesIO
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .prepare import read,save,TRAIN,sources
from .evaluate_external import CV,REAL,RELEASE,summarize


def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);fragments={};metas={}
    for split,path in [('dunhuang_cv',CV/'real/manifest.json'),('turufan',CV/'ood/manifest.json')]:
        meta=read(path);metas[split]=dict(pairs=meta['pairs'],fragment_ids=meta['fragment_ids'])
        prepared=Path(meta.get('prepared',REAL))
        with np.load(prepared/'inputs.npz',allow_pickle=False) as z:masks=z['packed_masks']
        for token,packed in zip(meta['fragment_ids'],masks):
            if token in fragments:continue
            mask=np.unpackbits(packed,axis=-1).astype(bool);yy,xx=np.nonzero(mask)
            if not len(yy):raise ValueError('empty input mask')
            y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
            crop=mask[y0:y1,x0:x1];rgba=np.full(crop.shape+(4,),255,np.uint8);rgba[...,3]=crop*255
            stream=BytesIO();Image.fromarray(rgba).save(stream,format='PNG',optimize=True)
            fragments[token]=dict(bbox_xywh=[x0,y0,x1-x0,y1-y0],area=int(mask.sum()),
                png='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode())
    save(out/'masks_and_pairs.json',dict(preprocessing='original frozen800 cache; no new smoothing/scale changes',fragments=fragments,manifests=metas))
    root=Path(a.evaluation);rr=[json.loads(x) for x in (root/'sim_test/case_diagnostics.jsonl').read_text().splitlines()]
    train_sources=set().union(*(sources(e['source_row']) for e in read(TRAIN)['entries']))
    cal=set().union(*(set(e['sources']) for v in ('clean','hard') for e in read(Path(a.data)/('cal_'+v+'.json'))['entries']))
    select=set().union(*(set(e['sources']) for v in ('clean','hard') for e in read(Path(a.data)/('select_'+v+'.json'))['entries']))
    src={p['pair_id']:sources(p) for p in [json.loads(x) for x in (RELEASE/'pairs/test.jsonl').read_text().splitlines()]}
    excluded={name:{pid for pid,s in src.items() if s&seen} for name,seen in [('train',train_sources),('cal',cal),('select',select)]}
    kept=[r for r in rr if not src[r['pair_id']]&(train_sources|cal|select)]
    save(out/'sim_strict_source_disjoint.json',dict(protocol='exclude train + CAL + SELECT source families from original SIM TEST, no re-inference',
        full_count=len(rr),overlap_counts={k:len(v) for k,v in excluded.items()},strict_count=len(kept),
        metrics=summarize(kept,.47),kept_pair_ids=[r['pair_id'] for r in kept],excluded_sources=sorted(train_sources|cal|select)))
    print(dict(fragments=len(fragments),strict_sim_pairs=len(kept)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--data',required=True)
    run(p.parse_args())

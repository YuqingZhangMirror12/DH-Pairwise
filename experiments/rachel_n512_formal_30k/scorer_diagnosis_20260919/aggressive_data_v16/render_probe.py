"""Audited preview only; not misrepresented as full per-type review coverage."""
import argparse,json,base64
from pathlib import Path
from .render_review import render

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    bundle=Path(a.bundle);out=Path(a.out);m=json.loads((bundle/'manifest.json').read_text());audit=json.loads((bundle/'pixel_audit.json').read_text())
    assert audit['status']=='passed';by={r['id']:r for r in audit['receipts']};rows=[]
    for r in m['entries']:
        img=out/'images'/(r['id']+'.png');render(r,bundle,img)
        rows.append(dict(id=r['id'],label='正例' if r['label'] else '负例',recipe=r['recipe'],groups=r['groups'],
            image='data:image/png;base64,'+base64.b64encode(img.read_bytes()).decode(),trim=r['detail']['trim'],
            paired_gap=by[r['id']]['paired_gap'],old_L40=r['baseline_metrics'].get('d40_length_px'),
            new_L40=r['new_metrics'].get('d40_length_px'),inherited_correspondences=r['inherited_correspondences']))
    out.mkdir(parents=True,exist_ok=True)
    (out/'rendered.json').write_text(json.dumps(dict(groups={'preview':dict(label='已审计预览（尚非每类10例）',ids=[r['id'] for r in rows])},rows=rows),ensure_ascii=False)+'\n')
    print(json.dumps(dict(pairs=len(rows),partial_preview=True)))

if __name__=='__main__':main()

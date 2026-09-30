"""Full, bidirectional GT-aligned contour distances and actual input scale.

Do not call whole-outline distance a semantic seam. Raw boundary coordinates
are not smoothed for distance; sigma3 is used only to obtain stable normals.
Both arc-weighted and pair-equal distributions are retained.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from ..distribution_audit_20260923.measure import (
    resample, outline, pair_metrics, unpack, clean_support)

cv2.setNumThreads(1)
EDGES=np.array([0,2,4,8,15,20,30,40,64,128,256,512,1024,np.inf])
FINE=np.r_[np.arange(0,64.01,.25),128,256,512,1024,2048,np.inf]
KINDS=('all','opposed_unbounded','seam40','seam64')
Q=(0,.1,.25,.5,.75,.9,.95,.99,1)


def read(path):return json.loads(Path(path).read_text())
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def raw_outline(mask):
    contours,_=cv2.findContours(np.uint8(mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:raise ValueError('empty mask')
    p=max(contours,key=cv2.contourArea)[:,0,::-1].astype(float)
    signed=np.sum(p[:,1]*np.roll(p[:,0],-1)-np.roll(p[:,1],-1)*p[:,0])
    if signed<0:p=p[::-1]
    p,perimeter,step=resample(p)
    smooth=gaussian_filter1d(p,3,axis=0,mode='wrap')
    tangent=np.roll(smooth,-4,axis=0)-np.roll(smooth,4,axis=0)
    normals=np.c_[-tangent[:,1],tangent[:,0]]
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    return dict(points=p,normals=normals,step=step,perimeter=perimeter)


def side_distances(a,b,shift):
    p=a['points'];q=b['points']+shift
    d,j=cKDTree(q).query(p,workers=1)
    vector=q[j]-p
    opposed=(a['normals']*b['normals'][j]).sum(1)<=-.5
    exterior=((vector*a['normals']).sum(1)>=-3)&((-vector*b['normals'][j]).sum(1)>=-3)
    ok=opposed&exterior
    use={'all':np.ones(len(p),bool),'opposed_unbounded':ok}
    for limit in (40,64):
        # Do not include gap-bridged distances above the advertised bound.
        use['seam'+str(limit)]=clean_support(ok&(d<=limit),a['step'])&ok&(d<=limit)
    return d,use


def quantile(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
        **{'p'+str(int(100*q)):float(np.quantile(a,q)) if len(a) else None for q in Q})


def measured_pair(ma,mb,t,points=None,valid=None):
    smooth=[outline(m) for m in (ma,mb)]
    row=pair_metrics(*smooth,t)
    for i,s in enumerate('ab'):
        if points is not None:
            p=points[i][valid[i]];steps=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
            row['tokens_'+s]=len(p)
            row['model_step_mean_'+s]=float(steps.mean())
            row['model_step_p90_'+s]=float(np.quantile(steps,.9))
        row['window32_over_sqrt_area_'+s]=32/np.sqrt(row['area_px_'+s])
        row['window64_over_sqrt_area_'+s]=64/np.sqrt(row['area_px_'+s])
    result={}
    if t is not None:
        a,b=[raw_outline(m) for m in (ma,mb)]
        sides=[side_distances(a,b,-np.asarray(t)),side_distances(b,a,np.asarray(t))]
        for kind in KINDS:
            values=[d[use[kind]] for d,use in sides]
            joined=np.concatenate(values)
            # Arc weights are slightly below1 due to ceil(perimeter) sampling.
            h=[np.histogram(v,FINE)[0]*g['step'] for v,g in zip(values,(a,b))]
            coarse=[np.histogram(v,EDGES)[0]*g['step'] for v,g in zip(values,(a,b))]
            arc=np.sum(h,axis=0);c=np.sum(coarse,axis=0)
            equal=np.mean([hh/hh.sum() for hh in h if hh.sum()>0],axis=0) if arc.sum()>0 else np.zeros(len(FINE)-1)
            row[kind+'_gap']=quantile(joined)
            row[kind+'_hist_arc_px']=c.tolist()
            row[kind+'_arc_a_px']=float(h[0].sum());row[kind+'_arc_b_px']=float(h[1].sum())
            result[kind]=dict(arc=arc,equal=equal,measurable=bool(arc.sum()))
    return row,result


def archive_job(payload):
    root,e=payload
    with np.load(Path(root)/e['artifact_path'],allow_pickle=False) as z:
        label=bool(z['label'].item());ma,mb=[unpack(z,s) for s in 'ab']
        t=z['translation_a_to_b_rc'] if label and bool(z['translation_valid']) else None
        p=[z['points_rc_'+s] for s in 'ab'];v=[z['contour_valid_'+s] for s in 'ab']
        row,h=measured_pair(ma,mb,t,p,v)
    return dict(row,pair_id=e['pair_id'],label=label,recipe=e.get('corrosion_recipe',e.get('s7_recipe',e.get('compound_recipe'))),
                fragment_a_id=e.get('fragment_a_token'),fragment_b_id=e.get('fragment_b_token'),artifact_path=e['artifact_path']),h


class Collector:
    def __init__(self,out,name):
        self.out=Path(out)/name;self.out.mkdir(parents=True,exist_ok=True)
        self.stream=(self.out/'pairs.jsonl').open('w');self.rows=[]
        self.arc={k:np.zeros(len(FINE)-1) for k in KINDS}
        self.equal={k:np.zeros(len(FINE)-1) for k in KINDS};self.n={k:0 for k in KINDS}
    def add(self,row,h):
        self.stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.rows.append(row)
        for k,x in h.items():
            self.arc[k]+=x['arc'];self.equal[k]+=x['equal'];self.n[k]+=int(x['measurable'])
    def finish(self,source):
        self.stream.close();positive=[r for r in self.rows if r['label']]
        stats=dict(pairs=len(self.rows),positive_pairs=len(positive),negative_pairs=len(self.rows)-len(positive),
            fine_edges=[float(x) if np.isfinite(x) else None for x in FINE],
            coarse_edges=[float(x) if np.isfinite(x) else None for x in EDGES],
            histograms={k:dict(measurable_pairs=self.n[k],arc_px=self.arc[k].tolist(),
                point_weighted_share=(self.arc[k]/self.arc[k].sum()).tolist() if self.arc[k].sum() else None,
                pair_equal_share=(self.equal[k]/self.n[k]).tolist() if self.n[k] else None) for k in KINDS},source=source)
        stats['positive_metrics']={k:quantile([r[k] for r in positive if r.get(k) is not None]) for k in
            ('d20_length_px','d10_breaks_mean','d40_gap_mean_px','area_ratio','mean_fragment_area_px')}
        for label in (True,False):
            rr=[r for r in self.rows if r['label']==label]
            stats['positive_fragments' if label else 'negative_fragments']={name:quantile(
                [r[key+'_'+s] for r in rr for s in 'ab' if r.get(key+'_'+s) is not None]) for name,key in
                [('area_px2','area_px'),('perimeter_px','perimeter_px'),('tokens','tokens'),
                 ('step_mean_px','model_step_mean'),('step_p90_px','model_step_p90'),
                 ('window32_over_sqrt_area','window32_over_sqrt_area'),('window64_over_sqrt_area','window64_over_sqrt_area')]}
        save(self.out/'summary.json',stats)
        return stats


def simulation(a):
    m=read(a.manifest);entries=m['entries'];start=time.time();c=Collector(a.out,a.name)
    if a.limit:entries=entries[:a.limit]
    with ProcessPoolExecutor(a.workers) as pool:
        for i,(row,h) in enumerate(pool.map(archive_job,((m['artifact_root'],e) for e in entries),chunksize=10),1):
            c.add(row,h)
            if i%2000==0:print(json.dumps(dict(dataset=a.name,done=i,total=len(entries),seconds=time.time()-start)),flush=True)
    c.finish(dict(manifest=a.manifest,sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
        measurement='raw outer-contour positions, normals sigma3, uniform1px arclength; bidirectional GT distance',
        gap_not_corrosion_depth=True,seconds=time.time()-start))
    print(json.dumps(dict(dataset=a.name,status='complete',pairs=len(entries))),flush=True)


def real(a):
    meta=read(a.manifest);gt={r['pair_id']:r for r in read(a.gt)['positive_pairs']}
    excluded={r['pair_id'] for r in read(a.exclusions)['records']}
    pairs=[p for p in meta['pairs'] if p['label']]
    if len(pairs)!=295:raise ValueError('expected original retained295 positives')
    prepared=Path(meta['prepared']);index={fid:i for i,fid in enumerate(meta['fragment_ids'])}
    with np.load(prepared/'inputs.npz',allow_pickle=False) as z:
        masks=np.unpackbits(z['packed_masks'],axis=-1).astype(bool);points=z['points'];valid=z['valid']
    allc=Collector(a.out,'Dunhuang295');cleanc=Collector(a.out,'Dunhuang292')
    for p in pairs:
        g=gt[p['pair_id']];fa,fb=p['fragment_a_id'],p['fragment_b_id'];ia,ib=index[fa],index[fb]
        assert (fa,fb)==(g['fragment_a_token'],g['fragment_b_token'])
        row,h=measured_pair(masks[ia],masks[ib],g['translation_gt_a_to_b_rc'],[points[ia],points[ib]],[valid[ia],valid[ib]])
        row.update(pair_id=p['pair_id'],label=True,fragment_a_id=fa,fragment_b_id=fb,
            gt_excluded=p['pair_id'] in excluded,translation_gt_rc=g['translation_gt_a_to_b_rc'],case_cluster=p.get('case_cluster'))
        allc.add(row,h)
        if not row['gt_excluded']:cleanc.add(row,h)
    source=dict(manifest=a.manifest,gt=a.gt,exclusions=a.exclusions,
                masks=str(prepared/'inputs.npz'),original_masks_changed=False,semantic_seam_gt_available=False)
    allc.finish(source);cleanc.finish(source)
    print(json.dumps(dict(status='complete',original=295,clean=len(cleanc.rows))),flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('sim');s.add_argument('--manifest',required=True);s.add_argument('--name',required=True)
    s.add_argument('--out',required=True);s.add_argument('--workers',type=int,default=8);s.add_argument('--limit',type=int,default=0)
    r=sub.add_parser('real');r.add_argument('--manifest',required=True);r.add_argument('--gt',required=True)
    r.add_argument('--exclusions',required=True);r.add_argument('--out',required=True)
    a=p.parse_args();simulation(a) if a.mode=='sim' else real(a)


if __name__=='__main__':main()

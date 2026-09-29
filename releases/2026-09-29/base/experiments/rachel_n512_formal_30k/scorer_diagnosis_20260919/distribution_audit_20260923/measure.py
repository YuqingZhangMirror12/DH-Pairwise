"""Mask/GT-only domain measurements. No model inference or annotation writes.

Lengths are geometric near-contact proxies, not hand-annotated semantic seams.
Turufan seam statistics are intentionally disabled until the user finishes review.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

cv2.setNumThreads(1)
THRESHOLDS = (4, 10, 20, 40, 64)


def resample(points, step=1.):
    d = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    good = d > 1e-9
    points = points[good]
    d = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    cum = np.r_[0., np.cumsum(d)]
    n = max(4, int(np.ceil(cum[-1] / step)))
    s = np.arange(n) * cum[-1] / n
    ix = np.minimum(np.searchsorted(cum, s, side='right') - 1, len(d) - 1)
    p = points[ix] + ((s-cum[ix])/d[ix])[:, None] * (np.roll(points, -1, axis=0)[ix]-points[ix])
    return p, float(cum[-1]), float(cum[-1]/n)


def outline(mask):
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('empty input mask')
    chosen = max(contours, key=cv2.contourArea)
    p = chosen[:, 0, ::-1].astype(float)
    # Standardize orientation for outward normals in row/column coordinates.
    signed = np.sum(p[:, 1]*np.roll(p[:, 0], -1)-np.roll(p[:, 1], -1)*p[:, 0])
    if signed < 0:
        p = p[::-1]
    raw, perimeter_raw, _ = resample(p)
    sm = gaussian_filter1d(raw, sigma=3., axis=0, mode='wrap')
    p, perimeter, step = resample(sm)
    t = np.roll(p, -4, axis=0) - np.roll(p, 4, axis=0)
    n = np.c_[-t[:, 1], t[:, 0]]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rc = np.argwhere(mask)
    shape = rc.max(0) - rc.min(0) + 1
    return dict(points=p, normals=n, step=step,
                metadata=dict(area_px=int(mask.sum()), perimeter_px=perimeter,
                              perimeter_raw_px=perimeter_raw, bbox_height_px=int(shape[0]),
                              bbox_width_px=int(shape[1]), external_component_count=len(contours),
                              roughness_raw_over_smooth=perimeter_raw/perimeter))


def circular_runs(bits):
    bits = np.asarray(bits, bool)
    if not bits.any():
        return []
    if bits.all():
        return [np.arange(len(bits))]
    starts = np.flatnonzero(bits & ~np.roll(bits, 1))
    runs = []
    for start in starts:
        ids = []
        for j in range(len(bits)):
            i = (int(start) + j) % len(bits)
            if not bits[i]:
                break
            ids.append(i)
        runs.append(np.array(ids, dtype=int))
    return runs


def clean_support(bits, step):
    # Close raster/normal jitter gaps <=3 px, then remove components <8 px.
    bits = np.asarray(bits, bool).copy()
    if not bits.any():
        return bits
    for run in circular_runs(~bits):
        if len(run) * step <= 3.:
            bits[run] = True
    for run in circular_runs(bits):
        if len(run) * step < 8.:
            bits[run] = False
    return bits


def support_stats(bits, step):
    runs = circular_runs(bits)
    lengths = [len(r)*step for r in runs]
    holes = sorted([len(r)*step for r in circular_runs(~bits)], reverse=True)
    # The longest complementary arc is outside the contact envelope, not a break.
    internal = holes[1:] if runs else []
    return dict(length=float(sum(lengths)), longest=float(max(lengths, default=0.)),
                segment_count=len(runs), span=float(len(bits)*step-holes[0]) if holes and runs else float(sum(lengths)),
                break_count_8_to_100=sum(8. <= x <= 100. for x in internal),
                long_separation_count=sum(x > 100. for x in internal),
                break_lengths_8_to_100=[x for x in internal if 8. <= x <= 100.])


def proximity_side(a, b, shift):
    p, q = a['points'], b['points'] + np.asarray(shift)
    d, j = cKDTree(q).query(p, workers=1)
    vector = q[j] - p
    opposed = np.sum(a['normals']*b['normals'][j], axis=1) <= -.5
    # Small raster / GT inaccuracies allowed; deep interpenetration is not gap.
    exterior = ((vector*a['normals']).sum(1) >= -3.) & ((-vector*b['normals'][j]).sum(1) >= -3.)
    return d, opposed & exterior


def geometry(a, b, translation):
    # t maps A coordinates to B coordinates, so B is placed at -t in A frame.
    da, oka = proximity_side(a, b, -np.asarray(translation))
    db, okb = proximity_side(b, a, np.asarray(translation))
    result = {}
    for threshold in THRESHOLDS:
        ba = clean_support(oka & (da <= threshold), a['step'])
        bb = clean_support(okb & (db <= threshold), b['step'])
        sa, sb = support_stats(ba, a['step']), support_stats(bb, b['step'])
        # Gap values exclude bridge-added points; length includes at most 3px smoothing gaps.
        gaps = np.r_[da[ba & oka & (da <= threshold)], db[bb & okb & (db <= threshold)]]
        key = 'd'+str(threshold)
        result.update({key+'_length_px': .5*(sa['length']+sb['length']),
            key+'_length_a_px':sa['length'], key+'_length_b_px':sb['length'],
            key+'_longest_px':.5*(sa['longest']+sb['longest']),
            key+'_span_px':.5*(sa['span']+sb['span']),
            key+'_gap_mean_px':float(gaps.mean()) if len(gaps) else None,
            key+'_gap_median_px':float(np.median(gaps)) if len(gaps) else None,
            key+'_gap_p90_px':float(np.quantile(gaps,.9)) if len(gaps) else None,
            key+'_segments_mean':.5*(sa['segment_count']+sb['segment_count']),
            key+'_breaks_mean':.5*(sa['break_count_8_to_100']+sb['break_count_8_to_100']),
            key+'_long_separations_mean':.5*(sa['long_separation_count']+sb['long_separation_count'])})
    ca=result['d20_length_a_px']/a['metadata']['perimeter_px']
    cb=result['d20_length_b_px']/b['metadata']['perimeter_px']
    result['d20_contact_fraction_min']=min(ca,cb)
    result['d20_contact_fraction_max']=max(ca,cb)
    result['d20_contact_fraction_asymmetry']=max(ca,cb)/min(ca,cb) if min(ca,cb)>0 else None
    result['d40_gap_mean_over_sqrt_smaller_area']=result['d40_gap_mean_px']/np.sqrt(min(a['metadata']['area_px'],b['metadata']['area_px'])) if result['d40_gap_mean_px'] is not None else None
    return result


def pair_metrics(a, b, translation=None):
    result = {k+'_'+side:v for side,obj in [('a',a),('b',b)] for k,v in obj['metadata'].items()}
    areas = [a['metadata']['area_px'], b['metadata']['area_px']]
    result['area_ratio']=min(areas)/max(areas)
    result['mean_fragment_area_px']=np.mean(areas).item()
    if translation is not None:
        result.update(geometry(a,b,translation))
    return result


def unpack(z, side):
    shape = tuple(z['mask_'+side+'_shape'])
    return np.unpackbits(z['mask_'+side+'_packed'],axis=-1)[...,:shape[-1]].reshape(shape).squeeze().astype(bool)


def aug_record(report, label):
    result = dict(changed=bool(report.get('changed_pair')), fallback_reason=report.get('fallback_reason'))
    sides=[report.get('side_'+s,{}) for s in 'ab']
    result['removed_area_px']=sum(s.get('removed_area_px',0) for s in sides)
    result['legacy_requested_max_depth_px']=max((s.get('config',{}).get('max_depth_px',0) for s in sides),default=0)
    detail=report.get('s7',{}).get('detail',{})
    members=detail.get('members',[])
    if members:
        member=members[0 if label else 1]
        for key in ('source_seam_length_px','source_seam_retention','physically_retained_source_seam_length_px','retained_supervised_seam_length_px','material_retention'):
            result[key]=member.get(key)
    details=[detail.get(s,{}) for s in 'ab']
    result['measured_max_erosion_depth_px']=max((x.get('applied_max_depth_px',0.) for x in details),default=0.) if report.get('s7',{}).get('recipe') in {'wave','local','seam_gaps'} else None
    result['applied_gap_count_per_pair']=sum(x.get('gap_k_applied',0) for x in details)
    result['applied_local_notch_count_per_pair']=sum(x.get('local_notch_count_applied',0) for x in details)
    result['applied_sides']=sum(bool(x.get('applied')) for x in sides)
    gp=report.get('gen5_partition',{})
    result['gen5_patterns']=[p.get('ordered_pattern') for p in gp.get('geometry_provenance',[])]
    return result


def one_sim(payload):
    root, entry = payload
    with np.load(Path(root)/entry['artifact_path'],allow_pickle=False) as z:
        label=bool(z['label'].item())
        assert label==bool(entry['label']) and str(z['pair_id'])==entry['pair_id']
        ma,mb=unpack(z,'a'),unpack(z,'b')
        a,b=outline(ma),outline(mb)
        translation=z['translation_a_to_b_rc'] if label and bool(z['translation_valid']) else None
        report=json.loads(str(z['report_json']))
        result=dict(dataset='S7',pair_id=entry['pair_id'],label=label,recipe=entry['s7_recipe'],
            source_stratum=entry.get('source_stratum'),group_index=entry['s7_group_index'],
            s7_h=bool(entry['changed_pair']) and entry['s7_recipe'] in {'local','seam_gaps','partial_curve'},
            translation_gt_rc=translation.tolist() if translation is not None else None,
            augmentation=aug_record(report,label),**pair_metrics(a,b,translation))
        assert result['augmentation']['changed']==bool(entry['changed_pair'])
        return result


def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def do_sim(args):
    manifest=json.loads(Path(args.manifest).read_text())
    entries=manifest['entries']; root=Path(manifest['artifact_root'])
    if args.limit:
        entries=entries[:args.limit]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();n=0
    with (out/'sim_pairs.jsonl').open('w') as output, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(one_sim,((str(root),x) for x in entries),chunksize=20):
            output.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');n+=1
            if n%2000==0:
                output.flush(); print(json.dumps(dict(completed=n,elapsed_seconds=time.monotonic()-start)),flush=True)
    save_json(out/'sim_receipt.json',dict(manifest=str(args.manifest),n=n,
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        protocol=manifest.get('protocol'),elapsed_seconds=time.monotonic()-start,workers=args.workers))
    print(json.dumps(dict(complete=True,n=n,elapsed_seconds=time.monotonic()-start)),flush=True)


def do_real(args):
    data=json.loads(Path(args.inputs).read_text())
    snapshot=json.loads(Path(args.snapshot).read_text())
    cases={x['pair_id']:x for x in snapshot['queries']['cases']['rows']}
    fragments={}
    fragment_rows=[]
    for dataset,manifest in data['manifests'].items():
        for fid in manifest['fragment_ids']:
            f=data['fragments'][fid]
            image=np.asarray(Image.open(io.BytesIO(base64.b64decode(f['png'].split(',')[1]))))
            mask=image[:,:,3]>0
            assert int(mask.sum())==f['area'],(fid,int(mask.sum()),f['area'])
            obj=outline(mask)
            obj['points']+=np.array(f['bbox_xywh'][:2][::-1])
            fragments[fid]=obj
            fragment_rows.append(dict(dataset=dataset,fragment_id=fid,**obj['metadata']))
    rows=[]
    for dataset,manifest in data['manifests'].items():
        for pair in manifest['pairs']:
            c=cases[pair['pair_id']]
            assert bool(c['label'])==bool(pair['label'])
            # Explicit user gate: NEVER use Turufan model positions or unfinished annotations.
            translation=c['gt'] if dataset=='dunhuang_cv' and pair['label'] else None
            if dataset=='dunhuang_cv' and pair['label']:
                assert translation is not None
            a,b=fragments[pair['fragment_a_id']],fragments[pair['fragment_b_id']]
            rows.append(dict(dataset=dataset,pair_id=pair['pair_id'],label=pair['label'],case_name=c['case_name'],
                fragment_a_id=pair['fragment_a_id'],fragment_b_id=pair['fragment_b_id'],
                translation_gt_rc=translation,**pair_metrics(a,b,translation)))
    save_json(Path(args.out)/'real_fragments.json',fragment_rows)
    save_json(Path(args.out)/'real_pairs.json',rows)
    if args.full_gt:
        full=[]
        for p in json.loads(Path(args.full_gt).read_text())['positive_pairs']:
            fid_a,fid_b=p['fragment_a_token'],p['fragment_b_token']
            t=p['translation_gt_a_to_b_rc']
            c=cases.get(p['pair_id'])
            if c and c['dataset']=='敦煌':
                assert np.allclose(t,c['gt'],atol=1e-9)
            full.append(dict(dataset='dunhuang_full',pair_id=p['pair_id'],label=True,
                fragment_a_id=fid_a,fragment_b_id=fid_b,retained=p['pair_id'] in cases,
                translation_gt_rc=t,**pair_metrics(fragments[fid_a],fragments[fid_b],t)))
        save_json(Path(args.out)/'dunhuang_full_pairs.json',full)
    save_json(Path(args.out)/'real_receipt.json',dict(n_pairs=len(rows),n_fragments=len(fragment_rows),
        masks_inputs_sha256=hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
        preprocessing=data['preprocessing'],turufan_seams='DEFERRED_BY_USER; no annotations read',
        measurement='dense external contour; 1px arc resampling; sigma3px measurement smoothing; no mask mutation',
        thresholds_px=THRESHOLDS,minimum_component_px=8,microgap_bridge_px=3,normal_cosine_max=-.5,
        gap_caveat='conditional distance within threshold, not original material loss',
        partial_caveat='real ancestry unobserved; coverage asymmetry is only a proxy'))
    print(json.dumps(dict(complete=True,pairs=len(rows),fragments=len(fragment_rows))),flush=True)


def selftest():
    m=np.zeros((180,180),bool);m[30:150,30:90]=True
    a=outline(m)
    # Two equal rectangles placed with boundary-center distance 1, 11, 21px.
    for gap in (0,10,20):
        r=geometry(a,a,np.array([0.,-(60.+gap)]))
        threshold=next(t for t in THRESHOLDS if t>gap+1)
        assert r['d'+str(threshold)+'_length_px']>90,r
        assert abs(r['d'+str(threshold)+'_gap_median_px']-(gap+1))<.5,r
        if gap>=10:
            assert r['d4_length_px']==0,r
    r=geometry(a,a,np.array([0.,400.]))
    assert r['d64_length_px']==0
    r1=geometry(a,a,np.array([50.,-60.]));r2=geometry(a,a,np.array([-50.,60.]))
    assert abs(r1['d20_length_px']-r2['d20_length_px'])<1e-8
    rr=pair_metrics(a,a,np.array([0.,-60.]))
    assert rr['area_ratio']==1 and abs(rr['area_px_a']-7200)==0
    bits=np.zeros(200,bool);bits[:50]=1;bits[70:110]=1
    stats=support_stats(bits,1.)
    assert stats['break_count_8_to_100']==1 and stats['span']==110
    print('geometry self-tests passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('selftest')
    s=sub.add_parser('sim');s.add_argument('--manifest',required=True);s.add_argument('--out',required=True)
    s.add_argument('--workers',type=int,default=2);s.add_argument('--limit',type=int,default=0)
    r=sub.add_parser('real');r.add_argument('--inputs',required=True);r.add_argument('--snapshot',required=True);r.add_argument('--out',required=True);r.add_argument('--full-gt')
    args=p.parse_args()
    if args.mode=='sim':do_sim(args)
    elif args.mode=='real':do_real(args)
    else:selftest()

"""Read-only gap audit with partner identity frozen BEFORE structural cutting.

The earlier diagnostic rematched points to the *retained* opposite source band.
At a cut endpoint that could replace a removed original partner with a distant
one. This audit keeps the original nearest supported source partner instead.
Ambiguous original source distances >3px and failed rays remain in the retained
source denominator and are reported as unresolved, never as zero or new GT.
No masks, correspondence labels, generator source, or training are modified.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2 import materialize
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.scale import pair_shared_scale
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.layered_geometry import canonical_mirrored_contours
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.seam_context_v3.augmentation import paired_mirror
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.latent_seam import source_band,ray_project
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.partial_v14 import retained_support
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.conservative_weather import parts


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(ok,message):
    if not ok:raise AssertionError(message)


def frozen_partners(p,q,shift):
    """Mapping depends ONLY on pre-cut source geometry, not final masks/gaps."""
    distance,index=cKDTree(q).query(p+shift)
    return q[index],distance<=3.+1e-6


def audit(record,baseline,out):
    path=Path(record['sample_path']);name=path.name
    sample,_=load_sample(path);old,old_report=load_sample(record['baseline_sample_path'])
    group=read(baseline/'groups'/(name.split('_')[0]+'.json'))
    original,_=pair_shared_scale(materialize.clean_positive(group['source_positive_pair_id']),
        old_report['pair_shared_scale']['requested_mean_area_px2'],topology_backoff=True,identity_fallback=True)
    if record['mirror']:original=canonical_mirrored_contours(paired_mirror(original,record['mirror']))
    with np.load(record['proof_path'],allow_pickle=False) as d:z={k:d[k] for k in d.files}
    stage='primary' if record['recipe']=='partial' else 'trim'
    trimmed=replace(original,**{'mask_'+s:np.unpackbits(z['packed_'+stage+'_'+s],axis=1)[None].astype(np.float32) for s in 'ab'})
    bands=source_band(original,bridge=0.);_,proof=retained_support(original,trimmed,bands)
    arrays={};all_gap=[];all_weight=[];active_new=[];active_old=[];denom=0.;ambiguous=0.;ray_unresolved=0.;changed=0
    for side,other,shift in [('a','b',sample.translation_a_to_b_rc),('b','a',-sample.translation_a_to_b_rc)]:
        keep=proof[side+'_physically_retained'];p=proof[side+'_source_points'][keep];w=proof[side+'_source_weights'][keep]
        q0,_,bits=bands[other];q,source_ok=frozen_partners(p,q0[bits],shift)
        need(np.allclose(p,z['gap_'+side+'_source_points']), 'same retained source denominator')
        need(np.allclose(q,proof[side+'_partner_points'][keep]),'frozen pre-cut partner consistency')
        old_q=z['gap_'+side+'_partner_source_points'];changed+=int((np.linalg.norm(q-old_q,axis=1)>.01).sum())
        pa,da,va=ray_project(getattr(original,'mask_'+side)[0],getattr(sample,'mask_'+side)[0],p)
        pb,db,vb=ray_project(getattr(original,'mask_'+other)[0],getattr(sample,'mask_'+other)[0],q)
        valid=source_ok&va&vb;gap=np.linalg.norm(pa+shift-pb,axis=1)
        oa,_,ova=ray_project(getattr(original,'mask_'+side)[0],getattr(old,'mask_'+side)[0],p)
        ob,_,ovb=ray_project(getattr(original,'mask_'+other)[0],getattr(old,'mask_'+other)[0],q)
        old_gap=np.linalg.norm(oa+shift-ob,axis=1);old_valid=source_ok&ova&ovb
        active=z['gap_'+side+'_primary_affected']
        arrays.update({side+'_'+k:v.astype(np.float32) if v.dtype!=bool else v for k,v in dict(
            source_points=p,partner_source_points=q,source_weight=w,source_valid=source_ok,
            projected_points=pa,partner_projected_points=pb,valid=valid,gap=gap,
            source_gap=np.linalg.norm(p+shift-q,axis=1),recession=da,partner_recession=db,
            primary_affected=active,old_gap=old_gap,old_valid=old_valid).items()})
        all_gap.extend(gap[valid]);all_weight.extend(w[valid]);active_new.extend(gap[valid&active]);active_old.extend(old_gap[old_valid&active])
        denom+=float(w.sum());ambiguous+=float(w[~source_ok].sum());ray_unresolved+=float(w[source_ok&~(va&vb)].sum())
    values=np.asarray(all_gap);weights=np.asarray(all_weight);major,_=parts(record['recipe'])
    maximum=max(active_new) if major else None
    if major:need(maximum is not None and 5<=maximum<=15+1e-4,'fixed-pair primary peak5–15')
    need(ambiguous+ray_unresolved<=denom+1e-5,'exclusive unresolved accounting')
    np.savez_compressed(out/(record['id']+'.npz'),**arrays)
    return dict(id=record['id'],status='passed',changed_partner_count=changed,
        retained_source_arc_length_px=denom/2,source_ambiguous_fraction=ambiguous/denom,
        ray_unresolved_fraction=ray_unresolved/denom,unresolved_fraction=(ambiguous+ray_unresolved)/denom,
        new_primary_gap_peak_px=float(maximum) if maximum is not None else None,
        old_primary_gap_peak_px=float(max(active_old)) if major and active_old else None,
        resolved_arc_gap_p10_p50_p90=np.quantile(values,[.1,.5,.9]).tolist(),
        all_resolved_gap_max_px=float(values.max()),
        arc_fraction_gap5to15=float(np.sum(weights*((values>=5)&(values<=15)))/weights.sum()),
        sample_sha256=sha(path),proof_sha256=sha(record['proof_path']),
        projection_sha256=sha(out/(record['id']+'.npz')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.pilot);out=Path(a.out)
    if out.exists():raise ValueError('new audit output required')
    out.mkdir(parents=True);records=read(root/'manifest.json')['entries'];protocol=read(root/'protocol.json');baseline=Path(protocol['baseline'])
    materialize.initialize(read(baseline/'protocol.json')['options'])
    receipts=[];errors=[]
    for r in records:
        if not r['label']:continue
        try:receipts.append(audit(r,baseline,out))
        except Exception as e:errors.append(dict(id=r['id'],error=repr(e)))
    result=dict(status='passed' if not errors else 'failed',positive_pairs=sum(bool(r['label']) for r in records),
        receipts=receipts,errors=errors,manifest_sha256=sha(root/'manifest.json'),
        definition='original TRAIN-supported source partners frozen before cutting; original source distance<=3px; normal-ray projection; unresolved stays in denominator',
        generator_masks_or_labels_changed=False,training_changed=False)
    (out/'projection_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],pairs=len(receipts),errors=errors)))
    if errors:raise ValueError('projection audit failed')


if __name__=='__main__':main()

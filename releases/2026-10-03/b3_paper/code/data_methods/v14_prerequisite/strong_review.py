"""Actual completed strong-data receipt and literal-mask review figures."""
import argparse
import base64
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..s7_compound_v1.materialize import read,save_json
from ..gap_distribution_v2.showcases import display
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample

NAMES={'clean':'无腐蚀','wave':'连续起伏退蚀','local_deep':'局部深腐蚀',
 'gaps':'1–5处接缝缺口','wave_gaps':'起伏退蚀＋缺口',
 'wave_local':'起伏退蚀＋局部深腐蚀','wave_local_gaps':'起伏退蚀＋深腐蚀＋缺口'}
NEG={'cross_gen':'跨Gen负例','cross_parent_same_gen':'同Gen跨原图负例',
 'same_parent_nonadjacent':'同原图明确非邻接负例'}


def q(values):
    x=np.array([v for v in values if v is not None],float)
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,
        median=float(np.median(x)) if len(x) else None,
        p10=float(np.quantile(x,.1)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None)


def damage(row):
    dd=[v for v in row['augmentation']['damage'].values() if v.get('applied')]
    return dict(peak=max([v.get('applied_max_depth_px',0) for v in dd],default=0),
        k=sum(v.get('notch_kinds',[]).count('gap') for v in dd),
        local=sum(v.get('notch_kinds',[]).count('local') for v in dd),
        extra=sum(sum(v.get('notch_independently_removed_pixels',[])) for v in dd))


def placement_summary(rows):
    """Expose the conditional near-island distribution, not its old recipe."""
    result=[]
    for label,value in [('正例',True),('负例',False)]:
        rr=[r for r in rows if bool(r['label'])==value and r['recipe']!='clean']
        islands=[d['survival_island'] for r in rr for d in r['augmentation']['damage'].values()
                 if d.get('applied') and d.get('survival_island')]
        widths=q(d['core_width_px'] for d in islands)
        attempts=q(r['augmentation']['weather_plan_selection']['attempts'] for r in rr
                   if r['augmentation'].get('weather_plan_selection'))
        result.append(dict(label=label,corrodedPairs=len(rr),islandSides=len(islands),
            coreWidthPx=widths,planAttempts=attempts,
            anchorCounts=dict(Counter(str(d['anchor_count']) for d in islands)),
            definition='Per weathered fragment; core excludes two 8px smooth shoulders. Positive near islands enclose inherited TRAIN anchors; negatives copy the physical width at a random own-contour location.'))
    return result


def compile_evidence(root):
    root=Path(root);s=read(root/'status.json');val=read(root/'validation.json')
    if s['status'] not in ('complete','pilot_complete') or val['status']!='passed':
        raise ValueError('completed data and passed loader checks required')
    manifest=read(root/'train_s7b_24k.json');protocol=read(root/'protocol.json')
    rows=[json.loads(l) for l in (root/'pair_metrics.jsonl').read_text().splitlines()]
    pos=[r for r in rows if r['label']];neg=[r for r in rows if not r['label']]
    assert len(rows)==s['sample_count']==len(manifest['entries'])
    recipes=[]
    for key,percent in protocol['recipe_percent'].items():
        rr=[r for r in pos if r['recipe']==key];nn=[r for r in neg if r['recipe']==key]
        recipes.append(dict(key=key,recipe=NAMES[key],category=rr[0]['corrosion_category'],
            positive=len(rr),negative=len(nn),share=len(rr)/len(pos),expectedShare=percent/100,
            partial=sum(r['partial'] for r in rr),peak=q(damage(r)['peak'] for r in rr if key!='clean'),
            latentLength=q(r['latent_seam']['source_length_px'] for r in rr),
            gapMean=q(r['latent_seam']['gap_mean_px'] for r in rr)))
    bin_edges=pos[0]['latent_seam']['gap_bins'];hist=[];summaries=[]
    for group,rr in [('所有正例',pos),('仅腐蚀正例',[r for r in pos if r['recipe']!='clean']),
                     ('含连续起伏退蚀',[r for r in pos if 'wave' in r['recipe']])]:
        h=np.mean([r['latent_seam']['gap_share'] for r in rr],axis=0)
        for i,v in enumerate(h):
            lo,hi=bin_edges[i:i+2]
            hist.append(dict(group=group,bin=f'{lo:g}–{hi:g}' if hi is not None else f'≥{lo:g}',share=float(v),pairs=len(rr)))
        summaries.append(dict(group=group,pairs=len(rr),
            meanGap=q(r['latent_seam']['gap_mean_px'] for r in rr),
            sourceLength=q(r['latent_seam']['source_length_px'] for r in rr),
            within5Mean=float(np.mean([r['latent_seam']['fraction_under5'] for r in rr])),
            over15Mean=float(np.mean([r['latent_seam']['fraction_over15'] for r in rr])),
            heterogeneousPairs=sum(r['latent_seam']['has_near_and_far'] for r in rr),
            resolvedMean=float(np.mean([r['latent_seam']['ray_resolved_fraction'] for r in rr]))))
    gapk=[]
    for label,rr in [('正例',pos),('负例',neg)]:
        rr=[r for r in rr if 'gaps' in r['recipe']];cnt=Counter(damage(r)['k'] for r in rr)
        assert all(1<=k<=5 for k in cnt)
        for k in range(1,6):gapk.append(dict(label=label,k=str(k),pairs=cnt[k],share=cnt[k]/len(rr),denominator=len(rr)))
    negcnt=Counter(r['negative_kind'] for r in neg)
    return dict(schema='strong-seam-review/1',root=str(root),trainingStarted=False,
        status=s['status'],pairs=len(rows),positives=len(pos),negatives=len(neg),
        recipeRows=recipes,latentHist=hist,latentSummaries=summaries,notches=gapk,
        placementSummary=placement_summary(rows),
        negativeSources=[dict(kind=k,name=NEG[k],pairs=v,share=v/len(neg)) for k,v in negcnt.items()],
        negativeGroupedPairs=sum(bool(r['anchor_group_id']) for r in neg),
        partial=sum(r['partial'] for r in pos),mirrors=dict(Counter(str(r['mirror']) for r in pos)),
        sourceStrata=dict(Counter(('positive:' if e['label'] else 'negative:')+e['source_stratum'] for e in manifest['entries'])),
        actualPeaks={k:q(damage(r)['peak'] for r in rows if r['label']==label and r['recipe']!='clean')
                     for k,label in [('positive',True),('negative',False)]},
        definition='Fixed pre-weather TRAIN-GT-supported seam. First remaining material along old inward normals; unresolved rays reported; no final-gap cutoff; not exact new match labels.',
        lengthQuotaBasis=protocol['length_quota_basis'],validation=val,
        generationSummary=read(root/'summary.json'),protocol=protocol),manifest,rows


def mirror_rc(p,axis):
    p=np.array(p,float).copy()
    if axis:p[...,1 if axis=='horizontal' else 0]=799.-p[...,1 if axis=='horizontal' else 0]
    return p


def figure(root,entry,row,out,key,original=None):
    sample,report=load_sample(root/entry['artifact_path']);axis=entry['offline_paired_mirror']
    fig=plt.figure(figsize=(15,7));grid=fig.add_gridspec(2,3,width_ratios=[1.15,1.15,1])
    ax0=fig.add_subplot(grid[:,0]);ax1=fig.add_subplot(grid[:,1])
    a2=fig.add_subplot(grid[0,2]);a3=fig.add_subplot(grid[1,2])
    gap_profile=[]
    if entry['label']:
        with np.load(root/entry['latent_seam_artifact'],allow_pickle=False) as z:
            bm=[np.unpackbits(z['packed_preweather_'+s],axis=1).astype(np.float32)[None] for s in 'ab']
            t=z['translation_a_to_b_rc'].copy()
            if axis:
                bm=[np.flip(x,axis=2 if axis=='horizontal' else 1).copy() for x in bm]
                t[1 if axis=='horizontal' else 0]*=-1
            before=replace(sample,mask_a=bm[0],mask_b=bm[1],translation_a_to_b_rc=t)
            display(ax0,original if original is not None else before,
                    'Original / before Partial + scaling' if original is not None else 'Before corrosion / same GT')
            display(ax1,sample,'After augmentation / same GT')
            # Use identical world extents before/after. Neither drawing is
            # rescaled independently to make removed material look equal.
            if original is None:
                ax1.set_xlim(ax0.get_xlim());ax1.set_ylim(ax0.get_ylim())
            valid=z['a_valid'];gap=z['a_gap'][valid]
            arc=np.cumsum(z['a_source_weight'])-.5*z['a_source_weight']
            ids=np.unique(np.linspace(0,len(arc)-1,min(200,len(arc))).astype(int))
            gaps_at=np.r_[False,np.linalg.norm(np.diff(z['a_source_points'],axis=0),axis=1)>3.]
            # Preserve a break even when the display's decimation skips the
            # first point of a new supported segment.
            for ordinal,i in enumerate(ids):
                previous=ids[ordinal-1] if ordinal else i
                split=bool(gaps_at[previous+1:i+1].any())
                gap_profile.append(dict(arcPx=float(arc[i]),
                    gapPx=float(z['a_gap'][i]) if valid[i] and not split else None,
                    rawGapPx=float(z['a_gap'][i]) if valid[i] else None,
                    displayLineBreak=split,originalGapPx=float(z['a_source_gap'][i])))
            pa=mirror_rc(z['a_projected_points'][valid],axis)
            pb=mirror_rc(z['a_partner_projected_points'][valid],axis)-t
            points=(pa+pb)/2
            for ax,quantile,title in ((a2,.1,'Lower-gap section'),(a3,.9,'Higher-gap section')):
                idx=int(np.argmin(abs(gap-np.quantile(gap,quantile))))
                center=points[idx]
                display(ax,sample,f'{title}: projected gap {gap[idx]:.1f}px',center)
                ax.plot([pa[idx,1],pb[idx,1]],[pa[idx,0],pb[idx,0]],color='#70257f',lw=1.5)
                ax1.scatter([center[1]],[center[0]],facecolors='none',edgecolors='#70257f',s=100,lw=1.5)
    else:
        display(ax0,sample,'Negative A + B / unplaced')
        # Outer contour shown at literal crop scale: no invented negative GT.
        for ax,side in ((ax1,'a'),(a2,'b')):
            m=getattr(sample,'mask_'+side)[0]
            ax.imshow(m,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest');p=np.argwhere(m)
            lo=p.min(0);hi=p.max(0)
            ax.set_xlim(lo[1]-20,hi[1]+20);ax.set_ylim(hi[0]+20,lo[0]-20)
            ax.set_aspect('equal');ax.set_title('Augmented fragment '+side.upper());ax.axis('off')
        a3.axis('off');a3.text(.03,.9,'NEGATIVE\nNo GT seam or layout\nRandom own-contour damage\nNo fabricated correspondence',va='top',fontsize=13)
    fig.suptitle(f'S7-E strong | {row["recipe"]} | partial={row["partial"]} | label={int(entry["label"])}',fontsize=15)
    fig.tight_layout();path=out/(key+'.png');fig.savefig(path,dpi=125);plt.close(fig)
    d=damage(row);lat=row.get('latent_seam')
    caption=f"{NAMES[row['recipe']]}；Partial={'是' if row['partial'] else '否'}；镜像={axis or '无'}；面积比1:{1/row['area_ratio']:.2f}。"
    if row['recipe']!='clean':caption+=f"实际最大退蚀{d['peak']:.1f}px，缺口{d['k']}处、局部深腐蚀{d['local']}处。"
    if lat:
        caption+=f"腐蚀前GT支持段长{lat['source_length_px']:.1f}px；全支持段投影gap P10/P50/P90={lat['gap_p10_px']:.1f}/{lat['gap_p50_px']:.1f}/{lat['gap_p90_px']:.1f}px；可测弧长{lat['ray_resolved_fraction']:.1%}。"
        if original is None:caption+='左图已完成Partial/共同缩放，尚未腐蚀；与中图使用相同GT及显示尺度。'
        else:caption+='左图是未Partial裁切、未共同缩放的原始正例；中图为曲线Partial后。两图各有100px标尺，不按显示大小推断面积变化。'
        caption+='紫线为诊断投影，不是训练标签。右侧小图均为160px范围。'
    else:caption+='此为明确负例，只有分开展示，不暗示两片可拼；负例没有接缝间隙。'
    return dict(id=key,title=NAMES[row['recipe']]+(' · 正例' if entry['label'] else ' · '+NEG[entry['negative_kind']]),
        pair_id=entry['pair_id'],recipe=row['recipe'],label=bool(entry['label']),partial=bool(row['partial']),
        gapProfile=gap_profile,
        source_stratum=entry['source_stratum'],caption=caption,
        image='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),
        source_artifact=str(root/entry['artifact_path']),latent_source=str(root/entry['latent_seam_artifact']) if entry['label'] else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--evidence-only',action='store_true',help='Refresh summaries, retaining verified cases from this exact data root')
    a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    evidence,manifest,rows=compile_evidence(root)
    if a.evidence_only:
        prior=read(out/'evidence.json');cases=read(out/'showcases.json')
        if Path(prior['root']).resolve()!=root.resolve():raise ValueError('case root mismatch')
        pair_ids={r['pair_id'] for r in rows}
        if not cases or any(c['pair_id'] not in pair_ids for c in cases):raise ValueError('case scope mismatch')
        evidence['cases']=[{k:v for k,v in c.items() if k!='image'} for c in cases]
        save_json(out/'evidence.json',evidence)
        print(json.dumps(dict(pairs=evidence['pairs'],cases=len(cases),images_reused=True,out=str(out))))
        return
    metrics={r['pair_id']:r for r in rows};entries=manifest['entries'];selected=[];used=set()
    for recipe in evidence['protocol']['recipe_percent']:
        for partial in (False,True):
            pool=[e for e in entries if e['label'] and e['corrosion_recipe']==recipe and e['partial_applied']==partial]
            if not pool:continue
            # Prespecified typical example by mean gap, not maximum damage.
            mid=np.median([metrics[e['pair_id']]['latent_seam']['gap_mean_px'] for e in pool])
            chosen=min(pool,key=lambda e:abs(metrics[e['pair_id']]['latent_seam']['gap_mean_px']-mid))
            selected.append((recipe+('-partial' if partial else '-whole'),chosen));used.add(chosen['pair_id'])
    predicates=[('area-small',lambda e:e['label'] and metrics[e['pair_id']]['area_ratio']<.25),
        ('horizontal',lambda e:e['label'] and e['offline_paired_mirror']=='horizontal'),
        ('vertical',lambda e:e['label'] and e['offline_paired_mirror']=='vertical')]
    strata=sorted(set(e['source_stratum'] for e in entries if e['label'] and e['source_stratum']!='native'))
    for i,stratum in enumerate(strata):predicates.append(('structure-'+str(i),lambda e,s=stratum:e['label'] and e['source_stratum']==s))
    for kind in NEG:predicates.append((kind,lambda e,k=kind:not e['label'] and e['negative_kind']==k and e['corrosion_recipe']!='clean'))
    predicates.append(('negative-anchor',lambda e:not e['label'] and bool(e['anchor_group_id'])))
    for key,test in predicates:
        pool=[e for e in entries if test(e) and e['pair_id'] not in used]
        if not pool:continue
        mid=np.median([metrics[e['pair_id']]['mean_fragment_area_px'] for e in pool])
        chosen=min(pool,key=lambda e:abs(metrics[e['pair_id']]['mean_fragment_area_px']-mid))
        selected.append((key,chosen));used.add(chosen['pair_id'])
    from .materialize import initialize,clean_positive
    from ..seam_context_v3.augmentation import paired_mirror
    initialize(evidence['protocol']['options'])
    cases=[]
    for key,e in selected:
        original=None
        if key=='clean-partial':
            original=clean_positive(e['source_pair_id'])
            if e['offline_paired_mirror']:original=paired_mirror(original,e['offline_paired_mirror'])
        cases.append(figure(root,e,metrics[e['pair_id']],out,key,original))
    evidence['cases']=[{k:v for k,v in c.items() if k!='image'} for c in cases]
    save_json(out/'evidence.json',evidence);save_json(out/'showcases.json',cases)
    print(json.dumps(dict(pairs=evidence['pairs'],cases=len(cases),out=str(out))))


if __name__=='__main__':main()

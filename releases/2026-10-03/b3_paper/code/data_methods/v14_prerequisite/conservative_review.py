"""Full realised v12 measurements and ten literal-mask examples per type."""
import argparse
import base64
from collections import Counter
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..s7_compound_v1.materialize import read,save_json
from ..gap_distribution_v2.showcases import display
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from .conservative_weather import NAMES,parts
from .strong_review import q,NEG


def audit(root,manifest,rows):
    """Recompute numerical fields from every archive, not configured maxima."""
    metrics={r['pair_id']:r for r in rows};counts=Counter();max_depth=0.;max_coverage=0.;weak_modes=Counter()
    for entry in manifest['entries']:
        r=metrics[entry['pair_id']];major,weak=parts(r['recipe'])
        with np.load(root/entry['weather_artifact'],allow_pickle=False) as z:
            for side in 'ab':
                if side+'_total' not in z:continue
                total=z[side+'_total'];base=z[side+'_major'];w=z[side+'_weak'];edge=z[side+'_edge'];el=z[side+'_eligible']
                assert np.all(np.isfinite(total)) and np.all(total>=0) and total.max()<=9.
                assert np.allclose(total,np.minimum(9.,base+w),atol=1e-6)
                assert np.all(total[~el]==0)
                fraction=float(edge[total>0].sum()/edge[el].sum())
                assert fraction<=.5+1e-6
                d=r['augmentation']['damage'][side]
                assert abs(fraction-d['affected_fraction'])<1e-6
                assert np.count_nonzero(base)>0 if major else np.count_nonzero(base)==0
                assert np.count_nonzero(w)>0 if weak else np.count_nonzero(w)==0
                max_depth=max(max_depth,float(total.max()));max_coverage=max(max_coverage,fraction)
                if weak:
                    assert w.max()<=4.
                    weak_modes[d['weak_mode']]+=1
                if major=='gaps':assert 1<=d['notch_count']<=3
        counts[r['recipe']]+=1
    return dict(status='passed',archives=len(manifest['entries']),actual_field_max_px=max_depth,
        actual_max_affected_fraction=max_coverage,weak_modes=dict(weak_modes),recipe_counts=dict(counts),
        numerical_fields_from_all_archives=True,depth_is_not_sum_of_both_fragments=True)


def summarise(root):
    status=read(root/'status.json');validation=read(root/'validation.json')
    assert status['status'] in ('complete','pilot_complete') and validation['status']=='passed'
    manifest=read(root/'train_s7b_24k.json');protocol=manifest['protocol']
    assert protocol['distribution_revision'].get('conservative_v12')
    rows=[json.loads(l) for l in (root/'pair_metrics.jsonl').open()]
    pos=[r for r in rows if r['label']];neg=[r for r in rows if not r['label']]
    recipes=[]
    for key in protocol['recipe_percent']:
        rr=[r for r in pos if r['recipe']==key];nn=[r for r in neg if r['recipe']==key]
        dd=[d for r in rr for d in r['augmentation']['damage'].values() if d.get('applied')]
        recipes.append(dict(key=key,name=NAMES[key],positive=len(rr),negative=len(nn),share=len(rr)/len(pos),
            actualPeak=q(d['applied_max_depth_px'] for d in dd),
            affectedFraction=q(d['affected_fraction'] for d in dd),
            sourceLength=q(r['latent_seam']['source_length_px'] for r in rr),
            gapMean=q(r['latent_seam']['gap_mean_px'] for r in rr)))
    numerical=audit(root,manifest,rows)
    return dict(schema='conservative-seam-review/12',root=str(root),pairs=len(rows),positives=len(pos),negatives=len(neg),
        trainingStarted=False,status=status['status'],recipeRows=recipes,numericalAudit=numerical,
        validation=validation,protocol=protocol,summary=read(root/'summary.json'),
        gapCounts={str(label):dict(Counter(sum(d.get('notch_count',0) for d in r['augmentation']['damage'].values())
            for r in rows if r['label']==label and 'gaps' in r['recipe'])) for label in (True,False)},
        localCounts={k:sum(r['recipe'].startswith(k) for r in pos) for k in ('local_abrupt','local_gradual')},
        negativeSources=[dict(key=k,name=NEG[k],pairs=n,share=n/len(neg)) for k,n in Counter(r['negative_kind'] for r in neg).items()],
        metricDefinition='Affected fraction = field>0 source-arc length / full pre-weather TRAIN-supported arc on the selected side; includes weak overlay. One side per pair. Negative denominator is its own random corridor, not a true seam.'),manifest,rows


def add_layer_statistics(evidence,manifest,rows):
    """Two levels; primary damage counts are mutually exclusive, not axes."""
    if not evidence['protocol']['distribution_revision'].get('layered_damage_exclusive'):return
    from .layered_geometry import corrosion_layer
    names={'none':'完全无腐蚀／严丝合缝','partial':'Partial Seam（仅加1–3px轻退化）',
        'weak':'仅连续弱腐蚀','strong':'起伏／局部腐蚀（可附加弱腐蚀）','notch':'缺口（可附加弱腐蚀）'}
    layer=[]
    for key,name in names.items():
        p=sum(r['label'] and corrosion_layer(r['recipe'])==key for r in rows)
        n=sum(not r['label'] and corrosion_layer(r['recipe'])==key for r in rows)
        layer.append(dict(key=key,name=name,positive=p,negative=n,share=p/evidence['positives']))
    p=[r for r in rows if r['partial']]
    light=[d for r in rows for d in r.get('background_degradation',{}).values()]
    micro=[]
    for recipe in evidence['protocol']['recipe_percent']:
        rr=[r for r in rows if r['recipe']==recipe];dd=[d for r in rr for d in r.get('background_degradation',{}).values()]
        micro.append(dict(key=recipe,name=NAMES[recipe],pairs=len(rr),fragments=len(dd),
            coverage=q(d['actual_affected_fraction'] for d in dd),
            eligibleLength=q(d['eligible_length_px'] for d in dd),
            peak=q(d['applied_max_depth_px'] for d in dd),
            plannedCoverage=q(d['field_support_fraction'] for d in dd)))
    evidence.update(schema='layered-seam-review/13',layered=True,corrosionLayerRows=layer,
        backgroundRows=micro,backgroundSummary=dict(fragments=len(light),
            actualCoverage=q(d['actual_affected_fraction'] for d in light),
            targetFraction=.7,tolerance=.02,coverageIsPerFragmentLengthNotSampleProbability=True),
        damageRecipeExclusivity=dict(partial_pairs=len(p),partial_with_weather=sum(bool(r['augmentation']['damage']) for r in p),
            partial_with_primary_weak=sum(bool(r['augmentation']['weak_overlay']) for r in p),
            partial_with_light=sum(bool(r.get('background_degradation')) for r in p),
            all_mirror_before_damage=all(r['augmentation_layers']['mirror_before_damage'] for r in rows)),
        fragmentOccurrences=2*len(rows),statisticalUnit='pair; fragmentOccurrences counts occurrences, not distinct source fragments',
        partialRetainedFraction=q(r['augmentation']['partial']['source_seam_retention'] for r in p if r['label']))
    audit_path=Path(evidence['root'])/'independent_pixel_audit.json'
    if audit_path.exists():
        evidence['pixelAudit']={k:v for k,v in read(audit_path).items() if k!='rows'}
    if evidence['protocol']['distribution_revision'].get('partial_min_smaller_perimeter_fraction'):
        evidence['schema']='layered-seam-review/14'
        subtypes=[]
        for mode,name in (('end','端部裁切'),('middle','中段切除、保留两侧')):
            positive=[r for r in p if r['label'] and r['augmentation']['partial']['mode']==mode]
            negative=[r for r in p if not r['label'] and r['augmentation']['partial']['mode']==mode]
            details=[r['augmentation']['partial'] for r in positive]
            receipts=[d['support_constraint'] for d in details]
            subtypes.append(dict(key=mode,name=name,positive=len(positive),negative=len(negative),
                ratio=q(d['common_over_smaller_perimeter'] for d in receipts),
                ratioMinimum=min(d['common_over_smaller_perimeter'] for d in receipts),
                ratioMaximum=max(d['common_over_smaller_perimeter'] for d in receipts),
                commonLength=q(d['common_retained_length_px'] for d in receipts),
                smallerPerimeter=q(d['full_perimeter_px'][d['smaller_fragment']] for d in receipts),
                minFlank=q(min(d['retained_flanks_px']) for d in details if 'retained_flanks_px' in d),
                removedMiddle=q(d['removed_middle_arc_px'] for d in details if 'removed_middle_arc_px' in d)))
        evidence['partialSubtypeRows']=subtypes
        evidence['partialConstraintDefinition']='After primary crop, before light: minimum of bilateral surviving original TRAIN-supported arcs / full perimeter of smaller-by-area fragment. Removed middle and new cut receive no seam credit. Only positive pairs have this ratio.'


def figure(root,entry,row,out,key):
    if 'augmentation_layers' in entry:
        from .layered_figure import figure as layered_figure
        return layered_figure(root,entry,row,out,key)
    sample,report=load_sample(root/entry['artifact_path']);axis=entry['offline_paired_mirror']
    layered='augmentation_layers' in entry
    with np.load(root/entry['weather_artifact'],allow_pickle=False) as z:
        masks=[np.unpackbits(z['packed_preweather_'+s],axis=1).astype(np.float32)[None] for s in 'ab']
        t=z['translation_a_to_b_rc'].copy()
        if axis and str(z['coordinate_frame'])=='pre_mirror_800px':
            masks=[np.flip(m,axis=2 if axis=='horizontal' else 1).copy() for m in masks]
            t[1 if axis=='horizontal' else 0]*=-1
        before=replace(sample,mask_a=masks[0],mask_b=masks[1],translation_a_to_b_rc=t)
        selected=next((s for s in 'ab' if s+'_total' in z),None)
        field={k:z[selected+'_'+k].copy() for k in ('points','arc','edge','total','major','weak','eligible')} if selected else None
    fig,axs=plt.subplots(2,2,figsize=(13,8),gridspec_kw={'height_ratios':[1.25,1]})
    left,right,zoom,curve=axs.ravel()
    suffix='same GT pose' if sample.label else 'unplaced negative / NO GT layout'
    display(left,before,'Before corrosion / '+suffix)
    display(right,sample,'After corrosion / '+suffix)
    right.set_xlim(left.get_xlim());right.set_ylim(left.get_ylim())
    partial_only=layered and row['recipe']=='partial'
    if selected or partial_only:
        if partial_only:selected=report['compound']['partial']['sides'][0 if sample.label else 1]
        original=getattr(before,'mask_'+selected)[0].astype(bool);now=getattr(sample,'mask_'+selected)[0].astype(bool)
        rgb=np.ones((*original.shape,3));rgb[original]=[.68,.72,.73];rgb[now]=[.12,.30,.35];rgb[original & ~now]=[.90,.28,.19]
        zoom.imshow(rgb,interpolation='nearest')
        changed=np.argwhere(original & ~now);lo=changed.min(0);hi=changed.max(0);center=(lo+hi)/2
        radius=max(20.,float(np.max(hi-lo)/2)+8)
        zoom.set_xlim(center[1]-radius,center[1]+radius);zoom.set_ylim(center[0]+radius,center[0]-radius)
        zoom.set_aspect('equal');zoom.set_title('Selected edge: red = removed material (px)')
    if field is not None:
        active=field['total']>0;arcs=field['arc'];perimeter=field['edge'].sum()
        active_arcs=np.sort(arcs[active]);gaps=np.diff(np.r_[active_arcs,active_arcs[0]+perimeter]);start=active_arcs[(np.argmax(gaps)+1)%len(active_arcs)]
        u=(arcs-start+10)%perimeter;order=np.argsort(u);span=float((perimeter-gaps.max())+20)
        ix=order[u[order]<=span]
        curve.plot(u[ix],field['major'][ix],color='#216778',label='Major damage',linewidth=1.4)
        curve.plot(u[ix],field['weak'][ix],color='#ad762c',label='Smooth weak',linewidth=1.4)
        curve.plot(u[ix],field['total'][ix],color='#cb4836',label='Final capped field',linewidth=1.5,linestyle='--')
        curve.set_ylim(-.2,10);curve.set_xlabel('Original contour arc (px)');curve.set_ylabel('Inward recession (px)')
        curve.legend(fontsize=8);curve.grid(alpha=.2);curve.set_title('Continuous original-arc depth field')
    else:
        curve.axis('off')
        if partial_only:
            info=report['compound']['partial']
            curve.text(.02,.8,'Partial Seam ONLY\nNo weak / wave / local / notch overlay\n'+
                f'Original seam retained: {info["source_seam_retention"]:.1%}' if sample.label else
                'Partial-shaped negative\nNo weak / wave / local / notch overlay\nNO GT seam',fontsize=12,va='top')
        else:
            zoom.axis('off');zoom.text(.03,.8,'No seam damage applied\nFragment scale / mirror only' if layered else
                'No corrosion applied\nPartial / scale / mirror may still be present',fontsize=13)
    fig.suptitle(f'{"Layered v13" if layered else "Conservative v12"} | {row["recipe"]} | positive={sample.label} | Partial={row["partial"]}',fontsize=14)
    fig.tight_layout();file=out/(key+'.png');fig.savefig(file,dpi=115);plt.close(fig)
    dd=[d for d in report['compound']['damage'].values() if d.get('applied')]
    d=dd[0] if dd else None
    caption=f"{NAMES[row['recipe']]}；{'正例：原GT不变' if sample.label else '明确负例：分开展示，无GT摆放'}；Partial={'是' if row['partial'] else '否'}；镜像={axis or '无'}。"
    if d:
        caption+=f"实际最大退蚀{d['applied_max_depth_px']:.2f}px；受影响弧长{d['affected_length_px']:.1f}/{d['eligible_length_px']:.1f}px（{d['affected_fraction']:.1%}）。"
        if d['weak_applied']:caption+=f"弱腐蚀为{'渐进波状' if d['weak_mode']=='weak_gradual' else '整段平滑内收'}，额外删除{d['weak_independently_removed_pixels']}像素。"
        if 'gaps' in row['recipe']:caption+=f"实际缺口{d['notch_count']}处。"
    caption+=('左上图已完成碎片层级的缩放与镜像，尚未施加任何损伤；右上只增加所选损伤。Partial不叠加任何腐蚀。' if layered else
        '左上图为Partial/共同缩放后的腐蚀前视图，与右上同尺度；')
    caption+='红色为真实删除材料；有深度曲线时来自实际生成场，不是模型预测。'
    return dict(id=key,pair_id=entry['pair_id'],recipe=row['recipe'],name=NAMES[row['recipe']],
        label=bool(sample.label),partial=bool(row['partial']),caption=caption,
        source_stratum=entry['source_stratum'],artifact=str(root/entry['artifact_path']),
        weatherSource=str(root/entry['weather_artifact']),
        image='data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--per-type',type=int,default=10)
    a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    evidence,manifest,rows=summarise(root);metrics={r['pair_id']:r for r in rows};entries=manifest['entries'];groups=[];cases=[];used={}
    add_layer_statistics(evidence,manifest,rows)
    predicates=[(r,NAMES[r],lambda e,r=r:e['corrosion_recipe']==r) for r in evidence['protocol']['recipe_percent']]
    if evidence.get('partialSubtypeRows'):
        predicates=[('partial_middle','Partial：中段切除／两侧保留',lambda e:e['label'] and e['partial_applied'] and metrics[e['pair_id']]['augmentation']['partial']['mode']=='middle'),
                    ('partial_end','Partial：端部裁切',lambda e:e['label'] and e['partial_applied'] and metrics[e['pair_id']]['augmentation']['partial']['mode']=='end')]+predicates
    if not evidence.get('layered'):predicates += [('partial','曲线Partial',lambda e:e['partial_applied'] and e['label'])]
    predicates += [('mirror_h','水平镜像',lambda e:e['offline_paired_mirror']=='horizontal' and e['label']),
        ('mirror_v','垂直镜像',lambda e:e['offline_paired_mirror']=='vertical' and e['label']),
        ('area_ratio','面积悬殊（面积比≤1:4）',lambda e:e['label'] and metrics[e['pair_id']]['area_ratio']<=.25)]
    for s in sorted(set(e['source_stratum'] for e in entries if e['label'] and e['source_stratum']!='native')):
        predicates.append(('structure_'+s,s,lambda e,s=s:e['label'] and e['source_stratum']==s))
    for k in NEG:
        predicates.append(('negative_'+k,NEG[k],lambda e,k=k:not e['label'] and e['negative_kind']==k))
    for group,name,predicate in predicates:
        pool=[e for e in entries if predicate(e)]
        if len(pool)<a.per_type:raise ValueError(f'{group}: only {len(pool)} examples')
        selected=[]
        # Five positive and five negative where both apply; fixed quantile
        # spread over actual area, not just attractive examples.
        labels=[True,False] if all(any(e['label']==v for e in pool) for v in (True,False)) else [pool[0]['label']]
        for label in labels:
            pp=[e for e in pool if e['label']==label]
            pp.sort(key=lambda e:metrics[e['pair_id']]['mean_fragment_area_px'])
            count=a.per_type//len(labels)
            selected.extend(pp[i] for i in np.linspace(0,len(pp)-1,count).astype(int))
        ids=[]
        for entry in selected:
            pid=entry['pair_id']
            if pid not in used:
                key='case_'+str(len(cases)+1).zfill(3)
                cases.append(figure(root,entry,metrics[pid],out,key));used[pid]=key
            ids.append(used[pid])
        family='corrosion' if group in evidence['protocol']['recipe_percent'] or group in ('partial_middle','partial_end') else 'negative_source' if group.startswith('negative_') else 'fragment'
        groups.append(dict(id=group,name=name,caseIds=ids,count=len(ids),layer=family))
    evidence['caseGroups']=groups;evidence['uniqueCases']=len(cases)
    save_json(out/'evidence.json',evidence);save_json(out/'showcases.json',cases)
    print(json.dumps(dict(status='ready',pairs=evidence['pairs'],groups=len(groups),cases=len(cases),per_type=a.per_type)))


if __name__=='__main__':main()

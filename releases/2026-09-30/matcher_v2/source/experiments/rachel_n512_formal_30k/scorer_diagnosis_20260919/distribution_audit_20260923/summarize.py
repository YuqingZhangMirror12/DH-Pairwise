"""Reproducible descriptive summaries, without changing any training artifact."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np


def stats(values):
    a=np.array([x for x in values if x is not None and np.isfinite(x)],float)
    if not len(a):return dict(n=0,missing=len(values))
    return dict(n=len(a),missing=len(values)-len(a),mean=float(a.mean()),
        **{k:float(v) for k,v in zip(('p10','p25','median','p75','p90'),np.quantile(a,[.1,.25,.5,.75,.9]))},
        zero_count=int((a==0).sum()),min=float(a.min()),max=float(a.max()))


def csv_write(path, rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def run(root):
    root=Path(root)
    sim=[json.loads(x) for x in (root/'sim_pairs.jsonl').read_text().splitlines()]
    real=json.loads((root/'real_pairs.json').read_text())
    full=json.loads((root/'dunhuang_full_pairs.json').read_text())
    assert len(sim)==24000 and sum(x['label'] for x in sim)==12000
    assert len({x['pair_id'] for x in sim})==24000
    assert len(full)==508 and sum(x['retained'] for x in full)==295
    assert not any('d20_length_px' in x for x in real if x['dataset']=='turufan')
    cohorts={
        'S7 positive':[x for x in sim if x['label']],
        'S7-H positive':[x for x in sim if x['label'] and x['s7_h']],
        'Dunhuang retained':[x for x in real if x['dataset']=='dunhuang_cv' and x['label']],
        'Dunhuang full':full,
        'Dunhuang excluded':[x for x in full if not x['retained']],
        'Turufan positive':[x for x in real if x['dataset']=='turufan' and x['label']],
        'S7 negative':[x for x in sim if not x['label']],
        'S7-H negative':[x for x in sim if not x['label'] and x['s7_h']],
        'Dunhuang negative':[x for x in real if x['dataset']=='dunhuang_cv' and not x['label']],
        'Turufan negative':[x for x in real if x['dataset']=='turufan' and not x['label']]}
    recipes=sorted({x['recipe'] for x in sim})
    for recipe in recipes:
        cohorts['S7 '+recipe]=[x for x in sim if x['label'] and x['recipe']==recipe]
        cohorts['S7 '+recipe+' applied']=[x for x in sim if x['label'] and x['recipe']==recipe and x['augmentation']['changed']]
    metrics=['area_ratio','mean_fragment_area_px','area_px_a','area_px_b','perimeter_px_a','perimeter_px_b',
        'd20_contact_fraction_min','d20_contact_fraction_max','d20_contact_fraction_asymmetry','d40_gap_mean_over_sqrt_smaller_area']
    metrics += [f'd{d}_{m}' for d in (4,10,20,40,64) for m in ('length_px','longest_px','gap_mean_px','gap_median_px','gap_p90_px','segments_mean','breaks_mean')]
    output={};metric_rows=[];rates=[]
    for name, rows in cohorts.items():
        if not rows:continue
        result=dict(pair_count=len(rows),metrics={})
        for metric in metrics:
            s=stats([x.get(metric) for x in rows]); result['metrics'][metric]=s
            metric_rows.append(dict(cohort=name,metric=metric,**s))
        # Endpoint occurrence weighted fragment distribution, not unique identities.
        for metric in ('area_px','perimeter_px','roughness_raw_over_smooth'):
            s=stats([x[metric+'_'+side] for x in rows for side in 'ab'])
            result['metrics']['endpoint_'+metric]=s
            metric_rows.append(dict(cohort=name,metric='endpoint_'+metric,**s))
        predicates={'area_ratio_lt_1_4':lambda x:x['area_ratio']<.25,
            'area_ratio_lt_1_8':lambda x:x['area_ratio']<.125}
        if 'd20_length_px' in rows[0]:
            predicates.update(contact20_lt64=lambda x:x['d20_length_px']<64,
                contact20_lt128=lambda x:x['d20_length_px']<128,
                contact20_lt256=lambda x:x['d20_length_px']<256,
                contact20_longest_lt128=lambda x:x['d20_longest_px']<128,
                contact20_fraction_min_lt10pct=lambda x:x['d20_contact_fraction_min']<.1,
                contact10_has_break=lambda x:x['d10_breaks_mean']>0,
                contact20_has_break=lambda x:x['d20_breaks_mean']>0,
                gap40_mean_gt5=lambda x:x['d40_gap_mean_px'] is not None and x['d40_gap_mean_px']>5,
                gap40_mean_2_to_10=lambda x:x['d40_gap_mean_px'] is not None and 2<=x['d40_gap_mean_px']<=10,
                gap40_mean_gt10=lambda x:x['d40_gap_mean_px'] is not None and x['d40_gap_mean_px']>10,
                no_40px_contact_proxy=lambda x:x['d40_gap_mean_px'] is None)
        result['rates']={}
        for key,pred in predicates.items():
            n=sum(bool(pred(x)) for x in rows)
            result['rates'][key]=dict(n=n,denominator=len(rows),fraction=n/len(rows))
            rates.append(dict(cohort=name,metric=key,count=n,denominator=len(rows),fraction=n/len(rows)))
        output[name]=result
    aug=[]
    for recipe in recipes:
        for label in (True,False):
            rows=[x for x in sim if x['recipe']==recipe and x['label']==label]
            changed=[x for x in rows if x['augmentation']['changed']]
            aug.append(dict(recipe=recipe,label=label,requested=len(rows),changed=len(changed),
                fraction_of_all_12k=len(changed)/12000,acceptance=len(changed)/len(rows),
                measured_max_depth=stats([x['augmentation']['measured_max_erosion_depth_px'] for x in changed]),
                gap_count=stats([x['augmentation']['applied_gap_count_per_pair'] for x in changed]),
                partial_retention=stats([x['augmentation'].get('source_seam_retention') for x in changed]),
                partial_source_seam_length=stats([x['augmentation'].get('source_seam_length_px') for x in changed]),
                partial_retained_seam_length=stats([x['augmentation'].get('physically_retained_source_seam_length_px') for x in changed]),
                fallback_counts=dict(Counter(x['augmentation']['fallback_reason'] for x in rows if not x['augmentation']['changed']))))
    fragment_rows=json.loads((root/'real_fragments.json').read_text())
    unique=[]
    for ds in ('dunhuang_cv','turufan'):
        ids={x['fragment_'+s+'_id'] for x in real if x['dataset']==ds and x['label'] for s in 'ab'}
        fr=[x for x in fragment_rows if x['dataset']==ds and x['fragment_id'] in ids]
        assert len(fr)==len(ids)
        unique.append(dict(dataset=ds,positive_unique_fragments=len(fr),
            area=stats([x['area_px'] for x in fr]),perimeter=stats([x['perimeter_px'] for x in fr])))
    report=dict(cohorts=output,augmentation=aug,unique_positive_fragments=unique,
        source_strata={str(label):dict(Counter(x['source_stratum'] for x in sim if x['label']==label)) for label in (True,False)},
        notes=['All seam measures use GT/masks only, never inferred model matches.',
               'Turufan seam statistics are deferred by user instruction.',
               'S7-H is nested in S7, not an independent sample.',
               'Gap40 is conditional on detected near-facing boundary within 40px, not material lost.',
               'Breaks are mean counts across two sides of 8-100px internal unsupported arcs at distance10px; no ancestral truth implied.',
               'Real partial-seam ancestry and pre-erosion contours are not available.'])
    (root/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    csv_write(root/'distribution_quantiles.csv',metric_rows);csv_write(root/'distribution_rates.csv',rates)
    compact_keys=['pair_id','dataset','label','recipe','s7_h','area_ratio','mean_fragment_area_px','d4_length_px','d10_length_px','d20_length_px','d40_gap_mean_px','d40_gap_median_px','d10_breaks_mean','d20_breaks_mean','d20_contact_fraction_min','d20_contact_fraction_max','d20_contact_fraction_asymmetry']
    csv_write(root/'pair_metrics.csv',[{k:x.get(k) for k in compact_keys} for x in sim+real])
    # Keep plotting data compact and portable; no model results or human labels included.
    plot_metrics=['d20_length_px','d40_gap_mean_px','d10_breaks_mean','mean_fragment_area_px','area_ratio','d20_contact_fraction_min']
    (root/'plot_data.json').write_text(json.dumps({k:[{m:x.get(m) for m in plot_metrics} for x in cohorts[k]]
        for k in ['S7 positive','S7-H positive','Dunhuang retained','Dunhuang full','Turufan positive']},allow_nan=False))
    for name in ['S7 positive','S7-H positive','Dunhuang retained','Dunhuang full','Turufan positive']:
        c=output[name];print(name,c['pair_count'])
        print(json.dumps(c['rates']))
    print('Unique positive fragments',unique)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)

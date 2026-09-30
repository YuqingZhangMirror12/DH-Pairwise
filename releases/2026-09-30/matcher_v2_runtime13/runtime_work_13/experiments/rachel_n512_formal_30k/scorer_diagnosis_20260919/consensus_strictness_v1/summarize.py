"""Aggregate completed phase1 only. No hyperparameter fitting or model calls."""
from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def quant(values):
    a=np.asarray([x for x in values if x is not None],dtype=float)
    if not len(a): return dict(n=0,min=None,p10=None,p25=None,p50=None,p75=None,p90=None,p95=None,max=None,mean=None)
    assert np.isfinite(a).all()
    return dict(zip(['min','p10','p25','p50','p75','p90','p95','max'],np.quantile(a,[0,.1,.25,.5,.75,.9,.95,1]).tolist()),n=len(a),mean=float(a.mean()))


def histogram(values,limits):
    a=np.array([x for x in values if x is not None],dtype=float)
    return dict(boundaries=limits,counts=np.histogram(a,bins=limits)[0].tolist(),n=len(a))


def short_row(r):
    return {k:v for k,v in r.items() if k not in ('hypotheses','clusters','cluster_pairs','merge_trace')}


def group_summary(rows):
    scalar=['median_contour_spacing_a_px','median_contour_spacing_b_px','median_edgecloud_spacing_px',
        'common_center_radius_px','pair_prescreen_distance_px','hypothesis_count','nonempty_hypothesis_count',
        'prebudget_cluster_count','retained_cluster_count','dropped_cluster_count','absolute_q_mass',
        'largest_sparse_cluster_mass_share','largest_fullq_share_union']
    out=dict(pairs=len(rows),positives=sum(r['label'] for r in rows),negatives=sum(not r['label'] for r in rows),
        numeric_invalid=sum(not r['numeric_valid'] for r in rows),
        scalars={k:quant([r[k] for r in rows]) for k in scalar},
        tangent_half_bandwidth_pair_medians=quant([r['tangent_half_bandwidth_px']['p50'] for r in rows]),
        retained_cluster_count_histogram=dict(sorted(Counter(r['retained_cluster_count'] for r in rows).items())))
    cps=[x for r in rows for x in r['cluster_pairs'] if x['both_retained']]
    # These are descriptive sensitivity bins; none becomes a merge rule.
    out['redundancy']=dict(cluster_pairs=len(cps),
        pairs_with_exact_duplicate_support=sum(any(x['both_retained'] and x['equal_nonempty_edges'] for x in r['cluster_pairs']) for r in rows),
        exact_duplicate_cluster_pairs=sum(x['equal_nonempty_edges'] for x in cps),
        pairs_with_jaccard_080=sum(any(x['both_retained'] and (x['jaccard'] or 0)>=.8 for x in r['cluster_pairs']) for r in rows),
        pairs_with_jaccard_090=sum(any(x['both_retained'] and (x['jaccard'] or 0)>=.9 for x in r['cluster_pairs']) for r in rows),
        near_duplicate_jaccard090_pose_distance=quant([x['pose_distance_px'] for x in cps if (x['jaccard'] or 0)>=.9]),
        near_duplicate_jaccard090_normal_shift=quant([x['shared_normal_shift_px'] for x in cps if (x['jaccard'] or 0)>=.9]),
        near_duplicate_jaccard090_tangent_shift=quant([x['shared_tangent_shift_px'] for x in cps if (x['jaccard'] or 0)>=.9]),
        cluster_pair_pose_distance=quant([x['pose_distance_px'] for x in cps]))
    gt=[r for r in rows if r['label'] and r['gt_known']]
    if gt:
        gd=[r['gt_diagnostic'] for r in gt]
        covered_h=[r for r in gt if r['gt_diagnostic']['correct_hypothesis_count']]
        covered_c=[r for r in gt if r['gt_diagnostic']['coverage_retained']]
        herrors=[h['gt_error_px'] for r in gt for h in r['hypotheses'] if h['edge_ids']]
        cerrors=[c['gt_error_px'] for r in gt for c in r['clusters'] if c['retained']]
        near_distances=[]; all_correct_distances=[]; nearest_intercluster_distances=[]
        for r in gt:
            h=[x for x in r['hypotheses'] if x['edge_ids'] and x['gt_error_px']<=20]
            if len(h)>1:
                pts=np.array([x['translation_rc'] for x in h]);d=np.linalg.norm(pts[:,None]-pts[None],axis=-1)
                all_correct_distances.extend(d[np.triu_indices(len(h),1)].tolist())
                np.fill_diagonal(d,np.inf);near_distances.extend(d.min(axis=1).tolist())
            c=[x for x in r['clusters'] if x['retained'] and x['contains_correct_hypothesis']]
            if len(c)>1:
                pts=np.array([x['translation_rc'] for x in c]);d=np.linalg.norm(pts[:,None]-pts[None],axis=-1)
                np.fill_diagonal(d,np.inf);nearest_intercluster_distances.extend(d.min(axis=1).tolist())
        out['gt']=dict(positive_pairs=len(gt),hypothesis_coverage=sum(x['correct_hypothesis_count']>0 for x in gd),
            retained_coverage=sum(x['coverage_retained'] for x in gd),prebudget_coverage=sum(x['coverage_prebudget'] for x in gd),
            top_cluster_correct=sum(x['top_cluster_correct'] for x in gd),
            split_correct_pairs=sum(x['correct_hypothesis_cluster_count']>1 for x in gd),
            split_correct_pairs_prebudget=sum(x['correct_hypothesis_cluster_count_prebudget']>1 for x in gd),
            mixed_pairs=sum(x['mixed_cluster_count']>0 for x in gd),mixed_clusters=sum(x['mixed_cluster_count'] for x in gd),
            mixed_clusters_prebudget=sum(x['mixed_cluster_count_prebudget'] for x in gd),
            correct_hypotheses_dropped=sum(x['correct_hypotheses_dropped'] for x in gd),
            correct_hypothesis_count=quant([x['correct_hypothesis_count'] for x in gd]),
            correct_hypothesis_cluster_count=quant([r['gt_diagnostic']['correct_hypothesis_cluster_count'] for r in covered_h]),
            original_union_coverage={s:quant([r['gt_diagnostic']['largest_correct_original_union_coverage'][s] for r in covered_h]) for s in ('edges','endpoints_a','endpoints_b')},
            compatible_coverage={s:quant([r['gt_diagnostic']['largest_correct_compatible_coverage'][s] for r in covered_h]) for s in ('edges','endpoints_a','endpoints_b')},
            fullq_coverage={s:quant([r['gt_diagnostic']['largest_correct_fullq_coverage'][s] for r in covered_h]) for s in ('edges','endpoints_a','endpoints_b')},
            endpoint_coverage_under_090=sum(any((r['gt_diagnostic']['largest_correct_compatible_coverage'][s] or 0)<.9 for s in ('endpoints_a','endpoints_b')) for r in covered_h),
            hypothesis_gt_error=quant(herrors),cluster_gt_error=quant(cerrors),
            hypothesis_error_bins=histogram(herrors,[0,5,10,20,40,80,160,320,1000000]),
            cluster_error_bins=histogram(cerrors,[0,5,10,20,40,80,160,320,1000000]),
            correct_hypothesis_all_pose_distances=quant(all_correct_distances),
            correct_hypothesis_nearest_distance=quant(near_distances),
            correct_cluster_nearest_distance=quant(nearest_intercluster_distances))
    else: out['gt']=None
    neg=[r for r in rows if not r['label']]
    out['negatives']=dict(pairs=len(neg),no_cluster=sum(r['retained_cluster_count']==0 for r in neg),
        retained_cluster_count=quant([r['retained_cluster_count'] for r in neg]),
        largest_sparse_cluster_mass_share=quant([r['largest_sparse_cluster_mass_share'] for r in neg]),
        largest_fullq_share_union=quant([r['largest_fullq_share_union'] for r in neg]),
        largest_fullq_absolute_mass=quant([r['clusters'][0]['full_q_weighted_mass'] for r in neg if r['clusters']]),
        largest_fullq_fraction_of_all_Q=quant([r['clusters'][0]['full_q_absolute_mass_fraction'] for r in neg if r['clusters']]))
    retained=[c for r in rows for c in r['clusters'] if c['retained']]
    out['evidence_retention']=dict(retained_clusters=len(retained),
        original_union_to_compatible_count=quant([x['sparse_union_recall'] for x in retained]),
        original_union_to_fullq_count=quant([x['full_q_union_recall'] for x in retained]),
        lost_original_explained_mass=quant([x['reference_mass_lost_fraction'] for x in retained]))
    parity=[r['live_cache_parity'] for r in rows if r.get('live_cache_parity')]
    out['cpu_gpu_cache_parity']=dict(pairs=len(parity),
        changed_cluster_counts=sum(p['cached_clusters']!=p['measured_clusters'] for p in parity),
        sparse_q_max_abs_difference=quant([p['q_max_abs_difference'] for p in parity]),
        edge_jaccard=quant([p['edge_jaccard'] for p in parity]),
        ordered_pose_max_difference=quant([p['ordered_pose_max_difference'] for p in parity]))
    return out


def run(root):
    root=Path(root);complete=read(root/'complete.json');protocol=read(root/'protocol.json')
    assert complete['status']=='complete'
    groups={};allrows=[]
    for split,n in protocol['expected'].items():
        rows=[read(root/split/(str(i).zfill(5)+'.json')) for i in range(n)]
        assert len({r['pair_id'] for r in rows})==n
        assert all(r['split']==split and r['index']==i for i,r in enumerate(rows))
        groups[split]=group_summary(rows)
        if split=='dunhuang_cv':
            excluded=[r for r in rows if r['gt_excluded']]
            assert len(excluded)==3 and sum(r['label'] for r in excluded)==3
            groups['dunhuang_corrected']=group_summary([r for r in rows if not r['gt_excluded']])
        if split=='turufan': assert not any(r['gt_known'] for r in rows)
        allrows.extend(rows)
    # Fixed cases were registered before inference; main252 denotes two pairs.
    by_id={r['pair_id']:r for r in allrows}
    cases=[]
    for c in protocol['cases']['cases']:
        r=by_id[c['pair_id']]
        cases.append(dict(alias=c['alias'],**short_row(r),hypotheses=r['hypotheses'],clusters=r['clusters'],cluster_pairs=r['cluster_pairs']))
    report=dict(status='complete',protocol=protocol,completion=complete,groups=groups,
        synthetic=read(root/'synthetic_production_scale.json'),
        strata_select={recipe:group_summary([r for r in allrows if r['split']=='sim_select' and r['recipe']==recipe])
            for recipe in sorted({r['recipe'] for r in allrows if r['split']=='sim_select'})})
    for name,data in [('summary.json',report),('fixed_cases.json',cases),('pair_summary.json',[short_row(r) for r in allrows])]:
        (root/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:dict(pairs=v['pairs'],gt=v['gt'],redundancy=v['redundancy']) for k,v in groups.items()},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)

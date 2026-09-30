"""Observed short-support, pooling, layer and gradient evidence; no model fitting."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def stats(xs):
    xs=[float(x) for x in xs if x is not None and np.isfinite(x)]
    return {'n':len(xs),'q10_q25_median_q75_q90':np.quantile(xs,[.1,.25,.5,.75,.9]).tolist() if xs else None}
def winner(r): return r['candidates'][r['winner_index']] if r['has_candidate'] else None


def run(root):
    root=Path(root);probes=read(root/'probes_v3/records.json');grads=read(root/'gradient_probe/records.json')
    repair={p['pair_id']:p for p in read(root/'probes_v3_turufan_unique/records.json')}
    probes=[repair.get(p['pair_id'],p) for p in probes]
    out={'definitions':{'short_support':'candidate correspondence edge count, not true seam length',
        'pooling':'actual learned attention weights; not causal attribution',
        'gradient':'60 recipe-by-label training samples, eval activations, no optimizer update',
        'diagnostic_sample':'36 outcome-stratified cases, not representative prevalence estimate'},'cohorts':{}}
    for split in ('dunhuang_cv','turufan','sim_test'):
        rr=rows(root/split/'case_diagnostics.jsonl')
        groups={'positive_accepted':[r for r in rr if r['label'] and r['score']>=.47],
                'positive_rejected':[r for r in rr if r['label'] and r['score']<.47],
                'negative_accepted':[r for r in rr if not r['label'] and r['score']>=.47],
                'negative_rejected':[r for r in rr if not r['label'] and r['score']<.47]}
        if split!='turufan':
            groups.update(correct_layout_rejected=[r for r in rr if r['label'] and r['layout20'] and r['score']<.47],
                          correct_candidate_wrong_winner=[r for r in rr if r['label'] and r['refined_coverage8'] and not r['layout20']],
                          no_correct_candidate=[r for r in rr if r['label'] and not r['refined_coverage8']])
        ds={}
        for name,rs in groups.items():
            cc=[winner(r) for r in rs if r['has_candidate']]
            ec=[c['edge_count'] for c in cc]
            ds[name]=dict(n=len(rs),has_candidate=len(cc),edges=stats(ec),single_edge=sum(x==1 for x in ec),
                up_to3=sum(x<=3 for x in ec),up_to8=sum(x<=8 for x in ec),
                arc_points=stats([min(c['arc_point_counts']) for c in cc]),
                score=stats([r['score'] for r in rs]),residual=stats([c['residual_median_px'] for c in cc]),
                overlap=stats([c['evidence'][5] for c in cc]),q1=stats([c['evidence'][0] for c in cc]),
                quality=stats([c['quality_logit'] for c in cc]),null=stats([c['null_logit'] for c in cc]))
        out['cohorts'][split]=ds
    pg=defaultdict(list)
    for p in probes:pg[p['reason']].append(p)
    out['pooling_by_stratum']={}
    for name,pp in pg.items():
        pools=[x for p in pp for x in p['pooling']]
        d={'cases':len(pp)}
        for key in ('support_mass','internal_gap_mass','exterior_context_mass','attention80_span_px','support_arc_span_px','effective_points','point_count','support_points'):
            d[key]=stats([x[key] for x in pools])
        d['effective_point_fraction']=stats([x['effective_points']/x['point_count'] for x in pools if x['point_count']])
        d['support_mass_relative_to_uniform']=stats([x['support_mass']/(x['support_points']/x['point_count']) for x in pools if x['support_points']])
        d['interventions']={}
        for v in ('residual_half','residual_zero','overlap_zero','dustbin_zero'):
            pairs=[(p['prediction']['score'],p['diagnostic_interventions'][v]) for p in pp if v in p['diagnostic_interventions']]
            d['interventions'][v]=dict(delta=stats([b-a for a,b in pairs]),rescued_at047=sum(a<.47<=b for a,b in pairs),
                                       lost_at047=sum(b<.47<=a for a,b in pairs))
        out['pooling_by_stratum'][name]=d
    out['gradient_by_group']={}
    gg={'all':grads,'positive':[r for r in grads if r['label']],'negative':[r for r in grads if not r['label']]}
    for recipe in sorted({r['recipe'] for r in grads}):gg['recipe:'+recipe]=[r for r in grads if r['recipe']==recipe]
    for name,rs in gg.items():
        out['gradient_by_group'][name]={}
        for part in ('encoder','context','shared_total'):
            cos=[r['gradients'][part]['cosine'] for r in rs if r['gradients'][part]['cosine'] is not None]
            out['gradient_by_group'][name][part]=dict(cosine=stats(cos),negative=sum(x<0 for x in cos),strong_negative=sum(x<-.1 for x in cos))
    old={r['pair_id']:r for r in read(root/'probes_baseline_strict/predictions.json')}
    parity=[];valid_mismatches=[]
    for file in ('matched_tokens_real_cv','matched_tokens_turufan'):
        for r in read(root/'baseline'/f'{file}.json')['rows']:
            p=old[r['pair_id']];parity.append(abs(r['score']-p['score']))
            if r['decision_valid']!=p['decision_valid']:valid_mismatches.append(r['pair_id'])
    out['parity']=dict(baseline_count=len(parity),baseline_max_score_delta=max(parity),baseline_valid_mismatches=valid_mismatches,
        v3_max_score_delta=max(p['parity']['score_delta'] for p in probes),
        v3_max_pose_delta=max(p['parity']['translation_delta_px'] or 0 for p in probes))
    out['layer_proxies']=[dict(pair_id=p['pair_id'],reason=p['reason'],**p['proxies'],broad_edges=p['broad_edges']) for p in probes]
    (root/'mechanism_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for split,groups in out['cohorts'].items():
        for group,d in groups.items():
            print(split,group,'n',d['n'],'single',d['single_edge'],'<=3',d['up_to3'],'<=8',d['up_to8'],'edge quantiles',d['edges'])
    print('PARITY',out['parity'])
    for key in ('correct_layout_rejected','layout_regression','old_accepted_new_rejected'):
        print(key,out['pooling_by_stratum'].get(key))
    print('GRADS', {k:out['gradient_by_group'][k] for k in ('all','positive','negative')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)

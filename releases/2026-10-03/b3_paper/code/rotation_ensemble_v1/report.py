"""Posthoc CPU analysis. Development locks the choice BEFORE one TEST pass."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .core import combine,annotate,VIEW_SETS,METHODS,candidate_key,candidate_score
from .metrics import summarize,evaluate_setting
from .io import read,checked,sha,ref,save


def load_rows(root,phase,matcher):
    out=root/phase/matcher;done=read(out/'complete.json');ret=read(out/'actual_return.json')
    if ret['returncode']!=0 or done['models_before']!=done['models_after'] or checked(done['protocol'])!=read(root/'protocol.json'):
        raise ValueError('no verified successful inference')
    result={}
    for name,s in done['datasets'].items():
        if sha(s['predictions']['path'])!=s['predictions']['sha256']:raise ValueError('prediction bytes changed')
        rows=[json.loads(line) for line in Path(s['predictions']['path']).open()];target=checked(s['targets'])
        if len(rows)!=s['pairs'] or len(target)!=len(rows) or {r['pair_id'] for r in rows}!=set(target):raise ValueError('prediction population differs')
        for r in rows:
            r['views']={int(a):v for a,v in r['views'].items()}
            if set(r['views'])!=set(s['angles']) or any(v['pair_id']!=r['pair_id'] for v in r['views'].values()):raise ValueError('view identities differ')
            if any('label' in v or 'target_translation_rc' in v for v in r['views'].values()):raise ValueError('GT present in inference views')
        result[name]=(rows,target)
    return result


def annotate_compact(p,t,pid):
    x=annotate(p,t['label'],t['target'])
    for k in ('union_candidates','deduplicated_candidates'):x.pop(k)
    return dict(x,pair_id=pid,fold=t['fold'],seam_type=t['seam_type'])


def settings_rows(data,angles,head):
    """Share the expensive, GT-free pool dedup across A/B/C; same exact rules."""
    output={m:[] for m in METHODS};raw,targets=data
    for row in raw:
        views=row['views'];pid=row['pair_id'];b=combine(views,angles,'B',head)
        winners=[max(views[a]['candidates'],key=lambda c:candidate_key(c,head)) if views[a]['numeric_valid'] and views[a]['candidates'] else None for a in angles]
        a=dict(b,method='A');c=dict(b,method='C')
        if b['has_candidate']:
            w=max((w for w in winners if w is not None),key=lambda w:(w['q_sum'],-w['angle'],-w['index']))
            a.update(translation=list(w['translation']),winner=dict(angle=w['angle'],index=w['index']),score=candidate_score(w,head),
                agreement_views=int(sum(z is not None and np.linalg.norm(np.asarray(z['translation'])-w['translation'])<=16 for z in winners)))
            c['score']=sum(0. if w is None else candidate_score(w,head) for w in winners)/len(angles)
        for method,p in [('A',a),('B',b),('C',c)]:output[method].append(annotate_compact(p,targets[pid],pid))
    return output


def timing(data,angles,head):
    values=[];full=[]
    for r in data[0]:
        ts=[r['views'][a]['timing'] for a in angles]
        values.append(sum(t['matcher_seconds_per_pair']+t['builder_seconds']+t['head_seconds'].get(head,0.) for t in ts))
        full.append(sum(t['batch_amortized_seconds'] for t in ts))
    return dict(pairs=len(values),component_estimate_seconds_per_pair=float(np.mean(values)),
        component_p50_seconds=float(np.median(values)),all_four_heads_measured_amortized_seconds_per_pair=float(np.mean(full)),
        caveat='Component accounting from the shared four-head sweep, not a separate end-to-end latency benchmark; CPU aggregation excluded.')


def sensitivity(data):
    raw,targets=data;out={}
    for angle in (0,90,180,270):
        counts=dict(positives=0,coverage=0,layout=0,coverage_flips_vs_identity=0,layout_flips_vs_identity=0,rescued_coverage=0,lost_coverage=0)
        for row in raw:
            t=targets[row['pair_id']]
            if t['target'] is None:continue
            def status(a):
                cs=row['views'][a]['candidates'];w=max(cs,key=lambda c:(c['q_sum'],-c['index'])) if cs else None
                return (any(np.linalg.norm(np.asarray(c['translation'])-t['target'])<=20 for c in cs),
                        w is not None and np.linalg.norm(np.asarray(w['translation'])-t['target'])<=20)
            before=status(0);after=status(angle);counts['positives']+=1
            counts['coverage']+=int(after[0]);counts['layout']+=int(after[1])
            counts['coverage_flips_vs_identity']+=int(before[0]!=after[0]);counts['layout_flips_vs_identity']+=int(before[1]!=after[1])
            counts['rescued_coverage']+=int(after[0] and not before[0]);counts['lost_coverage']+=int(before[0] and not after[0])
        out[str(angle)]=counts
    return out


def selection_key(row):
    if row['head']=='q':raise ValueError('raw Q is not an eligible trained classifier')
    d=row['populations']['dunhuang_select']['development_empirical_roc']
    return (d['0.02']['correct_and_accepted'],row['populations']['dunhuang_select']['primary']['auc'],
            d['0.01']['correct_and_accepted'],d['0.05']['correct_and_accepted'],-len(row['angles']),
            -METHODS.index(row['method']),row['head'],row['matcher'])


def development(root):
    protocol=read(root/'protocol.json');report=[];directions={};started=time.time()
    if (root/'development_report.json').exists() or (root/'default.json').exists():raise ValueError('development report/default already frozen')
    for matcher in ('matcher_sim','matcher_endpoint'):
        data=load_rows(root,'development',matcher);directions[matcher]=sensitivity(data['dunhuang_cv'])
        for head in ('q','patch_sim','stats_sim','patch_real','stats_real'):
            for angles in VIEW_SETS:
                population={n:settings_rows(v,angles,head) for n,v in data.items()}
                for method in METHODS:
                    dun=population['dunhuang_cv'][method];sim=population['sim_cal'][method];turu=population['turufan'][method]
                    if any(r['fold']==0 for r in dun+turu):raise ValueError('TEST rows in development analysis')
                    cal=[r for r in dun if r['fold']==1] if head.endswith('_real') else sim
                    select=[r for r in dun if r['fold'] in (2,3,4)]
                    row=dict(matcher=matcher,head=head,angles=list(angles),method=method,
                        pairing='original_trained_pair' if matcher=='matcher_sim' or head=='q' else 'endpoint_Matcher_swap_with_existing_SIM_Matcher_trained_head',
                        populations=dict(dunhuang_development=evaluate_setting(dun,cal,head),
                            dunhuang_select=evaluate_setting(select,cal,head),turufan_development=evaluate_setting(turu,sim,head)),
                        calibration=dict(dunhuang='real_fold1' if head.endswith('_real') else 'original_SIM_CAL',turufan='original_SIM_CAL'),
                        timing={name:timing(data[name],angles,head) for name in ('dunhuang_cv','turufan')})
                    thresholds=row['populations']['dunhuang_development'];row['seam_types']={}
                    for seam in ('J','R','C','unknown'):
                        subset=[r for r in dun if r['label'] and r['seam_type']==seam]
                        row['seam_types'][seam]=None if not subset else dict(primary=summarize(subset,thresholds['primary']['threshold']),
                            frozen_cal_fpr={key:summarize(subset,value['threshold']) for key,value in thresholds['frozen_cal_fpr'].items()},
                            development_empirical_roc={key:summarize(subset,value['threshold']) for key,value in thresholds['development_empirical_roc'].items()})
                    report.append(row)
                print('analyzed',matcher,head,list(angles),flush=True)
    candidates=[r for r in report if r['head']!='q'];best=max(candidates,key=selection_key)
    matched=max((r for r in candidates if r['matcher']=='matcher_sim'),key=selection_key)
    save(root/'development_report.json',dict(protocol=ref(root/'protocol.json'),settings=report,direction_sensitivity=directions,
        elapsed_seconds=time.time()-started,selected_only_on='Dunhuang folds2,3,4',test_used=False,
        best_matched_pipeline={k:matched[k] for k in ('matcher','head','angles','method')},
        selection_overfit_caveat='120 development settings compared on a small, historically exposed real development set; no guarantee of unseen-domain gains.'))
    baseline=next(r for r in report if r['matcher']==best['matcher'] and r['head']==best['head'] and r['angles']==[0] and r['method']=='B')
    save(root/'default.json',dict(protocol_sha256=sha(root/'protocol.json'),development_report=ref(root/'development_report.json'),
        locked_unix=time.time(),setting={k:best[k] for k in ('matcher','head','angles','method','pairing')},
        thresholds={name:dict(primary=best['populations'][name]['primary']['threshold'],
            frozen_cal_fpr={r:v['threshold'] for r,v in best['populations'][name]['frozen_cal_fpr'].items()}) for name in ('dunhuang_development','turufan_development')},
        baseline_thresholds={name:dict(primary=baseline['populations'][name]['primary']['threshold'],
            frozen_cal_fpr={r:v['threshold'] for r,v in baseline['populations'][name]['frozen_cal_fpr'].items()}) for name in ('dunhuang_development','turufan_development')},
        default_choice_rank=list(selection_key(best)),test_used=False))


def test_report(root):
    if (root/'test_report.json').exists():raise ValueError('retained TEST already reported; no repeat selection')
    default=read(root/'default.json');setting=default['setting'];data=load_rows(root,'test',setting['matcher']);summary={};cases=[]
    for name,item in data.items():
        if any(t['fold']!=0 for t in item[1].values()):raise ValueError('non-TEST row in retained report')
        p='dunhuang_development' if name=='dunhuang_cv' else 'turufan_development'
        rows=settings_rows(item,tuple(setting['angles']),setting['head'])[setting['method']]
        baseline=settings_rows(item,(0,),setting['head'])['B'];summary[name]={}
        for label,rs,ts in [('default',rows,default['thresholds'][p]),('identity_baseline',baseline,default['baseline_thresholds'][p])]:
            summary[name][label]=dict(primary=summarize(rs,ts['primary']),frozen_cal_fpr={r:summarize(rs,t) for r,t in ts['frozen_cal_fpr'].items()},
                timing=timing(item,setting['angles'] if label=='default' else [0],setting['head']))
        for r,b in zip(rows,baseline):cases.append(dict(dataset=name,default=r,identity_baseline=b))
    save(root/'test_cases.json',cases)
    save(root/'test_report.json',dict(default=ref(root/'default.json'),populations=summary,cases=ref(root/'test_cases.json'),
        no_test_threshold_fit=True,no_test_setting_selection=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['development','test'],required=True);a=p.parse_args()
    (development if a.phase=='development' else test_report)(a.root)

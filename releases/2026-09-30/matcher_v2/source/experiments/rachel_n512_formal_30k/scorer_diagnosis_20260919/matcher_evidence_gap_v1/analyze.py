"""Independent population reconciliation and full-distribution comparisons."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from geometry_diagnostics import quant


def read(p):return json.loads(p.read_text())


def analyze(root):
    complete=read(root/'complete.json')
    assert complete['status']=='complete' and complete['pairs']==2303 and complete['full_population']
    rows={s:[read(p) for p in sorted((root/s).glob('*.json'))] for s in ('sim_select','dunhuang_cv')}
    assert len(rows['sim_select'])==1500 and len(rows['dunhuang_cv'])==803
    positives={s:[r for r in rr if r['usable_gt']] for s,rr in rows.items()}
    assert len(positives['sim_select'])==750 and len(positives['dunhuang_cv'])==292
    out={'scope':{'sim_select_positives':750,'dunhuang_valid_positives':292,'pairs':2303,
                  'matcher_forward':False,'scorer_forward':False,'production_changed':False},'distributions':{},'threshold_change':{},'recipe':{},'cross_strata':[]}
    for s,pp in positives.items():
        dist={}
        for name in ('length_10_px','length_20_px','length_40_px','length_64_px','length10_over40','length20_over40',
                     'model_spacing_mean_px','area_mean_px','potential_tokens_20_mean','segments_10_ge8','gap40_over10_share','gap64_over20_share'):
            dist[name]=quant([r['geometry'][name] for r in pp])
        for name in ('correct_n_zero_filled','correct_mass_zero_filled','correct_edges_per100px_L40','correct_endpoints_over_potential20'):
            dist[name]=quant([r['threshold16'][name] for r in pp])
        for stage,metric in [('raw_cloud','gt20_edges'),('native_correct_union','n')]:
            dist[stage+'_'+metric]=quant([r[stage][metric] for r in pp])
        for stage,metric in [('raw_cloud','gt20_edges'),('native_correct_union','n')]:
            dist[stage+'_per100px_L40']=quant([100*r[stage][metric]/r['geometry']['length_40_px'] for r in pp if r['geometry']['length_40_px']>0])
        out['distributions'][s]=dist
        for r in pp:
            assert r['threshold16']['correct_n_zero_filled']==(r['threshold16']['best_correct']['n'] if r['threshold16']['best_correct'] else 0)
        negatives=[r for r in rows[s] if not r['label']]
        x={}
        for t in ('threshold16','threshold20'):
            x[t]={k:sum(r[t][k] for r in pp) for k in ('top_correct','coverage_retained','coverage_prebudget','budget_loss','single_complete_correct_native','native_mixed20_40','native_mixed20_20')}
            x[t]['n_all']=quant([r[t]['correct_n_zero_filled'] for r in pp])
            x[t]['n_when_correct']=quant([r[t]['best_correct']['n'] for r in pp if r[t]['best_correct']])
            x[t]['negative_max_mass']=quant([r[t]['max_cluster_raw_q_arc_mass_px'] for r in negatives])
        x['coverage_gained']=[r['pair_id'] for r in pp if not r['threshold16']['coverage_retained'] and r['threshold20']['coverage_retained']]
        x['coverage_lost']=[r['pair_id'] for r in pp if r['threshold16']['coverage_retained'] and not r['threshold20']['coverage_retained']]
        x['top_pose_gained']=sum(not r['threshold16']['top_correct'] and r['threshold20']['top_correct'] for r in pp)
        x['top_pose_lost']=sum(r['threshold16']['top_correct'] and not r['threshold20']['top_correct'] for r in pp)
        both=[r for r in pp if r['threshold16']['best_correct'] and r['threshold20']['best_correct']]
        delta=[r['threshold20']['best_correct']['n']-r['threshold16']['best_correct']['n'] for r in both]
        x['paired_n_change_both_correct']=quant(delta)
        x['both_correct']=len(both);x['n_increased']=sum(v>0 for v in delta);x['n_decreased']=sum(v<0 for v in delta);x['n_unchanged']=sum(v==0 for v in delta)
        x['top_n_recovery_examples']=[dict(pair_id=r['pair_id'],index=r['index'],n16=r['threshold16']['correct_n_zero_filled'],n20=r['threshold20']['correct_n_zero_filled'],
            L40=r['geometry']['length_40_px']) for r in sorted(pp,key=lambda r:r['threshold20']['correct_n_zero_filled']-r['threshold16']['correct_n_zero_filled'],reverse=True)[:5]]
        out['threshold_change'][s]=x
        out['recipe'][s]={recipe:dict(n=len(rr),length40=quant([r['geometry']['length_40_px'] for r in rr]),
            N16=quant([r['threshold16']['correct_n_zero_filled'] for r in rr]),
            gap40_over10=quant([r['geometry']['gap40_over10_share'] for r in rr])) for recipe in sorted({r['recipe'] for r in pp})
            for rr in [[r for r in pp if r['recipe']==recipe]]}
    # No causal regression: common geometry/continuity bins are descriptive, with counts shown.
    for lo,hi in ((0,256),(256,512),(512,768),(768,1024),(1024,np.inf)):
        for cl,ch in ((0,.5),(.5,.8),(.8,1.00001)):
            bins={s:[r for r in pp if lo<=r['geometry']['length_40_px']<hi and r['geometry']['length10_over40'] is not None and cl<=r['geometry']['length10_over40']<ch]
                  for s,pp in positives.items()}
            out['cross_strata'].append(dict(length40=[lo,hi if np.isfinite(hi) else None],continuity=[cl,min(ch,1)],
                groups={s:dict(n=len(rr),N16=quant([r['threshold16']['correct_n_zero_filled'] for r in rr]),
                    raw_gt20=quant([r['raw_cloud']['gt20_edges'] for r in rr]),density16=quant([r['threshold16']['correct_edges_per100px_L40'] for r in rr]),
                    spacing=quant([r['geometry']['model_spacing_mean_px'] for r in rr])) for s,rr in bins.items()}))
    out['file_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for s in rows for p in sorted((root/s).glob('*.json'))}
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    result=analyze(root);(root/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('file_sha256','recipe','cross_strata')},ensure_ascii=False))

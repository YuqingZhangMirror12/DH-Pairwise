"""Distribution follow-up: pre-Scorer best-M correct cluster per covered pair.

All counts are sample frequencies, never normal-curve approximations. Uncovered
positives are reported separately rather than imputed as a measured zero cluster.
"""
import json
from pathlib import Path
import numpy as np
from .analyze import read, sha


def stats(values):
    a=np.asarray(values,dtype=float)
    if len(a)==0:return dict(n=0,minimum=None,p10=None,p25=None,median=None,p75=None,p90=None,p95=None,maximum=None)
    q=np.quantile(a,[0,.1,.25,.5,.75,.9,.95,1])
    return dict(n=len(a),**dict(zip(('minimum','p10','p25','median','p75','p90','p95','maximum'),map(float,q))))


def main():
    out=Path('artifacts/threshold_rootcause_20260927');sources={};observations=[];coverage=[]
    for split,name in (('sim_select','仿真 SELECT'),('dunhuang_cv','敦煌')):
        path=out/(split+'.json');sources[str(path)]=sha(path)
        rows=[r for r in read(path) if r['usable_gt'] and not r['gt_excluded']]
        kept=0
        for r in rows:
            cs=[c for c in r['clusters'] if c['proposal_gt_error']<=20]
            if not cs:continue
            c=max(cs,key=lambda x:x['mass']);kept+=1
            observations.append(dict(dataset=name,split=split,pair_id=r['pair_id'],cluster_id=c['cluster_id'],
                proposal_gt_error=c['proposal_gt_error'],**{k:c[k] for k in ('n','q_mean','mass','endpoints_a','endpoints_b','effective_edges')}))
        coverage.append(dict(dataset=name,total_positives=len(rows),covered=kept,missing=len(rows)-kept,
                             coveredShare=kept/len(rows)))
    samples={name:[r for r in observations if r['dataset']==name] for name in ('仿真 SELECT','敦煌')}
    measures={'n':'点对数量','q_mean':'单点平均Q','mass':'累计证据M'}
    bounds={'n':[0,64,128,192,256,320,384,448,512,576],
            'q_mean':[0,.002,.004,.006,.008,.010,.012,.014,.016,.018],
            'mass':[0,2,4,6,8,10,15,20,30,40]}
    quantiles=[];bins=[];cdf=[];strata=[]
    for name,rows in samples.items():
        for key,title in measures.items():
            values=np.array([r[key] for r in rows]);quantiles.append(dict(dataset=name,measure=key,title=title,**stats(values)))
            breaks=bounds[key]
            assert values.min()>breaks[0] and values.max()<=breaks[-1]
            for lo,hi in zip(breaks,breaks[1:]):
                # Equal-width N/Q bins; M uses explicit unequal ranges as categories.
                n=int(((values>lo)&(values<=hi)).sum())
                if key=='n':label=f'{int(lo)+1}–{int(hi)}'
                else:label=f'({lo:g}, {hi:g}]'
                bins.append(dict(dataset=name,measure=key,title=title,bin=label,lo=lo,hi=hi,count=n,
                                 sample_count=len(rows),sampleShare=n/len(rows)))
            for x in breaks:
                n=int((values<=x).sum())
                cdf.append(dict(dataset=name,measure=key,upper=x,count=n,sample_count=len(rows),cumulativeShare=n/len(rows)))
        for lo,hi in ((0,64),(64,128),(128,256),(256,512),(512,1024)):
            members=[r for r in rows if lo<r['n']<=hi]
            strata.append(dict(dataset=name,bin=f'{lo+1}–{hi}',**stats([r['q_mean'] for r in members])))
    for name,rows in samples.items():
        for key in measures:assert sum(r['count'] for r in bins if r['dataset']==name and r['measure']==key)==len(rows)
    a=np.array([r['n'] for r in samples['仿真 SELECT']]);b=np.array([r['n'] for r in samples['敦煌']])
    x=np.unique(np.r_[a,b]);gap=np.array([(b<=z).mean()-(a<=z).mean() for z in x])
    result=dict(schema='threshold-input-distributions/1',status='complete',coverage=coverage,
        quantiles=quantiles,bins=bins,cdf=cdf,q_within_count_strata=strata,observations=observations,
        count_distribution_comparison=dict(random_dun_less_than_random_sim=float((b[:,None]<a).mean()),
            ties=float((b[:,None]==a).mean()),empirical_cdf_dun_minus_sim_min=float(gap.min()),
            empirical_cdf_dun_minus_sim_max=float(gap.max()),
            note='Descriptive frequency over these exposed datasets, not independent sampling probability or causal effect.'),
        sources=sources,caveats=['Conditional on at least one correct pre-head proposal; missing positives separately counted.',
            'One largest-M correct cluster per pair, not an independent row per edge.',
            'Q strata control point-count bin only, not geometry, seam length, damage, or source; n=0 remains null.',
            'SIM SELECT1500 is not TEST3000. GT20 is pose correctness, not exact edge correspondence GT.'])
    (out/'distributions.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    assert all(sha(k)==v for k,v in sources.items())
    print(json.dumps({k:v for k,v in result.items() if k not in ('bins','cdf','observations','sources')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

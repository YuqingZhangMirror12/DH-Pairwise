"""Recompute report evidence from measured full distributions, never model scores."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
from .measure import read,save,EDGES


def aggregate(summary,kind,weight):
    edges=np.asarray([np.inf if x is None else x for x in summary['fine_edges']])
    values=np.asarray(summary['histograms'][kind][weight])
    return [float(values[(edges[:-1]>=lo)&(edges[:-1]<hi)].sum()) for lo,hi in zip(EDGES[:-1],EDGES[1:])]


def compile_report(root,new=None):
    root=Path(root);names=['Dunhuang292','Dunhuang295','S7','S7-C']+([new] if new else [])
    summaries={n:read(root/n/'summary.json') for n in names}
    records={n:[json.loads(x) for x in (root/n/'pairs.jsonl').read_text().splitlines()] for n in names}
    labels={'Dunhuang292':'敦煌292','Dunhuang295':'敦煌295（含错误GT）','S7':'旧S7','S7-C':'S7-C',new:'S7-D（试产）' if new and 'pilot' in new else 'S7-D'}
    sources=[dict(dataset=n,file=str((root/n/'summary.json').resolve()),
                  summary_sha256=hashlib.sha256((root/n/'summary.json').read_bytes()).hexdigest(),**summaries[n]['source']) for n in names]
    rows=[]
    for kind in ('seam40','seam64','opposed_unbounded','all'):
        for weight in ('pair_equal_share','point_weighted_share'):
            for n,s in summaries.items():
                values=aggregate(s,kind,weight)
                assert abs(sum(values)-1)<1e-8
                for i,v in enumerate(values):
                    lo,hi=EDGES[i:i+2]
                    rows.append(dict(dataset=labels[n],key=n,kind=kind,weight=weight,
                        bin=f'{lo:g}–{hi:g}' if np.isfinite(hi) else f'≥{lo:g}',bin_index=i,
                        gapShare=v,pairs=s['histograms'][kind]['measurable_pairs']))
    metrics=[];areas={};areas_hist=[];relative=[];gap_cdf=[]
    ref=summaries['Dunhuang292'];refgap=np.array(aggregate(ref,'seam40','pair_equal_share'))
    for n,s in summaries.items():
        pos=[r for r in records[n] if r['label']]
        area=np.array([r['area_px_'+side] for r in pos for side in 'ab']);areas[n]=area
        f=s['positive_fragments'];m=s['positive_metrics']
        gap=np.array(aggregate(s,'seam40','pair_equal_share'))
        metrics.append(dict(dataset=labels[n],key=n,positivePairs=len(pos),negativePairs=s['negative_pairs'],
            measurablePairs=s['histograms']['seam40']['measurable_pairs'],
            areaMean=float(area.mean()),areaMedian=float(np.median(area)),areaP10=float(np.quantile(area,.1)),areaP90=float(np.quantile(area,.9)),
            negativeAreaMean=s['negative_fragments']['area_px2']['mean'],
            stepMedian=f['step_mean_px']['p50'],stepP10=f['step_mean_px']['p10'],stepP90=f['step_mean_px']['p90'],
            tokensMedian=f['tokens']['p50'],patchRelativeMedian=f['window32_over_sqrt_area']['p50'],
            lengthMean=m['d20_length_px']['mean'],lengthMedian=m['d20_length_px']['p50'],
            gapTV=float(abs(gap-refgap).sum()/2),gapUnder2=gap[0],gap2to8=float(gap[1:3].sum()),
            gap8to15=float(gap[3]),breakMean=m['d10_breaks_mean']['mean'],
            sameAreaReference='mean of two endpoint occurrences per positive pair'))
        bins=[0,25000,50000,75000,100000,125000,150000,175000,200000,250000,300000,400000,np.inf]
        counts=np.histogram(area,bins)[0]
        for i,v in enumerate(counts/len(area)):
            lo,hi=bins[i:i+2]
            areas_hist.append(dict(dataset=labels[n],key=n,bin=f'{lo/1000:g}–{hi/1000:g}' if np.isfinite(hi) else '≥400',areaShare=float(v),index=i))
        for p in np.linspace(0,1,21):
            relative.append(dict(dataset=labels[n],key=n,percentile=int(p*100),areaPx2=float(np.quantile(area,p)),
                patch32OverSqrtArea=float(np.quantile([32/np.sqrt(x) for x in area],p))))
        cdf=np.cumsum(s['histograms']['seam40']['pair_equal_share'])
        for i in range(160):gap_cdf.append(dict(dataset=labels[n],key=n,gapPx=float(s['fine_edges'][i+1]),cumulativeShare=float(cdf[i])))
    for r in metrics:r['areaWassersteinPx2']=float(wasserstein_distance(areas[r['key']],areas['Dunhuang292']))
    extra=read(root/'patch_scale.json') if (root/'patch_scale.json').exists() else None
    patch=[]
    for w in (7,16,32,64):
        q=extra['summary']['window%d_raw_px'%w] if extra else {}
        patch.append(dict(window=w,tensor='16×16',inputMaskWidth=w,gridSpan=w-1,samplePitch=(w-1)/15,
            dunhuangRawMedian=q.get('p50'),dunhuangRawP10=q.get('p10'),dunhuangRawP90=q.get('p90'),
            simulationCanonicalPx=w))
    def query(rr,definition,ids):
        return dict(rows=rr,source=dict(label='已完成的GT轮廓分布统计',tables=[x['file'] for x in sources],
            filters=['仅正例几何；敦煌剔除3个错误GT；S7/S7-C全量12K正例'],
            metricDefinitions=[dict(label='统计口径',definition=definition,componentIds=ids)],
            provenance=sources))
    snapshot=dict(title='敦煌接缝与碎片尺度：分布对照和数据修订',surface='report',status='reviewed',
        generatedAt=datetime.now(timezone.utc).isoformat(),report=dict(asOf='2026-09-23'),filters=[],
        queries=dict(gaps=query(rows,'双向最近轮廓距离；pair_equal每对等权、两侧等权，point_weighted按轮廓弧长。seam40为几何近接带而非语义GT接缝。',['gap-bars','gap-cdf','gap-methods','whole-outline','findings']),
            metrics=query(metrics,'每个正例贡献两次碎片观测；长度沿相向且≤20px的几何近接带，断口为≤10px支持带中的缺口代理。中心间距为每片有效点的相邻直线距离均值再汇总；面积负例列单独按负例端点等权计算。',['comparison-table','sampling-table','class-area-table','findings','revision']),
            area=query(areas_hist,'每个正例的两个端点等权，原始模型输入mask前景像素数，非原始扫描图像像素数。',['area-chart']),
            quantiles=query(relative,'实测分位数；32px / sqrt(fragment area)，比较相对尺度而不是物理毫米。',['area-cdf']),
            patch=query(patch,'grid_sample四种输入像素窗口，统一16×16；Dunhuang按800/max(parent canvas)共享缩放。',['patch-table','patch-explanation']),
            cdf=query(gap_cdf,'近接带距离的pair-equal累积分布。',['gap-cdf'])),
        methodology=dict(original295=True,valid292=True,real_reference_no_longer_untouched_test=True,
            nominal_patch_scale_equal_not_physical_scale_evidence=True,newDataset=new,newTrainingStarted=False),
        patchScale=extra['summary'] if extra else None,showcases=[])
    snapshot['queries']['patch']['source']['tables'] += [str((root/'patch_scale.json').resolve()),
        str(Path('staging/pairwise_v0_2/models/rachel_n512.py').resolve()),
        str(Path('staging/pairwise_v0_2/baselines/rachel_n512_real_external.py').resolve())]
    save(root/'reviewed_snapshot.json',snapshot)
    save(root/'comparison.json',dict(metrics=metrics,source=sources,patch=patch))
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--new')
    a=p.parse_args();compile_report(a.root,a.new)

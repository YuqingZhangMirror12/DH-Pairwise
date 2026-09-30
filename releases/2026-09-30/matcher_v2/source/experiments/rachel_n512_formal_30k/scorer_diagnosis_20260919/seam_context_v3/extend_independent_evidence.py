"""Add completed F/I probes and explicitly revised GT cohort to a candidate app.

The original human-review cases, fragment assets and original metrics are kept.
No inference, calibration, or annotation mutation is performed here.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from .prepare import read, save, sha
from .build_report_snapshot import heat
from .extend_independent_report import APP_ID, ARMS
from .revised_real_cohort import revise

LABELS = dict(S7_M12_matched_C16='旧 S7 · 端点 C16', v3_B22='v3 · B22',
    S7_H='S7-H · 纯困难微调', **ARMS)
MAP_LABELS = {'q0':'Matcher · 第一次Sinkhorn Q0', 'q1':'Matcher · 第二次Sinkhorn Q1',
    'arc_context.blocks.0.cross_attn_0':'Matcher Context · 第1层',
    'arc_context.blocks.3.cross_attn_0':'Matcher Context · 第4层',
    'scorer_features.arc_context.blocks.0.cross_attn_0':'独立Scorer Context · 第1层',
    'scorer_features.arc_context.blocks.3.cross_attn_0':'独立Scorer Context · 第4层',
    'verifier.blocks.0.cross_attn_0':'Verifier · 第1层',
    'verifier.blocks.1.cross_attn_0':'Verifier · 第2层'}


def revised_metric_rows(result):
    if result['protocol']['thresholds_refit'] or result['protocol']['original_predictions_modified']:
        raise ValueError('only a recount with frozen thresholds is supported')
    rows = []
    for r in result['rows']:
        rows.append(dict(dataset=r['dataset'], model=LABELS[r['model_id']],
            model_id=r['model_id'], primary=True,
            policy='SIM冻结' if r['policy']=='SIM冻结' else '真实五折',
            threshold=r['threshold_median'], thresholds=r['threshold_values'],
            **{k:r[k] for k in ('accuracy','precision','recall','f1','auroc',
                'tp','fp','fn','tn','n','positive','negative')},
            layout=r.get('layout_correct_total'), joint=r.get('layout_correct_accepted')))
    return rows


def run(a):
    root, original, out = Path(a.root), Path(a.snapshot), Path(a.out)
    if out.exists() or out.resolve()==original.resolve():
        raise ValueError('write a new candidate; never overwrite current user report')
    snapshot=read(original)
    if snapshot['id']!=APP_ID or 'independent_cases' not in snapshot['queries']:
        raise ValueError('requires the reviewed F/I extension of the existing report')
    exclusions=read(a.exclusions)
    revised=revise(snapshot,exclusions)
    requested={(r['split'],r['pair_id']) for r in read(a.cases)['cases']}
    heatrows=[]
    files=[]
    for arm in ARMS:
        path=root/('probes_'+arm)
        status=read(path/'status.json')
        records=read(path/'records.json')
        protocols=[read(root/arm/split/'protocol.json') for split in ('dunhuang_cv','turufan')]
        if (status.get('status')!='complete' or status.get('count')!=len(requested)
                or status.get('model_provenance',{}).get('arm')!=arm
                or any(p['checkpoint_sha256']!=status['checkpoint_sha256'] for p in protocols)):
            raise ValueError('probe identity or completion gate failed')
        if len(records)!=len(requested) or {(r['split'],r['pair_id']) for r in records}!=requested:
            raise ValueError('incomplete preselected probe cohort')
        for r in records:
            if r['parity']['score_delta']>2e-5 or (r['parity']['translation_delta_px'] or 0)>.01:
                raise ValueError('probe prediction differs from frozen evaluation')
            maps={key:heat(r['heatmaps'][key]) for key in MAP_LABELS if key in r['heatmaps']}
            heatrows.append(dict(arm=arm,pair_id=r['pair_id'],split=r['split'],reason=r['reason'],
                maps=maps,map_labels={key:MAP_LABELS[key] for key in maps},pooling=r['pooling'],
                interventions=r['diagnostic_interventions'],proxies=r['proxies'],
                broad_edges=r['broad_edges'],raw_array_file=str(path/r['raw_array_file']),
                raw_array_sha256=r['raw_array_sha256'],checkpoint_sha256=status['checkpoint_sha256']))
        files.extend([str(path/'records.json'),str(path/'status.json')])
    q=snapshot['queries']
    q['gt_exclusions']=dict(rows=exclusions['records'],source=dict(label='用户确认错误GT的3例',
        files=[str(a.exclusions)],caveats=['仅应用于修订总体；原始预测、阈值和人工标签未改变。']))
    q['metrics_revised']=dict(rows=revised_metric_rows(revised),source=dict(
        label='剔除3例错误GT后的冻结阈值重计',files=[str(original),str(a.exclusions)],
        evidenceFlow=[dict(title='重计而非重新校准',detail='原始五折每对阈值保持不变，剔除3个用户指定敦煌正例。')],
        caveats=[revised['protocol']['caveat'],'修订后敦煌800对/292正例；Turufan602对/301正例。']))
    q['independent_heatmaps']=dict(rows=heatrows,source=dict(label='F/I固定36例冻结前向与逐层输出',files=files,
        caveats=['诊断分层样本不是总体；注意力不是因果归因。','统一色阶和索引区间最大值降采样；原始矩阵独立保存。',
            'F使用冻结Matcher特征，无独立Scorer Context；I单独展示两条特征路径。']))
    mechanism=read(root/'mechanism_independent.json')
    q['independent_mechanism']=dict(rows=mechanism['paired'],source=dict(label='同轮廓点的特征路径对照',
        files=[str(root/'mechanism_independent.json')],caveats=[mechanism['protocol']['pooling_caveat'],
            mechanism['protocol']['feature_caveat']]))
    snapshot['buildStatus']='updating'
    snapshot['generatedAt']=datetime.now(timezone.utc).isoformat()
    save(out,snapshot)
    save(out.with_suffix('.revised_metrics.json'),revised)
    print(dict(output=str(out),heatmaps=len(heatrows),original_cases=len(q['cases']['rows']),
        source_snapshot_sha256=sha(original)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','snapshot','out','exclusions','cases'):
        p.add_argument('--'+key,required=True)
    run(p.parse_args())

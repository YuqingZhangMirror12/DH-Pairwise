"""Append threshold diagnosis to the existing report; preserve every old query."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .analyze import DIAG, read, sha
from ..consensus_real_analysis_v1.bind_report import slim_matrix


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main(project, artifact, complete=False):
    project,artifact=Path(project),Path(artifact)
    file=project/'src/data.json';snapshot=read(file)
    annotation=Path('artifacts/human_layout_review_20260923/annotations.json')
    annotation_sha=sha(annotation)
    old={k:digest(v) for k,v in snapshot['queries'].items() if not k.startswith('td_')}
    summary=read(artifact/'summary.json');review=read(artifact/'reviewed_c10.json');replay=read(artifact/'head_replay.json')
    assert summary['status']=='complete' and replay['status']=='passed' and len(review['cases'])==11
    aliases={c['pair_id']:c['alias'] for c in read(DIAG/'s7_consensus_eval_v14/case_plan.json')['cases']}
    pair_rows=read(artifact/'pair_diagnosis.json')
    pair_by={(r['dataset'],r['pair_id']):r for r in pair_rows}
    replay_by={r['pair_id']:r for r in replay['cases']}
    legacy={r['pair_id']:r for r in snapshot['queries']['mergefix_cases']['rows']}
    probes=[]
    for case in review['cases']:
        assert case['audit']['status']=='passed' and case['semantics']['evidence_mode']=='exact_union_q'
        job='m12_'+case['provenance']['split'];row=pair_by[(job,case['pair_id'])]
        c=legacy[case['pair_id']]
        p=dict(pair_id=case['pair_id'],alias=aliases[case['pair_id']],dataset=c['dataset'],
            gt=c['gt'],fragment_a=c['fragment_a'],fragment_b=c['fragment_b'],
            label=c['label'],selected_cluster_id=case['selected_cluster_id'],
            score=case['score'],threshold=case['threshold'],error_px=case['posthoc_evaluation']['error_px'],
            q=slim_matrix(case['q']),clusters=[],replay=replay_by[case['pair_id']],
            source=case['source'],semantics=case['semantics'],audit=case['audit'])
        for cl,diag in zip(case['clusters'],row['candidates']):
            assert cl['cluster_id']==diag['cluster_id']
            p['clusters'].append(dict(diag,
                union_edge_count=len(cl['proposal'].get('original_union_edge_ids',cl['proposal']['edge_ids'])),
                merged_hypothesis_count=len(cl['proposal']['merged_hypothesis_ids']),
                heatmaps={k:slim_matrix(v) for k,v in cl['heatmaps'].items()},
                attention={stage:{key:dict({k:v for k,v in a.items() if k not in ('matrix','key_measure')},
                    matrix=slim_matrix(a['matrix'])) for key,a in block.items()} for stage,block in cl['attention'].items()},
                links=cl['links'],points=cl['points']['final']))
        probes.append(p)
    assert len(probes)==11
    model_results=read('artifacts/threshold_m12_analysis_20260927/frozen_real_comparison.json')
    metric_rows=[]
    for split,s in model_results['summary']['splits'].items():
        for model in ('S7_M12_matched_C16','S7_H','mergefix_m12','mergefix_scratch','threshold_m12'):
            for policy in ('sim_frozen','bounded_max_f1','fixed03'):
                values=s['models'][model]['policies'][policy]
                metric_rows.append(dict(dataset='敦煌' if split=='dunhuang_cv' else 'Turufan',model=model,
                    policy=policy,thresholds=[x['threshold'] for x in values['folds']] if 'folds' in values else [values['threshold']],
                    **values['corrected']))
    distributions=[];funnel=[];counterfactual=[]
    labels={'m12_sim_test_v14':'仿真 TEST','m12_dunhuang_cv':'敦煌','m12_turufan':'Turufan'}
    for job,j in summary['jobs'].items():
        for group,values in j['distributions'].items():
            for measure,stats in values.items():
                distributions.append(dict(dataset=labels[job],group=group,measure=measure,**stats))
        for stage,count in j['funnel'].items():
            funnel.append(dict(dataset=labels[job],stage=stage,count=count,denominator=j['positives'],failureShare=count/j['positives']))
        for mode,counts in j['counterfactual_readouts'].items():
            counterfactual.append(dict(dataset=labels[job],mode=mode,**counts))
    real_cases=[]
    for r in pair_rows:
        if r['dataset']=='m12_sim_test_v14':continue
        c=legacy[r['pair_id']]
        real_cases.append(dict(pair_id=r['pair_id'],case_name=c['case_name'],dataset=c['dataset'],label=r['label'],
            fragment_a=c['fragment_a'],fragment_b=c['fragment_b'],gt=c['gt'],
            score=r['score'],translation=r['translation'],error_px=r['error_px'],layout20=r['layout20'] if r['gt_known'] else None,
            accepted=r['accepted'],candidate_coverage=r['candidate_coverage'] if r['gt_known'] else None,
            candidates=r['candidates']))
    layers=read(artifact/'layer_weights.json')['layers']
    provenance=dict(label='阈值M12 E8冻结预测、真实NPZ与权重；本地解析复算/有限头部干预',
        files=[str(artifact/f) for f in ('summary.json','layer_weights.json','head_replay.json','reviewed_c10.json')],
        evidenceFlow=[dict(title='全量分解',detail='3000仿真TEST、敦煌修订800、Turufan602；全部保留候选精确复算分数，原预测不变。'),
                      dict(title='有限干预',detail='固定11例原赢家及最终位姿；CPU只重放Scorer，基线核对后置零指定层/特征；0优化器更新。'),
                      dict(title='实际图像',detail='NPZ来自11个预先指定案例。所有候选可选；图中最多128条最高贡献连线，仅限制显示，不限制模型输入。')],
        caveats=['Turufan没有布局GT。','真实数据已参与设计，不是全新盲测。',
                 '解析上限不保证这些点对真的正确；特征置零可能离开训练分布，不能直接证明域差原因。',
                 '质量长度为Q×可观测弧长，不是实际完整接缝长度。'])
    queries={
        'td_summary':[summary], 'td_metrics':metric_rows, 'td_distributions':distributions,
        'td_funnel':funnel,'td_counterfactual':counterfactual,'td_cases':real_cases,
        'td_c10':probes,'td_layers':layers,
        'td_scales':[dict(quantity=k,**v) for k,v in review['scales'].items()],
    }
    for key,rows in queries.items():
        snapshot['queries'][key]=dict(rows=rows,source=dict(provenance,metricDefinitions=[
            dict(label='范围与口径',definition=('Pair F1不要求布局正确；布局误差≤20px。统计为实际保存结果；解析反事实单列，不是新训练模型。'
                 if key!='td_c10' else '阈值版评分W=Q×精确去重并集；Q不乘方向核。支持/冲突图是质量长度贡献；Attention是实际执行四头均值。'),
                 componentIds=['td-summary','td-metrics','td-funnel','td-counterfactual','td-cases','td-c10','td-layer-weights','td-q','td-w','td-attention','td-case-score'])]))
    snapshot['buildStatus']='complete' if complete else 'updating'
    snapshot['generatedAt']=datetime.now(timezone.utc).isoformat()
    assert all(digest(snapshot['queries'][k])==v for k,v in old.items())
    file.write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    assert sha(annotation)==annotation_sha
    receipt=dict(status='passed',old_queries_unchanged=list(old),annotation_sha256=annotation_sha,
        c10_cases=len(probes),real_cases=len(real_cases),snapshot_sha256=sha(file),new_queries=list(queries))
    (artifact/'report_binding.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--complete',action='store_true')
    args=parser.parse_args();main('reports/seam_v3_real_analysis_20260923','artifacts/threshold_diagnosis_20260927',args.complete)

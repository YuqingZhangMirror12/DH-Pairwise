"""Append reviewed mergefix data without changing historical rows or annotations."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from .analyze import DIAG, read, rows, sha, require

LABELS = {'S7_M12_matched_C16':'旧 S7', 'S7_H':'S7-H 困难微调', 'v3_B22':'连续接缝 v3',
          'frozen_features':'F 冻结特征', 'independent_features':'I 独立特征',
          'mergefix_m12':'归并修复 · M12', 'mergefix_scratch':'归并修复 · scratch'}
POLICY_LABELS = {'sim_frozen':'仿真冻结', 'fixed03':'固定0.30',
                 'bounded_max_f1':'真实五折F1', 'bounded_recall95':'真实五折Recall95目标'}


def slim_matrix(m):
    result = {k:v for k,v in m.items() if k != 'rows'}
    if not m['rows']:
        return dict(result, means=[], a_ranges=[], b_ranges=[])
    na = max(r['a_bin'] for r in m['rows'])+1
    nb = max(r['b_bin'] for r in m['rows'])+1
    means = [[None]*nb for _ in range(na)]
    ar, br = [None]*na, [None]*nb
    for r in m['rows']:
        means[r['a_bin']][r['b_bin']] = r['mean']
        ar[r['a_bin']] = [r['a_start'], r['a_end_exclusive']]
        br[r['b_bin']] = [r['b_start'], r['b_end_exclusive']]
    require(all(v is not None for row in means for v in row), 'missing matrix bin')
    return dict(result, means=means, a_ranges=ar, b_ranges=br)


def c10_rows(path, case_plan):
    data = read(path)
    require(len(data['cases']) == 22 and not data['model_inference_performed'], 'wrong C10 export')
    aliases = {c['pair_id']:c['alias'] for c in case_plan['cases']}
    out = []
    for c in data['cases']:
        require(c['audit']['status'] == 'passed', 'C10 audit failed')
        keep = ('pair_id','selected_cluster_id','has_candidate','numeric_valid','score','threshold',
                'accepted','translation_a_to_b_rc','posthoc_evaluation','source','semantics','audit')
        r = {k:c[k] for k in keep}
        r.update(arm=c['provenance']['arm'], alias=aliases[c['pair_id']],
                 checkpoint_sha256=c['provenance']['checkpoint_sha256'], q=slim_matrix(c['q']),
                 seed_count=len(c['trace']['seeds']), hypothesis_count=len(c['trace']['hypotheses']),
                 cluster_count=len(c['clusters']), candidates=[], winner=None)
        errors=c['posthoc_evaluation']['candidate_errors_px']
        for j, cl in enumerate(c['clusters']):
            proposal=cl['proposal']
            scalar_readout={k:v for k,v in cl['readout'].items() if not isinstance(v,(list,dict))}
            simple=dict(cluster_id=cl['cluster_id'], selected=cl['selected'],
                initial_translation_rc=cl['initial_translation_rc'],
                refined_translation_rc=cl['refined_translation_rc'],
                error_px=errors[j] if j<len(errors) else None,
                seed_count=len(proposal['initial_seed_ids']),
                merged_hypothesis_count=len(proposal['merged_hypothesis_ids']),
                sparse_edge_count=len(proposal['edge_ids']),
                union_edge_count=len(proposal.get('original_union_edge_ids', proposal['edge_ids'])),
                added_correspondence_count=cl['added_correspondence_count'],
                added_support_contribution_px=cl['added_support_contribution_px'],
                underconstrained=cl['underconstrained'], **scalar_readout)
            r['candidates'].append(simple)
            if cl['selected']:
                require(r['winner'] is None, 'multiple selected clusters')
                attention={stage:{key:dict({k:v for k,v in a.items() if k not in ('matrix','key_measure')},
                    matrix=slim_matrix(a['matrix'])) for key,a in block.items()}
                    for stage,block in cl['attention'].items()}
                r['winner']=dict(simple, heatmaps={k:slim_matrix(v) for k,v in cl['heatmaps'].items()},
                    attention=attention, final_links=cl['links']['final_support'],
                    points=cl['points']['final'], overlap=cl['overlap'])
        out.append(r)
    return out, data['scales']


def main(args):
    app=Path(args.project); artifact=Path(args.artifact); artifact.mkdir(parents=True,exist_ok=True)
    p=app/'src/data.json'; snapshot=read(p)
    backup=artifact/'report_before_mergefix.json'
    if not backup.exists(): shutil.copy2(p,backup)
    originals={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
               for k,v in snapshot['queries'].items() if not k.startswith('mergefix_')}
    annotations=Path(args.annotations); annotation_sha=sha(annotations)
    comparison_path=artifact/'frozen_real_comparison.json'
    comparison=read(comparison_path); summary=comparison['summary']
    require(summary['status']=='complete' and summary['all_source_hashes_unchanged'], 'unverified metrics')
    root=Path(args.evaluation); plan=read(DIAG/'s7_consensus_eval_v14/case_plan.json')
    metric_rows=[]
    for split,s in summary['splits'].items():
        for model,m in s['models'].items():
            for policy,values in m['policies'].items():
                ts=[f['threshold'] for f in values['folds']] if 'folds' in values else [values['threshold']]
                for cohort in ('original','corrected'):
                    metric_rows.append(dict(dataset='敦煌' if split=='dunhuang_cv' else 'Turufan',
                        model=model, model_label=LABELS[model], policy=policy,
                        policy_label=POLICY_LABELS[policy], cohort=cohort,thresholds=ts,
                        **values[cohort]))
    legacy=copy.deepcopy(snapshot['queries']['independent_cases']['rows'])
    case_lookup={c['pair_id']:c for c in legacy}
    excluded=set(plan['user_confirmed_gt_exclusions'])
    for c in legacy:
        c['gt_excluded']=c['pair_id'] in excluded
        for key,m in c['models'].items():
            m['thresholds']={'sim_frozen':m['threshold_sim'], 'fixed03':.3,
                             'bounded_max_f1':m['threshold_cv']}
    for arm in ('m12','scratch'):
        for split in ('dunhuang_cv','turufan'):
            key='mergefix_'+arm
            protocol=read(root/(arm+'_'+split)/'protocol.json')
            oof={p:{r['pair_id']:r['threshold'] for r in rr}
                 for p,rr in comparison['out_of_fold'][split][key].items()}
            for r in rows(root/(arm+'_'+split)/'case_diagnostics.jsonl'):
                c=case_lookup[r['pair_id']]
                require(bool(c['label'])==bool(r['label']) and c['fold']==r['fold'],'case identity mismatch')
                c['models'][key]=dict(score=r['score'],translation=r['translation'],
                    decision_valid=bool(r['has_candidate'] and r['numeric_valid']),
                    layout_error_px=r['error_px'],layout20=bool(r['layout20']) if r['gt_known'] else None,
                    thresholds=dict(sim_frozen=protocol['threshold'],fixed03=.3,
                                    **{p:v[r['pair_id']] for p,v in oof.items()}),
                    candidate_coverage=r['candidate_coverage'] if r['gt_known'] else None)
    require(len(case_lookup)==1405,'case cohort changed')
    c10,scales=c10_rows(artifact/'reviewed_c10_cases.json', plan)
    now=datetime.now(timezone.utc).isoformat()
    files=[str(comparison_path),str(artifact/'reviewed_c10_cases.json'),str(root/'evaluation_complete.json')]
    source=dict(label='冻结预测＋原来源五折，非重新推理',files=files,
        executedAt=now,evidenceFlow=[
            dict(title='冻结评价',detail='六组TEST/敦煌/Turufan评价已完成；预测、权重不变；22例按事先案例清单导出。'),
            dict(title='独立复算',detail='逐例混淆计数、GT20及Joint计数复核；复用既定0.20–0.80来源五折；排除3例错误GT仅重计，不重拟合。'),
            dict(title='显示压缩',detail='C10只在HTML展示赢家热图；保留所有候选摘要。16×16均值包括全部原矩阵条目，不重新归一化。完整NPZ与所有候选导出仍保留。')],
        caveats=['Turufan无GT位移，Layout/Joint必须为空。','单种子、不同历史训练曝光，非单因素严格消融。',
                 '真实数据曾用于设计诊断，不是全新盲测。','Attention不等于因果解释；质量长度不等于GT物理接缝长度。'])
    queries={
        'mergefix_metrics':dict(rows=metric_rows, source=dict(source,metricDefinitions=[
            dict(label='Pair F1',definition='2TP/(2TP+FP+FN)，判断是否可拼，不要求Layout正确。',componentIds=['mergefix-metrics','mergefix-summary']),
            dict(label='Joint F1',definition='真阳性必须接受且GT位移误差≤20px；正例错位接受也记FP；负例接受记FP。',componentIds=['mergefix-metrics','mergefix-summary']),
            dict(label='Layout',definition='不看分类接受，所有正例赢家位移误差≤20px的数量；Turufan未知。',componentIds=['mergefix-metrics','mergefix-summary'])])),
        'mergefix_cases':dict(rows=legacy,source=dict(source,metricDefinitions=[
            dict(label='摆放坐标',definition='保存位移为A→B的[row,col]；展示B移动为负位移。未因显示重新拟合。',componentIds=['mergefix-cases'])])),
        'mergefix_c10':dict(rows=c10,source=dict(source,metricDefinitions=[
            dict(label='Q与W',definition='该mergefix版实际评分W=Q×方向性几何核；不是阈值版的Q×并集。矩阵为全部格均值；百分数色条仅改变显示单位，不重新归一。',componentIds=['mergefix-q','mergefix-initial','mergefix-final','mergefix-candidates','mergefix-c10-note','mergefix-case-reading']),
            dict(label='Scorer Attention',definition='真实执行的4头Attention的均值；显示分箱后不重新按行归一。与Matcher Q不等价。',componentIds=['mergefix-attention'])])),
        'mergefix_scales':dict(rows=[dict(quantity=k,**v) for k,v in scales.items()],source=source),
        'mergefix_audit':dict(rows=summary['download_verification'],source=source),
    }
    snapshot['queries'].update(queries)
    if args.complete:
        checked=read(artifact/'report_checks.json')
        require(checked.get('status')=='passed' and not checked.get('errors'), 'browser checks missing')
        require(checked.get('annotationSha256')==annotation_sha, 'annotation audit changed')
    snapshot['buildStatus']='complete' if args.complete else 'updating'
    snapshot['generatedAt']=now
    snapshot['report']['asOf']='2026-09-27'
    for k,v in originals.items():
        require(hashlib.sha256(json.dumps(snapshot['queries'][k],sort_keys=True).encode()).hexdigest()==v,
                'historical query changed: '+k)
    p.write_text(json.dumps(snapshot,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    require(sha(annotations)==annotation_sha,'human annotations changed')
    receipt=dict(status='passed',new_queries=list(queries),cases=1405,c10_cases=len(c10),
        historical_queries_unchanged=list(originals),annotations_sha256=annotation_sha,
        snapshot_sha256=sha(p),comparison_sha256=sha(comparison_path),schema='mergefix-report-bind/1')
    (artifact/'report_binding.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--project',required=True)
    p.add_argument('--artifact',required=True);p.add_argument('--evaluation',required=True)
    p.add_argument('--annotations',required=True); p.add_argument('--complete',action='store_true'); main(p.parse_args())

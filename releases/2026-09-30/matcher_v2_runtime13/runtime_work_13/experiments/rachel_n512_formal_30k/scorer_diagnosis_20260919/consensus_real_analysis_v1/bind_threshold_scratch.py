"""Append the completed E32 threshold arm to the established review report.

No neural forward passes; no epoch/geometry selection. Preserves every existing
query and the external human annotation store. All candidate detail is retained
for the eleven prespecified cases, not retrospectively selected examples.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

from .analyze import DIAG, read, rows, sha, require, close
from .bind_report import LABELS as OLD_LABELS, slim_matrix

MODEL = 'threshold_scratch_fixed'
CHECKPOINT = '20d9295f9b66bbc04c2ef93cda3f20c0774694269150707b6fba3e7780a4441e'
LABELS = dict(OLD_LABELS, threshold_m12='固定16px · M12',
              threshold_scratch_fixed='固定16px · E32')
PREFIX = 'threshold_e32_'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checked_cluster(cluster, audit, error):
    require(audit['cluster_id'] == cluster['cluster_id']
            and audit['evidence_mode'] == 'exact_union_q'
            and audit['directional_mass_used_for_scorer'] is False, 'wrong evidence mode')
    proposal = cluster['proposal']
    union = proposal['original_union_edge_ids']
    require(len(union) == len(set(map(tuple, union))), 'duplicate union point pair')
    require(proposal['pose_diameter_px'] == 16
            and proposal['actual_diameter_px'] <= 16.00001, 'wrong pose diameter')
    for key in ('initial_recalled', 'final_recalled'):
        # The reviewed display sum uses float64; the captured audit reduces the
        # same entries in FP32. This is not a score/geometry comparison tolerance.
        require(math.isclose(cluster['heatmaps'][key]['total'], audit['union_absolute_q_mass'],
                             rel_tol=1e-6, abs_tol=1e-7), 'absolute union Q sum differs')
    require(cluster['heatmaps']['initial_recalled'] == cluster['heatmaps']['final_recalled'],
            'pose-invariant exact union Q changed across the two passes')
    for link in cluster['links'].values():
        require(link['used_to_limit_model_input'] is False
                and link['displayed_edges'] == len(link['rows'])
                and link['displayed_edges'] + link['omitted_edges'] == link['eligible_edges'],
                'display limit confused with model input')
    scalar = {k: v for k, v in cluster['readout'].items() if isinstance(v, (int, float, bool))}
    attention = {stage: {key: dict(
        {k: v for k, v in att.items() if k not in ('matrix', 'key_measure')},
        matrix=slim_matrix(att['matrix'])) for key, att in block.items()}
        for stage, block in cluster['attention'].items()}
    return dict(cluster_id=cluster['cluster_id'], selected=cluster['selected'],
        initial_translation_rc=cluster['initial_translation_rc'],
        refined_translation_rc=cluster['refined_translation_rc'],
        error_px=error, union_edge_count=len(union),
        actual_diameter_px=proposal['actual_diameter_px'],
        local_hypotheses=len(proposal['merged_hypothesis_ids']),
        union_absolute_q_mass=audit['union_absolute_q_mass'],
        underconstrained=cluster['underconstrained'], overlap=cluster['overlap'],
        heatmaps={k: slim_matrix(v) for k, v in cluster['heatmaps'].items()},
        attention=attention, final_links=cluster['links']['final_support'], **scalar)


def c10_rows(review, plan, historical):
    require(len(review['cases']) == 11 and not review['model_inference_performed'], 'wrong C10 export')
    plan_ids = {r['pair_id']: r for r in plan['cases']}
    require(set(plan_ids) == {r['pair_id'] for r in review['cases']}, 'changed prespecified cases')
    result = []
    for case in review['cases']:
        p = case['provenance']
        require(p['arm'] == 'scratch_fixed' and p['variant'] == 'threshold'
                and p['checkpoint_sha256'] == CHECKPOINT and p['selected_epoch'] == 22
                and p['threshold'] == .32 and p['evidence_mode'] == 'exact_union_q', 'wrong frozen model')
        require(case['audit']['status'] == 'passed', 'failed C10 audit')
        old = historical[case['pair_id']]
        require(bool(old['label']) == bool(case['posthoc_evaluation']['label']), 'label differs')
        errors = case['posthoc_evaluation']['candidate_errors_px']
        require(len(errors) == len(case['clusters']), 'candidate error alignment')
        audits = {r['cluster_id']: r for r in case['audit']['clusters']}
        clusters = [checked_cluster(cl, audits[cl['cluster_id']], errors[i])
                    for i, cl in enumerate(case['clusters'])]
        winners = [cl for cl in clusters if cl['selected']]
        require(len(winners) == int(case['has_candidate']), 'winner count')
        if winners:
            require(winners[0]['cluster_id'] == case['selected_cluster_id'], 'winner identity')
            close(winners[0]['score'], case['score'], 'winner score')
        result.append(dict(pair_id=case['pair_id'], alias=plan_ids[case['pair_id']]['alias'],
            dataset=old['dataset'], case_name=old['case_name'], fragment_a=old['fragment_a'],
            fragment_b=old['fragment_b'], label=old['label'], gt=old['gt'],
            comparisons={k: copy.deepcopy(old['models'][k]) for k in
                         ('S7_M12_matched_C16', 'mergefix_scratch')},
            checkpoint_sha256=CHECKPOINT, epoch=22, threshold=.32,
            selected_cluster_id=case['selected_cluster_id'], score=case['score'],
            accepted=case['accepted'], posthoc_evaluation=case['posthoc_evaluation'],
            q=slim_matrix(case['q']), clusters=clusters,
            source=case['source'], audit=case['audit']))
    return sorted(result, key=lambda r: list(plan_ids).index(r['pair_id']))


def main(args):
    out = Path(args.receipt)
    require(not out.exists(), 'preserve prior binding receipt')
    source_file = Path(args.project) / 'src/data.json'
    snapshot = read(source_file)
    require(snapshot['id'] == 'report:4b51ee5a-d8bb-4663-8305-69763a99c639', 'wrong report')
    require(not any(k.startswith(PREFIX) for k in snapshot['queries']), 'already imported')
    original = {k: digest(v) for k, v in snapshot['queries'].items()}
    annotation_sha = sha(args.annotations)
    inputs = {str(Path(p).resolve()): sha(p) for p in
              (args.analysis, args.review, DIAG / 's7_consensus_eval_v14/case_plan.json')}
    data = read(args.analysis)
    require(data['summary']['status'] == 'complete'
            and data['summary']['all_source_hashes_unchanged'], 'incomplete analysis')
    extension = data['summary']['extensions'][-1]
    require(extension['arm'] == MODEL and extension['original_results_preserved']
            and extension['fixed_cases_verified'] == 11, 'wrong analysis extension')
    for p, value in extension['input_sha256'].items():
        require(sha(p) == value, 'analysis evidence changed: ' + p)
    metrics = []
    for split, group in data['summary']['splits'].items():
        require(set(group['models']) == set(LABELS), 'missing nine-model comparison')
        for key, model in group['models'].items():
            for policy, p in model['policies'].items():
                ts = [x['threshold'] for x in p['folds']] if 'folds' in p else [p['threshold']]
                for cohort in ('original', 'corrected'):
                    metrics.append(dict(model=key, model_label=LABELS[key], split=split,
                        dataset='敦煌' if split == 'dunhuang_cv' else 'Turufan',
                        policy=policy, cohort=cohort, thresholds=ts, **p[cohort]))
    plan = read(DIAG / 's7_consensus_eval_v14/case_plan.json')
    review = read(args.review)
    historical = {r['pair_id']: r for r in snapshot['queries']['mergefix_cases']['rows']}
    cases = c10_rows(review, plan, historical)
    now = datetime.now(timezone.utc).isoformat()
    source = dict(label='冻结E32阈值版预测与11例实际网络输出；本次仅CPU复算/显示',
        files=list(inputs), inputSha256=inputs, executedAt=now,
        evidenceFlow=[
            dict(title='身份与终态', detail='28轮平台停止，仿真选择头E22/阈值0.32；E32 Matcher冻结。三个评价成功，11个事先固定案例数值审计通过。'),
            dict(title='统计口径', detail='沿用来源五折与0.20–0.80分类阈值网格；原803对校准后剔除3例错GT。冻结0.32、固定0.30、真实五折分开。'),
            dict(title='显示压缩', detail='Q与权重16×16全元素均值，无重新归一；保留11例全部候选。实际支持连线仅显示权重最大的128条，不截断网络输入。')],
        caveats=['真实数据曾参与设计诊断，是开发性评估而非新盲测。',
                 '原始点对并集、Q×方向核定位、学习支持和Attention不是同一数量。',
                 'Attention不证明因果；11例不代表总体发生率。',
                 'Turufan无GT位移；Layout与Joint为空。',
                 '单种子且旧模型训练历史不同，不能宣称单因素因果或统计显著。'],
        metricDefinitions=[
            dict(label='Pair F1', definition='2TP/(2TP+FP+FN)，只评价可拼分类。', componentIds=['threshold-e32-metrics']),
            dict(label='Layout≤20px', definition='不看接受分数，正例赢家位移误差≤20px的数量。', componentIds=['threshold-e32-metrics']),
            dict(label='Joint F1', definition='接受且GT20正确才计TP；负例接受和正例错位接受均计FP。', componentIds=['threshold-e32-metrics']),
            dict(label='并集Q', definition='每个精确(i,j)保留一次原始Q，和不归一成1，不重复Sinkhorn。', componentIds=['threshold-e32-candidates','threshold-e32-union'])])
    new = {PREFIX + 'metrics': dict(rows=metrics, source=source),
           PREFIX + 'c10': dict(rows=cases, source=source),
           PREFIX + 'scales': dict(rows=[dict(quantity=k, **v) for k, v in review['scales'].items()], source=source)}
    out.parent.mkdir(parents=True, exist_ok=True)
    backup = out.parent / 'report_before_threshold_e32.json'
    require(not backup.exists(), 'preserve report backup')
    shutil.copy2(source_file, backup)
    snapshot['queries'].update(new)
    snapshot['buildStatus'] = 'updating'
    snapshot['generatedAt'] = now
    snapshot['report']['asOf'] = '2026-09-28'
    tmp = source_file.with_suffix('.json.e32.tmp')
    require(not tmp.exists(), 'unfinished report write')
    with tmp.open('x') as f:
        json.dump(snapshot, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        f.write('\n')
    require(sha(source_file) == sha(backup), 'report changed while binding')
    require(sha(args.annotations) == annotation_sha and all(sha(p) == h for p, h in inputs.items()),
            'annotations or frozen evidence changed')
    tmp.replace(source_file)
    written = read(source_file)
    require(all(digest(written['queries'][k]) == h for k, h in original.items()), 'historical query changed')
    receipt = dict(status='passed', report_id=written['id'], new_queries=list(new),
        metrics=len(metrics), cases=len(cases), clusters=sum(len(c['clusters']) for c in cases),
        input_sha256=inputs, source_sha256=sha(__file__), annotations_sha256=annotation_sha,
        historical_queries_unchanged=list(original), snapshot_sha256=sha(source_file),
        neural_forward_passes=0, epoch_or_geometry_selection=False)
    with out.open('x') as f:
        json.dump(receipt, f, indent=2); f.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('project', 'analysis', 'review', 'annotations', 'receipt'):
        p.add_argument('--' + field, required=True)
    main(p.parse_args())

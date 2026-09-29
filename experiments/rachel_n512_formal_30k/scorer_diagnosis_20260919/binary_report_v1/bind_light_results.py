"""Prepare source-bound report queries from completed frozen imports only.

This is an offline display adapter, not an evaluation or training launcher.
No result is installed until a real six-job import exists. The caller builds
the existing report after binding and performs a bounded rendered check.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .assemble import EXPERIMENTS, REAL_PLAN_SHA, tasks
from .baselines import digest_object
from .bind_baselines import LABELS as BASELINE_LABELS
from .export import EXPECTED_CASES, read, require, sha


METRICS_QUERY = 'light_scorer_frozen_results'
CASES_QUERY = 'light_scorer_fixed_cases'
REPORT_ID = 'report:4b51ee5a-d8bb-4663-8305-69763a99c639'
LABELS = dict(BASELINE_LABELS, **{k: v['title'] for k, v in EXPERIMENTS.items()})


def query_rows(bundle):
    require(bundle.get('schema') == 'binary-report-all-models/1'
            and bundle.get('status') == 'evidence_ready_not_final_conclusions'
            and bundle.get('neural_inference_repeated') is False
            and bundle.get('thresholds_refitted') is False,
            'completed frozen comparison import required')
    available = bundle['available_light_experiments']
    require(available and len(available) == len(set(available))
            and set(available) <= set(EXPERIMENTS), 'no completed light results to bind')
    metrics, keys = [], set()
    for original in bundle['rows']:
        if original['population'] not in ('real_test', 'all'):
            continue  # Full-development / CAL / SELECT never get pooled with TEST.
        row = copy.deepcopy(original)
        model, choice, split, policy = [row[k] for k in ('model', 'selection_kind', 'split', 'policy')]
        require(model in LABELS and choice in ('sim', 'real')
                and policy in ('primary', 'fixed03'), 'unregistered comparison scope')
        if model in EXPERIMENTS:
            require(model in available and (choice, split) in tasks(model), 'light dataset scope differs')
        else:
            require(choice == 'sim' and split in ('sim_test_v14', 'dunhuang_cv', 'turufan'),
                    'do not invent a REAL-selected historical baseline')
        expected = (3000, 1500, 1500) if split.startswith('sim_test_') else (
            (161, 59, 102) if split == 'dunhuang_cv' else (122, 61, 61))
        require(row['population'] == ('all' if split.startswith('sim_test_') else 'real_test'),
                'population and dataset differ')
        m = row['metrics']
        require(tuple(m[k] for k in ('pairs', 'positives', 'negatives')) == expected,
                'registered TEST class counts differ')
        if not split.startswith('sim_test_'):
            require(row['real_plan_sha256'] == REAL_PLAN_SHA, 'REAL role binding differs')
        if split == 'turufan':
            require(m['layout20_count'] is None and m['joint_f1'] is None, 'Turufan has no Layout GT')
        key = (model, choice, split, policy)
        require(key not in keys, 'duplicate comparison row'); keys.add(key)
        metrics.append(dict(row, model_label=LABELS[model], **m))
    for model in available:
        require({(c, s, p) for m, c, s, p in keys if m == model}
                == {(c, s, p) for c, s in tasks(model) for p in ('primary', 'fixed03')},
                'each imported experiment requires six jobs and both score policies')

    cases, groups, case_sets = [], set(), {}
    lookup = {(r['model'], r['selection_kind'], r['split']): r
              for r in metrics if r['policy'] == 'primary'}
    for group in bundle['binary_case_groups']:
        key = (group['experiment'], group['selection_kind'], group['split'])
        require(key not in groups and key in lookup and key[0] in available,
                'case group differs from completed metrics'); groups.add(key)
        model, choice, split = key
        row = lookup[key]
        ids = {c['pair_id'] for c in group['cases']}
        require(len(ids) == len(group['cases']) == EXPECTED_CASES[split], 'fixed case set incomplete or duplicated')
        # Every model/checkpoint displays the same predeclared cases, not selected successes.
        require(split not in case_sets or case_sets[split] == ids, 'fixed cases changed across models')
        case_sets[split] = ids
        for case in group['cases']:
            require(case.get('schema') == 'binary-report-case/1'
                    and case['variant'] == EXPERIMENTS[model]['variant'], 'case head type differs')
            identity = case['provenance']
            require(all(identity.get(k) == v for k, v in dict(selection_kind=choice, split=split,
                        selected_epoch=row['selected_epoch'], checkpoint_sha256=row['checkpoint_sha256']).items())
                    and case['threshold'] == row['threshold'], 'case checkpoint/threshold differs')
            require(identity.get('execution_continuation') == row.get('execution_continuation'),
                    'case execution continuation differs from result row')
            require(case['verification']['imported_numeric_audit_status'] == 'passed'
                    and case['verification']['this_export_repeated_model_inference'] is False,
                    'case numeric audit missing')
            require(all(case['semantics'][k] is False for k in (
                'attention_present', 'local_conflict_classifier', 'learned_refinement',
                'pooling_weights_are_attention', 'layer_values_are_causal_importance')),
                'light head semantics differ')
            cases.append(dict(case_key='|'.join((*key, str(case['pair_id']))), model=model,
                model_label=LABELS[model], selection_kind=choice, split=split, pair_id=case['pair_id'],
                selected_epoch=row['selected_epoch'], checkpoint_sha256=row['checkpoint_sha256'],
                threshold=case['threshold'], score=case['score'], has_candidate=case['has_candidate'],
                record_json=json.dumps(case, ensure_ascii=False, separators=(',', ':'), allow_nan=False)))
    require(groups == {(m, c, s) for m in available for c, s in tasks(m)},
            'all six fixed-case groups required, including empty SIM case inventories')
    return metrics, cases


def bind_snapshot(snapshot, bundle, evidence_path, evidence_hash):
    require(snapshot['id'] == REPORT_ID, 'only update the established real-data report')
    metrics, cases = query_rows(bundle)
    result = copy.deepcopy(snapshot)
    now = datetime.now(timezone.utc).isoformat()
    definitions = [
        dict(label='分类准确率', definition='正确接受的正例TP与正确拒绝的负例TN之和除以该行总对数；不是正例Layout成功率。',
             componentIds=['light-scorer-summary', 'light-scorer-results-scope', 'light-scorer-results-table'],
             formula='(tp + negatives - fp) / pairs', dependencies=['tp', 'negatives', 'fp', 'pairs']),
        dict(label='同口径比较', definition='数据集、来源TEST范围、SIM/REAL选模方式和分类阈值策略同时一致，才能逐行比较。',
             componentIds=['light-scorer-summary', 'light-scorer-results-scope', 'light-scorer-results-table']),
        dict(label='固定案例', definition='每个已完成实验、每种选模方式各10个敦煌和1个Turufan固定案例；不是22个独立样本，也不保证均属于REAL-TEST。',
             componentIds=['light-scorer-case-view']),
        dict(label='点对Q与条件权重', definition='精确(i,j)并集保留原Q；Q乘观测弧长得到质量，质量归一化只用于条件汇聚，不是Attention。',
             componentIds=['light-scorer-case-view']),
    ]
    source = dict(label='已完成六作业冻结评价与固定数值案例', files=[str(evidence_path)], sha256=evidence_hash,
        executedAt=now, realPlanSha256=REAL_PLAN_SHA, metricDefinitions=definitions,
        evidenceFlow=[dict(title='终态与选择', detail='assemble核验六作业成功回执、已选checkpoint和预测SHA；compare保持同口径分组。'),
                      dict(title='案例数值', detail='复用同次前向的固定JSON/NPZ及既有独立数值审计；此展示不重新推理。')],
        caveats=bundle['caveats'] + ['固定案例是预先约定的开发诊断，不是REAL-TEST总体的随机抽样。'])
    for qid, rows in ((METRICS_QUERY, metrics), (CASES_QUERY, cases)):
        if qid in result['queries']:
            old = result['queries'][qid]['rows']
            key = (lambda r: r['case_key']) if qid == CASES_QUERY else (
                lambda r: '|'.join(r[k] for k in ('model', 'selection_kind', 'split', 'policy')))
            new = {key(r): digest_object(r) for r in rows}
            require(all(new.get(key(r)) == digest_object(r) for r in old),
                    'refresh must preserve already imported immutable results')
        result['queries'][qid] = dict(rows=rows, source=copy.deepcopy(source))
    result['queries'][CASES_QUERY]['payloadColumns'] = ['record_json']
    result['buildStatus'] = 'updating'
    result['generatedAt'] = now
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    require(not args.receipt.exists(), 'preserve prior receipts')
    target = args.project / 'src/data.json'
    source_hash, evidence_hash, annotation_hash = sha(target), sha(args.evidence), sha(args.annotations)
    original = read(target)
    result = bind_snapshot(original, read(args.evidence), args.evidence.resolve(), evidence_hash)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    backup = args.receipt.parent / 'report_before_light_results.json'
    require(not backup.exists(), 'preserve earlier backup'); shutil.copy2(target, backup)
    temp = target.with_suffix('.json.light.tmp')
    require(not temp.exists(), 'unfinished report write exists')
    with temp.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    require(sha(target) == sha(backup) == source_hash and sha(args.evidence) == evidence_hash
            and sha(args.annotations) == annotation_hash, 'report/evidence/annotations changed during binding')
    temp.replace(target)
    untouched = [q for q in original['queries'] if q not in (METRICS_QUERY, CASES_QUERY)]
    require(all(original['queries'][q] == result['queries'][q] for q in untouched), 'unrelated evidence changed')
    receipt = dict(schema='light-scorer-report-binding/1', status='bound_pending_build_and_visual_check',
        evidence_sha256=evidence_hash, snapshot_sha256=sha(target), annotations_sha256=annotation_hash,
        unchanged_queries=untouched, counts={q: len(result['queries'][q]['rows']) for q in (METRICS_QUERY, CASES_QUERY)},
        training_or_inference_performed=False)
    with args.receipt.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

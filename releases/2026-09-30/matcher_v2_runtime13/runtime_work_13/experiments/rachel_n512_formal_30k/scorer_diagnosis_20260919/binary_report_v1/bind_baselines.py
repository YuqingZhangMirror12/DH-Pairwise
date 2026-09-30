"""Append a matched-population baseline section to the existing local report."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .baselines import digest_object
from .assemble import REAL_PLAN_SHA
from .export import read, require, sha


QUERY = 'matched_real_test_baselines'
LABELS = {
    'mergefix_m12': '归并修复 · M12',
    'mergefix_scratch': '归并修复 · E32',
    'threshold_m12': '固定16px · M12',
    'threshold_scratch_fixed': '固定16px · E32',
}
REQUIRED_BASELINES = {'mergefix_m12', 'mergefix_scratch', 'threshold_m12'}


def assert_append_only(previous, updated):
    key = lambda r: (r['model'], r['split'], r['policy'])
    lookup = {key(r): r for r in updated}
    require(len(lookup) == len(updated), 'duplicate baseline row')
    require(all(key(r) in lookup and digest_object(r) == digest_object(lookup[key(r)])
                for r in previous), 'existing baseline rows changed or removed')


def reviewed_rows(bundle):
    require(bundle['schema'] == 'binary-report-all-models/1'
            and bundle['neural_inference_repeated'] is False
            and bundle['thresholds_refitted'] is False, 'frozen report evidence required')
    rows = []
    for row in bundle['rows']:
        if row['model'] not in LABELS or row['population'] != 'real_test':
            continue
        require(row['selection_kind'] == 'sim' and row['real_plan_sha256'] == REAL_PLAN_SHA,
                'do not mix selection protocols or source roles')
        require(row['split'] in ('dunhuang_cv', 'turufan')
                and row['policy'] in ('primary', 'fixed03'), 'unexpected matched population')
        metric = row['metrics']
        expected = (161, 59, 102) if row['split'] == 'dunhuang_cv' else (122, 61, 61)
        require(tuple(metric[k] for k in ('pairs', 'positives', 'negatives')) == expected,
                'registered REAL-TEST class counts differ')
        if row['split'] == 'turufan':
            require(metric['layout20_count'] is None and metric['joint_f1'] is None,
                    'do not invent Turufan Layout GT')
        rows.append(dict(row, model_label=LABELS[row['model']],
                         dataset_label='敦煌' if row['split'] == 'dunhuang_cv' else 'Turufan',
                         **metric))
    models = {r['model'] for r in rows}
    expected = {(m, s, p) for m in models for s in ('dunhuang_cv', 'turufan')
                for p in ('primary', 'fixed03')}
    require(REQUIRED_BASELINES <= models and len(rows) == len(expected)
            and {(r['model'], r['split'], r['policy']) for r in rows} == expected,
            'all completed baseline populations and both policies required')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    require(not args.receipt.exists(), 'keep prior binding receipts')
    bundle = read(args.evidence)
    evidence_hash = sha(args.evidence)
    rows = reviewed_rows(bundle)
    annotations_hash = sha(args.annotations)
    target = args.project / 'src/data.json'
    snapshot = read(target)
    original_id = snapshot['id']
    require(original_id == 'report:4b51ee5a-d8bb-4663-8305-69763a99c639',
            'only revise the established real-data review report')
    previous = snapshot['queries'].get(QUERY, {}).get('rows', [])
    assert_append_only(previous, rows)
    before = {key: digest_object(value) for key, value in snapshot['queries'].items() if key != QUERY}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    backup = args.receipt.parent / 'report_before_matched_baselines.json'
    require(not backup.exists(), 'preserve earlier report backup')
    shutil.copy2(target, backup)
    now = datetime.now(timezone.utc).isoformat()
    snapshot['queries'][QUERY] = dict(rows=rows, source=dict(
        label='旧冻结预测按已登记 REAL-TEST 来源重新统计；零次新推理',
        files=[str(args.evidence.resolve())],
        sha256=evidence_hash, realPlanSha256=REAL_PLAN_SHA, executedAt=now,
        evidenceFlow=[
            dict(title='保持已选模型', detail='所有列示复杂头的 checkpoint 和 SIM-CAL 阈值不变；保留原冻结预测及其哈希。'),
            dict(title='只重计同一子集', detail='按已登记来源折0划出 Dunhuang 161 对和 Turufan 122 对；不重新拟合任何阈值或选择轮次。'),
            dict(title='独立复算', detail='逐例复算 TP/FP/FN/TN、GT20 和 Joint，并与原全量冻结统计交叉核对；新增模型不改变已有行。'),
        ],
        caveats=[
            '来源隔离不等于全新盲测：这批真实数据已经参与开发诊断。',
            '本节不是原全量真实五折；两个评估口径分开，原报告保留。',
            '没有导入新轻量头成绩；空缺不填零，也不据此推断远端当前进度。',
            'Turufan没有GT位移，Layout与Joint均为空。',
            '不同Matcher起点和归并方式，不是只改变Scorer的因果消融。',
        ],
        metricDefinitions=[
            dict(label='Pair F1', definition='2TP/(2TP+FP+FN)，只评价可拼分类。', componentIds=['matched-real-test-table']),
            dict(label='Layout≤20px', definition='不论是否接受，正例最终赢家与GT位移距离≤20px的数量。', componentIds=['matched-real-test-table']),
            dict(label='Joint F1', definition='接受且GT20正确才计真阳性；正例错位接受和负例接受均计假阳性。', componentIds=['matched-real-test-table']),
        ],
    ))
    snapshot['buildStatus'] = 'updating'
    snapshot['generatedAt'] = now
    # Evidence cutoff remains unchanged: no new inference has occurred.
    temp = target.with_suffix('.json.matched.tmp')
    require(not temp.exists(), 'unfinished prior report write exists')
    with temp.open('x') as stream:
        json.dump(snapshot, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    require(sha(target) == sha(backup), 'report was edited while binding; do not overwrite')
    require(sha(args.annotations) == annotations_hash and sha(args.evidence) == evidence_hash,
            'annotations or evidence changed during report binding')
    temp.replace(target)
    after = read(target)
    require(after['id'] == original_id
            and all(digest_object(after['queries'][key]) == value for key, value in before.items()),
            'existing report identity or reviewed data changed')
    result = dict(schema='matched-real-test-report-bind/1', status='passed',
        query=QUERY, rows=len(rows), evidence_sha256=evidence_hash,
        historical_queries_unchanged=list(before), annotations_sha256=annotations_hash,
        previous_rows_preserved=len(previous),
        report_id=original_id, snapshot_sha256=sha(target),
        new_inference=False, changed_thresholds=False, real_development_not_blind=True)
    with args.receipt.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

"""Append the completed v17 experiment to the established report, without inference.

The original E32/head results, all other report queries, and human annotations
are immutable. Full cohorts and source-held-out TEST remain separate records.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .assemble import comparison_bundle, REAL_PLAN_SHA
from .bind_light_results import CASES_QUERY, LABELS, METRICS_QUERY, REPORT_ID, query_rows
from .compare import combined_evidence
from .export import read, require, sha
from .full_review import build as build_review

MODEL = 'aggressive_binary_patch'
POS_QUERY = 'light_scorer_all_positive_cases'
CF_QUERY = 'light_scorer_readout_interventions'
QUERIES = (METRICS_QUERY, CASES_QUERY, POS_QUERY, CF_QUERY)


def identity(query, row):
    if query in (CASES_QUERY, POS_QUERY):
        return row['case_key']
    fields = ('model', 'selection_kind', 'split', 'population', 'policy')
    if query == CF_QUERY:
        fields += ('intervention',)
    return tuple(row[k] for k in fields)


def additions(bundle, analysis):
    require(bundle['available_light_experiments'] == [MODEL], 'only append v17 here')
    _, fixed = query_rows(bundle)  # Reuse strict checkpoint/case/threshold validation.
    metrics = []
    for r in bundle['rows']:
        require(r['model'] == MODEL, 'old model must not be reimported')
        require(r['split'] != 'sim_test_v14', 'v17 simulation is not v14 TEST')
        if r['split'] == 'turufan':
            require(r['metrics']['layout20_count'] is None
                    and r['metrics']['joint_f1'] is None, 'Turufan has no Layout GT')
        metrics.append(dict(r, model_label=LABELS[MODEL], **r['metrics']))
    require(len(metrics) == 40 and len(fixed) == 22, 'six jobs and all recorded groups required')
    require(analysis['status'] == 'complete' and analysis['inference_repeated'] is False,
            'full-cohort numeric verification missing')
    require({(r['model'], r['selection_kind'], r['split']) for r in analysis['checks']}
            == {(MODEL, c, s) for c in ('sim', 'real') for s in ('dunhuang_cv', 'turufan')},
            'four full-real scopes required')
    gallery = analysis['all_positive_cases']
    for c in ('sim', 'real'):
        for s, n in (('dunhuang_cv', 292), ('turufan', 301)):
            require(sum(r['selection_kind'] == c and r['split'] == s for r in gallery) == n,
                    'positive gallery cohort differs')
    cf = analysis['interventions']
    require(len(cf) == 16 and all(not r['explicit_conflict_present'] for r in cf),
            'light head must not invent a conflict classifier')
    for r in cf:
        require(r['model'] == MODEL and r['changed_winners'] == 0, 'unexpected light-head reranking')
        actual = next(a for a in cf if all(a[k] == r[k] for k in
                      ('selection_kind', 'split', 'policy')) and a['intervention'] == 'actual')
        require(all(r[k] == actual[k] for k in ('tp', 'fp', 'fn', 'tn', 'layout20_count',
                    'joint_f1', 'winner_correct_but_rejected')), 'no-conflict must be an exact no-op')
    return dict(zip(QUERIES, (metrics, fixed, gallery, cf)))


def append_queries(snapshot, new_rows, provenance):
    require(snapshot['id'] == REPORT_ID, 'only update the established report')
    require(set(new_rows) == set(QUERIES), 'append all four related evidence queries')
    result = copy.deepcopy(snapshot)
    for query, rows in new_rows.items():
        require(query in result['queries'] and rows, 'existing query and new evidence required')
        current = result['queries'][query]
        old_keys = {identity(query, r) for r in current['rows']}
        keys = [identity(query, r) for r in rows]
        require(len(set(keys)) == len(keys) and not (old_keys & set(keys)),
                'do not duplicate or overwrite imported results')
        require(all(r['model'] == MODEL for r in rows), 'this append is v17 only')
        current['rows'].extend(copy.deepcopy(rows))
        current.setdefault('source', {})['v17Extension'] = copy.deepcopy(provenance)
    result['buildStatus'] = 'updating'
    result['generatedAt'] = datetime.now(timezone.utc).isoformat()
    require(all(snapshot['queries'][q] == result['queries'][q]
                for q in snapshot['queries'] if q not in QUERIES), 'unrelated query changed')
    return result


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--baselines', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    require(not args.out.exists(), 'preserve earlier report receipts')
    target = args.project / 'src/data.json'
    before, annotation_before = sha(target), sha(args.annotations)
    snapshot = read(target)
    # All six completed jobs, bound predictions and all 22 numeric traces are read once.
    imported = comparison_bundle({MODEL: args.evaluation})
    bundle = combined_evidence(read(args.baselines), imported)
    bundle['rows'] = [r for r in bundle['rows'] if r['model'] == MODEL]
    args.out.mkdir(parents=True)
    write_new(args.out / 'frozen_import.json', imported)
    bundle['input_sha256'] = {str((args.out / 'frozen_import.json').resolve()):
                             sha(args.out / 'frozen_import.json')}
    write_new(args.out / 'all_models.json', bundle)
    analysis = build_review(args.out / 'all_models.json', args.weights, snapshot, expected_scope_count=4)
    write_new(args.out / 'analysis.json', analysis)
    rows = additions(bundle, analysis)
    source = dict(label='v17从零Matcher＋Patch/Context轻头：六组冻结评价与逐例核验',
        files=[str((args.out / name).resolve()) for name in ('all_models.json', 'analysis.json')],
        fileSha256={name: sha(args.out / name) for name in ('all_models.json', 'analysis.json')},
        realPlanSha256=REAL_PLAN_SHA,
        metricDefinitions=[
            dict(label='v17实验边界', definition='v17从零训练Matcher、冻结SIM选中的E16，再训练34,529参数Patch/Context轻头；不是课程学习，也不是残差头。', componentIds=['v17-light-summary', 'v17-light-comparison']),
            dict(label='来源TEST与全量', definition='完整Dun800/Turu602包含CAL、SELECT和TEST；来源隔离TEST为Dun161/Turu122，不可相加。均属于开发性真实评价。', componentIds=['v17-light-summary', 'v17-light-comparison']),
            dict(label='完整失败案例', definition='两种选模方式各含全部292敦煌正例及301Turufan正例；Turu无布局GT。', componentIds=['light-full-failures'])],
        caveats=bundle['caveats'] + ['v17改变了训练数据和Matcher，不能把差异单独归因于Scorer。',
                                   '该头没有局部冲突分类器，移除显式冲突项是精确无操作。'])
    result = append_queries(snapshot, rows, source)
    shutil.copy2(target, args.out / 'report_before.json')
    require(sha(target) == before and sha(args.annotations) == annotation_before,
            'report or human annotations changed during binding')
    temp = target.with_suffix('.json.v17.tmp')
    write_new(temp, result)
    require(sha(target) == before and sha(args.annotations) == annotation_before,
            'concurrent update detected')
    temp.replace(target)
    receipt = dict(status='bound_pending_build_and_visual_check', old_snapshot_sha256=before,
        snapshot_sha256=sha(target), annotations_sha256=annotation_before,
        added_counts={q: len(r) for q, r in rows.items()},
        total_counts={q: len(result['queries'][q]['rows']) for q in QUERIES},
        inference_repeated=False, training_modified=False, thresholds_refitted=False)
    write_new(args.out / 'binding.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

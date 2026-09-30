"""Project verified frozen controls into the existing report's reviewed queries.

No inference, calibration, remote access, HTML renderer, or live-app mutation.
The CLI refuses incomplete analyses and existing output paths. Synthetic test
fixtures belong in temporary directories and must never be published as results.
"""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_factor_sweep import MODELS, SEARCHES, expected_readouts, require


MODEL_LABELS = {
    'threshold_scratch_fixed': '固定16px复杂头 · E32',
    'binary_patch': 'Patch/Context轻头 · E32',
    'binary_stats': 'Q/几何轻头 · E32',
    'aggressive_binary_patch': 'v17 Matcher＋Patch/Context轻头',
}
SEARCH_LABELS = {
    'baseline': '原搜索：Top-2／128模式／16种子',
    'top3_only': '只改行列Top-3',
    'all_modes_only': '只取消128模式截断',
    'seeds32_only': '只改32个初始种子',
    'combined': 'Top-3＋全部模式＋32种子',
}
READOUT_LABELS = {
    'baseline': '原Scorer读出',
    'raw_sum_q_rank': '按原Q总量选簇，保留其Scorer分数',
    'raw_max_q_rank': '按最大单点Q选簇，保留其Scorer分数',
    'no_overlap_penalty': '去掉显式重叠扣分',
    'no_conflict_penalty': '去掉显式冲突扣分',
    'no_conflict_or_overlap': '去掉显式冲突＋重叠扣分',
    'zero_overlap_feature': '旧轻头的重叠统计输入置零',
    'zero_mean': '旧轻头的均值特征置零',
    'zero_max': '旧轻头的最大值特征置零',
    'zero_mean_max': '旧轻头的均值＋最大值特征置零',
}
DOMAIN_LABELS = {'dunhuang_cv': '敦煌', 'turufan': 'Turufan', 'sim_select': '仿真SELECT'}
ROLE_SIZE = {('sim_select', 'sim_select'): 1500,
             ('dunhuang_cv', 'real_cal'): 160, ('dunhuang_cv', 'real_select'): 479,
             ('turufan', 'real_cal'): 120, ('turufan', 'real_select'): 360}
MODES = {'original_frozen_cal': '冻结原SIM-CAL阈值',
         'separate_real_cal': '仅REAL-CAL重校准后评价'}
DELTA_FIELDS = ('accuracy', 'f1', 'fp', 'layout20_count', 'candidate_coverage_count',
                'joint_tp', 'joint_f1', 'winner_correct_but_rejected')
LAYOUT_FIELDS = ('layout20_count', 'candidate_coverage_count', 'layout20',
                 'candidate_coverage', 'winner_correct_but_rejected',
                 'covered_but_winner_wrong', 'positive_no_correct_candidate',
                 'wrong_pose_accepted', 'joint_tp', 'joint_fp', 'joint_fn',
                 'joint_f1', 'joint_precision', 'joint_recall', 'known_positive_layouts')


def _close(a, b):
    return a is None and b is None or (a is not None and b is not None and
                                      abs(a - b) <= 1e-10)


def _key(row):
    return tuple(row[k] for k in ('split', 'role', 'search', 'readout', 'threshold_mode'))


def _expected(model):
    return {(split, role, search, readout, mode)
            for split, role in ROLE_SIZE for search in SEARCHES
            for readout in expected_readouts(model)
            for mode in (('original_frozen_cal',) if split == 'sim_select' else MODES)}


def _validate_metric(metric, split, size):
    require(metric['pairs'] == size, 'wrong metric population')
    for key in ('positives', 'negatives', 'tp', 'tn', 'fp', 'fn'):
        require(type(metric[key]) is int and metric[key] >= 0, 'invalid confusion count')
    p, n = metric['positives'], metric['negatives']
    require(p + n == size and metric['tp'] + metric['fn'] == p and
            metric['fp'] + metric['tn'] == n, 'confusion counts do not reconcile')
    require(_close(metric['accuracy'], (metric['tp'] + metric['tn']) / size) and
            _close(metric['f1'], 2 * metric['tp'] / max(1, 2 * metric['tp'] + metric['fp'] + metric['fn'])),
            'reported classification arithmetic differs')
    require(.20 <= metric['threshold'] <= .80, 'threshold outside locked range')
    if split == 'turufan':
        require(all(metric[key] is None for key in LAYOUT_FIELDS), 'Turufan has invented Layout metric')
    else:
        good, covered, joint = (metric[k] for k in ('layout20_count', 'candidate_coverage_count', 'joint_tp'))
        require(0 <= joint <= good <= covered <= p, 'Layout counts do not reconcile')
        require(metric['winner_correct_but_rejected'] == good - joint and
                metric['covered_but_winner_wrong'] == covered - good and
                metric['positive_no_correct_candidate'] == p - covered,
                'Layout failure categories do not reconcile')
        require(metric['joint_fp'] == metric['fp'] + metric['wrong_pose_accepted'] and
                metric['joint_fn'] == p - joint and
                _close(metric['joint_f1'], 2 * joint / max(1, 2 * joint + metric['joint_fp'] + metric['joint_fn'])),
                'reported joint arithmetic differs')


def validate(analysis, *, completed_models=None):
    selected = tuple(MODELS if completed_models is None else completed_models)
    require(bool(selected) and len(set(selected)) == len(selected) and
            set(selected) <= set(MODELS), 'invalid completed-model scope')
    partial = set(selected) != set(MODELS)
    require(analysis['schema'] == 'decoder-controls-independent-analysis/1' and
            analysis['status'] == ('model_complete' if partial else 'complete'),
            'complete independent analysis required for the declared scope')
    require(all(analysis[k] is False for k in ('test_used', 'training_performed', 'model_modified')),
            'not frozen development controls')
    require(analysis['historical_real_exposure'] is True, 'development exposure caveat missing')
    require(set(analysis['models']) == set(selected), 'exact declared completed models required')
    if partial:
        require(analysis['all_models_analyzed'] is False and
                set(analysis['requested_models']) == set(selected), 'partial scope mislabeled as full')
    # This also rejects NaN/Infinity rather than allowing JSON null/zero substitution.
    json.dumps(analysis, allow_nan=False)
    for model, result in analysis['models'].items():
        if partial:
            proof = result['exit_evidence']
            require(proof['returncode'] == 0 and len(proof['sha256']) == 64 and
                    proof['kind'] in ('controller_exit_receipt', 'independently_observed_linux_exit'),
                    'completed model lacks verified successful process exit')
        require(result['model'] == model and result['protocol']['model'] == model, 'model identity mismatch')
        require(result['automatic_production_change'] is False and
                result['new_pooling_heads_included'] is False, 'new training or promotion mixed into controls')
        by_key = {_key(row): row for row in result['rows']}
        require(len(by_key) == len(result['rows']) and set(by_key) == _expected(model),
                'missing, duplicate, TEST or extra report rows')
        protocol_threshold = result['protocol']['threshold']
        distribution_seen = {}
        for key, row in by_key.items():
            split, role, search, readout, mode = key
            metric = row['metrics']; _validate_metric(metric, split, ROLE_SIZE[(split, role)])
            baseline = by_key[(split, role, 'baseline', 'baseline', mode)]['metrics']
            require(_close(row['baseline_threshold'], baseline['threshold']), 'wrong comparison threshold')
            if mode == 'original_frozen_cal':
                require(_close(metric['threshold'], protocol_threshold), 'frozen threshold silently changed')
            else:
                cal = by_key[(split, 'real_cal', search, readout, mode)]['metrics']
                require(_close(metric['threshold'], cal['threshold']), 'SELECT differs from its CAL threshold')
            for name in DELTA_FIELDS:
                expected = metric[name] - baseline[name] if metric[name] is not None else None
                require(_close(row['delta'][name], expected), 'delta differs from same-population baseline')
            paired = row['paired']
            require(set(paired['counts']) == set(paired['pair_ids']), 'paired identities missing')
            for name, count in paired['counts'].items():
                ids = paired['pair_ids'][name]
                require(count is None and ids is None or
                        type(count) is int and isinstance(ids, list) and len(ids) == len(set(ids)) == count,
                        'paired event count differs from IDs')
            for left, right in (('classification_gained', 'classification_lost'),
                                ('positive_recovered', 'positive_lost'),
                                ('false_positive_added', 'false_positive_removed'),
                                ('layout_gained', 'layout_lost'),
                                ('coverage_gained', 'coverage_lost'),
                                ('correct_and_accepted_gained', 'correct_and_accepted_lost')):
                require(not set(paired['pair_ids'][left] or []) & set(paired['pair_ids'][right] or []),
                        'same pair reported as both gained and lost')
            counts = paired['counts']
            require(counts['classification_gained'] - counts['classification_lost'] ==
                    metric['tp'] + metric['tn'] - baseline['tp'] - baseline['tn'], 'paired class delta differs')
            require(counts['false_positive_added'] - counts['false_positive_removed'] == row['delta']['fp'],
                    'paired false-positive delta differs')
            if split != 'turufan':
                require(counts['layout_gained'] - counts['layout_lost'] == row['delta']['layout20_count'] and
                        counts['correct_and_accepted_gained'] - counts['correct_and_accepted_lost'] == row['delta']['joint_tp'],
                        'paired Layout delta differs')
            expected_mechanism = readout in ('zero_overlap_feature', 'zero_mean', 'zero_max', 'zero_mean_max')
            require(row['mechanism_only'] == expected_mechanism and
                    row['not_a_calibrated_q_classifier'] == readout.startswith('raw_'), 'diagnostic semantic flag changed')
            expected_cohorts = dict(all_positive_winners=metric['positives'],
                                    all_negative_winners=metric['negatives'],
                                    correct_positive_winners=metric['layout20_count'] or 0)
            require(set(row['distributions']) == set(expected_cohorts), 'missing distribution cohort')
            for cohort, fields in row['distributions'].items():
                require(set(fields) == {'pairs', 'sum_q', 'max_q', 'candidate_count', 'score'},
                        'missing evidence distribution')
                for quantiles in fields.values():
                    require(quantiles['count'] == expected_cohorts[cohort], 'distribution survivor population differs')
                    values = [quantiles['p' + str(q)] for q in (10, 25, 50, 75, 90)]
                    require(all(v is None for v in values + [quantiles['mean']]) if quantiles['count'] == 0
                            else all(type(v) in (int, float) for v in values + [quantiles['mean']]) and
                            all(a <= b for a, b in zip(values, values[1:])), 'invalid or invented distribution')
            dk = (split, role, search, readout)
            if dk in distribution_seen:
                require(row['distributions'] == distribution_seen[dk], 'threshold-independent evidence differs')
            distribution_seen[dk] = row['distributions']
            strict = row['strict_cpu_gpu_reproduction_subset']
            if split == 'sim_select':
                require(strict is None, 'invented SIM CPU/GPU subset')
            else:
                require(strict is not None and strict['threshold_refitted_on_subset'] is False and
                        strict['pairs'] + strict['excluded_pairs'] == metric['pairs'], 'invalid strict subset')
                if strict['metrics'] is not None:
                    _validate_metric(strict['metrics'], split, strict['pairs'])
                    require(_close(strict['metrics']['threshold'], metric['threshold']), 'strict subset recalibrated')
                else:
                    require(strict['pairs'] == 0, 'nonempty strict subset lacks metrics')
    return analysis


def project(analysis, *, source_name, source_sha256, completed_models=None):
    validate(analysis, completed_models=completed_models)
    selected = [model for model in MODELS if model in analysis['models']]
    require(Path(source_name).name == source_name and len(source_sha256) == 64 and
            all(c in '0123456789abcdef' for c in source_sha256), 'safe source identity required')
    tables = {name: [] for name in ('metrics', 'paired', 'distributions', 'strict', 'models')}
    for model in selected:
        result = analysis['models'][model]; seen_distributions = set()
        tables['models'].append(dict(model=model, model_label=MODEL_LABELS[model],
            sim_data='v17' if model == 'aggressive_binary_patch' else 'v14',
            frozen_threshold=result['protocol']['threshold'],
            checkpoint_sha256=result['protocol']['checkpoint_sha256'],
            sim_select_manifest_sha256=result['sim_select_manifest_sha256'],
            real_cal_pairs=280, real_select_pairs=839, sim_select_pairs=1500,
            new_pooling_training_included=False, test_used=False))
        for row in result['rows']:
            split, role, search, readout, mode = _key(row)
            key = ':'.join((model, split, role, search, readout, mode))
            base = dict(id=key, model=model, model_label=MODEL_LABELS[model],
                split=split, dataset=DOMAIN_LABELS[split], role=role,
                search=search, search_label=SEARCH_LABELS[search],
                readout=readout, readout_label=READOUT_LABELS[readout],
                threshold_mode=mode, threshold_label=MODES[mode],
                sim_data='v17' if model == 'aggressive_binary_patch' else 'v14',
                mechanism_only=row['mechanism_only'], raw_q_ranking_only=row['not_a_calibrated_q_classifier'],
                control_class=('原始基线' if search == 'baseline' and readout == 'baseline' else
                               '单因素' if (search == 'baseline') != (readout == 'baseline') and
                               search != 'combined' and readout not in ('no_conflict_or_overlap', 'zero_mean_max') else '联合干预'))
            tables['metrics'].append(dict(base, **row['metrics'], baseline_threshold=row['baseline_threshold'],
                **{'delta_' + k: v for k, v in row['delta'].items()},
                decode_seconds_mean=row['runtime_seconds']['mean'],
                decode_seconds_p90=row['runtime_seconds']['p90']))
            tables['paired'].append(dict(base, **row['paired']['counts'],
                pair_ids_json=json.dumps(row['paired']['pair_ids'], ensure_ascii=False, sort_keys=True)))
            strict = row['strict_cpu_gpu_reproduction_subset']
            if strict is not None:
                tables['strict'].append(dict(base, pairs=strict['pairs'], excluded_pairs=strict['excluded_pairs'],
                    threshold_refitted_on_subset=False,
                    metrics_json=json.dumps(strict['metrics'], ensure_ascii=False, sort_keys=True),
                    paired_json=json.dumps(strict['paired'], ensure_ascii=False, sort_keys=True)))
            # Evidence values do not depend on the calibration threshold; do not duplicate them.
            dk = (split, role, search, readout)
            if dk not in seen_distributions:
                seen_distributions.add(dk)
                distribution_base = {k: v for k, v in base.items() if k not in ('id', 'threshold_mode', 'threshold_label')}
                for cohort, fields in row['distributions'].items():
                    for field, quantiles in fields.items():
                        tables['distributions'].append(dict(distribution_base,
                            id=':'.join((model, *dk, cohort, field)), cohort=cohort, field=field, **quantiles))
    caveats = [
        '完整开发集单次冻结CPU对照，不是重新训练后的新模型，也不构成全新盲测。',
        'REAL-CAL为来源fold1；REAL-SELECT为fold2/3/4；保留fold0及仿真TEST均未用于选择。',
        '冻结SIM-CAL阈值与每个干预在REAL-CAL重校准的结果分开；SELECT不调分类阈值。',
        'Turufan没有Layout GT，Layout与Joint指标保留null，不显示为0或正确。',
        '原Q排序只改变赢家，分类仍使用该赢家的原Scorer分数，不是新Q分类器。',
        '旧头输入置零属于分布外机制诊断；新Pooling头训练结果不在本表中。',
        'Top-3、模式和种子是候选搜索控制，T16直径、最终8候选及物理阻断不变。',
        '固定E32模型使用v14仿真SELECT，v17模型使用自己的v17 SELECT；不合并为统一总体准确率。',
        '全CPU总体与严格CPU/GPU复现子集分开，子集不重校准；单项收益不能直接相加。',
    ]
    if len(selected) != len(MODELS):
        caveats.insert(0, f'阶段报告：仅{len(selected)}/4个已完整验收模型；其余模型不纳入，不将部分结果称为四模型完成。')
    queries = {}
    for name, rows in tables.items():
        qid = 'decoder_factor_' + name
        query = dict(rows=rows, source=dict(type='file', files=[source_name],
            evidenceFlow=[dict(title='完整冻结对照的独立复算',
                detail='分析文件SHA256：' + source_sha256 + '；只做展示投影，不运行推理或调阈值。')],
            caveats=caveats, metricDefinitions=[dict(label='分类与布局的同总体配对比较',
                definition='每条记录对应一个固定模型、数据角色、搜索和读出干预。分类F1与JointF1不可互换；率差使用百分点展示。',
                componentIds=['decoder-factor-' + name])]))
        if name == 'paired': query['payloadColumns'] = ['pair_ids_json']
        if name == 'strict': query['payloadColumns'] = ['metrics_json', 'paired_json']
        queries[qid] = query
    return dict(schema='decoder-controls-report-projection/1', status=analysis['status'], queries=queries,
                completed_models=selected, total_planned_models=len(MODELS),
                new_pooling_heads_included=False, live_report_modified=False,
                source_analysis_sha256=source_sha256,
                notes='Prepared query bundle only; merge into the existing report and verify rendered components after actual evidence is complete.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--completed-model', action='append', choices=MODELS,
                        help='Explicit verified model-complete subset; omission requires all four.')
    args = parser.parse_args()
    path = Path(args.analysis); raw = path.read_bytes()
    value = project(json.loads(raw), source_name=path.name, source_sha256=hashlib.sha256(raw).hexdigest(),
                    completed_models=args.completed_model)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status='projection_complete', output=str(out),
        row_counts={name: len(query['rows']) for name, query in value['queries'].items()},
        live_report_modified=False)))


if __name__ == '__main__':
    main()

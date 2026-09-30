"""Prepare an additive report snapshot from completed F/I frozen evaluation.

The current report and its three-model human-review cases are read-only inputs.
No inference, calibration, training, browser storage or annotation-file writes.
This produces a separate candidate snapshot for the final report update.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median

ARMS = {
    'frozen_features': 'F · 冻结特征续训',
    'independent_features': 'I · 独立Scorer特征',
}
SPLITS = {'dunhuang_cv': '敦煌', 'turufan': 'Turufan'}
APP_ID = 'report:4b51ee5a-d8bb-4663-8305-69763a99c639'


def read(path):
    return json.loads(Path(path).read_text())


def index(rows):
    result = {r['pair_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate pair ids')
    return result


def extend(snapshot, comparison, oof, bundles):
    """Pure merge; preserve original cases/assets/presentation and legacy metrics."""
    if snapshot.get('id') != APP_ID:
        raise ValueError('not the existing seam report')
    extension = comparison.get('independent_feature_extension', {})
    if (extension.get('same_source_folds') is not True
            or extension.get('checkpoint_selection') != 'simulation CAL/SELECT only'
            or extension.get('score_rescaling') is not False):
        raise ValueError('missing frozen same-fold comparison')
    result = copy.deepcopy(snapshot)
    queries = result['queries']
    cases = snapshot['queries']['cases']['rows']
    if not cases or set(c['dataset'] for c in cases) != set(SPLITS.values()):
        raise ValueError('report population missing a dataset')
    additions = {}
    metrics = [r for r in queries['metrics']['rows'] if r['model'] not in ARMS.values()]
    summaries = []
    for split, dataset in SPLITS.items():
        existing = index([c for c in cases if c['dataset'] == dataset])
        for arm, label in ARMS.items():
            bundle = bundles[arm][split]
            protocol, status = bundle['protocol'], bundle['status']
            model = comparison['splits'][split]['models'][arm]
            if (protocol.get('status') != 'complete' or status.get('status') != 'complete'
                    or protocol.get('arm') != arm or protocol.get('split') != split
                    or protocol.get('frozen_matcher_verified') is not True
                    or any(protocol.get(k) is not False for k in (
                        'real_used_to_select_checkpoint', 'threshold_fitting',
                        'gt_used_to_generate_candidates'))):
                raise ValueError('evaluation incomplete or incompatible')
            if (protocol['checkpoint_sha256'] != model['checkpoint_sha256']
                    or protocol['selected_epoch'] != model['selected_epoch']
                    or protocol['threshold'] != model['sim_frozen']['threshold']):
                raise ValueError('model/threshold binding mismatch')
            pred = index(bundle['rows'])
            fold = index(oof[split][arm]['bounded_max_f1'])
            if (pred.keys() != existing.keys() or pred.keys() != fold.keys()
                    or status['count'] != len(existing)
                    or protocol['sample_count'] != len(existing)):
                raise ValueError('report/inference/calibration population mismatch')
            for pair_id, p in pred.items():
                f, c = fold[pair_id], existing[pair_id]
                valid = bool(p['numeric_valid'] and p['has_candidate'])
                if (p['score'] != f['score'] or bool(p['label']) != bool(c['label'])
                        or f['accepted'] != (valid and p['score'] >= f['threshold'])
                        or p['target_translation_rc'] != c['gt']):
                    raise ValueError('score, target or decision changed during join')
                if split == 'turufan' and p['gt_known']:
                    raise ValueError('Turufan must not acquire invented layout GT')
                winner = p['candidates'][p['winner_index']] if p['has_candidate'] else None
                additions.setdefault((dataset, pair_id), {})[arm] = dict(
                    score=p['score'], translation=p['translation'],
                    threshold_cv=f['threshold'], threshold_sim=protocol['threshold'],
                    accepted_cv=f['accepted'], decision_valid=valid,
                    layout_error_px=p['error_px'],
                    layout20=p['layout20'] if p['gt_known'] else None,
                    selected_epoch=protocol['selected_epoch'],
                    checkpoint_sha256=protocol['checkpoint_sha256'],
                    winner_index=p['winner_index'],
                    edge_count=winner['edge_count'] if winner else 0,
                    residual=winner['residual_median_px'] if winner else None)
            for policy in ('SIM冻结', '真实五折'):
                frozen = policy == 'SIM冻结'
                cv = model['cv']['bounded_max_f1']
                m = model['sim_frozen'] if frozen else cv['pooled_out_of_fold']
                thresholds = [protocol['threshold']] if frozen else [f['threshold'] for f in cv['folds']]
                metrics.append(dict(dataset=dataset, model=label, policy=policy,
                    threshold=median(thresholds), thresholds=thresholds,
                    **{k: m[k] for k in ('accuracy', 'precision', 'recall', 'f1', 'auroc',
                        'tp', 'fp', 'fn', 'tn', 'n', 'positive', 'negative')},
                    layout=m.get('layout_correct_total'), joint=m.get('layout_correct_accepted'),
                    primary=True, selected_epoch=model['selected_epoch'],
                    checkpoint_sha256=model['checkpoint_sha256']))
            summaries.append(dict(dataset=dataset, arm=arm, model=label,
                selected_epoch=model['selected_epoch'], checkpoint_sha256=model['checkpoint_sha256'],
                sim_threshold=protocol['threshold'],
                proposal_comparison=comparison['splits'][split]['independent_feature_proposals'],
                paired_changes={k: v for k, v in comparison['splits'][split]['independent_feature_paired'].items()
                    if k.startswith(arm + '_vs_')}))
    # Do not add fields/models to the source cases used by human-review fingerprints.
    independent_cases = []
    for c in cases:
        row = {k: copy.deepcopy(c[k]) for k in (
            'pair_id', 'dataset', 'label', 'fragment_a', 'fragment_b', 'case_name', 'fold', 'gt')}
        row['models'] = copy.deepcopy(c['models'])
        row['models'].update(additions[(c['dataset'], c['pair_id'])])
        independent_cases.append(row)
    queries['metrics']['rows'] = metrics
    queries['independent_cases'] = dict(rows=independent_cases)
    queries['independent_features'] = dict(rows=summaries)
    return result


def run(a):
    root, snapshot_path, output = Path(a.root), Path(a.snapshot), Path(a.out)
    if output.resolve() == snapshot_path.resolve() or output.exists():
        raise ValueError('output must be a new file; do not replace the live report')
    original = snapshot_path.read_bytes()
    bundles = {arm: {} for arm in ARMS}
    files = [root/'comparison.json', root/'oof_predictions.json']
    for arm in ARMS:
        for split in SPLITS:
            p = root/arm/split
            bundles[arm][split] = dict(protocol=read(p/'protocol.json'), status=read(p/'status.json'),
                rows=[json.loads(s) for s in (p/'case_diagnostics.jsonl').read_text().splitlines()])
            files.extend(p/name for name in ('protocol.json', 'status.json', 'case_diagnostics.jsonl'))
    result = extend(json.loads(original), read(files[0]), read(files[1]), bundles)
    caveats = [
        'F/I权重只由仿真CAL/SELECT选择；真实阈值使用既定来源五折及0.20–0.80网格。',
        '训练均完成16轮预算，非充分收敛证明；单次训练不能宣称统计显著性。',
        'Turufan无GT Layout，不报告其客观布局准确率。',
        '原三模型人工审核数据与指纹保持不变，F/I仅加入独立对照数据。',
        '原有heatmaps仍是旧S7/v3；F/I热图需从各自胜出权重另行导出。',
    ]
    source = dict(label='F/I冻结真实评价与原模型同源五折比较', files=[str(f) for f in files],
        caveats=caveats, preserved_snapshot_sha256=hashlib.sha256(original).hexdigest())
    for key in ('independent_cases', 'independent_features'):
        result['queries'][key]['source'] = copy.deepcopy(source)
    metric_source = result['queries']['metrics']['source']
    metric_source['files'] = list(dict.fromkeys(metric_source.get('files', []) + source['files']))
    metric_source['caveats'] = list(dict.fromkeys(metric_source.get('caveats', []) + caveats[:3]))
    result['generatedAt'] = datetime.now(timezone.utc).isoformat()
    result['buildStatus'] = 'updating'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        f.write(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), app_id=result['id'],
        unchanged_human_review_cases=len(result['queries']['cases']['rows']),
        independent_cases=len(result['queries']['independent_cases']['rows']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'snapshot', 'out'):
        parser.add_argument('--'+name, required=True)
    run(parser.parse_args())

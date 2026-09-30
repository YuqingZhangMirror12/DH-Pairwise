"""Extend the existing reviewed report in place; retain all legacy rows/assets."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median

KEY = 'S7_H'
LABEL = 'S7-H · 纯困难微调'


def read(path):
    return json.loads(Path(path).read_text())


def index(values):
    result = {r['pair_id']: r for r in values}
    if len(result) != len(values):
        raise ValueError('duplicate pair ids')
    return result


def run(a):
    root, path = Path(a.root).resolve(), Path(a.app).resolve() / 'src/data.json'
    comparison, oof = read(root / 'comparison.json'), read(root / 'oof_predictions.json')
    status = read(root.parent / 'training/status.json')
    if status['status'] != 'training_complete':
        raise ValueError('training is not complete')
    current = read(path)
    if current['id'] != 'report:4b51ee5a-d8bb-4663-8305-69763a99c639':
        raise ValueError('not the existing seam report')
    queries = current['queries']
    metrics = [r for r in queries['metrics']['rows'] if r['model'] != LABEL]
    additions, comparisons = {}, []
    for split, dataset in [('dunhuang_cv', '敦煌'), ('turufan', 'Turufan')]:
        model = comparison['splits'][split]['models'][KEY]
        pred = index([json.loads(s) for s in (root / split / 'case_diagnostics.jsonl').read_text().splitlines()])
        fold = index(oof[split][KEY]['bounded_max_f1'])
        if pred.keys() != fold.keys():
            raise ValueError('prediction/calibration population mismatch')
        for policy in ['SIM冻结', '真实五折']:
            result = model['sim_frozen'] if policy == 'SIM冻结' else model['cv']['bounded_max_f1']['pooled_out_of_fold']
            thresholds = [model['sim_frozen']['threshold']] if policy == 'SIM冻结' else [r['threshold'] for r in model['cv']['bounded_max_f1']['folds']]
            metrics.append(dict(dataset=dataset, model=LABEL, policy=policy,
                threshold=median(thresholds), thresholds=thresholds,
                **{k: result[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'tp', 'fp', 'fn', 'tn', 'n', 'positive', 'negative']},
                layout=result.get('layout_correct_total'), joint=result.get('layout_correct_accepted'), primary=True))
        for pair_id, p in pred.items():
            f = fold[pair_id]
            if f['score'] != p['score'] or pair_id in additions:
                raise ValueError('score changed during calibration or duplicate population')
            additions[pair_id] = dict(score=p['score'], translation=p['translation'],
                threshold_cv=f['threshold'], threshold_sim=model['sim_frozen']['threshold'],
                accepted_cv=f['accepted'], decision_valid=p['decision_valid'],
                layout_error_px=p['layout_error_px'], layout20=p['layout_good_20'],
                endpoints_a=p['endpoints_a'], endpoints_b=p['endpoints_b'],
                inlier_count=p['inlier_count'], used_fallback=p['used_fallback'])
        comparisons.append(dict(dataset=dataset, selected_matcher_epoch=model['selected_matcher_epoch'],
            selected_scorer_epoch=model['selected_scorer_epoch'], checkpoint_sha256=model['checkpoint_sha256'],
            paired_changes=comparison['splits'][split]['s7_hard_vs_original']))
    cases = queries['cases']['rows']
    if set(additions) != set(index(cases)) or len(cases) != 1405:
        raise ValueError('report/new inference population mismatch')
    for c in cases:
        c['models'][KEY] = additions[c['pair_id']]
    queries['metrics']['rows'] = metrics
    for name, files in [('metrics', ['comparison.json']), ('cases', ['oof_predictions.json', 'dunhuang_cv/case_diagnostics.jsonl', 'turufan/case_diagnostics.jsonl'])]:
        s = queries[name]['source']
        s['files'] = list(dict.fromkeys(s['files'] + [str(root / f) for f in files]))
        caveat = 'S7-H权重只由独立仿真SELECT选择；真实阈值仍采用原来源五折与0.20–0.80网格。'
        s['caveats'] = list(dict.fromkeys(s.get('caveats', []) + [caveat]))
    queries['s7_hard'] = dict(rows=comparisons, source=dict(
        label='S7-H纯困难微调完成及冻结真实结果',
        files=[str(root.parent / 'training/status.json'), str(root / 'comparison.json'),
               str(root / 'dunhuang_cv/protocol.json'), str(root / 'turufan/protocol.json')],
        caveats=['既有S7有效困难子集7832对，非新增来源；3916正/3916负。',
                 'M8→C8有限微调预算不等于证明收敛；新优化器和有效batch64也有变化，非纯数据因素对照。',
                 'F/I按用户要求暂停；本轮未恢复。',
                 '页面逐层热图仍属于原S7/v3诊断，不是S7-H热图。']))
    current['buildStatus'] = a.build_status
    current['generatedAt'] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(current, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps(dict(app_id=current['id'], cases=len(cases), metrics=len(metrics), extended_model=KEY)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--app', required=True)
    p.add_argument('--build-status', choices=['updating', 'complete'], default='updating')
    run(p.parse_args())

"""Local, read-only recount of saved predictions; no inference or fitting.

Writes only generated evidence and table fragments in this handoff directory.
The common baseline cohort is intersected by exact pair ID, with ordered
fragment IDs, labels and existing translation GT checked independently.
"""
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HIST = ROOT / 'artifacts/matcher_v2_20260930/historical_b3_review_01'
SOURCES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, expected=None, lines=False):
    path = Path(path)
    digest = sha(path)
    if expected:
        assert digest == expected, str(path)
    SOURCES[str(path.relative_to(ROOT))] = digest
    text = path.read_text()
    return [json.loads(x) for x in text.splitlines()] if lines else json.loads(text)


def summarize(rows, threshold, geometry):
    pos = [r for r in rows if r['label']]
    neg = [r for r in rows if not r['label']]
    accept = lambda r: r['valid'] and r['score'] >= threshold
    tp = sum(accept(r) for r in pos)
    fp = sum(accept(r) for r in neg)
    fn = len(pos) - tp
    jt = sum(accept(r) and r['layout20'] for r in pos) if geometry else None
    wrong = sum(accept(r) and not r['layout20'] for r in pos) if geometry else None
    pscore = [r['score'] if r['valid'] else 0. for r in pos]
    nscore = sorted(r['score'] if r['valid'] else 0. for r in neg)
    auc = (sum(bisect_left(nscore, x) + bisect_right(nscore, x) for x in pscore)
           / (2 * len(pos) * len(neg))) if pos and neg else None
    return dict(pairs=len(rows), positives=len(pos), negatives=len(neg), threshold=threshold,
                tp=tp, fp=fp, fn=fn, tn=len(neg)-fp, accuracy=(tp+len(neg)-fp)/len(rows),
                recall=tp/len(pos), f1=2*tp/max(1, 2*tp+fp+fn), auc=auc,
                layout20_count=sum(r['layout20'] for r in pos) if geometry else None,
                joint_tp=jt, wrong_pose_accepted=wrong,
                joint_f1=2*jt/max(1, 2*jt+fp+wrong+len(pos)-jt) if geometry else None)


def check_metrics(actual, expected, mapping):
    for key, oldkey in mapping.items():
        a, b = actual[key], expected[oldkey]
        assert (a is None and b is None) or (a is not None and b is not None
                                              and abs(a-b) < 1e-10), (key, a, b)


def main():
    verify = read(HIST/'verification.json')
    ledger = read(HIST/'analysis.json')
    raw = read(HIST/'before_data.json')['queries']['cases']['rows']
    bad = set(verify['excluded_bad_gt'])
    cases = {r['pair_id']: r for r in raw if r['pair_id'] not in bad}
    assert len(cases) == 1402
    dun = {k:v for k,v in cases.items() if v['dataset'] == '敦煌'}
    turu = {k:v for k,v in cases.items() if v['dataset'] == 'Turufan'}
    assert (len(dun), len(turu)) == (800, 602)
    modern, thresholds, identities = {}, {}, {}
    full_metrics, test_metrics = [], []
    labels = {'aggressive_binary_patch':'v17 · Patch', 'b3_patch':'B3-H · Patch', 'b3_stats':'B3-H · Stats'}
    mapping = {k:k for k in ['pairs','positives','negatives','tp','fp','fn','tn','accuracy','f1',
                              'layout20_count','joint_tp','wrong_pose_accepted','joint_f1']}
    for s in verify['frozen_sources']:
        model, dataset = s['model'], s['dataset']
        if model not in labels or s['selection'] != 'sim':
            continue
        pred = read(ROOT/s['predictions_file'], s['predictions_sha256'], lines=True)
        read(ROOT/s['summary_file'], s['summary_sha256'])
        cohort = dun if dataset == 'dunhuang_cv' else turu
        rows = {}
        for p in pred:
            if p['pair_id'] in bad:
                continue
            c = cohort[p['pair_id']]
            valid = bool(p['numeric_valid'] and p['has_candidate'])
            good = bool(c['gt'] is not None and valid and p['translation'] is not None
                        and math.dist(c['gt'], p['translation']) <= 20)
            rows[p['pair_id']] = dict(pair_id=p['pair_id'], label=c['label'], fold=c['fold'],
                                     score=p['score'], valid=valid, layout20=good)
        assert set(rows) == set(cohort)
        modern[(model,dataset)] = rows
        thresholds[model] = s['threshold']
        identities[model] = s['checkpoint_sha256']
        for population in ['full','test']:
            sel = [v for v in rows.values() if population == 'full' or v['fold'] == 0]
            metric = summarize(sel, s['threshold'], dataset == 'dunhuang_cv')
            expected = next(m for m in ledger['metrics'] if m['model']==model and m['selection']=='sim'
                            and m['dataset']==dataset and m['population']==population and m['policy']=='primary')
            check_metrics(metric, expected, mapping)
            metric.update(model=labels[model], model_id=model, dataset=dataset, population=population)
            (full_metrics if population == 'full' else test_metrics).append(metric)

    baselines = {}
    baseline_identity, baseline_thresholds = {}, {}
    for name in ['pairingnet','shreddingnet']:
        r = ROOT/f'reports/rachel_recall_benchmarks_20260911_001/completed/{name}/real'
        receipt = read(r/'receipt.json')
        summ = read(r/'summary.json',receipt['summary_sha256'])
        pred = read(r/'pair_results.jsonl',receipt['pair_results_sha256'],lines=True)
        assert len(pred)==len({x['pair_id'] for x in pred})==1016
        baselines[name] = {x['pair_id']:x for x in pred}
        baseline_identity[name] = receipt['model_identity']
        baseline_thresholds[name] = {k:summ['operating_points'][k]['classification']['threshold']
                                     for k in ['max_f1','recall_first']}
    common = set(dun).intersection(*(set(v) for v in baselines.values()))
    assert len(common)==331 and sum(dun[k]['label'] for k in common)==292
    assert len(set(dun)-common)==469 and not any(dun[k]['label'] for k in set(dun)-common)
    common_metrics, positive_turu = [], []
    for name, pred in baselines.items():
        rows = []
        for k in sorted(common):
            p, c = pred[k], dun[k]
            assert (p['fragment_a'],p['fragment_b'],bool(p['label'])) == (c['fragment_a'],c['fragment_b'],c['label'])
            if c['label']:
                assert math.dist(c['gt'],p['target_translation_rc']) < 1e-6
            pose = p['layouts'][f'{name}_native_translation_consensus']
            good = bool(c['label'] and pose['valid'] and pose['translation_rc'] is not None
                        and math.dist(c['gt'],pose['translation_rc']) <= 20)
            rows.append(dict(label=c['label'],score=p['pair_probability'],valid=p['decision_valid'],layout20=good))
        for policy, threshold in baseline_thresholds[name].items():
            m = summarize(rows, threshold, True)
            m.update(model=name, policy=policy)
            common_metrics.append(m)
        t = read(ROOT/f'reports/turufan_ood_pairwise_20260912_001/evaluation_v1/{name}_predictions.json')
        tids = {x['pair_id'] for x in t['predictions']}
        assert len(tids)==301 and tids=={k for k,c in turu.items() if c['label']}
        assert t['identity']['checkpoint_sha256_by_stage'] == baseline_identity[name]['checkpoint_sha256_by_stage']
        assert t['thresholds']==baseline_thresholds[name]
        for policy, threshold in t['thresholds'].items():
            tp = sum(p['decision_valid'] and p['score'] >= threshold for p in t['predictions'])
            assert tp == sum(p['accepted'][policy] for p in t['predictions'])
            positive_turu.append(dict(model=name,policy=policy,threshold=threshold,tp=tp,positive=301,recall=tp/301))
    for model, label in labels.items():
        m = summarize([modern[(model,'dunhuang_cv')][k] for k in sorted(common)], thresholds[model], True)
        m.update(model=label,policy='frozen SIM-CAL')
        common_metrics.append(m)
        rows = [x for x in modern[(model,'turufan')].values() if x['label']]
        m = summarize(rows,thresholds[model],False)
        positive_turu.append(dict(model=label,policy='frozen SIM-CAL',threshold=thresholds[model],
                                  tp=m['tp'],positive=301,recall=m['recall']))

    endpoint = read(HERE/'endpoint_completion_verified.json')
    assert endpoint['completion_verified']
    report = endpoint['files']['test_report.json']['data']
    endpoint_metrics = []
    for dataset, settings in report['populations'].items():
        for setting, policies in settings.items():
            rr = [r[setting] for r in endpoint['cases'] if r['dataset']==dataset]
            cohort = dun if dataset=='dunhuang_cv' else turu
            assert {r['pair_id'] for r in rr}=={k for k,c in cohort.items() if c['fold']==0}
            for r in rr:
                c = cohort[r['pair_id']]
                assert bool(r['label'])==c['label']
                if c['gt'] is not None:
                    assert math.dist(c['gt'],r['target_translation_rc'])<1e-6
                    assert r['layout20']==bool(r['numeric_valid'] and r['has_candidate']
                                              and r['translation'] is not None and math.dist(c['gt'],r['translation'])<=20)
            for policy, val in [('primary',policies['primary'])]+list(policies['frozen_cal_fpr'].items()):
                rows = [dict(label=r['label'],score=r['score'],valid=r['numeric_valid'] and r['has_candidate'],
                             layout20=r['layout20']) for r in rr]
                m = summarize(rows,val['threshold'],dataset=='dunhuang_cv')
                mp = dict(pairs='pairs',positives='positive',negatives='negative',tp='pair_tp',fp='pair_fp',
                          fn='pair_fn',tn='pair_tn',accuracy='accuracy',f1='pair_f1',auc='auc',
                          layout20_count='layout_correct',joint_tp='correct_and_accepted',
                          wrong_pose_accepted='wrong_pose_accepted',joint_f1='joint_f1')
                check_metrics(m,val,mp)
                m.update(dataset=dataset,setting=setting,policy=policy,candidate_coverage=val['candidate_coverage'])
                endpoint_metrics.append(m)
    out = dict(schema='b3-paper-evidence/1',no_new_inference=True,no_threshold_refit=True,
               common_dunhuang=dict(count=331,positive=292,negative=39,ids=sorted(common),
                 excluded_bad_gt=sorted(bad),new_negative_pairs_without_baseline_predictions=sorted(set(dun)-common)),
               baseline_training=baseline_identity,baseline_thresholds=baseline_thresholds,
               common_dunhuang_metrics=common_metrics,turufan_positive301_metrics=positive_turu,
               historical_full_metrics=full_metrics,historical_test_metrics=test_metrics,
               endpoint_test_metrics=endpoint_metrics,sources=SOURCES)
    (HERE/'evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k.endswith('metrics')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

"""Full-population display and exact saved-readout interventions, without inference."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

from .bind_light_results import LABELS, METRICS_QUERY, REPORT_ID

FULL = {'dunhuang_cv': 'gt_corrected_800_development_context',
        'turufan': 'all_development_context'}
COUNTS = {'dunhuang_cv': (800, 292, 508), 'turufan': (602, 301, 301)}


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def candidate_scores(c, weights):
    """No-conflict does NOT reassign conflict probability to support."""
    if weights is None:
        assert c.get('local_classification_present') is False
        return dict(actual=c['score'], no_conflict=c['score']), 0.
    p, n, m = [c[k] for k in ('positive_evidence_px', 'conflict_evidence_px', 'observed_mass_length_px')]
    o = c['overlap']['fraction_min_area']
    assert o is not None and min(p, n, m) >= 0 and p+n <= m+2e-4
    w = weights; length = w['length_scale_px']
    support = w['positive'] * math.log1p(p / length)
    conflict = w['conflict'] * math.log1p(n / length)
    overlap = w['overlap'] * o
    actual = w['bias'] + support - conflict - overlap
    assert abs(actual - c['logit']) < 3e-6
    assert abs(sigmoid(actual) - c['score']) < 2e-6
    return dict(actual=c['score'], no_conflict=sigmoid(actual+conflict),
                all_support=sigmoid(w['bias']+w['positive']*math.log1p(m/length)-overlap)), abs(actual-c['logit'])


def decide(row, intervention, threshold):
    candidates = row['candidates']
    winner = max(candidates, key=lambda c: c['scores'][intervention]) if candidates else None
    valid = bool(row['numeric_valid'] and winner is not None)
    score = winner['scores'][intervention] if winner else 0.
    accepted = valid and score >= threshold
    correct = bool(valid and winner['gt_error_px'] is not None and winner['gt_error_px'] <= 20)
    return winner, accepted, correct


def tally(rows, intervention, threshold, layout):
    out = dict(pairs=len(rows), positives=sum(bool(r['label']) for r in rows),
               tp=0, fp=0, tn=0, fn=0, layout20_count=0, joint_tp=0,
               winner_correct_but_rejected=0, wrong_pose_accepted=0, changed_winners=0)
    out['negatives'] = out['pairs'] - out['positives']
    for r in rows:
        winner, accepted, correct = decide(r, intervention, threshold)
        positive = bool(r['label'])
        out['tp' if positive and accepted else 'fn' if positive else 'fp' if accepted else 'tn'] += 1
        original = next((c for c in r['candidates'] if c['selected']), None)
        out['changed_winners'] += int((winner['cluster_id'] if winner else None) != (original['cluster_id'] if original else None))
        if positive and layout:
            out['layout20_count'] += int(correct)
            out['joint_tp'] += int(correct and accepted)
            out['winner_correct_but_rejected'] += int(correct and not accepted)
            out['wrong_pose_accepted'] += int(not correct and accepted)
    out['accuracy'] = (out['tp']+out['tn'])/len(rows)
    out['f1'] = 2*out['tp']/(2*out['tp']+out['fp']+out['fn'])
    out['joint_f1'] = (2*out['joint_tp']/(out['tp']+out['fp']+out['positives'])) if layout else None
    if not layout:
        for k in ('layout20_count','joint_tp','winner_correct_but_rejected','wrong_pose_accepted'): out[k] = None
    return out


def build(bundle_path, weights_path, snapshot, expected_scope_count=16):
    bundle, weights = read(bundle_path), read(weights_path)['models']
    assert bundle['neural_inference_repeated'] is False and bundle['thresholds_refitted'] is False
    rows = bundle['rows']
    excluded = {r['pair_id'] for r in snapshot['queries']['gt_exclusions']['rows']}
    cases = {r['pair_id']: r for r in snapshot['queries']['cases']['rows']}
    inputs = {str(bundle_path): sha(bundle_path), str(weights_path): sha(weights_path)}
    for p, h in bundle['input_sha256'].items(): assert sha(p) == h
    interventions, gallery, checks = [], [], []
    full_rows = [dict(r,threshold=r['metrics']['threshold']) for r in rows
                 if r['split'] in FULL and r['population'] == FULL[r['split']] and r['policy'] == 'primary']
    assert len(full_rows) == expected_scope_count
    for scope in full_rows:
        model, selection, split = [scope[k] for k in ('model','selection_kind','split')]
        root = Path(scope['source']['job'])
        for name in ('protocol.json','prediction_complete.json','case_diagnostics.jsonl','pair_predictions.jsonl'):
            inputs[str(root/name)] = sha(root/name)
        protocol, complete = read(root/'protocol.json'), read(root/'prediction_complete.json')
        assert complete['status'] == 'all_predictions_frozen' and complete['model_state_unchanged'] is True
        assert inputs[str(root/'pair_predictions.jsonl')] == complete['sha256'] == scope['source']['predictions_sha256']
        assert protocol['checkpoint_sha256'] == scope['checkpoint_sha256']
        if 'diagnostics_sha256' in scope['source']:
            assert inputs[str(root/'case_diagnostics.jsonl')] == scope['source']['diagnostics_sha256']
        raw = [json.loads(line) for line in (root/'case_diagnostics.jsonl').read_text().splitlines()]
        assert len(raw) == len({r['pair_id'] for r in raw}) == complete['pairs']
        actual_predictions = {r['pair_id']: r for r in (json.loads(l) for l in (root/'pair_predictions.jsonl').read_text().splitlines())}
        w = weights.get(model)
        if w:
            assert w['checkpoint_sha256'] == scope['checkpoint_sha256'] and w['epoch'] == scope['selected_epoch']
        kept, max_error = [], 0.
        for original in raw:
            pid = original['pair_id']
            prediction = actual_predictions[pid]
            assert prediction['score'] == original['score'] and prediction['translation'] == original['translation']
            if split == 'dunhuang_cv' and pid in excluded: continue
            r = copy.deepcopy(original)
            assert r['candidate_count'] == len(r['candidates']) == len(r['candidate_errors_px'])
            for i, c in enumerate(r['candidates']):
                c['scores'], error = candidate_scores(c, w['readout_weights'] if w else None)
                max_error = max(max_error, error)
                c['gt_error_px'] = r['candidate_errors_px'][i] if r['gt_known'] else None
                if r['gt_known']:
                    assert abs(math.dist(c['refined_translation'], r['target_translation_rc'])-c['gt_error_px']) < 1e-4
            winner, accepted, correct = decide(r, 'actual', scope['threshold'])
            assert accepted == r['accepted'] and (winner is not None) == bool(r['has_candidate'])
            if winner:
                assert winner['cluster_id'] == r['selected_cluster_id'] and winner['score'] == r['score']
            kept.append(r)
            if not r['label']: continue
            c = cases[pid]
            assert c['label'] and c['fold'] == r['fold']
            if r['gt_known']: assert math.dist(c['gt'], r['target_translation_rc']) < 1e-4
            gallery.append(dict(case_key='|'.join((model,selection,split,pid)), model=model,
                model_label=LABELS[model], selection_kind=selection, split=split, pair_id=pid,
                case_name=c['case_name'], fragment_a=c['fragment_a'],fragment_b=c['fragment_b'],
                gt=c['gt'] if split=='dunhuang_cv' else None, fold=r['fold'],
                source_role={0:'real_test',1:'real_cal',2:'real_select',3:'real_select',4:'real_select'}[r['fold']],
                translation=r['translation'] if winner else None,score=r['score'],threshold=scope['threshold'],
                selected_epoch=scope['selected_epoch'],checkpoint_sha256=scope['checkpoint_sha256'],
                has_candidate=bool(r['has_candidate']),numeric_valid=bool(r['numeric_valid']),
                accepted=accepted,layout20=correct if split=='dunhuang_cv' else None,
                error_px=r['error_px'] if split=='dunhuang_cv' else None,
                candidate_coverage=any(c['gt_error_px'] is not None and c['gt_error_px']<=20 for c in r['candidates']) if split=='dunhuang_cv' else None,
                selected_cluster_id=r['selected_cluster_id'],
                candidate_json=json.dumps(r['candidates'],ensure_ascii=False,separators=(',',':'),allow_nan=False)))
        assert (len(kept),sum(r['label'] for r in kept),sum(not r['label'] for r in kept)) == COUNTS[split]
        for policy, threshold in [('primary',scope['threshold']),('fixed03',.3)]:
            baseline = next(x['metrics'] for x in rows if x['model']==model and x['selection_kind']==selection and x['split']==split and x['population']==FULL[split] and x['policy']==policy)
            for mode in (['actual','no_conflict','all_support'] if w else ['actual','no_conflict']):
                stats=tally(kept,mode,threshold,split=='dunhuang_cv')
                if mode=='actual':
                    for key,value in stats.items():
                        if key in baseline:
                            assert (value is None and baseline[key] is None) or (value is not None and baseline[key] is not None and abs(value-baseline[key])<1e-8), (model,key,value,baseline.get(key))
                    assert stats['changed_winners']==0
                interventions.append(dict(model=model,model_label=LABELS[model],selection_kind=selection,
                    split=split,population=FULL[split],policy=policy,threshold=threshold,
                    selected_epoch=scope['selected_epoch'],intervention=mode,
                    explicit_conflict_present=bool(w),checkpoint_sha256=scope['checkpoint_sha256'],**stats))
        checks.append(dict(model=model,selection_kind=selection,split=split,pairs=len(kept),
                           candidates=sum(len(r['candidates']) for r in kept),max_reconstruction_error=max_error))
    assert all(sha(p)==h for p,h in inputs.items())
    return dict(schema='full-scorer-review/1',status='complete',all_positive_cases=gallery,
        interventions=interventions,checks=checks,input_sha256=inputs,
        inference_repeated=False,threshold_refitted=False,training_modified=False,
        note='Analytic readout interventions preserve all Q/candidates/poses; not removal/retraining of the local classifier. Full real cohorts are developmental, not blind TEST.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True)
    args=p.parse_args();assert not args.out.exists()
    target=args.project/'src/data.json';snapshot=read(target);assert snapshot['id']==REPORT_ID
    before=sha(target);annotation_sha=sha(args.annotations)
    analysis=build(args.bundle,args.weights,snapshot)
    args.out.mkdir(parents=True)
    (args.out/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,allow_nan=False)+'\n')
    shutil.copy2(target,args.out/'report_before.json')
    updated=copy.deepcopy(snapshot)
    existing=updated['queries'][METRICS_QUERY]['rows']
    def identity(r): return tuple(r[k] for k in ('model','selection_kind','split','population','policy'))
    old_keys={identity(r) for r in existing}
    for r in read(args.bundle)['rows']:
        if r['population'] in ('all_development_context','gt_corrected_800_development_context'):
            assert identity(r) not in old_keys
            existing.append(dict(r,model_label=LABELS[r['model']],**r['metrics']))
    source=dict(label='完整冻结预测、实际权重标量与逐候选读出复算',
        files=[str((args.out/'analysis.json').resolve())],sha256=sha(args.out/'analysis.json'),
        caveats=[analysis['note'],'Full Dun800 excludes three previously confirmed incorrect GT records; Turufan has no Layout GT.'],
        metricDefinitions=[dict(label='移除冲突扣分',definition='仅令显式冲突惩罚项为0；不把冲突概率重新分配给支持；固定原Q/候选/精修位姿并重新选分数最大簇。',componentIds=['light-full-ablation']),
                           dict(label='所有局部支持',definition='仅复杂头令正证据P=M、冲突C=0；保留重叠项。不是重新训练，也不是全局Layout理论上限。',componentIds=['light-full-ablation']),
                           dict(label='失败全集',definition='每个模型/选模方式全部292敦煌正例可筛摆错或摆对拒绝；Turufan301正例只展示分类拒绝，不推断Layout对错。',componentIds=['light-full-failures'])])
    for qid,rows in [('light_scorer_all_positive_cases',analysis['all_positive_cases']),('light_scorer_readout_interventions',analysis['interventions'])]:
        assert qid not in updated['queries'];updated['queries'][qid]=dict(rows=rows,source=source)
    updated['queries']['light_scorer_all_positive_cases']['payloadColumns']=['candidate_json']
    updated['queries'][METRICS_QUERY]['source']['fullPopulationExtension']=source
    updated['generatedAt']=datetime.now(timezone.utc).isoformat();updated['buildStatus']='updating'
    assert sha(target)==before and sha(args.annotations)==annotation_sha
    for q in snapshot['queries']:
        if q!=METRICS_QUERY: assert snapshot['queries'][q]==updated['queries'][q]
    assert snapshot['queries'][METRICS_QUERY]['rows']==existing[:len(snapshot['queries'][METRICS_QUERY]['rows'])]
    target.write_text(json.dumps(updated,ensure_ascii=False,allow_nan=False)+'\n')
    receipt=dict(status='bound_pending_build',old_snapshot_sha256=before,snapshot_sha256=sha(target),
                 annotations_sha256=annotation_sha,counts={k:len(updated['queries'][k]['rows']) for k in (METRICS_QUERY,'light_scorer_all_positive_cases','light_scorer_readout_interventions')})
    (args.out/'binding.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__': main()

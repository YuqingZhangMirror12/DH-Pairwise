"""Frozen-model comparisons and source-disjoint real threshold cross-fitting.

No checkpoint/model selection or score transformation. The existing 0.20–0.80
grid and five source-grouped folds are reused unchanged from September 21.
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.bounded_real_calibration_v2.common import (
    read, save, rows, sha, metrics, crossfit, GRID,
)

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
FIXED_EIGHT = (
    "b6909377cdcb1b5a2ecbe7e371ae3d1b1a234cbdc0f6a8ffbaf8c677c72a16fb",
    "4a042cd1d230a3d13a435fbfe9954b905d3aaebb0e9374f14810b946fc5176b6",
    "7bef64454389af468ea75d1f29e6ccdd93ad5eae57c22fc2bce74d145e28d956",
    "6b0e600e633b6476eb5a98fb2f7c75ced69e08c17dcf9d4c6c1e8c6f2f1ae620",
    "1be7314690e4cfe02f96b1f98aeee13a86ac0c42acd6d2bbf8728f33fd7565a9",
    "99e4a457c7649de8e0d98bac24eb51b305b1fa8e2220c0a5fca4455e011e80fb",
    "66faf2192d044095a567449ac9201b0de4a7ccf59a301a74f2a5012865783ec1",
    "8ac539b5c6b9f609f636b00e9713ca233a22ba8e4f3b096d1fb4febfa1be7a95",
)


def compact(r):
    return dict(pair_id=r['pair_id'], score=r['score'],
        decision_valid=bool(r['numeric_valid'] and r['has_candidate']),
        layout_error_px=r['error_px'], layout_good_20=r['layout20'] if r['gt_known'] else None)


def summarize(labels, predictions, threshold, real):
    accepted = [r['decision_valid'] and r['score'] >= threshold for r in predictions]
    m = metrics(labels, accepted, [r['score'] for r in predictions],
                [r['layout_good_20'] for r in predictions] if real else None)
    return dict(threshold=threshold, **m)


def quantiles(values):
    values = [x for x in values if x is not None and np.isfinite(x)]
    return dict(n=len(values), quantiles=np.quantile(values, [0,.1,.25,.5,.75,.9,1]).tolist()) if values else dict(n=0, quantiles=None)


def run(root):
    root = Path(root)
    result = dict(protocol=dict(threshold_grid=list(GRID), source_disjoint_five_fold=True,
        score_rescaling=False, model_selection_on_real=False, prior_real_design_exposure=True,
        negative_labels='existing source-based constructed negatives; not all human verified'), splits={})
    registry = read(DIAG/'bounded_real_calibration_v2/results/registry.json')['old_matched_tokens']
    baseline_threshold = registry['provenance']['real']['model']['operating_points']['thresholds']['max_f1']
    selection = read(root/'selection.json')
    probe = []
    all_real = rows(root/'dunhuang/case_diagnostics.jsonl')
    by_real = {r['pair_id']:r for r in all_real}
    for suffix in FIXED_EIGHT:
        pid = 'pair/sha256/'+suffix
        assert pid in by_real
        probe.append(dict(split='dunhuang', pair_id=pid, reason='fixed historical eight (selected by S6 strata, not necessarily S7 failures)'))
    oof_bundle = {}
    for name, split, baseline_name in [('dunhuang_cv','real','matched_tokens_real_cv'),('turufan','ood','matched_tokens_turufan')]:
        meta = read(DIAG/f'real_domain_calibration_v1/results/{split}/manifest.json')
        meta['split'] = split
        detailed = rows(root/name/'case_diagnostics.jsonl')
        old = read(root/'baseline'/f'{baseline_name}.json')
        assert old['checkpoint_sha256'] == registry['checkpoint_sha256']
        predictions = {'v3_B22':[compact(r) for r in detailed], 'S7_M12_matched_C16':old['rows']}
        labels = [p['label'] for p in meta['pairs']]
        entry = dict(n=len(labels), positives=sum(labels), negatives=len(labels)-sum(labels), models={})
        oof_bundle[name] = {}
        for key, pred in predictions.items():
            assert [r['pair_id'] for r in pred] == [r['pair_id'] for r in meta['pairs']]
            assert len(set(r['pair_id'] for r in pred)) == len(pred)
            t = selection['threshold'] if key=='v3_B22' else baseline_threshold
            policies={}
            for policy in ('bounded_max_f1','bounded_recall95'):
                policies[policy], heldout = crossfit(meta, pred, policy)
                oof_bundle[name].setdefault(key,{})[policy] = heldout
            entry['models'][key]=dict(sim_frozen=summarize(labels,pred,t,split=='real'),
                fixed03=summarize(labels,pred,.3,split=='real'), cv=policies)
        a,b=predictions['v3_B22'],predictions['S7_M12_matched_C16']
        if split=='real':
            positive=[i for i,y in enumerate(labels) if y]
            layout=Counter(('new_good' if a[i]['layout_good_20'] else 'new_bad')+'__'+
                           ('old_good' if b[i]['layout_good_20'] else 'old_bad') for i in positive)
            entry['paired_layout']=dict(layout)
        for policy in ('bounded_max_f1','bounded_recall95'):
            x={r['pair_id']:r for r in oof_bundle[name]['v3_B22'][policy]}
            y={r['pair_id']:r for r in oof_bundle[name]['S7_M12_matched_C16'][policy]}
            entry.setdefault('paired_classification',{})[policy]=dict(Counter(
                ('new_correct' if x[p['pair_id']]['accepted']==bool(p['label']) else 'new_wrong')+'__'+
                ('old_correct' if y[p['pair_id']]['accepted']==bool(p['label']) else 'old_wrong') for p in meta['pairs']))
        conditions = {
            'layout_gain':lambda i:labels[i] and a[i]['layout_good_20'] and not b[i]['layout_good_20'],
            'layout_regression':lambda i:labels[i] and b[i]['layout_good_20'] and not a[i]['layout_good_20'],
            'correct_layout_rejected':lambda i:labels[i] and a[i]['layout_good_20'] and a[i]['score']<selection['threshold'],
            'correct_candidate_wrong_winner':lambda i:labels[i] and detailed[i]['refined_coverage8'] and not a[i]['layout_good_20'],
        } if split=='real' else {
            'old_accepted_new_rejected':lambda i:labels[i] and b[i]['score']>=baseline_threshold and a[i]['score']<selection['threshold'],
            'new_accepted_old_rejected':lambda i:labels[i] and a[i]['score']>=selection['threshold'] and b[i]['score']<baseline_threshold,
            'negative_old_false_positive':lambda i:not labels[i] and b[i]['score']>=baseline_threshold,
            'positive_both_rejected':lambda i:labels[i] and b[i]['score']<baseline_threshold and a[i]['score']<selection['threshold'],
        }
        for reason, condition in conditions.items():
            eligible=sorted([i for i in range(len(a)) if condition(i)],key=lambda i:a[i]['pair_id'])
            # First IDs within an explicitly outcome-stratified diagnostic sample.
            chosen=[i for i in eligible if not any(p['pair_id']==a[i]['pair_id'] for p in probe)][:4]
            probe.extend(dict(split=name, pair_id=a[i]['pair_id'], reason=reason) for i in chosen)
        buckets={
            'positive_accepted':[r for r in detailed if r['label'] and r['score']>=selection['threshold']],
            'positive_rejected':[r for r in detailed if r['label'] and r['score']<selection['threshold']],
            'negative_accepted':[r for r in detailed if not r['label'] and r['score']>=selection['threshold']],
            'negative_rejected':[r for r in detailed if not r['label'] and r['score']<selection['threshold']],
        }
        if split=='real':buckets['correct_layout_rejected']=[r for r in detailed if r['label'] and r['layout20'] and r['score']<selection['threshold']]
        diagnostics={}
        for bucket, rr in buckets.items():
            cc=[r['candidates'][r['winner_index']] for r in rr if r['has_candidate']]
            diagnostics[bucket]=dict(count=len(rr),candidate_count=len(cc),scores=quantiles([r['score'] for r in rr]),
                quality_logit=quantiles([c['quality_logit'] for c in cc]),null_logit=quantiles([c['null_logit'] for c in cc]),
                residual=quantiles([c['residual_median_px'] for c in cc]),
                evidence=[quantiles([c['evidence'][j] for c in cc]) for j in range(8)])
        entry['network_output_diagnostics']=diagnostics
        result['splits'][name]=entry
    save(root/'comparison.json',result)
    save(root/'oof_predictions.json',oof_bundle)
    save(root/'probe_cases.json',dict(selection='historical8 plus outcome-stratified ID-ordered sample; diagnostic only',cases=probe))
    for split,d in result['splits'].items():
        for model,m in d['models'].items():
            cv=m['cv']['bounded_max_f1']
            print(split,model,'SIM',m['sim_frozen'],'REAL_CV',cv['pooled_out_of_fold'],
                  'thresholds',[f['threshold'] for f in cv['folds']])
        print('paired layout',d.get('paired_layout'))
    print('probe_count',len(probe))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)

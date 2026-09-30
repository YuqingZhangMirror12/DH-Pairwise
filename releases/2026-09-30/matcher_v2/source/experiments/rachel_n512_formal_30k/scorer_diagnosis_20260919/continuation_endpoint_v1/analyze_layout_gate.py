"""Paired C8/C16 classification changes conditional on fixed REAL layout.

Read saved outputs only. Never fit thresholds or select checkpoints on REAL.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def accepted(row, threshold):
    return bool(row['decision_valid']) and row['classification']['fused'] >= threshold


def main():
    raw = ROOT / 'raw'
    a = raw/'attention_depth_20260915/s4_cross_attention_depth2/evaluation/fixed_epoch/real'
    b = raw/'scorer_diagnosis_20260919/continuation_v1/s6_d2_c16/evaluation/fixed_epoch/real'
    old, new = read_rows(a/'pair_results.jsonl'), read_rows(b/'pair_results.jsonl')
    assert [(r['pair_id'],r['label'],r['review_status']) for r in old] == [(r['pair_id'],r['label'],r['review_status']) for r in new]
    assert all(x['layouts'] == y['layouts'] and x['target_translation_rc'] == y['target_translation_rc'] for x,y in zip(old,new))
    thresholds = [json.loads((p/'summary.json').read_text())['groups']['kept_plus_all_negative']['classification']['fused']['max_f1']['threshold'] for p in (a,b)]
    kept = [(x,y) for x,y in zip(old,new) if x['label'] and x['review_status']=='keep']
    assert len(kept)==295
    keptgood, keptbad = [], []
    for x,y in kept:
        pose = x['layouts']['full_top2_mode']
        good = pose['valid'] and pose['translation_l2_px'] is not None and pose['translation_l2_px'] <= 20.
        (keptgood if good else keptbad).append((x,y))
    groups = dict(kept_positive=kept, kept_layout_good=keptgood, kept_layout_bad=keptbad,
        all_negative=[(x,y) for x,y in zip(old,new) if not x['label']])
    result = dict(source='same fixed-epoch C8 versus C16, frozen shared Matcher',
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a/'pair_results.jsonl',b/'pair_results.jsonl')},
        GT_layout_gate_px=20, same_pair_ids_labels_review=True, layouts_exactly_equal=True,
        old_simval_threshold=thresholds[0],new_simval_threshold=thresholds[1],
        real_threshold_fitting=False,groups={})
    for name,pairs in groups.items():
        scenarios={}
        for scenario,ta,tb in [('same_old_threshold',thresholds[0],thresholds[0]),('own_simval_threshold',*thresholds)]:
            transitions={k:[] for k in ('accept_accept','accept_reject','reject_accept','reject_reject')}
            for x,y in pairs:
                ka='accept' if accepted(x,ta) else 'reject'
                kb='accept' if accepted(y,tb) else 'reject'
                transitions[ka+'_'+kb].append(x['pair_id'])
            scenarios[scenario]=dict(threshold_c8=ta,threshold_c16=tb,
                accepted_c8=sum(accepted(x,ta) for x,y in pairs),accepted_c16=sum(accepted(y,tb) for x,y in pairs),
                transition_counts={k:len(v) for k,v in transitions.items()},transition_pair_ids=transitions)
        result['groups'][name]=dict(count=len(pairs),scenarios=scenarios)
    path=ROOT/'layout_gate_analysis.json'
    with path.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    for name,group in result['groups'].items():
        print(name,group['count'],{k:{j:v for j,v in value.items() if j!='transition_pair_ids'} for k,value in group['scenarios'].items()})


if __name__=='__main__':
    main()

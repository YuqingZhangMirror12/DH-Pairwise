"""Deterministic saved-output examples, not visual selection or new inference."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def rows(path):
    values=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    index={r['pair_id']:r for r in values}
    assert len(index)==len(values)
    return index

old=rows(ROOT/'s7_direct/matched_tokens/evaluation/c16/real/pair_results.jsonl')
new=rows(ROOT/'s7_direct/matched_edges/evaluation/c16/real/pair_results.jsonl')
support=rows(ROOT/'s7_matched_support_strata_v2/cases.jsonl')
margin=json.loads((HERE/'margin_vs_matched_tokens.json').read_text())
ids=margin['populations']['real_layout20_good']['working_points']['max_f1']['classes']['lost_accept']['pair_ids']
assert len(ids)==14
examples=[]
for key in ids:
    a,b,s=old[key],new[key],support[key]
    assert a['layouts']==b['layouts'] and a['label'] and a['review_status']=='keep'
    layout=a['layouts']['full_top2_mode']
    assert layout['valid'] and layout['translation_l2_px']<=20
    examples.append(dict(pair_id=key,fragment_a=a['fragment_a'],fragment_b=a['fragment_b'],
        old_score=a['classification']['fused'],new_score=b['classification']['fused'],
        logit_change=b['candidate_details']['deployed_logit']-a['candidate_details']['deployed_logit'],
        unchanged_translation_rc=layout['translation_rc'],gt_error_px=layout['translation_l2_px'],
        **{k:s[k] for k in ('min_endpoints','max_endpoints','inlier_edges','residual_px','area_ratio')}))
examples.sort(key=lambda r:(r['logit_change'],r['pair_id']))
result=dict(status='complete',selection='all14 newly rejected Layout20-correct REAL pairs; sorted by logit decrease',
    thresholds=dict(matched_tokens=.8344629406929016,matched_edges=.9202651381492615),
    caveat='descriptive examples, not a causal or representative sample',examples=examples)
(HERE/'newly_rejected_examples.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(examples[:3],ensure_ascii=False,indent=2))

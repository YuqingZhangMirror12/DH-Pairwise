"""Add completed F/I results to the unchanged B22/legacy real cross-fit table.

This selects thresholds only, never training epochs or model weights. The
historical result directory is read-only; outputs go to the new evaluation root.
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .analyze_external import compact,summarize,DIAG
from ..bounded_real_calibration_v2.common import read,save,rows,sha,crossfit

# File identities only; analysis deliberately has no PyTorch/GPU dependency.
ARMS=('frozen_features','independent_features')


def reference_tables(b22,s7_hard=None):
    result=read(b22/'comparison.json');oof=read(b22/'oof_predictions.json')
    if s7_hard is None:return result,oof
    enriched=read(s7_hard/'comparison.json');enriched_oof=read(s7_hard/'oof_predictions.json')
    if enriched.get('s7_hard_extension',{}).get('prior_sha256')!=sha(b22/'comparison.json'):
        raise ValueError('S7-H was not compared with this exact B22 reference')
    for name in ('dunhuang_cv','turufan'):
        for model,metrics in result['splits'][name]['models'].items():
            if enriched['splits'][name]['models'].get(model)!=metrics:
                raise ValueError('S7-H extension changed a historical reference')
        for model,predictions in oof[name].items():
            if enriched_oof[name].get(model)!=predictions:
                raise ValueError('S7-H extension changed historical out-of-fold predictions')
        if 'S7_H' not in enriched['splits'][name]['models'] or 'S7_H' not in enriched_oof[name]:
            raise ValueError('S7-H reference missing')
    return enriched,enriched_oof


def paired(meta,left,right,left_layout,right_layout):
    a={r['pair_id']:r for r in left};b={r['pair_id']:r for r in right}
    result=dict(classification=dict(Counter(
        ('left_correct' if a[p['pair_id']]['accepted']==bool(p['label']) else 'left_wrong')+'__'+
        ('right_correct' if b[p['pair_id']]['accepted']==bool(p['label']) else 'right_wrong') for p in meta['pairs'])))
    if meta['split']=='real':
        pos=[p for p in meta['pairs'] if p['label']]
        result['correct_layout_accepted']=dict(Counter(
            ('left_pass' if a[p['pair_id']]['accepted'] and left_layout[p['pair_id']] else 'left_fail')+'__'+
            ('right_pass' if b[p['pair_id']]['accepted'] and right_layout[p['pair_id']] else 'right_fail') for p in pos))
    return result


def proposal_differences(left,right):
    if len(left)!=len(right):raise ValueError('different population length')
    count_mismatch=0;max_delta=0.;mass_delta=0.
    for a,b in zip(left,right):
        if a['pair_id']!=b['pair_id']:raise ValueError('different population order')
        aa,bb=a['candidates'],b['candidates'];mass_delta=max(mass_delta,abs(a['q1_mass']-b['q1_mass']))
        if len(aa)!=len(bb):count_mismatch+=1;continue
        for ca,cb in zip(aa,bb):
            max_delta=max(max_delta,float(np.linalg.norm(np.array(ca['proposal_translation_rc'])-cb['proposal_translation_rc'])))
    return dict(pairs=len(left),candidate_count_mismatches=count_mismatch,
        max_aligned_proposal_translation_difference_px=max_delta,max_q1_total_mass_difference=mass_delta,
        note='Compare target-blind Matcher proposals, not trainable Verifier refinements or ranking.')


def run(a):
    root,b22=Path(a.root),Path(a.b22)
    s7_hard=Path(a.s7_hard) if getattr(a,'s7_hard',None) else None
    result,oof=reference_tables(b22,s7_hard)
    result['independent_feature_extension']=dict(source_b22=str(b22),original_comparison_sha256=sha(b22/'comparison.json'),
        same_source_folds=True,checkpoint_selection='simulation CAL/SELECT only',score_rescaling=False,
        endpoint_model_is_legacy_reference=True,training_replicates=1,
        s7_hard_reference=str(s7_hard) if s7_hard else None)
    for name,split in [('dunhuang_cv','real'),('turufan','ood')]:
        meta=read(DIAG/f'real_domain_calibration_v1/results/{split}/manifest.json');meta['split']=split
        ids=[p['pair_id'] for p in meta['pairs']];labels=[p['label'] for p in meta['pairs']]
        parent_protocol=read(b22/name/'protocol.json');detail={'v3_B22':rows(b22/name/'case_diagnostics.jsonl')}
        if s7_hard:
            p=s7_hard/name;protocol=read(p/'protocol.json')
            if protocol['status']!='complete' or read(p/'status.json')['status']!='complete':
                raise ValueError('S7-H evaluation incomplete')
            if protocol['source']!=parent_protocol['source'] or protocol['real_used_to_select_checkpoint'] or protocol['threshold_fitting']:
                raise ValueError('different S7-H evaluation protocol')
            rr=rows(p/'case_diagnostics.jsonl')
            if [r['pair_id'] for r in rr]!=ids:raise ValueError('different S7-H population')
            if protocol['checkpoint_sha256']!=result['splits'][name]['models']['S7_H']['checkpoint_sha256']:
                raise ValueError('S7-H summary belongs to different weights')
            detail['S7_H']=[dict(r,layout20=r['layout_good_20']) for r in rr]
        for arm in ARMS:
            p=root/arm/name;protocol=read(p/'protocol.json')
            if protocol['status']!='complete' or read(p/'status.json')['status']!='complete':raise ValueError('evaluation incomplete')
            if protocol['arm']!=arm or protocol['real_used_to_select_checkpoint'] or protocol['threshold_fitting']:
                raise ValueError('different evaluation protocol')
            if protocol['base_checkpoint_sha256']!=parent_protocol['checkpoint_sha256'] or protocol['source']!=parent_protocol['source']:
                raise ValueError('different base or input cache')
            rr=rows(p/'case_diagnostics.jsonl');detail[arm]=rr
            if [r['pair_id'] for r in rr]!=ids or len(set(ids))!=len(ids):raise ValueError('different/missing/duplicate pairs')
            pred=[compact(r) for r in rr];policies={};oof[name][arm]={}
            for policy in ('bounded_max_f1','bounded_recall95'):
                policies[policy],oof[name][arm][policy]=crossfit(meta,pred,policy)
            result['splits'][name]['models'][arm]=dict(selected_epoch=protocol['selected_epoch'],checkpoint_sha256=protocol['checkpoint_sha256'],
                sim_frozen=summarize(labels,pred,protocol['threshold'],split=='real'),fixed03=summarize(labels,pred,.3,split=='real'),cv=policies)
        result['splits'][name]['independent_feature_proposals']=proposal_differences(detail[ARMS[0]],detail[ARMS[1]])
        result['splits'][name]['independent_feature_paired']={}
        comparisons=[(ARMS[1],ARMS[0]),(ARMS[1],'v3_B22'),(ARMS[0],'v3_B22')]
        if s7_hard:comparisons.extend((arm,'S7_H') for arm in ARMS)
        for left,right in comparisons:
            result['splits'][name]['independent_feature_paired'][left+'_vs_'+right]=paired(meta,oof[name][left]['bounded_max_f1'],oof[name][right]['bounded_max_f1'],
                {r['pair_id']:r['layout20'] for r in detail[left]}, {r['pair_id']:r['layout20'] for r in detail[right]})
    save(root/'comparison.json',result);save(root/'oof_predictions.json',oof)
    for split,d in result['splits'].items():
        for arm in ARMS:
            m=d['models'][arm];print(split,arm,'epoch',m['selected_epoch'],'OOF',m['cv']['bounded_max_f1']['pooled_out_of_fold'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--b22',required=True)
    p.add_argument('--s7-hard',help='Completed S7-H evaluation root to retain as an additional reference')
    run(p.parse_args())

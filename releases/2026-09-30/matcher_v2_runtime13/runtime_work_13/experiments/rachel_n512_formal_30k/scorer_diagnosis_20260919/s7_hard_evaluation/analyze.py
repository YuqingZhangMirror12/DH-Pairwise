"""Append S7-H frozen predictions to existing source-grouped cross-fit results."""
import argparse
from collections import Counter
from pathlib import Path
from ..bounded_real_calibration_v2.common import read,save,rows,sha,crossfit
from ..seam_context_v3.analyze_external import summarize,DIAG


def run(a):
    root,prior=Path(a.root),Path(a.prior)
    result=read(prior/'comparison.json');oof=read(prior/'oof_predictions.json')
    result['s7_hard_extension']=dict(prior=str(prior),prior_sha256=sha(prior/'comparison.json'),
        checkpoint_selection='simulation only',real_threshold_crossfit=True,model_key='S7_H',
        training_replicates=1,data='7832 effective S7 hard pairs; mirrored; existing sources not new data')
    for name,split,baseline in (('dunhuang_cv','real','matched_tokens_real_cv'),('turufan','ood','matched_tokens_turufan')):
        meta=read(DIAG/f'real_domain_calibration_v1/results/{split}/manifest.json');meta['split']=split
        p=root/name;protocol=read(p/'protocol.json')
        if protocol['status']!='complete' or read(p/'status.json')['status']!='complete':raise ValueError('incomplete evaluation')
        if protocol['real_used_to_select_checkpoint'] or protocol['threshold_fitting']:raise ValueError('changed selection protocol')
        if protocol['source']!=read(prior/name/'protocol.json')['source']:raise ValueError('different real inputs')
        pred=rows(p/'case_diagnostics.jsonl');ids=[r['pair_id'] for r in pred]
        if ids!=[r['pair_id'] for r in meta['pairs']] or len(set(ids))!=len(ids):raise ValueError('different population')
        labels=[p['label'] for p in meta['pairs']];policies={};oof[name]['S7_H']={}
        for policy in ('bounded_max_f1','bounded_recall95'):
            policies[policy],oof[name]['S7_H'][policy]=crossfit(meta,pred,policy)
        result['splits'][name]['models']['S7_H']=dict(
            selected_matcher_epoch=protocol['selected_matcher_epoch'],selected_scorer_epoch=protocol['selected_scorer_epoch'],
            checkpoint_sha256=protocol['checkpoint_sha256'],sim_frozen=summarize(labels,pred,protocol['threshold'],split=='real'),
            fixed03=summarize(labels,pred,.3,split=='real'),cv=policies)
        new={r['pair_id']:r for r in oof[name]['S7_H']['bounded_max_f1']}
        old={r['pair_id']:r for r in oof[name]['S7_M12_matched_C16']['bounded_max_f1']}
        paired=dict(classification=dict(Counter(
            ('new_correct' if new[p['pair_id']]['accepted']==bool(p['label']) else 'new_wrong')+'__'+
            ('old_correct' if old[p['pair_id']]['accepted']==bool(p['label']) else 'old_wrong') for p in meta['pairs'])))
        if split=='real':
            base=read(prior/'baseline'/f'{baseline}.json')['rows']
            if [r['pair_id'] for r in base]!=ids:raise ValueError('different legacy layout population')
            paired['layout']=dict(Counter(('new_good' if n['layout_good_20'] else 'new_bad')+'__'+
                ('old_good' if b['layout_good_20'] else 'old_bad') for y,n,b in zip(labels,pred,base) if y))
            paired['joint_accept']=dict(Counter(('new_good' if n['layout_good_20'] and new[n['pair_id']]['accepted'] else 'new_bad')+'__'+
                ('old_good' if b['layout_good_20'] and old[b['pair_id']]['accepted'] else 'old_bad') for y,n,b in zip(labels,pred,base) if y))
        result['splits'][name]['s7_hard_vs_original']=paired
        print(name,policies['bounded_max_f1']['pooled_out_of_fold'],flush=True)
    save(root/'comparison.json',result);save(root/'oof_predictions.json',oof)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--prior',required=True);run(p.parse_args())

"""Evaluate completed frozen predictions; never lower thresholds below .2."""
import argparse
from pathlib import Path
from common import read,save,crossfit,metrics,GRID


def evaluate(root):
    root=Path(root);registry=read(root/'registry.json');all_results={};complete=True
    for split in ('real','ood'):
        meta=read(root/split/'manifest.json');meta['split']=split
        labels=[r['label'] for r in meta['pairs']];models={}
        for key,reg in registry.items():
            path=(Path(reg['reuse_predictions']) if 'reuse_predictions' in reg else root/'predictions'/key)/(split+'.json')
            if not path.exists():complete=False;continue
            pred=read(path)
            if pred['status']!='complete':complete=False;continue
            assert pred['checkpoint_sha256']==reg['checkpoint_sha256']
            assert [r['pair_id'] for r in pred['rows']]==[r['pair_id'] for r in meta['pairs']]
            scores=[r['score'] for r in pred['rows']];valid=[r['decision_valid'] for r in pred['rows']]
            good=[r['layout_good_20'] for r in pred['rows']] if split=='real' else None
            receipt=reg.get('model_receipt') or reg['provenance'][split]['model']
            old_t=receipt['operating_points']['thresholds']['max_f1']
            before=metrics(labels,[v and s>=old_t for s,v in zip(scores,valid)],scores,good)
            fixed=metrics(labels,[v and s>=.3 for s,v in zip(scores,valid)],scores,good)
            policies={}
            for policy in ('bounded_max_f1','bounded_recall95'):
                policies[policy],out=crossfit(meta,pred['rows'],policy)
                save(root/'oof'/split/key/(policy+'.json'),out)
            models[key]=dict(cohort=reg['cohort'],old_sim_threshold=old_t,old_sim=before,
                fixed_03=fixed,cv=policies,checkpoint_sha256=reg['checkpoint_sha256'])
        all_results[split]=dict(positive=sum(labels),negative=len(labels)-sum(labels),models=models)
    save(root/'results.json',dict(status='complete' if complete else 'partial',
        registered_models=len(registry),threshold_grid=list(GRID),scores_rescaled=False,
        test_labels_used_to_select_thresholds=False,model_weights_changed=False,results=all_results))
    print({split:{key:dict(threshold=v['cv']['bounded_max_f1']['threshold_median'],
        f1=v['cv']['bounded_max_f1']['pooled_out_of_fold']['f1'],
        recall=v['cv']['bounded_max_f1']['pooled_out_of_fold']['recall'])
        for key,v in data['models'].items()} for split,data in all_results.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);evaluate(p.parse_args().root)

"""Independent sklearn recount of final out-of-fold aggregates."""
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from common import read,save


def verify(root):
    root=Path(root);data=read(root/'results.json');registry=read(root/'registry.json')
    assert data['status']=='complete' and len(registry)==26
    count=0;max_delta=0.
    for split in ('real','ood'):
        meta=read(root/split/'manifest.json');expected={r['pair_id'] for r in meta['pairs']}
        for key,item in data['results'][split]['models'].items():
            for policy,result in item['cv'].items():
                rows=read(root/'oof'/split/key/(policy+'.json'))
                assert len(rows)==len(expected) and {r['pair_id'] for r in rows}==expected
                assert all(.2<=r['threshold']<=.8 for r in rows)
                y=[r['label'] for r in rows];a=[r['accepted'] for r in rows];s=[r['score'] for r in rows]
                derived=dict(accuracy=accuracy_score(y,a),precision=precision_score(y,a,zero_division=0),
                    recall=recall_score(y,a),f1=f1_score(y,a),auroc=roc_auc_score(y,s))
                for name,value in derived.items():
                    delta=abs(value-result['pooled_out_of_fold'][name])
                    max_delta=max(max_delta,delta);assert delta<1e-12
                count+=1
    save(root/'verification.json',dict(status='complete',independent_metric_library='sklearn',
        dataset_model_policy_tables=count,max_absolute_metric_delta=max_delta,
        thresholds_within_user_constraint=True,all_heldout_ids_exact_once=True))
    print(dict(status='complete',checked=count,max_delta=max_delta))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);verify(p.parse_args().root)

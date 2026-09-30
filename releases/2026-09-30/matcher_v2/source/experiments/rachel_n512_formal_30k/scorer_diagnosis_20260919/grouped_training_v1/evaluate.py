"""Final C16 only. SIMTEST + unchanged source-separated real calibration folds."""
import argparse
from dataclasses import asdict
import importlib.util
from types import FunctionType
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from support import *
from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as original
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.local_evidence_v2.model import make
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.local_evidence_v2.evaluate import LocalEvidenceInference

REAL_PROTOCOL=R/'bounded_real_calibration_v2_20260921'
GT=Path('/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json')

def metric_module():
    p=R/'bounded_real_calibration_v2_code_20260921/common.py'
    spec=importlib.util.spec_from_file_location('grouped_bounded_metrics',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def load_frozen_model(root,selection):
    if selection!='fixed_epoch':raise ValueError('fixed C16 only')
    root=Path(root);freeze=read(root/'freeze.json');identity=freeze['identity'];checkpoint=root/freeze['checkpoint']
    if freeze['schema']!=SCHEMA or freeze['status']!='complete' or freeze['real_ood_used_for_fit']:
        raise ValueError('not completed group training')
    if data.sha(checkpoint)!=freeze['checkpoint_sha256']:raise ValueError('checkpoint changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['completed_segments']!=64 or saved['matcher_updated']:raise ValueError('incomplete C16')
    if json.dumps(saved['identity'],sort_keys=True)!=json.dumps(identity,sort_keys=True):raise ValueError('identity differs')
    head=make('reference512_h4');head.load_state_dict(saved['model'],strict=True)
    base=cache.old.load_decoupled_checkpoint(cache.source_checkpoint()).base_model
    net=LocalEvidenceInference(base,head).eval().requires_grad_(False);op=freeze['operating_points']
    receipt=dict(training_run=str(root),selection=selection,budget=28,epoch=28,head_epoch=16,seed=260914,
        checkpoint_path=str(checkpoint),checkpoint_sha256=freeze['checkpoint_sha256'],
        freeze_path=str(root/'freeze.json'),freeze_sha256=data.sha(root/'freeze.json'),
        model_config=asdict(base.config),sampling='original512',architecture='grouped_training_v1:'+identity['arm'],
        classifier_thresholds={k:op['thresholds']['max_f1'] for k in original.core.BRANCHES},
        operating_points=op,training_identity=identity,model_design=net.metadata(),
        classifier_only_pair_bce=identity['arm']==ARMS[0],local_and_fused_are_same_single_classifier=True,
        coarse_is_untrained_diagnostic=True,test_or_real_used_for_fit=False,ood_used_for_fit=False,decoder_unchanged=True)
    return net,receipt

def summarize_rows(rows,labels,threshold,metric):
    scores=[float(r['classification']['fused']) for r in rows];valid=[bool(r['decision_valid']) for r in rows]
    good=[bool(r['layouts']['full_top2_mode']['valid'] and
        r['layouts']['full_top2_mode'].get('translation_l2_px') is not None and
        r['layouts']['full_top2_mode']['translation_l2_px']<=20) for r in rows]
    return metric.metrics(labels,[v and s>=threshold for v,s in zip(valid,scores)],scores,good)

def run(root,arm):
    root=Path(root);training=root/'training'/arm;dest=root/'evaluation'/arm;dest.mkdir(parents=True,exist_ok=True)
    metric=metric_module();test=dest/'test'
    if not (test/'summary.json').exists():
        args=original.parser().parse_args(['--training-run',str(training),'--selection','fixed_epoch',
            '--split','test','--output',str(test),'--batch-size','4','--workers','2'])
        fn=FunctionType(original.run.__code__,dict(original.run.__globals__,load_frozen_model=load_frozen_model),
            original.run.__name__,original.run.__defaults__,original.run.__closure__)
        fn(args)
    test_rows=[json.loads(x) for x in (test/'pair_results.jsonl').read_text().splitlines() if x]
    summary=dict(arm=arm,status='evaluating',test_fixed03=summarize_rows(test_rows,[int(r['label']) for r in test_rows],.3,metric))
    net,receipt=load_frozen_model(training,'fixed_epoch');torch.set_num_threads(2)
    original.core.sealed._set_determinism(260914);net=net.to('cuda:0')
    save(dest/'model_receipt.json',receipt)
    for split in ('real','ood'):
        meta=read(REAL_PROTOCOL/split/'manifest.json');meta['split']=split
        folder=dest/split;folder.mkdir(exist_ok=True)
        predictions_path=folder/'pair_predictions.json'
        if predictions_path.exists():rows=read(predictions_path)
        else:
            with np.load(Path(meta['prepared'])/'inputs.npz',allow_pickle=False) as f:
                arrays={k:f[k] for k in ('packed_masks','points','valid')}
            batches=original.core.real.input_batches(dict(fragment_ids=meta['fragment_ids'],pairs=meta['pairs']),arrays,4)
            rows=[]
            with torch.inference_mode():
                for b in batches:rows.extend(original.core.predict_batch(net,b,torch.device('cuda:0')))
            if [r['pair_id'] for r in rows]!=[r['pair_id'] for r in meta['pairs']]:raise ValueError('prediction ID mismatch')
            save(predictions_path,rows) # Freeze before reading held-out Layout GT.
        gt={r['pair_id']:r for r in read(GT)['positive_pairs']} if split=='real' else {}
        compact=[]
        for row,pair in zip(rows,meta['pairs']):
            error=None;l=row['layouts']['full_top2_mode']
            if split=='real' and pair['label']:
                target=gt[pair['pair_id']]
                if (target['fragment_a_token'],target['fragment_b_token'])!=(pair['fragment_a_id'],pair['fragment_b_id']):
                    raise ValueError('GT endpoints differ')
                if l['valid']:error=float(np.linalg.norm(np.asarray(l['translation_rc'])-target['translation_gt_a_to_b_rc']))
            compact.append(dict(pair_id=pair['pair_id'],score=float(row['classification']['fused']),
                decision_valid=bool(row['decision_valid']),layout_error_px=error,
                layout_good_20=bool(l['valid'] and error is not None and error<=20) if split=='real' else None))
        save(folder/'predictions.json',compact)
        y=[int(p['label']) for p in meta['pairs']];s=[p['score'] for p in compact];v=[p['decision_valid'] for p in compact]
        good=[p['layout_good_20'] for p in compact] if split=='real' else None
        fixed=metric.metrics(y,[bool(a and b>=.3) for a,b in zip(v,s)],s,good)
        cv,oof=metric.crossfit(meta,compact,'bounded_max_f1');save(folder/'oof.json',oof)
        summary[split]=dict(positive=sum(y),negative=len(y)-sum(y),fixed03=fixed,bounded_cv=cv,
            matched_prior_population=True,checkpoints_frozen_before_real_inference=True)
        save(dest/'summary.json',summary)
    summary['status']='complete';save(dest/'summary.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--arm',choices=ARMS,required=True)
    a=p.parse_args();run(a.root,a.arm)

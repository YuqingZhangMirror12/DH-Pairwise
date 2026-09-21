"""New negatives only, using each model's original frozen-source loader."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import importlib
import os
from pathlib import Path
import sys
import time

from common import read,save,rows,sha


@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def compact(row):
    layout=row.get('layouts',{}).get('full_top2_mode',{})
    error=layout.get('translation_l2_px')
    return dict(pair_id=row['pair_id'],score=float(row['classification']['fused']),
        decision_valid=bool(row['decision_valid']),layout_error_px=error,
        layout_good_20=None if error is None else bool(layout.get('valid') and error<=20))


def execute(root, key):
    started=time.time();root=Path(root);reg=read(root/'registry.json')[key]
    destination=root/'predictions'/key
    if (destination/'status.json').exists() and read(destination/'status.json')['status']=='complete':
        return
    destination.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=reg['pythonpath'].split(':')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluator
    from experiments.rachel_n512_formal_30k.run_real_layout_decoder_experiment import input_batches
    torch.set_num_threads(1)
    evaluator.core.sealed._set_determinism(260913)
    module=importlib.import_module(reg['module'])
    if reg['matcher_epoch']:
        bridge=module.EndpointBridge(reg['matcher_checkpoint'],reg['matcher_epoch'])
        loader=bridge.evaluate.load_frozen_model
    else:
        loader=module.load_frozen_model
    kwargs={'budget':reg['head_budget']} if reg['head_budget'] is not None else {}
    model,receipt=loader(Path(reg['training_run']),'fixed_epoch',**kwargs)
    if receipt['checkpoint_sha256']!=reg['checkpoint_sha256']:
        raise ValueError('frozen checkpoint differs from original endpoint')
    save(destination/'model_receipt.json',receipt)
    device=torch.device('cuda:0')
    model=model.to(device).eval().requires_grad_(False)
    spectral='spectral_training' in reg['module']
    for split in ('real','ood'):
        outpath=destination/(split+'.json')
        if outpath.exists() and read(outpath)['status']=='complete':continue
        meta=read(root/split/'manifest.json')
        original={r['pair_id']:r for r in rows(reg['provenance'][split]['pair_results'])}
        reused=[]
        for r in meta['pairs']:
            if not r['reuse_original_score']:continue
            old=original[r['pair_id']]
            if (old['fragment_a'],old['fragment_b'])!=(r['fragment_a_id'],r['fragment_b_id']):
                raise ValueError('reused endpoints differ')
            reused.append(dict(compact(old),prediction_source='original_fixed_checkpoint'))
        new_rows=[r for r in meta['pairs'] if not r['reuse_original_score']]
        with np.load(Path(meta['prepared'])/'inputs.npz',allow_pickle=False) as f:
            arrays={k:f[k] for k in ('packed_masks','points','valid')}
        def batches(entries):
            return input_batches(dict(fragment_ids=meta['fragment_ids'],pairs=entries),arrays,1)
        if spectral:
            # Preserve the original CPU-batch1 summary semantics, including SVD;
            # no summary normalization is fitted on calibration/test data.
            path=root/'spectral_cache'/(split+'.json')
            with lock(root/'spectral_cache'/'.lock'):
                if not path.exists():
                    items=module.cache.pair_items(batches(new_rows))
                    values=[]
                    for _,record,_ in module.cache.bounded_compute(items,4,str(module.train.shared.SOURCE)):
                        values.append(asdict(record))
                        if len(values)%100==0:
                            save(destination/'status.json',dict(status='spectral_precompute',split=split,done=len(values),total=len(new_rows)))
                    save(path,dict(status='complete',source=str(module.train.shared.SOURCE),
                        source_sha256=module.train.shared.SOURCE_SHA,records=values))
                cache_record=read(path)
            lookup={r['pair_id']:module.cache.f._read_record(r) for r in cache_record['records']}
            # Reuse official input-bound batch method; this inference-only view
            # cannot fit statistics and has precisely the new pair records.
            class SourceBoundView:
                batch=module.cache.f.FrozenSummaryCache.batch
                def __init__(self, records):self._lookup=records
            endpoint_cache=SourceBoundView(lookup)
            def predict(batch):
                model.bind_batch(batch,endpoint_cache)
                return evaluator.core.predict_batch(model,batch,device)
        else:
            def predict(batch):return evaluator.core.predict_batch(model,batch,device)
        # A small exact-input check catches a mismatched historical loader.
        # Spectral models use the published original cache for this check.
        checks=[r for r in meta['pairs'] if r['reuse_original_score']][:2]
        parity=[]
        for batch in batches(checks):
            if spectral:
                old_root=Path(reg['provenance'][split]['protocol']).parent
                old_receipt=read(old_root/'protocol.json')['model']
                bundle=old_receipt['endpoint_summary_cache']
                cached=module.cache.f.load_cache(bundle['path'],bundle['identity'],bundle['sha256'])
                model.bind_batch(batch,cached)
                output=evaluator.core.predict_batch(model,batch,device)
            else:output=predict(batch)
            for row in output:
                before=compact(original[row['pair_id']]);after=compact(row)
                delta=abs(after['score']-before['score'])
                parity.append(dict(pair_id=row['pair_id'],score_delta=delta))
                if delta>1e-4 or before['decision_valid']!=after['decision_valid']:
                    raise ValueError('historical inference parity differs: '+str(parity[-1]))
        save(destination/(split+'_parity.json'),parity)
        predictions=[]
        with torch.inference_mode():
            for batch in batches(new_rows):
                predictions.extend(dict(compact(r),prediction_source='new_cross_source_inference') for r in predict(batch))
                if len(predictions)%100==0:
                    save(destination/'status.json',dict(status='inference',key=key,split=split,
                        new_done=len(predictions),new_total=len(new_rows),batch=1,training=False))
        merged={r['pair_id']:r for r in reused+predictions}
        assert set(merged)=={r['pair_id'] for r in meta['pairs']}
        assert all(np.isfinite(r['score']) and 0<=r['score']<=1 for r in merged.values())
        save(outpath,dict(status='complete',key=key,split=split,checkpoint_sha256=reg['checkpoint_sha256'],
            trained=False,reused_count=len(reused),new_inferred_count=len(predictions),
            rows=[merged[r['pair_id']] for r in meta['pairs']]))
    save(destination/'status.json',dict(status='complete',training=False,elapsed_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--key',required=True)
    p.add_argument('--gpu',required=True);a=p.parse_args()
    with lock(Path(a.root)/'gpu_locks'/(a.gpu+'.lock')):execute(a.root,a.key)

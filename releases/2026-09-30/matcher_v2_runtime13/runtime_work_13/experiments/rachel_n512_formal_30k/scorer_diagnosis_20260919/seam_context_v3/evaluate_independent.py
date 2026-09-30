"""Frozen external evaluation of the two independently selected Scorer arms.

Reuses the exact B22 external input/GT/metric adapter. Never trains or changes
thresholds. Normal evaluation requires the predeclared training budget complete;
the only exception is an explicitly requested epoch0 input/output parity check.
"""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from .config import Config
from .independent_features import IndependentFeatureModel
from .prepare import read, save, sha
from .evaluate_external import NAMES, load_inputs, describe_prediction, attach_targets, summarize

ARMS=('frozen_features','independent_features')


def validate_selection(cp, selection, status, base_sha, initial_parity=False):
    """Keep model/threshold selection separate from real testing."""
    if cp.get('stage')!='independent_scorer_adaptation' or cp.get('arm') not in ARMS:
        raise ValueError('not an independent Scorer checkpoint')
    binding=cp['binding'];chosen=cp['selection_state']
    if binding!=selection['binding'] or binding['base_sha256']!=base_sha or binding['arm']!=cp['arm']:
        raise ValueError('checkpoint/base/selection binding mismatch')
    if any(chosen[k]!=selection[k] for k in ('epoch','key','threshold')):
        raise ValueError('not the selected checkpoint')
    if cp['epoch']!=chosen['epoch']+1 or cp['offset']!=0 or cp['updates']!=chosen['epoch']*750:
        raise ValueError('best checkpoint is not the selected epoch boundary')
    if binding['epochs']!=16 or binding['effective_batch']!=32:
        raise ValueError('different experiment budget')
    if not .2<=chosen['threshold']<=.80000001 or chosen['threshold']!=selection['report']['threshold']:
        raise ValueError('threshold not bound to the CAL report')
    if initial_parity:
        if chosen['epoch']!=0 or cp['updates']!=0:
            raise ValueError('initial parity permits only the original epoch0')
    else:
        if status.get('status')!='training_complete' or status.get('stop_reason')!='prespecified16epoch_budget_completed':
            raise ValueError('training not complete; no premature real evaluation')
        if (status.get('last_epoch'),status.get('updates'),status.get('exposures'))!=(16,12000,384000):
            raise ValueError('completed training budget differs')
        if status.get('matcher_unchanged') is not True or status['best']!=chosen:
            raise ValueError('terminal state differs from selected frozen checkpoint')


def load_winner(run, base, initial_parity=False):
    run,base=Path(run),Path(base)
    # torch.load holds one atomic checkpoint generation; binding checks reject a
    # concurrent selection change rather than silently mixing generations.
    cp=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    selection=read(run/'selection.json');status=read(run/'status.json')
    validate_selection(cp,selection,status,sha(base),initial_parity)
    parent=torch.load(base,map_location='cpu',weights_only=False)
    if cp['config']!=parent['config'] or parent['stage']!='B':
        raise ValueError('base model architecture changed')
    cfg=Config(**{f.name:cp['config'][f.name] for f in fields(Config)})
    model=IndependentFeatureModel(cfg,parent['model'],cp['arm'])
    model.load_state_dict(cp['model'],strict=True)
    for key,value in parent['model'].items():
        if not key.startswith('verifier.') and not torch.equal(value,cp['model'][key]):
            raise ValueError('frozen Matcher changed: '+key)
    model.eval().requires_grad_(False)
    return model,cp,selection


def setup(seed):
    torch.set_num_threads(1);random.seed(seed);np.random.seed(seed)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False


def run(a):
    setup(260923)
    model,cp,selection=load_winner(a.run,a.base)
    device=torch.device('cuda:0');model=model.to(device)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.time()
    meta,batches,source,dataset=load_inputs(a.split,a.microbatch)
    protocol=dict(status='running',arm=cp['arm'],split=a.split,checkpoint=str(Path(a.run)/'best.pt'),
        checkpoint_sha256=sha(Path(a.run)/'best.pt'),base_checkpoint_sha256=cp['binding']['base_sha256'],
        selection=str(Path(a.run)/'selection.json'),selected_epoch=selection['epoch'],threshold=selection['threshold'],
        initialization='B22_finetune_not_scratch',frozen_matcher_verified=True,config=cp['config'],source=source,
        sample_count=len(meta['pairs']),microbatch=a.microbatch,precision='fp32',pid=os.getpid(),
        model_input_fields=list(NAMES),gt_used_to_generate_candidates=False,real_used_to_select_checkpoint=False,
        threshold_fitting=False,historical_real_design_exposure=True,script_sha256=sha(__file__),
        adapter_sha256=sha(Path(__file__).with_name('evaluate_external.py')))
    save(out/'protocol.json',protocol)
    try:
        predictions=[]
        with (out/'pair_predictions.jsonl').open('x') as stream,torch.no_grad():
            for pairs,batch in batches:
                tensors=[torch.as_tensor(np.array(batch[k],copy=True),device=device,
                    dtype=torch.bool if k.startswith('contour_valid') else torch.float32) for k in NAMES]
                o=model(*tensors,decode=True,verify=True)
                for b,pair in enumerate(pairs):
                    row=describe_prediction(o,b,pair,model.cfg)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');predictions.append(row)
                stream.flush()
                if len(predictions)%64<a.microbatch or len(predictions)==len(meta['pairs']):
                    save(out/'status.json',dict(status='inference',processed=len(predictions),total=len(meta['pairs']),
                        seconds=time.time()-started,pid=os.getpid()))
            os.fsync(stream.fileno())
        if [r['pair_id'] for r in predictions]!=[p['pair_id'] for p in meta['pairs']]:
            raise ValueError('missing or reordered predictions')
        save(out/'prediction_complete.json',dict(status='all_predictions_frozen',count=len(predictions),
            checkpoint_sha256=protocol['checkpoint_sha256'],predictions_sha256=sha(out/'pair_predictions.jsonl')))
        # Targets are joined only after every model prediction has been saved.
        rows=attach_targets(predictions,meta,a.split,dataset)
        with (out/'case_diagnostics.jsonl').open('x') as stream:
            for r in rows:stream.write(json.dumps(r,allow_nan=False)+'\n')
        groups={'all':rows}
        if a.split=='turufan':groups['positive301']=[r for r in rows if r['label']]
        summary=dict(status='complete',arm=cp['arm'],split=a.split,threshold=selection['threshold'],
            groups={k:dict(primary=summarize(v,selection['threshold']),fixed03=summarize(v,.3)) for k,v in groups.items()},
            checkpoint_sha256=protocol['checkpoint_sha256'],layout_gt_available=a.split!='turufan')
        save(out/'summary.json',summary)
        protocol.update(status='complete',seconds=time.time()-started,max_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        save(out/'protocol.json',protocol);save(out/'status.json',dict(status='complete',count=len(rows),seconds=time.time()-started))
        print(json.dumps(dict(status='complete',arm=cp['arm'],split=a.split,seconds=time.time()-started)),flush=True)
    except Exception as error:
        save(out/'failure.json',dict(error=repr(error),pid=os.getpid()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','base','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--split',choices=('dunhuang_cv','turufan'),required=True)
    p.add_argument('--microbatch',type=int,default=8)
    run(p.parse_args())

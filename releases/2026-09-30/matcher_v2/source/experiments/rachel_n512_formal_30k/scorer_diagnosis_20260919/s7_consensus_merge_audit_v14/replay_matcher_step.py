"""Disposable CPU step comparing old/new source from one preserved checkpoint.

This is not a migrated formal checkpoint or a GPU/DDP resume test. The two
logical ranks' microbatch gradients are averaged sequentially on CPU. No
formal file, optimizer, budget, selection state, or process is modified.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DistributedSampler


MODULE='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def next_batches(dataset,record,config):
    if record['offset']%config.effective_batch:
        raise ValueError('checkpoint is not at a completed optimizer update')
    if record['offset']+config.effective_batch>len(dataset):
        raise ValueError('choose a mid-epoch snapshot; do not silently advance the sampler')
    result=[]
    for rank in range(config.world_size):
        sampler=DistributedSampler(dataset,num_replicas=config.world_size,rank=rank,
            shuffle=True,seed=config.data_seed,drop_last=False)
        sampler.set_epoch(record['epoch'])
        indices=list(sampler);start=record['offset']//config.world_size
        for micro in range(config.accumulate):
            first=start+micro*config.microbatch
            result.append((rank,micro,indices[first:first+config.microbatch]))
    return result


def run(source,checkpoint,reference,contract,calibration,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if Path.cwd().resolve()!=source or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('must run explicitly CPU-only from the requested isolated source')
    if 'diagnostics' not in out.parts or out.exists():
        raise ValueError('new diagnostics output required; never reuse a formal root')
    train=importlib.import_module(MODULE+'.train')
    data=importlib.import_module(MODULE+'.data')
    if Path(train.__file__).resolve().parents[4]!=source:
        raise ValueError('mixed source imports')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=train.TrainingConfig();binding=saved['binding']
    if saved['stage']!='matcher' or binding['arm']!='scratch' or saved['world_size']!=2:
        raise ValueError('preserved two-rank scratch Matcher checkpoint required')
    old_cfg=binding['config'];new_cfg=json.loads(json.dumps(cfg.record()))
    extras={'proposal_revision','merge_repair_policy','validation_checkpoint_archive'}
    if {k:v for k,v in new_cfg.items() if k not in extras}!=old_cfg:
        raise ValueError('optimizer/training configuration differs')
    manifest=json.loads(Path(contract).read_text())
    for path,key in ((contract,'data_contract_sha256'),(calibration,'geometry_calibration_sha256'),
                     (reference,'reference_checkpoint_sha256')):
        if digest(path)!=binding[key]:
            raise ValueError('preserved checkpoint input binding differs')
    adapter=train.S7MatcherAdapter.from_s7_m12(reference)
    adapter=train.fresh_matcher(adapter.base.config,cfg.matcher_seed)
    geometry=train.CompatibilityConfig.from_calibration(json.loads(Path(calibration).read_text()))
    model=train.S7Consensus(adapter,geometry,head=train.fresh_head(cfg.head_seed))
    model.matcher.set_frozen(False);model.head.requires_grad_(False);model.train()
    model.load_state_dict(saved['model'],strict=True)
    def prohibited(*args,**kwargs):
        raise AssertionError('Matcher training unexpectedly called the candidate decoder/Scorer')
    model.builder=prohibited;model.score_pair=prohibited
    raw=train.TrainModule(model,'matcher',cfg);raw.epoch=saved['epoch']
    named=[(n,p) for n,p in raw.named_parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([p for _,p in named],lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    optimizer.load_state_dict(saved['optimizer'])
    if not equal_tree(optimizer.state_dict(),saved['optimizer']):
        raise AssertionError('optimizer restore changed state')
    dataset=data.Dataset(manifest['train']['path'],manifest['train']['sha256'])
    schedule=next_batches(dataset,saved,cfg)
    if sum(len(ids) for _,_,ids in schedule)!=cfg.effective_batch:
        raise AssertionError('incomplete effective batch')
    optimizer.zero_grad(set_to_none=True);rows=[];start=time.time()
    for rank in range(cfg.world_size):
        rng=saved['rng'][rank]
        random.setstate(rng['python']);np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        # CUDA RNG is preserved in the source checkpoint but is not applied on CPU.
        for r,micro,indices in schedule:
            if r!=rank:
                continue
            batch=data.collate([dataset[i] for i in indices])
            loss,parts,counts=raw(batch)
            (loss/(cfg.world_size*cfg.accumulate)).backward()
            rows.append(dict(rank=rank,micro=micro,indices=indices,pair_ids=batch['pair_ids'],
                loss=float(loss.detach()),parts={k:float(v) for k,v in parts.items()},counts=counts))
            del batch,loss,parts
    gradients={n:p.grad.detach().clone() for n,p in named if p.grad is not None}
    if len(gradients)!=len(named) or not all(bool(torch.isfinite(g).all()) for g in gradients.values()):
        raise AssertionError('missing/nonfinite Matcher gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(raw.parameters(),cfg.gradient_clip_norm))
    if not math.isfinite(norm):
        raise AssertionError('nonfinite gradient norm')
    optimizer.step()
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),saved['model'][k])]
    if not changed or any(not k.startswith('matcher.') for k in changed):
        raise AssertionError('expected only disposable Matcher weights to change')
    core=Path(train.__file__).parent
    result=dict(schema='s7-matcher-disposable-step/1',status='passed',source=str(source),
        checkpoint_sha256=digest(checkpoint),source_sha256={p.name:digest(p) for p in core.glob('*.py')},
        from_epoch=saved['epoch'],from_offset=saved['offset'],from_updates=saved['updates'],
        physical_microbatch=cfg.microbatch,logical_world_size=cfg.world_size,accumulate=cfg.accumulate,
        effective_batch=cfg.effective_batch,optimizer_updates_disposable=1,formal_updates=0,
        decoder_and_scorer_not_called=True,parameter_names=[n for n,_ in named],
        restored_rng_ranks=len(saved['rng']),cuda_rng_replayed=False,
        gradient_norm=norm,changed_model_tensors=changed,rows=rows,seconds=time.time()-start,
        limitation='Sequential CPU averaging; not a real two-GPU DDP or GPU RNG replay')
    out.mkdir(parents=True,exist_ok=False)
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    optimizer=optimizer.state_dict(),gradients=gradients),out/'disposable.pt')
    result['disposable_sha256']=digest(out/'disposable.pt')
    (out/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ('status','seconds','from_epoch','from_offset','from_updates',
        'effective_batch','gradient_norm','decoder_and_scorer_not_called','limitation')}))


def compare(old,new,out):
    old,new,out=Path(old),Path(new),Path(out)
    if out.exists():raise ValueError('preserve prior comparison')
    a=json.loads((old/'receipt.json').read_text());b=json.loads((new/'receipt.json').read_text())
    for k in ('checkpoint_sha256','from_epoch','from_offset','from_updates','physical_microbatch',
              'logical_world_size','accumulate','effective_batch','parameter_names','rows','gradient_norm'):
        if a[k]!=b[k]:raise AssertionError('CPU replay differs: '+k)
    states=[]
    for directory,receipt in ((old,a),(new,b)):
        if digest(directory/'disposable.pt')!=receipt['disposable_sha256']:
            raise ValueError('disposable state changed')
        states.append(torch.load(directory/'disposable.pt',map_location='cpu',weights_only=False))
    for key in ('model','optimizer','gradients'):
        if not equal_tree(states[0][key],states[1][key]):raise AssertionError('CPU states differ: '+key)
    record=dict(schema='s7-matcher-source-step-parity/1',status='passed',old_receipt_sha256=digest(old/'receipt.json'),
        new_receipt_sha256=digest(new/'receipt.json'),exact_equal=['sample_order','loss_components','gradients',
        'gradient_norm','updated_model','updated_optimizer'],pairs=a['effective_batch'],
        trainable_tensors=len(a['parameter_names']),formal_updates=0,
        limitation='One discarded CPU step; selection/plateau and GPU resume still need separate verification')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(record,f,indent=2,allow_nan=False)
    print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    step=sub.add_parser('step')
    for name in ('source','checkpoint','reference','contract','calibration','out'):step.add_argument('--'+name,required=True)
    diff=sub.add_parser('compare')
    for name in ('old','new','out'):diff.add_argument('--'+name,required=True)
    args=vars(parser.parse_args());mode=args.pop('mode')
    (run if mode=='step' else compare)(**args)

"""Scratch-branch batch/gradient test; optimizer steps are discarded."""
import argparse
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import torch

from . import matcher as matcher_module
from .data import Dataset,collate,to_device
from .matcher import S7MatcherAdapter,INPUTS
from .preflight_matcher import digest,state_digest
from .scratch_matcher import fresh_matcher,matching_loss


def run(args):
    output=Path(args.output)
    if output.exists():
        raise ValueError('preserve previous gate receipt')
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    reference=S7MatcherAdapter.from_s7_m12(args.checkpoint)
    model=fresh_matcher(reference.base.config).to(args.device).train()
    if torch.equal(model.base.primal.weight.cpu(),reference.base.primal.weight):
        raise AssertionError('scratch unexpectedly equals historical weights')
    del reference
    initial=state_digest(model)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    data=Dataset(args.manifest)
    result=dict(status='running',scope='scratch actual-data FP32 gates, NOT formal training',
        source_architecture_checkpoint_sha256=digest(args.checkpoint),manifest_sha256=data.sha256,
        batch_size=args.batch_size,updated_weights_discarded=True,batches=[],optimizer_updates=0,
        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    def save():
        output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    start=time.time();save()
    try:
        for step in range(3):
            batch=to_device(collate([data[step*args.batch_size+i] for i in range(args.batch_size)]),args.device)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.time()
            with patch.object(matcher_module,'dustbin_sinkhorn',wraps=matcher_module.dustbin_sinkhorn) as sink:
                o=model(*(batch[k] for k in INPUTS))
            if sink.call_count!=1:
                raise AssertionError('expected single Sinkhorn')
            loss,parts,counts=matching_loss(model,o,batch)
            if not torch.isfinite(loss):
                raise AssertionError('nonfinite scratch loss')
            loss.backward()
            for branch in ('patch_encoder','context','primal','dual'):
                grads=[p.grad for p in getattr(model.base,branch).parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):
                    raise AssertionError('bad gradient path: '+branch)
            if any(p.grad is not None for m in (model.base.coarse,model.base.local_head,model.base.fusion) for p in m.parameters()):
                raise AssertionError('unused legacy classifier gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            receipt=dict(pair_ids=batch['pair_ids'],loss=float(loss.detach()),
                components={k:float(v.detach()) for k,v in parts.items()},counts=counts,
                elapsed_seconds=time.time()-begin,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,sinkhorn_calls=1)
            result['batches'].append(receipt);result['optimizer_updates']+=1;save()
            print(json.dumps({k:receipt[k] for k in ('loss','elapsed_seconds','peak_allocated_mb','counts')}),flush=True)
            del o,loss,parts,batch
        result.update(status='passed',parameters_changed=state_digest(model)!=initial,elapsed_seconds=time.time()-start)
        if not result['parameters_changed']:
            raise AssertionError('no actual scratch optimizer update')
        save()
    except Exception as error:
        result.update(status='failed',failure_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=8)
    run(p.parse_args())

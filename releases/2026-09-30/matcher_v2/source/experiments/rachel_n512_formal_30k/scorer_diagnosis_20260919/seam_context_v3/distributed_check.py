"""Disposable A/B synchronization check; no formal checkpoint output."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import os
import time
import gc
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .train import TrainingSystem,optimizer_for
from .config import Config
from .data import collate,to_device
from .preflight import sample_items,full_arc_trial
from .prepare import read,save


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True)
    p.add_argument('--preflight',required=True);p.add_argument('--checkpoint')
    p.add_argument('--microbatch',type=int);p.add_argument('--full-arc-stress',action='store_true');args=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    torch.manual_seed(260921);cfg=Config();device=torch.device('cuda',rank);world=dist.get_world_size()
    if 32%world:raise RuntimeError('world size must divide effective32')
    per_rank=32//world
    preflight=read(args.preflight)
    if not preflight['passed']:raise RuntimeError('preflight did not pass')
    items=sample_items(32)[rank::world]
    system=TrainingSystem(cfg).to(device);events=[]
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    if payload:system.model.load_state_dict(payload['model'])
    for stage in ('A','B'):
        opt=optimizer_for(system,stage,cfg)
        if payload and stage==payload['stage']:opt.load_state_dict(payload['optimizer'])
        wrapped=DDP(system,device_ids=[rank],broadcast_buffers=False)
        micro=args.microbatch or preflight['microbatch'][stage];accum=per_rank//micro
        if not accum or per_rank%micro:raise RuntimeError('microbatch must divide per-rank effective batch')
        torch.cuda.reset_peak_memory_stats();dist.barrier();begin=time.time()
        opt.zero_grad(set_to_none=True);local_loss=0.
        for k in range(accum):
            batch=to_device(collate(items[k*micro:(k+1)*micro]),device)
            with wrapped.no_sync() if k+1<accum else nullcontext():
                loss,parts,counts=wrapped(batch,stage,.5 if stage=='B' else 0.)
                (loss/accum).backward();local_loss+=float(loss)/accum
        torch.nn.utils.clip_grad_norm_(system.parameters(),5.,error_if_nonfinite=True);opt.step()
        flat=torch.cat([p.detach().flatten() for p in system.parameters()])
        other=flat.clone();dist.broadcast(other,0)
        difference=float((flat-other).abs().max())
        if difference>1e-6:raise RuntimeError('DDP weights diverged')
        torch.cuda.synchronize()
        event=dict(stage=stage,local_loss=local_loss,max_parameter_difference=difference,
                   microbatch=micro,accumulation=accum,world_size=world,effective_batch=32,
                   seconds=time.time()-begin,peak_mb=torch.cuda.max_memory_allocated()/2**20)
        gathered=[None]*world;dist.all_gather_object(gathered,event)
        events.append(dict(event,all_ranks=gathered))
        if rank==0:print(gathered,flush=True)
        state={k:v.detach().clone() for k,v in system.model.state_dict().items()};del wrapped,opt,batch,loss
        with torch.random.fork_rng(devices=[rank]):new=TrainingSystem(cfg).to(device)
        new.model.load_state_dict(state);system=new
        del state;gc.collect();torch.cuda.empty_cache()
    stress=None
    if args.full_arc_stress:
        opt=optimizer_for(system,'B',cfg)
        stress=full_arc_trial(system,opt,items,args.microbatch or preflight['microbatch']['B'],device)
        if stress['peak_mb']>.85*torch.cuda.get_device_properties(rank).total_memory/2**20:
            raise RuntimeError('insufficient full-arc memory headroom')
        gathered=[None]*world;dist.all_gather_object(gathered,stress);stress=gathered
    if rank==0:save(args.report,dict(passed=True,events=events,full_arc_stress=stress,
                                    weights_saved=False,source_checkpoint=args.checkpoint))
    dist.destroy_process_group()

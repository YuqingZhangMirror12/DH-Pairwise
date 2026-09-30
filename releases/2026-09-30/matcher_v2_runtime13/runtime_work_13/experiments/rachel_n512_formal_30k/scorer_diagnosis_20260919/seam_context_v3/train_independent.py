"""Equal-budget frozen-Matcher Scorer adaptation, isolated from formal v3."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from .data import Dataset, collate, to_device
from .independent_features import from_checkpoint, ScorerTrainingSystem
from .prepare import read, save, sha
from .train import atomic_checkpoint, rng_state, restore_rng, initial_digest
from .validate import validate


def run(a):
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    if a.microbatch*world*a.accumulation!=32:raise ValueError('effective batch must remain32')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(hours=3))
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    seed=260923;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);data=Path(a.data)
    if rank==0:
        lock=(out/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight=read(a.preflight)
    if not preflight.get('passed') or preflight['microbatch']!=a.microbatch:raise ValueError('preflight required')
    binding={'base_sha256':sha(a.base),'arm':a.arm,'seed':seed,'epochs':16,'effective_batch':32,
        'microbatch':a.microbatch,'accumulation':a.accumulation,'world':world,
        'data':{p:sha(data/p) for p in ('protocol.json','train.json','cal_clean.json','cal_hard.json','select_clean.json','select_hard.json')},
        'code':{p:sha(Path(__file__).parent/p) for p in ('independent_features.py','train_independent.py','model.py','seam_verifier.py','seam_proposals.py','losses.py','arc_context.py')}}
    model=from_checkpoint(a.base,a.arm).to(device);system=ScorerTrainingSystem(model)
    groups=[dict(params=list(model.verifier.parameters()),lr=5e-5,name='verifier')]
    if model.scorer_features is not None:groups.append(dict(params=list(model.scorer_features.parameters()),lr=2.5e-5,name='scorer_features'))
    opt=torch.optim.AdamW(groups,weight_decay=1e-4)
    matcher_initial={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith(('verifier.','scorer_features.'))}
    frozen_digest=initial_digest(model);epoch=1;offset=0;updates=0;best=None
    if (out/'last.pt').exists():
        if not a.resume:raise ValueError('existing run requires --resume')
        cp=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if cp['binding']!=binding:raise ValueError('resume binding mismatch')
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);restore_rng(cp['rng_by_rank'][rank])
        epoch,offset,updates,best=cp['epoch'],cp['offset'],cp['updates'],cp['selection_state']
    def unchanged():
        return all(torch.equal(v,model.state_dict()[k]) for k,v in matcher_initial.items())
    def checkpoint(ne,no,is_best=False):
        states=[rng_state()]
        if world>1:
            states=[None]*world
            dist.all_gather_object(states,rng_state())
        if rank==0:
            if not unchanged():raise RuntimeError('frozen Matcher changed')
            cp=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=model.cfg.record(),arm=a.arm,
                epoch=ne,offset=no,updates=updates,exposures=updates*32,optimizer=opt.state_dict(),rng_by_rank=states,
                selection_state=best,binding=binding,stage='independent_scorer_adaptation')
            atomic_checkpoint(out/'last.pt',cp)
            if is_best:atomic_checkpoint(out/'best.pt',cp)
        if world>1:dist.barrier()
    def validation(e):
        nonlocal best
        report,rr=validate(model,data,'B',device,4,a.workers)
        key=[report['selection_value']]+report['tiebreak']
        improved=best is None or key>best['key']
        if improved:best=dict(epoch=e,key=key,threshold=report['threshold'])
        if rank==0:
            save(out/f'epoch_{e:03d}_validation.json',report)
            if improved:
                save(out/'selection.json',dict(**best,report=report,binding=binding,checkpoint='best.pt'))
                save(out/'best_rows.json',rr)
            print(json.dumps(dict(event='validation',arm=a.arm,epoch=e,best=best,improved=improved)),flush=True)
        checkpoint(e+1,0,improved)
    if rank==0:
        save(out/'CONFIG.json',dict(binding=binding,model_config=model.cfg.record(),initial_digest=frozen_digest,
            parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_matcher=True,final_refinement_trainable=True,real_used_for_selection=False,
            loss_weights=dict(candidate=1.,rank=.5,pose=.25),fp32=True))
        save(out/'status.json',dict(status='initial_validation' if best is None else 'resuming',arm=a.arm,epoch=epoch,updates=updates))
    if best is None and a.initial_validation_cache:
        cache=Path(a.initial_validation_cache);selected=read(cache/'selection.json')
        if selected['epoch']!=0 or selected['binding']['base_sha256']!=binding['base_sha256'] or selected['binding']['arm']!=a.arm or selected['binding']['data']!=binding['data']:
            raise ValueError('initial validation cache is not the identical B22/data/arm')
        best={k:selected[k] for k in ('epoch','key','threshold')}
        if rank==0:
            save(out/'epoch_000_validation.json',selected['report'])
            save(out/'selection.json',dict(**best,report=selected['report'],binding=binding,checkpoint='best.pt'))
            save(out/'best_rows.json',read(cache/'best_rows.json'))
            save(out/'initial_validation_reuse.json',dict(source=str(cache),reason='already completed identical epoch0 inference; previous attempt failed saving RNG before any optimizer updates'))
        checkpoint(1,0,True)
    elif best is None:validation(0)
    ddp=DistributedDataParallel(system,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False) if world>1 else system
    dataset=Dataset(data/'train.json',train_mirror_probability=.1,seed=seed)
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=False)
    denom=a.microbatch*world;started=time.time()
    while epoch<=16:
        sampler.set_epoch(epoch);dataset.set_epoch(epoch)
        loader=DataLoader(dataset,batch_size=a.microbatch,sampler=sampler,collate_fn=collate,
            num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
        system.train();opt.zero_grad(set_to_none=True);sums={};n=0;t0=time.time();initial_offset=offset
        for i,batch in enumerate(loader):
            if i*denom<offset:continue
            batch=to_device(batch,device);sync=(i+1)%a.accumulation==0
            with nullcontext() if sync or world==1 else ddp.no_sync():
                loss,parts,counts=ddp(batch)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                (loss/a.accumulation).backward()
            if sync:
                torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
                opt.step();opt.zero_grad(set_to_none=True);updates+=1
            for k,v in parts.items():sums[k]=sums.get(k,0.)+float(v)
            n+=1
            if sync and (updates<=3 or updates%50==0) and rank==0:
                save(out/'status.json',dict(status='training',arm=a.arm,epoch=epoch,total_epochs=16,
                    completed_pairs=(i+1)*denom,epoch_pairs=len(dataset),updates=updates,exposures=updates*32,
                    microbatch_per_gpu=a.microbatch,accumulation=a.accumulation,effective_batch=32,gpus=world,
                    latest_loss={k:float(v) for k,v in parts.items()},last_counts=counts,
                    peak_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20,elapsed_seconds=time.time()-started,best=best))
            if sync and updates%200==0:checkpoint(epoch,(i+1)*denom)
        offset=0
        if epoch in (8,12):
            for group in opt.param_groups:group['lr']*=.5
        if rank==0:
            save(out/f'epoch_{epoch:03d}_train.json',dict(epoch=epoch,seconds=time.time()-t0,
                trained_pairs_this_segment=len(dataset)-initial_offset,loss={k:v/max(n,1) for k,v in sums.items()},
                updates=updates,learning_rates=[g['lr'] for g in opt.param_groups],matcher_unchanged=unchanged()))
        if epoch%2==0:validation(epoch)
        else:checkpoint(epoch+1,0)
        epoch+=1
    if rank==0:
        save(out/'status.json',dict(status='training_complete',arm=a.arm,stop_reason='prespecified16epoch_budget_completed',
            last_epoch=16,best=best,updates=updates,exposures=updates*32,matcher_unchanged=unchanged(),
            convergence_claim=False,next_action='evaluate frozen winner; real data not used for checkpoint selection'))
    if world>1:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=('frozen_features','independent_features'))
    for key in ('base','data','out','preflight'):p.add_argument('--'+key,required=True)
    p.add_argument('--microbatch',type=int,default=8);p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--resume',action='store_true')
    p.add_argument('--initial-validation-cache');a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if int(os.environ.get('RANK','0'))==0:save(Path(a.out)/'failure.json',dict(error=repr(e),action='inspect; no automatic training changes'))
        raise


if __name__=='__main__':main()

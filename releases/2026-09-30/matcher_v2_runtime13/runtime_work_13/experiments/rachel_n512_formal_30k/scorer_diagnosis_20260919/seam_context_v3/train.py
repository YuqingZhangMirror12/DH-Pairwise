"""Two-stage scratch DDP training with resumable, bounded platform-based schedule."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from .config import Config
from .model import SeamContextModel
from .losses import compute_loss
from .data import Dataset, collate, to_device, INPUTS
from .prepare import save, read, sha
from .validate import validate


class TrainingSystem(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.model=SeamContextModel(cfg)

    def forward(self,batch,stage,teacher_weight=0.):
        o=self.model(*(batch[k] for k in INPUTS),decode=stage=='B',verify=stage=='B')
        loss,parts,counts=compute_loss(self.model,o,batch,stage,teacher_weight)
        # Uniform DDP participation for empty/unknown supervision. Such zero
        # terms are NOT used as evidence of nonzero module learning in preflight.
        loss=loss+sum(p.flatten()[0]*0 for p in self.parameters() if p.requires_grad)
        return loss,{k:v.detach() for k,v in parts.items()},counts


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state(r['cuda'])


def check_resume_migration(payload,binding,cfg,world,batching,manifest,checkpoint):
    """Explicit approved topology transitions; never relax model/data checks."""
    if not manifest:
        raise RuntimeError('world-size/code migration requires an explicit resume manifest')
    if manifest.get('authorized_2_to_4_resume'):
        if (payload['world_size'],world)!=(2,4) or payload['offset']!=0 or payload['stage']!='A':
            raise RuntimeError('2->4 migration requires the approved A epoch boundary')
        if batching!={'A':8,'B':8} or set(manifest['new_rank_seeds'])!={'2','3'}:
            raise RuntimeError('2->4 migration requires microbatch8 and explicit new-rank RNG streams')
    elif manifest.get('authorized_2_to_4_B_resume'):
        if (payload['world_size'],world)!=(2,4) or payload['stage']!='B':
            raise RuntimeError('2->4 B migration requires the approved joint-stage continuation')
        if not 0<=payload['offset']<=24000 or payload['offset']%cfg.effective_batch:
            raise RuntimeError('2->4 B resume must be at a committed global optimizer boundary')
        if batching!={'A':8,'B':8} or set(manifest.get('new_rank_seeds',{}))!={'2','3'}:
            raise RuntimeError('2->4 B migration requires microbatch8 and explicit new-rank RNG streams')
        if len(payload['rng_by_rank'])!=2:
            raise RuntimeError('2->4 B source checkpoint must contain both rank RNG states')
    elif manifest.get('authorized_4_to_2_resume'):
        if (payload['world_size'],world)!=(4,2) or payload['stage']!='B':
            raise RuntimeError('4->2 migration requires the approved B continuation')
        if not 0<=payload['offset']<=24000 or payload['offset']%cfg.effective_batch:
            raise RuntimeError('4->2 resume must be at a committed global optimizer boundary')
        if batching.get('A')!=16 or batching.get('B') not in (8,16):
            raise RuntimeError('4->2 migration requires tested microbatch16 or B8 with accumulation2')
        if manifest.get('new_rank_seeds')!={} or manifest.get('retired_ranks')!=[2,3]:
            raise RuntimeError('4->2 migration restores ranks0/1 and retires ranks2/3')
        if len(payload['rng_by_rank'])!=4:
            raise RuntimeError('4->2 source checkpoint must contain all four rank RNG states')
    else:
        raise RuntimeError('unapproved topology migration')
    if payload['config']!=cfg.record() or cfg.effective_batch!=32:
        raise RuntimeError('migration must preserve model/loss config and effective32')
    if manifest['source_checkpoint_sha256']!=sha(checkpoint):
        raise RuntimeError('migration checkpoint identity mismatch')
    if manifest['source_binding']!=payload['binding'] or manifest['destination_binding']!=binding:
        raise RuntimeError('migration source/destination bindings do not match')
    if {k:v for k,v in binding.items() if k!='implementation'}!={k:v for k,v in payload['binding'].items() if k!='implementation'}:
        raise RuntimeError('migration cannot change datasets or validation protocol')
    old=payload['binding']['implementation'];new=binding['implementation']
    changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    if not changed<= {'train.py','launch.py','distributed_check.py'}:
        raise RuntimeError('migration changed model, losses or data code: '+str(sorted(changed)))
    if manifest['resume_position']!={k:payload[k] for k in ('stage','epoch','offset','updates','exposures')}:
        raise RuntimeError('migration position mismatch')
    if manifest['target_world_size']!=world or manifest['microbatch']!=batching:
        raise RuntimeError('migration target topology mismatch')
    return manifest


def atomic_checkpoint(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);temporary.replace(path)


def initial_digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_for(system,stage,cfg):
    core,heads=[],[]
    for name,p in system.model.named_parameters():
        is_head=name.startswith('verifier.')
        p.requires_grad_(stage=='B' or not is_head)
        if p.requires_grad:(heads if is_head else core).append(p)
    groups=[dict(params=core,lr=cfg.lr_a if stage=='A' else cfg.lr_b)]
    if heads:groups.append(dict(params=heads,lr=cfg.lr_new))
    return torch.optim.AdamW(groups,weight_decay=cfg.weight_decay)


def run(args):
    cfg=Config();rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(hours=3))
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
    out=Path(args.out).resolve();data=Path(args.data).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=None
    if rank==0:
        lock=(out/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(data/'protocol.json')
    if protocol['status']!='ready' or protocol['train_count']!=24000:raise RuntimeError('data protocol not ready')
    binding={p.name:sha(p) for p in (data/'protocol.json',data/'train.json',data/'cal_clean.json',data/'cal_hard.json',data/'select_clean.json',data/'select_hard.json')}
    binding['implementation']={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    binding['implementation']['beam_kernel.cpp']=sha(Path(__file__).with_name('beam_kernel.cpp'))
    preflight=read(Path(args.preflight));
    if not preflight.get('passed'):raise RuntimeError('preflight did not pass')
    batching=preflight['microbatch'];system=TrainingSystem(cfg).to(device)
    initial=initial_digest(system.model);stage='A';epoch=1;offset=0;updates=0;exposures=0
    state=dict(best=None,schedule_best=None,initial_metric=None,bad=0,lr_bad=0,reductions=0,since_reduction=0)
    payload=None;last=out/'last.pt';migration=None;topology_changed=False
    if last.exists():
        if not args.resume:raise RuntimeError('existing run requires --resume')
        payload=torch.load(last,map_location='cpu',weights_only=False)
        if payload['config']!=cfg.record():raise RuntimeError('resume config mismatch')
        if payload['binding']!=binding or payload['world_size']!=world:
            manifest=read(Path(args.resume_migration)) if args.resume_migration else None
            migration=check_resume_migration(payload,binding,cfg,world,batching,manifest,last)
            topology_changed=True
        else:migration=payload.get('resume_migration')
        system.model.load_state_dict(payload['model']);stage=payload['stage'];epoch=payload['epoch'];offset=payload['offset']
        updates=payload['updates'];exposures=payload['exposures'];state=payload['selection_state'];initial=payload['initial_digest']
    opt=optimizer_for(system,stage,cfg)
    if payload is not None:
        opt.load_state_dict(payload['optimizer'])
        if topology_changed and rank>=payload['world_size']:
            seed=int(migration['new_rank_seeds'][str(rank)])
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        else:restore_rng(payload['rng_by_rank'][rank])
        if topology_changed and initial_digest(system.model)!=migration['model_digest']:
            raise RuntimeError('restored model differs from paused checkpoint')
    if rank==0:
        save(out/'CONFIG.json',dict(**cfg.record(),microbatch=batching,world_size=world,
            precision='FP32 (initial release; no silent AMP)',source_binding=binding,initial_digest=initial,
            deterministic_note='seeded; GPU reductions not claimed bitwise identical across hardware',
            resume_migration=migration))
    dataset=Dataset(data/'train.json',train_mirror_probability=cfg.train_mirror_probability,seed=cfg.seed)
    started=time.time()
    def save_checkpoint(next_epoch,next_offset,extra_path=None):
        nonlocal stage
        states=[rng_state()]
        if world>1:
            states=[None]*world;dist.all_gather_object(states,rng_state())
        if rank==0:
            cp=dict(config=cfg.record(),binding=binding,initial_digest=initial,stage=stage,
                epoch=next_epoch,offset=next_offset,updates=updates,exposures=exposures,
                model={k:v.detach().cpu() for k,v in system.model.state_dict().items()},
                optimizer=opt.state_dict(),rng_by_rank=states,selection_state=state,
                sampler_seed=cfg.seed,world_size=world,resume_migration=migration)
            atomic_checkpoint(last,cp)
            if extra_path is not None:atomic_checkpoint(extra_path,cp)
        if world>1:dist.barrier()
    resume_start_updates=updates
    if topology_changed:
        # Commit the identical model/optimizer and sampler position with the
        # destination rank RNG states before any resumed optimizer update.
        save_checkpoint(epoch,offset)
        if rank==0:
            event=f"resumed_{payload['world_size']}_to_{world}"
            rng_note=('ranks0/1 restored; new ranks2/3 separately seeded' if world==4 else
                      'ranks0/1 restored; ranks2/3 retired after their saved updates')
            save(out/f'resume_event_{world}gpu.json',dict(event=event,stage=stage,epoch=epoch,
                offset=offset,updates=updates,exposures=exposures,world_size=world,microbatch=batching,
                effective_batch=cfg.effective_batch,model_digest=migration['model_digest'],
                source_checkpoint_sha256=migration['source_checkpoint_sha256'],
                preserved=['model','optimizer','selection_state','sampler_seed','epoch','offset'],
                rng_note=rng_note+'; topology changes are not bitwise-equivalent'))
            print(json.dumps(dict(event=event,epoch=epoch,updates=updates,
                world_size=world,microbatch=batching,effective_batch=cfg.effective_batch)),flush=True)
    while True:
        micro=int(batching[stage]);denom=world*micro
        if cfg.effective_batch % denom:raise RuntimeError('microbatch/world must divide effective32')
        accumulation=cfg.effective_batch//denom
        ddp=DistributedDataParallel(system,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False) if world>1 else system
        sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=cfg.seed,drop_last=False)
        maximum=cfg.stage_a_max if stage=='A' else cfg.stage_b_max
        minimum=cfg.stage_a_min if stage=='A' else cfg.stage_b_min
        stage_finished=False
        while epoch<=maximum:
            sampler.set_epoch(epoch+(0 if stage=='A' else 10000))
            dataset.set_epoch(epoch+(0 if stage=='A' else 10000))
            loader=DataLoader(dataset,batch_size=micro,sampler=sampler,collate_fn=collate,
                num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
            system.train();opt.zero_grad(set_to_none=True);start_epoch=time.time();sums={};nsteps=0
            epoch_start_offset=offset
            teacher=max(0.,.5*(1-(epoch-1)/(cfg.teacher_zero_epoch-1))) if stage=='B' else 0.
            for ix,batch in enumerate(loader):
                processed=ix*denom
                if processed<offset:continue
                batch=to_device(batch,device)
                last_micro=(ix+1)%accumulation==0
                context=nullcontext() if last_micro or world==1 else ddp.no_sync()
                with context:
                    loss,parts,counts=ddp(batch,stage,teacher)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                    (loss/accumulation).backward()
                if last_micro:
                    if stage=='A' and epoch==1:
                        opt.param_groups[0]['lr']=cfg.lr_a*min(1.,(ix+1)*denom/24000)
                    torch.nn.utils.clip_grad_norm_(system.parameters(),5.,error_if_nonfinite=True)
                    opt.step();opt.zero_grad(set_to_none=True);updates+=1
                    exposures+=cfg.effective_batch
                for k,v in parts.items():sums[k]=sums.get(k,0.)+float(v)
                nsteps+=1
                if last_micro and (updates<=3 or updates-resume_start_updates<=3 or updates%50==0) and rank==0:
                    save(out/'status.json',dict(status='training',stage=stage,epoch=epoch,
                        completed_pairs=(ix+1)*denom,epoch_pairs=24000,updates=updates,exposures=exposures,
                        microbatch_per_gpu=micro,effective_batch=32,gpus=world,teacher_weight=teacher,
                        epoch_augmentation_counts=dataset.augmentation_counts(),
                        latest_losses_rank0={k:float(v) for k,v in parts.items()},last_counts_rank0=counts,
                        peak_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20,elapsed_seconds=time.time()-started))
                if last_micro and updates%200==0:save_checkpoint(epoch,(ix+1)*denom)
            offset=0
            if rank==0:
                save(out/f'{stage}_epoch_{epoch:03d}_train.json',dict(stage=stage,epoch=epoch,
                    seconds=time.time()-start_epoch,teacher_weight=teacher,loss={k:v/max(nsteps,1) for k,v in sums.items()},
                    resumed_pair_offset=epoch_start_offset,trained_pairs_this_segment=24000-epoch_start_offset,
                    effective_batch=32,microbatch_per_gpu=micro,world_size=world,updates=updates,exposures=exposures))
            if epoch%cfg.validate_every==0:
                report,rows=validate(system.model,data,stage,device,args.val_microbatch,args.workers)
                metric=report['selection_value'];key=[metric]+report['tiebreak']
                if state['initial_metric'] is None:state['initial_metric']=metric
                eligible=stage=='A' or (epoch>=minimum and epoch-cfg.teacher_zero_epoch+1>=cfg.native_only_min)
                better=eligible and (state['best'] is None or key>state['best']['key'])
                if better:state['best']=dict(key=key,epoch=epoch,threshold=report['threshold'])
                if state['schedule_best'] is None or metric>state['schedule_best']+cfg.min_delta:
                    state['schedule_best']=metric;state['bad']=0;state['lr_bad']=0
                else:state['bad']+=1;state['lr_bad']+=1
                state['since_reduction']+=1
                if state['lr_bad']>=3:
                    old=[g['lr'] for g in opt.param_groups]
                    for g in opt.param_groups:g['lr']=max(cfg.min_lr,g['lr']*.5)
                    if any(g['lr']<v for g,v in zip(opt.param_groups,old)):
                        state['reductions']+=1;state['since_reduction']=0
                    state['lr_bad']=0
                if rank==0:
                    save(out/f'{stage}_epoch_{epoch:03d}_validation.json',report)
                    if better:
                        save(out/f'{stage}_best_rows.json',rows)
                        save(out/f'{stage}_selection.json',dict(**state['best'],report=report,binding=binding,
                            checkpoint='best_stageA_matcher.pt' if stage=='A' else 'best_joint.pt'))
                    print(json.dumps(dict(event='validation_complete',stage=stage,epoch=epoch,value=metric,
                        eligible=eligible,best=state['best'],lr=[g['lr'] for g in opt.param_groups])),flush=True)
                bestpath=out/('best_stageA_matcher.pt' if stage=='A' else 'best_joint.pt') if better else None
                save_checkpoint(epoch+1,0,bestpath)
                if stage=='A' and epoch>=4 and state['best']['key'][0]==0:
                    if rank==0:save(out/'status.json',dict(status='needs_diagnosis',stage='A',epoch=epoch,
                        reason='no native correct candidate in either first validation event; do not blindly run12 epochs'))
                    if world>1:dist.destroy_process_group()
                    return
                plateau=(epoch>=minimum and state['reductions']>=2 and state['since_reduction']>=2 and state['bad']>=6 and eligible)
                if plateau:
                    if stage=='A' and state['best']['key'][0]<.9 and state['best']['key'][0]<state['initial_metric']+cfg.min_delta:
                        if rank==0:save(out/'status.json',dict(status='needs_diagnosis',stage='A',epoch=epoch,
                            reason='native coverage never improved; cannot call a flat scratch model converged'))
                        if world>1:dist.destroy_process_group()
                        return
                    stage_finished=True;break
            else:save_checkpoint(epoch+1,0)
            epoch+=1
        if not stage_finished:
            if rank==0:save(out/'status.json',dict(status='budget_limited',stage=stage,last_epoch=maximum,
                next_action='request budget review; no claim of convergence or unapproved extension',best=state['best']))
            break
        if stage=='B':
            if rank==0:save(out/'status.json',dict(status='training_complete',stage='B',stop_reason='qualified_plateau',
                best=state['best'],updates=updates,exposures=exposures,next_action='frozen evaluation; no real-based reselection'))
            break
        if world>1:dist.barrier()
        checkpoint=torch.load(out/'best_stageA_matcher.pt',map_location='cpu',weights_only=False)
        del ddp
        # New module instance avoids stale DDP reducer hooks when the previously
        # frozen verifier joins the optimizer; preserve this run's RNG stream.
        with torch.random.fork_rng(devices=[local]):
            next_system=TrainingSystem(cfg).to(device)
        next_system.model.load_state_dict(checkpoint['model']);system=next_system
        stage='B';epoch=1;offset=0;opt=optimizer_for(system,stage,cfg)
        state=dict(best=None,schedule_best=None,initial_metric=None,bad=0,lr_bad=0,reductions=0,since_reduction=0)
        save_checkpoint(1,0)
        if rank==0:print(json.dumps(dict(event='stage_B_started',source='best_stageA_matcher.pt',backbone_frozen=False)),flush=True)
    if world>1:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--preflight',required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--val-microbatch',type=int,default=2);p.add_argument('--resume',action='store_true')
    p.add_argument('--resume-migration');a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if int(os.environ.get('RANK','0'))==0:save(Path(a.out)/'failure.json',dict(error=repr(e),recovery='resume last committed optimizer boundary'))
        raise


if __name__=='__main__':main()

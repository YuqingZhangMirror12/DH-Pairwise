"""Finite M8 -> C8 hard-only fine-tuning; source-disjoint SIM selection only."""
import argparse
from dataclasses import asdict
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
from torch.utils.data import DataLoader, DistributedSampler, Subset
from sklearn.metrics import average_precision_score
from .data import Dataset,collate,to_device,SEED
from .model import load_origin,HEAD_SHA
from ..matched_only.cache import SOURCE_SHA
from ..seam_context_v3.prepare import read,save,sha
from ..seam_context_v3.train import atomic_checkpoint,rng_state,restore_rng,initial_digest

EPOCHS=8
GRID=np.arange(20,81,dtype=float)/100


def metrics(rows,threshold):
    labels=np.array([r['label'] for r in rows],bool);scores=np.array([r['score'] for r in rows]);p=scores>=threshold
    tp=int((p&labels).sum());fp=int((p&~labels).sum());fn=int((~p&labels).sum());tn=int((~p&~labels).sum())
    return dict(count=len(rows),tp=tp,fp=fp,fn=fn,tn=tn,accuracy=(tp+tn)/len(rows),
        precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),
        ap=float(average_precision_score(labels,scores)),layout20=sum(r['layout20'] for r in rows)/max(int(labels.sum()),1),
        correct_layout_accepted=sum(r['layout20'] and bool(a) for r,a in zip(rows,p)))


def fit_threshold(views):
    # Each CAL pair has two correlated views, never count them as independent.
    return float(max(GRID,key=lambda t:(np.mean([metrics(v,float(t))['f1'] for v in views]),-abs(float(t)-.3),-float(t))))


@torch.no_grad()
def validate(model,data,device,rank,world,workers):
    model.eval();names=('select_clean','select_hard') if model.phase=='matcher' else ('cal_clean','cal_hard','select_clean','select_hard')
    outputs={};losses={};started=time.time()
    for name in names:
        dataset=Dataset(data/(name+'.json'))
        loader=DataLoader(Subset(dataset,list(range(rank,len(dataset),world))),batch_size=8,
            collate_fn=collate,num_workers=workers,pin_memory=True)
        rows=[];loss_sum=0.;count=0
        for batch in loader:
            batch=to_device(batch,device);o,selected,logit=model.evidence(batch,score=model.phase=='scorer')
            if model.phase=='matcher':
                loss,_=model.matcher_loss(o,batch);scores=torch.zeros_like(batch['labels'])
            else:
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(logit,batch['labels'],reduction='none')*o.training_valid).mean()
                scores=torch.where(o.training_valid,logit,torch.zeros_like(logit)).sigmoid()
            loss_sum+=float(loss)*len(batch['labels']);count+=len(batch['labels'])
            for i,pid in enumerate(batch['pair_ids']):
                label=bool(batch['labels'][i]);valid=bool(selected.layout_valid[i])
                error=float(torch.linalg.vector_norm(selected.translation_a_to_b_rc[i]-batch['translation_a_to_b_rc'][i])) if label and valid else None
                rows.append(dict(pair_id=pid,label=int(label),score=float(scores[i]),layout_valid=valid,
                    layout_error_px=error,layout20=bool(error is not None and error<=20),endpoints_a=int(selected.mask_a[i].sum()),endpoints_b=int(selected.mask_b[i].sum())))
        gathered=[None]*world
        if world>1:dist.all_gather_object(gathered,dict(rows=rows,loss_sum=loss_sum,count=count))
        else:gathered=[dict(rows=rows,loss_sum=loss_sum,count=count)]
        table={r['pair_id']:r for part in gathered for r in part['rows']}
        if len(table)!=len(dataset) or sum(len(p['rows']) for p in gathered)!=len(dataset):raise ValueError('validation duplication or missing pairs')
        outputs[name]=[table[e['pair_id']] for e in dataset.entries]
        losses[name]=sum(p['loss_sum'] for p in gathered)/sum(p['count'] for p in gathered)
    threshold=None if model.phase=='matcher' else fit_threshold([outputs['cal_clean'],outputs['cal_hard']])
    summaries={name:dict(metrics(rows,.5 if threshold is None else threshold),loss=losses[name]) for name,rows in outputs.items()}
    if model.phase=='matcher':
        key=[summaries['select_hard']['layout20'],-losses['select_hard'],summaries['select_clean']['layout20']]
    else:
        key=[summaries['select_hard']['f1'],np.mean([summaries[n]['f1'] for n in ('select_clean','select_hard')]),summaries['select_hard']['ap']]
    return dict(stage=model.phase,key=[float(k) for k in key],threshold=threshold,views=summaries,seconds=time.time()-started,
        checkpoint_selected_on_real=False,cal_select_source_disjoint=True,independent_pair_counts={'cal':82,'select':370}),outputs


def optimizer(model,phase):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5 if phase=='matcher' else 2e-5,weight_decay=1e-4)


def run(a):
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(hours=2))
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(SEED+rank);np.random.seed(SEED+rank);torch.manual_seed(SEED+rank);torch.cuda.manual_seed_all(SEED+rank)
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True);data=Path(a.data)
    model=load_origin(a.matcher,a.head).to(device);dataset=Dataset(data/'train.json',train=True)
    if a.preflight:
        reports=[]
        for phase in ('matcher','scorer'):
            model.set_phase(phase);opt=optimizer(model,phase)
            ddp=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
            before=initial_digest(model.base_model);torch.cuda.reset_peak_memory_stats(device)
            for step in range(2):
                indices=[(rank*a.microbatch+i+step*a.microbatch*world)%len(dataset) for i in range(a.microbatch)]
                batch=to_device(collate([dataset[i] for i in indices]),device);opt.zero_grad(set_to_none=True)
                loss,_=ddp(batch);loss.backward()
                grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                if not grads or not any(bool(g.abs().max()>0) for g in grads):raise ValueError('no learning gradient')
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True);opt.step()
            unchanged=before==initial_digest(model.base_model)
            if phase=='scorer' and not unchanged:raise ValueError('Scorer stage changed Matcher')
            reports.append(dict(stage=phase,loss=float(loss),peak_MiB=torch.cuda.max_memory_allocated()/2**20,matcher_unchanged=unchanged))
            del ddp,opt
        all_reports=[None]*world
        if world>1:dist.all_gather_object(all_reports,reports)
        else:all_reports=[reports]
        if rank==0:save(root/f'preflight_mb{a.microbatch}.json',dict(passed=True,microbatch=a.microbatch,world=world,reports=all_reports,formal_updates=0))
        if world>1:dist.destroy_process_group()
        return
    preflight=read(root/f'preflight_mb{a.microbatch}.json')
    if not preflight.get('passed') or preflight['world']!=world:raise ValueError('matching four-card preflight required')
    if rank==0:
        lock=(root/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binding=dict(schema='s7-hard-only-finetune/1',matcher_sha256=sha(a.matcher),head_sha256=sha(a.head),seed=SEED,
        epochs_per_stage=EPOCHS,world=world,microbatch=a.microbatch,effective_batch=a.microbatch*world,
        data={p.name:sha(p) for p in data.glob('*.json')},code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    resume=None
    if (root/'last.pt').exists():
        if not a.resume:raise ValueError('existing run requires explicit resume')
        resume=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        if resume['binding']!=binding:raise ValueError('resume protocol mismatch')
    if rank==0:
        save(root/'CONFIG.json',dict(binding=binding,model_config=asdict(model.base_model.config),
            initialization='historical S7 M12 Matcher plus matched-token C16 head, not v3 or random',
            training_data=read(data/'protocol.json'),matcher_epochs=8,scorer_epochs=8,
            matcher_lr=[1e-5,5e-6],scorer_lr=[2e-5,1e-5],lr_drop_after_epoch=4,
            optimizer='AdamW fresh per stage',weight_decay=1e-4,fp32=True,layout_decoder_unchanged=True,
            scorer='same 2-layer 4-head matched-token PairBCE, no new loss or architecture',
            selection='Matcher: hard SELECT Layout20 then loss; Scorer: hard SELECT Pair F1 at independent CAL threshold',
            counts_include_last_partial_batch=True,clean_validation_is_reported=True,real_selection=False))
    for phase in ('matcher','scorer'):
        if resume is not None and resume['phase']=='scorer' and phase=='matcher':continue
        if phase=='scorer':
            model=load_origin(a.matcher,a.head).to(device)
            selected=torch.load(root/'best_matcher.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(selected['model'],strict=True)
        model.set_phase(phase);opt=optimizer(model,phase);epoch=1;offset=0;updates=0;exposures=0;best=None
        if resume is not None and resume['phase']==phase:
            model.load_state_dict(resume['model'],strict=True);opt.load_state_dict(resume['optimizer'])
            restore_rng(resume['rng_by_rank'][rank]);epoch=resume['next_epoch'];offset=resume['offset']
            updates=resume['updates'];exposures=resume['exposures'];best=resume['best'];resume=None
        frozen_digest=initial_digest(model.base_model) if phase=='scorer' else None
        def checkpoint(ne,no,is_best=False):
            states=[None]*world
            if world>1:dist.all_gather_object(states,rng_state())
            else:states=[rng_state()]
            if rank==0:
                payload=dict(binding=binding,phase=phase,next_epoch=ne,offset=no,updates=updates,exposures=exposures,
                    model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),
                    rng_by_rank=states,best=best,model_config=asdict(model.base_model.config),loss_config=asdict(model.loss_config))
                atomic_checkpoint(root/'last.pt',payload)
                if is_best:atomic_checkpoint(root/f'best_{phase}.pt',payload)
            if world>1:dist.barrier()
        def validation(e):
            nonlocal best
            report,rows=validate(model,data,device,rank,world,a.workers)
            improved=best is None or report['key']>best['key']
            if improved:best=dict(epoch=e,key=report['key'],threshold=report['threshold'])
            if rank==0:
                save(root/f'{phase}_{e:03d}_validation.json',report)
                if improved:save(root/f'{phase}_selection.json',dict(**best,report=report,binding=binding))
                print(json.dumps(dict(event='validation',phase=phase,epoch=e,best=best,improved=improved)),flush=True)
            checkpoint(e+1,0,improved)
        if best is None:validation(0)
        ddp=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
        sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=SEED,drop_last=False)
        if len(dataset)%world:raise ValueError('avoid duplicated distributed training samples')
        while epoch<=EPOCHS:
            sampler.set_epoch(epoch);dataset.set_epoch(epoch)
            loader=DataLoader(dataset,batch_size=a.microbatch,sampler=sampler,collate_fn=collate,num_workers=a.workers,pin_memory=True)
            model.train();t0=time.time();loss_sum=0.;n=0;start_offset=offset
            for i,batch in enumerate(loader):
                if i*a.microbatch*world<offset:continue
                batch=to_device(batch,device);opt.zero_grad(set_to_none=True);loss,parts=ddp(batch);loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
                opt.step();updates+=1;seen=len(batch['labels'])*world;exposures+=seen;n+=seen;loss_sum+=float(loss)*seen
                next_offset=min((i+1)*a.microbatch*world,len(dataset))
                if updates%25==0 or updates<=3:
                    if rank==0:save(root/'status.json',dict(status='training',phase=phase,epoch=epoch,total_epochs=EPOCHS,
                        completed_pairs=next_offset,epoch_pairs=len(dataset),updates=updates,exposures=exposures,
                        microbatch=a.microbatch,world=world,effective_batch=a.microbatch*world,last_batch=seen,
                        loss=float(loss),best=best,peak_MiB=torch.cuda.max_memory_allocated()/2**20))
                if updates%50==0:checkpoint(epoch,next_offset)
                if updates%10==0:
                    requested=torch.tensor(int(rank==0 and (root/'pause.request').exists()),device=device)
                    if world>1:dist.broadcast(requested,src=0)
                    if int(requested):
                        checkpoint(epoch,next_offset)
                        if rank==0:save(root/'status.json',dict(status='paused',phase=phase,epoch=epoch,offset=next_offset,updates=updates,exposures=exposures,reason='user_pause_request'))
                        if world>1:dist.destroy_process_group()
                        return
            if epoch==4:
                for g in opt.param_groups:g['lr']*=.5
            if phase=='scorer' and initial_digest(model.base_model)!=frozen_digest:raise RuntimeError('frozen Matcher changed during scorer fine-tuning')
            totals=torch.tensor([loss_sum,n],dtype=torch.float64,device=device)
            if world>1:dist.all_reduce(totals)
            if rank==0:save(root/f'{phase}_{epoch:03d}_train.json',dict(epoch=epoch,seconds=time.time()-t0,
                trained_pairs_this_segment=len(dataset)-start_offset,updates=updates,exposures=exposures,loss=float(totals[0]/totals[1].clamp_min(1)),
                lr=opt.param_groups[0]['lr'],matcher_frozen=phase=='scorer'))
            offset=0
            if epoch%2==0:validation(epoch)
            else:checkpoint(epoch+1,0)
            epoch+=1
        if rank==0:save(root/f'{phase}_complete.json',dict(status='stage_complete',epochs=8,updates=updates,exposures=exposures,best=best))
        del ddp,opt
        if world>1:dist.barrier()
    if rank==0:save(root/'status.json',dict(status='training_complete',stop_reason='prespecified_M8_C8_finetune_budget_completed',
        matcher=read(root/'matcher_complete.json'),scorer=read(root/'scorer_complete.json'),
        selected_checkpoint='best_scorer.pt',real_evaluation_pending=True,convergence_claim=False))
    if world>1:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('matcher','head','data','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--microbatch',type=int,default=16);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--preflight',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if int(os.environ.get('RANK','0'))==0:save(Path(a.out)/('preflight_failure.json' if a.preflight else 'failure.json'),dict(error=repr(error)))
        raise

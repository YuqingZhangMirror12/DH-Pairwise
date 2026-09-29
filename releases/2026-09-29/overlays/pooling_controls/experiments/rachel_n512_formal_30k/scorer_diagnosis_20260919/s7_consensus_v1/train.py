"""Queued threshold-union comparison: two fixed Matchers, fresh identical heads.

Only after the CURRENT two-arm experiment completes. No Matcher retraining,
no old head import, no real-data selection. --preflight-steps is disposable.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,DistributedSampler

from .compatibility import CompatibilityConfig
from .config import Plateau,TrainingConfig
from ..decoder_readout_v1.head import EvidenceClusterHead
from .data import Dataset,collate,pair_labels,to_device
from .evaluation import validate
from .evaluation_checkpoint import save_evaluation_checkpoint
from .evidence import PairEvidence
from ..binary_scorer_v1.loss import batch_loss
from .matcher import INPUTS,S7MatcherAdapter
from .frozen_start import load_selected_matcher
from .real_development import bind_plan,RealDevelopment
from .migration import load_migration,load_bound,read_plan,rebase_plateau
from ..binary_scorer_v1.model import BinaryConsensus as S7Consensus
from .preflight_matcher import digest,state_digest
from .proposal_cache import ProposalCache
from .scratch_matcher import fresh_matcher,matching_loss
from .validation_protocol import layout as validation_layout,learning_curve_csv


def save_json(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temporary,path)


def save_torch(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    torch.save(record,temporary);os.replace(temporary,path)


def barrier():
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def restore_rng(record):
    random.setstate(record['python']);np.random.set_state(record['numpy'])
    torch.set_rng_state(record['torch'].cpu());torch.cuda.set_rng_state(record['cuda'].cpu())


def gathered(value,world):
    if world==1:
        return [value]
    values=[None]*world;dist.all_gather_object(values,value);return values


def fresh_head(seed, variant):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return EvidenceClusterHead(variant, remove_overlap=False)


def canonical_record(value):
    """Identical type representation in JSON manifests and torch checkpoints."""
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def make_binding(args,config,contract):
    if contract.get('status')!='passed' or not contract['source_disjoint'] or contract['train']['pairs']!=24000:
        raise ValueError('full source-isolated TRAIN contract required')
    validation_kind,_=validation_layout(contract)
    paths=[contract['train']]+list(contract['validation'].values())
    for record in paths:
        if digest(record['path'])!=record['sha256']:
            raise ValueError('bound data changed: '+record['path'])
    binding=dict(schema='s7-consensus-experiment/1',arm=args.arm,
        reference_checkpoint_sha256=digest(args.checkpoint),data_contract_sha256=digest(args.contract),
        geometry_calibration_sha256=digest(args.calibration),config=config.record(),
        implementation_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        matcher_config_only_for_scratch=False,validation_contract=contract['validation'],
        validation_design=contract.get('validation_design',dict(kind=validation_kind)),
        formal_training=not bool(args.preflight_steps),preflight_steps=args.preflight_steps)
    if args.arm=='scratch_fixed':
        if not args.frozen_matcher_state or digest(args.frozen_matcher_state)!=args.frozen_matcher_sha256:
            raise ValueError('explicit selected scratch Matcher path/hash required')
        if args.frozen_matcher_sha256 != '80cac47d5bc5340df35a7a7c36ab4a3580a9eea8c99adf797ff5744cd2068b17':
            raise ValueError('experiments1/2 require the confirmed E32 Matcher')
        binding['binary_scorer_sha256']={p.name:digest(p) for p in (Path(__file__).parent.parent/'binary_scorer_v1').glob('*.py')}
        binding['pooling_scorer_sha256']={p.name:digest(p) for p in (Path(__file__).parent.parent/'decoder_readout_v1').glob('*.py')}
        binding['real_development']=bind_plan(args.real_split)
        binding['fixed_matcher']=dict(path=str(args.frozen_matcher_state),
            sha256=args.frozen_matcher_sha256,head_imported=False,matcher_training=False)
    origin=getattr(args,'topology_origin',None)
    if origin:
        record=json.loads(Path(origin).read_text())
        if (args.scorer_variant!='patch' or record.get('kind')!='binary-topology-2to1'
                or record.get('effective_batch')!=32 or record.get('from_world_size')!=2
                or record.get('to_world_size')!=1 or digest(record['checkpoint'])!=record['checkpoint_sha256']):
            raise ValueError('explicit unchanged Patch checkpoint required for topology migration')
        binding['topology_transition']=dict(kind=record['kind'],manifest=str(origin),
            manifest_sha256=digest(origin),checkpoint_sha256=record['checkpoint_sha256'],
            original_updates=record['updates'],original_exposures=record['exposures'])
    return canonical_record(binding)


def make_caches(stage_root,binding,matcher,contract,rank):
    all_records={'train':contract['train'],**contract['validation']}
    result={}
    identity=dict(experiment=binding,matcher_state_sha256=state_digest(matcher),
        full_q_precision='float32 online',features='F/H online from identical frozen Matcher',
        cached_content='deterministic proposals only, not selected endpoint features')
    for name,record in all_records.items():
        root=stage_root/'proposal_cache'/name
        record_binding=dict(**identity,manifest_sha256=record['sha256'])
        if rank==0:
            ProposalCache(root,record_binding)
        barrier()
        result[name]=ProposalCache(root,record_binding)
    return result


class TrainModule(nn.Module):
    def __init__(self,model,stage,config,cache=None):
        super().__init__();self.model=model;self.stage=stage;self.config=config;self.cache=cache;self.epoch=0

    def forward(self,batch):
        output=self.model.matcher(*(batch[k] for k in INPUTS))
        if self.stage=='matcher':
            loss,components,counts=matching_loss(self.model.matcher,output,batch)
        else:
            pairs=[PairEvidence.from_matcher(output,i,batch['mask_a'],batch['mask_b']) for i in range(len(batch['labels']))]
            proposals=[self.model.builder(p) if self.cache is None else self.cache.get(key,self.model.builder,p)
                       for p,key in zip(pairs,batch['pair_ids'])]
            predictions=[self.model.score_pair(p,proposals=q) for p,q in zip(pairs,proposals)]
            flags=[int(hashlib.sha256((str(self.epoch)+':'+s).encode()).hexdigest()[:8],16)%self.config.extension_modulus==0
                   for s in batch['pair_ids']]
            loss,items=batch_loss(self.model,predictions,pair_labels(batch),extension_flags=flags)
            components={k:torch.stack([x.components[k].detach() for x in items]).mean() for k in items[0].components}
            counts={k:sum(x.counts.get(k,0) for x in items) for k in items[0].counts}
        # Every rank follows the same reducer graph even for no-candidate rows.
        loss=loss+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite training loss')
        return loss,{k:v.detach() for k,v in components.items()},counts


def run_stage(model,stage,args,config,contract,binding,device,rank,world):
    root=Path(args.out)/stage
    root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():
        complete=json.loads((root/'complete.json').read_text())
        if complete['binding']!=binding:
            raise ValueError('completed stage identity changed')
        return complete
    model.matcher.set_frozen(stage=='scorer');model.head.requires_grad_(stage=='scorer')
    model.train()
    dataset=Dataset(contract['train']['path'],contract['train']['sha256'])
    if len(dataset)!=24000 or len(dataset)%config.effective_batch:
        raise ValueError('formal fixed full-batch population mismatch')
    frozen_hash=state_digest(model.matcher) if stage=='scorer' else None
    initial_head_hash=state_digest(model.head)
    caches=make_caches(root,binding,model.matcher,contract,rank) if stage=='scorer' else None
    development=None if args.preflight_steps else RealDevelopment(args.real_split,binding['real_development'])
    raw=TrainModule(model,stage,config,None if caches is None else caches['train'])
    wrapped=DistributedDataParallel(raw,device_ids=[device.index],broadcast_buffers=False,
        find_unused_parameters=False) if world>1 else raw
    optimizer=torch.optim.AdamW([p for p in raw.parameters() if p.requires_grad],
        lr=config.learning_rate,weight_decay=config.weight_decay)
    plateau=Plateau();epoch=1;offset=0;updates=0;exposures=0;best=None;best_layout=None;best_real=None;curve=[]
    resume=root/'last.pt';migration=None;migration_origin=None
    preflight_origin=dict(updates=0,exposures=0)
    if resume.exists():
        if not args.resume:
            raise ValueError('existing checkpoint requires explicit --resume')
        record=torch.load(resume,map_location=device,weights_only=False)
        if record['binding']!=binding or record['world_size']!=world or record['stage']!=stage:
            raise ValueError('resume identity/topology/stage differs')
        model.load_state_dict(record['model'],strict=True);optimizer.load_state_dict(record['optimizer'])
        plateau=Plateau(**record['plateau'])
        epoch,offset,updates,exposures=(record[k] for k in ('epoch','offset','updates','exposures'))
        best,best_layout,curve=record['best'],record['best_layout'],record['curve']
        best_real=record.get('best_real')
        preflight_origin=record.get('preflight_origin',preflight_origin)
        migration_origin=record.get('migration_origin')
        restore_rng(record['rng'][rank])
    elif stage=='matcher' and getattr(args,'migration_plan',None):
        plan,origin,record=load_migration(args.migration_plan,binding,config)
        model.load_state_dict(record['model'],strict=True)
        optimizer.load_state_dict(record['optimizer'])
        # The saved E26 training batches are complete; validation was pending.
        # Preserve every learned parameter, moment, RNG and consumed update.
        epoch=record['epoch']+1;offset=0
        updates,exposures=record['updates'],record['exposures']
        plateau=Plateau(**record['plateau'])
        preflight_origin=dict(updates=updates,exposures=exposures)
        migration_origin=dict(manifest_sha256=digest(args.migration_plan),
            checkpoint=plan['resume'],retained_updates=updates,retained_exposures=exposures,
            original_plateau=record['plateau'],retained_learning_rates=[g['lr'] for g in optimizer.param_groups],
            pending_original_validation_not_reused=True,old_head_not_continued=True)
        migration=(plan,origin,record)
        restore_rng(record['rng'][rank])
        if rank==0:
            save_json(root/'migration_started.json',dict(status='imported_pending_reselection',
                binding=binding,**migration_origin))

    def checkpoint(next_epoch,next_offset):
        rng=gathered(rng_state(),world)
        if rank==0:
            save_torch(resume,dict(binding=binding,stage=stage,world_size=world,epoch=next_epoch,offset=next_offset,
                updates=updates,exposures=exposures,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),plateau=asdict(plateau),best=best,best_layout=best_layout,
                curve=curve,rng=rng,preflight_origin=preflight_origin,migration_origin=migration_origin,best_real=best_real))

    def evaluate(epoch_number,migration_snapshot=None):
        nonlocal best,best_layout,best_real
        report,predictions=validate(model,contract,stage,device,config,caches)
        real_report,real_predictions=development.evaluate(model,device,config)
        report['real_development']=real_report
        predictions['real_development']=real_predictions
        real_improve=epoch_number>0 and (best_real is None or tuple(real_report['key'])>tuple(best_real['key']))
        if real_improve:
            best_real=dict(epoch=epoch_number,key=real_report['key'],thresholds=real_report['thresholds'])
        row_updates=updates if migration_snapshot is None else epoch_number*24000//config.effective_batch
        row_exposures=exposures if migration_snapshot is None else epoch_number*24000
        row=dict(epoch=epoch_number,updates=row_updates,exposures=row_exposures,**report)
        if migration_snapshot is not None:
            row['migration_reselection_snapshot']=migration_snapshot
            row['historical_metrics_not_used']=True
        curve.append(row)
        improve=best is None or tuple(report['key'])>tuple(best['key'])
        layout_key=report['key'][1]
        layout_improve=best_layout is None or layout_key>best_layout['layout']
        if improve:
            best=dict(epoch=epoch_number,key=report['key'],threshold=report['threshold'])
        if layout_improve:
            best_layout=dict(epoch=epoch_number,layout=layout_key,threshold=report['threshold'])
        if rank==0:
            save_json(root/f'epoch_{epoch_number:03d}_validation.json',row)
            save_json(root/f'epoch_{epoch_number:03d}_predictions.json',predictions)
            save_evaluation_checkpoint(root/f'epoch_{epoch_number:03d}_weights.pt',
                dict(binding=binding,stage=stage,epoch=epoch_number,updates=row_updates,exposures=row_exposures,
                     threshold=report['threshold'],metrics=report,
                     model={k:v.detach().cpu() for k,v in model.state_dict().items()}))
            for changed,name in ((improve,'best_joint.pt'),(layout_improve,'best_layout.pt')):
                if changed:
                    save_torch(root/name,dict(binding=binding,stage=stage,epoch=epoch_number,
                        threshold=report['threshold'],metrics=report,model={k:v.detach().cpu() for k,v in model.state_dict().items()}))
            if real_improve:
                save_torch(root/'best_real.pt',dict(binding=binding,stage=stage,epoch=epoch_number,
                    thresholds=real_report['thresholds'],metrics=real_report,model={k:v.detach().cpu() for k,v in model.state_dict().items()}))
            save_json(root/'selection.json',dict(status='provisional',best=best,best_layout=best_layout,
                best_real=best_real,binding=binding,selection_on_real=True,test_used=False))
            save_json(root/'learning_curve.json',curve)
            (root/'learning_curve.csv').write_text(learning_curve_csv(curve))
        return report

    if migration is not None and not args.preflight_steps:
        plan,origin,imported=migration
        # Reevaluate only genuinely retained checkpoints. Missing historical
        # epochs are not reconstructed, and old cached proposals are not read.
        for snapshot in plan['reevaluate']:
            saved=load_bound(snapshot,origin)
            model.load_state_dict(saved['model'],strict=True)
            if rank==0:
                save_json(root/'status.json',dict(status='migration_reselection',arm=args.arm,stage=stage,
                    imported_epoch=imported['epoch'],reevaluating_epoch=snapshot['epoch'],
                    updates=updates,exposures=exposures,binding=binding,last_update_unix=time.time()))
            baseline=evaluate(snapshot['epoch'],migration_snapshot=snapshot)
        model.load_state_dict(imported['model'],strict=True)
        restore_rng(imported['rng'][rank])
        plateau=rebase_plateau(imported['plateau'],baseline['selection_value'])
        checkpoint(epoch,offset)
        if rank==0:
            save_json(root/'migration_complete.json',dict(status='migration_complete',binding=binding,
                **migration_origin,new_plateau=asdict(plateau),next_epoch=epoch,
                selection=best,reevaluated_epochs=[r['epoch'] for r in plan['reevaluate']],
                missing_historical_weights_not_reconstructed=True,
                maximum_total_matcher_epochs=config.maximum_epochs))
        barrier()
    elif not args.preflight_steps and not curve:
        baseline=evaluate(0)
        plateau.observe(baseline['selection_value'],0,config)
        checkpoint(1,0)
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=config.data_seed,drop_last=False)
    stop_reason=None
    while epoch<=config.maximum_epochs:
        sampler.set_epoch(epoch);raw.epoch=epoch;model.train()
        generator=torch.Generator().manual_seed(config.data_seed+epoch*world+rank)
        loader=DataLoader(dataset,batch_size=config.microbatch,sampler=sampler,collate_fn=collate,
            num_workers=config.workers_per_rank,pin_memory=True,generator=generator)
        begin=time.time();start_offset=offset;local_loss=0.;local_pairs=0;component_sums={};count_sums={}
        optimizer.zero_grad(set_to_none=True)
        for i,batch in enumerate(loader):
            next_offset=(i+1)*config.microbatch*world
            if next_offset<=start_offset:
                continue
            batch=to_device(batch,device)
            sync=(i+1)%config.accumulate==0
            context=wrapped.no_sync() if world>1 and not sync else nullcontext()
            with context:
                loss,parts,counts=wrapped(batch)
                (loss/config.accumulate).backward()
            n=len(batch['labels']);local_loss+=float(loss.detach())*n;local_pairs+=n
            for key,value in parts.items():
                component_sums[key]=component_sums.get(key,0.)+float(value)*n
            for key,value in counts.items():
                count_sums[key]=count_sums.get(key,0)+value
            offset=next_offset
            if sync:
                norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),config.gradient_clip_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError('nonfinite gradient norm')
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                updates+=1;exposures+=config.effective_batch
                if rank==0 and (updates%25==0 or updates-preflight_origin['updates']<=3):
                    save_json(root/'status.json',dict(status='preflight' if args.preflight_steps else 'training',
                        arm=args.arm,stage=stage,epoch=epoch,completed_pairs=offset,epoch_pairs=len(dataset),
                        updates=updates,exposures=exposures,world_size=world,microbatch=config.microbatch,
                        accumulate=config.accumulate,effective_batch=config.effective_batch,
                        learning_rate=optimizer.param_groups[0]['lr'],best=best,last_update_unix=time.time()))
                if args.preflight_steps and updates==preflight_origin['updates']+1:
                    # A disposable real optimizer/RNG checkpoint lets the gate
                    # compare uninterrupted update2 with replay from update1.
                    checkpoint(epoch,offset)
                if args.preflight_steps and updates-preflight_origin['updates']>=args.preflight_steps:
                    hashes=gathered(state_digest(model),world)
                    if len(set(hashes))!=1:
                        raise AssertionError('DDP ranks have divergent updated model states')
                    frozen_unchanged=stage!='scorer' or state_digest(model.matcher)==frozen_hash
                    if not frozen_unchanged:
                        raise AssertionError('frozen Matcher changed')
                    if state_digest(model.head)==initial_head_hash:
                        raise AssertionError('fresh head did not update')
                    receipt=dict(status='passed',formal_training=False,updated_weights_discarded=True,
                        head_updated=True,initial_head_sha256=initial_head_hash,final_head_sha256=state_digest(model.head),
                        arm=args.arm,stage=stage,updates=updates-preflight_origin['updates'],
                        exposures=exposures-preflight_origin['exposures'],
                        total_updates=updates,total_exposures=exposures,world_size=world,
                        microbatch=config.microbatch,accumulate=config.accumulate,effective_batch=config.effective_batch,
                        model_state_hashes=hashes,matcher_unchanged=frozen_unchanged,
                        seconds=time.time()-begin,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                        binding=binding,migration_origin=migration_origin)
                    if rank==0:
                        replay=args.resume and not getattr(args,'topology_gate_initial',False)
                        destination=root/('preflight_resumed.json' if replay else 'preflight.json')
                        if destination.exists():
                            raise ValueError('preserve earlier distributed gate receipt')
                        if replay:
                            initial=json.loads((root/'preflight.json').read_text())
                            if initial['model_state_hashes']!=hashes:
                                raise AssertionError('same-topology checkpoint replay differs from uninterrupted update2')
                            receipt['resume_matches_uninterrupted']=True
                        save_json(destination,receipt)
                    barrier();return receipt
                if not args.preflight_steps and updates%config.checkpoint_every_updates==0:
                    checkpoint(epoch,offset)
            del loss,batch
        if offset!=len(dataset):
            raise AssertionError('incomplete epoch without declared pause')
        totals=gathered(dict(loss_sum=local_loss,pairs=local_pairs,components=component_sums,counts=count_sums),world)
        count=sum(x['pairs'] for x in totals)
        train_record=dict(epoch=epoch,stage=stage,trained_pairs_this_segment=count,updates=updates,exposures=exposures,
            seconds=time.time()-begin,loss=sum(x['loss_sum'] for x in totals)/max(1,count),
            components={k:sum(x['components'].get(k,0.) for x in totals)/max(1,count) for k in component_sums},
            counts={k:sum(x['counts'].get(k,0) for x in totals) for k in count_sums})
        if rank==0:
            train_path=root/f'epoch_{epoch:03d}_train.json'
            if train_path.exists():
                train_path=root/f'epoch_{epoch:03d}_train_from_{start_offset:05d}.json'
            save_json(train_path,train_record)
        if epoch%config.validation_every_epochs==0:
            report=evaluate(epoch)
            action=plateau.observe(report['selection_value'],epoch,config)
            if action=='reduce_lr':
                for group in optimizer.param_groups:
                    group['lr']*=config.learning_rate_factor
            elif action!='continue':
                stop_reason=action
        checkpoint(epoch+1,0)
        barrier()
        if stop_reason:
            break
        epoch+=1;offset=0
    if stop_reason is None:
        stop_reason='budget_limit_not_claimed_converged'
    if stage=='scorer' and state_digest(model.matcher)!=frozen_hash:
        raise AssertionError('frozen Matcher changed during training')
    if best_real is None:raise AssertionError('no trained REAL-SELECT checkpoint')
    selection=dict(status='selected',best=best,best_layout=best_layout,stop_reason=stop_reason,
        binding=binding,actual_epochs=epoch,updates=updates,exposures=exposures,
        best_joint_sha256=digest(root/'best_joint.pt'),best_layout_sha256=digest(root/'best_layout.pt'),
        selection_on_real=True,test_used=False,best_real=best_real,best_real_sha256=digest(root/'best_real.pt'),
        matcher_unchanged=stage=='scorer',migration_origin=migration_origin)
    complete=dict(status='stage_complete',**{k:v for k,v in selection.items() if k!='status'})
    if rank==0:
        save_json(root/'selection.json',selection);save_json(root/'complete.json',complete)
        save_json(root/'status.json',dict(**complete,last_update_unix=time.time()))
    barrier()
    return complete


def run(args):
    if getattr(args,'topology_gate_initial',False) and not (args.resume and args.preflight_steps==12 and args.topology_origin):
        raise ValueError('initial topology gate requires explicit preserved origin and 12-step resume')
    if args.arm != 'scratch_fixed':
        raise ValueError('simplified comparison trains heads only')
    if args.arm=='m12' and (args.frozen_matcher_state or args.frozen_matcher_sha256):
        raise ValueError('historical M12 arm cannot silently use another Matcher')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    config=TrainingConfig(scorer_variant=args.scorer_variant)
    if world!=config.world_size:
        raise ValueError('registered single-GPU execution requires world_size=1 and accumulate=4')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(hours=2))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(config.data_seed+rank);np.random.seed(config.data_seed+rank);torch.manual_seed(config.data_seed+rank)
    contract=json.loads(Path(args.contract).read_text());binding=make_binding(args,config,contract)
    root=Path(args.out)
    if rank==0:
        root.mkdir(parents=True,exist_ok=True)
        if (root/'CONFIG.json').exists() and json.loads((root/'CONFIG.json').read_text())!=binding:
            raise ValueError('experiment identity changed')
        save_json(root/'CONFIG.json',binding)
    barrier()
    geometry=CompatibilityConfig.from_calibration(json.loads(Path(args.calibration).read_text()))
    adapter=S7MatcherAdapter.from_s7_m12(args.checkpoint)
    if args.arm=='scratch_fixed':
        imported=load_selected_matcher(adapter,args.frozen_matcher_state,args.frozen_matcher_sha256)
        if rank==0:save_json(root/'fixed_matcher_import.json',imported)
    model=S7Consensus(adapter,geometry,head=fresh_head(config.head_seed,config.scorer_variant)).to(device)
    stages=('scorer',)
    try:
        for stage in stages:
            outcome=run_stage(model,stage,args,config,contract,binding,device,rank,world)
        if rank==0 and not args.preflight_steps:
            save_json(root/'training_complete.json',dict(status='training_complete',arm=args.arm,
                stages=list(stages),last_stage=outcome,binding=binding,real_evaluation_pending=True))
    except Exception as error:
        if rank==0:
            save_json(root/'failure.json',dict(status='failed',time_unix=time.time(),
                type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),binding=binding))
        raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('scratch_fixed',),required=True)
    parser.add_argument('--scorer-variant',choices=('patch_mean','patch_sum'),required=True)
    for name in ('out','checkpoint','contract','calibration'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--real-split',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--topology-origin',help='explicit preserved two-rank Patch checkpoint migration')
    parser.add_argument('--topology-gate-initial',action='store_true',help='disposable first 12 steps from the explicit topology origin')
    parser.add_argument('--preflight-steps',type=int,default=0)
    parser.add_argument('--frozen-matcher-state',help='completed prior scratch Matcher best_joint.pt')
    parser.add_argument('--frozen-matcher-sha256',help='verified selected Matcher SHA256')
    run(parser.parse_args())

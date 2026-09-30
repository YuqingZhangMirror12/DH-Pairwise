"""Disposable TRAIN-only full-size GPU gradients, batching and learning check."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from .config import Config
from .train import TrainingSystem, optimizer_for
from .data import collate, to_device
from .prepare import TRAIN, read, save, sha
from .targets import base_target_metadata
from .seam_proposals import Candidate
from .losses import compute_loss
from .augmentation import paired_mirror
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample


def sample_items(count=32):
    source=read(TRAIN);entries=source['entries'];chosen=[]
    # Fixed source order; includes positive and negative members of recipe slots.
    for recipe in ('reference_e1','wave','local','seam_gaps','partial_curve','gen5_partition'):
        chosen += [e for e in entries if e['s7_recipe']==recipe][:4]
    seen={e['pair_id'] for e in chosen}
    chosen += [e for e in entries if e['pair_id'] not in seen][:max(0,count-len(chosen))]
    output=[]
    for i,e in enumerate(chosen[:count]):
        s,r=load_sample(Path(source['artifact_root'])/e['artifact_path'])
        # Explicitly exercise both newly authorized reflection directions in
        # the disposable panel; production uses exactly5% +5% per epoch.
        if i<2:
            axis='horizontal' if i==0 else 'vertical'
            s=paired_mirror(s,axis);r=dict(r,v3_paired_mirror=axis)
        output.append((s,r,base_target_metadata(s),e))
    return output


def full_arc_trial(system, opt, items, micro, device):
    """Separate scope releases every graph after success or a caught OOM.

    An exception is caught by the caller only after this frame has unwound;
    failed trials must not retain GPU tensors into the next smaller batch.
    """
    opt.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    batch=to_device(collate(items[:micro]),device)
    o=system.model(*(batch[k] for k in ('mask_a','mask_b','points_rc_a','points_rc_b',
        'contour_valid_a','contour_valid_b')),decode=False)
    base,_,_=compute_loss(system.model,o,batch,'A')
    evidence=[];lengths=[]
    for b,record in enumerate(o.records):
        ids=np.unique(np.linspace(0,len(record.edges)-1,min(512,len(record.edges))).astype(np.int64))
        candidates=[Candidate(ids,np.array([k*16.,k*8.],np.float32),0.) for k in range(8)]
        v=system.model.verify_candidates(o,b,candidates)
        evidence.append(v.logits.square().mean());lengths.extend(v.arc_point_counts)
    stress=base+torch.stack(evidence).mean();stress.backward();torch.cuda.synchronize()
    return dict(stage='B_full_arc_stress8',microbatch=micro,
        peak_mb=torch.cuda.max_memory_allocated()/2**20,
        max_arc_points=max(max(x) for x in lengths),status='passed')


def run(out,learn_steps=8):
    cfg=Config();torch.set_num_threads(2);torch.manual_seed(cfg.seed)
    torch.cuda.set_device(0);device='cuda:0';items=sample_items(32)
    system=TrainingSystem(cfg).to(device);opt=optimizer_for(system,'A',cfg)
    results=[];chosen={}
    for stage in ('A','B'):
        opt=optimizer_for(system,stage,cfg)
        chosen[stage]=None
        for micro in (4,8,16):
            try:
                gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                opt.zero_grad(set_to_none=True)
                batch=to_device(collate(items[:micro]),device)
                begin=time.time();loss,parts,counts=system(batch,stage,.5 if stage=='B' else 0.)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward();torch.cuda.synchronize()
                failures=[name for name,p in system.model.named_parameters() if p.requires_grad and
                          p.grad is not None and not torch.isfinite(p.grad).all()]
                if failures:raise FloatingPointError('nonfinite gradients: '+str(failures))
                peak=torch.cuda.max_memory_allocated()/2**20
                rec=dict(stage=stage,microbatch=micro,seconds=time.time()-begin,loss=float(loss),
                    peak_mb=peak,parts={k:float(v) for k,v in parts.items()},counts=counts,
                    primal_gradient=float(system.model.primal_dual.primal.weight.grad.norm()),
                    dual_gradient=float(system.model.primal_dual.dual.weight.grad.norm()),
                    cnn_gradient=float(system.model.patch_encoder.projection[2].weight.grad.norm()))
                results.append(rec);save(out/'batch_trials.json',results);print(json.dumps(rec),flush=True)
                if peak < .82*(torch.cuda.get_device_properties(0).total_memory/2**20):chosen[stage]=micro
                del batch,loss,parts
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                results.append(dict(stage=stage,microbatch=micro,status='OOM'));break
        if chosen[stage] is None:
            raise RuntimeError('microbatch4 not safe; explicit smaller preflight required')
    # A trained matcher may produce long arcs, unlike random short proposals.
    # Stress8 full-arc hypotheses before choosing B's production microbatch.
    safe_b=None
    for micro in sorted(set((chosen['B'],8,4)),reverse=True):
        try:
            rec=full_arc_trial(system,opt,items,micro,device)
            if rec['peak_mb']<.85*(torch.cuda.get_device_properties(0).total_memory/2**20):safe_b=micro
        except torch.cuda.OutOfMemoryError:
            rec=dict(stage='B_full_arc_stress8',microbatch=micro,status='OOM')
        opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        results.append(rec);save(out/'batch_trials.json',results);print(json.dumps(rec),flush=True)
        if safe_b is not None:break
    if safe_b is None:raise RuntimeError('long-arc memory preflight requires smaller explicit batch')
    chosen['B']=safe_b
    # A fresh disposable model, not the one used in memory trials, learns a
    # fixed32-TRAIN panel. Never save these weights as a formal initialization.
    del system,opt;gc.collect();torch.cuda.empty_cache();torch.manual_seed(cfg.seed)
    system=TrainingSystem(cfg).to(device);opt=optimizer_for(system,'A',cfg)
    micro=chosen['A'];curves=[]
    for step in range(learn_steps):
        opt.zero_grad(set_to_none=True);total=0.
        for start in range(0,32,micro):
            batch=to_device(collate(items[start:start+micro]),device)
            loss,parts,counts=system(batch,'A');(loss*len(batch['labels'])/32).backward()
            total+=float(loss)*len(batch['labels'])/32
        torch.nn.utils.clip_grad_norm_(system.parameters(),5.,error_if_nonfinite=True);opt.step()
        curves.append(total);print(json.dumps(dict(event='disposable_learning_check',step=step+1,loss=total)),flush=True)
    passed=bool(np.isfinite(curves).all() and np.mean(curves[-2:])<np.mean(curves[:2]))
    result=dict(passed=passed,microbatch=chosen,effective_batch=32,intended_world_size=2,
        precision='fp32',batch_trials=results,learning_curve=curves,training_examples=32,
        initialization_weights_saved=False,formal_training_started=False,
        device=torch.cuda.get_device_name(0),parameters=sum(p.numel() for p in system.parameters()))
    result['mirror_panel']={'horizontal':1,'vertical':1,'unmodified':len(items)-2}
    result['source_hashes']={x.name:sha(x) for x in Path(__file__).parent.glob('*.py') if x.name not in ('launch.py','gpu_tests.py')}
    result['source_hashes']['beam_kernel.cpp']=sha(Path(__file__).with_name('beam_kernel.cpp'))
    save(out/'preflight.json',result)
    if not passed:raise RuntimeError('small TRAIN learning check failed; formal training prohibited')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--learn-steps',type=int,default=8);a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    run(root,a.learn_steps)

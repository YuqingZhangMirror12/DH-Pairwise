"""Disposable32-pair/two-step GPU smoke; NEVER a formal checkpoint.

Only complete formal TRAIN caches are accepted. Uses the same fresh head,
FP32 batch16 AdamW/BCE/clip5 path, then discards all weights and moments.
Writes JSON only. No Matcher inference, validation, queueing or resume.
"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from . import data,stage_cache,train

SCHEMA="matched-only-disposable32-smoke/1"
COUNT=32


def require_cuda(device):
    if (device!="cuda:0" or platform.system()!="Linux"
            or not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError("smoke CLI requires Linux and exactly one exposed CUDA GPU0")
    return torch.device(device)


def synchronize(device):
    if device.type=="cuda": torch.cuda.synchronize(device)


def memory_stats(device):
    if device.type!="cuda":
        return dict(peak_allocated_bytes=None,peak_reserved_bytes=None)
    return dict(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))


def summarize_steps(raw):
    result=[]
    for row in raw:
        values={}
        for key,value in row.items():
            if torch.is_tensor(value): value=float(value.detach().cpu())
            values[key]=value
        result.append(values)
    return result


def two_steps(model,dataset,indices,device,*,receipts=None):
    """Numerical helper for CPU tests. CLI has no CPU/limit/batch override."""
    device=torch.device(device)
    ids=np.asarray(indices,dtype=np.int64)
    if (ids.shape!=(COUNT,) or len(np.unique(ids))!=COUNT or (ids<0).any()
            or (ids>=len(dataset)).any() or train.BATCH!=16
            or any(p.dtype!=torch.float32 for p in model.parameters())):
        raise ValueError("requires32 distinct rows, physical/effective16 and FP32 head")
    steps=[] if receipts is None else receipts
    if steps: raise ValueError("smoke must start fresh, not resume partial updates")
    optimizer=train.create_optimizer(model)
    model.train()
    synchronize(device)
    started=time.perf_counter()
    for offset in (0,16):
        batch=dataset.batch(ids[offset:offset+16],device)
        if any(v.requires_grad for v in batch.model_args[:2] if torch.is_tensor(v)):
            raise ValueError("frozen Matcher cached features must not require gradients")
        optimizer.zero_grad(set_to_none=True)
        output=model(*batch.model_args,**batch.model_kwargs)
        loss=train.pair_loss(output.logit,batch.labels,batch.training_valid)
        loss.backward()
        active_gradients=sum(p.grad is not None for p in model.parameters())
        if not active_gradients: raise RuntimeError("head graph produced no parameter gradients")
        # Same check/clip as formal training. Any NaN/Inf gradient makes this
        # total norm nonfinite and raises BEFORE the optimizer update.
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        steps.append(dict(step=len(steps)+1,samples=16,loss=loss.detach(),
            gradient_norm_before_clip=norm.detach(),finite_gradient_check=True,
            active_gradient_tensors=active_gradients,
            training_valid_count=batch.training_valid.sum().detach(),
            fallback_count=output.used_fallback.sum().detach(),pair_ids=list(batch.pair_ids)))
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    elapsed=time.perf_counter()-started
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("nonfinite head after smoke update")
    counters=[int(s["step"]) for s in optimizer.state.values() if "step" in s]
    if not counters or min(counters)<1 or max(counters)>2:
        raise RuntimeError("invalid disposable Adam update counters")
    result=dict(samples=COUNT,optimizer_updates=2,physical_microbatch=16,effective_batch=16,
        accumulation_steps=1,precision="fp32",elapsed_s=elapsed,
        throughput_pairs_s=COUNT/elapsed,steps=summarize_steps(steps),
        finite_parameters_after=True,finite_gradient_check=True,
        active_parameter_adam_step_min=min(counters),active_parameter_adam_step_max=max(counters),
        timing_scope="cold two-step compute+batch reads; excludes construction/report IO; NOT steady-state epoch speed",
        training_counted=False,formal_pair_exposures=0,formal_optimizer_updates=0,
        checkpoint_usable_for_formal=False,weights_persisted=False,formal_reuse_forbidden=True)
    # No state_dict, optimizer_state, torch.save or checkpoint return path.
    del optimizer
    return result


def load_dataset(args):
    source=data.FormalCache(args.train_cache,"train")
    if len(source)!=train.TRAIN_COUNT:
        raise ValueError("smoke requires the entire completed formal TRAIN24K cache, not probe32")
    if args.arm in stage_cache.STAGES:
        if not args.stage_cache: raise ValueError("stage arm requires complete TRAIN stage side-cache")
        return data.CandidateStageCache(source,args.stage_cache,args.arm)
    if args.stage_cache: raise ValueError("original3 arms must not receive stage side-cache")
    return source


def run(args):
    if args.arm not in train.ARMS or args.device!="cuda:0":
        raise ValueError("registered arm and CUDA0 required")
    dataset=load_dataset(args)
    root=Path(args.output).resolve()
    inputs=[Path(args.train_cache).resolve()]+([Path(args.stage_cache).resolve()] if args.stage_cache else [])
    if any(root==p or root in p.parents or p in root.parents for p in inputs):
        raise ValueError("smoke JSON output must be outside cache trees")
    # Same nonblocking child-owned lock as real training. No waiting, stealing,
    # killing or starting behind the current GPU owner's back.
    with train.lock_owner.gpu_lock():
        device=require_cuda(args.device)
        root.mkdir(parents=True,exist_ok=False)
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        train.runner._set_determinism(train.HEAD_SEED)
        model=train.make_scorer(args.arm,seed=train.HEAD_SEED)
        initial_sha=train.state_digest(model)
        protocol=dict(schema=SCHEMA,status="running",arm=args.arm,
            training_counted=False,checkpoint_usable_for_formal=False,
            formal_pair_exposures=0,formal_optimizer_updates=0,requested_samples=COUNT,
            physical_microbatch=16,effective_batch=16,precision="fp32",
            source_binding=dataset.binding,source_mutation_policy="read-only cached features; no Matcher loaded",
            initial_model_sha256=initial_sha,model_metadata=model.metadata(),
            initialization="same fresh HEAD260914 as formal; never carry smoke weights into training",
            sampling="first32 of original data seed260913 / absolute epoch13 permutation",
            implementation_sha256=dict(smoke=data.sha(__file__),**train.implementation_binding()),
            execution_device=str(device),torch_version=str(torch.__version__),
            started_unix=time.time(),real_ood_used=False,validation_performed=False)
        train.save(root/"protocol.json",protocol)
        receipts=[]
        try:
            if device.type=="cuda":
                torch.cuda.reset_peak_memory_stats(device)
                protocol["gpu_name"]=torch.cuda.get_device_name(device)
            model=model.to(device=device,dtype=torch.float32)
            indices=train.runner.epoch_indices(len(dataset),seed=train.DATA_SEED,epoch=13,limit=None)[:COUNT]
            result=two_steps(model,dataset,indices,device,receipts=receipts)
            result.update(memory_stats(device))
            # Reload read-only manifests/array headers to detect accidental
            # source modification. Base cache lacks full-array SHA; its existing
            # size/mtime binding is preserved, not overstated as content proof.
            source_after=load_dataset(args)
            if source_after.binding!=dataset.binding:
                raise RuntimeError("smoke source cache binding changed")
            result.update(status="complete",initial_model_sha256=initial_sha,
                disposable_final_model_sha256=train.state_digest(model),source_binding_unchanged=True,
                weights_and_optimizer_discarded=True,
                output_policy="JSON only; no weights, optimizer, resume checkpoint or formal progress")
            protocol.update(result)
            train.save(root/"results.json",result)
            train.save(root/"protocol.json",protocol)
            return result
        except BaseException as error:
            protocol.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
                error=repr(error),completed_disposable_steps=len(receipts),steps=summarize_steps(receipts),
                formal_pair_exposures=0,formal_optimizer_updates=0)
            train.save(root/"protocol.json",protocol)
            raise
        finally:
            del model


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=train.ARMS,required=True)
    p.add_argument("--train-cache",required=True)
    p.add_argument("--stage-cache",help="complete TRAIN stage side-cache for edge_seed/edge_multi")
    p.add_argument("--output",required=True)
    p.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    return p


if __name__=="__main__":
    print(json.dumps(run(parser().parse_args()),sort_keys=True))

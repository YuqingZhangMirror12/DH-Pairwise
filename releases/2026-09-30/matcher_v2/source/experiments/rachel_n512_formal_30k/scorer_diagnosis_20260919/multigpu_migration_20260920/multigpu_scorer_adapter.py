"""Tensor-only four-GPU runtime adapter for the existing edge_multi Scorer.

The original base object owns parameters/checkpoints/AdamW. This module changes
execution topology only; callers own GPU leases, boundary resume and receipts.
It does not bypass the original trainer's one-visible-GPU entrypoint guard.
"""
from copy import deepcopy
import time

import torch
from torch import nn

try:
    from matched_only import train as original, stage_cache
    from matched_only.model import CandidateSelection
except ModuleNotFoundError as error:
    if error.name != "matched_only":
        raise
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only import train as original, stage_cache
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.model import CandidateSelection

GLOBAL_BATCH = 16
WORLD_SIZE = 4
SELECTION_FIELDS = ("mask_a", "mask_b", "layout_valid", "translation_a_to_b_rc",
                    "candidate_indices", "candidate_valid", "candidate_inliers")
GROUP_FIELDS = ("candidate_inliers", "translation_rc", "present", "eligible", "ranks",
                "seed_candidate_id", "inlier_count", "status_code")


def require_base(base):
    if not isinstance(base, original.CandidateStageScorer) or base.arm != "edge_multi":
        raise ValueError("only the original edge_multi CandidateStageScorer is supported")


def flatten_batch(batch):
    """22 batch-leading tensors; labels/training_valid NEVER enter the head."""
    if len(batch.model_args) != 5:
        raise ValueError("expected four feature/valid tensors plus CandidateSelection")
    a, b, va, vb, selection = batch.model_args
    groups = batch.model_kwargs["groups"]
    if not isinstance(selection, CandidateSelection) or not isinstance(groups, stage_cache.StageGroups) or groups.stage != "edge_multi":
        raise ValueError("requires exact original selection and edge_multi StageGroups")
    keys = {"groups", "candidate_weights", "points_a_rc", "points_b_rc"}
    if set(batch.model_kwargs) != keys:
        raise ValueError("unexpected cached scorer kwargs")
    result = (a, b, va, vb, *(getattr(selection, k) for k in SELECTION_FIELDS),
              *(batch.model_kwargs[k] for k in ("candidate_weights", "points_a_rc", "points_b_rc")),
              *(getattr(groups, k) for k in GROUP_FIELDS))
    if len(result) != 22 or any(not isinstance(v, torch.Tensor) or v.ndim < 1 or len(v) != len(a) for v in result):
        raise ValueError("all scattered values must be tensors with the same leading batch dimension")
    return result


def unflatten_tensors(values):
    if len(values) != 22 or any(not isinstance(v, torch.Tensor) for v in values):
        raise ValueError("expected exactly22 tensors")
    count = len(values[0])
    if any(v.ndim < 1 or len(v) != count for v in values):
        raise ValueError("scattered leading dimensions differ")
    # Reasons are non-numeric diagnostics, not model inputs. The source head
    # validates their shape/type but does not use their text for scoring.
    selection = CandidateSelection(*values[4:11], tuple("parallel_cached_selection" for _ in range(count)))
    groups = stage_cache.StageGroups("edge_multi", *values[14:22])
    return (*values[:4], selection), dict(candidate_weights=values[11],
        points_a_rc=values[12], points_b_rc=values[13], groups=groups)


class TensorOnlyEdgeMultiAdapter(nn.Module):
    def __init__(self, base):
        super().__init__()
        require_base(base)
        self.base = base

    def forward(self, *values):
        args, kwargs = unflatten_tensors(values)
        result = self.base(*args, **kwargs)
        return result.logit, result.used_fallback


def validate_devices(device_ids):
    ids = tuple(device_ids)
    if len(ids) != WORLD_SIZE or len(set(ids)) != WORLD_SIZE or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("exactly four distinct nonnegative logical GPU indices required")
    return ids


def make_parallel(base, device_ids=(0, 1, 2, 3)):
    require_base(base)
    ids = validate_devices(device_ids)
    if not torch.cuda.is_available() or max(ids) >= torch.cuda.device_count():
        raise RuntimeError("four assigned visible CUDA devices required")
    primary = torch.device("cuda", ids[0])
    if any(p.device != primary or p.dtype != torch.float32 for p in base.parameters()):
        raise ValueError("original base must already be FP32 on primary GPU")
    return nn.DataParallel(TensorOnlyEdgeMultiAdapter(base), device_ids=list(ids), output_device=ids[0], dim=0)


def require_optimizer(base, optimizer):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("preserve original AdamW")
    actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
    expected = [id(p) for p in base.parameters()]
    if actual != expected or len(optimizer.param_groups) != 1:
        raise ValueError("optimizer must own original base parameters in original order")


def move_tensors(values, device):
    return tuple(v.to(device) for v in values)


def optimization_step(base, forward, tensors, labels, training_valid, optimizer, *, capture_gradients=False):
    """One GLOBAL update, also usable in a disposable CPU equivalence test."""
    require_base(base)
    require_optimizer(base, optimizer)
    if len(labels) != GLOBAL_BATCH or len(tensors[0]) != GLOBAL_BATCH:
        raise ValueError("each update must contain the original global16 rows")
    optimizer.zero_grad(set_to_none=True)
    logits, fallback = forward(*tensors)
    labels, training_valid = labels.to(logits.device), training_valid.to(logits.device)
    loss = original.pair_loss(logits, labels, training_valid)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite parallel scorer loss")
    loss.backward()
    gradients = ({name: None if p.grad is None else p.grad.detach().clone()
                  for name, p in base.named_parameters()} if capture_gradients else None)
    norm = torch.nn.utils.clip_grad_norm_(base.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=loss.detach(), fallback=fallback.detach(), logits=logits.detach(),
                preclip_gradient_norm=norm.detach(), preclip_gradients=gradients)


def train_segment(base, dataset, indices, optimizer, device, *, parallel=None, device_ids=(0, 1, 2, 3)):
    """Replacement for original.train_segment; no checkpoint/source changes."""
    ids = validate_devices(device_ids)
    device = torch.device(device)
    if device != torch.device("cuda", ids[0]) or not len(indices) or len(indices) % GLOBAL_BATCH:
        raise ValueError("primary CUDA device and full global16 batches required")
    require_base(base)
    require_optimizer(base, optimizer)
    parallel = make_parallel(base, ids) if parallel is None else parallel
    if (not isinstance(parallel, nn.DataParallel) or not isinstance(parallel.module, TensorOnlyEdgeMultiAdapter)
            or parallel.module.base is not base or tuple(parallel.device_ids) != ids):
        raise ValueError("parallel adapter must own this original base and assigned four GPUs")
    base.train(); parallel.train()
    started = time.perf_counter()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    valid_count = fallback_count = 0
    for offset in range(0, len(indices), GLOBAL_BATCH):
        batch = dataset.batch(indices[offset:offset+GLOBAL_BATCH], "cpu")
        result = optimization_step(base, parallel, flatten_batch(batch), batch.labels, batch.training_valid, optimizer)
        loss_sum += result["loss"].double() * GLOBAL_BATCH
        valid_count += int(batch.training_valid.sum())
        fallback_count += int(result["fallback"].sum())
    optimizer.zero_grad(set_to_none=True)
    return dict(samples=len(indices), optimizer_updates=len(indices)//GLOBAL_BATCH,
        mean_loss=float(loss_sum.cpu())/len(indices), training_valid_count=valid_count,
        fallback_count=fallback_count, elapsed_s=time.perf_counter()-started,
        physical_microbatch=4, per_gpu_microbatch=4, global_microbatch=16,
        effective_batch=16, accumulation_steps=1, parallel_world_size=4,
        runtime="tensor-only DataParallel; original base/AdamW; not bitwise single-GPU equivalent")


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return deepcopy(value)


def tree_difference(left, right):
    """Exact structure/None checks plus tensor absolute error, no scalar masking."""
    result = dict(maximum_absolute_error=0., maximum_reference_magnitude=0., worst_path=None,
                  structure_mismatches=[], nonfinite_paths=[])
    def walk(a,b,path):
        if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                result["structure_mismatches"].append(path); return
            x,y=a.detach().cpu().double(),b.detach().cpu().double()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                result["nonfinite_paths"].append(path); return
            error=float((x-y).abs().max()) if x.numel() else 0.
            magnitude=float(x.abs().max()) if x.numel() else 0.
            result["maximum_reference_magnitude"]=max(result["maximum_reference_magnitude"],magnitude)
            if error>result["maximum_absolute_error"]:
                result.update(maximum_absolute_error=error,worst_path=path)
        elif isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys():
            for key in a: walk(a[key],b[key],path+"/"+str(key))
        elif isinstance(a,(tuple,list)) and type(a)==type(b) and len(a)==len(b):
            for i,(x,y) in enumerate(zip(a,b)): walk(x,y,path+"/"+str(i))
        elif type(a)!=type(b) or a!=b:
            result["structure_mismatches"].append(path)
    walk(left,right,"root")
    result["maximum_error_over_global_reference_scale"]=result["maximum_absolute_error"]/max(1e-12,result["maximum_reference_magnitude"])
    return result


def benchmark(base, optimizer, dataset, indices, *, device_ids=(0,1,2,3)):
    """Disposable comparison: warmup1 + parity1 + timed4 batches PER topology.

    Exactly64 supplied rows, not score-selected. Original weights, Adam, grads
    and RNG remain unchanged. No file writes. Copies reset after warmup/parity;
    return all errors, not a fabricated equivalence/speedup pass threshold.
    Caller must lease all GPUs before calling. Timing includes H2D transfer and
    excludes cache read, snapshot restore and numerical comparison operations.
    """
    ids=validate_devices(device_ids)
    require_base(base); require_optimizer(base,optimizer)
    if len(indices)!=64 or len(set(indices))!=64:
        raise ValueError("benchmark requires exactly64 distinct predetermined rows")
    device=torch.device("cuda",ids[0])
    state,adam=cpu_tree(base.state_dict()),cpu_tree(optimizer.state_dict())
    original_gradients=cpu_tree({n:p.grad for n,p in base.named_parameters()})
    original_mode=base.training
    rng=original.capture_rng_state()
    copies=[]
    try:
        batches=[dataset.batch(indices[i:i+16],"cpu") for i in range(0,64,16)]
        flats=[flatten_batch(batch) for batch in batches]
        for _ in range(2):
            copied=deepcopy(base).to(device=device,dtype=torch.float32).train()
            copied.load_state_dict(state,strict=True)
            opt=original.create_optimizer(copied); opt.load_state_dict(deepcopy(adam))
            opt.zero_grad(set_to_none=True)
            copies.append((copied,opt))
        forwards=[TensorOnlyEdgeMultiAdapter(copies[0][0]),make_parallel(copies[1][0],ids)]
        def sync():
            for index in ids: torch.cuda.synchronize(index)
        def reset(i):
            model,opt=copies[i];model.load_state_dict(state,strict=True)
            opt.load_state_dict(deepcopy(adam));opt.zero_grad(set_to_none=True)
        def step(i,j,gradients=False):
            batch=batches[j]
            values=move_tensors(flats[j],device) if i==0 else flats[j]
            return optimization_step(copies[i][0],forwards[i],values,batch.labels,batch.training_valid,
                                     copies[i][1],capture_gradients=gradients)
        for i in range(2):
            step(i,0);sync();reset(i)
        parity=[step(i,0,True) for i in range(2)]
        sync()
        comparison=dict(logits=tree_difference(parity[0]["logits"],parity[1]["logits"]),
            loss=tree_difference(parity[0]["loss"],parity[1]["loss"]),
            preclip_gradients=tree_difference(parity[0]["preclip_gradients"],parity[1]["preclip_gradients"]),
            parameters_after_one_update=tree_difference(copies[0][0].state_dict(),copies[1][0].state_dict()),
            adam_after_one_update=tree_difference(copies[0][1].state_dict(),copies[1][1].state_dict()))
        timings=[]
        for i in range(2):
            reset(i);sync();started=time.perf_counter()
            results=[step(i,j) for j in range(4)]
            sync();elapsed=time.perf_counter()-started
            timings.append(dict(topology="single_gpu" if i==0 else "four_gpu_dp",elapsed_s=elapsed,
                pairs=64,optimizer_updates=4,pairs_per_second=64/elapsed,
                losses=[float(v["loss"].cpu()) for v in results]))
        comparison.update(parameters_after_four_updates=tree_difference(copies[0][0].state_dict(),copies[1][0].state_dict()),
            adam_after_four_updates=tree_difference(copies[0][1].state_dict(),copies[1][1].state_dict()))
        unchanged=dict(parameters=tree_difference(state,base.state_dict()),optimizer=tree_difference(adam,optimizer.state_dict()),
            gradients=tree_difference(original_gradients,{n:p.grad for n,p in base.named_parameters()}),mode_unchanged=base.training==original_mode)
        return dict(schema="edge-multi-four-gpu-discard-benchmark/1",status="complete_discarded",formal_training_counted=False,
            original_unchanged=unchanged,global_batch=16,per_gpu_batch=4,gpus=list(ids),timing=timings,
            speedup=timings[0]["elapsed_s"]/timings[1]["elapsed_s"],comparison=comparison,
            discarded_updates_per_topology=6,discarded_pair_exposures_per_topology=96,
            budget_note="warmup1 and parity1 reset before timed4; all copies discarded",checkpoint_written=False,
            caveat="Floating-point reduction differs; reported errors are measurements, not bitwise equivalence.")
    finally:
        original.restore_rng_state(rng)

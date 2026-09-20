"""Isolated finite G0/G1 trainer; import is inert and no scheduler is included.

Original24K ordinary16 BCE plus a separately counted same-anchor ranking group
on1470/1500 updates. No auxiliary BCE, cached feature reuse or Matcher forward.
Only cleanSIMVAL is evaluated/calibrated. Each6000 ordinary pairs is an atomic
Adam/RNG recovery commit. Formal CLI and discard32 require the shared GPU lock.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from . import data as auxiliary_data, model as architecture
from ..matched_only import cache as source_cache, data as cache_data, train as prior
from ..pair_grid_readout import model as grid_readout
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedRachelDataset
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset, collate_rachel_pairs
from experiments.rachel_n512_formal_30k import train_score_decoupled as source_training

SCHEMA = "s7-m12-scorer-feature-adaptation-training/1"
HEAD_SEED, DATA_SEED, AUX_SEED = 260914, 260913, 260929
BATCH, SEGMENT_SIZE, HEAD_EPOCHS = 16, 6000, 16
TRAIN_COUNT, VAL_COUNT, AUX_GROUPS = 24000, 3000, 1470
SEGMENTS_PER_EPOCH, TOTAL_SEGMENTS = 4, 64
RANK_WEIGHT, MARGIN = .3, .15
save, learning_rate, pair_loss = prior.save, prior.learning_rate, prior.pair_loss
capture_rng_state, restore_rng_state = prior.capture_rng_state, prior.restore_rng_state
state_digest, runner = prior.state_digest, prior.runner


def plan():
    return [dict(number=(e-1)*SEGMENTS_PER_EPOCH+s+1, head_epoch=e, absolute_epoch=12+e,
        offset=s*SEGMENT_SIZE, step_offset=s*SEGMENT_SIZE//BATCH, count=SEGMENT_SIZE,
        epoch_complete=s==SEGMENTS_PER_EPOCH-1, head_lr=learning_rate(e))
        for e in range(1, HEAD_EPOCHS+1) for s in range(SEGMENTS_PER_EPOCH)]


def auxiliary_schedule(head_epoch, group_count, *, updates=None):
    updates = TRAIN_COUNT//BATCH if updates is None else updates
    if type(head_epoch) is not int or not 1 <= head_epoch <= HEAD_EPOCHS or not 0 < group_count <= updates:
        raise ValueError("invalid auxiliary epoch/population budget")
    rng = np.random.default_rng(AUX_SEED + 1_000_003*head_epoch)
    slots, groups = rng.permutation(updates), rng.permutation(group_count)
    result = [None]*updates
    for slot, group in zip(slots, groups):
        result[int(slot)] = int(group)
    return tuple(result)


def exposure_ledger(completed_segments, negative_counts):
    if type(completed_segments) is not int or not 0 <= completed_segments <= TOTAL_SEGMENTS:
        raise ValueError("invalid committed segment count")
    groups = pairs = no_aux = 0
    for segment in plan()[:completed_segments]:
        schedule = auxiliary_schedule(segment["head_epoch"], len(negative_counts))
        selected = schedule[segment["step_offset"]:segment["step_offset"]+SEGMENT_SIZE//BATCH]
        for index in selected:
            if index is None:
                no_aux += 1
            else:
                groups += 1
                pairs += 1 + negative_counts[index]
    ordinary = completed_segments*SEGMENT_SIZE
    return dict(ordinary_pairs=ordinary, auxiliary_groups=groups, auxiliary_pairs=pairs,
        pair_forwards=ordinary+pairs, no_aux_updates=no_aux, optimizer_updates=ordinary//BATCH)


@dataclass(frozen=True)
class RawBatch:
    inputs: tuple
    labels: torch.Tensor
    training_valid: torch.Tensor
    decision_valid: torch.Tensor
    pair_ids: tuple


class RawPopulation:
    """Fresh six-input dataset + frozen Matcher validity, never cached features.

    FormalCache performs its existing complete/source/header checks once. Only
    small bool arrays and identity records are copied, then its memmaps can be
    released. Its batch() API is NEVER called. Each pair's freshly loaded six
    inputs are fingerprint-checked on first use in this process.
    """
    def __init__(self, dataset, validity_cache, manifest_path):
        self.dataset, self.split = dataset, validity_cache.split
        self.records = deepcopy(validity_cache.records)
        if len(dataset) != len(self.records) or dataset.split != self.split:
            raise ValueError("raw/cache populations differ")
        if cache_data.sha(manifest_path) != validity_cache.binding["population"]["manifest_sha256"]:
            raise ValueError("raw manifest differs from completed Matcher cache")
        self.lookup = {r["pair_id"]:i for i,r in enumerate(self.records)}
        self.training_valid = np.array(validity_cache.arrays["training_valid"], copy=True)
        self.decision_valid = np.array(validity_cache.arrays["decision_valid"], copy=True)
        if self.training_valid.dtype != np.bool_ or self.decision_valid.dtype != np.bool_:
            raise ValueError("validity cache arrays must be bool")
        self.checked = set()
        self.binding = dict(raw_manifest=str(Path(manifest_path).resolve()),
            cache=deepcopy(validity_cache.binding), cached_features_used=False,
            training_valid_sha256=hashlib.sha256(self.training_valid.tobytes()).hexdigest(),
            decision_valid_sha256=hashlib.sha256(self.decision_valid.tobytes()).hexdigest())

    def __len__(self):
        return len(self.dataset)

    def batch(self, indices, device="cpu"):
        samples = [self.dataset[int(i)] for i in indices]
        batch = collate_rachel_pairs(samples, contour_cap=512)
        arrays = tuple(np.asarray(getattr(batch, k), dtype=np.bool_ if k.startswith("contour_valid")
                                  else np.float32) for k in source_cache.INPUTS)
        cache_indices = []
        for row, (pair_id, label) in enumerate(zip(batch.pair_ids, batch.labels)):
            if pair_id not in self.lookup:
                raise ValueError("loaded raw pair absent from frozen Matcher cache")
            i = self.lookup[pair_id]
            if float(label) != self.records[i]["label"]:
                raise ValueError("raw/cache pair label differs")
            if pair_id not in self.checked:
                fingerprint = hashlib.sha256()
                for name, value in zip(source_cache.INPUTS, arrays):
                    fingerprint.update(name.encode()); fingerprint.update(value[row].tobytes())
                if fingerprint.hexdigest() != self.records[i].get("input_sha256"):
                    raise ValueError("raw/cache six-input fingerprint differs: " + pair_id)
                self.checked.add(pair_id)
            cache_indices.append(i)
        def tensor(value):
            return torch.from_numpy(np.array(value, copy=True)).to(device)
        return RawBatch(tuple(tensor(a) for a in arrays), tensor(batch.labels),
            tensor(self.training_valid[cache_indices]), tensor(self.decision_valid[cache_indices]), batch.pair_ids)


def optimizer_groups(model):
    groups = [("head", model.score_head, 1.)]
    if model.feature_trainable:
        groups.append(("stem", model.scorer_stem, .1))
    names = {id(p):n for n,p in model.named_parameters()}
    result = []
    for name, module, multiplier in groups:
        params = [p for p in module.parameters() if p.requires_grad]
        result.append(dict(name=name, lr_scale=multiplier, params=params,
                           parameter_names=[names[id(p)] for p in params]))
    included = {id(p) for g in result for p in g["params"]}
    if included != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("optimizer must contain exactly trainable Scorer parameters")
    return result


def optimizer_schema(model):
    return [dict(name=g["name"], lr_scale=g["lr_scale"], parameter_names=g["parameter_names"],
                 shapes=[list(p.shape) for p in g["params"]]) for g in optimizer_groups(model)]


def create_optimizer(model):
    groups = optimizer_groups(model)
    for group in groups:
        group["lr"] = learning_rate(1)*group["lr_scale"]
    return torch.optim.AdamW(groups, lr=learning_rate(1), weight_decay=1e-4)


def set_learning_rate(optimizer, epoch):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(epoch)*group["lr_scale"]


def raw_group_rank(raw_similarity, group):
    """Rank only: never calculate/backpropagate auxiliary PairBCE or sigmoid."""
    n = len(group.labels)
    edges = torch.tensor([[0,i] for i in range(1,n)], dtype=torch.long)
    if (not 2 <= n <= 4 or raw_similarity.shape != (n,) or not torch.isfinite(raw_similarity).all()
            or group.same_anchor_confirmed is not True or len(set(group.anchor_ids)) != 1
            or not torch.equal(group.labels.detach().cpu(), torch.tensor([1.]+[0.]*(n-1)))
            or not torch.equal(group.known_negative_pairs.detach().cpu(), edges)):
        raise ValueError("one positive and confirmed same-anchor negative group required")
    return F.relu(MARGIN - raw_similarity[0] + raw_similarity[1:].max())


def train_segment(model, training, auxiliary, indices, slots, optimizer, device):
    if not len(indices) or len(indices)%BATCH or len(slots) != len(indices)//BATCH:
        raise ValueError("segment must align ordinary full batches and global auxiliary slots")
    model.train()
    started, bce_sum, rank_sum, valid_count, aux_pairs, aux_groups = time.perf_counter(), 0., 0., 0, 0, 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    for step, start in enumerate(range(0,len(indices),BATCH)):
        batch = training.batch(indices[start:start+BATCH], device)
        optimizer.zero_grad(set_to_none=True)
        ordinary = model.scorer_forward(*batch.inputs)
        bce = pair_loss(ordinary.calibrated_logit, batch.labels, batch.training_valid)
        if not torch.isfinite(bce):
            raise FloatingPointError("nonfinite ordinary PairBCE")
        bce.backward()  # Free ordinary graph before the auxiliary feature pass.
        bce_sum += float(bce.detach())*len(batch.labels)
        valid_count += int(batch.training_valid.sum())
        del ordinary, bce
        if slots[step] is not None:
            group = auxiliary[int(slots[step])]
            result = model.scorer_forward(*(x.to(device) for x in group.inputs))
            rank = raw_group_rank(result.raw_similarity, group)
            (RANK_WEIGHT*rank).backward()
            rank_sum += float(rank.detach()); aux_groups += 1; aux_pairs += len(group.labels)
            del result, rank, group
        torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(ordinary_pairs=len(indices), auxiliary_groups=aux_groups, auxiliary_pairs=aux_pairs,
        pair_forwards=len(indices)+aux_pairs, optimizer_updates=len(slots), no_aux_updates=len(slots)-aux_groups,
        ordinary_mean_pair_bce=bce_sum/len(indices), rank_sum=rank_sum,
        mean_raw_rank_over_groups=rank_sum/max(1,aux_groups),
        mean_objective_per_update=bce_sum/len(indices)+RANK_WEIGHT*rank_sum/len(slots),
        ordinary_training_valid_count=valid_count, ordinary_physical_batch=BATCH,
        auxiliary_batch="separate2/3pairs per scheduled group; no auxiliary BCE; not effective batch16",
        elapsed_s=time.perf_counter()-started)


@torch.no_grad()
def evaluate(model, validation, device):
    from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
    model.eval()
    rows, loss_sum = [], 0.
    for start in range(0,len(validation),BATCH):
        batch = validation.batch(range(start,min(start+BATCH,len(validation))), device)
        result = model.scorer_forward(*batch.inputs)
        loss_sum += float(pair_loss(result.calibrated_logit,batch.labels,batch.training_valid))*len(batch.labels)
        logit = torch.where(batch.training_valid,result.calibrated_logit,torch.zeros_like(result.calibrated_logit))
        values = [x.cpu().tolist() for x in (batch.labels,logit,logit.sigmoid(),result.calibrated_logit,
                                            result.raw_similarity,batch.training_valid,batch.decision_valid)]
        for i,pair_id in enumerate(batch.pair_ids):
            rows.append(dict(zip(("pair_id","label","logit","score","raw_head_logit","raw_similarity",
                                  "training_valid","decision_valid"), [pair_id]+[x[i] for x in values])))
    labels,scores = [r["label"] for r in rows],[r["score"] for r in rows]
    op = fit_operating_points(labels,scores)
    best = op["validation"]["max_f1"]
    return dict(sample_count=len(rows),positive_count=int(sum(labels)),mean_loss=loss_sum/len(rows),
        training_valid_count=sum(r["training_valid"] for r in rows),
        decision_valid_count=sum(r["decision_valid"] for r in rows),operating_points=op,
        max_f1_selection_key=[best["f1"],best["auprc"]],recall95_selection_key=op["selection_key"],
        score_policy="fresh Scorer features; fixed M12 !training_valid logit0/probability.5; allSIMVAL calibrated"),rows


def frozen_digests(model):
    result = dict(base=state_digest(model.base_model))
    if not model.feature_trainable:
        result["stem"] = state_digest(model.scorer_stem)
    return result


def parameter_steps(model, optimizer):
    return {name:float(optimizer.state[p]["step"]) if p in optimizer.state and "step" in optimizer.state[p] else None
            for name,p in model.named_parameters() if p.requires_grad}


def checkpoint_payload(model, optimizer, ident, completed, winners, role):
    if frozen_digests(model) != ident["frozen_digests"]:
        raise ValueError("frozen Matcher/G0 stem changed")
    return dict(schema=SCHEMA,identity=ident,model_metadata=model.metadata(),
        model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer_state_dict=optimizer.state_dict(),optimizer_schema=optimizer_schema(model),
        optimizer_parameter_steps=parameter_steps(model,optimizer),rng_state=capture_rng_state(),
        completed_segments=completed,exposures=exposure_ledger(completed,ident["negative_counts"]),
        winners=deepcopy(winners),role=role,matcher_updated=False,phase="classifier",formal_training_counted=True)


def restore(model,optimizer,saved,ident):
    completed = saved.get("completed_segments")
    if (saved.get("schema") != SCHEMA or saved.get("identity") != ident
            or saved.get("model_metadata") != model.metadata() or saved.get("matcher_updated") is not False
            or saved.get("formal_training_counted") is not True or saved.get("phase") != "classifier"
            or saved.get("optimizer_schema") != optimizer_schema(model)
            or saved.get("exposures") != exposure_ledger(completed,ident["negative_counts"])
            or set(saved.get("rng_state",{})) != {"python","numpy","torch","cuda"}):
        raise ValueError("resume source/progress/optimizer/RNG identity differs")
    model.load_state_dict(saved["model_state_dict"],strict=True)
    if frozen_digests(model) != ident["frozen_digests"]:
        raise ValueError("restored frozen Matcher/G0 stem differs")
    expected_ids=[g["params"] for g in optimizer.state_dict()["param_groups"]]
    if [g["params"] for g in saved["optimizer_state_dict"]["param_groups"]] != expected_ids:
        raise ValueError("Adam serialized parameter order differs")
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    schema = optimizer_schema(model)
    if len(optimizer.param_groups) != len(schema):
        raise ValueError("Adam group count differs")
    epoch = max(1,(completed+SEGMENTS_PER_EPOCH-1)//SEGMENTS_PER_EPOCH)
    for group,expected in zip(optimizer.param_groups,schema):
        if (any(group.get(k) != expected[k] for k in ("name","lr_scale","parameter_names"))
                or group["lr"] != learning_rate(epoch)*expected["lr_scale"]
                or group["weight_decay"] != 1e-4 or tuple(group["betas"]) != (.9,.999) or group["eps"] != 1e-8):
            raise ValueError("Adam group order/names/hyperparameters differ")
    updates = saved["exposures"]["optimizer_updates"]
    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise ValueError("nonfinite saved parameter")
        state = optimizer.state.get(p)
        if state is not None and (not p.requires_grad or not {"step","exp_avg","exp_avg_sq"} <= set(state)
                or not 0 < float(state["step"]) <= updates or float(state["step"]) != int(state["step"])
                or any(state[k].shape != p.shape or not torch.isfinite(state[k]).all() for k in ("exp_avg","exp_avg_sq"))):
            raise ValueError("invalid Adam state/step/moments")
    if ((updates==0 and optimizer.state) or (updates>0 and not optimizer.state)
            or saved.get("optimizer_parameter_steps") != parameter_steps(model,optimizer)):
        raise ValueError("Adam parameter-step ledger differs")
    restore_rng_state(saved["rng_state"])
    return completed,deepcopy(saved["winners"])


def freeze_endpoint(root,ident,epoch,winners,report):
    selected = dict(fixed_endpoint=dict(head_epoch=epoch,absolute_epoch=12+epoch,
        checkpoint="head_epoch_%03d.pt"%epoch),**deepcopy(winners))
    for name,row in selected.items():
        path = root/("validation_head_%03d.json"%row["head_epoch"])
        val = json.loads(path.read_text())
        row.update(validation=path.name,validation_sha256=cache_data.sha(path),
            checkpoint_sha256=cache_data.sha(root/row["checkpoint"]),operating_points=val["operating_points"],
            primary_pair_threshold=val["operating_points"]["thresholds"]["recall_95" if name=="recall95" else "max_f1"])
    save(root/"freezes"/("c%d.json"%epoch),dict(schema=SCHEMA,status="complete_endpoint",identity=ident,
        budget_head_epochs=epoch,primary=epoch==16,primary_selection="fixed_endpoint",selections=selected,
        selection_population="clean SIMVAL3000 only",real_ood_used=False))


def execute(args,model,optimizer,training,validation,auxiliary,ident,device):
    root = Path(args.output)
    completed,winners = 0,{}
    target = args.stop_after_head_epoch*SEGMENTS_PER_EPOCH
    if args.resume:
        completed,winners = restore(model,optimizer,torch.load(root/"last.pt",map_location="cpu",weights_only=False),ident)
        if completed>target:
            raise ValueError("cannot resume backwards")
    elif (root/"last.pt").exists():
        raise ValueError("refusing existing checkpoint")
    protocol = dict(schema=SCHEMA,status="running",identity=ident,arguments=vars(args),
        completed_segments=completed,plan=plan(),real_ood_used=False,
        exposures=exposure_ledger(completed,ident["negative_counts"]))
    save(root/"protocol.json",protocol)
    if not args.resume:
        runner._atomic_torch_save(root/"last.pt",checkpoint_payload(model,optimizer,ident,0,{},"initial"))
    try:
        for endpoint in (8,16):
            if completed>=endpoint*SEGMENTS_PER_EPOCH:
                payload=torch.load(root/("head_epoch_%03d.pt"%endpoint),map_location="cpu",weights_only=False)
                report=json.loads((root/("validation_head_%03d.json"%endpoint)).read_text())
                if payload["identity"] != ident or payload["completed_segments"] != endpoint*SEGMENTS_PER_EPOCH:
                    raise ValueError("committed endpoint identity differs")
                freeze_endpoint(root,ident,endpoint,payload["winners"],report)
        for segment in plan():
            number,epoch = segment["number"],segment["head_epoch"]
            if number<=completed or epoch>args.stop_after_head_epoch:
                continue
            set_learning_rate(optimizer,epoch)
            order=runner.epoch_indices(len(training),seed=DATA_SEED,epoch=12+epoch,limit=None)
            indices=order[segment["offset"]:segment["offset"]+SEGMENT_SIZE]
            schedule=auxiliary_schedule(epoch,len(auxiliary))
            slots=schedule[segment["step_offset"]:segment["step_offset"]+SEGMENT_SIZE//BATCH]
            train_report=train_segment(model,training,auxiliary,indices,slots,optimizer,device)
            before=exposure_ledger(completed,ident["negative_counts"])
            after=exposure_ledger(number,ident["negative_counts"])
            if any(train_report[k] != after[k]-before[k] for k in before):
                raise ValueError("ordinary/auxiliary exposure ledger differs")
            save(root/("segment_%03d.json"%number),dict(segment=segment,training=train_report))
            if segment["epoch_complete"]:
                rng=capture_rng_state()
                try:
                    report,rows=evaluate(model,validation,device)
                finally:
                    restore_rng_state(rng)
                save(root/("validation_head_%03d.json"%epoch),report)
                save(root/("validation_head_%03d_rows.json"%epoch),rows)
                winners=prior.update_winners(winners,epoch,report)
            payload=checkpoint_payload(model,optimizer,ident,number,winners,"epoch_anchor" if segment["epoch_complete"] else "segment_recovery")
            if segment["epoch_complete"]:
                runner._atomic_torch_save(root/("head_epoch_%03d.pt"%epoch),payload)
            runner._atomic_torch_save(root/"last.pt",payload)
            completed=number
            if segment["epoch_complete"] and epoch in (8,16):
                freeze_endpoint(root,ident,epoch,winners,report)
            protocol.update(completed_segments=completed,exposures=after,head_epoch=epoch)
            save(root/"protocol.json",protocol)
            save(root/"status.json",dict(status="running",completed_segments=completed,exposures=after))
            if segment["epoch_complete"]:
                print(json.dumps(dict(event="head_epoch_complete",arm=args.arm,head_epoch=epoch,exposures=after)),flush=True)
        if completed != target:
            raise ValueError("incomplete requested endpoint")
        protocol.update(status="complete" if completed==TOTAL_SEGMENTS else "paused",
            completed_head_epochs=completed//SEGMENTS_PER_EPOCH)
        save(root/"protocol.json",protocol);save(root/"status.json",protocol)
        return dict(status=protocol["status"],completed_segments=completed,exposures=protocol["exposures"])
    except BaseException as error:
        protocol.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
            error=repr(error),completed_segments=completed,recovery="explicit resume last.pt; repeat only uncommitted segment")
        save(root/"protocol.json",protocol);save(root/"status.json",protocol)
        raise


def implementation_binding():
    files=dict(train=__file__,data=auxiliary_data.__file__,model=architecture.__file__,
        pair_grid=grid_readout.__file__,prior=prior.__file__,cache=source_cache.__file__,
        cache_validator=cache_data.__file__,source_training=source_training.__file__,
        raw_loader=sys.modules[RachelPairDataset.__module__].__file__,
        materialized_loader=sys.modules[MaterializedRachelDataset.__module__].__file__,
        rachel_model=sys.modules[architecture.RachelN512Pairwise.__module__].__file__,
        ca_model=sys.modules[architecture.CrossAttentionPairHead.__module__].__file__,
        rng=sys.modules[capture_rng_state.__module__].__file__,gpu_lock=prior.lock_owner.__file__)
    return {k:cache_data.sha(v) for k,v in files.items()}


def identity(arm,model,training,validation,auxiliary,availability):
    negative_counts=[len(g["negatives"]) for g in auxiliary.index["groups"]]
    if (len(training),len(validation),len(auxiliary),sum(negative_counts)) != (TRAIN_COUNT,VAL_COUNT,AUX_GROUPS,1558):
        raise ValueError("formal populations differ from verified24000/3000/1470+1558")
    if set(training.lookup)&set(validation.lookup):
        raise ValueError("TRAIN/VAL pair IDs overlap")
    return dict(schema=SCHEMA,arm=arm,model=model.metadata(),initial_state_sha256=state_digest(model),
        frozen_digests=frozen_digests(model),optimizer_schema=optimizer_schema(model),
        source_checkpoint_sha256=source_cache.SOURCE_SHA,source_matcher_epochs=12,
        populations=dict(train=training.binding,val=validation.binding),availability=availability,
        negative_counts=negative_counts,head_seed=HEAD_SEED,data_seed=DATA_SEED,aux_seed=AUX_SEED,
        head_epochs=16,retained_endpoints=[8,16],primary_endpoint=16,segments=64,
        segment_ordinary_pairs=6000,ordinary_physical_batch=16,optimizer_updates_per_epoch=1500,
        ordinary_pairs_per_epoch=24000,auxiliary_groups_per_epoch=1470,auxiliary_pairs_per_epoch=3028,
        training_pair_forwards_per_epoch=27028,no_aux_updates_per_epoch=30,
        effective_batch_claim="ordinary16 PLUS separately backwarded auxiliary2/3pairs; not effective16 overall",
        loss="mean ordinary PairBCE*fixedM12training_valid over16 + .3*single-group rawsim hinge(.15)",
        auxiliary_bce=False,auxiliary_matcher_validity_reused=False,cached_features_used=False,
        optimizer="AdamW",weight_decay=1e-4,clip_grad_norm=5.,precision="fp32",stem_lr_scale=.1,
        head_lr=[learning_rate(e) for e in range(1,17)],real_ood_used=False,
        selection="cleanSIMVAL only; fixed C16 primary, C8 retained; auxiliary maxF1/R95 within budget",
        implementation_sha256=implementation_binding(),
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,python=platform.python_version()))


def discard_smoke(args,model,optimizer,training,validation,auxiliary,ident,device):
    order=runner.epoch_indices(len(training),seed=DATA_SEED,epoch=13,limit=32)
    slots=[i for i in auxiliary_schedule(1,len(auxiliary)) if i is not None][:2]
    before=state_digest(model)
    report=train_segment(model,training,auxiliary,order,slots,optimizer,device)
    model.eval()
    with torch.no_grad():
        batch=validation.batch(range(32),device)
        output=model.scorer_forward(*batch.inputs)
        if not torch.isfinite(output.calibrated_logit).all():
            raise FloatingPointError("smoke validation nonfinite")
    if frozen_digests(model)!=ident["frozen_digests"] or state_digest(model)==before:
        raise ValueError("smoke frozen/trainable-state contract failed")
    result=dict(schema=SCHEMA,status="complete_discard_smoke",arm=args.arm,formal_training_counted=False,
        model_weights_discarded=True,checkpoint_saved=False,identity=ident,training=report,
        val_forward_pairs=32,validation_calibrated=False,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        smoke_schedule="ordinary first32; first2 nonempty auxiliary slots; not a formal segment")
    save(Path(args.output)/"smoke.json",result)
    return result


def run(args):
    if args.smoke not in (None,32) or (args.smoke and args.resume):
        raise ValueError("only new discard32 smoke is allowed")
    root=Path(args.output).resolve()
    with prior.lock_owner.gpu_lock():
        if platform.system()!="Linux" or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError("remote Linux single CUDA GPU required")
        torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
        source=source_cache.source_checkpoint(args.source)
        origin=source["resume_identity"]["populations"]
        manifest=Path(origin["train"]["manifest"])
        val_manifest=Path(origin["val"]["manifest"])
        caches={s:cache_data.FormalCache(getattr(args,s+"_cache"),s) for s in ("train","val")}
        training=RawPopulation(MaterializedRachelDataset(manifest),caches["train"],manifest)
        validation=RawPopulation(RachelPairDataset(val_manifest.parents[1],"val"),caches["val"],val_manifest)
        del caches  # No frozen token/cache-feature object survives into training.
        availability_root=Path(args.availability).resolve(strict=True)
        receipt=json.loads((availability_root/"summary.json").read_text())
        if receipt.get("status")!="complete" or receipt.get("manifest_sha256")!=source_cache.TRAIN_SHA:
            raise ValueError("verified auxiliary availability receipt required")
        for name,digest in receipt["artifacts"].items():
            if cache_data.sha(availability_root/name)!=digest:
                raise ValueError("auxiliary source artifact SHA changed")
        ledger=json.loads((availability_root/"scale_ledger.json").read_text())
        auxiliary=auxiliary_data.SameAnchorTripletDataset(manifest,scale_ledger=ledger)
        if auxiliary.index!=json.loads((availability_root/"triplet_index.json").read_text()):
            raise ValueError("reconstructed auxiliary index differs from verified receipt")
        source_model=source_training.load_decoupled_checkpoint(source)
        runner._set_determinism(HEAD_SEED)
        g0,g1=architecture.build_g0_g1(source_model)
        model=g0 if args.arm=="G0" else g1
        del source,source_model,g0,g1
        ident=identity(args.arm,model,training,validation,auxiliary,
            dict(root=str(availability_root),summary_sha256=cache_data.sha(availability_root/"summary.json"),
                 artifacts=receipt["artifacts"]))
        for input_path in (manifest.parent,val_manifest.parent,Path(args.source).resolve().parent,
                           Path(args.train_cache).resolve(),Path(args.val_cache).resolve(),availability_root):
            if root==input_path or input_path in root.parents or root in input_path.parents:
                raise ValueError("output must be separate from source/input/cache trees")
        if args.resume:
            if not root.is_dir():raise ValueError("resume output missing")
        else:
            root.mkdir(parents=True,exist_ok=False)
        args.output=str(root)
        device=torch.device(args.device)
        model.to(device=device,dtype=torch.float32)
        optimizer=create_optimizer(model)
        with prior.run_lock(root):
            if args.smoke:
                return discard_smoke(args,model,optimizer,training,validation,auxiliary,ident,device)
            return execute(args,model,optimizer,training,validation,auxiliary,ident,device)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=("G0","G1"),required=True)
    p.add_argument("--source",default=str(source_cache.SOURCE))
    p.add_argument("--train-cache",required=True);p.add_argument("--val-cache",required=True)
    p.add_argument("--availability",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    p.add_argument("--stop-after-head-epoch",type=int,choices=(8,16),default=16)
    p.add_argument("--resume",action="store_true");p.add_argument("--smoke",type=int,choices=(32,))
    return p


if __name__=="__main__":
    result=run(parser().parse_args())
    print(json.dumps({k:result[k] for k in result if k not in ("identity",)},sort_keys=True))

"""Finite fresh-C16 experiment; frozen Matcher cache, batch48, per-GPU lease."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from ..matched_only import cache, data as cached, train as prior
from .runtime import lease
from .model import CONFIGS, make
from .data import TrainingEvidence

SCHEMA = "local-evidence-v2-training/1"
BATCH, EPOCHS, SEGMENT = 48, 16, 6000
SEED = 260914
save = prior.save


def training_loss(output, batch, evidence=None, indices=None):
    pair = prior.pair_loss(output.logit, batch.labels, batch.training_valid)
    auxiliary = pair.new_zeros(())
    if evidence is not None:
        target = torch.as_tensor(evidence.correct[indices], dtype=output.logit.dtype, device=output.logit.device)
        known = torch.as_tensor(evidence.known[indices], dtype=torch.bool, device=output.logit.device)
        known = known & batch.training_valid & output.has_decoded_candidate
        losses = F.binary_cross_entropy_with_logits(output.candidate_logit, target, reduction="none")
        auxiliary = (losses*known).sum()/known.sum().clamp_min(1)
    return pair + .5*auxiliary, pair, auxiliary


def binding():
    return {name: cached.sha(Path(__file__).with_name(name))
            for name in ("model.py", "data.py", "train.py", "runtime.py")}


def execute(args):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one assigned CUDA GPU must be visible")
    torch.set_num_threads(2)
    prior.runner._set_determinism(SEED)
    if CONFIGS[args.arm].graph == "shredding":
        # PyG scatter kernels can be nondeterministic on CUDA. Do not silently
        # pretend an exact bitwise guarantee for this explicitly recorded arm.
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_default_dtype(torch.float32)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    training, validation = cached.FormalCache(args.train_cache, "train"), cached.FormalCache(args.val_cache, "val")
    cfg = CONFIGS[args.arm]
    evidence = TrainingEvidence(args.train_diagnostics, training) if cfg.joint_d else None
    net = make(args.arm).to("cuda:0")
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    identity = dict(schema=SCHEMA, arm=args.arm, model=net.metadata(), source_checkpoint_sha256=cache.SOURCE_SHA,
        matcher_frozen=True, head_seed=SEED, data_seed=prior.DATA_SEED,
        physical_microbatch=BATCH, accumulation_steps=1, effective_batch=BATCH, precision="fp32",
        epochs=EPOCHS, train_count=24000, val_count=3000, updates_per_epoch=24000//BATCH,
        train_cache=training.binding, val_cache=validation.binding,
        data_intervention=evidence.summary if evidence else None,
        implementation=binding(), validation_only_selection=True, real_ood_used_for_fit=False,
        reference_note="new batch48 control; old batch16 C16 endpoint is an additional historical reference")
    last = root/"last.pt"
    completed = 0
    if last.exists():
        if not args.resume:
            raise ValueError("existing checkpoint; use explicit resume")
        previous = torch.load(last, map_location="cpu", weights_only=False)
        if previous["identity"] != identity:
            raise ValueError("resume identity differs")
        net.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        prior.restore_rng_state(previous["rng"])
        completed = previous["completed_segments"]
    elif args.resume:
        raise ValueError("resume checkpoint is missing")
    save(root/"protocol.json", dict(status="running", identity=identity, started_unix=time.time()))
    if evidence:
        save(root/"sampling_definition.json", evidence.summary)
    prior.BATCH = BATCH  # isolated process; only reuse the original VAL readout.
    started = time.perf_counter()
    try:
        for epoch in range(1, EPOCHS+1):
            if epoch*4 <= completed:
                continue
            if evidence:
                order, ledger = evidence.order(epoch)
                save(root/("sampling_epoch_%03d.json" % epoch), ledger)
            else:
                order = prior.runner.epoch_indices(24000, seed=prior.DATA_SEED, epoch=12+epoch, limit=None)
            lr = prior.learning_rate(epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            for segment in range(4):
                number = (epoch-1)*4+segment+1
                if number <= completed:
                    continue
                net.train()
                segment_start = time.perf_counter()
                sums = np.zeros(3)
                examples = 0
                torch.cuda.reset_peak_memory_stats()
                selected_ids = order[segment*SEGMENT:(segment+1)*SEGMENT]
                for offset in range(0, SEGMENT, BATCH):
                    ids = selected_ids[offset:offset+BATCH]
                    batch = training.batch(ids, "cuda:0")
                    optimizer.zero_grad(set_to_none=True)
                    output = net(*batch.model_args, **batch.model_kwargs)
                    loss, pair, auxiliary = training_loss(output, batch, evidence, ids)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step()
                    sums += np.array([loss.item(), pair.item(), auxiliary.item()])*len(ids)
                    examples += len(ids)
                if examples != SEGMENT:
                    raise ValueError("partial training segment")
                torch.cuda.synchronize()
                report = dict(epoch=epoch, segment=segment+1, samples=examples,
                    seconds=time.perf_counter()-segment_start,
                    loss=float(sums[0]/examples), pair_bce=float(sums[1]/examples), candidate_bce=float(sums[2]/examples),
                    microbatch=BATCH, effective_batch=BATCH, updates=SEGMENT//BATCH,
                    cuda_peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                    cuda_peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
                save(root/("segment_%03d.json" % number), report)
                if segment == 3:
                    val, rows = prior.evaluate(net, validation, torch.device("cuda:0"))
                    save(root/("validation_%03d.json" % epoch), val)
                    save(root/("validation_%03d_rows.json" % epoch), rows)
                payload = dict(schema=SCHEMA, identity=identity, completed_segments=number,
                    head_epoch=epoch, model={k:v.detach().cpu() for k,v in net.state_dict().items()},
                    optimizer=optimizer.state_dict(), rng=prior.capture_rng_state(), matcher_updated=False)
                if segment == 3:
                    prior.runner._atomic_torch_save(root/("head_epoch_%03d.pt" % epoch), payload)
                prior.runner._atomic_torch_save(last, payload)
                completed = number
                save(root/"status.json", dict(status="training", arm=args.arm, completed_segments=completed,
                    epoch=epoch, elapsed_seconds=time.perf_counter()-started, last_segment=report))
                print(json.dumps(dict(event="segment_complete", arm=args.arm, **report)), flush=True)
        # Freeze before opening any TEST / REAL / OOD input.
        val = json.loads((root/"validation_016.json").read_text())
        head = root/"head_epoch_016.pt"
        save(root/"freeze.json", dict(schema=SCHEMA, status="complete", identity=identity,
            selection="fixed_C16", checkpoint=head.name, checkpoint_sha256=cached.sha(head),
            validation="validation_016.json", operating_points=val["operating_points"],
            real_ood_used_for_fit=False))
        save(root/"status.json", dict(status="training_complete", arm=args.arm, completed_segments=64,
            completed_head_epochs=16, updates=24000*16//BATCH, training_exposures=384000,
            matcher_updated=False, elapsed_seconds=time.perf_counter()-started))
        return identity
    except BaseException as error:
        save(root/"status.json", dict(status="failed", arm=args.arm, completed_segments=completed,
            error=repr(error), recovery="explicit resume from last committed segment"))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=tuple(CONFIGS), required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--train-diagnostics", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--lock-root", required=True)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    with lease(args.gpu_uuid, args.lock_root):
        execute(args)

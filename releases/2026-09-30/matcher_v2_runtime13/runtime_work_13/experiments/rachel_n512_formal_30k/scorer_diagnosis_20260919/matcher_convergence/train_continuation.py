"""Finite S7 M13..M20 GPU training only; no dispatch, evaluation or selection.

Requires an unused output directory (or explicit --resume of its last.pt), the
original S7 M12 SHA, and the same shared nonblocking GPU lock as other jobs.
M16 may pause; M20 reports training complete and fixed Matcher evaluation still
pending. --smoke 32 restores exact GPU state, performs two disposable updates,
and never creates a reusable checkpoint. Nothing starts on import.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local.train import gpu_lock
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence import continuation_core as core

old = core.old
SCHEMA = "rachel-s7-matcher-continuation-training/1"
SOURCE = Path("/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt")


def training_identity(identity):
    """Bind this loop as well as the unchanged kernel and source populations."""
    files = dict(entry=__file__, weathering_loader=sys.modules[old.make_weathering_loader.__module__].__file__,
        materialized_dataset=sys.modules[old.MaterializedWeatheredDataset.__module__].__file__,
        clean_dataset=sys.modules[old.RachelPairDataset.__module__].__file__, runner=old.runner.__file__)
    return dict(schema_version=SCHEMA, continuation_identity_sha256=old.canonical_digest(identity),
        implementation_sha256={key: old._sha256(path) for key, path in files.items()},
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, python=platform.python_version()),
        physical_microbatch=16, logical_microbatch=1, effective_batch=16, workers=4,
        precision="fp32", fixed_learning_rate=2e-5, classifier_transition=False,
        validation_in_training=False, real_ood_used=False, primary_endpoint=20, fixed_endpoints=[16, 20])


def validate_arguments(args):
    if args.stop_after_epoch not in (16, 20) or args.device != "cuda:0":
        raise ValueError("only fixed M16/M20 with one visible cuda:0 is supported")
    if args.smoke not in (None, 32) or args.smoke and args.resume:
        raise ValueError("discard32 smoke must be a new run, never a resume")


def validate_output(output, source_path, populations):
    """Do not create output inside, or as an ancestor of, any original inputs."""
    roots = [Path(source_path).resolve().parent,
        Path(populations["train"]["manifest"]).resolve().parent,
        Path(populations["val"]["manifest"]).resolve().parents[1]]
    for root in roots:
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("output must be isolated from immutable source/data trees")


def _runtime_args(output):
    return SimpleNamespace(output=str(output), microbatch=1, physical_microbatch=16,
        effective_batch=16, runtime_effective_batch=None, workers=4, log_every=1000)


def _loader(training, segment, count, cap):
    order = old.runner.epoch_indices(24000, seed=260913, epoch=segment["epoch"], limit=None)
    indices = order[segment["offset"]:segment["offset"] + count]
    return old.make_weathering_loader(training, indices, batch_size=16, num_workers=4,
        seed=segment["loader_seed"], contour_cap=cap)


def _report_contract(report, count):
    if (report.get("samples"), report.get("optimizer_updates"), report.get("phase"),
            report.get("pair_bce_weight")) != (count, count // 16, "matcher", 0.):
        raise RuntimeError("kernel executed wrong phase/loss/budget")


def _status(output, protocol, **fields):
    protocol.update(fields)
    old.save_json(output / "protocol.json", protocol)
    old.save_json(output / "status.json", dict(schema_version=SCHEMA, pid=os.getpid(), **fields))


def execute(args, model, optimizer, training, loss, source, identity, device):
    """Locked loop; internal CPU tests replace only numerical/source boundaries.

    Production callers must use run(), which owns BOTH GPU and output locks.
    last.pt is the commit authority, not an ahead-of-commit segment report or
    epoch anchor. An interruption never serializes partially updated weights.
    """
    output = Path(args.output)
    binding = training_identity(identity)
    runtime = _runtime_args(output)
    completed, context = 48, None
    checkpoint = source
    if args.resume:
        previous = json.loads((output / "protocol.json").read_text())
        if (previous.get("schema_version") != SCHEMA or previous.get("smoke") or
                previous.get("training_identity") != binding):
            raise ValueError("output ownership/implementation differs; no automatic migration")
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("training_identity") != binding:
            raise ValueError("last committed checkpoint belongs to a different training loop")
        completed = core.validate_progress(checkpoint, identity, rng_mode="exact_gpu")
        if completed > args.stop_after_epoch * 4:
            raise ValueError("requested stop precedes already committed progress")
    protocol = dict(schema_version=SCHEMA, training_identity=binding, continuation_identity=identity,
        plan=core.continuation_plan(), arguments=vars(args), status="initializing",
        smoke=bool(args.smoke), formal_training_counted=not bool(args.smoke),
        fixed_simval_evaluation_complete=False, full_experiment_complete=False,
        ready_for_fixed_matcher_evaluation=False)
    started = time.monotonic()
    try:
        # All construction, population checks, CUDA setup and protocol binding
        # above precede RNG restoration. No seed reset occurs after this point.
        if args.resume:
            context = core.restore_continuation(model, optimizer, checkpoint, identity, rng_mode="exact_gpu")
        else:
            context = core.restore_initial(model, optimizer, source, identity, rng_mode="exact_gpu")
        if context["completed_segments"] != completed:
            raise ValueError("restored progress differs from the committed checkpoint")
        _status(output, protocol, status="running", phase="matcher", completed_segments=completed,
            last_committed_segment=completed, global_exposure=completed * 6000)
        if not args.resume and not args.smoke:
            initial = core.checkpoint_payload(model, optimizer, identity=identity,
                completed_segments=48, resume_context=context, role="initial_recovery")
            initial["training_identity"] = deepcopy(binding)
            old.runner._atomic_torch_save(output / "last.pt", initial)
            del initial
        for segment in core.continuation_plan():
            number, epoch = segment["number"], segment["epoch"]
            if number <= completed or epoch > args.stop_after_epoch:
                continue
            model.set_phase("matcher").train()
            for group in optimizer.param_groups:
                group["lr"] = segment["learning_rate"]
            count = args.smoke or 6000
            loader = _loader(training, segment, count, model.config.contour_cap)
            _status(output, protocol, status="running", phase="matcher", epoch=epoch, segment=number,
                completed_segments=completed, last_committed_segment=completed,
                global_exposure=completed * 6000)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            before_base = old.state_digest(model.base_model) if args.smoke else None
            report = core.private_train_segment(model, loader, optimizer, loss, device, runtime, epoch)
            _report_contract(report, count)
            if args.smoke:
                core.validate_optimizer(dict(optimizer_state_dict=optimizer.state_dict()), model,
                    expected_updates=18002)
                if (core.frozen_digests(model) != identity["source_frozen_digests"] or
                        old.state_digest(model.base_model) == before_base):
                    raise RuntimeError("discard smoke violated frozen/trainable state boundary")
                result = dict(status="smoke_complete", phase="matcher", source_epoch=12, probe_epoch=13,
                    weights_and_optimizer_discarded=True, no_checkpoint_written=True,
                    discarded_pair_exposures=32, discarded_optimizer_updates=2,
                    formal_training_counted=False, formal_pair_exposures=0,
                    exact_gpu_rng_restored=True, training=report, frozen_branches_unchanged=True,
                    ready_for_fixed_matcher_evaluation=False, full_experiment_complete=False)
                old.save_json(output / "smoke.json", result)
                _status(output, protocol, **result, elapsed_s=time.monotonic() - started)
                return result
            saved = core.checkpoint_payload(model, optimizer, identity=identity,
                completed_segments=number, resume_context=context,
                role="epoch_anchor" if segment["epoch_complete"] else "recovery")
            saved["training_identity"] = deepcopy(binding)
            saved["segment_record"] = dict(segment=segment, training=report, formal_training_counted=True)
            # Anchor first, last.pt second: only last.pt commits a segment.
            # A crash between writes replays this segment from the prior last.
            if segment["epoch_complete"]:
                old.runner._atomic_torch_save(output / ("epoch_%03d.pt" % epoch), saved)
            old.runner._atomic_torch_save(output / "last.pt", saved)
            completed = number
            old.save_json(output / ("segment_%03d.json" % number), saved["segment_record"])
            del saved
            if segment["epoch_complete"]:
                print(json.dumps(dict(event="matcher_epoch_committed", epoch=epoch,
                    completed_segments=completed, global_exposure=completed * 6000)), flush=True)
        if completed != args.stop_after_epoch * 4:
            raise RuntimeError("fixed requested Matcher budget was not reached")
        result = dict(status="training_complete" if completed == 80 else "paused", phase="matcher",
            epoch=completed // 4, completed_segments=completed, last_committed_segment=completed,
            global_exposure=completed * 6000, optimizer_updates=completed * 375,
            continuation_pair_exposures=(completed - 48) * 6000,
            continuation_optimizer_updates=(completed - 48) * 375, formal_training_counted=True,
            fixed_epoch_anchors_ready=[e for e in (16, 20) if e * 4 <= completed],
            ready_for_fixed_matcher_evaluation=completed == 80,
            fixed_simval_evaluation_complete=False, full_experiment_complete=False,
            evaluation_pending=True, can_resume_to_epoch=20 if completed < 80 else None)
        _status(output, protocol, **result, elapsed_s=time.monotonic() - started)
        return result
    except BaseException as error:
        _status(output, protocol, status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            phase="matcher", error=repr(error), last_committed_segment=completed,
            completed_segments=completed, resumable_exposure=completed * 6000,
            explicit_resume_required=True, reusable_checkpoint_exists=(output / "last.pt").is_file(),
            ready_for_fixed_matcher_evaluation=False, full_experiment_complete=False)
        raise


def run(args):
    validate_arguments(args)
    output = Path(args.output).resolve()
    source_path = Path(args.source).resolve(strict=True)
    with gpu_lock():  # Nonblocking, fail on foreign compute; never wait/kill/steal.
        if platform.system() != "Linux" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("formal continuation requires Linux and exactly one visible CUDA GPU")
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        old.runner._set_determinism(old.SEED)  # flags/setup BEFORE source RNG restoration
        source_sha = old._sha256(source_path)
        if source_sha != core.SOURCE_SHA256:
            raise ValueError("source file is not the immutable S7 M12 checkpoint")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        model = core.validate_source(source, source_sha)
        identity = core.build_identity(source, source_sha, rng_mode="exact_gpu")
        populations = source["resume_identity"]["populations"]
        training, validation, cap, records = old.make_populations(SimpleNamespace(sampling="original512",
            train_materialized_manifest=populations["train"]["manifest"],
            dataset=str(Path(populations["val"]["manifest"]).parents[1])))
        if old.canonical_digest(records) != old.canonical_digest(populations) or cap != 512:
            raise ValueError("original S7 TRAIN/clean SIMVAL populations changed")
        del validation  # No validation iterator or scoring exists in this entry.
        validate_output(output, source_path, populations)
        if args.resume:
            if not output.is_dir():
                raise ValueError("resume requires an existing owned output directory")
        else:
            output.mkdir(parents=True, exist_ok=False)
        args.output = str(output)
        device = torch.device(args.device)
        model.to(device=device, dtype=torch.float32)
        optimizer = core.create_optimizer(model)
        loss = old.RachelN512LossConfig(**source["loss_config"])
        with old.run_lock(output):
            return execute(args, model, optimizer, training, loss, source, identity, device)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", default=str(SOURCE))
    result.add_argument("--output", required=True)
    result.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    result.add_argument("--stop-after-epoch", type=int, choices=(16, 20), default=20)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--smoke", type=int, choices=(32,))
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))

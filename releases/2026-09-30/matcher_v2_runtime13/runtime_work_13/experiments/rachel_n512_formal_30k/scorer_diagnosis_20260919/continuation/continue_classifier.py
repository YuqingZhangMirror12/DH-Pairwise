"""Isolated C8 -> C16 continuation; never modifies the original M12+C8 run.

Reuses the original train_segment bytecode with a private globals dictionary
whose ONLY substituted dependency is the epoch/phase guard. The public module,
loss, batching, model, data, shuffle, optimizer and validation remain unchanged.
No TEST/REAL/OOD reader is reachable from this training entry point.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
from types import FunctionType, SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import torch
from experiments.rachel_n512_formal_30k import train_score_decoupled as old
from staging.pairwise_v0_2.models.rachel_decoupled_score import load_decoupled_score_checkpoint

SCHEMA = "rachel-frozen-classifier-continuation/1"
START, STOP, FIRST_SEGMENT, LAST_SEGMENT = 20, 28, 80, 112
ROOT = Path("/root/autodl-tmp/rachel_score_design_20260913_001")
ARMS = {
    "s6_d2": (ROOT / "attention_depth_20260915/s4_cross_attention_depth2/training", 2),
    "s7": (ROOT / "s6_s7_20260915/priority_after_s5/s7_augmented_full24/training", 1),
}
EXPECTED_TRAIN_MANIFEST_SHA = {
    "s6_d2": "ca4a794aa2a5d196cc17c3e20144e3e5f9e2370eb952530193e6539de0b330af",
    "s7": "79a9e959f32ef9899116e299425d6350b17f6a04bd5070b5aa9a730319447c36",
}


def phase_for_epoch(epoch):
    if type(epoch) is not int or not 21 <= epoch <= STOP:
        raise ValueError("continuation is classifier-only absolute epochs21..28")
    return "classifier"


def private_function(function, **overrides):
    """Rebind a function without changing its code or its originating module."""
    namespace = dict(function.__globals__, **overrides)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


train_segment = private_function(old.train_segment, phase_for_epoch=phase_for_epoch)
_payload = private_function(old.checkpoint_payload,
    phase_for_epoch=lambda epoch: "classifier" if START <= epoch <= STOP else phase_for_epoch(epoch))


def plan():
    return [dict(number=(epoch - 1) * 4 + i + 1, epoch=epoch, phase="classifier",
        offset=i * old.SEGMENT_SIZE, count=old.SEGMENT_SIZE,
        global_start=(epoch - 1) * old.TRAIN_COUNT + i * old.SEGMENT_SIZE,
        global_stop=(epoch - 1) * old.TRAIN_COUNT + (i + 1) * old.SEGMENT_SIZE,
        epoch_complete=i == 3, learning_rate=2e-5)
        for epoch in range(21, 29) for i in range(4)]


def check_optimizer(checkpoint, model, *, expected_head_step):
    """Reject incomplete Adam recovery; unused no_evidence_logit is explicit."""
    state = checkpoint.get("optimizer_state_dict")
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("exact resume requires complete optimizer_state_dict; use explicit warm-start otherwise")
    groups = state["param_groups"]
    if [g.get("phase_family") for g in groups] != ["base", "new_head"]:
        raise ValueError("optimizer lifetime groups differ")
    own_matcher_optimizer = (not checkpoint["resume_identity"].get("matcher_checkpoint_sha256")
        and checkpoint.get("continuation_identity", {}).get("restore_mode", "exact") == "exact")
    for group, module in zip(groups, (model.base_model, model.score_head)):
        named = list(module.named_parameters())
        if len(group["params"]) != len(named) or group["lr"] != 2e-5 or group["weight_decay"] != 1e-4:
            raise ValueError("optimizer group structure/LR/weight decay differs")
        for key, (name, parameter) in zip(group["params"], named):
            item = state["state"].get(key)
            required_base = (group["phase_family"] == "base" and own_matcher_optimizer
                and not name.startswith(("coarse.", "local_head.", "fusion.")))
            if item is None:
                if group["phase_family"] == "base" and not required_base:
                    continue
                if name == "no_evidence_logit" or expected_head_step == 0:
                    continue
                raise ValueError("missing active optimizer state: " + name)
            if not {"step", "exp_avg", "exp_avg_sq"} <= set(item):
                raise ValueError("incomplete Adam moments: " + name)
            for field in ("exp_avg", "exp_avg_sq"):
                if item[field].shape != parameter.shape or not torch.isfinite(item[field]).all():
                    raise ValueError("invalid Adam moment: " + name)
            step = int(item["step"])
            if required_base and step != 18000:
                raise ValueError("frozen Matcher Adam step differs from M12: " + name)
            if group["phase_family"] == "new_head" and name != "no_evidence_logit" and step != expected_head_step:
                raise ValueError("head Adam step differs from registered history: " + name)
    rng = checkpoint.get("rng_state")
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("exact resume requires complete Python/NumPy/Torch/CUDA RNG state")
    return dict(head_step=expected_head_step, optimizer_restored=True, rng_restored=True,
                allowed_unused_head_parameter="no_evidence_logit")


def validate_source(checkpoint, arm):
    ident = checkpoint.get("resume_identity", {})
    old.validate_checkpoint_progress(checkpoint, ident)
    if checkpoint.get("inference_only") or checkpoint["completed_segments"] != FIRST_SEGMENT:
        raise ValueError("source must be the original resumable epoch20, not a selected earlier winner")
    if (ident.get("head_kind") != "cross_attention" or ident.get("sampling") != "original512"
            or ident.get("contour_cap") != 512 or ident.get("effective_batch") != 16
            or ident.get("microbatch") != 1 or ident.get("workers") != 4
            or ident.get("precision") != "fp32"
            or ident.get("populations", {}).get("train", {}).get("manifest_sha256") != EXPECTED_TRAIN_MANIFEST_SHA[arm]
            or ident.get("model_options", {}).get("cross_attention_depth", 1) != ARMS[arm][1]):
        raise ValueError("source differs from registered S6-D2/S7 configuration")
    runtime = checkpoint.get("runtime_batching", {})
    if runtime.get("physical_microbatch") != 16:
        raise ValueError("source must already use physical16/effective16; no new batch migration")
    model = old.load_decoupled_checkpoint(checkpoint)
    return model


def make_identity(args, source, source_path, records):
    origin = source["resume_identity"]
    if old.canonical_digest(records) != old.canonical_digest(origin["populations"]):
        raise ValueError("TRAIN/clean SIMVAL identity differs from the epoch20 source")
    return dict(schema_version=SCHEMA, arm=args.arm, schedule="M12_C8_plus_C8_frozen_matcher",
        source_checkpoint=str(source_path), source_checkpoint_sha256=old._sha256(source_path),
        source_resume_identity_sha256=old.canonical_digest(origin), source_epoch=20,
        source_classifier_epochs=8, final_epoch=28, final_classifier_epochs=16,
        additional_epochs=8, additional_pair_exposures=192000, additional_optimizer_updates=12000,
        lr=2e-5, optimizer="AdamW", weight_decay=1e-4, physical_microbatch=16,
        logical_microbatch=1, effective_batch=16, precision="fp32", workers=4,
        classifier_loss="unchanged samplewise PairBCE only", matcher_frozen=True,
        base_state_sha256=source["matcher_pretraining_receipt"]["base_state_sha256"],
        restore_mode=args.restore_mode, optimizer_reset=args.restore_mode == "warm-start",
        rng_reset=args.restore_mode == "warm-start", warm_start_seed=260919 if args.restore_mode == "warm-start" else None,
        bitwise_trajectory_equivalence_claimed=False, populations=records,
        primary_selection="fixed_epoch28", auxiliary_eligible_epoch_range=[13, 28],
        auxiliary_rules=["SIMVAL max-F1 (original tie-break)", "SIMVAL P@Recall95 (original tie-break)"],
        selection_population="clean SIM VAL3000 only", held_out_used_for_fit=False,
        endpoint_evaluation_only=True, implementation_sha256=old._sha256(__file__),
        reused_trainer_sha256=old._sha256(old.__file__))


def validate_continuation(checkpoint, identity):
    if (checkpoint.get("continuation_schema") != SCHEMA or checkpoint.get("inference_only")
            or checkpoint.get("continuation_identity") != identity):
        raise ValueError("continuation identity or schema differs")
    origin = checkpoint["resume_identity"]
    if old.canonical_digest(origin) != identity["source_resume_identity_sha256"]:
        raise ValueError("origin resume identity changed")
    number = checkpoint.get("completed_segments")
    if type(number) is not int or not FIRST_SEGMENT <= number <= LAST_SEGMENT:
        raise ValueError("continuation committed segments must be80..112")
    expected = dict(epoch=(number + 3) // 4, global_exposure=number * 6000,
        optimizer_updates=number * 375, phase="classifier", continuation_segments=number - 80,
        continuation_pair_exposures=(number - 80) * 6000,
        continuation_optimizer_updates=(number - 80) * 375)
    if any(checkpoint.get(k) != value for k, value in expected.items()):
        raise ValueError("continuation epoch/exposure/update ledger differs")
    if checkpoint.get("decoupled_training_schema") != old.SCHEMA or checkpoint.get("decoupled_score_schema") != old.MODEL_SCHEMA:
        raise ValueError("base checkpoint type differs")
    old.validate_head_revision(checkpoint["decoupled_score"], origin)
    if (checkpoint["decoupled_score"]["base_model_config"] != origin["base_model_config"]
            or checkpoint["decoupled_score"]["phase"] != "classifier"):
        raise ValueError("model configuration or phase changed")
    old.validate_runtime_batching(checkpoint.get("runtime_batching"), origin, number)
    return number


def restore(model, optimizer, checkpoint, identity, *, initial=False):
    number = FIRST_SEGMENT if initial else validate_continuation(checkpoint, identity)
    exact = identity["restore_mode"] == "exact"
    if exact or not initial:
        step = (number - (48 if exact else 80)) * 375
        check_optimizer(checkpoint, model, expected_head_step=step)
        if torch.cuda.is_available() and len(checkpoint["rng_state"]["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology differs: expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.set_phase("classifier").train()
    old.verify_receipt(model, checkpoint["matcher_pretraining_receipt"], checkpoint["resume_identity"])
    if exact or not initial:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        old.restore_rng_state(checkpoint["rng_state"])
    else:
        # Deliberate fallback only. Never silently call this an exact resume.
        old.runner._set_determinism(identity["warm_start_seed"])
    for group in optimizer.param_groups:
        group["lr"] = 2e-5
    return number


def payload(model, optimizer, source, identity, number, winners, role):
    result = _payload(model, optimizer, identity=source["resume_identity"],
        loss_config=old.RachelN512LossConfig(**source["loss_config"]), completed=number,
        receipt=source["matcher_pretraining_receipt"], winners=winners, role=role,
        runtime_batching=source["runtime_batching"])
    result.update(continuation_schema=SCHEMA, continuation_identity=identity,
        continuation_segments=number - 80, continuation_pair_exposures=(number - 80) * 6000,
        continuation_optimizer_updates=(number - 80) * 375,
        initialization="full epoch20 weights + optimizer/RNG" if identity["restore_mode"] == "exact"
            else "explicit epoch20 weight-only warm-start; optimizer/RNG reset")
    validate_continuation(result, identity)
    return result


def update_winners(winners, root, epoch, report, points):
    result = deepcopy(winners)
    if report["decision_coverage"] != 1.:
        raise ValueError("SIMVAL must retain complete decision coverage")
    for name in ("max_f1", "recall95"):
        candidate = old.winner_record(root, epoch, report, points, name)
        if name not in result or tuple(candidate["selection_key"]) > tuple(result[name]["selection_key"]):
            result[name] = candidate
    if epoch == STOP:
        result["fixed_epoch"] = old.winner_record(root, epoch, report, points, "fixed_epoch")
    return result


def publish_freeze(root, identity, winners):
    if set(winners) != {"max_f1", "recall95", "fixed_epoch"} or winners["fixed_epoch"]["selected_epoch"] != STOP:
        raise ValueError("cannot freeze/evaluate before fixed C16 endpoint")
    selections = {key: dict(value, checkpoint_sha256=old._sha256(value["checkpoint"]))
                  for key, value in winners.items()}
    old.save_json(root / "classifier_freezes/freeze.json", dict(schema_version=SCHEMA,
        status="complete", budget_epochs=28, matcher_epochs=12, classifier_epochs=16,
        primary_selection="fixed_epoch", eligible_epoch_range=[13, 28], selections=selections,
        continuation_identity=identity, continuation_identity_sha256=old.canonical_digest(identity),
        selection_population="clean SIM VAL3000 only", held_out_used_for_fit=False,
        no_GT_layout_used_for_selection=True, endpoint_evaluation_only=True))


def install_first_update_receipt(optimizer, model, output, *, smoke=False):
    """Bounded launch evidence, not repeated per-batch progress reporting."""
    updates = []
    def after_step(current, _args, _kwargs):
        if len(updates) >= 2:
            return
        steps = sorted({int(current.state[p]["step"]) for p in model.score_head.parameters()
                        if p in current.state and "step" in current.state[p]})
        updates.append(dict(update_since_launch=len(updates) + 1, head_adam_steps=steps,
            lr=[group["lr"] for group in current.param_groups], physical_microbatch=16,
            effective_batch=16, matcher_frozen=not any(p.requires_grad for p in model.base_model.parameters())))
        old.save_json(output / "first_updates.json", dict(pid=os.getpid(), smoke=smoke,
            formal_training_counted=not smoke, updates=updates))
    return optimizer.register_step_post_hook(after_step)


def run(args):
    if args.resume and args.smoke:
        raise ValueError("discard smoke cannot resume")
    if platform.system() != "Linux" or not torch.cuda.is_available() or args.device != "cuda:0":
        raise RuntimeError("remote Linux CUDA only; expose the intended card as cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU using CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(1)
    old.runner._set_determinism(old.SEED)
    source_path = (Path(args.source_training) if args.source_training else ARMS[args.arm][0]).resolve() / "epoch_020.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    model = validate_source(source, args.arm)
    origin = source["resume_identity"]
    data_args = SimpleNamespace(sampling="original512", train_materialized_manifest=origin["populations"]["train"]["manifest"],
        dataset=str(Path(origin["populations"]["val"]["manifest"]).parents[1]))
    training, validation, cap, records = old.make_populations(data_args)
    identity = make_identity(args, source, source_path, records)
    output = Path(args.output).resolve()
    if output == source_path.parent or source_path.parent in output.parents:
        raise ValueError("all output must be outside the immutable source training directory")
    if args.resume:
        if not output.is_dir():
            raise ValueError("resume output does not exist")
    else:
        output.mkdir(parents=True, exist_ok=False)
    args.output = str(output)
    # Same exact data-loader/runtime settings as the completed source arm.
    args.microbatch, args.physical_microbatch, args.effective_batch = 1, 16, 16
    args.runtime_effective_batch, args.workers, args.log_every = None, 4, 1000
    device = torch.device(args.device)
    model = model.to(device)
    optimizer = old.create_optimizer(model)
    with old.run_lock(output):
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False) if args.resume else source
        completed = restore(model, optimizer, checkpoint, identity, initial=not args.resume)
        winners = deepcopy(checkpoint["winners"])
        if not args.resume:
            winners.pop("fixed_epoch", None)  # primary C16 is not the historical C8 endpoint
        protocol = dict(schema_version=SCHEMA, continuation_identity=identity, arguments=vars(args),
            status="running", plan=plan(), smoke=bool(args.smoke), formal_training_counted=not bool(args.smoke),
            runtime=dict(torch=str(torch.__version__), python=platform.python_version(), cuda=torch.version.cuda),
            source_state_receipt=dict(source_epoch=20, source_complete_segments=80,
                optimizer_restored=identity["restore_mode"] == "exact", rng_restored=identity["restore_mode"] == "exact",
                source_cuda_rng_count=len(source.get("rng_state", {}).get("cuda", []))))
        old.save_json(output / "protocol.json", protocol)
        update_hook = install_first_update_receipt(optimizer, model, output, smoke=bool(args.smoke))
        if not args.resume and not args.smoke:
            old.runner._atomic_torch_save(output / "last.pt", payload(model, optimizer, source, identity, completed, winners, "continuation_anchor"))
        loss = old.RachelN512LossConfig(**source["loss_config"])
        try:
            for segment in plan():
                number, epoch = segment["number"], segment["epoch"]
                if number <= completed:
                    continue
                model.set_phase("classifier").train()  # no classifier RNG reset
                if model.base_model.training or any(p.requires_grad for p in model.base_model.parameters()):
                    raise RuntimeError("base must remain frozen/eval")
                order = old.runner.epoch_indices(old.TRAIN_COUNT, seed=old.SEED, epoch=epoch, limit=None)
                count = args.smoke or old.SEGMENT_SIZE
                indices = order[segment["offset"]:segment["offset"] + count]
                loader = old.make_weathering_loader(training, indices, batch_size=16, num_workers=4,
                    seed=old.SEED + number, contour_cap=cap)
                old.save_json(output / "status.json", dict(status="running", pid=os.getpid(), epoch=epoch,
                    phase="classifier", segment=number, continuation_pair_exposures=(completed - 80) * 6000))
                torch.cuda.reset_peak_memory_stats(device)
                report = train_segment(model, loader, optimizer, loss, device, args, epoch)
                if (report["samples"], report["optimizer_updates"]) != (count, count // 16):
                    raise RuntimeError("actual exposure/update count differs")
                old.verify_receipt(model, source["matcher_pretraining_receipt"], origin)
                old.save_json(output / ("segment_%03d.json" % number), dict(segment=segment, training=report,
                    formal_training_counted=not bool(args.smoke)))
                if args.smoke:
                    result = dict(status="smoke_complete", weights_discarded=True, formal_training_counted=False,
                        no_checkpoint_written=True, pair_exposures=count, optimizer_updates=count // 16,
                        frozen_base_unchanged=True, training=report)
                    old.save_json(output / "smoke.json", result)
                    break
                if segment["epoch_complete"]:
                    from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
                    old.save_json(output / "status.json", dict(status="running", phase="SIMVAL", epoch=epoch))
                    val_loader = old.make_ablation_loader(validation, list(range(old.VAL_COUNT)), batch_size=1,
                        num_workers=4, seed=old.SEED, contour_cap=cap)
                    report_val, rows = old.evaluate_pair_validation(model, val_loader, device)
                    points = fit_operating_points([r["label"] for r in rows], [r["classification"]["fused"] for r in rows])
                    old.save_json(output / ("validation_%03d_rows.json" % epoch), rows)
                    old.save_json(output / ("validation_%03d.json" % epoch), dict(epoch=epoch, validation=report_val,
                        operating_points=points, selection_eligible=True, global_exposure=number * 6000))
                    winners = update_winners(winners, output, epoch, report_val, points)
                saved = payload(model, optimizer, source, identity, number, winners, "epoch_anchor" if segment["epoch_complete"] else "recovery")
                if segment["epoch_complete"]:
                    old.runner._atomic_torch_save(output / ("epoch_%03d.pt" % epoch), saved)
                old.runner._atomic_torch_save(output / "last.pt", saved)
                completed = number
                del saved
            else:
                if completed != LAST_SEGMENT:
                    raise RuntimeError("C16 fixed budget incomplete")
                publish_freeze(output, identity, winners)
                result = dict(status="complete", epoch=28, classifier_epochs=16, completed_segments=112,
                    additional_pair_exposures=192000, additional_optimizer_updates=12000, global_exposure=672000,
                    optimizer_updates=42000, phase="train_val_complete", held_out_evaluated=False)
            protocol.update(result)
            old.save_json(output / "status.json", result)
            old.save_json(output / "protocol.json", protocol)
            update_hook.remove()
            return result
        except BaseException as error:
            # Never serialize mid-segment weights as if they were committed.
            protocol.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            error=repr(error), last_committed_segment=completed,
                            recovery="--resume reloads last.pt and repeats only uncommitted work")
            old.save_json(output / "protocol.json", protocol)
            old.save_json(output / "status.json", protocol)
            update_hook.remove()
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--arm", choices=tuple(ARMS), required=True)
    result.add_argument("--source-training", help="optional relocated original directory; never a winner checkpoint")
    result.add_argument("--output", required=True)
    result.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    result.add_argument("--restore-mode", choices=("exact", "warm-start"), default="exact")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--smoke", type=int, choices=(16, 32, 64))
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))

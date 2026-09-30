"""S7 M12 -> M20 state/step core only: no CLI, I/O loop, queue or selection.

The caller verifies the source file SHA and population manifests, constructs
the original wrapper on its final device, and creates the original optimizer
BEFORE restoring RNG. ``private_train_segment`` is the original numerical
kernel with only its phase lookup privately rebound. The caller must use the
plan's LR and original loader/order seeds, and hold the shared GPU lock.

``cpu_only_probe`` restores only Python/NumPy/CPU Torch and carries the saved
CUDA bytes unchanged. Its checkpoints are explicitly non-formal and cannot be
resumed as ``exact_gpu``. Neither mode claims equivalence to the original C13
trajectory: continuing Matcher optimization is a new, fixed-budget control.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import random
from types import FunctionType

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import train_score_decoupled as old
from experiments.rachel_n512_formal_30k import decoupled_samplewise_loss
from staging.pairwise_v0_2.models import rachel_decoupled_score

SCHEMA = "rachel-s7-matcher-only-continuation/1"
START_EPOCH, STOP_EPOCH, FIRST_SEGMENT, LAST_SEGMENT = 12, 20, 48, 80
SOURCE_SHA256 = "d8a93af1eb5f3b02baaf7d42b9d8675242a446a1b11e43cde0561ba89e670e07"
SOURCE_BASE_SHA256 = "90eb37d6525d59a9e27cd23618f1fedd4cbbac320cf24623dc3de97b7733a00b"
TRAIN_SHA256 = "79a9e959f32ef9899116e299425d6350b17f6a04bd5070b5aa9a730319447c36"
VAL_SHA256 = "daa6ccdd7686e93ba91ddfb1452c987145c26898a1917d2ac7d3180e199a8af8"
TRAINER_SHA256 = "146967835979362dbf20ccc2a7d2fd375b6fc324a16a8936969a03dc1afe97b8"
LOSS_SHA256 = "7e39e68002f10386e7d9f5b4c4fe95f2122e3412d17aee6087ae186ab99c479d"
RNG_MODES = ("exact_gpu", "cpu_only_probe")
FROZEN_BASE_NAMES = ("coarse", "local_head", "fusion")


def private_phase_for_epoch(epoch):
    if type(epoch) is not int or not 13 <= epoch <= STOP_EPOCH:
        raise ValueError("Matcher-only continuation accepts absolute epochs13..20")
    return "matcher"


private_train_segment = FunctionType(old.train_segment.__code__,
    dict(old.train_segment.__globals__, phase_for_epoch=private_phase_for_epoch),
    old.train_segment.__name__, old.train_segment.__defaults__, old.train_segment.__closure__)
private_train_segment.__kwdefaults__ = deepcopy(old.train_segment.__kwdefaults__)


def continuation_plan():
    return [dict(number=(epoch - 1) * 4 + i + 1, epoch=epoch, phase="matcher",
        offset=i * 6000, count=6000, global_start=(epoch - 1) * 24000 + i * 6000,
        global_stop=(epoch - 1) * 24000 + (i + 1) * 6000, epoch_complete=i == 3,
        learning_rate=2e-5, order_seed=260913, loader_seed=260913 + (epoch - 1) * 4 + i + 1,
        retain_endpoint=i == 3 and epoch in (16, 20))
        for epoch in range(13, 21) for i in range(4)]


def implementation_bindings():
    result = dict(core=old._sha256(__file__), trainer=old._sha256(old.__file__),
        samplewise_loss=old._sha256(decoupled_samplewise_loss.__file__),
        model_wrapper=old._sha256(rachel_decoupled_score.__file__))
    if result["trainer"] != TRAINER_SHA256 or result["samplewise_loss"] != LOSS_SHA256:
        raise ValueError("original trainer/samplewise implementation changed")
    return result


def frozen_digests(model):
    return dict(score_head=old.state_digest(model.score_head),
        **{name: old.state_digest(getattr(model.base_model, name)) for name in FROZEN_BASE_NAMES})


def _check_model(model, metadata=None):
    if metadata is not None and model.metadata() != metadata:
        raise ValueError("original source wrapper/model metadata changed")
    if model.phase != "matcher":
        raise ValueError("Matcher continuation cannot enter classifier phase")
    active = []
    for name, parameter in model.named_parameters():
        expected = name.startswith("base_model.") and not name.startswith(
            tuple("base_model." + part + "." for part in FROZEN_BASE_NAMES))
        if parameter.requires_grad != expected:
            raise ValueError("Matcher/frozen parameter boundary differs: " + name)
        if expected:
            active.append(name)
        elif parameter.grad is not None:
            raise ValueError("frozen parameter has a gradient: " + name)
    if len(active) != 43:
        raise ValueError("expected exactly43 active S7 Matcher parameter tensors")
    frozen = [model.score_head] + [getattr(model.base_model, part) for part in FROZEN_BASE_NAMES]
    if any(module.training for module in frozen):
        raise ValueError("frozen diagnostic/head branches must stay in eval mode")
    return active


def validate_optimizer(checkpoint, model, *, expected_updates):
    """Check original lifetime groups and every active Adam state, not just LR."""
    _check_model(model)
    state = checkpoint.get("optimizer_state_dict")
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("complete AdamW optimizer state required")
    groups = state["param_groups"]
    if [group.get("phase_family") for group in groups] != ["base", "new_head"]:
        raise ValueError("optimizer lifetime groups changed")
    seen, active = set(), []
    for group, prefix, module, count in zip(groups, ("base_model", "score_head"),
            (model.base_model, model.score_head), (81, 21)):
        parameters = list(module.named_parameters())
        if len(parameters) != count or len(group["params"]) != count:
            raise ValueError("optimizer parameter count/order contract changed")
        if (group.get("lr") != 2e-5 or group.get("weight_decay") != 1e-4 or
                tuple(group.get("betas", ())) != (.9, .999) or group.get("eps") != 1e-8 or
                group.get("amsgrad") is not False or group.get("maximize", False) or
                group.get("capturable", False) or group.get("differentiable", False)):
            raise ValueError("original AdamW hyperparameters changed")
        for key, (name, parameter) in zip(group["params"], parameters):
            if key in seen:
                raise ValueError("duplicate optimizer parameter identity")
            seen.add(key)
            item = state["state"].get(key)
            if not parameter.requires_grad:
                if item is not None:
                    raise ValueError("unused head/diagnostic branch acquired Adam state")
                continue
            active.append(prefix + "." + name)
            if not isinstance(item, dict) or set(item) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("missing/incomplete active Matcher Adam state: " + name)
            step = item["step"]
            if torch.is_tensor(step):
                if step.numel() != 1 or not torch.isfinite(step).all():
                    raise ValueError("invalid Adam step")
                step = step.item()
            if step != expected_updates:
                raise ValueError("active Matcher Adam step differs: " + name)
            for field in ("exp_avg", "exp_avg_sq"):
                value = item[field]
                if (not torch.is_tensor(value) or value.shape != parameter.shape or
                        value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                    raise ValueError("invalid Matcher Adam moment: " + name)
    if not set(state["state"]) <= seen or len(state["state"]) != 43:
        raise ValueError("unexpected optimizer states outside the43 active Matcher parameters")
    return active


def _check_rng(state):
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("complete Python/NumPy/CPU/CUDA RNG state required")
    if (not isinstance(state["cuda"], (tuple, list)) or len(state["cuda"]) != 1 or
            not isinstance(state["python"], tuple) or not isinstance(state["numpy"], tuple)):
        raise ValueError("expected source single-GPU RNG topology")
    for value in [state["torch"], *state["cuda"]]:
        if not torch.is_tensor(value) or value.dtype != torch.uint8 or value.ndim != 1 or not value.numel():
            raise ValueError("invalid serialized RNG tensor")


def validate_source(checkpoint, source_sha256):
    """SHA is computed over the actual file by the caller; no file loading here."""
    if source_sha256 != SOURCE_SHA256:
        raise ValueError("requires the exact registered S7 epoch012 checkpoint SHA")
    origin = checkpoint.get("resume_identity", {})
    old.validate_checkpoint_progress(checkpoint, origin)
    if (checkpoint.get("inference_only") or checkpoint.get("completed_segments") != 48 or
            checkpoint.get("checkpoint_role") != "epoch_anchor" or
            checkpoint.get("inherited_matcher_exposures") != 0 or
            checkpoint.get("formal_training_counted") is not True or
            origin.get("matcher_checkpoint_sha256") is not None or
            checkpoint.get("loss_config") != origin.get("loss_config")):
        raise ValueError("source must be the original full-state, non-import S7 M12 anchor")
    expected = dict(head_kind="cross_attention", sampling="original512", contour_cap=512,
        microbatch=1, effective_batch=16, workers=4, precision="fp32", train_count=24000,
        validation_count=3000, segment_pairs=6000, seed=260913)
    if any(origin.get(key) != value for key, value in expected.items()) or origin.get("model_options", {}):
        raise ValueError("source S7 model/data/batching configuration differs")
    for split, count, sha in (("train", 24000, TRAIN_SHA256), ("val", 3000, VAL_SHA256)):
        record = origin.get("populations", {}).get(split, {})
        if record.get("count") != count or record.get("split") != split or record.get("manifest_sha256") != sha:
            raise ValueError("source TRAIN/SIMVAL population differs")
    loss = old.RachelN512LossConfig(**checkpoint["loss_config"])
    if (origin.get("matcher_loss_config") != asdict(old.matcher_loss_config(loss)) or
            loss.translation_scale_px != 32. or loss.sinkhorn_residual_target != .001):
        raise ValueError("source Matcher loss differs")
    runtime = checkpoint.get("runtime_batching") or {}
    history = runtime.get("history", [])
    if (runtime.get("physical_microbatch") != 16 or len(history) != 1 or
            history[0].get("committed_segments") != 0 or history[0].get("physical_microbatch") != 16):
        raise ValueError("source must use physical16 from segment0, without a new migration")
    model = old.load_decoupled_checkpoint(checkpoint)
    if old.state_digest(model.base_model) != SOURCE_BASE_SHA256:
        raise ValueError("S7 M12 Matcher digest differs")
    validate_optimizer(checkpoint, model, expected_updates=18000)
    _check_rng(checkpoint.get("rng_state"))
    return model


def build_identity(source, source_sha256, *, rng_mode="exact_gpu"):
    if rng_mode not in RNG_MODES:
        raise ValueError("use explicit exact_gpu or cpu_only_probe RNG mode")
    model = validate_source(source, source_sha256)
    return dict(schema_version=SCHEMA, schedule="S7_M12_plus_M8_matcher_only",
        source_checkpoint_sha256=source_sha256, source_resume_identity=deepcopy(source["resume_identity"]),
        source_resume_identity_sha256=old.canonical_digest(source["resume_identity"]),
        source_model_metadata=deepcopy(source["decoupled_score"]),
        origin_matcher_pretraining_receipt=deepcopy(source["matcher_pretraining_receipt"]),
        origin_runtime_batching=deepcopy(source["runtime_batching"]),
        loss_config=deepcopy(source["loss_config"]), source_frozen_digests=frozen_digests(model),
        active_parameter_names=_check_model(model),
        optimizer_parameter_names=[[name for name, _ in module.named_parameters()]
            for module in (model.base_model, model.score_head)],
        source_epoch=12, final_epoch=20, fixed_endpoints=[16, 20], primary_endpoint=20,
        additional_exposures=192000, additional_updates=12000,
        physical_microbatch=16, logical_microbatch=1, effective_batch=16, workers=4,
        learning_rate=2e-5, optimizer="AdamW", weight_decay=1e-4, grad_clip_norm=5., precision="fp32",
        rng_mode=rng_mode, formal_training_counted=rng_mode == "exact_gpu",
        optimizer_reset=False, rng_reset=False, classifier_phase_used=False,
        original_plan_trajectory_equivalence_claimed=False,
        implementation_bindings=implementation_bindings())


def _validate_identity(identity, rng_mode):
    if (identity.get("schema_version") != SCHEMA or rng_mode not in RNG_MODES or
            identity.get("rng_mode") != rng_mode or
            identity.get("formal_training_counted") != (rng_mode == "exact_gpu") or
            identity.get("source_checkpoint_sha256") != SOURCE_SHA256 or
            old.canonical_digest(identity["source_resume_identity"]) != identity.get("source_resume_identity_sha256") or
            identity.get("implementation_bindings") != implementation_bindings()):
        raise ValueError("continuation identity/source/implementation/RNG mode differs")


def create_optimizer(model):
    _check_model(model)
    optimizer = old.create_optimizer(model)
    for group in optimizer.param_groups:
        group["lr"] = 2e-5
    return optimizer


def _check_optimizer_binding(optimizer, model):
    if not isinstance(optimizer, torch.optim.AdamW) or len(optimizer.param_groups) != 2:
        raise ValueError("original two-group AdamW required")
    for group, module, family in zip(optimizer.param_groups,
            (model.base_model, model.score_head), ("base", "new_head")):
        if (group.get("phase_family") != family or
                [id(p) for p in group["params"]] != [id(p) for p in module.parameters()]):
            raise ValueError("optimizer is not bound to original model parameter order")


def _restore_rng(state, model, rng_mode):
    _check_rng(state)
    devices = {p.device for p in model.parameters()}
    if rng_mode == "exact_gpu":
        if (not torch.cuda.is_available() or torch.cuda.device_count() != len(state["cuda"]) or
                devices != {torch.device("cuda:0")}):
            raise ValueError("exact_gpu requires original single visible GPU and model on cuda:0")
        old.restore_rng_state(state)
    elif rng_mode == "cpu_only_probe":
        if devices != {torch.device("cpu")}:
            raise ValueError("cpu_only_probe requires a CPU model")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
    else:
        raise ValueError("unknown RNG mode")


def _restore(model, optimizer, checkpoint, identity, number, rng_mode):
    _check_model(model, identity["source_model_metadata"])
    _check_optimizer_binding(optimizer, model)
    validate_optimizer(checkpoint, model, expected_updates=number * 375)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.set_phase("matcher").train()
    if frozen_digests(model) != identity["source_frozen_digests"]:
        raise ValueError("head/coarse/local/fusion tensor or buffer changed")
    expected_base = SOURCE_BASE_SHA256 if number == 48 else checkpoint["current_base_state_sha256"]
    if old.state_digest(model.base_model) != expected_base:
        raise ValueError("current Matcher state digest differs")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    context = dict(completed_segments=number, rng_mode=rng_mode,
        identity_sha256=old.canonical_digest(identity),
        saved_cuda_rng=[value.detach().cpu().clone() for value in checkpoint["rng_state"]["cuda"]],
        formal_training_counted=rng_mode == "exact_gpu", gpu_rng_restored=rng_mode == "exact_gpu")
    # MUST be last: model construction/strict checks above may consume CPU RNG.
    _restore_rng(checkpoint["rng_state"], model, rng_mode)
    return context


def restore_initial(model, optimizer, source, identity, *, rng_mode="exact_gpu"):
    _validate_identity(identity, rng_mode)
    expected = build_identity(source, identity["source_checkpoint_sha256"], rng_mode=rng_mode)
    if expected != identity:
        raise ValueError("initial source and immutable continuation identity differ")
    return _restore(model, optimizer, source, identity, 48, rng_mode)


def _progress(number):
    if type(number) is not int or not 48 <= number <= 80:
        raise ValueError("committed segments must be48..80")
    return dict(completed_segments=number, epoch=(number + 3) // 4, phase="matcher",
        global_exposure=number * 6000, optimizer_updates=number * 375,
        continuation_segments=number - 48, continuation_pair_exposures=(number - 48) * 6000,
        continuation_optimizer_updates=(number - 48) * 375)


def validate_progress(checkpoint, identity, *, rng_mode="exact_gpu"):
    _validate_identity(identity, rng_mode)
    if (checkpoint.get("matcher_continuation_schema") != SCHEMA or
            checkpoint.get("continuation_identity") != identity or checkpoint.get("inference_only") or
            "decoupled_training_schema" in checkpoint or "matcher_pretraining_receipt" in checkpoint):
        raise ValueError("not this new M-only continuation schema; old M12 receipt is provenance only")
    expected = _progress(checkpoint.get("completed_segments"))
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("Matcher-only progress/exposure/update ledger differs")
    for key, value in dict(resume_identity=identity["source_resume_identity"],
            decoupled_score_schema=old.MODEL_SCHEMA, decoupled_score=identity["source_model_metadata"],
            origin_matcher_pretraining_receipt=identity["origin_matcher_pretraining_receipt"],
            runtime_batching=identity["origin_runtime_batching"], loss_config=identity["loss_config"],
            rng_mode=rng_mode, formal_training_counted=rng_mode == "exact_gpu").items():
        if checkpoint.get(key) != value:
            raise ValueError("continuation provenance/model/runtime differs: " + key)
    old.validate_runtime_batching(checkpoint["runtime_batching"], checkpoint["resume_identity"], expected["completed_segments"])
    _check_rng(checkpoint.get("rng_state"))
    if checkpoint.get("checkpoint_role") not in ("initial_recovery", "recovery", "epoch_anchor"):
        raise ValueError("unknown checkpoint role")
    if checkpoint["checkpoint_role"] == "epoch_anchor" and expected["completed_segments"] % 4:
        raise ValueError("epoch anchor must be a complete epoch")
    return expected["completed_segments"]


def checkpoint_payload(model, optimizer, *, identity, completed_segments,
                       resume_context, role="recovery"):
    mode = resume_context.get("rng_mode")
    _validate_identity(identity, mode)
    if (resume_context.get("identity_sha256") != old.canonical_digest(identity) or
            not resume_context["completed_segments"] <= completed_segments):
        raise ValueError("resume context/committed progress differs")
    _check_model(model, identity["source_model_metadata"])
    _check_optimizer_binding(optimizer, model)
    if frozen_digests(model) != identity["source_frozen_digests"]:
        raise ValueError("frozen head/diagnostic tensors or buffers changed")
    progress = _progress(completed_segments)
    state = deepcopy(optimizer.state_dict())
    validate_optimizer(dict(optimizer_state_dict=state), model, expected_updates=progress["optimizer_updates"])
    if mode == "exact_gpu":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or any(p.device != torch.device("cuda:0") for p in model.parameters()):
            raise ValueError("formal checkpoint requires original single visible GPU")
        rng = old.capture_rng_state()
    else:
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("CPU probe checkpoint requires CPU model")
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
            cuda=[value.detach().cpu().clone() for value in resume_context["saved_cuda_rng"]])
    result = dict(matcher_continuation_schema=SCHEMA, continuation_identity=deepcopy(identity),
        resume_identity=deepcopy(identity["source_resume_identity"]), **progress,
        decoupled_score_schema=old.MODEL_SCHEMA, decoupled_score=deepcopy(model.metadata()),
        model_state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        optimizer_state_dict=state, rng_state=rng, rng_mode=mode,
        formal_training_counted=mode == "exact_gpu", checkpoint_role=role,
        current_base_state_sha256=old.state_digest(model.base_model),
        loss_config=deepcopy(identity["loss_config"]), runtime_batching=deepcopy(identity["origin_runtime_batching"]),
        origin_matcher_pretraining_receipt=deepcopy(identity["origin_matcher_pretraining_receipt"]))
    validate_progress(result, identity, rng_mode=mode)
    return result


def restore_continuation(model, optimizer, checkpoint, identity, *, rng_mode="exact_gpu"):
    number = validate_progress(checkpoint, identity, rng_mode=rng_mode)
    return _restore(model, optimizer, checkpoint, identity, number, rng_mode)

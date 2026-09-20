"""Exact S4-D1/S6-D4 C8 -> C16 controls; default is metadata-only preflight.

Private FunctionType adapters reuse the existing continuation bytecode without
mutating its globals or source. --execute is a separate, fail-closed opt-in.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import pickletools
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import continue_classifier as base

SCHEMA = "rachel-frozen-classifier-depth-continuation/1"
ROOT = base.ROOT
GPU_LOCK = ROOT / "scorer_diagnosis_20260919/heatmap-gpu.lock"
ARMS = {
    "s4_d1": (ROOT / "new_s345_20260914/s4_cross_attention/training", 1),
    "s6_d4": (ROOT / "attention_depth_20260915/s4_cross_attention_depth4/training", 4),
}
EXPECTED = {
    "s4_d1": {
        "checkpoint_sha256": "942732735b33e51cd646d99d95166204a5f50f3f3bf6223d9165089466b801ea",
        "resume_identity_sha256": "04a2abe00c3bb3a7ca181183f74b5f904590b8f2a7fe5b26be55ec207853e24a",
        "checkpoint_bytes": 2986654,
    },
    "s6_d4": {
        "checkpoint_sha256": "8c20dc3351877d35da8bcca30f1462f417f06efea4caf4ed57fc8e9c48e88f43",
        "resume_identity_sha256": "0cd6ae77b60d5eaf7af318d05d2541e45798e4abba1a3a41eaefd8bd464a02a5",
        "checkpoint_bytes": 5721374,
    },
}
EXPECTED_TRAIN_MANIFEST_SHA = dict.fromkeys(ARMS,
    "ca4a794aa2a5d196cc17c3e20144e3e5f9e2370eb952530193e6539de0b330af")
VAL_SHA = "daa6ccdd7686e93ba91ddfb1452c987145c26898a1917d2ac7d3180e199a8af8"
MATCHER_CHECKPOINT_SHA = "74fd1852df99b999d8f95f4ac1b013a05d5773c0ba85e616bb1ffc606be45129"
MATCHER_STATE_SHA = "175f765ada84c504ae854f9234e11db7659f53a84f2ef6be00eab75a528fe6c8"
old = base.old
private_function = base.private_function
load_decoupled_score_checkpoint = base.load_decoupled_score_checkpoint
plan = base.plan
train_segment = base.train_segment


def check_optimizer(checkpoint, model, *, expected_head_step):
    result = base.check_optimizer(checkpoint, model, expected_head_step=expected_head_step)
    state = checkpoint["optimizer_state_dict"]
    all_ids = [key for group in state["param_groups"] for key in group["params"]]
    if len(set(all_ids)) != len(all_ids) or set(state["state"]) - set(all_ids):
        raise ValueError("optimizer parameter IDs are duplicated or contain unknown state")
    for group, module in zip(state["param_groups"], (model.base_model, model.score_head)):
        if (tuple(group.get("betas", ())) != (0.9, 0.999) or group.get("eps") != 1e-8
                or group.get("amsgrad") is not False or group.get("maximize") is not False):
            raise ValueError("optimizer AdamW hyperparameters differ from the source recipe")
        for key, (name, parameter) in zip(group["params"], module.named_parameters()):
            item = state["state"].get(key)
            if item is None:
                continue
            step = item["step"]
            if (not isinstance(step, base.torch.Tensor) or step.numel() != 1
                    or not base.torch.isfinite(step).all() or float(step) != int(step)):
                raise ValueError("optimizer step is not a finite scalar integer: " + name)
            for field in ("exp_avg", "exp_avg_sq"):
                if item[field].dtype != parameter.dtype:
                    raise ValueError("optimizer moment dtype differs: " + name)
    return result


def validate_origin(origin, arm):
    if arm not in ARMS:
        raise ValueError("unregistered depth-control arm")
    if old.canonical_digest(origin) != EXPECTED[arm]["resume_identity_sha256"]:
        raise ValueError("source resume identity is not the pinned S4-D1/S6-D4 origin")
    populations = origin.get("populations", {})
    if (origin.get("model_options", {}).get("cross_attention_depth", 1) != ARMS[arm][1]
            or origin.get("matcher_checkpoint_sha256") != MATCHER_CHECKPOINT_SHA
            or populations.get("train", {}).get("manifest_sha256") != EXPECTED_TRAIN_MANIFEST_SHA[arm]
            or populations.get("val", {}).get("manifest_sha256") != VAL_SHA):
        raise ValueError("depth, shared Matcher, TRAIN or clean SIMVAL differs")


_validate_source = private_function(base.validate_source, ARMS=ARMS,
    EXPECTED_TRAIN_MANIFEST_SHA=EXPECTED_TRAIN_MANIFEST_SHA)


def validate_source(checkpoint, arm):
    validate_origin(checkpoint.get("resume_identity", {}), arm)
    if checkpoint.get("matcher_pretraining_receipt", {}).get("base_state_sha256") != MATCHER_STATE_SHA:
        raise ValueError("source Matcher state is not the pinned shared M12")
    model = _validate_source(checkpoint, arm)
    # No weight-only fallback: all used head parameters need real Adam moments.
    check_optimizer(checkpoint, model, expected_head_step=12000)
    return model


_make_identity = private_function(base.make_identity, SCHEMA=SCHEMA, __file__=__file__)


def make_identity(args, source, source_path, records):
    if args.restore_mode != "exact":
        raise ValueError("depth controls require exact optimizer/RNG recovery")
    validate_origin(source["resume_identity"], args.arm)
    if old._sha256(source_path) != EXPECTED[args.arm]["checkpoint_sha256"]:
        raise ValueError("source checkpoint differs from the pinned epoch20 SHA256")
    result = _make_identity(args, source, source_path, records)
    result.update(cross_attention_depth=ARMS[args.arm][1],
        source_registry=EXPECTED[args.arm],
        source_runtime_batching_sha256=old.canonical_digest(source["runtime_batching"]),
        reused_continuation_sha256=old._sha256(base.__file__),
        source_optimizer="same epoch_020.pt optimizer_state_dict; no external optimizer donor",
        shared_gpu_lock=str(GPU_LOCK))
    validate_identity(result)
    return result


def validate_identity(identity):
    arm = identity.get("arm")
    if arm not in ARMS:
        raise ValueError("unregistered depth-control continuation identity")
    required = dict(schema_version=SCHEMA, source_registry=EXPECTED[arm],
        source_checkpoint_sha256=EXPECTED[arm]["checkpoint_sha256"],
        source_resume_identity_sha256=EXPECTED[arm]["resume_identity_sha256"],
        cross_attention_depth=ARMS[arm][1], restore_mode="exact", optimizer_reset=False,
        rng_reset=False, source_epoch=20, final_epoch=28, source_classifier_epochs=8,
        final_classifier_epochs=16, additional_pair_exposures=192000,
        additional_optimizer_updates=12000, lr=2e-5, physical_microbatch=16,
        logical_microbatch=1, effective_batch=16, matcher_frozen=True,
        base_state_sha256=MATCHER_STATE_SHA, endpoint_evaluation_only=True,
        held_out_used_for_fit=False, primary_selection="fixed_epoch28")
    if any(identity.get(k) != v for k, v in required.items()):
        raise ValueError("depth-control identity/source/budget/restore contract differs")
    runtime_sha = identity.get("source_runtime_batching_sha256")
    if not isinstance(runtime_sha, str) or len(runtime_sha) != 64:
        raise ValueError("missing exact source runtime batching binding")


_validate_continuation = private_function(base.validate_continuation, SCHEMA=SCHEMA)


def validate_continuation(checkpoint, identity):
    validate_identity(identity)
    validate_origin(checkpoint.get("resume_identity", {}), identity["arm"])
    if old.canonical_digest(checkpoint.get("runtime_batching")) != identity["source_runtime_batching_sha256"]:
        raise ValueError("source runtime batching changed during continuation")
    return _validate_continuation(checkpoint, identity)


_restore = private_function(base.restore, validate_continuation=validate_continuation,
    check_optimizer=check_optimizer)


def restore(model, optimizer, checkpoint, identity, *, initial=False):
    validate_identity(identity)
    return _restore(model, optimizer, checkpoint, identity, initial=initial)


payload = private_function(base.payload, SCHEMA=SCHEMA, validate_continuation=validate_continuation)
publish_freeze = private_function(base.publish_freeze, SCHEMA=SCHEMA)
_run = private_function(base.run, SCHEMA=SCHEMA, ARMS=ARMS,
    validate_source=validate_source, make_identity=make_identity, restore=restore,
    payload=payload, publish_freeze=publish_freeze)


def preflight(arm, source_training=None):
    """Read JSON/stat/zip pickle metadata only; never deserialize tensor data."""
    if arm not in ARMS:
        raise ValueError("unregistered depth-control arm")
    root = Path(source_training or ARMS[arm][0]).resolve(strict=True)
    status = json.loads((root / "status.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    freeze = json.loads((root / "classifier_freezes/freeze.json").read_text())
    expected_status = dict(status="complete", epoch=20, completed_segments=80,
        global_exposure=480000, optimizer_updates=30000)
    if any(status.get(k) != v for k, v in expected_status.items()):
        raise ValueError("source status has not reached the completed C8 endpoint")
    if freeze.get("status") != "complete":
        raise ValueError("source classifier freeze is incomplete")
    validate_origin(freeze["resume_identity"], arm)
    runtime = protocol.get("runtime_batching", {})
    if (runtime.get("physical_microbatch") != 16 or runtime.get("effective_batch") != 16
            or runtime.get("logical_microbatch") != 1
            or runtime.get("origin_resume_identity_sha256") != EXPECTED[arm]["resume_identity_sha256"]):
        raise ValueError("source physical16/effective16 migration identity differs")
    selected = freeze["selections"]["fixed_epoch"]
    if (selected.get("selected_epoch") != 20
            or selected.get("checkpoint_sha256") != EXPECTED[arm]["checkpoint_sha256"]):
        raise ValueError("freeze does not bind the registered epoch20 hash")
    checkpoint_path = root / "epoch_020.pt"
    size = checkpoint_path.stat().st_size
    if size != EXPECTED[arm]["checkpoint_bytes"]:
        raise ValueError("source epoch20 checkpoint size changed")
    with zipfile.ZipFile(checkpoint_path) as archive:
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith("/data.pkl")))
    strings = {a for o, a, _ in pickletools.genops(metadata)
               if o.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE")}
    required = {"optimizer_state_dict", "param_groups", "exp_avg", "exp_avg_sq", "rng_state", "cuda"}
    if not required <= strings or "inference_only" in strings:
        raise ValueError("resumable optimizer/RNG pickle metadata absent or inference-only marker present")
    return dict(schema_version=SCHEMA, arm=arm, source_checkpoint=str(checkpoint_path),
        checkpoint_bytes=size, recorded_checkpoint_sha256=selected["checkpoint_sha256"],
        resume_identity_sha256=EXPECTED[arm]["resume_identity_sha256"],
        cross_attention_depth=ARMS[arm][1], physical_microbatch=16, effective_batch=16,
        optimizer_source=str(checkpoint_path) + ":optimizer_state_dict",
        rng_source=str(checkpoint_path) + ":rng_state", metadata_only=True,
        tensor_payloads_read=False, gpu_used=False, pending_execute_checks=[
            "recompute full checkpoint SHA256", "validate all active Adam moments/head step12000",
            "validate complete RNG and one-CUDA-device topology", "verify frozen shared Matcher digest"],
        additional_pair_exposures=192000, additional_optimizer_updates=12000,
        lock=str(GPU_LOCK), launches_training=False)


@contextmanager
def gpu_lock():
    """Same nonblocking lock as the active C16/heatmap suite, never bypassed."""
    with GPU_LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("shared GPU lock busy; no waiting or overlapping training") from error
        occupied = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if occupied:
            raise RuntimeError("GPU compute processes already present; will not overlap: " + occupied)
        yield


def run(args):
    if getattr(args, "restore_mode", "exact") != "exact":
        raise ValueError("warm-start is prohibited for the same-budget depth controls")
    receipt = preflight(args.arm, args.source_training)
    if not args.execute:
        return receipt
    if not args.output:
        raise ValueError("--output is required with --execute")
    output = Path(args.output).resolve()
    immutable = [p.resolve() for p, _ in ARMS.values()] + [Path(receipt["source_checkpoint"]).parent]
    if any(output == p or p in output.parents for p in immutable):
        raise ValueError("output must not be within either immutable source training directory")
    args.restore_mode = "exact"
    with gpu_lock():
        # Hash before the inherited torch.load; pickle metadata alone is not proof.
        if old._sha256(receipt["source_checkpoint"]) != EXPECTED[args.arm]["checkpoint_sha256"]:
            raise ValueError("source checkpoint differs from the pinned epoch20 SHA256")
        return _run(args)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--arm", required=True, choices=tuple(ARMS))
    result.add_argument("--source-training", help="optional byte-identical relocated source directory")
    result.add_argument("--output")
    result.add_argument("--execute", action="store_true", help="opt in only after queue authorization")
    result.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--smoke", type=int, choices=(16, 32, 64))
    result.set_defaults(restore_mode="exact")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True, indent=2))

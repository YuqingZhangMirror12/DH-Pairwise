"""Finite, isolated C1/C2 C8->C16 training. Nothing runs on import.

Only TRAIN/clean SIMVAL are reachable. Original C0 source/trainer are read-only.
No warm-start fallback: full global Adam/RNG are required, one new cold group.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import continue_classifier as cont
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local import model as local
from staging.pairwise_v0_2.models.rachel_decoupled_score import build_decoupled_score_model

old = cont.old
SCHEMA = "rachel-candidate-local-training/1"
SOURCE_SHA = "56d3a4949e9d7e10f6ab5bdadc0b8a50d17192a7130584c3b5d22e21ce4d2076"
MATCHER_SHA = "175f765ada84c504ae854f9234e11db7659f53a84f2ef6be00eab75a528fe6c8"
SOURCE = cont.ARMS["s6_d2"][0] / "epoch_020.pt"
GPU_LOCK = cont.ROOT / "scorer_diagnosis_20260919/heatmap-gpu.lock"
MODES = {"c1": "predicted_inliers", "c2": "all_valid_control"}


@contextmanager
def gpu_lock(path=GPU_LOCK, *, check_gpu=True):
    """Fail immediately on busy lock/device; never wait, kill or steal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if check_gpu:
            occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                "--format=csv,noheader"], text=True).strip().splitlines()
            foreign = [p.strip() for p in occupied if p.strip() and p.strip() != str(os.getpid())]
            if foreign:
                raise RuntimeError("GPU occupied; refusing overlap: " + ",".join(foreign))
        yield


def read_source(path):
    path = Path(path).resolve(strict=True)
    if old._sha256(path) != SOURCE_SHA:
        raise ValueError("source is not the pinned S6-D2 original epoch20 SHA")
    source = torch.load(path, map_location="cpu", weights_only=False)
    cont.validate_source(source, "s6_d2")  # rejects S8/step2048 and changed manifests
    if source["matcher_pretraining_receipt"]["base_state_sha256"] != MATCHER_SHA:
        raise ValueError("source frozen Matcher differs from S6-D2 M12")
    return source


def make_identity(arm, source, source_path, records):
    origin = source["resume_identity"]
    if old.canonical_digest(records) != old.canonical_digest(origin["populations"]):
        raise ValueError("TRAIN/clean SIMVAL manifest identity changed")
    return dict(schema_version=SCHEMA, arm=arm, mode=MODES[arm],
        source_checkpoint=str(source_path), source_checkpoint_sha256=SOURCE_SHA,
        source_resume_identity_sha256=old.canonical_digest(origin),
        base_state_sha256=MATCHER_SHA, source_epoch=20, final_epoch=28,
        source_classifier_epochs=8, final_classifier_epochs=16,
        additional_pair_exposures=192000, additional_optimizer_updates=12000,
        auxiliary_eligible_epoch_range=[21, 28], primary_selection="fixed_epoch28",
        physical_microbatch=16, effective_batch=16, logical_microbatch=1,
        lr=2e-5, weight_decay=1e-4, precision="fp32", workers=4,
        optimizer_groups=["base", "new_head", "candidate_local"],
        original_optimizer_and_rng_restored=True, new_group_cold=True,
        matcher_frozen=True, loss="unchanged samplewise PairBCE only",
        initialization="identical copied C8 CA; residual final readout strictly zero",
        decoder_config=asdict(local.DECODER_CONFIG), both_arms_decode_every_forward=True,
        target_blind_selection=True, selection_population="clean SIMVAL3000 only",
        held_out_used_for_fit=False, endpoint_evaluation_only=True,
        populations=records, bitwise_trajectory_equivalence_claimed=False,
        implementation_sha256={Path(p).name: old._sha256(p) for p in
            (__file__, local.__file__, cont.__file__, old.__file__)})


def validate_payload(saved, identity):
    if (saved.get("candidate_training_schema") != SCHEMA or saved.get("inference_only")
            or old.canonical_digest(saved.get("candidate_identity")) != old.canonical_digest(identity)):
        raise ValueError("candidate schema/identity mismatch")
    n = saved.get("completed_segments")
    if type(n) is not int or not 80 <= n <= 112:
        raise ValueError("candidate committed segments must be80..112")
    expected = dict(epoch=(n + 3) // 4, phase="classifier", global_exposure=n * 6000,
        optimizer_updates=n * 375, additional_pair_exposures=(n - 80) * 6000,
        additional_optimizer_updates=(n - 80) * 375)
    if any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError("candidate exposure/update ledger mismatch")
    if old.canonical_digest(saved["resume_identity"]) != identity["source_resume_identity_sha256"]:
        raise ValueError("source identity changed")
    old.validate_runtime_batching(saved.get("runtime_batching"), saved["resume_identity"], n)
    if (identity["source_checkpoint_sha256"] != SOURCE_SHA or identity["base_state_sha256"] != MATCHER_SHA
            or identity["mode"] != MODES[identity["arm"]]
            or saved["candidate_model"]["mode"] != identity["mode"]
            or saved["candidate_model"]["decoder_config"] != asdict(local.DECODER_CONFIG)):
        raise ValueError("source/arm/decoder not registered")
    if (saved["candidate_model"]["source_model"]["base_model_config"] != saved["resume_identity"]["base_model_config"]
            or saved["matcher_pretraining_receipt"]["base_state_sha256"] != identity["base_state_sha256"]):
        raise ValueError("candidate base/source receipt differs")
    return n


def load_model(saved):
    validate_payload(saved, saved["candidate_identity"])
    rng = old.capture_rng_state()
    try:
        metadata = saved["candidate_model"]["source_model"]
        source_model = build_decoupled_score_model(metadata["base_model_config"],
            "cross_attention", phase="classifier", model_options={"cross_attention_depth": 2})
        model = local.FrozenCandidateLocalModel(source_model, saved["candidate_identity"]["mode"])
        model.load_state_dict(saved["model_state_dict"], strict=True)
        if old.canonical_digest(model.metadata()) != old.canonical_digest(saved["candidate_model"]):
            raise ValueError("candidate model metadata mismatch")
        old.verify_receipt(model, saved["matcher_pretraining_receipt"], saved["resume_identity"])
        return model
    finally:
        old.restore_rng_state(rng)


def check_optimizer(saved, model):
    n = validate_payload(saved, saved["candidate_identity"])
    state = saved["optimizer_state_dict"]
    groups = state["param_groups"]
    if [g.get("phase_family") for g in groups] != ["base", "new_head", "candidate_local"]:
        raise ValueError("expected retained base/global plus exactly one candidate group")
    old_saved = dict(saved, optimizer_state_dict=dict(state=state["state"], param_groups=groups[:2]))
    facade = SimpleNamespace(base_model=model.base_model, score_head=model.score_head.global_head)
    cont.check_optimizer(old_saved, facade, expected_head_step=12000 + (n - 80) * 375)
    group, named = groups[2], list(model.score_head.local_head.named_parameters())
    if (len(group["params"]) != len(named) or group["lr"] != 2e-5 or group["weight_decay"] != 1e-4):
        raise ValueError("new group structure/LR differs")
    # All-ineligible batches do not traverse local CA, so those Adam steps may
    # legitimately lag total updates. Persist the exact per-parameter ledger;
    # never fabricate missing moments or pretend every parameter was active.
    expected_step = (n - 80) * 375
    for key, (name, parameter) in zip(group["params"], named):
        item = state["state"].get(key)
        recorded = saved["candidate_optimizer_steps"][name]
        if item is None and recorded is None:
            continue
        if item is None or not {"step", "exp_avg", "exp_avg_sq"} <= set(item):
            raise ValueError("missing candidate Adam state: " + name)
        if int(item["step"]) != recorded or not 0 <= recorded <= expected_step:
            raise ValueError("candidate Adam step differs: " + name)
        if any(item[k].shape != parameter.shape or not torch.isfinite(item[k]).all()
               for k in ("exp_avg", "exp_avg_sq")):
            raise ValueError("invalid candidate Adam moments: " + name)


def restore_rng(saved, device):
    if device.type == "cuda" and len(saved["rng_state"]["cuda"]) != torch.cuda.device_count():
        raise ValueError("CUDA RNG topology differs; exact restore requires one visible GPU")
    old.restore_rng_state(saved["rng_state"])


def payload(model, optimizer, source, identity, number, winners):
    saved = dict(candidate_training_schema=SCHEMA, candidate_identity=identity,
        candidate_model=model.metadata(), model_state_dict=old._cpu_model_state(model),
        optimizer_state_dict=optimizer.state_dict(), rng_state=old.capture_rng_state(),
        resume_identity=source["resume_identity"], runtime_batching=source["runtime_batching"],
        matcher_pretraining_receipt=source["matcher_pretraining_receipt"],
        loss_config=source["loss_config"], seed=old.SEED, phase="classifier",
        completed_segments=number, epoch=(number + 3) // 4, global_exposure=number * 6000,
        optimizer_updates=number * 375, additional_pair_exposures=(number - 80) * 6000,
        additional_optimizer_updates=(number - 80) * 375, winners=deepcopy(winners),
        candidate_optimizer_steps={name: int(optimizer.state[p]["step"])
            if p in optimizer.state and "step" in optimizer.state[p] else None
            for name, p in model.score_head.local_head.named_parameters()},
        formal_training_counted=True)
    check_optimizer(saved, model)
    return saved


def publish_freeze(root, identity, winners):
    if (set(winners) != {"fixed_epoch", "max_f1", "recall95"}
            or winners["fixed_epoch"]["selected_epoch"] != 28
            or any(not 21 <= v["selected_epoch"] <= 28 for v in winners.values())):
        raise ValueError("requires complete own-C9..C16 SIMVAL selections")
    old.save_json(root / "classifier_freezes/freeze.json", dict(schema_version=SCHEMA,
        status="complete", budget_epochs=28, eligible_epoch_range=[21, 28],
        candidate_identity=identity, candidate_identity_sha256=old.canonical_digest(identity),
        selections={k: dict(v, checkpoint_sha256=old._sha256(v["checkpoint"])) for k, v in winners.items()},
        primary_selection="fixed_epoch", held_out_used_for_fit=False,
        selection_population="clean SIMVAL3000 only", endpoint_evaluation_only=True))


def run(args):
    if args.resume and args.smoke:
        raise ValueError("discard smoke cannot resume")
    live = cont.ROOT / "scorer_diagnosis_20260919/continuation_v1"
    output = Path(args.output).resolve()
    if output == live or live in output.parents:
        raise ValueError("candidate experiment may not write into live C0 continuation outputs")
    if platform.system() != "Linux" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal/smoke CLI requires remote Linux with exactly one visible CUDA GPU")
    torch.set_num_threads(1)
    # Shared nonblocking lock is acquired BEFORE model CUDA allocation.
    with gpu_lock():
        return run_locked(args)


def run_locked(args, *, device=None):
    old.runner._set_determinism(old.SEED)
    source_path = Path(args.source).resolve(strict=True)
    source = read_source(source_path)
    origin = source["resume_identity"]
    data_args = SimpleNamespace(sampling="original512",
        train_materialized_manifest=origin["populations"]["train"]["manifest"],
        dataset=str(Path(origin["populations"]["val"]["manifest"]).parents[1]))
    training, validation, cap, records = old.make_populations(data_args)
    identity = make_identity(args.arm, source, source_path, records)
    output = Path(args.output).resolve()
    if output == source_path.parent or source_path.parent in output.parents:
        raise ValueError("output may not overlap original source directory")
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    if args.resume and not (output / "last.pt").is_file():
        raise ValueError("resume requires committed last.pt")
    args.output = str(output)
    args.microbatch, args.physical_microbatch, args.effective_batch = 1, 16, 16
    args.runtime_effective_batch, args.workers, args.log_every = None, 4, 1000
    device = torch.device("cuda:0") if device is None else device  # CPU injected only by synthetic tests
    model = local.build_from_s6_epoch20(source, MODES[args.arm]).to(device)
    optimizer, optimizer_receipt = local.restore_source_optimizer(source, model)
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False) if args.resume else source
    completed, winners = 80, {}
    if args.resume:
        completed = validate_payload(checkpoint, identity)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        check_optimizer(checkpoint, model)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        winners = deepcopy(checkpoint["winners"])
    # Source RNG restoration follows ALL initialization, datasets and optimizer.
    restore_rng(checkpoint, device)
    model.set_phase("classifier").train()
    old.verify_receipt(model, source["matcher_pretraining_receipt"], origin)
    protocol = dict(schema_version=SCHEMA, candidate_identity=identity, arguments=vars(args),
        optimizer_receipt=optimizer_receipt, status="running", plan=cont.plan(),
        smoke=bool(args.smoke), formal_training_counted=not bool(args.smoke),
        pid=os.getpid(), gpu_lock=str(GPU_LOCK), queue_does_not_overlap_live_C0=True)
    old.save_json(output / "protocol.json", protocol)
    hook = cont.install_first_update_receipt(optimizer, model, output, smoke=bool(args.smoke))
    if not args.resume and not args.smoke:
        old.runner._atomic_torch_save(output / "last.pt", payload(model, optimizer, source, identity, 80, {}))
    try:
        for segment in cont.plan():
            number, epoch = segment["number"], segment["epoch"]
            if number <= completed:
                continue
            model.set_phase("classifier").train()
            order = old.runner.epoch_indices(old.TRAIN_COUNT, seed=old.SEED, epoch=epoch, limit=None)
            count = args.smoke or old.SEGMENT_SIZE
            indices = order[segment["offset"]:segment["offset"] + count]
            loader = old.make_weathering_loader(training, indices, batch_size=16, num_workers=4,
                seed=old.SEED + number, contour_cap=cap)
            old.save_json(output / "status.json", dict(status="running", pid=os.getpid(), epoch=epoch,
                phase="classifier", segment=number, additional_pair_exposures=(completed - 80) * 6000))
            model.reset_decoder_cost()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            report = cont.train_segment(model, loader, optimizer,
                old.RachelN512LossConfig(**source["loss_config"]), device, args, epoch)
            report["candidate_decoder_cost"] = deepcopy(model.decoder_cost)
            if (report["samples"], report["optimizer_updates"], model.decoder_cost["pairs"]) != (count, count // 16, count):
                raise ValueError("actual sample/update/decoder budget mismatch")
            old.verify_receipt(model, source["matcher_pretraining_receipt"], origin)
            old.save_json(output / ("segment_%03d.json" % number), dict(segment=segment, training=report,
                formal_training_counted=not bool(args.smoke)))
            if args.smoke:
                result = dict(status="smoke_complete", pair_exposures=count, optimizer_updates=count // 16,
                    formal_training_counted=False, weights_discarded=True, no_checkpoint_written=True,
                    frozen_base_unchanged=True, candidate_identity=identity, training=report)
                old.save_json(output / "smoke.json", result)
                break
            if segment["epoch_complete"]:
                from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
                old.save_json(output / "status.json", dict(status="running", phase="SIMVAL", epoch=epoch))
                loader_val = old.make_ablation_loader(validation, list(range(old.VAL_COUNT)), batch_size=1,
                    num_workers=4, seed=old.SEED, contour_cap=cap)
                model.reset_decoder_cost()
                val, rows = old.evaluate_pair_validation(model, loader_val, device)
                points = fit_operating_points([r["label"] for r in rows], [r["classification"]["fused"] for r in rows])
                old.save_json(output / ("validation_%03d_rows.json" % epoch), rows)
                old.save_json(output / ("validation_%03d.json" % epoch), dict(epoch=epoch,
                    validation=val, operating_points=points, candidate_decoder_cost=model.decoder_cost))
                winners = cont.update_winners(winners, output, epoch, val, points)
            saved = payload(model, optimizer, source, identity, number, winners)
            if segment["epoch_complete"]:
                old.runner._atomic_torch_save(output / ("epoch_%03d.pt" % epoch), saved)
            old.runner._atomic_torch_save(output / "last.pt", saved)
            completed = number
        else:
            if completed != 112:
                raise ValueError("fixedC16 incomplete")
            publish_freeze(output, identity, winners)
            result = dict(status="complete", epoch=28, completed_segments=112,
                additional_pair_exposures=192000, additional_optimizer_updates=12000,
                phase="train_val_complete", held_out_evaluated=False)
        protocol.update(result)
        old.save_json(output / "protocol.json", protocol)
        old.save_json(output / "status.json", result)
        return result
    except BaseException as error:
        protocol.update(status="failed", error=repr(error), last_committed_segment=completed,
            recovery="explicit --resume only; reload committed last.pt and repeat uncommitted segment")
        old.save_json(output / "protocol.json", protocol)
        old.save_json(output / "status.json", protocol)
        raise
    finally:
        hook.remove()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=tuple(MODES), required=True)
    p.add_argument("--source", default=str(SOURCE), help="relocated source allowed only with same pinned SHA")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", type=int, choices=(16, 32, 64))
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))

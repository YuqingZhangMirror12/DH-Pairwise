"""Fixed S6-D2 C8->C16 spectral controls; import is side-effect free."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training import cache, model as net
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_head import features as f, model as head
from staging.pairwise_v0_2.models.rachel_decoupled_score import build_decoupled_score_model

shared, old, cont = cache.shared, cache.shared.old, cache.shared.cont
SCHEMA = "rachel-spectral-training/1"
VARIANTS = head.VARIANTS


def identity(args, source, bundle, populations):
    origin = source["resume_identity"]
    if old.canonical_digest(populations) != old.canonical_digest(origin["populations"]):
        raise ValueError("training population differs from same S6 source")
    for split in ("train", "val"):
        record = bundle["splits"][split]
        if (record["pair_count"] != cache.SPLIT_COUNTS[split]
                or record["identity"]["input_manifest_sha256"] != origin["populations"][split]["manifest_sha256"]):
            raise ValueError("cache does not cover complete same TRAIN/VAL manifest")
    return dict(schema_version=SCHEMA, variant=args.variant,
        source_checkpoint=str(Path(args.source).resolve()), source_checkpoint_sha256=shared.SOURCE_SHA,
        source_resume_identity_sha256=old.canonical_digest(origin), base_state_sha256=shared.MATCHER_SHA,
        cache_bundle=str(Path(args.cache_bundle).resolve()), cache_bundle_sha256=args.cache_bundle_sha,
        cache_records={s: bundle["splits"][s] for s in ("train", "val")},
        normalizer=bundle["normalizer"], normalizer_sha256=f.digest_json(bundle["normalizer"]),
        source_classifier_epochs=8, final_classifier_epochs=16, final_epoch=28,
        additional_pair_exposures=192000, additional_optimizer_updates=12000,
        physical_microbatch=16, effective_batch=16, logical_microbatch=1, workers=4,
        lr=2e-5, weight_decay=1e-4, precision="fp32", residual_seed=net.RESIDUAL_SEED,
        ca_trainable=True, matcher_frozen=True, optimizer_groups=["base", "new_head", "spectral_residual"],
        original_optimizer_rng_restored=True, new_residual_parameters=193, new_group_cold=True,
        loss="unchanged samplewise PairBCE only", all_arms_share_10D_normalizer_before_mask=True,
        primary_selection="fixed_epoch28", auxiliary_eligible_epoch_range=[21, 28],
        selection_population="clean SIMVAL3000 only", held_out_used_for_fit=False,
        endpoint_evaluation_only=True, svd_in_training=False, populations=populations,
        implementation_sha256={str(Path(p).resolve()): old._sha256(p) for p in
            (__file__, net.__file__, cache.__file__, head.__file__, f.__file__, cont.__file__, old.__file__)})


def cpu_state(module):
    # Spectral normalizer and scorer have checked dict-valued extra_state.
    return {key: value.detach().cpu() if torch.is_tensor(value) else deepcopy(value)
            for key, value in module.state_dict().items()}


def validate_payload(saved, ident):
    if (saved.get("spectral_training_schema") != SCHEMA
            or f.digest_json(saved.get("spectral_identity")) != f.digest_json(ident)):
        raise ValueError("spectral checkpoint schema/identity changed")
    n = saved.get("completed_segments")
    if type(n) is not int or not 80 <= n <= 112:
        raise ValueError("expected committed segments80..112")
    expected = dict(epoch=(n + 3) // 4, phase="classifier", global_exposure=n * 6000,
        optimizer_updates=n * 375, additional_pair_exposures=(n - 80) * 6000,
        additional_optimizer_updates=(n - 80) * 375)
    if any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError("spectral epoch/exposure/update ledger differs")
    if (ident["source_checkpoint_sha256"] != shared.SOURCE_SHA or ident["base_state_sha256"] != shared.MATCHER_SHA
            or ident["variant"] not in VARIANTS or ident["normalizer_sha256"] != f.digest_json(ident["normalizer"])
            or f.digest_json(saved["resume_identity"]) != ident["source_resume_identity_sha256"]
            or saved["model_design"]["scorer"]["variant"] != ident["variant"]
            or saved["model_design"]["scorer"]["standardizer_sha256"] != ident["normalizer_sha256"]):
        raise ValueError("source/variant/TRAIN normalizer differs")
    old.validate_runtime_batching(saved["runtime_batching"], saved["resume_identity"], n)
    return n


def check_optimizer(saved, model):
    n = validate_payload(saved, saved["spectral_identity"])
    state = saved["optimizer_state_dict"]
    if [g.get("phase_family") for g in state["param_groups"]] != ["base", "new_head", "spectral_residual"]:
        raise ValueError("original groups plus exactly one residual group required")
    facade = SimpleNamespace(base_model=model.base_model, score_head=model.score_head.ca)
    cont.check_optimizer(dict(saved, optimizer_state_dict=dict(state=state["state"],
        param_groups=state["param_groups"][:2])), facade, expected_head_step=12000 + (n - 80) * 375)
    group = state["param_groups"][2]
    named = list(model.score_head.residual.named_parameters())
    if len(group["params"]) != len(named) or group["lr"] != 2e-5 or group["weight_decay"] != 1e-4:
        raise ValueError("residual optimizer structure differs")
    for key, (name, p) in zip(group["params"], named):
        item = state["state"].get(key)
        if item is None and n == 80:
            continue
        if (item is None or not {"step", "exp_avg", "exp_avg_sq"} <= set(item)
                or int(item["step"]) != (n - 80) * 375
                or any(item[k].shape != p.shape or not torch.isfinite(item[k]).all() for k in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("residual Adam state missing/invalid: " + name)


def payload(model, optimizer, source, ident, number, winners):
    saved = dict(spectral_training_schema=SCHEMA, spectral_identity=ident, model_design=model.metadata(),
        model_state_dict=cpu_state(model), optimizer_state_dict=optimizer.state_dict(), rng_state=old.capture_rng_state(),
        # Keep original CA only for deterministic reconstruction/extra-state verification;
        # no original full checkpoint or optimizer duplicated here.
        source_ca_state_dict={k[len("score_head."):]: v.detach().cpu() for k, v in source["model_state_dict"].items()
                              if k.startswith("score_head.")},
        resume_identity=source["resume_identity"], runtime_batching=source["runtime_batching"],
        matcher_pretraining_receipt=source["matcher_pretraining_receipt"], loss_config=source["loss_config"],
        seed=old.SEED, phase="classifier", completed_segments=number, epoch=(number + 3) // 4,
        global_exposure=number * 6000, optimizer_updates=number * 375,
        additional_pair_exposures=(number - 80) * 6000, additional_optimizer_updates=(number - 80) * 375,
        winners=deepcopy(winners), formal_training_counted=True)
    check_optimizer(saved, model)
    return saved


def load_model(saved):
    ident = saved["spectral_identity"]
    validate_payload(saved, ident)
    rng = old.capture_rng_state()
    try:
        original = build_decoupled_score_model(saved["resume_identity"]["base_model_config"],
            "cross_attention", phase="classifier", model_options={"cross_attention_depth": 2})
        original.score_head.load_state_dict(saved["source_ca_state_dict"], strict=True)
        model = net.FrozenSpectralModel(original, ident["normalizer"], ident["variant"])
        model.load_state_dict(saved["model_state_dict"], strict=True)
        if f.digest_json(model.metadata()) != f.digest_json(saved["model_design"]):
            raise ValueError("model/scorer metadata mismatch")
        old.verify_receipt(model, saved["matcher_pretraining_receipt"], saved["resume_identity"])
        return model
    finally:
        old.restore_rng_state(rng)


def publish_freeze(root, ident, winners):
    if (set(winners) != {"fixed_epoch", "max_f1", "recall95"}
            or winners["fixed_epoch"]["selected_epoch"] != 28
            or any(not 21 <= x["selected_epoch"] <= 28 for x in winners.values())):
        raise ValueError("requires ownC9..C16 completed selections")
    old.save_json(root / "classifier_freezes/freeze.json", dict(schema_version=SCHEMA, status="complete",
        spectral_identity=ident, spectral_identity_sha256=f.digest_json(ident), budget_epochs=28,
        eligible_epoch_range=[21, 28], held_out_used_for_fit=False,
        selections={k: dict(v, checkpoint_sha256=old._sha256(v["checkpoint"])) for k, v in winners.items()}))


def run_locked(args, *, device=None):
    source_path = Path(args.source).resolve(strict=True)
    source = shared.read_source(source_path)
    bundle, caches = cache.load_bundle(args.cache_bundle, args.cache_bundle_sha)
    training, validation, cap, populations = cache.populations(source)
    ident = identity(args, source, bundle, populations)
    root = Path(args.output).resolve()
    if root == source_path.parent or source_path.parent in root.parents:
        raise ValueError("output inside immutable source")
    root.mkdir(parents=True, exist_ok=bool(args.resume))
    args.output = str(root)
    args.microbatch, args.physical_microbatch, args.effective_batch = 1, 16, 16
    args.runtime_effective_batch, args.workers, args.log_every = None, 4, 1000
    device = torch.device("cuda:0") if device is None else device
    model = net.build(source, bundle["normalizer"], args.variant).to(device)
    optimizer = net.restore_optimizer(source, model)
    checkpoint = torch.load(root / "last.pt", map_location="cpu", weights_only=False) if args.resume else source
    completed, winners = 80, {}
    if args.resume:
        completed = validate_payload(checkpoint, ident)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        check_optimizer(checkpoint, model)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        winners = deepcopy(checkpoint["winners"])
    shared.restore_rng(checkpoint, device)  # AFTER every construction and cache read
    model.set_phase("classifier").train()
    old.verify_receipt(model, source["matcher_pretraining_receipt"], source["resume_identity"])
    protocol = dict(schema_version=SCHEMA, spectral_identity=ident, status="running", plan=cont.plan(),
        arguments=vars(args), smoke=bool(args.smoke), formal_training_counted=not bool(args.smoke))
    old.save_json(root / "protocol.json", protocol)
    hook = cont.install_first_update_receipt(optimizer, model, root, smoke=bool(args.smoke))
    if not args.resume and not args.smoke:
        old.runner._atomic_torch_save(root / "last.pt", payload(model, optimizer, source, ident, 80, {}))
    try:
        for segment in cont.plan():
            number, epoch = segment["number"], segment["epoch"]
            if number <= completed:
                continue
            model.set_phase("classifier").train()
            order = old.runner.epoch_indices(old.TRAIN_COUNT, seed=old.SEED, epoch=epoch, limit=None)
            count = args.smoke or 6000
            indices = order[segment["offset"]:segment["offset"] + count]
            loader = old.make_weathering_loader(training, indices, batch_size=16, num_workers=4,
                seed=old.SEED + number, contour_cap=cap)
            loader = net.BoundLoader(loader, model, caches["train"])
            model.reset_binding_cost()
            old.save_json(root / "status.json", dict(status="running", phase="classifier", epoch=epoch, segment=number))
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            report = cont.train_segment(model, loader, optimizer, old.RachelN512LossConfig(**source["loss_config"]), device, args, epoch)
            report["cached_summary_binding"] = deepcopy(model.binding_cost)
            if (report["samples"], report["optimizer_updates"], model.binding_cost["pairs"]) != (count, count // 16, count):
                raise ValueError("sample/update/cache binding budget differs")
            old.verify_receipt(model, source["matcher_pretraining_receipt"], source["resume_identity"])
            old.save_json(root / ("segment_%03d.json" % number), dict(segment=segment, training=report))
            if args.smoke:
                result = dict(status="smoke_complete", pair_exposures=count, optimizer_updates=count // 16,
                    formal_training_counted=False, weights_discarded=True, no_checkpoint_written=True,
                    frozen_base_unchanged=True, spectral_identity=ident, training=report)
                old.save_json(root / "smoke.json", result)
                break
            if segment["epoch_complete"]:
                from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
                loader_val = old.make_ablation_loader(validation, list(range(old.VAL_COUNT)), batch_size=1,
                    num_workers=4, seed=old.SEED, contour_cap=cap)
                model.reset_binding_cost()
                val, rows = old.evaluate_pair_validation(model, net.BoundLoader(loader_val, model, caches["val"]), device)
                points = fit_operating_points([r["label"] for r in rows], [r["classification"]["fused"] for r in rows])
                old.save_json(root / ("validation_%03d_rows.json" % epoch), rows)
                old.save_json(root / ("validation_%03d.json" % epoch), dict(epoch=epoch, validation=val,
                    operating_points=points, cached_summary_binding=model.binding_cost))
                winners = cont.update_winners(winners, root, epoch, val, points)
            saved = payload(model, optimizer, source, ident, number, winners)
            if segment["epoch_complete"]:
                old.runner._atomic_torch_save(root / ("epoch_%03d.pt" % epoch), saved)
            old.runner._atomic_torch_save(root / "last.pt", saved)
            completed = number
        else:
            if completed != 112:
                raise ValueError("C16 incomplete")
            publish_freeze(root, ident, winners)
            result = dict(status="complete", epoch=28, completed_segments=112,
                additional_pair_exposures=192000, additional_optimizer_updates=12000, held_out_evaluated=False)
        protocol.update(result)
        old.save_json(root / "protocol.json", protocol)
        old.save_json(root / "status.json", result)
        return result
    except BaseException as error:
        protocol.update(status="failed", error=repr(error), last_committed_segment=completed,
            recovery="explicit --resume only; replay uncommitted segment")
        old.save_json(root / "protocol.json", protocol)
        old.save_json(root / "status.json", protocol)
        raise
    finally:
        hook.remove()


def run(args):
    if args.resume and args.smoke:
        raise ValueError("discard smoke cannot resume")
    if platform.system() != "Linux" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal CLI requires one visible remote CUDA GPU")
    live = cont.ROOT / "scorer_diagnosis_20260919/continuation_v1"
    if Path(args.output).resolve() == live or live in Path(args.output).resolve().parents:
        raise ValueError("cannot write into live C0")
    torch.set_num_threads(1)
    with shared.gpu_lock():
        return run_locked(args)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--source", default=str(shared.SOURCE))
    p.add_argument("--cache-bundle", required=True)
    p.add_argument("--cache-bundle-sha", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", type=int, choices=(16, 32, 64))
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))

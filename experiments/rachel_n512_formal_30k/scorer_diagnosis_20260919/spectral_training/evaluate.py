"""Original final evaluator + input-bound offline summary, no SVD at inference."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as original
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training import train, cache


def training_implementation_sha256():
    """Exactly the implementation closure recorded by train.identity."""
    return {str(Path(p).resolve()): train.old._sha256(p) for p in
            (train.__file__, train.net.__file__, cache.__file__, train.head.__file__,
             train.f.__file__, train.cont.__file__, train.old.__file__)}


def inference_runtime(device, summary_bundle_sha256, summary_record):
    """Bind scorer code AND offline summary in fields PredictionReuse compares."""
    runtime = original.inference_runtime(device)
    root = Path(original.__file__).resolve().parents[2]
    sources = training_implementation_sha256()
    sources[str(Path(__file__).resolve())] = train.old._sha256(__file__)
    runtime["source_sha256"] = dict(runtime["source_sha256"])
    runtime["source_sha256"].update({str(Path(path).relative_to(root)): sha for path, sha in sources.items()})
    runtime["spectral_summary_binding"] = dict(schema_version="rachel-spectral-endpoint-runtime/1",
        bundle_sha256=summary_bundle_sha256, cache_sha256=summary_record["sha256"],
        cache_identity_sha256=train.f.digest_json(summary_record["identity"]))
    return runtime


def load_frozen_model(root, selection):
    root = Path(root).resolve(strict=True)
    freeze_path = root / "classifier_freezes/freeze.json"
    freeze = json.loads(freeze_path.read_text())
    ident = freeze.get("spectral_identity", {})
    status = json.loads((root / "status.json").read_text())
    if (freeze.get("schema_version") != train.SCHEMA or freeze.get("status") != "complete"
            or freeze.get("budget_epochs") != 28 or freeze.get("eligible_epoch_range") != [21, 28]
            or freeze.get("held_out_used_for_fit") is not False
            or freeze.get("spectral_identity_sha256") != train.f.digest_json(ident)
            or status.get("status") != "complete" or status.get("completed_segments") != 112):
        raise ValueError("held-out evaluation requires complete same-budget C16 freeze")
    if ident.get("implementation_sha256") != training_implementation_sha256():
        raise ValueError("spectral implementation changed since training freeze; require frozen source")
    chosen = freeze["selections"][selection]
    epoch = chosen["selected_epoch"]
    path = Path(chosen["checkpoint"]).resolve(strict=True)
    if (not 21 <= epoch <= 28 or selection == "fixed_epoch" and epoch != 28
            or chosen.get("test_or_real_or_ood_used_for_fit") is not False
            or path != root / ("epoch_%03d.pt" % epoch)
            or train.old._sha256(path) != chosen["checkpoint_sha256"]):
        raise ValueError("selection path/epoch/hash differs")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    train.validate_payload(saved, ident)
    if saved["completed_segments"] != epoch * 4:
        raise ValueError("selection is not complete epoch")
    model = train.load_model(saved)
    origin = saved["resume_identity"]
    receipt = dict(training_run=str(root), budget=28, selection=selection, epoch=epoch,
        freeze_path=str(freeze_path), freeze_sha256=train.old._sha256(freeze_path),
        checkpoint_path=str(path), checkpoint_sha256=chosen["checkpoint_sha256"],
        seed=origin["seed"], model_config=asdict(model.config), architecture="ca_spectral_residual",
        sampling=origin["sampling"], classifier_thresholds=chosen["classifier_thresholds"],
        operating_points=chosen["operating_points"], winner_record=chosen,
        training_identity=origin, spectral_identity=ident, model_design=model.metadata(),
        classifier_only_pair_bce=True, coarse_is_untrained_diagnostic=True,
        local_and_fused_are_same_single_classifier=True, test_or_real_used_for_fit=False,
        ood_used_for_fit=False, endpoint_only=True, adapter_sha256=train.old._sha256(__file__))
    return model.eval().requires_grad_(False), receipt


def bind_and_predict(model, batch, device):
    model.bind_batch(batch, model.endpoint_cache)
    return original.core.predict_batch(model, batch, device)


def run(args):
    bundle, caches = cache.load_bundle(args.summary_bundle, args.summary_bundle_sha, require_training=False)
    if set(caches) != {args.split} or len(caches[args.split].records) != cache.SPLIT_COUNTS[args.split]:
        raise ValueError("endpoint requires matching complete split cache, no combined fit")

    def loader(root, selection):
        model, receipt = load_frozen_model(root, selection)
        model.endpoint_cache = caches[args.split]
        receipt["endpoint_summary_cache"] = bundle["splits"][args.split]
        receipt["summary_bundle_sha256"] = args.summary_bundle_sha
        return model, receipt

    core = SimpleNamespace(**{**vars(original.core), "predict_batch": bind_and_predict})
    def runtime(device):
        return inference_runtime(device, args.summary_bundle_sha, bundle["splits"][args.split])
    evaluate = train.cont.private_function(original.run, core=core, load_frozen_model=loader,
                                          inference_runtime=runtime)
    if args.device != "cuda:0":
        raise ValueError("formal endpoint uses one visible CUDA GPU; CPU tests call loader directly")
    with train.shared.gpu_lock():
        return evaluate(args)


def parser():
    p = original.parser()
    p.add_argument("--summary-bundle", required=True)
    p.add_argument("--summary-bundle-sha", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

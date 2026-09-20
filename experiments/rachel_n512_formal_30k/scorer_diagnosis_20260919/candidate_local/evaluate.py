"""Endpoint-only original evaluator with a strict candidate-local loader."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as original
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local import train


def training_implementation_sha256():
    """Match the source closure recorded by candidate train.make_identity."""
    return {Path(path).name: train.old._sha256(path) for path in
            (train.__file__, train.local.__file__, train.cont.__file__, train.old.__file__)}


def inference_runtime(device):
    """Bind candidate forward/adapter code in the reuse-comparison contract.

    The original runtime omits this out-of-tree head. Rebinding privately keeps
    the public evaluator unchanged while preventing stale sibling reuse after
    a parameter-free candidate/decoder code change.
    """
    runtime = original.inference_runtime(device)
    root = Path(original.__file__).resolve().parents[2]
    files = (Path(train.local.__file__), Path(train.__file__), Path(__file__),
             Path(train.local.estimate_translation_layout.__code__.co_filename))
    runtime["source_sha256"] = dict(runtime["source_sha256"])
    runtime["source_sha256"].update({str(path.resolve().relative_to(root)): train.old._sha256(path)
                                     for path in files})
    runtime["candidate_runtime_binding"] = "candidate-local-head-adapter-decoder/1"
    return runtime


def load_frozen_model(root, selection):
    root = Path(root).resolve(strict=True)
    freeze_path = root / "classifier_freezes/freeze.json"
    freeze = json.loads(freeze_path.read_text())
    ident = freeze.get("candidate_identity", {})
    status = json.loads((root / "status.json").read_text())
    if (freeze.get("schema_version") != train.SCHEMA or freeze.get("status") != "complete"
            or freeze.get("held_out_used_for_fit") is not False
            or freeze.get("eligible_epoch_range") != [21, 28] or freeze.get("budget_epochs") != 28
            or freeze.get("candidate_identity_sha256") != train.old.canonical_digest(ident)
            or status.get("status") != "complete" or status.get("completed_segments") != 112):
        raise ValueError("held-out evaluation requires completed fixed-C16 SIMVAL-only freeze")
    if ident.get("implementation_sha256") != training_implementation_sha256():
        raise ValueError("candidate implementation changed since training freeze; evaluation requires the frozen source")
    chosen = freeze["selections"][selection]
    epoch = chosen["selected_epoch"]
    if (not 21 <= epoch <= 28 or selection == "fixed_epoch" and epoch != 28
            or chosen.get("test_or_real_or_ood_used_for_fit") is not False):
        raise ValueError("unregistered candidate epoch or target-informed selection")
    path = Path(chosen["checkpoint"]).resolve(strict=True)
    if path != root / ("epoch_%03d.pt" % epoch) or train.old._sha256(path) != chosen["checkpoint_sha256"]:
        raise ValueError("selected checkpoint ownership/hash mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    train.validate_payload(saved, ident)
    if saved["completed_segments"] != epoch * 4:
        raise ValueError("selected checkpoint is not an epoch boundary")
    model = train.load_model(saved)
    origin = saved["resume_identity"]
    receipt = dict(training_run=str(root), budget=28, selection=selection, epoch=epoch,
        freeze_path=str(freeze_path), freeze_sha256=train.old._sha256(freeze_path),
        checkpoint_path=str(path), checkpoint_sha256=chosen["checkpoint_sha256"],
        seed=origin["seed"], model_config=asdict(model.config), architecture="candidate_local_ca_residual",
        sampling=origin["sampling"], classifier_thresholds=chosen["classifier_thresholds"],
        operating_points=chosen["operating_points"], winner_record=chosen,
        training_identity=origin, candidate_identity=ident, model_design=model.metadata(),
        classifier_only_pair_bce=True, coarse_is_untrained_diagnostic=True,
        local_and_fused_are_same_single_classifier=True, test_or_real_used_for_fit=False,
        ood_used_for_fit=False, endpoint_only=True,
        evaluation_adapter_sha256=train.old._sha256(__file__))
    return model.eval().requires_grad_(False), receipt


# Public evaluator/GT-free prediction decoder remain byte-for-byte unchanged.
_run = train.cont.private_function(original.run, load_frozen_model=load_frozen_model,
                                  inference_runtime=inference_runtime)


def run(args):
    if args.device != "cuda:0":
        raise ValueError("formal endpoint CLI uses cuda:0; CPU tests call loader directly")
    with train.gpu_lock():
        return _run(args)


if __name__ == "__main__":
    run(original.parser().parse_args())

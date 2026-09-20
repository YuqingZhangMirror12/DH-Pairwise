"""Endpoint-only adapter: original evaluator/decoder, honest C16 provenance.

Do not pass continuation checkpoints to the original C8 evaluator. It correctly
rejects epoch>20. This adapter substitutes only its frozen-model loader in a
private function namespace, without changing public code or pretending C16=C8.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluator
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import continue_classifier as train


def load_frozen_model(root, selection):
    root = Path(root).resolve(strict=True)
    path = root / "classifier_freezes/freeze.json"
    freeze = json.loads(path.read_text())
    identity = freeze.get("continuation_identity", {})
    if (freeze.get("schema_version") != train.SCHEMA or freeze.get("status") != "complete"
            or freeze.get("eligible_epoch_range") != [13, 28]
            or freeze.get("held_out_used_for_fit") is not False
            or freeze.get("budget_epochs") != 28
            or freeze.get("continuation_identity_sha256") != train.old.canonical_digest(identity)):
        raise ValueError("requires complete fixed-C16 SIMVAL-only freeze")
    status = json.loads((root / "status.json").read_text())
    if status.get("status") != "complete" or status.get("completed_segments") != 112:
        raise ValueError("TEST/REAL/OOD prohibited before the full C16 endpoint")
    selected = freeze["selections"][selection]
    epoch = selected["selected_epoch"]
    if (not 13 <= epoch <= 28 or selection == "fixed_epoch" and epoch != 28
            or selected.get("test_or_real_or_ood_used_for_fit") is not False):
        raise ValueError("unregistered winner epoch/selection population")
    checkpoint_path = Path(selected["checkpoint"]).resolve(strict=True)
    owner = Path(identity["source_checkpoint"]).resolve().parent if epoch <= 20 else root
    if checkpoint_path != owner / ("epoch_%03d.pt" % epoch):
        raise ValueError("winner is outside the registered origin/continuation directories")
    sha = train.old._sha256(checkpoint_path)
    if sha != selected["checkpoint_sha256"]:
        raise ValueError("checkpoint changed after the SIMVAL freeze")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    origin = payload.get("resume_identity", {})
    if train.old.canonical_digest(origin) != identity["source_resume_identity_sha256"]:
        raise ValueError("winner origin differs from C16 protocol")
    if payload.get("epoch") != epoch or payload.get("global_exposure") != epoch * 24000:
        raise ValueError("winner checkpoint budget differs")
    if epoch <= 20:
        model = train.old.load_decoupled_checkpoint(payload)
    else:
        train.validate_continuation(payload, identity)
        model = train.load_decoupled_score_checkpoint(payload)
        train.old.verify_receipt(model, payload["matcher_pretraining_receipt"], origin)
    receipt = dict(training_run=str(root), budget=28, selection=selection,
        freeze_path=str(path), freeze_sha256=train.old._sha256(path), checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=sha, epoch=epoch, seed=origin["seed"], model_config=asdict(model.config),
        architecture=origin["head_kind"], sampling=origin["sampling"],
        classifier_thresholds=selected["classifier_thresholds"], operating_points=selected["operating_points"],
        winner_record=selected, training_identity=origin, continuation_identity=identity,
        model_design=model.metadata(), classifier_only_pair_bce=True,
        coarse_is_untrained_diagnostic=True, local_and_fused_are_same_single_classifier=True,
        test_or_real_used_for_fit=False, ood_used_for_fit=False,
        endpoint_only=True, evaluation_adapter_sha256=train.old._sha256(__file__))
    return model.eval().requires_grad_(False), receipt


run = train.private_function(evaluator.run, load_frozen_model=load_frozen_model)


if __name__ == "__main__":
    run(evaluator.parser().parse_args())

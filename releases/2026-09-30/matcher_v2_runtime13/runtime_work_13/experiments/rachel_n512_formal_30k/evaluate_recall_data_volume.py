"""Held-out evaluation of the frozen high-recall data-volume winner."""
import argparse
import json
from functools import partial
from pathlib import Path

from . import evaluate_realism_checkpoint as fixed
from .train_recall_data_volume import SCHEMA, TOTAL
from .recall_operating_points import fit_operating_points
from .run_layout_decoder_experiment import classification
from .train_realism_data_ablation import save_json
from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


def load_winner(training_run, *, role="winner"):
    root = Path(training_run)
    if role not in ("winner", "six_epoch_snapshot"):
        raise ValueError("unknown checkpoint role")
    freeze = json.loads((root / ("train_val_freeze.json" if role == "winner" else "six_epoch_freeze.json")).read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if (protocol["schema_version"] != SCHEMA or protocol["status"] != "complete"
        or protocol["completed_exposures"] != TOTAL or freeze["status"] != "complete"
        or freeze["test_or_real_used_for_fit"] is not False):
        raise ValueError("requires completed recall-volume training")
    checkpoint_path = root / (role + ".pt")
    digest = sealed._sha256_file(checkpoint_path)
    if digest != freeze["checkpoint_sha256"]:
        raise ValueError("frozen winner changed")
    checkpoint = sealed._torch_load_checkpoint(checkpoint_path)
    if (checkpoint["epoch"] != freeze["selected_epoch"]
        or checkpoint["global_exposure"] != freeze["selected_global_exposure"]):
        raise ValueError("winner and selection event differ")
    model = load_rachel_checkpoint(checkpoint)
    architecture = fixed.check_fixed_architecture(model, checkpoint)
    thresholds = dict(freeze["classifier_thresholds"])
    thresholds["fused"] = freeze["primary_pair_threshold"]
    identity = dict(checkpoint_path=str(checkpoint_path), checkpoint_sha256=digest,
        seed=int(checkpoint["seed"]), precision="fp32", unique_count=protocol["unique_count"],
        model_metadata=architecture, original_fused_threshold=thresholds["fused"],
        branch_validation_thresholds=thresholds, model_selection_rule=freeze["selection_rule"],
        training_run=str(root), global_exposure=checkpoint["global_exposure"],
        operating_points=freeze["operating_points"])
    return model.eval().requires_grad_(False), identity, thresholds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run", required=True)
    p.add_argument("--checkpoint-role", choices=("winner", "six_epoch_snapshot"), default="winner")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True, choices=("test", "real"))
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prepared-cache", default=str(fixed.DEFAULT_PREPARED_CACHE))
    p.add_argument("--translation-gt-json")
    args = p.parse_args()
    fixed.run(args, winner_loader=partial(load_winner, role=args.checkpoint_role))
    # Thresholds are already frozen before fixed.run opens held-out inputs.
    freeze = json.loads((Path(args.training_run) / ("train_val_freeze.json" if args.checkpoint_role == "winner" else "six_epoch_freeze.json")).read_text())
    rows = [json.loads(line) for line in (Path(args.output) / "pair_results.jsonl").read_text().splitlines()]
    result = {}
    for name, threshold in freeze["operating_points"]["thresholds"].items():
        metrics = classification([r["label"] for r in rows], [r["classification"]["fused"] for r in rows], threshold)
        good = [r for r in rows if r["label"] and r["layouts"][fixed.DECODER_NAME]["valid"]
                and r["layouts"][fixed.DECODER_NAME]["translation_l2_px"] is not None
                and r["layouts"][fixed.DECODER_NAME]["translation_l2_px"] <= 10]
        metrics["accepted_good_layout_count"] = sum(r["classification"]["fused"] >= threshold for r in good)
        result[name] = metrics
    save_json(Path(args.output) / "recall_operating_points.json", dict(status="complete", split=args.split,
        methods=result, test_or_real_used_for_fit=False, threshold_source="selected cleanVAL3000",
        caveat="requested recall describes VAL only, not guaranteed REAL recall"))


if __name__ == "__main__":
    main()

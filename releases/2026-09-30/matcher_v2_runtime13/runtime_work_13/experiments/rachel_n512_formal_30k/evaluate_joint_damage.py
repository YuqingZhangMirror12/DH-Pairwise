"""Fixed TEST/REAL readout for complete random-init 6-epoch damage arms.

Only the training-receipt loader differs from evaluate_realism_checkpoint.
The original populations, score thresholds, decoder, and prediction-before-GT
ordering are retained; older 120k fine-tunes cannot masquerade as these arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.rachel_n512_formal_30k import evaluate_realism_checkpoint as fixed
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import check_fixed_architecture
from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


TOTAL_EXPOSURES, UPDATES, VAL_EVENTS = 144000, 9000, 6


def validate_freeze(freeze):
    expected = dict(status="complete", completed_global_exposures=TOTAL_EXPOSURES,
        completed_optimizer_updates=UPDATES, completed_validation_events=VAL_EVENTS,
        unique_count=24000, test_or_real_used_for_fit=False,
        original_validation_unchanged=True, initialization="random")
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise ValueError("incomplete or different joint-damage training protocol: " + key)
    validation = freeze["validation"]
    if ((validation.get("sample_count"), validation.get("positive_count"), validation.get("negative_count"))
            != (3000, 1500, 1500) or validation.get("pose_used_for_selection") is not False
            or validation.get("decision_coverage") != 1.0):
        raise ValueError("winner requires complete original clean VAL pair-only selection")
    thresholds = freeze["classifier_thresholds"]
    if (set(thresholds) != {"coarse", "local", "fused"}
            or thresholds != validation["thresholds"]
            or any(not np.isfinite(v) or not 0 <= v <= 1 for v in thresholds.values())):
        raise ValueError("three branch thresholds must be the unchanged VAL values")
    if not isinstance(freeze.get("initial_weights_sha256"), str) or len(freeze["initial_weights_sha256"]) != 64:
        raise ValueError("missing random initialization identity")
    return thresholds


def load_joint_training_winner(training_run):
    root = Path(training_run).resolve(strict=True)
    freeze_path, checkpoint_path = root / "train_val_freeze.json", root / "winner.pt"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    thresholds = validate_freeze(freeze)
    checkpoint = sealed._torch_load_checkpoint(checkpoint_path)
    for saved, selected in (("epoch", "selected_epoch"),
                            ("global_exposure", "selected_global_exposure"),
                            ("optimizer_updates", "selected_optimizer_updates"),
                            ("validation_event", "selected_validation_event"),
                            ("unique_count", "unique_count"),
                            ("initial_weights_sha256", "initial_weights_sha256")):
        if checkpoint.get(saved) != freeze.get(selected):
            raise ValueError("winner checkpoint differs from its freeze: " + saved)
    if checkpoint.get("initialization") != "random":
        raise ValueError("new schedule arms must start from random weights")
    if checkpoint.get("precision") != "fp32" or freeze.get("precision") != "fp32":
        raise ValueError("registered full-network comparison uses FP32")
    model = load_rachel_checkpoint(checkpoint)
    architecture = check_fixed_architecture(model, checkpoint)
    checkpoint_sha = sealed._sha256_file(checkpoint_path)
    if freeze.get("checkpoint_sha256", checkpoint_sha) != checkpoint_sha:
        raise ValueError("winner identity changed after freezing")
    identity = dict(training_run=str(root), training_freeze=str(freeze_path),
        training_freeze_sha256=sealed._sha256_file(freeze_path),
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
        checkpoint_epoch=checkpoint["epoch"], global_exposure=checkpoint["global_exposure"],
        optimizer_updates=checkpoint["optimizer_updates"], validation_event=checkpoint["validation_event"],
        unique_count=checkpoint["unique_count"], seed=int(checkpoint["seed"]), precision="fp32",
        model_metadata=architecture, loss_config=checkpoint["loss_config"],
        original_fused_threshold=float(thresholds["fused"]), branch_validation_thresholds=thresholds,
        model_selection_rule=freeze["selection_rule"], initialization="random",
        initial_weights_sha256=freeze["initial_weights_sha256"],
        completed_training_exposures=TOTAL_EXPOSURES, completed_training_updates=UPDATES,
        completed_training_validation_events=VAL_EVENTS)
    return model.eval().requires_grad_(False), identity, thresholds


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=("test", "real"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prepared-cache", default=str(fixed.DEFAULT_PREPARED_CACHE))
    p.add_argument("--translation-gt-json")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.split == "real" and not args.translation_gt_json:
        raise ValueError("REAL evaluation requires the existing translation GT path")
    fixed.run(args, winner_loader=load_joint_training_winner)

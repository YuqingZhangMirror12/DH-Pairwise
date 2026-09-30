"""Restricted coarse256/512 TEST/REAL evaluation, after BOTH arms freeze.

Reuse the unchanged FP32/batch8 fixed-Top2 prediction and target-blind REAL
evaluation pipeline. Neither a threshold nor a decoder is selected here.
--training-root must contain coarse256/ and coarse512/ with complete 120k
exposure, five-VAL-event receipts. The old coarse128 data-arm evaluator keeps
its original strict validator and default CLI behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k import evaluate_realism_checkpoint as evaluate
from experiments.rachel_n512_formal_30k.train_coarse_resolution_ablation import (
    SCHEMA, SELECTION_RULE, SOURCE_CHECKPOINT_SHA256, check_resolution_architecture, same_json,
)


def check_frozen_stage(training_root):
    """Small receipts only: fail closed before either external set is opened."""
    root = Path(training_root).resolve(strict=True)
    paths, reference = {}, None
    for size in (256, 512):
        arm = root / ("coarse%d" % size)
        protocol = json.loads((arm / "protocol.json").read_text())
        freeze = json.loads((arm / "train_val_freeze.json").read_text())
        if not (arm / "winner.pt").is_file():
            raise ValueError("both resolution winners must exist before external evaluation")
        initial = protocol.get("initial_model_metadata", {})
        effective = protocol.get("effective_model_metadata", {})
        source_config = initial.get("model_config", {})
        data = protocol.get("training_data", {})
        if (protocol.get("schema_version") != SCHEMA or protocol.get("arm") != "coarse%d" % size
                or protocol.get("status") != "complete" or freeze.get("status") != "complete"
                or protocol.get("initial_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
                or freeze.get("initial_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
                or protocol.get("seed") != 260909 or freeze.get("seed") != 260909
                or protocol.get("precision") != "fp32" or freeze.get("precision") != "fp32"
                or protocol.get("learning_rate") != 2e-5
                or protocol.get("initialization") != "warm-start"
                or protocol.get("full_network_trainable") is not True
                or protocol.get("seam_loss_enabled") is not False
                or protocol.get("seam_loss_weight") != 0.0
                or protocol.get("microbatch_size") != 4 or protocol.get("effective_batch_size") != 16
                or protocol.get("selection_rule") != SELECTION_RULE or freeze.get("selection_rule") != SELECTION_RULE
                or protocol.get("test_or_real_used_for_fit") is not False
                or freeze.get("test_or_real_used_for_fit") is not False
                or freeze.get("original_validation_unchanged") is not True
                or not same_json(protocol.get("source_loss_config"), protocol.get("effective_loss_config"))
                or data.get("kind") != "original_released_train" or data.get("sample_count") != 24000
                or data.get("unique_count") != 24000
                or any(receipt.get(key) != expected for receipt in (protocol, freeze)
                       for key, expected in (("completed_global_exposures", 120000),
                                             ("completed_optimizer_updates", 7500),
                                             ("completed_validation_events", 5),
                                             ("completed_dataset_epochs", 5)))):
            raise ValueError("both resolution arms require matching complete initialization/data/budget freezes")
        if (initial.get("model_kind") != "full" or initial.get("model_options") != {}
                or effective.get("model_kind") != "full" or effective.get("model_options") != {}
                or (source_config.get("canvas_size"), source_config.get("coarse_size"),
                    source_config.get("contour_cap"), source_config.get("patch_size")) != (800, 128, 512, 16)
                or tuple(source_config.get("window_sizes_px", ())) != (7.0, 16.0, 32.0, 64.0)
                or not same_json(effective.get("model_config"), dict(source_config, coarse_size=size))
                or not same_json(protocol.get("changed_config"),
                                 {"coarse_size": {"before": 128, "after": size}})):
            raise ValueError("frozen architectures must have coarse_size as their only configuration change")
        val = freeze.get("validation", {})
        if ((val.get("sample_count"), val.get("positive_count"), val.get("negative_count")) != (3000, 1500, 1500)
                or val.get("pose_used_for_selection") is not False or val.get("decision_coverage") != 1.0):
            raise ValueError("both winners require original balanced VAL and full-coverage row-F1 selection")
        identity = dict(initial=initial, training_data=data, loss=protocol["source_loss_config"],
                        control=protocol.get("control_training_run"), validation_root=protocol.get("validation_root"))
        if reference is not None and not same_json(reference, identity):
            raise ValueError("resolution arms disagree on the fixed control/data/loss/initial architecture")
        reference = identity
        paths[size] = arm
    return paths


def run(args):
    paths = check_frozen_stage(args.training_root)
    if args.coarse_size not in paths:
        raise ValueError("only the registered coarse256/coarse512 winners can be evaluated")
    args.training_run = str(paths[args.coarse_size])
    protocol = json.loads((paths[args.coarse_size] / "protocol.json").read_text())
    if Path(args.dataset).resolve() != Path(protocol["validation_root"]).resolve():
        raise ValueError("external TEST must use the unchanged original release recorded by training")

    def load_winner(training_run):
        model, identity, thresholds = evaluate.load_training_winner(
            training_run, architecture_validator=check_resolution_architecture)
        if model.config.coarse_size != args.coarse_size:
            raise ValueError("requested coarse size differs from the frozen checkpoint")
        identity.update(experiment_kind=SCHEMA, coarse_size=args.coarse_size,
                        changed_config={"coarse_size": {"before": 128, "after": args.coarse_size}},
                        both_resolution_arms_frozen=True,
                        all_resolution_training_roots={str(size): str(path) for size, path in paths.items()})
        return model, identity, thresholds

    evaluate.run(args, winner_loader=load_winner)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-root", required=True, help="parent of complete coarse256 and coarse512 arm directories")
    p.add_argument("--coarse-size", type=int, choices=(256, 512), required=True)
    p.add_argument("--split", choices=("test", "real"), required=True)
    p.add_argument("--dataset", required=True, help="unchanged original release TEST; never used for fitting")
    p.add_argument("--output", required=True, help="new result directory; no overwrite")
    p.add_argument("--prepared-cache", type=Path, default=evaluate.DEFAULT_PREPARED_CACHE)
    p.add_argument("--translation-gt-json", type=Path, default=evaluate.real.DEFAULT_TRANSLATION_GT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

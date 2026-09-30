"""Prepare, never launch, the S4 classifier-depth 1/2/4 comparison.

Depth1 is the already-completed S4 reference. Depth2/4 each import its exact
M12 source and train only C13..20 with a fresh head. Queue placement remains a
user decision: this module does not inspect processes, choose priority, copy
source, fit thresholds, or modify an existing configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k.insert_score_decoupled_queues import (
    PREFIX, TRAIN_MANIFEST, stage, write_new,
)
from experiments.rachel_n512_formal_30k.prepare_score_budget_queue import (
    DATASET, METADATA_CHECKPOINT, PYTHON, SPLITS,
)

SCHEMA = "rachel-attention-depth-preparation/1"
REFERENCE_ROOT = "/root/autodl-tmp/rachel_score_design_20260913_001"
DEPTHS = (2, 4)
SELECTIONS = ("fixed_epoch", "max_f1", "recall95")


def _absolute(value, name):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(name + " must be an absolute path without parent traversal")
    return path


def build_config(run_root, source, *, reference_root=REFERENCE_ROOT, python=PYTHON):
    """Pure construction; imported M12 identity is verified by the trainer."""
    root = _absolute(run_root, "run_root")
    source = _absolute(source, "source")
    reference = _absolute(reference_root, "reference_root")
    original_run = reference / "new_s345_20260914"
    if source != root / "source":
        raise ValueError("source must be the new isolated run_root/source snapshot")
    if root == reference or root == original_run or original_run in root.parents:
        raise ValueError("new depth outputs must be outside the existing S3/S4/S5 run")
    executable = str(_absolute(python, "python"))
    matcher = original_run / "s3_matrix/training/epoch_012.pt"
    depth1 = original_run / "s4_cross_attention"
    stages = []
    for depth in DEPTHS:
        name = "s4_cross_attention_depth%d" % depth
        arm = root / name
        training, smoke = arm / "training", arm / "smoke32"
        command = [executable, "-m", PREFIX + "train_score_decoupled",
            "--checkpoint", METADATA_CHECKPOINT, "--dataset", DATASET,
            "--train-materialized-manifest", TRAIN_MANIFEST,
            "--head-kind", "cross_attention", "--cross-attention-depth", str(depth),
            "--sampling", "original512", "--matcher-checkpoint", str(matcher),
            "--stop-after-epoch", "20", "--microbatch", "1",
            "--physical-microbatch", "16", "--effective-batch", "16",
            "--device", "cuda:0", "--workers", "4"]
        stages.append(stage(name + "_classifier_smoke32", command + [
            "--output", str(smoke), "--smoke", "32", "--smoke-phase", "classifier"],
            smoke, "smoke.json", ("smoke_complete",)))
        stages.append(stage(name + "_C13_C20", command + ["--output", str(training)],
            training, resume=True))
        for selection in SELECTIONS:
            for split in SPLITS:
                output = arm / "evaluation" / selection / split
                evaluate = [executable, "-m", PREFIX + "evaluate_score_decoupled",
                    "--training-run", str(training), "--selection", selection,
                    "--split", split, "--dataset", DATASET, "--output", str(output),
                    "--device", "cuda:0", "--batch-size", "1", "--workers", "4",
                    "--baseline-evaluation", str(depth1 / "evaluation" / selection / split)]
                if split == "real":
                    evaluate += ["--keep-ids", str(reference / "keep_ids.json")]
                stages.append(stage(name + "_" + selection + "_" + split,
                                    evaluate, output, "protocol.json"))
    return dict(root=str(root / "queue"), source=str(source), dependencies=[], stages=stages,
        preparation=dict(schema_version=SCHEMA, status="prepared_awaiting_queue_priority",
            launch_authorized=False, process_started=False, existing_queue_modified=False,
            deployment_performed=False, queue_insertion_performed=False,
            queue_priority="pending_user_choice: insert_next_or_after_existing_queue",
            dependencies_bound=False,
            source_snapshot_copied=False, source_snapshot_required=str(source),
            comparison_depths=[1, 2, 4], new_depths=list(DEPTHS), depth1_retrained=False,
            comparison_reference="depth1: existing completed S4; depth2/4: fresh C8 only",
            depth1_training_run=str(depth1 / "training"),
            depth1_classifier_freeze=str(depth1 / "training/classifier_freezes/freeze.json"),
            matcher_checkpoint=str(matcher), matcher_weights_loaded_by_preparer=False,
            matcher_import_contract="exact M12 receipt/base/data/loss; no inherited C head, optimizer or winners",
            changed_axis="classification cross-attention depth only; matcher and layout unchanged",
            model_options_field="cross_attention_depth; explicit checkpoint metadata for new depths",
            full24_manifest=TRAIN_MANIFEST, sampling="original512", precision="fp32",
            logical_microbatch=1, physical_microbatch=16, effective_batch=16,
            seed=260913, head_seed=260914, classifier_phase_seed=260915,
            inherited_matcher_epochs=12, inherited_matcher_exposures=288000,
            new_classifier_epochs=8, new_classifier_exposures=192000,
            total_budget_epochs=20, total_budget_exposures=480000,
            primary_selection="fixed_epoch20", auxiliary_epoch_range=[13, 20],
            selection_population="clean SIM VAL3000 only",
            all_thresholds_from_each_selected_checkpoint_VAL=True,
            held_out_read_by_preparer=False, no_GT_layout_used_for_selection=True))


def prepare(run_root, source, output, *, reference_root=REFERENCE_ROOT, python=PYTHON):
    """Write exactly one new config; no source copy, deployment or launch."""
    config = build_config(run_root, source, reference_root=reference_root, python=python)
    output = Path(output)
    write_new(output, config)  # Exclusive x mode; never overwrite an earlier plan.
    return dict(status=config["preparation"]["status"], output=str(output.resolve()),
                stage_count=len(config["stages"]), process_started=False,
                queue_priority=config["preparation"]["queue_priority"])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, help="new isolated remote experiment root")
    p.add_argument("--source", required=True, help="new run-root/source snapshot; not copied here")
    p.add_argument("--reference-root", default=REFERENCE_ROOT)
    p.add_argument("--python", default=PYTHON)
    p.add_argument("--output", required=True, help="new local or remote JSON path; exclusive create")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    print(json.dumps(prepare(args.run_root, args.source, args.output,
        reference_root=args.reference_root, python=args.python), ensure_ascii=False))


if __name__ == "__main__":
    main()

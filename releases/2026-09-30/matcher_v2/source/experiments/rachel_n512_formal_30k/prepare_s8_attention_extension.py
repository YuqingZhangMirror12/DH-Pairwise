"""Prepare S8: the authorized S5 step3-cap2048 + S6 depth2 extension.

This is a NEW C8-only arm, not a resume/overwrite of the completed MatrixCNN
run. Its exact S5 M12 base and original TRAIN/VAL density manifests are reused.
The preparer never contacts a GPU, copies source, or modifies/launches queues.
Depth2 is declared before looking at new REAL/OOD results; depth4 is not added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.rachel_n512_formal_30k.insert_score_decoupled_queues import (
    PREFIX, TRAIN_MANIFEST, stage, write_new,
)
from experiments.rachel_n512_formal_30k.prepare_attention_depth_ablation import (
    REFERENCE_ROOT, SELECTIONS, _absolute,
)
from experiments.rachel_n512_formal_30k.prepare_score_budget_queue import (
    DATASET, METADATA_CHECKPOINT, PYTHON, SPLITS,
)

SCHEMA = "rachel-s8-step2048-attention-extension/1"
ARM = "s8_step2048_attention_depth2"


def build_config(run_root, source, *, reference_root=REFERENCE_ROOT, python=PYTHON,
                 physical_microbatch=4):
    root = _absolute(run_root, "run_root")
    source = _absolute(source, "source")
    reference = _absolute(reference_root, "reference_root")
    original = reference / "new_s345_20260914"
    if source != root / "source":
        raise ValueError("source must be the isolated new run_root/source snapshot")
    if root == reference or root == original or original in root.parents:
        raise ValueError("new outputs must not overwrite the completed S5 experiment")
    if type(physical_microbatch) is not int or physical_microbatch != 4:
        raise ValueError("this S8 capacity gate registers physical_microbatch4 only")
    executable = str(_absolute(python, "python"))
    baseline = original / "s5_step3_cap2048"
    matcher = baseline / "training/epoch_012.pt"
    arm = root / ARM
    training, smoke = arm / "training", arm / "classifier_smoke32"
    train_manifest = original / "step_data/train/manifest.json"
    val_manifest = original / "step_data/val/manifest.json"
    command = [executable, "-m", PREFIX + "train_score_decoupled",
        "--checkpoint", METADATA_CHECKPOINT, "--dataset", DATASET,
        "--train-materialized-manifest", TRAIN_MANIFEST,
        "--head-kind", "cross_attention", "--cross-attention-depth", "2",
        "--sampling", "step3", "--density-train-manifest", str(train_manifest),
        "--clean-val-manifest", str(val_manifest), "--matcher-checkpoint", str(matcher),
        "--stop-after-epoch", "20", "--microbatch", "1",
        "--physical-microbatch", str(physical_microbatch), "--effective-batch", "16",
        "--device", "cuda:0", "--workers", "4"]
    capacity = arm / "capacity_gate"
    capacity_selection = original / "s5_capacity_v3_20260914/cpu_plan/selection.json"
    capacity_command = [executable, "-m", PREFIX + "benchmark_s8_attention_capacity",
        "--matcher-checkpoint", str(matcher), "--train-manifest", str(train_manifest),
        "--selection-file", str(capacity_selection), "--output", str(capacity)]
    stages = [stage(ARM + "_capacity_gate", capacity_command, capacity, "result.json"),
        stage(ARM + "_classifier_smoke32", command + ["--output", str(smoke),
        "--smoke", "32", "--smoke-phase", "classifier"], smoke, "smoke.json", ("smoke_complete",)),
        stage(ARM + "_C13_C20", command + ["--output", str(training)], training, resume=True)]
    for selection in SELECTIONS:
        for split in SPLITS:
            output = arm / "evaluation" / selection / split
            evaluate = [executable, "-m", PREFIX + "evaluate_score_decoupled",
                "--training-run", str(training), "--selection", selection,
                "--split", split, "--dataset", DATASET, "--output", str(output),
                "--device", "cuda:0", "--batch-size", "1", "--workers", "4",
                "--baseline-evaluation", str(baseline / "evaluation" / selection / split)]
            if split == "real":
                evaluate += ["--keep-ids", str(reference / "keep_ids.json")]
            stages.append(stage(ARM + "_" + selection + "_" + split,
                                evaluate, output, "protocol.json"))
    return dict(root=str(root / "queue"), source=str(source), dependencies=[], stages=stages,
        preparation=dict(schema_version=SCHEMA, experiment_name="S8", status="prepared_not_launched",
            process_started=False, source_snapshot_copied=False,
            deployment_performed=False, existing_outputs_modified=False,
            queue_priority="bind behind S6 on its exclusive GPU; S7 occupies the other GPU",
            dependency_binding_required=True, device_mapping="CUDA_VISIBLE_DEVICES chosen by launcher; cuda:0 is process-local",
            changed_axis="replace S5 MatrixCNN-v3 classifier with S6 depth2 attention; matcher and layout fixed",
            sampling="step3", contour_cap=2048, physical_step_px=3.,
            depth=2, depth_choice="predeclared before new REAL/OOD results", depth4_added=False,
            baseline_training_run=str(baseline / "training"), matcher_checkpoint=str(matcher),
            matcher_import_contract="same original M12 data/cap/loss/random-base receipt; no old head, optimizer or C winners",
            matcher_retrained=False, old_classifier_resumed=False,
            original_full24_manifest=TRAIN_MANIFEST,
            density_train_manifest=str(train_manifest), clean_val_manifest=str(val_manifest),
            seed=260913, head_seed=260914, classifier_phase_seed=260915,
            precision="fp32", logical_microbatch=1, physical_microbatch=physical_microbatch,
            effective_batch=16, gradient_accumulation=16 // physical_microbatch,
            batching_gate="first queue stage: real TRAIN extreme16, depth2/cap2048 physical4, two discarded effective16 groups, memory<80%, frozen base unchanged",
            capacity_selection=str(capacity_selection), capacity_search=False,
            physical4_reference="previous S5 cap2048 physical4; attention capacity still must be checked",
            inherited_matcher_epochs=12, inherited_matcher_exposures=288000,
            new_classifier_epochs=8, new_classifier_exposures=192000,
            total_budget_epochs=20, total_budget_exposures=480000,
            primary_selection="fixed_epoch20", auxiliary_selection_epochs=list(range(13, 21)),
            selection_population="SIM VAL3000 only", final_evaluation_count=9,
            held_out_used_for_selection=False, no_GT_layout_used_for_selection=True,
            ood_has_layout_GT=False))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--reference-root", default=REFERENCE_ROOT)
    p.add_argument("--python", default=PYTHON)
    p.add_argument("--physical-microbatch", type=int, default=4)
    p.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = build_config(args.run_root, args.source, reference_root=args.reference_root,
                          python=args.python, physical_microbatch=args.physical_microbatch)
    write_new(Path(args.output), config)


if __name__ == "__main__":
    main()

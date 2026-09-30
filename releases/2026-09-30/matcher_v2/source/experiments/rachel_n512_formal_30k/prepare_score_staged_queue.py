"""Prepare the S3 M12+C8 versus joint20 queue; never launches a process.

The caller chooses candidate_pair or candidate_dual before held-out inspection.
Reuse the existing joint trajectory, but select its S3 checkpoints anew from
saved clean VAL13..20. Its ordinary [5,20] winner is never imported. The main
comparison is the fixed epoch20 anchor; all outputs use separate directories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k.prepare_score_budget_queue import (
    DATASET, MATERIALIZED, METADATA_CHECKPOINT, PYTHON, SPLITS,
)
from experiments.rachel_n512_formal_30k.freeze_score_staged_comparison import (
    SELECTIONS, publish_s3_freeze, validate_s3_freeze, write_new_json,
)


def locations(root, architecture):
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("absolute experiment root required")
    if architecture not in ("candidate_pair", "candidate_dual"):
        raise ValueError("S3 requires a registered candidate architecture")
    return dict(root=root, joint=root / architecture / "training",
        staged=root / "s3" / architecture / "training",
        queue=root / "queues" / ("s3_" + architecture),
        smoke=root / "smokes" / ("s3_" + architecture + "_32_before_queue"),
        comparison=root / "s3" / architecture / "paired_freeze.json")


def evaluation_path(paths, schedule, selection, split):
    return paths[schedule].parent / "evaluation" / "s3_epoch020" / selection / split


def build_config(root, architecture, *, python=PYTHON):
    """One changed design axis: registered training phases, not the matcher."""
    paths = locations(root, architecture)
    stages = []
    train_args = [python, "-m", "experiments.rachel_n512_formal_30k.train_score_staged",
        "--checkpoint", METADATA_CHECKPOINT, "--dataset", DATASET,
        "--train-materialized-manifest", MATERIALIZED, "--architecture", architecture,
        "--workers", "4", "--stop-after-epoch", "20"]
    stages.append(dict(name="s3_matcher_smoke32", command=train_args + [
        "--output", str(paths["smoke"]), "--smoke", "32"], marker=str(paths["smoke"]),
        completion_path=str(paths["smoke"] / "smoke.json"),
        completion_statuses=["smoke_complete"]))
    stages.append(dict(name="s3_m12_c8_train20", command=train_args + [
        "--output", str(paths["staged"])], marker=str(paths["staged"]),
        completion_path=str(paths["staged"] / "status.json"),
        completion_statuses=["complete"], resume_arguments=["--resume"]))
    stages.append(dict(name="freeze_equal_budget_s3_comparison", command=[python,
        "-m", "experiments.rachel_n512_formal_30k.freeze_score_staged_comparison",
        "--joint-run", str(paths["joint"]), "--staged-run", str(paths["staged"]),
        "--output", str(paths["comparison"])], marker=str(paths["comparison"]),
        completion_path=str(paths["comparison"]), completion_statuses=["paired_frozen"]))
    for schedule in ("joint", "staged"):
        for selection in SELECTIONS:
            for split in SPLITS:
                output = evaluation_path(paths, schedule, selection, split)
                command = [python, "-m", "experiments.rachel_n512_formal_30k.evaluate_score_staged",
                    "--training-run", str(paths[schedule]), "--selection", selection,
                    "--split", split, "--output", str(output), "--batch-size", "4", "--workers", "4"]
                if split == "real":
                    command += ["--keep-ids", str(paths["root"] / "keep_ids.json")]
                if schedule == "staged":
                    command += ["--baseline-evaluation", str(evaluation_path(paths, "joint", selection, split))]
                stages.append(dict(name="s3_%s_%s_%s" % (schedule, selection, split),
                    command=command, marker=str(output), completion_path=str(output / "protocol.json"),
                    completion_statuses=["complete"]))
    return dict(root=str(paths["queue"]), source=str(paths["root"] / "source"),
                dependencies=[], stages=stages)


def prepare(root, architecture):
    """Prepare only when the existing joint20 data/VAL checkpoints are real."""
    root = Path(root).resolve(strict=True)
    paths = locations(root, architecture)
    # Do not prepare over an earlier attempt or change its experiment identity.
    if paths["queue"].exists() or paths["staged"].exists():
        raise FileExistsError("S3 queue or staged training already exists; inspect/resume that run")
    kept = json.loads((root / "keep_ids.json").read_text())["kept_positive_pair_ids"]
    if len(kept) != 295 or len(set(kept)) != 295:
        raise ValueError("requires the unchanged295-positive review export")
    joint_freeze = validate_s3_freeze(publish_s3_freeze(paths["joint"], schedule="joint"))
    if joint_freeze["architecture"] != architecture:
        raise ValueError("joint trajectory architecture differs from the requested S3 comparison")
    config = build_config(root, architecture)
    paths["queue"].mkdir(parents=True, exist_ok=False)
    write_new_json(paths["queue"] / "config.json", config)
    write_new_json(paths["queue"] / "preparation.json", dict(status="prepared_not_launched",
        architecture=architecture, primary_selection="fixed_epoch20", auxiliary_epoch_range=[13, 20],
        joint_freeze=str(paths["joint"] / "s3_freezes" / "freeze.json"),
        common_training_contract_sha256=joint_freeze["common_training_contract_sha256"],
        total_exposures_each=480000, joint_retrained=False,
        architecture_choice="caller-specified; no TEST/REAL/OOD metrics read by this preparer",
        held_out_used_for_fit=False))
    return dict(status="prepared_not_launched", config=str(paths["queue"] / "config.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--architecture", required=True, choices=("candidate_pair", "candidate_dual"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.architecture)))


if __name__ == "__main__":
    main()

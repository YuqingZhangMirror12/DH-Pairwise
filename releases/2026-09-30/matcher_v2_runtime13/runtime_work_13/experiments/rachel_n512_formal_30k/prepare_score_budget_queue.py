"""Prepare, but never launch, the next equal-budget S0/S1/S2 continuation.

Run on the experiment server after the preceding three-arm budget and all
held-out evaluations have completed. Decisions here depend only on completion
and protocol identity, never on TEST/REAL/OOD metric values. Optimizer/RNG and
the original fifty-epoch learning-rate timeline are resumed, not restarted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BUDGETS = (5, 10, 20, 30, 50)
ARMS = ("original", "candidate_pair", "candidate_dual")
SELECTIONS = ("max_f1", "recall95")
SPLITS = ("test", "real", "ood")
PYTHON = "/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python"
DATASET = "/root/autodl-tmp/dataset_rachel_pairwise_n512_v1"
MATERIALIZED = "/root/autodl-tmp/rachel_recall_benchmarks_20260911_001/data/train_e1_24k.json"
METADATA_CHECKPOINT = "/root/autodl-tmp/rachel_recall_benchmarks_20260911_001/full_e1_24k/training/winner.pt"


def next_budget(previous):
    if previous not in BUDGETS[:-1]:
        raise ValueError("previous must be5,10,20,30;50 has no continuation")
    return BUDGETS[BUDGETS.index(previous) + 1]


def evaluation_path(root, arm, budget, selection, split):
    return root / arm / "evaluation" / ("budget_%03d" % budget) / selection / split


def build_config(root, previous, *, python=PYTHON):
    """Pure command construction; one changed variable is the training budget."""
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("absolute experiment root required")
    budget = next_budget(previous)
    stages = []
    for arm in ARMS:
        training = root / arm / "training"
        name = "%s_joint_%03d" % (arm, budget)
        command = [python, "-m", "experiments.rachel_n512_formal_30k.train_score_design",
            "--checkpoint", METADATA_CHECKPOINT, "--dataset", DATASET,
            "--train-materialized-manifest", MATERIALIZED, "--output", str(training),
            "--architecture", arm, "--training-mode", "joint", "--workers", "4",
            "--max-epochs", "50", "--stop-after-epoch", str(budget), "--resume"]
        stages.append(dict(name=name, command=command, marker=str(training),
            completion_path=str(training / "status.json"),
            completion_statuses=["complete" if budget == 50 else "budget_complete"],
            resume_arguments=["--resume"]))
        for selection in SELECTIONS:
            for split in SPLITS:
                output = evaluation_path(root, arm, budget, selection, split)
                command = [python, "-m", "experiments.rachel_n512_formal_30k.evaluate_score_design",
                    "--training-run", str(training), "--budget", str(budget),
                    "--selection", selection, "--split", split, "--output", str(output),
                    "--batch-size", "4", "--workers", "4"]
                if split == "real":
                    command += ["--keep-ids", str(root / "keep_ids.json")]
                if arm != "original":
                    control = "original" if arm == "candidate_pair" else "candidate_pair"
                    command += ["--baseline-evaluation",
                        str(evaluation_path(root, control, budget, selection, split))]
                stages.append(dict(name="%s_%03d_%s_%s" % (arm, budget, selection, split),
                    command=command, marker=str(output), completion_path=str(output / "protocol.json"),
                    completion_statuses=["complete"]))
    return dict(root=str(root / "queues" / ("budget%03d" % budget)), source=str(root / "source"),
                dependencies=[], stages=stages)


def read(path):
    return json.loads(Path(path).read_text())


def verify_previous(root, previous):
    """Require exact complete populations; do not use their scores to decide."""
    root = Path(root)
    next_budget(previous)
    records = []
    for arm in ARMS:
        training = root / arm / "training"
        state = read(training / "status.json")
        if (state.get("status") != "budget_complete" or state.get("epoch") != previous
                or state.get("global_exposure") != previous * 24000
                or state.get("optimizer_updates") != previous * 1500):
            raise ValueError(arm + ": previous exact training budget is not complete")
        freeze_path = training / "budget_freezes" / ("%03d" % previous) / "freeze.json"
        freeze = read(freeze_path)
        if (freeze.get("schema_version") != "rachel-score-design-training/1"
                or freeze.get("status") != "frozen_at_budget" or freeze.get("budget_epochs") != previous
                or freeze.get("held_out_used_for_fit") is not False
                or set(freeze.get("winners", {})) != set(SELECTIONS)):
            raise ValueError(arm + ": previous VAL freeze is incomplete")
        for selection in SELECTIONS:
            winner = freeze["winners"][selection]
            for split in SPLITS:
                output = evaluation_path(root, arm, previous, selection, split)
                protocol = read(output / "protocol.json")
                model = protocol.get("model", {})
                if (protocol.get("schema_version") != "rachel-score-design-evaluation/1"
                        or protocol.get("status") != "complete" or protocol.get("split") != split
                        or protocol.get("sample_count") != {"test": 3000, "real": 1016, "ood": 301}[split]
                        or model.get("budget") != previous or model.get("selection") != selection
                        or model.get("architecture") != arm
                        or model.get("checkpoint_sha256") != winner.get("checkpoint_sha256")):
                    raise ValueError(str(output) + ": previous frozen evaluation incomplete or different")
                # Presence and completion of both files, not a metric threshold,
                # establishes that the named evaluation actually finished.
                if read(output / "summary.json").get("status") != "complete":
                    raise ValueError(str(output) + ": missing complete summary")
                if not (output / "pair_results.jsonl").is_file():
                    raise ValueError(str(output) + ": missing per-pair results")
        records.append(dict(arm=arm, previous_budget=previous, freeze=str(freeze_path),
                            validation_winner_sha256={k: v["checkpoint_sha256"] for k, v in freeze["winners"].items()}))
    return records


def prepare(root, previous):
    root = Path(root).resolve(strict=True)
    config = build_config(root, previous)
    records = verify_previous(root, previous)
    destination = Path(config["root"])
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in (("config.json", config), ("preparation.json", dict(status="prepared_not_launched",
            previous_budget=previous, next_budget=next_budget(previous),
            prior_completions=records, held_out_metrics_used_to_choose_next_action=False))):
        with (destination / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(status="prepared_not_launched", config=str(destination / "config.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--previous-budget", required=True, type=int, choices=BUDGETS[:-1])
    arguments = parser.parse_args()
    prepare(arguments.root, arguments.previous_budget)

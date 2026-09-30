"""Prepare a finite first5 -> budget10 -> budget20 continuation, never launch it.

The existing first5 queue must finish before any action. Each next-budget
preparer checks all three completed arms and their six frozen evaluations.
Nested queues execute as owned children, not detached grandchildren. This
chain stops at20 so the registered staged comparison can precede input axes.
No checkpoint, architecture or stopping decision is made from held-out scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k.prepare_score_budget_queue import PYTHON


def build_config(root, dependency_pid, *, python=PYTHON):
    root = Path(root)
    if not root.is_absolute() or type(dependency_pid) is not int or dependency_pid <= 0:
        raise ValueError("absolute experiment root and positive first5 queue PID required")
    first = root / "queues" / "candidates5"
    stages = []
    for previous, budget in ((5, 10), (10, 20)):
        queue = root / "queues" / ("budget%03d" % budget)
        stages.append(dict(name="prepare_budget_%03d" % budget,
            command=[python, "-m", "experiments.rachel_n512_formal_30k.prepare_score_budget_queue",
                     "--root", str(root), "--previous-budget", str(previous)],
            marker="prepare_score_budget_queue", completion_path=str(queue / "preparation.json"),
            completion_statuses=["prepared_not_launched"]))
        stages.append(dict(name="run_budget_%03d" % budget,
            command=[python, "-m", "experiments.rachel_n512_formal_30k.run_recall_benchmark_queue",
                     "--config", str(queue / "config.json")],
            marker=str(queue / "config.json"), completion_path=str(queue / "queue_state.json"),
            completion_statuses=["complete"], resume_arguments=["--resume"]))
    return dict(root=str(root / "queues" / "continuation_to020"), source=str(root / "source"),
        dependencies=[dict(name="three_arm_first5_with_frozen_evaluations", pid=dependency_pid,
            marker=str(first / "config.json"), status_path=str(first / "queue_state.json"))],
        stages=stages)


def prepare(root, dependency_pid):
    root = Path(root).resolve(strict=True)
    config = build_config(root, dependency_pid)
    first = root / "queues/candidates5"
    state = json.loads((first / "queue_state.json").read_text())
    if state.get("pid") != dependency_pid or state.get("status") not in (
            "starting", "waiting_for_dependency", "running", "complete"):
        raise ValueError("first5 dependency PID/status differs from the actual queue")
    proc = Path("/proc") / str(dependency_pid) / "cmdline"
    live = proc.exists() and str(first / "config.json").encode() in proc.read_bytes()
    if state["status"] != "complete" and not live:
        raise ValueError("first5 is unfinished and its actual queue process is stopped")
    if not (root / "source").is_dir():
        raise ValueError("task-owned source directory is absent")
    for budget in (10, 20):
        if (root / "queues" / ("budget%03d" % budget)).exists():
            raise ValueError("next-budget output already exists; inspect it before preparing a chain")
    destination = Path(config["root"])
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in (("config.json", config), ("preparation.json", dict(
            status="prepared_not_launched", first5_dependency_pid=dependency_pid,
            sequential_budgets=[10, 20], stops_before_staged_and_input_experiments=True,
            all_three_arms_each_budget=True, held_out_metrics_used_to_choose_next_action=False))):
        with (destination / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    result = dict(status="prepared_not_launched", config=str(destination / "config.json"))
    print(json.dumps(result))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--first5-queue-pid", required=True, type=int)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    prepare(args.root, args.first5_queue_pid)

"""Prepare a finite continuation020 -> candidate_dual S3 -> joint030 ->050 tail.

Preparation never launches a child, prepares a future S3 freeze, or selects
an architecture/budget from held-out scores. The current continuation_to020
queue is an explicit live-process dependency. Once it completes, the original
strict S3/budget preparers run unchanged, then their queues run as owned child
processes. The main three-arm optimizer/LR/RNG trajectories remain continuous.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k.prepare_score_budget_queue import PYTHON, ARMS

ARCHITECTURE = "candidate_dual"  # Fixed experimental order, never metric-selected.
ACTIVE_STATUSES = ("starting", "resuming", "waiting_for_dependency", "running", "complete")


def build_config(root, continuation_queue_pid, *, python=PYTHON):
    root = Path(root)
    if not root.is_absolute() or type(continuation_queue_pid) is not int or continuation_queue_pid <= 0:
        raise ValueError("absolute experiment root and positive continuation queue PID required")
    dependency = root / "queues/continuation_to020"
    s3 = root / "queues/s3_candidate_dual"
    stages = [dict(name="prepare_candidate_dual_s3", command=[python, "-m",
        "experiments.rachel_n512_formal_30k.prepare_score_staged_queue", "--root", str(root),
        "--architecture", ARCHITECTURE], marker="prepare_score_staged_queue",
        completion_path=str(s3 / "preparation.json"), completion_statuses=["prepared_not_launched"]),
        dict(name="run_candidate_dual_s3", command=[python, "-m",
        "experiments.rachel_n512_formal_30k.run_recall_benchmark_queue", "--config", str(s3 / "config.json")],
        marker=str(s3 / "config.json"), completion_path=str(s3 / "queue_state.json"),
        completion_statuses=["complete"], resume_arguments=["--resume"])]
    for previous, budget in ((20, 30), (30, 50)):
        queue = root / "queues" / ("budget%03d" % budget)
        stages.append(dict(name="prepare_budget_%03d" % budget, command=[python, "-m",
            "experiments.rachel_n512_formal_30k.prepare_score_budget_queue", "--root", str(root),
            "--previous-budget", str(previous)], marker="prepare_score_budget_queue",
            completion_path=str(queue / "preparation.json"), completion_statuses=["prepared_not_launched"]))
        stages.append(dict(name="run_budget_%03d" % budget, command=[python, "-m",
            "experiments.rachel_n512_formal_30k.run_recall_benchmark_queue", "--config", str(queue / "config.json")],
            marker=str(queue / "config.json"), completion_path=str(queue / "queue_state.json"),
            completion_statuses=["complete"], resume_arguments=["--resume"]))
    return dict(root=str(root / "queues/after020_s3_to050"), source=str(root / "source"),
        dependencies=[dict(name="continuation_to020_with_all_frozen_evaluations", pid=continuation_queue_pid,
            marker=str(dependency / "config.json"), status_path=str(dependency / "queue_state.json"))], stages=stages)


def dependency_live(pid, config_path):
    """Use the existing /proc PID+command marker semantics, not a guessed PID."""
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return b"run_recall_benchmark_queue" in command and str(config_path).encode() in command


def verify_dependency(root, pid):
    directory = root / "queues/continuation_to020"
    config_path, state_path = directory / "config.json", directory / "queue_state.json"
    config, state = json.loads(config_path.read_text()), json.loads(state_path.read_text())
    if (config.get("root") != str(directory) or config.get("source") != str(root / "source")
            or state.get("config") != config or state.get("pid") != pid
            or state.get("status") not in ACTIVE_STATUSES):
        raise ValueError("continuation dependency config/PID/status differs from the actual owned queue")
    # The producer is specifically the finite10->20 chain, not another queue
    # with a copied name or a process belonging to a training child.
    if [s.get("name") for s in config.get("stages", [])] != [
            "prepare_budget_010", "run_budget_010", "prepare_budget_020", "run_budget_020"]:
        raise ValueError("dependency is not the registered continuation_to020 chain")
    live = dependency_live(pid, config_path)
    if state["status"] != "complete" and not live:
        raise ValueError("continuation_to020 is unfinished and its actual queue process is stopped")
    return dict(queue=str(directory), config=str(config_path), pid=pid,
        observed_status=state["status"], observed_live=live,
        completion_gate="owned queue complete and its process stopped, checked by existing queue runner")


def future_outputs(root):
    """Only paths this tail would create; existing main training dirs are reused."""
    paths = [root / "queues/after020_s3_to050", root / "queues/s3_candidate_dual",
        root / "s3/candidate_dual/training", root / "s3/candidate_dual/paired_freeze.json",
        root / "s3/candidate_dual/evaluation/s3_epoch020",
        root / "smokes/s3_candidate_dual_32_before_queue",
        root / "candidate_dual/training/s3_freezes/freeze.json",
        root / "candidate_dual/evaluation/s3_epoch020"]
    for budget in (30, 50):
        paths.append(root / "queues" / ("budget%03d" % budget))
        for arm in ARMS:
            paths += [root / arm / "evaluation" / ("budget_%03d" % budget),
                root / arm / "training/budget_freezes" / ("%03d" % budget),
                root / arm / "training" / ("epoch_%03d.pt" % budget)]
    return paths


def prepare(root, continuation_queue_pid):
    root = Path(root).resolve(strict=True)
    config = build_config(root, continuation_queue_pid)
    dependency = verify_dependency(root, continuation_queue_pid)
    if not (root / "source").is_dir():
        raise ValueError("task-owned source directory is absent")
    occupied = [str(path) for path in future_outputs(root) if path.exists()]
    if occupied:
        raise FileExistsError("future tail output already exists; inspect/resume it instead: " + occupied[0])
    destination = Path(config["root"])
    destination.mkdir(parents=True, exist_ok=False)
    preparation = dict(status="prepared_not_launched", continuation_dependency=dependency,
        continuation_queue_pid=continuation_queue_pid, sequential_steps=[s["name"] for s in config["stages"]],
        s3_architecture=ARCHITECTURE, s3_primary="fixed_epoch20 joint20 versus M12+C8",
        s3_auxiliary_window=[13, 20], subsequent_three_arm_budgets=[30, 50],
        main_arms=list(ARMS), initialization_or_optimizer_reset=False,
        original_lr_and_rng_trajectory_preserved=True, original_preparation_gates_unchanged=True,
        future_freezes_prepared=False, children="owned synchronous queues; scoped --resume only",
        architecture_choice="fixed candidate_dual after added P/R supervision; not chosen by held-out scores",
        held_out_metrics_used_to_choose_next_action=False, launches_processes=False)
    for name, value in (("config.json", config), ("preparation.json", preparation)):
        with (destination / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    result = dict(status="prepared_not_launched", config=str(destination / "config.json"))
    print(json.dumps(result))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--continuation-queue-pid", type=int, required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    prepare(args.root, args.continuation_queue_pid)

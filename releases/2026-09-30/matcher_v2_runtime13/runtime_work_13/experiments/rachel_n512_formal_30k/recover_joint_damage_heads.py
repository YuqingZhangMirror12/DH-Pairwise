"""Resume only the failed, pre-extraction heads tail; preserve completed arms."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil

from . import run_joint_damage_pipeline as queue


def require_stopped(pid):
    if pid is None:
        return
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return
    raise RuntimeError("refusing recovery while recorded process exists: " + str(pid))


def validate_startup_failure(root, state):
    if state.get("status") != "failed" or state.get("active_stage") != "fit_evaluate_separate_heads":
        raise ValueError("only the recorded failed heads stage may be recovered")
    require_stopped(state.get("pid"))
    stages = state["stages"]
    index = next(i for i, stage in enumerate(stages) if stage["name"] == "fit_evaluate_separate_heads")
    if any(stage.get("status") != "complete" for stage in stages[:index]):
        raise ValueError("all prior model training/evaluations must already be complete")
    if stages[index].get("status") != "failed":
        raise ValueError("heads stage must have an observed failure")
    if [(s["name"], s["status"]) for s in stages[index + 1:]] != [("summarize_fixed_straight_strata", "queued")]:
        raise ValueError("unexpected remaining queue; recovery must not overwrite completed work")
    require_stopped(stages[index].get("pid"))
    heads = root / "heads"
    if {str(p.relative_to(heads)) for p in heads.rglob("*") if p.is_file()} != {"status.json"}:
        raise ValueError("heads progressed beyond initialization; use a different recovery procedure")
    status = queue.read(heads / "status.json")
    if status.get("status") != "failed" or "'str' object has no attribute 'open'" not in status.get("error", ""):
        raise ValueError("not the diagnosed pre-extraction path type failure")
    if not Path(stages[index]["log"]).is_file():
        raise ValueError("original failure log is missing")
    return index


def run(root):
    root = Path(root).resolve()
    state_path = root / "pipeline_state.json"
    with (root / ".heads_recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = queue.read(state_path)
        index = validate_startup_failure(root, state)
        args = argparse.Namespace(**state["arguments"])
        if Path(args.root).resolve() != root:
            raise ValueError("stored queue root differs from requested root")
        archive = root / "recovery" / "heads_path_type_startup_failure"
        archive.mkdir(parents=True, exist_ok=False)
        shutil.copy2(state_path, archive / "pipeline_state.json")
        old_log = Path(state["stages"][index]["log"])
        old_log.rename(archive / old_log.name)
        (root / "heads").rename(archive / "heads")
        state.setdefault("recoveries", []).append(dict(
            at=queue.now(), reason="normalize CLI train-manifest to Path before hashing",
            archive=str(archive), previous_parent_pid=state.get("pid"),
            previous_child_pid=state["stages"][index].get("pid"),
            completed_full_network_arms_preserved=True, prior_head_training_updates=0))
        state.update(pid=os.getpid(), status="running", error=None)
        queue.save(state_path, state)
        try:
            for stage in state["stages"][index:]:
                queue.run_stage(args, state, stage, state_path)
            queue.summarize(args)
            state.update(status="complete", active_stage=None, completed_at=queue.now())
        except Exception as error:
            state.update(status="failed", error=repr(error), failed_at=queue.now())
            raise
        finally:
            queue.save(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    run(parser.parse_args().root)

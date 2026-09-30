"""Resume the missing-module summary tail without retraining or inference."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil

from . import run_joint_damage_pipeline as queue
from .recover_joint_damage_heads import require_stopped

STAGE = "summarize_fixed_straight_strata"
MODULE = "experiments.rachel_n512_formal_30k.summarize_real_straight_strata"


def run(root):
    root = Path(root).resolve()
    path = root / "pipeline_state.json"
    with (root / ".summary_recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = queue.read(path)
        if state.get("status") != "failed" or state.get("active_stage") != STAGE:
            raise ValueError("only the failed summary tail may be recovered")
        require_stopped(state.get("pid"))
        stage = state["stages"][-1]
        if (stage["name"], stage["status"], stage["module"]) != (STAGE, "failed", MODULE):
            raise ValueError("unexpected final stage")
        require_stopped(stage.get("pid"))
        if any(s["status"] != "complete" for s in state["stages"][:-1]):
            raise ValueError("all training and evaluation must already be complete")
        old_log = Path(stage["log"])
        if "No module named " + MODULE not in old_log.read_text():
            raise ValueError("not the observed missing summary module failure")
        if (root / "straight_strata").exists() or (root / "summary.json").exists():
            raise ValueError("summary outputs already exist; do not overwrite")
        args = argparse.Namespace(**state["arguments"])
        if Path(args.root).resolve() != root:
            raise ValueError("stored root mismatch")
        if not (Path(args.source) / Path(*MODULE.split("."))).with_suffix(".py").is_file():
            raise ValueError("deploy the missing summary module first")
        archive = root / "recovery" / "missing_summary_module"
        archive.mkdir(parents=True, exist_ok=False)
        shutil.copy2(path, archive / path.name)
        old_log.rename(archive / old_log.name)
        state.setdefault("recoveries", []).append(dict(at=queue.now(),
            reason="deploy missing CPU straight-strata summary module",
            archive=str(archive), previous_parent_pid=state.get("pid"),
            previous_child_pid=stage.get("pid"), all_training_and_evaluation_preserved=True))
        state.update(pid=os.getpid(), status="running", error=None)
        queue.save(path, state)
        try:
            queue.run_stage(args, state, stage, path)
            queue.summarize(args)
            state.update(status="complete", active_stage=None, completed_at=queue.now())
        except Exception as error:
            state.update(status="failed", error=repr(error), failed_at=queue.now())
            raise
        finally:
            queue.save(path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    run(parser.parse_args().root)

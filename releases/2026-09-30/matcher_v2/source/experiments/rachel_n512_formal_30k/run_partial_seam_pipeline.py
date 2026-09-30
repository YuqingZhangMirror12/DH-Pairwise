"""Finite E4/E5 continuation, preserving the active E0-E3 and gap queue.

Wait by exact predecessor PID (no GPU context), then require its completion
receipt before sequential TRAIN/TEST/REAL jobs. Never restart a live job.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

from experiments.rachel_n512_formal_30k.run_all_weathering_ablations import (
    read_json, save_json, now)


def require_gap_complete(path):
    state = read_json(path)
    stages = state.get("stages", [])
    if state.get("status") != "complete" or len(stages) != 9 or any(
            s.get("status") != "complete" or s.get("returncode") != 0 for s in stages):
        raise RuntimeError("preceding full gap evaluation did not finish successfully")
    return state


def wait_for_gap(root):
    root = Path(root).resolve()
    path = root / "pipeline_state.json"
    state = read_json(path)
    if state.get("status") == "complete":
        return require_gap_complete(path)
    if state.get("status") not in ("waiting_for_e0_e3", "running"):
        raise RuntimeError("gap predecessor is neither active nor complete")
    pid = state.get("pid")
    if type(pid) is not int or pid <= 1:
        raise RuntimeError("an exact predecessor PID is required")
    try:
        cmdline = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().replace("\0", " ")
    except FileNotFoundError:
        return require_gap_complete(path)
    if "run_gap_stress_pipeline" not in cmdline or str(root) not in cmdline:
        return require_gap_complete(path)
    subprocess.run(["tail", "--pid=" + str(pid), "--sleep-interval=10", "-f", "/dev/null"], check=True)
    return require_gap_complete(path)


def stage_plan(args):
    root = Path(args.root).resolve()
    common = ["--dataset", args.dataset, "--device", "cuda:0", "--workers", str(args.workers)]
    stages = []
    for arm in ("e4", "e5"):
        train = root / arm / "training"
        stages.append(("train_" + arm, "experiments.rachel_n512_formal_30k.train_partial_seam",
            common + ["--arm", arm, "--checkpoint", args.checkpoint,
                "--train-manifest", args.train_manifest, "--cache-dir", str(root / "weathering_cache"),
                "--output", str(train), "--log-every", "100"]))
        for split in ("test", "real"):
            params = common + ["--training-run", str(train), "--split", split,
                                "--output", str(root / arm / "evaluation" / split)]
            if split == "real":
                params += ["--prepared-cache", args.prepared_cache, "--translation-gt-json", args.translation_gt_json]
            stages.append(("evaluate_" + arm + "_" + split,
                "experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint", params))
    return stages


def summarize(root):
    arms = {}
    for arm in ("e4", "e5"):
        train = read_json(root / arm / "training" / "train_val_freeze.json")
        if (train.get("status"), train.get("completed_global_exposures"), train.get("completed_optimizer_updates")) != ("complete", 120000, 7500):
            raise RuntimeError("incomplete equal-budget partial training arm")
        evaluations = {}
        for split in ("test", "real"):
            path = root / arm / "evaluation" / split
            receipt = read_json(path / "receipt.json")
            if receipt.get("status") != "complete":
                raise RuntimeError("incomplete held-out evaluation")
            summary = read_json(path / "summary.json")
            evaluations[split] = dict(classification=summary["classification"],
                layout=summary["layout"], source=str(path / "summary.json"))
        arms[arm] = dict(training_freeze=train, evaluations=evaluations)
    result = dict(schema_version="rachel-partial-seam-results/1", status="complete",
        complete=True, arms=arms, selection_on_real=False,
        comparison_controls="E0 for E4; E1 for E5; both fixed common warm start and 120k exposure budget",
        caveat="REAL was used for qualitative design and repeated research diagnostics; not an untouched blind test")
    save_json(root / "summary.json", result)
    return result


def run(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pipeline_state.json"
    if path.exists():
        raise FileExistsError("partial-seam pipeline already exists; do not duplicate it")
    prep = read_json(root / "preparation" / "summary.json")
    preflight = read_json(root / "preflight" / "preflight.json")
    if (prep.get("status"), prep.get("full_train"), prep.get("counts", {}).get("rows")) != ("complete", True, 24000):
        raise RuntimeError("full 24k generation manifest must be complete before training")
    if prep["counts"].get("applied_positive", 0) <= 0 or prep["counts"]["applied_positive"] != prep["counts"].get("applied_negative"):
        raise RuntimeError("partial recipe has no balanced actual application")
    if preflight.get("status") != "complete" or not preflight.get("basic_contract_passed") or preflight.get("weights_saved") is not False:
        raise RuntimeError("bounded E4/E5 GPU smoke must finish; smoke weights cannot be reused")
    source = Path(__file__).resolve().parents[2]
    state = dict(schema_version="rachel-partial-seam-pipeline/1", status="waiting_for_gap",
        pid=os.getpid(), started_at=now(), source=str(source), arguments=vars(args),
        active_stage="waiting_for_gap", predecessor=str(Path(args.gap_root) / "pipeline_state.json"),
        stages=[dict(name=name, module=module, arguments=params, status="queued")
                for name, module, params in stage_plan(args)],
        registered_budget_per_arm=dict(exposures=120000, optimizer_updates=7500, clean_val_events=5),
        original_files_modified=False, test_or_real_used_for_fit=False)
    save_json(path, state)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(source), PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        wait_for_gap(args.gap_root)
        state["status"] = "running"
        for stage in state["stages"]:
            stage.update(status="running", started_at=now(), log=str(root / (stage["name"] + ".log")))
            state["active_stage"] = stage["name"]
            save_json(path, state)
            command = [args.python, "-u", "-m", stage["module"]] + stage["arguments"]
            with Path(stage["log"]).open("x") as log:
                child = subprocess.Popen(command, cwd=str(source), env=environment,
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                stage["pid"] = child.pid
                save_json(path, state)
                code = child.wait()
            stage.update(status="complete" if code == 0 else "failed", returncode=code, completed_at=now())
            save_json(path, state)
            if code:
                raise RuntimeError(stage["name"] + " failed; see " + stage["log"])
        summarize(root)
        state.update(status="complete", active_stage=None, completed_at=now())
        save_json(path, state)
    except Exception as error:
        state.update(status="failed", failed_at=now(), error=repr(error))
        save_json(path, state)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "gap-root", "dataset", "train-manifest", "checkpoint", "prepared-cache", "translation-gt-json"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--preparation-pid", type=int,
                   help="exact active CPU preparation PID, awaited once before recipe checks")
    return p


def launch(args):
    root = Path(args.root).resolve()
    launch_path = root / "launch_state.json"
    if launch_path.exists():
        raise FileExistsError("partial continuation launcher already exists")
    record = dict(status="waiting_for_preparation", pid=os.getpid(), started_at=now(),
                  preparation_pid=args.preparation_pid, arguments=vars(args))
    save_json(launch_path, record)
    try:
        if args.preparation_pid:
            pid = args.preparation_pid
            if pid <= 1:
                raise ValueError("invalid preparation PID")
            try:
                command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().replace("\0", " ")
            except FileNotFoundError:
                command = None
            if command is not None:
                if "prepare_partial_seam" not in command or str(root / "preparation") not in command:
                    raise RuntimeError("preparation PID belongs to a different process")
                subprocess.run(["tail", "--pid=" + str(pid), "--sleep-interval=10", "-f", "/dev/null"], check=True)
        record.update(status="continuation_running", preparation_finished_at=now())
        save_json(launch_path, record)
        run(args)
        record.update(status="complete", completed_at=now())
    except Exception as error:
        record.update(status="failed", error=repr(error), failed_at=now())
        save_json(launch_path, record)
        raise
    save_json(launch_path, record)


if __name__ == "__main__":
    launch(parser().parse_args())

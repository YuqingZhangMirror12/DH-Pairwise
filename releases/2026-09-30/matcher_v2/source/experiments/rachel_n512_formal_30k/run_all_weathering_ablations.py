"""Finite continuation: wait for existing E1, run E2, fixed E3, then summarize.

The existing E1 process is never interrupted or modified. Linux pidfd (or GNU
tail --pid on Python3.8) waits without a GPU context or dataset inspection.
This is a single dependency queue, not a recurring task or hyperparameter search.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import subprocess
import sys


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def require_complete_predecessor(path):
    state = read_json(path)
    if state.get("status") != "complete":
        raise RuntimeError("E1 predecessor is not complete: " + str(state.get("status")))
    if any(stage.get("status") != "complete" or stage.get("returncode") != 0
           for stage in state.get("stages", [])) or len(state.get("stages", [])) != 3:
        raise RuntimeError("E1 must finish its training, TEST and REAL evaluation before the continuation")
    return state


def wait_for_e1(root):
    path = Path(root) / "pipeline_state.json"
    before = read_json(path)
    if before.get("status") == "complete":
        return require_complete_predecessor(path)
    if before.get("status") != "running":
        raise RuntimeError("E1 is not running and cannot be awaited: " + str(before.get("status")))
    pid = before["pid"]
    if not isinstance(pid, int) or pid <= 1:
        raise RuntimeError("An exact live E1 PID is required")
    if not hasattr(os, "pidfd_open"):
        # The server's Python3.8 lacks os.pidfd_open. GNU tail provides the
        # ordinary process-dependency wait; it only observes process liveness.
        try:
            command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().replace("\x00", " ")
        except FileNotFoundError:
            return require_complete_predecessor(path)
        if "run_edge_weathering_pipeline" not in command or str(Path(root).resolve()) not in command:
            return require_complete_predecessor(path)
        subprocess.run(["tail", "--pid=" + str(pid), "--sleep-interval=10", "-f", "/dev/null"], check=True)
        return require_complete_predecessor(path)
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return require_complete_predecessor(path)
    try:
        # Confirm the descriptor was opened on the intended pipeline, not a
        # recycled PID. If it has just exited, only its complete receipt counts.
        try:
            command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().replace("\x00", " ")
        except FileNotFoundError:
            return require_complete_predecessor(path)
        if ("run_edge_weathering_pipeline" not in command or str(Path(root).resolve()) not in command):
            return require_complete_predecessor(path)
        select.select([descriptor], [], [])
    finally:
        os.close(descriptor)
    return require_complete_predecessor(path)


def stage_plan(args):
    root = Path(args.root).resolve()
    e2, e3 = root / "e2", root / "e3"
    training = e2 / "training" / "e2"
    common = ["--dataset", args.dataset, "--device", "cuda:0", "--workers", str(args.workers)]
    return [
        ("train_e2", "experiments.rachel_n512_formal_30k.train_edge_consistency", common + [
            "--checkpoint", args.checkpoint, "--train-manifest", args.train_manifest,
            "--output", str(training), "--cache-dir", args.augmentation_cache, "--log-every", "100"]),
        ("evaluate_e2_test", "experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint", common + [
            "--training-run", str(training), "--split", "test", "--output", str(e2 / "evaluation" / "test")]),
        ("evaluate_e2_real", "experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint", common + [
            "--training-run", str(training), "--split", "real", "--output", str(e2 / "evaluation" / "real"),
            "--prepared-cache", args.prepared_cache, "--translation-gt-json", args.translation_gt_json]),
        ("fit_e3_and_evaluate", "experiments.rachel_n512_formal_30k.run_seam_geometry_head", common + [
            "--training-run", str(training), "--train-manifest", args.train_manifest,
            "--prepared-cache", args.prepared_cache, "--translation-gt-json", args.translation_gt_json,
            "--output", str(e3)]),
        ("summarize_e0_e1_e2_e3", "experiments.rachel_n512_formal_30k.summarize_weathering_ablations", [
            "--e0-root", args.e0_root, "--e1-root", args.e1_root, "--e2-root", str(e2),
            "--e3-root", str(e3), "--output", str(root / "summary")]),
    ]


def run(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "pipeline_state.json"
    if state_path.exists():
        raise FileExistsError("Refusing to start a duplicate ablation continuation")
    preflight = read_json(root / "e2" / "preflight" / "preflight.json")
    if (preflight.get("status") != "complete" or not preflight.get("basic_contract_passed")
            or preflight.get("weights_saved") is not False):
        raise RuntimeError("E2 must pass its bounded GPU preflight; pilot weights may not be reused")
    e0 = read_json(Path(args.e0_root) / "realism_training" / "matched24k" / "train_val_freeze.json")
    if (e0.get("status"), e0.get("unique_count"), e0.get("completed_global_exposures"),
            e0.get("completed_optimizer_updates")) != ("complete", 24000, 120000, 7500):
        raise RuntimeError("Existing E0 is not the completed matched24k equal-budget control")
    source = Path(__file__).resolve().parents[2]
    stages = [dict(name=name, status="queued", module=module, arguments=arguments)
              for name, module, arguments in stage_plan(args)]
    state = dict(schema_version="rachel-all-weathering-ablations/1", status="waiting_for_e1",
        started_at=now(), pid=os.getpid(), source=str(source), arguments=vars(args), stages=stages,
        predecessor=str(Path(args.e1_root) / "pipeline_state.json"),
        active_stage="waiting_for_e1", e0_reused=True, e0_completed_exposures=120000,
        e2_budget=dict(pair_draws=120000, optimizer_updates=7500, clean_teacher_forward_is_extra_compute=True),
        e3_anchor="E2 clean-VAL-selected winner, irrespective of REAL results",
        e3_primary_head="score_geometry", e3_attribution_control="score_only",
        model_or_threshold_selection_on_test_or_real=False,
        all_inference_layouts="unchanged fixed Top2 pure-translation decoder")
    save_json(state_path, state)
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        predecessor = wait_for_e1(args.e1_root)
        state.update(status="running", predecessor_completed_at=predecessor.get("completed_at"))
        save_json(state_path, state)
        for stage in state["stages"]:
            command = [args.python, "-u", "-m", stage["module"]] + stage["arguments"]
            stage.update(status="running", started_at=now(), command=command,
                         log=str(root / (stage["name"] + ".log")))
            state["active_stage"] = stage["name"]
            save_json(state_path, state)
            with Path(stage["log"]).open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=str(source), env=environment,
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                stage["pid"] = child.pid
                save_json(state_path, state)
                code = child.wait()
            if code == 0 and stage["name"] == "summarize_e0_e1_e2_e3":
                if read_json(root / "summary" / "summary.json").get("complete") is not True:
                    code = 1
                    stage["failure_reason"] = "Summary exists but one or more required ablation results are incomplete"
            stage.update(status="complete" if code == 0 else "failed", returncode=code, completed_at=now())
            save_json(state_path, state)
            if code:
                raise RuntimeError(stage["name"] + " failed; see " + stage["log"])
        state.update(status="complete", completed_at=now(), active_stage=None)
        save_json(state_path, state)
    except Exception as error:
        state.update(status="failed", failed_at=now(), error=repr(error))
        save_json(state_path, state)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "e0-root", "e1-root", "checkpoint", "dataset", "train-manifest",
                 "augmentation-cache", "prepared-cache", "translation-gt-json"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--workers", type=int, default=4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

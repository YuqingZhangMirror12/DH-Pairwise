"""One finite E1 training run followed by fixed TEST and REAL evaluation.

This is a detached, sequential experiment process, not a recurring monitor.
It never retunes a classifier/decoder or launches additional ablation arms.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def run(args):
    root = Path(args.root).resolve()
    source = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "pipeline_state.json"
    if state_path.exists():
        raise FileExistsError("Refusing to duplicate an existing E1 pipeline: " + str(state_path))
    preflight = json.loads((root / "preflight" / "preflight.json").read_text())
    if preflight.get("status") != "complete":
        raise RuntimeError("E1 GPU preflight must complete before formal training")
    training = root / "training" / "e1"
    common = ["--dataset", args.dataset, "--device", "cuda:0", "--workers", str(args.workers)]
    stages = [
        ("train_e1", "experiments.rachel_n512_formal_30k.train_edge_weathering", common + [
            "--checkpoint", args.checkpoint, "--train-manifest", args.train_manifest,
            "--output", str(training), "--cache-dir", str(root / "augmentation_cache"),
            "--log-every", "100"]),
        ("evaluate_test", "experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint", common + [
            "--training-run", str(training), "--split", "test", "--output", str(root / "evaluation" / "test")]),
        ("evaluate_real", "experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint", common + [
            "--training-run", str(training), "--split", "real", "--output", str(root / "evaluation" / "real"),
            "--prepared-cache", args.prepared_cache, "--translation-gt-json", args.translation_gt_json]),
    ]
    state = dict(schema_version="rachel-edge-weathering-finite-pipeline/1", status="running",
                 started_at=now(), pid=os.getpid(), source=str(source), stages=[],
                 design="E1 only: inward weathering with source-arc targets and clean-only old pose auxiliary",
                 classifier_or_decoder_changed=False, planned_training_exposures=120000,
                 clean_control_training=args.clean_control_training,
                 clean_control_policy="Existing matched24k E0; same initialization, data manifest and 120k budget. No duplicate E0 run.",
                 evaluation_policy="Clean VAL winner and thresholds, fixed original Top2 layout, no REAL fitting",
                 preflight_weights_used=False)
    write_json(state_path, state)
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        for name, module, arguments in stages:
            command = [args.python, "-u", "-m", module] + arguments
            stage = dict(name=name, status="running", started_at=now(), command=command,
                         log=str(root / (name + ".log")))
            state["stages"].append(stage)
            state["active_stage"] = name
            write_json(state_path, state)
            with Path(stage["log"]).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=str(source), env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                stage["pid"] = process.pid
                write_json(state_path, state)
                returncode = process.wait()
            stage.update(status="complete" if returncode == 0 else "failed", returncode=returncode,
                         completed_at=now())
            write_json(state_path, state)
            if returncode:
                raise RuntimeError(name + " exited with code " + str(returncode))
        state.update(status="complete", completed_at=now(), active_stage=None)
        write_json(state_path, state)
    except Exception as error:
        state.update(status="failed", failed_at=now(), error=repr(error))
        write_json(state_path, state)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--clean-control-training", required=True)
    p.add_argument("--prepared-cache", required=True)
    p.add_argument("--translation-gt-json", required=True)
    p.add_argument("--workers", type=int, default=4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

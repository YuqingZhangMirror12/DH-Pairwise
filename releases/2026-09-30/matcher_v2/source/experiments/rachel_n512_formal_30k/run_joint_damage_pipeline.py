"""Finite single-GPU queue for matched scratch schedules and separate heads.

No original run is restarted or edited. CPU guided-data preparation can run
alongside the first two training arms; partial arms require its explicit ready
receipt. All four matchers are evaluated on the same unchanged TEST and REAL.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ARMS = ("staged", "joint", "partial_low", "partial_high")
TRAIN = "experiments.rachel_n512_formal_30k.train_joint_damage"
EVALUATE = "experiments.rachel_n512_formal_30k.evaluate_joint_damage"
HEADS = "experiments.rachel_n512_formal_30k.run_damage_separate_heads"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def stage_plan(args):
    root = Path(args.root).resolve()
    data = ["--dataset", args.dataset, "--train-manifest", args.train_manifest,
            "--checkpoint", args.checkpoint, "--device", "cuda:0", "--workers", str(args.workers)]
    stages = []

    def training(name, arm, output, *, smoke=None, epoch=None):
        parameters = data + ["--arm", arm, "--output", str(output),
            "--cache-dir", str(root / "weathering_cache"), "--log-every", "100"]
        if arm.startswith("partial_"):
            parameters += ["--outline-bank", args.outline_bank]
        if smoke:
            parameters += ["--smoke=" + smoke, "--smoke-epoch", str(epoch)]
        stages.append(dict(name=name, module=TRAIN, parameters=parameters, status="queued"))

    training("smoke_staged_clean", "staged", root / "smoke/staged_clean", smoke="128/8", epoch=1)
    training("smoke_staged_weather", "staged", root / "smoke/staged_weather", smoke="128/8", epoch=4)
    training("smoke_joint", "joint", root / "smoke/joint", smoke="128/8", epoch=1)
    for arm in ARMS:
        if arm == "partial_low":
            stages.append(dict(name="wait_guided_preparation", status="queued", internal="guided_ready"))
        if arm.startswith("partial_"):
            training("smoke_" + arm, arm, root / "smoke" / arm, smoke="128/8", epoch=1)
        training("train_" + arm, arm, root / arm / "training")
        for split in ("test", "real"):
            parameters = ["--training-run", str(root / arm / "training"), "--dataset", args.dataset,
                "--split", split, "--output", str(root / arm / "evaluation" / split),
                "--device", "cuda:0", "--workers", str(args.workers)]
            if split == "real":
                parameters += ["--prepared-cache", args.prepared_cache,
                               "--translation-gt-json", args.translation_gt_json]
            stages.append(dict(name="evaluate_" + arm + "_" + split, module=EVALUATE,
                               parameters=parameters, status="queued"))
    stages.append(dict(name="fit_evaluate_separate_heads", module=HEADS, status="queued", parameters=[
        "--checkpoint", args.head_checkpoint, "--dataset", args.dataset,
        "--train-manifest", args.train_manifest, "--prepared-cache", args.prepared_cache,
        "--translation-gt-json", args.translation_gt_json, "--output", str(root / "heads"),
        "--cache-dir", str(root / "head_weathering_cache"), "--device", "cuda:0",
        "--workers", str(args.workers)]))
    parameters = ["--strata", args.straight_strata, "--output", str(root / "straight_strata")]
    for arm in ARMS:
        parameters += ["--evaluation", arm + "=" + str(root / arm / "evaluation/real")]
    stages.append(dict(name="summarize_fixed_straight_strata", status="queued",
        module="experiments.rachel_n512_formal_30k.summarize_real_straight_strata", parameters=parameters))
    return stages


def wait_guided(args, state, stage, path):
    stage.update(status="running", started_at=now(), receipt=str(Path(args.guided_ready).resolve()))
    state.update(active_stage=stage["name"], status="waiting_for_guided_preparation")
    save(path, state)
    # This is a remote finite queue dependency, not a new GPU worker. A failed
    # preparation receipt is terminal; absence never causes a duplicate launch.
    while not Path(args.guided_ready).exists():
        failure = Path(args.guided_ready).with_name("failure.json")
        if failure.exists():
            raise RuntimeError("guided preparation failed: " + str(failure))
        time.sleep(30)
    ready = read(args.guided_ready)
    if (ready.get("status") != "complete" or ready.get("train_only") is not True
            or ready.get("hard_negative_enabled") is not False
            or ready.get("canonical_weather_epochs") != [1, 2, 3]
            or ready.get("paired_low_high_verified") is not True):
        raise ValueError("guided preparation receipt does not cover the registered low/high data comparison")
    stage.update(status="complete", completed_at=now(), returncode=0, readiness=ready)
    save(path, state)


def run_stage(args, state, stage, path):
    root = Path(args.root).resolve()
    source = Path(args.source).resolve()
    command = [args.python, "-u", "-m", stage["module"]] + stage["parameters"]
    log_path = root / (stage["name"] + ".log")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(source), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    stage.update(status="running", started_at=now(), command=command, log=str(log_path))
    state.update(active_stage=stage["name"], status="running")
    save(path, state)
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=source, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        stage["pid"] = child.pid
        save(path, state)
        returncode = child.wait()
    if returncode == 0 and stage["name"].startswith("smoke_"):
        params = stage["parameters"]
        output = Path(params[params.index("--output") + 1])
        result = read(output / "smoke.json")
        if (result.get("status"), result.get("weights_discarded"),
                result.get("formal_training_budget_counted"), result.get("training", {}).get("samples"),
                result.get("training", {}).get("optimizer_updates")) != ("complete", True, False, 128, 8):
            raise RuntimeError("discard-only smoke did not complete the required actual updates")
        digest = result["initial_weights_sha256"]
        if state.get("initial_weights_sha256", digest) != digest:
            raise RuntimeError("smoke initialization differs between training arms")
        state["initial_weights_sha256"] = digest
        stage["smoke_result"] = result
    stage.update(status="complete" if returncode == 0 else "failed", returncode=returncode, completed_at=now())
    save(path, state)
    if returncode:
        raise RuntimeError(stage["name"] + " failed; inspect its original log and handle")


def summarize(args):
    root = Path(args.root).resolve()
    models, init_ids = {}, set()
    for arm in ARMS:
        freeze = read(root / arm / "training/train_val_freeze.json")
        if (freeze.get("status"), freeze.get("completed_global_exposures"),
                freeze.get("completed_optimizer_updates"), freeze.get("completed_validation_events"),
                freeze.get("initialization")) != ("complete", 144000, 9000, 6, "random"):
            raise ValueError("incomplete full-network arm: " + arm)
        init_ids.add(freeze["initial_weights_sha256"])
        evaluations = {}
        for split in ("test", "real"):
            directory = root / arm / "evaluation" / split
            summary, receipt = read(directory / "summary.json"), read(directory / "receipt.json")
            if summary.get("status") != "complete" or receipt.get("status") != "complete":
                raise ValueError("incomplete held-out evaluation: " + arm + "/" + split)
            evaluations[split] = dict(classification=summary["classification"], layout=summary["layout"],
                size_strata=summary.get("size_strata"), source=str(directory / "summary.json"))
        models[arm] = dict(training_freeze=freeze, evaluations=evaluations)
    if len(init_ids) != 1:
        raise ValueError("full-network arms did not start from identical random weights")
    heads = {split: read(root / "heads" / split / "summary.json") for split in ("test", "real")}
    if any(report.get("status") != "complete" for report in heads.values()):
        raise ValueError("independent pairability/reliability heads are not fully evaluated")
    straight = read(root / "straight_strata/summary.json")
    if straight.get("status") != "complete":
        raise ValueError("fixed straight strata not complete")
    result = dict(schema_version="rachel-joint-damage-results/1", status="complete", completed_at=now(),
        arms=models, separate_heads=heads, straight_strata=str(root / "straight_strata/summary.json"),
        initial_weights_sha256=next(iter(init_ids)), old_e1_historical_reference_only=True,
        hard_negative_overlay_used=False, test_or_real_used_for_fit=False,
        caveat="REAL is the same repeatedly inspected research set, not a new blind holdout")
    save(root / "summary.json", result)


def run(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pipeline_state.json"
    if path.exists():
        raise FileExistsError("queue already exists; inspect/recover its actual stage, do not duplicate")
    state = dict(schema_version="rachel-joint-damage-pipeline/1", status="running", pid=os.getpid(),
        started_at=now(), arguments=vars(args), active_stage=None, stages=stage_plan(args),
        error=None, full_network_budget_per_arm=dict(exposures=144000, updates=9000, epochs=6),
        no_extra_straight_negatives=True, original_natural_straight_pairs_retained=True)
    save(path, state)
    try:
        for stage in state["stages"]:
            if stage.get("internal") == "guided_ready":
                wait_guided(args, state, stage, path)
            else:
                run_stage(args, state, stage, path)
        summarize(args)
        state.update(status="complete", active_stage=None, completed_at=now())
    except Exception as error:
        state.update(status="failed", error=repr(error), failed_at=now())
        raise
    finally:
        save(path, state)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "python", "checkpoint", "head-checkpoint", "dataset", "train-manifest",
                 "outline-bank", "prepared-cache", "translation-gt-json", "straight-strata", "guided-ready"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--workers", type=int, default=4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

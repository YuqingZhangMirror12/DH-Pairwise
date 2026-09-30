"""Finite CPU precomputation for all spectral ablation input populations.

No GPU, training, retry, or source/output replacement. Uses the registered cache
CLI unchanged, then writes explicit digests for later training/evaluation.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training import cache

SCHEMA = "rachel-spectral-cpu-preparation/1"
MODULE = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training.cache"


def stages(root, python, workers):
    return [dict(name=name, output=str(root / name), command=[python, "-u", "-m", MODULE,
        "--output", str(root / name), "--split", split, "--workers", str(workers)])
        for name, split in (("train_val", "train_val"), ("test", "test"), ("real", "real"), ("ood", "ood"))]


def register(stage):
    path = Path(stage["output"]) / "bundle.json"
    sha = cache.f.file_sha256(path)
    bundle, caches = cache.load_bundle(path, sha, require_training=stage["name"] == "train_val")
    expected = {"train", "val"} if stage["name"] == "train_val" else {stage["name"]}
    if set(caches) != expected or any(len(caches[s].records) != cache.SPLIT_COUNTS[s] for s in expected):
        raise ValueError("CPU cache does not cover the complete unchanged population")
    return dict(bundle=str(path), sha256=sha, pairs={s: len(caches[s].records) for s in expected},
                wall_seconds={s: bundle["splits"][s]["elapsed_wall_s"] for s in expected})


def run(args):
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be1..8")
    source = Path(args.source_root).resolve(strict=True)
    if source != Path(__file__).resolve().parents[4]:
        raise ValueError("run from the dedicated source snapshot")
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = dict(schema_version=SCHEMA, status="running", pid=os.getpid(), source_root=str(source),
        started_at_unix=time.time(), workers=args.workers, device="cpu", GPU_jobs_started=False,
        source_checkpoint_sha256=cache.shared.SOURCE_SHA, stages=stages(root, sys.executable, args.workers),
        registrations={}, completed_stages=0)
    status_path = root / "status.json"
    save = lambda: cache.shared.old.save_json(status_path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    with (root / "prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        try:
            for index, stage in enumerate(state["stages"]):
                log_path = root / (stage["name"] + ".log")
                with log_path.open("x") as log:
                    child = subprocess.Popen(stage["command"], cwd=source, env=env,
                        stdout=log, stderr=subprocess.STDOUT)
                    stage.update(status="running", pid=child.pid, log=str(log_path), started_at_unix=time.time())
                    state.update(active_name=stage["name"], active_pid=child.pid)
                    save()
                    print(json.dumps(dict(stage=stage["name"], status="started", pid=child.pid)), flush=True)
                    code = child.wait()
                stage.update(returncode=code, finished_at_unix=time.time())
                if code:
                    raise RuntimeError(stage["name"] + " failed; no automatic retry")
                state["registrations"][stage["name"]] = register(stage)
                stage["status"] = "complete"
                state.update(completed_stages=index + 1, active_pid=None)
                save()
            registry = {s: {k: state["registrations"][s][k] for k in ("bundle", "sha256")}
                        for s in ("test", "real", "ood")}
            registry_path = root / "endpoint_registry.json"
            cache.shared.old.save_json(registry_path, registry)
            state.update(status="complete", active_name=None, endpoint_registry=str(registry_path),
                         endpoint_registry_sha256=cache.f.file_sha256(registry_path))
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            state["finished_at_unix"] = time.time()
            save()
    print(json.dumps(dict(status=state["status"], output=str(root))), flush=True)
    return state


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--workers", type=int, default=4)
    run(p.parse_args())

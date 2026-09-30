"""Explicit finite30-stage queue; all three smokes/cache registration required."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training import train
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import run_continuation_queue as prior

MODULE = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training."


def load_registry(path, sha):
    if train.f.file_sha256(path) != sha:
        raise ValueError("endpoint registry SHA differs")
    registry = json.loads(Path(path).read_text())
    if set(registry) != {"test", "real", "ood"}:
        raise ValueError("requires three separate complete held-out input caches")
    for split, record in registry.items():
        _, caches = train.cache.load_bundle(record["bundle"], record["sha256"], require_training=False)
        if set(caches) != {split} or len(caches[split].records) != train.cache.SPLIT_COUNTS[split]:
            raise ValueError("endpoint cache split/count differs")
    return registry


def stage_plan(root, training_bundle, training_sha, registry, python=sys.executable):
    template = prior.stage_plan(root, python)[:10]
    result = []
    for variant in train.VARIANTS:
        destination = root / (variant + "_c16")
        command = [python, "-m", MODULE + "train", "--variant", variant,
            "--cache-bundle", str(training_bundle), "--cache-bundle-sha", training_sha,
            "--output", str(destination)]
        result.append(dict(name=variant + "_C9_C16", kind="training", command=command,
            completion=str(destination / "status.json")))
        for old in template[1:]:
            command = [s.replace(str(root / "s6_d2_c16"), str(destination)) for s in old["command"]]
            command[2] = MODULE + "evaluate"
            split = command[command.index("--split") + 1]
            command += ["--summary-bundle", registry[split]["bundle"], "--summary-bundle-sha", registry[split]["sha256"]]
            result.append(dict(name=old["name"].replace("s6_d2", variant), kind="endpoint_evaluation", command=command,
                completion=old["completion"].replace(str(root / "s6_d2_c16"), str(destination))))
    return result


def validate_smokes(root, bundle_sha):
    normalization = set()
    for variant in train.VARIANTS:
        row = json.loads((root / (variant + "_smoke32/smoke.json")).read_text())
        ident = row.get("spectral_identity", {})
        if (row.get("status") != "smoke_complete" or row.get("pair_exposures") != 32
                or row.get("optimizer_updates") != 2 or row.get("formal_training_counted") is not False
                or row.get("frozen_base_unchanged") is not True or ident.get("variant") != variant
                or ident.get("cache_bundle_sha256") != bundle_sha
                or ident.get("source_checkpoint_sha256") != train.shared.SOURCE_SHA):
            raise ValueError("three same-source/cache discard32 smokes required")
        for path, sha in ident["implementation_sha256"].items():
            if train.old._sha256(path) != sha:
                raise ValueError("code changed after smoke")
        normalization.add(ident["normalizer_sha256"])
    if len(normalization) != 1:
        raise ValueError("all three controls must share one TRAIN10D normalizer")


def run(args):
    root, source = Path(args.output_root).resolve(strict=True), Path(args.source_root).resolve(strict=True)
    train.cache.load_bundle(args.cache_bundle, args.cache_bundle_sha)
    registry = load_registry(args.endpoint_registry, args.endpoint_registry_sha)
    validate_smokes(root, args.cache_bundle_sha)
    status_path = root / "queue_status.json"
    with train.old.run_lock(root):
        if status_path.exists():
            raise ValueError("queue exists; no duplicate dispatch or automatic retry")
        status = dict(schema_version="rachel-spectral-queue/1", status="running", pid=os.getpid(),
            started_at_unix=time.time(), per_epoch_heldout_evaluation=False,
            stages=stage_plan(root, args.cache_bundle, args.cache_bundle_sha, registry))
        prior.save(status_path, status)
        try:
            for index, stage in enumerate(status["stages"]):
                log_path = root / ("%02d_%s.log" % (index + 1, stage["name"]))
                stage.update(status="running", started_at_unix=time.time(), log=str(log_path))
                with log_path.open("x") as log:
                    child = subprocess.Popen(stage["command"], cwd=source, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONUNBUFFERED="1"))
                    stage["pid"] = child.pid
                    status.update(active_stage=index + 1, active_name=stage["name"], active_pid=child.pid)
                    prior.save(status_path, status)
                    code = child.wait()
                stage.update(returncode=code, finished_at_unix=time.time())
                if code:
                    raise RuntimeError(stage["name"] + " failed; no retry")
                receipt = json.loads(Path(stage["completion"]).read_text())
                if receipt.get("status") != "complete" or (stage["kind"] == "training"
                        and receipt.get("completed_segments") != 112):
                    raise ValueError("stage did not complete registered budget")
                stage["status"] = "complete"
                prior.save(status_path, status)
            status.update(status="complete", completed_stages=30, active_pid=None, active_name=None)
        except BaseException as error:
            status.update(status="failed", error=repr(error))
            raise
        finally:
            status["finished_at_unix"] = time.time()
            prior.save(status_path, status)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "output-root", "cache-bundle", "cache-bundle-sha", "endpoint-registry", "endpoint-registry-sha"):
        p.add_argument("--" + name, required=True)
    run(p.parse_args())

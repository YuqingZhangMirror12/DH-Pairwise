"""Explicit, finite sequential C1/C2 queue. No detach or automatic launch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local import train
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import run_continuation_queue as prior

MODULE = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local."


def stage_plan(root, python=sys.executable):
    """Same 24K data and original evaluator dataset arguments as C0."""
    template = prior.stage_plan(root, python)[:10]
    result = []
    for arm in train.MODES:
        training = root / (arm + "_c16")
        result.append(dict(name=arm + "_C9_C16", kind="training",
            command=[python, "-m", MODULE + "train", "--arm", arm, "--output", str(training)],
            completion=str(training / "status.json")))
        for old_stage in template[1:]:
            cmd = [s.replace(str(root / "s6_d2_c16"), str(training)) for s in old_stage["command"]]
            cmd[2] = MODULE + "evaluate"
            result.append(dict(name=old_stage["name"].replace("s6_d2", arm), kind="endpoint_evaluation",
                command=cmd, completion=old_stage["completion"].replace(str(root / "s6_d2_c16"), str(training))))
    return result


def validate_smokes(root):
    for arm in train.MODES:
        receipt = json.loads((root / (arm + "_smoke32/smoke.json")).read_text())
        ident = receipt.get("candidate_identity", {})
        if (receipt.get("status") != "smoke_complete" or receipt.get("pair_exposures") != 32
                or receipt.get("optimizer_updates") != 2 or receipt.get("formal_training_counted") is not False
                or receipt.get("frozen_base_unchanged") is not True
                or ident.get("mode") != train.MODES[arm] or ident.get("source_checkpoint_sha256") != train.SOURCE_SHA):
            raise ValueError("requires both independent exact-source discard32 smokes")
        current = {Path(p).name: train.old._sha256(p) for p in
            (train.__file__, train.local.__file__, train.cont.__file__, train.old.__file__)}
        if ident.get("implementation_sha256") != current:
            raise ValueError("code changed since smoke; repeat in a new disposable output")


def run(args):
    source = Path(args.source_root).resolve(strict=True)
    root = Path(args.output_root).resolve(strict=True)
    validate_smokes(root)
    status_path = root / "queue_status.json"
    # Each child holds the shared GPU lock for its full stage, nonblocking.
    # The queue's own directory lock stops duplicate dispatch; no parent GPU
    # lock is retained (which would deadlock a child). If another run claims
    # the GPU between stages, the next child fails and this queue stops.
    with train.old.run_lock(root):
        if status_path.exists():
            raise ValueError("queue already exists; never auto-restart completed/failed work")
        status = dict(status="running", schema_version="rachel-candidate-local-queue/1",
            pid=os.getpid(), started_at_unix=time.time(), stages=stage_plan(root),
            gpu_lock=str(train.GPU_LOCK), per_epoch_held_out_evaluation=False,
            sequence="C1 train+endpoint9 then C2 train+endpoint9")
        prior.save(status_path, status)
        try:
            for index, stage in enumerate(status["stages"]):
                stage.update(status="running", started_at_unix=time.time())
                log_path = root / ("%02d_%s.log" % (index + 1, stage["name"]))
                with log_path.open("x") as log:
                    child = subprocess.Popen(stage["command"], cwd=source, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONUNBUFFERED="1"))
                    stage.update(pid=child.pid, log=str(log_path))
                    status.update(active_stage=index + 1, active_name=stage["name"], active_pid=child.pid)
                    prior.save(status_path, status)
                    code = child.wait()
                stage.update(returncode=code, finished_at_unix=time.time())
                if code:
                    raise RuntimeError(stage["name"] + " failed; no automatic retries")
                receipt = json.loads(Path(stage["completion"]).read_text())
                if receipt.get("status") != "complete" or (stage["kind"] == "training"
                        and receipt.get("completed_segments") != 112):
                    raise ValueError("stage did not reach its registered complete endpoint")
                stage["status"] = "complete"
                prior.save(status_path, status)
            status.update(status="complete", completed_stages=20, active_pid=None, active_name=None)
        except BaseException as error:
            status.update(status="failed", error=repr(error))
            raise
        finally:
            status["finished_at_unix"] = time.time()
            prior.save(status_path, status)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output-root", required=True)
    run(p.parse_args())

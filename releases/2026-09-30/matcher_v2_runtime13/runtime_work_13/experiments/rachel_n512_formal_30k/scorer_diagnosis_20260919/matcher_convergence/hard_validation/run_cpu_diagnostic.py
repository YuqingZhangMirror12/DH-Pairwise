"""Finite CPU preparation then frozen M12 hard-VAL evaluation; no GPU queue.

This is a one-shot two-stage diagnostic, not a scheduler or training process.
It never resumes/retries, mutates source data, or acquires the GPU training lock.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .materialize import save, sha

PREFIX = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence"
ROOT = Path("/root/autodl-tmp/rachel_score_design_20260913_001")
CHECKPOINT = ROOT/"s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt"


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU diagnostic requires CUDA_VISIBLE_DEVICES='' before launch")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data, evaluation = output/"data", output/"m12"
    commands = [
        [sys.executable, "-m", PREFIX+".hard_validation.materialize",
            "--output", str(data), "--workers", str(args.workers)],
        [sys.executable, "-m", PREFIX+".evaluate_hard_validation",
            "--manifest", str(data/"manifest.json"), "--checkpoint", str(CHECKPOINT),
            "--epoch", "12", "--workers", str(args.workers),
            "--output", str(evaluation), "--execute"],
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    status = dict(schema="s7-hard-simval-cpu-pipeline/1", status="running", pid=os.getpid(),
        process_start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        completed_stages=0, active_pid=None, diagnostic_only=True, gpu_used=False,
        cpu_workers=args.workers, commands=commands, cwd=str(Path.cwd()),
        implementation_sha256=sha(__file__), training_queue_changed=False)
    started = time.monotonic()
    save(output/"status.json", status)
    try:
        for index, command in enumerate(commands):
            with (output/("stage_%d.log" % (index+1))).open("x") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                status.update(active_stage=index+1, active_pid=child.pid)
                save(output/"status.json", status)
                returncode = child.wait()
            if returncode:
                raise RuntimeError("CPU stage%d failed, exit%d" % (index+1, returncode))
            receipt = data/"manifest.json" if index == 0 else evaluation/"summary.json"
            record = json.loads(receipt.read_text())
            count = record.get("summary", {}).get("count") if index == 0 else record.get("count")
            if record.get("status") != "complete" or count != 6000:
                raise ValueError("CPU stage emitted incomplete/nonformal diagnostic population")
            status.update(completed_stages=index+1, active_pid=None, elapsed_s=time.monotonic()-started)
            save(output/"status.json", status)
        status.update(status="complete", result=str(evaluation/"summary.json"))
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        status["elapsed_s"] = time.monotonic()-started
        save(output/"status.json", status)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, choices=(1,2,3,4), default=4)
    print(json.dumps(run(p.parse_args()), indent=2))

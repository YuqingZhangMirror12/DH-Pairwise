"""Finite single-GPU C16 queue: two arms, endpoint-only held-out evaluations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCHEMA = "rachel-frozen-classifier-continuation-queue/1"
MODULE = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation."
R = Path("/root/autodl-tmp/rachel_score_design_20260913_001")
BASELINES = {"s6_d2": R / "attention_depth_20260915/s4_cross_attention_depth2/evaluation",
             "s7": R / "s6_s7_20260915/priority_after_s5/s7_augmented_full24/evaluation"}


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stage_plan(root, python=sys.executable):
    result = []
    for arm in ("s6_d2", "s7"):
        training = root / (arm + "_c16")
        result.append(dict(name=arm + "_C9_C16", kind="training",
            command=[python, "-m", MODULE + "continue_classifier", "--arm", arm,
                     "--output", str(training)], completion=str(training / "status.json")))
        for selection in ("fixed_epoch", "max_f1", "recall95"):
            for split in ("test", "real", "ood"):
                output = training / "evaluation" / selection / split
                command = [python, "-m", MODULE + "evaluate_continuation", "--training-run", str(training),
                    "--selection", selection, "--split", split, "--output", str(output),
                    "--device", "cuda:0", "--batch-size", "1", "--workers", "4",
                    "--dataset", "/root/autodl-tmp/dataset_rachel_pairwise_n512_v1",
                    "--prepared-cache", "/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared",
                    "--ood-prepared", "/root/autodl-tmp/turufan_ood_pairwise_20260912_001/prepared",
                    "--translation-gt-json", "/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json",
                    "--baseline-evaluation", str(BASELINES[arm] / selection / split)]
                if split == "real":
                    command.extend(["--keep-ids", str(R / "keep_ids.json")])
                result.append(dict(name=arm + "_" + selection + "_" + split, kind="endpoint_evaluation",
                    command=command, completion=str(output / "protocol.json")))
    return result


def main(args):
    source = Path(args.source_root).resolve(strict=True)
    root = Path(args.output_root).resolve(strict=True)
    status_path = root / "queue_status.json"
    if status_path.exists():
        raise ValueError("queue already exists; no duplicate or automatic restart")
    for arm in ("s6_d2", "s7"):
        smoke = json.loads((root / (arm + "_smoke32") / "smoke.json").read_text())
        if (smoke.get("status") != "smoke_complete" or smoke.get("pair_exposures") != 32
                or smoke.get("optimizer_updates") != 2 or smoke.get("frozen_base_unchanged") is not True
                or smoke.get("formal_training_counted") is not False):
            raise ValueError("both exact-restore discard32 smokes must pass before formal training")
    # Shared with the completed heatmap suite, not just a private queue lock.
    lock_path = root.parent / "heatmap-gpu.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if occupied:
            raise RuntimeError("GPU already has compute processes; will not overlap: " + occupied)
        started = time.time()
        status = dict(schema_version=SCHEMA, status="running", pid=os.getpid(), source_root=str(source),
            started_at_unix=started, gpu_lock=str(lock_path), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            sequence="S6-D2 C16 + final evaluations, then S7 C16 + final evaluations",
            per_epoch_held_out_evaluation=False, stages=stage_plan(root))
        save(status_path, status)
        try:
            for index, stage in enumerate(status["stages"]):
                log_path = root / ("%02d_%s.log" % (index + 1, stage["name"]))
                stage.update(status="running", started_at_unix=time.time(), log=str(log_path))
                with log_path.open("x") as log:
                    child = subprocess.Popen(stage["command"], cwd=source, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONUNBUFFERED="1"))
                    stage["pid"] = child.pid
                    status.update(active_stage=index + 1, active_name=stage["name"], active_pid=child.pid)
                    save(status_path, status)
                    code = child.wait()
                stage.update(returncode=code, finished_at_unix=time.time())
                if code:
                    stage["status"] = "failed"
                    raise RuntimeError(stage["name"] + " failed; inspect its log, no automatic retry")
                result = json.loads(Path(stage["completion"]).read_text())
                if result.get("status") != "complete":
                    raise RuntimeError(stage["name"] + " exited without completed receipt")
                if stage["kind"] == "training" and result.get("completed_segments") != 112:
                    raise RuntimeError("training did not reach its fixed C16 endpoint")
                stage["status"] = "complete"
                save(status_path, status)
            status.update(status="complete", active_pid=None, active_name=None, completed_stages=len(status["stages"]))
        except BaseException as error:
            status.update(status="failed", error=repr(error))
            raise
        finally:
            status.update(finished_at_unix=time.time(), elapsed_s=time.time() - started)
            save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    main(parser.parse_args())

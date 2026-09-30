"""One bounded frozen-geometry follow-up; never changes the training queues.

Two completed, matched checkpoints each get their own full VAL selection,
then frozen TEST and REAL evaluation. Physical defaults are the existing
registered Top2/normals/overlap/both comparison, not a new parameter sweep.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SOURCE = Path("/root/autodl-tmp/rachel_ablation_v3_source_20260907_008")
ROOT = Path("/root/autodl-tmp/rachel_ablation_v3_20260907_001")
OUTPUT = ROOT / "trained_physical_followup_20260908_001"
DATA = "/root/autodl-tmp/dataset_rachel_pairwise_n512_v1"
PREPARED = "/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared"
GT = "/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json"
ARMS = (("baseline_control", "0.9611544609069824"),
        ("coarse512", "0.16633746027946472"))


def commands():
    jobs = []
    for arm, threshold in ARMS:
        checkpoint = ROOT / "training_screen_attempt002" / arm / "winner.pt"
        out = OUTPUT / arm
        common = ["--checkpoint", str(checkpoint), "--pair-threshold", threshold,
                  "--precision", "fp32", "--seed", "260907", "--batch-size", "4"]
        for split in ("val", "test", "real"):
            module = "experiments.rachel_n512_formal_30k.run_" + (
                "real_contiguous_seam_ablation" if split == "real" else "contiguous_seam_ablation")
            cmd = [sys.executable, "-u", "-m", module] + common + ["--output", str(out / split)]
            if split != "real":
                cmd += ["--split", split, "--dataset", DATA, "--workers", "4"]
            else:
                cmd += ["--prepared-cache", PREPARED, "--translation-gt-json", GT]
            if split == "val":
                cmd += ["--physical-ablation"]
            else:
                cmd += ["--freeze", str(out / "val" / "validation_freeze.json")]
            jobs.append({"arm": arm, "split": split, "command": cmd})
    return jobs


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = commands()
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "protocol.json", {
        "schema": "matched-trained-physical-followup/1", "source_root": str(SOURCE),
        "training_modified": False, "classifier_modified": False,
        "registered_15_training_arms_unchanged": True,
        "selection_source": "each_checkpoint_own_full_validation_only",
        "selection_rule": "mean R@2/5/8/10, then assembly F1@10, then lower P90; exact tie Top2",
        "physical_configs": "existing defaults: Top2, normals, overlap, both; support_mode=sum",
        "hyperparameter_sweep": False, "seam_quality_evaluated": False,
        "test_and_real_previously_viewed": True, "jobs": jobs,
    })
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for job in jobs:
        log = OUTPUT / (job["arm"] + "_" + job["split"] + ".log")
        started = time.time()
        print(json.dumps(dict(job, event="stage_start", started_at=started)), flush=True)
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(job["command"], cwd=str(SOURCE), env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        receipt = dict(job, returncode=result.returncode, elapsed_s=time.time() - started)
        write_json(OUTPUT / (job["arm"] + "_" + job["split"] + "_receipt.json"), receipt)
        print(json.dumps(dict(receipt, event="stage_exit")), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    write_json(OUTPUT / "complete.json", {"status": "complete", "completed_stages": len(jobs)})
    print(json.dumps({"event": "followup_complete", "completed_stages": len(jobs)}), flush=True)


if __name__ == "__main__":
    main()

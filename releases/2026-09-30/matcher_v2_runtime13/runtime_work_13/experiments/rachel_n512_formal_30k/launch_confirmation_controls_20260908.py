"""Two predeclared warm-start baseline controls, each with frozen evaluation.

No candidate is chosen here. Existing screen queues and source snapshots are
untouched; later candidates must use these same fresh seeds and fixed budget.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SOURCE = Path("/root/autodl-tmp/rachel_ablation_v3_source_20260907_008")
OUTPUT = Path("/root/autodl-tmp/rachel_ablation_v3_20260907_001/confirmation_controls_20260908_001")
RUN = "/root/autodl-tmp/rachel_n512_convergence_20260901_001/run-convergence-50a8cffb0ae92614"
DATA = "/root/autodl-tmp/dataset_rachel_pairwise_n512_v1"
PREPARED = "/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared"
GT = "/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json"
SEEDS = (260908, 260909)
PACKAGE = "experiments.rachel_n512_formal_30k."


def train_command(seed):
    return [sys.executable, "-u", "-m", PACKAGE + "run_architecture_ablation",
            "--run", RUN, "--dataset", DATA,
            "--output", str(OUTPUT / ("seed%d" % seed) / "training"),
            "--variants", "baseline_control", "--seed", str(seed), "--epochs", "5",
            "--batch-size", "4", "--effective-batch-size", "16", "--workers", "4",
            "--learning-rate", "2e-5", "--log-every", "50"]


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_stage(seed_root, stage, command):
    started = time.time()
    event = {"event": "stage_start", "seed": int(seed_root.name[4:]),
             "stage": stage, "command": command, "started_at": started}
    write_json(seed_root / (stage + "_command.json"), event)
    print(json.dumps(event), flush=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (seed_root / (stage + ".log")).open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=str(SOURCE), env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    receipt = dict(event, event="stage_exit", returncode=result.returncode,
                   elapsed_s=time.time() - started)
    write_json(seed_root / (stage + "_receipt.json"), receipt)
    print(json.dumps(receipt), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"training_commands": [train_command(s) for s in SEEDS],
                          "after_each_training": "own full VAL/refinement/seam-quality, then frozen TEST and REAL"}, indent=2))
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "protocol.json", {
        "schema": "matched-confirmation-controls/1", "source_root": str(SOURCE),
        "seeds": SEEDS, "variant": "baseline_control", "candidate_selected": False,
        "initialization": "warm-start", "run": RUN, "dataset": DATA,
        "epochs": 5, "precision": "fp32", "batch_size": 4, "effective_batch_size": 16,
        "learning_rate": 2e-5, "weight_decay": 1e-4, "clip_norm": 5,
        "candidate_matching_required": True, "all_seeds_must_be_reported": True,
        "registered_15_screen_unchanged": True, "from_scratch_claim": False,
        "test_and_real_previously_viewed": True, "test_or_real_used_for_fit": False,
        "checkpoint_selection": "sqrt(sqrt(fused_AUROC*fused_AUPRC)*mean_R2_R5_R8_R10)",
        "decoder_selection": "own validation mean R2/R5/R8/R10, jointF1@10, lowerP90, exact tie Top2",
        "training_commands": [train_command(s) for s in SEEDS],
    })
    for seed in SEEDS:
        seed_root = OUTPUT / ("seed%d" % seed)
        seed_root.mkdir()
        run_stage(seed_root, "training", train_command(seed))
        arm = seed_root / "training" / "baseline_control"
        with (arm / "train_val_freeze.json").open(encoding="utf-8") as stream:
            freeze = json.load(stream)
        if freeze.get("status") != "complete" or freeze.get("test_or_real_used_for_fit") is not False:
            raise RuntimeError("training did not produce a complete validation-only winner")
        common = ["--checkpoint", str(arm / "winner.pt"), "--pair-threshold",
                  str(freeze["classifier_thresholds"]["fused"]),
                  "--precision", "fp32", "--seed", str(seed), "--batch-size", "4"]
        evaluation = seed_root / "evaluation"
        for split in ("val", "test", "real"):
            module = "run_real_contiguous_seam_ablation" if split == "real" else "run_contiguous_seam_ablation"
            command = [sys.executable, "-u", "-m", PACKAGE + module] + common
            command += ["--output", str(evaluation / split), "--save-seam-membership"]
            if split == "real":
                command += ["--prepared-cache", PREPARED, "--translation-gt-json", GT]
            else:
                command += ["--split", split, "--dataset", DATA, "--workers", "4", "--seam-quality"]
            if split == "val":
                command += ["--refinement-ablation"]
            else:
                command += ["--freeze", str(evaluation / "val" / "validation_freeze.json")]
            run_stage(seed_root, split, command)
    write_json(OUTPUT / "complete.json", {"status": "complete", "seeds": SEEDS, "completed_stages": 8})
    print(json.dumps({"event": "confirmation_controls_complete", "seeds": SEEDS}), flush=True)


if __name__ == "__main__":
    main()

"""Finite CPU preparation for all three canonical weather epochs.

Emit readiness only from measured full TRAIN24k runs, not the small probe.
The same deterministic generator is used by later GPU training. This pass
records geometry recipes and actual coverage, not model-derived labels.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from experiments.rachel_n512_formal_30k.run_joint_damage_pipeline import save, read
from staging.pairwise_v0_2.pairwise_data.rachel_guided_partial_dataset import SCHEMA_VERSION


def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "suite_state.json"
    if state_path.exists():
        raise FileExistsError("preparation exists; inspect the live handle before any recovery")
    state = dict(status="running", pid=os.getpid(), stages=[], arguments=vars(args),
        started_at=datetime.now(timezone.utc).isoformat())
    save(state_path, state)
    try:
        summaries = []
        for epoch in (1, 2, 3):
            output = root / ("full_epoch%d" % epoch)
            command = [sys.executable, "-u", "-m", "experiments.rachel_n512_formal_30k.prepare_guided_partial",
                "--dataset", args.dataset, "--train-manifest", args.train_manifest,
                "--outline-bank", args.outline_bank, "--output", str(output), "--limit", "24000",
                "--epoch", str(epoch), "--seed", "260910", "--max-attempts", "12",
                "--workers", str(args.workers)]
            stage = dict(epoch=epoch, status="running", output=str(output), command=command)
            state["stages"].append(stage)
            with (root / ("full_epoch%d.log" % epoch)).open("x") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL)
                stage["pid"] = child.pid
                save(state_path, state)
                returncode = child.wait()
            stage.update(returncode=returncode, status="complete" if returncode == 0 else "failed")
            save(state_path, state)
            if returncode:
                raise RuntimeError("full guided preparation failed at canonical epoch %d" % epoch)
            summary = read(output / "summary.json")
            if (summary.get("status") != "complete" or summary.get("selected_rows") != 24000
                    or summary.get("full_train") is not True or summary.get("probe_only") is not False
                    or summary.get("guided_topology_connectivity") != 8
                    or summary.get("canonical_weather_epoch") != epoch
                    or summary.get("hard_negative_overlay_used") is not False):
                raise ValueError("full measured TRAIN preparation is incomplete")
            for arm in ("partial_low", "partial_high"):
                count = summary["counts"][arm]
                if count["positive"] != 12000 or count["negative"] != 12000 or count["applied_positive"] != count["applied_negative"]:
                    raise ValueError("native positive/negative geometry applications are unbalanced")
            summaries.append(summary)
        total = {arm: sum(s["counts"][arm]["applied_positive"] + s["counts"][arm]["applied_negative"]
            for s in summaries) for arm in ("partial_low", "partial_high")}
        high_rate = total["partial_high"] / 72000
        low_rate = total["partial_low"] / 72000
        if high_rate < .6 or high_rate <= low_rate:
            raise ValueError("measured high coverage remains below the 60% weather-slot target: %.5f" % high_rate)
        ready = dict(status="complete", train_only=True, hard_negative_enabled=False,
            canonical_weather_epochs=[1, 2, 3], paired_low_high_verified=True,
            verification="same probability-independent generator/proposal RNG; fixed shared request coin; coupled native labels",
            full_training_rows_per_epoch=24000, canonical_row_exposures=72000,
            actual_applied=total,
            actual_weather_slot_rates=dict(partial_low=low_rate, partial_high=high_rate),
            actual_six_epoch_rates=dict(partial_low=low_rate / 2, partial_high=high_rate / 2),
            geometry_schema=SCHEMA_VERSION, topology_connectivity=8, minimum_retained_source_seam_px=32,
            minimum_original_matches=4, artificial_cut_edge_exclusion_px=8,
            input_identity=summaries[0]["input_identity"],
            summaries=[str(root / ("full_epoch%d/summary.json" % e)) for e in (1, 2, 3)],
            notes="Full geometry measured before E1 weathering. Later actual TRAIN sidecars count final use; no held-out inputs or labels used.")
        if any(s["input_identity"] != ready["input_identity"] for s in summaries):
            raise ValueError("canonical preparations used different source inputs")
        save(root / "ready.json", ready)
        state.update(status="complete", readiness=ready)
    except Exception as error:
        state.update(status="failed", error=repr(error))
        save(root / "failure.json", state)
        raise
    finally:
        save(state_path, state)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("dataset", "train-manifest", "outline-bank", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--workers", type=int, default=4)
    run(p.parse_args())

"""Finish saved-result reporting after the existing experiment queues exit.

This is CPU-only postprocessing: no matcher/head execution, retraining, model
selection, or threshold fitting. It preserves previous report snapshots.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

from experiments.rachel_n512_formal_30k.run_pairability_stage1_followup import completed


def reporting_plan(root, stage):
    root = Path(root).resolve()
    if stage not in ("stage1", "stage2"):
        raise ValueError("unknown reporting stage")
    dependencies = ([root / "stage1_followup_state.json"] if stage == "stage1" else
                    [root / "stage2_state.json", root / "stage1_reports_state.json"])
    commands = []
    if stage == "stage1":
        for split in ("test", "real"):
            commands.append(("transport_native_" + split, [sys.executable, "-m",
                "experiments.rachel_n512_formal_30k.report_native_score_probabilities",
                "--source", str(root / "heads" / ("matrix_transport_only_" + split) / split),
                "--cache", str(root / "cache" / split), "--freeze",
                str(root / "heads/matrix_transport_only/validation_freeze.json"), "--output",
                str(root / "heads" / ("matrix_transport_only_" + split + "_native") / split)]))
    snapshot = "pairability_summary_003" if stage == "stage1" else "pairability_summary_004"
    commands.append(("summary", [sys.executable, "-m",
        "experiments.rachel_n512_formal_30k.summarize_pairability_overnight", "--root", str(root),
        "--output", str(root / "reports" / snapshot)]))
    return dependencies, commands


def run(root, stage):
    root = Path(root).resolve()
    dependencies, commands = reporting_plan(root, stage)
    for dependency in dependencies:
        completed(dependency)
    rows = []
    for name, command in commands:
        row = dict(name=name, command=command, status="running", started_unix=time.time())
        rows.append(row)
        with (root / (stage + "_reports_" + name + ".log")).open("x") as stream:
            child = subprocess.Popen(command, cwd=root / "source", stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            row["pid"] = child.pid
            (root / (stage + "_reports_steps.json")).write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
            code = child.wait()
        row.update(status="complete" if code == 0 else "failed", exit_code=code,
                   ended_unix=time.time())
        (root / (stage + "_reports_steps.json")).write_text(json.dumps(rows, indent=2))
        if code:
            raise RuntimeError("reporting failed: " + name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    args = p.parse_args()
    run(args.root, args.stage)

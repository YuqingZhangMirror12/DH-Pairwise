"""Finite NEW E4v2/E5v2 queue: CPU preparation, existing GAP, smoke, fit/eval.

The queued parent imports no torch and opens no CUDA context. The existing
E0-E3/GAP processes and files are read-only predecessors; nothing is killed,
restarted, or overwritten. GPU smoke waits until GAP is completely finished.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys

from experiments.rachel_n512_formal_30k.run_all_weathering_ablations import (
    now, read_json, save_json,
)
from experiments.rachel_n512_formal_30k.run_partial_seam_pipeline import wait_for_gap


ARMS = ("e4v2", "e5v2")
PREPARATION_MODULE = "experiments.rachel_n512_formal_30k.prepare_curve_hardneg"
TRAINING_MODULE = "experiments.rachel_n512_formal_30k.train_curve_hardneg"
EVALUATION_MODULE = "experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint"
SOURCE_SHA256 = "350bf95413f828698f3396294a317569b013891d0906e2d4212d046fba059233"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(args, *, smoke=False):
    """Hash only declared TRAIN recipe artifacts, never VAL/TEST/REAL samples."""
    result = {key: str(Path(getattr(args, key)).resolve(strict=True)) for key in
              ("dataset", "train_manifest", "outline_bank", "negative_overlay")}
    result.update(train_manifest_sha256=_sha256(args.train_manifest),
                  negative_overlay_sha256=_sha256(args.negative_overlay),
                  outline_profiles_sha256=_sha256(Path(args.outline_bank) / "profiles.npz"),
                  outline_metadata_sha256=_sha256(Path(args.outline_bank) / "bank.json"))
    if smoke:
        result.update(checkpoint=str(Path(args.checkpoint).resolve(strict=True)),
                      checkpoint_sha256=_sha256(args.checkpoint))
        if result["checkpoint_sha256"] != SOURCE_SHA256:
            raise RuntimeError("both new arms require the original common 350bf9 warm start")
    return result


def require_preparation(path, identity):
    prep = read_json(path)
    if (prep.get("schema_version") != "rachel-curve-hardneg-train-preparation/1"
            or prep.get("status") != "complete" or prep.get("full_train") is not True
            or prep.get("probe_only") is not False or prep.get("epoch") != 1
            or prep.get("positive_arrays_identical") is not True):
        raise RuntimeError("full new curved/hard-negative TRAIN preparation is incomplete")
    if prep.get("input_identity") != identity:
        raise RuntimeError("preparation receipt does not match the current frozen input artifacts")
    counts = prep.get("counts", {})
    for arm in ARMS:
        count = counts.get(arm, {})
        if (count.get("rows"), count.get("positive"), count.get("negative")) != (24000, 12000, 12000):
            raise RuntimeError("preparation must cover unchanged balanced24k rows in both new arms")
    e4, e5 = counts["e4v2"], counts["e5v2"]
    if e4.get("applied_positive", 0) <= 0 or e4["applied_positive"] != e4.get("applied_negative"):
        raise RuntimeError("reference curved preparation lacks balanced actual applications")
    if e5.get("applied_positive") != e4["applied_positive"]:
        raise RuntimeError("new arms do not preserve the same actual curved positives")
    replacements = e5.get("replacements_negative")
    if type(replacements) is not int or not 0 < replacements <= 2400:
        raise RuntimeError("new hard-negative arm needs actual bounded negative replacements")
    return prep


def require_preflight(path, identity):
    receipt = read_json(path)
    if (receipt.get("schema_version") != "rachel-curve-hardneg-gpu-preflight/1"
            or receipt.get("status") != "complete" or receipt.get("basic_contract_passed") is not True
            or receipt.get("positive_arrays_identical") is not True
            or receipt.get("positive_weathered_arrays_identical") is not True
            or receipt.get("weights_saved") is not False
            or receipt.get("input_identity") != identity):
        raise RuntimeError("new bounded GPU smoke receipt is absent, unsafe, or input-mismatched")
    for arm in ARMS:
        report = receipt.get("results", {}).get(arm, {})
        if ((report.get("samples"), report.get("optimizer_updates")) != (128, 8)
                or not isinstance(report.get("mean_loss"), (int, float))
                or not math.isfinite(report["mean_loss"])):
            raise RuntimeError("new arm smoke did not meet the discarded128/8 finite-loss budget")
    return receipt


def wait_for_preparation(args):
    """Await only an explicitly supplied exact CPU preparation PID; never poll GPU."""
    pid = args.preparation_pid
    if pid is None:
        return
    if type(pid) is not int or pid <= 1:
        raise ValueError("invalid preparation PID")
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().replace("\0", " ")
    except FileNotFoundError:
        return  # Its full completion receipt is still mandatory below.
    if PREPARATION_MODULE not in command or str(Path(args.root).resolve() / "preparation") not in command or "--smoke" in command:
        raise RuntimeError("preparation PID belongs to a different process")
    subprocess.run(["tail", "--pid=" + str(pid), "--sleep-interval=10", "-f", "/dev/null"], check=True)


def stage_plan(args):
    root = Path(args.root).resolve()
    common = ["--dataset", args.dataset, "--device", "cuda:0", "--workers", str(args.workers)]
    recipe = ["--train-manifest", args.train_manifest, "--outline-bank", args.outline_bank,
              "--negative-overlay", args.negative_overlay]
    stages = [("smoke_e4v2_e5v2", PREPARATION_MODULE, common + recipe + [
        "--smoke", "--checkpoint", args.checkpoint, "--output", str(root / "preflight"),
        "--cache-dir", str(root / "smoke_weathering_cache")])]
    for arm in ARMS:
        training = root / arm / "training"
        stages.append(("train_" + arm, TRAINING_MODULE, common + recipe + [
            "--arm", arm, "--checkpoint", args.checkpoint, "--output", str(training),
            "--cache-dir", str(root / "weathering_cache"), "--log-every", "100"]))
        for split in ("test", "real"):
            params = common + ["--training-run", str(training), "--split", split,
                               "--output", str(root / arm / "evaluation" / split)]
            if split == "real":
                params += ["--prepared-cache", args.prepared_cache, "--translation-gt-json", args.translation_gt_json]
            stages.append(("evaluate_" + arm + "_" + split, EVALUATION_MODULE, params))
    return stages


def summarize(root, *, checkpoint):
    root, arms = Path(root).resolve(), {}
    for arm in ARMS:
        training = read_json(root / arm / "training/train_val_freeze.json")
        if ((training.get("status"), training.get("unique_count"), training.get("completed_global_exposures"),
             training.get("completed_optimizer_updates"), training.get("completed_validation_events"))
                != ("complete", 24000, 120000, 7500, 5)
                or training.get("original_validation_unchanged") is not True
                or training.get("test_or_real_used_for_fit") is not False
                or Path(training.get("initial_checkpoint", "")).resolve() != Path(checkpoint).resolve()):
            raise RuntimeError("incomplete or nonmatched common-warm-start curved arm")
        evaluations = {}
        for split in ("test", "real"):
            directory = root / arm / "evaluation" / split
            receipt, summary = read_json(directory / "receipt.json"), read_json(directory / "summary.json")
            if (receipt.get("status") != "complete" or summary.get("status") != "complete"
                    or summary.get("test_or_real_used_for_fit") is not False
                    or summary.get("branch_validation_thresholds") != training.get("classifier_thresholds")):
                raise RuntimeError("incomplete evaluation or changed clean-VAL thresholds")
            evaluations[split] = dict(classification=summary["classification"], layout=summary["layout"],
                                     source=str(directory / "summary.json"))
        arms[arm] = dict(training_freeze=training, evaluations=evaluations)
    result = dict(schema_version="rachel-curve-hardneg-results/1", status="complete", complete=True,
        arms=arms, selection_on_real=False, selection_on_test=False,
        comparison_controls="E4v2 vs E1 adds curved partial-data; E5v2 vs E4v2 adds only native straight hard-negative replacements",
        registered_budget_per_arm=dict(exposures=120000, optimizer_updates=7500, clean_val_events=5),
        original_e0_e3_and_gap_unchanged=True,
        caveat="REAL informed qualitative research design and repeated diagnostics; it is not a new untouched blind test")
    save_json(root / "summary.json", result)
    return result


def _run_stage(args, state, stage, state_path, source, environment):
    root = Path(args.root).resolve()
    command = [args.python, "-u", "-m", stage["module"]] + stage["arguments"]
    stage.update(status="running", started_at=now(), command=command,
                 log=str(root / (stage["name"] + ".log")))
    state.update(status="running", active_stage=stage["name"])
    save_json(state_path, state)
    with Path(stage["log"]).open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=str(source), env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        stage["pid"] = child.pid
        save_json(state_path, state)
        code = child.wait()
    stage.update(status="complete" if code == 0 else "failed", returncode=code, completed_at=now())
    save_json(state_path, state)
    if code:
        raise RuntimeError(stage["name"] + " failed; see " + stage["log"])


def run(args):
    root, gap_root = Path(args.root).resolve(), Path(args.gap_root).resolve()
    dataset_root = Path(args.dataset).resolve()
    if root == gap_root or gap_root in root.parents or root in gap_root.parents:
        raise ValueError("NEW output root must be separate from the original GAP tree")
    if root == dataset_root or dataset_root in root.parents:
        raise ValueError("NEW output root must not modify the original dataset release")
    if type(args.workers) is not int or args.workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pipeline_state.json"
    if path.exists():
        raise FileExistsError("new curved pipeline already exists; do not duplicate or restart it")
    source = Path(__file__).resolve().parents[2]
    state = dict(schema_version="rachel-curve-hardneg-pipeline/1", status="waiting_for_preparation",
        pid=os.getpid(), started_at=now(), source=str(source), arguments=vars(args),
        active_stage="waiting_for_preparation", predecessor=str(gap_root / "pipeline_state.json"),
        preparation_summary=str(root / "preparation/summary.json"),
        stages=[dict(name=name, module=module, arguments=params, status="queued")
                for name, module, params in stage_plan(args)],
        registered_budget_per_arm=dict(exposures=120000, optimizer_updates=7500, clean_val_events=5),
        smoke_discarded_exposures_per_arm=128, smoke_weights_reused=False,
        original_files_modified=False, original_e0_e3_and_gap_unchanged=True,
        test_or_real_used_for_fit=False, clean_val_only_model_and_threshold_selection=True,
        positive_augmented_inputs_shared_between_arms=True)
    save_json(path, state)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(source), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        wait_for_preparation(args)
        identity = input_identity(args)
        require_preparation(root / "preparation/summary.json", identity)
        state.update(status="waiting_for_gap", active_stage="waiting_for_gap", preparation_completed_at=now(),
                     input_identity=identity)
        save_json(path, state)
        predecessor = wait_for_gap(args.gap_root)  # Preserve the original exact-PID/receipt gate.
        state.update(gap_completed_at=predecessor.get("completed_at"), gap_receipt_checked_at=now())
        save_json(path, state)
        # Recheck recipe bytes after a possibly long predecessor wait.
        smoke_identity = input_identity(args, smoke=True)
        require_preparation(root / "preparation/summary.json", {key: value for key, value in smoke_identity.items()
            if key not in ("checkpoint", "checkpoint_sha256")})
        preflight = root / "preflight/preflight.json"
        for stage in state["stages"]:
            if stage["name"] == "smoke_e4v2_e5v2" and preflight.exists():
                require_preflight(preflight, smoke_identity)
                stage.update(status="complete", returncode=0, reused_completed_receipt=str(preflight),
                             receipt_sha256=_sha256(preflight), completed_at=now())
                save_json(path, state)
                continue
            if stage["name"] == "smoke_e4v2_e5v2" and preflight.parent.exists():
                raise RuntimeError("partial prior smoke output exists without receipt; refusing to overwrite or rerun")
            if stage["name"].startswith("train_"):
                if input_identity(args, smoke=True) != smoke_identity:
                    raise RuntimeError("frozen preparation artifacts or common checkpoint changed before training")
                require_preflight(preflight, smoke_identity)
            _run_stage(args, state, stage, path, source, environment)
            if stage["name"] == "smoke_e4v2_e5v2":
                require_preflight(preflight, smoke_identity)
        summarize(root, checkpoint=args.checkpoint)
        state.update(status="complete", active_stage=None, completed_at=now())
        save_json(path, state)
    except Exception as error:
        active = next((stage for stage in state["stages"] if stage["name"] == state.get("active_stage")), None)
        if active and active.get("status") == "running":
            active.update(status="failed", error=repr(error), failed_at=now())
        state.update(status="failed", failed_at=now(), error=repr(error))
        save_json(path, state)
        raise
    return state


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "dataset", "train-manifest", "outline-bank", "negative-overlay"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--gap-root", "--gaproot", dest="gap_root", required=True)
    p.add_argument("--checkpoint", "--warmstart", dest="checkpoint", required=True)
    p.add_argument("--prepared-cache", "--realprep", dest="prepared_cache", required=True)
    p.add_argument("--translation-gt-json", "--realgt", dest="translation_gt_json", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--preparation-pid", "--prepare-pid", dest="preparation_pid", type=int,
                   help="exact active CPU prepare_curve_hardneg PID to await before full receipt checks")
    return p


def launch(args):
    # A single finite queue; no daemon, retry scheduler, training launch on import,
    # or attempt to cancel the unrelated old axis-recipe continuation.
    return run(args)


if __name__ == "__main__":
    launch(parser().parse_args())

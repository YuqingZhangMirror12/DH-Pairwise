"""Sequential val -> frozen test -> prepared-cache real evaluation of completed arms.

Default: one scan. --follow repeats while a supplied training PID is actually
alive or newly completed arms remain runnable. Training status files never
prove process liveness. This is an experiment dependency pipeline, not a
notification service or scheduled automation. It imports no model/GPU code.

Each stage gets one exclusive attempt directory under ARM/_pipeline/STAGE,
with command.json, stdout.log, stderr.log, and result.json (exit_code). Actual
runner outputs live at ARM/val, ARM/test, ARM/real. Existing completed results
are reused; existing partial/failed attempts require manual attention and are
NEVER deleted, overwritten, or automatically retried by this entry point.

--seam-quality requests synthetic val/test seam metrics and rejects reuse of
older results without them. Real data has no target seam GT, so this flag is
never sent to its runner. --save-seam-membership is forwarded to all stages.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


SEAM_MODULE = "experiments.rachel_n512_formal_30k.run_contiguous_seam_ablation"
REAL_MODULE = "experiments.rachel_n512_formal_30k.run_real_contiguous_seam_ablation"
DEFAULT_PREPARED_CACHE = "/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared"
STAGES = ("val", "test", "real")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        return
    # Only pipeline bookkeeping uses replacement, never runner results/logs.
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def output_key(training_root, arm_name):
    root = Path(training_root).resolve()
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", root.name + "--" + arm_name)
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:10]
    return readable + "--" + digest


@dataclass(frozen=True)
class CompletedArm:
    training_root: Path
    directory: Path
    threshold: float

    @property
    def checkpoint(self):
        return self.directory / "winner.pt"

    @property
    def key(self):
        return output_key(self.training_root, self.directory.name)


def discover_arms(training_roots):
    """Read one directory level; provisional freezes never become eligible."""
    eligible, waiting, issues = [], [], []
    for root in sorted({Path(p).resolve() for p in training_roots}):
        if not root.is_dir():
            waiting.append(dict(training_root=str(root), reason="training_root_not_present"))
            continue
        for freeze_path in sorted(root.glob("*/train_val_freeze.json")):
            arm = freeze_path.parent
            identity = dict(training_root=str(root), arm=arm.name, key=output_key(root, arm.name))
            try:
                freeze = read_json(freeze_path)
                if freeze.get("status") != "complete":
                    waiting.append(dict(identity, reason="training_not_complete", status=freeze.get("status")))
                    continue
                if not (arm / "winner.pt").is_file():
                    waiting.append(dict(identity, reason="completed_freeze_without_winner_file"))
                    continue
                if freeze.get("test_or_real_used_for_fit") is not False:
                    raise ValueError("training freeze must explicitly exclude test/real fitting")
                threshold = freeze["classifier_thresholds"]["fused"]
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                    raise ValueError("frozen fused threshold must be numeric")
                if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                    raise ValueError("frozen fused threshold must be finite in [0,1]")
                eligible.append(CompletedArm(root, arm, float(threshold)))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append(dict(identity, status="needs_attention", reason=str(exc)))
    return eligible, waiting, issues


def pid_is_alive(pid):
    """Actual OS process existence; Linux zombies do not count as active."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno != errno.EPERM:
            raise
    stat = Path("/proc") / str(pid) / "stat"
    try:
        # comm may itself contain spaces/parentheses, so split after the last ).
        state = stat.read_text().rsplit(")", 1)[1].strip().split()[0]
        if state in {"Z", "X", "x"}:
            return False
    except FileNotFoundError:
        if Path("/proc").is_dir():
            return False
    except (OSError, IndexError):
        pass  # Non-Linux hosts retain the OS kill(pid, 0) result.
    return True


def formal_validation_freeze(arm, destination, args):
    freeze = read_json(destination / "val" / "validation_freeze.json")
    required = dict(schema_version="contiguous-seam-ablation/1", source_split="validation",
                    sample_count=3000, probe_only=False, test_or_real_used_for_fit=False)
    if any(freeze.get(key) != value for key, value in required.items()):
        raise ValueError("validation freeze is not a formal 3000-pair validation-only freeze")
    if freeze.get("original_fused_threshold") != arm.threshold:
        raise ValueError("geometry freeze differs from the training-frozen fused threshold")
    if freeze.get("precision") != args.precision:
        raise ValueError("geometry freeze differs from requested precision")
    if not re.fullmatch(r"[0-9a-f]{64}", str(freeze.get("checkpoint_sha256", ""))):
        raise ValueError("geometry freeze lacks a valid checkpoint SHA-256 identity")
    if freeze.get("checkpoint_path") and Path(freeze["checkpoint_path"]).resolve() != arm.checkpoint.resolve():
        raise ValueError("geometry freeze references a different checkpoint path")
    decoders = freeze.get("decoders", {})
    if not decoders or freeze.get("selected_full_decoder") not in decoders:
        raise ValueError("geometry freeze lacks its selected decoder")
    if args.refinement_ablation and not all(
        "full_contiguous_seam_gap%d_mode_refine" % gap in decoders for gap in (20, 40)
    ):
        raise ValueError("existing validation freeze lacks requested refinement ablation")
    if args.seam_quality and freeze.get("seam_quality_evaluated") is not True:
        raise ValueError("validation freeze lacks requested seam quality evaluation")
    return freeze


def validate_requested_metrics(summary, stage, args):
    """Use one optional-metric contract for completed children and cached reuse."""
    if args.seam_quality and stage in {"val", "test"}:
        if summary.get("seam_quality_evaluated") is not True:
            raise ValueError("synthetic summary lacks requested seam quality evaluation")
        if not isinstance(summary.get("seam_quality"), dict) or not summary["seam_quality"]:
            raise ValueError("synthetic summary lacks requested seam quality metrics")


def stage_state(arm, destination, stage, args):
    """Classify a stage as complete, ready, or blocked; never repair its files."""
    result_dir = destination / stage
    attempt_dir = destination / "_pipeline" / stage
    try:
        if attempt_dir.exists():
            receipt_path = attempt_dir / "result.json"
            if not receipt_path.is_file():
                return "needs_attention", "existing unfinished stage attempt"
            receipt = read_json(receipt_path)
            if receipt.get("exit_code") != 0 or receipt.get("status") != "complete":
                return "needs_attention", "existing failed stage attempt"
        summary_path = result_dir / "summary.json"
        if summary_path.is_file():
            summary = read_json(summary_path)
            if (summary.get("status") != "complete" or summary.get("probe_only") is True
                    or summary.get("sample_count") != (1016 if stage == "real" else 3000)
                    or summary.get("split") != stage):
                return "needs_attention", "existing summary is incomplete or has a different population"
            if summary.get("precision") != args.precision:
                return "needs_attention", "existing summary differs in precision"
            validate_requested_metrics(summary, stage, args)
            freeze = formal_validation_freeze(arm, destination, args)
            if summary.get("selected_full_decoder") != freeze["selected_full_decoder"]:
                return "needs_attention", "summary differs from validation-selected decoder"
            if summary.get("checkpoint_sha256") != freeze.get("checkpoint_sha256"):
                return "needs_attention", "summary differs from validation checkpoint identity"
            return "complete", "existing complete result reused"
        if result_dir.exists() or attempt_dir.exists():
            return "needs_attention", "existing partial output or attempt; automatic retry disabled"
        return "ready", "not started"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return "needs_attention", str(exc)


def stage_command(arm, destination, stage, args):
    command = [args.python_executable, "-m", REAL_MODULE if stage == "real" else SEAM_MODULE,
               "--checkpoint", str(arm.checkpoint), "--pair-threshold", repr(arm.threshold),
               "--output", str(destination / stage), "--precision", args.precision,
               "--seed", str(args.seed), "--device", args.device,
               "--batch-size", str(args.real_batch_size if stage == "real" else args.batch_size)]
    if stage == "real":
        command += ["--prepared-cache", str(args.prepared_cache)]
        if args.translation_gt_json:
            command += ["--translation-gt-json", str(args.translation_gt_json)]
    else:
        command += ["--split", stage, "--workers", str(args.workers)]
        if args.dataset:
            command += ["--dataset", str(args.dataset)]
        if args.seam_quality:
            command += ["--seam-quality"]
    if args.save_seam_membership:
        command += ["--save-seam-membership"]
    if stage != "val":
        command += ["--freeze", str(destination / "val" / "validation_freeze.json")]
    elif args.refinement_ablation:
        command += ["--refinement-ablation"]
    return command


def execute_stage(arm, destination, stage, args, run_process):
    attempt_dir = destination / "_pipeline" / stage
    attempt_dir.mkdir(parents=True, exist_ok=False)  # Exclusive one-attempt claim.
    command = stage_command(arm, destination, stage, args)
    started = now()
    write_json(attempt_dir / "command.json", dict(command=command, cwd=str(args.source_root),
               started_at=started, stage=stage, checkpoint=str(arm.checkpoint)), exclusive=True)
    exit_code, error = None, None
    with (attempt_dir / "stdout.log").open("x", encoding="utf-8") as stdout, \
            (attempt_dir / "stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            completed = run_process(command, cwd=str(args.source_root), stdout=stdout, stderr=stderr, check=False)
            exit_code = int(completed.returncode)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
            stderr.write(error + "\n")
    # Temporarily omit the attempt from completion checks: its final receipt
    # does not exist yet. Validate the child result before marking it complete.
    status, reason = "needs_attention", error or "subprocess failed"
    if exit_code == 0:
        try:
            summary = read_json(destination / stage / "summary.json")
            if (summary.get("status") != "complete" or summary.get("sample_count") != (1016 if stage == "real" else 3000)
                    or summary.get("split") != stage or summary.get("probe_only") is True):
                raise ValueError("subprocess exited zero without a formal complete summary")
            validate_requested_metrics(summary, stage, args)
            freeze = formal_validation_freeze(arm, destination, args)
            if (summary.get("checkpoint_sha256") != freeze.get("checkpoint_sha256")
                    or summary.get("precision") != args.precision
                    or summary.get("selected_full_decoder") != freeze["selected_full_decoder"]):
                raise ValueError("completed summary disagrees with formal validation freeze")
            status, reason = "complete", "subprocess and formal result completed"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reason = str(exc)
    receipt = dict(status=status, reason=reason, exit_code=exit_code, started_at=started, finished_at=now())
    write_json(attempt_dir / "result.json", receipt, exclusive=True)
    return status, reason


def evaluate_arm(arm, args, run_process=subprocess.run):
    destination = args.output_root / arm.key
    destination.mkdir(parents=True, exist_ok=True)
    result = dict(training_root=str(arm.training_root), arm=arm.directory.name, key=arm.key,
                  checkpoint=str(arm.checkpoint), pair_threshold=arm.threshold, stages={})
    for stage in STAGES:
        status, reason = stage_state(arm, destination, stage, args)
        if status == "ready":
            try:
                status, reason = execute_stage(arm, destination, stage, args, run_process)
            except (OSError, ValueError) as exc:
                status, reason = "needs_attention", str(exc)
        result["stages"][stage] = dict(status=status, reason=reason)
        if status != "complete":
            result["status"] = "needs_attention"
            break
    else:
        result["status"] = "complete"
    write_json(destination / "pipeline_status.json", dict(result, updated_at=now()))
    return result


def pending_eligible(args):
    arms, _, _ = discover_arms(args.training_root)
    pending = []
    for arm in arms:
        destination = args.output_root / arm.key
        for stage in STAGES:
            status, _ = stage_state(arm, destination, stage, args)
            if status == "ready":
                pending.append(arm.key)
                break
            if status == "needs_attention":
                break
    return pending


def run(args, *, run_process=subprocess.run, alive=pid_is_alive, sleep=time.sleep):
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.source_root.is_dir():
        raise ValueError("--source-root must be the existing child-process working directory")
    previous = None
    while True:
        arms, waiting, issues = discover_arms(args.training_root)
        results = [evaluate_arm(arm, args, run_process) for arm in arms]
        live = [pid for pid in args.training_pid if alive(pid)]
        pending = pending_eligible(args) if args.follow else []
        snapshot = dict(updated_at=now(), follow=args.follow, live_training_pids=live,
                        pending_eligible=pending, waiting_for_training=waiting,
                        discovery_needs_attention=issues, arms=results)
        snapshot["needs_attention_count"] = len(issues) + sum(row["status"] == "needs_attention" for row in results)
        snapshot["status"] = ("single_pass_complete" if not args.follow else
                              "finished" if not live and not pending else "following")
        write_json(args.output_root / "pipeline_snapshot.json", snapshot)
        event = {key: snapshot[key] for key in ("status", "live_training_pids", "pending_eligible", "needs_attention_count")}
        event["complete_arm_count"] = sum(row["status"] == "complete" for row in results)
        if event != previous:
            print(json.dumps(event), flush=True)
            previous = event
        if not args.follow or (not live and not pending):
            return snapshot
        if not pending:
            sleep(args.poll_seconds)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--refinement-ablation", action="store_true")
    parser.add_argument("--seam-quality", action="store_true", help="Require synthetic val/test seam quality metrics; never request real seam GT")
    parser.add_argument("--save-seam-membership", action="store_true", help="Save decoder memberships for val/test/real")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--prepared-cache", type=Path, default=Path(DEFAULT_PREPARED_CACHE))
    parser.add_argument("--translation-gt-json", type=Path)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--real-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    if min(args.batch_size, args.real_batch_size) <= 0 or args.workers < 0:
        parser.error("batch sizes must be positive and workers nonnegative")
    if any(pid <= 0 for pid in args.training_pid):
        parser.error("--training-pid must be a positive process id")
    args.training_root = [path.resolve() for path in args.training_root]
    args.training_pid = sorted(set(args.training_pid))
    for name in ("output_root", "source_root", "dataset", "prepared_cache", "translation_gt_json"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


if __name__ == "__main__":
    result = run(parse_args())
    raise SystemExit(2 if result["needs_attention_count"] else 0)

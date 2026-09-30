"""Sealed, finite follow-up after the exact registered C16 queue exits.

Code only until --seal then --execute are explicitly invoked. No GPU lock is
held here; each training/evaluation child owns its existing nonblocking lock.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import run_continuation_queue as prior
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local import queue as candidate
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation_depth_controls import continue_depth_controls as depth

SCHEMA = "rachel-finite-followup-supervisor/1"
MODULE = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919."
SELF = MODULE + "continuation_depth_controls.followup_supervisor"
DEPENDENCY_PID, DEPENDENCY_STARTTICKS = 99017, 932052031
DEPENDENCY_ROOT = prior.R / "scorer_diagnosis_20260919/continuation_v1"
DEPENDENCY_SOURCE = prior.R / "scorer_diagnosis_20260919/source"
BASELINES = {
    "s4_d1": prior.R / "new_s345_20260914/s4_cross_attention/evaluation",
    "s6_d4": prior.R / "attention_depth_20260915/s4_cross_attention_depth4/evaluation",
}


def save(path, value):
    prior.save(Path(path), value)


def source_hashes(root):
    """Seal repository code, never datasets, artifacts or checkpoint bytes."""
    root = Path(root).resolve(strict=True)
    files = set()
    for relative in ("experiments/rachel_n512_formal_30k", "staging/pairwise_v0_2"):
        directory = root / relative
        if not directory.is_dir():
            raise ValueError("snapshot missing required source tree: " + relative)
        files.update(directory.rglob("*.py"))
    for relative in ("experiments/__init__.py", "staging/__init__.py"):
        path = root / relative
        if path.exists():
            files.add(path)
    if any(p.is_symlink() for p in files):
        raise ValueError("sealed source files must not be symlinks")
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def verify_source(plan):
    if source_hashes(plan["source_root"]) != plan["source_sha256"]:
        raise ValueError("sealed next_source changed; no dispatch/retry")


def process_identity(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(state=fields[0], startticks=int(fields[19]))


def dependency_state(receipt, process):
    if process is not None and process["startticks"] != DEPENDENCY_STARTTICKS:
        raise RuntimeError("dependency PID reused; refusing another process identity")
    if (receipt.get("pid") != DEPENDENCY_PID or receipt.get("schema_version") != prior.SCHEMA
            or Path(receipt.get("source_root", "")).resolve() != DEPENDENCY_SOURCE):
        raise RuntimeError("dependency queue identity differs")
    status = receipt.get("status")
    if status not in ("running", "complete"):
        raise RuntimeError("dependency failed/interrupted/unknown: " + str(status))
    if status == "complete":
        stages = receipt.get("stages", [])
        if receipt.get("completed_stages") != 20 or len(stages) != 20 or any(s.get("status") != "complete" for s in stages):
            raise RuntimeError("dependency complete receipt has incomplete 20-stage ledger")
    terminated = process is None or process["state"] == "Z"
    if terminated and status != "complete":
        raise RuntimeError("dependency process missing/terminated without complete receipt")
    return "ready" if terminated and status == "complete" else "waiting"


def stage_plan(root, python):
    root = Path(root)
    cr, dr = root / "candidate_local", root / "depth_controls"
    rows = []
    for arm in ("c1", "c2"):
        output = cr / (arm + "_smoke32")
        rows.append(dict(name=arm + "_discard32", kind="candidate_smoke",
            command=[python, "-m", MODULE + "candidate_local.train", "--arm", arm,
                     "--smoke", "32", "--output", str(output)], completion=str(output / "smoke.json")))
    rows.append(dict(name="candidate_C1_C2_queue", kind="candidate_queue",
        command=[python, "-m", SELF, "--run-candidate-queue", "--plan", str(root / "launch_plan.json")],
        completion=str(cr / "queue_status.json")))
    for arm in depth.ARMS:
        output = dr / (arm + "_smoke32")
        rows.append(dict(name=arm + "_discard32", kind="depth_smoke", arm=arm,
            command=[python, "-m", MODULE + "continuation_depth_controls.continue_depth_controls",
                     "--arm", arm, "--execute", "--smoke", "32", "--output", str(output)],
            completion=str(output / "smoke.json")))
    template = prior.stage_plan(dr, python)[:10]
    for arm in depth.ARMS:
        training = dr / (arm + "_c16")
        rows.append(dict(name=arm + "_C9_C16", kind="depth_training", arm=arm,
            command=[python, "-m", MODULE + "continuation_depth_controls.continue_depth_controls",
                     "--arm", arm, "--execute", "--output", str(training)],
            completion=str(training / "status.json")))
        for old in template[1:]:
            command = [s.replace(str(dr / "s6_d2_c16"), str(training)) for s in old["command"]]
            command[2] = MODULE + "continuation_depth_controls.evaluate_depth_controls"
            index = command.index("--baseline-evaluation") + 1
            command[index] = command[index].replace(str(prior.BASELINES["s6_d2"]), str(BASELINES[arm]))
            rows.append(dict(name=old["name"].replace("s6_d2", arm), kind="endpoint_evaluation",
                command=command, completion=old["completion"].replace(str(dr / "s6_d2_c16"), str(training))))
    return rows


def validate_depth_smokes(root):
    for arm in depth.ARMS:
        output = Path(root) / (arm + "_smoke32")
        receipt = json.loads((output / "smoke.json").read_text())
        expected = dict(status="smoke_complete", pair_exposures=32, optimizer_updates=2,
            formal_training_counted=False, weights_discarded=True, no_checkpoint_written=True,
            frozen_base_unchanged=True)
        if any(receipt.get(k) != v for k, v in expected.items()) or list(output.glob("*.pt")):
            raise ValueError("both depth discard32 smoke receipts required")
        protocol = json.loads((output / "protocol.json").read_text())
        identity = protocol.get("continuation_identity", {})
        depth.validate_identity(identity)
        if identity["arm"] != arm or identity["implementation_sha256"] != depth.old._sha256(depth.__file__):
            raise ValueError("depth smoke source/code binding differs")
        updates = json.loads((output / "first_updates.json").read_text())["updates"]
        if ([u.get("head_adam_steps") for u in updates] != [[12001], [12002]]
                or any(u.get("physical_microbatch") != 16 or u.get("effective_batch") != 16
                       or u.get("matcher_frozen") is not True for u in updates)):
            raise ValueError("depth smoke Adam/batch/frozen-Matcher launch evidence differs")


def validate_completion(stage):
    receipt = json.loads(Path(stage["completion"]).read_text())
    expected = "smoke_complete" if stage["kind"].endswith("smoke") else "complete"
    if receipt.get("status") != expected:
        raise RuntimeError("child did not write its complete receipt: " + stage["name"])
    if stage["kind"] == "candidate_queue" and receipt.get("completed_stages") != 20:
        raise RuntimeError("candidate queue did not complete20 stages")
    if stage["kind"] == "depth_training" and any(receipt.get(k) != v for k, v in dict(
            completed_segments=112, additional_pair_exposures=192000, additional_optimizer_updates=12000).items()):
        raise RuntimeError("depth training did not reach fixed C16 budget")


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != SCHEMA or not 0 < plan.get("deadline_hours", 0) <= 48:
        raise ValueError("invalid sealed supervisor plan/deadline")
    if (plan.get("dependency_pid"), plan.get("dependency_startticks")) != (DEPENDENCY_PID, DEPENDENCY_STARTTICKS):
        raise ValueError("dependency identity not registered")
    if Path(plan["source_root"]).resolve() != Path(__file__).resolve().parents[4]:
        raise ValueError("execute must import the sealed next_source, not live source")
    if (Path(plan["dependency_status"]) != DEPENDENCY_ROOT / "queue_status.json"
            or Path(plan["output_root"]).resolve() != Path(path).resolve().parent
            or plan["python"] != sys.executable):
        raise ValueError("sealed dependency/output/interpreter binding differs")
    if plan.get("stages") != stage_plan(plan["output_root"], plan["python"]):
        raise ValueError("sealed stage plan changed")
    verify_source(plan)
    return plan


def seal(args):
    source = Path(args.source_root).resolve(strict=True)
    if source == DEPENDENCY_SOURCE or source != Path(__file__).resolve().parents[4]:
        raise ValueError("invoke the supervisor from its independent next_source snapshot")
    if not 0 < args.deadline_hours <= 48:
        raise ValueError("deadline must be greater than0 and at most48 hours")
    root = Path(args.output_root).resolve()
    if root == DEPENDENCY_ROOT or DEPENDENCY_ROOT in root.parents or source == root or source in root.parents:
        raise ValueError("new output must be outside live outputs and next_source")
    hashes = source_hashes(source)
    root.mkdir(parents=True, exist_ok=False)
    plan = dict(schema_version=SCHEMA, source_root=str(source), output_root=str(root),
        python=sys.executable, source_sha256=hashes, dependency_pid=DEPENDENCY_PID,
        dependency_startticks=DEPENDENCY_STARTTICKS, dependency_status=str(DEPENDENCY_ROOT / "queue_status.json"),
        deadline_hours=args.deadline_hours, stages=stage_plan(root, sys.executable))
    save(root / "launch_plan.json", plan)
    return dict(status="sealed_not_started", plan=str(root / "launch_plan.json"), stages=len(plan["stages"]))


def run_candidate_queue(plan_path):
    """Use candidate.queue bytecode; guard its20 nested Popen dispatches too."""
    plan = load_plan(plan_path)
    def guarded_popen(*args, **kwargs):
        verify_source(plan)
        return subprocess.Popen(*args, **kwargs)
    proxy = SimpleNamespace(Popen=guarded_popen, STDOUT=subprocess.STDOUT)
    run = depth.private_function(candidate.run, subprocess=proxy)
    return run(SimpleNamespace(source_root=plan["source_root"],
        output_root=str(Path(plan["output_root"]) / "candidate_local")))


def execute(plan_path):
    plan = load_plan(plan_path)
    root = Path(plan["output_root"])
    status_path = root / "supervisor_status.json"
    with (root / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            raise ValueError("supervisor already started; no resume/retry/overwrite")
        started = time.time()
        deadline = started + plan["deadline_hours"] * 3600
        status = dict(schema_version=SCHEMA, status="waiting_dependency", pid=os.getpid(),
            started_at_unix=started, dependency_deadline_unix=deadline,
            deadline_scope="predecessor wait only; never interrupts dispatched fixed-budget training",
            completed_stages=0, stages=plan["stages"])
        def transition(**fields):
            status.update(fields)
            save(status_path, status)
            print(json.dumps({k: status.get(k) for k in ("status", "active_name", "completed_stages", "error")}), flush=True)
        def check_deadline():
            if time.time() >= deadline:
                raise TimeoutError("predecessor wait deadline reached (maximum48h)")
        transition()
        child = None
        child_group = None
        try:
            while True:
                check_deadline()
                receipt = json.loads(Path(plan["dependency_status"]).read_text())
                if dependency_state(receipt, process_identity(DEPENDENCY_PID)) == "ready":
                    break
                time.sleep(min(30, max(.01, deadline - time.time())))  # intentionally silent
            for directory in (root / "candidate_local", root / "depth_controls"):
                directory.mkdir(exist_ok=False)
            transition(status="dependency_complete")
            for index, stage in enumerate(status["stages"]):
                verify_source(plan)
                if Path(stage["completion"]).exists():
                    raise ValueError("existing child completion will not be overwritten")
                if stage["kind"] == "candidate_queue":
                    candidate.validate_smokes(root / "candidate_local")
                if stage["kind"] == "depth_training":
                    validate_depth_smokes(root / "depth_controls")
                log_path = root / ("%02d_%s.log" % (index + 1, stage["name"]))
                with log_path.open("x") as log:
                    child = subprocess.Popen(stage["command"], cwd=plan["source_root"],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1"))
                    child_group = child.pid
                    stage.update(status="running", pid=child.pid, log=str(log_path), started_at_unix=time.time())
                    transition(status="running", active_name=stage["name"], active_stage=index + 1, active_pid=child.pid)
                    # Stage epochs/sequence are already finite. A dependency
                    # wait deadline must never truncate normal training.
                    stage["returncode"] = child.wait()
                    child = None
                if stage["returncode"] != 0:
                    raise RuntimeError(stage["name"] + " failed; no automatic retry")
                validate_completion(stage)
                child_group = None
                stage.update(status="complete", finished_at_unix=time.time())
                transition(completed_stages=index + 1, active_pid=None)
            transition(status="complete", active_name=None, finished_at_unix=time.time())
            return status
        except BaseException as error:
            if child_group is not None:
                # Only the process group created by this supervisor is stopped.
                try:
                    os.killpg(child_group, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if child is not None:
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child_group, signal.SIGKILL)
                        child.wait(timeout=10)
            transition(status="failed", error=repr(error), active_pid=None, finished_at_unix=time.time())
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    group = result.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--run-candidate-queue", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--source-root")
    result.add_argument("--output-root")
    result.add_argument("--plan")
    result.add_argument("--deadline-hours", type=float, default=48)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if args.seal:
        print(json.dumps(seal(args)))
    elif args.run_candidate_queue:
        run_candidate_queue(args.plan)
    else:
        def interrupted(signum, _frame):
            raise KeyboardInterrupt("supervisor received signal " + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        execute(args.plan)

"""Finite S7 Matcher/Scorer follow-up AFTER the existing priority queue exits.

No handoff, signals, restarts or GPU locking in this supervisor. Every child
keeps its own original lock. A failed dependency/stage stops this new queue;
an observation failure never authorizes restarting or terminating a process.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCHEMA = "s7-matcher-after-priority/1"
PRIORITY_SCHEMA = "matched-only-explicit-priority-handoff/1"
DIAGNOSIS = Path("/root/autodl-tmp/rachel_score_design_20260913_001/scorer_diagnosis_20260919")
PREDECESSOR = DIAGNOSIS/"priority_s7_matched_g_v1"
PREDECESSOR_SHA = "d6cc98be4ed46911eafd0de274c1655a77859061979b0338b1c3585eef9ecc4d"
PREDECESSOR_PID, PREDECESSOR_STARTTICKS = 126580, 935304609
TRANSACTION = "116ddabd-667f-413f-8f17-1d1c10e9923b"
REPO = Path(__file__).resolve().parents[4]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_subset(actual, expected, context="receipt"):
    if not isinstance(actual, dict):
        raise ValueError(context+" is not an object")
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(context+" missing "+key)
        if isinstance(value, dict):
            require_subset(actual[key], value, context+"."+key)
        elif actual[key] != value:
            raise ValueError(context+" differs at "+key)


def process_identity(pid):
    path = Path("/proc")/str(pid)
    try:
        fields = (path/"stat").read_text().rsplit(")", 1)[1].split()
        argv = [x.decode() for x in (path/"cmdline").read_bytes().split(b"\0") if x]
    except FileNotFoundError:
        return None
    return dict(pid=pid, state=fields[0], startticks=int(fields[19]), argv=argv)


def dependency_state(receipt, process, spec):
    """Ready means exact complete ledger AND observed process termination."""
    owner = spec["process"]
    require_subset(receipt, {spec["schema_field"]: spec["schema"], "pid": owner["pid"],
        "transaction_id": spec["transaction_id"]}, "predecessor")
    if process is not None:
        if (process.get("pid") != owner["pid"] or process.get("startticks") != owner["startticks"]):
            raise RuntimeError("predecessor PID reused/identity differs")
        if process["state"] not in ("Z", "X") and process.get("argv") != owner["argv"]:
            raise RuntimeError("predecessor command differs")
        if process["state"] in ("T", "t"):
            raise RuntimeError("predecessor stopped; no automatic signals/recovery")
    status = receipt.get("status")
    if status not in ("waiting_current_queue", "running", "complete"):
        raise RuntimeError("predecessor failed/interrupted/unknown: "+str(status))
    stages, expected = receipt.get("stages", []), spec["stage_ledger"]
    if len(stages) != len(expected) or len(stages) != spec["completed_stages"]:
        raise ValueError("predecessor finite stage population changed")
    for actual, registered in zip(stages, expected):
        require_subset(actual, registered, "predecessor stage")
    if status == "complete":
        if (receipt.get("completed_stages") != len(expected) or receipt.get("active_pid") is not None
                or receipt.get("active_name") is not None
                or any(s.get("status") != "complete" or s.get("returncode") != 0 for s in stages)):
            raise ValueError("predecessor complete receipt has unfinished stages/active child")
    terminal = process is None or process["state"] in ("Z", "X")
    if terminal and status != "complete":
        raise RuntimeError("predecessor exited without complete ledger")
    return "ready" if terminal and status == "complete" else "waiting"


def source_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.py")) if p.is_file()}


def verify_sources(plan):
    if source_hashes(plan["source_root"]) != plan["source_sha256"]:
        raise ValueError("new sealed source changed")
    for name, expected in plan["predecessor_plan_sha256"].items():
        if digest(name) != expected:
            raise ValueError("original priority plan changed")


def seal(args):
    from .followup_plan import stage_plan, required_receipts
    source, output = Path(args.source_root).resolve(strict=True), Path(args.output_root).resolve()
    if source != REPO or source == output or source in output.parents or output in source.parents:
        raise ValueError("invoke from independent snapshot, with separate output")
    if not 0 < args.deadline_hours <= 48:
        raise ValueError("predecessor-only wait deadline must be within48h")
    for name in ("launch_plan.json", "executed_plan.json"):
        if digest(PREDECESSOR/name) != PREDECESSOR_SHA:
            raise ValueError("existing priority registration differs")
    original = read(PREDECESSOR/"executed_plan.json")
    receipt = read(PREDECESSOR/"status.json")
    process = process_identity(PREDECESSOR_PID)
    if process is None or process["startticks"] != PREDECESSOR_STARTTICKS:
        raise RuntimeError("seal requires observing the exact live priority process")
    spec = dict(process=process, status_path=str(PREDECESSOR/"status.json"),
        schema_field="schema", schema=PRIORITY_SCHEMA, transaction_id=TRANSACTION,
        completed_stages=60, stage_ledger=original["stages"])
    dependency_state(receipt, process, spec)
    plan = dict(schema=SCHEMA, source_root=str(source), output_root=str(output),
        python=sys.executable, dependency=spec, deadline_hours=args.deadline_hours,
        source_sha256=source_hashes(source),
        predecessor_plan_sha256={str(PREDECESSOR/name): PREDECESSOR_SHA
            for name in ("launch_plan.json", "executed_plan.json")},
        stages=stage_plan(output, sys.executable, source), required_receipts=required_receipts(),
        existing_queue_modified=False, source_data="fixed S7 TRAIN24K", inference_selection="SIMVAL only")
    output.mkdir(parents=True, exist_ok=False)
    save(output/"launch_plan.json", plan)
    return dict(status="sealed_not_started", stages=len(plan["stages"]), plan=str(output/"launch_plan.json"))


def load_plan(path):
    from .followup_plan import stage_plan, required_receipts
    path = Path(path).resolve(strict=True)
    plan = read(path)
    if (plan.get("schema") != SCHEMA or Path(plan["source_root"]) != REPO
            or Path(plan["output_root"]) != path.parent or plan.get("python") != sys.executable
            or not 0 < plan.get("deadline_hours", 0) <= 48
            or plan["stages"] != stage_plan(path.parent, sys.executable, REPO)
            or plan["required_receipts"] != required_receipts()):
        raise ValueError("new source/plan/output/stage binding changed")
    verify_sources(plan)
    require_subset(plan["dependency"], dict(status_path=str(PREDECESSOR/"status.json"),
        schema_field="schema", schema=PRIORITY_SCHEMA, transaction_id=TRANSACTION,
        completed_stages=60, process=dict(pid=PREDECESSOR_PID, startticks=PREDECESSOR_STARTTICKS),
        stage_ledger=read(PREDECESSOR/"executed_plan.json")["stages"]), "registered dependency")
    return plan


def execute(plan_path):
    plan = load_plan(plan_path)
    root = Path(plan["output_root"])
    with (root/"supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/"status.json").exists():
            raise ValueError("already started; no implicit resume/retry")
        if os.getsid(0) != os.getpid():
            raise RuntimeError("start supervisor in a separate session")
        deadline = time.monotonic()+plan["deadline_hours"]*3600
        status = dict(schema=SCHEMA, status="waiting_priority", pid=os.getpid(),
            process_startticks=process_identity(os.getpid())["startticks"],
            dependency=plan["dependency"], completed_stages=0, stages=deepcopy(plan["stages"]),
            active_pid=None, active_name=None, gpu_lock_owned=False, existing_queue_modified=False,
            deadline_scope="predecessor waiting only; no timeout applied to dispatched training")
        save(root/"status.json", status)
        child = None
        try:
            while True:
                state = dependency_state(read(plan["dependency"]["status_path"]),
                    process_identity(plan["dependency"]["process"]["pid"]), plan["dependency"])
                if state == "ready":
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("predecessor wait expired; predecessor untouched")
                time.sleep(min(30., max(.01, deadline-time.monotonic())))
            for receipt in plan["required_receipts"]:
                require_subset(read(receipt["path"]), receipt["expect"], receipt["path"])
            for index, stage in enumerate(status["stages"]):
                verify_sources(plan)
                if Path(stage["completion"]).exists():
                    raise ValueError("new child output already exists: "+stage["name"])
                log_path = root/("%02d_%s.log" % (index+1, stage["name"]))
                with log_path.open("x") as log:
                    child = subprocess.Popen(stage["command"], cwd=stage["cwd"], env=stage["env"],
                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                    stage.update(status="running", pid=child.pid, log=str(log_path))
                    status.update(status="running", active_pid=child.pid, active_name=stage["name"])
                    save(root/"status.json", status)
                    returncode = child.wait()
                child = None
                stage["returncode"] = returncode
                if returncode:
                    raise RuntimeError("child failed; no automatic retry: "+stage["name"])
                require_subset(read(stage["completion"]), stage["completion_expect"], stage["name"])
                for report in stage.get("reports", []):
                    require_subset(read(report["path"]), report["expect"], report["path"])
                for checkpoint in stage.get("checkpoint_contracts", []):
                    if not Path(checkpoint["path"]).is_file():
                        raise ValueError("declared checkpoint missing: "+checkpoint["path"])
                stage["status"] = "complete"
                status.update(completed_stages=index+1, active_pid=None, active_name=None)
                save(root/"status.json", status)
            status["status"] = "complete"
            save(root/"status.json", status)
        except BaseException as error:
            status.update(status="failed", error=repr(error), active_child_may_still_run=child is not None,
                recovery="inspect exact child identity and ledger; no automatic resume/restart/signals")
            save(root/"status.json", status)
            raise
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--source-root")
    parser.add_argument("--output-root")
    parser.add_argument("--deadline-hours", type=float, default=48.)
    parser.add_argument("--plan")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.seal:
        if args.execute or not args.source_root or not args.output_root:
            parser.error("--seal requires source/output roots and cannot execute")
        result = seal(args)
    else:
        if not args.plan:
            parser.error("--plan required")
        result = execute(args.plan) if args.execute else dict(status="validated_not_started", stages=len(load_plan(args.plan)["stages"]))
    print(json.dumps(result, sort_keys=True))

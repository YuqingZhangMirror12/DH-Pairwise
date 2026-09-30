"""Explicit finite handoff of WAITING dispatchers, never active training.

INERT ON IMPORT. This module is not permission to run the handoff. Root must
inspect live PID/startticks/argv, receipts and child relationships and seal an
explicit plan before `--execute` is used. No GPU lock is taken by this parent.

Transaction:
  1. Own NEW output .priority.lock; validate the two old waiting dispatchers,
     their exact children, the running candidate queue, and protected trainers.
  2. SIGSTOP only those two exact dispatcher PIDs; revalidate everything.
  3. Durably publish handoff.json(status=committed), then SIGKILL the STOPPED
     dispatcher PIDs individually. NEVER SIGTERM or killpg: the old SIGTERM
     handler would kill the still-running candidate queue's process group.
  4. Wait for the untouched candidate queue's real complete20 receipt AND
     termination; then execute the explicit finite stage list in order.

Before commit, exceptions resume only dispatcher identities stopped by this
transaction. AFTER COMMIT, never resume old dispatchers automatically. If this
process crashes/stops mid-commit, both old dispatchers may remain stopped, one
may already be dead, and the candidate queue continues. Manual recovery must
inspect handoff.json, exact process identities and the stage ledger: finish
terminating the remaining exact STOPPED dispatchers, adopt/wait any live new
child, and explicitly construct a successor plan from uncompleted stages.
Never restart this output or forge old complete receipts. A stage failure or
observer interruption does not trigger an automatic retry or kill its child.

The supplied per-stage env is passed as the ENTIRE environment, not merged with
this process's environment. PYTHONPATH must be explicit. Retained old commands
may name a JSON-pointer verbatim reference, which is checked before dispatch.
Retaining the OLD after_dependencies supervisor command is not a valid spectral
continuation: it waits for the superseded parent to complete, which cannot happen.
Use the actual spectral smoke/queue commands with resolved CPU cache inputs.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid


SCHEMA = "matched-only-explicit-priority-handoff/1"
LEDGER_KEYS = ("name", "kind", "command", "completion")
TERMINAL = ("Z", "X")
STOPPED = ("T", "t")


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    """Replace + fsync file and directory; used BEFORE irreversible signals."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def require_subset(actual, expected, context):
    if not isinstance(actual, dict):
        raise ValueError(context + ": expected receipt object")
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(context + ": missing " + key)
        if isinstance(value, dict):
            require_subset(actual[key], value, context + "." + key)
        elif actual[key] != value:
            raise ValueError(context + ": differs at " + key)


def require_identity(value, expected, *, allow_terminal=False):
    if value is None:
        raise RuntimeError("process missing: " + str(expected["pid"]))
    if value.get("pid") != expected["pid"] or value.get("startticks") != expected["startticks"]:
        raise RuntimeError("PID reused or identity differs: " + str(expected["pid"]))
    if allow_terminal and value.get("state") in TERMINAL:
        return value
    if value.get("argv") != expected["argv"] or value.get("state") in TERMINAL:
        raise RuntimeError("command changed or process terminated: " + str(expected["pid"]))
    return value


def _pointer(value, pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("verbatim reference needs an absolute JSON pointer")
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def verify_sources(plan):
    for filename, expected in plan["source_bindings"].items():
        path = Path(filename)
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("registered source changed: " + filename)
    for stage in plan["stages"]:
        reference = stage.get("verbatim_reference")
        if reference and _pointer(read_json(reference["path"]), reference["pointer"]) != stage["command"]:
            raise ValueError("retained command differs from explicit reference: " + stage["name"])


def validate_plan(plan):
    if plan.get("schema") != SCHEMA or not Path(plan.get("output_root", "")).is_absolute():
        raise ValueError("wrong schema or output root")
    timeout = plan.get("predecessor_timeout_s")
    if not isinstance(timeout, (float, int)) or not 0 < timeout <= 48 * 3600:
        raise ValueError("finite predecessor wait must be at most48h")
    dispatchers = plan.get("dispatchers", [])
    if len(dispatchers) != 2:
        raise ValueError("exactly the two waiting dispatcher identities are required")
    processes = [d["process"] for d in dispatchers] + [plan["current_queue"]["process"]] + plan["protected_processes"]
    for process in processes:
        if (type(process.get("pid")) is not int or process["pid"] <= 1
                or type(process.get("startticks")) is not int or process["startticks"] <= 0
                or not isinstance(process.get("argv"), list) or not process["argv"]
                or not all(isinstance(x, str) and x for x in process["argv"])):
            raise ValueError("exact PID/startticks/nonempty argv required")
    old = [d["process"]["pid"] for d in dispatchers]
    protected = {p["pid"] for p in processes[2:]}
    if len(set(old)) != 2 or set(old) & protected:
        raise ValueError("dispatcher PID duplicates a protected queue/trainer")
    current = plan["current_queue"]
    if (current.get("completed_stages") != 20 or len(current.get("stage_ledger", [])) != 20
            or not current.get("schema") or not Path(current.get("status_path", "")).is_absolute()):
        raise ValueError("current queue needs exact20-stage ledger and status identity")
    if not plan["protected_processes"]:
        raise ValueError("register the current active trainer identity explicitly")
    child_owners = 0
    for row in dispatchers:
        if (not Path(row.get("status_path", "")).is_absolute()
                or not isinstance(row.get("status_expect"), dict)
                or not isinstance(row.get("expected_children"), list)
                or row["status_expect"].get("status") not in ("running", "waiting_dependencies")):
            raise ValueError("dispatcher must explicitly declare waiting state and children")
        if row["expected_children"] == [current["process"]["pid"]]:
            child_owners += 1
            if (row["status_expect"].get("status") != "running"
                    or row["status_expect"].get("active_pid") != current["process"]["pid"]):
                raise ValueError("candidate parent must be waiting on the registered queue")
        elif row["expected_children"] != [] or row["status_expect"].get("status") != "waiting_dependencies":
            raise ValueError("other dispatcher must have no child and wait for dependencies")
    if child_owners != 1:
        raise ValueError("exactly one dispatcher must parent the current queue")
    stages = plan.get("stages", [])
    if not stages or len({s["name"] for s in stages}) != len(stages):
        raise ValueError("a finite list of uniquely named stages is required")
    for stage in stages:
        if (not isinstance(stage.get("command"), list) or not stage["command"]
                or not all(isinstance(x, str) and x for x in stage["command"])
                or not Path(stage["command"][0]).is_absolute()
                or not Path(stage.get("cwd", "")).is_absolute()
                or not Path(stage.get("completion", "")).is_absolute()
                or not isinstance(stage.get("completion_expect"), dict)
                or not stage["completion_expect"].get("status")
                or not isinstance(stage.get("env"), dict) or "PYTHONPATH" not in stage["env"]
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in stage["env"].items())):
            raise ValueError("stage requires explicit argv/cwd/full env/completion expectations")
        if any("after_dependencies" in arg for arg in stage["command"]):
            raise ValueError("old spectral dependency waiter cannot survive supersession; use its actual child stages")
    if not isinstance(plan.get("source_bindings"), dict) or not plan["source_bindings"]:
        raise ValueError("explicit source hash bindings required")
    for path, digest in plan["source_bindings"].items():
        if not Path(path).is_absolute() or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid source binding")
    return plan


class Runtime:
    """Only real side-effect boundary; tests replace it with simulated processes."""
    monotonic = staticmethod(time.monotonic)
    now = staticmethod(time.time)
    sleep = staticmethod(time.sleep)

    @staticmethod
    def assert_detached():
        if os.getsid(0) != os.getpid() or os.getpgrp() != os.getpid():
            raise RuntimeError("start the new supervisor in its own session before --execute")

    @staticmethod
    def process(pid):
        path = Path("/proc") / str(pid)
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            argv = [x.decode("utf-8", errors="strict") for x in (path / "cmdline").read_bytes().split(b"\0") if x]
        except FileNotFoundError:
            return None
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), startticks=int(fields[19]), argv=argv)

    @staticmethod
    def children(pid):
        root = Path("/proc") / str(pid) / "task"
        result = set()
        for thread in root.iterdir():
            result.update(int(x) for x in (thread / "children").read_text().split())
        return sorted(result)

    @staticmethod
    def signal(pid, signum):
        os.kill(pid, signum)  # NEVER killpg or SIGTERM.

    @staticmethod
    def spawn(stage, log):
        return subprocess.Popen(stage["command"], cwd=stage["cwd"], env=dict(stage["env"]),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def queue_receipt(plan):
    spec = plan["current_queue"]
    receipt = read_json(spec["status_path"])
    if receipt.get("schema_version") != spec["schema"] or receipt.get("pid") != spec["process"]["pid"]:
        raise RuntimeError("candidate queue receipt identity differs")
    actual = receipt.get("stages", [])
    if len(actual) != 20 or [{k: s.get(k) for k in LEDGER_KEYS} for s in actual] != spec["stage_ledger"]:
        raise RuntimeError("candidate queue stage ledger changed")
    if receipt.get("status") not in ("running", "complete"):
        raise RuntimeError("candidate queue failed/interrupted: " + str(receipt.get("status")))
    if receipt["status"] == "complete" and (receipt.get("completed_stages") != 20
            or any(s.get("status") != "complete" for s in actual)):
        raise RuntimeError("candidate queue incomplete complete20 receipt")
    return receipt


def validate_live_handoff(plan, runtime, *, stopped=False):
    verify_sources(plan)
    snapshots = []
    for row in plan["dispatchers"]:
        process = require_identity(runtime.process(row["process"]["pid"]), row["process"])
        if (stopped and process["state"] not in STOPPED) or (not stopped and process["state"] in STOPPED):
            raise RuntimeError("dispatcher stop state differs from this transaction")
        receipt = read_json(row["status_path"])
        require_subset(receipt, row["status_expect"], row["name"])
        if sorted(runtime.children(process["pid"])) != sorted(row["expected_children"]):
            raise RuntimeError("dispatcher children changed: " + row["name"])
        snapshots.append(dict(name=row["name"], process=process, receipt=receipt))
    queue = require_identity(runtime.process(plan["current_queue"]["process"]["pid"]), plan["current_queue"]["process"])
    parent = next(d["process"]["pid"] for d in plan["dispatchers"] if d["expected_children"])
    if queue["ppid"] != parent or queue["state"] in STOPPED:
        raise RuntimeError("current queue parent/state changed before handoff")
    if queue_receipt(plan)["status"] != "running":
        raise RuntimeError("queue finished while planning handoff; re-observe, do not guess the next stage")
    for spec in plan["protected_processes"]:
        value = require_identity(runtime.process(spec["pid"]), spec)
        if value["state"] in STOPPED:
            raise RuntimeError("protected trainer unexpectedly stopped")
    return snapshots


def wait_stopped(spec, runtime):
    deadline = runtime.monotonic() + 5.
    while True:
        value = require_identity(runtime.process(spec["pid"]), spec)
        if value["state"] in STOPPED:
            return
        if runtime.monotonic() >= deadline:
            raise TimeoutError("dispatcher did not stop; rollback only our stopped identities")
        runtime.sleep(.01)


def wait_current_queue(plan, runtime):
    deadline = runtime.monotonic() + plan["predecessor_timeout_s"]
    while True:
        receipt = queue_receipt(plan)
        value = runtime.process(plan["current_queue"]["process"]["pid"])
        if value is not None:
            require_identity(value, plan["current_queue"]["process"], allow_terminal=True)
        terminal = value is None or value["state"] in TERMINAL
        if terminal:
            if receipt["status"] != "complete":
                raise RuntimeError("candidate queue terminated without complete20 receipt")
            return receipt
        if value["state"] in STOPPED:
            raise RuntimeError("candidate queue unexpectedly stopped; never auto-signal it")
        if runtime.monotonic() >= deadline:
            raise TimeoutError("candidate predecessor wait expired; no training was interrupted")
        runtime.sleep(min(15., max(.01, deadline - runtime.monotonic())))


def execute(plan, *, runtime=None):
    plan = validate_plan(copy.deepcopy(plan))
    runtime = runtime or Runtime()
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    status_path, handoff_path = root / "status.json", root / "handoff.json"
    with (root / ".priority.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists() or handoff_path.exists():
            raise ValueError("output already started; explicit recovery in a new plan/output, never auto-resume")
        runtime.assert_detached()
        transaction = str(uuid.uuid4())
        status = dict(schema=SCHEMA, status="prepared", pid=os.getpid(), transaction_id=transaction,
            started_at_unix=runtime.now(), completed_stages=0, stages=copy.deepcopy(plan["stages"]),
            old_receipts_modified=False, gpu_lock_owned=False, active_pid=None)
        atomic_json(root / "executed_plan.json", plan)
        atomic_json(status_path, status)
        paused, committed, child = [], False, None
        try:
            before = validate_live_handoff(plan, runtime)
            for row in plan["dispatchers"]:
                spec = row["process"]
                require_identity(runtime.process(spec["pid"]), spec)
                runtime.signal(spec["pid"], signal.SIGSTOP)
                paused.append(spec)
                wait_stopped(spec, runtime)
            after = validate_live_handoff(plan, runtime, stopped=True)
            receipt = dict(schema=SCHEMA, status="committed", transaction_id=transaction,
                committed_at_unix=runtime.now(), new_supervisor_pid=os.getpid(),
                dispatchers=plan["dispatchers"], stopped_dispatcher_pids=[s["pid"] for s in paused],
                current_queue=plan["current_queue"], protected_processes=plan["protected_processes"],
                before=before, after=after, termination_signals=[],
                old_dispatchers_superseded=True, old_receipts_modified=False,
                recovery="After commit NEVER automatically resume old dispatchers; inspect exact identities, any live child and ledger before explicit successor.")
            atomic_json(handoff_path, receipt)
            committed = True
            status.update(status="handoff_committed", handoff=str(handoff_path))
            atomic_json(status_path, status)
            for spec in reversed(paused):
                current = runtime.process(spec["pid"])
                if current is None:
                    continue
                require_identity(current, spec, allow_terminal=True)
                if current["state"] in TERMINAL:
                    continue
                if current["state"] not in STOPPED:
                    raise RuntimeError("committed dispatcher is no longer stopped; manual recovery required")
                runtime.signal(spec["pid"], signal.SIGKILL)
                receipt["termination_signals"].append(dict(pid=spec["pid"], signal="SIGKILL", unix=runtime.now()))
                atomic_json(handoff_path, receipt)
            status.update(status="waiting_current_queue")
            atomic_json(status_path, status)
            completed_queue = wait_current_queue(plan, runtime)
            atomic_json(root / "predecessor_complete.json", completed_queue)
            for index, stage in enumerate(status["stages"]):
                verify_sources(plan)
                if Path(stage["completion"]).exists():
                    raise ValueError("new stage already has a completion file: " + stage["name"])
                logfile = root / ("%03d_%s.log" % (index + 1, stage["name"]))
                with logfile.open("x") as log:
                    child = runtime.spawn(stage, log)
                    stage.update(status="running", pid=child.pid, started_at_unix=runtime.now(), log=str(logfile))
                    status.update(status="running", active_pid=child.pid, active_stage=index + 1, active_name=stage["name"])
                    atomic_json(status_path, status)
                    code = child.wait()  # no predecessor deadline applies to active GPU work
                child = None  # wait() returned: no observer uncertainty about this child
                stage["returncode"] = code
                if code:
                    raise RuntimeError("stage failed, no automatic retry: " + stage["name"])
                require_subset(read_json(stage["completion"]), stage["completion_expect"], stage["name"])
                stage.update(status="complete", returncode=code, finished_at_unix=runtime.now())
                child = None
                status.update(completed_stages=index + 1, active_pid=None, active_name=None)
                atomic_json(status_path, status)
            status.update(status="complete", finished_at_unix=runtime.now())
            atomic_json(status_path, status)
            return status
        except BaseException as error:
            # An fsync error after atomic replace may have published a commit.
            # Conservatively inspect its transaction ID before any rollback.
            if not committed and handoff_path.exists():
                found = read_json(handoff_path)
                committed = found.get("status") == "committed" and found.get("transaction_id") == transaction
            rollback = []
            if not committed:
                for spec in reversed(paused):
                    try:
                        value = require_identity(runtime.process(spec["pid"]), spec)
                        if value["state"] in STOPPED:
                            runtime.signal(spec["pid"], signal.SIGCONT)
                            rollback.append(dict(pid=spec["pid"], resumed=True))
                    except Exception as rollback_error:
                        rollback.append(dict(pid=spec["pid"], resumed=False, error=repr(rollback_error)))
            status.update(status="failed", error=repr(error), committed_handoff=committed,
                rollback=rollback, active_child_may_still_run=child is not None,
                finished_at_unix=runtime.now(),
                recovery="manual identity/ledger inspection; no automatic retry/resume; never signal active candidate queue/trainer")
            atomic_json(status_path, status)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--execute", action="store_true", help="deliberate dispatcher handoff; root must review first")
    args = parser.parse_args()
    plan = validate_plan(read_json(args.plan))
    verify_sources(plan)
    if args.execute:
        print(json.dumps(execute(plan), sort_keys=True))
    else:
        print(json.dumps(dict(status="validated_not_executed", stages=len(plan["stages"]))))

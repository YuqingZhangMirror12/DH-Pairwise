"""Retain the sealed depth/spectral work after the explicit priority handoff.

Inert on import. This helper does not signal the superseded dispatchers, alter
their receipts, acquire a GPU lock, resume, retry, or invent completed work.
Each child is the verbatim command from its original sealed implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SCHEMA = "matched-only-retained-legacy-phase/1"
HANDOFF_SCHEMA = "matched-only-explicit-priority-handoff/1"
REGISTERED_DISPATCHERS = {101639: 932414692, 103819: 932543651}
MODULE = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919."
LEDGER_KEYS = ("name", "kind", "command", "completion")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(state=fields[0], startticks=int(fields[19]))


def require_terminal(spec, inspect=process_identity):
    observed = inspect(spec["pid"])
    if observed is not None:
        if observed["startticks"] != spec["startticks"]:
            raise RuntimeError("registered PID reused; refusing a different process")
        if observed["state"] not in ("Z", "X"):
            raise RuntimeError("registered predecessor is still alive")


def validate_handoff(path, inspect=process_identity):
    receipt = read_json(path)
    dispatchers = receipt.get("dispatchers", [])
    identities = {row.get("process", {}).get("pid"): row.get("process", {}).get("startticks")
                  for row in dispatchers}
    signals = receipt.get("termination_signals", [])
    if (receipt.get("schema") != HANDOFF_SCHEMA or receipt.get("status") != "committed"
            or receipt.get("old_dispatchers_superseded") is not True
            or receipt.get("old_receipts_modified") is not False
            or len(dispatchers) != 2 or identities != REGISTERED_DISPATCHERS
            or len(signals) != 2
            or {row.get("pid") for row in signals} != set(REGISTERED_DISPATCHERS)
            or any(row.get("signal") != "SIGKILL" for row in signals)):
        raise RuntimeError("both exact dispatchers require a committed termination receipt")
    for row in dispatchers:
        require_terminal(row["process"], inspect)
    return receipt


def validate_candidate_queue(legacy, plan, handoff, inspect=process_identity):
    root = Path(plan["output_root"]) / "candidate_local"
    spec = handoff["current_queue"]
    if Path(spec["status_path"]) != root / "queue_status.json":
        raise RuntimeError("handoff candidate queue path differs from the sealed plan")
    receipt = read_json(spec["status_path"])
    expected = legacy.candidate.stage_plan(root, plan["python"])
    actual = receipt.get("stages", [])
    ledger = lambda rows: [{k: row.get(k) for k in LEDGER_KEYS} for row in rows]
    if (receipt.get("schema_version") != "rachel-candidate-local-queue/1"
            or receipt.get("pid") != spec["process"]["pid"]
            or receipt.get("status") != "complete" or receipt.get("completed_stages") != 20
            or len(actual) != 20 or len(expected) != 20 or ledger(actual) != ledger(expected)
            or any(row.get("status") != "complete" for row in actual)):
        raise RuntimeError("actual candidate queue must complete its unchanged 20-stage ledger")
    require_terminal(spec["process"], inspect)
    return receipt


def load_legacy(phase):
    suffix = {"depth": "continuation_depth_controls.followup_supervisor",
              "spectral": "spectral_training.after_dependencies"}[phase]
    return importlib.import_module(MODULE + suffix)


class Runtime:
    inspect = staticmethod(process_identity)

    @staticmethod
    def spawn(stage, source, env, log):
        return subprocess.Popen(stage["command"], cwd=source, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)

    @staticmethod
    def stop_child(child):
        # Only this helper's newly created child session, never the old queue.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def reject_existing_output(stage):
    if Path(stage["completion"]).exists():
        raise ValueError("existing legacy completion will not be overwritten: " + stage["name"])
    command = stage["command"]
    if "--output" in command and Path(command[command.index("--output") + 1]).exists():
        raise ValueError("existing legacy output will not be resumed or overwritten: " + stage["name"])


def execute(args, *, legacy=None, runtime=None):
    runtime = Runtime() if runtime is None else runtime
    legacy = load_legacy(args.phase) if legacy is None else legacy
    plan_path = Path(args.plan).resolve(strict=True)
    if sha256(plan_path) != args.plan_sha256:
        raise ValueError("original sealed plan digest differs")
    plan = legacy.load_plan(plan_path)
    handoff_path = Path(args.handoff_receipt).resolve(strict=True)
    handoff = validate_handoff(handoff_path, runtime.inspect)
    root = Path(args.output).resolve()
    source, old_root = Path(plan["source_root"]).resolve(), Path(plan["output_root"]).resolve()
    if any(root == p or p in root.parents for p in (source, old_root)):
        raise ValueError("new phase receipts must be outside sealed source and original outputs")
    root.mkdir(parents=True, exist_ok=False)
    status = dict(schema_version=SCHEMA, status="preparing", phase=args.phase, pid=os.getpid(),
                  started_at_unix=time.time(), original_plan=str(plan_path), original_plan_sha256=args.plan_sha256,
                  handoff_receipt=str(handoff_path), handoff_receipt_sha256=sha256(handoff_path),
                  old_supervisor_receipts_modified=False, completed_stages=0, stages=[])
    status_path = root / "status.json"
    child = None
    atomic_json(status_path, status)
    try:
        inputs = None
        if args.phase == "depth":
            candidate = validate_candidate_queue(legacy, plan, handoff, runtime.inspect)
            status["candidate_completion"] = dict(pid=candidate["pid"], completed_stages=20,
                receipt=str(old_root / "candidate_local/queue_status.json"),
                sha256=sha256(old_root / "candidate_local/queue_status.json"))
            generated = legacy.stage_plan(plan["output_root"], plan["python"])
            if generated != plan["stages"] or len(generated) != 25:
                raise ValueError("sealed original 25-stage depth plan changed")
            stages = generated[3:]
            if len(stages) != 22:
                raise ValueError("exact remaining 22 depth stages required")
        else:
            # The old GPU supervisor is demonstrably superseded. Resolve the
            # actual completed CPU registrations with the original validator.
            inputs = legacy.resolve_inputs()
            registration = old_root / "resolved_inputs.json"
            if registration.exists():
                raise ValueError("existing spectral input registration will not be overwritten")
            legacy.save(registration, inputs)  # Exact original serialization.
            status["resolved_inputs"] = str(registration)
            status["resolved_inputs_sha256"] = sha256(registration)
            stages = legacy.stage_plan(plan, inputs, status["resolved_inputs_sha256"])
            if len(stages) != 4 or stages[-1]["kind"] != "queue":
                raise ValueError("exact three spectral smokes plus original queue required")
        # No command edits or inherited new-source shadowing for legacy jobs.
        env = dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
        status.update(status="ready", stages=stages, child_cwd=str(source),
                      child_env_overrides={k: env[k] for k in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "PYTHONUNBUFFERED")})
        atomic_json(status_path, status)
        for index, stage in enumerate(stages):
            if sha256(plan_path) != args.plan_sha256:
                raise ValueError("original sealed plan changed before dispatch")
            legacy.verify_source(plan)
            reject_existing_output(stage)
            if args.phase == "depth" and stage["kind"] == "depth_training":
                legacy.validate_depth_smokes(old_root / "depth_controls")
            if args.phase == "spectral" and stage["kind"] == "queue":
                legacy.queue.validate_smokes(old_root, inputs["cache_bundle_sha"])
            log_path = root / ("%02d_%s.log" % (index + 1, stage["name"]))
            with log_path.open("x") as log:
                child = runtime.spawn(stage, str(source), env, log)
                stage.update(status="running", pid=child.pid, log=str(log_path), started_at_unix=time.time())
                status.update(status="running", active_name=stage["name"], active_stage=index + 1, active_pid=child.pid)
                atomic_json(status_path, status)
                code = child.wait()
                child = None
            stage["returncode"] = code
            if code:
                raise RuntimeError(stage["name"] + " failed; no retry")
            legacy.validate_completion(stage)
            stage.update(status="complete", finished_at_unix=time.time())
            status.update(completed_stages=index + 1, active_pid=None)
            atomic_json(status_path, status)
        status.update(status="complete", active_name=None, active_pid=None)
        return status
    except BaseException as error:
        if child is not None:
            runtime.stop_child(child)
        status.update(status="failed", error=repr(error), active_pid=None)
        raise
    finally:
        status["finished_at_unix"] = time.time()
        atomic_json(status_path, status)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", choices=("depth", "spectral"), required=True)
    for name in ("plan", "plan-sha256", "output", "handoff-receipt"):
        result.add_argument("--" + name, required=True)
    return result


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise KeyboardInterrupt("legacy phase received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    execute(parser().parse_args())

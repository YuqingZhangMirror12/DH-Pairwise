"""Finite sequential GPU queue with explicit live-process dependencies."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def live(pid, marker):
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        return marker.encode() in path.read_bytes()
    except FileNotFoundError:
        return False


def replace_dependency_handles(state, replacements):
    """Replace only stopped dependency processes; leave experiment config frozen."""
    known = {row["name"]: row for row in state["config"]["dependencies"]}
    overrides = dict(state.get("dependency_pid_overrides", {}))
    changes = []
    for name, pid in replacements.items():
        if name not in known or type(pid) is not int or pid <= 0:
            raise ValueError("invalid dependency process replacement: " + name)
        dependency = known[name]
        old_pid = overrides.get(name, dependency["pid"])
        if pid == old_pid:
            continue
        if live(old_pid, dependency["marker"]):
            raise RuntimeError("old dependency is still live: " + name)
        if not live(pid, dependency["marker"]):
            raise RuntimeError("replacement dependency is not live with expected marker: " + name)
        changes.append(dict(name=name, previous_pid=old_pid, replacement_pid=pid, recorded_at=now()))
        overrides[name] = pid
    state["dependency_pid_overrides"] = overrides
    state.setdefault("dependency_replacements", []).extend(changes)


def retry_unstarted_stage(state, name):
    """Explicit recovery for an import/pre-start failure with no output tree."""
    matches = [row for row in state["stages"] if row["name"] == name]
    if len(matches) != 1 or matches[0]["status"] != "failed":
        raise ValueError("retry requires one failed stage: " + name)
    stage = matches[0]
    if stage.get("pid") and live(stage["pid"], stage["marker"]):
        raise RuntimeError("failed stage process is still live: " + name)
    command = stage["command"]
    paths = [Path(command[i + 1]) for i, arg in enumerate(command[:-1])
             if arg in ("--output", "--output-root")]
    if not paths or any(path.exists() for path in paths) or Path(stage["completion_path"]).exists():
        raise RuntimeError("fresh retry requires absent stage outputs: " + name)
    fields = ("pid", "returncode", "started_at", "finished_at", "log")
    stage.setdefault("attempt_history", []).append(dict(reason="explicit retry after no-output failure",
        recorded_at=now(), **{key: stage[key] for key in fields if key in stage}))
    for key in fields:
        stage.pop(key, None)
    stage["status"] = "queued"


def run(config_path, resume=False, dependency_pids=None, retry_stages=None):
    if (dependency_pids or retry_stages) and not resume:
        raise ValueError("recovery options require explicit --resume")
    config = read(config_path)
    root = Path(config["root"])
    state_path = root / "queue_state.json"
    with (root / ".queue.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            if not resume:
                raise ValueError("queue state exists; explicit --resume required")
            state = read(state_path)
            if state["config"] != config:
                raise ValueError("queue config differs from frozen queue")
            previous = state.get("active_child")
            if previous and live(previous["pid"], previous["marker"]):
                raise RuntimeError("previous child still live; do not start duplicate")
        else:
            state = dict(config=config, status="starting", stages=[dict(s, status="queued") for s in config["stages"]],
                         started_at=now())
        replace_dependency_handles(state, dependency_pids or {})
        for name in retry_stages or []:
            retry_unstarted_stage(state, name)
        if resume:
            state.setdefault("resumptions", []).append(dict(at=now(), previous_status=state["status"],
                previous_error=state.pop("error", None), previous_failed_at=state.pop("failed_at", None)))
        state.update(pid=os.getpid(), status="resuming" if resume else "starting")
        save(state_path, state)
        try:
            for dependency in config["dependencies"]:
                while True:
                    status_path = Path(dependency["status_path"])
                    current = read(status_path) if status_path.exists() else {}
                    dependency_pid = state.get("dependency_pid_overrides", {}).get(dependency["name"], dependency["pid"])
                    is_live = live(dependency_pid, dependency["marker"])
                    if current.get("status") == "complete" and not is_live:
                        break
                    if current.get("status") == "failed" or not is_live:
                        raise RuntimeError("dependency not completed and process stopped: " + dependency["name"])
                    state.update(status="waiting_for_dependency", active_stage=dependency["name"],
                                 dependency_stage=current.get("stage", current.get("phase")), observed_at=now())
                    save(state_path, state)
                    time.sleep(30)
            for stage in state["stages"]:
                if stage["status"] == "complete":
                    continue
                command = list(stage["command"])
                if stage["status"] in ("running", "failed"):
                    if not stage.get("resume_arguments"):
                        raise RuntimeError("stage requires scoped manual recovery: " + stage["name"])
                    command += stage["resume_arguments"]
                pythonpath = os.pathsep.join([config["source"]] + stage.get("extra_pythonpath", []))
                environment = dict(os.environ, PYTHONPATH=pythonpath, OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
                log_path = root / (stage["name"] + ".log")
                stage.update(status="running", started_at=now(), log=str(log_path))
                state.update(status="running", active_stage=stage["name"])
                with log_path.open("a") as log:
                    child = subprocess.Popen(command, cwd=config["source"], env=environment,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    stage["pid"] = child.pid
                    state["active_child"] = dict(pid=child.pid, marker=stage["marker"])
                    save(state_path, state)
                    result = child.wait()
                stage.update(returncode=result, finished_at=now())
                state.pop("active_child", None)
                if result:
                    stage["status"] = "failed"
                    raise RuntimeError(stage["name"] + " returned" + str(result))
                receipt = read(stage["completion_path"])
                if receipt.get("status") not in stage.get("completion_statuses", ["complete"]):
                    stage["status"] = "failed"
                    raise RuntimeError("stage returned without completion receipt: " + stage["name"])
                stage["status"] = "complete"
                save(state_path, state)
            state.update(status="complete", active_stage=None, completed_at=now())
            save(state_path, state)
        except Exception as error:
            state.update(status="failed", error=repr(error), failed_at=now())
            save(state_path, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dependency-pid", action="append", default=[], metavar="NAME=PID",
                        help="On resume, bind a stopped dependency to its verified replacement process")
    parser.add_argument("--retry-unstarted-stage", action="append", default=[], metavar="NAME",
                        help="Explicitly retry a failed stage only when its output tree is absent")
    args = parser.parse_args()
    replacements = {}
    for item in args.dependency_pid:
        name, separator, pid = item.partition("=")
        if not separator or not pid.isdecimal() or name in replacements:
            parser.error("dependency-pid must be a unique NAME=positive PID")
        replacements[name] = int(pid)
    run(args.config, args.resume, replacements, args.retry_unstarted_stage)

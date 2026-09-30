"""External, target-blind ownership registry for untouched GPU experiment tails.

No CUDA import, signals, source edits or automatic retry. A delegated original
wrapper waits without a GPU lease; a failed delegation never falls back to it.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

SCHEMA = "stage-transfer/1"
POLL_SECONDS = 2.0
STATUSES = frozenset(("queued", "running", "complete", "failed"))


def _key(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("absolute runtime receipt path required")
    return str(path.resolve())


def read_registry(root):
    path = Path(root) / "registry.json"
    if not path.exists():
        return dict(schema=SCHEMA, experiments={}, stages={}, claims={})
    with path.open() as stream:
        value = json.load(stream)
    if (value.get("schema") != SCHEMA or
            any(not isinstance(value.get(k), dict) for k in ("experiments", "stages", "claims"))):
        raise ValueError("invalid stage-transfer registry schema")
    return value


def _save(root, value):
    root = Path(root)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=str(root), prefix=".registry-",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, root / "registry.json")
        temporary = None
        descriptor = os.open(str(root), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def locked_registry(root):
    """Short global dispatch transaction; exceptions do not persist mutations."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "dispatch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            value = read_registry(root)
            yield value
            _save(root, value)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _process_stat(pid):
    if isinstance(pid, bool):
        return None
    try:
        pid = int(pid)
        if pid <= 0:
            return None
        raw = Path("/proc/%d/stat" % pid).read_text()
        fields = raw[raw.rindex(")") + 2:].split()
        return fields[0], int(fields[19])
    except (OSError, ValueError, IndexError, TypeError):
        return None


def process_startticks(pid):
    value = _process_stat(pid)
    return None if value is None else value[1]


def process_alive(pid, startticks):
    value = _process_stat(pid)
    try:
        return (not isinstance(startticks, bool) and value is not None
                and value[0] not in ("Z", "X") and value[1] == int(startticks))
    except (TypeError, ValueError):
        return False


def _require_subset(actual, expected):
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ValueError("completion object/expectation must be dictionaries")
    for key, value in expected.items():
        if key not in actual:
            raise ValueError("missing completion field: " + key)
        if isinstance(value, dict):
            _require_subset(actual[key], value)
        elif actual[key] != value:
            raise ValueError("completion field differs: " + key)


def validate_completion(stage):
    """Read authoritative original leaf proofs, including additional receipts."""
    proofs = [dict(path=stage["completion"], expect=stage["completion_expect"])]
    proofs += stage.get("additional_completions", [])
    for proof in proofs:
        with Path(proof["path"]).open() as stream:
            _require_subset(json.load(stream), proof["expect"])
    return True


def register_experiments(root, packages):
    """Atomically transfer ONLY untouched complete experiment tails.

    Completion/runtime files and prior original claims prevent registration,
    even if those files say complete or their former owner has exited.
    """
    packages = list(packages)
    if not packages:
        raise ValueError("at least one experiment package required")
    with locked_registry(root) as registry:
        reserved_outputs = {
            str(Path(path).resolve())
            for prior in registry["experiments"].values() for stage in prior["stages"]
            for path in [stage["completion"], stage.get("output")]
            if path
        }
        for source in packages:
            package = deepcopy(source)
            name = package.get("name")
            if (not isinstance(name, str) or not name or name in registry["experiments"]
                    or not isinstance(package.get("origin_lane"), str)
                    or not package["origin_lane"] or not package.get("stages")):
                raise ValueError("invalid/duplicate experiment package")
            priority = package.get("priority", 0)
            if type(priority) is not int:
                raise ValueError("integer experiment priority required")
            stage_names = set()
            for stage in package["stages"]:
                receipt = _key(stage["original_runtime_receipt"])
                stage["original_runtime_receipt"] = receipt
                if (stage.get("gpu") is not True or not stage.get("name")
                        or stage["name"] in stage_names or not isinstance(stage.get("command"), list)
                        or not stage["command"] or not Path(stage["cwd"]).is_absolute()
                        or not Path(stage["completion"]).is_absolute() or not stage.get("completion_expect")):
                    raise ValueError("only unique original GPU leaf stages may be transferred")
                stage_names.add(stage["name"])
                paths = [receipt, stage["completion"]]
                paths += [p["path"] for p in stage.get("additional_completions", [])]
                output_keys = {str(Path(p).resolve()) for p in
                               (stage["completion"], stage.get("output")) if p}
                if (receipt in registry["stages"] or receipt in registry["claims"]
                        or output_keys & reserved_outputs
                        or any(Path(path).exists() for path in paths)
                        or (stage.get("output") and Path(stage["output"]).exists())):
                    raise ValueError("stage already claimed, started or has existing output: " + receipt)
                reserved_outputs.update(output_keys)
                registry["stages"][receipt] = dict(experiment=name, stage_name=stage["name"],
                    status="queued", original_runtime_receipt=receipt)
            package.update(status="queued", priority=priority, assigned_gpu=None,
                           worker_pid=None, worker_startticks=None, registered_at_unix=time.time())
            registry["experiments"][name] = package
    return [p["name"] for p in packages]


def _stage_spec(registry, receipt):
    entry = registry["stages"][receipt]
    experiment = registry["experiments"].get(entry["experiment"])
    if not isinstance(experiment, dict):
        raise ValueError("delegation references a missing experiment")
    matches = [s for s in experiment["stages"] if s["original_runtime_receipt"] == receipt]
    if len(matches) != 1 or matches[0]["name"] != entry["stage_name"]:
        raise ValueError("delegated stage identity differs")
    return experiment, entry, matches[0]


def _check_progress(experiment, entry):
    if experiment.get("status") not in STATUSES or entry.get("status") not in STATUSES:
        raise RuntimeError("invalid delegated execution status")
    if experiment["status"] == "failed" or entry["status"] == "failed":
        raise RuntimeError("delegated experiment/stage failed; original fallback forbidden")
    if entry["status"] == "complete":
        if type(entry.get("returncode")) is not int or entry["returncode"] != 0:
            raise RuntimeError("delegated completion lacks successful process exit")
        return True
    if experiment["status"] == "complete":
        raise RuntimeError("completed experiment contains unfinished delegated stage")
    if entry["status"] == "running" and experiment["status"] != "running":
        raise RuntimeError("running delegated stage lacks running experiment owner")
    if experiment["status"] == "running" and not process_alive(
            experiment.get("worker_pid"), experiment.get("worker_startticks")):
        raise RuntimeError("delegated worker missing or PID identity changed; no automatic retry")
    # A child can exit just before its live worker records/validates completion.
    # Do not declare that normal reap window failed from child PID alone.
    return False


def original_dispatch_gate(root, receipt):
    """Return False for an atomically claimed original, True for completed proxy.

    Must run before CUDA initialization, original runtime file or GPU lease.
    Queued packages may wait for the coordinator; running orphan owners fail.
    No GPU or registry lock is held while sleeping or reading completion files.
    """
    receipt = _key(receipt)
    pid = os.getpid()
    ticks = process_startticks(pid)
    if ticks is None:
        raise RuntimeError("Linux process identity required for dispatch ownership")
    with locked_registry(root) as registry:
        if receipt not in registry["stages"]:
            if receipt in registry["claims"] or Path(receipt).exists():
                raise RuntimeError("original stage already claimed/started; no implicit retry")
            registry["claims"][receipt] = dict(role="original", pid=pid, startticks=ticks,
                                               claimed_at_unix=time.time())
            return False
        experiment, entry, _ = _stage_spec(registry, receipt)
        previous = entry.get("waiting_proxy")
        if previous and (previous.get("pid"), previous.get("startticks")) != (pid, ticks):
            raise RuntimeError("delegated stage already has an original waiting proxy")
        entry["waiting_proxy"] = dict(pid=pid, startticks=ticks, status="waiting",
                                      started_at_unix=time.time())
    while True:
        registry = read_registry(root)
        if receipt not in registry["stages"]:
            raise RuntimeError("delegation disappeared; original fallback forbidden")
        experiment, entry, stage = _stage_spec(registry, receipt)
        if _check_progress(experiment, entry):
            validate_completion(stage)
            with locked_registry(root) as current:
                exp_now, row_now, spec_now = _stage_spec(current, receipt)
                if not _check_progress(exp_now, row_now) or spec_now != stage:
                    raise RuntimeError("delegated completion changed during acceptance")
                row_now["waiting_proxy"].update(status="complete", finished_at_unix=time.time())
            return True
        time.sleep(POLL_SECONDS)

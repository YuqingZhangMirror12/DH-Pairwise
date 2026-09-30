"""Supersede only an unstarted old axis-recipe queue after its replacement exists.

Linux /proc support is required. Only the verified old parent
is briefly stopped for a final race-safe recheck; only that parent and its exact
GNU tail waiter receive SIGTERM. The watched GAP process is never signalled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import tempfile
import time


PREFIX = "experiments.rachel_n512_formal_30k."
OLD_STAGES = ("train_e4", "evaluate_e4_test", "evaluate_e4_real",
              "train_e5", "evaluate_e5_test", "evaluate_e5_real")
NEW_STAGES = ("smoke_e4v2_e5v2", "train_e4v2", "evaluate_e4v2_test", "evaluate_e4v2_real",
              "train_e5v2", "evaluate_e5v2_test", "evaluate_e5v2_real")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent),
                                     prefix="." + path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _process(pid):
    path = Path("/proc") / str(pid)
    stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, state=stat[0], parent_pid=int(stat[1]), start_ticks=int(stat[19]),
                argv=[part.decode() for part in (path / "cmdline").read_bytes().split(b"\0") if part])


def _children(pid):
    return tuple(int(value) for value in (Path("/proc") / str(pid) / "task" / str(pid) / "children").read_text().split())


def _argument(argv, name):
    values = [argv[index + 1] for index, token in enumerate(argv[:-1]) if token == name]
    values += [token.split("=", 1)[1] for token in argv if token.startswith(name + "=")]
    if len(values) != 1:
        raise RuntimeError("expected one exact process argument: " + name)
    return values[0]


def _module_process(process, module, root):
    argv = process["argv"]
    if (process["state"] in ("Z", "X") or "-m" not in argv
            or argv[argv.index("-m") + 1] != PREFIX + module
            or Path(_argument(argv, "--root")).resolve() != root):
        raise RuntimeError("PID is not the expected live module and exact root")


def _queued(state, names, statuses, pid=None):
    if (state.get("status") not in statuses
            or tuple(stage.get("name") for stage in state.get("stages", [])) != names
            or any(stage.get("status") != "queued" or stage.get("pid") is not None
                   for stage in state.get("stages", []))
            or (pid is not None and state.get("pid") != pid)):
        raise RuntimeError("queue is not entirely unstarted in the expected waiting state")


def _tail(process, parent_pid, gap_pid):
    argv = process["argv"]
    if (process["parent_pid"] != parent_pid or process["state"] in ("Z", "X")
            or not argv or Path(argv[0]).name != "tail"
            or argv[1:] != ["--pid=" + str(gap_pid), "--sleep-interval=10", "-f", "/dev/null"]
            or _children(process["pid"])):
        raise RuntimeError("old child is not the exact childless GAP tail waiter")


def _same_process(before):
    after = _process(before["pid"])
    if any(after[key] != before[key] for key in ("start_ticks", "argv", "parent_pid")):
        raise RuntimeError("process identity changed")
    return after


def _owned_alive(before):
    """Recheck start time and argv; a recycled PID is never an owned process."""
    try:
        after = _process(before["pid"])
    except (FileNotFoundError, ProcessLookupError):
        return False
    if after["start_ticks"] != before["start_ticks"] or after["state"] in ("Z", "X"):
        return False
    if after["argv"] != before["argv"]:
        raise RuntimeError("owned process argv changed; refusing to signal it")
    return True


def _send_owned(before, signum):
    # The old parent is stopped before final guards. The tail may be reparented
    # after its parent exits, so identity here uses immutable start ticks/argv.
    if not _owned_alive(before):
        return False
    try:
        os.kill(before["pid"], signum)
    except ProcessLookupError:
        return False
    return True


def _wait_exit(before, timeout=5):
    deadline = time.monotonic() + timeout
    while _owned_alive(before):
        if time.monotonic() >= deadline:
            return False
        time.sleep(.05)
    return True


def _no_old_training(root):
    """Read-only bounded process check; unrelated E2/GAP jobs are ignored."""
    matches = []
    for number, path in enumerate(Path("/proc").iterdir()):
        if number > 100000:
            raise RuntimeError("unexpectedly large process directory; refusing an incomplete audit")
        if not path.name.isdigit():
            continue
        try:
            process = _process(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        argv = process["argv"]
        if process["state"] in ("Z", "X") or "-m" not in argv:
            continue
        module = argv[argv.index("-m") + 1]
        if not module.startswith(PREFIX + "train_"):
            continue
        try:
            output = Path(_argument(argv, "--output")).resolve()
        except RuntimeError:
            continue
        if output == root or root in output.parents:
            matches.append(process["pid"])
    if matches:
        raise RuntimeError("live training exists under old root: " + repr(matches))
    return matches


def supersede(args):
    if not Path("/proc/self/stat").exists():
        raise RuntimeError("Linux /proc process identity checks are required")
    old_root, new_root = Path(args.old_root).resolve(strict=True), Path(args.new_root).resolve(strict=True)
    if old_root == new_root or old_root in new_root.parents or new_root in old_root.parents:
        raise ValueError("old and new queue roots must be separate")
    old_pid = args.old_pid
    if type(old_pid) is not int or old_pid <= 1:
        raise ValueError("an exact old PID is required")
    old_path, new_path = old_root / "pipeline_state.json", new_root / "pipeline_state.json"
    receipt_path = new_root / "supersession.json"
    if receipt_path.exists():
        raise FileExistsError("supersession receipt already exists; refusing duplicate signalling")
    old_bytes, old_state, new_state = old_path.read_bytes(), _read(old_path), _read(new_path)
    _queued(old_state, OLD_STAGES, {"waiting_for_gap"}, old_pid)
    _queued(new_state, NEW_STAGES, {"waiting_for_preparation", "waiting_for_gap"})
    new_pid = new_state.get("pid")
    if type(new_pid) is not int or new_pid <= 1:
        raise RuntimeError("replacement needs an exact live PID")
    gap_path = Path(old_state["predecessor"]).resolve(strict=True)
    if gap_path != Path(new_state["predecessor"]).resolve(strict=True):
        raise RuntimeError("old and new queues do not share the same GAP predecessor")
    gap_state = _read(gap_path)
    gap_pid = gap_state.get("pid")
    if type(gap_pid) is not int or gap_pid <= 1 or len({old_pid, new_pid, gap_pid, os.getpid()}) != 4:
        raise RuntimeError("parent, replacement, GAP, and helper PIDs must be distinct")
    old_process, new_process, gap_process = _process(old_pid), _process(new_pid), _process(gap_pid)
    _module_process(old_process, "run_partial_seam_pipeline", old_root)
    _module_process(new_process, "run_curve_hardneg_pipeline", new_root)
    _module_process(gap_process, "run_gap_stress_pipeline", gap_path.parent)
    children = _children(old_pid)
    if len(children) != 1 or children[0] in (old_pid, new_pid, gap_pid, os.getpid()):
        raise RuntimeError("old parent must have exactly one distinct tail child")
    tail_process = _process(children[0])
    _tail(tail_process, old_pid, gap_pid)
    _no_old_training(old_root)
    launch_path = old_root / "launch_state.json"
    launch_bytes = launch_path.read_bytes() if launch_path.exists() else None
    launch_state = json.loads(launch_bytes) if launch_bytes is not None else None
    parent_stopped, term_started = False, False
    receipt = dict(schema_version="rachel-waiting-partial-supersession/1", status="checking",
        started_at=_now(), helper_pid=os.getpid(), old_root=str(old_root), replacement_root=str(new_root),
        old_pid=old_pid, old_tail_pid=tail_process["pid"], replacement_pid=new_pid,
        watched_gap_pid=gap_pid, watched_gap_signalled=False, e2_or_gap_modified=False,
        old_parent_process=old_process, old_tail_process=tail_process, signals=[])
    try:
        for process in (old_process, tail_process):
            _same_process(process)
            if not _owned_alive(process):
                raise RuntimeError("verified old process exited before supersession")
        if not _send_owned(old_process, signal.SIGSTOP):
            raise RuntimeError("old parent exited before it could be stopped")
        parent_stopped = True
        receipt["signals"].append(dict(pid=old_pid, signal="SIGSTOP"))
        deadline = time.monotonic() + 2
        while _same_process(old_process)["state"] not in ("T", "t"):
            if time.monotonic() >= deadline:
                raise RuntimeError("old parent did not stop promptly")
            time.sleep(.01)
        # The old parent cannot advance from its waiter while these guards run.
        if old_path.read_bytes() != old_bytes:
            raise RuntimeError("old queue state changed before final stopped-parent check")
        _queued(_read(old_path), OLD_STAGES, {"waiting_for_gap"}, old_pid)
        _queued(_read(new_path), NEW_STAGES, {"waiting_for_preparation", "waiting_for_gap"}, new_pid)
        _same_process(new_process)
        _same_process(gap_process)
        if _children(old_pid) != children:
            raise RuntimeError("old child tree changed; no process will be terminated")
        _tail(_same_process(tail_process), old_pid, gap_pid)
        _no_old_training(old_root)
        # TERM is queued while stopped; CONT lets the pending termination run.
        # The parent is confirmed dead before its tail is unblocked/terminated.
        if not _send_owned(old_process, signal.SIGTERM):
            raise RuntimeError("old parent exited before verified termination")
        term_started = True
        receipt["signals"].append(dict(pid=old_pid, signal="SIGTERM"))
        _send_owned(old_process, signal.SIGCONT)
        parent_stopped = False
        receipt["signals"].append(dict(pid=old_pid, signal="SIGCONT"))
        if not _wait_exit(old_process):
            raise RuntimeError("old parent did not terminate; tail is deliberately left blocked")
        if _send_owned(tail_process, signal.SIGTERM):
            receipt["signals"].append(dict(pid=tail_process["pid"], signal="SIGTERM"))
        if not _wait_exit(tail_process):
            raise RuntimeError("old tail did not terminate")
        _no_old_training(old_root)
        if old_path.read_bytes() != old_bytes:
            raise RuntimeError("old state changed unexpectedly; refusing to overwrite it")
        completed = _now()
        for stage in old_state["stages"]:
            stage.update(previous_status=stage["status"], status="cancelled_before_start", cancelled_at=completed)
        old_state.update(previous_status=old_state["status"], status="superseded", active_stage=None,
                         replacement_root=str(new_root), superseded_at=completed,
                         supersession_receipt=str(receipt_path), training_started=False)
        _write(old_path, old_state)
        launch_updated = False
        if launch_state is not None and launch_state.get("pid") == old_pid:
            if launch_path.read_bytes() != launch_bytes:
                raise RuntimeError("old launcher state changed; refusing to overwrite it")
            launch_state.update(previous_status=launch_state.get("status"), status="superseded",
                                replacement_root=str(new_root), superseded_at=completed)
            _write(launch_path, launch_state)
            launch_updated = True
        receipt.update(status="superseded", completed_at=completed, old_parent_terminated=True,
                       old_tail_terminated=True, old_live_training_pids=[], cancelled_before_start=6,
                       old_launch_state_updated=launch_updated, original_artifacts_preserved=True)
        _write(receipt_path, receipt)
        return receipt
    except Exception as error:
        if parent_stopped:
            try:
                _send_owned(old_process, signal.SIGCONT)
                receipt["signals"].append(dict(pid=old_pid, signal="SIGCONT", reason="restore_after_abort"))
            except ProcessLookupError:
                pass
            parent_stopped = False
        if receipt["signals"]:
            receipt.update(status="failed", failed_at=_now(), error=repr(error), termination_started=term_started)
            _write(receipt_path, receipt)
        raise
def finalize_terminated(args):
    """Complete receipts after termination succeeded but exit observation raced."""
    old_root, new_root = Path(args.old_root).resolve(), Path(args.new_root).resolve()
    path = new_root / "supersession.json"
    receipt = _read(path)
    if receipt.get("status") != "failed" or not receipt.get("termination_started") or receipt["old_pid"] != args.old_pid:
        raise RuntimeError("not a failed post-termination receipt")
    for key in ("old_parent_process", "old_tail_process"):
        if _owned_alive(receipt[key]):
            raise RuntimeError("old process still lives; receipt-only finalization refused")
    new = _read(new_root / "pipeline_state.json")
    _queued(new, NEW_STAGES, {"waiting_for_preparation", "waiting_for_gap"}, receipt["replacement_pid"])
    _module_process(_process(new["pid"]), "run_curve_hardneg_pipeline", new_root)
    gap_path = Path(new["predecessor"])
    _module_process(_process(receipt["watched_gap_pid"]), "run_gap_stress_pipeline", gap_path.parent)
    old = _read(old_root / "pipeline_state.json")
    _queued(old, OLD_STAGES, {"waiting_for_gap"}, args.old_pid)
    _no_old_training(old_root)
    timestamp = _now()
    for stage in old["stages"]:
        stage.update(previous_status="queued", status="cancelled_before_start", cancelled_at=timestamp)
    old.update(previous_status="waiting_for_gap", status="superseded", active_stage=None,
        replacement_root=str(new_root), superseded_at=timestamp, supersession_receipt=str(path), training_started=False)
    _write(old_root / "pipeline_state.json", old)
    launch_path = old_root / "launch_state.json"
    launch = _read(launch_path)
    if launch.get("pid") == args.old_pid:
        launch.update(previous_status=launch.get("status"), status="superseded",
            replacement_root=str(new_root), superseded_at=timestamp)
        _write(launch_path, launch)
    receipt.update(status="superseded", initial_receipt_error=receipt.pop("error", None),
        completed_at=timestamp, finalized_without_additional_signals=True, old_parent_terminated=True,
        old_tail_terminated=True, old_live_training_pids=[], cancelled_before_start=6,
        old_launch_state_updated=launch.get("pid")==args.old_pid, original_artifacts_preserved=True)
    _write(path, receipt)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--old-root", required=True)
    result.add_argument("--new-root", required=True)
    result.add_argument("--old-pid", type=int, required=True)
    result.add_argument("--finalize-terminated", action="store_true")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps((finalize_terminated if args.finalize_terminated else supersede)(args), sort_keys=True))

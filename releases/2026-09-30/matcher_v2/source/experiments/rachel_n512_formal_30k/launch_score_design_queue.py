"""Launch one finite, resumable score-design queue detached from SSH.

The existing queue owns its lock, child handles and completion checks. This
launcher never kills an existing process or retries an ambiguous live job.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def launch(config_path, *, resume=False):
    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text())
    root, source = Path(config["root"]), Path(config["source"])
    if not root.is_dir() or not source.is_dir():
        raise ValueError("existing task-owned queue root/source required")
    state_path = root / "queue_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        pid = state.get("pid")
        command = Path("/proc") / str(pid) / "cmdline"
        if command.exists() and b"run_recall_benchmark_queue" in command.read_bytes():
            print(json.dumps(dict(status="already_live", pid=pid, state_path=str(state_path))))
            return
        if not resume:
            raise RuntimeError("queue state exists; inspect its process/children before explicit resume")
    command = [sys.executable, "-m", "experiments.rachel_n512_formal_30k.run_recall_benchmark_queue",
               "--config", str(config_path)]
    if resume:
        command.append("--resume")
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    log_path = root / "queue.log"
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    receipt = dict(status="dispatched", pid=process.pid, command=command,
                   launched_at=datetime.now(timezone.utc).isoformat(), log=str(log_path),
                   config=str(config_path), resume=resume)
    temporary = root / "launch.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, root / "launch.json")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    launch(args.config, resume=args.resume)

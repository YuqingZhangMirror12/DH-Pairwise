"""Exclusive lease of a user-assigned physical GPU; no global GPU takeover."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess


@contextmanager
def lease(uuid, root):
    if not uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("one explicit full physical GPU UUID required")
    root=Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root/(uuid+".lock")).open("a+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=subprocess.check_output(["nvidia-smi","-i",uuid,"--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"],text=True)
        for line in raw.splitlines():
            observed,pid=[v.strip() for v in line.split(",")]
            if observed==uuid and int(pid)!=os.getpid():
                raise RuntimeError("assigned GPU already has a foreign compute process")
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)

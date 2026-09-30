"""Finite diagnostic launcher; reuses the existing physical-GPU exclusive lease.

No training dispatch, retry, live wrapper edit, or changes to experiment plans.
Select visibility and acquire ownership before importing Torch/the diagnostic.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--lock-root", required=True)
    parser.add_argument("--receipt", required=True)
    args, remainder = parser.parse_known_args()
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    from device_runtime import DeviceLease, inventory, resolve_gpu, visible_environment, save
    if "torch" in sys.modules:
        raise RuntimeError("GPU visibility must be selected before importing Torch")
    gpu = resolve_gpu(args.gpu, inventory())
    os.environ.update(visible_environment(gpu))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    sys.dont_write_bytecode = True
    receipt = Path(args.receipt).resolve()
    if receipt.exists():
        raise FileExistsError("new receipt required, never auto-resume")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = dict(schema="hard-scorer-diagnostic-gpu-lease/1", status="starting", pid=os.getpid(),
                  physical_gpu=gpu, training=False, arguments=remainder)
    save(receipt, status)
    try:
        with DeviceLease(gpu["uuid"], args.lock_root).gpu_lock():
            import evaluate_hard_scorers as diagnostic
            status["status"] = "running"
            save(receipt, status)
            result = diagnostic.run(diagnostic.parser().parse_args(remainder))
        status.update(status="complete", result=result)
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        status["elapsed_s"] = time.monotonic()-started
        save(receipt, status)
    print(json.dumps(status, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

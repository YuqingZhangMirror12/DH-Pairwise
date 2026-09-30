"""One finite execution-input diagnostic under the existing GPU UUID lease."""
import argparse
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
        raise RuntimeError("Select GPU ownership before importing Torch")
    gpu = resolve_gpu(args.gpu, inventory())
    os.environ.update(visible_environment(gpu))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    sys.dont_write_bytecode = True
    receipt = Path(args.receipt).resolve()
    if receipt.exists():
        raise FileExistsError("New receipt required; no automatic retry")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = dict(schema="execution-input-gpu-lease/1", status="starting",
                  pid=os.getpid(), physical_gpu=gpu, arguments=remainder,
                  training=False)
    save(receipt, status)
    try:
        with DeviceLease(gpu["uuid"], args.lock_root).gpu_lock():
            import probe_execution_inputs as diagnostic
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


if __name__ == "__main__":
    main()

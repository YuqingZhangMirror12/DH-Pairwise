"""One GPU leaf on one physical UUID; experiment source/math stay unchanged.

Run this wrapper separately for EVERY train/smoke/evaluate leaf. Queue and
supervisor entrypoints are intentionally unsupported: Python monkeypatches do
not propagate to subprocess children. CPU cache preparation is also separate.
The wrapper owns a UUID-specific flock, patches only GPU lock functions and
their imported aliases, and calls the original module's parser/run entrypoint.
Do not use runpy here: reexecuting candidate_local.train as __main__ would
redefine its original global GPU lock after patching the imported module.
"""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

if __package__:
    from .transfer_registry import original_dispatch_gate
else:
    from transfer_registry import original_dispatch_gate

SCHEMA = "rachel-assigned-physical-gpu-runtime/1"
PREFIX = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919."
UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
SUPPORTED = {
    "candidate_local.train", "candidate_local.evaluate",
    "continuation.continue_classifier", "continuation.evaluate_continuation",
    "continuation_depth_controls.continue_depth_controls",
    "continuation_depth_controls.evaluate_depth_controls",
    "matched_only.train", "matched_only.smoke", "matched_only.evaluate",
    "scorer_feature_adaptation_v1.train", "scorer_feature_adaptation_v1.evaluate",
    "spectral_training.train", "spectral_training.evaluate",
    "matcher_convergence.train_continuation", "matcher_convergence.evaluate_continuation",
    "matcher_convergence.evaluate_matcher_simval", "matcher_convergence.scorer_bridge",
}
LOCK_FUNCTIONS = {
    "candidate_local.train": "gpu_lock",
    "continuation_depth_controls.continue_depth_controls": "gpu_lock",
    "matcher_convergence.evaluate_matcher_simval": "exclusive_gpu",
}


def short_name(module):
    return module[len(PREFIX):] if module.startswith(PREFIX) else module


def check_entrypoint(module, arguments):
    name = short_name(module)
    if name not in SUPPORTED:
        raise ValueError("unsupported GPU leaf; flatten queues and wrap each leaf: " + module)
    if name == "matcher_convergence.scorer_bridge" and (
            not arguments or arguments[0] not in ("train", "evaluate")):
        raise ValueError("Scorer bridge GPU wrapper supports train/evaluate only; cache stays CPU")
    return name


def inventory(check_output=subprocess.check_output):
    raw = check_output(["nvidia-smi", "--query-gpu=index,uuid,name",
                        "--format=csv,noheader,nounits"], text=True)
    rows = []
    for cells in csv.reader(raw.splitlines()):
        if not cells:
            continue
        if len(cells) != 3:
            raise ValueError("invalid physical GPU inventory row")
        index, uuid, name = (x.strip() for x in cells)
        if not index.isdecimal() or not UUID_RE.fullmatch(uuid):
            raise ValueError("physical GPU index/full UUID required; MIG not supported")
        rows.append(dict(index=int(index), uuid=uuid, name=name))
    if not rows or len({r["index"] for r in rows}) != len(rows) or len({r["uuid"] for r in rows}) != len(rows):
        raise ValueError("empty or ambiguous physical GPU inventory")
    return rows


def resolve_gpu(selector, rows):
    selector = str(selector)
    matches = [r for r in rows if selector == r["uuid"] or selector == str(r["index"])]
    if len(matches) != 1:
        raise ValueError("select exactly one physical GPU index or full UUID")
    return dict(matches[0])


def visible_environment(gpu, environ=None):
    result = dict(os.environ if environ is None else environ)
    result.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                  PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    return result


def foreign_pids(uuid, *, pid=None, check_output=subprocess.check_output):
    """Scope BOTH query and returned rows; other physical GPUs are irrelevant."""
    raw = check_output(["nvidia-smi", "-i", uuid, "--query-compute-apps=gpu_uuid,pid",
                        "--format=csv,noheader,nounits"], text=True)
    own = os.getpid() if pid is None else pid
    foreign = set()
    for cells in csv.reader(raw.splitlines()):
        if not cells:
            continue
        if len(cells) != 2:
            raise RuntimeError("cannot parse assigned-GPU compute process query")
        observed_uuid, observed_pid = (x.strip() for x in cells)
        if observed_uuid != uuid:
            continue
        if not observed_pid.isdecimal() or int(observed_pid) <= 0:
            raise RuntimeError("assigned-GPU process PID unavailable; refusing overlap")
        if int(observed_pid) != own:
            foreign.add(int(observed_pid))
    return sorted(foreign)


class DeviceLease:
    """Nonblocking per-UUID lock, reentrant only within this wrapper instance."""
    def __init__(self, uuid, root, *, check_output=subprocess.check_output):
        if not UUID_RE.fullmatch(uuid):
            raise ValueError("full physical GPU UUID required for lock identity")
        self.uuid = uuid
        self.path = Path(root).resolve() / (uuid + ".lock")
        self.check_output = check_output
        self.depth = 0
        self.handle = None

    @contextmanager
    def gpu_lock(self, _legacy_path=None, *, check_gpu=True):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != self.uuid:
            raise RuntimeError("assigned physical GPU visibility changed")
        outer = self.depth == 0
        if outer:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("a+")
            try:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                self.handle.close()
                self.handle = None
                raise
        try:
            if check_gpu:
                occupied = foreign_pids(self.uuid, check_output=self.check_output)
                if occupied:
                    raise RuntimeError("assigned GPU occupied; refusing overlap: " + repr(occupied))
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1
        finally:
            if outer:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
                self.handle.close()
                self.handle = None


def project_module(name):
    return name.startswith("experiments.rachel_n512_formal_30k.") or any(
        name == group or name.startswith(group + ".") for group in
        ("matched_only", "candidate_local", "continuation", "continuation_depth_controls",
         "scorer_feature_adaptation_v1", "spectral_training", "matcher_convergence"))


def patch_lock_aliases(lease, modules=None):
    """Replace exact known lock function identities, including from-import aliases.

    Call before importing the leaf and again afterward. The Scorer bridge creates
    its private copies later, from these already scoped module references.
    """
    modules = sys.modules if modules is None else modules
    replacements = []
    scoped = lease.gpu_lock
    owner_modules = []
    for name, module in list(modules.items()):
        if module is None or short_name(name) not in LOCK_FUNCTIONS:
            continue
        key = LOCK_FUNCTIONS[short_name(name)]
        if hasattr(module, key):
            replacements.append((getattr(module, key), scoped))
            owner_modules.append(module)
    changed = []
    for name, module in list(modules.items()):
        if module is None or not project_module(name):
            continue
        for key, value in list(vars(module).items()):
            for old, new in replacements:
                if value is old:
                    setattr(module, key, new)
                    changed.append(name + "." + key)
                    break
    for module in owner_modules:
        if hasattr(module, "GPU_LOCK"):
            module.GPU_LOCK = lease.path
    return sorted(set(changed))


def invoke(module, name, arguments):
    """Original leaf CLI dispatch; no training/evaluation implementation copied."""
    if name == "matcher_convergence.scorer_bridge":
        return module.main(arguments)
    if name in ("matcher_convergence.evaluate_continuation", "matcher_convergence.evaluate_matcher_simval"):
        args = module.parser().parse_args(arguments)
        plan = module.preflight(args)
        return module.execute(args, plan) if args.execute else {k: v for k, v in plan.items() if k != "pair_ids"}
    if name in ("matched_only.evaluate", "scorer_feature_adaptation_v1.evaluate"):
        parser = module.original.parser()
        parser.add_argument("--head-budget", type=int, choices=(8, 16), default=16)
    elif name == "candidate_local.evaluate":
        parser = module.original.parser()
    elif name == "continuation.evaluate_continuation":
        parser = module.evaluator.parser()
    elif name == "continuation_depth_controls.evaluate_depth_controls":
        parser = module.base.evaluator.parser()
    else:
        parser = module.parser()
    args = parser.parse_args(arguments)
    if name == "matched_only.smoke":
        # The sealed smoke resets allocator peaks before its first .to(cuda).
        # is_available()/device_count() do not initialize the CUDA allocator;
        # on a cold process reset_peak_memory_stats can reject cuda:0. Select
        # visibility and acquire the outer UUID lease before reaching here.
        # The smoke subsequently applies its original seed before construction.
        module.torch.cuda.init()
    return module.run(args)


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run(args):
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    name = check_entrypoint(args.module, arguments)
    if "torch" in sys.modules:
        raise RuntimeError("start a fresh wrapper process; select UUID before importing Torch")
    # Future leaves in an already-running static lane must not duplicate work
    # transferred to a freed GPU. The proxy waits before CUDA or the GPU lease.
    # Dynamic workers use distinct receipts and therefore acquire ordinary
    # dispatch claims through this same gate.
    receipt = Path(args.receipt).resolve()
    transfer_root = Path(args.lock_root).resolve().parent / "dynamic_pool"
    if original_dispatch_gate(transfer_root, receipt):
        return {"status": "delegated_complete", "original_runtime_receipt": str(receipt)}
    gpu = resolve_gpu(args.gpu, inventory())
    os.environ.update(visible_environment(gpu))
    sys.dont_write_bytecode = True
    source_roots = args.source_root or [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if not source_roots:
        raise ValueError("explicit --source-root or stage PYTHONPATH is required")
    roots = [Path(p).resolve(strict=True) for p in source_roots]
    sys.path[:0] = [str(p) for p in roots]
    if any(receipt == root or root in receipt.parents for root in roots):
        raise ValueError("runtime receipt must be outside sealed source roots")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    lease = DeviceLease(gpu["uuid"], args.lock_root)
    record = dict(schema=SCHEMA, status="starting", pid=os.getpid(), gpu=gpu,
        logical_device="cuda:0", CUDA_VISIBLE_DEVICES=gpu["uuid"], lock_path=str(lease.path),
        module=args.module, arguments=arguments, source_roots=[str(p) for p in roots],
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_files_modified=False, model_math_modified=False, checkpoint_identity_modified=False,
        child_policy="one GPU leaf only; scheduler wraps every subprocess separately",
        cuda_initialization=("explicit before cold allocator reset" if name == "matched_only.smoke"
                             else "unchanged original leaf initialization"),
        started_unix=time.time())
    with receipt.open("x") as handle:
        json.dump(record, handle, indent=2)
    try:
        with lease.gpu_lock():
            owner = importlib.import_module(PREFIX + "candidate_local.train")
            if not any(root in Path(owner.__file__).resolve().parents for root in roots):
                raise RuntimeError("GPU-lock owner imported from outside supplied source roots")
            patched = patch_lock_aliases(lease)
            module = importlib.import_module(args.module)
            path = Path(module.__file__).resolve()
            if not any(root in path.parents for root in roots):
                raise RuntimeError("leaf imported from outside supplied source roots")
            patched += patch_lock_aliases(lease)
            record.update(status="running", leaf_file=str(path),
                leaf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                patched_lock_bindings=sorted(set(patched)))
            save(receipt, record)
            result = invoke(module, name, arguments)
            if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu["uuid"]:
                raise RuntimeError("leaf modified assigned GPU visibility")
            record.update(status="complete", finished_unix=time.time())
        save(receipt, record)
        return result
    except BaseException as error:
        record.update(status="failed", error=repr(error), finished_unix=time.time())
        save(receipt, record)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", "--gpu-uuid", dest="gpu", required=True,
                   help="physical full GPU UUID preferred; inventory index also accepted")
    p.add_argument("--lock-root", required=True, help="one common per-UUID lock directory across all workers")
    p.add_argument("--source-root", action="append", help="sealed import roots; otherwise explicit stage PYTHONPATH")
    p.add_argument("--receipt", required=True, help="unique external runtime JSON, never experiment protocol")
    p.add_argument("--module", required=True)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    return p


if __name__ == "__main__":
    value = run(parser().parse_args())
    if value is not None:
        print(json.dumps(value, sort_keys=True, default=str))

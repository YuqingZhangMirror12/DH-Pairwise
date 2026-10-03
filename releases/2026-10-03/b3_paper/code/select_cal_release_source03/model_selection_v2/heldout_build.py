"""One-shot new heldout plan, real CPU pilot, then full build after admission.

All source dependencies are loaded from the explicitly bound frozen runtime.
No models are loaded, no GPUs are selected, and old outputs are never edited.
The controller records actual subprocess return codes separately from reports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--parents", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--frozen-runtime", required=True, type=Path)
    parser.add_argument("--phase", choices=("controller", "plan"), default="controller")
    args = parser.parse_args()
    root, runtime = args.root.resolve(), args.frozen_runtime.resolve(strict=True)
    if not root.is_relative_to(Path("/root/autodl-tmp/model_selection_v2_20261002")):
        raise ValueError("new evaluation-only output root required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONPATH"] = str(runtime)
    sys.path.insert(0, str(runtime))
    if args.phase == "plan":
        import heldout_plan
        value = heldout_plan.build(args.catalog, args.parents, root / "plan", args.profile)
        print(json.dumps(dict(status="registered", roles=value["roles"],
                              planned_pairs_per_role=value["desired_pairs_per_role"])), flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path(__file__).with_name("heldout_plan.py"),
             Path(__file__).with_name("heldout_run.py"), Path(__file__).with_name("heldout_augment.py"),
             args.catalog, args.parents, args.profile]
    bindings = {str(path.resolve()): sha(path) for path in paths}
    inventory = dict(root=str(runtime), files={str(path.relative_to(runtime)):sha(path)
                     for path in sorted(runtime.rglob("*.py"))})
    if not inventory["files"]:
        raise ValueError("empty frozen runtime")
    save(root / "frozen_source_inventory.json", inventory)
    bindings[str(root / "frozen_source_inventory.json")] = sha(root / "frozen_source_inventory.json")
    save(root / "controller_launch.json", dict(pid=os.getpid(), started_unix=time.time(),
         frozen_runtime=str(runtime), bindings=bindings, cuda_visible_devices="",
         workers=2, threads_per_worker=1, models_loaded=False, original_outputs_changed=False))
    common = [sys.executable, str(Path(__file__).resolve()), "--root", str(root),
              "--catalog", str(args.catalog.resolve()), "--parents", str(args.parents.resolve()),
              "--profile", str(args.profile.resolve()), "--frozen-runtime", str(runtime), "--phase", "plan"]
    stages = [("plan", common),
              ("pilot", [sys.executable, str(Path(__file__).with_name("heldout_run.py")), "--root", str(root), "--pilot"]),
              ("full", [sys.executable, str(Path(__file__).with_name("heldout_run.py")), "--root", str(root)])]
    for stage, command in stages:
        started = time.time()
        result = subprocess.run(command, cwd=root, env=os.environ.copy())
        save(root / (stage + "_actual_return.json"), dict(command=command, returncode=result.returncode,
             started_unix=started, finished_unix=time.time(), bindings=bindings))
        if result.returncode:
            raise SystemExit(result.returncode)
        if stage != "plan":
            folder = "pilot_receipts" if stage == "pilot" else "build_receipts"
            wanted = "pilot_passed" if stage == "pilot" else "complete"
            receipts = {role: read(root / folder / role / "complete.json") for role in ("cal", "select")}
            if any(value["status"] != wanted for value in receipts.values()):
                save(root / (stage + "_shortfall.json"), dict(status="shortfall", stage=stage,
                     roles={role: {k: v[k] for k in ("status", "planned_quotas", "admitted_pairs")}
                            for role, v in receipts.items()}, no_next_stage_launched=True))
                raise SystemExit(2)
        if any(sha(path) != expected for path, expected in bindings.items()):
            raise ValueError("bound input/source changed")
        if any(sha(runtime / relative) != expected for relative, expected in inventory["files"].items()):
            raise ValueError("frozen runtime source changed")
        print(json.dumps(dict(stage=stage, status="passed", actual_return=0)), flush=True)
    save(root / "controller_complete.json", dict(status="curriculum_complete_pending_combined_release_audit",
         finished_unix=time.time(), bindings=bindings, no_model_selection=True, no_training=True,
         no_gpu=True, desired_curriculum_pairs_per_role=2880))


if __name__ == "__main__":
    main()

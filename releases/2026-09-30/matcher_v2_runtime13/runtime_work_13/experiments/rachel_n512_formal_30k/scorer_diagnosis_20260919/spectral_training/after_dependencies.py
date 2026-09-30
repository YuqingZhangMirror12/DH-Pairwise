"""Explicitly seal, then wait for two registered jobs before finite spectral GPU work.

No GPU lock, deployment, retries or automatic resume in this supervisor.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation_depth_controls import followup_supervisor as previous
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training import prepare_all, queue

SCHEMA = "rachel-spectral-after-dependencies/1"
SELF = queue.MODULE + "after_dependencies"
D = previous.prior.R / "scorer_diagnosis_20260919"
DEPENDENCIES = [
    dict(name="followup", pid=101639, startticks=932414692,
         schema_version=previous.SCHEMA, root=str(D / "followup_v1"),
         source_root=str(D / "followup_source_v1"), status_file="supervisor_status.json", stages=25),
    dict(name="cache", pid=102752, startticks=932472715,
         schema_version=prepare_all.SCHEMA, root=str(D / "spectral_training_v1/cache_v1"),
         source_root=str(D / "spectral_source_v1"), status_file="status.json", stages=4),
]
save, verify_source = previous.save, previous.verify_source
process_identity = previous.process_identity


def read_dependency(spec):
    root = Path(spec["root"])
    receipt = json.loads((root / spec["status_file"]).read_text())
    launch = json.loads((root / "launch_plan.json").read_text()) if spec["name"] == "followup" else receipt
    return receipt, launch


def dependency_state(spec, receipt, process, launch):
    if process is not None and process["startticks"] != spec["startticks"]:
        raise RuntimeError(spec["name"] + ": PID reused; refusing new identity")
    if (receipt.get("pid") != spec["pid"] or receipt.get("schema_version") != spec["schema_version"]
            or launch.get("schema_version") != spec["schema_version"]
            or launch.get("source_root") != spec["source_root"]):
        raise RuntimeError(spec["name"] + ": dependency source/schema/PID differs")
    if spec["name"] == "followup":
        if launch.get("output_root") != spec["root"]:
            raise RuntimeError("followup launch output differs")
        expected = launch.get("stages", [])
        actual = receipt.get("stages", [])
        if (len(expected) != 25 or len(actual) != 25
                or any(any(a.get(k) != e.get(k) for k in ("name", "kind", "command", "completion"))
                       for a, e in zip(actual, expected))):
            raise RuntimeError("followup stage ledger differs from launch plan")
    elif [s.get("name") for s in receipt.get("stages", [])] != ["train_val", "test", "real", "ood"]:
        raise RuntimeError("cache stage ledger differs")
    status = receipt.get("status")
    if status not in ("waiting_dependency", "dependency_complete", "running", "complete"):
        raise RuntimeError(spec["name"] + ": failed/interrupted/unknown status " + str(status))
    if status == "complete":
        stages = receipt.get("stages", [])
        if (receipt.get("completed_stages") != spec["stages"] or len(stages) != spec["stages"]
                or any(s.get("status") != "complete" for s in stages)):
            raise RuntimeError(spec["name"] + ": incomplete final stage ledger")
    terminated = process is None or process["state"] == "Z"
    if terminated and status != "complete":
        raise RuntimeError(spec["name"] + ": process missing/terminated without complete receipt")
    return "ready" if terminated and status == "complete" else "waiting"


def dependencies_ready():
    states = []
    for spec in DEPENDENCIES:
        receipt, launch = read_dependency(spec)
        states.append(dependency_state(spec, receipt, process_identity(spec["pid"]), launch))
    return all(state == "ready" for state in states)


def resolve_inputs():
    """Use actual CPU completion registrations, never guessed paths/digests."""
    spec = DEPENDENCIES[1]
    receipt, launch = read_dependency(spec)
    if dependency_state(spec, receipt, process_identity(spec["pid"]), launch) != "ready":
        raise RuntimeError("CPU prep must complete and exit before binding inputs")
    registrations = receipt["registrations"]
    if set(registrations) != {"train_val", "test", "real", "ood"}:
        raise ValueError("incomplete CPU prep registrations")
    training = registrations["train_val"]
    bundle, caches = queue.train.cache.load_bundle(training["bundle"], training["sha256"])
    if (set(caches) != {"train", "val"} or any(len(caches[s].records) != queue.train.cache.SPLIT_COUNTS[s]
            for s in caches)):
        raise ValueError("training cache does not cover unchanged complete population")
    registry_path, registry_sha = receipt["endpoint_registry"], receipt["endpoint_registry_sha256"]
    registry = queue.load_registry(registry_path, registry_sha)
    if registry != {s: {k: registrations[s][k] for k in ("bundle", "sha256")} for s in ("test", "real", "ood")}:
        raise ValueError("registry differs from CPU completion registrations")
    return dict(cache_bundle=training["bundle"], cache_bundle_sha=training["sha256"],
        endpoint_registry=registry_path, endpoint_registry_sha=registry_sha,
        cpu_completion_receipt=str(Path(spec["root"]) / spec["status_file"]),
        cpu_completion_sha256=queue.train.f.file_sha256(Path(spec["root"]) / spec["status_file"]),
        normalizer_sha256=queue.train.f.digest_json(bundle["normalizer"]))


def stage_plan(plan, inputs, inputs_sha):
    root, python = Path(plan["output_root"]), plan["python"]
    rows = []
    for variant in queue.train.VARIANTS:
        output = root / (variant + "_smoke32")
        rows.append(dict(name=variant + "_discard32", kind="smoke", command=[python, "-m", queue.MODULE + "train",
            "--variant", variant, "--smoke", "32", "--output", str(output),
            "--cache-bundle", inputs["cache_bundle"], "--cache-bundle-sha", inputs["cache_bundle_sha"]],
            completion=str(output / "smoke.json")))
    rows.append(dict(name="spectral_30_stage_queue", kind="queue", command=[python, "-m", SELF,
        "--run-queue", "--plan", str(root / "launch_plan.json"), "--inputs-sha", inputs_sha],
        completion=str(root / "queue_status.json")))
    return rows


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if (plan.get("schema_version") != SCHEMA or plan.get("dependencies") != DEPENDENCIES
            or not 0 < plan.get("deadline_hours", 0) <= 48):
        raise ValueError("unregistered dependency/schema/deadline")
    if (Path(plan["source_root"]).resolve() != Path(__file__).resolve().parents[4]
            or Path(plan["output_root"]).resolve() != Path(path).resolve().parent
            or plan["python"] != sys.executable):
        raise ValueError("sealed source/output/interpreter binding differs")
    verify_source(plan)
    return plan


def seal(args):
    source = Path(args.source_root).resolve(strict=True)
    if source != Path(__file__).resolve().parents[4] or source == Path(DEPENDENCIES[0]["source_root"]):
        raise ValueError("invoke from independent spectral snapshot, never sealed followup source")
    if not 0 < args.deadline_hours <= 48:
        raise ValueError("dependency wait deadline must be greater than0 and at most48 hours")
    root = Path(args.output_root).resolve()
    protected = [source] + [Path(d["root"]) for d in DEPENDENCIES]
    if any(root == p or p in root.parents for p in protected):
        raise ValueError("output must be outside source and dependency outputs")
    # Validate observed job identities now, but never wait or dispatch at seal.
    dependencies_ready()
    hashes = previous.source_hashes(source)
    root.mkdir(parents=True, exist_ok=False)
    plan = dict(schema_version=SCHEMA, dependencies=DEPENDENCIES, source_root=str(source),
        source_sha256=hashes, output_root=str(root), python=sys.executable, deadline_hours=args.deadline_hours)
    save(root / "launch_plan.json", plan)
    return dict(status="sealed_not_started", plan=str(root / "launch_plan.json"), outer_stages=4, queue_stages=30)


def run_queue(plan_path, inputs_sha):
    plan = load_plan(plan_path)
    path = Path(plan["output_root"]) / "resolved_inputs.json"
    if queue.train.f.file_sha256(path) != inputs_sha:
        raise ValueError("resolved CPU input receipt changed")
    inputs = json.loads(path.read_text())
    def guarded_popen(*args, **kwargs):
        verify_source(plan)
        return subprocess.Popen(*args, **kwargs)
    proxy = SimpleNamespace(Popen=guarded_popen, STDOUT=subprocess.STDOUT)
    run = previous.depth.private_function(queue.run, subprocess=proxy)
    return run(SimpleNamespace(source_root=plan["source_root"], output_root=plan["output_root"],
        **{key: inputs[key] for key in ("cache_bundle", "cache_bundle_sha", "endpoint_registry", "endpoint_registry_sha")}))


def validate_completion(stage):
    receipt = json.loads(Path(stage["completion"]).read_text())
    if stage["kind"] == "queue":
        if (receipt.get("schema_version") != "rachel-spectral-queue/1" or receipt.get("status") != "complete"
                or receipt.get("completed_stages") != 30 or len(receipt.get("stages", [])) != 30
                or any(s.get("status") != "complete" for s in receipt["stages"])):
            raise RuntimeError("spectral queue did not complete all30 stages")
    elif any(receipt.get(k) != v for k, v in dict(status="smoke_complete", pair_exposures=32,
            optimizer_updates=2, weights_discarded=True, no_checkpoint_written=True,
            formal_training_counted=False, frozen_base_unchanged=True).items()):
        raise RuntimeError("invalid discard32 smoke receipt")


def execute(plan_path):
    plan = load_plan(plan_path)
    root = Path(plan["output_root"])
    status_path = root / "after_dependencies_status.json"
    with (root / "after_dependencies.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            raise ValueError("already started; no automatic resume/retry/overwrite")
        started = time.time()
        deadline = started + plan["deadline_hours"] * 3600
        status = dict(schema_version=SCHEMA, status="waiting_dependencies", pid=os.getpid(),
            source_root=plan["source_root"], started_at_unix=started, dependency_deadline_unix=deadline,
            deadline_scope="dependency wait only", completed_stages=0, stages=[])
        def transition(**fields):
            status.update(fields)
            save(status_path, status)
            print(json.dumps({k: status.get(k) for k in ("status", "active_name", "completed_stages", "error")}), flush=True)
        transition()
        child = child_group = None
        try:
            while True:
                if time.time() >= deadline:
                    raise TimeoutError("dependency wait deadline reached; no dispatch")
                if dependencies_ready():
                    break
                time.sleep(min(30, max(.01, deadline - time.time())))  # intentionally silent
            inputs = resolve_inputs()
            path = root / "resolved_inputs.json"
            if path.exists():
                raise ValueError("existing input registration will not be overwritten")
            save(path, inputs)
            status["stages"] = stage_plan(plan, inputs, queue.train.f.file_sha256(path))
            transition(status="dependencies_complete")
            for index, stage in enumerate(status["stages"]):
                verify_source(plan)
                if Path(stage["completion"]).exists():
                    raise ValueError("existing child completion will not be overwritten")
                if stage["kind"] == "queue":
                    queue.validate_smokes(root, inputs["cache_bundle_sha"])
                log_path = root / ("after_%02d_%s.log" % (index + 1, stage["name"]))
                with log_path.open("x") as log:
                    child = subprocess.Popen(stage["command"], cwd=plan["source_root"], stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1"))
                    child_group = child.pid
                    stage.update(status="running", pid=child.pid, log=str(log_path))
                    transition(status="running", active_name=stage["name"], active_pid=child.pid)
                    code = child.wait()  # no deadline or periodic source hashing while child runs
                    child = None
                stage["returncode"] = code
                if code:
                    raise RuntimeError(stage["name"] + " failed; no retry")
                validate_completion(stage)
                child_group = None
                stage.update(status="complete", finished_at_unix=time.time())
                transition(completed_stages=index + 1, active_pid=None)
            transition(status="complete", active_name=None, finished_at_unix=time.time())
            return status
        except BaseException as error:
            if child_group is not None:
                try:
                    os.killpg(child_group, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if child is not None:
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child_group, signal.SIGKILL)
                        child.wait(timeout=10)
            transition(status="failed", error=repr(error), active_pid=None, finished_at_unix=time.time())
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    group = result.add_mutually_exclusive_group(required=True)
    for mode in ("seal", "execute", "run-queue"):
        group.add_argument("--" + mode, action="store_true", help=argparse.SUPPRESS if mode == "run-queue" else None)
    for name in ("source-root", "output-root", "plan", "inputs-sha"):
        result.add_argument("--" + name)
    result.add_argument("--deadline-hours", type=float, default=48)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if args.seal:
        print(json.dumps(seal(args)))
    elif args.run_queue:
        run_queue(args.plan, args.inputs_sha)
    else:
        def interrupted(signum, _frame):
            raise KeyboardInterrupt("supervisor received signal " + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        execute(args.plan)

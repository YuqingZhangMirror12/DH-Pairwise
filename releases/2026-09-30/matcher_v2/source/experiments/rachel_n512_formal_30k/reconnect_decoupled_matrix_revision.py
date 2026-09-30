"""One-shot explicit re-registration after the matrix-head BatchNorm defect.

Default is read-only planning. --execute is for the supervising root agent on
the remote server only. Exact stopped handles and frozen queue state are
rechecked; old config/state are never edited. Any error leaves the invalid C
trajectory stopped. There is deliberately no automatic rollback/SIGCONT.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

QUEUE_MODULE = "experiments.rachel_n512_formal_30k.run_recall_benchmark_queue"
TRAIN_MODULE = "experiments.rachel_n512_formal_30k.train_score_decoupled"
SCHEMA = "decoupled-matrix-head-queue-reconnect/2"
REVISION = "bn_relu_pool_v2"
SUFFIX = "_matrix_bnfix_v2_20260914"
PHYSICAL_ENV = "RACHEL_SCORE_PHYSICAL_MICROBATCH_N512"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def update_receipt(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def process(pid):
    root = Path("/proc") / str(pid)
    try:
        argv = root.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
        stat = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    if not argv or argv == [""]:
        return None
    return dict(pid=pid, argv=argv, state=stat[0], start_time_ticks=stat[19])


def option(command, name):
    if command.count(name) != 1:
        raise ValueError("expected one argument " + name)
    return command[command.index(name) + 1]


def physical_environment(pid):
    """Read only the already-scoped runtime key; never expose other env data."""
    key = (PHYSICAL_ENV + "=").encode()
    values = [item[len(key):].decode() for item in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
              if item.startswith(key)]
    if values != ["16"]:
        raise RuntimeError("expected the registered physical16/logical1/effective16 environment")
    return values[0]


def add_option(command, name, value):
    if name in command:
        raise ValueError("refusing to silently replace existing argument " + name)
    return list(command) + [name, str(value)]


def prepare_configs(snapshots, new_source, archived_training):
    """Pure transformation: first two stage commands + explicit dependency/root revision.

    All43 later inserted stages and every tail stage stay byte-equivalent as
    parsed JSON. Existing runtime dependency PID overrides are materialized
    into the new registered config, so already-completed prerequisites survive.
    """
    if len(snapshots) != 4:
        raise ValueError("requires exactly the inserted queue and three waiting tails")
    configs = []
    for item in snapshots:
        original, state = item["config"], item["state"]
        if state["config"] != original:
            raise ValueError("queue config differs from frozen state")
        clone = copy.deepcopy(original)
        clone["root"] = original["root"] + SUFFIX
        overrides = state.get("dependency_pid_overrides", {})
        known = {d["name"] for d in clone["dependencies"]}
        if not set(overrides) <= known:
            raise ValueError("unknown dependency override")
        for dependency in clone["dependencies"]:
            if dependency["name"] in overrides:
                dependency["pid"] = overrides[dependency["name"]]
        configs.append(clone)
    first, state = configs[0], snapshots[0]["state"]
    if len(first["stages"]) != 45 or len(state["stages"]) != 45:
        raise ValueError("registered inserted queue must retain all45 stages")
    names = [s["name"] for s in first["stages"]]
    if names[:2] != ["s3_matrix_smoke32", "s3_matrix_M12_C8"]:
        raise ValueError("unexpected first two S3 stage identities")
    if [s["status"] for s in state["stages"][:2]] != ["complete", "running"]:
        raise ValueError("expected completed old smoke and currently stopped S3 C training")
    if any(s["status"] != "queued" for s in state["stages"][2:]):
        raise ValueError("later inserted stage already started; manual reconciliation required")
    for item in snapshots[1:]:
        if item["state"].get("active_child") or any(s["status"] != "queued" for s in item["state"]["stages"]):
            raise ValueError("tail is not a pure waiting observer; refusing to replay work")
    first["source"] = str(new_source)
    matcher = Path(archived_training) / "epoch_012.pt"
    smoke, training = first["stages"][:2]
    smoke["command"] = add_option(smoke["command"], "--matcher-checkpoint", matcher)
    smoke["command"] = add_option(smoke["command"], "--smoke-phase", "classifier")
    training["command"] = add_option(training["command"], "--matcher-checkpoint", matcher)
    if "--resume" in training["command"]:
        raise ValueError("new repaired S3 identity must not resume defective C")
    assert first["stages"][2:] == snapshots[0]["config"]["stages"][2:]
    # Chain process handles are filled at launch. Keep exact dependency names;
    # other dependencies retain the effective overrides copied above.
    bindings = [None]
    for index in range(1, 4):
        previous_path = str(snapshots[index - 1]["path"])
        links = [d for d in configs[index]["dependencies"] if d["marker"] == previous_path]
        if len(links) != 1:
            raise ValueError("tail must depend exactly once on its immediate previous queue")
        dependency = links[0]
        dependency["marker"] = str(Path(configs[index - 1]["root"]) / "config.json")
        dependency["status_path"] = str(Path(configs[index - 1]["root"]) / "queue_state.json")
        dependency["pid"] = None  # Never executable before the preceding launch.
        bindings.append(dependency["name"])
        assert configs[index]["stages"] == snapshots[index]["config"]["stages"]
    return configs, bindings


def inspect(args):
    pids = [args.trainer_pid] + args.queue_pids
    if len(args.queue_pids) != 4 or len(set(pids)) != 5:
        raise ValueError("requires one trainer and four distinct queue parents")
    handles = []
    for index, pid in enumerate(pids):
        current = process(pid)
        expected = TRAIN_MODULE if index == 0 else QUEUE_MODULE
        if current is None or expected not in current["argv"] or current["state"] not in ("T", "t"):
            raise RuntimeError("exact expected process is not paused: " + str(pid))
        handles.append(current)
    snapshots = []
    for handle in handles[1:]:
        path = Path(option(handle["argv"], "--config")).resolve(strict=True)
        config = read(path)
        state_path = Path(config["root"]) / "queue_state.json"
        state = read(state_path)
        if state.get("pid") != handle["pid"]:
            raise RuntimeError("queue state PID differs from exact paused process")
        snapshots.append(dict(path=str(path), config=config, state=state,
            config_sha256=digest(path), state_sha256=digest(state_path)))
    if snapshots[0]["state"].get("active_child", {}).get("pid") != args.trainer_pid:
        raise RuntimeError("paused trainer is not the first queue's active child")
    training = Path(args.training).resolve(strict=True)
    if Path(option(handles[0]["argv"], "--output")).resolve() != training:
        raise RuntimeError("paused trainer output differs from explicitly scoped training path")
    status = read(training / "status.json")
    if status.get("phase") != "classifier" or status.get("pid") != args.trainer_pid:
        raise RuntimeError("expected stopped classifier phase; do not discard matcher work")
    new_source = Path(args.new_source).resolve(strict=True)
    if new_source == Path(snapshots[0]["config"]["source"]).resolve():
        raise ValueError("repaired source must have a new isolated identity/path")
    for relative in ("staging/pairwise_v0_2/models/rachel_decoupled_score.py",
                     "experiments/rachel_n512_formal_30k/train_score_decoupled.py",
                     "experiments/rachel_n512_formal_30k/run_recall_benchmark_queue.py"):
        if not (new_source / relative).is_file():
            raise ValueError("missing repaired source file " + relative)
    model_path = new_source / "staging/pairwise_v0_2/models/rachel_decoupled_score.py"
    defaults = [node.value.value for node in ast.parse(model_path.read_text()).body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        and any(isinstance(t, ast.Name) and t.id == "DEFAULT_MATRIX_HEAD_REVISION" for t in node.targets)]
    if defaults != [REVISION]:
        raise ValueError("new source default is not the explicitly registered repaired matrix revision")
    import torch
    checkpoint = torch.load(training / "epoch_012.pt", map_location="cpu", weights_only=False)
    if (checkpoint.get("epoch") != 12 or checkpoint.get("completed_segments") != 48 or
            checkpoint.get("phase") != "matcher" or checkpoint.get("global_exposure") != 288000 or
            checkpoint.get("decoupled_training_schema") != "rachel-score-decoupled-training/1"):
        raise RuntimeError("archive source is not the exact registered M12 checkpoint")
    receipt = checkpoint.get("matcher_pretraining_receipt", {})
    if receipt.get("completed_epochs") != 12 or receipt.get("pair_exposures") != 288000:
        raise RuntimeError("M12 pretraining receipt is absent/incomplete")
    for key in ("model_state_dict", "optimizer_state_dict", "rng_state"):
        if key not in checkpoint:
            raise RuntimeError("M12 recovery state missing " + key)
    del checkpoint
    work = Path(args.work).resolve()
    archived = work / "archive" / "s3_matrix" / "training"
    smoke = Path(option(snapshots[0]["config"]["stages"][0]["command"], "--output")).resolve(strict=True)
    if Path(option(snapshots[0]["config"]["stages"][1]["command"], "--output")).resolve() != training:
        raise ValueError("queue training output differs")
    configs, bindings = prepare_configs(snapshots, new_source, archived)
    for path in [archived, work / "archive" / "s3_matrix" / "smoke32", work / "reconnect_receipt.json"] + [Path(c["root"]) for c in configs]:
        if path.exists():
            raise RuntimeError("one-shot destination already exists: " + str(path))
    return dict(schema_version=SCHEMA, status="dry_plan", repair_revision=REVISION,
        work=str(work), training=str(training), archived_training=str(archived),
        original_smoke=str(smoke), archived_smoke=str(work / "archive" / "s3_matrix" / "smoke32"),
        new_source=str(new_source), handles=handles, snapshots=snapshots,
        new_configs=configs, chain_dependency_names=bindings,
        M12_sha256=digest(training / "epoch_012.pt"),
        runtime_batching=dict(physical_microbatch=16,logical_microbatch=1,effective_batch=16,
            environment_key=PHYSICAL_ENV,environment_value=physical_environment(args.trainer_pid),
            inherited_first_parent_environment=physical_environment(args.queue_pids[0]),
            applies_to="original512 and paired512 only, never step3-cap2048"),
        preserved_old_C_reason="matrix per-pair final BN followed directly by spatial mean loses input signal",
        rerun_matcher_epochs=0, rerun_classifier_epochs=8, mandatory_repaired_smoke=True,
        original_queue_config_and_state_modified=False, automatic_resume_invalid_C=False)


def verify_handle(expected):
    actual = process(expected["pid"])
    if actual is None or any(actual[k] != expected[k] for k in ("argv", "start_time_ticks")):
        raise RuntimeError("process identity changed: " + str(expected["pid"]))
    if actual["state"] not in ("T", "t"):
        raise RuntimeError("process is no longer paused: " + str(expected["pid"]))


def wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.25)
    raise TimeoutError("bounded queue-reconnection wait timed out")


def terminate_paused(expected):
    verify_handle(expected)
    # Pending SIGTERM terminates on continuation; no invalid-C recovery attempt.
    os.kill(expected["pid"], signal.SIGTERM)
    os.kill(expected["pid"], signal.SIGCONT)
    wait_for(lambda: process(expected["pid"]) is None)


def execute(args, plan):
    work = Path(plan["work"])
    work.mkdir(parents=True, exist_ok=True)
    for handle in plan["handles"]:
        verify_handle(handle)
    for index, snapshot in enumerate(plan["snapshots"]):
        if (digest(snapshot["path"]) != snapshot["config_sha256"] or
                digest(Path(snapshot["config"]["root"]) / "queue_state.json") != snapshot["state_sha256"]):
            raise RuntimeError("frozen queue config/state changed after planning")
        backup = work / "snapshots" / ("queue_%d" % index)
        backup.mkdir(parents=True, exist_ok=False)
        shutil.copy2(snapshot["path"], backup / "config.json")
        shutil.copy2(Path(snapshot["config"]["root"]) / "queue_state.json", backup / "queue_state.json")
    write_new(work / "reconnect_plan.json", plan)
    receipt = dict(schema_version=SCHEMA, status="validated_paused_targets", repair_revision=REVISION,
        old_handles=plan["handles"], M12_sha256=plan["M12_sha256"], new_queue_pids=[],
        archives=dict(training=plan["archived_training"], smoke=plan["archived_smoke"]),
        runtime_batching=plan["runtime_batching"],
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipt_path = work / "reconnect_receipt.json"
    write_new(receipt_path, receipt)
    try:
        terminate_paused(plan["handles"][0])
        for handle in reversed(plan["handles"][1:]):
            terminate_paused(handle)
        receipt["status"] = "old_invalid_C_and_observers_terminated"
        update_receipt(receipt_path, receipt)
        for source, destination in ((plan["training"], plan["archived_training"]),
                                    (plan["original_smoke"], plan["archived_smoke"])):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            Path(source).rename(destination)
        if digest(Path(plan["archived_training"]) / "epoch_012.pt") != plan["M12_sha256"]:
            raise RuntimeError("M12 archive checksum changed")
        receipt["status"] = "old_training_and_smoke_archived"
        update_receipt(receipt_path, receipt)
        replacements = []
        for index, prototype in enumerate(plan["new_configs"]):
            config = copy.deepcopy(prototype)
            if index:
                name = plan["chain_dependency_names"][index]
                matches = [d for d in config["dependencies"] if d["name"] == name]
                matches[0]["pid"] = replacements[-1]
            root = Path(config["root"])
            root.mkdir(parents=True, exist_ok=False)
            path = root / "config.json"
            write_new(path, config)
            environment = dict(os.environ, PYTHONPATH=config["source"], PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            # Preserve only the registered runtime override in the inserted
            # queue. Its reader ignores this512 override for step3-cap2048.
            environment.pop(PHYSICAL_ENV, None)
            if index == 0:
                environment[PHYSICAL_ENV] = plan["runtime_batching"]["environment_value"]
            with (root / "parent.log").open("x") as log:
                child = subprocess.Popen([args.python, "-u", "-m", QUEUE_MODULE, "--config", str(path)],
                    cwd=config["source"], env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            replacements.append(child.pid)
            receipt["new_queue_pids"] = list(replacements)
            receipt.setdefault("new_config_paths", []).append(str(path))
            update_receipt(receipt_path, receipt)
            def registered():
                if child.poll() is not None:
                    raise RuntimeError("new queue exited; inspect " + str(root / "parent.log"))
                current = read(root / "queue_state.json") if (root / "queue_state.json").exists() else {}
                return current if current.get("pid") == child.pid and current.get("status") in ("running", "waiting_for_dependency") else None
            state = wait_for(registered)
            if index == 0:
                receipt["first_active_stage"] = state.get("active_stage")
                receipt["first_child"] = state.get("active_child")
        receipt.update(status="repaired_queues_launched", original_C_automatically_resumed=False,
            original_configs_preserved=True, all_45_inserted_stages_retained=True,
            remaining_43_inserted_stage_commands_unchanged=True,
            completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        update_receipt(receipt_path, receipt)
        return receipt
    except BaseException as error:
        receipt.update(status="reconnection_error_requires_manual_attention", error=repr(error),
            automatic_rollback=False, old_invalid_C_must_not_resume=True)
        update_receipt(receipt_path, receipt)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--new-source", required=True)
    p.add_argument("--trainer-pid", type=int, required=True)
    p.add_argument("--queue-pids", nargs=4, type=int, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--execute", action="store_true")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    plan = inspect(arguments)
    result = execute(arguments, plan) if arguments.execute else dict(
        schema_version=SCHEMA,status="dry_plan",exact_paused_pids=[h["pid"] for h in plan["handles"]],
        repair_revision=REVISION,training=plan["training"],archived_training=plan["archived_training"],
        original_smoke=plan["original_smoke"],archived_smoke=plan["archived_smoke"],
        new_source=plan["new_source"],M12_sha256=plan["M12_sha256"],runtime_batching=plan["runtime_batching"],
        new_queue_roots=[c["root"] for c in plan["new_configs"]],
        stage_counts=[len(c["stages"]) for c in plan["new_configs"]],
        revised_S3_commands=[s["command"] for s in plan["new_configs"][0]["stages"][:2]],
        remaining_43_inserted_stages_unchanged=True,all_tail_stages_unchanged=True,
        original_configs_and_states_unchanged=True,execute_requires="same command plus --execute")
    print(json.dumps(result, ensure_ascii=False, indent=2))

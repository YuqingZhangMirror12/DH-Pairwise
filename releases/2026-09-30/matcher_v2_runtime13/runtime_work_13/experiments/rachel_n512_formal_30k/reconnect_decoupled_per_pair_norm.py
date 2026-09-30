"""Plan a versioned observer handoff without interrupting the live S4 trainer.

Default is read-only. The supervisor must explicitly pause only the four queue
parents before --execute. This tool never signals, restarts, archives, or
overwrites a trainer. A partial launch is recorded for manual reconciliation;
there is no automatic rollback or continuation of the old queue parents.
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
import signal
import subprocess
import time

PREFIX = "experiments.rachel_n512_formal_30k."
QUEUE_MODULE = PREFIX + "run_recall_benchmark_queue"
TRAIN_MODULE = PREFIX + "train_score_decoupled"
EVAL_MODULE = PREFIX + "evaluate_score_decoupled"
REVALIDATE_MODULE = PREFIX + "revalidate_decoupled_per_pair_norm"
SCHEMA = "decoupled-per-pair-norm-queue-reconnect/1"
REVISION = "per_pair_norm_v3"
SUFFIX = "_per_pair_norm_v3_20260914"
PHYSICAL_ENV = "RACHEL_SCORE_PHYSICAL_MICROBATCH_N512"
SELECTIONS = ("fixed_epoch", "max_f1", "recall95")
SPLITS = ("test", "real", "ood")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish complete JSON without overwriting a concurrently-created receipt.
    temporary = path.with_name(path.name + ".%d.tmp" % os.getpid())
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update_receipt(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def option(command, name):
    if command.count(name) != 1 or command.index(name) + 1 >= len(command):
        raise ValueError("expected exactly one valued argument " + name)
    return command[command.index(name) + 1]


def replace_option(command, name, value):
    option(command, name)
    result = list(command)
    result[result.index(name) + 1] = str(value)
    return result


def process(pid):
    root = Path("/proc") / str(pid)
    try:
        argv = root.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
        stat = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    if not argv or argv == [""] or stat[0] == "Z":
        return None
    return dict(pid=pid, argv=argv, state=stat[0], ppid=int(stat[1]), start_time_ticks=stat[19])


def verify_handle(expected, *, paused):
    actual = process(expected["pid"])
    if actual is None or any(actual[key] != expected[key] for key in ("argv", "start_time_ticks")):
        raise RuntimeError("exact process identity changed: " + str(expected["pid"]))
    if (actual["state"] in ("T", "t")) != paused:
        raise RuntimeError("unexpected paused state: " + str(expected["pid"]))
    return actual


def expected_names():
    names = []
    for arm in ("s3_matrix", "s4_cross_attention", "s5_control512", "s5_step3_cap2048"):
        if arm == "s5_control512":
            names.append("await_complete_step_data")
        names.extend([arm + "_smoke32", arm + "_M12_C8"])
        names.extend(arm + "_" + selection + "_" + split for selection in SELECTIONS for split in SPLITS)
    return names


def prepare_configs(snapshots, new_source, child_handle, training_status):
    """Pure, strict transformation of the registered45-stage S4-running state."""
    if len(snapshots) != 4:
        raise ValueError("requires exactly four queue snapshots")
    configs = []
    for item in snapshots:
        original, state = item["config"], item["state"]
        if state["config"] != original or len(state["stages"]) != len(original["stages"]):
            raise ValueError("queue state differs from its registered configuration")
        for definition, progress in zip(original["stages"], state["stages"]):
            if any(progress.get(key) != value for key, value in definition.items()):
                raise ValueError("stage definition changed in progress snapshot")
        clone = copy.deepcopy(original)
        clone["root"] += SUFFIX
        overrides = state.get("dependency_pid_overrides", {})
        if not set(overrides) <= {row["name"] for row in clone["dependencies"]}:
            raise ValueError("unknown dependency override")
        for dependency in clone["dependencies"]:
            dependency["pid"] = overrides.get(dependency["name"], dependency["pid"])
        configs.append(clone)
    original = snapshots[0]["config"]
    state = snapshots[0]["state"]
    if [stage["name"] for stage in original["stages"]] != expected_names():
        raise ValueError("expected the exact original45-stage S3/S4/S5 registration")
    if ([row["status"] for row in state["stages"][:13]] != ["complete"] * 12 + ["running"] or
            any(row["status"] != "queued" for row in state["stages"][13:])):
        raise ValueError("requires S3 all11 complete, S4 smoke complete, S4 training running, later queued")
    for item in snapshots[1:]:
        if (item["state"].get("active_child") or
                item["state"].get("status") != "waiting_for_dependency" or
                any(row["status"] != "queued" for row in item["state"]["stages"])):
            raise ValueError("tail must be a pure waiting observer with no started stages")
    s4_stage = original["stages"][12]
    training = option(s4_stage["command"], "--output")
    if (TRAIN_MODULE not in child_handle["argv"] or child_handle["state"] in ("T", "t", "Z") or
            option(child_handle["argv"], "--output") != training or
            option(child_handle["argv"], "--head-kind") != "cross_attention" or
            state.get("active_child", {}).get("pid") != child_handle["pid"] or
            state["stages"][12].get("pid") != child_handle["pid"] or
            training_status.get("pid") != child_handle["pid"] or
            training_status.get("status") != "running"):
        raise ValueError("exact live nonpaused S4 trainer and running training receipt required")
    source_training = option(original["stages"][1]["command"], "--output")
    target_training = str(Path(source_training).parent.parent / "s3_matrix_per_pair_norm_v3" / "training")
    template = original["stages"][2]
    dataset = option(template["command"], "--dataset")
    command = [template["command"][0], "-m", REVALIDATE_MODULE,
        "--source-training-run", source_training, "--output", target_training,
        "--dataset", dataset, "--device", "cuda:0", "--batch-size", "1", "--workers", "4"]
    revalidation = dict(name="s3_matrix_per_pair_norm_v3_revalidate_C13_C20", command=command,
        marker=target_training, completion_path=str(Path(target_training) / "status.json"),
        completion_statuses=["complete"], resume_arguments=["--resume"])
    new_evals = []
    for stage in original["stages"][2:11]:
        if EVAL_MODULE not in stage["command"] or "--baseline-evaluation" in stage["command"]:
            raise ValueError("unexpected old S3 evaluation command")
        replacement = copy.deepcopy(stage)
        replacement["name"] = stage["name"].replace("s3_matrix_", "s3_matrix_per_pair_norm_v3_", 1)
        output = str(Path(target_training).parent / "evaluation" /
            option(stage["command"], "--selection") / option(stage["command"], "--split"))
        replacement["command"] = replace_option(stage["command"], "--training-run", target_training)
        replacement["command"] = replace_option(replacement["command"], "--output", output)
        replacement["marker"] = output
        replacement["completion_path"] = str(Path(output) / "protocol.json")
        new_evals.append(replacement)
    remaining = copy.deepcopy(original["stages"][22:])
    for stage in remaining:
        if TRAIN_MODULE in stage["command"]:
            if not stage["name"].startswith(("s5_control512_", "s5_step3_cap2048_")):
                raise ValueError("unexpected pending trainer")
            if "--matrix-head-revision" in stage["command"] or "--resume" in stage["command"]:
                raise ValueError("new S5 revision requires a fresh not-yet-started command")
            stage["command"] += ["--matrix-head-revision", REVISION]
    first = configs[0]
    first["source"] = str(new_source)
    first["dependencies"] = [dict(name="existing_S4_training_complete", pid=child_handle["pid"],
        marker=training, status_path=str(Path(training) / "status.json"))]
    first["stages"] = copy.deepcopy(original["stages"][13:22]) + [revalidation] + new_evals + remaining
    bindings = [None]
    for index in range(1, 4):
        links = [row for row in configs[index]["dependencies"] if row["marker"] == snapshots[index - 1]["path"]]
        if len(links) != 1:
            raise ValueError("tail must depend exactly once on its immediate predecessor")
        link = links[0]
        link.update(pid=None, marker=str(Path(configs[index - 1]["root"]) / "config.json"),
            status_path=str(Path(configs[index - 1]["root"]) / "queue_state.json"))
        bindings.append(link["name"])
    mapping = [dict(original_name=row["name"], disposition=("already_complete_no_replay" if i < 12 else
        "running_child_preserved_no_replay" if i == 12 else "retained_pending"),
        new_name=row["name"] if i > 12 else None) for i, row in enumerate(original["stages"])]
    return configs, bindings, dict(original_stage_mapping=mapping,
        inserted_stage_names=[revalidation["name"]] + [row["name"] for row in new_evals],
        source_training=source_training, target_training=target_training,
        S4_training_restarted=False, S3_revalidation_additional_training_exposure=0,
        S5_common_revision=REVISION, original_first_dependencies=copy.deepcopy(original["dependencies"]))


def physical_environment(pid):
    key = (PHYSICAL_ENV + "=").encode()
    values = [row[len(key):].decode() for row in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
              if row.startswith(key)]
    if values != ["16"]:
        raise RuntimeError("expected registered N512 physical16 environment")
    return values[0]


def inspect(args):
    pids = list(args.queue_pids)
    if len(pids) != 4 or len(set(pids + [args.trainer_pid])) != 5:
        raise ValueError("requires four distinct parents and a distinct protected child")
    parents, snapshots = [], []
    for pid in pids:
        handle = process(pid)
        if handle is None or QUEUE_MODULE not in handle["argv"]:
            raise RuntimeError("expected queue process missing: " + str(pid))
        if args.execute and handle["state"] not in ("T", "t"):
            raise RuntimeError("execute requires explicitly paused queue parents")
        parents.append(handle)
        path = Path(option(handle["argv"], "--config")).resolve(strict=True)
        config = read(path)
        state_path = Path(config["root"]) / "queue_state.json"
        state = read(state_path)
        if state.get("pid") != pid:
            raise RuntimeError("queue receipt PID does not match process")
        snapshots.append(dict(path=str(path), config=config, state=state,
            config_sha256=digest(path), state_sha256=digest(state_path)))
    child = process(args.trainer_pid)
    if child is None or child["ppid"] != pids[0]:
        raise RuntimeError("protected S4 child is not the first parent direct live child")
    training = Path(option(child["argv"], "--output"))
    source = Path(args.new_source).resolve(strict=True)
    if source == Path(snapshots[0]["config"]["source"]).resolve():
        raise ValueError("requires a new isolated source directory")
    directory = source / "experiments/rachel_n512_formal_30k"
    for name in ("revalidate_decoupled_per_pair_norm", "evaluate_score_decoupled", "train_score_decoupled", "run_recall_benchmark_queue"):
        if not directory.joinpath(name + ".py").is_file():
            raise ValueError("missing new source module " + name)
    model = source / "staging/pairwise_v0_2/models/rachel_decoupled_score.py"
    tree = ast.parse(model.read_text())
    defaults = [node.value.value for node in tree.body if isinstance(node, ast.Assign) and
        isinstance(node.value, ast.Constant) and any(isinstance(t, ast.Name) and
        t.id == "DEFAULT_MATRIX_HEAD_REVISION" for t in node.targets)]
    if defaults != ["bn_relu_pool_v2"] or REVISION not in model.read_text():
        raise ValueError("requires explicit v3 support while preserving the v2 default")
    if "--matrix-head-revision" not in (directory / "train_score_decoupled.py").read_text():
        raise ValueError("new source trainer lacks the explicit revision CLI")
    configs, bindings, changes = prepare_configs(snapshots, source, child, read(training / "status.json"))
    work = Path(args.work).resolve()
    for path in [work / "reconnect_receipt.json", work / "reconnect_plan.json", Path(changes["target_training"]).parent] + [Path(c["root"]) for c in configs]:
        if path.exists():
            raise RuntimeError("one-shot destination already exists: " + str(path))
    for stage in snapshots[0]["config"]["stages"][13:]:
        if Path(stage["completion_path"]).exists():
            raise RuntimeError("pending stage already has completion evidence: " + stage["name"])
        if "--output" in stage["command"] and Path(option(stage["command"], "--output")).exists():
            raise RuntimeError("pending stage already has an output directory: " + stage["name"])
    return dict(schema_version=SCHEMA, status="dry_plan", work=str(work), new_source=str(source),
        parent_handles=parents, protected_child=child, snapshots=snapshots, new_configs=configs,
        chain_dependency_names=bindings, changes=changes,
        first_parent_physical_environment=physical_environment(pids[0]))


def wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.25)
    raise TimeoutError("bounded observer registration/retirement timed out")


def retire_parent(expected, *, protected_child_pid, allowed_parent_pids):
    """Only individual allowlisted queue parents; never signal a process group."""
    pid = expected["pid"]
    if pid <= 0 or pid == protected_child_pid or pid not in allowed_parent_pids or QUEUE_MODULE not in expected["argv"]:
        raise ValueError("refusing any signal outside the four queue-parent handles")
    verify_handle(expected, paused=True)
    os.kill(pid, signal.SIGTERM)
    if process(pid) is not None:
        verify_handle(expected, paused=True)
        os.kill(pid, signal.SIGCONT)
    wait_for(lambda: process(pid) is None)


def execute(args, plan):
    parents, child = plan["parent_handles"], plan["protected_child"]
    for handle in parents:
        verify_handle(handle, paused=True)
    verify_handle(child, paused=False)
    for item in plan["snapshots"]:
        if (digest(item["path"]) != item["config_sha256"] or
                digest(Path(item["config"]["root"]) / "queue_state.json") != item["state_sha256"]):
            raise RuntimeError("paused queue snapshot changed after planning")
    work = Path(plan["work"])
    write_new(work / "reconnect_plan.json", plan)
    receipt_path = work / "reconnect_receipt.json"
    receipt = dict(schema_version=SCHEMA, status="validated_paused_observers_live_child",
        parent_handles=parents, protected_child=child, new_queue_pids=[], new_config_paths=[],
        retired_parent_pids=[], changes=plan["changes"], child_signalled=False,
        original_configs_and_outputs_modified=False,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_new(receipt_path, receipt)
    try:
        for index, prototype in enumerate(plan["new_configs"]):
            config = copy.deepcopy(prototype)
            if index:
                name = plan["chain_dependency_names"][index]
                next(row for row in config["dependencies"] if row["name"] == name)["pid"] = receipt["new_queue_pids"][-1]
            root = Path(config["root"])
            root.mkdir(parents=True, exist_ok=False)
            path = root / "config.json"
            write_new(path, config)
            env = dict(os.environ, PYTHONPATH=config["source"], PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            env.pop(PHYSICAL_ENV, None)
            if index == 0:
                env[PHYSICAL_ENV] = plan["first_parent_physical_environment"]
            with (root / "parent.log").open("x") as log:
                observer = subprocess.Popen([args.python, "-u", "-m", QUEUE_MODULE, "--config", str(path)],
                    cwd=config["source"], env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            receipt["new_queue_pids"].append(observer.pid)
            receipt["new_config_paths"].append(str(path))
            update_receipt(receipt_path, receipt)
            def registered():
                if observer.poll() is not None:
                    raise RuntimeError("new observer exited; inspect " + str(root / "parent.log"))
                state_path = root / "queue_state.json"
                state = read(state_path) if state_path.exists() else {}
                return state.get("pid") == observer.pid and state.get("status") in ("running", "waiting_for_dependency")
            wait_for(registered)
        receipt["status"] = "all_replacement_observers_registered"
        update_receipt(receipt_path, receipt)
        # Old scheduler has no signal handler/child-cleanup handler. TERM of
        # each paused parent retires observation only; the S4 child is orphaned
        # and continues. Never send a signal to its process group or the child.
        for handle in reversed(parents):
            retire_parent(handle, protected_child_pid=child["pid"], allowed_parent_pids=[p["pid"] for p in parents])
            receipt["retired_parent_pids"].append(handle["pid"])
            update_receipt(receipt_path, receipt)
        actual_child = process(child["pid"])
        if actual_child is not None:
            verify_handle(child, paused=False)
            receipt["protected_child_after_handoff"] = actual_child
        else:
            training = option(child["argv"], "--output")
            if read(Path(training) / "status.json").get("status") != "complete":
                raise RuntimeError("protected child stopped without a complete training receipt")
            receipt["protected_child_finished_naturally"] = True
        receipt.update(status="replacement_queues_launched", child_restarted=False,
            all_tail_stage_commands_unchanged=True,
            completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        update_receipt(receipt_path, receipt)
        return receipt
    except BaseException as error:
        receipt.update(status="reconnection_error_requires_manual_attention", error=repr(error),
            automatic_rollback=False, never_resume_old_parents_while_replacements_exist=True)
        update_receipt(receipt_path, receipt)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--work", required=True)
    result.add_argument("--new-source", required=True)
    result.add_argument("--trainer-pid", type=int, required=True)
    result.add_argument("--queue-pids", nargs=4, type=int, required=True)
    result.add_argument("--python", required=True)
    result.add_argument("--execute", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    plan = inspect(arguments)
    result = execute(arguments, plan) if arguments.execute else dict(schema_version=SCHEMA, status="dry_plan",
        protected_child=plan["protected_child"], queue_parent_pids=[p["pid"] for p in plan["parent_handles"]],
        new_queue_roots=[c["root"] for c in plan["new_configs"]],
        stage_counts=[len(c["stages"]) for c in plan["new_configs"]], changes=plan["changes"],
        execute_requires="same command plus --execute, four parents paused, protected child live and not paused")
    print(json.dumps(result, ensure_ascii=False, indent=2))

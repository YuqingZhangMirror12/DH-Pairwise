"""Insert S6 then S7 after S5, replacing waiting observers, never trainers.

Default is a read-only plan. --hold pauses only four verified, wholly unstarted
queue observers. --execute requires that hold receipt and complete S7 data;
it publishes new immutable configs and retires only those held observers.
No existing experiment config, output, checkpoint, or source is edited.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from experiments.rachel_n512_formal_30k.insert_score_decoupled_queues import stage
from experiments.rachel_n512_formal_30k.prepare_attention_depth_ablation import (
    build_config as depth_config,
)
from experiments.rachel_n512_formal_30k.reconnect_decoupled_per_pair_norm import (
    PHYSICAL_ENV, QUEUE_MODULE, TRAIN_MODULE, EVAL_MODULE, digest, option,
    process, read, replace_option, update_receipt, wait_for, write_new,
)

SCHEMA = "s6-s7-after-s5-queue-insertion/1"
SELECTIONS = ("fixed_epoch", "max_f1", "recall95")
SPLITS = ("test", "real", "ood")


def absolute(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("an absolute path without parent traversal is required")
    return path


def dependency(name, pid, path, config_root):
    return dict(name=name, pid=pid, marker=str(path),
                status_path=str(Path(config_root) / "queue_state.json"))


def validate_waiting(snapshot):
    """Reject started stages even when a stale top-level state says waiting."""
    config, state, handle = snapshot["config"], snapshot["state"], snapshot["handle"]
    if (state.get("config") != config or state.get("pid") != handle["pid"] or
            state.get("status") != "waiting_for_dependency" or state.get("active_child") or
            len(state.get("stages", [])) != len(config["stages"])):
        raise ValueError("observer must be registered, waiting, and have no active child")
    for definition, progress in zip(config["stages"], state["stages"]):
        if (progress.get("status") != "queued" or
                any(progress.get(key) != value for key, value in definition.items()) or
                any(key in progress for key in ("pid", "started_at", "finished_at", "returncode", "attempt_history"))):
            raise ValueError("observer has a started or altered stage; do not hold it")
    if (QUEUE_MODULE not in handle["argv"] or
            option(handle["argv"], "--config") != snapshot["path"]):
        raise ValueError("queue argv does not identify the registered config")


def effective_config(snapshot):
    clone = copy.deepcopy(snapshot["config"])
    overrides = snapshot["state"].get("dependency_pid_overrides", {})
    if not set(overrides) <= {item["name"] for item in clone["dependencies"]}:
        raise ValueError("unknown dependency handle override")
    for item in clone["dependencies"]:
        item["pid"] = overrides.get(item["name"], item["pid"])
    return clone


def replace_dependency(config, old, new):
    links = [item for item in config["dependencies"] if item["marker"] == old["path"]]
    if len(links) != 1 or links[0]["pid"] != old["handle"]["pid"]:
        raise ValueError("expected exact single predecessor PID/config dependency")
    if links[0]["status_path"] != str(Path(old["config"]["root"]) / "queue_state.json"):
        raise ValueError("predecessor status path does not match its config root")
    name = links[0]["name"]
    links[0].update(new)
    return name


def without_option(command, name):
    option(command, name)
    at = command.index(name)
    return command[:at] + command[at + 2:]


def build_s7(reference, insertion_root, source, train_manifest):
    """S4 depth1 from scratch M12+C8; only the fixed TRAIN manifest changes."""
    root, source, train_manifest = map(absolute, (insertion_root, source, train_manifest))
    arm = root / "s7_augmented_full24"
    old_train = reference["stages"][1]["command"]
    command = without_option(copy.deepcopy(old_train), "--matcher-checkpoint")
    command = without_option(command, "--output")
    command = replace_option(command, "--cross-attention-depth", "1")
    command = replace_option(command, "--train-materialized-manifest", train_manifest)
    for name, value in (("--head-kind", "cross_attention"), ("--sampling", "original512"),
                        ("--microbatch", "1"), ("--physical-microbatch", "16"),
                        ("--effective-batch", "16"), ("--stop-after-epoch", "20")):
        if option(command, name) != value:
            raise ValueError("S7 reference differs from the fixed S4 design: " + name)
    smoke, training = arm / "smoke32", arm / "training"
    stages = [stage("s7_augmented_full24_smoke32", command + ["--output", str(smoke), "--smoke", "32"],
                    smoke, "smoke.json", ("smoke_complete",)),
              stage("s7_augmented_full24_M12_C8", command + ["--output", str(training)], training, resume=True)]
    for old in reference["stages"][2:11]:
        selection, split = option(old["command"], "--selection"), option(old["command"], "--split")
        output = arm / "evaluation" / selection / split
        cmd = replace_option(old["command"], "--training-run", training)
        cmd = replace_option(cmd, "--output", output)
        stages.append(stage("s7_augmented_full24_" + selection + "_" + split,
                            cmd, output, "protocol.json"))
    return dict(root=str(root / "queues/s7_augmented_full24"), source=str(source),
                dependencies=[], stages=stages)


def prepare_configs(s5, observers, *, insertion_root, s7_source, s7_train_manifest,
                    s7_readiness_path, reference_root, python):
    """Pure: observers are [old tail1, tail2, tail3, old depth2/4 tail]."""
    root = absolute(insertion_root)
    if len(observers) != 4:
        raise ValueError("exactly three original tails and one depth tail are required")
    for item in observers:
        validate_waiting(item)
    pids = [s5["handle"]["pid"]] + [item["handle"]["pid"] for item in observers]
    if len(set(pids)) != 5 or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise ValueError("protected S5 and four observer handles must be distinct")
    originals = [effective_config(item) for item in observers]
    for index, config in enumerate(originals):
        previous = s5 if index == 0 else observers[index - 1]
        probe = copy.deepcopy(config)
        replace_dependency(probe, previous, {})
    depth = originals[3]
    source = absolute(depth["source"])
    expected = depth_config(source.parent, source, reference_root=reference_root, python=python)
    if depth["stages"] != expected["stages"]:
        raise ValueError("S6 must reuse the exact registered depth2/4 commands and outputs")
    if str(s7_train_manifest) == option(depth["stages"][1]["command"], "--train-materialized-manifest"):
        raise ValueError("S7 requires a new augmented manifest, not the old Full24 manifest")
    s6 = copy.deepcopy(depth)
    s6["root"] = str(root / "queues/s6_depth_2_4")
    replace_dependency(s6, observers[2], dependency("S5_complete", s5["handle"]["pid"],
        s5["path"], s5["config"]["root"]))
    for row in s6["stages"]:
        row["name"] = row["name"].replace("s4_cross_attention_depth", "s6_depth", 1)
    s6["preparation"].update(status="registered_after_S5_before_S7", queue_priority="immediately_after_S5",
        dependencies_bound=True, depth1_retrained=False, replaces_old_depth_tail=observers[3]["path"])
    s7 = build_s7(depth, root, s7_source, s7_train_manifest)
    s7["dependencies"] = [dependency("S6_complete", None,
        Path(s6["root"]) / "config.json", s6["root"])]
    s7["preparation"] = dict(schema_version=SCHEMA, changed_axis="augmented fixed TRAIN data only",
        head_kind="cross_attention", cross_attention_depth=1, sampling="original512",
        matcher_epochs=12, classifier_epochs=8, inherited_matcher_epochs=0,
        train_manifest=str(absolute(s7_train_manifest)), readiness_path=str(absolute(s7_readiness_path)),
        total_training_pairs=24000, precision="fp32", logical_microbatch=1,
        physical_microbatch=16, effective_batch=16,
        primary_selection="fixed_epoch20", auxiliary_epoch_range=[13, 20],
        selection_population="clean SIM VAL3000 only", S6_winner_used=False)
    configs, bindings = [s6, s7], [None, "S6_complete"]
    for index, original in enumerate(originals[:3]):
        clone = copy.deepcopy(original)
        clone["root"] = str(root / ("queues/original_tail_%d" % (index + 1)))
        old_previous = s5 if index == 0 else observers[index - 1]
        new_previous = configs[-1]
        name = replace_dependency(clone, old_previous, dict(pid=None,
            marker=str(Path(new_previous["root"]) / "config.json"),
            status_path=str(Path(new_previous["root"]) / "queue_state.json")))
        configs.append(clone)
        bindings.append(name)
    destinations = [config["root"] for config in configs]
    old_roots = {s5["config"]["root"]} | {x["config"]["root"] for x in observers}
    if len(set(destinations)) != 5 or set(destinations) & old_roots:
        raise ValueError("new observer roots must not overlap existing observers")
    if any(link["pid"] in pids[1:] for config in configs for link in config["dependencies"]):
        raise ValueError("an additional dependency still points at a retiring observer")
    return dict(configs=configs, chain_dependency_names=bindings,
        mapping=dict(S5_preserved=True, S6_stage_count=22, S7_stage_count=11,
            original_tail_counts=[len(item["stages"]) for item in originals[:3]],
            old_depth_tail_replaced_not_duplicated=True, S6_output_paths_unchanged=True,
            original_tail_commands_unchanged=True, order=["S5", "S6_depth2_depth4", "S7", "original_tail1", "original_tail2", "original_tail3"]),
        retired_observer_pids=[item["handle"]["pid"] for item in observers])


def same_handle(expected, *, paused):
    current = process(expected["pid"])
    if (current is None or any(current[key] != expected[key] for key in ("argv", "start_time_ticks")) or
            (current["state"] in ("T", "t")) != paused):
        raise RuntimeError("exact process/paused identity changed: " + str(expected["pid"]))
    return current


def snapshot(pid):
    handle = process(pid)
    if handle is None or QUEUE_MODULE not in handle["argv"]:
        raise RuntimeError("queue process not found: " + str(pid))
    path = str(absolute(option(handle["argv"], "--config")))
    config = read(path)
    state_path = str(Path(config["root"]) / "queue_state.json")
    state = read(state_path)
    if state.get("pid") != pid or state.get("config") != config:
        raise RuntimeError("queue process is not bound to its persisted config/state")
    return dict(path=path, handle=handle, config=config, state=state,
        config_sha256=digest(path), state_path=state_path)


def runtime_overrides(pid):
    prefix = (PHYSICAL_ENV + "=").encode()
    values = [x[len(prefix):].decode() for x in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
              if x.startswith(prefix)]
    if len(values) > 1 or (values and (not values[0].isdigit() or int(values[0]) < 1)):
        raise ValueError("invalid registered physical microbatch environment")
    return {PHYSICAL_ENV: values[0]} if values else {}


def validate_protected(s5, child):
    same_handle(s5["handle"], paused=False)
    current = same_handle(child, paused=False)
    state = read(s5["state_path"])
    running = [row for row in state.get("stages", []) if row.get("status") == "running"]
    allowed_commands = []
    if len(running) == 1:
        allowed_commands = [running[0]["command"], running[0]["command"] + running[0].get("resume_arguments", [])]
    if (current["ppid"] != s5["handle"]["pid"] or current["argv"] not in allowed_commands or
            state.get("status") != "running" or state.get("active_child", {}).get("pid") != current["pid"]):
        raise RuntimeError("protected S5 parent/current stage child changed; do not hand off")


def validate_ready(path, manifest):
    """Small complete receipt + exact manifest identity, never test-based gating."""
    path, manifest = absolute(path), absolute(manifest)
    record = read(path)
    if (record.get("status") != "complete" or record.get("manifest") != str(manifest) or
            record.get("sample_count") != 24000 or record.get("positive_count") != 12000):
        raise ValueError("S7 readiness must certify this complete TRAIN24000 manifest")
    table = read(manifest)
    entries = table.get("entries", [])
    if (table.get("schema_version") != "rachel-materialized-e1-train/1" or
            table.get("split") != "train" or len(entries) != 24000 or
            sum(row.get("label") == 1 for row in entries) != 12000 or
            sum(row.get("label") == 0 for row in entries) != 12000 or
            any(row.get("source_row", {}).get("split") != "train" for row in entries) or
            any(not isinstance(row.get("pair_id"), str) or not row["pair_id"] for row in entries) or
            len({row.get("pair_id") for row in entries}) != 24000):
        raise ValueError("S7 manifest must be unique TRAIN24000 with 12000 positive/12000 negative pairs")
    return dict(record, verified_manifest_sha256=digest(manifest))


def inspect(args):
    s5 = snapshot(args.s5_pid)
    child = process(args.trainer_pid)
    if child is None:
        raise RuntimeError("protected trainer missing")
    validate_protected(s5, child)
    observers = [snapshot(pid) for pid in args.waiting_pids]
    if child["pid"] in args.waiting_pids:
        raise ValueError("trainer cannot be a replaceable observer")
    for item in observers:
        validate_waiting(item)
        item["runtime_overrides"] = runtime_overrides(item["handle"]["pid"])
    prepared = prepare_configs(s5, observers, insertion_root=args.insertion_root,
        s7_source=args.s7_source, s7_train_manifest=args.s7_train_manifest,
        s7_readiness_path=args.s7_readiness, reference_root=args.reference_root, python=args.python)
    root = absolute(args.insertion_root)
    for config in prepared["configs"]:
        if Path(config["root"]).exists():
            raise RuntimeError("new queue root already exists; reconcile instead of duplicate launching")
    if (root / "s7_augmented_full24").exists():
        raise RuntimeError("new S7 arm already exists; do not replay its training")
    for item in observers:
        for row in item["config"]["stages"]:
            if Path(row["completion_path"]).exists():
                raise RuntimeError("queued stage already has a completion receipt: " + row["name"])
            if "--output" in row["command"] and Path(option(row["command"], "--output")).exists():
                raise RuntimeError("queued stage already has output: " + row["name"])
    return dict(schema_version=SCHEMA, status="read_only_plan", root=str(root), s5=s5,
                protected_child=child, observers=observers, **prepared)


def producer_progress(pid, readiness_path):
    """Only this producer's CPU ticks and its small status receipt, not data scans."""
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        cpu_ticks = int(fields[11]) + int(fields[12])
    except FileNotFoundError:
        cpu_ticks = None
    path = Path(readiness_path)
    try:
        info = path.stat()
        receipt = (info.st_mtime_ns, info.st_size)
    except FileNotFoundError:
        receipt = None
    return dict(cpu_ticks=cpu_ticks, receipt=receipt)


def bind_current_s5_child(args):
    """Refresh only after data is ready; never carry a 30-minute-old child PID."""
    # Check every replaced observer first. If S5 completed and an old tail
    # started, this fails before any signal, launch, or protected-child change.
    for pid in args.waiting_pids:
        validate_waiting(snapshot(pid))
    s5 = snapshot(args.s5_pid)
    if s5["state"].get("status") != "running" or not s5["state"].get("active_child"):
        raise RuntimeError("needs_attention: S5 has completed or has no current active child; do not reorder active tails")
    refreshed = copy.copy(args)
    refreshed.trainer_pid = s5["state"]["active_child"]["pid"]
    return refreshed


def when_ready(args):
    """Bounded producer-aware auto insertion. This is explicit, not default."""
    if (not args.producer_pid or not args.producer_start_ticks or
            args.producer_pid in [args.s5_pid] + list(args.waiting_pids) or
            args.wait_timeout_seconds <= 0 or args.poll_seconds <= 0 or args.progress_timeout_seconds <= 0):
        raise ValueError("when-ready requires a distinct producer PID/start ticks and positive bounded timeouts")
    root = absolute(args.insertion_root)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "when_ready_status.json"
    with (root / ".when_ready.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt_path.exists() or (root / "insertion_receipt.json").exists() or (root / "hold_receipt.json").exists():
            raise RuntimeError("one-shot waiting/insertion receipt exists; reconcile instead of duplicate launching")
        expected = process(args.producer_pid)
        if expected is None or expected["start_time_ticks"] != str(args.producer_start_ticks):
            raise RuntimeError("expected live S7 producer identity is absent or reused")
        record = dict(schema_version=SCHEMA, status="waiting_for_S7_data", pid=os.getpid(),
            producer=expected, protected_s5_pid=args.s5_pid, observer_pids=list(args.waiting_pids),
            readiness_path=str(absolute(args.s7_readiness)), train_manifest=str(absolute(args.s7_train_manifest)),
            wait_timeout_seconds=args.wait_timeout_seconds, progress_timeout_seconds=args.progress_timeout_seconds,
            S5_or_training_signalled=False, insertion_started=False)
        write_new(receipt_path, record)
        started = time.monotonic()
        last_progress_at, previous = started, producer_progress(args.producer_pid, args.s7_readiness)
        try:
            while True:
                current = process(args.producer_pid)
                if current is not None and any(current[key] != expected[key] for key in ("argv", "start_time_ticks")):
                    raise RuntimeError("producer PID identity changed; no insertion")
                if current is not None and current["state"] in ("T", "t"):
                    raise RuntimeError("producer is paused; no unattended insertion")
                try:
                    ready = read(args.s7_readiness)
                except FileNotFoundError:
                    ready = {}
                except json.JSONDecodeError:
                    # A live producer might still be publishing its small
                    # receipt; an exited producer gets no such retry.
                    if current is None:
                        raise RuntimeError("producer exited with incomplete readiness JSON")
                    ready = {}
                if ready.get("status") == "complete":
                    validate_ready(args.s7_readiness, args.s7_train_manifest)
                    record.update(status="S7_data_complete_refreshing_S5", producer_complete=True,
                                  insertion_started=True)
                    update_receipt(receipt_path, record)
                    refreshed = bind_current_s5_child(args)
                    plan = inspect(refreshed)
                    record["protected_child_at_insertion"] = plan["protected_child"]
                    update_receipt(receipt_path, record)
                    hold(plan)
                    result = execute(refreshed, plan)
                    record.update(status="complete", insertion_status=result["status"],
                        replacements=result["replacements"], retired_pids=result["retired_pids"])
                    update_receipt(receipt_path, record)
                    return record
                if ready.get("status") in ("failed", "error") or current is None:
                    raise RuntimeError("S7 producer failed or exited without complete data; no insertion")
                now = time.monotonic()
                progress = producer_progress(args.producer_pid, args.s7_readiness)
                if progress != previous:
                    last_progress_at, previous = now, progress
                if now - started >= args.wait_timeout_seconds:
                    raise TimeoutError("bounded S7 data wait expired; no insertion")
                if now - last_progress_at >= args.progress_timeout_seconds:
                    raise TimeoutError("S7 producer/receipt made no observable progress within the limit")
                record.update(elapsed_seconds=round(now - started, 3), producer_progress=progress,
                              data_status=ready.get("status"), last_observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                update_receipt(receipt_path, record)
                time.sleep(min(args.poll_seconds, args.wait_timeout_seconds - (now - started)))
        except BaseException as error:
            record.update(status="needs_attention", error=repr(error),
                automatic_old_observer_resume=False,
                inspect_hold_and_insertion_receipts_before_recovery=True)
            update_receipt(receipt_path, record)
            raise


def refreshed_waiting(expected, *, paused):
    same_handle(expected["handle"], paused=paused)
    current = snapshot(expected["handle"]["pid"])
    validate_waiting(current)
    if current["config_sha256"] != expected["config_sha256"]:
        raise RuntimeError("immutable config changed")
    return current


def hold(plan):
    """The only STOP targets are exact, unstarted observer handles."""
    receipt_path = Path(plan["root"]) / "hold_receipt.json"
    if receipt_path.exists():
        raise RuntimeError("hold receipt already exists; inspect it before another operation")
    validate_protected(plan["s5"], plan["protected_child"])
    for item in plan["observers"]:
        refreshed_waiting(item, paused=False)
    paused = []
    try:
        for item in reversed(plan["observers"]):
            refreshed_waiting(item, paused=False)
            os.kill(item["handle"]["pid"], signal.SIGSTOP)
            paused.append(item)
            wait_for(lambda: process(item["handle"]["pid"])["state"] in ("T", "t"), timeout=5)
            refreshed_waiting(item, paused=True)
        validate_protected(plan["s5"], plan["protected_child"])
        value = dict(schema_version=SCHEMA, status="held_waiting_observers_only", observers=plan["observers"],
            protected_s5=plan["s5"]["handle"], protected_child=plan["protected_child"],
            trainer_signalled=False, held_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_new(receipt_path, value)
        return value
    except BaseException:
        # No replacements exist in hold mode. Restore only our own exact STOPs
        # if a waiting->running race occurred before the hold was committed.
        for item in paused:
            same_handle(item["handle"], paused=True)
            os.kill(item["handle"]["pid"], signal.SIGCONT)
        raise


def launch(config, python, overrides):
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=False)
    path = root / "config.json"
    write_new(path, config)
    environment = dict(os.environ, PYTHONPATH=config["source"], PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    environment.pop(PHYSICAL_ENV, None)
    environment.update(overrides)
    if environment.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("refusing to inherit the CPU-check-only empty CUDA visibility")
    with (root / "parent.log").open("x") as log:
        observer = subprocess.Popen([python, "-u", "-m", QUEUE_MODULE, "--config", str(path)],
            cwd=config["source"], env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return observer, path


def retire(item, allowed, protected):
    handle = item["handle"]
    if handle["pid"] not in allowed or handle["pid"] in protected or handle["pid"] <= 0:
        raise ValueError("refusing to retire anything except held waiting observers")
    refreshed_waiting(item, paused=True)
    os.kill(handle["pid"], signal.SIGTERM)
    if process(handle["pid"]) is not None:
        same_handle(handle, paused=True)
        os.kill(handle["pid"], signal.SIGCONT)
    wait_for(lambda: process(handle["pid"]) is None)


def execute(args, plan):
    root = Path(plan["root"])
    held = read(root / "hold_receipt.json")
    if (held.get("schema_version") != SCHEMA or held.get("status") != "held_waiting_observers_only" or
            held.get("protected_s5") != plan["s5"]["handle"] or
            held.get("protected_child") != plan["protected_child"]):
        # State letter may change R/S naturally; identity comparison below is
        # exact for argv/start ticks, but not a changing scheduler state.
        for key, current in (("protected_s5", plan["s5"]["handle"]), ("protected_child", plan["protected_child"])):
            old = held.get(key, {})
            if any(old.get(field) != current[field] for field in ("pid", "argv", "start_time_ticks")):
                raise ValueError("hold belongs to another protected process")
        if held.get("schema_version") != SCHEMA or held.get("status") != "held_waiting_observers_only":
            raise ValueError("requires an explicit successful scoped hold receipt")
    if len(held.get("observers", [])) != 4:
        raise ValueError("hold must contain exactly four observers")
    for old, current in zip(held["observers"], plan["observers"]):
        if any(old["handle"][key] != current["handle"][key] for key in ("pid", "argv", "start_time_ticks")):
            raise ValueError("observer hold identity changed")
        if (old["config_sha256"] != current["config_sha256"] or
                old.get("runtime_overrides", {}) != current.get("runtime_overrides", {})):
            raise ValueError("held observer config or registered runtime changed")
        refreshed_waiting(current, paused=True)
    validate_protected(plan["s5"], plan["protected_child"])
    readiness = validate_ready(args.s7_readiness, args.s7_train_manifest)
    for config in plan["configs"][:2]:
        source = Path(config["source"])
        for module in (QUEUE_MODULE, TRAIN_MODULE, EVAL_MODULE):
            if not source.joinpath(*module.split(".")).with_suffix(".py").is_file():
                raise ValueError("missing isolated deployment module: " + module)
    if (root / "insertion_receipt.json").exists():
        raise RuntimeError("insertion has already been attempted; never launch twice")
    write_new(root / "insertion_plan.json", plan)
    receipt_path = root / "insertion_receipt.json"
    receipt = dict(schema_version=SCHEMA, status="launching", replacements=[], retired_pids=[],
        mapping=plan["mapping"], s7_readiness=readiness, original_configs_modified=False,
        S5_parent_signalled=False, trainer_signalled=False, automatic_rollback=False)
    write_new(receipt_path, receipt)
    try:
        for index, prototype in enumerate(plan["configs"]):
            config = copy.deepcopy(prototype)
            if index:
                name = plan["chain_dependency_names"][index]
                next(row for row in config["dependencies"] if row["name"] == name)["pid"] = receipt["replacements"][-1]["pid"]
            overrides = {} if index < 2 else plan["observers"][index - 2]["runtime_overrides"]
            observer, path = launch(config, args.python, overrides)
            receipt["replacements"].append(dict(pid=observer.pid, config=str(path)))
            update_receipt(receipt_path, receipt)
            def registered():
                if observer.poll() is not None:
                    raise RuntimeError("new observer exited before registration")
                status_path = Path(config["root"]) / "queue_state.json"
                state = read(status_path) if status_path.exists() else {}
                return state.get("pid") == observer.pid and state.get("status") in ("waiting_for_dependency", "running")
            wait_for(registered)
        allowed = plan["retired_observer_pids"]
        protected = [plan["s5"]["handle"]["pid"], plan["protected_child"]["pid"]]
        for item in reversed(plan["observers"]):
            retire(item, allowed, protected)
            receipt["retired_pids"].append(item["handle"]["pid"])
            update_receipt(receipt_path, receipt)
        receipt["status"] = "S6_S7_inserted_after_S5_original_tails_preserved"
        update_receipt(receipt_path, receipt)
        return receipt
    except BaseException as error:
        receipt.update(status="partial_insertion_requires_manual_reconciliation", error=repr(error),
            never_resume_old_observers_while_replacements_exist=True)
        update_receipt(receipt_path, receipt)
        raise


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--insertion-root", required=True)
    parser.add_argument("--s5-pid", required=True, type=int)
    parser.add_argument("--trainer-pid", type=int,
                        help="exact current protected child for immediate modes; derived after readiness in --when-ready")
    parser.add_argument("--waiting-pids", required=True, type=int, nargs=4,
                        help="old tail1 tail2 tail3 and appended depth-tail PID, in this order")
    parser.add_argument("--s7-source", required=True)
    parser.add_argument("--s7-train-manifest", required=True)
    parser.add_argument("--s7-readiness", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--python", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--hold", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--when-ready", action="store_true")
    parser.add_argument("--producer-pid", type=int)
    parser.add_argument("--producer-start-ticks")
    parser.add_argument("--wait-timeout-seconds", type=float, default=21600)
    parser.add_argument("--progress-timeout-seconds", type=float, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=30)
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    if args.when_ready:
        print(json.dumps(when_ready(args), ensure_ascii=False, indent=2))
        return
    if not args.trainer_pid:
        raise ValueError("immediate modes require --trainer-pid; only --when-ready refreshes it automatically")
    plan = inspect(args)
    if args.hold:
        value = hold(plan)
    elif args.execute:
        value = execute(args, plan)
    else:
        value = dict(schema_version=SCHEMA, status="read_only_plan", mapping=plan["mapping"],
            new_queue_roots=[config["root"] for config in plan["configs"]],
            requires="separate --hold then --execute; S7 TRAIN24000 readiness required before launch")
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

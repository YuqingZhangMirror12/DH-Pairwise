"""Read-only by default: register measured batch4 only in pending S5 step3.

Execution requires four explicitly paused queue parents and preserves their
live direct child. No model/source/default, old config, or result is modified.
The existing runner only accepts dependency status ``complete``; if caught in
a smoke child, wait for the next normal stage before pausing the parents.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import subprocess

from experiments.rachel_n512_formal_30k import reconnect_decoupled_per_pair_norm as prior

SCHEMA = "decoupled-future-step3-batching-reconnect/1"
BENCHMARK_SCHEMA = "step3-cap2048-discard-batching-benchmark/2"
SUFFIX = "_s5_step_batch4_20260914"
TARGETS = ("s5_step3_cap2048_smoke32", "s5_step3_cap2048_M12_C8")
ADDITION = ["--physical-microbatch", "4", "--runtime-effective-batch", "16"]


def validate_benchmark(benchmark):
    if (benchmark.get("schema_version") != BENCHMARK_SCHEMA or benchmark.get("status") != "complete"
            or benchmark.get("matrix_head_revision") != prior.REVISION
            or benchmark.get("physical_candidates") != [1, 2, 4]
            or benchmark.get("phases") != ["matcher", "classifier"]
            or benchmark.get("effective_batch") != 16 or benchmark.get("guard_reserved_fraction") != .8
            or benchmark.get("weights_discarded") is not True
            or benchmark.get("formal_training_counted") is not False
            or benchmark.get("no_formal_batch_selected_or_changed") is not True
            or benchmark.get("capacity_evidence_only") is not True
            or benchmark.get("automatic_deployment") is not False
            or benchmark.get("stop_reason") is not None):
        raise ValueError("requires completed v3 step3 discard-only capacity evidence, not a prior formal change")
    rows = benchmark.get("results", [])
    if len(rows) != 6 or {(r.get("physical_microbatch"), r.get("phase")) for r in rows} != {
            (batch, phase) for batch in (1, 2, 4) for phase in ("matcher", "classifier")}:
        raise ValueError("requires all six distinct M/C capacity trials")
    for row in rows:
        fraction, peak, total = (row.get(k) for k in ("reserved_fraction", "peak_reserved_gpu_bytes", "total_gpu_bytes"))
        if (row.get("schema_version") != BENCHMARK_SCHEMA or row.get("status") != "complete"
                or row.get("matrix_head_revision") != prior.REVISION
                or row.get("logical_microbatch") != 1 or row.get("effective_batch") != 16
                or row.get("precision") != "fp32" or row.get("AMP") is not False
                or row.get("weights_discarded") is not True or row.get("formal_training_counted") is not False
                or row.get("same_padding_as_formal") is not True
                or row.get("source_optimizer_states_loaded_for_memory") is not True
                or not isinstance(fraction, (float, int)) or not math.isfinite(fraction) or not 0 < fraction < .8
                or type(peak) is not int or type(total) is not int or not 0 < peak <= total
                or not math.isclose(fraction, peak / total, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("capacity trial failed revision/effective-batch/memory/discard guard")
    return [copy.deepcopy(row) for row in rows if row["physical_microbatch"] == 4]


def prepare_configs(snapshots, child, *, suffix=SUFFIX):
    """Pure transformation: remove complete/current prefix, change two commands."""
    if len(snapshots) != 4 or not suffix.startswith("_") or any(c in suffix for c in "/\\"):
        raise ValueError("requires four queue snapshots and a simple versioned suffix")
    clones = []
    for index, item in enumerate(snapshots):
        config, state = item["config"], item["state"]
        if state.get("config") != config or len(state["stages"]) != len(config["stages"]):
            raise ValueError("queue snapshot differs from frozen configuration")
        for definition, progress in zip(config["stages"], state["stages"]):
            if any(progress.get(k) != v for k, v in definition.items()):
                raise ValueError("stage definition changed in progress receipt")
        if index and (state.get("active_child") or state.get("status") != "waiting_for_dependency"
                or any(row["status"] != "queued" for row in state["stages"])):
            raise ValueError("tails must be waiting parents with no started child/stage")
        clone = copy.deepcopy(config)
        clone["root"] += suffix
        overrides = state.get("dependency_pid_overrides", {})
        if not set(overrides) <= {r["name"] for r in clone["dependencies"]}:
            raise ValueError("unknown dependency override")
        for dependency in clone["dependencies"]:
            dependency["pid"] = overrides.get(dependency["name"], dependency["pid"])
        clones.append(clone)
    first, state = snapshots[0]["config"], snapshots[0]["state"]
    running = [i for i, row in enumerate(state["stages"]) if row["status"] == "running"]
    if len(running) != 1:
        raise ValueError("requires exactly one current live first-queue stage")
    position = running[0]
    current = first["stages"][position]
    if (state.get("status") != "running" or state.get("active_stage") != current["name"]
            or any(r["status"] != "complete" for r in state["stages"][:position])
            or any(r["status"] != "queued" for r in state["stages"][position + 1:])
            or current.get("completion_statuses", ["complete"]) != ["complete"]
            or not current["name"].startswith(("s3_matrix_per_pair_norm_v3_", "s5_control512_"))):
        raise ValueError("requires completed prefix, one v3-eval/control normal child, and wholly unstarted suffix; smoke cannot be a dependency")
    commands = [current["command"], current["command"] + current.get("resume_arguments", [])]
    if (child["state"] in ("T", "t", "Z") or child["argv"] not in commands
            or child["ppid"] != state["pid"] or state.get("active_child", {}).get("pid") != child["pid"]
            or state["stages"][position].get("pid") != child["pid"]
            or state.get("active_child", {}).get("marker") != current["marker"]):
        raise ValueError("exact live unpaused child differs from its current stage")
    remaining = copy.deepcopy(first["stages"][position + 1:])
    if [r["name"] for r in remaining if r["name"] in TARGETS] != list(TARGETS):
        raise ValueError("both S5 step3 smoke and training must still be unstarted")
    for row in remaining:
        if row["name"] not in TARGETS:
            continue
        command = row["command"]
        if (prior.TRAIN_MODULE not in command or "--resume" in command
                or any(flag in command for flag in ADDITION[::2])
                or prior.option(command, "--sampling") != "step3"
                or prior.option(command, "--microbatch") != "1"
                or prior.option(command, "--effective-batch") != "16"
                or prior.option(command, "--matrix-head-revision") != prior.REVISION):
            raise ValueError("pending step3 is not the unchanged v3 logical1/effective16 command")
        row["command"] += ADDITION
    clones[0]["stages"] = remaining
    clones[0]["dependencies"] = [dict(name="preserved_current_child_complete", pid=child["pid"],
        marker=current["marker"], status_path=current["completion_path"])]
    bindings = [None]
    for index in range(1, 4):
        links = [r for r in clones[index]["dependencies"] if r["marker"] == snapshots[index - 1]["path"]]
        if len(links) != 1:
            raise ValueError("tail must depend exactly once on its immediate predecessor")
        links[0].update(pid=None, marker=str(Path(clones[index - 1]["root"]) / "config.json"),
            status_path=str(Path(clones[index - 1]["root"]) / "queue_state.json"))
        bindings.append(links[0]["name"])
    return clones, bindings, dict(current_stage=copy.deepcopy(current), current_stage_index=position,
        dropped_complete_stage_names=[r["name"] for r in first["stages"][:position]],
        preserved_child_stage_name=current["name"], modified_stage_names=list(TARGETS),
        added_arguments=ADDITION, logical_microbatch=1, runtime_effective_batch=16,
        N512_physical_microbatch=16, step3_physical_microbatch=4, user_authorized_batching_optimization=True,
        model_or_algorithm_changed=False, source_runtime_modified=False, active_child_restarted=False)


def inspect(args):
    pids = args.queue_pids
    if len(set(pids)) != 4 or any(pid <= 0 for pid in pids):
        raise ValueError("four distinct positive queue parent PIDs required")
    parents, snapshots = [], []
    for pid in pids:
        handle = prior.process(pid)
        if handle is None or prior.QUEUE_MODULE not in handle["argv"]:
            raise RuntimeError("exact queue parent missing: " + str(pid))
        if args.execute and handle["state"] not in ("T", "t"):
            raise RuntimeError("execute requires all four explicitly paused parents")
        path = Path(prior.option(handle["argv"], "--config")).resolve(strict=True)
        config = prior.read(path)
        state_path = Path(config["root"]) / "queue_state.json"
        state = prior.read(state_path)
        if state.get("pid") != pid:
            raise RuntimeError("queue receipt/process PID mismatch")
        parents.append(handle)
        snapshots.append(dict(path=str(path), config=config, state=state,
            config_sha256=prior.digest(path), state_sha256=prior.digest(state_path)))
    child = prior.process(snapshots[0]["state"].get("active_child", {}).get("pid"))
    if child is None or child["pid"] in pids:
        raise RuntimeError("first parent has no distinct protected live child")
    configs, bindings, changes = prepare_configs(snapshots, child, suffix=args.suffix)
    benchmark_path = Path(args.benchmark).resolve(strict=True)
    benchmark = prior.read(benchmark_path)
    changes["measured_batch4_trials"] = validate_benchmark(benchmark)
    target = next(r for r in configs[0]["stages"] if r["name"] == TARGETS[1])
    manifest = Path(prior.option(target["command"], "--density-train-manifest")).resolve(strict=True)
    if manifest != Path(benchmark["train_manifest"]).resolve() or prior.digest(manifest) != benchmark["manifest_sha256"]:
        raise ValueError("capacity evidence is not for the registered step3 TRAIN manifest")
    work = Path(args.work).resolve()
    if work.exists() or any(Path(c["root"]).exists() for c in configs):
        raise RuntimeError("one-shot handoff output or versioned queue root already exists")
    for row in configs[0]["stages"]:
        if row["name"] in TARGETS and (Path(row["completion_path"]).exists()
                or Path(prior.option(row["command"], "--output")).exists()):
            raise RuntimeError("step3 target already has output; not a future-only change")
    for row in snapshots[0]["config"]["stages"][:changes["current_stage_index"]]:
        if prior.read(row["completion_path"]).get("status") not in row.get("completion_statuses", ["complete"]):
            raise RuntimeError("completed prefix lacks its existing completion receipt")
    return dict(schema_version=SCHEMA, status="dry_plan", work=str(work), parent_handles=parents,
        protected_child=child, snapshots=snapshots, new_configs=configs, chain_dependency_names=bindings,
        changes=changes, benchmark_path=str(benchmark_path), benchmark_sha256=prior.digest(benchmark_path),
        first_parent_physical_environment=prior.physical_environment(pids[0]))


def execute(args, plan):
    """Use the proven observer launch/retire order; no child signal or restart."""
    parents, child = plan["parent_handles"], plan["protected_child"]
    for handle in parents:
        prior.verify_handle(handle, paused=True)
    prior.verify_handle(child, paused=False)
    for item in plan["snapshots"]:
        if (prior.digest(item["path"]) != item["config_sha256"] or
                prior.digest(Path(item["config"]["root"]) / "queue_state.json") != item["state_sha256"]):
            raise RuntimeError("paused queue snapshot changed after planning")
    if prior.digest(plan["benchmark_path"]) != plan["benchmark_sha256"]:
        raise RuntimeError("benchmark receipt changed after planning")
    work = Path(plan["work"])
    prior.write_new(work / "reconnect_plan.json", plan)
    path = work / "reconnect_receipt.json"
    receipt = dict(schema_version=SCHEMA, status="validated_paused_observers_live_child",
        parent_handles=parents, protected_child=child, new_queue_pids=[], new_config_paths=[],
        retired_parent_pids=[], changes=plan["changes"], benchmark_path=plan["benchmark_path"],
        benchmark_sha256=plan["benchmark_sha256"], child_signalled=False, child_restarted=False,
        original_configs_and_outputs_modified=False, source_runtime_modified=False)
    prior.write_new(path, receipt)
    try:
        for index, prototype in enumerate(plan["new_configs"]):
            config = copy.deepcopy(prototype)
            if index:
                next(r for r in config["dependencies"] if r["name"] == plan["chain_dependency_names"][index])["pid"] = receipt["new_queue_pids"][-1]
            root = Path(config["root"])
            root.mkdir(parents=True, exist_ok=False)
            config_path = root / "config.json"
            prior.write_new(config_path, config)
            env = dict(os.environ, PYTHONPATH=config["source"], PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            env.pop(prior.PHYSICAL_ENV, None)
            if index == 0:
                env[prior.PHYSICAL_ENV] = plan["first_parent_physical_environment"]
            with (root / "parent.log").open("x") as log:
                observer = subprocess.Popen([args.python, "-u", "-m", prior.QUEUE_MODULE, "--config", str(config_path)],
                    cwd=config["source"], env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            receipt["new_queue_pids"].append(observer.pid)
            receipt["new_config_paths"].append(str(config_path))
            prior.update_receipt(path, receipt)
            def registered():
                if observer.poll() is not None:
                    raise RuntimeError("replacement observer exited; inspect " + str(root / "parent.log"))
                state_path = root / "queue_state.json"
                state = prior.read(state_path) if state_path.exists() else {}
                return state.get("pid") == observer.pid and state.get("status") in ("running", "waiting_for_dependency")
            prior.wait_for(registered)
        for handle in reversed(parents):
            prior.retire_parent(handle, protected_child_pid=child["pid"], allowed_parent_pids=[p["pid"] for p in parents])
            receipt["retired_parent_pids"].append(handle["pid"])
            prior.update_receipt(path, receipt)
        if prior.process(child["pid"]) is not None:
            prior.verify_handle(child, paused=False)
        elif prior.read(plan["changes"]["current_stage"]["completion_path"]).get("status") != "complete":
            raise RuntimeError("protected child stopped without its original complete receipt")
        else:
            receipt["protected_child_finished_naturally"] = True
        receipt.update(status="replacement_queues_launched", all_tail_stage_commands_unchanged=True,
            completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        prior.update_receipt(path, receipt)
        return receipt
    except BaseException as error:
        receipt.update(status="reconnection_error_requires_manual_attention", error=repr(error),
            automatic_rollback=False, never_resume_old_parents_while_replacements_exist=True)
        prior.update_receipt(path, receipt)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--work", required=True)
    result.add_argument("--benchmark", required=True)
    result.add_argument("--queue-pids", nargs=4, type=int, required=True)
    result.add_argument("--python", required=True)
    result.add_argument("--suffix", default=SUFFIX)
    result.add_argument("--execute", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    plan = inspect(args)
    result = execute(args, plan) if args.execute else plan
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()

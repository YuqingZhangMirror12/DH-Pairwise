"""Prepare one finite, pre-registered eight-block input queue after joint050.

No processes or future child preparers are launched here. Architecture is
candidate_dual, schedule is joint, and each formal run has twenty epochs.
Each experiment starts from the common default, not the preceding variant.
The density comparison alone retrains its paired512 control with identical
source supervision. All original preparer/freeze/evaluation gates remain in
the owned child workflows and are evaluated only when their turn is reached.
V4 uses a new parent queue; an existing V3 queue/config is never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k.prepare_score_budget_queue import PYTHON
from experiments.rachel_n512_formal_30k.prepare_score_after020_chain import (
    ACTIVE_STATUSES, build_config as after020_config, dependency_live)

ARCHITECTURE = "candidate_dual"
BUDGET = 20
SOURCE_VERSION = "v3"
MODULE_ROOT = "experiments.rachel_n512_formal_30k."
INPUT_OPTIONS = (
    ("coarse256", ("--coarse-size", "256")),
    ("coarse512", ("--coarse-size", "512")),
    ("density_v3", ()),
    ("single7", ("--single-window", "7")),
    ("single16", ("--single-window", "16")),
    ("single32", ("--single-window", "32")),
    ("single64", ("--single-window", "64")),
    ("post", ("--transport-fusion", "post")),
)


def run_name(axis):
    return "dual_joint020_" + axis


def input_options(source_density_version=SOURCE_VERSION):
    if source_density_version not in ("v3", "v4"):
        raise ValueError("input tail requires explicit supported v3/v4 density supervision")
    return tuple(("density_" + source_density_version if axis == "density_v3" else axis, options)
                 for axis, options in INPUT_OPTIONS)


def locations(root, source_density_version=SOURCE_VERSION):
    input_options(source_density_version)
    root = Path(root)
    suffix = "" if source_density_version == "v3" else "_" + source_density_version
    return dict(root=root, source=root / "source", queue=root / ("queues/after050_inputs020" + suffix),
        input_source=root / "input_source", density_source=root / "density_runtime_source",
        train_root=root / ("paired_source_density_full24_" + source_density_version) / "data",
        clean_root=root / ("paired_density_clean_eval_" + source_density_version) / "data",
        producer_source=root / ("paired_source_density_full24_" + source_density_version) / "source",
        comparison=root / "s3/candidate_dual/paired_freeze.json")


def build_config(root, after020_queue_pid, train_density_queue_pid, clean_density_queue_pid,
                 *, s3_comparison, python=PYTHON, source_density_version=SOURCE_VERSION):
    root = Path(root)
    pids = (after020_queue_pid, train_density_queue_pid, clean_density_queue_pid)
    if not root.is_absolute() or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise ValueError("absolute root and three explicit positive queue PIDs required")
    paths = locations(root, source_density_version)
    if Path(s3_comparison) != paths["comparison"]:
        raise ValueError("S3 comparison must be the exact owned candidate_dual paired_freeze.json")
    dependencies = [dict(name=name, pid=pid, marker=str(root / "queues" / name / "config.json"),
        status_path=str(root / "queues" / name / "queue_state.json")) for name, pid in zip(
            ("after020_s3_to050", "train_density_preparation_" + source_density_version,
             "clean_density_preparation_" + source_density_version), pids)]
    stages = []
    for axis, options in input_options(source_density_version):
        density = axis.startswith("density_")
        kind = "density" if density else "input"
        private_source = paths["density_source" if density else "input_source"]
        child_queue = root / "queues" / (kind + "_" + run_name(axis))
        # The existing runner has only a global cwd/PYTHONPATH. env -C plus
        # replacement PYTHONPATH prevents main-source modules shadowing these
        # independent runtimes. env execs the preparer; it never detaches it.
        command = ["/usr/bin/env", "-C", str(private_source), "PYTHONPATH=" + str(private_source),
            python, "-m", MODULE_ROOT + "prepare_score_" + kind + "_queue", "--root", str(root),
            "--architecture", ARCHITECTURE, "--budget", str(BUDGET), "--run-name", run_name(axis),
            "--s3-comparison", str(paths["comparison"])]
        if density:
            command += ["--train-root", str(paths["train_root"]), "--clean-root", str(paths["clean_root"]),
                        "--source-density-version", source_density_version]
        else:
            command += list(options)
        stages.append(dict(name="prepare_" + axis, command=command,
            marker=MODULE_ROOT + "prepare_score_" + kind + "_queue",
            completion_path=str(child_queue / "preparation.json"), completion_statuses=["prepared_not_launched"]))
        stages.append(dict(name="run_" + axis, command=[python, "-m", MODULE_ROOT + "run_recall_benchmark_queue",
            "--config", str(child_queue / "config.json")], marker=str(child_queue / "config.json"),
            completion_path=str(child_queue / "queue_state.json"), completion_statuses=["complete"],
            resume_arguments=["--resume"]))
    return dict(root=str(paths["queue"]), source=str(paths["source"]), dependencies=dependencies, stages=stages)


def read(path):
    return json.loads(Path(path).read_text())


def verify_dependency(root, dependency, source_density_version=SOURCE_VERSION):
    directory = root / "queues" / dependency["name"]
    config_path = directory / "config.json"
    config, state = read(config_path), read(directory / "queue_state.json")
    if (config.get("root") != str(directory) or state.get("config") != config
            or state.get("pid") != dependency["pid"] or state.get("status") not in ACTIVE_STATUSES):
        raise ValueError("dependency owned config/PID/status differs: " + dependency["name"])
    paths = locations(root, source_density_version)
    if dependency["name"] == "after020_s3_to050":
        handles = config.get("dependencies", [])
        if len(handles) != 1 or config != after020_config(root, handles[0].get("pid")):
            raise ValueError("dependency is not the exact registered after020 S3-to050 chain")
    else:
        if config.get("source") != str(paths["producer_source"]) or config.get("dependencies") != []:
            raise ValueError("density preparation dependency uses another producer/source")
        train = dependency["name"] == "train_density_preparation_" + source_density_version
        expected = [("prepare_full_train24000_both_caps_" + source_density_version,
            "materialize_source_density_" + source_density_version, "--output",
            paths["train_root"], paths["train_root"] / "run_state.json")] if train else [
            ("prepare_full_" + split + "3000_both_caps_" + source_density_version,
                "prepare_clean_density_" + source_density_version, "--cache-root",
                paths["clean_root"], paths["clean_root"] / ("clean_" + split + "_preparation.json"))
            for split in ("val", "test")]
        stages = config.get("stages", [])
        if len(stages) != len(expected):
            raise ValueError("density preparation dependency has wrong stage population")
        for index, (stage, (name, module, flag, output, receipt)) in enumerate(zip(stages, expected)):
            command = stage.get("command", [])
            if (stage.get("name") != name or len(command) < 3 or command[1:3] != ["-m", MODULE_ROOT + module]
                    or command.count(flag) != 1 or command[command.index(flag) + 1] != str(output)
                    or stage.get("completion_path") != str(receipt) or stage.get("completion_statuses") != ["complete"]
                    or any(arg in command for arg in ("--limit", "--smoke", "--pair-ids", "--max-pairs", "--stop-after-pairs"))):
                raise ValueError("density dependency is not the registered full-population producer")
            if not train and (command.count("--split") != 1 or
                    command[command.index("--split") + 1] != ("val", "test")[index]):
                raise ValueError("clean density dependency must prepare VAL then TEST")
    live = dependency_live(dependency["pid"], config_path)
    if state["status"] != "complete" and not live:
        raise ValueError("unfinished dependency process is stopped: " + dependency["name"])
    return dict(name=dependency["name"], config=str(config_path), pid=dependency["pid"],
        observed_status=state["status"], observed_live=live,
        execution_gate="queue status complete and process exited; original child preparer gates remain mandatory")


def future_outputs(root, source_density_version=SOURCE_VERSION):
    paths = [locations(root, source_density_version)["queue"]]
    for axis, _ in input_options(source_density_version):
        kind = "density" if axis.startswith("density_") else "input"
        paths += [root / "queues" / (kind + "_" + run_name(axis)), root / (kind + "_variants") / run_name(axis)]
    return paths


def verify_sources(root, source_density_version=SOURCE_VERSION):
    """Presence only; do not import private model code into the parent process."""
    paths = locations(root, source_density_version)
    required = [paths["source"] / "experiments/rachel_n512_formal_30k/run_recall_benchmark_queue.py"]
    for kind in ("input", "density"):
        source = paths[kind + "_source"]
        for entry in ("prepare_score_" + kind + "_queue", "train_score_" + ("input_variant" if kind == "input" else "density"),
                      "evaluate_score_" + ("input_variant" if kind == "input" else "density")):
            required.append(source / "experiments/rachel_n512_formal_30k" / (entry + ".py"))
    required.append(paths["density_source"] / "staging/pairwise_v0_2/pairwise_data" /
                    ("rachel_density_ownership_" + source_density_version + ".py"))
    if source_density_version == "v4":
        required.append(paths["density_source"] / "staging/pairwise_v0_2/pairwise_data/rachel_density_outer_certificate.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("independent runtime entry missing: " + missing[0])
    return [str(path) for path in required]


def verify_v3_predecessor_stopped(root, source_density_version):
    """Both tails share ordinary input outputs; never allow two live owners."""
    if source_density_version != "v4":
        return None
    previous = locations(root, "v3")["queue"]
    if not previous.exists():
        return None
    config_path, state_path = previous / "config.json", previous / "queue_state.json"
    config, state = read(config_path), read(state_path)
    if state.get("config") != config or config.get("root") != str(previous):
        raise ValueError("previous v3 input parent identity is inconsistent")
    pid = state.get("pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError("previous v3 input parent lacks an actual PID")
    if dependency_live(pid, config_path):
        raise ValueError("retire the still-live v3 input parent before preparing v4")
    if state.get("active_child") or any(s.get("status") not in (None, "queued") for s in state.get("stages", [])):
        raise ValueError("previous v3 parent has entered child work; inspect/resume rather than duplicate")
    return dict(config=str(config_path), pid=pid, observed_live=False,
                status_snapshot=state.get("status"), no_child_work_started=True,
                previous_config_overwritten=False)


def prepare(root, after020_queue_pid, train_density_queue_pid, clean_density_queue_pid, *, s3_comparison,
            source_density_version=SOURCE_VERSION):
    root = Path(root).resolve(strict=True)
    config = build_config(root, after020_queue_pid, train_density_queue_pid, clean_density_queue_pid,
        s3_comparison=s3_comparison, source_density_version=source_density_version)
    observed = [verify_dependency(root, row, source_density_version) for row in config["dependencies"]]
    sources = verify_sources(root, source_density_version)
    predecessor = verify_v3_predecessor_stopped(root, source_density_version)
    occupied = [str(path) for path in future_outputs(root, source_density_version) if path.exists()]
    if occupied:
        raise FileExistsError("future input output already exists; inspect/resume it instead: " + occupied[0])
    destination = Path(config["root"])
    destination.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="prepared_not_launched", dependencies=observed, architecture=ARCHITECTURE,
        training_mode="joint", budget=BUDGET, input_order=[axis for axis, _ in input_options(source_density_version)],
        all_variants_relative_to_common_default=True, axes_accumulated=False,
        parent_stage_count=16, input_child_queues=7, input_child_stage_count=8,
        density_child_queues=1, density_child_stage_count=16, formal_training_runs=9,
        formal_exposures_per_run=BUDGET * 24000, smoke_replaces_formal_training=False,
        s3_comparison=str(locations(root, source_density_version)["comparison"]),
        paired_density_version=source_density_version,
        paired_density512_control_retrained=True, historical512_used_as_density_control=False,
        child_working_directories="private input_source/density_runtime_source via env -C and replacement PYTHONPATH",
        checked_runtime_entries=sources, original_preparation_gates_unchanged=True,
        architecture_budget_choice="fixed candidate_dual joint20 before results; no S3/REAL/OOD winner selection",
        held_out_metrics_used_to_choose_next_action=False, future_child_queues_prepared=False,
        future_freezes_prepared=False, launches_processes=False)
    if source_density_version == "v4":
        receipt["v3_parent_retirement_observation"] = predecessor
    for name, value in (("config.json", config), ("preparation.json", receipt)):
        with (destination / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    result = dict(status="prepared_not_launched", config=str(destination / "config.json"))
    print(json.dumps(result))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--after020-queue-pid", type=int, required=True)
    p.add_argument("--train-density-queue-pid", type=int, required=True)
    p.add_argument("--clean-density-queue-pid", type=int, required=True)
    p.add_argument("--s3-comparison", required=True)
    p.add_argument("--source-density-version", choices=("v3", "v4"), default=SOURCE_VERSION)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    prepare(args.root, args.after020_queue_pid, args.train_density_queue_pid, args.clean_density_queue_pid,
        s3_comparison=args.s3_comparison, source_density_version=args.source_density_version)

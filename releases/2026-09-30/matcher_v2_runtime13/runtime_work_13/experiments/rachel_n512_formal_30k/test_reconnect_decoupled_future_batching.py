"""CPU synthetic queue handles; no real launch, signal, model or remote I/O."""
import argparse
import copy
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch

import pytest

from experiments.rachel_n512_formal_30k import reconnect_decoupled_future_batching as subject
from experiments.rachel_n512_formal_30k.test_reconnect_decoupled_per_pair_norm import fixture as old_fixture


def benchmark():
    result = dict(schema_version=subject.BENCHMARK_SCHEMA, status="complete",
        matrix_head_revision=subject.prior.REVISION, physical_candidates=[1, 2, 4],
        phases=["matcher", "classifier"], effective_batch=16, guard_reserved_fraction=.8,
        weights_discarded=True, formal_training_counted=False,
        no_formal_batch_selected_or_changed=True, capacity_evidence_only=True,
        automatic_deployment=False, stop_reason=None, results=[])
    for batch in (1, 2, 4):
        for phase in ("matcher", "classifier"):
            result["results"].append(dict(schema_version=subject.BENCHMARK_SCHEMA,
                status="complete", matrix_head_revision=subject.prior.REVISION,
                physical_microbatch=batch, phase=phase, logical_microbatch=1, effective_batch=16,
                precision="fp32", AMP=False, weights_discarded=True, formal_training_counted=False,
                same_padding_as_formal=True, source_optimizer_states_loaded_for_memory=True,
                total_gpu_bytes=10000, peak_reserved_gpu_bytes=7840 if batch == 4 else 4000,
                reserved_fraction=.784 if batch == 4 else .4))
    return result


def fixture(root, current_name="s3_matrix_per_pair_norm_v3_fixed_epoch_real"):
    original, child, status = old_fixture(root)
    configs, _, _ = subject.prior.prepare_configs(original, str(root / "isolated_v3_source"), child, status)
    snapshots = []
    for index, config in enumerate(configs):
        if index:
            config["dependencies"][0]["pid"] = index + 1
        state = dict(config=copy.deepcopy(config), pid=index + 1,
            status="running" if index == 0 else "waiting_for_dependency",
            stages=[dict(row, status="queued") for row in config["stages"]])
        if not index:
            position = next(i for i, row in enumerate(config["stages"]) if row["name"] == current_name)
            for row in state["stages"][:position]:
                row["status"] = "complete"
            stage = config["stages"][position]
            state["stages"][position].update(status="running", pid=9)
            state.update(active_stage=current_name, active_child=dict(pid=9, marker=stage["marker"]))
            child = dict(pid=9, ppid=1, state="R", start_time_ticks="900", argv=copy.deepcopy(stage["command"]))
        snapshots.append(dict(path=str(Path(config["root"]) / "config.json"), config=config, state=state))
    return snapshots, child


def save_fixture(root):
    snapshots, child = fixture(root)
    parents = []
    for index, item in enumerate(snapshots):
        subject.prior.write_new(item["path"], item["config"])
        state_path = Path(item["config"]["root"]) / "queue_state.json"
        subject.prior.write_new(state_path, item["state"])
        item.update(config_sha256=subject.prior.digest(item["path"]), state_sha256=subject.prior.digest(state_path))
        parents.append(dict(pid=index + 1, state="T", start_time_ticks=str(index + 100),
            argv=["python", "-m", subject.prior.QUEUE_MODULE, "--config", item["path"]]))
    for row in snapshots[0]["state"]["stages"]:
        if row["status"] == "complete":
            subject.prior.write_new(row["completion_path"], dict(status=row["completion_statuses"][0]))
    target = next(r for r in snapshots[0]["config"]["stages"] if r["name"] == subject.TARGETS[1])
    manifest = Path(subject.prior.option(target["command"], "--density-train-manifest"))
    subject.prior.write_new(manifest, {"fixture": "TRAIN step3 only"})
    proof = benchmark()
    proof.update(train_manifest=str(manifest), manifest_sha256=subject.prior.digest(manifest))
    benchmark_path = root / "benchmark.json"
    subject.prior.write_new(benchmark_path, proof)
    args = argparse.Namespace(queue_pids=[1, 2, 3, 4], benchmark=str(benchmark_path),
        work=str(root / "work"), suffix=subject.SUFFIX, python="/python", execute=False)
    return args, snapshots, parents, child


@pytest.mark.parametrize("current_name", ("s3_matrix_per_pair_norm_v3_fixed_epoch_real",
    "s3_matrix_per_pair_norm_v3_recall95_ood", "s5_control512_M12_C8", "s5_control512_max_f1_test"))
def test_only_future_step3_two_commands_change_and_current_child_not_replayed(current_name):
    snapshots, child = fixture(Path("/fixture"), current_name)
    before = copy.deepcopy(snapshots)
    configs, bindings, changes = subject.prepare_configs(snapshots, child)
    position = changes["current_stage_index"]
    assert snapshots == before
    assert configs[0]["source"] == snapshots[0]["config"]["source"]
    for new, old in zip(configs[0]["stages"], snapshots[0]["config"]["stages"][position + 1:]):
        expected = copy.deepcopy(old)
        if old["name"] in subject.TARGETS:
            expected["command"] += subject.ADDITION
        assert new == expected
    assert len(configs[0]["stages"]) == len(snapshots[0]["config"]["stages"]) - position - 1
    assert current_name not in [r["name"] for r in configs[0]["stages"]]
    dependency = configs[0]["dependencies"][0]
    assert dependency["pid"] == child["pid"]
    assert dependency["status_path"] == changes["current_stage"]["completion_path"]
    assert dependency["marker"] == changes["current_stage"]["marker"]
    assert bindings == [None, "previous", "previous", "previous"]
    for i in range(1, 4):
        assert configs[i]["stages"] == snapshots[i]["config"]["stages"]
        assert configs[i]["source"] == snapshots[i]["config"]["source"]
        assert configs[i]["dependencies"][0]["marker"] == str(Path(configs[i - 1]["root"]) / "config.json")
        assert configs[i]["dependencies"][1] == snapshots[i]["config"]["dependencies"][1]


@pytest.mark.parametrize("change", ("revision", "effective", "memory", "peak_mismatch", "missing_C4", "formal", "failed"))
def test_capacity_guard_rejects_invalid_or_incomplete_evidence(change):
    proof = benchmark()
    assert len(subject.validate_benchmark(proof)) == 2
    if change == "revision": proof["matrix_head_revision"] = "bn_relu_pool_v2"
    elif change == "effective": proof["results"][-1]["effective_batch"] = 32
    elif change == "memory": proof["results"][-2].update(reserved_fraction=.8, peak_reserved_gpu_bytes=8000)
    elif change == "peak_mismatch": proof["results"][-2]["reserved_fraction"] = .1
    elif change == "missing_C4": proof["results"].pop()
    elif change == "formal": proof["no_formal_batch_selected_or_changed"] = False
    elif change == "failed": proof["results"][-1]["status"] = "failed"
    with pytest.raises(ValueError):
        subject.validate_benchmark(proof)


@pytest.mark.parametrize("change", ("child_paused", "child_argv", "prefix", "tail_started", "duplicate_batch", "started_step3", "smoke_child"))
def test_progress_guard_rejects_unsafe_changes(change):
    snapshots, child = fixture(Path("/fixture"), "s5_control512_smoke32" if change == "smoke_child" else
        "s3_matrix_per_pair_norm_v3_fixed_epoch_real")
    if change == "child_paused": child["state"] = "T"
    elif change == "child_argv": child["argv"] += ["unregistered"]
    elif change == "prefix": snapshots[0]["state"]["stages"][0]["status"] = "queued"
    elif change == "tail_started": snapshots[1]["state"]["stages"][0]["status"] = "complete"
    elif change == "duplicate_batch":
        row = next(r for r in snapshots[0]["config"]["stages"] if r["name"] == subject.TARGETS[0])
        row["command"] += subject.ADDITION
        snapshots[0]["state"]["config"] = copy.deepcopy(snapshots[0]["config"])
    elif change == "started_step3":
        snapshots, child = fixture(Path("/fixture"), subject.TARGETS[1])
    with pytest.raises(ValueError):
        subject.prepare_configs(snapshots, child)


def test_inspect_is_read_only_and_requires_paused_parents_only_for_execute(tmp_path):
    args, snapshots, parents, child = save_fixture(tmp_path)
    processes = {h["pid"]: copy.deepcopy(h) for h in parents + [child]}
    processes[1]["state"] = "S"
    with patch.object(subject.prior, "process", side_effect=lambda pid: processes.get(pid)), \
            patch.object(subject.prior, "physical_environment", return_value="16"):
        plan = subject.inspect(args)
        assert plan["protected_child"] == child
        assert not Path(args.work).exists()
        assert all(not Path(c["root"]).exists() for c in plan["new_configs"])
        args.execute = True
        with pytest.raises(RuntimeError, match="explicitly paused"):
            subject.inspect(args)
    for item in snapshots:
        assert subject.prior.digest(item["path"]) == item["config_sha256"]


@pytest.mark.parametrize("natural_completion", (False, True))
def test_execute_mock_signals_only_four_parents_and_preserves_live_or_naturally_finished_eval(tmp_path, natural_completion):
    args, snapshots, parents, child = save_fixture(tmp_path)
    args.execute = True
    processes = {h["pid"]: copy.deepcopy(h) for h in parents + [child]}
    launched, signals = [], []
    with patch.object(subject.prior, "process", side_effect=lambda pid: processes.get(pid)), \
            patch.object(subject.prior, "physical_environment", return_value="16"):
        plan = subject.inspect(args)

        def launch(command, **kwargs):
            pid = 1001 + len(launched)
            config = subject.prior.read(subject.prior.option(command, "--config"))
            launched.append((config, kwargs))
            subject.prior.write_new(Path(config["root"]) / "queue_state.json",
                dict(pid=pid, status="waiting_for_dependency"))
            return argparse.Namespace(pid=pid, poll=lambda: None)

        def kill(pid, sig):
            assert len(launched) == 4 and pid in (1, 2, 3, 4)
            signals.append((pid, sig))
            if sig == signal.SIGCONT:
                processes.pop(pid)
                if pid == 1 and natural_completion:
                    processes.pop(child["pid"])
                    subject.prior.write_new(plan["changes"]["current_stage"]["completion_path"], {"status": "complete"})

        with patch.object(subject.subprocess, "Popen", side_effect=launch), patch.object(subject.prior.os, "kill", side_effect=kill):
            result = subject.execute(args, plan)
    assert result["status"] == "replacement_queues_launched"
    assert result["retired_parent_pids"] == [4, 3, 2, 1]
    assert signals == [(pid, sig) for pid in (4, 3, 2, 1) for sig in (signal.SIGTERM, signal.SIGCONT)]
    assert result["child_signalled"] is result["child_restarted"] is False
    assert result.get("protected_child_finished_naturally", False) == natural_completion
    assert launched[0][1]["env"][subject.prior.PHYSICAL_ENV] == "16"
    for i in range(1, 4):
        assert launched[i][0]["dependencies"][0]["pid"] == 1000 + i
        assert subject.prior.PHYSICAL_ENV not in launched[i][1]["env"]
    for item in snapshots:
        assert subject.prior.digest(item["path"]) == item["config_sha256"]
        assert subject.prior.digest(Path(item["config"]["root"]) / "queue_state.json") == item["state_sha256"]


def test_parser_and_help_are_cpu_only_and_execute_explicit():
    args = subject.parser().parse_args(["--work", "/work", "--benchmark", "/benchmark.json",
        "--queue-pids", "1", "2", "3", "4", "--python", "/python"])
    assert not args.execute
    subprocess.run([sys.executable, "-c", "from experiments.rachel_n512_formal_30k import reconnect_decoupled_future_batching; import sys; assert 'torch' not in sys.modules"], check=True)
    result = subprocess.run([sys.executable, "-m", subject.__name__, "--help"], check=True, capture_output=True, text=True)
    assert "--benchmark" in result.stdout and "--execute" in result.stdout

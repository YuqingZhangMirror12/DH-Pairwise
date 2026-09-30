"""Synthetic queue/process fixtures only; no remote processes or model runs."""
import argparse
import copy
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from experiments.rachel_n512_formal_30k import reconnect_decoupled_per_pair_norm as subject
from experiments.rachel_n512_formal_30k.insert_score_decoupled_queues import new_configurations, stage


def fixture(root):
    _, first = new_configurations(str(root), 99)
    wait = stage("await_complete_step_data", ["python", "wait"], root / "wait")
    first["stages"].insert(22, wait)
    configs = [first]
    for index in range(1, 4):
        configs.append(dict(root=str(root / ("tail%d" % index)), source="/unchanged/tail-source-%d" % index,
            dependencies=[dict(name="previous", pid=index, marker=str(Path(configs[-1]["root"]) / "config.json"),
                status_path=str(Path(configs[-1]["root"]) / "queue_state.json")),
                dict(name="other", pid=98, marker="/producer/config.json", status_path="/producer/queue_state.json")],
            stages=[stage("unchanged_tail_%d" % index, ["python", "-m", "tail", "--value", str(index)],
                root / ("tail-output-%d" % index))]))
    snapshots = []
    for index, config in enumerate(configs):
        state = dict(config=copy.deepcopy(config), pid=index + 1,
            status="running" if index == 0 else "waiting_for_dependency",
            stages=[dict(row, status="queued") for row in config["stages"]])
        if index == 0:
            for row in state["stages"][:12]:
                row["status"] = "complete"
            state["stages"][12].update(status="running", pid=9)
            state["active_child"] = dict(pid=9, marker=config["stages"][12]["marker"])
        else:
            state["dependency_pid_overrides"] = {"other": 999}
        snapshots.append(dict(path=str(Path(config["root"]) / "config.json"), config=config, state=state))
    child = dict(pid=9, ppid=1, state="R", start_time_ticks="900",
                 argv=copy.deepcopy(first["stages"][12]["command"]))
    return snapshots, child, dict(status="running", pid=9, phase="classifier")


class ReconnectTests(unittest.TestCase):
    def test_exact_order_pending_work_and_both_s5_revisions(self):
        snapshots, child, status = fixture(Path("/fixture"))
        original = copy.deepcopy(snapshots)
        configs, bindings, changes = subject.prepare_configs(snapshots, "/isolated/v3", child, status)
        self.assertEqual(snapshots, original)
        stages = configs[0]["stages"]
        self.assertEqual(len(stages), 42)
        self.assertEqual(stages[:9], snapshots[0]["config"]["stages"][13:22])
        self.assertEqual(stages[9]["name"], "s3_matrix_per_pair_norm_v3_revalidate_C13_C20")
        self.assertEqual(stages[9]["resume_arguments"], ["--resume"])
        self.assertEqual(subject.option(stages[9]["command"], "--source-training-run"), changes["source_training"])
        self.assertEqual(subject.option(stages[9]["command"], "--output"), changes["target_training"])
        self.assertEqual([row["name"] for row in stages[10:19]], [
            "s3_matrix_per_pair_norm_v3_" + selection + "_" + split
            for selection in subject.SELECTIONS for split in subject.SPLITS])
        for row, old in zip(stages[10:19], snapshots[0]["config"]["stages"][2:11]):
            expected = subject.replace_option(old["command"], "--training-run", changes["target_training"])
            expected = subject.replace_option(expected, "--output", row["marker"])
            self.assertEqual(row["command"], expected)
        self.assertEqual(stages[19]["name"], "await_complete_step_data")
        count = 0
        for row, old in zip(stages[19:], snapshots[0]["config"]["stages"][22:]):
            expected = copy.deepcopy(old)
            if subject.TRAIN_MODULE in expected["command"]:
                expected["command"] += ["--matrix-head-revision", "per_pair_norm_v3"]
                count += 1
            self.assertEqual(row, expected)
        self.assertEqual(count, 4)
        self.assertFalse(any(row["name"] in ("s4_cross_attention_M12_C8", "s3_matrix_M12_C8") for row in stages))
        self.assertEqual(configs[0]["dependencies"], [dict(name="existing_S4_training_complete", pid=9,
            marker=subject.option(child["argv"], "--output"),
            status_path=str(Path(subject.option(child["argv"], "--output")) / "status.json"))])
        self.assertEqual(bindings, [None, "previous", "previous", "previous"])
        for index in range(1, 4):
            self.assertEqual(configs[index]["stages"], snapshots[index]["config"]["stages"])
            self.assertEqual(configs[index]["source"], snapshots[index]["config"]["source"])
            self.assertIsNone(configs[index]["dependencies"][0]["pid"])
            self.assertEqual(configs[index]["dependencies"][1]["pid"], 999)
        self.assertEqual(len(changes["original_stage_mapping"]), 45)
        self.assertEqual(changes["S3_revalidation_additional_training_exposure"], 0)

    def test_reject_progress_mutations_and_ambiguous_handles(self):
        mutations = [
            lambda s, c, r: s[0]["state"]["stages"][11].update(status="queued"),
            lambda s, c, r: s[0]["state"]["stages"][12].update(status="complete"),
            lambda s, c, r: s[0]["state"]["stages"][13].update(status="running"),
            lambda s, c, r: s[1]["state"]["stages"][0].update(status="complete"),
            lambda s, c, r: s[1]["state"].update(active_child={"pid": 77}),
            lambda s, c, r: c.update(state="T"),
            lambda s, c, r: c.update(pid=88),
            lambda s, c, r: r.update(status="complete"),
            lambda s, c, r: s[0]["state"]["stages"][13].update(command=["changed"]),
            lambda s, c, r: s[1]["state"]["config"].update(source="tampered"),
        ]
        for mutation in mutations:
            snapshots, child, status = fixture(Path("/fixture"))
            mutation(snapshots, child, status)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                subject.prepare_configs(snapshots, "/isolated/v3", child, status)

    def test_reject_existing_revision_or_broken_chain(self):
        for kind in ("revision", "chain"):
            snapshots, child, status = fixture(Path("/fixture"))
            if kind == "revision":
                snapshots[0]["config"]["stages"][23]["command"] += ["--matrix-head-revision", "legacy"]
            else:
                snapshots[2]["config"]["dependencies"][0]["marker"] = "/wrong/config.json"
            for item in snapshots:
                item["state"]["config"] = copy.deepcopy(item["config"])
                for row, original in zip(item["state"]["stages"], item["config"]["stages"]):
                    row.update(copy.deepcopy(original))
            with self.assertRaises(ValueError):
                subject.prepare_configs(snapshots, "/isolated/v3", child, status)

    def test_child_group_and_unknown_pid_signals_are_prohibited(self):
        with patch.object(subject.os, "kill") as kill:
            for pid in (9, -1, 0, 99):
                handle = dict(pid=pid, argv=["python", "-m", subject.QUEUE_MODULE])
                with self.assertRaises(ValueError):
                    subject.retire_parent(handle, protected_child_pid=9, allowed_parent_pids=[1, 2, 3, 4])
            kill.assert_not_called()

    def test_process_reuse_or_unpaused_parent_is_rejected_before_signal(self):
        expected = dict(pid=1, argv=["python", "-m", subject.QUEUE_MODULE], start_time_ticks="100", state="T")
        for change in (dict(start_time_ticks="101"), dict(state="S"), dict(argv=["other"])):
            actual = dict(expected, **change)
            with patch.object(subject, "process", return_value=actual), patch.object(subject.os, "kill") as kill:
                with self.assertRaises(RuntimeError):
                    subject.retire_parent(expected, protected_child_pid=9, allowed_parent_pids=[1, 2, 3, 4])
                kill.assert_not_called()

    def test_mock_execute_preserves_child_and_old_sources_and_binds_new_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots, child, status = fixture(root)
            configs, bindings, changes = subject.prepare_configs(snapshots, str(root / "newsource"), child, status)
            parents = []
            for index, snapshot in enumerate(snapshots):
                subject.write_new(snapshot["path"], snapshot["config"])
                state_path = Path(snapshot["config"]["root"]) / "queue_state.json"
                subject.write_new(state_path, snapshot["state"])
                snapshot.update(config_sha256=subject.digest(snapshot["path"]), state_sha256=subject.digest(state_path))
                parents.append(dict(pid=index + 1, state="T", start_time_ticks=str(index + 100),
                    argv=["python", "-m", subject.QUEUE_MODULE, "--config", snapshot["path"]]))
            plan = dict(work=str(root / "work"), parent_handles=parents, protected_child=child,
                snapshots=snapshots, new_configs=configs, chain_dependency_names=bindings, changes=changes,
                first_parent_physical_environment="16")
            processes = {row["pid"]: copy.deepcopy(row) for row in parents + [child]}
            launched, signals = [], []

            def fake_launch(command, **kwargs):
                pid = 1001 + len(launched)
                config = subject.read(subject.option(command, "--config"))
                launched.append((config, kwargs))
                subject.write_new(Path(config["root"]) / "queue_state.json",
                    dict(pid=pid, status="waiting_for_dependency"))
                return argparse.Namespace(pid=pid, poll=lambda: None)

            def fake_kill(pid, sig):
                self.assertEqual(len(launched), 4, "all replacement observers register before retirement")
                self.assertNotEqual(pid, child["pid"])
                signals.append((pid, sig))
                if sig == signal.SIGCONT:
                    processes.pop(pid)

            with patch.object(subject, "process", side_effect=lambda pid: processes.get(pid)), \
                    patch.object(subject.subprocess, "Popen", side_effect=fake_launch), \
                    patch.object(subject.os, "kill", side_effect=fake_kill):
                receipt = subject.execute(argparse.Namespace(python="/python"), plan)
            self.assertEqual(receipt["status"], "replacement_queues_launched")
            self.assertEqual(receipt["retired_parent_pids"], [4, 3, 2, 1])
            self.assertEqual(signals, [(pid, sig) for pid in (4, 3, 2, 1) for sig in (signal.SIGTERM, signal.SIGCONT)])
            self.assertEqual(processes[9], child)
            self.assertFalse(receipt["child_signalled"])
            self.assertEqual(launched[0][0]["dependencies"][0]["pid"], 9)
            self.assertEqual(launched[0][1]["env"][subject.PHYSICAL_ENV], "16")
            for index in range(1, 4):
                self.assertEqual(launched[index][0]["dependencies"][0]["pid"], 1000 + index)
                self.assertNotIn(subject.PHYSICAL_ENV, launched[index][1]["env"])
            for snapshot in snapshots:
                self.assertEqual(subject.digest(snapshot["path"]), snapshot["config_sha256"])
                self.assertEqual(subject.digest(Path(snapshot["config"]["root"]) / "queue_state.json"), snapshot["state_sha256"])

    def test_default_is_read_only_and_publication_never_overwrites(self):
        args = subject.parser().parse_args(["--work", "/work", "--new-source", "/source", "--trainer-pid", "9",
            "--queue-pids", "1", "2", "3", "4", "--python", "/python"])
        self.assertFalse(args.execute)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            subject.write_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                subject.write_new(path, {"original": False})
            self.assertEqual(subject.read(path), {"original": True})


if __name__ == "__main__":
    unittest.main()

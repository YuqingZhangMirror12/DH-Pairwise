"""Bounded orchestration/real CLI parsing tests; never launch model training."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.rachel_n512_formal_30k import prepare_score_after020_chain as tail
from experiments.rachel_n512_formal_30k import prepare_score_budget_queue as budget
from experiments.rachel_n512_formal_30k import prepare_score_continuation_chain as prior


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def dependency_fixture(root, *, status="complete", pid=99999999):
    (root / "source").mkdir()
    config = prior.build_config(root, 99999998)
    directory = root / "queues/continuation_to020"
    write_json(directory / "config.json", config)
    write_json(directory / "queue_state.json", dict(config=config, status=status, pid=pid))
    return config


def previous_budget_fixture(root, epoch):
    """Only existing preparer's completion metadata gates, not fake model data."""
    for arm in budget.ARMS:
        training = root / arm / "training"
        write_json(training / "status.json", dict(status="budget_complete", epoch=epoch,
            global_exposure=epoch * 24000, optimizer_updates=epoch * 1500))
        winners = {s: dict(checkpoint_sha256=arm + s) for s in budget.SELECTIONS}
        write_json(training / "budget_freezes" / ("%03d" % epoch) / "freeze.json",
            dict(schema_version="rachel-score-design-training/1", status="frozen_at_budget",
                 budget_epochs=epoch, held_out_used_for_fit=False, winners=winners))
        for selection in budget.SELECTIONS:
            for split in budget.SPLITS:
                output = budget.evaluation_path(root, arm, epoch, selection, split)
                write_json(output / "protocol.json", dict(
                    schema_version="rachel-score-design-evaluation/1", status="complete", split=split,
                    sample_count={"test": 3000, "real": 1016, "ood": 301}[split],
                    model=dict(budget=epoch, selection=selection, architecture=arm,
                               checkpoint_sha256=winners[selection]["checkpoint_sha256"])))
                write_json(output / "summary.json", dict(status="complete"))
                # Original preparer checks presence, never reads any metric.
                (output / "pair_results.jsonl").write_text('{"fixture_only": true}\n')


class After020Tests(unittest.TestCase):
    def test_fixed_six_owned_steps_and_child_resume_semantics(self):
        config = tail.build_config("/experiment", 123)
        self.assertEqual(config["root"], "/experiment/queues/after020_s3_to050")
        self.assertEqual(config["source"], "/experiment/source")
        self.assertEqual(config["dependencies"], [dict(
            name="continuation_to020_with_all_frozen_evaluations", pid=123,
            marker="/experiment/queues/continuation_to020/config.json",
            status_path="/experiment/queues/continuation_to020/queue_state.json")])
        stages = config["stages"]
        self.assertEqual([s["name"] for s in stages], ["prepare_candidate_dual_s3", "run_candidate_dual_s3",
            "prepare_budget_030", "run_budget_030", "prepare_budget_050", "run_budget_050"])
        self.assertEqual(stages[0]["command"][-2:], ["--architecture", "candidate_dual"])
        for index, previous in ((2, 20), (4, 30)):
            self.assertEqual(stages[index]["command"][-2:], ["--previous-budget", str(previous)])
        for index, stage in enumerate(stages):
            self.assertNotIn("launch_score_design_queue", " ".join(stage["command"]))
            if index % 2:
                self.assertEqual(stage["command"][2], "experiments.rachel_n512_formal_30k.run_recall_benchmark_queue")
                self.assertEqual(stage["command"][-2:], ["--config", stage["marker"]])
                self.assertEqual(stage["resume_arguments"], ["--resume"])
                self.assertEqual(stage["completion_statuses"], ["complete"])
            else:
                self.assertNotIn("resume_arguments", stage)
                self.assertEqual(stage["completion_statuses"], ["prepared_not_launched"])

    def test_invalid_identity_and_exact_process_marker(self):
        for root, pid in (("relative", 1), ("/x", 0), ("/x", True), ("/x", -2)):
            with self.assertRaises(ValueError):
                tail.build_config(root, pid)
        marker = "/x/queues/continuation_to020/config.json"
        for command, expected in (
            (b"python\0-m\0experiments.run_recall_benchmark_queue\0--config\0" + marker.encode(), True),
            (b"python\0-m\0train_score_design\0" + marker.encode(), False),
            (b"python\0-m\0run_recall_benchmark_queue\0--config\0/wrong/config.json", False)):
            with patch.object(Path, "read_bytes", return_value=command):
                self.assertEqual(tail.dependency_live(123, marker), expected)

    def test_prepares_once_without_future_freezes_or_subprocesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            dependency_fixture(root)
            with patch.object(subprocess, "Popen", side_effect=AssertionError("must not launch")):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = tail.prepare(root, 99999999)
            self.assertEqual(result["status"], "prepared_not_launched")
            receipt = json.loads((root / "queues/after020_s3_to050/preparation.json").read_text())
            self.assertFalse(receipt["future_freezes_prepared"])
            self.assertFalse(receipt["launches_processes"])
            self.assertFalse(receipt["held_out_metrics_used_to_choose_next_action"])
            self.assertEqual(receipt["s3_architecture"], "candidate_dual")
            self.assertEqual(receipt["s3_auxiliary_window"], [13, 20])
            self.assertEqual(receipt["subsequent_three_arm_budgets"], [30, 50])
            self.assertEqual([p for p in tail.future_outputs(root) if p.exists()],
                             [root / "queues/after020_s3_to050"])
            with self.assertRaises(FileExistsError):
                tail.prepare(root, 99999999)

    def test_running_dependency_requires_actual_pid_and_live_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            dependency_fixture(root, status="running")
            with patch.object(tail, "dependency_live", return_value=False):
                with self.assertRaisesRegex(ValueError, "process is stopped"):
                    tail.prepare(root, 99999999)
            with patch.object(tail, "dependency_live", return_value=True) as live:
                with contextlib.redirect_stdout(io.StringIO()):
                    tail.prepare(root, 99999999)
                live.assert_called_once_with(99999999, root / "queues/continuation_to020/config.json")

    def test_wrong_dependency_and_preexisting_future_output_rejected(self):
        for variant in ("pid", "status", "config", "stage_order", "future"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                config = dependency_fixture(root)
                state_path = root / "queues/continuation_to020/queue_state.json"
                state = json.loads(state_path.read_text())
                if variant == "pid":
                    state["pid"] += 1
                elif variant == "status":
                    state["status"] = "failed"
                elif variant == "config":
                    state["config"]["source"] = "/other/source"
                elif variant == "stage_order":
                    config["stages"].reverse()
                    state["config"] = config
                    write_json(root / "queues/continuation_to020/config.json", config)
                else:
                    (root / "candidate_pair/training/budget_freezes/030").mkdir(parents=True)
                write_json(state_path, state)
                with self.assertRaises((ValueError, FileExistsError)):
                    tail.prepare(root, 99999999)
                self.assertFalse((root / "queues/after020_s3_to050").exists())

    def test_actual_tail_cli_is_prepare_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            dependency_fixture(root)
            command = [sys.executable, "-m", "experiments.rachel_n512_formal_30k.prepare_score_after020_chain",
                "--root", str(root), "--continuation-queue-pid", "99999999"]
            completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            self.assertEqual(json.loads(completed.stdout)["status"], "prepared_not_launched")
            self.assertFalse((root / "queues/after020_s3_to050/queue_state.json").exists())
            self.assertFalse((root / "queues/s3_candidate_dual").exists())

    def test_real_s3_cli_parses_fixed_dual_and_config_has21_steps(self):
        from experiments.rachel_n512_formal_30k import prepare_score_staged_queue as staged
        command = tail.build_config("/experiment", 123)["stages"][0]["command"]
        # Execute the actual CLI parser, but future freeze preparation is forbidden now.
        with patch.object(sys, "argv", [command[2]] + command[3:]):
            with patch.object(staged, "prepare", return_value={"status": "parser_fixture"}) as prepare:
                with contextlib.redirect_stdout(io.StringIO()):
                    staged.main()
                prepare.assert_called_once_with("/experiment", "candidate_dual")
        config = staged.build_config("/experiment", "candidate_dual")
        self.assertEqual(len(config["stages"]), 21)
        self.assertEqual([s["name"] for s in config["stages"][:3]],
            ["s3_matcher_smoke32", "s3_m12_c8_train20", "freeze_equal_budget_s3_comparison"])
        training = [s for s in config["stages"] if "train_score_staged" in s["command"][2]]
        self.assertEqual(len(training), 2)  # One discard smoke and one staged run; joint20 is reused.
        for stage in training:
            self.assertEqual(stage["command"][stage["command"].index("--architecture") + 1], "candidate_dual")
            self.assertEqual(stage["command"][stage["command"].index("--stop-after-epoch") + 1], "20")

    def test_actual_budget_preparation_clis_keep_three_continuous_trajectories(self):
        from experiments.rachel_n512_formal_30k.train_score_design import parser as train_parser
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for previous, target in ((20, 30), (30, 50)):
                previous_budget_fixture(root, previous)
                command = tail.build_config(root, 123, python=sys.executable)["stages"][2 if previous == 20 else 4]["command"]
                completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
                config = json.loads(Path(json.loads(completed.stdout)["config"]).read_text())
                self.assertEqual(len(config["stages"]), 21)
                trains = [s for s in config["stages"] if s["command"][2].endswith(".train_score_design")]
                self.assertEqual(len(trains), 3)
                self.assertEqual(len(config["stages"]) - len(trains), 18)
                for stage, arm in zip(trains, budget.ARMS):
                    args = train_parser().parse_args(stage["command"][3:])
                    self.assertEqual(args.architecture, arm)
                    self.assertEqual(args.training_mode, "joint")
                    self.assertTrue(args.resume)
                    self.assertEqual(args.max_epochs, 50)
                    self.assertEqual(args.stop_after_epoch, target)
                    self.assertEqual(args.output, str(root / arm / "training"))
                    self.assertEqual(args.workers, 4)
                    self.assertEqual(stage["completion_statuses"], ["complete" if target == 50 else "budget_complete"])
                self.assertFalse((root / "queues" / ("budget%03d" % target) / "queue_state.json").exists())


if __name__ == "__main__":
    unittest.main()

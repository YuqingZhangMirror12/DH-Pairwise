"""Finite orchestration tests; no remote process or model is launched."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.rachel_n512_formal_30k.prepare_score_continuation_chain import build_config, prepare


class ContinuationTests(unittest.TestCase):
    def test_serial_preparation_and_execution_stops_at20(self):
        config = build_config("/experiment", 123)
        self.assertEqual(len(config["dependencies"]), 1)
        self.assertEqual(config["dependencies"][0]["pid"], 123)
        self.assertEqual(config["dependencies"][0]["marker"], "/experiment/queues/candidates5/config.json")
        self.assertEqual([s["name"] for s in config["stages"]],
            ["prepare_budget_010", "run_budget_010", "prepare_budget_020", "run_budget_020"])
        for index, previous in ((0, 5), (2, 10)):
            stage = config["stages"][index]
            self.assertEqual(stage["command"][-2:], ["--previous-budget", str(previous)])
            self.assertNotIn("resume_arguments", stage)
            run = config["stages"][index + 1]
            self.assertEqual(run["resume_arguments"], ["--resume"])
            self.assertIn("run_recall_benchmark_queue", run["command"][2])
            self.assertNotIn("launch_score_design_queue", " ".join(run["command"]))

    def test_invalid_identity(self):
        for root, pid in (("relative", 1), ("/x", 0), ("/x", True)):
            with self.assertRaises(ValueError):
                build_config(root, pid)

    def test_completed_dependency_prepares_once_without_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source").mkdir()
            first = root / "queues/candidates5"
            first.mkdir(parents=True)
            (first / "queue_state.json").write_text(json.dumps(dict(pid=99999999, status="complete")))
            result = prepare(root, 99999999)
            self.assertEqual(result["status"], "prepared_not_launched")
            with self.assertRaises(FileExistsError):
                prepare(root, 99999999)

    def test_stopped_unfinished_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "queues/candidates5"
            first.mkdir(parents=True)
            (first / "queue_state.json").write_text(json.dumps(dict(pid=99999999, status="running")))
            with self.assertRaisesRegex(ValueError, "process is stopped"):
                prepare(root, 99999999)


if __name__ == "__main__":
    unittest.main()

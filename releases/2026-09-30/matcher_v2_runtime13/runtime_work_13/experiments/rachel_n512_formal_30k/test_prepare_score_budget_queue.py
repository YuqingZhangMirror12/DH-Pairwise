"""Pure continuation queue tests. No remote process is launched."""
from pathlib import Path
import json
import tempfile
import unittest

from experiments.rachel_n512_formal_30k.prepare_score_budget_queue import (
    ARMS, BUDGETS, SELECTIONS, SPLITS, build_config, evaluation_path, next_budget, verify_previous,
)


class BudgetQueueTests(unittest.TestCase):
    def make_completed_fixture(self, root):
        def save(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        # These are synthetic protocol fixtures, not model performance data.
        for arm in ARMS:
            training = root / arm / "training"
            save(training / "status.json", dict(status="budget_complete", epoch=5,
                global_exposure=120000, optimizer_updates=7500))
            save(training / "budget_freezes/005/freeze.json", dict(
                schema_version="rachel-score-design-training/1", status="frozen_at_budget",
                budget_epochs=5, held_out_used_for_fit=False,
                winners={s: dict(checkpoint_sha256="0" * 64) for s in SELECTIONS}))
            for selection in SELECTIONS:
                for split in SPLITS:
                    output = evaluation_path(root, arm, 5, selection, split)
                    save(output / "protocol.json", dict(schema_version="rachel-score-design-evaluation/1",
                        status="complete", split=split, sample_count={"test": 3000, "real": 1016, "ood": 301}[split],
                        model=dict(budget=5, selection=selection, architecture=arm, checkpoint_sha256="0" * 64)))
                    save(output / "summary.json", dict(status="complete", ignored_metric_value=-999))
                    (output / "pair_results.jsonl").touch()

    def test_every_next_budget_has_equal_budget_arms_and_complete_evaluations(self):
        for previous in BUDGETS[:-1]:
            config = build_config(Path("/experiment"), previous)
            self.assertEqual(len(config["stages"]), 21)
            self.assertEqual(len({s["name"] for s in config["stages"]}), 21)
            budget = next_budget(previous)
            for i, arm in enumerate(ARMS):
                group = config["stages"][i * 7:(i + 1) * 7]
                train = group[0]["command"]
                self.assertIn("--resume", train)
                self.assertEqual(train[train.index("--architecture") + 1], arm)
                self.assertEqual(train[train.index("--stop-after-epoch") + 1], str(budget))
                self.assertEqual(train[train.index("--max-epochs") + 1], "50")
                self.assertEqual(group[0]["completion_statuses"], ["complete" if budget == 50 else "budget_complete"])
                combinations = set()
                for evaluation in group[1:]:
                    command = evaluation["command"]
                    combinations.add((command[command.index("--selection") + 1], command[command.index("--split") + 1]))
                    self.assertEqual(command[command.index("--budget") + 1], str(budget))
                    self.assertNotIn("--checkpoint", command)
                    self.assertEqual(evaluation["completion_statuses"], ["complete"])
                    if arm != "original":
                        control = "original" if arm == "candidate_pair" else "candidate_pair"
                        self.assertIn("/" + control + "/evaluation/", command[command.index("--baseline-evaluation") + 1])
                self.assertEqual(len(combinations), 6)

    def test_only_registered_adjacent_budgets_and_absolute_root(self):
        for previous in (0, 6, 15, 50):
            with self.assertRaises(ValueError):
                next_budget(previous)
        with self.assertRaises(ValueError):
            build_config("relative", 5)

    def test_previous_completion_requires_every_frozen_model_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_completed_fixture(root)
            records = verify_previous(root, 5)
            self.assertEqual([r["arm"] for r in records], list(ARMS))
            output = evaluation_path(root, "candidate_dual", 5, "recall95", "ood")
            path = output / "protocol.json"
            protocol = json.loads(path.read_text())
            protocol["status"] = "running"
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "evaluation incomplete"):
                verify_previous(root, 5)
            protocol["status"] = "complete"
            protocol["model"]["checkpoint_sha256"] = "1" * 64
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "evaluation incomplete"):
                verify_previous(root, 5)

    def test_continuation_does_not_select_using_held_out_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_completed_fixture(root)
            before = verify_previous(root, 5)
            for arm in ARMS:
                for selection in SELECTIONS:
                    for split in SPLITS:
                        path = evaluation_path(root, arm, 5, selection, split) / "summary.json"
                        path.write_text(json.dumps(dict(status="complete", ignored_metric_value=12345)))
            self.assertEqual(verify_previous(root, 5), before)


if __name__ == "__main__":
    unittest.main()

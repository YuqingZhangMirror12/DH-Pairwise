"""S3 scheduling tests only; fixtures do not represent trained results."""
from pathlib import Path
import unittest

from experiments.rachel_n512_formal_30k.prepare_score_staged_queue import build_config, locations
from experiments.rachel_n512_formal_30k import train_score_staged, evaluate_score_staged


class StagedQueueTests(unittest.TestCase):
    def test_registered_architectures_absolute_paths_and_no_joint_retraining(self):
        for architecture in ("candidate_pair", "candidate_dual"):
            config = build_config("/experiment", architecture)
            self.assertEqual(len(config["stages"]), 21)
            self.assertEqual(len({s["marker"] for s in config["stages"]}), 21)
            self.assertEqual(config["dependencies"], [])
            self.assertFalse(any("experiments.rachel_n512_formal_30k.train_score_design" in s["command"]
                                 for s in config["stages"]))
        with self.assertRaises(ValueError):
            locations("relative", "candidate_dual")
        with self.assertRaises(ValueError):
            locations("/experiment", "original")

    def test_actual_cli_training_phase_budget_and_smoke_not_formal(self):
        stages = build_config("/experiment", "candidate_dual")["stages"]
        smoke, train = [train_score_staged.parser().parse_args(s["command"][3:]) for s in stages[:2]]
        self.assertEqual((smoke.smoke, smoke.stop_after_epoch), (32, 20))
        self.assertIsNone(train.smoke)
        self.assertEqual(train.stop_after_epoch, 20)
        self.assertNotEqual(smoke.output, train.output)
        self.assertEqual(stages[0]["completion_statuses"], ["smoke_complete"])
        self.assertEqual(stages[1]["resume_arguments"], ["--resume"])
        self.assertEqual(stages[2]["completion_statuses"], ["paired_frozen"])

    def test_each_schedule_same_selection_split_and_only_staged_has_baseline(self):
        stages = build_config("/experiment", "candidate_pair")["stages"][3:]
        seen = {"joint": set(), "staged": set()}
        for stage in stages:
            args = evaluate_score_staged.parser().parse_args(stage["command"][3:])
            schedule = "staged" if "/s3/" in args.training_run else "joint"
            seen[schedule].add((args.selection, args.split))
            self.assertNotIn("budget_020", args.output)
            self.assertIn("s3_epoch020", args.output)
            self.assertNotIn("--checkpoint", stage["command"])
            self.assertEqual(args.keep_ids is not None, args.split == "real")
            if schedule == "staged":
                self.assertEqual(args.baseline_evaluation,
                    "/experiment/candidate_pair/evaluation/s3_epoch020/%s/%s" % (args.selection, args.split))
            else:
                self.assertIsNone(args.baseline_evaluation)
        self.assertEqual(seen["joint"], seen["staged"])
        self.assertEqual(len(seen["joint"]), 9)


if __name__ == "__main__":
    unittest.main()

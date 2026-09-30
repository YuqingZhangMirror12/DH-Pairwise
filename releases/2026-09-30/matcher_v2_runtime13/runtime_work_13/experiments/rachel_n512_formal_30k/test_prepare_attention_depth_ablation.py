"""Synthetic config fixtures only: no datasets, GPU or trained results."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.rachel_n512_formal_30k import prepare_attention_depth_ablation as prep


def flag(command, name):
    return command[command.index(name) + 1]


class AttentionDepthPreparationTests(unittest.TestCase):
    def setUp(self):
        self.config = prep.build_config("/experiment/depth_ablation",
            "/experiment/depth_ablation/source", reference_root="/experiment")

    def test_only_two_new_depths_same_m12_and_fair_classifier_budget(self):
        config = self.config
        self.assertEqual(len(config["stages"]), 22)
        self.assertEqual(len({s["marker"] for s in config["stages"]}), 22)
        trainings = [s for s in config["stages"] if s["name"].endswith("_C13_C20")]
        self.assertEqual(len(trainings), 2)
        self.assertEqual({flag(s["command"], "--cross-attention-depth") for s in trainings}, {"2", "4"})
        for stage in trainings:
            command = stage["command"]
            for name, value in {"--head-kind": "cross_attention", "--sampling": "original512",
                    "--stop-after-epoch": "20", "--microbatch": "1", "--physical-microbatch": "16",
                    "--effective-batch": "16", "--workers": "4"}.items():
                self.assertEqual(flag(command, name), value)
            self.assertEqual(flag(command, "--matcher-checkpoint"),
                "/experiment/new_s345_20260914/s3_matrix/training/epoch_012.pt")
            self.assertEqual(flag(command, "--train-materialized-manifest"), prep.TRAIN_MANIFEST)
            self.assertNotIn("--resume", command)
            self.assertNotIn("--matrix-head-revision", command)
            self.assertEqual(stage["resume_arguments"], ["--resume"])
        contract = config["preparation"]
        self.assertEqual((contract["inherited_matcher_exposures"], contract["new_classifier_exposures"],
                          contract["total_budget_exposures"]), (288000, 192000, 480000))
        self.assertEqual(contract["precision"], "fp32")
        self.assertEqual(contract["comparison_depths"], [1, 2, 4])
        self.assertFalse(contract["depth1_retrained"])
        self.assertFalse(contract["queue_insertion_performed"])
        self.assertFalse(contract["deployment_performed"])

    def test_smokes_are_classifier_only_and_all_evaluations_match_depth1_selection(self):
        stages = self.config["stages"]
        smokes = [s for s in stages if s["name"].endswith("_classifier_smoke32")]
        self.assertEqual(len(smokes), 2)
        for stage in smokes:
            self.assertEqual(flag(stage["command"], "--smoke"), "32")
            self.assertEqual(flag(stage["command"], "--smoke-phase"), "classifier")
            self.assertEqual(stage["completion_statuses"], ["smoke_complete"])
            self.assertNotIn("resume_arguments", stage)
        evaluations = [s for s in stages if "evaluate_score_decoupled" in s["command"][2]]
        self.assertEqual(len(evaluations), 18)
        seen = {2: set(), 4: set()}
        for stage in evaluations:
            command = stage["command"]
            selection, split = flag(command, "--selection"), flag(command, "--split")
            depth = 2 if "depth2" in flag(command, "--training-run") else 4
            seen[depth].add((selection, split))
            self.assertEqual(flag(command, "--baseline-evaluation"),
                "/experiment/new_s345_20260914/s4_cross_attention/evaluation/%s/%s" % (selection, split))
            self.assertEqual("--keep-ids" in command, split == "real")
            if split == "real":
                self.assertEqual(flag(command, "--keep-ids"), "/experiment/keep_ids.json")
            self.assertNotIn("--threshold", command)
            self.assertNotIn("--checkpoint", command)
            self.assertEqual(flag(command, "--batch-size"), "1")
        expected = {(s, p) for s in prep.SELECTIONS for p in prep.SPLITS}
        self.assertEqual(seen, {2: expected, 4: expected})

    def test_current_training_and_evaluation_parsers_accept_all_commands(self):
        from experiments.rachel_n512_formal_30k import train_score_decoupled, evaluate_score_decoupled
        for stage in self.config["stages"]:
            command = stage["command"]
            module = train_score_decoupled if command[2].endswith("train_score_decoupled") else evaluate_score_decoupled
            args = module.parser().parse_args(command[3:])
            if module is train_score_decoupled:
                self.assertIn(args.cross_attention_depth, (2, 4))
                self.assertEqual(args.physical_microbatch, 16)

    def test_exclusive_write_prepared_only_and_no_rewrite_or_output_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "config.json"
            run_root = base / "new_remote_run"
            result = prep.prepare(str(run_root), str(run_root / "source"), output,
                                  reference_root="/experiment")
            self.assertEqual(result["status"], "prepared_awaiting_queue_priority")
            self.assertFalse(result["process_started"])
            payload = json.loads(output.read_text())
            self.assertFalse(payload["preparation"]["launch_authorized"])
            self.assertFalse(payload["preparation"]["dependencies_bound"])
            self.assertEqual(payload["dependencies"], [])
            self.assertIn("pending_user_choice", payload["preparation"]["queue_priority"])
            self.assertFalse(run_root.exists())
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                prep.prepare(str(run_root), str(run_root / "source"), output, reference_root="/experiment")
            self.assertEqual(output.read_bytes(), original)

    def test_reject_existing_run_source_and_relative_or_traversing_paths(self):
        for run, source in [("relative", "/experiment/source"),
                ("/experiment/new", "/experiment/source"),
                ("/experiment/new_s345_20260914", "/experiment/new_s345_20260914/source"),
                ("/experiment/new_s345_20260914/depth", "/experiment/new_s345_20260914/depth/source"),
                ("/experiment/../depth", "/experiment/../depth/source")]:
            with self.assertRaises(ValueError):
                prep.build_config(run, source, reference_root="/experiment")


if __name__ == "__main__":
    unittest.main()

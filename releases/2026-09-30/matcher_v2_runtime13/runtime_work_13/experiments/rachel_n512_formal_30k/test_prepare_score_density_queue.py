"""Pure planned density workflow checks; no real training or server actions."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.rachel_n512_formal_30k.prepare_score_density_queue import build_config, clean_split_receipt


class DensityQueueTests(unittest.TestCase):
    def test_each_cap_is_trained_and_tested_at_identical_budget(self):
        for budget in (5, 10, 20, 30, 50):
            config = build_config("/experiment", "candidate_dual", budget, "paired",
                train_root="/density", clean_root="/clean")
            self.assertEqual(config["source"], "/experiment/density_runtime_source")
            self.assertEqual(len(config["stages"]), 16)
            self.assertEqual(len({s["name"] for s in config["stages"]}), 16)
            for start, cap in ((0, 512), (8, 1024)):
                group = config["stages"][start:start + 8]
                train = group[1]["command"]
                self.assertEqual(train[train.index("--contour-cap") + 1], str(cap))
                self.assertEqual(train[train.index("--stop-after-epoch") + 1], str(budget))
                self.assertNotIn("--resume", train)
                self.assertEqual(group[1]["resume_arguments"], ["--resume"])
                self.assertEqual(group[1]["completion_statuses"], ["complete" if budget == 50 else "budget_complete"])
                self.assertEqual(group[0]["completion_statuses"], ["smoke_complete"])
                for evaluation in group[2:]:
                    command = evaluation["command"]
                    self.assertEqual(command[command.index("--budget") + 1], str(budget))
                    if cap == 1024:
                        baseline = command[command.index("--baseline-evaluation") + 1]
                        self.assertIn("/density_variants/paired/n512/evaluation/", baseline)
                        self.assertNotIn("/original/evaluation/", baseline)
                    else:
                        self.assertNotIn("--baseline-evaluation", command)

    def test_generated_arguments_match_actual_entry_parsers(self):
        from experiments.rachel_n512_formal_30k import train_score_density, evaluate_score_density
        for version in ("v1", "v2", "v3", "v4"):
            config = build_config("/experiment", "candidate_pair", 5, "paired",
                train_root="/d", clean_root="/c", source_density_version=version)
            for stage in config["stages"]:
                command = stage["command"]
                parser = train_score_density.parser() if "train_score_density" in command[2] else evaluate_score_density.parser()
                parser.parse_args(command[3:])

    def test_explicit_source_version_identical_for_both_caps(self):
        old = build_config("/experiment", "candidate_dual", 20, "paired_v1", train_root="/d", clean_root="/c")
        for version in ("v2", "v3", "v4"):
            new = build_config("/experiment", "candidate_dual", 20, "paired_" + version,
                train_root="/d" + version, clean_root="/c" + version, source_density_version=version)
            for start in (0, 8):
                for offset in (0, 1):
                    command = new["stages"][start + offset]["command"]
                    self.assertEqual(command[command.index("--source-density-version") + 1], version)
                    self.assertNotIn("--source-density-version", old["stages"][start + offset]["command"])
                # Evaluator obtains the version from the frozen training identity,
                # rather than accepting an independently overridable target version.
                for stage in new["stages"][start + 2:start + 8]:
                    self.assertNotIn("--source-density-version", stage["command"])
        with self.assertRaisesRegex(ValueError, "version"):
            build_config("/experiment", "candidate_dual", 20, "bad", train_root="/d", clean_root="/c",
                source_density_version="auto")

    def test_wrong_budget_or_unscoped_names_rejected(self):
        for name, budget in (("../other", 5), ("x", 6)):
            with self.assertRaises(ValueError):
                build_config("/experiment", "candidate_pair", budget, name, train_root="/d", clean_root="/c")

    def test_subset_preparation_cannot_masquerade_as_complete_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pairs").mkdir()
            # Real-size synthetic identity fixture, not experimental data.
            rows = [dict(pair_id="fixture_%d" % i, label=i < 1500) for i in range(3000)]
            (root / "pairs/val.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            (root / "clean_val_preparation.json").write_text(json.dumps(dict(status="complete", full_split=False,
                split="val", selected_count=3, cached_pair_count=3, failures=[])))
            with self.assertRaisesRegex(ValueError, "subset smoke"):
                clean_split_receipt(root, "val", root)


if __name__ == "__main__":
    unittest.main()

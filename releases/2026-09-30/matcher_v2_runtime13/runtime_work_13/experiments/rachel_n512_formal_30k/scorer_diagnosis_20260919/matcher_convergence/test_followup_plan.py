"""CPU parser/receipt tests only; never dispatch a command or read a checkpoint."""
import builtins
import json
from pathlib import PurePosixPath
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import followup_plan as plan


class PlanTests(unittest.TestCase):
    def test_builder_is_pure_deterministic_and_never_executes(self):
        with patch.object(builtins, "open", side_effect=AssertionError("no I/O")), \
             patch("subprocess.Popen", side_effect=AssertionError("no launch")):
            first = plan.build_plan()
            second = plan.build_plan()
        self.assertEqual(first, second)
        self.assertEqual(len(first["stages"]), 18)
        self.assertEqual(first["status"], "proposed_not_executed")
        self.assertEqual(json.loads(json.dumps(first)), first)
        self.assertFalse(first["m12_retrained"])

    def test_serial_stage_resources_outputs_and_no_resume(self):
        stages = plan.stage_plan("/new/output", "/env/python", "/new/source")
        self.assertEqual(len({s["name"] for s in stages}), 18)
        self.assertEqual(len({s["output"] for s in stages}), 18)
        for index, stage in enumerate(stages):
            self.assertEqual(stage["cwd"], "/new/source")
            self.assertEqual(stage["command"][:2], ["/env/python", "-m"])
            self.assertEqual(stage["env"]["PYTHONPATH"], "/new/source")
            self.assertEqual(stage["env"]["CUDA_VISIBLE_DEVICES"], "" if stage["kind"] == "cpu" else "0")
            self.assertEqual(stage["env_mode"], "complete_explicit_environment")
            self.assertEqual(stage["env"]["HOME"], "/root")
            self.assertIn("/usr/bin", stage["env"]["PATH"])
            self.assertNotIn("--resume", stage["command"])
            self.assertNotIn("--limit", stage["command"])
            self.assertNotIn("--allow-pilot", stage["command"])
            self.assertIn(PurePosixPath(stage["output"]), PurePosixPath(stage["completion"]).parents)
            if index:
                self.assertEqual(stage["depends_on"], [stages[index-1]["name"]])
        self.assertEqual(stages[0]["completion_expect"]["discarded_optimizer_updates"], 2)
        self.assertEqual(stages[1]["completion_expect"]["completed_segments"], 80)
        self.assertEqual(stages[1]["completion_expect"]["continuation_pair_exposures"], 192000)

    def test_exact_own_caches_fresh_heads_fixed_six_endpoints(self):
        stages = plan.stage_plan()
        caches = [s for s in stages if s["name"].endswith("_cache")]
        self.assertEqual(len(caches), 4)
        self.assertEqual(sorted(s["completion_expect"]["completed_pairs"] for s in caches), [3000,3000,24000,24000])
        for epoch in (16,20):
            owned = [s for s in stages if s["name"].startswith("M%d_" % epoch)]
            for stage in owned:
                command = stage["command"]
                checkpoint_flag = "--checkpoint" if "hard_SIMVAL" in stage["name"] else "--matcher-checkpoint"
                self.assertTrue(command[command.index(checkpoint_flag)+1].endswith("epoch_%03d.pt" % epoch))
            training = next(s for s in owned if s["name"].endswith("C16_train"))
            expected = training["completion_expect"]
            self.assertEqual((expected["completed_segments"], expected["completed_head_epochs"], expected["absolute_epoch"]), (64,16,epoch+16))
            self.assertEqual(expected["head_data_shuffle_epoch"], 28)
            self.assertEqual(expected["optimizer_updates"], 24000)
        evaluations = [s for s in stages if "--head-budget" in s["command"]]
        self.assertEqual(len(evaluations), 6)
        for stage in evaluations:
            command = stage["command"]
            self.assertEqual(command[command.index("--head-budget")+1], "16")
            self.assertEqual(command[command.index("--selection")+1], "fixed_epoch")
            self.assertEqual("--keep-ids" in command, stage["completion_expect"]["split"] == "real")
        self.assertEqual(len([s for s in stages if s["command"][2].endswith("evaluate_continuation")]), 0)
        self.assertEqual(len([s for s in stages if "hard_SIMVAL6000" in s["name"]]), 2)

    def test_receipt_names_match_real_smoke_cache_and_training_contracts(self):
        stages = plan.stage_plan()
        for stage in stages:
            if "discard32" in stage["name"]:
                self.assertTrue(stage["completion"].endswith("smoke.json"))
                self.assertEqual(stage["completion_expect"]["status"], "smoke_complete")
            if stage["name"].endswith("_cache"):
                self.assertTrue(stage["completion"].endswith("protocol.json"))
                self.assertTrue(stage["completion_expect"]["formal_training_eligible"])
        prerequisites = plan.prerequisite_contracts()
        self.assertEqual(prerequisites["priority"]["expect"]["completed_stages"], 60)
        self.assertEqual(prerequisites["baseline_status"]["path"], str(plan.BASELINE/"status.json"))
        self.assertEqual(len(prerequisites["baseline_evaluations"]), 3)
        self.assertEqual(prerequisites["hard_m12"]["expect"]["count"], 6000)
        receipts = plan.required_receipts()
        self.assertEqual(len(receipts), 7)
        self.assertTrue(all(set(r) == {"path", "expect"} for r in receipts))

    def test_reject_overlapping_or_relative_paths(self):
        for root in (str(plan.PRIORITY/"child"), str(plan.DEFAULT_SOURCE/"results"), str(plan.D), "relative", "/root/../bad"):
            with self.subTest(root=root), self.assertRaises(ValueError):
                plan.stage_plan(root)


class ActualParserTests(unittest.TestCase):
    def test_every_command_uses_implemented_parser_without_loading_models(self):
        from . import train_continuation, scorer_bridge, evaluate_hard_validation
        accepted = []
        fake = SimpleNamespace(cache=SimpleNamespace(run=lambda args: args),
            train=SimpleNamespace(run=lambda args: args), evaluate=SimpleNamespace(run=lambda args: args))
        # EndpointBridge is the only bridge construction boundary, replaced with
        # a passive receiver. This exercises the actual nested CLI parsers.
        with patch.object(scorer_bridge, "EndpointBridge", return_value=fake) as factory:
            for stage in plan.stage_plan():
                command = stage["command"]
                args = command[3:]
                with self.subTest(stage=stage["name"]):
                    if command[2].endswith("train_continuation"):
                        parsed = train_continuation.parser().parse_args(args)
                        train_continuation.validate_arguments(parsed)
                    elif command[2].endswith("evaluate_hard_validation"):
                        parsed = evaluate_hard_validation.parser().parse_args(args)
                        self.assertTrue(parsed.execute)
                        self.assertFalse(parsed.allow_pilot)
                        self.assertEqual(parsed.workers, 4)
                    else:
                        with patch.dict(scorer_bridge.os.environ, stage["env"]):
                            parsed = scorer_bridge.main(args)
                    self.assertEqual(str(parsed.output), stage["output"])
                    accepted.append(stage["name"])
            self.assertEqual(factory.call_count, 14)
        self.assertEqual(len(accepted), 18)


if __name__ == "__main__":
    unittest.main()

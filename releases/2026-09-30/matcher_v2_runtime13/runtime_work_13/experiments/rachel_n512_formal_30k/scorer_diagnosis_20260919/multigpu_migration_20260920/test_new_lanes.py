"""Pure lane extraction/runner contract tests; no SSH, GPUs or dispatch."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from . import new_lanes as n
from . import lane_runner
from ..matcher_convergence import followup_plan


def fixtures():
    priority = json.loads((Path(__file__).resolve().parents[1]/"priority_s7_matched_g_v1/launch_plan.json").read_text())
    matcher = dict(stages=followup_plan.stage_plan(), required_receipts=followup_plan.required_receipts(),
                   dependency=dict(pid=126580, completed_stages=60))
    return priority, matcher


class LaneTests(unittest.TestCase):
    def test_exact_registered_leaves_preserved_without_input_mutation(self):
        priority, matcher = fixtures(); originals = deepcopy((priority,matcher))
        with patch("subprocess.Popen", side_effect=AssertionError("no dispatch")), \
             patch.object(Path, "read_text", side_effect=AssertionError("pure builder")):
            lanes = n.derive_lanes(priority, matcher)
        self.assertEqual((priority,matcher), originals)
        self.assertEqual({k:len(v["stages"]) for k,v in lanes.items()}, dict(lane1=40,lane2=8,lane3=8,lane6=18))
        sources = {s["name"]:s for p in (priority,matcher) for s in p["stages"]}
        for lane in lanes.values():
            for stage in lane["stages"]:
                source = sources[stage["name"]]
                for field in ("command", "cwd", "env", "completion", "completion_expect"):
                    self.assertEqual(stage[field], source[field])
                lane_runner.validate_stage(stage)
                self.assertNotIn("depends_on", stage)
        encoded = json.dumps(lanes)
        for forbidden in ("119795", "126580", "after_priority", "priority_supervisor", "wait_S7", "retained_depth", "retained_spectral"):
            self.assertNotIn(forbidden, encoded)

    def test_direct_five_order_and_42_existing_endpoints(self):
        lanes = n.derive_lanes(*fixtures())
        direct = lanes["lane1"]["stages"]
        self.assertEqual([s["name"] for s in direct[:5]], [a+"_discard32" for a in n.DIRECT_ARMS])
        self.assertEqual([s["name"] for s in direct if s["name"].endswith("C16_train")], [a+"_C16_train" for a in n.DIRECT_ARMS])
        endpoints = [s for name in ("lane1","lane2","lane3") for s in lanes[name]["stages"] if "--head-budget" in s["command"]]
        self.assertEqual(len(endpoints), 42)
        self.assertEqual(sum(n._flag(s["command"],"--head-budget")=="16" for s in endpoints),21)
        self.assertEqual(sum(n._flag(s["command"],"--head-budget")=="8" for s in endpoints),21)

    def test_matcher_immediate_only_baseline_endpoint_waits_lane1(self):
        lanes = n.derive_lanes(*fixtures()); stages=lanes["lane6"]["stages"]
        self.assertEqual(stages[0]["prerequisites"], [])
        self.assertEqual(stages[1]["prerequisites"], [])
        self.assertEqual(stages[1]["completion_expect"]["completed_segments"],80)
        self.assertEqual([s["gpu"] for s in stages[:5]], [True,True,False,False,False])
        endpoints = [s for s in stages if "--head-budget" in s["command"]]
        self.assertEqual(len(endpoints),6)
        for stage in endpoints:
            self.assertEqual(len(stage["prerequisites"]),3)
            self.assertTrue(all(r["producer_lane"]=="lane1" for r in stage["prerequisites"]))
            self.assertTrue(all("training/all_tokens" in r["path"] for r in stage["prerequisites"]))
            self.assertNotIn("completed_stages", json.dumps(stage["prerequisites"]))
        for stage in stages:
            if stage not in endpoints:
                self.assertFalse(any(r.get("producer_lane")=="lane1" for r in stage["prerequisites"]))

    def test_cpu_stages_bypass_device_wrapper_and_gpu_leaf_wraps(self):
        stages = n.derive_lanes(*fixtures())["lane6"]["stages"]
        lane=dict(name="lane6",gpu_uuid="GPU-fixture")
        plan=dict(device_wrapper="/isolated/device_wrapper.py",output_root="/isolated/lane_outputs")
        for stage in stages:
            cmd=lane_runner.wrapped_command(stage,lane,plan)
            if stage["gpu"]:
                self.assertEqual(cmd[1],plan["device_wrapper"])
                self.assertIn("GPU-fixture",cmd)
            else:
                self.assertEqual(cmd,stage["command"])
        self.assertEqual(sum(not s["gpu"] for s in stages),6)

    def test_baseline_identity_or_budget_mismatch_rejected(self):
        priority,matcher=fixtures()
        changed=deepcopy(matcher)
        next(r for r in changed["required_receipts"] if r["path"].endswith("freezes/c16.json"))["expect"]["identity"]["source_checkpoint_sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"baseline"): n.derive_lanes(priority,changed)
        changed=deepcopy(priority)
        row=next(s for s in changed["stages"] if s["name"]=="G0_C16_train")
        row["command"] += ["--stop-after-head-epoch","8"]
        with self.assertRaisesRegex(ValueError,"shorten"): n.derive_lanes(changed,matcher)
        changed=deepcopy(matcher); changed["stages"]=changed["stages"][:-1]
        with self.assertRaisesRegex(ValueError,"18-stage"): n.derive_lanes(priority,changed)

    def test_architecture_notes_preserve_scientific_distinctions(self):
        notes=n.architecture_summary()
        self.assertIn("edge pairing is not retained",notes["direct5"]["matched_tokens"])
        self.assertIn("paired features retained",notes["direct5"]["matched_edges"])
        self.assertIn("no cached features",notes["G0_G1"]["input"])
        self.assertIn("0.3",notes["G0_G1"]["loss"])
        self.assertIn("no PairBCE",notes["Matcher"]["loss"])


if __name__ == "__main__":
    unittest.main()

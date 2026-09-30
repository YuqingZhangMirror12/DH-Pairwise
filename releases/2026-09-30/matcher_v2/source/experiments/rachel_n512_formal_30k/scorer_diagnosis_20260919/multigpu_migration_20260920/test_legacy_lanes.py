import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.multigpu_migration_20260920 import legacy_lanes as m


class LegacyLanesTest(unittest.TestCase):
    def stage(self, root, name, kind="endpoint_evaluation", module="evaluate"):
        out = root / name
        return dict(name=name, kind=kind, command=["/python", "-m", "original."+module,
                    "--output", str(out)], completion=str(out / "status.json"))

    def test_c2_resume_is_only_change_and_original_completed_endpoint_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = self.stage(root, "c2_C9_C16", "training", "train")
            endpoint = self.stage(root, "c2_fixed_test")
            Path(training["completion"]).parent.mkdir()
            Path(training["completion"]).write_text('{"status":"running"}')
            Path(endpoint["completion"]).parent.mkdir()
            Path(endpoint["completion"]).write_text('{"status":"complete"}')
            stages = m.decorate([training, endpoint], {"source_root": "/frozen"}, "lane0", resume_c2=True)
            self.assertEqual(len(stages), 1)
            self.assertEqual(stages[0]["command"], training["command"] + ["--resume"])
            self.assertEqual(stages[0]["cwd"], "/frozen")
            self.assertEqual(stages[0]["resume_completed_segments"], 81)
            self.assertEqual(json.loads(Path(endpoint["completion"]).read_text()), {"status":"complete"})

    def test_refuse_partial_output_and_old_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = self.stage(root, "existing")
            Path(stage["completion"]).parent.mkdir()
            with self.assertRaisesRegex(ValueError, "will not be overwritten"):
                m.decorate([stage], {"source_root":"/frozen"}, "lane4")
            queue = self.stage(root, "queue", module="queue")
            with self.assertRaisesRegex(ValueError, "leaf modules"):
                m.decorate([queue], {"source_root":"/frozen"}, "lane5")

    def test_exact_arm_counts_c1_exclusion_and_dependency_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = [self.stage(root, arm+"_C9_C16", "training", "train")
                         if i == 0 else self.stage(root, arm+"_endpoint"+str(i))
                         for arm in ("c1", "c2") for i in range(10)]
            depth = [self.stage(root, arm+"_smoke", "depth_smoke", "train") for arm in ("d1","d4")]
            depth += [self.stage(root, arm+"_C9_C16", "depth_training", "train")
                      if i == 0 else self.stage(root, arm+"_endpoint"+str(i))
                      for arm in ("d1", "d4") for i in range(10)]
            spectral = [self.stage(root, arm+"_smoke", "smoke", "train") for arm in ("z","m","s")]
            spectral += [self.stage(root, arm+"_C9_C16", "training", "train")
                         if i == 0 else self.stage(root, arm+"_endpoint"+str(i))
                         for arm in ("z", "m", "s") for i in range(10)]
            plan = dict(source_root="/frozen", output_root=str(root))
            reader = lambda p, phase, *args: {"candidate":candidate,"depth":depth} if phase=="followup" else {"spectral":spectral}
            with patch.object(m,"read",return_value=plan), patch.object(m,"sha256",return_value=m.C2_PAUSED_SHA), patch.object(m,"spectral_inputs",return_value={}):
                lanes = m.derive_legacy_lanes("followup", "spectral", "cpu", stage_reader=reader)
            self.assertEqual({k:len(v) for k,v in lanes.items()}, dict(lane0=10,lane4=22,lane5=33))
            self.assertTrue(all(s["name"].startswith("c2_") for s in lanes["lane0"]))
            self.assertEqual(lanes["lane4"][2]["prerequisites"][0]["producer_lane"], "lane4")
            self.assertEqual(lanes["lane5"][4]["prerequisites"][0]["expect"]["completed_segments"],112)
            self.assertFalse(any("pid" in s for rows in lanes.values() for s in rows))


if __name__ == "__main__":
    unittest.main()

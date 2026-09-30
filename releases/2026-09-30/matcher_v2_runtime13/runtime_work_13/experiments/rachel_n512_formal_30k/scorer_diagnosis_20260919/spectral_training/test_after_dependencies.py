"""CPU-only finite supervisor tests; all subprocesses are synthetic."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training import after_dependencies as sup


def fixture(index, status="complete"):
    spec = sup.DEPENDENCIES[index]
    names = ["train_val", "test", "real", "ood"] if index else ["stage%d" % i for i in range(25)]
    rows = [dict(name=name, status="complete") for name in names]
    receipt = dict(pid=spec["pid"], schema_version=spec["schema_version"], source_root=spec["source_root"],
        status=status, completed_stages=spec["stages"] if status == "complete" else 0, stages=rows)
    launch = dict(receipt, output_root=spec["root"])
    return receipt, launch


class DependencyTests(unittest.TestCase):
    def test_both_registered_processes_must_exit(self):
        for index, spec in enumerate(sup.DEPENDENCIES):
            r, launch = fixture(index)
            live = dict(state="S", startticks=spec["startticks"])
            self.assertEqual(sup.dependency_state(spec, r, live, launch), "waiting")
            self.assertEqual(sup.dependency_state(spec, r, None, launch), "ready")
            self.assertEqual(sup.dependency_state(spec, r, dict(live, state="Z"), launch), "ready")
        with patch.object(sup, "read_dependency", side_effect=lambda d: fixture(sup.DEPENDENCIES.index(d))), \
             patch.object(sup, "process_identity", side_effect=[None, dict(state="S", startticks=932472715)]):
            self.assertFalse(sup.dependencies_ready())

    def test_failed_missing_reused_incomplete_wrong_source_stop(self):
        for index, spec in enumerate(sup.DEPENDENCIES):
            r, launch = fixture(index)
            live = dict(state="S", startticks=spec["startticks"])
            cases = [(dict(r, status="failed"), live, launch),
                     (dict(r, status="running"), None, launch),
                     (r, dict(live, startticks=1), launch),
                     (dict(r, completed_stages=0), None, launch),
                     (r, live, dict(launch, source_root="/wrong")),
                     (dict(r, pid=1), live, launch)]
            for row, process, plan in cases:
                with self.subTest(index=index, receipt=row), self.assertRaises(RuntimeError):
                    sup.dependency_state(spec, row, process, plan)

    def test_real_cpu_receipt_inputs_and_registry_must_agree(self):
        r, _ = fixture(1)
        r["registrations"] = {s: dict(bundle="/actual/" + s, sha256=s + "-sha")
            for s in ("train_val", "test", "real", "ood")}
        r.update(endpoint_registry="/actual/registry.json", endpoint_registry_sha256="registry-sha")
        registry = {s: r["registrations"][s] for s in ("test", "real", "ood")}
        caches = {s: SimpleNamespace(records=range(sup.queue.train.cache.SPLIT_COUNTS[s])) for s in ("train", "val")}
        with patch.object(sup, "read_dependency", return_value=(r, r)), \
             patch.object(sup, "process_identity", return_value=None), \
             patch.object(sup.queue.train.cache, "load_bundle", return_value=({"normalizer": {"mean": [0]}}, caches)) as bundle, \
             patch.object(sup.queue, "load_registry", return_value=registry), \
             patch.object(sup.queue.train.f, "file_sha256", return_value="receipt-sha"):
            result = sup.resolve_inputs()
            self.assertEqual(result["cache_bundle"], "/actual/train_val")
            self.assertEqual(result["endpoint_registry_sha"], "registry-sha")
            self.assertEqual(result["normalizer_sha256"], sup.queue.train.f.digest_json({"mean": [0]}))
            bundle.assert_called_once_with("/actual/train_val", "train_val-sha")
            registry["ood"] = dict(bundle="/wrong", sha256="bad")
            with self.assertRaisesRegex(ValueError, "registry differs"):
                sup.resolve_inputs()

    def test_three_smokes_then_original_thirty_stage_queue(self):
        inputs = dict(cache_bundle="/actual/train", cache_bundle_sha="train-sha")
        rows = sup.stage_plan(dict(output_root="/new", python="/python"), inputs, "inputs-sha")
        self.assertEqual([r["name"] for r in rows], ["zero_discard32", "mass_discard32",
            "mass_spectral_discard32", "spectral_30_stage_queue"])
        for row in rows[:3]:
            self.assertIn("32", row["command"])
            self.assertIn("/actual/train", row["command"])
            self.assertNotIn("--resume", row["command"])
        self.assertIn("inputs-sha", rows[-1]["command"])
        registry = {s: dict(bundle="/" + s, sha256=s) for s in ("test", "real", "ood")}
        nested = sup.queue.stage_plan(Path("/new"), "/actual/train", "train-sha", registry, python="/python")
        self.assertEqual(len(nested), 30)
        self.assertEqual(sum(r["kind"] == "training" for r in nested), 3)

    def test_nested_queue_dispatch_private_guard_does_not_mutate_original(self):
        inputs = dict(cache_bundle="a", cache_bundle_sha="b", endpoint_registry="c", endpoint_registry_sha="d")
        plan = dict(source_root="/source", output_root="/new")
        def fake_run(args):
            for i in range(30):
                subprocess.Popen(["synthetic", str(i)])
            return args.cache_bundle
        original_globals = dict(fake_run.__globals__)
        with patch.object(sup, "load_plan", return_value=plan), \
             patch.object(sup.queue.train.f, "file_sha256", return_value="bound-sha"), \
             patch.object(Path, "read_text", return_value=sup.json.dumps(inputs)), \
             patch.object(sup, "verify_source") as verify, \
             patch.object(sup.queue, "run", fake_run), patch.object(sup.subprocess, "Popen") as popen:
            self.assertEqual(sup.run_queue("unused", "bound-sha"), "a")
            self.assertEqual(verify.call_count, 30)
            self.assertEqual(popen.call_count, 30)
            with self.assertRaisesRegex(ValueError, "receipt changed"):
                sup.run_queue("unused", "wrong-sha")
        self.assertNotIn("subprocess", original_globals)
        self.assertNotIn("subprocess", fake_run.__globals__)

    def test_wait_deadline_never_interrupts_dispatched_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(output_root=str(root), source_root="/unit-source", deadline_hours=48)
            rows = [dict(name="unit%d" % i, kind="smoke", command=["synthetic", str(i)],
                         completion=str(root / ("done%d.json" % i))) for i in range(2)]
            now, waits = [0.], []
            class Child:
                pid = 777777
                def __init__(self, command, **kwargs):
                    self.index = int(command[1])
                def wait(self):
                    waits.append(self.index)
                    now[0] = 49 * 3600.
                    return 0
            with patch.object(sup, "load_plan", return_value=plan), \
                 patch.object(sup, "dependencies_ready", return_value=True), \
                 patch.object(sup, "resolve_inputs", return_value={}), \
                 patch.object(sup, "stage_plan", return_value=rows), \
                 patch.object(sup, "validate_completion"), patch.object(sup, "verify_source") as verify, \
                 patch.object(sup.time, "time", side_effect=lambda: now[0]), \
                 patch.object(sup.time, "sleep", side_effect=AssertionError("no child polling")), \
                 patch.object(sup.subprocess, "Popen", Child), patch.object(sup.os, "killpg") as kill, \
                 patch("builtins.print"):
                result = sup.execute("unused")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(waits, [0, 1])
            self.assertEqual(verify.call_count, 2)
            kill.assert_not_called()

    def test_missing_dependency_no_dispatch_and_no_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = dict(output_root=directory, source_root="/source", deadline_hours=48)
            with patch.object(sup, "load_plan", return_value=plan), \
                 patch.object(sup, "dependencies_ready", side_effect=FileNotFoundError("missing")), \
                 patch.object(sup.subprocess, "Popen") as popen, patch("builtins.print"):
                with self.assertRaises(FileNotFoundError):
                    sup.execute("unused")
                with self.assertRaisesRegex(ValueError, "already started"):
                    sup.execute("unused")
            popen.assert_not_called()

    def test_timeout_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = dict(output_root=directory, source_root="/source", deadline_hours=48)
            with patch.object(sup, "load_plan", return_value=plan), \
                 patch.object(sup.time, "time", side_effect=[0., 49 * 3600., 49 * 3600.]), \
                 patch.object(sup.subprocess, "Popen") as popen, patch("builtins.print"):
                with self.assertRaises(TimeoutError):
                    sup.execute("unused")
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()

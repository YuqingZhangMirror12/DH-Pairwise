"""Stdlib-only registry/dispatch tests; no training, CUDA or live processes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import transfer_registry as registry


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dynamic_pool"
        self.identity = patch.object(registry, "process_startticks", return_value=123)
        self.identity.start(); self.addCleanup(self.identity.stop)

    def package(self, name="one"):
        base = Path(self.temp.name).resolve() / name
        stage = dict(name=name+"_train", gpu=True, command=["python", "-m", "unit.leaf"],
            cwd=str(base), output=str(base / "output"), completion=str(base / "output/status.json"),
            completion_expect=dict(status="complete", completed=16),
            original_runtime_receipt=str(base / "runtime.json"),
            additional_completions=[dict(path=str(base / "proof.json"), expect=dict(valid=True))])
        return dict(name=name, origin_lane="lane1", priority=2, stages=[stage])

    def write(self, path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def complete(self, package):
        stage = package["stages"][0]
        self.write(stage["completion"], dict(status="complete", completed=16))
        self.write(stage["additional_completions"][0]["path"], dict(valid=True))
        with registry.locked_registry(self.root) as state:
            state["experiments"][package["name"]]["status"] = "complete"
            state["stages"][stage["original_runtime_receipt"]].update(status="complete", returncode=0)

    def test_registration_atomicity_and_cpu_rejection(self):
        first, second = self.package(), self.package("two")
        second["stages"][0]["gpu"] = False
        with self.assertRaises(ValueError):
            registry.register_experiments(self.root, [first, second])
        self.assertEqual(registry.read_registry(self.root)["experiments"], {})
        second["stages"][0]["gpu"] = True
        self.assertEqual(registry.register_experiments(self.root, [first, second]), ["one", "two"])
        with self.assertRaises(ValueError):
            registry.register_experiments(self.root, [first])

    def test_original_claim_wins_registration_race(self):
        package = self.package(); receipt = package["stages"][0]["original_runtime_receipt"]
        self.assertFalse(registry.original_dispatch_gate(self.root, receipt))
        self.assertEqual(registry.read_registry(self.root)["claims"][receipt]["role"], "original")
        with self.assertRaises(ValueError):
            registry.register_experiments(self.root, [package])
        with self.assertRaises(RuntimeError):
            registry.original_dispatch_gate(self.root, receipt)

    def test_distinct_receipts_cannot_reserve_same_output(self):
        one, two = self.package(), self.package("two")
        two["stages"][0]["completion"] = one["stages"][0]["completion"]
        with self.assertRaises(ValueError):
            registry.register_experiments(self.root, [one, two])
        self.assertEqual(registry.read_registry(self.root)["experiments"], {})

    def test_existing_runtime_or_output_never_transferred(self):
        package = self.package()
        self.write(package["stages"][0]["original_runtime_receipt"], dict(status="complete"))
        with self.assertRaises(ValueError):
            registry.register_experiments(self.root, [package])
        Path(package["stages"][0]["original_runtime_receipt"]).unlink()
        self.write(package["stages"][0]["completion"], dict(status="complete"))
        with self.assertRaises(ValueError):
            registry.register_experiments(self.root, [package])

    def test_delegated_gate_waits_outside_lock_then_accepts_proofs(self):
        package = self.package(); receipt = package["stages"][0]["original_runtime_receipt"]
        registry.register_experiments(self.root, [package])
        with patch.object(registry.time, "sleep", side_effect=lambda _: self.complete(package)) as sleep:
            self.assertTrue(registry.original_dispatch_gate(self.root, receipt))
        sleep.assert_called_once()
        state = registry.read_registry(self.root)
        self.assertNotIn(receipt, state["claims"])
        self.assertEqual(state["stages"][receipt]["waiting_proxy"]["status"], "complete")

    def test_completion_requires_exit_zero_and_additional_proof(self):
        package = self.package(); stage = package["stages"][0]; receipt = stage["original_runtime_receipt"]
        registry.register_experiments(self.root, [package]); self.complete(package)
        with registry.locked_registry(self.root) as state:
            state["stages"][receipt]["returncode"] = 1
        with self.assertRaisesRegex(RuntimeError, "successful process exit"):
            registry.original_dispatch_gate(self.root, receipt)
        with registry.locked_registry(self.root) as state:
            state["stages"][receipt]["returncode"] = 0
        self.write(stage["additional_completions"][0]["path"], dict(valid=False))
        with self.assertRaisesRegex(ValueError, "differs"):
            registry.original_dispatch_gate(self.root, receipt)

    def test_failed_or_dead_worker_never_falls_back(self):
        package = self.package(); receipt = package["stages"][0]["original_runtime_receipt"]
        registry.register_experiments(self.root, [package])
        with registry.locked_registry(self.root) as state:
            state["experiments"]["one"].update(status="running", worker_pid=321, worker_startticks=456)
            state["stages"][receipt]["status"] = "running"
        with patch.object(registry, "process_alive", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "worker missing"):
                registry.original_dispatch_gate(self.root, receipt)
        with registry.locked_registry(self.root) as state:
            state["experiments"]["one"]["status"] = "failed"
        with self.assertRaisesRegex(RuntimeError, "fallback forbidden"):
            registry.original_dispatch_gate(self.root, receipt)
        self.assertNotIn(receipt, registry.read_registry(self.root)["claims"])

    def test_transaction_exception_rolls_back_and_process_identity(self):
        with self.assertRaisesRegex(RuntimeError, "unit abort"):
            with registry.locked_registry(self.root) as state:
                state["claims"]["unit"] = {}
                raise RuntimeError("unit abort")
        self.assertEqual(registry.read_registry(self.root)["claims"], {})
        with patch.object(registry, "_process_stat", return_value=("R", 19)):
            self.assertTrue(registry.process_alive(1, 19))
            self.assertFalse(registry.process_alive(1, 20))
        with patch.object(registry, "_process_stat", return_value=("Z", 19)):
            self.assertFalse(registry.process_alive(1, 19))


if __name__ == "__main__":
    unittest.main()

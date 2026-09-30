"""No GPU, remote access, or real signals: test retained queue boundaries."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from . import legacy_tail as tail


class FakeRuntime:
    def __init__(self, *, fail_index=None):
        self.processes = {}
        self.commands = []
        self.environments = []
        self.fail_index = fail_index
        self.stopped = []

    def inspect(self, pid):
        return self.processes.get(pid)

    def spawn(self, stage, source, env, log):
        index = len(self.commands)
        self.commands.append(copy.deepcopy(stage["command"]))
        self.environments.append((source, dict(env)))
        path = Path(stage["completion"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tail.atomic_json(path, {"status": "complete"})
        return SimpleNamespace(pid=800000 + index,
                               wait=lambda: 1 if index == self.fail_index else 0)

    def stop_child(self, child):
        self.stopped.append(child.pid)


class LegacyTailTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "sealed_source"
        self.source.mkdir()
        self.old_root = self.root / "old_outputs"
        self.old_root.mkdir()
        self.plan_path = self.old_root / "launch_plan.json"
        self.handoff_path = self.root / "handoff.json"
        self.plan = dict(source_root=str(self.source), output_root=str(self.old_root), python="python-exact")
        self.handoff = dict(schema=tail.HANDOFF_SCHEMA, status="committed",
            old_dispatchers_superseded=True, old_receipts_modified=False,
            dispatchers=[dict(process=dict(pid=pid, startticks=ticks, argv=["old", "--execute"]))
                         for pid, ticks in tail.REGISTERED_DISPATCHERS.items()],
            termination_signals=[dict(pid=pid, signal="SIGKILL") for pid in tail.REGISTERED_DISPATCHERS],
            current_queue=dict(process=dict(pid=119795, startticks=934449976),
                status_path=str(self.old_root / "candidate_local/queue_status.json")))
        tail.atomic_json(self.handoff_path, self.handoff)
        self.validated = []
        self.smoke_validations = []
        self.source_checks = []
        self.runtime = FakeRuntime()

    def make_stage(self, index, kind="endpoint_evaluation"):
        output = self.old_root / ("stage_%02d" % index)
        return dict(name="original_%02d" % index, kind=kind,
                    command=["python-exact", "-m", "original.sealed.module", "--output", str(output)],
                    completion=str(output / "complete.json"))

    def validate_completion(self, stage):
        self.assertEqual(tail.read_json(stage["completion"])["status"], "complete")
        self.validated.append(stage["name"])

    def depth(self):
        self.candidate_stages = [self.make_stage(i + 100) for i in range(20)]
        candidate = dict(schema_version="rachel-candidate-local-queue/1", pid=119795,
                         status="complete", completed_stages=20,
                         stages=[dict(row, status="complete") for row in self.candidate_stages])
        path = Path(self.handoff["current_queue"]["status_path"])
        path.parent.mkdir()
        tail.atomic_json(path, candidate)
        stages = [self.make_stage(i) for i in range(25)]
        stages[3]["kind"] = stages[4]["kind"] = "depth_smoke"
        stages[5]["kind"] = stages[15]["kind"] = "depth_training"
        self.plan["stages"] = copy.deepcopy(stages)
        return SimpleNamespace(load_plan=lambda path: copy.deepcopy(self.plan),
            stage_plan=lambda root, python: copy.deepcopy(stages),
            candidate=SimpleNamespace(stage_plan=lambda root, python: copy.deepcopy(self.candidate_stages)),
            validate_depth_smokes=lambda root: self.smoke_validations.append(str(root)),
            validate_completion=self.validate_completion,
            verify_source=lambda plan: self.source_checks.append(plan["source_root"]))

    def spectral(self):
        stages = [self.make_stage(i, "smoke") for i in range(3)]
        stages.append(dict(name="spectral_30_stage_queue", kind="queue",
            command=["python-exact", "-m", "old.after_dependencies", "--run-queue", "--plan", str(self.plan_path)],
            completion=str(self.old_root / "queue_status.json")))
        self.spectral_stages = copy.deepcopy(stages)
        return SimpleNamespace(load_plan=lambda path: copy.deepcopy(self.plan),
            resolve_inputs=lambda: dict(cache_bundle="actual", cache_bundle_sha="verified-cpu-sha"),
            save=tail.atomic_json,
            stage_plan=lambda plan, inputs, sha: copy.deepcopy(stages),
            queue=SimpleNamespace(validate_smokes=lambda root, sha: self.smoke_validations.append((str(root), sha))),
            validate_completion=self.validate_completion,
            verify_source=lambda plan: self.source_checks.append(plan["source_root"]))

    def args(self, phase):
        tail.atomic_json(self.plan_path, self.plan)
        return SimpleNamespace(phase=phase, plan=str(self.plan_path), plan_sha256=tail.sha256(self.plan_path),
                               output=str(self.root / "new_phase"), handoff_receipt=str(self.handoff_path))

    def test_depth_retains_exact_22_commands_and_child_environment(self):
        legacy = self.depth()
        expected = copy.deepcopy(self.plan["stages"][3:])
        args = self.args("depth")
        old_plan_bytes = self.plan_path.read_bytes()
        status = tail.execute(args, legacy=legacy, runtime=self.runtime)
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["completed_stages"], 22)
        self.assertEqual(self.runtime.commands, [row["command"] for row in expected])
        self.assertEqual(len(self.validated), 22)
        self.assertEqual(len(self.smoke_validations), 2)
        self.assertEqual(len(self.source_checks), 22)
        self.assertEqual(self.plan_path.read_bytes(), old_plan_bytes)
        self.assertFalse((self.old_root / "supervisor_status.json").exists())
        for cwd, env in self.runtime.environments:
            self.assertEqual(cwd, str(self.source))
            self.assertEqual(env["PYTHONPATH"], str(self.source))
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_spectral_resolves_original_inputs_once_then_exact_four_commands(self):
        legacy = self.spectral()
        args = self.args("spectral")
        status = tail.execute(args, legacy=legacy, runtime=self.runtime)
        self.assertEqual(status["completed_stages"], 4)
        self.assertEqual(self.runtime.commands, [row["command"] for row in self.spectral_stages])
        self.assertEqual(self.smoke_validations, [(str(self.old_root), "verified-cpu-sha")])
        self.assertEqual(status["resolved_inputs_sha256"], tail.sha256(self.old_root / "resolved_inputs.json"))
        self.assertFalse((self.old_root / "after_dependencies_status.json").exists())
        with self.assertRaises(FileExistsError):
            tail.execute(args, legacy=legacy, runtime=self.runtime)
        self.assertEqual(len(self.runtime.commands), 4)

    def test_committed_before_kills_is_not_sufficient(self):
        legacy = self.depth()
        self.handoff["termination_signals"] = []
        tail.atomic_json(self.handoff_path, self.handoff)
        with self.assertRaisesRegex(RuntimeError, "termination receipt"):
            tail.execute(self.args("depth"), legacy=legacy, runtime=self.runtime)
        self.assertEqual(self.runtime.commands, [])

    def test_live_or_reused_dispatcher_rejected(self):
        for state, ticks in (("T", 932414692), ("Z", 1)):
            with self.subTest(state=state, ticks=ticks):
                self.runtime.processes[101639] = dict(state=state, startticks=ticks)
                with self.assertRaises(RuntimeError):
                    tail.validate_handoff(self.handoff_path, self.runtime.inspect)

    def test_uncompleted_candidate_is_not_inferred_from_missing_process(self):
        legacy = self.depth()
        path = Path(self.handoff["current_queue"]["status_path"])
        receipt = tail.read_json(path)
        receipt["stages"][-1]["status"] = "running"
        tail.atomic_json(path, receipt)
        with self.assertRaisesRegex(RuntimeError, "20-stage"):
            tail.execute(self.args("depth"), legacy=legacy, runtime=self.runtime)
        self.assertEqual(self.runtime.commands, [])

    def test_live_candidate_rejected_even_with_complete_receipt(self):
        legacy = self.depth()
        self.runtime.processes[119795] = dict(state="S", startticks=934449976)
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            tail.execute(self.args("depth"), legacy=legacy, runtime=self.runtime)
        self.assertEqual(self.runtime.commands, [])

    def test_existing_child_output_never_overwritten(self):
        legacy = self.depth()
        existing = Path(self.plan["stages"][3]["command"][-1])
        existing.mkdir()
        sentinel = existing / "partial.pt"
        sentinel.write_bytes(b"original")
        with self.assertRaisesRegex(ValueError, "not be resumed"):
            tail.execute(self.args("depth"), legacy=legacy, runtime=self.runtime)
        self.assertEqual(sentinel.read_bytes(), b"original")
        self.assertEqual(self.runtime.commands, [])

    def test_failure_halts_without_retry(self):
        legacy = self.depth()
        self.runtime.fail_index = 2
        with self.assertRaisesRegex(RuntimeError, "no retry"):
            tail.execute(self.args("depth"), legacy=legacy, runtime=self.runtime)
        self.assertEqual(len(self.runtime.commands), 3)
        status = tail.read_json(self.root / "new_phase/status.json")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["completed_stages"], 2)
        self.assertEqual(self.runtime.stopped, [])

    def test_plan_digest_mismatch_prevents_any_new_output(self):
        legacy = self.depth()
        args = self.args("depth")
        args.plan_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest differs"):
            tail.execute(args, legacy=legacy, runtime=self.runtime)
        self.assertFalse(Path(args.output).exists())

    def test_existing_spectral_registration_never_overwritten(self):
        legacy = self.spectral()
        registration = self.old_root / "resolved_inputs.json"
        registration.write_text("original\n")
        with self.assertRaisesRegex(ValueError, "registration will not be overwritten"):
            tail.execute(self.args("spectral"), legacy=legacy, runtime=self.runtime)
        self.assertEqual(registration.read_text(), "original\n")
        self.assertEqual(self.runtime.commands, [])


if __name__ == "__main__":
    unittest.main()

"""Pure CPU decision/plan/seal tests; never dispatch a real subprocess."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation_depth_controls import followup_supervisor as sup


def receipt(status="running"):
    return dict(pid=sup.DEPENDENCY_PID, schema_version=sup.prior.SCHEMA,
        source_root=str(sup.DEPENDENCY_SOURCE), status=status, completed_stages=20 if status == "complete" else 0,
        stages=[dict(status="complete")] * 20 if status == "complete" else [])


class SupervisorTests(unittest.TestCase):
    def test_wait_exact_process_even_if_receipt_already_complete(self):
        process = dict(startticks=sup.DEPENDENCY_STARTTICKS, state="S")
        self.assertEqual(sup.dependency_state(receipt(), process), "waiting")
        self.assertEqual(sup.dependency_state(receipt("complete"), process), "waiting")
        self.assertEqual(sup.dependency_state(receipt("complete"), None), "ready")
        self.assertEqual(sup.dependency_state(receipt("complete"), dict(process, state="Z")), "ready")

    def test_failed_missing_reused_pid_and_incomplete_ledger_stop(self):
        process = dict(startticks=sup.DEPENDENCY_STARTTICKS, state="S")
        for r, p in ((receipt("failed"), process), (receipt(), None),
                (receipt("complete"), dict(process, startticks=123)),
                (dict(receipt("complete"), completed_stages=19), None),
                (dict(receipt(), pid=123), process), (dict(receipt(), source_root="/wrong"), process)):
            with self.subTest(receipt=r, process=p), self.assertRaises(RuntimeError):
                sup.dependency_state(r, p)

    def test_finite_order_and_correct_depth_baselines(self):
        rows = sup.stage_plan(Path("/new"), "/env/python")
        self.assertEqual(len(rows), 25)
        self.assertEqual([x["name"] for x in rows[:5]], ["c1_discard32", "c2_discard32",
            "candidate_C1_C2_queue", "s4_d1_discard32", "s6_d4_discard32"])
        self.assertEqual(rows[5]["kind"], "depth_training")
        self.assertEqual(rows[15]["kind"], "depth_training")
        self.assertEqual(sum(x["kind"] == "endpoint_evaluation" for x in rows), 18)
        for index, arm in ((6, "s4_d1"), (16, "s6_d4")):
            command = rows[index]["command"]
            path = command[command.index("--baseline-evaluation") + 1]
            self.assertTrue(path.startswith(str(sup.BASELINES[arm])))
        for row in rows:
            self.assertNotIn("--resume", row["command"])
            if row["name"].endswith("_real"):
                self.assertIn("--keep-ids", row["command"])

    def test_source_snapshot_change_and_added_code_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("experiments/rachel_n512_formal_30k", "staging/pairwise_v0_2"):
                (root / name).mkdir(parents=True)
            # Only synthetic fixture files, not project source/data.
            path = root / "experiments/rachel_n512_formal_30k/a.py"
            path.touch()
            plan = dict(source_root=str(root), source_sha256=sup.source_hashes(root))
            sup.verify_source(plan)
            (path.parent / "added.py").touch()
            with self.assertRaisesRegex(ValueError, "changed"):
                sup.verify_source(plan)

    def test_candidate_queue_internal_dispatch_is_privately_guarded(self):
        plan = dict(source_root="/source", output_root="/new")
        calls = []
        def run(args):
            subprocess.Popen(["synthetic"])
            return args.output_root
        # Supply a fake original function to verify the private namespace path;
        # the actual subprocess is mocked and never launched.
        with patch.object(sup, "load_plan", return_value=plan), \
             patch.object(sup, "verify_source", side_effect=lambda p: calls.append(p)), \
             patch.object(sup.candidate, "run", run), \
             patch.object(sup.subprocess, "Popen", return_value=object()) as popen:
            self.assertEqual(sup.run_candidate_queue("unused"), "/new/candidate_local")
        self.assertEqual(len(calls), 1)
        popen.assert_called_once_with(["synthetic"])

    def test_maximum_deadline_and_changed_plan_rejected(self):
        plan = dict(schema_version=sup.SCHEMA, deadline_hours=49)
        with patch.object(sup.json, "loads", return_value=plan), patch.object(Path, "read_text", return_value=""):
            with self.assertRaisesRegex(ValueError, "deadline"):
                sup.load_plan("unused")

    def test_dispatch_wait_is_not_killed_by_expired_predecessor_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root / "dependency.json"
            sup.save(dependency, receipt("complete"))
            rows = [dict(name="unit%d" % i, kind="endpoint_evaluation", command=["synthetic", str(i)],
                         completion=str(root / ("done%d.json" % i))) for i in range(2)]
            plan = dict(output_root=str(root), source_root="/unit-source", deadline_hours=48,
                        dependency_status=str(dependency), stages=rows)
            now = [0.]
            waits = []
            class Child:
                pid = 777777
                def __init__(self, command, **kwargs):
                    self.index = int(command[1])
                def wait(self):
                    waits.append(self.index)
                    # Simulate an already dispatched model finishing after48h.
                    now[0] = 49 * 3600.
                    sup.save(rows[self.index]["completion"], dict(status="complete"))
                    return 0
            with patch.object(sup, "load_plan", return_value=plan), \
                 patch.object(sup, "process_identity", return_value=None), \
                 patch.object(sup, "verify_source") as verify, \
                 patch.object(sup.time, "time", side_effect=lambda: now[0]), \
                 patch.object(sup.time, "sleep", side_effect=AssertionError("no active-child polling")), \
                 patch.object(sup.subprocess, "Popen", Child), patch.object(sup.os, "killpg") as kill, \
                 patch("builtins.print"):
                result = sup.execute("unit-plan")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed_stages"], 2)
            self.assertEqual(waits, [0, 1])
            self.assertEqual(verify.call_count, 2)  # only before each dispatch
            kill.assert_not_called()

    def test_missing_dependency_fails_without_child_and_no_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(output_root=str(root), source_root="/unit-source", deadline_hours=48,
                        dependency_status=str(root / "missing.json"), stages=[])
            with patch.object(sup, "load_plan", return_value=plan), \
                 patch.object(sup.subprocess, "Popen") as popen, patch("builtins.print"):
                with self.assertRaises(FileNotFoundError):
                    sup.execute("unit-plan")
                with self.assertRaisesRegex(ValueError, "already started"):
                    sup.execute("unit-plan")
            popen.assert_not_called()

    def test_predecessor_timeout_stops_before_any_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(output_root=str(root), source_root="/unit-source", deadline_hours=48,
                        dependency_status=str(root / "unused.json"), stages=[])
            with patch.object(sup, "load_plan", return_value=plan), \
                 patch.object(sup.time, "time", side_effect=[0., 49 * 3600., 49 * 3600.]), \
                 patch.object(sup.subprocess, "Popen") as popen, patch("builtins.print"):
                with self.assertRaisesRegex(TimeoutError, "predecessor wait"):
                    sup.execute("unit-plan")
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()

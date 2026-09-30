"""Simulated-process safety tests only: NEVER signals or launches real jobs."""
import copy
import hashlib
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import unittest

from . import priority_supervisor as supervisor


def write(path, value):
    Path(path).write_text(json.dumps(value))


def identity(pid, argv=None):
    return dict(pid=pid, startticks=pid*17, argv=argv or ["/python", "-m", "fixture."+str(pid)])


def make_plan(root):
    source = root / "source.py"
    source.write_text("# synthetic fixture, never executed\n")
    queue, trainer, parent, spectral = [identity(pid) for pid in (119795, 119924, 101639, 103819)]
    ledger = [dict(name="old_stage_%02d" % i, kind="training" if i in (0,10) else "endpoint_evaluation",
                   command=["/python", "old_stage", str(i)], completion=str(root/("old_completion_%d.json" % i)))
              for i in range(20)]
    write(root/"candidate.json", dict(schema_version="rachel-candidate-local-queue/1", pid=queue["pid"],
        status="running", stages=[dict(s, status="running" if i==0 else "pending") for i,s in enumerate(ledger)]))
    write(root/"parent.json", dict(status="running", active_pid=queue["pid"], active_name="candidate_C1_C2_queue"))
    write(root/"spectral.json", dict(status="waiting_dependencies", active_pid=None))
    return dict(schema=supervisor.SCHEMA, output_root=str(root/"new_priority"), predecessor_timeout_s=45,
        source_bindings={str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
        dispatchers=[dict(name="followup", process=parent, status_path=str(root/"parent.json"),
            status_expect=dict(status="running", active_pid=queue["pid"], active_name="candidate_C1_C2_queue"),
            expected_children=[queue["pid"]]),
            dict(name="spectral", process=spectral, status_path=str(root/"spectral.json"),
                status_expect=dict(status="waiting_dependencies",active_pid=None),expected_children=[])],
        current_queue=dict(process=queue,status_path=str(root/"candidate.json"),
            schema="rachel-candidate-local-queue/1",completed_stages=20,stage_ledger=ledger),
        protected_processes=[trainer], stages=[dict(name="new_arm",command=["/python","new_training"],
            cwd=str(root),env={"PYTHONPATH":"/isolated/new", "CUDA_VISIBLE_DEVICES":"0"},
            completion=str(root/"new_complete.json"),completion_expect=dict(status="complete",completed_segments=64))])


class FakeChild:
    def __init__(self, runtime, stage):
        self.runtime, self.stage, self.pid = runtime, stage, 900001

    def wait(self):
        self.runtime.clock += 1_000_000  # predecessor wait bound must NOT kill training
        if self.runtime.stage_exit == 0:
            write(self.stage["completion"], self.stage["completion_expect"])
        return self.runtime.stage_exit


class FakeRuntime:
    def __init__(self, plan):
        self.plan, self.clock = plan, 0.
        self.signals, self.spawned, self.sleep_calls = [], [], 0
        self.on_signal, self.on_sleep = None, None
        self.stage_exit = 0
        queue_pid = plan["current_queue"]["process"]["pid"]
        parent_pid = plan["dispatchers"][0]["process"]["pid"]
        self.processes = {}
        for spec in [d["process"] for d in plan["dispatchers"]] + [plan["current_queue"]["process"]] + plan["protected_processes"]:
            ppid = parent_pid if spec["pid"] == queue_pid else queue_pid if spec["pid"] == 119924 else 1
            self.processes[spec["pid"]] = dict(spec, state="R" if spec["pid"] in (queue_pid,119924) else "S", ppid=ppid)

    def process(self, pid):
        return copy.deepcopy(self.processes.get(pid))

    def assert_detached(self):
        pass  # simulated new supervisor is already detached; no real os signals

    def children(self, pid):
        return sorted(p for p,v in self.processes.items() if v["ppid"] == pid)

    def signal(self, pid, signum):
        self.signals.append((pid,signum))
        if self.on_signal: self.on_signal(pid,signum)
        if signum == signal.SIGSTOP: self.processes[pid]["state"] = "T"
        elif signum == signal.SIGCONT: self.processes[pid]["state"] = "S"
        elif signum == signal.SIGKILL:
            del self.processes[pid]
            for row in self.processes.values():
                if row["ppid"] == pid: row["ppid"] = 1
        else:
            raise AssertionError("unexpected signal, especially SIGTERM/killpg")

    def finish_queue(self, *, terminate=True):
        receipt = supervisor.read_json(self.plan["current_queue"]["status_path"])
        receipt.update(status="complete", completed_stages=20)
        for row in receipt["stages"]: row["status"] = "complete"
        write(self.plan["current_queue"]["status_path"],receipt)
        if terminate:
            self.processes.pop(119795,None)
            self.processes.pop(119924,None)

    def sleep(self, seconds):
        self.clock += seconds
        self.sleep_calls += 1
        if self.on_sleep: self.on_sleep(seconds)
        else: self.finish_queue()

    def monotonic(self): return self.clock
    def now(self): return 1000 + self.clock

    def spawn(self, stage, log):
        if 119795 in self.processes or 119924 in self.processes:
            raise AssertionError("new stage overlaps the still-running predecessor")
        self.spawned.append(copy.deepcopy(stage))
        return FakeChild(self,stage)


class PrioritySupervisorTests(unittest.TestCase):
    def run_fixture(self, callback):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            plan=make_plan(root)
            return callback(root,plan,FakeRuntime(plan))

    def test_success_kills_only_stopped_dispatchers_and_leaves_training_untouched(self):
        def case(root,plan,runtime):
            parent_before=(root/"parent.json").read_bytes()
            spectral_before=(root/"spectral.json").read_bytes()
            seen=[]
            def observe(pid,signum):
                self.assertNotIn(pid,(119795,119924))
                self.assertEqual(runtime.processes[119795]["state"],"R")
                self.assertEqual(runtime.processes[119924]["state"],"R")
                if signum==signal.SIGKILL:
                    self.assertEqual(runtime.processes[pid]["state"],"T")
                    receipt=supervisor.read_json(Path(plan["output_root"])/"handoff.json")
                    self.assertEqual(receipt["status"],"committed")
                    seen.append(pid)
            runtime.on_signal=observe
            result=supervisor.execute(plan,runtime=runtime)
            self.assertEqual(result["status"],"complete")
            self.assertEqual(set(seen),{101639,103819})
            self.assertEqual([s for _,s in runtime.signals],[signal.SIGSTOP,signal.SIGSTOP,signal.SIGKILL,signal.SIGKILL])
            self.assertEqual(runtime.spawned[0]["env"],plan["stages"][0]["env"])
            self.assertEqual((root/"parent.json").read_bytes(),parent_before)
            self.assertEqual((root/"spectral.json").read_bytes(),spectral_before)
            handoff=supervisor.read_json(Path(plan["output_root"])/"handoff.json")
            self.assertEqual(len(handoff["termination_signals"]),2)
            self.assertFalse(result["gpu_lock_owned"])
        self.run_fixture(case)

    def test_post_stop_validation_failure_rolls_back_only_our_parents(self):
        def case(root,plan,runtime):
            def change(pid,signum):
                if pid==103819 and signum==signal.SIGSTOP:
                    write(root/"spectral.json",dict(status="running",active_pid=777))
            runtime.on_signal=change
            with self.assertRaisesRegex(ValueError,"differs"):
                supervisor.execute(plan,runtime=runtime)
            self.assertEqual(runtime.signals,[(101639,signal.SIGSTOP),(103819,signal.SIGSTOP),
                                             (103819,signal.SIGCONT),(101639,signal.SIGCONT)])
            self.assertEqual(runtime.processes[119795]["state"],"R")
            self.assertEqual(runtime.processes[119924]["state"],"R")
            self.assertFalse((Path(plan["output_root"])/"handoff.json").exists())
            self.assertFalse(runtime.spawned)
        self.run_fixture(case)

    def test_second_stop_failure_resumes_first_only(self):
        def case(root,plan,runtime):
            def fail(pid,signum):
                if pid==103819 and signum==signal.SIGSTOP: raise PermissionError("simulated")
            runtime.on_signal=fail
            with self.assertRaises(PermissionError): supervisor.execute(plan,runtime=runtime)
            self.assertEqual(runtime.signals[-1],(101639,signal.SIGCONT))
            self.assertNotIn((103819,signal.SIGCONT),runtime.signals)
            self.assertFalse(any(s==signal.SIGKILL for _,s in runtime.signals))
        self.run_fixture(case)

    def test_reused_pid_rejected_before_any_signals(self):
        def case(root,plan,runtime):
            runtime.processes[101639]["startticks"]+=1
            with self.assertRaisesRegex(RuntimeError,"identity differs"):
                supervisor.execute(plan,runtime=runtime)
            self.assertEqual(runtime.signals,[])
        self.run_fixture(case)

    def test_parent_starts_different_child_before_stop_causes_rollback(self):
        def case(root,plan,runtime):
            def changed(pid,signum):
                if pid==101639 and signum==signal.SIGSTOP:
                    runtime.processes[777]=dict(identity(777),state="R",ppid=101639)
            runtime.on_signal=changed
            with self.assertRaisesRegex(RuntimeError,"children changed"):
                supervisor.execute(plan,runtime=runtime)
            self.assertFalse(any(s==signal.SIGKILL for _,s in runtime.signals))
            self.assertEqual(runtime.processes[777]["state"],"R")
        self.run_fixture(case)

    def test_queue_missing_without_complete_receipt_is_failure_not_readiness(self):
        def case(root,plan,runtime):
            runtime.on_sleep=lambda _:runtime.processes.pop(119795,None)
            with self.assertRaisesRegex(RuntimeError,"without complete20"):
                supervisor.execute(plan,runtime=runtime)
            self.assertFalse(runtime.spawned)
            self.assertFalse(any(s==signal.SIGCONT for _,s in runtime.signals))
            self.assertEqual(runtime.processes[119924]["state"],"R")
            state=supervisor.read_json(Path(plan["output_root"])/"status.json")
            self.assertTrue(state["committed_handoff"])
        self.run_fixture(case)

    def test_complete_receipt_alone_waits_for_actual_queue_exit(self):
        def case(root,plan,runtime):
            def progress(_):
                runtime.finish_queue(terminate=runtime.sleep_calls>=2)
            runtime.on_sleep=progress
            supervisor.execute(plan,runtime=runtime)
            self.assertEqual(runtime.sleep_calls,2)
        self.run_fixture(case)

    def test_wait_timeout_does_not_interrupt_candidate_queue_or_trainer(self):
        def case(root,plan,runtime):
            plan["predecessor_timeout_s"]=1
            runtime.on_sleep=lambda _:None
            with self.assertRaises(TimeoutError): supervisor.execute(plan,runtime=runtime)
            self.assertEqual(runtime.processes[119795]["state"],"R")
            self.assertEqual(runtime.processes[119924]["state"],"R")
            self.assertFalse(runtime.spawned)
        self.run_fixture(case)

    def test_partial_committed_termination_never_automatically_resumes_old_parents(self):
        def case(root,plan,runtime):
            def fail(pid,signum):
                if pid==101639 and signum==signal.SIGKILL: raise PermissionError("simulated partial commit")
            runtime.on_signal=fail
            with self.assertRaises(PermissionError): supervisor.execute(plan,runtime=runtime)
            self.assertNotIn(103819,runtime.processes)
            self.assertEqual(runtime.processes[101639]["state"],"T")
            self.assertEqual(runtime.processes[119795]["state"],"R")
            self.assertFalse(any(s==signal.SIGCONT for _,s in runtime.signals))
            self.assertTrue(supervisor.read_json(Path(plan["output_root"])/"status.json")["committed_handoff"])
        self.run_fixture(case)

    def test_stage_failure_no_retry_and_no_output_restart(self):
        def case(root,plan,runtime):
            runtime.stage_exit=3
            with self.assertRaisesRegex(RuntimeError,"no automatic retry"):
                supervisor.execute(plan,runtime=runtime)
            self.assertEqual(len(runtime.spawned),1)
            with self.assertRaisesRegex(ValueError,"already started"):
                supervisor.execute(plan,runtime=runtime)
            self.assertEqual(len(runtime.spawned),1)
        self.run_fixture(case)

    def test_no_inherited_environment_or_legacy_dependency_waiter(self):
        def case(root,plan,runtime):
            del plan["stages"][0]["env"]["PYTHONPATH"]
            with self.assertRaisesRegex(ValueError,"explicit"):
                supervisor.validate_plan(plan)
            plan["stages"][0]["env"]["PYTHONPATH"]="/new"
            plan["stages"][0]["command"].append("spectral_training.after_dependencies")
            with self.assertRaisesRegex(ValueError,"old spectral"):
                supervisor.validate_plan(plan)
        self.run_fixture(case)

    def test_source_and_verbatim_command_bindings(self):
        def case(root,plan,runtime):
            old=root/"original_plan.json"
            write(old,{"stages":[{"command":plan["stages"][0]["command"]}]})
            plan["stages"][0]["verbatim_reference"]={"path":str(old),"pointer":"/stages/0/command"}
            supervisor.verify_sources(plan)
            plan["stages"][0]["command"]=plan["stages"][0]["command"]+["changed"]
            with self.assertRaisesRegex(ValueError,"retained command"):
                supervisor.verify_sources(plan)
        self.run_fixture(case)


if __name__=="__main__": unittest.main()

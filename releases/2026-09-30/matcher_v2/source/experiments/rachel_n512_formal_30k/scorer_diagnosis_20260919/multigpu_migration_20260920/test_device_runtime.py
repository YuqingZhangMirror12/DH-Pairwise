"""No Torch/CUDA or experiment execution: assigned-device lock/dispatch tests."""
import argparse
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import device_runtime as runtime

A = "GPU-11111111-1111-1111-1111-111111111111"
B = "GPU-22222222-2222-2222-2222-222222222222"


class DeviceRuntimeTests(unittest.TestCase):
    def test_delegated_proxy_never_queries_or_initializes_gpu(self):
        args = runtime.parser().parse_args(["--gpu-uuid", B, "--lock-root", "/new/gpu_locks",
            "--receipt", "/new/lane1/runtime_edge_multi.json", "--module", "matched_only.train"])
        with patch.object(runtime, 'original_dispatch_gate', return_value=True) as gate, \
                patch.object(runtime, 'inventory') as inventory, \
                patch.object(runtime, 'DeviceLease') as lease:
            result = runtime.run(args)
        self.assertEqual(result['status'], 'delegated_complete')
        gate.assert_called_once_with(Path('/new/dynamic_pool'), Path(args.receipt))
        inventory.assert_not_called()
        lease.assert_not_called()

    def test_physical_mapping_exposes_uuid_and_keeps_logical_zero(self):
        query = Mock(return_value=f"0, {A}, NVIDIA RTX 4090 D\n6, {B}, NVIDIA RTX 4090 D\n")
        rows = runtime.inventory(query)
        self.assertEqual(runtime.resolve_gpu("6", rows), runtime.resolve_gpu(B, rows))
        env = runtime.visible_environment(runtime.resolve_gpu("6", rows), {"CUDA_VISIBLE_DEVICES": "0,1"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], B)
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        with self.assertRaises(ValueError):
            runtime.resolve_gpu("GPU-222", rows)
        with self.assertRaises(ValueError):
            runtime.inventory(Mock(return_value=f"0,{A},x\n0,{B},y\n"))

    def test_foreign_gpu_does_not_block_and_assigned_gpu_does(self):
        query = Mock(return_value=f"{B}, 456\n{A}, 123\n")
        self.assertEqual(runtime.foreign_pids(A, pid=123, check_output=query), [])
        command = query.call_args.args[0]
        self.assertEqual(command[command.index("-i") + 1], A)
        query.return_value += f"{A}, 789\n"
        self.assertEqual(runtime.foreign_pids(A, pid=123, check_output=query), [789])
        query.return_value = f"{A}, [N/A]\n"
        with self.assertRaises(RuntimeError):
            runtime.foreign_pids(A, pid=123, check_output=query)

    def test_same_uuid_lock_conflicts_but_other_uuid_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            empty = Mock(return_value="")
            first = runtime.DeviceLease(A, temp, check_output=empty)
            second = runtime.DeviceLease(A, temp, check_output=empty)
            other = runtime.DeviceLease(B, temp, check_output=empty)
            self.assertEqual(first.path, second.path)
            self.assertNotEqual(first.path, other.path)
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=A):
                with first.gpu_lock():
                    with first.gpu_lock("/ignored/legacy/heatmap-gpu.lock"):
                        self.assertEqual(first.depth, 2)
                    with self.assertRaises(BlockingIOError):
                        with second.gpu_lock():
                            self.fail("duplicate UUID owner must fail")
                    with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=B):
                        with other.gpu_lock():
                            self.assertEqual(other.depth, 1)
                with second.gpu_lock():
                    self.assertEqual(second.depth, 1)
            self.assertTrue(first.path.exists())  # Never unlink/replace lock inode.

    def test_busy_failure_releases_lock_and_visibility_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            query = Mock(return_value=f"{A}, 999999\n")
            lease = runtime.DeviceLease(A, temp, check_output=query)
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=A):
                with self.assertRaisesRegex(RuntimeError, "occupied"):
                    with lease.gpu_lock():
                        self.fail("same-GPU compute must block")
                self.assertEqual(lease.depth, 0)
                self.assertIsNone(lease.handle)
                query.return_value = ""
                with lease.gpu_lock():
                    pass
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=B):
                with self.assertRaisesRegex(RuntimeError, "visibility"):
                    with lease.gpu_lock():
                        pass

    def test_direct_from_import_aliases_and_owner_constant_are_patched(self):
        def original(*args, **kwargs):
            raise AssertionError("old global guard used")
        owner = ModuleType(runtime.PREFIX + "candidate_local.train")
        owner.gpu_lock, owner.GPU_LOCK = original, Path("/global.lock")
        alias = ModuleType(runtime.PREFIX + "matcher_convergence.train_continuation")
        alias.gpu_lock = original
        unrelated = ModuleType("unrelated")
        unrelated.gpu_lock = original
        with tempfile.TemporaryDirectory() as temp:
            lease = runtime.DeviceLease(A, temp, check_output=Mock(return_value=""))
            bindings = runtime.patch_lock_aliases(lease, {m.__name__: m for m in (owner, alias, unrelated)})
            self.assertEqual(len(bindings), 2)
            self.assertIs(owner.gpu_lock.__self__, lease)
            self.assertIs(alias.gpu_lock.__self__, lease)
            self.assertIs(unrelated.gpu_lock, original)
            self.assertEqual(owner.GPU_LOCK, lease.path)

    def test_separate_matcher_guard_is_scoped_too(self):
        module = ModuleType(runtime.PREFIX + "matcher_convergence.evaluate_matcher_simval")
        module.exclusive_gpu = lambda path: None
        lease = runtime.DeviceLease(A, "/tmp/device-runtime-unused-test", check_output=Mock(return_value=""))
        runtime.patch_lock_aliases(lease, {module.__name__: module})
        self.assertIs(module.exclusive_gpu.__self__, lease)

    def test_queues_cpu_bridge_and_unknown_entrypoints_rejected(self):
        for name in ("candidate_local.queue", "spectral_training.queue", "matched_only.priority_supervisor",
                     "continuation.run_continuation_queue", "arbitrary.training"):
            with self.assertRaises(ValueError):
                runtime.check_entrypoint(runtime.PREFIX + name, [])
        with self.assertRaises(ValueError):
            runtime.check_entrypoint("matcher_convergence.scorer_bridge", ["cache"])
        self.assertEqual(runtime.check_entrypoint(runtime.PREFIX + "matcher_convergence.scorer_bridge", ["train"]),
                         "matcher_convergence.scorer_bridge")

    def test_original_parser_run_and_preflight_dispatch(self):
        def parser():
            p = argparse.ArgumentParser()
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--resume", action="store_true")
            return p
        module = SimpleNamespace(parser=parser, run=Mock(return_value={"status": "unit-only"}))
        runtime.invoke(module, "candidate_local.train", ["--resume"])
        received = module.run.call_args.args[0]
        self.assertTrue(received.resume)
        self.assertEqual(received.device, "cuda:0")
        def eval_parser():
            p = argparse.ArgumentParser()
            p.add_argument("--execute", action="store_true")
            return p
        evaluate = SimpleNamespace(parser=eval_parser, preflight=Mock(return_value={"pair_ids": ["unit"], "ok": True}),
                                   execute=Mock(return_value="unit-execute"))
        self.assertEqual(runtime.invoke(evaluate, "matcher_convergence.evaluate_continuation", []), {"ok": True})
        evaluate.execute.assert_not_called()
        self.assertEqual(runtime.invoke(evaluate, "matcher_convergence.evaluate_continuation", ["--execute"]), "unit-execute")

    def test_cli_accepts_root_runner_shape_with_explicit_receipt(self):
        args = runtime.parser().parse_args(["--gpu-uuid", B, "--lock-root", "/new/locks",
            "--receipt", "/new/stages/one.runtime.json", "--module", "matched_only.train", "--", "--resume"])
        self.assertEqual(args.gpu, B)
        self.assertIsNone(args.source_root)
        self.assertEqual(args.arguments, ["--", "--resume"])

    def test_cold_smoke_initializes_cuda_before_original_run_only(self):
        order = []
        def parser():
            p = argparse.ArgumentParser()
            p.add_argument("--device", default="cuda:0")
            return p
        cuda = SimpleNamespace(init=Mock(side_effect=lambda: order.append("cuda.init")))
        module = SimpleNamespace(parser=parser, torch=SimpleNamespace(cuda=cuda),
            run=Mock(side_effect=lambda args: order.append("run:" + args.device)))
        runtime.invoke(module, "matched_only.smoke", [])
        self.assertEqual(order, ["cuda.init", "run:cuda:0"])
        order.clear()
        cuda.init.reset_mock()
        runtime.invoke(module, "matched_only.train", [])
        self.assertEqual(order, ["run:cuda:0"])
        cuda.init.assert_not_called()


if __name__ == "__main__":
    unittest.main()

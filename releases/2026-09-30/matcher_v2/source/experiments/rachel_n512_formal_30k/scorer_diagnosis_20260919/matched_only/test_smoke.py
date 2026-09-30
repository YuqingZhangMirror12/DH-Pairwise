"""CPU numerical helpers + explicitly mocked CLI CUDA guard, never remote."""
from contextlib import contextmanager,redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import data,smoke,stage_cache,train
from . import test_training as fixtures


@contextmanager
def full_fixture():
    with tempfile.TemporaryDirectory() as directory,patch.dict(data.FULL_COUNTS,{"train":32}), \
            patch.object(train,"TRAIN_COUNT",32):
        root=Path(directory);base=root/"base"
        fixtures.make_cache(base,count=32)
        source=data.FormalCache(base,"train")
        side=root/"stages";stage_cache.prepare(source,side)
        yield root,base,side,source


def args(root,base,side,arm):
    return SimpleNamespace(arm=arm,train_cache=str(base),
        stage_cache=str(side) if arm in stage_cache.STAGES else None,
        output=str(root/("smoke_"+arm)),device="cuda:0")


@contextmanager
def mocked_cpu_guard(root):
    """Only tests replace environment/device; shared file-lock remains real."""
    original=train.lock_owner.gpu_lock
    held=[]
    @contextmanager
    def lock():
        with original(root/"heatmap-gpu.lock",check_gpu=False):
            held.append(True)
            try: yield
            finally: held.pop()
    def guard(device):
        if device!="cuda:0" or not held: raise AssertionError("CUDA guard must follow child lock")
        return torch.device("cpu")
    with patch.object(train.lock_owner,"gpu_lock",lock),patch.object(smoke,"require_cuda",side_effect=guard):
        yield


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_all5_helpers_two_steps_fp32_finite_disposable_and_same_fresh_init(self):
        with full_fixture() as (root,base,side,source), \
                patch.object(torch,"save",side_effect=AssertionError("smoke must never save weights")):
            feature_sha=data.sha(base/"features_a.npy")
            for arm in train.ARMS:
                ds=source if arm not in stage_cache.STAGES else data.CandidateStageCache(source,side,arm)
                net=train.make_scorer(arm);initial=train.state_digest(net)
                result=smoke.two_steps(net,ds,np.arange(32),"cpu")
                self.assertEqual(result["samples"],32)
                self.assertEqual(result["optimizer_updates"],2)
                self.assertEqual(result["physical_microbatch"],16)
                self.assertEqual(result["effective_batch"],16)
                self.assertEqual(result["formal_optimizer_updates"],0)
                self.assertFalse(result["training_counted"])
                self.assertFalse(result["checkpoint_usable_for_formal"])
                self.assertTrue(result["finite_gradient_check"])
                self.assertTrue(result["finite_parameters_after"])
                self.assertGreater(result["throughput_pairs_s"],0.)
                self.assertEqual(len(result["steps"]),2)
                self.assertNotEqual(train.state_digest(net),initial)
                self.assertEqual(train.state_digest(train.make_scorer(arm)),initial)
                self.assertTrue(all(p.dtype==torch.float32 for p in net.parameters()))
                self.assertTrue(all(r["active_gradient_tensors"]>0 for r in result["steps"]))
                self.assertFalse(set(result)&{"model_state_dict","optimizer_state_dict","checkpoint"})
                json.dumps(result,allow_nan=False)
            self.assertEqual(data.sha(base/"features_a.npy"),feature_sha)

    def test_mocked_full_entry_all5_json_only_source_binding_and_peaks_unavailable_cpu(self):
        with full_fixture() as (root,base,side,source),mocked_cpu_guard(root), \
                patch.object(torch,"save",side_effect=AssertionError("no checkpoint export")):
            for arm in train.ARMS:
                request=args(root,base,side,arm)
                result=smoke.run(request)
                out=Path(request.output)
                self.assertEqual({p.name for p in out.iterdir()},{"protocol.json","results.json"})
                self.assertEqual(result["status"],"complete")
                self.assertIsNone(result["peak_allocated_bytes"])
                self.assertIsNone(result["peak_reserved_bytes"])
                self.assertTrue(result["source_binding_unchanged"])
                protocol=json.loads((out/"protocol.json").read_text())
                self.assertEqual(protocol["execution_device"],"cpu") # explicitly mocked, not GPU evidence
                self.assertFalse(protocol["training_counted"])
                self.assertFalse(protocol["validation_performed"])
                self.assertFalse(protocol["real_ood_used"])
                self.assertEqual(protocol["formal_pair_exposures"],0)
                self.assertEqual(protocol["initial_model_sha256"],train.state_digest(train.make_scorer(arm)))

    def test_nonfinite_gradient_fails_before_optimizer_step(self):
        net=fixtures.TinyModel()
        next(net.parameters()).register_hook(lambda g:torch.full_like(g,float("inf")))
        receipts=[]
        with self.assertRaises(RuntimeError):
            smoke.two_steps(net,fixtures.TinyData(32,"train"),np.arange(32),"cpu",receipts=receipts)
        self.assertEqual(receipts,[])

    def test_failed_cli_receipt_is_not_complete_or_usable_for_training(self):
        with full_fixture() as (root,base,side,source),mocked_cpu_guard(root):
            make=train.make_scorer
            def broken(arm,seed):
                net=make(arm,seed=seed)
                # First registered parameter is the no-evidence fallback and
                # has no gradient on this all-valid all_tokens fixture.
                net.head.classifier[-1].weight.register_hook(lambda g:torch.full_like(g,float("inf")))
                return net
            request=args(root,base,side,"all_tokens")
            with patch.object(train,"make_scorer",side_effect=broken):
                with self.assertRaises(RuntimeError):smoke.run(request)
            out=Path(request.output)
            self.assertEqual({p.name for p in out.iterdir()},{"protocol.json"})
            protocol=json.loads((out/"protocol.json").read_text())
            self.assertEqual(protocol["status"],"failed")
            self.assertEqual(protocol["completed_disposable_steps"],0)
            self.assertFalse(protocol["checkpoint_usable_for_formal"])
            self.assertFalse(protocol["training_counted"])

    def test_probe_or_incomplete_source_rejected_before_gpu_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);base=root/"tiny"
            fixtures.make_cache(base,count=4)
            request=args(root,base,None,"all_tokens")
            with patch.object(train.lock_owner,"gpu_lock",side_effect=AssertionError("lock too early")) as lock:
                with self.assertRaises(ValueError):smoke.run(request)
                lock.assert_not_called()
            self.assertFalse(Path(request.output).exists())

    def test_busy_child_lock_does_not_initialize_model_or_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);request=args(root,root/"base",None,"all_tokens")
            with patch.object(smoke,"load_dataset",return_value=object()), \
                    patch.object(train.lock_owner,"gpu_lock",side_effect=BlockingIOError("busy")), \
                    patch.object(train,"make_scorer",side_effect=AssertionError("must not allocate")) as factory:
                with self.assertRaises(BlockingIOError):smoke.run(request)
                factory.assert_not_called()
            self.assertFalse(Path(request.output).exists())

    def test_strict32_fp32_and_cli_gpu_only_no_batch_resume_controls(self):
        net=fixtures.TinyModel();ds=fixtures.TinyData(32,"train")
        for indices in (np.arange(16),np.zeros(32,dtype=int),np.arange(1,33)):
            with self.assertRaises(ValueError):smoke.two_steps(net,ds,indices,"cpu")
        with self.assertRaises(ValueError):smoke.two_steps(net.double(),ds,np.arange(32),"cpu")
        parsed=smoke.parser().parse_args(["--arm","edge_multi","--train-cache","a","--stage-cache","b","--output","c"])
        for forbidden in ("batch","resume","epochs","limit","real","val"):
            self.assertFalse(hasattr(parsed,forbidden))
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            smoke.parser().parse_args(["--arm","all_tokens","--train-cache","a","--output","c","--device","cpu"])
        with patch.object(smoke.platform,"system",return_value="Linux"), \
                patch.object(torch.cuda,"is_available",return_value=False):
            with self.assertRaises(RuntimeError):smoke.require_cuda("cuda:0")
        with patch.object(smoke.platform,"system",return_value="Linux"), \
                patch.object(torch.cuda,"is_available",return_value=True), \
                patch.object(torch.cuda,"device_count",return_value=2):
            with self.assertRaises(RuntimeError):smoke.require_cuda("cuda:0")

    def test_peak_metrics_calls_exact_cuda_device(self):
        device=torch.device("cuda:0")
        with patch.object(torch.cuda,"max_memory_allocated",return_value=1024) as allocated, \
                patch.object(torch.cuda,"max_memory_reserved",return_value=2048) as reserved:
            self.assertEqual(smoke.memory_stats(device),dict(peak_allocated_bytes=1024,peak_reserved_bytes=2048))
            allocated.assert_called_once_with(device);reserved.assert_called_once_with(device)


if __name__=="__main__":unittest.main()

"""CPU tests of finite loop ownership/commit/recovery, not GPU training claims."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import train_continuation as train


class ToyMatcher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.nn.Linear(1, 1)
        self.config = SimpleNamespace(contour_cap=512)
        self.phase = "matcher"

    def set_phase(self, phase):
        if phase != "matcher":
            raise AssertionError("classifier transition attempted")
        self.phase = phase
        return self


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])


class TrainContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7); random.seed(7); np.random.seed(7)
        self.source_model = ToyMatcher()
        self.source_optimizer = torch.optim.AdamW(self.source_model.parameters(), lr=2e-5)
        self.source = self.snapshot(self.source_model, self.source_optimizer, 48)
        self.calls = []
        self.fail_number = None
        self.identity = dict(source_frozen_digests={})

    def snapshot(self, model, optimizer, number):
        return dict(completed_segments=number, model_state_dict=deepcopy(model.state_dict()),
            optimizer_state_dict=deepcopy(optimizer.state_dict()), rng_state=rng())

    def restore(self, model, optimizer, checkpoint, identity, **kwargs):
        self.assertEqual(kwargs["rng_mode"], "exact_gpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng(checkpoint["rng_state"])
        return dict(completed_segments=checkpoint["completed_segments"])

    def payload(self, model, optimizer, *, identity, completed_segments, resume_context, role):
        result = self.snapshot(model, optimizer, completed_segments)
        result["checkpoint_role"] = role
        return result

    def kernel(self, model, loader, optimizer, loss, device, args, epoch):
        number, count = loader
        self.calls.append((number, epoch, count))
        self.assertEqual((args.microbatch, args.physical_microbatch, args.effective_batch, args.workers), (1,16,16,4))
        self.assertTrue(all(group["lr"] == 2e-5 for group in optimizer.param_groups))
        optimizer.zero_grad(set_to_none=True)
        # One lightweight synthetic step stands for a segment. Its random
        # inputs exercise Python/NumPy/Torch and Adam state across restarts.
        value = torch.rand(1, 1) + random.random() + float(np.random.random())
        model.base_model(value).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if number == self.fail_number:
            raise RuntimeError("injected after uncommitted update")
        return dict(samples=count, optimizer_updates=count // 16, phase="matcher", pair_bce_weight=0.)

    @contextmanager
    def mocked_core(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(train, "training_identity", return_value={"bound": "same"}))
            stack.enter_context(patch.object(train.core, "restore_initial", side_effect=self.restore))
            stack.enter_context(patch.object(train.core, "restore_continuation", side_effect=self.restore))
            stack.enter_context(patch.object(train.core, "validate_progress", side_effect=lambda c, i, **k: c["completed_segments"]))
            stack.enter_context(patch.object(train.core, "checkpoint_payload", side_effect=self.payload))
            stack.enter_context(patch.object(train.core, "private_train_segment", side_effect=self.kernel))
            stack.enter_context(patch.object(train, "_loader", side_effect=lambda t,s,n,c: (s["number"],n)))
            stack.enter_context(patch.object(train.core, "validate_optimizer", return_value=[]))
            stack.enter_context(patch.object(train.core, "frozen_digests", return_value={}))
            stack.enter_context(patch("sys.stdout", new=io.StringIO()))
            yield

    def execute(self, directory, *, stop=20, resume=False, smoke=None):
        model = ToyMatcher()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.5)
        args = SimpleNamespace(output=str(directory), stop_after_epoch=stop, resume=resume,
            smoke=smoke, device="cuda:0", source="unused_in_locked_loop")
        result = train.execute(args, model, optimizer, object(), object(), self.source,
            self.identity, torch.device("cpu"))
        return result, model, optimizer

    def test_fixed_m16_pause_then_m20_resume_without_replaying_commits(self):
        with tempfile.TemporaryDirectory() as directory, self.mocked_core():
            result, _, _ = self.execute(directory, stop=16)
            self.assertEqual((result["status"], result["completed_segments"]), ("paused",64))
            self.assertFalse(result["ready_for_fixed_matcher_evaluation"])
            result, _, _ = self.execute(directory, resume=True)
            self.assertEqual((result["status"], result["completed_segments"]), ("training_complete",80))
            self.assertTrue(result["ready_for_fixed_matcher_evaluation"])
            self.assertFalse(result["full_experiment_complete"])
            self.assertEqual([n for n,e,c in self.calls], list(range(49,81)))
            self.assertEqual(sorted(p.name for p in Path(directory).glob("epoch_*.pt")),
                ["epoch_%03d.pt" % e for e in range(13,21)])
            saved = torch.load(Path(directory)/"epoch_020.pt", weights_only=False)
            self.assertEqual(saved["checkpoint_role"], "epoch_anchor")

    def test_interruption_repeats_only_uncommitted_segment_and_restores_adam_rng(self):
        with tempfile.TemporaryDirectory() as reference, tempfile.TemporaryDirectory() as restarted, self.mocked_core():
            _, expected_model, expected_optimizer = self.execute(reference)
            expected_rng = rng()
            self.calls = []; self.fail_number = 50
            with self.assertRaisesRegex(RuntimeError, "uncommitted"):
                self.execute(restarted)
            saved = torch.load(Path(restarted)/"last.pt", weights_only=False)
            self.assertEqual(saved["completed_segments"],49)
            status = json.loads((Path(restarted)/"status.json").read_text())
            self.assertEqual(status["last_committed_segment"],49)
            self.fail_number = None
            _, actual_model, actual_optimizer = self.execute(restarted, resume=True)
            self.assertEqual([n for n,e,c in self.calls], [49,50]+list(range(50,81)))
            for name, value in actual_model.state_dict().items():
                self.assertTrue(torch.equal(value,expected_model.state_dict()[name]))
            for key, values in actual_optimizer.state_dict()["state"].items():
                for name, value in values.items():
                    self.assertTrue(torch.equal(value,expected_optimizer.state_dict()["state"][key][name]))
            actual_rng = rng()
            self.assertEqual(actual_rng["python"],expected_rng["python"])
            np.testing.assert_array_equal(actual_rng["numpy"][1],expected_rng["numpy"][1])
            self.assertTrue(torch.equal(actual_rng["torch"],expected_rng["torch"]))

    def test_foreign_resume_binding_rejected_before_restore_or_updates(self):
        with tempfile.TemporaryDirectory() as directory, self.mocked_core():
            self.execute(directory, stop=16)
            path = Path(directory)/"protocol.json"
            record = json.loads(path.read_text()); record["training_identity"]={"foreign":True}
            path.write_text(json.dumps(record))
            self.calls=[]
            with self.assertRaisesRegex(ValueError,"ownership"):
                self.execute(directory,resume=True)
            self.assertEqual(self.calls,[])

    def test_gpu_discard32_path_never_writes_checkpoint_or_formal_budget(self):
        with tempfile.TemporaryDirectory() as directory, self.mocked_core():
            result, _, _ = self.execute(directory, smoke=32)
            self.assertEqual(self.calls,[(49,13,32)])
            self.assertEqual(result["discarded_optimizer_updates"],2)
            self.assertFalse(result["formal_training_counted"])
            self.assertEqual(list(Path(directory).glob("*.pt")),[])
            self.assertFalse(json.loads((Path(directory)/"protocol.json").read_text())["formal_training_counted"])

    def test_loader_exact_original_order_seed_and_slice(self):
        segment=train.core.continuation_plan()[5]
        with patch.object(train.old.runner,"epoch_indices",return_value=tuple(range(24000))) as order, \
                patch.object(train.old,"make_weathering_loader") as loader:
            training=object()
            train._loader(training,segment,6000,512)
            order.assert_called_once_with(24000,seed=260913,epoch=14,limit=None)
            self.assertEqual(loader.call_args.args[1],tuple(range(6000,12000)))
            self.assertEqual(loader.call_args.kwargs,dict(batch_size=16,num_workers=4,seed=260967,contour_cap=512))

    def test_busy_shared_gpu_lock_prevents_any_source_load_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"source.pt"; source.touch()
            output=Path(directory)/"unused"
            args=train.parser().parse_args(["--source",str(source),"--output",str(output)])
            with patch.object(train,"gpu_lock",side_effect=BlockingIOError("busy")), \
                    patch.object(train.torch,"load") as load:
                with self.assertRaises(BlockingIOError): train.run(args)
                load.assert_not_called()
            self.assertFalse(output.exists())

    def test_output_isolation_and_disallow_smoke_resume(self):
        args=train.parser().parse_args(["--output","unused","--smoke","32","--resume"])
        with self.assertRaises(ValueError): train.validate_arguments(args)
        populations=dict(train=dict(manifest="/immutable/data/train.json"),val=dict(manifest="/clean/pairs/val.jsonl"))
        for output in (Path("/immutable"),Path("/immutable/training/new"),Path("/clean/new")):
            with self.assertRaises(ValueError):
                train.validate_output(output,"/immutable/training/epoch_012.pt",populations)


if __name__ == "__main__":
    unittest.main()

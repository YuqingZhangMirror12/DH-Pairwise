"""Tiny CPU continuation wiring checks, not formal training results."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from . import continuation_core as core
from experiments.rachel_n512_formal_30k.test_train_score_decoupled import model, tiny_loader


class MatcherContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schedule_continues_matcher_not_classifier(self):
        plan = core.continuation_plan()
        self.assertEqual((len(plan),plan[0]["number"],plan[-1]["number"]),(32,49,80))
        self.assertEqual(sum(r["count"] for r in plan),192000)
        self.assertEqual((plan[0]["epoch"],plan[-1]["epoch"]),(13,20))
        self.assertTrue(all(r["phase"]=="matcher" and r["learning_rate"]==2e-5 for r in plan))
        self.assertEqual(sum(bool(r["epoch_complete"]) for r in plan),8)

    def test_phase_rebinding_does_not_change_old_trainer(self):
        self.assertIs(core.private_train_segment.__code__,core.old.train_segment.__code__)
        self.assertIsNot(core.private_train_segment.__globals__,core.old.train_segment.__globals__)
        for epoch in (13,16,20):
            self.assertEqual(core.old.phase_for_epoch(epoch),"classifier")
            self.assertEqual(core.private_phase_for_epoch(epoch),"matcher")
        for value in (None,True,12,21,13.):
            with self.assertRaises(ValueError):
                core.private_phase_for_epoch(value)

    def test_real_training_kernel_at_epoch13_updates_only_matcher(self):
        net = model("cross_attention").set_phase("matcher").train()
        optimizer = core.old.create_optimizer(net)
        active = [p for p in net.parameters() if p.requires_grad]
        # Explicitly artificial moments: tests mechanics, not the real S7 source.
        for p in active:
            p.grad = torch.ones_like(p)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for value in optimizer.state.values():
            value["step"].fill_(18000)
        for group in optimizer.param_groups:
            group["lr"] = 2e-5
        frozen = dict(head=deepcopy(net.score_head.state_dict()),
            coarse=deepcopy(net.base_model.coarse.state_dict()),
            local=deepcopy(net.base_model.local_head.state_dict()),
            fusion=deepcopy(net.base_model.fusion.state_dict()))
        before = core.old.state_digest(net.base_model)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=directory,microbatch=1,physical_microbatch=16,
                effective_batch=16,log_every=1000)
            report = core.private_train_segment(net,tiny_loader(16,16,balanced=True),optimizer,
                core.old.RachelN512LossConfig(),torch.device("cpu"),args,13)
        self.assertEqual(report["phase"],"matcher")
        self.assertEqual(report["pair_bce_weight"],0.)
        self.assertEqual((report["samples"],report["optimizer_updates"]),(16,1))
        self.assertNotEqual(before,core.old.state_digest(net.base_model))
        for key,module in (("head",net.score_head),("coarse",net.base_model.coarse),
                           ("local",net.base_model.local_head),("fusion",net.base_model.fusion)):
            self.assertEqual(set(frozen[key]),set(module.state_dict()))
            for name,value in module.state_dict().items():
                self.assertTrue(torch.equal(value,frozen[key][name]),(key,name))
        self.assertTrue(all(int(optimizer.state[p]["step"])==18001 for p in active))
        self.assertTrue(all(p.grad is None for p in net.parameters()))


if __name__ == "__main__":
    unittest.main()

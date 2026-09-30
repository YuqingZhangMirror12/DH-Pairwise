"""Numerical probe helper tests; no checkpoints, data population, training or GPU."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import geometry

# probe.py is also a standalone remote script and deliberately imports its
# adjacent geometry module absolutely. Supply that module for package tests;
# restore its script-only process environment changes after this inert import.
with patch.dict(sys.modules, {"geometry": geometry}), patch.dict(os.environ), \
        patch.object(sys, "dont_write_bytecode", sys.dont_write_bytecode):
    spec = importlib.util.spec_from_file_location("physical_probe_under_test", Path(__file__).with_name("probe.py"))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def snapshots(self):
        valid = (torch.tensor([[True, True, False]]), torch.tensor([[False, True, False]]))
        # Production patches are[B,N,S,1,P,P], not[B,N,S,P,P].
        ref = dict(patches=tuple(torch.zeros(1,3,4,1,2,2) for _ in range(2)),
                   encoded=tuple(torch.ones(1,3,2) for _ in range(2)),
                   context=tuple(torch.ones(1,3,2)*2 for _ in range(2)))
        cur = {key:tuple(x.clone() for x in values) for key,values in ref.items()}
        # Very large/nonfinite padding must neither change metrics nor fail a
        # finite valid-token comparison.
        for side in range(2):
            for key in ("patches", "encoded", "context"):
                cur[key][side][~valid[side]] = float("nan")
        return ref, cur, valid

    def test_padding_excluded_and_patch_rates_are_one_number_per_scale(self):
        ref,cur,valid = self.snapshots()
        cur["patches"][0][0,0,2,0,0,0] = 1
        cur["patches"][1][0,1,0] = 1
        cur["encoded"][0][0,1,0] += 2
        cur["context"][1][0,1,1] += 3
        result = probe.compare_snapshots(ref,cur,valid)
        self.assertEqual(result["a"]["patch_difference_rate_by_scale"], [0.,0.,1/8,0.])
        self.assertEqual(result["b"]["patch_difference_rate_by_scale"], [1.,0.,0.,0.])
        self.assertEqual(result["a"]["encoded"], dict(relative_l2=1., absolute_l2=2., max_abs=2.))
        self.assertEqual(result["a"]["context"]["absolute_l2"], 0.)
        self.assertEqual(result["b"]["encoded"]["absolute_l2"], 0.)
        self.assertEqual(result["b"]["context"]["absolute_l2"], 3.)

    def test_empty_valid_and_nonfinite_valid_features_reject(self):
        ref,cur,valid = self.snapshots()
        with self.assertRaises(ValueError):
            probe.compare_snapshots(ref,cur,(torch.zeros_like(valid[0]),valid[1]))
        cur["encoded"][0][0,0,0] = float("inf")
        with self.assertRaisesRegex(ValueError,"finite"):
            probe.compare_snapshots(ref,cur,valid)
        self.assertIsNone(probe.difference(torch.zeros(2),torch.ones(2))["relative_l2"])

    def test_factorial_is_raw_logit_interaction_not_probability(self):
        values = dict(R=1.,B=10.,C=4.,D=3.)
        rows = {key:dict(logit=value, probability=999.) for key,value in values.items()}
        self.assertEqual(probe.factorial(rows),4.)
        rows["B"]["logit"] = 6.
        self.assertEqual(probe.factorial(rows),0.)

    def test_summary_signed_absolute_drift_and_strict_closeness(self):
        rows = []
        for i, reference in enumerate((1.,5.)):
            branches = {k:dict(logit=values[i]) for k,values in
                        dict(A=(4.,1.),B=(2.,7.),C=(1.5,8.),D=(0.,5.)).items()}
            identity = {k:dict(logit=reference) for k in "ABCD"}
            rows.append(dict(R=dict(logit=reference),scales={
                "0.5":dict(branches=branches,factorial_logit_interaction=probe.factorial(dict(branches,R=dict(logit=reference)))),
                "0.75":dict(branches=identity,factorial_logit_interaction=0.)}))
        result = probe.summarize(rows)
        half = result["scales"]["0.5"]
        self.assertEqual(result["case_count"],2)
        self.assertEqual(result["metrics"],"score sensitivity only; not accuracy")
        self.assertEqual(half["branches"]["A"],dict(mean_signed_logit_drift=-.5,
            mean_absolute_logit_drift=3.5, median_absolute_logit_drift=3.5,max_absolute_logit_drift=4.))
        self.assertEqual(half["B_closer_to_R_than_A_count"],2)
        self.assertEqual(half["C_closer_to_R_than_B_count"],1)
        self.assertEqual(half["mean_signed_interaction"],.25)
        self.assertEqual(half["mean_absolute_interaction"],1.25)
        self.assertEqual(result["scales"]["0.75"]["B_closer_to_R_than_A_count"],0)

    def test_encode_and_score_argument_order_inference_only(self):
        test = self
        masks = (torch.ones(1,1,2,2),torch.ones(1,1,2,2)*2)
        points = (torch.ones(1,3,2)*3,torch.ones(1,3,2)*4)
        valid = (torch.tensor([[True,True,False]]),torch.tensor([[True,False,True]]))
        sampled, encoded_calls, context_calls, head_calls = [],[],[],[]

        class FakeSampler(torch.nn.Module):
            def forward(self, mask, coords, keep):
                side = len(sampled)
                test.assertIs(mask,masks[side]);test.assertIs(coords,points[side]);test.assertIs(keep,valid[side])
                test.assertFalse(torch.is_grad_enabled())
                result = mask.new_full((1,3,4,1,2,2),float(side+1))
                sampled.append(result)
                return result

        class FakeBase:
            def _encode_patches(self, patches, keep):
                side = len(encoded_calls)
                test.assertIs(patches,sampled[side]);test.assertIs(keep,valid[side])
                test.assertFalse(torch.is_grad_enabled())
                value = torch.full((1,3,2),float(side+10))
                encoded_calls.append(value)
                return value

            def context(self, fa, fb, va, vb, pa, pb, canvas):
                for got,wanted in zip((fa,fb,va,vb,pa,pb),(*encoded_calls,*valid,*points)):
                    test.assertIs(got,wanted)
                test.assertEqual(canvas,800);test.assertFalse(torch.is_grad_enabled())
                context_calls.append(True)
                return fa+pa,fb+pb

        class FakeHead(torch.nn.Module):
            def forward(self, ca, cb, va, vb):
                test.assertIs(va,valid[0]);test.assertIs(vb,valid[1])
                test.assertFalse(torch.is_grad_enabled())
                head_calls.append((ca,cb))
                return ca[va].sum()-cb[vb].sum()

        base = FakeBase()
        evidence = probe.encode(base,FakeSampler(),masks,points,valid)
        result,snapshot = probe.score(SimpleNamespace(base_model=base,score_head=FakeHead()),evidence,points,valid)
        self.assertEqual(result["logit"],-8.)
        self.assertAlmostEqual(result["probability"],float(torch.tensor(-8.).sigmoid()))
        self.assertEqual(len(context_calls),1);self.assertEqual(len(head_calls),1)
        self.assertIs(snapshot["patches"],evidence["patches"])
        self.assertIs(snapshot["encoded"],evidence["encoded"])
        self.assertIs(snapshot["context"][0],head_calls[0][0])

    def test_score_rejects_more_than_one_or_nonfinite_logit(self):
        base = SimpleNamespace(context=lambda *args:(torch.ones(1),torch.ones(1)))
        evidence = dict(encoded=(torch.ones(1),torch.ones(1)))
        for answer in (torch.tensor([1.,2.]),torch.tensor([float("nan")])):
            model = SimpleNamespace(base_model=base,score_head=lambda *args:answer)
            with self.assertRaisesRegex(ValueError,"one finite"):
                probe.score(model,evidence,(None,None),(None,None))


if __name__=="__main__":
    unittest.main()

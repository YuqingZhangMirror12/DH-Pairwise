from dataclasses import replace
import unittest

import torch

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from experiments.rachel_n512_formal_30k.decoupled_samplewise_loss import compute_samplewise_phase_loss
from .config import Plateau,TrainingConfig
from .data import recipe_for
from .scratch_matcher import fresh_matcher,matching_loss
from .test_matcher import inputs


class TrainingContractTests(unittest.TestCase):
    def test_binding_json_and_checkpoint_types_are_identical(self):
        import json
        from .train import canonical_record
        record=canonical_record(TrainingConfig().record())
        self.assertEqual(record,json.loads(json.dumps(record)))
        self.assertIsInstance(record['proposals']['normal_vote_fractions'],list)

    def test_plateau_observes_both_lr_reductions_before_stopping(self):
        cfg=TrainingConfig();state=Plateau();events=[]
        for epoch in range(0,49,2):
            event=state.observe(.5,epoch,cfg)
            if event!='continue':
                events.append((epoch,event))
            if event not in ('continue','reduce_lr'):
                break
        self.assertEqual(events,[(16,'reduce_lr'),(22,'reduce_lr'),(28,'simulation_plateau_after_lr_reductions')])

    def test_improving_at_budget_does_not_claim_convergence(self):
        cfg=TrainingConfig();state=Plateau()
        for epoch in range(0,49,2):
            action=state.observe(epoch*.002,epoch,cfg)
        self.assertEqual(action,'budget_limit_not_claimed_converged')
        self.assertEqual(state.reductions,0)

    def test_recipe_missing_is_not_assumed_clean(self):
        self.assertEqual(recipe_for(dict(view='hard'),dict(changed_pair=True)),'hard_damage_unspecified')
        self.assertEqual(recipe_for(dict(view='hard'),dict(changed_pair=False)),'clean')
        with self.assertRaises(ValueError):
            recipe_for({},dict(changed_pair=False))

    def test_scratch_seed_reproducible_not_reference_tensors(self):
        cfg=RachelN512Config(canvas_size=32,coarse_size=32,contour_cap=16,landmark_count=2,context_layers=2,activation_checkpointing=False)
        a,b=fresh_matcher(cfg,22),fresh_matcher(cfg,22)
        other=fresh_matcher(cfg,23)
        torch.testing.assert_close(a.base.primal.weight,b.base.primal.weight,atol=0,rtol=0)
        self.assertFalse(torch.equal(a.base.primal.weight,other.base.primal.weight))
        self.assertFalse(torch.equal(a.base.primal.weight,a.base.dual.weight))

    def test_matching_loss_matches_original_per_pair_coefficients(self):
        cfg=RachelN512Config(canvas_size=32,coarse_size=32,contour_cap=16,landmark_count=2,context_layers=2,activation_checkpointing=False)
        adapter=fresh_matcher(cfg,22).eval()
        args=inputs();output=adapter(*args)
        a=torch.tensor([[0,1,2,3,-2,-2]])
        batch=dict(labels=torch.ones(1),target_a=a,target_b=a.clone(),
            translation_a_to_b_rc=torch.tensor([[3.,14.]]),translation_valid=torch.ones(1,dtype=torch.bool),
            pose_enabled=torch.ones(1,dtype=torch.bool))
        old=adapter.base(*args)
        oldloss,_=compute_samplewise_phase_loss(old,(batch['labels'],a,a,batch['translation_a_to_b_rc'],
            batch['translation_valid']),batch['pose_enabled'],RachelN512LossConfig(),'matcher')
        loss,_,counts=matching_loss(adapter,output,batch)
        torch.testing.assert_close(loss,oldloss)
        self.assertEqual(counts['exact_pose_pairs'],1)
        batch['pose_enabled'].zero_()
        damaged,parts,counts=matching_loss(adapter,output,batch)
        self.assertEqual(float(parts['translation_smooth_l1']),0.)
        self.assertEqual(counts['exact_pose_pairs'],0)
        damaged.backward()
        self.assertGreater(float(adapter.base.primal.weight.grad.abs().sum()),0)
        self.assertGreater(float(adapter.base.dual.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in adapter.base.coarse.parameters()))


if __name__=='__main__':
    unittest.main()

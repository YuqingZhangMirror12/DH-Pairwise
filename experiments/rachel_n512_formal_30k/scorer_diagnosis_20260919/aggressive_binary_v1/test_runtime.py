from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from .runtime import make_model, fresh_patch_head, selected_matcher
from .admission import sha
from ..binary_scorer_v1.loss import pair_loss
from ..s7_consensus_v1.compatibility import CompatibilityConfig
from ..s7_consensus_v1.config import TrainingConfig
from ..s7_consensus_v1.preflight_matcher import state_digest
from ..s7_consensus_v1.targets import PairLabels
from ..s7_consensus_v1.test_matcher import inputs
from ..s7_consensus_v1.test_threshold_joint import setup_pair
from ..s7_consensus_v1.train import TrainModule


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = TrainingConfig()
        self.reference = RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
            landmark_count=2, context_layers=2, activation_checkpointing=False)
        self.geometry = CompatibilityConfig(.5, .5, .5, .5, 1., 15.)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def model(self, stage='matcher', selected=None):
        return make_model(self.reference, self.geometry, self.config, stage, selected)

    def saved(self, model):
        path = self.root / 'new_matcher.pt'
        # A deliberately poisoned old head must be ignored by the phase bridge.
        state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        for name in state:
            if name.startswith('head.'): state[name].fill_(99.)
        torch.save(dict(stage='matcher', epoch=16, model=state, optimizer={'poison': True}), path)
        return dict(path=str(path), sha256=sha(path))

    def test_random_initialization_seed_and_rng_isolation(self):
        rng = torch.get_rng_state().clone()
        a, receipt = self.model(); b, _ = self.model()
        self.assertEqual(state_digest(a), state_digest(b))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(receipt['initialization'], 'random_seed_no_weight_import')
        self.assertFalse(receipt['reference_weights_imported'])
        self.assertTrue(any(p.requires_grad for p in a.matcher.parameters()))
        self.assertFalse(any(p.requires_grad for p in a.head.parameters()))

    def test_matcher_stage_rejects_import_and_scorer_requires_selected(self):
        with self.assertRaises(ValueError): self.model(selected={'path':'E32'})
        with self.assertRaises(ValueError): self.model('scorer')

    def test_actual_matcher_backward_changes_matcher_only(self):
        model, _ = self.model(); head_sha = state_digest(model.head); matcher_sha = state_digest(model.matcher)
        args = inputs(); target = torch.tensor([[0,1,2,3,-2,-2]])
        keys = ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b')
        batch = dict(zip(keys,args)); batch.update(labels=torch.ones(1), target_a=target, target_b=target.clone(),
            translation_a_to_b_rc=torch.tensor([[3.,14.]]), translation_valid=torch.ones(1,dtype=torch.bool),
            pose_enabled=torch.zeros(1,dtype=torch.bool))
        raw = TrainModule(model, 'matcher', self.config)
        optimizer = torch.optim.AdamW([p for p in raw.parameters() if p.requires_grad], lr=1e-4)
        loss, _, counts = raw(batch); loss.backward(); optimizer.step()
        self.assertNotEqual(matcher_sha, state_digest(model.matcher))
        self.assertEqual(head_sha, state_digest(model.head))
        self.assertEqual(counts['exact_pose_pairs'], 0)
        self.assertTrue(all(p.grad is None for p in model.head.parameters()))

    def test_bridge_imports_only_new_matcher_and_fresh_head(self):
        model, _ = self.model()
        with torch.no_grad(): model.matcher.base.primal.weight.add_(.123)
        selected = self.saved(model)
        head_model, receipt = self.model('scorer', selected)
        self.assertEqual(state_digest(model.matcher), state_digest(head_model.matcher))
        self.assertEqual(state_digest(head_model.head), state_digest(fresh_patch_head(self.config.head_seed)))
        head_model.train()
        self.assertFalse(any(p.requires_grad for p in head_model.matcher.parameters()))
        self.assertFalse(head_model.matcher.base.training)
        self.assertTrue(all(p.requires_grad for p in head_model.head.parameters()))
        self.assertFalse(receipt['old_head_imported']); self.assertFalse(receipt['optimizer_imported'])
        optimizer = torch.optim.AdamW([p for p in head_model.parameters() if p.requires_grad], lr=1e-4)
        self.assertEqual(len(optimizer.state), 0)

    def test_actual_production_head_step_preserves_full_matcher(self):
        initial, _ = self.model(); model, _ = self.model('scorer', self.saved(initial))
        matcher_sha = state_digest(model.matcher); head_sha = state_digest(model.head)
        _, pair, proposals, _ = setup_pair()
        pair = replace(pair, **{name:getattr(pair,name).detach().repeat(1,24)
                               for name in ('local_a','local_b','context_a','context_b')})
        n = len(pair.q); labels = PairLabels(True, True, torch.tensor([7.,0.]),
            torch.arange(n), torch.arange(n), torch.ones(n,dtype=torch.bool), torch.ones(n,dtype=torch.bool))
        loss = pair_loss(model, model.score_pair(pair, proposals=proposals), labels).total
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
        loss.backward(); optimizer.step()
        self.assertNotEqual(head_sha, state_digest(model.head))
        self.assertEqual(matcher_sha, state_digest(model.matcher))
        self.assertEqual(sum(p.numel() for p in model.head.parameters()), 34529)

    def selected_fixture(self):
        model, _ = self.model(); root = self.root / 'matcher'; root.mkdir()
        common = dict(data_contract_sha256='NEW', geometry_calibration_sha256='NEW-CAL', admission={},
            implementation_sha256={}, aggressive_implementation_sha256={}, binary_scorer_sha256={},
            config={'synthetic_fixture':True})
        binding = dict(common, stage='matcher', matcher_initialization='random_seed_no_weight_import')
        curve = [dict(epoch=e, key=[1. if e==0 else .8 if e==4 else .7, .5, -1.]) for e in range(0,17,2)]
        best = dict(epoch=4,key=curve[2]['key'],threshold=.3)
        torch.save(dict(binding=binding, stage='matcher', epoch=4, model=model.state_dict()), root/'best_joint.pt')
        selection = dict(status='selected', binding=binding, selection_on_real=False, migration_origin=None,
            actual_epochs=16, updates=12000, exposures=384000, best=best,
            best_joint_sha256=sha(root/'best_joint.pt'), stop_reason='synthetic_test_fixture')
        for name,value in (('selection.json',selection),('complete.json',dict(selection,status='stage_complete')),
                           ('learning_curve.json',curve)):
            (root/name).write_text(json.dumps(value))
        return root, common, selection

    def test_selection_excludes_epoch0_and_uses_new_trained_matcher(self):
        root, binding, _ = self.selected_fixture()
        receipt = selected_matcher(root, binding, self.config)
        self.assertEqual(receipt['epoch'], 4)

    def test_wrong_dataset_or_incomplete_terminal_rejected(self):
        root, binding, _ = self.selected_fixture()
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            selected_matcher(root, dict(binding,data_contract_sha256='v14'), self.config)
        (root/'failure.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'failure'): selected_matcher(root,binding,self.config)

    def test_modified_selected_checkpoint_rejected(self):
        root, binding, _ = self.selected_fixture()
        with (root/'best_joint.pt').open('ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'file changed'): selected_matcher(root,binding,self.config)

    def test_nonwinning_or_missing_validation_cannot_be_imported(self):
        root, binding, _ = self.selected_fixture()
        curve = json.loads((root/'learning_curve.json').read_text()); curve[3]['key'][0] = .95
        (root/'learning_curve.json').write_text(json.dumps(curve))
        with self.assertRaisesRegex(ValueError, 'selection differs'): selected_matcher(root,binding,self.config)
        (root/'learning_curve.json').write_text(json.dumps(curve[:-1]))
        with self.assertRaisesRegex(ValueError, 'history incomplete'): selected_matcher(root,binding,self.config)


if __name__ == '__main__': unittest.main()

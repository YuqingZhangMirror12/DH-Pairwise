"""Synthetic CPU construction/export tests; not formal training evidence."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from ..curriculum_training_v1.checkpoint_io import file_sha, tree_sha
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.runtime_plan import lock_record, check_light_heads
from ..curriculum_training_v1.test_validation_adapter import setup_plan
from ..curriculum_training_v1.training_core import Topology
from ..s7_consensus_v1.compatibility import CompatibilityConfig
from ..s7_consensus_v1.scratch_matcher import fresh_matcher
from ..s7_consensus_v1.test_matcher import inputs
from .additive_exposure import build_additive_ledger
from .test_additive_exposure import extras
from .runtime_schedule import compile_plan
from .model_runtime import make_components, load_export_matcher, load_export_scorer
from .validation import RULES


def plan_fixture(arm='B3', module='matcher'):
    base, template = setup_plan(module)
    record = copy.deepcopy(template.record); record['selection_rule']['id'] = RULES[module]
    original = lock_record(record, base)
    if arm == 'B2':return compile_plan(original, base, arm)
    extra = build_additive_ledger(base, extras(), ((0, 3, 1), (3, 7, 3), (7, 12, 2), (12, 16, 2)), seed=base.seed)
    return compile_plan(original, base, arm, additive=extra, admitted_data_sha256=digest('synthetic combined admission'))


def exported_fixture(parts, updates=7):
    """Actual tensors, explicitly fabricated metadata, NOT a terminal receipt."""
    b = parts['binding']
    return dict(schema='curriculum-model-export/1', binding=b, module=b['module'], order=b['order'],
                selection_kind='sim_best', updates=updates, model=copy.deepcopy(parts['model'].state_dict()))


class ModelRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path(__file__).resolve().parents[4]
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.arch = RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
            landmark_count=2, context_layers=2, activation_checkpointing=False)
        self.geometry = CompatibilityConfig(.5, .5, .5, .5, 1., 15.)

    def build(self, arm='B3', module='matcher', selected=None, cache=None):
        _, plan, schedule = plan_fixture(arm, module)
        return make_components(plan, schedule, Topology(0, 1, 4, 1), self.root,
                               self.arch, self.geometry, selected=selected, cache=cache)

    def selected(self, arm='B3'):
        parts = self.build(arm); model = parts['model']
        with torch.no_grad():
            for p in model.matcher.parameters():
                if p.requires_grad:p.add_(.003)
            for p in model.head.parameters():p.add_(.5)  # Must NOT be imported.
        path = self.out / (arm+'.pt'); torch.save(exported_fixture(parts), path)
        return dict(path=str(path), sha256=file_sha(path), updates=7, source_binding=parts['binding'])

    def test_B1_exact_legacy_initialization_and_forward(self):
        parts = self.build('B1'); matcher = parts['model'].matcher
        old = fresh_matcher(self.arch, parts['binding']['common_plan']['model_seed'])
        self.assertEqual(tree_sha(old.state_dict()), tree_sha(matcher.state_dict()))
        self.assertIsNone(matcher.upgrades)
        torch.testing.assert_close(old(*inputs()).assignment, matcher(*inputs()).assignment, atol=0, rtol=0)

    def test_B2_B3_identical_random_models_no_rng_consumption(self):
        before = torch.get_rng_state().clone(); a = self.build('B2'); b = self.build('B3')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(tree_sha(a['model'].state_dict()), tree_sha(b['model'].state_dict()))
        self.assertTrue(all(p.requires_grad for p in b['model'].matcher.upgrades.parameters()))
        self.assertFalse(any(p.requires_grad for p in b['model'].head.parameters()))

    def test_each_head_imports_all_selected_matcher_weights_and_new_head(self):
        selected = self.selected(); saved = torch.load(selected['path'], weights_only=False)
        a = self.build(module='scorer_patch', selected=selected)
        b = self.build(module='scorer_stats', selected=selected)
        self.assertEqual(check_light_heads(a['binding'], b['binding'])['status'], 'matched')
        expected = {k[8:]: v for k, v in saved['model'].items() if k.startswith('matcher.')}
        for parts, count in ((a, 34529), (b, 3201)):
            self.assertEqual(tree_sha(parts['model'].matcher.state_dict()), tree_sha(expected))
            self.assertFalse(any(p.requires_grad for p in parts['model'].matcher.parameters()))
            self.assertEqual(sum(p.numel() for p in parts['model'].head.parameters()), count)
            parts['module'].train()
            self.assertFalse(parts['model'].matcher.base.training)
            self.assertFalse(parts['model'].matcher.upgrades.training)
        old = {k[5:]: v for k, v in saved['model'].items() if k.startswith('head.')}
        self.assertNotEqual(tree_sha(a['model'].head.state_dict()), tree_sha(old))

    def test_export_loads_nonzero_v2_not_legacy_constructor(self):
        selected = self.selected(); saved = torch.load(selected['path'], weights_only=False)
        before = torch.get_rng_state().clone(); restored = load_export_matcher(saved, self.root)
        parts = self.build(module='scorer_patch', selected=selected)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(tree_sha(restored.state_dict()), tree_sha(parts['model'].matcher.state_dict()))
        torch.testing.assert_close(restored(*inputs()).assignment,
                                   parts['model'].matcher(*inputs()).assignment, atol=0, rtol=0)

    def test_full_scorer_export_roundtrip_both_variants(self):
        selected = self.selected()
        for module in ('scorer_patch', 'scorer_stats'):
            parts = self.build(module=module, selected=selected)
            with torch.no_grad():
                for p in parts['model'].head.parameters():p.add_(.01)
            saved = exported_fixture(parts); restored = load_export_scorer(saved, self.root)
            self.assertEqual(tree_sha(restored.state_dict()), tree_sha(parts['model'].state_dict()))
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
        with self.assertRaisesRegex(ValueError, 'trained Scorer'):
            load_export_scorer(torch.load(selected['path'], weights_only=False), self.root)

    def test_cross_arm_selected_matcher_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'own arm'):
            self.build('B1', 'scorer_patch', selected=self.selected('B3'))

    def test_missing_explicit_architecture_or_parameter_is_rejected(self):
        saved = exported_fixture(self.build())
        del saved['binding']['model_spec']['matcher_implementation']
        with self.assertRaises(ValueError):load_export_matcher(saved, self.root)
        saved = exported_fixture(self.build()); del saved['model']['matcher.upgrades.log_sharpness']
        with self.assertRaises(RuntimeError):load_export_matcher(saved, self.root)

    def test_export_nonfinite_or_wrong_config_is_rejected(self):
        saved = exported_fixture(self.build())
        saved['model']['matcher.upgrades.log_sharpness'].fill_(float('nan'))
        with self.assertRaisesRegex(ValueError, 'non-finite'):load_export_matcher(saved, self.root)
        saved = exported_fixture(self.build())
        saved['binding']['model_spec']['matcher_implementation']['matcher_v2']['use_cross'] = False
        with self.assertRaisesRegex(ValueError, 'architecture'):load_export_matcher(saved, self.root)

    def test_schedule_mismatch_unknown_arm_and_matcher_cache_rejected(self):
        _, plan, schedule = plan_fixture(); schedule['arm'] = 'B4'
        with self.assertRaises(ValueError):make_components(plan, schedule, Topology(0, 1, 4, 1), self.root, self.arch, self.geometry)
        _, plan, schedule = plan_fixture(); schedule['actual_updates'] += 1
        with self.assertRaises(ValueError):make_components(plan, schedule, Topology(0, 1, 4, 1), self.root, self.arch, self.geometry)
        with self.assertRaisesRegex(ValueError, 'cache'):self.build(cache=object())


if __name__ == '__main__':unittest.main()

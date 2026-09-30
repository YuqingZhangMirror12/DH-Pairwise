import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .frozen import (ARMS, TrainingConfig, terminal_training, validate_selection,
                     verify_scratch_matcher, verify_fixed_matcher, registered_protocol, sha, training)


def fixture(arm='m12'):
    binding = dict(arm=arm, formal_training=True, preflight_steps=0,
                   config=training.canonical_record(TrainingConfig().record()))
    chosen = dict(epoch=22, key=[.8, .9, .95], threshold=.3)
    cp = dict(stage='scorer', binding=binding, epoch=22, threshold=.3,
              metrics=dict(key=chosen['key'], threshold=.3))
    sel = dict(status='selected', binding=binding, best=chosen, actual_epochs=32,
               updates=24000, exposures=768000, best_joint_sha256='selected_hash',
               matcher_unchanged=True, selection_on_real=False,
               stop_reason='simulation_plateau_after_lr_reductions')
    complete = dict(sel, status='stage_complete')
    terminal = dict(status='training_complete', arm=arm, binding=binding, last_stage=complete,
                    stages=['matcher', 'scorer'] if arm == 'scratch' else ['scorer'])
    return cp, sel, complete, terminal


class FrozenSelectionTests(unittest.TestCase):
    def test_parallel_scope_never_accepts_an_unfinished_selected_arm(self):
        module = terminal_training.__module__
        with tempfile.TemporaryDirectory() as temp, patch(
                module+'.registered_protocol', return_value=dict(variant='threshold')):
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'selected arm must finish'):
                terminal_training(root, completed_arm='m12')
            selected = root/'formal_m12'; selected.mkdir()
            (selected/'training_complete.json').write_text(json.dumps(fixture()[-1]))
            # The other arm has not finished; no receipts or metrics are read.
            self.assertEqual(set(terminal_training(root, completed_arm='m12')), {'m12'})
            with self.assertRaisesRegex(ValueError, 'both arms must finish'):
                terminal_training(root)
            (selected/'failure.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'failure'):
                terminal_training(root, completed_arm='m12')

    def test_parallel_scope_is_not_a_legacy_or_unknown_arm_bypass(self):
        module = terminal_training.__module__
        with patch(module+'.registered_protocol', return_value=dict(variant='mergefix')):
            with self.assertRaisesRegex(ValueError, 'scope is not registered'):
                terminal_training('/unopened', completed_arm='m12')
        with patch(module+'.registered_protocol', return_value=dict(variant='simple')):
            with self.assertRaisesRegex(ValueError, 'scope is not registered'):
                terminal_training('/unopened', completed_arm='unregistered')

    def test_parallel_scope_keeps_terminal_stage_contract(self):
        module = terminal_training.__module__
        with tempfile.TemporaryDirectory() as temp, patch(
                module+'.registered_protocol', return_value=dict(variant='simple')):
            root = Path(temp); selected = root/'formal_m12'; selected.mkdir()
            record = fixture()[-1]; record['stages'] = ['matcher']
            (selected/'training_complete.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'invalid terminal'):
                terminal_training(root, completed_arm='m12')

    def test_both_arms_selected_joint_only(self):
        for arm in ARMS:
            self.assertEqual(validate_selection(*fixture(arm), arm)['arm'], arm)

    def test_reject_premature_real_or_test_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'both arms must finish'):
                terminal_training(root)
            m12 = root / 'formal_m12'; m12.mkdir()
            (m12 / 'training_complete.json').write_text(json.dumps(fixture()[-1]))
            with self.assertRaisesRegex(ValueError, 'scratch'):
                terminal_training(root)
            other = root / ('formal_'+ARMS[1]); other.mkdir()
            (other / 'training_complete.json').write_text(json.dumps(fixture(ARMS[1])[-1]))
            self.assertEqual(set(terminal_training(root)), set(ARMS))
            (other / 'failure.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'failure'):
                terminal_training(root)

    def test_reject_diagnostic_layout_checkpoint(self):
        x = list(fixture()); x[0] = dict(x[0], epoch=30)
        with self.assertRaisesRegex(ValueError, 'selected joint'):
            validate_selection(*x, 'm12')

    def test_reject_changed_threshold(self):
        x = list(fixture()); x[0] = dict(x[0], threshold=.4)
        with self.assertRaisesRegex(ValueError, 'selected joint'):
            validate_selection(*x, 'm12')

    def test_reject_preflight_and_wrong_architecture(self):
        for key, value in [('formal_training', False), ('preflight_steps', 12), ('config', {})]:
            x = copy.deepcopy(fixture()); x[0]['binding'][key] = value
            with self.assertRaisesRegex(ValueError, 'formal Scorer'):
                validate_selection(*x, 'm12')

    def test_reject_foreign_binding(self):
        x = list(copy.deepcopy(fixture())); x[1] = dict(x[1], binding={'arm': 'scratch'})
        with self.assertRaisesRegex(ValueError, 'identity differ'):
            validate_selection(*x, 'm12')

    def test_reject_short_or_inconsistent_budget(self):
        for field, value in [('actual_epochs', 2), ('updates', 1), ('exposures', 1),
                             ('matcher_unchanged', False), ('stop_reason', 'process_disappeared')]:
            cp, sel, complete, terminal = copy.deepcopy(fixture())
            sel[field] = complete[field] = value
            with self.assertRaisesRegex(ValueError, 'budget'):
                validate_selection(cp, sel, complete, terminal, 'm12')

    def test_reject_real_selected_weight(self):
        cp, sel, complete, terminal = fixture(); sel['selection_on_real'] = True
        with self.assertRaisesRegex(ValueError, 'real-selected'):
            validate_selection(cp, sel, complete, terminal, 'm12')

    def test_epoch_zero_best_remains_visible_if_selected(self):
        cp, sel, complete, terminal = fixture()
        cp['epoch'] = sel['best']['epoch'] = 0
        self.assertEqual(validate_selection(cp, sel, complete, terminal, 'm12')['arm'], 'm12')

    def test_scratch_matcher_must_equal_its_sim_selected_state(self):
        binding = {'arm': 'scratch'}
        best = dict(epoch=16,key=[.9,.8,-.1])
        selection = dict(status='selected',binding=binding,best=best,
                         selection_on_real=False,best_joint_sha256='hash')
        complete = dict(selection,status='stage_complete')
        checkpoint = dict(stage='matcher',epoch=16,binding=binding,metrics={'key':best['key']},
                          model={'matcher.test':torch.tensor([1.,2.]),'head.test':torch.tensor([0.])})
        scorer = dict(model={'matcher.test':torch.tensor([1.,2.]),'head.test':torch.tensor([99.])})
        with patch('{}.read'.format(verify_scratch_matcher.__module__),side_effect=[selection,complete]*2), \
             patch('{}.sha'.format(verify_scratch_matcher.__module__),return_value='hash'), \
             patch('torch.load',return_value=checkpoint):
            receipt=verify_scratch_matcher(Path('/unopened'),binding,scorer)
            self.assertEqual(receipt['selected_epoch'],16)
            scorer['model']['matcher.test'][0]=3.
            with self.assertRaisesRegex(ValueError,'changed in Scorer'):
                verify_scratch_matcher(Path('/unopened'),binding,scorer)

    def test_protocol_derived_from_exact_bound_revision(self):
        self.assertEqual(registered_protocol({})['arms'], ('m12', 'scratch'))
        config = dict(proposal_revision='native-hypothesis-complete-link-union/1-diameter16',
            threshold_policy=dict(pose_diameter_px=16., candidate_budget=8, maximum_interpenetration_sum=.1))
        self.assertEqual(registered_protocol(config)['arms'], ('m12', 'scratch_fixed'))
        config['threshold_policy']['pose_diameter_px'] = 20.
        with self.assertRaisesRegex(ValueError, 'fixed16'):
            registered_protocol(config)
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            registered_protocol({'proposal_revision': 'unreviewed'})


class FixedMatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'new'; self.root.mkdir()
        self.path = Path(self.temp.name)/'prior/formal_scratch/matcher/best_joint.pt'
        self.path.parent.mkdir(parents=True)
        source = self.path.parents[2]/'source'/Path(*training.__package__.split('.'))
        source.mkdir(parents=True)
        (source/'fixture.py').write_text('# synthetic source binding fixture\n')
        common = dict(data_contract_sha256='contract', geometry_calibration_sha256='geometry',
                      reference_checkpoint_sha256='reference')
        origin = dict(**common, arm='scratch', formal_training=True, preflight_steps=0,
                      implementation_sha256={'fixture.py': sha(source/'fixture.py')})
        best = dict(epoch=32, key=[.99, .99, -.1])
        cp = dict(stage='matcher', binding=origin, epoch=32, metrics={'key': best['key']},
                  model={'matcher.test': torch.tensor([1., 2.]), 'head.test': torch.tensor([9.])})
        torch.save(cp, self.path)
        self.selection = dict(status='selected', binding=origin, best=best,
            actual_epochs=32, updates=24000, exposures=768000,
            stop_reason='simulation_plateau_after_lr_reductions', selection_on_real=False,
            best_joint_sha256=sha(self.path))
        self.binding = dict(**common, arm='scratch_fixed', fixed_matcher=dict(path=str(self.path),
            sha256=sha(self.path), head_imported=False, matcher_training=False))
        self.imported = dict(path=str(self.path), sha256=sha(self.path), epoch=32,
                             old_head_imported=False, optimizer_imported=False, matcher_frozen=True)
        (self.root/'formal_scratch_fixed').mkdir()
        self.scorer = dict(model={'matcher.test': torch.tensor([1., 2.]), 'head.test': torch.tensor([7.])})
        self.write_records()

    def write_records(self):
        (self.path.parent/'selection.json').write_text(json.dumps(self.selection))
        (self.path.parent/'complete.json').write_text(json.dumps(dict(self.selection, status='stage_complete')))
        (self.root/'formal_scratch_fixed/fixed_matcher_import.json').write_text(json.dumps(self.imported))

    def test_import_has_exact_prior_selected_matcher_but_new_head(self):
        value = verify_fixed_matcher(self.root, self.binding, self.scorer)
        self.assertEqual(value['selected_epoch'], 32)
        self.assertFalse(value['matcher_retrained_for_this_head'])

    def test_reject_real_selected_or_partial_origin(self):
        for key, val in [('selection_on_real', True), ('actual_epochs', 12), ('updates', 1)]:
            with self.subTest(key=key):
                before = self.selection[key]
                self.selection[key] = val; self.write_records()
                with self.assertRaisesRegex(ValueError, 'completed simulation'):
                    verify_fixed_matcher(self.root, self.binding, self.scorer)
                self.selection[key] = before

    def test_reject_altered_matcher_or_missing_tensor(self):
        self.scorer['model']['matcher.test'][0] = 3.
        with self.assertRaisesRegex(ValueError, 'changed in Scorer'):
            verify_fixed_matcher(self.root, self.binding, self.scorer)
        del self.scorer['model']['matcher.test']
        with self.assertRaisesRegex(ValueError, 'membership'):
            verify_fixed_matcher(self.root, self.binding, self.scorer)

    def test_reject_changed_calibration_import_or_source(self):
        self.binding['geometry_calibration_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            verify_fixed_matcher(self.root, self.binding, self.scorer)
        self.binding['geometry_calibration_sha256'] = 'geometry'
        self.imported['old_head_imported'] = True; self.write_records()
        with self.assertRaisesRegex(ValueError, 'import receipt'):
            verify_fixed_matcher(self.root, self.binding, self.scorer)
        self.imported['old_head_imported'] = False; self.write_records()
        source = self.path.parents[2]/'source'/Path(*training.__package__.split('.'))/'fixture.py'
        source.write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'source binding'):
            verify_fixed_matcher(self.root, self.binding, self.scorer)


if __name__ == '__main__':
    unittest.main()

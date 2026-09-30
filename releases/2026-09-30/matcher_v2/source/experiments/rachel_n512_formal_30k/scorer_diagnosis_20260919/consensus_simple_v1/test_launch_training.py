"""Fail-closed checks for the independent simple training controller."""
from copy import deepcopy
import unittest

from .launch_training import validate_gate, validate_simple_config


class GateTests(unittest.TestCase):
    def setUp(self):
        self.binding = {'source': 'bound-source', 'data': 'bound-data'}
        self.receipt = dict(status='passed', formal_training=False, updated_weights_discarded=True,
            arm='m12', stage='scorer', updates=12, exposures=384, world_size=2,
            microbatch=8, accumulate=2, effective_batch=32, matcher_unchanged=True,
            model_state_hashes=['sha', 'sha'], binding=self.binding,
            resume_matches_uninterrupted=True)

    def test_valid_uninterrupted_and_replay(self):
        validate_gate(self.receipt, self.binding, 'm12')
        validate_gate(self.receipt, self.binding, 'm12', replay=True)

    def test_rejects_changed_training_protocol(self):
        changes = [('status', 'failed'), ('formal_training', True),
            ('updated_weights_discarded', False), ('arm', 'scratch_fixed'), ('stage', 'matcher'),
            ('updates', 0), ('exposures', 0), ('world_size', 3), ('microbatch', 16),
            ('accumulate', 1), ('effective_batch', 48), ('matcher_unchanged', False)]
        for key, value in changes:
            with self.subTest(key=key):
                receipt = deepcopy(self.receipt)
                receipt[key] = value
                with self.assertRaises(ValueError):
                    validate_gate(receipt, self.binding, 'm12')

    def test_rejects_missing_or_divergent_gpu_hashes(self):
        for hashes in ([], ['sha'], ['sha', 'other'], ['', ''], ['sha', 'sha', 'sha']):
            with self.subTest(hashes=hashes):
                receipt = deepcopy(self.receipt)
                receipt['model_state_hashes'] = hashes
                with self.assertRaises(ValueError):
                    validate_gate(receipt, self.binding, 'm12')

    def test_rejects_foreign_binding(self):
        with self.assertRaises(ValueError):
            validate_gate(self.receipt, {'source': 'different'}, 'm12')

    def test_replay_confirmation_is_required(self):
        receipt = deepcopy(self.receipt)
        del receipt['resume_matches_uninterrupted']
        with self.assertRaises(ValueError):
            validate_gate(receipt, self.binding, 'm12', replay=True)


class SimpleConfigTests(unittest.TestCase):
    def config(self):
        from types import SimpleNamespace
        result = SimpleNamespace(world_size=2, microbatch=8, accumulate=2,
                                 effective_batch=32, minimum_epochs=16, maximum_epochs=48)
        result.record = lambda: {
            'proposal_revision': 'raw-displacement-modes-simple/1-sim16',
            'simple_policy': dict(pose_radius_px=16., candidate_budget=8,
                                  maximum_interpenetration_sum=.10)}
        return result

    def test_registered_simple_config(self):
        validate_simple_config(self.config())

    def test_rejects_threshold_builder_or_changed_policy(self):
        for record in (
            {'proposal_revision': 'threshold16'},
            {'proposal_revision': 'raw-displacement-modes-simple/1-sim16',
             'simple_policy': dict(pose_radius_px=10., candidate_budget=8,
                                  maximum_interpenetration_sum=.10)},
        ):
            cfg = self.config()
            cfg.record = lambda: record
            with self.assertRaises(ValueError):
                validate_simple_config(cfg)

    def test_rejects_changed_batch_or_budget(self):
        for key, value in [('world_size', 4), ('microbatch', 16), ('accumulate', 1),
                           ('effective_batch', 64), ('minimum_epochs', 8),
                           ('maximum_epochs', 60)]:
            cfg = self.config()
            setattr(cfg, key, value)
            with self.assertRaises(ValueError):
                validate_simple_config(cfg)


if __name__ == '__main__':
    unittest.main()


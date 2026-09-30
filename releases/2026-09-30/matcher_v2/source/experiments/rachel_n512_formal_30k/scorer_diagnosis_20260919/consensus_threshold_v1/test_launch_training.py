"""Fail-closed checks for the independent threshold training controller."""
from copy import deepcopy
import unittest

from .launch_training import validate_gate


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


if __name__ == '__main__':
    unittest.main()

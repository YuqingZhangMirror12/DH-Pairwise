"""A short successful run is not authorization to reuse different inputs."""
import copy
import unittest

from .launch import validate_gate_receipt


class LaunchGateTests(unittest.TestCase):
    def fixture(self,arm='m12'):
        binding=dict(arm=arm,data_contract_sha256='new-24k',geometry_calibration_sha256='new-geometry',
            reference_checkpoint_sha256='m12',implementation_sha256={'train.py':'numeric-repair'},
            config={'effective_batch':32},formal_training=False,preflight_steps=12)
        receipt=dict(status='passed',resume_matches_uninterrupted=True,formal_training=False,
            updated_weights_discarded=True,arm=arm,stage='scorer' if arm=='m12' else 'matcher',
            world_size=2,microbatch=8,accumulate=2,effective_batch=32,updates=12,exposures=384,
            model_state_hashes=['equal','equal'],matcher_unchanged=True,binding=copy.deepcopy(binding))
        return receipt,binding

    def test_both_registered_arms_pass(self):
        for arm in ('m12','scratch'):
            gate,binding=self.fixture(arm)
            validate_gate_receipt(gate,arm,gate['stage'],binding)

    def test_old_data_geometry_source_or_config_rejected(self):
        for key in ('data_contract_sha256','geometry_calibration_sha256','reference_checkpoint_sha256',
                    'implementation_sha256','config'):
            gate,binding=self.fixture()
            gate['binding'][key]='old'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'binding differs'):
                validate_gate_receipt(gate,'m12','scorer',binding)

    def test_two_updates_not_enough_to_cover_prior_failure(self):
        gate,binding=self.fixture();gate.update(updates=2,exposures=64)
        with self.assertRaisesRegex(ValueError,'gate failed'):
            validate_gate_receipt(gate,'m12','scorer',binding)

    def test_frozen_matcher_and_distributed_resume_required(self):
        for change in ({'matcher_unchanged':False},{'resume_matches_uninterrupted':False},
                       {'model_state_hashes':['a','b']},{'updated_weights_discarded':False},
                       {'world_size':1},{'stage':'matcher'}):
            gate,binding=self.fixture();gate.update(change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'gate failed'):
                validate_gate_receipt(gate,'m12','scorer',binding)


if __name__=='__main__':
    unittest.main()

"""Same-call export tests run against each independent immutable model source."""
import copy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .audit import validate_arrays
from .frozen import TrainingConfig, registered_protocol
from .snapshot import snapshot_prediction
from .view_data import build_case
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.consensus_head import ConsensusEvidenceHead
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.model import S7Consensus
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_evidence import fixture


class VersionSnapshotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.protocol = registered_protocol(TrainingConfig().record())
        self.model = S7Consensus(None, CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                                head=ConsensusEvidenceHead(feature_dim=4)).eval()
        self.pair = fixture()
        with torch.no_grad():
            self.pred = self.model.score_pair(self.pair, capture_diagnostics=True)
        self.provenance = dict(variant=self.protocol['variant'], evidence_mode=self.protocol['evidence_mode'],
                               purpose='CPU synthetic fixture, not an experimental result')
        with patch.object(self.model, 'score_pair', side_effect=AssertionError('extra inference')):
            self.meta, self.arrays = snapshot_prediction('toy', self.pair, self.pred,
                threshold=.5, provenance=self.provenance)
        self.assertTrue(self.meta['clusters'])

    def rehash(self, key):
        self.meta['arrays'][key]['sha256'] = hashlib.sha256(self.arrays[key].tobytes()).hexdigest()

    def require_union(self):
        if self.protocol['evidence_mode'] != 'exact_union_q':
            self.skipTest('this check applies to the threshold snapshot only')

    def test_registered_mode_actual_values_and_no_additional_forward(self):
        report = validate_arrays(self.meta, self.arrays)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(self.meta['semantics']['evidence_mode'], self.protocol['evidence_mode'])
        for saved, cluster in zip(self.meta['clusters'], self.pred.clusters):
            np.testing.assert_array_equal(self.arrays[saved['stages']['final']['weights']],
                                          cluster.encoded.evidence.weights.detach().numpy())
        self.assertFalse(self.meta['semantics']['attention_weights_exported'])

    def test_provenance_cannot_mislabel_the_bound_variant(self):
        wrong = dict(self.provenance, variant='another variant')
        with self.assertRaisesRegex(ValueError, 'bound implementation'):
            snapshot_prediction('toy', self.pair, self.pred, threshold=.5, provenance=wrong)

    def test_union_weights_and_directional_diagnostic_are_separate(self):
        self.require_union()
        view = build_case(self.meta, self.arrays)
        for saved, displayed in zip(self.meta['clusters'], view['clusters']):
            stage = saved['stages']['final']
            mask = self.arrays[stage['kernels']]
            weights = self.arrays[stage['weights']]
            np.testing.assert_array_equal(weights, self.pair.q.numpy()*mask)
            np.testing.assert_array_equal(self.arrays[stage['directional_q_diagnostic']],
                self.pair.q.numpy()*self.arrays[stage['directional_kernels']])
            self.assertIn('final_directional_diagnostic', displayed['heatmaps'])
            self.assertIn('union_membership', displayed['heatmaps'])
            self.assertEqual(displayed['joint_constraint']['threshold_px'], 16.)
        self.assertIn('NOT used to attenuate', self.meta['semantics']['directional_kernels'])

    def test_union_cannot_be_relabelled_as_directional_recall(self):
        self.require_union()
        self.meta['semantics']['evidence_mode'] = 'directional_full_q'
        with self.assertRaisesRegex(ValueError, 'evidence meaning'):
            validate_arrays(self.meta, self.arrays)

    def test_wrong_diagnostic_q_fails_even_with_recomputed_hash(self):
        self.require_union()
        key = self.meta['clusters'][0]['stages']['final']['directional_q_diagnostic']
        self.arrays[key] = self.arrays[key]+.1
        self.rehash(key)
        with self.assertRaisesRegex(ValueError, 'diagnostic directional Q'):
            validate_arrays(self.meta, self.arrays)

    def test_chain_or_changed16_policy_rejected(self):
        self.require_union()
        self.meta['clusters'][0]['proposal']['pose_diameter_px'] = 32.
        with self.assertRaisesRegex(ValueError, 'fixed16'):
            validate_arrays(self.meta, self.arrays)


if __name__ == '__main__':
    unittest.main()

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .compatibility import CompatibilityConfig
from .consensus_head import ConsensusEvidenceHead
from .diagnostics import snapshot_prediction, union_arc_length, write_snapshot
from .evidence import PairEvidence
from .geometry import compact_contour
from .model import S7Consensus
from .test_evidence import fixture


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = S7Consensus(None, CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                                 head=ConsensusEvidenceHead(feature_dim=4))
        self.pair = fixture()

    def snapshot(self, pair=None, prediction=None):
        pair = self.pair if pair is None else pair
        if prediction is None:
            prediction = self.model.score_pair(pair, capture_diagnostics=True)
        return snapshot_prediction('synthetic-fixture', pair, prediction, threshold=.5,
                                   provenance={'purpose': 'untrained CPU structural test'})

    def test_capture_preserves_prediction_parameters_and_gradients(self):
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        ordinary = self.model.score_pair(self.pair)
        sum(c.readout.logit for c in ordinary.clusters).backward()
        gradients = {k: p.grad.clone() for k, p in self.model.named_parameters() if p.grad is not None}
        self.model.zero_grad(set_to_none=True)
        captured = self.model.score_pair(self.pair, capture_diagnostics=True)
        sum(c.readout.logit for c in captured.clusters).backward()
        for k, p in self.model.named_parameters():
            if k in gradients:
                torch.testing.assert_close(gradients[k], p.grad, rtol=0, atol=0)
        torch.testing.assert_close(ordinary.score, captured.score, rtol=0, atol=0)
        torch.testing.assert_close(ordinary.translation_a_to_b_rc, captured.translation_a_to_b_rc, rtol=0, atol=0)
        self.assertEqual(ordinary.accepted, captured.accepted)
        self.assertTrue(all(c.initial_encoded is None for c in ordinary.clusters))
        self.assertTrue(all(c.initial_encoded is not None for c in captured.clusters))
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(state[k], v, rtol=0, atol=0)

    def test_export_uses_captured_outputs_without_network_replay(self):
        with patch.object(self.model.head, 'forward_many', wraps=self.model.head.forward_many) as call:
            prediction = self.model.score_pair(self.pair, capture_diagnostics=True)
            self.assertEqual(call.call_count, 2)
            rng = torch.get_rng_state().clone()
            metadata, arrays = self.snapshot(prediction=prediction)
            self.assertEqual(call.call_count, 2)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertFalse(metadata['semantics']['attention_weights_exported'])
        self.assertFalse(metadata['semantics']['gt_used_by_exporter'])
        np.testing.assert_array_equal(arrays[metadata['pair']['q']], self.pair.q.numpy())
        for description, actual in zip(metadata['clusters'], prediction.clusters):
            np.testing.assert_array_equal(arrays[description['refinement']['localization_weights']],
                                          actual.refinement.localization_weights.detach().numpy())
            np.testing.assert_array_equal(arrays[description['stages']['final']['pose']],
                                          actual.translation.detach().numpy())
            np.testing.assert_allclose(arrays[description['edge_contributions']['support']].sum(),
                arrays[description['readout']['positive_evidence_px']], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(arrays[description['edge_contributions']['conflict']].sum(),
                arrays[description['readout']['conflict_evidence_px']], rtol=1e-6, atol=1e-6)
        # Detached diagnostic arrays must not alias live inference tensors.
        arrays[metadata['pair']['q']][:] = 0
        self.assertGreater(float(self.pair.q.sum()), 0)

    def test_compact_correspondences_keep_original_storage_indices(self):
        pair = replace(self.pair, original_a=torch.tensor([2,4,6,8,10]),
                       original_b=torch.tensor([1,5,9,13,17]))
        metadata, arrays = self.snapshot(pair=pair)
        for cluster in metadata['clusters']:
            ids = arrays[cluster['correspondence_ids']]
            original = arrays[cluster['correspondence_original_ids']]
            np.testing.assert_array_equal(original[:,0], pair.original_a.numpy()[ids[:,0]])
            np.testing.assert_array_equal(original[:,1], pair.original_b.numpy()[ids[:,1]])
            proposed = {tuple(x) for x in arrays[cluster['proposal']['edge_ids']]}
            np.testing.assert_array_equal(arrays[cluster['added_to_sparse_proposal']],
                                           [tuple(x) not in proposed for x in ids])

    def test_final_evidence_is_not_capped_to512_entries(self):
        n = 40
        angle = torch.arange(n)*2*torch.pi/n
        points = torch.stack((angle.sin(), angle.cos()),1)
        geometry = compact_contour(points[None], torch.ones(1,n,dtype=torch.bool))
        q = torch.ones(n,n)*.01
        feature = torch.ones(n,4)
        pair = PairEvidence(feature,feature,feature,feature,q,1-q.sum(1),1-q.sum(0),
                            geometry,geometry,torch.arange(n),torch.arange(n))
        metadata, arrays = self.snapshot(pair=pair)
        self.assertTrue(metadata['clusters'])
        self.assertGreater(max(len(arrays[c['correspondence_ids']]) for c in metadata['clusters']),512)
        self.assertEqual(arrays[metadata['pair']['q']].shape,(40,40))

    def test_initial_capture_required_instead_of_fabricated_replay(self):
        with self.assertRaisesRegex(ValueError, 'initial pass was not captured'):
            self.snapshot(prediction=self.model.score_pair(self.pair))

    def test_pose_score_and_acceptance_mismatch_are_rejected(self):
        prediction = self.model.score_pair(self.pair, capture_diagnostics=True)
        with self.assertRaisesRegex(ValueError, 'reported pose'):
            self.snapshot(prediction=replace(prediction,
                translation_a_to_b_rc=prediction.translation_a_to_b_rc+1))
        winner = prediction.clusters[prediction.selected_cluster_id]
        changed = replace(winner.encoded, evidence=replace(winner.encoded.evidence, pose=winner.translation+1))
        clusters = list(prediction.clusters)
        clusters[prediction.selected_cluster_id] = replace(winner,encoded=changed)
        with self.assertRaisesRegex(ValueError, 'final score input'):
            self.snapshot(prediction=replace(prediction,clusters=tuple(clusters)))
        with self.assertRaisesRegex(ValueError, 'acceptance'):
            self.snapshot(prediction=replace(prediction,accepted=not prediction.accepted))

    def test_empty_pair_has_no_fake_pose_or_support(self):
        pair = fixture(torch.zeros(5,5))
        metadata, arrays = self.snapshot(pair=pair)
        self.assertFalse(metadata['has_candidate'])
        self.assertFalse(metadata['accepted'])
        self.assertEqual(metadata['selected_cluster_id'],-1)
        self.assertIsNone(metadata['translation_a_to_b_rc'])
        self.assertIsNone(metadata['canvas_b_shift_rc'])
        self.assertEqual(metadata['clusters'],[])
        self.assertEqual(arrays[metadata['proposals']['seeds']].shape,(0,2))

    def test_arc_union_wrap_duplicate_and_gap(self):
        # [-2,2] wraps; [8,12] is the same circular observation, not another4px.
        self.assertEqual(union_arc_length([[-2,2],[8,12],[4,5]],10),5)
        self.assertEqual(union_arc_length([[1,2],[8,9]],10),2)
        self.assertEqual(union_arc_length([],10),0)
        self.assertEqual(union_arc_length([[1,11]],10),10)
        with self.assertRaises(ValueError):
            union_arc_length([[4,3]],10)

    def test_nonfinite_input_recorded_without_invalid_json_numbers(self):
        q = torch.zeros(5,5)
        q[0,0] = float('nan')
        pair = replace(fixture(q), numeric_valid=False)
        metadata, arrays = self.snapshot(pair=pair)
        self.assertFalse(metadata['numeric_valid'])
        self.assertEqual(metadata['arrays']['pair/q']['nonfinite_count'],1)
        json.dumps(metadata,allow_nan=False)
        self.assertTrue(np.isnan(arrays['pair/q'][0,0]))

    def test_npz_roundtrip_digests_and_no_overwrite(self):
        metadata, arrays = self.snapshot()
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root)/'case'
            saved = write_snapshot(destination,metadata,arrays)
            self.assertEqual(saved,json.loads((destination/'evidence.json').read_text()))
            self.assertEqual(hashlib.sha256((destination/'arrays.npz').read_bytes()).hexdigest(),
                             saved['sidecar']['sha256'])
            with np.load(destination/'arrays.npz',allow_pickle=False) as restored:
                self.assertEqual(set(restored.files),set(arrays))
                for name,value in arrays.items():
                    np.testing.assert_array_equal(restored[name],value)
                    self.assertEqual(hashlib.sha256(value.tobytes()).hexdigest(),
                                     saved['arrays'][name]['sha256'])
            with self.assertRaises(FileExistsError):
                write_snapshot(destination,metadata,arrays)


if __name__ == '__main__':
    unittest.main()

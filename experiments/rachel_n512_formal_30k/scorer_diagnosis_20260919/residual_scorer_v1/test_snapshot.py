from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from ..binary_scorer_v1.head import BinaryClusterHead
from .head import ARCHITECTURE
from .test_head import fixture
from .trace import ResidualTrace, LAYER_KINDS
from .snapshot import snapshot_prediction, write_snapshot, audit_snapshot, VARIANT


class ResidualSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.model, self.pair, self.proposals, _ = fixture()
        self.provenance = dict(variant=VARIANT, architecture=ARCHITECTURE, synthetic_fixture=True)

    def capture(self, proposals=None):
        with torch.no_grad(), ResidualTrace(self.model.head) as trace:
            pred = self.model.score_pair(self.pair, threshold=.3, proposals=proposals or self.proposals)
        meta, arrays = snapshot_prediction('synthetic', self.pair, pred, threshold=.3,
                                           provenance=self.provenance, trace=trace)
        return meta, arrays, trace

    def audit(self, meta, arrays):
        # Rebind every byte hash after an intentional corruption. Failing
        # arithmetic, rather than just a checksum, must catch the change.
        meta = deepcopy(meta)
        for key, value in arrays.items():
            meta['arrays'][key] = dict(meta['arrays'][key], shape=list(value.shape), dtype=value.dtype.str,
                                       sha256=hashlib.sha256(value.tobytes()).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'new'
            write_snapshot(root, meta, arrays)
            return audit_snapshot(root / 'evidence.json')

    def test_actual_graph_and_all_evidence_replay(self):
        meta, arrays, trace = self.capture()
        result = self.audit(meta, arrays)
        self.assertEqual(result['status'], 'passed', result)
        self.assertEqual(result['clusters'], 2)
        self.assertTrue(result['residual_addition_replayed'])
        self.assertEqual(list(trace.clusters[0]), list(LAYER_KINDS))

    def test_hooks_preserve_output_parameters_rng_and_gradients(self):
        head = self.model.head
        state = {k: v.clone() for k, v in head.state_dict().items()}
        rng = torch.get_rng_state().clone()
        a = self.model.score_pair(self.pair, proposals=self.proposals)
        grad_a = torch.autograd.grad(sum(c.readout.logit for c in a.clusters), tuple(head.parameters()), retain_graph=True)
        with ResidualTrace(head) as trace:
            b = self.model.score_pair(self.pair, proposals=self.proposals)
        grad_b = torch.autograd.grad(sum(c.readout.logit for c in b.clusters), tuple(head.parameters()))
        for ca, cb in zip(a.clusters, b.clusters):
            torch.testing.assert_close(ca.readout.logit, cb.readout.logit, rtol=0, atol=0)
        for ga, gb in zip(grad_a, grad_b):
            torch.testing.assert_close(ga, gb, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, head.state_dict()[k]) for k, v in state.items()))
        self.assertFalse(trace.handles)

    def test_residual_sum_tampering_detected(self):
        meta, arrays, _ = self.capture()
        layer = next(x for x in meta['clusters'][0]['layers'] if x['kind'] == 'residual_add')
        arrays[layer['output']] = arrays[layer['output']] + .1
        result = self.audit(meta, arrays)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('layer replay differs: cluster_mlp.2' in e for e in result['errors']))

    def test_using_branch_alone_in_final_projection_is_detected(self):
        meta, arrays, _ = self.capture()
        lookup = {x['name']: x for x in meta['clusters'][0]['layers']}
        arrays[lookup['cluster_mlp.3']['input']] = arrays[lookup['cluster_mlp.2.up']['output']].copy()
        result = self.audit(meta, arrays)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('residual sum is not final projection input', result['errors'])

    def test_q_tampering_detected(self):
        meta, arrays, _ = self.capture()
        arrays[meta['clusters'][0]['inputs']['q']] *= 2
        result = self.audit(meta, arrays)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('online Q differs', result['errors'])

    def test_skip_trace_must_not_be_omitted(self):
        meta, arrays, _ = self.capture()
        meta['clusters'][0]['layers'] = [x for x in meta['clusters'][0]['layers'] if x['kind'] != 'residual_add']
        with self.assertRaises(ValueError):
            self.audit(meta, arrays)

    def test_mislabeled_old_variant_rejected(self):
        with torch.no_grad(), ResidualTrace(self.model.head) as trace:
            pred = self.model.score_pair(self.pair, proposals=self.proposals)
        with self.assertRaises(ValueError):
            snapshot_prediction('bad', self.pair, pred, threshold=.3,
                                provenance=dict(self.provenance, variant='binary_patch'), trace=trace)
        with self.assertRaises(ValueError):
            with ResidualTrace(BinaryClusterHead('patch')):
                pass

    def test_empty_candidates_export_without_fake_layers(self):
        meta, arrays, trace = self.capture(replace(self.proposals, clusters=()))
        self.assertEqual(trace.clusters, [])
        self.assertEqual(self.audit(meta, arrays)['status'], 'passed')

    def test_trace_one_context_only_and_exception_cleanup(self):
        trace = ResidualTrace(self.model.head)
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with trace:
                self.model.head(self.pair, self.proposals.clusters[0])
                raise RuntimeError('deliberate')
        self.assertFalse(trace.handles)
        self.assertTrue(all(not m._forward_hooks for m in self.model.head.modules()))
        with self.assertRaises(ValueError):
            with trace:
                pass


if __name__ == '__main__':
    unittest.main()

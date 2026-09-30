import hashlib
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from . import probe_execution_inputs as probe


@dataclass(frozen=True)
class Selection:
    mask_a: torch.Tensor
    reasons: tuple


class ProbeExecutionTests(unittest.TestCase):
    def test_fingerprint_is_original_cache_named_float32_bool_bytes(self):
        batch = SimpleNamespace(mask_a=np.array([[[0, 1], [1, 0]]], dtype=np.uint8),
            contour_valid_a=np.array([[True, False]]))
        names = ('mask_a', 'contour_valid_a')
        expected = hashlib.sha256(b'mask_a'+batch.mask_a[0].astype(np.float32).tobytes()
            +b'contour_valid_a'+batch.contour_valid_a[0].astype(np.bool_).tobytes()).hexdigest()
        self.assertEqual(probe.input_fingerprint(batch, 0, names)['sha256'], expected)
        batch.mask_a[0, 0, 0] = 1
        self.assertNotEqual(probe.input_fingerprint(batch, 0, names)['sha256'], expected)

    def test_four_cells_keep_neighbors_and_originals_unchanged(self):
        online_f = (torch.tensor([[[2.]], [[7.]]]), torch.zeros(2, 1, 1))
        cache_f = (torch.tensor([[[20.]], [[70.]]]), torch.zeros(2, 1, 1))
        def block(weights, mask):
            return dict(valid=(torch.ones(2, 1, dtype=torch.bool),) * 2,
                selection=Selection(torch.tensor([[mask], [mask]]), ('one', 'two')),
                kwargs=dict(candidate_weights=torch.tensor(weights).reshape(2, 1),
                            points_a_rc=torch.zeros(2, 1, 2), points_b_rc=torch.zeros(2, 1, 2)),
                training_valid=torch.ones(2, dtype=torch.bool), decision_valid=torch.ones(2, dtype=torch.bool))
        ob, cb = block([1., 3.], False), block([10., 30.], True)
        calls = []
        def head(fa, fb, va, vb, selection, **kwargs):
            calls.append((fa.clone(), kwargs['candidate_weights'].clone(), selection.mask_a.clone()))
            return SimpleNamespace(logit=fa[:, 0, 0]+kwargs['candidate_weights'][:, 0],
                                   used_fallback=torch.zeros(2, dtype=torch.bool))
        result = probe.four_cells({'arm': head}, online_f, cache_f, ob, cb, [0])
        self.assertEqual([result[0]['arm'][c]['logit'] for c in probe.CELLS], [30., 12., 21., 3.])
        for fa, weights, mask in calls:
            self.assertEqual(float(fa[1, 0, 0]), 7.)
            self.assertEqual(float(weights[1, 0]), 3.)
            self.assertFalse(bool(mask[1, 0]))
        self.assertEqual(online_f[0][0, 0, 0], 2.)
        self.assertFalse(bool(ob['selection'].mask_a[0, 0]))
        self.assertTrue(bool(cb['selection'].mask_a[0, 0]))

    def test_exchange_dataclass_reasons_and_set_overlap(self):
        old = Selection(torch.tensor([[False], [True]]), ('a', 'b'))
        cached = Selection(torch.tensor([[True], [False]]), ('c', 'd'))
        result = probe.exchange_rows(old, cached, [0])
        self.assertEqual(result.reasons, ('c', 'b'))
        self.assertEqual(old.reasons, ('a', 'b'))
        self.assertEqual(result.mask_a.tolist(), [[True], [True]])
        self.assertEqual(probe.jaccard({(1, 2), (2, 3)}, {(2, 3), (3, 4)}), 1/3)
        self.assertEqual(probe.jaccard(set(), set()), 1.)


if __name__ == '__main__':
    unittest.main()

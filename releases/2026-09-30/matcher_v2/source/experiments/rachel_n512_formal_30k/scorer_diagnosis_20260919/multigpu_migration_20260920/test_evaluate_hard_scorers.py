"""Small CPU checks; no checkpoint, dataset, GPU or remote access."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from . import evaluate_hard_scorers as diagnostic


class HardScorerTests(unittest.TestCase):
    def test_six_heads_share_exact_single_forward_and_selection(self):
        inputs = tuple(torch.zeros(2, 3, 2) for _ in range(4)) + (torch.ones(2, 3, dtype=torch.bool),) * 2
        original = SimpleNamespace(assignment=torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
            token_features_a=torch.ones(2, 3, 4), token_features_b=torch.full((2, 3, 4), 2.),
            training_valid=torch.tensor([True, False]))
        selected = SimpleNamespace(candidate_indices=torch.tensor([[[0, 2], [-1, -1]], [[1, 0], [-1, -1]]]),
            candidate_valid=torch.tensor([[True, False], [True, False]]))
        base, selector = Mock(return_value=original), Mock(return_value=selected)
        heads = {str(i): Mock(return_value=SimpleNamespace(logit=torch.tensor([4., -2.]),
                          used_fallback=torch.tensor([False, True]))) for i in range(6)}
        out, selection, scores = diagnostic.shared_forward(base, heads, inputs, selector)
        self.assertIs(out, original)
        self.assertIs(selection, selected)
        base.assert_called_once_with(*inputs)
        selector.assert_called_once()
        weight_ids = set()
        for key, head in heads.items():
            args, kwargs = head.call_args
            self.assertIs(args[0], original.token_features_a)
            self.assertIs(args[1], original.token_features_b)
            self.assertIs(args[4], selected)
            self.assertIs(kwargs['points_a_rc'], inputs[2])
            self.assertIs(kwargs['points_b_rc'], inputs[3])
            torch.testing.assert_close(kwargs['candidate_weights'], torch.tensor([[2., 0.], [12., 0.]]))
            weight_ids.add(id(kwargs['candidate_weights']))
            self.assertEqual(scores[key]['logit'], [4., 0.])
            self.assertEqual(scores[key]['raw_logit'], [4., -2.])
            self.assertEqual(scores[key]['probability'][1], .5)
        self.assertEqual(len(weight_ids), 1)
        with self.assertRaises(ValueError):
            diagnostic.shared_forward(base, heads, inputs + (torch.tensor([1., 0.]),), selector)

    def test_rows_keep_invalid_positive_and_negative_geometry_unknown(self):
        entries = [dict(pair_id=str(i), source_pair_id='source'+str(i), recipe='wave',
                        label=i < 2, changed_pair=True, assigned_recipe='wave',
                        fallback_reason=None, source_family_overlap=False) for i in range(3)]
        reports = [dict(pose_supervision_enabled=False) for _ in entries]
        batch = SimpleNamespace(translation_a_to_b_rc=np.zeros((3, 2)))
        original = SimpleNamespace(training_valid=torch.tensor([True, False, True]),
                                   decision_valid=torch.tensor([True, False, True]))
        selected = SimpleNamespace(layout_valid=torch.tensor([True, False, True]),
            translation_a_to_b_rc=torch.tensor([[3., 4.], [float('nan'), float('nan')], [1., 2.]]),
            candidate_valid=torch.ones(3, 2, dtype=torch.bool),
            candidate_inliers=torch.tensor([[True, False]] * 3),
            mask_a=torch.tensor([[True, False]] * 3), mask_b=torch.ones(3, 2, dtype=torch.bool),
            reasons=('ok', 'too_few', 'ok'))
        scores = {'head': dict(logit=[1., 0., 2.], raw_logit=[1., -3., 2.],
                               probability=[.7, .5, .8], used_fallback=[False, True, False])}
        expected = {str(i): dict(raw_layout_valid=i != 1, raw_translation_l2_px=6. if i == 0 else None,
                                 raw_layout20_correct=i == 0) for i in range(3)}
        rows = diagnostic.rows_from_batch(entries, reports, batch, original, selected, scores, expected, 8)
        self.assertEqual(rows[0]['raw_translation_l2_px'], 5.)
        self.assertEqual(rows[0]['expected_cpu_layout']['error_delta_px'], -1.)
        self.assertEqual(rows[0]['ordinal'], 8)
        self.assertIsNone(rows[1]['raw_translation_l2_px'])
        self.assertIsNone(rows[1]['translation_a_to_b_rc'])
        self.assertFalse(rows[1]['raw_layout20_correct'])
        self.assertIsNone(rows[2]['raw_translation_l2_px'])
        self.assertFalse(rows[2]['raw_layout20_correct'])
        sanity = diagnostic.layout_sanity(rows)
        self.assertEqual(sanity['positive_count'], 2)
        self.assertEqual(sanity['positive_layout20_agreement'], 2)
        self.assertEqual(sanity['comparable_error_count'], 1)
        self.assertEqual(sanity['maximum_absolute_error_delta_px'], 1.)

    def test_cli_requires_explicit_full_or_pilot(self):
        args = ['--manifest', 'm', '--source-val-manifest', 'v', '--expected-layout-dir', 'e',
                '--training-root', 't', '--output', 'o']
        parsed = diagnostic.parser().parse_args(args + ['--full'])
        self.assertTrue(parsed.full)
        self.assertIsNone(parsed.limit)
        self.assertEqual(parsed.batch_size, 8)
        parsed = diagnostic.parser().parse_args(args + ['--limit', '32'])
        self.assertFalse(parsed.full)
        self.assertEqual(parsed.limit, 32)


if __name__ == '__main__':
    unittest.main()

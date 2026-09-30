from dataclasses import replace
import copy
import importlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from .matcher_diagnostics import distribution, inspect_pair, summarize

BASE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'


class MatcherDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['CURRICULUM_BASELINE_SOURCE']).resolve()
        cls.fixture_module = importlib.import_module(BASE + 's7_consensus_v1.test_threshold_joint')
        cls.head_module = importlib.import_module(BASE + 'binary_scorer_v1.head')
        if cls.root not in Path(cls.head_module.__file__).resolve().parents:
            raise ValueError('use bound threshold/light source for actual union features')

    def setUp(self):
        _, self.pair, self.proposals, _ = self.fixture_module.setup_pair()
        self.target = self.proposals.clusters[0].translation

    def inspect(self, name='fixture', **kwargs):
        return inspect_pair(name, self.pair, self.proposals, label=True, gt_pose=self.target, **kwargs)

    def test_all_clusters_inspected_without_neural_scorer(self):
        with patch.object(self.head_module.BinaryClusterHead, 'forward', side_effect=AssertionError('Scorer used')):
            row = self.inspect()
        self.assertEqual(row['retained_count'], 2)
        self.assertFalse(row['scorer_used']); self.assertFalse(row['gt_used_in_proposal'])
        self.assertTrue(row['retained_correct_coverage'])

    def test_duplicate_edges_do_not_inflate_counts_or_q(self):
        a = self.inspect()
        changed = tuple(replace(c, edge_ids=c.edge_ids.repeat(3, 1)) for c in self.proposals.clusters)
        b = inspect_pair('fixture', self.pair, replace(self.proposals, clusters=changed),
                         label=True, gt_pose=self.target)
        self.assertEqual(a, b)

    def test_raw_q_arc_and_edge_counts_agree_with_actual_head_inputs(self):
        row = self.inspect(capture_edges=True)
        for item, cluster in zip(row['candidates'], self.proposals.clusters):
            values = self.head_module.inputs_for_cluster(self.pair, cluster, False)
            self.assertEqual(item['unique_edge_count'], len(values.edge_ids))
            self.assertAlmostEqual(item['q_sum'], float(values.q.double().sum()))
            self.assertAlmostEqual(item['q_arc_mass_px'], float(values.mass_weights.double().sum()))
            self.assertEqual(item['edges']['raw_q'], values.q.tolist())
            self.assertEqual(item['edges']['compact_indices'], values.edge_ids.tolist())

    def test_raw_q_and_q_arc_winners_can_be_different_and_both_reported(self):
        n = len(self.pair.q)
        q = torch.diag(torch.tensor([.1] * 16 + [.2] * (n - 16)))
        steps = torch.tensor([[.1] * 16 + [10.] * (n - 16)])
        pair = replace(self.pair, q=q, ga=replace(self.pair.ga, next_step_px=steps),
                       gb=replace(self.pair.gb, next_step_px=steps))
        row = inspect_pair('different-winners', pair, self.proposals, label=True, gt_pose=self.target)
        self.assertEqual(row['q_sum_winner'], 0)
        self.assertEqual(row['q_arc_winner'], 1)
        self.assertTrue(row['q_sum_winner_layout20'])
        self.assertFalse(row['q_arc_winner_layout20'])

    def test_gt_only_annotates_fixed_proposals(self):
        a = self.inspect(capture_edges=True)
        b = inspect_pair('fixture', self.pair, self.proposals, label=True,
                         gt_pose=self.target + torch.tensor([100., 0.]), capture_edges=True)
        self.assertNotEqual(a['retained_correct_coverage'], b['retained_correct_coverage'])
        for first, second in zip(a['candidates'], b['candidates']):
            for field in ('candidate_sha256', 'edges', 'translation_rc', 'q_sum', 'q_arc_mass_px'):
                self.assertEqual(first[field], second[field])

    def test_budget_loss_only_when_prebudget_candidates_are_available(self):
        unknown = self.inspect()
        self.assertIsNone(unknown['budget_lost_correct'])
        first, second = self.proposals.clusters
        retained = replace(self.proposals, clusters=(second,))
        row = inspect_pair('lost', self.pair, retained, label=True, gt_pose=self.target,
                           all_clusters=(second, first))
        self.assertFalse(row['retained_correct_coverage'])
        self.assertTrue(row['prebudget_correct_coverage']); self.assertTrue(row['budget_lost_correct'])

    def test_other_pair_prebudget_list_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'prefix'):
            self.inspect(all_clusters=tuple(reversed(self.proposals.clusters)))

    def test_invalid_pair_cannot_reuse_previous_builder_list(self):
        with self.assertRaisesRegex(ValueError, 'invalid pair'):
            inspect_pair('bad', replace(self.pair, numeric_valid=False), self.proposals, label=False)

    def test_empty_retained_cannot_hide_stale_full_list(self):
        with self.assertRaisesRegex(ValueError, 'stale'):
            inspect_pair('bad', self.pair, replace(self.proposals, clusters=()), label=False,
                         all_clusters=self.proposals.clusters)

    def test_turufan_unknown_gt_is_null_not_zero(self):
        row = inspect_pair('turu', self.pair, self.proposals, label=True)
        summary = summarize([row])
        for key in ('q_sum_winner_layout20', 'retained_correct_coverage', 'budget_lost_correct'):
            self.assertIsNone(row[key])
        for key in ('q_sum_layout20', 'q_arc_layout20', 'correct_coverage_count', 'joint_f1'):
            self.assertIsNone(summary[key])
        self.assertEqual(summary['positive_layout_gt_count'], 0)

    def test_negative_has_mass_but_no_fake_common_layout(self):
        row = inspect_pair('negative', self.pair, self.proposals, label=False)
        self.assertGreater(row['negative_max_q_sum'], 0)
        self.assertIsNone(row['q_sum_winner_layout20'])
        with self.assertRaisesRegex(ValueError, 'negative'):
            inspect_pair('negative', self.pair, self.proposals, label=False, gt_pose=self.target)

    def test_no_candidate_pairs_remain_in_class_and_layout_denominators(self):
        positive = inspect_pair('positive-empty', self.pair, replace(self.proposals, clusters=()),
                                label=True, gt_pose=self.target, all_clusters=())
        negative = inspect_pair('negative-empty', self.pair, replace(self.proposals, clusters=()), label=False)
        summary = summarize([positive, negative])
        self.assertEqual(summary['pairs'], 2); self.assertEqual(summary['positive_no_candidate'], 1)
        self.assertEqual(summary['negative_no_candidate'], 1)
        self.assertEqual(summary['q_sum_layout20'], 0.)
        self.assertEqual(summary['negative_max_q_sum']['count'], 0)
        self.assertFalse(summary['no_candidate_negative_mass_imputed'])

    def test_observed_arc_normalization_is_explicit_not_gt_seam_accuracy(self):
        row = self.inspect(seam_reference={'length_px': 100., 'definition': 'synthetic original uncut GT arc'})
        item = row['candidates'][0]
        self.assertAlmostEqual(item['represented_arc_over_reference'], item['represented_bilateral_arc_px']/100.)
        self.assertFalse(row['seam_normalization_is_gt_correspondence_accuracy'])
        self.assertIsNone(self.inspect()['candidates'][0]['represented_arc_over_reference'])

    def test_missing_normalization_reference_keeps_correct_distribution_denominator(self):
        one = self.inspect('with-reference', seam_reference={'length_px': 100., 'definition': 'synthetic'})
        two = self.inspect('without-reference')
        result = summarize([one, two])
        self.assertEqual(result['correct_cluster_population'], 2)
        self.assertEqual(result['correct_cluster_pair_distributions']['represented_arc_over_reference']['count'], 1)
        self.assertEqual(result['correct_cluster_pair_distributions']['unique_edge_count']['count'], 2)

    def test_nonfinite_or_unidentified_seam_reference_rejected(self):
        for reference in ({'length_px': 0., 'definition': 'bad'}, {'length_px': 10.},
                          {'length_px': float('nan'), 'definition': 'bad'}):
            with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, 'length'):
                self.inspect(seam_reference=reference)

    def test_patch_feature_values_are_not_used_for_native_evidence(self):
        expected = self.inspect()
        changed = replace(self.pair, local_a=torch.full_like(self.pair.local_a, float('nan')),
                          context_b=torch.full_like(self.pair.context_b, float('nan')))
        actual = inspect_pair('fixture', changed, self.proposals, label=True, gt_pose=self.target)
        self.assertEqual(expected, actual)

    def test_inputs_parameters_gradients_and_rng_not_modified(self):
        before_q = self.pair.q.clone(); before_rng = torch.get_rng_state().clone()
        before_proposals = copy.deepcopy(self.proposals)
        self.inspect(capture_edges=True)
        self.assertTrue(torch.equal(before_q, self.pair.q))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertIsNone(self.pair.local_a.grad)
        self.assertTrue(torch.is_grad_enabled())
        for a, b in zip(before_proposals.clusters, self.proposals.clusters):
            self.assertTrue(torch.equal(a.translation, b.translation))
            self.assertTrue(torch.equal(a.edge_ids, b.edge_ids))

    def test_original_edge_indices_retained_for_image_overlays(self):
        changed = replace(self.pair, original_a=self.pair.original_a * 2,
                          original_b=self.pair.original_b * 3)
        row = inspect_pair('original-indices', changed, self.proposals, label=True,
                           gt_pose=self.target, capture_edges=True)
        for candidate in row['candidates']:
            for compact, original in zip(candidate['edges']['compact_indices'], candidate['edges']['original_indices']):
                self.assertEqual(original, [compact[0] * 2, compact[1] * 3])

    def test_distributions_include_tails_not_only_median(self):
        result = distribution([1, 2, 3, 4, 100])
        self.assertEqual(result['quantiles']['p50'], 3.)
        self.assertEqual(result['quantiles']['max'], 100.)
        self.assertGreater(result['quantiles']['p95'], result['quantiles']['p75'])
        self.assertAlmostEqual(result['mean'], 22.)
        self.assertIsNone(distribution([])['mean'])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            distribution([1, float('nan')])

    def test_duplicate_pairs_not_counted_as_independent_observations(self):
        row = self.inspect()
        with self.assertRaisesRegex(ValueError, 'unique'):
            summarize([row, row])

    def test_incorrect_strong_candidate_not_labelled_best_correct(self):
        good = self.inspect(); c = good['candidates'][good['best_correct_by_q_arc']]
        self.assertTrue(c['layout20'])
        no_correct = inspect_pair('no-correct', self.pair, self.proposals, label=True,
                                  gt_pose=torch.tensor([1000., 1000.]))
        self.assertIsNone(no_correct['best_correct_by_q_arc'])
        result = summarize([good, no_correct])
        self.assertEqual(result['correct_cluster_population'], 1)
        self.assertEqual(result['positive_layout_gt_count'], 2)


if __name__ == '__main__':
    unittest.main()

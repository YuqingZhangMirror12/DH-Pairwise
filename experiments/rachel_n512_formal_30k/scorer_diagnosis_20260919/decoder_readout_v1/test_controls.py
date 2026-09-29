import unittest
from dataclasses import replace
import torch
from decoder import SearchPolicy, CONTROLS, copy_with_search
from head import EvidenceClusterHead, raw_q_baselines, sum_evidence, ablate_frozen_binary_overlap
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_threshold_joint import setup_pair
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.legacy_pose_consensus import _cloud
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.binary_scorer_v1.head import BinaryClusterHead, SCALAR_NAMES

torch.set_num_threads(1)


class Controls(unittest.TestCase):
    def setUp(self):
        self.model, self.pair, self.proposals, self.features = setup_pair()

    def test_baseline_seed_bit_identity(self):
        b = self.model.builder; cloud = _cloud(self.pair, b.config)
        other = copy_with_search(b, SearchPolicy())
        self.assertTrue(torch.equal(b._seeds(cloud), other._seeds(cloud)))
        self.assertIsNot(b, other)
        self.assertFalse(hasattr(b, 'search_policy'))

    def test_top3_does_not_mutate_original_or_geometry(self):
        b = self.model.builder; c = copy_with_search(b, CONTROLS['top3_only'])
        self.assertEqual(b.config.row_column_topk, 2)
        self.assertEqual(c.config.row_column_topk, 3)
        self.assertIs(c.geometry, b.geometry); self.assertIs(c.policy, b.policy)
        old = set(map(tuple, _cloud(self.pair, b.config).ids.tolist()))
        new = set(map(tuple, _cloud(self.pair, c.config).ids.tolist()))
        self.assertTrue(old <= new)

    def test_seed_budget_no_longer_changes_mode_budget(self):
        b = self.model.builder; cloud = _cloud(self.pair, b.config)
        a = copy_with_search(b, SearchPolicy()); c = copy_with_search(b, CONTROLS['seeds32_only'])
        x, y = a._seeds(cloud), c._seeds(cloud)
        self.assertTrue(torch.equal(x, y[:len(x)]))
        self.assertEqual(c.search_audit['policy']['mode_limit'], 128)

    def test_all_modes_are_eligible_and_empty_safe(self):
        b = copy_with_search(self.model.builder, CONTROLS['all_modes_only'])
        cloud = _cloud(self.pair, b.config); b._seeds(cloud)
        self.assertEqual(b.search_audit['bins'], b.search_audit['eligible_modes'])
        empty = replace(cloud, ids=cloud.ids[:0], displacement=cloud.displacement[:0])
        self.assertEqual(tuple(b._seeds(empty).shape), (0, 2))

    def test_invalid_budgets_fail_closed(self):
        for args in [(0, 128, 16), (2, 8, 16), (2, None, 0)]:
            with self.assertRaises(ValueError): SearchPolicy(*args)

    def test_meanmax_baseline_can_exactly_load_old_weights(self):
        old = BinaryClusterHead('patch', 4)
        new = EvidenceClusterHead('patch_meanmax', 4, remove_overlap=False)
        new.load_state_dict(old.state_dict(), strict=True)
        p = self.proposals.clusters[0]
        torch.testing.assert_close(old(self.pair, p).logit, new(self.pair, p).logit, rtol=0, atol=0)

    def test_union_duplicates_have_no_effect_all_heads(self):
        p = self.proposals.clusters[0]
        for variant in ('patch_sum', 'patch_mean', 'patch_meanmax', 'stats'):
            h = EvidenceClusterHead(variant, 4)
            a = h(self.pair, p); b = h(self.pair, replace(p, edge_ids=p.edge_ids.repeat(3, 1)))
            torch.testing.assert_close(a.logit, b.logit)
            self.assertEqual(len(a.inputs.q), len(p.edge_ids))
        self.assertEqual(raw_q_baselines(self.pair, p)['unique_count'], len(p.edge_ids))

    def test_sum_is_permutation_invariant_and_no_mass_normalization(self):
        w = torch.tensor([.3, .7, .2]); g = torch.rand(3, 32)
        z, parts = sum_evidence(w, g)
        order = torch.tensor([2, 0, 1])
        zz, _ = sum_evidence(w[order], g[order])
        torch.testing.assert_close(z, zz)
        torch.testing.assert_close(z.expm1(), parts.sum(0))
        small, _ = sum_evidence(w * .1, g)
        self.assertTrue(bool((small < z).all()))

    def test_sum_each_edge_contribution_is_bounded_by_q_arc(self):
        h = EvidenceClusterHead('patch_sum', 4); out = h(self.pair, self.proposals.clusters[0])
        self.assertTrue(bool((out.contributions >= 0).all()))
        self.assertTrue(bool((out.contributions <= out.inputs.mass_weights[:, None]).all()))
        out.logit.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters()))
        for f in self.features: self.assertGreater(float(f.grad.abs().sum()), 0.)

    def test_no_overlap_feature_has_no_gradient_path(self):
        p = self.proposals.clusters[0]
        for variant in ('patch_sum', 'patch_mean', 'patch_meanmax', 'stats'):
            h = EvidenceClusterHead(variant, 4)
            base = h.input_builder(self.pair, p, variant != 'stats')
            stats = base.statistics.detach().clone().requires_grad_(True)
            h.input_builder = lambda *_: replace(base, statistics=stats)
            value = h(self.pair, p).logit
            gradient = torch.autograd.grad(value, stats)[0]
            self.assertEqual(float(gradient[SCALAR_NAMES.index('overlap_min_area')]), 0.)
            self.assertNotIn('overlap_min_area', h.scalar_names)

    def test_stats_does_not_read_patch_context(self):
        h = EvidenceClusterHead('stats', 4); p = self.proposals.clusters[0]
        a = h(self.pair, p)
        pair = replace(self.pair, local_a=None, local_b=None, context_a=None, context_b=None)
        torch.testing.assert_close(a.logit, h(pair, p).logit)

    def test_overlap_intervention_does_not_mutate_input_or_weights(self):
        h = BinaryClusterHead('patch', 4)
        x = torch.rand(80); original = x.clone(); state = {k:v.clone() for k,v in h.state_dict().items()}
        actual = ablate_frozen_binary_overlap(h, x, SCALAR_NAMES)
        altered = x.clone(); altered[77] = 0.
        torch.testing.assert_close(actual, h.cluster_mlp(altered).squeeze())
        self.assertTrue(torch.equal(x, original))
        self.assertTrue(all(torch.equal(v, h.state_dict()[k]) for k,v in state.items()))


if __name__ == '__main__': unittest.main()

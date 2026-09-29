from dataclasses import replace
import unittest

import torch
from torch import nn

from ..binary_scorer_v1.head import BinaryClusterHead
from ..binary_scorer_v1.model import BinaryConsensus
from ..binary_scorer_v1.loss import pair_loss
from ..s7_consensus_v1.test_threshold_joint import setup_pair
from ..s7_consensus_v1.targets import PairLabels
from .head import ResidualClusterHead, ResidualBlock, architecture_record


def fixture():
    original, pair, proposals, features = setup_pair()
    torch.manual_seed(26092406)
    model = BinaryConsensus(None, original.geometry, head=ResidualClusterHead(4))
    return model, pair, proposals, features


class ResidualHeadTests(unittest.TestCase):
    def test_parameter_count_and_explicit_identity(self):
        baseline, head = BinaryClusterHead('patch'), ResidualClusterHead()
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), 34529)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 36673)
        self.assertEqual(architecture_record(head)['parameter_count'], 36673)
        self.assertEqual(head.experiment_module, 'scorer_patch_residual')
        self.assertEqual(head.feature_dim, 96)

    def test_only_cluster_graph_changed(self):
        torch.manual_seed(81); baseline = BinaryClusterHead('patch')
        torch.manual_seed(81); head = ResidualClusterHead()
        for key, value in baseline.edge_mlp.state_dict().items():
            self.assertTrue(torch.equal(value, head.edge_mlp.state_dict()[key]), key)
        for a, b in ((baseline.cluster_mlp[0], head.cluster_mlp[0]),
                     (baseline.cluster_mlp[2], head.cluster_mlp[2].down)):
            for key, value in a.state_dict().items():
                self.assertTrue(torch.equal(value, b.state_dict()[key]), key)
        self.assertIs(ResidualClusterHead.forward, BinaryClusterHead.forward)

    def test_residual_is_exact_addition_without_post_activation(self):
        block = ResidualBlock()
        h = torch.randn(7, 64)
        torch.testing.assert_close(block(h), h + block.up(block.activation(block.down(h))), rtol=0, atol=0)
        with torch.no_grad():
            block.up.weight.zero_(); block.up.bias.zero_()
        torch.testing.assert_close(block(h), h, rtol=0, atol=0)

    def test_identity_skip_gradient_when_branch_zero(self):
        block = ResidualBlock()
        with torch.no_grad():
            block.up.weight.zero_(); block.up.bias.zero_()
        h = torch.randn(64, requires_grad=True)
        block(h).sum().backward()
        torch.testing.assert_close(h.grad, torch.ones_like(h), rtol=0, atol=0)

    def test_no_attention_batchnorm_local_class_or_neural_refinement(self):
        head = ResidualClusterHead()
        self.assertFalse(any(isinstance(m, (nn.MultiheadAttention, nn.modules.batchnorm._BatchNorm)) for m in head.modules()))
        self.assertFalse(any('conflict' in k or 'localizer' in k for k in head.state_dict()))
        self.assertEqual(head.cluster_mlp[-1].out_features, 1)

    def test_reject_old_head_state_in_both_directions(self):
        baseline, head = BinaryClusterHead('patch'), ResidualClusterHead()
        with self.assertRaises(RuntimeError):
            head.load_state_dict(baseline.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            baseline.load_state_dict(head.state_dict(), strict=True)

    def test_exact_union_q_statistics_and_pose_unchanged(self):
        model, pair, proposals, _ = fixture()
        baseline = BinaryClusterHead('patch', 4)
        for proposal in proposals.clusters:
            a, b = model.head(pair, proposal), baseline(pair, proposal)
            for name in ('edge_ids', 'q', 'arc_px', 'mass_weights', 'normalized_weights',
                         'edge_geometry', 'statistics', 'patch_context', 'pose'):
                torch.testing.assert_close(getattr(a.inputs, name), getattr(b.inputs, name), rtol=0, atol=0)
            torch.testing.assert_close(a.inputs.pose, proposal.translation, rtol=0, atol=0)

    def test_repeated_edge_union_does_not_multiply_q_or_score(self):
        model, pair, proposals, _ = fixture()
        proposal = proposals.clusters[0]
        a = model.head(pair, proposal)
        b = model.head(pair, replace(proposal, edge_ids=proposal.edge_ids.repeat(3, 1)))
        torch.testing.assert_close(a.logit, b.logit, rtol=0, atol=0)
        torch.testing.assert_close(a.inputs.q, b.inputs.q, rtol=0, atol=0)

    def test_raw_q_scale_survives_conditional_mean(self):
        model, pair, proposals, _ = fixture(); proposal = proposals.clusters[0]
        a = model.head(pair, proposal).inputs
        b = model.head(replace(pair, q=pair.q * .1), proposal).inputs
        torch.testing.assert_close(a.normalized_weights, b.normalized_weights)
        self.assertGreater(float(a.statistics[1]), float(b.statistics[1]))
        self.assertGreater(float(a.statistics[2]), float(b.statistics[2]))

    def test_all_member_features_receive_gradient(self):
        model, pair, proposals, features = fixture()
        grad = torch.autograd.grad(model.head(pair, proposals.clusters[0]).logit, features)
        for start in (0, 4, 8, 12):
            self.assertGreater(sum(float(g[start:start + 4].abs().sum()) for g in grad), 0.)

    def test_loss_unchanged_wrong_pose_is_not_positive(self):
        model, pair, proposals, _ = fixture(); n = len(pair.q)
        labels = PairLabels(True, True, torch.tensor([7., 0.]), torch.arange(n), torch.arange(n),
                            torch.ones(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool))
        prediction = model.score_pair(pair, proposals=proposals)
        loss = pair_loss(model, prediction, labels)
        self.assertEqual(loss.counts['correct_clusters'], 1)
        self.assertEqual(loss.counts['wrong_clusters'], 1)
        self.assertEqual(set(loss.components), {'candidate', 'ranking'})
        loss.total.backward()
        for name, parameter in model.head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)

    def test_empty_candidate_has_no_invented_pose(self):
        model, pair, proposals, _ = fixture()
        prediction = model.score_pair(pair, proposals=replace(proposals, clusters=()))
        self.assertFalse(prediction.has_candidate)
        self.assertIsNone(prediction.translation_a_to_b_rc)
        self.assertEqual(float(prediction.score), 0.)

    def test_fragment_exchange_invariant(self):
        model, pair, proposals, _ = fixture(); proposal = proposals.clusters[0]
        swapped = replace(pair, local_a=pair.local_b, local_b=pair.local_a, context_a=pair.context_b,
            context_b=pair.context_a, q=pair.q.T, unmatched_a=pair.unmatched_b, unmatched_b=pair.unmatched_a,
            ga=pair.gb, gb=pair.ga, original_a=pair.original_b, original_b=pair.original_a)
        other = replace(proposal, translation=-proposal.translation, edge_ids=proposal.edge_ids.flip(1))
        torch.testing.assert_close(model.head(pair, proposal).logit, model.head(swapped, other).logit)

    def test_invalid_and_zero_q_rejected(self):
        model, pair, proposals, _ = fixture(); proposal = proposals.clusters[0]
        for value in (float('nan'), -1., 0.):
            q = pair.q.clone()
            q[proposal.edge_ids[:, 0], proposal.edge_ids[:, 1]] = value
            with self.assertRaises(ValueError):
                model.head(replace(pair, q=q), proposal)

    def test_feature_dimension_required(self):
        for value in (0, -2, True, 4.5):
            with self.assertRaises(ValueError):
                ResidualClusterHead(value)


if __name__ == '__main__':
    unittest.main()

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, RachelN512Pairwise
from ..s7_consensus_v1 import matcher as legacy
from ..s7_consensus_v1 import test_matcher as legacy_tests
from ..s7_consensus_v1.scratch_matcher import fresh_matcher
from .adapter import MatcherV2Adapter, fresh_matcher_v2
from .network import MatcherV2Config, ContourSelfBlock, CrossBlock, arc_positions, rotary, deterministic_prefix_sum
from .spec import model_spec, from_model_spec


class V2AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(26093031)
        self.arch = RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
            window_sizes_px=(7., 16., 32., 64.), landmark_count=2,
            activation_checkpointing=False)
        self.base = RachelN512Pairwise(self.arch).eval()
        self.cfg = MatcherV2Config(enabled=True)

    def model(self, config=None, frozen=False):
        return MatcherV2Adapter(copy.deepcopy(self.base), config=config or self.cfg, frozen=frozen)

    def test_default_disabled_preserves_state_keys_rng_and_forward(self):
        old = legacy.S7MatcherAdapter(copy.deepcopy(self.base))
        rng = torch.get_rng_state().clone()
        new = MatcherV2Adapter(copy.deepcopy(self.base))
        self.assertIsNone(new.upgrades)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        new.load_state_dict(old.state_dict(), strict=True)
        a, b = old(*legacy_tests.inputs()), new(*legacy_tests.inputs())
        torch.testing.assert_close(a.assignment, b.assignment, atol=0, rtol=0)

    def test_enabled_zero_step_exact_old_evidence(self):
        old = legacy.S7MatcherAdapter(copy.deepcopy(self.base))
        rng = torch.get_rng_state().clone()
        new = self.model(frozen=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        a, b = old(*legacy_tests.inputs()), new(*legacy_tests.inputs())
        for field in ('local_a', 'local_b', 'context_a', 'context_b', 'affinity',
                      'assignment', 'unmatched_a', 'unmatched_b', 'numeric_valid'):
            torch.testing.assert_close(getattr(a, field), getattr(b, field), atol=0, rtol=0, msg=field)

    def test_same_seed_legacy_initialization_and_no_outer_rng_consumption(self):
        state = torch.get_rng_state().clone()
        old = fresh_matcher(self.arch, seed=26092407)
        new = fresh_matcher_v2(self.arch, config=self.cfg, seed=26092407)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for key, value in old.base.state_dict().items():
            torch.testing.assert_close(value, new.base.state_dict()[key], atol=0, rtol=0)

    def test_single_sinkhorn_two_encodings_no_legacy_head_or_decoder(self):
        new = self.model(frozen=True)
        with patch.object(legacy, 'dustbin_sinkhorn', wraps=legacy.dustbin_sinkhorn) as sink, \
             patch.object(new.base.patch_encoder, 'forward', wraps=new.base.patch_encoder.forward) as enc, \
             patch.object(new.base.coarse, 'forward', side_effect=AssertionError('coarse')), \
             patch.object(new.base.local_head, 'forward', side_effect=AssertionError('legacy head')), \
             patch.object(new.base.fusion, 'forward', side_effect=AssertionError('fusion')), \
             patch.object(new.base, '_translation', side_effect=AssertionError('decoder')):
            new(*legacy_tests.inputs())
        self.assertEqual(sink.call_count, 1)
        self.assertEqual(enc.call_count, 2)

    def test_freeze_and_unfreeze_include_all_new_modules(self):
        new = self.model(frozen=True).train()
        self.assertFalse(new.base.training)
        self.assertFalse(new.upgrades.training)
        self.assertFalse(any(p.requires_grad for p in new.parameters()))
        self.assertFalse(new(*legacy_tests.inputs()).assignment.requires_grad)
        new.set_frozen(False)
        self.assertTrue(new.upgrades.training)
        self.assertTrue(all(p.requires_grad for p in new.upgrades.parameters()))
        self.assertFalse(any(p.requires_grad for p in new.base.coarse.parameters()))

    def test_gradients_first_and_second_update(self):
        new = self.model().train()
        opt = torch.optim.AdamW((p for p in new.parameters() if p.requires_grad), lr=1e-3)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            result = new(*legacy_tests.inputs())
            loss = -result.assignment[:, 0, 1].clamp_min(1e-20).log().mean()
            loss.backward()
            for name, value in new.upgrades.named_parameters():
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(bool(torch.isfinite(value.grad).all()), name)
            if step == 0:
                self.assertEqual(float(new.upgrades.self_blocks[0].qkv.weight.grad.abs().sum()), 0)
                self.assertGreater(float(new.upgrades.self_blocks[0].out.weight.grad.abs().sum()), 0)
            else:
                for value in (new.upgrades.self_blocks[0].qkv.weight,
                              new.upgrades.cross_blocks[0].q.weight,
                              new.upgrades.concat.weight,
                              new.upgrades.scale_primal[0].weight,
                              new.upgrades.log_sharpness):
                    self.assertGreater(float(value.grad.abs().sum()), 0)
            opt.step()

    def test_checkpoints_forward_and_backward_agree(self):
        normal = self.model().train()
        # Exercise nonzero residual paths, not only the zero-step identity.
        with torch.no_grad():
            for name, value in normal.upgrades.named_parameters():
                if 'out.' in name or 'ff.2.' in name or 'scale_weight' in name:
                    value.normal_(0, .015)
        checked = copy.deepcopy(normal)
        checked.base.config = replace(self.arch, activation_checkpointing=True)
        outputs = []
        for model in (normal, checked):
            result = model(*legacy_tests.inputs())
            (-result.assignment[:, 0, 1].log().mean()).backward()
            outputs.append(result)
        torch.testing.assert_close(outputs[0].assignment, outputs[1].assignment, atol=0, rtol=0)
        for (name, a), (_, b) in zip(normal.named_parameters(), checked.named_parameters()):
            if a.requires_grad:
                self.assertIsNotNone(a.grad, name)
                torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-6, msg=name)

    def test_swap_uses_simultaneous_cross_updates(self):
        new = self.model(frozen=True)
        with torch.no_grad():
            for name, value in new.upgrades.named_parameters():
                if 'out.' in name or 'ff.2.' in name or 'scale_weight' in name:
                    value.normal_(0, .025)
        args = legacy_tests.inputs()
        a = new(*args)
        b = new(args[1], args[0], args[3], args[2], args[5], args[4])
        torch.testing.assert_close(a.context_a, b.context_b, atol=0, rtol=0)
        torch.testing.assert_close(a.context_b, b.context_a, atol=0, rtol=0)
        # CPU GEMM transpose changes reduction order (~3e-8 observed);
        # contexts themselves must still be bitwise symmetric above.
        torch.testing.assert_close(a.affinity, b.affinity.transpose(1, 2), atol=5e-8, rtol=3e-7)

    def test_nan_padding_not_content(self):
        new = self.model(frozen=True)
        args = list(legacy_tests.inputs())
        expected = new(*args)
        args[2][~args[4]] = float('nan')
        args[3][~args[5]] = float('inf')
        actual = new(*args)
        torch.testing.assert_close(expected.assignment, actual.assignment, atol=0, rtol=0)

    def test_matchability_optional_neutral_zero_step(self):
        old = legacy.S7MatcherAdapter(copy.deepcopy(self.base))(*legacy_tests.inputs())
        new = self.model(replace(self.cfg, use_matchability=True), frozen=True)(*legacy_tests.inputs())
        torch.testing.assert_close(old.affinity, new.affinity, atol=0, rtol=0)
        torch.testing.assert_close(old.assignment, new.assignment, atol=0, rtol=0)

    def test_transport_fp32_and_endpoint_mass(self):
        result = self.model(frozen=True)(*legacy_tests.inputs())
        self.assertEqual(result.assignment.dtype, torch.float32)
        torch.testing.assert_close(result.assignment.sum(2) + result.unmatched_a,
                                   result.valid_a.float(), atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(result.assignment.sum(1) + result.unmatched_b,
                                   result.valid_b.float(), atol=2e-4, rtol=2e-4)

    def test_gain_cap_and_all_ablation_paths_are_used(self):
        for flag in ('use_multiscale', 'use_sharpness', 'use_cross'):
            model = self.model(replace(self.cfg, **{flag: False})).train()
            (-model(*legacy_tests.inputs()).assignment[:, 0, 1].log().sum()).backward()
            self.assertTrue(all(p.grad is not None for p in model.upgrades.parameters()))
        model = self.model(frozen=True)
        with torch.no_grad():
            model.upgrades.log_sharpness.fill_(20)
        base = legacy.S7MatcherAdapter(copy.deepcopy(self.base))(*legacy_tests.inputs())
        actual = model(*legacy_tests.inputs())
        torch.testing.assert_close(actual.affinity, base.affinity * 5, atol=1e-6, rtol=1e-6)

    def test_explicit_model_spec_strict_roundtrip(self):
        model = self.model(frozen=True)
        spec = model_spec(model, 26092407)
        restored = from_model_spec(spec, frozen=True, state=model.state_dict())
        torch.testing.assert_close(model(*legacy_tests.inputs()).assignment,
                                   restored(*legacy_tests.inputs()).assignment, atol=0, rtol=0)
        damaged = copy.deepcopy(spec)
        damaged['matcher_v2']['use_cross'] = False
        with self.assertRaises(RuntimeError):
            from_model_spec(damaged, frozen=True, state=model.state_dict())
        with self.assertRaises(ValueError):
            from_model_spec({'architecture': spec['architecture']}, frozen=True)


class ValidContourTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(73001)
        self.cfg = MatcherV2Config(enabled=True)

    def test_arc_closes_at_valid_count_with_prefix_and_holes(self):
        p = torch.tensor([[[0., 0.], [0., 3.], [4., 3.], [4., 0.], [float('nan'), 0.]]])
        valid = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
        arc, perimeter = arc_positions(p, valid)
        torch.testing.assert_close(perimeter, torch.tensor([14.]), atol=0, rtol=0)
        torch.testing.assert_close(arc, torch.tensor([[0., 3., 7., 10., 0.]]), atol=0, rtol=0)
        order = torch.tensor([0, 4, 1, 2, 3])
        other_arc, other_per = arc_positions(p[:, order], valid[:, order])
        torch.testing.assert_close(other_arc, arc[:, order], atol=0, rtol=0)
        torch.testing.assert_close(other_per, perimeter, atol=0, rtol=0)

    def test_deterministic_scan_matches_reference_without_cumsum_call(self):
        for n in (1, 3, 37, 512):
            values = torch.rand(8, n, dtype=torch.float32) * 8
            expected = values.double().cumsum(1).float()
            with patch.object(torch, 'cumsum', side_effect=AssertionError('CUDA-incompatible cumulative kernel')):
                actual = deterministic_prefix_sum(values)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_prefix_scan_gradient_and_repeatability(self):
        x = torch.randn(2, 37, dtype=torch.float64, requires_grad=True)
        a = deterministic_prefix_sum(x); b = deterministic_prefix_sum(x)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        a.sum().backward()
        torch.testing.assert_close(x.grad, torch.arange(37,0,-1,dtype=x.dtype).expand_as(x), atol=0, rtol=0)

    def test_integer_harmonics_periodic_rotary(self):
        block = ContourSelfBlock(96, self.cfg).double()
        h = block.harmonics
        self.assertEqual(len(h), 12)
        self.assertEqual(len(set(h.tolist())), 12)
        self.assertTrue(bool((h == h.round()).all()))
        self.assertEqual(float(h.max()), 256)
        x = torch.randn(1, 4, 7, 24, dtype=torch.float64)
        angle = torch.randn(1, 1, 7, 1, dtype=torch.float64) * h
        torch.testing.assert_close(rotary(x, angle), rotary(x, angle + 2 * torch.pi * h), atol=1e-12, rtol=1e-12)

    def test_nonzero_self_block_cyclic_valid_rotation_and_padding(self):
        block = ContourSelfBlock(96, self.cfg).double()
        with torch.no_grad():
            block.out.weight.normal_(0, .02)
            block.ff[-1].weight.normal_(0, .02)
        p = torch.randn(1, 37, 2, dtype=torch.float64).cumsum(1)
        x = torch.randn(1, 37, 96, dtype=torch.float64)
        v = torch.ones(1, 37, dtype=torch.bool)
        expected = block(x, v, *arc_positions(p, v))
        shift = 13
        rp, rx = p.roll(shift, 1), x.roll(shift, 1)
        actual = block(rx, v, *arc_positions(rp, v))
        torch.testing.assert_close(actual, expected.roll(shift, 1), atol=2e-12, rtol=2e-12)
        vp = torch.cat((v, torch.zeros(1, 9, dtype=torch.bool)), 1)
        pp = torch.cat((p, torch.full((1, 9, 2), float('nan'), dtype=torch.float64)), 1)
        xp = torch.cat((x, torch.full((1, 9, 96), float('nan'), dtype=torch.float64)), 1)
        padded = block(xp, vp, *arc_positions(pp, vp))
        torch.testing.assert_close(padded[:, :37], expected, atol=2e-12, rtol=2e-12)
        self.assertEqual(float(padded[:, 37:].abs().sum()), 0)

    def test_long_range_gradient_reaches_distant_contour_token(self):
        block = ContourSelfBlock(96, self.cfg)
        with torch.no_grad():
            block.out.weight.normal_(0, .02)
        x = torch.randn(1, 128, 96, requires_grad=True)
        p = torch.randn(1, 128, 2).cumsum(1)
        v = torch.ones(1, 128, dtype=torch.bool)
        out = block(x, v, *arc_positions(p, v))
        out[0, 0, 0].backward()
        self.assertGreater(float(x.grad[0, 90].abs().sum()), 1e-7)

    def test_cross_ignores_padding_and_empty_key_set_is_finite(self):
        block = CrossBlock(96, self.cfg)
        with torch.no_grad():
            block.out.weight.normal_(0, .02)
        x, y = torch.randn(1, 5, 96), torch.randn(1, 7, 96)
        xv, yv = torch.ones(1, 5, dtype=torch.bool), torch.ones(1, 7, dtype=torch.bool)
        expected = block(x, xv, y, yv)
        yp = torch.cat((y, torch.full((1, 3, 96), float('nan'))), 1)
        yvp = torch.cat((yv, torch.zeros(1, 3, dtype=torch.bool)), 1)
        torch.testing.assert_close(block(x, xv, yp, yvp), expected, atol=3e-7, rtol=2e-6)
        empty = block(x, xv, yp, torch.zeros_like(yvp))
        self.assertTrue(bool(torch.isfinite(empty).all()))
        self.assertEqual(float(block(x, torch.zeros_like(xv), y, yv).abs().sum()), 0)

    def test_all_invalid_self_contour_and_arc(self):
        block = ContourSelfBlock(96, self.cfg)
        valid = torch.zeros(1, 4, dtype=torch.bool)
        p = torch.full((1, 4, 2), float('nan'))
        arc, per = arc_positions(p, valid)
        self.assertEqual(float(arc.abs().sum()), 0)
        self.assertEqual(float(per[0]), 1)
        output = block(torch.full((1, 4, 96), float('nan')), valid, arc, per)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertEqual(float(output.abs().sum()), 0)

    def test_invalid_configs_fail_before_allocating_training(self):
        for kwargs in ({'enabled': 1}, {'layers': 0}, {'max_sharpness': float('nan')},
                       {'enabled': True, 'use_long_context': False, 'use_cross': False}):
            with self.assertRaises((ValueError, TypeError)):
                MatcherV2Config(**kwargs)
        with self.assertRaises(ValueError):
            ContourSelfBlock(95, self.cfg)
        with self.assertRaises(ValueError):
            ContourSelfBlock(96, replace(self.cfg, max_harmonic=3))


if __name__ == '__main__':
    unittest.main()

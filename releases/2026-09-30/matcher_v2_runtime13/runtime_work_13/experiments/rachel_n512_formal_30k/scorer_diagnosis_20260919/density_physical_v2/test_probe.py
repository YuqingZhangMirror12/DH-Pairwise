"""Small synthetic CPU tests; no checkpoint/data inference or remote access."""
import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.density_physical_v2 import probe
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.context_dilation_v1.functional_context import context_forward
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, RachelN512Pairwise
from staging.pairwise_v0_2.models.rachel_decoupled_score import CrossAttentionPairHead
from staging.pairwise_v0_2.models.translation_layout import TranslationLayoutConfig, estimate_translation_layout
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import _dense_external_contour, _arc_resample_closed


class DensityPhysicalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20260920)

    def test_short_contours_always_resampled_and_nested(self):
        a = np.zeros((32, 32), np.uint8)
        a[6:23, 9:21] = 1
        points, detail = probe.uniform_sets(dict(a=a, b=a.T.copy()), _dense_external_contour, _arc_resample_closed)
        for side in "ab":
            self.assertLess(detail[side]["dense_count"], 512)
            self.assertEqual(points[512][side].shape, (512, 2))
            self.assertEqual(points[1024][side].shape, (1024, 2))
            np.testing.assert_array_equal(points[512][side], points[1024][side][::2])

    def test_hook_runs_full_matcher_retains_padding_and_removes_itself(self):
        config = RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=32,
            feature_dim=8, num_heads=2, landmark_count=4, context_layers=1,
            window_sizes_px=(7., 16.), sinkhorn_iterations=10)
        base = RachelN512Pairwise(config).eval().requires_grad_(False)
        model = SimpleNamespace(base_model=base, score_head=CrossAttentionPairHead(8, 2).eval())
        masks = {s: np.zeros((32, 32), np.uint8) for s in "ab"}
        masks["a"][5:24, 7:22] = 1
        masks["b"][8:27, 4:19] = 1
        points, valid = {}, {}
        for side, n in (("a", 12), ("b", 16)):
            points[side] = np.zeros((16, 2), np.float32)
            points[side][:n] = _arc_resample_closed(_dense_external_contour(masks[side].astype(bool)), n)
            valid[side] = np.arange(16) < n
        seen = []
        def decoder(a, b, q, va, vb, config):
            seen.append(q.copy())
            return estimate_translation_layout(a, b, q, va, vb, config=config)
        before = probe.state_digest(base)
        cfg = TranslationLayoutConfig(correspondence_mode="topk_union")
        row1, snap1 = probe.forward(model, masks, points, valid, context_forward, decoder, cfg, verify_d1=True)
        row2, snap2 = probe.forward(model, masks, points, valid, context_forward, decoder, cfg, dilation=2)
        self.assertEqual(row1["full"]["valid_counts"], [12, 16])
        self.assertEqual(snap1["context"][0].shape, (1, 16, 8))
        self.assertFalse(torch.equal(snap1["context"][0], snap2["context"][0]))
        self.assertFalse(np.array_equal(seen[0], seen[1]))
        self.assertEqual(probe.state_digest(base), before)
        self.assertFalse(base.context._forward_hooks)
        self.assertFalse(base.patch_sampler._forward_hooks)

    def test_error_does_not_leave_hook(self):
        config = RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
            feature_dim=8, num_heads=2, context_layers=1)
        base = RachelN512Pairwise(config).eval()
        model = SimpleNamespace(base_model=base)
        mask = np.zeros((32, 32), np.uint8)
        points = np.zeros((16, 2), np.float32)
        with self.assertRaises(Exception):
            probe.forward(model, dict(a=mask, b=mask), dict(a=points, b=points),
                dict(a=np.ones(16, bool), b=np.ones(16, bool)), context_forward,
                estimate_translation_layout, TranslationLayoutConfig())
        self.assertFalse(base.context._forward_hooks)
        self.assertFalse(base.patch_sampler._forward_hooks)

    def test_zero_reference_is_explicitly_undefined_relative_error(self):
        self.assertIsNone(probe.difference(torch.zeros(2), torch.zeros(2))["relative_l2"])


if __name__ == "__main__":
    unittest.main()

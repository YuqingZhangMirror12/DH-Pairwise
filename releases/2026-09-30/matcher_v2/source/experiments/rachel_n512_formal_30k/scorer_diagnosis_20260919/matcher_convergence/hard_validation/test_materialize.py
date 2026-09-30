"""Tiny CPU-only validation fixtures; no real data, model or worker launch."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample
from . import materialize as m


def sample(positive=True):
    mask = np.zeros((1, 8, 8), np.float32)
    mask[:, 1:7, 1:7] = 1
    points = np.asarray([[1, 1], [1, 3], [1, 6], [0, 0]], np.float32)
    valid = np.asarray([True, True, True, False])
    target = np.asarray([0, 1, -1, -2] if positive else [-1, -1, -1, -2], np.int64)
    translation = np.asarray([2, 3] if positive else [np.nan, np.nan], np.float32)
    return RachelPairSample(pair_id="positive" if positive else "negative",
        fragment_a_token="a", fragment_b_token="b", mask_a=mask.copy(), mask_b=mask.copy(),
        coarse_mask_a=mask.copy(), coarse_mask_b=mask.copy(),
        points_rc_a=points.copy(), points_rc_b=points.copy(),
        contour_valid_a=valid.copy(), contour_valid_b=valid.copy(),
        target_a=target.copy(), target_b=target.copy(), label=np.float32(positive),
        translation_a_to_b_rc=translation,
        translation_a_to_b_xy_cartesian=translation[::-1].copy(),
        translation_valid=np.bool_(positive))


def rows():
    return [dict(pair_id="source-%d" % i, label=i % 2, split="val") for i in range(16)]


class HardValidationMaterializeTests(unittest.TestCase):
    def test_plan_is_deterministic_balanced_and_does_not_change_source_order(self):
        source = rows()
        before = deepcopy(source)
        plan = m.group_plan(source)
        self.assertEqual(plan, m.group_plan(source))
        self.assertEqual(source, before)
        self.assertEqual(Counter(p["recipe"] for p in plan), {r: 2 for r in m.RECIPES})
        self.assertEqual([p["source_indices"] for p in plan], [[i+1, i] for i in range(0, 16, 2)])
        self.assertEqual(sorted(i for p in plan for i in p["source_indices"]), list(range(16)))
        for bad in ([], source[:-1], source[:6]):
            with self.subTest(size=len(bad)), self.assertRaises(ValueError):
                m.group_plan(bad)

    def test_identity_gt_and_original_frame_subset_are_preserved(self):
        old = sample()
        mask = old.mask_a.copy()
        mask[0, 1, 1] = 0
        new = replace(old, mask_a=mask)
        report = m.changed_report(old, new, "wave")
        self.assertTrue(m.validate_variant(old, new, report))
        self.assertFalse(report["pose_supervision_enabled"])
        for update in (dict(pair_id="other"), dict(fragment_b_token="other"),
                dict(label=np.float32(0)), dict(translation_valid=False),
                dict(translation_a_to_b_rc=np.asarray([3, 3])),
                dict(translation_a_to_b_xy_cartesian=np.asarray([3, 3])),
                dict(mask_a=np.ones_like(old.mask_a)), dict(mask_a=old.mask_a[:, :-1])):
            with self.subTest(fields=list(update)), self.assertRaises(ValueError):
                m.validate_variant(old, replace(old, **update), report)
        with self.assertRaises(ValueError):
            m.validate_variant(old, new, dict(report, changed_pair=False))
        with self.assertRaises(ValueError):
            m.validate_variant(old, new, dict(report, pose_supervision_enabled=True))

    def test_empty_foreground_is_not_a_valid_damage_variant(self):
        old = sample()
        for updates in (dict(mask_a=np.zeros_like(old.mask_a)),
                dict(mask_a=np.zeros_like(old.mask_a), mask_b=np.zeros_like(old.mask_b))):
            new = replace(old, **updates)
            with self.subTest(sides=list(updates)), self.assertRaises(ValueError):
                m.validate_variant(old, new, m.changed_report(old, new, "wave"))

    def test_positive_targets_are_reciprocal_and_only_on_valid_endpoints(self):
        old = sample()
        self.assertFalse(m.validate_variant(old, old, m.changed_report(old, old, "clean")))
        for updates in (dict(target_b=np.asarray([-1, 1, -1, -2])),
                dict(target_a=np.asarray([4, 1, -1, -2])),
                dict(target_a=np.asarray([0, 1, -1])),
                dict(target_a=np.asarray([0, 1, -3, -2])),
                dict(contour_valid_b=np.asarray([False, True, True, False]))):
            new = replace(old, **updates)
            with self.subTest(fields=list(updates)), self.assertRaises(ValueError):
                m.validate_variant(old, new, m.changed_report(old, old, "clean"))

    def test_negative_nan_gt_and_dustbin_ignore_targets_stay_negative(self):
        old = sample(False)
        self.assertFalse(m.validate_variant(old, old, m.changed_report(old, old, "clean")))
        for side in "ab":
            new = replace(old, **{"target_"+side: np.asarray([0, -1, -1, -2])})
            with self.subTest(side=side), self.assertRaises(ValueError):
                m.validate_variant(old, new, m.changed_report(old, old, "clean"))

    def test_clean_and_requested_fallback_ids_are_both_retained(self):
        originals = (sample(), sample(False))
        fallback = "coupled_geometry_or_supervision_rejection"
        variants = tuple((s, m.changed_report(s, s, "wave", fallback_reason=fallback)) for s in originals)
        state = dict(dataset=originals, options=dict(output="/unused/tiny-diagnostic", seed=m.SEED,
            family_flags={"positive": True, "negative": False}))
        plan = dict(group_index=3, source_indices=[0, 1], recipe="wave")
        with patch.dict(m.STATE, state, clear=True), patch.object(m, "strong_group", return_value=variants), \
                patch.object(m, "save_sample") as save_sample, patch.object(m, "save"):
            entries = m.materialize_group(plan)
        self.assertEqual(len(entries), 4)
        self.assertEqual({e["pair_id"] for e in entries},
            {"hardval-"+recipe+"::"+s.pair_id for recipe in ("clean", "wave") for s in originals})
        self.assertEqual([s.pair_id for s in originals], ["positive", "negative"])
        self.assertEqual(save_sample.call_count, 4)
        for call, entry in zip(save_sample.call_args_list, entries):
            _, saved, report = call.args
            self.assertEqual(saved.pair_id, entry["pair_id"])
            self.assertEqual(report["source_split"], "val")
            self.assertTrue(report["diagnostic_only"])
            self.assertFalse(entry["changed_pair"])
            self.assertEqual(entry["assigned_recipe"], "wave")
            self.assertEqual(entry["pose_supervision_enabled"], entry["label"])
            self.assertEqual(entry["source_family_overlap"], entry["label"])
            self.assertEqual(entry["fallback_reason"], fallback if entry["recipe"] == "wave" else None)

    def test_curve_adapter_keeps_val_identity_when_seam_is_unavailable(self):
        positive, negative = sample(), sample(False)
        adapter = m.ValidationCurvePair(positive, negative, SimpleNamespace(), m.SEED)
        with patch("staging.pairwise_v0_2.pairwise_data.rachel_guided_partial_dataset.source_seam_context",
                   return_value=None):
            values, report = adapter.materialize(0)
        self.assertIs(values[0], positive)
        self.assertIs(values[1], negative)
        self.assertEqual(adapter.split, "val")
        self.assertTrue(report["requested"])
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "original_source_seam_unavailable")

    def test_pilot_interleaves_requested_recipes_before_outcomes_and_is_not_train(self):
        seen = []
        source = rows()
        def generate(_function, plans, chunksize):
            self.assertEqual(chunksize, 1)
            seen.extend(plans)
            for plan in plans:
                yield [dict(pair_id="%d-%s-%s" % (plan["group_index"], label, recipe),
                    label=label, recipe=recipe, changed_pair=False,
                    fallback_reason=None if recipe == "clean" else "test_fallback")
                    for label in (True, False) for recipe in ("clean", plan["recipe"])]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(dataset=str(root/"source"), outline_bank=str(root/"bank"),
                train_manifest=str(root/"train.json"), output=str(root/"diagnostic"), workers=1,
                limit_groups=5)
            with patch.dict(m.os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                    patch.object(m, "source_rows", return_value=source), \
                    patch.object(m, "source_provenance", return_value=({}, {})), \
                    patch.object(m, "sha", return_value="fixture-sha"), \
                    patch.object(m, "ProcessPoolExecutor") as pool:
                pool.return_value.__enter__.return_value.map.side_effect = generate
                summary = m.run(args)
            manifest = json.loads((root/"diagnostic/manifest.json").read_text())
            status = json.loads((root/"diagnostic/status.json").read_text())
        self.assertEqual([p["recipe"] for p in seen], list(m.RECIPES)+[m.RECIPES[0]])
        self.assertEqual(summary["count"], 20)
        self.assertEqual(summary["fallback_reasons"], {"test_fallback": 10})
        self.assertEqual(manifest["split"], "val")
        self.assertEqual(status["status"], "pilot_complete")
        self.assertTrue(manifest["protocol"]["diagnostic_only"])
        for flag in ("training_eligible", "threshold_fitting", "model_selection", "real_ood_used"):
            self.assertFalse(manifest["protocol"][flag])
        for counts in summary["by_recipe"].values():
            self.assertEqual(counts["positive"], counts["negative"])
            self.assertEqual(counts["changed_positive"], 0)
            self.assertEqual(counts["changed_negative"], 0)


if __name__ == "__main__":
    unittest.main()

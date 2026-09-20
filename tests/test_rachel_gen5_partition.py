"""Small CPU-only source-geometry fixtures; no released or held-out data."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest

import numpy as np

from staging.pairwise_v0_2.pairwise_data import rachel_gen5_partition as target
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import RachelPreprocessConfig, _make_pair


CONFIG = target.UnionConfig(canvas_size=96, contour_cap=128, smoothing_sigma=0,
                           minimum_seam_px=24, minimum_token_matches=1)


def strips():
    masks = {}
    for i in range(5):
        masks[str(i)] = np.zeros((96, 96), bool)
        masks[str(i)][8:88, 8 + 16 * i:24 + 16 * i] = True
    return masks


def make(plan=None, masks=None, **overrides):
    args = dict(group_id="synthetic", lineage_id="train-parent", lineage_splits={"train-parent": "train"},
                neighbor_edges=[("0", "1"), ("1", "2"), ("2", "3"), ("3", "4")],
                plan=plan or target.Gen5PartitionPlan((("0",), ("1", "2"), ("3", "4"))), config=CONFIG)
    args.update(overrides)
    return target.build_gen5_partition(strips() if masks is None else masks, **args)


class Gen5PartitionTests(unittest.TestCase):
    def test_enumeration_counts_equivalent_orders_once(self):
        plans = target.enumerate_gen5_partition_plans(["4", "1", "3", "0", "2"])
        self.assertEqual(len(plans), 25)
        self.assertEqual(len({p.canonical_groups for p in plans}), 25)
        small = [p for p in plans if sorted(map(len, p.groups)) == [1, 2, 2]]
        large = [p for p in plans if sorted(map(len, p.groups)) == [1, 1, 3]]
        self.assertEqual((len(small), len(large)), (15, 10))
        for plan in plans:
            self.assertEqual(sorted(x for group in plan.groups for x in group), list("01234"))
        self.assertTrue(all(p.ordered_patterns == ((1, 2, 2), (2, 1, 2)) for p in small))
        self.assertTrue(all(p.ordered_patterns == ((3, 1, 1),) for p in large))
        one = target.enumerate_gen5_partition_plans(list("01234"), ((2, 1, 2),))
        self.assertEqual(len(one), 15)
        self.assertTrue(all(tuple(map(len, p.groups)) == (2, 1, 2) for p in one))
        self.assertEqual({p.canonical_groups for p in one}, {p.canonical_groups for p in small})

    def test_invalid_ids_patterns_and_plan_cannot_drop_or_repeat_members(self):
        for ids in (["0", "1", "2", "3"], ["0"] * 5, [0, "1", "2", "3", "4"]):
            with self.subTest(ids=ids), self.assertRaises(target.UnionAugmentationError):
                target.enumerate_gen5_partition_plans(ids)
        for pattern in (((1, 1, 3),), ((4, 1, 0),), ((True, 2, 2),), ()):
            with self.subTest(pattern=pattern), self.assertRaises(target.UnionAugmentationError):
                target.enumerate_gen5_partition_plans(list("01234"), pattern)
        for plan in (target.Gen5PartitionPlan((("0",), ("1", "2"), ("3", "3"))),
                     target.Gen5PartitionPlan((("0",), ("1",), ("2", "3", "4"))),
                     target.Gen5PartitionPlan((("0",), ("1", "2"), ("3", "4")), ((3, 1, 1),))):
            with self.subTest(plan=plan), self.assertRaises(target.UnionAugmentationError):
                make(plan)

    def test_exact_or_disjoint_support_third_group_and_no_source_mutation(self):
        masks = strips()
        before = deepcopy(masks)
        result = make(masks=masks)
        for fragment, group in zip(result.fragments, result.metadata["groups"]):
            expected = np.logical_or.reduce([masks[x] for x in group["member_ids"]])
            np.testing.assert_array_equal(fragment.parent_mask, expected)
        union = np.stack([f.parent_mask for f in result.fragments])
        self.assertEqual(int(union.sum(0).max()), 1)
        np.testing.assert_array_equal(union.any(0), np.stack(list(masks.values())).any(0))
        for key in masks:
            np.testing.assert_array_equal(masks[key], before[key])
        self.assertEqual(len(result.pairs), 3)
        for item in result.pairs:
            info = item.metadata
            selected = sum(info["selected_group_members"], [])
            third = info["third_group"]["member_ids"]
            self.assertFalse(set(selected) & set(third))
            self.assertEqual(sorted(selected + third), list("01234"))
            self.assertEqual(info["area_a_px"] + info["area_b_px"] + info["third_group"]["area_px"],
                             info["original_parent_area_px"])
            self.assertTrue(info["third_group_not_in_pair"])
            self.assertEqual(info["omitted_parent_members"], [])
            self.assertFalse(info["holes_filled"] or info["resized"] or info["gap_closing_applied"])
            json.dumps(info, allow_nan=False)

    def test_csv_labels_and_parent_translation_match_existing_preprocessor(self):
        result = make()
        self.assertEqual([p.pair.label for p in result.pairs], [True, False, True])
        for item in result.pairs:
            pair = item.pair
            cfg = RachelPreprocessConfig(canvas_size=96, contour_cap=128, minimum_positive_seam=24,
                                         contour_smoothing_sigma=0)
            expected = _make_pair(item.fragment_a, item.fragment_b, label=pair.label, config=cfg)
            np.testing.assert_array_equal(pair.dense_correspondences, expected.dense_correspondences)
            np.testing.assert_array_equal(pair.token_correspondences, expected.token_correspondences)
            self.assertEqual(pair.translation_a_to_b_rc, expected.translation_a_to_b_rc)
            self.assertEqual(pair.main_training_eligible, expected.main_training_eligible)
            self.assertTrue(pair.main_training_eligible)
            if pair.label:
                shift = np.subtract(item.fragment_b.model.parent_to_model_offset_rc,
                                    item.fragment_a.model.parent_to_model_offset_rc)
                np.testing.assert_array_equal(pair.translation_a_to_b_rc, shift)
                self.assertEqual(pair.translation_a_to_b_xy, (float(shift[1]), float(-shift[0])))
                self.assertGreater(len(pair.token_correspondences), 0)
            else:
                self.assertIsNone(pair.translation_a_to_b_rc)
                self.assertIsNone(item.metadata["translation_a_to_b_rc"])
                self.assertEqual(len(pair.token_correspondences), 0)

    def test_three_one_one_is_supported_with_all_five_in_metadata(self):
        result = make(target.Gen5PartitionPlan((("0", "1", "2"), ("3",), ("4",))))
        self.assertEqual(result.metadata["ordered_pattern"], [3, 1, 1])
        self.assertEqual([x.pair.label for x in result.pairs], [True, False, True])
        self.assertTrue(all(x.pair.main_training_eligible for x in result.pairs))

    def test_csv_geometry_disagreement_is_quarantined_not_relabelled(self):
        missing = make(neighbor_edges=[])
        self.assertTrue(all(not x.pair.label for x in missing.pairs))
        self.assertEqual([x.pair.main_training_eligible for x in missing.pairs], [False, True, False])
        self.assertEqual(missing.pairs[0].pair.status, "quarantined")
        self.assertIsNone(missing.pairs[0].pair.translation_a_to_b_rc)
        extra = make(neighbor_edges=[("0", "4")])
        self.assertTrue(extra.pairs[1].pair.label)
        self.assertFalse(extra.pairs[1].pair.main_training_eligible)
        self.assertEqual(extra.pairs[1].pair.status, "quarantined")

    def test_short_positive_seams_remain_positive_excluded(self):
        result = make(config=replace(CONFIG, minimum_seam_px=1000))
        self.assertEqual([x.pair.label for x in result.pairs], [True, False, True])
        self.assertTrue(all(not x.pair.main_training_eligible for x in result.pairs if x.pair.label))
        few = make(config=replace(CONFIG, minimum_token_matches=1000))
        self.assertTrue(all(x.pair.selection_exclusion_reason == "too_few_true_seam_token_matches"
                            for x in few.pairs if x.pair.label))

    def test_train_guard_precedes_geometry_and_generator_scale_are_strict(self):
        for split in (None, "val", "test", "real", "ood"):
            with self.subTest(split=split), self.assertRaisesRegex(target.UnionAugmentationError, "frozen TRAIN"):
                make(masks={}, lineage_splits={"train-parent": split})
        for override in (dict(generator="gen4voronoi"), dict(scale=.5), dict(scale=True),
                         dict(neighbor_edges=[("0", "x")]), dict(neighbor_edges=[("0", "0")])):
            with self.subTest(override=override), self.assertRaises(target.UnionAugmentationError):
                make(**override)

    def test_overlaps_and_disconnected_unions_are_rejected_never_repaired(self):
        masks = strips()
        masks["1"][8, 8] = True
        with self.assertRaisesRegex(target.UnionGeometryRejected, "source_fragment_overlap"):
            make(masks=masks)
        with self.assertRaisesRegex(target.UnionGeometryRejected, "partition_group_disconnected"):
            make(target.Gen5PartitionPlan((("0", "2"), ("1",), ("3", "4"))))

    def test_holes_are_not_filled(self):
        masks = {str(i): np.zeros((96, 96), bool) for i in range(5)}
        masks["0"][8:40, 8:40] = True
        masks["0"][16:32, 16:32] = False
        for i in range(1, 5):
            masks[str(i)][52:80, 8 + (i - 1) * 16:24 + (i - 1) * 16] = True
        result = make(masks=masks, neighbor_edges=[("1", "2"), ("2", "3"), ("3", "4")])
        np.testing.assert_array_equal(result.fragments[0].parent_mask, masks["0"])
        self.assertEqual(result.fragments[0].foreground_area, 32 * 32 - 16 * 16)
        self.assertEqual(result.fragments[0].model.model_mask.sum(), masks["0"].sum())

    def test_reordered_aliases_have_same_ids_and_geometry_and_are_deduplicated(self):
        first = make()
        reordered = make(target.Gen5PartitionPlan((("1", "2"), ("0",), ("3", "4")), ((2, 1, 2),)))
        self.assertEqual(first.metadata["partition_id"], reordered.metadata["partition_id"])
        self.assertEqual({p.metadata["pair_id"] for p in first.pairs},
                         {p.metadata["pair_id"] for p in reordered.pairs})
        self.assertEqual({p.metadata["geometry_signature"] for p in first.pairs},
                         {p.metadata["geometry_signature"] for p in reordered.pairs})
        unique = target.unique_partition_pairs([first, reordered, first])
        self.assertEqual(len(unique), 3)
        self.assertTrue(all(len(item.metadata["geometry_provenance"]) == 2 for item in unique))
        self.assertNotIn("geometry_provenance", first.pairs[0].metadata)

    def test_geometry_deduplication_does_not_depend_on_source_ids_and_checks_tampering(self):
        first, second = make(), make(group_id="same-geometry-alternate-source")
        unique = target.unique_partition_pairs([first, second])
        self.assertEqual(len(unique), 3)
        self.assertTrue(all(len(x.metadata["geometry_provenance"]) == 2 for x in unique))
        item = replace(first.pairs[0], metadata=dict(first.pairs[0].metadata, geometry_signature="wrong"))
        with self.assertRaisesRegex(target.UnionAugmentationError, "signature"):
            target.unique_partition_pairs([replace(first, pairs=(item,))])
        item = replace(first.pairs[0], metadata=dict(first.pairs[0].metadata, split="test"))
        with self.assertRaisesRegex(target.UnionAugmentationError, "TRAIN"):
            target.unique_partition_pairs([replace(first, pairs=(item,))])


if __name__ == "__main__":
    unittest.main()

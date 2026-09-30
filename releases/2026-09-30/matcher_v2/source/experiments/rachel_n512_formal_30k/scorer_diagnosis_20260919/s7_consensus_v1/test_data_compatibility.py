"""No network execution: exercise real collation and supervision adapters."""
from dataclasses import replace
import unittest

import numpy as np
import torch

from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample
from .check_data_compatibility import check_batch
from .data import collate


def item(recipe='partial', positive=True, mode='middle'):
    target = np.array([0, 1, -2, -1], np.int64) if positive else np.full(4, -1, np.int64)
    mask = np.ones((1, 8, 8), np.float32)
    points = np.array([[0, 0], [0, 7], [7, 7], [7, 0]], np.float32)
    valid = np.ones(4, np.bool_)
    sample = RachelPairSample('example', 'a', 'b', mask, mask.copy(), mask.copy(), mask.copy(),
        points, points.copy(), valid, valid.copy(), target, target.copy(), np.float32(positive),
        np.array([3, 7], np.float32), np.array([7, -3], np.float32), np.bool_(positive))
    report = dict(compound=dict(recipe=recipe, partial=dict(mode=mode) if recipe=='partial' else {}))
    return sample, report, dict(pair_id=sample.pair_id, label=positive, recipe=recipe)


class DataCompatibilityTests(unittest.TestCase):
    def test_partial_keeps_global_gt_but_not_precise_anchors(self):
        items = [item()]
        rows = check_batch(items, collate(items))
        self.assertTrue(rows[0]['global_gt_retained'])
        self.assertEqual(rows[0]['inherited_pairs'], 2)
        self.assertEqual(rows[0]['precise_pairs'], 0)
        self.assertEqual(rows[0]['partial_mode'], 'middle')
        self.assertEqual(rows[0]['unknown_valid_points'], dict(a=1,b=1))

    def test_clean_and_negative_in_same_batch(self):
        items = [item('clean'), item('gaps', positive=False)]
        rows = check_batch(items, collate(items))
        self.assertEqual([r['precise_pairs'] for r in rows], [2, 0])
        self.assertEqual([r['inherited_pairs'] for r in rows], [2, 0])

    def test_nonpartial_materialized_report_may_have_null_partial(self):
        s, r, e = item('mild')
        r['compound']['partial'] = None
        items = [(s, r, e)]
        rows = check_batch(items, collate(items))
        self.assertIsNone(rows[0]['partial_mode'])

    def test_wrong_recipe_is_rejected(self):
        s, r, e = item()
        e = dict(e, recipe='clean')
        with self.assertRaisesRegex(ValueError, 'materialized damage'):
            check_batch([(s,r,e)], collate([(s,r,e)]))

    def test_collation_must_not_change_masks_coordinates_or_labels(self):
        for key in ('mask_a', 'points_rc_a', 'target_a', 'translation_a_to_b_rc'):
            with self.subTest(key=key):
                items = [item()]
                batch = collate(items)
                batch[key].flatten()[0] += 1
                with self.assertRaisesRegex(ValueError, 'archived'):
                    check_batch(items, batch)

    def test_padding_must_remain_unknown(self):
        items = [item()]
        batch = collate(items)
        batch['target_a'][0, -1] = -1
        with self.assertRaisesRegex(ValueError, 'padding'):
            check_batch(items, batch)

    def test_background_damage_not_inferred_as_precise_from_small_residual(self):
        s, r, e = item('mild')
        s = replace(s, translation_a_to_b_rc=np.zeros(2, np.float32))
        items = [(s,r,e)]
        rows = check_batch(items, collate(items))
        self.assertEqual(rows[0]['precise_pairs'], 0)
        batch = collate(items)
        batch['precise_recipe'][:] = True
        with self.assertRaisesRegex(ValueError, 'incorrectly treated as clean'):
            check_batch(items, batch)


if __name__ == '__main__':
    unittest.main()

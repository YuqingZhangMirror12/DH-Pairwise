from collections import Counter
from dataclasses import replace
import unittest

from ..curriculum_training_v1 import test_exposure as baseline
from ..curriculum_training_v1.exposure import RankMicrobatches, digest
from .additive_exposure import StraightSampleRef, build_additive_ledger


def extras():
    return tuple(StraightSampleRef('sseam-' + str(i), 'source-' + str(i), digest(['in', i]),
        '/admitted/' + str(i) + '.npz', digest(['file', i]), bool(i % 2)) for i in range(12))


def build():
    return build_additive_ledger(baseline.ledger(), extras(), ((0, 3, 1), (3, 7, 3),
                                (7, 12, 2), (12, 16, 2)), seed=26093053)


class AdditiveExposureTests(unittest.TestCase):
    def test_preserve_every_original_exposure_in_original_order(self):
        original, added = baseline.ledger(), build()
        for order in ('curriculum', 'mixed'):
            retained = [i for i in added.sequence(order) if i < len(original.catalog)]
            self.assertEqual(tuple(retained), original.sequence(order))
            self.assertEqual(Counter(retained), Counter(original.sequence(order)))
        self.assertEqual(added.total_updates, 24)
        self.assertFalse(added.summary()['same_budget_as_base'])

    def test_starts_in_first_and_early_followup_window_not_only_late(self):
        added = build()
        self.assertEqual(added.slots[0], ('straight', 0, 0))
        self.assertTrue(any(k == 'straight' and p == 3 for k, _, p in added.slots))

    def test_all_batches_balanced_unique_and_all_added_seen(self):
        added = build()
        for order in ('curriculum', 'mixed'):
            for batch in baseline.batches(added.sequence(order), 4):
                self.assertEqual(len(set(batch)), 4)
                self.assertEqual(sum(added.catalog[i].label for i in batch), 2)
            self.assertEqual(set(i for i in added.sequence(order) if i >= added.base_catalog_size),
                             set(range(added.base_catalog_size, len(added.catalog))))

    def test_extra_rng_does_not_perturb_base_counts_or_order(self):
        a = build()
        b = build_additive_ledger(baseline.ledger(), reversed(extras()), a.windows, seed=88)
        self.assertNotEqual(a.sha256, b.sha256)
        self.assertEqual([i for i in a.curriculum if i < a.base_catalog_size],
                         [i for i in b.curriculum if i < b.base_catalog_size])
        self.assertEqual(Counter(a.curriculum), Counter(a.mixed))

    def test_resume_cursor_is_actual_update_and_all_rank_slices_reconstruct(self):
        ledger = build()
        ranks = [RankMicrobatches(ledger, 'curriculum', 5, rank, 2, 1, 2) for rank in range(2)]
        pieces = [list(rank) for rank in ranks]
        merged = []
        for i in range(len(pieces[0])):
            for rank in range(2):
                merged.extend(pieces[rank][i])
        self.assertEqual(tuple(merged), ledger.curriculum[5 * 4:])
        cursor = ranks[0].cursor(9)
        resumed = RankMicrobatches.from_cursor(ledger, cursor, 1, 'curriculum', 2, 1, 2)
        self.assertEqual(list(resumed), list(RankMicrobatches(ledger, 'curriculum', 9, 1, 2, 1, 2)))

    def test_lr_clock_does_not_skip_original_updates(self):
        added = build()
        actual_base = 0
        for step, (kind, _, _) in enumerate(added.slots):
            self.assertEqual(added.base_completed(step), actual_base)
            actual_base += kind == 'base'
        self.assertEqual(added.base_completed(added.total_updates), 16)
        with self.assertRaises(ValueError):
            added.base_completed(25)

    def test_invalid_windows_and_heldout_or_duplicate_inputs_rejected(self):
        base = baseline.ledger()
        for windows in ((), ((0, 15, 8),), ((1, 16, 8),), ((0, 16, 0),), ((0, 3, 2), (2, 16, 8))):
            with self.assertRaises(ValueError):
                build_additive_ledger(base, extras(), windows, seed=9)
        for change in (dict(split='select'), dict(label=1), dict(sample_path='relative'),
                       dict(model_input_sha256=base.catalog[0].model_input_sha256),
                       dict(pair_id=base.catalog[0].pair_id)):
            samples = list(extras()); samples[0] = replace(samples[0], **change)
            with self.assertRaises(ValueError):
                build_additive_ledger(base, samples, ((0, 16, 8),), seed=9)


if __name__ == '__main__':
    unittest.main()

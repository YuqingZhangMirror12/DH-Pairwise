"""Sampling tests, not GPU or pixel-admission evidence."""
from collections import Counter
from dataclasses import replace
import unittest

from .exposure import (STAGES, SampleRef, RankMicrobatches, build_ledger,
                       canonical_catalog, digest, learning_rate_at)


def samples(counts=(5, 4, 3)):
    rows = []
    for stage, count in zip(STAGES, counts):
        for label in (False, True):
            for ordinal in range(count):
                name = '%s-%s-%d' % (stage, label, ordinal)
                rows.append(SampleRef(stage, name, 'base::' + name, digest(['input', name]),
                                      '/unused/' + name + '.npz', digest(['file', name]), label))
    return rows


def ledger():
    return build_ledger(samples(), dict(zip(STAGES, (7, 5, 4))), seed=260928,
                        effective_batch=4)


def batches(sequence, batch):
    return [tuple(sequence[i:i + batch]) for i in range(0, len(sequence), batch)]


class ExposureTests(unittest.TestCase):
    def test_same_each_sample_exposures_not_only_total(self):
        value = ledger()
        self.assertEqual(Counter(value.curriculum), Counter(value.mixed))
        self.assertEqual(set(value.curriculum), set(range(len(value.catalog))))
        self.assertEqual(value.total_updates, 16)

    def test_same_batch_contents_only_order_changes(self):
        value = ledger()
        self.assertEqual(Counter(batches(value.curriculum, 4)), Counter(batches(value.mixed, 4)))
        self.assertNotEqual(value.curriculum, value.mixed)
        self.assertTrue(value.summary()['same_global_batches'])

    def test_curriculum_stage_boundaries(self):
        value = ledger()
        actual = [value.catalog[b[0]].stage for b in batches(value.curriculum, 4)]
        self.assertEqual(actual, [STAGES[0]] * 7 + [STAGES[1]] * 5 + [STAGES[2]] * 4)
        self.assertEqual([x['end_update'] for x in value.summary()['stage_plan']], [7, 12, 16])

    def test_mixed_stages_not_another_three_phase_schedule(self):
        value = ledger()
        stages = [value.catalog[b[0]].stage for b in batches(value.mixed, 4)]
        switches = sum(a != b for a, b in zip(stages, stages[1:]))
        self.assertGreater(switches, 2)
        self.assertGreater(len(set(stages[:7])), 1)

    def test_every_update_balanced_and_no_duplicate_sample(self):
        value = ledger()
        for order in ('curriculum', 'mixed'):
            for batch in batches(value.sequence(order), 4):
                self.assertEqual(sum(value.catalog[i].label for i in batch), 2)
                self.assertEqual(len(set(batch)), 4)

    def test_odd_catalog_size_no_padding_or_dropped_exposure(self):
        value = ledger()
        self.assertEqual(len(value.curriculum), 16 * 4)
        for stage, updates in zip(STAGES, value.stage_updates):
            self.assertEqual(sum(value.catalog[i].stage == stage for i in value.curriculum), updates * 4)

    def test_canonical_input_order_does_not_change_ledger(self):
        value = build_ledger(reversed(samples()), dict(zip(STAGES, (7, 5, 4))), 260928, 4)
        self.assertEqual(value, ledger())

    def test_seed_changes_binding_and_order_not_budget(self):
        first = ledger()
        other = build_ledger(samples(), dict(zip(STAGES, (7, 5, 4))), 99, 4)
        self.assertNotEqual(first.sha256, other.sha256)
        self.assertNotEqual(first.curriculum, other.curriculum)
        self.assertEqual(first.total_updates, other.total_updates)

    def test_catalog_artifact_identity_is_bound(self):
        original = samples(); changed = list(original)
        changed[0] = replace(changed[0], sample_sha256=digest('changed'))
        a = build_ledger(original, dict(zip(STAGES, (7, 5, 4))), 1, 4)
        b = build_ledger(changed, dict(zip(STAGES, (7, 5, 4))), 1, 4)
        self.assertNotEqual(a.sha256, b.sha256)

    def test_cross_stage_base_reuse_rejected(self):
        rows = samples(); rows[10] = replace(rows[10], source_base_key=rows[0].source_base_key)
        with self.assertRaisesRegex(ValueError, 'identity reused'):
            canonical_catalog(rows)

    def test_repeated_hard_base_rejected_even_if_input_differs(self):
        rows = samples(); rows[11] = replace(rows[11], source_base_key=rows[10].source_base_key)
        with self.assertRaisesRegex(ValueError, 'identity reused'):
            canonical_catalog(rows)

    def test_distinct_v17_views_remain_visible_as_shared_original_identity(self):
        rows = samples(); rows[1] = replace(rows[1], source_base_key=rows[0].source_base_key)
        value = build_ledger(rows, dict(zip(STAGES, (7, 5, 4))), 1, 4)
        report = value.summary()['stage_plan'][0]
        self.assertEqual(report['unique_pairs'], 10)
        self.assertEqual(report['unique_original_pairs'], 9)

    def test_duplicate_actual_input_rejected(self):
        rows = samples(); rows[1] = replace(rows[1], model_input_sha256=rows[0].model_input_sha256)
        with self.assertRaisesRegex(ValueError, 'actual model input'):
            canonical_catalog(rows)

    def test_duplicate_id_rejected(self):
        rows = samples(); rows[1] = replace(rows[1], pair_id=rows[0].pair_id)
        with self.assertRaisesRegex(ValueError, 'Pair ID'):
            canonical_catalog(rows)

    def test_invalid_catalog_rejected(self):
        for change in [dict(label=2), dict(sample_sha256='bad'), dict(sample_path=''), dict(stage='v19')]:
            rows = samples(); rows[0] = replace(rows[0], **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                canonical_catalog(rows)

    def test_all_three_stages_and_both_classes_required(self):
        for rows in [[r for r in samples() if r.stage != STAGES[2]],
                     [r for r in samples() if r.stage != STAGES[2] or r.label]]:
            with self.assertRaises(ValueError):
                canonical_catalog(rows)

    def test_no_hidden_budget_auto_selection(self):
        for stages in [dict(zip(STAGES, (0, 5, 4))), dict(zip(STAGES, (1, 5, 4))),
                       dict(zip(STAGES, (True, 5, 4))), {'v17_filtered': 7},
                       dict(zip(STAGES, (36000, 5, 4)))]:
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                build_ledger(samples(), stages, 1, 4)

    def test_cannot_raise_total_cap_or_use_odd_batch(self):
        for kwargs in [dict(max_updates=108000), dict(effective_batch=3)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_ledger(samples(), dict(zip(STAGES, (7, 5, 4))), 1, **kwargs)


class CursorTests(unittest.TestCase):
    def test_two_ranks_and_accumulation_reassemble_global_order(self):
        value = ledger()
        for order in ('curriculum', 'mixed'):
            ranks = [list(RankMicrobatches(value, order, 0, r, 2, 1, 2)) for r in range(2)]
            reconstructed = []
            for update in range(value.total_updates):
                for accumulate in range(2):
                    for rank in range(2):
                        reconstructed.extend(ranks[rank][update * 2 + accumulate])
            self.assertEqual(tuple(reconstructed), value.sequence(order))

    def test_one_gpu_scorer_same_global_examples(self):
        value = ledger()
        loader = RankMicrobatches(value, 'curriculum', 0, 0, 1, 4, 1)
        self.assertEqual(list(map(tuple, loader)), batches(value.curriculum, 4))

    def test_resume_every_boundary_exact_without_padding(self):
        value = ledger()
        for order in ('curriculum', 'mixed'):
            for rank in (0, 1):
                original = RankMicrobatches(value, order, 0, rank, 2, 1, 2)
                all_batches = list(original)
                for completed in range(value.total_updates + 1):
                    cursor = original.cursor(completed)
                    resumed = RankMicrobatches.from_cursor(value, cursor, rank, order, 2, 1, 2)
                    self.assertEqual(list(resumed), all_batches[completed * 2:])
                    self.assertEqual(len(resumed), len(all_batches) - completed * 2)

    def test_ledger_order_topology_and_exposure_tampering_rejected(self):
        value = ledger(); sampler = RankMicrobatches(value, 'curriculum', 0, 0, 2, 1, 2)
        for key, changed in [('ledger_sha256', digest('changed')), ('order', 'mixed'),
                             ('world_size', 1), ('microbatch', 2), ('accumulate', 1),
                             ('completed_exposures', 99), ('completed_updates', True)]:
            cursor = sampler.cursor(3); cursor[key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                RankMicrobatches.from_cursor(value, cursor, 0, 'curriculum', 2, 1, 2)

    def test_wrong_world_or_effective_batch_rejected(self):
        for args in [(0, 2, 2, 1, 2), (0, 0, 2, 2, 2), (17, 0, 2, 1, 2)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                RankMicrobatches(ledger(), 'curriculum', *args)

    def test_cannot_record_backward_or_future_cursor(self):
        sampler = RankMicrobatches(ledger(), 'curriculum', 5, 0, 2, 1, 2)
        for at in (4, 17, -1):
            with self.assertRaises(ValueError):
                sampler.cursor(at)

    def test_common_global_lr_no_stage_reset(self):
        knots = [(0, .0001), (8, .00005), (13, .000025)]
        self.assertEqual([learning_rate_at(x, knots) for x in [0, 7, 8, 12, 13, 16]],
                         [.0001, .0001, .00005, .00005, .000025, .000025])

    def test_invalid_lr_plan_rejected(self):
        for knots in [[], [(1, .001)], [(0, float('nan'))], [(0, -1)],
                      [(0, .1), (0, .01)], [(0, .1), (False, .01)]]:
            with self.subTest(knots=knots), self.assertRaises(ValueError):
                learning_rate_at(3, knots)


if __name__ == '__main__':
    unittest.main()

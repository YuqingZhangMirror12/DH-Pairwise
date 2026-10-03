import copy
import unittest

from . import protocol as p
from . import selection as s
from .test_protocol import Fixture, row


def report(plan, update, loss=1.):
    rows = []
    for entry in plan['manifest_bindings']['select']['manifest']['entries']:
        rows.append(dict({k: entry[k] for k in ('pair_id', 'sample_sha256', 'stage', 'generator', 'label')},
            gt_known=entry['label'], layout20=entry['label'], candidate_coverage=entry['label'],
            numeric_valid=True, has_candidate=True, matcher_loss=loss, loss_semantics=p.LOSS_SEMANTICS,
            loss_components=dict(match_nll=2*loss, dustbin_nll=0., translation_smooth_l1=0., sinkhorn_residual=0.)))
    return dict(schema=s.REPORT_SCHEMA, protocol_sha256=plan['sha256'], update=update, role='select',
        manifest_sha256=plan['manifest_bindings']['select']['sha256'], status='completed', returncode=0,
        checkpoint_sha256=f'{update:064x}', model_state_sha256=f'{update+1:064x}', rows=rows)


def set_loss(item, loss):
    item['matcher_loss'] = loss
    item['loss_components']['match_nll'] = 2*loss


class SelectionTests(Fixture):
    def test_minimum_loss_then_layout_then_earliest_update(self):
        plan = self.plan()
        values = [report(plan, 300, .5), report(plan, 200, .5), report(plan, 100, .8)]
        result = s.select_checkpoint(plan, values)
        self.assertEqual(result['selected_update'], 200)
        self.assertEqual(result['selection_kind'], 'mixed_sim_v2_best')
        self.assertFalse(result['end_to_end_training_ready'])

    def test_complete_candidate_inventory_required(self):
        plan = self.plan(); values = [report(plan, n) for n in plan['candidate_updates']]
        for bad in (values[:-1], values + [values[0]], values + [report(plan, 400)]):
            with self.assertRaisesRegex(ValueError, 'candidate report set'):s.select_checkpoint(plan, bad)

    def test_row_missing_duplicate_and_identity_hash_rejected(self):
        plan = self.plan([100]); base = report(plan, 100)
        changes = (lambda r: r['rows'].pop(), lambda r: r['rows'].__setitem__(0, r['rows'][1]),
                   lambda r: r['rows'][0].__setitem__('sample_sha256', 'b'*64),
                   lambda r: r.__setitem__('manifest_sha256', 'c'*64))
        for change in changes:
            current = copy.deepcopy(base); change(current)
            with self.assertRaises(ValueError):s.select_checkpoint(plan, [current])

    def test_nonfinite_negative_and_boolean_loss_rejected(self):
        plan = self.plan([100])
        for value in (float('nan'), float('inf'), -1., True):
            current = report(plan, 100); current['rows'][0]['matcher_loss'] = value
            with self.assertRaises(ValueError):s.select_checkpoint(plan, [current])

    def test_batch_mean_semantics_or_field_rejected(self):
        plan = self.plan([100])
        for field, value in (('loss_semantics', 'physical_batch_mean/1'), ('matcher_batch_loss', 1.)):
            current = report(plan, 100); current['rows'][0][field] = value
            with self.assertRaises(ValueError):s.select_checkpoint(plan, [current])
        current = report(plan, 100); del current['rows'][0]['loss_components']
        with self.assertRaises(ValueError):s.select_checkpoint(plan, [current])

    def test_full_loss_components_checked(self):
        plan = self.plan([100]); current = report(plan, 100)
        current['rows'][0]['loss_components']['dustbin_nll'] = 10.
        with self.assertRaisesRegex(ValueError, 'unreduced components'):s.select_checkpoint(plan, [current])

    def test_native_microbatch_one_provenance(self):
        plan = self.plan([100]); current = report(plan, 100)
        for item in current['rows']:
            del item['loss_components']
            item.update(loss_implementation=s.NATIVE_MICROBATCH1, physical_microbatch=1)
        self.assertEqual(s.select_checkpoint(plan, [current])['selected_update'], 100)
        for size in (8, True):
            current['rows'][0]['physical_microbatch'] = size
            with self.assertRaisesRegex(ValueError, 'microbatch1 loss provenance'):
                s.select_checkpoint(plan, [current])

    def test_numeric_failure_blocks_but_no_candidate_is_kept(self):
        plan = self.plan([100]); current = report(plan, 100)
        item = next(r for r in current['rows'] if r['label'])
        item.update(layout20=False, candidate_coverage=False, has_candidate=False)
        result = s.summarize_report(plan, current)
        self.assertEqual(result['stages'][item['stage']]['strata'][item['generator']]['positives'], 1)
        self.assertLess(result['macro_layout'], 1.)
        item['numeric_valid'] = False
        with self.assertRaisesRegex(ValueError, 'numeric-invalid'):
            s.select_checkpoint(plan, [current])

    def test_cal_test_real_train_roles_rejected(self):
        plan = self.plan([100])
        for role in ('cal', 'test', 'train', 'real_select'):
            current = report(plan, 100); current['role'] = role
            with self.assertRaisesRegex(ValueError, 'SELECT only'):s.select_checkpoint(plan, [current])

    def test_unsuccessful_report_rejected(self):
        plan = self.plan([100]); current = report(plan, 100); current['returncode'] = 1
        with self.assertRaisesRegex(ValueError, 'successful actual'):s.select_checkpoint(plan, [current])

    def test_stage_generator_macro_not_pooled_rows(self):
        # Unequal generator populations must not dominate the stage average.
        for role in ('cal', 'select'):
            self.manifests[role]['entries'] += [row(role, 'v17_filtered', 'Gen2', False, i) for i in range(1, 10)]
        plan = self.plan([100]); current = report(plan, 100)
        for item in current['rows']:
            set_loss(item, 5. if item['stage'] == 'v17_filtered' and item['generator'] == 'Gen2' else 1.)
        metrics = s.summarize_report(plan, current)
        self.assertAlmostEqual(metrics['stages']['v17_filtered']['loss'], 2.)
        self.assertAlmostEqual(metrics['macro_loss'], 1.3)

    def test_layout_band_fixed_absolute_and_loss_cannot_override(self):
        # 20 positives per stratum: one strict_strip failure costs .005;
        # one failure in v18 Gen2 costs .00375.
        for role in ('cal', 'select'):
            self.manifests[role]['entries'] += [row(role, stage, generator, True, i)
                for stage in p.STAGE_WEIGHTS for generator in p.DEFAULT_GENERATORS[stage] for i in range(1, 20)]
        plan = self.plan(); values = [report(plan, 100, 3.), report(plan, 200, 2.), report(plan, 300, .1)]
        for entry in [r for r in values[1]['rows'] if r['label'] and r['stage'] == 'strict_straight' and r['generator'] == 'straight_strip'][:1]:
            entry['layout20'] = False
        for entry in [r for r in values[2]['rows'] if r['label'] and r['stage'] == 'v18' and r['generator'] == 'Gen2'][:2]:
            entry['layout20'] = False
        result = s.select_checkpoint(plan, values)
        self.assertEqual(result['eligible_updates'], [100, 200])
        self.assertEqual(result['selected_update'], 200)
        set_loss(values[0]['rows'][0], 2.)  # No effect on band membership.

    def test_equal_loss_prefers_higher_layout_within_band(self):
        for role in ('cal', 'select'):
            self.manifests[role]['entries'] += [row(role, 'strict_straight', 'straight_strip', True, i) for i in range(1, 20)]
        plan = self.plan([100, 200]); values = [report(plan, n) for n in plan['candidate_updates']]
        item = next(r for r in values[0]['rows'] if r['label'] and r['stage'] == 'strict_straight' and r['generator'] == 'straight_strip')
        item['layout20'] = False
        self.assertEqual(s.select_checkpoint(plan, values)['selected_update'], 200)

    def test_coverage_never_changes_selection(self):
        plan = self.plan(); values = [report(plan, n, n/100.) for n in plan['candidate_updates']]
        for current in values:
            for item in current['rows']:
                if item['label']:item['layout20'] = False
        expected = s.select_checkpoint(plan, values)['selected_update']
        for item in values[0]['rows']:item['candidate_coverage'] = False
        self.assertEqual(s.select_checkpoint(plan, values)['selected_update'], expected)

    def test_row_order_irrelevant(self):
        plan = self.plan([100]); current = report(plan, 100)
        before = s.summarize_report(plan, current)
        current['rows'].reverse(); after = s.summarize_report(plan, current)
        self.assertEqual(before['macro_loss'], after['macro_loss'])
        self.assertEqual(before['stages'], after['stages'])


if __name__ == '__main__':unittest.main()

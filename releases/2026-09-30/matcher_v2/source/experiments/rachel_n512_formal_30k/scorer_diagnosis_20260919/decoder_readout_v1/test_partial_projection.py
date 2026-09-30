"""Synthetic admission checks; these are not measured model results."""
from copy import deepcopy
import unittest
import report_projection as report
from test_report_projection import fixture_analysis


class PartialProjection(unittest.TestCase):
    def setUp(self):
        self.models = ['binary_patch', 'binary_stats', 'aggressive_binary_patch']
        self.data = fixture_analysis()
        self.data.update(status='model_complete', all_models_analyzed=False,
                         requested_models=self.models)
        self.data['models'].pop('threshold_scratch_fixed')
        for value in self.data['models'].values():
            value['exit_evidence'] = dict(kind='independently_observed_linux_exit',
                returncode=0, sha256='f'*64, parent_receipt_pending=True)

    def test_explicit_complete_subset_is_labeled_partial(self):
        out = report.project(self.data, source_name='analysis.json', source_sha256='f'*64,
                             completed_models=self.models)
        self.assertEqual(out['status'], 'model_complete')
        self.assertEqual(len(out['queries']['decoder_factor_models']['rows']), 3)
        self.assertEqual(out['total_planned_models'], 4)
        self.assertIn('3/4', out['queries']['decoder_factor_metrics']['source']['caveats'][0])

    def test_default_still_requires_four_models(self):
        with self.assertRaises(ValueError): report.validate(self.data)

    def test_extra_or_missing_model_never_slips_in(self):
        for selected in (self.models[:2], self.models+['threshold_scratch_fixed'], [], self.models*2):
            with self.assertRaises(ValueError): report.validate(self.data, completed_models=selected)

    def test_successful_exit_required(self):
        self.data['models']['binary_patch']['exit_evidence']['returncode'] = 1
        with self.assertRaises(ValueError): report.validate(self.data, completed_models=self.models)

    def test_full_population_rows_still_required(self):
        self.data['models']['binary_patch']['rows'].pop()
        with self.assertRaises(ValueError): report.validate(self.data, completed_models=self.models)

    def test_partial_cannot_claim_full_complete(self):
        for field, value in [('status', 'complete'), ('all_models_analyzed', True)]:
            bad = deepcopy(self.data); bad[field] = value
            with self.assertRaises(ValueError): report.validate(bad, completed_models=self.models)


if __name__ == '__main__': unittest.main()

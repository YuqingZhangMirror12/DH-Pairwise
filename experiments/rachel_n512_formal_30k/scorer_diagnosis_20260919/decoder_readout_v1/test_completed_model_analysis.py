"""Admission tests; mocked results are not experiments or published evidence.

Actual per-model record/SHA/membership admission remains covered by the full
2619-row fixture in test_factor_analysis. Here we test the new completion scope.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_factor_sweep as analysis


class CompletedModelAnalysis(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roles = self.root/'roles.json'; self.roles.write_text('{}')
        self.source = self.root/'source.json'; self.source.write_text('{}')
        self.sim = {}
        for model in analysis.MODELS:
            path = self.root/(model+'.json'); path.write_text('{}'); self.sim[model] = path
            folder = self.root/'full_development_01'/model; folder.mkdir(parents=True)
            (folder/'complete.json').write_text('{"status":"complete"}')
        self.real = {'dunhuang_cv': 'verified_elsewhere', 'turufan': 'verified_elsewhere'}

    def controller_complete(self):
        (self.root/'complete.json').write_text(json.dumps(dict(
            status='complete', pairs=10476, models=list(analysis.MODELS))))

    def collect(self, selected=None, **options):
        names = analysis.MODELS if selected is None else selected
        sim = {key: self.sim[key] for key in set(names) if key in self.sim}
        sim.update(options.get('extra_sim', {}))
        return analysis.collect_analysis(self.root, self.roles, self.source, sim,
            options.get('real', self.real), selected)

    def mock_verified(self):
        def load(root, model, roles, source, sim, real):
            return {'already_verified_full_population': model}, {'model': model, 'threshold': .3}, {}
        loaded = patch.object(analysis, 'load_verified', side_effect=load).start()
        grouped = patch.object(analysis, 'analyze_groups',
            side_effect=lambda groups, protocol, model, eq: dict(model=model, rows=[])).start()
        self.addCleanup(patch.stopall)
        return loaded, grouped

    def test_single_completed_model_does_not_require_or_claim_controller_complete(self):
        loaded, grouped = self.mock_verified()
        value = self.collect(['binary_patch'])
        self.assertEqual(value['status'], 'model_complete')
        self.assertEqual(set(value['models']), {'binary_patch'})
        self.assertFalse(value['all_models_analyzed'])
        self.assertFalse(value['controller_completion_observed'])
        self.assertIsNone(value['controller_complete_sha256'])
        self.assertEqual(loaded.call_count, 1); self.assertEqual(grouped.call_count, 1)
        self.assertEqual(value['models']['binary_patch']['sim_select_manifest_sha256'],
                         analysis.sha(self.sim['binary_patch']))

    def test_subset_never_loads_unrequested_model(self):
        loaded, _ = self.mock_verified()
        value = self.collect(['binary_patch', 'binary_stats'])
        self.assertEqual([call.args[1] for call in loaded.call_args_list], ['binary_patch', 'binary_stats'])
        self.assertEqual(value['status'], 'model_complete')

    def test_all_models_still_require_controller_terminal_receipt(self):
        loaded, _ = self.mock_verified()
        for names in (None, list(analysis.MODELS)):
            with self.assertRaisesRegex(ValueError, 'controller not complete'): self.collect(names)
        self.assertEqual(loaded.call_count, 0)
        self.controller_complete()
        value = self.collect()
        self.assertEqual(value['status'], 'complete'); self.assertTrue(value['all_models_analyzed'])
        self.assertTrue(value['controller_completion_observed']); self.assertEqual(loaded.call_count, 4)

    def test_controller_completion_does_not_make_subset_a_four_model_report(self):
        self.controller_complete(); self.mock_verified()
        value = self.collect(['binary_patch'])
        self.assertEqual(value['status'], 'model_complete')
        self.assertFalse(value['all_models_analyzed']); self.assertTrue(value['controller_completion_observed'])

    def test_failure_precedes_completed_file(self):
        self.controller_complete(); (self.root/'controller_failure.json').write_text('{}')
        loaded, _ = self.mock_verified()
        with self.assertRaisesRegex(ValueError, 'failure takes precedence'): self.collect(['binary_patch'])
        self.assertEqual(loaded.call_count, 0)

    def test_incomplete_model_cannot_be_admitted_by_subset_mode(self):
        with patch.object(analysis, 'load_verified', side_effect=ValueError('full development evidence incomplete')):
            with self.assertRaisesRegex(ValueError, 'incomplete'): self.collect(['binary_patch'])

    def test_bad_controller_receipt_not_silently_ignored_in_subset_mode(self):
        (self.root/'complete.json').write_text(json.dumps(dict(status='running', pairs=2619, models=['binary_patch'])))
        loaded, _ = self.mock_verified()
        with self.assertRaisesRegex(ValueError, 'receipt invalid'): self.collect(['binary_patch'])
        self.assertEqual(loaded.call_count, 0)

    def test_unknown_duplicate_empty_or_wrong_manifest_set_rejected(self):
        self.mock_verified()
        for names in ([], ['unknown'], ['binary_patch', 'binary_patch']):
            with self.assertRaises(ValueError): self.collect(names)
        with self.assertRaisesRegex(ValueError, 'exact model-specific'):
            self.collect(['binary_patch'], extra_sim={'binary_stats': self.sim['binary_stats']})
        with self.assertRaisesRegex(ValueError, 'both actual real'):
            self.collect(['binary_patch'], real={'dunhuang_cv': 'only_one'})

    def test_duplicate_or_incomplete_assignment_is_not_silently_overwritten(self):
        self.assertEqual(analysis._assignments(['model=a=b.json']), {'model': 'a=b.json'})
        for values in (['x=a','x=b'], ['x'], ['=a'], ['x=']):
            with self.assertRaises(ValueError): analysis._assignments(values)


if __name__ == '__main__': unittest.main()

"""Report-only continuation fixtures: no trained weights or real predictions."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from . import assemble as a
from . import export
from .test_assemble import fixture, selected, save
from .test_assemble_integration import materialize


def execution_record():
    return dict(schema='binary-microbatch-continuation/1', updates=200, exposures=6400,
                original_checkpoint_sha256='a' * 64, resume_origin_sha256='b' * 64)


class ExecutionProvenanceTests(unittest.TestCase):
    def models(self):
        models = selected('binary_patch')
        for model in models.values():
            model['execution_continuation'] = execution_record()
        return models

    def test_different_metadata_hashes_are_valid_continuation_not_new_weights(self):
        models = self.models()
        a.validate_selection(models, 'binary_patch')
        self.assertNotEqual(models['sim']['execution_continuation']['original_checkpoint_sha256'],
                            models['sim']['execution_continuation']['resume_origin_sha256'])

    def test_sim_real_must_share_continuation_even_if_selected_epochs_differ(self):
        models = self.models()
        models['real']['execution_continuation']['resume_origin_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'execution continuation differs'):
            a.validate_selection(models, 'binary_patch')

    def test_missing_one_selection_receipt_is_not_an_unmigrated_run(self):
        models = self.models()
        del models['real']['execution_continuation']
        with self.assertRaisesRegex(ValueError, 'execution continuation differs'):
            a.validate_selection(models, 'binary_patch')

    def test_invalid_continuation_counts_hash_or_schema_rejected(self):
        for key, value in [('updates', True), ('exposures', 6401), ('updates', -1),
                           ('original_checkpoint_sha256', 'not-a-sha'),
                           ('resume_origin_sha256', 'g' * 64), ('schema', 'unknown/1')]:
            with self.subTest(key=key, value=value):
                models = self.models()
                for model in models.values(): model['execution_continuation'][key] = value
                with self.assertRaisesRegex(ValueError, 'invalid execution continuation'):
                    a.validate_selection(models, 'binary_patch')

    def test_job_protocol_must_match_selected_continuation(self):
        with fixture(execution=execution_record()) as (root, values):
            task = next(iter(values)); path = root / task / 'protocol.json'
            protocol = a.read(path)
            protocol['execution_continuation']['resume_origin_sha256'] = 'c' * 64
            save(path, protocol)
            with patch.object(a, 'export_job', side_effect=lambda folder, _: deepcopy(values[Path(folder).name])):
                with self.assertRaisesRegex(ValueError, 'job source lineage differs'):
                    a.assemble_evaluation(root, 'binary_patch')

    def check_six_jobs(self, experiment):
        with fixture(experiment, execution=execution_record()) as (root, values):
            materialize(root, experiment, values)
            before = {str(p.relative_to(root)): a.sha(p) for p in root.rglob('*') if p.is_file()}
            result = a.assemble_evaluation(root, experiment)
            self.assertEqual(len(result['rows']), 40)
            for row in result['rows']:
                self.assertEqual(row['execution_continuation'], execution_record())
                source = values[row['selection_kind'] + '_' + row['split']]
                self.assertEqual(row['metrics'], source['groups'][row['population']][row['policy']])
            for job in result['jobs']:
                for case in job['cases']:
                    self.assertEqual(case['provenance']['execution_continuation'], execution_record())
            self.assertEqual(before, {str(p.relative_to(root)): a.sha(p) for p in root.rglob('*') if p.is_file()})
            # A resealed summary with conflicting execution provenance is still
            # invalid. Matching score/epoch/checkpoint alone must not hide it.
            task = 'sim_turufan'; path = root / task / 'summary.json'
            summary = a.read(path)
            summary['execution_continuation']['resume_origin_sha256'] = 'c' * 64
            save(path, summary)
            verified_path = root / (task + '_verified.json')
            verified = a.read(verified_path); verified['summary_sha256'] = a.sha(path)
            save(verified_path, verified)
            with self.assertRaisesRegex(ValueError, 'frozen execution continuation differs'):
                export.export_job(root / task, verified_path)

    def test_patch_continuation_through_six_actual_exporter_jobs(self):
        self.check_six_jobs('binary_patch')

    def test_stats_continuation_through_six_actual_exporter_jobs(self):
        self.check_six_jobs('binary_stats')

    def test_legacy_and_new_data_imports_do_not_invent_continuation(self):
        for experiment in a.EXPERIMENTS:
            models = selected(experiment)
            a.validate_selection(models, experiment)
            self.assertTrue(all('execution_continuation' not in model for model in models.values()))


if __name__ == '__main__':
    unittest.main()

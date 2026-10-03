"""Presentation-only regressions using the completed source05 probe evidence."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from .build_snapshot_unprotected import prepare, REVISION


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[4]
        base=root/'artifacts/curriculum_review_20260928'
        cls.artifact=json.loads((base/'unprotected_light4_rendered_01/rendered.json').read_text())
        cls.probe=json.loads((base/'unprotected_light4_probe_01/probe_complete.json').read_text())
        cls.data=json.loads((root/'artifacts/binary_scorer_20260927/review_app/src/data.json').read_text())

    def run_prepare(self, artifact=None):
        return prepare(deepcopy(artifact or self.artifact), self.probe, deepcopy(self.data), 'actual-source05-rendered.json')

    def test_complete_record_is_not_complete_full_dataset(self):
        data, receipt=self.run_prepare()
        self.assertEqual(data['buildStatus'], 'complete')
        self.assertEqual(data['curriculumReview']['revision'], REVISION)
        self.assertTrue(data['curriculumReview']['record_complete'])
        self.assertFalse(data['curriculumReview']['full_generation_started'])
        self.assertTrue(data['curriculumReview']['full_generation_authorized'])
        self.assertEqual(receipt['unique_pairs'], 22)
        self.assertEqual(receipt['display_slots'], 147)
        self.assertTrue(receipt['archive_queries_preserved'])
        for key in self.data['queries']:
            if not key.startswith('curriculum_'):
                self.assertEqual(self.data['queries'][key], data['queries'][key])

    def test_zero_pristine_is_valid_measurement(self):
        positives=[r for r in self.artifact['population'] if r['label']]
        self.assertEqual(min(r['pristine_fraction'] for r in positives), 0.)
        self.assertEqual(sum(r['pristine_fraction']<.25 for r in positives), 8)
        data, _=self.run_prepare()
        metric=next(r for r in data['queries']['curriculum_summary']['rows'] if r['version']=='v18' and r['metric'].startswith('双侧严格'))
        self.assertEqual(metric['min'], 0.)

    def test_old_protection_rejected(self):
        artifact=deepcopy(self.artifact)
        artifact['rows'][0]['detail']['spec']['pristine_protection_enabled']=True
        with self.assertRaisesRegex(ValueError, 'stale protected'):self.run_prepare(artifact)

    def test_actual_depth_above4_rejected(self):
        artifact=deepcopy(self.artifact)
        artifact['rows'][0]['detail']['background']['a']['applied_max_depth_px']=4.01
        with self.assertRaisesRegex(ValueError, 'light depth'):self.run_prepare(artifact)

    def test_omitted_sample_is_not_full_probe(self):
        artifact=deepcopy(self.artifact);artifact['rows'].pop()
        with self.assertRaisesRegex(ValueError, 'reconcile'):self.run_prepare(artifact)

    def test_old_revision_rejected(self):
        artifact=deepcopy(self.artifact);artifact['protocol']['revision']='curriculum-v17p5-v18-review/3-pristine25'
        with self.assertRaisesRegex(ValueError, 'source05'):self.run_prepare(artifact)


if __name__=='__main__':unittest.main()

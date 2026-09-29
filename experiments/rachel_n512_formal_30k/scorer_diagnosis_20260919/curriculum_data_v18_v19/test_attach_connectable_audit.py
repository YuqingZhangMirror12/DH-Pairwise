"""Report adapter gates; no generation, inference, or report mutations."""
from copy import deepcopy
import unittest

from .attach_connectable_audit import attach


class AttachAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(id=v, version=v, actual_pixels_checked=True,
                          original20_pass=True, final30_pass=True,
                          eligible=True) for v in ('v17.5', 'v18')]
        self.summary = dict(status='complete',
            contract='curriculum-original20-final30-light4/2-common-arc',
            actual_positive_pixels_audited=2, rows_sha256='fixture',
            versions={v: dict(positive=1, original20_excluded=0,
                extra_final30_excluded=0, eligible=1, recipes={})
                for v in ('v17.5', 'v18')})
        self.data = dict(id='report:1b87f48a-d200-4451-96aa-32b1bdf5fcaa',
            title='User title', theme='User theme',
            curriculumReview=dict(revision='historical-r5'),
            queries=dict(curriculum_cases=dict(rows=[dict(id='v17.5', image='original-image')]),
                         original_archive=dict(rows=[dict(value=7)])))

    def test_preserves_input_identity_images_and_queries(self):
        before = deepcopy(self.data)
        updated, receipt = attach(self.data, self.summary, self.rows)
        self.assertEqual(self.data, before)
        for key in ('id', 'title', 'theme', 'curriculumReview'):
            self.assertEqual(updated[key], before[key])
        for key, value in before['queries'].items():
            self.assertEqual(updated['queries'][key], value)
        self.assertTrue(receipt['original_queries_unchanged'])
        self.assertFalse(updated['curriculumQualification']['new_per_type10_complete'])
        self.assertEqual(updated['curriculumQualification']['new_samples_generated'], 0)

    def test_rejects_incomplete_or_old_contract(self):
        for field, value in [('status', 'running'), ('contract', 'old-v1')]:
            bad = deepcopy(self.summary); bad[field] = value
            with self.assertRaises(ValueError): attach(self.data, bad, self.rows)

    def test_rejects_duplicate_or_unchecked_pixels(self):
        bad = deepcopy(self.rows); bad[1]['id'] = bad[0]['id']
        with self.assertRaises(ValueError): attach(self.data, self.summary, bad)
        bad = deepcopy(self.rows); bad[0]['actual_pixels_checked'] = False
        with self.assertRaises(ValueError): attach(self.data, self.summary, bad)

    def test_rejects_summary_mismatch(self):
        bad = deepcopy(self.summary); bad['versions']['v18']['eligible'] = 0
        with self.assertRaises(ValueError): attach(self.data, bad, self.rows)

    def test_requires_audit_for_every_displayed_image(self):
        bad = deepcopy(self.data); bad['queries']['curriculum_cases']['rows'][0]['id'] = 'unknown'
        with self.assertRaises(ValueError): attach(bad, self.summary, self.rows)

    def test_wrong_report_rejected(self):
        bad = deepcopy(self.data); bad['id'] = 'another-report'
        with self.assertRaises(ValueError): attach(bad, self.summary, self.rows)


if __name__ == '__main__':
    unittest.main()

"""New tests for the explicitly authorized manifest-only deduplication."""
import copy
import json
import os
from pathlib import Path
import unittest
from . import heldout_reduced_release as m
from .test_reduced_release import population


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ['DEDUP_AUDIT_PATH'])
        assert m.h.sha(path) == m.DEDUP_AUDIT_SHA
        cls.audit = json.loads(path.read_text())

    def rows(self, role):
        rows = population(role)
        used = set()
        for group in self.audit['folds'][role]['duplicate_groups']:
            for e in group:
                i = next(i for i, r in enumerate(rows) if i not in used and
                         all(r[k] == e[k] for k in ('stage', 'generator', 'label')))
                rows[i] = copy.deepcopy(e); used.add(i)
        return rows

    def test_cal_actual_1587_and_exact_exclusions(self):
        self.check_role('cal')

    def test_select_actual_1596_and_exact_exclusions(self):
        self.check_role('select')

    def check_role(self, role):
        rows = self.rows(role); m.validate_population(rows, role, allow_exact_duplicates=True)
        original = copy.deepcopy(rows)
        result, receipt = m.authorized_deduplication(rows, role, self.audit)
        self.assertEqual(rows, original)
        self.assertEqual(len(result), m.FINAL_COUNTS[role][0])
        self.assertEqual({e['pair_id'] for e in rows} - {e['pair_id'] for e in result},
                         set(self.audit['proposed_manifest_only_deduplication'][role]['excluded_pair_ids']))
        self.assertTrue(receipt['no_files_deleted'])
        for group in receipt['groups']:
            self.assertLess(group['retained_pair_id'], min(group['excluded_pair_ids']))

    def test_unregistered_duplicate_rejected(self):
        rows = self.rows('cal')
        clean = [e for e in rows if 'recipe' not in e]
        clean[0]['model_tensors_sha256'] = clean[1]['model_tensors_sha256']
        with self.assertRaisesRegex(ValueError, 'unapproved duplicate'):
            m.authorized_deduplication(rows, 'cal', self.audit)

    def test_changed_member_bytes_rejected(self):
        rows = self.rows('cal'); next(e for e in rows if 'recipe' in e)['sample_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'identity/content'):
            m.authorized_deduplication(rows, 'cal', self.audit)

    def test_conflicting_supervision_never_deduplicated(self):
        audit = copy.deepcopy(self.audit); e = audit['folds']['cal']['duplicate_groups'][0][0]
        rows = self.rows('cal'); next(r for r in rows if r['pair_id'] == e['pair_id'])['supervised_tensors_sha256'] = 'conflict'
        e['supervised_tensors_sha256'] = 'conflict'
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate supervision'):
            m.authorized_deduplication(rows, 'cal', audit)

    def test_different_retention_policy_rejected(self):
        audit = copy.deepcopy(self.audit)
        audit['proposed_manifest_only_deduplication']['cal']['excluded_pair_ids'][0] = 'not-authorized'
        with self.assertRaisesRegex(ValueError, 'excluded IDs'):
            m.authorized_deduplication(self.rows('cal'), 'cal', audit)


if __name__ == '__main__': unittest.main()

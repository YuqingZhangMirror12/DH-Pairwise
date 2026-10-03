"""Small pure-CPU normalization/admission tests; no generation or model use."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import heldout_release as r
from . import heldout_straight as h
from .protocol import ROW_FIELDS


class StrictReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.donors = [dict(path='/source/left.png', file_sha256='a'*64,
                            fragment_token='Gen2/original-fragment', source_family='family-a'),
                       dict(path='/source/right.png', file_sha256='b'*64,
                            fragment_token='Gen3/original-fragment', source_family='family-b')]
        self.plan = dict(role='cal', dataset_id='new-strict-dataset',
                         masterseeds={'cal': 2026100243, 'select': 2026100244}, source_rows=self.donors)
        self.entry = dict(split='cal', pair_id='new/cal/pair', id='M_pos_cal_00000', label=True,
                          sample_sha256='c'*64, pre_damage_pair_sha256='d'*64,
                          attempted_donor_references=copy.deepcopy(self.donors), tries=3,
                          meta={'a': {'source_family': 'family-a'}})
        self.tokens = {'a': 'native/no-seed/a', 'b': 'native/no-seed/b'}

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True))
        return {'path': str(path), 'sha256': h.file_sha(path)}

    def test_exact_protocol_projection_and_keep_native_provenance(self):
        value = r.normalize_identities(self.plan, self.entry, self.tokens)
        self.assertEqual(ROW_FIELDS, set(value) & ROW_FIELDS)
        self.assertEqual(('strict_straight', 'straight_strip'), (value['stage'], value['generator']))
        self.assertEqual(['family-a'], value['parent_ids'])
        self.assertEqual(['family-a', 'family-b'], value['donor_parent_ids'])
        self.assertEqual([], value['donor_base_pair_ids'])
        self.assertEqual(self.tokens, value['native_fragment_tokens'])
        self.assertEqual(['strict-cut-sha256/'+'d'*64], value['base_pair_ids'])
        self.assertTrue(all('new-strict-dataset/cal/new/cal/pair/' in x for x in value['fragment_ids']))
        self.assertEqual('native/no-seed/a', self.tokens['a'])

    def test_attempted_not_only_accepted_donors_are_disclosed(self):
        value = r.normalize_identities(self.plan, self.entry, self.tokens)
        self.assertIn('family-b', value['donor_parent_ids'])
        self.assertNotIn('family-b', value['parent_ids'])
        self.assertEqual([d['fragment_token'] for d in self.donors], value['donor_fragment_ids'])

    def test_reject_foreign_or_untracked_accepted_donor(self):
        bad = copy.deepcopy(self.entry)
        bad['attempted_donor_references'][0]['file_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'foreign attempted donor'):
            r.normalize_identities(self.plan, bad, self.tokens)
        bad = copy.deepcopy(self.entry); bad['meta']['a']['source_family'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'absent from attempted lineage'):
            r.normalize_identities(self.plan, bad, self.tokens)

    def test_procedural_parent_is_explicit_role_seed_not_fake_manuscript(self):
        pure = copy.deepcopy(self.entry); pure.update(meta={'base': 'torn_strip'}, attempted_donor_references=[])
        first = r.normalize_identities(self.plan, pure, self.tokens)
        self.assertEqual([], first['donor_parent_ids'])
        self.assertEqual(1, len(first['synthetic_parent_ids']))
        self.assertTrue(first['parent_ids'][0].startswith('procedural-strip/cal/2026100243/'))
        next_plan = dict(self.plan, role='select')
        next_entry = dict(pure, split='select', pair_id='new/select/pair')
        second = r.normalize_identities(next_plan, next_entry, self.tokens)
        self.assertFalse(set(first['parent_ids']) & set(second['parent_ids']))
        self.assertFalse(set(first['fragment_ids']) & set(second['fragment_ids']))

    def test_negative_does_not_invent_common_seam_or_parent(self):
        negative = dict(self.entry, label=False, meta={'base': 'strip', 'a': {}, 'b': {}},
                        attempted_donor_references=[], pre_damage_pair_sha256=None)
        value = r.normalize_identities(self.plan, negative, self.tokens)
        self.assertEqual(2, len(value['parent_ids']))
        self.assertTrue(value['base_pair_ids'][0].startswith('procedural-negative-pair/'))
        self.assertFalse(any(x.startswith('strict-cut-sha256/') for x in value['base_pair_ids']))

    def test_reject_unlabeled_wrong_role_or_missing_positive_base(self):
        for change in ({'label': 1}, {'split': 'test'}, {'pre_damage_pair_sha256': None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                r.normalize_identities(self.plan, dict(self.entry, **change), self.tokens)
        with self.assertRaisesRegex(ValueError, 'native fragment tokens'):
            r.normalize_identities(self.plan, self.entry, {'a': 'token'})

    def test_parent_and_all_retained_variant_bytes_are_bound(self):
        image = self.root / 'image.png'; image.write_bytes(b'original-parent-rgba')
        variant = self.root / 'variant.png'; variant.write_bytes(b'original-sibling-rgba')
        value = {'family': 'family-a', 'parent_image': {'path': str(image), 'file_sha256': h.file_sha(image),
                 'has_embedded_original_alpha': True, 'alpha_sha256': '1'*64},
                 'allowed_same_family_variants': [{'path': str(variant), 'file_sha256': h.file_sha(variant)}]}
        parent_plan = {'folds': {'cal': [value], 'select': []}}
        result = r._check_parent_bytes(parent_plan, {'family-a'})
        self.assertEqual(value, result['family-a'])
        result['family-a']['family'] = 'changed-copy'
        self.assertEqual('family-a', value['family'])
        variant.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bound file changed'):
            r._check_parent_bytes(parent_plan, {'family-a'})

    def test_receipt_hash_changed_rejected(self):
        receipt = self.write('receipt.json', {'status': 'passed'})
        Path(receipt['path']).write_text('{"status":"different"}')
        with self.assertRaisesRegex(ValueError, 'bound file changed'): r._read(receipt)

    def test_native_tuple_intervals_equal_exact_persisted_json_not_changed_values(self):
        native = {'nick_intervals': [(1.0, 3.0), (8.0, 9.0)], 'correspondence_count': 12}
        persisted = json.loads(json.dumps(native))
        self.assertTrue(r._audit_equal(native, persisted))
        persisted['nick_intervals'][1][1] = 9.01
        self.assertFalse(r._audit_equal(native, persisted))
        with self.assertRaises(ValueError): r._audit_equal({'bad': float('nan')}, {'bad': None})

    def test_no_pilot_or_mixed_completion_is_accepted(self):
        for status, rows in [('pilot_passed', 12), ('complete_mixed6400', 6400)]:
            ref = self.write('complete.json', dict(status=status, rows=rows, full_mixed6400_complete=False))
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, 'strict640 completion'):
                r.normalize_strict_release(ref['path'], ref['sha256'], 'cal', load_sample=None,
                    loader_source_receipt={}, target_builder_path='/unused', expected_target_builder_sha='0'*64)

    def test_actual_nonzero_return_cannot_be_admitted(self):
        ret = self.write('return.json', dict(returncode=1, command=['cpu-worker'], source_sha256='f'*64))
        launch = self.write('launch.json', dict(command=['cpu-worker'], controller_sha256='f'*64))
        ref = self.write('complete.json', dict(status='complete_strict640_only', rows=640,
            full_mixed6400_complete=False, actual_process_return=ret, controller=launch))
        with self.assertRaisesRegex(ValueError, 'actual launch/return identity'):
            r.normalize_strict_release(ref['path'], ref['sha256'], 'cal', load_sample=None,
                loader_source_receipt={}, target_builder_path='/unused', expected_target_builder_sha='0'*64)


if __name__ == '__main__': unittest.main()

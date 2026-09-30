import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from .source_audit import audit_sources, mask_identity


class SourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.entries, self.paths = {}, {}
        for i, split in enumerate(('train', 'select', 'test')):
            mask = np.zeros((800, 800), np.uint8)
            mask[100:200 + i, 200:300 + 2 * i] = 255
            Image.fromarray(mask).save(self.root / f'{split}.png')
            frag = dict(fragment_token=f'rachel/g/{i}/0', split_unit_id=f'family-{i}', model_mask_path=f'{split}.png')
            self.entries[split] = [dict(pair_id=split, source_row=dict(fragment_a=frag, fragment_b=frag.copy()))]
            self.paths[split] = self.root / f'{split}.json'

    def run_audit(self, **kwargs):
        for split, entries in self.entries.items():
            self.paths[split].write_text(json.dumps(dict(entries=entries)))
        return audit_sources(self.paths, self.root, **kwargs)

    def test_distinct_pools_pass_and_dedup_references(self):
        result = self.run_audit()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['counts']['train']['unique_original_masks'], 1)
        self.assertEqual(result['counts']['train']['original_mask_references'], 2)
        self.assertTrue(all(n == 0 for n in result['collision_counts'].values()))
        self.assertFalse(result['model_inference'])

    def test_same_family_different_pixels_is_leakage(self):
        for frag in self.entries['select'][0]['source_row'].values():
            frag['split_unit_id'] = 'family-0'
        result = self.run_audit()
        self.assertEqual(result['status'], 'not_admitted')
        self.assertEqual(result['collision_counts']['source_family'], 1)

    def test_copied_or_relocated_mask_is_leakage(self):
        with Image.open(self.root / 'train.png') as image:
            relocated = np.roll(np.asarray(image), 10, axis=0)
        Image.fromarray(relocated).save(self.root / 'select.png')
        result = self.run_audit()
        self.assertEqual(result['collision_counts']['pixel_sha256'], 0)
        self.assertEqual(result['collision_counts']['crop_pixel_sha256'], 1)
        self.assertEqual(result['status'], 'not_admitted')

    def test_union_exclusion_is_recorded_not_hidden(self):
        union = dict(fragment_token='rachel-union/a+b', model_mask_path='absent.png', split_unit_id='x')
        self.entries['train'].append(dict(source_row=dict(fragment_a=union, fragment_b=union)))
        result = self.run_audit()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['counts']['train']['excluded_reference_counts']['derived_union_not_original_mask'], 2)

    def test_missing_metadata_and_files_fail_closed(self):
        self.entries['train'][0]['source_row']['fragment_a'] = dict(fragment_token='rachel/a')
        self.entries['select'][0]['source_row']['fragment_a']['model_mask_path'] = 'missing.png'
        result = self.run_audit()
        self.assertEqual(result['status'], 'not_admitted')
        self.assertTrue({'missing_source_family', 'missing_or_outside_source_root'} <= {e['reason'] for e in result['errors']})

    def test_actual_derived_token_forms_excluded_but_family_still_checked(self):
        group = dict(fragment_token='gen5-group-abcdef', split_unit_id='family-1')
        union = dict(fragment_token='rachel-union-abcdef/a', split_unit_id='family-0', model_mask_path='missing.png')
        self.entries['train'].append(dict(source_row=dict(fragment_a=group, fragment_b=union)))
        result = self.run_audit()
        self.assertEqual(result['counts']['train']['excluded_reference_counts'],
                         dict(derived_gen5_group_no_original_mask=1, derived_union_not_original_mask=1))
        self.assertEqual(result['collision_counts']['all_v14_declared_source_families'], 1)
        self.assertEqual(result['status'], 'not_admitted')

    def test_manifest_only_is_not_pixel_admission(self):
        result = self.run_audit(hash_pixels=False)
        self.assertEqual(result['status'], 'not_admitted')
        self.assertFalse(result['pixel_hashes_checked'])

    def test_blank_mask_rejected(self):
        Image.fromarray(np.zeros((800, 800), np.uint8)).save(self.root / 'train.png')
        with self.assertRaisesRegex(ValueError, 'invalid original mask'):
            mask_identity(self.root / 'train.png')


if __name__ == '__main__':
    unittest.main()

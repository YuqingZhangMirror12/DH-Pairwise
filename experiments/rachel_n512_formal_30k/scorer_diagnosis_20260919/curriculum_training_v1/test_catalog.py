from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from .catalog import (InspectedSample, MATCHER_INPUTS, TARGETS, tensor_digest,
                      legacy_numerical_digest, inspect_sample, deduplicate_basic)
from .exposure import digest
from .test_exposure import samples


def fixture():
    return SimpleNamespace(pair_id='pair', label=np.float32(1.),
        mask_a=np.ones((1, 4, 4), dtype=np.float32), mask_b=np.ones((1, 4, 4), dtype=np.float32),
        points_rc_a=np.array([[0., 1.], [1., 2.]], dtype=np.float32),
        points_rc_b=np.array([[2., 1.], [3., 2.]], dtype=np.float32),
        contour_valid_a=np.ones(2, dtype=np.bool_), contour_valid_b=np.ones(2, dtype=np.bool_),
        target_a=np.array([0, 1], dtype=np.int64), target_b=np.array([0, 1], dtype=np.int64),
        translation_a_to_b_rc=np.array([2., 0.], dtype=np.float32), translation_valid=np.bool_(True))


class CatalogTests(unittest.TestCase):
    def test_input_fingerprint_does_not_include_gt(self):
        sample = fixture(); before = tensor_digest(sample, MATCHER_INPUTS)
        target = tensor_digest(sample, TARGETS); legacy = legacy_numerical_digest(sample)
        sample.translation_a_to_b_rc += 3
        self.assertEqual(before, tensor_digest(sample, MATCHER_INPUTS))
        self.assertNotEqual(target, tensor_digest(sample, TARGETS))
        self.assertNotEqual(legacy, legacy_numerical_digest(sample))

    def test_validity_is_input_and_changes_hash(self):
        sample = fixture(); before = tensor_digest(sample, MATCHER_INPUTS)
        sample.contour_valid_a[1] = False
        self.assertNotEqual(before, tensor_digest(sample, MATCHER_INPUTS))

    def test_pair_names_not_in_numeric_inputs(self):
        sample = fixture(); before = tensor_digest(sample, MATCHER_INPUTS)
        sample.pair_id = 'different-name'
        self.assertEqual(before, tensor_digest(sample, MATCHER_INPUTS))

    def test_shape_is_bound(self):
        sample = fixture(); before = tensor_digest(sample, MATCHER_INPUTS)
        sample.mask_a = sample.mask_a.reshape(1, 2, 8)
        self.assertNotEqual(before, tensor_digest(sample, MATCHER_INPUTS))

    def test_nonfinite_input_is_rejected(self):
        sample = fixture(); sample.points_rc_a[0, 0] = np.nan
        with self.assertRaises(ValueError):
            tensor_digest(sample, MATCHER_INPUTS)

    def test_actual_file_and_legacy_receipt_bound_before_loader_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sample.npz'; path.write_bytes(b'synthetic loader fixture')
            sha = hashlib.sha256(path.read_bytes()).hexdigest(); sample = fixture()
            row = dict(pair_id='pair', label=True, source_root='/data', source_pair_id='original',
                       source_row=dict(split='train'), sample_path=str(path), sample_sha256=sha)
            audit = dict(id='pair', status='passed', sample_sha256=sha,
                         model_input_sha256=legacy_numerical_digest(sample))
            result = inspect_sample('v18', row, audit, lambda p: (sample, {}))
            self.assertEqual(result.ref.source_base_key, '/data::original')
            self.assertEqual(result.ref.model_input_sha256, tensor_digest(sample, MATCHER_INPUTS))
            with self.assertRaisesRegex(ValueError, 'TRAIN'):
                inspect_sample('v18', dict(row, source_row=dict(split='test')), audit, lambda p: (sample, {}))
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'sample file'):
                inspect_sample('v18', row, audit, lambda p: (sample, {}))

    def partition(self):
        rows = [InspectedSample(r, digest(['targets', r.label]), digest(['legacy', r.pair_id])) for r in samples()]
        return rows[:10], rows[10:]

    def test_basic_exact_duplicates_removed_without_editing_hard(self):
        basic, hard = self.partition()
        duplicate = replace(basic[0], ref=replace(basic[0].ref, pair_id='z-duplicate',
                                                source_base_key='other-original'))
        result, receipt = deduplicate_basic(basic + [duplicate], hard)
        self.assertEqual(receipt['basic_after'], 10)
        self.assertEqual(len(result), len(samples()))
        self.assertEqual(len(receipt['discarded_exact_duplicates']), 1)
        self.assertEqual(len(hard), receipt['hard_rows_unchanged'])

    def test_conflicting_gt_does_not_evade_input_dedup(self):
        basic, hard = self.partition()
        duplicate = replace(basic[0], target_sha256=digest('conflicting'),
                            ref=replace(basic[0].ref, pair_id='z-duplicate'))
        with self.assertRaisesRegex(ValueError, 'conflicting supervision'):
            deduplicate_basic(basic + [duplicate], hard)

    def test_hard_input_duplicate_is_not_silently_repaired(self):
        basic, hard = self.partition()
        with self.assertRaisesRegex(ValueError, 'hard release'):
            deduplicate_basic(basic, hard + [hard[0]])

    def test_stale_basic_source_exclusion_rejected(self):
        basic, hard = self.partition()
        basic[0] = replace(basic[0], ref=replace(basic[0].ref, source_base_key=hard[0].ref.source_base_key))
        with self.assertRaisesRegex(ValueError, 'did not exclude'):
            deduplicate_basic(basic, hard)


if __name__ == '__main__':
    unittest.main()

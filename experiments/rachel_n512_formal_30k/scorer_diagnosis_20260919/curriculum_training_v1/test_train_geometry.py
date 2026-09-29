"""Synthetic TRAIN protocol and actual inherited-edge helper checks, CPU only."""
import copy
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import train_geometry as api
from .checkpoint_io import file_sha, write_json
from .exposure import STAGES, canonical_catalog, digest
from .test_exposure import samples


class FakeDataset:
    def __init__(self, sample):
        self.sample = sample
        self.entries = [dict(pair_id='explicit-synthetic', label=True, recipe='wave',
            sample_sha256=digest('sample'), source_base_key='synthetic::source',
            actual_matcher_input_sha256=digest('input'), effective_training_target_sha256=digest('target'))]

    def __getitem__(self, index):
        return self.sample, dict(synthetic_fixture=True), self.entries[index]


class TrainGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ['CURRICULUM_BASELINE_SOURCE'])
        cls.calibration = importlib.import_module(api.BASE + 'calibrate_geometry')
        cls.legacy_fixture = importlib.import_module(api.BASE + 'test_calibration')

    def setUp(self):
        torch.set_num_threads(1); self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        catalog = [asdict(row) for row in canonical_catalog(samples((1, 1, 1)))]
        self.admission = dict(status='passed', schema='curriculum-data-admission/1', gpu_used=False,
            catalog=catalog, catalog_sha256=digest(catalog))
        self.path = self.root / 'admission.json'; write_json(self.path, self.admission)
        self.dataset = FakeDataset(self.legacy_fixture.sample())

    def measurements(self):
        pairs, blocks = [], []
        for i, entry in enumerate(self.admission['catalog']):
            if not entry['label']:
                continue
            recipe = 'clean' if entry['stage'] == 'v17_filtered' else 'wave'
            row = dict(catalog_index=i, stage=entry['stage'], pair_id=entry['pair_id'], recipe=recipe,
                inherited_edges=120)
            # Repeat a single synthetic measured edge solely to test the
            # existing >=100 calibration guard and arithmetic; NOT real data.
            edge = self.calibration.inherited_edges(self.legacy_fixture.sample(), recipe)
            block = np.repeat(edge, 120, axis=0)
            if recipe == 'clean': block[:, 4] = .5; block[:, 6] = .5
            pairs.append(row); blocks.append(block)
        return pairs, blocks

    def test_catalog_view_does_not_invent_training_budget(self):
        _, view = api.catalog_view(self.path, file_sha(self.path))
        self.assertEqual(len(view.catalog), 6); self.assertFalse(hasattr(view, 'total_updates'))

    def test_changed_admission_not_loaded(self):
        old = file_sha(self.path); self.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'admission changed'): api.catalog_view(self.path, old)

    def test_all_three_tiers_and_both_labels_required(self):
        for omit in (lambda row: row['stage'] == 'v18', lambda row: row['stage'] == 'v18' and row['label']):
            value = copy.deepcopy(self.admission); value['catalog'] = [r for r in value['catalog'] if not omit(r)]
            value['catalog_sha256'] = digest(value['catalog']); write_json(self.path, value, replace=True)
            with self.assertRaises(ValueError): api.catalog_view(self.path, file_sha(self.path))

    def test_reordered_catalog_not_silently_remapped(self):
        value = copy.deepcopy(self.admission); value['catalog'].reverse(); write_json(self.path, value, replace=True)
        with self.assertRaisesRegex(ValueError, 'catalog identity'): api.catalog_view(self.path, file_sha(self.path))

    def test_inspect_uses_actual_inherited_targets_not_proximity(self):
        pair, edges = api.inspect_index(0, self.dataset, self.calibration, 'v18')
        self.assertEqual(pair['stage'], 'v18'); self.assertEqual(len(edges), 1)
        np.testing.assert_allclose(edges[0, 4:7], [5., 0., 5.])
        self.assertEqual(pair['precise_anchor_edges'], 0)

    def test_negative_pair_cannot_calibrate_geometry(self):
        self.dataset.entries[0]['label'] = False
        with self.assertRaisesRegex(ValueError, 'positive'): api.inspect_index(0, self.dataset, self.calibration, 'v18')

    def test_missing_reciprocal_target_not_replaced_by_nearest_edge(self):
        self.dataset.sample.target_b[4] = -2
        with self.assertRaisesRegex(ValueError, 'nonreciprocal'): api.inspect_index(0, self.dataset, self.calibration, 'v18')

    def test_exact_original_derive_formula_used_on_combined_pool(self):
        pairs, blocks = self.measurements()
        edges, indices, strata, parameters = api.summarize_measurements(pairs, blocks, self.calibration)
        self.assertEqual(edges.shape, (360, 11)); self.assertEqual(len(indices), 360)
        self.assertEqual(parameters, self.calibration.derive_parameters(strata))
        self.assertEqual(parameters['damage_normal_upper_px'], 5.)
        self.assertEqual(strata['stage:v18']['pair_count'], 1)

    def test_equal_pair_weight_not_extra_weight_for_longer_seam(self):
        pairs, blocks = self.measurements()
        blocks[1] = np.repeat(blocks[1], 4, axis=0); pairs[1]['inherited_edges'] *= 4
        blocks[2][:, 4] = 15.
        _, _, strata, _ = api.summarize_measurements(pairs, blocks, self.calibration)
        self.assertAlmostEqual(strata['corroded']['pair_weighted']['normal_px']['mean'], 10.)

    def test_missing_clean_support_does_not_fallback_to_real(self):
        pairs, blocks = self.measurements(); pairs[0]['recipe'] = 'wave'; blocks[0][:, -1] = 0
        with self.assertRaisesRegex(ValueError, 'no validation fallback'):
            api.summarize_measurements(pairs, blocks, self.calibration)

    def test_short_reliable_support_rejected_without_lowering_guard(self):
        pairs, blocks = self.measurements(); pairs[0]['inherited_edges'] = 2; blocks[0] = blocks[0][:2]
        with self.assertRaisesRegex(ValueError, 'insufficient reliable'):
            api.summarize_measurements(pairs, blocks, self.calibration)

    def test_duplicate_missing_or_forged_rows_not_completed(self):
        pairs, blocks = self.measurements()
        for changed in (pairs[:-1], [pairs[0], pairs[0], pairs[2]]):
            with self.subTest(rows=changed), self.assertRaisesRegex(ValueError, 'every admitted positive'):
                api.save_calibration(self.root / 'out', self.path, self.admission, changed,
                                     blocks, self.calibration, {}, 0.)
        self.assertFalse((self.root / 'out').exists())

    def test_artifacts_bind_input_and_keep_pose16_label20_unchanged(self):
        pairs, blocks = self.measurements(); out = self.root / 'out'
        record = api.save_calibration(out, self.path, self.admission, pairs, blocks, self.calibration, {}, 0.)
        self.assertEqual(record['curriculum_data_admission_sha256'], file_sha(self.path))
        self.assertEqual(record['fixed_pose_diameter_px'], 16.); self.assertEqual(record['layout_correctness_label_px'], 20.)
        for key in ('real_used', 'test_used', 'cal_used', 'select_used', 'predictions_used', 'gpu_used', 'pose_threshold_fitted'):
            self.assertFalse(record[key])
        for name, sha in record['artifacts'].items(): self.assertEqual(file_sha(out / name), sha)
        with np.load(out / 'inherited_edge_audit.npz', allow_pickle=False) as data:
            self.assertEqual(data['values'].shape, (360, 11)); self.assertTrue(data['source_support_known'].all())

    def test_existing_outputs_not_overwritten(self):
        pairs, blocks = self.measurements(); out = self.root / 'out'; out.mkdir()
        with self.assertRaisesRegex(ValueError, 'preserve earlier'):
            api.save_calibration(out, self.path, self.admission, pairs, blocks, self.calibration, {}, 0.)

    def test_resource_limits_and_cuda_visibility_precede_any_loading(self):
        for workers in (0, 5, True):
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), self.subTest(workers=workers), self.assertRaisesRegex(ValueError, 'four'):
                api.run('missing', 'missing', 'missing', 'missing', self.root / 'out', workers)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'), self.assertRaisesRegex(ValueError, 'hide CUDA'):
            api.run('missing', 'missing', 'missing', 'missing', self.root / 'out')


if __name__ == '__main__':
    unittest.main()

"""CPU contracts for display conversion. Fixtures are NOT experiment results."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .frozen import sha
from .view_data import matrix_view, mask_runs, bounded_links, build_case, shared_scales, load_evaluation, write_review
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.consensus_head import ConsensusEvidenceHead
from .snapshot import snapshot_prediction
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.diagnostics import write_snapshot
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.model import S7Consensus
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_evidence import fixture


class MatrixDisplayTests(unittest.TestCase):
    def test_all_matrix_entries_accounted_with_unequal_bins(self):
        matrix = np.arange(77).reshape(7, 11)/100
        view = matrix_view(matrix, bins=4)
        self.assertEqual(sum(r['cells'] for r in view['rows']), 77)
        self.assertAlmostEqual(sum(r['total'] for r in view['rows']), matrix.sum())
        self.assertAlmostEqual(sum(r['mean']*r['cells'] for r in view['rows']), matrix.sum())
        self.assertEqual(len(view['rows']), 16)
        self.assertEqual(view['normalization'], 'none')
        self.assertEqual(view['rows'][0]['a_start'], 0)
        self.assertEqual(view['rows'][-1]['b_end_exclusive'], 11)

    def test_no_fake_zero_for_nan_but_structural_zeros_retained(self):
        matrix = np.zeros((2, 3)); matrix[0, 1] = float('nan')
        self.assertFalse(matrix_view(matrix)['available'])
        self.assertIsNone(matrix_view(matrix)['total'])
        zero = matrix_view(np.zeros((2, 3)))
        self.assertTrue(zero['available'])
        self.assertEqual(len(zero['rows']), 6)
        self.assertTrue(all(r['mean'] == 0 for r in zero['rows']))

    def test_link_sign_rank_and_omitted_mass(self):
        a = np.array([[2., 5.], [3., 6.]])
        b = np.array([[12., 25.], [13., 26.]])
        weights = np.array([[.4, .1], [.0, .4]])
        view = bounded_links(weights, a, b, [7, 8], [4, 5], [10, 20], limit=1)
        row = view['rows'][0]
        self.assertEqual((row['a_compact'], row['b_compact']), (0, 0))
        self.assertEqual((row['ax'], row['ay']), (row['bx'], row['by']))
        self.assertEqual((row['a_original'], row['b_original']), (7, 4))
        self.assertEqual(view['omitted_edges'], 2)
        self.assertAlmostEqual(view['omitted_weight'], .5)
        self.assertAlmostEqual(view['displayed_weight']+view['omitted_weight'], view['eligible_weight'])
        self.assertFalse(view['used_to_limit_model_input'])

    def test_added_link_subset_and_empty_input(self):
        weights = np.array([[.9, .001], [.0, .0]])
        points = np.zeros((2, 2)); eligible = np.array([[False, True], [False, False]])
        view = bounded_links(weights, points, points, [0, 1], [0, 1], [0, 0], eligible=eligible)
        self.assertEqual(view['eligible_edges'], 1)
        self.assertEqual(view['rows'][0]['weight'], .001)
        empty = bounded_links(weights*0, points, points, [0, 1], [0, 1], [0, 0])
        self.assertIsNone(empty['display_fraction'])

    def test_exact_raster_roundtrip_including_holes(self):
        mask = np.array([[0, 1, 1, 0, 1], [1, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=np.uint8)
        value = mask_runs(mask)
        restored = np.zeros((value['height'], value['width']), dtype=np.uint8)
        for row, start, end in value['runs']:
            restored[row, start:end] = 1
        np.testing.assert_array_equal(restored, mask)
        self.assertEqual(value['foreground_pixels'], mask.sum())
        with self.assertRaisesRegex(ValueError, 'binary'):
            mask_runs(np.array([[0, .5, 1]]))


class CaseDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(12)
        cls.model = S7Consensus(None, CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                               head=ConsensusEvidenceHead(feature_dim=4)).eval()
        cls.pair = fixture()
        cls.provenance = dict(arm='m12', split='turufan', checkpoint_sha256='CPU-FIXTURE',
            selected_epoch=2, threshold=.5, selection_sha256='CPU-FIXTURE-selection',
            data_contract_sha256='CPU-FIXTURE-contract', geometry_calibration_sha256='CPU-FIXTURE-geometry',
            model_selection_on_test_or_real=False, threshold_refitted=False,
            purpose='unit test only; not trained real performance')
        with torch.no_grad():
            cls.pred = cls.model.score_pair(cls.pair, capture_diagnostics=True)
            cls.meta, cls.arrays = snapshot_prediction('toy', cls.pair, cls.pred,
                threshold=.5, provenance=cls.provenance)

    def test_actual_two_pass_evidence_to_rows_preserves_units(self):
        view = build_case(self.meta, self.arrays, bins=3, link_limit=1)
        self.assertEqual(view['audit']['status'], 'passed')
        self.assertFalse(view['semantics']['attention_weights_available'])
        self.assertFalse(view['audit']['layout_correctness_verified'])
        for c, original in zip(view['clusters'], self.meta['clusters']):
            total = .5*sum(row['support_weight_px'] for side in 'ab' for row in c['points']['final'][side])
            self.assertAlmostEqual(total, c['readout']['positive_evidence_px'], places=5)
            self.assertEqual(c['links']['initial_localization']['pose_rc'], c['initial_translation_rc'])
            self.assertEqual(c['links']['final_support']['pose_rc'], c['refined_translation_rc'])
            source = self.arrays[original['refinement']['localization_weights']]
            self.assertAlmostEqual(c['heatmaps']['localization']['total'], source.sum(dtype=np.float64))
        self.assertEqual(view['score'], self.meta['score'])

    def test_no_per_image_normalization_shared_absolute_scale(self):
        one = build_case(self.meta, self.arrays, bins=5)
        two = copy.deepcopy(one)
        for row in two['q']['rows']:
            row['mean'] *= .001
        scales = shared_scales([one, two])
        self.assertAlmostEqual(scales['q']['maximum'], float(self.pair.q.max()))
        self.assertEqual(scales['q']['transformation'], 'linear')

    def test_corrupt_arrays_cannot_enter_review(self):
        arrays = dict(self.arrays)
        key = self.meta['pair']['q']; arrays[key] = arrays[key]+.01
        with self.assertRaisesRegex(ValueError, 'array content'):
            build_case(self.meta, arrays)

    def test_invalid_no_candidate_does_not_invent_heatmap_or_layout(self):
        q = torch.zeros(5, 5); q[0, 0] = float('nan')
        pair = replace(fixture(q), numeric_valid=False)
        with torch.no_grad():
            pred = self.model.score_pair(pair, capture_diagnostics=True)
            meta, arrays = snapshot_prediction('invalid', pair, pred, threshold=.5,
                                               provenance=self.provenance)
        view = build_case(meta, arrays)
        self.assertFalse(view['q']['available'])
        self.assertIsNone(view['translation_a_to_b_rc'])
        self.assertEqual(view['clusters'], [])
        json.dumps(view, allow_nan=False)

    def write_fixture_evaluation(self, directory):
        directory = Path(directory)
        written = write_snapshot(directory/'evidence/toy', self.meta, self.arrays)
        target = {k: self.meta[k] for k in ('score','accepted','selected_cluster_id','has_candidate','numeric_valid')}
        target.update(pair_id='toy', label=True, gt_known=False, target_translation_rc=None,
                      translation=self.meta['translation_a_to_b_rc'], error_px=None, layout20=False,
                      candidate_errors_px=[None]*len(self.meta['clusters']))
        (directory/'pair_predictions.jsonl').write_text(json.dumps(target)+'\n')
        (directory/'case_diagnostics.jsonl').write_text(json.dumps(target)+'\n')
        files = {
            'protocol.json': dict(status='complete', **self.provenance),
            'prediction_complete.json': dict(status='all_predictions_frozen', model_state_unchanged=True,
                sha256=sha(directory/'pair_predictions.jsonl'), **self.provenance),
            'status.json': dict(status='complete'),
            'diagnostic_index.json': dict(selected_by_new_results=False,
                cases=[dict(pair_id='toy', evidence='evidence/toy/evidence.json',
                            sidecar_sha256=written['sidecar']['sha256'])]),
        }
        for name, record in files.items():
            (directory/name).write_text(json.dumps(record))

    def test_complete_readback_preserves_turufan_missing_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_fixture_evaluation(tmp)
            result = load_evaluation(tmp, bins=3, link_limit=2)
            self.assertIsNone(result[0]['posthoc_evaluation']['layout20'])
            self.assertFalse(result[0]['posthoc_evaluation']['gt_known'])
            out = Path(tmp)/'review.json'
            self.assertEqual(write_review([tmp], out, bins=3)['cases'], 1)
            exported = json.loads(out.read_text())
            self.assertEqual(exported['cases'][0]['source']['predictions_sha256'],
                             sha(Path(tmp)/'pair_predictions.jsonl'))
            self.assertFalse(exported['model_inference_performed'])
            with self.assertRaises(FileExistsError):
                write_review([tmp], out, bins=3)

    def test_partial_or_changed_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_fixture_evaluation(tmp)
            path = Path(tmp)/'status.json'; path.write_text('{"status":"inference"}')
            with self.assertRaisesRegex(ValueError, 'complete frozen'):
                load_evaluation(tmp)
            path.write_text('{"status":"complete"}')
            with (Path(tmp)/'pair_predictions.jsonl').open('a') as f:
                f.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'complete frozen'):
                load_evaluation(tmp)

    def test_checkpoint_or_translation_relabelling_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_fixture_evaluation(tmp)
            path = Path(tmp)/'protocol.json'; data = json.loads(path.read_text())
            data['checkpoint_sha256'] = 'another model'; path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                load_evaluation(tmp)
            data['checkpoint_sha256'] = self.provenance['checkpoint_sha256']; path.write_text(json.dumps(data))
            path = Path(tmp)/'case_diagnostics.jsonl'; data = json.loads(path.read_text())
            data['translation'][0] += 1; path.write_text(json.dumps(data)+'\n')
            with self.assertRaisesRegex(ValueError, 'layout differs'):
                load_evaluation(tmp)


if __name__ == '__main__':
    unittest.main()

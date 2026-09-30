import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import ridge_runner as runner
from diagnostic_metrics import compact_evidence
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, RachelN512Pairwise
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1 import matcher as legacy
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_matcher import inputs
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_v2_v1.adapter import MatcherV2Adapter
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_v2_v1.network import MatcherV2Config


class ActualCaptureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(26093032)
        config = RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
            window_sizes_px=(7., 16., 32., 64.), landmark_count=2, activation_checkpointing=False)
        self.base = RachelN512Pairwise(config)
        self.tensors = dict(zip(runner.INPUTS, inputs()))

    def test_actual_legacy_one_forward_one_sinkhorn_unchanged_q(self):
        model = legacy.S7MatcherAdapter(self.base, frozen=True).eval()
        before = copy.deepcopy(model.state_dict())
        with patch.object(legacy, 'dustbin_sinkhorn', wraps=legacy.dustbin_sinkhorn) as sink, \
             patch.object(model, 'forward', wraps=model.forward) as forward, \
             patch.object(self.base.coarse, 'forward', side_effect=AssertionError('coarse')), \
             patch.object(self.base.local_head, 'forward', side_effect=AssertionError('head')), \
             patch.object(self.base.fusion, 'forward', side_effect=AssertionError('fusion')):
            captured, valid = runner.capture(model, self.tensors)
        self.assertEqual(sink.call_count, 1)
        self.assertEqual(forward.call_count, 1)
        self.assertIs(type(valid), bool)
        np.testing.assert_array_equal(captured['affinity'], captured['context_cosine'])
        expected = model(**self.tensors)
        np.testing.assert_array_equal(captured['q'], expected.assignment[0, :4, :4].numpy())
        np.testing.assert_array_equal(captured['original_a'], np.arange(4))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_actual_v2_ungained_cosine_is_not_final_affinity(self):
        model = MatcherV2Adapter(self.base, config=MatcherV2Config(enabled=True), frozen=True).eval()
        with torch.no_grad():
            model.upgrades.log_sharpness.fill_(.7)
            model.upgrades.scale_weight.fill_(.03)
        with patch.object(legacy, 'dustbin_sinkhorn', wraps=legacy.dustbin_sinkhorn) as sink:
            captured, _ = runner.capture(model, self.tensors)
        self.assertEqual(sink.call_count, 1)
        self.assertGreater(float(np.max(np.abs(captured['affinity'] - captured['context_cosine']))), .01)
        expected = model(**self.tensors)
        np.testing.assert_array_equal(captured['q'], expected.assignment[0, :4, :4].numpy())
        np.testing.assert_array_equal(captured['affinity'], expected.affinity[0, :4, :4].numpy())

    def test_gt_label_and_trainable_model_rejected_before_forward(self):
        model = legacy.S7MatcherAdapter(self.base, frozen=True).eval()
        with patch.object(model, 'forward', side_effect=AssertionError('must not run')):
            with self.assertRaisesRegex(ValueError, 'six Matcher'):
                runner.capture(model, dict(self.tensors, label=torch.tensor([True])))
            model.set_frozen(False)
            with self.assertRaisesRegex(ValueError, 'frozen'):
                runner.capture(model, self.tensors)

    def test_train_mode_double_precision_and_multi_pair_rejected(self):
        model = legacy.S7MatcherAdapter(self.base, frozen=True)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            runner.capture(model.train(), self.tensors)
        model.eval()
        with self.assertRaisesRegex(ValueError, 'FP32'):
            runner.capture(model, dict(self.tensors, mask_a=self.tensors['mask_a'].double()))
        with self.assertRaisesRegex(ValueError, 'one pair'):
            runner.capture(model, {k: torch.cat([v, v]) for k, v in self.tensors.items()})

    def test_non_cpu_environment_rejected_before_terminal_loading(self):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}), \
             patch.object(runner, 'terminal_model', side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                runner.run(SimpleNamespace())


class FrozenTargetJoinTests(unittest.TestCase):
    def make_fixture(self, root):
        (root/'raw').mkdir()
        points = np.array([[0., 0.], [0., 10.], [0., 20.]], dtype=np.float32)
        q = np.eye(3, dtype=np.float32) / 3
        evidence = compact_evidence(points, points + [6, -8], np.ones(3, bool), np.ones(3, bool),
            q, q, q, np.zeros(3), np.zeros(3))
        runner.save_arrays(root/'raw/fixture.npz', evidence)
        rows = [dict(pair_id=str(i), numeric_valid=True, raw_file='fixture.npz',
                     raw_sha256=runner.sha(root/'raw/fixture.npz'), model_inputs_sha256='fixture') for i in range(233)]
        runner.save_rows(root/'raw_predictions.jsonl', rows)
        plan = dict(rows=[dict(pair_id=str(i), label=True, role='real_cal' if i < 58 else 'real_select',
            seam_group='J', fragment_a_id='a'+str(i), fragment_b_id='b'+str(i)) for i in range(233)])
        targets = [dict(pair_id=str(i), fragment_a_token='a'+str(i), fragment_b_token='b'+str(i),
                        translation_gt_a_to_b_rc=[6, -8]) for i in range(233)]
        runner.save(root/'gt.json', dict(positive_pairs=targets))
        plan['inputs'] = dict(layout_gt=runner.bind(root/'gt.json'))
        proof = dict(predictions=runner.bind(root/'raw_predictions.jsonl'), targets_joined=False,
            pairs=233, model_state_unchanged=True)
        return plan, rows, proof

    def test_gt_not_read_without_durable_prediction_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, _ = self.make_fixture(root)
            seen = []
            original = runner.read
            def guarded(path):
                seen.append(Path(path).name)
                return original(path)
            with patch.object(runner, 'read', side_effect=guarded), self.assertRaises(FileNotFoundError):
                runner.targets_after_freeze(plan, rows, root)
            self.assertNotIn('gt.json', seen)

    def test_complete_233_pairs_reopen_and_no_evidence_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, proof = self.make_fixture(root)
            runner.save(root/'prediction_complete.json', proof)
            actual = runner.targets_after_freeze(plan, rows, root)
            self.assertEqual(len(actual), 233)
            self.assertEqual(actual[0]['diagnostic']['directions']['a_to_b']['metrics']['true_partner_q_share']['p50'], 1.)
            self.assertEqual(runner.sha(root/'raw/fixture.npz'), rows[0]['raw_sha256'])
            self.assertEqual(actual, runner.targets_after_freeze(plan, rows, root))

    def test_reordered_or_incomplete_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, proof = self.make_fixture(root)
            runner.save(root/'prediction_complete.json', proof)
            with self.assertRaisesRegex(ValueError, 'membership/order'):
                runner.targets_after_freeze(plan, list(reversed(rows)), root)
            with self.assertRaisesRegex(ValueError, 'durable complete'):
                runner.targets_after_freeze(plan, rows[:-1], root)

    def test_raw_tensor_or_gt_endpoint_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, proof = self.make_fixture(root)
            runner.save(root/'prediction_complete.json', proof)
            bad = copy.deepcopy(rows); bad[0]['raw_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'raw tensor'):
                runner.targets_after_freeze(plan, bad, root)
            plan['rows'][0]['fragment_a_id'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'endpoint order'):
                runner.targets_after_freeze(plan, rows, root)

    def test_frozen_strata_sha_and_exclusive_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.json'
            runner.save(path, {})
            with self.assertRaisesRegex(ValueError, 'strata changed'):
                runner.verify_strata(path)
            with self.assertRaises(FileExistsError):
                runner.save(path, {})
            with self.assertRaises(FileExistsError):
                runner.save_arrays(path, {'q': np.zeros(1)})


if __name__ == '__main__':
    unittest.main()

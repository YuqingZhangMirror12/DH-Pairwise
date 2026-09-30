import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ..curriculum_training_v1.checkpoint_io import file_sha
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.training_core import Topology
from ..s7_consensus_v1.metrics import summarize
from ..s7_consensus_v1.config import TrainingConfig
from .test_model_runtime import plan_fixture
from .validation import (bind_dunhuang, check_report, select_dunhuang, select_history,
                         ValidationAdapter, DunhuangDevelopment)


def rows_fixture():
    def pair(name, positive, score, layout):
        return dict(pair_id=name, label=positive, gt_known=positive, score=score,
                    has_candidate=True, numeric_valid=True, layout20=layout, candidate_coverage=layout)
    return {'dunhuang_cv': {'real_cal': [pair('cp', True, .5, True), pair('cn', False, .3, False)],
                            'real_select': [pair('sp', True, .7, True), pair('sn', False, .6, False)]}}


class ValidationTests(unittest.TestCase):
    def report(self):return select_dunhuang(rows_fixture(), SimpleNamespace(threshold_tie_preference=.3), summarize)

    def test_cal_threshold_not_select_or_turufan(self):
        report = self.report()
        self.assertEqual(report['thresholds'], {'dunhuang_cv': .31})
        self.assertEqual(report['domains']['dunhuang_cv']['select']['fp'], 1)
        self.assertFalse(report['turufan_used'])
        self.assertEqual(report['key'][0], report['domains']['dunhuang_cv']['select']['joint_f1'])

    def test_reject_real_test_turufan_duplicates_or_missing_GT(self):
        for mutate in (lambda x: x.update(turufan={}),
                       lambda x: x['dunhuang_cv'].update(real_test=[]),
                       lambda x: x['dunhuang_cv']['real_select'][0].update(pair_id='cp'),
                       lambda x: x['dunhuang_cv']['real_select'][0].update(gt_known=False)):
            rows = rows_fixture(); mutate(rows)
            with self.assertRaises(ValueError):select_dunhuang(rows, SimpleNamespace(threshold_tie_preference=.3), summarize)

    def test_matcher_cannot_select_on_real(self):
        simulation = dict(stage='matcher', real_used=False, key=[.4, .3, -.1], selection_value=.4)
        check_report('matcher', simulation, None)
        with self.assertRaises(ValueError):check_report('matcher', simulation, self.report())

    def test_real_evaluator_uses_actual_native_config_and_six_inputs(self):
        # Synthetic prepacked arrays exercise the production inference loop,
        # rather than the report selector alone. No real dataset is opened.
        evaluator = object.__new__(DunhuangDevelopment)
        evaluator.arrays = dict(packed_masks=np.zeros((2, 800, 100), dtype=np.uint8),
            points=np.zeros((2, 512, 2), dtype=np.float32), valid=np.ones((2, 512), dtype=bool))
        evaluator.lookup = dict(a=0, b=1)
        evaluator.pairs = {role: [dict(pair_id=role+str(p), label=p, fragment_a_id='a', fragment_b_id='b')
                                 for p in (False, True)] for role in ('real_cal', 'real_select')}
        evaluator.gt = {role+'True': dict(fragment_a_token='a', fragment_b_token='b',
                       translation_gt_a_to_b_rc=[0., 0.]) for role in evaluator.pairs}
        evaluator.metric = summarize
        evaluator.matcher_api = SimpleNamespace(INPUTS=('mask_a', 'mask_b', 'points_rc_a',
                                                         'points_rc_b', 'contour_valid_a', 'contour_valid_b'))
        calls = []
        def matcher(*inputs):
            self.assertEqual(len(inputs), 6)
            self.assertEqual(inputs[0].shape[1:], (1, 800, 800))
            self.assertEqual(inputs[2].shape[1:], (512, 2))
            calls.append(len(inputs[0])); return None
        evaluator.evidence = SimpleNamespace(PairEvidence=SimpleNamespace(from_matcher=lambda *args:
            SimpleNamespace(q=torch.ones(1))))
        model = SimpleNamespace(eval=lambda: None, matcher=matcher, score_pair=lambda pair:
            SimpleNamespace(translation_a_to_b_rc=torch.zeros(2), has_candidate=True,
                            numeric_valid=True, score=torch.tensor(.5), clusters=[]))
        report, rows = evaluator.evaluate(model, 'cpu', TrainingConfig(microbatch=1))
        self.assertEqual(calls, [1, 1, 1, 1])
        self.assertEqual(set(rows), {'dunhuang_cv'})
        self.assertFalse(report['turufan_used'])
        with self.assertRaisesRegex(ValueError, 'at most eight'):
            evaluator.evaluate(model, 'cpu', TrainingConfig(microbatch=32))

    def test_observation_and_history_replay_with_Dun_only(self):
        _, plan, _ = plan_fixture(module='scorer_patch')
        simulation = dict(stage='scorer', real_used=False, key=[.9, .8, .7], selection_value=.9, threshold=.5)
        written = []
        def writer(update, record, rows):
            written.append(rows)
            return dict(path='/explicit-synthetic-test/'+str(update), sha256=digest(rows))
        adapter = ValidationAdapter(plan, Topology(0, 1, 4, 1), lambda: (simulation, []),
                                    lambda: (self.report(), rows_fixture()), writer)
        observations = [dict(update=i, report=adapter(i)) for i in plan.validation_updates]
        selection = select_history(plan, observations, plan.record['total_updates'])
        self.assertEqual(selection['best_real']['update'], plan.validation_updates[1])
        self.assertFalse(selection['turufan_used'])
        self.assertTrue(selection['fixed_budget_reached'])
        self.assertTrue(all(set(r['real_development']) == {'dunhuang_cv'} for r in written))
        bad = copy.deepcopy(observations); bad[1]['report']['real_development']['thresholds']['turufan'] = .4
        with self.assertRaises(ValueError):select_history(plan, bad, plan.record['total_updates'])

    def test_nonzero_rank_receives_writer_failure(self):
        _, plan, _ = plan_fixture()
        sim = dict(stage='matcher', real_used=False, key=[.4, .3, -.1], selection_value=.4)
        adapter = ValidationAdapter(plan, Topology(1, 2, 1, 2), lambda: (sim, []), None, lambda *x: {},
                                    broadcast=lambda _: dict(ok=False, message='disk full'))
        with self.assertRaisesRegex(ValueError, 'disk full'):adapter(0)

    def test_bind_dunhuang_does_not_open_Turufan_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); prepared = root/'prepared'; prepared.mkdir()
            (prepared/'inputs.npz').write_bytes(b'explicit synthetic binding-only test')
            gt = root/'gt.json'; gt.write_text('{}')
            pairs = [dict(pair_id=f'p{i}', fold=i, fragment_a_id=f'a{i}', fragment_b_id=f'b{i}') for i in range(5)]
            meta = dict(pairs=pairs, fragment_source_group={f+s: s for s in map(str, range(5)) for f in 'ab'})
            manifest = root/'manifest.json'; manifest.write_text(json.dumps(meta))
            folds = {'real_cal': [1], 'real_select': [2, 3, 4], 'real_test': [0]}
            spec = dict(remote_manifest=str(manifest), manifest_sha256=file_sha(manifest), prepared=str(prepared),
                        excluded_gt_pair_ids=[], roles={r: dict(pair_ids=[p['pair_id'] for p in pairs if p['fold'] in fs])
                                                      for r, fs in folds.items()})
            plan = dict(schema='threshold-joint-real-split/1', source_disjoint=True, role_folds=folds, gt_path=str(gt),
                        datasets={'dunhuang_cv': spec, 'turufan': {'remote_manifest': '/THIS_MUST_NOT_BE_OPENED'}})
            path = root/'plan.json'; path.write_text(json.dumps(plan))
            binding = bind_dunhuang(path)
            self.assertFalse(binding['turufan_opened']); self.assertFalse(binding['test_inferred'])
            self.assertEqual(binding['inference_roles'], ['real_cal', 'real_select'])
            meta['fragment_source_group']['a0'] = meta['fragment_source_group']['a1']
            manifest.write_text(json.dumps(meta)); spec['manifest_sha256'] = file_sha(manifest)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'leakage'):bind_dunhuang(path)


if __name__ == '__main__':unittest.main()

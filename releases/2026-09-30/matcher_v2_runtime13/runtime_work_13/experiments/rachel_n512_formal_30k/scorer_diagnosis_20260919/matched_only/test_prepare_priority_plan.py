import unittest
from pathlib import Path
from unittest.mock import patch

from . import prepare_priority_plan as p


class PriorityPlanTests(unittest.TestCase):
    def stages(self):
        with patch.object(p, 'sha', return_value='a' * 64):
            return p.make_stages(Path('/tmp/priority'), dict(pid=124783, startticks=935064765))

    def test_population_and_budget(self):
        rows = self.stages()
        self.assertEqual(len(rows), 60)
        self.assertEqual(sum(x['name'].endswith('_discard32') for x in rows), 7)
        training = [x for x in rows if 'matched_only.train' in x['command']]
        self.assertEqual(len(training), 5)
        self.assertTrue(all(x['completion_expect']['classifier_pair_exposures'] == 384000 for x in training))
        evaluation = [x for x in rows if 'matched_only.evaluate' in x['command']]
        self.assertEqual(len(evaluation), 30)
        self.assertEqual({x['command'][x['command'].index('--head-budget') + 1] for x in evaluation}, {'8', '16'})
        self.assertTrue(all(x['command'][x['command'].index('--selection') + 1] == 'fixed_epoch' for x in evaluation))
        adaptation = [x for x in rows if x['name'] in ('G0_C16_train', 'G1_C16_train')]
        self.assertEqual(len(adaptation), 2)
        self.assertTrue(all(x['completion_expect']['exposures']['pair_forwards'] == 432448 for x in adaptation))
        self.assertEqual(sum(p.ADAPT_PACKAGE + 'evaluate' in x['command'] for x in rows), 12)

    def test_distinct_candidate_inputs_and_legacy_order(self):
        rows = self.stages()
        for row in rows:
            command = row['command']
            if 'matched_only.train' in command:
                arm = command[command.index('--arm') + 1]
                self.assertEqual('--train-stage-cache' in command, arm in ('edge_seed', 'edge_multi'))
        self.assertEqual([x['name'] for x in rows[-2:]], ['retained_depth', 'retained_spectral'])
        self.assertIn(str(p.SPECTRAL), rows[-1]['env']['PYTHONPATH'])
        self.assertNotIn(str(p.OLD), rows[-1]['env']['PYTHONPATH'])
        self.assertEqual(rows[0]['env']['CUDA_VISIBLE_DEVICES'], '')

    def test_cache_missing_is_not_success(self):
        base = dict(pid=4, gpu_jobs_started=False, status='waiting_feature_cache', completed_splits=[])
        with self.assertRaisesRegex(RuntimeError, 'exited without complete'):
            p.cache_ready(base, None, 4, 20)
        with self.assertRaisesRegex(RuntimeError, 'reused'):
            p.cache_ready(base, dict(startticks=21, state='S'), 4, 20)
        base.update(status='complete', completed_splits=['train', 'val'])
        self.assertFalse(p.cache_ready(base, dict(startticks=20, state='R'), 4, 20))
        self.assertTrue(p.cache_ready(base, None, 4, 20))


if __name__ == '__main__':
    unittest.main()

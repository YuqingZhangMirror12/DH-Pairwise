from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from consensus_binary_eval_common import evaluate as common
from ..curriculum_scorer_eval_v1 import evaluate, audit
from ..curriculum_scorer_eval_v1 import test_evaluate as scorer_fixtures
from ..curriculum_training_v1.checkpoint_io import file_sha
from . import population
from . import test_data_runtime as data_fixtures
from .prepare_data import canonical_entry
from ..curriculum_training_v1.exposure import digest


class PopulationTests(unittest.TestCase):
    def test_fixed_ten_positive_each_type_before_any_scores(self):
        rows = [dict(pair_id=f'{r}-{i:02d}', recipe='straight_'+r, label=i%2 == 0)
                for r in 'MJR' for i in range(30)]
        chosen = population.fixed_straight_examples(rows)
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertEqual(len(chosen), 30)
        self.assertEqual(chosen, population.fixed_straight_examples(list(reversed(rows))))
        groups = population.population_groups(rows, 'sim_straight_select', {})
        self.assertEqual({name:len(r) for name, r in groups.items()}, dict(all=90, straight_M=30, straight_J=30, straight_R=30))
        with self.assertRaises(ValueError):population.fixed_straight_examples(rows[:30])

    def test_actual_light_head_straight_subgroups_and_post_prediction_GT(self):
        helper = scorer_fixtures.InferenceTests()
        for variant in ('patch', 'stats'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                model, pair, proposals, plan, origin, meta, targets, batch = helper.setup_fixture(root, variant)
                split = 'sim_straight_select'; plan['pair_counts'] = {split:6}
                origin.update(thresholds={split:.3}, threshold_origins={split:'SIM-CAL; no straight fitting'})
                for i, row in enumerate(meta['pairs']):row['recipe'] = 'straight_'+'MJR'[i//2]
                for row in targets:row['gt_pose'] = [0., 0.] if row['label'] else None
                out = root/'out'; actual = model.score_pair
                def join(*args):
                    self.assertTrue((out/'prediction_complete.json').exists())
                    return targets
                with ExitStack() as stack:
                    stack.enter_context(patch.object(common.PairEvidence, 'from_matcher', return_value=pair))
                    stack.enter_context(patch.object(model, 'score_pair', side_effect=lambda p, **kw:actual(p, proposals=proposals, **kw)))
                    result = evaluate.run_population(model, None, plan, split, origin, out, 'cpu',
                        registered_splits=population.SPLITS,
                        population_loader=lambda *args:(meta, iter([(meta['pairs'], batch)]), {'CPU_fixture':True}, None),
                        group_builder=population.population_groups, target_loader=join)
                self.assertEqual(result['main_group'], 'all')
                self.assertEqual(result['threshold_origin'], 'SIM-CAL; no straight fitting')
                self.assertEqual(result['groups']['straight_J']['primary']['pairs'], 2)
                self.assertEqual(audit.verify_population(out)['pairs'], 6)

    def test_straight_loader_recounts_canonical_hash_and_refuses_change(self):
        fixture = data_fixtures.CanonicalLoaderTests(); fixture.setUp()
        try:
            row = canonical_entry(fixture.entry, fixture.audit, 'select', 42, fixture.loader)
            with patch.object(population, 'bound_module') as api:
                api.return_value.load_sample = fixture.loader
                dataset = population.StraightPopulation([fixture.entry], {'records':[fixture.audit]},
                    'select', 42, digest([row]), None)
                self.assertIs(dataset[0][0], fixture.sample)
                fixture.path.write_bytes(b'changed after complete audit')
                with self.assertRaisesRegex(ValueError, 'sample changed'):dataset[0]
        finally:fixture.doCleanups()


if __name__ == '__main__':unittest.main()

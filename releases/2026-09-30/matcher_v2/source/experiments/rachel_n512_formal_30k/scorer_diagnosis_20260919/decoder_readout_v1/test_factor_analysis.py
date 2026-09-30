"""Independent arithmetic, scope and completion tests; no remote inference."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import analyze_factor_sweep as analysis


def row(pid, label, score, *, good=False, covered=False, valid=True, known=None):
    return dict(pair_id=pid, label=label, score=score, good=good, covered=covered,
                valid=valid, known=label if known is None else known,
                pose=[0., 0.] if good else [50., 0.], winner=0,
                pairs=10, sum_q=.3, max_q=.1, candidate_count=2, seconds=.1)


def six_rows():
    return [row('p1', True, .8, good=True, covered=True),
            row('p2', True, .7, covered=True),
            row('p3', True, .1, good=True, covered=True),
            row('p4', True, .9, valid=False),
            row('n1', False, .8), row('n2', False, .1)]


def record(model='binary_patch', *, split='dunhuang_cv', label=True):
    candidates = [dict(translation=[0., 0.], pairs=8, sum_q=.8, max_q=.2,
                       score=.6, logit=math.log(.6 / .4), members=[1]),
                  dict(translation=[50., 0.], pairs=4, sum_q=.7, max_q=.3,
                       score=.8, logit=math.log(.8 / .2), members=[2])]
    predictions = {}
    for search in analysis.SEARCHES:
        variants = {name: dict(winner=1, score=.8, logits=[c['logit'] for c in candidates])
                    for name in analysis.expected_readouts(model)}
        variants['raw_sum_q_rank'] = dict(winner=0, score=.6)
        variants['raw_max_q_rank'] = dict(winner=1, score=.8)
        predictions[search] = dict(candidates=deepcopy(candidates), readouts=variants,
            numeric_valid=True, seconds=.1,
            search_audit=dict(gt_used=False, policy=dict(
                row_column_topk=3 if search in ('top3_only', 'combined') else 2,
                mode_limit=None if search in ('all_modes_only', 'combined') else 128,
                initial_seeds=32 if search in ('seeds32_only', 'combined') else 16)))
    return dict(pair_id='a', split=split, role='real_cal' if split != 'sim_select' else 'sim_select',
                label=label, target=[0., 0.] if label and split != 'turufan' else None,
                predictions=predictions, reference_equivalence=dict(strict=True))


def check_record(value, model='binary_patch'):
    return analysis.check_record(value, model,
        {'dunhuang_cv': {'a': 'real_cal'}, 'turufan': {'a': 'real_cal'}},
        {'a': value['label']}, {'dunhuang_cv': {'a': value['label']}, 'turufan': {'a': value['label']}})


class Arithmetic(unittest.TestCase):
    def test_manual_pair_and_joint_denominators(self):
        m = analysis.metrics(six_rows(), .3, layout_available=True)
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['tn']), (2, 1, 2, 1))
        self.assertEqual(m['accuracy'], .5)
        self.assertAlmostEqual(m['f1'], 4 / 7)
        self.assertEqual((m['layout20_count'], m['candidate_coverage_count']), (2, 3))
        self.assertEqual((m['joint_tp'], m['joint_fp'], m['joint_fn']), (1, 2, 3))
        self.assertAlmostEqual(m['joint_f1'], 2 / 7)
        self.assertEqual(m['winner_correct_but_rejected'], 1)
        self.assertEqual(m['wrong_pose_accepted'], 1)

    def test_matches_independent_existing_numpy_formula(self):
        # Production helper is imported only in this TEST, not in the analyzer.
        source = Path(__file__).parent.parent / 's7_consensus_v1/metrics.py'
        spec = importlib.util.spec_from_file_location('production_metric_reference', source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        rows = six_rows()
        translated = [dict(label=r['label'], gt_known=r['known'], score=r['score'],
                           numeric_valid=r['valid'], has_candidate=r['valid'],
                           layout20=r['good'], candidate_coverage=r['covered']) for r in rows]
        actual = analysis.metrics(rows, .3, layout_available=True)
        original = module.summarize(translated, .3)
        for key in set(actual) & set(original):
            self.assertEqual(actual[key], original[key], key)

    def test_invalid_high_score_is_rejected(self):
        self.assertFalse(analysis.accepted(row('x', True, 1., valid=False), .2))

    def test_threshold_boundary_is_greater_equal(self):
        self.assertTrue(analysis.accepted(row('x', True, .30), .30))
        self.assertFalse(analysis.accepted(row('x', True, .30 - 1e-12), .30))

    def test_tied_average_precision_and_auc(self):
        rows = [row('a', True, .5, good=True, covered=True), row('b', False, .5)]
        m = analysis.metrics(rows, .3, layout_available=True)
        self.assertEqual(m['ap'], .5); self.assertEqual(m['auroc'], .5)
        self.assertEqual(analysis.average_precision(list(reversed(rows))), .5)

    def test_no_gt_domain_is_null_not_zero(self):
        rows = [row('a', True, .8, known=False), row('b', False, .1)]
        m = analysis.metrics(rows, .3, layout_available=False)
        self.assertEqual(m['f1'], 1.)
        for key in ('layout20_count', 'known_positive_layouts', 'joint_tp', 'joint_f1'):
            self.assertIsNone(m[key])
        delta = analysis.transitions(rows, rows, .3, .3, layout_available=False)
        self.assertIsNone(delta['counts']['layout_gained'])

    def test_missing_positive_gt_is_not_silently_excluded(self):
        with self.assertRaisesRegex(ValueError, 'incomplete Layout GT'):
            analysis.metrics([row('a', True, .8, known=False)], .3, layout_available=True)

    def test_calibration_only_real_cal_grid(self):
        rows = [row('a', True, .6, good=True, covered=True), row('b', False, .4)]
        m = analysis.calibrate(rows, 'dunhuang_cv')
        self.assertEqual(m['threshold'], .41)
        self.assertEqual(m['joint_f1'], 1.)
        for role in ('real_select', 'real_test', 'sim_select'):
            with self.assertRaises(ValueError): analysis.calibrate(rows, 'dunhuang_cv', role=role)
        with self.assertRaises(ValueError): analysis.calibrate(rows, 'sim_select')

    def test_separate_real_domains_get_separate_thresholds(self):
        dun = [row('a', True, .6, good=True, covered=True), row('b', False, .4)]
        turu = [row('a', True, .29, known=False), row('b', False, .22)]
        self.assertEqual(analysis.calibrate(dun, 'dunhuang_cv')['threshold'], .41)
        self.assertEqual(analysis.calibrate(turu, 'turufan')['threshold'], .29)

    def test_paired_gains_include_new_false_positives(self):
        before = [row('p', True, .1, good=True, covered=True), row('n', False, .1)]
        after = [dict(before[1], score=.8), dict(before[0], score=.8)]
        delta = analysis.transitions(before, after, .3, .3, layout_available=True)
        self.assertEqual(delta['counts']['positive_recovered'], 1)
        self.assertEqual(delta['counts']['false_positive_added'], 1)
        self.assertEqual(delta['counts']['classification_gained'], 1)
        self.assertEqual(delta['counts']['classification_lost'], 1)
        self.assertEqual(delta['pair_ids']['false_positive_added'], ['n'])

    def test_pairing_rejects_missing_duplicate_or_changed_truth(self):
        rows = six_rows()
        for bad in (rows[:-1], rows[:-1] + rows[:1], [dict(r, label=not r['label']) for r in rows]):
            with self.assertRaises(ValueError):
                analysis.transitions(rows, bad, .3, .3, layout_available=True)

    def test_quantiles_include_spread_and_empty_null(self):
        q = analysis.quantiles([0, 0, 0, 100])
        self.assertEqual(q['p50'], 0); self.assertAlmostEqual(q['p90'], 70)
        self.assertEqual(q['mean'], 25)
        self.assertIsNone(analysis.quantiles([])['p50'])


class ScopeAndSemantics(unittest.TestCase):
    def test_valid_record_all_model_types(self):
        for model in analysis.MODELS:
            check_record(record(model), model)

    def test_no_test_or_excluded_identity(self):
        for change in ({'role': 'real_test'}, {'pair_id': 'not-in-plan'}, {'split': 'sim_test'}):
            value = record(); value.update(change)
            with self.assertRaises(ValueError): check_record(value)

    def test_turufan_and_negative_cannot_have_layout_gt(self):
        for value in (record(split='turufan'), record(label=False)):
            check_record(value)
            value['target'] = [0., 0.]
            with self.assertRaises(ValueError): check_record(value)

    def test_seed32_cannot_secretly_double_mode_budget(self):
        value = record(); value['predictions']['seeds32_only']['search_audit']['policy']['mode_limit'] = 256
        with self.assertRaisesRegex(ValueError, 'confounded'): check_record(value)

    def test_q_ranking_keeps_original_winner_score_not_q(self):
        value = record(); check_record(value)
        actual = analysis.outcome(value, 'baseline', 'raw_sum_q_rank')
        self.assertEqual(actual['winner'], 0)
        self.assertEqual(actual['score'], .6)
        self.assertEqual(actual['sum_q'], .8)
        self.assertTrue(actual['good'])
        value['predictions']['baseline']['readouts']['raw_sum_q_rank']['score'] = .8
        with self.assertRaisesRegex(ValueError, 'inconsistent'): check_record(value)

    def test_layout_ranking_differs_from_candidate_coverage(self):
        value = record()
        baseline = analysis.outcome(value, 'baseline', 'baseline')
        self.assertTrue(baseline['covered']); self.assertFalse(baseline['good'])
        self.assertEqual(baseline['error'], 50.)

    def test_reject_nonfinite_missing_control_wrong_winner_or_budget(self):
        variants = []
        bad = record(); bad['predictions']['baseline']['candidates'][0]['sum_q'] = math.nan; variants.append(bad)
        bad = record(); del bad['predictions']['top3_only']; variants.append(bad)
        bad = record(); bad['predictions']['baseline']['readouts']['baseline']['winner'] = 0; variants.append(bad)
        bad = record(); bad['predictions']['baseline']['candidates'] *= 5; variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError): check_record(bad)

    def test_failure_beats_complete_and_pilot_never_completes_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); folder = root / 'full_development_01/binary_patch'; folder.mkdir(parents=True)
            failure = root / 'controller_failure.json'; failure.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'failure takes precedence'):
                analysis.load_verified(root, 'binary_patch', 'unused', 'unused', 'unused', {})
            failure.unlink()
            (folder / 'complete.json').write_text(json.dumps(dict(status='pilot_complete')))
            (folder / 'protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                analysis.load_verified(root, 'binary_patch', 'unused', 'unused', 'unused', {})

    def test_complete_file_receipts_and_full_membership_end_to_end(self):
        # Synthetic fixture tests admission, NOT an experimental result.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'source_01'; source.mkdir()
            for name in ('run_factor_sweep.py', 'decoder.py'):
                (source / name).write_text('# synthetic test binding\n')
            source_plan = root / 'source_plan.json'; source_plan.write_text('{}')
            folder = root / 'full_development_01/binary_stats'; folder.mkdir(parents=True)
            plan = dict(source_disjoint=True, role_folds={'real_cal': [1], 'real_select': [2, 3, 4],
                                                        'real_test': [0]}, datasets={})
            specs = {'dunhuang_cv': (160, 479, 161), 'turufan': (120, 360, 122)}
            real_paths = {}; identities = []
            for split, counts in specs.items():
                manifest = []; roles = {}; index = 0
                for role, count in zip(('real_cal', 'real_select', 'real_test'), counts):
                    ids = []
                    for _ in range(count):
                        pid = split + str(index); label = bool(index % 2); index += 1
                        manifest.append(dict(pair_id=pid, label=label)); ids.append(pid)
                        if role != 'real_test': identities.append((split, role, pid, label))
                    roles[role] = dict(pair_ids=ids)
                path = root / (split + '.json'); path.write_text(json.dumps(dict(pairs=manifest)))
                real_paths[split] = path
                plan['datasets'][split] = dict(roles=roles, excluded_gt_pair_ids=[],
                                              manifest_sha256=analysis.sha(path))
            plan_path = root / 'real_split.json'; plan_path.write_text(json.dumps(plan))
            sim_path = root / 'sim.json'
            sim = [dict(pair_id='sim' + str(i), label=bool(i % 2)) for i in range(1500)]
            sim_path.write_text(json.dumps(dict(entries=sim)))
            identities += [('sim_select', 'sim_select', r['pair_id'], r['label']) for r in sim]
            path = folder / 'records.jsonl'
            templates = {(split, label): record('binary_stats', split=split, label=label)
                         for split in analysis.POPULATION for label in (False, True)}
            with path.open('w') as stream:
                for split, role, pid, label in identities:
                    r = dict(templates[(split, label)], pair_id=pid, role=role)
                    stream.write(json.dumps(r) + '\n')
            (folder / 'summary.json').write_text('{}')
            protocol = dict(model='binary_stats', pilot_limit=None, threshold=.3,
                            source_plan_sha256=analysis.sha(source_plan),
                            real_roles_sha256=analysis.sha(plan_path),
                            script_sha256=analysis.sha(source / 'run_factor_sweep.py'),
                            decoder_sha256=analysis.sha(source / 'decoder.py'),
                            threshold_refitted=False, training_performed=False, gpu_used=False, test_used=False)
            (folder / 'protocol.json').write_text(json.dumps(protocol))
            (folder / 'complete.json').write_text(json.dumps(dict(
                status='complete', model_unchanged=True, pairs=2619, population=analysis.POPULATION,
                records_sha256=analysis.sha(path), summary_sha256=analysis.sha(folder / 'summary.json'))))
            (root / 'full_development_01_binary_stats_exit.json').write_text('{"returncode":0}')
            groups, actual, eq = analysis.load_verified(root, 'binary_stats', plan_path,
                                                        source_plan, sim_path, real_paths)
            self.assertEqual(len(groups[('sim_select', 'sim_select', 'baseline', 'baseline')]), 1500)
            self.assertEqual(len(groups[('dunhuang_cv', 'real_select', 'baseline', 'baseline')]), 479)
            self.assertEqual(eq['turufan']['strict'], 480)
            self.assertEqual(actual['threshold'], .3)
            # A file appended after a completed receipt cannot be admitted.
            with path.open('a') as stream: stream.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                analysis.load_verified(root, 'binary_stats', plan_path, source_plan, sim_path, real_paths)

    def test_group_analysis_uses_cal_without_select_labels_and_excludes_pooling_diagnostics(self):
        groups = {}
        for split in analysis.POPULATION:
            roles = ('sim_select',) if split == 'sim_select' else ('real_cal', 'real_select')
            for role in roles:
                values = [row('p', True, .6, good=split != 'turufan', covered=split != 'turufan',
                              known=split != 'turufan'), row('n', False, .4)]
                for search in analysis.SEARCHES:
                    for kind in analysis.expected_readouts('binary_patch'):
                        groups[(split, role, search, kind)] = deepcopy(values)
        equivalence = {split: {'sensitive_ids': ['p']} for split in ('dunhuang_cv', 'turufan')}
        result = analysis.analyze_groups(groups, {'threshold': .3}, 'binary_patch', equivalence)
        real = [r for r in result['rows'] if r['threshold_mode'] == 'separate_real_cal']
        self.assertTrue(all(r['metrics']['threshold'] == .41 for r in real))
        self.assertTrue(all(r['strict_cpu_gpu_reproduction_subset']['pairs'] == 1 for r in real))
        self.assertTrue(all(r['strict_cpu_gpu_reproduction_subset']['metrics']['threshold'] == .41 for r in real))
        self.assertTrue(all(r['metrics']['pairs'] == 2 for r in real))
        self.assertTrue(all(r['readout'] not in analysis.POOLING_DIAGNOSTICS
                            for r in result['development_ranking']))
        self.assertTrue(all('sim_select_delta' in r for r in result['development_ranking']))
        self.assertFalse(result['automatic_production_change'])
        # Change SELECT label/score; its threshold must still come only from CAL.
        changed = deepcopy(groups)
        for key, values in changed.items():
            if key[1] == 'real_select':
                for value in values: value['score'] = .9
        other = analysis.analyze_groups(changed, {'threshold': .3}, 'binary_patch')
        real_other = [r for r in other['rows'] if r['threshold_mode'] == 'separate_real_cal']
        self.assertTrue(all(r['metrics']['threshold'] == .41 for r in real_other))


if __name__ == '__main__':
    unittest.main()

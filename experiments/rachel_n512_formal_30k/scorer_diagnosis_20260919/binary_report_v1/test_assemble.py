"""Synthetic terminal fixtures; no server, trained weights, or real data."""
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import assemble as a


def save(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + '\n')


def selected(experiment):
    spec = a.EXPERIMENTS[experiment]
    origin = dict(sha256=a.E32_SHA, selected_epoch=32, historical_matcher_epochs=36,
                  matcher_retrained_for_this_head=False)
    if experiment == 'aggressive_binary_patch':
        origin = dict(sha256='synthetic-new-Matcher', selected_epoch=6,
                      matcher_trained_from_random_in_this_experiment=True, epoch_selected_on_real=False)
    result = {}
    for choice in ('sim', 'real'):
        thresholds = {split: .21 for _, split in a.tasks(experiment)}
        if choice == 'real':
            thresholds.update(dunhuang_cv=.42, turufan=.53)
        result[choice] = dict(variant='binary_' + spec['variant'], experiment_variant=experiment,
            selection_kind=choice, checkpoint_sha256='synthetic-' + experiment + '-' + choice,
            selected_epoch=8 if choice == 'sim' else 10, last_epoch=28, thresholds=thresholds,
            selection_on_test=False, selection_on_real=choice == 'real', real_used_for_stopping=False,
            matcher_origin=deepcopy(origin), matcher_updated_during_training=False,
            matcher_updated_during_scorer_training=False, real_plan_sha256=a.REAL_PLAN_SHA,
            data_contract_sha256='synthetic-new-data' if experiment == 'aggressive_binary_patch' else 'synthetic-v14',
            geometry_calibration_sha256='synthetic-geometry', selection_sha256='synthetic-selection',
            terminal_receipt_sha256='synthetic-terminal', stop_reason='simulation_plateau_after_lr_reductions')
    return result


def job_value(root, experiment, choice, split, model):
    counts = {'all': 3000} if split.startswith('sim_test_') else a.REAL_GROUP_COUNTS[split]
    groups = {name: {policy: dict(pairs=n, threshold=threshold, f1=.5, joint_f1=None, layout20=None)
                    for policy, threshold in (('primary', model['thresholds'][split]), ('fixed03', .3))}
              for name, n in counts.items()}
    ncases = 10 if split == 'dunhuang_cv' else 1 if split == 'turufan' else 0
    return dict(schema='binary-report-job/1', experiment=experiment, selection_kind=choice, split=split,
        selected_epoch=model['selected_epoch'], checkpoint_sha256=model['checkpoint_sha256'],
        threshold=model['thresholds'][split], main_group='all' if split.startswith('sim_test_') else 'real_test',
        groups=groups, cases=[dict(pair_id='synthetic-' + str(i)) for i in range(ncases)],
        source=dict(job=str(root), summary_sha256='synthetic-summary', predictions_sha256='synthetic-predictions'))


@contextmanager
def fixture(experiment='binary_patch', execution=None):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        spec = a.EXPERIMENTS[experiment]
        models = selected(experiment)
        if execution is not None:
            for model in models.values():
                model['execution_continuation'] = deepcopy(execution)
        prefix = 'aggressive-binary' if experiment == 'aggressive_binary_patch' else 'binary'
        gate = dict(schema=prefix + '-frozen-selection-gate/1', variant=spec['variant'], status='passed',
                    trained_checkpoints_inspected=True, real_inference_performed=False,
                    selected=models, source_bindings={'synthetic': True})
        plan = dict(schema=prefix + '-frozen-evaluation-queue/1', variant=spec['variant'],
                    tasks=sorted(a.tasks(experiment)), selected=models, source_bindings=gate['source_bindings'])
        save(root / 'plan.json', plan); save(root / 'selected_models.json', gate)
        jobs, exports = [], {}
        for choice, split in sorted(a.tasks(experiment)):
            task = choice + '_' + split
            folder = root / task; folder.mkdir()
            save(folder / 'protocol.json', models[choice])
            proof = dict(status='passed', fixture=True, selection_kind=choice, split=split)
            save(root / (task + '_verified.json'), proof)
            save(root / (task + '_exit.json'), dict(returncode=0))
            jobs.append(dict(task=task, selection_kind=choice, split=split, status='complete', returncode=0, verified=proof))
            exports[task] = job_value(folder, experiment, choice, split, models[choice])
        complete = dict(status='complete', variant=spec['variant'], all_six_populations_verified=True,
                        fixed_case_evaluations=22, training_modified=False, plan_sha256=a.sha(root / 'plan.json'), jobs=jobs)
        save(root / 'evaluation_complete.json', complete); save(root / 'driver_status.json', complete)
        yield root, exports


def reseal(root, *, plan=False, complete=False):
    value = a.read(root / 'evaluation_complete.json')
    if plan: value['plan_sha256'] = a.sha(root / 'plan.json')
    if plan or complete:
        save(root / 'evaluation_complete.json', value); save(root / 'driver_status.json', value)


class AssemblyTests(unittest.TestCase):
    def run_fixture(self, root, values, experiment='binary_patch'):
        with patch.object(a, 'export_job', side_effect=lambda directory, receipt: deepcopy(values[Path(directory).name])):
            return a.assemble_evaluation(root, experiment)

    def test_six_jobs_preserve_every_population_and_policy(self):
        with fixture() as (root, values):
            result = self.run_fixture(root, values)
            self.assertEqual(len(result['jobs']), 6)
            self.assertEqual(len(result['rows']), 40)
            self.assertEqual(sum(r['is_main'] for r in result['rows']), 6)
            for row in result['rows']:
                original = values[row['selection_kind'] + '_' + row['split']]
                self.assertEqual(row['metrics'], original['groups'][row['population']][row['policy']])
            real = [r for r in result['rows'] if r['selection_kind'] == 'real' and r['is_main']]
            self.assertEqual({r['split']: r['metrics']['threshold'] for r in real},
                             {'sim_test_v14': .21, 'dunhuang_cv': .42, 'turufan': .53})

    def test_missing_experiments_are_unimported_not_zeros(self):
        with fixture() as (root, values):
            with patch.object(a, 'export_job', side_effect=lambda directory, _: deepcopy(values[Path(directory).name])):
                result = a.comparison_bundle({'binary_patch': root})
            self.assertEqual(result['unavailable_experiments'], ['binary_stats', 'aggressive_binary_patch'])
            self.assertFalse(result['all_three_frozen_evaluations_imported'])
            self.assertTrue(all(r['experiment'] == 'binary_patch' for r in result['rows']))

    def test_new_dataset_and_random_matcher_are_not_v14_or_e32(self):
        with fixture('aggressive_binary_patch') as (root, values):
            result = self.run_fixture(root, values, 'aggressive_binary_patch')
            self.assertEqual({r['split'] for r in result['rows']}, {'sim_test_aggressive', 'dunhuang_cv', 'turufan'})
            self.assertTrue(all(r['matcher_sha256'] == 'synthetic-new-Matcher' for r in result['rows']))

    def test_terminal_failures_override_complete(self):
        for name in ('failure.json', 'controller_failure.json', 'launch_failure.json'):
            with self.subTest(name=name), fixture() as (root, values):
                save(root / name, dict(status='failed'))
                with self.assertRaisesRegex(ValueError, 'failure precedes'):
                    self.run_fixture(root, values)

    def test_missing_or_duplicate_jobs_do_not_become_complete(self):
        for kind in ('missing', 'duplicate'):
            with self.subTest(kind=kind), fixture() as (root, values):
                terminal = a.read(root / 'evaluation_complete.json')
                if kind == 'missing': terminal['jobs'].pop()
                else: terminal['jobs'][-1] = terminal['jobs'][0]
                save(root / 'evaluation_complete.json', terminal); reseal(root, complete=True)
                with self.assertRaisesRegex(ValueError, 'six distinct'):
                    self.run_fixture(root, values)

    def test_nonzero_process_exit_rejected(self):
        with fixture() as (root, values):
            task = next(iter(values)); save(root / (task + '_exit.json'), dict(returncode=1))
            with self.assertRaisesRegex(ValueError, 'exit successfully'):
                self.run_fixture(root, values)

    def test_changed_verification_rejected(self):
        with fixture() as (root, values):
            save(root / (next(iter(values)) + '_verified.json'), dict(status='passed', changed=True))
            with self.assertRaisesRegex(ValueError, 'verification differs'):
                self.run_fixture(root, values)

    def test_changed_job_checkpoint_or_threshold_rejected(self):
        for key, value in (('checkpoint_sha256', 'different'), ('threshold', .4), ('selected_epoch', 12)):
            with self.subTest(key=key), fixture() as (root, values):
                values[next(iter(values))][key] = value
                with self.assertRaisesRegex(ValueError, 'selected model'):
                    self.run_fixture(root, values)

    def test_population_or_fixed_policy_cannot_silently_change(self):
        for key, value in (('pairs', 999), ('threshold', .4)):
            with self.subTest(key=key), fixture() as (root, values):
                values['sim_turufan']['groups']['real_test']['fixed03'][key] = value
                with self.assertRaisesRegex(ValueError, 'population count|policy threshold'):
                    self.run_fixture(root, values)

    def test_terminal_selection_rules(self):
        for key, value in (('selection_on_test', True), ('real_plan_sha256', 'other'),
                           ('real_used_for_stopping', True), ('last_epoch', 99)):
            with self.subTest(key=key):
                models = selected('binary_patch'); models['sim'][key] = value
                with self.assertRaises(ValueError): a.validate_selection(models, 'binary_patch')
        models = selected('binary_patch'); models['sim']['matcher_updated_during_training'] = True
        with self.assertRaisesRegex(ValueError, 'frozen E32'): a.validate_selection(models, 'binary_patch')

    def test_two_heads_require_same_frozen_matcher_and_data(self):
        with fixture('binary_patch') as (ra, va), fixture('binary_stats') as (rb, vb):
            imports = {'binary_patch': ra, 'binary_stats': rb}
            def exported(directory, _):
                directory = Path(directory)
                return deepcopy((va if directory.parent == ra else vb)[directory.name])
            with patch.object(a, 'export_job', side_effect=exported):
                self.assertEqual(len(a.comparison_bundle(imports)['experiments']), 2)
                gate = a.read(rb / 'selected_models.json'); plan = a.read(rb / 'plan.json')
                for model in gate['selected'].values(): model['data_contract_sha256'] = 'different-data'
                plan['selected'] = gate['selected']
                save(rb / 'selected_models.json', gate); save(rb / 'plan.json', plan); reseal(rb, plan=True)
                for folder in rb.iterdir():
                    if folder.is_dir():
                        protocol = a.read(folder / 'protocol.json'); protocol['data_contract_sha256'] = 'different-data'
                        save(folder / 'protocol.json', protocol)
                with self.assertRaisesRegex(ValueError, 'same frozen Matcher and data'):
                    a.comparison_bundle(imports)

    def test_assembly_does_not_modify_frozen_files(self):
        with fixture() as (root, values):
            before = {str(p.relative_to(root)): a.sha(p) for p in root.rglob('*') if p.is_file()}
            self.run_fixture(root, values)
            self.assertEqual(before, {str(p.relative_to(root)): a.sha(p) for p in root.rglob('*') if p.is_file()})

    def test_import_does_not_load_torch(self):
        code = ('import sys;sys.path.insert(0,' + repr(str(Path(a.__file__).parents[1])) + ');'
                'import binary_report_v1.assemble;assert "torch" not in sys.modules')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()

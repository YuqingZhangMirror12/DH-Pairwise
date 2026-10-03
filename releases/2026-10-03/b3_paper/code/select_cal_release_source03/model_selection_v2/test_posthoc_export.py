import copy
import json
from pathlib import Path
import unittest
import weakref

from . import posthoc_export as x
from .test_protocol import Fixture
from .test_selection import report


class FakeModel(dict):
    pass


class FakeBackend:
    def __init__(self, fixture):
        self.f = fixture; self.verified = False; self.states_read = []
        self.model_refs = {}; self.live_before_checkpoint = []

    def verify_original(self, root, spec, arm):
        if self.f.original_verification_fails:
            raise ValueError('strict original terminal failed')
        self.verified = True
        return copy.deepcopy(self.f.endpoint), {'real_used_for_selection': False}

    def checkpoint_at(self, root, update, binding):
        self.live_before_checkpoint.append([u for u, ref in self.model_refs.items() if ref() is not None])
        self.states_read.append(update)
        return dict(binding=copy.deepcopy(binding), sampling={'completed_updates': update},
                    model={'model.matcher.w': update, 'model.head.w': 0}), x.receipt(root / f'update_{update:06d}/committed.json')

    def model_only(self, state, binding):
        result = FakeModel({k[len('model.'):]: v for k, v in state['model'].items()})
        self.model_refs[state['sampling']['completed_updates']] = weakref.ref(result)
        return result

    def tree_sha(self, value):
        return x.digest(value)

    def save_model(self, path, record):
        self.live_at_publication = [u for u, ref in self.model_refs.items() if ref() is not None]
        self.f.saved_record = copy.deepcopy(record)
        path.write_text(json.dumps(record, sort_keys=True))
        return dict(**x.receipt(path), model_state_sha256=self.tree_sha(record['model']), update=record['updates'])


class ExportTests(Fixture):
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return x.receipt(path)

    def setUp(self):
        super().setUp()
        self.original_verification_fails = False; self.saved_record = None
        self.training = (self.root / 'old_training').resolve(); self.training.mkdir()
        self.spec = self.root / 'execution.json'; self.write(self.spec, {'unchanged': True})
        common = dict(module='matcher', total_updates=300, effective_batch=32)
        self.binding = dict(module='matcher', order='curriculum', run_mode='formal',
                            common_plan=common, common_plan_sha256=x.digest(common))
        self.endpoint = dict(binding=self.binding, module='matcher', selection_kind='equal_budget_endpoint',
                             updates=300, total_completed_updates=300, optimizer_imported=False, training_rng_included=False)
        launch = self.write(self.training / 'formal_launch.json', {'phase': 'formal'})
        self.write(self.training / 'formal_return.json', dict(phase='formal', returncode=0, launch_sha256=launch['sha256']))
        for name in ('controller_complete.json', 'export_process_return.json', 'gpu_gate.json',
                     'formal/training_complete.json', 'formal/exports/training_complete.json', 'formal/exports/selection.json'):
            self.write(self.training / name, {'old_artifact': name})
        self.plan_value = self.plan()
        self.protocol = self.write(self.root / 'protocol.json', self.plan_value)
        materialized_entries = []
        for i, entry in enumerate(self.manifests['select']['entries']):
            sample = self.root / f'sample_{i}.npz'; sample.write_bytes(entry['pair_id'].encode())
            materialized_entries.append(dict(pair_id=entry['pair_id'], label=entry['label'],
                sample_path=str(sample.resolve()), recipe='frozen_sim_recipe'))
        self.materialized_value = dict(split='select', real_used=False, test_used=False, entries=materialized_entries)
        self.native_inputs = dict(
            materialized_select=self.write(self.root / 'materialized_select.json', self.materialized_value),
            source_inventory=self.write(self.root / 'source_inventory.json', {'evaluation.py': 'a' * 64}),
            evaluation_config=self.write(self.root / 'evaluation_config.json', {'microbatch': 1}))
        self.backend = FakeBackend(self)
        items = []
        for update in [100, 200, 300]:
            ck = self.training / f'formal/checkpoints/update_{update:06d}'
            committed = self.write(ck / 'committed.json', {'update': update})
            rank = self.write(ck / 'rank_00.pt', {'tensor_stub': update})
            model = {'matcher.w': update, 'head.w': 0}
            items.append(dict(update=update, committed_checkpoint=committed,
                              checkpoint_file=rank, model_state_sha256=self.backend.tree_sha(model)))
        self.candidate_value = dict(schema=x.CANDIDATE_SCHEMA, arm='B3', source_controller_root=str(self.training),
            source_binding_sha256=x.digest(self.binding), candidates=items)
        self.candidate = self.write(self.root / 'candidates.json', self.candidate_value)
        self.evaluations = {}
        for item in items:
            u = item['update']; r = report(self.plan_value, u, {100: .8, 200: .5, 300: .6}[u])
            r.update(checkpoint_sha256=item['checkpoint_file']['sha256'], model_state_sha256=item['model_state_sha256'])
            report_ref = self.write(self.root / f'report_{u}.json', r)
            launch_ref = self.write(self.root / f'launch_{u}.json', dict(schema='model-selection-evaluation-launch/2',
                phase='matcher_select', role='select', protocol_sha256=self.plan_value['sha256'],
                candidate_manifest_sha256=self.candidate['sha256'], update=u,
                checkpoint_sha256=item['checkpoint_file']['sha256'], model_state_sha256=item['model_state_sha256'],
                report_path=report_ref['path'], command=['python', 'isolated_evaluate.py', '--update', str(u)],
                native_evaluator_revision='frozen_native_evaluate_view_microbatch1/1', **self.native_inputs))
            return_ref = self.write(self.root / f'return_{u}.json', dict(schema='model-selection-evaluation-return/2',
                phase='matcher_select', returncode=0, launch_sha256=launch_ref['sha256'],
                report_path=report_ref['path'], report_sha256=report_ref['sha256']))
            self.evaluations[u] = dict(report=report_ref, launch=launch_ref, process_return=return_ref)
        self.before = {str(p.relative_to(self.training)): x.file_sha(p) for p in self.training.rglob('*') if p.is_file()}

    def run_export(self, output=None):
        return x.export_reselected_matcher(output or self.root / 'new_export', self.training, self.spec, 'B3',
            self.protocol, self.candidate, self.evaluations, backend=self.backend)

    def rewrite_eval(self, update, part, mutate):
        ref = self.evaluations[update][part]; value = x.read(ref['path']); mutate(value)
        self.evaluations[update][part] = self.write(Path(ref['path']), value)

    def rewrite_launch(self, update, mutate):
        self.rewrite_eval(update, 'launch', mutate)
        self.rewrite_eval(update, 'process_return',
            lambda r: r.__setitem__('launch_sha256', self.evaluations[update]['launch']['sha256']))

    def test_success_is_new_schema_and_preserves_all_original_files(self):
        complete = self.run_export()
        self.assertEqual(complete['selected_update'], 200)
        self.assertTrue(self.backend.verified)
        self.assertEqual(self.backend.states_read, [100, 200, 300])
        self.assertEqual(self.backend.live_before_checkpoint, [[], [], [200]])
        self.assertEqual(self.backend.live_at_publication, [200])
        self.assertEqual(self.saved_record['schema'], x.EXPORT_SCHEMA)
        self.assertEqual(self.saved_record['selection_kind'], 'mixed_sim_v2_best')
        self.assertEqual(self.saved_record['binding'], self.binding)
        self.assertNotIn('observation', self.saved_record)
        self.assertNotIn('optimizer', self.saved_record)
        self.assertFalse(complete['head_training_completed'])
        after = {str(p.relative_to(self.training)): x.file_sha(p) for p in self.training.rglob('*') if p.is_file()}
        self.assertEqual(self.before, after)

    def test_original_terminal_failure_prevents_creation(self):
        self.original_verification_fails = True
        with self.assertRaisesRegex(ValueError, 'strict original'): self.run_export()
        self.assertFalse((self.root / 'new_export').exists())

    def test_actual_original_return_nonzero_prevents_creation(self):
        p = self.training / 'formal_return.json'; value = x.read(p); value['returncode'] = 1; self.write(p, value)
        with self.assertRaisesRegex(ValueError, 'actual successful'): self.run_export()

    def test_actual_original_return_false_is_not_integer_success(self):
        p = self.training / 'formal_return.json'; value = x.read(p); value['returncode'] = False; self.write(p, value)
        with self.assertRaisesRegex(ValueError, 'actual successful'): self.run_export()

    def test_existing_or_historical_output_rejected(self):
        for path in (self.training / 'new_export', self.training, self.root):
            with self.assertRaisesRegex(ValueError, 'fresh output'): self.run_export(path)

    def test_current_failure_precedes_old_success(self):
        self.write(self.training / 'formal/failure_attempt_003.json', {'failed': True})
        with self.assertRaisesRegex(ValueError, 'unresolved formal'): self.run_export()

    def test_missing_candidate_evaluation_rejected(self):
        del self.evaluations[300]
        with self.assertRaisesRegex(ValueError, 'complete frozen'): self.run_export()

    def test_changed_checkpoint_bytes_rejected(self):
        Path(self.candidate_value['candidates'][1]['checkpoint_file']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint identity'): self.run_export()

    def test_wrong_actual_model_hash_rejected(self):
        self.backend.tree_sha = lambda model: 'a' * 64
        with self.assertRaisesRegex(ValueError, 'actual candidate'): self.run_export()

    def test_actual_eval_returncode_and_launch_binding_required(self):
        for key, value in (('returncode', 1), ('launch_sha256', 'e' * 64)):
            with self.subTest(key=key):
                old = copy.deepcopy(self.evaluations[100]); path = Path(old['process_return']['path']); content = x.read(path)
                self.rewrite_eval(100, 'process_return', lambda r: r.__setitem__(key, value))
                with self.assertRaisesRegex(ValueError, 'actual evaluation return'): self.run_export()
                self.write(path, content); self.evaluations[100] = old

    def test_old_report_sha_binding_cannot_accept_modified_result(self):
        self.rewrite_eval(100, 'report', lambda r: r.__setitem__('role', 'test'))
        with self.assertRaisesRegex(ValueError, 'actual evaluation return'): self.run_export()
        self.assertEqual(self.backend.states_read, [])

    def test_cal_test_real_roles_never_export(self):
        self.rewrite_eval(100, 'launch', lambda r: r.__setitem__('role', 'real'))
        with self.assertRaisesRegex(ValueError, 'evaluation launch'): self.run_export()

    def test_committed_receipt_binding_rejected(self):
        self.backend.checkpoint_at = lambda root, u, b: ({'binding': b, 'sampling': {'completed_updates': u}}, {'path': 'foreign', 'sha256': 'b' * 64})
        with self.assertRaisesRegex(ValueError, 'committed checkpoint'): self.run_export()

    def test_native_revision_and_input_receipts_are_mandatory(self):
        original = x.read(self.evaluations[100]['launch']['path'])
        for key in ('native_evaluator_revision', 'materialized_select', 'source_inventory', 'evaluation_config'):
            with self.subTest(key=key):
                self.rewrite_launch(100, lambda r: r.pop(key))
                with self.assertRaisesRegex(ValueError, 'native evaluator revision|input path/SHA'): self.run_export()
                self.rewrite_launch(100, lambda r: r.update(copy.deepcopy(original)))
        self.assertEqual(self.backend.states_read, [])

    def test_changed_native_input_bytes_rejected(self):
        for key, ref in self.native_inputs.items():
            with self.subTest(key=key):
                path = Path(ref['path']); content = path.read_bytes(); path.write_bytes(content + b'\n')
                with self.assertRaisesRegex(ValueError, 'bound artifact changed'): self.run_export()
                path.write_bytes(content)

    def test_candidate_specific_materialization_source_or_config_rejected(self):
        for key, ref in self.native_inputs.items():
            with self.subTest(key=key):
                value = x.read(ref['path'])
                if key == 'materialized_select': value['entries'][0]['recipe'] = 'different_recipe'
                else: value['candidate_specific_change'] = True
                different = self.write(self.root / ('different_' + key + '.json'), value)
                self.rewrite_launch(200, lambda r: r.__setitem__(key, different))
                with self.assertRaisesRegex(ValueError, 'identical materialized SELECT/source/config'): self.run_export()
                self.rewrite_launch(200, lambda r: r.__setitem__(key, ref))
        self.assertEqual(self.backend.states_read, [])

    def test_identical_native_input_bytes_at_different_paths_are_accepted(self):
        for key, ref in self.native_inputs.items():
            duplicate = self.write(self.root / ('same_' + key + '.json'), x.read(ref['path']))
            self.assertEqual(duplicate['sha256'], ref['sha256'])
            self.rewrite_launch(200, lambda r: r.__setitem__(key, duplicate))
        self.assertEqual(self.run_export()['selected_update'], 200)

    def test_materialized_split_population_label_recipe_are_verified(self):
        mutations = (
            lambda r: r.__setitem__('split', 'test'),
            lambda r: r.__setitem__('real_used', True),
            lambda r: r.__setitem__('test_used', True),
            lambda r: r['entries'][0].__setitem__('pair_id', 'foreign'),
            lambda r: r['entries'][0].__setitem__('label', not r['entries'][0]['label']),
            lambda r: r['entries'][0].pop('recipe'),
        )
        ref = self.native_inputs['materialized_select']; path = Path(ref['path'])
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = copy.deepcopy(self.materialized_value); mutate(value)
                changed = self.write(path, value)
                self.rewrite_launch(100, lambda r: r.__setitem__('materialized_select', changed))
                with self.assertRaises(ValueError): self.run_export()
                self.write(path, self.materialized_value)
                self.rewrite_launch(100, lambda r: r.__setitem__('materialized_select', ref))
        self.assertEqual(self.backend.states_read, [])

    def test_materialized_actual_sample_hash_is_verified(self):
        sample = Path(self.materialized_value['entries'][0]['sample_path']); sample.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'actual SELECT sample bytes'): self.run_export()
        self.assertEqual(self.backend.states_read, [])

    def test_native_input_mutation_during_checkpoint_verification_blocks_publication(self):
        initial = self.backend.checkpoint_at
        for key, ref in self.native_inputs.items():
            with self.subTest(key=key):
                path = Path(ref['path']); content = path.read_bytes()
                def mutate(root, update, binding):
                    result = initial(root, update, binding)
                    if update == 300: path.write_bytes(content + b'\n')
                    return result
                self.backend.checkpoint_at = mutate
                with self.assertRaisesRegex(ValueError, 'input changed before'): self.run_export()
                self.assertFalse((self.root / 'new_export').exists())
                path.write_bytes(content)

    def test_sample_mutation_during_checkpoint_verification_blocks_publication(self):
        initial = self.backend.checkpoint_at
        sample = Path(self.materialized_value['entries'][0]['sample_path'])
        def mutate(root, update, binding):
            result = initial(root, update, binding)
            if update == 300: sample.write_bytes(b'changed')
            return result
        self.backend.checkpoint_at = mutate
        with self.assertRaisesRegex(ValueError, 'actual SELECT sample bytes'): self.run_export()
        self.assertFalse((self.root / 'new_export').exists())


if __name__ == '__main__': unittest.main()

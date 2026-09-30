import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dispatch_controls as q


class AssignmentsTests(unittest.TestCase):
    def test_two_old_lanes_exclude_b3_and_joint(self):
        q.validate_lanes(q.LANES)
        self.assertEqual({r['arm']: r['gpus'] for r in q.LANES}, {'B1': [1, 2], 'B2': [6, 7]})

    def test_modified_assignment_is_rejected(self):
        for devices in ([0, 5], [3, 4], [1, 1], [2, 1]):
            lanes = copy.deepcopy(q.LANES); lanes[0]['gpus'] = devices
            with self.assertRaises(ValueError):
                q.validate_lanes(lanes)

    def test_only_declared_tasks_have_paths(self):
        for arm in ('B1', 'B2'):
            for module in q.MODULES:
                spec, root = q.paths(arm, module)
                self.assertIn(arm.lower()+'_'+module, str(root))
                self.assertEqual(spec.name, 'execution.json')
        for arm, module in (('B3', 'matcher'), ('B0', 'matcher'), ('B1', 'residual')):
            with self.assertRaises(ValueError):
                q.paths(arm, module)

    def test_child_keeps_full_pipeline_no_budget_override(self):
        args = q.task_args('/spec', '/out', [1, 2]); cmd = q.command(args)
        self.assertEqual(cmd[-3:], ['--gpus', '1', '2'])
        self.assertEqual(cmd[2], q.PACKAGE+'.matcher_v2_v1.pipeline')
        self.assertNotIn('--gate-only', cmd)
        self.assertNotIn('--updates', cmd)
        self.assertIn(str(q.CANONICAL), cmd)

    def test_head_command_has_exactly_one_assigned_device(self):
        self.assertEqual(q.command(q.task_args('/spec', '/out', [7]))[-2:], ['--gpus', '7'])

    def test_environment_does_not_inherit_gpu_visibility(self):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,5'}):
            result = q.task_environment([6, 7])
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], '6,7')
        self.assertEqual(result['PYTHONPATH'], str(q.SOURCE))
        for devices in ([0], [5], [3, 4], []):
            with self.assertRaises(ValueError):
                q.task_environment(devices)


class EvidenceTests(unittest.TestCase):
    def test_binding_mutation_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data'; q.save(path, {})
            with patch.object(q, 'BINDINGS', {path: q.sha(path)}):
                q.check_bindings()
            with patch.object(q, 'BINDINGS', {path: '0'*64}), self.assertRaises(ValueError):
                q.check_bindings()

    def test_terminal_immutable_status_atomically_replaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'status.json'; q.save(path, {'step': 1}, replace=True)
            q.save(path, {'step': 2}, replace=True)
            self.assertEqual(q.read(path), {'step': 2})
            self.assertFalse(path.with_name('status.json.tmp').exists())
            with self.assertRaises(FileExistsError):
                q.save(path, {})

    def test_failed_current_resume_does_not_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            q.save(Path(tmp)/'failure.json', {})
            lane = dict(q.LANES[0], resume=tmp)
            previous = SimpleNamespace(old_release=Mock())
            with self.assertRaisesRegex(ValueError, 'predecessor training failed'):
                q.previous_release(lane, previous)
            previous.old_release.assert_not_called()

    def test_no_terminal_waits_instead_of_inferring_release(self):
        previous = SimpleNamespace(old_release=lambda lane: None)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(q.previous_release(dict(q.LANES[0], resume=tmp), previous))

    def test_wrong_prior_devices_or_formal_model_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); receipt = root/'release.json'
            q.save(receipt, dict(gpus='1,2', branches=[dict(formal_root='/wrong/formal_m12')]))
            proof = dict(kind='simple', gpus=[1, 2], receipt=str(receipt))
            previous = SimpleNamespace(old_release=lambda lane: proof)
            with self.assertRaisesRegex(ValueError, 'not the assigned old task'):
                q.previous_release(dict(q.LANES[0], resume=tmp), previous)
            proof['gpus'] = [6, 7]
            with self.assertRaisesRegex(ValueError, 'wrong predecessor'):
                q.previous_release(dict(q.LANES[0], resume=tmp), previous)

    def test_valid_previous_release_is_bound_to_its_actual_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); receipt = root/'release.json'
            q.save(receipt, dict(gpus='6,7', branches=[dict(formal_root=str(q.SIMPLE/'formal_scratch_fixed'))]))
            proof = dict(kind='simple', gpus=[6, 7], receipt=str(receipt))
            previous = SimpleNamespace(old_release=lambda lane: proof)
            self.assertEqual(q.previous_release(dict(q.LANES[1], resume=tmp), previous), proof)

    def test_failure_precedes_stale_pipeline_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); q.save(root/'complete.json', dict(status='complete'))
            nested = root/'training/formal'; nested.mkdir(parents=True)
            q.save(nested/'failure_attempt_01.json', {})
            with self.assertRaisesRegex(ValueError, 'failure precedes'):
                q.fail_first(root)

    def test_pipeline_reopens_actual_full_verifier_not_only_complete_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec = root/'execution.json'; q.save(spec, {})
            q.save(root/'complete.json', dict(status='complete', evaluation_complete_sha256='hash', completed_unix=2))
            args = q.task_args(spec, root, [1, 2])
            verify = Mock(return_value=dict(status='complete', evaluation_complete_sha256='hash', completed_unix=3))
            modules = dict(pipeline=SimpleNamespace(verify_all=verify, commands=lambda *a: 'bound_commands'),
                runtime_inputs=SimpleNamespace(load_inputs=lambda spec: {'plan': 'proof'}))
            self.assertEqual(q.verify_pipeline(args, modules), {'plan': 'proof'})
            verify.assert_called_once_with(root, args, {'plan': 'proof'}, 'bound_commands')
            verify.return_value['evaluation_complete_sha256'] = 'tampered'
            with self.assertRaises(ValueError):
                q.verify_pipeline(args, modules)

    def test_selected_matcher_requires_successful_export_and_same_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'training').mkdir(); (root/'exports').mkdir()
            q.save(root/'training/export_process_return.json', dict(returncode=0, export_root=str(root/'exports')))
            q.save(root/'exports/training_complete.json', dict(binding=dict(common_plan_sha256='plan')))
            selected = q.selected_from_terminal(root, {'plan': SimpleNamespace(sha256='plan')})
            self.assertEqual(selected['common_plan_sha256'], 'plan')
            with self.assertRaisesRegex(ValueError, 'another control plan'):
                q.selected_from_terminal(root, {'plan': SimpleNamespace(sha256='other')})


class HeadTests(unittest.TestCase):
    def compile_fixture(self, tmp, arm='B1'):
        root = Path(tmp); selected = dict(export_root='/exports/'+arm, process_return='/return', common_plan_sha256='plan')
        result = dict(arm=arm, module='scorer_patch', selected_matcher=selected, topology=dict(world_size=1, microbatch=32))
        compiler = SimpleNamespace(compile_execution=Mock(return_value=result))
        return selected, result, compiler, root/'locked/execution.json', root/'pipeline'

    def test_b1_head_keeps_extra_data_and_own_matcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected, _, compiler, spec, root = self.compile_fixture(tmp)
            with patch.object(q, 'paths', return_value=(spec, root)):
                q.compile_head('B1', 'scorer_patch', selected, compiler)
            compiler.compile_execution.assert_called_once_with(q.BASE, 'B1', 'scorer_patch', spec.parent,
                combined_admission=q.COMBINED, selected=selected)

    def test_b2_head_does_not_add_straight_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected, _, compiler, spec, root = self.compile_fixture(tmp, 'B2')
            with patch.object(q, 'paths', return_value=(spec, root)):
                q.compile_head('B2', 'scorer_patch', selected, compiler)
            self.assertIsNone(compiler.compile_execution.call_args.kwargs['combined_admission'])

    def test_wrong_arm_or_larger_global_batch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected, result, compiler, spec, root = self.compile_fixture(tmp)
            with patch.object(q, 'paths', return_value=(spec, root)):
                result['arm'] = 'B3'
                with self.assertRaises(ValueError):
                    q.compile_head('B1', 'scorer_patch', selected, compiler)
                result['arm'] = 'B1'; result['topology']['microbatch'] = 64
                with self.assertRaises(ValueError):
                    q.compile_head('B1', 'scorer_patch', selected, compiler)

    def test_existing_head_output_never_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected, _, compiler, spec, root = self.compile_fixture(tmp); root.mkdir()
            with patch.object(q, 'paths', return_value=(spec, root)), self.assertRaises(ValueError):
                q.compile_head('B1', 'scorer_patch', selected, compiler)
            compiler.compile_execution.assert_not_called()


class LaneLifecycleTests(unittest.TestCase):
    def test_waits_for_verified_release_then_matcher_then_two_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); lane = copy.deepcopy(q.LANES[0]); events = []
            selected = dict(export_root='/verified', process_return='/returned', common_plan_sha256='plan')
            def task(arm, module, gpus, root, modules, selection=None):
                events.append((module, gpus, selection))
                return dict(status='complete', selected=selected) if module == 'matcher' else dict(status='complete', module=module)
            launcher = SimpleNamespace(check_free=Mock())
            modules = dict(previous='bound_old_verifier', launcher=launcher)
            with patch.object(q, 'check_bindings'), patch.object(q, 'previous_release', side_effect=[None, {'proof': 'release'}]), \
                 patch.object(q.time, 'sleep') as sleep, patch.object(q, 'run_task', side_effect=task):
                result = q.run_lane(lane, out, modules)
            sleep.assert_called_once_with(60); launcher.check_free.assert_called_once_with([1, 2], 2)
            self.assertEqual(events[0], ('matcher', [1, 2], None))
            self.assertCountEqual(events[1:], [('scorer_patch', [1], selected), ('scorer_stats', [2], selected)])
            self.assertEqual(result['status'], 'control_arm_training_and_required_evaluations_complete')
            self.assertTrue((out/'B1/complete.json').exists())

    def test_matcher_failure_never_launches_heads_or_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); modules = dict(previous=Mock(), launcher=SimpleNamespace(check_free=Mock()))
            with patch.object(q, 'check_bindings'), patch.object(q, 'previous_release', return_value={'proof': 'release'}), \
                 patch.object(q, 'run_task', side_effect=ValueError('failed Matcher')) as task:
                with self.assertRaisesRegex(ValueError, 'failed Matcher'):
                    q.run_lane(q.LANES[1], out, modules)
            task.assert_called_once()
            self.assertTrue((out/'B2/failure.json').exists())
            self.assertFalse((out/'B2/complete.json').exists())

    def test_busy_released_device_blocks_new_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            modules = dict(previous=Mock(), launcher=SimpleNamespace(check_free=Mock(side_effect=ValueError('busy'))))
            with patch.object(q, 'check_bindings'), patch.object(q, 'previous_release', return_value={'proof': 'release'}), \
                 patch.object(q, 'run_task', side_effect=AssertionError('must not run')) as task:
                with self.assertRaisesRegex(ValueError, 'busy'):
                    q.run_lane(q.LANES[0], Path(tmp), modules)
            task.assert_not_called()

    def test_faster_resource_poll_not_allowed(self):
        with self.assertRaisesRegex(ValueError, 'interval'):
            q.run_lane(q.LANES[0], Path('/no-write'), {}, tick=1)

    def test_task_failure_saved_and_existing_children_not_signalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); failure = ValueError('bad binding')
            with patch.object(q, 'check_bindings', side_effect=failure):
                with self.assertRaisesRegex(ValueError, 'bad binding'):
                    q.run_task('B1', 'matcher', [1, 2], out, {})
            self.assertTrue((out/'matcher/failure.json').exists())

    def test_preparation_code_binding_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'proof'
            q.save(path, dict(status='passed', tests=24, errors=0, failures=0, skipped=0,
                source_sha256={'wrong': 'source'}, cuda_initialized=False))
            with self.assertRaisesRegex(ValueError, 'CPU control-queue'):
                q.verify_preparation(path)

    def test_fixtures_without_actual_runtime_preflight_do_not_authorize_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'proof'
            q.save(path, dict(status='passed', tests=28, errors=0, failures=0, skipped=0,
                source_sha256=q.own_code(), cuda_initialized=False, runtime_preflight=None))
            with self.assertRaisesRegex(ValueError, 'remote CPU import/plan'):
                q.verify_preparation(path)

    def test_successful_bound_runtime_preflight_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'proof'
            q.save(path, dict(status='passed', tests=28, errors=0, failures=0, skipped=0,
                source_sha256=q.own_code(), cuda_initialized=False,
                runtime_preflight=dict(status='passed', nvidia_or_gpu_probe_used=False,
                    immutable_bindings={str(p):v for p,v in q.BINDINGS.items()})))
            q.verify_preparation(path)

    def test_changed_runtime_binding_rejects_otherwise_successful_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'proof'
            q.save(path, dict(status='passed', tests=28, errors=0, failures=0, skipped=0,
                source_sha256=q.own_code(), cuda_initialized=False,
                runtime_preflight=dict(status='passed', nvidia_or_gpu_probe_used=False,
                    immutable_bindings={})))
            with self.assertRaisesRegex(ValueError, 'remote CPU import/plan'):
                q.verify_preparation(path)


if __name__ == '__main__':
    unittest.main()

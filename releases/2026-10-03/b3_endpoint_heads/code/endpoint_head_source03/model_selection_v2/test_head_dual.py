"""Only new dual-GPU routing guards; real resume/gradient admission runs on GPUs."""
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import head_endpoint as endpoint, head_launcher as launcher, head_pipeline as pipeline
from . import head_terminal as terminal, head_terminal_queue as queue, posthoc_export as ex


class DualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = dict(module='scorer_stats', topology=endpoint.DUAL_TOPOLOGY)
        self.path = self.root/'spec.json'; self.path.write_text(json.dumps(self.spec))
        self.args = SimpleNamespace(spec=self.path, gpu=None, gpus=[2, 3], preparation=None,
            gpu_admission=True, python=Path(sys.executable), canonical=self.root/'canonical',
            case_plan=self.root/'cases', out=self.root/'out', gate_only=False)

    def test_torchrun_two_ranks_preserves_actual_gate_and_resume(self):
        for stop, resume in ((12, False), (1, False), (12, True)):
            cmd = launcher.command('/python', self.path, self.root/'out', 2, 'gate', stop=stop, resume=resume)
            self.assertEqual(cmd[:9], ['/python', '-m', 'torch.distributed.run', '--standalone',
                '--nnodes=1', '--nproc_per_node=2', '--module', 'model_selection_v2.head_execution', 'train'])
            self.assertEqual('--resume' in cmd, resume)
        with self.assertRaises(ValueError):
            launcher.command('/python', self.path, self.root/'out', 2, 'formal')

    def test_exact_disjoint_gpu_pairs_and_rank_count(self):
        self.assertEqual(launcher.assigned_gpus(self.args, self.spec), [2, 3])
        for assignment in ([0, 1], [2, 2], [2], [True, 3], [3, 2]):
            with self.assertRaises(ValueError):
                launcher.assigned_gpus(SimpleNamespace(gpu=None, gpus=assignment), self.spec)
        self.assertEqual(launcher.assigned_gpus(SimpleNamespace(gpu=None, gpus=[0, 1]),
            dict(self.spec, module='scorer_patch')), [0, 1])

    def test_topology_request_preserves_global_batch32(self):
        request = dict(schema=endpoint.REQUEST_SCHEMA, arm='B3', matcher_update=31667,
            matcher_file_sha256=endpoint.MODEL_SHA, source_registry={}, validation_population_protocol={},
            user_instruction='Explicit two GPUs per head', train_labels_unchanged=True,
            task3_overlay_applied=False, head_topology=endpoint.DUAL_TOPOLOGY)
        endpoint.check_request(request)
        for change in ({'microbatch': 32}, {'world_size': 1}, {'accumulate': True}):
            with self.assertRaises(ValueError):
                endpoint.check_request(dict(request, head_topology=dict(endpoint.DUAL_TOPOLOGY, **change)))

    def test_direct_gpu_admission_cannot_be_used_for_other_topology(self):
        adoption = dict(head_topology=endpoint.DUAL_TOPOLOGY)
        with patch.object(endpoint, 'check'):
            self.assertIsNone(launcher.preparation(self.args, self.spec, dict(selected_matcher_adoption=adoption)))
            with self.assertRaises(ValueError):
                launcher.preparation(self.args, dict(self.spec, topology=dict(world_size=1)),
                                     dict(selected_matcher_adoption=adoption))

    def test_pipeline_uses_two_training_gpus_and_one_terminal_gpu(self):
        commands = pipeline.commands(self.args, self.root/'pipeline')
        train = commands['training']; evaluate = commands['terminal']
        self.assertEqual(train[train.index('--gpus')+1:train.index('--gpus')+3], ['2', '3'])
        self.assertIn('--gpu-admission', train); self.assertNotIn('--preparation', train)
        self.assertEqual(evaluate[evaluate.index('--gpu')+1], '2')
        self.assertEqual(queue.cuda_assignment([2, 3]), '2,3')

    def test_busy_pair_does_not_launch_or_create_output(self):
        support = SimpleNamespace(check_free=lambda *_: (_ for _ in ()).throw(ValueError('busy')))
        with patch.object(launcher.execution, 'load_inputs', return_value=(dict(self.spec, runtime='/runtime'), {})), \
             patch.object(launcher, 'preparation', return_value=None), \
             patch.object(launcher, 'module', return_value=support):
            with self.assertRaisesRegex(ValueError, 'busy'): launcher.run(self.args)
        self.assertFalse(self.args.out.exists())

    def test_actual_return_requires_exact_two_rank_command(self):
        command = launcher.command('/python', self.path, self.root/'out', 2, 'formal', gate=self.root/'gate')
        launch = self.root/'formal_launch.json'; returned = self.root/'formal_return.json'
        launch.write_text(json.dumps(dict(phase='formal', command=command)))
        returned.write_text(json.dumps(dict(phase='formal', returncode=0, launch_sha256=ex.file_sha(launch))))
        terminal.returned_phase(self.root, 'formal', 'model_selection_v2.head_execution', 'train',
                                {'--mode': 'formal'}, world=2)
        with self.assertRaises(ValueError):
            terminal.returned_phase(self.root, 'formal', 'model_selection_v2.head_execution', 'train', {}, world=1)


if __name__ == '__main__':
    unittest.main()

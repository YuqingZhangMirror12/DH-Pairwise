from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from . import pipeline


class PipelineTests(unittest.TestCase):
    def test_separate_new_training_and_terminal_entries_same_explicit_devices(self):
        args = SimpleNamespace(spec='/s', preparation='/p', python='/python', gpus=[3, 4],
            canonical_straight='/data', case_plan='/cases')
        a, b = pipeline.commands(args, '/task', 2)
        self.assertIn(pipeline.__package__+'.launcher', a)
        self.assertIn(pipeline.__package__+'.evaluation_controller', b)
        self.assertEqual(a[-2:], ['3', '4']); self.assertEqual(b[-2:], ['3', '4'])
        self.assertEqual(b[b.index('--controller-root')+1], '/task/training')
        with self.assertRaises(ValueError):pipeline.commands(args, '/task', 1)

    def test_process_exit_and_actual_command_are_both_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); values = ['/python', '-m', 'fake.CPU.test']
            launch = root/'training_launch.json'; returned = root/'training_return.json'
            write_json(launch, dict(phase='training', command=values))
            record = dict(phase='training', returncode=0, launch_sha256=file_sha(launch))
            write_json(returned, record)
            self.assertIn('return_sha256', pipeline.verify_child(root, 'training', values))
            with self.assertRaises(ValueError):pipeline.verify_child(root, 'training', values+['--altered'])
            write_json(returned, dict(record, returncode=7), replace=True)
            with self.assertRaises(ValueError):pipeline.verify_child(root, 'training', values)


if __name__ == '__main__':unittest.main()

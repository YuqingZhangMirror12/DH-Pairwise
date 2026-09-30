import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import launch_training as launch
from recovery_contract import ROOT, FAILED


class RecoveryRoutingTests(unittest.TestCase):
    def contracts(self):
        path = Path(__file__).parent / 'evaluation_queue/contracts.py'
        spec = importlib.util.spec_from_file_location('recovery_route_fixture', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_root_does_not_overwrite_failed_attempt(self):
        self.assertEqual(launch.FORMAL_ROOT, ROOT)
        self.assertEqual(ROOT.parent, FAILED)
        self.assertNotEqual(ROOT, FAILED)

    def test_all_joint_eval_commands_use_repaired_root_source_and_receipt(self):
        module = self.contracts()
        for choice in ('sim', 'real'):
            for split in module.SPLITS:
                command = module.evaluator_command('python', 'joint', choice, split, '/out')
                self.assertEqual(command[command.index('--root') + 1], str(ROOT))
                self.assertTrue(command[command.index('--joint-source') + 1].endswith('/training_source_03'))
                self.assertTrue(command[command.index('--preparation') + 1].endswith('/evaluation_source_03/preparation.json'))

    def test_frozen_comparator_is_not_retrained_or_redirected(self):
        module = self.contracts()
        args = module.args_for('frozen', '/out')
        self.assertEqual(args.root, module.CONTROL)
        self.assertEqual(args.real_selection, ROOT / 'postprocess_joint_01/control_reselection/real_selection.json')
        self.assertNotIn(('frozen', 'sim', 'turufan'), module.TASKS)

    def test_registered_failure_root_cannot_be_relaunched(self):
        argv = ['launch_training.py', '--root', str(FAILED), '--gpus', '3,4', '--release-evaluation', '/not-opened']
        with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'unregistered'):
            launch.main()


def load_tests(loader, tests, pattern):
    # Re-exercise the seven existing gate contracts against THIS copied launcher.
    # This is new routing verification, not a claim that GPU gates ran.
    folder = Path(__file__).parent.parent / 'threshold_joint_v1'
    if not folder.exists():
        folder = Path('/root/autodl-tmp/consensus_threshold_joint_20260927/threshold_joint_v1')
    package = types.ModuleType('recovery_gate_fixtures'); package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    sys.modules[package.__name__ + '.launch_training'] = launch
    tests.addTests(loader.loadTestsFromName(package.__name__ + '.test_launch_training'))
    return tests

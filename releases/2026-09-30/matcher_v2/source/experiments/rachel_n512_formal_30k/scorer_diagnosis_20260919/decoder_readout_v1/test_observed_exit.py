"""Read-only Linux exit-evidence admission, using synthetic temporary records."""
import json
from pathlib import Path
import tempfile
import unittest

import analyze_factor_sweep as a


def stat(pid, parent, start, state='Z', code=0):
    fields = ['0'] * 50
    fields[0], fields[1], fields[19], fields[49] = state, str(parent), str(start), str(code)
    return str(pid)+' (python worker) '+' '.join(fields)+'\n'


class TerminalEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.model = 'binary_patch'
        (self.root/'evidence_inputs').mkdir()
        self.folder = self.root/'full_development_01'/self.model; self.folder.mkdir(parents=True)
        self.complete = self.folder/'complete.json'; self.complete.write_text('{}')
        self.controller = dict(pid=5, starttime=80, cmdline='python factor_controller.py --driver')
        (self.root/'controller_launch.json').write_text(json.dumps(self.controller))
        self.identity = dict(pid=6, starttime=90, cmdline='python run_factor_sweep.py')
        self.launch = self.root/('full_development_01_'+self.model+'_launch.json')
        self.launch.write_text(json.dumps(dict(identity=self.identity)))
        self.proof = dict(kind='linux_zombie_exit', launch_sha256=a.sha(self.launch),
            complete_sha256=a.sha(self.complete), identity=self.identity,
            raw_proc_stat=stat(6, 5, 90), wait_status=0, returncode=0)
        self.observation = dict(schema='frozen-factor-terminal-observation/1', read_only=True,
            observed_at='2026-09-29T05:20:00Z', boot_id='synthetic-test-only',
            controller=self.controller, parent_stat=stat(5, 1, 80, 'S'),
            parent_cmdline=self.controller['cmdline'], models={self.model:self.proof})

    def check(self):
        (self.root/'evidence_inputs/terminal_observation.json').write_text(json.dumps(self.observation))
        return a.verify_worker_exit(self.root, self.model)

    def test_bound_zero_exit_is_accepted_without_reaping(self):
        result = self.check()
        self.assertEqual(result['kind'], 'independently_observed_linux_exit')
        self.assertTrue(result['parent_receipt_pending'])

    def test_live_worker_is_not_complete(self):
        self.proof['raw_proc_stat'] = stat(6, 5, 90, 'R')
        with self.assertRaisesRegex(ValueError, 'live or identity'): self.check()

    def test_nonzero_or_signalled_exit_rejected(self):
        for code in (256, 9, 15):
            self.proof['raw_proc_stat'] = stat(6, 5, 90, code=code)
            with self.assertRaisesRegex(ValueError, 'successfully'): self.check()

    def test_pid_reuse_or_wrong_parent_rejected(self):
        for raw in (stat(7, 5, 90), stat(6, 5, 91), stat(6, 4, 90)):
            self.proof['raw_proc_stat'] = raw
            with self.assertRaisesRegex(ValueError, 'identity'): self.check()

    def test_changed_complete_file_rejected(self):
        self.complete.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'another launch or output'): self.check()

    def test_parent_identity_rejected(self):
        self.observation['parent_stat'] = stat(5, 1, 81, 'S')
        with self.assertRaisesRegex(ValueError, 'parent identity'): self.check()

    def test_controller_failure_exit_overrides_zero_observation(self):
        path = self.root/('full_development_01_'+self.model+'_exit.json')
        path.write_text('{"returncode":1}')
        with self.assertRaisesRegex(ValueError, 'successfully'): self.check()

    def test_complete_file_alone_is_insufficient(self):
        with self.assertRaisesRegex(ValueError, 'evidence absent'):
            a.verify_worker_exit(self.root, self.model)


if __name__ == '__main__':
    unittest.main()

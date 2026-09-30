import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ..curriculum_training_v1.runtime_io import read
from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_scorer_eval_v1 import controller as shared
from . import evaluation_controller as api


class EvaluationControllerTests(unittest.TestCase):
    def test_fixed_final_registry_and_distinct_entry(self):
        args = SimpleNamespace(spec='/s', preparation='/p', controller_root='/c')
        for module in ('matcher', 'scorer_patch', 'scorer_stats'):
            jobs = api.jobs(module)
            self.assertEqual(len(jobs), 12); self.assertEqual(len({j['name'] for j in jobs}), 12)
            self.assertEqual({j['split'] for j in jobs}, set(api.SPLITS))
            if module == 'matcher':self.assertNotIn('real_best', {j['selection'] for j in jobs})
            else:self.assertNotIn('equal_budget_endpoint', {j['selection'] for j in jobs})
            values = api.command('/python', args, '/plan', '/out', jobs[0], module)
            self.assertIn(api.__package__+'.evaluate', values); self.assertNotIn('--order', values)
        with self.assertRaises(ValueError):api.jobs('unregistered_head')

    def test_explicit_registered_jobs_actual_short_CPU_children(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); checks = []; registered = api.jobs('matcher')[:2]
            def verify(root, job):
                self.assertEqual(read(root/(job['name']+'_return.json'))['returncode'], 0)
                return dict(job=job)
            with patch.object(shared, 'identity', side_effect=lambda pid:dict(pid=pid, synthetic_identity=True)):
                done = shared.execute_queue(root, [0, 5], dict(os.environ, PYTHONPATH=str(root)),
                    lambda job, out:[sys.executable, '-c', 'print("explicit CPU queue fixture")'],
                    free_check=lambda *args:checks.append(args), verify=verify, registered_jobs=registered)
            self.assertEqual(len(done), 2); self.assertEqual(len(checks), 2)
            self.assertEqual({p.name for p in root.glob('*_return.json')}, {j['name']+'_return.json' for j in registered})

    def test_nonzero_return_never_reaches_artifact_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); job = api.jobs('matcher')[0]; name = job['name']
            launch = root/(name+'_launch.json'); write_json(launch, dict(job=job, command=[]))
            write_json(root/(name+'_return.json'), dict(returncode=7, launch_sha256=file_sha(launch)))
            write_json(root/'controller_launch.json', {})
            with patch.object(api, 'verify_native') as audit, self.assertRaisesRegex(ValueError, 'child return'):
                api.verify_job(root, job, 'matcher')
            audit.assert_not_called()


if __name__ == '__main__':unittest.main()

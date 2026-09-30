import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training import prepare_all as p


class PrepareTests(unittest.TestCase):
    def test_finite_stage_plan_and_worker_count(self):
        plan = p.stages(Path('/tmp/new'), 'python', 4)
        self.assertEqual([s['name'] for s in plan], ['train_val', 'test', 'real', 'ood'])
        self.assertTrue(all(s['command'][-2:] == ['--workers','4'] for s in plan))
        self.assertTrue(all('--probe' not in s['command'] for s in plan))

    def test_complete_cpu_flow_no_real_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'out'
            args = SimpleNamespace(output_root=str(root), source_root=str(Path(p.__file__).resolve().parents[4]), workers=4)
            with patch.object(p.subprocess,'Popen') as popen, patch.object(p,'register') as reg:
                popen.return_value.pid = 123
                popen.return_value.wait.return_value = 0
                reg.side_effect = lambda s: dict(bundle=s['output']+'/bundle.json',sha256='a'*64)
                result = p.run(args)
                self.assertEqual(result['completed_stages'], 4)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(popen.call_count, 4)
                self.assertTrue(all(c.kwargs['env']['CUDA_VISIBLE_DEVICES']=='' for c in popen.call_args_list))
                self.assertEqual(set(json.loads((root/'endpoint_registry.json').read_text())), {'test','real','ood'})
            with self.assertRaises(FileExistsError):
                p.run(args)

    def test_stops_after_first_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output_root=str(Path(tmp)/'out'),source_root=str(Path(p.__file__).resolve().parents[4]),workers=1)
            with patch.object(p.subprocess,'Popen') as popen:
                popen.return_value.pid=123
                popen.return_value.wait.return_value=7
                with self.assertRaisesRegex(RuntimeError,'no automatic retry'):
                    p.run(args)
                self.assertEqual(popen.call_count,1)


if __name__ == '__main__':
    unittest.main()

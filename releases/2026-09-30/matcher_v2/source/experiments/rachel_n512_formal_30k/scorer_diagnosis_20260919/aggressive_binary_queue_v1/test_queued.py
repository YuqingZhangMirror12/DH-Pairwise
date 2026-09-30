from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import run_queued as worker
import register


class QueueTests(unittest.TestCase):
    def test_priority_uses_actual_dispatches_not_stale_aggregate_status(self):
        import json
        common=Mock();common.read.side_effect=lambda p:json.loads(Path(p).read_text());common.live.return_value=True
        with tempfile.TemporaryDirectory() as t,patch.object(worker,'QUEUE',Path(t)):
            root=Path(t);(root/'launches').mkdir()
            for index,name in enumerate(('binary_patch','binary_stats')):
                work=root/name;work.mkdir()
                (root/'launches'/('0'+str(index)+'_'+name+'.json')).write_text(json.dumps(dict(operation=name,work=str(work),identity={'pid':index})))
            worker.check_dispatched(common) # no combined driver_status exists yet
            common.live.return_value=False
            with self.assertRaises(ValueError):worker.check_dispatched(common)
            for name in ('binary_patch','binary_stats'):(root/name/'complete.json').write_text('{"status":"complete"}')
            worker.check_dispatched(common)
            (root/'launches/joint_e32.json').write_text('{}')
            with self.assertRaises(ValueError):worker.check_dispatched(common)

    def test_failed_lightweight_task_cannot_release_future_work(self):
        import json
        common=Mock();common.read.side_effect=lambda p:json.loads(Path(p).read_text());common.live.return_value=True
        with tempfile.TemporaryDirectory() as t,patch.object(worker,'QUEUE',Path(t)):
            root=Path(t);(root/'launches').mkdir();work=root/'patch';work.mkdir()
            (work/'failure.json').write_text('{}')
            (root/'launches/01_binary_patch.json').write_text(json.dumps(dict(operation='binary_patch',work=str(work),identity={'pid':1})))
            with self.assertRaisesRegex(ValueError,'failed'):worker.check_dispatched(common)

    def fixture(self):
        return dict(status='complete',all_six_populations_verified=True,fixed_case_evaluations=22,
            jobs=[dict(selection_kind=c,split=s,status='complete',returncode=0,verified={'status':'passed'})
                  for c in ('sim','real') for s in ('sim_test_aggressive','dunhuang_cv','turufan')])

    def test_completion_requires_all_six_new_populations_and_numeric_proofs(self):
        result=self.fixture();worker.verified_evaluation(result)
        for case in ('missing','duplicate','old_test','failed','unverified','cases'):
            bad=deepcopy(result)
            if case=='missing':bad['jobs'].pop()
            elif case=='duplicate':bad['jobs'][1]=bad['jobs'][0]
            elif case=='old_test':bad['jobs'][0]['split']='sim_test_v14'
            elif case=='failed':bad['jobs'][0]['returncode']=1
            elif case=='unverified':bad['jobs'][0]['verified']['status']='pending'
            else:bad['fixed_case_evaluations']=0
            with self.assertRaises(ValueError):worker.verified_evaluation(bad)

    def test_missing_or_failed_data_never_admits_or_starts_gpu(self):
        common=Mock();common.read.side_effect=FileNotFoundError('not ready')
        with tempfile.TemporaryDirectory() as t,patch.object(register,'DATA',Path(t)):
            with self.assertRaises(FileNotFoundError):register.ready_data(common)
            (Path(t)/'pipeline_failure.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'failure'):register.ready_data(common)
        common.launch_wait.assert_not_called()

    def test_full_data_terminal_hashes_and_admission_are_required(self):
        common=Mock();common.sha.return_value='actual'
        common.read.return_value=dict(status='complete',pairs=30000,data_contract_sha256='actual',geometry_sha256='actual')
        admission=Mock();admission.validate_data.return_value={'passed':True}
        with tempfile.TemporaryDirectory() as t,patch.object(register,'DATA',Path(t)),patch.object(register,'load',return_value=admission):
            self.assertEqual(register.ready_data(common),{'passed':True});admission.validate_data.assert_called_once()
            common.read.return_value['geometry_sha256']='old'
            with self.assertRaises(ValueError):register.ready_data(common)


if __name__=='__main__':unittest.main()

"""Recompute synthetic real-head artifacts and run actual short CPU children."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ..curriculum_training_v1.checkpoint_io import file_sha,write_json
from ..curriculum_training_v1.runtime_io import read
from . import controller, test_evaluate as fixture
from .audit import verify_population


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.helper=fixture.InferenceTests()
        self.helper.run_fixture(self.root,'patch')
        self.out=self.root/'out'

    def test_actual_prediction_metrics_and_mlp_snapshot_recompute(self):
        audit=verify_population(self.out)
        self.assertEqual(audit['pairs'],6);self.assertEqual(audit['diagnostic_cases'],1)
        self.assertTrue(audit['actual_rows_and_numeric_evidence_recomputed'])
        self.assertTrue(audit['process_return_must_be_checked_separately'])

    def test_failure_overrides_success(self):
        write_json(self.out/'failure.json',dict(synthetic=True))
        with self.assertRaisesRegex(ValueError,'failure'):verify_population(self.out)

    def test_summary_change_even_with_new_hash_cannot_pass(self):
        summary=read(self.out/'summary.json');summary['groups']['real_test']['primary']['f1']=.1234567
        write_json(self.out/'summary.json',summary,replace=True)
        complete=read(self.out/'evaluation_complete.json')
        complete['files']['summary.json']=complete['summary_sha256']=file_sha(self.out/'summary.json')
        write_json(self.out/'evaluation_complete.json',complete,replace=True)
        with self.assertRaisesRegex(ValueError,'summary differs'):verify_population(self.out)

    def test_missing_or_changed_prediction_file_fails(self):
        with (self.out/'pair_predictions.jsonl').open('a') as f:f.write('{}\n')
        with self.assertRaisesRegex(ValueError,'artifact'):verify_population(self.out)

    def test_snapshot_sidecar_not_just_json_flags_is_verified(self):
        case=read(self.out/'diagnostic_index.json')['cases'][0]
        evidence=self.out/case['evidence'];metadata=read(evidence)
        side=evidence.parent/metadata['sidecar']['path']
        side.write_bytes(side.read_bytes()+b'corrupt')
        with self.assertRaisesRegex(ValueError,'sidecar'):verify_population(self.out)

    def test_population_metadata_is_not_unbound(self):
        value=read(self.out/'population.json');value['pair_ids'].reverse()
        write_json(self.out/'population.json',value,replace=True)
        complete=read(self.out/'evaluation_complete.json');complete['files']['population.json']=file_sha(self.out/'population.json')
        write_json(self.out/'evaluation_complete.json',complete,replace=True)
        with self.assertRaisesRegex(ValueError,'population'):verify_population(self.out)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()

    def test_only_six_registered_jobs_no_new_endpoint_or_select_inference(self):
        jobs=controller.jobs();self.assertEqual(len(jobs),6)
        self.assertEqual(len({j['name'] for j in jobs}),6)
        self.assertEqual({j['selection'] for j in jobs},{'sim_best','real_best'})
        self.assertEqual({j['split'] for j in jobs},{'sim_test','dunhuang_cv','turufan'})

    def test_commands_bind_explicit_source_population_model_and_output(self):
        args=SimpleNamespace(spec=self.root/'spec',preparation=self.root/'prepared',controller_root=self.root/'training',
            common_source=self.root/'common',binary_source=self.root/'binary')
        job=controller.jobs()[0]
        values=controller.command('/python',args,self.root/'population',self.root/job['name'],job)
        for flag,value in [('--selection',job['selection']),('--split','sim_test'),('--out',str(self.root/job['name'])),
                           ('--common-source',str(args.common_source)),('--device','cuda:0')]:
            self.assertEqual(values[values.index(flag)+1],value)
        with self.assertRaisesRegex(ValueError,'unregistered'):
            controller.command('/python',args,self.root/'population',self.root/'out',dict(job,selection='equal_budget_endpoint'))

    def test_six_actual_short_cpu_children_complete_without_gpu_calls(self):
        checks=[];verified=[]
        def free(gpus,world):checks.append((gpus,world))
        def verify(root,job):
            value=read(root/(job['name']+'_return.json'));self.assertEqual(value['returncode'],0)
            self.assertFalse(value['automatic_retry']);verified.append(job['name']);return dict(job=job)
        values=lambda job,out:[sys.executable,'-c','import os; print(os.environ["CUDA_VISIBLE_DEVICES"])']
        env=dict(os.environ,PYTHONPATH=str(self.root))
        with patch.object(controller,'identity',side_effect=lambda pid:dict(pid=pid,synthetic_identity=True)):
            done=controller.execute_queue(self.root,[0,5],env,values,free_check=free,verify=verify)
        self.assertEqual(len(done),6);self.assertEqual(len(checks),6)
        self.assertEqual(set(verified),{j['name'] for j in controller.jobs()})
        for job in controller.jobs():
            launch=read(self.root/(job['name']+'_launch.json'))
            self.assertIn(launch['gpu'],[0,5])
            self.assertEqual((self.root/(job['name']+'.log')).read_text().strip(),str(launch['gpu']))

    def test_failed_child_stops_dispatch_no_retry(self):
        calls=[]
        def values(job,out):calls.append(job);return [sys.executable,'-c','raise SystemExit(7)']
        with patch.object(controller,'identity',side_effect=lambda pid:dict(pid=pid,synthetic_identity=True)), \
                self.assertRaisesRegex(ValueError,'incomplete'):
            controller.execute_queue(self.root,[0],dict(os.environ,PYTHONPATH=str(self.root)),values,
                free_check=lambda *a:None,verify=lambda *a:self.fail('failed process cannot be verified'))
        self.assertEqual(len(calls),1)
        status=read(self.root/'driver_status.json');self.assertEqual(status['status'],'draining_after_failure')
        self.assertEqual(len(status['pending']),5)

    def test_nonzero_return_is_not_accepted_as_completed_output(self):
        job=controller.jobs()[0];launch=self.root/(job['name']+'_launch.json')
        write_json(launch,dict(job=job,command=[]))
        write_json(self.root/(job['name']+'_return.json'),dict(returncode=9,launch_sha256=file_sha(launch)))
        with self.assertRaisesRegex(ValueError,'success'):controller.verify_job(self.root,job)

    def test_occupied_card_prevents_child_launch(self):
        def busy(*args):raise ValueError('occupied')
        with patch.object(controller.subprocess,'Popen') as process,self.assertRaisesRegex(ValueError,'occupied'):
            controller.execute_queue(self.root,[0],dict(os.environ,PYTHONPATH=str(self.root)),lambda *a:[],free_check=busy)
        process.assert_not_called()


if __name__=='__main__':unittest.main()

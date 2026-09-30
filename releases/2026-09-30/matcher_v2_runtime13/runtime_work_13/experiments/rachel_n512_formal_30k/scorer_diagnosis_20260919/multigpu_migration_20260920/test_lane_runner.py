import copy
import unittest
from lane_runner import require_subset, validate_plan, validate_stage, wrapped_command


class LaneTests(unittest.TestCase):
    def setUp(self):
        self.stage=dict(name='train',command=['/python','-m','matched_only.train','--device','cuda:0'],
                        cwd='/source',completion='/out/status.json',completion_expect={'status':'complete'})
        self.plan=dict(schema='seven-gpu-migration/1',output_root='/migration',
                       device_wrapper='/wrapper.py',lanes=[dict(name='lane'+str(i),gpu_uuid='GPU-'+str(i),
                                                              stages=[copy.deepcopy(self.stage)]) for i in range(7)])

    def test_distinct_lanes(self):
        self.assertIs(validate_plan(self.plan),self.plan)
        self.plan['lanes'][6]['gpu_uuid']='GPU-0'
        with self.assertRaises(ValueError): validate_plan(self.plan)

    def test_gpu_wrapper_preserves_leaf_args(self):
        command=wrapped_command(self.stage,self.plan['lanes'][3],self.plan)
        self.assertEqual(command[-3:],['--','--device','cuda:0'])
        self.assertIn('GPU-3',command)
        self.assertIn('matched_only.train',command)

    def test_cpu_bypasses_wrapper(self):
        self.stage['gpu']=False
        self.assertEqual(wrapped_command(self.stage,self.plan['lanes'][0],self.plan),self.stage['command'])

    def test_old_supervisors_rejected(self):
        for module in ['matched_only.priority_supervisor','candidate_local.queue','x.after_priority']:
            self.stage['command'][2]=module
            with self.assertRaises(ValueError): validate_stage(self.stage)

    def test_completion_is_not_file_existence(self):
        require_subset({'status':'complete','nested':{'count':6000}}, {'status':'complete','nested':{'count':6000}})
        for receipt in [{'status':'running'}, {'status':'complete','nested':{'count':32}}]:
            with self.assertRaises(ValueError): require_subset(receipt,{'status':'complete','nested':{'count':6000}})


if __name__=='__main__': unittest.main()

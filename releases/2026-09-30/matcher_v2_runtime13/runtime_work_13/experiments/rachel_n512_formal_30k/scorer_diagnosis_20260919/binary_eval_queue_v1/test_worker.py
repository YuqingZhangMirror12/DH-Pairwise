import importlib
import os
import unittest
from worker import trace_check

class WorkerTests(unittest.TestCase):
    def test_production_size_actual_trace_on_cpu(self):
        training=importlib.import_module('experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.train')
        variant=os.environ['BINARY_VERIFY_VARIANT'];r=trace_check(training,variant,'cpu')
        self.assertEqual(r['status'],'passed',r)
        self.assertEqual(r['head_parameters'],34529 if variant=='patch' else 3201)
        self.assertEqual(r['optimizer_updates'],0);self.assertFalse(r['real_inference_performed'])
        self.assertTrue(r['capture_bitwise_equal']);self.assertTrue(r['rng_unchanged'])
        self.assertTrue(r['parameters_unchanged']);self.assertTrue(r['numeric_replay_passed'])

if __name__=='__main__':unittest.main()

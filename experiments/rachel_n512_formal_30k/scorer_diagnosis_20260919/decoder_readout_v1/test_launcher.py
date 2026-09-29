import copy
import unittest
from unittest.mock import patch
import launch_priority as l


class LaunchTests(unittest.TestCase):
    def valid(self):
        return dict(status='passed',formal_training=False,updated_weights_discarded=True,arm='scratch_fixed',
            stage='scorer',updates=12,exposures=384,world_size=1,microbatch=32,accumulate=1,effective_batch=32,
            matcher_unchanged=True,head_updated=True,binding={'synthetic':1},model_state_hashes=['trained'],
            initial_head_sha256='initial',final_head_sha256='trained',resume_matches_uninterrupted=True)
    def test_gate_and_replay_exact_requirements(self):
        r=self.valid();l.validate_gate(r,r['binding']);l.validate_gate(r,r['binding'],True)
        for key in ('matcher_unchanged','head_updated','updated_weights_discarded','resume_matches_uninterrupted'):
            x=copy.deepcopy(r);x[key]=False
            with self.assertRaises(ValueError):l.validate_gate(x,r['binding'],True)
    def test_gate_wrong_batch_budget_source_rejected(self):
        r=self.valid()
        for key,value in [('updates',1),('exposures',32),('microbatch',16),('world_size',2),('binding',{})]:
            x=copy.deepcopy(r);x[key]=value
            with self.assertRaises(ValueError):l.validate_gate(x,r['binding'])
    def test_zero_update_or_multiple_hashes_rejected(self):
        r=self.valid()
        for key,value in [('final_head_sha256','initial'),('model_state_hashes',['a','b'])]:
            x=copy.deepcopy(r);x[key]=value
            with self.assertRaises(ValueError):l.validate_gate(x,r['binding'])
    def test_fresh_formal_never_imports_gate_optimizer_or_resume(self):
        for v in l.GPUS:
            c=l.command(v,l.ROOT/('formal_'+v))
            self.assertNotIn('--resume',c);self.assertNotIn('--preflight-steps',c)
            self.assertIn('--nproc_per_node=1',c);self.assertIn(l.E32_SHA,c)
            self.assertEqual(c[c.index('--scorer-variant')+1],v)
    def test_gate_replay_is_only_explicit_resume(self):
        c=l.command('patch_mean',l.ROOT/'gate_patch_mean',True,True)
        self.assertIn('--resume',c);self.assertEqual(c[c.index('--preflight-steps')+1],'12')
    def test_busy_gpu_never_claimed(self):
        with patch.object(l.subprocess,'check_output',side_effect=['3, GPU-three\n4, GPU-four\n','GPU-three\n']):
            with self.assertRaises(ValueError):l.free('3')
    def test_unoccupied_registered_gpu(self):
        with patch.object(l.subprocess,'check_output',side_effect=['3, GPU-three\n4, GPU-four\n','GPU-four\n']):
            self.assertEqual(l.free('3'),'GPU-three')
    def test_pid_reuse_detected(self):
        a=dict(pid=1,starttime=22,cmdline='python train');b=dict(a,starttime=23)
        self.assertFalse(l.same_process(a,b));self.assertTrue(l.same_process(a,dict(a)))


if __name__=='__main__':unittest.main()

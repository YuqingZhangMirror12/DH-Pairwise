from copy import deepcopy
import unittest
from .launch_training import validate_gate,validate_release,validate_evaluation_preparation


class GateTests(unittest.TestCase):
    def setUp(self):
        self.binding={'source':'source','data':'data'}
        self.receipt=dict(status='passed',formal_training=False,updated_weights_discarded=True,
            arm='scratch_joint',stage='scorer',updates=12,exposures=384,world_size=2,
            microbatch=8,accumulate=2,effective_batch=32,matcher_unchanged=False,
            cached_proposals=False,old_unused_heads_unchanged=True,model_state_hashes=['sha','sha'],
            binding=self.binding,resume_matches_uninterrupted=True,
            matcher_change=dict(changed_tensors=10,relative_l2=1e-5,max_absolute=1e-5,unused_changed=[]),
            head_change=dict(changed_tensors=5,relative_l2=1e-3,max_absolute=1e-3,unused_changed=[]),
            gradient_l2_max=dict(scorer=1.,matcher=.1))
    def test_valid_joint_and_replay(self):
        validate_gate(self.receipt,self.binding,'scratch_joint')
        validate_gate(self.receipt,self.binding,'scratch_joint',replay=True)
    def test_reject_frozen_cache_wrong_protocol(self):
        for key,value in [('matcher_unchanged',True),('cached_proposals',True),('old_unused_heads_unchanged',False),
                          ('updates',0),('world_size',3),('effective_batch',16),('updated_weights_discarded',False)]:
            receipt=deepcopy(self.receipt);receipt[key]=value
            with self.assertRaises(ValueError):validate_gate(receipt,self.binding,'scratch_joint')
    def test_reject_no_update_or_unintended_parameter_update(self):
        for key,value in [('changed_tensors',0),('relative_l2',0),('max_absolute',float('nan')),('unused_changed',['base.coarse.weight'])]:
            receipt=deepcopy(self.receipt);receipt['matcher_change'][key]=value
            with self.assertRaises(ValueError):validate_gate(receipt,self.binding,'scratch_joint')
    def test_reject_no_gradient_or_nonfinite(self):
        for value in (0.,float('inf'),float('nan')):
            receipt=deepcopy(self.receipt);receipt['gradient_l2_max']['matcher']=value
            with self.assertRaises(ValueError):validate_gate(receipt,self.binding,'scratch_joint')
    def test_reject_bad_hash_binding_and_resume(self):
        for key,value in [('model_state_hashes',['a','b']),('binding',{}),('resume_matches_uninterrupted',False)]:
            receipt=deepcopy(self.receipt);receipt[key]=value
            with self.assertRaises(ValueError):validate_gate(receipt,self.binding,'scratch_joint',replay=True)
    def test_released_branch_requires_three_successful_evaluations(self):
        receipt=dict(status='complete',jobs=[dict(split=s,status='complete',returncode=0) for s in ('sim_test_v14','dunhuang_cv','turufan')])
        validate_release(receipt)
        receipt['jobs'][0]['returncode']=1
        with self.assertRaises(ValueError):validate_release(receipt)
        receipt['jobs'].pop()
        with self.assertRaises(ValueError):validate_release(receipt)

    def test_joint_launch_requires_tested_joint_and_frozen_evaluation(self):
        receipt=dict(schema='threshold-joint-evaluation-preparation/1',status='cpu_preparation_passed',
            errors=0,failures=0,both_implementations_import_verified=True,real_inference_performed=False,
            implementations=dict(joint={'a':'a'},frozen={'b':'b'}))
        validate_evaluation_preparation(receipt)
        for field,value in [('errors',1),('failures',1),('status','pending'),
                            ('both_implementations_import_verified',False),('implementations',{'joint':{}})]:
            with self.assertRaises(ValueError):validate_evaluation_preparation(dict(receipt,**{field:value}))


if __name__=='__main__':unittest.main()

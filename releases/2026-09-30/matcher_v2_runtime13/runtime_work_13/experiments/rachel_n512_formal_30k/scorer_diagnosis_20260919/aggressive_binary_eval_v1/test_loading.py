"""Actual state loading with synthetic files, not actual training checkpoints."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from consensus_binary_eval_common import frozen as common
from consensus_binary_eval_adapter.test_loading import FrozenMatcher
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.binary_scorer_v1.test_binary import fixture as network_fixture
from . import loading
from .contracts import read,save,sha,inventory,STOPS
from .test_contracts import fixture,curve


class LoadingTests(unittest.TestCase):
    def setup_case(self,base,choice):
        root=base/'run';formal=root/'formal_scratch_aggressive';stage=formal/'scorer';stage.mkdir(parents=True)
        reference=base/'reference.pt';reference.write_bytes(b'architecture-only synthetic reference')
        roles=base/'real_plan.json';save(roles,dict(synthetic=True))
        save(root/'review_approval.json',dict(synthetic_not_user_approval=True))
        save(root/'data_contract.json',dict(synthetic_only=True))
        geometry=network_fixture('patch')[0].geometry
        save(root/'geometry_calibration_v2/geometry_calibration.json',dict(status='complete',
             schema='s7-consensus-train-geometry/2',parameters=asdict(geometry)))
        config=common.training.TrainingConfig(scorer_variant='patch')
        model=loading.runtime.BinaryConsensus(FrozenMatcher(),geometry,head=loading.runtime.fresh_patch_head(config.head_seed))
        initial=loading.runtime.state_digest(model.matcher)
        cp,s,c,t,config_record=fixture(choice);b=cp['binding'];code=Path(common.training.__file__).parent
        b.update(implementation_sha256=inventory(code),binary_scorer_sha256=inventory(code.parent/'binary_scorer_v1'),
            aggressive_implementation_sha256=inventory(code.parent/'aggressive_binary_v1'),
            data_contract_sha256=sha(root/'data_contract.json'),
            geometry_calibration_sha256=sha(root/'geometry_calibration_v2/geometry_calibration.json'),
            reference_checkpoint_sha256=sha(reference),
            admission={'synthetic_bound_admission':True,'approval':{'sha256':sha(root/'review_approval.json')}},
            matcher_initialization='random_seed_no_weight_import',real_development={'synthetic_roles':True})
        b.pop('fixed_matcher')
        matcher_dir=formal/'matcher';matcher_dir.mkdir()
        origin_binding=dict(b,stage='matcher')
        matcher_history=[dict(epoch=e,updates=e*750,exposures=e*24000,
            key=[1. if e==0 else .9 if e==4 else .8],threshold=.3,real_used=False)
            for e in range(0,18,2)]
        metrics={k:v for k,v in matcher_history[2].items() if k not in ('epoch','updates','exposures')}
        origin_state=deepcopy(model.state_dict())
        # Poison only the old head; the Scorer bridge must never use it.
        origin_state['head.cluster_mlp.4.bias']+=99
        origin=matcher_dir/'best_joint.pt'
        torch.save(dict(stage='matcher',epoch=4,binding=origin_binding,metrics=metrics,model=origin_state),origin)
        osel=dict(status='selected',binding=origin_binding,selection_on_real=False,best={'epoch':4,'key':[.9],'threshold':.3},
            best_joint_sha256=sha(origin),actual_epochs=16,updates=12000,exposures=384000,stop_reason=STOPS[0],test_used=False)
        save(matcher_dir/'selection.json',osel);save(matcher_dir/'complete.json',dict(osel,status='stage_complete'))
        save(matcher_dir/'learning_curve.json',matcher_history)
        spec=loading.runtime.selected_matcher(matcher_dir,b,config);b['fixed_matcher']=spec;t['selected_matcher']=spec
        save(formal/'matcher_initialization.json',dict(initialization='random_seed_no_weight_import',seed=config.matcher_seed,
            initial_state_sha256=initial,reference_weights_imported=False,optimizer_imported=False,old_head_imported=False))
        save(formal/'scorer_initialization.json',dict(initialization='new_aggressive_selected_matcher',
            path=str(origin),sha256=sha(origin),epoch=4,old_head_imported=False,optimizer_imported=False,matcher_frozen=True))
        history=curve()
        for row in history:save(stage/f"epoch_{row['epoch']:03d}_validation.json",row)
        cp['model']=deepcopy(model.state_dict());cp['metrics']=({k:v for k,v in history[1].items() if k not in ('epoch','updates','exposures')}
            if choice=='sim' else history[1]['real_development'])
        path=stage/('best_joint.pt' if choice=='sim' else 'best_real.pt')
        self.seal(path,cp,s,c,t,formal)
        return SimpleNamespace(root=root,formal=formal,stage=stage,reference=reference,roles=roles,choice=choice,cp=cp,s=s,c=c,t=t,
            path=path,binding=b,model=model,initial=initial,helper=SimpleNamespace(bind_plan=lambda _:b['real_development']))

    def seal(self,path,cp,s,c,t,formal):
        torch.save(cp,path);s[path.stem+'_sha256']=sha(path);c[path.stem+'_sha256']=sha(path);t['last_stage']=c
        save(formal/'scorer/selection.json',s);save(formal/'scorer/complete.json',c);save(formal/'training_complete.json',t)

    def load(self,x):
        with ExitStack() as stack:
            stack.enter_context(patch.object(loading,'PLAN_SHA',sha(x.roles)))
            stack.enter_context(patch.object(loading,'validate_data',return_value=x.binding['admission']))
            stack.enter_context(patch.object(common.S7MatcherAdapter,'from_s7_m12',return_value=SimpleNamespace(base=SimpleNamespace(config='synthetic'))))
            stack.enter_context(patch.object(loading.runtime,'make_model',return_value=(x.model,{'initial_state_sha256':x.initial})))
            return loading.load_selected(x.root,x.reference,x.choice,x.roles,x.helper)

    def test_actual_head_tensors_load_and_entire_matcher_is_frozen(self):
        for choice in ('sim','real'):
            with tempfile.TemporaryDirectory() as t:
                x=self.setup_case(Path(t),choice);expected=deepcopy(x.cp['model']);model,_,p=self.load(x)
                self.assertEqual(sum(q.numel() for q in model.head.parameters()),34529)
                self.assertTrue(all(not q.requires_grad for q in model.parameters()))
                for k,v in expected.items():self.assertTrue(torch.equal(model.state_dict()[k],v),k)
                self.assertTrue(p['matcher_origin']['matcher_trained_from_random_in_this_experiment'])
                self.assertFalse(p['matcher_updated_during_scorer_training']);self.assertFalse(p['attention_present'])
                self.assertEqual(p['matcher_origin']['selected_epoch'],4)
                self.assertEqual(p['thresholds']['sim_test_aggressive'],.3)
                self.assertEqual(p['thresholds']['turufan'],.4 if choice=='real' else .3)

    def test_changed_matcher_rejected_even_after_resealing_hash(self):
        with tempfile.TemporaryDirectory() as t:
            x=self.setup_case(Path(t),'real');x.cp['model']['matcher.running']+=1
            self.seal(x.path,x.cp,x.s,x.c,x.t,x.formal)
            with self.assertRaisesRegex(ValueError,'Matcher changed'):self.load(x)

    def test_initialization_receipts_forbid_old_weights_and_optimizer(self):
        for filename,key in [('matcher_initialization.json','reference_weights_imported'),('scorer_initialization.json','optimizer_imported'),
                             ('scorer_initialization.json','old_head_imported')]:
            with tempfile.TemporaryDirectory() as t:
                x=self.setup_case(Path(t),'sim');p=x.formal/filename;r=read(p);r[key]=True;save(p,r)
                with self.assertRaises(ValueError):self.load(x)

    def test_missing_history_failure_or_changed_source_rejected(self):
        for case in ('history','failure','source'):
            with tempfile.TemporaryDirectory() as t:
                x=self.setup_case(Path(t),'sim')
                if case=='history':(x.stage/'epoch_004_validation.json').unlink()
                elif case=='failure':save(x.formal/'failure_matcher.json',dict(status='failed'))
                else:
                    x.binding['aggressive_implementation_sha256']['runtime.py']='changed'
                    self.seal(x.path,x.cp,x.s,x.c,x.t,x.formal)
                with self.assertRaises((ValueError,FileNotFoundError)):self.load(x)


if __name__=='__main__':unittest.main()

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch

from consensus_binary_eval_common import frozen as common
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.binary_scorer_v1.test_binary import fixture as network_fixture
from . import loading
from .contracts import read,save,sha,inventory,STOPS
from .test_contracts import fixture,curve

VARIANT=os.environ.get('BINARY_VERIFY_VARIANT','patch')

class FrozenMatcher(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.25,.75]))
        self.register_buffer('running',torch.tensor([2.]))
    def set_frozen(self,value):
        self.requires_grad_(not value);self.eval() if value else self.train()

class LoadingTests(unittest.TestCase):
    def fixture(self,base,choice):
        root=base/'run';formal=root/('formal_'+VARIANT);stage=formal/'scorer';stage.mkdir(parents=True)
        reference=base/'reference.pt';reference.write_bytes(b'synthetic reference identity; no real checkpoint')
        real=base/'roles.json';save(real,{'synthetic_source_roles':True})
        contract=root/'data_contract.json';save(contract,dict(status='passed',schema='s7-consensus-data-contract/3',source_disjoint=True))
        geometry=network_fixture(VARIANT)[0].geometry
        calibration=root/'geometry_calibration_v2/geometry_calibration.json'
        save(calibration,dict(status='complete',schema='s7-consensus-train-geometry/2',contract_sha256=sha(contract),parameters=asdict(geometry)))
        cp,s,c,t,config=fixture(choice);b=cp['binding']
        code=Path(common.training.__file__).parent
        b.update(implementation_sha256=inventory(code),binary_scorer_sha256=inventory(code.parent/'binary_scorer_v1'),
            data_contract_sha256=sha(contract),geometry_calibration_sha256=sha(calibration),
            reference_checkpoint_sha256=sha(reference),real_development={'synthetic_bound_roles':True})
        model=common.training.S7Consensus(FrozenMatcher(),geometry,
            head=common.training.fresh_head(config['head_seed'],VARIANT))
        cp['model']=model.state_dict()
        history=curve();cp['metrics']=({k:v for k,v in history[1].items() if k not in ('epoch','updates','exposures')}
                                     if choice=='sim' else history[1]['real_development'])
        origin=base/'origin/formal_scratch/matcher/best_joint.pt';origin.parent.mkdir(parents=True)
        source=base/'origin/source'/Path(*common.training.__package__.split('.'));source.mkdir(parents=True)
        (source/'origin.py').write_text('ORIGIN = "synthetic"\n')
        ob={k:b[k] for k in ('data_contract_sha256','geometry_calibration_sha256','reference_checkpoint_sha256')}
        ob.update(arm='scratch',formal_training=True,preflight_steps=0,implementation_sha256=inventory(source))
        oc=dict(stage='matcher',epoch=32,binding=ob,metrics={'key':[.9,.8]},model=deepcopy(cp['model']))
        # Old Scorer parameters may differ; only Matcher is imported.
        oc['model']['head.cluster_mlp.4.bias']=oc['model']['head.cluster_mlp.4.bias']+1
        torch.save(oc,origin);e32=sha(origin)
        osel=dict(status='selected',binding=ob,selection_on_real=False,best={'epoch':32,'key':[.9,.8]},
            best_joint_sha256=e32,actual_epochs=32,updates=24000,exposures=768000,stop_reason=STOPS[0])
        save(origin.parent/'selection.json',osel);save(origin.parent/'complete.json',dict(osel,status='stage_complete'))
        b['fixed_matcher']=dict(path=str(origin),sha256=e32,head_imported=False,matcher_training=False)
        save(formal/'fixed_matcher_import.json',dict(path=str(origin),sha256=e32,epoch=32,
             old_head_imported=False,optimizer_imported=False,matcher_frozen=True))
        for row in history:save(stage/f"epoch_{row['epoch']:03d}_validation.json",row)
        path=stage/('best_joint.pt' if choice=='sim' else 'best_real.pt')
        for record in (s,c,t):record['binding']=b
        self.seal(path,cp,s,c,t,formal)
        return SimpleNamespace(root=root,formal=formal,stage=stage,reference=reference,plan=real,e32=e32,
            helper=SimpleNamespace(bind_plan=lambda _:b['real_development']),path=path,cp=cp,s=s,c=c,t=t,
            choice=choice,expected=model.state_dict(),binding=b)

    def seal(self,path,cp,s,c,t,formal):
        torch.save(cp,path);key=path.stem+'_sha256';s[key]=sha(path);c[key]=sha(path);t['last_stage']=c
        save(formal/'scorer/selection.json',s);save(formal/'scorer/complete.json',c);save(formal/'training_complete.json',t)

    def load(self,x):
        with ExitStack() as stack:
            stack.enter_context(patch.object(loading,'E32_SHA',x.e32))
            stack.enter_context(patch.object(loading,'PLAN_SHA',sha(x.plan)))
            stack.enter_context(patch.object(common.S7MatcherAdapter,'from_s7_m12',side_effect=lambda _:FrozenMatcher()))
            return loading.load_selected(x.root,VARIANT,x.reference,x.choice,x.plan,x.helper)

    def test_actual_binary_state_loads_and_all_matcher_tensors_frozen(self):
        for choice in ('sim','real'):
            with tempfile.TemporaryDirectory() as tmp:
                x=self.fixture(Path(tmp),choice);model,_,p=self.load(x)
                self.assertEqual(model.head.variant,VARIANT)
                self.assertEqual(sum(v.numel() for v in model.head.parameters()),34529 if VARIANT=='patch' else 3201)
                self.assertTrue(all(not v.requires_grad for v in model.parameters()))
                for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,x.expected[k]),k)
                self.assertEqual(p['selection_kind'],choice);self.assertEqual(p['selected_epoch'],2)
                self.assertFalse(p['matcher_updated_during_training']);self.assertFalse(p['attention_present'])
                self.assertEqual(p['thresholds']['sim_test_v14'],.3)
                self.assertEqual(p['thresholds']['turufan'],.4 if choice=='real' else .3)

    def test_changed_frozen_matcher_rejected_even_with_resealed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            x=self.fixture(Path(tmp),'real');x.cp['model']['matcher.running']+=1
            self.seal(x.path,x.cp,x.s,x.c,x.t,x.formal)
            with self.assertRaisesRegex(ValueError,'Matcher changed'):self.load(x)

    def test_import_must_exclude_old_head_and_optimizer(self):
        for field in ('old_head_imported','optimizer_imported'):
            with tempfile.TemporaryDirectory() as tmp:
                x=self.fixture(Path(tmp),'real');p=x.formal/'fixed_matcher_import.json';r=read(p);r[field]=True;save(p,r)
                with self.assertRaisesRegex(ValueError,'import receipt'):self.load(x)

    def test_wrong_variant_source_or_checkpoint_content_rejected(self):
        for case in ('variant','source','content'):
            with tempfile.TemporaryDirectory() as tmp:
                x=self.fixture(Path(tmp),'real')
                if case=='content':x.path.write_bytes(b'not the selected content')
                else:
                    if case=='variant':x.binding['config']['scorer_variant']='stats' if VARIANT=='patch' else 'patch'
                    else:x.binding['binary_scorer_sha256']['head.py']='wrong'
                    self.seal(x.path,x.cp,x.s,x.c,x.t,x.formal)
                with self.assertRaises(ValueError):self.load(x)

    def test_missing_or_newly_better_validation_is_not_accepted(self):
        for case in ('missing','better'):
            with tempfile.TemporaryDirectory() as tmp:
                x=self.fixture(Path(tmp),'real');p=x.stage/'epoch_004_validation.json'
                if case=='missing':p.unlink()
                else:
                    r=read(p);r['real_development']['key'][0]=.99;save(p,r)
                with self.assertRaises((ValueError,FileNotFoundError)):self.load(x)

    def test_failure_precedes_stale_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            x=self.fixture(Path(tmp),'sim');save(x.formal/'failure.json',{'error':'synthetic'})
            with self.assertRaisesRegex(ValueError,'failure'):self.load(x)

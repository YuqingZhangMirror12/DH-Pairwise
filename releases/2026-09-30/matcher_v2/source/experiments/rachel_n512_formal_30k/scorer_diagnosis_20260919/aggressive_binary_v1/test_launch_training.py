"""CPU-only external-controller contracts; never launches training."""
from copy import deepcopy
from pathlib import Path
import tempfile
import json
import unittest
from . import launch_training as controller


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class LauncherTests(unittest.TestCase):
    def gate(self, stage):
        return dict(status='passed', formal_training=False, updated_weights_discarded=True,
            arm='scratch_aggressive', stage=stage, updates=12, exposures=384, world_size=2,
            microbatch=8, accumulate=2, effective_batch=32, matcher_unchanged=stage == 'scorer',
            head_unchanged=stage == 'matcher', matcher_frozen_expected=stage == 'scorer',
            binding={'stage':stage}, model_state_hashes=['same','same'], resume_matches_uninterrupted=True)

    def test_both_stages_require_actual_complementary_updates(self):
        for stage in ('matcher','scorer'):
            r = self.gate(stage); controller.validate_gate(r, r['binding'], stage, True)
            for field in ('matcher_unchanged','head_unchanged','matcher_frozen_expected'):
                bad = deepcopy(r); bad[field] = not r[field]
                with self.assertRaises(ValueError): controller.validate_gate(bad,r['binding'],stage,True)

    def test_hash_replay_batch_binding_and_nonzero_update(self):
        r = self.gate('scorer')
        for k,v in [('updates',0),('model_state_hashes',['one','two']),('effective_batch',64),
                    ('resume_matches_uninterrupted',False),('binding',{}),('arm','scratch_fixed')]:
            bad = deepcopy(r); bad[k] = v
            with self.assertRaises(ValueError): controller.validate_gate(bad,r['binding'],'scorer',True)

    def test_commands_never_import_e32_or_old_head_optimizer(self):
        root=Path('/synthetic')
        for stage in ('matcher','scorer'):
            a=controller.arguments(root,stage,12); cmd=controller.command(a,root/'disposable')
            self.assertIn('--nproc_per_node=2',cmd);self.assertIn('--preflight-steps',cmd)
            self.assertNotIn('--resume',cmd);self.assertIn('--resume',controller.command(a,root/'disposable',True))
            self.assertNotIn('--frozen-matcher-state',cmd);self.assertNotIn('--optimizer',cmd)
            if stage=='matcher':self.assertNotIn('--selected-matcher-stage',cmd)
            else:self.assertEqual(cmd[cmd.index('--selected-matcher-stage')+1],'/synthetic/formal_scratch_aggressive/matcher')
            a.preflight_steps=0
            self.assertNotIn('--preflight-steps',controller.command(a,root/'formal_scratch_aggressive'))

    def test_cpu_receipt_requires_exact_source_and_real_tests(self):
        with tempfile.TemporaryDirectory() as t:
            source=Path(t);(source/'test.py').write_text('x=1\n')
            r=dict(schema='aggressive-binary-cpu-preparation/1',status='passed',tests=1,errors=0,failures=0,
                skipped=0,source_unchanged=True,head_parameters=34529,formal_training_started=False,gpu_preflight=False,
                source_sha256=controller.inventory(source))
            controller.validate_cpu(r,source)
            for k,v in [('tests',0),('skipped',1),('source_unchanged',False),('head_parameters',3201),('gpu_preflight',True)]:
                bad=deepcopy(r);bad[k]=v
                with self.assertRaises(ValueError):controller.validate_cpu(bad,source)
            (source/'test.py').write_text('x=2\n')
            with self.assertRaises(ValueError):controller.validate_cpu(r,source)

    def fixture(self,root,stage):
        d=root/stage;d.mkdir();(d/'best_joint.pt').write_bytes(b'synthetic SIM')
        (d/'best_real.pt').write_bytes(b'synthetic REAL')
        binding={'config':dict(minimum_epochs=16,maximum_epochs=48),'stage':stage}
        selected=dict(status='selected',binding=binding,actual_epochs=16,updates=12000,exposures=384000,
            stop_reason='simulation_plateau_after_lr_reductions',test_used=False,
            best_joint_sha256=controller.sha(d/'best_joint.pt'),best={'epoch':2},
            best_real={'epoch':2} if stage=='scorer' else None,selection_on_real=stage=='scorer',
            best_real_sha256=controller.sha(d/'best_real.pt'),matcher_unchanged=stage=='scorer')
        self.seal(d,selected);return d,binding,selected

    def seal(self,d,s):
        save(d/'selection.json',s);save(d/'complete.json',dict(s,status='stage_complete'))

    def test_actual_terminal_counts_hashes_and_selection_semantics(self):
        for stage in ('matcher','scorer'):
            with tempfile.TemporaryDirectory() as t:
                d,b,s=self.fixture(Path(t),stage);controller.terminal(d,b,stage)
                for field,value in [('updates',0),('test_used',True),('actual_epochs',48),('selection_on_real',stage=='matcher')]:
                    bad=deepcopy(s);bad[field]=value;self.seal(d,bad)
                    with self.assertRaises(ValueError):controller.terminal(d,b,stage)
                self.seal(d,s);(d/'best_joint.pt').write_bytes(b'mutated')
                with self.assertRaises(ValueError):controller.terminal(d,b,stage)

    def test_matcher_epoch_zero_and_scorer_changed_matcher_rejected(self):
        for stage in ('matcher','scorer'):
            with tempfile.TemporaryDirectory() as t:
                d,b,s=self.fixture(Path(t),stage)
                if stage=='matcher':s['best']['epoch']=0
                else:s['matcher_unchanged']=False
                self.seal(d,s)
                with self.assertRaises(ValueError):controller.terminal(d,b,stage)

    def test_failure_takes_precedence(self):
        for stage in ('matcher','scorer'):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);d,b,s=self.fixture(root,stage);save(root/('failure_'+stage+'.json'),{'status':'failed'})
                with self.assertRaisesRegex(ValueError,'failure'):controller.terminal(d,b,stage)


if __name__=='__main__':unittest.main()

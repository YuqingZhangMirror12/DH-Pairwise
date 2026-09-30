"""Synthetic completed CPU states plus real small-canvas model load checks."""
import copy
from dataclasses import asdict
import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ..curriculum_training_v1 import launcher, test_runtime_io as fixture
from ..curriculum_training_v1.checkpoint_io import file_sha, tree_sha, write_json
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.runtime_io import export_completed, read
from ..curriculum_training_v1.runtime_plan import experiment_binding
from .terminal import verified_export, load_model, thresholds_for_export

BASE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'


class TerminalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.spec = self.root/'spec.json'; self.out = self.root/'formal'
        write_json(self.spec, dict(synthetic_cpu_fixture=True))

    def make(self, module='scorer_patch'):
        def binding(*args):
            return dict(experiment_binding(*args), run_mode='formal', execution_manifest_sha256=file_sha(self.spec))
        with patch.object(fixture, 'experiment_binding', side_effect=binding):
            _, self.plan, self.binding, _ = fixture.train_fixture(self.out, module=module)
        complete = export_completed(self.out/'exports', self.out/'checkpoints', self.plan, self.binding)
        complete['export_root'] = str(self.out/'exports'); write_json(self.out/'training_complete.json', complete)
        write_json(self.root/'formal_launch.json', dict(phase='formal', command=['/python', '--spec', str(self.spec),
            '--order', 'curriculum', '--mode', 'formal', '--out', str(self.out)]))
        write_json(self.root/'formal_return.json', dict(phase='formal', returncode=0,
            launch_sha256=file_sha(self.root/'formal_launch.json')))
        write_json(self.root/'gpu_gate.json', dict(status='passed', synthetic_parser_fixture_only=True,
            formal_binding_sha256=digest({k:v for k,v in self.binding.items() if k != 'run_mode'})))
        result = launcher.verify_formal(self.out, self.spec, self.plan, 'curriculum')
        write_json(self.root/'export_process_return.json', result)
        write_json(self.root/'controller_complete.json', dict(result,
            successful_return_sha256=file_sha(self.root/'export_process_return.json'),
            formal_return_sha256=file_sha(self.root/'formal_return.json'), gpu_gate_sha256=file_sha(self.root/'gpu_gate.json')))

    def verify(self, choice='sim_best'):
        return verified_export(self.root, self.spec, self.plan, choice)

    def test_patch_three_actual_export_files_have_correct_update_thresholds(self):
        self.make()
        for choice, update in [('sim_best',7), ('real_best',12), ('equal_budget_endpoint',16)]:
            saved, origin = self.verify(choice)
            self.assertEqual(origin['selected_updates'], update); self.assertIsNone(origin['selected_epoch'])
            self.assertEqual(origin['thresholds']['sim_test'], .3)
            self.assertEqual(origin['thresholds']['dunhuang_cv'], .24 if choice == 'real_best' else .3)
            self.assertEqual(origin['model_state_sha256'], tree_sha(saved['model']))
            self.assertFalse(origin['matcher_updated_during_training']); self.assertFalse(origin['claimed_converged'])

    def test_stats_module_is_not_mislabelled_patch(self):
        self.make('scorer_stats'); _, origin = self.verify()
        self.assertEqual(origin['variant'], 'binary_stats')

    def test_failure_overrides_stale_success(self):
        self.make(); write_json(self.root/'controller_failure.json', {'synthetic':True})
        with self.assertRaisesRegex(ValueError, 'failure'): self.verify()

    def test_nonzero_process_return_not_called_completed(self):
        self.make(); row = read(self.root/'formal_return.json'); row['returncode'] = 9
        write_json(self.root/'formal_return.json', row, replace=True)
        with self.assertRaisesRegex(ValueError, 'return'): self.verify()

    def test_actual_weight_file_corruption_refused(self):
        self.make(); (self.out/'exports/sim_best.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'model changed'): self.verify()

    def test_selected_observation_file_change_refused(self):
        self.make(); saved, _ = self.verify()
        Path(saved['observation']['artifact']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'observation artifact'): self.verify()

    def test_controller_chain_identity_required(self):
        self.make(); row = read(self.root/'controller_complete.json'); row['formal_return_sha256'] = 'a'*64
        write_json(self.root/'controller_complete.json', row, replace=True)
        with self.assertRaisesRegex(ValueError, 'controller/export'): self.verify()

    def test_matcher_or_unknown_selection_refused(self):
        self.make()
        with self.assertRaisesRegex(ValueError, 'registered'): self.verify('latest')
        with self.assertRaisesRegex(ValueError, 'registered'):
            verified_export(self.root, self.spec, fixture.setup_plan()[1], 'sim_best')

    def test_no_epoch_zero_or_test_or_outside_threshold(self):
        self.make(); saved, _ = self.verify('real_best')
        for field, value in [('update',0), ('test_used',True), ('selection_eligible',False)]:
            changed = copy.deepcopy(saved); changed['observation'][field] = value
            with self.assertRaises(ValueError): thresholds_for_export(changed)
        changed = copy.deepcopy(saved); changed['observation']['real_development']['thresholds']['turufan'] = .1
        with self.assertRaisesRegex(ValueError, 'threshold'): thresholds_for_export(changed)


class ActualModelLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = Path(os.environ['CURRICULUM_BASELINE_SOURCE']).resolve()
        cls.arch_api = importlib.import_module('staging.pairwise_v0_2.models.rachel_n512')
        cls.scratch = importlib.import_module(BASE+'s7_consensus_v1.scratch_matcher')
        cls.geometry_api = importlib.import_module(BASE+'s7_consensus_v1.compatibility')
        cls.head_api = importlib.import_module(BASE+'binary_scorer_v1.head')
        cls.model_api = importlib.import_module(BASE+'binary_scorer_v1.model')
        cls.source_sha = digest({str(p.relative_to(cls.baseline)):file_sha(p) for p in cls.baseline.rglob('*.py')})

    def setUp(self):
        torch.set_num_threads(1); self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)

    def make(self, variant='patch'):
        architecture = self.arch_api.RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
            landmark_count=2, context_layers=2, activation_checkpointing=False)
        geometry = self.geometry_api.CompatibilityConfig(.5,.5,.5,.5,1.,15.)
        matcher = self.scratch.fresh_matcher(architecture,26092407).set_frozen(True).eval()
        model = self.model_api.BinaryConsensus(matcher,geometry,head=self.head_api.BinaryClusterHead(variant))
        common = dict(model_seed=26092407,head_seed=26092406,baseline_sources_sha256=self.source_sha,
            data_admission_sha256='a'*64,geometry_sha256='b'*64,synthetic_cpu_fixture=True)
        prior_binding = dict(module='matcher',order='curriculum',common_plan=common,
            model_spec=dict(architecture=asdict(architecture),geometry=asdict(geometry)))
        prior = dict(schema='curriculum-model-export/1',module='matcher',order='curriculum',selection_kind='sim_best',
            updates=12,binding=prior_binding,model={'matcher.'+k:v for k,v in matcher.state_dict().items()})
        path = Path(self.tmp.name)/'matcher.pt'; torch.save(prior,path)
        selected = dict(path=str(path),sha256=file_sha(path),source_binding=prior_binding,updates=12,
            old_head_imported=False,optimizer_imported=False)
        spec = dict(architecture=asdict(architecture),geometry=asdict(geometry),scorer_variant=variant,
            proposal_revision='native-hypothesis-complete-link-union/1-diameter16',
            initialization='selected_curriculum_matcher_new_head',selected_matcher=selected,
            initial_matcher_state_sha256=tree_sha(matcher.state_dict()),old_head_imported=False,optimizer_imported=False)
        saved = dict(model=model.state_dict(),binding=dict(module='scorer_'+variant,order='curriculum',
            common_plan=common,model_spec=spec))
        origin = dict(schema='curriculum-scorer-evaluation-origin/1',module='scorer_'+variant,
            common_plan_sha256=digest(common),model_state_sha256=tree_sha(model.state_dict()),
            matcher_state_sha256=tree_sha(matcher.state_dict()))
        return saved, origin

    def test_both_actual_light_heads_load_exactly_and_remain_frozen(self):
        for variant, count in [('patch',34529),('stats',3201)]:
            saved, origin = self.make(variant); rng = torch.get_rng_state().clone()
            model, meta = load_model(saved,origin,self.baseline)
            self.assertEqual(tree_sha(model.state_dict()),origin['model_state_sha256'])
            self.assertTrue(torch.equal(rng,torch.get_rng_state())); self.assertFalse(model.training)
            self.assertTrue(model.matcher.frozen); self.assertFalse(any(p.requires_grad for p in model.parameters()))
            self.assertEqual(sum(p.numel() for p in model.head.parameters()),count)
            self.assertFalse(meta['old_head_imported'])

    def test_changed_export_or_origin_or_baseline_rejected(self):
        saved, origin = self.make()
        with self.assertRaisesRegex(ValueError,'identity'): load_model(saved,dict(origin,model_state_sha256='c'*64),self.baseline)
        with self.assertRaisesRegex(ValueError,'baseline'): load_model(saved,origin,Path(self.tmp.name))
        Path(saved['binding']['model_spec']['selected_matcher']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Matcher file'): load_model(saved,origin,self.baseline)

    def test_matcher_tensor_drift_detected_even_if_export_hash_redeclared(self):
        saved, origin = self.make(); key = next(k for k,v in saved['model'].items()
            if k.startswith('matcher.') and v.is_floating_point())
        saved['model'][key] = saved['model'][key]+1
        origin['model_state_sha256'] = tree_sha(saved['model'])
        with self.assertRaisesRegex(ValueError,'frozen Matcher'): load_model(saved,origin,self.baseline)

    def test_other_head_variant_and_nonfinite_state_refused(self):
        saved, origin = self.make(); bad = copy.deepcopy(saved); bad['binding']['model_spec']['scorer_variant']='stats'
        with self.assertRaisesRegex(ValueError,'registered'): load_model(bad,origin,self.baseline)
        key = next(k for k in saved['model'] if k.startswith('head.'))
        saved['model'][key] = torch.full_like(saved['model'][key],float('nan'))
        origin['model_state_sha256'] = tree_sha(saved['model'])
        with self.assertRaisesRegex(ValueError,'nonfinite'): load_model(saved,origin,self.baseline)


if __name__ == '__main__': unittest.main()

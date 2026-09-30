"""Actual small-shape CPU loss/AdamW/checkpoint/export path; not a CUDA gate."""
import copy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from ..curriculum_training_v1.checkpoint_io import DistributedCheckpoint, file_sha, tree_sha, write_json
from ..curriculum_training_v1.runtime_io import ObservationWriter
from ..curriculum_training_v1.training_core import Topology, run_updates
from ..s7_consensus_v1.compatibility import CompatibilityConfig
from ..s7_consensus_v1.test_matcher import inputs
from ..s7_consensus_v1.metrics import summarize
from .model_runtime import make_components, load_export_matcher, load_export_scorer
from .runtime_io import export_completed, selected_matcher
from .test_model_runtime import plan_fixture
from .test_validation import rows_fixture
from .validation import ValidationAdapter, select_dunhuang
from .terminal import load_model


def run_fixture(directory, module='matcher', selected=None, resume=None, stop=None):
    torch.set_num_threads(1); torch.manual_seed(901); random.seed(903); np.random.seed(904)
    source = Path(__file__).resolve().parents[4]
    ledger, plan, schedule = plan_fixture('B3', module)
    architecture = RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
        landmark_count=2, context_layers=2, activation_checkpointing=False)
    geometry = CompatibilityConfig(.5, .5, .5, .5, 1., 15.)
    topology = Topology(0, 1, 2, 2)
    parts = make_components(plan, schedule, topology, source, architecture, geometry, selected)
    binding = dict(parts['binding'], explicit_small_CPU_integration_fixture=True)
    raw = parts['module']; initial = tree_sha(raw.model.matcher.state_dict())
    initial_head = tree_sha(raw.model.head.state_dict())
    data = []
    keys = ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b')
    for i, ref in enumerate(ledger.catalog):
        row = {key: value[0].detach().clone() for key, value in zip(keys, inputs())}
        target = torch.tensor([0, 1, 2, 3, -2, -2] if ref.label else [-1, -1, -1, -1, -2, -2])
        row.update(labels=torch.tensor(float(ref.label)), target_a=target, target_b=target.clone(),
            translation_a_to_b_rc=torch.tensor([3., 14.]) if ref.label else torch.zeros(2),
            translation_valid=torch.tensor(ref.label), pose_enabled=torch.tensor(False),
            precise_recipe=torch.tensor(False), recipes='cpu_damaged_fixture',
            pair_ids=f'cpu-fixture-{i}')
        data.append(row)
    sim = dict(stage='matcher' if module == 'matcher' else 'scorer', real_used=False,
               key=[.9, .8, .7], selection_value=.9, threshold=.5)
    development = None if module == 'matcher' else lambda: (
        select_dunhuang(rows_fixture(), SimpleNamespace(threshold_tie_preference=.3), summarize), rows_fixture())
    writer = ObservationWriter(directory/'validation', plan, binding)
    evaluator = ValidationAdapter(plan, topology, lambda: (sim, []), development, writer)
    optimizer = torch.optim.AdamW([p for p in raw.parameters() if p.requires_grad], lr=plan.learning_rate_knots[0][1])
    gradient_steps = []; seen = {}
    hooks = [p.register_hook(lambda value, name=name: seen.update({name: value.detach().abs().sum().item()}))
             for name, p in raw.named_parameters() if p.requires_grad]
    def step(record):
        gradient_steps.append(dict(update=record['completed_updates'], gradients=dict(seen))); seen.clear()
    state = run_updates(raw, optimizer, data, ledger, 'curriculum', topology, 'cpu',
        plan.learning_rate_knots, plan.validation_updates, evaluate=evaluator, resume=resume,
        binding=binding, checkpoint_every=4, stop_after=stop, on_update=step,
        on_checkpoint=DistributedCheckpoint(directory/'checkpoints', 0, 1))
    for hook in hooks:hook.remove()
    return dict(state=copy.deepcopy(state), parts=parts, plan=plan, binding=binding, ledger=ledger,
                gradients=gradient_steps, initial_matcher=initial, initial_head=initial_head)


class RuntimePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_actual_v2_loss_updates_all_new_branches_and_resumes_update1(self):
        full = run_fixture(self.root/'full')
        partial = run_fixture(self.root/'resume', stop=1)
        # Receipt paths are run-specific, so compare model, optimizer and rank
        # RNG independently; schedule and observation metrics must also agree.
        resumed = run_fixture(self.root/'resume', resume=partial['state'])
        for field in ('model', 'optimizer', 'rng', 'sampling'):
            self.assertEqual(tree_sha(full['state'][field]), tree_sha(resumed['state'][field]), field)
        self.assertNotEqual(full['initial_matcher'], tree_sha(full['parts']['model'].matcher.state_dict()))
        self.assertEqual(full['initial_head'], tree_sha(full['parts']['model'].head.state_dict()))
        for step in full['gradients']:
            self.assertTrue(step['gradients'])
            self.assertTrue(all(np.isfinite(v) for v in step['gradients'].values()))
        required = {n for n, p in full['parts']['module'].named_parameters() if p.requires_grad}
        self.assertTrue(all(set(step['gradients']) == required for step in full['gradients']))
        for name in ('model.matcher.upgrades.self_blocks.0.qkv.weight',
                     'model.matcher.upgrades.cross_blocks.0.q.weight',
                     'model.matcher.upgrades.scale_primal.0.weight'):
            self.assertEqual(full['gradients'][0]['gradients'][name], 0)
            self.assertGreater(full['gradients'][1]['gradients'][name], 0)

    def test_completed_matcher_export_new_heads_and_native_reload(self):
        trained = run_fixture(self.root/'matcher')
        complete = export_completed(self.root/'export', self.root/'matcher/checkpoints', trained['plan'], trained['binding'])
        returned = self.root/'return.json'
        write_json(returned, dict(returncode=0, training_complete_sha256=file_sha(self.root/'export/training_complete.json')))
        selected = selected_matcher(self.root/'export', returned, trained['plan'].sha256, 'B3')
        self.assertGreater(selected['updates'], 0)
        loaded = load_export_matcher(torch.load(selected['path'], weights_only=False), Path(__file__).resolve().parents[4])
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
        for module in ('scorer_patch', 'scorer_stats'):
            out = self.root/module; head = run_fixture(out, module, selected=selected)
            self.assertEqual(head['initial_matcher'], tree_sha(head['parts']['model'].matcher.state_dict()))
            self.assertNotEqual(head['initial_head'], tree_sha(head['parts']['model'].head.state_dict()))
            exported = export_completed(out/'export', out/'checkpoints', head['plan'], head['binding'])
            saved = torch.load(exported['exports']['real_best']['path'], weights_only=False)
            self.assertFalse(saved['observation']['real_development']['turufan_used'])
            restored = load_export_scorer(saved, Path(__file__).resolve().parents[4])
            self.assertEqual(tree_sha(restored.state_dict()), tree_sha(saved['model']))
            origin = dict(schema='matcher-v2-terminal-origin/1', module=module, arm='B3',
                common_plan_sha256=head['plan'].sha256, model_state_sha256=tree_sha(saved['model']),
                matcher_state_sha256=head['initial_matcher'])
            terminal, geometry, info = load_model(saved, origin, Path(__file__).resolve().parents[4])
            self.assertEqual(tree_sha(terminal.state_dict()), tree_sha(saved['model']))
            self.assertFalse(info['optimizer_imported']); self.assertFalse(info['old_head_imported'])
        self.assertEqual(complete['completed_updates'], trained['ledger'].total_updates)


if __name__ == '__main__':unittest.main()

"""Independent CPU parity check against the user-supplied in-tree reference.

Reference files are loaded under separate module names, never installed over
the immutable baseline. Real examples are forward-only diagnostics; backward
uses only the four supplied synthetic examples. No pilot accuracy is claimed.
"""
import argparse
import copy
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, RachelN512Pairwise
from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.model_adapter import require
from ..s7_consensus_v1.matcher import S7MatcherAdapter, INPUTS
from ..s7_consensus_v1.scratch_matcher import matching_loss
from .adapter import MatcherV2Adapter
from .network import MatcherV2Config


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def reference_name(name):
    parts = name.split('.')
    if parts[0] == 'self_blocks':
        parts[2] = dict(norm1='norm_attention', norm2='norm_feedforward', out='output',
                        ff='feedforward').get(parts[2], parts[2])
        return 'token_context.'+'.'.join(parts)
    if parts[0] == 'cross_blocks':
        parts[2] = dict(norm_q='norm_query', norm_kv='norm_source', norm2='norm_feedforward',
                        q='query', kv='key_value', out='output', ff='feedforward').get(parts[2], parts[2])
        return 'token_context.'+'.'.join(parts)
    parts[0] = dict(concat='scale_concat', scale_weight='scale_logit_weight',
                    log_sharpness='log_inverse_temperature_gain').get(parts[0], parts[0])
    return '.'.join(parts)


def compare(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only comparison required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(26093051)
    root = args.reference.resolve(); out = args.output_new.resolve()
    require(not out.exists(), 'new comparison receipt required')
    files = [root/'rachel_n512_reference.py', root/'matcher_reference.py',
             root/'results/reference.json', root/'results/reference.npz', root/'results/reference_e32_base_state.pt']
    before = {str(p):file_sha(p) for p in files}
    net = load_file('staging.pairwise_v0_2.models._external_intree_v2', files[0])
    adapter_module = load_file('experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1._external_v2_adapter', files[1])
    meta = json.loads(files[2].read_text()); arrays = np.load(files[3])
    config = dict(meta['config']); config['window_sizes_px'] = tuple(config['window_sizes_px'])
    arch = RachelN512Config(**config)
    base = RachelN512Pairwise(arch)
    base.load_state_dict(torch.load(files[4], map_location='cpu', weights_only=False), strict=True)
    old = S7MatcherAdapter(copy.deepcopy(base), frozen=True).eval()
    own = MatcherV2Adapter(copy.deepcopy(base), config=MatcherV2Config(enabled=True), frozen=False).eval()
    ref = adapter_module.S7MatcherAdapter(net.RachelN512Pairwise(net.RachelN512V2Config(**asdict(arch))), frozen=False).eval()
    state = ref.base.state_dict()
    for name, value in base.state_dict().items():state[name] = value.clone()
    for name, value in own.upgrades.named_parameters():state[reference_name(name)] = value.detach().clone()
    ref.base.load_state_dict(state, strict=True)
    require(sum(p.numel() for p in own.upgrades.parameters()) == 410213, 'reference architecture count differs')
    report = dict(status='running', source_sha256=before, n_pairs=10,
        real_examples_forward_only=6, synthetic_examples=4, new_parameters=410213,
        mapped_parameter_tensors=len(list(own.upgrades.parameters())), cuda_initialized=False)
    fields = ('local_a', 'local_b', 'context_a', 'context_b', 'affinity', 'assignment')
    def inputs(indices):return [torch.from_numpy(arrays['in_'+key][indices]) for key in INPUTS]
    def difference(a, b):return {k:float((getattr(a,k)-getattr(b,k)).abs().max()) for k in fields}
    began = time.monotonic()
    with torch.no_grad():
        neutral = []
        for i in range(10):
            values = inputs([i]); a, b, c = old(*values), own(*values), ref(*values)
            own_diff, ref_diff = difference(a,b), difference(a,c)
            require(max(own_diff.values()) == 0 and max(ref_diff.values()) == 0, 'zero-init legacy parity failed')
            neutral.append(dict(index=i, own=own_diff, reference=ref_diff))
        report['zero_init'] = neutral
        generator = torch.Generator().manual_seed(26093052)
        for name, value in own.upgrades.named_parameters():
            value.add_(.025*torch.randn(value.shape, generator=generator))
        state = ref.base.state_dict()
        for name, value in own.upgrades.named_parameters():state[reference_name(name)] = value.detach().clone()
        ref.base.load_state_dict(state, strict=True)
        nonzero = []
        for i in range(10):
            values = inputs([i]); a, b = own(*values), ref(*values)
            for field in fields:torch.testing.assert_close(getattr(a,field), getattr(b,field), atol=3e-5, rtol=3e-5)
            nonzero.append(dict(index=i, maximum_absolute=difference(a,b)))
        report['nonzero_mapped_weights'] = nonzero
    # Gradient comparison: synthetic examples ONLY, existing matching loss.
    indices = [6, 7, 8, 9]; batch = dict(zip(INPUTS, inputs(indices)))
    batch.update(labels=torch.from_numpy(arrays['in_label'][indices]).float(),
        target_a=torch.from_numpy(arrays['in_target_a'][indices]), target_b=torch.from_numpy(arrays['in_target_b'][indices]),
        translation_a_to_b_rc=torch.from_numpy(arrays['in_translation'][indices]).float())
    batch['translation_valid'] = batch['labels'] == 1
    batch['pose_enabled'] = batch['translation_valid'] & ~torch.from_numpy(arrays['in_changed'][indices])
    gradients = []; losses = []
    for model in (own, ref):
        model.train(); model.zero_grad(set_to_none=True)
        output = model(*(batch[k] for k in INPUTS)); loss, _, _ = matching_loss(model, output, batch)
        loss.backward(); losses.append(float(loss.detach()))
        params = dict(model.upgrades.named_parameters()) if model is own else dict(model.base.named_parameters())
        gradients.append({name:params[name if model is own else reference_name(name)].grad.detach().clone()
                          for name, _ in own.upgrades.named_parameters()})
    errors = {}
    for name in gradients[0]:
        a, b = gradients[0][name], gradients[1][name]
        require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), 'nonfinite reference gradient')
        torch.testing.assert_close(a, b, atol=3e-5, rtol=5e-4, msg=name)
        errors[name] = float((a-b).abs().max())
    require(before == {str(p):file_sha(p) for p in files}, 'reference changed during comparison')
    require(not torch.cuda.is_initialized(), 'CPU comparison touched CUDA')
    report.update(status='passed', seconds=time.monotonic()-began, synthetic_training_losses=losses,
        gradient_maximum_absolute=errors, matching_loss_unchanged=True, slide_loss_used=False,
        real_examples_backpropagated=0, formal_training=False, accuracy_claim=False)
    write_json(out, report)
    print(json.dumps(dict(status='passed', pairs=10, new_parameters=410213,
        nonzero_output_max=max(max(row['maximum_absolute'].values()) for row in nonzero),
        gradient_max=max(errors.values()), real_backprop=0, sha256=file_sha(out))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True); p.add_argument('--output-new', type=Path, required=True)
    compare(p.parse_args())


if __name__ == '__main__':main()

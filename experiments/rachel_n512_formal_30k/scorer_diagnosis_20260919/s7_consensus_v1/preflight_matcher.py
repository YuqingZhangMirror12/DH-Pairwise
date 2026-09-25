"""Read-only checkpoint/input parity check; never runs an optimizer."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs
from .matcher import S7MatcherAdapter, INPUTS, SOURCE_SHA


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, x in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(checkpoint, manifest, output, device):
    output = Path(output)
    if output.exists():
        raise ValueError('preserve previous preflight; choose a new output path')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    adapter = S7MatcherAdapter.from_s7_m12(checkpoint).to(device).eval()
    before = state_digest(adapter)
    record = json.loads(Path(manifest).read_text())
    # Balanced recipe sample, selected without reading any predicted score.
    seen, selected = set(), []
    for entry in record['entries']:
        key = (entry.get('recipe', entry.get('s7_recipe')), int(entry['label']))
        if key not in seen:
            seen.add(key)
            selected.append(entry)
    if not selected:
        raise ValueError('empty manifest')
    receipts = []
    start = time.time()
    for entry in selected:
        path = entry.get('sample_path') or str(Path(record['artifact_root']) / entry['artifact_path'])
        sample, _ = load_sample(path)
        if sample.pair_id != entry['pair_id']:
            raise ValueError('sample identity differs')
        values = collate_rachel_pairs([sample]).as_dict()
        args = [torch.from_numpy(np.array(values[k], copy=True)).to(device) for k in INPUTS]
        with torch.inference_mode():
            original = adapter.base(*args)
            evidence = adapter(*args)
        errors = {}
        for name, old, new in (
            ('S', original.affinity, evidence.affinity),
            ('Q', original.assignment, evidence.assignment),
            ('H_a', original.token_features_a, evidence.context_a),
            ('H_b', original.token_features_b, evidence.context_b),
            ('unmatched_a', original.unmatched_a, evidence.unmatched_a),
            ('unmatched_b', original.unmatched_b, evidence.unmatched_b),
        ):
            torch.testing.assert_close(old, new, rtol=1e-6, atol=1e-7, msg=name)
            errors[name] = float((old - new).abs().max())
        receipts.append(dict(pair_id=sample.pair_id, sample_sha256=digest(path),
            recipe=entry.get('recipe', entry.get('s7_recipe')), label=int(entry['label']),
            max_abs_errors=errors, numeric_valid=bool(evidence.numeric_valid[0]),
            transport_converged=bool(evidence.transport.diagnostics.converged[0]),
            row_mass_residual=float(evidence.transport.diagnostics.row_residual_max[0]),
            col_mass_residual=float(evidence.transport.diagnostics.col_residual_max[0])))
    after = state_digest(adapter)
    if after != before:
        raise AssertionError('read-only preflight changed parameters/buffers')
    result = dict(status='passed', scope='historical S7 M12 adapter parity; not full consensus preflight',
        checkpoint=str(checkpoint), checkpoint_sha256=SOURCE_SHA, manifest=str(manifest),
        manifest_sha256=digest(manifest), device=str(device), config=asdict(adapter.base.config),
        elapsed_seconds=time.time()-start, optimizer_updates=0, state_unchanged=True,
        original_state_sha256=before, samples=receipts,
        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        padding_limit='Legacy Context storage-width/cyclic-origin sensitivity retained for parity; new modules tested separately.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','scope','elapsed_seconds','optimizer_updates','state_unchanged')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'manifest', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    run(args.checkpoint, args.manifest, args.output, args.device)

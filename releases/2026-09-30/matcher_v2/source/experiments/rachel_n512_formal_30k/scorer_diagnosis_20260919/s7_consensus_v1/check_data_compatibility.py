"""CPU-only archive -> training batch -> supervision checks; never train.

This check may inspect a human-review pilot. A passing receipt is NOT approval
for full generation, a formal data contract, geometry calibration, inference,
or a replacement for the separately required training preflights.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .data import Dataset, collate, pair_labels, recipe_for
from .matcher import INPUTS
from .targets import candidate_quality, local_targets


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, name):
    expected = torch.from_numpy(np.array(expected, copy=True))
    require(actual.dtype == expected.dtype and torch.equal(actual, expected),
            'batch changed archived ' + name)


def check_batch(items, batch):
    """Check the real collator/label adapters, without a neural forward pass."""
    labels = pair_labels(batch)
    rows = []
    for i, ((sample, report, entry), label) in enumerate(zip(items, labels)):
        require(batch['pair_ids'][i] == sample.pair_id, 'batch Pair identity changed')
        recipe = recipe_for(entry, report)
        declared = report.get('compound', {}).get('recipe')
        if declared is not None:
            require(recipe == declared, 'batch recipe differs from materialized damage')
        require(batch['recipes'][i] == recipe, 'batch recipe changed')
        require(label.label == bool(sample.label), 'Pair label changed')
        require(label.translation_known == bool(sample.translation_valid), 'GT validity changed')
        same(label.translation_a_to_b_rc, sample.translation_a_to_b_rc, 'GT displacement')
        same(batch['translation_a_to_b_xy_cartesian'][i],
             sample.translation_a_to_b_xy_cartesian, 'Cartesian GT displacement')
        require(bool(batch['precise_recipe'][i]) == (recipe == 'clean'),
                'damaged recipe incorrectly treated as clean')
        require(bool(batch['pose_enabled'][i]) == (recipe == 'clean' and label.translation_known),
                'legacy exact-pose supervision differs from clean-only rule')
        for name in ('mask_a', 'mask_b', 'coarse_mask_a', 'coarse_mask_b'):
            same(batch[name][i], getattr(sample, name), name)
        compact_ids = {}
        inherited = {}
        unknown = {}
        for side in ('a', 'b'):
            n = len(getattr(sample, 'target_' + side))
            for stem in ('points_rc_', 'contour_valid_', 'target_'):
                same(batch[stem + side][i, :n], getattr(sample, stem + side), stem + side)
            valid = batch['contour_valid_' + side][i]
            target = getattr(label, 'target_' + side)
            require(not valid[n:].any() and (target[n:] == -2).all(),
                    'padding became valid or known supervision')
            precise = getattr(label, 'precise_anchor_' + side)
            require(torch.equal(precise, (target >= 0) & (recipe == 'clean')),
                    'damage residual was used to fabricate a precise anchor')
            compact_ids[side] = valid.nonzero(as_tuple=False).flatten()
            inherited[side] = int((target >= 0).sum())
            unknown[side] = int(((target == -2) & valid).sum())
        require(inherited['a'] == inherited['b'], 'reciprocal target count differs')
        ia, ib = compact_ids['a'], compact_ids['b']
        # local_targets needs shape only. This synthetic zero tensor is not a
        # Matcher output and is never interpreted as a prediction or confidence.
        shape_only = SimpleNamespace(original_a=ia, original_b=ib,
                                     q=torch.zeros(len(ia), len(ib)))
        local = local_targets(shape_only, label)
        require(int(local.source_support_known.sum()) == inherited['a'],
                'local supervision lost or created inherited correspondences')
        expected_precise = inherited['a'] if recipe == 'clean' else 0
        require(int(local.precise_anchor_known.sum()) == expected_precise,
                'incorrect precise-localization target count')
        if label.label:
            require(label.translation_known, 'damaged positive lost global GT')
            require(not local.edge_target_known[label.target_a[ia] == -2].any(),
                    'unknown/artificial-cut A points became known targets')
            require(not local.edge_target_known[:, label.target_b[ib] == -2].any(),
                    'unknown/artificial-cut B points became known targets')
        else:
            require(inherited['a'] == 0 and local.edge_target_known.all(),
                    'negative supervision contains positive correspondences')
        # GT is used solely to test supervision; no prediction is generated.
        poses = [label.translation_a_to_b_rc,
                 label.translation_a_to_b_rc + torch.tensor([21., 0.])]
        quality, known = candidate_quality(poses, label)
        expected = torch.tensor([1., 0.] if label.label else [0., 0.])
        require(known.all() and torch.equal(quality, expected),
                'wrong-pose candidate inherited the positive Pair label')
        for key in INPUTS:
            require(isinstance(batch[key], torch.Tensor), 'non-tensor Matcher input')
            require(torch.isfinite(batch[key]).all(), 'nonfinite Matcher input')
        partial = report.get('compound', {}).get('partial') or {}
        rows.append(dict(pair_id=sample.pair_id, label=int(label.label), recipe=recipe,
                         partial_mode=partial.get('mode'), inherited_pairs=inherited['a'],
                         unknown_valid_points=unknown,
                         precise_pairs=expected_precise,
                         global_gt_retained=label.translation_known))
    require(len(rows) == len(items), 'batch label count changed')
    return rows


def run(manifest, output, *, expected_count, batch_size=2):
    output = Path(output)
    require(not output.exists(), 'preserve previous compatibility receipt')
    require(expected_count > 0 and batch_size > 0, 'invalid count or batch size')
    torch.set_num_threads(2)
    data = Dataset(manifest)
    require(len(data) == expected_count, 'unexpected pilot/archive population')
    start = time.monotonic()
    rows = []
    with torch.no_grad():
        for offset in range(0, len(data), batch_size):
            items = [data[j] for j in range(offset, min(offset + batch_size, len(data)))]
            rows.extend(check_batch(items, collate(items)))
    require(hashlib.sha256(Path(manifest).read_bytes()).hexdigest() == data.sha256,
            'manifest changed while checking')
    positive = [r for r in rows if r['label']]
    require(all(r['inherited_pairs'] >= 4 for r in positive),
            'positive archive has fewer than four inherited correspondences')
    counts = Counter((r['recipe'], r['label'], r['partial_mode']) for r in rows)
    result = dict(status='passed', scope='CPU data/target compatibility, not training or inference',
        schema='s7-consensus-pilot-compatibility/1',
        manifest=str(Path(manifest).resolve()), manifest_sha256=data.sha256,
        checked_pairs=len(rows), positive_pairs=len(positive), negative_pairs=len(rows)-len(positive),
        batch_size=batch_size, optimizer_updates=0, model_forwards=0,
        gpu_used=False, formal_training_approval=False, full_data_generation_approval=False,
        online_augmentation_applied=False, source_archives_modified=False,
        matcher_inputs=list(INPUTS),
        positive_inherited_pair_count=dict(min=min(r['inherited_pairs'] for r in positive),
            max=max(r['inherited_pairs'] for r in positive)),
        strata=[dict(recipe=recipe,label=label,partial_mode=mode,pairs=count)
                for (recipe,label,mode),count in sorted(counts.items(), key=lambda x:str(x[0]))],
        checks=['archive arrays unchanged through collate', 'damaged-positive global GT retained',
                'reciprocal local support unchanged', 'unknown/artificial-cut targets remain masked',
                'only clean recipes supply precise anchors', 'wrong-pose candidates remain negative',
                'negative pairs contain no positive correspondence targets'],
        elapsed_seconds=time.monotonic()-start, pairs=rows,
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).with_name('data.py'),
            Path(__file__).with_name('targets.py'), Path(__file__).with_name('matcher.py'))})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','checked_pairs','positive_pairs',
        'negative_pairs','optimizer_updates','model_forwards','elapsed_seconds')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--expected-count', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    run(args.manifest, args.output, expected_count=args.expected_count, batch_size=args.batch_size)

"""Fixed S6-D2 weights and physical masks; fresh contour-count intervention.

CPU-only diagnostic, not S5/S8 training or a selected operating configuration.
Re-extraction control separates original prepared points from count changes.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import torch
from scipy.spatial import cKDTree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def anchor_indices(reference, current):
    distance, ids = cKDTree(current).query(reference)
    return ids, dict(reference_count=len(reference), current_count=len(current),
        unique_current_anchors=len(np.unique(ids)), maximum_distance_px=float(distance.max()),
        mean_distance_px=float(distance.mean()), exact_fraction=float(np.mean(distance < 1e-5)))


@torch.inference_mode()
def forward(model, masks, points, config, decoder):
    arrays = [masks[s][None, None].astype(np.float32) for s in 'ab']
    arrays += [points[s][None].astype(np.float32) for s in 'ab']
    arrays += [np.ones((1, len(points[s])), bool) for s in 'ab']
    tensors = [torch.from_numpy(x.copy()) for x in arrays]
    base = model.base_model(*tensors)
    z = model.score_head(base.token_features_a, base.token_features_b, *tensors[-2:])[0]
    if not bool(base.training_valid[0]) or not torch.isfinite(z):
        raise ValueError('invalid model output; do not silently use fallback')
    q = base.assignment[0].numpy()
    pose = decoder.estimate_translation_layout(points['a'], points['b'], q,
        np.ones(len(points['a']), bool), np.ones(len(points['b']), bool), config=config)
    features = {s:getattr(base, 'token_features_' + s).clone() for s in 'ab'}
    row = dict(logit=float(z), probability=float(z.sigmoid()),
        count_a=len(points['a']), count_b=len(points['b']),
        q_mass=float(q.astype(np.float64).sum()),
        q_mass_per_mean_token=float(q.astype(np.float64).sum() / ((len(points['a']) + len(points['b'])) / 2.)),
        sinkhorn_converged=bool(base.transport.diagnostics.converged[0]),
        layout=dict(valid=bool(pose.valid), candidate_count=int(pose.candidate_count),
            inlier_count=int(pose.inlier_count), residual_px=pose.residual_px,
            t_a_to_b_rc=pose.t_a_to_b_rc.tolist() if pose.valid else None))
    return row, features


@torch.inference_mode()
def anchor_head(head, reference, points, ref_features, features):
    selected, sides = {}, {}
    for s in 'ab':
        ids, detail = anchor_indices(reference[s], points[s])
        selected[s] = features[s][:, ids, :]
        cosine = torch.nn.functional.cosine_similarity(ref_features[s], selected[s], dim=-1)
        detail['context_mean_cosine_to_resampled512'] = float(cosine.mean())
        sides[s] = detail
    if any(v['unique_current_anchors'] != v['reference_count'] for v in sides.values()):
        return dict(available=False, reason='nearest anchors are not unique', sides=sides)
    z = head(selected['a'], selected['b'],
        *[torch.ones(1, len(reference[s]), dtype=torch.bool) for s in 'ab'])[0]
    return dict(available=True, logit=float(z), probability=float(z.sigmoid()), sides=sides,
        interpretation='Scorer rerun on nearest resampled512 anchors from this denser Matcher context; not fresh512 Matcher')


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, args.source_root)
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluation
    from staging.pairwise_v0_2.models import translation_layout as decoder
    from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour
    from staging.pairwise_v0_2.pairwise_data.rachel_step_density import sample_pair_step_contours
    root, out = Path(args.probe_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source = json.loads((root / 'protocol.json').read_text())
    cases = {r['pair_id']:r for r in json.loads((root / 'cases.json').read_text())}
    previous = json.loads(Path(args.prior_results).read_text())
    selected = previous['rows']
    if len(selected) != 8:
        raise ValueError('same eight token-multiplicity examples required')
    model, identity = evaluation.load_frozen_model(source['model']['training_run'], source['model']['selection'])
    if identity['checkpoint_sha256'] != previous['source_checkpoint_sha256']:
        raise ValueError('different source checkpoint')
    model.cpu().eval().requires_grad_(False)
    original_config = asdict(model.base_model.config)
    # Shape allowance only, with no parameter changes and no source-file edits.
    model.config = model.base_model.config = replace(model.base_model.config, contour_cap=2048)
    cfg = evaluation.core.fixed.TOP2_CONFIG
    receipt = dict(status='running', schema_version='same-weights-mask-contour-count/1',
        source_checkpoint_sha256=identity['checkpoint_sha256'], original_config=original_config,
        runtime_cap_allowance=2048, prior_selection_sha256=sha(args.prior_results),
        script_sha256=sha(__file__), decoder_config=asdict(cfg), cpu_threads=1,
        trained_or_threshold_fitted=False, GPU_used=False, physical_masks_resized=False,
        GT_used_for_sampling_or_decoding=False, fresh_matcher_and_sinkhorn=True,
        selection='same eight previously selected token-multiplicity examples, not a representative population',
        caveats=['Fixed512-trained checkpoint; not density-retrained S5/S8.',
            'Original extractor retains dense contour if shorter than cap; actual valid counts reported.',
            'Anchor intervention changes Scorer input only after the denser Matcher forward.',
            'Step3 is a different sampling policy; cap1024/2048 are not exact S5 inputs.',
            'GT offset error is attached after inference; OOD has no layout GT.'])
    save(out / 'protocol.json', receipt)
    rows, started = [], time.monotonic()
    try:
        for chosen in selected:
            case = cases[chosen['pair_id']]
            path = root / case['arrays_path']
            if sha(path) != case['arrays_sha256']:
                raise ValueError('saved tensors changed')
            with np.load(path, allow_pickle=False) as z:
                masks = {s:z['mask_' + s].copy() for s in 'ab'}
                original = {s:z['points_rc_' + s][z['valid_' + s]].copy() for s in 'ab'}
            if any(not np.isin(masks[s], (0, 1)).all() for s in 'ab'):
                raise ValueError('nonbinary physical mask')
            reference = {s:extract_ordered_outer_contour(masks[s].astype(bool), cap=512, smoothing_sigma=3.)[0] for s in 'ab'}
            row = dict(pair_id=case['pair_id'], dataset=case['dataset'], name=chosen['name'],
                stratum=chosen['stratum'], label=case['label'], threshold=chosen['threshold'],
                saved_arrays_sha256=case['arrays_sha256'], variants=[])
            ref_features = None
            for name in ('original512', 'resampled512', 'resampled1024', 'resampled2048', 'step3cap2048'):
                tick = time.monotonic()
                if name == 'original512':
                    points, detail = original, None
                elif name == 'step3cap2048':
                    a, b, detail = sample_pair_step_contours(masks['a'], masks['b'])
                    points = dict(a=a, b=b)
                else:
                    cap = int(name[len('resampled'):])
                    points = {s:extract_ordered_outer_contour(masks[s].astype(bool), cap=cap, smoothing_sigma=3.)[0] for s in 'ab'}
                    detail = dict(cap=cap, smoothing_sigma=3.)
                measured, features = forward(model, masks, points, cfg, decoder)
                measured.update(variant=name, sampling=detail)
                if name == 'original512':
                    error = abs(measured['logit'] - case['raw_head_logit'])
                    if error > 2e-4:
                        raise ValueError('historical original512 replay failed: ' + str(error))
                    measured['historical_logit_replay_error'] = error
                elif name == 'resampled512':
                    ref_features = features
                    measured['anchor_512'] = anchor_head(model.score_head, reference, points, ref_features, features)
                else:
                    measured['anchor_512'] = anchor_head(model.score_head, reference, points, ref_features, features)
                # Labels/pose targets are diagnostics only, never inputs above.
                gt = case.get('target_translation_rc') if case.get('layout_gt_available') and case['label'] else None
                measured['layout']['translation_l2_px'] = (float(np.linalg.norm(np.asarray(measured['layout']['t_a_to_b_rc']) - gt))
                    if gt is not None and measured['layout']['valid'] else None)
                measured['seconds'] = time.monotonic() - tick
                row['variants'].append(measured)
                save(out / 'partial_results.json', rows + [row])
                print(json.dumps(dict(name=row['name'], variant=name, probability=measured['probability'],
                    layout_error=measured['layout']['translation_l2_px'], seconds=measured['seconds']), ensure_ascii=False), flush=True)
            rows.append(row)
        receipt.update(status='complete', completed_cases=len(rows), elapsed_seconds=time.monotonic()-started)
        save(out / 'results.json', dict(rows=rows, protocol=receipt))
    except BaseException as error:
        receipt.update(status='failed', completed_cases=len(rows), error=repr(error))
        raise
    finally:
        save(out / 'protocol.json', receipt)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'probe-root', 'prior-results', 'output'):
        p.add_argument('--'+name, required=True)
    run(p.parse_args())

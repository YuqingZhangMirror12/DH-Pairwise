"""Read saved Q only: full matrix -> Top2 union -> original cap512.

CPU numpy diagnostic, no model inference/optimization and no production edits.
GT is an after-the-fact support reference, not a decoder input.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effective_count(values):
    total, squares = float(values.sum()), float((values ** 2).sum())
    return total * total / squares if squares else 0.


def support(q, displacement_error, selected, radius):
    geometric = displacement_error <= radius
    mask = selected & geometric
    weighted = np.where(mask, q, 0.)
    qa, qb = weighted.sum(1), weighted.sum(0)
    mass = float(weighted.sum())
    return dict(geometric_cells_in_stage=int(mask.sum()), positive_q_cells=int((mask & (q > 0)).sum()),
        raw_q_mass=mass, mean_q_per_geometric_cell=mass / int(mask.sum()) if mask.any() else None,
        fraction_full_q_mass=mass / float(q.sum()) if q.sum() else None,
        unique_a=int((qa > 0).sum()), unique_b=int((qb > 0).sum()),
        effective_support_a=effective_count(qa), effective_support_b=effective_count(qb),
        max_q=float(q[mask].max()) if mask.any() else None)


def load_decoder(path):
    # Avoid importing models/__init__, which imports torch and unrelated models.
    spec = importlib.util.spec_from_file_location('saved_q_original_decoder', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    started = time.monotonic()
    root, output = Path(args.diagnosis_root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    decoder = load_decoder(Path(args.source_root) / 'staging/pairwise_v0_2/models/translation_layout.py')
    selected_path = Path(args.selection_json)
    selected = json.loads(selected_path.read_text())
    wanted = [r for r in selected if r['dataset'] == 'real' and r['stratum'] == 'real_TP_bad']
    for stratum in ('real_TP_good', 'real_FN_good', 'real_FP', 'real_TN'):
        wanted.append(next(r for r in selected if r['stratum'] == stratum))
    if len(wanted) != 8:
        raise ValueError('expected frozen four TP-bad plus four fixed first-stratum controls')
    cases_path = root / 'heatmaps_v1/s6/cases.json'
    cases = {r['pair_id']:r for r in json.loads(cases_path.read_text())}
    rows = []
    for sample in wanted:
        case = cases[sample['pair_id']]
        array_path = cases_path.parent / case['arrays_path']
        if sha(array_path) != case['arrays_sha256']:
            raise ValueError('saved model tensors changed')
        with np.load(array_path, allow_pickle=False) as archive:
            q = archive['sinkhorn_assignment'].astype(np.float64)
        a, b = np.asarray(case['points_rc_a']), np.asarray(case['points_rc_b'])
        va, vb = np.asarray(case['valid_a'], bool), np.asarray(case['valid_b'], bool)
        valid = va[:, None] & vb[None, :]
        q = np.where(valid, q, 0.)
        cfg = decoder.TranslationLayoutConfig(**case['layout']['decoder_config'])
        original = decoder.estimate_translation_layout(a, b, q, va, vb, config=cfg)
        saved = case['layout']
        if (not original.valid or not np.array_equal(original.candidate_indices, saved['candidate_indices'])
                or not np.array_equal(original.inlier_mask, saved['inlier_mask'])
                or np.linalg.norm(original.t_a_to_b_rc - saved['t_a_to_b_rc']) > 1e-7):
            raise ValueError('baseline decoder did not replay')
        # Set cap above all possible cells: actual union still contains only Top2.
        uncut_cfg = replace(cfg, max_candidates=int(valid.sum()) + 1)
        union_indices, _, _ = decoder._candidates(a, b, q, va, vb, uncut_cfg)
        union, capped = np.zeros(q.shape, bool), np.zeros(q.shape, bool)
        union[union_indices[:, 0], union_indices[:, 1]] = True
        capped[original.candidate_indices[:, 0], original.candidate_indices[:, 1]] = True
        stages = dict(full=valid, top2_uncapped=union, top2_cap512=capped)
        delta = b[None, :, :] - a[:, None, :]
        centers = dict(predicted=original.t_a_to_b_rc)
        gt = case.get('target_translation_rc') if case.get('layout_gt_available') and case['label'] else None
        if gt is not None:
            centers['gt'] = np.asarray(gt)
        measured = {}
        for center_name, center in centers.items():
            distance = np.linalg.norm(delta - center, axis=2)
            entry = dict(minimum_valid_cell_residual_px=float(distance[valid].min()), radii={})
            for radius in (10, 20):
                values = {name:support(q, distance, mask, radius) for name, mask in stages.items()}
                full_mass = values['full']['raw_q_mass']
                union_mass = values['top2_uncapped']['raw_q_mass']
                cap_mass = values['top2_cap512']['raw_q_mass']
                entry['radii'][str(radius)] = dict(stages=values,
                    full_to_top2_mass_retention=union_mass / full_mass if full_mass else None,
                    top2_to_cap_mass_retention=cap_mass / union_mass if union_mass else None)
            measured[center_name] = entry
        # A single cap intervention; no GT, no score calibration, no TopK search.
        uncut_layout = decoder.estimate_translation_layout(a, b, q, va, vb, config=uncut_cfg)
        raw_variants = {}
        for name, result in (('original_cap512', original), ('uncapped_top2', uncut_layout)):
            raw_variants[name] = dict(candidate_count=result.candidate_count, valid=result.valid,
                translation_rc=result.t_a_to_b_rc.tolist() if result.valid else None,
                inlier_count=result.inlier_count, residual_px=result.residual_px,
                translation_gt_error_px=float(np.linalg.norm(result.t_a_to_b_rc - gt)) if result.valid and gt is not None else None)
        rows.append(dict(model='s6', pair_id=case['pair_id'], name=sample.get('name'),
            stratum=sample['stratum'], label=case['label'], saved_score=case['score'],
            arrays_sha256=case['arrays_sha256'], full_q_mass=float(q.sum()),
            stage_edge_counts={name:int(mask.sum()) for name, mask in stages.items()},
            support=measured, decoder_cap_intervention=raw_variants))
    output.mkdir(parents=True)
    result = dict(schema_version='saved-sinkhorn-truncation/1', rows=rows,
        cases_sha256=sha(cases_path), selection_sha256=sha(selected_path),
        decoder_sha256=sha(Path(args.source_root) / 'staging/pairwise_v0_2/models/translation_layout.py'),
        probe_sha256=sha(__file__), status='complete', elapsed_seconds=time.monotonic() - started,
        baseline_candidate_order_inliers_translation_replayed=True,
        network_inference_performed=False, parameters_or_thresholds_fitted=False, device='CPU numpy',
        sampling='four previously fixed REAL TP-bad and first TP-good/FN-good/FP/TN from fixed selection',
        caveats=['Illustrative cases, not population rates or trainable supervision.',
            'GT-consistent offsets are not exact point-level ground truth under corrosion.',
            'Effective support count is concentration, not ordered seam length or correctness.',
            'Negative pairs have no GT layout; only their predicted neighborhood is measured.',
            'Scorer remains unchanged; the uncapped decoder variant is a diagnostic, not deployed.'])
    (output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete', cases=len(rows), elapsed_seconds=result['elapsed_seconds'])), flush=True)
    for row in rows:
        if row['stratum'] == 'real_TP_bad':
            print(json.dumps(dict(name=row['name'], gt10=row['support']['gt']['radii']['10'],
                                  layouts=row['decoder_cap_intervention']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis-root', required=True)
    p.add_argument('--selection-json', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--output', required=True)
    run(p.parse_args())

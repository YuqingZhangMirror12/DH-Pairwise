"""TRAIN-only inherited-correspondence audit, not a model optimization.

No CAL/SELECT/real images, predicted matches or scores are read. Saved TRAIN ray
projections are diagnostics only: their long-tail gaps are NOT trusted token
correspondences or a calibration bound. Missing precise-anchor facts stay unknown.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from .geometry import compact_contour, pair_frame


RELIABLE_NORMAL_MIN = .5  # Registered geometric reliability cutoff, not fitted to real data.
COLUMNS = ('i', 'j', 'residual_r', 'residual_c', 'normal_px', 'tangent_px',
           'residual_norm_px', 'spacing_a_px', 'spacing_b_px', 'normal_reliability',
           'precise_anchor_known')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, weights=None):
    values = np.asarray(values, float)
    if not len(values):
        return dict(n=0, mean=None, p01=None, p50=None, p90=None, p95=None, p99=None)
    if weights is None:
        weights = np.ones(len(values), float)
    weights = np.asarray(weights, float)
    if not np.isfinite(values).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('invalid measured distribution')
    order = np.argsort(values, kind='stable')
    x, w = values[order], weights[order]
    cdf = (np.cumsum(w) - .5 * w) / w.sum()
    q = np.interp([.01, .5, .9, .95, .99], cdf, x)
    return dict(n=len(x), mean=float(np.average(x, weights=w)),
                **{k:float(v) for k,v in zip(('p01','p50','p90','p95','p99'),q)})


def inherited_edges(sample, recipe):
    """Only existing reciprocal targets. Never label proximity/notches positive."""
    if not sample.label or not sample.translation_valid:
        raise ValueError('calibration requires a TRAIN positive with GT translation')
    i = np.flatnonzero((sample.target_a >= 0) & sample.contour_valid_a)
    j = sample.target_a[i]
    if (not len(i) or np.any(j >= len(sample.target_b)) or
            not np.array_equal(sample.target_b[j], i) or not sample.contour_valid_b[j].all()):
        raise ValueError('missing or nonreciprocal inherited supervision')
    geometries = []
    for side in 'ab':
        p = torch.from_numpy(np.array(getattr(sample, 'points_rc_'+side), copy=True)).double()[None]
        v = torch.from_numpy(np.array(getattr(sample, 'contour_valid_'+side), copy=True))[None]
        geometries.append(compact_contour(p, v))
    ga, gb = geometries
    ia, jb = ga.original_to_compact[0, i], gb.original_to_compact[0, j]
    normal, tangent, reliability = pair_frame(ga.outward_normal_rc[0, ia],
        gb.outward_normal_rc[0, jb], ga.normal_reliability[0, ia], gb.normal_reliability[0, jb])
    residual = (gb.points[0, jb] - ga.points[0, ia] -
                torch.from_numpy(np.array(sample.translation_a_to_b_rc, copy=True)).double())
    # Clean means no corrosion, not necessarily no partial cut. Inherited clean
    # targets already exclude artificial cuts. No residual-based anchor selection.
    precise = np.full(len(i), recipe == 'clean', bool)
    return np.column_stack((i, j, residual.numpy(), (residual*normal).sum(-1).numpy(),
        (residual*tangent).sum(-1).numpy(), residual.norm(dim=-1).numpy(),
        ga.cell_px[0, ia].numpy(), gb.cell_px[0, jb].numpy(), reliability.numpy(), precise))


def worker_init():
    torch.set_num_threads(1)


def inspect_entry(item):
    index, entry, root = item
    root = Path(root)
    path = (root/entry['artifact_path']).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('sample is outside bound TRAIN root')
    sample, report = load_sample(path)
    recipe = entry['corrosion_recipe']
    if (sample.pair_id != entry['pair_id'] or not sample.label
            or report['compound']['recipe'] != recipe or entry['source_row']['split'] != 'train'):
        raise ValueError('TRAIN archive identity mismatch')
    edges = inherited_edges(sample, recipe)
    pair = dict(index=index, pair_id=entry['pair_id'], source_pair_id=entry['source_pair_id'],
        sample_sha256=sha(path), recipe=recipe, partial=bool(entry['partial_applied']),
        mirrored=entry['offline_paired_mirror'], source_support_edges=len(edges),
        precise_anchor_edges=int(edges[:, -1].sum()),
        source_class='clean_inherited' if recipe == 'clean' else 'corroded_inherited',
        changed_a=bool(report['changed_a']), changed_b=bool(report['changed_b']),
        weather_removed_area_px={s:int(report['side_'+s].get('weather_only_removed_area_px', 0)) for s in 'ab'},
        ignored_tokens={s:int(np.sum((getattr(sample,'target_'+s) == -2) &
                                    getattr(sample,'contour_valid_'+s))) for s in 'ab'})
    # These projected locations need not be sampled tokens, nor exact ancestors.
    # Keep their audit separate from inherited labels AND parameter calibration.
    latent_path = entry.get('latent_seam_artifact')
    if latent_path:
        lp = (root/latent_path).resolve()
        if root.resolve() not in lp.parents:
            raise ValueError('latent projection is outside TRAIN root')
        with np.load(lp, allow_pickle=False) as z:
            gaps, weights, retreats = [], [], []
            unresolved, total = 0., 0.
            for side in 'ab':
                valid = z[side+'_valid'].astype(bool)
                weight = z[side+'_source_weight'].astype(float)
                total += weight.sum()
                unresolved += weight[~valid].sum()
                gaps.append(z[side+'_gap'][valid])
                weights.append(weight[valid])
                retreats.append(z[side+'_recession'][valid])
            pair['projected_gap_px'] = quantile(np.concatenate(gaps), np.concatenate(weights))
            pair['measured_recession_px'] = quantile(np.concatenate(retreats), np.concatenate(weights))
            pair['projection_unresolved_fraction'] = float(unresolved/max(total, 1e-12))
            pair['latent_sha256'] = sha(lp)
    return pair, edges


def summarize_edges(edges, pair_index, chosen):
    mask = np.isin(pair_index, chosen)
    e, p = edges[mask], pair_index[mask]
    # Each pair contributes equal total weight. Long, dense seams do not define
    # all uncertainty parameters. Edge-unweighted stats are separately retained.
    counts = Counter(p.tolist())
    w = np.array([1/counts[k] for k in p], float)
    reliable = e[:, 9] >= RELIABLE_NORMAL_MIN
    summaries = {}
    for column, name in ((4,'normal_px'), (5,'tangent_px'), (6,'residual_norm_px'),
                         (7,'spacing_a_px'), (8,'spacing_b_px'), (9,'normal_reliability')):
        use = reliable if column in (4,5) else np.ones(len(e), bool)
        summaries[name] = quantile(e[use,column], w[use])
    spacing = np.maximum(.5*(e[:,7]+e[:,8]), 1e-6)
    for column, name in ((4,'abs_normal_per_spacing'), (5,'abs_tangent_per_spacing'),
                         (6,'norm_per_spacing')):
        use = reliable if column in (4,5) else np.ones(len(e), bool)
        summaries[name] = quantile(np.abs(e[use,column])/spacing[use], w[use])
    return dict(pair_count=len(chosen), edges=len(e), reliable_normal_edges=int(reliable.sum()),
        reliable_normal_fraction=float(reliable.mean()) if len(e) else None,
        far_residual_edges=int((e[:,6] >= 15).sum()),
        precise_anchor_edges=int(e[:,10].sum()), pair_weighted=summaries,
        edge_weighted_residual_norm_px=quantile(e[:,6]))


def derive_parameters(summaries):
    """Only reliable inherited edges define parameters, not dense projection tails."""
    clean = summaries['recipe:clean']['pair_weighted']
    damaged = summaries['corroded']['pair_weighted']
    if summaries['recipe:clean']['reliable_normal_edges'] < 100 or summaries['corroded']['reliable_normal_edges'] < 100:
        raise ValueError('insufficient reliable TRAIN anchors/support for calibration')
    return dict(normal_reliability_min=RELIABLE_NORMAL_MIN,
        normal_sigma_per_spacing=max(.1, clean['abs_normal_per_spacing']['p95']/1.96),
        tangent_sigma_per_spacing=max(.1, clean['abs_tangent_per_spacing']['p95']/1.96),
        evidence_tangent_sigma_per_spacing=max(.1, clean['abs_tangent_per_spacing']['p95']/1.96,
                                               damaged['abs_tangent_per_spacing']['p95']/1.96),
        fallback_sigma_per_spacing=max(.1, clean['norm_per_spacing']['p95']/2.448),
        sigma_floor_px=1.,
        damage_normal_upper_px=float(np.ceil(max(damaged['normal_px']['p99'],1.))),
        damage_normal_lower_px=0., damage_tangent_offset_px=0.,
        unreliable_normal_damage_offset_px=0.,
        interpretation='finite positive-normal interval; separately calibrated evidence vs location tangential Sigma; unreliable normal has no signed compensation')


def run(contract_path, output, workers):
    output = Path(output)
    if output.exists():
        raise ValueError('preserve audit receipts; choose a new output directory')
    contract = json.loads(Path(contract_path).read_text())
    manifest_path = Path(contract['train']['archive_manifest'])
    if (contract['status'] != 'passed' or not contract['source_disjoint'] or
            sha(manifest_path) != contract['train']['archive_manifest_sha256']):
        raise ValueError('completed source-isolated TRAIN contract differs')
    manifest = json.loads(manifest_path.read_text())
    if manifest['split'] != 'train' or len(manifest['entries']) != 24000:
        raise ValueError('requires the registered full24K TRAIN population')
    root = manifest_path.parent
    positive = [e for e in manifest['entries'] if e['label']]
    if len(positive) != 12000:
        raise ValueError('requires all12000 positive pairs')
    output.mkdir(parents=True)
    start = time.time()
    tasks = [(i,e,str(root)) for i,e in enumerate(positive)]
    pairs, blocks, indices = [], [], []
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
            for pair, edges in pool.map(inspect_entry, tasks, chunksize=16):
                pairs.append(pair); blocks.append(edges)
                indices.append(np.full(len(edges), pair['index'], np.int32))
        edges, pair_index = np.concatenate(blocks), np.concatenate(indices)
        strata = {'all':list(range(len(pairs)))}
        for recipe in sorted(set(e['recipe'] for e in pairs)):
            strata['recipe:'+recipe] = [e['index'] for e in pairs if e['recipe'] == recipe]
        strata['corroded'] = [e['index'] for e in pairs if e['recipe'] != 'clean']
        strata['partial'] = [e['index'] for e in pairs if e['partial']]
        summaries = {k:summarize_edges(edges,pair_index,v) for k,v in strata.items()}
        projected = [e['projected_gap_px']['p99'] for e in pairs if e['recipe'] != 'clean']
        # Registered conservative first configuration, NOT a validation optimum.
        # Bound uses reliable inherited residuals; neither requested10–30 nor
        # layout_success20 appears in its calculation.
        parameters = derive_parameters(summaries)
        np.savez_compressed(output/'inherited_edge_audit.npz',
            values=edges.astype(np.float32), pair_index=pair_index,
            columns=np.array(COLUMNS), source_support_known=np.ones(len(edges), bool))
        (output/'pairs.json').write_text(json.dumps(pairs, ensure_ascii=False, allow_nan=False)+'\n')
        record = dict(status='complete', schema='s7-consensus-train-geometry/2',
            contract=str(Path(contract_path).resolve()), contract_sha256=sha(contract_path),
            manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
            pairs=len(pairs), inherited_edges=len(edges), elapsed_seconds=time.time()-start,
            strata=summaries, parameters=parameters,
            calibration_rule='Pair-equal clean p95 divided by1.96 (2D2.448) for localization Sigma; corroded tangential p95/1.96 for evidence Sigma; ceil(corroded inherited signed-normal p99) for finite damage extent. Only reliable inherited edges calibrate; dense ray projections never set tolerances.',
            projected_pair_p99_distribution=quantile(projected),
            source_support_known='Only existing reciprocal TRAIN targets; does not include filled-in sparse intervals or artificial cuts.',
            precise_anchor_known='Only inherited targets in clean-corrosion recipe. No residual threshold converts damaged support to precise anchors.',
            missing='Per-token original ancestor IDs and displacements are not in current archives; precise surviving anchors inside corroded samples remain unknown. No inference from GT residual is used to label them.',
            projection_caveat='Saved dense TRAIN ray projections are diagnostic only, not precise token ancestry, match/local labels or tolerance calibration. Their long tails and unresolved projections remain reported separately.',
            model_inputs_note='No GT/recipe/anchor flag is an inference feature; these facts are calibration/supervision only.',
            selection_note='No CAL/SELECT or real sample/score read; initial geometric parameters are not claimed optimal.',
            artifacts={p.name:sha(p) for p in (output/'pairs.json', output/'inherited_edge_audit.npz')},
            code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
        (output/'geometry_calibration.json').write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        lines=['# TRAIN-only geometry calibration', '',
            f"{len(pairs):,} positive pairs; {len(edges):,} inherited reciprocal edges. No optimization or real evaluation.", '',
            '| Stratum | Pairs | Edges | Reliable normal | Residual ≥15px |',
            '| --- | ---: | ---: | ---: | ---: |']
        for name, r in summaries.items():
            lines.append(f"| {name} | {r['pair_count']} | {r['edges']} | {r['reliable_normal_fraction']:.3%} | {r['far_residual_edges']} |")
        lines += ['', '## Frozen initial parameters', '', '```json', json.dumps(parameters,indent=2), '```', '',
            record['calibration_rule'], '', record['precise_anchor_known'], '', record['missing'], '',
            record['projection_caveat'], '', record['selection_note']]
        (output/'geometry_calibration_report.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps({k:record[k] for k in ('status','pairs','inherited_edges','elapsed_seconds','parameters')}))
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(exc), elapsed_seconds=time.time()-start), indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error('workers must be1…64')
    run(args.contract, args.out, args.workers)

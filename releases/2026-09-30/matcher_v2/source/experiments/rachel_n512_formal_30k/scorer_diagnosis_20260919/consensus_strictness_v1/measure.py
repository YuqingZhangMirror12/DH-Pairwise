"""Read-only full-population measurement using the bound production builder.

Runs in an isolated postprocess directory with the immutable production source
first on PYTHONPATH. No training, model selection, or TEST access.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1 import pose_consensus_repair
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.data import Dataset, collate
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import PairEvidence, observed_arc_cells
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.geometry import pair_frame
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.matcher import S7MatcherAdapter, INPUTS, SOURCE_SHA
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.pose_consensus import PoseConsensusBuilder, ProposalConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_consensus import make_cloud

FORMAL = Path('/root/autodl-tmp/s7_consensus_layered_v14_mergefix_20260925')
M12 = Path('/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt')
CV = Path('/root/autodl-tmp/rachel_score_design_20260913_001/real_domain_calibration_v1_20260921')
GT = Path('/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temp, path)


def quant(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, min=None, p10=None, p25=None, p50=None, p75=None, p90=None, p95=None, max=None, mean=None)
    v = np.quantile(a, [0, .1, .25, .5, .75, .9, .95, 1]).tolist()
    return dict(zip(['min', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'max'], v), n=len(a), mean=float(a.mean()))


def edge_set(ids):
    return {tuple(x) for x in ids.tolist()}


def endpoints(edges, side):
    return {x[side] for x in edges}


def ratio(a, b):
    return a / b if b else None


def set_stats(a, b):
    return dict(intersection=len(a & b), union=len(a | b), jaccard=ratio(len(a & b), len(a | b)),
                recall_of_reference=ratio(len(a & b), len(b)))


def model_digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def observe_builder(builder, pair):
    """Capture untruncated final list at return, without replacing any decisions."""
    captured = {}
    code = pose_consensus_repair.RepairedPoseConsensusBuilder.build_from_cloud.__wrapped__.__code__
    def observer(frame, event, value):
        if event == 'return' and frame.f_code is code:
            captured['clusters'] = tuple(frame.f_locals.get('clusters', ()))
    old = sys.getprofile()
    try:
        sys.setprofile(observer)
        result = builder(pair)
    finally:
        sys.setprofile(old)
    all_clusters = captured.get('clusters')
    if all_clusters is None:
        if pair.numeric_valid:
            raise RuntimeError('pre-budget observer did not capture production builder')
        all_clusters = ()
    if len(result.clusters) != min(len(all_clusters), builder.config.max_clusters):
        raise AssertionError('observer cluster count differs')
    for a, b in zip(result.clusters, all_clusters):
        if a is not b:
            raise AssertionError('observer changed production ordering or objects')
    return result, all_clusters


def arc_distance(cloud, edges_a, edges_b):
    """Minimum cyclic arc separation of supporting endpoints, not a merge gate."""
    index = {tuple(x): i for i, x in enumerate(cloud.ids.tolist())}
    out = {}
    for side in ('a', 'b'):
        a = torch.tensor([index[x] for x in edges_a], dtype=torch.long)
        b = torch.tensor([index[x] for x in edges_b], dtype=torch.long)
        if not len(a) or not len(b):
            out[side] = None; continue
        s = getattr(cloud, 'arc_' + side)
        perimeter = getattr(cloud, 'perimeter_' + side)
        distance = (s[a, None] - s[b][None]).abs()
        distance = torch.minimum(distance, (perimeter - distance).clamp_min(0))
        out[side] = float(distance.min())
    return out


@torch.no_grad()
def measure_pair(builder, pair, proposals, all_clusters):
    cloud = proposals.cloud
    cfg = builder.geometry
    gate = math.exp(-.5 * builder.config.membership_sigma ** 2)
    spacing = .5 * (cloud.spacing_a + cloud.spacing_b)
    radius = math.sqrt(2) * builder.config.merge_sigma * builder._local_scale(cloud)
    se = (spacing * cfg.evidence_tangent_sigma_per_spacing).clamp_min(cfg.sigma_floor_px)
    sn = (spacing * cfg.normal_sigma_per_spacing).clamp_min(cfg.sigma_floor_px)
    sf = (spacing * cfg.fallback_sigma_per_spacing).clamp_min(cfg.sigma_floor_px)
    cloud_edges = edge_set(cloud.ids)
    hyps = []
    for i, h in enumerate(proposals.hypotheses):
        edges = edge_set(h.edge_ids)
        hyps.append(dict(index=i, seed_rc=proposals.seeds[i].tolist(), translation_rc=h.translation.tolist(),
            edge_ids=sorted(edges), unique_a=len(endpoints(edges, 0)), unique_b=len(endpoints(edges, 1)),
            support_mass_px=h.absolute_support_mass_px, underconstrained=h.underconstrained))
    aa, _ = observed_arc_cells(pair.ga, builder.config.observation_radius_px)
    ab, _ = observed_arc_cells(pair.gb, builder.config.observation_radius_px)
    arc_weight = .5 * (aa[:, None] + ab[None])
    q_mass_px = pair.q * arc_weight
    clusters = []
    max_kernel = torch.zeros_like(pair.q)
    for i, c in enumerate(all_clusters):
        edges = edge_set(c.edge_ids)
        union = edge_set(getattr(c, 'original_union_edge_ids', c.edge_ids))
        kernels = torch.cat([pair.compatibility(c.translation, cfg, j, min(j + 64, len(pair.local_a)))[1].kernel
                             for j in range(0, len(pair.local_a), 64)])
        w = pair.q * kernels
        usable = (kernels >= gate) & (pair.q >= builder.config.minimum_absolute_q)
        full_ids = edge_set(usable.nonzero())
        if i < len(proposals.clusters):
            max_kernel = torch.maximum(max_kernel, kernels)
        reference = torch.zeros(len(cloud.ids))
        for hi in c.merged_hypothesis_ids:
            h = proposals.hypotheses[hi]
            member = pose_consensus_repair.edge_membership(cloud.ids, h.edge_ids)
            reference = torch.maximum(reference, cloud.q * cloud.arc_weight * cloud.compatibility(h.translation, cfg).kernel * member)
        current_member = pose_consensus_repair.edge_membership(cloud.ids, c.edge_ids)
        current_mass = cloud.q * cloud.arc_weight * kernels[cloud.ids[:, 0], cloud.ids[:, 1]]
        clusters.append(dict(index=i, retained=i < len(proposals.clusters), translation_rc=c.translation.tolist(),
            hypothesis_ids=list(c.merged_hypothesis_ids), edge_ids=sorted(edges), original_union_ids=sorted(union),
            unique_a=len(endpoints(edges, 0)), unique_b=len(endpoints(edges, 1)),
            union_unique_a=len(endpoints(union, 0)), union_unique_b=len(endpoints(union, 1)),
            sparse_union_recall=ratio(len(edges & union), len(union)),
            reference_mass_lost_fraction=ratio(float(reference[~current_member].sum()), float(reference.sum())),
            sparse_weight_mass_ratio_to_original=ratio(float(current_mass.sum()), float(reference.sum())),
            support_mass_px=c.absolute_support_mass_px, overlap=c.overlap, underconstrained=c.underconstrained,
            full_q_weighted_mass=float(w.sum()), full_q_weighted_mass_px=float((w * arc_weight).sum()),
            full_q_absolute_mass_fraction=ratio(float(w.sum()), float(pair.q.sum())),
            full_q_significant_ids=sorted(full_ids), full_q_unique_a=len(endpoints(full_ids, 0)),
            full_q_unique_b=len(endpoints(full_ids, 1)),
            full_q_union_recall=ratio(len(full_ids & union), len(union))))
    pairwise = []
    normal, tangent, rel = pair_frame(cloud.normal_a, cloud.normal_b, cloud.reliability_a, cloud.reliability_b)
    index = {tuple(x): i for i, x in enumerate(cloud.ids.tolist())}
    for a, ca in enumerate(clusters):
        ea = set(map(tuple, ca['edge_ids']))
        for b in range(a + 1, len(clusters)):
            cb = clusters[b]; eb = set(map(tuple, cb['edge_ids']))
            delta = torch.tensor(cb['translation_rc']) - torch.tensor(ca['translation_rc'])
            shared = ea & eb
            ids = torch.tensor([index[x] for x in shared], dtype=torch.long)
            nr = None; tr = None; reliable_share = None
            if len(ids):
                weights = cloud.q[ids] * cloud.arc_weight[ids] * (rel[ids] >= cfg.normal_reliability_min)
                reliable_share = ratio(float(weights.sum()), float((cloud.q[ids] * cloud.arc_weight[ids]).sum()))
                if float(weights.sum()) > 0:
                    nr = float(((normal[ids] * delta).sum(1).abs() * weights).sum() / weights.sum())
                    tr = float(((tangent[ids] * delta).sum(1).abs() * weights).sum() / weights.sum())
            stats = set_stats(ea, eb)
            pairwise.append(dict(a=a, b=b, both_retained=ca['retained'] and cb['retained'],
                pose_distance_px=float(delta.norm()), delta_rc=delta.tolist(),
                **stats, equal_nonempty_edges=bool(ea and ea == eb),
                overlap_fraction_smaller=ratio(len(shared), min(len(ea), len(eb))),
                shared_normal_shift_px=nr, shared_tangent_shift_px=tr,
                shared_normal_reliable_mass_fraction=reliable_share,
                minimum_arc_separation_px=arc_distance(cloud, ea, eb)))
    retained = clusters[:len(proposals.clusters)]
    total = sum(c['support_mass_px'] for c in retained)
    max_c = retained[0] if retained else None
    return dict(numeric_valid=pair.numeric_valid, valid_points_a=len(pair.local_a), valid_points_b=len(pair.local_b),
        median_contour_spacing_a_px=float(pair.ga.cell_px[0].median()),
        median_contour_spacing_b_px=float(pair.gb.cell_px[0].median()),
        median_edgecloud_spacing_px=float(spacing.median()) if len(spacing) else None,
        cloud_edges=len(cloud_edges), local_scale_px=builder._local_scale(cloud), common_center_radius_px=radius,
        pair_prescreen_distance_px=2 * radius, tangent_half_bandwidth_px=quant(3 * se),
        normal_extra_half_bandwidth_px=quant(3 * sn), isotropic_fallback_radius_px=quant(3 * sf),
        normal_material_offset_range_px=[0, cfg.damage_normal_upper_px],
        absolute_q_mass=float(pair.q.sum()), unmatched_a_mean=float(pair.unmatched_a.mean()),
        unmatched_b_mean=float(pair.unmatched_b.mean()),
        all_q_kernel_union_mass_px=float((q_mass_px * max_kernel).sum()),
        hypothesis_count=len(hyps), nonempty_hypothesis_count=sum(bool(h['edge_ids']) for h in hyps),
        prebudget_cluster_count=len(clusters), retained_cluster_count=len(retained),
        dropped_cluster_count=len(clusters)-len(retained),
        largest_sparse_cluster_mass_share=ratio(max_c['support_mass_px'], total) if max_c else None,
        largest_fullq_share_union=ratio(max_c['full_q_weighted_mass_px'], float((q_mass_px * max_kernel).sum())) if max_c else None,
        hypotheses=hyps, clusters=clusters, cluster_pairs=pairwise, merge_trace=proposals.merge_trace)


def attach_label(row, item, target, exclusions):
    """Called only after frozen builder output/snapshot exists."""
    row = dict(row, label=int(item['label']), gt_known=target is not None,
        gt_excluded=item['pair_id'] in exclusions, target_translation_rc=target,
        recipe=item.get('recipe'), fold=item.get('fold'),
        source_family=item.get('source_family'), base_pair_id=item.get('base_pair_id'))
    def error(t):
        return float(np.linalg.norm(np.asarray(t)-np.asarray(target))) if target is not None else None
    for h in row['hypotheses']:
        h['gt_error_px'] = error(h['translation_rc'])
    correct_h = [h for h in row['hypotheses'] if h['edge_ids'] and h['gt_error_px'] is not None and h['gt_error_px'] <= 20]
    correct_ids = {h['index'] for h in correct_h}
    correct_union = set().union(*(set(map(tuple, h['edge_ids'])) for h in correct_h)) if correct_h else set()
    touching = []; touching_pre = []; mixed = []; mixed_pre = []
    for c in row['clusters']:
        c['gt_error_px'] = error(c['translation_rc'])
        c['contains_correct_hypothesis'] = bool(set(c['hypothesis_ids']) & correct_ids)
        c['contains_gt40_wrong_hypothesis'] = any(row['hypotheses'][hi]['gt_error_px'] is not None and row['hypotheses'][hi]['gt_error_px'] > 40 for hi in c['hypothesis_ids'])
        if c['contains_correct_hypothesis']:
            touching_pre.append(c['index'])
            if c['retained']: touching.append(c['index'])
        if c['contains_correct_hypothesis'] and c['contains_gt40_wrong_hypothesis']:
            mixed_pre.append(c['index'])
            if c['retained']: mixed.append(c['index'])
    correct_c = [c for c in row['clusters'] if c['retained'] and c['gt_error_px'] is not None and c['gt_error_px'] <= 20]
    largest = max(correct_c, key=lambda c:c['support_mass_px']) if correct_c else None
    def coverage(key):
        e = set(map(tuple, largest[key])) if largest else set()
        return dict(edges=ratio(len(e & correct_union), len(correct_union)),
            endpoints_a=ratio(len(endpoints(e, 0) & endpoints(correct_union, 0)), len(endpoints(correct_union, 0))),
            endpoints_b=ratio(len(endpoints(e, 1) & endpoints(correct_union, 1)), len(endpoints(correct_union, 1))))
    row['gt_diagnostic'] = None if target is None else dict(correct_hypothesis_count=len(correct_h),
        correct_hypothesis_cluster_count=len(touching), correct_hypothesis_cluster_count_prebudget=len(touching_pre),
        correct_union_edges=len(correct_union), correct_union_endpoints_a=len(endpoints(correct_union, 0)),
        correct_union_endpoints_b=len(endpoints(correct_union, 1)),
        correct_hypotheses_dropped=sum(not any(h['index'] in c['hypothesis_ids'] for c in row['clusters'] if c['retained']) for h in correct_h),
        mixed_cluster_count=len(mixed), mixed_cluster_count_prebudget=len(mixed_pre),
        correct_cluster_count=len(correct_c), largest_correct_cluster_id=largest['index'] if largest else None,
        coverage_retained=bool(correct_c), coverage_prebudget=any(c['gt_error_px'] <= 20 for c in row['clusters']),
        top_cluster_correct=bool(row['clusters'] and row['clusters'][0]['gt_error_px'] <= 20),
        largest_correct_original_union_coverage=coverage('original_union_ids'),
        largest_correct_compatible_coverage=coverage('edge_ids'),
        largest_correct_fullq_coverage=coverage('full_q_significant_ids'))
    return row


def initialize(formal, checkpoint, out, case_plan):
    global W
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    formal = Path(formal); out = Path(out)
    contract = read(formal/'data_contract.json')
    select = contract['validation']['select_mixed']
    dataset = Dataset(select['path'], select['sha256'])
    binding = read(formal/'formal_m12/scorer/proposal_cache/select_mixed/binding.json')
    geometry = CompatibilityConfig.from_calibration(read(formal/'geometry_calibration_v2/geometry_calibration.json'))
    builder = PoseConsensusBuilder(geometry, ProposalConfig(**binding['experiment']['config']['proposals']))
    model = S7MatcherAdapter.from_s7_m12(checkpoint).eval()
    real = {}
    for name, sub in [('dunhuang_cv','real'), ('turufan','ood')]:
        meta = read(CV/sub/'manifest.json')
        with np.load(Path(meta['prepared'])/'inputs.npz', allow_pickle=False) as z:
            arrays = {k:z[k] for k in ('packed_masks', 'points', 'valid')}
        real[name] = (meta, arrays, {f:i for i,f in enumerate(meta['fragment_ids'])})
    W = dict(formal=formal, out=out, dataset=dataset, builder=builder, model=model, real=real,
        initial_digest=model_digest(model), gt=None, case_plan=read(case_plan))


@torch.no_grad()
def worker(task):
    split, i = task
    start = time.time(); dataset = W['dataset']
    target = None
    if split == 'sim_select':
        item = dataset.entries[i]
        sample, report, _ = dataset[i]
        batch = collate([(sample, report, item)])
    else:
        meta, arrays, index = W['real'][split]
        item = meta['pairs'][i]; batch = {}
        for s in ('a','b'):
            ix = index[item['fragment_'+s+'_id']]
            batch['mask_'+s] = torch.from_numpy(np.unpackbits(arrays['packed_masks'][ix:ix+1], axis=-1).astype(np.float32)[:,None])
            batch['points_rc_'+s] = torch.from_numpy(arrays['points'][ix:ix+1].astype(np.float32))
            batch['contour_valid_'+s] = torch.from_numpy(arrays['valid'][ix:ix+1].astype(bool))
    output = W['model'](*(batch[k] for k in INPUTS))
    pair = PairEvidence.from_matcher(output, 0, batch['mask_a'], batch['mask_b'])
    proposals, all_clusters = observe_builder(W['builder'], pair)
    row = measure_pair(W['builder'], pair, proposals, all_clusters)
    row.update(pair_id=item['pair_id'], split=split, index=i, backend='CPU FP32, deterministic, batch1')
    relative = split+'/'+hashlib.sha256(item['pair_id'].encode()).hexdigest()+'.pt'
    raw = W['out']/'evidence'/relative
    raw.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(pair_id=item['pair_id'], proposals=proposals, prebudget_clusters=all_clusters,
        q=pair.q, unmatched_a=pair.unmatched_a, unmatched_b=pair.unmatched_b,
        points_a=pair.points_a, points_b=pair.points_b, geometry_a=pair.ga, geometry_b=pair.gb), raw)
    row['evidence_file'] = 'evidence/'+relative
    row['evidence_sha256'] = sha(raw)
    if model_digest(W['model']) != W['initial_digest']:
        raise AssertionError('frozen Matcher tensors changed')
    # GT enters only after all pose hypotheses, memberships, and evidence files
    # are fixed; never passed into inference/build_from_cloud.
    if split == 'sim_select' and sample.label and sample.translation_valid:
        target = np.asarray(sample.translation_a_to_b_rc).tolist()
    elif split == 'dunhuang_cv' and item['label']:
        if W['gt'] is None:
            W['gt'] = {x['pair_id']:x for x in read(GT)['positive_pairs']}
        gt = W['gt'][item['pair_id']]
        if (gt['fragment_a_token'], gt['fragment_b_token']) != (item['fragment_a_id'], item['fragment_b_id']):
            raise AssertionError('GT order mismatch')
        target = gt['translation_gt_a_to_b_rc']
    row = attach_label(row, item, target, set(W['case_plan']['user_confirmed_gt_exclusions']))
    if split == 'sim_select':
        cachekey = hashlib.sha256(item['pair_id'].encode()).hexdigest()
        cache = W['formal']/'formal_m12/scorer/proposal_cache/select_mixed'/cachekey[:2]/(cachekey+'.pt')
        if cache.exists():
            rec = torch.load(cache, map_location='cpu', weights_only=False)
            if rec['pair_id'] != item['pair_id']: raise AssertionError('cache ID mismatch')
            old = rec['proposals']; old_cloud = {tuple(ij):float(q) for ij,q in zip(old.cloud.ids.tolist(), old.cloud.q)}
            new_cloud = {tuple(ij):float(q) for ij,q in zip(proposals.cloud.ids.tolist(), proposals.cloud.q)}
            shared = old_cloud.keys() & new_cloud.keys()
            row['live_cache_parity'] = dict(cached_clusters=len(old.clusters), measured_clusters=len(proposals.clusters),
                edge_jaccard=ratio(len(shared), len(old_cloud.keys() | new_cloud.keys())),
                q_max_abs_difference=max((abs(old_cloud[x]-new_cloud[x]) for x in shared), default=0.),
                ordered_pose_max_difference=max((float((a.translation-b.translation).norm()) for a,b in zip(old.clusters, proposals.clusters)), default=None),
                cache_sha256=sha(cache))
    row['elapsed_seconds'] = time.time()-start
    save(W['out']/split/(str(i).zfill(5)+'.json'), row)
    return dict(split=split, index=i, pair_id=item['pair_id'], seconds=row['elapsed_seconds'])


def synthetic(builder):
    records = []
    for spacing in (3.,4.,5.):
        for name in ('four_tangent', 'four_diagonal', 'corroded_seam8', 'separated60', 'drift80'):
            if name.startswith('four'):
                centers=[0.,4.,10.,15.,200.]
            elif name == 'separated60': centers=[0.,60.]
            elif name == 'drift80': centers=[0.,20.,40.,60.,80.]
            else: centers=[0.]
            cloud = make_cloud(centers)
            cloud = replace(cloud, spacing_a=torch.full_like(cloud.spacing_a, spacing),
                spacing_b=torch.full_like(cloud.spacing_b, spacing), arc_weight=torch.full_like(cloud.arc_weight, spacing))
            seeds=torch.tensor([[x,0.] for x in centers])
            if name == 'four_diagonal':
                vector=torch.tensor([math.sqrt(.5),math.sqrt(.5)])
                cloud=replace(cloud, displacement=cloud.displacement[:,0,None]*vector)
                seeds=seeds[:,0,None]*vector
            if name == 'corroded_seam8':
                cloud=replace(cloud, displacement=cloud.displacement+torch.tensor([0.,8.]))
                seeds=None  # production finite-damage seed votes, not hand-picked zero
            out=builder.build_from_cloud(cloud,seeds)
            shifted=replace(cloud, arc_a=cloud.arc_a*100, arc_b=cloud.arc_b*100,
                perimeter_a=cloud.perimeter_a*100, perimeter_b=cloud.perimeter_b*100)
            fixed_seeds=out.seeds
            again=builder.build_from_cloud(shifted,fixed_seeds)
            arc_invariant=(len(out.clusters)==len(again.clusters) and all(torch.equal(a.translation,b.translation) and torch.equal(a.edge_ids,b.edge_ids)
                           for a,b in zip(out.clusters,again.clusters)))
            records.append(dict(name=name, spacing_px=spacing, common_center_radius_px=math.sqrt(2)*builder.config.merge_sigma*builder._local_scale(cloud),
                clusters=[dict(translation_rc=c.translation.tolist(),edges=len(c.edge_ids),hypothesis_ids=c.merged_hypothesis_ids,
                    union_edges=len(getattr(c,'original_union_edge_ids',c.edge_ids))) for c in out.clusters],
                seeds=out.seeds.tolist(), arc_distance_change_fixed_seeds_invariant=arc_invariant))
    return records


def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    formal=Path(args.formal); contract=read(formal/'data_contract.json')
    binding=read(formal/'formal_m12/scorer/proposal_cache/select_mixed/binding.json')
    package=Path(inspect.getfile(pose_consensus_repair)).parent
    for name,value in binding['experiment']['implementation_sha256'].items():
        if (package/name).exists() and sha(package/name)!=value:
            raise ValueError('production code hash mismatch: '+name)
    calibration=formal/'geometry_calibration_v2/geometry_calibration.json'
    if sha(calibration)!=binding['experiment']['geometry_calibration_sha256'] or sha(args.checkpoint)!=SOURCE_SHA:
        raise ValueError('frozen model/calibration mismatch')
    cfg=CompatibilityConfig.from_calibration(read(calibration))
    builder=PoseConsensusBuilder(cfg,ProposalConfig(**binding['experiment']['config']['proposals']))
    expected={'sim_select':1500,'dunhuang_cv':803,'turufan':602}
    tasks=[(s,i) for s,n in expected.items() for i in range(n)]
    protocol=dict(schema='consensus-strictness/1', scope='authorized phase1; no changed algorithm, constants or training',
        expected=expected, formal_root=str(formal), source=str(package), source_hashes=binding['experiment']['implementation_sha256'],
        checkpoint=str(args.checkpoint), checkpoint_sha256=SOURCE_SHA, geometry=asdict(cfg), proposal_config=asdict(builder.config),
        calibration_sha256=sha(calibration), select_spec=contract['validation']['select_mixed'],
        real_manifests={s:dict(path=str(CV/p/'manifest.json'),sha256=sha(CV/p/'manifest.json')) for s,p in [('dunhuang_cv','real'),('turufan','ood')]},
        cases=read(args.case_plan), script_sha256=sha(__file__), guide_sha256=args.guide_sha,
        inference='CPU FP32 batch1; no training GPU allocation; actual inputs unchanged; GT joined after frozen geometry',
        prebudget_observation='Python return observer captures original list before budget slicing, without changing outputs',
        full_q_significance_definition='diagnostic only: original Q>=1e-4 and compatibility>=exp(-4.5); scoring keeps all nonzero weights',
        membership_bandwidth='reported pure tangent half-width at zero unexplained normal residual, not an isotropic tolerance',
        correct_hypothesis_definition='nonempty support and Euclidean 2D GT error<=20px; diagnostic, never a merge gate',
        contamination='a merged cluster contains both a <=20px hypothesis and a >40px hypothesis; 20..40 is intermediate',
        largest_cluster='maximum existing sparse absolute support mass, never selected using new scorer',
        workers=args.workers, started_at=time.time())
    for s in expected: (out/s).mkdir()
    save(out/'protocol.json',protocol)
    torch.set_num_threads(1)
    save(out/'synthetic_production_scale.json',synthetic(builder))
    counts=Counter(); start=time.time()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                initializer=initialize,initargs=(args.formal,args.checkpoint,args.out,args.case_plan)) as pool:
            for result in pool.map(worker,tasks,chunksize=1):
                counts[result['split']]+=1
                save(out/'status.json',dict(status='measuring',completed=dict(counts),expected=expected,elapsed_seconds=time.time()-start,pid=os.getpid()))
        if dict(counts)!=expected: raise AssertionError('incomplete populations')
        save(out/'complete.json',dict(status='complete',counts=dict(counts),elapsed_seconds=time.time()-start,protocol_sha256=sha(out/'protocol.json')))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),completed=dict(counts),elapsed_seconds=time.time()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--formal',default=str(FORMAL));p.add_argument('--checkpoint',default=str(M12))
    p.add_argument('--out',required=True);p.add_argument('--case-plan',required=True);p.add_argument('--guide-sha',required=True)
    p.add_argument('--workers',type=int,default=16);run(p.parse_args())

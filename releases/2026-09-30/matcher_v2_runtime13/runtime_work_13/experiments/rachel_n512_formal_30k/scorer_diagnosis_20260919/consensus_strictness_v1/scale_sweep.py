"""Replay frozen phase1 proposals; select on SIM SELECT, then report real only."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch

from measure import (FORMAL, CV, Dataset, CompatibilityConfig, ProposalConfig, PairEvidence,
    observed_arc_cells, read, save, sha, quant, edge_set, endpoints, synthetic)
from pose_scale import IsotropicPoseBuilder, PoseScalePolicy
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import material_overlap

VARIANTS = [('s8', PoseScalePolicy(8.)), ('s10', PoseScalePolicy(10.)),
            ('s12', PoseScalePolicy(12.)), ('s16', PoseScalePolicy(16.)),
            ('adaptive', PoseScalePolicy(adaptive=True))]
PHASE1 = Path('/root/autodl-tmp/consensus_strictness_20260925/results_phase1')


def metrics(row, proposals, clusters, pair, builder):
    """GT is accessed only after ALL inference decisions have been made."""
    truth = row['target_translation_rc']
    def error(p):
        return None if truth is None else float(np.linalg.norm(np.asarray(p)-np.asarray(truth)))
    h = proposals.hypotheses
    correct = {i for i, x in enumerate(h) if len(x.edge_ids) and truth is not None and error(x.translation.tolist()) <= 20.}
    wrong40 = {i for i, x in enumerate(h) if len(x.edge_ids) and truth is not None and error(x.translation.tolist()) > 40.}
    wrong20 = {i for i, x in enumerate(h) if len(x.edge_ids) and truth is not None and error(x.translation.tolist()) > 20.}
    reference = set().union(*(edge_set(h[i].edge_ids) for i in correct)) if correct else set()
    crows = []
    for i, c in enumerate(clusters):
        e = edge_set(c.edge_ids)
        union = edge_set(getattr(c, 'original_union_edge_ids', c.edge_ids))
        members = set(c.merged_hypothesis_ids)
        crows.append(dict(index=i, retained=i < len(proposals.clusters), translation_rc=c.translation.tolist(),
            gt_error_px=error(c.translation.tolist()), hypothesis_ids=sorted(members),
            edge_ids=sorted(e), original_union_ids=sorted(union), support_mass_px=c.absolute_support_mass_px,
            overlap=c.overlap, underconstrained=c.underconstrained,
            contains_correct=bool(members & correct), mixed20_40=bool(members & correct and members & wrong40),
            mixed20_20=bool(members & correct and members & wrong20),
            center_distance_max_px=float(torch.stack([(h[j].translation-c.translation).norm() for j in members]).max()),
            mass_loss_guard=getattr(c, 'maximum_lost_explained_mass_fraction', None)))
    kept = [c for c in crows if c['retained']]
    good = [c for c in kept if c['gt_error_px'] is not None and c['gt_error_px'] <= 20]
    best_good = max(good, key=lambda c: c['support_mass_px']) if good else None
    full_target = best_good if row['label'] else (kept[0] if kept else None)
    full_ids = set()
    if full_target is not None:
        pose = pair.q.new_tensor(full_target['translation_rc'])
        kernels = torch.cat([pair.compatibility(pose, builder.geometry, start, min(start+64, len(pair.local_a)))[1].kernel
            for start in range(0, len(pair.local_a), 64)])
        weights = pair.q * kernels
        full_ids = edge_set(((pair.q >= builder.config.minimum_absolute_q) &
            (kernels >= math.exp(-.5 * builder.config.membership_sigma**2))).nonzero())
        aa, _ = observed_arc_cells(pair.ga, builder.config.observation_radius_px)
        ab, _ = observed_arc_cells(pair.gb, builder.config.observation_radius_px)
        full_target['full_q_mass'] = float(weights.sum())
        full_target['full_q_mass_px'] = float((weights * .5 * (aa[:, None]+ab[None])).sum())
        full_target['full_q_fraction_all_q'] = float(weights.sum()/pair.q.sum().clamp_min(1e-30))

    def cover(edges):
        return dict(edges=len(edges & reference)/len(reference) if reference else None,
            endpoints_a=len(endpoints(edges, 0) & endpoints(reference, 0))/len(endpoints(reference, 0)) if reference else None,
            endpoints_b=len(endpoints(edges, 1) & endpoints(reference, 1))/len(endpoints(reference, 1)) if reference else None)
    touches = [c for c in kept if c['contains_correct']]
    complete = bool(correct and len(touches) == 1 and correct <= set(touches[0]['hypothesis_ids'])
        and touches[0]['gt_error_px'] <= 20 and not touches[0]['mixed20_40'])
    pairs = []
    for n, a in enumerate(kept):
        ea = set(map(tuple, a['edge_ids']))
        for b in kept[n+1:]:
            eb = set(map(tuple, b['edge_ids']))
            if ea or eb:
                pairs.append(len(ea & eb)/len(ea | eb))
    return dict(pair_id=row['pair_id'], index=row['index'], split=row['split'], label=row['label'],
        gt_known=row['gt_known'], gt_excluded=row['gt_excluded'], recipe=row['recipe'],
        source_family=row.get('source_family'), base_pair_id=row.get('base_pair_id'),
        evidence_file=row['evidence_file'], evidence_sha256=row['evidence_sha256'],
        hypothesis_count=len(h), correct_hypothesis_count=len(correct), correct_hypothesis_ids=sorted(correct),
        complete_correct_cluster=complete, prebudget_cluster_count=len(crows), retained_cluster_count=len(kept),
        hypothesis_coverage=bool(correct), coverage_retained=bool(good),
        coverage_prebudget=any(c['gt_error_px'] is not None and c['gt_error_px'] <= 20 for c in crows),
        top_cluster_correct=bool(kept and kept[0]['gt_error_px'] is not None and kept[0]['gt_error_px'] <= 20),
        correct_cluster_count=len(good), correct_hypothesis_cluster_count=len(touches),
        mixed20_40=sum(c['mixed20_40'] for c in crows), mixed20_20=sum(c['mixed20_20'] for c in crows),
        largest_correct_original_union_coverage=cover(set(map(tuple, best_good['original_union_ids'])) if best_good else set()),
        largest_correct_compatible_coverage=cover(set(map(tuple, best_good['edge_ids'])) if best_good else set()),
        largest_correct_fullq_coverage=cover(full_ids if best_good else set()),
        largest_sparse_mass_px=kept[0]['support_mass_px'] if kept else 0.,
        largest_sparse_mass_share=kept[0]['support_mass_px']/sum(c['support_mass_px'] for c in kept) if kept and sum(c['support_mass_px'] for c in kept)>0 else None,
        negative_top_full_q_mass=(full_target.get('full_q_mass') if full_target is not None and not row['label'] else None),
        negative_top_full_q_fraction_all_q=(full_target.get('full_q_fraction_all_q') if full_target is not None and not row['label'] else None),
        jaccard090_remaining=sum(x >= .9 for x in pairs), clusters=crows)


def initialize(phase1, output, formal):
    global W
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    p1 = read(Path(phase1)/'protocol.json')
    geometry = CompatibilityConfig(**p1['geometry'])
    builders = {key: IsotropicPoseBuilder(geometry, ProposalConfig(**p1['proposal_config']), policy) for key, policy in VARIANTS}
    ds = Dataset(p1['select_spec']['path'], p1['select_spec']['sha256'])
    W = dict(phase1=Path(phase1), out=Path(output), builders=builders, ds=ds, real={})


def real_masks(split, row):
    if split not in W['real']:
        path = CV / ('real' if split == 'dunhuang_cv' else 'ood') / 'manifest.json'
        meta = read(path)
        with np.load(Path(meta['prepared'])/'inputs.npz', allow_pickle=False) as z:
            masks = z['packed_masks']
        W['real'][split] = (meta, masks, {f:i for i, f in enumerate(meta['fragment_ids'])})
    meta, masks, index = W['real'][split]
    item = meta['pairs'][row['index']]
    assert item['pair_id'] == row['pair_id']
    return [torch.from_numpy(np.unpackbits(masks[index[item['fragment_'+s+'_id']]], axis=-1).astype(np.float32)) for s in ('a','b')]


@torch.no_grad()
def worker(task):
    split, i = task
    row = read(W['phase1']/split/(str(i).zfill(5)+'.json'))
    path = W['phase1']/row['evidence_file']
    if sha(path) != row['evidence_sha256']:
        raise ValueError('frozen phase1 evidence changed')
    raw = torch.load(path, map_location='cpu', weights_only=False)
    if raw['pair_id'] != row['pair_id']:
        raise AssertionError('raw pair mismatch')
    if split == 'sim_select':
        sample, _, _ = W['ds'][i]
        masks = [torch.from_numpy(np.array(m.squeeze(), dtype=np.float32, copy=True)) for m in (sample.mask_a, sample.mask_b)]
    else:
        masks = real_masks(split, row)
    na, nb = raw['q'].shape
    dummy_a = torch.zeros((na, 1)); dummy_b = torch.zeros((nb, 1))
    pair = PairEvidence(dummy_a, dummy_b, dummy_a, dummy_b, raw['q'], raw['unmatched_a'], raw['unmatched_b'],
        raw['geometry_a'], raw['geometry_b'], torch.arange(na), torch.arange(nb), *masks, row['numeric_valid'])
    baseline = raw['proposals']
    result = {}
    for name, builder in W['builders'].items():
        proposals = builder.build_from_cloud(baseline.cloud, baseline.seeds,
            overlap_fn=lambda t: material_overlap(pair, t), frozen_hypotheses=baseline.hypotheses)
        # Labels have not been passed to any builder method above.
        r = metrics(row, proposals, builder.all_clusters, pair, builder)
        r['variant'] = name
        r['audit'] = dict(builder.audit)
        r['merge_trace'] = list(proposals.merge_trace)
        save(W['out']/name/split/(str(i).zfill(5)+'.json'), r)
        result[name] = {k:v for k,v in r.items() if k not in ['clusters','merge_trace']}
    return result


def summarize(rows):
    gt = [r for r in rows if r['label'] and r['gt_known'] and not r['gt_excluded']]
    negatives = [r for r in rows if not r['label']]
    summary = dict(pairs=len(rows), positives=sum(r['label'] for r in rows),
        gt_positive_pairs=len(gt), negative_pairs=len(negatives),
        complete_correct_count=sum(r['complete_correct_cluster'] for r in gt),
        complete_correct_fraction=sum(r['complete_correct_cluster'] for r in gt)/len(gt) if gt else None,
        hypothesis_coverage=sum(r['hypothesis_coverage'] for r in gt),
        prebudget_coverage=sum(r['coverage_prebudget'] for r in gt),
        retained_coverage=sum(r['coverage_retained'] for r in gt),
        budget_loss=sum(r['coverage_prebudget'] and not r['coverage_retained'] for r in gt),
        top_cluster_correct=sum(r['top_cluster_correct'] for r in gt),
        mixed20_40=sum(r['mixed20_40'] for r in gt), mixed20_20=sum(r['mixed20_20'] for r in gt),
        jaccard090_pairs=sum(r['jaccard090_remaining'] > 0 for r in rows),
        radius_px=quant([r['audit']['radius_px'] for r in rows]),
        clusters=quant([r['retained_cluster_count'] for r in rows]),
        negatives=dict(largest_sparse_mass_px=quant([r['largest_sparse_mass_px'] for r in negatives]),
            largest_sparse_mass_share=quant([r['largest_sparse_mass_share'] for r in negatives if r['largest_sparse_mass_share'] is not None]),
            top_full_q_mass=quant([r['negative_top_full_q_mass'] for r in negatives if r['negative_top_full_q_mass'] is not None]),
            top_full_q_fraction_all_q=quant([r['negative_top_full_q_fraction_all_q'] for r in negatives if r['negative_top_full_q_fraction_all_q'] is not None]),
            clusters=quant([r['retained_cluster_count'] for r in negatives])),
        audit_totals={k:sum(r['audit'][k] for r in rows) for k in rows[0]['audit'] if k != 'radius_px'} if rows else {})
    for kind in ('original_union', 'compatible', 'fullq'):
        key = 'largest_correct_'+kind+'_coverage'
        summary[kind+'_coverage'] = {s:quant([r[key][s] for r in gt if r[key][s] is not None]) for s in ('edges','endpoints_a','endpoints_b')}
    return summary


def controls(builder):
    rows = synthetic(builder)
    for r in rows:
        c = r['clusters']
        name = r['name']
        if name == 'separated60':
            r['passed'] = len(c) == 2 and all(len(x['hypothesis_ids']) == 1 for x in c)
        elif name == 'drift80':
            r['passed'] = len(c) > 1 and all(len(x['hypothesis_ids']) < 5 for x in c)
        elif name.startswith('four'):
            r['passed'] = len(c) == 2 and sorted(len(x['hypothesis_ids']) for x in c) == [1,4]
        else:
            r['passed'] = len(c) == 1
        r['s_pose_px'] = max(10., 3*r['spacing_px']) if builder.scale_policy.adaptive else builder.scale_policy.radius_px
        # This was an old formula printed by the shared phase1 fixture, not the
        # scale used by this builder; remove it to avoid a misleading label.
        del r['common_center_radius_px']
    return rows


def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    original = read(Path(args.phase1)/'protocol.json')
    assert read(Path(args.phase1)/'complete.json')['status'] == 'complete'
    code_dir = Path(inspect.getfile(IsotropicPoseBuilder)).parent
    production = Path(args.formal)/'source/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1'
    for name, digest in original['source_hashes'].items():
        assert sha(production/name) == digest, name
    geometry = CompatibilityConfig(**original['geometry'])
    control_rows = {name:controls(IsotropicPoseBuilder(geometry, ProposalConfig(**original['proposal_config']), policy)) for name, policy in VARIANTS}
    protocol = dict(schema='consensus-pose-scale-sweep/1', source_phase1=str(args.phase1),
        phase1_protocol_sha256=sha(Path(args.phase1)/'protocol.json'),
        variants={name:asdict(policy) for name,policy in VARIANTS},
        scales_are='pixel radii, no x3; adaptive max(10,3*median mean A/B spacing over fixed proposal cloud)',
        contact='only weak normal eigendirection; weighted 5th-percentile contact; step<=existing9px damage band; preserve95% original directional mass',
        unchanged=['Matcher','Q/dustbin','initial seeds/hypotheses','directional localization and final evidence kernels','damage bound','GT20 label','budget8','training'],
        selection=dict(population='sim_select only, all750 positive pairs denominator',
            complete_correct='one retained <=20px cluster contains ALL original <=20px hypothesis IDs; no <=20/>40 mixed cluster',
            requirements=['complete_correct_fraction>=0.9','mixed20_40=0 (prebudget)','60px and drift80 controls pass at spacing3/4/5'],
            additional_guide_guardrails=['sim coverage, top correctness, compatible evidence mean not below phase1'],
            tie_break='smallest fixed radius passing; adaptive considered only if no fixed passes',
            no_pass='no radius selected, no deployment/training; real results descriptive only'),
        mixed_cluster_caveat='primary <=20 plus >40 reproduces phase1; stricter <=20 plus >20 also reported',
        correctness_is_not_merge_radius=True, backend='CPU FP32 from same saved phase1 raw evidence; no model rerun',
        controls=control_rows, scripts={p.name:sha(p) for p in code_dir.glob('*.py')},
        workers=args.workers, started_unix=time.time(), no_test_access=True, no_training_changes=True)
    save(out/'protocol.json', protocol)
    for name,_ in VARIANTS:
        for split in original['expected']:
            (out/name/split).mkdir(parents=True)
    counts = Counter(); results = {name:{} for name,_ in VARIANTS}; compact = {name:[] for name,_ in VARIANTS}
    start=time.time()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                initializer=initialize, initargs=(args.phase1,args.out,args.formal)) as pool:
            # Real metrics cannot influence selection: finish/write immutable
            # simulation selection receipt BEFORE submitting any real job.
            for split,n in original['expected'].items():
                grouped={name:[] for name,_ in VARIANTS}
                for item in pool.map(worker, [(split,i) for i in range(n)], chunksize=1):
                    counts[split]+=1
                    for name, r in item.items():
                        grouped[name].append(r); compact[name].append(r)
                    save(out/'status.json',dict(status='measuring',counts=dict(counts),current_split=split,
                        elapsed_seconds=time.time()-start,pid=os.getpid()))
                for name,rows in grouped.items():
                    results[name][split]=summarize(rows)
                if split=='sim_select':
                    baseline = read(Path(args.phase1)/'summary.json')['groups']['sim_select']['gt']
                    passing=[]; all_guardrails_passing=[]; gate_records={}
                    for name,_ in VARIANTS:
                        r=results[name][split]
                        control=control_rows[name]
                        checks=dict(complete=r['complete_correct_fraction']>=.9,
                            no_mixed=r['mixed20_40']==0,
                            separated_and_drift=all(x['passed'] for x in control if x['name'] in ('separated60','drift80')),
                            four_group_and_corroded_controls=all(x['passed'] for x in control),
                            coverage_nonregression=r['retained_coverage']>=baseline['retained_coverage'],
                            top_nonregression=r['top_cluster_correct']>=baseline['top_cluster_correct'],
                            evidence_nonregression=r['compatible_coverage']['edges']['mean']+1e-9>=baseline['compatible_coverage']['edges']['mean'])
                        user_pass=all(checks[k] for k in ('complete','no_mixed','separated_and_drift'))
                        if user_pass: passing.append(name)
                        if all(checks.values()): all_guardrails_passing.append(name)
                        gate_records[name]=dict(user_criteria_pass=user_pass, all_guardrails_pass=all(checks.values()),checks=checks,summary=r)
                    selected=passing[0] if passing else None
                    save(out/'selection.json',dict(selected=selected,passing=passing,all_guardrails_passing=all_guardrails_passing,candidates=gate_records,
                        selected_before_real=True,real_pairs_measured=0,selection_unix=time.time(),protocol_sha256=sha(out/'protocol.json')))
                save(out/'summary_partial.json',dict(groups=results,counts=dict(counts)))
        assert dict(counts)==original['expected']
        save(out/'pair_summary.json',compact)
        save(out/'summary.json',dict(groups=results,controls=control_rows,selection=read(out/'selection.json'),protocol=protocol))
        save(out/'complete.json',dict(status='complete',counts=dict(counts),elapsed_seconds=time.time()-start,
            summary_sha256=sha(out/'summary.json'),selection_sha256=sha(out/'selection.json')))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),counts=dict(counts)))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase1',default=str(PHASE1));p.add_argument('--formal',default=str(FORMAL))
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=16)
    run(p.parse_args())

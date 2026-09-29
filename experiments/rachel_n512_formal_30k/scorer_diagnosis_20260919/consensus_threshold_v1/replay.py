"""CPU-only frozen native-hypothesis replay. No Matcher/Scorer or TEST run.

Fixed predeclared T=16 DIAMETER, not a search using real-domain results.
All labels are used only after the builder returns a complete result.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.data import Dataset
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import PairEvidence, observed_arc_cells
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.pose_consensus import ProposalConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.threshold_builder import ThresholdPoseBuilder, ThresholdPolicy, diameter
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.threshold_evidence import recall_union

CV = Path('/root/autodl-tmp/rachel_score_design_20260913_001/real_domain_calibration_v1_20260921')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(temp, path)


def edges(ids):
    return set(map(tuple, ids.tolist()))


def quant(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return dict(n=0, mean=None, p50=None, p90=None, max=None)
    return dict(n=len(values), mean=float(np.mean(values)), p50=float(np.median(values)),
                p90=float(np.quantile(values, .9)), max=max(values))


def init(phase1, out):
    global ROOT, OUT, GEOMETRY, CONFIG, DATASET, REAL
    ROOT, OUT = Path(phase1), Path(out)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    GEOMETRY = CompatibilityConfig(**protocol['geometry'])
    CONFIG = ProposalConfig(**protocol['proposal_config'])
    DATASET = Dataset(protocol['select_spec']['path'], protocol['select_spec']['sha256'])
    REAL = {}


def masks_for(split, row):
    if split == 'sim_select':
        sample, _, _ = DATASET[row['index']]
        return [torch.from_numpy(np.array(m.squeeze(), dtype=np.float32, copy=True))
                for m in (sample.mask_a, sample.mask_b)]
    if split not in REAL:
        meta = json.loads((CV / ('real' if split == 'dunhuang_cv' else 'ood') / 'manifest.json').read_text())
        with np.load(Path(meta['prepared']) / 'inputs.npz', allow_pickle=False) as z:
            masks = z['packed_masks']
        REAL[split] = (meta, masks, {f: i for i, f in enumerate(meta['fragment_ids'])})
    meta, masks, index = REAL[split]
    item = meta['pairs'][row['index']]
    assert item['pair_id'] == row['pair_id']
    return [torch.from_numpy(np.unpackbits(masks[index[item['fragment_' + s + '_id']]], axis=-1).astype(np.float32))
            for s in ('a', 'b')]


def describe(clusters, retained, raw, row, union_admission):
    cloud, hypotheses = raw['proposals'].cloud, raw['proposals'].hypotheses
    usable = bool(row['label'] and row['gt_known'] and not row['gt_excluded'])
    truth = torch.tensor(row['target_translation_rc']) if usable else None
    error = lambda pose: float((pose - truth).norm()) if usable else None
    active = {i for i, h in enumerate(hypotheses) if len(h.edge_ids)}
    good = {i for i in active if error(hypotheses[i].translation) <= 20} if usable else set()
    bad40 = {i for i in active if error(hypotheses[i].translation) > 40} if usable else set()
    reference = set().union(*(edges(hypotheses[i].edge_ids) for i in good)) if good else set()
    edge_index = {tuple(e): i for i, e in enumerate(cloud.ids.tolist())}
    records = []
    for i, c in enumerate(clusters):
        union = edges(c.edge_ids)
        ids = list(c.merged_hypothesis_ids)
        raw_errors = [(edge, error(cloud.displacement[edge_index[edge]])) for edge in union] if usable else []
        kernel = cloud.compatibility(c.translation, GEOMETRY).kernel
        effective = union if union_admission else {e for e, k in edge_index.items() if kernel[k] >= math.exp(-4.5)}
        record = dict(index=i, retained=i < retained, members=ids, translation_rc=c.translation.tolist(),
                      gt_error_px=error(c.translation), union_edge_count=len(union),
                      member_pose_diameter_px=diameter(torch.stack([hypotheses[j].translation for j in ids])),
                      raw_q_arc_mass_px=float(sum(cloud.q[edge_index[e]] * cloud.arc_weight[edge_index[e]] for e in union)),
                      reference_union_recall=len(union & reference) / len(reference) if reference else None,
                      reference_effective_recall=len(effective & reference) / len(reference) if reference else None,
                      mixed_native20_40=bool(set(ids) & good and set(ids) & bad40),
                      mixed_native20_20=bool(set(ids) & good and set(ids) & (active - good)),
                      raw_mixed20_40=bool(any(v <= 20 for _, v in raw_errors) and any(v > 40 for _, v in raw_errors)),
                      overlap=c.overlap)
        records.append(record)
    correct = [r for r in records if usable and r['gt_error_px'] <= 20]
    correct_kept = [r for r in correct if r['retained']]
    touching = [r for r in records if r['retained'] and set(r['members']) & good]
    complete = bool(good and len(touching) == 1 and good <= set(touching[0]['members']) and touching[0]['gt_error_px'] <= 20)
    winner = correct_kept[0] if correct_kept else None
    return dict(top_correct=bool(records and usable and records[0]['gt_error_px'] <= 20),
                coverage_prebudget=bool(correct), coverage_retained=bool(correct_kept),
                budget_loss=bool(correct and not correct_kept), single_complete_correct_native=complete,
                good_native_count=len(good), reference_edges=len(reference),
                reference_union_recall=(winner['reference_union_recall'] if winner else 0.) if reference else None,
                reference_effective_recall=(winner['reference_effective_recall'] if winner else 0.) if reference else None,
                native_mixed20_40=sum(r['mixed_native20_40'] for r in records),
                native_mixed20_20=sum(r['mixed_native20_20'] for r in records),
                retained_raw_mixed20_40=sum(r['raw_mixed20_40'] for r in records[:retained]),
                prebudget=len(records), retained=retained,
                max_cluster_raw_q_arc_mass_px=max((r['raw_q_arc_mass_px'] for r in records[:retained]), default=0.),
                max_original_pose_diameter_px=max((r['member_pose_diameter_px'] for r in records), default=0.),
                clusters=records)


@torch.no_grad()
def worker(task):
    split, index = task
    row = json.loads((ROOT / split / f'{index:05d}.json').read_text())
    path = ROOT / row['evidence_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['evidence_sha256']
    raw = torch.load(path, map_location='cpu', weights_only=False)
    masks = masks_for(split, row)
    na, nb = raw['q'].shape
    a, b = torch.zeros((na, 1)), torch.zeros((nb, 1))
    pair = PairEvidence(a, b, a, b, raw['q'], raw['unmatched_a'], raw['unmatched_b'],
                        raw['geometry_a'], raw['geometry_b'], torch.arange(na), torch.arange(nb), *masks, row['numeric_valid'])
    cells = tuple(observed_arc_cells(g, CONFIG.observation_radius_px)[0] for g in (pair.ga, pair.gb))
    area_a, area_b = [float(m.sum()) for m in masks]
    def overlap(pose):
        dr, dc = map(int, pose.round().tolist())
        aa, bb = masks
        r0, c0 = max(0, -dr), max(0, -dc)
        r1, c1 = min(aa.shape[0], bb.shape[0] - dr), min(aa.shape[1], bb.shape[1] - dc)
        n = float((aa[r0:r1, c0:c1] * bb[r0+dr:r1+dr, c0+dc:c1+dc]).sum()) if r1 > r0 and c1 > c0 else 0.
        return dict(available=True, intersection_px=n, fraction_min_area=n/max(1., min(area_a, area_b)),
                    fraction_sum_area=n/max(1., area_a+area_b))
    builder = ThresholdPoseBuilder(GEOMETRY, CONFIG)
    old = raw['proposals']
    result = builder.build_from_hypotheses(old.cloud, old.hypotheses, old.seeds, overlap_fn=overlap, cells=cells)
    # Structural assertions do not use GT.
    for cluster in builder.all_clusters:
        expected = set().union(*(edges(old.hypotheses[j].edge_ids) for j in cluster.merged_hypothesis_ids))
        assert edges(cluster.edge_ids) == expected and len(cluster.edge_ids) == len(expected)
        assert cluster.actual_diameter_px <= 16.0001
        assert bool(((cluster.translation - cluster.member_translations_rc).norm(dim=1) <= 16.0001).all())
        assert bool(torch.isfinite(cluster.translation).all())
    if result.clusters:
        c = result.clusters[0]
        ev = recall_union(pair, c.translation, GEOMETRY, c.edge_ids)
        ids = c.edge_ids
        assert torch.equal(ev.weights[ids[:, 0], ids[:, 1]], pair.q[ids[:, 0], ids[:, 1]])
        assert int((ev.weights > 0).sum()) == len(ids)
    answer = dict(index=index, pair_id=row['pair_id'], split=split, label=row['label'],
                  usable_gt=bool(row['label'] and row['gt_known'] and not row['gt_excluded']),
                  gt_excluded=row['gt_excluded'], recipe=row['recipe'],
                  baseline=describe(raw['prebudget_clusters'], len(old.clusters), raw, row, False),
                  threshold16=describe(builder.all_clusters, len(result.clusters), raw, row, True),
                  audit=builder.audit)
    save(OUT / split / f'{index:05d}.json', answer)
    return {**{k: v for k, v in answer.items() if k not in ('baseline', 'threshold16')},
            **{name: {k: v for k, v in answer[name].items() if k != 'clusters'} for name in ('baseline', 'threshold16')}}


def summarize(rows):
    positive = [r for r in rows if r['usable_gt']]
    negative = [r for r in rows if not r['label']]
    out = dict(pairs=len(rows), usable_gt_positive_pairs=len(positive), negative_pairs=len(negative))
    for name in ('baseline', 'threshold16'):
        p = [r[name] for r in positive]
        out[name] = {key: sum(r[key] for r in p) for key in ('top_correct', 'coverage_prebudget',
                     'coverage_retained', 'budget_loss', 'single_complete_correct_native',
                     'native_mixed20_40', 'native_mixed20_20', 'retained_raw_mixed20_40')}
        out[name].update({key: quant(r[key] for r in p) for key in ('reference_union_recall', 'reference_effective_recall')})
        out[name]['negative_max_cluster_raw_q_arc_mass_px'] = quant(r[name]['max_cluster_raw_q_arc_mass_px'] for r in negative)
        out[name]['maximum_diameter_all_pairs'] = max((r[name]['max_original_pose_diameter_px'] for r in rows), default=0.)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase1', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = Path(args.out)
    out.mkdir(exist_ok=False, parents=True)
    save(out / 'protocol.json', dict(policy=asdict(ThresholdPolicy()), fixed_before_results=True,
         scale_selection='predeclared 16px DIAMETER research control; not claimed selected/optimal',
         source='frozen S7 M12 native hypotheses, same phase1 population',
         models_run=False, gpu_used=False, test_accessed=False, gt_used_by_builder=False,
         real_used_to_choose_threshold=False, raw_q_membership_not_directional_membership=True))
    start = time.time()
    rows, summaries, count = {}, {}, 0
    try:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                                 initializer=init, initargs=(args.phase1, args.out)) as pool:
            for split, size in [('sim_select', 1500), ('dunhuang_cv', 803), ('turufan', 602)]:
                rows[split] = []
                for answer in pool.map(worker, [(split, i) for i in range(min(size, args.limit or size))]):
                    rows[split].append(answer)
                    count += 1
                    if count % 100 == 0:
                        save(out / 'status.json', dict(status='replaying', pairs=count, seconds=time.time()-start))
                summaries[split] = summarize(rows[split])
                save(out / 'summary.json', summaries)
                save(out / 'pair_summary.json', rows)
        save(out / 'complete.json', dict(status='complete', pairs=count, seconds=time.time()-start,
             structural_verification_passed=True, limited=args.limit, learned_score_evaluated=False))
    except Exception as exc:
        save(out / 'failure.json', dict(error=repr(exc), pairs=count))
        raise
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()

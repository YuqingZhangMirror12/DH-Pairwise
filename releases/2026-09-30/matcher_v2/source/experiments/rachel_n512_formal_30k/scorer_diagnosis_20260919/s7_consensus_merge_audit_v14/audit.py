"""Explain the existing merge gates; never change proposals or trained weights.

Only TRAIN proposal caches are accepted. Case selection is recipe/label and ID
based, before inspecting merge results. Material overlap and the final refit
are not reconstructed: an earlier veto is sufficient to explain rejection,
whereas passing these audited gates does not prove a merge would be accepted.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1 import pose_consensus
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.pose_consensus import PoseConsensusBuilder, ProposalConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def edge_membership(cloud_ids, edge_ids):
    stride = int(cloud_ids[:, 1].max()) + 1
    return torch.isin(cloud_ids[:, 0] * stride + cloud_ids[:, 1],
                      edge_ids[:, 0] * stride + edge_ids[:, 1])


@torch.no_grad()
def audit_pair(builder, cloud, ca, cb):
    radius = math.sqrt(2) * builder.config.merge_sigma * builder._local_scale(cloud)
    distance = float((ca.translation - cb.translation).norm())
    result = dict(fitted_center_distance_px=distance, common_radius_px=radius)
    if distance > 2 * radius:
        return dict(result, first_veto='fitted_centers_far')
    union = torch.unique(torch.cat((ca.edge_ids, cb.edge_ids)), dim=0)
    allowed = edge_membership(cloud.ids, union)
    in_a, in_b = [edge_membership(cloud.ids, x.edge_ids) for x in (ca, cb)]
    candidate = builder._hypothesis(cloud, .5 * (ca.translation + cb.translation),
        ca.initial_seed_ids + cb.initial_seed_ids,
        torch.cat((ca.seed_translations, cb.seed_translations)),
        ca.merged_hypothesis_ids + cb.merged_hypothesis_ids, allowed=allowed)
    comp = cloud.compatibility(candidate.translation, builder.geometry)
    gate = math.exp(-.5 * builder.config.membership_sigma ** 2)
    raw_distance = float((candidate.seed_translations - candidate.translation).norm(dim=-1).max())
    fitted_distance = float((torch.stack((ca.translation, cb.translation)) - candidate.translation).norm(dim=-1).max())
    kernel_rejected = allowed & (comp.kernel < gate)
    base_mass = cloud.q * cloud.arc_weight * allowed
    explained_mass = base_mass * comp.kernel
    ids, d = cloud.ids[allowed], cloud.displacement[allowed]
    alternate_scale = max(builder._local_scale(cloud),
        float((.5 * (cloud.spacing_a + cloud.spacing_b)).median()) *
        builder.geometry.evidence_tangent_sigma_per_spacing)
    exclusive = ((ids[:, None] == ids[None]).any(-1) &
                 ((d[:, None] - d[None]).norm(dim=-1) > 2 * alternate_scale))
    aa, bb = in_a[allowed], in_b[allowed]
    preexisting = (aa[:, None] & aa[None]) | (bb[:, None] & bb[None])
    introduced = exclusive & ~preexisting
    upper = torch.triu(torch.ones_like(exclusive), diagonal=1)
    exclusive_pairs = int((exclusive & upper).sum())
    introduced_pairs = int((introduced & upper).sum())
    vetoes = dict(raw_seed_radius=raw_distance > radius,
                  union_kernel=bool(kernel_rejected.any()),
                  same_endpoint=bool(exclusive_pairs))
    first = next((k for k, value in vetoes.items() if value), 'passes_audited_gates')
    result.update(first_veto=first, vetoes=vetoes,
        raw_seed_radius_px=raw_distance, fitted_member_radius_px=fitted_distance,
        union_edges=len(union), intersection_edges=int((in_a & in_b).sum()),
        union_kernel_rejected_edges=int(kernel_rejected.sum()),
        rejected_base_mass_fraction=float(base_mass[kernel_rejected].sum() / base_mass.sum().clamp_min(1e-30)),
        rejected_explained_mass_fraction=float(explained_mass[kernel_rejected].sum() / explained_mass.sum().clamp_min(1e-30)),
        exclusive_pairs=exclusive_pairs, newly_introduced_exclusive_pairs=introduced_pairs,
        alternate_distance_limit_px=2 * alternate_scale,
        equal_input_edge_sets=bool(torch.equal(in_a, in_b)),
        candidate_translation=candidate.translation.tolist())
    conflicts = (exclusive & upper).nonzero()
    if len(conflicts):
        x, y = map(int, conflicts[0]); indices = allowed.nonzero().flatten()
        result['first_conflict'] = dict(edge_ids=[ids[x].tolist(), ids[y].tolist()],
            q=[float(cloud.q[indices[z]]) for z in (x, y)],
            kernel=[float(comp.kernel[indices[z]]) for z in (x, y)],
            displacement_distance_px=float((d[x] - d[y]).norm()),
            already_within_input_hypothesis=bool(preexisting[x, y]))
    return result


@torch.no_grad()
def audit_proposals(builder, proposals):
    hypotheses = [x for x in proposals.hypotheses if len(x.edge_ids)]
    pairs = []
    for a, ca in enumerate(hypotheses):
        for b in range(a + 1, len(hypotheses)):
            pairs.append(dict(a=a, b=b, **audit_pair(builder, proposals.cloud, ca, hypotheses[b])))
    # Idempotence diagnostic: a candidate and an identical copy are the same
    # evidence and the same pose. They must not create two independent modes.
    duplicate = [dict(hypothesis=i, **audit_pair(builder, proposals.cloud, c, c))
                 for i, c in enumerate(hypotheses)]
    return dict(original_seed_count=len(proposals.seeds),
        original_hypothesis_count=len(hypotheses), original_cluster_count=len(proposals.clusters),
        original_merge_count=len(proposals.merge_trace), pairs=pairs, identical_candidate_pairs=duplicate)


def run(root, out, per_stratum=1):
    root, out = Path(root), Path(out)
    if not 1 <= per_stratum <= 4:
        raise ValueError('bounded diagnostic sample required')
    contract = json.loads((root / 'data_contract.json').read_text())
    cache = root / 'formal_m12/scorer/proposal_cache/train'
    binding = json.loads((cache / 'binding.json').read_text())
    source = Path(inspect.getfile(pose_consensus))
    if sha(source) != binding['experiment']['implementation_sha256']['pose_consensus.py']:
        raise ValueError('audit must explain the actual bound builder')
    manifest = Path(contract['train']['path'])
    if sha(manifest) != contract['train']['sha256'] or sha(manifest) != binding['manifest_sha256']:
        raise ValueError('TRAIN manifest binding mismatch')
    calibration = root / 'geometry_calibration_v2/geometry_calibration.json'
    if sha(calibration) != binding['experiment']['geometry_calibration_sha256']:
        raise ValueError('geometry calibration binding mismatch')
    builder = PoseConsensusBuilder(CompatibilityConfig.from_calibration(json.loads(calibration.read_text())),
        ProposalConfig(**binding['experiment']['config']['proposals']))
    entries = json.loads(manifest.read_text())['entries']
    available = defaultdict(list)
    for e in sorted(entries, key=lambda x: x['pair_id']):
        key = hashlib.sha256(e['pair_id'].encode()).hexdigest()
        p = cache / key[:2] / (key + '.pt')
        if p.exists():
            recipe = e.get('recipe', e.get('s7_recipe'))
            if recipe is None:
                raise ValueError('missing declared TRAIN recipe')
            available[(str(recipe), int(e['label']))].append(dict(pair_id=e['pair_id'],
                recipe=recipe, label=int(e['label']), cache_path=str(p), cache_sha256=sha(p)))
    selected = [e for k in sorted(available) for e in available[k][:per_stratum]]
    if not selected:
        raise ValueError('no available TRAIN proposal snapshots')
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(schema='s7-merge-audit-selection/1', scope='TRAIN only; no new network inference',
        selection='lexicographic ID within recipe/label among already cached TRAIN records',
        count=len(selected), per_stratum=per_stratum, cache_binding_sha256=sha(cache / 'binding.json'),
        manifest_sha256=sha(manifest), builder_sha256=sha(source), calibration_sha256=sha(calibration),
        available_by_stratum={str(k):len(v) for k,v in available.items()}, cases=selected)
    (out / 'selection.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    start = time.time(); rows = []
    for e in selected:
        p = Path(e['cache_path'])
        if sha(p) != e['cache_sha256']:
            raise ValueError('cached input changed after selection')
        record = torch.load(p, map_location='cpu', weights_only=False)
        if record['pair_id'] != e['pair_id'] or record['binding_signature'] != signature(binding):
            raise ValueError('cached proposal identity mismatch')
        rows.append(dict(e, **audit_proposals(builder, record['proposals'])))
    first = Counter(x['first_veto'] for r in rows for x in r['pairs'])
    duplicate = Counter(x['first_veto'] for r in rows for x in r['identical_candidate_pairs'])
    report = dict(schema='s7-merge-audit/1', status='complete',
        limitation='first merge iteration, pre-overlap gates only; not final acceptance, correctness or new model performance',
        selection_sha256=sha(out / 'selection.json'), elapsed_seconds=time.time()-start,
        first_veto_counts=dict(first), identical_candidate_first_veto_counts=dict(duplicate), cases=rows)
    (out / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--per-stratum', type=int, default=1); a = p.parse_args()
    torch.set_num_threads(2)
    run(a.root, a.out, a.per_stratum)

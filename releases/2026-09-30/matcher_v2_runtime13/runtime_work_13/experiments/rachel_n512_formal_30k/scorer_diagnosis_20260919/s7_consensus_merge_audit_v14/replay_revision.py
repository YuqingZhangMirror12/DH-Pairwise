"""Replay the preregistered TRAIN snapshots with their actual raster masks.

CPU-only, no Matcher call or optimizer. Proposed repairs remain outside formal
source. GT is attached only after candidate generation, as a TRAIN diagnostic.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import material_overlap
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.pose_consensus import ProposalConfig
from .audit import sha, signature
from .builder_revision import MergeRepairPolicy, RepairedPoseConsensusBuilder


def main(root, selection, out):
    root, selection, out = Path(root), Path(selection), Path(out)
    plan = json.loads(selection.read_text())
    if plan.get('scope') != 'TRAIN only; no new network inference':
        raise ValueError('only the prespecified TRAIN diagnostic plan is accepted')
    binding_path = root/'formal_m12/scorer/proposal_cache/train/binding.json'
    if sha(binding_path) != plan['cache_binding_sha256']:
        raise ValueError('cache binding changed')
    binding = json.loads(binding_path.read_text())
    contract = json.loads((root/'data_contract.json').read_text())
    manifest = Path(contract['train']['path'])
    if sha(manifest) != plan['manifest_sha256']:
        raise ValueError('TRAIN manifest changed')
    manifest_data = json.loads(manifest.read_text())
    entries = {e['pair_id']:e for e in manifest_data['entries']}
    artifact_root = Path(manifest_data.get('artifact_root',manifest.parent))
    geometry = CompatibilityConfig.from_calibration(json.loads((root/'geometry_calibration_v2/geometry_calibration.json').read_text()))
    policy = MergeRepairPolicy()
    builder = RepairedPoseConsensusBuilder(geometry, ProposalConfig(**binding['experiment']['config']['proposals']), policy)
    out.mkdir(parents=True,exist_ok=False)
    protocol = dict(schema='s7-merge-repair-replay/1', selection_sha256=sha(selection), policy=asdict(policy),
        proposal_config=asdict(builder.config), inference='cached TRAIN cloud/seeds and unmodified TRAIN masks only',
        builder_sha256=sha(Path(__file__).with_name('builder_revision.py')), formal_model_modified=False,
        no_new_learned_inference=True, no_optimizer_updates=True,
        note='TRAIN diagnostics only, not validation selection or a classification result')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    start = time.time(); rows=[]
    for item in plan['cases']:
        path = Path(item['cache_path'])
        if sha(path) != item['cache_sha256']:
            raise ValueError('cached proposal snapshot changed')
        saved = torch.load(path,map_location='cpu',weights_only=False)
        if saved['binding_signature'] != signature(binding) or saved['pair_id'] != item['pair_id']:
            raise ValueError('proposal identity mismatch')
        old = saved['proposals']; e = entries[item['pair_id']]
        sample_path = Path(e.get('sample_path') or artifact_root/e['artifact_path'])
        sample, _ = load_sample(sample_path)
        if sample.pair_id != item['pair_id']:
            raise ValueError('mask/sample identity mismatch')
        pair = SimpleNamespace(mask_a=torch.from_numpy(np.array(sample.mask_a,copy=True)).squeeze(),
                               mask_b=torch.from_numpy(np.array(sample.mask_b,copy=True)).squeeze())
        overlap = lambda t:material_overlap(pair,t)
        new = builder.build_from_cloud(old.cloud,old.seeds,overlap_fn=overlap)
        before = torch.stack([c.translation for c in old.clusters]) if old.clusters else torch.empty((0,2))
        after = torch.stack([c.translation for c in new.clusters]) if new.clusters else torch.empty((0,2))
        # Labels and GT have not been passed into proposal generation above.
        gt = torch.as_tensor(np.array(sample.translation_a_to_b_rc,copy=True))
        correct = bool(sample.label) and bool(sample.translation_valid)
        row = dict(pair_id=item['pair_id'], recipe=item['recipe'], label=item['label'],
            archive_sha256=sha(sample_path), old_candidates=len(old.clusters), new_candidates=len(new.clusters),
            old_merges=len(old.merge_trace), new_merges=len(new.merge_trace),
            old_poses=before.tolist(), new_poses=after.tolist(), new_merge_trace=new.merge_trace,
            old_gt_errors_px=(before-gt).norm(dim=-1).tolist() if correct else None,
            new_gt_errors_px=(after-gt).norm(dim=-1).tolist() if correct else None,
            new_overlaps=[c.overlap for c in new.clusters])
        failures=[]; tested=0
        for index, seed in enumerate(old.seeds):
            one = builder.build_from_cloud(old.cloud,seed[None],overlap_fn=overlap)
            two = builder.build_from_cloud(old.cloud,seed[None].repeat(2,1),overlap_fn=overlap)
            tested += 1
            if (len(one.clusters) != len(two.clusters) or any(
                not torch.equal(a.translation,b.translation) or a.absolute_support_mass_px != b.absolute_support_mass_px
                for a,b in zip(one.clusters,two.clusters))):
                failures.append(index)
        row['duplicate_seed_trials']=tested;row['duplicate_seed_failures']=failures
        rows.append(row)
    report = dict(protocol,status='complete',elapsed_seconds=time.time()-start,cases=rows,
        old_candidate_histogram=dict(Counter(r['old_candidates'] for r in rows)),
        new_candidate_histogram=dict(Counter(r['new_candidates'] for r in rows)),
        total_merges=sum(r['new_merges'] for r in rows),
        duplicate_seed_trials=sum(r['duplicate_seed_trials'] for r in rows),
        duplicate_seed_failures=sum(len(r['duplicate_seed_failures']) for r in rows))
    (out/'replay.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','proposal_config')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--selection',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    main(a.root,a.selection,a.out)

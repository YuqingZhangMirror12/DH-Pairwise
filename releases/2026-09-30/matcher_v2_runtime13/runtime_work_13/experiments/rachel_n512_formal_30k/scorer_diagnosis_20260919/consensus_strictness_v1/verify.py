"""Independent checks of the phase1 observations and completed denominators."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from measure import sha, read, save, CV, PoseConsensusBuilder, ProposalConfig, CompatibilityConfig, edge_set


def verify(root):
    root=Path(root);protocol=read(root/'protocol.json');complete=read(root/'complete.json')
    if complete['status']!='complete': raise AssertionError('run incomplete')
    builder=PoseConsensusBuilder(CompatibilityConfig(**protocol['geometry']),ProposalConfig(**protocol['proposal_config']))
    exclusions=set(protocol['cases']['user_confirmed_gt_exclusions'])
    checks={};corrected=0;seen=set();fixed={r['pair_id'] for r in protocol['cases']['cases']}
    for split,n in protocol['expected'].items():
        rows=[read(root/split/(str(i).zfill(5)+'.json')) for i in range(n)]
        counts=dict(positive=sum(r['label'] for r in rows),negative=sum(not r['label'] for r in rows),
            gt=sum(r['gt_known'] for r in rows),excluded=sum(r['gt_excluded'] for r in rows))
        if split=='sim_select': assert counts==dict(positive=750,negative=750,gt=750,excluded=0)
        if split=='dunhuang_cv': assert counts==dict(positive=295,negative=508,gt=295,excluded=3)
        if split=='turufan': assert counts==dict(positive=301,negative=301,gt=0,excluded=0)
        checks[split]=counts
        for i,r in enumerate(rows):
            assert r['index']==i and r['split']==split
            assert r['pair_id'] not in seen;seen.add(r['pair_id'])
            assert r['gt_excluded']==(r['pair_id'] in exclusions)
            assert r['retained_cluster_count']==min(r['prebudget_cluster_count'],8)
            for c in r['clusters']:
                assert c['unique_a']==len({e[0] for e in c['edge_ids']})
                assert c['unique_b']==len({e[1] for e in c['edge_ids']})
                # Every sparse compatible cloud edge is also present in full-Q
                # significant membership; extra full-Q edges were never TopK cut.
                assert set(map(tuple,c['edge_ids'])) <= set(map(tuple,c['full_q_significant_ids']))
            if r['gt_known']:
                target=np.array(r['target_translation_rc'])
                for h in r['hypotheses']:
                    assert abs(h['gt_error_px']-np.linalg.norm(np.array(h['translation_rc'])-target))<1e-7
                for c in r['clusters']:
                    assert abs(c['gt_error_px']-np.linalg.norm(np.array(c['translation_rc'])-target))<1e-7
            # Fixed requested examples plus one ordinary record per population.
            if r['pair_id'] in fixed or i==0:
                path=root/r['evidence_file'];assert sha(path)==r['evidence_sha256']
                raw=torch.load(path,map_location='cpu',weights_only=False)
                p=raw['proposals'];assert len(p.hypotheses)==r['hypothesis_count']
                expected=np.sqrt(2)*builder.config.merge_sigma*builder._local_scale(p.cloud)
                assert abs(expected-r['common_center_radius_px'])<1e-8
                for h,rec in zip(p.hypotheses,r['hypotheses']):
                    assert edge_set(h.edge_ids)==set(map(tuple,rec['edge_ids']))
                for c,rec in zip(p.clusters,r['clusters']):
                    assert edge_set(c.edge_ids)==set(map(tuple,rec['edge_ids']))
        if split=='dunhuang_cv':corrected=sum(r['label'] and not r['gt_excluded'] for r in rows)
    assert corrected==292 and fixed<=seen and len(seen)==2905
    sources={}
    for split,sub in [('dunhuang_cv','real'),('turufan','ood')]:
        manifest=CV/sub/'manifest.json';meta=read(manifest);inputs=Path(meta['prepared'])/'inputs.npz'
        sources[split]=dict(manifest_sha256=sha(manifest),inputs_sha256=sha(inputs))
        assert sources[split]['manifest_sha256']==protocol['real_manifests'][split]['sha256']
    record=dict(status='passed',independent_checks=checks,corrected_dunhuang_gt=corrected,
        total_pairs=len(seen),fixed_pairs=len(fixed),source_hashes=sources,
        all_rows_checked_for_counts_unique_endpoints_gt_errors_and_sparse_subset_of_fullq=True,
        raw_hash_and_membership_checked='all 11 fixed pairs plus first row of each population',
        test_or_real_used_to_select_parameters=False,script_sha256=sha(__file__))
    save(root/'verification.json',record);print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();torch.set_num_threads(1);verify(a.root)

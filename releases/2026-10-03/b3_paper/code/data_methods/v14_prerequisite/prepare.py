"""Construct exact TRAIN source quotas without reading model scores or TEST."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial import cKDTree
from ..s7_compound_v1.materialize import read, save_json, digest
from staging.pairwise_v0_2.pairwise_data.rachel_composite_training import _fragment


def gen_family(generator):
    m = re.match(r'gen(\d+)', generator)
    if not m:
        raise ValueError('unknown generator: '+generator)
    return int(m.group(1))


def negative_kind(a, b):
    if a['parent_group_id'] == b['parent_group_id']:
        return 'same_parent_nonadjacent'
    if a['split_unit_id'] == b['split_unit_id']:
        return None
    if gen_family(a['generator']) != gen_family(b['generator']):
        return 'cross_gen'
    if a['generator'] == b['generator']:
        return 'cross_parent_same_gen'
    return None


def rank(seed, *values):
    return hashlib.sha256(json.dumps([seed, values], sort_keys=True).encode()).hexdigest()


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def run(args):
    out=Path(args.out)
    if out.exists():
        raise ValueError('new source plan required')
    old=read(args.s7_manifest); options=old['protocol']['options']
    root=Path(options['dataset_root'])
    splits=read(root/'pairs/lineage_splits.json')
    fragments={r['fragment_token']:r for r in records(root/'manifests/fragments.jsonl')
               if splits.get(r['split_unit_id'])=='train'}
    positives=[e for e in old['entries'] if e['label']]
    assert len(positives)==12000
    gen5=read(options['gen5_manifest'])
    retained=[e for e in gen5['entries'] if not e['label']]
    assert len(retained)==1200
    old_stats=Counter(e['source_stratum'] for e in old['entries'] if not e['label'])
    # Existing Gen5 partition/omission geometry is kept, not replaced with rectangles.
    negatives=[]
    for e in retained:
        kind=('same_parent_nonadjacent' if e['source_stratum']=='gen5_partition_within_negative'
              else 'cross_parent_same_gen')
        if kind=='same_parent_nonadjacent' and e['partition_provenance']['crossing_neighbor_edges']:
            raise ValueError('Gen5 negative has a crossing adjacency')
        negatives.append(dict(mode='gen5',pair_id=e['pair_id'],kind=kind,
                              anchor_group_id=None,source_stratum=e['source_stratum']))
    quota={'same_parent_nonadjacent':3600,'cross_parent_same_gen':4200,'cross_gen':4200}
    needed=Counter(quota);needed.subtract(Counter(r['kind'] for r in negatives))
    hard=[]
    for r in records(root/'manifests/within_candidates.jsonl'):
        if (r['label'] or not r['main_training_eligible'] or r['status']!='accepted'
                or r['fragment_a_token'] not in fragments or r['fragment_b_token'] not in fragments):
            continue
        a,b=fragments[r['fragment_a_token']],fragments[r['fragment_b_token']]
        if (negative_kind(a,b)!='same_parent_nonadjacent' or r['seam_length_px']!=0
                or r['metadata']['seam_match_count']!=0
                or r['label_origin']!='rachel_csv_same_folder_nonneighbor_no_seam'):
            raise ValueError('nonadjacent source lacks explicit label evidence')
        hard.append(r)
    hard.sort(key=lambda r:rank(args.seed,r['pair_id']))
    if len(hard)<needed['same_parent_nonadjacent']:
        raise ValueError('insufficient distinct confirmed same-parent negatives')
    values=sorted(fragments.values(),key=lambda r:rank(args.seed,r['fragment_token']))
    features=np.log([[r['foreground_area'],r['bbox_aspect_ratio']] for r in values])
    tree=cKDTree(features)
    # Search inside the allowed source stratum first. A global nearest-2048
    # search can exclude an entire Gen folder at extreme fragment sizes.
    source_indices={}
    for generator in {r['generator'] for r in values}:
        source_indices[('cross_parent_same_gen',generator)]=np.array(
            [i for i,r in enumerate(values) if r['generator']==generator])
    for family in {gen_family(r['generator']) for r in values}:
        source_indices[('cross_gen',family)]=np.array(
            [i for i,r in enumerate(values) if gen_family(r['generator'])!=family])
    source_trees={key:cKDTree(features[ids]) for key,ids in source_indices.items()}
    pos_area=np.array([[e['source_row']['fragment_'+side]['foreground_area'] for side in 'ab']
                       for e in positives],float)
    # Symmetrize which endpoint acts as anchor, using the unchanged positive mix.
    area_pairs=np.concatenate([pos_area,pos_area[:,::-1]])
    pos_tree=cKDTree(np.log(area_pairs[:,0:1]))
    rng=np.random.default_rng(args.seed)
    seen=set();usage=Counter();partner_usage=Counter();packed=0

    def append_native(a,b,kind,group=None,original=None):
        key=tuple(sorted((a['fragment_token'],b['fragment_token'])))
        if key in seen or negative_kind(a,b)!=kind:
            raise ValueError('duplicate or wrong negative stratum')
        seen.add(key);usage.update(key)
        pid=(original['pair_id'] if original else 's7b2-neg-'+rank(args.seed,key)[:24])
        row=dict(pair_id=pid,split='train',label=False,fragment_a=_fragment(a),fragment_b=_fragment(b),
                 correspondence_path=None,translation_a_to_b_rc=None,translation_a_to_b_xy_cartesian=None,
                 label_origin=(original['label_origin'] if original else 'different_frozen_TRAIN_lineages'),
                 negative_origin=kind)
        negatives.append(dict(mode='native',pair_id=pid,row=row,kind=kind,anchor_group_id=group,
                              anchor_fragment_token=a['fragment_token'],
                              source_stratum='balanced_'+kind,
                              parent_a=a['parent_group_id'],parent_b=b['parent_group_id'],
                              generator_a=a['generator'],generator_b=b['generator']))
        needed[kind]-=1

    def partner(anchor,kind):
        _,near=pos_tree.query([[np.log(anchor['foreground_area'])]],k=32)
        desired=float(area_pairs[int(rng.choice(near[0])),1])
        query=np.log([desired,anchor['bbox_aspect_ratio']])
        pool_key=(kind,anchor['generator'] if kind=='cross_parent_same_gen'
                  else gen_family(anchor['generator']))
        ids=source_indices[pool_key];candidate_tree=source_trees[pool_key]
        for count in dict.fromkeys((min(2048,len(ids)),len(ids))):
            _,candidates=candidate_tree.query(query,k=count)
            for j in np.atleast_1d(candidates):
                b=values[int(ids[int(j)])]
                key=tuple(sorted((anchor['fragment_token'],b['fragment_token'])))
                if (negative_kind(anchor,b)==kind and key not in seen
                        and partner_usage[b['fragment_token']]<8):
                    partner_usage[b['fragment_token']]+=1
                    return b
        raise ValueError('no size/aspect-near TRAIN partner for '+kind)

    # 50% of all negative pairs are explicitly in 2000 three-partner anchor sets.
    chosen=[];anchor_seen=set()
    for r in hard:
        a,b=fragments[r['fragment_a_token']],fragments[r['fragment_b_token']]
        if a['fragment_token'] in anchor_seen:
            a,b=b,a
        if a['fragment_token'] in anchor_seen:
            continue
        anchor_seen.add(a['fragment_token']);chosen.append((r,a,b))
        if len(chosen)==2000:
            break
    if len(chosen)!=2000:
        raise ValueError('insufficient unique multi-negative anchors')
    selected_hard=set()
    for r,a,b in chosen:
        group='s7b2-anchor-'+rank(args.seed,a['fragment_token'])[:24]
        append_native(a,b,'same_parent_nonadjacent',group,r);selected_hard.add(r['pair_id'])
        for kind in ('cross_parent_same_gen','cross_gen'):
            append_native(a,partner(a,kind),kind,group)
        packed+=1
    for r in hard:
        if needed['same_parent_nonadjacent']==0:
            break
        if r['pair_id'] not in selected_hard:
            append_native(fragments[r['fragment_a_token']],fragments[r['fragment_b_token']],
                          'same_parent_nonadjacent',original=r)
    for kind in ('cross_parent_same_gen','cross_gen'):
        for _ in range(needed[kind]):
            # Source anchors follow positive areas instead of uniform Gen2/Gen4 counts.
            target=area_pairs[int(rng.integers(len(area_pairs))),0]
            _,idx=tree.query([np.log(target),float(rng.choice(features[:,1]))],k=32)
            anchor=values[int(rng.choice(idx))]
            append_native(anchor,partner(anchor,kind),kind)
    if any(needed.values()) or len(negatives)!=12000:
        raise ValueError('negative source quota incomplete')
    counts=Counter(r['kind'] for r in negatives)
    groups=Counter(r['anchor_group_id'] for r in negatives if r['anchor_group_id'])
    assert counts==quota and len(groups)==2000 and set(groups.values())=={3}
    rng.shuffle(negatives)
    rng.shuffle(positives)
    plan=dict(schema='s7-balanced-source-plan/2',seed=args.seed,s7_manifest=args.s7_manifest,
              dataset_root=str(root),positive=[dict(pair_id=e['pair_id'],source_stratum=e['source_stratum'])
                                               for e in positives],negative=negatives,
              stats=dict(positive_count=12000,negative_count=12000,
                  positive_strata=dict(Counter(e['source_stratum'] for e in positives)),
                  negative_kind=dict(counts),multi_negative_groups=len(groups),
                  grouped_negative_pairs=sum(groups.values()),retained_gen5_negative_pairs=len(retained),
                  native_confirmed_nonadjacent_available=len(hard),
                  old_s7_negative_strata=dict(old_stats),
                  max_native_fragment_uses=max(usage.values())),
              provenance=dict(s7_manifest_sha256=digest(args.s7_manifest),
                  lineage_splits_sha256=digest(root/'pairs/lineage_splits.json'),
                  within_negative_label='CSV nonneighbor AND independently zero seam; unknown ignored',
                  cross_label='different frozen TRAIN original image lineage',
                  cross_gen_definition='Gen2/3/4/5 family differs; same-gen uses identical generator folder',
                  model_scores_used=False,held_out_masks_or_targets_read=False,
                  original_manifests_modified=False))
    save_json(out,plan)
    print(json.dumps(plan['stats'],ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--s7-manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=26092331)
    run(p.parse_args())

"""Two-record CPU diagnosis; does not repair production or change thresholds.

Unknown source-owner pixels remain unassigned in a diagnostic ancestry view.
Enumerating their possible existing owners proves whether the actual A/B seam
depends on them. Physical masks and labels never change; no GT-derived match.
"""
import argparse
from dataclasses import replace
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from staging.pairwise_v0_2.pairwise_data.rachel_source_density_records import SourceDensityRecordResolver,_mask
from staging.pairwise_v0_2.pairwise_data.rachel_source_density import (
    source_boundary,shared_source_seams,resample_source_density,inherit_dense_source_targets)
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import rgb_to_binary_mask,RachelPreprocessConfig


def interface(masks,ids):
    a,b=(source_boundary(masks[i]) for i in ids)
    seam,components,detail=shared_source_seams(a,b)
    return seam,components,detail


def run(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    resolver=SourceDensityRecordResolver(args.manifest,args.canonical_root)
    run_config=json.loads((resolver.canonical_root/'run_config.json').read_text())
    rawroot=Path(run_config['source_root'])
    summary=dict(schema_version='source-density-ownership-two-record-diagnosis/1',indices=[505,807],
        production_source_changed=False,physical_masks_changed=False,thresholds_changed=False,
        complete_population_repair=False,gpu_used=False,gt_translation_used_for_matching=False,records=[])
    for index in (505,807):
        entry=resolver.entries[index];row=entry['source_row'];tokens=[row['fragment_'+s]['fragment_token'] for s in 'ab']
        parts=tokens[0].split('/');group,path=resolver._group(parts[1],parts[2]);parents=resolver._parents(group)
        ids=sorted(parents);ab=[token.split('/')[-1] for token in tokens]
        stack=np.stack([parents[k] for k in ids]);depth=np.stack([ndimage.distance_transform_edt(m) for m in stack])
        overlap=stack.sum(0)>1;second=np.sort(depth,axis=0)[-2];unknown=overlap&(second>3.)
        locations=np.argwhere(unknown);fraction=float(overlap.sum()/stack.any(0).sum())
        if fraction>.02 or len(locations)>6:raise ValueError('diagnosis bounded to sparse second-depth failures')
        pair_stats=[]
        for a,b in itertools.combinations(range(len(ids)),2):
            ov=stack[a]&stack[b];deep=ov&(np.minimum(depth[a],depth[b])>3.)
            pair_stats.append(dict(members=[ids[a],ids[b]],overlap_px=int(ov.sum()),
                second_depth_max_px=float(np.minimum(depth[a],depth[b])[ov].max()) if ov.any() else 0.,
                deep_pixels=int(deep.sum()),deep_coordinates_rc=np.argwhere(deep).tolist()))
        # Same existing winner rule and SAME3px criterion on trusted pixels.
        # Unknown pixels have no owner, not an invented larger accepted depth.
        known=stack.copy();safe=overlap&~unknown
        known[:,safe]=np.arange(len(ids))[:,None]==depth[:,safe].argmax(0)[None,:]
        known[:,unknown]=False
        assert np.array_equal(known.any(0),stack.any(0)&~unknown)
        assert np.array_equal(known[:,~overlap],stack[:,~overlap])
        known_masks={k:known[i] for i,k in enumerate(ids)}
        seam,components,topology=interface(known_masks,ab)
        if not seam:raise ValueError('no reliable A/B seam away from unknown owners')
        seam_mid=np.asarray([(np.asarray(k[0])+np.asarray(k[1]))*.5 for k in seam])
        distance_to_seam=cKDTree(seam_mid).query(locations+.5)[0]
        choices=[np.flatnonzero(stack[:,r,c]).tolist() for r,c in locations]
        variants=[]
        for owners in itertools.product(*choices):
            variant=known.copy()
            for (r,c),owner in zip(locations,owners):variant[owner,r,c]=True
            possible,_,_=interface({k:variant[i] for i,k in enumerate(ids)},ab)
            variants.append(dict(owner_ids=[ids[i] for i in owners],same_AB_seam=set(possible)==set(seam),
                added_AB_edges=len(set(possible)-set(seam)),lost_AB_edges=len(set(seam)-set(possible))))
        if not all(v['same_AB_seam'] for v in variants):
            raise ValueError('A/B interface depends on unresolved source owners')
        raw=rawroot/parts[1]/'no_erode'/parts[2]
        rgb_checks={}
        for f in group['fragments']:
            with Image.open(raw/(f['fragment_id']+'.jpg')) as image:rgb=np.asarray(image.convert('RGB'))
            recreated=rgb_to_binary_mask(rgb,RachelPreprocessConfig(**run_config['preprocess_config']))
            rgb_checks[f['fragment_id']]=dict(exact_original_parent_reproduction=bool(np.array_equal(recreated,parents[f['fragment_id']])))
        sample,report=load_sample(resolver.artifact_root/entry['artifact_path'])
        fragments={s:next(f for f in group['fragments'] if f['fragment_token']==row['fragment_'+s]['fragment_token']) for s in 'ab'}
        offsets={s:fragments[s]['target_audit']['parent_to_model_offset_rc'] for s in 'ab'}
        clean={s:_mask(Path(entry['source_root'])/row['fragment_'+s]['model_mask_path']) for s in 'ab'}
        for s in 'ab':resolver._verify_centerpad(parents[ab['ab'.index(s)]],clean[s],offsets[s])
        source={s:known_masks[fragments[s]['fragment_id']] for s in 'ab'}
        cap_results={}
        for cap in (512,1024):
            derived,detail,views=resample_source_density(sample,report,source,offsets,clean,cap=cap,ownership_allowance_px=3.)
            suppressed={}
            # Any projection whose point can reach unknown ownership is ignore,
            # including nonseam/dustbin targets. No uncertain cell gets a label.
            for s in 'ab':
                parent_points=getattr(derived,'points_rc_'+s)+.5-np.asarray(offsets[s])
                distance=cKDTree(locations+.5).query(parent_points)[0]
                margin=views[s]['projection_radius_px']+2.
                unsafe=distance<=margin
                views[s]['component'][unsafe]=-1;views[s]['seam_s'][unsafe]=np.nan
                views[s]['trusted'][unsafe]=False;views[s]['nonseam'][unsafe]=False
                suppressed[s]=dict(tokens=int(unsafe.sum()),positive_tokens_before=int(np.count_nonzero((getattr(derived,'target_'+s)>=0)&unsafe)),
                    exclusion_radius_px=margin,min_point_distance_to_unknown_px=float(distance.min()))
            ta,tb,chosen=inherit_dense_source_targets(views['a'],views['b'],components)
            if not chosen:raise ValueError('positive reduced to zero targets')
            # This discarded diagnostic result never enters any dataset loader.
            derived=replace(derived,target_a=ta,target_b=tb)
            fields=('mask_a','mask_b','coarse_mask_a','coarse_mask_b','label','translation_a_to_b_rc',
                'translation_a_to_b_xy_cartesian','translation_valid')
            assert all(np.array_equal(getattr(sample,k),getattr(derived,k),equal_nan=True) for k in fields)
            cap_results[str(cap)]=dict(matches_before_uncertainty_exclusion=detail['density']['new_match_count'],
                matches_after_uncertainty_exclusion=len(chosen),points=[len(ta),len(tb)],
                uncertainty_exclusion=suppressed,physical_masks_and_gt_unchanged=True,
                exact_e1_replay=all(v['exact_replay'] for v in detail['density']['replay'].values()))
        record=dict(index=index,pair_id=entry['pair_id'],actual_pair_members=ab,group_path=str(path),
            source_image_lineage=group['image_name'],source_raw_directory=str(raw),raw_files=sorted(p.name for p in raw.iterdir()),
            raw_to_saved_parent=rgb_checks,original_sparse_match_count=int((sample.target_a>=0).sum()),
            group_overlap_px=int(overlap.sum()),group_overlap_fraction=fraction,group_second_depth_max_px=float(second[overlap].max()),
            uncertain_source_pixels=locations.tolist(),pairwise_overlap_stats=pair_stats,
            AB_source_interface=topology,AB_components=components,unknown_to_AB_seam_distance_px=distance_to_seam.tolist(),
            all_possible_existing_owners=variants,all_owner_assignments_preserve_AB_seam=True,
            unknown_source_pixels_not_assigned=True,caps=cap_results)
        summary['records'].append(record)
        np.savez_compressed(output/('index%d_source_geometry.npz'%index),original_parent_masks=stack,
            diagnostic_known_owner_masks=known,unknown_owner_mask=unknown,member_ids=np.asarray(ids),
            shared_AB_edge_midpoints=seam_mid)
    (output/'diagnosis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True)
    p.add_argument('--canonical-root',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())

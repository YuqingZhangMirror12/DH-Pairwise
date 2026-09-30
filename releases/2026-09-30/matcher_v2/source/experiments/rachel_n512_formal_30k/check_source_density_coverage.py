"""Small read-only CPU coverage check on actual manifest source layouts.

Select at most12 pairs: native / union positives, hard / cross negatives,
each actual clean /2px /4px. No derived dataset, GPU, or training is created.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_source_density_records import SourceDensityRecordResolver
from staging.pairwise_v0_2.pairwise_data.rachel_source_density import resample_source_density


def run(args):
    resolver=SourceDensityRecordResolver(args.manifest,args.canonical_root)
    selected,examined=resolver.stratified_indices(args.per_family_scan)
    summary=dict(schema_version='rachel-source-density-stratified-pilot/1',
        manifest=str(resolver.manifest_path),prototype_only=True,formal_training_eligible=False,
        full24k_materialized=False,gpu_used=False,physical_masks_modified=False,
        source_records_automatically_resolved=True,metadata_rows_examined_by_family=examined,
        requested_strata=12,selected_strata=len(selected),pairs=[])
    for selection in selected:
        row=dict(selection)
        try:
            sample,report,source,offsets,clean,allowance,provenance=resolver.resolve(selection['index'])
            row.update(pair_id=sample.pair_id,source=provenance,caps={})
            for cap in (512,1024):
                changed,derived,_=resample_source_density(sample,report,source,offsets,clean,
                    cap=cap,ownership_allowance_px=allowance)
                for side in 'ab':
                    if not np.array_equal(getattr(sample,'mask_'+side),getattr(changed,'mask_'+side)):
                        raise AssertionError('fixed physical mask changed')
                chosen=np.flatnonzero(changed.target_a>=0)
                if not np.array_equal(changed.target_b[changed.target_a[chosen]],chosen):
                    raise AssertionError('nonreciprocal targets')
                canary,_,_=resample_source_density(replace(sample,
                    translation_a_to_b_rc=sample.translation_a_to_b_rc+1000.),report,source,offsets,clean,
                    cap=cap,ownership_allowance_px=allowance)
                if not np.array_equal(changed.target_a,canary.target_a) or not np.array_equal(changed.target_b,canary.target_b):
                    raise AssertionError('GT translation used for target matching')
                d=derived['density']
                row['caps'][str(cap)]=dict(real_points={s:len(getattr(changed,'points_rc_'+s)) for s in 'ab'},
                    new_match_count=d['new_match_count'],old_match_count=d['old_match_count'],
                    ignored={s:d['sides'][s]['ignored'] for s in 'ab'},source_topology=d['topology'],
                    original_runtime_exact_replay=d['replay'],gt_translation_canary_passed=True,
                    pose_supervision_enabled=derived['pose_supervision_enabled'])
            row['status']='passed'
        except Exception as error:
            row.update(status='failed_explicitly',error_type=type(error).__name__,error=str(error))
        summary['pairs'].append(row)
    summary['passed_pairs']=sum(r['status']=='passed' for r in summary['pairs'])
    summary['failed_pairs']=sum(r['status']!='passed' for r in summary['pairs'])
    text=json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)
    if args.json_output:
        output=Path(args.json_output)
        if output.exists():raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text+'\n')
    print(text,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--canonical-root',required=True)
    parser.add_argument('--per-family-scan',type=int,default=128)
    parser.add_argument('--json-output')
    run(parser.parse_args())

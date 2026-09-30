"""Version4 clean SIM VAL/TEST cache with bounded unknown-owner isolation.

Original labels and membership remain fixed. An explicit --pair-ids subset is
a preparation smoke only; it is never reported as a complete3000-pair split.
Failures are recorded and reject successful completion, not silently filtered.
"""
import argparse
import json
from pathlib import Path
import sys

from staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v4 import CleanSourceDensityV4Dataset as CleanSourceDensityDataset
from experiments.rachel_n512_formal_30k.materialize_source_density import save_json


def run(args):
    root=Path(args.cache_root).resolve();root.mkdir(parents=True,exist_ok=True)
    datasets={cap:CleanSourceDensityDataset(args.dataset,args.split,cap,root/(args.split+'_n'+str(cap)))
              for cap in (512,1024)}
    rows=datasets[512].rows;lookup={r['pair_id']:i for i,r in enumerate(rows)}
    ids=json.loads(Path(args.pair_ids).read_text()) if args.pair_ids else list(lookup)
    if not ids or len(set(ids))!=len(ids) or any(pid not in lookup for pid in ids):
        raise ValueError('invalid explicit original split selection')
    summary=dict(schema_version='clean-source-density-preparation/1',source_density_version='v4',
        ownership_protocol='density-local-unknown-ownership/4',split=args.split,
        full_split=args.pair_ids is None,original_split_count=len(rows),selected_count=len(ids),
        pair_ids=ids,cache_root=str(root),model_inference=False,gpu_used=False,
        model_or_threshold_selected=False,records=[],failures=[])
    for pid in ids:
        index=lookup[pid];row=dict(pair_id=pid,label=bool(rows[index]['label']),caps={})
        try:
            for cap,dataset in datasets.items():
                sample,report=dataset.weathered(index)
                row['caps'][str(cap)]=dict(real_points=[len(sample.points_rc_a),len(sample.points_rc_b)],
                    positive_matches=report['density']['new_match_count'],
                    source_split=report['source_split'],pose_supervision_enabled=report['pose_supervision_enabled'])
            summary['records'].append(row)
        except Exception as error:
            summary['failures'].append(dict(pair_id=pid,error_type=type(error).__name__,error=str(error)))
    summary['status']='complete' if not summary['failures'] and len(summary['records'])==len(ids) else 'incomplete'
    summary['cached_pair_count']=len(summary['records'])
    save_json(root/('clean_'+args.split+'_preparation.json'),summary)
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['status']=='complete' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--split',choices=('val','test'),required=True)
    p.add_argument('--cache-root',required=True);p.add_argument('--pair-ids')
    sys.exit(run(p.parse_args()))

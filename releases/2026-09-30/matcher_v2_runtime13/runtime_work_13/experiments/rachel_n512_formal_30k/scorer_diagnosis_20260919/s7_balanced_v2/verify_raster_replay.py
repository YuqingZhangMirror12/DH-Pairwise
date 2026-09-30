"""Verify a lossless metadata repair without changing any generated samples."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from . import materialize
from .audit_layered import one
from ..s7_compound_v1.materialize import read, save_json


def compare_archive(old, new, allow_exact=False):
    with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
        extra=set(b.files)-set(a.files)
        if set(a.files)-set(b.files) or (extra and (not allow_exact or any(
                name not in {'a_total_exact','b_total_exact'} for name in extra))):
            raise ValueError('archive fields changed: '+str(old))
        for name in a.files:
            if a[name].dtype!=b[name].dtype or not np.array_equal(a[name],b[name]):
                raise ValueError('original archive value changed: '+str(old)+' '+name)
        return len(a.files)


def compare_group(task):
    old_root, new_root, index = task
    old_root,new_root=Path(old_root),Path(new_root)
    relative='groups/%05d.json'%index
    old=read(old_root/relative);new=read(new_root/relative)
    expected=json.loads(json.dumps(old).replace(str(old_root),str(new_root)))
    if new!=expected:
        raise ValueError('group source, schedule, labels or report changed: '+str(index))
    fields=0
    for a,b in zip(old['entries'],new['entries']):
        fields+=compare_archive(old_root/a['artifact_path'],new_root/b['artifact_path'])
        fields+=compare_archive(a['target_metadata'],b['target_metadata'])
        for key in ('weather_artifact','background_artifact','latent_seam_artifact'):
            if a.get(key):
                fields+=compare_archive(old_root/a[key],new_root/b[key],allow_exact=True)
    return dict(slot=index,pairs=len(new['entries']),unchanged_array_fields=fields)


def run(old_root,new_root,index=None,workers=8):
    old_root,new_root=Path(old_root).resolve(),Path(new_root).resolve()
    start=time.time()
    if index is not None:
        if new_root.exists():raise ValueError('replay output already exists')
        new_root.mkdir(parents=True)
        options=dict(read(old_root/'protocol.json')['options'],out=str(new_root),
            profile=str(Path(__file__).with_name('distribution_profile_v14.json').resolve()))
        materialize.initialize(options)
        group=materialize.slot(index)
        rows=[compare_group((str(old_root),str(new_root),index))]
        audits=[one((str(new_root),e)) for e in group['entries']]
        if not all(a['raster_depth_lossless'] for a in audits):
            raise ValueError('lossless audit field missing')
    else:
        indexes=sorted(int(p.stem) for p in (old_root/'groups').glob('*.json'))
        if len(indexes)!=12000 or indexes!=list(range(12000)):
            raise ValueError('full original24K required for parity verification')
        with ProcessPoolExecutor(workers) as pool:
            rows=list(pool.map(compare_group,[(str(old_root),str(new_root),i) for i in indexes],chunksize=8))
        audits=None
    result=dict(status='passed',old_root=str(old_root),new_root=str(new_root),
        replay_slot=index,pairs=sum(r['pairs'] for r in rows),
        unchanged_array_fields=sum(r['unchanged_array_fields'] for r in rows),
        original_masks_points_targets_labels_gt_reports_unchanged=True,
        only_new_fields='float64 a_total_exact/b_total_exact in auxiliary damage receipts',
        elapsed_seconds=time.time()-start,pixel_audits=audits)
    save_json(new_root/'raster_replay_parity.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-root',required=True);p.add_argument('--new-root',required=True)
    p.add_argument('--slot',type=int);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();run(a.old_root,a.new_root,a.slot,a.workers)

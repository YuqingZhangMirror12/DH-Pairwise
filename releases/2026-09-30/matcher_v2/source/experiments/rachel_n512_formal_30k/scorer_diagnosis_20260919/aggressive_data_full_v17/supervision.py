"""Replay contour inheritance and exercise the actual downstream label adapters."""
from dataclasses import replace
import hashlib
import numpy as np
from PIL import Image
from ..aggressive_data_v15.geometry import crop
from ..s7_consensus_v1.check_data_compatibility import check_batch
from ..s7_consensus_v1.data import collate
from ..seam_context_v3.targets import known_gap_links
from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import _changed_view
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import inherit_pair_targets
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import _readonly


def changed(sample, masks, details, light=False):
    views={}; updates={}
    for side in 'ab':
        info=details[side]
        view=_changed_view(sample,side,masks[side],info,
            'wave' if light or not info.get('ignore_source_regions') else 'local')
        views[side]=view
        coarse=np.asarray(Image.fromarray(np.uint8(view.mask)*255).resize((128,128),Image.Resampling.NEAREST))>0
        updates.update({'mask_'+side:_readonly(view.mask[None],np.float32),
            'coarse_mask_'+side:_readonly(coarse[None],np.float32),
            'points_rc_'+side:view.points,'contour_valid_'+side:view.valid})
    ta,tb=inherit_pair_targets(sample,views['a'],views['b'])
    return replace(sample,**updates,target_a=_readonly(ta,np.int64),target_b=_readonly(tb,np.int64))


def audit_supervision(record,base,final,report,z):
    def masks(stage):return {s:np.unpackbits(z['packed_'+stage+'_'+s],axis=1).astype(bool) for s in 'ab'}
    current=base; plan=record['detail']['trim']; trim=masks('trim')
    current=crop(current,plan['side'],trim[plan['side']])
    recipe=record['recipe']; primary=masks('primary')
    if recipe=='partial':
        for side in 'ab': current=crop(current,side,primary[side])
    elif recipe!='clean':
        current=changed(current,primary,record['detail']['primary_damage'])
    if recipe!='clean':current=changed(current,masks('final'),record['detail']['background'],light=True)
    for side in 'ab':
        for stem in ('mask_','coarse_mask_','points_rc_','contour_valid_','target_'):
            name=stem+side
            if not np.array_equal(getattr(current,name),getattr(final,name)):
                raise ValueError('actual inherited supervision differs:'+name)
    target=record['target_metadata']
    if hashlib.sha256(open(target,'rb').read()).hexdigest()!=record['target_metadata_sha256']:
        raise ValueError('target metadata hash mismatch')
    expected=known_gap_links(final,base,report)
    with np.load(target,allow_pickle=False) as saved:
        if set(saved.files)!=set(expected) or any(not np.array_equal(saved[k],expected[k]) for k in expected):
            raise ValueError('target metadata not reproduced')
    items=[(final,report,record)]
    receipt=check_batch(items,collate(items))[0]
    return dict(actual_training_loader_checked=True,stagewise_ancestry_replayed=True,
        wrong_pose_is_negative=True,**receipt)


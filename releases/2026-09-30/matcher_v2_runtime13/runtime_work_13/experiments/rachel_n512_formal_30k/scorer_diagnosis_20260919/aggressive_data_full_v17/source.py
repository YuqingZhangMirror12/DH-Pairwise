"""Reconstruct audited v14 fragment stages; never generate new base tears."""
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import os
import numpy as np
from ..s7_balanced_v2 import materialize as old
from ..s7_balanced_v2.scale import pair_shared_scale
from ..s7_balanced_v2.layered_geometry import canonical_mirrored_contours
from ..seam_context_v3.augmentation import paired_mirror
from ..s7_compound_v1.materialize import read
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample

STATE = {}


def initialize(config, split):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    import cv2, torch
    cv2.setNumThreads(1); torch.set_num_threads(1)
    old.STATE.clear(); old.clean_positive.cache_clear(); baseline.cache_clear()
    root = Path(config['baselines'][split])
    if split == 'train':
        old.initialize(read(root/'protocol.json')['options'])
        negative = {e['pair_id']: e for e in old.STATE['plan']['negative']}
    else:
        from ..s7_consensus_v1.heldout_v14 import initialize as heldout_initialize
        heldout_initialize(dict(sources=config['heldout_sources'], split=split, out=str(root)))
        negative = {e['pair_id']: e for e in old.STATE['negative_plan']}
    manifest = root/('train_s7b_24k.json' if split == 'train' else 'archive_manifest.json')
    entries = read(manifest)['entries']
    groups = {}
    for entry in entries:
        slot, ordinal = map(int, Path(entry['artifact_path']).stem.split('_'))
        groups.setdefault(slot, {})[ordinal] = entry
    if any(set(g) != {0, 1} for g in groups.values()):
        raise ValueError('baseline positive/negative pairing changed')
    STATE.clear(); STATE.update(config=config, split=split, root=root, negative=negative,
        groups=groups, bank=old.STATE['bank'])


@lru_cache(maxsize=6)
def baseline(slot, ordinal):
    root = STATE['root']; entry = STATE['groups'][slot][ordinal]
    original = (old.clean_positive(entry['source_pair_id']) if ordinal == 0 else
                old.negative_source(STATE['negative'][entry['source_pair_id']])[0])
    prior, report = load_sample(root/entry['artifact_path'])
    scale = report['pair_shared_scale']
    if scale.get('heldout_original_geometry_unchanged'):
        before = original
    else:
        before, _ = pair_shared_scale(original, scale['requested_mean_area_px2'],
                                      topology_backoff=True, identity_fallback=True)
    if entry['offline_paired_mirror']:
        before = canonical_mirrored_contours(paired_mirror(before, entry['offline_paired_mirror']))
    with np.load(root/entry['weather_artifact'], allow_pickle=False) as z:
        fields = {k:z[k].copy() for k in z.files}
    for side in 'ab':
        if not np.array_equal(getattr(before, 'mask_'+side)[0],
                              np.unpackbits(fields['packed_preweather_'+side], axis=1)):
            raise ValueError('cannot reconstruct audited fragment raster')
    if entry.get('background_artifact'):
        with np.load(root/entry['background_artifact'], allow_pickle=False) as z:
            primary = replace(before, **{'mask_'+s:np.unpackbits(z['packed_before_'+s], axis=1)
                              .astype(np.float32)[None] for s in 'ab'})
    else:
        primary = prior
    return dict(entry=entry, original=before, old_primary=primary, fields=fields,
                old_sample=prior, old_report=report, slot=slot, ordinal=ordinal)


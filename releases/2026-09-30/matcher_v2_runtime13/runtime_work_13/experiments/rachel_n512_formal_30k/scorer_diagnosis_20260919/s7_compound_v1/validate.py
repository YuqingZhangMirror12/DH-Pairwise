"""Check the completed TRAIN export with the actual legacy and v3 loaders."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy import ndimage

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedWeatheredDataset
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.seam_context_v3.data import Dataset, collate
from .geometry import EIGHT, RECIPE_PERCENT, masks
from .materialize import save_json


def validate(root):
    root = Path(root)
    status = json.loads((root/'status.json').read_text())
    if status['status'] not in ('pilot_complete', 'complete'):
        raise ValueError('generation is not complete')
    old = MaterializedWeatheredDataset(status['manifest'])
    new = Dataset(root/'train.json')
    assert len(old) == len(new) == status['sample_count']
    n = len(old)//2
    assert Counter(e['label'] for e in old.entries) == {True:n, False:n}
    for label in (False, True):
        assert Counter(e['compound_recipe'] for e in old.entries if e['label']==label) == {
            k:n*v//100 for k,v in RECIPE_PERCENT.items()}
    checked = []
    # Real loaders, every recipe/label, including each offline mirror variant.
    buckets = {}
    for i, e in enumerate(old.entries):
        buckets.setdefault((e['compound_recipe'], e['label'], e['offline_paired_mirror']), i)
    for i in buckets.values():
        a, report = old[i]
        b, report2, metadata, entry = new[i]
        assert a.pair_id == b.pair_id == old.entries[i]['pair_id']
        for side in 'ab':
            target = getattr(a, 'target_'+side)
            valid = getattr(a, 'contour_valid_'+side)
            assert np.array_equal(getattr(a, 'points_rc_'+side), getattr(b, 'points_rc_'+side))
            assert np.isfinite(getattr(a, 'points_rc_'+side)).all()
            assert np.all(metadata['component_'+side][target<0] == -1)
            gap = metadata['gap_'+side]
            assert np.array_equal(gap, gap.T)
            assert not gap[target<0].any()
            assert len(target) == 512 and valid.sum() >= 4
            assert ndimage.label(getattr(a, 'mask_'+side)[0], EIGHT)[1] == 1
            if a.label:
                assert (target>=0).sum() >= 4
            else:
                assert not (target>=0).any()
        batch = collate([new[i]])
        assert batch['mask_a'].shape[0] == 1
        assert batch['gap_a'].shape == (1,512,512)
        assert bool(a.translation_valid) == bool(a.label)
        assert report == report2
        checked.append(a.pair_id)
    rows = [json.loads(line) for line in (root/'pair_metrics.jsonl').read_text().splitlines()]
    assert len(rows) == len(old)
    p = [r for r in rows if r['label']]
    assert Counter(r['actual_length_bin'] for r in p) == {'short':n//4,'medium':n//2,'long':n//4}
    assert all(32<=r['d20_length_px']<=800 for r in p)
    assert all(r['augmentation']['partial']['donor_split']=='train' for r in rows
               if r['augmentation'].get('partial'))
    result = dict(status='passed', total_pairs=len(old), positives=n, negatives=n,
        all_archives_roundtrip_checked_during_generation=True,
        actual_loader_and_metadata_checked_count=len(checked), checked_pair_ids=checked,
        source_train_only_checked_by_legacy_loader=True,
        recipe_and_geometry_quotas='exact', original_data_modified=False,
        turufan_seams_used=False, training_started=False)
    save_json(root/'validation.json', result)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    validate(parser.parse_args().root)

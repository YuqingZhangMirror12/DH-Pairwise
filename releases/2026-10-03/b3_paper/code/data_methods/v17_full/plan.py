"""Fixed split/recipe/cut-side/K assignment before geometric rejection sampling."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import numpy as np
from . import COUNTS, REVISION, SEED
from ..s7_compound_v1.materialize import read, save_json, digest


def side_schedule(n, seed):
    if n % 10:
        raise ValueError('70/30 quota needs complete ten-group blocks')
    rng = np.random.default_rng(seed)
    return [str(x) for _ in range(n//10) for x in rng.permutation(['smaller']*7+['larger']*3)]


def make_tasks(root, split):
    root = Path(root)
    manifest = root/('train_s7b_24k.json' if split == 'train' else 'archive_manifest.json')
    entries = read(manifest)['entries']; n = COUNTS[split]//2
    if len(entries) != 2*n:
        raise ValueError('baseline population count changed')
    by_slot = defaultdict(dict)
    for e in entries:
        slot, ordinal = map(int, Path(e['artifact_path']).stem.split('_'))
        by_slot[slot][ordinal] = e
    rng = np.random.default_rng(SEED+list(COUNTS).index(split)); sides = side_schedule(n, SEED+71+list(COUNTS).index(split))
    tasks = []; buckets = defaultdict(list)
    # Existing v14 lengths, structural strata, Partial mode and mirrors remain
    # the sampling strata. Negative identity is NEVER replaced on a rejection.
    for slot in range(n):
        pair = by_slot[slot]
        if set(pair) != {0,1} or not pair[0]['label'] or pair[1]['label']:
            raise ValueError('positive/negative slot mismatch')
        group = read(root/'groups'/f'{slot:05d}.json')
        recipe = pair[0]['corrosion_recipe']
        partial = group['statistics'][0]['augmentation'].get('partial') or {}
        key = (recipe, pair[0]['source_stratum'], group['length_bin'],
               pair[0]['offline_paired_mirror'], partial.get('mode'))
        buckets[key].append(slot)
        tasks.append(dict(slot=slot, recipe=recipe, key=key, size_class=sides[slot],
            mode='one' if slot % 2 else 'both', allowed_modes=['one','both'], k=0,
            negative_kind=pair[1]['negative_kind'], partial_mode=partial.get('mode')))
    gap_ids = [t['slot'] for t in tasks if t['recipe'].startswith('gaps')]
    k_values = np.resize(np.array([1,2,3,4]), len(gap_ids)); rng.shuffle(k_values)
    for slot, k in zip(gap_ids, k_values): tasks[slot]['k'] = int(k)
    for t in tasks:
        others = [i for i in buckets[t.pop('key')] if i != t['slot']]
        local = np.random.default_rng(SEED+t['slot']+100000*list(COUNTS).index(split))
        # Latest explicit user direction: never replace the original example
        # merely to force the new cut. Retain its audited v14 pair on rejection.
        t['positive_candidates'] = [t['slot']]
    return dict(split=split, revision=REVISION, tasks=tasks,
        baseline_archive_manifest=str(manifest), baseline_archive_sha256=digest(manifest),
        baseline_recipe_counts=dict(Counter(t['recipe'] for t in tasks)),
        side_counts=dict(Counter(t['size_class'] for t in tasks)),
        notch_counts=dict(Counter(str(t['k']) for t in tasks if t['k'])),
        rejection_policy='same original pair only; bounded one/both endpoint attempts; if no admissible v17 result preserve BOTH original v14 samples exactly',
        original_attempts=12, alternative_attempts=8,
        final_gap_gate_px=[5,35], notches=[1,4], no_new_base_tears=True)


def pilot_slots(plan):
    chosen = {}
    for t in plan['tasks']:
        # Cover all exclusive recipes/Partial modes, both side sizes, and each K.
        key = (t['recipe'], t['partial_mode'], t['size_class'], t['k'])
        chosen.setdefault(key, t['slot'])
    return sorted(chosen.values())

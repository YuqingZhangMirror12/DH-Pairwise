"""Scratch-model raw data/target metadata and source-disjoint simulation views.

--accept-validation-rebuild is an explicit protocol amendment, not a silent
fallback to random Pair splitting. Does not read any real dataset.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .targets import base_target_metadata, known_gap_links


ROOT = Path('/root/autodl-tmp')
TRAIN = ROOT/'rachel_score_design_20260913_001/s6_s7_20260915/preparation/data/train_s7_24k.json'
RELEASE = ROOT/'dataset_rachel_pairwise_n512_v1'


def read(p): return json.loads(Path(p).read_text())


def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    t.replace(p)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def family(name):
    stem = Path(name).stem.lower()
    return re.split(r'[_\-]?(?:recto|verso|seite\d+|detail|total)', stem, maxsplit=1)[0]


def sources(row):
    return {family(row['fragment_'+s]['split_unit_id']) for s in 'ab'}


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def row_dataset(root, selected, split):
    from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset, RachelDatasetConfig, _read_selected_pair
    dataset = object.__new__(RachelPairDataset)
    dataset.root, dataset.split, dataset.config = Path(root), split, RachelDatasetConfig()
    dataset._rows = tuple(_read_selected_pair(Path(root), split, row, i+1) for i, row in enumerate(selected))
    return dataset


def clean_report(sample):
    from staging.pairwise_v0_2.pairwise_data.rachel_staged_damage_dataset import clean_report as original
    return original(sample, epoch=1)


def train_metadata(task):
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
    entry, sample_path, out = task
    p = Path(out)
    if p.exists():
        with np.load(p,allow_pickle=False) as z:
            return dict(path=str(p),cached=True,recipe=entry['s7_recipe'],
                        gap_links_a=int(z['gap_a'].sum()),gap_links_b=int(z['gap_b'].sum()))
    sample, report = load_sample(sample_path)
    extra = base_target_metadata(sample)
    reason = 'no documented removed source arc; observed GT runs only'
    if entry['s7_recipe'] in ('seam_gaps', 'local') and report['changed_pair'] and sample.label:
        source = row_dataset(entry['source_root'], [entry['source_row']], 'train')[0]
        extra = known_gap_links(sample, source, report)
        reason = 'own-source arcs + original GT + recorded removal; unknown links ignored'
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **extra)
    return dict(path=str(p), gap_links_a=int(extra['gap_a'].sum()), gap_links_b=int(extra['gap_b'].sum()),
                reason=reason, recipe=entry['s7_recipe'], changed=bool(report['changed_pair']))


def audit_sources():
    train = read(TRAIN)
    val = list(rows(RELEASE/'pairs/val.jsonl'))
    ts = set().union(*(sources(e['source_row']) for e in train['entries']))
    vs = set().union(*(sources(e) for e in val))
    retained = [r for r in val if not sources(r)&ts]
    independent = vs-ts
    # Inspect only original SIMVAL SOURCE inventory, not TEST performance.
    positives = Counter(); negatives = Counter()
    for row in rows(RELEASE/'manifests/within_candidates.jsonl'):
        src = family(row.get('metadata', {}).get('image_name', ''))
        if src in independent and row.get('main_training_eligible'):
            (positives if row['label'] else negatives)[src] += 1
    return train, ts, vs, dict(train_groups=len(ts), original_val_groups=len(vs),
        overlapping_families=sorted(ts&vs), retained_original_pairs=len(retained),
        retained_original_labels=dict(Counter(str(r['label']) for r in retained)),
        independent_val_sources=sorted(independent), available_positive=dict(positives),
        available_within_negative=dict(negatives), normalization='explicit recto/verso/seite/detail/total prefix',
        caveat='known naming aliases, not an externally complete manuscript catalogue')


def build_validation(out, independent, seed=260921):
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import save_sample
    from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import strong_pair, stable_rng
    rng = np.random.default_rng(seed)
    groups = sorted(independent)
    rng.shuffle(groups)
    cal = set(groups[:max(1, round(len(groups)/3))]); select = set(groups)-cal
    fragments = {}
    for row in rows(RELEASE/'manifests/fragments.jsonl'):
        if family(row['split_unit_id']) in independent:
            # Never pass parent audit fields to the model-facing loader.
            fragments[row['fragment_token']] = {k: row[k] for k in
                ('fragment_token', 'split_unit_id', 'model_mask_path', 'contour_path')}
    by_source = defaultdict(lambda: {True: [], False: []})
    for row in rows(RELEASE/'manifests/within_candidates.jsonl'):
        if not row.get('main_training_eligible'):
            continue
        ta, tb = row['fragment_a_token'], row['fragment_b_token']
        if ta not in fragments or tb not in fragments:
            continue
        a, b = fragments[ta], fragments[tb]
        src = family(a['split_unit_id'])
        simple = dict(pair_id=row['pair_id'], label=bool(row['label']), split='val', fragment_a=a, fragment_b=b,
            correspondence_path=row['correspondence_path'] if row['label'] else None)
        by_source[src][bool(row['label'])].append(simple)
    selected = [];used_cross_pairs = set()
    for source in sorted(by_source):
        pool = by_source[source]
        # 75 positives/source: at most3000 balanced pairs, preserve source diversity.
        order = rng.permutation(len(pool[True]))[:75]
        pos = [pool[True][int(i)] for i in order]
        neg = [pool[False][int(i)] for i in rng.permutation(len(pool[False]))[:len(pos)//2]]
        fold_sources = cal if source in cal else select
        opposite = sorted(k for k, frag in fragments.items() if family(frag['split_unit_id']) in fold_sources-{source})
        for ix in range(len(pos)-len(neg)):
            first = pos[ix % len(pos)]['fragment_a']
            for _ in range(10000):
                other = fragments[opposite[int(rng.integers(len(opposite)))]]
                identity = tuple(sorted((first['fragment_token'],other['fragment_token'])))
                if identity not in used_cross_pairs:
                    used_cross_pairs.add(identity);break
            else:raise RuntimeError('independent negative pool exhausted')
            ident = hashlib.sha256('|'.join(identity).encode()).hexdigest()[:24]
            neg.append(dict(pair_id='v3-val-cross-'+ident, label=False, split='val', fragment_a=first,
                            fragment_b=other, correspondence_path=None))
        selected.extend(pos+neg)
    if len(selected) < 256:
        raise RuntimeError('insufficient independent simulation validation sources')
    if len({r['pair_id'] for r in selected})!=len(selected):
        raise RuntimeError('duplicate validation Pair ID')
    manifests = defaultdict(list)
    for ix, (row, sample) in enumerate(zip(selected, row_dataset(RELEASE, selected, 'val'))):
        side = 'cal' if sources(row) <= cal else 'select'
        if not (sources(row) <= (cal if side == 'cal' else select)):
            raise ValueError('cross-fold negative/source leak')
        for view in ('clean', 'hard'):
            current, report = sample, clean_report(sample)
            if view == 'hard':
                mode = ('wave', 'local', 'seam_gaps')[ix % 3]
                current, report = strong_pair(sample, mode, stable_rng(seed, sample.pair_id, 'validation-hard'))
            target = out/'validation_samples'/f'{ix:05d}_{view}.npz'
            save_sample(target, current, report)
            extra = known_gap_links(current, sample, report) if report.get('changed_pair') else base_target_metadata(current)
            meta = out/'validation_targets'/f'{ix:05d}_{view}.npz'
            meta.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(meta, **extra)
            manifests[side+'_'+view].append(dict(pair_id=sample.pair_id, sample_path=str(target),
                target_metadata=str(meta), label=int(sample.label), sources=sorted(sources(row)),
                view=view, changed=bool(report.get('changed_pair')), fallback=report.get('fallback_reason')))
    for name, entries in manifests.items():
        save(out/(name+'.json'), dict(schema='seam-context-v3-data/1', entries=entries))
    return dict(cal_sources=sorted(cal), select_sources=sorted(select),
                counts={k: len(v) for k, v in manifests.items()},
                labels={k:dict(Counter(str(e['label']) for e in v)) for k,v in manifests.items()},
                hard_actual_changed={k: sum(e['changed'] for e in v) for k, v in manifests.items()})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True)
    p.add_argument('--audit-only', action='store_true'); p.add_argument('--accept-validation-rebuild', action='store_true')
    p.add_argument('--workers', type=int, default=4); args = p.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    train, ts, vs, audit = audit_sources(); save(out/'source_audit.json', audit); print(json.dumps(audit), flush=True)
    if args.audit_only: return
    if not args.accept_validation_rebuild:
        raise RuntimeError('source overlap requires explicit validation-protocol confirmation')
    entries, tasks = [], []
    for ix, e in enumerate(train['entries']):
        src = str(Path(train['artifact_root'])/e['artifact_path']); meta = str(out/'train_targets'/f'{ix:05d}.npz')
        entries.append(dict(pair_id=e['pair_id'], sample_path=src, target_metadata=meta,
            label=int(e['label']), sources=sorted(sources(e['source_row'])), recipe=e['s7_recipe']))
        tasks.append((e, src, meta))
    counts = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for ix, result in enumerate(pool.map(train_metadata, tasks, chunksize=16)):
            counts.append(result)
            if (ix+1) % 4000 == 0: print(json.dumps(dict(event='targets_prepared', pairs=ix+1)), flush=True)
    save(out/'train.json', dict(schema='seam-context-v3-data/1', original_manifest=str(TRAIN),
        original_manifest_sha256=sha(TRAIN), entries=entries))
    save(out/'target_inventory.json', counts)
    val = build_validation(out, vs-ts)
    save(out/'protocol.json', dict(status='ready', train_count=len(entries), original_s7_manifest_sha256=sha(TRAIN),
        validation_amendment='same original SIMVAL independent source pool; aliases excluded; negatives rebuilt within folds',
        validation_rebuild_approved=True,
        source_audit=audit, validation=val, real_or_test_used=False,
        gap_policy='documented source removal inside fully observed clean GT run; other links ignore'))
    print(json.dumps(dict(event='data_ready', **val)), flush=True)


if __name__ == '__main__': main()

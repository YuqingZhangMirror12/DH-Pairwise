"""Read-only admission of the colleague's unchanged900-pair v4.2 SELECT.

This is a supplemental developmental evaluation population, never TRAIN and
never a second independent900-pair source sample. No model is run here.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919'
ROOT = Path('/root/autodl-tmp/matcher_v2_20260930')
REFERENCE = Path('/root/autodl-tmp/claudecode0929/data/sseam_v42_select/manifest.json')
BOUND = {
    'reference': (REFERENCE, '92413b080e72df7e13b122110f0ef159ff43d4584ef79425cf38eb8addc5f436'),
    'sources': (ROOT/'source_admission_02/source_audit.json', '9a5cb65870507134900921560291606f1e695e07b6bffa7985ff50ab5800f91a'),
    'real_exclusion': (ROOT/'real_exclusion_audit_01.json', '497d57147b5efc28282fdba99b4c9ba865200e521f5adde53b86bae151dd2dbe'),
    'training': (ROOT/'strict_training_admission_01/combined_admission.json', '07eb6da665f88088f872e64eb647ba419d3016f5c159326b7876cec21faed379'),
    'new_select': (ROOT/'v42_strict_dataset_01/select900/manifest.json', 'ee98a3a466b04d7c86aae2ffe4ae573fd5ad26a058577a5ae713e4344f00fcda'),
}
COUNTS = {'straight_M': 100, 'straight_J': 200, 'straight_R': 150}  # per label
GENERATORS = {'gen2voronoi_1', 'gen2voronoi_2', 'gen3voronoi', 'gen4voronoi',
              'gen4voronoi_1_3', 'gen5voronoi_1_1_3'}
BASES = {'M': {'rachel'}, 'J': {'torn_rachel', 'torn_strip', 'margin_fragment'}, 'R': {'strip'}}
PROCEDURAL = {'strip', 'torn_strip', 'margin_fragment'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read_bound(path, expected):
    require(sha(path) == expected, 'bound input changed: ' + str(path))
    return read(path)


def validate_manifest(record):
    require(record.get('schema') == 'claude-straight-seam-v4/1'
            and record.get('split') == 'select' and record.get('revision') == 'v4.2'
            and record.get('seed') == 26093015 and record.get('failed') == [],
            'unchanged complete reference SELECT required')
    entries = record['entries']
    require(len(entries) == len({e['pair_id'] for e in entries}) == len({e['id'] for e in entries}) == 900,
            'complete900 unique reference identities required')
    require(all(type(e['label']) is bool and e['split'] == 'select'
                and e['generator'] == 'claude-straight-seam-v4' for e in entries), 'reference roles/labels differ')
    expected = Counter({(k, label): n for k, n in COUNTS.items() for label in (True, False)})
    require(Counter((e['recipe'], e['label']) for e in entries) == expected, 'reference type/label counts differ')
    return entries


def source_lookup(audit, real):
    require(audit['status'] == 'passed' and audit['errors'] == []
            and all(n == 0 for n in audit['collision_counts'].values()), 'source separation audit failed')
    require(real['source_admission_sha256'] == BOUND['sources'][1], 'real exclusion belongs to different inventory')
    screen = real['synthetic_inventory']['select']
    require(not screen['invalid_synthetic_paths'] and not screen['exact_prepared_mask_matches']
            and not screen['potential_turufan_parent_aliases'], 'known real donor match in SELECT inventory')
    selected = audit['inventory']['select']
    others = audit['inventory']['train'] + audit['inventory']['test']
    for field in ('path', 'source_family', 'file_sha256', 'pixel_sha256', 'crop_pixel_sha256'):
        require(not ({r[field] for r in selected} & {r[field] for r in others}), 'SELECT/source collision: ' + field)
    lookup = {r['path']: r for r in selected}
    require(len(lookup) == len(selected) == 150, 'fixed SELECT150 donor inventory required')
    return Path(audit['source_root']).resolve(), lookup


def donor_records(entry, source_root, lookup):
    """Retain procedural ancestry explicitly; never guess a real donor alias."""
    require(Path(entry['source_root']).resolve() == source_root, 'unregistered source root')
    meta = entry['meta']; kind = entry['recipe'][-1]
    require(meta['type'] == kind and meta['base'] in BASES[kind]
            and meta['generator_revision'] == 'v4.2', 'reference generator metadata differs')
    leaves = [meta] if entry['label'] else [meta['a'], meta['b']]
    result = []
    for leaf in leaves:
        require(leaf['type'] == kind and leaf['base'] == meta['base'], 'piece source type/base differs')
        if leaf['base'] in PROCEDURAL:
            require(not leaf.get('source_mask') and not leaf.get('source_family'), 'procedural leaf has undeclared donor')
            result.append(dict(kind='procedural', base=leaf['base']))
        else:
            relative = Path(leaf['source_mask'])
            require(not relative.is_absolute() and len(relative.parts) == 5
                    and relative.parts[:2] == ('model', 'masks_800')
                    and relative.parts[2] in GENERATORS and relative.parts[3].isdigit()
                    and relative.suffix == '.png' and relative.stem.isdigit(), 'non-synthetic or unsafe donor path')
            path = (source_root/relative).resolve()
            require(source_root in path.parents and str(path) in lookup, 'donor not in source-isolated SELECT allowlist')
            source = lookup[str(path)]
            require(source['source_family'] == leaf['source_family'], 'donor source-family differs')
            result.append(dict(kind='synthetic_voronoi', **source))
    if not entry['label'] and all(r['kind'] == 'synthetic_voronoi' for r in result):
        require(result[0]['source_family'] != result[1]['source_family'], 'negative reuses same synthetic parent')
    return result


def inspect_entry(entry, loader, catalog, *, reference_root=None):
    path = Path(entry['sample_path']).resolve()
    if reference_root is not None:
        require(path.parent == (reference_root/'samples').resolve()
                and (reference_root/entry['artifact_path']).resolve() == path, 'reference sample escaped its fixed directory')
    require(sha(path) == entry['sample_sha256'], 'materialized sample hash differs')
    sample, report = loader(path)
    require(sample.pair_id == entry['pair_id'] and bool(sample.label) == entry['label'], 'official loader identity differs')
    for side in 'ab':
        mask = getattr(sample, 'mask_' + side)
        points = getattr(sample, 'points_rc_' + side)
        valid = getattr(sample, 'contour_valid_' + side)
        require(mask.shape == (1, 800, 800) and points.shape == (512, 2) and valid.shape == (512,)
                and mask.dtype == points.dtype == np.float32 and valid.dtype == np.bool_
                and np.isin(mask, [0, 1]).all() and valid.sum() >= 4,
                'unaltered800/N512 binary-mask inputs required')
    require(bool(sample.translation_valid) == entry['label'], 'reference label/GT presence differs')
    if reference_root is not None:
        require(report['split'] == 'select' and report['generator_meta'] == entry['meta']
                and report['gt_used_for_inputs'] is False, 'reference report metadata differs')
        if entry['label']:
            expected = np.asarray(entry['meta']['shift_b']) - np.asarray(entry['meta']['shift_a'])
            require(np.array_equal(sample.translation_a_to_b_rc, expected), 'centering-derived GT translation differs')
        else:
            require((sample.target_a < 0).all() and (sample.target_b < 0).all(), 'negative has matched targets')
    return dict(pair_id=entry['pair_id'], label=entry['label'], recipe=entry['recipe'],
        sample_path=str(path), sample_sha256=entry['sample_sha256'],
        model_input_sha256=catalog.tensor_digest(sample, catalog.MATCHER_INPUTS),
        target_sha256=catalog.tensor_digest(sample, catalog.TARGETS))


def duplicate_report(reference, train, new_select):
    """Exact six-input identity is not invariant pre-damage or manuscript identity."""
    def index(rows):
        result = {}
        for row in rows:
            result.setdefault(row['model_input_sha256'], []).append(row['pair_id'])
        return result
    a, b, c = map(index, (reference, train, new_select))
    return dict(within_reference=[ids for ids in a.values() if len(ids) > 1],
        with_training=[dict(reference=a[h], training=b[h], sha256=h) for h in sorted(a.keys() & b.keys())],
        with_new_select=[dict(reference=a[h], new_select=c[h], sha256=h) for h in sorted(a.keys() & c.keys())])


def run(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only admission required')
    require(not args.out.exists(), 'exclusive new admission output required')
    inputs = {k: read_bound(*bound) for k, bound in BOUND.items()}
    entries = validate_manifest(inputs['reference'])
    source_root, lookup = source_lookup(inputs['sources'], inputs['real_exclusion'])
    train = inputs['training']
    require(train['training_admitted'] is True and train['original_exposures_removed'] == 0
            and len(train['catalog']) == 27179, 'current complete B3 training catalog required')
    newer = inputs['new_select']
    require(newer['split'] == 'select' and not newer['failed'] and len(newer['entries']) == 900,
            'complete new900 SELECT required; no TEST inputs loaded')
    source = args.source_root.resolve(); sys.path.insert(0, str(source))
    loader_api = importlib.import_module('staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset')
    catalog = importlib.import_module(PACKAGE + '.curriculum_training_v1.catalog')
    for module in (loader_api, catalog):
        require(source in Path(module.__file__).resolve().parents, 'wrong official loader/source imported')
    args.out.mkdir(); started = time.time()
    code = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    bound = {k: dict(path=str(p), sha256=h) for k, (p, h) in BOUND.items()}
    bound['official_loader'] = binding(loader_api.__file__); bound['canonical_inputs'] = binding(catalog.__file__)
    save(args.out/'protocol.json', dict(schema='reference-select-admission-protocol/1', inputs=bound,
        code_sha256=code, training_admitted=False, device='cpu', neural_inference=False, test_samples_loaded=False))
    rows = []; all_donors = {}
    for entry in entries:
        donors = donor_records(entry, source_root, lookup)
        for donor in donors:
            if donor['kind'] == 'synthetic_voronoi':
                all_donors[donor['path']] = donor
        row = inspect_entry(entry, loader_api.load_sample, catalog, reference_root=REFERENCE.parent)
        row['donors'] = donors; rows.append(row)
    for path, donor in all_donors.items():
        require(sha(path) == donor['file_sha256'], 'actual synthetic donor changed')
    other = [inspect_entry(e, loader_api.load_sample, catalog) for e in newer['entries']]
    duplicates = duplicate_report(rows, train['catalog'], other)
    save(args.out/'duplicate_audit.json', duplicates)
    require(not duplicates['with_training'] and not duplicates['within_reference'], 'reference/TRAIN or internal duplicate')
    other_donors = {r['path'] for e in newer['entries'] for r in donor_records(e, source_root, lookup)
                    if r['kind'] == 'synthetic_voronoi'}
    groups = {'all': [r['pair_id'] for r in rows]}
    for recipe in COUNTS:
        groups[recipe] = [r['pair_id'] for r in rows if r['recipe'] == recipe]
    fixed = [i for recipe in COUNTS for i in sorted(r['pair_id'] for r in rows if r['recipe'] == recipe and r['label'])[:10]]
    summary = dict(schema='reference-select-admission/1', status='admitted_for_supplemental_evaluation_only',
        input_bindings=bound, pairs=900, positives=450, negatives=450,
        type_counts={k: len(groups[k]) for k in COUNTS}, rows=rows, groups=groups, fixed_diagnostic_ids=fixed,
        within_reference_duplicates=0, exact_training_input_duplicates=0,
        exact_new_select_duplicates=len(duplicates['with_new_select']),
        synthetic_donor_masks=len(all_donors), synthetic_donor_families=len({r['source_family'] for r in all_donors.values()}),
        procedural_pairs=sum(all(d['kind'] == 'procedural' for d in r['donors']) for r in rows),
        new_select_shared_donor_masks=len(set(all_donors) & other_donors),
        source_disjoint_from_registered_sim_train_and_test=True,
        direct_real_donor_screen_passed=True, complete_real_manuscript_alias_independence_proved=False,
        independently_sampled_source_population=False, pre_damage_pair_identity_proved=False,
        legacy_damage_ignore_targets_independently_proved=False,
        thresholds_or_checkpoint_selected_here=False, training_admitted=False,
        reference_samples_modified=False, model_scores_used_for_filtering=False,
        test_samples_loaded=False, neural_inference=False, cuda_initialized=False,
        caveats=['Same heldout donor pool as the new SELECT; report separately, not1800 independent samples.',
                 'Legacy reference lacks explicit pre-damage/mismatch trace; this is not training admission.',
                 'Exact six-input and registered-source disjointness do not prove all manuscript aliases or transformed ancestry.'],
        elapsed_seconds=time.time()-started)
    for view in bound.values():
        require(sha(view['path']) == view['sha256'], 'input changed during admission')
    require(code == {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}, 'companion source changed during admission')
    save(args.out/'admission.json', summary)
    complete = dict(status='complete', pairs=900, training_admitted=False, files={name: sha(args.out/name)
        for name in ('protocol.json', 'duplicate_audit.json', 'admission.json')},
        actual_process_return_must_be_verified_separately=True)
    save(args.out/'complete.json', complete)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'groups', 'fixed_diagnostic_ids')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except BaseException as error:
        if not existed:
            args.out.mkdir(parents=True, exist_ok=True)
            save(args.out/'failure.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()

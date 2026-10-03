"""Explicit new CAL/SELECT Gen4/5 groups through unchanged Rachel preprocessing.

Never calls the historical inventory, 30k selector, or TRAIN/VAL/TEST splitter.
The role remains CAL or SELECT even when using native loader primitives. CSV
adjacency and independent seam/residual gates remain those of the bound source.
Rejected groups/pairs are recorded, never relabelled or silently repaired.
"""
from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import sys


SCHEMA = 'mixed-heldout-base-release/1'
ROLES = ('cal', 'select')
GENERATORS = {'gen4voronoi': 4, 'gen4voronoi_1_3': 4, 'gen5voronoi_1_1_3': 5}
MODULE_PREFIX = 'staging.pairwise_v0_2.pairwise_data.'
MODULES = ('rachel_preprocess', 'rachel_30k_selection', 'prepare_rachel_pairwise',
           'rachel_training_dataset')
EXCLUSIONS = ('all_train_including_donors', 'all_original_and_published_test_including_donors')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_file(path, sha):
    p = Path(path)
    require(p.is_absolute() and not p.is_symlink() and p.is_file(), 'absolute regular source required')
    require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha), 'source SHA required')
    require(file_sha(p) == sha, 'source changed: ' + str(p))
    return p.resolve(strict=True)


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def approved_parents(plan):
    require(plan.get('schema') == 'task1-evaluation-only-parent-plan/1', 'wrong parent plan')
    require(set(plan.get('folds', {})) == set(ROLES), 'only CAL/SELECT allowed')
    blocked = set(plan.get('hard_exclusions', {}))
    for key in EXCLUSIONS:
        require(isinstance(plan.get('exclusion_families', {}).get(key), list), 'missing exclusions')
        blocked.update(plan['exclusion_families'][key])
    result, image_roles = {}, {}
    for role, rows in plan['folds'].items():
        require(isinstance(rows, list) and rows, 'empty parent role')
        result[role] = {}
        for row in rows:
            family = row['family']
            require(row['role'] == role and family not in blocked, 'excluded/misassigned family')
            require(family not in result[role], 'duplicate family')
            require(row.get('edge_donors_must_be_from_same_role_families') is True, 'donor boundary missing')
            images = row['allowed_same_family_variants']
            require(row['parent_image'] in images, 'primary parent must be an approved image')
            for image in images:
                sha = image['file_sha256']
                require(image.get('mode') == 'RGBA' and image.get('has_embedded_original_alpha') is True,
                        'original RGBA/alpha required')
                require(image_roles.setdefault(sha, role) == role, 'parent image shared across roles')
            result[role][family] = row
    require(not set(result['cal']) & set(result['select']), 'parent family shared across roles')
    return result


def validate_group(group, parents, *, verify_files=True):
    """Validate actual CSV/JPEG identities plus every sampled edge donor."""
    role, generator, family = group['role'], group['generator'], group['family']
    require(role in ROLES and generator in GENERATORS, 'new CAL/SELECT Gen4/5 only')
    require(family in parents[role], 'primary parent outside role')
    gid = group['group_id']
    require(isinstance(gid, str) and re.fullmatch('[A-Za-z0-9_-]+', gid), 'unsafe group ID')
    require(gid.startswith(role + '__' + family + '__'), 'new group ID must contain role and family')
    primary = parents[role][family]['parent_image']
    require(group['parent_image_path'] == primary['path'] and
            group['parent_image_sha256'] == primary['file_sha256'], 'primary image binding mismatch')
    checks = {primary['path']: primary['file_sha256']}
    donors = group.get('edge_donors')
    require(isinstance(donors, list) and len(donors) == GENERATORS[generator], 'exact edge donor list required')
    variants = set()
    for donor in donors:
        require(donor['family'] in parents[role], 'edge donor outside role or excluded')
        variants_allowed = parents[role][donor['family']]['allowed_same_family_variants']
        require(any(donor['parent_image_path'] == i['path'] and donor['parent_image_sha256'] == i['file_sha256']
                    for i in variants_allowed), 'edge donor parent image not approved')
        require(type(donor.get('seed')) is int and donor['seed'] >= 0, 'edge RNG seed required')
        for path_key, sha_key in (('parent_image_path', 'parent_image_sha256'),
                                  ('raw_edge_path', 'raw_edge_sha256'),
                                  ('variant_path', 'variant_sha256')):
            path, sha = donor[path_key], donor[sha_key]
            require(path not in checks or checks[path] == sha, 'conflicting source identity')
            checks[path] = sha
        require(donor['variant_sha256'] not in variants, 'duplicate sampled edge pixels/file')
        variants.add(donor['variant_sha256'])
    csv_path = Path(group['csv_path'])
    require(csv_path.name == 'label.csv', 'original label.csv required')
    checks[str(csv_path)] = group['csv_sha256']
    fragments = group['fragments']
    require(len(fragments) == GENERATORS[generator], 'fragment count changed')
    require(len({f['id'] for f in fragments}) == len(fragments), 'duplicate fragment id')
    require(len({f['mask_id'] for f in fragments}) == len(fragments), 'duplicate mask id')
    for fragment in fragments:
        p = Path(fragment['path'])
        require(p.parent == csv_path.parent and p.name == str(fragment['mask_id']) + '.jpg',
                'fragment JPEG must use original mask_id beside CSV')
        checks[str(p)] = fragment['sha256']
    if 'source_derivation' in group:
        derivation = group['source_derivation']
        require(derivation.get('policy') == 'parent-inherited-outer-boundary-exact/1' and
                derivation.get('exact_one_block_patch') is True and
                derivation.get('all_other_original_gates_preserved') is True,
                'unapproved base boundary derivation')
        require(group.get('parent_alpha_intersection_verified') is True,
                'new base alpha intersection receipt missing')
        require(isinstance(group.get('parent_boundary_events'), list), 'boundary trace missing')
        require(len(group.get('applied_fragment_alpha', [])) == len(fragments), 'pristine applied alpha missing')
        refs = [group['regularized_parent'], group['composite'], group['provenance_receipt'],
                *group['masks'], *group['applied_fragment_alpha']]
        for ref in refs:
            checks[ref['path']] = ref['sha256']
        for prefix in ('policy', 'derived_export'):
            checks[derivation[prefix+'_path']] = derivation[prefix+'_sha256']
    if verify_files:
        for path, sha in checks.items():
            checked_file(path, sha)
        with csv_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == len(fragments), 'CSV fragment count mismatch')
        require({(r['id'].strip(), r['mask_id'].strip()) for r in rows} ==
                {(str(f['id']), str(f['mask_id'])) for f in fragments}, 'CSV fragment identities mismatch')
        require({r['image_name'].strip() for r in rows} == {Path(primary['path']).name},
                'CSV image_name is not bound original parent')
        actual = {p.name for p in csv_path.parent.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg')}
        require(actual == {Path(f['path']).name for f in fragments}, 'unlisted/extra JPEG in group')
    return checks


def native_modules(runtime_root, expected_sources):
    """Require actual imported modules to be the caller-bound frozen files."""
    runtime_root = Path(runtime_root).resolve(strict=True)
    require(set(expected_sources) == set(MODULES), 'all native source modules must be SHA-bound')
    sys.path.insert(0, str(runtime_root))
    loaded = {}
    for name in MODULES:
        expected_path = runtime_root / Path(*(MODULE_PREFIX + name).split('.')).with_suffix('.py')
        checked_file(expected_path, expected_sources[name])
        module = importlib.import_module(MODULE_PREFIX + name)
        require(Path(module.__file__).resolve() == expected_path.resolve(), 'native import came from another checkout')
        loaded[name] = module
    return loaded


def model_row(candidate, fragments, role):
    """Explicit allowlist: no RGB, parent masks, or parent transforms in inputs."""
    require(role in ROLES and type(candidate['label']) is bool, 'invalid heldout role/label')
    def fragment(token):
        row = fragments[token]
        return {key: row[key] for key in ('fragment_token', 'model_mask_path', 'contour_path')}
    return dict(pair_id=candidate['pair_id'], split=role, label=candidate['label'],
                fragment_a=fragment(candidate['fragment_a_token']),
                fragment_b=fragment(candidate['fragment_b_token']),
                correspondence_path=candidate.get('correspondence_path'),
                translation_a_to_b_rc=candidate.get('translation_a_to_b_rc'),
                translation_a_to_b_xy_cartesian=candidate.get('translation_a_to_b_xy_cartesian'))


def load_native_pair(root, role, row, runtime):
    """Reuse unchanged getitem and residual gates without inventing a VAL role."""
    require(role in ROLES, 'only heldout roles')
    dataset = object.__new__(runtime.RachelPairDataset)
    dataset.root = Path(root).resolve(strict=True)
    dataset.split, dataset.config = role, runtime.RachelDatasetConfig()
    dataset._rows = (runtime._read_selected_pair(dataset.root, role, row, 1),)
    return dataset[0]


def materialize_new_groups(*, parent_plan_path, expected_parent_sha, groups_manifest_path,
                           expected_groups_sha, output_new, frozen_runtime_root, expected_sources,
                           generation_plan_path=None, expected_generation_plan_sha=None):
    plan_file = checked_file(parent_plan_path, expected_parent_sha)
    groups_file = checked_file(groups_manifest_path, expected_groups_sha)
    plan = json.loads(plan_file.read_text())
    manifest = json.loads(groups_file.read_text())
    if generation_plan_path is not None:
        gp = checked_file(generation_plan_path, expected_generation_plan_sha)
        generation_plan = json.loads(gp.read_text())
        require(manifest.get('plan_sha256') == expected_generation_plan_sha, 'generation plan mismatch')
        if 'parent_plan_sha256' in generation_plan or 'parent_plan' in generation_plan:
            parent_binding = generation_plan.get('parent_plan', {})
            bound_sha = generation_plan.get('parent_plan_sha256', parent_binding.get('sha256'))
            require(bound_sha == expected_parent_sha, 'generation used another parent plan')
            if parent_binding:
                checked_file(parent_binding['path'], bound_sha)
            generation_binding = 'preregistered_parent_plan_sha'
        else:
            # The original pilot predates the combined plan/index JSON. Bind
            # its actual stored parent records; never backdate a new SHA claim.
            keys = ('role', 'family', 'path', 'file_sha256', 'decoded_rgba_sha256', 'alpha_sha256')
            expected = [{**row['parent_image'], 'role':role, 'family':row['family']}
                        for role, values in plan['folds'].items() for row in values]
            expected = {tuple(r[k] for k in keys) for r in expected}
            actual = {tuple(r[k] for k in keys) for r in generation_plan['parents']}
            require(actual == expected and len(actual) == len(generation_plan['parents']),
                    'stored generation parents differ from the audited plan')
            generation_binding = 'posthoc_exact_parent_records_crosscheck_not_preregistered_plan_sha'
        # A directly bound new parent plan does not waive the original source
        # and edge-pool byte binding used by the earlier pilot.
        for key in ('source_probe', 'edge_pool'):
            reference = generation_plan[key]
            checked_file(reference['path'], reference['sha256'])
        for path, sha in generation_plan['sources'].items():
            checked_file(path, sha)
        if 'source_derivation' in generation_plan:
            for key in ('parent_extension', 'regressions', 'replay'):
                reference = generation_plan[key]
                checked_file(reference['path'], reference['sha256'])
            derivation = generation_plan['source_derivation']
            for prefix in ('policy', 'derived_export'):
                checked_file(derivation[prefix+'_path'], derivation[prefix+'_sha256'])
    else:
        require(manifest.get('parent_plan_sha256') == expected_parent_sha, 'generation used another parent plan')
        generation_binding = 'preregistered_parent_plan_sha'
    require(manifest.get('status') in ('complete', 'pilot_shortfall', 'pilot_complete'), 'generation not finalized')
    groups = manifest['groups']
    require(isinstance(groups, list) and groups, 'no explicit generated groups')
    parents = approved_parents(plan)
    groups = deepcopy(groups)
    for group in groups:
        if 'label_csv' in group:
            group['csv_path'], group['csv_sha256'] = group['label_csv']['path'], group['label_csv']['sha256']
        for donor in group.get('edge_donors', []):
            if 'parent_path' in donor:
                donor['parent_image_path'], donor['parent_image_sha256'] = donor['parent_path'], donor['parent_sha256']
                donor['raw_edge_path'], donor['raw_edge_sha256'] = donor['path'], donor['sha256']
    all_checks, identities = {}, set()
    for group in groups:
        if generation_plan_path is not None and 'source_derivation' in generation_plan:
            require(group.get('source_derivation') == generation_plan['source_derivation'] and
                    group.get('parent_plan') == generation_plan['parent_plan'],
                    'group source/parent derivation differs from registered generation')
        identity = (group['generator'], group['group_id'])
        require(identity not in identities, 'duplicate generated group')
        identities.add(identity)
        for path, sha in validate_group(group, parents).items():
            require(path not in all_checks or all_checks[path] == sha, 'conflicting source SHA')
            all_checks[path] = sha
    modules = native_modules(frozen_runtime_root, expected_sources)
    output = Path(output_new).absolute()
    require(not output.exists(), 'output must be entirely new; no overwrite/resume')
    require(not any(output == p or output in p.parents or p in output.parents
                    for p in (plan_file, groups_file, *(Path(p) for p in all_checks))),
            'output overlaps protected source')
    output.mkdir(parents=True)
    preprocessing = modules['prepare_rachel_pairwise']
    runtime = modules['rachel_training_dataset']
    config = asdict(modules['rachel_preprocess'].RachelPreprocessConfig())
    records, rejected = [], []
    for group in groups:
        role, generator, gid = group['role'], group['generator'], group['group_id']
        source = output / 'input'
        target = source / generator / 'no_erode' / gid
        target.mkdir(parents=True)
        shutil.copy2(group['csv_path'], target / 'label.csv')
        for f in group['fragments']:
            shutil.copy2(f['path'], target / Path(f['path']).name)
        # Input copy hashes are checked as well; no links to a mutable old tree.
        require(file_sha(target / 'label.csv') == group['csv_sha256'], 'copied CSV changed')
        for f in group['fragments']:
            require(file_sha(target / Path(f['path']).name) == f['sha256'], 'copied JPEG changed')
        release = output / role
        marker = preprocessing._process_group((str(source), str(release), generator, gid, config))
        group_record = dict(role=role, generator=generator, family=group['family'], group_id=gid,
                            raw_source=group, marker=marker, admitted_positive_pairs=[])
        if marker['status'] == 'processed_group':
            require(marker['image_name'] == Path(group['parent_image_path']).name, 'native parent changed')
            fragments = {row['fragment_token']: row for row in marker['fragments']}
            require(all(row['split_unit_id'] == marker['image_name'] for row in fragments.values()),
                    'native family lineage was replaced by generated group identity')
            for candidate in marker['candidates']:
                if candidate['label'] is not True or candidate['main_training_eligible'] is not True:
                    continue
                row = model_row(candidate, fragments, role)
                try:
                    sample = load_native_pair(release, role, row, runtime)
                except runtime.RachelDatasetError as error:
                    rejected.append(dict(role=role, pair_id=candidate['pair_id'], reason=str(error),
                                         stage='unchanged_native_positive_target_residual'))
                    continue
                require(bool(sample.translation_valid), 'native positive lost pose supervision')
                group_record['admitted_positive_pairs'].append(row)
        else:
            rejected.append(dict(role=role, group_id=gid, reason=marker.get('error'), stage='native_group'))
        records.append(group_record)
    for path, sha in all_checks.items():
        checked_file(path, sha)
    checked_file(plan_file, expected_parent_sha)
    checked_file(groups_file, expected_groups_sha)
    native_modules(frozen_runtime_root, expected_sources)
    identities = {str(p.relative_to(output)): file_sha(p) for p in output.rglob('*') if p.is_file()}
    result = dict(schema=SCHEMA, status='preprocessed_not_mixed_selection_complete',
                  raw_generation_status=manifest['status'],
                  generation_plan_path=generation_plan_path,
                  generation_plan_sha256=expected_generation_plan_sha,
                  generation_binding=generation_binding,
                  parent_plan_sha256=expected_parent_sha, groups_manifest_sha256=expected_groups_sha,
                  original_preprocess_config=config, native_source_sha256=expected_sources,
                  groups=records, rejected_pairs_or_groups=rejected, output_files_sha256=identities,
                  no_train_or_test_split_created=True,
                  negative_catalog_policy='same generator, different approved family, within role; separately enumerate and validate')
    _json(output / 'base_release.json', result)
    return result


FRAGMENT_FIELDS = ('fragment_token', 'model_mask_path', 'contour_path', 'split_unit_id',
                   'foreground_area', 'bbox_aspect_ratio', 'parent_group_id', 'generator',
                   'image_name', 'fragment_id')


def merge_catalog(*, parent_plan_path, expected_parent_sha, existing_index_path,
                  expected_existing_sha, new_release_path, expected_new_release_sha,
                  output_new, frozen_runtime_root, expected_sources):
    """Copy a SHA-bound canonical release, retaining audit geometry separately.

    A native ``split=val`` carrier is required by historical v14 wrappers; it
    never determines CAL versus SELECT. That assignment is immutable external
    role metadata. No old source is modified, linked, renamed, or deleted.
    """
    plan_file = checked_file(parent_plan_path, expected_parent_sha)
    index_file = checked_file(existing_index_path, expected_existing_sha)
    new_file = checked_file(new_release_path, expected_new_release_sha)
    plan, index, new = [json.loads(p.read_text()) for p in (plan_file, index_file, new_file)]
    require(plan['existing_base_index_contract']['sha256'] == expected_existing_sha, 'old index not bound by parent plan')
    require(new['parent_plan_sha256'] == expected_parent_sha, 'new release parent plan mismatch')
    require(new['status'] == 'preprocessed_not_mixed_selection_complete', 'wrong native release stage')
    parents = approved_parents(plan)
    runtime = native_modules(frozen_runtime_root, expected_sources)['rachel_training_dataset']
    out = Path(output_new).absolute()
    require(not out.exists(), 'new canonical catalog output already exists')
    out.mkdir(parents=True)
    copied, input_sha, fragments, by_token, audits = {}, {}, [], {}, []

    def copy(path, relative, expected=None):
        path = Path(path).absolute()
        relative = Path(relative)
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe catalog path')
        require(relative.parts[:2] in (('model', 'masks_800'), ('model', 'contours_n512'),
                                        ('targets', 'parent_masks_800'), ('targets', 'pairs')),
                'unapproved artifact class')
        digest = file_sha(path) if expected is None else expected
        checked_file(path, digest)
        input_sha[str(path)] = digest
        target = out / relative
        if str(relative) in copied:
            require(copied[str(relative)] == digest and file_sha(target) == digest, 'catalog identity collision')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            require(file_sha(target) == digest, 'copy changed source bytes')
            copied[str(relative)] = digest
        return str(relative)

    def add_fragment(role, family, row, source_root, files=None):
        require(role in ROLES and family in parents[role], 'fragment outside approved family/role')
        require(row['fragment_token'] not in by_token, 'duplicate fragment token across sources')
        native = {key: row[key] for key in FRAGMENT_FIELDS if key in row}
        for key in ('model_mask_path', 'contour_path'):
            receipt = files[key] if files else None
            copy(Path(source_root) / row[key], row[key], receipt['sha256'] if receipt else None)
        audit = deepcopy(row.get('target_audit', {}))
        if audit:
            path = audit['parent_mask_path']
            copy(Path(source_root) / path, path, files['parent_mask_path']['sha256'] if files else None)
        record = dict(role=role, family=family, row=native)
        fragments.append(record)
        by_token[native['fragment_token']] = record
        audits.append(dict(role=role, family=family, fragment_token=native['fragment_token'], target_audit=audit))

    old_root = Path(index['release_root'])
    for token, record in index['fragments'].items():
        require(record['row']['fragment_token'] == token, 'old index token mismatch')
        add_fragment(record['role'], record['family'], record['row'], old_root, record['files'])
    for group in new['groups']:
        for row in group['marker']['fragments']:
            paths = {key:row[key] for key in ('model_mask_path','contour_path')}
            paths['parent_mask_path'] = row['target_audit']['parent_mask_path']
            files = {key:{'sha256':new['output_files_sha256'][group['role']+'/'+relative]}
                     for key,relative in paths.items()}
            add_fragment(group['role'], group['family'], row, new_file.parent / group['role'], files)
    positive, rejections = {role: [] for role in ROLES}, []

    def add_pair(role, candidate, source_root, expected_target_sha=None):
        a, b = by_token[candidate['fragment_a_token']], by_token[candidate['fragment_b_token']]
        require(a['role'] == b['role'] == role and a['family'] == b['family'], 'positive pair lineage mismatch')
        require(a['row']['generator'] == b['row']['generator'], 'positive generator mismatch')
        relative = candidate['correspondence_path']
        copy(Path(source_root) / relative, relative, expected_target_sha)
        row = dict(pair_id=candidate['pair_id'], label=True, split='val',
                   fragment_a=a['row'], fragment_b=b['row'], correspondence_path=relative,
                   label_origin=candidate.get('label_origin', 'native_csv_and_independent_contour'),
                   translation_a_to_b_rc=candidate.get('translation_a_to_b_rc'),
                   translation_a_to_b_xy_cartesian=candidate.get('translation_a_to_b_xy_cartesian'))
        # The exact native loader accepts CAL/SELECT directly; carry role here
        # during validation, then expose val solely to the frozen v14 wrapper.
        try:
            sample = load_native_pair(out, role, dict(row, split=role), runtime)
            require(int((sample.target_a >= 0).sum()) >= 4, 'native heldout positive has <4 matches')
        except (runtime.RachelDatasetError, ValueError) as error:
            rejections.append(dict(role=role, pair_id=row['pair_id'], reason=str(error)))
            return
        positive[role].append(row)

    for role, rows in index['eligible_positive_rows'].items():
        for candidate in rows:
            require(candidate['label'] is True, 'non-positive in old positive index')
            add_pair(role, candidate, old_root)
    for group in new['groups']:
        admitted = {row['pair_id'] for row in group['admitted_positive_pairs']}
        for candidate in group['marker']['candidates']:
            if candidate['pair_id'] in admitted:
                target_sha = new['output_files_sha256'][group['role']+'/'+candidate['correspondence_path']]
                add_pair(group['role'], candidate, new_file.parent / group['role'], target_sha)
    for path, sha in input_sha.items():
        checked_file(path, sha)
    checked_file(plan_file, expected_parent_sha)
    checked_file(index_file, expected_existing_sha)
    checked_file(new_file, expected_new_release_sha)
    native_modules(frozen_runtime_root, expected_sources)
    result = dict(schema='mixed-heldout-native-catalog/1', status='pilot_catalog_not_full_selection',
                  release_root=str(out), fragments=fragments, positive_rows=positive,
                  fragment_target_audit=audits, rejected_positive_rows=rejections,
                  parent_plan_sha256=expected_parent_sha, existing_index_sha256=expected_existing_sha,
                  new_release_sha256=expected_new_release_sha, native_source_sha256=expected_sources,
                  native_val_is_only_interface_carrier=True, role_assignment='external immutable CAL/SELECT',
                  copied_files_sha256=copied, original_files_sha256=input_sha)
    _json(out / 'catalog.json', result)
    return result


def extend_verified_catalog(*, parent_plan_path, expected_parent_sha,
                            previous_parent_plan_path, expected_previous_parent_sha,
                            verified_catalog_path, expected_catalog_sha,
                            verification_path, expected_verification_sha,
                            new_release_path, expected_new_release_sha,
                            output_new, frozen_runtime_root, expected_sources):
    """Append new native groups without re-running already verified old pairs.

    Previously admitted artifacts are copied after exact byte verification.
    Their old native checks remain attached through the frozen verification
    receipt. Only the genuinely new release's positives invoke the native
    target loader. No prior catalog or parent plan is modified.
    """
    paths = [checked_file(path, sha) for path, sha in (
        (parent_plan_path, expected_parent_sha),
        (previous_parent_plan_path, expected_previous_parent_sha),
        (verified_catalog_path, expected_catalog_sha),
        (verification_path, expected_verification_sha),
        (new_release_path, expected_new_release_sha))]
    plan, previous_plan, previous, verification, new = [json.loads(p.read_text()) for p in paths]
    parents = approved_parents(plan)
    old_parents = approved_parents(previous_plan)
    for role, families in old_parents.items():
        require(all(parents[role].get(family) == row for family, row in families.items()),
                'previous parent identity or fold changed')
    require(plan['exclusion_families'] == previous_plan['exclusion_families'] and
            plan['hard_exclusions'] == previous_plan['hard_exclusions'], 'exclusions changed')
    require(plan['evidence']['previous_parent_plan']['sha256'] == expected_previous_parent_sha,
            'combined plan does not bind previous parent plan')
    require(previous['parent_plan_sha256'] == expected_previous_parent_sha and
            new['parent_plan_sha256'] == expected_parent_sha, 'release parent plan mismatch')
    require(previous['schema'] == 'mixed-heldout-native-catalog/1' and
            new['status'] == 'preprocessed_not_mixed_selection_complete', 'wrong input release stage')
    require(Path(verification['catalog']['path']).resolve(strict=True) == paths[2] and
            verification['catalog']['sha256'] == expected_catalog_sha,
            'verification receipt does not bind actual prior catalog')
    require(verification['all_positive_rows'] == {r:len(v) for r,v in previous['positive_rows'].items()},
            'prior positive counts differ from native verification')
    require(verification['native_source_sha256'] == previous['native_source_sha256'] ==
            new['native_source_sha256'] == expected_sources, 'native source changed')
    require(verification['native_rejections'] == [] and verification['catalog_rejections'] == [] and
            verification['source_files_unchanged'] is True, 'prior native verification not clean')
    previous_base_ref = verification['base_release']
    previous_base_file = checked_file(previous_base_ref['path'],previous_base_ref['sha256'])
    previous_base = json.loads(previous_base_file.read_text())
    require(previous_base['parent_plan_sha256'] == expected_previous_parent_sha and
            previous['new_release_sha256'] == previous_base_ref['sha256'], 'prior base provenance mismatch')
    runtime = native_modules(frozen_runtime_root, expected_sources)['rachel_training_dataset']
    out = Path(output_new).absolute()
    require(not out.exists(), 'new canonical catalog output already exists')
    old_root, new_root = Path(previous['release_root']), paths[4].parent
    require(not any(out == p or out in p.parents or p in out.parents for p in (*paths, old_root, new_root)),
            'new output overlaps protected input')
    out.mkdir(parents=True)
    copied, input_sha = {}, {}

    def copy_bound(path, relative, digest):
        relative = Path(relative)
        require(not relative.is_absolute() and '..' not in relative.parts and
                relative.parts[:2] in (('model','masks_800'),('model','contours_n512'),
                                        ('targets','parent_masks_800'),('targets','pairs')),
                'unsafe canonical artifact path')
        source = checked_file(path, digest)
        input_sha[str(source)] = digest
        target = out / relative
        require(str(relative) not in copied, 'new/old canonical artifact collision')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(file_sha(target) == digest, 'canonical copy changed bytes')
        copied[str(relative)] = digest

    for relative, digest in previous['copied_files_sha256'].items():
        copy_bound(old_root / relative, relative, digest)
    fragments = deepcopy(previous['fragments'])
    audits = deepcopy(previous['fragment_target_audit'])
    positive = deepcopy(previous['positive_rows'])
    by_token = {}
    for record in fragments:
        role, family, row = record['role'], record['family'], record['row']
        require(family in parents[role] and set(row) <= set(FRAGMENT_FIELDS), 'unsafe prior fragment')
        require(row['fragment_token'] not in by_token, 'duplicate prior fragment')
        require(all(row[k] in copied for k in ('model_mask_path','contour_path')), 'prior file not bound')
        by_token[row['fragment_token']] = record
    previous_positive_counts = {r:len(v) for r,v in positive.items()}
    pair_ids = {r['pair_id'] for rows in positive.values() for r in rows}
    require(len(pair_ids) == sum(previous_positive_counts.values()), 'duplicate prior pair ID')
    new_positive_counts = {role:0 for role in ROLES}
    rejections = []
    for group in new['groups']:
        if group['marker']['status'] != 'processed_group':
            continue
        role, family = group['role'], group['family']
        require(family in parents[role], 'new group outside approved fold')
        for row in group['marker']['fragments']:
            token = row['fragment_token']
            require(token not in by_token, 'new fragment collides with previously verified source')
            require(row['split_unit_id'] == group['marker']['image_name'] ==
                    Path(parents[role][family]['parent_image']['path']).name,
                    'native split unit is not the bound original parent')
            native = {key:row[key] for key in FRAGMENT_FIELDS if key in row}
            for relative in (row['model_mask_path'],row['contour_path'],row['target_audit']['parent_mask_path']):
                copy_bound(new_root/role/relative, relative, new['output_files_sha256'][role+'/'+relative])
            record = dict(role=role,family=family,row=native)
            fragments.append(record); by_token[token] = record
            audits.append(dict(role=role,family=family,fragment_token=token,target_audit=deepcopy(row['target_audit'])))
        admitted = {row['pair_id'] for row in group['admitted_positive_pairs']}
        for candidate in group['marker']['candidates']:
            if candidate['pair_id'] not in admitted:
                continue
            require(candidate['pair_id'] not in pair_ids, 'new positive collides with old positive')
            a, b = by_token[candidate['fragment_a_token']], by_token[candidate['fragment_b_token']]
            require(a['role'] == b['role'] == role and a['family'] == b['family'] == family and
                    a['row']['generator'] == b['row']['generator'], 'new positive lineage changed')
            relative = candidate['correspondence_path']
            copy_bound(new_root/role/relative, relative, new['output_files_sha256'][role+'/'+relative])
            row = dict(pair_id=candidate['pair_id'],label=True,split='val',fragment_a=a['row'],fragment_b=b['row'],
                       correspondence_path=relative,label_origin=candidate.get('label_origin','native_csv_and_independent_contour'),
                       translation_a_to_b_rc=candidate.get('translation_a_to_b_rc'),
                       translation_a_to_b_xy_cartesian=candidate.get('translation_a_to_b_xy_cartesian'))
            try:
                sample = load_native_pair(out, role, dict(row,split=role), runtime)
                require(bool(sample.translation_valid) and int((sample.target_a>=0).sum())>=4,
                        'new native positive lost valid pose/matches')
            except (runtime.RachelDatasetError,ValueError) as error:
                rejections.append(dict(role=role,pair_id=row['pair_id'],reason=str(error)))
                continue
            positive[role].append(row); pair_ids.add(row['pair_id']); new_positive_counts[role] += 1
    for path, digest in input_sha.items():
        checked_file(path,digest)
    for path, digest in zip(paths,(expected_parent_sha,expected_previous_parent_sha,expected_catalog_sha,
                                   expected_verification_sha,expected_new_release_sha)):
        checked_file(path,digest)
    native_modules(frozen_runtime_root,expected_sources)
    provenance, base_refs = {}, []
    for base, base_file, base_sha in ((previous_base,previous_base_file,previous_base_ref['sha256']),
                                     (new,paths[4],expected_new_release_sha)):
        reference = dict(path=str(base_file),sha256=base_sha)
        base_refs.append(reference)
        generation_path = Path(base['generation_plan_path']) if base.get('generation_plan_path') else None
        raw_ref = dict(path=str(generation_path.parent/'complete.json'),sha256=base['groups_manifest_sha256']) if generation_path else None
        if raw_ref:
            checked_file(raw_ref['path'],raw_ref['sha256'])
        for i,group in enumerate(base['groups']):
            if group['marker']['status'] != 'processed_group':
                continue
            ids = {f['parent_group_id'] for f in group['marker']['fragments']}
            require(len(ids) == 1, 'native parent group is ambiguous')
            group_id = next(iter(ids))
            require(group_id not in provenance, 'raw base provenance collision')
            provenance[group_id] = dict(role=group['role'],family=group['family'],generator=group['generator'],
                raw_group_id=group['group_id'],base_release_ref=reference,base_release_group_index=i,
                raw_complete_ref=raw_ref,generation_plan_ref=dict(path=str(generation_path),sha256=base['generation_plan_sha256']) if generation_path else None,
                raw_source_pointer='groups/'+str(i)+'/raw_source',
                inherited_parent_boundary_policy=group['raw_source'].get('source_derivation'))
    result = dict(schema='mixed-heldout-native-catalog/1',status='native_catalog_pending_stage_augmentation',
                  release_root=str(out),fragments=fragments,positive_rows=positive,fragment_target_audit=audits,
                  rejected_positive_rows=rejections,parent_plan_sha256=expected_parent_sha,
                  previous_parent_plan_sha256=expected_previous_parent_sha,
                  previous_catalog=dict(path=str(paths[2]),sha256=expected_catalog_sha),
                  previous_native_verification=dict(path=str(paths[3]),sha256=expected_verification_sha),
                  base_release_refs=base_refs,base_provenance_by_parent_group_id=provenance,
                  previous_pairs_reused_without_inference_or_native_retest=previous_positive_counts,
                  new_positive_pairs_native_checked=new_positive_counts,
                  existing_index_sha256=previous['existing_index_sha256'],new_release_sha256=expected_new_release_sha,
                  native_source_sha256=expected_sources,native_val_is_only_interface_carrier=True,
                  role_assignment='external immutable CAL/SELECT',copied_files_sha256=copied,
                  original_files_sha256=input_sha,complete_mixed_select_cal=False)
    _json(out/'catalog.json',result)
    return result

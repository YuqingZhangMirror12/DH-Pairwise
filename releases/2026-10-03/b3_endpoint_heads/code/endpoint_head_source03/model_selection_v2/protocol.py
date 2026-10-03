"""Freeze explicit mixed simulation selection identities before predictions exist.

This standard-library module does not generate samples, infer provenance from
names, verify sample tensors, or launch inference. Callers must materialize the
normalized lineage manifests from authoritative records (including donors).
The exact manifest bytes and their declared sample hashes are bound here.
"""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path


MANIFEST_SCHEMA = 'model-selection-lineage-manifest/2'
PLAN_SCHEMA = 'model-selection-frozen-protocol/2'
RULE_ID = 'stratified_native_layout_band_loss/2'
SELECTION_KIND = 'mixed_sim_v2_best'
LOSS_SEMANTICS = 'per_pair_full_matcher_loss/1'
STAGE_WEIGHTS = {'v17_filtered': .3, 'v17.5': .3, 'v18': .3, 'strict_straight': .1}
LAYOUT_TOLERANCE = .005
GENERATORS = ('Gen2', 'Gen3', 'Gen4', 'Gen5')
DEFAULT_GENERATORS = {stage: (['straight_strip'] if stage == 'strict_straight' else list(GENERATORS))
                      for stage in STAGE_WEIGHTS}
ROLES = ('train', 'test', 'cal', 'select')
IDENTITIES = ('parent_ids', 'base_pair_ids', 'fragment_ids')
DONORS = tuple('donor_' + name for name in IDENTITIES)
ROW_FIELDS = {'pair_id', 'sample_sha256', 'stage', 'generator', 'label', *IDENTITIES, *DONORS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _strings(values, name, nonempty=False):
    require(isinstance(values, list) and (values or not nonempty)
            and all(isinstance(s, str) and s.strip() == s and bool(s) for s in values)
            and len(values) == len(set(values)), 'unique explicit identities required: ' + name)
    return set(values)


def _validate_manifest(manifest):
    require(isinstance(manifest, dict) and set(manifest) == {'schema', 'role', 'entries'}
            and manifest['schema'] == MANIFEST_SCHEMA and manifest['role'] in ROLES,
            'registered simulation lineage manifest/role required; real roles are forbidden')
    entries = manifest['entries']
    require(isinstance(entries, list) and bool(entries), 'empty lineage manifest')
    pairs = set()
    for row in entries:
        require(isinstance(row, dict) and set(row) == ROW_FIELDS,
                'explicit stage/generator/label/source and donor identities required')
        for key in ('pair_id', 'stage', 'generator'):
            require(isinstance(row[key], str) and row[key].strip() == row[key] and bool(row[key]),
                    'nonempty explicit ' + key + ' required')
        require(row['pair_id'] not in pairs, 'duplicate manifest pair_id')
        pairs.add(row['pair_id'])
        require(is_sha(row['sample_sha256']) and type(row['label']) is bool,
                'exact sample hash and Boolean label required')
        for key in IDENTITIES:
            _strings(row[key], key, nonempty=True)
        for key in DONORS:
            _strings(row[key], key)
        # A donor may coincide with a primary source. Its union still participates
        # in every exclusion check; no assumption about names is made.
    return entries


def bind_manifest(path, expected_sha256):
    """Read exactly one caller-named normalized manifest and bind its raw bytes."""
    require(is_sha(expected_sha256), 'expected manifest SHA256 required')
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'manifest file SHA256 differs')
    manifest = json.loads(raw)
    entries = _validate_manifest(manifest)
    return canonical(dict(path=str(path), sha256=expected_sha256,
        content_sha256=digest(manifest), pair_count=len(entries), manifest=manifest))


def _validate_binding(binding, role, verify_files=False):
    require(isinstance(binding, dict) and set(binding) == {
        'path', 'sha256', 'content_sha256', 'pair_count', 'manifest'}, 'complete manifest binding required')
    require(isinstance(binding['path'], str) and Path(binding['path']).is_absolute()
            and is_sha(binding['sha256']) and is_sha(binding['content_sha256']),
            'absolute manifest path and file/content hashes required')
    entries = _validate_manifest(binding['manifest'])
    require(binding['manifest']['role'] == role, 'phase/role differs from frozen manifest slot')
    require(type(binding['pair_count']) is int and binding['pair_count'] == len(entries)
            and digest(binding['manifest']) == binding['content_sha256'],
            'manifest row count or content hash differs')
    if verify_files:
        require(bind_manifest(binding['path'], binding['sha256']) == binding, 'bound manifest changed')
    return entries


def _lineages(rows):
    result = {key: set() for key in IDENTITIES}
    for row in rows:
        for key in IDENTITIES:
            result[key].update(row[key]); result[key].update(row['donor_' + key])
    return result


def _audit(bindings, required_generators, verify_files=False):
    require(isinstance(bindings, dict) and set(bindings) == set(ROLES),
            'complete TRAIN/TEST exclusion catalogs plus CAL/SELECT required')
    require(isinstance(required_generators, dict) and set(required_generators) == set(STAGE_WEIGHTS),
            'every registered stage needs a frozen explicit generator inventory')
    for stage, generators in required_generators.items():
        _strings(generators, stage + ' generators', nonempty=True)
        require(set(generators) == set(DEFAULT_GENERATORS[stage]),
                'curriculum stages require Gen2--Gen5; strict_straight requires straight_strip; missing generators cannot be relaxed')
    rows = {role: _validate_binding(bindings[role], role, verify_files) for role in ROLES}
    identities = {role: _lineages(values) for role, values in rows.items()}
    pair_ids = {role: {r['pair_id'] for r in values} for role, values in rows.items()}
    sample_hashes = {role: {r['sample_sha256'] for r in values} for role, values in rows.items()}
    # We audit the new decision folds against one another and both exclusion
    # catalogs, not re-certify the historical TRAIN versus TEST split itself.
    for left, right in itertools.combinations(ROLES, 2):
        if {left, right} == {'train', 'test'}:
            continue
        require(not pair_ids[left] & pair_ids[right], 'pair_id leakage: ' + left + '/' + right)
        require(not sample_hashes[left] & sample_hashes[right],
                'sample_sha256 leakage despite declared identities: ' + left + '/' + right)
        for key in IDENTITIES:
            require(not identities[left][key] & identities[right][key],
                    'primary/donor ' + key + ' leakage: ' + left + '/' + right)
    reuse = {}
    for role in ('cal', 'select'):
        actual = {(r['stage'], r['generator']) for r in rows[role]}
        expected = {(s, g) for s, gs in required_generators.items() for g in gs}
        require(actual == expected, 'missing or extra required stage/generator stratum: ' + role)
        for stage, generator in sorted(expected):
            labels = {r['label'] for r in rows[role] if (r['stage'], r['generator']) == (stage, generator)}
            require(labels == {False, True}, 'each stage/generator stratum requires both labels: ' + role)
        counts = Counter(identity for row in rows[role] for identity in row['base_pair_ids'])
        def population(items):
            all_sources = _lineages(items)
            return dict(rows=len(items), positives=sum(r['label'] for r in items),
                negatives=sum(not r['label'] for r in items),
                primary_parent_count=len({v for r in items for v in r['parent_ids']}),
                primary_base_pair_count=len({v for r in items for v in r['base_pair_ids']}),
                primary_fragment_count=len({v for r in items for v in r['fragment_ids']}),
                parent_count_including_donors=len(all_sources['parent_ids']),
                base_pair_count_including_donors=len(all_sources['base_pair_ids']),
                fragment_count_including_donors=len(all_sources['fragment_ids']))
        strata = {stage: {generator: population([r for r in rows[role]
                    if (r['stage'], r['generator']) == (stage, generator)])
                    for generator in required_generators[stage]} for stage in STAGE_WEIGHTS}
        reuse[role] = dict(**population(rows[role]), distinct_base_pairs=len(counts), strata=strata,
            base_pairs_reused=sum(n > 1 for n in counts.values()),
            extra_base_pair_appearances=sum(n - 1 for n in counts.values()),
            maximum_base_pair_multiplicity=max(counts.values()),
            interpretation='same-fold augmentation reuse allowed; rows are not independent base pairs')
    return reuse


def freeze_protocol(bindings, candidate_updates, required_generators=None):
    """Seal policy, candidate list, data/exclusions before accepting any reports.

    The three curriculum stages require Gen2--Gen5. Native strict-straight
    strips have their own straight_strip identity, never a fictitious Gen ID.
    Inventories cannot be inferred or relaxed by later predictions.
    """
    require(isinstance(candidate_updates, list) and bool(candidate_updates)
            and all(type(n) is int and n > 0 for n in candidate_updates)
            and candidate_updates == sorted(set(candidate_updates)),
            'sorted unique positive trained candidate updates required')
    generators = required_generators if required_generators is not None else DEFAULT_GENERATORS
    reuse = _audit(bindings, generators, verify_files=True)
    plan = canonical(dict(schema=PLAN_SCHEMA, state='frozen_before_reports', reports_seen_at_freeze=0,
        rule=dict(id=RULE_ID, selection_kind=SELECTION_KIND, stage_weights=STAGE_WEIGHTS,
            within_stage='equal_generator_macro', layout_population='positive_pairs',
            loss_population='all_pairs', loss_semantics=LOSS_SEMANTICS,
            absolute_layout_tolerance=LAYOUT_TOLERANCE,
            band_rule='macro_layout >= maximum_macro_layout - absolute_layout_tolerance',
            within_band_order=['minimum_macro_loss', 'maximum_macro_layout', 'earliest_update'],
            coverage_affects_selection=False), candidate_updates=candidate_updates,
        manifest_bindings=bindings, required_generators=generators,
        same_fold_base_pair_reuse=reuse, matcher_selection_role='select',
        calibration_role='cal', real_used=False, test_used_for_selection=False,
        train_used_for_selection=False, test_and_train_read_for_exclusion_only=True))
    plan['sha256'] = digest(plan)
    return plan


def validate_protocol(plan, verify_files=False):
    if isinstance(plan, dict) and plan.get('schema') == 'model-selection-published-release-protocol/1':
        from .released_protocol import validate_release_protocol
        return validate_release_protocol(plan, verify_files=verify_files)
    require(isinstance(plan, dict) and is_sha(plan.get('sha256')),
            'sealed protocol SHA256 required')
    body = {k: v for k, v in plan.items() if k != 'sha256'}
    require(digest(body) == plan['sha256'], 'frozen protocol digest differs')
    require(set(body) == {'schema', 'state', 'reports_seen_at_freeze', 'rule', 'candidate_updates',
            'manifest_bindings', 'required_generators', 'same_fold_base_pair_reuse',
            'matcher_selection_role', 'calibration_role', 'real_used', 'test_used_for_selection',
            'train_used_for_selection', 'test_and_train_read_for_exclusion_only'},
            'unexpected frozen protocol fields')
    require(body.get('schema') == PLAN_SCHEMA and body.get('state') == 'frozen_before_reports'
            and type(body.get('reports_seen_at_freeze')) is int and body['reports_seen_at_freeze'] == 0
            and body.get('matcher_selection_role') == 'select'
            and body.get('calibration_role') == 'cal' and body.get('real_used') is False
            and body.get('test_used_for_selection') is False and body.get('train_used_for_selection') is False
            and body.get('test_and_train_read_for_exclusion_only') is True,
            'decision role or freeze boundary differs')
    rule = body.get('rule', {})
    expected_rule = dict(id=RULE_ID, selection_kind=SELECTION_KIND, stage_weights=STAGE_WEIGHTS,
        within_stage='equal_generator_macro', layout_population='positive_pairs',
        loss_population='all_pairs', loss_semantics=LOSS_SEMANTICS,
        absolute_layout_tolerance=LAYOUT_TOLERANCE,
        band_rule='macro_layout >= maximum_macro_layout - absolute_layout_tolerance',
        within_band_order=['minimum_macro_loss', 'maximum_macro_layout', 'earliest_update'],
        coverage_affects_selection=False)
    require(rule == expected_rule, 'unregistered or changed selection rule')
    updates = body.get('candidate_updates')
    require(isinstance(updates, list) and bool(updates) and all(type(n) is int and n > 0 for n in updates)
            and updates == sorted(set(updates)), 'candidate update inventory differs')
    audit = _audit(body['manifest_bindings'], body['required_generators'], verify_files)
    require(body['same_fold_base_pair_reuse'] == audit, 'same-fold augmentation reuse disclosure differs')
    return plan

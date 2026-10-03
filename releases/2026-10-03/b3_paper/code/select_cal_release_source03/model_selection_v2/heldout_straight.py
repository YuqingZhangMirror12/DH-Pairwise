"""Isolated CPU-only straight-seam materialization for new CAL/SELECT.

This orchestration retains frozen v4.2 pixels and corrected supervision. It
never admits training data, selects a checkpoint, or claims an independent
pixel/target audit. All imports and the added CAL RNG domain are private to
this instance; the historical generator files and outputs stay unchanged.
"""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading
import types
import uuid


SCHEMA = 'mixed-heldout-straight-plan/1'
COUNTS = {'M': 40, 'J': 75, 'R': 45}
ROLE_CODES = {'select': 2, 'cal': 4}
EXPECTED_ROWS = 320
REFERENCE_NAMES = ('gen_straight_seam', 'gen_straight_seam_v3', 'gen_straight_seam_v4')
WRAPPER_NAMES = ('generate', 'supervision', 'full_generate', 'geometry_contract', 'pair_identity', 'repair_geometry')
EXCLUSION_KEYS = ('all_train_including_donors', 'all_original_and_published_test_including_donors')
OLD_SEEDS = {20260929, 26093004, 26093015, 26093016, 26093084, 26093085, 26093086, 26093087,
             26093094, 26093095, 26093096, 26093097}
_IMPORT_LOCK = threading.RLock()


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _receipt(path, expected):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and file_sha(path) == expected, 'bound file changed: ' + str(path))
    return {'path': str(path), 'sha256': expected}


def _read(receipt):
    _receipt(receipt['path'], receipt['sha256'])
    return json.loads(Path(receipt['path']).read_text())


def _bound_index(parents, receipt):
    admitted = parents.get('evidence', {}).get('existing_mask_source_index', {})
    # The immutable admitted bytes may be transported to a new host/path.
    require(admitted.get('sha256') == receipt['sha256'] and isinstance(admitted.get('path'), str),
            'source index must be the exact receipt admitted by parent plan')


def _families(parent_plan, role):
    require(parent_plan.get('schema') == 'task1-evaluation-only-parent-plan/1', 'wrong parent plan schema')
    folds = parent_plan.get('folds', {})
    require(set(folds) == set(ROLE_CODES), 'parent plan must contain only CAL and SELECT')
    exclusions = parent_plan.get('exclusion_families')
    require(isinstance(exclusions, dict) and all(k in exclusions for k in EXCLUSION_KEYS),
            'explicit TRAIN/TEST donor-family exclusions required')
    blocked = set(parent_plan.get('hard_exclusions', {}))
    for key in EXCLUSION_KEYS:
        require(isinstance(exclusions[key], list) and all(isinstance(x, str) for x in exclusions[key]),
                'malformed exclusion family list')
        blocked.update(exclusions[key])
    role_sets = {}
    for name, rows in folds.items():
        require(isinstance(rows, list) and rows, 'empty parent role')
        families = [r.get('family') for r in rows]
        require(all(isinstance(f, str) and f for f in families) and len(set(families)) == len(families),
                'duplicate or missing parent family')
        require(all(r.get('role') == name and r.get('edge_donors_must_be_from_same_role_families') is True
                    for r in rows), 'role/donor boundary missing')
        require(not set(families) & blocked, 'TRAIN/TEST or hard-excluded parent family')
        role_sets[name] = set(families)
    require(not role_sets['cal'] & role_sets['select'], 'CAL/SELECT parent overlap')
    return role_sets[role]


def _sources(parent_plan, source_index, role, verify_files):
    families = _families(parent_plan, role)
    entries = source_index.get('entries')
    require(isinstance(entries, list) and entries, 'bound source index requires entries')
    rows, seen, identities, sha_roles = [], set(), set(), {}
    for row in entries:
        require(isinstance(row, dict) and row.get('role') in ROLE_CODES, 'source index role must be CAL/SELECT')
        own_families = _families(parent_plan, row['role'])
        family = row.get('source_family')
        require(family in own_families, 'source family outside its role parent allowlist')
        path, sha, token = row.get('path'), row.get('file_sha256'), row.get('fragment_token')
        require(isinstance(path, str) and Path(path).is_absolute(), 'absolute admitted source path required')
        require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha), 'source SHA256 required')
        require(isinstance(token, str) and token, 'source fragment identity required')
        require(path not in seen and token not in identities, 'duplicate source path/fragment identity')
        seen.add(path); identities.add(token)
        require(sha not in sha_roles or sha_roles[sha] == row['role'], 'identical source bytes cross CAL/SELECT')
        sha_roles[sha] = row['role']
        if row['role'] == role:
            require(family in families, 'foreign source family')
            if verify_files:
                _receipt(path, sha)
            rows.append(copy.deepcopy(row))
    require(rows and len({r['source_family'] for r in rows}) >= 2,
            'at least two distinct same-role material families required for negatives')
    return sorted(rows, key=lambda r: (r['source_family'], r['fragment_token'], r['path']))


def _real_exclusion(receipt, index_ref, rows):
    value = _read(receipt)
    require(value.get('status') == 'passed_source_exclusion' and value.get('source_index_sha256') == index_ref['sha256'],
            'exact new source population real-exclusion receipt required')
    require(value.get('potential_turufan_parent_aliases') == [] and value.get('exact_prepared_mask_matches') == []
            and value.get('real_scores_read') is False, 'real alias/mask hit or non-provenance real-data access')
    checked = value.get('checked_sources', {})
    require(all(checked.get(r['fragment_token']) == r['file_sha256'] for r in rows), 'real exclusion did not cover every source')


def freeze_plan(*, parent_plan_path, expected_parent_sha, source_index_path, expected_index_sha,
                role, dataset_id, masterseeds, output_new, frozen_runtime_dir,
                reference_dir, preprocess_path, metrics_path, real_exclusion_path, expected_real_exclusion_sha):
    """Bind role, source list, seeds, code and exact new review quotas before generation."""
    require(role in ROLE_CODES, 'only CAL/SELECT may be materialized')
    require(isinstance(dataset_id, str) and re.fullmatch('[A-Za-z0-9_-]+', dataset_id), 'safe explicit dataset_id required')
    require(set(masterseeds) == set(ROLE_CODES), 'both role seeds must be declared together')
    require(all(type(s) is int and 0 < s < 2**32 and s not in OLD_SEEDS for s in masterseeds.values())
            and len(set(masterseeds.values())) == 2, 'independent new role masterseeds required')
    parent_ref = _receipt(parent_plan_path, expected_parent_sha)
    index_ref = _receipt(source_index_path, expected_index_sha)
    parents, index = _read(parent_ref), _read(index_ref)
    _bound_index(parents, index_ref)
    rows = _sources(parents, index, role, True)
    real_ref = _receipt(real_exclusion_path, expected_real_exclusion_sha)
    _real_exclusion(real_ref, index_ref, rows)
    output = Path(output_new).resolve()
    require(not output.exists(), 'new output directory required; never resume/overwrite historical output')
    require(role in output.parts, 'output path must include explicit role component')
    require(dataset_id in output.parts, 'output must use the new dataset namespace')
    files = {}
    for label, root, names in (('wrapper', frozen_runtime_dir, WRAPPER_NAMES),
                                ('reference', reference_dir, REFERENCE_NAMES)):
        for name in names:
            path = Path(root).resolve(strict=True) / (name + '.py')
            files[label + '/' + name + '.py'] = _receipt(path, file_sha(path))
    files['preprocess'] = _receipt(preprocess_path, file_sha(preprocess_path))
    files['metrics'] = _receipt(metrics_path, file_sha(metrics_path))
    files['external_wrapper'] = _receipt(__file__, file_sha(__file__))
    return dict(schema=SCHEMA, dataset_id=dataset_id, role=role, stage='strict_straight',
                generator='straight_strip', counts_per_label=dict(COUNTS), expected_rows=EXPECTED_ROWS,
                j_counts_per_label={'torn_rachel': 60, 'margin_fragment': 9, 'torn_strip': 6},
                quota_policy='new review M40/J75/R45; not historical M/J/R proportions',
                masterseeds=dict(masterseeds), rng_role_codes=dict(ROLE_CODES), output_new=str(output),
                parent_plan=parent_ref, source_index=index_ref, real_source_exclusion=real_ref, source_rows=rows, code=files,
                donor_policy='all attempted and accepted material donors restricted to same-role parent plan',
                retries='frozen up to300 pair attempts, same base; redraws source/shape/damage; no extra retries',
                labels_algorithm_modified=False, training_admitted=False, gpu_used=False,
                model_inference=False, checkpoint_selected=False, head_training_started=False)


def validate_plan(plan, *, verify_files=True):
    require(plan.get('schema') == SCHEMA and plan.get('role') in ROLE_CODES, 'wrong heldout straight plan')
    require(plan.get('counts_per_label') == COUNTS and plan.get('expected_rows') == EXPECTED_ROWS,
            'frozen exact quota mismatch')
    require(plan.get('j_counts_per_label') == {'torn_rachel': 60, 'margin_fragment': 9, 'torn_strip': 6}, 'J quotas changed')
    require(plan.get('rng_role_codes') == ROLE_CODES, 'RNG role domain mismatch')
    seeds = plan.get('masterseeds', {})
    require(set(seeds) == set(ROLE_CODES) and all(type(x) is int and 0 < x < 2**32 and x not in OLD_SEEDS for x in seeds.values())
            and len(set(seeds.values())) == 2, 'role masterseeds invalid')
    for key in ('labels_algorithm_modified', 'training_admitted', 'gpu_used', 'model_inference',
                'checkpoint_selected', 'head_training_started'):
        require(plan.get(key) is False, 'out-of-scope plan flag: ' + key)
    require(plan.get('stage') == 'strict_straight' and plan.get('generator') == 'straight_strip', 'do not mislabel strict generator as Gen4/5')
    parents, index = _read(plan['parent_plan']), _read(plan['source_index'])
    _bound_index(parents, plan['source_index'])
    require(_sources(parents, index, plan['role'], verify_files) == plan.get('source_rows'), 'source population changed')
    _real_exclusion(plan['real_source_exclusion'], plan['source_index'], plan['source_rows'])
    keys = {'preprocess', 'metrics', 'external_wrapper'} | {f'wrapper/{n}.py' for n in WRAPPER_NAMES} | {f'reference/{n}.py' for n in REFERENCE_NAMES}
    require(set(plan.get('code', {})) == keys, 'complete frozen source binding required')
    if verify_files:
        for receipt in plan['code'].values():
            _receipt(receipt['path'], receipt['sha256'])
    output = Path(plan['output_new'])
    require(output.is_absolute() and plan['role'] in output.parts, 'explicit absolute role output required')
    require(plan.get('dataset_id') in output.parts, 'new dataset namespace missing from output')
    return plan


def jobs(plan):
    validate_plan(plan)
    return [(kind, label, i, plan['role'], plan['output_new'], plan['masterseeds'][plan['role']])
            for kind, count in COUNTS.items() for label in (1, 0) for i in range(count)]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FrozenBackend:
    """Load bound source into fresh modules; never mutate shared frozen imports.

    The caller must run one backend per worker/process. Temporary aliases only
    satisfy the reference's absolute imports, and are restored immediately.
    """
    def __init__(self, plan, *, audit_only=False):
        validate_plan(plan)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicitly disable CUDA for data construction')
        self.plan = plan
        namespace = '_heldout_straight_' + uuid.uuid4().hex
        pkg = types.ModuleType(namespace)
        pkg.__path__ = []
        sys.modules[namespace] = pkg
        self.private_modules = [namespace]
        aliases = list(REFERENCE_NAMES) + ['staging.pairwise_v0_2.pairwise_data.rachel_preprocess']
        with _IMPORT_LOCK:
            previous = {name: sys.modules.get(name) for name in aliases}
            previous_path = sys.path[:]
            try:
                pre = _load(namespace + '.preprocess', plan['code']['preprocess']['path'])
                self.private_modules.append(pre.__name__)
                sys.modules[aliases[-1]] = pre
                refs = {}
                for name in REFERENCE_NAMES:
                    refs[name] = _load(namespace + '.' + name, plan['code']['reference/' + name + '.py']['path'])
                    self.private_modules.append(refs[name].__name__)
                    sys.modules[name] = refs[name]
                ref = refs['gen_straight_seam_v4']
                require(ref.GEN_REV == 'v4.2' and ref.cut is refs['gen_straight_seam_v3'].cut,
                        'frozen v4.2 geometry identity mismatch')
                require(ref.SPLIT_SEED == {'train': 1, 'select': 2, 'test': 3}, 'unexpected reference split domains')
                # Modify only this private instance. Historical module/file is untouched.
                ref.SPLIT_SEED['cal'] = ROLE_CODES['cal']
                g = _load(namespace + '.generate', plan['code']['wrapper/generate.py']['path'])
                s = _load(namespace + '.supervision', plan['code']['wrapper/supervision.py']['path'])
                full = _load(namespace + '.full_generate', plan['code']['wrapper/full_generate.py']['path'])
                self.private_modules.extend([g.__name__, s.__name__, full.__name__])
                geometry = _load(namespace + '.geometry_contract', plan['code']['wrapper/geometry_contract.py']['path'])
                identity = _load(namespace + '.pair_identity', plan['code']['wrapper/pair_identity.py']['path'])
                repair = _load(namespace + '.repair_geometry', plan['code']['wrapper/repair_geometry.py']['path'])
                metrics = _load(namespace + '.metrics', plan['code']['metrics']['path'])
                self.private_modules.extend([geometry.__name__, identity.__name__, repair.__name__, metrics.__name__])
                g.load_reference = lambda directory: (ref, {n: plan['code']['reference/' + n + '.py']['sha256'] for n in REFERENCE_NAMES})
                repair.metric_module = lambda path: metrics
                rows = {r['path']: copy.deepcopy(r) for r in plan['source_rows']}
                directory = str(Path(plan['code']['reference/gen_straight_seam.py']['path']).parent)
                if audit_only:
                    g.initialize(directory, rows)  # No corrected-target/generation finalizer.
                else:
                    repair.initialize(directory, rows, dict(COUNTS), plan['code']['metrics']['path'])
                self.full, self.reference, self.context = full, ref, g.CONTEXT
                self.geometry, self.identity, self.metrics, self.supervision = geometry, identity, metrics, s
            finally:
                sys.path[:] = previous_path
                for name, old in previous.items():
                    if old is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = old
        old_save = self.reference.save_sample

        def guarded_save(path, *args, **kwargs):
            target = Path(path).resolve()
            root = Path(plan['output_new']).resolve()
            require(root in target.parents and target.parent == root / 'samples' and not target.exists(),
                    'write outside new output or duplicate sample')
            return old_save(path, *args, **kwargs)
        self.reference.save_sample = guarded_save

    def generate_one(self, job):
        self.context['geometry_attempts'] = []
        entry = self.full.generate_one(job)
        entry['geometry_attempts'] = copy.deepcopy(self.context['geometry_attempts'])
        entry['geometry_contract'] = copy.deepcopy(self.geometry.CONTRACT)
        if not entry.get('failed') and entry['label']:
            proof = self.context['proof']
            entry['pre_damage_pair_sha256'] = self.identity.base_pair_sha256(
                self.full.s.unpack(proof, 'cut_a'), self.full.s.unpack(proof, 'cut_b'))
        return entry


def validate_entry(plan, job, entry):
    """Structural/provenance checks, not the independent pixel/target audit."""
    allowed = {r['path']: r for r in plan['source_rows']}
    require(isinstance(entry, dict) and isinstance(entry.get('attempted_donor_references'), list),
            'attempted donor references required, including failed jobs')
    for donor in entry['attempted_donor_references']:
        require(isinstance(donor, dict) and donor.get('path') in allowed and donor == allowed[donor['path']],
                'attempted donor outside exact admitted role sources')
    families = {r['source_family'] for r in plan['source_rows']}

    def check_meta(value):
        if isinstance(value, dict):
            if 'source_family' in value:
                require(value['source_family'] in families, 'accepted material donor is foreign')
            if 'source_mask' in value:
                require(value['source_mask'] in allowed, 'accepted material source path is foreign')
            for child in value.values():
                check_meta(child)
        elif isinstance(value, list):
            for child in value:
                check_meta(child)
    check_meta(entry.get('meta', {}))
    if entry.get('failed'):
        return
    require(entry.get('split') == plan['role'] and type(entry.get('label')) is bool
            and entry['label'] == bool(job[1]) and entry.get('recipe') == 'straight_' + job[0],
            'generated role, label or recipe mismatch')
    require(entry.get('training_admitted') is False, 'training admission forbidden')
    require(entry.get('wrapper_revision') == 'codex-v42-geometry-explicit-targets/1', 'corrected frozen supervision required')
    require(isinstance(entry.get('pair_id'), str) and entry['pair_id'], 'sample pair identity missing')
    require(isinstance(entry.get('piece_attempts'), list) and isinstance(entry.get('finalization_attempts'), list),
            'full attempt trace required')
    root = Path(plan['output_new']).resolve()
    for field, subdir in [('sample', 'samples')] + ([('proof', 'proof')] if entry['label'] else []):
        path = Path(entry[field + '_path']).resolve(strict=True)
        require(path.parent == root / subdir and file_sha(path) == entry[field + '_sha256'],
                'generated artifact path/hash mismatch')
    if entry['label']:
        require(isinstance(entry.get('target_audit'), dict) and isinstance(entry.get('accepted_damage_trace'), dict),
                'positive target/damage evidence missing')
        require(entry.get('geometry_attempts') and entry['geometry_attempts'][-1].get('passed') is True,
                'fixed strict geometry predicate did not pass')
        require(isinstance(entry.get('pre_damage_pair_sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', entry['pre_damage_pair_sha256']), 'pre-damage identity required')


def materialize(plan, *, backend_factory=FrozenBackend, pilot=False):
    """Materialize exactly one role, leaving independent validation pending.

    CPU fixtures can inject a backend; their output is explicitly marked and
    must not be promoted to a production admission. No automatic retries.
    """
    validate_plan(plan)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU-only environment required')
    out = Path(plan['output_new'])
    require(not out.exists(), 'new output directory required')
    out.mkdir(parents=True)
    (out / 'samples').mkdir(); (out / 'proof').mkdir()
    _save(out / 'plan.json', plan)
    backend = backend_factory(plan)
    entries, ids, failed, base_ids = [], set(), [], set()
    planned_jobs = jobs(plan)
    if pilot:
        require('pilot' in plan['dataset_id'], 'pilot must use a separate dataset namespace and plan')
        planned_jobs = [job for job in planned_jobs if job[2] == 0]
    _save(out / 'execution_plan.json', dict(mode='pilot6' if pilot else 'full320',
          expected=len(planned_jobs), jobs=planned_jobs, dataset_plan_sha256=file_sha(out / 'plan.json')))
    with (out / 'records.jsonl').open('x') as stream:
        for job in planned_jobs:
            entry = backend.generate_one(job)
            validate_entry(plan, job, entry)
            entry.update(actual_role=plan['role'], stage='strict_straight', generator_domain='straight_strip',
                         dataset_id=plan['dataset_id'], plan_sha256=file_sha(out / 'plan.json'),
                         parent_plan_sha256=plan['parent_plan']['sha256'], source_index_sha256=plan['source_index']['sha256'],
                         native_generator=entry.get('generator'), independent_audit_status='pending')
            stream.write(json.dumps(entry, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            if entry.get('failed'):
                failed.append(entry)
                continue
            require(entry['pair_id'] not in ids, 'duplicate generated pair ID')
            if entry['label']:
                require(entry['pre_damage_pair_sha256'] not in base_ids, 'duplicate pre-damage base pair')
                base_ids.add(entry['pre_damage_pair_sha256'])
            ids.add(entry['pair_id']); entries.append(entry)
    validate_plan(plan)  # Closing source/hash check, including actual donor bytes.
    counts = Counter((e['recipe'], e['label']) for e in entries)
    if not failed:
        require(all(counts['straight_' + k, label] == (1 if pilot else n)
                    for k, n in COUNTS.items() for label in (True, False)), 'quota not filled')
    manifest = dict(schema='mixed-heldout-straight-materialization/1', split=plan['role'], entries=entries,
                    failed=failed, plan_sha256=file_sha(out / 'plan.json'), real_used=False, test_used=False,
                    training_admitted=False, independent_audit_status='pending', pilot=pilot,
                    execution_plan_sha256=file_sha(out / 'execution_plan.json'))
    _save(out / 'manifest.json', manifest)
    receipt = dict(status='incomplete' if failed else 'materialized_pending_independent_audit',
                   generated=len(entries), expected=len(planned_jobs), failed=len(failed), pilot=pilot,
                   manifest_sha256=file_sha(out / 'manifest.json'), records_sha256=file_sha(out / 'records.jsonl'),
                   plan_sha256=file_sha(out / 'plan.json'), production_backend=backend_factory is FrozenBackend,
                   labels_algorithm_modified=False, gpu_used=False, model_inference=False,
                   checkpoint_selected=False, head_training_started=False, training_admitted=False,
                   independent_audit_status='pending')
    _save(out / 'materialization_receipt.json', receipt)
    return receipt


def _save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def audit_materialization(plan, *, load_sample, target_builder_path, expected_target_builder_sha):
    """Read-only pixel replay and target audit, writing a new independent receipt.

    The caller must bind the supplied frozen NPZ loader's source inventory in
    its launcher receipt. Target-builder bytes are separately bound here.
    """
    import numpy as np
    validate_plan(plan)
    out = Path(plan['output_new'])
    manifest = json.loads((out / 'manifest.json').read_text())
    complete = json.loads((out / 'materialization_receipt.json').read_text())
    require(complete.get('production_backend') is True and not manifest['failed']
            and complete['status'] == 'materialized_pending_independent_audit', 'production successful materialization required')
    require(file_sha(out / 'manifest.json') == complete['manifest_sha256']
            and file_sha(out / 'plan.json') == complete['plan_sha256'] == manifest['plan_sha256'], 'materialization receipt changed')
    require(len(manifest['entries']) == complete['generated'] == complete['expected'], 'incomplete audit population')
    receipt = _receipt(target_builder_path, expected_target_builder_sha)
    builder = _load('_heldout_target_audit_' + uuid.uuid4().hex, receipt['path'])
    backend = FrozenBackend(plan, audit_only=True)
    s, context = backend.supervision, backend.context
    records, seen_inputs, seen_base = [], set(), set()
    for entry in manifest['entries']:
        sample, report = load_sample(entry['sample_path'])
        require(file_sha(entry['sample_path']) == entry['sample_sha256']
                and sample.pair_id == entry['pair_id'] and bool(sample.label) == entry['label'], 'loaded sample identity differs')
        d = hashlib.sha256()
        for field in ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b'):
            value = np.ascontiguousarray(getattr(sample, field)); d.update(field.encode()); d.update(value.tobytes())
        input_sha = d.hexdigest()
        require(input_sha not in seen_inputs, 'duplicate actual model inputs')
        seen_inputs.add(input_sha)
        base_sha = None
        if entry['label']:
            require(file_sha(entry['proof_path']) == entry['proof_sha256'], 'proof changed')
            with np.load(entry['proof_path'], allow_pickle=False) as archive:
                proof = {k: archive[k] for k in archive.files}
            base_sha = backend.identity.base_pair_sha256(s.unpack(proof, 'cut_a'), s.unpack(proof, 'cut_b'))
            require(base_sha == entry['pre_damage_pair_sha256'] and base_sha not in seen_base, 'repeated/mismatched pre-damage base')
            seen_base.add(base_sha)
            context.update(donors={}, rejections=Counter(), proof=None)
            aa, bb, trace = s.replay_final_attempt(backend.reference, entry, plan['masterseeds'][plan['role']])
            for name, value in [('final_parent_a', aa), ('final_parent_b', bb),
                                ('cut_a', context['cut'][0]), ('cut_b', context['cut'][1]),
                                ('post_misfit_a', context['misfit'][2]), ('post_misfit_b', context['misfit'][3])]:
                np.testing.assert_array_equal(value, s.unpack(proof, name))
            original, reason = backend.reference.finalize(aa, bb, copy.deepcopy(entry['meta']), True)
            require(reason is None, 'accepted geometry does not replay')
            ia, ib, t, pa, va, pb, vb, ta, tb = original
            for field, value in [('mask_a', ia[None].astype(np.float32)), ('mask_b', ib[None].astype(np.float32)),
                                 ('points_rc_a', pa.astype(np.float32)), ('points_rc_b', pb.astype(np.float32)),
                                 ('contour_valid_a', va), ('contour_valid_b', vb), ('translation_a_to_b_rc', t)]:
                np.testing.assert_array_equal(getattr(sample, field), value, err_msg=entry['id'] + ':' + field)
            np.testing.assert_array_equal(proof['original_target_a'], ta)
            np.testing.assert_array_equal(proof['original_target_b'], tb)
            raw = types.SimpleNamespace(points_rc_a=pa, points_rc_b=pb, contour_valid_a=va,
                                        contour_valid_b=vb, target_a=ta, target_b=tb)
            target, details = s.build_supervision(raw, proof, trace, builder.projected_interval_damage)
            np.testing.assert_array_equal(sample.target_a, target['target_a'])
            np.testing.assert_array_equal(sample.target_b, target['target_b'])
            require(details['correspondence_count'] >= 8, 'too few healthy correspondences')
            require(not backend.geometry.violations(entry['recipe'][-1], backend.metrics.measure(sample)),
                    'frozen strict geometry predicate violated')
            validate_entry(plan, (entry['recipe'][-1], 1, 0, plan['role'], plan['output_new'], 0),
                           dict(entry, attempted_donor_references=list(context['donors'].values())))
        else:
            require(np.all(sample.target_a == -1) and np.all(sample.target_b == -1) and not sample.translation_valid,
                    'negative sample fabricated seam or translation supervision')
            left, right = entry['meta']['a'].get('source_family'), entry['meta']['b'].get('source_family')
            require(not left or not right or left != right, 'negative pieces share a material family')
            if entry['recipe'] == 'straight_R':
                ha, hb = entry['meta']['a']['strip_height'], entry['meta']['b']['strip_height']
                require(abs(hb-ha) <= .03*ha, 'R negative height control changed')
            details = {'correspondence_count': 0}
        records.append(dict(pair_id=entry['pair_id'], id=entry['id'], label=entry['label'],
                            sample_sha256=entry['sample_sha256'], model_input_sha256=input_sha,
                            pre_damage_pair_sha256=base_sha, target_audit=details,
                            strict_geometry_passed=True if entry['label'] else None))
    validate_plan(plan)
    require(file_sha(out / 'manifest.json') == complete['manifest_sha256'], 'manifest changed during audit')
    result = dict(status='passed_integrity_supervision_and_fixed_geometry', rows=len(records), records=records,
                  target_builder=receipt, materialization_receipt_sha256=file_sha(out / 'materialization_receipt.json'),
                  manifest_sha256=complete['manifest_sha256'], labels_algorithm_modified=False,
                  gpu_used=False, model_inference=False, training_admitted=False, pilot=complete['pilot'])
    _save(out / 'independent_audit.json', result)
    return result

"""Read-only, fail-closed admission of a completed new curriculum component.

This is not model inference, target repair or a complete mixed-data release.
The donor columns are conservative exclusion bounds, NOT lists of donors all
actually used. Accepted donors and historically unavailable logs are separate.
"""
from collections import Counter
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import sys

try:
    from .heldout_augment import tensor_identity, STAGES
    from .heldout_base_release import approved_parents, validate_group
    from .heldout_plan import canonical_generator
    from .heldout_run import validate_tasks
    from .protocol import ROW_FIELDS
except ImportError:
    from heldout_augment import tensor_identity, STAGES
    from heldout_base_release import approved_parents, validate_group
    from heldout_plan import canonical_generator
    from heldout_run import validate_tasks
    from protocol import ROW_FIELDS


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Evidence:
    """Hash each actual file, reject conflicting receipts and concurrent writes."""
    def __init__(self):
        self.files = {}
        self.objects = {}

    def file(self, path, expected=None):
        p = Path(path)
        need(p.is_absolute() and not p.is_symlink() and p.is_file(), 'absolute regular evidence file required: '+str(p))
        p = p.resolve(strict=True)
        s = p.stat()
        signature = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if p not in self.files:
            self.files[p] = (sha(p), signature)
        actual, old = self.files[p]
        need(signature == old, 'evidence changed during admission: '+str(p))
        if expected is not None:
            need(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) and actual == expected,
                 'evidence SHA mismatch: '+str(p))
        return dict(path=str(p), sha256=actual)

    def read(self, path, expected=None):
        ref = self.file(path, expected)
        if ref['path'] not in self.objects:
            self.objects[ref['path']] = json.loads(Path(ref['path']).read_text())
        return self.objects[ref['path']]

    def ref(self, ref):
        return self.read(ref['path'], ref['sha256'])

    def mapping(self, values, root=None):
        need(isinstance(values, dict) and values, 'nonempty source/file binding map required')
        for name, expected in values.items():
            p = Path(name)
            if root is not None:
                need(not p.is_absolute() and '..' not in p.parts, 'unsafe relative source path')
                p = Path(root)/p
            self.file(p, expected)

    def finish(self):
        for p, (expected, _) in list(self.files.items()):
            self.file(p, expected)
            if p.suffix in ('.json', '.py'):
                need(sha(p) == expected, 'metadata/source bytes changed during admission')
        return [dict(path=str(p), sha256=value[0]) for p, value in sorted(self.files.items())]


def _controller(root, ev):
    launch = ev.read(root/'controller_launch.json')
    final = ev.read(root/'controller_complete.json')
    actual = ev.read(root/'full_actual_return.json')
    need(final.get('status') == 'curriculum_complete_pending_combined_release_audit', 'controller incomplete')
    need(all(final.get(k) is True for k in ('no_model_selection','no_training','no_gpu')), 'not evaluation-only generation')
    need(final.get('desired_curriculum_pairs_per_role') == 2880, 'wrong controller budget')
    need(type(actual.get('returncode')) is int and actual['returncode'] == 0, 'actual full process did not return integer zero')
    bindings = launch.get('bindings')
    need(isinstance(bindings, dict) and bindings and final.get('bindings') == actual.get('bindings') == bindings,
         'controller/actual return source bindings disagree')
    ev.mapping(bindings)
    command = actual.get('command')
    need(isinstance(command, list) and len(command) == 4 and command[2:] == ['--root', str(root)],
         'actual command is not full heldout_run --root')
    script = Path(command[1])
    extension = launch.get('schema') == 'mixed-heldout-extension-controller/1'
    if extension:
        need(script.name == 'heldout_extend_run.py' and str(script) in bindings and
             launch.get('total_registered_reserves') == final.get('reserve_count') == 12 and
             launch.get('old_pixel_reconstruction_not_repeated') is True, 'extension runner/configuration differs')
        admission_path = str(root/'extension_admission.json')
        need(launch.get('extension_admission_path') == admission_path and
             launch.get('extension_admission_sha256') == bindings.get(admission_path), 'extension admission not launch-bound')
        ev.read(admission_path,bindings[admission_path])
        plan_path = str(root/'plan'/'generation_plan.json')
        need(plan_path in bindings and ev.read(plan_path,bindings[plan_path]).get('schema') ==
             'mixed-sim-heldout-generation-plan/2', 'extension requires bound v2 plan')
        payload_path = str(root/'catalog_payload_admission.json')
        need(payload_path in bindings, 'extension native payload admission missing')
        payload_ref = dict(path=payload_path,sha256=bindings[payload_path])
        payload = ev.ref(payload_ref)
        need(payload.get('schema') == 'heldout-catalog-payload-admission/1' and payload.get('status') == 'byte_exact' and
             final.get('catalog_payload_after') == dict(status='unchanged',admission=payload_ref),
             'extension native payload before/after admission differs')
        ev.mapping(payload['files'])
    else:
        need(launch.get('schema') is None and script.name == 'heldout_run.py' and str(script) in bindings,
             'actual full runner unbound or controller schema unsupported')
    need(launch.get('cuda_visible_devices') == '' and launch.get('models_loaded') is False and
         launch.get('original_outputs_changed') is False and launch.get('workers') == 2 and
         launch.get('threads_per_worker') == 1, 'unsafe controller configuration')
    need(launch['started_unix'] <= actual['started_unix'] <= actual['finished_unix'] <= final['finished_unix'],
         'completion chronology invalid')
    inventory_path = root/'frozen_source_inventory.json'
    need(str(inventory_path) in bindings, 'frozen runtime inventory not bound at launch')
    inventory = ev.read(inventory_path, bindings[str(inventory_path)])
    need(inventory.get('root') == launch.get('frozen_runtime'), 'runtime identity differs')
    ev.mapping(inventory['files'], inventory['root'])
    return launch, inventory


def _extension_context(root, launch, plan, ev):
    """Only the exact append-only v2 plan and byte-preserved adoption scheme."""
    need(launch.get('schema') == 'mixed-heldout-extension-controller/1', 'v2 plan has no extension controller')
    try:
        from .heldout_extend_plan import validate_extension
        from .heldout_adopt import check_admission
    except ImportError:
        from heldout_extend_plan import validate_extension
        from heldout_adopt import check_admission
    validated, sources = validate_extension(root/'plan'/'generation_plan.json')
    need(validated == plan and plan['reserve_count'] == 12, 'deterministic extension validation differs')
    old = Path(plan['extension_of']['build_root']).resolve(strict=True)
    need(str(old) == launch['old_build_root'] and old != root, 'extension original build identity differs')
    admission = ev.read(launch['extension_admission_path'],launch['extension_admission_sha256'])
    check_admission(admission)
    need(admission['old_build_root'] == str(old) and admission['old_plan'] == plan['extension_of']['generation_plan'] and
         admission['old_plan_sha256'] == plan['extension_of']['generation_plan']['sha256'] and
         admission['original_receipts']['old_sources'] == plan['extension_of']['source_plan'],
         'independent adoption refers to another original build/registration')
    ev.mapping(admission['bound_files'])
    for ref in plan['extension_of'].values():
        if isinstance(ref, dict):
            ev.ref(ref)
    payload = ev.read(root/'catalog_payload_admission.json')
    cat = ev.read(sources['catalog_path'],sources['catalog_sha256'])
    expected = {str(Path(cat['release_root'])/rel):value for rel,value in cat['copied_files_sha256'].items()}
    expected.update(cat['original_files_sha256'])
    need(payload['files'] == expected and payload['catalog'] ==
         dict(path=sources['catalog_path'],sha256=sources['catalog_sha256']) and
         payload['copied_count'] == len(cat['copied_files_sha256']) and
         payload['original_count'] == len(cat['original_files_sha256']), 'extension payload scope is incomplete')
    return dict(admission=admission,old_root=old,sources=sources)


def _imported_group(group, task, root, role, extension, ev):
    """Return the actual final-pixel root after exact old-commit admission."""
    if 'imported_from' not in group:
        return root
    need(extension is not None, 'legacy group cannot import another build')
    imported = group['imported_from']
    need(set(imported) == {'commit','old_plan','old_pilot_complete'} and
         group.get('old_pixels_regenerated') is False, 'imported proof schema differs')
    expected = extension['admission']['roles'][role]['commits'].get(f"{task['stage']}:{task['slot']}")
    need(imported == expected, 'group import is not in independently checked admission index')
    old = extension['old_root']
    need(imported['commit']['path'] == str(old/'augmented'/role/task['stage']/'groups'/f"{task['slot']:05d}.json") and
         imported['old_plan']['path'] == str(old/'plan'/'generation_plan.json') and
         imported['old_pilot_complete']['path'] == str(old/'pilot_receipts'/role/'complete.json'),
         'imported paths do not identify original registered role/slot')
    prior = ev.ref(imported['commit'])
    old_plan = ev.ref(imported['old_plan'])
    pilot = ev.ref(imported['old_pilot_complete'])
    need(prior.get('status') == 'committed' and prior['task'] == task and task in old_plan['tasks'] and
         prior['plan_sha256'] == pilot['plan_sha256'] == imported['old_plan']['sha256'] and
         pilot['role'] == role and pilot['pilot_only'] is True and pilot['status'] in ('pilot_passed','shortfall') and
         prior['records'] == group['records'] and prior['audit_rows'] == group['audit_rows'] and
         all(record in pilot['records'] for record in prior['records']), 'old success proof changed or not pilot-admitted')
    return old


def _baseline_root(root, role, slot, audit, extension, ev):
    current = root/'baseline'/role
    if extension is None:
        need('imported_from' not in audit and 'root' not in audit, 'legacy baseline has foreign adoption fields')
        return current
    adopted = extension['admission']['roles'][role]['baselines'].get(str(slot))
    if adopted is None:
        need(audit.get('root') == str(current) and 'imported_from' not in audit, 'new baseline path not current build')
        return current
    prior = Path(adopted['root'])
    need(prior == extension['old_root']/'baseline'/role and audit.get('root') == str(prior) and
         audit.get('imported_from') == adopted['group'] and audit['group_sha256'] == adopted['group']['sha256'] and
         audit['audit_rows'] == adopted['audit_rows'] and
         adopted['group']['path'] == str(prior/'groups'/f'{int(slot):05d}.json'), 'baseline import differs from exact adoption index')
    ev.ref(adopted['group'])
    return prior


def _native_loader(inventory):
    """Import only the registered native sample loader, never a model."""
    root = Path(inventory['root']).resolve(strict=True)
    prefix = 'staging.pairwise_v0_2'
    for name, module in list(sys.modules.items()):
        p = getattr(module, '__file__', None)
        if name.startswith(prefix) and p:
            need(Path(p).resolve().is_relative_to(root), 'preloaded native module belongs to another runtime')
    sys.path.insert(0, str(root))
    module = importlib.import_module(prefix+'.pairwise_data.rachel_materialized_dataset')
    expected = root/'staging/pairwise_v0_2/pairwise_data/rachel_materialized_dataset.py'
    need(Path(inspect.getsourcefile(module.load_sample)).resolve() == expected, 'wrong native load_sample implementation')
    for name, module_loaded in list(sys.modules.items()):
        p = getattr(module_loaded, '__file__', None)
        if name.startswith(prefix) and p and Path(p).suffix == '.py':
            p = Path(p).resolve()
            need(p.is_relative_to(root) and inventory['files'].get(str(p.relative_to(root))) == sha(p),
                 'native dependency not bound to frozen source')
    return module.load_sample


def _catalog(sources, launch, ev):
    for key in ('catalog', 'parent_plan', 'profile'):
        p, expected = sources[key+'_path'], sources[key+'_sha256']
        need(launch['bindings'].get(p) == expected, key+' differs from launch registration')
    cat = ev.read(sources['catalog_path'], sources['catalog_sha256'])
    parents = ev.read(sources['parent_plan_path'], sources['parent_plan_sha256'])
    allowed = approved_parents(parents)
    need(cat['parent_plan_sha256'] == sources['parent_plan_sha256'] and
         cat['release_root'] == sources['dataset_root'], 'catalog/source parent or data root differs')
    # This is final byte admission, not a claim the generation launcher checked
    # every native file before launch. Catalog admission itself predates launch.
    ev.mapping(cat['copied_files_sha256'], cat['release_root'])
    ev.mapping(cat['original_files_sha256'])
    for name, expected in cat['native_source_sha256'].items():
        need(re.fullmatch('[A-Za-z0-9_]+', name), 'unsafe native preprocessing module name')
        ev.file(Path(launch['frozen_runtime'])/'staging/pairwise_v0_2/pairwise_data'/(name+'.py'), expected)
    previous_admission = ev.ref(cat['previous_native_verification'])
    need(previous_admission.get('source_files_unchanged') is True and
         previous_admission.get('catalog') == cat['previous_catalog'], 'old native admission is not bound to reused catalog')
    ev.ref(cat['previous_catalog'])
    fragments = {}
    for f in cat['fragments']:
        token = f['row']['fragment_token']
        need(token not in fragments, 'duplicate catalog fragment token')
        need(f['role'] in allowed and f['family'] in allowed[f['role']], 'catalog primary outside registered fold')
        for key in ('model_mask_path','contour_path'):
            need(f['row'][key] in cat['copied_files_sha256'], 'native model file missing catalog byte receipt')
        fragments[token] = f
    releases = {r['path']: ev.ref(r) for r in cat['base_release_refs']}
    ref_shas = {r['path']:r['sha256'] for r in cat['base_release_refs']}
    admission_found = False
    for ref in cat['base_release_refs']:
        # The native adapter's fixed sibling receipt is not an ID inference:
        # its contents must bind this exact catalog and this exact base release.
        admission = ev.read(Path(ref['path']).parent/'verification.json')
        if admission.get('catalog') == dict(path=sources['catalog_path'],sha256=sources['catalog_sha256']):
            need(admission.get('status') == 'native_catalog_admitted_pending_stage_augmentation' and
                 admission.get('base_release') == ref and admission.get('source_files_unchanged') is True and
                 admission.get('native_source_sha256') == cat['native_source_sha256'], 'current native admission differs')
            exclusion = ev.ref(admission['real_source_exclusion'])
            need(exclusion.get('status') == 'passed_source_exclusion' and
                 exclusion.get('parent_plan_sha256') == sources['parent_plan_sha256'] and
                 exclusion.get('exact_prepared_mask_matches') == [] and
                 exclusion.get('real_labels_read') is False and exclusion.get('real_scores_read') is False,
                 'new base real-source exclusion receipt is not admitted')
            ev.ref(exclusion['combined_parent_guard'])
            ev.ref(exclusion['reused_real_hash_index'])
            checked = {r['fragment_token']:r for r in exclusion['checked_gen45_fragment_masks']}
            expected_tokens = {token for token,f in fragments.items() if canonical_generator(f['row']['generator']) in ('Gen4','Gen5')}
            need(set(checked) == expected_tokens, 'real-source exclusion omitted a new base fragment')
            for token, row in checked.items():
                f = fragments[token]
                need(row['role'] == f['role'] and row['family'] == f['family'] and
                     Path(row['model_mask_path']) == Path(cat['release_root'])/f['row']['model_mask_path'],
                     'real-source exclusion fragment identity differs')
                ev.file(row['model_mask_path'],row['file_sha256'])
            admission_found = True
    need(admission_found, 'no original native admission bound to final catalog')
    group_edges = {}
    for gid, item in cat['base_provenance_by_parent_group_id'].items():
        ref = item['base_release_ref']
        need(ref_shas.get(ref['path']) == ref['sha256'], 'base release ref is not catalog bound')
        release = releases[ref['path']]
        index = item['base_release_group_index']
        need(type(index) is int and 0 <= index < len(release['groups']), 'bad base release group index')
        group = release['groups'][index]
        need(item['raw_source_pointer'] == f'groups/{index}/raw_source', 'raw source pointer mismatch')
        need(all(group[k] == item[k] for k in ('role','family','generator')) and group['group_id'] == item['raw_group_id'],
             'catalog/base source identity disagreement')
        raw = group['raw_source']
        ev.mapping(validate_group(raw, allowed, verify_files=False))
        ev.mapping(raw['source_binding'])
        ev.ref(raw['edge_pool'])
        complete = ev.ref(item['raw_complete_ref'])
        plan = ev.ref(item['generation_plan_ref'])
        need(release['generation_plan_sha256'] == item['generation_plan_ref']['sha256'], 'base generation plan binding differs')
        raw_groups = complete.get('groups', []) + [g for job in complete.get('jobs', []) for g in job['successful_groups']]
        need(any(g['group_id'] == raw['group_id'] and g['role'] == raw['role'] and
                 g['generator'] == raw['generator'] for g in raw_groups), 'raw completed group missing')
        need(isinstance(plan, dict) and plan and complete['plan_sha256'] == item['generation_plan_ref']['sha256'],
             'raw completion is not bound to its generation plan')
        for path_key, hash_key in (('parent_image_path','parent_image_sha256'), ('csv_path','csv_sha256')):
            ev.file(raw[path_key], raw[hash_key])
        for fragment in raw['fragments']:
            ev.file(fragment['path'], fragment['sha256'])
        edges = []
        for donor in raw['edge_donors']:
            for pkey, skey in (('parent_image_path','parent_image_sha256'),('raw_edge_path','raw_edge_sha256'),
                              ('variant_path','variant_sha256')):
                ev.file(donor[pkey], donor[skey])
            edges.append(dict(parent_id=donor['family'],
                              fragment_ids=['tearing-edge-sha256/'+donor['raw_edge_sha256'],
                                            'tearing-edge-variant-sha256/'+donor['variant_sha256']],
                              source=donor))
        marker = group['marker']
        need(marker['status'] == 'processed_group', 'native preprocessing not complete')
        matched = [f for f in fragments.values() if f['row']['parent_group_id'] == gid]
        need(matched, 'unreferenced base provenance group')
        marker_tokens = {f['fragment_token']:f for f in marker['fragments']}
        for f in matched:
            need(f['role'] == group['role'] and f['family'] == group['family'], 'exact group provenance mismatch')
            native = marker_tokens.get(f['row']['fragment_token'], {})
            need(all(native.get(k) == v for k,v in f['row'].items()), 'fragment differs from original native marker')
        group_edges[gid] = dict(edges=edges, provenance=item)
    for f in fragments.values():
        gen = canonical_generator(f['row']['generator'])
        need(gen not in ('Gen4','Gen5') or f['row']['parent_group_id'] in group_edges,
             'new Gen4/5 lacks exact underlying edge provenance')
    return cat, allowed, fragments, group_edges


def _donor_bounds(role, source_role, fragments, group_edges, ev):
    import numpy as np
    bankref = source_role['donor_bank']
    bankroot = Path(bankref['path'])
    bank = ev.read(bankroot/'bank.json', bankref['metadata_sha256'])
    ev.file(bankroot/'profiles.npz', bankref['profiles_sha256'])
    need(bank.get('split') == role and bank.get('train_profiles_used') is False, 'bank role/training contamination')
    arcs = bank['arcs']
    need(len(arcs) == bankref['profile_count'] and arcs, 'bank profile population changed')
    with np.load(bankroot/'profiles.npz',allow_pickle=False) as archive:
        profiles = archive['profiles'].copy()
    need(len(profiles) == len(arcs) and np.isfinite(profiles).all(), 'actual bank profile arrays differ')
    parents, tokens = set(), set()
    for arc in arcs:
        f = fragments.get(arc['fragment_token'])
        need(f is not None and f['role'] == role and arc['split'] == role and arc['family'] == f['family'] and
             arc['lineage'] == f['row']['split_unit_id'], 'bank arc not bound to exact same-fold fragment')
        parents.add(f['family']); tokens.add(arc['fragment_token'])
        for edge in group_edges.get(f['row']['parent_group_id'], {}).get('edges', []):
            parents.add(edge['parent_id']); tokens.update(edge['fragment_ids'])
    return dict(bank,_profiles=profiles), dict(parent_ids=sorted(parents), base_pair_ids=[], fragment_ids=sorted(tokens))


def _source_row(record, task, source_role, cat, fragments, role):
    ordinal, slot = record['baseline_ordinal'], task['slot']
    pid = task['base_pair_ids'][ordinal]
    need(record['source_pair_id'] == pid, 'record base pair not preregistered')
    if ordinal == 0:
        pool = {r['pair_id']:r for r in cat['positive_rows'][role]}
        need(source_role['positive'][slot]['pair_id'] == pid and pid in pool, 'positive not catalog-admitted')
        row = pool[pid]
        need(row in source_role['positive_pool'], 'positive absent from fixed source pool')
    else:
        item = source_role['negative'][slot]
        need(item['mode'] == 'native' and item['pair_id'] == pid, 'negative source registration mismatch')
        row = item['row']
        need(row['label'] is False and row.get('correspondence_path') is None and
             row.get('label_origin') == 'distinct_canonical_manuscript_families', 'negative label rule changed')
    need(type(row['label']) is bool and row['label'] == record['label'], 'native source label differs')
    need({k:v for k,v in record['source_row'].items() if k != 'pair_id'} ==
         {k:v for k,v in row.items() if k != 'pair_id'}, 'source row altered beyond baseline ID namespace')
    found = []
    for side in 'ab':
        source = row['fragment_'+side]
        f = fragments.get(source['fragment_token'])
        need(f is not None and f['row'] == source and f['role'] == role, 'exact primary catalog token mismatch')
        need(canonical_generator(source['generator']) == task['generator'], 'native generator relabelled')
        found.append(f)
    if ordinal == 1:
        need(found[0]['family'] != found[1]['family'], 'cross-manuscript negative shares a canonical parent')
    return row, found


def _accepted_donors(record, report, baseline_report, bank, fragments, group_edges, role):
    trim = record['detail']['trim']
    partial = baseline_report['compound']['partial'] if record['partial_applied'] else None
    need(report['compound'].get('partial') == partial, 'partial metadata differs from original baseline')
    accepted = []
    for kind, detail in (('trim', trim if trim.get('applied', True) else None), ('partial', partial)):
        if not detail or detail.get('applied', True) is False:
            continue
        index = detail.get('donor_index')
        need(type(index) is int and 0 <= index < len(bank['arcs']), 'accepted donor index missing or invalid')
        arc = bank['arcs'][index]
        if kind == 'trim':
            need(detail.get('donor') == arc, 'trim actual donor differs from bound bank arc')
            import numpy as np
            profile = bank['_profiles'][index]
            need(detail.get('profile_sha256') == hashlib.sha256(profile.tobytes()).hexdigest() and
                 np.array_equal(np.asarray(detail.get('profile'),dtype=profile.dtype),profile),
                 'trim actual curve profile differs from bound donor index')
        else:
            need(detail.get('donor_lineage') == arc['lineage'] and detail.get('donor_family') == arc['family'] and
                 detail.get('donor_split') == role, 'partial donor does not match bound bank')
        f = fragments[arc['fragment_token']]
        accepted.append(dict(kind=kind, bank_index=index, parent_id=f['family'], fragment_id=arc['fragment_token'],
                             lineage=arc['lineage'], underlying_edge_sources=group_edges.get(
                                 f['row']['parent_group_id'], {}).get('edges', [])))
    need(record['augmentation_donor_sources'] == [d['lineage'] for d in accepted], 'accepted donor lineages differ')
    return accepted


def _audit_hash(sample):
    h = hashlib.sha256()
    for key in ('mask_a','mask_b','points_rc_a','points_rc_b','target_a','target_b','translation_a_to_b_rc'):
        h.update(key.encode()); h.update(getattr(sample,key).tobytes())
    return h.hexdigest()


def _record(record, task, group, source_role, cat, fragments, group_edges, bank, bounds, root, role, ev, loader,
            *, sample_root=None, baseline_root=None, numeric_mirror_schedule=False):
    sample_root = root if sample_root is None else sample_root
    baseline_root = root/'baseline'/role if baseline_root is None else baseline_root
    ordinal = record['baseline_ordinal']
    need(type(ordinal) is int and ordinal in (0,1) and type(record['label']) is bool and record['label'] == (ordinal == 0),
         'actual paired positive/negative ordinals differ')
    need(record['baseline_slot'] == task['slot'] and record['data_role'] == role and record['stage'] == task['stage'] and
         record['version'] == task['stage'] and record['generator'] == task['generator'] and
         record['recipe'] == record['corrosion_recipe'] == task['recipe'] and record['v14_fallback'] is False,
         'record metadata differs from preregistered task')
    need(source_role['slot_generators'][task['slot']] == task['generator'] and
         source_role['schedule']['recipes'][task['slot']] == task['recipe'], 'source/task recipe or Gen mismatch')
    detail = record['detail']
    need(record['requested_gap_count'] == task['k'] and detail['requested_gap_count'] ==
         (task['k'] if task['recipe'].startswith('gaps') else 0) and
         detail['trim']['size_class'] == task['size_class'], 'actual trim size or gap request differs from task')
    if task['stage'] != 'v17_filtered':
        need(detail['trim']['target_removed_fraction'] == task['trim_target'] and
             detail['depth_revision'] == task['stage'] and record['partial_applied'] == detail['partial_crop_applied'],
             'actual curriculum stage/trim/partial policy differs')
    else:
        need(record['partial_applied'] == (task['recipe'] == 'partial'), 'v17 baseline partial recipe differs')
    mirror = source_role['schedule']['mirrors'][task['slot']]
    if numeric_mirror_schedule:
        need(type(mirror) is int and mirror in (0, 1, 2), 'unregistered mirror code')
        mirror = {0: None, 1: 'horizontal', 2: 'vertical'}[mirror]
    need(record['offline_paired_mirror'] == mirror, 'registered paired mirror differs')
    basename = f"{task['slot']:05d}_{ordinal}.npz"
    paths = [('sample_path','sample_sha256',sample_root/'augmented'/role/task['stage']/'samples'/basename),
             ('proof_path','proof_sha256',sample_root/'augmented'/role/task['stage']/'proof'/basename),
             ('target_metadata','target_metadata_sha256',sample_root/'augmented'/role/task['stage']/'targets'/basename),
             ('unaugmented_sample_path','unaugmented_sample_sha256',sample_root/'augmented'/role/'unaugmented'/basename),
             ('baseline_sample_path','baseline_sample_sha256',None),
             ('baseline_group_path','baseline_group_sha256',baseline_root/'groups'/f"{task['slot']:05d}.json")]
    if ordinal == 0:
        paths.append(('latent_seam_artifact','latent_sha256',sample_root/'augmented'/role/task['stage']/'latent'/basename))
    else:
        need(record.get('latent_seam_artifact') is None and record.get('latent_sha256') is None, 'negative has fabricated latent seam')
    for pkey, skey, expectedpath in paths:
        need(expectedpath is None or Path(record[pkey]) == expectedpath, 'record path escaped registered output slot')
        ev.file(record[pkey], record[skey])
    baseline = ev.read(record['baseline_group_path'], record['baseline_group_sha256'])
    need(len(baseline['entries']) == 2, 'baseline group incomplete')
    entry = baseline['entries'][ordinal]
    for key in ('source_pair_id','source_row','source_root','negative_kind','offline_paired_mirror'):
        need(entry[key] == record[key], 'baseline provenance changed: '+key)
    need(entry['corrosion_recipe'] == task['recipe'] and entry['pair_id'] == record['source_row']['pair_id'] and
         Path(record['baseline_sample_path']) == baseline_root/entry['artifact_path'], 'baseline actual archive mismatch')
    need(record['source_root'] == cat['release_root'], 'native source root changed')
    source_row, primary = _source_row(record,task,source_role,cat,fragments,role)
    sample, report = loader(Path(record['sample_path']))
    clean, _ = loader(Path(record['unaugmented_sample_path']))
    baseline_sample, baseline_report = loader(Path(record['baseline_sample_path']))
    need(sample.pair_id == record['pair_id'] == record['id'] and bool(sample.label) == record['label'] and
         baseline_sample.pair_id == entry['pair_id'] and bool(baseline_sample.label) == record['label'] and
         bool(clean.label) == record['label'], 'actual native archive ID/label mismatch')
    need(report.get('data_role') == role and report.get('recipe') == task['recipe'] and
         report.get('paired_review') == record['detail'] and report.get('source_pair_id') == record['source_pair_id'] and
         report.get('base_v14_pair_id') == entry['pair_id'] and report.get('v14_fallback') is False,
         'actual archive metadata disagrees with records')
    for key, value in (('model_tensors_sha256',tensor_identity(sample)),
                       ('supervised_tensors_sha256',tensor_identity(sample,include_supervision=True)),
                       ('unaugmented_model_tensors_sha256',tensor_identity(clean))):
        need(record[key] == value, 'actual model/supervision tensor identity differs: '+key)
    audits = {a['id']:a for a in group['audit_rows']}
    need(len(audits) == 2 and record['id'] in audits, 'independent admitted pixel audit missing')
    audit = audits[record['id']]
    need(audit.get('status') == 'passed' and audit.get('sample_sha256') == record['sample_sha256'] and
         audit.get('proof_sha256') == record['proof_sha256'] and audit.get('model_input_sha256') == _audit_hash(sample),
         'original independent pixel audit not bound to actual output')
    accepted = _accepted_donors(record, report, baseline_report, bank, fragments, group_edges, role)
    parent_edges = [edge for f in primary for edge in group_edges.get(f['row']['parent_group_id'],{}).get('edges',[])]
    donor_parents = set(bounds['parent_ids']) | {e['parent_id'] for e in parent_edges}
    donor_fragments = set(bounds['fragment_ids']) | {i for e in parent_edges for i in e['fragment_ids']}
    normalized = dict(pair_id=record['pair_id'],sample_sha256=record['sample_sha256'],stage=task['stage'],
         generator=task['generator'],label=record['label'],parent_ids=sorted({f['family'] for f in primary}),
         base_pair_ids=[source_row['pair_id']],fragment_ids=sorted({f['row']['fragment_token'] for f in primary}),
         donor_parent_ids=sorted(donor_parents),donor_base_pair_ids=[],donor_fragment_ids=sorted(donor_fragments),
         recipe=task['recipe'],sample_path=record['sample_path'],model_tensors_sha256=record['model_tensors_sha256'],
         supervised_tensors_sha256=record['supervised_tensors_sha256'],native_record=record,
         registered_generation_task=task,
         primary_base_group_provenance=[group_edges[f['row']['parent_group_id']]['provenance'] for f in primary
             if f['row']['parent_group_id'] in group_edges],
         accepted_augmentation_donors=accepted,primary_base_edge_sources=parent_edges,
         imported_original_commit=group.get('imported_from'),
         donor_identity_semantics='accepted plus conservative possible attempted-bank superset; NOT all actually used',
         attempted_augmentation_donor_log_complete=False,historical_gen23_tearing_edge_trace_complete=False,
         decision_role=role)
    need(ROW_FIELDS <= set(normalized), 'normalization schema incomplete')
    return normalized


def _population(entries):
    base_counts = Counter(p for e in entries for p in e['base_pair_ids'])
    return dict(rows=len(entries),primary_parent_count=len({p for e in entries for p in e['parent_ids']}),
                base_pair_count=len(base_counts),fragment_count=len({p for e in entries for p in e['fragment_ids']}),
                model_input_count=len({e['model_tensors_sha256'] for e in entries}),
                max_views_per_base_pair=max(base_counts.values(),default=0),
                reused_base_pairs=sum(n>1 for n in base_counts.values()))


def normalize_curriculum_release(root, role):
    """Verify the complete 2,880-row curriculum fold; perform no mutations."""
    need(role in ('cal','select'), 'only CAL or SELECT can be normalized')
    root = Path(root).resolve(strict=True)
    ev = Evidence()
    launch, inventory = _controller(root,ev)
    plan = ev.read(root/'plan'/'generation_plan.json')
    extension = None
    if plan.get('schema') == 'mixed-sim-heldout-generation-plan/1':
        need(launch.get('schema') is None, 'old plan cannot use extension controller')
        validate_tasks(plan)
        reserves = 3
    elif plan.get('schema') == 'mixed-sim-heldout-generation-plan/2':
        extension = _extension_context(root,launch,plan,ev)
        reserves = 12
    else:
        raise ValueError('unsupported curriculum generation plan schema')
    need(plan['source_plan_path'] == str(root/'plan'/'sources.json') and plan.get('model_outputs_used') is False and
         plan.get('no_train_or_test_generation') is True, 'invalid preregistered source plan')
    sources = ev.read(plan['source_plan_path'],plan['source_plan_sha256'])
    if extension is not None:
        need(sources == extension['sources'], 'actual extension source plan differs')
    cat, parents, fragments, edges = _catalog(sources,launch,ev)
    source_role = sources['splits'][role]
    need(set(source_role['source_families']) == set(parents[role]), 'source role parent registration differs')
    bank, bounds = _donor_bounds(role,source_role,fragments,edges,ev)
    need(set(bounds['parent_ids']) <= set(parents[role]), 'potential augmentation/edge donor crosses fold')
    complete = ev.read(root/'build_receipts'/role/'complete.json')
    wanted_schema = 'mixed-sim-heldout-role-build/2' if extension else 'mixed-sim-heldout-role-build/1'
    need(complete.get('schema') == wanted_schema and complete.get('role') == role and
         complete.get('status') == 'complete' and complete.get('pilot_only') is False and
         complete.get('planned_quotas') == 1440 and complete.get('admitted_pairs') == 2880 and
         complete.get('gpu_used') is False and complete.get('model_outputs_used') is False and
         complete.get('plan_sha256') == ev.file(root/'plan'/'generation_plan.json')['sha256'], 'role is not fully complete')
    if extension is not None:
        need(complete.get('reserve_count') == 12 and complete.get('extension_admission') ==
             dict(path=launch['extension_admission_path'],sha256=launch['extension_admission_sha256']),
             'role completion not bound to independent extension admission')
    need(not any(f.get('phase') == 'quota_exhausted' for f in complete['failures']), 'quota exhaustion cannot be complete')
    ev.mapping(complete['frozen_source_sha256'])
    for p, value in complete['frozen_source_sha256'].items():
        path = Path(p)
        need(path.is_relative_to(inventory['root']) and inventory['files'].get(str(path.relative_to(inventory['root']))) == value,
             'actual generation dependency outside frozen source')
    records = complete['records']
    wanted = Counter({(stage,gen,label):120 for stage in STAGES for gen in ('Gen2','Gen3','Gen4','Gen5') for label in (False,True)})
    need(len(records) == 2880 and Counter((r['stage'],r['generator'],r['label']) for r in records) == wanted,
         'actual role population differs from fixed stage/Gen/label quotas')
    need(len({r['pair_id'] for r in records}) == 2880 and len({r['model_tensors_sha256'] for r in records}) == 2880,
         'duplicate actual pair/input in component')
    tasks = {(t['stage'],t['slot']):t for t in plan['tasks'] if t['role'] == role}
    need(len(tasks) == 1440*reserves, 'duplicate registered stage/slot identities')
    groups, quotas, pixel_roots = {}, set(), {}
    for record in records:
        key = (record['stage'],record['baseline_slot'])
        need(key in tasks, 'admitted record has no preregistered task')
        if key not in groups:
            task = tasks[key]
            group = ev.read(root/'augmented'/role/key[0]/'groups'/f'{key[1]:05d}.json')
            need(group['status'] == 'committed' and group['task'] == task and group['plan_sha256'] == complete['plan_sha256'] and
                 group.get('source_replaced') is False and group.get('v14_fallback') is False and len(group['records']) == 2,
                 'admitted group or task binding differs')
            quota = (task['stage'],task['generator'],task['quota_slot'])
            need(quota not in quotas, 'two reserves admitted for one fixed quota')
            quotas.add(quota); groups[key] = group
            pixel_roots[key] = _imported_group(group,task,root,role,extension,ev)
        need(record in groups[key]['records'], 'complete record differs from committed actual group')
    need(len(groups) == len(quotas) == 1440, 'missing actual committed groups')
    if extension is not None:
        need(complete.get('imported_groups') == sum('imported_from' in g for g in groups.values()),
             'imported group count does not match actual admitted wrappers')
    needed_slots = {str(r['baseline_slot']) for r in records}
    need(needed_slots <= set(complete['base_audit']), 'independent baseline audits missing')
    baseline_roots = {}
    for slot, audit in complete['base_audit'].items():
        baseline_roots[slot] = _baseline_root(root,role,slot,audit,extension,ev)
        baseline = ev.read(baseline_roots[slot]/'groups'/f'{int(slot):05d}.json',audit['group_sha256'])
        need(len(audit['audit_rows']) == 2 and {a['pair_id'] for a in audit['audit_rows']} ==
             {e['pair_id'] for e in baseline['entries']}, 'baseline actual audit identity mismatch')
    loader = _native_loader(inventory)
    entries = [_record(r,tasks[(r['stage'],r['baseline_slot'])],groups[(r['stage'],r['baseline_slot'])],source_role,
                       cat,fragments,edges,bank,bounds,root,role,ev,loader,
                       sample_root=pixel_roots[(r['stage'],r['baseline_slot'])],
                       baseline_root=baseline_roots[str(r['baseline_slot'])]) for r in records]
    files = ev.finish()
    return dict(schema='mixed-heldout-curriculum-component/1',status='verified_component_only',role=role,rows=len(entries),
        entries=entries,bound_receipts=dict(actual_files=files),
        provenance_summary=dict(rows=len(entries),committed_groups=len(groups),
            primary_parent_count=len({p for e in entries for p in e['parent_ids']}),
            base_pair_count=len({p for e in entries for p in e['base_pair_ids']}),
            donor_semantics='same-fold bank conservative exclusion upper bound, not actual use count',
            accepted_donors_separately_recorded=True,attempted_augmentation_donor_log_complete=False,
            historical_gen23_tearing_edge_trace_complete=False,new_gen45_actual_edge_trace_complete=True,
            legacy_curve_origin_strings_preserved_but_actual_source_admitted_by_bank_token_and_sha=True,
            reserve_count=reserves,imported_original_groups=sum('imported_from' in g for g in groups.values()),
            population=_population(entries),
            per_cell={stage:{gen:{str(label).lower():_population([e for e in entries if
                (e['stage'],e['generator'],e['label']) == (stage,gen,label)]) for label in (False,True)}
                for gen in ('Gen2','Gen3','Gen4','Gen5')} for stage in STAGES},
            catalog_files_checked_at_normalization=True,
            catalog_payload_checked_by_this_build_launcher=extension is not None,
            historical_source03_launcher_payload_check_claimed=False,
            complete_mixed_6400_release=False,model_inference=False,training=False,target_repair=False))

"""One independent, CPU-only publication of completed build05.

Preserves its real return2; accepts ONLY the two human-authorized CAL quotas.
Reads existing pixel audits, never regenerates/replays pixels or changes labels.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import heldout_curriculum_release as h, heldout_reduce_plan as rp
from .heldout_parallel_plan import validate_shards
from .heldout_publish import save, population
from .protocol import ROW_FIELDS, MANIFEST_SCHEMA, _validate_manifest

WAIVED = {('cal', 'Gen3', 19, 'v17_filtered'), ('cal', 'Gen5', 77, 'v17_filtered')}
BASE = Path('/root/autodl-tmp/model_selection_v2_20261002')
DEDUP_AUDIT_SHA = '6c30e269cdead51913a38b78053869c51fb98c15712ae80eccd5de506bf5e473'
RELEASE_STATUS = 'released_with_authorized_shortfalls_and_exact_deduplication'
FINAL_COUNTS = {'cal': (1587, 793, 794), 'select': (1596, 796, 800)}


def tree_refs(value, ev):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            ev.file(value['path'], value['sha256'])
        for v in value.values(): tree_refs(v, ev)
    elif isinstance(value, list):
        for v in value: tree_refs(v, ev)


def check_waivers(missing, actual_return, failures):
    h.need(set(missing) == WAIVED and len(missing) == len(WAIVED), 'unapproved or duplicate missing quota')
    h.need(type(actual_return) is int and actual_return == 2, 'preserve expected actual builder return2')
    h.need(not failures, 'unexpected failure cannot be waived')


def validate_population(rows, role, removed=(), allow_exact_duplicates=False):
    wanted = Counter({(s, g, label): 60 for s in rp.STAGES for g in rp.GENS for label in (False, True)})
    if role == 'cal':
        for _, g, _, s in WAIVED:
            for label in (False, True): wanted[s, g, label] -= 1
    wanted.update({('strict_straight', 'straight_strip', label): 80 for label in (False, True)})
    wanted.subtract(Counter((e['stage'], e['generator'], e['label']) for e in removed))
    h.need(Counter((e['stage'], e['generator'], e['label']) for e in rows) == wanted, 'actual stage/Gen/label population differs')
    keys = ('pair_id', 'sample_sha256') if allow_exact_duplicates else ('pair_id', 'sample_sha256', 'model_tensors_sha256')
    for key in keys:
        h.need(len({e[key] for e in rows}) == len(rows), 'duplicate final ' + key)


def authorized_deduplication(rows, role, audit):
    """Apply only the exact, human-approved audit; retain every original file."""
    h.need(audit['schema'] == 'cross-shard-exact-input-duplicate-diagnosis/1', 'wrong duplicate audit')
    planned = audit['proposed_manifest_only_deduplication'][role]
    groups = defaultdict(list)
    for e in rows: groups[e['model_tensors_sha256']].append(e)
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    registered = {g[0]['model_tensors_sha256']: g for g in audit['folds'][role]['duplicate_groups']}
    h.need(set(duplicates) == set(registered), 'unapproved duplicate group')
    removed, receipts = [], []
    fields = ('pair_id', 'stage', 'generator', 'label', 'recipe', 'sample_path', 'sample_sha256',
              'model_tensors_sha256', 'supervised_tensors_sha256')
    for digest, group in duplicates.items():
        ordered = sorted(group, key=lambda e: e['pair_id'])
        expected = sorted(registered[digest], key=lambda e: e['pair_id'])
        h.need([{k: e[k] for k in fields} for e in ordered] == [{k: e[k] for k in fields} for e in expected],
               'approved duplicate identity/content changed')
        h.need(len({e['supervised_tensors_sha256'] for e in group}) == 1 and
               len({e['label'] for e in group}) == 1, 'conflicting duplicate supervision')
        removed.extend(ordered[1:])
        receipts.append(dict(model_tensors_sha256=digest, retained_pair_id=ordered[0]['pair_id'],
                             excluded_pair_ids=[e['pair_id'] for e in ordered[1:]]))
    excluded = {e['pair_id'] for e in removed}
    h.need(excluded == set(planned['excluded_pair_ids']) and len(excluded) == len(removed), 'unapproved excluded IDs')
    retained = [e for e in rows if e['pair_id'] not in excluded]
    counts = (len(retained), sum(e['label'] for e in retained), sum(not e['label'] for e in retained))
    h.need(counts == FINAL_COUNTS[role] == (planned['pairs'], planned['positive'], planned['negative']), 'deduplicated count differs')
    validate_population(retained, role, removed=removed)
    return retained, dict(policy=planned['policy'], user_approved=True, raw_pairs=len(rows),
        actual_pairs=len(retained), positive=counts[1], negative=counts[2], groups=receipts,
        excluded_rows=[{k: e[k] for k in fields} for e in removed], no_files_deleted=True, no_replacement_generation=True)


def cross_fold(rows, allowed, forbidden):
    for role, entries in rows.items():
        for e in entries:
            families = (set(e['parent_ids']) | set(e['donor_parent_ids']))
            families = {f for f in families if not f.startswith('procedural-strip/')}
            h.need(families <= allowed[role] and not families & forbidden, 'primary/donor source outside heldout role')
    overlaps = {}
    for field in ('pair_id', 'sample_sha256', 'model_tensors_sha256', 'supervised_tensors_sha256'):
        overlaps[field] = len({e[field] for e in rows['cal']} & {e[field] for e in rows['select']})
    for field in ('parent_ids', 'base_pair_ids', 'fragment_ids'):
        sets = [{v for e in rows[r] for k in (field, 'donor_' + field) for v in e[k]} for r in rp.ROLES]
        overlaps[field] = len(sets[0] & sets[1])
    h.need(not any(overlaps.values()), 'cross-role overlap: ' + repr(overlaps))
    return overlaps


def committed_chain(path, task, plan_ref, extension, roots, ev):
    group = ev.read(path)
    h.need(group.get('status') == 'committed' and group['task'] == task and
           group['plan_sha256'] == plan_ref['sha256'] and group.get('source_replaced') is False and
           group.get('v14_fallback') is False and len(group['records']) == 2, 'commit/task mismatch')
    original = group
    seen = set()
    while 'reused_from' in group:
        reuse = group['reused_from']; ref = reuse['commit']; prior_path = Path(ref['path'])
        h.need(ref['path'] not in seen and len(seen) < 5, 'cyclic/excess reuse chain')
        seen.add(ref['path'])
        root = prior_path.parents[4]
        h.need(root in roots and prior_path == root/'augmented'/task['role']/task['stage']/'groups'/f"{task['slot']:05d}.json",
               'unregistered reused commit path')
        h.need(reuse['original_generation_plan'] == plan_ref and group.get('old_pixels_regenerated') is False,
               'reuse plan/pixel policy changed')
        prior = ev.ref(ref)
        h.need(all(prior[k] == group[k] for k in ('status', 'task', 'plan_sha256', 'records', 'audit_rows')) and
               prior.get('source_replaced') is False and prior.get('v14_fallback') is False, 'reuse content changed')
        group = prior; path = prior_path
    root = Path(path).parents[4]
    pixel_root = h._imported_group(group, task, root, task['role'], extension, ev)
    return original, pixel_root, len(seen)


def strict_rows(ref, role, ev, loader):
    subset = ev.ref(ref)
    h.need(subset['schema'] == 'mixed-heldout-strict-user-subset/1' and subset['role'] == role and
           subset['rows'] == 160 and subset['subset_uses_model_outputs'] is False and
           subset['pixel_generation_repeated'] is False, 'strict subset identity/policy differs')
    ev.mapping(subset['source_files'])
    seals = subset['original_normalization_seals']
    done, launch, actual = (ev.ref(seals[k]) for k in ('complete.json', 'launch.json', 'actual_process_return.json'))
    h.need(done['schema'] == 'strict-normalization-completion/1' and done['rows'] == 640 and
           done['status'] == 'verified_strict_component_only' and actual['returncode'] == 0 and
           actual['command'] == launch['command'] and actual['source'] == launch['source'], 'strict actual normalization failed')
    ev.mapping(done['bound_source_sha256']); tree_refs(done, ev)
    original = ev.ref(subset['original_expanded'])
    h.need(next(r for r in done['roles'] if r['role'] == role)['expanded'] == subset['original_expanded'] and
           len(original['entries']) == 320 and original['status'] == 'verified_component_only', 'strict original output differs')
    tree_refs(original['bound_receipts'], ev)
    raw = ev.ref(done['original_completion'])
    raw_return, raw_launch = ev.ref(raw['actual_process_return']), ev.ref(raw['controller'])
    h.need(raw_return['returncode'] == 0 and raw_return['command'] == raw_launch['command'], 'strict raw generation return mismatch')
    tree_refs(raw['roles'][role]['refs'], ev)
    cells = defaultdict(list)
    for e in original['entries']:
        cells[e['label'], e['recipe'], e['native_entry']['meta']['base'] if e['recipe'] == 'straight_J' else ''].append(e)
    selected = []
    for label in (False, True):
        up = (int(label) + rp.ROLES.index(role)) % 2
        counts = {('straight_M', ''): 20, ('straight_R', ''): 23-up, ('straight_J', 'torn_rachel'): 30,
                  ('straight_J', 'margin_fragment'): 4+up, ('straight_J', 'torn_strip'): 3}
        for (recipe, subtype), n in counts.items():
            ordered = sorted(cells[label, recipe, subtype], key=lambda e: hashlib.sha256(
                ('strict-reduce160/1:' + role + ':' + e['pair_id']).encode()).hexdigest())
            selected.extend(ordered[:n])
    h.need(selected == subset['entries'] and len(selected) == 160, 'strict metadata-only subset selection changed')
    result = copy.deepcopy(selected)
    for e in result:
        ev.file(e['sample_path'], e['sample_sha256'])
        native = e['native_entry']
        for p, s in (('proof_path', 'proof_sha256'),):
            if native.get(p): ev.file(native[p], native[s])
        sample, report = loader(Path(e['sample_path']))
        h.need(sample.pair_id == e['pair_id'] and bool(sample.label) == e['label'] and report['split'] == role,
               'strict actual row identity differs')
        d = hashlib.sha256()
        for key in ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b'):
            d.update(key.encode()); d.update(getattr(sample, key).tobytes())
        h.need(d.hexdigest() == e['model_input_sha256'], 'strict normalized input changed')
        e['model_tensors_sha256'] = h.tensor_identity(sample)
        e['supervised_tensors_sha256'] = h.tensor_identity(sample, include_supervision=True)
    return result


def supplement(ev):
    path = BASE/'shortfall_cal_gen3_q19_01'
    done = ev.read(path/'complete.json'); tree_refs(done, ev)
    h.need(done['status'] == 'bounded_batch_exhausted_shortfall_authorized' and done['unresolved_missing_pairs'] == 2 and
           done['provisional_selected_index'] is None and len(done['candidates']) == 4, 'supplement outcome differs')
    actual = ev.ref(done['actual_return']); h.need(actual['returncode'] == 0, 'supplement unexpected return')
    h.need(done['context']['sha256'] == '59326078e09ebae4dce02c2b8c39507bb731a89128f6a651f1dc7358bee33bcc', 'supplement context changed')
    for c in done['candidates']:
        h.need(c['status'] == 'rejected' and ev.ref(c['actual_return'])['returncode'] == 0 and
               ev.ref(c['outcome'])['status'] == 'rejected', 'supplement candidate unexpectedly failed or accepted')
    return ev.file(path/'complete.json')


def statistics(rows):
    result = population(rows)
    result['by_gen'] = {g: population([e for e in rows if e['generator'] == g]) for g in sorted({e['generator'] for e in rows})}
    result['by_stage'] = {s: population([e for e in rows if e['stage'] == s]) for s in (*rp.STAGES, 'strict_straight')}
    result['gen_stage_recipe_label'] = [{'gen': g, 'stage': s, 'recipe': r, 'label': l, 'count': n}
        for (g, s, r, l), n in sorted(Counter((e['generator'], e['stage'], e['recipe'], e['label']) for e in rows).items())]
    result['accepted_parameters'] = [dict(pair_id=e['pair_id'], stage=e['stage'], generator=e['generator'], recipe=e['recipe'],
        label=e['label'], detail=e['native_record']['detail'], mirror=e['native_record']['offline_paired_mirror'],
        partial_applied=e['native_record']['partial_applied']) if 'native_record' in e else
        dict(pair_id=e['pair_id'], stage=e['stage'], recipe=e['recipe'], label=e['label'], detail=e['native_entry']['meta']) for e in rows]
    result['max_augmented_views_per_base'] = max(Counter(p for e in rows for p in e['base_pair_ids']).values())
    return result


def work(output, root):
    ev = h.Evidence()
    dedup_path = BASE/'release_duplicate_audit_01'/'audit.json'
    dedup_audit = ev.read(dedup_path, DEDUP_AUDIT_SHA)
    dedup_receipts = {}
    launch = ev.read(root/'controller_launch.json'); actual = ev.read(root/'full_actual_return.json')
    h.need(launch['schema'] == 'mixed-heldout-parallel-controller/1' and actual['bindings'] == launch['bindings'] and
           actual['command'][2:] == ['--root', str(root), '--worker'] and
           Path(actual['command'][1]).name == 'heldout_parallel_build.py' and actual['command'][1] in launch['bindings'] and
           launch['started_unix'] <= actual['started_unix'] <= actual['finished_unix'], 'build launch/return mismatch')
    h.need(launch['workers'] == 32 and launch['cuda_visible_devices'] == '' and launch['models_loaded'] is False, 'wrong build resource/model policy')
    ev.mapping(launch['bindings'])
    h.need(ev.read(root/'full_shortfall.json')['status'] == 'shortfall', 'actual full shortfall missing')
    failures = list(root.glob('failure*.json')) + list((root/'shard_receipts').glob('*/failure.json'))
    projection = ev.ref(launch['reduction_plan']); plan_ref = projection['original_generation_plan']; plan = ev.ref(plan_ref)
    sources = ev.read(plan['source_plan_path'], plan['source_plan_sha256'])
    rp.validate(projection, plan, sources); validate_shards(launch['shards'], projection, plan)
    inventory = ev.ref(launch['frozen_source_inventory'])
    h.need(inventory['root'] == launch['frozen_runtime'], 'native runtime differs')
    admission = ev.ref(launch['original_pilot_admission']); ev.mapping(admission['bound_files'])
    extension = dict(admission=admission, old_root=Path(admission['old_build_root']), sources=sources)
    roots = {root, *map(Path, launch['reuse_roots']), extension['old_root']}
    cat, allowed, fragments, edges = h._catalog(sources, launch, ev)
    # approved_parents returns family -> parent records for each role.
    allowed = {role: set(records) for role, records in allowed.items()}
    parents = ev.read(sources['parent_plan_path'], sources['parent_plan_sha256'])
    forbidden = set().union(*map(set, parents['exclusion_families'].values()), set(parents.get('hard_exclusions', {})))
    h.need(not allowed['cal'] & allowed['select'] and not (allowed['cal'] | allowed['select']) & forbidden, 'parent plan exclusions differ')
    loader = h._native_loader(inventory)
    missing, all_rows, summaries, commit_counts = [], {}, {}, {}
    for role in rp.ROLES:
        banks, bounds = h._donor_bounds(role, sources['splits'][role], fragments, edges, ev)
        tasks = {(t['stage'], t['slot']): t for t in rp.chosen_tasks(projection, plan, role)}
        observed, entries, bases = set(), [], {}
        for shard in (s for s in launch['shards'] if s['role'] == role):
            receipt_root = root/'shard_receipts'/shard['name']
            done = ev.read(receipt_root/'complete.json')
            h.need(done['schema'] == 'mixed-sim-heldout-parallel-shard/1' and done['shard'] == shard and done['role'] == role and
                   done['planned_quotas'] == 45 and done['pilot_only'] is False and done['gpu_used'] is False and
                   done['model_outputs_used'] is False and done['reduction_plan'] == launch['reduction_plan'] and
                   done['original_generation_plan'] == plan_ref, 'shard identity/registration mismatch')
            for path, sha in done['frozen_source_sha256'].items():
                p = Path(path); h.need(p.is_relative_to(inventory['root']) and
                    inventory['files'].get(str(p.relative_to(inventory['root']))) == sha, 'generation used unbound source')
                ev.file(p, sha)
            exhausted = [f for f in done['failures'] if f['phase'] == 'quota_exhausted']
            h.need(done['status'] == ('shortfall' if exhausted else 'complete') and
                   done['admitted_pairs'] == len(done['records']) == 90-2*len(exhausted), 'shard completion/count mismatch')
            for failure in exhausted:
                key = (role, *failure['quota']); missing.append(key)
                h.need(failure['error'] == 'all 12 fixed candidates rejected', 'unrecognized quota failure')
                registered = [t for t in tasks.values() if (t['generator'], t['quota_slot'], t['stage']) == tuple(failure['quota'])]
                rejected = [f for f in done['failures'] if f.get('task') in registered]
                h.need(len(registered) == 12 and {json.dumps(f['task'], sort_keys=True) for f in rejected} ==
                       {json.dumps(t, sort_keys=True) for t in registered} and
                       all(f['phase'] in ('augmentation', 'baseline_geometry_exhausted') for f in rejected), 'shortfall does not prove twelve ordinary rejections')
            grouped = defaultdict(list)
            for rec in done['records']: grouped[rec['stage'], rec['baseline_slot']].append(rec)
            for key, records in grouped.items():
                task = tasks[key]; quota = (role, task['generator'], task['quota_slot'], task['stage'])
                h.need(task['generator'] == shard['generator'] and task['quota_slot'] in shard['quota_slots'] and
                       quota not in observed and len(records) == 2, 'duplicate/foreign quota ownership')
                observed.add(quota)
                group, pixel_root, depth = committed_chain(root/'augmented'/role/task['stage']/'groups'/f"{task['slot']:05d}.json",
                    task, plan_ref, extension, roots, ev)
                h.need(records == group['records'], 'shard rows differ from exact commit')
                audit = done['base_audit'][str(task['slot'])]; baseline_root = Path(audit['root'])
                h.need(baseline_root in {r/'baseline'/role for r in roots}, 'baseline root not admitted')
                baseline = ev.read(baseline_root/'groups'/f"{task['slot']:05d}.json", audit['group_sha256'])
                h.need(len(audit['audit_rows']) == 2 and {a['pair_id'] for a in audit['audit_rows']} ==
                       {e['pair_id'] for e in baseline['entries']}, 'baseline audit identity mismatch')
                source_audit = ev.read(receipt_root/'baseline_admissions'/f"{task['slot']:05d}.json")
                h.need(source_audit == audit, 'persisted baseline audit differs')
                tree_refs(audit, ev)
                for r in records:
                    entries.append(h._record(r, task, group, sources['splits'][role], cat, fragments, edges,
                        banks, bounds, root, role, ev, loader, sample_root=pixel_root,
                        baseline_root=baseline_root, numeric_mirror_schedule=True))
                bases[key] = depth
        expected = {(role, t['generator'], t['quota_slot'], t['stage']) for t in tasks.values()}
        h.need(expected-observed == {w for w in WAIVED if w[0] == role}, 'missing quota not exactly authorized')
        entries.extend(strict_rows(launch['strict_subsets'][role], role, ev, loader))
        entries.sort(key=lambda e: (e['stage'], e['generator'], e['pair_id']))
        validate_population(entries, role, allow_exact_duplicates=True)
        entries, dedup_receipts[role] = authorized_deduplication(entries, role, dedup_audit)
        all_rows[role] = entries; summaries[role] = statistics(entries)
        commit_counts[role] = dict(paired_groups=len(bases), reused_groups=sum(d > 0 for d in bases.values()))
        print(json.dumps(dict(role=role, rows=len(entries), status='normalized_no_pixel_replay')), flush=True)
    check_waivers(missing, actual['returncode'], failures)
    supplement_ref = supplement(ev)
    overlaps = cross_fold(all_rows, allowed, forbidden)
    files = ev.finish()
    evidence_ref = save(output/'evidence.json', dict(files=files, source_build_actual_return=actual,
        authorized_missing_quotas=[list(x) for x in sorted(WAIVED)], supplement=supplement_ref,
        original_builder_return_preserved=2, generated_pixels=0, pixel_replays=0, model_forwards=0))
    dedup_ref = save(output/'authorized_deduplication.json', dict(
        audit=dict(path=str(dedup_path), sha256=DEDUP_AUDIT_SHA),
        authorization='User approved the proposed exact-input manifest-only deduplication on 2026-10-03.',
        folds=dedup_receipts, original_files_unchanged=True, model_outputs_used=False))
    outputs = {}
    for role, rows in all_rows.items():
        lineage = dict(schema=MANIFEST_SCHEMA, role=role, entries=[{k: e[k] for k in ROW_FIELDS} for e in rows])
        _validate_manifest(lineage)
        compact_summary = {k:v for k,v in summaries[role].items() if k != 'accepted_parameters'}
        manifest = dict(schema_version='mixed-simulation-heldout/1', split=role, entries=rows,
                        summary=compact_summary, train_used=False, test_used=False, real_used=False,
                        original_training_or_test_modified=False, checkpoint_selection_performed=False)
        outputs[role] = dict(manifest=save(output/role/(role+'.json'), manifest),
                             lineage=save(output/role/'lineage.json', lineage),
                             statistics=save(output/role/'statistics.json', summaries[role]))
    return save(output/'verification.json', dict(schema='mixed-heldout-reduced-release/1', status='verified_data_release',
        rows=3183, raw_generated_rows=3196, desired_rows=3200, outputs=outputs, evidence=evidence_ref,
        authorized_deduplication=dedup_ref,
        folds={r:{k:v for k,v in s.items() if k != 'accepted_parameters'} for r,s in summaries.items()},
        commit_counts=commit_counts, cross_role_overlaps=overlaps, train_test_source_overlap=0,
        authorized_missing_quotas=[list(x) for x in sorted(WAIVED)], original_builder_actual_return=2,
        historical_gen23_base_edge_donor_provenance_complete=False, attempted_augmentation_donor_log_complete=False,
        donor_bounds_are_conservative_not_actual_counts=True, frozen_label_semantics_not_repaired=True,
        task3_overlay_applied=False, no_pixels_regenerated=True, no_model_inference=True,
        endpoint_cal_inference_authorized_by_user=True, checkpoint_selection_performed=False))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--build-root', type=Path, required=True); ap.add_argument('--worker', action='store_true')
    a=ap.parse_args(); root=a.root.resolve(); build=a.build_root.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if a.worker:
        try: work(root, build)
        except BaseException as error:
            save(root/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), time=time.time())); raise
        return
    root.mkdir(parents=True, exist_ok=False)
    command=[sys.executable, '-m', 'model_selection_v2.heldout_reduced_release', '--root', str(root), '--build-root', str(build), '--worker']
    source={str(p.resolve()):h.sha(p) for p in Path(__file__).parent.glob('*.py')}
    launch=dict(command=command, source=source, started_unix=time.time(), pid=os.getpid(), original_build=str(build))
    save(root/'launch.json', launch)
    with (root/'worker.log').open('x') as log:
        result=subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    actual=dict(command=command, source=source, returncode=result.returncode, started_unix=launch['started_unix'], finished_unix=time.time())
    save(root/'actual_return.json', actual)
    if result.returncode: raise SystemExit(result.returncode)
    h.need(all(h.sha(p)==sha for p,sha in source.items()), 'publisher source changed')
    verification=root/'verification.json'
    save(root/'complete.json', dict(status=RELEASE_STATUS, rows=3183,
        verification=dict(path=str(verification), sha256=h.sha(verification)),
        actual_return=dict(path=str(root/'actual_return.json'), sha256=h.sha(root/'actual_return.json')),
        finished_unix=time.time()))
    print(json.dumps(dict(status=RELEASE_STATUS, root=str(root), rows=3183)), flush=True)


if __name__ == '__main__': main()

"""Read-only normalization/admission of the new strict CAL/SELECT component.

Only this component is admitted here, never the entire mixed selection set.
No files are rewritten; native tokens/entries and raw provenance are retained.
"""
from collections import Counter
import copy
import hashlib
import inspect
import json
from pathlib import Path
import types
import uuid

from . import heldout_straight as h
from .protocol import ROW_FIELDS


def _read(receipt):
    h._receipt(receipt['path'], receipt['sha256'])
    return json.loads(Path(receipt['path']).read_text())


def _audit_equal(replayed, persisted):
    # Frozen nick_intervals returns tuples; its immutable JSON audit necessarily
    # stores those as arrays/lists. Compare exact JSON content, not Python
    # container classes. Numeric changes remain differences; NaN is rejected.
    return h.digest(replayed) == h.digest(persisted)


def _source_families(meta):
    result = set()
    if isinstance(meta, dict):
        if meta.get('source_family'):
            result.add(meta['source_family'])
        for value in meta.values(): result.update(_source_families(value))
    elif isinstance(meta, list):
        for value in meta: result.update(_source_families(value))
    return result


def normalize_identities(plan, entry, native_tokens):
    """Explicit provenance identities, not inference from generator-like IDs."""
    role, dataset, pair_id = plan['role'], plan['dataset_id'], entry['pair_id']
    h.require(role in ('cal', 'select') and entry['split'] == role, 'strict role mismatch')
    h.require(type(entry.get('label')) is bool, 'explicit Boolean label required')
    h.require(set(native_tokens) == {'a', 'b'} and all(isinstance(x, str) and x for x in native_tokens.values()),
              'both native fragment tokens required')
    donors = entry['attempted_donor_references']
    allowed = {r['path']: r for r in plan['source_rows']}
    for donor in donors:
        h.require(donor.get('path') in allowed and donor == allowed[donor['path']], 'foreign attempted donor')
    accepted = sorted(_source_families(entry.get('meta', {})))
    all_families = sorted({d['source_family'] for d in donors})
    h.require(set(accepted) <= set(all_families), 'accepted material donor absent from attempted lineage')
    # Procedural strips have no original manuscript image. Their parent IDs
    # identify independent RNG construction, never a fictitious source family.
    synthetic = []
    if not accepted:
        seed = plan['masterseeds'][role]
        identity = h.digest([dataset, role, seed, entry['id'], entry['label'], entry['tries']])
        sides = ['shared'] if entry['label'] else ['a', 'b']
        synthetic = [f'procedural-strip/{role}/{seed}/{identity}/{side}' for side in sides]
    base = entry.get('pre_damage_pair_sha256')
    if entry['label']:
        h.require(isinstance(base, str) and len(base) == 64, 'positive pre-damage identity missing')
        base_ids = ['strict-cut-sha256/' + base]
    else:
        # A negative has no common seam/base pair GT. Its construction identity
        # is explicit and separate from positive pre-damage pair identities.
        base_ids = ['procedural-negative-pair/' + h.digest(
            [dataset, role, plan['masterseeds'][role], entry['id'], entry['tries'], entry['meta']])]
    fields = dict(pair_id=pair_id, sample_sha256=entry['sample_sha256'], stage='strict_straight',
                  generator='straight_strip', label=entry['label'], parent_ids=accepted + synthetic,
                  base_pair_ids=base_ids,
                  fragment_ids=[f'derived/{dataset}/{role}/{pair_id}/{side}' for side in ('a', 'b')],
                  donor_parent_ids=all_families, donor_base_pair_ids=[],
                  donor_fragment_ids=sorted({d['fragment_token'] for d in donors}))
    h.require(set(fields) == ROW_FIELDS and fields['parent_ids'], 'protocol projection fields differ')
    for name in ('parent_ids', 'base_pair_ids', 'fragment_ids', 'donor_parent_ids', 'donor_base_pair_ids', 'donor_fragment_ids'):
        h.require(len(fields[name]) == len(set(fields[name])), 'duplicate explicit identity: ' + name)
    return dict(fields, native_fragment_tokens=dict(native_tokens), synthetic_parent_ids=synthetic,
                donor_base_pair_note='Material donors are single fragments, not labeled base pairs; none invented.')


def _check_parent_bytes(parent_plan, families):
    rows = {r['family']: r for group in parent_plan['folds'].values() for r in group}
    refs = {}
    for family in sorted(families):
        h.require(family in rows, 'missing parent provenance')
        parent = rows[family]
        image = parent.get('parent_image', {})
        h.require(image.get('has_embedded_original_alpha') is True and image.get('alpha_sha256'),
                  'original parent alpha provenance missing')
        h._receipt(image['path'], image['file_sha256'])
        variants = parent.get('allowed_same_family_variants', [])
        for variant in variants:
            h._receipt(variant['path'], variant['file_sha256'])
        refs[family] = copy.deepcopy(parent)
    return refs


def _replay_targets(backend, entry, sample, proof, seed, builder):
    import numpy as np
    s, context = backend.supervision, backend.context
    context.update(donors={}, rejections=Counter(), proof=None)
    aa, bb, trace = s.replay_final_attempt(backend.reference, entry, seed)
    for name, value in [('final_parent_a', aa), ('final_parent_b', bb),
                        ('cut_a', context['cut'][0]), ('cut_b', context['cut'][1]),
                        ('post_misfit_a', context['misfit'][2]), ('post_misfit_b', context['misfit'][3])]:
        np.testing.assert_array_equal(value, s.unpack(proof, name))
    original, reason = backend.reference.finalize(aa, bb, copy.deepcopy(entry['meta']), True)
    h.require(reason is None, 'accepted strict geometry no longer replays')
    ia, ib, t, pa, va, pb, vb, ta, tb = original
    for field, value in [('mask_a', ia[None].astype(np.float32)), ('mask_b', ib[None].astype(np.float32)),
                         ('points_rc_a', pa.astype(np.float32)), ('points_rc_b', pb.astype(np.float32)),
                         ('contour_valid_a', va), ('contour_valid_b', vb), ('translation_a_to_b_rc', t)]:
        np.testing.assert_array_equal(getattr(sample, field), value)
    np.testing.assert_array_equal(proof['original_target_a'], ta)
    np.testing.assert_array_equal(proof['original_target_b'], tb)
    raw = types.SimpleNamespace(points_rc_a=pa, points_rc_b=pb, contour_valid_a=va,
                                contour_valid_b=vb, target_a=ta, target_b=tb)
    target, audit = s.build_supervision(raw, proof, trace, builder.projected_interval_damage)
    np.testing.assert_array_equal(sample.target_a, target['target_a'])
    np.testing.assert_array_equal(sample.target_b, target['target_b'])
    h.require(audit['correspondence_count'] >= 8, 'too few healthy targets')
    h.require(not backend.geometry.violations(entry['recipe'][-1], backend.metrics.measure(sample)),
              'fixed strict geometry predicate no longer passes')
    return audit


def normalize_strict_release(completion_receipt_path, expected_completion_sha, role, *,
                             load_sample, loader_source_receipt, target_builder_path,
                             expected_target_builder_sha):
    """Read and verify an actual complete320 role, then return expanded rows.

    ``entries`` extend the strict protocol row with paths/provenance. Project
    each entry onto ``protocol.ROW_FIELDS`` for the separately frozen protocol
    manifest. Parent IDs are normalized source-family strings from parent_plan;
    callers must use that same family vocabulary for other stage manifests.
    """
    import numpy as np
    h.require(role in ('cal', 'select'), 'only new CAL/SELECT strict roles permitted')
    complete_ref = h._receipt(completion_receipt_path, expected_completion_sha)
    complete = _read(complete_ref)
    h.require(complete.get('status') == 'complete_strict640_only' and complete.get('rows') == 640
              and complete.get('full_mixed6400_complete') is False, 'full strict640 completion receipt required')
    actual, launch = _read(complete['actual_process_return']), _read(complete['controller'])
    h.require(actual['returncode'] == 0 and actual['command'] == launch['command']
              and actual['source_sha256'] == launch['controller_sha256'], 'actual launch/return identity mismatch')
    info = complete['roles'][role]
    bound = {name: _read(receipt) for name, receipt in info['refs'].items()}
    plan, manifest, audit = bound['plan.json'], bound['manifest.json'], bound['independent_audit.json']
    material, summary = bound['materialization_receipt.json'], bound['verified_summary.json']
    h.validate_plan(plan)
    h.require(plan['role'] == manifest['split'] == role and info['rows'] == 320 and not manifest['failed']
              and len(manifest['entries']) == audit['rows'] == 320, 'strict role/population mismatch')
    h.require(material['production_backend'] is True and material['pilot'] is False
              and material['generated'] == material['expected'] == 320 and material['failed'] == 0,
              'not successful production320')
    refs = info['refs']
    h.require(material['manifest_sha256'] == refs['manifest.json']['sha256'] == audit['manifest_sha256']
              and material['plan_sha256'] == refs['plan.json']['sha256'] == manifest['plan_sha256']
              and audit['materialization_receipt_sha256'] == refs['materialization_receipt.json']['sha256']
              and audit['status'] == 'passed_integrity_supervision_and_fixed_geometry', 'strict receipt chain mismatch')
    h.require(loader_source_receipt == summary['loader'], 'loader source must match actual independent audit')
    h._receipt(loader_source_receipt['path'], loader_source_receipt['sha256'])
    h.require(Path(inspect.getsourcefile(load_sample)).resolve() == Path(loader_source_receipt['path']).resolve(),
              'injected NPZ loader is not the bound native implementation')
    target_ref = h._receipt(target_builder_path, expected_target_builder_sha)
    h.require(target_ref == audit['target_builder'], 'independent target builder differs')
    builder = h._load('_strict_release_target_' + uuid.uuid4().hex, target_ref['path'])
    backend = h.FrozenBackend(plan, audit_only=True)
    observed = {r['pair_id']: r for r in audit['records']}
    h.require(len(observed) == 320, 'duplicate audited identities')
    parents = _read(plan['parent_plan'])
    parent_refs = _check_parent_bytes(parents, {r['source_family'] for r in plan['source_rows']})
    entries, seen, base_ids, tensor_hashes = [], set(), set(), set()
    for entry in manifest['entries']:
        h.require(entry['pair_id'] in observed and entry['pair_id'] not in seen, 'unaudited/duplicate row')
        seen.add(entry['pair_id'])
        job = (entry['recipe'][-1], int(entry['label']), 0, role, plan['output_new'], plan['masterseeds'][role])
        h.validate_entry(plan, job, entry)
        sample, report = load_sample(entry['sample_path'])
        h.require(sample.pair_id == entry['pair_id'] and bool(sample.label) == entry['label']
                  and bool(sample.translation_valid) == entry['label'], 'actual NPZ ID/label differs')
        h.require(report['split'] == role and report['supervision_revision'] == 'codex-v42-geometry-explicit-targets/1',
                  'actual NPZ role or supervision revision differs')
        d = hashlib.sha256()
        for name in ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b'):
            value = np.ascontiguousarray(getattr(sample, name)); d.update(name.encode()); d.update(value.tobytes())
        tensor_sha = d.hexdigest()
        h.require(tensor_sha == observed[entry['pair_id']]['model_input_sha256'] and tensor_sha not in tensor_hashes,
                  'actual model input hash mismatch/duplicate')
        tensor_hashes.add(tensor_sha)
        supervision = dict(correspondence_count=0)
        if entry['label']:
            with np.load(entry['proof_path'], allow_pickle=False) as archive:
                proof = {k: archive[k] for k in archive.files}
            base = backend.identity.base_pair_sha256(backend.supervision.unpack(proof, 'cut_a'),
                                                     backend.supervision.unpack(proof, 'cut_b'))
            h.require(base == entry['pre_damage_pair_sha256'] == observed[entry['pair_id']]['pre_damage_pair_sha256']
                      and base not in base_ids, 'pre-augmentation proof mismatch/duplicate')
            base_ids.add(base)
            supervision = _replay_targets(backend, entry, sample, proof, plan['masterseeds'][role], builder)
            h.require(_audit_equal(supervision, observed[entry['pair_id']]['target_audit']), 'replayed target audit differs')
        else:
            h.require(np.all(sample.target_a == -1) and np.all(sample.target_b == -1), 'negative has fabricated correspondence')
        normalized = normalize_identities(plan, entry, {'a': sample.fragment_a_token, 'b': sample.fragment_b_token})
        donor_families = normalized['donor_parent_ids']
        normalized.update(role=role, recipe=entry['recipe'], sample_path=entry['sample_path'],
            model_input_sha256=tensor_sha, native_entry=copy.deepcopy(entry),
            upstream_parent_provenance={f: parent_refs[f] for f in donor_families},
            corrected_supervision_audit=supervision,
            target_storage='target_a/target_b inside sample NPZ; original targets and pre-damage pixels inside positive proof NPZ')
        entries.append(normalized)
    counts = Counter((e['recipe'], e['label']) for e in entries)
    h.require(all(counts['straight_' + k, label] == n for k, n in h.COUNTS.items() for label in (True, False)),
              'strict per-label M/J/R quota differs')
    for label in (True, False):
        bases = Counter(e['native_entry']['meta']['base'] for e in entries if e['recipe'] == 'straight_J' and e['label'] == label)
        h.require(bases == {'torn_rachel': 60, 'margin_fragment': 9, 'torn_strip': 6}, 'strict J subtype quota differs')
    h.validate_plan(plan)
    _read(complete_ref)
    for receipt in info['refs'].values(): h._receipt(receipt['path'], receipt['sha256'])
    return dict(schema='normalized-strict-heldout-component/1', status='verified_component_only', role=role,
                entries=entries, rows=320, provenance_summary=dict(
                    parent_ids=sorted({x for e in entries for x in e['parent_ids']}),
                    donor_parent_ids=sorted({x for e in entries for x in e['donor_parent_ids']}),
                    donor_fragment_ids=sorted({x for e in entries for x in e['donor_fragment_ids']}),
                    synthetic_parent_count=len({x for e in entries for x in e['synthetic_parent_ids']})),
                bound_receipts=dict(completion=complete_ref, loader=loader_source_receipt,
                                    target_builder=target_ref, role_receipts=info['refs']),
                full_mixed6400_complete=False, inputs_modified=False, labels_algorithm_modified=False,
                model_inference=False, checkpoint_selected=False, head_training_started=False)

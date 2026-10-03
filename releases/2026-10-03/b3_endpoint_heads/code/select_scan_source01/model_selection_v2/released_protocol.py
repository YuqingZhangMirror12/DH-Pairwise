"""Bind the already-verified published_03 release without inventing TRAIN rows.

Exclusion provenance comes from the byte-bound publication verification. This
does not rerun generation, claim complete historical donor tracing, or repair
point supervision. The decision rule remains the pre-existing v2 rule.
"""
from pathlib import Path
import hashlib
import json

from .protocol import (DEFAULT_GENERATORS, LOSS_SEMANTICS, LAYOUT_TOLERANCE,
    RULE_ID, SELECTION_KIND, STAGE_WEIGHTS, bind_manifest, canonical, digest,
    require, _validate_binding, _lineages)

SCHEMA = 'model-selection-published-release-protocol/1'
COUNTS = {'select': (1596, 796, 800), 'cal': (1587, 793, 794)}
VERIFICATION_SHA = 'a8e7da99581e0ac762e8198af9c498cc52c7edb97cbb94129581d8149cff7a06'


def checked(ref):
    require(set(ref) == {'path', 'sha256'} and Path(ref['path']).is_absolute(), 'exact absolute receipt required')
    raw = Path(ref['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'published receipt changed')
    return json.loads(raw)


def rule():
    return dict(id=RULE_ID, selection_kind=SELECTION_KIND, stage_weights=STAGE_WEIGHTS,
        within_stage='equal_generator_macro', layout_population='positive_pairs',
        loss_population='all_pairs', loss_semantics=LOSS_SEMANTICS,
        absolute_layout_tolerance=LAYOUT_TOLERANCE,
        band_rule='macro_layout >= maximum_macro_layout - absolute_layout_tolerance',
        within_band_order=['minimum_macro_loss', 'maximum_macro_layout', 'earliest_update'],
        coverage_affects_selection=False)


def audit_bindings(bindings, publication, verify_files):
    require(set(bindings) == {'select', 'cal'}, 'published SELECT and CAL lineage required')
    lineage = {}
    for role, counts in COUNTS.items():
        b = bindings[role]; rows = _validate_binding(b, role, verify_files)
        require((len(rows), sum(r['label'] for r in rows), sum(not r['label'] for r in rows)) == counts,
                'published count/label population differs')
        require({k: b[k] for k in ('path', 'sha256')} == publication['outputs'][role]['lineage'],
                'lineage not bound to verified publication')
        require({(r['stage'], r['generator']) for r in rows} ==
                {(s, g) for s, gs in DEFAULT_GENERATORS.items() for g in gs}, 'missing stratum')
        for stage, gens in DEFAULT_GENERATORS.items():
            for gen in gens:
                require({r['label'] for r in rows if (r['stage'], r['generator']) == (stage, gen)} ==
                        {True, False}, 'stratum missing a label')
        lineage[role] = _lineages(rows)
    for key in lineage['select']:
        require(not lineage['select'][key] & lineage['cal'][key], 'published lineage overlap')
    for key in ('pair_id', 'sample_sha256'):
        require(not {r[key] for r in bindings['select']['manifest']['entries']} &
                    {r[key] for r in bindings['cal']['manifest']['entries']}, 'published identity overlap')


def freeze_release_protocol(publication_ref, updates):
    publication = checked(publication_ref)
    bindings = {r: bind_manifest(**dict(path=publication['outputs'][r]['lineage']['path'],
        expected_sha256=publication['outputs'][r]['lineage']['sha256'])) for r in COUNTS}
    plan = canonical(dict(schema=SCHEMA, state='frozen_before_reports', reports_seen_at_freeze=0,
        candidate_updates=updates, rule=rule(), required_generators=DEFAULT_GENERATORS,
        manifest_bindings=bindings, publication=publication_ref, matcher_selection_role='select',
        real_used=False, test_used_for_selection=False, cal_used_for_matcher_selection=False,
        exclusion_evidence='bound verified publication; no fabricated TRAIN/TEST lineage rows',
        historical_gen23_base_edge_donor_provenance_complete=False,
        frozen_label_semantics_not_repaired=True, task3_overlay_applied=False,
        correlated_same_fold_augmentation_views=True))
    plan['sha256'] = digest(plan)
    validate_release_protocol(plan, verify_files=True)
    return plan


def validate_release_protocol(plan, verify_files=False):
    body = {k: v for k, v in plan.items() if k != 'sha256'}
    require(digest(body) == plan.get('sha256'), 'published protocol changed')
    require(plan.get('schema') == SCHEMA and plan.get('state') == 'frozen_before_reports'
            and plan.get('reports_seen_at_freeze') == 0 and plan.get('rule') == rule()
            and plan.get('required_generators') == DEFAULT_GENERATORS, 'selection policy changed')
    require(plan.get('matcher_selection_role') == 'select' and plan.get('real_used') is False
            and plan.get('test_used_for_selection') is False and plan.get('cal_used_for_matcher_selection') is False,
            'SELECT-only decision boundary required')
    require(plan.get('historical_gen23_base_edge_donor_provenance_complete') is False
            and plan.get('frozen_label_semantics_not_repaired') is True
            and plan.get('task3_overlay_applied') is False
            and plan.get('correlated_same_fold_augmentation_views') is True, 'publication caveat removed')
    updates = plan.get('candidate_updates')
    require(isinstance(updates, list) and updates and all(type(u) is int and u > 0 for u in updates)
            and updates == sorted(set(updates)), 'candidate set required')
    require(plan['publication']['sha256'] == VERIFICATION_SHA, 'only published_03 admitted here')
    publication = checked(plan['publication'])
    require(publication.get('status') == 'verified_data_release' and publication['rows'] == 3183
            and publication['train_test_source_overlap'] == 0
            and all(type(n) is int and n == 0 for n in publication['cross_role_overlaps'].values())
            and publication['task3_overlay_applied'] is False, 'publication isolation/identity differs')
    audit_bindings(plan['manifest_bindings'], publication, verify_files)
    return plan

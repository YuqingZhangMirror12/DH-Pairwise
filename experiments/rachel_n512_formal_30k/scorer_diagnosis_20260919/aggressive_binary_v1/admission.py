"""Read-only admission for experiment 3. This module grants no user approval.

The full-generation writer must supply these receipts after human review.
Neither a ten-pair probe nor the old v14 contract can start this experiment.
"""
import hashlib
import json
from pathlib import Path

REVISION = 'aggressive-v17-depth35/1'
COUNTS = {'train': 24000, 'cal': 1500, 'select': 1500, 'test': 3000}
REVIEW_GROUPS = (
    'clean', 'mild', 'wave', 'wave_weak', 'local_abrupt', 'local_abrupt_weak',
    'local_gradual', 'local_gradual_weak', 'gaps', 'gaps_weak', 'partial_end',
    'partial_middle', 'mirror_h', 'mirror_v', 'native', 'gen5', 'union_tiny',
    'unequal', 'negative_cross_gen', 'negative_cross_parent_same_gen',
    'negative_same_parent_nonadjacent', 'trim_one', 'trim_both', 'light70',
    'crop_smaller', 'crop_larger', 'notch_k4')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def need(condition, message):
    if not condition:
        raise ValueError(message)


def bound(spec):
    need(sha(spec['path']) == spec['sha256'], 'bound receipt changed: ' + spec['path'])
    return read(spec['path'])


def validate_approval(path):
    approval = read(path)
    need(approval.get('schema') == 'aggressive-data-human-approval/2'
         and approval.get('status') == 'approved'
         and approval.get('scope') == 'full30k_after_full_per_type10_review'
         and approval.get('evidence_kind') == 'explicit_user_message_in_current_thread'
         and approval.get('user_approval_text') and approval.get('approved_at_iso'),
         'explicit approval AFTER the complete per-type review is required')
    need(hashlib.sha256(approval['user_approval_text'].encode()).hexdigest()==approval.get('user_approval_text_sha256')
         and approval.get('fallback_authorization_text')
         and hashlib.sha256(approval['fallback_authorization_text'].encode()).hexdigest()==approval.get('fallback_authorization_sha256')
         and approval.get('authorized_delta',{}).get('after')==[5,35]
         and approval.get('authorized_delta',{}).get('retain_original_v14_on_failed_new_augmentation') is True,
         'approved 5–35px and explicit v14 fallback policy must be bound')
    need(approval.get('augmentation_revision') == REVISION, 'review revision changed')
    generation = bound(approval['review_generation'])
    audit = bound(approval['review_pixel_audit'])
    rendered = bound(approval['review_rendered'])
    need(generation.get('status') == 'generated_pending_pixel_audit_and_review'
         and generation.get('missing') == {}, 'a probe is not the complete review set')
    n = generation['pairs']
    need(audit.get('status') == 'passed' and audit.get('pairs') == n
         and not audit.get('errors') and len(audit.get('receipts', [])) == n,
         'complete independent review pixel audit is required')
    receipts = audit['receipts']
    need(all(r.get('status') == 'passed' for r in receipts), 'review audit contains failures')
    audited_ids = {r['id'] for r in receipts}
    need(len(audited_ids) == n, 'duplicate review audit IDs')
    need(set(rendered['groups']) == set(REVIEW_GROUPS), 'all 27 review groups are required')
    rendered_ids = {r['id'] for r in rendered['rows']}
    need(len(rendered_ids) == len(rendered['rows']) and rendered_ids <= audited_ids,
         'rendered review contains unaudited or duplicate rows')
    for group in rendered['groups'].values():
        need(len(group['ids']) == 10 and len(set(group['ids'])) == 10
             and set(group['ids']) <= rendered_ids, 'ten audited images per review group required')
    side = audit['positive_size_counts']
    need(side['smaller'] + side['larger'] == n // 2
         and side['smaller'] * 10 == (n // 2) * 7, 'accepted review 70/30 quota changed')
    return dict(path=str(Path(path).resolve()), sha256=sha(path),
                augmentation_revision=REVISION,
                review_pixel_audit_sha256=approval['review_pixel_audit']['sha256'],
                review_rendered_sha256=approval['review_rendered']['sha256'])


def validate_data(contract_path, calibration_path, approval_path):
    approval = validate_approval(approval_path)
    contract = read(contract_path)
    need(contract.get('status') == 'passed' and contract.get('source_disjoint') is True
         and contract.get('augmentation_revision') == REVISION,
         'completed NEW aggressive source-isolated data contract required; not v14')
    need(contract.get('review_approval_sha256') == approval['sha256'], 'full data approval differs')
    need(contract.get('online_mirror_probability') == 0, 'no extra online augmentation')
    specs = dict(train=contract['train'], cal=contract['validation']['cal_mixed'],
                 select=contract['validation']['select_mixed'], test=contract['test']['mixed'])
    hashes, all_ids = {}, set()
    for name, spec in specs.items():
        manifest = bound(spec)
        rows = manifest['entries']
        need(manifest.get('split') == name and manifest.get('augmentation_revision') == REVISION,
             'wrong manifest population or augmentation revision: ' + name)
        need(manifest.get('not_full_training_dataset') is not True,
             'review pilot must never become a training manifest')
        ids = {r['pair_id'] for r in rows}
        need(len(rows) == COUNTS[name] and len(ids) == len(rows) and not (all_ids & ids),
             'missing or duplicate materialized pairs: ' + name)
        need(all(r['label'] in (0, 1, False, True) for r in rows)
             and sum(bool(r['label']) for r in rows) * 2 == len(rows),
             'each split must retain equal positive/negative counts')
        all_ids |= ids
        hashes[name] = spec['sha256']
    families = contract['source_families']
    used = set()
    for name in COUNTS:
        sources = set(families[name])
        need(sources and not (sources & used), 'manuscript source leakage: ' + name)
        used |= sources
    heldout = set().union(*(set(families[k]) for k in ('cal', 'select', 'test')))
    need(not (heldout & set(families['historical_s7_train']))
         and not (heldout & set(families['donor_train'])), 'historical TRAIN/donor leakage')
    full = bound(contract['aggressive_full_audit'])
    need(full.get('schema') == 'aggressive-v17-full-audit/1' and full.get('status') == 'passed'
         and full.get('checked_pairs') == 30000 and full.get('failures') == 0
         and full.get('manifest_sha256') == hashes
         and full.get('review_approval_sha256') == approval['sha256']
         and full.get('source_disjoint') is True
         and full.get('endpoint_and_area_pixel_audit') is True
         and full.get('target_and_donor_audit') is True
         and full.get('fallback_numerical_identity_audit') is True,
         'all 30K pixels, targets, endpoints, area and sources must be audited')
    calibration = read(calibration_path)
    need(calibration.get('schema') == 's7-consensus-train-geometry/2'
         and calibration.get('status') == 'complete'
         and calibration.get('pairs') == 12000
         and calibration.get('inherited_edges', 0) > 0
         and calibration.get('contract_sha256') == sha(contract_path)
         and calibration.get('manifest_sha256') == contract['train']['archive_manifest_sha256']
         and sha(contract['train']['archive_manifest']) == contract['train']['archive_manifest_sha256'],
         'recompute geometry on this NEW TRAIN only; old 0..9px calibration is invalid')
    # The requested final two-sided gap is not an isotropic pose tolerance,
    # nor a hardcoded replacement for the measured normal-damage p99.
    return dict(approval=approval, manifest_sha256=hashes,
                full_audit_sha256=contract['aggressive_full_audit']['sha256'],
                contract_sha256=sha(contract_path), geometry_sha256=sha(calibration_path),
                geometry_from_new_train_only=True, test_used_for_training_or_selection=False)

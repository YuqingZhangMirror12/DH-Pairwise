"""Make a new training catalog from completed, independently audited releases.

This is not a geometry audit, data generator, budget choice or GPU launcher.
Original archives remain unchanged. Pure Matcher inputs and effective training
supervision are different identities, including the clean-only pose auxiliary.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

from .catalog import inspect_sample, deduplicate_basic
from .exposure import STAGES, digest, is_sha

HARD_COUNTS = {'v17.5': 6000, 'v18': 3000}
REVISION = 'aggressive-v17-depth35/1'


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Bindings:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve(); signature = file_sha(path)
        require(expected is None or (is_sha(expected) and expected == signature),
                'bound artifact differs: ' + str(path))
        require(str(path) not in self.files or self.files[str(path)] == signature,
                'artifact changed during admission: ' + str(path))
        self.files[str(path)] = signature
        return path

    def read(self, path, expected=None):
        path = self.bind(path, expected)
        value = json.loads(path.read_text())
        require(file_sha(path) == self.files[str(path)], 'artifact changed while reading')
        return value

    def verify(self):
        for path, expected in self.files.items():
            require(file_sha(path) == expected, 'admission input changed: ' + path)


def indexed(rows, key, description):
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), 'duplicate ' + description)
    return result


def original_key(row):
    return str(row['source_root']) + '::' + str(row['source_pair_id'])


def normalize_archive_row(row, root):
    result = dict(row)
    result['artifact_path'] = str((root / row['artifact_path']).resolve())
    return result


def basic_rows(root, complete, contract_path, bindings):
    """Follow original v17 full-audit -> group-receipt bindings, not a pilot."""
    contract = bindings.read(contract_path)
    require(contract['status'] == 'passed' and contract['source_disjoint'] is True
            and contract['augmentation_revision'] == REVISION
            and contract['online_mirror_probability'] == 0., 'wrong v17 data contract')
    spec = contract['train']; original_path = bindings.bind(spec['path'], spec['sha256'])
    archive_path = bindings.bind(spec['archive_manifest'], spec['archive_manifest_sha256'])
    original = bindings.read(original_path)
    require(original == bindings.read(archive_path), 'TRAIN and archive manifests differ')
    require(original['split'] == 'train' and original['augmentation_revision'] == REVISION,
            'not original v17 TRAIN')
    rows = original['entries']; by_id = indexed(rows, 'pair_id', 'basic Pair ID')
    require(len(rows) == spec['pairs'] == spec['pair_count'], 'original basic row count differs')
    require(Counter(r['label'] for r in rows) == {True: spec['positives'], False: spec['negatives']},
            'original basic labels differ')
    full_spec = contract['aggressive_full_audit']
    full = bindings.read(full_spec['path'], full_spec['sha256'])
    require(full['status'] == 'passed' and full['failures'] == 0
            and full['source_disjoint'] is True and full['target_and_donor_audit'] is True
            and full['endpoint_and_area_pixel_audit'] is True
            and full['fallback_numerical_identity_audit'] is True
            and full['manifest_sha256']['train'] == spec['sha256'], 'v17 full audit failed or stale')
    split_spec = full['split_audits']['train']
    audit = bindings.read(split_spec['path'], split_spec['sha256'])
    require(audit['status'] == 'passed' and audit['checked_pairs'] == len(rows)
            and audit['manifest_sha256'] == spec['sha256']
            and audit['actual_loader_and_supervision_checked_all'] is True
            and audit['all_pixels_and_endpoints_checked'] is True, 'v17 TRAIN pixel audit differs')
    receipts = []; group_paths = set(); train_root = original_path.parent
    for group_spec in audit['group_receipts']:
        path = bindings.bind(group_spec['path'], group_spec['sha256'])
        require(path.parent == train_root / 'audits' and path not in group_paths,
                'duplicate or foreign group audit path')
        group_paths.add(path); group = bindings.read(path)
        require(group['status'] == 'passed' and len(group['rows']) == 2, 'incomplete v17 group audit')
        bindings.bind(train_root / 'groups' / path.name, group['group_sha256'])
        receipts.extend(group['rows'])
    checks = indexed(receipts, 'id', 'basic audit ID')
    require(set(checks) == set(by_id) and all(r['status'] == 'passed' for r in receipts),
            'v17 audit membership differs')
    exclusion = bindings.read(root / 'curriculum_base_exclusion.json', complete['base_exclusion_sha256'])
    keys = exclusion['source_pair_keys']
    require(exclusion['status'] == 'full_generation_passed' and len(keys) == len(set(keys)),
            'incomplete or duplicated original-pair exclusion')
    reference = bindings.read(root / 'v17_curriculum_base_reference_manifest.json')
    require(Path(reference['archive_manifest']).resolve() in {original_path, archive_path},
            'basic reference points to another original archive')
    bindings.bind(reference['archive_manifest'], reference['archive_sha256'])
    expected = [normalize_archive_row(r, train_root) for r in rows if original_key(r) not in set(keys)]
    actual = indexed(reference['entries'], 'pair_id', 'basic reference Pair ID')
    require(actual == indexed(expected, 'pair_id', 'expected basic Pair ID'),
            'reference is not exactly the original rows minus hard-stage base identities')
    require(reference['original_rows'] == len(rows)
            and reference['excluded_rows'] == len(rows) - len(expected)
            and reference['training_started'] is False, 'basic reference accounting differs')
    return expected, checks, set(keys), len(rows), reference['excluded_rows']


def inspect_training_row(stage, row, audit, loader=None):
    inspected = inspect_sample(stage, row, audit, loader)
    recipe = row.get('recipe')
    require(isinstance(recipe, str) and bool(recipe), 'explicit training recipe required')
    # The bound data.collate uses recipe == clean for precise correspondence
    # labels and its clean-only pose auxiliary. Equal target arrays alone are
    # therefore insufficient to declare two training examples equivalent.
    effective_target = digest(dict(materialized_targets=inspected.target_sha256,
                                   precise_recipe=recipe == 'clean'))
    return replace(inspected, target_sha256=effective_target)


def prepare(root, pipeline, verification_path, contract_path, loader=None):
    root = Path(root).resolve(); pipeline = Path(pipeline).resolve(); bindings = Bindings()
    for path in (root / 'pipeline_failure.json', pipeline / 'pipeline_failure.json',
                 pipeline / 'admission_shortfall.json'):
        require(not path.exists(), 'failure/shortfall takes precedence over completion')
    verification = bindings.read(verification_path)
    require(verification['status'] == 'passed' and verification['training_started'] is False
            and verification['gpu_used'] is False and not verification['heldout_family_overlap'],
            'independent final verification required')
    complete = bindings.read(root / 'pipeline_complete.json', verification['pipeline_complete_sha256'])
    require(complete['status'] == 'complete' and complete == bindings.read(root / 'generation_complete.json'),
            'data generation did not complete consistently')
    controller = bindings.read(pipeline / 'pipeline_complete.json')
    require(controller['status'] == 'complete'
            and controller['complete_sha256'] == verification['pipeline_complete_sha256']
            and bindings.read(pipeline / 'full_generation_return.json')['returncode'] == 0,
            'generation controller did not complete successfully')
    require(complete['source_binding_sha256'] == verification['source_binding_sha256'],
            'final verification source binding differs')
    basic, basic_checks, excluded_bases, original_count, excluded_rows = basic_rows(
        root, complete, contract_path, bindings)
    hard = []; hard_keys = set(); metadata = {}; versions = {}
    for stage, count in HARD_COUNTS.items():
        result = verification['versions'][stage]
        manifest = bindings.read(root / stage / 'manifest.json', result['manifest_sha256'])
        audit = bindings.read(root / stage / 'pixel_audit.json', result['audit_sha256'])
        rows = manifest['entries']; checks = indexed(audit['receipts'], 'id', 'hard audit ID')
        require(manifest['split'] == 'train' and len(rows) == count == result['pairs'] == audit['pairs'],
                'full hard release size differs')
        require(audit['status'] == 'passed' and audit['all_rows_rechecked_after_generation'] is True,
                'hard release independent final pixel audit missing')
        require(set(checks) == set(indexed(rows, 'pair_id', 'hard Pair ID'))
                and Counter(r['label'] for r in rows) == {True: count // 2, False: count // 2},
                'hard release membership or labels differ')
        for row in rows:
            require(row['version'] == stage, 'wrong hard version')
            item = inspect_training_row(stage, row, checks[row['pair_id']], loader)
            require(item.ref.source_base_key not in hard_keys, 'repeated hard original pair')
            hard_keys.add(item.ref.source_base_key); hard.append(item)
            metadata[(stage, row['pair_id'])] = row
        versions[stage] = dict(pairs=count, positives=count // 2, negatives=count // 2)
    require(hard_keys == excluded_bases, 'base exclusion does not match actual hard population')
    inspected_basic = []
    for row in basic:
        item = inspect_training_row('v17_filtered', row, basic_checks[row['pair_id']], loader)
        inspected_basic.append(item); metadata[('v17_filtered', row['pair_id'])] = row
    catalog, dedup = deduplicate_basic(inspected_basic, hard)
    inspected_by_id = {(x.ref.stage, x.ref.pair_id): x for x in inspected_basic + hard}
    manifests = {}
    for stage in STAGES:
        entries = []
        for ref in catalog:
            if ref.stage != stage:
                continue
            row = metadata[(stage, ref.pair_id)]; item = inspected_by_id[(stage, ref.pair_id)]
            entries.append(dict(pair_id=ref.pair_id, label=ref.label, recipe=row['recipe'],
                sample_path=ref.sample_path, sample_sha256=ref.sample_sha256,
                source_base_key=ref.source_base_key, source_entry_sha256=digest(row),
                actual_matcher_input_sha256=ref.model_input_sha256,
                effective_training_target_sha256=item.target_sha256,
                legacy_pixel_audit_numerical_sha256=item.legacy_audit_sha256))
        manifests[stage] = dict(schema='curriculum-admitted-train/1', stage=stage,
            split='train', entries=entries, online_augmentation=False)
        versions[stage] = dict(pairs=len(entries), positives=sum(e['label'] for e in entries),
                               negatives=sum(not e['label'] for e in entries))
    bindings.verify()
    return dict(schema='curriculum-data-admission/1', status='passed', versions=versions,
        original_basic_rows=original_count, basic_rows_excluded_by_original_identity=excluded_rows,
        exact_input_dedup=dedup, catalog=[asdict(r) for r in catalog],
        catalog_sha256=digest([asdict(r) for r in catalog]),
        manifests=manifests, bound_inputs=bindings.files,
        actual_sample_files_hashed=len(basic) + len(hard),
        geometry_reaudit_performed=False, final_pixel_verification_reused=True,
        original_archives_modified=False, class_balance_is_exposure_not_row_deletion=True,
        gpu_used=False, training_started=False, budget_locked=False)


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'pipeline', 'verification', 'basic-contract', 'baseline-source',
                 'baseline-preparation', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'), 'hide GPUs before CPU admission')
    require(not args.out.exists(), 'new output directory required; preserve earlier evidence')
    if hasattr(os, 'getpriority'):
        os.nice(max(0, 15 - os.getpriority(os.PRIO_PROCESS, 0)))
    # Mount the explicitly bound immutable training loader, not a similarly
    # named historical package in the current workspace.
    prior = json.loads(args.baseline_preparation.read_text()); baseline = args.baseline_source.resolve()
    actual = {str(p.relative_to(baseline)): file_sha(p) for p in baseline.rglob('*.py')}
    require(prior['status'] == 'passed' and actual == prior['baseline_python_sha256'],
            'actual training loader/source differs from tested binding')
    from .verify_validation_preparation import bind_baseline
    bind_baseline(baseline)
    result = prepare(args.root, args.pipeline, args.verification, args.basic_contract)
    require(actual == {str(p.relative_to(baseline)): file_sha(p) for p in baseline.rglob('*.py')},
            'baseline source changed during admission')
    result.update(baseline_source=str(baseline), baseline_python_sha256=actual,
                  baseline_preparation_sha256=file_sha(args.baseline_preparation))
    args.out.mkdir(parents=True)
    output_hashes = {}
    for stage, manifest in result.pop('manifests').items():
        path = args.out / (stage + '.json')
        path.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + '\n')
        output_hashes[stage] = dict(path=str(path.resolve()), sha256=file_sha(path))
    result['training_manifests'] = output_hashes
    path = args.out / 'admission_complete.json'
    path.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'versions', 'budget_locked', 'training_started')}))


if __name__ == '__main__':
    main()

"""Read-only requalification of an archived review under the new contract."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .seam_contract import CONTRACT, evaluate_masks


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_record(task):
    root, row, old = task
    root = Path(root) / row['version']
    proof = root / 'proof' / Path(row['proof_path']).name
    sample = root / 'samples' / Path(row['sample_path']).name
    if sha(proof) != row['proof_sha256'] or sha(sample) != row['sample_sha256']:
        raise ValueError('actual archived sample/proof SHA changed')
    if old['status'] != 'passed' or old['sample_sha256'] != row['sample_sha256'] or old['proof_sha256'] != row['proof_sha256']:
        raise ValueError('old independent pixel audit binding missing')
    with np.load(proof, allow_pickle=False) as z:
        stages = {stage: {s: np.unpackbits(z['packed_'+stage+'_'+s], axis=1).astype(bool)
                         for s in 'ab'} for stage in ('fragment', 'primary', 'final')}
        ref = {k.removeprefix('pristine_'): z[k] for k in z.files if k.startswith('pristine_')}
    with np.load(sample, allow_pickle=False) as z:
        if not bool(z['label']):
            raise ValueError('only positive geometry may be measured')
        for side in 'ab':
            actual = np.unpackbits(z['mask_'+side+'_packed']).reshape(tuple(z['mask_'+side+'_shape'])).astype(bool)
            if not np.array_equal(actual.squeeze(), stages['final'][side]):
                raise ValueError('final archive differs from pixel proof')
        translation = np.asarray(z['translation_a_to_b_rc'])
    summary, _ = evaluate_masks(stages['fragment'], stages['primary'], stages['final'], ref, translation)
    old_before = row['detail']['crop_only_seam_floor']['before']
    if abs(summary['original_over_smaller_perimeter'] - old_before['common_over_smaller_perimeter']) > 1e-9:
        raise ValueError('independent original20 recomputation differs')
    return dict(id=row['id'], version=row['version'], recipe=row['recipe'], groups=row['groups'],
                baseline_slot=row['baseline_slot'], sample_sha256=row['sample_sha256'],
                proof_sha256=row['proof_sha256'], actual_pixels_checked=True, **summary)


def quantiles(values):
    return np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--workers', type=int, default=2)
    args = p.parse_args(); start = time.monotonic()
    if args.out.exists():
        raise ValueError('new audit output only; do not overwrite historical records')
    if not 1 <= args.workers <= 4:
        raise ValueError('bounded CPU audit:1–4 single-thread workers')
    if (args.root/'pipeline_failure.json').exists() or read(args.root/'pipeline_complete.json')['status'] != 'complete':
        raise ValueError('only a completed, nonfailed review can be requalified')
    inputs = {}; all_entries = {}; tasks = []
    for version in ('v17.5', 'v18'):
        manifest = args.root/version/'manifest.json'; audit = args.root/version/'pixel_audit.json'
        entries = read(manifest)['entries']; previous = read(audit)
        if previous['status'] != 'passed':
            raise ValueError('old pixel audit not passed')
        by_id = {r['id']: r for r in previous['receipts']}
        if len(by_id) != len(entries) or len({r['id'] for r in entries}) != len(entries):
            raise ValueError('row identity or old audit completeness mismatch')
        all_entries[version] = entries
        inputs[version] = dict(manifest_sha256=sha(manifest), pixel_audit_sha256=sha(audit))
        tasks.extend((str(args.root), r, by_id[r['id']]) for r in entries if r['label'])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(check_record, tasks, chunksize=2))
    versions = {}; retained = []
    for version, entries in all_entries.items():
        rr = [r for r in rows if r['version'] == version]
        good = [r for r in rr if r['eligible']]; slots = {r['baseline_slot'] for r in good}
        by_recipe = {}
        for recipe in sorted({r['recipe'] for r in rr}):
            part = [r for r in rr if r['recipe'] == recipe]
            by_recipe[recipe] = dict(positive=len(part), eligible=sum(r['eligible'] for r in part),
                original20_excluded=sum(not r['original20_pass'] for r in part),
                extra_final30_excluded=sum(r['original20_pass'] and not r['final30_pass'] for r in part))
        selected = [r for r in entries if r['baseline_slot'] in slots]
        counts=Counter(r['label'] for r in selected)
        if counts[True] != len(good) or counts[False] != len(good):
            raise ValueError('paired negatives missing from review-only qualified view')
        retained.extend(selected)
        versions[version] = dict(positive=len(rr), original20_excluded=sum(not r['original20_pass'] for r in rr),
            original20_pass=sum(r['original20_pass'] for r in rr),
            extra_final30_excluded=sum(r['original20_pass'] and not r['final30_pass'] for r in rr),
            eligible=len(good), eligible_review_pairs=len(selected), recipes=by_recipe,
            original_ratio_quantiles=quantiles([r['original_over_smaller_perimeter'] for r in rr]),
            final_ratio_quantiles=quantiles([r['final_over_original'] for r in rr]))
    for version, binding in inputs.items():
        if sha(args.root/version/'manifest.json') != binding['manifest_sha256'] or sha(args.root/version/'pixel_audit.json') != binding['pixel_audit_sha256']:
            raise ValueError('input changed during read-only audit')
    args.out.mkdir(parents=True)
    def save(name, value):
        (args.out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    save('rows.json', rows)
    save('qualified_review_manifest.json', dict(contract=CONTRACT, review_only=True,
        training_dataset=False, negative_geometry_not_evaluated=True, entries=retained))
    save('summary.json', dict(status='complete', contract=CONTRACT, versions=versions, inputs=inputs,
        rows_sha256=sha(args.out/'rows.json'), qualified_review_manifest_sha256=sha(args.out/'qualified_review_manifest.json'),
        source_sha256={x.name:sha(x) for x in [Path(__file__),Path(__file__).with_name('seam_contract.py')]},
        actual_positive_pixels_audited=len(rows), negative_geometry_not_evaluated=True,
        old_data_unchanged=True, gpu_used=False, new_samples_generated=0,
        seconds=time.monotonic()-start))
    print(json.dumps(read(args.out/'summary.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()

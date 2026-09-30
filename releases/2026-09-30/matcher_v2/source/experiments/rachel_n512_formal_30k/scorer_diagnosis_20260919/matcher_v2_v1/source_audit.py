"""Read-only admission audit for straight-seam source pools.

This checks the ORIGINAL masks referenced by each v14 split, not augmented
training inputs or ground-truth labels. A source pool is never repaired by
silently dropping a cross-split collision. No generation or model inference is
performed here. TEST images are used only for identity/format auditing.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = ('train', 'select', 'test')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def mask_identity(path):
    """Hash both bytes and decoded binary pixels; crop hash catches relocation."""
    path = Path(path)
    data = path.read_bytes()
    with Image.open(path) as image:
        if image.mode not in ('1', 'L', 'P'):
            raise ValueError(f'non-mask image mode: {image.mode}')
        mask = np.asarray(image.convert('L')) > 0
    if mask.shape != (800, 800) or not mask.any() or mask.all():
        raise ValueError(f'invalid original mask: shape={mask.shape}, area={int(mask.sum())}')
    r, c = np.nonzero(mask)
    crop = mask[r.min():r.max() + 1, c.min():c.max() + 1]
    def pixel_hash(array):
        return sha256(np.asarray(array.shape, dtype='<u4').tobytes()
                      + np.packbits(array, bitorder='little').tobytes())
    return dict(file_sha256=sha256(data), pixel_sha256=pixel_hash(mask),
                crop_pixel_sha256=pixel_hash(crop), foreground_area=int(mask.sum()))


def _collisions(inventory, field):
    owners = defaultdict(lambda: defaultdict(list))
    for split, rows in inventory.items():
        for row in rows:
            if field in row:
                owners[row[field]][split].append(row['path'])
    return [dict(identity=identity, owners=dict(sorted(splits.items())))
            for identity, splits in sorted(owners.items()) if len(splits) > 1]


def audit_sources(manifests, source_root, *, workers=2, hash_pixels=True):
    if set(manifests) != set(SPLITS):
        raise ValueError('all and only train/select/test manifests are required')
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('source audit permits 1..4 CPU readers')
    source_root = Path(source_root).resolve(strict=True)
    inventory, counts, errors, manifest_hashes, declared_families = {}, {}, [], {}, {}
    for split in SPLITS:
        manifest = Path(manifests[split])
        data = manifest.read_bytes()
        entries = json.loads(data)['entries']
        manifest_hashes[split] = dict(path=str(manifest.resolve()), sha256=sha256(data))
        found, excluded, families_by_path = {}, Counter(), defaultdict(set)
        original_references = 0
        all_families = {}
        for entry in entries:
            source = entry.get('source_row')
            if not isinstance(source, dict):
                errors.append(dict(split=split, pair_id=entry.get('pair_id'), reason='missing_source_row'))
                continue
            for side in ('fragment_a', 'fragment_b'):
                fragment = source.get(side, {})
                token = fragment.get('fragment_token', '')
                family = fragment.get('split_unit_id')
                if not isinstance(family, str) or not family:
                    errors.append(dict(split=split, token=token, reason='missing_source_family'))
                    continue
                all_families[(token, family)] = dict(path=token, source_family=family)
                if token.startswith(('rachel-union/', 'rachel-union-')):
                    excluded['derived_union_not_original_mask'] += 1
                    continue
                if token.startswith('gen5-group-') and 'model_mask_path' not in fragment:
                    excluded['derived_gen5_group_no_original_mask'] += 1
                    continue
                if not token.startswith('rachel/'):
                    errors.append(dict(split=split, pair_id=entry.get('pair_id'), side=side,
                                       reason='not_identified_original_rachel', token=token))
                    continue
                family, relative = fragment.get('split_unit_id'), fragment.get('model_mask_path')
                if not isinstance(family, str) or not family or not isinstance(relative, str):
                    errors.append(dict(split=split, token=token, reason='missing_family_or_path'))
                    continue
                path = (source_root / relative).resolve()
                if not path.is_relative_to(source_root) or not path.is_file():
                    errors.append(dict(split=split, token=token, path=str(path), reason='missing_or_outside_source_root'))
                    continue
                families_by_path[str(path)].add(family)
                original_references += 1
                found[str(path)] = dict(path=str(path), source_family=family, fragment_token=token)
        for path, families in families_by_path.items():
            if len(families) != 1:
                errors.append(dict(split=split, path=path, reason='conflicting_family_for_same_mask',
                                   families=sorted(families)))
        inventory[split] = [found[path] for path in sorted(found)]
        declared_families[split] = list(all_families.values())
        counts[split] = dict(manifest_entries=len(entries), original_mask_references=original_references,
                            unique_original_masks=len(found), source_families=len({r['source_family'] for r in found.values()}),
                            all_v14_declared_source_families=len({r['source_family'] for r in all_families.values()}),
                            excluded_reference_counts=dict(excluded))
        if not found:
            errors.append(dict(split=split, reason='empty_source_pool'))
    if hash_pixels:
        paths = sorted({row['path'] for rows in inventory.values() for row in rows})
        def inspect(path):
            try:
                return path, mask_identity(path), None
            except Exception as exc:
                return path, None, f'{type(exc).__name__}: {exc}'
        identities = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for path, identity, error in pool.map(inspect, paths):
                if error:
                    errors.append(dict(path=path, reason='mask_decode_or_pixel_contract_failed', detail=error))
                else:
                    identities[path] = identity
        for rows in inventory.values():
            for row in rows:
                row.update(identities.get(row['path'], {}))
    fields = ['path', 'source_family'] + (['pixel_sha256', 'crop_pixel_sha256'] if hash_pixels else [])
    collisions = {field: _collisions(inventory, field) for field in fields}
    collisions['all_v14_declared_source_families'] = _collisions(declared_families, 'source_family')
    passed = hash_pixels and not errors and not any(collisions.values())
    result = dict(schema='straight-seam-source-admission/1', status='passed' if passed else 'not_admitted',
                  read_only=True, model_inference=False, test_used_for='identity and format only; never selection',
                  source_root=str(source_root), manifests=manifest_hashes, counts=counts,
                  pixel_hashes_checked=hash_pixels, collision_counts={k: len(v) for k, v in collisions.items()},
                  collisions=collisions, errors=errors, inventory=inventory,
                  source_family_definition='original split_unit_id; not a claim of inferred manuscript identity')
    result['inventory_sha256'] = sha256(canonical_json(inventory).encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in SPLITS:
        parser.add_argument('--' + split, required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output-new', type=Path, help='Save generated audit JSON; refuse to overwrite any file')
    args = parser.parse_args()
    result = audit_sources({split: getattr(args, split) for split in SPLITS}, args.source_root, workers=args.workers)
    encoded = canonical_json(result)
    if args.output_new:
        args.output_new.parent.mkdir(parents=True, exist_ok=True)
        with args.output_new.open('x') as stream:
            stream.write(encoded + '\n')
        summary = {k: v for k, v in result.items() if k not in ('inventory', 'collisions', 'errors')}
        summary.update(output=str(args.output_new), report_sha256=sha256((encoded + '\n').encode()),
                       error_count=len(result['errors']), first_errors=result['errors'][:12])
        print(canonical_json(summary))
    else:
        print(encoded)
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())

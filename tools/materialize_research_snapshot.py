"""Materialize a hash-verified code snapshot in a NEW output directory.

This command never starts training and does not contain data or checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def checked_relative(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('unsafe manifest path')
    return p


def materialize(release, variant, destination):
    release, destination = Path(release).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing directory')
    manifest = json.loads((release/'manifests'/f'{variant}.json').read_text())
    if manifest.get('schema') != 'research-source-capsule/1' or manifest.get('variant') != variant:
        raise ValueError('invalid source manifest')
    overlay = release/checked_relative(manifest['source_subdirectory'])
    sources = []
    for name, expected in manifest['source_files'].items():
        rel = checked_relative(name)
        src = overlay/rel
        if not src.exists():
            src = release/'base'/rel
        if hashlib.sha256(src.read_bytes()).hexdigest() != expected:
            raise ValueError('source SHA256 mismatch: '+name)
        sources.append((src, rel))
    destination.mkdir(parents=True, exist_ok=False)
    for src, rel in sources:
        target = destination/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    (destination/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return dict(status='verified_code_only', variant=variant, files=len(sources),
                training_started=False, output=str(destination))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--release', default=str(Path(__file__).resolve().parents[1]/'releases/2026-09-29'))
    p.add_argument('--variant', required=True, choices=['threshold_complex','mergefix_complex',
        'simple_complex','binary_e32_micro32','v17_matcher_binary','pooling_controls'])
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(materialize(args.release, args.variant, args.output), indent=2))

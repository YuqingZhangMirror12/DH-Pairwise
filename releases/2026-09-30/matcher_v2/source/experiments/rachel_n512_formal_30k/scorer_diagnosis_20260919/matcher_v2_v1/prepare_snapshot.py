"""Copy a verified snapshot and refresh only explicit packages into a NEW root."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-snapshot', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--refresh-package', action='append', required=True)
    parser.add_argument('--copy-asset', action='append', default=[])
    parser.add_argument('--baseline-execution', type=Path)
    parser.add_argument('--baseline-local-root', type=Path)
    parser.add_argument('--output-new', type=Path, required=True)
    args = parser.parse_args()
    base, repo, out = args.base_snapshot.resolve(), args.repo_root.resolve(), args.output_new.resolve()
    binding = json.loads((base / 'source_binding.json').read_text())
    sources = {}
    for name, expected in binding.items():
        path = (base / name).resolve()
        if not path.is_relative_to(base) or sha(path) != expected:
            raise ValueError('base snapshot changed: ' + name)
        sources[name] = path
    baseline_receipt = None
    if bool(args.baseline_execution) != bool(args.baseline_local_root):
        raise ValueError('baseline execution and local source must be specified together')
    if args.baseline_execution:
        execution = json.loads(args.baseline_execution.read_text())
        source = args.baseline_local_root.resolve()
        baseline = execution['baseline']
        for name, expected in baseline['python_sha256'].items():
            path = (source / name).resolve(strict=True)
            if not path.is_relative_to(source) or sha(path) != expected:
                raise ValueError('original production baseline changed: ' + name)
            sources[name] = path
        baseline_receipt = dict(execution_sha256=sha(args.baseline_execution),
            original_root=baseline['root'], local_root=str(source),
            python_sha256=baseline['python_sha256'])
    for package in args.refresh_package:
        folder = (repo / package).resolve()
        if not folder.is_relative_to(repo) or not folder.is_dir():
            raise ValueError('explicit repository package required')
        for path in folder.rglob('*.py'):
            if '__pycache__' not in path.parts:
                if baseline_receipt and str(path.relative_to(repo)) in baseline_receipt['python_sha256']:
                    raise ValueError('refresh must not overwrite the preserved original baseline')
                sources[str(path.relative_to(repo))] = path
    out.mkdir(parents=True, exist_ok=False)
    result = {}
    for name, source in sorted(sources.items()):
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        result[name] = sha(target)
    with (out / 'source_binding.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    if baseline_receipt:
        with (out / 'baseline_composition.json').open('x') as stream:
            json.dump(baseline_receipt, stream, indent=2, sort_keys=True)
            stream.write('\n')
    assets = {}
    for name in args.copy_asset:
        source = (repo/name).resolve(strict=True)
        if not source.is_relative_to(repo) or not source.is_file() or source.suffix == '.py':
            raise ValueError('explicit non-Python repository asset required')
        target = out/source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(source, target); assets[str(source.relative_to(repo))] = sha(target)
    with (out/'asset_binding.json').open('x') as stream:
        json.dump(assets, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps(dict(source_root=str(out), files=len(result), binding_sha256=sha(out / 'source_binding.json'),
                          refreshed_packages=args.refresh_package)))


if __name__ == '__main__':
    main()

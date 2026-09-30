"""Preserve existing remote code, selected weights and metric receipts once."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=False)
    base = Path('/root/autodl-tmp')
    code_names = ['rachel_ablation_v3_source_20260907_008',
                  'rachel_ablation_v3_source_20260907_009',
                  'rachel_pairwise_benchmark_source_direct_20260904_001',
                  'rachel_pairwise_eval_source_exact6_20260906_003']
    result_names = ['rachel_ablation_v3_20260907_001',
                    'rachel_same_data_benchmark_direct_20260904_001',
                    'rachel_same_data_final_eval_exact6_20260906_004',
                    'rachel_n512_convergence_20260901_001']
    selected = set()
    for name in code_names:
        root = base / name
        if not root.is_dir():
            raise FileNotFoundError(root)
        selected.update(x for x in root.rglob('*') if x.is_file()
                        and not any(v in x.parts for v in ('__pycache__', '.pytest_cache', '.git'))
                        and x.suffix in ('.py', '.json', '.md', '.yaml', '.yml', '.toml', '.sh', '.txt'))
    for name in result_names:
        root = base / name
        if not root.is_dir():
            raise FileNotFoundError(root)
        for x in root.rglob('*'):
            if not x.is_file():
                continue
            if x.name == 'winner.pt' or x.name == 'epoch-047.pt':
                selected.add(x)
            elif x.suffix in ('.json', '.md', '.yaml', '.yml', '.toml') and x.stat().st_size < 10_000_000:
                selected.add(x)
    data = base / 'dataset_rachel_pairwise_n512_v1'
    for name in ('preprocess_receipt.json', 'run_config.json', 'pairs/summary.json',
                 'pairs/train.jsonl', 'pairs/val.jsonl', 'pairs/test.jsonl', 'qa/preprocess_summary.json'):
        x = data / name
        if x.is_file():
            selected.add(x)
    rows = [{'original': str(x), 'archive_path': str(x.relative_to(base)), 'bytes': x.stat().st_size}
            for x in sorted(selected)]
    manifest = dict(schema_version='rachel-preserved-baseline/v1', created_unix=time.time(),
                    original_files_preserved=True, entries=rows,
                    training_dataset=str(data), raw_rachel=str(base / 'dataset_rachel/datasets'),
                    data_policy='Data files stay at original absolute paths; exact split manifests are included.',
                    active_matcher=str(base / 'rachel_ablation_v3_20260907_001/confirmation_candidate_20260908_001/seed260909/training/multiscale7_16_32_64/winner.pt'))
    mp = dest / 'manifest.json'
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    archive = dest / 'baseline_code_weights_receipts.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
        tar.add(mp, arcname='manifest.json')
        for row in rows:
            tar.add(row['original'], arcname=row['archive_path'], recursive=False)
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    with tarfile.open(archive, 'r:gz') as tar:
        count = len(tar.getmembers())
    assert count == len(rows) + 1
    receipt = dict(status='complete', archive=str(archive), sha256=h.hexdigest(),
                   archived_files=count, bytes=archive.stat().st_size,
                   source_bytes=sum(r['bytes'] for r in rows))
    (dest / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

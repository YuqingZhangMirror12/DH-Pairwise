"""Record CPU preparation tests, optionally inspect existing probe inputs only."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import unittest

from .catalog import inspect_sample


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def inspect_probe(root):
    complete = root / 'probe_complete.json'
    if not complete.is_file() or (root / 'pipeline_failure.json').exists():
        raise ValueError('existing completed probe required; do not generate or restart')
    result = []
    for stage in ('v17.5', 'v18'):
        manifest = root / stage / 'manifest.json'
        audit_path = root / stage / 'pixel_audit.json'
        rows = read(manifest)['entries']; audited = read(audit_path)
        if audited['status'] != 'passed' or audited['pairs'] != len(rows):
            raise ValueError('probe audit incomplete')
        checks = {r['id']: r for r in audited['receipts']}
        if len(checks) != len(rows) or set(checks) != {r['pair_id'] for r in rows}:
            raise ValueError('probe row membership differs')
        for row in rows:
            sample = inspect_sample(stage, row, checks[row['pair_id']])
            result.append(dict(stage=stage, pair_id=sample.ref.pair_id,
                source_base_key=sample.ref.source_base_key,
                sample_sha256=sample.ref.sample_sha256,
                matcher_input_sha256=sample.ref.model_input_sha256,
                target_sha256=sample.target_sha256,
                legacy_audit_sha256=sample.legacy_audit_sha256))
    return dict(status='passed', root=str(root), complete_sha256=sha(complete),
        actual_existing_probe_rows=len(result), rows=result,
        geometry_regenerated=False, neural_inference=False,
        actual_loader_and_six_input_fingerprints_checked=True,
        full_data_admission=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--existing-probe')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('explicitly hide all GPUs for this CPU preparation check')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    before = {p.name: sha(p) for p in source.glob('*.py')}
    begin = time.time(); stream = io.StringIO()
    names = [__package__ + '.' + x for x in ('test_exposure', 'test_catalog', 'test_optimizer_cursor',
                                            'test_training_core', 'test_model_integration')]
    tests = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    (out / 'tests.log').write_text(stream.getvalue())
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed',
        scope='CPU sampler, catalog, update loop, synthetic optimizer/RNG continuation and bound96 Matcher/light-head fixtures; not production training',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), source_sha256=before, test_log_sha256=sha(out / 'tests.log'),
        gpu_used=False, formal_training_started=False, training_launcher_ready=False,
        protocol_budget_locked=False, elapsed_seconds=time.time() - begin)
    if result.wasSuccessful() and args.existing_probe:
        receipt['existing_probe_inputs'] = inspect_probe(Path(args.existing_probe))
    if before != {p.name: sha(p) for p in source.glob('*.py')}:
        raise ValueError('preparation source changed while verifying')
    (out / 'preparation.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'tests', 'failures', 'errors', 'gpu_used', 'formal_training_started')},
                     ensure_ascii=False))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Test the independent exporter against the prepared production source."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob('*.py'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('explicit CPU-only tests required')
    here = Path(__file__).resolve().parent
    base = here.parent
    args.out.mkdir(parents=True, exist_ok=False)
    roots = dict(exporter=here, training=args.source.resolve(), evaluator=base / 'binary_eval_v1')
    before = {name: inventory(root) for name, root in roots.items()}
    # Import fixture/model code from the deployed preparation, not the older
    # generic s7_consensus package beside this new display-only code.
    spec = importlib.util.spec_from_file_location('binary_report_bootstrap', base / 'binary_eval_v1/entry.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.bootstrap(args.source, base / 's7_consensus_eval_v14')
    namespace = types.ModuleType('binary_report_checks')
    namespace.__path__ = [str(here)]
    sys.modules['binary_report_checks'] = namespace
    start = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromName('binary_report_checks.test_export')
    with (args.out / 'tests.log').open('x') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    unchanged = before == {name: inventory(root) for name, root in roots.items()}
    passed = result.wasSuccessful() and unchanged
    record = dict(schema='binary-report-exporter-preparation/1', status='passed' if passed else 'failed',
        tests=result.testsRun, errors=len(result.errors), failures=len(result.failures), skipped=len(result.skipped),
        synthetic_cases_only=True, trained_checkpoint_opened=False, real_data_opened=False,
        gpu_used=False, inference_of_real_samples_performed=False, source_files_unchanged=unchanged,
        source_python_sha256=before, elapsed_seconds=time.time() - start)
    (args.out / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('status', 'tests', 'errors', 'failures', 'skipped', 'source_files_unchanged')}))
    if not passed:
        print((args.out / 'tests.log').read_text())
        raise SystemExit(1)


if __name__ == '__main__':
    main()

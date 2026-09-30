"""Bounded local checks for new result assembly; no server or real inference."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path)
    parser.add_argument('--common', type=Path)
    parser.add_argument('--execution-receipt', type=Path, action='append', default=[])
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only check required')
    here = Path(__file__).resolve().parent
    evaluator = (args.evaluator or here.parent / 'binary_eval_v1').resolve()
    common = (args.common or here.parent / 's7_consensus_eval_v14').resolve()
    spec = importlib.util.spec_from_file_location('assembly_test_bootstrap', evaluator / 'entry.py')
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    training, _ = entry.bootstrap(args.source, common)
    namespace = types.ModuleType('binary_report_checks'); namespace.__path__ = [str(here)]
    sys.modules['binary_report_checks'] = namespace
    from binary_report_checks.verify import inventory
    from binary_report_checks import assemble
    from binary_report_checks.test_assemble import selected
    roots = dict(report_source=here, prepared_training=args.source.resolve(),
                 evaluator=evaluator, common=common)
    before = {name: inventory(root) for name, root in roots.items()}
    receipts = []
    for path in args.execution_receipt:
        value = assemble.read(path)
        if (value.get('schema') != 'binary-microbatch-continuation/1'
                or value.get('status') != 'imported' or value.get('variant') not in ('patch', 'stats')
                or value.get('binding', {}).get('config') != training.canonical_record(
                    training.TrainingConfig(scorer_variant=value['variant']).record())):
            raise ValueError('current execution receipt/config mismatch')
        execution = {key: value[key] for key in ('schema', 'updates', 'exposures',
                     'original_checkpoint_sha256', 'resume_origin_sha256')}
        models = selected('binary_' + value['variant'])
        for model in models.values(): model['execution_continuation'] = execution
        assemble.validate_selection(models, 'binary_' + value['variant'])
        receipts.append(dict(path=str(path.resolve()), sha256=assemble.sha(path),
                             metadata_schema_compatible=True, variant=value['variant']))
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in (
        'binary_report_checks.test_assemble', 'binary_report_checks.test_assemble_integration',
        'binary_report_checks.test_export', 'binary_report_checks.test_execution_provenance'))
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    with (args.out / 'tests.log').open('x') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    unchanged = (before == {name: inventory(root) for name, root in roots.items()}
                 and all(assemble.sha(r['path']) == r['sha256'] for r in receipts))
    passed = result.wasSuccessful() and unchanged
    record = dict(schema='binary-result-assembly-preparation/1', status='passed' if passed else 'failed',
        tests=result.testsRun, errors=len(result.errors), failures=len(result.failures), skipped=len(result.skipped),
        source_files_unchanged=unchanged, source_python_sha256=before,
        execution_receipt_metadata_checks=receipts,
        synthetic_terminal_fixtures_only=True, real_results=False, remote_server_queried=False,
        trained_checkpoint_opened=False, gpu_used=False, real_data_opened=False,
        six_job_integration_fixtures=5, six_job_fixture_snapshot_records=110,
        six_job_fixture_unique_untrained_forward_traces=5,
        independent_experiments_measured=0,
        scope='assembly, exporter and execution-provenance tests; five synthetic six-job integrations',
        elapsed_seconds=time.time() - start)
    (args.out / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('status', 'tests', 'errors', 'failures', 'skipped', 'source_files_unchanged')}))
    if not passed:
        print((args.out / 'tests.log').read_text())
        raise SystemExit(1)


if __name__ == '__main__':
    main()

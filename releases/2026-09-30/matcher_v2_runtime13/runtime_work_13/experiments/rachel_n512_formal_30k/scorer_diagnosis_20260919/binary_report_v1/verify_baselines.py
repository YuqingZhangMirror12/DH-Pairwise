"""Local reporting tests and an already-generated real-data import receipt."""
import argparse
import io
import json
from pathlib import Path
import unittest

from . import baselines, compare, test_baselines
from .export import read, require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual-baselines', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'preserve previous verification')
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_baselines))
    actual = read(args.actual_baselines)
    require(result.wasSuccessful(), log.getvalue())
    models = set(actual['selected'])
    require(actual['schema'] == 'binary-report-complex-baselines/1'
            and {'mergefix_m12', 'mergefix_scratch', 'threshold_m12'} <= models
            and models <= set(baselines.BASELINES) and len(actual['rows']) == 20 * len(models),
            'actual completed-baseline import required')
    require(all(sha(path) == value for path, value in actual['input_sha256'].items()),
            'actual consumed files changed')
    main_real = [r for r in actual['rows'] if r['is_main'] and r['population'] == 'real_test']
    require(len(main_real) == 2 * len(models), 'models by two real TEST domains required')
    for row in main_real:
        counts = ((161, 59, 102) if row['split'] == 'dunhuang_cv' else (122, 61, 61))
        require(tuple(row['metrics'][k] for k in ('pairs', 'positives', 'negatives')) == counts,
                'actual main REAL-TEST count differs')
    files = [Path(module.__file__).resolve() for module in (baselines, compare, test_baselines)] + [Path(__file__).resolve()]
    receipt = dict(status='passed', tests_run=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), skipped=len(result.skipped),
        actual_baselines_sha256=sha(args.actual_baselines), actual_models=list(actual['selected']),
        actual_metric_rows=len(actual['rows']), actual_completed_jobs_consumed=3 * len(models),
        actual_main_real_rows=len(main_real),
        input_files_unchanged=True, new_model_forward_passes=0, thresholds_fitted=0,
        source_sha256={p.name: sha(p) for p in files})
    args.out.mkdir(parents=True)
    (args.out / 'tests.log').write_text(log.getvalue())
    (args.out / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

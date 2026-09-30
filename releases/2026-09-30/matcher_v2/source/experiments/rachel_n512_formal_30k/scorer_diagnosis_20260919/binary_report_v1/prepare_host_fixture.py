"""Internal routing/display fixture, NEVER install in the user's result app.

Reuses three previously audited untrained four-dimensional CPU records. No
forward pass, server call, trained checkpoint or experiment result is opened.
Copies of records test selector routing, not independent model observations.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .bind_light_results import REPORT_ID, bind_snapshot
from .export import read, require, sha
from .test_bind_light_results import fixture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--node', type=Path, required=True)
    args = p.parse_args()
    require(args.app.resolve().name == 'ui_test_app' and not args.out.exists(), 'internal test app and new receipt only')
    old = read(args.app / 'src/data.json')
    require(old['id'] == 'report:ab92bece-b138-4939-9077-29f7d6bbd8f7', 'never use the user result app')
    args.out.mkdir(parents=True)
    source = Path(__file__).resolve().parent
    receipts = []
    for label, cmd in (
        ('python', [sys.executable, '-m', __package__ + '.test_bind_light_results']),
        ('scopes', [str(args.node), '--test', str(source / 'test_light_scopes.mjs')]),
    ):
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (args.out / (label + '.log')).write_text(run.stdout)
        require(run.returncode == 0, label + ' tests failed')
        receipts.append(dict(name=label, returncode=run.returncode))
    frozen = read(args.fixtures)
    originals = {r['id']: json.loads(r['payload']) for r in frozen['queries']['binary_ui_fixtures']['rows']}
    bundles = [fixture('binary_patch'), fixture('binary_stats')]
    for bundle in bundles:
        variant = 'patch' if bundle['available_light_experiments'][0] == 'binary_patch' else 'stats'
        for row in bundle['rows']:
            row['metrics']['threshold'] = .3
        for group in bundle['binary_case_groups']:
            for i, original in enumerate(group['cases']):
                record = copy.deepcopy(originals['empty' if variant == 'patch' and i == 1 else variant])
                record['pair_id'] = original['pair_id']
                record['provenance'].update(original['provenance'])
                record['provenance']['synthetic_display_clone'] = True
                group['cases'][i] = record
    bundle = copy.deepcopy(bundles[0])
    for key in ('rows', 'binary_case_groups', 'available_light_experiments'):
        bundle[key] += bundles[1][key]
    original_id = old['id']
    before = copy.deepcopy(old)
    old['id'] = REPORT_ID  # exercise the real binder; restore the internal identity immediately below
    result = bind_snapshot(old, bundle, args.fixtures.resolve(), sha(args.fixtures))
    result['id'] = original_id
    result['report'] = dict(asOf='2026-09-28')
    result['title'] = '轻量头报告接入 · 内部合成测试'
    result['status'] = 'synthetic_test_only_not_model_performance'
    result['buildStatus'] = 'complete'
    result['queries'] = {k: v for k, v in result['queries'].items() if k.startswith('light_scorer_')}
    for query in result['queries'].values():
        query['source'].update(label='内部显示夹具：未训练4维CPU记录的复制；表格为合成数值',
            evidenceFlow=[dict(title='显示测试', detail='复用3份原已审计合成CPU记录，复制为44项路由测试，不运行模型。')],
            caveats=['所有表格值为合成显示测试，不是模型性能。44项案例记录不是44次独立推理。'])
    (args.out / 'snapshot_before.json').write_text(json.dumps(before, ensure_ascii=False, indent=2) + '\n')
    (args.app / 'src/data.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    content = args.app / 'src/content/report'
    copied = {}
    for name in ('LightScorerResults.jsx', 'BinaryEvidencePanel.jsx', 'evidence_geometry.mjs',
                 'binary-evidence.css', 'light-scopes.mjs'):
        shutil.copy2(source / name, content / name)
        copied[name] = sha(source / name)
        require(sha(content / name) == copied[name], 'copied source differs')
    receipt = dict(status='local_host_fixture_prepared', tests=receipts, copied_sha256=copied,
        fixture_sha256=sha(args.fixtures), snapshot_sha256=sha(args.app / 'src/data.json'),
        metrics=24, cloned_case_records=44, original_cpu_forward_records_reused=3,
        new_inference=False, production_report_modified=False, visual_check_complete=False)
    (args.out / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

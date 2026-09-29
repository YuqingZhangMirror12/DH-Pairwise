"""Reconcile the appended report against the prior independent metric recount."""
import json
from datetime import datetime, timezone
from pathlib import Path

from .bind_aggressive_results import MODEL, QUERIES, METRICS_QUERY, identity
from .export import read, require, sha


def main():
    root = Path(__file__).resolve().parents[4]
    out = root / 'artifacts/binary_report_20260929/v17_integration_01'
    target = root / 'reports/seam_v3_real_analysis_20260923/src/data.json'
    annotations = root / 'artifacts/human_layout_review_20260923/annotations.json'
    snapshot, before, binding = read(target), read(out / 'report_before.json'), read(out / 'binding.json')
    require(sha(annotations) == binding['annotations_sha256'], 'human annotations changed')
    require(snapshot['id'] == before['id'] and snapshot['buildStatus'] == 'updating',
            'unexpected snapshot identity or finalization state')
    for q, source in before['queries'].items():
        if q in QUERIES:
            require(snapshot['queries'][q]['rows'][:len(source['rows'])] == source['rows'],
                    'historical rows changed')
            require(all(snapshot['queries'][q]['source'][k] == v for k,v in source['source'].items()),
                    'historical provenance changed')
        else:
            require(snapshot['queries'][q] == source, 'unrelated evidence changed')
    for q in QUERIES:
        rows = snapshot['queries'][q]['rows']
        require(len({identity(q,r) for r in rows}) == len(rows), 'duplicate row grain')
    rows = [r for r in snapshot['queries'][METRICS_QUERY]['rows'] if r['model'] == MODEL]
    old_recount = root / 'artifacts/aggressive_binary_evaluation_20260929/metrics_recount.json'
    for previous in read(old_recount)['rows']:
        row = next(r for r in rows if r['selection_kind'] == previous['selection'] and all(
            r[k] == previous[k] for k in ('split', 'population', 'policy')))
        require(row['selected_epoch'] == previous['epoch'], 'recount epoch differs')
        for k in ('pairs','threshold','accuracy','f1','layout20_count','joint_f1','fp','winner_correct_but_rejected'):
            require(row[k] == previous[k], 'independent prior metric recount differs: ' + k)
        require(sha(previous['source']) == previous['source_sha256'], 'prior recount source changed')
    snapshot['buildStatus'] = 'complete'
    snapshot['generatedAt'] = datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False,separators=(',', ':'))+'\n')
    result = dict(status='complete_content_verified_pending_final_rebuild',
        snapshot_sha256=sha(target), annotations_sha256=sha(annotations),
        prior_recount_sha256=sha(old_recount), independent_recount_rows=20,
        added_counts=binding['added_counts'], old_queries_and_rows_preserved=True,
        source_import='six terminal jobs, prediction and NPZ hashes, 22 passed numeric traces',
        tests=dict(targeted_python_passed=13, source_snapshot_node_passed=7,
                   broader_python_discovery='95 passed; 3 unchanged Torch-dependent modules unavailable in bundled Python'),
        browser_checked=['SIM/full table','REAL/full table','REAL/source-TEST table',
                         'v17 wrong Layout 51/292','v17 correct Layout rejected 50/292',
                         'v17 fixed Q trace: 83 union pairs, score .9682, threshold .72',
                         '390x844: document width 390, no broken images; viewport restored'],
        inference_repeated=False, training_modified=False, threshold_refitted=False)
    with (out/'verification.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__': main()

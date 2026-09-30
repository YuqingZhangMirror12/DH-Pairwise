"""Independently recount frozen binary predictions; never fit or infer."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .assemble import REAL_PLAN_SHA
from .export import read, require, sha


def recount(rows, threshold):
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    for row in rows:
        require(type(row['label']) is bool, 'boolean label required')
        require(type(row['score']) in (int, float) and math.isfinite(row['score']),
                'finite score required')
        accepted = row['has_candidate'] and row['numeric_valid'] and row['score'] >= threshold
        key = ('tp' if row['label'] else 'fp') if accepted else ('fn' if row['label'] else 'tn')
        counts[key] += 1
    n = len(rows); p = counts['tp'] + counts['fn']; neg = counts['tn'] + counts['fp']
    require(n and p and neg, 'both classes required')
    return dict(counts, pairs=n, positives=p, negatives=neg,
                accuracy=(counts['tp'] + counts['tn'])/n,
                precision=counts['tp']/max(1, counts['tp']+counts['fp']),
                recall=counts['tp']/p,
                f1=2*counts['tp']/max(1, 2*counts['tp']+counts['fp']+counts['fn']))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--real-plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    require(not args.out.exists(), 'preserve prior audit')
    require(sha(args.real_plan) == REAL_PLAN_SHA, 'source role plan changed')
    plan = read(args.real_plan); evidence = read(args.evidence)
    source_labels = {}
    for split, spec in plan['datasets'].items():
        manifest = Path(spec['local_manifest'])
        require(sha(manifest) == spec['manifest_sha256'], 'real source manifest changed')
        source_labels[split] = {r['pair_id']: r['label'] for r in read(manifest)['pairs']}
    results = []; labels_by_split = {}
    for experiment in evidence['experiments']:
        for job in experiment['jobs']:
            root = Path(job['source']['job']); raw_path = root/'pair_predictions.jsonl'
            labeled_path = root/'case_diagnostics.jsonl'
            require(sha(raw_path) == job['source']['predictions_sha256'], 'raw predictions changed')
            raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
            labeled = [json.loads(line) for line in labeled_path.read_text().splitlines()]
            require(len(raw) == len(labeled) and len({r['pair_id'] for r in raw}) == len(raw),
                    'labeled or raw rows missing/duplicated')
            for a, b in zip(raw, labeled):
                require(all(b.get(k) == v for k, v in a.items()), 'target join changed a prediction')
                expected = a['has_candidate'] and a['numeric_valid'] and a['score'] >= job['threshold']
                require(a['accepted'] is expected, 'accepted flag disagrees with frozen threshold')
            split = job['split']; labels = {r['pair_id']: r['label'] for r in labeled}
            require(labels_by_split.get(split, labels) == labels, 'targets differ across checkpoints')
            labels_by_split[split] = labels
            if split.startswith('sim_test_'):
                groups = {'all': labeled}
            else:
                require(labels == source_labels[split], 'real labels differ from bound source')
                spec = plan['datasets'][split]
                groups = {'all_development_context': labeled}
                if split == 'dunhuang_cv':
                    excluded = set(spec['excluded_gt_pair_ids'])
                    groups['gt_corrected_800_development_context'] = [r for r in labeled if r['pair_id'] not in excluded]
                for name, role in spec['roles'].items():
                    ids = set(role['pair_ids'])
                    groups[name] = [r for r in labeled if r['pair_id'] in ids]
                    require(len(groups[name]) == len(ids), 'source role membership missing')
            require(set(groups) == set(job['groups']), 'reported populations differ')
            checked = 0
            for name, rows in groups.items():
                for policy, metrics in job['groups'][name].items():
                    threshold = job['threshold'] if policy == 'primary' else .3
                    value = recount(rows, threshold)
                    require(metrics['threshold'] == threshold, 'threshold changed')
                    require(all(k == 'tn' and k not in metrics or math.isclose(v, metrics[k], rel_tol=1e-12, abs_tol=1e-12)
                                for k, v in value.items()), 'frozen summary differs from exact recount')
                    checked += 1
            results.append(dict(experiment=experiment['experiment'], selection=job['selection_kind'],
                split=split, rows=len(raw), populations_and_policies=checked,
                predictions_sha256=sha(raw_path), labeled_rows_sha256=sha(labeled_path)))
    result = dict(status='passed', jobs=len(results), results=results,
        real_labels_rechecked_against_bound_manifests=True,
        raw_predictions_unchanged_by_target_join=True,
        simulation_labels_rechecked_across_all_four_checkpoints=True,
        new_inference=False, threshold_fitting=False,
        evidence_sha256=sha(args.evidence), real_plan_sha256=REAL_PLAN_SHA)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], jobs=len(results),
                         policies=sum(r['populations_and_policies'] for r in results), sha256=sha(args.out))))


if __name__ == '__main__': main()

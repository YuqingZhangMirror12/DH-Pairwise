"""Raw-row S7 C16 comparison using the independent recount, not endpoint summaries."""
import argparse
import hashlib
import json
from pathlib import Path

from . import recount_matcher_scorers as raw

ARMS = ('all_tokens', 'matched_tokens', 'edge_seed', 'matched_edges', 'edge_multi')


def run(root, output, arms=ARMS):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('new result required; preserve prior comparisons')
    if len(set(arms)) != len(arms) or not arms or any(a not in ARMS for a in arms):
        raise ValueError('unique registered S7 arms required')
    report = dict(schema='independent-s7-c16-recount/1', status='complete',
        arms=list(arms), operating_points=list(raw.OPS), sources={}, metrics={}, transitions={},
        no_inference=True, no_threshold_fit=True, production_layout_unchanged=True,
        implementation_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), Path(raw.__file__).resolve())},
        caveats=['Own frozen clean SIMVAL thresholds, no REAL/OOD fitting.',
            'REAL keep803 is post-review: kept positive295 plus all negative508.',
            'Negative39 originals and negative469 constructed remain separate.',
            'OOD301 is positive-only with no layout GT; binary Accuracy/F1/AUC unavailable.',
            'Original production pose is evaluated; scorer-selected candidate poses are separate diagnostics.',
            'Some arms differ in edge encoding/parameters and execution topology, not a pure input-only contrast.'])
    for split in raw.SPLITS:
        loaded = {a: raw.load(root/a/'evaluation/c16', split) for a in arms}
        reference = loaded[arms[0]][0]
        matcher = loaded[arms[0]][2]['matcher_sha256']
        for arm, (rows, thresholds, source) in loaded.items():
            if rows.keys() != reference.keys() or source['matcher_sha256'] != matcher:
                raise ValueError('exact same pair membership and Matcher required')
            for pair, row in rows.items():
                for field in ('label', 'target_translation_rc', 'review_status', 'strict_member'):
                    if row[field] != reference[pair][field]:
                        raise ValueError('pair identity/label/review changed: '+pair)
                for field in ('predicted_translation_rc', 'layout_valid', 'layout_error'):
                    if row[field] != reference[pair][field]:
                        raise ValueError('production layout differs; do not claim pure scoring change: '+pair)
            if any(t is None or not 0 <= t <= 1 for t in thresholds.values()):
                raise ValueError('missing frozen operating point')
            report['sources'].setdefault(arm, {})[split] = source
        for population, ids in raw.populations(reference, split).items():
            report['metrics'][population] = {arm: {op: raw.metrics([rows[k] for k in ids], ts[op])
                for op in raw.OPS} for arm, (rows, ts, _) in loaded.items()}
            report['transitions'][population] = {}
            for before in arms:
                for after in arms:
                    if before == after or after != 'edge_multi':
                        continue
                    a, at, _ = loaded[before]
                    b, bt, _ = loaded[after]
                    report['transitions'][population][before+'->'+after] = {
                        op: raw.transitions(a, b, ids, at[op], bt[op]) for op in raw.OPS}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=ARMS, action='append')
    args = parser.parse_args()
    result = run(args.root, args.output, args.arm or ARMS)
    print(json.dumps(dict(status=result['status'], arms=result['arms'], output=args.output)))

"""Reuse the historical diagnostic cases with canonical F/I evaluation splits.

This does not select cases from new F/I outcomes or alter the original cohort.
Only the legacy ``dunhuang`` alias is mapped to ``dunhuang_cv`` after checking
membership in the frozen comparison population. User-rejected GT is omitted.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(original, populations, excluded):
    cases, omitted, seen = [], [], set()
    for row in original['cases']:
        split = 'dunhuang_cv' if row['split'] == 'dunhuang' else row['split']
        key = (split, row['pair_id'])
        if key in seen:
            raise ValueError('duplicate case after canonical split mapping')
        seen.add(key)
        if split not in populations or row['pair_id'] not in populations[split]:
            raise ValueError('historical diagnostic case absent from F/I population')
        if row['pair_id'] in excluded:
            omitted.append(dict(row, split=split))
        else:
            cases.append(dict(row, split=split))
    if not cases:
        raise ValueError('empty diagnostic cohort')
    return dict(selection=original['selection'], cases=cases,
                preparation=dict(selected_from_new_model_outcomes=False,
                    historical_case_count=len(original['cases']),
                    alias_mapping={'dunhuang': 'dunhuang_cv'},
                    user_gt_exclusions_applied=sorted(excluded),
                    omitted_invalid_gt=omitted))


def run(args):
    prior, out = Path(args.prior), Path(args.out)
    if out.exists():
        raise ValueError('output must be a new file; preserve original cohorts')
    original = json.loads((prior/'probe_cases.json').read_text())
    populations = {}
    for split in ('dunhuang_cv', 'turufan'):
        rows = [json.loads(line) for line in
                (prior/split/'case_diagnostics.jsonl').read_text().splitlines()]
        ids = [row['pair_id'] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate case in frozen comparison population')
        populations[split] = set(ids)
    exclusions = json.loads(Path(args.exclusions).read_text())['records']
    result = prepare(original, populations, {r['pair_id'] for r in exclusions})
    result['preparation']['source_sha256'] = {
        'probe_cases.json': hashlib.sha256((prior/'probe_cases.json').read_bytes()).hexdigest(),
        'gt_exclusions.json': hashlib.sha256(Path(args.exclusions).read_bytes()).hexdigest(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(cases=len(result['cases']),
                          omitted=len(result['preparation']['omitted_invalid_gt']),
                          output=str(out))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prior', 'exclusions', 'out'):
        parser.add_argument('--'+name, required=True)
    run(parser.parse_args())

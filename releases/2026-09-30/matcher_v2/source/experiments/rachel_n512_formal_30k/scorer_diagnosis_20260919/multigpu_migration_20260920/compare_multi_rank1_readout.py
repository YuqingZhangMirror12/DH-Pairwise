"""Fixed-weight inference counterfactual: rank-1 group only vs saved multi-group max."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sigmoid(z):
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('new result required')
    root = json.loads((source/'results.json').read_text())
    if root['status'] != 'complete':
        raise ValueError('complete candidate-pose record required')
    counters, cases = {}, []
    for line in (source/'cases.jsonl').open():
        row = json.loads(line)
        if row['cohort'] == 'real_excluded_positive213':
            continue
        ranks = [g for g in row['groups'] if g['proposal_rank'] == 1 and g['eligible']]
        if len(ranks) != 1:
            raise ValueError('no unique eligible first proposal; do not silently invent fallback')
        probability = sigmoid(ranks[0]['logit'])
        thresholds = root['endpoints'][row['model']+':'+row['split']]['thresholds']
        for op, threshold in thresholds.items():
            key = row['model']+':'+row['cohort']+':'+op
            c = counters.setdefault(key, Counter())
            c['count'] += 1
            # Saved group logits are FP32, while this diagnostic uses a double
            # sigmoid. Do not count boundary-sensitive reconstructed decisions.
            if abs(probability-threshold) <= 2e-7:
                c['boundary_ambiguous'] += 1
                cases.append(dict(pair_id=row['pair_id'], comparison=key,
                    boundary_ambiguous=True, first_group_probability=probability))
                continue
            single = row['decision_valid'] and probability >= threshold
            multi = row['accepted'][op]
            if single and not multi:
                raise ValueError('maximum-group monotonicity violated')
            c['single_accepted'] += int(single)
            c['multi_accepted'] += int(multi)
            if multi and not single:
                c['added_positive' if row['label'] else 'added_negative'] += 1
                gt = row['gt_diagnostic']
                if gt:
                    c['added_positive_production_correct'] += int(gt['production_layout20'])
                    c['added_positive_highest_scorer_correct'] += int(gt['highest_scorer_layout20'])
                cases.append(dict(pair_id=row['pair_id'], comparison=key,
                    label=row['label'], single_probability=probability,
                    multi_probability=row['score'], selected_group_rank=row['selected_group_rank'],
                    production_layout20=None if gt is None else gt['production_layout20'],
                    hypothetical_highest_scorer_layout20=None if gt is None else gt['highest_scorer_layout20']))
    result = dict(schema='fixed-weight-multi-vs-first-readout/1', status='complete',
        no_training=True, no_inference=True, no_threshold_fit=True, no_deployment_change=True,
        source=str(source.resolve()), input_sha256={f: hashlib.sha256((source/f).read_bytes()).hexdigest()
            for f in ('results.json', 'cases.jsonl')},
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        boundary_probability_tolerance=2e-7, groups={k:dict(v) for k,v in counters.items()}, cases=cases,
        caveats=['Same trained multi-candidate weights, own frozen threshold; not a retrained single-candidate arm.',
            'Rank 1 is original target-blind proposal order, not GT or Scorer rank.',
            'Hypothetical pose correctness is not deployed; OOD has no GT.',
            'Boundary-sensitive reconstructed probabilities excluded from decision comparison and explicitly counted.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    r = run(args.source, args.output)
    print(json.dumps(dict(status=r['status'], groups=r['groups'])))

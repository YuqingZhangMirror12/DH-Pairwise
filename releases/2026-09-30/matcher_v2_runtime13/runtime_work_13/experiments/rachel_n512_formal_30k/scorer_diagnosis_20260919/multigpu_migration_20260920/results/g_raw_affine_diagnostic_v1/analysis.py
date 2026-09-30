"""Bounded saved-output-only G0/G1 diagnostic. No Torch, model I/O or inference."""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] * (hi - x) + values[hi] * (x - lo) if hi != lo else values[lo]


def distribution(values):
    assert values and all(math.isfinite(v) for v in values)
    return dict(n=len(values), min=min(values), p10=quantile(values, .1),
                median=quantile(values, .5), p90=quantile(values, .9), max=max(values),
                near_negative_099=sum(v <= -.99 for v in values),
                near_positive_099=sum(v >= .99 for v in values),
                near_negative_0999=sum(v <= -.999 for v in values),
                near_positive_0999=sum(v >= .999 for v in values),
                outside_cosine_bounds_tolerance_1e_minus6=sum(abs(v) > 1.000001 for v in values))


def affine(rows):
    x = [r['candidate_details']['raw_similarity'] for r in rows]
    y = [r['candidate_details']['calibrated_logit'] for r in rows]
    xm, ym = math.fsum(x) / len(x), math.fsum(y) / len(y)
    variance = math.fsum((v - xm) ** 2 for v in x)
    assert variance > 0
    slope = math.fsum((a - xm) * (b - ym) for a, b in zip(x, y)) / variance
    intercept = ym - slope * xm
    errors = [abs(b - (slope * a + intercept)) for a, b in zip(x, y)]
    return dict(n=len(rows), slope=slope, intercept=intercept,
                max_abs_residual=max(errors), rms_residual=math.sqrt(math.fsum(e*e for e in errors)/len(errors)),
                positive_slope=slope > 0, within_fp32_absolute_tolerance_2e_minus6=max(errors) <= 2e-6)


def auc(pos, neg, getter):
    negatives = sorted(getter(r) for r in neg)
    wins = ties = 0
    for row in pos:
        score = getter(row)
        lo, hi = bisect.bisect_left(negatives, score), bisect.bisect_right(negatives, score)
        wins += lo
        ties += hi - lo
    return dict(auroc=(wins + .5 * ties) / (len(pos) * len(neg)),
                wins=wins, ties=ties, positive_count=len(pos), negative_count=len(neg))


def run(input_root):
    result = dict(schema='g-saved-raw-affine/1', status='complete',
                  source='complete saved pair_results only; no fitting of classifier or operating thresholds',
                  quantile='linear interpolation on sorted values, index (n-1)*q',
                  affine='OLS estimates from saved raw/calibrated pairs, not checkpoint parameter reads',
                  near_endpoint='descriptive raw <= -0.99 or >= 0.99; stricter 0.999 also counted; not an internal-causality test',
                  checkpoints={}, limitations=[
                      'Endpoint-near aggregate LME scores do not establish token/gradient saturation or its cause.',
                      'Positive affine calibration preserves raw ranking except finite-precision ties; it can change fixed-threshold acceptance.',
                      'OOD has positives only and no layout ground truth; no OOD AUROC, F1 or layout correctness is inferred.',
                      'REAL kept-positive and two negative cohorts have distinct definitions; excluded positives are not included in their distributions.',
                      'No new inference, training, threshold selection, heatmap or model-weight read.'])
    reference_ids = {}
    for arm in ('G0', 'G1'):
        for endpoint in ('c8', 'c16'):
            key = f'{arm}_{endpoint}'
            splits, sources, checkpoint_shas = {}, [], set()
            for split, expected in (('test', 3000), ('real', 1016), ('ood', 301)):
                folder = input_root / arm / 'evaluation' / endpoint / split
                protocol = json.loads((folder / 'protocol.json').read_text())
                summary = json.loads((folder / 'summary.json').read_text())
                assert protocol['status'] == summary['status'] == 'complete'
                checkpoint_shas.add(summary['model']['checkpoint_sha256'])
                rows = [json.loads(line) for line in (folder / 'pair_results.jsonl').read_text().splitlines()]
                ids = {r['pair_id']: (r['fragment_a'], r['fragment_b'], r['label'],
                                    r.get('review_status'), r.get('strict_member')) for r in rows}
                assert len(rows) == len(ids) == expected
                if split in reference_ids:
                    assert ids == reference_ids[split]
                else:
                    reference_ids[split] = ids
                for r in rows:
                    d = r['candidate_details']
                    assert all(math.isfinite(d[f]) for f in ('raw_similarity', 'calibrated_logit', 'final_logit'))
                    assert math.isfinite(r['classification']['fused'])
                splits[split] = rows
                sources.append(dict(split=split, files={n: dict(path=str((folder/n).resolve()), sha256=sha(folder/n))
                    for n in ('protocol.json', 'summary.json', 'pair_results.jsonl')}))
            assert len(checkpoint_shas) == 1
            test, real, ood = (splits[s] for s in ('test', 'real', 'ood'))
            tp = [r for r in test if r['label']]
            tn = [r for r in test if not r['label']]
            rp = [r for r in real if r['label'] and r['review_status'] == 'keep']
            sn = [r for r in real if not r['label'] and r['strict_member']]
            cn = [r for r in real if not r['label'] and not r['strict_member']]
            assert tuple(map(len, (tp, tn, rp, sn, cn))) == (1500, 1500, 295, 39, 469)
            assert all(r['label'] and r['target_translation_rc'] is None for r in ood)
            all_rows = test + real + ood
            populations = dict(test_all=test, test_positive=tp, test_negative=tn,
                               real_kept_positive=rp, real_strict_negative=sn,
                               real_constructed_negative=cn, ood_positive_layout_unknown=ood)
            fit = affine(all_rows)
            by_split = {s: affine(rs) for s, rs in splits.items()}
            for s, rs in splits.items():
                by_split[s]['max_abs_residual_using_global_affine'] = max(abs(
                    r['candidate_details']['calibrated_logit'] - (fit['slope'] * r['candidate_details']['raw_similarity'] + fit['intercept'])) for r in rs)
            getters = dict(raw=lambda r: r['candidate_details']['raw_similarity'],
                           calibrated_logit=lambda r: r['candidate_details']['calibrated_logit'],
                           probability=lambda r: r['classification']['fused'])
            rankings = {p: {f: auc(a, b, get) for f, get in getters.items()}
                        for p, a, b in [('test3000', tp, tn), ('real_keep803', rp, sn+cn), ('real_strict334', rp, sn)]}
            def sigmoid(v):
                return 1/(1+math.exp(-v)) if v >= 0 else math.exp(v)/(1+math.exp(v))
            result['checkpoints'][key] = dict(checkpoint_sha256=checkpoint_shas.pop(),
                sources=sources, total_rows=len(all_rows), global_affine=fit, split_affine=by_split,
                training_valid_count=sum(r['candidate_details']['training_valid'] for r in all_rows),
                decision_valid_count=sum(r['decision_valid'] for r in all_rows),
                calibrated_final_unequal_count=sum(r['candidate_details']['calibrated_logit'] != r['candidate_details']['final_logit'] for r in all_rows),
                max_abs_probability_sigmoid_error=max(abs(r['classification']['fused'] - sigmoid(r['candidate_details']['final_logit'])) for r in all_rows),
                probability_min=min(r['classification']['fused'] for r in all_rows),
                probability_max=max(r['classification']['fused'] for r in all_rows),
                global_raw=distribution([r['candidate_details']['raw_similarity'] for r in all_rows]),
                raw_distributions={p: distribution([r['candidate_details']['raw_similarity'] for r in rs]) for p, rs in populations.items()},
                ranking=rankings)
    result['exact_pair_ids_endpoints_labels_review_membership_equal'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path(__file__).resolve().parents[1]/'g_adaptation')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results.json'))
    args = parser.parse_args()
    assert quantile([0, 10], .1) == 1 and quantile([0, 10], .9) == 9
    assert auc([1, 2], [1, 3], lambda x: x)['auroc'] == .375
    result = run(args.input_root)
    result['analysis_script_sha256'] = sha(Path(__file__))
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checkpoints=len(result['checkpoints']), output=str(args.output))))


if __name__ == '__main__':
    main()

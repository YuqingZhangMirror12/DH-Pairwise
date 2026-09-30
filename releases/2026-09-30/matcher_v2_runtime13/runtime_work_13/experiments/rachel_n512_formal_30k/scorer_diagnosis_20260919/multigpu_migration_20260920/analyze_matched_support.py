"""Offline support-stratified diagnosis of a frozen matched-endpoint Scorer.

Counts are selected contour tokens, NOT pixel seam lengths. Conditioning on
correct Layout is an end-to-end failure diagnostic, not an unbiased task metric
or a causal intervention. Never fit a threshold or infer OOD layout correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from collections import Counter

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.endpoint_compare_v1 import compare

BINS = ((1, 16), (17, 32), (33, 64), (65, 128), (129, 512))
FEATURES = ('min_endpoints', 'max_endpoints', 'inlier_edges', 'residual_px',
            'runner_up_support_ratio', 'support_weight', 'overlap_small_fraction', 'area_ratio')


def median(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(values) if values else None


def cell(rows):
    accepted = sum(r['accepted'] for r in rows)
    return dict(n=len(rows), accepted=accepted, rejected=len(rows)-accepted,
                acceptance_rate=accepted/len(rows) if rows else None,
                median_score=median(rows, 'score'),
                medians={k: median(rows, k) for k in FEATURES},
                missing={k: sum(r.get(k) is None for r in rows) for k in FEATURES})


def group(rows):
    bins = {f'{lo}_{hi}': cell([r for r in rows if lo <= r['min_endpoints'] <= hi])
            for lo, hi in BINS}
    if sum(c['n'] for c in bins.values()) != len(rows):
        raise ValueError('unexpected endpoint count outside predefined bins')
    return dict(overall=cell(rows), by_min_endpoints=bins,
                accepted=cell([r for r in rows if r['accepted']]),
                rejected=cell([r for r in rows if not r['accepted']]))


def analyze(root):
    endpoints = {split: compare.load_endpoint(root/split, split) for split in ('test', 'real', 'ood')}
    if any(e['status'] != 'complete' for e in endpoints.values()):
        raise ValueError('all three authoritative endpoints must be complete')
    threshold = endpoints['test']['thresholds']['max_f1']
    if any(e['thresholds']['max_f1'] != threshold for e in endpoints.values()):
        raise ValueError('threshold differs across splits')
    records = []
    for split, endpoint in endpoints.items():
        for r in endpoint['rows'].values():
            c = r['candidate_details']; layout = r['layouts']['full_top2_mode']; d = layout['diagnostics']
            a, b = c['selected_token_count_a'], c['selected_token_count_b']
            if any(type(x) is not int or x < 1 or x > 512 for x in (a, b)):
                raise ValueError('invalid selected endpoint count')
            if c['used_fallback'] or not c['has_decoded_candidate'] or not r['decision_valid']:
                raise ValueError('fallback/invalid case requires explicit separate analysis')
            records.append(dict(pair_id=r['pair_id'], split=split, label=bool(r['label']),
                review_status=r.get('review_status'), strict_member=r.get('strict_member'),
                score=r['classification']['fused'], accepted=r['classification']['fused'] >= threshold,
                layout20=compare.layout_state(r) if split != 'ood' and r['label'] else None,
                min_endpoints=min(a, b), max_endpoints=max(a, b), inlier_edges=c['inlier_edge_count'],
                residual_px=d.get('residual_px'), runner_up_support_ratio=d.get('runner_up_support_ratio'),
                support_weight=d.get('support_weight'), overlap_small_fraction=layout.get('overlap_small_fraction'),
                area_ratio=r.get('area_ratio')))
    selectors = {
        'test_positive_good_layout': lambda r: r['split']=='test' and r['label'] and r['layout20'] is True,
        'test_negative': lambda r: r['split']=='test' and not r['label'],
        'real_keep_positive_good_layout': lambda r: r['split']=='real' and r['label'] and r['review_status']=='keep' and r['layout20'] is True,
        'real_keep_positive_bad_layout': lambda r: r['split']=='real' and r['label'] and r['review_status']=='keep' and r['layout20'] is False,
        'real_strict_negative': lambda r: r['split']=='real' and not r['label'] and r['strict_member'],
        'real_constructed_distractor': lambda r: r['split']=='real' and not r['label'] and not r['strict_member'],
        'ood_positive_unknown_layout': lambda r: r['split']=='ood' and r['label'],
    }
    groups = {name: group([r for r in records if select(r)]) for name, select in selectors.items()}
    real = groups['real_keep_positive_good_layout']; test = groups['test_positive_good_layout']
    bins = real['by_min_endpoints']
    supported = [name for name,c in bins.items() if c['n'] and test['by_min_endpoints'][name]['n']]
    common_n = sum(bins[name]['n'] for name in supported)
    standardized = sum(bins[name]['n'] * test['by_min_endpoints'][name]['acceptance_rate'] for name in supported)
    result = dict(schema='matched-support-stratification/1', status='complete', threshold=threshold,
        bins=BINS, selection='own frozen SIMVAL max-F1, unchanged across all three endpoints',
        groups=groups, source={split: dict(path=e['path'], sources=e['sources']) for split,e in endpoints.items()},
        descriptive_standardization=dict(common_real_good_n=common_n,
            expected_accepts_using_TEST_within_bin_rates=standardized,
            observed_accepts_same_bins=sum(bins[name]['accepted'] for name in supported)),
        caveats=[
            'Endpoint count is neither independent seam length nor a sampling-density measurement.',
            'Broad-bin standardization is an accounting comparison, not causal attribution; within-bin distributions differ.',
            'Correct-layout conditioning is post-model diagnostic selection, not overall classification performance.',
            'Original negative labels and constructed cross-case distractors are separated; the latter are not independently verified GT negatives.',
            'OOD has only positives and no layout GT; no OOD Accuracy, F1 or layout-success label is inferred.',
            'All inputs are frozen original model outputs; no training, new inference or threshold fitting.'])
    return result, records


def training_support(diagnostic_root, training_protocol):
    summary = json.loads((diagnostic_root/'summary.json').read_text())
    protocol = json.loads(training_protocol.read_text())
    if summary['status'] != 'complete' or protocol['status'] != 'complete':
        raise ValueError('incomplete source')
    source = summary['sources']; binding = protocol['identity']['cache_bindings']['train']
    expected = {'cache':binding['root'], 'cache_protocol_sha256':binding['protocol_sha256'],
        'source_checkpoint_sha256':binding['source_checkpoint_sha256'],
        'manifest_sha256':binding['population']['manifest_sha256']}
    if any(source[k] != v for k,v in expected.items()):
        raise ValueError('training diagnostic and actual Scorer training cache differ')
    rows = [json.loads(line) for line in (diagnostic_root/'rows.jsonl').read_text().splitlines()]
    if len(rows) != 24000 or len({r['pair_id'] for r in rows}) != 24000 or sum(r['label'] for r in rows) != 12000:
        raise ValueError('wrong training population')
    bins = {}
    for lo,hi in BINS:
        selected = [r for r in rows if lo <= r['unique_endpoints_min'] <= hi]
        positive = [r for r in selected if r['label']]
        bins[f'{lo}_{hi}'] = dict(n=len(selected), positive=len(positive), negative=len(selected)-len(positive),
            positive_fraction=len(positive)/len(selected) if selected else None,
            layout20_good_positive=sum(r['raw_layout20_success'] for r in positive),
            positive_recipes=dict(Counter(r['s7_recipe'] for r in positive)),
            actually_augmented_positive=sum(r['actual_changed_pair'] for r in positive))
    if sum(c['n'] for c in bins.values()) != len(rows):
        raise ValueError('unaccounted training endpoints')
    return dict(n=len(rows), positives=12000, negatives=12000, by_min_endpoints=bins,
        source=dict(diagnostic=str(diagnostic_root), protocol=str(training_protocol), **expected),
        definition='Unique endpoints from the same final predicted inlier edges; labels remain original pair adjacency.',
        limitations='Matcher has seen these training pairs. No Scorer score is inferred for training rows.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-diagnostics', type=Path)
    p.add_argument('--training-protocol', type=Path)
    args = p.parse_args()
    result, records = analyze(args.endpoint_root.resolve())
    if bool(args.train_diagnostics) != bool(args.training_protocol):
        p.error('train diagnostics and training protocol must be supplied together')
    if args.train_diagnostics:
        result['training_support'] = training_support(args.train_diagnostics.resolve(), args.training_protocol.resolve())
    result['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output/'cases.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
    print(json.dumps({'status':result['status'], 'threshold':result['threshold'],
        'groups':{n:g['overall'] for n,g in result['groups'].items()},
        'descriptive_standardization':result['descriptive_standardization']}))


if __name__ == '__main__':
    main()

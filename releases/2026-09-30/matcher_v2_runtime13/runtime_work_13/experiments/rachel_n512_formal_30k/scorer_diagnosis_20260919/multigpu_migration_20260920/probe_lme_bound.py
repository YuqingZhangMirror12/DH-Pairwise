"""Saved G-head LME outputs bound the largest post-CA cosine, without inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path


ALLOWANCE = 1e-4  # Conservative descriptive FP32 allowance, not interval arithmetic.


def maximum_upper(raw, cap=512, tau=15.):
    if not math.isfinite(raw) or cap < 1 or tau <= 0:
        raise ValueError('finite raw, positive cap/tau required')
    return min(1. + ALLOWANCE, raw + math.log(cap * cap) / tau + ALLOWANCE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(row, split):
    if split == 'test':
        return ['test_positive' if row['label'] else 'test_negative']
    if split == 'ood':
        assert row['label'] and row['target_translation_rc'] is None
        return ['ood_positive_layout_unknown']
    if not row['label']:
        return ['real_original_negative' if row['strict_member'] else 'real_constructed_distractor']
    if row['review_status'] != 'keep':
        return ['real_excluded_positive']
    layout = row['layouts']['full_top2_mode']
    error = layout['translation_l2_px']
    assert error is not None and math.isfinite(error)
    return ['real_kept_positive', 'real_kept_layout20_correct' if layout['valid'] and error <= 20.
            else 'real_kept_layout20_incorrect']


def run(root):
    output = dict(schema='saved-lme-maximum-bound/1', status='complete', no_inference=True,
        no_training=True, no_threshold_fit=True, floating_point_allowance=ALLOWANCE,
        formula='max(cos) <= raw_LME + log(Na*Nb)/tau <= raw_LME + log(cap^2)/tau',
        applies_to='G0/G1 post-Scorer-Cross-Attention cosine grid, NOT Matcher affinity or Sinkhorn',
        limitations=['Mathematical bound from the recorded readout, not a recovered heatmap.',
            'A nonpositive cosine does not prove a pair or seam is physically nonmatching.',
            'Positive upper bound does not prove a high cosine entry exists.',
            'Does not locate a failing network layer or establish feature/gradient saturation.',
            'The 1e-4 numerical allowance is descriptive, not certified interval arithmetic.'],
        checkpoints={})
    for arm in ('G0', 'G1'):
        for budget in ('c8', 'c16'):
            population, cases, sources, checkpoints = {}, [], [], set()
            for split, count in (('test', 3000), ('real', 1016), ('ood', 301)):
                folder = root / arm / 'evaluation' / budget / split
                summary = json.loads((folder/'summary.json').read_text())
                protocol = json.loads((folder/'protocol.json').read_text())
                assert summary['status'] == protocol['status'] == 'complete'
                model = summary['model']
                checkpoints.add(model['checkpoint_sha256'])
                assert model['model_design']['tau'] == 15.
                assert model['model_config']['contour_cap'] == 512
                assert model['model_design']['readout'] == 'CA tokens -> L2 cosine grid -> LME raw_similarity -> positive affine calibrated_logit'
                threshold = model['operating_points']['thresholds']['max_f1']
                rows = [json.loads(line) for line in (folder/'pair_results.jsonl').read_text().splitlines()]
                assert len(rows) == len({r['pair_id'] for r in rows}) == count
                for row in rows:
                    raw = row['candidate_details']['raw_similarity']
                    assert -1.00001 <= raw <= 1.00001
                    assert row['decision_valid'] and row['candidate_details']['training_valid']
                    bound = maximum_upper(raw)
                    accepted = row['classification']['fused'] >= threshold
                    for group in groups(row, split):
                        record = population.setdefault(group, dict(n=0, max_cos_upper_below_zero=0,
                            max_cos_upper_below_half=0, upper_below_zero_accepted=0))
                        record['n'] += 1
                        record['max_cos_upper_below_zero'] += int(bound < 0.)
                        record['max_cos_upper_below_half'] += int(bound < .5)
                        record['upper_below_zero_accepted'] += int(bound < 0. and accepted)
                    if bound < 0.:
                        cases.append(dict(split=split, pair_id=row['pair_id'], groups=groups(row, split),
                            raw_similarity=raw, max_cos_upper_bound=bound,
                            score=row['classification']['fused'], accepted_at_frozen_max_f1=accepted))
                sources.append(dict(split=split, files={name:dict(path=str((folder/name).resolve()),
                    sha256=digest(folder/name)) for name in ('summary.json','protocol.json','pair_results.jsonl')}))
            assert len(checkpoints) == 1
            output['checkpoints'][arm+'_'+budget] = dict(checkpoint_sha256=next(iter(checkpoints)),
                cap=512, tau=15., upper_bound_addend=math.log(512**2)/15.+ALLOWANCE,
                populations=population, cases_with_negative_upper_bound=cases, sources=sources)
    return output


def self_test():
    # Check the inequality on constant grids and a sparse high-entry grid.
    for values in ([.2]*4, [-.95]*64, [.9]+[-.9]*63, [-.2,.1,.4,-.8]):
        top = max(values)
        raw = top + math.log(sum(math.exp(15*(v-top)) for v in values))/15 - math.log(len(values))/15
        cap = math.ceil(math.sqrt(len(values)))
        assert raw <= top + 1e-10 and top <= maximum_upper(raw, cap)
    assert maximum_upper(-.95) < 0.
    assert maximum_upper(.5) >= .5


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    self_test()
    result = run(args.input_root)
    result['analysis_script_sha256'] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v['populations'] for k,v in result['checkpoints'].items()}))

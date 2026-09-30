"""Describe saved spectral logit additions; no model inference or recalibration."""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sigmoid(x):
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def percentile(values, q):
    values = sorted(values)
    a = (len(values)-1)*q
    lo, hi = math.floor(a), math.ceil(a)
    return values[lo] if lo == hi else values[lo]*(hi-a)+values[hi]*(a-lo)


def auc(rows, score):
    pos, neg = [r for r in rows if r['label']], [r for r in rows if not r['label']]
    if not pos or not neg:
        return None
    neg = sorted(score(r) for r in neg)
    wins = sum((bisect.bisect_left(neg, score(r))+bisect.bisect_right(neg, score(r)))/2 for r in pos)
    return wins/(len(pos)*len(neg))


def describe(rows, threshold):
    delta = [r['candidate_details']['spectral_residual_logit'] for r in rows]
    z = math.log(threshold/(1-threshold))
    before = [r['candidate_details']['ca_logit'] >= z for r in rows]
    after = [r['classification']['fused'] >= threshold for r in rows]
    return dict(n=len(rows), delta_p10=percentile(delta,.1), delta_median=percentile(delta,.5),
        delta_p90=percentile(delta,.9), delta_min=min(delta), delta_max=max(delta),
        mean_abs_delta=math.fsum(map(abs,delta))/len(delta),
        source_full_threshold=threshold,
        full_accept=sum(after), arithmetic_without_branch_accept=sum(before),
        added_accept=sum(not a and b for a,b in zip(before,after)),
        removed_accept=sum(a and not b for a,b in zip(before,after)),
        auc_ca_only=auc(rows,lambda r:r['candidate_details']['ca_logit']),
        auc_full=auc(rows,lambda r:r['classification']['fused']))


def run(root):
    result = dict(schema='saved-spectral-addition-diagnostic/1',status='complete',
        no_inference=True,no_training=True,no_threshold_fit=True,
        arithmetic_intervention='remove saved spectral_residual_logit; keep the original full-model SIMVAL threshold',
        caveats=['CA parameters were co-trained with the residual; ca_only is not an independently trained baseline.',
            'No ca_only recalibration. Acceptance changes describe immediate arithmetic, not a new model selection result.',
            'This residual contains mass AND spectral inputs in mass_spectral; removing it is not spectral-only ablation.',
            'All-token CA branch can change across arms; compare registered complete arms separately.',
            'Positive-only OOD has no AUROC or layout-accuracy inference.'],arms={})
    for arm in ('zero_c16','mass_c16','mass_spectral_c16'):
        populations, sources, thresholds, total = {}, [], set(), []
        for split, n in (('test',3000),('real',1016),('ood',301)):
            folder = root/arm/'evaluation/fixed_epoch'/split
            summary=json.loads((folder/'summary.json').read_text())
            protocol=json.loads((folder/'protocol.json').read_text())
            assert summary['status']==protocol['status']=='complete'
            threshold=summary['model']['operating_points']['thresholds']['max_f1']
            thresholds.add(threshold)
            rows=[json.loads(line) for line in (folder/'pair_results.jsonl').read_text().splitlines()]
            assert len(rows)==len({r['pair_id'] for r in rows})==n
            for row in rows:
                d=row['candidate_details']
                assert row['decision_valid'] and len(d['spectral_branch_input'])==10
                assert all(math.isfinite(x) for x in (d['ca_logit'],d['spectral_residual_logit'],*d['spectral_branch_input']))
                assert abs(sigmoid(d['ca_logit']+d['spectral_residual_logit'])-row['classification']['fused']) < 2e-7
            total.extend(rows)
            if split=='test':
                populations['test3000']=rows
                populations['test_positive']=[r for r in rows if r['label']]
                populations['test_negative']=[r for r in rows if not r['label']]
            elif split=='ood':
                assert all(r['label'] and r['target_translation_rc'] is None for r in rows)
                populations['ood_positive_layout_unknown']=rows
            else:
                kept=[r for r in rows if r['label'] and r['review_status']=='keep']
                original=[r for r in rows if not r['label'] and r['strict_member']]
                distractor=[r for r in rows if not r['label'] and not r['strict_member']]
                assert tuple(map(len,(kept,original,distractor)))==(295,39,469)
                populations.update(real_keep803=kept+original+distractor,
                    real_strict334=kept+original,real_kept_positive=kept,
                    real_original_negative=original,real_constructed_distractor=distractor,
                    real_kept_layout20_correct=[r for r in kept if r['layouts']['full_top2_mode']['valid']
                        and r['layouts']['full_top2_mode']['translation_l2_px'] <=20.])
            sources.append(dict(split=split,files={f:dict(path=str((folder/f).resolve()),sha256=digest(folder/f))
                for f in ('protocol.json','summary.json','pair_results.jsonl')}))
        assert len(thresholds)==1
        threshold=next(iter(thresholds))
        result['arms'][arm]=dict(populations={k:describe(v,threshold) for k,v in populations.items()},
            nonzero_input_rows_by_dimension=[sum(r['candidate_details']['spectral_branch_input'][i]!=0 for r in total) for i in range(10)],
            source_count=len(total),sources=sources)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert percentile([0,10],.1)==1
    assert auc([dict(label=True,x=2),dict(label=False,x=2)],lambda r:r['x'])==.5
    result=run(args.input_root)
    result['script_sha256']=digest(Path(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({k:{g:v['populations'][g] for g in ('real_keep803','real_kept_layout20_correct','ood_positive_layout_unknown')}
        for k,v in result['arms'].items()}))

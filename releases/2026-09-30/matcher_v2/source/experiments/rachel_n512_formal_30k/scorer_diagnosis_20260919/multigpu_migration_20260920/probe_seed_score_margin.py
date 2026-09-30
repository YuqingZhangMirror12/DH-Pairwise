"""Saved-score decomposition, not a new threshold selection or inference."""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit(p):
    assert 0 < p < 1
    return math.log(p / (1-p))


def sigmoid(x):
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def quantiles(xs):
    if not xs:
        return None
    xs = sorted(xs)
    def at(q):
        a = (len(xs)-1)*q
        lo, hi = math.floor(a), math.ceil(a)
        return xs[lo]*(hi-a)+xs[hi]*(a-lo) if lo != hi else xs[lo]
    return dict(n=len(xs), p10=at(.1), median=at(.5), p90=at(.9),
                minimum=xs[0], maximum=xs[-1])


def load_endpoint(folder):
    summary = json.loads((folder/'summary.json').read_text())
    protocol = json.loads((folder/'protocol.json').read_text())
    assert summary['status'] == protocol['status'] == 'complete'
    rows = [json.loads(line) for line in (folder/'pair_results.jsonl').read_text().splitlines()]
    keyed = {r['pair_id']: r for r in rows}
    assert len(rows) == len(keyed)
    for r in rows:
        assert r['decision_valid'] and not r['candidate_details']['used_fallback']
        z = r['candidate_details']['deployed_logit']
        assert math.isfinite(z)
        assert abs(sigmoid(z)-r['classification']['fused']) < 2e-7
    return keyed, summary['model']['operating_points']['thresholds'], {
        f: dict(path=str((folder/f).resolve()), sha256=sha(folder/f))
        for f in ('pair_results.jsonl', 'summary.json', 'protocol.json')}


def pair_auc(positives, negatives, arm):
    neg = sorted(x[arm]['candidate_details']['deployed_logit'] for x in negatives)
    if not positives or not neg:
        return None
    wins = sum((bisect.bisect_left(neg, x[arm]['candidate_details']['deployed_logit'])
              + bisect.bisect_right(neg, x[arm]['candidate_details']['deployed_logit']))/2
               for x in positives)
    return wins/(len(positives)*len(neg))


def main(root, reference='matched_tokens', new_arm='edge_seed'):
    output = dict(schema='saved-seed-score-margin/1', no_inference=True,
        no_training=True, no_threshold_fit=True, populations={}, sources={},
        reference_arm=reference, new_arm=new_arm,
        caveats=[
            'Arithmetic transplantation of the reference numerical threshold is NOT calibrated deployment.',
            'Raw logit changes across co-trained models are descriptive, not an isolated causal effect.',
            'AUC is computed on saved logits to avoid probability saturation ties.',
            'Candidate stage, edge encoding and supplied metadata change together.',
            'REAL reviewed positives and constructed distractors are diagnostic cohorts.',
            'OOD is positive-only with no Layout GT.'])
    for split, expected in (('test', 3000), ('real', 1016), ('ood', 301)):
        old, old_t, old_src = load_endpoint(root/reference/'evaluation/c16'/split)
        new, new_t, new_src = load_endpoint(root/new_arm/'evaluation/c16'/split)
        assert old.keys() == new.keys() and len(old) == expected
        rows = []
        for key in old:
            a,b = old[key],new[key]
            for field in ('label','review_status','strict_member','fragment_a','fragment_b',
                          'target_translation_rc','layouts'):
                assert (field in a) == (field in b), (key,field,'presence')
                if field in a:
                    assert a[field] == b[field], (key,field)
            rows.append((a,b))
        if split == 'real':
            pos = [x for x in rows if x[0]['label'] and x[0]['review_status']=='keep']
            original = [x for x in rows if not x[0]['label'] and x[0]['strict_member']]
            distractor = [x for x in rows if not x[0]['label'] and not x[0]['strict_member']]
            good = [x for x in pos if x[0]['layouts']['full_top2_mode']['valid'] and
                    x[0]['layouts']['full_top2_mode']['translation_l2_px'] <= 20]
            good_ids = {x[0]['pair_id'] for x in good}
            wrong = [x for x in pos if x[0]['pair_id'] not in good_ids]
            assert (len(pos),len(original),len(distractor),len(good)) == (295,39,469,216)
            groups = dict(real_keep803=pos+original+distractor, real_positive=pos,
                          real_layout20_good=good, real_layout20_wrong=wrong,
                          real_original_negative=original,
                          real_distractor=distractor)
            output['rank_checks'] = {name: {a:pair_auc(p,n,i) for i,a in enumerate((reference,new_arm))}
                for name,p,n in [('positive_vs_original',pos,original),
                                  ('positive_vs_distractor',pos,distractor),
                                  ('layoutgood_vs_original',good,original),
                                  ('layoutgood_vs_distractor',good,distractor),
                                  ('layoutwrong_vs_original',wrong,original),
                                  ('layoutwrong_vs_distractor',wrong,distractor)]}
        else:
            groups = {split: rows}
        output['sources'][split] = dict(reference=old_src,new=new_src)
        for name, group in groups.items():
            result = dict(n=len(group), working_points={})
            for point in ('max_f1','recall_95','recall_99'):
                a,b = old_t[point],new_t[point]
                za,zb = logit(a),logit(b)
                classes = {k:[] for k in ('both_accept','lost_accept','gained_accept','both_reject')}
                for oldrow,newrow in group:
                    ap,bp = oldrow['classification']['fused']>=a,newrow['classification']['fused']>=b
                    key = ('both_accept' if ap and bp else 'lost_accept' if ap else
                           'gained_accept' if bp else 'both_reject')
                    classes[key].append((oldrow,newrow))
                descriptions={}
                for key,subset in classes.items():
                    oz=[x[0]['candidate_details']['deployed_logit'] for x in subset]
                    nz=[x[1]['candidate_details']['deployed_logit'] for x in subset]
                    descriptions[key]=dict(n=len(subset),
                        source_logit=quantiles(oz),new_logit=quantiles(nz),
                        logit_change=quantiles([b-a for a,b in zip(oz,nz)]),
                        source_margin=quantiles([z-za for z in oz]),
                        new_margin=quantiles([z-zb for z in nz]),
                        new_accepts_at_reference_numerical_threshold=sum(z>=za for z in nz),
                        pair_ids=[x[0]['pair_id'] for x in subset] if key in ('lost_accept','gained_accept') else None)
                result['working_points'][point]=dict(reference_threshold=a,new_threshold=b,
                    threshold_logit_change=zb-za,
                    identity='new_margin - reference_margin = logit_change - threshold_logit_change',
                    classes=descriptions)
            output['populations'][name]=result
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',default='matched_tokens',choices=('all_tokens','matched_tokens','edge_seed','matched_edges'))
    parser.add_argument('--new',default='edge_seed',choices=('all_tokens','matched_tokens','edge_seed','matched_edges'))
    args=parser.parse_args()
    if args.reference == args.new:
        parser.error('reference and new must differ')
    result=main(args.root,args.reference,args.new)
    result['script_sha256']=sha(Path(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    good=result['populations']['real_layout20_good']['working_points']
    print(json.dumps(dict(rank_checks=result['rank_checks'], good={k:{
        'threshold_logit_change':v['threshold_logit_change'],
        'lost':v['classes']['lost_accept']['n'],
        'lost_logit_change':v['classes']['lost_accept']['logit_change'],
        'lost_new_accept_at_old_numerical_threshold':v['classes']['lost_accept']['new_accepts_at_reference_numerical_threshold']}
        for k,v in good.items()}),ensure_ascii=False))

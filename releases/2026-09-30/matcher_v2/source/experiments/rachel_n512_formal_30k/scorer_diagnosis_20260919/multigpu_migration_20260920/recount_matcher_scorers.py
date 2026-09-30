"""Independent raw-row M12/M16/M20 C16 recount; no comparator imports."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

OPS = ('max_f1', 'recall_99')
SPLITS = {'test': (3000, 1500), 'real': (1016, 508), 'ood': (301, 301)}


def load(root, split):
    root = root / split
    protocol = json.loads((root/'protocol.json').read_text())
    model = protocol['model']
    if protocol['status'] != 'complete' or model['selection'] != 'fixed_epoch' or model['head_epoch'] != 16:
        raise ValueError('completed fixed C16 endpoint required')
    rows, digest = {}, hashlib.sha256()
    with (root/'pair_results.jsonl').open('rb') as stream:
        for line in stream:
            digest.update(line)
            raw = json.loads(line)
            pair = raw['pair_id']
            if pair in rows:
                raise ValueError('duplicate pair ID')
            layout = raw['layouts']['full_top2_mode']
            score = raw['classification']['fused']
            if not math.isfinite(score):
                raise ValueError('nonfinite probability')
            rows[pair] = dict(pair_id=pair, label=bool(raw['label']), score=score,
                decision_valid=bool(raw['decision_valid']), strict_member=raw.get('strict_member'),
                review_status=raw.get('review_status'), target_translation_rc=raw.get('target_translation_rc'),
                layout_valid=bool(layout['valid']), layout_error=layout['translation_l2_px'],
                predicted_translation_rc=layout['translation_rc'])
    if (len(rows), sum(r['label'] for r in rows.values())) != SPLITS[split] or len(rows) != protocol['sample_count']:
        raise ValueError('complete split counts differ')
    thresholds = {op: model['operating_points']['thresholds'].get(op) for op in OPS}
    return rows, thresholds, dict(directory=str(root), pair_results_sha256=digest.hexdigest(),
        protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
        matcher_sha256=model['source_matcher_sha256'], head_sha256=model['checkpoint_sha256'],
        thresholds=thresholds, threshold_policy='each model own frozen clean SIMVAL operating point; no fit')


def populations(rows, split):
    if split != 'real':
        return {split: list(rows)}
    result = dict(real_all=list(rows), real_keep=[k for k, r in rows.items() if not r['label'] or r['review_status']=='keep'],
        real_kept_positive=[k for k,r in rows.items() if r['label'] and r['review_status']=='keep'],
        real_original_negative=[k for k,r in rows.items() if not r['label'] and r['strict_member']],
        real_constructed_negative=[k for k,r in rows.items() if not r['label'] and not r['strict_member']])
    if [len(result[k]) for k in ('real_all','real_keep','real_kept_positive','real_original_negative','real_constructed_negative')] != [1016,803,295,39,469]:
        raise ValueError('review and negative source populations differ')
    return result


def layout_state(row):
    if not row['label'] or row['target_translation_rc'] is None:
        return 'unknown'
    e = row['layout_error']
    return 'good' if row['layout_valid'] and e is not None and math.isfinite(e) and e <= 20 else 'bad'


def accepted(row, threshold):
    return row['decision_valid'] and row['score'] >= threshold


def auc(rows):
    positives = sum(r['label'] for r in rows)
    negatives = len(rows)-positives
    if not positives or not negatives:
        return None
    groups = {}
    for row in rows:
        score = row['score'] if row['decision_valid'] else -1.0
        groups.setdefault(score, [0, 0])[int(row['label'])] += 1
    below, wins = 0, 0.
    for _, (n, p) in sorted(groups.items()):
        wins += p * (below + .5*n)
        below += n
    return wins/(positives*negatives)


def metrics(rows, threshold):
    positives = [r for r in rows if r['label']]
    negatives = [r for r in rows if not r['label']]
    states = Counter(layout_state(r) for r in positives)
    no_positive_layout_gt = bool(positives) and states['unknown'] == len(positives)
    result = dict(count=len(rows), positive_count=len(positives), negative_count=len(negatives),
        layout_positive_states=dict(states), layout20_correct=None if no_positive_layout_gt else states['good'],
        layout20_rate_all_positive=states['good']/len(positives) if positives and not states['unknown'] else None,
        auroc=auc(rows), threshold=threshold)
    if threshold is None:
        return dict(result, threshold_missing=True)
    tp = sum(accepted(r, threshold) for r in positives)
    fp = sum(accepted(r, threshold) for r in negatives)
    fn, tn = len(positives)-tp, len(negatives)-fp
    result.update(tp=tp, fp=fp, fn=fn, tn=tn,
        recall=tp/len(positives) if positives else None,
        fpr=fp/len(negatives) if negatives else None,
        accuracy=(tp+tn)/len(rows) if positives and negatives else None,
        precision=tp/(tp+fp) if positives and negatives and tp+fp else None,
        f1=2*tp/(2*tp+fp+fn) if positives and negatives and 2*tp+fp+fn else None,
        accepted_positive_layout_states=dict(Counter(layout_state(r) for r in positives if accepted(r,threshold))),
        accepted_correct=None if no_positive_layout_gt else sum(accepted(r,threshold) and layout_state(r)=='good' for r in positives),
        rejected_correct=None if no_positive_layout_gt else sum(not accepted(r,threshold) and layout_state(r)=='good' for r in positives))
    return result


def transitions(old, new, ids, old_t, new_t):
    positive = [k for k in ids if old[k]['label']]
    raw_cells, raw_gain, raw_loss = {}, [], []
    for k in positive:
        a, b = layout_state(old[k]), layout_state(new[k])
        raw_cells.setdefault(a+'->'+b, []).append(k)
        if a=='bad' and b=='good': raw_gain.append(k)
        if a=='good' and b=='bad': raw_loss.append(k)
    result = dict(count=len(ids), positive_count=len(positive), raw_layout_cells=raw_cells,
        raw_layout_gain_ids=raw_gain, raw_layout_loss_ids=raw_loss,
        predicted_translation_changed_count=sum(old[k]['predicted_translation_rc'] != new[k]['predicted_translation_rc'] for k in ids))
    if old_t is None or new_t is None:
        return dict(result, threshold_missing=True)
    groups = {key: [] for key in ('positive_fn_to_tp','positive_tp_to_fn','negative_fp_to_tn','negative_tn_to_fp',
                                  'accepted_correct_gain','accepted_correct_loss')}
    correct_cells = {'gain': {}, 'loss': {}}
    for k in ids:
        a, b = accepted(old[k], old_t), accepted(new[k], new_t)
        pos = old[k]['label']
        if pos and not a and b: groups['positive_fn_to_tp'].append(k)
        if pos and a and not b: groups['positive_tp_to_fn'].append(k)
        if not pos and a and not b: groups['negative_fp_to_tn'].append(k)
        if not pos and not a and b: groups['negative_tn_to_fp'].append(k)
        if pos:
            la, lb = layout_state(old[k]), layout_state(new[k])
            ca, cb = a and la=='good', b and lb=='good'
            if ca != cb:
                direction = 'gain' if cb else 'loss'
                groups['accepted_correct_'+direction].append(k)
                correct_cells[direction].setdefault(la+'->'+lb, []).append(k)
    result.update(groups=groups, counts={key: len(v) for key,v in groups.items()},
                  accepted_correct_changes_by_both_layout_states=correct_cells)
    return result


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('new independent result only')
    paths = dict(m12=root/'s7_direct/all_tokens/evaluation/c16', m16=root/'matcher_m16/scorer_c16/evaluation/c16',
                 m20=root/'matcher_m20/scorer_c16/evaluation/c16')
    report = dict(schema='independent-matcher-scorer-c16-recount/1', source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='complete', operating_points=list(OPS), methodology='raw JSONL only; no comparator calls; own frozen SIMVAL thresholds',
        caveats=['OOD positive-only; no binary Accuracy/F1 or layout ground truth',
                 'Fresh heads and Matcher differ; classification changes cannot be assigned to Matcher alone',
                 'Invalid predictions stay in denominators; no threshold fitting or model selection'], sources={}, metrics={}, transitions={})
    for split in SPLITS:
        loaded = {name: load(path, split) for name,path in paths.items()}
        baseline = loaded['m12'][0]
        for name,(rows,thresholds,source) in loaded.items():
            if rows.keys() != baseline.keys():
                raise ValueError('exact pair membership required, not intersection')
            for key,row in rows.items():
                if any(row[field] != baseline[key][field] for field in ('label','target_translation_rc','review_status','strict_member')):
                    raise ValueError('pair labels/GT/review/source identity changed')
            report['sources'].setdefault(name,{})[split] = source
        for population,ids in populations(baseline,split).items():
            report['metrics'][population] = {name: {op: metrics([rows[k] for k in ids], thresholds[op]) for op in OPS}
                                             for name,(rows,thresholds,_) in loaded.items()}
            report['transitions'][population] = {}
            for before,after in (('m12','m16'),('m16','m20'),('m12','m20')):
                a,at,_ = loaded[before]; b,bt,_ = loaded[after]
                report['transitions'][population][before+'->'+after] = {
                    op: transitions(a,b,ids,at[op],bt[op]) for op in OPS}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args(); r=run(a.root,a.output)
    print(json.dumps({'status':r['status'],'output':a.output,'populations':list(r['metrics'])}))

"""Re-use frozen E1/Full24k predictions as explicitly unequal-budget references.

No fitting, inference, remote writes, or checkpoint loading. max-F1, VAL R95,
and legacy VAL R99 thresholds remain separate; R99 is never renamed R95.
The established review cohort and current S0 pair identities are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(rows, labeled=True):
    keys = ('pair_id', 'label', 'fragment_a', 'fragment_b', 'target_translation_rc') if labeled else ('pair_id',)
    result = {row['pair_id']: [row[k] for k in keys] for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate pair ID in reference predictions')
    return result


def positive_only(rows, threshold):
    """A positive-only OOD set supports recall, not binary accuracy or layout."""
    if not rows or not math.isfinite(threshold):
        raise ValueError('Expected nonempty OOD population and finite threshold')
    if any(type(r['decision_valid']) is not bool or not math.isfinite(r['score'])
           or not 0 <= r['score'] <= 1 for r in rows):
        raise ValueError('Invalid OOD prediction')
    accepted = sum(r['decision_valid'] and r['score'] >= threshold for r in rows)
    return dict(sample_count=len(rows), positive_count=len(rows), negative_count=0,
        classification=dict(threshold=threshold, tp=accepted, fn=len(rows)-accepted,
            fp=None, tn=None, recall=accepted/len(rows), precision=None, f1=None,
            accuracy=None, ap=None, auroc=None), positive_only_layout=None,
        layout_unavailable_reason='No OOD ground-truth translation',
        decision_valid_count=sum(r['decision_valid'] for r in rows))


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if not k.endswith('_pair_ids')}
    return value


def build(workspace):
    workspace = Path(workspace).resolve()
    reports = workspace/'reports'
    curated = reports/'rachel_curated_review_20260912_001'
    bench = reports/'rachel_recall_benchmarks_20260911_001'
    formal = reports/'rachel_score_design_20260913_001'
    ood_root = reports/'turufan_ood_pairwise_20260912_001/evaluation_v1'
    spec = importlib.util.spec_from_file_location('curated_reference_reader', curated/'compare_kept_models.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sources = {}

    def source(path, lines=False):
        path = Path(path).resolve()
        sources[str(path)] = sha(path)
        return read_rows(path) if lines else read(path)

    saved = source(curated/'curated_v2/metrics_full_and_keep_only.json')
    previous = source(curated/'historical_e1_setting_comparison.json')
    if saved['status'] != 'complete' or previous['status'] != 'complete':
        raise ValueError('Historical curation/reference is incomplete')
    annotation = saved['annotation_manifest']
    keep = set(annotation['kept_positive_pair_ids'])
    negatives = set(annotation['unchanged_negative_pair_ids'])
    if len(keep) != 295 or len(negatives) != 508 or keep & negatives:
        raise ValueError('Established REAL review cohort changed')
    current = {
        'test': source(formal/'s0_s1_first5_independent_test/inputs/original/pair_results.jsonl', True),
        'real': source(formal/'s0_s1_first5_independent_real_ood/inputs/original/real/pair_results.jsonl', True),
        'ood': source(formal/'s0_s1_first5_independent_real_ood/inputs/original/ood/pair_results.jsonl', True),
    }
    result = {}
    for name, display in (('historical_e1', 'Historical E1'), ('full_e1_24k_winner', 'Full-E1-24K')):
        old = saved['models'][name]
        real_path = Path(old['sources']['pair_results'])
        real = source(real_path, True)
        if sources[str(real_path)] != old['sources']['pair_results_sha256']:
            raise ValueError('Historical REAL predictions differ from curated source')
        receipt = source(old['sources']['receipt'])
        checkpoint_sha = old['identity']['checkpoint_sha256']
        if name == 'historical_e1':
            test_root = reports/'rachel_weathering_all_20260909_001/e0_e1_completed_snapshot/e1/evaluation/test'
            train_protocol = source(reports/'rachel_weathering_all_20260909_001/e0_e1_completed_snapshot/e1/training/e1/protocol.json')
            if train_protocol['status'] != 'complete' or train_protocol['initialization'] != 'warm-start':
                raise ValueError('Historical E1 warm-start evidence differs')
            freeze = source(bench/'historical_e1_operating_points/validation_freeze.json')
            if freeze['real_or_test_opened_for_fit'] is not False or freeze['threshold_fit_split'] != 'val':
                raise ValueError('Historical threshold source must be VAL-only')
            values = freeze['thresholds']
            training = dict(selected_final_stage_epoch=train_protocol['selected_epoch'],
                final_stage_selected_exposures=train_protocol['selected_global_exposure'],
                inherited_pretraining=True, total_selected_exposures=None,
                total_exposure_note='Final-stage120K excludes inherited pretraining; not equal to random-init120K',
                seed=train_protocol['seed'], equal_budget_control=False,
                initial_checkpoint=train_protocol['initial_checkpoint'],
                checkpoint_selection_rule=train_protocol['selection_rule'])
            ood_name = 'historical_e1'
        else:
            test_root = bench/'completed/full_e1_24k/evaluation/winner/test'
            train_protocol = source(bench/'completed/full_e1_24k/training/protocol.json')
            if (train_protocol['status'] != 'complete' or train_protocol['initialization'] != 'random'
                    or train_protocol['source_weights_loaded'] is not False):
                raise ValueError('Full24K random-initialization evidence differs')
            freeze = source(bench/'completed/full_e1_24k/training/train_val_freeze.json')
            if freeze['status'] != 'complete' or freeze['test_or_real_used_for_fit'] is not False:
                raise ValueError('Full24K freeze must be complete and VAL-only')
            if freeze['checkpoint_sha256'] != checkpoint_sha:
                raise ValueError('Full24K checkpoint source mismatch')
            values = freeze['operating_points']['thresholds']
            training = dict(selected_epoch=freeze['selected_epoch'],
                total_selected_exposures=freeze['selected_global_exposure'],
                inherited_pretraining=False, seed=train_protocol['seed'], equal_budget_control=False,
                checkpoint_selection_rule=freeze['selection_rule'])
            ood_name = 'full_e1_24k'
        test = source(test_root/'pair_results.jsonl', True)
        test_receipt = source(test_root/'receipt.json')
        if sources[str((test_root/'pair_results.jsonl').resolve())] != test_receipt['pair_results_sha256']:
            raise ValueError('Historical TEST predictions differ from receipt')
        ood_doc = source(ood_root/(ood_name+'_predictions.json'))
        if (ood_doc['status'] != 'complete' or ood_doc['labels_used_by_forward'] is not False
                or ood_doc['identity']['checkpoint_sha256'] != checkpoint_sha):
            raise ValueError('OOD predictions do not use the historical checkpoint')
        ood = ood_doc['predictions']
        if len(ood) != 301:
            raise ValueError('OOD population is not301 positive pairs')
        for split, rows in (('test', test), ('real', real), ('ood', ood)):
            if identities(rows, split != 'ood') != identities(current[split], split != 'ood'):
                raise ValueError(name+' '+split+' identity/label/endpoint/GT differs from current S0')
        if any(r['decision_valid'] is not True for r in test+real):
            raise ValueError('Unexpected invalid decisions need explicit denominator treatment')
        real_norm = helper.normalize(real, 'classification.fused', 'full_top2_mode')
        test_norm = helper.normalize(test, 'classification.fused', 'full_top2_mode')
        if ({r['pair_id'] for r in real if not r['label']} != negatives
                or not keep <= {r['pair_id'] for r in real if r['label']}):
            raise ValueError('Review identities not aligned with REAL')
        groups = {'test_full': test_norm, 'real_full': real_norm,
            'real_keep_plus_all_negative': [r for r in real_norm if r['pair_id'] in keep | negatives]}
        policies = {'max_f1': values['max_f1'], 'val_recall95': values['recall_95'],
                    'legacy_val_recall99': values['recall_99']}
        if (policies['max_f1'] != old['thresholds']['max_f1']
                or policies['legacy_val_recall99'] != old['thresholds']['recall_first']):
            raise ValueError('Legacy threshold names do not match frozen numeric values')
        measurements = {policy: {group: compact(helper.summarize(rows, threshold))
            for group, rows in groups.items()} for policy, threshold in policies.items()}
        for policy, threshold in policies.items():
            measurements[policy]['ood_positive_only'] = positive_only(ood, threshold)
        # Check the independent new extraction against already delivered max-F1
        # counts. Do not re-run any image/model pipeline to recreate a baseline.
        for group, previous_group in (('test_full', 'test3000'), ('real_full', 'real1016'),
                                     ('real_keep_plus_all_negative', 'kept803')):
            measured = measurements['max_f1'][group]
            reference = previous['models'][name][previous_group]['max_f1']
            for key in ('tp', 'fp', 'tn', 'fn', 'recall', 'precision', 'f1', 'accuracy', 'ap', 'auroc'):
                if abs(measured['classification'][key] - reference[key]) > 1e-10:
                    raise ValueError('Historical max-F1 metric mismatch: '+name+'/'+group+'/'+key)
            if measured['positive_only_layout']['20']['raw_good_count'] != reference['raw20_count']:
                raise ValueError('Historical raw layout20 differs')
        result[name] = dict(display_name=display, checkpoint_sha256=checkpoint_sha,
            training=training, threshold_policies=policies, measurements=measurements,
            role='historical performance reference, not equal-budget architecture ablation',
            checkpoint_reloaded=False, current_pair_population_exact_match=True,
            ood_endpoint_check='pair_id equality only; historical OOD rows do not expose endpoints/GT')
    sources[str(curated/'compare_kept_models.py')] = sha(curated/'compare_kept_models.py')
    return dict(schema_version='rachel-score-historical-references/1', status='complete',
        models=result, source_sha256=sources, new_inference=False, thresholds_refitted=False,
        original_artifacts_modified=False, layout_tolerances_px=[10,20],
        notes=['max_f1 and val_recall95 are operating points on each already-selected historical checkpoint.',
               'Historical recall_first means VALR99, not the new experiment selection named recall95.',
               'Historical E1 final-stage exposure excludes inherited pretraining; total is unavailable here.',
               'Historical checkpoint selection and new budgeted selection differ; compare as references only.',
               'REAL keep cohort is fixed retrospective human selection, not an untouched test population.',
               'OOD score-only diagnostics have301positives, no negative-set specificity or layout accuracy.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--output', required=True)
    parser.add_argument('--refresh-own-output', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        if (not args.refresh_own_output or not output.is_dir()
                or {p.name for p in output.iterdir()} != {'references.json'}
                or read(output/'references.json').get('schema_version') != 'rachel-score-historical-references/1'):
            raise ValueError('Use new output or explicitly refresh only this collectors own reference artifact')
    result = build(args.workspace)
    output.mkdir(parents=True, exist_ok=args.refresh_own_output)
    (output/'references.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], models=list(result['models']),
        output=str(output.resolve()/'references.json'), new_inference=False)))


if __name__ == '__main__':
    main()

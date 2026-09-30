"""Posthoc CAL-only FPR budgets from existing native Matcher predictions.

No neural forward, candidate construction, reranking or model fitting. Original
whole-population files are integrity checked; only the frozen639 development
identities enter any new metric or threshold calculation.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def require(value, message):
    if not value: raise ValueError(message)


def native_development_rows(native_rows, strata, scoring):
    require(scoring in ('q_sum', 'q_arc'), 'predeclared raw-Q readout required')
    by_id = {r['pair_id']:r for r in native_rows}
    require(len(by_id) == len(native_rows), 'duplicate frozen prediction')
    ids = [r['pair_id'] for r in strata]
    require(len(set(ids)) == len(ids) and set(ids).issubset(by_id), 'missing/duplicate development membership')
    result = []
    for item in strata:
        require(item['role'] in ('real_cal', 'real_select') and item['fold'] != 0, 'TEST cannot enter budgets')
        row = by_id[item['pair_id']]
        require(row['schema'] == 'curriculum-matcher-pair/1' and row['scorer_used'] is False
            and row['q_modified'] is False and row['gt_used_in_proposal'] is False
            and type(row['label']) is bool and row['label'] == item['label']
            and type(row['numeric_valid']) is bool, 'native prediction identity/labels differ')
        if item['label']:
            require(row['gt_known'] is True and type(row['retained_correct_coverage']) is bool,
                    'known positive Layout GT required')
        winner = row[scoring+'_winner']
        candidates = row['candidates']; retained = row['retained_count']
        require(type(retained) is int and 0 <= retained <= len(candidates), 'invalid retained budget')
        require(winner is None or (type(winner) is int and 0 <= winner < retained
                                  and candidates[winner]['index'] == winner), 'winner outside retained budget')
        correct = row[scoring+'_winner_layout20']
        require(not item['label'] or (correct is None if winner is None else type(correct) is bool),
                'missing winner correctness')
        score = 0. if winner is None else candidates[winner]['q_sum' if scoring == 'q_sum' else 'q_arc_mass_px']
        result.append(dict(pair_id=item['pair_id'], role=item['role'], label=item['label'],
            numeric_valid=row['numeric_valid'], has_candidate=winner is not None, score=score,
            layout20=bool(correct) if item['label'] else None,
            candidate_coverage=row['retained_correct_coverage'] if item['label'] else None))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('diagnostics-source', 'source-root', 'evaluation-root', 'strata', 'out'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--arm',choices=('B0','B1','B2','B3'),required=True)
    args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only analysis required')
    require(not args.out.exists(), 'new analysis output required')
    sys.path.insert(0,str(args.diagnostics_source.resolve()))
    from ridge_runner import sha, read, read_rows, save, save_rows, bind, verify_strata
    from diagnostic_metrics import calibrate_negative_budget, score_at_calibrated_budget
    require(sha(args.diagnostics_source/'diagnostic_metrics.py') ==
        '8f59fae15b0ad5833f167501b10bd00060a949e01831744b5ad6b52bbaa8bb2e', 'changed metric implementation')
    plan, _ = verify_strata(args.strata)
    sys.path.insert(0,str(args.source_root.resolve()))
    package='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'
    name='curriculum_training_v1.matcher_eval_controller' if args.arm=='B0' else 'matcher_v2_v1.evaluation_controller'
    controller=importlib.import_module(package+name)
    require(args.source_root.resolve() in Path(controller.__file__).resolve().parents, 'wrong verifier source')
    job=dict(name='sim_best_dunhuang_cv',selection='sim_best',split='dunhuang_cv')
    proof=controller.verify_job(args.evaluation_root,job) if args.arm=='B0' else controller.verify_job(args.evaluation_root,job,'matcher')
    origin=proof['provenance']
    require(origin['module']=='matcher' and origin['scorer_used'] is False
        and origin['real_used_for_selection'] is False and origin['real_split_sha256']==plan['inputs']['real_split']['sha256'],
        'SIM-selected Matcher and bound real roles required')
    require(args.arm=='B0' or origin['arm']==args.arm,'wrong experiment arm')
    frozen=args.evaluation_root/job['name']/'case_diagnostics.jsonl'
    original=read_rows(frozen)
    groups={r['pair_id']:r['seam_group'] for r in plan['rows'] if r['label'] and r['role']=='real_select'}
    results={}; outputs={}
    for scoring in ('q_sum','q_arc'):
        rows=native_development_rows(original,plan['rows'],scoring)
        cal=[r for r in rows if r['role']=='real_cal']; selected=[r for r in rows if r['role']=='real_select']
        require(len(cal)==160 and len(selected)==479 and sum(r['label'] for r in cal)==58
                and sum(r['label'] for r in selected)==175,'complete development population required')
        outputs[scoring]=rows
        results[scoring]={str(fraction):score_at_calibrated_budget(selected,calibrate_negative_budget(cal,fraction),groups)
            for fraction in (.01,.02,.05)}
    args.out.mkdir()
    save(args.out/'summary.json',dict(schema='matcher-v2-native-cal-budgets/1',arm=args.arm,
        origin=origin,proof=proof,strata=bind(args.strata),frozen_source=bind(frozen),metrics=results,
        test_used_for_analysis=False,inference_repeated=False,candidates_modified=False,
        original_sim_threshold_results_overwritten=False,model_effectiveness_not_implied=True))
    for scoring,rows in outputs.items():save_rows(args.out/(scoring+'_development.jsonl'),rows)
    complete=dict(status='complete',arm=args.arm,development_pairs=639,negative_cal=102,negative_select=304,
        source_sha256=sha(__file__),files={p.name:sha(p) for p in args.out.iterdir() if p.is_file()},
        gpu_used=False,model_inference=False,return_must_be_verified_separately=True)
    save(args.out/'complete.json',complete)
    print(json.dumps(complete),flush=True)


if __name__=='__main__':main()

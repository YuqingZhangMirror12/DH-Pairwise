"""Existing immutable evaluators, never launch training or repeat ordinary TEST."""
import importlib
from pathlib import Path
import sys

import contracts as c

KINDS = {'matcher':('native_reference','ridge','native_budgets'),
         'scorer_patch':('scorer_reference',),'scorer_stats':('scorer_reference',)}


def registry():
    return [dict(id=arm.lower()+'_'+module+'_'+kind,arm=arm,module=module,kind=kind)
            for arm in c.ARMS for module in c.MODULES for kind in KINDS[module]]


def check_job(job):c.require(job in registry(),'unregistered supplemental task')


def command(job,out,ready):
    check_job(job); arm,module,kind=(job[k] for k in ('arm','module','kind'))
    spec,root=c.paths(arm,module)
    c.require(ready['execution']==c.bound(spec),'verified execution changed before child')
    common=['--source-root',str(c.SOURCE),'--spec',str(spec),'--spec-sha',ready['execution']['sha256'],
            '--training-root',str(root/'training'),'--arm',arm,'--selection','sim_best','--out',str(out)]
    if kind=='native_reference':
        return [str(c.PYTHON),str(c.ROOT/'reference_select_source_01/evaluate_reference_select.py'),*common,
            '--preparation',str(c.PREPARATION),'--admission',str(c.ADMISSION),'--admission-sha',c.ADMISSION_SHA]
    if kind=='ridge':
        return [str(c.PYTHON),str(c.ROOT/'diagnostics_source_03/ridge_runner.py'),*common,'--strata',str(c.STRATA)]
    if kind=='native_budgets':
        return [str(c.PYTHON),str(c.ROOT/'cal_budget_source_01/score_native.py'),
            '--source-root',str(c.SOURCE),'--diagnostics-source',str(c.ROOT/'diagnostics_source_03'),
            '--evaluation-root',str(root/'evaluation'),'--strata',str(c.STRATA),'--out',str(out),'--arm',arm]
    return [str(c.PYTHON),str(c.ROOT/'reference_scorer_source_01/evaluate_reference_scorer.py'),*common,
        '--engine-root',str(c.SOURCE),'--module',module,'--preparation',str(c.PREPARATION),
        '--reference-source',str(c.ROOT/'reference_select_source_01'),
        '--admission',str(c.ADMISSION),'--admission-sha',c.ADMISSION_SHA]


def dep_module(folder,name):
    directory=c.ROOT/folder;sys.path.insert(0,str(directory))
    module=importlib.import_module(name)
    c.require(Path(module.__file__).resolve().parent==directory,'wrong companion dependency '+name)
    return module


def file_set(root,complete,expected):
    c.require(not (root/'failure.json').exists(),'child failure precedes completion')
    c.require(set(complete['files'])==set(expected) and
        all(c.sha(root/name)==value for name,value in complete['files'].items()),'complete artifact membership/hash differs')


def verify_reference(job,out,ready):
    complete=c.read(out/'evaluation_complete.json');origin=complete['provenance']
    c.check_origin(origin,ready)
    c.require(origin['split']=='sim_reference_straight_select','wrong supplemental reference population')
    admission=dep_module('reference_select_source_01','evaluate_reference_select')
    plan,_=admission.verify_admission(c.ADMISSION,c.ADMISSION_SHA)
    population=c.read(out/'population.json')
    c.require(population['pair_ids']==plan['groups']['all'] and population['groups']==plan['groups']
        and population['fixed_diagnostic_ids']==sorted(plan['fixed_diagnostic_ids']),
        'reference order/group/fixed-case identities changed')
    if job['kind']=='native_reference':
        verifier=importlib.import_module(c.PACKAGE+'.curriculum_training_v1.matcher_run')
        audit=verifier.verify_population(out);recorded=c.read(out/'independent_artifact_audit.json')
        c.require(audit['pairs']==900 and audit['diagnostic_cases']==30
            and all(recorded[k]==v for k,v in audit.items()) and recorded['rng_unchanged'] is True
            and recorded['cuda_initialized'] is False and recorded['reference_unmodified'] is True
            and recorded['classification_accuracy'] is None,'complete independent native reference required')
    else:
        verifier=dep_module('reference_scorer_source_01','run_scorer_reference')
        spec,root=c.paths(job['arm'],job['module'])
        audit=verifier.verify_completed(dict(arm=job['arm'],module=job['module'],source_root=str(c.SOURCE),
            engine_root=str(c.SOURCE),selection='sim_best'),out)
    return dict(evaluation_complete=c.bound(out/'evaluation_complete.json'),
        independent_audit=c.bound(out/'independent_artifact_audit.json'),verified=audit)


def verify_ridge(out,ready):
    complete=c.read(out/'complete.json')
    file_set(out,complete,('protocol.json','raw_predictions.jsonl','prediction_complete.json','case_diagnostics.jsonl','summary.json'))
    c.require(complete['schema']=='matcher-v2-ridge-complete/1' and complete['status']=='complete'
        and complete['pairs']==233 and complete['model_state_unchanged'] is True
        and complete['original_raw_files_verified'] is True and complete['cuda_initialized'] is False
        and complete['real_backpropagation'] is False and complete['test_inferred'] is False,
        'complete CPU-only development ridge evidence required')
    c.check_origin(c.read(out/'protocol.json')['origin'],ready)
    api=dep_module('diagnostics_source_03','ridge_runner');plan,_=api.verify_strata(c.STRATA)
    raw=c.rows(out/'raw_predictions.jsonl');labeled=c.rows(out/'case_diagnostics.jsonl')
    c.require(len(raw)==233 and sum(not row['numeric_valid'] for row in raw)==complete['numeric_invalid_pairs'],
              'ridge numerical validity count differs')
    actual=api.targets_after_freeze(plan,raw,out)
    c.require(actual==labeled,'independent raw-array ridge recount differs')
    summary=c.read(out/'summary.json');c.check_origin(summary['origin'],ready)
    c.require(api.summarize_ridge_population(actual)=={k:v for k,v in summary.items()
        if k not in ('origin','numeric_invalid_pair_ids')},'ridge group/distribution summary differs')
    return dict(complete=c.bound(out/'complete.json'),summary=c.bound(out/'summary.json'),pairs=233,
        numeric_invalid_pairs=complete['numeric_invalid_pairs'],raw_arrays_independently_recomputed=True)


def verify_budgets(out,ready):
    complete=c.read(out/'complete.json');summary=c.read(out/'summary.json')
    file_set(out,complete,('summary.json','q_sum_development.jsonl','q_arc_development.jsonl'))
    c.require(complete['status']=='complete' and complete['arm']==ready['arm']
        and complete['development_pairs']==639 and complete['negative_cal']==102 and complete['negative_select']==304
        and complete['gpu_used'] is False and complete['model_inference'] is False
        and summary['test_used_for_analysis'] is False and summary['inference_repeated'] is False
        and summary['candidates_modified'] is False,'posthoc-only complete development budgets required')
    c.check_origin(summary['origin'],ready)
    api=dep_module('diagnostics_source_03','ridge_runner');plan,_=api.verify_strata(c.STRATA)
    scorer=dep_module('cal_budget_source_01','score_native')
    metrics=dep_module('diagnostics_source_03','diagnostic_metrics')
    source=summary['frozen_source'];c.require(c.bound(source['path'])==source,'frozen source predictions changed')
    original=c.rows(source['path'])
    groups={row['pair_id']:row['seam_group'] for row in plan['rows'] if row['label'] and row['role']=='real_select'}
    for method in ('q_sum','q_arc'):
        actual=scorer.native_development_rows(original,plan['rows'],method)
        c.require(actual==c.rows(out/(method+'_development.jsonl')),'development-only readout differs')
        cal=[r for r in actual if r['role']=='real_cal'];selected=[r for r in actual if r['role']=='real_select']
        expected={str(f):metrics.score_at_calibrated_budget(selected,metrics.calibrate_negative_budget(cal,f),groups)
                  for f in (.01,.02,.05)}
        c.require(summary['metrics'][method]==expected,'CAL-only threshold/SELECT metric recount differs')
    return dict(complete=c.bound(out/'complete.json'),summary=c.bound(out/'summary.json'),development_pairs=639,
        original_predictions_independently_recounted=True,test_used=False)


def verify_job(job,out,ready):
    check_job(job);out=Path(out)
    c.require(not (out/'failure.json').exists(),'child failure precedes completion')
    if job['kind'].endswith('_reference'):return verify_reference(job,out,ready)
    if job['kind']=='ridge':return verify_ridge(out,ready)
    return verify_budgets(out,ready)

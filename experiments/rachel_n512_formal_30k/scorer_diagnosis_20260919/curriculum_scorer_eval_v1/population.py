"""Same heldout input populations as native Matcher evaluation, with a head."""
from ..curriculum_training_v1.matcher_population import (
    freeze_plan as matcher_plan, load_population, population_groups, tensor_inputs,
    targets_after_prediction, INPUTS, REAL_PLAN_SHA, CASE_PLAN_SHA)
from ..curriculum_training_v1.model_adapter import require
from ..curriculum_training_v1.runtime_io import read

SPLITS = ('sim_test', 'dunhuang_cv', 'turufan')


def freeze_plan(spec_path, case_plan, source_root):
    plan = matcher_plan(spec_path, case_plan, source_root)
    plan['schema'] = 'curriculum-scorer-population-plan/1'
    plan['splits'] = list(SPLITS)
    plan['simulation'] = {'sim_test': plan['simulation']['sim_test']}
    plan['pair_counts'].pop('sim_select')
    plan['scorer_used'] = True
    plan.pop('real_used_for_matcher_selection')
    plan['real_development_selection_permitted'] = True
    return plan


def validate_plan(plan, spec_path, source_root):
    require(plan.get('schema') == 'curriculum-scorer-population-plan/1'
            and plan == freeze_plan(spec_path, plan['case_plan']['path'], source_root),
            'frozen Scorer population/source binding changed')


def attach_targets(predictions, targets, meta):
    """Join only after prediction_complete; no model calls or selection here."""
    import numpy as np
    ids = [r['pair_id'] for r in predictions]
    require(ids and len(ids) == len(set(ids))
            and ids == [r['pair_id'] for r in targets] == [r['pair_id'] for r in meta['pairs']],
            'target/population membership or order differs')
    rows = []
    for prediction, target, item in zip(predictions, targets, meta['pairs']):
        require('label' not in prediction and 'gt_known' not in prediction
                and type(target['label']) is bool, 'unlabelled prediction and binary target required')
        pose = target['gt_pose']
        require(pose is None or (target['label'] and np.asarray(pose).shape == (2,)
                and np.isfinite(pose).all()), 'finite positive layout target required')
        def error(translation):
            return (None if pose is None or translation is None else
                    float(np.linalg.norm(np.asarray(translation, dtype=np.float64)-pose)))
        row = dict(prediction, label=target['label'], gt_known=pose is not None,
            target_translation_rc=pose, recipe=item.get('recipe'), fold=item.get('fold'),
            error_px=error(prediction['translation']))
        row['layout20'] = bool(pose is not None and row['numeric_valid']
                              and row['error_px'] is not None and row['error_px'] <= 20)
        row['proposal_errors_px'] = [error(c['proposal_translation']) for c in row['candidates']]
        row['candidate_errors_px'] = [error(c['refined_translation']) for c in row['candidates']]
        row['proposal_coverage'] = any(e is not None and e <= 20 for e in row['proposal_errors_px'])
        row['candidate_coverage'] = any(e is not None and e <= 20 for e in row['candidate_errors_px'])
        rows.append(row)
    return rows

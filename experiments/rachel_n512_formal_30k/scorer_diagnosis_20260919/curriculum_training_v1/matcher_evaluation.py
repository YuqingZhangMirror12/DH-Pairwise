"""Native Matcher evaluation primitives; never instantiate or call a Scorer.

Prediction receives only the six image/contour inputs. Ground truth joins a
completed, serialized prediction afterwards. A caller still has to verify a
training terminal, bind the heldout data, persist prediction_complete, and
run the population/controller workflow. This module does not launch that job.
"""
import copy

import numpy as np
import torch

from .checkpoint_io import tree_sha
from .exposure import digest
from .matcher_diagnostics import inspect_pair, summarize
from .model_adapter import BASE, bound_module, require

TARGET_FIELDS = ('label', 'gt_known', 'target_translation_rc', 'q_sum_winner_layout20',
    'q_arc_winner_layout20', 'q_sum_winner_error_px', 'q_arc_winner_error_px',
    'retained_correct_coverage', 'prebudget_correct_coverage', 'budget_lost_correct',
    'best_correct_by_q_arc', 'negative_max_q_sum', 'negative_max_q_arc_px')


def prediction_sha(row):
    """All evidence, poses, memberships and winner identities, not GT labels."""
    value = copy.deepcopy(row)
    for field in TARGET_FIELDS:
        value.pop(field, None)
    for candidate in value['candidates']:
        candidate.pop('pose_error_px', None); candidate.pop('layout20', None)
    return digest(value)


@torch.no_grad()
def predict_batch(matcher, geometry, source_root, inputs, pair_ids, capture_ids=()):
    """Actual single-Sinkhorn adapter plus native T16 builder, with no targets.

    A fresh builder per pair prevents its diagnostic prebudget list leaking
    from a valid pair into a subsequent invalid pair. This is not a new pose
    algorithm: both the retained and prebudget candidates come from the same
    invocation of the immutable training builder.
    """
    matcher_api = bound_module(BASE + 's7_consensus_v1.matcher', source_root)
    evidence_api = bound_module(BASE + 's7_consensus_v1.evidence', source_root)
    policy = bound_module(BASE + 's7_consensus_v1.pose_consensus', source_root)
    require(isinstance(matcher, matcher_api.S7MatcherAdapter)
            and matcher.frozen and not matcher.training
            and not any(p.requires_grad for p in matcher.parameters()), 'frozen eval Matcher required')
    require(policy.REVISION == 'native-hypothesis-complete-link-union/1-diameter16', 'fixed T16 builder required')
    require(set(inputs) == set(matcher_api.INPUTS), 'only the six Matcher inputs are allowed')
    require(pair_ids and len(set(pair_ids)) == len(pair_ids)
            and all(isinstance(value, str) and value for value in pair_ids), 'unique batch pair identities required')
    for name, value in inputs.items():
        expected_dtype = torch.bool if name.startswith('contour_valid') else torch.float32
        require(isinstance(value, torch.Tensor) and value.dtype == expected_dtype
                and len(value) == len(pair_ids), 'FP32/boolean input batch differs: ' + name)
    evidence = matcher(**inputs)
    rows = []
    for index, pair_id in enumerate(pair_ids):
        pair = evidence_api.PairEvidence.from_matcher(evidence, index, inputs['mask_a'], inputs['mask_b'])
        builder = policy.PoseConsensusBuilder(geometry)
        proposals = builder(pair)
        row = inspect_pair(pair_id, pair, proposals, label=None, all_clusters=builder.all_clusters,
                           capture_edges=pair_id in capture_ids)
        row['model_inputs_sha256'] = tree_sha({key:value[index] for key, value in inputs.items()})
        if pair_id in capture_ids:
            row['contour_points'] = dict(a=pair.points_a.cpu().tolist(), b=pair.points_b.cpu().tolist(),
                original_a=pair.original_a.cpu().tolist(), original_b=pair.original_b.cpu().tolist(),
                coordinates='compact contour row,column in unchanged model input canvas')
        rows.append(row)
    return rows


def annotate_prediction(row, *, label, gt_pose=None):
    """Posthoc target join. Does not rebuild/re-rank/refine/normalize evidence."""
    require(row.get('schema') == 'curriculum-matcher-pair/1' and row.get('label') is None
            and row.get('gt_known') is False and row.get('scorer_used') is False
            and row.get('gt_used_in_proposal') is False and row.get('q_modified') is False,
            'unlabelled native prediction required before target join')
    require(type(label) is bool, 'binary target label required')
    require(gt_pose is None or label, 'negative pair cannot have a common layout GT')
    target = None if gt_pose is None else np.asarray(gt_pose, dtype=np.float64)
    require(target is None or (target.shape == (2,) and np.isfinite(target).all()), 'finite 2D positive GT required')
    for key in TARGET_FIELDS:
        if key not in ('label', 'gt_known'):
            require(row.get(key) is None, 'prediction already contains a target-derived field: ' + key)
    require(all(c['pose_error_px'] is None and c['layout20'] is None for c in row['candidates']),
            'prediction candidates already contain GT')
    before = prediction_sha(row); result = copy.deepcopy(row)
    known = target is not None
    result.update(label=label, gt_known=known, target_translation_rc=None if target is None else target.tolist())
    candidates = result['candidates']; retained = candidates[:result['retained_count']]
    require([c['index'] for c in candidates] == list(range(len(candidates)))
            and 0 <= result['retained_count'] <= len(candidates)
            and [c['retained'] for c in candidates] == [i < len(retained) for i in range(len(candidates))],
            'candidate indices or retained prefix differ')
    for candidate in candidates:
        pose = np.asarray(candidate['translation_rc'], dtype=np.float64)
        require(pose.shape == (2,) and np.isfinite(pose).all(), 'finite predicted 2D pose required')
        error = None if not known else float(np.linalg.norm(pose - target))
        candidate.update(pose_error_px=error, layout20=None if not known else error <= 20.)
    for prefix in ('q_sum', 'q_arc'):
        winner = result[prefix + '_winner']
        require(winner is None or (type(winner) is int and 0 <= winner < len(retained)), 'winner outside retained candidates')
        result[prefix + '_winner_layout20'] = None if winner is None or not known else candidates[winner]['layout20']
        result[prefix + '_winner_error_px'] = None if winner is None else candidates[winner]['pose_error_px']
    result['retained_correct_coverage'] = None if not known else any(c['layout20'] for c in retained)
    audited = result['prebudget_count'] is not None
    require(not audited or result['prebudget_count'] == len(candidates), 'prebudget count differs')
    precoverage = None if not known or not audited else any(c['layout20'] for c in candidates)
    result['prebudget_correct_coverage'] = precoverage
    result['budget_lost_correct'] = None if precoverage is None else bool(precoverage and not result['retained_correct_coverage'])
    good = [c for c in retained if c['layout20'] is True]
    result['best_correct_by_q_arc'] = None if not good else max(good, key=lambda c:(c['q_arc_mass_px'], -c['index']))['index']
    for field, winner_name, mass_name in [('negative_max_q_sum', 'q_sum_winner', 'q_sum'),
                                        ('negative_max_q_arc_px', 'q_arc_winner', 'q_arc_mass_px')]:
        winner = result[winner_name]
        result[field] = None if label or winner is None else candidates[winner][mass_name]
    require(prediction_sha(result) == before, 'target join changed predictions')
    return result


def annotate_population(predictions, targets):
    require(predictions and len(predictions) == len(targets)
            and len({r['pair_id'] for r in predictions}) == len(predictions), 'complete unique population required')
    result = []
    for row, target in zip(predictions, targets):
        require(row['pair_id'] == target['pair_id'] and set(target) == {'pair_id', 'label', 'gt_pose'},
                'target identity/order/schema differs')
        result.append(annotate_prediction(row, label=target['label'], gt_pose=target['gt_pose']))
    return result


def compare_orders(curriculum, mixed, provenance_c, provenance_m):
    """Paired native evidence comparison; no Scorer accuracy is fabricated.

    Provenances must come from the verified terminal loader, not from user
    supplied model names. Artifact/source verification remains the caller's
    responsibility; this function validates the comparison contract itself.
    """
    for record, order in ((provenance_c, 'curriculum'), (provenance_m, 'mixed')):
        require(record['schema'] == 'curriculum-matcher-evaluation-origin/1' and record['order'] == order
                and record['module'] == 'matcher' and record['stop_reason'] == 'fixed_shared_update_budget'
                and record['scorer_used'] is False and record['real_used_for_selection'] is False,
                'verified native completed C/M origins required')
        require(record['selection_kind'] in ('sim_best', 'equal_budget_endpoint'), 'only registered Matcher choices allowed')
        require(type(record['total_completed_updates']) is int and 0 < record['updates'] <= record['total_completed_updates'],
                'completed training/update identity invalid')
        if record['selection_kind'] == 'equal_budget_endpoint':
            require(record['updates'] == record['total_completed_updates'], 'endpoint is not the budget endpoint')
    for field in ('common_plan_sha256', 'selection_kind', 'total_completed_updates', 'initial_matcher_state_sha256',
                  'architecture', 'geometry', 'proposal_revision', 'evaluation_population_sha256'):
        require(provenance_c[field] == provenance_m[field], 'C/M comparison differs: ' + field)
    require(curriculum and len(curriculum) == len(mixed)
            and len({r['pair_id'] for r in curriculum}) == len(curriculum), 'paired unique evaluation population required')
    for a, b in zip(curriculum, mixed):
        for field in ('pair_id', 'label', 'gt_known', 'target_translation_rc', 'model_inputs_sha256'):
            require(field in a and field in b and a[field] == b[field], 'C/M data/GT/input identity differs: ' + field)
        require(isinstance(a['model_inputs_sha256'], str) and len(a['model_inputs_sha256']) == 64, 'actual model input SHA required')
    known = [(a,b) for a,b in zip(curriculum,mixed) if a['label'] and a['gt_known']]
    transitions = {}
    for field in ('retained_correct_coverage', 'q_sum_winner_layout20', 'q_arc_winner_layout20'):
        transitions[field] = dict(positive_layout_gt_count=len(known),
            both_correct=sum(a[field] is True and b[field] is True for a,b in known),
            curriculum_only=sum(a[field] is True and b[field] is not True for a,b in known),
            mixed_only=sum(a[field] is not True and b[field] is True for a,b in known),
            neither_correct=sum(a[field] is not True and b[field] is not True for a,b in known))
    return dict(schema='curriculum-matcher-paired-comparison/1',
        curriculum=summarize(curriculum), mixed=summarize(mixed), paired_layout_transitions=transitions,
        selection_kind=provenance_c['selection_kind'], common_plan_sha256=provenance_c['common_plan_sha256'],
        both_trained_total_updates=provenance_c['total_completed_updates'],
        selected_updates=dict(curriculum=provenance_c['updates'], mixed=provenance_m['updates']),
        same_selected_update=provenance_c['updates'] == provenance_m['updates'],
        classification_accuracy=None, joint_f1=None, scorer_used=False,
        caution='Native Matcher candidates/evidence only; no classification head. Same budget is not equal selected epoch. Real data is developmental; Turufan has no Layout GT.')

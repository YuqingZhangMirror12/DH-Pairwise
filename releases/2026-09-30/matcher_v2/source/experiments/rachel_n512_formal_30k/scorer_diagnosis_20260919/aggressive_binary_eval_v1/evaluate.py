"""Terminal evaluation; separate SIM/REAL selection, no threshold fitting here."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch

from consensus_binary_eval_common import evaluate as common
from .contracts import PLAN_SHA, SIM_SPLIT, read, sha, save, partition_real, validate_threshold
from .loading import load_selected
from .population import load_population, attach_targets
from consensus_binary_eval_adapter.trace import MLPTrace
from consensus_binary_eval_adapter.snapshot import snapshot_prediction,write_snapshot,audit_snapshot


def prediction_record(pair_id,pair,pred):
    candidates=[]
    for i,c in enumerate(pred.clusters):
        x=c.readout.inputs
        candidates.append(dict(cluster_id=i,selected=i==pred.selected_cluster_id,
            proposal_translation=c.proposal.translation.detach().cpu().tolist(),
            refined_translation=c.translation.detach().cpu().tolist(),learned_refinement=False,
            score=float(c.readout.score),logit=float(c.readout.logit),
            union_pair_count=len(x.edge_ids),union_q_sum=float(x.q.sum()),
            observed_mass_length_px=float(x.mass_weights.sum()),
            underconstrained=bool(c.proposal.underconstrained),
            positive_evidence_px=None,conflict_evidence_px=None,local_classification_present=False))
    row=dict(pair_id=pair_id,has_candidate=pred.has_candidate,numeric_valid=pred.numeric_valid,
        score=float(pred.score),accepted=pred.accepted,selected_cluster_id=pred.selected_cluster_id,
        translation=None if pred.translation_a_to_b_rc is None else pred.translation_a_to_b_rc.detach().cpu().tolist(),
        pose_uncertainty=pred.pose_uncertainty,candidates=candidates,seed_count=len(pred.proposals.seeds),
        hypotheses_count=len(pred.proposals.hypotheses),candidate_count=len(candidates),
        merge_trace_count=len(pred.proposals.merge_trace),valid_points_a=pair.q.shape[0],valid_points_b=pair.q.shape[1],
        absolute_q_mass=common.finite_or_none(pair.q.sum()),unmatched_a_mean=common.finite_or_none(pair.unmatched_a.mean()),
        unmatched_b_mean=common.finite_or_none(pair.unmatched_b.mean()))
    json.dumps(row,allow_nan=False)
    return row


def selection_for_split(provenance, split):
    value = provenance['thresholds'][split]
    validate_threshold(value)
    return value


def population_summary(rows, threshold, split):
    result = common.population_summary(rows, threshold)
    if split == 'turufan':
        for key in list(result):
            if key.startswith('joint_') or key in ('layout20', 'layout20_count',
                    'candidate_coverage', 'candidate_coverage_count', 'covered_but_winner_wrong',
                    'winner_correct_but_rejected', 'positive_no_correct_candidate', 'wrong_pose_accepted'):
                result[key] = None
    return result


def make_summary(rows, split, plan, provenance):
    threshold = selection_for_split(provenance, split)
    groups = {'all': rows} if split == SIM_SPLIT else partition_real(rows, plan, split)
    return dict(status='complete', **provenance,
        layout_gt_available=split != 'turufan', threshold_refitting=False,
        main_group='all' if split == SIM_SPLIT else 'real_test',
        real_test_is_historically_unseen=False,
        groups={name: dict(primary=population_summary(part, threshold, split),
                          fixed03=population_summary(part, .3, split)) for name, part in groups.items()})


def run(args, helper):
    # Terminal/source/origin audits must finish BEFORE opening heldout inputs.
    model, contract, provenance = load_selected(args.root,args.reference,args.selection,args.real_plan,helper)
    if sha(args.real_plan) != PLAN_SHA:
        raise ValueError('registered real split required')
    role_plan = read(args.real_plan)
    plan = read(args.case_plan)
    common.validate_case_plan(plan)
    if set(plan['user_confirmed_gt_exclusions']) != set(role_plan['datasets']['dunhuang_cv']['excluded_gt_pair_ids']):
        raise ValueError('fixed diagnostic GT exclusions differ')
    wanted = {r['pair_id'] for r in plan['cases'] if r['split'] == args.split}
    threshold = selection_for_split(provenance, args.split)
    provenance.update(threshold=threshold, split=args.split,
        threshold_origin=('synthetic CAL at selected epoch' if args.split == SIM_SPLIT or args.selection == 'sim'
                          else 'source-isolated REAL-CAL at REAL-SELECT epoch'),
        real_plan_sha256=PLAN_SHA, postprocess_preparation_sha256=sha(args.preparation),
        case_plan_sha256=sha(args.case_plan), model_input_fields=list(common.INPUTS),
        gt_used_for_prediction=False, microbatch=8, historical_real_development_exposure=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    start = time.time(); device = torch.device(args.device)
    model.to(device); before = common.state_digest(model)
    try:
        meta, batches, source, dataset = load_population(args.split, contract, 8)
        ids = [p['pair_id'] for p in meta['pairs']]
        if len(set(ids)) != len(ids) or not wanted <= set(ids):
            raise ValueError('incomplete or duplicated inference population')
        if args.split != SIM_SPLIT:
            partition_real([dict(pair_id=i) for i in ids], role_plan, args.split)
            if source.get('manifest_sha256') != role_plan['datasets'][args.split]['manifest_sha256']:
                raise ValueError('real population source changed')
            if source.get('inputs_sha256') != provenance['real_development_binding']['sources'][args.split]['inputs_sha256']:
                raise ValueError('real image/contour cache differs from selection binding')
        provenance.update(source=source, total_pairs=len(ids))
        save(out/'protocol.json', dict(status='running', **provenance))
        rows, diagnostics = [], []
        with (out/'pair_predictions.jsonl').open('x') as stream, torch.no_grad():
            for items, batch in batches:
                inputs = common.tensor_inputs(batch, device)
                evidence = model.matcher(*(inputs[k] for k in common.INPUTS))
                for i, item in enumerate(items):
                    pair = common.PairEvidence.from_matcher(evidence, i, inputs['mask_a'], inputs['mask_b'])
                    capture = item['pair_id'] in wanted
                    with (MLPTrace(model.head) if capture else nullcontext()) as trace:
                        pred = model.score_pair(pair, threshold=threshold, capture_diagnostics=capture)
                    row = prediction_record(item['pair_id'], pair, pred)
                    stream.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
                    if capture:
                        relative = 'evidence/'+hashlib.sha256(item['pair_id'].encode()).hexdigest()[:24]
                        metadata, arrays = snapshot_prediction(item['pair_id'], pair, pred,
                            threshold=threshold, provenance=provenance, trace=trace)
                        written = write_snapshot(out/relative, metadata, arrays)
                        audit = audit_snapshot(out/relative/'evidence.json')
                        if audit['status'] != 'passed':
                            raise ValueError('fixed diagnostic numeric audit did not pass')
                        save(out/relative/'audit.json', audit)
                        diagnostics.append(dict(pair_id=item['pair_id'], evidence=relative+'/evidence.json',
                            sidecar_sha256=written['sidecar']['sha256'],
                            numerical_audit=relative+'/audit.json', numerical_audit_status=audit['status']))
                    del pred, pair
                stream.flush()
                save(out/'status.json', dict(status='inference', processed=len(rows), total=len(ids),
                                             elapsed_seconds=time.time()-start, pid=os.getpid()))
            os.fsync(stream.fileno())
        if [r['pair_id'] for r in rows] != ids or {r['pair_id'] for r in diagnostics} != wanted:
            raise ValueError('missing frozen predictions or fixed diagnostic snapshots')
        if common.state_digest(model) != before:
            raise ValueError('frozen evaluation changed model tensors')
        save(out/'prediction_complete.json', dict(status='all_predictions_frozen', pairs=len(rows),
            sha256=sha(out/'pair_predictions.jsonl'), model_state_unchanged=True, **provenance))
        # Label/GT joining and grouping occur after predictions become immutable.
        gt = ({r['pair_id']: r for r in read(role_plan['gt_path'])['positive_pairs']}
              if args.split == 'dunhuang_cv' else None)
        labeled = attach_targets(rows, meta, args.split, dataset, gt)
        with (out/'case_diagnostics.jsonl').open('x') as stream:
            for row in labeled:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        summary = make_summary(labeled, args.split, role_plan, provenance)
        summary['diagnostic_cases'] = diagnostics
        save(out/'summary.json', summary)
        save(out/'diagnostic_index.json', dict(cases=diagnostics, selected_by_new_results=False))
        save(out/'protocol.json', dict(status='complete', **provenance))
        save(out/'status.json', dict(status='complete', pairs=len(rows), elapsed_seconds=time.time()-start,
                                     pid=os.getpid()))
    except BaseException as error:
        save(out/'failure.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
                                     elapsed_seconds=time.time()-start, pid=os.getpid()))
        raise

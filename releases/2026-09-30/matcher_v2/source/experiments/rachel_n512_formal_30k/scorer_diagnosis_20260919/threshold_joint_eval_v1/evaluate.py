"""Terminal evaluation; separate SIM/REAL selection, no threshold fitting here."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch

from consensus_joint_eval_common import evaluate as common
from consensus_joint_eval_common.frozen import registered_protocol, TrainingConfig
from .contracts import PLAN_SHA, read, sha, save, partition_real, validate_threshold
from .loading import load_joint, load_frozen_real


def selection_for_split(provenance, split):
    value = provenance['thresholds'][split]
    validate_threshold(value)
    return value


def snapshot_provenance(provenance):
    """Training experiment and forward evidence protocol are different labels.

    Joint fine-tuning changes the weights, not the threshold architecture. The
    shared numeric exporter must still enforce its exact-union-Q protocol.
    """
    protocol = registered_protocol(TrainingConfig().record())
    if protocol['variant'] != 'threshold' or protocol['evidence_mode'] != 'exact_union_q':
        raise ValueError('joint/control diagnostics require the fixed16 threshold architecture')
    return dict(provenance, experiment_variant=provenance.get('variant'),
                variant=protocol['variant'], evidence_mode=protocol['evidence_mode'])


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
    groups = {'all': rows} if split == 'sim_test_v14' else partition_real(rows, plan, split)
    return dict(status='complete', **provenance,
        layout_gt_available=split != 'turufan', threshold_refitting=False,
        main_group='all' if split == 'sim_test_v14' else 'real_test',
        real_test_is_historically_unseen=False,
        groups={name: dict(primary=population_summary(part, threshold, split),
                          fixed03=population_summary(part, .3, split)) for name, part in groups.items()})


def run(args, helper):
    # Terminal/source/origin audits must finish BEFORE opening heldout inputs.
    if args.model_kind == 'joint':
        model, contract, provenance = load_joint(args.root, args.reference, args.selection,
                                                 args.real_plan, helper)
    else:
        model, contract, provenance = load_frozen_real(args.root, args.reference,
            args.real_selection, args.real_plan, helper)
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
        threshold_origin=('synthetic CAL at selected epoch' if args.split == 'sim_test_v14' or args.selection == 'sim'
                          else 'source-isolated REAL-CAL at REAL-SELECT epoch'),
        real_plan_sha256=PLAN_SHA, postprocess_preparation_sha256=sha(args.preparation),
        case_plan_sha256=sha(args.case_plan), model_input_fields=list(common.INPUTS),
        gt_used_for_prediction=False, microbatch=8, historical_real_development_exposure=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    start = time.time(); device = torch.device(args.device)
    model.to(device); before = common.state_digest(model)
    try:
        meta, batches, source, dataset = common.load_population(args.split, contract, 8)
        ids = [p['pair_id'] for p in meta['pairs']]
        if len(set(ids)) != len(ids) or not wanted <= set(ids):
            raise ValueError('incomplete or duplicated inference population')
        if args.split != 'sim_test_v14':
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
                    with (common.AttentionTrace(model.head) if capture else nullcontext()) as attention:
                        pred = model.score_pair(pair, threshold=threshold, capture_diagnostics=capture)
                    row = common.prediction_record(item['pair_id'], pair, pred)
                    stream.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
                    if capture:
                        relative = 'evidence/'+hashlib.sha256(item['pair_id'].encode()).hexdigest()[:24]
                        metadata, arrays = common.snapshot_prediction(item['pair_id'], pair, pred,
                            threshold=threshold, provenance=snapshot_provenance(provenance), attention_trace=attention)
                        written = common.write_snapshot(out/relative, metadata, arrays)
                        audit = common.audit_snapshot(out/relative/'evidence.json')
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
        labeled = common.attach_targets(rows, meta, args.split, dataset, gt)
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

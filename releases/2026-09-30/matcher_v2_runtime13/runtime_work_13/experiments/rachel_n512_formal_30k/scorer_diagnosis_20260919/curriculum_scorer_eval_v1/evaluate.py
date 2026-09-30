"""Immutable predictions and actual union/Q/MLP snapshots, then target joining."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch

from consensus_binary_eval_common import evaluate as common
from consensus_binary_eval_adapter.evaluate import prediction_record, population_summary
from consensus_binary_eval_adapter.snapshot import snapshot_prediction, write_snapshot, audit_snapshot
from consensus_binary_eval_adapter.trace import MLPTrace
from ..curriculum_training_v1.checkpoint_io import file_sha, tree_sha, write_json
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.model_adapter import require
from ..curriculum_training_v1.runtime_io import read
from .population import (SPLITS, INPUTS, load_population, population_groups,
                         tensor_inputs, targets_after_prediction, attach_targets)


def make_summary(rows, groups, provenance, diagnostics):
    by_id = {r['pair_id']:r for r in rows}; split = provenance['split']
    require(len(by_id) == len(rows) and rows and groups, 'unique rows and declared groups required')
    require(all(ids and len(ids) == len(set(ids)) and set(ids) <= set(by_id) for ids in groups.values()),
            'evaluation group membership differs')
    main = 'all' if split.startswith('sim_') else 'real_test'
    require(main in groups, 'primary evaluation population missing')
    threshold = provenance['threshold']
    return dict(status='complete', **provenance, layout_gt_available=split != 'turufan',
        threshold_refitting=False, main_group=main, real_test_is_historically_unseen=False,
        groups={name:dict(primary=population_summary([by_id[i] for i in ids], threshold, split),
            fixed03=population_summary([by_id[i] for i in ids], .3, split)) for name,ids in groups.items()},
        diagnostic_cases=diagnostics)


def run_population(model, source_root, plan, split, provenance, out, device, *,
                   registered_splits=None, population_loader=None, group_builder=None,
                   target_loader=None, extra_diagnostic_ids=()):
    # Defaults are resolved at call time so existing terminal callers and tests
    # retain their original population policy. New adapters are explicit.
    registered_splits = SPLITS if registered_splits is None else registered_splits
    population_loader = load_population if population_loader is None else population_loader
    group_builder = population_groups if group_builder is None else group_builder
    target_loader = targets_after_prediction if target_loader is None else target_loader
    require(split in registered_splits and not model.training and all(not p.requires_grad for p in model.parameters()),
            'only registered frozen Scorer inference permitted')
    require(provenance['model_state_sha256'] == tree_sha(model.state_dict()), 'loaded model differs from verified export')
    threshold = provenance['thresholds'][split]
    require(type(threshold) in (int, float) and .2 <= threshold <= .8, 'frozen CAL threshold required')
    cases = read(plan['case_plan']['path']); common.validate_case_plan(cases)
    roles = read(plan['real_split']['path'])
    wanted = {r['pair_id'] for r in cases['cases'] if r['split'] == split} | set(extra_diagnostic_ids)
    out = Path(out); out.mkdir(parents=True, exist_ok=False); start = time.time()
    before = tree_sha(model.state_dict()); rng = torch.get_rng_state().clone()
    try:
        meta, batches, source, dataset = population_loader(split, plan, source_root)
        ids = [r['pair_id'] for r in meta['pairs']]
        require(len(ids) == plan['pair_counts'][split] and len(ids) == len(set(ids)) and wanted <= set(ids),
                'complete registered population and fixed cases required')
        grouped = group_builder(meta['pairs'], split, roles)
        population = dict(source=source, pair_ids=ids,
            groups={name:[r['pair_id'] for r in group] for name,group in grouped.items()},
            fixed_diagnostic_ids=sorted(wanted),
            case_metadata=[{k:r[k] for k in ('pair_id','recipe','fold') if k in r} for r in meta['pairs']])
        provenance = dict(provenance, split=split, source=source, total_pairs=len(ids), threshold=threshold,
            threshold_origin=provenance.get('threshold_origins', {}).get(split, ('source-isolated REAL-CAL at REAL-SELECT update' if
                split != 'sim_test' and provenance['selection_kind'] == 'real_best' else 'SIM-CAL at this update')),
            real_plan_sha256=plan['real_split']['sha256'], case_plan_sha256=plan['case_plan']['sha256'],
            microbatch=8, gt_used_for_prediction=False, model_input_fields=list(INPUTS),
            simulation_revision=plan['simulation_revision'], evaluation_population_sha256=digest(population))
        write_json(out/'population.json', population)
        write_json(out/'protocol.json', dict(status='running', **provenance))
        rows, diagnostics = [], []
        with (out/'pair_predictions.jsonl').open('x') as stream, torch.no_grad():
            for items, batch in batches:
                inputs = tensor_inputs(batch, device)
                evidence = model.matcher(*(inputs[k] for k in INPUTS))
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
                        require(audit['status'] == 'passed', 'fixed actual union/Q/MLP numeric audit failed')
                        write_json(out/relative/'audit.json', audit)
                        diagnostics.append(dict(pair_id=item['pair_id'], evidence=relative+'/evidence.json',
                            evidence_sha256=file_sha(out/relative/'evidence.json'), sidecar_sha256=written['sidecar']['sha256'],
                            numerical_audit=relative+'/audit.json', numerical_audit_status='passed'))
                    del pred, pair
                stream.flush()
                write_json(out/'status.json', dict(status='inference', processed=len(rows), total=len(ids),
                    elapsed_seconds=time.time()-start, pid=os.getpid()), replace=True)
            os.fsync(stream.fileno())
        require([r['pair_id'] for r in rows] == ids and {r['pair_id'] for r in diagnostics} == wanted,
                'missing/duplicate frozen prediction or fixed diagnostic')
        require(before == tree_sha(model.state_dict()) and torch.equal(rng, torch.get_rng_state()),
                'evaluation changed model or CPU torch RNG')
        write_json(out/'prediction_complete.json', dict(status='all_predictions_frozen', pairs=len(rows),
            sha256=file_sha(out/'pair_predictions.jsonl'), model_state_unchanged=True, torch_cpu_rng_unchanged=True,
            **provenance))
        # This callback is intentionally invoked only after durable prediction completion.
        targets = target_loader(meta, split, plan, dataset)
        labeled = attach_targets(rows, targets, meta)
        with (out/'case_diagnostics.jsonl').open('x') as stream:
            for row in labeled:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        summary = make_summary(labeled, population['groups'], provenance, diagnostics)
        write_json(out/'summary.json', summary)
        write_json(out/'diagnostic_index.json', dict(cases=diagnostics, selected_by_new_results=False))
        write_json(out/'protocol.json', dict(status='complete', **provenance), replace=True)
        write_json(out/'status.json', dict(status='complete', pairs=len(rows), elapsed_seconds=time.time()-start,
            pid=os.getpid()), replace=True)
        files = ('population.json','protocol.json','pair_predictions.jsonl','prediction_complete.json',
                 'case_diagnostics.jsonl','summary.json','diagnostic_index.json','status.json')
        write_json(out/'evaluation_complete.json', dict(schema='curriculum-scorer-evaluation-complete/1',
            status='evaluation_complete', pairs=len(rows), provenance=provenance,
            files={name:file_sha(out/name) for name in files},
            predictions_sha256=file_sha(out/'pair_predictions.jsonl'),
            labeled_sha256=file_sha(out/'case_diagnostics.jsonl'), summary_sha256=file_sha(out/'summary.json'),
            prediction_complete_sha256=file_sha(out/'prediction_complete.json'),
            diagnostic_index_sha256=file_sha(out/'diagnostic_index.json'), model_state_unchanged=True,
            split=split, selection_kind=provenance['selection_kind'], actual_updates=provenance['selected_updates'],
            process_success_not_yet_certified=True))
        return summary
    except BaseException as error:
        write_json(out/'failure.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.time()-start, pid=os.getpid(), automatic_retry=False))
        raise

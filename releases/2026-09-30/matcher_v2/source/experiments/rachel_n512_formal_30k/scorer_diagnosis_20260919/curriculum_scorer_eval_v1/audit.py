"""Reopen and recompute actual saved predictions/evidence; no model calls."""
from pathlib import Path
import json

from consensus_binary_eval_adapter.snapshot import audit_snapshot
from ..curriculum_training_v1.checkpoint_io import file_sha
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.model_adapter import require
from ..curriculum_training_v1.runtime_io import read
from .evaluate import make_summary
from .population import attach_targets


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def within(root, relative):
    path = (root/relative).resolve()
    require(root in path.parents, 'diagnostic file outside this output root')
    return path


def verify_population(out):
    out = Path(out).resolve()
    require(not (out/'failure.json').exists(), 'failure precedes stale evaluation completion')
    complete = read(out/'evaluation_complete.json'); origin = complete['provenance']
    require(complete.get('schema') == 'curriculum-scorer-evaluation-complete/1'
            and complete['status'] == 'evaluation_complete' and complete['model_state_unchanged'] is True
            and complete['split'] == origin['split'] and complete['selection_kind'] == origin['selection_kind']
            and complete['actual_updates'] == origin['selected_updates'] > 0,
            'not this completed curriculum Scorer evaluation')
    expected = {'population.json','protocol.json','pair_predictions.jsonl','prediction_complete.json',
                'case_diagnostics.jsonl','summary.json','diagnostic_index.json','status.json'}
    require(set(complete['files']) == expected
            and all(file_sha(out/name) == sha for name,sha in complete['files'].items()),
            'completed evaluation artifact changed')
    population = read(out/'population.json'); frozen = read(out/'prediction_complete.json')
    require(origin['evaluation_population_sha256'] == digest(population)
            and read(out/'protocol.json') == dict(status='complete', **origin)
            and frozen == dict(status='all_predictions_frozen', pairs=complete['pairs'],
                sha256=file_sha(out/'pair_predictions.jsonl'),model_state_unchanged=True,
                torch_cpu_rng_unchanged=True,**origin), 'frozen prediction/protocol/population binding differs')
    for key, filename in (('predictions_sha256','pair_predictions.jsonl'),('labeled_sha256','case_diagnostics.jsonl'),
                          ('summary_sha256','summary.json'),('prediction_complete_sha256','prediction_complete.json'),
                          ('diagnostic_index_sha256','diagnostic_index.json')):
        require(complete[key] == file_sha(out/filename), 'completion file alias differs: '+key)
    raw = read_rows(out/'pair_predictions.jsonl'); labeled = read_rows(out/'case_diagnostics.jsonl')
    require(len(raw) == len(labeled) == complete['pairs'] == origin['total_pairs']
            and [r['pair_id'] for r in raw] == [r['pair_id'] for r in labeled] == population['pair_ids'],
            'missing/reordered/duplicated evaluation rows')
    targets = [dict(pair_id=r['pair_id'],label=r['label'],gt_pose=r['target_translation_rc']) for r in labeled]
    require(attach_targets(raw, targets, dict(pairs=population['case_metadata'])) == labeled,
            'posthoc targets changed predictions or derived layout metrics')
    index = read(out/'diagnostic_index.json'); cases = index['cases']
    require(index['selected_by_new_results'] is False and len({r['pair_id'] for r in cases}) == len(cases)
            and {r['pair_id'] for r in cases} == set(population['fixed_diagnostic_ids']),
            'fixed diagnostic membership changed')
    raw_by_id = {r['pair_id']:r for r in raw}
    for item in cases:
        evidence = within(out,item['evidence']); metadata = read(evidence)
        require(file_sha(evidence) == item['evidence_sha256']
                and metadata['sidecar']['sha256'] == item['sidecar_sha256']
                and metadata['pair_id'] == item['pair_id'] and metadata['provenance'] == origin,
                'fixed diagnostic identity differs')
        audit = audit_snapshot(evidence)
        require(audit['status'] == item['numerical_audit_status'] == 'passed'
                and read(within(out,item['numerical_audit'])) == audit, 'fixed union/Q/MLP audit changed')
        prediction = raw_by_id[item['pair_id']]
        for key in ('score','accepted','translation','selected_cluster_id','has_candidate','numeric_valid'):
            require(metadata[key] == prediction[key], 'snapshot differs from actual prediction: '+key)
        require(len(metadata['clusters']) == len(prediction['candidates']), 'snapshot candidate count differs')
        for captured, candidate in zip(metadata['clusters'], prediction['candidates']):
            require(captured['cluster_id'] == candidate['cluster_id']
                    and captured['score'] == candidate['score'] and captured['logit'] == candidate['logit'],
                    'snapshot candidate scores differ')
    require(make_summary(labeled,population['groups'],origin,cases) == read(out/'summary.json'),
            'summary differs from actual per-pair predictions')
    status = read(out/'status.json')
    require(status['status'] == 'complete' and status['pairs'] == complete['pairs'], 'terminal status missing')
    return dict(status='passed',pairs=len(raw),diagnostic_cases=len(cases),
        evaluation_complete_sha256=file_sha(out/'evaluation_complete.json'),
        prediction_sha256=file_sha(out/'pair_predictions.jsonl'),model_state_unchanged=True,
        selection_kind=origin['selection_kind'],selected_updates=origin['selected_updates'],
        population_sha256=origin['evaluation_population_sha256'],
        actual_rows_and_numeric_evidence_recomputed=True,process_return_must_be_checked_separately=True)

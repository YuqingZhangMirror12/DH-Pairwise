"""Terminal-only v14 TEST/real inference and exact C10 evidence export.

The six image/contour tensors alone reach Matcher. Labels/GT are attached after
all predictions have been durably saved. No threshold or epoch is selected here.
Real folds and user GT exclusions remain metadata, not model inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
from contextlib import nullcontext

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from .frozen import ARMS, TrainingConfig, load_selected, read, sha
from .audit import audit_snapshot
from .snapshot import snapshot_prediction
from .attention_trace import AttentionTrace
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.data import Dataset, collate
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.diagnostics import write_snapshot
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import PairEvidence
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.matcher import INPUTS
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.metrics import summarize
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.preflight_matcher import state_digest
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.train import save_json


SPLITS = ('sim_test_v14', 'dunhuang_cv', 'turufan')


def finite_or_none(value):
    number = float(value)
    return number if np.isfinite(number) else None


def validate_case_plan(plan):
    if (plan.get('schema') != 's7-consensus-fixed-diagnostics/1'
            or plan.get('selected_by_new_results') is not False
            or len(plan.get('cases', [])) != 11
            or len({r['pair_id'] for r in plan['cases']}) != 11
            or len(set(plan.get('user_confirmed_gt_exclusions', []))) != 3
            or any(r['split'] not in ('dunhuang_cv', 'turufan') for r in plan['cases'])):
        raise ValueError('registered fixed case plan required')


def tensor_inputs(batch, device):
    """Explicit allowlist: Pair labels, GT and recipe cannot enter prediction."""
    result = {}
    for key in INPUTS:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            value = torch.from_numpy(np.array(value, copy=True))
        result[key] = value.to(device=device,
            dtype=torch.bool if key.startswith('contour_valid') else torch.float32)
    return result


def prediction_record(pair_id, pair, prediction):
    candidates = []
    for i, cluster in enumerate(prediction.clusters):
        candidates.append(dict(cluster_id=i, selected=i == prediction.selected_cluster_id,
            proposal_translation=cluster.proposal.translation.detach().cpu().tolist(),
            refined_translation=cluster.translation.detach().cpu().tolist(),
            score=float(cluster.readout.score), logit=float(cluster.readout.logit),
            positive_evidence_px=float(cluster.readout.positive_evidence_px),
            conflict_evidence_px=float(cluster.readout.conflict_evidence_px),
            observed_mass_length_px=float(cluster.readout.observed_mass_length_px),
            underconstrained=cluster.refinement.underconstrained, overlap=cluster.overlap))
    row = dict(pair_id=pair_id, has_candidate=prediction.has_candidate,
        numeric_valid=prediction.numeric_valid, score=float(prediction.score),
        accepted=prediction.accepted, selected_cluster_id=prediction.selected_cluster_id,
        translation=None if prediction.translation_a_to_b_rc is None
            else prediction.translation_a_to_b_rc.detach().cpu().tolist(),
        pose_uncertainty=prediction.pose_uncertainty, candidates=candidates,
        seed_count=len(prediction.proposals.seeds),
        hypotheses_count=len(prediction.proposals.hypotheses),
        candidate_count=len(candidates), merge_trace_count=len(prediction.proposals.merge_trace),
        valid_points_a=len(pair.local_a), valid_points_b=len(pair.local_b),
        absolute_q_mass=finite_or_none(pair.q.sum()),
        unmatched_a_mean=finite_or_none(pair.unmatched_a.mean()),
        unmatched_b_mean=finite_or_none(pair.unmatched_b.mean()))
    json.dumps(row, allow_nan=False)
    return row


def load_population(split, contract, batch_size):
    if split == 'sim_test_v14':
        spec = contract['test']['mixed']
        if spec['pair_count'] != 3000:
            raise ValueError('v14 fixed TEST requires3000 pairs')
        dataset = Dataset(spec['path'], spec['sha256'])
        if len(dataset) != 3000:
            raise ValueError('fixed TEST size changed')

        def batches():
            for start in range(0, len(dataset), batch_size):
                indices = range(start, min(start + batch_size, len(dataset)))
                yield [dataset.entries[j] for j in indices], collate([dataset[j] for j in indices])

        return dict(pairs=dataset.entries), batches(), dict(manifest=spec['path'],
            manifest_sha256=spec['sha256'], source_count=spec['source_count'],
            base_pair_count=spec['base_pair_count'], preprocessing='immutable v14 TEST archives'), dataset
    if split not in SPLITS:
        raise ValueError('unregistered external population')
    # Data-only reuse of the previously frozen800px/N512 real cache adapter.
    # Its v3 model/decoder/score functions are NEVER called by this module.
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.seam_context_v3.evaluate_external import load_inputs
    return load_inputs(split, batch_size)


def attach_targets(predictions, meta, split, dataset=None, ground_truth=None):
    if len(predictions) != len(meta['pairs']):
        raise ValueError('missing predictions')
    rows = []
    for i, (prediction, item) in enumerate(zip(predictions, meta['pairs'])):
        if prediction['pair_id'] != item['pair_id']:
            raise ValueError('reordered predictions')
        target = None
        if split == 'sim_test_v14':
            sample, _, _ = dataset[i]
            label = bool(sample.label)
            if label and sample.translation_valid:
                target = np.asarray(sample.translation_a_to_b_rc)
        else:
            label = bool(item['label'])
            if split == 'dunhuang_cv' and label:
                gt = ground_truth[item['pair_id']]
                if (gt['fragment_a_token'], gt['fragment_b_token']) != (
                        item['fragment_a_id'], item['fragment_b_id']):
                    raise ValueError('Dunhuang GT endpoint order differs')
                target = np.asarray(gt['translation_gt_a_to_b_rc'])
        known = target is not None

        def error(translation):
            return float(np.linalg.norm(np.asarray(translation) - target)) if known and translation is not None else None

        row = dict(prediction, label=label, gt_known=known,
            target_translation_rc=target.tolist() if known else None,
            recipe=item.get('recipe'), fold=item.get('fold'),
            label_source=item.get('label_source'),
            negative_kind=item.get('negative_kind', item.get('source_row', {}).get('negative_kind')),
            error_px=error(prediction['translation']))
        row['layout20'] = bool(known and row['numeric_valid'] and row['error_px'] is not None and row['error_px'] <= 20)
        row['proposal_errors_px'] = [error(c['proposal_translation']) for c in row['candidates']]
        row['candidate_errors_px'] = [error(c['refined_translation']) for c in row['candidates']]
        row['proposal_coverage'] = any(e is not None and e <= 20 for e in row['proposal_errors_px'])
        row['candidate_coverage'] = any(e is not None and e <= 20 for e in row['candidate_errors_px'])
        rows.append(row)
    return rows


def population_summary(rows, threshold):
    value = summarize(rows, threshold)
    scores = np.array([r['score'] if r['has_candidate'] and r['numeric_valid'] else 0. for r in rows])
    labels = np.array([r['label'] for r in rows], bool)
    positive, negative = scores[labels], scores[~labels]
    value['auroc'] = (float(((positive[:, None] > negative).astype(float)
        + .5 * (positive[:, None] == negative)).mean()) if len(positive) and len(negative) else None)
    if not value['known_positive_layouts']:
        for key in ('layout20_count', 'candidate_coverage_count', 'covered_but_winner_wrong',
                    'winner_correct_but_rejected', 'positive_no_correct_candidate',
                    'wrong_pose_accepted', 'joint_tp', 'joint_fp', 'joint_fn'):
            value[key] = None
    return value


def run(args):
    # This guard occurs before opening any TEST or real model input/target.
    model, contract, provenance = load_selected(args.root, args.arm, args.reference,
        completed_arm_only=getattr(args, 'completed_arm_only', False))
    plan = read(args.case_plan)
    validate_case_plan(plan)
    wanted = {r['pair_id'] for r in plan['cases'] if r['split'] == args.split}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    start = time.time(); config = TrainingConfig(); device = torch.device(args.device)
    random.seed(config.data_seed); np.random.seed(config.data_seed); torch.manual_seed(config.data_seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = model.to(device); before = state_digest(model)
    try:
        meta, batches, source, dataset = load_population(args.split, contract, 8)
        ids = [p['pair_id'] for p in meta['pairs']]
        if len(set(ids)) != len(ids) or not wanted <= set(ids):
            raise ValueError('missing fixed diagnostic case or duplicate population ID')
        provenance.update(split=args.split, source=source, case_plan_sha256=sha(args.case_plan),
            microbatch=8, model_input_fields=list(INPUTS), gt_used_for_prediction=False,
            postprocess_code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
            total_pairs=len(ids), historical_real_development_exposure=True)
        save_json(out/'protocol.json', dict(status='running', **provenance))
        rows, diagnostic_index = [], []
        with (out/'pair_predictions.jsonl').open('x') as stream, torch.no_grad():
            for items, batch in batches:
                inputs = tensor_inputs(batch, device)
                evidence = model.matcher(*(inputs[k] for k in INPUTS))
                for i, item in enumerate(items):
                    pair = PairEvidence.from_matcher(evidence, i, inputs['mask_a'], inputs['mask_b'])
                    capture = item['pair_id'] in wanted
                    with (AttentionTrace(model.head) if capture else nullcontext()) as attention:
                        pred = model.score_pair(pair, threshold=provenance['threshold'], capture_diagnostics=capture)
                    row = prediction_record(item['pair_id'], pair, pred)
                    stream.write(json.dumps(row, allow_nan=False) + '\n'); rows.append(row)
                    if capture:
                        relative = 'evidence/' + hashlib.sha256(item['pair_id'].encode()).hexdigest()[:24]
                        metadata, arrays = snapshot_prediction(item['pair_id'], pair, pred,
                            threshold=provenance['threshold'], provenance=provenance, attention_trace=attention)
                        written = write_snapshot(out/relative, metadata, arrays)
                        audit = audit_snapshot(out/relative/'evidence.json')
                        save_json(out/relative/'audit.json', audit)
                        diagnostic_index.append(dict(pair_id=item['pair_id'], evidence=relative+'/evidence.json',
                            sidecar_sha256=written['sidecar']['sha256'],
                            numerical_audit=relative+'/audit.json', numerical_audit_status=audit['status']))
                    del pred, pair
                stream.flush()
                save_json(out/'status.json', dict(status='inference', processed=len(rows),
                    total=len(ids), elapsed_seconds=time.time()-start, pid=os.getpid()))
            os.fsync(stream.fileno())
        if [r['pair_id'] for r in rows] != ids or {r['pair_id'] for r in diagnostic_index} != wanted:
            raise ValueError('incomplete inference/evidence population')
        if state_digest(model) != before:
            raise ValueError('frozen inference changed model tensors')
        save_json(out/'prediction_complete.json', dict(status='all_predictions_frozen', pairs=len(rows),
            sha256=sha(out/'pair_predictions.jsonl'), model_state_unchanged=True, **provenance))
        # Only now read/join GT for metrics. No second model call or tuning.
        gt = None
        if args.split == 'dunhuang_cv':
            from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.seam_context_v3.evaluate_external import GT
            gt = {r['pair_id']: r for r in read(GT)['positive_pairs']}
        labeled = attach_targets(rows, meta, args.split, dataset, gt)
        with (out/'case_diagnostics.jsonl').open('x') as stream:
            for row in labeled:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        groups = {'all': labeled}
        if args.split == 'dunhuang_cv':
            excluded = set(plan['user_confirmed_gt_exclusions'])
            if not excluded <= set(ids):
                raise ValueError('registered GT exclusions absent from Dunhuang population')
            groups['gt_corrected_800'] = [r for r in labeled if r['pair_id'] not in excluded]
        if args.split == 'sim_test_v14':
            # Negative-only strata report FPR, not an invented positive recall.
            archive = read(Path(source['manifest']).parent/'archive_manifest.json')['entries']
            kind = {e['pair_id']: e['negative_kind'] for e in archive if not e['label']}
            negative_strata = {k: [r for r in labeled if kind.get(r['pair_id']) == k] for k in set(kind.values())}
        else:
            negative_strata = {}
        summary = dict(status='complete', **provenance, layout_gt_available=args.split != 'turufan',
            groups={name: dict(primary=population_summary(part, provenance['threshold']),
                fixed03=population_summary(part, .3)) for name, part in groups.items()},
            negative_subtypes={name: dict(count=len(part), accepted=sum(r['accepted'] for r in part),
                false_positive_rate=sum(r['accepted'] for r in part)/len(part)) for name, part in negative_strata.items()},
            threshold_refitting=False, diagnostic_cases=diagnostic_index)
        save_json(out/'summary.json', summary)
        save_json(out/'diagnostic_index.json', dict(cases=diagnostic_index, selected_by_new_results=False))
        save_json(out/'protocol.json', dict(status='complete', **provenance))
        save_json(out/'status.json', dict(status='complete', pairs=len(rows),
            elapsed_seconds=time.time()-start, pid=os.getpid()))
    except BaseException as exc:
        save_json(out/'failure.json', dict(status='failed', error=repr(exc),
            traceback=traceback.format_exc(), pid=os.getpid(), elapsed_seconds=time.time()-start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'reference', 'out'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--completed-arm-only', action='store_true',
        help='Parallel threshold/simple queue only; the selected arm must be fully terminal.')
    parser.add_argument('--case-plan', default=str(Path(__file__).with_name('case_plan.json')))
    run(parser.parse_args())

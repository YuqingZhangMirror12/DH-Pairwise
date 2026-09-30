"""Complete CPU-only D17/Q diagnostics on a preselected finished Matcher.

Uses the existing strict terminal loaders. Predicts all233 development positive
pairs with six inputs; GT is parsed only after a durable prediction_complete.
Does not redo the existing Layout/classification evaluation or touch GPU jobs.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from diagnostic_metrics import compact_evidence, diagnose_pair, require, summarize_ridge_population

PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919'
INPUTS = ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b')
STRATA_SHA = '5f0d7a7be2f39857a92935a262c078f540890d36794dbea5ee91cebc17ebc357'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def save_rows(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def save_arrays(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


@torch.no_grad()
def capture(matcher, tensors):
    """The exact six-field inference; a second readout never calls Sinkhorn."""
    require(set(tensors) == set(INPUTS), 'only the six Matcher inputs are allowed')
    require(matcher.frozen and not matcher.training and not any(p.requires_grad for p in matcher.parameters()),
            'completed frozen evaluation Matcher required')
    for key, tensor in tensors.items():
        require(tensor.device.type == 'cpu' and tensor.shape[0] == 1,
                'one pair on CPU only; no GPU allocation')
        require(tensor.dtype == (torch.bool if key.startswith('contour_valid') else torch.float32),
                'FP32/bool inputs required')
    ev = matcher(**tensors)
    base = matcher.base
    ap = F.normalize(base.primal(ev.context_a), dim=2, eps=1e-6)
    bp = F.normalize(base.primal(ev.context_b), dim=2, eps=1e-6)
    ad = F.normalize(base.dual(ev.context_a), dim=2, eps=1e-6)
    bd = F.normalize(base.dual(ev.context_b), dim=2, eps=1e-6)
    cosine = .5 * (ap @ bd.transpose(1, 2) + ad @ bp.transpose(1, 2))
    array = lambda value: value[0].detach().cpu().numpy()
    result = compact_evidence(array(ev.points_rc_a), array(ev.points_rc_b), array(ev.valid_a), array(ev.valid_b),
        array(ev.affinity), array(cosine), array(ev.assignment), array(ev.unmatched_a), array(ev.unmatched_b))
    return result, bool(ev.numeric_valid[0])


def terminal_model(args):
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    spec = read(args.spec)
    if args.arm == 'B0':
        execution = importlib.import_module(PACKAGE + '.curriculum_training_v1.execution')
        inputs = execution.load_inputs(spec)
        terminal = importlib.import_module(PACKAGE + '.curriculum_training_v1.matcher_terminal')
        saved, origin = terminal.verified_export(args.training_root, args.spec, inputs['plan'], 'curriculum', args.selection)
        importer = importlib.import_module(PACKAGE + '.curriculum_training_v1.verify_validation_preparation')
        importer.bind_baseline(inputs['baseline'])
        model, _, origin = terminal.load_matcher(saved, origin, inputs['baseline'])
    else:
        runtime = importlib.import_module(PACKAGE + '.matcher_v2_v1.runtime_inputs')
        inputs = runtime.load_inputs(spec)
        require(spec['arm'] == args.arm and spec['module'] == 'matcher', 'wrong Matcher arm/module')
        terminal = importlib.import_module(PACKAGE + '.matcher_v2_v1.terminal')
        saved, origin = terminal.verified_export(args.training_root, args.spec, inputs['plan'], args.selection)
        model, _, origin = terminal.load_model(saved, origin, inputs['source'])
    require(source in Path(terminal.__file__).resolve().parents, 'terminal verifier imported from wrong source')
    require(saved['module'] == 'matcher' and origin['scorer_used'] is False
            and origin['real_used_for_selection'] is False, 'SIM-preselected native Matcher only')
    require(model.base.config.canvas_size == 800 and model.base.config.contour_cap == 512,
            'unchanged full800/N512 model required')
    model.to('cpu'); model.eval(); model.requires_grad_(False)
    return model, origin


def verify_strata(path):
    require(sha(path) == STRATA_SHA, 'fixed development strata changed')
    plan = read(path)
    require(plan['schema'] == 'matcher-v2-development-strata/1' and plan['status'] == 'locked'
            and plan['model_scores_used'] is False and plan['test_measured'] is False,
            'mask-only development strata required')
    for bound in plan['inputs'].values():
        require(sha(bound['path']) == bound['sha256'], 'strata source changed: ' + bound['path'])
    rows = [r for r in plan['rows'] if r['label']]
    require(len(plan['rows']) == 639 and len(rows) == len({r['pair_id'] for r in rows}) == 233
            and all(r['role'] in ('real_cal', 'real_select') and r['fold'] != 0 for r in rows),
            'complete development-positive membership required')
    return plan, rows


def targets_after_freeze(plan, raw_rows, root):
    proof = read(root/'prediction_complete.json')
    require(proof['predictions'] == bind(root/'raw_predictions.jsonl') and proof['targets_joined'] is False
            and proof['pairs'] == len(raw_rows) == 233 and proof['model_state_unchanged'] is True,
            'durable complete predictions required before GT join')
    require(sha(plan['inputs']['layout_gt']['path']) == plan['inputs']['layout_gt']['sha256'], 'GT changed')
    source = read(plan['inputs']['layout_gt']['path'])['positive_pairs']
    gt = {r['pair_id']: r for r in source}
    require(len(gt) == len(source), 'duplicate GT identity')
    strata = {r['pair_id']: r for r in plan['rows'] if r['label']}
    require([r['pair_id'] for r in raw_rows] == list(strata), 'predicted membership/order differs')
    result = []
    for row in raw_rows:
        frozen = root/'raw'/row['raw_file']
        require(frozen.resolve().parent == (root/'raw').resolve() and sha(frozen) == row['raw_sha256'], 'raw tensor file changed')
        with np.load(frozen, allow_pickle=False) as arrays:
            evidence = {key: arrays[key] for key in arrays.files}
        target, item = gt[row['pair_id']], strata[row['pair_id']]
        require((target['fragment_a_token'], target['fragment_b_token']) ==
                (item['fragment_a_id'], item['fragment_b_id']), 'GT endpoint order differs')
        result.append(dict(pair_id=row['pair_id'], role=item['role'], seam_group=item['seam_group'],
            numeric_valid=row['numeric_valid'], model_inputs_sha256=row['model_inputs_sha256'],
            diagnostic=diagnose_pair(evidence, target['translation_gt_a_to_b_rc'])))
    return result


def run(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only invocation required')
    require(not args.out.exists(), 'new diagnostic output directory required')
    require(sha(args.spec) == args.spec_sha, 'bound execution manifest changed')
    require(args.arm in ('B0', 'B1', 'B2', 'B3'), 'unregistered experiment arm')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model, origin = terminal_model(args)  # Completed training verified BEFORE heldout loading.
    checkpoint_api = importlib.import_module(PACKAGE + '.curriculum_training_v1.checkpoint_io')
    before = checkpoint_api.tree_sha(model.state_dict())
    require(before == origin['matcher_state_sha256'], 'actual Matcher state differs from verified export')
    plan, items = verify_strata(args.strata)
    meta = read(plan['inputs']['manifest']['path'])
    with np.load(plan['inputs']['prepared_inputs']['path'], allow_pickle=False) as arrays:
        packed, points, valid = (arrays[key] for key in ('packed_masks', 'points', 'valid'))
    lookup = {name: i for i, name in enumerate(meta['fragment_ids'])}
    require(packed.shape == (len(lookup), 800, 100) and points.shape == (len(lookup), 512, 2)
            and valid.shape == (len(lookup), 512), 'unchanged model input cache required')
    args.out.mkdir(); (args.out/'raw').mkdir()
    started = time.time()
    code = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    save(args.out/'protocol.json', dict(schema='matcher-v2-ridge-protocol/1', arm=args.arm,
        origin=origin, spec=bind(args.spec), strata=bind(args.strata), code_sha256=code,
        inputs=list(INPUTS), gt_used_for_forward=False, device='cpu', threads=1, microbatch=1,
        model_state_before=before, test_inferred=False, scorer_used=False, pid=os.getpid(), started_unix=started))
    rows = []
    for number, item in enumerate(items):
        tensors = {}
        for side in 'ab':
            i = lookup[item['fragment_' + side + '_id']]
            tensors['mask_' + side] = torch.from_numpy(np.unpackbits(packed[i:i+1], axis=-1).astype(np.float32)[:, None])
            tensors['points_rc_' + side] = torch.from_numpy(points[i:i+1].astype(np.float32))
            tensors['contour_valid_' + side] = torch.from_numpy(valid[i:i+1].astype(bool))
        captured, numeric_valid = capture(model, tensors)
        filename = '%04d.npz' % number
        save_arrays(args.out/'raw'/filename, captured)
        rows.append(dict(pair_id=item['pair_id'], numeric_valid=numeric_valid,
            raw_file=filename, raw_sha256=sha(args.out/'raw'/filename),
            model_inputs_sha256=checkpoint_api.tree_sha({k: v[0] for k, v in tensors.items()})))
        if number % 32 == 31:
            print(json.dumps(dict(processed=number+1, total=len(items), elapsed_seconds=time.time()-started)), flush=True)
    require(checkpoint_api.tree_sha(model.state_dict()) == before, 'frozen forward changed model state')
    require(not torch.cuda.is_initialized(), 'diagnostic unexpectedly initialized CUDA')
    save_rows(args.out/'raw_predictions.jsonl', rows)
    save(args.out/'prediction_complete.json', dict(pairs=len(rows), targets_joined=False,
        model_state_unchanged=True, predictions=bind(args.out/'raw_predictions.jsonl'), code_sha256=code))
    measured = targets_after_freeze(plan, rows, args.out)
    save_rows(args.out/'case_diagnostics.jsonl', measured)
    summary = summarize_ridge_population(measured)
    summary['numeric_invalid_pair_ids'] = [r['pair_id'] for r in rows if not r['numeric_valid']]
    summary['origin'] = origin
    save(args.out/'summary.json', summary)
    # Reopen and recompute from the immutable raw arrays, not the in-memory summaries.
    reopened = targets_after_freeze(plan, read_rows(args.out/'raw_predictions.jsonl'), args.out)
    require(reopened == read_rows(args.out/'case_diagnostics.jsonl'), 'reopened evidence metrics differ')
    require(summarize_ridge_population(reopened) == {k: v for k, v in read(args.out/'summary.json').items()
            if k not in ('origin', 'numeric_invalid_pair_ids')}, 'recounted summary differs')
    require(code == {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}, 'diagnostic code changed mid-run')
    require(checkpoint_api.tree_sha(model.state_dict()) == before, 'diagnostic changed Matcher state')
    complete = dict(schema='matcher-v2-ridge-complete/1', status='complete', arm=args.arm,
        pairs=len(rows), numeric_invalid_pairs=sum(not r['numeric_valid'] for r in rows),
        files={name: sha(args.out/name) for name in ('protocol.json', 'raw_predictions.jsonl',
            'prediction_complete.json', 'case_diagnostics.jsonl', 'summary.json')},
        original_raw_files_verified=True, model_state_unchanged=True, cuda_initialized=False,
        real_backpropagation=False, test_inferred=False, completed_unix=time.time(),
        actual_process_return_must_be_verified_separately=True)
    save(args.out/'complete.json', complete)
    print(json.dumps(complete, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('source-root', 'spec', 'training-root', 'strata', 'out'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--spec-sha', required=True)
    parser.add_argument('--arm', choices=('B0', 'B1', 'B2', 'B3'), required=True)
    parser.add_argument('--selection', choices=('sim_best', 'equal_budget_endpoint'), default='sim_best')
    args = parser.parse_args()
    for field in ('source_root', 'spec', 'training_root', 'strata', 'out'):
        setattr(args, field, getattr(args, field).resolve())
    existed = args.out.exists()
    try:
        run(args)
    except BaseException as error:
        if not existed:
            args.out.mkdir(exist_ok=True)
            save(args.out/'failure.json', dict(status='failed', error=repr(error),
                traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()

"""Outcome-selected cache/online input exchange, not training or a repair.

The external launcher must own the GPU lease before run(). Model inputs contain
no labels/GT. Preserve the original hard-VAL batch8 composition and exchange only
selected clean rows; all other batch members remain unchanged online inputs.
"""
import argparse
from dataclasses import fields, replace
import hashlib
import importlib
import json
from pathlib import Path
import time

SCHEMA = 'scorer-cache-online-input-exchange/1'
ARMS = ('all_tokens', 'matched_tokens', 'matched_edges')
CELLS = ('cache_features_cache_block', 'online_features_cache_block',
         'cache_features_online_block', 'online_features_online_block')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def input_fingerprint(batch, row, names):
    import numpy as np
    digest, shapes = hashlib.sha256(), {}
    for name in names:
        value = np.asarray(getattr(batch, name)[row], dtype=np.bool_ if name.startswith('contour_valid') else np.float32)
        digest.update(name.encode())
        digest.update(value.tobytes())
        shapes[name] = dict(shape=list(value.shape), dtype=str(value.dtype))
    return dict(sha256=digest.hexdigest(), arrays=shapes)


def exchange_rows(online, cached, indices):
    """Copy only declared rows; tensor and CandidateSelection originals untouched."""
    if hasattr(online, 'detach'):
        result = online.clone()
        result[indices] = cached[indices]
        return result
    if isinstance(online, tuple):
        result = list(online)
        for i in indices:
            result[i] = cached[i]
        return tuple(result)
    return replace(online, **{f.name: exchange_rows(getattr(online, f.name), getattr(cached, f.name), indices)
                              for f in fields(online)})


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def evidence_difference(online_features, cached_features, online_block, cached_block, row):
    import torch
    def arr(value):
        return value[row].detach().cpu()
    features = {}
    for side, online, cached, valid in zip('ab', online_features, cached_features, cached_block['valid']):
        o, c, mask = arr(online), arr(cached), arr(valid)
        delta = (o-c)[mask]
        features[side] = dict(valid_count=int(mask.sum()), relative_l2=float(delta.norm() / c[mask].norm().clamp_min(1e-12)),
            mean_absolute=float(delta.abs().mean()) if delta.numel() else None,
            maximum_absolute=float(delta.abs().max()) if delta.numel() else None)
    old, new = cached_block['selection'], online_block['selection']
    def edges(selection, inlier=False):
        valid = arr(selection.candidate_valid)
        if inlier:
            valid = valid & arr(selection.candidate_inliers) & arr(selection.layout_valid)
        return {tuple(pair) for pair in arr(selection.candidate_indices)[valid].tolist()}
    ca, oa, ci, oi = edges(old), edges(new), edges(old, True), edges(new, True)
    endpoints = {}
    for side in 'ab':
        c = set(arr(getattr(old, 'mask_'+side)).nonzero().flatten().tolist())
        o = set(arr(getattr(new, 'mask_'+side)).nonzero().flatten().tolist())
        endpoints[side] = dict(cache_count=len(c), online_count=len(o), jaccard=jaccard(c, o))
    cv, ov = bool(arr(old.layout_valid)), bool(arr(new.layout_valid))
    ct, ot = arr(old.translation_a_to_b_rc), arr(new.translation_a_to_b_rc)
    # Q weights are compared by edge identity, not slot index after a reorder.
    def weights(block):
        selection = block['selection']
        valid = arr(selection.candidate_valid)
        return {tuple(edge): float(weight) for edge, weight in zip(arr(selection.candidate_indices)[valid].tolist(),
                    arr(block['kwargs']['candidate_weights'])[valid].tolist())}
    cw, ow = weights(cached_block), weights(online_block)
    common = cw.keys() & ow.keys()
    qdelta = [abs(cw[e]-ow[e]) for e in common]
    return dict(features=features, candidates=dict(cache_count=len(ca), online_count=len(oa), jaccard=jaccard(ca, oa),
        identical_order=torch.equal(arr(old.candidate_indices), arr(new.candidate_indices)),
        shared_weight_count=len(qdelta), mean_absolute_q_delta=sum(qdelta)/len(qdelta) if qdelta else None,
        maximum_absolute_q_delta=max(qdelta) if qdelta else None),
        final_inlier_edges=dict(cache_count=len(ci), online_count=len(oi), jaccard=jaccard(ci, oi)),
        unique_endpoints=endpoints, layout=dict(cache_valid=cv, online_valid=ov,
        cache_translation_rc=ct.tolist() if cv else None, online_translation_rc=ot.tolist() if ov else None,
        translation_l2_difference_px=float((ot-ct).norm()) if cv and ov else None))


def four_cells(heads, online_features, cached_features, online_block, cached_block, rows):
    import torch
    result = {i: {} for i in rows}
    with torch.inference_mode():
        for cell in CELLS:
            use_cache_f = cell.startswith('cache_features')
            use_cache_b = cell.endswith('cache_block')
            features = tuple(exchange_rows(o, c, rows) for o, c in zip(online_features, cached_features)) if use_cache_f else online_features
            if use_cache_b:
                block = dict(valid=tuple(exchange_rows(o, c, rows) for o, c in zip(online_block['valid'], cached_block['valid'])),
                    selection=exchange_rows(online_block['selection'], cached_block['selection'], rows),
                    kwargs={k: exchange_rows(v, cached_block['kwargs'][k], rows) for k, v in online_block['kwargs'].items()},
                    training_valid=exchange_rows(online_block['training_valid'], cached_block['training_valid'], rows),
                    decision_valid=exchange_rows(online_block['decision_valid'], cached_block['decision_valid'], rows))
            else:
                block = online_block
            for arm, head in heads.items():
                output = head(*features, *block['valid'], block['selection'], **block['kwargs'])
                if not torch.isfinite(output.logit).all():
                    raise ValueError('nonfinite exchange logit')
                deployed = torch.where(block['training_valid'], output.logit, torch.zeros_like(output.logit))
                probability = deployed.sigmoid()
                for i in rows:
                    result[i].setdefault(arm, {})[cell] = dict(raw_logit=float(output.logit[i]), logit=float(deployed[i]),
                        probability=float(probability[i]), used_fallback=bool(output.used_fallback[i]),
                        training_valid=bool(block['training_valid'][i]), decision_valid=bool(block['decision_valid'][i]))
    return result


def run(args):
    started = time.monotonic()
    if args.device != 'cuda:0':
        raise ValueError('external assigned-GPU launcher must provide logical cuda:0')
    selection_path, manifest_path = Path(args.selection).resolve(strict=True), Path(args.manifest).resolve(strict=True)
    selection = json.loads(selection_path.read_text())
    selected_ids = [x['pair_id'] for x in selection['cases']] if 'cases' in selection else selection['pair_ids']
    if selection.get('schema') != 'execution-input-probe-selection/1' or not 1 <= len(selected_ids) <= 16 or len(set(selected_ids)) != len(selected_ids):
        raise ValueError('explicit unique outcome-selected cases1..16 required')
    reference_root = Path(args.reference_run).resolve(strict=True)
    previous = json.loads((reference_root/'protocol.json').read_text())
    if (previous.get('schema') != 'hard-simval-six-frozen-scorers/1' or previous.get('status') != 'complete'
            or previous.get('sample_count') != 6000 or previous.get('batch_size') != 8
            or previous.get('manifest_sha256') != sha(manifest_path)
            or previous.get('cases_sha256') != sha(reference_root/'cases.jsonl')):
        raise ValueError('requires exact completed hard6000 batch8 reference')
    reference = {r['pair_id']: r for r in map(json.loads, (reference_root/'cases.jsonl').read_text().splitlines())}
    manifest = json.loads(manifest_path.read_text())
    entries = manifest['entries']
    if len(reference) != 6000 or set(reference) != {r['pair_id'] for r in entries}:
        raise ValueError('full reference/manifest IDs differ')
    selected = [reference[pair_id] for pair_id in selected_ids]
    for row in selected:
        if row['recipe'] != 'clean' or entries[row['ordinal']]['pair_id'] != row['pair_id']:
            raise ValueError('requires clean rows at original exact ordinals')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('new output directory only')
    for protected in (Path(args.training_root).resolve(), Path(args.val_cache).resolve(), reference_root,
                      Path(manifest['artifact_root']).resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('output must be separate from source artifacts')
    import torch
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence import evaluate_hard_validation as hard
    from staging.pairwise_v0_2.training import rachel_n512_runner as runner
    from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs
    try:
        evaluation = importlib.import_module('matched_only.evaluate')
    except ModuleNotFoundError as error:
        if error.name != 'matched_only':
            raise
        evaluation = importlib.import_module('experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.evaluate')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    cache = evaluation.data.FormalCache(args.val_cache, 'val')
    cache_ids = {r['pair_id']: r['ordinal'] for r in cache.records}
    heads, receipts, old_rows, base = {}, {}, {}, None
    for arm in ARMS:
        training = Path(args.training_root)/arm
        adapter, receipt = evaluation.load_frozen_model(training, 'fixed_epoch', budget=16)
        prior = previous['heads'][arm+'_c16']
        if receipt['checkpoint_sha256'] != prior['checkpoint_sha256'] or receipt['source_matcher_sha256'] != prior['source_matcher_sha256']:
            raise ValueError('head/Matcher differs from completed online reference')
        if base is None:
            base = adapter.base_model.eval().requires_grad_(False).to(device)
        heads[arm] = adapter.score_head.eval().requires_grad_(False).to(device)
        receipts[arm] = receipt
        original_rows = json.loads((training/'validation_head_016_rows.json').read_text())
        old_rows[arm] = {r['pair_id']: r for r in original_rows}
        if len(original_rows) != 3000 or len(old_rows[arm]) != 3000 or set(old_rows[arm]) != set(cache_ids):
            raise ValueError('original validation rows/cache identities differ')
        del adapter
    protocol = dict(schema=SCHEMA, status='running', source_sha256=sha(__file__), selection=selection,
        selection_sha256=sha(selection_path), reference_protocol_sha256=sha(reference_root/'protocol.json'),
        manifest_sha256=sha(manifest_path), cache_binding=cache.binding, heads=receipts,
        outcome_selected=True, population_frequency_inference=False, sample_count=len(selected_ids),
        batch_size=8, source_batch_starts=sorted({r['ordinal']//8*8 for r in selected}),
        cells=list(CELLS), training_performed=False, thresholds_fitted=False,
        block_definition='selection, raw Q weights, points, contour-valid flags, training/decision validity',
        intervention='selected clean rows only; other original batch members remain online',
        caveat='diagnostic input exchanges are not deployable repairs or additive causal contributions',
        runtime=dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(device), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision()))
    output.mkdir(parents=True, exist_ok=False)
    save(output/'protocol.json', protocol)
    rows = []
    try:
        for start in protocol['source_batch_starts']:
            chunk = entries[start:start+8]
            loaded = [hard.load_entry(manifest['artifact_root'], entry) for entry in chunk]
            batch = collate_rachel_pairs([x[0] for x in loaded], contour_cap=512)
            local_selected, fingerprints = [], {}
            for i, entry in enumerate(chunk):
                if entry['pair_id'] not in selected_ids:
                    continue
                cached_record = cache.records[cache_ids[entry['source_pair_id']]]
                fingerprint = input_fingerprint(batch, i, evaluation.cache.INPUTS)
                fingerprint['cached_sha256'] = cached_record['input_sha256']
                fingerprint['equal'] = fingerprint['sha256'] == fingerprint['cached_sha256']
                fingerprints[i] = fingerprint
                if not fingerprint['equal']:
                    rows.append(dict(pair_id=entry['pair_id'], source_pair_id=entry['source_pair_id'], ordinal=start+i,
                        status='input_mismatch', fingerprint=fingerprint,
                        interpretation='not the same six input arrays; no feature/candidate attribution attempted'))
                else:
                    local_selected.append(i)
            if not local_selected:
                continue
            inputs, _ = runner._full_batch(batch, device)
            with torch.inference_mode():
                original = base(*inputs)
                chosen = evaluation.inference.select_predicted_inliers(original.assignment, *inputs[2:])
                safe = chosen.candidate_indices.clamp_min(0)
                bi = torch.arange(len(chunk), device=device)[:, None]
                weights = original.assignment[bi, safe[..., 0], safe[..., 1]]
                weights = torch.where(chosen.candidate_valid, weights, torch.zeros_like(weights))
            online_f = (original.token_features_a, original.token_features_b)
            online_b = dict(valid=inputs[4:], selection=chosen, training_valid=original.training_valid,
                decision_valid=original.decision_valid, kwargs=dict(candidate_weights=weights, points_a_rc=inputs[2], points_b_rc=inputs[3]))
            cached = cache.batch([cache_ids[e['source_pair_id']] for e in chunk], device)
            cached_f = cached.model_args[:2]
            cached_b = dict(valid=cached.model_args[2:4], selection=cached.model_args[4], kwargs=cached.model_kwargs,
                training_valid=cached.training_valid, decision_valid=cached.decision_valid)
            values = four_cells(heads, online_f, cached_f, online_b, cached_b, local_selected)
            for i in local_selected:
                entry = chunk[i]
                alignment = {}
                for arm in ARMS:
                    old = old_rows[arm][entry['source_pair_id']]
                    prior = reference[entry['pair_id']]['scores'][arm+'_c16']
                    c, o = values[i][arm][CELLS[0]], values[i][arm][CELLS[3]]
                    alignment[arm] = dict(cache_vs_original_val=dict(reference_logit=old['logit'], reference_probability=old['score'],
                        delta_logit=c['logit']-old['logit'], delta_probability=c['probability']-old['score']),
                        online_vs_previous_hard=dict(reference_logit=prior['logit'], reference_probability=prior['probability'],
                        delta_logit=o['logit']-prior['logit'], delta_probability=o['probability']-prior['probability']))
                rows.append(dict(pair_id=entry['pair_id'], source_pair_id=entry['source_pair_id'], label=entry['label'],
                    ordinal=start+i, original_batch_start=start, original_batch_pair_ids=[e['pair_id'] for e in chunk],
                    status='complete', fingerprint=fingerprints[i], scores=values[i], alignment=alignment,
                    evidence_difference=evidence_difference(online_f, cached_f, online_b, cached_b, i)))
        rows.sort(key=lambda row: selected_ids.index(row['pair_id']))
        save(output/'cases.json', rows)
        protocol.update(status='complete', completed_count=sum(r['status']=='complete' for r in rows),
            input_mismatch_count=sum(r['status']=='input_mismatch' for r in rows), cases_sha256=sha(output/'cases.json'),
            elapsed_seconds=time.monotonic()-started)
        save(output/'protocol.json', protocol)
        return dict(status='complete', output=str(output), completed_count=protocol['completed_count'],
            input_mismatch_count=protocol['input_mismatch_count'], elapsed_seconds=protocol['elapsed_seconds'])
    except BaseException as error:
        protocol.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic()-started)
        save(output/'protocol.json', protocol)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'manifest', 'val-cache', 'training-root', 'reference-run', 'output'):
        result.add_argument('--'+name, required=True, type=Path)
    result.add_argument('--device', default='cuda:0')
    return result


if __name__ == '__main__':
    raise SystemExit('Use the external assigned-GPU lease launcher, then parser()/run(args).')

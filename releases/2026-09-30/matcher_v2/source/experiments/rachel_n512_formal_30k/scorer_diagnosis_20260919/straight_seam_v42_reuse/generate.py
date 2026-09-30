"""Run frozen reference geometry, independent seeds, synthetic-only donors.

This is a visual/statistical reproduction, NOT a training-admission receipt.
Reference retries redraw difficulty; that behavior is preserved and disclosed.
The reference's correspondence filtering still requires an independent audit.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

ALLOWED = {'gen2voronoi_1', 'gen2voronoi_2', 'gen3voronoi', 'gen4voronoi',
           'gen4voronoi_1_3', 'gen5voronoi_1_1_3'}
CONTEXT = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load_reference(directory):
    modules = {}
    for name in ('gen_straight_seam', 'gen_straight_seam_v3', 'gen_straight_seam_v4'):
        path = Path(directory) / (name + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    reference = modules['gen_straight_seam_v4']
    assert reference.GEN_REV == 'v4.2'
    assert reference.cut is modules['gen_straight_seam_v3'].cut
    assert reference.finalize.__module__ == 'gen_straight_seam_v4'
    return reference, {name: sha(module.__file__) for name, module in modules.items()}


def source_pool(admission, real_audit, split):
    assert admission['status'] == 'passed' and not admission['errors']
    assert not any(admission['collision_counts'].values())
    root = Path(admission['source_root']).resolve()
    blocked = real_audit['synthetic_inventory'][split]['potential_turufan_parent_aliases']
    assert not real_audit['synthetic_inventory'][split]['exact_prepared_mask_matches']
    rows = {}
    excluded = []
    for row in admission['inventory'][split]:
        rel = Path(row['path']).resolve().relative_to(root)
        parts = rel.parts
        if (len(parts) != 5 or parts[:2] != ('model', 'masks_800') or parts[2] not in ALLOWED
                or row['fragment_token'] != '/'.join(('rachel', parts[2], parts[3], rel.stem))):
            raise ValueError('not an explicitly admitted synthetic Voronoi source')
        if row['source_family'] in blocked:
            excluded.append(row)
        else:
            rows[str(rel)] = row
    return root, rows, excluded


def choose_review_jobs(reference, out, seed):
    result = []
    for kind in 'MJR':
        if kind == 'J':
            quotas = {'torn_rachel': 6, 'margin_fragment': 1, 'torn_strip': 1}
            indices = []
            for i in range(200):
                base = reference.base_of(kind, seed, 'train', 1, i)
                if quotas[base] > 0:
                    indices.append(i)
                    quotas[base] -= 1
                if sum(quotas.values()) == 0:
                    break
            if sum(quotas.values()):
                raise ValueError('fixed-ID review pool lacks a J subtype')
        else:
            indices = list(range(8))
        result.extend((kind, 1, i, 'train', str(out), seed) for i in indices)
        result.extend((kind, 0, i, 'train', str(out), seed) for i in range(2))
    return result


def initialize(reference_dir, rows):
    reference, _ = load_reference(reference_dir)
    reference.SRCS = sorted((path, row['source_family']) for path, row in rows.items())
    original_load = reference.load_rachel
    original_cut = reference.cut
    original_misfit = reference.misfit
    original_piece = reference.piece_pair
    original_finalize = reference.finalize

    def guarded_load(path, scale):
        row = rows.get(path)
        if row is None or sha(row['path']) != row['file_sha256']:
            raise ValueError('source outside synthetic allowlist or changed SHA')
        CONTEXT['donors'][path] = row
        return original_load(path, scale)

    def tracked_cut(*args, **kwargs):
        a, b, geom = original_cut(*args, **kwargs)
        CONTEXT['cut'] = (a.copy(), b.copy(), geom)
        return a, b, geom

    def tracked_misfit(a, b, geom, rng, kind):
        pre_a, pre_b = a.copy(), b.copy()
        aa, bb, meta = original_misfit(a, b, geom, rng, kind)
        CONTEXT['misfit'] = (pre_a, pre_b, aa.copy(), bb.copy())
        return aa, bb, meta

    def tracked_piece(*args, **kwargs):
        a, b, meta = original_piece(*args, **kwargs)
        if a is None:
            CONTEXT['rejections'][str(meta)] += 1
        return a, b, meta

    def tracked_finalize(a, b, meta, label=True):
        result, error = original_finalize(a, b, meta, label)
        if result is None:
            CONTEXT['rejections'][str(error)] += 1
        elif label:
            ca, cb, geom = CONTEXT['cut']
            pre_a, pre_b, post_a, post_b = CONTEXT['misfit']
            cen, u, n, off, lo, profile = geom
            # Retain geometry before/after damage for later independent target
            # auditing. The reference's output pixels/targets are not altered.
            CONTEXT['proof'] = dict(
                cut_a=np.packbits(ca, axis=1), cut_b=np.packbits(cb, axis=1), shape=np.array(ca.shape),
                post_misfit_a=np.packbits(post_a, axis=1), post_misfit_b=np.packbits(post_b, axis=1),
                final_parent_a=np.packbits(a, axis=1), final_parent_b=np.packbits(b, axis=1),
                cut_centre=cen, cut_axis=u, cut_normal=n, cut_offset=np.array(off),
                profile_origin=np.array(lo), profile=profile,
                shift_a=np.asarray(meta['shift_a']), shift_b=np.asarray(meta['shift_b']))
        return result, error

    reference.load_rachel = guarded_load
    reference.cut = tracked_cut
    reference.misfit = tracked_misfit
    reference.piece_pair = tracked_piece
    reference.finalize = tracked_finalize
    CONTEXT['reference'] = reference


def generate_one(job):
    CONTEXT.update(donors={}, rejections=Counter(), proof=None)
    result = CONTEXT['reference'].one(job)
    result['rejections'] = dict(CONTEXT['rejections'])
    result['attempted_donor_references'] = list(CONTEXT['donors'].values())
    result['training_admitted'] = False
    if not result.get('failed') and result['label']:
        proof_path = Path(job[4]) / 'proof' / (result['id'] + '.npz')
        with proof_path.open('xb') as stream:
            np.savez_compressed(stream, **CONTEXT['proof'])
        result.update(proof_path=str(proof_path), proof_sha256=sha(proof_path))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir', 'source-admission', 'real-audit', 'output-new'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--mode', choices=['review30', 'select900'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=2)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicitly disable CUDA')
    if a.seed in (26093004, 26093015, 26093016):
        raise ValueError('Use an independent seed, not a reference release seed')
    reference, hashes = load_reference(a.reference_dir)
    admission = json.loads(a.source_admission.read_text())
    real_audit = json.loads(a.real_audit.read_text())
    assert real_audit['source_admission_sha256'] == sha(a.source_admission)
    split = 'train' if a.mode == 'review30' else 'select'
    root, rows, excluded = source_pool(admission, real_audit, split)
    assert str(root) == reference.SRC
    out = a.output_new.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'samples').mkdir(); (out / 'proof').mkdir()
    if a.mode == 'review30':
        jobs = choose_review_jobs(reference, out, a.seed)
    else:
        jobs = [(kind, label, i, split, str(out), a.seed) for kind, count in [('M', 100), ('J', 200), ('R', 150)]
                for label in (1, 0) for i in range(count)]
    save(out / 'plan.json', dict(mode=a.mode, seed=a.seed, split=split, expected=len(jobs), jobs=jobs,
        source_admission_sha256=sha(a.source_admission), real_audit_sha256=sha(a.real_audit),
        reference_sha256=hashes, wrapper_sha256=sha(__file__), source_count=len(rows),
        excluded_donors=excluded, SEAM=reference.SEAM, MIS=reference.MIS, J_BASES=reference.J_BASES,
        retries='Reference behavior: up to 300 attempts, parameters redrawn; not fixed-difficulty sampling.',
        no_real_masks_for_generation=True, training_admitted=False, gpu=False))
    entries = []
    started = time.time()
    with (out / 'records.jsonl').open('x') as log:
        with multiprocessing.get_context('spawn').Pool(a.workers, initializer=initialize,
                initargs=(a.reference_dir, rows)) as pool:
            for result in pool.imap_unordered(generate_one, jobs, chunksize=1):
                entries.append(result)
                log.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + '\n'); log.flush()
                if len(entries) % 10 == 0 or len(entries) == len(jobs):
                    print(json.dumps(dict(done=len(entries), expected=len(jobs), seconds=time.time()-started)), flush=True)
    successful = sorted([x for x in entries if not x.get('failed')], key=lambda x:x['id'])
    failed = [x for x in entries if x.get('failed')]
    save(out / 'manifest.json', dict(schema='codex-v42-reference-reuse/1', mode=a.mode, split=split,
        seed=a.seed, entries=successful, failed=failed, plan_sha256=sha(out/'plan.json'), training_admitted=False))
    receipt = dict(status='generated' if not failed else 'incomplete', expected=len(jobs), generated=len(successful),
        failed=len(failed), seconds=time.time()-started, manifest_sha256=sha(out/'manifest.json'),
        records_sha256=sha(out/'records.jsonl'), training_admitted=False, gpu_training_started=False)
    save(out / 'generation_complete.json', receipt)
    print(json.dumps(receipt), flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
